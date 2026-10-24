"""
================================================================================
Station C - Integrated Real-Time Optical Telemetry, GPIO & Live LSTM Engine (Pi 5)
================================================================================
"""
import gc
import json
import os
import random
import ssl
import threading
import time
import traceback
import urllib.request
import cv2  # type: ignore
from flask import Flask, Response, render_template_string, send_from_directory  # type: ignore
import joblib  # type: ignore

# Headless matplotlib configuration for Linux / Pi 5
import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from supabase import Client, create_client  # type: ignore
import tensorflow as tf  # type: ignore

# Suppress TensorFlow boot warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras  # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from keras.layers import (  # type: ignore
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTM,
)
from keras.models import Sequential, load_model  # type: ignore
from keras.regularizers import l2  # type: ignore

# --- RASPBERRY PI 5 GPIO CONFIGURATION ---
SIREN_PIN = 17       # GPIO 17 (Pin 11) -> Relay / Siren Control
RAIN_GPIO_PIN = 27   # GPIO 27 (Pin 13) -> Digital Rain Sensor
WIND_GPIO_PIN = 22   # GPIO 22 (Pin 15) -> Wind Anemometer Pulse Pin

try:
    from gpiozero import DigitalInputDevice, Output  # type: ignore

    siren = Output(SIREN_PIN)
    siren.off()
    rain_sensor = DigitalInputDevice(RAIN_GPIO_PIN, pull_up=True)
    wind_sensor = DigitalInputDevice(WIND_GPIO_PIN, pull_up=True)
    GPIO_AVAILABLE = True
    print("[INFO] Raspberry Pi 5 GPIO Hardware initialized successfully.")
except Exception as e:
    GPIO_AVAILABLE = False
    print(f"[WARNING] GPIO Initialization skipped ({e}). Running in simulation mode.")

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# --- LOCATION & THRESHOLD CONSTANTS ---
TANONG_LAT = 14.6542
TANONG_LON = 120.9508
SIREN_TRIGGER_RAIN_PERCENT = 50
SIREN_TRIGGER_WIND_KMH = 35.0

# --- MODEL PATHS & CONSTANTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_flood_regression_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "lstm_scaler.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
PLOT_HISTORY_PATH = os.path.join(BASE_DIR, "plot_training_history.png")
PLOT_SCATTER_PATH = os.path.join(BASE_DIR, "plot_actual_vs_predicted.png")

LOOKBACK_STEPS = 8
PREDICTION_HORIZON_STEPS = 2
RETRAIN_INTERVAL_SECONDS = 21600  # Automatically retrain every 6 hours

FEATURE_COLUMNS = [
    "rain_percent",
    "wind_speed_kmh",
    "camera_water_level_ft",
    "om_temp",
    "om_humidity",
    "om_weather_code",
    "rain_3p_avg",
    "level_delta",
    "rain_6p_sum",
    "water_velocity_3p",
    "rain_wind_interaction",
]
TARGET_COLUMN = "predicted_rise_ft"

# --- GLOBAL IN-MEMORY MODEL CACHE ---
global_lstm_model = None
global_scaler = None
model_lock = threading.Lock()

# Global Hardware & State Memory
cap = None
camera_index = 0
last_valid_level_ft = 0.0

current_data = {
    "color": "Unknown",
    "measurement": "0 cm",
    "status": "Idle",
    "water_tier": 0,
    "actual_water_level_ft": 0.0,
    "predicted_rise_ft": 0.0,
    "predicted_flood_risk": 0,
    "prediction_label": "SAFE (GREEN)",
    "rain_percent": 0,
    "wind_speed_kmh": 0.0,
}

training_status = {
    "is_training": False,
    "epoch_progress": "0/15",
    "train_mae": 0.0,
    "test_mae": 0.0,
    "acc_3in": 0.0,
    "acc_6in": 0.0,
    "last_trained": "Never",
}

rain_history = [0.0] * LOOKBACK_STEPS
wind_history = [0.0] * LOOKBACK_STEPS
history_index = 0
feature_history = []

# Cached Open-Meteo values (Updated every 10 minutes)
cached_weather = {
    "om_temp": 30.0,
    "om_humidity": 70,
    "rain_percent_api": 0.0,
    "om_weather_code": 0,
    "wind_speed_kmh_api": 0.0,
}


# ============================================================================
# 0. CUSTOM ASYMMETRIC LOSS FUNCTION & MODEL CACHE
# ============================================================================
@keras.saving.register_keras_serializable(package="Custom")
def asymmetric_mse(y_true, y_pred):
    """3.5x penalty multiplier on under-predictions to prevent false negatives."""
    error = y_true - y_pred
    weight = tf.where(error > 0, 3.5, 1.0)
    return tf.reduce_mean(weight * tf.square(error))


def load_cached_model_and_scaler():
    """Loads model and scaler into RAM to enable high-speed real-time predictions."""
    global global_lstm_model, global_scaler
    with model_lock:
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                global_scaler = joblib.load(SCALER_PATH)
                global_lstm_model = load_model(
                    MODEL_PATH,
                    custom_objects={"asymmetric_mse": asymmetric_mse},
                    compile=False,
                )
                print("[INFO] Model and Scaler successfully cached in memory.")
            else:
                print("[WARNING] Model or Scaler file missing. Running edge heuristic fallback.")
        except Exception as e:
            print(f"[ERROR] Loading model into cache failed: {e}")


load_cached_model_and_scaler()


# ============================================================================
# 1. OPTICAL & GPIO SENSOR DRIVERS
# ============================================================================
def init_camera():
    """Scans for active Logitech C270HD camera."""
    global cap, camera_index
    for idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                print(f"[INFO] Connected to USB Camera at index {idx}")
                cap = temp_cap
                camera_index = idx
                return True
            temp_cap.release()
    print("[WARNING] No active USB camera device found.")
    return False


init_camera()


def get_color_data(h, s, v):
    """Maps HSV space to 4-tier flood levels (Feet)."""
    if s < 50 or v < 50:
        return "Unknown", "N/A", "Unknown", 0, -1.0

    if (0 <= h <= 8) or (170 <= h <= 180):
        return "Red", "2-3 ft (20-30 cm)", "Critical Level", 3, 3.0
    elif 9 <= h <= 25:
        return "Orange", "1-2 ft (11-19 cm)", "Warning Level", 2, 2.0
    elif 36 <= h <= 85:
        return "Green", "0-1 ft (5-10 cm)", "Light Level", 1, 1.0

    return "Unknown", "N/A", "Unknown", 0, -1.0


def read_fast_rain_percent():
    if GPIO_AVAILABLE:
        return 100 if not rain_sensor.is_active else 0
    return 0


def read_fast_wind_kmh():
    if GPIO_AVAILABLE:
        pulse_count = 0
        start_time = time.time()
        last_state = wind_sensor.is_active

        while time.time() - start_time < 0.2:  # 200ms sampling
            current_state = wind_sensor.is_active
            if current_state and not last_state:
                pulse_count += 1
            last_state = current_state
            time.sleep(0.005)

        wind_kmh = (pulse_count / 0.2) * 2.4
        return round(wind_kmh, 2)
    return 0.0


def fetch_open_meteo_data():
    global cached_weather
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={TANONG_LAT}&longitude={TANONG_LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "StationC/1.0"})

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
                data = json.loads(response.read())
                current = data.get("current", {})
                cached_weather = {
                    "om_temp": current.get("temperature_2m", 30.0),
                    "om_humidity": current.get("relative_humidity_2m", 70),
                    "rain_percent_api": current.get("precipitation", 0.0) * 10.0,
                    "om_weather_code": current.get("weather_code", 0),
                    "wind_speed_kmh_api": current.get("wind_speed_10m", 0.0),
                }
                return
        except Exception:
            time.sleep(1)


def weather_fetch_loop():
    while True:
        fetch_open_meteo_data()
        time.sleep(600)


threading.Thread(target=weather_fetch_loop, daemon=True).start()


# ============================================================================
# 2. FAULT-TOLERANT CONTINUOUS LSTM TRAINING ENGINE
# ============================================================================
def build_lstm_regression(input_shape) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.1, kernel_regularizer=l2(0.001))),
        LayerNormalization(),
        Dropout(0.25),
        Bidirectional(LSTM(32, return_sequences=False, recurrent_dropout=0.1, kernel_regularizer=l2(0.001))),
        LayerNormalization(),
        Dropout(0.25),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=asymmetric_mse,
        metrics=["mae"],
    )
    return model


def plot_regression_metrics(history, y_true, y_pred):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["mae"], label="Train MAE", color="blue")
    if "val_mae" in history.history:
        plt.plot(history.history["val_mae"], label="Validation MAE", color="orange")
    plt.title("LSTM Model Error (Feet)")
    plt.ylabel("Mean Absolute Error (ft)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss (Asymmetric MSE)", color="blue")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss (Asymmetric MSE)", color="orange")
    plt.title("LSTM Model Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_HISTORY_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="teal")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--")
    plt.title("Actual vs Predicted Water Rise (ft)")
    plt.xlabel("Actual Rise (ft)")
    plt.ylabel("Predicted Rise (ft)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER_PATH, dpi=300)
    plt.close()


def fetch_supabase_training_data(supabase: Client) -> pd.DataFrame:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fetching historical dataset from Supabase (StationC)...")
    try:
        response = (
            supabase.table("StationC")
            .select("*")
            .order("id", desc=True)
            .limit(10000)
            .execute()
        )
        data = response.data
    except Exception as e:
        print(f"[ERROR] Failed to fetch training data from Supabase: {e}")
        return pd.DataFrame()

    if not data or len(data) == 0:
        print("[WARNING] 0 rows found in StationC table.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Safe column parsing safeguard to prevent KeyError
    required_columns = ["water_tier", "rain_percent", "wind_speed_kmh", "om_temp", "om_humidity", "om_weather_code"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values(by="id", ascending=True).reset_index(drop=True)

    df["camera_water_level_ft"] = df["water_tier"].map({3: 3.0, 2: 2.0, 1: 1.0}).fillna(0.0)
    df["rain_percent"] = df["rain_percent"].fillna(0.0)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0.0)
    df["om_temp"] = df["om_temp"].fillna(30.0)
    df["om_humidity"] = df["om_humidity"].fillna(70.0)
    df["om_weather_code"] = df["om_weather_code"].fillna(0.0)

    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().round(2)
    df["level_delta"] = df["camera_water_level_ft"].diff().fillna(0.0).round(2)
    df["rain_6p_sum"] = df["rain_percent"].rolling(window=6, min_periods=1).sum().round(2)
    df["water_velocity_3p"] = df["camera_water_level_ft"].diff(3).fillna(0.0).round(2)
    df["rain_wind_interaction"] = ((df["rain_percent"] / 100.0) * df["wind_speed_kmh"]).round(2)

    df["future_water_level_ft"] = df["camera_water_level_ft"].shift(-PREDICTION_HORIZON_STEPS)
    df[TARGET_COLUMN] = (df["future_water_level_ft"] - df["camera_water_level_ft"]).clip(lower=0).round(2)

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df


def create_3d_sequences(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    feature_data = df[FEATURE_COLUMNS].values
    targets = df[TARGET_COLUMN].values

    scaled_features = scaler.fit_transform(feature_data) if fit_scaler else scaler.transform(feature_data)

    X_seq, y_seq = [], []
    for i in range(len(df) - LOOKBACK_STEPS + 1):
        X_seq.append(scaled_features[i : i + LOOKBACK_STEPS])
        y_seq.append(targets[i + LOOKBACK_STEPS - 1])

    if len(X_seq) == 0:
        return np.empty((0, LOOKBACK_STEPS, len(FEATURE_COLUMNS))), np.empty((0,))

    return np.array(X_seq), np.array(y_seq)


class ProgressCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_status["epoch_progress"] = f"{epoch + 1}/15"


def run_training_cycle():
    global training_status
    if training_status["is_training"]:
        return

    print("\n--- Starting Station C LSTM Re-Training Cycle ---")
    training_status["is_training"] = True
    training_status["epoch_progress"] = "Starting..."

    try:
        df = fetch_supabase_training_data(supabase)
        min_rows_needed = LOOKBACK_STEPS + PREDICTION_HORIZON_STEPS + 2  # 12 rows minimum

        if df.empty or len(df) < min_rows_needed:
            rows_found = len(df) if not df.empty else 0
            print(f"[WARNING] Not enough valid rows in StationC table ({rows_found}/{min_rows_needed} rows found).")
            print("[HINT] Use '/seed_data' on dashboard to populate synthetic initial telemetry.")
            training_status["epoch_progress"] = f"Need 12+ DB Rows (Found {rows_found})"
            return

        split_idx = int(len(df) * 0.85)
        df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
        df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

        if len(df_train) < LOOKBACK_STEPS or len(df_test) < LOOKBACK_STEPS:
            df_train = df.copy()
            df_test = df.copy()

        scaler = StandardScaler()
        X_train, y_train = create_3d_sequences(df_train, scaler, fit_scaler=True)
        X_test, y_test = create_3d_sequences(df_test, scaler, fit_scaler=False)

        if len(X_train) == 0:
            print("[WARNING] Created 0 sequence samples. Gathering more dataset rows.")
            training_status["epoch_progress"] = "Need more samples"
            return

        val_data = (X_test, y_test) if len(X_test) > 0 else None

        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH, custom_objects={"asymmetric_mse": asymmetric_mse})
            except Exception:
                model = build_lstm_regression(input_shape=(X_train.shape[1], X_train.shape[2]))
        else:
            model = build_lstm_regression(input_shape=(X_train.shape[1], X_train.shape[2]))

        early_stop = EarlyStopping(monitor="val_loss" if val_data else "loss", patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss" if val_data else "loss", factor=0.5, patience=2, min_lr=1e-5)
        progress_cb = ProgressCallback()

        training_status["epoch_progress"] = "0/15"

        history = model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=15,
            batch_size=min(16, len(X_train)),
            callbacks=[early_stop, reduce_lr, progress_cb],
            verbose=0,
        )

        y_train_pred = model.predict(X_train, verbose=0).flatten()
        train_mae = mean_absolute_error(y_train, y_train_pred)

        if len(X_test) > 0:
            y_test_pred = model.predict(X_test, verbose=0).flatten()
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_abs_err = np.abs(y_test - y_test_pred)
            acc_3in = np.mean(test_abs_err <= 0.25) * 100
            acc_6in = np.mean(test_abs_err <= 0.50) * 100

            plot_regression_metrics(history, y_test, y_test_pred)

            training_status["train_mae"] = round(float(train_mae), 4)
            training_status["test_mae"] = round(float(test_mae), 4)
            training_status["acc_3in"] = round(float(acc_3in), 2)
            training_status["acc_6in"] = round(float(acc_6in), 2)

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        schema = {"feature_columns": FEATURE_COLUMNS, "lookback_steps": LOOKBACK_STEPS}
        with open(SCHEMA_PATH, "w") as f:
            json.dump(schema, f, indent=4)

        load_cached_model_and_scaler()

        training_status["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print("\n[SUCCESS] Model training complete and cached memory updated.")

    except Exception as e:
        print(f"[ERROR] Training cycle failed: {e}")
        traceback.print_exc()
        training_status["epoch_progress"] = "Error"
    finally:
        training_status["is_training"] = False
        keras.backend.clear_session()
        gc.collect()


def background_retrain_scheduler():
    while True:
        time.sleep(RETRAIN_INTERVAL_SECONDS)
        run_training_cycle()


train_thread = threading.Thread(target=background_retrain_scheduler, daemon=True)
train_thread.start()


# ============================================================================
# 3. HIGH-SPEED REAL-TIME PREDICTION & TELEMETRY ENGINE (1s LOOP)
# ============================================================================
def compute_edge_prediction(rain_percent, wind_kmh, om_weather_code, om_humidity):
    global rain_history, wind_history, history_index

    rain_history[history_index] = float(rain_percent)
    wind_history[history_index] = float(wind_kmh)
    history_index = (history_index + 1) % LOOKBACK_STEPS

    rain_sum = sum(rain_history[(history_index - 1 - i) % LOOKBACK_STEPS] for i in range(3))
    wind_sum = sum(wind_history[(history_index - 1 - i) % LOOKBACK_STEPS] for i in range(3))
    rain_3p_avg = rain_sum / 3.0
    wind_3p_avg = wind_sum / 3.0

    weather_severity_factor = 1.0
    if om_weather_code >= 95:
        weather_severity_factor = 1.35
    elif om_weather_code >= 63 or om_weather_code in [81, 82]:
        weather_severity_factor = 1.20
    elif om_weather_code >= 51 or om_weather_code == 80:
        weather_severity_factor = 1.10

    humidity_factor = 1.15 if om_humidity >= 85 else (1.05 if om_humidity >= 70 else 1.0)

    base_rise = (rain_3p_avg * 0.010) + (wind_3p_avg * 0.002)
    predicted_rise = base_rise * weather_severity_factor * humidity_factor
    return max(0.0, round(predicted_rise, 2))


def predict_lstm_water_rise_fast(features_dict):
    """High-speed prediction running against RAM cached model."""
    global feature_history, global_lstm_model, global_scaler

    feature_history.append(features_dict)
    if len(feature_history) > 10:
        feature_history.pop(0)

    df_hist = pd.DataFrame(feature_history)

    df_hist["rain_3p_avg"] = df_hist["rain_percent"].rolling(3, min_periods=1).mean()
    df_hist["level_delta"] = df_hist["camera_water_level_ft"].diff().fillna(0.0)
    df_hist["rain_6p_sum"] = df_hist["rain_percent"].rolling(6, min_periods=1).sum()
    df_hist["water_velocity_3p"] = df_hist["camera_water_level_ft"].diff(3).fillna(0.0)
    df_hist["rain_wind_interaction"] = (df_hist["rain_percent"] / 100.0) * df_hist["wind_speed_kmh"]

    if len(df_hist) < LOOKBACK_STEPS:
        return None

    current_window = df_hist[FEATURE_COLUMNS].tail(LOOKBACK_STEPS).values

    with model_lock:
        if global_lstm_model is not None and global_scaler is not None:
            try:
                scaled_features = global_scaler.transform(current_window)
                X_input = np.array([scaled_features])
                predicted_rise = float(global_lstm_model.predict(X_input, verbose=0)[0][0])
                return max(0.0, round(predicted_rise, 2))
            except Exception:
                return None
    return None


def real_time_telemetry_loop():
    """Runs high-frequency 1-second sampling and combined LSTM predictions."""
    global current_data, last_valid_level_ft
    while True:
        start_time = time.time()

        rain_pct = read_fast_rain_percent()
        wind_kmh = read_fast_wind_kmh()

        current_data["rain_percent"] = rain_pct
        current_data["wind_speed_kmh"] = wind_kmh

        active_level = current_data["actual_water_level_ft"]
        if active_level < 0:
            active_level = last_valid_level_ft
        else:
            last_valid_level_ft = active_level

        effective_rain = max(rain_pct, cached_weather["rain_percent_api"])
        effective_wind = max(wind_kmh, cached_weather["wind_speed_kmh_api"])

        features_dict = {
            "rain_percent": effective_rain,
            "wind_speed_kmh": effective_wind,
            "camera_water_level_ft": active_level,
            "om_temp": cached_weather["om_temp"],
            "om_humidity": cached_weather["om_humidity"],
            "om_weather_code": cached_weather["om_weather_code"],
        }

        edge_rise = compute_edge_prediction(
            effective_rain,
            effective_wind,
            cached_weather["om_weather_code"],
            cached_weather["om_humidity"],
        )
        lstm_rise = predict_lstm_water_rise_fast(features_dict)

        final_rise = lstm_rise if lstm_rise is not None else edge_rise
        current_data["predicted_rise_ft"] = final_rise

        is_rain_danger = effective_rain >= SIREN_TRIGGER_RAIN_PERCENT
        is_wind_danger = effective_wind >= SIREN_TRIGGER_WIND_KMH

        if active_level >= 3.0 or final_rise >= 3.0 or is_rain_danger or is_wind_danger:
            risk_code = 3
            risk_label = "DANGER ALERT (RED)"
            if GPIO_AVAILABLE:
                siren.on()
        elif active_level >= 2.0 or final_rise >= 2.0:
            risk_code = 2
            risk_label = "ALERT (ORANGE)"
            if GPIO_AVAILABLE:
                siren.off()
        elif active_level >= 1.0 or final_rise >= 1.0:
            risk_code = 1
            risk_label = "WARNING (YELLOW)"
            if GPIO_AVAILABLE:
                siren.off()
        else:
            risk_code = 0
            risk_label = "SAFE (GREEN)"
            if GPIO_AVAILABLE:
                siren.off()

        current_data["predicted_flood_risk"] = risk_code
        current_data["prediction_label"] = risk_label

        elapsed = time.time() - start_time
        time.sleep(max(0.1, 1.0 - elapsed))


threading.Thread(target=real_time_telemetry_loop, daemon=True).start()


def supabase_sync_loop():
    """Asynchronous background database sync every 5 seconds."""
    while True:
        time.sleep(5)
        try:
            data_to_insert = {
                "detected_color": current_data["color"],
                "water_tier": current_data["water_tier"],
                "distance_data": current_data["measurement"],
                "status": current_data["status"],
                "predicted_flood_risk": current_data["predicted_flood_risk"],
                "prediction_label": current_data["prediction_label"],
                "predicted_rise_ft": current_data["predicted_rise_ft"],
                "rain_percent": current_data["rain_percent"],
                "wind_speed_kmh": current_data["wind_speed_kmh"],
                "om_temp": cached_weather["om_temp"],
                "om_humidity": cached_weather["om_humidity"],
                "om_weather_code": cached_weather["om_weather_code"],
            }
            supabase.table("StationC").insert(data_to_insert).execute()
            print(f"[Supabase Sync -> StationC] Level: {current_data['actual_water_level_ft']}ft | Pred Rise: {current_data['predicted_rise_ft']}ft | Risk: {current_data['prediction_label']}")
        except Exception as e:
            print(f"[Supabase Sync Error]: {e}")


threading.Thread(target=supabase_sync_loop, daemon=True).start()


# ============================================================================
# 4. FLASK STREAM & WEB DASHBOARD ENDPOINTS
# ============================================================================
def generate_frames():
    global cap, current_data
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(1)
            init_camera()
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        height, width, _ = frame.shape
        cx, cy = width // 2, height // 2
        box_size = 25
        x1, y1 = cx - box_size, cy - box_size
        x2, y2 = cx + box_size, cy + box_size

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_roi)
        avg_h, avg_s, avg_v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

        color_name, dist_str, status_desc, tier, level_ft = get_color_data(avg_h, avg_s, avg_v)

        if color_name != "Unknown":
            current_data["color"] = color_name
            current_data["measurement"] = dist_str
            current_data["status"] = status_desc
            current_data["water_tier"] = tier
            current_data["actual_water_level_ft"] = level_ft
        else:
            current_data["actual_water_level_ft"] = -1.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {current_data['color']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Level: {current_data['measurement']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred Rise: {current_data['predicted_rise_ft']}ft", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template_string("""
        <html>
            <head>
                <title>Station C - Raspberry Pi 5 Telemetry Dashboard</title>
                <style>
                    body { font-family: sans-serif; background: #111; color: white; text-align: center; margin: 20px; }
                    .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
                    .card { background: #222; padding: 20px; border-radius: 10px; border: 1px solid #444; width: 640px; }
                    .stat { font-size: 1.1em; margin: 10px 0; }
                    .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; margin: 5px; display: inline-block; }
                    .btn:hover { background: #0056b3; }
                    .btn-seed { background: #28a745; }
                    .btn-seed:hover { background: #1e7e34; }
                    img { border-radius: 8px; max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h2>Station C: Integrated Optical & Continuous LSTM Dashboard</h2>
                <div class="container">
                    <div class="card">
                        <h3>Real-Time Camera Telemetry</h3>
                        <img src="/video_feed" width="640" height="480">
                    </div>
                    <div class="card">
                        <h3>LSTM Model Training Metrics</h3>
                        <div class="stat"><b>Status:</b> {{ "Training in progress (" + training_status.epoch_progress + ")" if training_status.is_training else "Idle (" + training_status.epoch_progress + ")" }}</div>
                        <div class="stat"><b>Last Trained:</b> {{ training_status.last_trained }}</div>
                        <div class="stat"><b>Test MAE Error:</b> {{ training_status.test_mae }} ft</div>
                        <div class="stat"><b>Accuracy (&le; 3 in / 0.25 ft):</b> {{ training_status.acc_3in }}%</div>
                        <div class="stat"><b>Accuracy (&le; 6 in / 0.50 ft):</b> {{ training_status.acc_6in }}%</div>
                        <br>
                        <a href="/trigger_train" class="btn">Trigger Model Training</a>
                        <a href="/seed_data" class="btn btn-seed">Seed 50 Synthetic Rows</a>
                    </div>
                </div>
                <br>
                <h2>Training Evaluation Graphs</h2>
                <div class="container">
                    <div class="card">
                        <h3>Training & Validation Error Curves</h3>
                        <img src="/plots/plot_training_history.png" width="600">
                    </div>
                    <div class="card">
                        <h3>Actual vs Predicted Regression Fit</h3>
                        <img src="/plots/plot_actual_vs_predicted.png" width="600">
                    </div>
                </div>
            </body>
        </html>
    """, training_status=training_status)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/plots/<filename>")
def serve_plot(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/trigger_train")
def trigger_train():
    if not training_status["is_training"]:
        t = threading.Thread(target=run_training_cycle, daemon=True)
        t.start()
        return "Training cycle triggered successfully! <a href='/'>Return to Dashboard</a>"
    return "Training is already running! <a href='/'>Return to Dashboard</a>"


@app.route("/seed_data")
def seed_data():
    """Generates 50 synthetic telemetry rows in Supabase to allow immediate LSTM model training."""
    colors = ["Green", "Orange", "Red"]
    tiers = [1, 2, 3]
    measurements = ["0-1 ft (5-10 cm)", "1-2 ft (11-19 cm)", "2-3 ft (20-30 cm)"]
    labels = ["SAFE (GREEN)", "WARNING (YELLOW)", "DANGER ALERT (RED)"]

    batch = []
    for _ in range(50):
        idx = random.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
        batch.append({
            "detected_color": colors[idx],
            "water_tier": tiers[idx],
            "distance_data": measurements[idx],
            "status": "Rain" if idx > 0 else "Clear",
            "predicted_flood_risk": idx,
            "prediction_label": labels[idx],
            "predicted_rise_ft": round(random.uniform(0.0, 3.5), 2),
            "rain_percent": random.randint(0, 100),
            "wind_speed_kmh": round(random.uniform(5.0, 45.0), 2),
            "om_temp": round(random.uniform(26.0, 32.0), 1),
            "om_humidity": random.randint(60, 95),
            "om_weather_code": random.choice([0, 51, 61, 80, 95]),
        })

    try:
        supabase.table("StationC").insert(batch).execute()
        return "Successfully seeded 50 telemetry rows into Supabase! <a href='/'>Return to Dashboard & Click 'Trigger Model Training'</a>"
    except Exception as e:
        return f"Failed to seed data: {e} <a href='/'>Return to Dashboard</a>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)