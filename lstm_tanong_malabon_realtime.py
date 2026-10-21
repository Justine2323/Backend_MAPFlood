"""
================================================================================
Continuous LSTM Water Rise Regression Model - Barangay Tañong, Malabon City
Target Supabase Table: Station A
================================================================================
"""
import gc
import json
import os
import random
import time
import urllib.request
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from supabase import create_client, Client

# Correct Keras 3 / TensorFlow imports
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    LSTM,
    LayerNormalization,
)
from keras.models import Sequential, load_model
from keras.regularizers import l2

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Direct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_flood_regression_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "lstm_scaler.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
PLOT_HISTORY_PATH = os.path.join(BASE_DIR, "plot_training_history.png")
PLOT_SCATTER_PATH = os.path.join(BASE_DIR, "plot_actual_vs_predicted.png")

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqhimswayyurxymltrxn.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S",
)

TANONG_LAT = 14.6542
TANONG_LON = 120.9508

# Hardware & Model Target Configuration (6.0 ft baseline height)
SENSOR_MOUNT_HEIGHT_MM = 1828.8  # Physical distance (mm) from sensor lens to dry baseline (6.0 ft)
PREDICTION_HORIZON_STEPS = 2     # Predict water level shift N steps into the future

FEATURE_COLUMNS = [
    "rain_percent",
    "wind_speed_kmh",
    "tof_distance_mm",
    "om_temp",
    "om_humidity",
    "om_weather_code",
    "rain_3p_avg",
    "tof_delta",
    "rain_6p_sum",
    "water_velocity_3p",
    "rain_wind_interaction",
]
TARGET_COLUMN = "predicted_rise_ft"
LOOKBACK_STEPS = 8
SLEEP_INTERVAL_SECONDS = 3600  # 1 hour sleep interval for background mode


# ============================================================================
# 0. CUSTOM ASYMMETRIC LOSS FUNCTION
# ============================================================================
@keras.saving.register_keras_serializable(package="Custom")
def asymmetric_mse(y_true, y_pred):
    """
    Custom Loss Function: Applies a 3.5x penalty multiplier when actual rise
    exceeds predicted rise (under-prediction), dramatically reducing false negatives.
    """
    error = y_true - y_pred
    weight = tf.where(error > 0, 3.5, 1.0)
    return tf.reduce_mean(weight * tf.square(error))


# ============================================================================
# 1. DATA SEEDER & SUPABASE FETCHING
# ============================================================================
def fetch_supabase_sensor_data(supabase: Client) -> pd.DataFrame:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching Supabase database (Station A)...")

    response = (
        supabase.table("Station A")
        .select("*")
        .order("id", desc=True)
        .limit(10000)
        .execute()
    )
    data = response.data

    if not data or len(data) < (LOOKBACK_STEPS + PREDICTION_HORIZON_STEPS):
        print(
            "[WARNING] Not enough data in Supabase for training sequence + target shift. Ensure ESP32 is pushing data."
        )
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values(by="id", ascending=True).reset_index(drop=True)

    # Missing value handling
    df["tof_distance_mm"] = df["tof_distance_mm"].ffill().fillna(SENSOR_MOUNT_HEIGHT_MM)
    df["rain_percent"] = df["rain_percent"].fillna(0)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0.0)
    df["om_temp"] = df["om_temp"].fillna(30.0)
    df["om_humidity"] = df["om_humidity"].fillna(70)
    df["om_weather_code"] = df["om_weather_code"].fillna(0)

    if "status" in df.columns:
        clear_mask = (
            df["status"]
            .astype(str)
            .str.upper()
            .str.contains("CLEAR|DRY|NORMAL", na=False)
        )
        df.loc[clear_mask, "rain_percent"] = 0

    # Primary rolling & derived features
    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().round(2)
    df["tof_delta"] = df["tof_distance_mm"].diff().fillna(0.0).round(2)

    # Enhanced trend features
    df["rain_6p_sum"] = df["rain_percent"].rolling(window=6, min_periods=1).sum().round(2)
    df["actual_water_level_ft"] = ((SENSOR_MOUNT_HEIGHT_MM - df["tof_distance_mm"]) / 304.8).clip(lower=0)
    df["water_velocity_3p"] = df["actual_water_level_ft"].diff(3).fillna(0.0).round(2)
    df["rain_wind_interaction"] = ((df["rain_percent"] / 100.0) * df["wind_speed_kmh"]).round(2)

    # Shift water levels into the future to establish ground-truth target
    df["future_water_level_ft"] = df["actual_water_level_ft"].shift(-PREDICTION_HORIZON_STEPS)

    # Target variable: Actual future water rise relative to current step
    df[TARGET_COLUMN] = (df["future_water_level_ft"] - df["actual_water_level_ft"]).clip(lower=0).round(2)

    # Drop trailing rows where future target cannot be known
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    print(f"[INFO] Successfully loaded and preprocessed {len(df)} valid training rows.")
    return df


def create_3d_sequences(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    feature_data = df[FEATURE_COLUMNS].values
    targets = df[TARGET_COLUMN].values

    if fit_scaler:
        scaled_features = scaler.fit_transform(feature_data)
    else:
        scaled_features = scaler.transform(feature_data)

    X_seq, y_seq, row_ids = [], [], []
    for i in range(len(df) - LOOKBACK_STEPS + 1):
        X_seq.append(scaled_features[i : i + LOOKBACK_STEPS])
        last_step_idx = i + LOOKBACK_STEPS - 1
        y_seq.append(targets[last_step_idx])
        row_ids.append(df["id"].iloc[last_step_idx])

    if len(X_seq) == 0:
        return np.empty((0, LOOKBACK_STEPS, len(FEATURE_COLUMNS))), np.empty((0,)), []

    return np.array(X_seq), np.array(y_seq), row_ids


# ============================================================================
# 2. MODEL ARCHITECTURE & GRAPHING
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
    print(f"[INFO] Saved regression plots to {BASE_DIR}")


# ============================================================================
# 3. FEEDBACK TO SUPABASE (4-TIER ALIGNMENT: GREEN, YELLOW, ORANGE, RED)
# ============================================================================
def push_predictions_to_supabase(supabase: Client, row_ids: list, y_pred: np.ndarray):
    print("\n[INFO] Updating Supabase database with 4-tier threshold predictions...")

    y_pred_flat = y_pred.flatten()
    payload = []

    for r_id, pred_rise in zip(row_ids, y_pred_flat):
        # 3.0 ft+ is DANGER LEVEL (RED / SIREN)
        if pred_rise >= 3.0:
            risk_code = 3
            label = "DANGER ALERT (RED)"
        # 2.0 ft to 2.99 ft is ALERT LEVEL (ORANGE)
        elif pred_rise >= 2.0:
            risk_code = 2
            label = "ALERT (ORANGE)"
        # 1.0 ft to 1.99 ft is WARNING LEVEL (YELLOW)
        elif pred_rise >= 1.0:
            risk_code = 1
            label = "WARNING (YELLOW)"
        # Below 1.0 ft is SAFE (GREEN)
        else:
            risk_code = 0
            label = "SAFE (GREEN)"

        payload.append({
            "id": int(r_id),
            "predicted_rise_ft": round(float(pred_rise), 2),
            "predicted_flood_risk": risk_code,
            "prediction_label": label,
        })

    batch_size = 200
    updated_count = 0

    for i in range(0, len(payload), batch_size):
        batch = payload[i : i + batch_size]
        try:
            supabase.table("Station A").upsert(batch).execute()
            updated_count += len(batch)
        except Exception as e:
            print(f"[ERROR] Failed to upsert batch starting at index {i}: {e}")

    print(f"[INFO] Successfully updated {updated_count} rows in Supabase.")


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================
def run_training_cycle():
    print("\n--- Starting Data Fetch and Training Cycle ---")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    df = fetch_supabase_sensor_data(supabase)

    if df.empty:
        print("[WARNING] DataFrame is empty. Aborting cycle.")
        return

    train_ratio = 0.85
    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[INFO] Data Split: {len(df_train)} Train rows | {len(df_test)} Test rows")

    # --- AUTOMATIC SCALER & MODEL RE-CALIBRATION SAFEGUARD ---
    need_new_scaler = True
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(FEATURE_COLUMNS):
                print("[INFO] Found valid matching scaler. Loading...")
                X_train, y_train, train_ids = create_3d_sequences(df_train, scaler, fit_scaler=False)
                need_new_scaler = False
            else:
                print(f"[WARNING] Saved scaler feature count ({getattr(scaler, 'n_features_in_', 'Unknown')}) does not match current feature count ({len(FEATURE_COLUMNS)}). Rebuilding scaler...")
        except Exception as e:
            print(f"[WARNING] Scaler load failed ({e}). Rebuilding...")

    if need_new_scaler:
        print("[INFO] Fitting fresh StandardScaler for 11 features...")
        scaler = StandardScaler()
        X_train, y_train, train_ids = create_3d_sequences(df_train, scaler, fit_scaler=True)
        joblib.dump(scaler, SCALER_PATH)
        # Force model rebuild if scaler was invalidated
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    X_test, y_test, test_ids = create_3d_sequences(df_test, scaler, fit_scaler=False)
    X_all, y_all, all_row_ids = create_3d_sequences(df, scaler, fit_scaler=False)

    val_data = (X_test, y_test) if len(X_test) > 0 else None

    if os.path.exists(MODEL_PATH):
        print(f"\n[INFO] Found existing model at {MODEL_PATH}. Loading weights...")
        try:
            model = load_model(MODEL_PATH, custom_objects={"asymmetric_mse": asymmetric_mse})
        except Exception as e:
            print(f"[WARNING] Could not load existing model architecture ({e}). Rebuilding new model...")
            model = build_lstm_regression(input_shape=(X_train.shape[1], X_train.shape[2]))
    else:
        print("\n[INFO] Building new architecture for 11 features...")
        model = build_lstm_regression(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

    print("\n[INFO] --- Fine-tuning LSTM Model ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=15,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # --- EVALUATION ON TRAIN & TEST SETS ---
    print("\n" + "=" * 60)
    print("LSTM REGRESSION ACCURACY EVALUATION METRICS")
    print("=" * 60)

    # 1. Train Performance
    y_train_pred = model.predict(X_train).flatten()
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_abs_err = np.abs(y_train - y_train_pred)
    train_acc_3in = np.mean(train_abs_err <= 0.25) * 100
    train_acc_6in = np.mean(train_abs_err <= 0.50) * 100

    print("--- TRAINING SET PERFORMANCE ---")
    print(f"Train MAE                      : {train_mae:.4f} ft")
    print(f"Train Accuracy (within 3 in)   : {train_acc_3in:.2f}%")
    print(f"Train Accuracy (within 6 in)   : {train_acc_6in:.2f}%")

    # 2. Test / Validation Performance
    if len(X_test) > 0:
        y_test_pred = model.predict(X_test).flatten()

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        test_abs_err = np.abs(y_test - y_test_pred)
        test_acc_3in = np.mean(test_abs_err <= 0.25) * 100
        test_acc_6in = np.mean(test_abs_err <= 0.50) * 100

        print("\n--- TEST / VALIDATION SET PERFORMANCE ---")
        print(f"Test Mean Absolute Error (MAE) : {test_mae:.4f} ft")
        print(f"Test Mean Squared Error (MSE)  : {test_mse:.4f}")
        print(f"Test R2 Score                  : {test_r2:.4f}")
        print("-" * 60)
        print(f"Test Accuracy (within 3 in / 0.25 ft) : {test_acc_3in:.2f}%")
        print(f"Test Accuracy (within 6 in / 0.50 ft) : {test_acc_6in:.2f}%")

        plot_regression_metrics(history, y_test, y_test_pred)

    print("=" * 60)

    if len(X_all) > 0:
        all_preds = model.predict(X_all)
        push_predictions_to_supabase(supabase, all_row_ids, all_preds)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    schema = {"feature_columns": FEATURE_COLUMNS, "lookback_steps": LOOKBACK_STEPS}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)

    print("\n[SUCCESS] Cycle complete. Artifacts saved.")

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    print("=====================================================")
    print("Starting Continuous LSTM Prediction Engine")
    print("=====================================================")

    EXECUTION_MODE = os.getenv("EXECUTION_MODE", "SINGLE")

    if EXECUTION_MODE == "SINGLE":
        run_training_cycle()
    else:
        while True:
            try:
                run_training_cycle()
                minutes = SLEEP_INTERVAL_SECONDS // 60
                print(f"\n[INFO] Sleeping for {minutes} minutes...")
                time.sleep(SLEEP_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                print("\n[INFO] Stopped by user.")
                break
            except Exception as e:
                print(f"\n[ERROR] Exception encountered: {e}")
                time.sleep(300)