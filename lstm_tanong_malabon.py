"""
================================================================================
LSTM Flood Risk Model with Supabase Feedback Loop - Barangay Tañong, Malabon City
================================================================================
"""

import json
import os
import random
import urllib.request
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from supabase import create_client, Client

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Input, LSTM
from keras.models import Sequential
from keras.regularizers import l2

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Direct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_flood_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "lstm_scaler.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
PLOT_HISTORY_PATH = os.path.join(BASE_DIR, "plot_training_history.png")
PLOT_CM_PATH = os.path.join(BASE_DIR, "plot_confusion_matrix.png")

# Supabase Credentials
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"

# Coordinates: Barangay Tañong, Malabon City
TANONG_LAT = 14.6542
TANONG_LON = 120.9508

FEATURE_COLUMNS = [
    "rain_percent",
    "wind_speed_kmh",
    "tof_distance_mm",
    "om_temp",
    "om_humidity",
    "om_weather_code",
    "rain_3p_avg",
    "tof_delta",
]
TARGET_COLUMN = "flood_risk"
CLASS_INDICES = [0, 1, 2]
CLASS_NAMES = ["Safe (LOW)", "Warning (MODERATE)", "Flood Alert (HIGH)"]
LOOKBACK_STEPS = 8


# ============================================================================
# 1. DATA SEEDER & SUPABASE FETCHING
# ============================================================================
def seed_tanong_data_from_open_meteo(supabase: Client):
    print(f"\n[AUTO-SEED] Clearing old table & seeding fresh cyclical weather data...")
    
    try:
        supabase.table("sensor_data").delete().neq("id", 0).execute()
    except Exception as e:
        print(f"Notice during reset: {e}")

    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={TANONG_LAT}&longitude={TANONG_LON}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,rain,weather_code,wind_speed_10m"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as response:
            meteo_data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch Open-Meteo data: {e}")
        return

    if "hourly" not in meteo_data:
        print("Invalid response from Open-Meteo API.")
        return

    hourly = meteo_data["hourly"]
    total_entries = len(hourly["time"])

    records = []
    current_tof = 1900.0  # Dry baseline (~1.9 meters distance to sensor)

    for i in range(total_entries):
        temp = float(hourly["temperature_2m"][i] or 30.0)
        humidity = int(round(hourly["relative_humidity_2m"][i] or 70))
        rain_mm = float(hourly["rain"][i] or 0.0)
        weather_code = int(hourly["weather_code"][i] or 0)
        wind_speed = float(hourly["wind_speed_10m"][i] or 0.0)

        rain_pct = int(min(100, round(rain_mm * 12)))

        # Dynamic physics simulation
        if rain_mm > 5.0:
            current_tof -= random.uniform(80.0, 160.0)
        elif rain_mm > 0.5:
            current_tof -= random.uniform(20.0, 50.0)
        else:
            if current_tof < 1900.0:
                current_tof += random.uniform(70.0, 130.0)
            else:
                current_tof += random.uniform(-10.0, 10.0)

        current_tof = max(100.0, min(current_tof, 2000.0))
        water_level_ft = max(0.0, (2000.0 - current_tof) / 304.8)
        
        # Calculate numerical predicted rise in ft
        pred_rise_ft = round((rain_pct * 0.035) + (wind_speed * 0.010), 2)

        records.append({
            "rain_percent": rain_pct,
            "wind_speed_kmh": round(wind_speed, 2),
            "tof_distance_mm": round(current_tof, 2),
            "water_level_ft": round(water_level_ft, 2),
            "predicted_rise_ft": pred_rise_ft,
            "om_temp": round(temp, 2),
            "om_humidity": humidity,
            "om_weather_code": weather_code,
            "status": "Clear" if rain_pct <= 10 else "Raining"
        })

    batch_size = 100
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            supabase.table("sensor_data").insert(batch).execute()
            inserted += len(batch)
        except Exception as e:
            print(f"Error inserting batch: {e}")

    print(f"✅ Re-seeded {inserted} multi-state historical records into Supabase!\n")


def fetch_supabase_sensor_data(supabase: Client) -> pd.DataFrame:
    print("Connecting to Supabase database...")
    
    response = supabase.table("sensor_data").select("*").order("id", desc=False).range(0, 5000).execute()
    data = response.data

    if not data or len(data) < LOOKBACK_STEPS:
        seed_tanong_data_from_open_meteo(supabase)
        response = supabase.table("sensor_data").select("*").order("id", desc=False).range(0, 5000).execute()
        data = response.data

    df = pd.DataFrame(data)
    print(f"Retrieved {len(df)} sensor entries from Supabase.")

    # Fill missing values
    df["tof_distance_mm"] = df["tof_distance_mm"].ffill().fillna(2000.0)
    df["rain_percent"] = df["rain_percent"].fillna(0)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0.0)
    df["om_temp"] = df["om_temp"].fillna(30.0)
    df["om_humidity"] = df["om_humidity"].fillna(70)
    df["om_weather_code"] = df["om_weather_code"].fillna(0)

    if "status" in df.columns:
        clear_mask = df["status"].astype(str).str.upper().str.contains("CLEAR|DRY|NORMAL", na=False)
        df.loc[clear_mask, "rain_percent"] = 0

    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().round(2)
    df["tof_delta"] = df["tof_distance_mm"].diff().fillna(0.0).round(2)

    # Calculate numerical predicted rise in ft
    df["predicted_rise_ft"] = ((df["rain_3p_avg"] * 0.035) + (df["wind_speed_kmh"] * 0.010)).round(2)

    # Multi-class Heuristic Labels calibrated for urban drainage
    raw_risk = []
    for rain_avg, tof in zip(df["rain_3p_avg"], df["tof_distance_mm"]):
        if tof <= 600.0 or (rain_avg >= 50 and tof <= 1000.0):
            raw_risk.append(2)  # Flood Alert (HIGH)
        elif tof <= 1300.0 or (rain_avg >= 20 and tof <= 1600.0):
            raw_risk.append(1)  # Warning (MODERATE)
        else:
            raw_risk.append(0)  # Safe (LOW)

    df[TARGET_COLUMN] = (
        pd.Series(raw_risk)
        .rolling(window=3, min_periods=1, center=True)
        .median()
        .astype(int)
    )

    return df

def create_3d_sequences(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    feature_data = df[FEATURE_COLUMNS].values
    targets = df[TARGET_COLUMN].values

    scaled_features = scaler.fit_transform(feature_data) if fit_scaler else scaler.transform(feature_data)

    X_seq, y_seq, row_ids = [], [], []
    for i in range(len(df) - LOOKBACK_STEPS):
        X_seq.append(scaled_features[i : i + LOOKBACK_STEPS])
        y_seq.append(targets[i + LOOKBACK_STEPS])
        row_ids.append(df["id"].iloc[i + LOOKBACK_STEPS])

    if len(X_seq) == 0:
        return np.empty((0, LOOKBACK_STEPS, len(FEATURE_COLUMNS))), np.empty((0,)), []

    return np.array(X_seq), np.array(y_seq), row_ids


# ============================================================================
# 2. MODEL ARCHITECTURE & GRAPHING FUNCTIONS
# ============================================================================
def build_lstm(input_shape) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.25),
        Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.25),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_HISTORY_PATH, dpi=300)
    print(f"\n[INFO] Saved training history plot to {PLOT_HISTORY_PATH}")
    plt.show()
    plt.close()


def plot_confusion_matrix_chart(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_INDICES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('LSTM Flood Risk Prediction Confusion Matrix (Brgy. Tañong)')
    plt.ylabel('Actual Risk Level')
    plt.xlabel('Predicted Risk Level')
    
    plt.tight_layout()
    plt.savefig(PLOT_CM_PATH, dpi=300)
    print(f"[INFO] Saved confusion matrix plot to {PLOT_CM_PATH}")
    plt.show()
    plt.close()


# ============================================================================
# 3. FEEDBACK TO SUPABASE (BATCHED UPSERT)
# ============================================================================
def push_predictions_to_supabase(supabase: Client, row_ids: list, y_pred: np.ndarray, df: pd.DataFrame):
    print("\nUpdating Supabase database with LSTM flood prediction feedback & numeric rise...")
    
    # Map calculated predicted_rise_ft by ID
    rise_map = dict(zip(df["id"], df["predicted_rise_ft"]))

    payload = [
        {
            "id": int(r_id),
            "predicted_flood_risk": int(pred_code),
            "prediction_label": CLASS_NAMES[pred_code],
            "predicted_rise_ft": float(rise_map.get(r_id, 0.0))
        }
        for r_id, pred_code in zip(row_ids, y_pred)
    ]

    batch_size = 200
    updated_count = 0

    for i in range(0, len(payload), batch_size):
        batch = payload[i : i + batch_size]
        try:
            supabase.table("sensor_data").upsert(batch).execute()
            updated_count += len(batch)
        except Exception as e:
            print(f"Failed to upsert batch starting at index {i}: {e}")

    print(f"Successfully updated {updated_count} rows in Supabase.")


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================
def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    df = fetch_supabase_sensor_data(supabase)

    print("\nSupabase Class Breakdown:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = (df[TARGET_COLUMN] == cls_idx).sum()
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  - {cls_name}: {count} records ({pct:.1f}%)")

    # Chronological Split
    train_ratio = 0.80
    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"\nChronological Split: {len(df_train)} Train rows | {len(df_test)} Test rows")

    scaler = StandardScaler()

    X_train, y_train, train_ids = create_3d_sequences(df_train, scaler, fit_scaler=True)
    X_test, y_test, test_ids = create_3d_sequences(df_test, scaler, fit_scaler=False)

    X_all, y_all, all_row_ids = create_3d_sequences(df, scaler, fit_scaler=False)

    val_data = (X_test, y_test) if len(X_test) > 0 else None
    monitor_metric = "val_loss" if val_data else "loss"

    present_classes = np.unique(y_train)
    class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0}
    if len(present_classes) > 1:
        weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
        for cls, weight in zip(present_classes, weights):
            class_weight_dict[int(cls)] = float(weight)

    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor=monitor_metric, patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=3, min_lr=1e-5)

    print("\n--- Training LSTM Model ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=40,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    plot_training_history(history)

    if len(X_test) > 0:
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\n" + "=" * 60)
        print("LSTM MODEL EVALUATION METRICS (Chronological Test Set)")
        print("=" * 60)
        print(
            classification_report(
                y_test,
                y_pred,
                labels=CLASS_INDICES,
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )
        plot_confusion_matrix_chart(y_test, y_pred, CLASS_NAMES)

    # Predict full sequence set & push feedback to Supabase
    if len(X_all) > 0:
        all_pred_probs = model.predict(X_all)
        all_preds = np.argmax(all_pred_probs, axis=1)
        push_predictions_to_supabase(supabase, all_row_ids, all_preds, df)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    schema = {"feature_columns": FEATURE_COLUMNS, "lookback_steps": LOOKBACK_STEPS}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)

    print(f"\n[SUCCESS] Model trained and database feedback complete.")


if __name__ == "__main__":
    main()