"""
================================================================================
Continuous LSTM Water Rise Regression Model - Barangay Tañong, Malabon City
================================================================================
"""

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

# Correct direct Keras imports for TF 2.21+ / Keras 3
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Input, LSTM
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
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"

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
# Targeting the continuous numerical rise
TARGET_COLUMN = "predicted_rise_ft" 
LOOKBACK_STEPS = 8
SLEEP_INTERVAL_SECONDS = 3600  # Run every 1 hour continuously


# ============================================================================
# 1. DATA SEEDER & SUPABASE FETCHING
# ============================================================================
def fetch_supabase_sensor_data(supabase: Client) -> pd.DataFrame:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching Supabase database...")
    
    # Fetch up to 10k latest rows for long-term operation
    response = supabase.table("sensor_data").select("*").order("id", desc=True).limit(10000).execute()
    data = response.data

    if not data or len(data) < LOOKBACK_STEPS:
        print("[WARNING] Not enough data in Supabase. Please ensure the ESP32 is sending data.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Sort back to chronological for LSTM sequence generation
    df = df.sort_values(by="id", ascending=True).reset_index(drop=True)

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

    # Ground truth baseline computation (what the model learns to predict)
    df["predicted_rise_ft"] = ((df["rain_3p_avg"] * 0.035) + (df["wind_speed_kmh"] * 0.010)).round(2)

    print(f"[INFO] Successfully loaded and preprocessed {len(df)} rows.")
    return df

def create_3d_sequences(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    feature_data = df[FEATURE_COLUMNS].values
    targets = df[TARGET_COLUMN].values

    if fit_scaler:
        scaled_features = scaler.fit_transform(feature_data)
    else:
        scaled_features = scaler.transform(feature_data)

    X_seq, y_seq, row_ids = [], [], []
    for i in range(len(df) - LOOKBACK_STEPS):
        X_seq.append(scaled_features[i : i + LOOKBACK_STEPS])
        y_seq.append(targets[i + LOOKBACK_STEPS])
        row_ids.append(df["id"].iloc[i + LOOKBACK_STEPS])

    if len(X_seq) == 0:
        return np.empty((0, LOOKBACK_STEPS, len(FEATURE_COLUMNS))), np.empty((0,)), []

    return np.array(X_seq), np.array(y_seq), row_ids


# ============================================================================
# 2. MODEL ARCHITECTURE (REGRESSION) & GRAPHING
# ============================================================================
def build_lstm_regression(input_shape) -> Sequential:
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
        # Single output neuron with Linear activation for exact float values (feet)
        Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",  
        metrics=["mae"], 
    )
    return model


def plot_regression_metrics(history, y_true, y_pred):
    # 1. Training History Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE', color='blue')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
    plt.title('LSTM Model Error (Feet)')
    plt.ylabel('Mean Absolute Error (ft)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss (MSE)', color='orange')
    plt.title('LSTM Model Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_HISTORY_PATH, dpi=300)
    plt.close()

    # 2. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title('Actual vs Predicted Water Rise (ft)')
    plt.xlabel('Actual Rise (ft)')
    plt.ylabel('Predicted Rise (ft)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER_PATH, dpi=300)
    plt.close()
    print(f"[INFO] Saved regression plots to {BASE_DIR}")


# ============================================================================
# 3. FEEDBACK TO SUPABASE (BATCHED UPSERT)
# ============================================================================
def push_predictions_to_supabase(supabase: Client, row_ids: list, y_pred: np.ndarray):
    print("\n[INFO] Updating Supabase database with exact LSTM predicted rise in feet...")
    
    y_pred_flat = y_pred.flatten()
    payload = []
    
    for r_id, pred_rise in zip(row_ids, y_pred_flat):
        if pred_rise >= 2.0:
            risk_code = 2
            label = "Flood Alert (HIGH)"
        elif pred_rise >= 1.0:
            risk_code = 1
            label = "Warning (MODERATE)"
        else:
            risk_code = 0
            label = "Safe (LOW)"

        payload.append({
            "id": int(r_id),
            "predicted_rise_ft": round(float(pred_rise), 2),
            "predicted_flood_risk": risk_code,
            "prediction_label": label
        })

    batch_size = 200
    updated_count = 0

    for i in range(0, len(payload), batch_size):
        batch = payload[i : i + batch_size]
        try:
            supabase.table("sensor_data").upsert(batch).execute()
            updated_count += len(batch)
        except Exception as e:
            print(f"[ERROR] Failed to upsert batch starting at index {i}: {e}")

    print(f"[INFO] Successfully updated {updated_count} rows in Supabase.")


# ============================================================================
# 4. MAIN PIPELINE (LONG-TERM EXECUTION LOOP)
# ============================================================================
def run_training_cycle():
    print("\n--- Starting New Data Fetch and Training Cycle ---")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    df = fetch_supabase_sensor_data(supabase)

    if df.empty:
        print("[WARNING] DataFrame is empty. Aborting this cycle.")
        return

    # Chronological Split
    train_ratio = 0.85
    split_idx = int(len(df) * train_ratio)

    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[INFO] Data Split: {len(df_train)} Train rows | {len(df_test)} Test rows")

    # Handle Scaler Continuous loading
    if os.path.exists(SCALER_PATH):
        print("[INFO] Found existing scaler. Loading...")
        scaler = joblib.load(SCALER_PATH)
        X_train, y_train, train_ids = create_3d_sequences(df_train, scaler, fit_scaler=False)
    else:
        print("[INFO] No existing scaler found. Fitting new scaler...")
        scaler = StandardScaler()
        X_train, y_train, train_ids = create_3d_sequences(df_train, scaler, fit_scaler=True)

    X_test, y_test, test_ids = create_3d_sequences(df_test, scaler, fit_scaler=False)
    X_all, y_all, all_row_ids = create_3d_sequences(df, scaler, fit_scaler=False)

    val_data = (X_test, y_test) if len(X_test) > 0 else None

    # Load existing model if operating long-term, otherwise build new
    if os.path.exists(MODEL_PATH):
        print(f"\n[INFO] Found existing model at {MODEL_PATH}. Loading weights for continuous training...")
        model = load_model(MODEL_PATH)
    else:
        print("\n[INFO] No existing model found. Building new architecture...")
        model = build_lstm_regression(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

    print("\n[INFO] --- Training / Fine-tuning LSTM Model ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=15, 
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print(f"REGRESSION EVALUATION METRICS")
        print("=" * 60)
        print(f"Mean Absolute Error (MAE) : {mae:.4f} ft")
        print(f"Mean Squared Error (MSE)  : {mse:.4f}")
        print(f"R2 Score                  : {r2:.4f}")
        
        plot_regression_metrics(history, y_test, y_pred)

    # Predict full sequence set & push continuous feedback to Supabase
    if len(X_all) > 0:
        all_preds = model.predict(X_all)
        push_predictions_to_supabase(supabase, all_row_ids, all_preds)

    # Save progress for the next cycle
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    schema = {"feature_columns": FEATURE_COLUMNS, "lookback_steps": LOOKBACK_STEPS}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)

    print(f"\n[SUCCESS] Cycle complete. Weights saved.")


if __name__ == "__main__":
    print("=====================================================")
    print("Starting Continuous LSTM Prediction Engine")
    print("=====================================================")
    
    while True:
        try:
            run_training_cycle()
            minutes = SLEEP_INTERVAL_SECONDS // 60
            print(f"\n[INFO] Sleeping for {minutes} minutes before the next cycle...")
            time.sleep(SLEEP_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n[INFO] Script stopped manually by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred during the cycle: {e}")
            print(f"[INFO] Retrying in 5 minutes...")
            time.sleep(300)