import os
import time
import json
import joblib
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from datetime import datetime
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Dropout, LayerNormalization, Input

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqhimswayyurxymltrxn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S")

LOOKBACK_STEPS = 8
PREDICTION_HORIZON_STEPS = 2
SLEEP_INTERVAL_SECONDS = 3600

SENSOR_MOUNT_HEIGHT_FT_A = 6.0 

# Updated features to include all requested Flood API ensemble variables
BASE_FEATURES = [
    "rain_percent", "wind_speed_kmh", "om_temp", "om_humidity", 
    "om_rain", "om_pressure", "rain_3p_avg", 
    "river_discharge", "river_discharge_mean", "river_discharge_median",
    "river_discharge_max", "river_discharge_min", "river_discharge_p25", "river_discharge_p75"
]
TARGET_COLUMN = "predicted_rise_ft"

# ============================================================================
# 0. CUSTOM ASYMMETRIC LOSS FUNCTION 
# ============================================================================
@keras.saving.register_keras_serializable(package="Custom")
def asymmetric_mse(y_true, y_pred):
    error = y_true - y_pred
    weight = tf.where(error > 0, 3.5, 1.0)
    return tf.reduce_mean(weight * tf.square(error))

# ============================================================================
# MODULE 1: STATION-SPECIFIC DATA PREPARATION
# ============================================================================
def fetch_flood_api_data() -> dict:
    url = "https://flood-api.open-meteo.com/v1/flood?latitude=14.6577&longitude=120.9537&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min,river_discharge_p25,river_discharge_p75&ensemble=true&models=seamless_v4"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})
        
        # Grab the latest daily entry [-1] for all ensemble variables
        return {
            "river_discharge": float(daily.get("river_discharge", [0])[-1] or 0.0),
            "river_discharge_mean": float(daily.get("river_discharge_mean", [0])[-1] or 0.0),
            "river_discharge_median": float(daily.get("river_discharge_median", [0])[-1] or 0.0),
            "river_discharge_max": float(daily.get("river_discharge_max", [0])[-1] or 0.0),
            "river_discharge_min": float(daily.get("river_discharge_min", [0])[-1] or 0.0),
            "river_discharge_p25": float(daily.get("river_discharge_p25", [0])[-1] or 0.0),
            "river_discharge_p75": float(daily.get("river_discharge_p75", [0])[-1] or 0.0)
        }
    except Exception as e:
        print(f"[WARNING] Failed to fetch Flood API data: {e}")
        # Return safe zeros if API fails
        return {k: 0.0 for k in [
            "river_discharge", "river_discharge_mean", "river_discharge_median", 
            "river_discharge_max", "river_discharge_min", "river_discharge_p25", "river_discharge_p75"
        ]}

def prep_station_a(df: pd.DataFrame) -> pd.DataFrame:
    df["tof_distance_ft"] = df["tof_distance_ft"].ffill().fillna(SENSOR_MOUNT_HEIGHT_FT_A)
    df["actual_water_level_ft"] = (SENSOR_MOUNT_HEIGHT_FT_A - df["tof_distance_ft"]).clip(lower=0)
    df["sensor_delta"] = df["tof_distance_ft"].diff().fillna(0.0)
    return df, BASE_FEATURES + ["tof_distance_ft", "sensor_delta"]

def prep_station_b(df: pd.DataFrame) -> pd.DataFrame:
    df["float_tier"] = df["float_tier"].ffill().fillna(0)
    df["actual_water_level_ft"] = df["float_tier"] * 1.5 
    df["sensor_delta"] = df["float_tier"].diff().fillna(0.0)
    return df, BASE_FEATURES + ["float_tier", "sensor_delta"]

def prep_station_c(df: pd.DataFrame) -> pd.DataFrame:
    # Safely create camera_est_ft from water_tier if it doesn't exist
    if "camera_est_ft" not in df.columns:
        if "water_tier" in df.columns:
            df["camera_est_ft"] = df["water_tier"].astype(float)
        else:
            df["camera_est_ft"] = 0.0

    df["camera_est_ft"] = df["camera_est_ft"].ffill().fillna(0.0)
    df["actual_water_level_ft"] = df["camera_est_ft"]
    df["sensor_delta"] = df["camera_est_ft"].diff().fillna(0.0)
    return df, BASE_FEATURES + ["camera_est_ft", "sensor_delta"]

def engineer_shared_features(df: pd.DataFrame, flood_metrics: dict) -> pd.DataFrame:
    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().fillna(0)
    
    # Broadcast API ensemble metrics into columns
    for key, val in flood_metrics.items():
        df[key] = val
        
    df["future_water_level_ft"] = df["actual_water_level_ft"].shift(-PREDICTION_HORIZON_STEPS)
    df[TARGET_COLUMN] = (df["future_water_level_ft"] - df["actual_water_level_ft"]).clip(lower=0)
    return df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

# ============================================================================
# MODULE 2: MODEL ARCHITECTURE & ACCURACY
# ============================================================================
def build_lstm_model(input_shape) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.1)),
        LayerNormalization(),
        Dropout(0.25),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=asymmetric_mse, metrics=["mae"])
    return model

def calculate_accuracy_percentage(y_true, y_pred) -> float:
    mae = np.mean(np.abs(y_true - y_pred))
    accuracy = max(0.0, 100.0 - ((mae / 3.0) * 100.0))
    return round(accuracy, 2)

# ============================================================================
# MODULE 3: PIPELINE EXECUTION PER STATION
# ============================================================================
def process_station(supabase: Client, table_name: str, prep_func, flood_metrics: dict):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing {table_name}...")
    
    response = supabase.table(table_name).select("*").order("id", desc=True).limit(2000).execute()
    df = pd.DataFrame(response.data)
    if df.empty or len(df) < (LOOKBACK_STEPS + PREDICTION_HORIZON_STEPS):
        print(f"[SKIP] Insufficient data for {table_name}.")
        return

    df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
    
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df, feature_cols = prep_func(df)
    df = engineer_shared_features(df, flood_metrics)
    
    clean_name = table_name.replace(" ", "_")
    scaler_path = f"scaler_{clean_name}.joblib"
    model_path = f"model_{clean_name}.keras"
    
    need_new_scaler = True
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
                scaled_features = scaler.transform(df[feature_cols].values)
                need_new_scaler = False
        except Exception:
            pass

    if need_new_scaler:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_cols].values)
        joblib.dump(scaler, scaler_path)
        if os.path.exists(model_path):
            os.remove(model_path) 

    X_seq, y_seq, row_ids = [], [], []
    for i in range(len(df) - LOOKBACK_STEPS + 1):
        X_seq.append(scaled_features[i : i + LOOKBACK_STEPS])
        y_seq.append(df[TARGET_COLUMN].iloc[i + LOOKBACK_STEPS - 1])
        row_ids.append(df["id"].iloc[i + LOOKBACK_STEPS - 1])
        
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    if os.path.exists(model_path):
        try:
            model = load_model(model_path, custom_objects={"asymmetric_mse": asymmetric_mse})
        except Exception:
            model = build_lstm_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
    else:
        model = build_lstm_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        model.fit(X_seq, y_seq, epochs=5, batch_size=16, verbose=0)
        model.save(model_path)

    predictions = model.predict(X_seq, verbose=0).flatten()
    accuracy_pct = calculate_accuracy_percentage(y_seq, predictions)
    print(f"[{table_name}] Model Accuracy: {accuracy_pct}%")

    payload = []
    for r_id, pred in zip(row_ids, predictions):
        pred_val = round(float(pred), 2)
        
        if pred_val >= 3.0:
            risk_code = 3
            label = "DANGER"
        elif pred_val >= 2.0:
            risk_code = 2
            label = "ALERT"
        elif pred_val >= 1.0:
            risk_code = 1
            label = "WARNING"
        else:
            risk_code = 0
            label = "SAFE"

        payload.append({
            "id": int(r_id),
            "predicted_rise_ft": pred_val,
            "prediction_accuracy_pct": accuracy_pct,
            "predicted_flood_risk": risk_code,
            "prediction_label": label
        })

    for i in range(0, len(payload), 200):
        supabase.table(table_name).upsert(payload[i : i + 200]).execute()
    print(f"[{table_name}] Successfully updated {len(payload)} rows in Supabase.")

# ============================================================================
# MAIN PIPELINE EXECUTION (SINGLE-PASS FOR GITHUB ACTIONS)
# ============================================================================
if __name__ == "__main__":
    print("=====================================================")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting LSTM Prediction Pass")
    print("=====================================================")
    
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        flood_metrics = fetch_flood_api_data()
        print(f"[INFO] Current Core River Discharge: {flood_metrics['river_discharge']} m³/s")
        print(f"[INFO] Current Mean Discharge: {flood_metrics['river_discharge_mean']} m³/s")

        process_station(supabase_client, "Station A", prep_station_a, flood_metrics)
        process_station(supabase_client, "Station B", prep_station_b, flood_metrics)
        process_station(supabase_client, "Station C", prep_station_c, flood_metrics)
        
        print("\n[SUCCESS] Pipeline completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")
        exit(1)
