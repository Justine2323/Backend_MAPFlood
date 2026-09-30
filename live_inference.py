"""
================================================================================
Continuous Real-Time Flood Risk Inference Engine - Barangay Tañong, Malabon City
================================================================================
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from supabase import create_client, Client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_flood_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "lstm_scaler.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"

CLASS_NAMES = ["Safe (LOW)", "Warning (MODERATE)", "Flood Alert (HIGH)"]
POLL_INTERVAL_SECONDS = 5  # Set to match your daemon polling frequency


def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(SCHEMA_PATH)):
        raise FileNotFoundError("Model artifacts missing. Run lstm_tanong_malabon.py first to train model.")

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, schema["feature_columns"], schema["lookback_steps"]


def process_latest_sensor_data(supabase: Client, model, scaler, feature_cols, lookback_steps):
    # Fetch latest entries from Supabase
    response = supabase.table("sensor_data").select("*").order("id", desc=True).limit(15).execute()
    data = response.data

    if not data or len(data) < lookback_steps:
        return

    # Sort to chronological order
    df = pd.DataFrame(data).sort_values("id").reset_index(drop=True)

    # Missing value cleaning
    df["tof_distance_mm"] = df["tof_distance_mm"].ffill().fillna(2000.0)
    df["rain_percent"] = df["rain_percent"].fillna(0)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0.0)
    df["om_temp"] = df["om_temp"].fillna(30.0)
    df["om_humidity"] = df["om_humidity"].fillna(70)
    df["om_weather_code"] = df["om_weather_code"].fillna(0)

    if "status" in df.columns:
        clear_mask = df["status"].astype(str).str.upper().str.contains("CLEAR|DRY|NORMAL", na=False)
        df.loc[clear_mask, "rain_percent"] = 0

    # Dynamic momentum calculations
    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().round(2)
    df["tof_delta"] = df["tof_distance_mm"].diff().fillna(0.0).round(2)

    # Extract sequence window
    latest_seq_df = df[feature_cols].tail(lookback_steps)
    latest_row = df.iloc[-1]
    latest_id = int(latest_row["id"])

    # Extract individual parameters for logging
    latest_tof = float(latest_row["tof_distance_mm"])
    latest_wind = float(latest_row["wind_speed_kmh"])
    latest_rain = int(latest_row["rain_percent"])
    timestamp = pd.Timestamp.now().isoformat()

    scaled_seq = scaler.transform(latest_seq_df.values)
    X_input = np.expand_dims(scaled_seq, axis=0)

    # Predict
    pred_probs = model.predict(X_input, verbose=0)
    pred_code = int(np.argmax(pred_probs, axis=1)[0])
    pred_label = CLASS_NAMES[pred_code]

    # Write back prediction results to Supabase
    supabase.table("sensor_data").update({
        "predicted_flood_risk": pred_code,
        "prediction_label": pred_label
    }).eq("id", latest_id).execute()

    # PRINT LOG FORMATTING (TOF, Wind, Rainfall)
    print(f"[{timestamp}] ID: {latest_id} | TOF: {latest_tof:.1f}mm | Wind: {latest_wind:.1f} km/h | Rain: {latest_rain}% -> Predicted: {pred_label}")


def main():
    print("🚀 Live Inference Daemon Started! Monitoring Supabase every 5s...\n")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    model, scaler, feature_cols, lookback_steps = load_artifacts()

    while True:
        try:
            process_latest_sensor_data(supabase, model, scaler, feature_cols, lookback_steps)
        except Exception as e:
            print(f"[{pd.Timestamp.now().isoformat()}] Error in inference loop: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()