import json
import os
import joblib
import numpy as np
import pandas as pd
import requests

# Model directories (points to the flood_svm_outputs folder)
MODEL_DIR = os.path.dirname(__file__) if os.path.basename(os.getcwd()) == "flood_svm_outputs" else "flood_svm_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_flood_classification_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")

RISK_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
RISK_ADVICE = {
    0: "Normal conditions in Barangay Tañong. Continue routine monitoring.",
    1: "ELEVATED RISK: Heavy rainfall or rising Tullahan river levels. Notify local Malabon DRRMO authorities.",
    2: "FLOOD WARNING: Critical water/rain levels in Tañong. Prepare for immediate evacuation of low-lying areas.",
}

def fetch_live_tanong_weather() -> dict:
    """Fetches real-time hourly forecast data for Tañong, Malabon City from Open-Meteo."""
    LATITUDE = 14.6542
    LONGITUDE = 120.9508

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"current=precipitation,relative_humidity_2m,soil_moisture_0_to_7cm,surface_pressure&"
        f"hourly=precipitation&past_days=1&forecast_days=1&"
        f"timezone=Asia%2FManila"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to reach weather API: {response.text}")

    data = response.json()
    current = data["current"]
    hourly_precip = data["hourly"]["precipitation"]

    # Calculate last 24-hour accumulated rainfall (mm)
    rainfall_24h_mm = round(float(sum(hourly_precip[-24:])), 2)
    soil_moisture_pct = round(float(current["soil_moisture_0_to_7cm"] * 100), 2)
    humidity_pct = round(float(current["relative_humidity_2m"]), 2)
    surface_pressure_hpa = round(float(current["surface_pressure"]), 2)

    # Calculate discharge and river water level proxy
    upstream_discharge_m3s = round(
        30 + (rainfall_24h_mm * (1 + (soil_moisture_pct / 100) ** 2) * 1.5), 2
    )
    river_water_level_m = round(
        0.8 + 0.85 * np.log1p(upstream_discharge_m3s / 40) + 0.01 * rainfall_24h_mm, 2
    )

    return {
        "rainfall_24h_mm": rainfall_24h_mm,
        "river_water_level_m": river_water_level_m,
        "soil_moisture_pct": soil_moisture_pct,
        "humidity_pct": humidity_pct,
        "surface_pressure_hpa": surface_pressure_hpa,
        "upstream_discharge_m3s": upstream_discharge_m3s,
    }

def run_live_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Model or scaler joblib file missing from '{MODEL_DIR}'. Run flood_svm_thesis.py first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    feature_order = schema["feature_order"]

    print("Fetching live weather conditions for Tañong, Malabon City...")
    live_data = fetch_live_tanong_weather()

    # Format features in exact schema order
    input_features = np.array([[live_data[f] for f in feature_order]])
    scaled_features = scaler.transform(input_features)

    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]

    print("\n" + "=" * 60)
    print("LIVE FLOOD RISK ASSESSMENT: BARANGAY TAÑONG, MALABON CITY")
    print("=" * 60)
    print(f"Current 24h Accumulated Rainfall: {live_data['rainfall_24h_mm']} mm")
    print(f"Current Soil Saturation:         {live_data['soil_moisture_pct']} %")
    print(f"Estimated River Water Level:     {live_data['river_water_level_m']} m")
    print(f"Estimated Upstream Discharge:    {live_data['upstream_discharge_m3s']} m³/s")
    print("-" * 60)
    print(f"PREDICTED RISK LEVEL: {RISK_LABELS[prediction]}")
    print(f"Confidence Score:     {round(float(np.max(probabilities) * 100), 2)}%")
    print(f"Action Protocol:     {RISK_ADVICE[prediction]}")
    print("=" * 60)

if __name__ == "__main__":
    run_live_prediction()