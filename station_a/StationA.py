import os
import numpy as np
import pandas as pd
import requests
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore

# ============================================================================
# CONFIGURATION
# ============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "Station A" # Target Station A

# ADJUSTABLE PREDICTION WINDOW
PREDICT_AHEAD_MINUTES = 30  # Forecast 30 minutes into the future
DATA_INTERVAL_SEC = 5       # ESP32 sends data every 5 seconds
STEPS_AHEAD = (PREDICT_AHEAD_MINUTES * 60) // DATA_INTERVAL_SEC # Calculate offset (360 rows)

# Tañong, Malabon Coordinates
LAT = 14.6542
LON = 120.9508

SENSOR_MOUNT_HEIGHT_MM = 1828.8 
LOOKBACK_STEPS = 6  

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not found in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# API HELPERS
# ============================================================================
def fetch_flood_api():
    """Fetches real-time river discharge data from Open-Meteo Flood API"""
    url = f"https://flood-api.open-meteo.com/v1/flood?latitude={LAT}&longitude={LON}&daily=river_discharge,river_discharge_mean,river_discharge_max,river_discharge_min&timezone=Asia%2FManila&forecast_days=1"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        discharge = daily.get("river_discharge", [0.0])[0]
        discharge_max = daily.get("river_discharge_max", [0.0])[0]
        
        return discharge, discharge_max
    except Exception as e:
        print(f"[WARNING] Failed to fetch Flood API data: {e}")
        return 0.0, 0.0

# ============================================================================
# MODEL SETUP
# ============================================================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def get_risk_label(predicted_level, current_level):
    predicted_rise = predicted_level - current_level
    if predicted_level >= 3.0 or predicted_rise >= 2.0:
        return 3, "DANGER"
    elif predicted_level >= 2.0 or predicted_rise >= 1.0:
        return 2, "WARNING"
    elif predicted_level >= 1.0 or predicted_rise >= 0.5:
        return 1, "CAUTION"
    else:
        return 0, "SAFE"

# ============================================================================
# SINGLE EXECUTION RUN
# ============================================================================
def run_prediction():
    print(f"Running LSTM Prediction Engine for {TABLE_NAME} (Forecast: +{PREDICT_AHEAD_MINUTES} mins)...")
    scaler = MinMaxScaler()
    
    # Fetch 1500 rows to ensure we have enough historical data for the 30min offset (360 rows)
    response = supabase.table(TABLE_NAME).select("*").order("id", desc=True).limit(1500).execute()
    data = response.data
    
    if len(data) < LOOKBACK_STEPS + STEPS_AHEAD + 1:
        print(f"Not enough data for a {PREDICT_AHEAD_MINUTES}m prediction. Need {LOOKBACK_STEPS + STEPS_AHEAD + 1} rows, currently have {len(data)}.")
        return
        
    df = pd.DataFrame(data)
    df = df.sort_values(by="id").reset_index(drop=True)
    
    if 'water_level_ft' not in df.columns:
        df['water_level_ft'] = 0.0
        
    df['calculated_water_level_ft'] = df['water_level_ft'].fillna(0.0).clip(lower=0.0)
    
    discharge, discharge_max = fetch_flood_api()
    df['river_discharge'] = discharge
    df['river_discharge_max'] = discharge_max
    
    features = [
        'calculated_water_level_ft', 
        'rain_percent', 
        'wind_speed_kmh', 
        'om_rain', 
        'om_pressure', 
        'river_discharge', 
        'river_discharge_max'
    ]
    df_features = df[features].fillna(0)
    
    scaled_data = scaler.fit_transform(df_features)
    
    # Shift the target 'y' forward by 'STEPS_AHEAD' to train the model to predict 30 minutes in the future
    X, y = [], []
    for i in range(len(scaled_data) - LOOKBACK_STEPS - STEPS_AHEAD):
        X.append(scaled_data[i:(i + LOOKBACK_STEPS)])
        y.append(scaled_data[i + LOOKBACK_STEPS + STEPS_AHEAD, 0]) 
        
    X = np.array(X)
    y = np.array(y)
    
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=4, verbose=0)
    
    latest_sequence = scaled_data[-LOOKBACK_STEPS:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)
    
    scaled_prediction = model.predict(latest_sequence, verbose=0)[0][0]
    
    dummy_array = np.zeros((1, len(features)))
    dummy_array[0, 0] = scaled_prediction
    predicted_water_level_ft = scaler.inverse_transform(dummy_array)[0, 0]
    
    current_water_level_ft = df['calculated_water_level_ft'].iloc[-1]
    predicted_rise_ft = float(predicted_water_level_ft - current_water_level_ft)
    
    if predicted_rise_ft < 0: 
        predicted_rise_ft = 0.0
        
    risk_code, risk_label = get_risk_label(predicted_water_level_ft, current_water_level_ft)
    latest_id = int(df['id'].iloc[-1])
    
    update_payload = {
        "predicted_rise_ft": round(predicted_rise_ft, 2),
        "predicted_flood_risk": risk_code,
        "prediction_label": risk_label
    }
    
    supabase.table(TABLE_NAME).update(update_payload).eq("id", latest_id).execute()
    print(f"[{pd.Timestamp.now()}] ID: {latest_id} | Current: {current_water_level_ft:.2f}ft | +{PREDICT_AHEAD_MINUTES}m Pred Rise: {predicted_rise_ft:.2f}ft | Status: {risk_label}")

if __name__ == "__main__":
    run_prediction()