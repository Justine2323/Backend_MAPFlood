import os
import requests
import numpy as np
import pandas as pd
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore

# --- 1. CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jqhimswayyurxymltrxn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing Supabase credentials.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. DATA ACQUISITION ---
def fetch_geographical_data() -> float:
    """Fetches real-time river discharge for Barangay Tañong, Malabon City."""
    om_url = "https://flood-api.open-meteo.com/v1/flood?latitude=14.6542&longitude=120.9508&daily=river_discharge&timezone=auto"
    try:
        res = requests.get(om_url, timeout=10).json()
        return res['daily']['river_discharge'][-1] if 'daily' in res else 0.0
    except Exception as e:
        print(f"Warning: Open-Meteo flood API fetch failed: {e}")
        return 0.0

def fetch_telemetry_data(supabase: Client, table_name: str = "StationB", limit: int = 300) -> pd.DataFrame:
    """Fetches telemetry data sent by the ESP32-S3."""
    response = supabase.table(table_name).select("*").order("id", desc=False).limit(limit).execute()
    data = response.data
    if not data or len(data) < 10:
        raise ValueError("Insufficient data records in Supabase for training.")
    return pd.DataFrame(data)

# --- 3. DATA PREPROCESSING ---
def preprocess_data(df: pd.DataFrame, river_discharge: float, seq_length: int = 5):
    """Cleans, scales, and formats features into LSTM sequences."""
    df['river_discharge'] = river_discharge
    features = ['rain_percent', 'wind_speed_kmh', 'water_level_ft', 'om_temp', 'river_discharge']
    df_features = df[features].fillna(0)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length][2])  # Target: index 2 (water_level_ft)

    return np.array(X), np.array(y), scaler, features, scaled_data

# --- 4. MODEL TRAINING ---
def train_and_evaluate_lstm(X: np.ndarray, y: np.ndarray, input_shape: tuple):
    """Trains the LSTM neural network and calculates performance accuracy."""
    if len(X) < 5:
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=8, verbose=0)

    predictions = model.predict(X_val, verbose=0)
    mape = mean_absolute_percentage_error(y_val + 1e-5, predictions + 1e-5) 
    accuracy_percentage = max(0.0, (1 - mape) * 100)

    return model, accuracy_percentage

# --- 5. INFERENCE & CLASSIFICATION ---
def predict_next_rise(model, scaler, scaled_data, features, seq_length: int = 5) -> float:
    """Calculates predicted water rise in feet."""
    last_sequence = scaled_data[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predicted_scaled = model.predict(last_sequence, verbose=0)

    dummy_array = np.zeros((1, len(features)))
    dummy_array[0][2] = predicted_scaled[0][0]
    predicted_ft = float(scaler.inverse_transform(dummy_array)[0][2])
    return max(0.0, predicted_ft)

def classify_risk(predicted_ft: float) -> tuple:
    if predicted_ft >= 2.5:
        return "DANGER", 3
    elif predicted_ft >= 1.5:
        return "ALERT", 2
    elif predicted_ft >= 0.5:
        return "WARNING", 1
    return "SAFE", 0

# --- 6. DATABASE SYNC ---
def update_prediction(supabase: Client, table_name: str, record_id: int, predicted_ft: float, label: str, risk: int, accuracy: float):
    """Pushes predicted values back to Supabase for ESP32 consumption."""
    supabase.table(table_name).update({
        "predicted_rise_ft": round(predicted_ft, 2),
        "predicted_flood_risk": risk,
        "prediction_label": label
    }).eq("id", record_id).execute()
    print(f"[SUCCESS] Updated ID {record_id} | Prediction: {predicted_ft:.2f}ft | Label: {label} | Acc: {accuracy:.2f}%")

# --- MAIN EXECUTION PIPELINE ---
def main():
    try:
        supabase = get_supabase_client()
        
        river_discharge = fetch_geographical_data()
        df = fetch_telemetry_data(supabase, table_name="StationB")
        
        seq_length = 5
        X, y, scaler, features, scaled_data = preprocess_data(df, river_discharge, seq_length)
        
        model, accuracy = train_and_evaluate_lstm(X, y, input_shape=(seq_length, len(features)))
        
        predicted_ft = predict_next_rise(model, scaler, scaled_data, features, seq_length)
        label, risk = classify_risk(predicted_ft)
        
        latest_id = df.iloc[-1]['id']
        update_prediction(supabase, "StationB", latest_id, predicted_ft, label, risk, accuracy)

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")

if __name__ == "__main__":
    main()