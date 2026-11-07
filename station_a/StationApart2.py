import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from supabase import create_client
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ============================================================
# CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "Station A"

PREDICT_AHEAD_MINUTES = 30
SAMPLE_MINUTES = 1
HORIZON_STEPS = PREDICT_AHEAD_MINUTES // SAMPLE_MINUTES
LOOKBACK_STEPS = 6

NAIVE_TIMESTAMP_TIMEZONE = "Asia/Manila"
MAX_READING_AGE_SECONDS = 180
PAGE_SIZE = 1000

# Fixed: double underscores for __file__
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "station_a_lstm_30min"

MODEL_PATH = ARTIFACT_DIR / "model.keras"
SCALER_PATH = ARTIFACT_DIR / "scalers.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

OPTIONAL_FEATURES = ["rain_percent", "wind_speed_kmh", "om_rain", "om_pressure"]

# ============================================================
# CONNECTION & UTILS
# ============================================================

def get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_timestamp(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Invalid created_at timestamp.")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NAIVE_TIMESTAMP_TIMEZONE, ambiguous="raise", nonexistent="raise")
    return timestamp.tz_convert("UTC")

def fetch_rows(client, limit):
    rows = []
    offset = 0
    while len(rows) < limit:
        count = min(PAGE_SIZE, limit - len(rows))
        response = client.table(TABLE_NAME).select("*").order("id", desc=True).range(offset, offset + count - 1).execute()
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows

def prepare_data(rows, feature_columns=None):
    df = pd.DataFrame(rows)
    required = ["id", "created_at", "water_level_ft"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    if df.empty:
        raise ValueError("No telemetry available.")
        
    df["timestamp"] = df["created_at"].map(parse_timestamp)
    df = df.sort_values(["timestamp", "id"]).drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    if feature_columns is None:
        feature_columns = ["water_level_ft"]
        for column in OPTIONAL_FEATURES:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="raise")
                if values.notna().all():
                    feature_columns.append(column)

    for column in feature_columns:
        if column not in df.columns:
            raise ValueError(f"Missing trained feature: {column}")
        df[column] = pd.to_numeric(df[column], errors="raise")
        values = df[column].dropna().to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {column}.")

    if (df["water_level_ft"].dropna() < 0).any():
        raise ValueError("Negative water level found. Check calibration and units.")

    indexed = df.set_index("timestamp")
    grid = indexed[feature_columns].resample("1min", closed="right", label="right").last()
    last_complete_minute = indexed.index.max().floor("min")
    grid = grid.loc[:last_complete_minute]

    return df, grid, feature_columns

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, activation="relu", return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def build_sequences(grid):
    X, y, origins, current_levels = [], [], [], []
    for end in range(LOOKBACK_STEPS - 1, len(grid) - HORIZON_STEPS):
        start = end - LOOKBACK_STEPS + 1
        target = end + HORIZON_STEPS
        window = grid.iloc[start:end + 1]
        level_span = grid["water_level_ft"].iloc[start:target + 1]

        if window.isna().any().any() or level_span.isna().any():
            continue

        X.append(window.to_numpy(dtype=float))
        y.append(float(grid["water_level_ft"].iloc[target]))
        origins.append(grid.index[end])
        current_levels.append(float(grid["water_level_ft"].iloc[end]))

    if not X:
        raise ValueError("No complete 30-minute training pairs. Collect more continuous telemetry.")
    return np.asarray(X), np.asarray(y), pd.DatetimeIndex(origins), np.asarray(current_levels)

def calculate_metrics(actual, predicted):
    return {
        "mae_ft": float(mean_absolute_error(actual, predicted)),
        "rmse_ft": float(np.sqrt(mean_squared_error(actual, predicted))),
    }

def train_model(client, row_limit, epochs):
    print("Starting training process...")
    rows = fetch_rows(client, row_limit)
    _, grid, feature_columns = prepare_data(rows)
    X, y, origins, current_levels = build_sequences(grid)

    if len(X) < 240:
        raise ValueError(f"Only {len(X)} usable training pairs. At least 240 are required.")

    split_index = int(len(X) * 0.8)
    test_start = origins[split_index]

    training_mask = origins + pd.Timedelta(minutes=PREDICT_AHEAD_MINUTES) < test_start
    testing_mask = origins >= test_start

    if training_mask.sum() < 120 or testing_mask.sum() < 30:
        raise ValueError("Insufficient training/test data after the 30-minute separation.")

    X_train, y_train = X[training_mask], y[training_mask]
    X_test, y_test = X[testing_mask], y[testing_mask]

    if np.std(y_train) < 1e-6:
        raise ValueError("Training water levels are effectively constant.")

    feature_scaler, target_scaler = MinMaxScaler(), MinMaxScaler()
    number_of_features = len(feature_columns)

    feature_scaler.fit(X_train.reshape(-1, number_of_features))
    target_scaler.fit(y_train.reshape(-1, 1))

    def scale_inputs(values):
        original_shape = values.shape
        return feature_scaler.transform(values.reshape(-1, number_of_features)).reshape(original_shape)

    X_train_scaled = scale_inputs(X_train)
    X_test_scaled = scale_inputs(X_test)
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1))

    model = build_lstm_model((LOOKBACK_STEPS, number_of_features))
    model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=32, shuffle=False, verbose=1)

    predicted_scaled = model.predict(X_test_scaled, verbose=0)
    predicted_levels = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
    predicted_levels = np.maximum(0.0, predicted_levels)

    baseline_levels = current_levels[testing_mask]
    model_metrics = calculate_metrics(y_test, predicted_levels)
    baseline_metrics = calculate_metrics(y_test, baseline_levels)

    report = {
        "model_type": "LSTM",
        "horizon_minutes": PREDICT_AHEAD_MINUTES,
        "sampling_minutes": SAMPLE_MINUTES,
        "lookback_steps": LOOKBACK_STEPS,
        "training_samples": int(training_mask.sum()),
        "test_samples": int(testing_mask.sum()),
        "first_test_origin": test_start.isoformat(),
        "model": model_metrics,
        "persistence_baseline": baseline_metrics,
        "beats_persistence_mae": model_metrics["mae_ft"] < baseline_metrics["mae_ft"],
    }

    metadata = {
        "feature_columns": feature_columns,
        "lookback_steps": LOOKBACK_STEPS,
        "horizon_minutes": PREDICT_AHEAD_MINUTES,
        "naive_timestamp_timezone": NAIVE_TIMESTAMP_TIMEZONE,
        "evaluation": report,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, SCALER_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTRAINING RESULTS:")
    print(json.dumps(report, indent=2))
    print("\nSaved model and artifacts successfully.")

def get_risk_label(predicted_level, current_level):
    predicted_change = predicted_level - current_level
    if predicted_level >= 3.0 or predicted_change >= 2.0: return 3, "DANGER"
    if predicted_level >= 2.0 or predicted_change >= 1.0: return 2, "WARNING"
    if predicted_level >= 1.0 or predicted_change >= 0.5: return 1, "CAUTION"
    return 0, "SAFE"

def run_prediction(client, row_limit, write):
    required_files = [MODEL_PATH, SCALER_PATH, METADATA_PATH]
    if not all(path.exists() for path in required_files):
        raise ValueError("Saved model is missing. Run --train first locally and push artifacts to GitHub.")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    scalers = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, compile=False)

    rows = fetch_rows(client, row_limit)
    df, grid, _ = prepare_data(rows, feature_columns=metadata["feature_columns"])

    if len(grid) < LOOKBACK_STEPS:
        raise ValueError("Not enough recent minute-level readings.")

    latest_window = grid.iloc[-LOOKBACK_STEPS:]
    if latest_window.isna().any().any():
        raise ValueError("Recent input history has missing readings.")

    issued_at = latest_window.index[-1]
    now = pd.Timestamp.now(tz="UTC")
    age = (now - issued_at).total_seconds()

    # Log warning instead of hard crash for stale data during GitHub automation
    if age < 0 or age > MAX_READING_AGE_SECONDS:
        print(f"[WARNING] Telemetry is {age}s old. Check sensor updates.", file=sys.stderr)

    scaled_window = scalers["feature_scaler"].transform(latest_window.to_numpy(dtype=float))
    model_input = np.expand_dims(scaled_window, axis=0)
    scaled_prediction = model.predict(model_input, verbose=0)
    
    predicted_level = float(scalers["target_scaler"].inverse_transform(np.asarray(scaled_prediction).reshape(-1, 1))[0, 0])
    predicted_level = max(0.0, predicted_level)

    current_level = float(latest_window["water_level_ft"].iloc[-1])
    predicted_change = predicted_level - current_level
    predicted_rise = max(0.0, predicted_change)
    risk_code, risk_label = get_risk_label(predicted_level, current_level)

    source_rows = df.loc[df["timestamp"] <= issued_at]
    source_id = int(source_rows.iloc[-1]["id"])
    target_at = issued_at + pd.Timedelta(minutes=PREDICT_AHEAD_MINUTES)

    if write:
        payload = {
            "predicted_rise_ft": round(predicted_rise, 6),
            "predicted_flood_risk": risk_code,
            "prediction_label": risk_label,
        }
        update_response = client.table(TABLE_NAME).update(payload).eq("id", source_id).execute()
        if not update_response.data:
            raise RuntimeError("No updated rows returned. Check Supabase permissions.")
        print(f"[SUCCESS] Updated DB row {source_id}")

    print(f"\nCurrent: {current_level:.6f} ft | Pred: {predicted_level:.6f} ft | Rise: {predicted_rise:.6f} ft | Status: {risk_label}\n")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Station A LSTM Predictor")
    parser.add_argument("--train", action="store_true", help="Train and save a new LSTM.")
    parser.add_argument("--no-write", action="store_true", help="Prevent writing to Supabase.")
    parser.add_argument("--limit", type=int, default=2000, help="Max rows to fetch.")
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    try:
        client = get_client()
        if args.train:
            train_model(client, row_limit=args.limit, epochs=args.epochs)
        else:
            # By default, runs prediction AND writes unless --no-write is passed
            run_prediction(client, row_limit=args.limit, write=not args.no-write)
    except Exception as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        sys.exit(1)

# Fixed: double underscores for __name__ and __main__
if __name__ == "__main__":
    main()