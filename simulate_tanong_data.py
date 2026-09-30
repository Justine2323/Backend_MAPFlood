"""
================================================================================
SVM Flood Risk Model with Supabase Feedback Loop - Barangay Tañong, Malabon City
--------------------------------------------------------------------------------
Task: Auto-fetch Open-Meteo Tañong weather data if database is low, fetch 
      sensor data from Supabase, train Support Vector Classifier (SVC), 
      generate evaluation plots, and push predictions back to Supabase.
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from supabase import create_client, Client

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Direct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_flood_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "svm_scaler.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")
PLOT_CM_PATH = os.path.join(BASE_DIR, "plot_svm_confusion_matrix.png")
PLOT_PROB_PATH = os.path.join(BASE_DIR, "plot_svm_class_distribution.png")

# Supabase Credentials
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"

# Location: Barangay Tañong, Malabon City Coordinates
TANONG_LAT = 14.654
TANONG_LON = 120.9535

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


# ============================================================================
# 1. OPEN-METEO TAÑONG DATA SEEDER (IF SUPABASE HAS NO DATA)
# ============================================================================
def seed_tanong_data_from_open_meteo(supabase: Client):
    """Fetches past 30 days of Open-Meteo weather for Tañong, Malabon and populates Supabase."""
    print(f"\n[AUTO-SEED] Fetching historical weather for Barangay Tañong ({TANONG_LAT}, {TANONG_LON})...")
    
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

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
    current_tof = 2000.0  # Clear height: 2000mm (2 meters)

    for i in range(total_entries):
        temp = float(hourly["temperature_2m"][i] or 30.0)
        humidity = int(round(hourly["relative_humidity_2m"][i] or 70))
        rain_mm = float(hourly["rain"][i] or 0.0)
        weather_code = int(hourly["weather_code"][i] or 0)
        wind_speed = float(hourly["wind_speed_10m"][i] or 0.0)

        rain_pct = int(min(100, round(rain_mm * 10)))

        # Simulate water distance based on rainfall in Malabon
        if rain_mm > 5.0:
            current_tof -= random.uniform(30.0, 90.0)
        elif rain_mm > 0.5:
            current_tof -= random.uniform(5.0, 25.0)
        else:
            current_tof += random.uniform(5.0, 20.0)

        current_tof = max(5.0, min(current_tof, 2000.0))

        record = {
            "rain_percent": int(rain_pct),
            "wind_speed_kmh": float(round(wind_speed, 2)),
            "tof_distance_mm": float(round(current_tof, 2)),
            "om_temp": float(round(temp, 2)),
            "om_humidity": int(humidity),
            "om_weather_code": int(weather_code)
        }
        records.append(record)

    batch_size = 100
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            supabase.table("sensor_data").insert(batch).execute()
            inserted += len(batch)
        except Exception as e:
            print(f"Error inserting batch: {e}")

    print(f"✅ Successfully seeded {inserted} Tañong historical records into Supabase!\n")


# ============================================================================
# 2. SUPABASE DATA FETCHING & FEATURE ENGINEERING
# ============================================================================
def fetch_supabase_sensor_data(supabase: Client) -> pd.DataFrame:
    print("Connecting to Supabase database...")
    
    response = supabase.table("sensor_data").select("*").order("id", desc=False).range(0, 5000).execute()
    data = response.data

    # Auto-seed if data is insufficient
    if not data or len(data) < 10:
        print(f"Insufficient data in database ({len(data) if data else 0} rows).")
        seed_tanong_data_from_open_meteo(supabase)
        response = supabase.table("sensor_data").select("*").order("id", desc=False).range(0, 5000).execute()
        data = response.data

    df = pd.DataFrame(data)
    print(f"Successfully retrieved {len(df)} sensor entries from Supabase for Tañong, Malabon.")

    # Fill missing values
    df["tof_distance_mm"] = df["tof_distance_mm"].ffill().fillna(2000.0)
    df["rain_percent"] = df["rain_percent"].fillna(0)
    df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(0)
    df["om_temp"] = df["om_temp"].fillna(30.0)
    df["om_humidity"] = df["om_humidity"].fillna(70)
    df["om_weather_code"] = df["om_weather_code"].fillna(0)

    # Momentum features
    df["rain_3p_avg"] = df["rain_percent"].rolling(window=3, min_periods=1).mean().round(2)
    df["tof_delta"] = df["tof_distance_mm"].diff().fillna(0.0).round(2)

    # Multi-class Heuristic Thresholds (Adjusted for mm measurements)
    raw_risk = []
    for rain, tof in zip(df["rain_percent"], df["tof_distance_mm"]):
        if tof <= 500.0 or rain >= 80:
            raw_risk.append(2)  # Flood Alert (HIGH)
        elif tof <= 1200.0 or rain >= 40:
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


def prepare_dataset(df: pd.DataFrame, scaler: StandardScaler, fit_scaler: bool = False):
    """Extracts 2D feature matrices and target arrays suitable for SVM tabular learning."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    row_ids = df["id"].tolist()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, row_ids


# ============================================================================
# 3. MODEL ARCHITECTURE & GRAPHING FUNCTIONS
# ============================================================================
def build_svm(class_weight_dict: dict = None) -> SVC:
    """Instantiates a Support Vector Machine classifier with RBF Kernel and probability estimates."""
    model = SVC(
        C=1.5,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight=class_weight_dict if class_weight_dict else "balanced",
        random_state=RANDOM_STATE,
    )
    return model


def plot_class_distribution(y_train, y_test):
    """Plots and displays the training vs test set class distribution."""
    plt.figure(figsize=(9, 5))
    
    train_counts = [np.sum(y_train == i) for i in CLASS_INDICES]
    test_counts = [np.sum(y_test == i) for i in CLASS_INDICES]
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.bar(x - width/2, train_counts, width, label='Train Set', color='navy')
    plt.bar(x + width/2, test_counts, width, label='Test Set', color='teal')

    plt.title('SVM Dataset Class Distribution (Brgy. Tañong)')
    plt.xlabel('Risk Categories')
    plt.ylabel('Sample Count')
    plt.xticks(x, CLASS_NAMES)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_PROB_PATH, dpi=300)
    print(f"\n[INFO] Saved dataset distribution plot to {PLOT_PROB_PATH}")
    plt.show()
    plt.close()


def plot_confusion_matrix_chart(y_true, y_pred, labels):
    """Plots, displays, and saves a heatmap for the SVM confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_INDICES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels)
    plt.title('SVM Flood Risk Prediction Confusion Matrix (Brgy. Tañong)')
    plt.ylabel('Actual Risk Level')
    plt.xlabel('Predicted Risk Level')
    
    plt.tight_layout()
    plt.savefig(PLOT_CM_PATH, dpi=300)
    print(f"[INFO] Saved confusion matrix plot to {PLOT_CM_PATH}")
    plt.show()
    plt.close()


# ============================================================================
# 4. FEEDBACK TO SUPABASE
# ============================================================================
def push_predictions_to_supabase(supabase: Client, row_ids: list, y_pred: np.ndarray):
    print("\nUpdating Supabase database with SVM flood prediction feedback...")
    updated_count = 0
    for row_id, pred_code in zip(row_ids, y_pred):
        label = CLASS_NAMES[pred_code]
        try:
            supabase.table("sensor_data").update({
                "predicted_flood_risk": int(pred_code),
                "prediction_label": label
            }).eq("id", row_id).execute()
            updated_count += 1
        except Exception as e:
            print(f"Failed to update row ID {row_id}: {e}")

    print(f"Successfully updated {updated_count} rows in Supabase.")


# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================
def main():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    df = fetch_supabase_sensor_data(supabase)

    print("\nSupabase Class Breakdown:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = (df[TARGET_COLUMN] == cls_idx).sum()
        pct = (count / len(df)) * 100
        print(f"  - {cls_name}: {count} records ({pct:.1f}%)")

    scaler = StandardScaler()

    # Small dataset logic
    if len(df) < 50:
        print("\n[INFO] Small dataset detected (< 50 rows). Training on 100% of data without test split.")
        X_train, y_train, train_ids = prepare_dataset(df, scaler, fit_scaler=True)
        X_test, y_test, test_ids = X_train, y_train, train_ids
    else:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train, y_train, train_ids = prepare_dataset(train_df, scaler, fit_scaler=True)
        X_test, y_test, test_ids = prepare_dataset(test_df, scaler, fit_scaler=False)

    # Class Weights for SVM
    present_classes = np.unique(y_train)
    class_weight_dict = "balanced"
    if len(present_classes) > 1:
        weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
        class_weight_dict = {cls: w for cls, w in zip(present_classes, weights)}

    # Build and Train SVM Model
    model = build_svm(class_weight_dict)

    print("\n--- Training Support Vector Machine (SVM) Model ---")
    model.fit(X_train, y_train)
    print("SVM model fitting complete.")

    # Plot Class Distribution
    plot_class_distribution(y_train, y_test)

    # Evaluation & Confusion Matrix
    if len(X_test) > 0:
        y_pred = model.predict(X_test)

        print("\n" + "=" * 60)
        print("SVM MODEL EVALUATION METRICS")
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

    # Full Dataset Inference & Feedback Loop to Supabase
    X_all, _, all_row_ids = prepare_dataset(df, scaler, fit_scaler=False)
    if len(X_all) > 0:
        all_preds = model.predict(X_all)
        push_predictions_to_supabase(supabase, all_row_ids, all_preds)

    # Save Model Artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    schema = {"feature_columns": FEATURE_COLUMNS}
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)

    print(f"\n[SUCCESS] SVM model trained, evaluated, and updated to database.")


if __name__ == "__main__":
    main()