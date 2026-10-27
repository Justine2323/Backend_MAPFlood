"""
================================================================================
Station C - Integrated Real-Time Optical Telemetry & GPIO (Pi 5)
================================================================================
"""
import json
import os
import ssl
import threading
import time
import urllib.request
import cv2  # type: ignore
from flask import Flask, Response, render_template_string  # type: ignore
from supabase import Client, create_client  # type: ignore

# --- RASPBERRY PI 5 GPIO CONFIGURATION ---
SIREN_PIN = 17       # GPIO 17 (Pin 11) -> Relay / Siren Control
RAIN_GPIO_PIN = 27   # GPIO 27 (Pin 13) -> Digital Rain Sensor
WIND_GPIO_PIN = 22   # GPIO 22 (Pin 15) -> Wind Anemometer Pulse Pin

try:
    from gpiozero import DigitalInputDevice, Output  # type: ignore

    siren = Output(SIREN_PIN)
    siren.off()
    rain_sensor = DigitalInputDevice(RAIN_GPIO_PIN, pull_up=True)
    wind_sensor = DigitalInputDevice(WIND_GPIO_PIN, pull_up=True)
    GPIO_AVAILABLE = True
    print("[INFO] Raspberry Pi 5 GPIO Hardware initialized successfully.")
except Exception as e:
    GPIO_AVAILABLE = False
    print(f"[WARNING] GPIO Initialization skipped ({e}). Running in simulation mode.")

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# --- LOCATION & THRESHOLD CONSTANTS ---
TANONG_LAT = 14.6542
TANONG_LON = 120.9508
SIREN_TRIGGER_RAIN_PERCENT = 50
SIREN_TRIGGER_WIND_KMH = 35.0

# Global Hardware & State Memory
cap = None
camera_index = 0

current_data = {
    "color": "Unknown",
    "measurement": "0 ft",
    "status": "Idle",
    "water_tier": 0,
    "actual_water_level_ft": 0.0,
    "predicted_flood_risk": 0,
    "prediction_label": "SAFE (GREEN)",
    "rain_percent": 0,
    "wind_speed_kmh": 0.0,
}

# Cached Open-Meteo values (Updated every 10 minutes)
cached_weather = {
    "om_temp": 30.0,
    "om_humidity": 70,
    "rain_percent_api": 0.0,
    "om_weather_code": 0,
    "wind_speed_kmh_api": 0.0,
}


# ============================================================================
# 1. OPTICAL & GPIO SENSOR DRIVERS
# ============================================================================
def init_camera():
    """Scans for active USB camera."""
    global cap, camera_index
    for idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                print(f"[INFO] Connected to USB Camera at index {idx}")
                cap = temp_cap
                camera_index = idx
                return True
            temp_cap.release()
    print("[WARNING] No active USB camera device found.")
    return False

init_camera()


def get_color_data(h, s, v):
    """Maps HSV space to 3-tier flood levels based on requested constraints."""
    if s < 50 or v < 50:
        return "Unknown", "N/A", "Unknown", 0, -1.0

    # RED (2 to 3ft) - ALERT
    if (0 <= h <= 8) or (170 <= h <= 180):
        return "Red", "2 to 3ft", "ALERT", 3, 3.0
    # ORANGE (1ft to 2ft) - WARNING
    elif 9 <= h <= 25:
        return "Orange", "1ft to 2ft", "WARNING", 2, 2.0
    # GREEN (0 to 1ft) - SAFE
    elif 36 <= h <= 85:
        return "Green", "0 to 1ft", "SAFE", 1, 1.0

    return "Unknown", "N/A", "Unknown", 0, -1.0


def read_fast_rain_percent():
    if GPIO_AVAILABLE:
        return 100 if not rain_sensor.is_active else 0
    return 0


def read_fast_wind_kmh():
    if GPIO_AVAILABLE:
        pulse_count = 0
        start_time = time.time()
        last_state = wind_sensor.is_active

        while time.time() - start_time < 0.2:  # 200ms sampling
            current_state = wind_sensor.is_active
            if current_state and not last_state:
                pulse_count += 1
            last_state = current_state
            time.sleep(0.005)

        wind_kmh = (pulse_count / 0.2) * 2.4
        return round(wind_kmh, 2)
    return 0.0


def fetch_open_meteo_data():
    global cached_weather
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={TANONG_LAT}&longitude={TANONG_LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "StationC/1.0"})

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
                data = json.loads(response.read())
                current = data.get("current", {})
                cached_weather = {
                    "om_temp": current.get("temperature_2m", 30.0),
                    "om_humidity": current.get("relative_humidity_2m", 70),
                    "rain_percent_api": current.get("precipitation", 0.0) * 10.0,
                    "om_weather_code": current.get("weather_code", 0),
                    "wind_speed_kmh_api": current.get("wind_speed_10m", 0.0),
                }
                return
        except Exception:
            time.sleep(1)

def weather_fetch_loop():
    while True:
        fetch_open_meteo_data()
        time.sleep(600)

threading.Thread(target=weather_fetch_loop, daemon=True).start()


# ============================================================================
# 2. REAL-TIME TELEMETRY ENGINE & DATABASE SYNC
# ============================================================================
def real_time_telemetry_loop():
    """Runs continuous hardware sampling for threshold rules."""
    global current_data
    while True:
        start_time = time.time()

        rain_pct = read_fast_rain_percent()
        wind_kmh = read_fast_wind_kmh()

        current_data["rain_percent"] = rain_pct
        current_data["wind_speed_kmh"] = wind_kmh

        active_level = current_data["actual_water_level_ft"]

        is_rain_danger = rain_pct >= SIREN_TRIGGER_RAIN_PERCENT
        is_wind_danger = wind_kmh >= SIREN_TRIGGER_WIND_KMH

        # Hardware & Visual Threshold Logic
        if active_level >= 3.0 or is_rain_danger or is_wind_danger:
            risk_code = 3
            risk_label = "ALERT (RED)"
            if GPIO_AVAILABLE:
                siren.on()
        elif active_level >= 2.0:
            risk_code = 2
            risk_label = "WARNING (ORANGE)"
            if GPIO_AVAILABLE:
                siren.off()
        elif active_level >= 1.0:
            risk_code = 1
            risk_label = "SAFE (GREEN)"
            if GPIO_AVAILABLE:
                siren.off()
        else:
            risk_code = 0
            risk_label = "UNKNOWN"
            if GPIO_AVAILABLE:
                siren.off()

        current_data["predicted_flood_risk"] = risk_code
        current_data["prediction_label"] = risk_label

        elapsed = time.time() - start_time
        time.sleep(max(0.1, 1.0 - elapsed))

threading.Thread(target=real_time_telemetry_loop, daemon=True).start()


def supabase_sync_loop():
    """Sends raw telemetry to Supabase continuously."""
    while True:
        time.sleep(5)  # Sync every 5 seconds
        try:
            data_to_insert = {
                "detected_color": current_data["color"],
                "water_tier": current_data["water_tier"],
                "distance_data": current_data["measurement"],
                "status": current_data["status"],
                "predicted_flood_risk": current_data["predicted_flood_risk"],
                "prediction_label": current_data["prediction_label"],
                # LSTM predictions removed; raw data only
                "rain_percent": current_data["rain_percent"],
                "wind_speed_kmh": current_data["wind_speed_kmh"],
                "om_temp": cached_weather["om_temp"],
                "om_humidity": cached_weather["om_humidity"],
                "om_weather_code": cached_weather["om_weather_code"],
            }
            supabase.table("StationC").insert(data_to_insert).execute()
            print(f"[Supabase Push] Level: {current_data['measurement']} | Color: {current_data['color']} | Status: {current_data['prediction_label']}")
        except Exception as e:
            print(f"[Supabase Sync Error]: {e}")

threading.Thread(target=supabase_sync_loop, daemon=True).start()


# ============================================================================
# 3. FLASK STREAM & WEB DASHBOARD ENDPOINTS
# ============================================================================
def generate_frames():
    global cap, current_data
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(1)
            init_camera()
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        height, width, _ = frame.shape
        cx, cy = width // 2, height // 2
        box_size = 25
        x1, y1 = cx - box_size, cy - box_size
        x2, y2 = cx + box_size, cy + box_size

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_roi)
        avg_h, avg_s, avg_v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

        color_name, dist_str, status_desc, tier, level_ft = get_color_data(avg_h, avg_s, avg_v)

        if color_name != "Unknown":
            current_data["color"] = color_name
            current_data["measurement"] = dist_str
            current_data["status"] = status_desc
            current_data["water_tier"] = tier
            current_data["actual_water_level_ft"] = level_ft
        else:
            current_data["actual_water_level_ft"] = -1.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {current_data['color']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Level: {current_data['measurement']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {current_data['status']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template_string("""
        <html>
            <head>
                <title>Station C - Raspberry Pi 5 Telemetry Node</title>
                <style>
                    body { font-family: sans-serif; background: #111; color: white; text-align: center; margin: 20px; }
                    .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
                    .card { background: #222; padding: 20px; border-radius: 10px; border: 1px solid #444; width: 640px; }
                    .stat { font-size: 1.1em; margin: 10px 0; }
                    img { border-radius: 8px; max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h2>Station C: Raw Telemetry Node (Pushing to Supabase)</h2>
                <div class="container">
                    <div class="card">
                        <h3>Real-Time Camera Telemetry</h3>
                        <img src="/video_feed" width="640" height="480">
                    </div>
                </div>
            </body>
        </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # Runs continuously
    app.run(host="0.0.0.0", port=5000, threaded=True)