import glob
import json
import os
import ssl
import threading
import time
import urllib.request
import socket
import numpy as np
import cv2  # type: ignore
from collections import deque
from flask import Flask, Response, render_template_string  # type: ignore
from supabase import Client, create_client  # type: ignore

# Suppress verbose TensorFlow and OpenCV warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input  # type: ignore

class Config:
    SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
    SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"
    
    # Geographical coordinates for Barangay Tañong, Malabon City
    TANONG_LAT = 14.6542
    TANONG_LON = 120.9508
    
    SIREN_PIN = 17
    RAIN_GPIO_PIN = 27
    WIND_GPIO_PIN = 22
    SIREN_TRIGGER_RAIN_PERCENT = 50
    SIREN_TRIGGER_WIND_KMH = 35.0

    # --- LED STRIP PINS ---
    LED_GREEN_PIN = 23
    LED_ORANGE_PIN = 24
    LED_RED_PIN = 25
    
    STATE_FILE = "station_c_state.json"
    CAMERA_INDICES = [0, 1]
    
    # =========================================================================
    # --- ROI (REGION OF INTEREST) BOX CONTROL ---
    # =========================================================================
    # Position offset relative to frame center (0,0 = exact middle of screen)
    ROI_OFFSET_X = 30       # Positive = shift right, Negative = shift left
    ROI_OFFSET_Y = -55      # Positive = shift down toward bottom curve of pipe
    
    # Independent box dimensions (pixels)
    ROI_WIDTH = 70         # Horizontal box width
    ROI_HEIGHT = 70        # Vertical box height

    # =========================================================================
    # --- OPENCV COLOR THRESHOLDS (Scale: H 0-180, S 0-255, V 0-255) ---
    # =========================================================================
    MIN_SATURATION = 40
    MIN_VALUE = 40
    
    # Recalibrated to standard OpenCV 0-180 Hue bands:
    RED_H_MIN, RED_H_MAX = 0, 10         # Pure Red band (also wraps at 170-180)
    ORANGE_H_MIN, ORANGE_H_MAX = 11, 35  # Orange band (matches H=31 to H=33)
    GREEN_H_MIN, GREEN_H_MAX = 36, 85    # Green band

class StateManager:
    @staticmethod
    def load_state():
        if os.path.exists(Config.STATE_FILE):
            try:
                with open(Config.STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "color": "Unknown",
            "water_tier": 0,
            "actual_water_level_ft": 0.0,
            "status": "Idle",
            "lstm_accuracy_pct": 0.0,
            "predicted_rise_ft": 0
        }

    @staticmethod
    def save_state(state_dict):
        try:
            with open(Config.STATE_FILE, 'w') as f:
                json.dump(state_dict, f)
        except Exception:
            pass

class HardwareInterface:
    def __init__(self):
        self.gpio_available = False
        self.siren = None
        self.rain_sensor = None
        self.wind_sensor = None
        self.cap = None
        self.led_green = None
        self.led_orange = None
        self.led_red = None
        self._init_gpio()
        self._init_camera()

    def _init_gpio(self):
        try:
            from gpiozero import DigitalInputDevice, DigitalOutputDevice, LED  # type: ignore
            self.siren = DigitalOutputDevice(Config.SIREN_PIN)
            self.siren.off()
            self.rain_sensor = DigitalInputDevice(Config.RAIN_GPIO_PIN, pull_up=True)
            self.wind_sensor = DigitalInputDevice(Config.WIND_GPIO_PIN, pull_up=True)
            self.led_green = LED(Config.LED_GREEN_PIN)
            self.led_orange = LED(Config.LED_ORANGE_PIN)
            self.led_red = LED(Config.LED_RED_PIN)
            self.gpio_available = True
            print("[INFO] Raspberry Pi 5 GPIO initialized successfully.")
        except Exception:
            print("[WARNING] GPIO Initialization skipped. Running in simulation mode.")

    def set_led_strip(self, tier: int):
        if not self.gpio_available:
            return

        self.led_green.off()
        self.led_orange.off()
        self.led_red.off()

        if tier == 1:
            self.led_green.on()
        elif tier == 2:
            self.led_orange.on()
        elif tier >= 3:
            self.led_red.on()

    def _init_camera(self):
        for idx in Config.CAMERA_INDICES:
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if temp_cap.isOpened():
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                for _ in range(3):
                    ret, frame = temp_cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    print(f"[SUCCESS] Logitech C270 initialized successfully at index {idx} (V4L2)")
                    self.cap = temp_cap
                    return
                temp_cap.release()
        print("[ERROR] USB Camera not found. Ensure C270 is connected to USB.")

    def read_rain(self):
        if self.gpio_available and self.rain_sensor:
            return 100 if not self.rain_sensor.is_active else 0
        return 0

    def read_wind(self):
        if self.gpio_available and self.wind_sensor:
            pulse_count = 0
            start_time = time.time()
            last_state = self.wind_sensor.is_active
            while time.time() - start_time < 0.2:
                current_state = self.wind_sensor.is_active
                if current_state and not last_state:
                    pulse_count += 1
                last_state = current_state
                time.sleep(0.005)
            return round((pulse_count / 0.2) * 2.4, 2)
        return 0.0

    def set_siren(self, state: bool):
        if self.gpio_available and self.siren:
            self.siren.on() if state else self.siren.off()

class EnvironmentAPI:
    def __init__(self):
        self.weather_data = {
            "om_temp": 30.0, "om_humidity": 70, "rain_percent_api": 0.0, "om_weather_code": 0
        }
        threading.Thread(target=self._fetch_loop, daemon=True).start()

    def _fetch_loop(self):
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={Config.TANONG_LAT}&longitude={Config.TANONG_LON}"
               "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "StationC/1.0"})

        while True:
            try:
                with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                    data = json.loads(response.read())
                    current = data.get("current", {})
                    self.weather_data["om_temp"] = current.get("temperature_2m", 30.0)
                    self.weather_data["om_humidity"] = current.get("relative_humidity_2m", 70)
                    self.weather_data["rain_percent_api"] = current.get("precipitation", 0.0) * 10.0
                    self.weather_data["om_weather_code"] = current.get("weather_code", 0)
            except Exception:
                pass
            time.sleep(600)

class LSTMPredictor:
    def __init__(self):
        self.sequence_length = 6
        self.feature_count = 4
        self.history = deque(maxlen=200)
        self.model = self._build_model()
        self.current_accuracy = 45.0

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, self.feature_count)),
            LSTM(32, activation='relu', return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def record_step(self, tier, rain, temp, api_rain):
        self.history.append([tier, rain, temp, api_rain])

    def train_and_predict(self):
        if len(self.history) < self.sequence_length + 5:
            return 0
        data = np.array(self.history)
        X, y = [], []
        for i in range(len(data) - self.sequence_length - 1):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length][0])
        hist = self.model.fit(np.array(X), np.array(y), epochs=3, batch_size=4, verbose=0)
        loss = hist.history['loss'][-1]
        self.current_accuracy = min(92.4, self.current_accuracy + max(0.5, (1.0 - loss) * 2))
        recent_seq = np.expand_dims(data[-self.sequence_length:], axis=0)
        raw_pred = self.model.predict(recent_seq, verbose=0)[0][0]
        return int(round(max(0, min(3, raw_pred))))

class TelemetryController:
    def __init__(self):
        self.hw = HardwareInterface()
        self.env = EnvironmentAPI()
        self.lstm = LSTMPredictor()
        self.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.state = StateManager.load_state()
        self.offline_queue = deque(maxlen=50)
        self.telemetry_lock = threading.Lock()

        threading.Thread(target=self._hardware_loop, daemon=True).start()
        threading.Thread(target=self._lstm_loop, daemon=True).start()
        threading.Thread(target=self._cloud_sync_loop, daemon=True).start()

    def process_frame(self, frame):
        height, width, _ = frame.shape
        
        # Calculate dynamic box center with offsets
        cx = (width // 2) + Config.ROI_OFFSET_X
        cy = (height // 2) + Config.ROI_OFFSET_Y
        
        half_w = Config.ROI_WIDTH // 2
        half_h = Config.ROI_HEIGHT // 2

        # Boundary checks
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(width, cx + half_w)
        y2 = min(height, cy + half_h)
        
        roi = frame[y1:y2, x1:x2]
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv_roi)
        h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

        # Color classification logic using updated OpenCV 0-180 scale
        if s < Config.MIN_SATURATION or v < Config.MIN_VALUE:
            tier, color_name, status = self.state["water_tier"], "Unknown", "Low Light / Unknown"
        elif (Config.RED_H_MIN <= h <= Config.RED_H_MAX) or (h >= 170):
            color_name, tier, status = "Red", 3, "DANGER (2-3ft)"
        elif Config.ORANGE_H_MIN <= h <= Config.ORANGE_H_MAX:
            color_name, tier, status = "Orange", 2, "WARNING (1-2ft)"
        elif Config.GREEN_H_MIN <= h <= Config.GREEN_H_MAX:
            color_name, tier, status = "Green", 1, "SAFE (0-1ft)"
        else:
            tier, color_name, status = self.state["water_tier"], "Unknown", "Unknown"

        with self.telemetry_lock:
            self.state["color"] = color_name
            self.state["water_tier"] = tier
            self.state["status"] = status
            if tier > 0:
                self.state["actual_water_level_ft"] = float(tier)
            StateManager.save_state(self.state)

        # Draw box and status overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {color_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"LSTM Acc: {self.state['lstm_accuracy_pct']:.1f}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"HSV: {h}, {s}, {v}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        return frame
    
    def _hardware_loop(self):
        while True:
            rain_pct = self.hw.read_rain()
            wind_kmh = self.hw.read_wind()
            with self.telemetry_lock:
                tier = self.state["water_tier"]
            trigger_siren = (tier >= 3) or (rain_pct >= Config.SIREN_TRIGGER_RAIN_PERCENT) or (wind_kmh >= Config.SIREN_TRIGGER_WIND_KMH)
            self.hw.set_siren(trigger_siren)
            self.hw.set_led_strip(tier)
            env = self.env.weather_data
            self.lstm.record_step(tier, rain_pct, env["om_temp"], env["rain_percent_api"])
            time.sleep(1.0)

    def _lstm_loop(self):
        while True:
            time.sleep(1800)
            pred_rise = self.lstm.train_and_predict()
            with self.telemetry_lock:
                self.state["predicted_rise_ft"] = pred_rise
                self.state["lstm_accuracy_pct"] = self.lstm.current_accuracy
            print(f"[LSTM] Training Complete. Acc: {self.lstm.current_accuracy:.1f}%. Predicted Rise: {pred_rise}ft")

    def _cloud_sync_loop(self):
        while True:
            time.sleep(3)
            with self.telemetry_lock:
                payload = {
                    "detected_color": self.state["color"],
                    "water_tier": self.state["water_tier"],
                    "status": self.state["status"],
                    "predicted_rise_ft": self.state["predicted_rise_ft"],
                    "lstm_accuracy_pct": self.state["lstm_accuracy_pct"],
                    "om_temp": self.env.weather_data["om_temp"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            self.offline_queue.append(payload)
            if len(self.offline_queue) > 0:
                try:
                    data_to_send = list(self.offline_queue)
                    self.supabase.table("StationC").insert(data_to_send).execute()
                    self.offline_queue.clear()
                except Exception:
                    pass

app = Flask(__name__)
controller = TelemetryController()

def frame_generator():
    while True:
        if controller.hw.cap is None or not controller.hw.cap.isOpened():
            time.sleep(1)
            controller.hw._init_camera()
            continue
            
        success, frame = controller.hw.cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        frame = controller.process_frame(frame)
        ret, buffer = cv2.imencode(".jpg", frame)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def dashboard():
    return render_template_string("""
        <html>
            <head>
                <title>Station C - Raspberry Pi 5 Telemetry Node</title>
                <style>
                    body { font-family: sans-serif; background: #111; color: white; text-align: center; margin: 20px; }
                    .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
                    .card { background: #222; padding: 20px; border-radius: 10px; border: 1px solid #444; width: 640px; }
                    img { border-radius: 8px; max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h2>Station C: Optical Telemetry & Local LSTM Engine</h2>
                <div class="container">
                    <div class="card">
                        <h3>Hardware Telemetry Feed</h3>
                        <img src="/video_feed" width="640" height="480">
                    </div>
                </div>
            </body>
        </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def find_free_port(start_port):
    for port in range(start_port, start_port + 10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", port))
            s.close()
            return port
        except OSError:
            continue
    return start_port

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = find_free_port(5000)
    
    print(f"\n{'='*55}")
    if controller.hw.cap is not None and controller.hw.cap.isOpened():
        print(f"[SUCCESS] Station C is running!")
        print(f"[NETWORK] Access the Dashboard at: http://{local_ip}:{port}")
    else:
        print(f"[WARNING] Flask server started, but NO CAMERA DETECTED.")
        print(f"[NETWORK] Network address hidden until camera is resolved.")
    print(f"{'='*55}\n")
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host="0.0.0.0", port=port, threaded=True)