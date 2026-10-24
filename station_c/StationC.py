"""
================================================================================
Station C - Optical Water Telemetry & Video Feed (Raspberry Pi 5)
================================================================================
"""
import threading
import time
import cv2 # type: ignore
from flask import Flask, Response, render_template_string # type: ignore
import numpy as np # type: ignore
from supabase import Client, create_client # type: ignore

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://jqhimswayyurxymltrxn.supabase.co"
SUPABASE_KEY = "sb_publishable_OeO0MsX7Aoa4bYlA8NhbZw_-JmxsF0S"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# Global camera handle & status dictionary
cap = None
camera_index = 0

current_data = {
    "color": "Unknown",
    "measurement": "0 cm",
    "status": "Idle",
    "water_tier": 0,
    "predicted_rise_ft": 0.0,
    "prediction_label": "SAFE",
}


def init_camera():
  """Scans camera indices 0, 1, and 2 to find an active USB optical device."""
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


# Initial camera setup
init_camera()


def get_color_data(h, s, v):
  """Maps HSV color space ranges to flood risk tiers and approximate height readings."""
  if s < 50 or v < 50:
    return "Unknown", "N/A", "Unknown", 0

  # Red (2-3ft / Critical)
  if (0 <= h <= 8) or (170 <= h <= 180):
    return "Red", "2-3 ft (20-30 cm)", "Critical Level", 3
  # Orange (1-2ft / Warning)
  elif 9 <= h <= 25:
    return "Orange", "1-2 ft (11-19 cm)", "Warning Level", 2
  # Green (0-1ft / Light)
  elif 36 <= h <= 85:
    return "Green", "0-1 ft (5-10 cm)", "Light Level", 1

  return "Unknown", "N/A", "Unknown", 0


# --- BACKGROUND THREAD: SUPABASE SYNC (TARGET: stationC) ---
def supabase_sync_loop():
  global current_data
  while True:
    time.sleep(5)  # Push telemetry every 5 seconds

    if current_data["color"] != "Unknown":
      try:
        data_to_insert = {
            "detected_color": current_data["color"],
            "water_tier": current_data["water_tier"],
            "distance_data": current_data["measurement"],
            "status": current_data["status"],
        }

        supabase.table("stationC").insert(data_to_insert).execute()
        print(
            f"[Supabase Sync -> stationC] Logged {current_data['color']} (Tier"
            f" {current_data['water_tier']})"
        )

      except Exception as e:
        print(f"[Supabase Sync Error]: {e}")


# Start background database worker thread
sync_thread = threading.Thread(target=supabase_sync_loop, daemon=True)
sync_thread.start()


def generate_frames():
  global cap, current_data
  while True:
    # Auto-reconnect camera if disconnected
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

    # Extract Center Region of Interest (ROI)
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv_roi)
    avg_h, avg_s, avg_v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    color_name, distance_data, status_desc, tier = get_color_data(
        avg_h, avg_s, avg_v
    )

    # Update thread state
    current_data["color"] = color_name
    current_data["measurement"] = distance_data
    current_data["status"] = status_desc
    current_data["water_tier"] = tier

    # Draw Target Box and Live Overlay
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"Station C Color: {color_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Level: {distance_data}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
      continue

    frame_bytes = buffer.tobytes()
    yield (
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        + frame_bytes
        + b"\r\n"
    )


@app.route("/")
def index():
  return render_template_string("""
        <html>
            <head><title>Station C - Camera Stream</title></head>
            <body style="font-family: sans-serif; text-align: center; background: #111; color: white;">
                <h2>Station C: Real-Time Optical Telemetry</h2>
                <img src="/video_feed" width="640" height="480" style="border: 3px solid #555; border-radius: 8px;">
            </body>
        </html>
    """)


@app.route("/video_feed")
def video_feed():
  return Response(
      generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, threaded=True)