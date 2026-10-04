import os
import time
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def run_realtime_pipeline():
    while True:
        try:
            print("[INFO] Fetching latest ESP32 telemetry & generating predictions...")
            # Place your existing data fetching, scaling, and prediction logic here
            
        except Exception as e:
            print(f"[ERROR] Real-time loop exception: {e}")
        
        # Wait 30 seconds before running the next prediction cycle
        time.sleep(30)

if __name__ == "__main__":
    run_realtime_pipeline()