import numpy as np
from keras.models import load_model
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import date
from sqlalchemy import create_engine
import src.database as db
import time
import requests
import json


project_root = Path(__file__).resolve().parents[1]
BASE_URL = "https://gold-price-monitoring-1.onrender.com/"

# Frontend to backend health checks

## Part 1: Frontend checks
def check_endpoint(endpoint: str):
    """Send a GET request to the endpoint and print response details."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return "✅ {endpoint} OK ->"
        else:
            return f"❌ {endpoint} FAILED -> Status: {response.status_code}, Reason: {response.reason}"
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error accessing {endpoint}: {e}")

def frontend_checker():
    print("🔍 Checking Gold Price Forecasting API health...\n")

    endpoints = {
     "root":   "/",            # root endpoint
      "About model":  "/about_model"  # model info endpoint
    }

    front_end_checklist = {}
    for identifier,ep in endpoints.items():
        response = check_endpoint(ep)
        front_end_checklist[identifier] = response

    return front_end_checklist

# Testing if our model is able to give predictions on live
def model_pred_test():
    features = np.random.rand(60).tolist()  # 60 random float values

    # Prepare JSON payload
    payload = {
        "features": features
    }

    # Send POST request
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    # Check and print response
    if response.status_code == 200:
        return f"✅ Success! Able to give predictions"
    
    else:
        return f"❌ Failed with status: {response.status_code}, Detail:{response.reason}"


# ---------- DATABASE CONNECTION ----------
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )

# Checking if database is accessible or not
def db_health():
    try:
        engine = get_engine()
        start = time.time()
        result = db.execute("SELECT 1;")
        latency = time.time() - start
        return {"status": "✅ ready", "latency": latency}
    except Exception as e:
        return {"status": "❌ error", "details": str(e)}, 500


# Checking if yfinance is accessible or not to be able to fetch data
def data_source_health():
    try:
      gold = yf.Ticker("GC=F")
      df = gold.history(start="2025-10-24", end=date.today(), interval="1d").reset_index()
      if df.empty:
            raise ValueError("Empty data returned")
      return {"status": "✅ ready", "details": "yfinance responding"}
    except Exception as e:
        return {"status": "❌ error", "details": str(e)}, 500


# Check if model is accessible or not and is loading or not
def model_health():
    scaler = MinMaxScaler()
    models_dir =project_root / "models"
    model_path = models_dir /"gold_lstm_model.keras"
    
    try:
        model = load_model(model_path)
        return {"status": " ✅ready", "details": "model loaded succsesfully"}
    except Exception as e:
        return {"status": "❌ Error while loading model", "details": str(e)},500
        

if __name__=='__main__':
    health_checks_checklist = {
        "GET request":frontend_checker,
        "POST request":model_pred_test,
        "Database connectivity":db_health,
        "Data source": data_source_health,
        "Model loading": model_health
    }
    
    # to store the status of each health check
    results = {}
    for name, func in health_checks_checklist.items():
        response = func()
        if isinstance(response, dict):
            for sub_key, sub_value in response.items():
                # Combine parent name and sub-check name for clarity
                identifier = f"{name}.{sub_key}"
                results[identifier] = sub_value
        else:
            results[name] = response
        
    for name, status in results.items():
        print(f'{name} : {status}')
        print("--------------------------------------------")
