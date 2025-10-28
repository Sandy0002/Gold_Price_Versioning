import numpy as np
from tensorflow.keras.models import load_model
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import date
from sqlalchemy import create_engine
import database as db
import time
import requests
import json


health_checklist={}

project_root = Path(__file__).resolve().parents[1]
scaler = MinMaxScaler()
models_dir =project_root / "models"
model_path = models_dir /"gold_lstm_model.h5"
model = load_model(model_path)
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

    for identifier,ep in endpoints.items():
        response = check_endpoint(ep)
        health_checklist[identifier] = response
        print("--------------------------------------------")

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
    project_root = Path(__file__).resolve().parents[1]
    scaler = MinMaxScaler()

    if model is None:
        # return {"status": "❌ Error", "details": str(e)},500
        print(f'status": "❌ Failed to load model')

    else:  
        # return {"status": " ✅ready", "details": "model loaded succsesfully"}
        print(f'status": "✅ready", "details": "model loaded succsesfully')
  
        


# def predict_health():
#     try:
#         start_time = time.time()

#         # 1️⃣ Load input file
#         input_folder = project_root / "test_inputs"
#         input_file = input_folder / "inputs.json"
#         with open(input_file, "r") as f:
#             test_input = json.load(f)

#         # Expecting structure like: { "features": [list of last 60 prices] }
#         if "features" not in test_input:
#             raise ValueError("Missing 'features' key in test_inputs/inputs.json")

#         # 2️⃣ Prepare data for LSTM
#         features = np.array(test_input["features"]).reshape(-1, 1)
#         if features.shape[0] != 60:
#             raise ValueError(f"Expected 60 lookback days, got {features.shape[0]}")

#         scaled_features = scaler.fit_transform(features)
#         X_input = scaled_features.reshape(1, 60, 1)
#         model = load_model(model_path)
        
#         # 3️⃣ Local model prediction (internal model sanity)
#         scaled_pred = model.predict(X_input)[0][0]
#         local_pred = scaler.inverse_transform([[scaled_pred]])[0][0]

#         # 4️⃣ Test deployed /predict endpoint
#         PREDICT_URL = "https://gold-price-monitoring.onrender.com/predict"
#         response = requests.post(PREDICT_URL, json=test_input, timeout=20)

#         if response.status_code != 200:
#             raise ValueError(f"/predict endpoint failed: {response.status_code} - {response.text}")

#         data = response.json()
#         if "prediction" not in data or data["prediction"] is None:
#             raise ValueError("Invalid prediction structure from endpoint")

#         latency = round(time.time() - start_time, 3)

#         return {
#             "status": "✅ ready",
#             "details": {
#                 "latency_seconds": latency   }
#         }

#     except Exception as e:
#         return {"status": "❌ Error", "details": str(e)}, 500

if __name__=='__main__':
    # frontend_checker()
    # model_pred_test()
    # db_health()
    # data_source_health()
    model_health()
