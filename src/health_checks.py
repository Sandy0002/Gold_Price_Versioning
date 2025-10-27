from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import date
from sqlalchemy import create_engine
import database as db
import time
import requests


project_root = Path(__file__).resolve().parents[1]
scaler = MinMaxScaler()
models_dir =project_root / "models"
model = models_dir /"gold_lstm_model.h5"


# ---------- DATABASE CONNECTION ----------
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )

# 1. Initialize FastAPI app
# ===============================
app = FastAPI(title="Health checks API", version="1.0")

# To check if home page is active
@app.get("/")
def root():
    return {"status": "alive", "version": "v1.0.0"}


# Check if model is accessible or not and is loading or not
@app.get("/health/model")
def model_health():
      # metadata = models_dir / "model_metadata.json"
      scaler = MinMaxScaler()
      try:
            model = load_model(model)
            # with open(metadata, "r") as f:
            #       metadata = json.load(f)
            return {"status": "ready", "details": "Model loaded successfully"}
      except Exception as e:
            return {"status": "error", "details": str(e)}, 500
            # model = None
            # metadata = {}


# Checking if yfinance is accessible or not to be able to fetch data
@app.get("/health/data_source")
def data_source_health():
    try:
      gold = yf.Ticker("GC=F")
      df = gold.history(start="2025-10-24", end=date.today(), interval="1d").reset_index()
      if df.empty:
            raise ValueError("Empty data returned")
      return {"status": "ready", "details": "yfinance responding"}
    except Exception as e:
        return {"status": "error", "details": str(e)}, 500

# Checking if database is accessible or not
@app.get("/health/db")
def db_health():
    try:
        engine = get_engine()
        start = time.time()
        result = db.execute("SELECT 1;")
        latency = time.time() - start
        return {"status": "ready", "latency": latency}
    except Exception as e:
        return {"status": "error", "details": str(e)}, 500


# Checking if model is able to predict or not
from fastapi import APIRouter
import numpy as np
import requests, json, time
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

router = APIRouter(prefix="/health")



@router.get("/health/predict")
def predict_health():
    try:
        start_time = time.time()

        # 1️⃣ Load input file
        input_folder = project_root / "test_inputs"
        input_file = input_folder / "inputs.json"
        with open(input_file, "r") as f:
            test_input = json.load(f)

        # Expecting structure like: { "features": [list of last 60 prices] }
        if "features" not in test_input:
            raise ValueError("Missing 'features' key in test_inputs/inputs.json")

        # 2️⃣ Prepare data for LSTM
        features = np.array(test_input["features"]).reshape(-1, 1)
        if features.shape[0] != 60:
            raise ValueError(f"Expected 60 lookback days, got {features.shape[0]}")

        scaled_features = scaler.fit_transform(features)
        X_input = scaled_features.reshape(1, 60, 1)

        # 3️⃣ Local model prediction (internal model sanity)
        scaled_pred = model.predict(X_input)[0][0]
        local_pred = scaler.inverse_transform([[scaled_pred]])[0][0]

        # 4️⃣ Test deployed /predict endpoint
        PREDICT_URL = "https://gold-price-monitoring.onrender.com/predict"
        response = requests.post(PREDICT_URL, json=test_input, timeout=20)

        if response.status_code != 200:
            raise ValueError(f"/predict endpoint failed: {response.status_code} - {response.text}")

        data = response.json()
        if "prediction" not in data or data["prediction"] is None:
            raise ValueError("Invalid prediction structure from endpoint")

        latency = round(time.time() - start_time, 3)

        return {
            "status": "ready",
            "details": {
                "latency_seconds": latency,
                "message": "Model and endpoint both responding correctly"
            }
        }

    except Exception as e:
        return {"status": "error", "details": str(e)}, 500
