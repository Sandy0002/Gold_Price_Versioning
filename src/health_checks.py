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
from sqlalchemy import create_engine, inspect
from database import execute
import time


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
      project_root = Path(__file__).resolve().parents[1]
      models_dir =project_root / "models"
      model = models_dir /"gold_lstm_model.h5"
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
