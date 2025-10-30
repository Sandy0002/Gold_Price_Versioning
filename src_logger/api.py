from src_logger.logging_config import setup_logger

# Initialize logger early
setup_logger()


from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import os
from pathlib import Path
import time
import sys
from src_logger.logger import get_logger  # ✅ import your centralized logger
from fastapi.responses import JSONResponse
import traceback


logger = get_logger(__name__)

logger.info("FastAPI app initialized")


# ===============================
# 1. Initialize FastAPI app
# ===============================

app = FastAPI(title="Gold Price Forecasting API")

# ===============================
# 2. Define input schema
# ===============================

class InputData(BaseModel):
    features: list[float]


# ===============================
# 3. Load model and metadata at startup
# ===============================

project_root = Path(__file__).resolve().parents[1]
models_dir = project_root / "models"
model_path = models_dir / "gold_lstm_model.keras"
metadata_path = models_dir / "model_metadata.json"
scaler = MinMaxScaler()

try:
    model = load_model(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info("✅ Model and metadata loaded successfully.")
except Exception as e:
    logger.exception(f"❌ Error loading model or metadata. Got this as :{e}")

    model = None
    metadata = {}

# ===============================
# 4. Middleware: Log each request
# ===============================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 3)
    logger.info(f"{request.method} {request.url.path} completed in {process_time}s with status {response.status_code}")
    return response


# ===============================
# 5. Root endpoint
# ===============================

@app.get("/")
def root():
    logger.debug("Root endpoint accessed.")
    return {"message": "Welcome to the Gold Price Forecasting API!"}


# ===============================
# 6. About model endpoint
# ===============================

@app.get("/about_model")
def about_model():
    """Return information about the currently deployed model."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    metadata_file = models_dir / "model_metadata.json"

    if not metadata_file.exists():
        logger.warning("Model metadata file not found.")
        raise HTTPException(status_code=404, detail="Model metadata not found")

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        logger.info("Model metadata retrieved successfully.")
    except Exception as e:
        logger.exception("Error reading metadata file.")
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")

    return {"status": "success", "model_info": metadata}


# ===============================
# 7. Prediction endpoint
# ===============================

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        logger.info("Received prediction request.")
        features = np.array(data.features).reshape(-1, 1)
        logger.debug(f"Input features length: {len(features)}")

        # Ensure input length matches model lookback
        if features.shape[0] != 60:
            logger.warning(f"Invalid input length: {features.shape[0]} (expected 60).")
            raise ValueError(f"Expected 60 time steps, got {features.shape[0]}")

        scaled_features = scaler.fit_transform(features)
        X_input = scaled_features.reshape(1, 60, 1)
        logger.debug(f"Input shape for model: {X_input.shape}")

        scaled_pred = model.predict(X_input)[0][0]
        predicted_price = float(scaler.inverse_transform([[scaled_pred]])[0][0])

        logger.info(f"Prediction generated successfully: {predicted_price:.2f}")
        return {
            "predicted_gold_price": predicted_price,
            "model_info": metadata.get("model_name", metadata),
        }

    except Exception as e:
        logger.exception("Error during prediction.")
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")
    