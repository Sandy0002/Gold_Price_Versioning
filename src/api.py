from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, APIRouter
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. Initialize FastAPI app
# ===============================
router = APIRouter(prefix="/gold", tags=["Gold Price Forecasting"])
# app = FastAPI()  # <-- Define app here
# app.include_router(router)

'''
1. class InputData(BaseModel):
You’re defining a data model (or schema) named InputData. It inherits from BaseModel, which comes from Pydantic — the library FastAPI uses for automatic validation and serialization of request data.

This means whenever a user sends JSON data to your API, FastAPI will automatically:
      Check that the data matches this model structure,
      Convert it to Python types, and
      Raise a clear validation error if it doesn’t match.

So, this defines what the input JSON structure should look like.
'''
# ===============================
# 2. Define input schema
# ===============================
class InputData(BaseModel):
    features: list[float] # This declares a field named features inside your input model. It specifies that features must be a list of floating-point numbers.


# Load model at startup
project_root = Path(__file__).resolve().parents[1]
models_dir =project_root / "models"
model = models_dir /"gold_lstm_model.h5"
metadata = models_dir / "model_metadata.json"
scaler = MinMaxScaler()

try:
    model = load_model(model)
    with open(metadata, "r") as f:
        metadata = json.load(f)
    print("✅ Model and metadata loaded successfully.")
except Exception as e:
    print("❌ Error loading model or metadata:", e)
    model = None
    metadata = {}

# 4. Root endpoint
# ===============================
@router.get("/")
def root():
    return {"message": "Welcome to the Gold Price Forecasting API!"}

@router.get("/about_model")
def about_model():
    """Return information about the currently deployed model."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    metadata_file = models_dir / "model_metadata.json"

    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found")

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")

    return {
        "status": "success",
        "model_info": metadata
    }

# 5. Prediction endpoint
# ===============================
@router.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
      features = np.array(data.features).reshape(-1, 1)
      # Ensure input length matches model lookback
      if features.shape[0]!=60:
            raise ValueError(f"Expected 60 time steps, got {features.shape[0]}")

      scaled_features = scaler.fit_transform(features)
      X_input = scaled_features.reshape(1, 60, 1)
      print("🔹 Input shape:", features.shape)
      scaled_pred = model.predict(X_input)[0][0]
      predicted_price =  float(scaler.inverse_transform([[scaled_pred]])[0][0])
      print("🔹 Raw prediction output:", predicted_price)

    except Exception as e:
            import traceback
            print("❌ Error during prediction:")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

    return {
        "predicted_gold_price": predicted_price,
        "model_info": metadata.get("model_name", metadata),
    }
# 6. Run manually (optional)
# ===============================
# if __name__ == "__main__":
#     import uvicorn
#     from fastapi import FastAPI
#     app = FastAPI()
#     app.include_router(router)
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)