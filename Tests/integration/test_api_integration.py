# This code is integration test for frontend to  backend

import pytest
from fastapi.testclient import TestClient
from src.api import app  # <-- adjust import path if needed
from src import api
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense
import json
import json


BASE_URL = "https://gold-price-monitoring-1.onrender.com/"


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model(tmp_path_factory):
    """
    Create a dummy LSTM model and metadata in the expected location 
    so FastAPI app can load it successfully.
    """
    # Path to the models directory inside src
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Define and save a small dummy model
    model = Sequential([
        LSTM(10, input_shape=(60, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    dummy_model_path = models_dir / "gold_lstm_model.keras"
    model.save(dummy_model_path)

    # Create dummy metadata
    metadata = {
        "model_name": "dummy_gold_lstm",
        "trained_on": "synthetic_data",
        "lookback": 60
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)


def test_root_endpoint(client):
    """✅ Test that root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_about_model_endpoint(client):
    """✅ Test model metadata retrieval."""
    response = client.get("/about_model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "model_info" in data


def test_predict_endpoint_with_valid_input(client,monkeypatch,tmp_path):
    """✅ Test /predict endpoint with valid 60-length input and patched model path."""

    # -----------------------------
    # 1️⃣ Create a dummy model and metadata dynamically
    # -----------------------------
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)

    dummy_model_path = models_dir / "gold_lstm_model.keras"
    dummy_metadata_path = models_dir / "model_metadata.json"

    # Create and save a small LSTM model
    model = Sequential([
        LSTM(10, input_shape=(60, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.save(dummy_model_path)

    # Create dummy metadata file
    metadata = {"model_name": "dummy_gold_lstm", "trained_on": "test_data"}
    with open(dummy_metadata_path, "w") as f:
        json.dump(metadata, f)

    # -----------------------------
    # 2️⃣ Patch API model + metadata paths dynamically
    # -----------------------------
    monkeypatch.setattr(api, "load_model", lambda path: load_model(dummy_model_path))
    monkeypatch.setattr(api, "model", model)
    monkeypatch.setattr(api, "metadata", metadata)



    # Note: In above we have create a temp setup as if we don't create it, it will try to load the real model folder and if it doens't able to fetch it will give 500 error
    """✅ Test /predict endpoint with valid 60-length input."""
    # Generate 60 dummy float values
    payload = {"features": list(np.linspace(1800, 1900, 60))}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_gold_price" in data
    assert isinstance(data["predicted_gold_price"], float)
    print("✅ Prediction output:", data)


def test_predict_endpoint_invalid_input(client,monkeypatch,tmp_path):
    """⚠️ Test /predict endpoint with wrong input length."""
    

    # -----------------------------
    # 1️⃣ Create a dummy model and metadata dynamically
    # -----------------------------
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)

    dummy_model_path = models_dir / "gold_lstm_model.keras"
    dummy_metadata_path = models_dir / "model_metadata.json"

    # Create and save a small LSTM model
    model = Sequential([
        LSTM(10, input_shape=(60, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.save(dummy_model_path)

    # Create dummy metadata file
    metadata = {"model_name": "dummy_gold_lstm", "trained_on": "test_data"}
    with open(dummy_metadata_path, "w") as f:
        json.dump(metadata, f)

    # -----------------------------
    # 2️⃣ Patch API model + metadata paths dynamically
    # -----------------------------
    monkeypatch.setattr(api, "load_model", lambda path: load_model(dummy_model_path))
    monkeypatch.setattr(api, "model", model)
    monkeypatch.setattr(api, "metadata", metadata)


    payload = {"features": list(np.linspace(1800, 1850, 10))}  # only 10 values

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Error in prediction" in response.json()["detail"]