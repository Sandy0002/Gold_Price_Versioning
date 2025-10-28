from fastapi.testclient import TestClient
from src.api import app
import numpy as np
from src.data_preprocess import fetch_data
from src.sequence_creator import create_sequences
from src.train_model import train_model
from src.forecaster import forecast_next
from tensorflow.keras.models import load_model



def test_end_to_end_training(tmp_path):
    # 1. Fetch and preprocess
    df = fetch_data("GC=F", "2024-01-01", "2024-01-20")
    data = df["Close"].values

    # 2. Create sequences
    X, y = create_sequences(data, lookback=60)
    X, y = np.array(X), np.array(y)

    # 3. Train model
    model_path = tmp_path / "model.h5"
    metadata_path = tmp_path / "metadata.json"
    metrics = train_model(X, y, model_path, metadata_path)

    assert "loss" in metrics

    # 4. Forecast
    model = load_model(model_path)
    pred = forecast_next(model, X[-1].reshape(1, 60, 1))
    assert isinstance(pred, float)


client = TestClient(app)

def test_about_model_endpoint():
    response = client.get("/about_model")
    assert response.status_code == 200
    assert "model_name" in response.json()

def test_predict_endpoint():
    data = {"features": [0.1]*60}
    response = client.post("/predict", json=data)
    assert response.status_code in [200, 400]  # depending on logic