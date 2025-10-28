import numpy as np
from src.sequence_creator import create_sequences
from src.forecaster import forecast_next
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
from src.data_preprocess import fetch_data_postgres
from src.train_model import train_model
import pytest




@pytest.mark.unit
def test_fetch_data_returns_dataframe():
    df = fetch_data_postgres()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])

@pytest.mark.unit
def test_sequence_creation_shape():
    print("\n🧪 Running unit test: test_create_sequences()")
    data = np.arange(100)
    X, y = create_sequences(data, lookback=10)
    assert X.shape[1] == 10
    assert len(X) == len(y)
    
    print("✅ Sequence creation logic passed — shapes and outputs look valid.")

@pytest.mark.unit
def test_forecast_next_output_shape():
    # Dummy model
    model = Sequential([LSTM(10, input_shape=(60,1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")

    sample_input = np.random.rand(1, 60, 1)
    prediction = forecast_next(model, sample_input)
    assert isinstance(prediction, float) or isinstance(prediction, np.floating)

@pytest.mark.unit
def test_train_model_returns_artifacts(tmp_path):
    # Dummy data
    X = np.random.rand(50, 60, 1)
    y = np.random.rand(50)

    model_path = tmp_path / "model.h5"
    metadata_path = tmp_path / "metadata.json"

    metrics = train_model(X, y, model_path, metadata_path)
    assert "loss" in metrics
    assert model_path.exists()
    assert metadata_path.exists()