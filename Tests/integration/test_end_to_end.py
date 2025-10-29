import numpy as np
import pytest
from pathlib import Path
import src.sequence_creator as sc
import src.train_model as tr
import tensorflow as tf


@pytest.mark.integration
def test_full_pipeline_end_to_end(monkeypatch, tmp_path):
    """
    Integration test: ensures the full data → training → save/load → scoring pipeline runs.
    """

    # 1️⃣ Mock data fetch (bypass database)
    def mock_fetch_data_postgres():
        # Create fake OHLCV data
        import pandas as pd
        rows = 200
        return pd.DataFrame({
            "Open": np.random.rand(rows),
            "High": np.random.rand(rows),
            "Low": np.random.rand(rows),
            "Close": np.linspace(100, 200, rows),  # linear increasing trend
            "Volume": np.random.randint(1000, 5000, rows)
        })

    # Monkeypatch DB fetch
    monkeypatch.setattr(sc, "fetch_data_postgres", mock_fetch_data_postgres)
    
    def mock_project_root():
    # Force all saves to go under tmp_path/models
    # if this is not there then the below line of monkeypatch.setattr for referncing parent path will refer to a non-existent path which will throw assertion error that model should be saved after training
      return tmp_path

    # Monkeypatch model saving path so it writes to tmp_path instead of real /models
    # simulates: project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(tr, "Path", lambda *a, **kw: Path(tmp_path))

    # 2️⃣ Prepare data (sequence creation + scaling)
    X_train, X_test, y_train, y_test, scaler = sc.prepare_data()

    assert X_train.ndim == 3, "X_train should be 3D for LSTM"
    assert len(X_train) > 0 and len(X_test) > 0, "Data split should not be empty"

    # 3️⃣ Train lightweight model (using fewer epochs for speed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(5, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=0)

    # 4️⃣ Save model and run modelUpdater logic
    tr.modelUpdater(model, X_test, y_test)

    # 5️⃣ Ensure model was saved
    model_path = tmp_path / "models" / "gold_lstm_model.h5"
    assert model_path.exists(), "Model should be saved after training"

    # 6️⃣ Load and evaluate
    reloaded = tf.keras.models.load_model(model_path)
    preds = reloaded.predict(X_test)
    assert preds.shape[0] == X_test.shape[0], "Predictions shape mismatch"

    print("✅ Integration pipeline ran successfully from data to model scoring.")
