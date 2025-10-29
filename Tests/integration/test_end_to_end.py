# This code is integration test for entire backend

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

    # Monkeypatch model saving path so it writes to tmp_path instead of real /models
    # simulates: project_root = Path(__file__).resolve().parents[1]
    # monkeypatch.setattr(tr, "Path", lambda *a, **kw: Path(tmp_path))
    
    # 2️⃣ Redirect model save path to tmp_path
    import src.train_model as tr
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tr, "Path", lambda *a, **k: tmp_path)

    # 3️⃣  Prepare data (sequence creation + scaling)
    X_train, X_test, y_train, y_test, scaler = sc.prepare_data()
    assert X_train.ndim == 3, "X_train should be 3D for LSTM"
    assert len(X_train) > 0 and len(X_test) > 0, "Data split should not be empty"

  # 4️⃣ Define a simple LSTM model (no InputLayer)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dense(1)
    ]) 

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=0)

    #  5️⃣ Save model using your logic
    tr.modelUpdater(model, X_test, y_test)

    # 6️⃣ Check model exists
    model_path = tmp_path / "models" / "gold_lstm_model.h5"
    assert model_path.exists(), f"Model not found at {model_path}"

    # for above line if you get:
    '''Tests/integration/test_end_to_end.py::test_full_pipeline_end_to_end - AssertionError: Model not found at /tmp/models/gold_lstm_model.h5 assert False + where False = exists() + where exists = PosixPath('/tmp/models/gold_lstm_model.h5').exists
    then give below as project root to use the actual path as root:
    
    import src.train_model as tr 
    project_root = Path(tr.__file__).resolve().parents[1]
    '''

    # 7️⃣ Reload model safely (disable compile)
    reloaded = tf.keras.models.load_model(model_path, compile=False)
    preds = reloaded.predict(X_test)


    assert preds.shape[0] == X_test.shape[0], "Predictions shape mismatch"
    print("✅ Integration pipeline ran successfully from data to model scoring.")
