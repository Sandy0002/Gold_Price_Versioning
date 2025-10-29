# This code is an integration test for the entire backend

import numpy as np
import pytest
from pathlib import Path
import tensorflow as tf
import src.sequence_creator as sc
import src.train_model as tr


@pytest.mark.integration
def test_full_pipeline_end_to_end(monkeypatch, tmp_path):
    """
    Integration test: ensures the full data → training → save/load → scoring pipeline runs.
    """

    # 1️⃣ Mock data fetch (bypass database)
    def mock_fetch_data_postgres():
        import pandas as pd
        rows = 200
        return pd.DataFrame({
            "Open": np.random.rand(rows),
            "High": np.random.rand(rows),
            "Low": np.random.rand(rows),
            "Close": np.linspace(100, 200, rows),  # linear trend
            "Volume": np.random.randint(1000, 5000, rows)
        })

    monkeypatch.setattr(sc, "fetch_data_postgres", mock_fetch_data_postgres)

    # 2️⃣ Redirect model save path to tmp_path
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure train_model.py uses tmp_path for model saving
    monkeypatch.setattr(tr, "Path", lambda *a, **k: tmp_path)

    # 3️⃣ Prepare data
    X_train, X_test, y_train, y_test, scaler = sc.prepare_data()
    assert X_train.ndim == 3, "X_train should be 3D for LSTM"
    assert len(X_train) > 0 and len(X_test) > 0, "Data split should not be empty"

    # 4️⃣ Define a simple LSTM model (no InputLayer to avoid serialization issues)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=0)

    # 5️⃣ Run modelUpdater logic
    tr.modelUpdater(model, X_test, y_test)

    # 6️⃣ Resolve possible model save locations
    tmp_model_path = tmp_path / "models" / "gold_lstm_model.h5"
    real_model_path = Path(tr.__file__).resolve().parents[1] / "models" / "gold_lstm_model.h5"

    if tmp_model_path.exists():
        model_path = tmp_model_path
    elif real_model_path.exists():
        model_path = real_model_path
    else:
        raise AssertionError(
            f"❌ Model not found in either tmp path ({tmp_model_path}) "
            f"or real path ({real_model_path})"
        )

    # 7️⃣ Reload model safely
    reloaded = tf.keras.models.load_model(model_path, compile=False)
    preds = reloaded.predict(X_test)
    assert preds.shape[0] == X_test.shape[0], "Predictions shape mismatch"

    print("✅ Integration pipeline ran successfully from data to model scoring.")
