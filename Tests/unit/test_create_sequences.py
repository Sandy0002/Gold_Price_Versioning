
import numpy as np
from src.sequence_creator import create_sequences,prepare_data
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler
import src.sequence_creator as sc


def mock_fetch_data_postgres():
    """Mocked database fetch that returns predictable DataFrame."""
    dates = pd.date_range("2023-01-01", periods=120)
    close_prices = np.linspace(100, 200, 120)
    df = pd.DataFrame({"Date": dates, "Close": close_prices})
    return df


# -------------------- TEST 1 --------------------
def test_create_sequences_shape_and_alignment():
    """Ensure create_sequences correctly forms aligned X, y arrays."""
    data = np.arange(100).reshape(-1, 1)
    lookback = 10
    X, y = create_sequences(data, lookback)

    # Expected shapes
    assert X.shape == (90, 10), f"Unexpected X shape {X.shape}"
    assert y.shape == (90,), f"Unexpected y shape {y.shape}"

    # Check first sequence values
    assert np.array_equal(X[0], np.arange(0, 10)), "First sequence incorrect"
    assert y[0] == 10, "First y target should be 10"
    print("✅ test_create_sequences_shape_and_alignment passed.")


# -------------------- TEST 2 --------------------
def test_prepare_data_output_shapes(monkeypatch):
    """Test prepare_data end-to-end using a mock fetch_data_postgres."""
    # Replace actual DB call with mock
    monkeypatch.setattr(sc, "fetch_data_postgres", mock_fetch_data_postgres)

    X_train, X_test, y_train, y_test, scaler = sc.prepare_data()

    # Basic shape checks
    assert X_train.ndim == 3 and X_train.shape[2] == 1, "X_train must be 3D for LSTM"
    assert X_test.ndim == 3 and X_test.shape[2] == 1, "X_test must be 3D for LSTM"
    assert isinstance(scaler, MinMaxScaler), "Returned scaler must be MinMaxScaler"

    total_sequences = len(X_train) + len(X_test)
    assert total_sequences + 60 == 120, "Mismatch in total sequences vs input size"

    # Ensure data splits around 70/30
    ratio = len(X_train) / (len(X_train) + len(X_test))
    assert 0.65 < ratio < 0.75, "Train/test split ratio incorrect"

    print("✅ test_prepare_data_output_shapes passed.")


# -------------------- TEST 3 --------------------
def test_prepare_data_handles_empty_dataframe(monkeypatch):
    """Ensure prepare_data handles empty DataFrame safely."""
    def mock_empty_fetch():
        return pd.DataFrame(columns=["Close"])

    monkeypatch.setattr(sc, "fetch_data_postgres", mock_empty_fetch)

    X_train, X_test, y_train, y_test, scaler = sc.prepare_data()
    assert X_train.size == 0 and X_test.size == 0, "Empty input should return empty arrays"
    assert y_train.size == 0 and y_test.size == 0, "Empty y arrays expected"
    assert scaler is None, "Scaler should be None when DataFrame empty"

    print("✅ test_prepare_data_handles_empty_dataframe passed.")