import numpy as np
import pandas as pd
from src.data_preprocess import fetch_data_postgres
import pytest
from sklearn.preprocessing import MinMaxScaler


def test_fetch_data_returns_dataframe():
    df = fetch_data_postgres()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])