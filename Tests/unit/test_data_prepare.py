import pandas as pd
from src.data_preprocess import fetch_data_postgres

def test_fetch_data_returns_dataframe():
    """Check if fetch_data_postgres returns a non-empty DataFrame with expected columns."""
    print("\n🧪 Running test_fetch_data_returns_dataframe...")
    df = fetch_data_postgres()
    assert isinstance(df, pd.DataFrame), "Returned object must be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]), \
        "Missing expected columns"
    print("✅ fetch_data_postgres() passed!")

