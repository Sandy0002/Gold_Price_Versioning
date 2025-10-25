import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import os
from pathlib import Path

# --- Configuration of file paths---
project_root = Path(__file__).resolve().parents[1]
# Create models directory under project root
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)
data_file_path = data_dir / "gold_data.xlsx"

# --- Fetch full history (if needed) ---
ticker = "GC=F"
gold = yf.Ticker(ticker)
df = gold.history(start="2020-01-01", interval="1d")

# --- Fetch today's rate ---
today = date.today()
tomorrow = today + timedelta(days=1)
latest_price = gold.history(start=today, end=tomorrow)

# --- Remove timezone info (important for Excel) ---
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

latest_price = latest_price.reset_index()
latest_price["Date"] = pd.to_datetime(latest_price["Date"]).dt.tz_localize(None)

# --- Save or update Excel file ---
if not os.path.exists(data_file_path):
    df.to_excel(data_file_path, index=False)
    print("✅ File created successfully.")
else:
    existing_df = pd.read_excel(data_file_path)
    updated_df = pd.concat([existing_df, latest_price], ignore_index=True)
    updated_df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    updated_df.to_excel(data_file_path, index=False)
    print("✅ File updated successfully.")