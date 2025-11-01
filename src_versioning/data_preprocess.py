import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv
import argparse
import datetime
from pathlib import Path

load_dotenv()


# ---------- DATABASE CONNECTION ----------
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )

# ---------- STEP 1: FETCH GOLD DATA ----------
def fetch_gold_data(table_name="gold_prices", ticker="GC=F"):
    """
    Fetch historical gold price data from Yahoo Finance.
    Updates data daily and avoids duplicates.
    """
    engine = get_engine()
    gold = yf.Ticker(ticker)

    inspector = inspect(engine)
    table_exists = table_name in inspector.get_table_names()

    # Determine start date for fetching
    if table_exists:
        try:
            last_date_query = pd.read_sql(f'SELECT MAX("Date") AS last_date FROM {table_name}', engine)
            last_date = last_date_query["last_date"][0]

            # if some entry is there in table and database
            if last_date is not None:
                start_date = pd.to_datetime(last_date).date() + timedelta(days=1)
            
            # if no entyr is there in table possibly no table
            else:
                start_date = date.today() - timedelta(days=5)
        
        # If anything goes wrong
        except Exception:
            start_date = date.today() - timedelta(days=5)
    else:
        start_date = "2020-01-01"

    end_date = date.today()

    df = gold.history(start=start_date, end=end_date, interval="1d").reset_index()
    if df.empty:
        print("ℹ️ No new data to fetch.")
        return pd.DataFrame()  # empty dataframe

    # Normalize Date column
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Filter out rows already in DB
    if table_exists:
        existing_dates = pd.read_sql(f'SELECT "Date" FROM {table_name}', engine)
        existing_dates['Date'] = pd.to_datetime(existing_dates['Date']).dt.date
        df = df[~df['Date'].isin(existing_dates['Date'])]

    print(f"✅ {len(df)} new rows ready to insert.")
    return df


# ---------- STEP 2: STORE DATA TO POSTGRES ----------
def store_data_postgres(df, table_name="gold_prices"):
    """
    Store fetched data into PostgreSQL.
    Creates table if not exists, appends new rows otherwise.
    """
    if df.empty:
        print(f"ℹ️ No new rows to append.")
        return

    engine = get_engine()
    inspector = inspect(engine)
    table_exists = table_name in inspector.get_table_names()

    if not table_exists:
        # Create table and insert data
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"✅ Table '{table_name}' created and {len(df)} rows inserted.")
    else:
        # Append new rows
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"✅ Appended {len(df)} new rows to '{table_name}'.")


# ---------- STEP 3: FETCH DATA BACK ----------
def fetch_data_postgres(table_name="gold_prices",engine=None):
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(f"✅ Retrieved {len(df)} rows from '{table_name}'.")
    
    df.to_csv()
    return df

# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    today = date.today()
    data_dir = project_root / "data/raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file_path = data_dir / f"gold_snapshot_{today}.csv"

    print(f"🧭 Project root: {project_root}")
    print(f"📂 Saving to: {data_file_path.resolve()}")
    print(f"File exists before saving? {data_file_path.exists()}")

    new_data = fetch_gold_data()
    if len(new_data) > 0:
        store_data_postgres(new_data)

    df_retrieved = fetch_data_postgres()
    print(f"📊 Retrieved {len(df_retrieved)} rows")

    df_retrieved.to_csv(data_file_path, index=False)
    print(f"✅ File created successfully with name {data_file_path}")


    print(df_retrieved.tail())
