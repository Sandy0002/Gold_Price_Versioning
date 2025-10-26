import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from datetime import date, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv


# Loading .env file to access the contents in the folder
load_dotenv()

# Getting database credentials
username = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
host = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
table_name="gold_prices"

# ---------- STEP 1: FETCH DATA FROM YFINANCE ----------
def fetch_gold_data(ticker="GC=F", period="5y", interval="1d"):
    """
    Fetch historical gold price data from Yahoo Finance.
    Default ticker 'GC=F' is Gold Futures.
    """

    engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")
    gold = yf.Ticker("GC=F")

    # Get the last date in the table (if table exists)
    try:
        last_date = pd.read_sql(f"SELECT MAX(Date) AS last_date FROM {table_name}", engine)["last_date"][0]
        if last_date is not None:
            start_date = pd.to_datetime(last_date).date() + timedelta(days=1)
        else:
            start_date = date.today() - timedelta(days=5)
    except Exception:
        start_date = date.today() - timedelta(days=5)

    end_date = date.today()

    df = gold.history(start=start_date, end=end_date, interval="1d").reset_index()
    if df.empty:
        print("ℹ️ No new data to fetch.")
        return df

    # Normalize Date to date only
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Fetch existing dates from DB
    existing_df = pd.read_sql(f"SELECT Date FROM {table_name}", engine)
    existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date

    # Keep only new rows
    new_rows = df[~df['Date'].isin(existing_df['Date'])]
    print(f"✅ {len(new_rows)} new rows ready to insert.")
    return new_rows


# ---------- STEP 2: STORE IN MYSQL ----------
def store_data_to_mysql(df, table_name="gold_prices"):
    """
    Store fetched data into MySQL database using SQLAlchemy.
    Update the connection string with your credentials.
    """

    # create connection engine
    engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")

    # Check if table exists
    if not df.empty:
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"✅ Appended {len(df)} new rows to '{table_name}'.")
    else:
        print(f"ℹ️ No new rows to append.")


# ---------- STEP 3: FETCH FROM MYSQL ----------
def fetch_data_from_mysql(table_name="gold_prices"):
    """
    Fetch stored data from MySQL database back into pandas DataFrame.
    """
    engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    print(f"✅ Retrieved {len(df)} rows from MySQL table '{table_name}'")
    return df


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    df = fetch_gold_data()
    if len(df)>0:
      store_data_to_mysql(df)
    df_retrieved = fetch_data_from_mysql()

    print(df_retrieved.tail())
