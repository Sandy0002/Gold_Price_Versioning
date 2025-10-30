import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv
from src_logger.logger import get_logger



load_dotenv()
logger = get_logger(__name__)

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
    logger.info(f"Starting gold data fetch for ticker '{ticker}' and table '{table_name}'")

    try:
        engine = get_engine()
        gold = yf.Ticker(ticker)
        inspector = inspect(engine)
        table_exists = table_name in inspector.get_table_names()

        # Determine start date for fetching
        if table_exists:
            try:
                last_date_query = pd.read_sql(f'SELECT MAX("Date") AS last_date FROM {table_name}', engine)
                last_date = last_date_query["last_date"][0]

                if last_date is not None:
                    start_date = pd.to_datetime(last_date).date() + timedelta(days=1)
                    logger.debug(f"Table '{table_name}' exists. Last date: {last_date}. Starting from {start_date}.")
                else:
                    start_date = date.today() - timedelta(days=5)
                    logger.debug(f"Table '{table_name}' empty. Starting from {start_date}.")
            except Exception:
                start_date = date.today() - timedelta(days=5)
                logger.warning("Failed to determine last date from DB. Defaulting to last 5 days.")
        else:
            start_date = "2020-01-01"
            logger.debug(f"Table '{table_name}' not found. Starting from {start_date}.")

        end_date = date.today()

        df = gold.history(start=start_date, end=end_date, interval="1d").reset_index()

        if df.empty:
            logger.info("No new gold price data to fetch.")
            return pd.DataFrame()

        # Normalize Date column
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        # Filter out rows already in DB
        if table_exists:
            existing_dates = pd.read_sql(f'SELECT "Date" FROM {table_name}', engine)
            existing_dates['Date'] = pd.to_datetime(existing_dates['Date']).dt.date
            before_count = len(df)
            df = df[~df['Date'].isin(existing_dates['Date'])]
            after_count = len(df)
            logger.debug(f"Filtered {before_count - after_count} duplicate rows. {after_count} new rows remain.")

        logger.info(f"{len(df)} new rows ready to insert.")
        return df

    except Exception as e:
        logger.exception("Error occurred while fetching gold price data.")
        raise


# ---------- STEP 2: STORE DATA TO POSTGRES ----------
def store_data_postgres(df, table_name="gold_prices"):
    """
    Store fetched data into PostgreSQL.
    Creates table if not exists, appends new rows otherwise.
    """
    logger.info(f"Storing data in table {table_name}")

    try:
        if df.empty:
            logger.info( "No new rows to append.")
            return

        engine = get_engine()
        inspector = inspect(engine)
        table_exists = table_name in inspector.get_table_names()

        if not table_exists:
            # Create table and insert data
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            logger.info(f"Table '{table_name}' created and {len(df)} rows inserted.")
        else:
            # Append new rows
            df.to_sql(table_name, engine, if_exists="append", index=False)
            logger.info(f"Appended {len(df)} new rows to '{table_name}'.")

    except Exception as e:
        logger.exception("Error occurred while fetching gold price data.")
        raise


# ---------- STEP 3: FETCH DATA BACK ----------
def fetch_data_postgres(table_name="gold_prices",engine=None):
    try:
        engine = get_engine()
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        logger.info(f"Retrieved {len(df)} rows from '{table_name}'.")
        return df
    
    except Exception as e:
        logger.exception("Error occurred while fetching gold price data.")
        raise

# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    new_data = fetch_gold_data()
    if len(new_data) >0:
        store_data_postgres(new_data)
    df_retrieved = fetch_data_postgres()
    print(df_retrieved.tail())
