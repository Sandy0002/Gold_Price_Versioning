# database.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT", 5432)
    )

def execute(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result
