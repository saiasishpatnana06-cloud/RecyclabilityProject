import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "recyclability_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            material TEXT,
            confidence REAL,
            recyclability_score REAL,
            status TEXT,
            disposal_tip TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(material, confidence, score, status, tip):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (timestamp, material, confidence, recyclability_score, status, disposal_tip)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, material, confidence, score, status, tip))
    conn.commit()
    conn.close()

def fetch_history_df():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df