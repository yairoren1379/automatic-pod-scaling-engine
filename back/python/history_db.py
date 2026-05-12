import sqlite3
import datetime
import os

DB_FILE = "recent_runs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            state INTEGER,
            action INTEGER,
            reward REAL,
            is_catastrophic BOOLEAN,
            new_q_value REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_run(state: int, action: int, reward: float, is_catastrophic: bool, new_q_value: float):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO runs (timestamp, state, action, reward, is_catastrophic, new_q_value)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, state, action, reward, is_catastrophic, new_q_value))
    
    conn.commit()
    conn.close()

init_db()