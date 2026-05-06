import sqlite3
import datetime
import os

DB_FILE = "recent_runs.db"

def init_db():
    """יוצר את מסד הנתונים המקומי ואת הטבלה אם הם לא קיימים"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # יצירת טבלת היסטוריה (SQL אמיתי שישמח את משרד החינוך!)
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
    """שומר ריצה חדשה למסד הנתונים הלוקאלי"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO runs (timestamp, state, action, reward, is_catastrophic, new_q_value)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, state, action, reward, is_catastrophic, new_q_value))
    
    conn.commit()
    conn.close()

# מפעילים את הפונקציה כדי לוודא שהטבלה קיימת כשהשרת עולה
init_db()