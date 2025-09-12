import sqlite3
from pathlib import Path


def project_root() -> Path:
    
    return Path(__file__).resolve().parent.parent


DB_PATH = project_root() / "finance.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t_date TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
    print(f"Initialized database at: {DB_PATH}")


if __name__ == "__main__":
    init_db()