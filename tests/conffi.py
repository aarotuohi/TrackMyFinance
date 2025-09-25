import os
import sqlite3
import tempfile
import pytest

import functions as f

#Test setup wiht temp db 

@pytest.fixture(autouse=True)
def temp_db(patch):
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_finance.db")
        patch.setattr(f, "DB_PATH", db_path)
        f.init_db()
        yield db_path  

@pytest.fixture
def sample_transactions():
    return [
        {"date": "2025-01-01", "category": "Food", "amount": 12.59, "description": "Lunch", "repeating": False},
        {"date": "2025-01-02", "category": "Transport", "amount": 2.50, "description": "Bus", "repeating": False},
        {"date": "2025-01-03", "category": "Food", "amount": 20.99, "description": "Dinner", "repeating": True},
    ]
