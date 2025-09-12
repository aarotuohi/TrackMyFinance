# TrackMyFinance

A simple personal finance tracking web app built with Streamlit and SQLite.

## Features
- Add transactions with date, category, amount, and description.
- Filter by date range and categories.
- Visualize spending with a pie chart (by category) and bar chart (by day).
- Download your filtered data as CSV.

## Quick start
1. Create and activate a Python 3.9+ environment.
2. Install dependencies.
3. Run the app.

### Windows PowerShell


pip install -r requirements.txt
streamlit run main.py


Then open the URL Streamlit prints (usually http://localhost:8501).

## Notes
- Data is stored locally in `finance.db` (SQLite) next to the app.
- To reset data, stop the app and delete `finance.db`.
