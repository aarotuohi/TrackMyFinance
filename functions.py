import sqlite3
from datetime import date, timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import calendar
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

try:
    import yfinance as yf  
except ImportError:  
    yf = None

# Use path for the database 
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_ROOT / "finance.db")

# Categories 
CATEGORIES = [
    "Groceries",
    "Transportation",
    "Restaurants",
    "Rent",
    "Utilities",
    "Entertainment",
    "Subscriptions",
    "Healthcare",
    "Education",
    "Travel",
    "Shopping",
    "Investments",
    "Other",
]


def _on_limit_slider_change():
    st.session_state["limit_number"] = st.session_state.get("limit_slider", 0.0)


def _on_limit_number_change():
    st.session_state["limit_slider"] = st.session_state.get("limit_number", 0.0)


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t_date TEXT NOT NULL,             
                description TEXT,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                repeating INTEGER NOT NULL DEFAULT 0,
                ticker TEXT,
                purchase_price REAL,
                shares REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        # Ensure columns
        cols = {row[1] for row in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        if "repeating" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN repeating INTEGER NOT NULL DEFAULT 0")
        if "ticker" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN ticker TEXT")
        if "purchase_price" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN purchase_price REAL")
        if "shares" not in cols:
            conn.execute("ALTER TABLE transactions ADD COLUMN shares REAL")

# insert transaction
def insert_transaction(t_date: date, description: str, category: str, amount: float, repeating: bool = False, ticker: Optional[str] = None):
    
    purchase_price = None
    shares = None
    if category == "Investments" and ticker and isinstance(ticker, str):
        if yf is not None:
            try:
                hist = yf.Ticker(ticker).history(period="1d", auto_adjust=True)
                if not hist.empty:
                    purchase_price = float(hist["Close"].iloc[-1])
                    if purchase_price > 0:
                        shares = float(amount) / purchase_price
            except Exception:
                st.error("Failed to fetch stock data. Please check the ticker symbol.")
                pass
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transactions (t_date, description, category, amount, repeating, ticker, purchase_price, shares) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                t_date.isoformat() if isinstance(t_date, date) else str(t_date),
                (description or "").strip(),
                category.strip(),
                float(amount),
                1 if repeating else 0,
                (ticker or None).strip().upper() if isinstance(ticker, str) else None,
                purchase_price,
                shares,
            ),
        )


def delete_transaction(tx_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))


def load_transactions(start: Optional[date] = None, end: Optional[date] = None, categories: Optional[List[str]] = None, repeating_only: Optional[bool] = None) -> pd.DataFrame:
    # Build query
    query = "SELECT id, t_date, description, category, amount, repeating, ticker, purchase_price, shares FROM transactions WHERE 1=1"
    params: list = []
    if start is not None:
        query += " AND t_date >= ?"
        params.append(start.isoformat())
    if end is not None:
        query += " AND t_date <= ?"
        params.append(end.isoformat())
    if categories:
        placeholders = ",".join(["?"] * len(categories))
        query += f" AND category IN ({placeholders})"
        params.extend(categories)
    if repeating_only is True:
        query += " AND repeating = 1"
    elif repeating_only is False:
        query += " AND repeating = 0"

    # Load into DataFrame
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["t_date"]) 

    # Post-process
    if not df.empty:
        df = df.sort_values("t_date")
        df["amount"] = df["amount"].astype(float)
        
        if "repeating" in df.columns:
            df["repeating"] = df["repeating"].astype(int).astype(bool)
    return df

# default period
def period_default() -> Tuple[date, date]:
    
    today = date.today()
    start = today.replace(day=1)

    last_day = calendar.monthrange(today.year, today.month)[1]
    end = today.replace(day=last_day)
    return start, end

# date default from pref
def period_default_from_pref(pref: Optional[str]) -> Tuple[date, date]:
   
    pref = (pref or "This month").lower()
    today = date.today()

    if pref in ("last 7 days", "7d"):
        start = today - timedelta(days=6)
        end = today
    elif pref in ("last 30 days", "30d"):
        start = today - timedelta(days=29)
        end = today
    elif pref in ("this year", "ytd"):
        start = today.replace(month=1, day=1)
        end = today
    else:  
        start = today.replace(day=1)
        last_day = calendar.monthrange(today.year, today.month)[1]
        end = today.replace(day=last_day)
    return start, end


def ensure_category(cat: str, other_text: Optional[str]) -> str:
   
    if cat == "Other":
        return (other_text or "Other").strip() or "Other"
    return cat


def render_summary(df: pd.DataFrame, start: date, end: date):
    
    if df.empty:
        st.info("No transactions in the selected period.")
        return

    n_days = (end - start).days + 1
    total_spent = float(df["amount"].sum())
    avg_daily = total_spent / max(n_days, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total spent", fmt_currency(total_spent))
    c2.metric("Avg per day", fmt_currency(avg_daily))
    c3.metric("Transactions", f"{len(df):,}")

    # Charts
    st.subheader("Spending breakdown")
    left, right = st.columns([1, 1])

    with left:
        by_cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig_pie = px.pie(by_cat, names="category", values="amount", title="By category")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie)

    with right:
        by_day = df.groupby("t_date", as_index=False)["amount"].sum()
        fig_day = px.bar(by_day, x="t_date", y="amount", title="By day")
        st.plotly_chart(fig_day)

    with st.expander("See table and export"):
        show = df.copy()
        show.rename(
            columns={
                "t_date": "Date",
                "description": "Description",
                "category": "Category",
                "amount": "Amount",
                "id": "ID",
                "repeating": "Repeating",
                "ticker": "Ticker",
                "purchase_price": "Purchase Price",
                "shares": "Shares",
            },
            inplace=True,
        )
        cols = [
            c
            for c in [
                "ID",
                "Date",
                "Category",
                "Description",
                "Amount",
                "Repeating",
                "Ticker",
                "Purchase Price",
                "Shares",
            ]
            if c in show.columns
        ]
        st.dataframe(show[cols], hide_index=True)

        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="transactions.csv", mime="text/csv")

#render delete
def render_delete(df: pd.DataFrame):
    
    if df.empty:
        return
    with st.expander("Delete a transaction"):
        options = {
            f"#{row.id} | {row.t_date.date()} | {fmt_currency(row.amount)} | {row.category} | {row.description or ''}": int(row.id)
            for _, row in df.iterrows()
        }
        label = st.selectbox("Select transaction to delete", ["-"] + list(options.keys()))
        if label != "-":
            if st.button("Delete selected", type="primary"):
                delete_transaction(options[label])
                st.success("Deleted.")
                st.rerun()

# render multiple days
def render_summary_for_dates(df: pd.DataFrame, selected_dates: List[date]):
    
    if not selected_dates:
        st.info("Select one or more dates to see stats.")
        return

    if df.empty:
        st.info("No transactions for the selected dates.")
        return

    days_count = len(set(selected_dates))
    total_spent = float(df["amount"].sum())
    avg_daily = total_spent / max(days_count, 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total spent", fmt_currency(total_spent))
    c2.metric("Avg per selected day", fmt_currency(avg_daily))
    c3.metric("Transactions", f"{len(df):,}")

    st.subheader("Spending breakdown")
    left, right = st.columns([1, 1])

    with left:
        by_cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        fig_pie = px.pie(by_cat, names="category", values="amount", title="By category")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie)

    with right:
        by_day = df.groupby("t_date", as_index=False)["amount"].sum()
        fig_day = px.bar(by_day, x="t_date", y="amount", title="By day")
        st.plotly_chart(fig_day)

    with st.expander("See table and export"):
        show = df.copy()
        show.rename(
            columns={
                "t_date": "Date",
                "description": "Description",
                "category": "Category",
                "amount": "Amount",
                "id": "ID",
                "repeating": "Repeating",
                "ticker": "Ticker",
                "purchase_price": "Purchase Price",
                "shares": "Shares",
            },
            inplace=True,
        )
        cols = [
            c
            for c in [
                "ID",
                "Date",
                "Category",
                "Description",
                "Amount",
                "Repeating",
                "Ticker",
                "Purchase Price",
                "Shares",
            ]
            if c in show.columns
        ]

        st.dataframe(show[cols], hide_index=True)
        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="transactions_selected_dates.csv",
            mime="text/csv",
        )


def daterange_list(start: date, end: date) -> List[date]:
    
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


# Theming utilities kept 
def apply_theme(theme: str = "light") -> None:
    
    theme = (theme or "light").lower()
    st.session_state["theme"] = theme
    
    pio.templates.default = "plotly_dark" if theme == "dark" else "plotly_white"
    
    if theme == "dark":
        st.markdown(
            """
            <style>
            body, .stApp { background-color: #0E1117; color: #FAFAFA; }
            </style>
            """,
            unsafe_allow_html=True,
        )

# plotly template
def get_plotly_template() -> str:

    return "plotly_dark" if st.session_state.get("theme") == "dark" else "plotly_white"

# currency getter
def get_currency_symbol() -> str:
    
    sym = st.session_state.get("currency_symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    return "€"

# currency formatter

def fmt_currency(value: Optional[float]) -> str:
    
    if value is None or pd.isna(value):
        return ""
    return f"{get_currency_symbol()}{float(value):,.2f}"