import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import calendar

import pandas as pd
import plotly.express as px
import streamlit as st

# Config the app
st.set_page_config(page_title="TrackMyFinance", page_icon="💸", layout="wide")

# Use path for the database (project root)
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_ROOT / "finance.db")

# Categories (May change in the future)
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

# --- Callbacks for synced widgets ---
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
				created_at TEXT DEFAULT (datetime('now'))
			)
			"""
		)
		# Ensure columns
		cols = {row[1] for row in conn.execute("PRAGMA table_info(transactions)").fetchall()}
		if "repeating" not in cols:
			conn.execute("ALTER TABLE transactions ADD COLUMN repeating INTEGER NOT NULL DEFAULT 0")


def insert_transaction(t_date: date, description: str, category: str, amount: float, repeating: bool = False):
	with get_conn() as conn:
		conn.execute(
			"INSERT INTO transactions (t_date, description, category, amount, repeating) VALUES (?, ?, ?, ?, ?)",
			(
				t_date.isoformat() if isinstance(t_date, date) else str(t_date),
				(description or "").strip(),
				category.strip(),
				float(amount),
				1 if repeating else 0,
			),
		)


def delete_transaction(tx_id: int):
	with get_conn() as conn:
		conn.execute("DELETE FROM transactions WHERE id = ?", (int(tx_id),))

# Load transactions with filters
def load_transactions(start: Optional[date] = None, end: Optional[date] = None, categories: Optional[List[str]] = None, repeating_only: Optional[bool] = None) -> pd.DataFrame:
	# Build query
	query = "SELECT id, t_date, description, category, amount, repeating FROM transactions WHERE 1=1"
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
		# repeating bool
		if "repeating" in df.columns:
			df["repeating"] = df["repeating"].astype(int).astype(bool)
	return df


# Default period, current month to date
def period_default() -> Tuple[date, date]:
	today = date.today()
	start = today.replace(day=1)

	last_day = calendar.monthrange(today.year, today.month)[1]
	end = today.replace(day=last_day)
	return start, end

# Helper
def ensure_category(cat: str, other_text: Optional[str]) -> str:
	if cat == "Other":
		return (other_text or "Other").strip() or "Other"
	return cat
# Helper
def render_summary(df: pd.DataFrame, start: date, end: date):
	if df.empty:
		st.info("No transactions in the selected period.")
		return

	n_days = (end - start).days + 1
	total_spent = float(df["amount"].sum())
	avg_daily = total_spent / max(n_days, 1)

	c1, c2, c3 = st.columns(3)
	c1.metric("Total spent", f"€{total_spent:,.2f}")
	c2.metric("Avg per day", f"€{avg_daily:,.2f}")
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
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating"}, inplace=True)
		cols = [c for c in ["ID", "Date", "Category", "Description", "Amount", "Repeating"] if c in show.columns]
		st.dataframe(show[cols], hide_index=True)

		csv = show.to_csv(index=False).encode("utf-8")
		st.download_button("Download CSV", data=csv, file_name="transactions.csv", mime="text/csv")


def render_delete(df: pd.DataFrame):
	if df.empty:
		return
	with st.expander("Delete a transaction"):
		options = {
			f"#{row.id} | {row.t_date.date()} | €{row.amount:.2f} | {row.category} | {row.description or ''}": int(row.id)
			for _, row in df.iterrows()
		}
		label = st.selectbox("Select transaction to delete", ["-"] + list(options.keys()))
		if label != "-":
			if st.button("Delete selected", type="primary"):
				delete_transaction(options[label])
				st.success("Deleted.")
				st.rerun()


# Helper to render multiple day view(dates)
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
	c1.metric("Total spent", f"€{total_spent:,.2f}")
	c2.metric("Avg per selected day", f"€{avg_daily:,.2f}")
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
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating"}, inplace=True)
		cols = [c for c in ["ID", "Date", "Category", "Description", "Amount", "Repeating"] if c in show.columns]

		st.dataframe(show[cols], hide_index=True)
		csv = show.to_csv(index=False).encode("utf-8")
		st.download_button("Download CSV", data=csv, file_name="transactions_selected_dates.csv", mime="text/csv")


def daterange_list(start: date, end: date) -> List[date]:
	days = (end - start).days
	return [start + timedelta(days=i) for i in range(days + 1)]


def main():
	init_db()

	st.title("💸 Track My Finance")
	st.caption("Simple personal spending tracker")

	# Simple routing between pages using session state
	if "page" not in st.session_state:
		st.session_state.page = "home"

	# Full screen state sidebar in filters (maybe located to somewhere else)

	if st.session_state.page == "multi":
		# Sidebar filters for multi-day view
		st.sidebar.header("Filters")
		category_filter = st.sidebar.multiselect("Categories", options=CATEGORIES, default=CATEGORIES)
		repeating_filter = st.sidebar.selectbox(
			"Repeating filter",
			options=["All", "Repeating", "Non-repeating"],
			index=0,
		)
		# Back to Home in sidebar
		if st.sidebar.button("← Back to Home"):
			st.session_state.page = "home"
			st.rerun()
		st.sidebar.markdown("---")
		

		st.subheader("Multi-day statistics")
		default_start, default_end = period_default()
		win = st.date_input(
			"Choose a date window",
			value=(default_start, default_end),
			help="Pick a period, then select specific dates from it",
		)
		if isinstance(win, tuple) and len(win) == 2:
			win_start, win_end = win
		else:
			win_start = win if isinstance(win, date) else date.today() - timedelta(days=7)
			win_end = win_start

		options = daterange_list(win_start, win_end)

		# Manage selected dates 
		key_sel = "multi_selected_dates"
		if key_sel not in st.session_state:
			st.session_state[key_sel] = options
		else:
			
			st.session_state[key_sel] = [d for d in st.session_state[key_sel] if d in options]

		col_a, col_b = st.columns([3, 1])
		with col_a:
			selected_dates = st.multiselect(
				"Select one or many dates",
				options=options,
				format_func=lambda d: d.strftime("%Y-%m-%d"),
				key=key_sel,
			)
		with col_b:
			if st.button("Select all"):
				st.session_state[key_sel] = options
				st.rerun()
			if st.button("Clear"):
				st.session_state[key_sel] = []
				st.rerun()

		# Map repeating filter to parameter
		repeating_only_multi = None
		if repeating_filter == "Repeating":
			repeating_only_multi = True
		elif repeating_filter == "Non-repeating":
			repeating_only_multi = False

		df = load_transactions(win_start, win_end, categories=category_filter, repeating_only=repeating_only_multi)
		if not df.empty:
		
			sel_set = set(selected_dates)
			if sel_set:
				df = df[df["t_date"].dt.date.isin(sel_set)]

		render_summary_for_dates(df, st.session_state.get(key_sel, []))
		return

	# Sidebar filters (Home)
	st.sidebar.header("Filters")
	default_start, default_end = period_default()
	period = st.sidebar.date_input(
		"Period",
		value=(default_start, default_end),
		help="Pick a start and end date for reports",
	)
	if isinstance(period, tuple) and len(period) == 2:
		start_date, end_date = period
	else:
	
		start_date = end_date = period if isinstance(period, date) else date.today()

	category_filter = st.sidebar.multiselect("Categories", options=CATEGORIES, default=CATEGORIES)

	# Add repeating filter on Home as well
	repeating_filter = st.sidebar.selectbox(
		"Repeating filter",
		options=["All", "Repeating", "Non-repeating"],
		index=0,
	)

	# Monthly spending limit controls (synced slider + number)
	st.sidebar.subheader("Monthly limit")
	if "limit_slider" not in st.session_state:
		st.session_state["limit_slider"] = 1000.0
		st.session_state["limit_number"] = 1000.0
	# Ensure number value is initialized even if slider already existed
	if "limit_number" not in st.session_state:
		st.session_state["limit_number"] = float(st.session_state.get("limit_slider", 1000.0))

	st.sidebar.slider(
		"Set limit (€)",
		min_value=0.0,
		max_value=10000.0,
		step=50.0,
		key="limit_slider",
		on_change=_on_limit_slider_change,
	)
	st.sidebar.number_input(
		"Or type limit (€)",
		min_value=0.0,
		max_value=100000.0,
		step=10.0,
		key="limit_number",
		on_change=_on_limit_number_change,
	)

	# Compute month-to-limit usage and show progress
	month_start, month_end = period_default()

	# Map filter for this computation
	repeating_only_for_limit = None
	if repeating_filter == "Repeating":

		repeating_only_for_limit = True
	elif repeating_filter == "Non-repeating":
		repeating_only_for_limit = False

	month_df = load_transactions(month_start, month_end, categories=category_filter, repeating_only=repeating_only_for_limit)
	month_total = float(month_df["amount"].sum()) if not month_df.empty else 0.0
	limit_val = float(st.session_state.get("limit_number", 1000.0))

	fraction = 0.0 if limit_val <= 0 else min(month_total / limit_val, 1.0)
	st.sidebar.progress(fraction, text=f"{month_total:,.2f} / {limit_val:,.2f} € this month")

	if limit_val > 0 and month_total > limit_val:
		st.sidebar.error("Monthly limit exceeded")
	else:
		st.sidebar.caption("Tracking monthly spend vs limit")

	st.sidebar.markdown("---")

	# Open multi-day stats button lives in the sidebar
	if st.session_state.page != "multi":
		if st.sidebar.button("Open multi-day stats", type="primary"):
			st.session_state.page = "multi"
			st.rerun()
	
	

	# Add transaction form
	st.subheader("Add a spending")

	# Category selection outside the form so it updates instantly
	col_cat, col_other = st.columns([1, 1])
	with col_cat:
		st.selectbox("Category", options=CATEGORIES, index=0, key="add_cat")
	with col_other:
		if st.session_state.get("add_cat", CATEGORIES[0]) == "Other":
			st.text_input("Insert other category", value="", key="add_other_cat")

	with st.form("add_tx_form", clear_on_submit=True):
		col1, col2 = st.columns([1, 1])
		with col1:
			t_date = st.date_input("Date", value=date.today())
		with col2:
			amount = st.number_input("Amount (€)", min_value=0.0, step=0.5, format="%.2f")

		description = st.text_input("Description (optional)")
		repeating_flag = st.checkbox("Monthly payment (repeating)")
		submitted = st.form_submit_button("Add spending", type="primary")
		if submitted:
			cate = st.session_state.get("add_cat", CATEGORIES[0])
			other_cat = st.session_state.get("add_other_cat", "") if cate == "Other" else ""
			final_cat = ensure_category(cate, other_cat)
			if amount <= 0:
				st.error("Amount must be greater than 0.")
			else:
				insert_transaction(t_date, description, final_cat, amount, repeating=repeating_flag)
				st.success("Added!")
				# Reseting the field after adding
				st.session_state["add_other_cat"] = ""

	# Load and render 
	repeating_ = None
	if repeating_filter == "Repeating":
		repeating_ = True
	elif repeating_filter == "Non-repeating":
		repeating_ = False

	df = load_transactions(start_date, end_date, categories=category_filter, repeating_only=repeating_)
	render_summary(df, start_date, end_date)
	render_delete(df)


if __name__ == "__main__":
	main()

