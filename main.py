import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

# Config the app
st.set_page_config(page_title="TrackMyFinance", page_icon="💸", layout="wide")

DB_PATH = "finance.db"
CATEGORIES = [
	"Groceries",
	"Transportation",
	"Restaurants",
	"Rent",
	"Utilities",
	"Entertainment",
	"Healthcare",
	"Education",
	"Travel",
	"Shopping",
	"Other",
]


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
		st.plotly_chart(fig_pie, use_container_width=True)

	with right:
		by_day = df.groupby("t_date", as_index=False)["amount"].sum()
		fig_day = px.bar(by_day, x="t_date", y="amount", title="By day")
		st.plotly_chart(fig_day, use_container_width=True)

	with st.expander("See table and export"):
		show = df.copy()
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID"}, inplace=True)
		st.dataframe(show[["ID", "Date", "Category", "Description", "Amount"]], use_container_width=True, hide_index=True)
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
		st.plotly_chart(fig_pie, use_container_width=True)

	with right:
		by_day = df.groupby("t_date", as_index=False)["amount"].sum()
		fig_day = px.bar(by_day, x="t_date", y="amount", title="By day")
		st.plotly_chart(fig_day, use_container_width=True)

	with st.expander("See table and export"):
		show = df.copy()
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID"}, inplace=True)
		st.dataframe(show[["ID", "Date", "Category", "Description", "Amount"]], use_container_width=True, hide_index=True)
		csv = show.to_csv(index=False).encode("utf-8")
		st.download_button("Download CSV", data=csv, file_name="transactions_selected_dates.csv", mime="text/csv")


def daterange_list(start: date, end: date) -> List[date]:
	days = (end - start).days
	return [start + timedelta(days=i) for i in range(days + 1)]


def main():
	#init_db()

	st.title("💸 Track My Finance")
	st.caption("Simple personal spending tracker")

	# Simple routing between pages using session state
	if "page" not in st.session_state:
		st.session_state.page = "home"

	# Top navigation buttons
	nav_col1, nav_col2 = st.columns([1, 1])
	with nav_col1:
		if st.session_state.page != "home":
			if st.button("← Back to Home", use_container_width=False):
				st.session_state.page = "home"
				st.rerun()
	with nav_col2:
		if st.session_state.page != "multi":
			if st.button("Open multi-day stats →", type="primary", use_container_width=False):
				st.session_state.page = "multi"
				st.rerun()

	if st.session_state.page == "multi":
		# Sidebar filters for multi-day view
		st.sidebar.header("Filters")
		category_filter = st.sidebar.multiselect("Categories", options=CATEGORIES, default=CATEGORIES)
		st.sidebar.markdown("---")
		st.sidebar.caption("Made with Streamlit")

		st.subheader("Multi-day statistics")
		# default_start, default_end = period_default()
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

		# Manage selected dates via session state
		key_sel = "multi_selected_dates"
		if key_sel not in st.session_state:
			st.session_state[key_sel] = options
		else:
			# Ensure values stay within options when window changes
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
			if st.button("Select all", use_container_width=True):
				st.session_state[key_sel] = options
				st.rerun()
			if st.button("Clear", use_container_width=True):
				st.session_state[key_sel] = []
				st.rerun()

		#df = load_transactions(win_start, win_end, categories=category_filter)
		if not df.empty:
			# Keep only selected dates
			sel_set = set(selected_dates)
			if sel_set:
				df = df[df["t_date"].dt.date.isin(sel_set)]

		render_summary_for_dates(df, st.session_state.get(key_sel, []))
		return

	# Sidebar filters
	st.sidebar.header("Filters")
	#default_start, default_end = period_default()
	period = st.sidebar.date_input(
		"Period",
		value=(default_start, default_end),
		help="Pick a start and end date for reports",
	)
	if isinstance(period, tuple) and len(period) == 2:
		start_date, end_date = period
	else:
		# Fallback for single-date selection
		start_date = end_date = period if isinstance(period, date) else date.today()

	category_filter = st.sidebar.multiselect("Categories", options=CATEGORIES, default=CATEGORIES)

	st.sidebar.markdown("---")
	

	# Add transaction form
	st.subheader("Add a spending")
	with st.form("add_tx_form", clear_on_submit=True):
		col1, col2, col3 = st.columns([1, 1, 1])
		with col1:
			t_date = st.date_input("Date", value=date.today())
		with col2:
			amount = st.number_input("Amount (€)", min_value=0.0, step=0.5, format="%.2f")
		with col3:
			cate = st.selectbox("Category", options=CATEGORIES, index=0)
			
			if cate == "Other":
				other_cat = st.text_input("Insert other category", value="")
			else:
				other_cat = ""

		description = st.text_input("Description (optional)")
		submitted = st.form_submit_button("Add spending", type="primary")
		if submitted:
			final_cat = ensure_category(cate, other_cat)
			if amount <= 0:
				st.error("Amount must be greater than 0.")
			else:
				insert_transaction(t_date, description, final_cat, amount)
				st.success("Added!")

	# Load and render data for selected period
	df = load_transactions(start_date, end_date, categories=category_filter)
	render_summary(df, start_date, end_date)
	render_delete(df)


if __name__ == "__main__":
	main()

