import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import calendar

import pandas as pd
import plotly.express as px
import streamlit as st
import json
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError
try:
	import yfinance as yf  
except ImportError:  
	yf = None
try:
	
	from streamlit_searchbox import st_searchbox  
except Exception:
	st_searchbox = None  

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


# --- Yahoo Finance symbol search helper ---
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_symbol_search(query: str, quotes_count: int = 20, region: str = "US") -> List[dict]:
	"""Search Yahoo Finance for symbols matching query. Returns list of dicts with keys: symbol, name, exch.

	- Filters to Yahoo Finance quotes and common instrument types
	- Does not raise on errors; returns [] instead
	"""
	q = (query or "").strip()
	if not q:
		return []
	params = {
		"q": q,
		"quotesCount": quotes_count,
		"newsCount": 0,
		"listsCount": 0,
		"enableFuzzyQuery": False,
		"quotesQueryId": "tss_match_phrase_query",
		"multiQuoteQueryId": "multi_quote_single_token_query",
		"newsQueryId": "news_cie_vespa",
		"enableCb": False,
		"region": region,
		"lang": "en-US",
	}
	url = "https://query1.finance.yahoo.com/v1/finance/search?" + urlencode(params)
	try:
		# Add a browser-like User-Agent to avoid being blocked
		req = Request(url, headers={
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
		})
		with urlopen(req, timeout=8) as resp:
			payload = resp.read().decode("utf-8", errors="ignore")
			data = json.loads(payload)
			quotes = data.get("quotes", []) or []
			results = []
			for qd in quotes:
				# Keep only proper Yahoo quotes
				if not qd.get("isYahooFinance", True):
					continue
				qt = (qd.get("quoteType") or "").upper()
				if qt not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY"}:
					continue
				sym = (qd.get("symbol") or "").strip()
				if not sym:
					continue
				# Only include symbols that start with the query (case-insensitive)
				if not sym.upper().startswith(q.upper()):
					continue
				name = qd.get("shortname") or qd.get("longname") or qd.get("name") or ""
				exch = qd.get("exchDisp") or qd.get("exchangeDisplay") or qd.get("exchange") or ""
				results.append({"symbol": sym, "name": name, "exch": exch})
			return results
	except (URLError, TimeoutError, json.JSONDecodeError, Exception):
		return []


def insert_transaction(t_date: date, description: str, category: str, amount: float, repeating: bool = False, ticker: Optional[str] = None):
	"""Insert a transaction. For Investments with a ticker and available yfinance, compute purchase_price and shares."""
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
				# Silent fallback; purchase_price/shares remain None
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

# Load transactions with filters
def load_transactions(start: Optional[date] = None, end: Optional[date] = None, categories: Optional[List[str]] = None, repeating_only: Optional[bool] = None) -> pd.DataFrame:
	# Build query
	query = "SELECT id, t_date, description, category, amount, repeating, ticker FROM transactions WHERE 1=1"
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
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating", "ticker": "Ticker"}, inplace=True)
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating", "ticker": "Ticker", "purchase_price": "Purchase Price", "shares": "Shares"}, inplace=True)
		cols = [c for c in ["ID", "Date", "Category", "Description", "Amount", "Repeating", "Ticker", "Purchase Price", "Shares"] if c in show.columns]
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
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating", "ticker": "Ticker"}, inplace=True)
		show.rename(columns={"t_date": "Date", "description": "Description", "category": "Category", "amount": "Amount", "id": "ID", "repeating": "Repeating", "ticker": "Ticker", "purchase_price": "Purchase Price", "shares": "Shares"}, inplace=True)
		cols = [c for c in ["ID", "Date", "Category", "Description", "Amount", "Repeating", "Ticker", "Purchase Price", "Shares"] if c in show.columns]

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

	# Apply pending resets for add_* inputs BEFORE rendering widgets
	if st.session_state.pop("reset_add_inputs", False):
		st.session_state["add_other_cat"] = ""
		st.session_state["add_ticker"] = ""

	with col_cat:
		st.selectbox("Category", options=CATEGORIES, index=0, key="add_cat")
	with col_other:
		current_cat = st.session_state.get("add_cat", CATEGORIES[0])
		if current_cat == "Other":
			st.text_input("Insert other category", value="", key="add_other_cat")
		elif current_cat == "Investments":
			# Ticker selection is now inside the form below
			st.caption("Select ticker in the form below")

	with st.form("add_tx_form", clear_on_submit=True):
		col1, col2 = st.columns([1, 1])
		with col1:
			t_date = st.date_input("Date", value=date.today())
		with col2:
			amount = st.number_input("Amount (€)", min_value=0.0, step=0.5, format="%.2f")
		# Ticker input and suggestions inside the form when Investments selected
		if st.session_state.get("add_cat") == "Investments":
			if st_searchbox is not None:
				def _search_func_form(search: str) -> List[str]:
					items = yahoo_symbol_search(search, quotes_count=25)
					return [f"{it['symbol']} — {it['name']} ({it['exch']})".strip() for it in items]
				picked_label = st_searchbox(
					_search_func_form,
					key="form_add_ticker_searchbox",
					placeholder="Ticker (e.g., AAPL, MSFT)",
					default=st.session_state.get("add_ticker", ""),
				)
				if picked_label:
					sym = picked_label.split(" — ")[0].strip()
					if sym:
						st.session_state["add_ticker"] = sym
			else:
				qcol1, qcol2 = st.columns([2, 2])
				with qcol1:
					st.text_input("Ticker", value=st.session_state.get("add_ticker", ""), key="add_ticker")
				with qcol2:
					q = (st.session_state.get("add_ticker", "") or "").strip()
					labels = []
					if len(q) >= 2:
						suggestions = yahoo_symbol_search(q, quotes_count=25)
						if suggestions:
							labels = [f"{it['symbol']} — {it['name']} ({it['exch']})".strip() for it in suggestions]
					picked = st.selectbox("Suggestions", options=["-"] + labels if labels else ["-"], index=0, key="form_add_ticker_pick")

		description = st.text_input("Description (optional)")
		repeating_flag = st.checkbox("Monthly payment (repeating)")
		submitted = st.form_submit_button("Add spending", type="primary")
		if submitted:
			cate = st.session_state.get("add_cat", CATEGORIES[0])
			other_cat = st.session_state.get("add_other_cat", "") if cate == "Other" else ""
			final_cat = ensure_category(cate, other_cat)

			# Validate ticker 
			ticker_val = None
			if final_cat == "Investments":

				# Prefer typed/selected ticker
				ticker_val = (st.session_state.get("add_ticker", "") or "").strip().upper()
				if not ticker_val:
					picked_label = st.session_state.get("form_add_ticker_pick", "-")
					if picked_label and picked_label != "-":
						# Derive symbol from label (format: SYMBOL — Name (Exch))
						ticker_val = picked_label.split(" — ")[0].strip().upper()
				if not ticker_val:
					st.error("Please provide a stock ticker for Investments (e.g., AAPL).")
					st.stop()
			if amount <= 0:
				st.error("Amount must be greater than 0.")
			else:
				insert_transaction(t_date, description, final_cat, amount, repeating=repeating_flag, ticker=ticker_val)
				if final_cat == "Investments" and yf is None:
					st.warning("Install 'yfinance' to compute shares & live profit: pip install yfinance")
				st.success("Added!")
				# Schedule input reset 
				st.session_state["reset_add_inputs"] = True
				st.rerun()

	# Load and render 
	repeating_ = None
	if repeating_filter == "Repeating":
		repeating_ = True
	elif repeating_filter == "Non-repeating":
		repeating_ = False

	df = load_transactions(start_date, end_date, categories=category_filter, repeating_only=repeating_)
	render_summary(df, start_date, end_date)

	# Investments tracker view when Investments is among selected categories
	if "Investments" in category_filter:
		st.subheader("Investments — stock tracker")
		with st.expander("Track your investment tickers"):
			# Gather available tickers from DB 
			inv_df = load_transactions(categories=["Investments"]) 
			available = sorted([t for t in inv_df.get("ticker", pd.Series(dtype=str)).dropna().unique()]) if inv_df is not None and not inv_df.empty else []
			if not available:
				st.info("Add an Investment with a Ticker to enable tracking.")
			else:
				col1, col2 = st.columns([2, 1])
				with col1:
					selected_ticker = st.selectbox("Ticker", options=available)
				with col2:
					period_choice = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=2)

				if yf is None:
					st.warning("Stock data requires the 'yfinance' package. Install it with: pip install yfinance")
				else:
					try:
						data = yf.Ticker(selected_ticker).history(period=period_choice, auto_adjust=True)
						if data is None or data.empty:
							st.error("No price data found for this ticker/period. Check the ticker symbol.")
						else:
							last_close = float(data["Close"].iloc[-1])
							first_close = float(data["Close"].iloc[0])
							pct = ((last_close / first_close) - 1.0) * 100.0 if first_close else 0.0
							mc1, mc2 = st.columns(2)
							mc1.metric("Last close", f"{last_close:,.2f}")
							mc2.metric("Change", f"{pct:+.2f}%")

							plot_df = data.reset_index()
							fig = px.line(plot_df, x=plot_df.columns[0], y="Close", title=f"{selected_ticker} price — {period_choice}")
							st.plotly_chart(fig)
					except Exception as e:
						st.error(f"Failed to load stock data for {selected_ticker}: {e}")

		# Portfolio summary 
		st.subheader("Portfolio summary")
		pdf = load_transactions(categories=["Investments"])
		if pdf is None or pdf.empty or "ticker" not in pdf.columns:
			st.info("No investment data yet.")
		else:
			# Ensure shares; if missing try to backfill using purchase_price
			inv_rows = pdf.dropna(subset=["ticker"]).copy()
			# Fetch current prices for unique tickers
			prices = {}
			if yf is not None:
				try:
					unique_tickers = sorted([t for t in inv_rows["ticker"].dropna().unique()])
					if unique_tickers:
						# use yfinance Tickers for efficiency
						for tck in unique_tickers:
							try:
								data_cur = yf.Ticker(tck).history(period="1d", auto_adjust=True)
								if data_cur is not None and not data_cur.empty:
									prices[tck] = float(data_cur["Close"].iloc[-1])
							except Exception:
								pass
				except Exception:
					st.warning("Failed fetching live prices for some tickers.")
			else:
				st.info("Install 'yfinance' for live portfolio valuation.")

			portfolio_rows = []
			for tck, group in inv_rows.groupby("ticker"):
				total_invested = float(group["amount"].sum())
				# derive or sum shares
				shares_vals = group["shares"].dropna()
				shares_sum = float(shares_vals.sum()) if not shares_vals.empty else None
				current_price = prices.get(tck)
				current_value = None
				profit = None
				pct = None
				if current_price and shares_sum:
					current_value = shares_sum * current_price
					profit = current_value - total_invested
					pct = (profit / total_invested * 100.0) if total_invested else None
				portfolio_rows.append({
					"Ticker": tck,
					"Invested": total_invested,
					"Shares": shares_sum,
					"Current Price": current_price,
					"Current Value": current_value,
					"Profit": profit,
					"Return %": pct,
				})

			if not portfolio_rows:
				st.info("No priced holdings yet (add investments or install yfinance).")
			else:
				port_df = pd.DataFrame(portfolio_rows)
				# Totals
				total_invested_all = port_df["Invested"].sum()
				current_value_all = port_df["Current Value"].sum(min_count=1)
				profit_all = None
				return_pct_all = None
				if pd.notna(current_value_all):
					profit_all = current_value_all - total_invested_all
					return_pct_all = (profit_all / total_invested_all * 100.0) if total_invested_all else None
				mc1, mc2, mc3 = st.columns(3)
				mc1.metric("Invested", f"€{total_invested_all:,.2f}")
				if current_value_all and pd.notna(current_value_all):
					mc2.metric("Current value", f"€{current_value_all:,.2f}")
				if profit_all is not None:
					mc3.metric("Return", f"€{profit_all:,.2f}" + (f" ({return_pct_all:+.2f}%)" if return_pct_all is not None else ""))

				show_cols = ["Ticker", "Invested", "Shares", "Current Price", "Current Value", "Profit", "Return %"]
				st.dataframe(port_df[show_cols], hide_index=True)
				csv_port = port_df.to_csv(index=False).encode("utf-8")
				st.download_button("Download portfolio CSV", data=csv_port, file_name="portfolio.csv", mime="text/csv")

	render_delete(df)


if __name__ == "__main__":
	main()

