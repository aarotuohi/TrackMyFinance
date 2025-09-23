from datetime import date, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st

from catalog import build_ticker_catalog, refresh_ticker_catalog
from functions import (
	init_db,
	insert_transaction,
	delete_transaction,
	load_transactions,
	period_default,
    period_default_from_pref,
	ensure_category,
	daterange_list,
	render_summary,
	render_delete,
	render_summary_for_dates,
	_on_limit_slider_change,
	_on_limit_number_change,
	apply_theme,
	get_plotly_template,
    get_currency_symbol,
    fmt_currency,
	CATEGORIES,
)
try:
	import yfinance as yf
except ImportError:
	yf = None


# Config the app
st.set_page_config(page_title="TrackMyFinance", page_icon="💸", layout="wide")


def main():
	init_db()

	st.title("💸 Track My Finance")
	st.caption("Simple personal spending tracker")

	# Simple routing between pages using session state
	if "page" not in st.session_state:
		st.session_state.page = "home"
	if "theme" not in st.session_state:
		st.session_state["theme"] = "light"

	# Apply theme at start of render
	apply_theme(st.session_state.get("theme"))

	# Theme slider 
	tr_left, tr_right = st.columns([0.75, 0.25])
	with tr_right:
		current_theme = st.session_state.get("theme", "light")
		dark_on = st.toggle(
			"Dark mode",
			value=(current_theme == "dark"),
			help="Toggle between light and dark theme",
			key="theme_toggle",
		)
		# Apply theme immediately 
		new_theme = "dark" if dark_on else "light"
		if new_theme != current_theme:
			apply_theme(new_theme)
			st.rerun()

	# Sidebar settings
	with st.sidebar.container():
		st.markdown("### Settings")
		# Theme
		dark_sidebar = st.toggle(
			"Dark mode",
			value=(st.session_state.get("theme") == "dark"),
			help="Quickly toggle theme",
			key="theme_toggle_sidebar",
		)
		if dark_sidebar != (st.session_state.get("theme") == "dark"):
			apply_theme("dark" if dark_sidebar else "light")
			st.rerun()

		# Currency
		curr = st.selectbox(
			"Currency",
			options=["€", "$", "£", "¥"],
			index=["€", "$", "£", "¥"].index(st.session_state.get("currency_symbol", "€")),
			help="Choose the currency symbol used in totals",
		)
		if curr != st.session_state.get("currency_symbol"):
			st.session_state["currency_symbol"] = curr
			st.rerun()

		# Default period preference
		per_pref = st.selectbox(
			"Default period",
			options=["This month", "Last 7 days", "Last 30 days", "This year"],
			index=["This month", "Last 7 days", "Last 30 days", "This year"].index(
				st.session_state.get("default_period_pref", "This month")
			),
			help="Controls the initial period shown on Home and Multi-day views",
		)
		if per_pref != st.session_state.get("default_period_pref"):
			st.session_state["default_period_pref"] = per_pref
			st.rerun()

		# Layout density
		compact = st.toggle(
			"Compact layout",
			value=bool(st.session_state.get("compact_layout", False)),
			help="Reduce spacing in tables and controls",
		)
		st.session_state["compact_layout"] = compact

		st.markdown("---")

	# Full screen state sidebar in filters (maybe located to somewhere else)
	if st.session_state.get("compact_layout"):
		st.markdown(
			"""
			<style>
			/* Reduce dataframe cell padding */
			[data-testid="stStyledDataFrame"] td, [data-testid="stStyledDataFrame"] th { padding: 0.25rem 0.5rem; }
			/* Reduce general element spacing a bit */
			.css-1dp5vir, .block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }
			</style>
			""",
			unsafe_allow_html=True,
		)

	# Settings page
	if st.session_state.page == "settings":
		st.subheader("Settings")
		choice = st.radio(
			"Theme",
			options=["Light", "Dark"],
			index=(1 if st.session_state.get("theme") == "dark" else 0),
			horizontal=True,
			help="Prefer using the top-right switch for quick changes.",
		)
		if choice:
			new_theme = choice.lower()
			if new_theme != st.session_state.get("theme"):
				apply_theme(new_theme)
				st.success(f"Theme switched to {choice}.")
				st.rerun()
		st.sidebar.header("Navigation")
		if st.sidebar.button("← Back to Home"):
			st.session_state.page = "home"
			st.rerun()
		return

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

	# Sidebar filters HOme
	st.sidebar.header("Filters")

	# Use preferred default period
	default_start, default_end = period_default_from_pref(st.session_state.get("default_period_pref"))
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

	# Add repeating filter on Home
	repeating_filter = st.sidebar.selectbox(
		"Repeating filter",
		options=["All", "Repeating", "Non-repeating"],
		index=0,
	)

	# Monthly spending limit controls
	st.sidebar.subheader("Monthly limit")
	if "limit_slider" not in st.session_state:
		st.session_state["limit_slider"] = 1000.0
		st.session_state["limit_number"] = 1000.0
	# Ensure number value is initialized 
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
	month_start, month_end = period_default_from_pref(st.session_state.get("default_period_pref"))

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
	st.sidebar.progress(
		fraction,
		text=f"{fmt_currency(month_total)} / {fmt_currency(limit_val)} this period",
	)

	if limit_val > 0 and month_total > limit_val:
		st.sidebar.error("Monthly limit exceeded")
	else:
		st.sidebar.caption("Tracking monthly spend vs limit")

	st.sidebar.markdown("---")

	# Open multi-day stats and Settings in the sidebar
	if st.session_state.page != "multi":
		if st.sidebar.button("Open multi-day stats", type="primary"):
			st.session_state.page = "multi"
			st.rerun()
	if st.session_state.page != "settings":
		if st.sidebar.button("Open settings"):
			st.session_state.page = "settings"
			st.rerun()
	
	

	# Add transaction form
	st.subheader("Add a spending")

	# Category selection outside the form so it updates instantly
	col_cat, col_other = st.columns([1, 1])

	# Apply pending resets for add_* inputs BEFORE rendering widgets
	if st.session_state.pop("reset_add_inputs", False):
		st.session_state["add_other_cat"] = ""
		st.session_state["add_ticker"] = ""
		st.session_state["add_ticker_label"] = ""
		st.session_state["add_asset_type"] = "Stocks"

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
		# Categorized dropdown 
		if st.session_state.get("add_cat") == "Investments":
			catalog = build_ticker_catalog()
			asset_types = list(catalog.keys())
			# Determine default
			non_empty_types = [t for t in asset_types if catalog.get(t)]
			default_type = st.session_state.get("add_asset_type") or (non_empty_types[0] if non_empty_types else (asset_types[0] if asset_types else "Stocks"))
			idx_default = asset_types.index(default_type) if default_type in asset_types else 0
			colt1, colt2, colt3 = st.columns([1, 2, 1])
			with colt1:
				st.selectbox("Asset type", options=asset_types, index=idx_default, key="add_asset_type")
			with colt3:
				refresh_clicked = st.form_submit_button("Refresh list")
				if refresh_clicked:
					refresh_ticker_catalog()
					st.rerun()
			with colt2:
				atype = st.session_state.get("add_asset_type", default_type)
				entries = catalog.get(atype, [])
				# Auto-switch to a non-empty category 
				if not entries and non_empty_types:
					atype = non_empty_types[0]
					entries = catalog.get(atype, [])
					st.session_state["add_asset_type"] = atype
					st.caption("Showing available category due to empty listings.")
				labels = [f"{it['symbol']} — {it['name']}".strip() for it in entries] if entries else ["No options available"]
				st.selectbox("Ticker", options=labels, key="add_ticker_label")

		description = st.text_input("Description (optional)")
		repeating_flag = st.checkbox("Monthly payment (repeating)")
		submitted = st.form_submit_button("Add spending", type="primary")

		
		
		if submitted and not st.session_state.get("_refresh_clicked", False):
			cate = st.session_state.get("add_cat", CATEGORIES[0])
			other_cat = st.session_state.get("add_other_cat", "") if cate == "Other" else ""
			final_cat = ensure_category(cate, other_cat)

			# Validate ticker derived 
			ticker_val = None
			if final_cat == "Investments":
				label = st.session_state.get("add_ticker_label", "")
				if label and label != "No options available":
					ticker_val = label.split(" — ")[0].strip().upper()
				if not ticker_val:
					st.error("Please select a ticker from the dropdown.")
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

	# Investments tracker view 
	if "Investments" in category_filter:
		st.subheader("Investments — stock tracker")
		with st.expander("Track your investment tickers"):
			# Gather available tickers DB
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
							fig.update_layout(template=get_plotly_template())
							st.plotly_chart(fig)
					except Exception as e:
						st.error(f"Failed to load stock data for {selected_ticker}: {e}")

		# Portfolio summary 
		st.subheader("Portfolio summary")
		pdf = load_transactions(categories=["Investments"])
		if pdf is None or pdf.empty or "ticker" not in pdf.columns:
			st.info("No investment data yet.")
		else:
			# Ensure shares
			inv_rows = pdf.dropna(subset=["ticker"]).copy()

			# Fetch 
			prices = {}
			if yf is not None:
				try:
					unique_tickers = sorted([t for t in inv_rows["ticker"].dropna().unique()])
					if unique_tickers:
						# use yfinance tickers
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
					"Profit %": pct,
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
				mc1, mc2, mc3, mc4 = st.columns(4)
				mc1.metric("Invested", fmt_currency(total_invested_all))
				if current_value_all and pd.notna(current_value_all):
					mc2.metric("Current value", fmt_currency(current_value_all))
				if profit_all is not None:
					mc3.metric("Profit", fmt_currency(profit_all) if profit_all is not None else "")
				if return_pct_all is not None:
					mc4.metric("Profit %", f"{return_pct_all:+.2f}%" if return_pct_all is not None else "")

				show_cols = ["Ticker", "Invested", "Shares", "Current Price", "Current Value", "Profit", "Profit %"]
				st.dataframe(port_df[show_cols], hide_index=True)
				csv_port = port_df.to_csv(index=False).encode("utf-8")
				st.download_button("Download portfolio CSV", data=csv_port, file_name="portfolio.csv", mime="text/csv")

	render_delete(df)


if __name__ == "__main__":
	main()

