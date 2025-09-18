from typing import Optional
from io import StringIO

import pandas as pd
import streamlit as st
from urllib.request import urlopen, Request


def _fetch_text(url: str, timeout: int = 12) -> Optional[str]:
    """Fetch text from a URL with a browser-like User-Agent; returns None on error."""
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


@st.cache_data(show_spinner=True, ttl=60 * 60 * 24)
def _get_us_equity_listings() -> pd.DataFrame:
    """Fetch US listings from NASDAQ Trader (NASDAQ + NYSE/AMEX).

    Returns a DataFrame with columns: symbol, name, is_etf.
    """
    urls = {
        "nasdaq": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    frames = []

    # NASDAQ
    text = _fetch_text(urls["nasdaq"]) or ""
    if text:
        df = pd.read_csv(StringIO(text), sep="|")
        if (
            "Symbol" in df.columns
            and "Security Name" in df.columns
            and "ETF" in df.columns
            and "Test Issue" in df.columns
        ):
            df = df[df["Test Issue"] != "Y"]
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": df["Symbol"]
                        .astype(str)
                        .str.strip()
                        .str.replace(".", "-", regex=False)
                        .str.upper(),
                        "name": df["Security Name"].astype(str).str.strip(),
                        "is_etf": df["ETF"].astype(str).str.upper().eq("Y"),
                    }
                )
            )

    # OTHER (NYSE/AMEX)
    text2 = _fetch_text(urls["other"]) or ""
    if text2:
        df2 = pd.read_csv(StringIO(text2), sep="|")
        if (
            "ACT Symbol" in df2.columns
            and "Security Name" in df2.columns
            and "ETF" in df2.columns
            and "Test Issue" in df2.columns
        ):
            df2 = df2[df2["Test Issue"] != "Y"]
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": df2["ACT Symbol"]
                        .astype(str)
                        .str.strip()
                        .str.replace(".", "-", regex=False)
                        .str.upper(),
                        "name": df2["Security Name"].astype(str).str.strip(),
                        "is_etf": df2["ETF"].astype(str).str.upper().eq("Y"),
                    }
                )
            )

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # drop duplicates (prefer first occurrence)
        all_df = all_df.drop_duplicates(subset=["symbol"])
        return all_df

    # If fetch fails, return empty DataFrame
    return pd.DataFrame(columns=["symbol", "name", "is_etf"])


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def build_ticker_catalog() -> dict:
    """Build categorized ticker catalog.

    Returns dict with keys: 'Stocks', 'Funds/ETFs', 'Crypto', 'Indexes', 'Currencies'
    mapping to list of dicts {symbol, name}.
    """
    cat = {"Stocks": [], "Funds/ETFs": [], "Crypto": [], "Indexes": [], "Currencies": []}

    # US equities/ETFs
    us = _get_us_equity_listings()
    if us is not None and not us.empty:
        stocks = us[~us["is_etf"]]
        funds = us[us["is_etf"]]
        cat["Stocks"] = stocks.sort_values("symbol")[
            ["symbol", "name"]
        ].to_dict(orient="records")
        cat["Funds/ETFs"] = funds.sort_values("symbol")[
            ["symbol", "name"]
        ].to_dict(orient="records")
    else:
        # Fallback curated lists when listings are unavailable (offline or blocked)
        cat["Stocks"] = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)"},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
        ]
        cat["Funds/ETFs"] = [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
            {"symbol": "VOO", "name": "Vanguard S&P 500 ETF"},
            {"symbol": "IVV", "name": "iShares Core S&P 500 ETF"},
        ]

    # Curated crypto (Yahoo symbols)
    cat["Crypto"] = [
        {"symbol": "BTC-USD", "name": "Bitcoin"},
        {"symbol": "ETH-USD", "name": "Ethereum"},
        {"symbol": "SOL-USD", "name": "Solana"},
        {"symbol": "ADA-USD", "name": "Cardano"},
        {"symbol": "XRP-USD", "name": "XRP"},
        {"symbol": "DOGE-USD", "name": "Dogecoin"},
    ]

    # Curated indexes
    cat["Indexes"] = [
        {"symbol": "^GSPC", "name": "S&P 500"},
        {"symbol": "^IXIC", "name": "Nasdaq Composite"},
        {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
        {"symbol": "^NDX", "name": "Nasdaq 100"},
    ]

    # Curated currency pairs (Yahoo FX symbols)
    cat["Currencies"] = [
        {"symbol": "EURUSD=X", "name": "EUR/USD"},
        {"symbol": "GBPUSD=X", "name": "GBP/USD"},
        {"symbol": "USDJPY=X", "name": "USD/JPY"},
        {"symbol": "USDCHF=X", "name": "USD/CHF"},
        {"symbol": "AUDUSD=X", "name": "AUD/USD"},
    ]

    return cat


def refresh_ticker_catalog() -> None:
    """Clear cached ticker data so the next access rebuilds it."""
    try:
        st.cache_data.clear()
    except Exception:
        # If cache clear fails, we fail silently to avoid breaking UI
        pass
