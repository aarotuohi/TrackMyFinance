import datetime as dt
import pandas as pd
import functions as f


def t_insert_and_load_transactions(sample_transactions):
    for t in sample_transactions:
        f.insert_transaction(**t)
    df = f.load_transactions()
    assert len(df) == 3

    assert set(df.category.unique()) == {"Food", "Transport"}


def t_filter_by_date(sample_transactions):
    for t in sample_transactions:
        f.insert_transaction(**t)
    start = dt.date(2025, 1, 2)
    end = dt.date(2025, 1, 3)
    df = f.load_transactions(start_date=start, end_date=end)
    assert len(df) == 2
    assert df.date.min().date() == start
    assert df.date.max().date() == end


def t_filter_by_category(sample_transactions):
    for t in sample_transactions:
        f.insert_transaction(**t)
    df = f.load_transactions(categories=["Food"])
    assert set(df.category.unique()) == {"Food"}

    assert len(df) == 2


def t_repeating_flag(sample_transactions):
    for t in sample_transactions:
        f.insert_transaction(**t)
    df = f.load_transactions()

    assert df.loc[df.description == "Dinner", "repeating"].iloc[0] is True


def t_ensure_category_creates_other():
    f.insert_transaction(date="2025-02-01", category="TestOther", amount=10.0, description="Test", repeating=False)
    df = f.load_transactions()

    assert "TestOther" in set(df.category.unique())


def t_currency_formatting():
    import streamlit as st
    if "currency" not in st.session_state:
        st.session_state.currency = "USD"
    formatted = f.fmt_currency(1234.5)

    assert "$" in formatted and "1,234.50" in formatted

    assert f.fmt_currency(None) == "-"


def t_export_dataframe(sample_transactions):

    for t in sample_transactions:
        f.insert_transaction(**t)
    df = f.load_transactions()
    
    for col in ["date", "category", "amount", "description", "repeating"]:
        assert col in df.columns

    assert df.amount.sum() == 37.5
