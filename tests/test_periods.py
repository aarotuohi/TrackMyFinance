import datetime as dt

import functions as f

def t_period_this_month():
    start, end = f.period_default()
    today = dt.date.today()
    assert start.month == today.month and start.year == today.year
    assert end == today


def t_period_from_pref_last_7_days():
    start, end = f.period_default_from_pref("Last 7 days")
    today = dt.date.today()
    assert end == today
    # this includes today
    assert (today - start).days == 6  


def t_period_from_pref_last_30_days():
    start, end = f.period_default_from_pref("Last 30 days")
    today = dt.date.today()
    assert end == today
    assert (today - start).days == 29


def t_period_from_pref_this_year():
    start, end = f.period_default_from_pref("This year")
    today = dt.date.today()
    assert start == dt.date(today.year, 1, 1)
    assert end == today


def t_period_from_pref_unknown():
    start, end = f.period_default_from_pref("Unknown Pref")
    today = dt.date.today()
    assert start.month == today.month and start.year == today.year


def t_daterange_list():
    today = dt.date.today()
    start = today - dt.timedelta(days=2)
    days = f.daterange_list(start, today)
    assert len(days) == 3
    assert days[0] == start and days[-1] == today
