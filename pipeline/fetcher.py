"""
fetcher.py — fetch news headlines and OHLCV price data via yfinance
"""
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)
def fetch_news(ticker: str) -> list[dict]:
    """
    Return list of news articles for the given ticker.
    Each dict: {title, publisher, providerPublishTime, link}
    Sorted most-recent first.
    """
    t = yf.Ticker(ticker)
    raw = t.news or []
    articles = []
    for item in raw:
        articles.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "providerPublishTime": item.get("providerPublishTime", 0),
            "link": item.get("link", ""),
        })
    articles.sort(key=lambda x: x["providerPublishTime"], reverse=True)
    return articles


@st.cache_data(ttl=300)
def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for the given ticker and period.
    Index: DatetimeIndex. Columns: Open, High, Low, Close, Volume.
    Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-level columns if present (yfinance >= 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()
