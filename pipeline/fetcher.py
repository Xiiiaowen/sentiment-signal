"""
fetcher.py — fetch news headlines and OHLCV price data via yfinance
"""
import time
import datetime as _dt

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Custom session with a browser-like User-Agent to avoid Yahoo Finance rate limits
# (Streamlit Cloud shared IPs get blocked with the default yfinance agent)
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
})


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol, session=_SESSION)


@st.cache_data(ttl=300)
def fetch_news(ticker: str) -> list[dict]:
    """
    Return list of news articles for the given ticker.
    Each dict: {title, publisher, providerPublishTime, link}
    Sorted most-recent first.
    Raises RuntimeError on rate limit so the caller can show a friendly message.
    """
    try:
        t = yf.Ticker(ticker, session=_SESSION)
        raw = t.news or []
    except Exception as e:
        err = type(e).__name__
        if any(k in err for k in ("RateLimit", "DataException", "Exception")) or "429" in str(e):
            raise RuntimeError(f"yf_error: {err}: {e}")
        raise

    articles = []
    for item in raw:
        # yfinance >= 0.2.50 nests everything under "content"
        content = item.get("content", item)
        title = content.get("title", "")
        publisher = (content.get("provider") or {}).get("displayName", "") or content.get("publisher", "")
        link = (
            (content.get("canonicalUrl") or {}).get("url", "")
            or (content.get("clickThroughUrl") or {}).get("url", "")
            or content.get("link", "")
        )
        pub_date = content.get("pubDate") or content.get("displayTime", "")
        if pub_date:
            try:
                ts = int(_dt.datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp())
            except Exception:
                ts = 0
        else:
            ts = int(content.get("providerPublishTime", 0))
        if not title:
            continue
        articles.append({
            "title": title,
            "publisher": publisher,
            "providerPublishTime": ts,
            "link": link,
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
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            session=_SESSION,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()
