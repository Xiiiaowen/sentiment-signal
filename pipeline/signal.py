"""
signal.py — aggregate sentiment scores → daily signal + correlation analysis
"""
import datetime

import numpy as np
import pandas as pd
from scipy import stats


def build_sentiment_df(articles: list[dict], scores: list[dict]) -> pd.DataFrame:
    """
    Merge article metadata with sentiment scores into a flat DataFrame.

    Columns: date, title, publisher, link, label, positive, negative, neutral, score
    """
    rows = []
    for article, score in zip(articles, scores):
        ts = article.get("providerPublishTime", 0)
        if ts:
            date = datetime.datetime.utcfromtimestamp(ts).date()
        else:
            date = None
        rows.append({
            "date": date,
            "title": article.get("title", ""),
            "publisher": article.get("publisher", ""),
            "link": article.get("link", ""),
            **score,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_daily_signal(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily level and join with price returns.

    Returns DataFrame with columns:
        date, daily_sentiment, rolling_sentiment, close, daily_return, next_day_return
    """
    if sentiment_df.empty or price_df.empty:
        return pd.DataFrame()

    # Daily mean sentiment
    daily = (
        sentiment_df.groupby("date")["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "daily_sentiment"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily.sort_values("date", inplace=True)

    # 3-day rolling mean (smoothed signal)
    daily["rolling_sentiment"] = (
        daily["daily_sentiment"].rolling(window=3, min_periods=1).mean()
    )

    # Price returns
    price = price_df[["Close"]].copy()
    price.index = pd.to_datetime(price.index).normalize()
    price.index.name = "date"
    price = price.reset_index()
    price.columns = ["date", "close"]
    price["daily_return"] = price["close"].pct_change() * 100
    price["next_day_return"] = price["daily_return"].shift(-1)

    merged = pd.merge(daily, price, on="date", how="inner")
    merged.sort_values("date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def compute_correlations(signal_df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlation between daily_sentiment and same-day / next-day return.

    Returns dict:
        {same_day: {r, p, n}, next_day: {r, p, n}}
    """
    result = {}

    def _corr(x, y):
        mask = x.notna() & y.notna()
        x_, y_ = x[mask], y[mask]
        n = len(x_)
        if n < 3:
            return {"r": None, "p": None, "n": n}
        r, p = stats.pearsonr(x_, y_)
        return {"r": round(float(r), 3), "p": round(float(p), 4), "n": n}

    if signal_df.empty:
        return {"same_day": {"r": None, "p": None, "n": 0},
                "next_day": {"r": None, "p": None, "n": 0}}

    result["same_day"] = _corr(signal_df["daily_sentiment"], signal_df["daily_return"])
    result["next_day"] = _corr(signal_df["daily_sentiment"], signal_df["next_day_return"])
    return result


def regression_line(x: pd.Series, y: pd.Series):
    """
    Return (x_line, y_line) arrays for a linear regression trend line,
    filtering NaN values first. Returns (None, None) if insufficient data.
    """
    mask = x.notna() & y.notna()
    x_, y_ = x[mask].values, y[mask].values
    if len(x_) < 3:
        return None, None
    slope, intercept, *_ = stats.linregress(x_, y_)
    x_line = np.linspace(x_.min(), x_.max(), 100)
    y_line = slope * x_line + intercept
    return x_line, y_line
