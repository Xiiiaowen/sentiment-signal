"""
app.py — Market Sentiment Analyser
Streamlit UI: 3 tabs — Sentiment Feed | Sentiment vs Price | Signal Analysis
"""
import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipeline.fetcher import fetch_news, fetch_prices
from pipeline.sentiment import score_headlines
from pipeline.signal import (
    build_sentiment_df,
    compute_correlations,
    compute_daily_signal,
    regression_line,
)

# ── Colors ──────────────────────────────────────────────────────────────────
COLOR_POS = "#2ecc71"
COLOR_NEG = "#e74c3c"
COLOR_NEU = "#95a5a6"
COLOR_PRICE = "#3498db"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyser",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Market Sentiment")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    period = st.selectbox("Price lookback", ["1mo", "3mo", "6mo", "1y"], index=1)
    st.caption("FinBERT · ProsusAI · yfinance")
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Fetches recent news headlines for any stock ticker, "
        "scores them with [FinBERT](https://huggingface.co/ProsusAI/finbert) "
        "(a finance-specific NLP model), and visualises the sentiment signal "
        "alongside price data."
    )

# ── Data loading ─────────────────────────────────────────────────────────────
with st.spinner(f"Loading data for {ticker}…"):
    articles = fetch_news(ticker)
    price_df = fetch_prices(ticker, period)

if not articles:
    st.warning(f"No news found for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

headlines = [a["title"] for a in articles]

with st.spinner("Running FinBERT sentiment analysis…"):
    scores = score_headlines(ticker, headlines)

sentiment_df = build_sentiment_df(articles, scores)
signal_df = compute_daily_signal(sentiment_df, price_df)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📰 Sentiment Feed", "📊 Sentiment vs Price", "🔬 Signal Analysis"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sentiment Feed
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"Sentiment Feed — {ticker}")

    total = len(sentiment_df)
    n_pos = (sentiment_df["label"] == "Positive").sum()
    n_neg = (sentiment_df["label"] == "Negative").sum()
    n_neu = (sentiment_df["label"] == "Neutral").sum()
    mean_score = sentiment_df["score"].mean()

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Articles", total)
    col2.metric("Positive", f"{n_pos / total * 100:.0f}%")
    col3.metric("Negative", f"{n_neg / total * 100:.0f}%")
    col4.metric("Neutral", f"{n_neu / total * 100:.0f}%")
    col5.metric("Mean Score", f"{mean_score:+.2f}")

    st.markdown("---")

    # Article list
    for _, row in sentiment_df.iterrows():
        label = row["label"]
        if label == "Positive":
            badge = "🟢 Positive"
            bar_color = COLOR_POS
        elif label == "Negative":
            badge = "🔴 Negative"
            bar_color = COLOR_NEG
        else:
            badge = "⚪ Neutral"
            bar_color = COLOR_NEU

        confidence = max(row["positive"], row["negative"], row["neutral"])
        date_str = row["date"].strftime("%d %b %Y") if pd.notna(row["date"]) else "—"

        with st.container():
            title = row["title"]
            link = row.get("link", "")
            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            info_col, badge_col, score_col = st.columns([3, 1, 1])
            info_col.caption(f"{row['publisher']}  ·  {date_str}")
            badge_col.markdown(f"**{badge}**")
            score_col.caption(f"Confidence: {confidence * 100:.0f}%")

            # Sentiment bar: score maps from -1→+1 to 0→100%
            bar_pct = int((row["score"] + 1) / 2 * 100)
            st.markdown(
                f"""<div style="background:#2d2d2d;border-radius:4px;height:6px;margin-bottom:12px;">
                <div style="width:{bar_pct}%;background:{bar_color};height:6px;border-radius:4px;"></div>
                </div>""",
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sentiment vs Price
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header(f"Sentiment vs Price — {ticker}")

    if price_df.empty:
        st.warning("Could not load price data. Check the ticker symbol.")
    else:
        # Build daily sentiment for scatter overlay
        daily_sent = (
            sentiment_df.groupby("date")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "mean_score"})
        )
        daily_sent["label_agg"] = daily_sent["mean_score"].apply(
            lambda s: "Positive" if s > 0.05 else ("Negative" if s < -0.05 else "Neutral")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Price line (left axis)
        price_close = price_df["Close"].copy()
        if isinstance(price_close.columns if hasattr(price_close, "columns") else None, pd.MultiIndex):
            price_close = price_close.iloc[:, 0]
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df["Close"].squeeze(),
                name="Close Price",
                line=dict(color=COLOR_PRICE, width=2),
                hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Per-article scatter (right axis)
        for label, color, symbol in [
            ("Positive", COLOR_POS, "circle"),
            ("Negative", COLOR_NEG, "circle"),
            ("Neutral", COLOR_NEU, "circle-open"),
        ]:
            mask = sentiment_df["label"] == label
            sub = sentiment_df[mask]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["score"],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=9, symbol=symbol, opacity=0.85),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Score: %{y:.2f}<extra></extra>"
                    ),
                    customdata=sub[["title", "publisher"]].values,
                ),
                secondary_y=True,
            )

        # Daily mean sentiment bars (right axis)
        bar_colors = [
            COLOR_POS if v > 0.05 else (COLOR_NEG if v < -0.05 else COLOR_NEU)
            for v in daily_sent["mean_score"]
        ]
        fig.add_trace(
            go.Bar(
                x=daily_sent["date"],
                y=daily_sent["mean_score"],
                name="Daily Mean Sentiment",
                marker_color=bar_colors,
                opacity=0.35,
                hovertemplate="Date: %{x|%d %b %Y}<br>Mean Score: %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(
            title_text="Sentiment Score",
            secondary_y=True,
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor="#555",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "**What to look for:** Do sentiment spikes (green/red dots) precede price moves? "
            "News sentiment is often a lagging indicator — markets price in information fast. "
            "Persistent multi-day sentiment shifts are more informative than single-article spikes."
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Signal Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header(f"Signal Analysis — {ticker}")

    st.info(
        "**Data limitation:** yfinance news returns only ~10–30 recent articles "
        "(typically the last 1–3 days). The correlations below are indicative only — "
        "with fewer than 30 data points, they are not statistically robust. "
        "A proper backtest would require 1–2 years of historical news data "
        "(e.g. NewsAPI, Alpha Vantage News)."
    )

    # ── Sentiment distribution ───────────────────────────────────────────────
    st.subheader("Sentiment Distribution")

    dist_col, pie_col = st.columns([1, 2])
    with dist_col:
        st.metric("Positive", f"{n_pos} articles ({n_pos / total * 100:.0f}%)")
        st.metric("Negative", f"{n_neg} articles ({n_neg / total * 100:.0f}%)")
        st.metric("Neutral", f"{n_neu} articles ({n_neu / total * 100:.0f}%)")

    with pie_col:
        pie_fig = go.Figure(
            go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[n_pos, n_neg, n_neu],
                marker=dict(colors=[COLOR_POS, COLOR_NEG, COLOR_NEU]),
                hole=0.4,
                textinfo="label+percent",
            )
        )
        pie_fig.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # ── Correlation table ────────────────────────────────────────────────────
    st.subheader("Sentiment–Return Correlation")

    if signal_df.empty:
        st.warning(
            "No overlapping dates between news and price data. "
            "This usually happens when all news articles are from **today** — "
            "end-of-day price data is not yet available for the current trading day. "
            "Try again after market close, or check a ticker with older cached news."
        )
    else:
        corr = compute_correlations(signal_df)

        def _fmt_corr(c):
            if c["r"] is None:
                return "—", "—", str(c["n"])
            return f"{c['r']:+.3f}", f"{c['p']:.4f}", str(c["n"])

        r_same, p_same, n_same = _fmt_corr(corr["same_day"])
        r_next, p_next, n_next = _fmt_corr(corr["next_day"])

        corr_df = pd.DataFrame(
            {
                "Correlation": [r_same, r_next],
                "p-value": [p_same, p_next],
                "N (days)": [n_same, n_next],
            },
            index=["Sentiment vs Same-Day Return", "Sentiment vs Next-Day Return (lagged)"],
        )
        st.dataframe(corr_df, use_container_width=True)
        st.caption(
            "Pearson r: +1 = perfect positive correlation, −1 = perfect negative, 0 = no linear relationship. "
            "p < 0.05 is the conventional significance threshold, but with N < 30 any result is unreliable."
        )

        # ── Scatter: sentiment vs next-day return ────────────────────────────
        st.subheader("Sentiment Score vs Next-Day Return")

        plot_df = signal_df[["date", "daily_sentiment", "next_day_return"]].dropna()
        if len(plot_df) >= 2:
            dot_colors = plot_df["daily_sentiment"].apply(
                lambda s: COLOR_POS if s > 0.05 else (COLOR_NEG if s < -0.05 else COLOR_NEU)
            )
            x_line, y_line = regression_line(plot_df["daily_sentiment"], plot_df["next_day_return"])

            scatter_fig = go.Figure()
            scatter_fig.add_trace(
                go.Scatter(
                    x=plot_df["daily_sentiment"],
                    y=plot_df["next_day_return"],
                    mode="markers",
                    marker=dict(color=dot_colors, size=11, opacity=0.8),
                    hovertemplate=(
                        "Date: %{customdata}<br>"
                        "Sentiment: %{x:.2f}<br>"
                        "Next-Day Return: %{y:.2f}%<extra></extra>"
                    ),
                    customdata=plot_df["date"].dt.strftime("%d %b %Y"),
                    name="Trading Day",
                )
            )
            if x_line is not None:
                scatter_fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#f39c12", width=2, dash="dash"),
                        name="Trend (linear fit)",
                    )
                )

            scatter_fig.add_hline(y=0, line_color="#555", line_dash="dot")
            scatter_fig.add_vline(x=0, line_color="#555", line_dash="dot")

            scatter_fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Daily Sentiment Score",
                yaxis_title="Next-Day Return (%)",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            r_val = corr["next_day"]["r"]
            if r_val is not None:
                direction = "positive" if r_val > 0 else "negative"
                strength = "weak" if abs(r_val) < 0.3 else ("moderate" if abs(r_val) < 0.6 else "strong")
                st.caption(
                    f"**Insight:** The trend line shows a **{strength} {direction}** relationship "
                    f"(r = {r_val:+.3f}) between news sentiment and the following day's return for {ticker}. "
                    "Remember: this is based on very few data points and should not be used to make trading decisions."
                )
        else:
            st.info("Not enough days with both news and price data to plot the scatter.")
