"""
app.py — SentimentSignal
Streamlit UI: 3 tabs — Sentiment Feed | Sentiment vs Price | Signal Analysis
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipeline.fetcher import fetch_news, fetch_prices
from pipeline.sentiment import score_headlines as _score_headlines
from pipeline.signal import (
    build_sentiment_df,
    compute_correlations,
    compute_daily_signal,
    regression_line,
)

@st.cache_data(ttl=3600, show_spinner=False)
def score_headlines(ticker: str, headlines: tuple) -> list[dict]:
    """Cached wrapper — tuple arg makes it hashable for st.cache_data."""
    return _score_headlines(ticker, list(headlines))


# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_POS   = "#2ecc71"
COLOR_NEG   = "#e74c3c"
COLOR_NEU   = "#7fb3d3"
COLOR_PRICE = "#3498db"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentSignal",
    page_icon="📈",
    layout="wide",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Article cards */
.article-card {
    background: #1e1e2e;
    border: 1px solid #2d2d42;
    border-radius: 10px;
    padding: 16px 20px 12px 20px;
    margin-bottom: 12px;
}
.article-title {
    font-size: 15px;
    font-weight: 600;
    color: #e0e0e0;
    line-height: 1.4;
    margin-bottom: 6px;
}
.article-title a {
    color: #e0e0e0;
    text-decoration: none;
}
.article-title a:hover {
    color: #3498db;
    text-decoration: underline;
}
.article-meta {
    font-size: 12px;
    color: #777;
    margin-bottom: 10px;
}
/* Sentiment badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-pos { background: #1a3d2b; color: #2ecc71; border: 1px solid #2ecc7155; }
.badge-neg { background: #3d1a1a; color: #e74c3c; border: 1px solid #e74c3c55; }
.badge-neu { background: #1a2a3d; color: #7fb3d3; border: 1px solid #7fb3d355; }
/* Sentiment bar */
.bar-track {
    background: #2a2a3e;
    border-radius: 4px;
    height: 5px;
    margin-top: 10px;
}
.bar-fill {
    height: 5px;
    border-radius: 4px;
}
/* Confidence text */
.conf-text {
    font-size: 12px;
    color: #666;
    float: right;
    margin-top: -18px;
}
/* Metric cards */
.metric-card {
    background: #1e1e2e;
    border: 1px solid #2d2d42;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #e0e0e0;
}
/* Page header */
.page-header {
    padding: 4px 0 20px 0;
    border-bottom: 1px solid #2d2d42;
    margin-bottom: 24px;
}
.page-header h2 {
    margin: 0 0 4px 0;
    font-size: 22px;
    color: #e0e0e0;
}
.page-header p {
    margin: 0;
    font-size: 13px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 SentimentSignal")
    st.markdown("<p style='color:#666;font-size:13px;margin-top:-8px;'>Financial News · FinBERT NLP</p>", unsafe_allow_html=True)
    st.markdown("---")
    ticker = st.text_input("Ticker symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").strip().upper()
    period = st.selectbox("Price lookback", ["1mo", "3mo", "6mo", "1y"], index=1)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:12px;color:#666;'>"
        "Powered by <a href='https://huggingface.co/ProsusAI/finbert' style='color:#3498db;'>ProsusAI/finbert</a> "
        "— a finance-specific BERT model fine-tuned on Financial PhraseBank. "
        "Price data via yfinance. No API keys required.</p>",
        unsafe_allow_html=True,
    )

# ── Data loading ──────────────────────────────────────────────────────────────
with st.spinner(f"Fetching news and prices for **{ticker}**…"):
    try:
        articles = fetch_news(ticker)
    except RuntimeError as e:
        st.error(
            f"**Yahoo Finance returned an error** — this is common on Streamlit Cloud shared IPs. "
            f"Wait 30–60 seconds and click **Rerun**, or run the app locally for best results. "
            f"\n\n`{e}`"
        )
        st.stop()
    price_df = fetch_prices(ticker, period)

if not articles:
    st.warning(f"No news found for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

headlines = [a["title"] for a in articles]

with st.spinner("Running FinBERT sentiment analysis…"):
    scores = score_headlines(ticker, tuple(headlines))

sentiment_df = build_sentiment_df(articles, scores)
signal_df    = compute_daily_signal(sentiment_df, price_df)

total      = len(sentiment_df)
n_pos      = int((sentiment_df["label"] == "Positive").sum())
n_neg      = int((sentiment_df["label"] == "Negative").sum())
n_neu      = int((sentiment_df["label"] == "Neutral").sum())
mean_score = sentiment_df["score"].mean()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📰  Sentiment Feed", "📊  Sentiment vs Price", "🔬  Signal Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sentiment Feed
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"""<div class='page-header'>
            <h2>Sentiment Feed — {ticker}</h2>
            <p>{total} articles analysed · sorted by most recent</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Summary metric row ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in [
        (c1, "Articles",   str(total),                    "#e0e0e0"),
        (c2, "Positive",   f"{n_pos / total * 100:.0f}%", COLOR_POS),
        (c3, "Negative",   f"{n_neg / total * 100:.0f}%", COLOR_NEG),
        (c4, "Neutral",    f"{n_neu / total * 100:.0f}%", COLOR_NEU),
        (c5, "Mean Score", f"{mean_score:+.2f}",          COLOR_POS if mean_score > 0.05 else (COLOR_NEG if mean_score < -0.05 else COLOR_NEU)),
    ]:
        col.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color};'>{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Article list ──────────────────────────────────────────────────────────
    for _, row in sentiment_df.iterrows():
        label = row["label"]
        if label == "Positive":
            badge_html = f"<span class='badge badge-pos'>● Positive</span>"
            bar_color  = COLOR_POS
        elif label == "Negative":
            badge_html = f"<span class='badge badge-neg'>● Negative</span>"
            bar_color  = COLOR_NEG
        else:
            badge_html = f"<span class='badge badge-neu'>● Neutral</span>"
            bar_color  = COLOR_NEU

        confidence = max(row["positive"], row["negative"], row["neutral"])
        date_str   = row["date"].strftime("%d %b %Y") if pd.notna(row["date"]) else "—"
        link       = row.get("link", "")
        title      = row["title"]
        title_html = f"<a href='{link}' target='_blank'>{title}</a>" if link else title
        bar_pct    = int((row["score"] + 1) / 2 * 100)

        st.markdown(
            f"""<div class='article-card'>
                <div class='article-title'>{title_html}</div>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                    <div>
                        <span style='font-size:12px;color:#888;font-weight:500;'>{row['publisher']}</span>
                        <span style='font-size:12px;color:#555;'>&nbsp;·&nbsp;{date_str}</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        {badge_html}
                        <span style='font-size:12px;color:#666;'>Confidence&nbsp;<b style='color:#aaa;'>{confidence * 100:.0f}%</b></span>
                        <span style='font-size:12px;color:#666;'>Score&nbsp;<b style='color:{bar_color};'>{row['score']:+.2f}</b></span>
                    </div>
                </div>
                <div class='bar-track'>
                    <div class='bar-fill' style='width:{bar_pct}%;background:{bar_color};'></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sentiment vs Price
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"""<div class='page-header'>
            <h2>Sentiment vs Price — {ticker}</h2>
            <p>Price history overlaid with per-article sentiment scores</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if price_df.empty:
        st.warning("Could not load price data. Check the ticker symbol.")
    else:
        daily_sent = (
            sentiment_df.groupby("date")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "mean_score"})
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df["Close"].squeeze(),
                name="Close Price",
                line=dict(color=COLOR_PRICE, width=2),
                hovertemplate="%{x|%d %b %Y}<br><b>$%{y:.2f}</b><extra>Close</extra>",
            ),
            secondary_y=False,
        )

        for label, color, symbol in [
            ("Positive", COLOR_POS, "circle"),
            ("Negative", COLOR_NEG, "circle"),
            ("Neutral",  COLOR_NEU, "circle-open"),
        ]:
            sub = sentiment_df[sentiment_df["label"] == label]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["score"],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=10, symbol=symbol, opacity=0.9,
                                line=dict(width=1, color=color)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Score: <b>%{y:.2f}</b><extra></extra>"
                    ),
                    customdata=sub[["title", "publisher"]].values,
                ),
                secondary_y=True,
            )

        bar_colors = [
            COLOR_POS if v > 0.05 else (COLOR_NEG if v < -0.05 else COLOR_NEU)
            for v in daily_sent["mean_score"]
        ]
        fig.add_trace(
            go.Bar(
                x=daily_sent["date"],
                y=daily_sent["mean_score"],
                name="Daily Mean",
                marker_color=bar_colors,
                opacity=0.3,
                hovertemplate="Date: %{x|%d %b %Y}<br>Mean Score: %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False,
                         gridcolor="#1e1e2e", zerolinecolor="#2d2d42")
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True,
                         range=[-1.2, 1.2], zeroline=True, zerolinecolor="#444",
                         gridcolor="#1e1e2e")
        fig.update_xaxes(gridcolor="#1e1e2e")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "<p style='font-size:13px;color:#666;padding:0 4px;'>"
            "<b style='color:#aaa;'>What to look for:</b> Do sentiment spikes (coloured dots) "
            "precede price moves? News is often a lagging signal — markets react in milliseconds. "
            "Persistent multi-day shifts are more informative than single-article spikes."
            "</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Signal Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"""<div class='page-header'>
            <h2>Signal Analysis — {ticker}</h2>
            <p>Sentiment distribution · correlation with price returns</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.info(
        "**Data note:** yfinance returns only ~10–30 recent articles (last 1–3 days). "
        "Correlations are indicative only — with N < 30 they are not statistically robust. "
        "A proper backtest needs 1–2 years of historical news (e.g. NewsAPI, Alpha Vantage)."
    )

    # ── Sentiment distribution ────────────────────────────────────────────────
    st.markdown("#### Sentiment Distribution")
    dist_col, pie_col = st.columns([1, 2])

    with dist_col:
        for label, count, color in [
            ("Positive", n_pos, COLOR_POS),
            ("Negative", n_neg, COLOR_NEG),
            ("Neutral",  n_neu, COLOR_NEU),
        ]:
            pct = count / total * 100
            st.markdown(
                f"""<div class='metric-card' style='margin-bottom:10px;text-align:left;
                            display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-size:13px;color:{color};font-weight:600;'>{label}</span>
                    <span style='font-size:18px;font-weight:700;color:#e0e0e0;'>{count}
                        <span style='font-size:13px;color:#666;font-weight:400;'>({pct:.0f}%)</span>
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

    with pie_col:
        pie_fig = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[n_pos, n_neg, n_neu],
            marker=dict(colors=[COLOR_POS, COLOR_NEG, COLOR_NEU],
                        line=dict(color="#0e1117", width=2)),
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # ── Correlation table ─────────────────────────────────────────────────────
    st.markdown("#### Sentiment–Return Correlation")

    if signal_df.empty:
        st.warning(
            "No overlapping dates between news and price data. "
            "This usually happens when all articles are from **today** — "
            "end-of-day price data is not yet available for the current trading day. "
            "Try again after market close, or try a different ticker."
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
            {"Pearson r": [r_same, r_next], "p-value": [p_same, p_next], "N (days)": [n_same, n_next]},
            index=["Sentiment vs Same-Day Return", "Sentiment vs Next-Day Return (lagged)"],
        )
        st.dataframe(corr_df, use_container_width=True)
        st.markdown(
            "<p style='font-size:12px;color:#666;'>"
            "Pearson r: +1 = perfect positive correlation, −1 = perfect negative, 0 = none. "
            "p &lt; 0.05 is the conventional threshold, but with N &lt; 30 any result is unreliable."
            "</p>",
            unsafe_allow_html=True,
        )

        # ── Scatter: sentiment vs next-day return ─────────────────────────────
        st.markdown("#### Sentiment Score vs Next-Day Return")

        plot_df = signal_df[["date", "daily_sentiment", "next_day_return"]].dropna()
        if len(plot_df) >= 2:
            dot_colors = plot_df["daily_sentiment"].apply(
                lambda s: COLOR_POS if s > 0.05 else (COLOR_NEG if s < -0.05 else COLOR_NEU)
            )
            x_line, y_line = regression_line(plot_df["daily_sentiment"], plot_df["next_day_return"])

            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=plot_df["daily_sentiment"],
                y=plot_df["next_day_return"],
                mode="markers",
                marker=dict(color=dot_colors, size=12, opacity=0.85,
                            line=dict(width=1, color="#0e1117")),
                hovertemplate=(
                    "Date: %{customdata}<br>"
                    "Sentiment: %{x:.2f}<br>"
                    "Next-Day Return: <b>%{y:.2f}%</b><extra></extra>"
                ),
                customdata=plot_df["date"].dt.strftime("%d %b %Y"),
                name="Trading Day",
            ))
            if x_line is not None:
                scatter_fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    line=dict(color="#f39c12", width=2, dash="dash"),
                    name="Linear trend",
                ))

            scatter_fig.add_hline(y=0, line_color="#333", line_dash="dot")
            scatter_fig.add_vline(x=0, line_color="#333", line_dash="dot")

            scatter_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                height=380,
                xaxis_title="Daily Sentiment Score",
                yaxis_title="Next-Day Return (%)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#333"),
                yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#333"),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            r_val = corr["next_day"]["r"]
            if r_val is not None:
                direction = "positive" if r_val > 0 else "negative"
                strength  = "weak" if abs(r_val) < 0.3 else ("moderate" if abs(r_val) < 0.6 else "strong")
                st.markdown(
                    f"<p style='font-size:13px;color:#666;'>"
                    f"<b style='color:#aaa;'>Insight:</b> The trend shows a <b style='color:#e0e0e0;'>"
                    f"{strength} {direction}</b> relationship (r = {r_val:+.3f}) between news sentiment "
                    f"and the following day's return for {ticker}. "
                    f"Based on very few data points — treat as exploratory only.</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Not enough days with both news and price data to plot the scatter.")
