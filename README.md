# Market Sentiment Analyser

A financial news sentiment analyser that fetches recent news headlines for any stock ticker, scores them with FinBERT (a finance-specific NLP model), and visualises the sentiment signal alongside price data.

---

## How It Works

```
User selects ticker + lookback period
  ↓
yfinance fetches recent news headlines + OHLCV price data
  ↓
ProsusAI/finbert scores each headline → Positive / Negative / Neutral
  sentiment_score = positive_prob − negative_prob  (range: −1.0 to +1.0)
  ↓
Results cached to disk (JSON keyed by headline hash)
  ↓
Daily sentiment aggregated → correlation with same-day and next-day returns
  ↓
3-tab Streamlit dashboard
```

The key insight: FinBERT is fine-tuned on financial text, so it correctly handles phrases like "revenue beat expectations" (Positive) and "missed guidance" (Negative) that generic sentiment models get wrong.

---

## Features

- **Domain-specific NLP** — ProsusAI/finbert, fine-tuned on the Financial PhraseBank dataset
- **No API keys needed** — yfinance is free; FinBERT runs locally on CPU
- **Disk cache** — FinBERT results saved by headline hash; re-running never scores the same headline twice
- **3-tab dashboard:**
  - **Sentiment Feed** — article list with badges, confidence scores, and colour-coded bars
  - **Sentiment vs Price** — dual-axis chart overlaying price and sentiment scatter
  - **Signal Analysis** — sentiment distribution, correlation stats, and scatter plot with trend line
- **Any ticker** — works with any valid yfinance symbol (stocks, ETFs, indices)

---

## Tech Stack

| Layer | Library | Notes |
|---|---|---|
| News data | yfinance `.news` | Free, no key, ~10–30 recent articles |
| Price data | yfinance `.download()` | OHLCV, multiple periods |
| Sentiment model | ProsusAI/finbert (HuggingFace) | Finance-specific BERT, 3-class |
| Inference | transformers pipeline | CPU inference, cached with @st.cache_resource |
| Disk cache | JSON files in data/cache/ | Keyed by MD5 hash of headline text |
| Charts | Plotly | Interactive, dark theme |
| UI | Streamlit | 3-tab layout |

---

## Getting Started

### 1. Install dependencies

```bash
cd market-sentiment
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

App runs at http://localhost:8501.

On first run, HuggingFace downloads ProsusAI/finbert (~440 MB) and caches it in `~/.cache/huggingface/`. Subsequent runs load from cache instantly.

---

## Project Structure

```
market-sentiment/
├── app.py                  # Streamlit UI — 3 tabs
├── pipeline/
│   ├── __init__.py
│   ├── fetcher.py          # yfinance: fetch news headlines + OHLCV price data
│   ├── sentiment.py        # FinBERT batch inference + disk cache
│   └── signal.py           # Aggregate scores → daily sentiment signal + analysis
├── data/
│   └── cache/              # JSON sentiment cache keyed by headline hash (gitignored)
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## Data Limitation

yfinance `.news` returns only ~10–30 recent articles (typically the last 1–3 days). There is no free API providing historical financial news going back months or years.

This means:
- **Sentiment Feed** — works perfectly; shows current sentiment picture
- **Sentiment vs Price** — shows sentiment scatter on the price chart for available dates
- **Signal Analysis** — shows correlation stats and methodology; with N < 30, treat as indicative only

---

## What Could Be Improved

**Historical news data**
With a paid NewsAPI or Alpha Vantage News key, you could backtest over 1–2 years of data. The signal analysis code would work unchanged — just more data points.

**Sentiment model alternatives**
FinBERT works well for headline-length text. For longer articles, a model with a larger context window (e.g. fin-distilroberta) would be more appropriate.

**Real-time streaming**
Currently refreshes every 5 minutes (ttl=300 cache). A WebSocket feed from a news provider would enable real-time sentiment tracking.

---

## What This Demonstrates

- **Domain-specific NLP** — FinBERT vs generic models; why model choice matters for financial text
- **Transformer inference pipeline** — loading, batching, and caching a HuggingFace model
- **Financial signal construction** — from raw text to a numerical time-series signal
- **Correlation analysis** — linking NLP output to market data (cross-disciplinary)
- **Honest limitations** — the UI and README acknowledge data constraints openly

---

## Disclaimer

For learning and demonstration purposes only. Not intended for trading or investment decisions.
