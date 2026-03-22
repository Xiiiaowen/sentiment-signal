# CLAUDE.md — market-sentiment

## What This Project Is

A financial news sentiment analyser that fetches recent news headlines for any stock ticker,
scores them with FinBERT (a finance-specific NLP model), and visualises the sentiment signal
alongside price data.

This project demonstrates a different AI pattern from the other portfolio projects:
- credit-ai-assessor: tabular ML + XAI
- annual-report-rag: RAG / embeddings / vector search
- market-sentiment: NLP pipeline with a domain-specific transformer model + financial signal analysis

## How to Run

```bash
cd "d:/Projects/AI Agent/Try/market-sentiment"
pip install -r requirements.txt
streamlit run app.py
```

App runs at http://localhost:8501.
No API keys needed. FinBERT runs locally. yfinance is free.

On first run, HuggingFace downloads ProsusAI/finbert (~440 MB) and caches it in
~/.cache/huggingface/. Subsequent runs load from cache instantly.

## Python Environment

- Python 3.11 (Windows)
- Run pip with: pip install <package>

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
├── README.md
└── CLAUDE.md
```

## Data Flow

```
User selects ticker (e.g. AAPL) + lookback period
        ↓
pipeline/fetcher.py
  → yfinance.Ticker(ticker).news          → list of {title, publisher, providerPublishTime}
  → yfinance.download(ticker, period)     → OHLCV DataFrame
        ↓
pipeline/sentiment.py
  → for each headline: check disk cache (data/cache/{ticker}.json)
  → if not cached: run ProsusAI/finbert  → {label, score} (Positive/Negative/Neutral)
  → sentiment_score = positive_prob - negative_prob  (range: -1.0 to +1.0)
  → save new results to cache
        ↓
pipeline/signal.py
  → build DataFrame: date | headline | sentiment_score | price_return
  → daily_sentiment = mean of all article scores on that date
  → rolling_sentiment = 3-day rolling mean (smoothed signal)
  → compute: correlation(sentiment, same_day_return), correlation(sentiment, next_day_return)
        ↓
app.py renders 3 tabs
```

## Tech Stack

| Layer | Library | Notes |
|---|---|---|
| News data | yfinance `.news` | Free, no key, ~10-30 recent articles |
| Price data | yfinance `.download()` | OHLCV, multiple periods |
| Sentiment model | ProsusAI/finbert (HuggingFace) | Finance-specific BERT, 3-class |
| Inference | transformers pipeline | CPU inference, cached with @st.cache_resource |
| Disk cache | JSON files in data/cache/ | Keyed by MD5 hash of headline text |
| Charts | Plotly | Interactive, consistent with other projects |
| UI | Streamlit | 3-tab layout |

## FinBERT Details (important for interviews)

- Model: ProsusAI/finbert — fine-tuned on financial PhraseBank dataset
- Labels: Positive, Negative, Neutral
- Why FinBERT over VADER: VADER is a rule-based lexicon trained on social media text.
  FinBERT understands financial language — e.g. "revenue beat expectations" → Positive,
  "missed guidance" → Negative. Generic models often get these wrong.
- Sentiment score: positive_prob - negative_prob (float -1.0 to +1.0)
  - +1.0 = completely positive
  - 0.0 = neutral or mixed
  - -1.0 = completely negative
- Batch size: process up to 32 headlines at once for efficiency
- Cache: results stored by headline hash so FinBERT never re-runs on same text

## yfinance News Limitation (be honest about this)

yfinance `.news` returns only ~10-30 recent articles (typically last 1-3 days).
There is no free API that provides historical financial news going back months/years.

This means:
- Tab 1 (Sentiment Feed): works perfectly — shows current sentiment picture
- Tab 2 (Sentiment vs Price): shows sentiment scatter on the price chart for available dates
- Tab 3 (Signal Analysis): shows correlation stats, sentiment distribution — NOT a full backtest

In the README "What Could Be Improved" section, mention:
- With a paid NewsAPI or Alpha Vantage News key, you could backtest over 1-2 years of data
- The signal analysis tab shows the methodology; the same code would work with richer data

## UI Layout — 3 Tabs

### Sidebar
- Ticker input (text box, default: AAPL)
- Lookback period selector: 1mo, 3mo, 6mo, 1y (for price chart)
- Caption: "FinBERT · ProsusAI · yfinance"

### Tab 1 — Sentiment Feed
- Summary metrics row: total articles, % positive, % negative, % neutral, mean score
- Article list: for each article show:
  - Headline (clickable link if URL available)
  - Publisher + date
  - Sentiment badge: 🟢 Positive / 🔴 Negative / ⚪ Neutral
  - Confidence score (e.g. 87%)
  - Sentiment bar (color-coded)
- Sort: most recent first

### Tab 2 — Sentiment vs Price
- Dual-axis Plotly chart:
  - Left axis: closing price (line, blue)
  - Right axis: sentiment score per article (scatter dots, colored by sentiment)
  - If multiple articles on same day: show mean sentiment for that day as a bar
- Caption explaining what to look for (do sentiment spikes precede price moves?)

### Tab 3 — Signal Analysis
- st.info() explaining the data limitation (only ~30 days of news available)
- Sentiment distribution: pie chart or bar chart (Positive / Negative / Neutral counts)
- Correlation table:
  - Sentiment vs same-day return
  - Sentiment vs next-day return (lagged 1 day)
  - Note: with N < 30 data points, treat as indicative only
- Scatter plot: sentiment score (x) vs next-day return % (y)
  - Each dot = one day with news
  - Trend line (linear regression)
- Key insight caption: explain what the correlation means and its limitations

## Caching Strategy

### Model cache (@st.cache_resource)
```python
@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
```
Loaded once per Streamlit session. Survives re-runs triggered by widget changes.

### Sentiment disk cache
- File: data/cache/{ticker}.json
- Format: {"<md5_of_headline>": {"label": "Positive", "positive": 0.87, "negative": 0.05, "neutral": 0.08, "score": 0.82}}
- Load on startup, save after each new batch
- Purpose: avoid re-running FinBERT on headlines already scored

### Price data cache (@st.cache_data, ttl=300)
```python
@st.cache_data(ttl=300)
def fetch_prices(ticker, period):
    ...
```

## UI Conventions

- Theme: Streamlit default (dark mode fine)
- Charts: Plotly with dark theme (template="plotly_dark")
- Colors:
  - Positive sentiment: #2ecc71 (green)
  - Negative sentiment: #e74c3c (red)
  - Neutral: #95a5a6 (grey)
  - Price line: #3498db (blue)
- Use st.metric() for summary numbers
- Use st.expander() if needed for long article lists

## What This Demonstrates (for portfolio/interviews)

- **Domain-specific NLP**: FinBERT vs generic models — shows understanding of why model choice matters for financial text
- **Transformer inference pipeline**: loading, batching, caching a HuggingFace model
- **Financial signal construction**: from raw text to a numerical time-series signal
- **Correlation analysis**: linking NLP output to market data (cross-disciplinary)
- **Honest limitations**: the README and UI acknowledge data constraints — shows intellectual honesty valued in data science roles
- **No API keys needed**: fully self-contained, easy for anyone to run

## What NOT to Do

- Do not claim the backtest is statistically significant — sample is too small
- Do not add pandas-ta (not compatible with Python 3.11) — use manual pandas calculations for returns
- Do not use LangChain or any LLM wrapper — FinBERT runs directly via transformers pipeline
- Do not hardcode tickers — the UI should accept any valid yfinance ticker
- Do not skip the disk cache — FinBERT inference takes ~0.1s per headline; re-running on every page refresh would be slow

## Known Constraints

- yfinance news: ~10-30 articles, last 1-3 days only
- FinBERT model size: ~440 MB (downloaded once to HuggingFace cache)
- Streamlit Cloud: FinBERT + torch may be close to the 1 GB RAM limit; test before deploying
- No historical news without a paid API

## Potential Interview Questions

Q: Why FinBERT instead of VADER or TextBlob?
A: VADER is rule-based and trained on social media. Financial headlines use specific jargon
("beat estimates", "missed guidance", "raised outlook") that VADER misclassifies.
FinBERT is fine-tuned on the Financial PhraseBank dataset specifically for this domain.

Q: How would you extend this to a real trading signal?
A: Need historical news (NewsAPI, Refinitiv, Bloomberg), more tickers, and proper
walk-forward backtesting. Would also add position sizing, transaction costs, and
regime filters (don't trade sentiment during earnings blackouts).

Q: What are the limitations of sentiment analysis for trading?
A: News is often already priced in by the time it's published. High-frequency traders
react in milliseconds. Retail-accessible news sentiment works better for weekly/monthly
signals than daily. Also, sentiment about a company may not reflect the stock's
actual drivers (e.g., AAPL is driven by iPhone sales cycles, not daily news).
