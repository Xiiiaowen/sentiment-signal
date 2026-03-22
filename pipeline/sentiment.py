"""
sentiment.py — FinBERT batch inference with disk cache
"""
import hashlib
import json
import os
from pathlib import Path

import streamlit as st

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_model():
    """Load ProsusAI/finbert once per Streamlit session."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
    )


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache(ticker: str) -> dict:
    path = CACHE_DIR / f"{ticker.upper()}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(ticker: str, cache: dict) -> None:
    path = CACHE_DIR / f"{ticker.upper()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def score_headlines(ticker: str, headlines: list[str]) -> list[dict]:
    """
    Score a list of headline strings with FinBERT.
    Uses disk cache keyed by MD5 hash of each headline.

    Returns list of dicts (same order as input):
        {label, positive, negative, neutral, score}
    where score = positive_prob - negative_prob ∈ [-1, 1]
    """
    if not headlines:
        return []

    cache = _load_cache(ticker)
    results = [None] * len(headlines)
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(headlines):
        key = _md5(text)
        if key in cache:
            results[i] = cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        model = load_model()
        # Batch in chunks of 32
        batch_size = 32
        all_raw = []
        for start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[start : start + batch_size]
            # Truncate long headlines to avoid token limit issues
            batch = [t[:512] for t in batch]
            all_raw.extend(model(batch))

        for idx, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
            raw = all_raw[idx]
            # raw is list of {label, score} dicts (top_k=None returns all classes)
            probs = {item["label"].lower(): item["score"] for item in raw}
            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            neu = probs.get("neutral", 0.0)
            label = max(probs, key=probs.get).capitalize()
            record = {
                "label": label,
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral": round(neu, 4),
                "score": round(pos - neg, 4),
            }
            results[orig_idx] = record
            cache[_md5(text)] = record

        _save_cache(ticker, cache)

    return results
