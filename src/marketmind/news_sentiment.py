"""Offline ticker news sentiment (yfinance headlines + VADER).

Used by the paper trader / quant scoring WITHOUT the News MCP server: a direct
yfinance fetch scored by VADER, degraded to neutral on any failure so callers
never crash on flaky news. Mirrors news_server's schema handling for titles.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("marketmind.news_sentiment")

NEUTRAL: dict[str, Any] = {"compound": 0.0, "count": 0}


def _title(item: dict) -> str:
    """Extract the headline from a yfinance news item (new or legacy schema)."""
    content = item.get("content")
    if isinstance(content, dict):  # new nested schema
        return str(content.get("title", "") or "")
    return str(item.get("title", "") or "")  # legacy flat schema


def ticker_sentiment(ticker: str, limit: int = 12) -> dict[str, Any]:
    """Mean VADER compound over the ticker's recent headlines.

    Args:
        ticker: US stock symbol, e.g. "NVDA".
        limit: Max headlines to score.

    Returns:
        {"compound": float -1..1, "count": int} — neutral (0.0, 0) when news
        is unavailable or empty. Never raises.
    """
    try:
        import yfinance as yf
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        raw = yf.Ticker(ticker).news or []
        titles = [t for t in (_title(it) for it in raw[: max(0, limit)] if isinstance(it, dict)) if t]
        if not titles:
            return dict(NEUTRAL)

        vader = SentimentIntensityAnalyzer()
        scores = [vader.polarity_scores(t)["compound"] for t in titles]
        return {"compound": round(sum(scores) / len(scores), 4), "count": len(scores)}
    except Exception as exc:  # noqa: BLE001 - news is advisory; degrade to neutral
        log.debug(f"{ticker}: news sentiment failed — {exc}")
        return dict(NEUTRAL)
