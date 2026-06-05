"""News MCP server (FastMCP, Streamable HTTP, port 8002, /mcp).

Two tools: get_recent_news (yfinance headlines) and score_sentiment (VADER).
No LLM, no vector store — sentiment is VADER-only for the MVP. See CLAUDE.md.
"""
from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf
from fastmcp import FastMCP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

mcp = FastMCP("news")

_vader = SentimentIntensityAnalyzer()


def _as_iso(value) -> str:
    """Best-effort convert a yfinance publish time to an ISO-8601 string."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):  # legacy epoch seconds
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    return str(value)  # new schema already gives an ISO date string


def _normalize(item: dict) -> dict:
    """Map one yfinance news item to {title, publisher, link, published, summary}.

    Handles both the legacy flat schema and the newer nested `content` schema.
    """
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content is not None:  # new schema
        provider = content.get("provider") or {}
        url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {})
        return {
            "title": content.get("title", ""),
            "publisher": provider.get("displayName", ""),
            "link": url.get("url", ""),
            "published": _as_iso(content.get("pubDate") or content.get("displayTime")),
            "summary": content.get("summary", "") or content.get("description", ""),
        }
    # legacy flat schema
    return {
        "title": item.get("title", ""),
        "publisher": item.get("publisher", ""),
        "link": item.get("link", ""),
        "published": _as_iso(item.get("providerPublishTime")),
        "summary": item.get("summary", ""),
    }


@mcp.tool
def get_recent_news(ticker: str, limit: int = 12) -> dict:
    """Fetch recent news headlines for a stock ticker.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        limit: Max number of articles to return (default 12).

    Returns:
        {"ticker": str,
         "articles": [{"title": str, "publisher": str, "link": str,
                       "published": str (ISO-8601), "summary": str}, ...]}
        Returns an empty articles list (never raises) when news is unavailable.
    """
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001 - yfinance news is flaky; degrade to empty
        raw = []

    articles = [_normalize(it) for it in raw[: max(0, limit)] if isinstance(it, dict)]
    return {"ticker": ticker.upper(), "articles": articles}


@mcp.tool
def score_sentiment(headlines: list[str]) -> dict:
    """Score a list of news headlines with VADER sentiment.

    Args:
        headlines: List of headline strings to score.

    Returns:
        {"compound": float,   # mean of per-headline VADER compound scores, -1..1
         "label": str,        # "positive" (>=0.05) / "negative" (<=-0.05) / "neutral"
         "per_headline": [{"text": str, "compound": float}, ...]}
        Empty input yields compound 0.0, label "neutral", empty per_headline.
    """
    per_headline = []
    for text in headlines:
        text = "" if text is None else str(text)
        compound = _vader.polarity_scores(text)["compound"]
        per_headline.append({"text": text, "compound": round(compound, 4)})

    if per_headline:
        mean_compound = sum(p["compound"] for p in per_headline) / len(per_headline)
    else:
        mean_compound = 0.0
    mean_compound = round(mean_compound, 4)

    if mean_compound >= 0.05:
        label = "positive"
    elif mean_compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"compound": mean_compound, "label": label, "per_headline": per_headline}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)
