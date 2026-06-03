"""Contract A: the LangGraph shared state and per-agent result schemas.

This module is the single source of truth for the data that flows between
nodes. Servers, agents, and the orchestrator all conform to these shapes.
"""
from typing import TypedDict, Literal


class QuantResult(TypedDict):
    signal: Literal["BUY", "HOLD", "SELL"]
    confidence: float        # 0.0 - 1.0
    rsi_14: float
    sma_50: float
    last_close: float
    above_sma_50: bool
    rationale: str


class SentimentResult(TypedDict):
    label: Literal["positive", "neutral", "negative"]
    score: float             # VADER compound, -1.0 - 1.0
    headline_count: int
    summary: str


class RiskResult(TypedDict):
    level: Literal["low", "moderate", "high"]
    current_weight: float    # percent of portfolio, current
    projected_weight: float  # percent after the proposed buy
    note: str


class MarketMindState(TypedDict, total=False):
    query: str               # raw user question
    ticker: str              # extracted, e.g. "NVDA"
    proposed_notional: float # $ to evaluate (default 10000)
    quant: QuantResult
    sentiment: SentimentResult
    risk: RiskResult
    run_sentiment: bool      # router flag set after the quant node
    report_markdown: str     # final cited report
    citations: list[dict]    # [{"claim": str, "agent": "quant"|"sentiment"|"risk"}]
    errors: list[str]
