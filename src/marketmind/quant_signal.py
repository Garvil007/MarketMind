"""Deterministic, script-first quant signal.

This is the single rule engine that turns a `get_technicals` payload (see
market_data_server.get_technicals) into a BUY/HOLD/SELL call WITHOUT an LLM. The
Quant agent receives this as a strong prior and may override it with a stated
reason ("script first, LLM may override"), but if the LLM is absent or fails the
prior is a complete, reproducible answer on its own.

The same function is reused by the backtest engine and the dataset labeler so
that "what the script would have decided" is defined in exactly one place.

Rules (mirrors the scanner's intent in scanner.py):
  - A full `buy_signal` (all six conditions) is the strongest BUY.
  - `rs_high` + uptrend (above SMA50, EMA10>EMA20>EMA50) is a solid BUY.
  - Above SMA50 with healthy RSI leans BUY/HOLD.
  - Below SMA50 with a falling EMA stack or weak RSI is SELL.
  - Everything else is HOLD.

Confidence is the share of bullish (or bearish) conditions that agree, clamped
to a sane band so a thin signal never reads as certainty.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

# RSI bands used by the rule engine.
_RSI_OVERBOUGHT = 75.0
_RSI_WEAK = 45.0
_RSI_HEALTHY_LOW = 50.0

# Distinct bullish/bearish condition slots used to scale confidence:
# buy_signal, rs_high, above_sma_50, ema_stack, golden_cross, rsi.
_MAX_CONDS = 6


class SignalDecision(TypedDict):
    signal: str            # "BUY" | "HOLD" | "SELL"
    confidence: float      # 0.0 - 1.0
    bull_conditions: list[str]
    bear_conditions: list[str]
    reasons: list[str]     # human-readable, for the rationale / prompt


def _f(tech: dict, key: str, default: float = 0.0) -> float:
    """Read a float field defensively (tools may emit None / strings)."""
    v = tech.get(key, default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _ema_stack_up(tech: dict) -> Optional[bool]:
    """EMA10 > EMA20 > EMA50, or None if the EMAs aren't all present."""
    if not all(k in tech and tech[k] is not None for k in ("ema_10", "ema_20", "ema_50")):
        return None
    return _f(tech, "ema_10") > _f(tech, "ema_20") > _f(tech, "ema_50")


def _golden(tech: dict) -> Optional[bool]:
    """SMA50 > SMA200 (golden-cross posture), or None if SMA200 is absent."""
    if tech.get("sma_200") is None:
        return None
    return _f(tech, "sma_50") > _f(tech, "sma_200")


def compute_signal(tech: dict) -> SignalDecision:
    """Turn a get_technicals payload into a deterministic BUY/HOLD/SELL decision.

    Args:
        tech: dict with at least rsi_14, sma_50, last_close, above_sma_50, and
            (when available) ema_10/20/50, sma_200, rs_high, buy_signal, rs_value.

    Returns:
        SignalDecision with the call, a 0-1 confidence, and the conditions that
        drove it (used for the rationale and as the LLM's prior).
    """
    rsi = _f(tech, "rsi_14")
    above_sma_50 = bool(tech.get("above_sma_50"))
    buy_signal = bool(tech.get("buy_signal"))
    rs_high = bool(tech.get("rs_high"))
    ema_up = _ema_stack_up(tech)
    golden = _golden(tech)

    bull: list[str] = []
    bear: list[str] = []
    reasons: list[str] = []

    if buy_signal:
        bull.append("buy_signal")
        reasons.append("six-condition buy_signal is TRUE")
    if rs_high:
        bull.append("rs_high")
        reasons.append("relative strength at a ~6-month new high")
    if above_sma_50:
        bull.append("above_sma_50")
    else:
        bear.append("below_sma_50")
    if ema_up is True:
        bull.append("ema_stack_up")
    elif ema_up is False:
        bear.append("ema_stack_down")
    if golden is True:
        bull.append("golden_cross")
    elif golden is False:
        bear.append("death_cross")
    if rsi >= _RSI_HEALTHY_LOW and rsi < _RSI_OVERBOUGHT:
        bull.append("rsi_healthy")
    elif rsi >= _RSI_OVERBOUGHT:
        bear.append("rsi_overbought")
    elif rsi <= _RSI_WEAK:
        bear.append("rsi_weak")

    # --- Decide ---------------------------------------------------------
    if buy_signal or (rs_high and above_sma_50 and ema_up is True):
        signal = "BUY"
    elif (not above_sma_50) and (ema_up is False or rsi <= _RSI_WEAK):
        signal = "SELL"
    elif above_sma_50 and rsi < _RSI_OVERBOUGHT and ema_up is not False:
        signal = "BUY" if (rs_high or ema_up is True) else "HOLD"
    else:
        signal = "HOLD"

    # --- Confidence: how many of the distinct condition slots agree -----
    # Slots: buy_signal, rs_high, above_sma_50, ema_stack, golden_cross, rsi.
    # Scaling by a fixed denominator (not the observed ratio) keeps a single
    # confirming condition from reading as near-certainty.
    if signal == "BUY":
        conf = 0.45 + 0.5 * (len(bull) / _MAX_CONDS)
        if buy_signal:
            conf = max(conf, 0.85)
    elif signal == "SELL":
        conf = 0.45 + 0.5 * (len(bear) / _MAX_CONDS)
    else:  # HOLD — weak/mixed evidence, kept in a modest band
        conf = 0.4 + 0.1 * (1.0 - abs(len(bull) - len(bear)) / _MAX_CONDS)

    conf = round(min(0.95, max(0.3, conf)), 2)

    if not reasons:
        reasons.append(
            f"trend {'above' if above_sma_50 else 'below'} SMA50, RSI {rsi:.1f}, "
            f"EMA stack {'up' if ema_up else ('down' if ema_up is False else 'n/a')}"
        )

    return SignalDecision(
        signal=signal,
        confidence=conf,
        bull_conditions=bull,
        bear_conditions=bear,
        reasons=reasons,
    )


def prior_block(decision: SignalDecision) -> str:
    """Render the deterministic decision as a prior block for the Quant prompt."""
    import json

    return json.dumps(
        {
            "script_signal": decision["signal"],
            "script_confidence": decision["confidence"],
            "bull_conditions": decision["bull_conditions"],
            "bear_conditions": decision["bear_conditions"],
            "reasons": decision["reasons"],
        },
        indent=2,
    )
