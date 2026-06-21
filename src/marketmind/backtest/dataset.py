"""Build training data from history for BOTH models.

For every historical bar (across one or many tickers) we compute the scanner
feature row and a forward-return label, then emit two artifacts:

  - a tabular dataset (CSV)  -> ml_model.py (scikit-learn classifier)
  - an instruction JSONL     -> llm_finetune.py (LoRA/QLoRA SFT)

Label: forward return over `horizon` bars maps to BUY / SELL / HOLD via
`up_thresh` / `down_thresh`. This is the supervised target — "given these
technicals, what should the call have been to capture the next move".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from marketmind import quant_signal
from marketmind.backtest.features import build_features, fetch_history, row_to_tech

# Columns fed to the scikit-learn model (order matters — ml_model relies on it).
FEATURE_COLUMNS = [
    "rsi_14", "sma_50", "sma_200", "ema_10", "ema_20", "ema_50",
    "pct_from_sma_50", "plus_di", "weekly_rsi", "rs_value",
    "above_sma_50", "rs_high",
    "cond1", "cond2", "cond3", "cond4", "cond5", "cond6", "buy_signal",
]
LABELS = ["SELL", "HOLD", "BUY"]


def _label(fwd_ret: float, up: float, down: float) -> str:
    if fwd_ret >= up:
        return "BUY"
    if fwd_ret <= -down:
        return "SELL"
    return "HOLD"


def label_features(
    feats: pd.DataFrame, horizon: int, up: float, down: float
) -> pd.DataFrame:
    """Attach forward return + 3-class label to a feature table.

    Drops the trailing `horizon` rows (no forward data) and any row with a NaN
    in the model feature columns.
    """
    if feats is None or feats.empty:
        return pd.DataFrame()
    df = feats.copy()
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    df["fwd_ret"] = fwd
    df = df.iloc[:-horizon] if horizon < len(df) else df.iloc[0:0]
    df = df.dropna(subset=["fwd_ret"] + [c for c in FEATURE_COLUMNS if c in df.columns])
    if df.empty:
        return df
    df["label"] = df["fwd_ret"].apply(lambda r: _label(float(r), up, down))
    # Booleans -> ints for the tabular model.
    for c in FEATURE_COLUMNS:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def build_dataset(
    tickers: Iterable[str],
    horizon: int = 20,
    up: float = 0.05,
    down: float = 0.05,
    period: str = "2y",
) -> pd.DataFrame:
    """Build a combined labeled feature table across tickers.

    Returns a DataFrame with FEATURE_COLUMNS + ticker, date, fwd_ret, label.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        data, idx = fetch_history(ticker, period=period)
        feats = build_features(data, idx)
        labeled = label_features(feats, horizon, up, down)
        if labeled.empty:
            continue
        labeled = labeled.copy()
        labeled.insert(0, "ticker", ticker.upper())
        labeled.insert(1, "date", labeled.index.astype(str))
        frames.append(labeled)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- LLM instruction JSONL --------------------------------------------------

_SYSTEM = (
    "You are a quantitative equities analyst. Given a stock's technical "
    "indicators, decide BUY, HOLD, or SELL and justify it in one sentence."
)


def _rationale(tech: dict, label: str, fwd_ret: float) -> str:
    """Build a one-line rationale grounded in the conditions and realized move."""
    dec = quant_signal.compute_signal(tech)
    bull = ", ".join(dec["bull_conditions"]) or "none"
    bear = ", ".join(dec["bear_conditions"]) or "none"
    return (
        f"{label}: bullish [{bull}], bearish [{bear}]; "
        f"RSI {tech['rsi_14']:.1f}, {'above' if tech['above_sma_50'] else 'below'} SMA50."
    )


def _user_prompt(tech: dict) -> str:
    fields = {
        "rsi_14": round(tech["rsi_14"], 2),
        "last_close": round(tech["last_close"], 2),
        "sma_50": round(tech["sma_50"], 2),
        "sma_200": tech["sma_200"],
        "above_sma_50": tech["above_sma_50"],
        "ema_10": round(tech["ema_10"], 2),
        "ema_20": round(tech["ema_20"], 2),
        "ema_50": round(tech["ema_50"], 2),
        "rs_high": tech["rs_high"],
        "buy_signal": tech["buy_signal"],
        "rs_value": round(tech["rs_value"], 2),
    }
    return "Technicals:\n" + json.dumps(fields, indent=2) + "\n\nDecide BUY, HOLD, or SELL with a one-sentence reason."


def to_chat_records(labeled: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a labeled feature table into chat-format SFT records.

    Each record: {"messages": [system, user, assistant]} — the format trl's
    SFTTrainer consumes with a chat template (Qwen/Llama).
    """
    records: list[dict[str, Any]] = []
    for _, row in labeled.iterrows():
        tech = row_to_tech(row)
        label = str(row["label"])
        rationale = _rationale(tech, label, float(row["fwd_ret"]))
        records.append({
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user_prompt(tech)},
                {"role": "assistant", "content": rationale},
            ]
        })
    return records


def save_dataset(
    labeled: pd.DataFrame,
    out_dir: str | Path,
    write_jsonl: bool = True,
) -> dict[str, str]:
    """Write the tabular CSV and (optionally) the chat JSONL. Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    csv_path = out / "dataset.csv"
    labeled.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)

    if write_jsonl:
        jsonl_path = out / "sft.jsonl"
        records = to_chat_records(labeled)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        paths["jsonl"] = str(jsonl_path)
    return paths
