"""Trade simulator + performance metrics for the deterministic script signal.

Walks a feature table (features.build_features) bar by bar, opens a long when the
entry rule fires, and closes on a stop-loss, take-profit, max-hold, or a flip to
SELL. Everything is causal: a decision at bar t uses only data through bar t and
the trade is marked at bar t's close.

The entry rule defaults to "the script (quant_signal.compute_signal) says BUY",
so the backtest measures the user's own scanner logic. Set entry="buy_signal" to
test only the strict six-condition signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from marketmind import quant_signal
from marketmind.backtest.features import row_to_tech


@dataclass
class BacktestConfig:
    entry: str = "script"          # "script" (compute_signal==BUY) | "buy_signal"
    hold_days: int = 20            # max bars to hold
    stop_loss: float = 0.08        # exit if return <= -8%
    take_profit: float = 0.20      # exit if return >= +20%
    exit_on_sell: bool = True      # exit early if the script flips to SELL
    allow_reentry: bool = True     # may open a new trade after closing one


@dataclass
class Trade:
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    bars_held: int
    ret: float
    reason: str


def _entry_signal(row: pd.Series, cfg: BacktestConfig) -> bool:
    if cfg.entry == "buy_signal":
        return bool(row["buy_signal"])
    return quant_signal.compute_signal(row_to_tech(row))["signal"] == "BUY"


def _is_sell(row: pd.Series) -> bool:
    return quant_signal.compute_signal(row_to_tech(row))["signal"] == "SELL"


def run_backtest(features_df: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict[str, Any]:
    """Simulate the strategy over a feature table and return trades + metrics.

    Args:
        features_df: output of features.build_features (one ticker).
        cfg: BacktestConfig; defaults to the script-BUY strategy.

    Returns:
        {"trades": [Trade...], "metrics": {...}, "equity_curve": pd.Series}
    """
    cfg = cfg or BacktestConfig()
    if features_df is None or features_df.empty:
        return {"trades": [], "metrics": _empty_metrics(), "equity_curve": pd.Series(dtype=float)}

    rows = list(features_df.iterrows())
    trades: list[Trade] = []
    i = 0
    n = len(rows)

    while i < n:
        date, row = rows[i]
        if not _entry_signal(row, cfg):
            i += 1
            continue

        entry_price = float(row["close"])
        entry_date = date
        exit_idx = i
        reason = "max_hold"
        for j in range(i + 1, min(i + 1 + cfg.hold_days, n)):
            d2, r2 = rows[j]
            ret = (float(r2["close"]) - entry_price) / entry_price
            exit_idx = j
            if ret <= -cfg.stop_loss:
                reason = "stop_loss"
                break
            if ret >= cfg.take_profit:
                reason = "take_profit"
                break
            if cfg.exit_on_sell and _is_sell(r2):
                reason = "sell_signal"
                break
            reason = "max_hold"
        exit_date, exit_row = rows[exit_idx]
        exit_price = float(exit_row["close"])
        trades.append(Trade(
            entry_date=entry_date, exit_date=exit_date,
            entry_price=round(entry_price, 2), exit_price=round(exit_price, 2),
            bars_held=exit_idx - i,
            ret=round((exit_price - entry_price) / entry_price, 4),
            reason=reason,
        ))
        i = exit_idx + 1 if cfg.allow_reentry else n

    metrics = _metrics(trades, features_df)
    equity = _equity_curve(trades)
    return {"trades": trades, "metrics": metrics, "equity_curve": equity}


def _equity_curve(trades: list[Trade]) -> pd.Series:
    """Compounded equity (start = 1.0) marked at each trade's exit date."""
    if not trades:
        return pd.Series(dtype=float)
    eq = 1.0
    dates, values = [], []
    for t in trades:
        eq *= (1.0 + t.ret)
        dates.append(t.exit_date)
        values.append(eq)
    return pd.Series(values, index=pd.Index(dates, name="date"), name="equity")


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(dd.min())


def _empty_metrics() -> dict[str, Any]:
    return {
        "n_trades": 0, "win_rate": 0.0, "avg_return": 0.0, "total_return": 0.0,
        "max_drawdown": 0.0, "avg_bars_held": 0.0, "sharpe": 0.0, "buy_hold_return": 0.0,
    }


def _metrics(trades: list[Trade], features_df: pd.DataFrame) -> dict[str, Any]:
    if not trades:
        m = _empty_metrics()
        if len(features_df) > 1:
            c = features_df["close"]
            m["buy_hold_return"] = round(float(c.iloc[-1] / c.iloc[0] - 1.0), 4)
        return m

    rets = np.array([t.ret for t in trades], dtype=float)
    eq = _equity_curve(trades)
    c = features_df["close"]
    buy_hold = float(c.iloc[-1] / c.iloc[0] - 1.0) if len(c) > 1 else 0.0
    return {
        "n_trades": len(trades),
        "win_rate": round(float((rets > 0).mean()), 4),
        "avg_return": round(float(rets.mean()), 4),
        "total_return": round(float(eq.iloc[-1] - 1.0), 4),
        "max_drawdown": round(_max_drawdown(eq), 4),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 2),
        "sharpe": round(float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0, 3),
        "buy_hold_return": round(buy_hold, 4),
    }
