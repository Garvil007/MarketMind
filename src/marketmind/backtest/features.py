"""Walk-forward feature table from historical OHLCV.

Computes, for EVERY bar with enough history, the same indicators and six buy
conditions that scanner.py evaluates only on the last bar — so a backtest sees
exactly what the live scanner would have seen on each historical day.

All indicators are causal (use only data up to bar t). Weekly RSI is mapped
as-of the last completed week to avoid intra-week lookahead.

`row_to_tech` reshapes one feature row into the dict that
market_data_server.get_technicals emits, so marketmind.quant_signal.compute_signal
can be applied per bar without modification.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from marketmind.scanner import INDEX_SYMBOL_USA, VOL_THRESHOLD_USA, _safe_array

# Minimum bars before the six-condition signal is considered defined.
MIN_BARS = 60


def _talib():
    import talib  # imported lazily so importing this module never requires TA-Lib
    return talib


def _weekly_rsi_daily(close: pd.Series) -> pd.Series:
    """Weekly RSI(14) reindexed onto the daily index, as-of the last full week."""
    talib = _talib()
    weekly_close = close.resample("W").last().dropna()
    if len(weekly_close) < 15:
        return pd.Series(np.nan, index=close.index)
    wr = talib.RSI(_safe_array(weekly_close), timeperiod=14)
    wr_series = pd.Series(wr, index=weekly_close.index)
    # ffill carries the last week-end value at/<= each daily bar (no lookahead).
    return wr_series.reindex(close.index, method="ffill")


def build_features(data_2y: pd.DataFrame, idx_2y: pd.DataFrame) -> pd.DataFrame:
    """Build a per-bar feature DataFrame for one ticker vs the S&P 500.

    Args:
        data_2y: ~2y daily OHLCV for the ticker (yfinance columns: Open/High/Low/
            Close/Volume, DatetimeIndex).
        idx_2y: ~2y daily OHLCV for the benchmark (^GSPC).

    Returns:
        DataFrame indexed by date with indicator columns, the six buy conditions
        (cond1..cond6), buy_signal, rs_high, and rs_value. Rows without enough
        history are dropped.
    """
    talib = _talib()
    if data_2y is None or data_2y.empty or idx_2y is None or idx_2y.empty:
        return pd.DataFrame()

    df = data_2y.copy()
    # Align ticker and index on common trading days.
    common = df.index.intersection(idx_2y.index)
    df = df.loc[common]
    idx_close = idx_2y["Close"].loc[common]
    if len(df) < MIN_BARS:
        return pd.DataFrame()

    close = _safe_array(df["Close"])
    high = _safe_array(df["High"])
    low = _safe_array(df["Low"])
    volume = _safe_array(df["Volume"])

    rsi_14 = talib.RSI(close, timeperiod=14)
    sma_50 = talib.SMA(close, timeperiod=50)
    sma_200 = talib.SMA(close, timeperiod=200)
    ema_10 = talib.EMA(close, timeperiod=10)
    ema_20 = talib.EMA(close, timeperiod=20)
    ema_50 = talib.EMA(close, timeperiod=50)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=5)
    vol_sma20 = talib.SMA(volume, timeperiod=20)
    weekly_rsi = _weekly_rsi_daily(df["Close"]).to_numpy()

    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["volume"] = volume
    out["rsi_14"] = rsi_14
    out["sma_50"] = sma_50
    out["sma_200"] = sma_200
    out["ema_10"] = ema_10
    out["ema_20"] = ema_20
    out["ema_50"] = ema_50
    out["plus_di"] = plus_di
    out["vol_sma20"] = vol_sma20
    out["weekly_rsi"] = weekly_rsi

    out["above_sma_50"] = out["close"] > out["sma_50"]
    out["pct_from_sma_50"] = (out["close"] - out["sma_50"]) / out["sma_50"] * 100.0

    # Relative strength vs the benchmark + its 123-bar rolling new-high flag.
    rs = (out["close"] * 7 * 1000) / idx_close.to_numpy()
    out["rs_value"] = rs
    rs_roll_max = rs.rolling(window=123, min_periods=1).max()
    out["rs_high"] = rs > rs_roll_max.shift(1)

    # Six buy conditions (mirrors scanner.scan_with_data).
    di_diff = out["plus_di"].diff()
    sma50_rising = (
        out["sma_50"].diff().gt(0)
        & out["sma_50"].diff().shift(1).gt(0)
        & out["sma_50"].diff().shift(2).gt(0)
        & out["sma_50"].diff().shift(3).gt(0)
    )
    out["cond1"] = di_diff >= 10
    out["cond2"] = (out["close"] * out["volume"]) > VOL_THRESHOLD_USA
    out["cond3"] = (out["ema_10"] > out["ema_20"]) & (out["ema_20"] > out["ema_50"])
    out["cond4"] = out["weekly_rsi"] >= 59
    out["cond5"] = sma50_rising
    out["cond6"] = out["volume"] > out["vol_sma20"]
    out["buy_signal"] = (
        out["cond1"] & out["cond2"] & out["cond3"]
        & out["cond4"] & out["cond5"] & out["cond6"]
    )

    # Drop warm-up rows where the core indicators aren't defined yet.
    out = out.dropna(subset=["rsi_14", "sma_50", "ema_50"])
    return out


def row_to_tech(row: pd.Series) -> dict[str, Any]:
    """Reshape one feature row into a get_technicals-style dict for compute_signal."""
    sma_200 = row.get("sma_200")
    return {
        "rsi_14": float(row["rsi_14"]),
        "sma_50": float(row["sma_50"]),
        "sma_200": None if pd.isna(sma_200) else float(sma_200),
        "ema_10": float(row["ema_10"]),
        "ema_20": float(row["ema_20"]),
        "ema_50": float(row["ema_50"]),
        "last_close": float(row["close"]),
        "above_sma_50": bool(row["above_sma_50"]),
        "pct_from_sma_50": float(row["pct_from_sma_50"]),
        "rs_high": bool(row["rs_high"]),
        "buy_signal": bool(row["buy_signal"]),
        "rs_value": float(row["rs_value"]),
    }


def fetch_history(ticker: str, period: str = "2y") -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch (ticker, benchmark) daily OHLCV via yfinance. Returns (None, None) on failure."""
    import yfinance as yf

    try:
        data = yf.Ticker(ticker).history(period=period, interval="1d")
        idx = yf.Ticker(INDEX_SYMBOL_USA).history(period=period, interval="1d")
        return data, idx
    except Exception:  # noqa: BLE001 - caller treats None as "no data"
        return None, None
