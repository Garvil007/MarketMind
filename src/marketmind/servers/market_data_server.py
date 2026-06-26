"""Market Data MCP server (FastMCP, Streamable HTTP, port 8001, /mcp).

Exposes two read-only price tools backed by yfinance: get_ohlcv and
get_technicals. No auth, no caching. See CLAUDE.md for the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from fastmcp import FastMCP

from marketmind import scanner
from marketmind.scanner import (
    INDEX_SYMBOL_USA,
    _safe_array,
    scan_with_data,
)

mcp = FastMCP("market-data")


def _history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch a yfinance OHLCV DataFrame. Empty frame means unknown ticker."""
    return yf.Ticker(ticker).history(period=period, interval=interval)


@mcp.tool
def get_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> dict:
    """Fetch historical OHLCV candles for a stock ticker.

    Args:
        ticker: Stock symbol, e.g. "NVDA" (US equities).
        period: Lookback window. One of yfinance periods:
            "1mo", "3mo", "6mo", "1y", "2y", "5y", "max". Default "6mo".
        interval: Candle size: "1d", "1wk", "1mo". Default "1d".

    Returns:
        On success:
            {"ticker": str,
             "rows": [{"date": "YYYY-MM-DD", "open": float, "high": float,
                       "low": float, "close": float, "volume": int}, ...]}
        On unknown ticker or failure:
            {"error": str}
    """
    try:
        df = _history(ticker, period, interval)
        if df is None or df.empty:
            return {"error": f"No data for ticker '{ticker}' (period={period}, interval={interval})."}
        rows = []
        for ts, row in df.iterrows():
            rows.append({
                "date": ts.date().isoformat(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return {"ticker": ticker.upper(), "rows": rows}
    except Exception as exc:  # noqa: BLE001 - surface as data, never raise to the agent
        return {"error": f"get_ohlcv failed for '{ticker}': {exc}"}


@mcp.tool
def get_technicals(ticker: str) -> dict:
    """Compute trend/momentum technicals for a stock ticker using TA-Lib.

    Fetches 2y of daily data (same window as the scanner) so SMA-200, RS, and
    the six-condition buy signal are all well-defined. Embeds scanner outputs
    (rs_high, buy_signal) so the Quant agent needs only one tool call for the
    full picture.

    Args:
        ticker: Stock symbol, e.g. "NVDA" (US equities).

    Returns:
        On success:
            {"ticker": str,
             "rsi_14": float,          # TA-Lib RSI(14) on daily closes
             "sma_50": float,          # TA-Lib SMA(50)
             "sma_200": float | null,  # TA-Lib SMA(200), null if < 200 rows
             "ema_10": float,          # TA-Lib EMA(10)
             "ema_20": float,          # TA-Lib EMA(20)
             "ema_50": float,          # TA-Lib EMA(50)
             "last_close": float,
             "above_sma_50": bool,
             "pct_from_sma_50": float,
             "rs_high": bool,          # RS vs S&P 500 at a ~6-month new high
             "buy_signal": bool,       # all six scanner conditions true
             "rs_value": float}        # raw RS score vs S&P 500
        On failure:
            {"error": str}
    """
    try:
        data_2y = yf.Ticker(ticker).history(period="2y", interval="1d")
        idx_2y = yf.Ticker(INDEX_SYMBOL_USA).history(period="2y", interval="1d")

        if data_2y is None or data_2y.empty:
            return {"error": f"No data for ticker '{ticker}'."}

        close_series = data_2y["Close"].dropna()
        if len(close_series) < 50:
            return {"error": f"Not enough history for '{ticker}' ({len(close_series)} rows)."}

        close = _safe_array(close_series)

        rsi_14_arr = talib.RSI(close, timeperiod=14)
        sma_50_arr = talib.SMA(close, timeperiod=50)
        sma_200_arr = talib.SMA(close, timeperiod=200)
        ema_10_arr = talib.EMA(close, timeperiod=10)
        ema_20_arr = talib.EMA(close, timeperiod=20)
        ema_50_arr = talib.EMA(close, timeperiod=50)

        last_close = round(float(close[-1]), 2)
        sma_50 = round(float(sma_50_arr[-1]), 2)
        sma_200 = None if np.isnan(sma_200_arr[-1]) else round(float(sma_200_arr[-1]), 2)
        above_sma_50 = last_close > sma_50
        pct_from_sma_50 = round((last_close - sma_50) / sma_50 * 100.0, 2) if sma_50 else 0.0

        scan = scan_with_data(ticker.upper(), "usa", data_2y, idx_2y)
        conds = scan.get("conditions", {})

        return {
            "ticker": ticker.upper(),
            "rsi_14": round(float(rsi_14_arr[-1]), 2),
            "sma_50": sma_50,
            "sma_200": sma_200,
            "ema_10": round(float(ema_10_arr[-1]), 2),
            "ema_20": round(float(ema_20_arr[-1]), 2),
            "ema_50": round(float(ema_50_arr[-1]), 2),
            "last_close": last_close,
            "above_sma_50": bool(above_sma_50),
            "pct_from_sma_50": pct_from_sma_50,
            "rs_high": scan["rs_high"],
            "buy_signal": scan["buy_signal"],
            "rs_value": scan["details"]["rs_value"],
            # Individual scanner conditions, so the ML model sees the same
            # features live as it was trained on (backtest.dataset.FEATURE_COLUMNS).
            "plus_di": conds.get("plus_di", 0.0),
            "weekly_rsi": conds.get("weekly_rsi", 0.0),
            "cond1": conds.get("cond1", False),
            "cond2": conds.get("cond2", False),
            "cond3": conds.get("cond3", False),
            "cond4": conds.get("cond4", False),
            "cond5": conds.get("cond5", False),
            "cond6": conds.get("cond6", False),
        }
    except Exception as exc:  # noqa: BLE001 - surface as data, never raise to the agent
        return {"error": f"get_technicals failed for '{ticker}': {exc}"}


@mcp.tool
def scan_signals(ticker: str) -> dict:
    """Run a personal RS-high + buy-signal momentum scan on a US stock ticker.

    Fetches ~2y of daily data for the ticker and the S&P 500 benchmark, then:
      - rs_high: True if relative strength vs the S&P 500 just made a new
        ~6-month high (123-bar rolling max).
      - buy_signal: True only if ALL six conditions hold: +DI(5) rising >=10,
        last-bar dollar volume above threshold, EMA10>EMA20>EMA50, weekly
        RSI(14) >= 59, 50-day SMA rising over 5 bars, and last volume above its
        20-day SMA.

    Args:
        ticker: US stock symbol, e.g. "NVDA".

    Returns:
        {"symbol": str, "market": "usa",
         "rs_high": bool, "buy_signal": bool,
         "details": {"symbol", "price", "change_pct", "volume", "avg_volume",
                     "volume_ratio", "rs_value"},
         "error": str | null}
    """
    return scanner.scan_signals(ticker)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
