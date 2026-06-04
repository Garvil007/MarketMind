"""Market Data MCP server (FastMCP, Streamable HTTP, port 8001, /mcp).

Exposes two read-only price tools backed by yfinance: get_ohlcv and
get_technicals. No auth, no caching. See CLAUDE.md for the contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from fastmcp import FastMCP

mcp = FastMCP("market-data")


def _history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch a yfinance OHLCV DataFrame. Empty frame means unknown ticker."""
    return yf.Ticker(ticker).history(period=period, interval=interval)


def _rsi(close: pd.Series, window: int = 14) -> float:
    """Wilder-style RSI over `close`. Returns the most recent value."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])


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
    """Compute trend/momentum technicals for a stock ticker.

    Uses ~1 year of daily closes so the 200-day SMA is well defined.

    Args:
        ticker: Stock symbol, e.g. "NVDA" (US equities).

    Returns:
        On success:
            {"ticker": str,
             "rsi_14": float,          # 14-period RSI, 0-100
             "sma_50": float,          # 50-day simple moving average
             "sma_200": float,         # 200-day simple moving average (null if too few rows)
             "last_close": float,
             "above_sma_50": bool,     # last_close > sma_50
             "pct_from_sma_50": float} # percent distance of last_close from sma_50
        On failure:
            {"error": str}
    """
    try:
        df = _history(ticker, period="1y", interval="1d")
        if df is None or df.empty:
            return {"error": f"No data for ticker '{ticker}'."}
        close = df["Close"].dropna()
        if len(close) < 50:
            return {"error": f"Not enough history for '{ticker}' to compute technicals ({len(close)} rows)."}

        last_close = float(close.iloc[-1])
        sma_50 = float(close.rolling(window=50).mean().iloc[-1])
        sma_200_raw = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else np.nan
        sma_200 = None if pd.isna(sma_200_raw) else round(float(sma_200_raw), 2)
        rsi_14 = _rsi(close, 14)
        above_sma_50 = last_close > sma_50
        pct_from_sma_50 = (last_close - sma_50) / sma_50 * 100.0 if sma_50 else 0.0

        return {
            "ticker": ticker.upper(),
            "rsi_14": round(rsi_14, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": sma_200,
            "last_close": round(last_close, 2),
            "above_sma_50": bool(above_sma_50),
            "pct_from_sma_50": round(pct_from_sma_50, 2),
        }
    except Exception as exc:  # noqa: BLE001 - surface as data, never raise to the agent
        return {"error": f"get_technicals failed for '{ticker}': {exc}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
