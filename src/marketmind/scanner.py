"""Personal RS-high + buy-signal scanner (USA equities), adapted for MarketMind.

Ported from a Kafka-based batch scanner. Kafka transport and bulk-fetch infra
removed; a single-ticker yfinance fetch added. The TA logic (RS new-high check
and the six-condition buy signal, TA-Lib backed) is preserved verbatim.

USA market only for now. Benchmark = S&P 500 (^GSPC).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("marketmind.scanner")

# ── USA benchmark + thresholds ──────────────────────────────────────────────
INDEX_SYMBOL_USA = "^GSPC"           # S&P 500, RS benchmark
VOL_THRESHOLD_USA = 20_000_000.0     # min dollar volume (last_close * volume) on last bar; tune


def _safe_array(series: pd.Series) -> np.ndarray:
    """Contiguous float64 array for TA-Lib (NaNs preserved)."""
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64))


def _empty_result(symbol: str, market: str, err: Optional[str]) -> Dict[str, Any]:
    return {
        "symbol": symbol, "market": market,
        "rs_high": False, "buy_signal": False,
        "details": {
            "symbol": symbol, "price": 0.0, "change_pct": 0.0,
            "volume": 0, "avg_volume": 0, "volume_ratio": 0.0, "rs_value": 0.0,
        },
        "conditions": {
            "cond1": False, "cond2": False, "cond3": False,
            "cond4": False, "cond5": False, "cond6": False,
            "plus_di": 0.0, "weekly_rsi": 0.0,
        },
        "error": err,
    }


def scan_with_data(symbol: str, market: str,
                   data_2y: Optional[pd.DataFrame],
                   idx_2y: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """RS-high + buy-signal checks given already-fetched OHLCV."""
    import talib

    if data_2y is None or data_2y.empty:
        return _empty_result(symbol, market, "empty OHLCV (rate-limited / delisted / wrong sym)")
    if idx_2y is None or idx_2y.empty:
        return _empty_result(symbol, market, "index data unavailable")

    vol_threshold = VOL_THRESHOLD_USA  # USA only
    result = _empty_result(symbol, market, None)

    try:
        n = len(data_2y)
        data_1y  = data_2y.iloc[max(0, n - 252):]
        data_6mo = data_2y.iloc[max(0, n - 126):]
        n_idx    = len(idx_2y)
        idx_1y   = idx_2y.iloc[max(0, n_idx - 252):]
        weekly_close = data_2y["Close"].resample("W").last().dropna()

        # ── RS new-high ───────────────────────────────────────────────
        try:
            close_1y     = data_1y["Close"]
            idx_close_1y = idx_1y["Close"]
            common = close_1y.index.intersection(idx_close_1y.index)
            if len(common) >= 50:
                sc = close_1y.loc[common]
                ic = idx_close_1y.loc[common]
                rs = (sc * 7 * 1000) / ic
                rs_high = rs.rolling(window=123, min_periods=1).max()
                if rs.iloc[-1] > rs_high.shift(1).iloc[-1]:
                    result["rs_high"] = True
        except Exception as exc:
            log.debug(f"{symbol}: RS — {exc}")

        # ── Buy-signal ────────────────────────────────────────────────
        try:
            idx_6mo = idx_2y.iloc[max(0, n_idx - 126):]
            common_6 = data_6mo.index.intersection(idx_6mo.index)
            df6 = data_6mo.loc[common_6].copy()
            if len(df6) < 60:
                raise ValueError("not enough 6mo data")

            high   = _safe_array(df6["High"])
            low    = _safe_array(df6["Low"])
            close  = _safe_array(df6["Close"])
            volume = _safe_array(df6["Volume"])

            plus_di  = talib.PLUS_DI(high, low, close, timeperiod=5)
            ema10    = talib.EMA(close, timeperiod=10)
            ema20    = talib.EMA(close, timeperiod=20)
            ema50    = talib.EMA(close, timeperiod=50)
            sma50    = talib.SMA(close, timeperiod=50)
            vol_sma20 = talib.SMA(volume, timeperiod=20)

            sma_trend = (
                not np.isnan(sma50[-1]) and len(sma50) >= 5
                and sma50[-1] > sma50[-2] > sma50[-3] > sma50[-4] > sma50[-5]
            )

            weekly_rsi_arr  = talib.RSI(_safe_array(weekly_close), timeperiod=14)
            last_weekly_rsi = float(weekly_rsi_arr[-1]) if len(weekly_rsi_arr) > 0 else 0.0

            di_diff   = float(plus_di[-1] - plus_di[-2]) if len(plus_di) >= 2 else 0.0
            vol_value = float(close[-1]) * float(volume[-1])

            cond1 = di_diff >= 10
            cond2 = vol_value > vol_threshold
            cond3 = (
                not np.isnan(ema10[-1]) and not np.isnan(ema20[-1]) and not np.isnan(ema50[-1])
                and float(ema10[-1]) > float(ema20[-1]) > float(ema50[-1])
            )
            cond4 = last_weekly_rsi >= 59
            cond5 = bool(sma_trend)
            cond6 = (
                not np.isnan(vol_sma20[-1])
                and float(volume[-1]) > float(vol_sma20[-1])
            )

            if all([cond1, cond2, cond3, cond4, cond5, cond6]):
                result["buy_signal"] = True

            # Expose the individual conditions so live features match training.
            result["conditions"] = {
                "cond1": bool(cond1), "cond2": bool(cond2), "cond3": bool(cond3),
                "cond4": bool(cond4), "cond5": bool(cond5), "cond6": bool(cond6),
                "plus_di": round(float(plus_di[-1]), 4) if len(plus_di) else 0.0,
                "weekly_rsi": round(last_weekly_rsi, 4),
            }
        except Exception as exc:
            log.debug(f"{symbol}: buy — {exc}")

        # ── Details ───────────────────────────────────────────────────
        try:
            close_1y     = data_1y["Close"]
            idx_close_1y = idx_1y["Close"]
            latest_price = float(close_1y.iloc[-1])
            prev_close   = float(close_1y.iloc[-2]) if len(close_1y) > 1 else latest_price
            change_pct   = ((latest_price - prev_close) / prev_close * 100) if prev_close else 0.0

            vol_series     = data_1y["Volume"].tail(20)
            latest_volume  = float(data_1y["Volume"].iloc[-1])
            avg_volume     = float(vol_series.mean()) if not vol_series.empty else 0.0

            common = close_1y.index.intersection(idx_close_1y.index)
            rs_val = (
                float((close_1y.loc[common].iloc[-1] * 7 * 1000) / idx_close_1y.loc[common].iloc[-1])
                if len(common) > 0 else 0.0
            )

            result["details"] = {
                "symbol":       symbol,
                "price":        round(latest_price, 2),
                "change_pct":   round(change_pct, 2),
                "volume":       int(latest_volume),
                "avg_volume":   int(avg_volume),
                "volume_ratio": round(latest_volume / avg_volume, 2) if avg_volume else 0.0,
                "rs_value":     round(rs_val, 2),
            }
        except Exception as exc:
            log.debug(f"{symbol}: details — {exc}")

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    return result


def scan_signals(symbol: str) -> Dict[str, Any]:
    """Fetch 2y daily OHLCV for `symbol` and the S&P 500, then run scan_with_data.

    Single-ticker, USA-only entrypoint that replaces the original Kafka batch loop.
    """
    market = "usa"
    try:
        data_2y = yf.Ticker(symbol).history(period="2y", interval="1d")
        idx_2y = yf.Ticker(INDEX_SYMBOL_USA).history(period="2y", interval="1d")
    except Exception as exc:  # noqa: BLE001 - surface as data, never raise to the agent
        return _empty_result(symbol, market, f"fetch failed: {exc}")
    return scan_with_data(symbol, market, data_2y, idx_2y)
