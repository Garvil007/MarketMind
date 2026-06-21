"""Backtest the deterministic script signal over one or more US tickers.

Fetches ~2y daily history, builds the scanner feature table, simulates trades,
and prints per-ticker + aggregate metrics. No servers required (offline, yfinance).

Run:
  python scripts/run_backtest.py NVDA AAPL MSFT
  python scripts/run_backtest.py --entry buy_signal --hold 15 --stop 0.07 NVDA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from marketmind.backtest.features import build_features, fetch_history  # noqa: E402

DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "AMD", "TSLA"]


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the script signal.")
    p.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS, help="US symbols")
    p.add_argument("--entry", choices=["script", "buy_signal"], default="script")
    p.add_argument("--hold", type=int, default=20)
    p.add_argument("--stop", type=float, default=0.08)
    p.add_argument("--tp", type=float, default=0.20)
    p.add_argument("--period", default="2y")
    args = p.parse_args()

    cfg = BacktestConfig(entry=args.entry, hold_days=args.hold, stop_loss=args.stop, take_profit=args.tp)
    tickers = args.tickers or DEFAULT_TICKERS

    all_returns: list[float] = []
    print(f"Backtest  entry={cfg.entry}  hold={cfg.hold_days}  stop={cfg.stop_loss}  tp={cfg.take_profit}\n")
    print(f"{'Ticker':<8}{'Trades':>7}{'Win%':>8}{'Avg%':>8}{'Total%':>9}{'MaxDD%':>9}{'B&H%':>9}")
    for ticker in tickers:
        data, idx = fetch_history(ticker, period=args.period)
        feats = build_features(data, idx)
        if feats.empty:
            print(f"{ticker:<8}{'(no data)':>7}")
            continue
        res = run_backtest(feats, cfg)
        m = res["metrics"]
        all_returns.extend(t.ret for t in res["trades"])
        print(f"{ticker:<8}{m['n_trades']:>7}{m['win_rate']*100:>8.1f}"
              f"{m['avg_return']*100:>8.1f}{m['total_return']*100:>9.1f}"
              f"{m['max_drawdown']*100:>9.1f}{m['buy_hold_return']*100:>9.1f}")

    if all_returns:
        import numpy as np
        r = np.array(all_returns)
        print(f"\nAggregate: {len(r)} trades  win={float((r>0).mean())*100:.1f}%  "
              f"avg={float(r.mean())*100:.2f}%  median={float(np.median(r))*100:.2f}%")


if __name__ == "__main__":
    main()
