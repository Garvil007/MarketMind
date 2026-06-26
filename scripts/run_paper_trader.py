"""Run the Claude paper-trading agent once (headless), then print the portfolio.

Fetches fresh prices for the watchlist, decides + executes virtual trades against
the $500 account, and prints the marked-to-market portfolio. No servers needed.

Run:
  python scripts/run_paper_trader.py
  python scripts/run_paper_trader.py NVDA AAPL MSFT
  python scripts/run_paper_trader.py --reset --cash 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind import paper_trader  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Claude paper-trading agent.")
    p.add_argument("tickers", nargs="*", help="watchlist (default: built-in list)")
    p.add_argument("--reset", action="store_true", help="reset the account before trading")
    p.add_argument("--cash", type=float, default=paper_trader.STARTING_CASH, help="starting cash")
    args = p.parse_args()

    if args.reset:
        paper_trader.reset_paper(starting_cash=args.cash)
        print(f"Account reset to ${args.cash:.2f}.")

    watchlist = args.tickers or None
    run = paper_trader.decide_and_trade(watchlist=watchlist)

    print("\n=== Trades this run ===")
    if run["trades"]:
        for tr in run["trades"]:
            print(f"  {tr['side']:<4} {tr['ticker']:<6} {tr['shares']:>10.4f} @ ${tr['price']:.2f}")
    else:
        print("  (no trades)")
    if run["skipped"]:
        print(f"  skipped (no data): {', '.join(run['skipped'])}")

    s = paper_trader.portfolio_summary()
    print("\n=== Claude portfolio ===")
    print(f"  Starting:    ${s['starting_cash']:.2f}")
    print(f"  Cash:        ${s['cash']:.2f}")
    print(f"  Positions:   ${s['positions_value']:.2f}")
    print(f"  TOTAL VALUE: ${s['total_value']:.2f}   "
          f"({s['total_return_pct']:+.2f}%, P&L ${s['total_pnl']:+.2f})")
    if s["positions"]:
        print(f"\n  {'Ticker':<8}{'Shares':>10}{'AvgCost':>10}{'Last':>10}{'MktVal':>10}{'Unrl%':>8}{'Sig':>6}")
        for r in s["positions"]:
            print(f"  {r['ticker']:<8}{r['shares']:>10.4f}{r['avg_cost']:>10.2f}"
                  f"{r['last']:>10.2f}{r['market_value']:>10.2f}{r['unrealized_pct']:>8.2f}{r['signal']:>6}")


if __name__ == "__main__":
    main()
