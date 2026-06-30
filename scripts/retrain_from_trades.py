"""Retrain the quant ML model on Claude Trader's own closed trades.

Closes the learning loop: every SELL logs the trade's entry features + realized
return as a labeled example (paper_trade_outcomes). This script merges those with
the historical backtest dataset and retrains data/models/quant_clf.joblib, so the
model improves from its own mistakes.

Usage:
    python scripts/retrain_from_trades.py                 # default account "claude"
    python scripts/retrain_from_trades.py --account claude --min 5
    python scripts/retrain_from_trades.py --no-base       # train on paper trades only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marketmind import paper_trader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain ML model from paper-trade outcomes.")
    ap.add_argument("--account", default=paper_trader.ACCOUNT_ID)
    ap.add_argument("--min", type=int, default=10, help="min closed trades required")
    ap.add_argument("--base", default="data/training/dataset.csv",
                    help="historical dataset CSV to merge with (default: %(default)s)")
    ap.add_argument("--no-base", action="store_true", help="ignore the base dataset")
    args = ap.parse_args()

    n = paper_trader.outcomes_count(args.account)
    print(f"Closed-trade outcomes logged for '{args.account}': {n}")

    result = paper_trader.retrain_from_trades(
        account=args.account,
        base_dataset=None if args.no_base else args.base,
        min_outcomes=args.min,
    )

    if result["status"] == "skipped":
        print(f"Retrain skipped — {result['reason']}.")
        print("Run the trader more (let positions flip to SELL) to accumulate outcomes.")
        return 0

    print(f"Retrained on {result['n_total']} rows "
          f"({result['n_new']} paper trades + {result['n_base']} historical).")
    if "accuracy" in result:
        print(f"Holdout accuracy: {result['accuracy']}")
    print(f"Model saved: {result.get('model_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
