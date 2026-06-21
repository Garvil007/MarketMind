"""Build training data (tabular CSV + chat JSONL) from historical bars.

For each ticker: fetch ~2y history, build scanner features, label each bar by its
forward return, and write data/training/dataset.csv (for the ML model) and
data/training/sft.jsonl (for LoRA/QLoRA fine-tuning).

Run:
  python scripts/build_dataset.py NVDA AAPL MSFT AMD TSLA
  python scripts/build_dataset.py --horizon 20 --up 0.05 --down 0.05 --out data/training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.backtest.dataset import build_dataset, save_dataset  # noqa: E402

DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "GOOGL", "AMZN", "META"]


def main() -> None:
    p = argparse.ArgumentParser(description="Build ML + LLM training data.")
    p.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS, help="US symbols")
    p.add_argument("--horizon", type=int, default=20, help="forward-return horizon in bars")
    p.add_argument("--up", type=float, default=0.05, help="BUY threshold (forward return)")
    p.add_argument("--down", type=float, default=0.05, help="SELL threshold (forward return)")
    p.add_argument("--period", default="2y")
    p.add_argument("--out", default="data/training")
    p.add_argument("--no-jsonl", action="store_true", help="skip the LLM JSONL")
    args = p.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    print(f"Building dataset for {len(tickers)} tickers (horizon={args.horizon}, up={args.up}, down={args.down})...")
    df = build_dataset(tickers, horizon=args.horizon, up=args.up, down=args.down, period=args.period)
    if df.empty:
        print("No data produced (rate-limited or bad symbols?).")
        return

    counts = df["label"].value_counts().to_dict()
    print(f"Rows: {len(df)}   Label balance: {counts}")
    paths = save_dataset(df, args.out, write_jsonl=not args.no_jsonl)
    for k, v in paths.items():
        print(f"  wrote {k}: {v}")


if __name__ == "__main__":
    main()
