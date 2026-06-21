"""Train the tabular ML classifier on the built dataset.

Reads data/training/dataset.csv, trains a gradient-boosted classifier with a
time-ordered split, prints metrics, and saves data/models/quant_clf.joblib.

Run:
  python scripts/build_dataset.py            # first
  python scripts/train_ml.py
  python scripts/train_ml.py --csv data/training/dataset.csv --out data/models/quant_clf.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from marketmind.backtest.ml_model import DEFAULT_MODEL_PATH, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train the tabular quant classifier.")
    p.add_argument("--csv", default="data/training/dataset.csv")
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    if not Path(args.csv).exists():
        print(f"Dataset not found: {args.csv}\nRun: python scripts/build_dataset.py")
        return

    df = pd.read_csv(args.csv)
    metrics = train(df, model_path=args.out, test_frac=args.test_frac)

    print(f"Trained on {metrics['n_train']} rows, tested on {metrics['n_test']}.")
    if "accuracy" in metrics:
        print(f"Test accuracy: {metrics['accuracy']}\n")
        print(metrics["report"])
    print("\nTop features:")
    for feat, imp in list(metrics["feature_importances"].items())[:8]:
        print(f"  {feat:<18}{imp}")
    print(f"\nSaved model -> {metrics['model_path']}")


if __name__ == "__main__":
    main()
