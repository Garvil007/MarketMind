"""Tabular ML model: predict BUY/HOLD/SELL from scanner features.

Trains a gradient-boosted classifier on the labeled dataset (dataset.py) with a
time-ordered train/test split (no shuffling — financial data leaks if shuffled).
Saves a joblib artifact that the quant pipeline can load to get a second,
data-driven opinion alongside the deterministic script signal.

scikit-learn + joblib are the only extra deps and live in requirements-train.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marketmind.backtest.dataset import FEATURE_COLUMNS, LABELS

DEFAULT_MODEL_PATH = Path("data/models/quant_clf.joblib")


def _split_time_ordered(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by row order so the test set is strictly later than train (if date-sorted)."""
    if "date" in df.columns:
        df = df.sort_values("date")
    cut = int(len(df) * (1.0 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def train(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_frac: float = 0.2,
) -> dict[str, Any]:
    """Train the classifier on a labeled dataset and persist it.

    Args:
        df: output of dataset.build_dataset (must contain FEATURE_COLUMNS + label).
        model_path: where to write the joblib artifact.
        test_frac: fraction of (time-ordered) rows held out for evaluation.

    Returns:
        metrics dict: accuracy, per-class report, feature importances, n rows.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report

    if df is None or df.empty:
        raise ValueError("Empty dataset — build it first (scripts/build_dataset.py).")

    # Features added after a dataset was built (e.g. news_sentiment, which has
    # no historical source) are backfilled as neutral 0.0 so old CSVs stay valid.
    df = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0

    if "label" not in df.columns:
        raise ValueError("Dataset missing 'label' column.")

    train_df, test_df = _split_time_ordered(df, test_frac)
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    metrics: dict[str, Any] = {"n_train": len(train_df), "n_test": len(test_df)}
    if len(test_df) > 0:
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy()
        preds = clf.predict(X_test)
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["report"] = classification_report(y_test, preds, zero_division=0, output_dict=False)

    metrics["feature_importances"] = dict(
        sorted(
            zip(FEATURE_COLUMNS, (round(float(v), 4) for v in clf.feature_importances_)),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_COLUMNS, "labels": list(clf.classes_)}, path)
    metrics["model_path"] = str(path)
    return metrics


def _tech_to_vector(tech: dict) -> np.ndarray:
    """Map a get_technicals dict to the model's feature vector order.

    Condition columns (cond1..6, buy_signal, rs_high) come from the tech dict
    where present; missing booleans default to 0.
    """
    row = {
        "rsi_14": tech.get("rsi_14", 0.0),
        "sma_50": tech.get("sma_50", 0.0),
        "sma_200": tech.get("sma_200") or 0.0,
        "ema_10": tech.get("ema_10", 0.0),
        "ema_20": tech.get("ema_20", 0.0),
        "ema_50": tech.get("ema_50", 0.0),
        "pct_from_sma_50": tech.get("pct_from_sma_50", 0.0),
        "plus_di": tech.get("plus_di", 0.0),
        "weekly_rsi": tech.get("weekly_rsi", 0.0),
        "rs_value": tech.get("rs_value", 0.0),
        "above_sma_50": int(bool(tech.get("above_sma_50"))),
        "rs_high": int(bool(tech.get("rs_high"))),
        "buy_signal": int(bool(tech.get("buy_signal"))),
        "news_sentiment": tech.get("news_sentiment", 0.0) or 0.0,
    }
    for c in ("cond1", "cond2", "cond3", "cond4", "cond5", "cond6"):
        row[c] = int(bool(tech.get(c)))
    return np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)


def load(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the persisted model bundle, or None if it hasn't been trained yet.

    Returns None (not an error) when the artifact is missing OR when the training
    deps (joblib/scikit-learn) aren't installed — the ML vote is advisory, so the
    caller falls back to the deterministic script signal.
    """
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None
    # Safe: artifact is self-generated locally by train() below, not an external source.
    return joblib.load(path)


def predict_from_tech(tech: dict, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    """Predict {signal, confidence, proba} from a get_technicals dict.

    Returns None if no model artifact exists yet (caller falls back to the
    deterministic script signal).
    """
    bundle = load(model_path)
    if bundle is None:
        return None
    clf = bundle["model"]
    # Artifact trained before a feature was added (e.g. news_sentiment) can't
    # score the wider vector — treat as "no model" until retrained.
    if list(bundle.get("features", [])) != list(FEATURE_COLUMNS):
        return None
    vec = _tech_to_vector(tech)
    proba = clf.predict_proba(vec)[0]
    classes = list(clf.classes_)
    best = int(np.argmax(proba))
    return {
        "signal": classes[best],
        "confidence": round(float(proba[best]), 4),
        "proba": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }
