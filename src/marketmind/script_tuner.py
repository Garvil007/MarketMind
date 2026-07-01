"""Tune the deterministic script from the paper trader's closed trades.

This is the "script learns too" half of the learning loop (the other half is
ml_model retraining). It reads paper_trade_outcomes, replays each closed BUY's
entry features through quant_signal.compute_signal to recover WHICH bull
conditions justified the entry, then scores every condition by its realized
win rate:

    weight = clamp(win_rate / 0.5, 0.25, 1.25)     (n >= min_condition_n)

A condition that kept appearing in losing entries (label SELL) sinks below 1.0
and drags future BUYs it supports under the min_buy_score gate; a condition
that keeps winning is restored/boosted. min_buy_score itself rises when the
account's overall win rate is poor, demanding more confirming evidence per BUY.

The tuning is recomputed from scratch on every call (idempotent — no drift),
bounded, and written to data/models/script_params.json, which
quant_signal.compute_signal reads on every decision.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from marketmind import quant_signal
from marketmind.portfolio_db import connect

log = logging.getLogger("marketmind.script_tuner")

# Bounds for the learned knobs — the script can be dampened, never disabled.
WEIGHT_MIN = 0.25
WEIGHT_MAX = 1.25
SCORE_TIERS = (            # (overall win rate below, min_buy_score)
    (0.45, 2.0),           # losing account -> demand 2+ strong conditions
    (0.55, 1.5),           # mediocre -> demand a bit more confluence
)
DEFAULT_MIN_BUY_SCORE = 1.0


def _outcome_rows(account: str) -> list[dict[str, Any]]:
    """Closed-trade outcomes (label + entry features) for one account."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT ticker, realized_return, label, features_json "
            "FROM paper_trade_outcomes WHERE account_id = ? ORDER BY id;", (account,)
        )
        return [dict(r) for r in cur.fetchall()]


def _entry_conditions(features_json: str) -> list[str]:
    """Recover the bull conditions that were true at entry from stored features.

    Condition membership in compute_signal does not depend on the learned
    weights, so replaying through it is stable regardless of current tuning.
    """
    try:
        tech = json.loads(features_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return []
    if not tech:
        return []
    return quant_signal.compute_signal(tech)["bull_conditions"]


def tune_from_trades(
    account: str = "claude",
    min_outcomes: int = 10,
    min_condition_n: int = 5,
) -> dict[str, Any]:
    """Recompute script condition weights + BUY gate from closed-trade win rates.

    Args:
        account: paper account whose outcomes drive the tuning.
        min_outcomes: minimum decisive (win/loss) closed trades before tuning.
        min_condition_n: minimum decisive trades a condition must appear in
            before its weight moves off 1.0.

    Returns:
        Status dict: {"status", "n_outcomes", "overall_win_rate",
        "condition_stats", "params", "params_path"} — or a "skipped" status
        with the reason when there isn't enough data yet.
    """
    rows = _outcome_rows(account)

    # Decisive outcomes only: label BUY = the entry won, SELL = it lost.
    # HOLD (small move either way) says little about the entry conditions.
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    n_win = n_loss = 0
    for r in rows:
        label = r["label"]
        if label not in ("BUY", "SELL"):
            continue
        is_win = label == "BUY"
        n_win += int(is_win)
        n_loss += int(not is_win)
        for cond in _entry_conditions(r["features_json"]):
            if is_win:
                wins[cond] = wins.get(cond, 0) + 1
            else:
                losses[cond] = losses.get(cond, 0) + 1

    n_decisive = n_win + n_loss
    if n_decisive < min_outcomes:
        return {"status": "skipped",
                "reason": f"only {n_decisive} decisive closed trades (need {min_outcomes})",
                "n_outcomes": len(rows)}

    # Per-condition weight from its win rate (recomputed fresh — idempotent).
    weights = dict(quant_signal.DEFAULT_PARAMS["condition_weights"])
    condition_stats: dict[str, dict[str, Any]] = {}
    for cond in weights:
        w, l = wins.get(cond, 0), losses.get(cond, 0)
        n = w + l
        if n >= min_condition_n:
            win_rate = w / n
            weights[cond] = round(min(WEIGHT_MAX, max(WEIGHT_MIN, win_rate / 0.5)), 2)
        else:
            win_rate = None
        condition_stats[cond] = {"wins": w, "losses": l, "win_rate": win_rate,
                                 "weight": weights[cond]}

    # Overall gate: a losing book must demand more confluence per BUY.
    overall = n_win / n_decisive
    min_buy_score = DEFAULT_MIN_BUY_SCORE
    for below, score in SCORE_TIERS:
        if overall < below:
            min_buy_score = score
            break

    params = {
        "condition_weights": weights,
        "min_buy_score": min_buy_score,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tuned_from": {"account": account, "n_decisive": n_decisive,
                       "overall_win_rate": round(overall, 4)},
    }
    path = quant_signal.save_params(params)
    log.info(f"script tuned from {n_decisive} trades (win rate {overall:.0%}) -> {path}")

    return {"status": "tuned", "n_outcomes": len(rows), "n_decisive": n_decisive,
            "overall_win_rate": round(overall, 4), "condition_stats": condition_stats,
            "params": params, "params_path": path}
