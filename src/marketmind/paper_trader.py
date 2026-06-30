"""Virtual ("paper") trading account driven by the script + ML signals.

A self-contained agent that starts with a fixed amount of virtual cash and
trades a small US watchlist offline (yfinance direct — no MCP servers, no Groq).
Each run fetches fresh prices, recomputes the deterministic script signal
(quant_signal.compute_signal) plus the advisory ML vote (ml_model), sells held
names that flip to SELL, buys fresh BUY candidates with available cash, and marks
the book to market. State persists in the same SQLite file as the portfolio under
account_id="claude".

This is VIRTUAL only — no broker, no order execution. Advisory per project scope.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from marketmind import quant_signal
from marketmind.backtest import ml_model
from marketmind.backtest.features import build_features, fetch_history, row_to_tech
from marketmind.portfolio_db import connect

log = logging.getLogger("marketmind.paper_trader")

ACCOUNT_ID = "claude"
STARTING_CASH = 500.0
MAX_POS_FRAC = 0.25          # cap one position at 25% of starting cash
MAX_POSITIONS = 6            # most names held at once
MIN_TRADE_USD = 1.0         # ignore trades smaller than this

# Learning loop: label a closed trade by its realized return. A BUY that gained
# more than UP_THRESH was "right" (label BUY), one that lost more than DOWN_THRESH
# was "wrong" (label SELL — should not have bought); in between is HOLD.
OUTCOME_UP_THRESH = 0.05     # +5% realized -> the BUY was correct
OUTCOME_DOWN_THRESH = -0.05  # -5% realized -> the BUY was a mistake

# How the (retrained) ML vote influences trades. When a model exists, a confident
# ML SELL vetoes a script BUY, and a confirming ML BUY lifts a candidate's rank.
# No model -> these are no-ops, so behavior matches the script-only baseline.
ML_VETO_CONF = 0.60          # ML must be this confident in SELL to veto a script BUY
ML_RANK_WEIGHT = 0.5         # blend weight of ML BUY-confidence into ranking

# Default watchlist — large/liquid US names; override per call/UI.
DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "AMD", "TSLA", "GOOGL", "AMZN", "META"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_account (
    account_id    TEXT PRIMARY KEY,
    cash          REAL NOT NULL,
    starting_cash REAL NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_positions (
    account_id TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    shares     REAL NOT NULL,
    avg_cost   REAL NOT NULL,
    entry_tech TEXT,
    entry_ts   TEXT,
    PRIMARY KEY (account_id, ticker)
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    side       TEXT NOT NULL,
    shares     REAL NOT NULL,
    price      REAL NOT NULL,
    signal     TEXT,
    reason     TEXT,
    ts         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trade_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    realized_return REAL NOT NULL,
    label           TEXT NOT NULL,
    features_json   TEXT NOT NULL,
    entry_ts        TEXT,
    exit_ts         TEXT NOT NULL
);
"""

# Columns added after the original paper_positions shipped; ALTER is a no-op-safe
# migration for databases created before the learning loop existed.
_MIGRATIONS = (
    "ALTER TABLE paper_positions ADD COLUMN entry_tech TEXT;",
    "ALTER TABLE paper_positions ADD COLUMN entry_ts TEXT;",
)


def _migrate(conn) -> None:
    """Add columns missing from pre-existing databases. Idempotent."""
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except Exception:  # column already exists -> sqlite raises OperationalError
            pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- account / persistence --------------------------------------------------

def init_paper(account: str = ACCOUNT_ID, starting_cash: float = STARTING_CASH) -> None:
    """Create tables and seed the account with starting cash if it doesn't exist."""
    with connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        row = conn.execute(
            "SELECT account_id FROM paper_account WHERE account_id = ?;", (account,)
        ).fetchone()
        if row is None:
            now = _now()
            conn.execute(
                "INSERT INTO paper_account (account_id, cash, starting_cash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?);",
                (account, starting_cash, starting_cash, now, now),
            )
        conn.commit()


def reset_paper(account: str = ACCOUNT_ID, starting_cash: float = STARTING_CASH) -> None:
    """Wipe an account's positions/trades and reset cash to starting_cash."""
    with connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        # Positions/trades are wiped on reset, but paper_trade_outcomes is kept on
        # purpose — accumulated learning data survives an account restart.
        conn.execute("DELETE FROM paper_positions WHERE account_id = ?;", (account,))
        conn.execute("DELETE FROM paper_trades WHERE account_id = ?;", (account,))
        now = _now()
        conn.execute(
            "INSERT INTO paper_account (account_id, cash, starting_cash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET cash=excluded.cash, "
            "starting_cash=excluded.starting_cash, updated_at=excluded.updated_at;",
            (account, starting_cash, starting_cash, now, now),
        )
        conn.commit()


def _load_account(conn, account: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT cash, starting_cash FROM paper_account WHERE account_id = ?;", (account,)
    ).fetchone()
    return {"cash": float(row["cash"]), "starting_cash": float(row["starting_cash"])}


def _load_positions(conn, account: str) -> dict[str, dict[str, Any]]:
    cur = conn.execute(
        "SELECT ticker, shares, avg_cost, entry_tech, entry_ts "
        "FROM paper_positions WHERE account_id = ?;", (account,)
    )
    return {r["ticker"]: {"shares": float(r["shares"]), "avg_cost": float(r["avg_cost"]),
                          "entry_tech": r["entry_tech"], "entry_ts": r["entry_ts"]}
            for r in cur.fetchall()}


# --- signals ----------------------------------------------------------------

def _latest_tech(ticker: str) -> Optional[tuple[dict, float]]:
    """(tech dict, last close) from the most recent bar, or None if no data."""
    data, idx = fetch_history(ticker)
    feats = build_features(data, idx)
    if feats is None or feats.empty:
        return None
    row = feats.iloc[-1]
    return row_to_tech(row), float(row["close"])


def evaluate(ticker: str) -> Optional[dict[str, Any]]:
    """Score one ticker: deterministic signal + advisory ML vote + last price."""
    got = _latest_tech(ticker)
    if got is None:
        return None
    tech, last = got
    dec = quant_signal.compute_signal(tech)
    ml = ml_model.predict_from_tech(tech)  # None if no trained model
    return {
        "ticker": ticker.upper(),
        "signal": dec["signal"],
        "confidence": dec["confidence"],
        "last": round(last, 2),
        "reasons": dec["reasons"],
        "ml": ml,
        "tech": tech,  # entry features, stored at BUY so closed trades can be labeled/retrained
    }


# --- learning: outcome capture ----------------------------------------------

def _label_for_return(realized: float) -> str:
    """Map a closed trade's realized return to a supervised label."""
    if realized >= OUTCOME_UP_THRESH:
        return "BUY"
    if realized <= OUTCOME_DOWN_THRESH:
        return "SELL"
    return "HOLD"


def _ml_vetoes_buy(ev: dict[str, Any]) -> bool:
    """True if a confident ML SELL vote should block a script BUY."""
    ml = ev.get("ml")
    return bool(ml and ml.get("signal") == "SELL"
                and float(ml.get("confidence", 0.0)) >= ML_VETO_CONF)


def _rank_score(ev: dict[str, Any]) -> float:
    """BUY ranking score: script confidence, lifted when ML also votes BUY."""
    score = float(ev["confidence"])
    ml = ev.get("ml")
    if ml and ml.get("signal") == "BUY":
        score = (1 - ML_RANK_WEIGHT) * score + ML_RANK_WEIGHT * float(ml.get("confidence", 0.0))
    return score


def _record_outcome(conn, account: str, ticker: str, pos: dict[str, Any], exit_price: float) -> None:
    """Log a closed position's entry features + realized return as a training row.

    This is the feedback step: a BUY that later lost money becomes a SELL-labeled
    example, so retrain_from_trades teaches the model to avoid that setup next time.
    """
    entry = float(pos.get("avg_cost") or 0.0)
    if entry <= 0:
        return
    realized = (exit_price - entry) / entry
    conn.execute(
        "INSERT INTO paper_trade_outcomes "
        "(account_id, ticker, entry_price, exit_price, realized_return, label, "
        " features_json, entry_ts, exit_ts) VALUES (?,?,?,?,?,?,?,?,?);",
        (account, ticker, entry, exit_price, realized, _label_for_return(realized),
         pos.get("entry_tech") or "{}", pos.get("entry_ts"), _now()),
    )


# --- trading ----------------------------------------------------------------

def decide_and_trade(account: str = ACCOUNT_ID, watchlist: Optional[list[str]] = None) -> dict[str, Any]:
    """Fetch fresh signals, sell SELLs, buy BUYs with cash, persist, return run log.

    Returns {"trades": [...], "evaluated": [...], "skipped": [...], "cash": float}.
    """
    init_paper(account)
    tickers = [t.strip().upper() for t in (watchlist or DEFAULT_WATCHLIST) if t.strip()]

    with connect() as conn:
        acct = _load_account(conn, account)
        cash = acct["cash"]
        starting = acct["starting_cash"]
        positions = _load_positions(conn, account)

        evals: dict[str, dict] = {}
        skipped: list[str] = []
        for t in tickers:
            ev = evaluate(t)
            if ev is None:
                skipped.append(t)
            else:
                evals[t] = ev

        trades: list[dict] = []

        # --- SELL: held names that now signal SELL -> liquidate to cash ----
        for t, pos in list(positions.items()):
            ev = evals.get(t)
            if ev and ev["signal"] == "SELL" and pos["shares"] > 0:
                proceeds = pos["shares"] * ev["last"]
                cash += proceeds
                trades.append({"ticker": t, "side": "SELL", "shares": round(pos["shares"], 4),
                               "price": ev["last"], "signal": "SELL",
                               "reason": "; ".join(ev["reasons"])[:200]})
                _record_outcome(conn, account, t, pos, exit_price=ev["last"])
                conn.execute("DELETE FROM paper_positions WHERE account_id=? AND ticker=?;", (account, t))
                del positions[t]

        # --- BUY: fresh BUY candidates, ranked by confidence --------------
        per_cap = starting * MAX_POS_FRAC
        candidates = sorted(
            (ev for t, ev in evals.items()
             if ev["signal"] == "BUY" and t not in positions and not _ml_vetoes_buy(ev)),
            key=_rank_score, reverse=True,
        )
        for ev in candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            budget = min(cash, per_cap)
            if budget < MIN_TRADE_USD or ev["last"] <= 0:
                continue
            shares = budget / ev["last"]
            cost = shares * ev["last"]
            if cost < MIN_TRADE_USD:
                continue
            cash -= cost
            entry_tech = json.dumps(ev.get("tech", {}))
            entry_ts = _now()
            positions[ev["ticker"]] = {"shares": shares, "avg_cost": ev["last"],
                                       "entry_tech": entry_tech, "entry_ts": entry_ts}
            conn.execute(
                "INSERT INTO paper_positions (account_id, ticker, shares, avg_cost, entry_tech, entry_ts) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(account_id, ticker) DO UPDATE SET shares=excluded.shares, "
                "avg_cost=excluded.avg_cost, entry_tech=excluded.entry_tech, entry_ts=excluded.entry_ts;",
                (account, ev["ticker"], shares, ev["last"], entry_tech, entry_ts),
            )
            trades.append({"ticker": ev["ticker"], "side": "BUY", "shares": round(shares, 4),
                           "price": ev["last"], "signal": "BUY",
                           "reason": "; ".join(ev["reasons"])[:200]})

        # --- persist trades + cash ----------------------------------------
        now = _now()
        for tr in trades:
            conn.execute(
                "INSERT INTO paper_trades (account_id, ticker, side, shares, price, signal, reason, ts) "
                "VALUES (?,?,?,?,?,?,?,?);",
                (account, tr["ticker"], tr["side"], tr["shares"], tr["price"],
                 tr["signal"], tr["reason"], now),
            )
        conn.execute("UPDATE paper_account SET cash=?, updated_at=? WHERE account_id=?;",
                     (cash, now, account))
        conn.commit()

    return {
        "trades": trades,
        "evaluated": sorted(evals.values(), key=lambda e: e["confidence"], reverse=True),
        "skipped": skipped,
        "cash": round(cash, 2),
    }


# --- valuation / reporting --------------------------------------------------

def portfolio_summary(account: str = ACCOUNT_ID) -> dict[str, Any]:
    """Mark the book to market and return a full portfolio report.

    Returns total_value, cash, total_return_pct, realized/unrealized P&L, and a
    per-position table with the current signal.
    """
    init_paper(account)
    with connect() as conn:
        acct = _load_account(conn, account)
        positions = _load_positions(conn, account)

    cash = acct["cash"]
    starting = acct["starting_cash"]
    rows: list[dict] = []
    positions_value = 0.0
    unrealized = 0.0
    for t, pos in positions.items():
        ev = evaluate(t)
        last = ev["last"] if ev else pos["avg_cost"]
        mkt = pos["shares"] * last
        pnl = (last - pos["avg_cost"]) * pos["shares"]
        positions_value += mkt
        unrealized += pnl
        rows.append({
            "ticker": t,
            "shares": round(pos["shares"], 4),
            "avg_cost": round(pos["avg_cost"], 2),
            "last": round(last, 2),
            "market_value": round(mkt, 2),
            "unrealized_pct": round((last - pos["avg_cost"]) / pos["avg_cost"] * 100, 2) if pos["avg_cost"] else 0.0,
            "signal": ev["signal"] if ev else "n/a",
        })

    total_value = cash + positions_value
    total_pnl = total_value - starting
    for r in rows:
        r["weight_pct"] = round(r["market_value"] / total_value * 100, 2) if total_value else 0.0

    return {
        "account_id": account,
        "starting_cash": round(starting, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / starting * 100, 2) if starting else 0.0,
        "unrealized_pnl": round(unrealized, 2),
        "realized_pnl": round(total_pnl - unrealized, 2),
        "positions": sorted(rows, key=lambda r: r["market_value"], reverse=True),
    }


def recent_trades(account: str = ACCOUNT_ID, limit: int = 25) -> list[dict[str, Any]]:
    """Most recent trades for the account, newest first."""
    init_paper(account)
    with connect() as conn:
        cur = conn.execute(
            "SELECT ticker, side, shares, price, signal, ts FROM paper_trades "
            "WHERE account_id = ? ORDER BY id DESC LIMIT ?;",
            (account, limit),
        )
        return [dict(r) for r in cur.fetchall()]


# --- learning: retrain from the account's own closed trades ------------------

def outcomes_count(account: str = ACCOUNT_ID) -> int:
    """How many labeled closed-trade outcomes have accumulated for retraining."""
    init_paper(account)
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM paper_trade_outcomes WHERE account_id = ?;", (account,)
        ).fetchone()
        return int(row["n"])


def build_outcomes_df(account: str = ACCOUNT_ID):
    """Turn logged closed-trade outcomes into a model-ready DataFrame.

    Each row is the trade's ENTRY features (same columns the ML model trains on)
    plus the realized-return label. Returns an empty DataFrame if nothing closed yet.
    """
    import pandas as pd

    from marketmind.backtest.dataset import FEATURE_COLUMNS

    init_paper(account)
    with connect() as conn:
        cur = conn.execute(
            "SELECT ticker, realized_return, label, features_json, exit_ts "
            "FROM paper_trade_outcomes WHERE account_id = ? ORDER BY id;", (account,)
        )
        raw = [dict(r) for r in cur.fetchall()]

    records: list[dict[str, Any]] = []
    for r in raw:
        try:
            tech = json.loads(r["features_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not tech:
            continue
        vec = ml_model._tech_to_vector(tech)[0]  # FEATURE_COLUMNS order
        row = {col: float(v) for col, v in zip(FEATURE_COLUMNS, vec)}
        row["label"] = r["label"]
        row["date"] = r["exit_ts"]  # lets ml_model time-order the split
        records.append(row)

    return pd.DataFrame(records, columns=[*FEATURE_COLUMNS, "label", "date"])


def retrain_from_trades(
    account: str = ACCOUNT_ID,
    base_dataset: str | None = "data/training/dataset.csv",
    model_path: str | None = None,
    min_outcomes: int = 10,
) -> dict[str, Any]:
    """Retrain the ML model on the account's closed trades (+ the base dataset).

    The new closed-trade outcomes are MERGED with the historical backtest dataset
    (if present) so the model learns from BOTH history and its own mistakes —
    "new takes are taken into account" on every retrain. Returns a status dict.
    """
    import pandas as pd

    df_new = build_outcomes_df(account)
    n_new = len(df_new)
    if n_new < min_outcomes:
        return {"status": "skipped", "reason": f"only {n_new} closed trades (need {min_outcomes})",
                "n_new": n_new}

    frames = [df_new]
    n_base = 0
    if base_dataset:
        base_path = Path(base_dataset)
        if base_path.exists():
            df_base = pd.read_csv(base_path)
            n_base = len(df_base)
            frames.append(df_base)

    df = pd.concat(frames, ignore_index=True, sort=False)
    kwargs: dict[str, Any] = {}
    if model_path:
        kwargs["model_path"] = model_path
    metrics = ml_model.train(df, **kwargs)
    return {"status": "trained", "n_new": n_new, "n_base": n_base,
            "n_total": len(df), **metrics}
