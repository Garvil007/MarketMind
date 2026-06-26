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

import logging
from datetime import datetime, timezone
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
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- account / persistence --------------------------------------------------

def init_paper(account: str = ACCOUNT_ID, starting_cash: float = STARTING_CASH) -> None:
    """Create tables and seed the account with starting cash if it doesn't exist."""
    with connect() as conn:
        conn.executescript(_SCHEMA)
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


def _load_positions(conn, account: str) -> dict[str, dict[str, float]]:
    cur = conn.execute(
        "SELECT ticker, shares, avg_cost FROM paper_positions WHERE account_id = ?;", (account,)
    )
    return {r["ticker"]: {"shares": float(r["shares"]), "avg_cost": float(r["avg_cost"])}
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
    }


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
                conn.execute("DELETE FROM paper_positions WHERE account_id=? AND ticker=?;", (account, t))
                del positions[t]

        # --- BUY: fresh BUY candidates, ranked by confidence --------------
        per_cap = starting * MAX_POS_FRAC
        candidates = sorted(
            (ev for t, ev in evals.items() if ev["signal"] == "BUY" and t not in positions),
            key=lambda e: e["confidence"], reverse=True,
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
            positions[ev["ticker"]] = {"shares": shares, "avg_cost": ev["last"]}
            conn.execute(
                "INSERT INTO paper_positions (account_id, ticker, shares, avg_cost) VALUES (?,?,?,?) "
                "ON CONFLICT(account_id, ticker) DO UPDATE SET shares=excluded.shares, avg_cost=excluded.avg_cost;",
                (account, ev["ticker"], shares, ev["last"]),
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
