"""SQLite access layer for portfolio holdings.

Plain sqlite3 — no ORM, no migrations (MVP). The database lives at
data/marketmind.db relative to the project root. The Portfolio MCP server and
scripts/seed_db.py are the only callers.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

# Project root = .../marketmind ; this file is at src/marketmind/portfolio_db.py
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "marketmind.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    account_id TEXT    NOT NULL,
    ticker     TEXT    NOT NULL,
    shares     REAL    NOT NULL,
    cost_basis REAL    NOT NULL,
    sector     TEXT    NOT NULL,
    PRIMARY KEY (account_id, ticker)
);
"""


def connect() -> sqlite3.Connection:
    """Open a connection to the portfolio db with dict-like rows."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the holdings table if it does not exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Drop and recreate the holdings table (idempotent seeding)."""
    conn.execute("DROP TABLE IF EXISTS holdings;")
    conn.executescript(SCHEMA)
    conn.commit()


def insert_holdings(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    """Bulk-insert holding rows (account_id, ticker, shares, cost_basis, sector)."""
    conn.executemany(
        "INSERT INTO holdings (account_id, ticker, shares, cost_basis, sector) "
        "VALUES (:account_id, :ticker, :shares, :cost_basis, :sector);",
        rows,
    )
    conn.commit()


def get_holdings(account_id: str = "default") -> List[Dict[str, Any]]:
    """Return all holding rows for an account as a list of plain dicts."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT account_id, ticker, shares, cost_basis, sector "
            "FROM holdings WHERE account_id = ? ORDER BY ticker;",
            (account_id,),
        )
        return [dict(r) for r in cur.fetchall()]
