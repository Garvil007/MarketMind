"""Seed the SQLite portfolio with a realistic 'default' account.

Idempotent: drops and recreates the holdings table on every run. Six holdings
across sectors with a clear tech tilt. Plain sqlite3, no ORM/migrations.

Run:  python scripts/seed_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind import portfolio_db  # noqa: E402

# Tech tilt: AAPL + MSFT + NVDA dominate; JPM/XOM/JNJ diversify.
SEED_HOLDINGS = [
    {"account_id": "default", "ticker": "AAPL", "shares": 120, "cost_basis": 175.40, "sector": "Technology"},
    {"account_id": "default", "ticker": "MSFT", "shares": 60,  "cost_basis": 330.10, "sector": "Technology"},
    {"account_id": "default", "ticker": "NVDA", "shares": 80,  "cost_basis": 110.25, "sector": "Technology"},
    {"account_id": "default", "ticker": "JPM",  "shares": 50,  "cost_basis": 180.75, "sector": "Financials"},
    {"account_id": "default", "ticker": "XOM",  "shares": 70,  "cost_basis": 104.60, "sector": "Energy"},
    {"account_id": "default", "ticker": "JNJ",  "shares": 40,  "cost_basis": 152.30, "sector": "Healthcare"},
]


def main() -> None:
    with portfolio_db.connect() as conn:
        portfolio_db.reset_db(conn)
        portfolio_db.insert_holdings(conn, SEED_HOLDINGS)
    print(f"Seeded {len(SEED_HOLDINGS)} holdings into {portfolio_db.DB_PATH}")


if __name__ == "__main__":
    main()
