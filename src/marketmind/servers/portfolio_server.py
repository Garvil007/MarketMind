"""Portfolio MCP server (FastMCP, Streamable HTTP, port 8003, /mcp).

Two tools: get_holdings and assess_position_risk. Holdings live in SQLite
(see portfolio_db); current prices come from yfinance. Plain sqlite3, no ORM.
See CLAUDE.md for the contract.
"""
from __future__ import annotations

from typing import Dict, List

import yfinance as yf
from fastmcp import FastMCP

from marketmind import portfolio_db

mcp = FastMCP("portfolio")

# Concentration thresholds on projected SECTOR exposure (percent of portfolio).
_MODERATE_SECTOR_PCT = 25.0
_HIGH_SECTOR_PCT = 40.0


def _last_price(ticker: str) -> float:
    """Most recent close for a ticker via yfinance; 0.0 if unavailable."""
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return 0.0
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:  # noqa: BLE001 - price is best-effort, never raise to the agent
        return 0.0


def _priced_holdings(account_id: str) -> List[Dict]:
    """Holdings rows enriched with current price and market_value."""
    rows = portfolio_db.get_holdings(account_id)
    for r in rows:
        price = _last_price(r["ticker"])
        r["price"] = round(price, 2)
        r["market_value"] = round(price * float(r["shares"]), 2)
    return rows


@mcp.tool
def get_holdings(account_id: str = "default") -> dict:
    """List a portfolio account's holdings with live valuation and weights.

    Args:
        account_id: Account key. Default "default" (the seeded account).

    Returns:
        {"holdings": [{"ticker", "shares", "cost_basis", "market_value",
                       "weight", "sector"}, ...],   # weight = percent of total
         "total_value": float,
         "by_sector": {sector: weight_percent, ...}}
        Empty account -> empty holdings, total_value 0.0, empty by_sector.
    """
    rows = _priced_holdings(account_id)
    total_value = round(sum(r["market_value"] for r in rows), 2)

    holdings = []
    by_sector: Dict[str, float] = {}
    for r in rows:
        weight = round(r["market_value"] / total_value * 100.0, 2) if total_value else 0.0
        holdings.append({
            "ticker": r["ticker"],
            "shares": r["shares"],
            "cost_basis": r["cost_basis"],
            "market_value": r["market_value"],
            "weight": weight,
            "sector": r["sector"],
        })
        by_sector[r["sector"]] = round(by_sector.get(r["sector"], 0.0) + weight, 2)

    return {"holdings": holdings, "total_value": total_value, "by_sector": by_sector}


@mcp.tool
def assess_position_risk(ticker: str, proposed_notional: float,
                         account_id: str = "default") -> dict:
    """Assess concentration risk of adding a dollar amount to a ticker.

    Computes the ticker's current weight, its weight after adding
    `proposed_notional`, and the resulting sector exposure, then grades
    concentration from the projected sector exposure.

    Args:
        ticker: Stock symbol to evaluate, e.g. "NVDA".
        proposed_notional: Dollar amount to hypothetically buy.
        account_id: Account key. Default "default".

    Returns:
        {"current_weight": float,          # percent of portfolio, now
         "projected_weight": float,        # percent after the buy
         "sector": str,                    # ticker's sector ("Unknown" if not held)
         "sector_exposure_after": float,   # sector percent after the buy
         "concentration_level": str,       # "low" / "moderate" / "high"
         "note": str}                      # one-sentence summary
    """
    ticker = ticker.upper()
    rows = _priced_holdings(account_id)
    total_value = sum(r["market_value"] for r in rows)

    held = next((r for r in rows if r["ticker"] == ticker), None)
    sector = held["sector"] if held else "Unknown"
    ticker_value = held["market_value"] if held else 0.0
    sector_value = sum(r["market_value"] for r in rows if r["sector"] == sector) if sector != "Unknown" else 0.0

    new_total = total_value + proposed_notional
    current_weight = round(ticker_value / total_value * 100.0, 2) if total_value else 0.0
    projected_weight = round((ticker_value + proposed_notional) / new_total * 100.0, 2) if new_total else 0.0
    sector_exposure_after = round((sector_value + proposed_notional) / new_total * 100.0, 2) if new_total else 0.0

    if sector_exposure_after >= _HIGH_SECTOR_PCT:
        level = "high"
    elif sector_exposure_after >= _MODERATE_SECTOR_PCT:
        level = "moderate"
    else:
        level = "low"

    if sector == "Unknown":
        note = (f"{ticker} is not currently held; adding ${proposed_notional:,.0f} would make it "
                f"{projected_weight:.1f}% of the portfolio.")
    else:
        note = (f"Adding ${proposed_notional:,.0f} to {ticker} lifts it to {projected_weight:.1f}% "
                f"and {sector} exposure to {sector_exposure_after:.1f}% ({level} concentration).")

    return {
        "current_weight": current_weight,
        "projected_weight": projected_weight,
        "sector": sector,
        "sector_exposure_after": sector_exposure_after,
        "concentration_level": level,
        "note": note,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
