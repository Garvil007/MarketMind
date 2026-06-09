"""End-to-end pipeline run.

Starts NOTHING — all three MCP servers must already be running:
  Market Data :8001  |  News :8002  |  Portfolio :8003
(and the portfolio DB seeded via scripts/seed_db.py).

Calls run_analysis on a sample query and prints the cited report. The full run
ingest -> quant -> (sentiment?) -> risk -> report is traced to LangSmith.

Run:  python scripts/run_pipeline.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make src/ importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.graph.orchestrator import run_analysis  # noqa: E402

QUERY = "Should I buy NVDA this week?"


async def main() -> None:
    state = await run_analysis(QUERY)

    print(f"Query : {QUERY}")
    print(f"Ticker: {state.get('ticker')}   Notional: {state.get('proposed_notional')}")
    print(f"Quant signal: {(state.get('quant') or {}).get('signal')}   "
          f"run_sentiment: {state.get('run_sentiment')}")
    if state.get("delegated_to"):
        print(f"A2A: Quant delegated_to -> {state['delegated_to']}")

    print("\n===== REPORT =====\n")
    print(state.get("report_markdown", "(no report produced)"))

    citations = state.get("citations", [])
    print(f"\n===== CITATIONS ({len(citations)}) =====\n")
    print(json.dumps(citations, indent=2))

    errors = state.get("errors", [])
    if errors:
        print("\n===== ERRORS =====")
        for e in errors:
            print(f"- {e}")


if __name__ == "__main__":
    asyncio.run(main())
