"""Smoke test for the Report Writer node.

No MCP servers needed — write_report is a single LLM call. Feeds hand-built
quant/sentiment/risk dicts, prints the tagged Markdown report and the citations
list. Traced to LangSmith (env vars come from config). Needs a real GROQ_API_KEY.

Run:  python scripts/test_report.py
      python scripts/test_report.py --no-sentiment   # exercise the absent-branch path
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make src/ importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.agents.nodes import write_report  # noqa: E402

STATE = {
    "query": "Should I buy NVDA?",
    "ticker": "NVDA",
    "proposed_notional": 10000,
    "quant": {
        "signal": "BUY",
        "confidence": 0.72,
        "rsi_14": 61.4,
        "sma_50": 168.30,
        "last_close": 182.05,
        "above_sma_50": True,
        "rationale": "Price above the 50-day SMA with a true buy_signal and rs_high vs the S&P 500.",
    },
    "sentiment": {
        "label": "positive",
        "score": 0.34,
        "headline_count": 10,
        "summary": "Coverage leans positive on data-center demand and new product launches.",
    },
    "risk": {
        "level": "high",
        "current_weight": 14.2,
        "projected_weight": 22.6,
        "note": "Technology sector exposure rises to 71% after the buy.",
    },
}


async def main(with_sentiment: bool) -> None:
    state = dict(STATE)
    if not with_sentiment:
        state.pop("sentiment")
    out = await write_report(state)
    print("===== report_markdown =====\n")
    print(out["report_markdown"])
    print("\n===== citations =====\n")
    print(json.dumps(out["citations"], indent=2))


if __name__ == "__main__":
    asyncio.run(main(with_sentiment="--no-sentiment" not in sys.argv))
