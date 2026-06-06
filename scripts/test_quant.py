"""Smoke test for the Quant agent.

Assumes the Market Data server is ALREADY running on :8001/mcp. Builds the Quant
agent, runs it on NVDA, parses the final message into a QuantResult, prints it.
The run is traced to LangSmith (env vars come from config).

Run:  python scripts/test_quant.py     (with the market data server up)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make src/ importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.agents.factory import build_agent          # noqa: E402
from marketmind.agents.prompts import QUANT_SYSTEM_PROMPT   # noqa: E402
from marketmind.state import QuantResult                    # noqa: E402

TICKER = "NVDA"

_EXPECTED_KEYS = {"signal", "confidence", "rsi_14", "sma_50", "last_close", "above_sma_50", "rationale"}


def _final_text(result: dict) -> str:
    """Extract the last message's text content (string or content-block list)."""
    msg = result["messages"][-1]
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _parse_quant(text: str) -> QuantResult:
    """Parse the agent's JSON output, tolerating stray prose / code fences."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in agent output:\n{text}")
    data = json.loads(text[start:end + 1])
    missing = _EXPECTED_KEYS - data.keys()
    if missing:
        raise ValueError(f"QuantResult missing keys {missing}: {data}")
    return data  # type: ignore[return-value]


async def main() -> None:
    agent = await build_agent("market_data", QUANT_SYSTEM_PROMPT)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Analyze {TICKER}."}]}
    )
    quant = _parse_quant(_final_text(result))
    print(json.dumps(quant, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
