"""Smoke test for the specialist agents (quant, sentiment, risk).

Each agent is scoped to ONE running MCP server:
  quant     -> Market Data :8001/mcp
  sentiment -> News        :8002/mcp
  risk      -> Portfolio   :8003/mcp

Builds each agent, runs it on NVDA, parses the final message into its *Result
dict, prints it. Runs are traced to LangSmith (env vars come from config).

Run one or more by name (default: all):
  python scripts/test_agents.py                 # all three
  python scripts/test_agents.py sentiment risk  # only these
Each agent's server must be up before it is exercised.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make src/ importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketmind.agents.factory import build_agent  # noqa: E402
from marketmind.agents.prompts import (             # noqa: E402
    QUANT_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)

TICKER = "NVDA"
PROPOSED_NOTIONAL = 10000

# name -> (server_name, system_prompt, expected JSON keys, user message)
AGENTS = {
    "quant": (
        "market_data",
        QUANT_SYSTEM_PROMPT,
        {"signal", "confidence", "rsi_14", "sma_50", "last_close", "above_sma_50", "rationale"},
        f"Analyze {TICKER}.",
    ),
    "sentiment": (
        "news",
        SENTIMENT_SYSTEM_PROMPT,
        {"label", "score", "headline_count", "summary"},
        f"Assess recent news sentiment for {TICKER}.",
    ),
    "risk": (
        "portfolio",
        RISK_SYSTEM_PROMPT,
        {"level", "current_weight", "projected_weight", "note"},
        f"Assess the risk of buying ${PROPOSED_NOTIONAL} of {TICKER}.",
    ),
}


def _final_text(result: dict) -> str:
    """Extract the last message's text content (string or content-block list)."""
    msg = result["messages"][-1]
    content = msg.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _parse(text: str, expected: set[str]) -> dict:
    """Parse the agent's JSON output, tolerating stray prose / code fences."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in agent output:\n{text}")
    data = json.loads(text[start:end + 1])
    missing = expected - data.keys()
    if missing:
        raise ValueError(f"Result missing keys {missing}: {data}")
    return data


async def run_agent(name: str) -> dict:
    server_name, prompt, expected, message = AGENTS[name]
    agent = await build_agent(server_name, prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    return _parse(_final_text(result), expected)


async def main(names: list[str]) -> None:
    for name in names:
        print(f"\n=== {name} agent ({AGENTS[name][0]}) ===")
        try:
            print(json.dumps(await run_agent(name), indent=2))
        except Exception as e:  # keep going so one dead server doesn't block the rest
            print(f"FAILED: {e}")


if __name__ == "__main__":
    selected = sys.argv[1:] or list(AGENTS)
    unknown = [n for n in selected if n not in AGENTS]
    if unknown:
        sys.exit(f"Unknown agent(s) {unknown}. Choose from {list(AGENTS)}.")
    asyncio.run(main(selected))
