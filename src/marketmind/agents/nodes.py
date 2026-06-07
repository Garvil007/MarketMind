"""LangGraph node functions.

Implemented so far: write_report — the Report Writer. It is a plain LLM call
(no MCP server, no tools): it takes the accumulated MarketMindState (query,
ticker, and the quant/sentiment/risk result dicts) and produces the final cited
Markdown report. It reuses the shared ChatGroq MODEL from the factory — single
provider, same key, traced to LangSmith on import.

TODO: intake_node, quant_node (sets run_sentiment), sentiment_node, risk_node.
"""
from __future__ import annotations

import json

from marketmind.agents.factory import MODEL
from marketmind.agents.prompts import REPORT_SYSTEM_PROMPT
from marketmind.state import MarketMindState


def _section(label: str, value: object, absent: str) -> str:
    """Render one agent's JSON block for the user message, or an absent marker."""
    body = json.dumps(value, indent=2) if value else absent
    return f"{label}:\n{body}"


def _build_user_message(state: MarketMindState) -> str:
    """Flatten the relevant state into the human message for the report LLM."""
    return "\n\n".join(
        [
            f"User query: {state.get('query', '(none)')}",
            f"Ticker: {state.get('ticker', '(unknown)')}",
            f"Proposed notional: {state.get('proposed_notional', 10000)}",
            _section("Quant agent output", state.get("quant"), "(absent)"),
            _section(
                "Sentiment agent output",
                state.get("sentiment"),
                "(absent — the sentiment branch was skipped)",
            ),
            _section("Risk agent output", state.get("risk"), "(absent)"),
        ]
    )


def _message_text(resp) -> str:
    """Extract text from an LLM response (string or content-block list)."""
    content = resp.content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _parse_report(text: str) -> dict:
    """Parse the report LLM's JSON, tolerating stray prose / code fences."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in report output:\n{text}")
    # strict=False: LLMs routinely put raw newlines inside the markdown string.
    data = json.loads(text[start:end + 1], strict=False)
    if "report_markdown" not in data:
        raise ValueError(f"Report output missing 'report_markdown': {data}")
    return {
        "report_markdown": data["report_markdown"],
        "citations": data.get("citations", []),
    }


async def write_report(state: MarketMindState) -> dict:
    """Report Writer node: synthesize agent outputs into a cited Markdown report.

    Args:
        state: MarketMindState with query/ticker and quant/sentiment/risk dicts
            (sentiment may be absent).

    Returns:
        {"report_markdown": str, "citations": [{"claim", "agent"}, ...]}
        — a partial state update for the orchestrator to merge.
    """
    user_message = _build_user_message(state)
    resp = await MODEL.ainvoke(
        [("system", REPORT_SYSTEM_PROMPT), ("human", user_message)]
    )
    return _parse_report(_message_text(resp))
