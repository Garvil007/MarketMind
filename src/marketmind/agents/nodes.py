"""LangGraph node functions: ingest, quant, sentiment, risk, report.

The three specialist nodes each build a server-scoped agent (via the factory),
run it on the ticker, and parse its JSON into the matching *Result dict. ingest
parses the raw query into ticker + proposed_notional. report is a plain LLM call
(write_report) that synthesizes everything into a cited Markdown report.

Every node body is wrapped by @_safe: on failure it appends to state["errors"]
and returns a no-op update instead of crashing the graph. All nodes reuse the
shared ChatGroq MODEL from the factory — single provider, traced to LangSmith.
"""
from __future__ import annotations

import functools
import json
import re

from marketmind import config, quant_signal
from marketmind.agents.factory import MODEL, build_agent, get_scoped_tools
from marketmind.agents.prompts import (
    QUANT_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
)
from marketmind.state import MarketMindState

# Confidence below this makes the quant node request the sentiment branch.
_SENTIMENT_CONFIDENCE_GATE = 0.6
# Confidence below this triggers in-process A2A delegation (only when ENABLE_A2A).
_A2A_CONFIDENCE_GATE = 0.4
_DEFAULT_NOTIONAL = 10000.0

_SENTIMENT_KEYS = {"label", "score", "headline_count", "summary"}

# Common all-caps tokens that are NOT tickers, so ingest doesn't mistake them.
_NOT_TICKERS = {"I", "A", "BUY", "SELL", "HOLD", "USD", "US", "USA", "THE", "DD", "CEO", "ETF", "IPO"}


def _section(label: str, value: object, absent: str) -> str:
    """Render one agent's JSON block for the user message, or an absent marker."""
    body = json.dumps(value, indent=2) if value else absent
    return f"{label}:\n{body}"


def _build_user_message(state: MarketMindState) -> str:
    """Flatten the relevant state into the human message for the report LLM."""
    parts = [
        f"User query: {state.get('query', '(none)')}",
        f"Ticker: {state.get('ticker', '(unknown)')}",
        f"Proposed notional: {state.get('proposed_notional', 10000)}",
    ]
    # Only present (and only when A2A fired) — keeps the off-path message identical.
    if state.get("delegated_to"):
        parts.append(f"Note: the Quant agent delegated directly to the {state['delegated_to']} agent (A2A).")
    return "\n\n".join(
        parts
        + [
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


def _extract_json(text: str) -> dict:
    """Pull the first/last-brace JSON object out of an LLM message.

    strict=False because LLMs routinely emit raw newlines inside string values.
    """
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in LLM output:\n{text}")
    return json.loads(text[start:end + 1], strict=False)


def _parse_result(text: str, required: set[str]) -> dict:
    """Parse an agent's JSON output and verify it carries the required keys."""
    data = _extract_json(text)
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Agent result missing keys {missing}: {data}")
    return data


def _parse_report(text: str) -> dict:
    """Parse the report LLM's JSON, tolerating stray prose / code fences."""
    data = _extract_json(text)
    if "report_markdown" not in data:
        raise ValueError(f"Report output missing 'report_markdown': {data}")
    return {
        "report_markdown": data["report_markdown"],
        "citations": data.get("citations", []),
    }


def _safe(node_name: str):
    """Wrap an async node so any exception appends to state['errors'] instead of crashing.

    On failure the node returns only an updated 'errors' list, leaving every
    other state field untouched (downstream nodes see the absence and adapt).
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state: MarketMindState) -> dict:
            try:
                return await fn(state)
            except Exception as e:  # noqa: BLE001 - graph must survive any node failure
                errors = list(state.get("errors", []))
                errors.append(f"{node_name}: {e}")
                return {"errors": errors}
        return wrapper
    return decorator


async def _run_scoped_agent(server_name: str, prompt: str, message: str, required: set[str]) -> dict:
    """Build a server-scoped agent, run it on one message, parse its JSON result."""
    agent = await build_agent(server_name, prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    return _parse_result(_message_text(result["messages"][-1]), required)


async def _call_tool(server_name: str, tool_name: str, args: dict) -> dict:
    """Invoke a single MCP tool directly (no agent) and return its dict payload.

    Used so the Quant node can compute the deterministic script prior from real
    technicals while still going through MCP (never touching yfinance directly).
    """
    tools = await get_scoped_tools(server_name)
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found on server '{server_name}'.")
    raw = await tool.ainvoke(args)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    # Some adapters return a content-block list; pull the text out and parse.
    return _extract_json(_message_text(raw) if hasattr(raw, "content") else str(raw))


# --- Graph nodes -----------------------------------------------------------

@_safe("ingest")
async def ingest(state: MarketMindState) -> dict:
    """Parse the raw query into a ticker symbol and a proposed notional amount."""
    query = state.get("query", "")

    # Ticker: first all-caps 1-5 letter token that isn't a common non-ticker word.
    ticker = None
    for token in re.findall(r"\b[A-Z]{1,5}\b", query):
        if token not in _NOT_TICKERS:
            ticker = token
            break
    if ticker is None:
        raise ValueError(f"No ticker symbol found in query: {query!r}")

    # Notional: first $-amount like $5,000 or $7500.50; default otherwise.
    notional = _DEFAULT_NOTIONAL
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", query)
    if m:
        notional = float(m.group(1).replace(",", ""))

    return {"ticker": ticker, "proposed_notional": notional}


@_safe("quant")
async def quant_node(state: MarketMindState) -> dict:
    """Run the Quant agent; set state['quant'] and the run_sentiment router flag.

    When ENABLE_A2A is on AND confidence is very low (< 0.4), the Quant node makes
    a lightweight in-process A2A call to the Sentiment agent directly — distinct
    from the graph's normal routing — and tags state with delegated_to="sentiment".
    With the flag off this path is dead, so the baseline is byte-for-byte unchanged.
    """
    ticker = state["ticker"]

    # Script-first: compute the deterministic prior from real technicals, then
    # hand it to the LLM as its default. The LLM may override it with a reason.
    decision = None
    prior_msg = ""
    try:
        tech = await _call_tool("market_data", "get_technicals", {"ticker": ticker})
        if "error" not in tech:
            decision = quant_signal.compute_signal(tech)
            prior_msg = (
                "\n\nSCRIPT PRIOR (deterministic rule engine — treat as your default "
                "unless the technicals clearly contradict it; if you override it, say "
                "why in the rationale):\n" + quant_signal.prior_block(decision)
            )
    except Exception:  # noqa: BLE001 - prior is best-effort; LLM still runs without it
        decision = None

    quant = await _run_scoped_agent(
        "market_data",
        QUANT_SYSTEM_PROMPT,
        f"Analyze {ticker}.{prior_msg}",
        {"signal", "confidence", "rsi_14", "sma_50", "last_close", "above_sma_50", "rationale"},
    )

    # Record the prior and whether the LLM overrode it, for transparency.
    if decision is not None:
        quant["script_signal"] = decision["signal"]
        quant["script_confidence"] = decision["confidence"]
        quant["overridden"] = quant.get("signal") != decision["signal"]

    confidence = float(quant["confidence"])
    update: dict = {"quant": quant, "run_sentiment": confidence < _SENTIMENT_CONFIDENCE_GATE}

    if config.ENABLE_A2A and confidence < _A2A_CONFIDENCE_GATE:
        # A2A: Quant asks Sentiment directly (horizontal), bypassing graph routing.
        sentiment = await _run_scoped_agent(
            "news",
            SENTIMENT_SYSTEM_PROMPT,
            f"Assess recent news sentiment for {ticker}.",
            _SENTIMENT_KEYS,
        )
        update["sentiment"] = sentiment
        update["delegated_to"] = "sentiment"
        # The read is already satisfied; skip the graph's sentiment node.
        update["run_sentiment"] = False

    return update


@_safe("sentiment")
async def sentiment_node(state: MarketMindState) -> dict:
    """Run the Sentiment agent; set state['sentiment']."""
    ticker = state["ticker"]
    sentiment = await _run_scoped_agent(
        "news",
        SENTIMENT_SYSTEM_PROMPT,
        f"Assess recent news sentiment for {ticker}.",
        _SENTIMENT_KEYS,
    )
    return {"sentiment": sentiment}


@_safe("risk")
async def risk_node(state: MarketMindState) -> dict:
    """Run the Risk agent; set state['risk']."""
    ticker = state["ticker"]
    notional = state.get("proposed_notional", _DEFAULT_NOTIONAL)
    risk = await _run_scoped_agent(
        "portfolio",
        RISK_SYSTEM_PROMPT,
        f"Assess the risk of buying ${notional} of {ticker}.",
        {"level", "current_weight", "projected_weight", "note"},
    )
    return {"risk": risk}


@_safe("report")
async def report_node(state: MarketMindState) -> dict:
    """Run the Report Writer; set state['report_markdown'] and state['citations']."""
    return await write_report(state)


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
