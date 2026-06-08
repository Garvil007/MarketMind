"""MarketMind Streamlit dashboard.

Calls the LangGraph orchestrator IN-PROCESS (no HTTP to the graph) and lights up
four agent cards as the real pipeline streams: Quant, Sentiment, Risk, Report.
Status is driven by orchestrator.run_analysis_stream — not a fake timer.

The three specialist agents reach their data through MCP servers (teal tag); the
Report writer has no MCP access (distinct grey tag). All three MCP servers must
be running (:8001 market data, :8002 news, :8003 portfolio) and the DB seeded.

Run:  streamlit run src/marketmind/app/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import re
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

# Make src/ importable when Streamlit runs this file by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marketmind import config  # noqa: E402

st.set_page_config(page_title="MarketMind", layout="wide")

# Importing the orchestrator pulls in agents.factory, which fails fast if the
# Groq key is missing. Surface that as a clean message, not a stack trace.
try:
    from marketmind.graph.orchestrator import run_analysis_stream  # noqa: E402
except config.ConfigError as e:
    st.error(f"⚙️ Configuration error: {e}")
    st.stop()

DEFAULT_QUERY = "Should I buy NVDA this week?"

# Human-readable server names for error messages, keyed by config.MCP_URLS key.
SERVER_LABELS = {"market_data": "Market Data", "news": "News", "portfolio": "Portfolio"}

# Node -> card title. ingest has no card (it only sets ticker/notional).
CARDS = ["quant", "sentiment", "risk", "report"]
CARD_META = {
    "quant": ("Quant", "Market Data MCP", "mcp"),
    "sentiment": ("Sentiment", "News MCP", "mcp"),
    "risk": ("Risk", "Portfolio MCP", "mcp"),
    "report": ("Report writer", "no MCP access", "nomcp"),
}

BADGE_COLORS = {
    "queued": ("#6b7280", "#1f2937"),    # grey
    "running": ("#f59e0b", "#3a2e12"),   # amber
    "done": ("#10b981", "#0f2e23"),      # green
    "skipped": ("#6b7280", "#1f2937"),   # grey
    "error": ("#ef4444", "#3a1717"),     # red
}

LANGSMITH_URL = "https://smith.langchain.com"


# --- rendering helpers ------------------------------------------------------

def _headline(node: str, state: dict) -> str:
    """One-line output summary for a finished node; rounds all numbers."""
    if node == "quant" and state.get("quant"):
        q = state["quant"]
        return f"{q['signal']} · conf {round(float(q['confidence']), 2)}"
    if node == "sentiment" and state.get("sentiment"):
        s = state["sentiment"]
        return f"{s['label']} · {round(float(s['score']), 2)} · {s['headline_count']} hl"
    if node == "risk" and state.get("risk"):
        r = state["risk"]
        return f"{r['level']} · proj {round(float(r['projected_weight']), 1)}%"
    if node == "report" and state.get("report_markdown"):
        return "report ready"
    return "—"


def _tag_html(label: str, kind: str) -> str:
    if kind == "mcp":
        return (f"<span style='background:#0f2e2b;color:#2dd4bf;border:1px solid #134e4a;"
                f"padding:1px 7px;border-radius:10px;font-size:11px'>{label}</span>")
    return (f"<span style='background:#1f2937;color:#9ca3af;border:1px dashed #4b5563;"
            f"padding:1px 7px;border-radius:10px;font-size:11px'>{label}</span>")


def _render_card(slot, node: str, status: str, headline: str) -> None:
    title, mcp_label, kind = CARD_META[node]
    fg, bg = BADGE_COLORS[status]
    badge = (f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:10px;"
             f"font-size:12px;font-weight:600;text-transform:uppercase'>{status}</span>")
    body = (f"<div style='color:#e5e7eb;font-size:15px;margin-top:8px'>{headline}</div>"
            if status in ("done", "error") else
            "<div style='color:#6b7280;font-size:13px;margin-top:8px'>awaiting…</div>")
    slot.markdown(
        f"""<div style='border:1px solid #374151;border-radius:12px;padding:14px;
        background:#111827;min-height:120px'>
        <div style='display:flex;justify-content:space-between;align-items:center'>
          <span style='color:#f9fafb;font-weight:600'>{title}</span>{badge}
        </div>
        <div style='margin-top:6px'>{_tag_html(mcp_label, kind)}</div>
        {body}</div>""",
        unsafe_allow_html=True,
    )


_TAG_RE = re.compile(r"\[(Quant|Sentiment|Risk)\]")
_TAG_BG = {"Quant": "#2dd4bf", "Sentiment": "#a78bfa", "Risk": "#f59e0b"}


def _highlight_citations(md: str) -> str:
    """Wrap inline [Quant]/[Sentiment]/[Risk] tags in colored pills (kept visible)."""
    def repl(m: re.Match) -> str:
        name = m.group(1)
        return (f"<span style='background:{_TAG_BG[name]};color:#0b0f19;font-size:11px;"
                f"font-weight:700;padding:1px 6px;border-radius:8px'>{name}</span>")
    return _TAG_RE.sub(repl, md)


# --- streaming driver -------------------------------------------------------

def _down_servers() -> list[tuple[str, int]]:
    """Return (server_key, port) for any MCP server not accepting TCP connections."""
    down = []
    for key, url in config.MCP_URLS.items():
        parsed = urlparse(url)
        host, port = parsed.hostname or "localhost", parsed.port or 80
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) != 0:
                down.append((key, port))
    return down


def _unreachable_messages(down: list[tuple[str, int]]) -> list[str]:
    """Friendly 'run `make servers`' message per unreachable server."""
    return [
        f"{SERVER_LABELS.get(key, key)} MCP not reachable on :{port} — run `make servers`"
        for key, port in down
    ]


def _iter_stream(query: str):
    """Drive the async run_analysis_stream generator from sync Streamlit code."""
    loop = asyncio.new_event_loop()
    agen = run_analysis_stream(query)
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()


def _next_running(finished: str, state: dict, status: dict) -> None:
    """After `finished` completes, mark the next node running (and skips)."""
    if finished == "ingest":
        status["quant"] = "running"
    elif finished == "quant":
        if state.get("run_sentiment"):
            status["sentiment"] = "running"
        else:
            status["sentiment"] = "skipped"
            status["risk"] = "running"
    elif finished == "sentiment":
        status["risk"] = "running"
    elif finished == "risk":
        status["report"] = "running"


def _run_pipeline(query: str, slots: dict) -> tuple[dict, float]:
    """Stream the graph, updating card slots live. Returns (final_state, latency_s)."""
    status = {c: "queued" for c in CARDS}
    headline = {c: "—" for c in CARDS}
    for c in CARDS:
        _render_card(slots[c], c, status[c], headline[c])

    final_state: dict = {"query": query}
    t0 = time.perf_counter()
    for node, state in _iter_stream(query):
        final_state = state
        if node in CARDS:
            errored = (node == "report" and not state.get("report_markdown")) or \
                      (node in ("quant", "sentiment", "risk") and not state.get(node))
            status[node] = "error" if errored else "done"
            headline[node] = "error" if errored else _headline(node, state)
        _next_running(node, state, status)
        for c in CARDS:
            _render_card(slots[c], c, status[c], headline[c])
    latency = time.perf_counter() - t0
    return final_state, latency


# --- page -------------------------------------------------------------------

st.session_state.setdefault("running", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("server_error", None)

st.markdown("## MarketMind")
st.caption("Multi-agent investment research")
st.markdown(
    "<div style='color:#10b981;font-size:13px'>● 3 MCP servers online "
    "<span style='color:#6b7280'>· Streamable HTTP</span></div>",
    unsafe_allow_html=True,
)
st.divider()

query = st.text_input("Query", value=DEFAULT_QUERY, label_visibility="collapsed")
analyze = st.button("Analyze", type="primary", disabled=st.session_state.running)

st.write("")
cols = st.columns(4)
slots = {c: cols[i].empty() for i, c in enumerate(CARDS)}
report_slot = st.empty()
obs_slot = st.empty()


def _render_static(result: dict | None) -> None:
    """Render cards + report from stored result (idle state, between runs)."""
    if not result:
        for c in CARDS:
            _render_card(slots[c], c, "queued", "—")
        return
    state = result["state"]
    ran_sentiment = state.get("run_sentiment", True)
    for c in CARDS:
        if c == "sentiment" and not ran_sentiment and not state.get("sentiment"):
            _render_card(slots[c], c, "skipped", "—")
        else:
            done = bool(state.get(c)) or (c == "report" and state.get("report_markdown"))
            _render_card(slots[c], c, "done" if done else "error", _headline(c, state))
    _render_report(state, result["latency"])


def _render_report(state: dict, latency: float) -> None:
    md = state.get("report_markdown")
    if md:
        report_slot.markdown(_highlight_citations(md), unsafe_allow_html=True)
    else:
        report_slot.info("No report produced.")
    obs_slot.markdown(
        f"<div style='color:#6b7280;font-size:12px;margin-top:10px'>"
        f"⏱ total latency {round(latency, 2)}s &nbsp;·&nbsp; "
        f"<a href='{LANGSMITH_URL}' target='_blank' style='color:#2dd4bf'>"
        f"LangSmith project: {config.LANGSMITH_PROJECT}</a></div>",
        unsafe_allow_html=True,
    )


# Two-phase to keep the button disabled while a run is in flight:
if analyze and not st.session_state.running:
    st.session_state.running = True
    st.session_state.pending_query = query
    st.session_state.server_error = None
    st.rerun()

if st.session_state.running:
    # Preflight: don't start the graph if a server is down — show a clear fix.
    down = _down_servers()
    if down:
        st.session_state.server_error = _unreachable_messages(down)
        st.session_state.running = False
        st.rerun()

    state, latency = _run_pipeline(st.session_state.pending_query, slots)
    errors = state.get("errors", [])
    _render_report(state, latency)
    if errors:
        st.warning("Some nodes reported errors:\n" + "\n".join(f"- {e}" for e in errors))
    st.session_state.result = {"state": state, "latency": latency}
    st.session_state.running = False
    st.rerun()
else:
    for msg in st.session_state.server_error or []:
        st.error(msg)
    _render_static(st.session_state.result)
