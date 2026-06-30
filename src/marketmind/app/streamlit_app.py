"""MarketMind Streamlit dashboard.

Two tabs:
  - Research: calls the LangGraph orchestrator IN-PROCESS (no HTTP to the graph)
    and lights up four agent cards as the real pipeline streams (Quant, Sentiment,
    Risk, Report). Status is driven by orchestrator.run_analysis_stream.
  - Claude Trader: a VIRTUAL paper-trading account ($500 by default) driven by the
    deterministic script signal + advisory ML vote (offline, yfinance direct — no
    MCP servers needed). Each run fetches fresh prices, trades, and reports the
    portfolio value. Advisory only; no real orders.

The three specialist agents reach their data through MCP servers (teal tag); the
Report writer has no MCP access (distinct grey tag). The Research tab needs all
three MCP servers running (:8001 market data, :8002 news, :8003 portfolio) and the
DB seeded. The Claude Trader tab does not.

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

# Paper trader is offline (no Groq/servers); safe to import unconditionally.
from marketmind import paper_trader  # noqa: E402

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


# --- Research tab -----------------------------------------------------------

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


def _render_research() -> None:
    """Research tab body. The card slots are declared global so the render helpers
    (_render_static / _render_report) can see them."""
    global slots, report_slot, obs_slot

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


# --- Claude Trader tab ------------------------------------------------------

def _evaluated_rows(evaluated: list[dict]) -> list[dict]:
    """Flatten the per-ticker evaluation list into display rows (drops bulky tech)."""
    rows = []
    for ev in evaluated:
        ml = ev.get("ml") or {}
        rows.append({
            "ticker": ev.get("ticker"),
            "signal": ev.get("signal"),
            "confidence": ev.get("confidence"),
            "last": ev.get("last"),
            "ml_signal": ml.get("signal", "—"),
            "ml_conf": ml.get("confidence", "—"),
            "why": "; ".join(ev.get("reasons", []))[:120],
        })
    return rows


def _trader_safe_summary() -> dict | None:
    """Portfolio summary, or None if the marking fetch fails (offline/rate-limit)."""
    try:
        return paper_trader.portfolio_summary()
    except Exception as e:  # noqa: BLE001 - surface in the UI, never crash the tab
        st.warning(f"Could not value the portfolio right now: {e}")
        return None


def _render_trader() -> None:
    """Claude Trader tab: virtual $500 agent, fresh data each run, portfolio report."""
    st.markdown("### 🤖 Claude Trader")
    st.caption("VIRTUAL paper trading — advisory only, no real orders. "
               "Script signal + advisory ML vote, offline (no MCP servers).")

    c1, c2 = st.columns([1, 3])
    cash = c1.number_input("Starting cash ($)", min_value=50.0, step=50.0,
                           value=float(paper_trader.STARTING_CASH))
    wl = c2.text_input("Watchlist (comma-separated US tickers)",
                       value=", ".join(paper_trader.DEFAULT_WATCHLIST))

    b1, b2, _ = st.columns([1, 1, 3])
    run_btn = b1.button("Run Claude Trader", type="primary")
    reset_btn = b2.button("Reset account")

    if reset_btn:
        paper_trader.reset_paper(starting_cash=cash)
        st.session_state.trader_run = None
        st.session_state.trader_summary = _trader_safe_summary()
        st.success(f"Account reset to ${cash:,.2f}.")

    if run_btn:
        tickers = [t.strip() for t in wl.split(",") if t.strip()]
        with st.spinner("Fetching fresh prices, computing signals, trading…"):
            try:
                st.session_state.trader_run = paper_trader.decide_and_trade(watchlist=tickers)
            except Exception as e:  # noqa: BLE001
                st.error(f"Trader run failed: {e}")
            st.session_state.trader_summary = _trader_safe_summary()

    summary = st.session_state.trader_summary or _trader_safe_summary()
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total value", f"${summary['total_value']:,.2f}",
                  f"{summary['total_return_pct']:+.2f}%")
        m2.metric("Cash", f"${summary['cash']:,.2f}")
        m3.metric("Positions value", f"${summary['positions_value']:,.2f}")
        m4.metric("Total P&L", f"${summary['total_pnl']:+,.2f}")
        if summary["positions"]:
            st.markdown("**Holdings** (marked to market)")
            st.dataframe(summary["positions"], use_container_width=True, hide_index=True)
        else:
            st.info("No open positions — all in cash.")

    run = st.session_state.trader_run
    if run:
        if run["trades"]:
            st.markdown("**Trades this run**")
            st.dataframe(run["trades"], use_container_width=True, hide_index=True)
        else:
            st.caption("No trades this run — no fresh BUY/SELL signals.")
        if run.get("evaluated"):
            st.markdown("**Evaluated signals** (every watchlist name this run)")
            st.dataframe(_evaluated_rows(run["evaluated"]),
                         use_container_width=True, hide_index=True)
        if run["skipped"]:
            st.caption("Skipped (no data): " + ", ".join(run["skipped"]))

    history = paper_trader.recent_trades()
    if history:
        with st.expander("Trade history"):
            st.dataframe(history, use_container_width=True, hide_index=True)

    _render_learning()


def _render_learning() -> None:
    """Learning loop: retrain the ML model on the account's own closed trades."""
    st.divider()
    st.markdown("**🧠 Learn from closed trades**")
    try:
        n = paper_trader.outcomes_count()
    except Exception:  # noqa: BLE001
        n = 0
    st.caption(
        f"{n} closed-trade outcome(s) logged. Each SELL records the entry setup + "
        "realized return; retraining folds these (plus history) into the ML model so "
        "it learns from its own mistakes. Need ≥10 to retrain."
    )
    if st.button("Retrain ML from trades", disabled=n < 10):
        with st.spinner("Merging closed trades with history and retraining…"):
            try:
                res = paper_trader.retrain_from_trades()
            except Exception as e:  # noqa: BLE001
                st.error(f"Retrain failed: {e}")
                return
        if res["status"] == "skipped":
            st.warning(f"Skipped — {res['reason']}.")
        else:
            acc = res.get("accuracy", "n/a")
            st.success(
                f"Retrained on {res['n_total']} rows "
                f"({res['n_new']} paper trades + {res['n_base']} historical). "
                f"Holdout accuracy: {acc}."
            )


# --- page -------------------------------------------------------------------

st.session_state.setdefault("running", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("server_error", None)
st.session_state.setdefault("trader_run", None)
st.session_state.setdefault("trader_summary", None)

st.markdown("## MarketMind")
st.caption("Multi-agent investment research")

tab_research, tab_trader = st.tabs(["Research", "Claude Trader"])
with tab_research:
    _render_research()
with tab_trader:
    _render_trader()
