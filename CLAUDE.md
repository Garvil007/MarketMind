# MarketMind — Project Source of Truth

Multi-agent financial research MVP. Three MCP servers expose data tools; a
LangGraph orchestrator runs specialist agents (quant, sentiment, risk) and
emits a cited markdown report.

---

## Architecture

```
                ┌─────────────────────────────────────────┐
   user query → │           LangGraph Orchestrator         │
                │   intake → quant → [router] → sentiment   │
                │                 ↘──────────→ risk → report │
                └───────┬───────────────┬───────────────┬────┘
                        │ MCP           │ MCP           │ MCP
                 ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
                 │ Market Data │  │   News     │  │  Portfolio  │
                 │  :8001/mcp  │  │  :8002/mcp │  │  :8003/mcp  │
                 │  yfinance   │  │  yf + VADER│  │  SQLite     │
                 └─────────────┘  └────────────┘  └─────────────┘
```

- **Transport:** every server is FastMCP over Streamable HTTP at path `/mcp`.
- **Agents:** `ChatGroq` (Llama, model from `AGENT_MODEL`, free tier) + MCP tools
  loaded per-agent via `langchain-mcp-adapters`. Each agent sees ONLY its server's tools.
- **Router:** after the quant node, `run_sentiment` flag decides whether the
  sentiment branch runs (e.g. skip on a clear SELL).
- **Tracing:** LangSmith, project `marketmind`.
- **Persistence:** portfolio holdings in SQLite under `data/` (gitignored).

---

## Hard Scope Boundary (MVP)

IN scope:
- One ticker per query, US equities.
- Three servers, six tools (below). Quant / sentiment / risk agents. One report.
- SQLite portfolio seeded by `scripts/seed_db.py`.

OUT of scope — do NOT build:
- Auth, user accounts, multi-tenant.
- Order execution / brokerage integration. Reports are advisory only.
- Real-time streaming, websockets, intraday tick data.
- Multiple tickers / basket optimization per query.
- LLM provider is Groq (Llama). Single provider — no multi-provider abstraction.
- Containerization, cloud deploy, CI beyond local.

---

## Conventions

- Source under `src/marketmind/`; run with `src/` on `PYTHONPATH` (`pip install -e .`
  later, or `cd src`). Import as `marketmind.<module>`.
- `config.py` is the only place that reads env vars. Everything imports from it.
- `state.py` (Contract A) is the only definition of inter-node data shapes.
- Agents never touch yfinance/SQLite directly — only through MCP tools.
- Each report claim must carry a citation to the agent that produced it.
- Python 3.10. Type-hint public functions.

---

## State Schema (Contract A — see `src/marketmind/state.py`)

```python
QuantResult:     signal(BUY|HOLD|SELL), confidence(0-1), rsi_14, sma_50,
                 last_close, above_sma_50, rationale
SentimentResult: label(positive|neutral|negative), score(VADER -1..1),
                 headline_count, summary
RiskResult:      level(low|moderate|high), current_weight, projected_weight, note

MarketMindState (total=False):
  query, ticker, proposed_notional(default 10000),
  quant, sentiment, risk,
  run_sentiment (router flag, set after quant node),
  report_markdown, citations[{claim, agent}], errors[str]
```

---

## MCP Tool Signatures (six tools)

### Market Data MCP — FastMCP, Streamable HTTP, port 8001, `/mcp`
```
get_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> dict
    -> {"ticker", "rows": [{"date","open","high","low","close","volume"}, ...]}

get_technicals(ticker: str) -> dict
    -> {"ticker","rsi_14","sma_50","sma_200","last_close","above_sma_50","pct_from_sma_50"}
```

Extension (personal scanner, USA only, TA-Lib backed — see `src/marketmind/scanner.py`):
```
scan_signals(ticker: str) -> dict     # RS-high + 6-condition buy signal vs S&P 500
    -> {"symbol","market":"usa","rs_high","buy_signal",
        "details": {"symbol","price","change_pct","volume","avg_volume",
                    "volume_ratio","rs_value"},
        "error"}
```

### News MCP — FastMCP, Streamable HTTP, port 8002, `/mcp`
```
get_recent_news(ticker: str, limit: int = 12) -> dict
    -> {"ticker","articles": [{"title","publisher","link","published","summary"}, ...]}

score_sentiment(headlines: list[str]) -> dict     # VADER
    -> {"compound","label","per_headline": [{"text","compound"}, ...]}
```

### Portfolio MCP — FastMCP, Streamable HTTP, port 8003, `/mcp`
```
get_holdings(account_id: str = "default") -> dict
    -> {"holdings": [{"ticker","shares","cost_basis","market_value","weight","sector"}, ...],
        "total_value", "by_sector": {sector: weight}}

assess_position_risk(ticker: str, proposed_notional: float, account_id: str = "default") -> dict
    -> {"current_weight","projected_weight","sector","sector_exposure_after",
        "concentration_level","note"}
```

---

## Layout

```
marketmind/
├── CLAUDE.md                 # this file
├── .env.example
├── requirements.txt
├── README.md
├── Makefile
├── data/                     # SQLite db lives here (gitignored)
├── scripts/
│   ├── seed_db.py
│   ├── run_servers.sh
│   ├── test_quant.py
│   └── run_pipeline.py
└── src/marketmind/
    ├── config.py
    ├── state.py              # Contract A
    ├── portfolio_db.py
    ├── servers/              # market_data_server, news_server, portfolio_server
    ├── agents/               # factory, prompts, nodes
    ├── graph/                # orchestrator (StateGraph + run_analysis)
    └── app/                  # streamlit_app
```

## Quant Signal — Script-First (see `src/marketmind/quant_signal.py`)

The Quant call is no longer left entirely to the LLM. `quant_signal.compute_signal`
turns a `get_technicals` payload into a deterministic BUY/HOLD/SELL + confidence
from the scanner conditions (buy_signal, rs_high, SMA/EMA trend, RSI). The quant
node fetches technicals via MCP, computes this **prior**, and injects it into the
agent prompt as the default. The LLM may override it but must say why; the result
records `script_signal`, `script_confidence`, and `overridden` on the quant dict.
This is the single definition of "what the script decides" — reused by backtest.

## Backtest & Training (extension — `src/marketmind/backtest/`)

Offline stack; no servers needed (yfinance direct). Heavy deps in
`requirements-train.txt` (`make install-train`).

- `features.py` — walk-forward indicator/condition table mirroring `scanner.py`.
- `engine.py`   — trade simulator + metrics for the script signal (`make backtest`).
- `dataset.py`  — labeled CSV (ML) + chat JSONL (LLM SFT) from forward returns (`make dataset`).
- `ml_model.py` — scikit-learn gradient-boosted classifier (`make train-ml`).
- `llm_finetune.py` — LoRA/QLoRA SFT of Qwen/Llama on the JSONL (`make train-llm`, GPU).

Artifacts under `data/` (gitignored): `data/training/`, `data/models/`.

**Live wiring:** the quant node calls `ml_model.predict_from_tech` and injects the
result as an ADVISORY **ML SECOND OPINION** alongside the SCRIPT PRIOR (returns
None → block skipped if no `data/models/quant_clf.joblib` yet). Quant dict gains
`ml_signal`, `ml_confidence`, `ml_proba`. To feed it the same features as
training, `get_technicals` now also returns `plus_di`, `weekly_rsi`, `cond1..6`
(via `scanner` result `conditions`). The LoRA/QLoRA LLM stays offline eval only.

## Claude Trader — Paper Trading (extension — `src/marketmind/paper_trader.py`)

A VIRTUAL paper-trading agent (advisory only, no broker). Starts with $500 cash
under `account_id="claude"` and trades a small US watchlist OFFLINE — yfinance
direct via `backtest.features`, scored by `quant_signal.compute_signal` + advisory
`ml_model` vote. No MCP servers / Groq needed. Each run fetches fresh prices,
sells held names that flip to SELL, buys fresh BUY candidates (ranked by
confidence, capped at `MAX_POS_FRAC` of starting cash, `MAX_POSITIONS` max),
marks to market, and reports total value / P&L / return %.

State persists in the same SQLite db (`data/marketmind.db`), new tables:
`paper_account`, `paper_positions` (now also `entry_tech` JSON + `entry_ts`),
`paper_trades` (ISO-8601 UTC timestamps). Surfaced as the **Claude Trader** tab in
`app/streamlit_app.py`; headless via `scripts/run_paper_trader.py` (`make paper`).

**Learning loop (retrain from own trades).** BOTH learners update from realized
P&L. On BUY, the entry `tech` features are stored on the position; on SELL,
`_record_outcome` writes the entry features + realized return + a label (`BUY` if ≥
`OUTCOME_UP_THRESH` +5%, `SELL` if ≤ `OUTCOME_DOWN_THRESH` −5%, else `HOLD`) to
table `paper_trade_outcomes`. `retrain_from_trades` then does two things:
1. **ML** — rebuilds outcomes into a feature/label frame (`build_outcomes_df`,
   columns = ml `FEATURE_COLUMNS` + label) and MERGES them with the historical
   `data/training/dataset.csv` before calling `ml_model.train`.
2. **Script** — `script_tuner.tune_from_trades` replays each closed trade's entry
   features through `compute_signal`, computes per-bull-condition win rates, and
   writes `data/models/script_params.json`: condition weights
   `clamp(win_rate/0.5, 0.25, 1.25)` (n ≥ 5 per condition) + a `min_buy_score`
   gate (2.0 if overall win rate < 45%, 1.5 if < 55%, else 1.0). `compute_signal`
   loads these params on every call: a BUY whose weighted bull score falls under
   the gate is downgraded to HOLD. Missing/default params = untuned behavior, so
   the fixed-script baseline is preserved on fresh checkouts. Tuning is
   recomputed from scratch each run (idempotent, bounded — dampens, never disables).

Needs ≥`min_outcomes` (10) decisive closed trades. Run via
`make retrain-from-trades` / `scripts/retrain_from_trades.py`, or the "Retrain ML
from trades" button in the Claude Trader tab. Outcomes survive `reset_paper`.

**News sentiment (live feature).** `paper_trader.evaluate` calls
`news_sentiment.ticker_sentiment` (offline: yfinance headlines + VADER, degrades
to neutral 0.0 on failure) and merges `news_sentiment` (compound −1..1) +
`news_count` into the tech dict. It feeds three places: (a) `compute_signal` —
`news_positive` bull / `news_negative` bear condition at ±0.2, and a hard BUY→HOLD
veto at ≤ −0.3; (b) the ML feature vector — `news_sentiment` is in
`FEATURE_COLUMNS` (historical rows backfilled 0.0-neutral at train time; live
closed trades carry the real entry value, so the retrained model learns its
effect); (c) the stored entry features for the learning loop. Old model
artifacts with the 19-feature schema return None from `predict_from_tech`
(advisory skipped) until retrained.

## Build Order (next, not yet done)

1. `portfolio_db.py` + `seed_db.py` (SQLite schema + sample holdings).
2. Three servers, one at a time; smoke-test each over HTTP.
3. `agents/factory.py` (scoped MCP tool loading) + `prompts.py`.
4. `agents/nodes.py` (intake, quant, sentiment, risk) + `graph/orchestrator.py`.
5. `scripts/run_pipeline.py` end-to-end, then `app/streamlit_app.py`.
