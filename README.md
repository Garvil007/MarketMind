# MarketMind

Multi-agent financial research MVP.

Three MCP servers (market data, news, portfolio) feed a LangGraph orchestrator
that runs quant, sentiment, and risk agents, then produces a cited markdown report.

## Status

Scaffolding only. Servers, agents, graph, and frontend not yet implemented.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in keys
```

## Layout

See `CLAUDE.md` for architecture, scope boundary, state schema, and MCP tool signatures.
