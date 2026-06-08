"""Central config. Loads environment via python-dotenv.

All runtime knobs live here so the rest of the package imports settings
rather than touching os.environ directly.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq, free Llama models) ---
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama-3.3-70b-versatile")


class ConfigError(RuntimeError):
    """Raised when required configuration (e.g. the LLM key) is missing."""


def require_groq_key() -> str:
    """Return the Groq API key, or fail fast with a clear, actionable message.

    Called at the LLM entry point (agents.factory) so a fresh checkout without a
    key gets a one-line fix instead of a deep provider stack trace.
    """
    if not GROQ_API_KEY:
        raise ConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your free "
            "Groq key (create one at https://console.groq.com/keys)."
        )
    return GROQ_API_KEY

# --- LangSmith tracing ---
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "marketmind")

# Mirror LangSmith vars into the names the SDK reads, so tracing "just works".
if LANGSMITH_TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# --- MCP server URLs ---
MARKET_DATA_MCP_URL: str = os.getenv("MARKET_DATA_MCP_URL", "http://localhost:8001/mcp")
NEWS_MCP_URL: str = os.getenv("NEWS_MCP_URL", "http://localhost:8002/mcp")
PORTFOLIO_MCP_URL: str = os.getenv("PORTFOLIO_MCP_URL", "http://localhost:8003/mcp")

MCP_URLS: dict[str, str] = {
    "market_data": MARKET_DATA_MCP_URL,
    "news": NEWS_MCP_URL,
    "portfolio": PORTFOLIO_MCP_URL,
}
