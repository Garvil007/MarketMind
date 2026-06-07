"""Agent factory: ChatGroq + scoped MCP tool loading.

The scoping rule: an agent connects to exactly ONE MCP server and receives only
that server's tools. get_scoped_tools is the single choke point that enforces it.
Importing this module also imports config, which activates LangSmith tracing.
"""
from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from marketmind import config  # noqa: F401 - side effect: sets LangSmith env vars

# One shared model instance for all agents; deterministic output for parsable JSON.
# Groq serves Llama models (free tier) with an OpenAI-compatible tool-calling API.
MODEL = ChatGroq(
    model=config.AGENT_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)


def _client_for(server_name: str) -> MultiServerMCPClient:
    """Build a client connected to ONLY the named server's URL."""
    if server_name not in config.MCP_URLS:
        raise ValueError(f"Unknown server '{server_name}'. Known: {list(config.MCP_URLS)}")
    url = config.MCP_URLS[server_name]
    # User-facing transport name is "http"; the adapter's literal is "streamable_http".
    return MultiServerMCPClient({server_name: {"transport": "streamable_http", "url": url}})


async def get_scoped_tools(server_name: str):
    """Connect to one MCP server and return its LangChain tools (and no others).

    Args:
        server_name: Key in config.MCP_URLS ("market_data" | "news" | "portfolio").

    Returns:
        list[BaseTool] for that server only.
    """
    client = _client_for(server_name)
    return await client.get_tools(server_name=server_name)


async def build_agent(server_name: str, system_prompt: str):
    """Build a LangGraph react agent scoped to one server's tools.

    Args:
        server_name: Which MCP server to scope the agent to.
        system_prompt: The agent's system prompt.

    Returns:
        A compiled react agent (call via .ainvoke({"messages": [...]})).
    """
    tools = await get_scoped_tools(server_name)
    return create_react_agent(MODEL, tools, prompt=system_prompt)
