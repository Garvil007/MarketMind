"""LangGraph StateGraph wiring + run_analysis() entrypoint.

Topology (sequential + one conditional, no parallel branches):

    START -> ingest -> quant -> [run_sentiment?] --true--> sentiment -> risk
                                               \--false--------------> risk
            risk -> report -> END

The conditional after quant routes to sentiment only when the quant node flagged
run_sentiment (low confidence); otherwise it skips straight to risk. A MemorySaver
checkpointer persists state per thread_id. Importing nodes activates LangSmith.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from marketmind.agents import nodes
from marketmind.state import MarketMindState


def _route_after_quant(state: MarketMindState) -> str:
    """Conditional edge: run the sentiment branch only if the quant node asked for it."""
    return "sentiment" if state.get("run_sentiment") else "risk"


def build_graph() -> StateGraph:
    """Assemble the StateGraph (uncompiled) over MarketMindState."""
    g = StateGraph(MarketMindState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("quant", nodes.quant_node)
    g.add_node("sentiment", nodes.sentiment_node)
    g.add_node("risk", nodes.risk_node)
    g.add_node("report", nodes.report_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "quant")
    g.add_conditional_edges(
        "quant", _route_after_quant, {"sentiment": "sentiment", "risk": "risk"}
    )
    g.add_edge("sentiment", "risk")
    g.add_edge("risk", "report")
    g.add_edge("report", END)

    return g


async def run_analysis(query: str, thread_id: str = "demo") -> MarketMindState:
    """Compile the graph with a MemorySaver checkpointer and run one query end-to-end.

    Args:
        query: The user's natural-language question (must name a ticker).
        thread_id: Checkpoint thread key; reuse to continue a conversation.

    Returns:
        The final MarketMindState (includes report_markdown, citations, errors).
    """
    graph = build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke({"query": query}, config=config)


async def run_analysis_stream(query: str, thread_id: str = "demo"):
    """Stream the pipeline, yielding (node_name, accumulated_state) as each node finishes.

    Uses LangGraph's "updates" stream mode: each chunk is {node_name: partial_update}.
    We merge updates into a running state and yield the snapshot so far, so a UI can
    light up each agent card the moment its node completes.

    Yields:
        tuple[str, MarketMindState]: the node that just finished and the state after it.
    """
    graph = build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    state: dict = {"query": query}
    async for chunk in graph.astream({"query": query}, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if update:
                state.update(update)
            yield node_name, dict(state)
