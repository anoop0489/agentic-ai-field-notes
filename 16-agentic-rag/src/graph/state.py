# ─────────────────────────────────────────────────────────────────────────────
# state.py — Typed state for the Agentic RAG graph
# ─────────────────────────────────────────────────────────────────────────────
# The graph state flows through every node and conditional edge.
# Each node receives the full state and returns a PARTIAL update
# (only the keys it changed). LangGraph merges automatically.
# ─────────────────────────────────────────────────────────────────────────────

from typing import List, TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[str]
