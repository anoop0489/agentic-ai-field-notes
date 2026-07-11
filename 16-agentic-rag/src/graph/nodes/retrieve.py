# ─────────────────────────────────────────────────────────────────────────────
# retrieve.py — Vectorstore retrieval node
# ─────────────────────────────────────────────────────────────────────────────
# Queries ChromaDB with the user's question and returns matching documents.
# The retriever is imported from ingestion.py (created once during ingestion).
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Pull relevant documents from ChromaDB for the user's question."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
