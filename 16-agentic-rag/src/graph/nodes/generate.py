# ─────────────────────────────────────────────────────────────────────────────
# generate.py — Answer generation node
# ─────────────────────────────────────────────────────────────────────────────
# Takes filtered/augmented documents + question and generates an answer
# using the rlm/rag-prompt from LangChain Hub via the generation chain.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate an answer from the graded documents + question."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
