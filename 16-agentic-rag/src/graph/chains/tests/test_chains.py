# ─────────────────────────────────────────────────────────────────────────────
# test_chains.py — Unit tests for Agentic RAG chains
# ─────────────────────────────────────────────────────────────────────────────
# Tests each grader chain independently before running the full graph.
# This is a production best practice: if the full graph fails, these tests
# pinpoint exactly which chain broke.
#
# Run:
#   cd 16-agentic-rag/src
#   uv run pytest graph/chains/tests/test_chains.py -s -v
# ─────────────────────────────────────────────────────────────────────────────

import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


# ─── Retrieval Grader Tests ──────────────────────────────────────────────────

def test_retrieval_grader_answer_yes() -> None:
    """A document about agents IS relevant to 'agent memory'."""
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    """A document about agents is NOT relevant to 'how to make pizza'."""
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )
    assert res.binary_score == "no"


# ─── Generation Chain Tests ──────────────────────────────────────────────────

def test_generation_chain() -> None:
    """Generation chain produces a non-empty answer from relevant docs."""
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(f"\nGeneration: {generation}")
    assert len(generation) > 0


# ─── Hallucination Grader Tests ──────────────────────────────────────────────

def test_hallucination_grader_answer_yes() -> None:
    """A generation FROM the docs should be grounded (not hallucinated)."""
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    """A pizza recipe answer is NOT grounded in agent docs (hallucinated)."""
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


# ─── Router Tests ────────────────────────────────────────────────────────────

def test_router_to_vectorstore() -> None:
    """Questions about agents should route to vectorstore."""
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    """Off-topic questions should route to web search."""
    question = "how to make pizza"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
