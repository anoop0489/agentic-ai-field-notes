# ─────────────────────────────────────────────────────────────────────────────
# graph.py — Full Agentic RAG graph definition
# ─────────────────────────────────────────────────────────────────────────────
# Assembles the StateGraph with 4 nodes and conditional edges:
#   - Conditional entry point (router decides vectorstore or web search)
#   - Document grading with web search fallback
#   - Self-RAG: hallucination check + answer relevance check after generation
#
# Architecture (Adaptive RAG + Self-RAG combined):
#
#                     +----------+
#                     |  START   |
#                     +----+-----+
#                          |
#                     +----v-----+
#                     |  ROUTER  | (LLM classifies question)
#                     +--+----+--+
#                        |    |
#             vectorstore|    |websearch
#                        |    |
#                  +-----v-+  |
#                  |RETRIEVE|  |
#                  +-----+--+  |
#                        |     |
#                  +-----v-----v-+
#                  | GRADE DOCS  |
#                  +--+-------+--+
#                     |       |
#             relevant|       |not relevant
#                     |       |
#                     |   +---v--------+
#                     |   | WEB SEARCH |
#                     |   +---+--------+
#                     |       |
#                  +--v-------v--+
#                  |  GENERATE   |<--- (retry if hallucinated)
#                  +------+------+
#                         |
#                  +------v------+
#                  | HALLUCINATE |
#                  | + RELEVANCE |
#                  +--+---+---+-+
#                     |   |   |
#              useful |   |   | not useful
#                     |   |   |
#              +------v+  |   +---> WEB SEARCH
#              |  END  |  |
#              +-------+  | not supported
#                         +---> GENERATE (retry)
#
# The compiled graph is exported as `app` for use by main.py.
#
# How to run:
#   cd 16-agentic-rag/src
#   uv run python main.py
#
# Prerequisites:
#   - OPENAI_API_KEY in .env
#   - TAVILY_API_KEY in .env
#   - ChromaDB ingested: uv run python ingestion.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ───────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


# ─── Conditional Edge: decide_to_generate ────────────────────────────────────
# After grading documents, decides whether we have enough relevant docs
# to generate an answer, or if we need to supplement with web search.
def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: DOCUMENTS NOT RELEVANT, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


# ─── Conditional Edge: Self-RAG check (hallucination + answer relevance) ─────
# Two-step verification AFTER generation:
#   1. Is the answer GROUNDED in the documents? (hallucination check)
#   2. Does the answer ADDRESS the user's question? (relevance check)
#
# Three outcomes:
#   "useful"        → grounded + relevant → END (success)
#   "not supported" → hallucinated → GENERATE again (retry)
#   "not useful"    → grounded but off-topic → WEBSEARCH (need different info)
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# ─── Conditional Entry Point: route_question ─────────────────────────────────
# This is the ADAPTIVE RAG entry point. Instead of always starting at RETRIEVE,
# the LLM classifies the question and decides:
#   - "vectorstore" → question is about agents/prompts/attacks → RETRIEVE
#   - "websearch"   → question is off-topic for our corpus → WEBSEARCH
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


# ─── Build the Graph ─────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# Register nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Entry: router decides where to start (Adaptive RAG)
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# Fixed edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(WEBSEARCH, GENERATE)

# After grading docs: generate if all relevant, web search if any irrelevant
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

# After generation: Self-RAG verification loop
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,   # Hallucinated → retry generation
        "useful": END,               # Good answer → done
        "not useful": WEBSEARCH,     # Doesn't answer question → get more info
    },
)

# ─── Compile ─────────────────────────────────────────────────────────────────
app = workflow.compile()
