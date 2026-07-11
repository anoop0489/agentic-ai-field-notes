# ─────────────────────────────────────────────────────────────────────────────
# web_search.py — Tavily web search fallback node
# ─────────────────────────────────────────────────────────────────────────────
# Called when:
#   (1) Router decides question is off-topic for vectorstore, OR
#   (2) grade_documents finds irrelevant docs, OR
#   (3) answer_grader says generation doesn't address the question.
#
# Appends Tavily web results to existing documents so the generate node
# has richer context on the next pass.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ───────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

# ─── Tavily setup ────────────────────────────────────────────────────────────
# max_results=3 keeps web search focused; too many results dilute quality.
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """Run Tavily search and append results as a Document."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents") or []

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    # Join all result contents into one Document for the generation chain
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    documents.append(web_results)
    return {"documents": documents, "question": question}
