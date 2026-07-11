# ─────────────────────────────────────────────────────────────────────────────
# router.py — Question routing chain (vectorstore vs web search)
# ─────────────────────────────────────────────────────────────────────────────
# Uses structured output to classify the user's question and decide whether
# to route it to the local vectorstore (agents, prompt eng, adversarial
# attacks) or to Tavily web search (everything else).
#
# This is the ADAPTIVE RAG entry point — the graph doesn't always start
# at the same node because of this router.
#
# Credit: The routing prompt and RouteQuery schema are adapted from
# LangChain's official open-source LangGraph Adaptive-RAG tutorial.
#
# Key technique:
#   with_structured_output(RouteQuery) forces the LLM to return one of two
#   choices ("vectorstore" or "websearch") — no freeform text.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from typing import Literal

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ───────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ─── Structured output schema ────────────────────────────────────────────────
# Pydantic model that constrains the LLM to ONLY return one of the two routes.
# The Literal type means the LLM physically cannot return anything else.
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# ─── Chain setup ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# System prompt tells the LLM what each route means.
# vectorstore = agents, prompt eng, adversarial attacks (our ingested corpus).
# websearch = anything else the corpus doesn't cover.
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# ─── Exported chain ──────────────────────────────────────────────────────────
question_router = route_prompt | structured_llm_router
