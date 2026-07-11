# ─────────────────────────────────────────────────────────────────────────────
# schemas.py — Pydantic models for structured LLM output (Reflexion Agent)
# ─────────────────────────────────────────────────────────────────────────────
# These schemas define the STRUCTURED OUTPUT that the LLM must return.
# By using tool_choice="AnswerQuestion", we force the LLM to always return
# data in this exact shape — no freeform text, only structured JSON.
#
# The key insight: the LLM doesn't just answer the question — it also
# produces its own CRITIQUE and SEARCH QUERIES in the same response.
# This is what makes it a "reflexion" agent: self-critique is baked into
# the output format itself.
# ─────────────────────────────────────────────────────────────────────────────

from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """Self-critique produced by the LLM alongside its answer.
    Forces the LLM to identify what's missing and what's unnecessary."""

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """The structured output for the FIRST response.
    The LLM must produce: answer + self-critique + search queries.

    The search_queries field is the bridge to the tool execution step —
    these queries will be sent to Tavily to fetch real-time data."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """The structured output for REVISED responses (extends AnswerQuestion).
    Adds a references field for citations from the search results.

    This inherits answer + reflection + search_queries from AnswerQuestion,
    plus adds citations so the revised answer is grounded in real data."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
