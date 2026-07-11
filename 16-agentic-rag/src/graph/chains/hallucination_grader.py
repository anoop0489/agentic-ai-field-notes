# ─────────────────────────────────────────────────────────────────────────────
# hallucination_grader.py — Generation grounding check (Self-RAG)
# ─────────────────────────────────────────────────────────────────────────────
# After generation, this chain checks whether the LLM's answer is actually
# SUPPORTED by the retrieved documents. Returns True if grounded, False if
# the LLM hallucinated (made up content not in the docs).
#
# This is the first half of the Self-RAG verification loop.
# If this fails → the graph retries generation (same docs, new attempt).
#
# Credit: The grading prompt and GradeHallucinations schema are adapted from
# LangChain's official open-source LangGraph Self-RAG tutorial.
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ─── Structured output schema ────────────────────────────────────────────────
# Note: binary_score is a BOOL here (not str). True = grounded, False = hallucinated.
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# ─── Chain setup ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# ─── Exported chain ──────────────────────────────────────────────────────────
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
