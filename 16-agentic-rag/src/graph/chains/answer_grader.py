# ─────────────────────────────────────────────────────────────────────────────
# answer_grader.py — Answer relevance check (Self-RAG step 2)
# ─────────────────────────────────────────────────────────────────────────────
# After confirming the generation is grounded (not hallucinated), this chain
# checks whether the answer actually ADDRESSES the user's question.
# A grounded answer that doesn't answer the question is still useless.
#
# This is the second half of the Self-RAG verification loop.
# If this fails → the graph goes to web search (need different information).
#
# Credit: The grading prompt and GradeAnswer schema are adapted from
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
# Note: binary_score is a BOOL here (not str). True = useful, False = not useful.
class GradeAnswer(BaseModel):
    """Binary score for whether the answer addresses the question."""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# ─── Chain setup ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# ─── Exported chain ──────────────────────────────────────────────────────────
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
