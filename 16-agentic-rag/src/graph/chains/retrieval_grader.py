# ─────────────────────────────────────────────────────────────────────────────
# retrieval_grader.py — Document relevance grading chain
# ─────────────────────────────────────────────────────────────────────────────
# Given a retrieved document and the user question, grades whether the
# document is relevant ("yes") or not ("no"). Used by the grade_documents
# node to filter out irrelevant chunks before generation.
#
# Credit: The grading prompt and GradeDocuments schema are adapted from
# LangChain's official open-source LangGraph Self-RAG / CRAG tutorial.
#
# Key technique:
#   with_structured_output(GradeDocuments) → the LLM MUST respond with
#   binary_score = "yes" or "no". No freeform rambling allowed.
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
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ─── Structured output schema ────────────────────────────────────────────────
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ─── Chain setup ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# ─── Exported chain ──────────────────────────────────────────────────────────
retrieval_grader = grade_prompt | structured_llm_grader
