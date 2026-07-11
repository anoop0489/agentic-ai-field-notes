# ─────────────────────────────────────────────────────────────────────────────
# generation.py — RAG generation chain
# ─────────────────────────────────────────────────────────────────────────────
# The actual answer generation step. Uses the standard rlm/rag-prompt from
# LangChain Hub which formats documents + question into a clean RAG prompt.
#
# This is the ONLY chain that returns freeform text (not structured output).
# All other chains use with_structured_output for binary yes/no grading.
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

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ─── Chain setup ─────────────────────────────────────────────────────────────
# hub.pull("rlm/rag-prompt") → community RAG prompt that formats {context} + {question}
# StrOutputParser() → extracts raw text from the AIMessage (no structured output needed here)
llm = ChatOpenAI(temperature=0)
prompt = hub.pull("rlm/rag-prompt")

# ─── Exported chain ──────────────────────────────────────────────────────────
generation_chain = prompt | llm | StrOutputParser()
