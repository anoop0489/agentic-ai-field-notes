# =============================================================================
# LANGSMITH TRACING & OBSERVABILITY: Debug the Black Box
# =============================================================================
# Demonstrates how LangSmith traces work without changing your code:
#   - Set env vars → tracing activates automatically
#   - Every chain step is logged: prompt rendering, LLM call, parsing
#   - View exact inputs, outputs, latency, and token costs in the UI
#
# WHAT IT DOES:
#   Runs a simple LCEL chain (template → model → parser) with LangSmith
#   tracing enabled. After running, check smith.langchain.com to see the
#   full execution trace.
#
# KEY CONCEPTS:
#   - LANGCHAIN_TRACING_V2=true: Master switch for tracing
#   - LANGCHAIN_API_KEY: Your LangSmith API key
#   - LANGCHAIN_PROJECT: Groups traces together in the UI
#   - Tuples vs Objects: ("human", "{var}") vs HumanMessage(content=f"{var}")
#
# HOW TRACING WORKS (Callback Architecture):
#   1. LangChain detects LANGCHAIN_TRACING_V2=true in env
#   2. Injects a LangChainTracer into the callback manager
#   3. Events fire: on_chain_start, on_llm_start, on_llm_end
#   4. Tracer sends HTTP POST to LangSmith API (non-blocking, async)
#   → Your code runs at normal speed; tracing happens in background
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true
#   2. LangSmith account at https://smith.langchain.com/
#   3. Packages: uv add langchain-core langchain-openai langsmith python-dotenv truststore
#
# USAGE:
#   uv run 02-gist-of-langchain/src/03_langsmith_tracing.py
#   → Then check https://smith.langchain.com/ to see the trace
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "reference-guides"))
from logger import log_header, log_info, log_success, log_error, log_warning

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def check_prerequisites():
    """Verify required env vars exist before making API calls."""
    errors = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        errors.append("LANGCHAIN_API_KEY")

    if errors:
        for var in errors:
            log_error(f"{var} not found in .env")
        sys.exit(1)

    # Report tracing status
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower()
    project = os.environ.get("LANGCHAIN_PROJECT", "(default)")

    log_success("Prerequisites met!")
    if tracing == "true":
        log_success(f"LangSmith tracing: ENABLED (project: {project})")
    else:
        log_warning("LangSmith tracing: DISABLED (set LANGCHAIN_TRACING_V2=true to enable)")


def main():
    check_prerequisites()

    log_header("LANGSMITH TRACING: Observability Without Code Changes")

    # ========================= THE DATA =========================
    information = "LangChain is a framework for developing applications powered by language models."
    log_info(f"Input data: \"{information[:50]}...\"")

    # ========================= THE TEMPLATE =========================
    # CRITICAL: We use TUPLES here (not HumanMessage objects)
    # Tuples are "schemas" — LangChain injects variables at runtime
    # HumanMessage objects "lock" the string immediately (see Glossary #5)
    messages = [
        ("system", "You are a helpful AI tutor. Summarize the following concept in exactly one sentence."),
        ("human", "{information}"),
    ]

    # .from_messages() is a Factory Method (see Glossary #2)
    chat_template = ChatPromptTemplate.from_messages(messages)
    log_info("Template created with roles: system, human")

    # ========================= THE MODEL =========================
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    log_info("Model: gpt-4o (temperature=0)")

    # ========================= THE PARSER =========================
    # Extracts .content string from AIMessage object
    # Without this, you get the full AIMessage (with response_metadata, usage, etc.)
    parser = StrOutputParser()
    log_info("Parser: StrOutputParser (extracts plain text from AIMessage)")

    # ========================= THE CHAIN =========================
    # LCEL pipe: template → model → parser
    # Each | triggers __or__ (operator overloading, see Glossary #3)
    chain = chat_template | llm | parser
    log_info("Chain built: chat_template | llm | parser")

    # ========================= EXECUTION =========================
    log_header("EXECUTING CHAIN (trace sent to LangSmith)")
    log_info("Behind the scenes:")
    log_info("  1. LangChain finds {information} placeholder in tuple")
    log_info("  2. Injects our string value from the dictionary")
    log_info("  3. Creates HumanMessage object")
    log_info("  4. Sends to OpenAI API")
    log_info("  5. Receives AIMessage → parser extracts .content")
    log_info("  6. Trace data sent to LangSmith (async, non-blocking)")
    print()

    response = chain.invoke(input={"information": information})

    log_success("Response received!")
    print()
    print("=" * 60)
    print(response)
    print("=" * 60)
    print()

    # ========================= WHAT TO CHECK IN LANGSMITH =========================
    log_header("NEXT STEPS: Check LangSmith UI")
    log_info("Go to: https://smith.langchain.com/")
    log_info("Look for:")
    log_info("  • Exact rendered prompt (after variable injection)")
    log_info("  • Token usage (input + output tokens = cost)")
    log_info("  • Latency per step (template rendering vs LLM call)")
    log_info("  • The full nested trace tree (template → LLM → parser)")


if __name__ == "__main__":
    main()
