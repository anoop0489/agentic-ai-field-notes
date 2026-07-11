# =============================================================================
# CHAT MODELS ARCHITECTURE: Message Protocol & Stateless Memory
# =============================================================================
# Demonstrates the core Chat Model concepts:
#   - SystemMessage, HumanMessage, AIMessage roles
#   - LLMs are STATELESS — you must re-send full history every call
#   - "Memory" is an illusion created by appending previous outputs to input
#
# WHAT IT DOES:
#   Constructs a raw message array simulating a conversation and shows
#   how the model "remembers" only because we feed it the full history.
#
# KEY CONCEPTS:
#   - SystemMessage: "God Mode" instructions (persona, constraints, rules)
#   - HumanMessage: The user's input
#   - AIMessage: Previous model output (injected to simulate memory)
#   - The model does NOT remember you between calls
#
# C# ANALOGY:
#   Like a REST API endpoint that takes a JSON array of Message DTOs.
#   Because HTTP is stateless, you must pass the entire session history
#   on every request — the server forgets you after each response.
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY
#   2. Packages: uv add langchain-core langchain-openai python-dotenv truststore
#
# USAGE:
#   uv run 02-gist-of-langchain/src/02_chat_models_architecture.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "reference-guides"))
from logger import log_header, log_info, log_success, log_error, log_warning

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def check_prerequisites():
    """Verify required env vars exist before making API calls."""
    if not os.environ.get("OPENAI_API_KEY"):
        log_error("OPENAI_API_KEY not found in .env")
        sys.exit(1)
    log_success("Prerequisites met — OPENAI_API_KEY found")


def main():
    check_prerequisites()

    log_header("CHAT MODELS: Message Protocol & Stateless Memory")

    # ========================= INITIALIZE MODEL =========================
    # temperature=0.7: Slightly creative (good for conversational demos)
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
    log_info("Model initialized: gpt-4o (temperature=0.7)")

    # ========================= THE MESSAGE ARRAY =========================
    # This is the ENTIRE "memory" — if you remove any message, the model
    # loses that context. There is NO server-side session storing history.
    log_header("CONSTRUCTING MESSAGE HISTORY")

    messages = [
        # SYSTEM: Sets persona and behavior rules (persists across conversation)
        # C# equivalent: AppSettings or GlobalConfiguration for the AI
        SystemMessage(content="You are a sarcastic senior engineer who loves Python."),

        # HUMAN: The user's first input
        HumanMessage(content="I am writing a script to parse CSVs."),

        # AI: We "fake" memory by injecting what the model previously said
        # In production, this comes from a database (Redis, SQL, etc.)
        AIMessage(content="Oh, thrilling. Another CSV parser. Groundbreaking work."),

        # HUMAN: Follow-up — the model will understand context because of the above
        HumanMessage(content="Hey, be nice! How do I handle missing values with pandas?"),
    ]

    log_info("Message array constructed:")
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        # Truncate long messages for display
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        log_info(f"  [{role:>6}] {content_preview}")

    # ========================= EXECUTION =========================
    log_header("SENDING TO MODEL")
    log_info("Passing full message array to llm.invoke()...")
    log_warning("Remember: We pass a LIST of messages, not a single string!")

    # .invoke() accepts List[BaseMessage] — NOT a concatenated string
    # LangChain handles converting these Python objects into the JSON schema
    # that OpenAI's Chat Completions API expects
    response = llm.invoke(messages)

    # ========================= OUTPUT ANALYSIS =========================
    log_header("RESPONSE ANALYSIS")
    log_info(f"Response type: {type(response).__name__}")
    log_info(f"Response role: AI (always AIMessage for chat models)")
    log_success("Model response:")
    print()
    print(f"  {response.content}")
    print()

    # ========================= THE STATELESNESS PROOF =========================
    log_header("STATELESSNESS PROOF")
    log_warning("If we make a NEW call without the history, the model forgets everything:")
    log_info("Sending ONLY a follow-up question (no history)...")

    # This call has NO context — the model doesn't know about CSVs or pandas
    isolated_response = llm.invoke([
        HumanMessage(content="What was I just asking about?")
    ])
    log_info(f"  Response: {isolated_response.content[:100]}...")
    log_success("Proof: The model has NO memory of the previous conversation!")
    log_info("To simulate 'memory', you must re-send the full message array every time.")


if __name__ == "__main__":
    main()
