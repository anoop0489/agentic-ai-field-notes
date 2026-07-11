# =============================================================================
# STREAMLIT CHAT UI: Documentation Assistant Frontend
# =============================================================================
# This is the user-facing chat interface for the documentation assistant.
# It uses Streamlit to create a web-based chat UI with:
#   - Chat message history (persisted via st.session_state)
#   - Source citations (collapsible per message)
#   - Clear chat button in the sidebar
#
# HOW STREAMLIT WORKS:
#   Streamlit reruns this ENTIRE script top-to-bottom on every interaction.
#   That means:
#   - Local variables RESET on every click/submit
#   - Only st.session_state PERSISTS between reruns
#   - The message display loop re-renders ALL messages each time
#
#   This is fundamentally different from React/Angular where state lives
#   in components. It's more like a Razor Page that re-renders on every POST
#   but uses Session to remember values.
#
# PREREQUISITES:
#   1. Run ingestion.py first (populates the vector store)
#   2. .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   3. Packages: uv add streamlit
#
# USAGE:
#   streamlit run main.py
#   → Opens http://localhost:8501 in your browser
# =============================================================================

import os
import sys
from typing import Any, Dict, List

import streamlit as st

# Add src directory to path so we can import backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Extract source URLs from Document objects for display."""
    return [
        str(meta.get("source") or "Unknown")
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


# ========================= PAGE CONFIG =========================
# Must be the FIRST Streamlit command (before any other st.* calls)
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")


# ========================= SIDEBAR =========================
with st.sidebar:
    st.subheader("Session")
    # Clear chat button — resets session_state and forces a rerun
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


# ========================= SESSION STATE (MEMORY) =========================
# st.session_state is a per-user dictionary that persists across reruns.
# Without this, chat messages would vanish on every interaction because
# Streamlit reruns the entire script each time.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. "
            "I'll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]


# ========================= DISPLAY MESSAGE HISTORY =========================
# This loop runs on EVERY rerun — it re-renders all previous messages.
# Each message is a dict: {"role": "user"|"assistant", "content": str, "sources": list}
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show source citations in a collapsible expander
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


# ========================= HANDLE USER INPUT =========================
# st.chat_input renders a fixed-position text box at the bottom (like ChatGPT)
# Returns None until the user submits, then returns the text
prompt = st.chat_input("Ask a question about LangChain…")

if prompt:
    # 1. Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response using the agentic RAG backend
    with st.chat_message("assistant"):
        try:
            # st.spinner shows a loading indicator while we wait for the LLM
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            # Display the answer
            st.markdown(answer)

            # Display source citations if we have them
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # 3. Save assistant message to session state (persists for next rerun)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
