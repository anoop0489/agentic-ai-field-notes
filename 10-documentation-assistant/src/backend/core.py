# =============================================================================
# BACKEND: Agentic RAG with Tool-Calling Agent
# =============================================================================
# This module creates an agent that decides WHEN to search the documentation
# vector store. Unlike Section 9's deterministic RAG (always searches), this
# agent uses function calling to decide if retrieval is needed.
#
# KEY CONCEPTS:
#   - create_agent(): LangChain's modern factory for tool-calling agents
#   - @tool decorator: Converts a Python function into a LangChain Tool
#   - response_format="content_and_artifact": Returns TWO values:
#       1. Serialized text → what the LLM reads (ToolMessage.content)
#       2. Raw objects → what the app uses (ToolMessage.artifact)
#   - init_chat_model(): Provider-agnostic model initialization
#
# WHY AGENTIC HERE (vs deterministic in Section 9)?
#   A documentation assistant gets varied questions:
#   - "What is LangChain?" → Agent knows this, no search needed
#   - "Show me ChatOpenAI API" → Agent MUST search the docs
#   The agent decides — this is appropriate when the tool isn't always needed.
#
# PREREQUISITES:
#   1. Run ingestion.py first to populate the Pinecone index
#   2. .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#
# USAGE:
#   Called by main.py (Streamlit), or run standalone:
#   uv run backend/core.py
# =============================================================================

import os
import sys
from typing import Any

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# ========================= SETUP =========================
# Initialize embeddings — MUST match what was used in ingestion.py
# (text-embedding-3-small produces 1536-dimensional vectors)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to the existing Pinecone index (populated by ingestion.py)
# Hardcoded for Section 10 — ignores .env INDEX_NAME (that's Section 9's index)
vectorstore = PineconeVectorStore(
    index_name="doc-helper-index",
    embedding=embeddings,
)

# init_chat_model() is provider-agnostic — swap "gpt-4o"/"openai" to
# "claude-3-5-sonnet"/"anthropic" without changing any other code
model = init_chat_model("gpt-4o", model_provider="openai")
# =============================================================


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain.

    This tool searches the Pinecone vector store for the top-4 most relevant
    document chunks matching the query.

    The response_format="content_and_artifact" means this function returns TWO things:
    1. serialized (str) → becomes ToolMessage.content (what the LLM reads)
    2. retrieved_docs (list) → becomes ToolMessage.artifact (what the app uses)

    This separation lets the LLM reason over formatted text while the app
    retains access to raw Document objects for source citations.
    """
    # Retrieve top-4 most similar chunks from Pinecone
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)

    # Format for the LLM — include source URL for citation
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return BOTH: (text for LLM, raw docs for app)
    # The LLM sees 'serialized', the app accesses 'retrieved_docs' via artifact
    return serialized, retrieved_docs


def run_llm(query: str) -> dict[str, Any]:
    """
    Run the agentic RAG pipeline to answer a query.

    Flow:
    1. create_agent() builds an agent with our retrieval tool
    2. Agent receives the user query
    3. Agent decides: "Should I call retrieve_context?"
    4. If yes: calls the tool, reads the results, generates answer
    5. If no: answers directly from parametric knowledge
    6. We extract both the answer and source documents

    Args:
        query: The user's question about LangChain

    Returns:
        dict with:
        - "answer": The generated answer (str)
        - "context": List of retrieved Document objects (for source citations)
    """
    # System prompt tells the agent its role and when to use the tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # create_agent() is the modern LangChain pattern (replaces AgentExecutor)
    # It builds a tool-calling agent that uses function calling under the hood
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Invoke the agent with the user's message
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    # Extract the answer — it's always the last AI message
    answer = response["messages"][-1].content

    # Extract source documents from ToolMessage artifacts
    # When the agent calls retrieve_context, the second return value
    # (retrieved_docs) is stored as ToolMessage.artifact
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    # Quick test — run standalone to verify the backend works
    test_query = "What are LangChain agents and how do they work?"
    print(f"🔍 Query: {test_query}")
    print()

    result = run_llm(query=test_query)

    print("💡 Answer:")
    print(result["answer"])
    print()

    if result["context"]:
        print(f"📚 Sources ({len(result['context'])} documents):")
        for doc in result["context"]:
            print(f"   - {doc.metadata.get('source', 'Unknown')}")
