# ─────────────────────────────────────────────────────────────────────────────
# ingestion.py — Agentic RAG Document Ingestion Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Loads web pages (Lilian Weng blog posts), splits into chunks, and stores
# in a ChromaDB vector store with OpenAI embeddings.
#
# Run ONCE to create the vector store, then the retriever is importable
# by the graph's retrieve node.
#
# How to run:
#   cd 16-agentic-rag/src
#   uv run python ingestion.py
#
# Prerequisites:
#   - OPENAI_API_KEY in .env
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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

# ─── Configuration ───────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./.chroma"
COLLECTION_NAME = "rag-chroma"

# Three blog posts that form our "knowledge base" for this project.
# Topics: LLM agents, prompt engineering, adversarial attacks on LLMs.
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


# ─── Validate env vars before doing anything ─────────────────────────────────
def check_prerequisites():
    required = ["OPENAI_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("   Add them to your .env file")
        sys.exit(1)


# ─── Ingestion function (run once) ───────────────────────────────────────────
def ingest_documents():
    """Load web pages, chunk with tiktoken, embed into ChromaDB."""
    print("Loading documents from web...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    print(f"   Loaded {len(docs_list)} documents")

    # 250 tiktoken tokens per chunk, no overlap.
    # Small chunks = more precise retrieval for grading.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"   Split into {len(doc_splits)} chunks")

    print("Embedding and storing in ChromaDB...")
    Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print("Ingestion complete!")


# ─── Retriever (importable by other modules) ─────────────────────────────────
# This runs at import time so the retrieve node can just do:
#   from ingestion import retriever
retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_prerequisites()
    ingest_documents()
