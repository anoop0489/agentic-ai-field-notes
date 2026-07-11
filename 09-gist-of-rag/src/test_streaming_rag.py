# =============================================================================
# STREAMING RAG: Token-by-Token Response Demo
# =============================================================================
# This example demonstrates the difference between .invoke() and .stream()
# on the SAME LCEL chain. No code change to the chain itself — just how
# you consume the output.
#
# THE PROBLEM:
#   With .invoke(), the user stares at a blank screen for 5-10 seconds
#   while the LLM generates the full response, then everything appears at once.
#
# THE SOLUTION:
#   With .stream(), tokens arrive one at a time as the LLM generates them.
#   The first token appears within 200-500ms. The user sees progress instantly.
#
# THIS SCRIPT SHOWS:
#   1. Building an LCEL RAG chain (same as main.py)
#   2. Running it with .invoke() — full wait, then dump
#   3. Running it with .stream() — token by token
#   4. Timing both to show the "time to first token" difference
#
# PREREQUISITES:
#   - Run test_multimodal_pdf_rag.py first to populate Pinecone
#   - .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#
# USAGE:
#   uv run test_streaming_rag.py
# =============================================================================

import os
import sys
import time

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()


def check_prerequisites():
    errors = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.environ.get("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY not found in .env")
    if not os.environ.get("INDEX_NAME"):
        errors.append("INDEX_NAME not found in .env")
    if errors:
        print("❌ Prerequisites check FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    check_prerequisites()

    QUERY = "How is the Parcel events flown from the Hub to the CTS System?"

    print("=" * 70)
    print("🔄 STREAMING RAG DEMO — .invoke() vs .stream()")
    print("=" * 70)
    print(f"   Index: {os.environ['INDEX_NAME']}")
    print(f"   Query: {QUERY}")
    print()

    # ---- Build the LCEL chain (SAME chain used for both demos) ----
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
    )

    # This is the SAME chain for both .invoke() and .stream()
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # ================================================================
    # DEMO 1: .invoke() — Full wait, then dump
    # ================================================================
    print("=" * 70)
    print("1️⃣  .invoke() — User waits for FULL response")
    print("=" * 70)

    start = time.time()
    result = chain.invoke({"question": QUERY})
    elapsed = time.time() - start

    print(result)
    print(f"\n⏱️  Total time: {elapsed:.1f}s (user saw NOTHING until now)")
    print()

    # ================================================================
    # DEMO 2: .stream() — Token by token
    # ================================================================
    print("=" * 70)
    print("2️⃣  .stream() — User sees tokens as they arrive")
    print("=" * 70)

    start = time.time()
    first_token_time = None

    for chunk in chain.stream({"question": QUERY}):
        if first_token_time is None:
            first_token_time = time.time() - start
        print(chunk, end="", flush=True)

    elapsed = time.time() - start
    print(f"\n\n⏱️  First token: {first_token_time:.1f}s | Total: {elapsed:.1f}s")
    print(f"   User saw the first word {elapsed - first_token_time:.1f}s EARLIER than with .invoke()!")

    print()
    print("=" * 70)
    print("KEY TAKEAWAY:")
    print("   Same chain, same result. Only the consumption method changed.")
    print("   .invoke() → user waits for everything")
    print("   .stream() → user sees progress immediately")
    print("=" * 70)
