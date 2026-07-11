# =============================================================================
# DOCUMENTATION INGESTION: Crawl → Chunk → Embed → Store
# =============================================================================
# This script crawls a documentation website using Tavily, chunks the content
# with RecursiveCharacterTextSplitter, and stores embeddings in Pinecone.
#
# WHAT IT DOES:
#   1. Uses TavilyCrawl to crawl a documentation site (LangChain docs)
#   2. Converts crawled pages into LangChain Document objects
#   3. Splits documents into chunks (4000 chars, 200 overlap) using
#      RecursiveCharacterTextSplitter (hierarchical: paragraphs → lines → words)
#   4. Embeds chunks using OpenAI text-embedding-3-small
#   5. Stores vectors in Pinecone using async batch processing
#
# KEY CONCEPTS:
#   - TavilyCrawl: AI-native web crawler that returns clean text (not HTML)
#   - RecursiveCharacterTextSplitter: Tries multiple separators in order
#     (paragraphs → lines → words → chars) for semantically coherent chunks
#   - Async batch ingestion: Process 50 docs at a time concurrently
#     using asyncio.gather() (like Task.WhenAll() in C#)
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME, TAVILY_API_KEY
#   2. Pinecone index: dimensions=1536, metric=cosine, serverless
#   3. Packages: uv add langchain-openai langchain-pinecone langchain-tavily
#      langchain-text-splitters python-dotenv truststore
#
# USAGE:
#   uv run ingestion.py
#
# NOTE: This runs ONCE to populate the vector store. After running, you can
#       use main.py (Streamlit) or backend/core.py to query the indexed docs.
# =============================================================================

import asyncio
import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ========================= CONFIGURATION =========================
# The documentation site to crawl
CRAWL_URL = "https://python.langchain.com/"

# How deep to follow links from the start URL
MAX_DEPTH = 1

# Max pages to process (set to None for all pages, or a number for testing)
# For testing: 1-3 pages is enough to verify the pipeline works end-to-end
MAX_PAGES = 1

# Chunk size for splitting (4000 = keeps full code examples + explanation together)
CHUNK_SIZE = 4000

# Overlap between adjacent chunks (preserves context at boundaries)
CHUNK_OVERLAP = 200

# How many documents to send to Pinecone in one async batch
BATCH_SIZE = 50

# Pinecone index name — hardcoded for Section 10 (ignores .env INDEX_NAME
# which is used by Section 9). Create in Pinecone: dims=1536, cosine, serverless
INDEX_NAME = "doc-helper-index"
# =================================================================


def check_prerequisites():
    """Verify all required env vars exist before proceeding."""
    errors = []

    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.environ.get("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY not found in .env")
    # INDEX_NAME is optional — defaults to "doc-helper-index" if not set
    # (see CONFIGURATION section above)
    if not os.environ.get("TAVILY_API_KEY"):
        errors.append("TAVILY_API_KEY not found in .env")

    if errors:
        print("❌ Prerequisites check FAILED:")
        for e in errors:
            print(f"   - {e}")
        print("\nSetup steps:")
        print("  1. Get a Tavily API key at https://app.tavily.com/")
        print("  2. Create a Pinecone index: dimensions=1536, metric=cosine")
        print("  3. Add all keys to your .env file")
        sys.exit(1)

    print("✅ All prerequisites met!")
    print(f"   OpenAI key:  {os.environ['OPENAI_API_KEY'][:8]}...")
    print(f"   Pinecone:    {os.environ['PINECONE_API_KEY'][:8]}...")
    print(f"   Index:       {INDEX_NAME}")
    print(f"   Tavily key:  {os.environ['TAVILY_API_KEY'][:8]}...")
    print()


def crawl_documentation() -> list[Document]:
    """
    Phase 1: Crawl the documentation site with TavilyCrawl.

    TavilyCrawl traverses the website, follows links up to max_depth,
    and extracts clean text content from each page. No HTML parsing needed —
    Tavily handles JavaScript rendering, nav stripping, and content extraction.

    Returns a list of LangChain Document objects (one per crawled page).
    """
    print("=" * 60)
    print("📡 PHASE 1: WEB CRAWLING")
    print("=" * 60)
    print(f"   URL:       {CRAWL_URL}")
    print(f"   Max depth: {MAX_DEPTH}")
    print()

    tavily_crawl = TavilyCrawl()

    print("   🕸️  Crawling (this may take 30-60 seconds)...")
    res = tavily_crawl.invoke({
        "url": CRAWL_URL,
        "max_depth": MAX_DEPTH,
        "extract_depth": "advanced",
    })

    # Convert Tavily results to LangChain Documents
    # Each result has: {"url": "...", "raw_content": "clean text..."}
    all_docs = []
    for item in res["results"]:
        all_docs.append(
            Document(
                page_content=item["raw_content"],
                metadata={"source": item["url"]},
            )
        )

    print(f"   ✅ Crawled {len(all_docs)} pages")

    # Limit pages for testing (remove or set MAX_PAGES=None for full ingestion)
    if MAX_PAGES and len(all_docs) > MAX_PAGES:
        all_docs = all_docs[:MAX_PAGES]
        print(f"   🧪 TEST MODE: Limited to {MAX_PAGES} page(s)")

    if all_docs:
        print(f"   📋 Sample: {all_docs[0].metadata['source']}")
        print(f"      Content: \"{all_docs[0].page_content[:100]}...\"")
    print()
    return all_docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Phase 2: Split documents into chunks using RecursiveCharacterTextSplitter.

    Why Recursive over Character?
    - CharacterTextSplitter uses ONE separator (\n\n). If a paragraph is
      longer than chunk_size, it produces an oversized chunk.
    - RecursiveCharacterTextSplitter tries separators IN ORDER:
      \n\n (paragraphs) → \n (lines) → " " (words) → "" (chars)
      This guarantees chunks stay within size limits while preserving
      the largest possible semantic units.

    Why chunk_size=4000?
    - Documentation pages have code examples + explanations together
    - 4000 chars keeps a full example in one chunk (better retrieval)
    - GPT-4o can easily handle 4000-char chunks in its context

    Why chunk_overlap=200?
    - Answers that span a chunk boundary won't be lost
    - 200 chars ≈ 1-2 sentences of shared context
    """
    print("=" * 60)
    print("✂️  PHASE 2: CHUNKING")
    print("=" * 60)
    print(f"   Strategy:  RecursiveCharacterTextSplitter")
    print(f"   Size:      {CHUNK_SIZE} chars")
    print(f"   Overlap:   {CHUNK_OVERLAP} chars")
    print(f"   Input:     {len(documents)} documents")
    print()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Default separators: ["\n\n", "\n", " ", ""]
        # Tries paragraphs first, then lines, then words, then characters
    )
    chunks = text_splitter.split_documents(documents)

    print(f"   ✅ Created {len(chunks)} chunks from {len(documents)} pages")
    if chunks:
        print(f"   📋 Sample chunk ({len(chunks[0].page_content)} chars):")
        print(f"      \"{chunks[0].page_content[:120]}...\"")
    print()
    return chunks


async def store_embeddings(chunks: list[Document]):
    """
    Phase 3: Embed chunks and store in Pinecone using async batch processing.

    Why async?
    - Embedding + storing is I/O-bound (waiting for OpenAI + Pinecone APIs)
    - Sequential processing of 1000+ chunks takes forever
    - Async batching sends multiple batches concurrently:
      Instead of: Batch1 → wait → Batch2 → wait → Batch3 → wait
      We get:     Batch1 → Batch2 → Batch3 (all waiting in parallel!)

    Why batch_size=50?
    - Too small (10): many network round-trips, under-utilizes parallelism
    - Too large (500): one failure loses 500 docs, memory pressure
    - 50 is the sweet spot for most vector store APIs

    asyncio.gather(*tasks, return_exceptions=True):
    - Runs all batch uploads concurrently (like Task.WhenAll() in C#)
    - return_exceptions=True means: if one batch fails, others continue
      (without it, first failure cancels everything)
    """
    print("=" * 60)
    print("🧮 PHASE 3: EMBEDDING + STORING")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # Split chunks into batches
    batches = [
        chunks[i : i + BATCH_SIZE]
        for i in range(0, len(chunks), BATCH_SIZE)
    ]
    print(f"   Model:     text-embedding-3-small (1536 dims)")
    print(f"   Index:     {INDEX_NAME}")
    print(f"   Chunks:    {len(chunks)}")
    print(f"   Batches:   {len(batches)} (batch_size={BATCH_SIZE})")
    print()

    async def add_batch(batch: list[Document], batch_num: int) -> bool:
        """Process a single batch — called concurrently for all batches."""
        try:
            # aadd_documents is the ASYNC version of add_documents
            # The "a" prefix is LangChain's convention for async methods
            await vectorstore.aadd_documents(batch)
            print(f"   ✅ Batch {batch_num}/{len(batches)} — {len(batch)} docs stored")
            return True
        except Exception as e:
            print(f"   ❌ Batch {batch_num}/{len(batches)} FAILED: {e}")
            return False

    # Run ALL batches concurrently — this is the key performance win
    # asyncio.gather() is Python's equivalent of C#'s Task.WhenAll()
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count results
    successful = sum(1 for r in results if r is True)
    failed = len(batches) - successful

    print()
    if failed == 0:
        print(f"   ✅ All {successful} batches stored successfully!")
    else:
        print(f"   ⚠️  {successful}/{len(batches)} batches succeeded, {failed} failed")


async def main():
    """
    Main async orchestrator — runs the full ingestion pipeline.

    asyncio.run(main()) starts the event loop from synchronous code.
    This is Python's equivalent of `static async Task Main()` in C#.
    """
    print()
    print("🚀 DOCUMENTATION INGESTION PIPELINE")
    print("=" * 60)
    print()

    # Phase 1: Crawl
    documents = crawl_documentation()
    if not documents:
        print("❌ No documents crawled. Check your Tavily API key and URL.")
        sys.exit(1)

    # Phase 2: Chunk
    chunks = chunk_documents(documents)
    if not chunks:
        print("❌ No chunks created. Check your documents.")
        sys.exit(1)

    # Phase 3: Embed + Store (async)
    await store_embeddings(chunks)

    # Summary
    print()
    print("=" * 60)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   📊 Pages crawled:  {len(documents)}")
    print(f"   📊 Chunks created: {len(chunks)}")
    print(f"   📊 Vectors stored: {len(chunks)}")
    print()
    print("   Next: Run the Streamlit app to query the docs:")
    print("   → streamlit run main.py")


if __name__ == "__main__":
    check_prerequisites()
    # asyncio.run() creates an event loop, runs main(), then shuts down
    # This is the ONLY way to enter async code from sync Python
    asyncio.run(main())
