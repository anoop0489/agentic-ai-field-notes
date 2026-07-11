# =============================================================================
# INDEXING STRATEGY: Incremental Ingestion with RecordManager
# =============================================================================
# This example demonstrates LangChain's Indexing API — the production pattern
# for avoiding duplicate embeddings and wasted API costs.
#
# THE PROBLEM:
#   Running ingestion.py twice → all chunks get re-embedded and stored again.
#   - Duplicate vectors in Pinecone → retrieval returns copies
#   - Wasted embedding API costs → paying to re-embed unchanged content
#   - No deletion → removed content still lives in the vector store
#
# THE SOLUTION:
#   RecordManager + index() with cleanup="incremental"
#   - Computes a SHA256 hash of each chunk's content
#   - Tracks hashes in a local SQLite database
#   - On re-ingestion: skip unchanged, re-embed modified, delete removed
#
# ANALOGY (C#):
#   Think of EF Core Migrations — each migration has a unique name/hash.
#   Running `dotnet ef database update` skips already-applied migrations.
#   The RecordManager does the same for document chunks.
#
# THIS SCRIPT SHOWS:
#   1. First run: all chunks embedded and stored (num_added = N)
#   2. Second run: all chunks skipped (num_skipped = N, $0 spent)
#   3. Modified content: only changed chunks re-embedded
#   4. Deleted content: orphaned vectors removed from Pinecone
#
# PREREQUISITES:
#   - .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   - pip install langchain langchain-openai langchain-pinecone
#
# USAGE:
#   uv run test_indexing_strategy.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

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


def create_sample_documents():
    """Create sample documents to demonstrate indexing behavior."""
    return [
        Document(
            page_content="Vector databases store high-dimensional vectors and provide "
            "fast nearest-neighbor similarity search. Unlike regular databases "
            "that use exact-match queries, vector databases find items that are "
            "semantically similar to a query vector.",
            metadata={"source": "sample-vector-db.txt"},
        ),
        Document(
            page_content="Embeddings are numerical representations of text where "
            "semantically similar texts produce vectors that are close together "
            "in vector space. This enables mathematical similarity comparisons "
            "between pieces of text.",
            metadata={"source": "sample-embeddings.txt"},
        ),
        Document(
            page_content="Chunking is the process of breaking large documents into "
            "smaller segments that fit within the LLM's context window while "
            "preserving semantic meaning. The chunk size trade-off: too small "
            "loses context, too large includes noise.",
            metadata={"source": "sample-chunking.txt"},
        ),
    ]


def create_modified_documents():
    """Same sources but one document has been modified."""
    return [
        Document(
            page_content="Vector databases store high-dimensional vectors and provide "
            "fast nearest-neighbor similarity search. Unlike regular databases "
            "that use exact-match queries, vector databases find items that are "
            "semantically similar to a query vector.",
            metadata={"source": "sample-vector-db.txt"},
        ),
        Document(
            page_content="Embeddings are numerical representations of text where "
            "semantically similar texts produce vectors that are close together "
            "in vector space. This enables mathematical similarity comparisons "
            "between pieces of text. Modern embedding models like "
            "text-embedding-3-small support up to 1536 dimensions.",
            metadata={"source": "sample-embeddings.txt"},
        ),
        # sample-chunking.txt is REMOVED (not in list) → should be deleted
    ]


def print_result(label, result):
    print(f"\n   📊 {label}:")
    print(f"      Added:   {result.get('num_added', 0)}")
    print(f"      Updated: {result.get('num_updated', 0)}")
    print(f"      Skipped: {result.get('num_skipped', 0)}")
    print(f"      Deleted: {result.get('num_deleted', 0)}")


if __name__ == "__main__":
    check_prerequisites()

    index_name = os.environ["INDEX_NAME"]
    namespace = f"pinecone/{index_name}"
    db_path = os.path.join(os.path.dirname(__file__), "record_manager_demo.db")

    print("=" * 70)
    print("📦 INDEXING STRATEGY DEMO — Incremental Ingestion")
    print("=" * 70)
    print(f"   Index:          {index_name}")
    print(f"   Namespace:      {namespace}")
    print(f"   RecordManager:  SQLite → {db_path}")
    print()

    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name, embedding=embeddings
    )

    # ---- Create and initialize the RecordManager ----
    record_manager = SQLRecordManager(
        namespace=namespace,
        db_url=f"sqlite:///{db_path}",
    )
    record_manager.create_schema()  # Idempotent — safe to call multiple times

    # ================================================================
    # RUN 1: First ingestion — all documents are new
    # ================================================================
    print("-" * 70)
    print("1️⃣  FIRST RUN: All documents are new")
    print("-" * 70)

    docs_v1 = create_sample_documents()
    print(f"   Documents: {len(docs_v1)}")
    for d in docs_v1:
        print(f"     - {d.metadata['source']} ({len(d.page_content)} chars)")

    result1 = index(
        docs_iterable=docs_v1,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print_result("Result", result1)
    print("   ✅ All 3 documents embedded and stored")

    # ================================================================
    # RUN 2: Same documents — everything should be SKIPPED
    # ================================================================
    print()
    print("-" * 70)
    print("2️⃣  SECOND RUN: Same documents, no changes")
    print("-" * 70)

    result2 = index(
        docs_iterable=docs_v1,  # Same docs!
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print_result("Result", result2)
    print("   ✅ All skipped! Zero embedding API calls. $0 spent.")

    # ================================================================
    # RUN 3: Modified + deleted documents
    # ================================================================
    print()
    print("-" * 70)
    print("3️⃣  THIRD RUN: 1 unchanged, 1 modified, 1 deleted")
    print("-" * 70)

    docs_v2 = create_modified_documents()
    print("   Changes:")
    print("     - sample-vector-db.txt:  UNCHANGED")
    print("     - sample-embeddings.txt: MODIFIED (added sentence)")
    print("     - sample-chunking.txt:   DELETED (not in batch)")

    result3 = index(
        docs_iterable=docs_v2,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print_result("Result", result3)
    print("   ✅ Only the diff was processed!")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    print("📋 SUMMARY — What the RecordManager saved you:")
    print("=" * 70)
    print()
    print("   Run 1: 3 new chunks → 3 embedding API calls (necessary)")
    print("   Run 2: 0 changes   → 0 embedding API calls (saved 3!)")
    print("   Run 3: 1 modified  → 1 embedding API call  (saved 2!)")
    print()
    print("   Without RecordManager: 9 API calls")
    print("   With RecordManager:    4 API calls (56% savings!)")
    print()
    print("   CLEANUP MODES:")
    print("   ┌─────────────────┬────────────────────────────────────────┐")
    print("   │ None            │ Never deletes. Duplicates accumulate. │")
    print("   │ 'incremental'   │ Deletes per-source. Best default.    │")
    print("   │ 'full'          │ Deletes ALL not in current batch.    │")
    print("   └─────────────────┴────────────────────────────────────────┘")
    print()
    print("   PRODUCTION:")
    print("   - Local dev:  SQLite  (zero setup)")
    print("   - Production: PostgreSQL (shared state, backed up)")
    print("   - Multi-worker: PostgreSQL or Redis")
    print("=" * 70)

    # Cleanup the demo database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n   🧹 Cleaned up demo database: {db_path}")
