# =============================================================================
# INGESTION PIPELINE: Load → Chunk → Embed → Store
# =============================================================================
# This script runs ONCE (or whenever documents change) to populate the vector
# store with embedded document chunks. It is the "offline" phase of RAG.
#
# Pipeline:
#   mediumblog.txt → TextLoader → Document → CharacterTextSplitter → Chunks
#                                                                        ↓
#                                    Pinecone ← OpenAIEmbeddings ← 20 Chunks
#
# Prerequisites:
#   1. Pinecone account with an index created (dimensions=1536, metric=cosine)
#   2. .env file with OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   3. Dependencies: uv add langchain langchain-openai langchain-pinecone
#                    langchain-community langchain-text-splitters python-dotenv
# =============================================================================

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from .env file
# Required: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")

    # =========================================================================
    # STEP 1: Load the document
    # =========================================================================
    # TextLoader reads a plain text file into a LangChain Document object.
    # The Document has two attributes:
    #   - page_content: The full text content of the file
    #   - metadata: {"source": "path/to/file"} (for grounding/provenance later)
    #
    # LangChain provides loaders for many formats (PDF, Notion, Slack, etc.)
    # The interface is always the same: loader.load() → List[Document]
    #
    # NOTE: If you get UnicodeDecodeError, add encoding="utf-8" parameter:
    #   TextLoader("mediumblog.txt", encoding="utf-8")
    loader = TextLoader("mediumblog.txt")
    document = loader.load()

    # =========================================================================
    # STEP 2: Split document into chunks
    # =========================================================================
    # Why chunk? Even with large context windows, sending too much text:
    #   - Degrades answer quality (needle in haystack problem)
    #   - Costs more tokens (wasted API spend)
    #   - Increases latency
    #
    # Parameters:
    #   chunk_size=1000:  Max characters per chunk. Heuristic: large enough to
    #                     have semantic meaning, small enough to fit several in
    #                     a prompt. Read a chunk — if it makes sense, it's good.
    #   chunk_overlap=0:  No shared text between adjacent chunks. Set > 0 if
    #                     answers might span chunk boundaries.
    #
    # The splitter defaults to splitting on "\n\n" (paragraph boundaries).
    # Each resulting chunk is still a LangChain Document (inherits metadata).
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # =========================================================================
    # STEP 3: Embed chunks and store in Pinecone
    # =========================================================================
    # OpenAIEmbeddings uses the text-embedding-ada-002 model by default.
    # It converts text → vector of 1536 dimensions.
    #
    # PineconeVectorStore.from_documents() does the heavy lifting:
    #   1. Iterates through all chunks
    #   2. Sends each chunk's page_content to OpenAI's embedding API
    #   3. Stores the resulting vector + metadata in Pinecone
    #   4. Handles batching and rate limits automatically
    #
    # After this runs, your Pinecone index will contain 20 vectors, each
    # representing one chunk of the blog post.
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting into vector store...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Finished! Vectors stored in Pinecone.")
