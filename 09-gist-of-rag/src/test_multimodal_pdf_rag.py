# =============================================================================
# RAG TEST: Production-Grade PDF Ingestion + Retrieval Demo
# =============================================================================
# This program demonstrates the full RAG pipeline with PDF documents, including
# a deep comparison of ALL available PDF loader strategies in LangChain.
#
# ⚠️  THE PROBLEM WITH BASIC PDF LOADERS:
#   Plain text PDF loaders (like PyPDFLoader) ONLY extract raw text.
#   They CANNOT understand:
#     - Diagrams and flowcharts (they see nothing — images are skipped entirely)
#     - Tables (columns get merged into garbled text)
#     - Scanned/image-based PDFs (no text layer = zero extraction)
#     - Complex layouts (multi-column, sidebars, callout boxes)
#
#   Production-grade PDFs (architecture docs, whitepapers, technical specs)
#   almost always contain these elements. Using a basic loader means your
#   RAG system silently LOSES critical information.
#
# THE SOLUTION — TIERED LOADER STRATEGY:
#   This file demonstrates 4 loader approaches, from simplest to most capable:
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │ Loader              │ Text  │ Tables │ Diagrams │ Scanned │ Cost   │
#   │─────────────────────│────── │────────│──────────│─────────│────────│
#   │ 1. PyPDFLoader       │ ✅   │ ❌      │ ❌        │ ❌       │ Free   │
#   │ 2. PyMuPDFLoader     │ ✅   │ ⚠️      │ ❌        │ ❌       │ Free   │
#   │ 3. UnstructuredLoader│ ✅   │ ✅      │ ❌        │ ✅       │ Free*  │
#   │ 4. Multimodal Vision │ ✅   │ ✅      │ ✅        │ ✅       │ $$$    │
#   └─────────────────────────────────────────────────────────────────────┘
#   * Unstructured requires system-level dependencies (Tesseract, Poppler)
#
#   This script defaults to OPTION 4 (Multimodal Vision) because it is the
#   only approach that truly understands diagrams, flowcharts, and complex
#   layouts — which is what production PDFs actually contain.
#
# PREREQUISITES:
#   1. Pinecone account: https://pinecone.io (free tier)
#   2. Create an index: dimensions=1536, metric=cosine, type=dense, serverless
#   3. .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   4. Packages:
#      uv add langchain-pinecone langchain-community langchain-text-splitters
#      uv add pypdf pymupdf pdf2image
#   5. For Option 4 (Multimodal — RECOMMENDED):
#      - No extra system dependencies! PyMuPDF handles PDF → image conversion
#        natively (pure Python, no Poppler needed).
#
# USAGE:
#   uv run test_pdf_rag.py
#
# C# ANALOGY:
#   Think of loader selection like choosing a deserialization strategy:
#     - PyPDFLoader     = Reading a .docx as plain text (loses formatting)
#     - PyMuPDFLoader   = Using a basic XML parser (preserves some structure)
#     - Unstructured    = Using a full HTML parser with CSS awareness
#     - Multimodal      = Taking a screenshot and having a human read it
# =============================================================================

import base64
import os
import sys

# Use the Windows certificate store for SSL — fixes corporate proxy issues
# (Without this, Python uses its own bundled CA certs which don't include
# your company's internal CA, causing SSL_CERTIFICATE_VERIFY_FAILED errors)
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# ========================= CONFIGURATION =========================
PDF_PATH = "event tracking.pdf"

QUERY = "How is the Parcel events flown from the Hub to the CTS System?"

TOP_K = 3

# LOADER STRATEGY — Change this to try different loaders:
#   "pypdf"       → Option 1: Basic text-only (fast, free, loses tables/diagrams)
#   "pymupdf"     → Option 2: Layout-aware text (free, better tables, no diagrams)
#   "unstructured"→ Option 3: OCR-capable (free, needs Tesseract/Poppler installed)
#   "multimodal"  → Option 4: GPT-4o Vision (best quality, handles everything, costs $)
LOADER_STRATEGY = "multimodal"
# =================================================================


def check_prerequisites():
    """Verify all required env vars and files exist before proceeding."""
    errors = []

    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.environ.get("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY not found in .env")
    if not os.environ.get("INDEX_NAME"):
        errors.append("INDEX_NAME not found in .env")
    if not os.path.exists(PDF_PATH):
        errors.append(f"PDF file not found: {PDF_PATH}")

    if errors:
        print("❌ Prerequisites check FAILED:")
        for e in errors:
            print(f"   - {e}")
        print("\nSetup steps:")
        print("  1. Create a Pinecone account at https://pinecone.io (free tier)")
        print("  2. Create an index: dimensions=1536, metric=cosine")
        print("  3. Add OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME to your .env file")
        print(f"  4. Place your PDF file at: {PDF_PATH}")
        sys.exit(1)

    print("✅ All prerequisites met!")
    print(f"   OpenAI key: {os.environ['OPENAI_API_KEY'][:8]}...")
    print(f"   Pinecone key: {os.environ['PINECONE_API_KEY'][:8]}...")
    print(f"   Index: {os.environ['INDEX_NAME']}")
    print(f"   PDF: {PDF_PATH}")
    print(f"   Loader: {LOADER_STRATEGY}")
    print()


# =============================================================================
# LOADER OPTION 1: PyPDFLoader (Basic Text-Only)
# =============================================================================
# WHAT IT DOES:
#   Extracts raw text from each PDF page using the `pypdf` library.
#   Creates one LangChain Document per page.
#
# STRENGTHS:
#   - Zero system dependencies (pure Python)
#   - Fast and lightweight
#   - Good for simple, text-heavy PDFs (articles, essays, plain reports)
#
# WEAKNESSES:
#   - Tables → columns merge into a single garbled line of text
#   - Diagrams/Flowcharts → completely invisible (skipped, no output at all)
#   - Scanned PDFs → returns empty strings (no OCR capability)
#   - Multi-column layouts → text from different columns gets interleaved
#
# WHEN TO USE:
#   Quick prototyping with simple text PDFs. Never for production documents
#   that contain any visual elements.
#
# INSTALL: uv add pypdf
# =============================================================================
def load_with_pypdf(pdf_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    print("📄 [PyPDFLoader] Loading PDF (text-only extraction)...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages (text only — diagrams/images SKIPPED)")
    return documents


# =============================================================================
# LOADER OPTION 2: PyMuPDFLoader (Layout-Aware Text)
# =============================================================================
# WHAT IT DOES:
#   Uses the `pymupdf` (fitz) library which is a binding to the MuPDF C library.
#   MuPDF understands the PDF's internal coordinate system, so it can preserve
#   reading order, detect columns, and extract text with better spatial awareness.
#
# STRENGTHS:
#   - Much better table extraction than PyPDFLoader (preserves column alignment)
#   - Preserves reading order in multi-column layouts
#   - Extracts metadata (author, title, creation date) automatically
#   - Faster than PyPDFLoader for large files (C library under the hood)
#
# WEAKNESSES:
#   - Diagrams/Flowcharts → still invisible (it reads text, not images)
#   - Scanned PDFs → still returns empty (no OCR)
#   - Complex nested tables → can still garble
#
# WHEN TO USE:
#   Good default for text-heavy production docs with tables but no diagrams.
#   Significant upgrade over PyPDFLoader at zero extra cost.
#
# INSTALL: uv add pymupdf
# =============================================================================
def load_with_pymupdf(pdf_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader

    print("📄 [PyMuPDFLoader] Loading PDF (layout-aware text extraction)...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   Loaded {len(documents)} pages (better tables, but diagrams still SKIPPED)")
    return documents


# =============================================================================
# LOADER OPTION 3: UnstructuredPDFLoader (OCR + Layout Parsing)
# =============================================================================
# WHAT IT DOES:
#   Uses the `unstructured` library which combines multiple strategies:
#   - For digital PDFs: extracts text with layout detection (headers, lists, tables)
#   - For scanned PDFs: runs Tesseract OCR to convert images to text
#   - Classifies content into element types (Title, NarrativeText, Table, etc.)
#
# STRENGTHS:
#   - Handles scanned/image-based PDFs via OCR
#   - Detects document structure (headers, footers, page numbers)
#   - Table extraction is significantly better than both options above
#   - Can partition documents into semantic elements (not just pages)
#
# WEAKNESSES:
#   - Diagrams/Flowcharts → OCR might extract text labels inside them,
#     but CANNOT understand the visual relationships (arrows, flow, hierarchy)
#   - Requires system dependencies: Tesseract OCR + Poppler (not pip-installable)
#   - Slower than PyPDF/PyMuPDF due to the multi-strategy pipeline
#   - The `unstructured` package is large (~500MB with all extras)
#
# SYSTEM DEPENDENCIES (must install separately):
#   - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
#   - Poppler: https://github.com/osssr/poppler-windows/releases
#   - uv add unstructured[pdf]
#
# WHEN TO USE:
#   When you have scanned PDFs or need structured element detection.
#   Still not enough for diagrams — use multimodal for that.
#
# NOTE: This function is included for reference. It requires extra system
# dependencies that are not installed by default in this course.
# =============================================================================
def load_with_unstructured(pdf_path: str) -> list[Document]:
    """
    ⚠️ Requires system dependencies (Tesseract, Poppler) and:
       uv add "unstructured[pdf]"
    Uncomment and use only if you have these installed.
    """
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
    except ImportError:
        print("❌ [UnstructuredPDFLoader] Not available.")
        print("   Install with: uv add \"unstructured[pdf]\"")
        print("   Also requires Tesseract and Poppler on your system PATH.")
        sys.exit(1)

    print("📄 [UnstructuredPDFLoader] Loading PDF (OCR + layout parsing)...")
    # mode="elements" returns each detected element (Title, Text, Table) separately
    # mode="single" returns the entire document as one chunk
    # mode="paged" returns one Document per page (like PyPDFLoader but with better parsing)
    loader = UnstructuredPDFLoader(pdf_path, mode="elements")
    documents = loader.load()
    print(f"   Loaded {len(documents)} elements (OCR-capable, but diagrams still NOT understood)")

    # Show the element types detected
    element_types = set(doc.metadata.get("category", "Unknown") for doc in documents)
    print(f"   Detected element types: {element_types}")
    return documents


# =============================================================================
# LOADER OPTION 4: Multimodal Vision (GPT-4o) — ⭐ RECOMMENDED FOR PRODUCTION
# =============================================================================
# WHAT IT DOES:
#   Converts each PDF page into an IMAGE, then sends it to GPT-4o's vision
#   capability. The LLM literally "sees" the page and describes everything:
#   text, tables, diagrams, flowcharts, architecture diagrams, screenshots, etc.
#
# HOW IT WORKS (Step by Step):
#   1. PyMuPDF (fitz) renders each PDF page → PNG image (no Poppler needed)
#   2. Each PNG is base64-encoded (the format GPT-4o expects for images)
#   3. GPT-4o receives the image and a prompt: "Extract ALL content from this page"
#   4. GPT-4o returns a rich text description including:
#      - All visible text (preserving structure)
#      - Table contents (properly formatted)
#      - Diagram descriptions ("This flowchart shows A → B → C with condition X")
#      - Chart data and trends
#   5. Each page's description becomes a LangChain Document for the RAG pipeline
#
# STRENGTHS:
#   - Understands EVERYTHING a human can see on the page
#   - Diagrams, flowcharts, architecture diagrams → fully described
#   - Tables → properly structured
#   - Scanned PDFs → no problem (it's looking at the image)
#   - Complex layouts → handled naturally (GPT-4o reads like a human)
#
# WEAKNESSES:
#   - Cost: ~$0.01-0.03 per page (GPT-4o vision pricing)
#   - Speed: ~2-5 seconds per page (API call per page)
#   - The LLM description is an interpretation, not exact text extraction
#     (minor wording differences possible vs the original PDF text)
#
# SYSTEM DEPENDENCIES:
#   - pymupdf (already installed) — handles PDF → image conversion natively
#   - No Poppler or Tesseract needed!
#   - uv add pymupdf  (already done)
#
# WHEN TO USE:
#   Any production PDF that contains diagrams, flowcharts, tables, or mixed
#   content. The cost is negligible for most use cases (a 50-page PDF costs
#   about $0.50-$1.50 to process once, and you only ingest once).
#
# C# ANALOGY:
#   This is like using Azure Computer Vision / Azure Document Intelligence
#   instead of a basic text parser. You're paying for an AI service to
#   "read" the document the way a human would.
# =============================================================================
def load_with_multimodal_vision(pdf_path: str) -> list[Document]:
    import fitz  # pymupdf — renders PDF pages as images without Poppler
    from langchain_openai import ChatOpenAI

    print("📄 [Multimodal Vision] Converting PDF pages to images...")

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ PyMuPDF failed to open PDF: {e}")
        sys.exit(1)

    page_count = len(pdf_doc)
    print(f"   Opened PDF: {page_count} pages")

    # Use GPT-4o (the multimodal model that can see images)
    llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

    documents = []
    for i in range(page_count):
        print(f"   🔍 Analyzing page {i + 1}/{page_count} with GPT-4o Vision...")

        # Render the page as a PNG image at 200 DPI using PyMuPDF
        page = pdf_doc[i]
        pix = page.get_pixmap(dpi=200)
        image_bytes = pix.tobytes("png")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Send the image to GPT-4o with a detailed extraction prompt
        # The prompt is critical — it tells GPT-4o exactly what to extract
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a document analysis expert. Extract ALL content "
                                "from this PDF page. Include:\n"
                                "1. All visible text (preserve headings, bullet points, numbering)\n"
                                "2. Tables (format as markdown tables)\n"
                                "3. Diagrams and flowcharts (describe the visual flow, "
                                "nodes, arrows, relationships, and any labels)\n"
                                "4. Charts and graphs (describe the data, axes, trends)\n"
                                "5. Any other visual elements\n\n"
                                "Be thorough and precise. Preserve the document's structure."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",  # Use high-res analysis
                            },
                        },
                    ],
                }
            ]
        )

        # Create a LangChain Document from the GPT-4o description
        page_content = response.content
        doc = Document(
            page_content=page_content,
            metadata={
                "source": pdf_path,
                "page": i,
                "loader": "multimodal_vision_gpt4o",
            },
        )
        documents.append(doc)
        print(f"      ✅ Page {i + 1}: extracted {len(page_content)} chars")

    pdf_doc.close()
    print(f"   ✅ All {len(documents)} pages analyzed with vision!\n")
    return documents


# =============================================================================
# LOADER DISPATCHER — Routes to the selected strategy
# =============================================================================
def load_pdf(pdf_path: str, strategy: str) -> list[Document]:
    """Load a PDF using the selected strategy.

    This is the central decision point. In production, you would typically:
    - Use multimodal for the initial ingestion of important docs
    - Use pymupdf as a fast fallback for simple text documents
    - Use unstructured when you have scanned PDFs at scale
    """
    loaders = {
        "pypdf": load_with_pypdf,
        "pymupdf": load_with_pymupdf,
        "unstructured": load_with_unstructured,
        "multimodal": load_with_multimodal_vision,
    }

    if strategy not in loaders:
        print(f"❌ Unknown loader strategy: '{strategy}'")
        print(f"   Available: {', '.join(loaders.keys())}")
        sys.exit(1)

    return loaders[strategy](pdf_path)


# =============================================================================
# INGESTION PIPELINE
# =============================================================================
def run_ingestion():
    """Phase 1: Load PDF → Chunk → Embed → Store in Pinecone."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain_text_splitters import CharacterTextSplitter

    # Step 1: Load PDF using the selected strategy
    documents = load_pdf(PDF_PATH, LOADER_STRATEGY)

    # Step 2: Split into chunks
    # Even with multimodal (where each page is already a separate doc),
    # pages can produce long descriptions. Chunking keeps retrieval precise.
    print("✂️  Splitting into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks from {len(documents)} pages")

    # Show a sample chunk
    if chunks:
        sample = chunks[0]
        print(f"\n   📋 Sample chunk (first 200 chars):")
        print(f"   \"{sample.page_content[:200]}...\"")
        print(f"   Metadata: {sample.metadata}")
        print()

    # Step 3: Embed and store
    print("🧮 Embedding and storing in Pinecone...")
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print(f"   ✅ Stored {len(chunks)} vectors in Pinecone!\n")


# =============================================================================
# RETRIEVAL PIPELINE
# =============================================================================
def run_retrieval(query: str):
    """Phase 2: Query → Retrieve → Augment → Generate."""
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
    )

    # ---- Without RAG (baseline) ----
    print("=" * 70)
    print("🤖 WITHOUT RAG (raw LLM — no context)")
    print("=" * 70)
    raw_response = llm.invoke([HumanMessage(content=query)])
    print(raw_response.content)
    print()

    # ---- With RAG ----
    print("=" * 70)
    print("📚 WITH RAG (grounded in your PDF)")
    print("=" * 70)

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    print(f"\n🔍 Retrieved {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs):
        loader_used = doc.metadata.get("loader", "text")
        source_info = f"page {doc.metadata.get('page', '?')}" if "page" in doc.metadata else doc.metadata.get("source", "?")
        print(f"   Chunk {i+1} ({source_info}, loader={loader_used}): \"{doc.page_content[:80]}...\"")
    print()

    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    print("💡 Answer:")
    print(response.content)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    check_prerequisites()

    print("=" * 70)
    print(f"PHASE 1: INGESTION (PDF → Pinecone) [Strategy: {LOADER_STRATEGY}]")
    print("=" * 70)
    run_ingestion()

    print("=" * 70)
    print("PHASE 2: RETRIEVAL (Query → RAG Answer)")
    print("=" * 70)
    run_retrieval(QUERY)

    print("\n✅ Done!")
    print("   To try a different loader, change LOADER_STRATEGY at the top of this file.")
    print("   Options: pypdf | pymupdf | unstructured | multimodal")
