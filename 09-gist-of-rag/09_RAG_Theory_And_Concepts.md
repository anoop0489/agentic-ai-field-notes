# 09. The GIST of RAG — Embeddings, Vector Databases & Retrieval

A comprehensive guide to Retrieval Augmented Generation: what it is, why it matters, and how every component fits together.

*Topic: The GIST of RAG*

---

## 📑 Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Key Definitions](#key-definitions-interview-ready) | 20+ terms with quick-recall openers for interviews |
| 2 | [The Problem RAG Solves](#the-problem-rag-solves) | Why LLMs fail without RAG, the 4 problems with stuffing |
| 3 | [The RAG Pipeline](#the-rag-pipeline-two-phases) | Ingestion (offline) + Retrieval (online) phases |
| 4 | [Document & Loaders](#deep-dive-langchain-document--document-loaders) | `Document` object, loader abstraction, source code |
| 5 | [Embeddings](#deep-dive-embeddings) | Vectors, distance metrics, model selection, same-model rule |
| 6 | [Vector Databases](#deep-dive-vector-databases) | Pinecone setup walkthrough, `from_documents()`, DB comparison |
| 7 | [Chunking](#deep-dive-chunking) | Size trade-offs, overlap, advanced strategies |
| 8 | [The Retriever](#deep-dive-the-retriever) | `as_retriever()`, inheritance chain, how search works |
| 9 | [Production Considerations](#production-considerations) | Agentic vs deterministic RAG, LangSmith tracing |
| 10 | [Streaming](#deep-dive-streaming-token-by-token-responses) | `.stream()` vs `.invoke()`, FastAPI example, method family |
| 11 | [Indexing Strategy](#deep-dive-indexing-strategy-incremental-ingestion) | RecordManager, content hashing, cleanup modes, SQLite vs Postgres |
| 12 | [Multimodal PDF RAG](#deep-dive-multimodal-pdf-rag) | Why text-only loaders fail, GPT-4o Vision approach, loader comparison |
| 13 | [Conversational RAG](#deep-dive-conversational-rag-multi-turn) | Question reformulation, memory strategies, follow-up vs new topic |
| 14 | [RAG Evaluation & Quality](#deep-dive-rag-evaluation--quality) | Three quality dimensions, failure modes, evaluation frameworks |
| 15 | [Interview Q&A](#interview-qa-anchors) | 18 interview questions with production-grade answers |

---

## What is this section about?

This section teaches you how to build a **RAG pipeline** — the most common production pattern for making LLMs answer questions about your private data. We start with the "why" (motivation), then move through every building block (embeddings, vector stores, chunking, retrieval), and end with two complete implementations.

---

## Key Definitions (Interview-Ready)

Use these as your opening sentence when asked "What is X?" in an interview:

| Term | Quick Recall (say this first) | Full Definition |
|------|------|------------|
| **RAG** | "Retrieve context, augment the prompt, generate the answer" | A technique that retrieves relevant documents from a knowledge base, injects them into the LLM's prompt as context, and generates a grounded answer — solving the problem of LLMs not knowing your private data. |
| **Embedding** | "Text → vector of numbers" | A numerical representation (vector) of text where semantically similar texts are close together in vector space, enabling mathematical similarity comparisons. Not all vectors are embeddings, but all embeddings are vectors. |
| **Vector** | "A list of numbers representing something" | A sequence of numerical values that represent the characteristics of an object (word, image, etc.) in a high-dimensional space. The building block of embeddings. |
| **Vector Database** | "Database optimized for similarity search" | A specialized database that stores high-dimensional vectors and provides fast nearest-neighbor search — enabling retrieval of semantically similar documents at scale. |
| **Chunking** | "Split large docs into digestible pieces" | The process of breaking a large document into smaller segments (chunks) that fit within the LLM's context window while preserving semantic meaning. |
| **Similarity Search** | "Find the closest vectors" | A query operation that takes a vector and returns the k nearest vectors in the database, measured by distance metrics like cosine similarity. |
| **Retriever** | "Query → relevant documents" | A component that takes a user query, embeds it, and returns the top-k most relevant document chunks from the vector store. In LangChain, a Runnable with `invoke()`. |
| **Context Window** | "Max tokens the LLM can process" | The hard limit on how many tokens (input + output) an LLM can handle in a single request — exceeded tokens are rejected. |
| **Document (LangChain)** | "`page_content` + `metadata`" | LangChain's universal data object for any loaded data. Has two attributes: `page_content` (the actual text) and `metadata` (a dict with `source`, custom fields for filtering/provenance). |
| **Document Loader** | "Any source → LangChain Document" | A LangChain abstraction that loads data from any source (PDF, text, Notion, Google Drive, Slack, WhatsApp) into a standardized `Document` object. Same `.load()` interface regardless of source. |
| **Text Splitter** | "Chunk with strategy" | A LangChain utility that splits documents into chunks using configurable strategies (character count, token count, recursive splitting) with optional overlap. |
| **Chunk Overlap** | "Shared text between adjacent chunks" | The number of characters/tokens that consecutive chunks share, preventing context loss when an answer spans a chunk boundary. |
| **`from_documents()`** | "Embed + store in one call" | LangChain's method on vector stores that iterates through documents, embeds each one, and stores vectors — handling batching, rate limits, and threading automatically. |
| **`as_retriever()`** | "Vector store → searchable Runnable" | Converts a vector store into a LangChain Retriever (a Runnable with `invoke()`) that wraps similarity search functionality. |
| **LCEL** | "Pipe operator for LangChain" | LangChain Expression Language — a declarative syntax using the `|` operator to compose Runnables into chains with built-in streaming, async, and batch support. |
| **RunnablePassthrough** | "Pass input through, optionally add keys" | A LangChain Runnable that forwards its input unchanged while optionally computing and assigning new keys to the output dictionary via `.assign()`. |
| **RunnableLambda** | "Auto-wrapped Python function" | When you pipe a regular Python function in an LCEL chain, LangChain auto-converts it into a `RunnableLambda` — giving it `invoke()`, `stream()`, and `batch()` capabilities. |
| **StrOutputParser** | "AIMessage → just the text" | A LangChain parser that extracts the `.content` string from an `AIMessage` response, so the chain output is a clean string instead of a message object. |
| **Grounding** | "Answer backed by evidence" | The practice of ensuring an LLM's response is based on retrieved factual data (the context) rather than its parametric knowledge or hallucination. |
| **Knowledge Cutoff** | "LLM doesn't know anything after date X" | The training data deadline of an LLM. Facts, products, or events after this date are unknown to the model — a key motivation for RAG. |
| **Hallucination** | "LLM confidently invents wrong facts" | When an LLM generates plausible-sounding but factually incorrect information because it lacks the real data. RAG directly combats this by providing actual source text. |
| **Question Reformulation** | "Rewrite follow-ups as standalone queries" | The technique of using an LLM to resolve pronouns and references in follow-up questions by incorporating chat history, enabling accurate retrieval in multi-turn conversations. |
| **Multimodal RAG** | "Vision model reads PDFs with diagrams" | An ingestion approach where PDF pages are rendered as images and processed by a Vision model (GPT-4o), capturing diagrams, tables, and layouts that text-only loaders miss. |
| **RecordManager** | "Track what's already been embedded" | A LangChain component that stores content hashes in a database (SQLite/PostgreSQL) to enable incremental ingestion — skipping unchanged chunks, re-embedding modified ones, and deleting removed ones. |
| **Content Hashing** | "SHA256 fingerprint of each chunk" | Each chunk's text is hashed to a unique fingerprint. On re-ingestion, matching hashes are skipped, changed hashes trigger re-embedding, and orphaned hashes trigger deletion. |

---

## The Problem RAG Solves

### Why can't we just ask the LLM directly?

LLMs are trained on public internet data up to a knowledge cutoff date. They have **no knowledge** of:
- Your company's internal documents
- Recent events after their training cutoff
- Private databases, PDFs, or codebases

### The Naive Solution (and why it fails)

> "Just stuff the entire document into the prompt."

This breaks for **four reasons**:

| Problem | Explanation |
|---------|-------------|
| **1. Token Limit** | LLMs have a hard cap (e.g., 4K, 128K, 1M tokens). Exceeding it = request rejected. |
| **2. Needle in a Haystack** | Research proves LLMs become less accurate with longer contexts — they "lose" information in the middle of long prompts. |
| **3. Cost** | More tokens = higher API costs. Sending 100K tokens when you need 500 is wasteful. |
| **4. Latency** | More tokens = slower processing time. Users wait longer for answers. |

### The RAG Solution

Instead of sending everything, we:
1. **Pre-process**: Split the document into small chunks and store them as vectors
2. **Retrieve**: Find only the chunks relevant to the user's question
3. **Augment**: Inject those chunks into the prompt as context
4. **Generate**: Let the LLM answer grounded in the relevant context

This solves all four problems: stays within token limits, focuses the LLM on relevant data, reduces cost, and improves latency.

### The Knowledge Cutoff Problem (Real-World Example)

We can demonstrate this live with the query *"What is Pinecone in machine learning?"*:

| Model | Response (No RAG) | Why? |
|-------|-------------------|------|
| **GPT-3.5** | "A pinecone algorithm is a method for searching hyperparameters..." | ❌ **Hallucination.** GPT-3.5 was trained before Pinecone (the vector DB) was widely known. |
| **GPT-4o/5** | "Pinecone is a managed vector database used in ML to store and index..." | ✅ Correct — trained on more recent data. |
| **GPT-3.5 + RAG** | "Pinecone is a fully managed cloud-based vector database..." | ✅ Correct — grounded in the retrieved blog content. |

**Key insight**: RAG lets even an older/cheaper model answer correctly by providing the source text. You don't need the most expensive model if you have good retrieval.

---

## The RAG Pipeline (Two Phases)

![Basic RAG Pipeline](./assets/basic_rag_pipeline.png)

### Phase 1: Ingestion (Offline, One-Time)

```
Document → Load → Chunk → Embed → Store in Vector DB
```

| Step | What Happens | LangChain Component |
|------|-------------|-------------------|
| **Load** | Read the file into a LangChain `Document` | `TextLoader`, `PyPDFLoader`, etc. |
| **Chunk** | Split into smaller pieces (e.g., 1000 chars) | `CharacterTextSplitter` |
| **Embed** | Convert each chunk into a vector | `OpenAIEmbeddings` |
| **Store** | Save vectors + metadata in vector DB | `PineconeVectorStore.from_documents()` |

### Phase 2: Retrieval + Generation (Online, Per-Query)

```
User Query → Embed → Similarity Search → Top-K Chunks → Augment Prompt → LLM → Answer
```

| Step | What Happens | LangChain Component |
|------|-------------|-------------------|
| **Embed Query** | Convert user question into a vector | `OpenAIEmbeddings` |
| **Search** | Find k nearest chunk vectors | `retriever.invoke(query)` |
| **Format** | Join chunks into a context string | `format_docs()` |
| **Augment** | Plug context + question into prompt template | `ChatPromptTemplate` |
| **Generate** | Send augmented prompt to LLM | `ChatOpenAI` |

---

## Deep Dive: LangChain Document & Document Loaders

### The `Document` Object

Everything in LangChain's RAG pipeline flows through `Document` objects. A Document has exactly two attributes:

```python
document.page_content   # str — the actual text content
document.metadata       # dict — metadata about the source
```

**Example after loading `mediumblog.txt`:**
```python
Document(
    page_content="Vector Database: What is it and why you should know it?...",
    metadata={"source": "mediumblog.txt"}
)
```

The `metadata.source` field is critical for **provenance** — it tells users (and the system) where the answer came from. In production, metadata can include page numbers, timestamps, authors, or custom fields for filtering.

### Document Loaders (The Abstraction)

LangChain implements loaders for dozens of data sources. The interface is always the same:

```python
loader = SomeLoader(source)
documents = loader.load()  # Always returns List[Document]
```

| Loader | Source | Example |
|--------|--------|---------|
| `TextLoader` | Plain text files | `.txt`, `.md` |
| `PyPDFLoader` | PDF documents | `.pdf` (page per Document) |
| `NotionDirectoryLoader` | Notion exports | Notion workspace |
| `SlackChatLoader` | Slack messages | Chat history |
| `YoutubeLoader` | YouTube transcripts | Video captions |
| `GoogleDriveLoader` | Google Drive | Any Drive file |
| `CSVLoader` | CSV files | Row per Document |

**The power of this abstraction**: Your downstream code (splitting, embedding, retrieval) doesn't care where the data came from. Swap `TextLoader` for `PyPDFLoader` and the rest of the pipeline is unchanged.

### Under the Hood (Source Code)

Diving into LangChain's source code shows how simple it is:

```python
# TextLoader source code (simplified):
class TextLoader:
    def load(self):
        with open(self.file_path) as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]
```

Even the WhatsApp loader is just: open file → regex to extract sender/date/text → format → return as Document. The pattern is always the same.

---

## Deep Dive: Embeddings

### What is an Embedding?

An embedding model is a **black box** that:
- **Input**: Text (word, sentence, paragraph)
- **Output**: A vector (list of numbers, e.g., 1536 dimensions)

The key property: **semantically similar texts produce vectors that are close together** in vector space.

> **Important distinction**: An embedding IS a vector representation, but not all vectors are embeddings. An embedding specifically captures semantic meaning learned through training.

### Example

```
"I want to order an extra large coffee"  →  [0.23, -0.45, 0.89, ...]
"I'll have a tall coffee"                →  [0.21, -0.43, 0.91, ...]  ← CLOSE!
"quiero pedir café extra grande"         →  [0.22, -0.44, 0.90, ...]  ← CLOSE! (cross-language)
"The stock market crashed today"         →  [0.78,  0.12, -0.56, ...] ← FAR!
```

The embedding model doesn't care about language — it captures **meaning**. Two sentences with the same semantic intent will be close in vector space even if one is in English and the other in Spanish.

### How It Enables RAG

When a user asks "How tall is the Burj Khalifa?", we embed this question into a vector. In the vector space, this vector will be close to the Wikipedia paragraph about the Burj Khalifa's height (if we've embedded and stored it). We retrieve that paragraph, inject it as context, and the LLM can answer accurately even if it wasn't trained on this information.

### Distance Metrics

| Metric | Use Case | How It Works |
|--------|----------|--------------|
| **Cosine Similarity** | Most common, default in Pinecone | Measures angle between vectors (0° = identical) |
| **Euclidean Distance** | When magnitude matters | Straight-line distance in vector space |
| **Dot Product** | Fast computation | Combination of angle and magnitude |

### Embedding Models

| Model | Provider | Dimensions | Notes |
|-------|----------|-----------|-------|
| `text-embedding-3-small` | OpenAI | 512–1536 | Good balance of cost/quality |
| `text-embedding-ada-002` | OpenAI | 1536 | Legacy but still widely used |
| `text-embedding-3-large` | OpenAI | 256–3072 | Highest quality, most expensive |

**Rule of thumb**: Longer vectors = more semantic information captured, but higher storage/compute cost.

### LangChain's Uniform Embedding Interface

LangChain provides **one interface** for all embedding providers. Whether you use OpenAI, Cohere, Hugging Face, or any other provider — the API is the same:

```python
# Swap providers by changing one import + one class name. Downstream code stays identical.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # Uses text-embedding-ada-002 by default

# OR:
from langchain_community.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings()

# The rest of your code doesn't change:
vector = embeddings.embed_query("What is Pinecone?")
```

This is the same pattern as `init_chat_model()` for LLMs — abstract the provider, keep the interface consistent.

### ⚠️ Critical Rule: Same Model for Ingestion AND Retrieval

The embedding model used to embed your documents MUST be the same model used to embed the user's query. If you embed documents with `text-embedding-3-small` (1536 dimensions) but query with `text-embedding-ada-002` (also 1536 dimensions), the vectors live in different vector spaces and similarity search will return garbage.

**If you change your embedding model, you must re-embed all your documents.**

---

## Deep Dive: Vector Databases

### What is a Vector Database?

A specialized database that:
1. **Stores** high-dimensional vectors alongside metadata
2. **Indexes** vectors for fast nearest-neighbor search
3. **Retrieves** the top-k most similar vectors to a query vector

### Why not just use a regular database?

Regular databases search by exact match (`WHERE name = 'laptop'`). Vector databases search by **semantic similarity** — "find me documents that mean something similar to this question."

### Pinecone (Used Here)

- **Managed**: Cloud-based, no infrastructure to manage
- **Serverless**: Pay per query, scales automatically
- **Free tier**: Sufficient for learning and prototyping

#### Pinecone Index Configuration

| Setting | Our Value | Why |
|---------|-----------|-----|
| **Dimensions** | 1536 | Must match embedding model output size |
| **Metric** | Cosine | Default, works well for text similarity |
| **Type** | Dense | Standard for text embeddings |
| **Mode** | Serverless | Cost-effective for development |
| **Cloud/Region** | AWS us-east-1 | For production: same region as your app to avoid egress costs |

> **Production tip**: Deploy your vector store in the same cloud region as your RAG application. Cross-region calls add latency and egress costs.

#### Setting Up Pinecone (Step-by-Step Console Walkthrough)

1. **Create an Account**
   - Go to [pinecone.io](https://pinecone.io) → Click **"Sign Up Free"**
   - You can sign up with Google, GitHub, or email
   - The **free tier** (Starter plan) is More than enough for this project — it gives you 1 project, 5 serverless indexes, and 2GB of storage

2. **Create a New Index**
   - After logging in, you land on the **Dashboard**
   - Click **"Create Index"** (top-right button)
   - Fill in the form:

   | Field | What to Select | Why |
   |-------|---------------|-----|
   | **Index Name** | `event-tracking-pdf-index` (or any descriptive name) | This goes into your `.env` as `INDEX_NAME`. Use lowercase with hyphens. |
   | **Dimensions** | `1536` | ⚠️ **Must match your embedding model's output size.** OpenAI's `text-embedding-ada-002` (LangChain's default) outputs 1536-dimensional vectors. If you use a different model, check its docs. |
   | **Metric** | `cosine` | Measures the angle between vectors. Best default for text similarity. Other options: `euclidean` (spatial distance) or `dotproduct` (magnitude-sensitive). |
   | **Type** | `Dense` | Standard for text embeddings. "Sparse" is for keyword/BM25-style search. |
   | **Capacity Mode** | `Serverless` | No infrastructure to manage, scales automatically, pay-per-query. The alternative "Pod-based" gives dedicated compute but costs more and requires capacity planning. |
   | **Cloud Provider** | `AWS` | Any provider works; AWS has the most regions. |
   | **Region** | `us-east-1` (Virginia) | Choose the region closest to you or your app. For production: **same region as your application** to avoid cross-region latency and egress costs. |

   - Click **"Create Index"**
   - Wait ~30 seconds for the index to become **"Ready"** (green status)

3. **Generate an API Key**
   - In the left sidebar, click **"API Keys"**
   - You should see a **Default** key already created
   - Click the **copy icon** to copy the key (it starts with `pcsk_...`)
   - ⚠️ **Save this immediately** — you can view it again later, but treat it like a password

4. **Add to Your `.env` File**
   ```bash
   PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxx
   INDEX_NAME=event-tracking-pdf-index
   ```

   > ⚠️ **`PINECONE_API_KEY` must use this exact name.** LangChain's Pinecone integration auto-detects this environment variable. If you rename it (e.g., `PINECONE_KEY`), LangChain won't find it and you'll get authentication errors.

5. **Verify Your Index Is Working**
   - Back in the Pinecone console, click on your index name
   - You should see a dashboard showing:
     - **Vectors**: 0 (empty, ready for ingestion)
     - **Dimensions**: 1536
     - **Metric**: cosine
     - **Status**: Ready
   - After running the ingestion script, come back here — you'll see the vector count increase

#### Common Mistakes When Creating the Index

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong dimensions (e.g., 768 instead of 1536) | `ValueError: vector dimension mismatch` during ingestion | Delete the index and recreate with 1536 (must match your embedding model) |
| Using "Pod-based" instead of "Serverless" | Higher costs, requires capacity planning | Serverless is fine for development and most production workloads |
| Wrong metric (e.g., `dotproduct` instead of `cosine`) | Retrieval returns irrelevant results | Delete and recreate — metric cannot be changed after creation |
| Forgetting to copy the API key | `AuthenticationError` when running scripts | Go to API Keys in the sidebar and copy again |
| Naming the env var `PINECONE_KEY` instead of `PINECONE_API_KEY` | `PineconeApiKeyError: API key is required` | LangChain expects the exact name `PINECONE_API_KEY` |

#### Data Structure in Pinecone

Each record contains:
```json
{
  "id": "unique-vector-id",
  "values": [0.23, -0.45, 0.89, ...],  // The embedding vector (1536 numbers)
  "metadata": {
	"text": "The actual chunk content...",
	"source": "mediumblog.txt"            // Where this chunk came from
  }
}
```

The `metadata` field enables **filtering** in production — restrict searches to specific documents, date ranges, or categories.

#### What `from_documents()` Does Internally

Looking at LangChain's source demystifies `PineconeVectorStore.from_documents()`:

```python
# Simplified internal logic:
def from_documents(texts, embeddings, index_name):
	for batch in batched(texts):
		vectors = embeddings.embed_documents([t.page_content for t in batch])
		index.upsert(vectors_with_metadata)
```

What you get for free by using LangChain instead of calling Pinecone directly:
- **Batching**: Splits large document sets into API-friendly batches
- **Rate limit handling**: Retries with backoff when API limits are hit
- **Threading/AsyncIO**: Concurrent embedding calls for faster ingestion
- **Universal interface**: Same code works with Chroma, Weaviate, Qdrant — swap one class name

Could we write this logic ourselves? Yes — but LangChain handles batching, rate limits, and async, and gives one interface for all vector stores.

### Other Vector Databases

| Database | Type | Key Differentiator |
|----------|------|-------|
| **Pinecone** | Managed (Cloud) | Fully managed, serverless, simple API, enterprise security. Used Here. |
| **Chroma** | Open Source (Local) | In-memory or persistent mode. Great for prototyping/POC. Python/JS SDKs. |
| **Weaviate** | Open Source / Cloud | Supports multi-modal (text, images, audio). Integrates with HuggingFace, OpenAI, LangChain. |
| **Qdrant** | Open Source / Cloud | Rust-based, high performance. Supports nested filtering, disk storage, scalar quantization. |
| **Milvus** | Open Source | Designed for trillion-scale. Separates storage/compute for horizontal scaling. |
| **FAISS** | Library (Meta) | Not a database — a similarity search library. Fast but no persistence/scaling. |
| **pgvector** | PostgreSQL Extension | Add vector search to existing Postgres. Good when you already use Postgres. |

### How to Choose a Vector Database (Production Criteria)

| Factor | What to Evaluate |
|--------|-----------------|
| **Scalability** | Can it handle your data volume growth without performance loss? |
| **Performance** | Query latency, throughput under load, multi-dimensional search speed |
| **Security** | Encryption, access controls, compliance (SOC2, HIPAA, GDPR) |
| **Cost** | Pricing model (per query, per vector, per GB) — LLM APIs already expensive |
| **Deployment** | Cloud-only vs self-hosted vs hybrid — matches your infra preferences? |
| **Integration** | Native LangChain support? SDKs in your language? |

---

## Deep Dive: Chunking

![Chunking Strategies Overview](assets/chunking_strategies.png)

### Why Chunk?

Even with 1M token context windows, chunking is critical because of **garbage in, garbage out**: sending irrelevant context degrades LLM answers, wastes tokens, and adds cost.

### Chunking Parameters

| Parameter | What It Controls | Our Value |
|-----------|-----------------|-----------|
| `chunk_size` | Maximum characters per chunk | 1000 |
| `chunk_overlap` | Characters shared between adjacent chunks | 0 |
| `separator` | Character(s) to split on | `\n\n` (double newline) |

### The Chunk Size Trade-Off

| Too Small | Too Large |
|-----------|-----------|
| Chunks lose context/meaning | Includes irrelevant information |
| Need more chunks to cover a topic | Wastes tokens and increases cost |
| More retrieval calls | Worse "needle in haystack" effect |

**Rule of thumb**: A chunk should make sense if you read it as a human. If it's gibberish in isolation, it's too small.

### Chunk Overlap

When `chunk_overlap > 0`, adjacent chunks share some text. This helps when an answer spans a chunk boundary — the overlapping text ensures both chunks contain the complete context.

```
Chunk 1: [A B C D E F G H]
Chunk 2:         [G H I J K L M N]   ← G,H appear in both (overlap = 2)
```

### Important: Chunks Are Still Documents

After splitting, each chunk is still a LangChain `Document` object — it inherits the metadata from the parent document (including `source`). So every chunk knows where it came from, enabling source attribution in answers.

### Chunk Sizes Are Approximate

Because splitting happens on separator characters (`\n\n`), some chunks may exceed `chunk_size`. If a paragraph is longer than 1000 characters and there's no separator within it, LangChain will produce a chunk larger than the limit. This is expected and usually not a problem with modern context windows (32K–1M tokens), but be aware of it.

### Advanced Chunking Strategies (Production)

| Strategy | When to Use |
|----------|-------------|
| **Character-based** | Simple text documents |
| **Token-based** | When you need precise token counts |
| **Recursive** | Structured documents (tries headers, then paragraphs, then sentences) |
| **Semantic** | When meaning boundaries don't align with formatting |
| **Code-aware** | Source code (respects function/class boundaries) |

---

## Deep Dive: The Retriever

### What is `as_retriever()`?

Calling `vectorstore.as_retriever()` returns a **VectorStoreRetriever** object — a LangChain Runnable that wraps the vector store's search functionality.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

- `k=3` means: return the top 3 most relevant chunks
- The retriever has an `invoke()` method (it's a Runnable)
- Under the hood, it calls the vendor's similarity search implementation

### Inheritance Chain

Tracing through the source code shows the class hierarchy:

```
VectorStoreRetriever → BaseRetriever → Runnable
```

Because it's a **Runnable**, the retriever has:
- `invoke()` — synchronous retrieval
- `ainvoke()` — async retrieval
- `batch()` — multiple queries at once
- It can be piped (`|`) in LCEL chains

Internally, `invoke()` calls `get_relevant_documents()`, which each vector store vendor implements differently (Pinecone uses their SDK, Chroma uses theirs, etc.).

### How Retrieval Works (Step by Step)

1. User asks: *"What is Pinecone in machine learning?"*
2. The query string is embedded into a vector using the same embedding model
3. The vector is sent to Pinecone for similarity search
4. Pinecone returns the top-k vectors (with their metadata/text)
5. LangChain wraps them as `Document` objects with `page_content` and `metadata`

### Why k=3?

- More chunks = more context = better coverage **but** also more noise
- Fewer chunks = more focused **but** might miss relevant info
- `k=3` is a practical starting point; tune based on your data and prompt size

---

## Production Considerations

### Why Not Use an Agent for RAG?

LangChain's documentation shows an "Agentic RAG" approach where retrieval is wrapped as a tool and a ReAct agent decides when to call it. **This is generally not recommended for production:**

| Agentic RAG (Not Recommended) | Deterministic RAG (Recommended) |
|------|------|
| LLM *decides* whether to search | Always searches the knowledge base |
| Can skip searches when needed | Predictable behavior every time |
| Can answer off-topic questions | Stays within business logic bounds |
| 2 LLM calls per query (reason + answer) | 1 LLM call per query |
| Higher latency and cost | Lower latency and cost |
| Vulnerable to jailbreaking | Controlled pipeline |

**When IS Agentic RAG appropriate?** When you genuinely need the agent to decide between multiple tools (web search, calculator, database, etc.) — not when retrieval should always happen.

### The Better Architecture (Coming Later)

These notes later cover **Agentic RAG with LangGraph** — based on research papers with:
- Hallucination checking
- Answer relevance validation
- Deterministic graph-based control flow
- Much more robust than a simple ReAct agent

### LangSmith Tracing

Without LCEL, each step (retrieval, formatting, LLM call) appears as separate traces — hard to debug.

With LCEL, the entire pipeline appears as **one unified trace** showing:
- Input question
- Each step's duration
- The bottleneck (usually the LLM call)
- Final output

This is critical for production debugging and optimization.

---

## Deep Dive: Streaming (Token-by-Token Responses)

### The Problem

Right now, every RAG call works like this:

```
User asks question → [8 seconds of silence] → Full answer appears at once
```

This is `.invoke()` — it waits for the **entire** LLM response to finish before returning anything. Your user stares at a blank screen for 5-10 seconds thinking the app is broken.

### What Streaming Does

Streaming sends the answer **token by token** as the LLM generates it — exactly like how ChatGPT types out its response word by word:

```
User asks question → "The" → "parcel" → "events" → "flow" → "from" → ...
```

The first token arrives within **200-500ms**. The user sees progress immediately.

### How to Use It (LCEL Makes It Free)

If you built your chain with LCEL, streaming is already available — just swap `.invoke()` for `.stream()`:

```python
# WITHOUT streaming (current) — user waits for full response
result = chain.invoke({"question": "What is CTS?"})
print(result)  # appears after 8 seconds of silence

# WITH streaming — tokens arrive immediately, one at a time
for chunk in chain.stream({"question": "What is CTS?"}):
    print(chunk, end="", flush=True)  # appears word by word, instantly
```

That's it. Same chain, same logic, one method change.

### Why Naive (Non-LCEL) Chains Can't Stream

In the naive approach, you call `llm.invoke(messages)` which returns a complete `AIMessage`. There's no way to get partial tokens because the function doesn't return until it's done.

LCEL chains are built as a **pipeline of Runnables**, and each Runnable knows how to yield partial outputs. When you call `.stream()`, the LLM Runnable yields tokens as they arrive from the API, and `StrOutputParser` passes each one through immediately.

### When to Use Streaming

| Scenario | Use `.invoke()` | Use `.stream()` |
|----------|:-:|:-:|
| Backend API returning JSON | ✅ | ❌ |
| Chatbot UI (web/mobile) | ❌ | ✅ |
| Batch processing (no user waiting) | ✅ | ❌ |
| Interactive CLI tool | ❌ | ✅ |

### Streaming in a Web API (FastAPI Example)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/ask")
async def ask(question: str):
    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
```

**C# Analogy:** This is the Python equivalent of using `IAsyncEnumerable<string>` in an ASP.NET controller — the client receives data as it's generated instead of waiting for the full response.

### The Full Method Family

| Method | What It Does | When to Use |
|--------|-------------|-------------|
| `chain.invoke()` | Runs synchronously, returns full result | Backend batch jobs |
| `chain.stream()` | Yields tokens one at a time (sync) | CLI tools |
| `chain.ainvoke()` | Async version of invoke | FastAPI/Django endpoints |
| `chain.astream()` | Async streaming (most common in web) | Chat UIs, real-time APIs |
| `chain.batch()` | Runs multiple inputs in parallel | Bulk processing |

> **Key takeaway:** You build the chain ONCE with LCEL. Then you choose `.invoke()`, `.stream()`, `.ainvoke()`, `.astream()`, or `.batch()` depending on how you want to consume it. The chain itself doesn't change.

---

## Deep Dive: Indexing Strategy (Incremental Ingestion)

### The Problem

Right now, your ingestion script re-embeds the **entire document every time** you run it:

```
Run 1: 27-page PDF → embed all 67 chunks → store in Pinecone     ✅ (first time, necessary)
Run 2: 27-page PDF → embed all 67 chunks AGAIN → duplicates!     ❌ (wasted $, duplicate vectors)
Run 3: Updated PDF (1 page changed) → embed all 67 chunks AGAIN  ❌ (only 1 chunk actually changed)
```

Problems:
- **Duplicate vectors** — same content stored multiple times, retrieval returns copies
- **Wasted money** — re-embedding unchanged content costs API calls
- **No deletion** — if you remove a section from the PDF, the old vectors are still in Pinecone

### The Solution: LangChain's Indexing API

LangChain provides a `RecordManager` + `index()` function that tracks what's already been ingested using **content hashing**:

```
Document text → SHA256 hash → "abc123"
```

Each chunk gets a unique hash based on its content. On re-ingestion:

| Chunk Status | Hash Match? | Action |
|-------------|:-----------:|--------|
| New content (never seen) | No match | Embed + insert |
| Unchanged content | Hash matches | **Skip** (save money) |
| Modified content | Hash changed | Re-embed + update |
| Deleted from source | Orphaned hash | **Remove** from vector store |

### How the RecordManager Works

The `RecordManager` is a lightweight database that stores three things per chunk:

```
┌──────────────────────────────────────────────────────┐
│ Record Manager (SQLite by default)                    │
│──────────────────────────────────────────────────────│
│ source_id     │ content_hash │ last_updated          │
│───────────────│──────────────│───────────────────────│
│ doc.pdf:p1:c1 │ sha256:abc.. │ 2025-01-15 10:30:00  │
│ doc.pdf:p1:c2 │ sha256:def.. │ 2025-01-15 10:30:00  │
│ doc.pdf:p2:c1 │ sha256:ghi.. │ 2025-01-15 10:30:00  │
└──────────────────────────────────────────────────────┘
```

When you run ingestion again, it compares the new chunks' hashes against this table and only processes the diff.

### Implementation

```python
from langchain.indexes import SQLRecordManager, index
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Create the vector store (same as before)
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name="my-index", embedding=embeddings
)

# 2. Create the Record Manager
# This uses a local SQLite file to track content hashes.
# The namespace should be unique per collection/index.
record_manager = SQLRecordManager(
    namespace="pinecone/my-index",
    db_url="sqlite:///record_manager.db"
)
record_manager.create_schema()  # Create the tracking table (first time only)

# 3. Index with cleanup mode
# cleanup="incremental" →
#   - New chunks: embed + insert
#   - Unchanged chunks: skip
#   - Modified chunks: re-embed + update
#   - Deleted chunks (from a given source): remove from vector store
result = index(
    docs_iterable=chunks,         # Your chunked documents
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental",        # The magic — only process the diff
    source_id_key="source",       # metadata key that identifies the document
)

print(result)
# {'num_added': 5, 'num_updated': 2, 'num_deleted': 1, 'num_skipped': 59}
#                                                        ^^^^^^^^^^^^^^
#                                      59 chunks unchanged = 59 embedding calls saved!
```

### Cleanup Modes Explained

| Mode | New | Changed | Deleted | Use Case |
|------|:---:|:-------:|:-------:|----------|
| `None` | Insert | Insert (duplicate!) | ❌ Never | Quick prototyping only |
| `"incremental"` | Insert | Update | Removes per-source | **Best default** — safe, efficient |
| `"full"` | Insert | Update | Removes ALL not in current batch | Full re-sync (like `git reset --hard`) |

**Best practice:** Use `"incremental"` for most cases. Use `"full"` only when you want to guarantee the vector store is an exact mirror of the source documents (e.g., nightly full re-index).

### Why SQLite? (And Production Alternatives)

The `SQLRecordManager` defaults to **SQLite** — a local file-based database. This is perfect because:
- Zero setup (no server to run)
- The tracking data is tiny (just hashes and timestamps)
- It's the same pattern NuGet uses for its local cache

**For production** (multiple ingestion workers, CI/CD pipelines):

| Environment | Record Manager Storage | Why |
|-------------|----------------------|-----|
| Local development | SQLite (`sqlite:///record_manager.db`) | Zero setup |
| Single-server production | PostgreSQL | Durable, backed up |
| Multi-worker / CI/CD | PostgreSQL or Redis | Shared state across workers |

The `SQLRecordManager` accepts any SQLAlchemy connection string:
```python
# Local SQLite (default, recommended for learning)
record_manager = SQLRecordManager("ns", db_url="sqlite:///record_manager.db")

# Production PostgreSQL
record_manager = SQLRecordManager("ns", db_url="postgresql://user:pass@host/db")
```

### C# Analogy

Think of the `RecordManager` like **EF Core Migrations**:
- Migrations track which schema changes have been applied (by hash/name)
- The RecordManager tracks which document chunks have been embedded (by content hash)
- Running migrations again skips already-applied ones — running indexing again skips already-embedded chunks
- Both prevent duplicates and keep state consistent

### When NOT to Use Indexing

- **One-time ingestion** (load once, never update) → overkill, just use `from_documents()`
- **Tiny datasets** (< 100 chunks) → re-embedding everything is cheap enough
- **Different embedding models** → hashes match but vectors don't — must re-embed everything anyway

---

## Deep Dive: Multimodal PDF RAG

### The Problem with Text-Only Loaders

Production PDFs aren't plain text. They contain **diagrams, flowcharts, tables, and complex layouts** that text-only loaders silently drop:

| Loader | Text | Tables | Diagrams/Flowcharts | Cost |
|--------|:----:|:------:|:-------------------:|------|
| `PyPDFLoader` | ✅ | ❌ | ❌ | Free |
| `PyMuPDFLoader` | ✅ | ⚠️ Partial | ❌ | Free |
| `UnstructuredPDFLoader` | ✅ | ✅ | ❌ | Free* |
| **Multimodal Vision (GPT-4o)** | ✅ | ✅ | ✅ | ~$0.01-0.03/page |

### The Multimodal Approach

Instead of extracting text, render each PDF page as an **image** and send it to GPT-4o Vision:

```
PDF Page → Render as Image (PyMuPDF) → GPT-4o Vision → Structured Text Description → Embed → Store
```

The Vision model "reads" everything a human would see — text, diagrams, flowcharts, tables, handwritten notes — and produces a rich text description that captures the full semantic content.

### When to Use Each Loader

| Scenario | Recommended Loader |
|----------|-------------------|
| Plain text PDFs (contracts, articles) | `PyPDFLoader` (free, fast) |
| PDFs with tables but no images | `UnstructuredPDFLoader` |
| Engineering docs, architecture diagrams | **Multimodal Vision** |
| Mixed batch (some simple, some complex) | **Multimodal Vision** (safest default) |

**Key insight:** You only ingest once. Spending $1 to properly process a 50-page PDF is worth it if every query returns accurate results instead of missing diagram context.

> → See working implementation: [`src/test_multimodal_pdf_rag.py`](src/test_multimodal_pdf_rag.py)

---

## Deep Dive: Conversational RAG (Multi-Turn)

### The Problem with Single-Shot RAG

Basic RAG is **stateless** — each query is independent. This breaks for follow-up questions:

```
Q1: "What is CTS?"                        → Retrieves CTS docs ✅
Q2: "How does the Hub connect to it?"      → Retrieves docs about "it" ❌ GARBAGE
```

The retriever doesn't know "it" refers to CTS from the previous question.

### The Solution: Question Reformulation

Before searching, use the LLM to **rewrite** follow-up questions into standalone queries:

```
Chat History: [("What is CTS?", "CTS is the Centralized Tracking System...")]
Follow-up:    "How does the Hub connect to it?"
Reformulated: "How does the Hub connect to the Centralized Tracking System (CTS)?"
```

The reformulated question is self-contained — the retriever can find relevant chunks without context.

### Architecture

```
User Question → Reformulate (with history) → Retrieve → Generate → Update History
```

### Two Cases to Handle

| Case | What Happens |
|------|-------------|
| **Follow-up** (references prior context) | Rewrite with resolved pronouns/references |
| **New topic** (completely different subject) | Return unchanged — don't over-connect to prior context |

This distinction is critical. A naive reformulation prompt will force connections between unrelated questions, producing worse retrieval than no reformulation at all.

### Memory Strategies (Production)

| Strategy | Best For | Trade-Off |
|----------|----------|-----------|
| **Last N turns** (list) | Short conversations | Simple, but unbounded growth |
| **Summary memory** | Long conversations | Compresses history, loses detail |
| **Database-backed** (Redis, PostgreSQL) | Persistent across sessions | Requires infra, but production-ready |

> → See working implementation: [`src/test_conversational_rag.py`](src/test_conversational_rag.py)

---

## Deep Dive: RAG Evaluation & Quality

### How Do You Know Your RAG System Works?

Building a RAG pipeline is only half the work. In production, you need to **measure** whether retrieval is returning the right chunks and whether the LLM is generating grounded answers.

### The Three Dimensions of RAG Quality

| Dimension | What It Measures | Question It Answers |
|-----------|-----------------|---------------------|
| **Retrieval Relevance** | Are the retrieved chunks actually relevant? | "Did the retriever find the right documents?" |
| **Answer Faithfulness** | Is the answer grounded in the context? | "Did the LLM stick to the retrieved facts?" |
| **Answer Relevance** | Does the answer address the question? | "Did the LLM actually answer what was asked?" |

### Common RAG Failure Modes

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| **Wrong chunks retrieved** | Correct answer exists but isn't found | Bad chunking, wrong k, embedding mismatch | Adjust chunk size, increase k, verify same model |
| **Hallucination despite context** | Answer includes facts not in chunks | Weak grounding prompt, model tendency | Strengthen "only use the context" instruction |
| **Partial answer** | Answer is correct but incomplete | Relevant info split across chunks not retrieved | Increase k, add chunk overlap |
| **Refusal to answer** | "I don't have enough information" | Overly strict grounding prompt | Relax constraints or improve retrieval |

### Quick Quality Checks (No Framework Needed)

Before reaching for evaluation frameworks, try these manual checks:

1. **Eyeball test**: Ask 10 questions you know the answer to. Check if retrieved chunks contain the answer.
2. **Source attribution**: Does the system tell you *which* chunks it used? If not, add metadata logging.
3. **Adversarial test**: Ask questions where the answer is NOT in the documents. Does it hallucinate or say "I don't know"?
4. **Edge cases**: Typos, multilingual queries, ambiguous questions — does retrieval degrade gracefully?

### Evaluation Frameworks (When You Scale)

| Framework | What It Does |
|-----------|-------------|
| **RAGAS** | Automated RAG evaluation with faithfulness, relevance, and context recall metrics |
| **LangSmith** | Trace-level debugging + dataset-based evaluation with custom evaluators |
| **DeepEval** | Unit testing for LLM outputs with built-in metrics |
| **TruLens** | Feedback functions that score retrieval and generation quality |

> **In practice**: LangSmith tracing gives you immediate visibility into what the retriever returns and what the LLM generates. Start there before adding automated evaluation.

> 🚀 **Runnable example:** [test_rag_evaluation.py](src/test_rag_evaluation.py) — LLM-as-judge evaluation scoring faithfulness, answer relevance, and retrieval relevance across a test dataset.

---

## Interview Q&A Anchors

**Q: What is RAG and why do we need it?**

> **A:** RAG stands for Retrieval Augmented Generation. It solves the problem of LLMs not knowing about private or recent data. Instead of stuffing the entire document into the prompt (which hits token limits, costs more, adds latency, and reduces accuracy), RAG retrieves only the relevant chunks, augments the prompt with them, and generates a grounded answer. The three letters map directly to the pipeline: **R**etrieve relevant chunks → **A**ugment the prompt with context → **G**enerate the answer.

**Q: What are the four problems with stuffing the entire document into the prompt?**

> **A:** (1) **Token limit** — exceeding the context window causes request failure. (2) **Needle in a haystack** — longer prompts reduce LLM accuracy (proven by research). (3) **Cost** — more tokens = higher API bills. (4) **Latency** — more tokens = slower response times.

**Q: What is an embedding and why is it useful for RAG?**

> **A:** An embedding is a vector (list of numbers) that represents text in a high-dimensional space where semantically similar texts are close together. This enables us to convert a user's question into a vector, then use mathematical distance calculations to find the stored chunks whose vectors are closest — meaning they're semantically most relevant to the question.

**Q: What is a vector database and how is it different from a regular database?**

> **A:** A vector database stores high-dimensional vectors and provides fast nearest-neighbor similarity search. Unlike regular databases that do exact-match queries (`WHERE id = 5`), vector databases find items that are *semantically similar* to a query vector — essential for finding relevant document chunks in RAG.

**Q: Why do we split documents into chunks even with 1M token context windows?**

> **A:** Because of "garbage in, garbage out." Even if the context window is large enough, sending irrelevant text (1) costs more tokens, (2) proven by research to degrade answer quality (needle in haystack problem), and (3) increases latency. Chunking lets us send only the relevant portions.

**Q: What is the difference between the naive RAG implementation and the LCEL version?**

> **A:** Both produce the same answer. The naive version calls each step manually (retrieve → format → prompt → LLM) as separate function calls. The LCEL version composes them into a single chain using the `|` operator, which gives you: built-in streaming (`.stream()`), async support (`.ainvoke()`), batch processing (`.batch()`), better composability with other chains, and unified LangSmith tracing where all steps appear under one trace.

**Q: What does `RunnablePassthrough.assign()` do?**

> **A:** It passes the input dictionary through unchanged while computing and adding new keys. In our RAG chain, the input is `{"question": "..."}`. `RunnablePassthrough.assign(context=...)` keeps the `question` key and adds a `context` key computed by the retrieval sub-chain. The output `{"question": "...", "context": "..."}` then feeds into the prompt template.

**Q: Why prefer deterministic RAG over agentic RAG for production?**

> **A:** In production (e.g., customer support), you always want to query the knowledge base — there's no decision to make. Wrapping retrieval as a tool inside a ReAct agent adds: an extra LLM call (to decide whether to search), latency, cost, and vulnerability to jailbreaking. It also means the agent might skip the search or answer off-topic questions. Deterministic RAG guarantees predictable behavior, lower cost, and stays within business logic.

**Q: What is the `Document` object in LangChain and why does it have a `metadata` field?**

> **A:** A LangChain `Document` has two attributes: `page_content` (the text) and `metadata` (a dict). The `metadata.source` field provides **provenance** — it tells users where the answer came from (which file, which page). In production, you can add custom metadata (timestamps, categories, access control) for filtering during retrieval.

**Q: Why does LangChain auto-convert Python functions to `RunnableLambda` in LCEL chains?**

> **A:** When you pipe a regular Python function (like `format_docs`) in an LCEL chain, LangChain automatically wraps it in a `RunnableLambda`. This gives it the full Runnable interface (`invoke()`, `stream()`, `batch()`, `ainvoke()`) so it can participate in the chain's streaming and async capabilities without you writing boilerplate.

**Q: What happens if you change your embedding model after ingestion?**

> **A:** The vectors become incompatible. Each embedding model maps text to a different vector space. Even if two models produce the same dimensionality (e.g., both 1536), the vector spaces are different. You must **re-embed all documents** with the new model. This is why embedding model choice is an important upfront decision.

**Q: What is the `length_function` parameter in text splitters and when would you change it?**

> **A:** By default, chunk size is measured in characters using `len()`. But if you need precise control over token counts (important for cost optimization), you can provide a custom function that counts tokens using the model's tokenizer (e.g., `tiktoken` for OpenAI). This ensures chunks never exceed a specific token count.

**Q: Why is LangChain's one-interface pattern important for vector stores specifically?**

> **A:** Vector stores differ widely in their APIs, authentication, query syntax, and data formats. LangChain abstracts this behind a common interface (`from_documents()`, `as_retriever()`, `similarity_search()`). This means you can prototype with Chroma (free, local) and deploy with Pinecone (managed, scalable) by changing one class name — no rewrite needed.

**Q: What is streaming in LCEL and why does it matter?**

> **A:** Streaming sends the LLM's response token by token as it's generated, rather than waiting for the full response. In a chat UI, this means the user sees the first word within 200-500ms instead of waiting 5-10 seconds for the complete answer. LCEL chains support this out of the box — you just call `.stream()` instead of `.invoke()` on the same chain. The naive (non-LCEL) approach can't stream because `llm.invoke()` blocks until the full response is ready.

**Q: What is LangChain's Indexing API and when would you use it?**

> **A:** The Indexing API (`RecordManager` + `index()`) solves the problem of re-ingesting unchanged documents. It computes a content hash for each chunk and stores it in a tracking database (SQLite locally, PostgreSQL in production). On re-ingestion, it compares hashes: new chunks get embedded, unchanged chunks are skipped, modified chunks are re-embedded, and deleted chunks are removed from the vector store. This saves embedding costs and prevents duplicate vectors. Use `cleanup="incremental"` as the default mode.

**Q: How do you handle PDFs with diagrams and flowcharts in RAG?**

> **A:** Text-only loaders (`PyPDFLoader`) silently drop all visual content. The multimodal approach renders each PDF page as an image and sends it to a Vision model (GPT-4o), which produces a rich text description of everything on the page — text, tables, diagrams, flowcharts. This text is then embedded and stored normally. You only ingest once, so the per-page vision cost (~$0.01-0.03) is worth it for accurate retrieval.

**Q: How do you handle follow-up questions in a RAG chatbot?**

> **A:** Basic RAG is stateless — it can't resolve pronouns like "it" or "that" from prior turns. The solution is **question reformulation**: before retrieval, send the chat history + follow-up to the LLM to rewrite it as a standalone query. "How does it connect?" becomes "How does the Hub connect to the CTS system?" The key nuance: completely unrelated new questions should be returned unchanged — don't over-connect to prior context.

**Q: What are the three dimensions of RAG quality?**

> **A:** (1) **Retrieval relevance** — did the retriever find the right chunks? (2) **Answer faithfulness** — is the answer grounded in the retrieved context, not hallucinated? (3) **Answer relevance** — does the answer actually address the question asked? You can measure these manually with eyeball tests or at scale with frameworks like RAGAS, LangSmith evaluations, or DeepEval.

---

## References

- [LangChain RAG Tutorial](https://docs.langchain.com/)
- [LangChain Python Build Overview](https://docs.langchain.com/oss/python/build-overview)
- [LangChain API Reference](https://reference.langchain.com/python/langchain)
- [What is RAG? (Medium)](https://medium.com/@drjulija/what-is-retrieval-augmented-generation-rag-938e4f6e03d1)
- [Needle in a Haystack Research](https://arxiv.org/abs/2307.03172)
- [Pinecone Documentation](https://docs.pinecone.io/)
