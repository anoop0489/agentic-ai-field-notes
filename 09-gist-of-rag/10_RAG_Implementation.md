# 10. RAG Implementation — From Ingestion to Retrieval

Step-by-step implementation guide: ingesting documents into a vector store, then retrieving relevant context to answer questions with an LLM.

*Topic: Medium Blog Analyzer Project*

---

## Project Overview

We build a complete RAG system that:
1. **Ingests** a Medium blog post about vector databases into Pinecone
2. **Retrieves** relevant chunks when asked "What is Pinecone in machine learning?"
3. **Generates** a grounded answer using the retrieved context

### Project Structure

```
09-gist-of-rag/
├── assets/
│   └── basic_rag_pipeline.png          # Visual diagram of the pipeline
├── src/
│   ├── mediumblog.txt                  # Sample document (Medium blog about vector DBs)
│   ├── ingestion.py                    # Phase 1: Load → Chunk → Embed → Store
│   ├── main.py                         # Phase 2: Retrieve → Augment → Generate
│   ├── test_multimodal_pdf_rag.py      # PDF RAG with 4-loader comparison (vision default)
│   ├── test_conversational_rag.py      # Multi-turn RAG with question reformulation
│   ├── test_streaming_rag.py           # .invoke() vs .stream() side-by-side comparison
│   ├── test_indexing_strategy.py       # Incremental ingestion with RecordManager
│   └── test_rag_evaluation.py          # LLM-as-judge evaluation (faithfulness, relevance)
├── 09_RAG_Theory_And_Concepts.md       # Theory & definitions (previous file)
└── 10_RAG_Implementation.md            # This file (implementation walkthrough)
```

---

## 📦 Dependencies

```bash
uv add langchain langchain-openai langchain-pinecone langchain-community langchain-text-splitters python-dotenv black isort
```

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
INDEX_NAME=medium-blogs-embeddings-index

# LangSmith (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-gist
```

---

## Phase 1: Ingestion Pipeline (`ingestion.py`)

The ingestion pipeline runs **once** (or when documents change) to populate the vector store.

### Pipeline Flow

```
mediumblog.txt → TextLoader → Document → CharacterTextSplitter → 20 Chunks
																	   ↓
											  Pinecone ← OpenAIEmbeddings ← Chunks
```

### Step-by-Step Walkthrough

#### Step 1: Load the Document

```python
loader = TextLoader("mediumblog.txt")
document = loader.load()  # Returns List[Document] with one Document
```

**What happens:**
- `TextLoader` opens the file and reads its content
- Returns a list containing one `Document` object
- The `Document` has:
  - `page_content`: The full text of the file
  - `metadata`: `{"source": "mediumblog.txt"}`

**Why LangChain's `TextLoader`?**  
Same interface for all sources. Replace `TextLoader` with `PyPDFLoader`, `NotionDirectoryLoader`, `SlackChatLoader`, etc. — the downstream code stays the same.

> **Encoding Tip:** If you get `UnicodeDecodeError`, add `encoding="utf-8"` or `autodetect_encoding=True` to the loader constructor.

#### Step 2: Split into Chunks

```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
# Result: 20 LangChain Document objects, each with ~1000 chars of content
```

**What happens:**
- The text is split on `\n\n` (double newline) by default
- Each chunk is ≤ 1000 characters
- `chunk_overlap=0` means no shared text between adjacent chunks
- Each chunk is still a `Document` with its own `page_content` and inherited `metadata`

**Why chunk_size=1000?**
- Small enough to fit several chunks in a prompt
- Large enough that each chunk has meaningful semantic content
- **Rule of thumb**: If you read a chunk as a human and it makes sense, the size is good

#### Step 3: Embed and Store

```python
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

PineconeVectorStore.from_documents(
	texts, embeddings, index_name=os.environ["INDEX_NAME"]
)
```

**What happens under the hood:**
1. LangChain iterates through all 20 chunks
2. Each chunk's `page_content` is sent to the OpenAI Embeddings API
3. OpenAI returns a 1536-dimension vector for each chunk
4. LangChain stores each vector + metadata in Pinecone
5. Rate limiting and batching are handled automatically

**Why use LangChain for this?**
- Handles batching (avoids rate limits)
- Supports async/threading for large document sets
- One interface for all vector stores (swap Pinecone for Chroma with one line)

---

## Phase 2: Retrieval Pipeline (`main.py`)

The retrieval pipeline runs **per user query** — it retrieves relevant context and generates an answer.

### Shared Setup

```python
# Initialize all components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()  # defaults to gpt-3.5-turbo (or latest)

vectorstore = PineconeVectorStore(
	index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
	"""Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)
```

**Key component: The Prompt Template**

This is the **augmentation** step of RAG. The template has two placeholders:
- `{context}` — where the retrieved chunks go
- `{question}` — the user's original query

The instruction "based **only** on the following context" prevents the LLM from using its parametric knowledge — it must ground its answer in the provided chunks.

---

### Implementation 1: Naive (Without LCEL)

The simplest possible RAG implementation — manual function calls, step by step.

```python
def retrieval_chain_without_lcel(query: str):
	# Step 1: Retrieve relevant documents from vector store
	docs = retriever.invoke(query)

	# Step 2: Format documents into a single context string
	context = format_docs(docs)

	# Step 3: Populate prompt template with context and question
	messages = prompt_template.format_messages(context=context, question=query)

	# Step 4: Invoke LLM with the augmented prompt
	response = llm.invoke(messages)

	# Step 5: Return the answer text
	return response.content
```

#### What Each Step Does

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| 1. Retrieve | `"what is Pinecone?"` (string) | `[Doc1, Doc2, Doc3]` (list of Documents) | Embeds query, searches Pinecone, returns top-3 |
| 2. Format | `[Doc1, Doc2, Doc3]` | `"chunk1\n\nchunk2\n\nchunk3"` (string) | Joins page_content with double newlines |
| 3. Prompt | context + question | `[HumanMessage(content="Answer the question...")]` | Fills template placeholders |
| 4. LLM | `[HumanMessage]` | `AIMessage(content="Pinecone is a...")` | Full LLM inference call |
| 5. Return | `AIMessage` | `"Pinecone is a..."` (string) | Extracts `.content` from response |

#### Limitations

- ❌ No streaming support (must wait for full response)
- ❌ No async support (blocks the thread)
- ❌ Hard to compose with other chains
- ❌ Each step traces separately in LangSmith (hard to debug)
- ❌ More verbose and error-prone

---

### Implementation 2: LCEL (LangChain Expression Language)

The same pipeline, but composed as a single declarative chain.

```python
def create_retrieval_chain_with_lcel():
	retrieval_chain = (
		RunnablePassthrough.assign(
			context=itemgetter("question") | retriever | format_docs
		)
		| prompt_template
		| llm
		| StrOutputParser()
	)
	return retrieval_chain
```

**Invocation:**
```python
chain = create_retrieval_chain_with_lcel()
result = chain.invoke({"question": "what is Pinecone in machine learning?"})
```

#### Breaking Down the Chain (Left to Right)

```
Input: {"question": "what is Pinecone?"}
		   │
		   ▼
┌───────────────────────────────────────────────-┐
│  RunnablePassthrough.assign(context=...)       │
│                                                │
│  1. Passes input through unchanged:            │
│     {"question": "what is Pinecone?"}          │
│                                                │
│  2. Computes new key "context":                │
│     itemgetter("question")                     │  → extracts "what is Pinecone?"
│       | retriever                              │  → [Doc1, Doc2, Doc3]
│       | format_docs                            │  → "chunk1\n\nchunk2\n\nchunk3"
│                                                │
│  Output: {"question": "...", "context": "..."} │
└─────────────────────────────────────────────-──┘
		   │
		   ▼
┌──────────────────────────┐
│  prompt_template         │  → HumanMessage with context + question filled in
└──────────────────────────┘
		   │
		   ▼
┌──────────────────────────┐
│  llm                     │  → AIMessage(content="Pinecone is a...")
└──────────────────────────┘
		   │
		   ▼
┌──────────────────────────┐
│  StrOutputParser()       │  → "Pinecone is a..." (just the string)
└──────────────────────────┘
```

#### Key Concepts

**`itemgetter("question")`**
- Python's `operator.itemgetter` — creates a callable that extracts a key from a dict
- Equivalent to: `lambda x: x["question"]`
- Used to pull out just the question string to pass to the retriever

**`RunnablePassthrough.assign(context=...)`**
- Passes the input dict through unchanged
- Computes a new key (`context`) by running the sub-chain
- Merges the new key into the output dict
- Result: `{"question": "...", "context": "retrieved text..."}`

**`format_docs` as a function in the chain**
- Regular Python functions in LCEL chains are auto-converted to `RunnableLambda`
- No need to wrap it — just pipe it directly
- The `RunnableLambda` wrapper gives it `invoke()`, `stream()`, `batch()`, and `ainvoke()` methods
- Note: when we use regular Python functions in a LCEL chain, LangChain automatically converts them into runnable lambdas that adhere to the runnable interface

**`StrOutputParser()`**
- The LLM returns an `AIMessage` object (with `.content`, `.tool_calls`, etc.)
- `StrOutputParser()` simply extracts the `.content` string
- Equivalent to: `lambda msg: msg.content`
- Makes the chain output a clean string instead of a message object

**Why the function returns a chain (not taking arguments)**
- Notice `create_retrieval_chain_with_lcel()` takes **no arguments** and returns a chain
- The chain itself is a Runnable — you call `.invoke({"question": "..."})` on it
- This is LangChain's pattern: build chains as reusable objects, invoke them separately

#### Advantages Over Naive Approach

| Feature | Naive | LCEL |
|---------|-------|------|
| **Streaming** | ❌ Manual | ✅ `chain.stream()` |
| **Async** | ❌ Manual | ✅ `chain.ainvoke()` |
| **Batch** | ❌ Manual loop | ✅ `chain.batch()` |
| **Composability** | ❌ Hard to reuse | ✅ Pipe into other chains |
| **Tracing** | ❌ Separate traces | ✅ One unified LangSmith trace |
| **Code style** | Imperative | Declarative |

---

## LangSmith Tracing Comparison

### Without LCEL (Fragmented)
Each component traces independently — you see separate entries for:
- Vector store retrieval
- LLM invocation

No parent trace connects them. Debugging requires manually correlating timestamps.

### With LCEL (Unified)
Everything appears under **one `RunnableSequence` trace**:

```
RunnableSequence (1.2s total)
├── RunnablePassthrough.assign (0.8s)
│   ├── itemgetter → "what is Pinecone?"
│   ├── VectorStoreRetriever → [3 documents]
│   └── format_docs → "chunk1\n\nchunk2..."
├── ChatPromptTemplate → [HumanMessage]
├── ChatOpenAI (0.4s) → AIMessage
└── StrOutputParser → "Pinecone is a..."
```

You can see:
- Total pipeline duration
- Which step is the bottleneck
- Input/output of every step
- The exact context that was retrieved

---

## Demonstrating RAG vs No-RAG

The `main.py` file runs three implementations to compare:

### Without RAG (Baseline)
```python
result = llm.invoke([HumanMessage(content="what is Pinecone in machine learning?")])
# GPT-3.5 answer: "A pinecone algorithm is a method used in machine learning..."
# ❌ WRONG — hallucinated a non-existent algorithm
```

### With RAG
```python
result = retrieval_chain_without_lcel("what is Pinecone in machine learning?")
# Answer: "Pinecone is a fully managed cloud-based vector database specifically 
# designed for businesses looking to build large-scale ML applications."
# ✅ CORRECT — grounded in the retrieved blog content
```

This demonstrates exactly why RAG exists: the LLM's parametric knowledge was wrong (or outdated), but with the relevant context injected, it answers correctly.

---

## The `format_docs` Helper

```python
def format_docs(docs):
	"""Format retrieved documents into a single string."""
	return "\n\n".join(doc.page_content for doc in docs)
```

Simple but critical:
- Takes a list of `Document` objects
- Extracts `page_content` from each
- Joins them with double newlines
- Returns a single string ready to inject into the prompt

---

## Running the Project

```bash
# 1. Ensure Pinecone index exists and .env is configured

# 2. Ingest the document (one-time)
uv run python 09-gist-of-rag/src/ingestion.py

# 3. Run the retrieval pipeline
uv run python 09-gist-of-rag/src/main.py
```

**Expected Output:**
```
IMPLEMENTATION 0: Raw LLM Invocation (No RAG)
Answer: [Incorrect/hallucinated answer about pinecones]

IMPLEMENTATION 1: Without LCEL
Answer: Pinecone is a fully managed cloud-based vector database...

IMPLEMENTATION 2: With LCEL - Better Approach
Answer: Pinecone is a fully managed cloud-based vector database...
```

---

## A Critique of LangChain's RAG Documentation

It's worth discussing the trade-offs in LangChain's official RAG tutorial:

### What the Docs Show (Agentic RAG)
LangChain's tutorial creates a ReAct agent with retrieval as a tool:
```python
# LangChain docs approach — NOT recommended for production
retriever_tool = create_retriever_tool(retriever, ...)
agent = create_react_agent(llm, [retriever_tool], prompt)
```

### The Criticism

| Issue | Problem |
|-------|---------|
| **LLM decides whether to search** | May skip search when it's needed, or search when unnecessary |
| **Two inference calls** | One to decide tool use, one to generate answer — double the cost/latency |
| **Agent can go off-script** | Can answer unrelated questions, vulnerable to jailbreaking |
| **`create_agent` is a black box** | Internal behavior can change between LangChain versions |
| **Too abstracted** | Hard to know what's happening under the hood |

### Recommended Approach
For standard RAG applications, use a **deterministic pipeline** (like our LCEL chain):
- Always retrieves context (no decision-making)
- Single LLM call
- Predictable, testable behavior
- Full control over every step

### When IS Agentic RAG Appropriate?
Later in these notes: **LangGraph-based Agentic RAG** — research-backed architecture with hallucination detection, answer validation, and graph-based control flow. This is the production-grade approach when you genuinely need agent autonomy.

---

## Summary

| Concept | One-Line Takeaway |
|---------|-------------------|
| **RAG** | Retrieve → Augment → Generate. Don't stuff, search. |
| **Ingestion** | Load → Chunk → Embed → Store (one-time offline job) |
| **Retrieval** | Query → Embed → Similarity Search → Top-K → Context |
| **Naive vs LCEL** | Same result, LCEL gives streaming/async/tracing for free |
| **RunnablePassthrough.assign** | Keep the input, compute and add new keys |
| **Agentic vs Deterministic** | If retrieval should always happen, don't make it optional |
| **LangSmith** | LCEL = one trace for the entire pipeline = easy debugging |

---

## Beyond Basic RAG: Production Extensions

The examples above cover the foundational RAG pipeline. Two real-world patterns extend this significantly:

### 1. Multimodal PDF RAG — [`src/test_multimodal_pdf_rag.py`](src/test_multimodal_pdf_rag.py)

**Problem:** Production PDFs contain diagrams, flowcharts, tables, and complex layouts. Text-only loaders (`PyPDFLoader`) silently lose all visual information.

**Solution:** Convert each PDF page to an image and use GPT-4o Vision to "read" it — text, tables, diagrams, and all.

| Loader | Text | Tables | Diagrams | Cost |
|--------|:----:|:------:|:--------:|------|
| PyPDFLoader | ✅ | ❌ | ❌ | Free |
| PyMuPDFLoader | ✅ | ⚠️ | ❌ | Free |
| UnstructuredPDFLoader | ✅ | ✅ | ❌ | Free* |
| **Multimodal Vision (GPT-4o)** | ✅ | ✅ | ✅ | ~$0.01-0.03/page |

**Key insight:** You only ingest once. Spending $1 to process a 50-page PDF properly is worth it if every query against it returns accurate results instead of missing diagram context.

→ Run: `uv run test_multimodal_pdf_rag.py`

---

### 2. Conversational RAG — [`src/test_conversational_rag.py`](src/test_conversational_rag.py)

**Problem:** Single-shot RAG forgets everything between questions. Follow-ups with pronouns break retrieval:
- Q1: "What is CTS?" → works fine
- Q2: "How does the Hub connect to **it**?" → retriever searches for "it" → garbage results

**Solution:** **Question reformulation** — before searching, use the LLM to rewrite follow-ups as standalone queries:
```
"How does the Hub connect to it?"
  → Reformulated: "How does the Hub connect to the Centralized Tracking System (CTS)?"
```

**Two cases handled:**
1. Follow-up referencing prior context → reformulate (resolve pronouns)
2. Completely new topic → return unchanged (don't over-connect)

**Architecture:** `User Question → Reformulate → Retrieve → Generate → Update History`

**Production memory strategies:**
- **Short conversations** → keep last N turns in a list (our approach)
- **Long conversations** → `ConversationSummaryMemory` (summarizes older turns)
- **Persistent across sessions** → store in a database (Redis, PostgreSQL)

→ Run: `uv run test_conversational_rag.py`

---

### 3. Streaming RAG — [`src/test_streaming_rag.py`](src/test_streaming_rag.py)

**Problem:** With `.invoke()`, the user stares at a blank screen for 5-10 seconds while the LLM generates the full response. The entire answer appears at once.

**Solution:** Swap `.invoke()` for `.stream()` on the **same LCEL chain** — tokens arrive one at a time as the LLM generates them. First token appears within 200-500ms.

| Method | Time to First Token | User Experience |
|--------|:-------------------:|----------------|
| `.invoke()` | 5-10 seconds | Blank screen → wall of text |
| `.stream()` | 200-500ms | Words appear as LLM thinks |

**Key insight:** You build the chain ONCE. Then choose how to consume: `.invoke()`, `.stream()`, `.ainvoke()`, `.astream()`, or `.batch()`. The chain logic doesn't change.

→ Run: `uv run test_streaming_rag.py`

---

### 4. Indexing Strategy — [`src/test_indexing_strategy.py`](src/test_indexing_strategy.py)

**Problem:** Running ingestion twice re-embeds every chunk — duplicates in the vector store, wasted embedding costs, and no way to delete removed content.

**Solution:** LangChain's `RecordManager` + `index()` with `cleanup="incremental"` — tracks content hashes in SQLite (or PostgreSQL in production). On re-ingestion:

| Chunk Status | Action | API Cost |
|:------------|:-------|:--------:|
| New (never seen) | Embed + insert | 💰 |
| Unchanged | **Skip** | 🆓 |
| Modified | Re-embed + update | 💰 |
| Deleted from source | **Remove** from vector store | 🆓 |

**C# Analogy:** Like EF Core Migrations — running `dotnet ef database update` skips already-applied migrations. The RecordManager does the same for embeddings.

→ Run: `uv run test_indexing_strategy.py`
