# 11. Documentation Assistant — Theory & Concepts

A comprehensive guide to building a production-grade documentation assistant: web crawling, agentic RAG, Streamlit UI, and conversational memory.

*Topic: Building a Documentation Assistant*

---

## 📑 Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Key Definitions](#key-definitions-interview-ready) | 15+ terms covering web crawling, agentic RAG, Streamlit, and memory |
| 2 | [What We're Building](#what-were-building) | End-to-end architecture of the documentation helper |
| 3 | [Web Crawling with Tavily](#deep-dive-web-crawling-with-tavily) | TavilyCrawl, TavilyMap, TavilyExtract — all three products in detail |
| 4 | [RecursiveCharacterTextSplitter](#deep-dive-recursivecharactertextsplitter) | Why it's better than CharacterTextSplitter, how it splits hierarchically |
| 5 | [Pinecone Integrated vs Bring Your Own](#deep-dive-pinecone-integrated-vs-bring-your-own-embeddings) | Why we embed with OpenAI and just store in Pinecone |
| 6 | [Async Batch Ingestion](#deep-dive-async-batch-ingestion) | Python async/await for C# devs, concurrent batching, `asyncio.gather()` |
| 7 | [Agentic RAG with Tools](#deep-dive-agentic-rag-with-tools) | `create_agent()`, `@tool` decorator, `content_and_artifact` pattern |
| 8 | [init_chat_model()](#deep-dive-init_chat_model) | Provider-agnostic model initialization |
| 9 | [Streamlit Chat UI](#deep-dive-streamlit-chat-ui) | Building interactive chat interfaces, session state, streaming |
| 10 | [Memory via Session State](#deep-dive-memory-via-session-state) | How Streamlit's session state provides conversational memory |
| 11 | [Deterministic vs Agentic RAG](#deterministic-vs-agentic-rag--why-agentic-here) | Why agentic for this project, agent message flow, multiple tool calls, reference code limitations |
| 12 | [Interview Q&A](#interview-qa-anchors) | 16 section-specific questions + link to architecture Q&As |

---

## What is this section about?

In Section 9, we built a **deterministic** RAG pipeline on a single PDF. In Section 10, we scale this into a **real application** — a documentation assistant (lightweight Cursor/chat.langchain.com clone) that:

1. **Crawls** an entire documentation site (not just one file)
2. **Embeds** thousands of pages using async batch processing
3. **Retrieves** context using an **agent** (not a fixed pipeline)
4. **Presents** results in a Streamlit chat UI with memory

This is the first time these notes build a **user-facing application** — the concepts from Sections 2-9 all come together here.

---

## Key Definitions (Interview-Ready)

| Term | Quick Recall (say this first) | Full Definition |
|------|------|------------|
| **Tavily** | "AI-native web search/crawl API" | A search and crawling service designed specifically for AI applications — returns structured content (not raw HTML) that's ready for LLM consumption. We use the **`langchain-tavily`** package (LangChain integration wrapper), not the raw `tavily-python` SDK. |
| **TavilyCrawl** | "Website → structured content at scale" | Tavily's crawling API that traverses a website following links, extracting clean text content from each page. Returns structured results with URL, raw_content, and metadata. |
| **TavilyMap** | "Discover all URLs on a site" | Maps a website's structure by finding all reachable URLs up to a configurable depth/breadth. Used to discover what pages exist before crawling them. |
| **TavilyExtract** | "Single URL → clean content" | Extracts structured content from a specific URL. Like a smart scraper that returns clean text instead of raw HTML. |
| **RecursiveCharacterTextSplitter** | "Split by structure, then by character" | A hierarchical text splitter that tries multiple separators in order (paragraphs → sentences → words), keeping chunks semantically coherent. Preferred over `CharacterTextSplitter` for production. |
| **`create_agent()`** | "Model + tools + prompt → ready agent" | LangChain's high-level factory function that creates a tool-calling agent from a model, list of tools, and optional system prompt. Replaces the older `initialize_agent()` pattern. |
| **`@tool`** | "Python function → LangChain Tool" | A decorator that converts a regular Python function into a LangChain Tool with auto-generated name, description (from docstring), and schema (from type hints). |
| **`response_format="content_and_artifact"`** | "Return text for LLM + raw data for app" | A tool configuration that returns two values: (1) serialized text the LLM sees as context, and (2) raw Python objects (like Document lists) the application code can access from the ToolMessage artifact. |
| **`init_chat_model()`** | "Provider-agnostic model factory" | A LangChain function that initializes any chat model by name + provider string. Swap `"gpt-4o"` → `"claude-3"` by changing two strings — no import changes needed. |
| **Streamlit** | "Python script → web app in minutes" | A Python framework that turns scripts into interactive web applications. No HTML/CSS/JS needed — just Python decorators and function calls. |
| **`st.session_state`** | "Per-user persistent dictionary" | Streamlit's mechanism for maintaining state between reruns. Survives widget interactions — used for chat history, user preferences, and any data that must persist across renders. |
| **`st.chat_message`** | "Render a chat bubble" | A Streamlit component that displays a message in a chat-style bubble with role avatar (user/assistant). Handles markdown formatting automatically. |
| **`st.chat_input`** | "Chat input box with Enter-to-send" | A Streamlit widget that renders a fixed-position text input at the bottom of the page, like ChatGPT's input box. Returns the text when submitted. |
| **ToolMessage** | "Tool execution result sent back to agent" | A LangChain message type that carries the output of a tool call back to the LLM. Has `content` (text the LLM reads) and optional `artifact` (raw data for the app). |
| **Artifact (Tool)** | "Raw structured data from tool execution" | The second return value when using `response_format="content_and_artifact"`. Contains Python objects (Document lists, dicts) that the app uses — separate from the serialized text the LLM reads. |

---

## What We're Building

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENTATION ASSISTANT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: INGESTION (one-time)                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Tavily  │ →  │  Chunk   │ →  │  Embed   │ →  │ Pinecone │      │
│  │  Crawl   │    │ (4000/   │    │ (OpenAI  │    │  Store   │      │
│  │          │    │  200)    │    │  3-small) │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  PHASE 2: RETRIEVAL (per-query, via agent)                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  User    │ →  │  Agent   │ →  │ retrieve │ →  │  Answer  │      │
│  │  Query   │    │ (GPT-4o) │    │ _context │    │ + Sources│      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  PHASE 3: UI (Streamlit)                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Chat Interface + Session State Memory + Source Citations     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Differs from Section 9

| Aspect | Section 9 (PDF RAG) | Section 10 (Doc Assistant) |
|--------|---------------------|---------------------------|
| **Data source** | Single PDF file | Entire documentation website |
| **Ingestion** | Load file locally | Crawl live website with Tavily |
| **Chunking** | CharacterTextSplitter (1000/0) | RecursiveCharacterTextSplitter (4000/200) |
| **Embedding model** | `text-embedding-ada-002` | `text-embedding-3-small` |
| **Retrieval** | Deterministic (always search) | Agentic (agent decides) |
| **Chain** | LCEL chain with `\|` operator | `create_agent()` with tool |
| **UI** | CLI print statements | Streamlit web app |
| **Memory** | Manual chat history list | Streamlit session state |
| **Scale** | ~67 chunks | 1000+ chunks (full docs site) |

---

## Deep Dive: Web Crawling with Tavily

### Why Not Just Download the Docs?

Documentation sites are **live, dynamic, and linked**. You can't just download a single file — you need to:
1. **Discover** all pages (follow links, respect depth limits)
2. **Extract** clean text content (strip HTML, nav bars, footers)
3. **Handle** dynamic content (JavaScript-rendered pages)

Tavily is an **AI-native** search/crawl API that handles all of this and returns clean, structured content ready for LLM consumption.

### `langchain-tavily` vs `tavily-python` — Which Package?

| Package | PyPI | What It Is |
|---------|------|------------|
| **`langchain-tavily`** | [langchain-tavily](https://pypi.org/project/langchain-tavily/) | LangChain integration — wraps Tavily's API into LangChain-compatible classes (`TavilyCrawl`, `TavilyMap`, `TavilyExtract`) that return results you can pipe directly into splitters and vector stores. **This is what we use.** |
| `tavily-python` | [tavily-python](https://pypi.org/project/tavily-python/) | Tavily's standalone Python SDK — lower-level, returns raw dicts. Use this if you're NOT using LangChain. |

```bash
# We install the LangChain wrapper (it pulls tavily-python as a dependency)
uv add langchain-tavily
```

```python
# Import from the LangChain integration package
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
```

The `langchain-tavily` classes give you `.invoke()` / `.ainvoke()` compatibility, meaning they slot directly into LCEL chains and agent tool pipelines.

### The Three Tavily Tools

| Tool | What It Does | When to Use |
|------|-------------|-------------|
| **`TavilyMap`** | Discovers all URLs on a site (like a sitemap) | First step — find what pages exist |
| **`TavilyCrawl`** | Crawls pages and extracts clean content | Main ingestion — get the actual text |
| **`TavilyExtract`** | Extracts content from a single URL | Targeted extraction of specific pages |

### How TavilyCrawl Works

```python
from langchain_tavily import TavilyCrawl

tavily_crawl = TavilyCrawl()

res = tavily_crawl.invoke({
	"url": "https://python.langchain.com/",
	"max_depth": 2,           # How many link-levels deep to follow
	"extract_depth": "advanced",  # Content extraction quality
})

# Results structure:
# res["results"] = [
#   {"url": "https://...", "raw_content": "The actual page text..."},
#   {"url": "https://...", "raw_content": "Another page..."},
#   ...
# ]
```

### Converting Crawl Results to Documents

```python
from langchain_core.documents import Document

all_docs = []
for item in res["results"]:
	all_docs.append(
		Document(
			page_content=item["raw_content"],
			metadata={"source": item["url"]},
		)
	)
```

Each crawled page becomes a `Document` with the URL as the `source` metadata — enabling source citations in the final answer.

### TavilyMap Configuration

```python
tavily_map = TavilyMap(
	max_depth=5,      # Follow links up to 5 levels deep
	max_breadth=20,   # Max 20 links per page
	max_pages=1000,   # Stop after discovering 1000 pages
)
```

### The Three Tavily Products in Detail

#### 1. TavilyMap — "Discover all URLs on a site"

TavilyMap automatically discovers and maps website structures by crawling through links. It does NOT extract content — it only finds URLs.

```python
from langchain_tavily import TavilyMap

# Initialize with crawl limits
tavily_map = TavilyMap(
	max_depth=3,        # Crawl up to 3 levels deep from the start URL
	max_breadth=15,     # Follow up to 15 links per page
	max_pages=50,       # Stop after discovering 50 total pages
)

# Discover all reachable URLs
site_map = tavily_map.invoke("https://python.langchain.com/docs/introduction/")

# Result: list of discovered URLs
urls = site_map.get("results", [])
# urls = [
#   "https://python.langchain.com/docs/concepts/",
#   "https://python.langchain.com/docs/how_to/tool_calling/",
#   "https://python.langchain.com/docs/integrations/",
#   ... (up to max_pages URLs)
# ]
```

**Use cases:**
- Pre-step before extraction (find pages, then extract selectively)
- Site auditing (discover all pages, check for broken links)
- Building a sitemap when none exists

#### 2. TavilyExtract — "Single/batch URL → clean content"

TavilyExtract takes a list of URLs and returns clean, structured text content. It removes HTML, navigation, ads, footers — only the main content remains.

```python
from langchain_tavily import TavilyExtract

tavily_extract = TavilyExtract()

# Extract content from specific URLs
result = await tavily_extract.ainvoke(input={
	"urls": [
		"https://python.langchain.com/docs/concepts/agents/",
		"https://python.langchain.com/docs/concepts/tools/",
	]
})

# Result structure:
extracted_docs = result.get("results", [])
# Each item: {"url": "https://...", "raw_content": "Clean extracted text..."}
```

**Batch processing pattern** (process many URLs efficiently):

```python
import asyncio

async def extract_batch(urls: list[str], batch_num: int) -> list:
	"""Extract documents from a batch of URLs."""
	docs = await tavily_extract.ainvoke(input={"urls": urls})
	return docs.get("results", [])

# Split URLs into batches of 3
url_batches = [urls[i:i+3] for i in range(0, len(urls), 3)]

# Process all batches concurrently
tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
batch_results = await asyncio.gather(*tasks)

# Flatten results
all_extracted = [doc for batch in batch_results for doc in batch]
```

**Use cases:**
- Selective extraction (only pages you care about, not entire site)
- When you already have a list of URLs from TavilyMap
- Precise control over which pages to include

#### 3. TavilyCrawl — "Crawl + extract in one step (with AI guidance)"

TavilyCrawl combines URL discovery AND content extraction in a single call. Its killer feature: **instruction-guided crawling** — you tell the AI WHAT you're looking for and it intelligently decides which links to follow.

```python
from langchain_tavily import TavilyCrawl

tavily_crawl = TavilyCrawl()

# WITHOUT instructions (baseline — crawls everything)
result_basic = tavily_crawl.invoke({
	"url": "https://python.langchain.com/",
	"max_depth": 2,
	"extract_depth": "advanced",
})
# Returns: ALL pages up to depth 2 (mixed content, lots of noise)

# WITH instructions (AI-guided — targeted content)
result_targeted = tavily_crawl.invoke({
	"url": "https://python.langchain.com/",
	"instructions": "Find all pages about AI agents",  # ← Natural language!
	"max_depth": 3,
	"extract_depth": "advanced",
})
# Returns: ONLY pages related to AI agents (highly relevant, minimal noise)
```

**The instructions parameter is the game-changer:**

| Approach | Pages Found | Content Quality | Post-Processing |
|----------|:-----------:|:---------------:|:---------------:|
| No instructions | Many (mixed) | Low relevance | Must filter manually |
| With instructions | Fewer (targeted) | High relevance | Ready to use |

**How instructions work internally:**
- Tavily's AI reads each discovered page
- Evaluates whether it matches your instructions
- Only follows links that lead toward relevant content
- Skips irrelevant branches entirely

**Use cases:**
- Main ingestion pipeline (crawl an entire docs site)
- When you want both discovery AND extraction in one API call
- When you need AI-guided filtering (e.g., "only find pages about RAG")

### Choosing the Right Tavily Tool

| Scenario | Tool | Why |
|----------|------|-----|
| "I want all URLs on a site" | `TavilyMap` | Discovery only, fast, no content |
| "I have specific URLs, give me their content" | `TavilyExtract` | Targeted extraction, precise control |
| "Crawl this site and give me everything" | `TavilyCrawl` (no instructions) | One-step discovery + extraction |
| "Find content about X on this site" | `TavilyCrawl` (with instructions) | AI-guided, most relevant results |
| "Map first, then extract selectively" | `TavilyMap` → `TavilyExtract` | Two-step: discover then choose |

### The Pipeline We Use

In this project, we use `TavilyCrawl` as the single ingestion step:

```
TavilyCrawl(url, max_depth=2, extract_depth="advanced")
	→ results[] (url + raw_content)
		→ Convert to Document objects
			→ Chunk → Embed → Store
```

For more control, you could use the two-step approach:
```
TavilyMap(url) → discover all URLs
	→ Filter/select relevant URLs
		→ TavilyExtract(selected_urls) → get content
			→ Convert to Documents → Chunk → Embed → Store
```

> → See the Tavily documentation for hands-on demos of `TavilyMap`, `TavilyExtract`, and `TavilyCrawl` (with and without instructions).

### Why Tavily Over BeautifulSoup/Scrapy?

| Feature | Manual Scraping | Tavily |
|---------|----------------|--------|
| JavaScript rendering | ❌ Need Selenium/Playwright | ✅ Handled automatically |
| Content extraction | Manual HTML parsing | ✅ Returns clean text |
| Rate limiting | Must implement yourself | ✅ Built-in |
| Pagination/depth | Custom crawl logic | ✅ Config parameters |
| AI-ready output | Post-processing needed | ✅ Ready for embeddings |

**C# Analogy:** Tavily is like using Azure Cognitive Search's web crawler instead of writing your own `HttpClient` + `HtmlAgilityPack` scraping pipeline.

---

## Deep Dive: RecursiveCharacterTextSplitter

### Why Not CharacterTextSplitter?

In Section 9, we used `CharacterTextSplitter(chunk_size=1000, separator="\n\n")`. This splits ONLY on double-newlines. If a paragraph is longer than 1000 characters, it produces an oversized chunk because it has no fallback strategy.

`RecursiveCharacterTextSplitter` tries **multiple separators in order**:

```python
# Default separator hierarchy (from most to least "structural"):
separators = ["\n\n", "\n", " ", ""]
#              ↑         ↑     ↑    ↑
#           paragraphs  lines  words  characters (last resort)
```

### How It Works

1. Try to split on `\n\n` (paragraph boundaries)
2. If a chunk is still too large, split on `\n` (line boundaries)
3. Still too large? Split on ` ` (word boundaries)
4. Absolute last resort: split on `""` (character-by-character)

This produces **semantically coherent chunks** because it preserves the largest structural unit that fits within the size limit.

### Configuration in This Project

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=4000,     # Larger chunks for documentation (more context per retrieval)
	chunk_overlap=200,   # 200 chars overlap to preserve boundary context
)
chunks = text_splitter.split_documents(all_docs)
```

### Why chunk_size=4000?

Documentation pages are longer and more structured than blog posts. Larger chunks (4000 vs 1000) mean:
- Each chunk contains more complete context (a full code example + explanation)
- Fewer chunks = fewer retrieval calls needed
- GPT-4o's large context window can handle bigger chunks easily

The trade-off: larger chunks are less precise (might include some irrelevant text alongside the answer).

### Comparison

| Splitter | Strategy | Best For |
|----------|----------|----------|
| `CharacterTextSplitter` | Single separator, no fallback | Simple text with consistent structure |
| **`RecursiveCharacterTextSplitter`** | Hierarchical separators | **Most production use cases** |
| `TokenTextSplitter` | Split by token count | When you need exact token budgets |
| `MarkdownTextSplitter` | Split on markdown headers | Documentation in markdown format |
| `CodeTextSplitter` | Language-aware splitting | Source code |

**Rule of thumb:** Use `RecursiveCharacterTextSplitter` as your default unless you have a specific reason not to.

---

## Deep Dive: Pinecone "Integrated" vs "Bring Your Own" Embeddings

### The Confusion

When you create a Pinecone index in their UI, they now ask you to **select an embedding model**. This is confusing because our code ALSO specifies an embedding model (`OpenAIEmbeddings`). Are they the same thing? **No — they are two completely different approaches.**

### The Two Approaches

#### Approach A: Pinecone Integrated (UI selects model)

```
Your code sends TEXT string → Pinecone → Pinecone calls embedding model → Pinecone stores vector
```

- You select a model in Pinecone's UI (e.g., "multilingual-e5-large")
- Your code sends **raw text** to Pinecone's API
- Pinecone does the embedding AND storing in one step
- You do NOT use `OpenAIEmbeddings` in your code
- Pinecone charges you extra for the embedding compute

#### Approach B: Bring Your Own Vectors ← **What we use**

```
Your code sends TEXT → OpenAI API → returns [0.023, -0.041, ..., 0.012] (1536 floats)
                                                         ↓
                                     Your code sends these floats → Pinecone stores them
```

- You do NOT select a model in Pinecone's UI — just set `dimensions=1536`
- Your code calls `OpenAIEmbeddings(model="text-embedding-3-small")` to get vectors
- Then sends those pre-computed vectors (arrays of 1536 numbers) to Pinecone
- Pinecone just stores numbers — it doesn't know or care what model made them
- You pay OpenAI directly for embedding (cheaper, more flexible)

### Side-by-Side Comparison

| | Approach A (Pinecone embeds) | Approach B (We embed) ← **Ours** |
|--|--|--|
| **What you send to Pinecone** | Raw text strings | Pre-computed float arrays |
| **Who calls the embedding model** | Pinecone (internally) | Our Python code |
| **Who pays for embedding** | Pinecone bill (more expensive) | OpenAI bill (cheaper) |
| **Model flexibility** | Only Pinecone's supported models | ANY model (OpenAI, Cohere, local) |
| **LangChain compatibility** | Doesn't work with `PineconeVectorStore` | Works perfectly |
| **When creating the index** | Select a model in UI | Select "Bring your own" / set dims manually |

### Why We Use Approach B

1. **LangChain's `PineconeVectorStore` sends pre-computed vectors** — it calls `embeddings.embed_documents()` first, then sends the resulting floats to Pinecone
2. **Flexibility** — we can swap from OpenAI to Cohere to a local model without touching Pinecone
3. **Cost** — OpenAI embedding is cheaper than Pinecone's integrated embedding
4. **Control** — we can cache embeddings, retry failed ones, batch them ourselves

### What to Select in Pinecone UI

When creating the index:
- ❌ Do NOT select an embedding model from their dropdown
- ✅ Choose **"Bring your own"** / **"Custom"** / **"Manual configuration"**
- ✅ Set **dimensions = 1536** (matches `text-embedding-3-small`)
- ✅ Set **metric = cosine**

Or skip the UI entirely and create programmatically:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="...")
pc.create_index(name="doc-helper-index", dimension=1536, metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
```

### C# Analogy

| Approach | C# Equivalent |
|----------|---------------|
| **A (Pinecone embeds)** | SQL Server Full-Text Search — the DB does indexing internally |
| **B (We embed)** | You compute a hash/vector in C# code, then store it in a `FLOAT[]` column — the DB just stores what you give it |

---

## Deep Dive: Async Batch Ingestion

### The Problem with Synchronous Ingestion

In Section 9, `PineconeVectorStore.from_documents(chunks)` processed everything sequentially. For 67 chunks, this was fine. For 1000+ chunks from a full documentation site, sequential processing is painfully slow.

### Python Async Fundamentals (for C# Developers)

If you come from C#, Python's `async/await` works the same conceptually — but with a few syntax differences:

| C# | Python | What It Does |
|----|--------|-------------|
| `async Task<T>` | `async def` | Declares a coroutine (async function) |
| `await task` | `await coroutine` | Suspends execution until result is ready |
| `Task.WhenAll(tasks)` | `asyncio.gather(*tasks)` | Runs multiple operations concurrently |
| `Task.Run(...)` | `asyncio.create_task(...)` | Schedules a coroutine to run |
| `static void Main()` → `async Task Main()` | `asyncio.run(main())` | Starts the event loop |

#### Key Concepts

```python
import asyncio

# 1. A coroutine — declared with "async def", MUST be awaited to execute
async def fetch_data():
	await asyncio.sleep(1)  # Non-blocking wait (like Task.Delay(1000))
	return "data"

# 2. WRONG — calling a coroutine without await does NOTHING
fetch_data()  # Returns a coroutine OBJECT, doesn't execute!

# 3. CORRECT — must await inside another async function
async def main():
	result = await fetch_data()  # Executes and waits for result

# 4. Running multiple coroutines concurrently (like Task.WhenAll)
async def main():
	# These run IN PARALLEL — total time ≈ 1 second, not 3
	results = await asyncio.gather(
		fetch_data(),
		fetch_data(),
		fetch_data(),
	)

# 5. Entry point — starts the event loop
asyncio.run(main())
```

#### Why Async for Ingestion?

Embedding and storing documents is **I/O-bound** — you're waiting for:
- OpenAI API to return embeddings (network I/O)
- Pinecone to store vectors (network I/O)

While waiting for one batch's API response, you could be sending the next batch. This is exactly what `asyncio` enables — concurrent I/O operations on a single thread.

```
SYNCHRONOUS (Section 9):
  Batch 1: [===send===][===wait===][===done===]
  Batch 2:                                      [===send===][===wait===][===done===]
  Batch 3:                                                                           [===send===]...
  Total: 3 × latency

ASYNC (Section 10):
  Batch 1: [===send===][===wait===][===done===]
  Batch 2: [===send===][===wait===][===done===]
  Batch 3: [===send===][===wait===][===done===]
  Total: 1 × latency (all running concurrently!)
```

**C# Analogy:** This is exactly the same reason you use `async/await` + `Task.WhenAll()` in C# for multiple HTTP calls — you don't want to wait for each one sequentially.

### The Solution: Concurrent Batch Processing

The ingestion pipeline splits documents into batches and processes them concurrently:

```python
import asyncio
from langchain_core.documents import Document

async def index_documents_async(documents: list[Document], batch_size: int = 50):
	"""Process documents in batches concurrently."""

	# Split into batches (list comprehension — like LINQ .Chunk(50))
	batches = [
		documents[i : i + batch_size]
		for i in range(0, len(documents), batch_size)
	]

	# Each batch is processed as an independent async task
	async def add_batch(batch: list[Document], batch_num: int):
		try:
			await vectorstore.aadd_documents(batch)  # Non-blocking!
			print(f"✅ Batch {batch_num}/{len(batches)} done")
		except Exception as e:
			print(f"❌ Batch {batch_num} failed: {e}")
			return False
		return True

	# Run ALL batches concurrently — like Task.WhenAll()
	tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
	results = await asyncio.gather(*tasks, return_exceptions=True)

	# Count successes
	successful = sum(1 for r in results if r is True)
	print(f"📊 {successful}/{len(batches)} batches processed successfully")
```

### The `asyncio.run()` Entry Point

```python
# In the main script — starts the async event loop
async def main():
	# ... crawl, chunk, then:
	await index_documents_async(chunks, batch_size=500)

if __name__ == "__main__":
	asyncio.run(main())  # Entry point for async code
```

`asyncio.run()` creates an event loop, runs the coroutine until complete, then shuts down. This is the Python equivalent of `static async Task Main(string[] args)` in C#.

### Why batch_size=50?

| Too Small (10) | Too Large (500) |
|---------------|-----------------|
| Many network round-trips | Single failure loses 500 docs |
| High overhead per batch | Memory pressure (500 embeddings in RAM) |
| Under-utilizes parallelism | API rate limits hit harder |

**50** is a balance: enough to amortize network overhead, small enough that a failure doesn't lose much work.

### `return_exceptions=True` — Don't Crash on Failures

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
```

Without `return_exceptions=True`, if ANY batch fails, the entire `gather()` raises an exception and all other tasks are cancelled. With it, failures are returned as exception objects in the results list — other batches continue processing.

**C# Equivalent:**
```csharp
// Without: await Task.WhenAll(tasks); — throws on first failure
// With: Task.WhenAll returns all tasks, you check each .Exception property
```

### Key Methods Summary

| Method | Sync/Async | What It Does |
|--------|-----------|--------------|
| `vectorstore.add_documents(docs)` | Sync | Adds documents one batch at a time |
| `vectorstore.aadd_documents(docs)` | **Async** | Non-blocking add — enables concurrency |
| `asyncio.gather(*tasks)` | Async | Runs multiple coroutines concurrently |
| `asyncio.run(main())` | Sync (entry) | Starts the event loop from sync code |

### The LangChain Async Naming Convention

LangChain follows a consistent pattern: every sync method has an async counterpart prefixed with `a`:

| Sync | Async | Used In |
|------|-------|---------|
| `.invoke()` | `.ainvoke()` | Chains, agents, tools |
| `.stream()` | `.astream()` | Streaming responses |
| `.batch()` | `.abatch()` | Parallel processing |
| `.add_documents()` | `.aadd_documents()` | Vector store ingestion |

This is the same `Async` suffix pattern as C# (`GetAsync()`, `SendAsync()`) — just with a prefix `a` instead.

---

## Deep Dive: Agentic RAG with Tools

### Why Agentic RAG Here?

In Section 9, we noted that deterministic RAG is better for production. So why use an agent here?

**Context matters.** A documentation assistant is different from a customer support bot:

| Use Case | Approach | Why |
|----------|----------|-----|
| Customer support bot | Deterministic | ALWAYS needs the knowledge base |
| **Documentation assistant** | **Agentic** | Sometimes needs docs, sometimes can answer directly |

A documentation assistant might get questions like:
- "What is LangChain?" → Agent can answer from parametric knowledge
- "Show me the API for ChatOpenAI" → Agent MUST search the docs
- "Compare LCEL to the old approach" → Agent searches, then synthesizes

The agent **decides** whether retrieval is needed — this is appropriate when the tool isn't always required.

### `create_agent()` — The Modern Pattern

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o", model_provider="openai")

agent = create_agent(
	model,
	tools=[retrieve_context],
	system_prompt="You are a helpful assistant that answers questions about LangChain..."
)

response = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

This replaces the older `initialize_agent()` / `AgentExecutor` pattern with a cleaner, more composable API.

### The `@tool` Decorator

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
	"""Retrieve relevant documentation to help answer user queries about LangChain."""
	retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

	# Serialized text → what the LLM sees
	serialized = "\n\n".join(
		f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
		for doc in retrieved_docs
	)

	# Return (content_for_llm, raw_artifact_for_app)
	return serialized, retrieved_docs
```

### What is `response_format="content_and_artifact"`?

This is a powerful pattern that separates **what the LLM reads** from **what the app uses**:

| Return Value | Who Uses It | Purpose |
|-------------|-------------|---------|
| `serialized` (first) | The LLM (as ToolMessage content) | Text the agent reasons over |
| `retrieved_docs` (second) | Your application code (as ToolMessage artifact) | Raw Document objects for source citations |

Without this, you'd have to either:
- Return raw Documents (LLM can't read Python objects)
- Return only text (app loses metadata for citations)

**C# Analogy:** Like returning `(string displayText, List<Document> rawData)` from a method — the UI shows `displayText`, but the backend uses `rawData` for further processing.

### Extracting Sources from the Agent Response

```python
from langchain.messages import ToolMessage

response = agent.invoke({"messages": messages})

# Get the answer (last AI message)
answer = response["messages"][-1].content

# Get source documents (from ToolMessage artifacts)
context_docs = []
for message in response["messages"]:
	if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
		if isinstance(message.artifact, list):
			context_docs.extend(message.artifact)
```

---

## Deep Dive: init_chat_model()

### The Provider-Agnostic Pattern

Instead of importing provider-specific classes:

```python
# OLD: Tightly coupled to OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# NEW: Provider-agnostic
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o", model_provider="openai")
```

### Why This Matters

Switching providers is now a **configuration change**, not a code change:

```python
# OpenAI
model = init_chat_model("gpt-4o", model_provider="openai")

# Anthropic
model = init_chat_model("claude-3-5-sonnet", model_provider="anthropic")

# Google
model = init_chat_model("gemini-pro", model_provider="google_genai")

# Local (Ollama)
model = init_chat_model("llama3", model_provider="ollama")
```

All return the same interface — `invoke()`, `stream()`, `ainvoke()`, `bind_tools()`, etc.

**C# Analogy:** This is like using `IServiceCollection.AddHttpClient<T>()` with different named clients — the consuming code doesn't know or care which implementation is behind the interface.

---

## Deep Dive: Streamlit Chat UI

### What is Streamlit?

Streamlit turns Python scripts into web applications. No HTML, CSS, or JavaScript needed. You write normal Python and Streamlit renders it as an interactive web page.

```python
import streamlit as st

st.title("My App")
st.write("Hello, world!")
# → Opens a browser with a title and text. That's it.
```

### The Chat Interface Pattern

```python
import streamlit as st

# Page configuration
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

# Initialize message history
if "messages" not in st.session_state:
	st.session_state.messages = [
		{"role": "assistant", "content": "Ask me anything about LangChain docs."}
	]

# Display all previous messages
for msg in st.session_state.messages:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"])

# Handle new user input
prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
	# Add user message
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)

	# Generate and display response
	with st.chat_message("assistant"):
		with st.spinner("Thinking…"):
			result = run_llm(prompt)
		st.markdown(result["answer"])

	# Save assistant message
	st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
```

### Key Streamlit Components for Chat

| Component | What It Does | Example |
|-----------|-------------|---------|
| `st.chat_message("user")` | Renders a chat bubble with user avatar | User messages |
| `st.chat_message("assistant")` | Renders a chat bubble with AI avatar | AI responses |
| `st.chat_input("...")` | Fixed input box at page bottom | Text entry |
| `st.spinner("...")` | Shows loading indicator | While waiting for LLM |
| `st.expander("Sources")` | Collapsible section | Source citations |
| `st.sidebar` | Left panel for controls | Clear chat button |

### How Streamlit Reruns Work

**Critical concept:** Streamlit reruns the **entire script** from top to bottom on every interaction (button click, input submit, etc.). This means:

1. Without `st.session_state`, all variables reset on each interaction
2. Chat history MUST be stored in `st.session_state` or it disappears
3. The display loop re-renders all previous messages on each rerun

This is fundamentally different from traditional web frameworks (React, Angular) where state persists in components.

**C# Analogy:** It's like a Razor Page where the entire page re-renders on every POST, but `TempData`/`Session` persists values across requests.

### Running Streamlit

```bash
streamlit run main.py
# Opens http://localhost:8501 in your browser
```

---

## Deep Dive: Memory via Session State

### How Memory Works in This Project

Unlike Section 9's conversational RAG (which used an explicit chat history list + reformulation), this project uses **Streamlit's session state as implicit memory**:

```python
# All messages are stored here — survives page reruns
st.session_state.messages = [
	{"role": "assistant", "content": "Welcome!", "sources": []},
	{"role": "user", "content": "What is LCEL?", "sources": []},
	{"role": "assistant", "content": "LCEL is...", "sources": ["https://..."]},
]
```

### Memory Scope

| Scope | Persists Across | Lost When |
|-------|----------------|-----------|
| `st.session_state` | Page reruns, widget interactions | Browser tab closed, server restart |
| Local variable | Nothing (reset each rerun) | Every interaction |
| Database-backed | Everything | Never (persistent) |

### The "Clear Chat" Pattern

```python
with st.sidebar:
	if st.button("Clear chat"):
		st.session_state.pop("messages", None)
		st.rerun()  # Force a full rerun with clean state
```

### Limitations of Session State Memory

| Limitation | Impact | Production Solution |
|-----------|--------|-------------------|
| Lost on page refresh | User loses history | Store in database (Redis/PostgreSQL) |
| No cross-tab sharing | Each tab is independent | Shared backend session store |
| No summarization | Long conversations grow unbounded | Add `ConversationSummaryMemory` |
| No persistence | Server restart = all memory gone | Database-backed session |

For a demo/learning project, session state is perfect. For production, you'd back this with a persistent store.

---

## Deterministic vs Agentic RAG — Why Agentic Here?

> 📖 **Full architecture guide:** [RAG Architecture Decisions](../reference-guides/RAG_Architecture_Decisions.md) — covers all three architectures (deterministic, conversational, agentic), cost breakdowns, memory strategies, and when to use which.

### Quick Recap

| | Deterministic (Section 9) | Agentic (Section 10) |
|--|--|--|
| **Who controls flow?** | You (hardcoded pipeline) | LLM (decides what tools to call) |
| **Retrieval** | Always runs | Agent decides if/when |
| **LLM calls** | 1 | 2+ |
| **Cost** | Lower, predictable | Higher, variable |

**Why agentic for THIS project?** A documentation assistant is exploratory — sometimes the user asks general questions the model knows, sometimes they need specific API details from docs. The agent decides when retrieval is necessary. A customer support bot (where you always need the knowledge base) would use deterministic.

### What Actually Happens Inside `agent.invoke()`

When `run_llm("What are deep agents?")` is called, `create_agent()` orchestrates this loop automatically:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ITERATION 1 — LLM decides to use the retrieval tool                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Sent to OpenAI API:                                                │
│    [SystemMessage]  "You are a helpful AI assistant..."             │
│    [HumanMessage]   "What are deep agents?"                         │
│                                                                     │
│  LLM responds with:                                                 │
│    [AIMessage]      tool_calls=[{                                   │
│                       name: "retrieve_context",                     │
│                       args: {query: "deep agents LangChain"}        │
│                     }]                                              │
│    → The LLM does NOT answer yet — it requests a tool call          │
│                                                                     │
│  Framework executes retrieve_context() automatically:               │
│    → Embeds query → searches vector store → gets top 4 chunks       │
│    → Returns (serialized, retrieved_docs) tuple                     │
│    → Creates ToolMessage:                                           │
│        .content  = serialized text    (LLM reads this)              │
│        .artifact = Document objects   (app extracts URLs from this) │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ITERATION 2 — AUGMENTATION: LLM reads context and answers          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Sent to OpenAI API (full conversation including tool result):      │
│    [SystemMessage]  "You are a helpful AI assistant..."             │
│    [HumanMessage]   "What are deep agents?"                         │
│    [AIMessage]      tool_calls=[{name: "retrieve_context", ...}]    │
│    [ToolMessage]    "Source: https://...\n\nContent: Deep agents     │
│                      are batteries-included..."                     │
│                      ↑ serialized text AUGMENTED into conversation   │
│                                                                     │
│  LLM responds with:                                                 │
│    [AIMessage]      "Deep agents in LangChain provide automatic     │
│                      context compression..."   ← FINAL ANSWER       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ AFTER agent.invoke() — Our code extracts results                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  response["messages"] contains all 4 messages:                      │
│    [0] HumanMessage   → user's question                             │
│    [1] AIMessage       → tool call request (not an answer)           │
│    [2] ToolMessage     → .content  = serialized (LLM already read)  │
│                          .artifact = [Document(...), Document(...)]  │
│    [3] AIMessage       → final answer text                           │
│                                                                     │
│  Our code extracts (in run_llm(), NOT in main.py):                  │
│    answer      = messages[-1].content         → final answer         │
│    context_docs = loop all ToolMessage.artifact → Documents for UI  │
│                                                                     │
│  Returned to Streamlit as:                                          │
│    {"answer": "Deep agents are...", "context": [Document, ...]}     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** `create_agent()` is **generic** — it's not RAG-specific. The tool you plug in defines the capability. Swap `retrieve_context` for `get_weather` and it becomes a weather agent. RAG is just one capability you attach.

### Multiple Tool Calls — Why the Loop

The LLM can call the tool **multiple times** in a single `agent.invoke()`:

```
User: "Compare LangChain agents with LangGraph agents"

[0] HumanMessage   "Compare LangChain agents with LangGraph agents"
[1] AIMessage       tool_calls=[{query: "LangChain agents"}]         ← 1st call
[2] ToolMessage     .artifact = [Doc, Doc, Doc, Doc]                 ← 4 docs
[3] AIMessage       tool_calls=[{query: "LangGraph agents"}]         ← 2nd call
[4] ToolMessage     .artifact = [Doc, Doc, Doc, Doc]                 ← 4 more docs
[5] AIMessage       "Here's the comparison..."                       ← FINAL ANSWER
```

That's why the code loops through **all** messages to collect artifacts (not just `messages[2]`) — we can't predict how many tool calls the LLM will make.

> 💡 You can reduce unnecessary tool calls via the system prompt (e.g., "use a single broad query"). See [RAG Architecture Decisions → Reducing Tool Calls](../reference-guides/RAG_Architecture_Decisions.md#reducing-tool-calls-via-system-prompt) for the full pattern and trade-offs.

### Limitations of the Reference Code (Section 10-Specific)

**1. No conversation memory:** The reference code passes only the current question to the agent — `st.session_state` stores messages for display only, not for the LLM. Follow-ups like "Tell me more about that" won't work.

**2. Duplicate source URLs:** The reference `_format_sources()` never deduplicated — multiple chunks from the same page showed the same URL repeatedly. Fixed with `dict.fromkeys()`.

**3. History would explode tokens if added naively:** Each tool call adds ~4 chunks × 4000 chars. After 5 questions, you'd be sending ~20,000 tokens of stale context. The production solutions (strip ToolMessages, sliding window, summarization) are covered in the [Memory & History Management](../reference-guides/RAG_Architecture_Decisions.md#memory--history-management) reference guide.

---

## Interview Q&A Anchors

**Q: What is Tavily and why use it instead of BeautifulSoup for documentation ingestion?**

> **A:** Tavily is an AI-native search and crawl API that returns structured, clean text content — not raw HTML. Unlike manual scraping with BeautifulSoup, Tavily handles JavaScript-rendered pages, automatic pagination, rate limiting, and content extraction out of the box. The output is immediately ready for embedding without post-processing.

**Q: Why use `RecursiveCharacterTextSplitter` instead of `CharacterTextSplitter`?**

> **A:** `RecursiveCharacterTextSplitter` tries multiple separators hierarchically — paragraphs, then lines, then words, then characters. This produces semantically coherent chunks because it always splits at the largest structural boundary that fits the size limit. `CharacterTextSplitter` only uses one separator and produces oversized chunks when the separator isn't found within the chunk size.

**Q: What does `response_format="content_and_artifact"` do on a `@tool`?**

> **A:** It allows a tool to return two values: (1) serialized text that becomes the `ToolMessage.content` the LLM reasons over, and (2) raw Python objects stored as `ToolMessage.artifact` that the application code can access. This separates what the LLM reads (formatted text) from what the app uses (Document objects with metadata for source citations).

**Q: How does `create_agent()` differ from the older `AgentExecutor` pattern?**

> **A:** `create_agent()` is LangChain's modern high-level factory — it takes a model, tools, and system prompt and returns a ready-to-use agent. It's simpler, more composable, and uses the latest tool-calling conventions (function calling) instead of the older ReAct prompt-based approach. The older `initialize_agent()` / `AgentExecutor` required more boilerplate and manual configuration.

**Q: Why use an agent for this project when deterministic RAG was recommended in Section 9?**

> **A:** Context determines the approach. A customer support bot ALWAYS needs to search the knowledge base (deterministic). A documentation assistant is exploratory — sometimes the user asks general questions the model knows, sometimes they need specific API details that require retrieval. The agent can decide when retrieval is necessary, making it appropriate for this use case.

**Q: How does Streamlit maintain chat history between interactions?**

> **A:** Streamlit reruns the entire script on every interaction. `st.session_state` is a per-user dictionary that persists across these reruns. Chat messages are stored there as a list of dicts with role, content, and sources. The display loop re-renders all messages on each rerun, giving the appearance of a persistent chat. This memory is lost when the browser tab closes — for persistence, you'd back it with a database.

**Q: What is `init_chat_model()` and why is it preferred?**

> **A:** It's a provider-agnostic model factory. Instead of importing `ChatOpenAI` or `ChatAnthropic` directly (tight coupling), you call `init_chat_model("gpt-4o", model_provider="openai")`. Switching providers becomes a configuration change (two strings) instead of a code change (imports + class names). All providers return the same interface.

**Q: Why use async batch ingestion instead of `from_documents()`?**

> **A:** `from_documents()` processes everything sequentially. For 1000+ documents from a crawled site, this takes too long. Async batch processing splits documents into groups (e.g., 50) and sends them to the vector store concurrently using `asyncio.gather()`. This dramatically reduces total ingestion time by utilizing network I/O parallelism.

**Q: What is the Streamlit rerun model and why does it matter?**

> **A:** Streamlit reruns the entire Python script top-to-bottom on every user interaction. This means local variables reset every time — only `st.session_state` persists. If you forget to store chat messages in session state, they vanish on the next interaction. It's fundamentally different from component-based frameworks like React where state persists in components.

**Q: How do you extract source documents from an agent's response?**

> **A:** When using `@tool(response_format="content_and_artifact")`, the raw documents are stored in the `ToolMessage.artifact` field. You iterate through `response["messages"]`, find `ToolMessage` instances, check for the `artifact` attribute, and extract the Document list. This gives you access to metadata (URLs, page numbers) for source citations.

**Q: What chunk_size would you use for documentation pages vs blog posts?**

> **A:** Documentation pages are longer and more structured — use 4000 chars with 200 overlap so each chunk contains a complete code example + its explanation. Blog posts are shorter and less structured — 1000 chars with minimal overlap works. The key trade-off: larger chunks = more context per retrieval but less precision.

**Q: How would you add persistence to this documentation assistant for production?**

> **A:** Replace Streamlit's `st.session_state` (ephemeral) with a database-backed session store. Store conversations in Redis (fast, TTL-based expiry) or PostgreSQL (permanent history, queryable). Send the last N messages as context to the agent on each query. For very long conversations, add `ConversationSummaryMemory` to compress older turns.

**Q: What is the difference between TavilyMap, TavilyExtract, and TavilyCrawl?**

> **A:** `TavilyMap` discovers URLs (no content extraction). `TavilyExtract` takes specific URLs and returns clean content. `TavilyCrawl` combines both — it discovers AND extracts in one call, with an optional `instructions` parameter that uses AI to guide which links to follow. For full-site ingestion, `TavilyCrawl` with instructions is the most efficient single-step approach.

**Q: What does `asyncio.gather(*tasks, return_exceptions=True)` do and why is `return_exceptions` important?**

> **A:** `asyncio.gather()` runs multiple coroutines concurrently (like `Task.WhenAll()` in C#). Without `return_exceptions=True`, if any task fails, the entire gather raises an exception and cancels remaining tasks. With it, failures are captured as exception objects in the results — other tasks continue processing. This is critical for batch ingestion where one failed batch shouldn't abort the entire pipeline.

**Q: What is the Python `async/await` equivalent of C#'s `Task.WhenAll()`?**

> **A:** `asyncio.gather(*tasks)` is the direct equivalent. You create a list of coroutines (tasks), pass them to `gather()`, and await the result. Unlike C#'s `Task.Run()`, Python coroutines don't start executing until they're awaited or scheduled — calling `async def func()` just returns a coroutine object. You must use `await`, `asyncio.gather()`, or `asyncio.create_task()` to actually execute them.

**Q: When creating a Pinecone index, should you select an embedding model in the UI?**

> **A:** Only if you want Pinecone to do the embedding for you (Integrated approach — you send raw text, Pinecone embeds + stores). If you're using LangChain's `PineconeVectorStore` with `OpenAIEmbeddings`, select "Bring your own vectors" and set dimensions=1536 manually. Your code calls OpenAI to get vectors, then sends those pre-computed floats to Pinecone for storage. This gives you full control over the embedding model and is cheaper.

**Q: Where does the "augmentation" in RAG actually happen inside `create_agent()`?**

> **A:** The framework does it automatically between agent loop iterations. After the tool executes and returns serialized text, the framework adds it as a `ToolMessage` to the conversation and sends the full message chain (including the tool result) back to the LLM. The LLM now has the retrieved context in its conversation history and uses it to generate the final answer. You never write this loop — `create_agent()` handles it internally.

**Q: `create_agent()` is RAG-specific, right?**

> **A:** No. `create_agent()` is a generic agent loop — it has no knowledge of retrieval. It becomes Agentic RAG only because you plug in a retrieval tool. Give it `get_weather` and it's a weather agent. Give it `run_sql_query` and it's a SQL agent. The tool defines the capability; the agent framework is the same in all cases.

**Q: Can the LLM make multiple tool calls in a single `agent.invoke()`?**

> **A:** Yes. For complex questions like "Compare LangChain agents with LangGraph", the LLM might call the retrieval tool twice — once for each topic — before synthesizing a final answer. That's why the source extraction code loops through **all** messages looking for ToolMessages with artifacts, not just a hardcoded index. The agent loop continues until the LLM produces a response without a tool call.

> 📖 **Architecture Q&As** (deterministic vs agentic cost, memory management, when to use which): See [RAG Architecture Decisions → Interview Q&A](../reference-guides/RAG_Architecture_Decisions.md#interview-qa-anchors)

---

## Runnable Source Files

All concepts in this doc are implemented as runnable scripts. See the implementation walkthrough and source code:

| File | What It Implements | Link |
|------|-------------------|------|
| Implementation guide | Full step-by-step walkthrough | [12_DocAssistant_Implementation.md](12_DocAssistant_Implementation.md) |
| Ingestion pipeline | TavilyCrawl → Chunk → Embed → Store | [src/ingestion.py](src/ingestion.py) |
| Agentic RAG backend | `create_agent()` + `@tool` + `content_and_artifact` | [src/backend/core.py](src/backend/core.py) |
| Streamlit chat UI | `st.session_state` + `st.chat_message` + sources | [src/main.py](src/main.py) |

---

## Production Reference: chat-langchain

A great production reference is [**chat-langchain**](https://github.com/langchain-ai/chat-langchain) — LangChain's own production documentation chatbot (the system behind `chat.langchain.com`). This is the real-world implementation our Section 10 assistant is modeled after.

### What It Demonstrates

| Feature | Implementation |
|---------|---------------|
| **Architecture** | Agentic RAG with LangGraph (state machine, not raw agent loop) |
| **Retrieval** | Pinecone vector store with curated documentation embeddings |
| **Memory** | LangGraph checkpointing (persistent, not just session state) |
| **Streaming** | Token-by-token via LangGraph's streaming events |
| **Multi-source** | Searches across LangChain, LangGraph, LangSmith docs simultaneously |
| **Deployment** | FastAPI backend + React frontend, containerized |
| **Feedback** | LangSmith feedback integration (thumbs up/down → evaluation dataset) |

### Why It Matters for Interviews

- Shows how a real team evolved from simple RAG to LangGraph-based agentic retrieval
- Demonstrates production concerns we studied: streaming, memory management, source citations
- Uses the same building blocks (Pinecone, OpenAI embeddings, tool-calling agents) — just with LangGraph for orchestration instead of raw `create_agent()`
- The repo structure (backend/frontend split, ingestion scripts, deployment configs) mirrors real production setups

**Key takeaway:** Our Section 10 assistant is a simplified but architecturally accurate version of `chat-langchain`. The main upgrade path from our code → production is: replace Streamlit with a proper frontend, replace `create_agent()` with LangGraph `StateGraph`, and add persistent memory + feedback loops.

> 🔗 **Repo:** [github.com/langchain-ai/chat-langchain](https://github.com/langchain-ai/chat-langchain)

---

## References

- [Tavily API Documentation](https://docs.tavily.com/)
- [Tavily Crawl API Reference](https://docs.tavily.com/documentation/api-reference/endpoint/crawl)
- [Tavily LangChain Integration — `langchain-tavily` (PyPI)](https://pypi.org/project/langchain-tavily/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Agents — create_agent](https://python.langchain.com/docs/how_to/tool_calling_agent/)
- [LangChain Tools — @tool decorator](https://python.langchain.com/docs/how_to/custom_tools/)
- [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
