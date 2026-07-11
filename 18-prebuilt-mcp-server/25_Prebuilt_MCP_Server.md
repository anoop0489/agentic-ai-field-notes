# 25. Using a Pre-Built MCP Server (mcpdoc) with AI Clients

> **Context:** The first practical MCP integration — connecting a pre-built MCP server (`mcpdoc`) to pre-built clients (Cursor, Claude Desktop). This is the "hello world" of MCP: no custom code, just configuration and understanding the flow.

---

## The Core Idea

> **Remember this, forget the rest.** Before building your own MCP server, learn to USE one. `mcpdoc` is a documentation-fetching server that gives your AI client (Cursor, Claude Desktop) real-time access to the latest LangChain/LangGraph docs. It uses `llms.txt` as a table of contents, then scrapes specific pages on demand. The result: your agent answers from LIVE docs, not stale training data.

**The technique in one sentence:**

> "Give your AI client an MCP server that fetches real-time documentation, so answers are grounded in the latest official docs — not hallucinated from outdated training data."

**What this section builds:**

```
┌──────────────────────────────────────────────────────────────┐
│ SECTION 18 GOAL                                              │
│                                                              │
│  Pre-built CLIENT          Pre-built SERVER                  │
│  ┌────────────────┐        ┌────────────────┐               │
│  │ Claude Desktop │──MCP──►│ mcpdoc         │               │
│  │ (or Cursor)    │        │ (LangChain's)  │               │
│  └────────────────┘        └───────┬────────┘               │
│                                    │                         │
│                                    ▼                         │
│                           ┌────────────────┐                 │
│                           │ llms.txt       │                 │
│                           │ (LangGraph     │                 │
│                           │  documentation)│                 │
│                           └────────────────┘                 │
│                                                              │
│  Result: Claude Desktop can answer "What is LangGraph        │
│  memory?" grounded in REAL-TIME official docs.               │
└──────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Are We Building? (Ch. 130)](#1-what-are-we-building-ch-130) | The goal — pre-built server + pre-built client |
| 2 | [MCP Inspector — Debugging Tool (Ch. 131)](#2-mcp-inspector--debugging-tool-ch-131) | How to test/debug MCP servers without a client |
| 3 | [llms.txt — The Website Index for AI (Ch. 132)](#3-llmstxt--the-website-index-for-ai-ch-132) | What llms.txt is, why it exists, when to use each variant |
| 4 | [mcpdoc — Real-Time Documentation Server (Ch. 133)](#4-mcpdoc--real-time-documentation-server-ch-133) | How mcpdoc works, setup, integration with Claude Desktop |
| 5 | [The Full Flow: Query → llms.txt → Scrape → Answer](#5-the-full-flow-query--llmstxt--scrape--answer) | Step-by-step execution trace |
| 6 | [mcpdoc vs RAG — Clarifying the Confusion](#6-mcpdoc-vs-rag--clarifying-the-confusion) | When mcpdoc is enough, when you need RAG, and image support |
| 7 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **mcpdoc** | MCP server for live documentation | A pre-built MCP server by LangChain that fetches real-time documentation from websites using their `llms.txt` file. Gives AI clients always-fresh docs without manual indexing. |
| **llms.txt** | Table of contents for AI agents | A Markdown file placed at a website's root that lists important pages with URLs and descriptions — designed for LLMs/agents to understand a site's structure and fetch relevant content. |
| **llms-full.txt** | The entire site content in one file | A variant of `llms.txt` that contains ALL page content inline (not just URLs). Huge file, suitable for RAG indexing or large-context LLMs with caching. |
| **MCP Inspector** | Debug/test tool for MCP servers | An open-source interactive dev tool by Anthropic for testing MCP servers — lets you inspect tools, resources, prompts, and execute them without needing a full client setup. |
| **Real-time grounding** | Answers from live data, not training data | The pattern where an agent fetches current documentation/data at query time, ensuring answers reflect the latest state — not what the LLM memorised during training (which goes stale). |
| **SSE (Server-Sent Events)** | Legacy HTTP streaming transport | An older MCP transport where the server pushes events to the client over a persistent HTTP connection. Now deprecated in favour of streamable HTTP. |
| **stdio** | Local subprocess transport | Transport where the client launches the server as a child process and communicates via stdin/stdout. Used by Claude Desktop and Cursor for local MCP servers. |
| **UVX** | UV's tool execution command | A `uv` command that runs Python tools/scripts in isolated environments without permanent installation — similar to `npx` for Node.js. |

---

## 1. What Are We Building? (Ch. 130)

### The Learning Journey

The MCP learning path is structured deliberately:

```
Step 1 (this section): Use pre-built server + pre-built client
  → Understand the protocol without writing code

Step 2 (next sections): Build your own MCP server
  → Understand server-side implementation

Step 3 (later): Build your own MCP client inside an agent
  → Understand client-side implementation
```

### This Section's Setup

| Component | What | Pre-built? |
|-----------|------|-----------|
| **Server** | `mcpdoc` (LangChain's documentation server) | ✅ Yes — clone from GitHub |
| **Client #1** | Cursor IDE (has built-in MCP client) | ✅ Yes — just configure |
| **Client #2** | Claude Desktop (has built-in MCP client) | ✅ Yes — just configure |
| **Data source** | LangChain/LangGraph `llms.txt` | ✅ Yes — hosted by LangChain |

### What `mcpdoc` Gives Your AI Client

Without mcpdoc:
- Claude Desktop answers from **training data** (months/years old)
- LangGraph APIs change frequently — training data goes stale fast
- Answers look correct but may reference deprecated APIs

With mcpdoc:
- Claude Desktop fetches **live documentation** at query time
- Always reflects the current state of LangGraph/LangChain docs
- Answers are grounded in real, scraped content

---

## 2. MCP Inspector — Debugging Tool (Ch. 131)

### What Is MCP Inspector?

An **interactive development tool** for testing and debugging MCP servers — without needing a full client like Claude Desktop or Cursor. Think of it as Postman for MCP.

| Feature | What It Does |
|---------|-------------|
| **Tools tab** | Lists all tools, their schemas, lets you execute them with custom inputs |
| **Resources tab** | Lists available resources, shows metadata, enables content inspection |
| **Prompts tab** | Displays prompt templates, shows arguments, allows testing with custom inputs |
| **Notifications pane** | Shows logs and notifications from the server |
| **Connection panel** | Connect to servers via stdio or HTTP (SSE/streamable-http) |

### Running MCP Inspector

```bash
# No installation needed — runs via npx
npx @modelcontextprotocol/inspector
```

This opens a web UI (typically on `http://localhost:3000`) where you can:
1. Connect to any running MCP server
2. Browse its capabilities
3. Test tools with custom inputs
4. See raw responses

### Why It Matters

When building or debugging MCP servers, you need to verify:
- Are tools being exposed correctly?
- Do tool schemas match what you expect?
- Does tool execution return proper results?
- Are resources/prompts loading?

MCP Inspector answers all of these **without writing client code**.

> ⚠️ **Note:** The demo connects to the server via "SSE". As of the MCP spec (2025-03-26), SSE is deprecated in favour of **streamable HTTP**. MCP Inspector supports both, but new servers should use `streamable-http`. The Inspector UI may still label the connection as "SSE" for backward compatibility.

---

## 3. llms.txt — The Website Index for AI (Ch. 132)

### What Is llms.txt?

A **Markdown file** placed at a website's root URL (e.g., `https://docs.langchain.com/llms.txt`) that provides:
- A list of important pages with their URLs
- Brief descriptions of what each page covers
- A machine-readable "table of contents" for AI agents

**Analogy:** If a website is a book, `llms.txt` is the table of contents page — it tells you what chapters exist and where to find them.

### The Two Variants

| Variant | Content | Size | URL Example |
|---------|---------|------|-------------|
| **llms.txt** | URLs + short descriptions (index only) | Small (few KB) | `https://docs.langchain.com/llms.txt` |
| **llms-full.txt** | Full page content inline | Large (can be MB+) | `https://docs.langchain.com/llms-full.txt` |

### When to Use Which

| Use Case | Which Variant | Pattern |
|----------|--------------|---------|
| Agent with scraping tool (Firecrawl, etc.) | `llms.txt` | Agent reads index → picks relevant URL → scrapes that page |
| RAG pipeline (index into vector store) | `llms-full.txt` | Download → chunk → embed → store in Pinecone/Chroma |
| Large-context LLM (100K+ tokens) | `llms-full.txt` | Send entire content in one prompt |
| Context caching (Gemini, Claude) | `llms-full.txt` | Cache the full content, query against it |

### The Agent Pattern with llms.txt

This is exactly what `mcpdoc` implements:

```
┌──────────────────────────────────────────────────────────────┐
│ AGENT + llms.txt PATTERN                                     │
│                                                              │
│ 1. Agent fetches llms.txt (the index)                        │
│    → Gets: list of URLs + descriptions                       │
│                                                              │
│ 2. LLM reads index, picks relevant URL for the user's query │
│    → "User asked about memory... this URL covers memory"     │
│                                                              │
│ 3. Agent scrapes that specific URL                           │
│    → Gets: full page content about memory                    │
│                                                              │
│ 4. LLM generates answer grounded in scraped content          │
│    → Real-time, accurate, from official docs                 │
│                                                              │
│ TRADEOFF: Higher latency (2-3 tool calls) but REAL-TIME data │
└──────────────────────────────────────────────────────────────┘
```

### Trade-offs

| Approach | Latency | Freshness | Cost |
|----------|---------|-----------|------|
| **llms.txt + scraping** (mcpdoc) | Higher (multiple tool calls + scraping) | Real-time — always current | Low (no indexing, pay per scrape) |
| **llms-full.txt + RAG** | Lower (vector search is fast) | Stale — only as fresh as last index | Higher (embedding + storage costs) |
| **llms-full.txt + context cache** | Lowest (cached in LLM provider) | Semi-fresh — as fresh as last cache refresh | Highest (cache storage at provider) |
| **Training data only** (no MCP) | Fastest | Most stale — frozen at training cutoff | Free (already trained) |

### Why Website Owners Want llms.txt

- Improves how AI extracts info from their site
- Enhances discoverability by AI-powered search
- Similar incentive as `robots.txt` — help machines understand your site
- Can improve SEO for AI-driven search engines

> **Note:** `llms.txt` is not an official web standard (no RFC), but is gaining rapid adoption in the GenAI community. LangChain, Anthropic, and many documentation sites already implement it.

---

## 4. mcpdoc — Real-Time Documentation Server (Ch. 133)

### What mcpdoc Does

[`mcpdoc`](https://github.com/langchain-ai/mcpdoc) is a pre-built MCP server by LangChain that:

1. **Stores** a reference to one or more `llms.txt` URLs
2. **Exposes two tools:**
   - `list_doc_sources` — returns the configured llms.txt URLs
   - `fetch_docs` — scrapes a given URL and returns its content
3. **Enables any MCP client** to fetch live documentation on demand

### The Two Tools

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `list_doc_sources` | None | URL(s) to the configured llms.txt file(s) | Agent discovers where the index lives |
| `fetch_docs` | A URL string | Scraped content of that URL | Agent fetches specific documentation pages |

### How the Agent Uses mcpdoc (3-Step Flow)

```
User: "What is LangGraph memory?"

Step 1 — Agent calls list_doc_sources()
  → Returns: "https://langchain-ai.github.io/langgraph/llms.txt"

Step 2 — Agent calls fetch_docs("https://langchain-ai.github.io/langgraph/llms.txt")
  → Returns: Index of all LangGraph doc pages with URLs
  → Agent reads index, identifies: "/concepts/memory" is relevant

Step 3 — Agent calls fetch_docs("https://langchain-ai.github.io/langgraph/concepts/memory")
  → Returns: Full content of the memory documentation page

Step 4 — LLM generates answer grounded in the scraped content
  → Answer reflects CURRENT LangGraph memory documentation
```

### Setup: Running mcpdoc Locally

```bash
# Clone the repo
git clone https://github.com/langchain-ai/mcpdoc.git
cd mcpdoc

# Create and activate virtual environment
uv venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv sync

# Run the server (SSE mode on port 8082 — for testing with Inspector)
uv run mcpdoc --urls "LangGraph=https://langchain-ai.github.io/langgraph/llms.txt" \
  --transport sse --port 8082
```

### Testing with MCP Inspector

```bash
# In a separate terminal
npx @modelcontextprotocol/inspector

# In the Inspector UI:
# 1. Connect to http://localhost:8082 (SSE)
# 2. Go to Tools tab → List Tools
# 3. Test list_doc_sources → see the configured URL
# 4. Test fetch_docs with the llms.txt URL → see the index content
```

### Integrating with Claude Desktop

Claude Desktop stores MCP server configs in a JSON file:

**Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**

```json
{
  "mcpServers": {
	"llms-txt": {
	  "command": "/full/path/to/uvx",
	  "args": [
		"--directory", "/full/path/to/mcpdoc",
		"mcpdoc",
		"--urls", "LangGraph=https://langchain-ai.github.io/langgraph/llms.txt",
		"--transport", "stdio",
		"--port", "8081"
	  ]
	}
  }
}
```

> ⚠️ **Critical debugging lesson:** You **must** use the full absolute path to `uvx` (and to the mcpdoc directory) in the config. Claude Desktop doesn't inherit your shell's PATH, so relative paths or just `uvx` will fail with `ENOENT`. Use `which uvx` (macOS/Linux) or `where uvx` (Windows) to find the full path.

### Common Pitfalls

| Problem | Symptom | Fix |
|---------|---------|-----|
| Relative path to `uvx` | `ENOENT` error in Claude Desktop logs | Use full absolute path from `which uvx` |
| Relative path to mcpdoc directory | Server starts but can't find source files | Use full absolute path in `--directory` |
| Wrong transport | Server runs but client can't connect | Claude Desktop uses `stdio`, Inspector uses `sse`/`http` |
| Forgot to restart Claude Desktop | Old config still active | Must fully restart after config changes |

### Result: Before vs After

| Question: "What is LangGraph memory?" | Without mcpdoc | With mcpdoc |
|----------------------------------------|----------------|-------------|
| **Source** | LLM training data (months old) | Live scrape of official docs |
| **Accuracy** | May reference deprecated APIs | Reflects current documentation |
| **Freshness** | Frozen at training cutoff | Real-time |
| **Tool calls** | 0 (direct generation) | 2-3 (list → fetch index → fetch page) |
| **Latency** | Fast (~2s) | Slower (~5-10s with scraping) |
| **Confidence** | Looks correct, may be wrong | Grounded in source material |

---

## 5. The Full Flow: Query → llms.txt → Scrape → Answer

### Complete Sequence Diagram

```
┌──────┐  ┌───────────────┐  ┌─────┐  ┌──────────────┐  ┌───────────────┐
│ USER │  │ CLAUDE DESKTOP│  │ LLM │  │ MCP CLIENT   │  │ mcpdoc SERVER │
└──┬───┘  └───────┬───────┘  └──┬──┘  └──────┬───────┘  └───────┬───────┘
   │              │              │             │                  │
   │ "What is     │              │             │                  │
   │  LangGraph   │              │             │                  │
   │  memory?"    │              │             │                  │
   ├─────────────►│              │             │                  │
   │              │              │             │                  │
   │              │  query +     │             │                  │
   │              │  tools       │             │                  │
   │              ├─────────────►│             │                  │
   │              │              │             │                  │
   │              │  "call       │             │                  │
   │              │  list_doc_   │             │                  │
   │              │  sources"    │             │                  │
   │              │◄─────────────┤             │                  │
   │              │              │             │                  │
   │              ├──────────────┼────────────►│                  │
   │              │              │             ├─────────────────►│
   │              │              │             │  list_doc_sources│
   │              │              │             │◄─────────────────┤
   │              │              │             │  URL: llms.txt   │
   │              │◄─────────────┼─────────────┤                  │
   │              │              │             │                  │
   │              ├─────────────►│             │                  │
   │              │  "call       │             │                  │
   │              │  fetch_docs  │             │                  │
   │              │  (llms.txt)" │             │                  │
   │              │◄─────────────┤             │                  │
   │              │              │             │                  │
   │              ├──────────────┼────────────►│                  │
   │              │              │             ├─────────────────►│
   │              │              │             │  fetch(llms.txt) │
   │              │              │             │◄─────────────────┤
   │              │              │             │  [index content] │
   │              │◄─────────────┼─────────────┤                  │
   │              │              │             │                  │
   │              ├─────────────►│             │                  │
   │              │ "fetch_docs  │             │                  │
   │              │ (memory URL)"│             │                  │
   │              │◄─────────────┤             │                  │
   │              │              │             │                  │
   │              ├──────────────┼────────────►│                  │
   │              │              │             ├─────────────────►│
   │              │              │             │ fetch(/memory)   │
   │              │              │             │◄─────────────────┤
   │              │              │             │ [memory content] │
   │              │◄─────────────┼─────────────┤                  │
   │              │              │             │                  │
   │              ├─────────────►│             │                  │
   │              │ query +      │             │                  │
   │              │ memory docs  │             │                  │
   │              │◄─────────────┤             │                  │
   │              │ final answer │             │                  │
   │◄─────────────┤              │             │                  │
   │  grounded    │              │             │                  │
   │  answer      │              │             │                  │
```

### Key Observations from the Flow

1. **3 tool calls minimum** — list sources, fetch index, fetch specific page
2. **LLM decides which page** — reads the index and picks the relevant URL (this is the "intelligence")
3. **All scraping happens in the server** — Claude Desktop doesn't need HTTP capabilities, only MCP protocol
4. **Protocol handles everything** — client doesn't need to know HOW the server scrapes, just calls the tool

---

## 6. mcpdoc vs RAG — Clarifying the Confusion

A common source of confusion: mcpdoc fetches relevant documentation and gives it to the LLM for a grounded answer — isn't that just RAG?

### The Honest Answer

**Yes, the goal is the same:** retrieve relevant information → give it to the LLM → get a grounded answer. That's retrieval-augmented generation in spirit. The difference is the **retrieval mechanism** and **when processing happens:**

| | Traditional RAG | mcpdoc (index + fetch) |
|---|-----------------|------------------------|
| **How it finds relevant content** | Embedding math — cosine similarity over vectors | LLM reads a table of contents and picks the right URL |
| **Pre-processing required** | Yes — chunk all docs, embed, store in vector DB (upfront, offline) | None — fetches live at query time |
| **What gets retrieved** | Small chunks (200-500 tokens) from across many documents | One entire page at a time |
| **Infrastructure** | Vector DB + embedding model + ingestion pipeline | Just an HTTP fetch. That's it. |
| **Freshness** | Stale — only as fresh as last re-index | Always live — scrapes on demand |

### Why mcpdoc Works Without RAG Infrastructure

The key insight: **mcpdoc only works because the index (`llms.txt`) is small enough for the LLM to read.**

LangGraph docs have ~50 pages listed in their `llms.txt`. That index is maybe 2,000-3,000 tokens — trivial for any modern LLM to read and reason over. The LLM acts as the "retriever" by reading the index and picking the right URL.

### When This Approach Breaks Down (RAG Becomes Necessary)

The mcpdoc/index pattern stops working when:

| Scenario | Why It Breaks | Solution |
|----------|---------------|----------|
| **Index too large** (1,000+ pages listed) | `llms.txt` itself won't fit in context window | Need vector search to find relevant items |
| **Multiple sources combined** (all Confluence + all Slack + all emails) | Combined index = tens of thousands of entries | Need embeddings across all sources |
| **Sub-page precision needed** | Answer is in one paragraph of a 50-page doc | Need chunking to isolate the relevant paragraph |
| **Semantic matching required** | User says "vacation policy" but doc says "PTO guidelines" | Need embedding similarity (understands synonyms) |

### The Practical Rule of Thumb

```
Single documentation site (30-200 pages):
  → mcpdoc/llms.txt is simpler, fresher, and sufficient
  → No infrastructure needed

Multiple heterogeneous sources (1,000+ documents):
  → RAG is necessary — index alone won't fit in context
  → Vector search scales regardless of corpus size

The boundary: Can the LLM read your entire index in one shot?
  Yes → mcpdoc pattern works
  No  → You need RAG's mathematical retrieval
```

### What About Images?

**`mcpdoc` is text-only.** Its `fetch_docs` tool scrapes web pages and returns the text/Markdown content. It does not download, process, or return images embedded in documentation pages.

If your documentation relies heavily on diagrams, screenshots, or visual content:
- `mcpdoc` will miss that information entirely
- The LLM will only see the text portions of the page
- For image-heavy docs, you'd need a multimodal RAG pipeline (extract images → describe with vision model → index descriptions)

The MCP protocol itself supports multimodal tool responses (images, audio — see Section 17), but `mcpdoc` specifically does not use this capability. It's a text-scraping tool.

---

## Interview Q&A Anchors

**Q: What is `llms.txt` and how does it relate to MCP?**
> **A:** `llms.txt` is a Markdown file at a website's root that lists important pages with URLs and descriptions — a machine-readable table of contents for AI agents. MCP servers like `mcpdoc` use it as an index: the agent first fetches `llms.txt` to discover what pages exist, then scrapes specific pages relevant to the user's question. It enables real-time documentation fetching without pre-indexing.

**Q: What's the difference between `llms.txt` and `llms-full.txt`?**
> **A:** `llms.txt` contains only URLs and short descriptions (like a book's table of contents) — lightweight, requires a follow-up scrape to get actual content. `llms-full.txt` contains all page content inline (like the entire book) — large file, suitable for RAG indexing, context caching, or large-context LLMs. Use `llms.txt` with an agent that can scrape; use `llms-full.txt` for batch indexing into a vector store.

**Q: What is MCP Inspector and when would you use it?**
> **A:** MCP Inspector is an interactive dev tool (run via `npx`) for testing and debugging MCP servers without needing a full client like Claude Desktop. It lets you connect to a server, browse its tools/resources/prompts, and execute them with custom inputs. Use it during development to verify your server exposes the correct tools with the right schemas before integrating with production clients.

**Q: Why would Claude Desktop without mcpdoc give potentially wrong answers about LangGraph?**
> **A:** Claude's training data has a knowledge cutoff date. LangGraph and LangChain update APIs frequently — what was correct 3 months ago may be deprecated today. Without mcpdoc, Claude generates from stale training data that looks correct but may reference old APIs. With mcpdoc, the answer is grounded in a live scrape of the official documentation, guaranteeing it reflects the current state.

**Q: What's the common mistake when configuring MCP servers in Claude Desktop?**
> **A:** Using relative paths. Claude Desktop doesn't inherit your shell's PATH or working directory, so `uvx` or `./mcpdoc` will fail with `ENOENT`. You must use the full absolute path to both the executable (`/usr/local/bin/uvx` or `C:\Users\...\.local\bin\uvx.exe`) and the server's directory. Find it with `which uvx` (macOS/Linux) or `where uvx` (Windows).

**Q: How does the agent decide which page to scrape from `llms.txt`?**
> **A:** The LLM reads the scraped `llms.txt` content (which contains URLs + descriptions) and uses its reasoning to match the user's question to the most relevant URL. For "What is LangGraph memory?", it finds the entry for `/concepts/memory` in the index. This is standard LLM tool-use reasoning — the agent autonomously decides the next action based on context.

**Q: What are the trade-offs of real-time doc fetching (mcpdoc) vs RAG-indexed docs?**
> **A:** Real-time fetching (mcpdoc) gives guaranteed freshness but higher latency (2-3 tool calls + scraping = 5-10s). RAG-indexed docs give lower latency (fast vector search) but go stale unless you re-index periodically. For rapidly changing docs (LangGraph), real-time fetching is better. For stable internal docs, RAG is more efficient. You can combine both — RAG for common queries, MCP fallback for freshness.

**Q: When does the mcpdoc/llms.txt pattern stop working and you need traditional RAG?**
> **A:** When the index itself is too large for the LLM to read in one context window. A documentation site with 30-200 pages works fine — the `llms.txt` is a few thousand tokens. But if you're combining thousands of heterogeneous documents (Confluence + Slack + emails), the combined index won't fit in context. At that point, you need vector search to mathematically narrow down relevant documents, which is exactly what RAG does.

**Q: Does mcpdoc support images or only text?**
> **A:** Text only. `mcpdoc`'s `fetch_docs` tool scrapes web pages and returns their text/Markdown content. It does not download, process, or return images embedded in those pages. If documentation relies on diagrams or screenshots, that visual information is invisible to the agent. For image-heavy content, you'd need a multimodal RAG pipeline that extracts images and describes them with a vision model. Note: the MCP protocol itself supports multimodal responses (images, audio), but `mcpdoc` specifically does not use this capability.

---

## References

- [mcpdoc GitHub Repository](https://github.com/langchain-ai/mcpdoc)
- [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) | [GitHub](https://github.com/modelcontextprotocol/inspector)
- [LangGraph llms.txt](https://langchain-ai.github.io/langgraph/llms.txt)
- [LangChain llms.txt](https://docs.langchain.com/llms.txt)
- [MCP Official Documentation](https://modelcontextprotocol.io/introduction)
