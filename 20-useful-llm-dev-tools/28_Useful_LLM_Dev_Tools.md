# 28. Useful Tools When Developing LLM Applications

> **Context:** A collection of essential developer tools in the LangChain ecosystem

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [LangChain's Official MCP Server (Ch. 143)](#1-langchains-official-mcp-server-ch-143) | Stop AI coding agents from generating deprecated code |
| 2 | [LangChain Hub (Ch. 144)](#2-langchain-hub-ch-144) | Community prompt repository, `hub.pull()` |
| 3 | [Text Splitting Playground (Ch. 145)](#3-text-splitting-playground-ch-145) | Visualise chunking strategies, official splitter reference |
| 4 | [LangChain vs LlamaIndex (Ch. 146)](#4-langchain-vs-llamaindex-ch-146) | When to use which framework |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **DocsByLangChain MCP** | Live docs for coding agents | A public, remote MCP server by LangChain that exposes a `SearchDocsByLangChain` tool — gives AI coding agents access to the latest documentation so they don't generate deprecated code. |
| **LangChain Hub** | Community prompt marketplace | A repository (part of LangSmith) for sharing, versioning, and downloading prompts. Access via `hub.pull("owner/prompt-name")`. |
| **Text Splitting Playground** | Visual chunk debugger | A Streamlit app that lets you paste text, tweak chunk size/overlap/splitter type, and see the resulting chunks visually — for optimising RAG ingestion. |
| **RecursiveCharacterTextSplitter** | Default splitter — start here | Splits text hierarchically (paragraphs → sentences → words), keeping semantic units intact. Best default for most RAG use cases. |
| **chat.langchain.com** | LangChain's own doc chatbot | An official chatbot that uses the same `SearchDocsByLangChain` MCP server to answer LangChain questions from live docs. |

---

## 1. LangChain's Official MCP Server (Ch. 143)

### The Problem

AI coding agents (Cursor, Claude Code, VS Code Copilot) generate **deprecated code** because their training data is frozen at a point in time. LangChain changes rapidly — APIs break, functions get renamed (e.g., `create_react_agent` → `create_agent`). Without live docs access, coding agents produce outdated, non-working code.

### The Solution: DocsByLangChain MCP Server

LangChain provides a **free, public, remote MCP server** — no API key required. It exposes one tool:

| Tool | Description | Input |
|------|-------------|-------|
| `SearchDocsByLangChain` | Search across LangChain docs for relevant information, code examples, API references, and guides | `query: str` |

### How to Use It

**In Cursor / VS Code (MCP config):**

```json
{
  "mcpServers": {
	"langchain-docs": {
	  "url": "https://docs.langchain.com/mcp",
	  "transport": "streamable_http"
	}
  }
}
```

That's it — one line in your MCP config. The coding agent will now query live LangChain docs before generating code.

### Demo: With vs Without the MCP Server

| | Without MCP Server | With MCP Server |
|---|---|---|
| **Prompt** | "Write me a LangChain agent" | Same prompt |
| **Result** | `create_react_agent` (deprecated) or `initialize_agent` (deprecated for years) | `create_agent` (current, correct) |
| **Why** | LLM relies on stale training data | Tool invokes `SearchDocsByLangChain` → gets latest docs |

### chat.langchain.com

The same MCP server powers LangChain's official documentation chatbot at [chat.langchain.com](https://chat.langchain.com). You can:
- Ask questions and get answers sourced from live docs
- Click "View trace" to see the full LangSmith trace (transparent tool calls)
- See the agent invoke `SearchDocsByLangChain` multiple times to refine answers

### Key Takeaway

> If you're using any AI coding agent to write LangChain/LangGraph code, **always have the LangChain docs MCP server enabled**. It's the difference between getting working code and wasting time debugging deprecated APIs.

---

## 2. LangChain Hub (Ch. 144)

### What It Is

A community repository (hosted within [LangSmith](https://smith.langchain.com/hub)) for sharing and versioning prompts. The idea: prompt engineering is hard, and reusing battle-tested prompts saves time.

### Features

| Feature | Details |
|---------|---------|
| **Browse by use case** | Agents, RAG, classification, code writing, SQL, extraction |
| **Filter by model** | OpenAI, Google, Meta — prompts optimised per vendor |
| **Version history** | Full commit log showing how prompts evolved |
| **Playground** | Test prompts with different parameters (temperature, model) directly in the browser |
| **Popularity metrics** | Downloads, likes, watchers |

### How to Use in Code

```python
from langchain import hub

# Download a community prompt by its identifier
prompt = hub.pull("rlm/rag-prompt")

# Use it in a chain
chain = prompt | llm | output_parser
```

The `hub.pull("owner/prompt-name")` pattern downloads the latest version of a shared prompt. You can pin to a specific commit hash for reproducibility.

### When to Use

- Starting a new RAG project → pull `rlm/rag-prompt` as a baseline
- Need a ReAct agent prompt → browse the "agents" category
- Want to A/B test prompts → compare multiple hub prompts in the playground
- Sharing prompts across a team → publish to your org's hub space

---

## 3. Text Splitting Playground (Ch. 145)

### The Problem

Before RAG ingestion, documents must be split into chunks. The parameters (chunk size, overlap, splitter type) dramatically affect retrieval quality, but it's hard to visualise the impact without trial and error.

### The Tool

**URL:** [langchain-text-splitter.streamlit.app](https://langchain-text-splitter.streamlit.app/)

A free Streamlit app where you:
1. Paste your text
2. Choose a splitter type and parameters (chunk size, chunk overlap, length function)
3. Click "Split Text"
4. See each chunk highlighted visually — verify they make semantic sense

### Official Text Splitter Reference

Install:
```bash
uv add langchain-text-splitters
```

### Splitting Strategies

| Strategy | Splitter | How It Works | Best For |
|----------|----------|-------------|----------|
| **Text structure** | `RecursiveCharacterTextSplitter` | Splits hierarchically: paragraphs → sentences → words. Keeps larger units intact, falls back to smaller if chunk exceeds size. | **Default for most use cases** |
| **Token-based** | `CharacterTextSplitter.from_tiktoken_encoder` | Splits by token count (not characters). Ensures chunks fit model context exactly. | When token budget is critical |
| **Character-based** | `CharacterTextSplitter` | Splits by character count. Simpler, more consistent across text types. | Simple documents, fixed-size needs |
| **Markdown** | `MarkdownHeaderTextSplitter` | Splits by headers (`#`, `##`, `###`). Preserves document structure. | Documentation, READMEs |
| **HTML** | `HTMLHeaderTextSplitter` | Splits by HTML tags. Preserves semantic structure. | Web pages |
| **JSON** | `RecursiveJsonSplitter` | Splits by object/array elements. | API responses, structured data |
| **Code** | `RecursiveCharacterTextSplitter.from_language` | Splits by functions, classes, logical blocks. Language-aware. | Source code |

### Code Examples

**Recursive (default — start here):**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)
```

**Token-based (tiktoken):**
```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
	encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

### How to Choose Parameters

There's no universal answer — every dataset is different. Use the playground to:
- Verify chunks contain coherent information (not cut mid-sentence)
- Check overlap captures enough context between adjacent chunks
- Compare splitter types on your actual data

> 💡 **Start with** `RecursiveCharacterTextSplitter`, chunk_size=1000, chunk_overlap=200. Adjust based on visual inspection in the playground.

---

## 4. LangChain vs LlamaIndex (Ch. 146)

### Quick Answer

> Both can build LLM applications. **Use LangChain** — it's more popular, more comprehensive for agents, and handles RAG equally well now.

### Comparison

| Aspect | LangChain | LlamaIndex |
|--------|-----------|------------|
| **Focus** | General-purpose LLM orchestration (agents + RAG + chains) | Data-centric (RAG, search, data connectors) |
| **Agents** | Robust — LangGraph, `create_agent`, tool calling, custom graphs | Basic — has ReAct, but less flexible |
| **RAG** | Full support — LCEL chains, retrievers, multiple vector stores | Strong — originally built for this |
| **Adoption** | More popular, larger community | Smaller community |
| **Flexibility** | High — LCEL, custom chains, mix-and-match components | More opinionated, fewer escape hatches |
| **Active development** | Very active — frequent updates, MCP support, LangGraph | Active, but slower pace |

### When to Pick Which

| Scenario | Choice |
|----------|--------|
| Building an agentic application (tool calling, reasoning loops) | **LangChain** — much more robust agent ecosystem |
| RAG-heavy app (chat with data, document QA) | **LangChain** — equivalent RAG support now, plus agents if you need them later |
| Need maximum flexibility in chain composition | **LangChain** — LCEL gives fine-grained control |
| Team already uses LlamaIndex | **LlamaIndex** — switching cost not worth it for pure RAG |

### A Practical Take

> Even if an application is very focused on data and has a lot of retrieval augmentation, LangChain answers those needs well. For an agentic application, LangChain is the stronger choice because it has a much more robust ecosystem.

---

## Interview Q&A Anchors

**Q: How do you prevent AI coding agents from generating deprecated LangChain code?**
> **A:** Enable LangChain's official DocsByLangChain MCP server in your coding agent's config. It exposes a `SearchDocsByLangChain` tool that queries live documentation, so the agent generates code against the current API rather than stale training data. It's a free, remote streamable HTTP server requiring no API key.

**Q: What is LangChain Hub and when would you use `hub.pull()`?**
> **A:** LangChain Hub is a community repository of versioned prompts hosted within LangSmith. You use `hub.pull("owner/prompt-name")` to download a battle-tested prompt as a starting point — especially useful for RAG prompts, agent system prompts, and classification tasks. It supports version pinning for reproducibility.

**Q: How do you choose text splitting parameters for RAG?**
> **A:** Start with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) — it preserves semantic structure by splitting hierarchically. Use the LangChain Text Splitting Playground to visually verify chunks contain coherent information. Adjust based on your actual data; there's no universal answer. For token-sensitive scenarios, use `from_tiktoken_encoder` to split by token count instead of characters.

**Q: LangChain vs LlamaIndex — which would you choose and why?**
> **A:** LangChain. Both handle RAG well, but LangChain has a significantly more robust agent ecosystem (LangGraph, `create_agent`, custom state graphs), more community adoption, and greater flexibility via LCEL. LlamaIndex is more data/search-focused and has basic agent support, but for production agentic applications, LangChain is the stronger choice.

---

## References

- [LangChain Docs MCP Server](https://docs.langchain.com/mcp) — official remote MCP server
- [chat.langchain.com](https://chat.langchain.com) — documentation chatbot powered by the same MCP
- [LangChain Hub](https://smith.langchain.com/hub) — community prompt repository
- [Text Splitting Playground](https://langchain-text-splitter.streamlit.app/) — visual chunk debugger
- [Text Splitter Docs](https://python.langchain.com/docs/concepts/text_splitters/) — official splitter documentation
- [LlamaIndex](https://www.llamaindex.ai/) — alternative LLM orchestration framework
