# 12. Documentation Assistant — Implementation

Step-by-step implementation guide: crawling documentation, building an agentic RAG backend, and wiring up a Streamlit chat UI.

*Topic: Documentation Helper Project*

> 📖 **Theory & Concepts:** [11_DocAssistant_Theory_And_Concepts.md](11_DocAssistant_Theory_And_Concepts.md)

---

## Project Overview

We build a complete **documentation assistant** that:
1. **Crawls** the LangChain documentation site using Tavily
2. **Chunks and embeds** thousands of pages into Pinecone
3. **Retrieves** relevant context using an agent with a retrieval tool
4. **Presents** answers with source citations in a Streamlit chat UI

### Project Structure

```
10-documentation-assistant/
├── 11_DocAssistant_Theory_And_Concepts.md   # Theory & definitions (previous file)
├── 12_DocAssistant_Implementation.md        # This file (implementation walkthrough)
└── src/
	├── ingestion.py                         # Phase 1: Crawl → Chunk → Embed → Store
	├── backend/
	│   ├── __init__.py
	│   └── core.py                          # Phase 2: Agent + retrieval tool
	├── main.py                              # Phase 3: Streamlit chat UI
	├── .streamlit/
	│   └── config.toml                      # Streamlit theme configuration
	├── test_tavily_extract.py               # Quick Tavily demo script
	└── _create_index.py                     # One-off Pinecone index creator
```

---

## 📦 Dependencies

```bash
uv add langchain langchain-openai langchain-pinecone langchain-tavily langchain-text-splitters python-dotenv streamlit truststore
```

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
INDEX_NAME=doc-helper-index
TAVILY_API_KEY=tvly-...

# LangSmith (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=doc-assistant
```

---

## Phase 1: Ingestion Pipeline (`ingestion.py`)

The ingestion pipeline runs **once** to crawl the documentation site and populate the vector store.

### Pipeline Flow

```
Tavily Crawl → LangChain Documents → RecursiveCharacterTextSplitter → OpenAI Embeddings → Pinecone
```

### Key Implementation Details

#### 1. Web Crawling with Tavily

```python
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document

tavily_crawl = TavilyCrawl()

res = tavily_crawl.invoke({
	"url": "https://python.langchain.com/",
	"max_depth": 2,
	"extract_depth": "advanced",
})

# Convert to LangChain Documents
all_docs = []
for item in res["results"]:
	all_docs.append(
		Document(
			page_content=item["raw_content"],
			metadata={"source": item["url"]},
		)
	)
```

#### 2. Chunking with RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)
```

Why `4000/200`?
- Documentation pages are structured and long — 4000 chars keeps complete code examples + explanations in one chunk
- 200 char overlap preserves context at chunk boundaries

#### 3. Async Batch Ingestion

```python
import asyncio
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name="langchain-docs-index", embedding=embeddings)

async def index_documents_async(documents, batch_size=50):
	batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

	async def add_batch(batch, num):
		await vectorstore.aadd_documents(batch)

	tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
	await asyncio.gather(*tasks, return_exceptions=True)
```

#### 4. Running Ingestion

```bash
uv run ingestion.py
```

Expected output:
```
🚀 DOCUMENTATION INGESTION PIPELINE
🗺️  TavilyCrawl: Crawling https://python.langchain.com/...
✅ Crawled 847 pages
✂️  Text Splitter: 847 documents → 2,341 chunks (4000/200)
📚 VectorStore: Adding 2,341 documents in 47 batches...
✅ All batches processed successfully!
```

---

## Phase 2: Backend — Agentic RAG (`backend/core.py`)

The backend creates an **agent** with a retrieval tool that decides when to search the docs.

### Key Implementation Details

#### 1. The Retrieval Tool

```python
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name="langchain-docs-index", embedding=embeddings)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
	"""Retrieve relevant documentation to help answer user queries about LangChain."""
	retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

	serialized = "\n\n".join(
		f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
		for doc in retrieved_docs
	)

	return serialized, retrieved_docs
```

**Why `content_and_artifact`?**
- First return value (`serialized`) → becomes `ToolMessage.content` that the LLM reads
- Second return value (`retrieved_docs`) → stored as `ToolMessage.artifact` for source citations

#### 2. The Agent

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o", model_provider="openai")

def run_llm(query: str) -> dict:
	system_prompt = (
		"You are a helpful AI assistant that answers questions about LangChain documentation. "
		"Use the tool to find relevant information before answering questions. "
		"Always cite the sources you use."
	)

	agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

	response = agent.invoke({"messages": [{"role": "user", "content": query}]})

	# Extract answer
	answer = response["messages"][-1].content

	# Extract source documents from ToolMessage artifacts
	context_docs = []
	for message in response["messages"]:
		if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
			if isinstance(message.artifact, list):
				context_docs.extend(message.artifact)

	return {"answer": answer, "context": context_docs}
```

---

## Phase 3: Streamlit Chat UI (`main.py`)

### Key Implementation Details

#### 1. Page Setup and Initial State

```python
import streamlit as st
from backend.core import run_llm

st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

if "messages" not in st.session_state:
	st.session_state.messages = [
		{"role": "assistant", "content": "Ask me anything about LangChain docs.", "sources": []}
	]
```

#### 2. Display Message History

```python
for msg in st.session_state.messages:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"])
		if msg.get("sources"):
			with st.expander("Sources"):
				for s in msg["sources"]:
					st.markdown(f"- {s}")
```

#### 3. Handle User Input

```python
prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
	st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

	with st.chat_message("assistant"):
		with st.spinner("Retrieving docs and generating answer…"):
			result = run_llm(prompt)
			answer = result.get("answer", "")
			sources = [doc.metadata.get("source", "Unknown") for doc in result.get("context", [])]

		st.markdown(answer)
		if sources:
			with st.expander("Sources"):
				for s in sources:
					st.markdown(f"- {s}")

	st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
```

#### 4. Sidebar Controls

```python
with st.sidebar:
	st.subheader("Session")
	if st.button("Clear chat", use_container_width=True):
		st.session_state.pop("messages", None)
		st.rerun()
```

#### 5. Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#1E1E1E"
secondaryBackgroundColor = "#252526"
textColor = "#FFFFFF"
font = "sans serif"
```

### Running the App

```bash
streamlit run main.py
# Opens http://localhost:8501
```

---

## How Everything Connects

```
User types question in Streamlit
	↓
st.chat_input() captures the text
	↓
run_llm(prompt) is called (backend/core.py)
	↓
create_agent() creates agent with retrieve_context tool
	↓
Agent decides: "I need to search the docs"
	↓
retrieve_context(query) is called
	↓
Pinecone similarity search returns top-4 chunks
	↓
Agent reads the retrieved context
	↓
Agent generates answer with source citations
	↓
Answer + sources returned to Streamlit
	↓
st.chat_message() renders the response
	↓
st.session_state saves for memory
```

---

## Key Differences from Common Reference Code (Our Adaptations)

| Common Reference | Our Adaptation | Why |
|-----------------|---------------|-----|
| `certifi` for SSL | `truststore.inject_into_ssl()` | Corporate proxy needs Windows cert store |
| `pipenv` | `uv` | Our standard package manager |
| `langchain_classic.text_splitter` | `langchain_text_splitters` | Use the dedicated package |
| `Chroma` (local) or `Pinecone` | `Pinecone` (serverless) | Consistent with our Section 9 setup |
| `gpt-5.2` model reference | `gpt-4o` | Available and production-proven |
| Custom `logger.py` with colors | Standard print with emoji | Simpler, consistent with our scripts |

---

## What's New in This Section (vs Section 9)

| Concept | Section 9 | Section 10 |
|---------|-----------|------------|
| Data source | Local file | Live website via Tavily |
| Splitter | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
| Embedding model | `text-embedding-ada-002` | `text-embedding-3-small` |
| Retrieval pattern | Deterministic LCEL chain | Agentic with `create_agent()` |
| Tool pattern | N/A | `@tool(response_format="content_and_artifact")` |
| Model init | `ChatOpenAI(model="gpt-4o")` | `init_chat_model("gpt-4o", ...)` |
| UI | CLI print statements | Streamlit web app |
| Memory | Manual list + reformulation | `st.session_state` |
| Ingestion scale | ~67 chunks | 1000+ chunks with async batching |

---

## Beyond Basic — Runnable Source Files

All scripts below are fully commented with prerequisites, config, and run instructions:

| File | What It Does | Run Command |
|------|-------------|-------------|
| [`src/ingestion.py`](src/ingestion.py) | Crawl → Chunk → Embed → Store (run once) | `uv run ingestion.py` |
| [`src/backend/core.py`](src/backend/core.py) | Agentic RAG backend (standalone test) | `uv run backend/core.py` |
| [`src/main.py`](src/main.py) | Streamlit chat UI (main app) | `streamlit run main.py` |

### Quick Start (after `.env` is configured)

```bash
# 1. Install dependencies
uv add langchain langchain-openai langchain-pinecone langchain-tavily langchain-text-splitters python-dotenv streamlit truststore

# 2. Populate the vector store (one-time)
cd 10-documentation-assistant/src
uv run ingestion.py

# 3. Launch the chat app
streamlit run main.py
```
