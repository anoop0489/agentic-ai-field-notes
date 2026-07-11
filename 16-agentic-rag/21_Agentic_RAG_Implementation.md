# 21. Agentic RAG — Implementation Guide

> Step-by-step implementation of Adaptive RAG with Self-Correction using LangGraph, ChromaDB, and Tavily.

---

## Project Structure

```
16-agentic-rag/
├── 20_Agentic_RAG.md                  # Theory & concepts
├── 21_Agentic_RAG_Implementation.md   # This file
├── 22_Production_Optimisation.md      # Cost/latency guide (rerankers, fewer LLM calls)
└── src/
	├── main.py                        # Entry point
	├── ingestion.py                   # ChromaDB ingestion (run once)
	└── graph/
		├── __init__.py
		├── consts.py                  # Node name constants
		├── state.py                   # GraphState TypedDict
		├── graph.py                   # Full graph definition
		├── chains/
		│   ├── __init__.py
		│   ├── router.py             # Question routing (vectorstore vs web)
		│   ├── retrieval_grader.py   # Document relevance scoring
		│   ├── hallucination_grader.py # Generation grounding check
		│   ├── answer_grader.py      # Answer relevance check
		│   ├── generation.py         # RAG generation chain
		│   └── tests/
		│       ├── __init__.py
		│       └── test_chains.py    # Unit tests for all chains
		└── nodes/
			├── __init__.py
			├── retrieve.py           # ChromaDB retrieval
			├── grade_documents.py    # Filter irrelevant docs
			├── generate.py           # Generate answer
			└── web_search.py         # Tavily fallback
```

---

## Dependencies

```bash
uv add langchain langchain-openai langchain-chroma langchain-community langchain-tavily langchainhub tiktoken python-dotenv truststore
```

---

## Environment Variables

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional — LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

---

## Step 1: Ingestion (Run Once)

Loads three blog posts, chunks them, and embeds into ChromaDB:

```bash
uv run python 16-agentic-rag/src/ingestion.py
```

This creates a `.chroma/` directory with the persisted vector store.

**Key decisions:**
- `chunk_size=250` (tiktoken tokens) — small enough for precise retrieval
- `chunk_overlap=0` — no overlap (keeps chunks independent)
- `OpenAIEmbeddings()` — defaults to `text-embedding-ada-002` (1536 dims)

---

## Step 2: Graph State

The state flows through every node and edge:

```python
class GraphState(TypedDict):
	question: str        # Original user question (never mutated)
	generation: str      # LLM-generated answer (set by generate node)
	web_search: bool     # Flag: should we do web search? (set by grade_documents)
	documents: List[str] # Retrieved/filtered/augmented documents
```

---

## Step 3: Chains (the 5 LLM "brains")

Each chain is a standalone LCEL pipe: `prompt | llm.with_structured_output(Model)`

| Chain | File | Input | Output |
|-------|------|-------|--------|
| Router | `chains/router.py` | `{question}` | `RouteQuery(datasource="vectorstore" \| "websearch")` |
| Retrieval Grader | `chains/retrieval_grader.py` | `{question, document}` | `GradeDocuments(binary_score="yes" \| "no")` |
| Hallucination Grader | `chains/hallucination_grader.py` | `{documents, generation}` | `GradeHallucinations(binary_score=True \| False)` |
| Answer Grader | `chains/answer_grader.py` | `{question, generation}` | `GradeAnswer(binary_score=True \| False)` |
| Generation | `chains/generation.py` | `{context, question}` | `str` (freeform answer) |

---

## Step 4: Nodes (the 4 graph steps)

Each node receives `GraphState`, does work, returns a partial state update:

| Node | What It Does | Key Logic |
|------|-------------|-----------|
| `retrieve` | Queries ChromaDB | `retriever.invoke(question)` |
| `grade_documents` | Loops through docs, grades each | Sets `web_search=True` if ANY doc is irrelevant |
| `generate` | Calls generation chain | `generation_chain.invoke({context, question})` |
| `web_search` | Queries Tavily | Appends web results to existing documents |

---

## Step 5: Graph Assembly

```python
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Entry: router decides where to start
workflow.set_conditional_entry_point(route_question, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE})

# Fixed edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(WEBSEARCH, GENERATE)

# Conditional: after grading docs
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})

# Conditional: after generation (Self-RAG checks)
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
	"not supported": GENERATE,  # Hallucinated → retry
	"useful": END,              # Good answer → done
	"not useful": WEBSEARCH,    # Doesn't answer question → need more info
})

app = workflow.compile()
```

---

## Step 6: Run It

```bash
uv run python 16-agentic-rag/src/main.py
```

**Expected output:**
```
---ROUTE QUESTION---
---ROUTE QUESTION TO RAG---
---RETRIEVE---
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---DECISION: GENERATE---
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---
```

---

## Step 7: Run Tests

```bash
uv run pytest 16-agentic-rag/src/graph/chains/tests/test_chains.py -s -v
```

---

## Beyond Basic

The theory file covers production considerations in detail:

→ [Section 10: Production Considerations](./20_Agentic_RAG.md#10-production-considerations) — infinite loop prevention, cost optimization, latency optimization

→ [Section 9: Comparison Matrix](./20_Agentic_RAG.md#9-comparison-deterministic-rag-vs-self-rag-vs-adaptive-rag) — when to use which RAG variant

→ [Section 12 of Reflexion Notes](../15-reflexion-agent/19_Reflexion_Agent.md#12-production-grade-checkpointers) — adding PostgresSaver for production persistence (applies to this graph too)
