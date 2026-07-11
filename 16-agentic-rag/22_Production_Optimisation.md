# 22. Production Optimisation — Agentic RAG Cost & Latency Guide

> **Context:** A companion to Section 16's Agentic RAG implementation. the reference graph is optimised for **learnability** (each concept is a separate testable node). This guide shows how to collapse that into fewer LLM calls for production systems where cost and latency matter.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [The Cost Problem](#1-the-cost-problem) | Token/call analysis of the reference implementation |
| 2 | [Replace Router with Retrieval Score](#2-replace-router-with-retrieval-score) | Eliminate 1 LLM call with embedding distance thresholds |
| 3 | [Replace Per-Doc Grading with Rerankers](#3-replace-per-doc-grading-with-rerankers) | Eliminate N LLM calls with a cross-encoder |
| 4 | [Collapse Post-Generation Checks into One Call](#4-collapse-post-generation-checks-into-one-call) | Self-assessment in the generation prompt |
| 5 | [The Minimal Architecture](#5-the-minimal-architecture) | 1-2 LLM calls total vs 8+ |
| 6 | [When to Keep the Decomposed Graph](#6-when-to-keep-the-decomposed-graph) | Trade-offs: debuggability vs efficiency |
| 7 | [Interview Q&A Anchors](#7-interview-qa-anchors) | How to talk about this in interviews |

---

## 1. The Cost Problem

the reference graph makes **8+ LLM calls per query** on the happy path (4 retrieved docs):

| Step | LLM Calls | Why It Exists |
|------|-----------|---------------|
| Router | 1 | Classify question → vectorstore or web |
| Retrieval Grader | 4 (one per doc) | Filter irrelevant chunks |
| Generation | 1 | Produce the answer |
| Hallucination Grader | 1 | Check grounding in docs |
| Answer Grader | 1 | Check relevance to question |
| **Total (happy path)** | **8** | |

If hallucination or answer check fails → retry loop adds more. A single query can hit **12-15 LLM calls**.

**Why it is done this way:** Each chain is independently testable, visible in LangSmith traces, and teaches one concept. This is the right design for learning. But in production, you'd consolidate.

---

## 2. Replace Router with Retrieval Score

**Problem:** The router uses an LLM call to decide "should I check my vectorstore or go to web search?"

**Better approach:** Just retrieve and check the similarity score. If the best result is far from the query, fall back to web search. Zero LLM calls.

```python
# Instead of: question_router.invoke({"question": question})
# Do this:
results = vectorstore.similarity_search_with_score(question, k=4)

# ChromaDB returns L2 distance (lower = more similar)
# Cosine stores return cosine similarity (higher = more similar)
best_score = results[0][1]  # score of top result

if best_score > DISTANCE_THRESHOLD:  # too far from any known doc
	# Route to web search — no LLM call spent
	return "websearch"
else:
	# Use these results — they're relevant enough
	return "vectorstore"
```

**Trade-off:** Less nuanced than an LLM classifier, but for well-scoped corpora (like "agents + prompt eng + adversarial attacks") a threshold works fine. You can calibrate with a few test queries.

---

## 3. Replace Per-Doc Grading with Rerankers

**Problem:** The `grade_documents` node calls the LLM once *per document* to check relevance. With 4 docs, that's 4 LLM calls (~4-8 seconds, ~$0.01-0.04).

**Better approach:** Use a **reranker** (cross-encoder model) that scores all documents in one fast pass.

### What's a Reranker?

Two-stage retrieval is the industry standard:

| Stage | Model Type | Speed | Accuracy | How It Works |
|-------|-----------|-------|----------|--------------|
| 1. Retrieve | Bi-encoder (embeddings) | Very fast | Good | Pre-computed vectors, cosine similarity |
| 2. Rerank | Cross-encoder | Fast (~50-200ms) | Excellent | Sees query + doc together, joint attention |

The bi-encoder (ChromaDB) gives you **recall** (find candidates). The cross-encoder gives you **precision** (rank them properly).

### LangChain Integration

```python
# Option A: Cohere Rerank API (~$2/1000 queries — per query, NOT per doc)
from langchain_cohere import CohereRerank

reranker = CohereRerank(top_n=3, model="rerank-v3.5")

# Option B: Free local cross-encoder (no API costs at all)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

cross_encoder = HuggingFaceCrossEncoder(
	model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Wrap your retriever — reranking happens automatically on retrieve
from langchain.retrievers import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
	base_compressor=reranker,
	base_retriever=retriever,  # your existing ChromaDB retriever
)

# Now retrieval returns only the top-3 most relevant docs
# No LLM grading needed — the cross-encoder already filtered
docs = compression_retriever.invoke("agent memory")
```

### Popular Rerankers

| Reranker | Type | Cost | Quality | Notes |
|----------|------|------|---------|-------|
| Cohere Rerank v3.5 | API | ~$2/1K queries | Excellent | Best API option |
| Jina Reranker v2 | API/local | Free tier | Very good | Can self-host |
| `ms-marco-MiniLM-L-6-v2` | Local (HF) | Free | Good | Lightweight, fast |
| BGE Reranker v2 (BAAI) | Local (HF) | Free | Very good | Strong open-source |
| Voyage AI Rerank 2 | API | Cheap | Very good | Good accuracy/cost ratio |

### Cost Comparison

| Approach | Calls per query | Latency | Cost (GPT-4o pricing) |
|----------|----------------|---------|----------------------|
| LLM grading (4 docs) | 4 LLM calls | ~4-8s | ~$0.01-0.04 |
| Cohere Rerank | 1 API call | ~100ms | ~$0.002 |
| Local cross-encoder | 1 model inference | ~50-200ms | $0 |

---

## 4. Collapse Post-Generation Checks into One Call

**Problem:** After generation, the graph makes 2 separate LLM calls:
1. Hallucination grader: "Is this grounded?"
2. Answer grader: "Does this address the question?"

**Better approach:** Ask the LLM to self-assess during generation itself.

```python
system = """Answer the question using ONLY the provided context documents.

After your answer, provide a self-assessment in this exact JSON format:
{
  "answer": "<your answer here>",
  "grounded": true/false,
  "addresses_question": true/false,
  "confidence": 0-100,
  "needs_web_search": true/false
}

Rules:
- If you cannot fully answer from the documents, set grounded=false
- If the documents don't cover the topic, set needs_web_search=true
- Be honest in your self-assessment — it's used for quality routing"""
```

This collapses **generation + hallucination check + answer check** into **1 LLM call**.

**Trade-off:** Self-assessment is less reliable than an independent grader (the model may not catch its own hallucinations). For high-stakes applications, keep the separate hallucination checker. For most use cases, self-assessment + retrieval score threshold is sufficient.

### Alternative: Use Logprobs for Confidence

```python
# GPT-4o can return token-level log probabilities
response = llm.invoke(prompt, logprobs=True, top_logprobs=5)

# If average confidence is high, skip verification entirely
avg_logprob = mean([token.logprob for token in response.logprobs])
if avg_logprob > HIGH_CONFIDENCE_THRESHOLD:
	return response  # Skip grading — model is confident
else:
	# Only run expensive verification on low-confidence answers
	...
```

---

## 5. The Minimal Architecture

Combining all optimisations:

```
Question
	│
	▼
Retrieve + Rerank (ChromaDB + cross-encoder)  ← 0 LLM calls
	│
	├─ Best reranker score < threshold? → Web search fallback
	│
	▼
Generate with self-assessment (1 LLM call)    ← 1 LLM call
	│
	├─ confidence high + grounded? → Return answer (done)
	├─ needs_web_search? → Supplement with Tavily, retry
	├─ not grounded? → Retry generation (same docs)
	│
	▼
Done: 1-2 LLM calls total
```

**Comparison:**

| Architecture | LLM Calls | Latency | Monthly cost (1K queries/day) |
|--------------|-----------|---------|-------------------------------|
| Reference graph (Section 16) | 8-15 | ~10-20s | ~$300-900 |
| Optimised (reranker + self-assess) | 1-2 | ~2-4s | ~$30-60 |

---

## 6. When to Keep the Decomposed Graph

This architecture isn't wrong — it's optimised for different goals:

| Keep decomposed graph when... | Collapse into fewer calls when... |
|------------------------------|-----------------------------------|
| You need full LangSmith observability per step | Cost/latency is the primary concern |
| Each grading decision needs audit trail (compliance) | You trust the model's self-assessment |
| You're building for a regulated domain (medical, legal) | Your corpus is well-scoped and retrieval is reliable |
| You want to A/B test individual graders | You need sub-3-second response times |
| Team debugging: each node is independently testable | Scale: thousands of queries per minute |

**The real-world middle ground:** Use rerankers for document grading (always — it's strictly better), keep the router if your corpus boundaries are fuzzy, and conditionally run hallucination checks only on low-confidence answers.

---

## 7. Interview Q&A Anchors

**Q: The Agentic RAG pattern uses N LLM calls per doc for grading. How would you reduce costs in production?**

> **A:** Replace per-document LLM grading with a cross-encoder reranker — it scores all documents in one pass (~50ms, zero LLM cost) while being more accurate than binary yes/no classification. For routing, use the vectorstore's relevance score threshold instead of an LLM classifier. For post-generation checks, embed self-assessment into the generation prompt to collapse 3 calls into 1.

**Q: When would you still use the decomposed multi-step pattern?**

> **A:** In regulated domains where every grading decision needs an audit trail, or when you need to A/B test individual components independently. The decomposed graph is also valuable during development because each node is independently testable and visible in LangSmith traces.

**Q: What's a reranker and why is it better than LLM-based document grading?**

> **A:** A reranker is a cross-encoder that scores query-document relevance by attending to both jointly — unlike bi-encoders (embeddings) that encode them separately. It's better because: (1) it's ~100x cheaper than an LLM call, (2) ~50x faster, (3) purpose-built for relevance scoring so often more accurate, and (4) it processes all documents in a single batch rather than N separate calls.

**Q: What's the trade-off of LLM self-assessment vs independent graders?**

> **A:** Self-assessment (asking the model to rate its own grounding and relevance) saves 2 LLM calls but is less reliable — models sometimes don't catch their own hallucinations. The pragmatic approach is: use self-assessment for most queries, and only invoke independent graders on low-confidence answers where the model itself signals uncertainty.

---

## References

- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking)
- [LangChain ContextualCompressionRetriever](https://python.langchain.com/docs/how_to/contextual_compression/)
- [Cross-Encoders vs Bi-Encoders (SBERT)](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Jina Reranker](https://jina.ai/reranker/)
- [BGE Reranker (BAAI)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Self-RAG Paper (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Adaptive RAG Paper (Jeong et al., 2024)](https://arxiv.org/abs/2403.14403)
