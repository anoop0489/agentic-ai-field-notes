# =============================================================================
# RAG EVALUATION: Measuring Retrieval & Generation Quality
# =============================================================================
# This script demonstrates how to evaluate a RAG system beyond "it looks right."
# It measures the three dimensions of RAG quality:
#   1. Retrieval Relevance — did the retriever find the right chunks?
#   2. Answer Faithfulness — is the answer grounded in the retrieved context?
#   3. Answer Relevance — does the answer actually address the question?
#
# THE PROBLEM:
#   You built a RAG pipeline and it "seems to work." But in production, you need
#   measurable metrics to catch regressions, compare chunk sizes, or justify
#   architecture decisions. "Eyeball testing" doesn't scale.
#
# THE APPROACH:
#   We use LLM-as-judge evaluation — the same LLM (or a cheaper one) scores the
#   output along each dimension. This is the pattern used by RAGAS, DeepEval, and
#   LangSmith evaluators under the hood.
#
# THIS SCRIPT SHOWS:
#   1. Building a standard LCEL RAG chain
#   2. Running it with retrieval + answer capture (not just the final string)
#   3. Evaluating faithfulness (grounding) with an LLM judge
#   4. Evaluating relevance with an LLM judge
#   5. Detecting hallucination by comparing answer claims to source chunks
#   6. Batch evaluation across a test dataset
#
# PREREQUISITES:
#   - Pinecone index populated (run ingestion.py first)
#   - .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#
# USAGE:
#   uv run test_rag_evaluation.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()


def check_prerequisites():
    errors = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.environ.get("PINECONE_API_KEY"):
        errors.append("PINECONE_API_KEY not found in .env")
    if not os.environ.get("INDEX_NAME"):
        errors.append("INDEX_NAME not found in .env")
    if errors:
        print("❌ Prerequisites check FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)


# --- Evaluation Prompts (LLM-as-Judge) ---

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are an evaluation judge. Given the CONTEXT (retrieved chunks) and the ANSWER,
determine if the answer is fully grounded in the context.

Score on a scale of 1-5:
  5 = Every claim in the answer is directly supported by the context
  4 = Most claims supported, minor inference that's reasonable
  3 = Some claims supported, some unsupported but plausible
  2 = Significant claims not found in context
  1 = Answer contradicts or fabricates information not in context

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence explanation>"}}"""
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    """You are an evaluation judge. Given the QUESTION and the ANSWER,
determine if the answer actually addresses what was asked.

Score on a scale of 1-5:
  5 = Directly and completely answers the question
  4 = Answers the question with minor gaps
  3 = Partially answers but misses key aspects
  2 = Tangentially related but doesn't answer the question
  1 = Completely off-topic or refuses to answer

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence explanation>"}}"""
)

RETRIEVAL_RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    """You are an evaluation judge. Given the QUESTION and the RETRIEVED CHUNKS,
determine if the retriever found relevant information.

Score on a scale of 1-5:
  5 = All chunks are highly relevant to answering the question
  4 = Most chunks relevant, one may be marginally useful
  3 = Mixed — some relevant, some irrelevant chunks
  2 = Mostly irrelevant chunks with traces of useful info
  1 = None of the chunks help answer the question

QUESTION:
{question}

RETRIEVED CHUNKS:
{context}

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence explanation>"}}"""
)


# --- Test Dataset ---
# In production, this would come from a CSV/JSON file or LangSmith dataset.
# Each entry has a question and (optionally) an expected answer for reference.

TEST_QUESTIONS = [
    {
        "question": "What is Pinecone in machine learning?",
        "expected_topic": "vector database",
    },
    {
        "question": "How do vector databases differ from traditional databases?",
        "expected_topic": "similarity search",
    },
    {
        "question": "What are embeddings and why are they useful?",
        "expected_topic": "numerical representation",
    },
    {
        "question": "Who won the 2024 Super Bowl?",
        "expected_topic": None,  # Not in our knowledge base — should say "I don't know"
    },
]


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def run_rag_with_retrieval_capture(chain, retriever, question):
    """Run RAG and capture both the answer AND the retrieved chunks."""
    retrieved_docs = retriever.invoke(question)
    context_text = format_docs(retrieved_docs)
    answer = chain.invoke({"question": question})
    return {
        "question": question,
        "context": context_text,
        "answer": answer,
        "num_chunks": len(retrieved_docs),
        "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
    }


def evaluate_single(judge_llm, result):
    """Run all three evaluations on a single RAG result."""
    faithfulness_chain = FAITHFULNESS_PROMPT | judge_llm | StrOutputParser()
    relevance_chain = RELEVANCE_PROMPT | judge_llm | StrOutputParser()
    retrieval_chain = RETRIEVAL_RELEVANCE_PROMPT | judge_llm | StrOutputParser()

    import json

    scores = {}

    for name, eval_chain, inputs in [
        ("faithfulness", faithfulness_chain, {"context": result["context"], "answer": result["answer"]}),
        ("answer_relevance", relevance_chain, {"question": result["question"], "answer": result["answer"]}),
        ("retrieval_relevance", retrieval_chain, {"question": result["question"], "context": result["context"]}),
    ]:
        raw = eval_chain.invoke(inputs)
        try:
            parsed = json.loads(raw)
            scores[name] = parsed
        except json.JSONDecodeError:
            scores[name] = {"score": 0, "reason": f"Failed to parse: {raw[:100]}"}

    return scores


if __name__ == "__main__":
    check_prerequisites()

    print("=" * 70)
    print("📊 RAG EVALUATION — Measuring Quality Across 3 Dimensions")
    print("=" * 70)
    print(f"   Index: {os.environ['INDEX_NAME']}")
    print(f"   Test questions: {len(TEST_QUESTIONS)}")
    print()

    # ---- Build the RAG chain ----
    embeddings = OpenAIEmbeddings()
    rag_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context.
If the context doesn't contain the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | rag_llm
        | StrOutputParser()
    )

    # ---- Run evaluation ----
    all_scores = []

    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        print(f"\n{'─' * 70}")
        print(f"  Question {i}/{len(TEST_QUESTIONS)}: {question}")
        print(f"{'─' * 70}")

        result = run_rag_with_retrieval_capture(chain, retriever, question)

        print(f"  📄 Retrieved: {result['num_chunks']} chunks")
        print(f"  💬 Answer: {result['answer'][:120]}...")
        print()

        scores = evaluate_single(judge_llm, result)
        all_scores.append({"question": question, **scores})

        for dimension, evaluation in scores.items():
            score = evaluation.get("score", "?")
            reason = evaluation.get("reason", "N/A")
            emoji = "✅" if score >= 4 else "⚠️" if score >= 3 else "❌"
            print(f"  {emoji} {dimension:22s} → {score}/5  ({reason})")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)

    for dimension in ["faithfulness", "answer_relevance", "retrieval_relevance"]:
        scores_list = [
            entry[dimension]["score"]
            for entry in all_scores
            if isinstance(entry.get(dimension, {}).get("score"), int)
        ]
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            emoji = "✅" if avg >= 4 else "⚠️" if avg >= 3 else "❌"
            print(f"  {emoji} {dimension:22s} → avg {avg:.1f}/5")

    print()
    print("=" * 70)
    print("KEY TAKEAWAYS:")
    print("  • Scores ≥ 4 = production-ready for that dimension")
    print("  • Scores 2-3 = investigate (bad chunks? weak prompt? wrong k?)")
    print("  • Score 1 = broken — likely wrong index or hallucination")
    print("  • The 'out-of-scope' question tests grounding (should refuse)")
    print("=" * 70)
