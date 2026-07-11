# =============================================================================
# CONVERSATIONAL RAG: Multi-Turn Q&A Over a Document
# =============================================================================
# This example demonstrates how to build a chatbot that can answer follow-up
# questions about a document. Unlike single-shot RAG (ask one question, get one
# answer, forget everything), this maintains conversation history.
#
# THE PROBLEM:
#   User: "What is CTS?"
#   Bot:  "CTS is the Centralized Tracking Service..."
#   User: "How does the Hub connect to it?"
#                                       ^^
#   The word "it" refers to CTS from the previous question.
#   But the retriever doesn't know that — it just sees "How does the Hub
#   connect to it?" and searches Pinecone for... what? "it" has no meaning
#   in isolation.
#
# THE SOLUTION — Question Reformulation:
#   Before searching the vector store, we use the LLM to REWRITE the follow-up
#   question into a standalone question using the chat history:
#
#   Chat History: [("What is CTS?", "CTS is the Centralized Tracking Service...")]
#   Follow-up:   "How does the Hub connect to it?"
#   Reformulated: "How does the Hub connect to the Centralized Tracking Service (CTS)?"
#                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   NOW the retriever can find relevant chunks because the query is self-contained.
#
# ARCHITECTURE:
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │ User Question ──→ Reformulate (if follow-up) ──→ Retrieve ──→ LLM   │
#   │       ↑                    ↑                                    │    │
#   │       │              Chat History                               │    │
#   │       │                                                         ↓    │
#   │       └──────────────── Answer + Update History ←───────────────┘    │
#   └──────────────────────────────────────────────────────────────────────┘
#
# C# ANALOGY:
#   Think of this like a SignalR chat with server-side session state.
#   Each user session holds a List<ChatMessage> that grows with each turn.
#   The "reformulation" step is like a middleware that preprocesses the
#   incoming message before it hits the business logic (retrieval).
#
# PREREQUISITES:
#   - Run test_multimodal_pdf_rag.py first to populate Pinecone with your PDF vectors
#   - .env file with: OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#
# USAGE:
#   uv run test_conversational_rag.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


# ========================= CONFIGURATION =========================
TOP_K = 3
# =================================================================


def check_prerequisites():
    """Verify environment is ready."""
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
        print("\n   Make sure you've run test_pdf_rag.py first to populate Pinecone.")
        sys.exit(1)


# =============================================================================
# STEP 1: Question Reformulation
# =============================================================================
# This is the KEY difference between single-shot RAG and conversational RAG.
# Without this step, follow-up questions with pronouns ("it", "that", "those")
# or implicit references will fail to retrieve relevant context.

REFORMULATION_PROMPT = ChatPromptTemplate.from_template(
    """Your job is to determine if a follow-up question references the prior conversation,
and ONLY if it does, rewrite it as a standalone question.

RULES:
1. If the question contains pronouns referring to prior context ("it", "they", "that",
   "this system", "the same"), rewrite it to replace those references with the actual
   entities from the chat history.
2. If the question is about a COMPLETELY DIFFERENT TOPIC that does NOT reference
   anything in the chat history, return it EXACTLY as-is. Do NOT add context from
   prior questions.
3. When in doubt, return the question unchanged. Less reformulation is better than
   wrong reformulation.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
)


def reformulate_question(llm, question: str, chat_history: list) -> str:
    """
    Rewrite a follow-up question into a standalone question.

    Examples:
        History: [("What is CTS?", "CTS is the Centralized Tracking Service")]
        Input:   "How does the Hub connect to it?"
        Output:  "How does the Hub connect to the Centralized Tracking Service (CTS)?"

        History: []
        Input:   "What is CTS?"
        Output:  "What is CTS?"  (unchanged — already standalone)
    """
    # If no history, the question is already standalone — skip the LLM call
    if not chat_history:
        return question

    # Format chat history as readable text
    history_text = ""
    for human_msg, ai_msg in chat_history:
        history_text += f"Human: {human_msg}\nAssistant: {ai_msg}\n\n"

    # Ask the LLM to reformulate
    messages = REFORMULATION_PROMPT.format_messages(
        chat_history=history_text, question=question
    )
    response = llm.invoke(messages)
    reformulated = response.content.strip()

    return reformulated


# =============================================================================
# STEP 2: RAG Answer Generation
# =============================================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a document.
Use ONLY the following context to answer. If the context doesn't contain
enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
)


def generate_answer(llm, retriever, question: str, reformulated_question: str) -> str:
    """
    Retrieve relevant chunks and generate an answer.

    Note: We search with the REFORMULATED question (better retrieval)
    but show the ORIGINAL question in the prompt (more natural).
    """
    # Retrieve using the reformulated question (has full context for better search)
    docs = retriever.invoke(reformulated_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    messages = ANSWER_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return response.content, docs


# =============================================================================
# MAIN: Interactive Conversation Loop
# =============================================================================

if __name__ == "__main__":
    check_prerequisites()

    print("=" * 70)
    print("📚 CONVERSATIONAL RAG — Multi-Turn Document Q&A")
    print("=" * 70)
    print(f"   Index: {os.environ['INDEX_NAME']}")
    print(f"   Top-K: {TOP_K}")
    print()
    print("   Ask questions about your document. Follow-ups are supported!")
    print("   The system will reformulate follow-up questions automatically.")
    print()
    print("   Type 'quit' to exit, 'history' to see conversation so far,")
    print("   or 'clear' to reset conversation history.")
    print("=" * 70)
    print()

    # Initialize components
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Chat history — list of (human_question, ai_answer) tuples
    # This is the simplest form of memory. In production, you'd use:
    # - LangChain's ConversationBufferMemory (for short conversations)
    # - ConversationSummaryMemory (for long conversations — summarizes old turns)
    # - A database (for persistent memory across sessions)
    chat_history: list[tuple[str, str]] = []

    turn = 0
    while True:
        # Get user input
        try:
            question = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            print("\nGoodbye!")
            break
        if question.lower() == "history":
            if not chat_history:
                print("   (no history yet)\n")
            else:
                print("\n   📜 Conversation History:")
                for i, (q, a) in enumerate(chat_history):
                    print(f"   Turn {i+1}:")
                    print(f"     Q: {q}")
                    print(f"     A: {a[:100]}...")
                    print()
            continue
        if question.lower() == "clear":
            chat_history.clear()
            turn = 0
            print("   ✅ History cleared!\n")
            continue

        turn += 1
        print()

        # Step 1: Reformulate the question (make it standalone)
        reformulated = reformulate_question(llm, question, chat_history)

        # Show the reformulation if it changed (educational — shows what happened)
        if reformulated != question and chat_history:
            print(f"   🔄 Reformulated: \"{reformulated}\"")
            print()

        # Step 2: Retrieve + Generate
        answer, retrieved_docs = generate_answer(
            llm, retriever, question, reformulated
        )

        # Show retrieved sources (abbreviated)
        print(f"   📎 Sources: {len(retrieved_docs)} chunks retrieved")
        for i, doc in enumerate(retrieved_docs):
            page = doc.metadata.get("page", "?")
            print(f"      - Chunk {i+1} (page {page}): \"{doc.page_content[:60]}...\"")
        print()

        # Show answer
        print(f"🤖 Assistant: {answer}")
        print()

        # Step 3: Update history
        chat_history.append((question, answer))

        # Keep history manageable (last 5 turns)
        # In production, use ConversationSummaryMemory for long conversations
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]
