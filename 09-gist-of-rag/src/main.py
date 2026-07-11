# =============================================================================
# RAG RETRIEVAL PIPELINE: Retrieve → Augment → Generate
# =============================================================================
# This script runs PER USER QUERY to retrieve relevant context from the vector
# store and generate a grounded answer. It is the "online" phase of RAG.
#
# It demonstrates THREE approaches:
#   0. Raw LLM call (no RAG) — baseline showing the problem
#   1. Naive implementation (manual step-by-step) — educational, shows the flow
#   2. LCEL implementation (declarative chain) — production-ready
#
# Prerequisites:
#   1. Run ingestion.py first to populate Pinecone with vectors
#   2. .env file with OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME
#   3. (Optional) LANGSMITH_TRACING=true for observability
# =============================================================================

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================
# All components are initialized once and shared across implementations.

print("Initializing components...")

# Embedding model — same model used during ingestion (MUST match dimensions)
embeddings = OpenAIEmbeddings()

# LLM for answer generation
llm = ChatOpenAI()

# Connect to the existing Pinecone index (populated by ingestion.py)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Create a retriever: wraps vectorstore.similarity_search() as a LangChain Runnable
# search_kwargs={"k": 3} means: return the 3 most similar chunks
# The retriever embeds the query, searches Pinecone, and returns Document objects.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template with two placeholders:
#   {context} — where retrieved chunks are injected (the "augmentation" in RAG)
#   {question} — the user's original query
# "based only on the following context" — prevents the LLM from using its own knowledge
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """
    Format retrieved documents into a single context string.

    Takes a list of LangChain Document objects and joins their page_content
    with double newlines. This string is what gets injected into the prompt
    template's {context} placeholder.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# =============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# =============================================================================
# This is the most transparent implementation — every step is explicit.
# Great for learning, but has limitations for production use.


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution (imperative)
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - Each step traces separately in LangSmith (fragmented debugging)
    """
    # Step 1: Retrieve relevant documents
    # Under the hood: query is embedded → Pinecone similarity search → top-3 docs
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    # Joins all page_content with "\n\n" separators
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    # Fills {context} and {question} placeholders in the template
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    # This is a standard LLM call — the augmented prompt goes in, answer comes out
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# =============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - PRODUCTION APPROACH
# =============================================================================
# Same logic as above, but composed as a single declarative chain.
# Benefits: streaming, async, batch, composability, unified tracing.


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain (Runnable) that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Unified tracing: Entire pipeline appears as one LangSmith trace
    """
    # The chain reads left-to-right, top-to-bottom:
    #
    # 1. RunnablePassthrough.assign(context=...)
    #    - Input: {"question": "what is Pinecone?"}
    #    - Passes input through (keeps "question" key)
    #    - Computes new "context" key by running the sub-chain:
    #        itemgetter("question") → extracts the question string
    #        | retriever            → embeds & searches Pinecone → [Doc1, Doc2, Doc3]
    #        | format_docs          → "chunk1\n\nchunk2\n\nchunk3"
    #    - Output: {"question": "what is Pinecone?", "context": "chunk1\n\n..."}
    #
    # 2. prompt_template
    #    - Input: dict with "question" and "context" keys
    #    - Fills the template placeholders
    #    - Output: ChatPromptValue (list of messages)
    #
    # 3. llm
    #    - Input: messages list
    #    - Output: AIMessage(content="Pinecone is a...")
    #
    # 4. StrOutputParser()
    #    - Input: AIMessage
    #    - Output: just the string content ("Pinecone is a...")
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# =============================================================================
# MAIN: Run all implementations for comparison
# =============================================================================

if __name__ == "__main__":
    print("Retrieving...")

    # The test query — Pinecone as a vector DB is in our ingested blog,
    # but GPT-3.5 might not know about it (or might hallucinate).
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG (demonstrates the PROBLEM)
    # ========================================================================
    # Without context, the LLM may hallucinate or give outdated information.
    # This is WHY we need RAG.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    # Same answer as Option 2, but each step is a separate function call.
    # Traces separately in LangSmith — harder to debug.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    # Production-ready: streaming, async, unified tracing.
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
