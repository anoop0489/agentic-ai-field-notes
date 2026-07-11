# =============================================================================
# LANGCHAIN FUNDAMENTALS: PromptTemplates & LCEL Chain
# =============================================================================
# Demonstrates the core building block of LangChain:
#   - ChatPromptTemplate (message-based prompts with roles)
#   - LCEL pipe operator (| chains components together)
#   - Model agnosticism (swap OpenAI ↔ Ollama without changing chain logic)
#   - StrOutputParser (extracts .content from AIMessage)
#
# WHAT IT DOES:
#   Takes a person's background information, passes it through a prompt
#   template + LLM chain, and generates a summary with interesting facts.
#
# KEY CONCEPTS:
#   - ChatPromptTemplate.from_messages(): Factory method (see Glossary #2)
#   - chain = template | llm | parser: Operator overloading (see Glossary #3)
#   - chain.invoke({"key": value}): Triggers execution, injects variables
#   - temperature=0: Deterministic output (no randomness)
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY
#   2. Packages: uv add langchain-core langchain-openai python-dotenv truststore
#
# USAGE:
#   uv run 02-gist-of-langchain/src/01_langchain_fundamentals.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

# Import logger utility for colored demo output
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "reference-guides"))
from logger import log_header, log_info, log_success, log_error

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def check_prerequisites():
    """Verify required env vars exist before making API calls."""
    if not os.environ.get("OPENAI_API_KEY"):
        log_error("OPENAI_API_KEY not found in .env")
        sys.exit(1)
    log_success("Prerequisites met — OPENAI_API_KEY found")


def main():
    check_prerequisites()

    log_header("LANGCHAIN FUNDAMENTALS: PromptTemplates & LCEL")

    # ========================= THE DATA =========================
    # Raw unstructured text — this is what we'll pass to the LLM
    information = """
    Anoop is a Principal/Senior Software Engineer with 12+ years of experience
    designing scalable, reliable, and maintainable software systems. Passionate
    about leveraging agile methodologies and cloud platforms to deliver innovative
    solutions to complex challenges.
    Professional Expertise: C#, Java, Golang, Python, Node.js, React, Angular,
    Kubernetes, Docker, Apache Kafka, RabbitMQ, Redis, AWS, Azure.
    Academic Background: Master's degree in Computer Science from the
    University of New Orleans.
    """

    # ========================= APPROACH 1: PromptTemplate =========================
    # String-based template — simple, for non-conversational tasks
    # C# equivalent: String.Format("given the info {0} about a person...", information)
    log_info("APPROACH 1: PromptTemplate (String-based)")
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    log_info(f"  Template variables: {summary_prompt_template.input_variables}")

    # ========================= APPROACH 2: ChatPromptTemplate =========================
    # Message-based template — production standard for Chat Models
    # Uses TUPLES (not HumanMessage objects) so LangChain can inject vars at runtime
    log_info("APPROACH 2: ChatPromptTemplate (Message-based) ← PRODUCTION STANDARD")
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Summarize the person's background and provide two interesting facts."),
        ("human", "{information}"),
    ])
    log_info(f"  Message roles: system, human")
    log_info(f"  Template variables: {chat_template.input_variables}")

    # ========================= THE MODEL =========================
    # ChatOpenAI: Paid cloud model (high intelligence)
    # temperature=0: Deterministic (no randomness) — good for factual tasks
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    log_info(f"  Model: gpt-4o (temperature=0)")

    # Alternative: Free local model (uncomment to use)
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(temperature=0, model="qwen3:1.7b")

    # ========================= THE PARSER =========================
    # StrOutputParser extracts .content from AIMessage → gives you a plain string
    # Without this, you get the full AIMessage object (with metadata, tokens, etc.)
    parser = StrOutputParser()

    # ========================= THE CHAIN (LCEL) =========================
    # Pipe operator | connects: template → model → parser
    # Under the hood: Python's __or__ dunder method (see Glossary #3)
    # C# equivalent: template.Pipe(llm).Pipe(parser) or a fluent builder
    chain = chat_template | llm | parser
    log_info("  Chain built: chat_template | llm | parser")

    # ========================= EXECUTION =========================
    log_header("EXECUTING CHAIN")
    log_info("Sending request to OpenAI...")

    # .invoke() triggers the entire pipeline:
    #   1. LangChain finds {information} in the tuple
    #   2. Injects the string value from our dictionary
    #   3. Creates the actual HumanMessage object
    #   4. Sends to OpenAI API
    #   5. Receives AIMessage, parser extracts .content
    response = chain.invoke(input={"information": information})

    log_success("Response received!")
    print()
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
