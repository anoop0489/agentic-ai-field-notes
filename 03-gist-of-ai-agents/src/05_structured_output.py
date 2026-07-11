# =============================================================================
# STRUCTURED OUTPUT: Pydantic + LLM = Typed JSON Responses
# =============================================================================
# Demonstrates how to force an LLM to return strongly-typed objects:
#   - Pydantic BaseModel = C# DTO / Java POJO with DataAnnotations
#   - .with_structured_output(): Forces LLM to return matching JSON
#   - Field(description="..."): Guides the LLM on how to fill each field
#   - Result is a Python object with dot-notation access (not a string)
#
# WHAT IT DOES:
#   Takes raw unstructured text about job postings and forces the LLM
#   to extract and return a strictly typed Python object with company
#   names and job titles.
#
# KEY CONCEPTS:
#   - Pydantic BaseModel: Defines the schema (like a C# DTO)
#   - Field(description=...): Acts as DataAnnotation/prompt for the LLM
#   - .with_structured_output(MyClass): Generic method that enforces JSON schema
#   - Response is an OBJECT, not a string — use dot-notation to access fields
#
# C# ANALOGY:
#   var result = JsonConvert.DeserializeObject<JobSearchResponse>(llmOutput);
#   Except LangChain does this automatically using OpenAI's Function Calling API
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY
#   2. Packages: uv add langchain-core langchain-openai python-dotenv pydantic truststore
#
# USAGE:
#   uv run 03-gist-of-ai-agents/src/05_structured_output.py
# =============================================================================

import os
import sys
from typing import List

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "reference-guides"))
from logger import log_header, log_info, log_success, log_error

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ========================= DEFINE THE SCHEMA (DTO) =========================
# These Pydantic classes define the EXACT structure we want from the LLM.
# The LLM reads the Field descriptions to understand how to map data.
# C# equivalent: public class JobPosting { [Description("...")] public string Title { get; set; } }

class JobPosting(BaseModel):
    """A single job posting extracted from text."""
    title: str = Field(description="The exact title of the job")
    company: str = Field(description="The company that is hiring")

class JobSearchResponse(BaseModel):
    """Container for multiple job postings."""
    postings: List[JobPosting] = Field(description="A list of the job postings found")


def check_prerequisites():
    """Verify required env vars exist before making API calls."""
    if not os.environ.get("OPENAI_API_KEY"):
        log_error("OPENAI_API_KEY not found in .env")
        sys.exit(1)
    log_success("Prerequisites met — OPENAI_API_KEY found")


def main():
    check_prerequisites()

    log_header("STRUCTURED OUTPUT: Pydantic + LLM = Typed Objects")

    # ========================= THE MODEL =========================
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    log_info("Model: gpt-4o (temperature=0 for consistent extraction)")

    # ========================= ENFORCE STRUCTURED OUTPUT =========================
    # .with_structured_output() is like a Generic Method in C#:
    #   var structuredLlm = llm.WithStructuredOutput<JobSearchResponse>();
    # It tells OpenAI to use Function Calling API to guarantee the output
    # matches our Pydantic schema EXACTLY (no free-form text allowed)
    structured_llm = llm.with_structured_output(JobSearchResponse)
    log_info("Structured output enforced: JobSearchResponse schema")
    log_info("  Under the hood: Pydantic class → JSON Schema → OpenAI Function Calling")

    # ========================= THE PROMPT =========================
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract job posting data from the user's text into the requested format."),
        ("human", "{input}"),
    ])
    log_info("Prompt: system (extraction instructions) + human (raw text)")

    # ========================= THE CHAIN =========================
    # prompt → structured_llm (forces JSON output matching our schema)
    chain = prompt | structured_llm
    log_info("Chain: prompt | structured_llm")

    # ========================= EXECUTION =========================
    log_header("EXTRACTING STRUCTURED DATA FROM RAW TEXT")
    raw_text = "TechCorp is hiring a Senior AI Engineer. DataFlow Inc needs an ML Ops Lead."
    log_info(f"Raw input: \"{raw_text}\"")
    log_info("Sending to LLM with JSON schema enforcement...")
    print()

    # The response is NOT a string — it's a fully typed Python object!
    # C# equivalent: var response = JsonConvert.DeserializeObject<JobSearchResponse>(json);
    response = chain.invoke({"input": raw_text})

    # ========================= OUTPUT =========================
    log_header("RESULTS (Typed Object, Not String)")
    log_success(f"Response type: {type(response).__name__} (not str!)")
    print()

    # Access data via dot-notation — exactly like a C# DTO
    for i, posting in enumerate(response.postings, 1):
        log_info(f"  Job {i}:")
        log_info(f"    Company: {posting.company}")
        log_info(f"    Title:   {posting.title}")
        print()

    # Demonstrate it's a real object, not a string
    log_success(f"Direct access: response.postings[0].company = \"{response.postings[0].company}\"")
    log_info("No string parsing needed — the LLM output is already deserialized!")


if __name__ == "__main__":
    main()
