# ─────────────────────────────────────────────────────────────────────────────
# chains.py — LLM chains for the Reflexion Agent (first_responder + revisor)
# ─────────────────────────────────────────────────────────────────────────────
# This file defines two LLM chains that share the same base prompt template
# but with different instructions:
#
# 1. first_responder — Generates the initial ~250 word answer + critique + queries
# 2. revisor — Revises the answer incorporating search results + adds citations
#
# IMPORTANT TECHNIQUE: Both chains use bind_tools() with tool_choice forced.
# This means the LLM MUST return structured output matching our Pydantic schema.
# It's not optional — the LLM has no choice but to fill in every field.
#
# How to run standalone (tests the first_responder chain only):
#   cd C:\Dev\akgit
#   uv run python 15-reflexion-agent/src/chains.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import datetime

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ──────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "reflexion-agent"

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

# ─── Reconfigure stdout for emoji/unicode on Windows ─────────────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ─── LLM Setup ──────────────────────────────────────────────────────────────
# Using o4-mini — needs strong reasoning to produce quality self-critique
llm = ChatOpenAI(model="o4-mini")

# Parsers for extracting structured tool calls from LLM response
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# ─── Shared Prompt Template (Actor Prompt) ───────────────────────────────────
# Both chains use this same template with different {first_instruction}.
# The prompt forces the LLM to:
#   1. Answer the question (instruction varies by chain)
#   2. Self-critique the answer (be severe!)
#   3. Generate search queries to improve the answer
#
# {time} is injected dynamically so the LLM knows the current date
# (important for searching recent events like startup funding)
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


# ─── Chain 1: First Responder ────────────────────────────────────────────────
# Generates the INITIAL answer. Instruction: "Provide a detailed ~250 word answer."
# Uses tool_choice="AnswerQuestion" to force structured output matching the schema.
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# ─── Chain 2: Revisor ────────────────────────────────────────────────────────
# Revises the answer using search results + previous critique.
# Key additions over first_responder:
#   - Must incorporate the critique from the previous step
#   - Must include numerical citations [1], [2], etc.
#   - Must add a References section with URLs
#   - Still limited to ~250 words
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# ─── Standalone Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing first_responder chain (standalone)")
    print("=" * 60)

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
