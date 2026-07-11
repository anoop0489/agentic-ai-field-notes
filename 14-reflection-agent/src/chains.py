# ─────────────────────────────────────────────────────────────────────────────
# chains.py — Prompt chains for the Reflection Agent
# ─────────────────────────────────────────────────────────────────────────────
# This file defines two LLM chains used by the reflection agent graph:
#
# 1. generate_chain — Takes tweet + conversation history, produces a revised tweet
# 2. reflect_chain — Takes tweet draft, produces critique and recommendations
#
# The two chains form a feedback loop: generate → reflect → generate → ...
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# ─── Reflection Prompt ───────────────────────────────────────────────────────
# This chain acts as a "critic" — a viral Twitter influencer grading the tweet.
# It receives all messages so far (original request + any prior drafts) and
# produces detailed feedback: length, virality, style, hooks, CTAs, etc.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ─── Generation Prompt ───────────────────────────────────────────────────────
# This chain acts as the "writer" — it generates or revises tweets.
# On the first call, it writes the best tweet it can.
# On subsequent calls, it sees the critique (fed back as a HumanMessage)
# and revises its previous attempt accordingly.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# ─── LLM and Chain Composition ───────────────────────────────────────────────
# Using ChatOpenAI (defaults to gpt-4o-mini via OPENAI_API_KEY in .env)
# Chain = prompt | llm  (LCEL pipe operator)
llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
