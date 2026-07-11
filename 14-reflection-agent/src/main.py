# ─────────────────────────────────────────────────────────────────────────────
# main.py — Reflection Agent (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
# What this does:
#   A LangGraph agent that iteratively improves a tweet by cycling between
#   a "generate" node (writes/revises the tweet) and a "reflect" node
#   (critiques the tweet). The loop continues until 6+ messages accumulate,
#   meaning roughly 3 generate/reflect cycles.
#
# Architecture (Mermaid — renders on GitHub):
#   graph LR;
#     __start__ --> generate;
#     generate --> reflect;
#     reflect -.-> generate;
#     generate -.-> __end__;
#
# ASCII view:
#   ┌──────────┐    ┌─────────┐
#   │ GENERATE │───▶│ REFLECT │
#   │ (writer) │◀───│ (critic)│
#   └──────────┘    └─────────┘
#        │
#        ▼ (after 6+ messages)
#      [END]
#
# How to run:
#   cd C:\Dev\akgit
#   uv run python 14-reflection-agent/src/main.py
#
# Prerequisites:
#   - OPENAI_API_KEY in .env
#   - Dependencies: langchain, langchain-openai, langgraph, python-dotenv, truststore
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()

from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ──────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "reflection-agent"

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

# ─── Reconfigure stdout for emoji/unicode on Windows ─────────────────────────
sys.stdout.reconfigure(encoding="utf-8")


# ─── State Definition ────────────────────────────────────────────────────────
# The state is simply a list of messages that grows with each iteration.
# add_messages is a reducer that appends new messages to the existing list
# (instead of replacing them).
#
# NOTE: This is the manual/old way. The modern equivalent is just:
#   from langgraph.graph import MessagesState
#   class MessageGraph(MessagesState):
#       pass
# MessagesState already has `messages: Annotated[list, add_messages]` built-in.
# It's written out explicitly here for teaching purposes.
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── Node Names ──────────────────────────────────────────────────────────────
REFLECT = "reflect"
GENERATE = "generate"


# ─── Node: Generate ──────────────────────────────────────────────────────────
# Calls the generation chain with all messages so far.
# On the first call: produces an initial tweet draft.
# On subsequent calls: sees critique and produces a revised tweet.
# Returns an AIMessage (the LLM's response).
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


# ─── Node: Reflect ───────────────────────────────────────────────────────────
# Calls the reflection chain to critique the latest tweet draft.
# IMPORTANT: wraps the critique in a HumanMessage so that when the
# generate node sees it next, it treats the critique as user feedback
# (not as its own prior output). This trick makes the conversation flow
# naturally: user request → AI draft → "user" critique → AI revision → ...
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


# ─── Build the Graph ─────────────────────────────────────────────────────────
builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)  # Start with generation


# ─── Conditional Edge: Should we keep iterating? ─────────────────────────────
# Stop condition: if there are more than 6 messages in state, we've done
# enough iterations (≈3 generate/reflect cycles). Otherwise, keep reflecting.
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# ─── Wire the edges ──────────────────────────────────────────────────────────
# After GENERATE → check if we should continue or stop
# path_map explicitly maps the return values of should_continue to node names.
# This helps LangGraph render the full graph correctly (showing both branches
# in Mermaid/ASCII diagrams). Without it, the graph visualization may omit
# the reflect node from the diagram even though execution still works.
builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
# After REFLECT → always go back to GENERATE (the cycle!)
builder.add_edge(REFLECT, GENERATE)

# ─── Compile ─────────────────────────────────────────────────────────────────
graph = builder.compile()

# Print Mermaid diagram (paste into https://mermaid.live to visualize)
print(graph.get_graph().draw_mermaid())
# Print ASCII diagram (renders directly in terminal)
graph.get_graph().print_ascii()

# Print the graph structure (ASCII art)
print("=" * 60)
print("📊 REFLECTION AGENT — GRAPH STRUCTURE")
print("=" * 60)
graph.get_graph().print_ascii()
print()


# ─── Execute ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🐦 REFLECTION AGENT — Tweet Improvement Loop")
    print("=" * 60)

    # The original tweet to improve (example input)
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }

    print("\n📝 Original tweet submitted for improvement...")
    print("-" * 60)

    response = graph.invoke(inputs)

    # Print all messages to see the full generate/reflect conversation
    print("\n" + "=" * 60)
    print("📬 FULL CONVERSATION (Generate ↔ Reflect cycles)")
    print("=" * 60)
    for i, msg in enumerate(response["messages"]):
        role = "🧑 USER" if msg.type == "human" else "🤖 AI"
        print(f"\n{'─' * 60}")
        print(f"Message {i+1} [{role}]:")
        print(f"{'─' * 60}")
        print(msg.content[:500])  # Truncate long messages for readability
        if len(msg.content) > 500:
            print("... (truncated)")

    print("\n" + "=" * 60)
    print("✅ Final refined tweet is the LAST AI message above.")
    print("=" * 60)
