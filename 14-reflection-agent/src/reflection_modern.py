# ─────────────────────────────────────────────────────────────────────────────
# reflection_modern.py — Reflection Agent using MessagesState (Modern Way)
# ─────────────────────────────────────────────────────────────────────────────
# What this does:
#   Same reflection agent as main.py but using the modern MessagesState
#   instead of manually defining TypedDict + add_messages reducer.
#   This is how you'd write it in production with LangGraph today.
#
# Difference from main.py:
#   main.py (explicit way):
#     class MessageGraph(TypedDict):
#         messages: Annotated[list[BaseMessage], add_messages]
#
#   This file (modern way):
#     from langgraph.graph import MessagesState
#     # That's it. MessagesState already has messages + add_messages built-in.
#
# How to run:
#   cd C:\Dev\akgit
#   uv run python 14-reflection-agent/src/reflection_modern.py
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

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ──────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "reflection-agent"

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState

from chains import generate_chain, reflect_chain

# ─── Reconfigure stdout for emoji/unicode on Windows ─────────────────────────
sys.stdout.reconfigure(encoding="utf-8")


# ─── State Definition (Modern Way) ──────────────────────────────────────────
# MessagesState already provides:
#   messages: Annotated[list[AnyMessage], add_messages]
# No need to define it ourselves. Just use it directly.
# If you need extra fields, subclass it:
#   class MyState(MessagesState):
#       iteration_count: int = 0


# ─── Node Names ──────────────────────────────────────────────────────────────
REFLECT = "reflect"
GENERATE = "generate"


# ─── Node: Generate ──────────────────────────────────────────────────────────
def generation_node(state: MessagesState):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


# ─── Node: Reflect ───────────────────────────────────────────────────────────
# Wraps critique as HumanMessage so the generator treats it as user feedback.
def reflection_node(state: MessagesState):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


# ─── Build the Graph ─────────────────────────────────────────────────────────
# Notice: StateGraph(MessagesState) — no custom class needed!
builder = StateGraph(MessagesState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# ─── Conditional Edge: Should we keep iterating? ─────────────────────────────
def should_continue(state: MessagesState):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# ─── Wire the edges ──────────────────────────────────────────────────────────
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

# ─── Compile ─────────────────────────────────────────────────────────────────
graph = builder.compile()

print("=" * 60)
print("📊 REFLECTION AGENT (Modern MessagesState)")
print("=" * 60)
graph.get_graph().print_ascii()
print()


# ─── Execute ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🐦 REFLECTION AGENT — Modern MessagesState Version")
    print("=" * 60)

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

    print("\n" + "=" * 60)
    print("📬 FULL CONVERSATION (Generate ↔ Reflect cycles)")
    print("=" * 60)
    for i, msg in enumerate(response["messages"]):
        role = "🧑 USER" if msg.type == "human" else "🤖 AI"
        print(f"\n{'─' * 60}")
        print(f"Message {i+1} [{role}]:")
        print(f"{'─' * 60}")
        print(msg.content[:500])
        if len(msg.content) > 500:
            print("... (truncated)")

    print("\n" + "=" * 60)
    print("✅ Final refined tweet is the LAST AI message above.")
    print("=" * 60)
