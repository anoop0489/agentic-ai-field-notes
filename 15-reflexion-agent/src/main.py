# ─────────────────────────────────────────────────────────────────────────────
# main.py — Reflexion Agent (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────
# What this does:
#   A LangGraph agent that produces high-quality research articles by:
#   1. Drafting an initial answer with self-critique and search queries
#   2. Executing search queries via Tavily for real-time data
#   3. Revising the answer incorporating search results + citations
#   4. Repeating steps 2-3 for MAX_ITERATIONS cycles
#
# Architecture (from the Reflexion paper — Shinn et al., 2023):
#
#   graph LR;
#     __start__ --> draft;
#     draft --> execute_tools;
#     execute_tools --> revise;
#     revise -.-> execute_tools;  (cycle — if iterations remain)
#     revise -.-> __end__;        (exit — if max iterations reached)
#
# ASCII view:
#   ┌───────┐    ┌───────────────┐    ┌────────┐
#   │ DRAFT │───▶│ EXECUTE_TOOLS │───▶│ REVISE │
#   └───────┘    └───────────────┘    └────┬───┘
#                        ▲                  │
#                        │                  │ (if iterations ≤ MAX)
#                        └──────────────────┘
#                                           │ (if iterations > MAX)
#                                           ▼
#                                        [END]
#
# Key difference from Section 14 (Reflection Agent):
#   - Section 14: No tools, pure LLM-to-LLM feedback
#   - Section 15: Uses Tavily search to ground revisions in real data
#   - Section 14: Stop condition = message count
#   - Section 15: Stop condition = number of tool executions (ToolMessages)
#
# How to run:
#   cd C:\Dev\akgit
#   uv run python 15-reflexion-agent/src/main.py
#
# Prerequisites:
#   - OPENAI_API_KEY in .env
#   - TAVILY_API_KEY in .env
#   - Dependencies: langchain, langchain-openai, langchain-tavily, langgraph,
#                   python-dotenv, truststore
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()

from typing import Literal

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ──────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "reflexion-agent"

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

# ─── Reconfigure stdout for emoji/unicode on Windows ─────────────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ─── Configuration ───────────────────────────────────────────────────────────
# How many revise→search cycles to allow before stopping.
# Each cycle adds one ToolMessage, so we count those to track iterations.
MAX_ITERATIONS = 2


# ─── Node: Draft ─────────────────────────────────────────────────────────────
# The FIRST node in the graph. Generates the initial answer with:
#   - A ~250 word response
#   - Self-critique (what's missing, what's superfluous)
#   - 1-3 search queries to improve the answer
# Output: AIMessage with tool_calls (structured output via AnswerQuestion schema)
def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ─── Node: Revise ────────────────────────────────────────────────────────────
# Takes the previous answer + search results and produces a BETTER answer.
# Uses the critique from the previous step to know what to improve.
# Adds citations from the search results (References section).
# Output: AIMessage with tool_calls (structured output via ReviseAnswer schema)
def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ─── Conditional Edge: Event Loop ────────────────────────────────────────────
# Decides whether to continue searching or stop.
# Stop condition: count the number of ToolMessages in the state.
# Each execute_tools call adds one ToolMessage, so this counts iterations.
# If we've done more than MAX_ITERATIONS searches → stop.
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# ─── Build the Graph ─────────────────────────────────────────────────────────
builder = StateGraph(MessagesState)

# Register nodes
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)  # ToolNode from tool_executor.py
builder.add_node("revise", revise_node)

# Wire edges
builder.add_edge(START, "draft")              # Start → draft the initial answer
builder.add_edge("draft", "execute_tools")    # Draft → search for data
builder.add_edge("execute_tools", "revise")   # Search results → revise answer

# Conditional: after revise, either search again or stop
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

# ─── Compile ─────────────────────────────────────────────────────────────────
graph = builder.compile()

# Print Mermaid diagram (paste into https://mermaid.live to visualize)
print(graph.get_graph().draw_mermaid())
# Print ASCII diagram (renders directly in terminal)
graph.get_graph().print_ascii()

# Print graph structure
print("=" * 60)
print("📊 REFLEXION AGENT — GRAPH STRUCTURE")
print("=" * 60)
print(graph.get_graph().draw_mermaid())
print()
graph.get_graph().print_ascii()
print()


# ─── Execute ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 REFLEXION AGENT — Research with Self-Improvement")
    print("=" * 60)

    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                }
            ]
        }
    )

    # Extract the final answer from the last AIMessage with tool calls
    print("\n" + "=" * 60)
    print("📝 FINAL ANSWER")
    print("=" * 60)
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        final = last_message.tool_calls[0]["args"]
        print(f"\n{final['answer']}")
        if "references" in final:
            print("\n📚 References:")
            for ref in final["references"]:
                print(f"  {ref}")
    else:
        print(last_message.content)

    # Print iteration summary
    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    tool_msgs = sum(isinstance(m, ToolMessage) for m in res["messages"])
    ai_msgs = sum(isinstance(m, AIMessage) for m in res["messages"])
    print(f"  Total messages: {len(res['messages'])}")
    print(f"  AI responses: {ai_msgs}")
    print(f"  Tool executions: {tool_msgs}")
    print(f"  Revision cycles: {tool_msgs}")
    print("=" * 60)
