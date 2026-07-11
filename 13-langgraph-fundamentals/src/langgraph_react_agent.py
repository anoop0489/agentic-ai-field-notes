"""
ReAct Agent with Function Calling — LangGraph Implementation (Chapters 90-97)

This is a first real LangGraph project. It implements the
classic ReAct (Reason + Act) agent loop using LangGraph's graph structure.

WHAT THIS PROGRAM DOES:
  Takes a user question → LLM reasons about what tools to use → calls tools →
  feeds results back → LLM reasons again → repeats until it has the final answer.

  Example: "What is the temperature in Tokyo? List it and then triple it"
    1. LLM decides: I need to search for Tokyo's temperature → calls TavilySearch
    2. Gets result: "72°F" → feeds back to LLM
    3. LLM decides: I need to triple 72 → calls triple(72)
    4. Gets result: 216.0 → feeds back to LLM
    5. LLM decides: I have everything → returns final answer to user

THE REACT LOOP (this is the entire pattern):
  ┌──────────────┐     ┌───────────────┐     ┌────────────┐
  │ agent_reason │────▶│ has tool_calls?│────▶│    tools    │
  │ (LLM thinks) │     │  YES → tools   │     │ (execute)  │
  └──────────────┘     │  NO  → END     │     └─────┬──────┘
        ▲              └───────────────┘            │
        │                                           │
        └───────────────────────────────────────────┘
                    (CYCLE: back to agent)

WHY THIS MATTERS:
  - This is the SIMPLEST possible LangGraph agent
  - It demonstrates the exact pattern that 90% of agents follow
  - The cycle (tools → agent → tools → agent) is what makes it an AGENT
  - Without the cycle, it's just a chain (one-directional)

TOOLS USED:
  1. TavilySearch — real-time web search (gets live data like weather)
  2. triple — a simple math tool (demonstrates custom tool creation)

PREREQUISITES:
  - OPENAI_API_KEY in .env
  - TAVILY_API_KEY in .env (get free at https://app.tavily.com)

Run: uv run python langgraph_react_agent.py
"""

import sys

# Fix Windows terminal encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Corporate proxy / SSL fix — must be before any network imports
import truststore
truststore.inject_into_ssl()

import os
from dotenv import load_dotenv
load_dotenv()

# Route LangSmith traces to a dedicated project (must be set BEFORE langchain imports)
os.environ["LANGSMITH_PROJECT"] = "langgraph-fundamentals"

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode


# =====================================================================
# TOOLS: Define what the agent can DO in the real world.
#
# Each tool is a Python function with:
#   - Type hints (become the schema the LLM sees)
#   - Docstring (becomes the description the LLM reads to decide when to use it)
#
# The LLM does NOT execute these. It only REQUESTS them.
# The ToolNode executes them and returns results.
# =====================================================================

# Tool 1: Web search — gets live information (weather, news, facts)
# TavilySearch is a pre-built LangChain tool. max_results=1 keeps it fast.
tavily_search = TavilySearch(max_results=1)

# Tool 2: Custom math tool — demonstrates how to write your own
@tool
def triple(num: float) -> float:
    """Triple a number. Use this when you need to multiply a number by 3.

    Args:
        num: The number to triple.

    Returns:
        The input number multiplied by 3.
    """
    return float(num) * 3


# Collect all tools into a list — this gets passed to both the LLM and the ToolNode
tools = [tavily_search, triple]


# =====================================================================
# MODEL: Initialize the LLM and bind tools to it.
#
# bind_tools() does NOT execute tools. It tells the LLM:
#   "Here are functions you can REQUEST. Include tool_calls in your
#    response if you want me to run them."
#
# The LLM then returns AIMessage with tool_calls=[{name, args, id}]
# instead of just content="..." when it wants to use a tool.
# =====================================================================
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# =====================================================================
# NODES: The two processing stations in our ReAct loop.
#
# Node 1: agent_reason — LLM thinks about what to do
# Node 2: tool_node — executes whatever tools the LLM requested
# =====================================================================

# System prompt: tells the LLM its identity and capabilities
SYSTEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."


def agent_reason(state: MessagesState):
    """The REASONING node — LLM decides what to do next.

    Receives: current state (all messages so far)
    Does: calls LLM with system prompt + conversation history
    Returns: AIMessage that either:
      - Has tool_calls (LLM wants to use a tool) → will route to tools node
      - Has content only (LLM has the final answer) → will route to END
    """
    # Prepend system message to give the LLM its identity/rules
    # (system message is NOT stored in state — it's injected fresh each time)
    response = llm_with_tools.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}] + state["messages"]
    )
    return {"messages": [response]}


# ToolNode is a prebuilt node that:
#   1. Reads tool_calls from the last AIMessage
#   2. Looks up the matching Python function by name
#   3. Calls it with the provided arguments
#   4. Wraps each result in a ToolMessage (with tool_call_id for correlation)
#   5. Returns all ToolMessages as a state update
tool_node = ToolNode(tools)


# =====================================================================
# CONDITIONAL EDGE: The routing logic that makes this an AGENT.
#
# After the LLM thinks (agent_reason), we check:
#   - Did it request tool calls? → Route to "act" (tools node)
#   - Did it give a final answer? → Route to END
#
# This is the DECISION POINT. Without it, there's no agent loop.
# =====================================================================
def should_continue(state: MessagesState) -> str:
    """Inspect the last message and decide: more tools or done?"""
    last_message = state["messages"][-1]

    # If the LLM's response has tool_calls → it wants to act
    if last_message.tool_calls:
        return "act"

    # Otherwise it gave a final text answer → we're done
    return END


# =====================================================================
# BUILD THE GRAPH: Wire nodes and edges into the ReAct loop.
#
# The graph structure IS the agent architecture:
#   START → agent_reason → (conditional) → act → agent_reason → ... → END
#                                            ↑___________________________|
#                                                    (THE CYCLE)
# =====================================================================

# Node name constants (avoids typos in string references)
AGENT_REASON = "agent_reason"
ACT = "act"

# Create the graph with MessagesState (just needs a messages list)
graph = StateGraph(MessagesState)

# Register nodes
graph.add_node(AGENT_REASON, agent_reason)  # LLM thinking node
graph.add_node(ACT, tool_node)               # Tool execution node

# Wire edges
graph.set_entry_point(AGENT_REASON)          # START → agent_reason

# After agent_reason: check if we need tools or are done
graph.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {END: END, ACT: ACT}  # Explicit mapping: return value → node name
)

# After tools execute: ALWAYS go back to agent_reason (THE CYCLE)
# This is what makes it an agent — the LLM gets to think again after seeing tool results
graph.add_edge(ACT, AGENT_REASON)

# Compile: validate the graph, freeze it, make it runnable
app = graph.compile()


# =====================================================================
# RUN THE AGENT
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 ReAct Agent with Function Calling (LangGraph)")
    print("=" * 60)

    # The question requires MULTIPLE tool calls in sequence:
    #   1. Search for Tokyo temperature (TavilySearch)
    #   2. Triple that number (triple tool)
    # The agent must REASON about the order and chain them correctly.
    question = "What is the temperature in Tokyo? List it and then triple it"

    print(f"\n👤 Question: {question}")
    print("-" * 60)

    # invoke() runs the full graph to completion:
    #   agent_reason → (tools?) → act → agent_reason → (tools?) → ... → END
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # The final message is the LLM's answer (no more tool_calls)
    final_answer = result["messages"][-1].content
    print(f"\n🤖 Answer: {final_answer}")

    # Show the full execution trace (what happened internally)
    print(f"\n{'─' * 60}")
    print(f"📋 Full execution trace ({len(result['messages'])} messages):")
    print(f"{'─' * 60}")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", [])
        content = getattr(msg, "content", "")

        if tool_calls:
            for tc in tool_calls:
                print(f"  [{i}] {role}: → calls {tc['name']}({tc['args']})")
        elif content:
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{i}] {role}: {preview}")
        else:
            print(f"  [{i}] {role}: (empty)")

    print(f"\n{'=' * 60}")
    print("✅ ReAct loop complete.")
    print("=" * 60)
