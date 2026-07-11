"""
Simplest LangGraph agent — using .invoke() (no streaming).
One input → graph runs → one output. That's it.

Run: uv run python langgraph_simple_invoke.py
"""

import os
import sys
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import truststore
truststore.inject_into_ssl()

import os
from dotenv import load_dotenv
load_dotenv()

# Route LangSmith traces to a dedicated project (must be set BEFORE langchain imports)
os.environ["LANGSMITH_PROJECT"] = "langgraph-fundamentals"

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# --- State: just use MessagesState (has messages built-in) ---
# No extra fields needed for this simple example.


# --- Model ---
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)


# --- One simple tool ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {"Tokyo": "72°F sunny", "London": "58°F cloudy", "Paris": "65°F rainy"}
    return weather.get(city, f"No weather data for {city}")


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)


# --- Nodes ---
def agent(state: MessagesState):
    """Call the LLM."""
    messages = [SystemMessage(content="You are a helpful assistant. Be concise.")] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# --- Build graph ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# CONFIG (RunnableConfig): The full configuration dict passed to every LangGraph call.
#
# It has COMPARTMENTS — each for a different purpose:
#
#   config = {
#       "configurable": {          ← YOUR parameters (things that change behavior)
#           "thread_id": "abc",        - Which conversation (required for checkpointing)
#           "model": "gpt-4o",         - Which model (if using configurable models)
#           "checkpoint_id": "...",     - Resume from specific checkpoint (time travel)
#       },
#       "tags": ["prod"],          ← Labels for tracing (show up in LangSmith)
#       "metadata": {"team": "X"},← Extra info attached to traces
#       "recursion_limit": 25,     ← Max graph loops before stopping
#       "callbacks": [...],        ← Hook functions for events
#   }
#
# WHY "configurable" as a nested dict (not flat)?
#   - LangGraph needs to separate YOUR custom keys from FRAMEWORK keys.
#   - "tags", "metadata", "callbacks" are framework features.
#   - "thread_id", "model" are YOUR parameters that change per call.
#   - The nesting prevents name collisions between the two.
#   - Think of it as: config is the envelope, configurable is YOUR letter inside.
#
# WHY IS thread_id NEEDED?
#   - LangGraph saves checkpoints (snapshots) after every node runs.
#   - But which conversation does each checkpoint belong to? → thread_id tells it.
#   - Same thread_id = same conversation (has history from previous turns).
#   - New thread_id = fresh conversation (no memory of past).
#
# WHERE IS CONFIG USED?
#   - app.invoke(input, config)    → Run graph, save checkpoints under this thread
#   - app.get_state(config)        → Read the latest checkpoint for this thread
#   - app.invoke(None, config)     → Resume a paused graph from this thread's checkpoint
#   - app.get_state_history(config)→ Browse all checkpoints for this thread
#
# C# ANALOGY:
#   - config ≈ HttpContext (the full request context object)
#   - configurable ≈ HttpContext.Items or RouteData (your custom per-request params)
#   - thread_id ≈ sessionId in ASP.NET (scopes the conversation)
#   - tags/metadata ≈ Activity.Tags in System.Diagnostics (tracing/telemetry)
#   - Checkpointer ≈ Session State Provider (in-memory, SQL, Redis)
# ---------------------------------------------------------------------------
thread_id = str(uuid.uuid4())  # Generate a unique ID for this conversation
config = {"configurable": {"thread_id": thread_id}}
# ^ Minimum required config for a graph with a checkpointer.
#   "configurable" = the compartment for YOUR params.
#   "thread_id" = which conversation's checkpoints to save/load.

print("=" * 50)
print("SIMPLE .invoke() EXAMPLE")
print("=" * 50)
print(f"Thread ID: {thread_id[:8]}... (this scopes all checkpoints below)")

# ─────────────────────────────────────────────────
# TURN 1: User asks about weather
# ─────────────────────────────────────────────────
print("\n--- Turn 1 ---")
print("Input:  'What's the weather in Tokyo?'")

result = app.invoke(
    {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
    config=config,
)

# result is the FINAL state after the entire graph ran to END
# result["messages"] has ALL messages: user → AI(tool_call) → tool_result → AI(final)
final_answer = result["messages"][-1].content
print(f"Output: '{final_answer}'")
print(f"Total messages in state: {len(result['messages'])}")

# Let's see what happened inside:
print("\n  What happened internally:")
for i, msg in enumerate(result["messages"]):
    role = msg.__class__.__name__
    content = getattr(msg, "content", "")[:60]
    tool_calls = getattr(msg, "tool_calls", [])
    if tool_calls:
        print(f"  [{i}] {role}: tool_calls={[tc['name'] for tc in tool_calls]}")
    else:
        print(f"  [{i}] {role}: {content}")

# ─────────────────────────────────────────────────
# TURN 2: Follow-up (same thread_id = remembers Turn 1)
# ─────────────────────────────────────────────────
print("\n--- Turn 2 ---")
print("Input:  'And in London?'")

result = app.invoke(
    {"messages": [HumanMessage(content="And in London?")]},
    config=config,  # Same thread_id → has Turn 1 history
)

final_answer = result["messages"][-1].content
print(f"Output: '{final_answer}'")
print(f"Total messages in state: {len(result['messages'])}")

# ─────────────────────────────────────────────────
# TURN 3: No tool needed (model answers directly)
# ─────────────────────────────────────────────────
print("\n--- Turn 3 ---")
print("Input:  'Thanks!'")

result = app.invoke(
    {"messages": [HumanMessage(content="Thanks!")]},
    config=config,
)

final_answer = result["messages"][-1].content
print(f"Output: '{final_answer}'")
print(f"Total messages in state: {len(result['messages'])}")

# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-IN-THE-LOOP (HIL): Pause before tool execution, inspect, then resume.
#
# Why? In production, you may want a human to APPROVE a tool call before it runs.
# Example: booking a flight, sending an email, charging a credit card.
#
# How it works:
#   1. Compile graph with interrupt_before=["tools"]
#   2. invoke() runs: START → agent → PAUSES (before tools node)
#   3. You inspect what the LLM wants to do (get_state)
#   4. You approve → invoke(None) to RESUME from where it paused
#   5. Graph continues: tools → agent → END
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("HUMAN-IN-THE-LOOP EXAMPLE")
print("=" * 50)

# Compile a NEW version of the same graph, but with interrupt_before
app_with_hil = graph.compile(
    checkpointer=MemorySaver(),       # Fresh checkpointer for this demo
    interrupt_before=["tools"],        # ← THIS is the magic line
)

# New thread_id = new conversation (no leftover history from above)
# WHY a new thread? Because get_state(config) looks up checkpoints by thread_id.
# If we reused the old one, it would find Turn 1-3 history from above.
hil_thread = str(uuid.uuid4())
hil_config = {
    "configurable": {"thread_id": hil_thread},
    # Tags + metadata so you can FIND these traces in LangSmith:
    # Filter by tag "hil-demo" to see both the paused and resumed runs.
    "tags": ["hil-demo", f"session:{hil_thread[:8]}"],
    "metadata": {"demo": "human-in-the-loop", "session_id": hil_thread},
    "run_name": "hil_weather_check",  # Shows as the trace name in LangSmith
}
print(f"HIL Thread ID: {hil_thread[:8]}... (separate conversation)")
# NOTE: In LangSmith you'll see TWO traces with tag "hil-demo":
#   1. First invoke() — runs agent, then PAUSES (no tool execution)
#   2. Second invoke(None) — RESUMES, runs tools, then agent gives final answer

# Step 1: User asks something that WILL trigger a tool call
print("\n--- Step 1: Invoke (graph will PAUSE before tools) ---")
print("Input:  'What's the weather in Paris?'")

result = app_with_hil.invoke(
    {"messages": [HumanMessage(content="What's the weather in Paris?")]},
    config=hil_config,
)

# The graph PAUSED — result is the state at the pause point
# The last message is an AIMessage WITH tool_calls (not yet executed)
print(f"Graph paused. Last message type: {result['messages'][-1].__class__.__name__}")

# Step 2: Inspect what the LLM wants to do
# get_state(config) → Opens the filing cabinet at hil_config's thread_id,
#                      reads the latest checkpoint (the paused state).
print("\n--- Step 2: Inspect pending tool call (human reviews) ---")
state = app_with_hil.get_state(hil_config)
# state.values  → The full state dict at the pause point
# state.next    → Which node(s) would run next if we resume
pending_tool_call = state.values["messages"][-1].tool_calls[0]
print(f"  Tool the LLM wants to call: {pending_tool_call['name']}")
print(f"  Arguments it wants to pass: {pending_tool_call['args']}")
print(f"  Next node (paused before):  {state.next}")  # ('tools',)

# Step 3: Human decides — approve or reject
print("\n--- Step 3: Human APPROVES → resume execution ---")

# To resume: invoke with None (no new input, just continue from checkpoint)
# invoke(None, config) means: "Don't add any new user message.
#   Just open the checkpoint for this thread_id and pick up where you left off."
# The graph continues: tools node runs → back to agent → agent gives final answer → END
result = app_with_hil.invoke(None, config=hil_config)

# Now the graph finished: tools ran → agent gave final answer
final_answer = result["messages"][-1].content
print(f"  Final answer: '{final_answer}'")

# Show the full message trace
print("\n  Full message trace:")
for i, msg in enumerate(result["messages"]):
    role = msg.__class__.__name__
    content = getattr(msg, "content", "")[:60]
    tool_calls = getattr(msg, "tool_calls", [])
    if tool_calls:
        print(f"  [{i}] {role}: tool_calls={[tc['name'] for tc in tool_calls]}")
    else:
        print(f"  [{i}] {role}: {content}")

print("\n" + "=" * 50)
print("DONE. Three turns + Human-in-the-Loop, all with .invoke().")
print("=" * 50)
