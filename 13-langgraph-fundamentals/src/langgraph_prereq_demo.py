"""
LangGraph Prerequisites — All 14 Topics in One Demo
====================================================
This script demonstrates every prerequisite concept needed before starting LangGraph.
Run it to see each concept in action with printed output.

Topics Covered:
 1. State & TypedDict
 2. Reducers (add_messages)
 3. Nodes (plain Python functions)
 4. Edges (add_edge — sequential flow)
 5. init_chat_model & bind_tools
 6. with_config (runtime model switching)
 7. System Instructions & Prompt Templates
 8. Conditional Edges (route_tools)
 9. Tools & ToolNode
 10. Compile (.compile())
 11. Memory & Checkpointers (MemorySaver)
 12. Configuration, Threads & Time Travel
 13. Streaming (stream_mode)
 14. Human-in-the-Loop & Advanced Tool Architecture

Requirements:
    pip install langchain langchain-openai langgraph python-dotenv

Set your OPENAI_API_KEY in a .env file or environment variable.
"""

import os
import uuid
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

# Route LangSmith traces to a dedicated project (must be set BEFORE langchain imports)
os.environ["LANGSMITH_PROJECT"] = "langgraph-fundamentals"

# ============================================================================
# TOPIC 1: State & TypedDict
# ============================================================================
# State is the structured data contract that holds ALL data flowing through
# your graph. It's a TypedDict — like a C# record/POCO but accessed as a dict.

from langgraph.graph.message import add_messages

print("=" * 70)
print("TOPIC 1: State & TypedDict")
print("=" * 70)


class AgentState(TypedDict):
    """The blueprint for ALL data in our graph.

    Every node receives this, every node returns partial updates to it.
    Think of it as the 'HttpContext' that flows through the pipeline.
    """
    messages: Annotated[list, add_messages]  # Topic 2: reducer attached
    user_name: str                            # Simple field (no reducer = replace)
    preferred_language: str                   # Will be set by a tool via Command


# You access fields with dict syntax: state["messages"], state["user_name"]
sample_state: AgentState = {
    "messages": [],
    "user_name": "Anoop",
    "preferred_language": "English",
}
print(f"Initial state: user_name={sample_state['user_name']}, "
      f"messages count={len(sample_state['messages'])}")
print()

# ============================================================================
# TOPIC 2: Reducers
# ============================================================================
# A reducer tells LangGraph HOW to merge a node's return into existing state.
# Without add_messages, returning {"messages": [new_msg]} REPLACES the list.
# With add_messages, it APPENDS (and deduplicates by ID).

print("=" * 70)
print("TOPIC 2: Reducers (add_messages)")
print("=" * 70)
print("""
  messages: Annotated[list, add_messages]
                       ^          ^
                    the type    the reducer function

  - add_messages → appends new messages (like List.AddRange())
  - No annotation → replaces entirely (like property = newValue)
  - Also handles deduplication: same message ID → update in place
  - Supports RemoveMessage for deletion
""")

# ============================================================================
# TOPIC 3: Nodes — The processing functions
# ============================================================================
print("=" * 70)
print("TOPIC 3: Nodes (Processing Functions)")
print("=" * 70)
print("""
  Nodes are plain Python functions that:
    1. Receive the full state as input
    2. Do some work (call LLM, run code, fetch data)
    3. Return a PARTIAL dict — only the fields they want to update

  Three archetypes:
    - Model Node: calls the LLM, returns AIMessage
    - Tool Node: executes tools, returns ToolMessages
    - Custom Node: any business logic
""")

# ============================================================================
# TOPIC 4: Edges — Sequential flow
# ============================================================================
print("=" * 70)
print("TOPIC 4: Edges (add_edge — Sequential Flow)")
print("=" * 70)
print("""
  graph.add_edge(START, "model")   → Start here
  graph.add_edge("tools", "model") → After tools, go back to model

  Edges define the FIXED paths. For dynamic routing, use conditional edges.
""")

# ============================================================================
# TOPIC 5: init_chat_model & bind_tools
# ============================================================================
print("=" * 70)
print("TOPIC 5: init_chat_model & bind_tools")
print("=" * 70)

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# init_chat_model: provider-agnostic factory. Switch models by changing a string.
model = init_chat_model(
    "openai:gpt-4o-mini",  # format: "provider:model_name"
    temperature=0,
    max_retries=3,
)
print(f"Model created: {model.__class__.__name__}")


# Define tools with @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when asked about weather."""
    # In production, this would call a real API
    weather_data = {"Boston": "72°F sunny", "London": "58°F cloudy", "Tokyo": "81°F humid"}
    return weather_data.get(city, f"No data for {city}")


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone. Use this for time-related queries."""
    return f"It is 2:30 PM in {timezone}"


tools = [get_weather, get_time]

# bind_tools: attach tool schemas to the model so it CAN request them
model_with_tools = model.bind_tools(tools)
print(f"Tools bound: {[t.name for t in tools]}")
print(f"The model now knows it can call: get_weather(city), get_time(timezone)")
print()

# ============================================================================
# TOPIC 6: with_config — Runtime model/config switching
# ============================================================================
print("=" * 70)
print("TOPIC 6: with_config (Runtime Configuration)")
print("=" * 70)

# with_config binds default config to a runnable (creates a new wrapped object)
tagged_model = model.with_config(
    tags=["production", "weather-agent"],
    metadata={"team": "AI", "version": "2.0"},
    run_name="weather_model",
)
print("Created tagged_model with default tags and metadata")
print("These propagate to ALL sub-calls without passing them every time")

# Runtime model switching via configurable
configurable_model = init_chat_model(temperature=0)  # No model specified
print("\nConfigurable model created — pick model at runtime via config:")
print('  config={"configurable": {"model": "gpt-4o"}}')
print('  config={"configurable": {"model": "claude-sonnet-4-6"}}')
print()

# ============================================================================
# TOPIC 7: System Instructions & Prompt Templates
# ============================================================================
print("=" * 70)
print("TOPIC 7: System Instructions & Prompt Templates")
print("=" * 70)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System message defines the agent's identity and rules
SYSTEM_PROMPT = """You are a helpful assistant named WeatherBot.
You can check weather and time. Be concise.
Always greet the user by name if known."""

# ChatPromptTemplate with dynamic variables
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are {persona}. The user's name is {user_name}. Respond in {language}."),
    MessagesPlaceholder("messages"),  # Existing conversation injected here
])

print(f"System prompt defined ({len(SYSTEM_PROMPT)} chars)")
print("ChatPromptTemplate supports dynamic variables: {persona}, {user_name}, {language}")
print("MessagesPlaceholder('messages') → injects state['messages'] into the template")
print()

# ============================================================================
# TOPIC 8: Conditional Edges — Dynamic routing
# ============================================================================
print("=" * 70)
print("TOPIC 8: Conditional Edges (Dynamic Routing)")
print("=" * 70)


def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    """Routing function: inspects state, returns next node name.

    This is the BRAIN of the agent loop:
    - If model returned tool_calls → go execute them
    - If model returned a final answer → END
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


print("""
  Conditional edge flow:
    model_node → [route_after_model] → "tools" (if tool_calls exist)
                                      → END    (if final answer)

  The routing function:
    1. Receives current state (read-only)
    2. Returns a string: name of the next node
    3. Does NOT modify state
""")

# ============================================================================
# TOPIC 9: Tools & ToolNode
# ============================================================================
print("=" * 70)
print("TOPIC 9: Tools & ToolNode (Prebuilt Executor)")
print("=" * 70)

from langgraph.prebuilt import ToolNode

# ToolNode does all the boilerplate:
#   1. Reads tool_calls from last AIMessage
#   2. Finds matching tool by name
#   3. Executes with provided args
#   4. Wraps result in ToolMessage with correct tool_call_id
#   5. Handles errors gracefully
tool_node = ToolNode(tools)
print(f"ToolNode created with {len(tools)} tools: {[t.name for t in tools]}")
print("It handles: name lookup, execution, ToolMessage creation, error handling")
print()

# ============================================================================
# TOPIC 10: Compile — Build the graph
# ============================================================================
print("=" * 70)
print("TOPIC 10: Compile (.compile())")
print("=" * 70)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Build the complete agent graph
graph = StateGraph(AgentState)

# Model node: calls LLM with system prompt + conversation history
def model_node(state: AgentState):
    """The model node — calls the LLM with tools bound."""
    # Prepend system message (NOT stored in state — fresh each call)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Register nodes
graph.add_node("model", model_node)
graph.add_node("tools", tool_node)

# Wire edges
graph.add_edge(START, "model")                              # Start → model
graph.add_conditional_edges("model", route_after_model)     # model → tools or END
graph.add_edge("tools", "model")                            # tools → back to model

# Compile with checkpointer (Topic 11)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

print("Graph compiled successfully!")
print("  Nodes: model, tools")
print("  Edges: START→model, model→[conditional], tools→model")
print("  Checkpointer: MemorySaver (in-memory, dev only)")
print()
print("What .compile() does:")
print("  1. Validates all edges point to existing nodes")
print("  2. Freezes the graph (no more add_node)")
print("  3. Attaches checkpointer for persistence")
print("  4. Returns a runnable with .invoke(), .stream(), .batch()")
print()

# ============================================================================
# TOPIC 11: Memory & Checkpointers
# ============================================================================
print("=" * 70)
print("TOPIC 11: Memory & Checkpointers")
print("=" * 70)
print("""
  Checkpointer saves state AFTER each node execution.
  This enables:
    - Resume after crashes (pick up where you left off)
    - Multi-turn conversations (state persists between requests)
    - Human-in-the-loop (pause, inspect, resume)
    - Time travel (browse history, fork from past states)

  Production checkpointers:
    MemorySaver   → dev/testing (lost on restart)
    PostgresSaver → production (durable, scalable)
    SqliteSaver   → single-machine production
""")

# ============================================================================
# TOPIC 12: Configuration, Threads & Time Travel
# ============================================================================
print("=" * 70)
print("TOPIC 12: Configuration, Threads & Time Travel")
print("=" * 70)

# thread_id: YOU generate it. It scopes the conversation.
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print(f"Generated thread_id: {thread_id}")
print(f"Config: {config}")
print()

# First turn
print("--- Turn 1: Asking about weather ---")
result = app.invoke(
    {
        "messages": [HumanMessage(content="What's the weather in Boston?")],
        "user_name": "Anoop",
        "preferred_language": "English",
    },
    config=config,
)
print(f"Response: {result['messages'][-1].content}")
print()

# Second turn — SAME thread_id, so it has the previous messages
print("--- Turn 2: Follow-up (same thread) ---")
result = app.invoke(
    {"messages": [HumanMessage(content="And what time is it there?")]},
    config=config,  # Same thread_id → conversation continues
)
print(f"Response: {result['messages'][-1].content}")
print()

# Time travel: inspect state history
print("--- Time Travel: Browsing checkpoints ---")
for i, state in enumerate(app.get_state_history(config)):
    msg_count = len(state.values.get("messages", []))
    next_node = state.next
    print(f"  Checkpoint {i}: {msg_count} messages, next={next_node}")
print()

# ============================================================================
# TOPIC 13: Streaming
# ============================================================================
print("=" * 70)
print("TOPIC 13: Streaming")
print("=" * 70)

# New thread for streaming demo
stream_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# stream_mode="updates" — see which node produced what
print("--- Streaming with mode='updates' (node-level progress) ---")
for chunk in app.stream(
    {
        "messages": [HumanMessage(content="Weather in Tokyo?")],
        "user_name": "Anoop",
        "preferred_language": "English",
    },
    config=stream_config,
    stream_mode="updates",
):
    for node_name, output in chunk.items():
        last_msg = output["messages"][-1]
        content = getattr(last_msg, "content", "")
        tool_calls = getattr(last_msg, "tool_calls", [])
        if tool_calls:
            print(f"  [{node_name}] → tool_calls: {[tc['name'] for tc in tool_calls]}")
        elif content:
            preview = content[:80] + "..." if len(content) > 80 else content
            print(f"  [{node_name}] → {preview}")
print()

# ============================================================================
# TOPIC 14: Human-in-the-Loop & Advanced Tool Architecture
# ============================================================================
print("=" * 70)
print("TOPIC 14: Human-in-the-Loop & Advanced Tool Architecture")
print("=" * 70)

# --- 14a: interrupt_before pattern ---
print("\n--- 14a: interrupt_before (intercept BEFORE tools execute) ---")

# Compile a NEW graph with interrupt_before
app_with_interrupt = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # PAUSE before tool execution
)

interrupt_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# This will run model → STOP (before tools)
result = app_with_interrupt.invoke(
    {
        "messages": [HumanMessage(content="What's the weather in London?")],
        "user_name": "Anoop",
        "preferred_language": "English",
    },
    config=interrupt_config,
)

# Inspect the frozen state
frozen_state = app_with_interrupt.get_state(interrupt_config)
print(f"  Graph paused! Next node: {frozen_state.next}")
pending_calls = frozen_state.values["messages"][-1].tool_calls
print(f"  Model wants to call: {pending_calls[0]['name']}({pending_calls[0]['args']})")

# Approve and resume (in production, a human reviews here)
print("  [Human approves] → Resuming...")
result = app_with_interrupt.invoke(None, config=interrupt_config)
print(f"  Final response: {result['messages'][-1].content}")
print()

# --- 14b: ToolRuntime (framework dependency injection) ---
print("--- 14b: ToolRuntime (injected context, hidden from LLM) ---")
print("""
  @tool
  def my_tool(query: str, runtime: ToolRuntime) -> str:
      # LLM only sees: my_tool(query: str)
      # But code has access to:
      runtime.state           # conversation messages, custom fields
      runtime.context         # immutable per-request data (user_id, role)
      runtime.store           # long-term memory (persists across threads)
      runtime.tool_call_id    # unique ID for this execution
      runtime.stream_writer   # emit progress updates to frontend
      runtime.execution_info  # thread_id, run_id, attempt number
""")

# --- 14c: Command (tool mutates graph state) ---
print("--- 14c: Command (tool updates graph state directly) ---")
print("""
  from langgraph.types import Command
  from langchain.messages import ToolMessage

  @tool
  def set_language(language: str, runtime: ToolRuntime) -> Command:
      return Command(update={
          "preferred_language": language,     # ← Mutates state field!
          "messages": [ToolMessage(
              content=f"Language set to {language}.",
              tool_call_id=runtime.tool_call_id,
          )]
      })

  Normal return → model sees text, state unchanged
  Command return → state is mutated (preferences, flags, counters)
""")

# --- 14d: return_direct=True (skip LLM post-processing) ---
print("--- 14d: return_direct=True (short-circuit the loop) ---")
print("""
  @tool(return_direct=True)
  def fetch_order_status(order_id: str) -> str:
      return f"Order {order_id}: Shipped, arrives in 2 days."

  With return_direct=True:
    Tool output → directly to user (NO extra LLM call)

  Without it (default):
    Tool output → back to model → model summarizes → user
""")

# ============================================================================
# SUMMARY: The Complete Agent Loop
# ============================================================================
print("=" * 70)
print("COMPLETE AGENT LOOP SUMMARY")
print("=" * 70)
print("""
  User Question
       │
       ▼
  ┌─────────────────────────────────┐
  │  MODEL NODE                     │
  │  1. Prepend SystemMessage       │
  │  2. Call model_with_tools       │
  │  3. Return AIMessage            │
  └──────────────┬──────────────────┘
                 │
       ┌─────────┴─────────┐
       │  CONDITIONAL EDGE  │
       │  tool_calls?       │
       └──┬──────────────┬──┘
          │              │
     YES  │              │  NO
          ▼              ▼
  ┌──────────────┐   ┌───────┐
  │  TOOL NODE   │   │  END  │
  │  Execute all │   └───────┘
  │  tool_calls  │
  └──────┬───────┘
         │
         └──────────→ back to MODEL NODE

  With Checkpointer:
    - State saved after EVERY node
    - thread_id scopes the conversation
    - Can pause (interrupt), inspect, modify, resume
    - Can time-travel to any past checkpoint
""")

print("\n✅ All 14 prerequisite topics demonstrated!")
print(f"   Thread used: {thread_id}")
print(f"   Total messages in main thread: {len(result['messages'])}")
