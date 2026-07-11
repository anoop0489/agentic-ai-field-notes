"""
A complete LangGraph agent that naturally uses all 14 prerequisite concepts
in a single cohesive program — a Travel Assistant that helps plan trips.

This is NOT a tutorial with sections. It's a real agent you'd build in production,
with comments pointing out which concept is being used and why.

Run: uv run python langgraph_prereq_complete.py
Requires (in pyproject.toml): langchain langchain-openai langgraph python-dotenv truststore
"""

import os
import sys
import uuid
from typing import Annotated, TypedDict

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
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# STATE: Inherit from MessagesState
# We only add our EXTRA fields on top. This is the idiomatic pattern —
# don't redefine the messages field yourself when LangGraph already provides it.
# ---------------------------------------------------------------------------
class TravelState(MessagesState):
    # MessagesState already gives us: messages: Annotated[list[AnyMessage], add_messages]
    # We just add domain-specific fields. No reducer = overwrite on update.
    traveler_name: str
    destination: str
    budget: str


# ---------------------------------------------------------------------------
# MODEL: init_chat_model is the provider-agnostic factory.
# Change "openai:gpt-4o-mini" to "anthropic:claude-sonnet-4-6" and nothing else breaks.
# ---------------------------------------------------------------------------
llm = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0,
    max_retries=3,   # auto-retries on 429/5xx with exponential backoff
)


# ---------------------------------------------------------------------------
# TOOLS: Plain functions decorated with @tool.
# The docstring becomes the description the LLM reads to decide when to call it.
# Type hints define the schema the LLM must follow.
# ---------------------------------------------------------------------------
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    # Simulated results — in production this hits an API
    flights = {
        "Tokyo": "Flight JL101: $850 departing 9:00 AM, Flight AA203: $720 departing 2:30 PM",
        "Paris": "Flight AF001: $650 departing 7:00 AM, Flight DL455: $580 departing 11:00 AM",
        "London": "Flight BA178: $550 departing 6:00 AM, Flight VS401: $490 departing 3:00 PM",
    }
    return flights.get(destination, f"No flights found to {destination} on {date}")


@tool
def search_hotels(city: str, check_in: str, nights: int) -> str:
    """Search for hotels in a city for a given check-in date and number of nights."""
    hotels = {
        "Tokyo": "Hotel Sakura: $120/night (4★), Park Hyatt: $450/night (5★)",
        "Paris": "Le Petit: $95/night (3★), Ritz Paris: $800/night (5★)",
        "London": "Premier Inn: $85/night (3★), The Savoy: $600/night (5★)",
    }
    return hotels.get(city, f"No hotels found in {city}")


@tool
def get_visa_requirements(nationality: str, destination: str) -> str:
    """Check visa requirements for a given nationality traveling to a destination country."""
    # Simplified — real tool would check a database
    visa_free = {
        ("Indian", "Japan"): "Visa required. Apply at Japanese Embassy, processing: 5-7 days.",
        ("Indian", "France"): "Schengen visa required. Apply online, processing: 15 days.",
        ("Indian", "UK"): "UK visa required. Apply online, processing: 3 weeks.",
        ("American", "Japan"): "No visa needed for stays under 90 days.",
        ("American", "France"): "No visa needed for stays under 90 days (EU citizen waiver).",
    }
    key = (nationality, destination)
    return visa_free.get(key, f"Please check embassy website for {nationality} → {destination}")


@tool(return_direct=True)
def emergency_contact(country: str) -> str:
    """Get emergency contact numbers for a country. Use ONLY for safety emergencies."""
    # return_direct=True means this output goes straight to the user
    # WITHOUT another LLM call to rephrase it. The raw text IS the answer.
    contacts = {
        "Japan": "Police: 110 | Ambulance: 119 | Fire: 119 | Embassy: +81-3-2224-5000",
        "France": "Police: 17 | Ambulance: 15 | Fire: 18 | Embassy: +33-1-4312-2222",
        "UK": "Emergency: 999 | Non-emergency: 101 | Embassy: +44-20-7499-9000",
    }
    return contacts.get(country, f"Call local emergency services in {country}")


tools = [search_flights, search_hotels, get_visa_requirements, emergency_contact]


# ---------------------------------------------------------------------------
# bind_tools: Tells the LLM about available tools so it can REQUEST them.
# This doesn't execute anything — it just gives the model the schemas/descriptions.
# ---------------------------------------------------------------------------
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT: Defines the agent's identity and rules.
# Using ChatPromptTemplate for dynamic variables injected at runtime.
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are TravelBot, a helpful travel planning assistant. "
     "The traveler's name is {traveler_name}. "
     "Their destination is {destination} with a budget of {budget}. "
     "Use your tools to find flights, hotels, and visa info. "
     "Be concise and helpful. Always greet the user by name on first interaction."),
    MessagesPlaceholder("messages"),  # Conversation history injected here
])


# ---------------------------------------------------------------------------
# NODES: The processing stations in our graph.
# Each node is a plain Python function: receives state, returns partial update.
# ---------------------------------------------------------------------------
def agent_node(state: TravelState):
    """The MODEL NODE — calls the LLM with context and tools.

    This is the 'brain'. It reads the full conversation, thinks,
    and either answers directly or requests tool calls.
    """
    # Format the prompt with state values (system instructions are NOT in state)
    formatted = prompt.invoke({
        "traveler_name": state.get("traveler_name", "traveler"),
        "destination": state.get("destination", "unknown"),
        "budget": state.get("budget", "flexible"),
        "messages": state["messages"],
    })
    # Call the model — it returns an AIMessage (possibly with tool_calls)
    response = llm_with_tools.invoke(formatted)
    return {"messages": [response]}


# ToolNode is the PREBUILT node that handles all tool execution automatically:
#   1. Reads tool_calls from the last AIMessage
#   2. Matches each call to the right Python function by name
#   3. Executes it with the provided arguments
#   4. Wraps results in ToolMessage with correct tool_call_id
#   5. Returns all ToolMessages as a state update
tool_node = ToolNode(tools)


def trim_history(state: TravelState):
    """A CUSTOM NODE — keeps conversation manageable by trimming old messages.

    Demonstrates that nodes can be ANY logic, not just LLM/tool calls.
    Also shows RemoveMessage for state management.
    """
    messages = state["messages"]
    # Keep only last 20 messages to prevent context overflow
    if len(messages) > 20:
        to_remove = [RemoveMessage(id=m.id) for m in messages[:-20]]
        return {"messages": to_remove}
    return {"messages": []}


# ---------------------------------------------------------------------------
# CONDITIONAL EDGE: The routing function that decides the next node.
# It inspects state (read-only) and returns the name of where to go.
# This is the decision point in the agent loop.
# ---------------------------------------------------------------------------
def should_continue(state: TravelState) -> str:
    """Route based on whether the model requested tool calls or gave a final answer."""
    last_message = state["messages"][-1]
    # If the LLM returned tool_calls → execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise it's a final answer → end the turn
    return END


# ---------------------------------------------------------------------------
# BUILD THE GRAPH: Wire nodes and edges together.
# add_edge = fixed path, add_conditional_edges = dynamic routing.
# ---------------------------------------------------------------------------
graph = StateGraph(TravelState)

# Register all nodes
graph.add_node("trim", trim_history)   # Custom node: manage history
graph.add_node("agent", agent_node)    # Model node: call LLM
graph.add_node("tools", tool_node)     # Tool node: execute tool calls

# Wire the flow
graph.add_edge(START, "trim")                          # Always trim first
graph.add_edge("trim", "agent")                        # Then think
graph.add_conditional_edges("agent", should_continue)  # Agent decides: tools or END
graph.add_edge("tools", "agent")                       # After tools → back to agent


# ---------------------------------------------------------------------------
# COMPILE: Validate, freeze, attach checkpointer, return runnable.
# After this, no more add_node/add_edge — the graph is immutable.
# ---------------------------------------------------------------------------
checkpointer = MemorySaver()  # In-memory persistence (use PostgresSaver in prod)
app = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# RUN THE AGENT: Multi-turn conversation with streaming and time travel.
# ---------------------------------------------------------------------------
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    GRAPH WORKFLOW DIAGRAM                                │
# │                                                                         │
# │   ┌───────┐     ┌──────┐     ┌───────┐     ┌─────────────┐            │
# │   │ START │────▶│ trim │────▶│ agent │────▶│ conditional │            │
# │   └───────┘     └──────┘     └───────┘     └──────┬──────┘            │
# │                                                     │                   │
# │                                        ┌────────────┴────────────┐      │
# │                                        │                         │      │
# │                                        ▼                         ▼      │
# │                                  tool_calls?              no tool_calls  │
# │                                        │                         │      │
# │                                        ▼                         ▼      │
# │                                  ┌───────────┐             ┌─────────┐  │
# │                                  │   tools   │             │   END   │  │
# │                                  └─────┬─────┘             └─────────┘  │
# │                                        │                                │
# │                                        └──────────▶ back to agent       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# CODE EXECUTION STEPS (what happens per .stream() call):
#
#   Step 1: START → trim node
#           - Checks if messages > 20, trims if needed
#           - State checkpoint saved
#
#   Step 2: trim → agent node
#           - Prepends SystemMessage (NOT stored in state)
#           - Calls LLM with tools bound (llm_with_tools.invoke)
#           - Returns AIMessage (may contain tool_calls)
#           - State checkpoint saved
#
#   Step 3: agent → conditional edge (should_continue)
#           - Reads last message from state
#           - If tool_calls exist → route to "tools"
#           - If no tool_calls → route to END
#
#   Step 4a (if tools): tools node
#           - ToolNode reads tool_calls from AIMessage
#           - Executes each tool function by name
#           - Returns ToolMessage(s) with results + tool_call_id
#           - State checkpoint saved
#           - Then → back to Step 2 (agent thinks again with results)
#
#   Step 4b (if END): Graph execution complete
#           - Final AIMessage.content is the answer
#           - State checkpoint saved (final)
#           - Control returns to your code
#
# ---------------------------------------------------------------------------
def main():
    # thread_id: YOU generate it. It scopes this conversation's checkpoints.
    # Same thread_id = continue conversation. New thread_id = fresh start.
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        # Tags let you filter all traces from this session in LangSmith.
        # Each .stream()/.invoke() is a separate trace, but they share this tag.
        "tags": [f"session:{thread_id[:8]}"],
        "metadata": {"script": "langgraph_prereq_complete", "session_id": thread_id},
    }

    print("🌍 TravelBot — Your AI Travel Assistant")
    print(f"   Session: {thread_id[:8]}...")
    print("=" * 60)

    # --- Turn 1: Initial planning request ---
    print("\n👤 User: I want to plan a trip to Tokyo next month. Budget is $2000.")
    print("-" * 60)

    # Streaming: Instead of .invoke() (blocking), use .stream() for real-time updates.
    # stream_mode="updates" shows which node produced what output.
    print("🤖 TravelBot: ", end="", flush=True)
    for chunk in app.stream(
        {
            "messages": [HumanMessage(content="I want to plan a trip to Tokyo next month. My budget is $2000.")],
            "traveler_name": "Anoop",
            "destination": "Tokyo",
            "budget": "$2000",
        },
        config=config,
        stream_mode="updates",
    ):
        for node_name, output in chunk.items():
            if node_name == "agent":
                last = output["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n   🔧 Calling: {[tc['name'] for tc in last.tool_calls]}")
                elif last.content:
                    print(last.content)
            elif node_name == "tools":
                print(f"   ✅ Tools returned results")

    # --- Turn 2: Follow-up (same thread_id = has full history) ---
    print("\n\n👤 User: What about hotels? Something mid-range, 5 nights.")
    print("-" * 60)
    print("🤖 TravelBot: ", end="", flush=True)

    for chunk in app.stream(
        {"messages": [HumanMessage(content="What about hotels? Something mid-range, 5 nights.")]},
        config=config,  # Same thread — agent remembers Tokyo, budget, etc.
        stream_mode="updates",
    ):
        for node_name, output in chunk.items():
            if node_name == "agent":
                last = output["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n   🔧 Calling: {[tc['name'] for tc in last.tool_calls]}")
                elif last.content:
                    print(last.content)
            elif node_name == "tools":
                print(f"   ✅ Tools returned results")

    # --- Turn 3: Visa check ---
    print("\n\n👤 User: Do I need a visa? I'm Indian.")
    print("-" * 60)
    print("🤖 TravelBot: ", end="", flush=True)

    for chunk in app.stream(
        {"messages": [HumanMessage(content="Do I need a visa? I'm Indian.")]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, output in chunk.items():
            if node_name == "agent":
                last = output["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n   🔧 Calling: {[tc['name'] for tc in last.tool_calls]}")
                elif last.content:
                    print(last.content)
            elif node_name == "tools":
                print(f"   ✅ Tools returned results")

    # --- TIME TRAVEL: Browse the full checkpoint history ---
    print("\n\n" + "=" * 60)
    print("⏰ TIME TRAVEL — Browsing checkpoint history")
    print("=" * 60)

    # get_state_history returns every saved snapshot (like git log)
    history = list(app.get_state_history(config))
    print(f"   Total checkpoints saved: {len(history)}")
    for i, snapshot in enumerate(history[:5]):  # Show first 5
        msg_count = len(snapshot.values.get("messages", []))
        next_node = snapshot.next
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id", "?")[:8]
        print(f"   [{i}] checkpoint={checkpoint_id}... | {msg_count} msgs | next={next_node}")

    # get_state: inspect current (latest) state
    current = app.get_state(config)
    print(f"\n   Current state:")
    print(f"     traveler_name: {current.values.get('traveler_name')}")
    print(f"     destination: {current.values.get('destination')}")
    print(f"     budget: {current.values.get('budget')}")
    print(f"     messages: {len(current.values['messages'])} total")

    # --- HUMAN-IN-THE-LOOP: Compile with interrupt, inspect, modify, resume ---
    print("\n\n" + "=" * 60)
    print("🛑 HUMAN-IN-THE-LOOP — interrupt_before demonstration")
    print("=" * 60)

    # Compile a version that pauses BEFORE tool execution
    app_guarded = graph.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )

    guard_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # This runs: trim → agent → STOPS (before tools)
    app_guarded.invoke(
        {
            "messages": [HumanMessage(content="Search flights from Delhi to Tokyo on March 15")],
            "traveler_name": "Anoop",
            "destination": "Tokyo",
            "budget": "$2000",
        },
        config=guard_config,
    )

    # Inspect what the model wants to do (before we let it)
    frozen = app_guarded.get_state(guard_config)
    pending_calls = frozen.values["messages"][-1].tool_calls
    print(f"\n   ⏸️  Graph PAUSED before tools node")
    print(f"   📋 Model wants to call: {pending_calls[0]['name']}")
    print(f"   📋 With arguments: {pending_calls[0]['args']}")
    print(f"   📋 Next node would be: {frozen.next}")

    # Human reviews and approves — resume by invoking with None
    print(f"\n   ✅ Human approves → resuming execution...")
    result = app_guarded.invoke(None, config=guard_config)
    final_msg = result["messages"][-1].content
    preview = final_msg[:100] + "..." if len(final_msg) > 100 else final_msg
    print(f"   🤖 Final answer: {preview}")

    # --- with_config demonstration ---
    print("\n\n" + "=" * 60)
    print("⚙️  with_config — Bind default config to a runnable")
    print("=" * 60)

    # Attach tags/metadata permanently — they flow to ALL sub-calls
    production_app = app.with_config(
        tags=["production", "travel-agent"],
        metadata={"version": "1.0", "team": "AI"},
        run_name="travel_assistant",
    )
    print("   Created production_app with bound config:")
    print("     tags: ['production', 'travel-agent']")
    print("     metadata: {'version': '1.0', 'team': 'AI'}")
    print("   These propagate to every node and tool call automatically.")
    print("   No need to pass them on every .invoke() call.")

    # --- Summary ---
    print("\n\n" + "=" * 60)
    print("✅ DEMO COMPLETE — All 14 prerequisite concepts demonstrated above.")
    print("=" * 60)
    print("""
    1.  State & TypedDict      → TravelState with messages, traveler_name, etc.
    2.  Reducers               → Annotated[list, add_messages] on messages field
    3.  Nodes                  → agent_node, tool_node, trim_history
    4.  Edges                  → add_edge(START→trim→agent), add_edge(tools→agent)
    5.  init_chat_model        → Provider-agnostic LLM factory
    6.  with_config            → production_app with bound tags/metadata
    7.  System Instructions    → ChatPromptTemplate with dynamic variables
    8.  Conditional Edges      → should_continue routes to tools or END
    9.  Tools & ToolNode       → 4 tools + ToolNode for auto-execution
    10. Compile                → graph.compile(checkpointer=...) freezes the graph
    11. Memory & Checkpointers → MemorySaver persists state across turns
    12. Threads & Time Travel  → thread_id scoping + get_state_history browsing
    13. Streaming              → stream_mode="updates" for node-level progress
    14. Human-in-the-Loop      → interrupt_before + get_state + invoke(None) resume
    """)

    # --- Interactive chat: YOUR turn to talk to the agent ---
    print("=" * 60)
    print("💬 INTERACTIVE MODE — Now YOU chat with TravelBot!")
    print("   Type your message and press Enter. Type 'quit' to exit.")
    print("   (Same thread as above — agent remembers the Tokyo trip)")
    print("=" * 60)

    while True:
        user_input = input("\n👤 You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        print("🤖 TravelBot: ", end="", flush=True)
        for chunk in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, output in chunk.items():
                if node_name == "agent":
                    last = output["messages"][-1]
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        print(f"\n   🔧 Calling: {[tc['name'] for tc in last.tool_calls]}")
                    elif last.content:
                        print(last.content)
                elif node_name == "tools":
                    print(f"   ✅ Tools returned results")


if __name__ == "__main__":
    main()
