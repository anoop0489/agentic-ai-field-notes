"""Advanced LangGraph agent — ToolRuntime, Command, HIL with state override.

Demonstrates advanced patterns beyond the simple invoke example:
  - ToolRuntime with typed context (dependency injection into tools)
  - Command for tool-driven state mutation
  - Human-in-the-Loop with update_state (manager overrides tool args)
  - return_direct tools (skip final LLM summary)

Run: uv run python langgraph_advanced_hil.py
"""

import sys
import uuid
from dataclasses import dataclass
from typing import Literal

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
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║          QUICK REFERENCE — HOW STATE GETS UPDATED IN LANGGRAPH       ║
# ╠═══════════════════════════════════════════════════════════════════════╣
# ║                                                                       ║
# ║  WHO CAN UPDATE STATE?                                                ║
# ║  ─────────────────────                                                ║
# ║  1. REGULAR NODE (e.g., agent_node)                                   ║
# ║     → Receives full state, returns a dict with fields to update       ║
# ║     → Can update ANY field: messages, order_status, budget, etc.      ║
# ║     → Example: return {"messages": [ai_msg], "order_status": "New"}   ║
# ║                                                                       ║
# ║  2. TOOLNODE (prebuilt)                                               ║
# ║     → Internally runs tool functions, wraps output in ToolMessage     ║
# ║     → Only updates: {"messages": [ToolMessage(...)]}                  ║
# ║     → CANNOT update custom fields (order_status, budget, etc.)        ║
# ║                                                                       ║
# ║  3. TOOL FUNCTION (plain @tool)                                       ║
# ║     → Just a function. Returns a string. That's it.                   ║
# ║     → It does NOT receive state, it does NOT update state.            ║
# ║     → ToolNode wraps its return value into a ToolMessage.             ║
# ║                                                                       ║
# ║  4. TOOL FUNCTION with COMMAND (@tool returning Command)              ║
# ║     → The ONLY way a tool function can update custom state fields.    ║
# ║     → Returns Command(update={"order_status": "Done", "messages": [..]})
# ║     → This bypasses ToolNode's normal wrapping — it writes directly.  ║
# ║                                                                       ║
# ║  WHEN DO YOU NEED COMMAND?                                            ║
# ║  ──────────────────────────                                           ║
# ║  • Your tool needs to WRITE to custom state fields (not just messages)║
# ║  • Example: tool sets order_status, budget, destination, etc.         ║
# ║  • Without Command, tool output is just a string in a ToolMessage —   ║
# ║    the LLM sees it, but your state fields remain unchanged.           ║
# ║                                                                       ║
# ║  SIMPLE MENTAL MODEL:                                                 ║
# ║  ────────────────────                                                 ║
# ║    Node    → "I have the state, I return updates"                     ║
# ║    Tool    → "I just compute and return a string"                     ║
# ║    Command → "I'm a tool, but I ALSO need to update state"            ║
# ║                                                                       ║
# ║  C# ANALOGY:                                                          ║
# ║    Node    = Service method that reads & writes to DbContext           ║
# ║    Tool    = Pure utility function (no DB access)                      ║
# ║    Command = Utility function that ALSO dispatches a domain event      ║
# ║                                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# =====================================================================
# 1. THE ARCHITECTURAL STATE DEFINITION (Topic 1 & 2)
# =====================================================================
class AgentState(MessagesState):
    """The structured data contract blue print for our application."""
    preferred_language: str
    order_status: str

# =====================================================================
# 2. RUNTIME CONTEXT DEFINITION (Topic 14)
# =====================================================================
@dataclass
class UserContext:
    """Immutable environmental data securely passed by the API gateway."""
    user_id: str
    account_type: Literal["Standard", "Premium"]

# =====================================================================
# 3. ADVANCED CONTEXT-AWARE TOOLS (Topic 3, 9, & 14)
# =====================================================================
# 3. ADVANCED CONTEXT-AWARE TOOLS (Topic 3, 9, & 14)
#
# NORMAL TOOL vs COMMAND TOOL — What's the difference?
#
#   NORMAL TOOL (returns str):
#     @tool
#     def get_weather(city: str) -> str:
#         return "72°F sunny"
#     → The string goes into a ToolMessage → LLM reads it → LLM responds
#     → Graph state is NOT changed (only messages list gets the ToolMessage)
#
#   COMMAND TOOL (returns Command):
#     @tool
#     def process_order(item: str, runtime: ToolRuntime) -> Command:
#         return Command(update={"order_status": "Completed", "messages": [...]})
#     → The Command WRITES to graph state (sets order_status = "Completed")
#     → AND adds a ToolMessage so the LLM knows what happened
#     → This is how a tool can MUTATE your custom state fields
#
#   WHY NOT JUST RETURN A STRING?
#     If process_order returned "Order completed", the LLM would see that text,
#     but order_status in your state would still be empty/unset. You'd have no
#     way to track the order programmatically. Command solves this by letting
#     the tool update state fields directly.
#
#   C# ANALOGY:
#     - Normal tool = Controller action that returns Ok("result")
#     - Command tool = Controller action that ALSO writes to HttpContext.Items
#       or dispatches a domain event before returning
# =====================================================================
@tool
def process_premium_order(item: str, runtime: ToolRuntime[UserContext]) -> Command:
    """Process an order for a high-value premium item. Requires manager approval."""

    # A. Dependency Injection via ToolRuntime:
    #    runtime.context gives us the UserContext dataclass that was passed
    #    via context= on app.invoke()/app.stream(). The LLM never sees this —
    #    it's injected by the framework, like [FromServices] in ASP.NET.
    secure_user = runtime.context.user_id      # "user_anoop"
    tier = runtime.context.account_type         # "Premium"

    # B. Read session state via ToolRuntime:
    #    runtime.state is the current graph state (AgentState dict).
    #    This lets a tool READ custom fields without them being tool parameters.
    current_lang = runtime.state.get("preferred_language", "English")

    # Execute backend ordering logic securely
    confirmation_text = f"Order processed for {item}. User: {secure_user} ({tier} Tier). Lang: {current_lang}"

    # C. Return a Command to WRITE BACK to graph state:
    #
    #    Command(update={...}) does TWO things at once:
    #
    #    1. MUTATES STATE: Sets order_status = "Completed" in AgentState
    #       → Without Command, there's no way for a tool to change custom state fields.
    #       → A normal string return only adds a ToolMessage — it can't touch order_status.
    #
    #    2. ADDS TOOLMESSAGE: The messages list gets a ToolMessage so the LLM
    #       knows the tool succeeded and what it did. The tool_call_id MUST match
    #       the original tool_call from the AIMessage — this is how LangGraph
    #       correlates "which tool call produced which result".
    #
    #    Think of it as: return value + side effect in one atomic operation.
    return Command(
        update={
            "order_status": "Completed",           # ← Mutates AgentState.order_status
            "messages": [
                ToolMessage(
                    content=confirmation_text,
                    tool_call_id=runtime.tool_call_id  # ← Correlates with the AIMessage's tool_call
                )
            ]
        }
    )


# ---------------------------------------------------------------------------
# return_direct=True — SHORT-CIRCUIT: Skip the final LLM call
#
# NORMAL FLOW (return_direct=False, the default):
#   User → Agent(LLM) → Tool runs → ToolMessage → Agent(LLM again) → Final answer
#   The LLM gets to READ the tool result and REPHRASE it for the user.
#
# SHORT-CIRCUIT FLOW (return_direct=True):
#   User → Agent(LLM) → Tool runs → Tool output IS the final answer → END
#   The LLM does NOT get another turn. The raw tool output goes to the user.
#
# WHEN TO USE return_direct=True:
#   - Database lookups where the result IS the answer (no rephrasing needed)
#   - Status checks, order tracking, price lookups
#   - When you want to SAVE an LLM call (faster + cheaper)
#
# WHEN NOT TO USE:
#   - Tool returns raw data that needs summarization
#   - Tool returns partial info that needs combining with other results
#   - When the LLM should reason about the result before responding
#
# HOW IT WORKS INTERNALLY:
#   ToolNode checks: does this tool have return_direct=True?
#   If yes → it sets a flag on the ToolMessage
#   → The conditional edge sees this flag → routes to END instead of back to agent
#   → The ToolMessage content becomes the user-facing response
#
# C# ANALOGY:
#   return_direct=True ≈ A middleware that returns a cached response
#   directly from Redis without ever hitting the controller (LLM).
# ---------------------------------------------------------------------------
@tool(return_direct=True)
def fetch_fast_status(order_id: str) -> str:
    """Fetch an instantaneous status update. Short-circuits the LLM loop."""
    return f"Order {order_id} is packing at fulfillment center."


# Wrap tools into the prebuilt execution container node
tool_node = ToolNode([process_premium_order, fetch_fast_status])

# =====================================================================
# 4. INITIALIZE AGENT ENGINE & ROUTING LOGIC (Topic 5, 7, & 8)
# =====================================================================
# Topic 5 & 7: Provider-agnostic model factory + bind tool schemas
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([process_premium_order, fetch_fast_status])

def agent_node(state: AgentState):
    """The core intelligence node processing incoming conversation threads."""
    # SystemMessage (not AIMessage!) sets the agent's identity/rules
    system_prompt = SystemMessage(
        content="You are a financial agent. "
        "When a user asks to order an item, you MUST use the process_premium_order tool. "
        "When a user asks for order status, you MUST use the fetch_fast_status tool. "
        "Never process orders or check status without using the appropriate tool."
    )
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def route_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """Topic 8: Conditional Edge tracks inspecting live tool invocation states."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# =====================================================================
# 5. ASSEMBLE WORKFLOW BLUEPRINT (Topic 4, 10, & 11)
# =====================================================================
workflow = StateGraph(AgentState)

# Wire up the functional execution track pathways
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_tools)
workflow.add_edge("tools", "agent")

# Topic 11: Attach a persistent database abstraction engine layer
checkpointer_db = MemorySaver()

# Topic 10 & 14: Lock tracks down and install structural brake guardrails
app = workflow.compile(
    checkpointer=checkpointer_db,
    interrupt_before=["tools"]  # Pulls emergency brake right before tool runs!
)

# =====================================================================
# 6. EXECUTING LIVE ARCHITECTURAL PATHWAYS (Topic 12, 13, & 14)
# =====================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                    THREE LEVELS OF HUMAN-IN-THE-LOOP                       │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                            │
# │  LEVEL 1 — SIMPLE (Approve / Reject)                                      │
# │  ─────────────────────────────────────                                     │
# │  Graph pauses → Human sees pending tool call → Approves or doesn't.       │
# │  Code: invoke(None) to approve, or just don't resume to reject.           │
# │  Example: "LLM wants to call search_flights(Tokyo) — OK? Yes → resume"   │
# │                                                                            │
# │  LEVEL 2 — MODERATE (Reject + Give Feedback)                              │
# │  ────────────────────────────────────────────                              │
# │  Graph pauses → Human rejects → Adds a HumanMessage with feedback →      │
# │  Re-runs the agent node so the LLM tries again with new instructions.     │
# │  Code: update_state({"messages": [HumanMessage("No, use a cheaper       │
# │         flight")]}, as_node="human") then invoke(None).                    │
# │  Example: "Don't book business class, try economy instead"                │
# │                                                                            │
# │  LEVEL 3 — ADVANCED (Rewrite Tool Args) ← THIS DEMO                      │
# │  ───────────────────────────────────────                                   │
# │  Graph pauses → Human MODIFIES the pending tool_call arguments →          │
# │  Writes a new checkpoint with corrected args via update_state() →         │
# │  Resumes, and the tool runs with the HUMAN'S args, not the LLM's.        │
# │  Code: update_state({"messages": [AIMessage(id=..., tool_calls=          │
# │         [{...corrected args...}])]}, as_node="agent")                     │
# │  Example: Manager changes order from "Paneer Biryani" to                  │
# │           "Vegetable Manchuria" before the order tool executes.            │
# │                                                                            │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                    EXECUTION FLOW OF THIS DEMO                             │
# │                                                                            │
# │  STAGE 1: app.stream(user_input, ...)                                      │
# │    ┌───────┐     ┌───────┐     ┌─────────────┐     ⏸️ PAUSED              │
# │    │ START │────▶│ agent │────▶│ has tools?   │────▶ interrupt_before      │
# │    └───────┘     └───────┘     │ YES → tools  │     ["tools"]              │
# │                                └──────────────┘                            │
# │    • LLM sees "order Paneer Biryani" → calls process_premium_order         │
# │    • Graph STOPS before tools node runs (checkpoint saved)                 │
# │    • Tool has NOT executed yet — args are still pending                    │
# │                                                                            │
# │  STAGE 2: app.update_state(corrected_payload, as_node="agent")             │
# │    ┌──────────────────────────────────────────────────────────────┐        │
# │    │ Manager reviews pending call via get_state()                  │        │
# │    │ Sees: process_premium_order(item="Paneer Biryani")            │        │
# │    │ Decides to change → item="Vegetable Manchuria"                │        │
# │    │ Writes corrected AIMessage with SAME id + tool_call_id        │        │
# │    │ update_state() saves new checkpoint (replaces pending call)   │        │
# │    └──────────────────────────────────────────────────────────────┘        │
# │    • The checkpoint now has the MANAGER'S args, not the LLM's             │
# │                                                                            │
# │  STAGE 3: app.invoke(None, ...)                                            │
# │    ┌───────────┐     ┌───────┐     ┌─────────────┐     ┌───────┐         │
# │    │   tools    │────▶│ agent │────▶│ has tools?   │────▶│  END  │         │
# │    │ (runs with │     │ (LLM  │     │ NO → END     │     └───────┘         │
# │    │ corrected  │     │ gives │     └──────────────┘                       │
# │    │ args!)     │     │ final │                                            │
# │    └───────────┘     │answer)│                                             │
# │                       └───────┘                                            │
# │    • process_premium_order runs with "Vegetable Manchuria"                 │
# │    • Command updates order_status="Completed" in state                    │
# │    • Agent loops back, LLM sees tool result, gives final answer            │
# │    • No more tool_calls → END                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
if __name__ == "__main__":
    # Topic 12: Generate a unique thread_id for this conversation
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Injected immutable security claims — passed as context= on invoke/stream
    # ToolRuntime[UserContext] will receive this inside the tool function.
    request_context = UserContext(user_id="user_anoop", account_type="Premium")

    user_input = {"messages": [HumanMessage(content="Please order me a Paneer Biryani.")]}

    print("\n" + "=" * 60)
    print("ADVANCED HIL — ToolRuntime, Command, State Override")
    print("=" * 60)
    print(f"Thread ID: {thread_id[:8]}...")

    # ─────────────────────────────────────────────────────────
    # STAGE 1: Initial request — graph runs until interrupt
    # ─────────────────────────────────────────────────────────
    print("\n--- STAGE 1: Initial request (graph will PAUSE before tools) ---")
    print(f"Input: '{user_input['messages'][0].content}'")

    # context= passes UserContext into ToolRuntime for any tool that declares it
    for chunk in app.stream(
        user_input,
        config=thread_config,
        context=request_context,
        stream_mode="updates",
    ):
        print(f"  Node executed: {list(chunk.keys())[0]}")

    # Inspect the frozen state — graph is paused before tools node
    frozen_state = app.get_state(thread_config)
    print(f"\n  Graph paused? {len(frozen_state.next) > 0}")
    print(f"  Next node:    {frozen_state.next}")
    print(f"  Pending call: {frozen_state.values['messages'][-1].tool_calls}")

    # ─────────────────────────────────────────────────────────
    # STAGE 2: Manager overrides the tool arguments
    #
    # This is the advanced HIL pattern: instead of just approving
    # the tool call, the manager CHANGES what the tool will receive.
    # update_state() writes a new checkpoint with modified args.
    # ─────────────────────────────────────────────────────────
    print("\n--- STAGE 2: Manager overrides tool args via update_state ---")
    print("  Original item: 'Paneer Biryani'")
    print("  Override item: 'Vegetable Manchuria'")

    # Build a corrected AIMessage that REPLACES the pending one
    # (must match the original message ID and tool_call ID)
    original_msg = frozen_state.values['messages'][-1]
    corrected_payload = {
        "messages": [
            AIMessage(
                id=original_msg.id,
                content="",
                tool_calls=[{
                    "name": "process_premium_order",
                    "args": {"item": "Vegetable Manchuria"},
                    "id": original_msg.tool_calls[0]["id"]
                }]
            )
        ]
    }

    # Write the corrected checkpoint as if the agent node produced it
    app.update_state(thread_config, corrected_payload, as_node="agent")
    print("  Checkpoint updated with corrected tool args.")

    # ─────────────────────────────────────────────────────────
    # STAGE 3: Resume — tools run with the overridden args
    # ─────────────────────────────────────────────────────────
    print("\n--- STAGE 3: Resume execution (tools run with corrected args) ---")

    # invoke(None) = continue from the latest checkpoint
    final_output = app.invoke(None, config=thread_config, context=request_context)

    print(f"\n  Final answer:  {final_output['messages'][-1].content}")
    print(f"  Order status:  {final_output.get('order_status')}")

    # Show the full message trace
    print("\n  Full message trace:")
    for i, msg in enumerate(final_output["messages"]):
        role = msg.__class__.__name__
        content = getattr(msg, "content", "")[:80]
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            print(f"  [{i}] {role}: tool_calls={[tc['name'] for tc in tool_calls]}")
        else:
            print(f"  [{i}] {role}: {content}")

    print("\n" + "=" * 60)
    print("DONE. Advanced HIL with state override.")
    print("=" * 60)