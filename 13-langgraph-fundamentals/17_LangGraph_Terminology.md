# LangGraph Terminology — Clear Definitions & Examples

> A focused reference to understand what each piece is, how the execution flows, and how state gets updated.

---

## The Big Picture — In Plain English

Here's what happens under the hood when you run a LangGraph agent, explained like telling a story:

1. **You (the user)** type something like "Book me a flight to Tokyo".
2. That text becomes a **HumanMessage** and gets placed into **state** (the shared data bag).
3. The **Agent Node** runs. It takes the messages from state and sends them to the **LLM** (like GPT-4).
4. The LLM reads your message and makes a **decision**:
   - **Option A:** "I can answer this directly" → it returns an **AIMessage** with text (like "Sure, here's info about Tokyo") → goes to **END**. Done.
   - **Option B:** "I need to use a tool" → it returns an **AIMessage** that says "call search_flights with destination=Tokyo" (this is a `tool_call`).
5. A **Conditional Edge** (a simple Python function) checks: does the AIMessage have `tool_calls`? If yes → route to **ToolNode**. If no → route to **END**.
6. The **ToolNode** runs. It looks at the `tool_calls`, finds the matching **tool function** (`search_flights`), and calls it with the args the LLM specified (`destination="Tokyo"`).
7. The **tool function** executes (calls an API, queries a database, whatever) and **returns a string** like "Flight JL101: $850, departing 2pm".
8. ToolNode takes that string, wraps it in a **ToolMessage**, and puts it into state's `messages`.
9. Control goes **back to the Agent Node**. Now the LLM sees the original conversation PLUS the tool result.
10. The LLM reads the tool result and gives a **final answer**: "I found a flight to Tokyo — JL101 for $850 departing at 2pm. Want me to book it?"
11. This time there are no `tool_calls` → **Conditional Edge** routes to **END**. Done.

**That's the complete cycle.** The LLM never directly calls Python functions — it just says "I want to call X with Y" and the framework does the actual execution.

---

## The Complete Flow Diagram

```
                         User: "Book me a flight to Tokyo"
                                      |
                                      v
                               [HumanMessage]
                                      |
                                      v
                              +-------+-------+
                              |    START      |
                              +-------+-------+
                                      |
                                      v
                    +-------------------------------------+
                    |           AGENT NODE                 |
                    |                                     |
                    |  1. Reads state["messages"]          |
                    |  2. Sends to LLM (GPT-4)            |
                    |  3. LLM returns AIMessage           |
                    |  4. Returns {"messages": [AIMsg]}   |
                    +------------------+------------------+
                                       |
                                       v
                    +-------------------------------------+
                    |        CONDITIONAL EDGE              |
                    |                                     |
                    |  Does AIMessage have tool_calls?    |
                    +--------+-----------------+----------+
                             |                 |
                         YES |                 | NO
                             v                 v
               +-------------------+    +------------+
               |    TOOL NODE      |    |    END     |
               |                   |    |            |
               | 1. Reads tool_calls|    | Final      |
               | 2. Finds @tool fn  |    | answer     |
               | 3. Calls with args |    | returned   |
               | 4. Wraps result in |    +------------+
               |    ToolMessage     |
               | 5. Returns         |
               |  {"messages":      |
               |   [ToolMessage]}   |
               +---------+---------+
                         |
                         | ToolMessage added to state
                         |
                         v
                  (back to AGENT NODE)
                         |
                         v
                 Conditional Edge again
                         |
                    no tool_calls
                         |
                         v
                  +------------+
                  |    END     |
                  +------------+
```

**What the diagram shows:**
- The **cycle** (Agent -> Tools -> Agent) repeats until the LLM has no more tool calls
- The **Conditional Edge** is the decision point that either loops or stops
- ToolNode always routes back to Agent - the LLM always gets to see tool results

---

## The Flow With Human-in-the-Loop (HIL)

When you add `interrupt_before=["tools"]`, the graph **pauses** before the ToolNode runs. A human can then inspect, approve, reject, or modify the pending tool call before resuming.

```
               User: "Order me a Paneer Biryani"
                              |
                              v
                       +-----------+
                       |   START   |
                       +-----+-----+
                             |
                             v
               +-------------------------------+
               |         AGENT NODE             |
               | LLM: "call process_order(      |
               |        item='Paneer Biryani')" |
               | Returns AIMessage + tool_calls |
               +---------------+---------------+
                               |
                               v
                      (Conditional Edge)
                        tool_calls? YES
                               |
                               v
               +===============================+
               ||  GRAPH PAUSES HERE           ||
               ||                             ||
               ||  interrupt_before=["tools"]  ||
               ||                             ||
               ||  State saved to checkpoint  ||
               ||  Tool has NOT run yet       ||
               ||  Args are just pending      ||
               +===============================+
                               |
                               v
               +-------------------------------+
               |   HUMAN REVIEWS & DECIDES     |
               |                               |
               |  A) APPROVE (Level 1):        |
               |     invoke(None, config)      |
               |     Tool runs as-is           |
               |                               |
               |  B) REJECT + FEEDBACK (Lv 2): |
               |     update_state(HumanMsg)    |
               |     Agent re-thinks           |
               |                               |
               |  C) REWRITE ARGS (Level 3):   |
               |     update_state(corrected    |
               |       AIMessage with new args)|
               |     Tool gets human's args    |
               +---------------+---------------+
                               |
                               v  (after approve/modify)
               +-------------------------------+
               |         TOOL NODE              |
               | Runs with (possibly corrected) |
               | args. Returns ToolMessage.     |
               +---------------+---------------+
                               |
                               v
               +-------------------------------+
               |         AGENT NODE             |
               | LLM reads result -> answer     |
               +---------------+---------------+
                               |
                               v
                        +-----------+
                        |    END    |
                        +-----------+
```

### The Three HIL Levels Explained Simply

| Level | What Happens | When To Use |
|-------|-------------|-------------|
| **Level 1 — Approve** | Human says "go ahead". Tool runs as-is. | Simple confirmation ("are you sure you want to delete?") |
| **Level 2 — Reject + Feedback** | Human says "no, do it differently". Agent re-thinks. | "Don't book business class, try economy" |
| **Level 3 — Rewrite Args** | Human directly changes what the tool will receive. | Manager overrides order item, corrects a typo, changes amount |

### How HIL Works Under the Hood

1. **Checkpoint saves state** — When the graph pauses, the current state (all messages, custom fields) is saved with a `thread_id`.
2. **`get_state(config)`** — You read the frozen state to see what tool call is pending.
3. **`update_state(config, new_values, as_node="...")`** — You write new values into the checkpoint. The `as_node` tells the graph "pretend this update came from that node" (important for routing).
4. **`invoke(None, config)`** — Passing `None` means "don't add new input, just continue from where you paused".

---

## Terminology — Definitions & Examples

### 1. State

The shared data container that flows through the entire graph. Every node can read it, and nodes return updates to it.

```python
from langgraph.graph import MessagesState

class TravelState(MessagesState):
	# MessagesState already gives you: messages: list[BaseMessage]
	# Below are YOUR custom fields:
	destination: str
	budget: float
	order_status: str
```

**Key point:** State is just a TypedDict (a dictionary with defined keys). When a node runs, it receives the current state and returns a partial dict of fields to update.

---

### 2. MessagesState

A built-in state that LangGraph provides. It has exactly one field:

```python
class MessagesState(TypedDict):
	messages: Annotated[list[BaseMessage], add_messages]
```

- `messages` is a list of LangChain message objects (HumanMessage, AIMessage, ToolMessage)
- `add_messages` is a **reducer** — it means "append new messages to existing ones" instead of replacing

When you inherit from `MessagesState`, you get `messages` for free and can add your own fields on top.

---

### 3. Node (The General Concept)

A **node** is any Python function registered in the graph. It receives state, does something, and returns fields to update. There are three common types:

| Type | What it does | Example |
|------|-------------|---------|
| Plain Node | Any logic — no LLM, no tools | Trim messages, validate data, set defaults |
| Agent Node | Calls the LLM | The "brain" that decides what to do next |
| ToolNode | Runs a collection of tool functions | Executes whatever the LLM asked for |

All three follow the same contract: **receive state → return partial state update**.

#### 3a. Plain Node (no LLM, no tools — just logic)

```python
def set_defaults(state: TravelState) -> dict:
	# No LLM, no tools — just pure Python logic
	return {"destination": "Unknown", "budget": 0.0}

def trim_old_messages(state: TravelState) -> dict:
	# Keep only last 10 messages to save tokens
	recent = state["messages"][-10:]
	return {"messages": recent}

def validate_budget(state: TravelState) -> dict:
	# Business rule: cap budget at 5000
	capped = min(state["budget"], 5000.0)
	return {"budget": capped}
```

**When to use:** Preprocessing, cleanup, validation, setting defaults — anything that doesn't need an LLM or external API.

#### 3b. Agent Node (where the LLM lives)

```python
def agent_node(state: TravelState) -> dict:
	# Read messages from state
	response = llm_with_tools.invoke(state["messages"])

	# Return the AIMessage to be appended to state["messages"]
	return {"messages": [response]}
```

**What it updates:** Typically just `messages` (appends an AIMessage). But it CAN update any field:

```python
def agent_node(state: TravelState) -> dict:
	response = llm_with_tools.invoke(state["messages"])

	# Update messages AND a custom field
	return {
		"messages": [response],
		"order_status": "Processing"  # ← agent node CAN do this
	}
```

**When to use:** Whenever you need the LLM to reason, decide, or generate a response.

#### 3c. ToolNode (a prebuilt node — collection of tool functions)

A **prebuilt** node from LangGraph that executes tool functions. You don't write its code — LangGraph provides it. You just give it a list of tools.

```python
from langgraph.prebuilt import ToolNode

# Give it your collection of tool functions
tool_node = ToolNode([search_flights, book_hotel, get_weather])
```

**What it does internally:**
1. Reads `state["messages"][-1].tool_calls` (the pending tool calls from the AIMessage)
2. Finds the matching tool function by name
3. Runs each tool function with the arguments the LLM specified
4. Wraps each tool's return value in a `ToolMessage`
5. Returns `{"messages": [ToolMessage(...), ToolMessage(...), ...]}`

**What it updates:** Only `messages`. It appends ToolMessages. It CANNOT update custom fields like `order_status` or `budget`.

#### How they all register in the graph:

```python
workflow = StateGraph(TravelState)

# All three are just "nodes" to the graph — same add_node() call
workflow.add_node("setup", set_defaults)          # plain node
workflow.add_node("agent", agent_node)            # agent node (has LLM)
workflow.add_node("tools", tool_node)             # ToolNode (collection of tools)

# Wire them up
workflow.add_edge(START, "setup")
workflow.add_edge("setup", "agent")
workflow.add_conditional_edges("agent", route_tools)
workflow.add_edge("tools", "agent")
```

**Key insight:** The graph doesn't care what's inside a node. It just calls the function, gets back a dict, and merges it into state. Whether that function uses an LLM, runs tools, or just does `return {"budget": 100}` — the graph treats them all the same.

---

### 4. Tool Function

A plain Python function decorated with `@tool`. It is NOT a node. It runs INSIDE the ToolNode.

```python
from langchain.tools import tool

@tool
def search_flights(destination: str) -> str:
	"""Search for available flights."""
	return f"Flight to {destination}: $500, departing 10am"
```

**Key facts:**
- It receives ONLY the arguments the LLM passed (like `destination`)
- It does NOT receive state
- It returns a string (or simple value)
- It does NOT update state — ToolNode wraps its return into a ToolMessage

---

### 5. Edges & Conditional Edges

**Edge** — A fixed connection between two nodes. "After A always go to B."

```python
workflow.add_edge(START, "agent")     # always start at agent
workflow.add_edge("tools", "agent")   # after tools, always go back to agent
```

**Conditional Edge** — A function that looks at state and decides where to go next.

```python
def route_tools(state: TravelState) -> str:
	last_msg = state["messages"][-1]
	if last_msg.tool_calls:
		return "tools"    # LLM wants to call a tool → go to ToolNode
	return END            # LLM gave a final answer → stop

workflow.add_conditional_edges("agent", route_tools)
```

This is how the graph knows whether to loop (tool call → agent → tool call → agent) or stop.

---

### 6. Command

The **only way** for a tool function to update custom state fields. Without it, tools can only return strings.

```python
@tool
def book_flight(destination: str, runtime: ToolRuntime) -> Command:
	return Command(
		update={
			"destination": destination,          # ← updates custom field!
			"budget": 500.0,                     # ← updates custom field!
			"messages": [ToolMessage(
				content=f"Booked flight to {destination}",
				tool_call_id=runtime.tool_call_id
			)]
		}
	)
```

**Why it exists:** If `book_flight` just returned a string like `"Booked!"`, the LLM would see that text, but `state["destination"]` and `state["budget"]` would remain empty. Command lets the tool write to state directly.

---

### 7. ToolRuntime

An object that gets **injected** into a tool function to give it access to things it normally can't see:

```python
@tool
def my_tool(query: str, runtime: ToolRuntime[MyContext]) -> Command:
	# runtime.state     → current graph state (read custom fields)
	# runtime.context   → your custom context object (user info, auth, etc.)
	# runtime.tool_call_id → needed for ToolMessage correlation
	...
```

The LLM never sees `runtime` — it's hidden from the tool schema. Only `query` appears as a parameter the LLM fills in.

---

### 8. Checkpointer & thread_id

**Checkpointer** — Saves the complete state after every single node execution. This enables pause/resume, time travel, and HIL.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # in-memory (use PostgresSaver for production)
app = workflow.compile(checkpointer=checkpointer)
```

**thread_id** — A unique ID that identifies which conversation's checkpoints to load/save.

```python
config = {"configurable": {"thread_id": "conversation-123"}}
app.invoke(user_input, config=config)
```

Without `thread_id`, the graph wouldn't know which conversation to resume when you call `invoke(None)`.

---

### 9. Reducer

A function attached to a state field that defines HOW updates are merged.

```python
from typing import Annotated
from langgraph.graph import add_messages

class MyState(TypedDict):
	messages: Annotated[list, add_messages]   # ← APPEND new messages to existing
	counter: int                               # ← no reducer = REPLACE the value
```

- `messages` uses `add_messages` reducer → new messages are **appended**
- `counter` has no reducer → new value **replaces** the old one

---

## How Each One Updates State — Complete Picture

### Agent Node updates state:

```python
def agent_node(state: TravelState) -> dict:
	response = llm.invoke(state["messages"])
	return {"messages": [response]}  # ← this merges into state
```

The graph takes `{"messages": [response]}` and appends it to `state["messages"]`.

### ToolNode updates state:

```python
# You don't write this code — it happens internally:
# ToolNode runs your tool function, gets back "Flight to Tokyo: $500"
# Then it does:
return {"messages": [ToolMessage(content="Flight to Tokyo: $500", tool_call_id="...")]}
```

Only messages get updated. Your custom fields (`budget`, `destination`) are untouched.

### Tool function does NOT update state:

```python
@tool
def search_flights(destination: str) -> str:
	return "Flight to Tokyo: $500"
	# This string goes to ToolNode → ToolNode wraps it → state["messages"] gets a ToolMessage
	# But state["budget"] or state["destination"]? UNCHANGED.
```

### Tool function with Command DOES update state:

```python
@tool
def book_flight(destination: str, runtime: ToolRuntime) -> Command:
	return Command(
		update={
			"destination": destination,          # ← updates custom field!
			"budget": 500.0,                     # ← updates custom field!
			"messages": [ToolMessage(
				content=f"Booked flight to {destination}",
				tool_call_id=runtime.tool_call_id
			)]
		}
	)
```

---

## When Does a Custom Field Actually Get Updated?

Given this state:
```python
class OrderState(MessagesState):
	order_status: str    # starts as ""
	item_name: str       # starts as ""
```

**Scenario A — Normal tool (no Command):**
```
User: "Order pizza"
Agent Node → returns {"messages": [AIMessage(tool_calls=[order_food(item="pizza")])]}
ToolNode   → runs order_food("pizza") → gets "Done" → returns {"messages": [ToolMessage("Done")]}
Agent Node → returns {"messages": [AIMessage("Your pizza is ordered!")]}

State at end:
  messages = [HumanMessage, AIMessage, ToolMessage, AIMessage]  ✅ updated
  order_status = ""   ← STILL EMPTY! Nobody wrote to it.
  item_name = ""      ← STILL EMPTY!
```

**Scenario B — Tool with Command:**
```
User: "Order pizza"
Agent Node → returns {"messages": [AIMessage(tool_calls=[order_food(item="pizza")])]}
ToolNode   → runs order_food("pizza") → tool returns Command(update={
				"order_status": "Completed",
				"item_name": "pizza",
				"messages": [ToolMessage("Done")]
			 })
Agent Node → returns {"messages": [AIMessage("Your pizza is ordered!")]}

State at end:
  messages = [HumanMessage, AIMessage, ToolMessage, AIMessage]  ✅ updated
  order_status = "Completed"  ✅ updated by Command!
  item_name = "pizza"         ✅ updated by Command!
```

**Scenario C — Agent node does it directly (no tools needed):**
```
User: "Set my destination to Paris"
Agent Node → returns {
	"messages": [AIMessage("Done! Destination set to Paris.")],
	"destination": "Paris"   ← agent node writes directly
}

State at end:
  messages = [HumanMessage, AIMessage]  ✅ updated
  destination = "Paris"                 ✅ updated by agent node!
```

---

## Summary Table

| Component | Receives State? | Updates State? | What It Updates |
|-----------|----------------|----------------|-----------------|
| Plain Node | ✅ Full state | ✅ Any field | Whatever you return in the dict |
| Agent Node | ✅ Full state | ✅ Any field | Usually `messages`, but CAN update custom fields |
| ToolNode | ✅ Full state (internally) | ✅ Only `messages` | Appends ToolMessages |
| Tool Function | ❌ Only its args | ❌ Nothing | Returns a string to ToolNode |
| Tool + Command | ✅ Via ToolRuntime | ✅ Any field | Whatever you put in `Command(update={...})` |

---

## One-Line Definitions

- **State** — The shared dictionary that all nodes read from and write to.
- **MessagesState** — A built-in state with just `messages: list[BaseMessage]`.
- **Node** — A function that takes state in and returns partial state updates out.
- **Plain Node** — A node with just Python logic. No LLM, no tools.
- **Agent Node** — A node where the LLM lives. Reads messages, returns LLM response.
- **ToolNode** — A prebuilt node that runs tool functions and wraps results in ToolMessages.
- **Tool Function** — A decorated Python function. Receives only its args. Returns a string.
- **Command** — An object a tool returns to write directly to state (the tool's escape hatch).
- **ToolRuntime** — Injected object giving a tool access to state and context it normally can't see.
- **Reducer** — A function that defines HOW a state field gets updated (e.g., append vs replace).
- **Edge** — A fixed connection: "after node A, always go to node B".
- **Conditional Edge** — A function that inspects state and decides which node runs next.
- **Checkpointer** — Saves state after every node so you can pause, resume, or time-travel.
- **thread_id** — A unique ID for each conversation, used to find the right checkpoint.
- **interrupt_before** — Tells the graph to pause before a specific node runs (enables HIL).
- **invoke(None)** — "Continue from where you paused" (resume after interrupt).
- **update_state()** — Write new values into a saved checkpoint (for HIL modifications).
- **get_state()** — Read the current frozen state (see what's pending).
- **HumanMessage** — A message from the user.
- **AIMessage** — A message from the LLM (may contain text or tool_calls).
- **ToolMessage** — A message containing a tool's result (links back via tool_call_id).
- **tool_calls** — A list inside AIMessage saying "I want to call these tools with these args".
- **tool_call_id** — Unique ID that links a ToolMessage back to the specific tool_call that produced it.

---

## Syntax Reference — How To Build a LangGraph Agent (Step by Step)

This is the complete recipe. Every LangGraph program follows these steps.

---

### Step 1: Define Your State

```python
from langgraph.graph import MessagesState

# Option A: Just use MessagesState (simplest — only messages)
# Use this when you only need conversation history.

# Option B: Extend it with custom fields
class MyState(MessagesState):
    destination: str
    budget: float
    order_status: str
```

---

### Step 2: Define Your Tools

```python
from langchain_core.tools import tool

@tool
def search_flights(destination: str) -> str:
    """Search for available flights to a destination."""
    return f"Flight to {destination}: $500"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

# Collect all tools into a list
tools = [search_flights, get_weather]
```

---

### Step 3: Initialize the LLM and Bind Tools

```python
from langchain.chat_models import init_chat_model

# Create the LLM
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

# Bind tools → tells the LLM "these are functions you can REQUEST"
llm_with_tools = llm.bind_tools(tools)
```

---

### Step 4: Define Your Nodes

```python
from langgraph.prebuilt import ToolNode

# Agent node — where the LLM lives
def agent_node(state: MyState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool node — prebuilt, executes tool functions
tool_node = ToolNode(tools)

# (Optional) Plain node — any custom logic
def setup_node(state: MyState):
    return {"destination": "Unknown", "budget": 0.0}
```

---

### Step 5: Define Conditional Edge (Routing Logic)

```python
from langgraph.graph import END

def should_continue(state: MyState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # LLM wants to use tools → go to tool node
    return END              # LLM gave final answer → stop
```

---

### Step 6: Build the Graph (Wire Everything Together)

```python
from langgraph.graph import StateGraph, START, END

# Create graph with your state type
workflow = StateGraph(MyState)

# Register nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set entry point (START → first node)
workflow.set_entry_point("agent")
# OR equivalently: workflow.add_edge(START, "agent")

# Add conditional edge (decision point after agent)
workflow.add_conditional_edges("agent", should_continue)
# OR with explicit mapping:
# workflow.add_conditional_edges("agent", should_continue, {END: END, "tools": "tools"})

# Add fixed edge (after tools → always go back to agent)
workflow.add_edge("tools", "agent")
```

---

### Step 7: Compile the Graph

```python
# Without checkpointer (no persistence, no HIL)
app = workflow.compile()

# With checkpointer (enables persistence, HIL, time travel)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# With HIL (pause before a node)
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]    # pause before tools run
)
```

---

### Step 8: Run the Graph

```python
from langchain_core.messages import HumanMessage

# Basic invoke (no persistence)
result = app.invoke({"messages": [HumanMessage(content="Hello")]})
print(result["messages"][-1].content)

# With thread_id (persistence enabled)
config = {"configurable": {"thread_id": "my-conversation-1"}}
result = app.invoke(
    {"messages": [HumanMessage(content="Book a flight to Tokyo")]},
    config=config
)

# Stream (see each node as it executes)
for chunk in app.stream(
    {"messages": [HumanMessage(content="Book a flight")]},
    config=config,
    stream_mode="updates"
):
    node_name = list(chunk.keys())[0]
    print(f"Node executed: {node_name}")
```

---

### Step 9: HIL — Pause, Inspect, Resume

```python
# After graph pauses (interrupt_before=["tools"]):

# Read the frozen state
frozen = app.get_state(config)
print(frozen.next)                              # ('tools',)
print(frozen.values["messages"][-1].tool_calls) # what tool the LLM wants

# Level 1: Just approve — resume as-is
result = app.invoke(None, config=config)

# Level 2: Reject + give feedback
from langchain_core.messages import HumanMessage
app.update_state(
    config,
    {"messages": [HumanMessage(content="No, search for cheaper flights")]},
    as_node="human"
)
result = app.invoke(None, config=config)

# Level 3: Rewrite the tool args
from langchain_core.messages import AIMessage
original = frozen.values["messages"][-1]
app.update_state(
    config,
    {"messages": [AIMessage(
        id=original.id,
        content="",
        tool_calls=[{
            "name": "search_flights",
            "args": {"destination": "Osaka"},   # changed from Tokyo to Osaka
            "id": original.tool_calls[0]["id"]
        }]
    )]},
    as_node="agent"
)
result = app.invoke(None, config=config)
```

---

### Step 10: Generate a Diagram

```python
# Save as PNG (requires graphviz or mermaid)
app.get_graph().draw_mermaid_png(output_file_path="my_agent_flow.png")
```

---

### Complete Minimal Example (Copy-Paste Starter)

```python
"""Minimal LangGraph agent — copy this as your starting template."""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# 1. Tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

tools = [multiply]

# 2. LLM + bind tools
llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Nodes
def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

# 4. Routing
def route(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# 5. Graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route)
graph.add_edge("tools", "agent")
app = graph.compile()

# 6. Run
result = app.invoke({"messages": [HumanMessage(content="What is 7 * 6?")]})
print(result["messages"][-1].content)
```
