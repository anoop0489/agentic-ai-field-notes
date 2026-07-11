# 15. LangGraph Prerequisites — Modern LangChain Patterns You Need First 🧱

*Pre-requisite knowledge for LangGraph, sourced from official LangChain docs (2025) + validated independently*

> ⚠️ **Why this exists:** The earlier sections (1-12) cover LangChain fundamentals, but LangGraph introduces newer patterns — `ToolRuntime`, `StateGraph`, `Command`, `init_chat_model`, middleware, and more — that are worth learning separately. This doc bridges the gap.

---

## 📋 Table of Contents

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 1 | [LangGraph Overview](#1-langgraph-overview) | What LangGraph is, why it exists, how it relates to LangChain |
| 2 | [StateGraph & MessagesState](#2-stategraph--messagesstate) | The core graph primitives — nodes, edges, state |
| 3 | [init_chat_model — Universal Model Factory](#3-init_chat_model--universal-model-factory) | Provider-agnostic model initialization, parameters, retries |
| 4 | [Modern Tool Patterns](#4-modern-tool-patterns) | `@tool` decorator, reserved argument names, advanced schemas |
| 5 | [ToolRuntime — Access Context](#5-toolruntime--access-context) | State, context, store, stream writer, execution info |
| 6 | [Tool Return Values & Command](#6-tool-return-values--command) | String, object, multimodal, `Command` for state updates |
| 7 | [Structured Output](#7-structured-output) | `with_structured_output`, Pydantic, TypedDict, JSON Schema |
| 8 | [Middleware & Dynamic Tool Selection](#8-middleware--dynamic-tool-selection) | `wrap_model_call`, `wrap_tool_call`, filtering tools at runtime |
| 9 | [Streaming & Batch](#9-streaming--batch) | `.stream()`, `.astream_events()`, `.batch()`, chunk accumulation |
| 10 | [Persistence & Checkpointers](#10-persistence--checkpointers) | Durable execution, `thread_id`, resume after failures |
| 11 | [Deep Dive: thread_id & RunnableConfig](#11-deep-dive-thread_id--runnableconfig) | What thread_id really is, the full config dict, when to reuse |
| 12 | [Conditional Edges & Routing](#12-conditional-edges--routing) | Dynamic paths, routing functions, branching logic |
| 13 | [The Agent Loop & bind_tools](#13-the-agent-loop--bind_tools) | The full Model → tool_calls → execute → ToolMessage cycle |
| 14 | [Python Syntax for C# Devs](#14-python-syntax-for-c-devs) | `Annotated`, `TypedDict`, `dataclass`, `Literal` — decoded |
| 15 | [create_react_agent / create_agent](#15-create_react_agent--create_agent) | The high-level prebuilt agent factories |
| 16 | [Graph Compilation & Recursion Limits](#16-graph-compilation--recursion-limits) | What `.compile()` does, preventing infinite loops |
| 17 | [add_messages Advanced Behavior](#17-add_messages-advanced-behavior) | Deduplication by ID, `RemoveMessage`, message management |
| 18 | [ToolNode — Prebuilt Tool Executor](#18-toolnode--prebuilt-tool-executor) | The prebuilt node that executes tool calls automatically |
| 19 | [Node Types in a Graph](#19-node-types-in-a-graph) | Model node vs tool node vs custom node — roles and responsibilities |
| 20 | [with_config & Runnable Binding](#20-with_config--runnable-binding) | Attaching config to runnables, how config flows through graphs |
| 21 | [System Instructions in Graphs](#21-system-instructions-in-graphs) | SystemMessage, prompt templates, agent identity |
| 22 | [Graph Streaming](#22-graph-streaming) | `stream_mode`, node updates vs token streaming, frontend patterns |
| 23 | [Time Travel & State History](#23-time-travel--state-history) | `get_state_history`, replaying, forking conversations |
| 24 | [Human-in-the-Loop Deep Dive](#24-human-in-the-loop-deep-dive) | `interrupt_before/after`, `get_state`, `update_state`, approval workflows |
| 25 | [Interview Q&A Anchors](#25-interview-qa-anchors) | Quick-fire answers for all prerequisites |

---

## 📚 Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|-----------------|
| **LangGraph** | Orchestration runtime for stateful agents | A low-level framework for building long-running, stateful workflows with persistence, human-in-the-loop, and streaming. Built on top of LangChain but usable independently. |
| **StateGraph** | A directed graph where nodes transform shared state | The core LangGraph primitive — you define nodes (functions), edges (transitions), and a shared state schema. Think of it as a state machine for AI workflows. |
| **MessagesState** | Pre-built state with a `messages` list | A convenience TypedDict with `messages: Annotated[list, add_messages]` — the default state for chatbot-style graphs. |
| **ToolRuntime** | Injected context object for tools | A parameter automatically injected into tools that provides access to state, context, store, stream writer, and execution info — hidden from the LLM schema. |
| **Command** | A tool return type that updates graph state | Allows tools to write back to graph state (e.g., set user preferences) rather than just returning text to the model. |
| **init_chat_model** | Provider-agnostic model factory | A single function that initializes any chat model from any provider using `"provider:model_name"` syntax. |
| **Middleware** | Hooks that wrap model/tool calls | Functions decorated with `@wrap_model_call` or `@wrap_tool_call` that intercept and modify requests/responses — used for dynamic tool selection, error handling, caching. |
| **Checkpointer** | Persistence layer for graph state | Saves graph state after each node execution so workflows can resume after crashes, handle long-running tasks, and support human-in-the-loop interrupts. |
| **thread_id** | Conversation scope identifier | A unique ID passed in `config["configurable"]["thread_id"]` that scopes a conversation's message history and checkpoints. Like a session ID. |
| **Reducer** | Conflict resolution for state updates | A function that determines how to merge concurrent updates to the same state field (e.g., `add_messages` appends instead of replacing). |
| **RunnableConfig** | The universal configuration dict | A dictionary passed to `.invoke()` containing `configurable` (thread_id, model), `tags`, `metadata`, `callbacks`, `max_concurrency`, and `recursion_limit`. |
| **Conditional Edge** | A routing function on graph edges | A function that examines current state and returns the name of the next node to execute — enables branching/routing. |
| **bind_tools** | Attach tools to a model | Makes the model aware of available tools so it can generate `tool_calls` in its response. |
| **create_react_agent** | Pre-built ReAct agent factory | A convenience function that builds a complete StateGraph with model node, tool node, conditional routing, and the full agent loop — no manual graph wiring needed. |
| **Annotated** | Python type hint + metadata | Python's way of attaching extra information (like a reducer function) to a type hint. LangGraph uses it to know *how* to update a field. |
| **recursion_limit** | Max graph cycles before forced stop | Prevents infinite agent loops — the graph stops if it exceeds this many node executions (default: 25). |
| **ToolNode** | Prebuilt node that executes tool calls | A LangGraph class that reads `tool_calls` from the last `AIMessage`, executes the matching tools, and returns `ToolMessage` objects — handles the entire "execute tools" step in one line. |
| **with_config** | Bind config to a runnable | Attaches default configuration (tags, metadata, run_name) to any runnable so it flows automatically through the chain without passing it at invocation time. |
| **Model Node** | The node that calls the LLM | In a graph, the node that invokes the chat model with the current messages and returns an `AIMessage` (potentially with `tool_calls`). |

---

## 1. LangGraph Overview

### What Is LangGraph?

LangGraph is a **low-level orchestration runtime** for building stateful, long-running AI agents and workflows. It sits *above* LangChain (which provides model/tool integrations) but *below* high-level agent harnesses.

```
┌─────────────────────────────────────────────────────┐
│  Deep Agents SDK (Harness — planning, subagents)    │  ← Highest level
├─────────────────────────────────────────────────────┤
│  LangGraph (Runtime — state, persistence, HIL)      │  ← This section
├─────────────────────────────────────────────────────┤
│  LangChain (Framework — models, tools, agents)      │  ← Sections 1-12
├─────────────────────────────────────────────────────┤
│  LangSmith (Platform — tracing, eval, deploy)       │  ← Observability
└─────────────────────────────────────────────────────┘
```

### Why LangGraph Exists (Problems It Solves)

| Problem | LangChain Alone | LangGraph Solution |
|---------|----------------|-------------------|
| Agent crashes mid-task | Lost all progress, start over | **Checkpointers** persist state — resume from last node |
| Need human approval before action | Not supported natively | **Interrupts** pause graph, wait for human, resume |
| Agent loop runs forever | Hard to detect/stop | **Recursion limits** + state inspection at each step |
| Multiple agents collaborating | Manual orchestration | **Subgraphs** — compose agents as nodes in a parent graph |
| Conversation memory across sessions | Manual DB integration | **Thread-scoped persistence** — automatic with `thread_id` |
| Need different logic paths | Simple if/else in chains | **Conditional edges** — route based on state |

### C# Analogy

Think of LangGraph as **Durable Functions (Azure)** or **Temporal.io workflows** for AI:
- Each **node** = an activity/step function
- The **state** = the workflow context object
- **Checkpointers** = the durable execution storage
- **Interrupts** = the `WaitForExternalEvent` / human approval pattern
- **Conditional edges** = the orchestrator's `switch` logic

### The Hello World

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
	return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

---

## 2. StateGraph & MessagesState

### StateGraph — The Core Primitive

A `StateGraph` is a directed graph where:
- **Nodes** are Python functions that receive state and return state updates
- **Edges** define the execution flow between nodes
- **State** is a shared TypedDict that flows through the entire graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Step 1: Define your state schema
class MyState(TypedDict):
	messages: Annotated[list, add_messages]  # Reducer: append, don't replace
	user_name: str                            # Custom field

# Step 2: Define node functions (receive state, return partial updates)
def greet(state: MyState):
	name = state.get("user_name", "stranger")
	return {"messages": [{"role": "ai", "content": f"Hello, {name}!"}]}

# Step 3: Build the graph
graph = StateGraph(MyState)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

# Step 4: Compile and run
app = graph.compile()
result = app.invoke({"messages": [], "user_name": "Anoop"})
```

### MessagesState — The Pre-Built Default

For chatbot-style agents, LangGraph provides `MessagesState` so you don't have to define the messages field yourself:

```python
from langgraph.graph import MessagesState

# This is equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
```

### Reducers — Why `Annotated[list, add_messages]`?

A **reducer** tells LangGraph how to merge a node's return value into existing state. Without it, returning `{"messages": [...]}` would *replace* the entire list. With `add_messages`, it *appends*.

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `add_messages` | Appends new messages to existing list | Chat history (default) |
| None (no annotation) | Replaces the field entirely | Simple values like `user_name` |
| Custom function | Your logic for merging | Counters, sets, complex objects |

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| `StateGraph` | A state machine (`Stateless` library) or Durable Functions orchestrator |
| `TypedDict` state | A POCO/DTO class with public properties |
| Node function | An Activity function in Durable Functions |
| `add_messages` reducer | `List<T>.AddRange()` vs assignment |
| `START` / `END` | Entry point / terminal state |
| Conditional edge | `switch` statement in orchestrator |

---

## 3. init_chat_model — Universal Model Factory

### The Problem It Solves

Before `init_chat_model`, switching providers meant changing imports and class names:

```python
# Old way — tightly coupled to provider
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatOpenAI(model="gpt-4o")  # ← Locked to OpenAI
```

### The Modern Way

```python
from langchain.chat_models import init_chat_model

# Provider-agnostic — switch by changing a string
model = init_chat_model("gpt-4o")                          # OpenAI (auto-detected)
model = init_chat_model("anthropic:claude-sonnet-4-6")     # Anthropic
model = init_chat_model("google_genai:gemini-2.5-flash")   # Google
model = init_chat_model("ollama:qwen3:1.7b")              # Local Ollama
```

### Key Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `model` | str | Model name, optionally prefixed with `"provider:"` |
| `temperature` | float | Randomness (0 = deterministic, 1 = creative) |
| `max_tokens` | int | Maximum output length |
| `timeout` | int | Seconds before request times out |
| `max_retries` | int | Retry count for failures (default: 6) |
| `api_key` | str | Provider API key (usually from env var) |

### Connection Resilience (Built-In Retries)

LangChain chat models automatically retry with exponential backoff:

```python
model = init_chat_model(
	"gpt-4o",
	max_retries=10,   # Increase for unreliable networks (default: 6)
	timeout=120,      # Seconds before timeout
)
```

**What gets retried:** Network errors, rate limits (429), server errors (5xx)
**What doesn't:** Client errors (401 unauthorized, 404 not found)

### Configurable Models (Runtime Provider Switching)

```python
# No model specified at creation — pick at runtime
configurable_model = init_chat_model(temperature=0)

# Same code, different model each time
configurable_model.invoke("hello", config={"configurable": {"model": "gpt-4o"}})
configurable_model.invoke("hello", config={"configurable": {"model": "claude-sonnet-4-6"}})
```

### C# Analogy

`init_chat_model` is like a **factory pattern** with dependency injection:

```csharp
// C# equivalent concept
IChatModel model = ChatModelFactory.Create("openai:gpt-4o", new ModelOptions {
	Temperature = 0,
	MaxRetries = 10,
	Timeout = TimeSpan.FromSeconds(120)
});
```

Or like `HttpClientFactory` — you don't `new HttpClient()` directly, you use the factory for configuration, resilience, and lifecycle management.

---

## 4. Modern Tool Patterns

### The `@tool` Decorator

The standard way to create tools in LangChain:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
	"""Get the weather at a location."""  # ← Docstring becomes the tool description
	return f"It's sunny in {location}."
```

**Key rules:**
- Type hints are **required** (they define the input schema the LLM sees)
- The docstring is the tool description the model reads to decide when to use it
- Prefer `snake_case` names (some providers reject spaces/special chars)

### Custom Name & Description

```python
@tool("web_search", description="Search the internet for current information. Use for any factual queries.")
def search(query: str) -> str:
	"""Search the web."""
	return f"Results for: {query}"
```

### Advanced Schema with Pydantic

For complex inputs with validation and descriptions:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
	"""Input for weather queries."""
	location: str = Field(description="City name or coordinates")
	units: Literal["celsius", "fahrenheit"] = Field(default="celsius")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius") -> str:
	"""Get current weather."""
	return f"22°C in {location}"
```

### ⚠️ Reserved Argument Names

These parameter names **cannot** be used as tool arguments — they cause runtime errors:

| Reserved Name | Why It's Reserved |
|--------------|-------------------|
| `config` | Used internally to pass `RunnableConfig` to tools |
| `runtime` | Reserved for `ToolRuntime` parameter (state, context, store access) |

```python
# ❌ BAD — will crash at runtime
@tool
def bad_tool(config: str) -> str:  # "config" is reserved!
	return config

# ✅ GOOD — use a different name
@tool
def good_tool(configuration: str) -> str:
	return configuration
```

### C# Analogy

| LangChain Tool | C# Equivalent |
|---------------|---------------|
| `@tool` decorator | `[Description("...")]` attribute on a method |
| `args_schema` (Pydantic) | A DTO class with `[Required]` and `[Description]` attributes |
| Reserved `config`/`runtime` | Like `HttpContext` in ASP.NET — injected by the framework, not by the caller |
| Tool name | The method name registered in a DI container |

---

## 5. ToolRuntime — Access Context

### What Is ToolRuntime?

`ToolRuntime` is an **automatically injected** parameter that gives your tool access to the execution context. The LLM never sees it — it's hidden from the schema.

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(query: str, runtime: ToolRuntime) -> str:
	"""Search something."""
	# LLM only sees: my_tool(query: str)
	# But your code has access to runtime.state, runtime.context, etc.
	return f"Results for {query}"
```

### What ToolRuntime Provides

```
┌────────────────────────────────────────────────────────────┐
│                      ToolRuntime                             │
├────────────────────────────────────────────────────────────┤
│  .state            → Short-term memory (current messages,   │
│                      custom fields for this conversation)   │
│  .context          → Immutable config (user_id, session)    │
│  .store            → Long-term memory (persists across      │
│                      conversations — like a database)        │
│  .stream_writer    → Emit real-time progress updates        │
│  .execution_info   → thread_id, run_id, attempt number     │
│  .server_info      → Assistant/graph ID (LangGraph Server)  │
│  .tool_call_id     → Unique ID for this specific call       │
│  .config           → Full RunnableConfig                    │
└────────────────────────────────────────────────────────────┘
```

### 5.1 State (Short-Term Memory)

State is the **current conversation's data** — messages, custom fields, counters. It exists only for the duration of a thread.

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_message_count(runtime: ToolRuntime) -> str:
	"""Get the number of messages in the conversation."""
	messages = runtime.state["messages"]
	return f"There are {len(messages)} messages in this conversation."

@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
	"""Get a user preference from the current session state."""
	preferences = runtime.state.get("user_preferences", {})
	return preferences.get(pref_name, "Not set")
```

### 5.2 Context (Immutable Per-Run Configuration)

Context carries **per-invocation data** like user ID, session info, or permissions. It's set when you call `.invoke()` and cannot be changed during execution.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
	user_id: str
	role: str  # "admin", "editor", "viewer"

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
	"""Get the current user's account information."""
	user_id = runtime.context.user_id  # ← Typed access
	role = runtime.context.role
	return f"User {user_id} with role {role}"

# When invoking the agent:
# agent.invoke(..., context=UserContext(user_id="abc123", role="admin"))
```

**Key distinction:**
- `thread_id` scopes the **conversation** (message history, checkpoints)
- `context` carries **per-run data** your tools read at invocation time

### 5.3 Store (Long-Term Memory)

Store is **persistent storage** that survives across conversations. Unlike state (which is per-thread), store data persists indefinitely.

```python
from langchain.tools import tool, ToolRuntime

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
	"""Save a user preference that persists across conversations."""
	store = runtime.store
	user_id = runtime.context.user_id

	# Namespace pattern: ("users", user_id) + key
	store.put(("preferences", user_id), key, {"value": value})
	return f"Saved {key}={value} for future conversations."

@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
	"""Retrieve a previously saved user preference."""
	store = runtime.store
	user_id = runtime.context.user_id

	item = store.get(("preferences", user_id), key)
	return item.value["value"] if item else "No preference saved."
```

| Memory Type | Scope | Lifetime | Analogy |
|-------------|-------|----------|---------|
| **State** | Per-thread (conversation) | Until thread is deleted | `HttpContext.Items` / session state |
| **Context** | Per-invocation | Single request | `HttpContext.User` / JWT claims |
| **Store** | Global (cross-thread) | Persistent (until deleted) | Database / Redis cache |

### 5.4 Stream Writer (Real-Time Progress)

Emit updates to the client while a tool is executing:

```python
@tool
def long_running_search(query: str, runtime: ToolRuntime) -> str:
	"""Search multiple databases (takes time)."""
	writer = runtime.stream_writer

	writer("Searching primary database...")
	# ... actual work ...
	writer("Searching secondary database...")
	# ... actual work ...
	writer("Compiling results...")

	return "Found 42 matching documents."
```

### 5.5 Execution Info

Access thread/run identity and retry state:

```python
@tool
def debug_tool(runtime: ToolRuntime) -> str:
	"""Log execution identity."""
	info = runtime.execution_info
	return f"Thread: {info.thread_id}, Run: {info.run_id}, Attempt: {info.node_attempt}"
```

### C# Analogy for ToolRuntime

| ToolRuntime Component | C# Equivalent |
|----------------------|---------------|
| `runtime.state` | `HttpContext.Items` or scoped service state |
| `runtime.context` | `HttpContext.User.Claims` (JWT claims, injected per-request) |
| `runtime.store` | `IDistributedCache` / EF Core `DbContext` |
| `runtime.stream_writer` | `IServerSentEventsWriter` / SignalR `IHubContext.SendAsync()` |
| `runtime.execution_info` | `Activity.Current` (OpenTelemetry trace/span info) |
| `runtime.tool_call_id` | Correlation ID from a message broker |

---

## 6. Tool Return Values & Command

### Return a String (Most Common)

The model sees this as text and decides what to do next:

```python
@tool
def get_weather(city: str) -> str:
	"""Get weather for a city."""
	return f"It is currently sunny in {city}."
# Model sees: ToolMessage(content="It is currently sunny in Boston.")
```

### Return an Object (Structured Data)

The model can reason over specific fields:

```python
@tool
def get_weather_data(city: str) -> dict:
	"""Get structured weather data."""
	return {"city": city, "temperature_c": 22, "conditions": "sunny"}
```

### Return Multimodal Content

Tools can return images, audio, etc. to vision-capable models:

```python
@tool
def capture_screenshot() -> list[dict]:
	"""Capture a screenshot."""
	return [
		{"type": "text", "text": "Screenshot of the current page:"},
		{"type": "image", "url": "https://example.com/page.png"},
	]
```

### Return a Command (Update Graph State)

**This is the key new pattern.** When a tool needs to *write back* to the graph's state (not just return text to the model), it returns a `Command`:

```python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
	"""Set the user's preferred response language."""
	return Command(
		update={
			"preferred_language": language,       # ← Updates graph state field
			"messages": [
				ToolMessage(
					content=f"Language set to {language}.",
					tool_call_id=runtime.tool_call_id,  # ← Required correlation ID
				)
			],
		}
	)
```

**Why Command exists:**
- Normal string returns → model sees text, state unchanged
- Command → state is mutated (e.g., setting preferences, flags, counters)
- The `ToolMessage` inside `Command` is optional but recommended so the model knows the action succeeded

### Return Direct (Skip Model Post-Processing)

When the tool's output IS the final answer — no further LLM reasoning needed:

```python
@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
	"""Fetch the current status of a customer order."""
	return f"Order {order_id} is shipped and will arrive in 2 days."
# Agent returns this directly to the user — no extra LLM call
```

**Use when:** The tool output is the complete, user-ready answer (e.g., database lookup).
**Don't use when:** The result needs summarization, further reasoning, or chaining.

### C# Analogy

| Tool Return | C# Equivalent |
|-------------|---------------|
| String return | An endpoint returning `Ok("result")` |
| Dict return | An endpoint returning `Ok(new { temp = 22 })` |
| `Command` | An endpoint that also writes to `HttpContext.Items` or dispatches a domain event |
| `return_direct=True` | A short-circuit middleware that returns a cached response without hitting the controller |

---

## 7. Structured Output

### The Problem

LLMs return free-form text. But you often need **structured data** (JSON matching a schema):

```python
# Without structured output:
response = model.invoke("Extract the movie title and year from: 'Inception came out in 2010'")
# → "The movie is Inception and it was released in 2010."  ← Free text, hard to parse
```

### The Solution: `with_structured_output`

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
	"""A movie with details."""
	title: str = Field(description="The title of the movie")
	year: int = Field(description="The year released")
	director: str = Field(description="The director")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Tell me about Inception")
# → Movie(title="Inception", year=2010, director="Christopher Nolan")  ← Typed object!
```

### Three Schema Formats

| Format | Best For | Validation |
|--------|----------|-----------|
| **Pydantic BaseModel** | Production code — rich validation, descriptions, nesting | Automatic |
| **TypedDict** | Simple cases — no runtime validation needed | Manual |
| **JSON Schema** (dict) | Max control, interoperability with other systems | Manual |

### Methods (How It Works Under the Hood)

| Method | How It Constrains | Provider Support |
|--------|-------------------|-----------------|
| `json_schema` | Provider's native structured output feature | OpenAI, Anthropic, Google |
| `function_calling` | Forces a tool call matching the schema | Most providers |
| `json_mode` | Generates valid JSON (schema in prompt) | Legacy — less reliable |

### C# Analogy

`with_structured_output` is like using `System.Text.Json` deserialization with a schema contract:

```csharp
// C# equivalent concept
var movie = await chatClient.GetStructuredResponseAsync<Movie>(
	"Tell me about Inception",
	JsonSchema.FromType<Movie>()
);
// movie.Title → "Inception", movie.Year → 2010
```

Or in Semantic Kernel: `kernel.InvokeAsync<Movie>(...)` with a schema constraint.

---

## 8. Middleware & Dynamic Tool Selection

### What Is Middleware?

Middleware are **hooks** that wrap model calls or tool calls, letting you intercept and modify behavior. Two types:

| Decorator | Wraps | Use Cases |
|-----------|-------|-----------|
| `@wrap_model_call` | The LLM invocation | Dynamic tool filtering, model switching, logging |
| `@wrap_tool_call` | Tool execution | Error handling, retries, authorization |

### wrap_model_call — Dynamic Tool Selection

Filter which tools the model sees based on state, permissions, or context:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def permission_filter(request: ModelRequest, handler) -> ModelResponse:
	"""Only show admin tools to admin users."""
	user_role = request.runtime.context.role

	if user_role != "admin":
		# Filter out admin-only tools
		safe_tools = [t for t in request.tools if not t.name.startswith("admin_")]
		request = request.override(tools=safe_tools)

	return handler(request)  # ← Pass to next middleware or model
```

### wrap_tool_call — Error Handling

Catch tool exceptions and return friendly messages:

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

@wrap_tool_call
def handle_errors(request: ToolCallRequest, handler) -> ToolMessage:
	"""Convert exceptions into ToolMessages the model can handle."""
	try:
		return handler(request)
	except Exception as e:
		return ToolMessage(
			content=f"Tool error: {e}. Please try a different approach.",
			tool_call_id=request.tool_call["id"],
		)
```

### Dynamic Model Selection

Switch models at runtime based on complexity:

```python
from langchain_openai import ChatOpenAI

basic = ChatOpenAI(model="gpt-4o-mini")
advanced = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def route_by_complexity(request: ModelRequest, handler) -> ModelResponse:
	"""Use advanced model for complex conversations."""
	if len(request.state["messages"]) > 10:
		return handler(request.override(model=advanced))
	return handler(request.override(model=basic))
```

### C# Analogy

| Middleware Concept | C# Equivalent |
|-------------------|---------------|
| `@wrap_model_call` | ASP.NET middleware pipeline / `DelegatingHandler` |
| `@wrap_tool_call` | Exception filter / `IAsyncActionFilter` |
| `handler(request)` | `next(context)` in the middleware pipeline |
| `request.override(...)` | Mutating `HttpContext` before passing downstream |
| Dynamic tool filtering | Authorization policy removing endpoints from route table |

---

## 9. Streaming & Batch

### Stream (Real-Time Token Output)

```python
# Synchronous streaming
for chunk in model.stream("Why do parrots talk?"):
	print(chunk.text, end="", flush=True)

# Async streaming
async for chunk in model.astream("Why do parrots talk?"):
	print(chunk.text, end="", flush=True)
```

### Chunk Accumulation

Chunks can be summed to reconstruct the full message:

```python
full = None
for chunk in model.stream("What color is the sky?"):
	full = chunk if full is None else full + chunk
# full is now equivalent to what .invoke() would return
```

### Streaming Events (Advanced)

Filter by event type for fine-grained control:

```python
async for event in model.astream_events("Hello"):
	if event["event"] == "on_chat_model_stream":
		print(event["data"]["chunk"].text, end="")
	elif event["event"] == "on_chat_model_end":
		print(f"\n\nFull: {event['data']['output'].text}")
```

### Batch (Parallel Requests)

```python
responses = model.batch([
	"Why do parrots talk?",
	"How do airplanes fly?",
	"What is quantum computing?"
])

# Control parallelism
responses = model.batch(
	list_of_inputs,
	config={"max_concurrency": 5}
)
```

### C# Analogy

| LangChain | C# Equivalent |
|-----------|---------------|
| `.stream()` | `IAsyncEnumerable<T>` / `await foreach` |
| `.astream()` | `IAsyncEnumerable<T>` with `async` context |
| Chunk accumulation (`full + chunk`) | `StringBuilder.Append()` pattern |
| `.batch()` | `Task.WhenAll(requests.Select(r => client.SendAsync(r)))` |
| `max_concurrency` | `SemaphoreSlim` / `Parallel.ForEachAsync(maxDegreeOfParallelism)` |
| `astream_events()` | Server-Sent Events / `IObservable<T>` (Rx) |

---

## 10. Persistence & Checkpointers

### The Problem

Without persistence, if an agent crashes mid-task or needs human approval, all progress is lost.

### The Solution: Checkpointers

A **checkpointer** saves the graph's state after each node execution:

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory (development/testing)
checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)

# Every invocation needs a thread_id to scope the conversation
result = app.invoke(
	{"messages": [{"role": "user", "content": "hello"}]},
	config={"configurable": {"thread_id": "conversation-123"}}
)
```

### thread_id — Conversation Scope

`thread_id` is how LangGraph knows which conversation you're continuing:

```python
# First turn
app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})

# Second turn — same thread_id, so it has the previous messages
app.invoke({"messages": [{"role": "user", "content": "follow up"}]},
		   config={"configurable": {"thread_id": "abc"}})

# Different conversation — different thread_id
app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "xyz"}})
```

### Production Checkpointers

| Checkpointer | Use Case |
|--------------|----------|
| `MemorySaver` | Development, testing (lost on restart) |
| `PostgresSaver` | Production — durable, scalable |
| `SqliteSaver` | Single-machine production |

### Human-in-the-Loop (Interrupts)

Checkpointers enable **interrupts** — pausing a graph to wait for human input:

```python
# Graph pauses before "dangerous_action" node
app = graph.compile(
	checkpointer=checkpointer,
	interrupt_before=["dangerous_action"]
)

# Invoke — graph runs until the interrupt point, then stops
result = app.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})
# result.status → "interrupted"

# Human reviews, then resumes
app.invoke(None, config={"configurable": {"thread_id": "abc"}})  # None = continue
```

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| Checkpointer | Azure Durable Functions `IDurableOrchestrationContext` |
| `thread_id` | `instanceId` in Durable Functions |
| `MemorySaver` | In-memory state (test double) |
| `PostgresSaver` | Azure Table Storage / SQL persistence provider |
| `interrupt_before` | `WaitForExternalEvent("approval")` in Durable Functions |
| Resume after interrupt | `RaiseEventAsync("approval", payload)` |

---

## 11. Deep Dive: thread_id & RunnableConfig

### What Exactly Is `thread_id`?

`thread_id` is a **string you generate** (typically a UUID) that scopes a conversation. It answers: "Which conversation are we continuing?"

```python
import uuid

# YOU generate it — it's not magic
thread_id = str(uuid.uuid4())  # e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Pass it on every invocation
config = {"configurable": {"thread_id": thread_id}}
result = app.invoke({"messages": [...]}, config=config)
```

### Is It Hardcoded?

**No.** It's dynamic — you create a new one per conversation:

| Scenario | How You Get `thread_id` |
|----------|------------------------|
| New conversation starts | Generate with `uuid.uuid4()` or `langchain_core.utils.uuid.uuid7()` |
| User sends follow-up | Reuse the **same** `thread_id` from their session |
| Different user | Different `thread_id` (per user + per conversation) |
| Testing/development | You can hardcode `"test-thread-1"` for reproducibility |
| Web app | Store in browser session / cookie, send on each request |

### When to Reuse vs Create New

```python
# Scenario 1: Multi-turn chat (SAME thread_id)
config = {"configurable": {"thread_id": "user-123-session-1"}}
app.invoke({"messages": [{"role": "user", "content": "What's the weather?"}]}, config=config)
app.invoke({"messages": [{"role": "user", "content": "And tomorrow?"}]}, config=config)
# ↑ Second call has full history because same thread_id

# Scenario 2: New conversation (NEW thread_id)
config = {"configurable": {"thread_id": "user-123-session-2"}}
app.invoke({"messages": [{"role": "user", "content": "Unrelated question"}]}, config=config)
# ↑ Starts fresh — no history from session-1
```

### The Full `RunnableConfig` Dictionary

`thread_id` is just one field inside the larger `RunnableConfig`. Here's everything you can pass:

```python
config = {
    # Required for persistence (checkpointers)
    "configurable": {
        "thread_id": "conversation-abc",   # Scopes the conversation
        "model": "gpt-4o",                 # For configurable models
        "model_provider": "openai",        # Provider override
    },

    # Optional but useful
    "tags": ["production", "user-123"],    # For LangSmith filtering
    "metadata": {"user_id": "123", "session_start": "2025-01-01"},  # Custom tracking
    "callbacks": [my_callback_handler],    # Logging, monitoring
    "max_concurrency": 5,                  # Limit parallel batch calls
    "recursion_limit": 30,                 # Max graph cycles (default: 25)
    "run_name": "weather_query",           # Name this run in traces
}
```

### RunnableConfig Explained Field by Field

| Field | Where It Goes | What It Does | Required? |
|-------|---------------|-------------|-----------|
| `configurable.thread_id` | Checkpointer | Scopes which conversation's state to load/save | Yes (if using checkpointer) |
| `configurable.model` | Configurable models | Picks which model to use at runtime | No |
| `tags` | All steps | Inherited labels for filtering in LangSmith | No |
| `metadata` | All steps | Custom key-value pairs for tracking | No |
| `callbacks` | All steps | Event handlers (logging, cost tracking) | No |
| `max_concurrency` | Batch/parallel | Max simultaneous operations | No |
| `recursion_limit` | Graph execution | Prevents infinite loops (default: 25) | No |
| `run_name` | This specific run | Label in traces (not inherited by sub-calls) | No |

### C# Analogy

| LangChain Config | C# Equivalent |
|-----------------|---------------|
| `thread_id` | Session ID / `HttpContext.Session.Id` |
| `configurable` | `IOptions<T>` / runtime settings from DI |
| `tags` | OpenTelemetry `Activity.Tags` |
| `metadata` | `Activity.Baggage` (propagated context) |
| `callbacks` | `ILogger` + middleware event hooks |
| `recursion_limit` | `CancellationToken` with timeout |

### Config vs ToolRuntime — How Are They Different?

This is a common source of confusion. Both carry "configuration" — but they serve **completely different purposes at different layers**.

**Simple analogy:**
- `config` = the **mailing address** on the envelope (tells the postal system WHERE to deliver)
- `ToolRuntime` = the **contents inside the envelope** (what the recipient reads and uses)

#### What Each One Is

| | **config (RunnableConfig)** | **ToolRuntime** |
|---|---|---|
| **What** | A dict you pass to `.invoke()` / `.stream()` | A parameter auto-injected INTO your tool function |
| **Who uses it** | The **framework** (LangGraph, checkpointer, LangSmith) | Your **tool code** (business logic inside the tool) |
| **Who creates it** | **You** (the developer calling the graph) | **LangGraph** (automatically, at tool execution time) |
| **Visible to LLM?** | No (it's infrastructure) | No (hidden from schema) |
| **Purpose** | Route, persist, trace, and scope the execution | Give tools access to state, memory, and streaming |

#### When Each One Is Used

```
YOUR CODE                         LANGGRAPH FRAMEWORK                    YOUR TOOL
─────────                         ──────────────────                    ─────────

app.invoke(                       Framework reads config:               @tool
  input,                            → thread_id → load checkpoints     def my_tool(query: str, runtime: ToolRuntime):
  config={                          → tags → attach to traces              # runtime.state → read messages
    "configurable": {               → metadata → send to LangSmith         # runtime.store → read/write long-term memory
      "thread_id": "abc"            → recursion_limit → set max loops      # runtime.context → read user_id, permissions
    },                                                                     # runtime.config → the SAME config dict!
    "tags": ["prod"]              Then framework INJECTS ToolRuntime        return f"Results for {query}"
  }                               into your tool with all the context
)                                 your tool needs to do its job.
```

#### The Key Difference in One Sentence

> **Config tells LangGraph HOW to run the graph. ToolRuntime tells your tool WHAT it can access while running.**

#### They're Connected (But Not the Same Thing)

`ToolRuntime` actually **contains** the config inside it (`runtime.config`). The framework reads your config, uses parts of it for routing/persistence, and then packages the relevant context into a `ToolRuntime` object for your tools.

```python
# You pass config at the TOP level:
config = {"configurable": {"thread_id": "abc"}, "tags": ["prod"]}
app.invoke(input, config=config)

# Inside your tool, runtime.config IS that same config:
@tool
def my_tool(query: str, runtime: ToolRuntime) -> str:
    thread = runtime.config["configurable"]["thread_id"]  # "abc"
    # But you'd typically use the higher-level accessors:
    thread = runtime.execution_info.thread_id              # "abc" (cleaner)
    return "..."
```

#### Comparison Table

| Aspect | config | ToolRuntime |
|--------|--------|-------------|
| **Layer** | Caller → Framework | Framework → Tool |
| **Direction** | You push it IN | LangGraph injects it FOR you |
| **Contains** | thread_id, tags, metadata, callbacks | state, context, store, stream_writer, config |
| **Analogy** | HTTP Request Headers | Dependency-injected services in a controller |
| **Lifetime** | Per `.invoke()` call | Per tool execution |
| **Without checkpointer** | Still works (tags, metadata flow) | Still works (state, context available) |
| **Without it** | Graph has no persistence, no traces | Tool can't read state or write to store |

#### C# Analogy (Side by Side)

```csharp
// CONFIG ≈ What you pass when making an HTTP request
var request = new HttpRequestMessage {
    Headers = {
        { "X-Session-Id", "abc-123" },     // ≈ thread_id
        { "X-Trace-Tag", "production" }     // ≈ tags
    }
};

// TOOLRUNTIME ≈ What your controller receives via DI
public class WeatherController : ControllerBase
{
    // These are INJECTED by the framework, not passed by the caller directly
    private readonly HttpContext _context;      // ≈ runtime.context
    private readonly ISession _session;         // ≈ runtime.state
    private readonly IDistributedCache _cache;  // ≈ runtime.store
    private readonly IHubContext _hub;          // ≈ runtime.stream_writer
}
```

#### When Do You Need Each?

| Scenario | What You Use |
|----------|--------------|
| "I want multi-turn memory" | `config` with `thread_id` + checkpointer |
| "My tool needs to read the current messages" | `ToolRuntime` → `runtime.state["messages"]` |
| "I want to filter traces in LangSmith" | `config` with `tags` |
| "My tool needs to save user preferences long-term" | `ToolRuntime` → `runtime.store` |
| "I want to pause and resume (HIL)" | `config` with same `thread_id` on resume |
| "My tool needs the user's ID for permissions" | `ToolRuntime` → `runtime.context` |
| "I want to limit graph loops" | `config` with `recursion_limit` |
| "My tool needs to stream progress to the UI" | `ToolRuntime` → `runtime.stream_writer` |

---

## 12. Conditional Edges & Routing

### The Problem

In a linear graph, every node always leads to the same next node. But agents need **branching**:
- If the model returned tool calls → go to tool execution node
- If the model gave a final answer → go to END
- If user input was flagged → go to escalation node

### How Conditional Edges Work

```python
from langgraph.graph import StateGraph, START, END

def router(state):
    """Look at the last message — did the model request tool calls?"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"       # ← Go to tool execution node
    return END               # ← Go to END (final answer)

graph = StateGraph(MyState)
graph.add_node("model", call_model)
graph.add_node("tools", execute_tools)

graph.add_edge(START, "model")
graph.add_conditional_edges("model", router)  # ← Branching!
graph.add_edge("tools", "model")              # ← After tools, go back to model
```

### The Routing Function Contract

A routing function:
1. **Receives** the current state (same as a node function)
2. **Returns** a string — the name of the next node to execute (or `END`)
3. **Does NOT modify** state — it's read-only

```python
def should_continue(state: MyState) -> str:
    """Decide next step based on state."""
    messages = state["messages"]
    last = messages[-1]

    # Option A: model wants to use tools
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"

    # Option B: model gave final answer
    return END
```

### Visual: The ReAct Agent Loop

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                         ▼
                ┌────────────────┐
           ┌───▶│   model_node   │◀──────────┐
           │    └────────┬───────┘           │
           │             │                    │
           │    (conditional edge)            │
           │     /              \             │
           │    ▼                ▼            │
           │  tool_calls?      no tool_calls │
           │    │                    │        │
           │    ▼                    ▼        │
           │  ┌──────────┐      ┌───────┐    │
           │  │tool_node │      │  END  │    │
           │  └────┬─────┘      └───────┘    │
           │       │                          │
           └───────┘  (always go back to model)
```

### Multiple Outputs (Routing Map)

You can provide a mapping for clarity:

```python
graph.add_conditional_edges(
    "model",
    router,
    {
        "tools": "tool_node",      # If router returns "tools" → go to tool_node
        "escalate": "human_node",  # If router returns "escalate" → go to human_node
        END: END,                  # If router returns END → finish
    }
)
```

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| `add_conditional_edges(node, func)` | `switch` in a Durable Functions orchestrator |
| Router function | A `Func<State, string>` that returns the next activity name |
| Routing map | A `Dictionary<string, string>` mapping logical names to actual nodes |

---

## 13. The Agent Loop & bind_tools

### The Full Tool-Calling Cycle

This is the most important loop to understand. Every agent follows this pattern:

```
User Question → Model (with tools bound) → 
    IF tool_calls: Execute tools → Feed results back → Model again
    IF no tool_calls: Return final answer to user
```

### Step by Step

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Define a tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72°F and sunny in {city}"

# 2. Create model and BIND tools to it
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([get_weather])  # ← Model now knows about the tool

# 3. Send a question
messages = [HumanMessage(content="What's the weather in Boston?")]
response = llm_with_tools.invoke(messages)

# 4. Model doesn't answer directly — it returns tool_calls
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_abc123'}]

# 5. YOU execute the tool
messages.append(response)  # Add the AIMessage with tool_calls
for tc in response.tool_calls:
    result = get_weather.invoke(tc)  # Execute it
    messages.append(result)          # ToolMessage with result

# 6. Send back to model — now it can answer with the data
final = llm_with_tools.invoke(messages)
print(final.content)
# "The weather in Boston is 72°F and sunny."
```

### What `bind_tools` Actually Does

`bind_tools` **doesn't change the model** — it creates a new model wrapper that:
1. Converts your Python tool definitions into JSON schemas
2. Passes those schemas to the provider API (OpenAI, Anthropic, etc.)
3. The provider API now knows the model CAN request these tools
4. The model decides IF and WHEN to call them based on the conversation

```python
# Without bind_tools — model can only respond with text
model = ChatOpenAI(model="gpt-4o")

# With bind_tools — model can also respond with tool_calls
model_with_tools = model.bind_tools([get_weather, search_docs])
# ↑ This is a NEW object. The original `model` is unchanged.
```

### The ToolMessage Contract

After executing a tool, you must send a `ToolMessage` back with:
- `content`: The tool's return value (string)
- `tool_call_id`: Must match the `id` from the tool call request

```python
# This correlation is CRITICAL — the model uses it to match results to requests
ToolMessage(
    content="72°F and sunny in Boston",
    tool_call_id="call_abc123"  # ← Same ID from response.tool_calls[0]["id"]
)
```

### Parallel Tool Calls

Models can request multiple tools simultaneously:

```python
# User: "What's the weather in Boston AND New York?"
# Model returns TWO tool_calls:
# [
#   {'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'city': 'New York'}, 'id': 'call_2'},
# ]

# You execute BOTH and append BOTH ToolMessages
for tc in response.tool_calls:
    result = get_weather.invoke(tc)
    messages.append(result)
# Then call model again — it has both results
```

### C# Analogy

| Agent Loop Step | C# Equivalent |
|----------------|---------------|
| `bind_tools` | Registering endpoints in a DI container / Swagger schema generation |
| Model with `tool_calls` | A controller returning `RedirectToAction("GetWeather", args)` |
| `ToolMessage` | The response from that redirected action, correlated by request ID |
| Parallel tool calls | `Task.WhenAll(toolCall1, toolCall2)` |
| The full loop | A `while(!done)` orchestrator pattern |

---

## 14. Python Syntax for C# Devs

### `TypedDict` — Like a C# Record/POCO

`TypedDict` is Python's way of defining a dictionary with known keys and types:

```python
from typing import TypedDict

# Python
class MyState(TypedDict):
    messages: list
    user_name: str
    score: int
```

```csharp
// C# equivalent
public record MyState(
    List<BaseMessage> Messages,
    string UserName,
    int Score
);
```

**Key difference:** A `TypedDict` is still a `dict` at runtime — you access fields with `state["messages"]`, not `state.messages`. The type hints are for tooling only.

### `Annotated` — Attaching Metadata to Types

`Annotated` adds extra information to a type hint without changing the type itself:

```python
from typing import Annotated

# Without Annotated — just a list
messages: list

# With Annotated — a list WITH a reducer function attached
messages: Annotated[list, add_messages]
#          ↑ type    ↑ metadata (tells LangGraph to APPEND, not REPLACE)
```

```csharp
// C# doesn't have a direct equivalent, but conceptually:
[Reducer(typeof(AddMessagesReducer))]  // ← Like an attribute on a property
public List<BaseMessage> Messages { get; set; }
```

**Why it matters:** LangGraph reads the `Annotated` metadata to know how to merge state updates. Without it, returning `{"messages": [new_msg]}` would REPLACE the entire list.

### `dataclass` — Like a C# Record with Auto-Properties

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str
    role: str = "viewer"  # Default value
```

```csharp
// C# equivalent
public record UserContext(string UserId, string Role = "viewer");
```

`dataclass` auto-generates `__init__`, `__repr__`, and `__eq__`. It's Python's version of a simple data container.

### `Literal` — Like a C# Enum Constraint

```python
from typing import Literal

def set_units(units: Literal["celsius", "fahrenheit"]) -> str:
    ...
```

```csharp
// C# equivalent
public enum Units { Celsius, Fahrenheit }
public string SetUnits(Units units) => ...;
```

### `Field` (Pydantic) — Like `[Required]` + `[Description]`

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    limit: int = Field(default=10, ge=1, le=100)
```

```csharp
// C# equivalent
public class SearchInput
{
    [Required, Description("The search query")]
    public string Query { get; set; }

    [Range(1, 100)]
    public int Limit { get; set; } = 10;
}
```

### `**kwargs` — Like `params object[]` or `Dictionary<string, object>`

```python
# Python accepts any keyword arguments
def init_model(**kwargs):
    temperature = kwargs.get("temperature", 0.7)
```

```csharp
// C# rough equivalent
void InitModel(Dictionary<string, object> kwargs) {
    var temperature = kwargs.GetValueOrDefault("temperature", 0.7);
}
```

### Summary Table

| Python | C# | Purpose |
|--------|-----|---------|
| `TypedDict` | `record` / POCO | Structured data container |
| `Annotated[T, metadata]` | `[Attribute] T` | Attach extra info to a type |
| `dataclass` | `record` with auto-props | Simple data class |
| `Literal["a", "b"]` | `enum { A, B }` | Restrict values |
| `Field(description=...)` | `[Description("...")]` | Metadata for validation/docs |
| `**kwargs` | `Dictionary<string, object>` | Arbitrary named arguments |
| `Optional[str]` / `str | None` | `string?` | Nullable type |
| `list[str]` | `List<string>` | Generic collection |

---

## 15. create_react_agent / create_agent

### The Problem

Building a graph manually requires:
1. Defining state
2. Creating a model node
3. Creating a tool node
4. Wiring conditional edges for the agent loop
5. Compiling with a checkpointer

That's ~30 lines of boilerplate for a standard agent. `create_react_agent` (now `create_agent`) does it all in one call.

### The High-Level Factory

```python
from langchain.agents import create_agent  # Modern API
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(model, tools=[get_weather])

# That's it. No StateGraph, no edges, no compile.
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in Boston?"}]},
    config={"configurable": {"thread_id": "session-1"}}
)
```

### What It Builds Internally

`create_agent` creates this graph for you:

```
START → model_node → [conditional] → tool_node → model_node → ... → END
                         ↓
                        END (if no tool_calls)
```

With:
- A `MessagesState` (or custom state if provided)
- A model node that calls `model.invoke(state["messages"])`
- A tool node that executes all requested tools
- A conditional edge that checks for `tool_calls`
- An optional checkpointer for persistence

### Customization Options

```python
agent = create_agent(
    model=model,
    tools=[get_weather, search_docs],
    system_prompt="You are a helpful assistant.",     # System message prepended
    middleware=[error_handler, permission_filter],   # Middleware hooks
    context_schema=UserContext,                      # Typed context injection
    store=InMemoryStore(),                           # Long-term memory
    checkpointer=MemorySaver(),                     # Persistence
)
```

### Legacy: `create_react_agent` (LangGraph < 2025)

Older code uses `create_react_agent` from `langgraph.prebuilt`:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(
    model,
    tools=[get_weather],
    checkpointer=MemorySaver(),
)
```

This is being superseded by `create_agent` from `langchain.agents`, which adds middleware, context, and store support.

### When to Use Factory vs Manual Graph

| Use Case | Approach |
|----------|----------|
| Standard ReAct agent (model + tools loop) | `create_agent` — no reason to build manually |
| Custom routing (multiple LLMs, branches) | Manual `StateGraph` |
| Multi-agent collaboration | Manual graph with subgraphs |
| Non-standard flow (parallel nodes, conditional tool sets) | Manual graph |
| Learning / interviews | Understand both! |

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| `create_agent(model, tools)` | `builder.AddControllers()` + automatic route registration in ASP.NET |
| Manual `StateGraph` | Manually defining a state machine with `Stateless` or Durable Functions |
| The factory doing all wiring | `WebApplication.CreateBuilder(args)` — convention-over-configuration |

---

## 16. Graph Compilation & Recursion Limits

### What Does `.compile()` Do?

When you call `graph.compile()`:

1. **Validates** — checks all edges point to existing nodes, no orphans
2. **Freezes** — the graph structure becomes immutable (no more `add_node`)
3. **Attaches checkpointer** — if provided, enables persistence
4. **Sets interrupt points** — if `interrupt_before` / `interrupt_after` specified
5. **Returns a runnable** — the compiled graph has `.invoke()`, `.stream()`, `.batch()`

```python
# graph.compile() → CompiledGraph (runnable)
app = graph.compile(
    checkpointer=MemorySaver(),          # Optional: persistence
    interrupt_before=["dangerous_node"],  # Optional: pause points
)

# Now you can invoke it
result = app.invoke({"messages": [...]}, config={...})
```

### Recursion Limit — Preventing Infinite Loops

If an agent keeps calling tools forever (model is confused, tool keeps failing), the graph needs a safety net:

```python
# Default: 25 cycles
result = app.invoke(
    {"messages": [...]},
    config={
        "configurable": {"thread_id": "abc"},
        "recursion_limit": 50,  # ← Override per-invocation
    }
)
```

**What counts as a "recursion"?** Each time execution passes through a node, the counter increments. A typical agent loop (model → tool → model → tool → model → END) uses 5 recursions.

**What happens when exceeded?** A `GraphRecursionError` is raised. In production, catch this and return a fallback response.

### C# Analogy

| LangGraph | C# Equivalent |
|-----------|---------------|
| `.compile()` | `app.Build()` in ASP.NET — validates and freezes the pipeline |
| `recursion_limit` | `CancellationTokenSource(TimeSpan)` / max loop iterations |
| `GraphRecursionError` | `TaskCanceledException` / `TimeoutException` |
| `interrupt_before` | Circuit breaker pattern / approval workflow step |

---

## 17. add_messages Advanced Behavior

### Basic Behavior (What We Already Know)

`add_messages` is a reducer that **appends** new messages instead of replacing:

```python
# State before: messages = [msg1, msg2]
# Node returns: {"messages": [msg3]}
# State after:  messages = [msg1, msg2, msg3]  ← Appended, not replaced
```

### Deduplication by ID

If a message has the same `id` as an existing message, `add_messages` **replaces** instead of appending:

```python
from langchain_core.messages import AIMessage

# State has: [AIMessage(id="msg-1", content="Hello")]
# Node returns: [AIMessage(id="msg-1", content="Hello (corrected)")]
# Result: The existing message is REPLACED (same id = update in place)
```

**Why this matters:** When streaming, the model sends chunks that accumulate. If you re-process, you don't want duplicates.

### RemoveMessage — Deleting Messages from State

LangGraph has a special message type for removing messages from state:

```python
from langchain_core.messages import RemoveMessage

def trim_old_messages(state):
    """Remove all but the last 10 messages."""
    messages = state["messages"]
    if len(messages) > 10:
        # Create RemoveMessage for each message to delete
        to_remove = [RemoveMessage(id=m.id) for m in messages[:-10]]
        return {"messages": to_remove}
    return {"messages": []}
```

**Use cases:**
- Implementing a sliding window (keep last N messages)
- Removing system messages that are no longer needed
- Clearing tool call/response pairs after they're consumed

### Custom Reducer Pattern

You can write your own reducer for any state field:

```python
from typing import Annotated

def add_to_set(existing: set, new: set) -> set:
    """Custom reducer: union of sets."""
    return existing | new

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    seen_topics: Annotated[set, add_to_set]  # ← Custom reducer
```

### C# Analogy

| add_messages Behavior | C# Equivalent |
|----------------------|---------------|
| Append (default) | `List<T>.AddRange(newItems)` |
| Deduplicate by ID | `Dictionary<string, T>[id] = newItem` (upsert) |
| `RemoveMessage` | `List<T>.RemoveAll(m => idsToRemove.Contains(m.Id))` |
| Custom reducer | A custom `ICollection<T>` implementation with merge logic |

---

## 18. ToolNode — Prebuilt Tool Executor

### The Problem

In the agent loop, after the model returns `tool_calls`, *someone* needs to:
1. Look up which tool function matches the name
2. Execute it with the provided arguments
3. Wrap the result in a `ToolMessage` with the correct `tool_call_id`
4. Handle errors gracefully

Doing this manually every time is boilerplate. `ToolNode` solves it.

### What Is ToolNode?

`ToolNode` is a **prebuilt LangGraph node** that does all of the above automatically:

```python
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Found docs about {query}"

# Create the node — pass it ALL your tools
tool_node = ToolNode([get_weather, search_docs])
```

### How It Works Internally

When `tool_node` is invoked as part of a graph:

```
1. Reads state["messages"][-1]  → the AIMessage with tool_calls
2. For EACH tool_call in that message:
   a. Finds the matching tool by name
   b. Calls tool.invoke(tool_call)
   c. Gets back a ToolMessage (content + tool_call_id)
3. Returns {"messages": [ToolMessage1, ToolMessage2, ...]}
   ↑ This becomes the state update (appended via add_messages)
```

### Using ToolNode in a Graph

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Tools
tools = [get_weather, search_docs]

# Model with tools bound
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# Node functions
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(tools))   # ← One line! Handles all tool execution

graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue)
graph.add_edge("tools", "model")  # After tools execute, go back to model

app = graph.compile()
```

### ToolNode vs Manual Tool Execution

| Approach | Code | Use When |
|----------|------|----------|
| `ToolNode(tools)` | 1 line — handles everything | Standard agent patterns |
| Manual loop | ~10 lines — custom logic per tool | Need special handling (logging, auth, partial execution) |

### Error Handling in ToolNode

By default, if a tool throws an exception, `ToolNode` catches it and returns an error `ToolMessage`:

```python
# If get_weather raises ValueError("City not found"):
# ToolNode returns:
# ToolMessage(content="Error: City not found", tool_call_id="call_xyz")
# The model sees this and can try a different approach
```

### Parallel Tool Execution

When the model returns multiple `tool_calls` in one message, `ToolNode` executes them all and returns all `ToolMessage` results:

```python
# Model returns: tool_calls = [
#   {"name": "get_weather", "args": {"city": "Boston"}, "id": "call_1"},
#   {"name": "get_weather", "args": {"city": "NYC"}, "id": "call_2"},
# ]
# ToolNode returns: {"messages": [ToolMessage(..., id="call_1"), ToolMessage(..., id="call_2")]}
```

### C# Analogy

| ToolNode Concept | C# Equivalent |
|-----------------|---------------|
| `ToolNode(tools)` | A `ControllerActivator` / `IServiceProvider.GetService(toolName)` that dispatches by name |
| Matching tool by name | `IServiceCollection` resolution — register by name, resolve by name |
| Error handling | Global exception filter returning error DTOs |
| Parallel execution | `Task.WhenAll(toolCalls.Select(tc => Execute(tc)))` |

---

## 19. Node Types in a Graph

### The Three Node Archetypes

Every LangGraph agent graph has these roles (whether you use prebuilt classes or write them manually):

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Graph                             │
├────────────────┬────────────────┬────────────────────────┤
│  MODEL NODE    │  TOOL NODE     │  CUSTOM NODE           │
│  (Agent Node)  │  (ToolNode)    │  (Your logic)          │
├────────────────┼────────────────┼────────────────────────┤
│  Calls the LLM │  Executes tools│  Any Python code       │
│  Returns       │  Returns       │  Returns               │
│  AIMessage     │  ToolMessages  │  State updates         │
│  (may have     │                │                        │
│  tool_calls)   │                │                        │
└────────────────┴────────────────┴────────────────────────┘
```

### Model Node (Agent Node)

The node that invokes the LLM. Its job:
1. Read messages from state
2. Call `model.invoke(messages)` (with tools bound)
3. Return the `AIMessage` as a state update

```python
def model_node(state: MessagesState):
    """The 'brain' — calls the LLM."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**What comes out:** An `AIMessage` that either:
- Has `content` (final answer) → route to END
- Has `tool_calls` (needs tools) → route to tool node

### Tool Node

The node that executes requested tools. Its job:
1. Read the last `AIMessage` from state
2. Execute each tool call
3. Return `ToolMessage` objects

```python
from langgraph.prebuilt import ToolNode

# Prebuilt — handles everything automatically
tool_node = ToolNode(tools)

# OR manual equivalent:
def manual_tool_node(state: MessagesState):
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        tool = tool_map[tc["name"]]
        result = tool.invoke(tc)
        results.append(result)
    return {"messages": results}
```

### Custom Nodes

Any logic that isn't LLM calls or tool execution:

```python
def check_guardrails(state: MessagesState):
    """Custom node: check if the response violates safety rules."""
    last = state["messages"][-1]
    if "forbidden_word" in last.content:
        return {"messages": [AIMessage(content="I can't help with that.")]}
    return {"messages": []}

def enrich_context(state: MyState):
    """Custom node: add user preferences to state before LLM call."""
    user_prefs = load_from_db(state["user_id"])
    return {"user_preferences": user_prefs}
```

### How They Connect (The Standard Pattern)

```python
graph = StateGraph(MessagesState)

# Register nodes
graph.add_node("agent", model_node)           # Model (agent) node
graph.add_node("tools", ToolNode(tools))      # Tool node
graph.add_node("guardrails", check_guardrails) # Custom node

# Wire edges
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)  # → "tools" or END
graph.add_edge("tools", "agent")                       # tools → back to agent
```

### C# Analogy

| Node Type | C# Equivalent |
|-----------|---------------|
| Model Node | A controller action that calls an AI service |
| Tool Node | A mediator/dispatcher that routes commands to handlers |
| Custom Node | Business logic middleware / validation service |
| The routing between them | ASP.NET pipeline with `UseWhen()` branching |

---

## 20. with_config & Runnable Binding

### The Problem

Sometimes you want to attach default configuration to a runnable so you don't have to pass it every time:

```python
# Annoying: passing tags/metadata on every call
model.invoke("hello", config={"tags": ["production"], "metadata": {"team": "AI"}})
model.invoke("world", config={"tags": ["production"], "metadata": {"team": "AI"}})
```

### The Solution: `with_config`

```python
# Bind config once — it flows automatically
configured_model = model.with_config(
    tags=["production"],
    metadata={"team": "AI"},
    run_name="my_model"
)

# Now just invoke — config is already attached
configured_model.invoke("hello")
configured_model.invoke("world")
```

### What `with_config` Does

It creates a **new runnable** that wraps the original with default config. The original is unchanged:

```python
original_model = ChatOpenAI(model="gpt-4o")

# Creates a new object — original is untouched
tagged_model = original_model.with_config(tags=["v2"], run_name="chat")

original_model.invoke("hi")   # No tags
tagged_model.invoke("hi")     # Has tags=["v2"], run_name="chat"
```

### Config Precedence (Merge Rules)

When you call `.invoke()` with config AND the runnable has `with_config`, they **merge**:

```python
configured = model.with_config(tags=["default"], metadata={"source": "app"})

# Invocation config merges ON TOP of with_config
configured.invoke("hello", config={
    "tags": ["override"],           # ← Extends: tags becomes ["default", "override"]
    "metadata": {"request": "123"}, # ← Merges: {"source": "app", "request": "123"}
    "run_name": "specific_call",    # ← Overrides run_name
})
```

| Field | Merge Behavior |
|-------|---------------|
| `tags` | Combined (union) |
| `metadata` | Merged (invocation wins on conflict) |
| `run_name` | Overridden (invocation wins) |
| `configurable` | Merged |
| `callbacks` | Combined |

### How Config Flows Through a Graph

When you call `app.invoke(input, config={...})`:

1. The config is passed to the **first node**
2. Each node inherits the config automatically
3. Sub-calls (model.invoke inside a node) also receive it
4. `tags` and `metadata` propagate to ALL sub-calls
5. `run_name` does NOT propagate (it's per-call only)

```python
# Config flows through the entire graph
app.invoke(
    {"messages": [...]},
    config={
        "configurable": {"thread_id": "abc"},  # → Checkpointer uses this
        "tags": ["user-request"],               # → All nodes/tools inherit
        "metadata": {"user_id": "123"},        # → All nodes/tools inherit
    }
)
```

### Related: `configurable_fields` (Runtime Model Switching)

For models, you can make specific fields configurable at runtime:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o",
    temperature=0,
    configurable_fields=("model", "temperature"),  # ← These can be overridden
)

# Override at runtime via config
model.invoke("hello", config={"configurable": {"model": "gpt-4o-mini", "temperature": 0.9}})
```

### Related: `with_retry` and Other Binding Methods

`with_config` is part of a family of binding methods on all runnables:

| Method | What It Does |
|--------|-------------|
| `with_config(...)` | Attach default config (tags, metadata, run_name) |
| `with_retry(...)` | Add retry logic with exponential backoff |
| `with_fallbacks(...)` | Try alternative runnables if this one fails |
| `bind(...)` | Bind keyword arguments to the runnable's invoke |
| `bind_tools(...)` | Bind tool schemas (model-specific) |

```python
# Chaining: model with tools, retries, and default config
robust_model = (
    ChatOpenAI(model="gpt-4o")
    .bind_tools(tools)
    .with_retry(stop_after_attempt=3)
    .with_config(tags=["production"])
)
```

### C# Analogy

| LangChain | C# Equivalent |
|-----------|---------------|
| `with_config(tags=[...])` | `.AddOptions(o => o.Tags = [...])` in DI / configuration binding |
| Config flowing through graph | `HttpContext` propagating through middleware pipeline |
| `configurable_fields` | `IOptionsSnapshot<T>` — values can change per-request |
| `with_retry` | Polly retry policy `.AddPolicyHandler(Policy.WaitAndRetry(...))` |
| `with_fallbacks` | Polly fallback policy / circuit breaker |
| `bind(...)` | Partial application / currying in functional C# |

---

## 21. System Instructions in Graphs

### The Pattern

In a LangGraph agent, system instructions define the agent's identity. There are two approaches:

**Approach 1: Prepend SystemMessage in the model node**

```python
from langchain_core.messages import SystemMessage

SYSTEM_PROMPT = """You are a helpful weather assistant.
Only use the get_weather tool when asked about weather.
Always be concise."""

def model_node(state: MessagesState):
    # Prepend system message to every LLM call
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
```

**Approach 2: Use `system_prompt` parameter in `create_agent`**

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant. Be concise.",
)
# The factory handles prepending SystemMessage internally
```

### ChatPromptTemplate (For Complex Prompts)

When your system prompt has dynamic variables:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {persona}. Respond in {language}."),
    MessagesPlaceholder("messages"),  # ← Existing conversation history injected here
])

def model_node(state: MessagesState):
    formatted = prompt.invoke({
        "persona": "a senior engineer",
        "language": "English",
        "messages": state["messages"],
    })
    response = model.invoke(formatted)
    return {"messages": [response]}
```

### Key Rule: System Message Is NOT Stored in State

The system prompt is typically **not** stored in `state["messages"]`. It's prepended fresh on every LLM call. This prevents it from being modified by tools or other nodes.

---

## 22. Graph Streaming

### The Difference: Graph Streaming vs Model Streaming

| Type | What Streams | API |
|------|-------------|-----|
| **Model streaming** | Individual tokens as the LLM generates | `model.stream("hello")` |
| **Graph streaming** | Node outputs as each node completes | `app.stream(input, stream_mode="updates")` |
| **Both combined** | Tokens within node execution | `app.astream_events(input)` |

### stream_mode Options

```python
# Option 1: "values" — full state after each node
for state_snapshot in app.stream(input, config=config, stream_mode="values"):
    print(state_snapshot["messages"][-1])
    # Emits the FULL state dict after each node completes

# Option 2: "updates" — only what changed
for update in app.stream(input, config=config, stream_mode="updates"):
    print(update)
    # {"model_node": {"messages": [AIMessage(...)]}}
    # {"tool_node": {"messages": [ToolMessage(...)]}}

# Option 3: "messages" — message-level streaming (tokens + node boundaries)
for msg, metadata in app.stream(input, config=config, stream_mode="messages"):
    print(msg.content, end="", flush=True)
    # Streams individual tokens as they arrive from the LLM
```

### Frontend Pattern (Node Progress + Token Streaming)

```python
# Show progress per node AND stream tokens
for chunk in app.stream(input, config=config, stream_mode="updates"):
    for node_name, node_output in chunk.items():
        if node_name == "model":
            print(f"🧠 Model thinking...")
        elif node_name == "tools":
            print(f"🔧 Executing tools...")
        print(node_output["messages"][-1].content)
```

### Async Streaming (Production)

```python
async for event in app.astream_events(input, config=config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        await send_to_frontend(token)
    elif event["event"] == "on_chain_end":
        node = event["name"]
        await send_progress(f"Completed: {node}")
```

---

## 23. Time Travel & State History

### What Is Time Travel?

Because checkpointers save state after **every** node execution, you have a complete history. Time travel lets you:
1. **Inspect** — see the state at any point in the conversation
2. **Replay** — re-execute from a previous checkpoint
3. **Fork** — branch off from a past state into a new timeline

### get_state — Inspect Current State

```python
# After invoking the graph:
current = app.get_state(config)
print(current.values["messages"])      # Current messages
print(current.next)                     # Which node would execute next (if interrupted)
print(current.config["configurable"])   # thread_id, checkpoint_id
```

### get_state_history — Browse All Checkpoints

```python
# Walk backwards through every state snapshot
for state in app.get_state_history(config):
    print(f"Step: {state.config['configurable']['checkpoint_id']}")
    print(f"  Next node: {state.next}")
    print(f"  Messages: {len(state.values['messages'])}")
    print("---")
```

Each checkpoint has:
- `values` — the full state at that point
- `next` — which node was about to execute
- `config` — includes `checkpoint_id` for targeting

### Forking — Branch From a Past State

```python
# Get history
history = list(app.get_state_history(config))

# Pick a past checkpoint (e.g., before the tool call went wrong)
past_state = history[2]  # 3rd checkpoint from the end

# Fork: invoke from that checkpoint with a different input
forked_config = past_state.config  # Contains the checkpoint_id
result = app.invoke(
    {"messages": [{"role": "user", "content": "Actually, try a different city"}]},
    config=forked_config,
)
# This creates a NEW timeline branch from that past state
```

### Why This Matters

| Capability | Use Case |
|-----------|----------|
| Inspect state | Debugging — "what did the model see at step 3?" |
| Replay | Testing — re-run with same inputs to verify fixes |
| Fork | User says "go back and try something else" |
| Audit | Compliance — full trace of agent decisions |

---

## 24. Human-in-the-Loop Deep Dive

### interrupt_before — Intercept Before Execution

The graph **stops** right before a node runs. The checkpoint stores the pending state including what the model requested:

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # ← Pause before tool execution
)

# Invoke — runs model_node, then STOPS before tools
result = app.invoke(
    {"messages": [{"role": "user", "content": "Delete all my files"}]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Graph is now frozen. The AIMessage with tool_calls is in state.
```

### Inspecting the Frozen State

```python
state = app.get_state(config)
print(state.next)  # ("tools",) — this is what would execute next
print(state.values["messages"][-1].tool_calls)
# [{'name': 'delete_files', 'args': {'path': '/'}, 'id': 'call_xyz'}]
# ↑ Human reviewer can see what the model wants to do
```

### update_state — Modify Before Resuming

```python
from langchain_core.messages import AIMessage

# Option A: Approve as-is — just resume
app.invoke(None, config=config)

# Option B: Modify the tool call arguments
corrected_msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "delete_files",
        "args": {"path": "/tmp/old_files"},  # ← Changed from "/" to safe path
        "id": "call_xyz",
    }]
)
app.update_state(config, {"messages": [corrected_msg]})
app.invoke(None, config=config)  # Resume with corrected arguments

# Option C: Reject — override with a final response
app.update_state(
    config,
    {"messages": [AIMessage(content="I cannot delete system files.")]},
    as_node="model",  # ← Pretend this came from the model node
)
# Graph sees a final answer (no tool_calls) and routes to END
```

### interrupt_after — Intercept After Execution

The graph **stops** after a node runs but before passing results downstream:

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["tools"],  # ← Pause AFTER tools execute, BEFORE model reads results
)

# Invoke — runs model → tools, then STOPS
result = app.invoke({"messages": [...]}, config=config)

# Check what the tools returned
state = app.get_state(config)
tool_result = state.values["messages"][-1]  # ToolMessage with raw output
print(tool_result.content)
# "User's SSN: 123-45-6789, Balance: $50,000"
# ↑ Admin can redact PII before LLM sees it

# Redact and resume
from langchain_core.messages import ToolMessage
redacted = ToolMessage(
    content="Balance: $50,000 [PII redacted]",
    tool_call_id=tool_result.tool_call_id,
    id=tool_result.id,  # Same ID to replace via add_messages dedup
)
app.update_state(config, {"messages": [redacted]})
app.invoke(None, config=config)  # Model now sees redacted version
```

### The Checkpoint Database (Conceptual)

Internally, checkpoints are stored with a compound key:

```
┌──────────────────────────────────────────────────────────┐
│  Checkpoint Store (e.g., PostgreSQL table)                │
├──────────────┬──────────────────┬────────────────────────┤
│  thread_id   │  checkpoint_id    │  state_json           │
├──────────────┼──────────────────┼────────────────────────┤
│  session-1   │  ckpt-001         │  {messages: [...]}    │
│  session-1   │  ckpt-002         │  {messages: [...]}    │
│  session-1   │  ckpt-003         │  {messages: [...]}    │ ← interrupted here
│  session-2   │  ckpt-001         │  {messages: [...]}    │
└──────────────┴──────────────────┴────────────────────────┘
```

Each row is like a git commit — immutable, ordered, and replayable.

---

## 25. Interview Q&A Anchors

**Q: What is LangGraph and how does it differ from LangChain?**
> **A:** LangChain provides model/tool integrations and agent abstractions (the "what" — models, prompts, tools). LangGraph is the orchestration runtime (the "how" — state management, persistence, human-in-the-loop, streaming). You can use LangGraph without LangChain, but they complement each other. Think of LangChain as the .NET SDK libraries and LangGraph as the ASP.NET runtime.

**Q: What is `ToolRuntime` and why does it exist?**
> **A:** `ToolRuntime` is an injected parameter that gives tools access to execution context — state (conversation messages), context (user identity), store (persistent memory), and stream writer (progress updates). It's hidden from the LLM schema so the model doesn't see it. It exists because tools often need more than just their arguments — they need to know who's calling, what the conversation history is, and where to persist data.

**Q: What are the reserved argument names for LangChain tools?**
> **A:** `config` and `runtime` are reserved. Using them as tool parameter names causes runtime errors because LangChain uses them internally for dependency injection. If you need to accept "config" data, name the parameter something else like `configuration` or `settings`.

**Q: What's the difference between State, Context, and Store in ToolRuntime?**
> **A:** State is short-term memory scoped to the current conversation thread (messages, custom fields). Context is immutable per-invocation configuration (user ID, permissions) — think JWT claims. Store is long-term persistent memory that survives across conversations — think database. A tool might read the user's context to know who they are, check state for conversation history, and write to store to remember preferences for next time.

**Q: What is a `Command` return type and when would you use it?**
> **A:** `Command` is a tool return type that updates graph state beyond just returning text. Use it when a tool needs to set flags, preferences, or counters in the graph's state — not just return information to the model. For example, a "set_language" tool should update the `preferred_language` field in state so subsequent nodes can use it, not just tell the model "language was set."

**Q: How does `init_chat_model` improve over direct class instantiation?**
> **A:** It's a provider-agnostic factory — you switch providers by changing a string (`"openai:gpt-4o"` → `"anthropic:claude-sonnet-4-6"`) without changing imports. It also supports runtime configurability (pick model per-request via `config`), consistent parameter handling across providers, and built-in retry/timeout configuration.

**Q: What is a Checkpointer and why is `thread_id` required?**
> **A:** A Checkpointer persists graph state after each node execution, enabling resume-after-crash, human-in-the-loop interrupts, and multi-turn conversations. `thread_id` is the scope key — it tells the checkpointer which conversation's state to load/save. Without it, the graph wouldn't know which user's conversation to resume. It's like a session ID in web apps.

**Q: How does middleware work in LangChain agents?**
> **A:** Middleware wraps model calls (`@wrap_model_call`) or tool calls (`@wrap_tool_call`) with interceptor logic. `wrap_model_call` can filter tools, switch models, or log requests before they hit the LLM. `wrap_tool_call` can catch errors, add retries, or gate execution. They compose — multiple middleware run in order, each calling `handler(request)` to pass to the next one. It's the same pipeline pattern as ASP.NET middleware.

**Q: What does `return_direct=True` do on a tool?**
> **A:** It short-circuits the agent loop — the tool's output goes directly to the user without another LLM call. Use it when the tool output IS the final answer (e.g., a database lookup returning a formatted status). Don't use it when the output needs summarization or further reasoning.

**Q: How does structured output work in LangChain?**
> **A:** `model.with_structured_output(Schema)` constrains the model to output data matching your schema (Pydantic model, TypedDict, or JSON Schema). Under the hood, it uses the provider's native structured output API or forces a tool call matching the schema. The result is a typed object you can use directly in code — no regex parsing needed.

**Q: What exactly is `thread_id` and who creates it?**
> **A:** `thread_id` is a string (usually a UUID) that **you generate** and pass in `config["configurable"]["thread_id"]`. It scopes the conversation — the checkpointer uses it to load/save the right state. You create a new one for each new conversation and reuse it for follow-ups. It's like a session ID in a web app — the framework doesn't auto-generate it, your app logic does.

**Q: What is `RunnableConfig` and what goes inside it?**
> **A:** It's the universal configuration dictionary passed to `.invoke()`. Contains `configurable` (thread_id, model overrides), `tags` (labels for tracing), `metadata` (custom key-value pairs), `callbacks` (event handlers), `max_concurrency` (parallelism limit), and `recursion_limit` (max graph cycles). Everything inside propagates to sub-calls except `run_name`.

**Q: How do conditional edges work in a StateGraph?**
> **A:** A conditional edge takes a routing function that receives the current state and returns a string — the name of the next node. The graph executes whichever node the function picks. Classic use: check if the model's last message has `tool_calls`; if yes → tool node, if no → END. It's the equivalent of a switch statement in a workflow orchestrator.

**Q: What does `bind_tools` do and is it the same as `create_agent`?**
> **A:** `bind_tools` attaches tool schemas to a model so it *can* request tool execution. But it doesn't execute anything — you still need to manually run tools and feed results back. `create_agent` builds the entire loop for you: bind_tools + model node + tool node + conditional routing + compilation. Use `bind_tools` when building custom graphs; use `create_agent` for standard ReAct agents.

**Q: What is a reducer in LangGraph and why does `messages` need one?**
> **A:** A reducer defines how to merge a node's output into existing state. Without one, returning `{"messages": [new_msg]}` would replace the entire message list. The `add_messages` reducer appends instead. It also handles deduplication by ID and supports `RemoveMessage` for deletion. You annotate the field with `Annotated[list, add_messages]` to attach the reducer.

**Q: What does `.compile()` actually do?**
> **A:** It validates the graph structure (no orphan nodes, all edges resolve), freezes it (no more modifications), attaches the checkpointer and interrupt points, and returns a runnable object with `.invoke()`, `.stream()`, and `.batch()` methods. Think of it like `app.Build()` in ASP.NET — it finalizes the pipeline configuration.

**Q: What is `recursion_limit` and when would you change it?**
> **A:** It's the maximum number of node executions before the graph force-stops (default: 25). It prevents infinite agent loops where the model keeps requesting tools that keep failing. Increase it for complex multi-step agents; decrease it for cost-sensitive applications. When exceeded, a `GraphRecursionError` is raised.

**Q: What is `ToolNode` and why would you use it instead of manually executing tools?**
> **A:** `ToolNode` is a prebuilt LangGraph class that handles the entire tool execution step: it reads `tool_calls` from the last `AIMessage`, finds the matching tool by name, executes it, wraps results in `ToolMessage` objects with correct `tool_call_id`, and handles errors. Use it to avoid 10+ lines of boilerplate in every agent graph. Use manual execution only when you need custom logic per tool (auth, logging, partial execution).

**Q: What are the typical node types in a LangGraph agent?**
> **A:** Three archetypes: (1) **Model node** — calls the LLM, returns `AIMessage` (may contain `tool_calls`); (2) **Tool node** — executes requested tools, returns `ToolMessage` objects; (3) **Custom nodes** — any business logic (guardrails, enrichment, database calls). The standard flow is: model node → conditional edge (tool_calls?) → tool node → back to model node.

**Q: What does `with_config` do and how is it different from passing config to `.invoke()`?**
> **A:** `with_config` binds default configuration (tags, metadata, run_name) to a runnable permanently, creating a new wrapped object. When you call `.invoke(config=...)`, the invocation config merges on top (tags combine, metadata merges, run_name overrides). Use `with_config` to avoid repeating the same config on every call; use invocation config for per-request overrides.

**Q: How does config flow through a LangGraph execution?**
> **A:** When you call `app.invoke(input, config={...})`, the config propagates to all nodes and their sub-calls. `tags` and `metadata` are inherited by every step (model calls, tool calls). `run_name` is per-call only. `configurable.thread_id` is read by the checkpointer. This means a single config dict at the top controls tracing, persistence, and runtime behavior for the entire graph execution.

**Q: How do system instructions work in a LangGraph agent?**
> **A:** You prepend a `SystemMessage` to the messages list inside your model node before calling the LLM. The system message is NOT stored in `state["messages"]` — it's added fresh on each call so nodes and tools can't accidentally mutate it. For `create_agent`, you pass `system_prompt="..."` and the factory handles it.

**Q: What are the graph streaming modes and when would you use each?**
> **A:** Three modes: `"values"` emits full state after each node (good for debugging); `"updates"` emits only the changes per node (good for progress indicators); `"messages"` streams individual tokens as the LLM generates them (good for real-time UX). Use `astream_events` when you need both token-level streaming AND node boundary events.

**Q: What is time travel in LangGraph?**
> **A:** Because checkpointers save state after every node, you can browse the full history with `get_state_history()`, inspect any past checkpoint, and fork a new conversation from any historical point by invoking with that checkpoint's config. It's like git — each checkpoint is an immutable commit, and you can branch from any commit.

**Q: Explain interrupt_before vs interrupt_after.**
> **A:** `interrupt_before=["tools"]` stops the graph BEFORE the tools node runs — you see what the model wants to do and can approve, modify, or reject. `interrupt_after=["tools"]` stops AFTER tools run but BEFORE the model reads results — you can redact sensitive data. Both patterns use `get_state()` to inspect, `update_state()` to modify, and `invoke(None)` to resume.

**Q: How does `update_state` work with `as_node`?**
> **A:** `update_state(config, values, as_node="model")` injects values into state as if they came from the named node. This matters for routing — if you inject an `AIMessage` without `tool_calls` as the model node, the conditional edge sees "no tool calls" and routes to END, effectively rejecting the tool request.

---

## References

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain Models](https://docs.langchain.com/oss/python/langchain/models)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangChain Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [LangGraph ToolNode](https://docs.langchain.com/oss/python/langgraph/workflows-agents#toolnode)
- [LangChain Runnable Interface](https://docs.langchain.com/oss/python/langchain/runnables)
