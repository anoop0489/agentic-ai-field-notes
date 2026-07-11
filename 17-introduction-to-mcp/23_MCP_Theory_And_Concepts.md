# 23. Model Context Protocol (MCP) — Theory & Concepts

> **Context:** This section covers MCP — the open protocol that standardizes how applications provide tools, resources, and context to LLMs. Content is sourced from the [official LangChain MCP documentation](https://docs.langchain.com/oss/python/langchain/mcp) and the [MCP specification](https://modelcontextprotocol.io/introduction).

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Is MCP?](#1-what-is-mcp) | The one-sentence definition and why it exists |
| 2 | [MCP vs Application Protocols (HTTP, gRPC, REST)](#2-mcp-vs-application-protocols-http-grpc-rest) | How MCP differs from traditional client-server communication |
| 3 | [MCP Architecture — Client/Server Model](#3-mcp-architecture--clientserver-model) | Host, client, server roles and how they interact |
| 4 | [The Three Primitives: Tools, Resources, Prompts](#4-the-three-primitives-tools-resources-prompts) | The core capabilities an MCP server exposes |
| 5 | [Transport Mechanisms](#5-transport-mechanisms) | stdio vs HTTP — when to use each |
| 6 | [Stateless vs Stateful Sessions](#6-stateless-vs-stateful-sessions) | Default behaviour and when you need persistence |
| 7 | [MCP in LangChain — langchain-mcp-adapters](#7-mcp-in-langchain--langchain-mcp-adapters) | MultiServerMCPClient, loading tools, using with agents |
| 8 | [Custom MCP Servers with FastMCP](#8-custom-mcp-servers-with-fastmcp) | Building your own tool servers |
| 9 | [Tool Interceptors](#9-tool-interceptors) | Middleware for MCP tool calls — auth, retry, context injection |
| 10 | [Advanced Features](#10-advanced-features) | Structured content, multimodal, progress, logging, elicitation |
| 11 | [References](#references) | Official docs and libraries |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **MCP (Model Context Protocol)** | Universal USB-C for AI tools | An open protocol that standardizes how applications expose tools, resources, and context to LLMs — so any client can connect to any server without custom integration code. |
| **MCP Server** | Exposes tools/resources/prompts | A lightweight process that exposes capabilities (tools, data, prompt templates) via the MCP protocol. Can be local (stdio) or remote (HTTP). |
| **MCP Client** | Connects to MCP servers | The component inside your application that maintains 1:1 connections with MCP servers, handles protocol negotiation, and routes tool calls. |
| **MCP Host** | The application containing client(s) | The user-facing application (IDE, chatbot, agent) that embeds one or more MCP clients. In LangChain, this is your agent script. |
| **Transport** | The wire format (stdio or HTTP) | The communication mechanism between client and server. stdio = local subprocess via stdin/stdout. HTTP = remote via streamable HTTP requests. |
| **FastMCP** | Python framework for building MCP servers | A high-level library that makes creating MCP-compliant tool servers trivial — decorate functions with `@mcp.tool()` and you're done. |
| **MultiServerMCPClient** | LangChain's multi-server connector | The LangChain adapter class that connects to one or more MCP servers and converts their tools into LangChain-compatible tools. |
| **Tool (MCP primitive)** | Executable function for LLMs | A function exposed by an MCP server that an LLM can invoke — e.g., `add(a, b)`, `get_weather(location)`. Converted to LangChain tools automatically. |
| **Resource (MCP primitive)** | Read-only data for context | Data exposed by an MCP server (files, DB records, API responses) that can be read by clients to provide context — not executable. |
| **Prompt (MCP primitive)** | Reusable prompt template | A parameterized prompt template exposed by an MCP server that clients can retrieve and use in their workflows. |
| **Interceptor** | Middleware for tool calls | An async function that wraps MCP tool execution — can modify requests, add auth, implement retries, access agent state, or short-circuit execution. |
| **Elicitation** | Server asks user for input mid-execution | A mechanism where an MCP server can request additional information from the user during tool execution, rather than requiring all inputs upfront. |
| **Stateless session** | Default: fresh session per tool call | `MultiServerMCPClient`'s default mode — each tool invocation creates a new MCP session, executes, and cleans up. No state persists between calls. |
| **Stateful session** | Persistent session via `client.session()` | Explicit session management where the MCP connection persists across multiple tool calls — required for servers that maintain context between interactions. |
| **stdio transport** | Local subprocess communication | Client launches the MCP server as a child process and communicates via stdin/stdout. Inherently stateful (process persists). Best for local tools. |
| **HTTP transport (streamable-http)** | Remote server communication | Client communicates with a remote MCP server over HTTP. Supports auth headers, custom auth flows, and is the standard for deployed services. |
| **Lifecycle** | Init → active → shutdown | The MCP session lifecycle: capability negotiation on connect, active tool/resource usage, then graceful shutdown. Stateful sessions must manage this explicitly. |

---

## 1. What Is MCP?

**Model Context Protocol (MCP)** is an open protocol created by Anthropic that standardizes how applications provide tools and context to LLMs. Think of it as **USB-C for AI** — a universal connector that lets any LLM-powered application talk to any tool server without custom integration.

**The problem MCP solves:**

Before MCP, every tool integration was bespoke:
- OpenAI function calling has its own JSON schema format
- Claude tools have a slightly different format
- LangChain tools use yet another abstraction
- Every new MCP server someone builds requires custom glue code

MCP provides **one standard** so:
- Tool authors write their server once
- Any MCP-compatible client (VS Code, Claude Desktop, your LangChain agent) can use it immediately
- No per-client integration effort

```
WITHOUT MCP:                           WITH MCP:
┌──────────┐   custom code   ┌─────┐   ┌──────────┐              ┌─────┐
│ Client A ├────────────────►│Tool1│   │ Client A ├──┐           │Tool1│
└──────────┘                 └─────┘   └──────────┘  │  MCP      └──┬──┘
┌──────────┐   custom code   ┌─────┐   ┌──────────┐  ├──protocol──►│
│ Client B ├────────────────►│Tool1│   │ Client B ├──┘           ┌──▼──┐
└──────────┘                 └─────┘   └──────────┘              │Tool2│
┌──────────┐   custom code   ┌─────┐                             └─────┘
│ Client C ├────────────────►│Tool2│
└──────────┘                 └─────┘
(N clients × M tools = N×M integrations)  (N + M integrations)
```

**The protocol in one sentence:**

> "MCP servers expose tools, resources, and prompts over a standardized protocol. MCP clients connect and use them. Any client works with any server."

---

## 2. MCP vs Application Protocols (HTTP, gRPC, REST)

MCP is **not** a replacement for HTTP or gRPC — it operates at a higher abstraction layer. Understanding where it sits is critical:

| Dimension | HTTP/REST | gRPC | MCP |
|-----------|-----------|------|-----|
| **Purpose** | General client-server communication | High-performance RPC between services | Standardised tool/context exposure for LLMs |
| **Target consumer** | Any application | Any application | LLM-powered applications specifically |
| **Schema definition** | OpenAPI/Swagger (optional) | Protobuf (mandatory) | MCP protocol spec (mandatory) |
| **Discovery** | Manual (read API docs) | Manual (read .proto files) | Automatic (client queries server for capabilities) |
| **What it exposes** | Endpoints (GET /users, POST /orders) | Methods (service.GetUser) | Three primitives: Tools, Resources, Prompts |
| **Transport** | HTTP itself IS the transport | HTTP/2 | Runs OVER stdio or HTTP (transport-agnostic) |
| **Statefulness** | Stateless (by convention) | Stateless or streaming | Configurable (stateless default, stateful opt-in) |
| **LLM-optimised** | No — returns raw JSON/HTML | No — returns protobuf bytes | Yes — returns clean text, structured content, metadata |
| **Tool invocation** | Client must know endpoint + payload format | Client must have generated stubs | Client asks "what tools exist?" then calls by name |

**Key distinction:**

- **HTTP/REST/gRPC** = "Here are my endpoints, read the docs and build a client"
- **MCP** = "Connect to me, I'll tell you what I can do, and your LLM can call my tools by name"

MCP is specifically designed for the **LLM tool-calling workflow**:
1. Server announces its tools (name, description, input schema)
2. LLM decides which tool to call based on the description
3. Client executes the tool call on behalf of the LLM
4. Result is returned in LLM-friendly format (clean text, not raw HTML/JSON blobs)

**Analogy for a .NET developer:**

| Concept | .NET Equivalent | MCP Equivalent |
|---------|-----------------|----------------|
| Endpoint definition | `[HttpGet("/api/weather")]` | `@mcp.tool() def get_weather()` |
| Schema/contract | Swagger/OpenAPI spec | MCP tool JSON schema (auto-generated from type hints) |
| Discovery | Swagger UI at `/swagger` | `client.get_tools()` — programmatic, not human-readable |
| Client generation | NSwag / AutoRest | `langchain-mcp-adapters` converts to LangChain tools |
| Middleware | ASP.NET middleware pipeline | MCP interceptors |

---

## 3. MCP Architecture — Client/Server Model

MCP follows a strict **Host → Client → Server** hierarchy:

```
┌─────────────────────────────────────────────────────────┐
│ HOST (your application / agent script)                   │
│                                                         │
│  ┌───────────────┐  ┌───────────────┐                  │
│  │ MCP Client 1  │  │ MCP Client 2  │  (1:1 per server)│
│  └───────┬───────┘  └───────┬───────┘                  │
└──────────┼───────────────────┼──────────────────────────┘
		   │ stdio             │ HTTP
		   ▼                   ▼
	┌──────────────┐    ┌──────────────┐
	│ MCP Server A │    │ MCP Server B │
	│ (local math) │    │ (remote API) │
	└──────────────┘    └──────────────┘
```

| Role | Responsibility | Example |
|------|---------------|---------|
| **Host** | The user-facing application. Contains one or more clients. Manages permissions and consent. | Your Python agent script, VS Code, Claude Desktop |
| **Client** | Maintains a 1:1 connection with a single server. Handles protocol negotiation and message routing. | `MultiServerMCPClient` creates one client per configured server |
| **Server** | Exposes tools, resources, and/or prompts. Runs as a subprocess (stdio) or a remote service (HTTP). | A FastMCP script with `@mcp.tool()` decorators |

**Lifecycle of an MCP connection:**

```
1. INITIALIZE  → Client sends capabilities, server responds with its capabilities
2. ACTIVE      → Client calls tools, reads resources, fetches prompts
3. SHUTDOWN    → Graceful disconnect (client or server initiated)
```

---

## 4. The Three Primitives: Tools, Resources, Prompts

MCP servers expose exactly three types of capabilities:

| Primitive | Direction | Who Controls It | Purpose | LangChain Equivalent |
|-----------|-----------|----------------|---------|---------------------|
| **Tools** | Server → Client → LLM invokes | LLM decides when/how to call | Executable functions (search, calculate, query DB) | `BaseTool` / `@tool` decorator |
| **Resources** | Server → Client reads | Application/user decides | Read-only data (files, DB records, API responses) | `Document` / `Blob` |
| **Prompts** | Server → Client fetches | User selects | Reusable prompt templates with parameters | `ChatPromptTemplate` |

### Tools — "Things the LLM can DO"

Tools are the most common primitive. The LLM sees the tool's name, description, and input schema, then decides to call it.

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
	"""Add two numbers"""
	return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
	"""Multiply two numbers"""
	return a * b
```

**How this becomes available to an LLM:**
1. Server starts → exposes `add` and `multiply` with JSON schemas
2. Client calls `get_tools()` → receives tool definitions
3. LangChain converts to `BaseTool` objects → passed to agent
4. LLM sees: "You have tools: add(a: int, b: int), multiply(a: int, b: int)"
5. LLM decides: "I need to call add(3, 5)"
6. Client routes the call to the server → server executes → returns result

### Resources — "Things the LLM can READ"

Resources provide context without execution. Think of them as file access or data feeds.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Load all resources from a server
blobs = await client.get_resources("server_name")

# Or load specific resources by URI
blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

for blob in blobs:
	print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
	print(blob.as_string())  # For text content
```

Resources are returned as LangChain `Blob` objects — unified interface for text and binary.

### Prompts — "Reusable prompt templates"

Prompts allow servers to expose parameterised templates that clients can fetch and use:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Load a prompt by name
messages = await client.get_prompt("server_name", "summarize")

# Load a prompt with arguments
messages = await client.get_prompt(
	"server_name",
	"code_review",
	arguments={"language": "python", "focus": "security"}
)
```

Prompts are returned as LangChain messages — ready to inject into any chat workflow.

---

## 5. Transport Mechanisms

MCP is **transport-agnostic** — it defines the protocol messages, not how they're delivered. Two transports are supported:

### stdio (Standard Input/Output)

```
┌────────────┐  stdin/stdout  ┌────────────┐
│   Client   │◄──────────────►│   Server   │
│            │                │ (subprocess)│
└────────────┘                └────────────┘
```

| Aspect | Detail |
|--------|--------|
| **How it works** | Client launches server as a child process, communicates via stdin/stdout |
| **When to use** | Local tools, development, CLI scripts, single-machine setups |
| **Statefulness** | Inherently stateful — process persists for the lifetime of the connection |
| **Security** | No network exposure. Process-level isolation only. |
| **Example** | A math server running as `python math_server.py` |

```python
client = MultiServerMCPClient(
	{
		"math": {
			"transport": "stdio",
			"command": "python",
			"args": ["/path/to/math_server.py"],
		}
	}
)
```

### HTTP (Streamable HTTP)

```
┌────────────┐   HTTP requests   ┌────────────┐
│   Client   │◄─────────────────►│   Server   │
│            │                   │  (remote)  │
└────────────┘                   └────────────┘
```

| Aspect | Detail |
|--------|--------|
| **How it works** | Client sends HTTP requests to a server endpoint over the network (TCP) |
| **When to use** | Remote tools, deployed services, shared infrastructure, multi-tenant |
| **Statefulness** | Stateless by default. Can be stateful with explicit session management. |
| **Security** | Supports auth headers, OAuth, custom `httpx.Auth` implementations |
| **Production example** | A weather service at `https://mcp-tools.mycompany.com/weather/mcp` |

> ⚠️ **Why do all the examples below use `localhost`?**
>
> Because in a learning/dev environment, you run **both** the client and the server on your own machine to test. The URL `http://localhost:8000/mcp` means "the server is running on MY machine, port 8000." In production, you'd swap that for a real deployed URL like `https://mcp-weather.azurewebsites.net/mcp`.
>
> **The key point:** stdio can ONLY ever be local (it spawns a subprocess). HTTP gives you the OPTION to go remote — same code, just change the URL. The examples use `localhost` purely for "run it yourself and learn" purposes.
>
> ```python
> # Dev (both client and server on your laptop):
> "url": "http://localhost:8000/mcp"
>
> # Production (server deployed to cloud, client is your agent):
> "url": "https://mcp-weather.azurewebsites.net/mcp"
> ```

```python
# Dev example — server running locally for testing
client = MultiServerMCPClient(
	{
		"weather": {
			"transport": "http",
			"url": "http://localhost:8000/mcp",  # ← local dev only!
			"headers": {
				"Authorization": "Bearer YOUR_TOKEN",
				"X-Custom-Header": "custom-value"
			},
		}
	}
)
```

#### HTTP Authentication

Two approaches for authenticating HTTP connections:

**1. Static headers** — simple Bearer token or API key:

```python
"headers": {"Authorization": "Bearer YOUR_TOKEN"}
```

**2. Custom auth flow** — implement `httpx.Auth` interface for OAuth, token rotation, etc.:

```python
client = MultiServerMCPClient(
	{
		"weather": {
			"transport": "http",
			"url": "http://localhost:8000/mcp",
			"auth": custom_auth_instance,
		}
	}
)
```

### Which Transport to Choose?

| Scenario | Transport | Why |
|----------|-----------|-----|
| Local dev / single machine | stdio | No network, simple, fast |
| Team shared tools | HTTP | Centrally deployed, accessible to everyone |
| Production deployment | HTTP | Scalable, auth, monitoring |
| Sensitive/private tools | stdio | No network exposure at all |
| Tools needing persistent state | stdio (inherent) or HTTP + stateful session | Process persists automatically with stdio |

---

## 6. Stateless vs Stateful Sessions

### Default: Stateless

`MultiServerMCPClient` is **stateless by default**. Each tool invocation:
1. Creates a fresh MCP `ClientSession`
2. Executes the tool
3. Cleans up the session

This means no state persists between tool calls. Each call is independent.

**When stateless is fine:** Most tool calls — math operations, web searches, API queries. Each call is self-contained.

### Explicit: Stateful Sessions

When an MCP server maintains context across calls (e.g., a database session, a conversation, a file being edited), you need a **persistent session**:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# Create a session explicitly — persists across all tool calls within the block
async with client.session("server_name") as session:
	tools = await load_mcp_tools(session)
	agent = create_agent("google_genai:gemini-3.5-flash", tools)
	# All tool calls within this block share the same session
```

| Mode | Session Lifetime | Use Case |
|------|-----------------|----------|
| **Stateless** (default) | Created & destroyed per tool call | Independent operations (search, calculate) |
| **Stateful** (`client.session()`) | Persists until `async with` block exits | Multi-step workflows on the same server |

---

## 7. MCP in LangChain — langchain-mcp-adapters

The [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) library bridges MCP servers into the LangChain ecosystem.

### Installation

```bash
uv add langchain-mcp-adapters
```

### Core Pattern: MultiServerMCPClient + Agent

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
	client = MultiServerMCPClient(
		{
			"math": {
				"transport": "stdio",
				"command": "python",
				"args": ["/path/to/math_server.py"],
			},
			"weather": {
				"transport": "http",
				"url": "http://localhost:8000/mcp",
			}
		}
	)

	tools = await client.get_tools()
	agent = create_agent("claude-sonnet-4-6", tools)

	math_response = await agent.ainvoke(
		{"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
	)
	weather_response = await agent.ainvoke(
		{"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
	)
	print(math_response)
	print(weather_response)

if __name__ == "__main__":
	asyncio.run(main())
```

### What Happens Under the Hood

```
1. MultiServerMCPClient connects to each configured server
2. client.get_tools() queries each server for its tool definitions
3. langchain-mcp-adapters converts MCP tools → LangChain BaseTool objects
4. create_agent() binds tools to the LLM (like create_tool_calling_agent)
5. agent.ainvoke() runs the agent — LLM can call any MCP tool
6. When LLM calls a tool → adapter routes it to the correct MCP server
7. Server executes → result flows back through adapter → back to LLM
```

### Error Handling

By default, MCP tool errors are returned to the model as a `ToolMessage` with `status="error"` — the agent can read the error and retry. To raise exceptions instead:

```python
client = MultiServerMCPClient({...}, handle_tool_errors=False)
```

| Error Type | Default Behaviour | With `handle_tool_errors=False` |
|------------|-------------------|--------------------------------|
| Tool execution error (`CallToolResult(isError=True)`) | Returned as error message to model | Raises `ToolException` |
| Transport/session failure | Always raises | Always raises |
| Content conversion failure | Always raises | Always raises |

---

## 8. Custom MCP Servers with FastMCP

[FastMCP](https://gofastmcp.com/getting-started/welcome) is the standard library for building MCP servers in Python.

### Installation

```bash
uv add fastmcp
```

### Math Server (stdio transport)

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
	"""Add two numbers"""
	return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
	"""Multiply two numbers"""
	return a + b

if __name__ == "__main__":
	mcp.run(transport="stdio")
```

### Weather Server (HTTP transport)

```python
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
	"""Get weather for location."""
	return "It's always sunny in New York"

if __name__ == "__main__":
	mcp.run(transport="streamable-http")
```

### What Makes a Good MCP Tool?

| Quality | Why It Matters |
|---------|---------------|
| **Descriptive name** | LLM uses the name to decide if it should call this tool |
| **Clear docstring** | The description is sent to the LLM — vague = wrong tool calls |
| **Typed parameters** | Auto-generates JSON schema — LLM knows what arguments to pass |
| **Focused scope** | One tool = one action. Don't combine "search" and "delete" in one tool. |

---

## 9. Tool Interceptors

Interceptors are **middleware for MCP tool calls**. They wrap tool execution with cross-cutting concerns: logging, auth, retries, context injection, state updates.

### Why Interceptors Exist

MCP servers run as **separate processes** — they can't access:
- LangGraph agent state
- LangGraph store (long-term memory)
- Runtime context (user ID, API keys)
- RunnableConfig

Interceptors bridge this gap by running **inside your agent process** but wrapping **outbound MCP tool calls**.

### Basic Pattern

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def logging_interceptor(
	request: MCPToolCallRequest,
	handler,
):
	"""Log tool calls before and after execution."""
	print(f"Calling tool: {request.name} with args: {request.args}")
	result = await handler(request)
	print(f"Tool {request.name} returned: {result}")
	return result

client = MultiServerMCPClient(
	{"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
	tool_interceptors=[logging_interceptor],
)
```

### Modifying Requests

Use `request.override()` to create modified requests (immutable pattern):

```python
async def double_args_interceptor(request: MCPToolCallRequest, handler):
	"""Double all numeric arguments before execution."""
	modified_args = {k: v * 2 for k, v in request.args.items()}
	modified_request = request.override(args=modified_args)
	return await handler(modified_request)
```

### Composing Interceptors (Onion Model)

Multiple interceptors execute in "onion" order — first in the list is the outermost layer:

```python
client = MultiServerMCPClient(
	{...},
	tool_interceptors=[outer_interceptor, inner_interceptor],
)

# Execution order:
# outer: before → inner: before → tool execution → inner: after → outer: after
```

### Retry with Exponential Backoff

```python
import asyncio

async def retry_interceptor(request: MCPToolCallRequest, handler, max_retries=3, delay=1.0):
	"""Retry failed tool calls with exponential backoff."""
	last_error = None
	for attempt in range(max_retries):
		try:
			return await handler(request)
		except Exception as e:
			last_error = e
			if attempt < max_retries - 1:
				wait_time = delay * (2 ** attempt)
				await asyncio.sleep(wait_time)
	raise last_error
```

### Accessing Runtime Context

When used within a LangChain agent, interceptors receive `ToolRuntime` context:

```python
from dataclasses import dataclass
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

@dataclass
class Context:
	user_id: str
	api_key: str

async def inject_user_context(request: MCPToolCallRequest, handler):
	"""Inject user credentials into MCP tool calls."""
	runtime = request.runtime
	user_id = runtime.context.user_id
	api_key = runtime.context.api_key

	modified_request = request.override(
		args={**request.args, "user_id": user_id}
	)
	return await handler(modified_request)
```

### State Updates with Command

Interceptors can return `Command` objects to update agent state or redirect graph flow:

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

async def handle_task_completion(request: MCPToolCallRequest, handler):
	"""Mark task complete and hand off to summary agent."""
	result = await handler(request)

	if request.name == "submit_order":
		return Command(
			update={
				"messages": [result] if isinstance(result, ToolMessage) else [],
				"task_status": "completed",
			},
			goto="summary_agent",
		)
	return result
```

---

## 10. Advanced Features

### Structured Content

MCP tools can return **structured content** (machine-parseable JSON) alongside human-readable text. Access via the `artifact` field on `ToolMessage`:

```python
from langchain.messages import ToolMessage

result = await agent.ainvoke(
	{"messages": [{"role": "user", "content": "Get data from the server"}]}
)

for message in result["messages"]:
	if isinstance(message, ToolMessage) and message.artifact:
		structured_content = message.artifact["structured_content"]
```

### Multimodal Tool Content

MCP tools can return images, text, and other media. Access via `content_blocks`:

```python
for message in result["messages"]:
	if message.type == "tool":
		for block in message.content_blocks:
			if block["type"] == "text":
				print(f"Text: {block['text']}")
			elif block["type"] == "image":
				print(f"Image URL: {block.get('url')}")
```

### Progress Notifications

Subscribe to progress updates for long-running tools:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

async def on_progress(progress: float, total: float | None, message: str | None, context: CallbackContext):
	percent = (progress / total * 100) if total else progress
	tool_info = f" ({context.tool_name})" if context.tool_name else ""
	print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

client = MultiServerMCPClient(
	{...},
	callbacks=Callbacks(on_progress=on_progress),
)
```

### Logging

Subscribe to server log messages:

```python
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.types import LoggingMessageNotificationParams

async def on_logging_message(params: LoggingMessageNotificationParams, context: CallbackContext):
	print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
	{...},
	callbacks=Callbacks(on_logging_message=on_logging_message),
)
```

### Elicitation

MCP servers can **request additional input from users during execution** — interactive tool flows without requiring all inputs upfront.

#### Server Side (requesting input)

```python
from pydantic import BaseModel
from mcp.server.fastmcp import Context, FastMCP

server = FastMCP("Profile")

class UserDetails(BaseModel):
	email: str
	age: int

@server.tool()
async def create_profile(name: str, ctx: Context) -> str:
	"""Create a user profile, requesting details via elicitation."""
	result = await ctx.elicit(
		message=f"Please provide details for {name}'s profile:",
		schema=UserDetails,
	)
	if result.action == "accept" and result.data:
		return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"
	if result.action == "decline":
		return f"User declined. Created minimal profile for {name}."
	return "Profile creation cancelled."
```

#### Client Side (handling the request)

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

async def on_elicitation(
	mcp_context: RequestContext,
	params: ElicitRequestParams,
	context: CallbackContext,
) -> ElicitResult:
	return ElicitResult(
		action="accept",
		content={"email": "user@example.com", "age": 25},
	)

client = MultiServerMCPClient(
	{"profile": {"url": "http://localhost:8000/mcp", "transport": "http"}},
	callbacks=Callbacks(on_elicitation=on_elicitation),
)
```

#### Elicitation Response Actions

| Action | Description |
|--------|-------------|
| `accept` | User provided valid input. Include data in the `content` field. |
| `decline` | User chose not to provide the requested information. |
| `cancel` | User cancelled the operation entirely. |

---

## References

- [MCP Official Documentation](https://modelcontextprotocol.io/introduction)
- [MCP Specification (2025-03-26)](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [LangChain MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)
- [`langchain-mcp-adapters` GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [FastMCP Documentation](https://gofastmcp.com/getting-started/welcome)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
