# 26. LangChain MCP Adapters — Official Documentation Reference

> **Context:** This document is a structured reference based on the official [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) documentation. It covers how to convert MCP tools into LangChain-compatible tools, connect to MCP servers from Python, and build LangGraph agents that use MCP servers.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Is langchain-mcp-adapters?](#1-what-is-langchain-mcp-adapters) | Purpose, installation, core value |
| 2 | [Quickstart: Single Server (stdio)](#2-quickstart-single-server-stdio) | Minimal server + client with stdio transport |
| 3 | [Multiple MCP Servers](#3-multiple-mcp-servers) | Connecting to multiple servers simultaneously |
| 4 | [Streamable HTTP Transport](#4-streamable-http-transport) | Modern HTTP-based MCP transport |
| 5 | [Runtime Headers (Auth)](#5-runtime-headers-auth) | Passing authentication headers to MCP servers |
| 6 | [Tool Error Handling](#6-tool-error-handling) | How execution errors flow back to the agent |
| 7 | [Using with LangGraph StateGraph](#7-using-with-langgraph-stategraph) | Manual graph construction with MCP tools |
| 8 | [Using with LangGraph API Server](#8-using-with-langgraph-api-server) | Deploying MCP-powered agents as a service |
| 9 | [API Reference Summary](#9-api-reference-summary) | Classes, functions, and types |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **langchain-mcp-adapters** | Bridge between MCP and LangChain | A Python package that converts MCP tools into LangChain-compatible tools, enabling LangGraph agents to use any MCP server without manual adaptation. |
| **MultiServerMCPClient** | Connect to many MCP servers at once | A client class that manages connections to multiple MCP servers (stdio, SSE, HTTP) and exposes all their tools as a unified list for LangGraph agents. |
| **load_mcp_tools()** | Convert MCP tools → LangChain tools | A function that takes an active MCP `ClientSession` and returns a list of LangChain-compatible tool objects that wrap the MCP server's tools. |
| **create_agent** | Create an agent from model + tools | A convenience function from `langchain.agents` that creates an agent given an LLM (or model string like `"openai:gpt-4.1"`) and a list of tools. Handles the tool-calling loop automatically. |
| **FastMCP** | Quick MCP server builder | A class from the `mcp` SDK that provides a decorator-based API for defining MCP tools, resources, and prompts with minimal boilerplate. |
| **StdioServerParameters** | Config for launching stdio servers | A dataclass specifying how to launch an MCP server as a subprocess (command, args, env vars). Used by clients connecting via stdio transport. |
| **handle_tool_errors** | Graceful error handling flag | When `True` (default), MCP execution errors are returned as error `ToolMessage`s so the agent can self-correct. When `False`, errors raise exceptions. |

---

## 1. What Is langchain-mcp-adapters?

A lightweight Python package that bridges two ecosystems:

```
┌─────────────────────────────────────────────────────────┐
│ THE BRIDGE                                              │
│                                                         │
│  MCP Ecosystem          langchain-mcp-adapters          │
│  ┌──────────────┐       ┌──────────────────┐           │
│  │ MCP Servers  │──────►│ Converts MCP     │           │
│  │ (any server) │       │ tools → LangChain│           │
│  └──────────────┘       │ compatible tools │           │
│                         └────────┬─────────┘           │
│                                  │                      │
│                                  ▼                      │
│                         ┌──────────────────┐           │
│                         │ LangGraph Agent  │           │
│                         │ (uses tools      │           │
│                         │  natively)       │           │
│                         └──────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Core Features

- 🛠️ Convert MCP tools into LangChain tools usable with LangGraph agents
- 📦 `MultiServerMCPClient` — connect to multiple MCP servers and load all tools
- 🔌 Supports all MCP transports: stdio, SSE (legacy), streamable HTTP, WebSocket

### Installation

```bash
pip install langchain-mcp-adapters
# or with uv:
uv add langchain-mcp-adapters langgraph langchain-openai
```

> **Note:** Installing `langchain-mcp-adapters` automatically installs the `mcp` SDK — you don't need to install it separately.

---

## 2. Quickstart: Single Server (stdio)

The simplest possible setup: one MCP server + one LangGraph agent.

### Server Side

```python
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
	"""Add two numbers"""
	return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
	"""Multiply two numbers"""
	return a * b

if __name__ == "__main__":
	mcp.run(transport="stdio")
```

### Client Side

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

server_params = StdioServerParameters(
	command="python",
	# MUST be full absolute path to the server script
	args=["/full/path/to/math_server.py"],
)

async with stdio_client(server_params) as (read, write):
	async with ClientSession(read, write) as session:
		await session.initialize()

		# Convert MCP tools → LangChain tools
		tools = await load_mcp_tools(session)

		# Create agent and invoke
		agent = create_agent("openai:gpt-4.1", tools)
		result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
```

### What Happens Under the Hood

```
1. stdio_client() launches math_server.py as a subprocess
2. ClientSession establishes the MCP protocol handshake
3. load_mcp_tools() calls session.list_tools() via MCP protocol
4. Each MCP tool is wrapped in a LangChain Tool object
   (name, description, args schema → all preserved)
5. create_agent() binds these tools to the LLM
6. Agent invokes tools via MCP protocol transparently
```

---

## 3. Multiple MCP Servers

`MultiServerMCPClient` connects to multiple servers and returns a unified tool list.

### Server Side (two servers)

```python
# math_server.py — transport: stdio (same as above)

# weather_server.py — transport: http
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
	"""Get weather for location."""
	return "It's always sunny in New York"

if __name__ == "__main__":
	mcp.run(transport="http")
```

```bash
# Start the HTTP server first (it needs to be running when the client connects)
python weather_server.py
```

### Client Side

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
	{
		"math": {
			"command": "python",
			"args": ["/full/path/to/math_server.py"],
			"transport": "stdio",
		},
		"weather": {
			# Server must already be running on this port
			"url": "http://localhost:8000/mcp",
			"transport": "http",
		}
	}
)
tools = await client.get_tools()
agent = create_agent("openai:gpt-4.1", tools)
math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
weather_response = await agent.ainvoke({"messages": "what is the weather in NYC?"})
```

### Key Point: Session Lifecycle

> By default, `MultiServerMCPClient` starts a **new session** for each tool invocation. If you need to reuse a session (e.g., for stateful servers), explicitly open one:
>
> ```python
> async with client.session("math") as session:
>     tools = await load_mcp_tools(session)
> ```

---

## 4. Streamable HTTP Transport

The modern MCP transport (replaced SSE as of spec 2025-03-26).

### Using Raw Client

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools

async with streamablehttp_client("http://localhost:3000/mcp") as (read, write, _):
	async with ClientSession(read, write) as session:
		await session.initialize()
		tools = await load_mcp_tools(session)
		agent = create_agent("openai:gpt-4.1", tools)
		result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
```

### Using MultiServerMCPClient

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
	{
		"math": {
			"transport": "http",
			"url": "http://localhost:3000/mcp"
		},
	}
)
tools = await client.get_tools()
agent = create_agent("openai:gpt-4.1", tools)
result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
```

---

## 5. Runtime Headers (Auth)

Pass authentication or custom headers when connecting to HTTP-based MCP servers.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
	{
		"weather": {
			"transport": "http",
			"url": "http://localhost:8000/mcp",
			"headers": {
				"Authorization": "Bearer YOUR_TOKEN",
				"X-Custom-Header": "custom-value"
			},
		}
	}
)
tools = await client.get_tools()
agent = create_agent("openai:gpt-4.1", tools)
response = await agent.ainvoke({"messages": "what is the weather in NYC?"})
```

> **Important:** Only `sse` and `http` (streamable HTTP) transports support runtime headers. Headers are sent with every HTTP request to the MCP server. stdio transport does not support headers (it's a subprocess, not HTTP).

---

## 6. Tool Error Handling

MCP distinguishes two types of failures:

| Error Type | What It Means | Default Behavior |
|------------|---------------|-----------------|
| **Execution error** (`isError=True`) | Tool ran but failed (e.g., "project not found") | Returned to agent as `ToolMessage(status="error")` — agent can self-correct |
| **Transport/protocol failure** | Connection dropped, timeout, etc. | Always raises an exception — cannot be suppressed |

### Default: Graceful Error Handling

```python
client = MultiServerMCPClient({...})
tools = await client.get_tools()  # handle_tool_errors=True by default
# Agent sees the error message and can retry or adjust
```

### Legacy: Raise Exceptions on Tool Errors

```python
client = MultiServerMCPClient({...}, handle_tool_errors=False)
# or at the tool-loading level:
tools = await load_mcp_tools(session, handle_tool_errors=False)
# Tool execution errors now raise ToolException
```

> **Why graceful is better:** The agent sees what went wrong (e.g., "No data found for location X") and can adjust its approach — "let me try a different location." Raising an exception crashes the agent loop entirely.

---

## 7. Using with LangGraph StateGraph

For custom graph topologies (not just pre-built ReAct):

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1")

client = MultiServerMCPClient(
	{
		"math": {
			"command": "python",
			"args": ["/full/path/to/math_server.py"],
			"transport": "stdio",
		},
		"weather": {
			"url": "http://localhost:8000/mcp",
			"transport": "http",
		}
	}
)
tools = await client.get_tools()

def call_model(state: MessagesState):
	response = model.bind_tools(tools).invoke(state["messages"])
	return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_node(ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")
graph = builder.compile()

math_response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
weather_response = await graph.ainvoke({"messages": "what is the weather in NYC?"})
```

### When to Use StateGraph vs create_agent

| Use Case | Which to Use |
|----------|-------------|
| Simple tool-calling agent | `create_agent` — handles the loop for you |
| Custom routing logic (e.g., human-in-the-loop, parallel tool calls) | `StateGraph` — full control over graph topology |
| Need to add memory, interrupts, or conditional branching | `StateGraph` — compose with other LangGraph features |

---

## 8. Using with LangGraph API Server

Deploy your MCP-powered agent as a hosted service:

```python
# graph.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def make_graph():
	client = MultiServerMCPClient(
		{
			"weather": {
				"url": "http://localhost:8000/mcp",
				"transport": "http",
			},
			"math": {
				"command": "python",
				"args": ["/full/path/to/math_server.py"],
				"transport": "stdio",
			},
		}
	)
	tools = await client.get_tools()
	agent = create_agent("openai:gpt-4.1", tools)
	return agent
```

```json
// langgraph.json
{
  "dependencies": ["."],
  "graphs": {
	"agent": "./graph.py:make_graph"
  }
}
```

> ⚠️ **Production warning from official docs:** MCP's stdio transport was designed for local desktop applications. Before using stdio in a web server context, evaluate whether there's a more appropriate solution. For remote/cloud deployments, prefer HTTP transport.

---

## 9. API Reference Summary

### Classes

| Class | Purpose |
|-------|---------|
| `MultiServerMCPClient` | Manages connections to multiple MCP servers |
| `StdioConnection` | Connection config for stdio-based servers |
| `SSEConnection` | Connection config for SSE servers (legacy) |
| `StreamableHttpConnection` | Connection config for modern HTTP servers |
| `WebsocketConnection` | Connection config for WebSocket servers |
| `MCPToolArtifact` | Wraps tool response artifacts |
| `McpHttpClientFactory` | Creates HTTP client instances for MCP |
| `ToolCallInterceptor` | Intercepts tool calls for logging/modification |
| `MCPToolCallRequest` | Represents a pending tool call |

### Functions

| Function | Purpose |
|----------|---------|
| `load_mcp_tools(session)` | Loads all tools from an MCP session as LangChain tools |
| `convert_mcp_tool_to_langchain_tool()` | Converts a single MCP tool to LangChain format |
| `to_fastmcp()` | Converts LangChain tools back to MCP tools (reverse direction) |
| `load_mcp_prompt(session, name)` | Loads a prompt template from an MCP server |
| `load_mcp_resources(session)` | Loads resources from an MCP server |
| `get_mcp_resource(session, uri)` | Gets a specific resource by URI |
| `create_session()` | Creates a new MCP client session |

### Types

| Type | Purpose |
|------|---------|
| `Connection` | Union type for all connection configs |
| `ToolMessageContentBlock` | Content block in a tool response |
| `ConvertedToolResult` | Result after converting MCP response |
| `MCPToolCallResult` | Raw result from intercepted tool call |

### Callbacks

| Callback | Purpose |
|----------|---------|
| `LoggingMessageCallback` | Logs all MCP messages for debugging |
| `ProgressCallback` | Reports progress during long operations |
| `ElicitationCallback` | Handles server-initiated prompts to the user |

---

## References

- [langchain-mcp-adapters GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [API Reference](https://reference.langchain.com/python/langchain-mcp-adapters)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Streamable HTTP Spec](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http)
