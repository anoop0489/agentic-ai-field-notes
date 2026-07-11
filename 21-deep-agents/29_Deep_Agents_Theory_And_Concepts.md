# 29. Deep Agents — Theory & Concepts

> **Context:** This section covers **Deep Agents** — LangChain's "agent harness" for building agents that can plan, spawn subagents, use a virtual filesystem, and manage growing context across long-running, multi-step tasks. Content is sourced from the [official LangChain Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/) and the [`deepagents` API reference](https://reference.langchain.com/python/deepagents/).

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Are Deep Agents?](#1-what-are-deep-agents) | The one-sentence definition and the "agent harness" idea |
| 2 | [Frameworks, Runtimes & Harnesses](#2-frameworks-runtimes--harnesses) | How Deep Agents relates to LangChain and LangGraph |
| 3 | [Quickstart — `create_deep_agent`](#3-quickstart--create_deep_agent) | The minimal agent and the `provider:model` string format |
| 4 | [The Four Core Capabilities](#4-the-four-core-capabilities) | The mental model that organizes everything else |
| 5 | [Execution Environment](#5-execution-environment) | Tools, virtual filesystem, permissions, code execution, streaming |
| 6 | [Context Management](#6-context-management) | Skills, memory, summarization/offloading, prompt caching |
| 7 | [Delegation](#7-delegation) | Task planning (`write_todos`) and subagents (`task`) |
| 8 | [Steering — Human-in-the-Loop](#8-steering--human-in-the-loop) | Pausing for approval with `interrupt_on` |
| 9 | [Middleware & the Default Stack](#9-middleware--the-default-stack) | How the harness is assembled and customized |
| 10 | [When to Use Deep Agents](#10-when-to-use-deep-agents) | Decision guide vs `create_agent` and raw LangGraph |
| 11 | [C# / .NET Analogies](#11-c--net-analogies) | Mapping harness concepts to ASP.NET Core patterns |
| 12 | [Interview Q&A Anchors](#interview-qa-anchors) | Production-grade answers to expected questions |
| 13 | [References](#references) | Official docs and libraries |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Deep Agent** | Agent + built-in planning, files, subagents, memory | An agent built with the `deepagents` library that ships with task planning, a virtual filesystem, subagent spawning, skills, and long-term memory out of the box — designed for complex, multi-step tasks. |
| **Agent harness** | The scaffolding around the tool-calling loop | The same core tool-calling loop as any agent, wrapped with reliability features (context management, delegation, steering, a filesystem) that make agents dependable on real tasks. Deep Agents *is* a harness. |
| **`create_deep_agent`** | The one factory function you call | The entry point that assembles a deep agent from a model, tools, and a system prompt — wiring in the default middleware stack automatically. |
| **`deepagents`** | The standalone library | A PyPI package built on LangChain's agent building blocks and the LangGraph runtime (for durable execution, streaming, human-in-the-loop). |
| **Virtual filesystem** | File tools over a pluggable backend | A configurable file layer (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, etc.) backed by in-memory state, local disk, a LangGraph store, or a custom backend. |
| **Backend** | Where files actually live | The pluggable storage implementation behind the virtual filesystem — `StateBackend` (in-memory), `FilesystemBackend` (disk), `StoreBackend` (LangGraph store), composite, or custom. |
| **Filesystem permissions** | Declarative allow/deny for paths | Rules (`operations`, `paths`, `mode`) evaluated in order, first-match-wins, controlling which paths the agent can read or write. |
| **Sandbox backend** | Isolated shell via `execute` | A backend that exposes an `execute` tool for running real shell commands (installs, tests, CLIs) in an isolated OS environment. |
| **Interpreter** | Scoped JS via `eval` | A QuickJS runtime that adds an `eval` tool for lightweight loops, batching, and programmatic tool calling — no shell, network, or filesystem access. |
| **Skill** | On-demand domain knowledge | A directory with a `SKILL.md` file (plus optional scripts/templates) loaded via progressive disclosure — frontmatter at startup, full content only when needed. |
| **Memory** | Always-loaded persistent context | `AGENTS.md` files passed via `memory=`, loaded on every run to carry preferences, conventions, and project rules across conversations. |
| **Context offloading** | Move big results out of the prompt | Storing large tool outputs/history in the filesystem instead of the live context window, keeping the prompt small. |
| **Summarization** | Compress conversation history | Automatic compression of older messages so long runs stay within token limits. |
| **`write_todos`** | Built-in planning tool | A tool that maintains a structured task list with `pending` / `in_progress` / `completed` statuses, persisted in agent state. |
| **Subagent** | Ephemeral child agent | A fresh, isolated agent spawned via the `task` tool that runs autonomously and returns a single compact report — keeping heavy work out of the main context. |
| **`task` tool** | Spawns subagents | The built-in tool the main agent calls to delegate an isolated subtask to a subagent. |
| **`general-purpose` subagent** | The default worker | The subagent enabled by default; you can also define custom subagents with narrower tools/permissions. |
| **Steering** | Human control at runtime | Human-in-the-loop approval and interrupts that pause the agent at sensitive tool calls. |
| **`interrupt_on`** | Pause-before-tool config | A mapping of tool names to interrupt configs (e.g., `{"edit_file": True}`) that pauses for approval before those calls. |
| **Harness profile** | Per-model harness tuning | A registered profile (`excluded_tools`, `excluded_middleware`, etc.) that customizes which tools/middleware a given model sees. |
| **Middleware** | Composable behavior layers | Units like `FilesystemMiddleware` and `SubAgentMiddleware` that assemble the harness; some are required scaffolding and cannot be removed. |

---

## 1. What Are Deep Agents?

**Deep Agents** is a library (`deepagents`) for building agents that can handle **complex, multi-step tasks** — the kind that overflow a single context window, require planning, or benefit from spinning up helpers. Instead of wiring these capabilities yourself, they come built in.

Out of the box, a deep agent can:

- **Take actions in an environment** — call tools, read/write files, execute code
- **Connect to your data** — load memories, skills, and domain knowledge at the right moment
- **Manage growing context** — summarize history and offload large results across long runs
- **Parallelize tasks** — delegate to general or specialized subagents in isolated context windows
- **Stay in the loop** — pause for human approval at critical decision points
- **Improve over time** — update memory, skills, and prompts based on real usage

**The one-sentence definition:**

> "Deep Agents is an *agent harness* — the standard tool-calling loop plus built-in planning, a virtual filesystem, subagents, memory, and human-in-the-loop — so agents stay reliable on long, complex tasks."

### The "agent harness" idea

An **agent harness** is the scaffolding *around* the core loop where an LLM reads messages, picks a tool, runs it, observes the result, and repeats. Every agent framework has that loop. What separates a toy from a production agent is everything around it: how it plans, how it keeps context small, how it delegates, and how a human stays in control.

```
		┌──────────────────────────────────────────────────────┐
		│                    DEEP AGENT HARNESS                 │
		│                                                       │
		│   ┌───────────────  CORE LOOP  ───────────────┐       │
		│   │  read messages → pick tool → run → observe │      │
		│   └────────────────────────────────────────────┘      │
		│                                                       │
		│   Execution env   Context mgmt   Delegation   Steering│
		│   (tools, files,  (skills,       (subagents,  (human- │
		│    code, stream)   memory,        planning)    in-loop)│
		│                    summarize)                         │
		└──────────────────────────────────────────────────────┘
```

Deep Agents is that harness, pre-assembled and configurable.

---

## 2. Frameworks, Runtimes & Harnesses

Newcomers frequently conflate **LangChain**, **LangGraph**, and **Deep Agents**. They sit at different layers and are used *together*, not as alternatives.

| Layer | What it is | You use it for |
|-------|-----------|----------------|
| **LangChain** | The **framework** — core building blocks | Models, tools, messages, prompts. `create_agent` builds a *custom* agent from these primitives. |
| **LangGraph** | The **runtime** — durable execution engine | Streaming, checkpointing, human-in-the-loop, cycles. It's what actually runs the graph underneath. |
| **Deep Agents** | The **harness** — batteries-included agent | Complex tasks where you want planning, files, subagents, and memory *without* assembling them by hand. |

```
┌─────────────────────────────────────────────┐
│  Deep Agents  (harness: built-in capabilities)│  ← create_deep_agent(...)
├─────────────────────────────────────────────┤
│  LangChain    (framework: models/tools/msgs) │  ← create_agent(...) for custom
├─────────────────────────────────────────────┤
│  LangGraph    (runtime: durable execution)   │  ← raw StateGraph for full control
└─────────────────────────────────────────────┘
```

**How to choose (official guidance):**

- **Deep Agents** → you want the built-in capabilities (planning, filesystem, subagents, memory) and a fast path to a reliable agent.
- **LangChain `create_agent`** → you want a custom agent *without* those built-ins, assembling exactly the pieces you need.
- **Raw LangGraph workflow** → you want full control over the control flow (like our Section 16 Agentic RAG graph, where we hand-built router → retrieve → grade → generate → re-check).

> 💡 **Connecting to earlier sections:** In Section 13 we learned LangGraph is the graph runtime. In Sections 14–16 we hand-built graphs (Reflection, Reflexion, Agentic RAG) node-by-node. Deep Agents is the opposite trade-off — instead of designing topology yourself, you accept a proven harness and customize it via middleware.

For a side-by-side comparison with Anthropic's harness, the docs provide [Deep Agents vs. Claude Agent SDK](https://docs.langchain.com/oss/python/deepagents/comparison).

---

## 3. Quickstart — `create_deep_agent`

A minimal deep agent needs three things: a **model**, a list of **tools**, and a **system prompt**.

```python
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
	"""Get weather for a given city."""
	return f"It's always sunny in {city}!"


agent = create_deep_agent(
	model="anthropic:claude-sonnet-4-5",
	tools=[get_weather],
	system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
	{"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

Install with:

```bash
uv add deepagents
```

### The `provider:model` string format

The `model=` argument accepts a **`"provider:model-name"`** string, and Deep Agents resolves the right chat model class for you. The same code works across providers by swapping only that string:

| Provider | Example string |
|----------|----------------|
| Anthropic | `"anthropic:claude-sonnet-4-5"` |
| OpenAI | `"openai:gpt-4o"` |
| Google | `"google_genai:gemini-2.5-flash"` |
| Ollama (local) | `"ollama:qwen3:1.7b"` |
| OpenRouter / Fireworks / Baseten | `"openrouter:<model>"`, `"fireworks:<model>"`, `"baseten:<model>"` |

> ⚠️ **Docs caveat:** The official quickstart uses forward-looking placeholder model names (e.g., `gpt-5.5`, `gemini-3.5-flash`, `claude-sonnet-4-6`) that may not exist yet. The **format** is what matters — substitute whatever model your provider currently offers. Examples above use models available at the time of writing; always check your provider's current lineup.

Instead of a string, you can also pass a fully instantiated LangChain chat model object (e.g., a configured `ChatOpenAI` with a custom SSL/`httpx` client — exactly the pattern we used behind the corporate proxy in earlier sections).

> 💡 **Observability:** Because Deep Agents runs on LangGraph, you get [LangSmith](https://smith.langchain.com/) tracing for free — set your `LANGSMITH_*` env vars and every tool call, subagent, and interrupt is traced. See the [observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart).

---

## 4. The Four Core Capabilities

Everything in Deep Agents organizes under **four pillars**. Keep this map in your head — every feature below hangs off one of these.

| Pillar | Icon idea | What it governs | Key pieces |
|--------|-----------|-----------------|-----------|
| **Execution environment** | ⚡ where the agent *acts* | How the agent takes actions | Tools & MCP, virtual filesystem, permissions, code execution, streaming |
| **Context management** | 🗄️ what the agent *knows* | What's in context and how it survives long runs | Skills, memory, summarization/offloading, prompt caching |
| **Delegation** | 🕸️ how work is *split* | Breaking big problems into parallel units | Task planning (`write_todos`), subagents (`task`) |
| **Steering** | 🧑 how humans stay *in control* | Runtime oversight | Human-in-the-loop (`interrupt_on`) |

```
EXECUTION ENV        CONTEXT MGMT         DELEGATION          STEERING
──────────────       ──────────────       ──────────         ──────────
Tools & MCP          Skills               write_todos        interrupt_on
Virtual FS           Memory (AGENTS.md)   Subagents (task)   HITL approval
Permissions          Summarization        general-purpose
Code execution       Context offloading   custom subagents
Streaming            Prompt caching
```

The next four sections walk each pillar in depth.

---

## 5. Execution Environment

The execution environment is **where an agent acts**. It has four layers plus streaming to observe them.

### 5.1 Tools & MCP

Pass custom Python functions, LangChain tools, or tools from any **MCP server** via `tools=`. Deep Agents fully supports the [Model Context Protocol](https://docs.langchain.com/oss/python/langchain/mcp) (which we covered in Sections 17–19), so you can connect databases, APIs, and file systems through a standard interface.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
	model="anthropic:claude-sonnet-4-5",
	tools=[search, fetch_page, run_query],   # plain functions or LangChain tools
)
```

> 💡 **Ties into Section 17–19:** The MCP tools you learned to expose with FastMCP and load with `MultiServerMCPClient` drop straight into a deep agent's `tools=` list. The harness doesn't care whether a tool is local Python or a remote MCP tool.

### 5.2 Virtual Filesystem

The harness provides a configurable **virtual filesystem** backed by a pluggable backend. This is a signature Deep Agents feature — the agent gets a real set of file tools it can use to offload context, save intermediate work, and read domain data.

| Tool | Description |
|------|-------------|
| `ls` | List files in a directory with metadata (size, modified time) |
| `read_file` | Read contents with line numbers; supports `offset`/`limit` for large files, and returns multimodal blocks for images/video/audio/documents |
| `write_file` | Create a new file, or overwrite an existing one |
| `edit_file` | Exact string replacements (with global replace mode) |
| `delete` | Delete a file, or a directory and its contents recursively |
| `glob` | Find files matching patterns (e.g., `**/*.py`) |
| `grep` | Search file contents (files-only, content-with-context, or counts) |
| `execute` | Run shell commands — **only** available with [sandbox backends](https://docs.langchain.com/oss/python/deepagents/sandboxes) |

**Backends** (where files actually live):

| Backend | Storage | Use when |
|---------|---------|----------|
| `StateBackend` | In-memory agent state | Ephemeral scratch space, tests, single run |
| `FilesystemBackend` | Local disk | Real files persist between runs |
| `StoreBackend` | LangGraph store | Long-term memory across threads/sessions |
| Composite | Routes paths to different backends | Mix disk + store + memory by path |
| Custom | Your implementation | Cloud storage, DB-backed files, policy hooks |

> **Version notes:** `delete` requires `deepagents >= 0.7.a1`; recursive directory deletion requires `0.7.a2`. Backends that don't support deletion have the tool auto-hidden from the model.

**Multimodal reads** — `read_file` can return image/video/audio/document content blocks for extensions like `.png`, `.jpg`, `.mp4`, `.mp3`, `.pdf`, `.pptx`, etc. (echoing the multimodal PDF work from Section 9).

**Hiding or restricting the file tools** — you're not forced to expose all of them:

```python
# Hide ALL filesystem tools via a harness profile
from deepagents import HarnessProfile, register_harness_profile

register_harness_profile(
	"anthropic:claude-sonnet-4-5",
	HarnessProfile(
		excluded_tools=frozenset(
			{"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"}
		),
	),
)
```

```python
# Expose only a READ-ONLY subset (requires deepagents >= 0.7.0a4)
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware

agent = create_deep_agent(
	model="claude-sonnet-4-5",
	middleware=[
		FilesystemMiddleware(backend=backend, tools=["read_file", "ls", "glob", "grep"]),
	],
)
```

> ⚠️ **Gotcha:** `read_file` **must** be in any `tools` allowlist — omitting it raises `ValueError`. And you cannot remove `FilesystemMiddleware` itself via `excluded_middleware` — it's required scaffolding. You can only hide the *model-visible tools*, not the middleware. (`execute`/`delete` are also dropped automatically when the backend doesn't support them.)

### 5.3 Filesystem Permissions

Declarative **permission rules** control which paths the agent can read or write. Each rule has:

- `operations`: `"read"` and/or `"write"`
- `paths`: glob patterns
- `mode`: `"allow"` or `"deny"`

Rules are evaluated **top to bottom, first-match-wins**. If no rule matches, the operation is **allowed** (default-allow).

```python
agent = create_deep_agent(
	model="anthropic:claude-sonnet-4-5",
	tools=[...],
	permissions=[
		{"operations": ["read", "write"], "paths": ["/workspace/**"], "mode": "allow"},
		{"operations": ["read", "write"], "paths": [".env", "**/secrets/**"], "mode": "deny"},
	],
)
```

This lets you confine an agent to `/workspace/`, protect `.env`/credentials, and give subagents **narrower** access than the parent. Permissions apply to the built-in filesystem tools — **not** to sandbox backends (which allow arbitrary `execute`). For custom validation, use [backend policy hooks](https://docs.langchain.com/oss/python/deepagents/backends#add-policy-hooks).

### 5.4 Code Execution — Sandboxes & Interpreters

Two distinct mechanisms, for two distinct needs:

| Mechanism | Tool | Gives you | Use when |
|-----------|------|-----------|----------|
| **Sandbox backend** | `execute` | Real shell in an isolated OS environment | Install deps, run tests, call CLIs, touch a real filesystem |
| **Interpreter** | `eval` | Scoped **QuickJS** (JavaScript) runtime | Loops, batching, deterministic data transforms, programmatic tool calling |

- **Sandboxes** implement `SandboxBackendProtocolV2`; when detected, the harness adds the `execute` tool. See [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes).
- **Interpreters** are lightweight — **no** shell, package installs, filesystem, or network access. Great for a safe programmable layer. See [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters).

> **Rule of thumb:** need the operating system? → sandbox. Need a safe little compute layer to orchestrate tools/data? → interpreter.

### 5.5 Streaming

[Event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming) exposes an agent run as **typed projections** — separate streams for messages, tool calls, values, and output. Uniquely, Deep Agents adds `stream.subagents`, so **each delegated task gets its own handle** with independent message, tool-call, and nested-subagent streams. That means you can render "Subagent 2 is running tool X" in a UI while the main agent keeps going.

---

## 6. Context Management

Context management controls **what the agent knows**, **how long it can operate within token limits**, and **what it retains across sessions**. Four layers.

### 6.1 Skills

**Skills** package specialized workflows, domain knowledge, and custom instructions. Each skill follows the [Agent Skills standard](https://agentskills.io/) and lives in a directory with a **`SKILL.md`** file (plus optional scripts, templates, and reference docs).

The key mechanism is **progressive disclosure**:

1. At **startup**, the agent reads only the `SKILL.md` **frontmatter** (name + description) — cheap, keeps context small.
2. Only when a task **needs** a skill does it read the full content.

```
skills/
├── pdf-processing/
│   ├── SKILL.md          ← frontmatter read at startup; body read on demand
│   ├── extract.py        ← supporting script
│   └── templates/
└── data-cleaning/
	└── SKILL.md
```

This is exactly the pattern this very repo's Copilot agent uses ("load the skill only when relevant"). It keeps a huge library of capabilities available without paying the token cost until you use one. See [Skills](https://docs.langchain.com/oss/python/deepagents/skills).

### 6.2 Memory

**Memory** gives the agent persistent context across conversations — coding style, preferences, conventions, project guidelines. It uses [`AGENTS.md` files](https://agents.md/) passed through the `memory=` parameter.

**Skills vs Memory — the critical distinction:**

| | Skills | Memory |
|-|--------|--------|
| **Loaded** | On demand (progressive) | **Always** (every run) |
| **File** | `SKILL.md` | `AGENTS.md` |
| **Purpose** | Rich capabilities you *sometimes* need | Preferences/rules you *always* want applied |
| **Cost** | Low until used | Paid every turn (keep it lean) |

The agent can also **update memory** from interactions and feedback, so learned preferences carry forward without re-stating them. Memory content is stored in the configured backend (`StateBackend`, `StoreBackend`, or `FilesystemBackend`). See [Memory](https://docs.langchain.com/oss/python/deepagents/customization#memory).

### 6.3 Summarization & Context Offloading

To survive long runs within token limits, the harness manages context in four parts:

1. **Input context** — system prompt, memory, skills, and tool prompts define the starting point.
2. **Compression** — built-in **summarization** compresses older conversation history, and **context offloading** moves large intermediate results into the virtual filesystem instead of the live prompt.
3. **Isolation** — subagents quarantine heavy subtasks and return only final results (see [Delegation](#7-delegation)).
4. **Long-term memory** — persistent storage in the virtual filesystem carries information across threads.

```
Live context window (small, hot)          Virtual filesystem (large, cold)
┌────────────────────────────┐            ┌──────────────────────────────┐
│ system prompt              │            │ big_search_results.json      │
│ memory (AGENTS.md)         │   offload  │ scraped_page_01.md           │
│ recent messages (summarized)│ ─────────► │ intermediate_report.md       │
│ pointers to offloaded files│            │ ...                          │
└────────────────────────────┘            └──────────────────────────────┘
```

Together these support multi-step tasks that **exceed a single context window** while reducing manual trimming and token usage. See [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering).

### 6.4 Prompt Caching

For **Anthropic** and **Amazon Bedrock** (Claude/Nova) models, `create_deep_agent` **automatically** applies prompt caching to the *static* sections of the system prompt — base instructions, memory, and skill content that repeat every turn. This avoids reprocessing the same tokens, cutting **latency and cost** on long-running agents. It's **on by default**, no config required. For other providers, see [provider-specific caching middleware](https://docs.langchain.com/oss/python/integrations/middleware#official-integrations).

---

## 7. Delegation

Delegation lets an agent break large problems into smaller, parallelizable units. Two layers.

### 7.1 Task Planning — `write_todos`

The harness provides a **`write_todos`** tool so the agent can maintain a structured task list during execution. Tasks track status:

- `pending` → not started
- `in_progress` → currently working
- `completed` → done

The list is **persisted in agent state**, giving a lightweight planning layer for long, multi-step work. (This is the same "visible plan the user can watch" idea behind the plan-tracking in this repo's Copilot agent.)

### 7.2 Subagents — the `task` tool

The built-in **`task`** tool lets the main agent spawn **ephemeral subagents** for isolated, long-running, or parallel work. Subagent execution provides:

- **Fresh context** — each invocation is a new agent instance with its own window.
- **Autonomous execution** — the subagent runs independently until done.
- **Single handoff** — it returns **one** final report to the main agent.
- **Configurable strategy** — use the default `general-purpose` subagent (enabled by default) or define **custom subagents** with narrower tools/permissions.
- **Stateless messaging** — subagents can't send multiple messages back; one report only.
- **Context & token efficiency** — heavy work stays isolated and is compressed into a compact result.

```
MAIN AGENT
   │  task("research competitor pricing")
   ├──────────────► SUBAGENT A (fresh context) ──► returns compact report ──┐
   │  task("summarize 200-page PDF")                                        │
   ├──────────────► SUBAGENT B (fresh context) ──► returns compact report ──┤
   │                                                                        ▼
   └──── continues with only the compact reports in its context ◄───────────┘
```

**Why this matters:** without subagents, a 200-page PDF summary would dump 200 pages into the main agent's context. With a subagent, those 200 pages live in the *subagent's* throwaway context, and the main agent only ever sees the 3-paragraph result.

You can run **without** subagents by disabling the auto-added one via a harness profile and passing no synchronous subagents — but you **cannot** remove `SubAgentMiddleware` via `excluded_middleware` (rejected by design). See [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents).

> 💡 **You've seen this pattern:** This repo's own Copilot Modernization Agent delegates each task to a `task-worker` sub-agent that runs in isolation and reports back — the exact same "orchestrator + ephemeral worker" architecture Deep Agents provides natively.

---

## 8. Steering — Human-in-the-Loop

Steering gives humans control at runtime. Deep Agents integrates with **LangGraph interrupts** so you can **pause for approval** on sensitive tool calls, via the `interrupt_on` parameter.

`interrupt_on` maps tool names to interrupt configs:

```python
agent = create_deep_agent(
	model="anthropic:claude-sonnet-4-5",
	tools=[edit_file_tool, send_email],
	interrupt_on={"edit_file": True},   # pause before EVERY edit_file call
)
```

When paused, you can:

- **Approve** the call as-is,
- **Add guidance**, or
- **Modify the tool inputs** before execution.

This is your runtime safety layer for destructive operations, expensive API calls, and interactive debugging. See [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop).

> 💡 **Connecting to Section 13:** This is LangGraph's `interrupt` primitive from the fundamentals section, surfaced as a first-class harness feature — you configure *which tools* pause instead of hand-wiring interrupt nodes.

---

## 9. Middleware & the Default Stack

Deep Agents is assembled from **middleware** — composable layers that add behavior. `create_deep_agent` wires a **default middleware stack** for the main agent, including `FilesystemMiddleware` (virtual FS) and `SubAgentMiddleware` (the `task` tool).

Key customization surfaces:

| Surface | What it does |
|---------|-------------|
| `middleware=[...]` | Add your own middleware, or pass a configured instance (e.g., `FilesystemMiddleware(tools=[...])`) to **override** a default |
| Harness **profiles** | Per-model tuning — `excluded_tools`, disabling the auto subagent, etc. |
| `excluded_tools` | Hide specific model-visible tools |
| `excluded_middleware` | Remove *optional* middleware — but **required scaffolding** (`FilesystemMiddleware`, `SubAgentMiddleware`) is rejected |

**Required vs optional:** Some middleware is structural. You can hide the *tools* it exposes, but you cannot delete the middleware itself. The intended lever is `excluded_tools` (surface) rather than `excluded_middleware` (scaffolding). See [Customization](https://docs.langchain.com/oss/python/deepagents/customization) and the [default middleware stack](https://docs.langchain.com/oss/python/deepagents/customization#default-stack-main-agent).

---

## 10. When to Use Deep Agents

A decision guide to place Deep Agents against the alternatives you've learned:

| Situation | Reach for | Why |
|-----------|-----------|-----|
| Complex, multi-step, long-running task | **Deep Agents** | Planning, files, subagents, memory built in |
| Agent that must offload/scan large data | **Deep Agents** | Virtual filesystem + context offloading |
| Parallel research / fan-out subtasks | **Deep Agents** | Native subagents via `task` |
| Simple, bounded agent, few tools | **LangChain `create_agent`** | Lighter, assemble only what you need |
| Fully custom, deterministic control flow | **Raw LangGraph graph** | Full topology control (like our Section 16 Agentic RAG) |
| One-shot LLM call, no tools | **Plain LCEL chain** | No agent loop needed at all |

**The trade-off in one line:** Deep Agents trades *control over topology* for *reliability out of the box*. Hand-built LangGraph gives you maximum control at maximum effort; Deep Agents gives you a proven harness you customize at the edges.

---

## 11. C# / .NET Analogies

Several harness concepts map cleanly onto patterns a senior .NET engineer already knows. These are the ones with a *genuine* structural parallel (not forced):

| Deep Agents concept | .NET / ASP.NET Core equivalent | Why the mapping holds |
|---------------------|-------------------------------|-----------------------|
| **Middleware stack** (`FilesystemMiddleware`, `SubAgentMiddleware`) | ASP.NET Core **middleware pipeline** (`app.UseX()`) | Both are ordered, composable layers wrapping a core loop; some are required, order matters, and you insert your own. |
| **Virtual filesystem + pluggable backend** | `IFileProvider` / `IFileSystem` abstraction over disk, memory, or cloud | Same file API; the backend (disk vs in-memory vs store) is swapped behind the interface. |
| **Filesystem permissions** (allow/deny, first-match) | **Authorization policies** / ordered `[Authorize]` + endpoint rules | Declarative rules evaluated in order, first-match-wins, default-allow if none match. |
| **Subagents via `task`** | `Task.Run` into a child DI scope that returns a result object | Fresh isolated context, runs autonomously, returns a single compact result — no shared mutable state leaking back. |
| **`write_todos` planning** | A `List<WorkItem>` with a `Status` enum (`Pending/InProgress/Completed`) persisted in state | A structured, observable task list tracked through the run. |
| **Skills (progressive disclosure)** | **Lazy loading** / `Lazy<T>` + plugin discovery reading only manifests until invoked | Read the manifest (frontmatter) cheaply; load the heavy body only on first real use. |
| **Memory (`AGENTS.md`, always loaded)** | `appsettings.json` / injected `IOptions<T>` read at startup | Always-present configuration/conventions applied to every request. |
| **`interrupt_on` (human-in-the-loop)** | An **approval gate** / manual checkpoint in a workflow (e.g., Durable Functions `WaitForExternalEvent`) | Execution pauses at a defined point until a human approves/edits, then resumes. |
| **Harness profile (`excluded_tools`)** | **Feature flags** / conditional service registration per environment | Toggle which capabilities a given model/environment sees without changing core code. |

> **Note:** Per this repo's convention, C# analogies are included only where the concept is a genuine core-programming or data-structure parallel. The *agent reasoning loop* itself has no meaningful C# equivalent — it's a statistical LLM behavior, not a control structure — so it's intentionally left off this table.

---

## Interview Q&A Anchors

**Q: What is Deep Agents and how is it different from `create_agent`?**
> **A:** Deep Agents is an "agent harness" — the standard tool-calling loop plus built-in capabilities: task planning (`write_todos`), a virtual filesystem, subagent spawning (`task`), skills, memory, summarization, and human-in-the-loop. `create_agent` builds a *custom* agent from LangChain primitives with none of that scaffolding. You reach for Deep Agents when you want reliability on complex, long-running tasks without assembling those pieces yourself; you reach for `create_agent` when you want a lean, bounded agent.

**Q: What does "agent harness" mean?**
> **A:** It's the scaffolding around the core agent loop (read messages → pick tool → run → observe → repeat). Every framework has that loop; the harness adds what makes agents production-reliable — context management, delegation, a filesystem, and steering. Deep Agents *is* a harness, pre-assembled and configurable via middleware.

**Q: How do Deep Agents relate to LangChain and LangGraph?**
> **A:** They're layers, not alternatives. LangChain is the framework (models, tools, messages). LangGraph is the runtime that actually executes the agent with durable state, streaming, and interrupts. Deep Agents is a harness built on both — it uses LangChain's building blocks and runs on the LangGraph runtime. You use all three together.

**Q: Why does a deep agent have a virtual filesystem?**
> **A:** Primarily for context management. Instead of stuffing large tool results or intermediate work into the live context window, the agent offloads them to files (`write_file`) and reads them back on demand. This keeps the prompt small and lets the agent handle tasks that exceed a single context window. The filesystem is pluggable — in-memory state, local disk, a LangGraph store, or a custom backend.

**Q: What's the difference between skills and memory?**
> **A:** Both provide context, but skills load on demand and memory loads always. Skills live in `SKILL.md` directories and use progressive disclosure — only the frontmatter is read at startup, and the full body loads only when a task needs it. Memory lives in `AGENTS.md` files, is loaded on every run, and holds persistent preferences/conventions. Rule of thumb: capabilities you sometimes need → skill; rules you always want → memory (keep it lean, it's paid every turn).

**Q: How do subagents help with context, and what are their constraints?**
> **A:** The `task` tool spawns an ephemeral subagent with a fresh, isolated context. It runs autonomously and returns a single compact report, so heavy work (e.g., reading a 200-page PDF) stays in the subagent's throwaway window and never pollutes the main agent's context. Constraints: subagents are stateless — they can't send multiple messages back, only one final handoff — and you can give them narrower tools/permissions than the parent.

**Q: What's the difference between a sandbox backend and an interpreter?**
> **A:** A sandbox backend exposes an `execute` tool for real shell commands in an isolated OS — use it to install dependencies, run tests, or call CLIs. An interpreter exposes an `eval` tool running scoped QuickJS (JavaScript) with no shell, network, or filesystem access — use it as a lightweight programmable layer for loops, batching, and programmatic tool calling. OS access → sandbox; safe compute layer → interpreter.

**Q: How do filesystem permissions work?**
> **A:** You pass a list of rules, each with `operations` (read/write), `paths` (globs), and `mode` (allow/deny). They're evaluated top to bottom, first-match-wins, and if no rule matches the operation is allowed by default. This confines agents to specific directories, protects secrets like `.env`, and lets subagents inherit narrower access. Note: permissions don't apply to sandbox backends, which allow arbitrary `execute`.

**Q: How do you keep a human in the loop?**
> **A:** Use the `interrupt_on` parameter, mapping tool names to interrupt configs (e.g., `{"edit_file": True}`). Built on LangGraph interrupts, it pauses before those tool calls so you can approve, add guidance, or modify the inputs before execution. It's the runtime safety layer for destructive operations and expensive API calls.

**Q: Why can't you remove `FilesystemMiddleware` or `SubAgentMiddleware`?**
> **A:** They're required scaffolding in the default stack, so removing them via `excluded_middleware` is intentionally rejected. Instead you hide their *model-visible tools* with `excluded_tools` (or a `FilesystemMiddleware(tools=[...])` allowlist), which removes the tools from the model's view while keeping the middleware in place. The design separates the tool *surface* (removable) from the structural middleware (not removable).

**Q: What does prompt caching do in Deep Agents and when is it automatic?**
> **A:** It caches the static sections of the system prompt — base instructions, memory, and skill content that repeat every turn — so those tokens aren't reprocessed on each call, cutting latency and cost. It's applied automatically for Anthropic and Amazon Bedrock (Claude/Nova) models with no configuration. For other providers you add provider-specific caching middleware.

---

## References

- [Deep Agents Overview (official)](https://docs.langchain.com/oss/python/deepagents/) — the source page for this section
- [Deep Agents Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart/)
- [Customization guide](https://docs.langchain.com/oss/python/deepagents/customization/) — default middleware stack, memory, overrides
- [Backends](https://docs.langchain.com/oss/python/deepagents/backends) — state, disk, store, composite, custom + policy hooks
- [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions) — rule structure and subagent inheritance
- [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) & [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters) — code execution
- [Skills](https://docs.langchain.com/oss/python/deepagents/skills) & the [Agent Skills standard](https://agentskills.io/)
- [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents) — default and custom subagents
- [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
- [Event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming) — typed projections + `stream.subagents`
- [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering) — summarization & offloading
- [Frameworks, runtimes, and harnesses](https://docs.langchain.com/oss/python/concepts/products) — where Deep Agents sits
- [Deep Agents vs. Claude Agent SDK](https://docs.langchain.com/oss/python/deepagents/comparison)
- [`deepagents` on PyPI](https://pypi.org/project/deepagents/) & [API reference](https://reference.langchain.com/python/deepagents/)
- [Agents.md standard](https://agents.md/) — the memory file format
