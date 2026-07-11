# 31. Deep Agents Overview — Official Documentation Reference

> **Context:** This document is a structured, faithful reference of the official [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/) documentation. It captures the full harness picture — the four core capabilities, the quickstart, and every built-in feature — so the upcoming **Deep Agent Skills** material has a complete foundation to build on. For the conceptual deep-dive and interview framing, see the companion [29. Deep Agents — Theory & Concepts](../21-deep-agents/29_Deep_Agents_Theory_And_Concepts.md).

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Is Deep Agents?](#1-what-is-deep-agents) | The library, its built-in capabilities, and the "agent harness" idea |
| 2 | [Quickstart](#2-quickstart) | Minimal `create_deep_agent` across providers |
| 3 | [Core Capabilities Map](#3-core-capabilities-map) | The four pillars at a glance |
| 4 | [Frameworks, Runtimes & Harnesses](#4-frameworks-runtimes--harnesses) | How Deep Agents relates to LangChain and LangGraph |
| 5 | [Execution Environment](#5-execution-environment) | Tools/MCP, virtual filesystem, permissions, code execution, streaming |
| 6 | [Context Management](#6-context-management) | Skills, memory, summarization/offloading, prompt caching |
| 7 | [Delegation](#7-delegation) | Task planning and subagents |
| 8 | [Steering](#8-steering) | Human-in-the-loop |
| 9 | [Get Started Links](#9-get-started-links) | Official next-step resources |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Deep Agents** | Batteries-included agent library | The easiest way to start building LLM-powered agents/applications, with built-in task planning, file systems for context management, subagent-spawning, and long-term memory. Works for any task, including complex multi-step ones. |
| **Agent harness** | Scaffolding around the tool-calling loop | The same core tool-calling loop as other frameworks, plus built-in capabilities that make agents reliable for real tasks. |
| **`create_deep_agent`** | The factory function | Builds a deep agent from a `model`, `tools`, and `system_prompt`, wiring the default middleware stack automatically. |
| **`deepagents`** | The PyPI package | A standalone library built on LangChain's core agent building blocks, running on the LangGraph runtime for durable execution, streaming, and human-in-the-loop. |
| **Execution environment** | Where the agent acts | Tools, virtual filesystem, filesystem permissions, code execution, and streaming. |
| **Context management** | What the agent knows over time | Skills, memory, summarization/offloading, and prompt caching. |
| **Delegation** | Splitting work | Task planning (`write_todos`) and subagents (`task`). |
| **Steering** | Human control at runtime | Human-in-the-loop approval via `interrupt_on`. |

---

## 1. What Is Deep Agents?

Deep Agents is the easiest way to start building agents and applications powered by LLMs — with **built-in capabilities** for task planning, file systems for context management, subagent-spawning, and long-term memory. You can use it for **any task**, including complex, multi-step tasks.

**Built-in capabilities (from the docs):**

| Capability | What it means |
|------------|---------------|
| **Take actions in an environment** | Take actions via tools, read and write files, execute code |
| **Connect to your data** | Load memories, skills, and domain knowledge at the right moment |
| **Manage growing context** | Summarize history and offload large results across long runs |
| **Parallelize tasks** | Delegate to general or specialized subagents in isolated context windows |
| **Stay in the loop** | Pause for human approval at critical decision points |
| **Improve over time** | Update memory, skills, and prompts based on real usage |

Deep Agents is an **["agent harness"](https://docs.langchain.com/oss/python/concepts/products#agent-harnesses-like-the-deep-agents-sdk)** — the same core tool-calling loop as other agent frameworks, but with built-in capabilities that make agents reliable for real tasks.

---

## 2. Quickstart

A minimal deep agent takes a `model`, `tools`, and a `system_prompt`. The `model` argument uses the `"provider:model-name"` string format, so the same code works across providers by swapping only that string.

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

**Provider strings shown in the official docs:**

| Provider | Model string (as documented) |
|----------|------------------------------|
| Google | `google_genai:gemini-3.5-flash` |
| OpenAI | `openai:gpt-5.5` |
| Anthropic | `anthropic:claude-sonnet-4-6` |
| OpenRouter | `openrouter:z-ai/glm-5.2` |
| Fireworks | `fireworks:accounts/fireworks/models/glm-5p2` |
| Baseten | `baseten:zai-org/GLM-5.2` |
| Ollama | `ollama:north-mini-code-1.0` |

> ⚠️ **Correction:** The official quickstart uses **forward-looking placeholder model names** (e.g., `gpt-5.5`, `gemini-3.5-flash`, `claude-sonnet-4-6`) that may not exist when you read this. Only the **`provider:model` format** is normative — substitute a model your provider currently offers (this repo's demo uses `openai:gpt-4o`).

> 💡 **Observability tip (from docs):** Trace requests, debug behavior, and evaluate outputs with [LangSmith](https://smith.langchain.com/). Follow the observability quickstart to set up, and see *Going to production* for deployment options. (This repo's demo pins the trace project via `os.environ["LANGSMITH_PROJECT"] = "deep-agents"`.)

---

## 3. Core Capabilities Map

Deep Agents organizes into **four pillars**. Everything else hangs off one of these.

```
┌───────────────────────┬───────────────────────┬───────────────────┬─────────────────┐
│ EXECUTION ENVIRONMENT │ CONTEXT MANAGEMENT     │ DELEGATION        │ STEERING        │
│ (⚡ where it acts)     │ (🗄️ what it knows)     │ (🕸️ split work)   │ (🧑 human control)│
├───────────────────────┼───────────────────────┼───────────────────┼─────────────────┤
│ Tools & MCP           │ Skills                 │ Task planning     │ Human-in-the-   │
│ Virtual filesystem    │ Memory                 │ (write_todos)     │ loop            │
│ Filesystem permissions│ Summarization &        │ Subagents (task)  │ (interrupt_on)  │
│ Code execution        │   context offloading   │                   │                 │
│ Streaming             │ Prompt caching         │                   │                 │
└───────────────────────┴───────────────────────┴───────────────────┴─────────────────┘
```

The official docs summarize the pillars in four cards:

| Pillar | Card description |
|--------|------------------|
| **Execution environment** | Tools, virtual filesystem, optional sandbox, and REPL (interpreter) |
| **Context management** | Skills, memory, summarization, context offloading, and prompt caching |
| **Delegation** | Subagent spawning and task planning |
| **Steering** | Human-in-the-loop approval and interrupts |

---

## 4. Frameworks, Runtimes & Harnesses

- **`deepagents`** is a standalone library built on top of **LangChain's** core agent building blocks. It uses the **LangGraph** runtime for durable execution, streaming, human-in-the-loop, and other features.
- **LangChain** is the framework that provides the core building blocks for your agents.

For building custom agents **without** these built-in capabilities, the docs recommend LangChain's [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) or a custom [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) workflow. For a side-by-side with Anthropic's harness, see [Deep Agents vs. Claude Agent SDK](https://docs.langchain.com/oss/python/deepagents/comparison).

```
deepagents  (harness)  ── built on ──►  LangChain (framework)
						── runs on ───►  LangGraph (runtime)
```

---

## 5. Execution Environment

The execution environment is where an agent acts. It has four layers, plus streaming to observe them.

### 5.1 Tools and MCP

Pass custom functions, LangChain tools, or tools from any [MCP server](https://docs.langchain.com/oss/python/langchain/mcp) via `tools=`. Deep Agents fully supports the Model Context Protocol.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
	model="anthropic:claude-sonnet-4-5",
	tools=[search, fetch_page, run_query],
)
```

### 5.2 Virtual Filesystem Access

A configurable virtual filesystem backed by [pluggable backends](https://docs.langchain.com/oss/python/deepagents/backends): in-memory state, local disk, LangGraph store, composite routing, or a custom backend (with permission rules).

**Backend file operations:**

| Tool | Description |
|------|-------------|
| `ls` | List files in a directory with metadata (size, modified time) |
| `read_file` | Read contents with line numbers; supports offset/limit for large files, and returns multimodal content blocks for non-text files |
| `write_file` | Create a new file, or overwrite an existing one |
| `edit_file` | Exact string replacements (with global replace mode) |
| `delete` | Delete a file, or a directory and its contents recursively |
| `glob` | Find files matching patterns (e.g., `**/*.py`) |
| `grep` | Search file contents (files only, content with context, or counts) |
| `execute` | Run shell commands — available with **sandbox backends** only |

> **Version notes (docs):** `delete` requires `deepagents` 0.7.a1+; recursive directory deletion requires 0.7.a2+. Backends that don't support deletion auto-hide the tool.

**Supported multimodal `read_file` extensions:**

| Type | Extensions |
|------|------------|
| Image | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`, `.heif` |
| Video | `.mp4`, `.mpeg`, `.mov`, `.avi`, `.flv`, `.mpg`, `.webm`, `.wmv`, `.3gpp` |
| Audio | `.wav`, `.mp3`, `.aiff`, `.aac`, `.ogg`, `.flac` |
| File | `.pdf`, `.ppt`, `.pptx` |

**Hiding all filesystem tools** — register a harness profile with `excluded_tools`:

```python
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

**Restricting to a subset** (requires `deepagents>=0.7.0a4`) — pass a `tools` allowlist to `FilesystemMiddleware`:

```python
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware

# Read-only agent: write_file, edit_file, delete, and execute are never shown
agent = create_deep_agent(
	model="claude-sonnet-4-5",
	middleware=[
		FilesystemMiddleware(backend=backend, tools=["read_file", "ls", "glob", "grep"]),
	],
)
```

> ⚠️ **Rules from the docs:** `read_file` **must** appear in any allowlist (omitting it raises `ValueError`). `FilesystemMiddleware` itself is **required scaffolding** — removing it via `excluded_middleware` is intentionally rejected; only hide the model-visible tools. `execute`/`delete` are auto-dropped when the backend doesn't support them.

The virtual filesystem underpins several other capabilities — **skills, memory, code execution, and context management** — and is available when building custom tools/middleware.

### 5.3 Filesystem Permissions

Declarative rules control which paths the agent can read/write. Each rule has:

- `operations`: `"read"` and/or `"write"`
- `paths`: glob patterns
- `mode`: `"allow"` or `"deny"`

Rules are evaluated **top to bottom, first-match-wins**; if none match, the operation is **allowed**. This restricts agents to specific directories (e.g., `/workspace/`), protects sensitive files (`.env`, credentials), and gives subagents narrower access than the parent. Permissions do **not** apply to sandbox backends (arbitrary `execute`). For custom logic, use [backend policy hooks](https://docs.langchain.com/oss/python/deepagents/backends#add-policy-hooks).

### 5.4 Code Execution

| Mechanism | Tool | Use for |
|-----------|------|---------|
| [Sandbox backends](https://docs.langchain.com/oss/python/deepagents/sandboxes) | `execute` | Install deps, run tests, call CLIs, work with an OS filesystem. Implements `SandboxBackendProtocolV2`; when detected, the harness adds `execute`. |
| [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters) | `eval` | Lightweight programmable layer for loops, batching, deterministic transforms, programmatic tool calling — scoped QuickJS (JavaScript), **no** shell/package/filesystem/network access. |

### 5.5 Streaming

[Event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming) exposes runs as typed projections for messages, tool calls, values, and output. Deep Agents adds **`stream.subagents`** so each delegated task gets its own handle with independent message, tool-call, and nested-subagent streams.

---

## 6. Context Management

Controls what the agent knows, how long it can operate within token limits, and what it retains across sessions. Four layers.

### 6.1 Skills

> 🎯 **This is the focus of Section 22.**

Skills package specialized workflows, domain knowledge, and custom instructions for your deep agent. Each skill follows the [Agent Skills standard](https://agentskills.io/) and lives in a directory with a **`SKILL.md`** file (and optionally scripts, templates, reference docs, and other resources).

Deep Agents load skills with **progressive disclosure**: the agent reads `SKILL.md` **frontmatter** at startup, then reads full skill content **only when a task needs it**. This keeps startup context compact while making rich capabilities available on demand.

See [Skills](https://docs.langchain.com/oss/python/deepagents/skills).

### 6.2 Memory

Memory gives persistent context across conversations (coding style, preferences, conventions, project guidelines). It uses [`AGENTS.md` files](https://agents.md/) passed via the `memory` parameter. Unlike skills, **memory files are always loaded**, and content is stored in the configured backend (`StateBackend`, `StoreBackend`, or `FilesystemBackend`). The agent can also **update memory** based on interactions and feedback.

### 6.3 Summarization and Context Offloading

The harness manages context so agents handle long-running work within token limits. The context flow has four parts:

- **Input context** — system prompt, memory, skills, and tool prompts define the start.
- **Compression** — built-in offloading and summarization compress conversation history and large intermediate results.
- **Isolation** — subagents quarantine heavy subtasks and return only final results.
- **Long-term memory** — persistent storage in the virtual filesystem carries information across threads.

See [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering) and [Multimodal](https://docs.langchain.com/oss/python/deepagents/multimodal).

### 6.4 Prompt Caching

For **Anthropic** and **Amazon Bedrock** (Claude/Nova) models, `create_deep_agent` **automatically** applies prompt caching to static system-prompt sections (base instructions, memory, and skill content that repeat each turn), reducing latency and cost. **Enabled by default, no configuration required.** For other providers, see [provider-specific caching middleware](https://docs.langchain.com/oss/python/integrations/middleware#official-integrations).

---

## 7. Delegation

Breaks large problems into smaller, parallelizable units. Two layers.

### 7.1 Task Planning

The harness provides a **`write_todos`** tool for a structured task list during execution. Tasks track status — `'pending'`, `'in_progress'`, `'completed'` — and are persisted in agent state, giving a lightweight planning layer for long, multi-step work.

### 7.2 Subagents

The built-in **`task`** tool lets the main agent create ephemeral subagents for isolated, long-running, multi-step, or parallel tasks:

- **Fresh context** — each invocation is a new agent instance with its own context.
- **Autonomous execution** — runs independently until completion.
- **Single handoff** — returns one final report to the main agent.
- **Configurable strategy** — use the default `general-purpose` subagent (enabled by default) or define custom subagents.
- **Stateless messaging** — subagents can't send multiple messages back.
- **Context/token efficiency** — heavy work stays isolated and is compressed into a compact result.

> **Note (docs):** To run without the `task` tool, disable the auto-added subagent via a harness profile and pass no synchronous subagents — do **not** remove `SubAgentMiddleware` via `excluded_middleware` (rejected). See [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents).

---

## 8. Steering

The steering component gives humans control over agent behavior at runtime (and sets filesystem permissions for agent work).

### Human-in-the-loop

Deep Agents integrates with LangGraph interrupts to pause for approval on sensitive tool calls via the `interrupt_on` parameter — a mapping of tool names to interrupt configs. For example, `interrupt_on={"edit_file": True}` pauses before every edit, letting you approve the call, add guidance, or modify tool inputs before execution. This is your runtime safety layer for destructive operations, expensive API calls, and interactive debugging. See [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop).

---

## 9. Get Started Links

Official next-step resources from the overview page:

| Resource | Link |
|----------|------|
| Quickstart | https://docs.langchain.com/oss/python/deepagents/quickstart |
| Customization | https://docs.langchain.com/oss/python/deepagents/customization |
| Deep Agents Code | https://docs.langchain.com/oss/python/deepagents/code/overview |
| ACP (use in code editors) | https://docs.langchain.com/oss/python/deepagents/acp |
| API Reference | https://reference.langchain.com/python/deepagents/ |

---

## References

- [Deep Agents Overview (source page)](https://docs.langchain.com/oss/python/deepagents/)
- [Documentation index (`llms.txt`)](https://docs.langchain.com/llms.txt)
- [Skills](https://docs.langchain.com/oss/python/deepagents/skills) · [Agent Skills standard](https://agentskills.io/)
- [Backends](https://docs.langchain.com/oss/python/deepagents/backends) · [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions)
- [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) · [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters)
- [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents) · [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
- Companion: [29. Deep Agents — Theory & Concepts](../21-deep-agents/29_Deep_Agents_Theory_And_Concepts.md)
