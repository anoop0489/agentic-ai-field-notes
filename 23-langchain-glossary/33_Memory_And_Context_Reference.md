# 33. Memory & Context in LangChain / LangGraph — Official Documentation Reference

see Section 22's [Deep Agent Skills overview](../22-deep-agent-skills/31_Deep_Agents_Official_Overview_Reference.md).

---

## The Core Idea

> **Remember this, forget the rest.** LLMs are **stateless** — each API call knows nothing about the last one. "Memory" is the machinery you build *around* the model to feed it the right past context. There are exactly **two kinds**, split by *how long the memory lives*:
> - **Short-term memory** = the current conversation (one **thread**), persisted by a **checkpointer**.
> - **Long-term memory** = facts that survive *across* conversations (many threads), persisted in a **store** under a **namespace**.
> Everything else — trimming, summarizing, profiles, semantic search — is just a technique for managing one of those two.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Why LLMs Need Memory](#1-why-llms-need-memory) | The stateless problem |
| 2 | [The Two Types of Memory](#2-the-two-types-of-memory) | Short-term vs long-term at a glance |
| 3 | [Short-Term Memory (Threads & Checkpointers)](#3-short-term-memory-threads--checkpointers) | State, threads, `thread_id`, checkpointers |
| 4 | [Managing Short-Term Memory](#4-managing-short-term-memory) | Trim, delete, summarize |
| 5 | [Long-Term Memory (Stores & Namespaces)](#5-long-term-memory-stores--namespaces) | Cross-thread persistence, semantic search |
| 6 | [The Three Types of Long-Term Memory](#6-the-three-types-of-long-term-memory) | Semantic, episodic, procedural |
| 7 | [Profile vs Collection](#7-profile-vs-collection) | Two ways to store semantic memory |
| 8 | [When to Write Memories: Hot Path vs Background](#8-when-to-write-memories-hot-path-vs-background) | The timing tradeoff |
| 9 | [Context Engineering: The Bigger Picture](#9-context-engineering-the-bigger-picture) | Mutability × lifetime; runtime context vs state vs store |
| 10 | [Production Backends](#10-production-backends) | Checkpointer/store databases |
| 11 | [C# Analogy](#11-c-analogy) | Mapping to session state, distributed cache, DI |
| 12 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |
| 13 | [References](#references) | Official docs |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Memory** | Remembering past interactions | A system that recalls information about previous interactions so an agent can be efficient and personalized. |
| **Short-term memory** | This conversation | Thread-scoped memory: the message history + state for a single session, persisted via a checkpointer. Read at the start of each step, updated at each step. |
| **Long-term memory** | Facts across conversations | Data shared *across* threads, recalled any time, in any thread. Stored as JSON in a **store** under custom **namespaces**. |
| **Thread** | One conversation | Groups multiple interactions in a session (like an email thread). Identified by `thread_id`. |
| **State** | The agent's working memory | The LangGraph state object (usually a `messages` list + custom fields). Acts as short-term memory during a run. |
| **Checkpointer** | Saves the thread | A `BaseCheckpointSaver` (e.g., `InMemorySaver`, `PostgresSaver`) that persists state so a thread can be paused/resumed. |
| **Store** | Saves cross-thread memory | A `BaseStore` (e.g., `InMemoryStore`, `PostgresStore`) that saves/recalls long-term memories by `namespace` + `key`. |
| **Namespace** | A folder for memories | A tuple (e.g., `(user_id, "memories")`) that organizes long-term memories hierarchically. |
| **Runtime context** | Injected dependencies | Static per-run data (user ID, DB connection, API clients) passed via the `context` argument — *not* the LLM's prompt. |

---

## 1. Why LLMs Need Memory

A chat model is **stateless**: it only knows what's in the messages you send *this call*. To hold a conversation, you must resend the growing history each turn. But history grows without bound, and that causes three problems:

1. **Context window overflow** — a full history may not fit, causing an irrecoverable error.
2. **Degraded quality** — even within the limit, LLMs get "distracted" by stale/off-topic content over long contexts.
3. **Cost & latency** — more tokens = slower and more expensive.

So memory is really two jobs: **(a) persist** the right information, and **(b) manage/trim** it so the context stays useful.

---

## 2. The Two Types of Memory

The docs classify memory by **recall scope** (how long it lives):

```
					 ┌──────────────────────────────────────────┐
					 │              MEMORY                        │
					 └──────────────────────────────────────────┘
						│                              │
		  ┌─────────────┘                              └──────────────┐
		  ▼                                                           ▼
  SHORT-TERM (thread-scoped)                          LONG-TERM (cross-thread)
  • one conversation / session                        • shared across all conversations
  • lives in agent STATE                              • lives in a STORE
  • persisted by a CHECKPOINTER                       • organized by NAMESPACE + KEY
  • keyed by thread_id                                • keyed by e.g. user_id
  • e.g. message history, uploaded files              • e.g. user profile, preferences, facts
```

| | Short-term | Long-term |
|---|-----------|-----------|
| **Scope** | One thread / conversation | Across all threads |
| **Where** | LangGraph **state** | LangGraph **store** |
| **Persistence** | **Checkpointer** | **Store** (with optional semantic search) |
| **Keyed by** | `thread_id` | `namespace` (e.g. `user_id`) + `key` |
| **Typical content** | Conversation history, intermediate results | User profile, preferences, learned facts |

---

## 3. Short-Term Memory (Threads & Checkpointers)

**Short-term memory = the conversation history for a single thread**, managed as part of the agent's **state**.

- A **thread** groups the interactions of one session (like email groups a conversation). It's identified by a **`thread_id`** you pass at invocation.
- LangGraph stores state as **thread-scoped checkpoints**. State updates when the agent is invoked or a step (e.g., a tool call) completes, and is read at the start of each step.
- To enable it, pass a **`checkpointer`** when creating the agent. That's the whole trick — same `thread_id` → the agent "remembers"; new `thread_id` → a fresh conversation.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
	model="openai:gpt-4o",
	tools=[get_user_info],
	checkpointer=InMemorySaver(),   # ← enables short-term memory
)

thread_config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": [{"role": "user", "content": "Hi! My name is Bob."}]}, thread_config)
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, thread_config)
# → "You are Bob!"  (because the same thread_id shares state)
```

> 💡 **State is more than messages.** Besides `messages`, state can hold uploaded files, retrieved documents, or generated artifacts. Extend `AgentState` and pass it via `state_schema` to add custom fields (e.g., `user_id`, `preferences`).

---

## 4. Managing Short-Term Memory

Long conversations eventually exceed the context window. Four common strategies (all applied **before** the LLM call, typically via `@before_model` middleware):

| Strategy | What it does | Tradeoff |
|----------|-------------|----------|
| **Trim** | Keep only the last *N* messages / tokens (`trim_messages` util) | Simple, but silently drops old info |
| **Delete** | Permanently remove messages from state via `RemoveMessage` (or `REMOVE_ALL_MESSAGES`) | Frees space, but info is gone forever |
| **Summarize** | Replace old messages with an LLM-generated summary (`SummarizationMiddleware`) | Preserves gist, costs an extra LLM call |
| **Custom** | Message filtering, relevance selection, etc. | Full control, more code |

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
	model="openai:gpt-4o",
	tools=[...],
	middleware=[
		SummarizationMiddleware(
			model="openai:gpt-4o-mini",
			trigger=("tokens", 4000),   # summarize once history passes 4k tokens
			keep=("messages", 20),      # always keep the last 20 messages verbatim
		)
	],
	checkpointer=InMemorySaver(),
)
```

> ⚠️ **Validity warning (from docs):** After deleting/trimming, the remaining history must still be **valid** for your provider. Many providers require the history to start with a `user` message, and require every `assistant` message with tool calls to be followed by matching `tool` result messages.

---

## 5. Long-Term Memory (Stores & Namespaces)

**Long-term memory persists across threads** — recall it any time, in any conversation. It lives in a **store**, not in thread state.

- Memories are JSON documents saved under a **`namespace`** (like a folder, e.g. `(user_id, "memories")`) and a **`key`** (like a filename).
- Enable it by passing a **`store`** when compiling the graph / creating the agent. Inside nodes/tools, access it via `runtime.store` (`.put`, `.get`, `.search`).

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("user_123", "memories")

store.put(namespace, "pref-1", {"rules": ["User likes short, direct answers"]})
item = store.get(namespace, "pref-1")
items = store.search(namespace, query="language preferences")   # semantic/filter search
```

**Semantic search** — configure the store with an embedding function so memories can be retrieved by *meaning*, not exact match:

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={"embed": init_embeddings("openai:text-embedding-3-small"), "dims": 1536})
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.search(("user_123", "memories"), query="I'm hungry", limit=1)   # → the pizza memory
```

> ⚠️ **Don't confuse the two "semantics."** *Semantic memory* (below) = a psychology term for storing **facts**. *Semantic search* = a **retrieval technique** using embeddings. They're unrelated concepts that happen to share a word.

---

## 6. The Three Types of Long-Term Memory

Borrowing from human psychology (and the [CoALA paper](https://arxiv.org/pdf/2309.02427)), long-term memory splits into three kinds:

| Type | Stores | Human analogy | Agent example | How it's implemented |
|------|--------|---------------|---------------|----------------------|
| **Semantic** | **Facts** | Things learned in school | Facts about a user | Profile or collection in the store |
| **Episodic** | **Experiences** | Things I did | Past successful actions | Few-shot examples (past input→output pairs) |
| **Procedural** | **Instructions / rules** | Instincts, motor skills | The agent's system prompt | Model weights + code + prompt; usually updated via prompt rewriting |

- **Semantic** → remember *what's true* (e.g., "the user is a plumber") to personalize responses.
- **Episodic** → remember *how a task was done well*, shown to the model as few-shot examples ("show, don't tell").
- **Procedural** → remember *the rules for behaving*. Agents rarely change weights/code, but they **can rewrite their own prompt** — often via **"Reflection"** / meta-prompting: feed the agent its current instructions + recent feedback and let it produce improved instructions.

---

## 7. Profile vs Collection

Two ways to organize **semantic** memory in the store:

| | **Profile** | **Collection** |
|---|-------------|----------------|
| **Shape** | One continuously-updated JSON document | Many small documents, appended over time |
| **Update** | Rewrite the whole profile each time (pass old → generate new, or JSON-patch) | Insert new docs; occasionally update/delete existing ones |
| **Strength** | Unified, comprehensive context in one place | Higher recall; easier to *add* new info without losing old |
| **Weakness** | Error-prone as it grows; needs strict schema/decoding | Harder search; model may over-insert or over-update |
| **Best when** | Small, well-scoped facts about one entity | Growing, varied knowledge where losing info is costly |

> 💡 The [Trustcall](https://github.com/hinthornw/trustcall) package helps manage collection updates (inserting vs patching), and evaluation (e.g., LangSmith) helps tune the behavior.

---

## 8. When to Write Memories: Hot Path vs Background

There are two moments an agent can *write* long-term memories:

| | **In the hot path** | **In the background** |
|---|--------------------|----------------------|
| **When** | During the run, before responding (agent uses a `save_memory`-style tool) | As a separate async task, after/between runs |
| **Pros** | Memories immediately usable; transparent to the user | No added latency; separates memory logic; can batch/dedupe |
| **Cons** | Adds latency; agent must multitask, hurting quality | Must decide *when* to trigger (timer, cron, manual); other threads may lag |
| **Example** | ChatGPT's `save_memories` tool | A scheduled memory-formation service |

---

## 9. Context Engineering: The Bigger Picture

**Context engineering** = building dynamic systems that give the LLM the right information and tools, in the right format, to do the task. The docs organize *all* context along two axes:

- **Mutability:** *Static* (doesn't change during a run — user metadata, tools, DB connections) vs *Dynamic* (evolves — conversation history, tool results).
- **Lifetime:** *Runtime* (one run) vs *Cross-conversation* (persists across sessions).

LangGraph gives **three** mechanisms combining these:

| Mechanism | Mutability | Lifetime | Access | Use for |
|-----------|-----------|----------|--------|---------|
| **Static runtime context** | Static | Single run | `context=` arg to `invoke`/`stream`; read via `runtime.context` | User IDs, DB connections, API clients (dependency injection) |
| **Dynamic runtime context (state)** | Dynamic | Single run | LangGraph **state** object | Conversation history, intermediate results = **short-term memory** |
| **Dynamic cross-conversation context (store)** | Dynamic | Cross-conversation | LangGraph **store** | User profiles, preferences = **long-term memory** |

> ⚠️ **Runtime context ≠ LLM context.** *Runtime context* is **dependency injection** — data your *code* needs (like a `user_id` used to fetch preferences). It is **not** the prompt sent to the LLM, nor the "context window" (max token count). You can *use* runtime context to *shape* the LLM context (e.g., look up preferences by `user_id` and add them to the prompt).

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class ContextSchema:
	user_name: str

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
	user_name = request.runtime.context.user_name        # static runtime context
	return f"You are a helpful assistant. Address the user as {user_name}."

agent = create_agent(model="openai:gpt-4o", tools=[...],
					 middleware=[personalized_prompt], context_schema=ContextSchema)

agent.invoke({"messages": [{"role": "user", "content": "hi"}]},
			 context=ContextSchema(user_name="John Smith"))
```

---

## 10. Production Backends

`InMemorySaver` / `InMemoryStore` are for development only (data lost on restart). In production, use a database-backed implementation:

| Backend | Checkpointer (short-term) | Store (long-term) |
|---------|---------------------------|-------------------|
| **Postgres** | `PostgresSaver` / `AsyncPostgresSaver` | `PostgresStore` / `AsyncPostgresStore` |
| **Redis** | `RedisSaver` / `AsyncRedisSaver` | `RedisStore` / `AsyncRedisStore` |
| **MongoDB** | `MongoDBSaver` / `AsyncMongoDBSaver` | — |
| **Oracle** | `OracleSaver` / `AsyncOracleSaver` | `OracleStore` / `AsyncOracleStore` |
| **SQLite / Azure Cosmos DB** | available | see docs |

**Two operational notes:**
1. **Migrations** — most DB-backed checkpointers/stores expose a `setup()` method to create the required schema. Call it once (the first time), ideally as a **dedicated deployment step** or at server startup.
2. **Subgraphs** — you only provide the checkpointer when compiling the **parent** graph; LangGraph propagates it to child subgraphs automatically.

You can also **inspect and clean up** thread state: `graph.get_state(config)` (current), `graph.get_state_history(config)` (all checkpoints), and `checkpointer.delete_thread(thread_id)` (delete a thread's checkpoints).

---

## 11. C# Analogy

For a .NET engineer, the mapping is intuitive:

| LangChain / LangGraph | C# / ASP.NET analogy |
|-----------------------|----------------------|
| Short-term memory (state + checkpointer) | **Session state** keyed by a session ID (`thread_id` ≈ session ID) |
| Checkpointer (`PostgresSaver`, `RedisSaver`) | The **session store provider** (SQL Server / Redis distributed cache backing `ISession`) |
| Long-term memory (store + namespace) | A **repository / database** keyed by `user_id` — data outliving any one session |
| Namespace tuple `(user_id, "memories")` | A **partition key** / table + key organization |
| Static runtime context (`context=`) | **Dependency injection** — scoped services (`DbContext`, `IHttpClientFactory`, current-user accessor) |
| `state_schema` extending `AgentState` | Extending a **DTO / view-model** with extra properties |
| Semantic search on the store | A **vector index** query (e.g., Azure AI Search vector search) |
| Trim / summarize middleware | A **request-pipeline middleware** that rewrites the payload before the downstream call |

> ⚠️ Analogy is for intuition. Unlike deterministic session state, an LLM's *use* of retrieved memory is probabilistic — retrieval puts facts in the prompt, but the model decides how to use them.

---

## Interview Q&A Anchors

**Q: What are the two types of memory in LangGraph and how do they differ?**
> Short-term (thread-scoped) memory is the conversation history/state for a single session, persisted by a **checkpointer** and keyed by `thread_id`. Long-term memory persists **across** threads, stored in a **store** as JSON under a `namespace` + `key` (e.g., keyed by `user_id`). Short-term is "this conversation"; long-term is "everything I know about this user."

**Q: How do you actually turn on memory?**
> For short-term, pass a `checkpointer` (e.g., `InMemorySaver` in dev, `PostgresSaver` in prod) and invoke with a `thread_id`. For long-term, pass a `store` and read/write it via `runtime.store.put/get/search` inside your nodes or tools.

**Q: A conversation is exceeding the context window. What are your options?**
> Trim to the last N tokens/messages, delete old messages with `RemoveMessage`, or summarize the history with `SummarizationMiddleware`. Summarization preserves the gist at the cost of an extra LLM call; trimming/deleting is cheap but drops information. Always ensure the remaining history stays provider-valid (starts with a user message; tool calls followed by tool results).

**Q: Explain semantic, episodic, and procedural memory.**
> Semantic stores **facts** (about a user), episodic stores **experiences** (past actions, used as few-shot examples), and procedural stores **rules/instructions** (usually the system prompt). Agents rarely change weights or code, but they can rewrite their own prompt — often via Reflection/meta-prompting.

**Q: Profile vs collection for semantic memory?**
> A profile is one continuously-rewritten JSON document — unified context, but error-prone as it grows. A collection is many small documents appended over time — higher recall and easier to add new facts, but harder to search and the model may over-insert/over-update. Choose profile for small well-scoped facts, collection when losing information is costly.

**Q: Hot path vs background memory writing?**
> Hot path writes memories during the run (immediately usable, transparent, but adds latency and splits the agent's attention). Background writes them asynchronously (no latency, cleaner separation, but you must decide when to trigger and other threads may briefly lack the new context).

**Q: What is "runtime context" and how is it different from the LLM context window?**
> Runtime context is **dependency injection** — static per-run data like a `user_id` or DB connection passed via the `context=` argument and read through `runtime.context`. It is *not* the prompt and *not* the token-limit "context window." You typically use it to *build* the LLM context, e.g., look up preferences by `user_id` and inject them into the system prompt.

**Q: Why can't the LLM just remember on its own?**
> LLMs are stateless — each call only sees the messages you pass. Any "memory" is application-layer machinery (state + checkpointer for short-term, store for long-term) that decides what past information to persist and re-inject into each call.

---

## References

- [Memory (conceptual overview)](https://docs.langchain.com/oss/python/concepts/memory)
- [Context (conceptual overview)](https://docs.langchain.com/oss/python/concepts/context)
- [Short-term memory in LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [Add memory (LangGraph)](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [Persistence / Checkpointers](https://docs.langchain.com/oss/python/langgraph/persistence) · [Stores](https://docs.langchain.com/oss/python/langgraph/stores)
- [CoALA paper (memory taxonomy)](https://arxiv.org/pdf/2309.02427) · [Trustcall](https://github.com/hinthornw/trustcall) · [Memory templates](https://github.com/langchain-ai/memory-agent)
