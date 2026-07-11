# 19. Reflexion Agent — Tool-Augmented Self-Improvement with LangGraph

> **Context:** A Reflexion agent that extends the basic reflection pattern (Section 14) with tool use — fetching real-time data via Tavily to ground revisions in facts and citations.

---

## The Core Idea (from the Reflexion paper)

> **Remember this, forget the rest.** The code is just plumbing. AI writes the code. You need to remember the TECHNIQUE.

**Paper:** [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366) — Northeastern, MIT, Princeton.

**The technique in one sentence:**

> "Write an answer, be honest about what's wrong with it, Google the gaps, rewrite — repeat."

**The technique in five steps:**

```
Step 1:  ANSWER     →  Write your best answer using what you know
Step 2:  CRITIQUE   →  Be brutally honest — what's wrong? what's missing?
Step 3:  RESEARCH   →  Use any tool to get real data about what you got wrong
Step 4:  REWRITE    →  Rewrite using the real data + add citations
Step 5:  REPEAT     →  Go back to Step 2 until good enough
```

**Why it works:**
- Step 2 forces **self-awareness** — the LLM can't pretend its answer is perfect
- Step 3 **grounds** the improvement in real data — not just rewording the same wrong info
- Step 4 adds **citations** — the answer becomes verifiable, not just confident-sounding

**The tool can be ANYTHING — web search is not mandatory:**

The paper's concept is **Reflection + any tool that fills the knowledge gap.** The tool depends on your domain:

| What the critique identifies | Tool that fills the gap |
|------------------------------|------------------------|
| "Missing recent funding data" | Web search (Tavily, Google) |
| "SQL query might be wrong" | Code executor (run the SQL, check results) |
| "Need to verify this API exists" | API call (hit the endpoint, check response) |
| "Math calculation seems off" | Calculator tool |
| "Need the actual document text" | RAG retriever (vector DB lookup) |
| "Not sure about this law/rule" | Database query (legal DB, compliance DB) |

This example uses Tavily (web search) because it's a research article. But if you were building a **code generator**, the tool would be a code executor. For a **data analyst**, it would be a SQL runner. The pattern stays the same — only the tool changes.

**Reflexion vs Reflection (the difference):**

```
Reflection (Section 14):  Answer → Critique → Rewrite → Critique → Rewrite
                          (no internet, just the LLM talking to itself)

Reflexion (Section 15):   Answer → Critique → GOOGLE → Rewrite → Critique → GOOGLE → Rewrite
                          (internet fills the gaps the critique identified)
```

**RAG vs Reflexion — the exam analogy:**

```
RAG       = Study your notes first, then take the exam
            (Retrieve from YOUR knowledge base → then answer)
            One retrieval pass — if your question is vague, you might
            miss relevant chunks buried deeper in your own corpus

Reflexion = Take the exam with whatever you know, check your answers,
            study ONLY what you got wrong, retake — repeat until you pass
            (Answer → critique → targeted search for gaps → rewrite → repeat)
            Each attempt is better because you know exactly what to fix
```

| | RAG | Reflexion |
|--|-----|-----------|
| **Setup cost** | High — ingest documents into vector store ONCE upfront | None — searches on-demand each iteration |
| **Data source** | Your pre-ingested corpus (vector DB) | Live/external sources (web, APIs, DBs) — whatever the tool queries |
| **When** you search | BEFORE answering | AFTER answering + critiquing |
| **What** you search | Your ingested corpus (broad similarity match on the question) | Only what the critique said is wrong (targeted queries) |
| **Who** decides what to search | The user's question | The LLM itself (it knows what it got wrong) |
| **How many passes** | One shot | Multiple — each pass catches new errors |
| **Best when** | You know nothing about the topic | You know most of it but need to verify |

Neither is "better" — they're for different situations. You can even combine them: RAG for the first draft, Reflexion to critique and improve.

**The concept vs the implementation:**

| The CONCEPT (remember this) | The IMPLEMENTATION (AI writes this) |
|-----------------------------|-------------------------------------|
| Answer + critique yourself | Pydantic schemas, `tool_choice`, `bind_tools` |
| Google the gaps | ToolNode, `StructuredTool.from_function`, name matching |
| Rewrite with real data | `MessagesPlaceholder`, `.partial()`, LCEL pipe |
| Repeat until done | Conditional edges, `ToolMessage` counting |

The left column is the technique you need for interviews and system design. The right column is LangChain-specific plumbing that changes with every library version.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [What Is a Reflexion Agent?](#1-what-is-a-reflexion-agent) | How it extends basic reflection with tools and structured output |
| 2 | [How It Actually Works](#2-how-it-actually-works--plain-english-walkthrough) | Plain English walkthrough, naming confusion cleared up, full step-by-step |
| 3 | [Architecture Overview](#3-architecture-overview) | Three nodes, one cycle, tool-augmented revision |
| 4 | [The Structured Output Trick](#4-the-structured-output-trick) | Forcing the LLM to produce answer + critique + queries in one call |
| 5 | [The Graph Structure](#5-the-graph-structure) | Nodes, edges, state, stop condition |
| 6 | [Advanced Prompt Engineering](#6-advanced-prompt-engineering) | How the prompt forces self-critique and improvement, `MessagesPlaceholder` deep dive |
| 7 | [The ToolNode Name-Matching Trick](#7-the-toolnode-name-matching-trick) | Why tools are named after Pydantic schemas |
| 8 | [Comparison: Reflection vs Reflexion](#8-comparison-reflection-vs-reflexion) | Side-by-side with Section 14 |
| 9 | [Code Walkthrough](#9-code-walkthrough) | Step-by-step through all 4 files |
| 10 | [LangGraph Workflow Patterns](#10-langgraph-workflow-patterns) | Prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer |
| 11 | [ToolNode Deep Dive](#11-toolnode-deep-dive) | How ToolNode works, input/output formats, error handling, ToolRuntime |
| 12 | [Production-Grade Checkpointers](#12-production-grade-checkpointers) | PostgresSaver, async, time travel, HIL — the production gap |
| 13 | [Interview Q&A Anchors](#interview-qa-anchors) | Quick-fire answers |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|-------------|----------------|
| **Reflexion** | Reflection + tools + citations | An agent architecture from the Reflexion paper (Shinn et al., 2023) that combines self-critique with tool execution to iteratively improve answers using external data. |
| **Structured Output** | LLM forced to return a schema | Using `tool_choice` to force the LLM to always return data matching a Pydantic model — no freeform text allowed. |
| **AnswerQuestion** | Schema for initial response | Pydantic model: answer + self-critique (Reflection) + search queries. |
| **ReviseAnswer** | Schema for revised response | Extends AnswerQuestion with a `references` field for citations. |
| **tool_choice** | Force a specific tool call | Parameter that tells the LLM "you MUST call this tool" — eliminates the option to respond with plain text. |
| **ToolNode** | Prebuilt node that executes tools | LangGraph's built-in node that looks up tools by name and executes them, returning ToolMessages. |
| **Actor Prompt** | Shared template for both chains | One prompt template with a variable `{first_instruction}` — instantiated differently for draft vs revise. |
| **MessagesPlaceholder** | Injects full history into prompt | A prompt slot that gets replaced at runtime with the entire `messages` list from graph state — lets the LLM see the user question, prior answers, critiques, and search results. |
| **Prompt Chaining** | Linear A → B → C | Each LLM call processes the output of the previous one. No branching, no cycles. |
| **Parallelization** | Multiple LLM calls at once | Independent subtasks run simultaneously, results aggregated. |
| **Routing** | LLM picks a branch | LLM classifies input and routes to specialized handlers. No cycles. |
| **Orchestrator-Worker** | Plan → spawn → synthesize | One LLM plans subtasks, dynamically spawns workers via Send API, collects results. |
| **Evaluator-Optimizer** | Generate ↔ Evaluate loop | One LLM generates, another evaluates. Cycle until quality threshold met. |
| **Send API** | Dynamic worker spawning | LangGraph API that creates node executions at runtime based on state content. |
| **ToolRuntime** | State injection for tools | Lets tools access graph state and run-scoped context that the LLM didn't generate. |

---

## 1. What Is a Reflexion Agent?

The Reflexion agent extends the basic reflection pattern (Section 14) with **three key additions**:

1. **Tools** — Uses Tavily search to fetch real-time data
2. **Structured output** — Forces the LLM to return answer + critique + search queries in one call
3. **Citations** — Revised answers must include numbered references to source URLs

```
┌─────────────────────────────────────────────────────────────┐
│              REFLEXION vs REFLECTION                         │
│                                                             │
│  Section 14 (Reflection):                                   │
│    Generate ←→ Reflect (pure LLM-to-LLM, no tools)         │
│                                                             │
│  Section 15 (Reflexion):                                    │
│    Draft → Search → Revise → Search → Revise → END         │
│    (LLM + tools + citations + structured output)            │
└─────────────────────────────────────────────────────────────┘
```

**Paper:** [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366)

**LangChain blog:** [Reflection Agents](https://www.langchain.com/blog/reflection-agents)

---

## 2. How It Actually Works — Plain English Walkthrough

> ⚠️ **Read this section first.** The rest of the doc goes deep into code and techniques. This section explains what actually happens at each step, with no code jargon.

### The Naming Confusion — Cleared Up

The same step is often given different names at different layers. Here's the cheat sheet:

```
What it DOES               Graph Node Name    Python Function    Chain (prompt+LLM)
─────────────────────────  ─────────────────  ─────────────────  ──────────────────
Writes the first answer    "draft"            draft_node()       first_responder
Googles the search terms   "execute_tools"    (ToolNode)         (Tavily search)
Rewrites with real data    "revise"           revise_node()      revisor
```

**Three names, three layers, same step.** When this doc says "draft node" or "first_responder" or "draft_node()" — it's all the same thing: the step that writes the first answer.

### What `tool_choice` Really Means

This is the most misleading name in the whole project:

```
tool_choice="AnswerQuestion"

❌ Does NOT mean: "Call a tool called AnswerQuestion"
✅ Actually means: "Return your answer as JSON matching this form"
```

The word "tool" is misleading. It's just a trick to force the LLM to return structured JSON instead of freeform text. Think of it as **"forced JSON format"**, not "call a tool."

### The Full Walkthrough — What Happens When You Run `main.py`

**You ask:** "Write about AI-Powered SOC startups that raised capital."

---

**Step 1 — DRAFT NODE (LLM writes, no internet)**

The LLM uses ONLY its training data. It has no internet access. It produces ONE JSON response with three parts:

```
┌─────────────────────────────────────────────────────────────┐
│ What the LLM returns (ONE call, ONE response):              │
│                                                             │
│ answer:                                                     │
│   "AI-Powered Security Operations Centers (SOC) represent   │
│    a growing market. Companies like SentinelOne and         │
│    CrowdStrike use AI for threat detection..."              │
│                                                             │
│ reflection:                                                 │
│   missing: "No specific funding amounts. No mention of      │
│            early-stage startups. Missing market size data."  │
│   superfluous: "Too much general background about           │
│                 traditional SOCs."                           │
│                                                             │
│ search_queries:      ← JUST TEXT. Not actual searches.      │
│   - "AI SOC startups 2024 funding rounds"                   │
│   - "autonomous security operations center market size"     │
│   - "AI-powered SOC startup seed series A 2024"             │
│                                                             │
│ ⚠️ The LLM did NOT Google anything.                        │
│    It just wrote down what it WOULD search for.             │
│    These are plain strings sitting in the JSON.             │
└─────────────────────────────────────────────────────────────┘
```

**State after Step 1:** `[HumanMessage, AIMessage(answer + critique + search strings)]`

---

**Step 2 — EXECUTE_TOOLS NODE (Tavily actually searches the internet)**

A completely separate node takes those search query strings and sends them to Tavily (a search engine API). The LLM is NOT involved here. This is pure Python code calling an API.

```
┌─────────────────────────────────────────────────────────────┐
│ Input: ["AI SOC startups 2024 funding rounds",              │
│         "autonomous SOC market size",                       │
│         "AI-powered SOC startup seed series A 2024"]        │
│                                                             │
│ Tavily searches the real internet and returns:              │
│                                                             │
│ Result 1: "Torq raised $70M Series C for AI SOC..."        │
│           URL: https://techcrunch.com/...                   │
│                                                             │
│ Result 2: "AI security market expected to reach $38B..."    │
│           URL: https://marketsandmarkets.com/...            │
│                                                             │
│ Result 3: "Intezer raised $33M for autonomous SOC..."       │
│           URL: https://venturebeat.com/...                  │
│                                                             │
│ ✅ THIS is where the internet is used.                     │
│    Real URLs. Real data. Real numbers.                      │
└─────────────────────────────────────────────────────────────┘
```

**State after Step 2:** `[HumanMessage, AIMessage(draft), ToolMessage(real search results)]`

---

**Step 3 — REVISE NODE (LLM rewrites using real data)**

Now the LLM sees EVERYTHING: the original question, its first draft, its own critique ("I'm missing funding data"), AND the real search results. It rewrites:

```
┌─────────────────────────────────────────────────────────────┐
│ What the LLM sees (via MessagesPlaceholder):                │
│                                                             │
│   1. Your question: "Write about AI-Powered SOC..."         │
│   2. Its own draft: "AI-Powered SOC represents..."          │
│   3. Its own critique: "Missing funding amounts"            │
│   4. Real search results: "Torq $70M, Intezer $33M..."     │
│                                                             │
│ What it returns (BETTER version):                           │
│                                                             │
│ answer:                                                     │
│   "AI-Powered SOCs are transforming cybersecurity.          │
│    Torq raised $70M Series C [1], Intezer secured           │
│    $33M [2], and the market is projected to reach            │
│    $38B by 2028 [3]..."                                     │
│                                                             │
│ reflection:                                                 │
│   missing: "Could add comparison of detection rates.        │
│            No mention of open-source alternatives."         │
│   superfluous: "Market size paragraph is too long."         │
│                                                             │
│ search_queries:                                             │
│   - "AI vs traditional SOC detection rate comparison"       │
│   - "open source AI security tools 2024"                    │
│                                                             │
│ references:         ← NEW field (only in ReviseAnswer)      │
│   - "[1] https://techcrunch.com/torq-70m-series-c"          │
│   - "[2] https://venturebeat.com/intezer-33m"               │
│   - "[3] https://marketsandmarkets.com/ai-security"         │
│                                                             │
│ ✅ Now the answer has real data AND citations.              │
└─────────────────────────────────────────────────────────────┘
```

**State after Step 3:** `[HumanMessage, AIMessage(draft), ToolMessage(results), AIMessage(revision1)]`

---

**Step 4 — event_loop checks: should we keep going?**

```
Count ToolMessages in state = 1
MAX_ITERATIONS = 2
1 ≤ 2 → YES, keep going → route back to EXECUTE_TOOLS
```

---

**Step 5 — EXECUTE_TOOLS again (new searches from the revision)**

Takes the NEW search queries from Step 3 ("AI vs traditional SOC detection rate", "open source AI security tools 2024") and Googles them. Returns fresh results.

---

**Step 6 — REVISE again (even better answer)**

LLM now sees: original question + draft + first search results + first revision + second search results. Writes an even better answer with more citations.

---

**Step 7 — event_loop checks again**

```
Count ToolMessages in state = 2
2 ≤ 2 → YES, one more round
```

---

**Step 8 — EXECUTE_TOOLS one more time**

---

**Step 9 — REVISE final time (best answer yet)**

---

**Step 10 — event_loop checks final time**

```
Count ToolMessages in state = 3
3 > 2 → STOP → route to END ✅
```

---

### Summary: The Three Roles

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  DRAFT / REVISE  │     │  EXECUTE_TOOLS   │     │    event_loop    │
│                  │     │                  │     │                  │
│  WHO: The LLM    │     │  WHO: Tavily API │     │  WHO: Python     │
│                  │     │                  │     │       code       │
│  DOES: Writes    │     │  DOES: Searches  │     │                  │
│  text, critiques │     │  the internet    │     │  DOES: Counts    │
│  itself, suggests│     │  for real data   │     │  iterations,     │
│  what to Google  │     │                  │     │  decides if we   │
│                  │     │  NO LLM here.    │     │  keep going or   │
│  NO internet     │     │  Just an API     │     │  stop            │
│  access here.    │     │  call.           │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
     Writes text           Gets real data         Controls the loop
```

---

## 3. Architecture Overview

### Mermaid Diagram (project graph)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph LR;
        __start__([<p>__start__</p>]):::first
        draft(draft)
        execute_tools(execute_tools)
        revise(revise)
        __end__([<p>__end__</p>]):::last
        __start__ --> draft;
        draft --> execute_tools;
        execute_tools --> revise;
        revise -.-> draft;
        revise -.-> execute_tools;
        revise -.-> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

### What Each Node Does

| Node | Input | Output | Purpose |
|------|-------|--------|---------|
| **draft** | User question | AIMessage with tool_calls (AnswerQuestion) | Initial answer + critique + search queries |
| **execute_tools** | AIMessage with search_queries | ToolMessage with search results | Fetch real-time data from Tavily |
| **revise** | All messages so far | AIMessage with tool_calls (ReviseAnswer) | Improved answer + new critique + new queries + citations |

---

## 4. The Structured Output Trick

The most powerful technique in this agent: **the LLM is forced to self-critique as part of its output format**.

```python
class AnswerQuestion(BaseModel):
	answer: str           # The actual ~250 word answer
	reflection: Reflection  # Self-critique (missing + superfluous)
	search_queries: List[str]  # Queries to improve the answer

class ReviseAnswer(AnswerQuestion):
	references: List[str]  # Citations from search results
```

**Why this works:**
- `tool_choice="AnswerQuestion"` means the LLM CANNOT return plain text
- It MUST fill in every field including `reflection` and `search_queries`
- The self-critique isn't optional — it's structurally enforced
- The search queries bridge the LLM's self-awareness gap with real data

**The revision chain sees:**
1. The original answer (from draft)
2. The critique (from the same draft response)
3. The search results (from Tavily)
4. And must produce a BETTER answer + NEW critique + NEW queries + CITATIONS

---

## 5. The Graph Structure

### State

Uses `MessagesState` directly — just a growing list of messages:

```python
from langgraph.graph import MessagesState
builder = StateGraph(MessagesState)
```

### Stop Condition

Counts `ToolMessage` instances to track iterations:

```python
MAX_ITERATIONS = 2

def event_loop(state: MessagesState):
	count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
	if count_tool_visits > MAX_ITERATIONS:
		return END
	return "execute_tools"
```

### Edges

| From | To | Type | Condition |
|------|-----|------|-----------|
| START | draft | Fixed | Always |
| draft | execute_tools | Fixed | Always (draft always produces search queries) |
| execute_tools | revise | Fixed | Always |
| revise | execute_tools or END | Conditional | ToolMessage count > MAX_ITERATIONS → END |

---

## 6. Advanced Prompt Engineering

Making the LLM **actually incorporate critique** is harder than generating the critique. Key techniques:

### Shared Actor Prompt

Both chains use the same template with different `{first_instruction}`:

```python
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),  # ← ALL history goes here
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
```

### MessagesPlaceholder — The History Injection Point

`MessagesPlaceholder` is **the most important piece** of this prompt template. Without it, the LLM would only see the system instructions — it would have no idea what the user asked, what it already answered, or what the search results said.

**What it does:** At runtime, `MessagesPlaceholder(variable_name="messages")` gets replaced with the **entire `messages` list** from the graph state. That list grows with every step:

```
Step 1 (draft):
  messages = [HumanMessage("Write about AI-Powered SOC...")]
  → MessagesPlaceholder injects: [HumanMessage]

Step 4 (first revise):
  messages = [HumanMessage, AIMessage(draft), ToolMessage(search results)]
  → MessagesPlaceholder injects: [HumanMessage, AIMessage, ToolMessage]

Step 7 (second revise):
  messages = [HumanMessage, AIMessage, ToolMessage, AIMessage(revision1), ToolMessage(new results)]
  → MessagesPlaceholder injects: ALL 5 messages
```

**What the LLM actually sees** (expanded prompt on the first revise):

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM: You are expert researcher. Current time: 2025-01-15... │
│   1. Revise your previous answer using the new information...  │
│   2. Reflect and critique your answer. Be severe...            │
│   3. Recommend search queries...                               │
├─────────────────────────────────────────────────────────────────┤
│ HUMAN: Write about AI-Powered SOC...          ← from state     │
│ AI: {answer: "...", reflection: {...}, ...}    ← draft output  │
│ TOOL: [Tavily search results...]              ← search data    │
├─────────────────────────────────────────────────────────────────┤
│ SYSTEM: Answer the user's question above using the required    │
│         format.                                                │
└─────────────────────────────────────────────────────────────────┘
```

**Why this matters for the Reflexion pattern:**
- The revisor sees its OWN previous answer (including the self-critique fields)
- It sees the search results that address the critique's gaps
- Each iteration adds more context, so each revision is better informed
- Without `MessagesPlaceholder`, the prompt would just be system instructions with no history — the LLM would start from scratch every time

**`MessagesPlaceholder` vs hardcoded `{input}`:**

| Approach | What Gets Injected | Supports Multi-Turn? |
|----------|-------------------|---------------------|
| `("human", "{input}")` | A single string variable | ❌ One message only |
| `MessagesPlaceholder("messages")` | An entire list of `BaseMessage` objects | ✅ Any number of `HumanMessage`, `AIMessage`, `ToolMessage` |

In agentic graphs, you almost always need `MessagesPlaceholder` because the state grows with each node execution — and each node needs to see the full conversation to make informed decisions.

### First Responder Instruction
```
"Provide a detailed ~250 word answer."
```

### Revisor Instruction
```
"Revise your previous answer using the new information.
- You should use the previous critique to add important information...
- You MUST include numerical citations...
- Add a References section...
- Make SURE it is not more than 250 words."
```

### How the Two Chains Are Built — The `.partial()` → `|` → `.bind_tools()` Pipeline

This is the most confusing part of the code if you come back later. Here's exactly what each piece does:

**Chain 1 — `first_responder` (draft):**

```python
# Step A: Take the shared template and fill in {first_instruction}
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# Step B: Pipe the prompt into the LLM, forcing it to return AnswerQuestion schema
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
```

**Chain 2 — `revisor` (revise):**

```python
# Same thing, but with different instruction and different schema
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions        # ← different instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")  # ← different form
```

**Breaking it down piece by piece:**

```
actor_prompt_template          ← The shared template with {first_instruction} and {time} slots
        │
        ▼
   .partial(first_instruction="...")   ← Fill in {first_instruction} ahead of time.
        │                                 Now the template only needs {messages} at runtime.
        │                                 Think of it like setting a DEFAULT value.
        ▼
        |                              ← The PIPE operator (LCEL). Means "pass the output
        │                                 of the left side as input to the right side."
        │                                 prompt output → LLM input. That's it.
        ▼
   llm.bind_tools(                     ← Tell the LLM "here are the tools you can call"
       tools=[AnswerQuestion],            These are Pydantic models, not actual functions.
       tool_choice="AnswerQuestion"    ← "You MUST call this tool. No freeform text."
   )
```

**What `.partial()` actually does — a simple analogy:**

Think of the template as a form letter with blanks:

```
Dear {name},
Your order for {product} ships on {date}.
```

`.partial(name="John")` fills in ONE blank ahead of time:

```
Dear John,
Your order for {product} ships on {date}.
```

Now when you send the letter, you only need to fill in `{product}` and `{date}`.

In our case, `actor_prompt_template` has three blanks: `{time}`, `{first_instruction}`, and `{messages}`:
- `{time}` is filled with `.partial(time=lambda: ...)` when the template is created
- `{first_instruction}` is filled with `.partial(first_instruction="...")` for each chain
- `{messages}` is filled at **runtime** when the chain is invoked with graph state

**Side-by-side — both chains are 90% identical:**

| Part | `first_responder` | `revisor` |
|------|-------------------|-----------|
| Base template | `actor_prompt_template` | `actor_prompt_template` (same) |
| `{time}` | Auto-filled (same) | Auto-filled (same) |
| `{first_instruction}` | "Provide a detailed ~250 word answer." | "Revise your previous answer..." (with citation rules) |
| `{messages}` | Filled at runtime (same) | Filled at runtime (same) |
| `bind_tools` schema | `AnswerQuestion` (answer + critique + queries) | `ReviseAnswer` (same + references) |
| `tool_choice` | `"AnswerQuestion"` | `"ReviseAnswer"` |

**The only differences:** the instruction text and the output schema. Everything else is shared. That's why we use `.partial()` — write the template once, customize it twice.

**What the `|` (pipe) operator does at runtime:**

```
chain.invoke({"messages": [HumanMessage("Write about AI SOC...")]})

Step 1: Prompt template fills in {messages} → produces a list of ChatMessages
Step 2: | pipes those ChatMessages into the LLM
Step 3: LLM is forced to return AnswerQuestion/ReviseAnswer structured output
Step 4: Result = AIMessage with tool_calls containing the filled-in schema
```

### Why "Be severe to maximize improvement"?

Without this directive, the LLM tends to say "great answer, no issues!" — which defeats the purpose of self-critique. The prompt explicitly instructs severity to force meaningful feedback.

---

## 7. The ToolNode Name-Matching Trick

The most subtle implementation detail:

```python
# In tool_executor.py
execute_tools = ToolNode([
	StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
	StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
])
```

**Why name tools after schemas?**

When you use `tool_choice="AnswerQuestion"`, the LLM's response has:
```json
{"tool_calls": [{"name": "AnswerQuestion", "args": {"answer": "...", "search_queries": [...]}}]}
```

`ToolNode` looks up tools BY NAME. So it needs a tool named `"AnswerQuestion"` to handle that call. Both tools run the exact same `run_queries` function — the name is just for routing.

---

## 8. Comparison: Reflection vs Reflexion

| Aspect | Section 14 (Reflection) | Section 15 (Reflexion) |
|--------|------------------------|------------------------|
| **Tools** | None | Tavily search |
| **Output format** | Freeform text (AIMessage) | Structured (Pydantic schema via tool_choice) |
| **Critique mechanism** | Separate reflect node | Built into the same response (Reflection field) |
| **Citations** | None | Required (References list) |
| **Stop condition** | Message count > 6 | ToolMessage count > MAX_ITERATIONS |
| **Nodes** | 2 (generate, reflect) | 3 (draft, execute_tools, revise) |
| **Grounding** | None (pure LLM) | Real-time web data |
| **Use case** | Tweet improvement | Research articles, factual content |
| **Quality driver** | LLM-to-LLM feedback | External data + self-critique |
| **Code complexity** | ~60 lines | ~100 lines (across 4 files) |

---

## 9. Code Walkthrough

### File Structure

```
15-reflexion-agent/
├── 19_Reflexion_Agent.md          # This theory file
└── src/
	├── schemas.py                 # Pydantic models (AnswerQuestion, ReviseAnswer)
	├── chains.py                  # LLM chains (first_responder, revisor)
	├── tool_executor.py           # Tavily search via ToolNode
	└── main.py                    # Graph definition + execution
```

### Execution Flow

```
Step 1: User sends question
		→ state: [HumanMessage("Write about AI-Powered SOC...")]

Step 2: draft_node runs first_responder chain
		→ state: [..., AIMessage(tool_calls=[{name: "AnswerQuestion",
				   args: {answer: "...", reflection: {...}, search_queries: [...]}}])]

Step 3: execute_tools (ToolNode) extracts search_queries, calls Tavily
		→ state: [..., ToolMessage(content="[search results...]")]

Step 4: revise_node runs revisor chain (sees answer + critique + search results)
		→ state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer",
				   args: {answer: "...", reflection: {...}, search_queries: [...], references: [...]}}])]

Step 5: event_loop checks ToolMessage count (1 ≤ 2) → route to execute_tools

Step 6: execute_tools runs NEW search queries from revise step
		→ state: [..., ToolMessage(content="[new search results...]")]

Step 7: revise_node runs again (sees all prior context + new data)
		→ state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer", args: {...}}])]

Step 8: event_loop checks ToolMessage count (2 ≤ 2) → route to execute_tools

Step 9: execute_tools runs yet another set of queries
		→ state: [..., ToolMessage(content="[more results...]")]

Step 10: revise_node runs final time
		 → state: [..., AIMessage(tool_calls=[{name: "ReviseAnswer", args: {...}}])]

Step 11: event_loop checks ToolMessage count (3 > 2) → route to END ✅
```

---

## 10. LangGraph Workflow Patterns

> These official LangGraph patterns show where the Reflexion agent fits in the broader landscape. They are from the [official LangGraph docs](https://langchain-ai.github.io/langgraph/).

### Pattern Taxonomy

| Pattern | Flow Shape | Cycles? | LLM Decides Flow? | Our Examples |
|---------|-----------|---------|-------------------|------|
| **Prompt Chaining** | A → B → C (linear) | ❌ | ❌ | RAG pipeline (Section 9) |
| **Parallelization** | START → [A, B, C] → Aggregator → END | ❌ | ❌ | — |
| **Routing** | Router → one-of-many paths → END | ❌ | ✅ (picks branch) | LangChain routers |
| **Orchestrator-Worker** | Orchestrator → spawn N workers → Synthesizer | ❌ | ✅ (plans subtasks) | — |
| **Evaluator-Optimizer** | Generate ←→ Evaluate (cycle) | ✅ | ✅ (decides quality) | Reflection Agent (Section 14) |
| **Agent (ReAct)** | Think → Act → Observe (cycle) | ✅ | ✅ (picks tools) | ReAct Agent (Section 13) |
| **Reflexion** | Draft → Search → Revise (cycle) | ✅ | ✅ (generates queries) | This section |

### Prompt Chaining

Each LLM call processes the output of the previous call. No branching, no cycles.

```
START → generate_joke → [check_punchline] → improve_joke → polish_joke → END
                              │
                              └─── (pass) ──→ END
```

**When to use:** Well-defined tasks with verifiable intermediate steps (translation, content verification).

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: State):
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: State):
    """Gate function — pure Python, no LLM needed"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
# Conditional: if punchline passes → END, otherwise → improve
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()
state = chain.invoke({"topic": "cats"})
```

### Parallelization

Multiple LLM calls run simultaneously, results aggregated.

```
         ┌─── call_llm_1 (joke) ───┐
START ───┼─── call_llm_2 (story) ──┼──→ aggregator → END
         └─── call_llm_3 (poem) ───┘
```

**When to use:** Independent subtasks that can run concurrently (speed), or running the same task multiple times (confidence).

**LangGraph feature:** Multiple edges from START to different nodes automatically run in parallel.

```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_1(state: State):
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    combined = f"JOKE:\n{state['joke']}\n\nSTORY:\n{state['story']}\n\nPOEM:\n{state['poem']}"
    return {"combined_output": combined}

builder = StateGraph(State)
builder.add_node("call_llm_1", call_llm_1)
builder.add_node("call_llm_2", call_llm_2)
builder.add_node("call_llm_3", call_llm_3)
builder.add_node("aggregator", aggregator)

# THREE edges from START → all three run in PARALLEL
builder.add_edge(START, "call_llm_1")
builder.add_edge(START, "call_llm_2")
builder.add_edge(START, "call_llm_3")
# All three converge into aggregator
builder.add_edge("call_llm_1", "aggregator")
builder.add_edge("call_llm_2", "aggregator")
builder.add_edge("call_llm_3", "aggregator")
builder.add_edge("aggregator", END)

parallel_workflow = builder.compile()
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])
```

### Routing

LLM classifies input and routes to specialized handlers. No cycles.

```
START → router_llm → {story: llm_1, joke: llm_2, poem: llm_3} → END
```

**Key technique:** Use `with_structured_output(Route)` to get a Pydantic model from the LLM, then route based on the field value.

```python
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        description="The next step in the routing process"
    )

# LLM with forced structured output — always returns a Route object
router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_router(state: State):
    """LLM classifies the input and decides the route"""
    decision = router.invoke([
        SystemMessage(content="Route the input to story, joke, or poem."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

def write_story(state: State):
    result = llm.invoke(f"Write a story: {state['input']}")
    return {"output": result.content}

def write_joke(state: State):
    result = llm.invoke(f"Write a joke: {state['input']}")
    return {"output": result.content}

def write_poem(state: State):
    result = llm.invoke(f"Write a poem: {state['input']}")
    return {"output": result.content}

def route_decision(state: State):
    """Conditional edge function — returns the name of the next node"""
    if state["decision"] == "story": return "write_story"
    elif state["decision"] == "joke": return "write_joke"
    elif state["decision"] == "poem": return "write_poem"

builder = StateGraph(State)
builder.add_node("llm_call_router", llm_call_router)
builder.add_node("write_story", write_story)
builder.add_node("write_joke", write_joke)
builder.add_node("write_poem", write_poem)

builder.add_edge(START, "llm_call_router")
# path_map: {return value of route_decision: node name to visit}
builder.add_conditional_edges("llm_call_router", route_decision, {
    "write_story": "write_story",
    "write_joke": "write_joke",
    "write_poem": "write_poem",
})
builder.add_edge("write_story", END)
builder.add_edge("write_joke", END)
builder.add_edge("write_poem", END)

router_workflow = builder.compile()
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])
```

### Orchestrator-Worker

One LLM plans subtasks, spawns workers dynamically, synthesizes results.

```
START → orchestrator (plans sections) → [Send("worker", section) for each] → synthesizer → END
```

**Key LangGraph feature:** The `Send` API lets you dynamically create worker nodes at runtime — you don't need to know how many workers you'll need at graph construction time.

```python
from typing import Annotated, List
import operator
from langgraph.types import Send

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the section.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

# Main graph state — shared across all workers
class State(TypedDict):
    topic: str
    sections: list[Section]
    # operator.add reducer — all workers APPEND to this list in parallel
    completed_sections: Annotated[list, operator.add]
    final_report: str

# Worker gets its OWN state with just one section
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    """LLM plans what sections the report needs"""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Report topic: {state['topic']}"),
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Each worker writes ONE section"""
    section = llm.invoke([
        SystemMessage(content="Write a report section. Use markdown."),
        HumanMessage(content=f"Section: {state['section'].name} — {state['section'].description}"),
    ])
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    return {"final_report": "\n\n---\n\n".join(state["completed_sections"])}

def assign_workers(state: State):
    """Dynamically spawn one worker per section via Send API"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_call", llm_call)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
# Conditional edge that SPAWNS workers dynamically
builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", END)

workflow = builder.compile()
state = workflow.invoke({"topic": "LLM scaling laws"})
print(state["final_report"])
```

### Evaluator-Optimizer

Generate → evaluate → (if not good enough) → generate again. Exactly what our Reflection Agent does.

```
START → generator → evaluator → {Accepted: END, Rejected: generator}
```

**When to use:** Tasks with clear success criteria that need iteration (translation quality, code correctness, joke humor).

```python
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Is the joke funny?")
    feedback: str = Field(description="How to improve if not funny.")

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state: State):
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} considering feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    grade = evaluator.invoke(f"Grade the joke: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    return "Rejected + Feedback"

builder = StateGraph(State)
builder.add_node("llm_call_generator", llm_call_generator)
builder.add_node("llm_call_evaluator", llm_call_evaluator)

builder.add_edge(START, "llm_call_generator")
builder.add_edge("llm_call_generator", "llm_call_evaluator")
# Cycle back to generator if rejected, otherwise end
builder.add_conditional_edges("llm_call_evaluator", route_joke, {
    "Accepted": END,
    "Rejected + Feedback": "llm_call_generator",
})

optimizer = builder.compile()
state = optimizer.invoke({"topic": "Cats"})
print(state["joke"])
```

### Where Reflexion Fits

The Reflexion agent is a **combination** of:
- **Evaluator-Optimizer** (self-critique drives revision)
- **Agent** (uses tools to fetch data)
- **Prompt Chaining** (structured progression from draft to final)

---

## 11. ToolNode Deep Dive

> `ToolNode` is a prebuilt LangGraph node that handles tool execution. It's used in this project and the concepts apply to all LangGraph agents.

### What ToolNode Does

| Input | Process | Output |
|-------|---------|--------|
| `AIMessage` with `tool_calls` | Looks up tool by name → invokes with args | `ToolMessage(content=result)` |

### Key Features

| Feature | Description |
|---------|-------------|
| **Name-based routing** | Matches `tool_call.name` to registered tool names |
| **Parallel execution** | If AIMessage has multiple tool_calls, executes all in parallel |
| **Error handling** | Configurable: catch errors and return them as ToolMessage, or propagate |
| **State injection** | Via `ToolRuntime` — tools can access graph state and run-scoped context |

### End-to-End Example: ToolNode in a Simple Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# 2. Bind tools to the LLM so it knows about them
tools = [multiply, add]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# 3. Create ToolNode — it will execute whichever tool the LLM calls
tool_node = ToolNode(tools)

# 4. Build the graph
def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")
# tools_condition: if LLM returned tool_calls → "tools", otherwise → END
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

graph = builder.compile()

# 5. Run it
result = graph.invoke({"messages": [HumanMessage("What is 3 * 12?")]})
print(result["messages"][-1].content)  # "3 * 12 = 36"
```

**What happens step by step:**

| Step | Node | What Happens |
|------|------|-------------|
| 1 | `call_model` | LLM sees `multiply` and `add` tools, decides to call `multiply(3, 12)` |
| 2 | `tools_condition` | Detects `tool_calls` in AIMessage → routes to `"tools"` |
| 3 | `tools` (ToolNode) | Finds tool named `"multiply"` → invokes `multiply(3, 12)` → returns `ToolMessage(content="36")` |
| 4 | `call_model` | LLM sees the ToolMessage result → responds with "3 * 12 = 36" |
| 5 | `tools_condition` | No tool_calls → routes to END |

### Input Formats

```python
# 1. Graph state (most common in LangGraph)
{"messages": [AIMessage(tool_calls=[{"name": "search", "args": {...}, "id": "abc"}])]}

# 2. Message list
[AIMessage(tool_calls=[...])]

# 3. Direct tool calls (for testing)
[{"name": "search", "args": {"query": "test"}, "id": "1", "type": "tool_call"}]
```

### Parallel Tool Execution Example

If the LLM returns multiple `tool_calls` in a single AIMessage, ToolNode executes ALL of them in parallel:

```python
from langchain_core.messages import AIMessage

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 25°C in {city}"

@tool
def get_population(city: str) -> int:
    """Get population of a city."""
    return {"London": 9_000_000, "Tokyo": 14_000_000}.get(city, 0)

tool_node = ToolNode([get_weather, get_population])

# Simulate an AIMessage that calls BOTH tools at once
ai_msg = AIMessage(content="", tool_calls=[
    {"name": "get_weather", "args": {"city": "London"}, "id": "1", "type": "tool_call"},
    {"name": "get_population", "args": {"city": "London"}, "id": "2", "type": "tool_call"},
])

# ToolNode executes both in parallel, returns TWO ToolMessages
result = tool_node.invoke({"messages": [ai_msg]})
for msg in result["messages"]:
    print(f"Tool: {msg.name} → {msg.content}")
# Tool: get_weather → Sunny, 25°C in London
# Tool: get_population → 9000000
```

### Error Handling Options

```python
# Default: catches invocation errors, propagates execution errors
ToolNode(tools)

# Catch ALL errors, return as ToolMessage (agent can recover)
ToolNode(tools, handle_tool_errors=True)

# Custom error message
ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# Custom handler function
ToolNode(tools, handle_tool_errors=lambda e: f"Error: {str(e)}")

# Disable error handling (exceptions propagate — crashes the graph)
ToolNode(tools, handle_tool_errors=False)
```

**Example with error handling:**

```python
@tool
def risky_tool(query: str) -> str:
    """A tool that might fail."""
    raise ValueError(f"Cannot process: {query}")

# With handle_tool_errors=True, the error becomes a ToolMessage
# instead of crashing the graph — the LLM can retry or adjust
tool_node = ToolNode([risky_tool], handle_tool_errors=True)

ai_msg = AIMessage(content="", tool_calls=[
    {"name": "risky_tool", "args": {"query": "test"}, "id": "1", "type": "tool_call"}
])
result = tool_node.invoke({"messages": [ai_msg]})
print(result["messages"][0].content)
# "Error: Cannot process: test"
# The LLM sees this as a ToolMessage and can decide what to do next
```

### ToolRuntime — Accessing State From Tools

Tools normally only receive the args the LLM generates. To access **graph state** or **run-scoped context**, use `ToolRuntime`:

```python
from langchain_core.tools import tool, ToolRuntime

class State(MessagesState):
    user_id: str
    permissions: list[str]

@tool
def get_user_info(runtime: ToolRuntime[None, State]) -> str:
    """Look up user information."""
    user_id = runtime.state["user_id"]
    permissions = runtime.state["permissions"]
    return f"User {user_id} has permissions: {', '.join(permissions)}"
```

**Important:** Tools can only access state values passed to the ToolNode. When ToolNode is a direct graph node, it receives the full state. If you invoke it manually from another node, pass the full state explicitly.

**Full ToolRuntime example in a graph:**

```python
@tool
def get_user_orders(runtime: ToolRuntime[None, State]) -> str:
    """Get orders for the current user from graph state."""
    user_id = runtime.state["user_id"]
    # In production: db_query(user_id)
    return f"User {user_id} has 3 pending orders"

class State(MessagesState):
    user_id: str

tools = [get_user_orders]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
tool_node = ToolNode(tools)

def call_model(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)  # ToolNode gets full State, including user_id

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

graph = builder.compile()

# user_id is in graph state — the LLM never sees it, but the tool can access it
result = graph.invoke({
    "messages": [HumanMessage("What are my orders?")],
    "user_id": "user-42",
})
print(result["messages"][-1].content)
```

### How It's Used in This Project

```python
# In tool_executor.py — tools named to match LLM's tool_choice
execute_tools = ToolNode([
    StructuredTool.from_function(run_queries, name="AnswerQuestion"),
    StructuredTool.from_function(run_queries, name="ReviseAnswer"),
])
```

The LLM calls `tool_choice="AnswerQuestion"` → ToolNode finds tool named `"AnswerQuestion"` → executes `run_queries` → returns ToolMessage with search results.

---

## 12. Production-Grade Checkpointers

> ⚠️ **Beyond the basic example.** The basic version uses `MemorySaver` (in-memory), which loses all state when the process dies. In production you need durable persistence — this section fills that gap.

### Why You Need a Production Checkpointer

`MemorySaver` is fine for demos but **catastrophic in production:**

| Scenario | MemorySaver | Production Checkpointer |
|----------|-------------|------------------------|
| Server restarts/deploys | ❌ All state lost | ✅ Resumes from last checkpoint |
| Multiple server instances | ❌ Each has its own memory | ✅ Shared DB, any instance can resume |
| Long-running agents (Reflexion) | ❌ Crash = start over | ✅ Picks up from last completed node |
| Human-in-the-loop (HIL) | ❌ User must stay connected | ✅ User can come back hours later |
| Debugging/auditing | ❌ No history | ✅ Full time-travel through every state |

### Available Checkpointers

| Checkpointer | Package | Best For |
|--------------|---------|----------|
| `MemorySaver` | `langgraph` (built-in) | Local dev, testing, demos |
| `PostgresSaver` (sync) | `langgraph-checkpoint-postgres` | Production APIs (FastAPI, Flask) |
| `AsyncPostgresSaver` | `langgraph-checkpoint-postgres` | Production async APIs (ASGI, Starlette) |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | Single-server production, edge deployments |
| `AsyncSqliteSaver` | `langgraph-checkpoint-sqlite` | Async single-server |

### Production Example: PostgresSaver with a Reflexion Agent

```python
"""
Production Reflexion agent with PostgreSQL checkpointing.

This is what a real deployment looks like:
- State survives server restarts
- Multiple workers can process the same conversation
- You can time-travel to debug any step
- Human-in-the-loop can pause for days

Prerequisites:
  pip install langgraph-checkpoint-postgres psycopg[binary]
  # or: uv add langgraph-checkpoint-postgres "psycopg[binary]"

  # PostgreSQL running with a database ready:
  # CREATE DATABASE langgraph_checkpoints;

Run:
  uv run python main_production.py
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# ─── 1. Database connection ─────────────────────────────────────────────────
# In production: use env var, never hardcode credentials
DB_URI = os.environ.get(
    "CHECKPOINT_POSTGRES_URI",
    "postgresql://user:password@localhost:5432/langgraph_checkpoints"
)

# ─── 2. Build the graph (same as before) ────────────────────────────────────
from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2

def draft_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState):
    count = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if count > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

# ─── 3. Compile WITH checkpointer (the only difference from demo code) ──────
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Create tables on first run (idempotent)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # ─── 4. Invoke with thread_id (required for checkpointing) ───────────────
    config = {"configurable": {"thread_id": "research-article-001"}}

    result = graph.invoke(
        {"messages": [HumanMessage("Write about AI-Powered SOC startups that raised capital")]},
        config=config,
    )

    print(result["messages"][-1].tool_calls[0]["args"]["answer"])

    # ─── 5. Resume later (even after server restart) ─────────────────────────
    # Same thread_id = picks up where it left off
    # If the graph already completed, this returns the final state immediately
    state = graph.get_state(config)
    print(f"Thread status: {state.next}")  # () = completed, ('execute_tools',) = in progress
```

### Async Version (for FastAPI / Production APIs)

```python
"""Async checkpointing — for use inside FastAPI or any async framework."""

import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def run_agent():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "user-42-session-7"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage("Write about AI-Powered SOC startups")]},
            config=config,
        )
        return result

# In FastAPI:
# @app.post("/research")
# async def research(question: str, thread_id: str):
#     config = {"configurable": {"thread_id": thread_id}}
#     result = await graph.ainvoke({"messages": [HumanMessage(question)]}, config)
#     return {"answer": result["messages"][-1].tool_calls[0]["args"]["answer"]}
```

### What the Checkpointer Actually Stores

After every node execution, the checkpointer saves:

```
┌─────────────────────────────────────────────────────────────────┐
│ thread_id: "research-article-001"                               │
│ checkpoint_id: "1ef7a8b2-..."  (auto-generated UUID)            │
│ parent_checkpoint_id: "1ef7a8a1-..."  (previous checkpoint)     │
│                                                                 │
│ channel_values:                                                  │
│   messages: [                                                   │
│     HumanMessage("Write about AI-Powered SOC..."),              │
│     AIMessage(tool_calls=[{name: "AnswerQuestion", ...}]),      │
│     ToolMessage("Torq raised $70M..."),                         │
│   ]                                                             │
│                                                                 │
│ metadata:                                                        │
│   source: "loop"                                                │
│   step: 3                                                       │
│   writes: {"revise": {"messages": [...]}}                       │
│                                                                 │
│ pending_writes: []  (empty = node completed successfully)       │
└─────────────────────────────────────────────────────────────────┘
```

### Time Travel — Debugging Production Issues

```python
# Get ALL checkpoints for a thread (full history of every node execution)
checkpoints = list(checkpointer.list(config))

for cp in checkpoints:
    print(f"Step {cp.metadata['step']}: wrote to {list(cp.metadata['writes'].keys())}")
    # Step 0: wrote to ['__start__']
    # Step 1: wrote to ['draft']
    # Step 2: wrote to ['execute_tools']
    # Step 3: wrote to ['revise']
    # Step 4: wrote to ['execute_tools']
    # Step 5: wrote to ['revise']

# Replay from a specific checkpoint (time travel)
old_config = {"configurable": {"thread_id": "research-article-001", "checkpoint_id": checkpoints[2].id}}
state_at_step_2 = graph.get_state(old_config)
print(state_at_step_2.values["messages"])  # See exactly what the state was after execute_tools
```

### Human-in-the-Loop with Checkpointer

```python
# Compile with an interrupt — graph PAUSES before revise node
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["revise"],  # Pause here, let human review
)

config = {"configurable": {"thread_id": "hil-session-1"}}

# First invoke — runs draft + execute_tools, then PAUSES
result = graph.invoke(
    {"messages": [HumanMessage("Write about AI-Powered SOC startups")]},
    config=config,
)
# Graph is now paused. User can review the search results.
# Hours/days can pass. Server can restart. State is in PostgreSQL.

# Later: human approves, resume from where it paused
result = graph.invoke(None, config=config)  # None = "continue from checkpoint"
```

### Key Takeaways for Interviews

| Point | Detail |
|-------|--------|
| **MemorySaver = dev only** | Never use in production. State lost on restart. |
| **PostgresSaver = production default** | Shared state, survives restarts, enables HIL |
| **thread_id is mandatory** | Without it, the checkpointer can't find the right conversation |
| **`checkpointer.setup()`** | Creates the required tables. Call once (idempotent). |
| **Time travel** | `checkpointer.list(config)` returns every checkpoint — full audit trail |
| **Interrupt + resume** | `interrupt_before`/`interrupt_after` + `invoke(None, config)` for HIL |
| **The graph code doesn't change** | Only the `compile(checkpointer=...)` line differs between dev and prod |

---

## Interview Q&A Anchors

**Q: What is a Reflexion agent and how does it differ from basic reflection?**
> **A:** A Reflexion agent combines self-critique with tool execution. Unlike basic reflection (LLM-to-LLM feedback only), reflexion fetches external data via tools to ground improvements in real facts. The LLM produces structured output containing answer, critique, AND search queries in one call — forcing self-awareness into the output format itself.

**Q: Why use `tool_choice` to force structured output?**
> **A:** `tool_choice="AnswerQuestion"` eliminates the LLM's option to respond with freeform text. It MUST return data matching the Pydantic schema, including the self-critique and search queries. This makes the reflection loop deterministic — every response always contains actionable feedback and queries for the next iteration.

**Q: How does the ToolNode know which function to call?**
> **A:** ToolNode matches tool calls by name. Since the LLM uses `tool_choice="AnswerQuestion"`, its response has `tool_calls[0].name == "AnswerQuestion"`. We register a `StructuredTool` with that exact name. Both `AnswerQuestion` and `ReviseAnswer` tools execute the same search function — the name is just a routing mechanism.

**Q: What's the stop condition in a Reflexion agent?**
> **A:** It counts `ToolMessage` instances in the state. Each tool execution adds one ToolMessage, so counting them equals counting iterations. When the count exceeds `MAX_ITERATIONS`, the conditional edge routes to END instead of back to execute_tools.

**Q: When would you use Reflexion over basic Reflection?**
> **A:** Use Reflexion when factual accuracy matters — research articles, technical documentation, reports that need citations. Basic reflection is sufficient for style/quality improvements (tweets, emails) where external data isn't needed. Reflexion costs more (tool calls + more LLM tokens) but produces grounded, verifiable output.

**Q: How does Reflexion differ from RAG?**
> **A:** RAG retrieves from your ingested knowledge base first, then answers — like studying your notes before an exam. Reflexion answers first with parametric knowledge, critiques itself, then does targeted retrieval for only the gaps it identified — like taking the exam, checking your answers, studying just what you got wrong, and retaking. RAG does one broad retrieval pass based on the user's question; Reflexion retrieval is narrow and driven by the LLM's own critique. You can combine both — RAG for the first draft, Reflexion to iterate and improve.

**Q: What are the main LangGraph workflow patterns?**
> **A:** Five core patterns: (1) Prompt chaining — linear sequence of LLM calls. (2) Parallelization — multiple independent calls at once. (3) Routing — LLM classifies input and routes to specialized handlers. (4) Orchestrator-worker — one LLM plans subtasks, spawns workers dynamically via Send API. (5) Evaluator-optimizer — generate/evaluate cycle until quality threshold met. Agents add tool use and cycles on top of these.

**Q: What is ToolNode and when would you use it vs writing your own tool execution?**
> **A:** ToolNode is a prebuilt LangGraph node that handles tool execution: name-based routing, parallel execution, error handling, and state injection via ToolRuntime. Use it when you want standard tool execution behavior. Write your own when you need custom routing logic, specialized error recovery, or non-standard argument transformation.

**Q: How does the Send API enable dynamic parallelism?**
> **A:** The Send API lets you create worker nodes at runtime based on state content. Instead of defining a fixed number of parallel branches at graph construction time, you return `[Send("worker", input) for input in dynamic_list]` from a conditional edge. Each Send spawns an independent execution of the target node with its own input, and results are collected via a shared state key with a reducer.

**Q: What is `MessagesPlaceholder` and why is it critical in agentic prompts?**
> **A:** `MessagesPlaceholder` is a slot in a `ChatPromptTemplate` that gets replaced at runtime with the entire `messages` list from graph state. Unlike `("human", "{input}")` which injects a single string, `MessagesPlaceholder` injects a full list of `HumanMessage`, `AIMessage`, and `ToolMessage` objects. In agentic graphs this is essential — each node needs to see the complete conversation history (prior answers, critiques, search results) to make informed decisions. Without it, the LLM would start from scratch on every iteration.

**Q: Why can't you use MemorySaver in production, and what do you use instead?**
> **A:** `MemorySaver` stores state in-process memory — it's lost on restart, can't be shared across server instances, and provides no audit trail. In production you use `PostgresSaver` (or `AsyncPostgresSaver` for async frameworks). It persists every checkpoint to a shared database, enabling server restarts without state loss, horizontal scaling, time-travel debugging via `checkpointer.list(config)`, and human-in-the-loop pauses that can last hours or days. The graph code stays identical — only the `compile(checkpointer=...)` line changes.

---

## Runnable Scripts

→ [`src/main.py`](./src/main.py) — Full agent execution
→ [`src/chains.py`](./src/chains.py) — Standalone chain test

---

## References

- [Reflexion Paper (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366) — The original paper from Northeastern, MIT, and Princeton
- [LangChain Blog — Reflection Agents](https://www.langchain.com/blog/reflection-agents) — LangChain team's implementation guide
- [LangGraph Reflexion Tutorial](https://www.langchain.com/blog/reflection-agents#reflexion) — Official LangGraph tutorial
- [Tavily Search](https://tavily.com/) — Search engine optimized for LLM applications
- [LangGraph Workflows and Agents Guide](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Official patterns documentation (prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer)
- [ToolNode API Reference](https://docs.langchain.com/oss/python/langgraph/workflows-agents#toolnode) — Prebuilt node for tool execution
