# 02. Chat Models: The Message Protocol & Architecture 🧠

*Topic: LangChain Basics*

> 🚀 **Runnable script:** [src/02_chat_models_architecture.py](src/02_chat_models_architecture.py)
> 🖼️ **Multimodal demo:** [src/test_multimodal_messages.py](src/test_multimodal_messages.py) — send a PDF page to GPT-4o Vision

## 🎯 What You Will Learn
* The architectural shift from Completion Models to Chat Models.
* The "Chat Markup Language" (System, Human, AI Roles).
* **Message anatomy** — content types (text, image, audio), metadata (usage, response), and the full type hierarchy.
* How to handle the inherently stateless nature of LLM APIs.
* Raw message construction vs. PromptTemplates.

## 📦 Dependency Setup
Run this exact command in your terminal using `uv`:
```bash
uv add langchain-core langchain-openai python-dotenv
```

---

## 1. Core Architectural Concepts

### Completion vs. Chat
This module defines the architectural shift from **Completion Models** (Legacy) to **Chat Models** (Modern).

* **Completion Models (LLMs):** Text-In $\to$ Text-Out. You give it a string ("The sky is"), and it completes it (" blue"). It has no concept of "talking."
* **Chat Models:** Messages-In $\to$ Message-Out. You give it a **List of Messages** with specific **Roles** (System, Human, AI). It outputs an `AIMessage`.

### C#/Java Analogy
* **Completion Model:** A pure static string manipulation function: `string response = StringUtils.CompleteText("The sky is ");`
* **Chat Model:** A REST API Endpoint that takes a JSON Array of Message DTOs. Because HTTP is fundamentally stateless, you must pass the entire session history array on every single request to simulate "memory".

---

## 2. Technical Deep Dive (Components)

### A. The Three Roles (The "Chat Markup Language")

Modern models (GPT-4, Claude, Gemini) are trained on structured dialogue, not just raw text. They expect inputs tagged with specific tokens indicating *who* is speaking.

| Role | Class | Purpose | Production Note |
| --- | --- | --- | --- |
| **System** | `SystemMessage` | Meta-instructions. Defines **who** the AI is and **how** it behaves. | **Security Critical.** This is where you place guardrails (e.g., "Do not reveal user PII"). |
| **Human** | `HumanMessage` | The user's input. | In production, strictly separate this from System instructions to prevent "Prompt Injection." |
| **AI** | `AIMessage` | The model's output. | Used to feed conversation history back into the model so it remembers the context. |

### B. Message Anatomy — The Three Layers

Every LangChain message (inheriting from `BaseMessage`) is more than just text. Each message is a structured object with three layers:

```
┌─────────────────────────────────────────────────────────┐
│                    BaseMessage                            │
├─────────────────────────────────────────────────────────┤
│  1. ROLE         │  type field (system/human/ai/tool)    │
│  2. CONTENT      │  str OR List[content blocks]          │
│  3. METADATA     │  response_metadata, usage_metadata,   │
│                  │  id, tool_calls, etc.                  │
└─────────────────────────────────────────────────────────┘
```

#### Layer 1: Role (Type)

Already covered above — `system`, `human`, `ai`, `tool`.

#### Layer 2: Content — Text vs. Multimodal Blocks

The `content` field can be **either** a plain string **or** a list of typed content blocks for multimodal input:

| Content Type | Block Format | Used For |
|-------------|-------------|----------|
| **Text** | `{"type": "text", "text": "..."}` | Standard text input/output |
| **Image URL** | `{"type": "image_url", "image_url": {"url": "..."}}` | Sending images to vision models (GPT-4o, Claude) |
| **Image (base64)** | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` | Inline image data without external URL |
| **Audio** | `{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}` | Audio input for speech models |
| **Tool Use** | Returned in `AIMessage.tool_calls` | When the model requests tool execution |

```python
from langchain_core.messages import HumanMessage

# Simple text content (string shorthand)
msg_text = HumanMessage(content="What is this image?")

# Multimodal content (list of blocks)
msg_multimodal = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}}
])
```

> **C# Analogy:** Think of `content` as a discriminated union / `OneOf<string, List<ContentBlock>>`. In C# terms, it's like having a property that can be `string` for simple cases or `List<ChatMessageContent>` for multimodal, similar to the Semantic Kernel's `ChatMessageContentItemCollection`.

#### ⚡ Direct Vision vs. RAG — When to Use Which

Sending images directly via multimodal content blocks is powerful but expensive. Here's when to choose each approach:

| | Direct Vision (this pattern) | RAG Pipeline (Section 9) |
|---|---|---|
| **How it works** | Render page → send image to GPT-4o on every query | Render once → GPT-4o describes → embed → store in vector DB |
| **Cost per query** | **High** — image tokens billed every call (~1000+ tokens per page at `detail: "high"`) | **Low** — text retrieval + text generation only |
| **Upfront cost** | Zero | One-time GPT-4o description + embedding pass |
| **Best for** | Ad-hoc inspection, one-off "what's on this page?", verifying layout/colors | Repeated queries against same documents, production systems |
| **Scales to** | 1-2 pages at a time | Hundreds/thousands of pages |

> **Rule of thumb:** If you'll ask more than 2-3 questions about the same document, pay the one-time cost to describe and embed it into a vector store. Reserve direct vision for development-time exploration or when you need the model to see exact pixel-level detail.
>
> See [09-gist-of-rag/src/test_multimodal_pdf_rag.py](../09-gist-of-rag/src/test_multimodal_pdf_rag.py) for the production RAG approach.

#### Layer 3: Metadata

`AIMessage` objects returned by the model carry rich metadata:

| Property | What It Contains | Why It Matters |
|----------|-----------------|---------------|
| `response_metadata` | Model name, finish reason, system fingerprint | Debugging which model/version produced a response |
| `usage_metadata` | `input_tokens`, `output_tokens`, `total_tokens` | **Cost tracking** — calculate spend per request |
| `id` | Unique message identifier (e.g., `run-abc123`) | Correlation in logging and tracing (LangSmith) |
| `tool_calls` | List of tool call requests `[{name, args, id}]` | Agents — the model is requesting tool execution |
| `additional_kwargs` | Vendor-specific fields (refusal, logprobs, etc.) | Advanced use cases |

```python
response = llm.invoke([HumanMessage(content="Hello")])

print(response.content)              # "Hi there!"
print(response.usage_metadata)       # {'input_tokens': 8, 'output_tokens': 12, 'total_tokens': 20}
print(response.response_metadata)    # {'model_name': 'gpt-4o', 'finish_reason': 'stop', ...}
print(response.id)                   # 'run-abc123-...'
print(response.tool_calls)           # [] (empty if no tools requested)
```

> **Production tip:** Always log `usage_metadata` in production to track costs. At scale, this is how you build token usage dashboards and catch runaway agent loops that burn through your budget.

#### The Complete Message Type Hierarchy

| Class | Role | Has `tool_calls`? | Has `usage_metadata`? | Typical Use |
|-------|------|-------------------|----------------------|-------------|
| `SystemMessage` | system | ❌ | ❌ | Set behavior, persona, guardrails |
| `HumanMessage` | human | ❌ | ❌ | User input (text, images, audio) |
| `AIMessage` | ai | ✅ | ✅ | Model output, tool requests |
| `ToolMessage` | tool | ❌ | ❌ | Return tool execution results to the model |
| `RemoveMessage` | — | ❌ | ❌ | Signal to delete a message from state (LangGraph) |

---

### C. "Stateless" Architecture

A critical concept for interviews: **LLMs are stateless.**

* The model does NOT remember you.
* When you send the second question ("How do I handle missing values?"), the model has already forgotten the first question.
* **Solution:** You must re-send the **entire chain** of messages (System + Human + AI + Human) every single time you call `.invoke()`. This is why the `messages` list in the code below is so important.

### D. `LLM` vs. `ChatModel` in LangChain

LangChain has two distinct class types:

1. **`LLM` classes (e.g., `OpenAI`):** Accept a string, return a string. (Deprecated for most modern uses).
2. **`ChatModel` classes (e.g., `ChatOpenAI`):** Accept `List[BaseMessage]`, return `AIMessage`.
* *Why it matters:* Even if a model is technically a text completion engine under the hood, the `ChatModel` wrapper handles the complex tokenization required to format the "System/Human" structure correctly for the API.

---

## 💻 Dual Examples

### 1. Example Implementation (General Chat API)
A common demonstration passes a hardcoded array of `SystemMessage`, `HumanMessage`, and `AIMessage` objects directly to `llm.invoke()` to prove that memory is just an illusion created by appending previous outputs to the input list.

### 2. Generic / Real-World Implementation (Customer Support Bot)
In a production C# backend, you don't hardcode messages. You retrieve the user's `SessionId`, query Redis or a SQL database to reconstruct the `List<BaseMessage>` of their last 10 interactions, append their new `HumanMessage`, and send the entire payload to OpenAI to generate the next response.

---

## 3. Code Implementation: Raw Message Structures

*This script demonstrates how to interact with a Chat Model using raw message objects, revealing what actually happens inside a `ChatPromptTemplate`.*

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Load Config
load_dotenv()

# 2. Initialize the Chat Model
llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

# 3. Define a "Conversation History" (The Context)
# Unlike completion models, we don't just send a query.
# We send a structured list representing the state of the conversation.
messages = [
    # SYSTEM: The "God Mode" instruction. Sets behavior/persona.
    SystemMessage(content="You are a sarcastic senior engineer who loves Python."),
    
    # HUMAN: The user's first input.
    HumanMessage(content="I am writing a script to parse CSVs."),
    
    # AI: A 'fake' history. We inject this to give the model memory of what it 'said'.
    AIMessage(content="Oh, thrilling. Another CSV parser. Groundbreaking work."),
    
    # HUMAN: The user's follow-up question.
    HumanMessage(content="Hey, be nice! How do I handle missing values with pandas?")
]

# 4. Execution
# We pass the LIST of messages, not a single string.
response = llm.invoke(messages)

# 5. Output Analysis
print(f"Role: {type(response).__name__}") # Expect: AIMessage
print(f"Content: {response.content}")
```

### Code Breakdown:
1. **Line 15 (`messages = [...]`):** We are manually constructing the state array. In a real application, this array is built dynamically.
2. **Line 17 (`SystemMessage`):** The global configuration for the LLM. It dictates the rules of engagement for the entire session.
3. **Line 23 (`AIMessage`):** We are "faking" memory here by hardcoding what the AI supposedly said previously. This is exactly how actual memory modules work under the hood—they just append the last API response to this list before the next loop.
4. **Line 31 (`llm.invoke(messages)`):** Notice we pass the `List` object, not a concatenated string. LangChain handles converting these Python objects into the specific JSON schema OpenAI expects.

---

## ⚠️ Production Notes (What Breaks & How to Fix It)

* **Context Window Exhaustion:** If you continuously append messages to the list, you will eventually exceed the model's token limit (e.g., 128k tokens) and the API will throw an HTTP 400 error.
  * **The Fix:** Implement a "Sliding Window" (only send the last N messages), or use an LLM to periodically summarize the oldest messages into a single dense context string.
* **Role Contamination (Prompt Injection):** If user input is accidentally passed as a `SystemMessage`, the user gains "God Mode" over your application.
  * **The Fix:** Strictly type your inputs. Dynamic user strings must *always* and *only* be wrapped in a `HumanMessage` class.

---

## 4. Interview Q&A Anchors

**Q: What is the difference between a SystemMessage and a HumanMessage?**
> **A:** A `SystemMessage` is a high-level instruction that sets the behavior, persona, and constraints of the AI (the "Director's notes"). A `HumanMessage` is the dynamic input from the end-user. Separating them is crucial for security, as it allows the developer to define hard constraints that the user theoretically cannot override.

**Q: Why do we pass a list of messages instead of a single string?**
> **A:** Because modern models are fine-tuned on conversation data. Passing a list allows the model to distinguish between instructions (System), past context (AI), and current input (Human). It also enables "few-shot prompting" where we provide examples of good interactions (User/AI pairs) to guide the model's performance.

**Q: If I use `ChatOpenAI`, does it remember my previous messages automatically?**
> **A:** No. The model is stateless. The developer (or the LangChain `Memory` module) must manage the list of past messages and re-send the full history with every new request.

**Q: What is the `content` field of a LangChain message — is it always a string?**
> **A:** No. For simple text, `content` is a plain string. But for multimodal input (images, audio), it becomes a list of typed content blocks — each block has a `type` field (`"text"`, `"image_url"`, `"input_audio"`) and corresponding data. This allows sending mixed content (text + images) in a single `HumanMessage` to vision models like GPT-4o.

**Q: How do you track token usage and cost per request in LangChain?**
> **A:** Every `AIMessage` returned by the model includes a `usage_metadata` property with `input_tokens`, `output_tokens`, and `total_tokens`. In production, you log this per request to build cost dashboards, detect runaway agent loops, and enforce per-user token budgets. The `response_metadata` also includes the model name and finish reason for debugging.

---