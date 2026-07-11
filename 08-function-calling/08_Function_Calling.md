# 08. Function Calling (Theory)

*Topic: Function Calling - Understanding Function Calling for LLMs*

## What You Will Learn
* What function calling (aka tool calling) is and how it differs from the ReAct prompt.
* How the LLM produces structured JSON instead of plain text when it needs a tool.
* Why function calling was introduced and what problem it solves.
* The two main capabilities: external tool integration + structured output.
* Advantages and the one key tradeoff (opaque reasoning).

---

## What is Function Calling?

Function calling (or tool calling) is an LLM capability where the model, instead of generating plain text, outputs a **structured JSON object** that specifies a function name and its arguments -- telling your application exactly which code to execute and with what inputs.

Key points:
- The LLM generates a **well-structured JSON response** that is easy to parse.
- The function call appears in a **special place** in the response (not in the regular text content).
- Function calling is a **capability of certain LLMs** -- not all models support it. However, all state-of-the-art models from OpenAI, Anthropic, and Google now include it as standard.

---

## How It Works

1. **You provide** the model with a list of function definitions (names, parameters, descriptions).
2. **The LLM decides** whether a function should be invoked based on the user's request.
3. **The LLM responds** with a JSON object specifying which function to call and with what arguments.
4. **Your application** takes the JSON, executes the actual function, and feeds the result back to the LLM.

Behind the scenes, the model is **fine-tuned** to:
- Detect when a function should be invoked
- Format its response as valid JSON adhering to the function's schema

### Example

User asks: *"What's the weather in Paris?"*

LLM response (instead of plain text):
```json
{
  "name": "get_current_weather",
  "arguments": {
	"location": "Paris",
	"unit": "celsius"
  }
}
```

Your application then:
1. Parses this JSON
2. Calls the real `get_current_weather("Paris", "celsius")` function
3. Gets the result (e.g., "22 degrees, sunny")
4. Sends the result back to the LLM to continue the conversation

---

## Why Was Function Calling Introduced?

**The motivation: the ReAct prompt is unreliable.**

Remember File 03 (`03_raw_react_prompt.py`)? The ReAct prompt requires the LLM to output text in a strict format (`Action: ... Action Input: ...`), and we parse it with regex. The problems:

- LLMs sometimes output **bad/unparseable text** that breaks the regex
- The format is **fragile** -- small deviations crash the program
- Debugging is difficult when the LLM doesn't follow the format

Function calling solves this by moving the heavy lifting to the LLM vendor:
- All reasoning happens inside the model
- What you get back is a **deterministic, parseable JSON object**
- Much more reliable than regex-parsed text

---

## Two Main Capabilities

### 1. Connect LLMs to External Tools
The original use case -- let the LLM call functions in your application (APIs, databases, calculators, etc.)

### 2. Structured Output from the LLM
Leverage the LLM's reasoning to extract information into specific fields and return it as organized JSON. You can then convert it into a Pydantic object and downstream to your application. This is very reliable.

---

## Advantages

### Structured and Reliable Integration
- Output is **machine-readable JSON** with a specific function name and arguments
- Easy to parse, less prone to misinterpretation (vs. ReAct prompt)
- The model is fine-tuned to adhere to the function schema strictly
- Reduces random formatting errors that we saw with the ReAct prompt

### Token Efficient
- No chain-of-thought text output (no `Thought: ...` verbose reasoning)
- The model skips verbose explanations and **directly returns the function call**
- Fewer tokens = lower cost and faster responses

---

## The One Tradeoff: Opaque Reasoning

When the model decides to call a function, it does so **without exposing its chain of thought**.

- We see the final function name and arguments, but NOT the justification
- We don't understand WHY it chose that particular function with those specific arguments
- Function calling becomes a **black box decision** -- no intermediate rationale exposed
- This makes debugging and auditing the model's decisions harder

**However:** de facto, function calling is now the standard. Nobody uses the raw ReAct prompt in production anymore. LLM vendors (OpenAI, Google, Anthropic) have perfected function calling to the point where it's reliable enough to build robust AI agents and applications.

---

## Comparison: ReAct Prompt vs Function Calling

| Aspect | ReAct Prompt (File 03) | Function Calling (Files 01 & 02) |
|--------|------------------------|----------------------------------|
| **Output format** | Raw text: `Action: tool_name` | Structured JSON: `{"name": "tool_name", "arguments": {...}}` |
| **Parsing** | Regex (fragile) | JSON parsing (reliable) |
| **Reasoning visibility** | Full chain-of-thought visible | Opaque -- reasoning is internal |
| **Token usage** | High (Thought + Action + verbose text) | Low (just the function call JSON) |
| **Reliability** | Can fail if LLM deviates from format | Very reliable (model fine-tuned for it) |
| **Who does the work** | You (prompt engineering + regex) | LLM vendor (fine-tuning + schema enforcement) |
| **Introduced** | 2022 (Yao et al. paper) | June 2023 (OpenAI) |

---

## How This Connects to What We Already Built

| File | Approach | Uses Function Calling? |
|------|----------|----------------------|
| `01_agent_loop_langchain_tool_calling.py` | LangChain | Yes -- `bind_tools()` sends schemas, LLM returns `tool_calls` |
| `02_agent_loop_raw_function_calling.py` | Raw Ollama SDK | Yes -- `tools=` param sends schemas, LLM returns `tool_calls` |
| `03_raw_react_prompt.py` | Raw ReAct prompt | No -- plain text + regex (pre-function-calling era) |

Files 01 and 02 are already using function calling. File 03 shows what we did BEFORE function calling existed. This section explains the theory behind why Files 01/02 are so much more reliable than File 03.

---

## Interview Q&A Anchors

**Q: What is function calling in the context of LLMs?**

> **A:** Function calling is a capability where the LLM, instead of generating plain text, produces a structured JSON object specifying which function to call and with what arguments. The developer provides function definitions (name, parameters, description) to the model, and the model is fine-tuned to detect when a function should be invoked and format its response as valid JSON adhering to the schema.

**Q: What problem does function calling solve that the ReAct prompt couldn't?**

> **A:** The ReAct prompt requires the LLM to output text in a strict format that we parse with regex. This is fragile -- the LLM can deviate from the format, producing unparseable output that crashes the application. Function calling moves the heavy lifting to the LLM vendor: the model is fine-tuned to output valid JSON, making it far more reliable and deterministic.

**Q: What are the two main capabilities function calling gives developers?**

> **A:** (1) Connecting LLMs to external tools -- letting the model invoke functions in your application (APIs, databases, etc.). (2) Getting structured output -- leveraging the LLM's reasoning to extract information into specific fields and return it as organized JSON that can be converted to typed objects (e.g., Pydantic models).

**Q: What is the main tradeoff of function calling vs the ReAct prompt?**

> **A:** Opaque reasoning. With ReAct, you see the full chain of thought (Thought/Action/Observation). With function calling, the reasoning is internal to the model -- you only see the final function name and arguments, not why the model chose them. This makes debugging harder, but the reliability gain is worth it.

**Q: Why is function calling more token-efficient than ReAct?**

> **A:** ReAct requires the model to output verbose chain-of-thought text (Thought: ..., Action: ..., etc.) before every tool call. Function calling skips this -- the model directly returns the function call JSON without verbose reasoning, using fewer tokens and reducing cost.

---

## Deep Dive: The Tool Calling Flow (from OpenAI Docs)

The tool calling flow is a **multi-step conversation** between your application and the model:

```
1. Your app sends a request with tools the model could call
2. Model returns a tool call (function name + arguments JSON)
3. Your app executes the function with those arguments
4. Your app sends the tool output back to the model
5. Model returns a final text response (or more tool calls)
```

Key vocabulary:
- **Tool** -- a piece of functionality you tell the model it has access to
- **Tool call** -- the model's request to use a tool (contains `name`, `arguments`, `call_id`)
- **Tool call output** -- the result you send back (referenced by `call_id`)

### Why `call_id` Matters

Each tool call the model generates has a unique `call_id` (e.g., `call_abc123`). When you send the result back, you MUST include this exact `call_id` so the model can match your result to the specific call it made.

```python
# File 01 does this:
messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
#                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
#                                                      This links result to the specific call
```

If the `call_id` is missing or wrong, the API returns HTTP 400 -- the model's state history is corrupted. In File 02 (Ollama), this isn't required because Ollama's SDK handles it implicitly.

---

## Function Schema Definition (OpenAI Format)

Each function definition has these fields:

| Field | Description |
|-------|-------------|
| `type` | Always `"function"` |
| `name` | The function's name (e.g., `get_weather`) |
| `description` | When and how to use the function |
| `parameters` | JSON Schema defining input arguments |
| `strict` | Whether to enforce strict schema adherence (recommended: `true`) |

Example:
```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Retrieves current weather for the given location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and country e.g. Paris, France"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location", "units"],
    "additionalProperties": false
  },
  "strict": true
}
```

This is exactly what we hand-wrote in `02_agent_loop_raw_function_calling.py` (PART 2).

---

## Tool Choice Configurations

You can control when the model uses tools:

| Setting | Behavior |
|---------|----------|
| `"auto"` (default) | Model decides -- may call zero, one, or multiple functions |
| `"required"` | Model MUST call at least one function |
| `{"type": "function", "name": "get_weather"}` | Model MUST call this specific function |
| `"none"` | Model cannot call any functions (text-only response) |

---

## Best Practices for Defining Functions (from OpenAI)

1. **Write clear descriptions** -- explicitly describe purpose, parameters, and output format
2. **Use enums** -- constrain values where possible (e.g., `"enum": ["celsius", "fahrenheit"]`)
3. **Don't make the model fill arguments you already know** -- if you have the `order_id`, don't make it a parameter
4. **Combine functions called in sequence** -- if you always call B after A, merge them
5. **Keep available functions small** -- aim for fewer than 20 tools at any time
6. **Enable strict mode** -- set `strict: true` for reliable schema adherence
7. **Token awareness** -- function definitions count as input tokens (they're injected into the system message)

---

## LangChain's Modern Agent API: `create_agent`

LangChain now provides `create_agent` -- a minimal, configurable harness:

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-4o",          # swap provider by changing this string
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
)
```

This is the evolution of what we used in File 01 (`init_chat_model` + `bind_tools` + manual loop).
Now LangChain handles the entire loop for you -- same provider-agnostic swap with one string.

Supported providers (same `model=` string pattern):
- `"openai:gpt-4o"` / `"openai:gpt-5.4"`
- `"anthropic:claude-sonnet-4-6"`
- `"google_genai:gemini-2.5-flash-lite"`
- `"ollama:qwen3:1.7b"`
- `"azure_openai:gpt-4o"`
- `"fireworks:..."` / `"bedrock_converse:..."` / `"huggingface:..."`

---

## Parallel Function Calling

Modern models can call **multiple functions in a single turn**. The `tool_calls` array may contain several entries.

In production, instead of just processing `tool_calls[0]`, you'd iterate over all calls:

```python
# Production pattern (handles parallel tool calls)
for tool_call in ai_message.tool_calls:
    result = tools[tool_call.name](**tool_call.args)
    messages.append(ToolMessage(content=str(result), tool_call_id=tool_call.id))
```

You can disable this with `parallel_tool_calls=false` to force one tool per turn (which is what our learning files do for clarity).

---

## LangChain vs LangGraph vs Deep Agents (Where This All Fits)

LangChain's ecosystem now has three levels. Knowing this helps you pick the right tool:

| Level | When to use | What it gives you |
|-------|-------------|-------------------|
| **LangChain** (`create_agent`) | Standard agent with tools | Minimal harness: model + tools + prompt + loop. Highly customizable. |
| **LangGraph** | Complex multi-step workflows | Low-level orchestration: deterministic + agentic workflows, persistence, human-in-the-loop. |
| **Deep Agents** | "Batteries included" agent | Auto context compression, virtual filesystem, sub-agent spawning. Built on top of LangChain agents. |

These notes are currently at the **LangChain** level (Files 01-04). LangGraph will come in later sections when we need more complex state management and multi-agent workflows.

---

## References

- [OpenAI: Function Calling and Other API Updates (June 2023 announcement)](https://openai.com/index/function-calling-and-other-api-updates/)
- [OpenAI: Function Calling Guide](https://developers.openai.com/api/docs/guides/function-calling)
- [LangChain Python Documentation](https://docs.langchain.com/oss/python/langchain/overview)
