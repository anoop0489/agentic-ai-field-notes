# ==========================================================================
# NOTE: This entire file is written specifically for the OLLAMA Python SDK.
# The SDK calls (ollama.chat()), message format (plain dicts with "role"),
# tool schema format (OpenAI-style JSON), and response objects (.message,
# .tool_calls, .function.name) are ALL Ollama-specific.
# If you switch to another provider (OpenAI SDK, Anthropic SDK, etc.),
# you would need to rewrite the API calls, message structures, and
# response parsing to match that provider's SDK.
# This is the exact problem LangChain solves -- one interface for all providers.
#
# WHY LANGCHAIN MATTERS (remember this):
#   Without LangChain: You are locked to ONE provider. Switching from Ollama to
#   OpenAI or Anthropic means rewriting: API calls, message dicts, tool schemas,
#   response parsing, and error handling. That's 80% of this file.
#   With LangChain:    You change ONE string: init_chat_model("openai:gpt-4o") ->
#   init_chat_model("ollama:qwen3") -> init_chat_model("anthropic:claude-3").
#   Everything else (tools, messages, loops) stays identical. That's the value.
# ==========================================================================

from dotenv import load_dotenv

# 1. Load environment variables (like your LANGSMITH_API_KEY) from the .env file
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ==========================================
# PART 1: DEFINING THE TOOLS
# ==========================================

# KEY DIFFERENCE FROM FILE 01:
# NO @tool decorator here. These are just plain Python functions.
# Without @tool, LangChain won't auto-generate JSON schemas for us.
# We'll have to write those schemas by hand (see PART 2 below).

# @traceable is NOT a LangChain thing -- it's from LangSmith for observability only.
# It does NOT generate JSON schemas. It just logs the function call for debugging.
# run_type= tells LangSmith how to categorize this in the trace UI:
#   "tool"      ? Tool execution (function the agent called)
#   "llm"       ? LLM API call (shows tokens, latency, model info)
#   "chain"     ? Chain/pipeline step (default if omitted)
#   "retriever" ? Retrieval step (for RAG workflows)

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# ==========================================
# PART 2: HAND-WRITTEN JSON TOOL SCHEMAS
# ==========================================

# !!! IMPORTANT: PROVIDER-SPECIFIC FORMAT !!!
# This JSON schema format is specific to OpenAI/Ollama.
# Each LLM provider uses a DIFFERENT format for tool definitions:
#   OpenAI/Ollama  -> {"type": "function", "function": {"name": ..., "parameters": {...}}}
#   Anthropic      -> {"name": ..., "description": ..., "input_schema": {...}}
#   Google Gemini  -> {"function_declarations": [{"name": ..., "parameters": {...}}]}
# If you switch providers, you MUST rewrite these schemas.
# This is exactly why LangChain's @tool + bind_tools() exists -- it handles this for you.

# DIFFERENCE 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
# Think of this like writing a Swagger/OpenAPI spec manually instead of using [ApiController].
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# ==========================================
# PART 3: THE AGENT LOOP (THE STATE MACHINE)
# ==========================================

# --- Helper: traced Ollama call ---
# DIFFERENCE 3: Without LangChain, we must manually trace LLM calls for LangSmith.
# In File 01, init_chat_model() returns a LangChain BaseChatModel which has AUTOMATIC
# LangSmith tracing built-in -- every .invoke() is logged without you doing anything.
# Here, ollama.chat() is a raw SDK call with ZERO tracing. LangSmith doesn't know it
# exists unless we manually wrap it with @traceable. That's what this function does.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# @traceable tells LangSmith to record everything that happens inside this function.
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    # 1. Create the tool registry (Service Locator Pattern)
    # When the LLM says "call get_product_price", we look up this dict to find the function.
    # In File 01, LangChain built this for us with: {t.name: t for t in tools}
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }


    print(f"Question: {question}")
    print("=" * 60)

    # 2. Create the "Agent Scratchpad" (Conversation History)
    # KEY DIFFERENCE: Messages are plain dicts instead of typed objects (SystemMessage, HumanMessage).
    # The LLM has no memory -- we must send the ENTIRE history every single time.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES -- you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price -- do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use -- do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # 3. START THE LOOP (Circuit Breaker pattern -- cap at MAX_ITERATIONS)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # STEP A: Send the full message history to the LLM via Ollama SDK directly.
        # DIFFERENCE 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # STEP B: Exit Condition -- if no tool calls, the LLM has the final answer.
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # STEP C: Process the Tool Request
        # Process only the FIRST tool call -- force one tool per iteration
        tool_call = tool_calls[0]
        # DIFFERENCE 6: Attribute access (.function.name) instead of dict access (.get("name"))
        # because Ollama returns typed objects, not dicts like LangChain.
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # STEP D: Execute the Python Code (Service Locator lookup)
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # DIFFERENCE 7: Direct function call with **kwargs instead of tool.invoke(args)
        # In File 01: observation = tool_to_use.invoke(tool_args)
        # Here:       observation = tool_to_use(**tool_args)
        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        # STEP E: Update the Memory (Scratchpad)
        # KEY DIFFERENCE: We append raw dicts instead of typed ToolMessage objects.
        # In File 01: messages.append(ToolMessage(content=str(observation), tool_call_id=...))
        # Here: We just append a plain {"role": "tool"} dict. No tool_call_id needed with Ollama.
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

        # The loop restarts, sending the updated history back to the LLM!

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Raw Function Calling Agent (No LangChain)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
