# CHANGE 1: Add re + inspect -- we'll parse tool calls from raw text instead of structured JSON.
# re = Regular Expressions for parsing raw text output from the LLM.
# inspect = Python's built-in reflection module. It can read a function's signature
#   (parameter names + types) and docstring AT RUNTIME without you hardcoding them.
#   This is how we auto-generate tool descriptions for the prompt -- similar to how
#   C#/Java reflection reads [Attributes] or annotations from methods at runtime.
import re
import inspect
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

# KEY DIFFERENCE FROM FILES 01 & 02:
# No @tool, no JSON schemas, no function calling API.
# Tools are just plain Python functions. We describe them to the LLM
# as plain text INSIDE the prompt itself (see PART 2 below).


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
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Tool registry -- same Service Locator pattern as Files 01 & 02.
# When we parse a tool name from the LLM's raw text, we look it up here.
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ==========================================
# PART 2: THE ReAct PROMPT TEMPLATE
# ==========================================

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.
# This is how agents worked BEFORE LLMs had built-in function calling (pre-June 2023).

def get_tool_descriptions(tools_dict):
    """Use Python's inspect module to auto-generate plain-text tool descriptions.
    This replaces the hand-written JSON schemas from File 02.
    The output looks like: get_product_price(product: str) -> float - Look up the price..."""
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # @traceable wraps our function and adds extra params like (*, config=None).
        # __wrapped__ is Python's standard way to get the ORIGINAL function back,
        # so inspect.signature() sees (product: str) instead of (product: str, *, config=None).
        original_function = getattr(tool_function, "__wrapped__", tool_function)

        # inspect.signature() reads the function's type hints at runtime.
        # For get_product_price, this returns: (product: str) -> float
        signature = inspect.signature(original_function)

        # inspect.getdoc() reads the function's docstring at runtime.
        # For get_product_price, this returns: "Look up the price of a product in the catalog."
        docstring = inspect.getdoc(tool_function) or ""

        # Combine into: "get_product_price(product: str) -> float - Look up the price..."
        # This plain-text description goes INTO the prompt so the LLM knows what tools exist.
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# The ReAct Prompt Template -- this is the ENTIRE "brain" of the agent.
# Instead of passing JSON tool schemas to the API, we describe everything in plain text.
# The LLM must follow this strict format, and we parse its output with regex.
# Based on the original ReAct paper: https://arxiv.org/abs/2210.03629
#
# THIS IS THE SAME TEMPLATE that LangChain's AgentExecutor uses internally
# (see: hwchase17/react on LangSmith Hub). The placeholders are:
#   {tool_descriptions} -- plain-text descriptions of available functions
#   {tool_names}        -- comma-separated list like "get_product_price, apply_discount"
#   {{question}}        -- the user's input (double-braces to escape f-string)
# After each iteration, we append the Thought/Action/Observation to a "scratchpad"
# string and re-inject it into the prompt. That scratchpad IS the agent's memory.
# Before June 2023 (when OpenAI added function calling), ALL agents worked this way.
react_prompt = f"""
STRICT RULES -- you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price -- do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use -- do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# WHY EACH LINE IN THE FORMAT ABOVE EXISTS (who writes it and why our code needs it):
# -----------------------------------------------------------------------
#   "Thought:"       -- Written by LLM. Forces chain-of-thought reasoning before
#                       acting. Without it, LLMs skip tools and guess answers.
#   "Action:"        -- Written by LLM. Our regex r"Action:\s*(.+)" grabs this as
#                       the tool name to execute. This IS the "function call".
#   "Action Input:"  -- Written by LLM. Our regex grabs this as the arguments to
#                       pass to the function. Replaces structured JSON args.
#   "Observation:"   -- Written by OUR CODE, never the LLM. We use stop=["\nObservation"]
#                       to halt the LLM before it writes this, then we inject the
#                       real tool result. Without stop, LLM hallucinates fake results.
#   "Final Answer:"  -- Written by LLM. Our regex exit condition. When found, loop stops.
#
# Summary: An "agent" is just a prompt + regex + stop token + loop. No magic.


# ==========================================
# PART 3: THE AGENT LOOP (THE STATE MACHINE)
# ==========================================

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent --
# all agency comes from the prompt above and our regex parsing below.
# We just send a plain text prompt and get plain text back. No structured tool_calls.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# @traceable tells LangSmith to record everything that happens inside this function.
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    # KEY DIFFERENCE: No messages array. No system/user roles.
    # The entire context is one giant string that grows with each iteration.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    # START THE LOOP (Circuit Breaker pattern -- cap at MAX_ITERATIONS)
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # STEP A: Send the prompt + scratchpad to the LLM.
        # The "stop" token is CRITICAL: it prevents the LLM from hallucinating
        # its own Observation. We stop it right before "\nObservation" so WE can
        # inject the real tool result instead of letting the LLM make one up.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # STEP B: Exit Condition -- check if the LLM produced a "Final Answer:" line.
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # STEP C: Parse the tool call from raw text using regex.
        # CHANGE 6: Parse tool calls from raw text with regex -- fragile if LLM doesn't follow format.
        # In Files 01 & 02, we read structured tool_calls from the API response.
        # Here, we literally regex the LLM's text output. If it deviates from the format, we break.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            # tools[tool_name] gets the function by name from the dict (Service Locator).
            # (*args) unpacks the list and calls it -- e.g., get_product_price("laptop").
            # The () IS the invocation -- in Python, any object with () after it gets called.
            # So tools["get_product_price"]("laptop") is the same as get_product_price("laptop").
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Raw ReAct Prompt Agent (No Function Calling, No LangChain)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
