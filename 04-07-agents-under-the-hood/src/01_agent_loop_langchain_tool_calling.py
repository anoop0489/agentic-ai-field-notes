import os
from dotenv import load_dotenv

# 1. Load environment variables (like your LANGSMITH_API_KEY) from the .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ==========================================
# PART 1: DEFINING THE TOOLS
# ==========================================

# @tool is a LangChain Magic Keyword (Decorator).
# LLMs cannot read Python code. They only read JSON. 
# @tool automatically reads the data types (product: str) and the docstring ("""Look up...""")
# and converts them into a JSON Instruction Manual so the LLM knows this tool exists.
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ==========================================
# PART 2: THE AGENT LOOP (THE STATE MACHINE)
# ==========================================

# @traceable tells LangSmith to record everything that happens inside this function.
# You can log into LangSmith's website later to see exactly what the LLM was thinking.
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    
    # 1. Group our tools together
    tools = [get_product_price, apply_discount]
    
    # 2. Create a lookup dictionary: {"get_product_price": <actual_function_code>}
    # We need this later so when the LLM says "Run get_product_price", we can find the code.
    tools_dict = {t.name: t for t in tools}

    # 3. Setup the AI Model using a Factory Method
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    
    # 4. Attach the tools to the AI
    # This does NOT run the tools. It just attaches the JSON Instruction Manuals
    # we created earlier to the LLM, saying: "You are allowed to use these if you need them."
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # 5. Create the "Agent Scratchpad" (The Conversation History)
    # The LLM has no memory. We must send it the ENTIRE history every single time we talk to it.
    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    # 6. START THE LOOP
    # We use a 'for' loop instead of 'while True' as a Circuit Breaker.
    # If the AI gets confused and loops forever, this forces it to stop at 10 to save money/CPU.
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # STEP A: Let the AI Think. We send it the history and wait for a response.
        ai_message = llm_with_tools.invoke(messages)

        # Look to see if the AI asked to use a tool
        tool_calls = ai_message.tool_calls

        # STEP B: The Exit Condition
        # If tool_calls is empty, the AI didn't need a tool. It figured out the final answer!
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # STEP C: Process the Tool Request
        # The AI asked for a tool. Let's get the details of what it wants.
        tool_call = tool_calls[0] 
        tool_name = tool_call.get("name")      # e.g., "get_product_price"
        tool_args = tool_call.get("args", {})  # e.g., {"product": "laptop"}
        tool_call_id = tool_call.get("id")     # A unique ID receipt for this specific request

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # STEP D: Execute the Python Code
        # Look up the string name in our dictionary to find the actual Python function
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Actually run the python code using the arguments the AI provided
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # STEP E: Update the Memory (Scratchpad)
        # First, save the AI's request to use the tool into history
        messages.append(ai_message)
        
        # Second, save the actual result of the tool into history.
        # We MUST use ToolMessage, and we MUST provide the exact same tool_call_id.
        # This tells the AI: "Here is the result for that specific tool you just asked for."
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

        # The loop now restarts from the top, sending this updated history back to the AI!

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
