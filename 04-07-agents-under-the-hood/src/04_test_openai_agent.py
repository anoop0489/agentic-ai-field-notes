"""
04_test_openai_agent.py -- Quick Test Runner (OpenAI Cloud)

PURPOSE:
  This file is a simplified version of 01_agent_loop_langchain_tool_calling.py
  that uses OpenAI (gpt-4o-mini) instead of a local Ollama model.
  Use this when you don't have Ollama installed or want to verify the agent
  works against a cloud LLM.

  It demonstrates the Factory Pattern: the exact same agent loop works with
  any provider -- just swap the model string.

PREREQUISITES:
  - OPENAI_API_KEY set in your .env file
  - Run with: uv run 04-07-agents-under-the-hood/src/04_test_openai_agent.py

CORPORATE NETWORK NOTE:
  This file includes SSL workarounds for corporate proxy/firewall environments
  that perform SSL inspection. If you're on a normal network, these are harmless.
"""
import os
import ssl
from dotenv import load_dotenv

# 1. Load environment variables (OPENAI_API_KEY) from the .env file
load_dotenv()

# LangSmith tracing -- disabled on corporate networks (SSL blocked).
# Set to "true" when running from home/non-firewalled network to see traces
# at https://smith.langchain.com project: "langchain-course"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_PROJECT"] = "langchain-course"

# Fix SSL for corporate networks (disables certificate verification globally)
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MODEL = "gpt-4o-mini"

# Create an httpx client that skips SSL verification (corporate proxy workaround)
http_client = httpx.Client(verify=False)


# ==========================================
# PART 1: DEFINING THE TOOLS
# ==========================================

# Same tools as Files 01–03. Using @tool so LangChain auto-generates JSON schemas.


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
# PART 2: THE AGENT LOOP
# ==========================================


def run_agent(question: str):
    # 1. Group tools and create the lookup registry
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # 2. Setup the AI Model
    # KEY DIFFERENCE from File 01: Using ChatOpenAI directly with http_client
    # for SSL workaround. In File 01 we use init_chat_model() factory.
    llm = ChatOpenAI(model=MODEL, temperature=0, http_client=http_client)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # 3. Create the conversation history
    messages = [
        SystemMessage(content="You are a helpful shopping assistant. Use the tools provided."),
        HumanMessage(content=question),
    ]

    # 4. The Agent Loop (same logic as File 01, simplified)
    for iteration in range(1, 6):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # Exit condition -- no tool calls means final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        # Append AI request + tool result to history for next iteration
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Running Agent with OpenAI (gpt-4o-mini)...")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")

