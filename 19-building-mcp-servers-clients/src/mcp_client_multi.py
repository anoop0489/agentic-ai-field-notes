"""
MCP Multi-Server Client (stdio + SSE transports)
==================================================
Demonstrates connecting to MULTIPLE MCP servers simultaneously using
LangChain's MultiServerMCPClient, then using all tools in a LangGraph agent.

What this script does:
1. Connects to math_server.py via stdio (launched as subprocess)
2. Connects to weather_server.py via SSE (must already be running on port 8000)
3. Loads tools from BOTH servers into a unified list
4. Creates a ReAct agent that can call tools from either server
5. Asks a math question (hits math server) and a weather question (hits weather server)

How to run:
    # Terminal 1: Start the SSE weather server first
    uv run 19-building-mcp-servers-clients/src/servers/weather_server.py

    # Terminal 2: Run the multi-server client
    uv run 19-building-mcp-servers-clients/src/mcp_client_multi.py

Prerequisites:
    - OPENAI_API_KEY in .env file
    - Weather server running on http://localhost:8000
    - langchain-mcp-adapters, langgraph, langchain-openai installed
"""

import asyncio
import os
from pathlib import Path

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


def check_prerequisites():
    """Validate that required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"❌ Missing environment variables: {', '.join(missing)}\n"
            f"   Add them to your .env file."
        )


async def main():
    check_prerequisites()

    llm = ChatOpenAI(model="gpt-4o")

    # Build the absolute path to the math server script.
    math_server_path = str(
        Path(__file__).parent / "servers" / "math_server.py"
    )

    print("=" * 60)
    print("🔌 Connecting to multiple MCP servers...")
    print("=" * 60)

    # MultiServerMCPClient manages connections to multiple MCP servers.
    # Each server is identified by a key name (used for logging/debugging).
    #
    # - "math": Uses stdio transport — the client launches math_server.py
    #   as a subprocess automatically. No manual server start needed.
    #
    # - "weather": Uses SSE transport — connects to an already-running
    #   HTTP server. You must start weather_server.py separately first.
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/sse",
                "transport": "sse",
            },
        }
    )

    # get_tools() connects to all servers, fetches their tool lists,
    # and returns them as a unified list of LangChain Tool objects.
    tools = await client.get_tools()
    print(f"✅ Loaded {len(tools)} tools: {[t.name for t in tools]}")

    # Create an agent that has access to ALL tools from both servers.
    # The LLM decides which tool to call based on the user's question.
    agent = create_agent(llm, tools)

    # Math question — will route to the math server's add/multiply tools
    print("-" * 60)
    math_question = "What is (3 + 5) x 12?"
    print(f"❓ Math question: {math_question}")
    math_result = await agent.ainvoke({"messages": math_question})
    print(f"💡 Answer: {math_result['messages'][-1].content}")

    # Weather question — will route to the weather server's get_weather tool
    print("-" * 60)
    weather_question = "What is the weather in New York?"
    print(f"❓ Weather question: {weather_question}")
    weather_result = await agent.ainvoke({"messages": weather_question})
    print(f"💡 Answer: {weather_result['messages'][-1].content}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
