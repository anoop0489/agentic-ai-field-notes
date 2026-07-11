"""
MCP Single-Server Client (stdio transport)
============================================
Demonstrates connecting to a SINGLE MCP server via stdio and using its tools
in a LangGraph ReAct agent.

What this script does:
1. Launches the math_server.py as a subprocess (stdio transport)
2. Establishes an MCP session and loads the server's tools
3. Converts MCP tools → LangChain tools via load_mcp_tools()
4. Creates a ReAct agent and asks it a math question
5. The agent calls the add/multiply tools via MCP protocol to solve it

How to run:
    uv run 19-building-mcp-servers-clients/src/mcp_client_single.py

Prerequisites:
    - OPENAI_API_KEY in .env file
    - langchain-mcp-adapters, langgraph, langchain-openai installed
"""

import asyncio
import os
from pathlib import Path

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
    # StdioServerParameters requires absolute paths — relative paths fail
    # because the subprocess doesn't inherit the working directory.
    math_server_path = str(
        Path(__file__).parent / "servers" / "math_server.py"
    )

    server_params = StdioServerParameters(
        command="python",
        args=[math_server_path],
    )

    print("=" * 60)
    print("🔌 Connecting to math MCP server (stdio)...")
    print("=" * 60)

    # stdio_client() launches math_server.py as a subprocess.
    # It returns read/write streams connected to the subprocess's stdin/stdout.
    async with stdio_client(server_params) as (read, write):
        # ClientSession establishes the MCP protocol handshake.
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("✅ MCP session initialized")

            # load_mcp_tools() calls session.list_tools() under the hood,
            # then wraps each MCP tool as a LangChain Tool object.
            tools = await load_mcp_tools(session)
            print(f"🛠️  Loaded {len(tools)} tools: {[t.name for t in tools]}")

            # Create an agent that can use the MCP tools.
            agent = create_agent(llm, tools)

            print("-" * 60)
            question = "What is 54 + 2 * 3?"
            print(f"❓ Question: {question}")
            print("-" * 60)

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]}
            )
            print(f"💡 Answer: {result['messages'][-1].content}")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
