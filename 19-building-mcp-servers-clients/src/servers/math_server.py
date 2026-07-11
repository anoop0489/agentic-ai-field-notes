"""
MCP Math Server (stdio transport)
==================================
A minimal MCP server that exposes two mathematical tools: add and multiply.
Communicates via stdio — the CLIENT launches this as a subprocess and talks
to it through stdin/stdout pipes.

How to run (for testing with MCP Inspector):
    uv run 19-building-mcp-servers-clients/src/servers/math_server.py

In production use: The client launches this automatically via StdioServerParameters.
You do NOT start stdio servers manually — the client manages the subprocess lifecycle.

Prerequisites:
    - mcp SDK installed (comes with langchain-mcp-adapters)

Transport: stdio (local subprocess, not network-accessible)
"""

from mcp.server.fastmcp import FastMCP

# Create a named MCP server instance.
# The name "Math" is metadata — clients see this when they connect.
mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    # transport="stdio" means this server communicates via stdin/stdout.
    # The client (e.g., langchain-mcp-adapters) spawns this as a subprocess
    # and sends/receives JSON-RPC messages through the pipes.
    mcp.run(transport="stdio")
