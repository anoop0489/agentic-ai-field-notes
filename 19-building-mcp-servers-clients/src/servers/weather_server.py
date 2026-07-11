"""
MCP Weather Server (SSE transport)
====================================
A minimal MCP server that exposes a weather tool.
Communicates via SSE (Server-Sent Events) — runs as an HTTP server on a port.
Clients connect to it over the network (or localhost for development).

How to run:
    uv run 19-building-mcp-servers-clients/src/servers/weather_server.py

The server starts on http://localhost:8000 by default.
Clients connect to http://localhost:8000/sse (the MCP SSE endpoint).

Prerequisites:
    - mcp SDK installed (comes with langchain-mcp-adapters)

Transport: SSE (HTTP-based, network-accessible)

Note: SSE is deprecated in the MCP spec (2025-03-26) in favour of streamable HTTP.
To use the modern transport, change transport="sse" to transport="http" below.
Both work identically from the server author's perspective.
"""

from mcp.server.fastmcp import FastMCP

# Create a named MCP server instance.
mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # In production, this would call a real weather API.
    # This is a stub that always returns the same response.
    return "It's always sunny in New York"


if __name__ == "__main__":
    # transport="sse" means this server runs as an HTTP server.
    # The client connects via HTTP POST to http://localhost:8000/sse.
    #
    # Unlike stdio servers, SSE/HTTP servers must be started MANUALLY
    # before the client connects. They run independently.
    #
    # To change the port: mcp.run(transport="sse", port=9000)
    mcp.run(transport="sse")
