# ─────────────────────────────────────────────────────────────────────────────
# tool_executor.py — Search tool execution for the Reflexion Agent
# ─────────────────────────────────────────────────────────────────────────────
# This file sets up the ToolNode that executes search queries generated
# by the LLM's structured output.
#
# KEY TRICK: The ToolNode expects tools whose NAMES match the tool_calls
# from the LLM. Since we forced the LLM to use tool_choice="AnswerQuestion"
# or "ReviseAnswer", the tool_calls will have those names. So we create
# StructuredTools with matching names that both run the same search function.
#
# Flow:
#   LLM returns tool_calls=[{name: "AnswerQuestion", args: {search_queries: [...]}}]
#   → ToolNode looks up tool by name "AnswerQuestion"
#   → Calls run_queries(search_queries=["query1", "query2"])
#   → Returns ToolMessage with Tavily search results
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

# Tavily search engine — optimized for LLM consumption, returns clean results
tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated search queries in batch via Tavily.

    The LLM generates 1-3 search queries as part of its structured output.
    We batch them all at once for efficiency, and the results are returned
    as a ToolMessage that gets appended to the conversation state."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


# Create ToolNode with tools named to match the LLM's tool_calls.
# When LLM uses tool_choice="AnswerQuestion", the tool_call name is "AnswerQuestion".
# When LLM uses tool_choice="ReviseAnswer", the tool_call name is "ReviseAnswer".
# Both execute the same search function — the name matching is what matters.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
