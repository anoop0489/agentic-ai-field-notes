# =============================================================================
# AI AGENTS & TOOLS: ReAct Agent with Tavily Search
# =============================================================================
# Demonstrates the shift from deterministic chains to autonomous agents:
#   - Agent uses LLM as a REASONING ENGINE to decide what to do
#   - ReAct loop: Thought → Action → Observation → (repeat or Final Answer)
#   - AgentExecutor: The while loop that physically runs tools
#   - agent_scratchpad: Memory injection for the stateless LLM
#
# WHAT IT DOES:
#   Creates a ReAct agent with a Tavily web search tool. The agent
#   autonomously decides to search the web, reads results, and generates
#   a final answer — all without hardcoded logic.
#
# KEY CONCEPTS:
#   - create_tool_calling_agent(): Factory that binds tools to LLM
#   - AgentExecutor: Runtime engine (while loop + state machine)
#   - agent_scratchpad: Where previous Thoughts/Actions/Observations are injected
#   - TavilySearchResults: AI-optimized search (clean text, not raw HTML)
#   - verbose=True: Prints the internal ReAct reasoning to console
#
# C# ANALOGY:
#   A Chain is: var result = Step3(Step2(Step1(input)));
#   An Agent is: while(!state.IsFinished) { var cmd = agent.Decide(state); state.Update(cmd.Execute()); }
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY, TAVILY_API_KEY
#   2. Packages: uv add langchain langchain-openai langchain-community python-dotenv truststore
#
# USAGE:
#   uv run 03-gist-of-ai-agents/src/04_agent_and_tools.py
# =============================================================================

import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "reference-guides"))
from logger import log_header, log_info, log_success, log_error, log_warning

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor


def check_prerequisites():
    """Verify required env vars exist before making API calls."""
    errors = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        errors.append("TAVILY_API_KEY")

    if errors:
        for var in errors:
            log_error(f"{var} not found in .env")
        sys.exit(1)
    log_success("Prerequisites met — OPENAI_API_KEY + TAVILY_API_KEY found")


def main():
    check_prerequisites()

    log_header("AI AGENTS: ReAct Agent with Tavily Search")

    # ========================= DEFINE TOOLS =========================
    # TavilySearchResults: AI-optimized web search
    # Unlike Google, Tavily returns clean text (not HTML) — saves tokens
    # max_results=2: Limits response size to avoid context window overflow
    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool]
    log_info(f"Tools registered: {[t.name for t in tools]}")
    log_info("  TavilySearch returns clean text, not raw HTML")

    # ========================= THE MODEL =========================
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    log_info("Model: gpt-4o (temperature=0 for deterministic reasoning)")

    # ========================= THE AGENT PROMPT =========================
    # The prompt MUST include {agent_scratchpad} — this is where the
    # AgentExecutor injects previous Thoughts/Actions/Observations
    # Without it, the agent forgets what it already searched for!
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful job search assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # REQUIRED for ReAct memory
    ])
    log_info("Prompt includes: system, human, agent_scratchpad (ReAct memory)")

    # ========================= CREATE THE AGENT =========================
    # This factory method:
    #   1. Converts our tools into JSON schema
    #   2. Sends schema to OpenAI so the model knows these functions exist
    #   3. Returns an agent that can generate tool-call requests
    # NOTE: This does NOT execute anything — it's just building the graph
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    log_info("Agent created (tools bound to LLM via JSON schema)")

    # ========================= CREATE THE EXECUTOR =========================
    # AgentExecutor is the RUNTIME ENGINE — it's a while loop that:
    #   1. Asks the LLM what to do
    #   2. If LLM says "call tool X" → executor runs X locally
    #   3. Feeds result back to LLM (via agent_scratchpad)
    #   4. Repeats until LLM says "I have the final answer"
    # verbose=True: Prints the internal Thought/Action/Observation loop
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    log_info("AgentExecutor ready (verbose=True for demo visibility)")

    # ========================= EXECUTION =========================
    log_header("RUNNING AGENT (watch the ReAct loop below)")
    question = "Find 2 Senior AI Engineer job postings in Texas."
    log_info(f"User question: \"{question}\"")
    log_warning("The agent will now AUTONOMOUSLY decide to search the web...")
    print()

    # .invoke() triggers the full ReAct loop
    # The agent may call Tavily multiple times before producing a final answer
    response = agent_executor.invoke({"input": question})

    # ========================= OUTPUT =========================
    log_header("FINAL ANSWER")
    log_success("Agent completed its reasoning loop!")
    print()
    print(response["output"])
    print()
    log_info("The agent decided WHAT to search and WHEN to stop — no hardcoded logic!")


if __name__ == "__main__":
    main()
