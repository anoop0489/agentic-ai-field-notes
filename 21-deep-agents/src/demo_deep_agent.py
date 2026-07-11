# ─────────────────────────────────────────────────────────────────────────────
# demo_deep_agent.py — Deep Agent SRE Triage Demo (Virtual Filesystem)
# ─────────────────────────────────────────────────────────────────────────────
# What this does:
#   Demonstrates a Deep Agent acting as an automated Site Reliability Engineer.
#   The agent is handed a virtual filesystem (backed by a real disk folder)
#   containing a crash log, then uses the built-in file tools (ls, read_file,
#   write_file) to:
#     1. List files at the virtual root "/"
#     2. Read /server_crash.log
#     3. Diagnose the root cause of the transaction failure
#     4. Write a Markdown triage report to /triage_report.md
#
#   This showcases the Deep Agents "execution environment" pillar — specifically
#   the virtual filesystem backed by a FilesystemBackend (see doc 29, §5.2).
#
# Filesystem mapping (virtual_mode=True maps "/" onto the workspace folder):
#   Agent path  "/"                 → ./demo_workspace/
#   Agent path  "/server_crash.log" → ./demo_workspace/server_crash.log
#   Agent path  "/triage_report.md" → ./demo_workspace/triage_report.md
#
#   The ./demo_workspace/ folder is created automatically at runtime by
#   setup_mock_production_environment() and is git-ignored (generated output).
#
# Prerequisites:
#   - OPENAI_API_KEY in .env  (this demo uses openai:gpt-4o)
#   - Dependencies: deepagents, langchain-openai, python-dotenv, truststore
#
# How to run:
#   cd 21-deep-agents/src
#   uv run python demo_deep_agent.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import shutil

# ─── Corporate proxy SSL fix (must be FIRST before any network imports) ──────
import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# ─── LangSmith tracing → dedicated project ───────────────────────────────────
os.environ["LANGSMITH_PROJECT"] = "deep-agents"

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


# ─── Validate env vars before doing anything ─────────────────────────────────
def check_prerequisites():
    required = ["OPENAI_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("   Add them to your .env file")
        sys.exit(1)


def setup_mock_production_environment(workspace_path: str):
    """Creates a mock staging folder to simulate a live server workspace."""
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.makedirs(workspace_path)

    crash_log_content = """2026-07-08T22:00:01Z [INFO] Gateway API initialized on port 8080.
2026-07-08T22:05:14Z [WARN] Database pool connection latency exceeded 500ms.
2026-07-08T22:10:00Z [CRITICAL] TRANSACTION FAILURE: Account ID 'ACC-9942' attempted transfer of $120,000.00.
2026-07-08T22:10:01Z [ERROR] ZeroDivisionError: division by zero in calculate_risk_multiplier() at line 42 of order_processor.py.
2026-07-08T22:10:02Z [CRITICAL] Microservice terminated abruptly. System memory dumped.
"""

    with open(os.path.join(workspace_path, "server_crash.log"), "w", encoding="utf-8") as f:
        f.write(crash_log_content)

    print(f"📁 Mock environment: '{workspace_path}' with 'server_crash.log'\n")


def run_enterprise_agent_demo():
    workspace_dir = "./demo_workspace"
    setup_mock_production_environment(workspace_dir)

    # virtual_mode=True: "/" maps to root_dir, agent sees /server_crash.log
    backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)

    agent = create_deep_agent(
        model="openai:gpt-4o",
        system_prompt=(
            "You are an automated Site Reliability Engineer (SRE). "
            "Read server logs, diagnose the root cause, and write a Markdown triage report. "
            "Your filesystem root '/' is your workspace. Files are at paths like /server_crash.log."
        ),
        backend=backend,
    )

    run_config = {"configurable": {"thread_id": "sre_triage_001"}}

    task_input = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "List files at /. Read /server_crash.log. "
                    "Analyze why the transaction failed and write a remediation report to /triage_report.md."
                ),
            }
        ]
    }

    print("⚡ Running agent...\n")

    for chunk in agent.stream(task_input, run_config, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print("--- Agent ---")
                print(last_message.content, "\n")
            elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"🛠️  Tool: {last_message.tool_calls[0]['name']}\n")

    # Check if the agent created the report
    report_path = os.path.join(workspace_dir, "triage_report.md")
    if os.path.exists(report_path):
        print("=" * 60)
        print("📄 GENERATED REPORT:")
        print("=" * 60)
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("⚠️  Agent did not create triage_report.md")


if __name__ == "__main__":
    check_prerequisites()
    run_enterprise_agent_demo()