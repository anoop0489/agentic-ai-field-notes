# ─────────────────────────────────────────────────────────────────────────────
# consts.py — Node name constants for the Agentic RAG graph
# ─────────────────────────────────────────────────────────────────────────────
# Using constants avoids typos and makes refactoring safe — change the string
# once, it updates everywhere (edges, conditionals, imports).
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
