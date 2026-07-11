# ─────────────────────────────────────────────────────────────────────────────
# grade_documents.py — Document relevance filtering node
# ─────────────────────────────────────────────────────────────────────────────
# Loops through ALL retrieved documents and grades each one for relevance.
# Relevant docs are kept; irrelevant docs are dropped.
# If ANY document is irrelevant, sets web_search=True to supplement later.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Filter documents by relevance; flag web_search if any are irrelevant."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
