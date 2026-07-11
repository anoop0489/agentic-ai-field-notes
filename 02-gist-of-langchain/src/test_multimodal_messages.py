# =============================================================================
# MULTIMODAL MESSAGES: Sending Images to GPT-4o via Content Blocks
# =============================================================================
# Demonstrates the multimodal content type from Section 2's Message Anatomy:
#   - HumanMessage with content as a LIST of typed blocks (not just a string)
#   - Sending a PDF page as a base64 image alongside a text question
#   - Getting GPT-4o to "see" and describe diagrams, tables, flowcharts
#
# WHAT IT DOES:
#   1. Opens a PDF file and renders a specific page as a PNG image
#   2. Base64-encodes the image (the format GPT-4o expects)
#   3. Sends a HumanMessage with [text_block, image_url_block] content
#   4. GPT-4o analyzes the image and answers your question about it
#
# THIS IS THE PATTERN:
#   Instead of: HumanMessage(content="plain string")
#   We use:     HumanMessage(content=[
#                   {"type": "text", "text": "..."},
#                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
#               ])
#
# C# ANALOGY:
#   Like using Azure AI Vision / Document Intelligence — you send an image
#   payload to an AI service and get back a structured description. The
#   content field is a discriminated union: string OR List<ContentBlock>.
#   In Semantic Kernel terms: ChatMessageContentItemCollection.
#
# PREREQUISITES:
#   1. .env file with: OPENAI_API_KEY
#   2. Packages: uv add langchain-openai pymupdf python-dotenv truststore
#   3. A PDF file (defaults to "event tracking.pdf" in 09-gist-of-rag/src/)
#
# USAGE:
#   uv run 02-gist-of-langchain/src/test_multimodal_messages.py
#
# OPTIONS:
#   Change PDF_PATH, PAGE_NUMBER, and QUESTION below to target any page/question.
# =============================================================================

import base64
import os
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIGURATION =========================
# Point this at any PDF you have locally
PDF_PATH = "09-gist-of-rag/src/event tracking.pdf"

# Which page to analyze (0-indexed)
PAGE_NUMBER = 8

# Your question about the page
QUESTION = "Summarize what this page shows. If there are any diagrams or flowcharts, describe them in detail."
# =================================================================


def check_prerequisites():
    errors = []
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY not found in .env")
    if not os.path.exists(PDF_PATH):
        errors.append(f"PDF file not found: {PDF_PATH}")
    if errors:
        print("❌ Prerequisites check failed:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)


def render_page_to_base64(pdf_path: str, page_num: int) -> str:
    """Render a PDF page as a PNG and return base64-encoded string."""
    import fitz  # pymupdf

    pdf_doc = fitz.open(pdf_path)
    if page_num >= len(pdf_doc):
        print(f"❌ Page {page_num} doesn't exist. PDF has {len(pdf_doc)} pages (0-indexed).")
        sys.exit(1)

    page = pdf_doc[page_num]
    pix = page.get_pixmap(dpi=200)
    image_bytes = pix.tobytes("png")
    pdf_doc.close()

    return base64.b64encode(image_bytes).decode("utf-8")


def ask_about_image(image_base64: str, question: str) -> str:
    """Send a multimodal message (text + image) to GPT-4o and return the response."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2000)

    # THIS IS THE KEY PATTERN — multimodal content blocks
    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}",
                "detail": "high",
            },
        },
    ])

    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    check_prerequisites()

    print("=" * 70)
    print("🖼️  MULTIMODAL MESSAGE DEMO — PDF Page → GPT-4o Vision")
    print("=" * 70)
    print(f"📄 PDF:      {PDF_PATH}")
    print(f"📃 Page:     {PAGE_NUMBER + 1} (0-indexed: {PAGE_NUMBER})")
    print(f"❓ Question: {QUESTION}")
    print("-" * 70)

    print("\n🔄 Rendering PDF page to image...")
    img_b64 = render_page_to_base64(PDF_PATH, PAGE_NUMBER)
    print(f"   ✅ Image encoded: {len(img_b64)} chars (base64)")

    print("\n🧠 Sending multimodal message to GPT-4o...")
    print("   (text block + image_url block in HumanMessage.content)")
    answer = ask_about_image(img_b64, QUESTION)

    print("\n" + "=" * 70)
    print("📝 GPT-4o Response:")
    print("=" * 70)
    print(answer)
    print("\n" + "-" * 70)
    print("✅ Done! This demonstrates HumanMessage with multimodal content blocks.")
