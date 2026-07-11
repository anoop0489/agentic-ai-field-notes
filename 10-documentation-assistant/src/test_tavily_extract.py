# =============================================================================
# TAVILY QUICK TEST: Demonstrates TavilyExtract (simplest Tavily concept)
# =============================================================================
# This script extracts clean content from a single URL using TavilyExtract.
# It shows how Tavily returns structured text (not HTML) ready for LLM use.
#
# USAGE: uv run test_tavily_extract.py
# =============================================================================

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

import os
import sys

if not os.environ.get("TAVILY_API_KEY"):
    print("❌ TAVILY_API_KEY not found in .env")
    sys.exit(1)

print("🧪 TAVILY EXTRACT — Quick Demo")
print("=" * 60)
print()

# --- TavilyExtract: Single URL → Clean Content ---
from langchain_tavily import TavilyExtract

tavily_extract = TavilyExtract()

target_url = "https://python.langchain.com/docs/concepts/agents/"
print(f"📡 Extracting content from:")
print(f"   {target_url}")
print()
print("   ⏳ Calling Tavily API...")

result = tavily_extract.invoke({"urls": [target_url]})

# Show what we got back
extracted = result.get("results", [])
print(f"   ✅ Got {len(extracted)} result(s)")
print()

if extracted:
    page = extracted[0]
    content = page.get("raw_content", "")
    print(f"📋 URL: {page.get('url', 'N/A')}")
    print(f"📏 Content length: {len(content)} characters")
    print()
    print("-" * 60)
    print("📄 FIRST 500 CHARACTERS:")
    print("-" * 60)
    print(content[:500])
    print()
    print("-" * 60)
    print()
    print("💡 KEY TAKEAWAY:")
    print("   Tavily returns CLEAN TEXT — no HTML tags, no nav bars,")
    print("   no JavaScript. This is ready to chunk → embed → store.")
    print()
    print("   In our ingestion.py, TavilyCrawl does this for MANY pages")
    print("   at once by following links automatically.")
