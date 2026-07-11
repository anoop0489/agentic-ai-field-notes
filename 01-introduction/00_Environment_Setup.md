# 00. Environment Setup & Visual Studio Integration ⚡

*Topic: Introduction & Setup*

## 🎯 What You Will Learn
* How to replace legacy Python tooling (pip, venv) with `uv`, a next-generation package manager.
* Understanding declarative project structures (`pyproject.toml`).
* How to wire up a Python virtual environment to the Visual Studio IDE for IntelliSense.
* Secure secret management using `.env` files.

## 📦 Dependency Setup
Since this module is the setup itself, open your **PowerShell** as Administrator and run the following command to install `uv` globally on your machine:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
*(Note: Close and reopen your terminal/Visual Studio after installation to ensure the `uv` command is recognized).*

### Per-Module Packages

Each module adds its own dependencies. Here's the full list across these notes:

| Module | Command | Key Packages |
| :--- | :--- | :--- |
| **01 - Fundamentals** | `uv add langchain langchain-openai python-dotenv truststore` | Core LangChain + OpenAI + secrets + SSL |
| **03 - LangSmith** | `uv add langsmith` | Tracing & observability |
| **04-07 - Agents Under the Hood** | `uv add ollama langsmith httpx` | Local LLM (Ollama), tracing, HTTP client |
| **09 - RAG (Embeddings & Retrieval)** | `uv add langchain-pinecone langchain-community langchain-text-splitters pymupdf pypdf pdf2image` | Vector store, PDF loaders, chunking |
| **10 - Documentation Assistant** | `uv add langchain-tavily streamlit` | Web crawling (Tavily), chat UI (Streamlit) |
| **13 - LangGraph Fundamentals** | `uv add langgraph` | Stateful agent orchestration, checkpointers, streaming |
| **14 - Reflection Agent** | *(no new packages)* | Uses `langgraph` + `langchain-openai` from earlier modules |
| **15 - Reflexion Agent** | `uv add langchain-tavily` | Web search tool for iterative refinement |
| **16 - Agentic RAG** | `uv add langchain-chroma langchain-community` | Local vector store (Chroma), document loaders (WebBaseLoader) |
| **17 - Introduction to MCP** | *(documentation only)* | Conceptual overview — no Python packages |
| **18 - Pre-built MCP Server** | *(external tool: `npx @nicepkg/mcpdoc`)* | Uses npx/Node.js, not a pip package |
| **19 - Building MCP Servers & Clients** | `uv add langchain-mcp-adapters langgraph langchain-openai` | MCP↔LangChain bridge, agent framework |
| **20 - Useful LLM Dev Tools** | *(external tools only)* | No new pip packages — uses MCP server config, LangSmith Hub, Streamlit playground |

> 💡 Some packages listed above depend on earlier modules. For example, Module 10 still needs `langchain-openai` and `langchain-pinecone` from Module 09.

Or install everything at once (matches `pyproject.toml`):
```bash
uv add langchain langchain-openai langchain-ollama langchain-pinecone langchain-community langchain-text-splitters langchain-tavily langchain-mcp-adapters langgraph langsmith ollama python-dotenv truststore httpx requests urllib3 pymupdf pypdf pdf2image streamlit langchain-chroma tiktoken
```

---

## 1. Core Architectural Concepts

### Prerequisites
* **VS Code** (Recommended Editor)
* **Git**: [Download Here](https://git-scm.com/downloads)
* **PowerShell** (Standard on Windows)

### Package & Environment Management (`uv`)
* **What it is:** `uv` is an ultra-fast, Rust-based Python project manager. It replaces the messy legacy stack of `pip`, `venv`, `pyenv`, and `requirements.txt` with a single tool that even manages downloading Python versions for you.
* **Declarative Dependency Management:** Instead of manually running `pip install X` (imperative), `uv` uses a `pyproject.toml` file to declare exactly what the project needs to run.

### The Virtual Environment (`.venv`)
* **What it is:** Python projects require "Virtual Environments." This is a hidden folder inside your project (`.venv`) that contains an isolated copy of Python.exe and all your downloaded packages. It prevents your project from conflicting with other Python projects on your computer.

### C#/Java Analogy
* **`uv`:** The exact equivalent of **NuGet + MSBuild** (C#) or **Maven/Gradle** (Java). 
* **`pyproject.toml`:** This is your **`.csproj`** or **`pom.xml`** file. It stores the metadata and package references.
* **`uv.lock`:** This is the equivalent of your **`packages.lock.json`**. It guarantees the exact same sub-dependency versions are installed on every machine.
* **`.venv` Folder:** Think of this exactly like your **`bin`** or **`obj`** folders. It is locally generated, ephemeral, and *never* committed to source control.

---

## 2. Component & Third-Party Overview

### A. The `python-dotenv` Library
* **What it is:** A lightweight package that reads key-value pairs from a local `.env` file and pushes them into Python's OS environment variables.
* **Why we use it:** To prevent hardcoding highly sensitive API keys (like `OPENAI_API_KEY`) directly into source code, which would expose them if pushed to GitHub.
* **C# Analogy:** This is the Python equivalent of using **`appsettings.Development.json`** combined with the `.NET Secret Manager` during local debugging.

---

## 🔑 Environment Variables & API Keys

All secrets are stored in a single `.env` file at the project root. A template is provided as `.env.example`.

```bash
# Copy the template
cp .env.example .env
```

Then fill in your actual keys:

| Variable | Where to Get It | Used By |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | All modules (GPT-4o) |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com/) → Settings → API Keys | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable tracing | LangSmith tracing |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) → Dashboard → API Keys | Agent search tool (Section 4-7+) |
| `PINECONE_API_KEY` | [pinecone.io](https://pinecone.io/) → API Keys (free tier) | RAG vector store (Section 9+) |
| `INDEX_NAME` | Pinecone console → your index name | RAG vector store (Section 9+) |

**Sample `.env` file:**
```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxx
INDEX_NAME=medium-blogs-embeddings-index
```

> ⚠️ **Never commit `.env` to Git.** It is already in `.gitignore`.

---

## 🦙 Ollama (Local LLM Setup)

Starting from Module 04, we use **Ollama** to run open-weights models locally (free, no API costs).

### Installation
Download from [ollama.com/download](https://ollama.com/download) and install.

### Pull a Model
```bash
# Lightweight model with tool-calling support (used in Module 04)
ollama pull qwen3:1.7b
```

### Start the Local Server
```bash
ollama serve
```
*(Leave this terminal open. Your Python code connects to `localhost:11434`.)*

### Verify
```bash
ollama run qwen3:1.7b
# Type "Hi" to test, then "/bye" to exit
```

**C# Analogy:** Think of Ollama as **Docker Desktop for LLMs**. You "pull" a model image and "run" it, exposing a local REST API that LangChain can talk to.

---

## 🌲 Pinecone (Vector Store for RAG — Section 9+)

Starting from Module 09, we use **Pinecone** as our managed vector database for RAG.

### Setup

1. **Sign up** at [pinecone.io](https://pinecone.io/) — the free tier (Starter plan) gives you 1 project, 5 indexes, and 2GB storage. More than enough for this project.
2. **Create an index** in the Pinecone console — click **"Create Index"** and fill in:
   - **Index Name**: A descriptive name like `event-tracking-pdf-index` (lowercase, hyphens only). This goes into your `.env` as `INDEX_NAME`.
   - **Dimensions**: `1536` — ⚠️ must match your embedding model. OpenAI's `text-embedding-ada-002` (LangChain's default) outputs 1536 dimensions. Wrong value = `ValueError` during ingestion.
   - **Metric**: `cosine` (default, best for text similarity)
   - **Type**: `Dense` (standard for text embeddings; "Sparse" is for keyword/BM25-style search)
   - **Capacity Mode**: `Serverless` (no infra to manage, scales automatically, pay-per-query)
   - **Cloud/Region**: `AWS` / `us-east-1` (or any region closest to you; for production use the same region as your app)
   - Click **"Create Index"** and wait ~30 seconds for status to turn green ("Ready")
3. **Generate an API key** — go to **API Keys** in the left sidebar. A default key is auto-created. Copy it (starts with `pcsk_...`).
4. **Add to your `.env`**:
   ```bash
   PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxx
   INDEX_NAME=event-tracking-pdf-index
   ```

> ⚠️ **`PINECONE_API_KEY` must be this exact name** — LangChain's Pinecone integration auto-detects it from the environment. If you rename it (e.g., `PINECONE_KEY`), LangChain won't find it and you'll get authentication errors.

> 💡 **Cannot change dimensions or metric after creation.** If you pick the wrong value, you must delete the index and recreate it. This is the #1 setup mistake.

### Verify

```python
import os
from dotenv import load_dotenv
load_dotenv()
print(f"Pinecone key loaded: {os.environ['PINECONE_API_KEY'][:8]}...")
print(f"Index name: {os.environ['INDEX_NAME']}")
```

---

## 🖥️ Visual Studio IDE Integration (C# Developer Workflow)

Since Python does not use `.sln` (Solution) files, you must configure Visual Studio IDE to understand your folder structure.

1. **Open the Project:** Open Visual Studio -> **"Open a local folder"** -> Select your repository folder.
2. **Wire up IntelliSense:** Visual Studio needs to know where your packages are installed. 
   * In the Solution Explorer, right-click your workspace folder.
   * Select **"Add/Change Python Environment"**.
   * Choose **"Add Existing Environment"**.
   * Browse to the `.venv/Scripts/python.exe` file that `uv` generated (you will do this after running `uv init` in the next step).
3. **Source Control:** Use the built-in **Git Changes** tab just like a standard C# project.

---

## 💻 Dual Examples

### 1. Example Implementation (Local Prototyping)
We use `uv init` to create a local workspace. We manually create a `.env` file on your machine to hold your OpenAI API key, and we use `uv run` to rapidly execute test scripts.

### 2. Generic / Real-World Implementation (CI/CD Pipeline)
In an enterprise Azure DevOps or GitHub Actions pipeline, you **never** use a `.env` file. Instead, your CI/CD runner executes `uv sync` (which reads the `uv.lock` file to perfectly replicate the production environment). The API keys are injected at runtime directly into the server's environment variables via Azure Key Vault or GitHub Secrets.

---

## 3. Code Implementation: Project Initialization & Test Script

Open Visual Studio's **Developer PowerShell** (View -> Terminal) and run:

```powershell
# 1. Initialize the project 
# (This creates pyproject.toml, .python-version, and a hello.py starter file)
uv init

# 2. Add our core dependencies 
# (This creates the hidden .venv folder and downloads the packages)
uv add langchain langchain-openai python-dotenv truststore
```

Next, right-click in Solution Explorer, add a new text file named `.env`, and add your API key:
```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```
*(⚠️ **CRITICAL:** Use Visual Studio's Git Changes GUI to ensure `.env` and `.venv/` are added to your `.gitignore` file before making any commits!)*

**Create a test file `test_setup.py`:**

```python
import langchain
from dotenv import load_dotenv
import os

# 1. INJECT SECRETS
# This reads the local .env file and pushes the variables into the runtime environment.
load_dotenv()

# 2. RETRIEVE SECRETS
# Safely pull the key from the OS environment dictionary.
api_key = os.getenv("OPENAI_API_KEY")

# 3. VALIDATION OUTPUT
print(f"✅ LangChain Version: {langchain.__version__}")

if api_key:
    # Print only the first 5 characters for security verification
    print(f"✅ API Key Loaded: {api_key[:5]}...")
else:
    print("❌ API Key NOT found.")
```

**Execute the script:**
```powershell
uv run test_setup.py
```

### Code Breakdown:
1. **`uv run` Execution:** In the old days, you had to manually "activate" a virtual environment before running a script. `uv run` handles this automatically. It guarantees **Ephemeral Execution**, ensuring the script runs *only* with the dependencies inside your `.venv` folder, preventing pollution from global Python packages.
2. **Line 11 (`os.getenv(...)`):** The `os` module is built into Python. `getenv` is the standard way to retrieve environment variables. If the key is missing, it returns `None` rather than throwing an exception (unlike C#'s `ConfigurationManager` which might throw a null reference).

---

## ⚠️ Production Notes (What Breaks & How to Fix It)

* **The "Works on my machine" Drift:** If you only commit `pyproject.toml` and forget to commit `uv.lock`, your production server might download LangChain `v1.3` while your local machine is on `v1.2`, breaking your app. 
  * **The Fix:** Treat `uv.lock` as a sacred source of truth. Always commit it to source control, and enforce `uv sync` in your build pipelines.
* **Secret Leakage:** Accidentally pushing your `.env` file to a public GitHub repo. OpenAI actively scans GitHub and will instantly revoke your API key if they find it.
  * **The Fix:** Ensure `.env` is in `.gitignore` on day one. In enterprise setups, utilize pre-commit hooks (like `git-secrets` or `trufflehog`) to scan code for API key regex patterns before allowing a commit to succeed.
* **Committing the `.venv` Folder:** Pushing the `.venv` folder to GitHub will upload thousands of files and ruin your repository history.
  * **The Fix:** Ensure `.venv/` is in your `.gitignore`. The `.venv` folder is machine-specific and should be re-generated on every new machine using `uv sync`.

---

## 4. Interview Q&A Anchors

**Q: Why migrate a Python backend from `pip` and `requirements.txt` to `uv` and `pyproject.toml`?**
> **A:** `requirements.txt` is an imperative, flat list that does not reliably resolve sub-dependency conflicts, leading to brittle deployments. `uv` utilizes `pyproject.toml` to create a **Declarative** setup, generating a strict lockfile (`uv.lock`). This guarantees deterministic builds across all environments. Additionally, `uv` is written in Rust, resolving dependencies orders of magnitude faster than `pip`.

**Q: What is the architectural purpose of a Python Virtual Environment (`.venv`)?**
> **A:** Unlike C# where NuGet packages are usually compiled directly into the application's build output or pulled from a global cache securely, Python packages modify the runtime environment. A virtual environment isolates the project's dependencies from the host OS and from other Python projects. It ensures that Project A can run Django 3 while Project B runs Django 4 on the same machine without conflict.

**Q: How do you handle environment variables in a local development environment versus a production Kubernetes cluster?**
> **A:** Locally, I use the `python-dotenv` library to read a `.env` file (which is strictly `.gitignore`'d) to populate `os.environ` for fast prototyping. In production, I do not deploy `.env` files. Instead, the orchestrator (like Kubernetes ConfigMaps/Secrets or Docker environment variables) securely injects the values directly into the container's OS environment at runtime. The Python application code (`os.getenv()`) remains identical in both scenarios.

---