# PDF RAG Pipeline

Retrieval-Augmented Generation pipeline that parses any PDF, chunks and embeds it into a FAISS vector index, retrieves relevant chunks for a user query, and generates a streamed LLM response.

Supports both fully local (Ollama + sentence-transformers) and cloud (OpenAI) modes — no API keys required for local usage.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (for local mode)

### Install

**Linux / macOS**

```bash
git clone <repo-url> && cd python-rag-script
python -m venv venv
source venv/bin/activate  # fish: source venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
git clone <repo-url>; cd python-rag-script
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Choose a Provider

**OpenAI (recommended for speed and quality)**

Set your key in `.env`:

```
TEAMIFIED_OPENAI_API_KEY=sk-...
```

When set, both embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`) switch to OpenAI automatically. `gpt-4o-mini` offers the best balance of cost and quality for RAG workloads — fast, cheap, and strong at grounded Q&A.

**Local (free, no API key)**

Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and Ollama for chat. Requires downloading a model and runs on your hardware, so speed depends on your machine.

```bash
ollama pull qwen3:8b   # or any model you prefer
ollama serve            # if not already running
```

Change the Ollama model via env:

```bash
OLLAMA_MODEL=llama3:8b
```

## Usage

```bash
python run.py
```

This starts interactive mode — the PDF is loaded and indexed once, then you enter queries in a loop. You can also pass a single query as an argument:

```bash
python run.py "When did the EDSA People Power Revolution happen?"
```

Type `quit`, `exit`, empty enter, or Ctrl+C / Ctrl+D to exit interactive mode.

To use a different PDF, set `PDF_PATH` in `.env`.

### Sample Queries

```
When did the EDSA People Power Revolution happen?
Who is Jose Rizal and why is he important?
Tell me about the Spanish colonization of the Philippines.
Give me a summary of Filipino history.
What were the major turning points in Philippine history from colonization to independence?
What were the causes and lasting effects of the Philippine-American War?
How did Spanish colonization shape Philippine culture, religion, and governance?
What are the differences between EDSA I and EDSA II?
Compare the goals of the Propaganda Movement and the Katipunan.
What were the events leading up to the EDSA II revolution?
Trace the timeline of Philippine independence from Spain through the American period.
```

### Query Decomposition

Complex queries (7+ words) are automatically decomposed into sub-questions. Each sub-question is searched independently and results are merged via reciprocal rank fusion (RRF) before being passed to the LLM. This improves recall for multi-faceted questions. Short queries skip decomposition and go straight to retrieval.

### Streaming Responses

LLM responses stream token-by-token into a Rich `Live` panel — you see the answer as it's generated instead of waiting for the full response.

### Embedding Cache

Embeddings are cached to `.cache/` on first run. Subsequent runs with the same PDF and config skip the embedding step entirely. Cache invalidates automatically when the PDF, chunk size, chunk overlap, or embedding model changes.

## Architecture

```
run.py              → CLI entry point, interactive TUI loop + single-query mode
config.py           → Env vars, constants, provider detection
pdf_loader.py       → PDF text extraction (PyMuPDF) + chunking (LangChain)
embedder.py         → Embedding + FAISS index + disk cache + hybrid search + reranking
query_intent.py     → Regex-based query intent classification (factual/comparative/timeline/analytical)
query_decompose.py  → Query decomposition into sub-questions + RRF merge
llm.py              → LLM calls (Ollama / OpenAI) with intent-aware RAG prompt + streaming
```

**Data flow:**

```
query → classify intent
      → decompose? (7+ words → LLM splits into sub-questions)
      → search original + each sub-question (MMR + BM25 + reranking per query)
      → RRF merge all result lists → top-K
      → enrich with page numbers
      → stream LLM response with intent-aware system prompt
```

## Tests

```bash
pytest tests/ -v
```

Tests use real sentence-transformers for embedding and mock LLM calls — no API keys needed.

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `TEAMIFIED_OPENAI_API_KEY` | unset | Set to use OpenAI for embeddings + chat |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model for local chat |
| `PDF_PATH` | auto-detected | Path to PDF file |

### Constants (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `10` | Number of chunks retrieved per query |
| `DECOMPOSE_ENABLED` | `True` | Enable/disable query decomposition |
| `DECOMPOSE_MAX_SUB` | `4` | Max sub-questions per decomposition |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature |
