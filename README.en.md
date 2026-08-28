# 🎨 ArtAgent — A Conversational Agent for Western Art History

<p align="right"><a href="README.md">中文</a> · <a href="README.en.md">English</a></p>

ArtAgent is a LangGraph-orchestrated conversational agent for Western art history. It answers from a local core art library first (about **53,000 source records**) and supports factual queries, style comparisons, timelines, preference-aware retrieval, image analysis, document understanding, and cross-session memory.

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Configure environment variables
# macOS / Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
# Required: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (chat model; no platform preset)
# Optional: VISION_MODEL / VISION_API_KEY / VISION_BASE_URL (vision; falls back to chat model config; LLM_MODEL must support image input)
# Optional: JUDGE_MODEL / JUDGE_API_KEY / JUDGE_BASE_URL (eval judge model; defaults to the chat model)
# Optional: PDF_IMAGE_EMBED_PROVIDER / MODEL / API_KEY / BASE_URL (PDF page-image embedding; DashScope by default, any OpenAI-compatible endpoint)
# Optional: RERANK_API_KEY (reranking), TAVILY_API_KEY (web fallback), MINERU_TOKEN (precise PDF parsing)
# config.yaml ships with the repo and contains no secrets: edit it directly for timeouts / concurrency / retrieval / governance; put models and keys in .env.

# On first visit, create an account on the registration page. Keep
# ARTAGENT_SEED_DEFAULT_ACCOUNT unset for shared deployments: it creates the
# known user/11111111 demo account and is suitable only for a temporary local demo.

# 3. Start the web UI
python api.py
# Open http://127.0.0.1:7860

# Optional frontend development (React + Vite; Node 20+)
cd frontend
npm install
npm run dev
# Open http://127.0.0.1:5173; /api and /static proxy to 7860

# Production frontend build: output goes to static/dist
npm run build
```

Run traces retain sanitized metadata for 30 days by default; set `TRACE_RETENTION_DAYS` to change this. When the model provider returns `usage_metadata`, ArtAgent uses it for token accounting and otherwise falls back to a character-based estimate.

> The repository does not include data assets. Running locally requires `data/` (core-library CSV, Chroma vector index, SQLite memory store, and `data/core/images/`). If you only have the CSV, build a matching core index in your data-preparation environment before starting the service. Do not commit data assets or local build caches.

## Use Cases

| Use case | What it does |
|---|---|
| Artwork & artist lookup | Search artworks and artists by title, artist, date, or movement |
| Style comparison & evolution | Compare styles of artists or artworks; trace an artist's or movement's evolution |
| Preference-aware retrieval | Use the current request and remembered preferences to retrieve and explain matching artists, works, or movements |
| Visual analysis | Analyze composition, color, and brushwork |
| Expert skills | Style comparison, timeline tracing, deep artwork analysis, document summarization, exhibition research |
| Document & spreadsheet Q&A | Upload PDF / Excel files and ask questions with page-level citations |
| Data statistics | Aggregate statistics on movements, dates, and techniques across the library |
| Memory & collections | Remember preferences across sessions; manage saved collections |

## Data

The core library (`dataset_id=core`, the default runtime dataset) is a merged, normalized dataset from three sources:

| Source | Content | Count |
|---|---|---:|
| Wikidata | Structured works / artists / movements / collections | 30,041 |
| SemArt | Descriptions of 8th–19th century European paintings | 19,862 |
| Art Institute of Chicago | Open collection data | 3,085 |

> Some records come from multiple sources (e.g., the same work matches both Wikidata and SemArt).

## Tech Stack

| Module | Choice |
|---|---|
| Agent orchestration | LangGraph: load_memory → ask_user → ReAct tool loop ⇄ tools → reflection → save_memory |
| Retrieval | BGE-M3 semantic vectors + lexical channel (core FTS5 / PDF BM25, on-demand translation) → weighted RRF fusion → Jina Reranker v3.5 API |
| Chat model | OpenAI-compatible API (no platform preset; configured via config.yaml / environment variables) |
| Vision model | OpenAI-compatible vision model (independently configurable; falls back to chat model config) |
| User system | Account registration / login / API-key auth; sessions, memory, documents, and feedback isolated per user |
| Memory & sessions | SQLite (memory items/events, conversations, rolling summaries, user documents, collections, feedback, extraction metrics) |
| Web frontend | React 19 + TypeScript + Vite (SSE streaming) |
| Testing & delivery | pytest offline regression, deterministic evaluation gates, and frontend type-check/build in GitHub Actions |

## Evaluation

```bash
pytest -q                                      # offline regression; does not consume model quota
python eval/memory_reliability_eval.py         # memory write, recall, conflict, and forgetting checks
python eval/agent_eval_v2.py --retrieval-n 100 # core-library offline Recall@5
python eval/agent_eval_v2.py                   # complete core online evaluation
```

## Project Structure

```text
api.py                 # FastAPI backend (SSE streaming; python api.py)
web/                   # service layer: LangGraph inference & rendering, painting-analysis SSE
frontend/              # React frontend (Vite + TypeScript; build output goes to static/dist)
static/                # static assets (svg) and frontend build output (dist/)
src/
├─ agent/              # LangGraph graph, nodes, context building (ReAct + clarification + reflection)
├─ memory/             # long-term memory: items / extraction / conflicts / profiles / summaries / collections / feedback
├─ tools/              # tool belt (retrieval / analysis / memory / collections / skills)
├─ retrieval/          # hybrid retrieval: semantic + lexical channels + RRF + reranking
├─ ingestion/          # PDF / Excel parsing (MinerU / page-image vision / tables)
├─ analysis/           # painting-analysis engine (visual metrics / validation / report storage)
├─ data/               # document status storage and the shared SQLite connection layer
├─ platform/           # users / auth / API keys
├─ skills/             # expert skill loading and activation
├─ subagents/          # delegate_task parallel sub-agent executor
├─ utils/              # config / LLM clients / logging / tool-execution governance
├─ tasks/              # document parsing task queue
└─ observability/      # run traces & /api/metrics
agent_skills/          # expert skill definitions (SKILL.md)
eval/                  # evaluation entry & test sets
tests/                 # pytest tests
```
