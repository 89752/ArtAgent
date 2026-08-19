# 🎨 ArtAgent — A Conversational Agent for Western Art History

<p align="right"><a href="README.md">中文</a> · <a href="README.en.md">English</a></p>

ArtAgent is a conversational agent for Western art history, orchestrated with LangGraph. It comes with a local art library of **55,000+ artwork records**, supporting factual queries, style comparisons, timelines, preference-based recommendations, image analysis, document understanding, and long-term memory across sessions.

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Required: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (chat model; no platform preset)
# Optional: VISION_MODEL / VISION_API_KEY / VISION_BASE_URL (vision; falls back to chat model config; LLM_MODEL must support image input)
# Optional: JUDGE_MODEL / JUDGE_API_KEY / JUDGE_BASE_URL (eval judge model; defaults to the chat model)
# Optional: PDF_IMAGE_EMBED_PROVIDER / MODEL / API_KEY / BASE_URL (PDF page-image embedding; DashScope by default, any OpenAI-compatible endpoint)
# Optional: RERANK_API_KEY (reranking), TAVILY_API_KEY (web fallback), MINERU_TOKEN (precise PDF parsing)
# config.yaml ships with the repo and contains no secrets: edit it directly for timeouts / concurrency / retrieval / governance; put models and keys in .env.

# Note: no default account (user/11111111) is created anymore. Bootstrap the first admin with
#   python scripts/manage_users.py create --name admin --username admin --password <strong-password> --admin
# Self-registration is available for regular accounts. Set ARTAGENT_SEED_DEFAULT_ACCOUNT=1 only for the legacy single-machine experience.

# 3. Start the web UI
python api.py
# Open http://127.0.0.1:7860
```

> The repository does not include data assets. Running locally requires `data/` (core library CSV, Chroma vector index, SQLite memory store, `data/core/images/` images); if you only have the CSV, rebuild the index with `python scripts/06_index_core.py --csv data/core/artworks_core.csv`.

## Use Cases

| Use case | What it does |
|---|---|
| Artwork & artist lookup | Search artworks and artists by title, artist, date, or movement |
| Style comparison & evolution | Compare styles of artists or artworks; trace an artist's or movement's evolution |
| Preference-based recommendation | Recommend artists and works based on your stated aesthetic preferences |
| Visual analysis | Analyze composition, color, and brushwork |
| Expert skills | Style comparison, timeline tracing, preference-based recommendation, deep artwork analysis, document summarization, exhibition research |
| Document & spreadsheet Q&A | Upload PDF / Excel files and ask questions with page-level citations |
| Data statistics | Aggregate statistics on movements, dates, and techniques across the library |
| Memory & collections | Remember preferences across sessions; manage saved collections |

## Highlights

- **Local library first**: answers are grounded in the merged core library (Wikidata + SemArt + AIC), with web fallback (Tavily / Wikipedia / Met Museum API) only when local data is insufficient — no fabrication.
- **ReAct + clarification**: asks clarifying questions when information is missing; comparisons / timelines / recommendations use expert skills; falls back to the web when retrieval comes up empty.
- **23 base tools + 6 expert skills**: base tools cover semantic search, exact lookup, painter knowledge, image lookup & visual analysis, PDF page reading, color analysis, aggregate statistics, museum search, Wikipedia lookup, web search, memory read/write/delete, collection CRUD, and parallel research; expert skills cover style comparison, timeline tracing, preference-based recommendation, deep artwork analysis, document summarization, and pre-exhibition research.
- **Long-term memory (optional enhancements)**: explicit "remember / forget" works out of the box; optional auto-extraction (`MEMORY_AUTO_EXTRACT=1`), semantic conflict resolution (`MEMORY_SMART_MERGE=1`), and cross-session user profiles (`MEMORY_PROFILE_REFRESH=1`); everything is stored in local SQLite and can be viewed or deleted per item in the memory panel.
- **Documents & spreadsheets**: upload PDF / Excel, optional MinerU precise parsing and visual reading of scanned pages; answers can cite a specific page of a document.
- **Web UI**: SSE streaming with visible reasoning steps, background generation across chats (start a new conversation before the answer finishes), stop generation, collapsible / resizable sidebar, dark mode, source citation cards, memory panel, and feedback.

## Data

The core library (`dataset_id=core`, the default runtime dataset) is a merged, normalized dataset from three sources:

| Source | Content | Count |
|---|---|---:|
| Wikidata | Structured works / artists / movements / collections | 30,041 |
| SemArt | Descriptions of 8th–19th century European paintings | 19,862 |
| Art Institute of Chicago | Open collection data | 3,085 |

> Some records come from multiple sources (e.g., the same work matches both Wikidata and SemArt).

After merging and deduplication: **55,000+ artwork records**, of which the records with descriptions are indexed in Chroma (BGE-M3 multilingual vectors; the exact count updates with core-library rebuilds). The collection is primarily 8th–19th century European painting (~83%), with a small number of early-20th-century works and about 7,000 records without a year; every record carries an image reference (local image or collection URL).

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
| Testing | pytest fast suite (11 test files / 470 offline cases, CI integrated) |

## Evaluation

```bash
python eval/agent_eval_v2.py --retrieval-n 100   # offline retrieval Recall@5
python eval/agent_eval_v2.py                     # full run (online evaluation)
pytest                                           # fast offline tests
```

Latest baseline (2026-08):

| Dimension | Result |
|---|---|
| Answer quality (30 golden cases) | avg 4.67/5 · pass rate (≥4) 93% |
| Factual accuracy | 27/31 (87%) |
| Multi-turn dialogue | 6/6 |
| Tool selection | 42/48 (88%) |
| Adversarial & safety | 8/10 |
| Intent diagnosis (soft signal) | 36/40 (90%) |
| Routing decisions | 13/15 |
| Retrieval Recall@5 | 90.0% (core · semantic+lexical hybrid · Jina API rerank) |

> Note: the table is the historical 2026-08 baseline. The current graph has converged to plain ReAct + skills; intent diagnosis now uses the rule-based `classify_intent`, and the routing dimension was removed with the old pipeline.

## Roadmap

**Completed**

- **Hybrid retrieval & reranking**: dual-channel recall (BGE-M3 semantic + FTS5/BM25 lexical) over the core library and user documents, with on-demand cross-language query translation, RRF fusion, and Jina Reranker v3.5 API reranking;
- **Mature tool belt**: ReAct tool belt + 23 base tools and 6 expert skills covering lookup, comparison, timelines, recommendations, visual analysis, statistics, memory, and collections;
- **Long-term memory**: explicit memory, auto-extraction, semantic conflict resolution, and cross-session profiles, visible and controllable;
- **Document understanding**: dual-channel PDF parsing (text layer + page images), spreadsheet Q&A, with page-level citations;
- **Evaluation system**: multi-dimensional test sets covering answer quality, facts, tools, multi-turn dialogue, adversarial cases, intent, routing, and retrieval;
- **Quality loop**: LLM judge scoring, rule-based checks, and state assertions; thumbs up/down feedback stored and exportable as evaluation candidates.
- **Multi-user & data isolation**: account registration / login / API-key auth; sessions, memory, documents, and feedback isolated per user (platform layer).

**Planned**

- **OpenAI-compatible API**: public streaming chat API with OpenAPI docs;
- **MCP tool integration**: import third-party tools;
- **Deployment & operations**: one-command Docker startup, CI gates.

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
tests/                 # pytest fast suite
```
