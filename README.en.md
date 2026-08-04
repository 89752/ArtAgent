# 🎨 ArtAgent — A Conversational Agent for Western Art History

<p align="right"><a href="README.md">中文</a> · <a href="README.en.md">English</a></p>

ArtAgent is a conversational agent for Western art history, orchestrated with LangGraph. It comes with a local art library of **53,912 artwork records and 10,107 artists**, supporting factual queries, style comparisons, timelines, preference-based recommendations, image analysis, document understanding, and long-term memory across sessions.

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Required: LLM_API_KEY (chat model)
# Optional: RERANK_API_KEY (reranking), TAVILY_API_KEY (web fallback), MINERU_TOKEN (precise PDF parsing)

# 3. Start the web UI
python api.py
# Open http://127.0.0.1:7860
```

> The repository does not include data assets. Running locally requires `data/` (core library CSV, Chroma vector index, SQLite memory store) and `SemArt/` (images); if you only have the CSV, rebuild the index with `python scripts/index_core.py --csv data/core/artworks_core.csv`.

## Use Cases

| Use case | What it does |
|---|---|
| Artwork & artist lookup | Search artworks and artists by title, artist, date, or movement |
| Style comparison & evolution | Compare styles of artists or artworks; trace an artist's or movement's evolution |
| Preference-based recommendation | Recommend artists and works based on your stated aesthetic preferences |
| Visual analysis | Analyze composition, color, and brushwork |
| Expert skills | In-depth artwork analysis, document summarization, exhibition research |
| Document & spreadsheet Q&A | Upload PDF / Excel files and ask questions with page-level citations |
| Data statistics | Aggregate statistics on movements, dates, and techniques across the library |
| Memory & collections | Remember preferences across sessions; manage saved collections |

## Highlights

- **Local library first**: answers are grounded in the merged core library (Wikidata + SemArt + AIC), with web fallback (Tavily / Wikipedia / Met Museum API) only when local data is insufficient — no fabrication.
- **Intent routing + ReAct**: greetings, definitions, and arithmetic are answered directly; real-time questions go to the web; comparisons / timelines / recommendations use dedicated capability tools; it asks clarifying questions when information is missing.
- **26 callable tools**: semantic search, exact lookup, painter knowledge, image lookup & visual analysis, PDF page reading, style comparison, timelines, preference recommendations, color analysis, aggregate statistics, museum search, Wikipedia lookup, web search, memory read/write/delete, collection CRUD, and 3 expert skills (deep artwork analysis / document summarization / exhibition research).
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

After merging and deduplication: **53,912 artwork records** and **10,107 artists**, of which **39,314 have descriptions and are indexed in Chroma** (BGE-M3 multilingual vectors). The collection is primarily 8th–19th century European painting (~83%), with a small number of early-20th-century works and about 7,000 records without a year; every record carries an image reference (local image or collection URL).

## Tech Stack

| Module | Choice |
|---|---|
| Agent orchestration | LangGraph: load_memory → rewrite_split → classify → rag_gate → multi_retrieve → ReAct tool loop → reflection → save_memory |
| Retrieval | Chroma + BGE-M3 local embeddings + weighted RRF fusion + Jina Reranker v3.5 (API / local) |
| Chat model | DeepSeek / Qwen (OpenAI-compatible API with primary / backup failover) |
| Vision model | Qwen-Omni (image analysis, PDF page images) |
| Memory & sessions | SQLite (memory_items / memory_events / memory_episodes / conversations) |
| Web frontend | FastAPI + SSE + vanilla HTML/CSS/JS (no Gradio) |
| Testing | pytest fast suite (54 test files, offline, CI integrated) |

## Evaluation

```bash
python eval/agent_eval_v2.py --retrieval-n 100   # offline retrieval Recall@5
python eval/agent_eval_v2.py                     # full run (requires API quota)
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
| Retrieval Recall@5 | 88.0% (core · Jina API rerank) |

## Roadmap

**Completed**

- **Hybrid retrieval & reranking**: unified vector search over the core art library and user documents, with API / local reranking options;
- **Mature tool belt**: intent routing + 26 tools covering lookup, comparison, timelines, recommendations, visual analysis, statistics, memory, and collections;
- **Long-term memory**: explicit memory, auto-extraction, semantic conflict resolution, and cross-session profiles, visible and controllable;
- **Document understanding**: dual-channel PDF parsing (text layer + page images), spreadsheet Q&A, with page-level citations;
- **Evaluation system**: multi-dimensional test sets covering answer quality, facts, tools, multi-turn dialogue, adversarial cases, intent, routing, and retrieval;
- **Quality loop**: LLM judge scoring, rule-based checks, and state assertions; thumbs up/down feedback stored and exportable as evaluation candidates.

**Planned**

- **Multi-user & data isolation**: user accounts with session and knowledge-base isolation;
- **OpenAI-compatible API**: public streaming chat API with OpenAPI docs;
- **MCP tool integration**: import third-party tools;
- **Deployment & operations**: one-command Docker startup, CI gates, cost & latency observability.

## Project Structure

```text
api.py                 # FastAPI backend (SSE streaming; python api.py)
web/                   # service layer: LangGraph inference & rendering
static/                # vanilla frontend (index.html + app.js + app.css)
src/
├─ agent/              # LangGraph graph, intent tree, rewriting, nodes, context
├─ memory/             # long-term memory: items / extraction / conflicts / profiles / episodes
├─ tools/              # tool belt (retrieval / analysis / memory / collections / skills)
├─ retrieval/          # hybrid retrieval: Chroma + BGE + RRF + reranking
├─ ingestion/          # PDF / Excel parsing (MinerU / Qwen-VL / tables)
├─ tasks/              # document parsing task queue
└─ observability/      # run traces & /api/metrics
agent_skills/          # expert skill definitions (SKILL.md)
eval/                  # evaluation entry & test sets
tests/                 # pytest fast suite
```
