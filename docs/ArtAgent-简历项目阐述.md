# ArtAgent 简历项目阐述

> 项目周期：2026.07 – 2026.08（个人项目，如为团队项目请自行调整为"参与/负责 XX 模块"）

## 算法岗版（推荐）

### 一句话版

独立设计并开发 ArtAgent——基于 LangGraph 的西方艺术问答 Agent：构建"意图分类 → 任务管线/ReAct 工具循环 → 反思兜底"的混合编排，自建 BGE 向量召回 + 双路精排的 RAG 链路与多维度评测体系；意图分类准确率 96.0%（Macro-F1 0.962），检索 Recall@5 达 88%（API 精排）/ 85%（本地精排）。

### 项目经历（4 条）

1. 负责 Agent 意图理解与路由：构建 203 条四类意图标注集（含跨领域负样本与"推荐"易混淆样本），以 temperature=0 的确定性链路评测，意图分类准确率 96.0%（Macro-F1 0.962），并基于混淆矩阵与误分类清单完成错误分析。
2. 负责检索质量优化：搭建"BGE 向量召回 + 精排"两阶段链路，对比 qwen3-rerank（API）与 bge-reranker-v2-m3（本地零额度）精排后端并做 embedding 模型 A/B；在 100 条自动标注、seed=42 可复现样本上 Recall@5 达 88%（API）/ 85%（本地）。
3. 设计多维度 Agent 评测体系：覆盖意图分类、检索 Recall@5、事实准确率（60 条 core 自动可验证题集，100%）、工具选择（30 条含"无工具"负样本）与行为化用例（8 类场景）五类指标，全部在真实组件链路上运行，支持 nightly 回归与 >2pp 波动告警。
4. 设计混合编排与记忆推理：意图路由 + 显式任务管线 + ReAct 工具循环 + 反思联网兜底，使规划/检索/反思各环节可观测、可调试；将用户偏好经 LLM 推理为结构化风格特征后再做向量检索，实现"理解 → 推理 → 检索"而非关键词匹配；落地 P50/P95 延迟、单轮成本、工具分布、兜底率观测与模型主备降级（额度耗尽场景实战验证）。

### 英文版

**ArtAgent — LangGraph-based Western Art Q&A Agent** *(Personal Project, 2026.07–2026.08)*

- Owned intent understanding & routing: built a 203-query four-class intent set with cross-domain negatives and hard "recommendation" samples; with temperature=0 deterministic evaluation, achieved 96.0% accuracy (Macro-F1 0.962), including confusion-matrix and error analysis.
- Optimized retrieval quality: built a two-stage BGE vector-recall + rerank pipeline, comparing qwen3-rerank (API) vs. bge-reranker-v2-m3 (local, zero quota) backends with embedding-model A/B tests; reached 88% (API) / 85% (local) Recall@5 on a reproducible 100-sample auto-labeled benchmark (seed=42).
- Designed a five-metric agent evaluation suite — intent classification, Recall@5, fact accuracy (60 core-verifiable queries, 100%), tool selection (30 queries incl. no-tool negatives), and behavioral scenarios (8 categories) — all executed on the real component pipeline with nightly regression and >2pp drift alerts.
- Designed hybrid orchestration & memory reasoning: intent-routed explicit pipelines + ReAct tool loop + reflection with web fallback, keeping planning/retrieval/reflection observable and debuggable; user preferences are reasoned into structured style features before vector retrieval ("understand → reason → retrieve"); added P50/P95 latency, per-turn cost, and tool-distribution observability, plus model primary/backup failover validated in a real quota-exhaustion incident.

---

## 通用/全栈版（备选）

### 一句话版

独立设计并开发 ArtAgent——基于 SemArt 数据集的西方艺术智能问答 Agent：以 LangGraph 实现"意图路由 + 四类任务管线 + ReAct 工具循环 + 反思联网兜底"的混合编排，集成 Chroma 向量检索、SQLite 长期记忆与 FastAPI/SSE 流式前端；意图分类准确率 96%，检索 Recall@5 达 88%。

### 中文完整版

1. 独立完成 Agent 核心架构：基于 LangGraph 设计"意图分类 → 分支管线（对比 / 时间线 / 推荐 / 通用问答）→ 反思 → 联网兜底"的混合编排，任务结构化时走显式工作流、开放问题时走 ReAct 工具循环，使每步决策可观测、可调试；意图分类准确率达 96%。
2. 构建 RAG 检索与工具链：基于 Chroma + BGE Embedding 对 1,384 幅 8–19 世纪欧洲绘画建立语义索引，封装语义检索、精确查询、画家知识、图像理解、联网搜索 5 类工具；在 100 条可复现样本（seed=42）上 Recall@5 达 88%。
3. 实现跨会话长期记忆与完整产品前端：以 SQLite + LangGraph MemorySaver 存储用户偏好，打通"偏好 → 推理结构化风格特征 → 向量检索推荐"链路并支持偏好管理；自建 FastAPI + SSE 流式双栏 Web 界面，逐节点展示推理链路、交互式来源引用与多轮对话。
4. 完成工程化与可观测性建设：统一工具超时/重试、任务队列、限流、模型主备降级与 HTML 白名单消毒；每轮对话落库 `agent_runs`，提供 P50/P95 延迟、成本、工具分布、反思兜底率等指标接口，配套回归脚本与 Docker 部署。

### 英文版

**ArtAgent — LangGraph-based Western Art Q&A Agent** *(Personal Project, 2026.07–2026.08)*

- Designed and built a hybrid-orchestration agent over the SemArt dataset (1,384 European paintings, 8th–19th c.): LangGraph routes intents into comparison / timeline / recommendation pipelines, with a ReAct tool loop for open-ended questions and a reflection node that triggers web-search fallback; achieved 96% intent-classification accuracy (50-sample baseline, v2 203-sample set pending).
- Built the RAG stack with Chroma + BGE embeddings and 5 tools (semantic search, exact lookup, painter knowledge, image analysis, web search), reaching 88% Recall@5 on a reproducible 100-sample benchmark (seed=42).
- Implemented SQLite + MemorySaver cross-session user memory and a self-built FastAPI + SSE streaming UI with per-node reasoning traces and interactive citations; shipped production hardening (task queueing, rate limiting, model failover, HTML sanitization), metrics endpoints (P50/P95 latency, cost, tool distribution), regression tests, and Docker deployment.

## 备注

数字取自仓库最新评测结果；若你复测后实际数字有变化，直接替换正文即可。"独立设计开发"请确认为个人独立完成。
