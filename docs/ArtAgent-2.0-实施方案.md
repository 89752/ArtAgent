# ArtAgent 2.0 实施方案（完善版 · 开发用）

> 本文档 = 原《ArtAgent-2.0-设计方案》+ 2026-07-31 代码级评估的 8 项补充 + Stage 1/2 实施记录。
> 是 Stage 3–7 后续开发的**唯一事实来源**；每个新会话开工前先读「§1 当前状态」和「§13 陷阱速查」。
>
> 原方案中的架构判断（三层结构、确定性编排 vs 真工具调用、collection 级隔离不做物理合并、
> 页级自适应路由、结构化/非结构化检索统一抽象）经代码核验全部成立，本文不再重复论证，只记结论。

---

## 0. 开发环境基线

| 项 | 值 | 注意 |
|---|---|---|
| Python 环境 | `C:\Users\86188\anaconda3\envs\artagent\python.exe`（3.10.20） | 旧 `llm` 环境已弃用；PATH 里的 python 均无项目依赖 |
| 关键依赖 | langchain 1.3.14 / langgraph 1.2.10 / chromadb 1.5.9 / sentence-transformers 5.6.1 / pyarrow 25.0.0 | **勿随意安装 HF `datasets` 包**：旧环境曾因 sklearn+datasets+pyarrow 三方 DLL 冲突导致 `import sentence_transformers` 段错误；新环境未装 datasets 才正常。若必须安装，装完先测该导入 |
| LLM/视觉/检索 API | DashScope（`.env` 中 `DEEPSEEK_API_KEY` 一把 key 通用） | 视觉模型 `qwen3.5-omni-plus`；联网兜底 Tavily |
| 向量模型 | BGE `bge-small-en-v1.5`，本地缓存 | `api.py` 已设 `HF_HUB_OFFLINE=1`；测试脚本不设也可跑（缓存命中） |
| 数据/索引 | SemArt（21,384 幅）+ `data/index/chroma/`（217MB，collections: `semart` + 遗留 `user_pdfs`） | `user_pdfs` 与 `data/uploads/` 是旧 PDF 实验遗产，**Stage 3/6 处置** |

**验证命令（全量约 25–35 分钟，走真实 API）：**

```bash
PY="C:/Users/86188/anaconda3/envs/artagent/python.exe"
$PY tests/test_access.py       # 纯单测，无 LLM，秒级
$PY tests/test_tools.py        # 工具冒烟（真实 API）
$PY tests/test_multi_tool.py   # 多工具链
$PY tests/test_pipelines.py    # 四分支端到端
$PY eval/run_eval.py           # 意图分类 + Recall@5（基线 96.0% / 64.0%；64.0% 为 n=25 口径，默认 n=20 为 70.0%，两者非回归关系）
```

---

## 1. 当前状态：Stage 1 / Stage 2 已完成 ✅

> Stage 1：commits `a08bda1`、`1aaf48c`；Stage 2：见 §1.4。

### 1.1 已落地的改动（Stage 1）

- **数据访问层 `src/data/access.py`**：`fuzzy_match`（精确→去冠词→分词包含 三级递进）、`row_to_artwork_dict`（兼容 DataFrame 行与 Chroma metadata）、`format_evidence_block`（模板化 + 空字段清理）、`EVIDENCE_SNIPPET_LEN=200`。配套 `tests/test_access.py` 17 个纯单测。
- **工具 7 → 5**：`semantic_search` / `exact_lookup` / `query_painter_knowledge`（去 LLM 化，返回结构化统计）/ `image_lookup`（吸收 analyze_image，`analyze=True` 触发视觉分析）/ `web_search`。`compare_artwork_styles` 已删除。`GENERAL_TOOLS` 与 `SYSTEM_PROMPT` 已同步。
- **管线去重**：`timeline_gather_periods` 不再 iterrows 拼第二遍证据；comparison/recommendation/web_fallback 的证据格式化统一走 `format_evidence_block`。
- **隐藏 bug 修复**：旧 `get_by_author` 整串 contains 匹配遇上 SemArt 的倒序 AUTHOR（`"GOGH, Vincent van"`）永远查空，`fuzzy_match` 分词匹配根治。
- **清理**：faiss 残留索引（49MB）、pdf_loader.py 孤儿代码、README/requirements 的 Gradio 失效引用；`.gitignore` 的 `data/` 改为 `/data/`（原规则会把 `src/data/` 源码也忽略）。

### 1.2 保留未动的遗产（Stage 3/6 处理）

- `data/uploads/`：12 个旧会话目录 + 真实 PDF（用户上传过的文件，勿直接删，Stage 6 文档管理体系建立时迁移或经用户确认后清理）。
- Chroma `user_pdfs` 集合：旧 chunk 策略的向量，Stage 3 新入库体系上线时废弃重建。

### 1.3 测试约定（全 Stage 通用）

- 每个新模块配**纯单测**（参照 `tests/test_access.py`：不加载数据集、不调 LLM、不联网、断言式 + `__main__` 直跑）。
- 验收标准用**无显著回归**（eval 指标波动 ≤ 2 个百分点），不用"完全一致"——意图分类走 LLM，存在天然波动。

### 1.4 Stage 2 实施记录（RAG 抽象层）✅

**已落地的改动：**

- **`src/retrieval/` 新包**：
  - `base.py`：`RetrievalResult(content/source/score/metadata/image_refs)`，source 枚举含 `semart/user_table/user_pdf_text/user_pdf_image/met_museum/rijksmuseum`（后两者 Stage 7 预留）；`BaseRetriever` Protocol（`search(query, top_k, filters)`）。
  - `structured_retriever.py`：`TableSchema`（entity_col/group_axis_col/description_col/image_col，`supports_timeline`/`supports_recommendation` 自动推出）+ `SEMART_SCHEMA` 配置化；`StructuredTableRetriever` 提供 `group_by_axis`（timeline）、`exclude_by_entity`/`exclude_from_results`（recommendation）、`search`；数据源注册表 `register_structured_dataset`/`get_structured_retriever`，SemArt 懒注册为 `dataset_id="semart"`。
  - `hybrid.py`：`HybridRetriever.search(query, top_k, sources, dataset_id)`——多源扇出 → RRF（k=60）按源内排名融合 → page_id/doc_id 去重；BGE/Chroma 单例从 `tools/retrieval.py` 迁入（`get_chroma_collection(name)` 参数化，为 Stage 3 多 collection 预留）；`get_hybrid_retriever()` 全局单例自动注册 SemArt。
- **集成点**（§3.2 三项全部落实）：`AgentState.dataset_id="semart"`；`web/service.py` 重置清单同步；`semantic_search` 改走 HybridRetriever 且返回形状不变（`title/author/date/technique/school/timeframe/image_file/description_snippet/relevance_score`）。
- **节点改造**：`timeline_gather_periods` 改用 `get_structured_retriever(state.dataset_id).group_by_axis(subject)`；`recommendation_feature_search` 的排除逻辑改用 `exclude_from_results`；路由层 `_route_by_intent` 加能力开关（只读 schema，SemArt 恒 True 不触发降级）。

**实施中的关键决策（对原方案的细化，后续 Stage 需知晓）：**

1. **注册全懒加载**：注册只落 schema + 若干 loader，df/Chroma/BGE 模型首次实际使用才解析——能力开关挂在意图路由上，不能让每次分类都付出数据集加载代价。
2. **`exclude_by_entity` vs `exclude_from_results`**：原方案签名是前者（返回 DataFrame），但 recommendation 的排除实际发生在 `semantic_search` 的结果字典上（向量检索走 Chroma 不过 DataFrame）。两者都实现、共享同一套分词逻辑，节点用后者；行为与 Stage 1 内联逻辑逐字一致（长度 >2 的词小写化做包含匹配）。
3. **RRF 只定序不改分**：各源原生 `score`（SemArt 为 `1 - distance`）原样保留，`relevance_score` 形状不变；Python sort 稳定，单源时 RRF 严格保持原顺序——这是 Recall@5 逐位持平的关键。
4. **去重只作用于带 page_id/doc_id 的结果**：SemArt 行无此二键，不参与去重；Stage 3 双路线页面共享 page_id 时该逻辑才生效。
5. **`StructuredTableRetriever.search` 双路径**：挂了向量集合走 BGE 向量检索（SemArt）；无索引表走 `fuzzy_match` 实体列 → 描述列包含的兜底路径（Stage 5 预览），支持 `{列: 值}` 等值 filters。

**验收数据（2026-08-01 全量跑完）：**

- 纯单测 53 个全绿：`test_structured_retriever.py` 23 + `test_hybrid.py` 13 + `test_access.py` 17（秒级，无 LLM）。
- `tests/test_tools.py` / `test_pipelines.py` / `test_multi_tool.py` 全绿，四分支与多工具链行为不回归。
- eval 意图分类 **98.0%**（49/50，基线 96.0%，波动在 ±2pp 内）。
- eval Recall@5：**新旧路径 25 条 query 逐位对比 0 不一致**（`tests/verify_recall_parity.py`，标题/顺序/条数完全相同）；**n=25 复现基线恰为 64.0%**（16/25）。注意口径：基线 64.0% 来自 n=25，`run_eval.py` 默认 `--retrieval-n 20` 跑出 70.0%（14/20）是同一样本种子下的不同口径，非行为变化。`verify_recall_parity.py` 留作永久校验工具，Stage 3/4 改动检索层后应复跑。

---

## 2. 目标架构（三层）

```
Agent 编排层（graph.py，路由基本不变）
  routing → {comparison / timeline / recommendation 确定性管线, general ReAct 循环}
        │ general 分支 bind_tools 接入真工具调用
能力层（主战场）
  ├─ RAG 能力（HybridRetriever，RRF 融合 + page_id/doc_id 去重）
  │   ├─ 结构化：StructuredTableRetriever（SemArt 首个注册实例 + 用户表格）
  │   └─ 非结构化：UserDocTextRetriever（BGE 空间）+ UserDocImageRetriever（DashScope 多模态空间）
  ├─ 外部实时工具（仅 general 分支）：museum_search（走 RAG 融合）/
  │   nearby_venues / wiki_lookup / color_analysis（独立 @tool，不参与融合）
  └─ web_search（保留兜底）
基础设施层：数据访问层 access.py（已建成）/ documents_store / 观测 traced
```

**术语原则**：只有 general 分支是真工具调用（LLM 自主决策）；三条管线是确定性编排（检索时机/参数写死），这是刻意保留的优点，不要在改造中泛化掉。所有新工具只返回结构化数据，不在内部再调 LLM 包装回答。

---

## 3. Stage 2：RAG 抽象层 ✅（已完成，实施记录见 §1.4）

**目标**：SemArt 从硬编码变成"第一个注册的数据源"；`timeline`/`recommendation` 依赖 `TableSchema` 抽象而非 SemArt 字段名。

### 3.1 交付物

- `src/retrieval/base.py`：`RetrievalResult(content, source, score, metadata, image_refs)`，`source` 枚举预留 `met_museum`/`rijksmuseum`；`BaseRetriever`（Protocol）。
- `src/retrieval/structured_retriever.py`：`TableSchema`（entity_col/group_axis_col/description_col/image_col，supports_timeline/supports_recommendation 自动推出）+ `StructuredTableRetriever`（group_by_axis 给 timeline、exclude_by_entity 给 recommendation、search 走 `access.fuzzy_match`）。
- `SEMART_SCHEMA` 配置 + 启动时注册 `dataset_id="semart"`。
- `src/retrieval/hybrid.py`：`HybridRetriever.search(query, top_k, sources=None, dataset_id=None)`——BGE 编码 query 查 BGE 空间 →（Stage 3 起另有 DashScope 多模态一路）→ RRF 融合 → page_id/doc_id 去重。其余检索器可先空实现。
- 路由层能力开关：进 `timeline`/`recommendation` 前查 `schema.supports_*`，不支持降级 general（Stage 2 恒 True 不触发）。

### 3.2 集成点（原方案未写，必须做）

1. **`AgentState` 加 `dataset_id: str = "semart"` 字段**。
2. **`web/service.py` 的 `stream_answer` 状态初始化清单同步加 `dataset_id`**（该函数显式列出每轮重置的所有标量字段，漏加会串味）。
3. **`semantic_search` 工具改为调用 HybridRetriever，但返回形状必须保持现状**（`title/author/date/technique/school/timeframe/image_file/description_snippet`）：`web/service.py::_parse_artworks_from_messages` 按 `title/author/date/image_file` 解析 ToolMessage，comparison/recommendation 节点消费 `description_snippet`，缺字段会静默断 UI 配图。

### 3.3 验收

- `structured_retriever` 纯单测（构造小 DataFrame，不依赖 SemArt）。
- `test_pipelines.py` 全绿；eval 无显著回归（检索 Recall@5 应恰好 64.0%——固定种子、本地 BGE，无 LLM 波动借口）。
- ✅ 已验收（§1.4）：**Recall@5 基线口径为 n=25（16/25=64.0%）**；`run_eval.py` 默认 n=20 跑出 70.0% 是同一 seed=42 下的不同样本量，不是回归。判据应是"新旧路径逐位一致"，用 `tests/verify_recall_parity.py` 验证。

---

## 4. Stage 3：PDF 解析与入库

**最大风险 Stage，估 4–6 天。**

### 4.1 页级自适应路由（零模型调用）

PyMuPDF 逐页采集信号（**先把 `PyMuPDF` 加回 requirements.txt**，旧 pdf_loader.py 遗留过 fitz 用法参考，文件已删，git 历史不可考）：

| 信号 | 用途 |
|---|---|
| `doc.metadata` producer/creator | 文档级先验（Office 系 vs 扫描仪系） |
| `page.get_text()` 字符数 | 核心信号：>200 文字路线候选，<50 多模态候选 |
| 图片面积占比 | >80% 多模态，<30% 文字路线 |
| `page.get_fonts()` 是否为空 | 扫描页佐证 |
| 公式符号密度（∫∑∏√±≤≥ + 上下标） | 高 → 强制 MinerU，不接受降级 |

判定：`text>200 且 img<30%` → 文字路线；`text<50 或 img>80%` → 多模态；中间地带 → **双路线**（都入库，共享 page_id，检索端去重；不纠结阈值精度）。阈值为起始值，须用真实 PDF 人工核对后微调。

### 4.2 两条路线

- **文字路线：MinerU 为主**（v3.x，Apache 2.0 系自定义协议，Windows/纯 CPU 可跑 pipeline 后端，首次下载约 4GB 模型，国内设 `MINERU_MODEL_SOURCE=modelscope`）。pdfplumber 仅兜底，公式密集页不得降级（宁可转多模态整页）。⚠️ **MinerU 官方 known issue：艺术图册/漫画解析不佳**——与本项目领域直接相关，双路线设计可对冲，但测试素材必须含真实画册验证（见 §12）。
- **多模态路线**：整页渲染 → DashScope 多模态 embedding → 独立 Chroma collection（与 BGE 空间隔离，维度/语义分布不同不可混库）。**选型建议默认 `tongyi-embedding-vision-plus`（1152 维，图片 $0.09/M token），`qwen3-vl-embedding`（2560 维，$0.258/M，约 3 倍价，单请求限 1 图 ≤5MB）做对照组**，小规模测试后定。
- **生成阶段**：命中整页图的页面图片喂 Qwen-VL 读图作答——这是成本大头，评测时单列"命中图片路线的平均生成开销"。

### 4.3 Chunk 策略（文字路线）

基于 MinerU 语义块（text/table/equation/image，带 page_idx/bbox）做归并/拆分，**不对纯文本套滑动窗口**：
1. text 块：短段向下归并（下限 ~150–200 字符），长小节才拆（10–15% 重叠），上限 300–500 字符是"该不该拆"的判断阈值非硬指标；metadata 带标题层级。
2. table 块（HTML）：整块入库不拆，`block_type=table`。
3. equation 块（LaTeX）：整块 + 前后短上下文，独立小 chunk。
4. MinerU 抠出的内嵌小图：Qwen-Omni 生成 caption → 入 BGE 空间，`block_type=image_caption`，保留图片路径。
5. 统一 metadata：`doc_id / page_id / block_type / section / kb_id`（kb_id 为 Phase 2 预留）。

### 4.4 入库流程（Phase 1 修订版）

**修订（评估补充）**：原方案"Phase 1 接受同步阻塞"在 MinerU CPU 下不可行（100 页约 5–10 分钟，浏览器/uvicorn 会超时）。**Phase 1 即采用 `fastapi.BackgroundTasks` + 状态轮询**（成本极低），或先限制上传页数/大小。流程：上传 → 存 `uploads/{kb_id}/{doc_id}.pdf` → 记录元数据 → 后台解析（逐页路由 → 两路入库）→ 写解析结果元数据（页数/路由分布/chunk 数/耗时/状态）→ SSE/轮询可见进度。

### 4.5 遗产处置（评估补充）

新体系上线时：废弃 Chroma `user_pdfs` 旧集合（删或留作只读对照）；`data/uploads/` 旧会话目录里的 PDF 经用户确认后迁移或清理。

### 4.6 验收

- 页级路由纯单测（构造 mock 页面对象或小型 fixture PDF）。
- 真实测试 PDF 跑通：路由分布合理、双路线页面检索不重复引用、文字 chunk 命中可用。

---

## 5. Stage 4：检索质量层

1. **上下文头（context header）**：向量化前给 PDF 文字 chunk 拼接 `[文档 | 章节 | 实体]` 头（只影响向量与展示，不改存储）；SemArt/结构化表不加。
2. **Rerank**：RRF 粗排 → top 15–20 → **qwen3-rerank** 精排 → top 5–8。已核实：DashScope OpenAI 兼容端点 `POST https://dashscope.aliyuncs.com/compatible-api/v1/reranks`，单次 ≤500 文档、单文档 ≤4000 token，支持 `instruct` 参数；gte-rerank-v2 已下线勿用。彩蛋：`qwen3-vl-rerank` 可给多模态整页图做精排，可选。
3. **结果相关性校正通用化**：把 `recommendation` 的 `rec_filter` 思路提炼为 HybridRetriever 后的通用轻量 LLM 过滤步骤，所有分支受益。
4. 不做（Phase 2 再说）：BM25/关键词混合、结果压缩；永不做：向量数值量化。

**验收**：reranker/relevance_filter 纯单测（mock 候选集）；eval Recall@5 应较 64.0% 基线**提升**（这是本 Stage 的价值度量）。

---

## 6. Stage 5：结构化表格上传

依托 Stage 2 的 `StructuredTableRetriever` + `TableSchema`：

1. **文件类型路由**（零模型调用）：.csv/.xlsx/.xls → 表格通道；.pdf → Stage 3 通道。
2. **Schema 推断必须有人工确认**：LLM 看表头+前几行猜列角色 → 用户确认/纠正后生效。原因：猜错 entity_col 会让 recommendation 的排除逻辑**静默出错**（排除了错误的行而不报错），比明显报错更危险。
3. **能力开关真正生效**：无时间/分类列的表，`timeline` 意图降级 general（Stage 2 预留的判断在此启用）。
4. 明确不做：MinerU 表格块升格为结构化表（HTML→DataFrame 可靠性不足，后续阶段再议）。

**验收**：schema 推断单测；准备两份测试表格（字段完整支持 timeline+recommendation 的 vs 不支持的纯列表）验证推断与降级。

---

## 7. Stage 6：文档/数据源生命周期管理（Phase 1 基线，非可选）

1. **`src/data/documents_store.py`（SQLite `documents` 表）**：上传时间/大小/原始文件名/页数/路由分布/解析状态(pending/processing/done/failed)/耗时/chunk 数。Stage 3/5 的解析元数据在此落库。路由分布与 chunk 数是排查"为什么检索不到"的第一手诊断信息。
2. **文件库列表页（最小能力）**：列表（文件名/时间/大小/状态/页数）+ **删除**。
3. **删除必须级联**：删文件时连带清理 Chroma 中该 `doc_id` 的所有向量（两个 collection 都要清），不是只删文件——易漏，列为验收硬项。
4. Phase 2 再做：重新解析（阈值调整后重跑）、解析详情页（逐页路由分布）。

---

## 8. Stage 7：能力层扩展（须在 Stage 1 之后；可与 3–6 部分并行）

顺序与要点：

1. **`color_analysis`（最先做）**：本地纯算法（K-means 主色调、色彩直方图、对称性/构图网格），输出结构化数值。**实现用 numpy/PIL（`Image.quantize` 或手写 K-means），不引入 scikit-learn 依赖**。复用 `image_lookup` 的图片定位。
2. **`museum_search`**：`MuseumAPIRetriever` 接入 HybridRetriever（`RetrievalResult.source` 已预留）。**Met API：免 key、限速 80 req/s、CC0 仅限公共领域作品**（`/public/collection/v1/search` 返回 objectID 列表，再逐件取 metadata，top_k 小所以 N+1 请求可接受）。Rijksmuseum：需申请免费 key（用户自备）。RRF 按源内排名融合，无需 Met 提供相似度分数。
3. **`nearby_venues`**：高德 Web 服务 POI（免费 key 用户自备）。只管"在哪"；**"现在展什么"无统一免费 API，退回 web_search 接力**——文档和工具描述里写清这个分工。
4. **`wiki_lookup`**：Wikidata/Wikipedia 免费接口，优先级最低。
5. **工具选择准确率评估集**（Phase 1 末期雏形）：参照 `eval/intent_testset.json` 做"该调哪个工具"的人工标注集，衡量 8–9 个工具下的误选率。

明确不做：拍卖价格数据（付费）、日历/天气（凑数）、票务平台（无公开 API）、Google Arts & Culture（权限不确定）。

**新工具纪律**（Stage 1 立的原则）：只返回结构化数据，不在内部调 LLM 包装回答；`nearby_venues`/`wiki_lookup`/`color_analysis` 不实现 `BaseRetriever`，不参与自动融合。

---

## 9. Web UI（拆进各 Stage，**不是独立的第 8 步**）

原落地顺序把 Web UI 列为第 8 步整体推进——不可行，各 Stage 验收依赖各自的 UI 切片：

| 随哪个 Stage | UI 切片 |
|---|---|
| Stage 3 | 上传入口 + 入库进度展示（BackgroundTasks 状态轮询） |
| Stage 5 | schema 推断确认/纠正交互（文本框确认 or 字段映射 UI，随本 Stage 设计定稿） |
| Stage 6 | 文件库列表页（列表+删除） |
| Stage 3/6 | 回答中的来源标签 UI（`RetrievalResult.source` → 前端徽标） |

前端基座沿用现有 `static/`（原生 HTML/CSS/JS + SSE），不引框架。

---

## 10. Phase 2 摘要（Stage 1–7 完成后）

多租户（kb_id+user_id，Chroma metadata filter 逻辑隔离）→ 异步任务化深化 → 可观测性扩展（各数据源贡献条数/图片路线生成开销/工具调用分布，复用 `traced`）→ 评测集扩展（PDF Recall@K、跨源路由、页级路由准确率、图文关联、工具选择、schema 推断）→ Docker Compose（api + chroma 卷）。

---

## 11. 验收规约（每 Stage 收尾必做）

1. 新模块纯单测全绿（无 LLM、秒级）。
2. `tests/test_tools.py` + `test_pipelines.py` + `test_multi_tool.py` 全绿。
3. `eval/run_eval.py`：意图分类 ≥ 94%（基线 96.0%，允许 ≤2pp 波动）、Recall@5 对比基线 64.0%（Stage 2 应持平，Stage 4 应提升）。
4. 涉及状态的改动：检查 `web/service.py` 的重置清单与 `_chain_detail` 节点标签是否需要同步。
5. commit 信息注明 Stage 与验收数据。

---

## 12. 开放问题（原 §13 更新版）

| 项 | 状态 |
|---|---|
| Met API 接入细节 | ✅ 已定：免 key、80 req/s、CC0 仅公共领域 |
| 多模态 embedding 选型 | ✅ 已定策略：默认 tongyi-embedding-vision-plus（便宜 3 倍），qwen3-vl-embedding 对照，小规模测试后定稿 |
| 测试 PDF 素材 | ⏳ 待准备：2–3 份美术图文 PDF（**至少 1 份真实画册**，验 MinerU 短板）+ 1 份含公式技术 PDF |
| 页级路由阈值 | ⏳ 200字符/30%/80% 为起始值，真实 PDF 跑后微调 |
| 测试表格素材 | ⏳ 待准备：支持 timeline+recommendation 的完整表 ×1、不支持的纯列表 ×1 |
| schema 确认交互形式 | ⏳ 随 Stage 5 Web 设计定稿 |
| Rijksmuseum / 高德 key | ⏳ 用户自备免费 key（Met/Wikidata 无需 key） |

---

## 13. 陷阱速查（新会话开工前必读）

1. **SemArt AUTHOR 是倒序大写格式**（`"GOGH, Vincent van"`）：任何按作者名的匹配必须走 `access.fuzzy_match`，禁止自己写 contains——整串匹配必查空。
2. **Chroma 现有两个 collection**：`semart`（正式）+ `user_pdfs`（旧实验遗产，Stage 3 处置）。`_get_collection()` 目前硬编码取 `semart`。
3. **`.gitignore` 的 `/data/` 只忽略根目录数据**；历史上写成 `data/` 曾把 `src/data/` 源码静默排除出版本管理，勿回退。
4. **环境**：用 `anaconda3/envs/artagent/python.exe`；勿装 `datasets` 包（会重建 DLL 冲突链）。
5. **`web/service.py` 的 `stream_answer` 有显式的每轮状态重置清单**：`AgentState` 加字段必须同步加进清单，否则跨轮串味。
6. **工具返回形状的隐形消费者**：`_parse_artworks_from_messages` 认 `title/author/date/image_file`；各合成节点认 `description_snippet`。改 RetrievalResult/工具输出时对齐。
7. **测试全是真实 API 调用**：全量验证约 25–35 分钟且有 API 成本；开发期优先跑纯单测，验收再跑全量。
8. **工具内不许藏 LLM 调用**：结构化数据进、结构化数据出，组织语言是 general_agent 的事。
9. **BGE/Chroma 单例已迁到 `src/retrieval/hybrid.py`**（`get_chroma_collection(name)` / `get_bge_embed_fn()`）；`tools/retrieval.py` 不再有 `_get_collection`/`_get_embedding_model`，Stage 3 新增 collection 直接调前者。
10. **数据源注册表全懒加载**：`get_structured_retriever("semart")` 只注册 schema + loader，不读 CSV/不开 Chroma/不加载 BGE——能力开关可以挂在每次意图路由上零成本调用；但别在纯单测里访问 `.df` 或 `search`（会真加载 SemArt）。
11. **跨源排序只信 RRF 排名**：各源 `score` 绝对值不可比（SemArt 是 1-cosine distance，未来 museum API 无相似度分数），HybridRetriever 融合时不得对 score 做跨源比较；`relevance_score=1-distance` 是 SemArt 专属形状，仅在其结果上成立。
