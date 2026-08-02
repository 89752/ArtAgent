# ArtAgent 2.0 实施方案（完善版 · 开发用）

> 本文档 = 原《ArtAgent-2.0-设计方案》+ 2026-07-31 代码级评估的 8 项补充 + Stage 1–6 实施记录。
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
| LLM/视觉/检索 API | DashScope（`.env` 中 `DEEPSEEK_API_KEY` 一把 key 通用） | **对话模型现为 `glm-4.7`**（2026-08-01 切换：glm-5 免费额度耗尽；GLM 系列各有独立 100 万 token 免费额度，glm-4.6 可作后备）；视觉模型 `qwen3.5-omni-plus`；联网兜底 Tavily |
| 向量模型 | BGE `bge-small-en-v1.5`，本地缓存 | `api.py` 已设 `HF_HUB_OFFLINE=1`；测试脚本不设也可跑（缓存命中） |
| 数据/索引 | SemArt（21,384 幅）+ `data/index/chroma/`（217MB，collections: `semart` + 遗留 `user_pdfs`） | `user_pdfs` 与 `data/uploads/` 是旧 PDF 实验遗产，**Stage 3/6 处置** |

**验证命令（全量约 25–35 分钟，走真实 API）：**

```bash
PY="C:/Users/86188/anaconda3/envs/artagent/python.exe"
$PY tests/test_access.py       # 纯单测，无 LLM，秒级
$PY tests/test_tools.py        # 工具冒烟（真实 API）
$PY tests/test_multi_tool.py   # 多工具链
$PY tests/test_pipelines.py    # 四分支端到端
$PY eval/run_eval.py           # 意图分类 + Recall@5（基线 96.0% / 64.0%；64.0% 为 n=25 粗排口径，默认 n=20 为 70.0%；Stage 4 起默认开精排，n=25 为 76.0%——两个口径别混，A/B 用 RERANK_ENABLED=0 跑对照）
```

---

## 1. 当前状态：Stage 1 / 2 / 3 / 4 / 5 / 6 已完成 ✅

> Stage 1：commits `a08bda1`、`1aaf48c`；Stage 2：见 §1.4；Stage 3：见 §1.5；Stage 4：见 §1.6；Stage 5：见 §1.7。

### 1.1 已落地的改动（Stage 1）

- **数据访问层 `src/data/access.py`**：`fuzzy_match`（精确→去冠词→分词包含 三级递进）、`row_to_artwork_dict`（兼容 DataFrame 行与 Chroma metadata）、`format_evidence_block`（模板化 + 空字段清理）、`EVIDENCE_SNIPPET_LEN=200`。配套 `tests/test_access.py` 17 个纯单测。
- **工具 7 → 5**：`semantic_search` / `exact_lookup` / `query_painter_knowledge`（去 LLM 化，返回结构化统计）/ `image_lookup`（吸收 analyze_image，`analyze=True` 触发视觉分析）/ `web_search`。`compare_artwork_styles` 已删除。`GENERAL_TOOLS` 与 `SYSTEM_PROMPT` 已同步。
- **管线去重**：`timeline_gather_periods` 不再 iterrows 拼第二遍证据；comparison/recommendation/web_fallback 的证据格式化统一走 `format_evidence_block`。
- **隐藏 bug 修复**：旧 `get_by_author` 整串 contains 匹配遇上 SemArt 的倒序 AUTHOR（`"GOGH, Vincent van"`）永远查空，`fuzzy_match` 分词匹配根治。
- **清理**：faiss 残留索引（49MB）、pdf_loader.py 孤儿代码、README/requirements 的 Gradio 失效引用；`.gitignore` 的 `data/` 改为 `/data/`（原规则会把 `src/data/` 源码也忽略）。

### 1.2 保留未动的遗产（Stage 6 处理）

- `data/uploads/`：旧会话目录 + 真实 PDF（用户上传过的文件，勿直接删，Stage 6 文档管理体系建立时迁移或经用户确认后清理）。新体系的上传落在 `data/uploads/{kb_id}/{doc_id}/`（与旧目录结构不同，不冲突）。
- Chroma `user_pdfs` 集合：旧 chunk 策略的向量，新体系用 `user_pdf_text`/`user_pdf_images` 两个新 collection，`user_pdfs` 已无消费者，Stage 6 废弃。

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

### 1.5 Stage 3 实施记录（PDF 解析与入库）✅

**已落地的改动：**

- **`src/ingestion/` 新包**：
  - `page_classifier.py`：PyMuPDF 逐页信号采集（text_len/image_ratio/has_fonts/公式符号密度 + producer 文档级先验），确定性规则判定 text/multimodal/dual 路由，公式密集页标 `force_mineru`。阈值全部做成模块常量（起始值 200字符/30%/80%）。
  - `blocks.py`：`Block`（解析器产出的语义块）/ `Chunk`（入库检索单元，page_id/chroma_id/metadata 统一）。
  - `pdfplumber_fallback.py`：兜底解析器（全部产出 text 块，section 留空）；与 MinerU 约定同签名接口（`parse_pages(pdf_path, page_nos) -> list[Block]`），MinerU 接入时可直接替换。已压制 pdfminer 字体 warning 刷屏。
  - `chunker.py`：按 block_type 分流——text 短段归并（同页同小节，下限 180）/ 长块滑窗拆分（上限 500、12% 重叠）/ table 整块 / equation 整块+前后 80 字符上下文 / image_caption 独立。
  - `multimodal_indexer.py`：整页渲染（150 DPI PNG 落盘）→ DashScope `tongyi-embedding-vision-plus`（实测 1152 维，文本/图片同空间，文搜图跨模态成立）→ `user_pdf_images` collection。
  - `pipeline.py`：编排（路由→两路入库→状态落盘）。公式密集页在 MinerU 不可用时退多模态整页图（不硬用 pdfplumber 解公式）。状态用 JSON 文件（`data/index/doc_status.json`）支撑轮询，**Stage 6 换 SQLite documents_store 时整体替换**。
- **检索层扩展**：`userdoc_text_retriever.py`（BGE 空间，`user_pdf_text` collection）+ `userdoc_image_retriever.py`（DashScope 空间）实现 BaseRetriever 并注册进 `get_hybrid_retriever()`；`_dedup` 改为**路线感知**：同页文字 chunk 与整页图同时命中时丢弃整页图（文字证据更精确，且 LLM 尚不能读图），同页多个文字 chunk 全保留。`semantic_search` 按 source 分形状输出：semart→画作字典（不变）；PDF→`{source, title="《doc》第N页", content, description_snippet, page, ...}`。
- **Web 链路**：`POST /api/documents/upload`（multipart，50MB 上限，`BackgroundTasks` 后台解析）+ `GET /api/documents[/{doc_id}]` 状态轮询；前端侧栏"我的文档"面板（上传按钮 + 状态徽标：解析中/N 片段/失败，3s 轮询）。新增依赖：`PyMuPDF/pdfplumber/dashscope/python-multipart`（已入 requirements）。
- **UI 防污染**：`collect_artworks` 与 `_parse_artworks_from_messages` 跳过带 `source` 键的文档片段（不进配图卡片），但保留在 retrieved_docs 作 LLM 证据——`format_evidence_block` 输出形如 `- 《画册》第3页: 内容…`，溯源自然成立。

**实施中的关键决策（对原方案的修订/细化）：**

1. **MinerU 精准解析 API 已接入（2026-08-01 闭环 ✅）**。路线选型：官方 v4 云端 API，本地安装方案否决（~4GB 模型 + `datasets`/pyarrow 冲突前科 + CPU 慢；且当日上午用 16 页全扫描画册实测云端 API 质量优秀——OCR 连贯、图注完整，本地无质量收益）。落地为 `src/ingestion/mineru_parser.py`：`MINERU_TOKEN` 配置即启用，未配置/调用失败自动降级 pdfplumber（公式密集页仍退多模态整页图）；v4 流程 = `file-urls/batch` 申请签名 URL → PUT 上传（自动开解析）→ `extract-results/batch/{batch_id}` 轮询 → zip 取 `*_content_list.json`；`model_version=vlm`；**整份文档上传解析后按页过滤**（`page_ranges` 参数的 page_idx 语义未明，Phase 2 实测后再改按页解析省 quota，注意 quota 按整文档页数计）；块映射 text/list/code→text、equation→LaTeX 整块、table→caption+HTML+footnote 整块、image/chart→文档自带图注（无图注不产块，视觉内容走整页图路线）、header/footer/page_number 等噪声丢弃；内嵌图落盘 `work_dir/images/` 备用。实测：RAG PDF 5 页云端 8.8s 解析、160 blocks → 52 chunks（含 1 个 table 块——chunker 的 table 路径首次命中真实数据）；`tests/test_mineru_parser.py` 13 个纯单测。新增显式依赖 `requests`（原为传递依赖）。
2. **Qwen-VL 读图作答（2026-08-01 已闭环 ✅）**。落地方式为 **`read_page_image` 工具**而非 ToolMessage 多模态化：对话模型 glm-4.7 是纯文本大脑，继续负责工具决策；命中整页图（`source=user_pdf_image`，结果带 `image_path` + `read_hint`）时 Agent 自主调用 `read_page_image`，由 `qwen3.5-omni-plus` 读图返回文字描述。这保留了"工具内不偷藏 LLM 调用"的纪律（该工具的存在意义就是视觉读取，与 `image_lookup analyze=True` 同类），且每次读图调用在日志中可观测（正是"命中图片路线的生成开销"度量点）。路径安全校验：只放行 `data/uploads/` 下的图片文件。已实测：含纯图版页的测试画册 → Agent 依次调 `semantic_search` → `read_page_image` → 准确答出《星夜》画面内容（旋转星空/柏树/村庄/厚涂笔触）。视觉客户端统一收敛到 `utils/llm.py::get_vision_llm`（`image_lookup` 同步复用）。
3. **去重从"page_id/doc_id 塌缩"改为路线感知**：Stage 2 的朴素 page_id 去重会把同页多个文字 chunk 误杀——同页多 chunk 内容不同必须全保留；只抑制"同页整页图 vs 文字 chunk"的冗余命中。
4. **eval 与 parity 校验锁定 semart 源**：`semantic_search` 融合用户 PDF 后，`eval/run_eval.py` 的 Recall@5 与 `verify_recall_parity.py` 一律 `sources=["semart"]`，保证 64.0% 基线口径不被开发库里的测试文档污染。
5. **测试文档级联清理已验证**：`collection.get(where={"doc_id": ...}) → delete(ids=...)` 两个 collection 各清一遍即可（Stage 6 删除按钮复用此流程）。沙箱环境下 `shutil.rmtree` 被 safe-delete 拦截，工作目录残留文件需用户手动清理（`data/uploads/default/`，已被 gitignore）。
6. **DashScope 多模态查询成本**：`user_pdf_image` 检索每次查询调一次多模态编码 API；collection 为空时短路返回，零成本。

**验收数据（2026-08-01）：**

- 纯单测累计 **81 个全绿**：access 17 + structured_retriever 23 + hybrid 14 + page_classifier 15 + chunker 12。
- 端到端（真实服务）：上传百度百科 RAG PDF（5页：4 文字路线+1 双路线 → 21 chunks + 1 整页图）与 Renaissance 测试 PDF（3页全双路线 → 6 chunks + 3 整页图），混合检索来源标签正确、双路线页面文字命中时整页图被抑制、提问回答正确声明"上传文档讨论的是 RAG 技术"并溯源。
- 回归：`verify_recall_parity` 锁 semart 源 25 条 query 逐位 0 不一致；`test_pipelines` 5 问全过；`test_tools` 4 工具全过。
- eval（补跑完成，glm-4.7）：意图分类 **96.0%**（48/50，基线 96.0%，达标）；Recall@5 n=20 口径 70.0%（锁 semart 源；n=25 口径 64.0% 已于上午验证持平）。
- 事故与修复：全量回归曾两次在 `rec_filter` 节点卡死——DashScope 偶发连接挂起不返回（额度耗尽前兆），而 LLM 客户端未设超时导致无限等待。已在 `src/utils/llm.py` 与 `image_lookup.py` 加 `request_timeout=180` + `max_retries=2`（修复后同一调用 161s 正常完成）。**教训：任何外部 API 客户端必须显式设超时。**
- **模型切换（2026-08-01 下午）**：原对话模型 glm-5 免费额度耗尽（403 FreeTierOnly），切换为 **glm-4.7**（百炼 GLM 系列各有独立 100 万 token 免费额度；glm-4.6 亦可作后备，glm-4.5 仅支持 stream 模式不可用于 invoke）。`DEEPSEEK_MODEL` 环境变量名已是误称——它只是 DashScope 模型 ID 的槽位，改值即换模型，后续 Stage 留意评测口径以 glm-4.7 为准。实测 glm-4.7 工具调用正常、意图分类比 glm-5 快约 40%。

### 1.6 Stage 4 实施记录（检索质量层）✅

**已落地的改动：**

- **上下文头（context header）**：`pipeline.py::_context_header`——PDF 文字 chunk 向量化时拼 `《文档》 | 章节` 头（只影响向量与展示，不改存储：documents 仍是原始 content，header 落 `metadata.context_header`）；SemArt/结构化表不加。展示侧 `tools/retrieval.py` 把章节并进标题（`《画册》第3页 · 章节名`，旧文档无此字段自动跳过）。方案的"实体"位无确定性来源（NER/LLM 抽取留 Phase 2），Phase 1 为文档名 + MinerU 标题层级两档。新索引的文档即生效；旧文档需 Stage 6 重解析才补头。
- **Rerank（qwen3-rerank 精排）**：`src/retrieval/reranker.py`（DashScope 兼容端点，显式 30s 超时 + 2 次重试，任何失败返回 None 由调用方降级）；`hybrid.py::_rerank_fused`——RRF 粗排后取池 `RERANK_POOL=40` 送精排，文本候选按槽位重排、`user_pdf_image` 整页图保持原槽位（文本精排器读不懂图，占位标签只会被打低分错误压底）；`rerank_score` 写 metadata（原生 score 不动，跨源不可比纪律不变）；开关 env `RERANK_ENABLED`（默认开）/ `search(rerank=False)`（eval A/B 用）。
- **相关性校正通用化**：`src/retrieval/relevance.py::llm_relevance_filter`——rec_filter 思路的通用轻量版：编号候选 → 确定性 LLM 选相关编号 → **只删不重排**；任何失败返回原列表；LLM 过度过滤时按原序兜底补足 `min_keep`（永不返回空证据）。挂在编排层两处：`comparison_retrieve`（每对象检索后过滤再进合成）与 `general_tools` 节点（ToolNode 包成普通节点，对 semantic_search 的 ToolMessage 过滤后重序列化，query 取自 tool_call args；节点名不变，service.py 标签无需同步）。recommendation 保留专用 rec_filter（特征匹配+排除+理由生成，不止相关性）；timeline 无向量检索不适用。开关 env `RELEVANCE_FILTER_ENABLED`（默认开）。

**实施中的关键决策（后续 Stage 需知晓）：**

1. **精排候选池取 40 而非方案的 15–20**：实测（n=25 口径）pool=20 池召回 68.0%、pool=40 76.0%，精排两次都 100% 兑现池内召回——瓶颈在池召回不在排序。代价是每次检索各源多取候选 + 精排文档数翻倍（仍在单次 500 文档限额内）。
2. **精排 API 部分响应必须整体回退**：按槽位重排时若响应条数少于候选数，被移动的文档会在原槽位复制一份（同一文档占两位）——已加守卫 `len(ranked) != len(text_slots)` 时回粗排原序（test_reranker 覆盖）。
3. **semantic_search 工具保持无 LLM**：相关性过滤放图节点层而非工具内部——工具是 eval Recall@5 与 test_tools 的直接消费者，必须确定性可测；LLM 判断留编排层且每次过滤日志可观测（`relevance_filter` 事件含 in/kept/dropped）。与"工具内不偷藏 LLM 调用"纪律一致。
4. **过滤器只删不重排、永不返回空**：顺序已由 RRF+精排决定，LLM 只剔除噪声（提示词写明"拿不准的保留"）；`min_keep=2` 兜底防合成端拿到空证据答非所问。超过 `max_candidates=12` 的尾部候选不参与过滤、原样透传。
5. **延迟/成本上升是设计内开销**：每次 semantic_search 现在 = BGE 编码 + 40 候选 + 1 次 qwen3-rerank +（编排层）1 次 glm 过滤调用；一轮全量回归时长约为 Stage 3 的 1.5–2 倍。怀疑卡死时先看日志——只要 `rerank`/`relevance_filter` 事件在滚动就是正常的（180s 超时兜底）。
6. **UI 顺带收益**：过滤后 ToolMessage 重序列化仍为 list[dict]，`_parse_artworks_from_messages` 无感知——无关画作不再进配图卡片。

**验收数据（2026-08-01，glm-4.7）：**

- 纯单测累计 **132 个全绿**：+`test_reranker` 15（mock API：解析/截断/重试/开关/槽位保持/部分响应回退）、+`test_relevance` 15（fake LLM：过滤/兜底/降级 + general_tools 消息重写）。
- **eval（锁 semart 源，n=25 基线口径）**：意图分类 **98.0%**（49/50）；**Recall@5 = 76.0%（19/25），较 64.0% 基线 +12pp**——逐条 A/B（`RERANK_ENABLED=0` vs 默认开）：3 条被精排救回、0 条由命中变未命中，纯增益。
- 回归：`verify_recall_parity`（`RERANK_ENABLED=0`）25 条逐位 0 不一致、n=25 复现 64.0% 基线；`test_tools` / `test_multi_tool` / `test_pipelines` 全绿（13 分钟）。

### 1.7 Stage 5 实施记录（结构化表格上传）✅

**已落地的改动：**

- **文件类型路由与加载（`src/ingestion/table_loader.py`，零模型调用）**：`classify_upload` 按扩展名分通道（.pdf → Stage 3，.csv/.xlsx/.xls → 表格）；CSV 编码兜底 utf-8→gb18030→latin1（中文 Excel 导出常是 GBK）；xlsx/xls 多 sheet 确定性选择——`有效列数×数据行数`打分（有效列=非 `Unnamed:N` 且非全空），学习计划首表是说明页也能选对「每日打卡表」。新增依赖 openpyxl/xlrd（纯 Python）。
- **Schema 推断（`src/ingestion/schema_inference.py`）**：LLM 看表头（含 dtype）+ 前 4 行猜 4 个列角色 + 显示名；输出逐列与真实表头校验，幻觉列置空告警（宁可少判不可错判）；任何失败返回全空建议（用户仍可手填）。`SCHEMA_INFER_PROMPT` 明确 **entity_col 是"归属主体"而非记录标题**（画作表取 AUTHOR 不取 TITLE——初版提示词实测踩坑：取 TITLE 时 capability 门显示支持、管线却按画名查作者静默查空，正是方案§6.2 要防的静默出错；提示词加"作品-创作者结构取创作者"规则与反例后修正）。
- **确认注册流程（`src/ingestion/table_pipeline.py`）**：状态机 `processing → pending_confirm → active/failed`（与 PDF 共用 doc_status.json，kind 字段区分）；确认时 `entity_col` 必填+逐列校验→注册 `StructuredTableRetriever`（df 懒加载，首访才读盘）+ 挂进 HybridRetriever（source 名=dataset_id=`table_{doc_id}`）；`restore_active_tables()` 在 api lifespan 启动时从状态存储重建（注册表/Hybrid 是内存单例）；`unregister_table()` 为 Stage 6 删除级联预留。
- **数据源切换**：`HybridRetriever.active_dataset`（默认 semart，`set_active_dataset` 校验已注册）；`semantic_search` 工具按它过滤（选表时 SemArt 不参与、选 semart 时表不参与；用户 PDF 两路无 dataset_id 属性不受切换影响）；`service.stream_answer` 每轮从单例读进 `state.dataset_id`（重置清单纪律不变）；端点 `GET /api/datasets`（清单+当前项）/ `POST /api/dataset/active`（切换，未注册 404）。
- **能力开关真正生效**：graph 层 `_capability_supported` 无需改动（Stage 2 预留）——无轴表 timeline 意图、无描述表 recommendation 意图自动降级 general；未注册 dataset_id 同样降级。
- **表格检索第三级兜底（`structured_retriever._fuzzy_search` 扩展）**：实体 fuzzy → 描述整串包含 → **词重叠打分**（长 query 按内容词在实体+描述列的命中率排序，≤20 词、>3 字母、确定性无模型）。没有第三级时，recommendation 的 30–60 词 extracted_features 整串包含必空、推荐管线在用户表上必瘫。
- **user_table 结果形状（`tools/retrieval.py::_format_result` 第三形状）**：原始列全带上（小写键，`exclude_from_results` 按 `entity_col.lower()` 定位靠它）+ 通用 `title/description_snippet`（证据模板与 Stage 4 相关性过滤拼候选靠它）+ `source` 键（不进 UI 配图卡片，陷阱#13）。
- **Web UI 切片**：上传按钮泛化 PDF/CSV/XLSX/XLS；文档列表表格徽标（待确认 schema 可点击/已启用 N 行+能力）；**schema 确认弹窗**（4 个角色下拉映射列名、显示名输入、推断依据展示——§12 确认交互形式定稿为字段映射下拉 UI）；侧栏**数据源切换器**（含能力提示）。新端点 `POST /api/documents/{doc_id}/schema`（确认/纠正）。

**实施中的关键决策（后续 Stage 需知晓）：**

1. **确认前不注册**：推断完成只落 pending_confirm 状态与建议 schema，用户确认/纠正才进注册表——方案§6.2"猜错 entity_col 会静默出错"的防线；active 状态也允许重确认改 schema。
2. **表格 Phase 1 不入向量库**：注册即走 Stage 2 预留的 fuzzy 兜底路径；词重叠打分是长 query 的救命路径。向量化用户表格（description 列入 BGE 库）留 Phase 2。
3. **active_dataset 三处口径**：工具读 `hybrid.active_dataset`（semantic_search 无状态）、节点读 `state.dataset_id`、service 每轮从单例同步进 state——切换只改单例一处，三处自然一致；eval/test_tools 默认 semart 不受影响。
4. **正样本的现实约束**：SemArt 名画家 TIMEFRAME 全部单桶（50 年桶 > 个人生涯），多组 group_by_axis 只有跨世纪集合实体可验——子集含 ROMANESQUE PAINTER, Italian（8 时期 29 幅）专验多组；4 名画家×50 幅验指名查询与推荐排除。
5. **刁难样本的真实价值**：学习计划推断实测猜 entity=学习主题、axis=阶段（过度猜测的典型），确认 UI 纠正为无轴+desc=具体内容后 timeline=False/recommendation=True——人工纠正确实是必需品不是摆设。
6. **多 sheet 选择不需要 LLM**：`有效列×行数`打分在学习计划上 98:22:16:4 碾压性区分，零模型调用原则守住。

**验收数据（2026-08-01，glm-4.7）：**

- 纯单测累计 **156 个全绿**：+`test_table_ingest` 21（路由/sheet 选择/编码兜底/推断 5 例/确认注册/恢复/能力门/空角色守卫/结果形状/词重叠 2 例）、+`test_reranker` 3（双端点主备接力，见下）。
- **HTTP e2e 三件套全过**（`scripts/e2e_stage5_http.py`，真实服务+真实推断）：正样本推断全中（AUTHOR/TIMEFRAME/DESCRIPTION/IMAGE_FILE，双能力 True）；负样本无轴无描述（双 False）；刁难样本 sheet 选对+纠正确认生效；清单/切换/404 正常。
- **对话 e2e 全过**（`scripts/e2e_stage5_chat.py`，真实 graph）：timeline 全管线在画作集上运行（梵高单时期）；recommendation 全管线在画作集运行（词重叠喂候选、梵高排除）；书单 timeline 意图降级 general（无 tl_* 节点）；`restore_active_tables` 重启恢复 3 表验证。
- 回归：parity（关精排）25 条逐位 0 不一致、复现 64.0%；test_tools/test_multi_tool/test_pipelines 全绿（日志 `logs/stage5_regression.log`）；eval 意图分类 94.0%（47/50，基线 96.0% 的 ±2pp 容差内达标）、Recall@5 76.0% 持平。
- **精排额度事故与双端点接力（当日插曲）**：回归跑到一半 qwen3-rerank 免费额度耗尽（17:29 起连续 403）。当日已先知先觉把 `reranker.py` 重写为**双端点主备接力**——按模型名自动选端点（gte-rerank-v2/qwen3-vl-rerank 走原生 `services/rerank` 报文；其余走兼容报文），`RERANK_MODEL`/`RERANK_FALLBACK_MODEL` env 槽位，主模型重试耗尽自动接力后备、双失败才降级粗排；事故发生时**热备 gte-rerank-v2 一次调用即接管**，回归全程无感知。事后实测 gte-rerank-v2 主力口径 **Recall@5 = 76.0%（19/25），与 qwen3-rerank 逐条一致**——满额 100 万免费额度，`.env` 已切其为新主力（跳过死模型重试税）。**订正旧记载："gte-rerank-v2 已下线"为误传**（混淆了 gte-rerank v1 的 2026-05-30 停服）；兼容端点仅 qwen3-rerank 一个文本精排，vl-rerank 在其上 404 属正常（走原生端点）。按量付费价约 0.5 元/百万 token（国际站 $0.1/1M）。

### 1.8 Stage 6 实施记录（文档/数据源生命周期管理）✅

**已落地的改动：**

- **`src/data/documents_store.py`（SQLite `documents` 表）**：替换 Stage 3/5 的 JSON 状态文件（`data/index/doc_status.json`），统一持久化 doc_id/kb_id/kind/doc_name/status/started_at/finished_at/file_path/file_size/pages/text_chunks/image_pages/elapsed_sec/error，以及 kind-specific 的 metadata JSON（PDF 路由分布、表格 schema/行列数/显示名/能力开关）。对外 `get_document` / `list_documents` 仍返回与旧 `list_doc_status` 兼容的扁平 dict，前端与管线代码改动最小化。启动时 `init_db()` 自动建表；若表为空且旧 JSON 存在，则一次性迁移并重命名原文件为 `.json.migrated`。
- **状态存储切换**：`src/ingestion/pipeline.py` 与 `table_pipeline.py` 的 `update_doc_status` / `get_doc_status` / `list_doc_status` 全部改走 `documents_store`，不再直接读写 JSON；`api.py` lifespan 先 `documents_store.init_db()` 再 `service.restore_tables()`。
- **级联删除**：`web/service.py::delete_document` 统一处理 PDF/表格删除——PDF 调用 `pipeline.delete_pdf_vectors` 清理 `user_pdf_text` 与 `user_pdf_images` 两个 collection 中该 `doc_id` 的全部向量；表格调用 `table_pipeline.unregister_table` 从 `_REGISTRY` 与 HybridRetriever 移除，若其为当前生效数据源则复位为 `semart`；随后 `shutil.rmtree` 删除上传文件目录，最后删除 SQLite 记录。`DELETE /api/documents/{doc_id}` 已接入。
- **文件库列表 UI**：侧栏「我的文档」面板渲染所有文档（文件名/状态/页数/chunk 数/路由分布/失败原因），解析中的文档显示进度徽标与轮询，已落定文档显示「×」删除按钮；删除前二次确认并提示「将同时删除上传文件和索引向量」。
- **并发与过滤加固（Stage 6 验证期间顺手修复）**：`hybrid.py` 的 Chroma PersistentClient 改为**线程本地单例**——FastAPI BackgroundTasks（线程池）与主线程共用一份缓存 Collection 在 Windows 下会触发 "attempt to write a readonly database"（`data/uploads/default/threadpool-test-001` 即当时的复现目录）；`relevance.py::llm_relevance_filter` 对 `source="user_pdf_image"` 的整页图结果**一律保留不送文本过滤**——其文本 snippet 只是占位标题，LLM 过滤会误杀 `read_page_image` 的唯一入口，与 Stage 4 精排"整页图保持原槽位"同纪律。两者各配 1 条纯单测。

**实施中的关键决策（后续 Stage 需知晓）：**

1. **一上传即落库**：`service.save_upload` 在文件写入磁盘后立即 `documents_store.add_document(kind=..., status='processing')`，后台解析任务再调用 `update_doc_status` 补充结果；前端轮询 `/api/documents` 时即使解析极快也能看到记录，避免 JSON 时代「先写文件后写状态」的竞态空窗。
2. **metadata 合并而非覆盖**：`update_document(metadata={...})` 会与现有 metadata 做 dict.update，表格确认时追加 `confirmed_schema` / `supports_*` 等字段不会冲掉之前的 `proposed_schema` / `rows` / `cols`。
3. **删除时禁止处理中状态由 UI 兜底**：后端 `delete_document` 不硬拦 `processing` 文档，因为后台任务异常时用户需要能清理脏记录；但 UI 在解析中时禁用删除按钮，避免正常流程冲突。
4. **旧 JSON 迁移是冷切换**：新库首次启动自动迁移旧状态，迁移成功后重命名原文件；若 `.migrated` 已存在则先删除再重命名，避免 WinError 183 警告。
5. **未做独立文件库页面**：按 §9「Web UI 拆进各 Stage」原则，Stage 6 的最小能力落在侧栏面板而非新增页面；后续 Phase 2 如需「解析详情页/逐页路由分布」再扩展。

**验收数据（2026-08-02，glm-4.7）：**

- 纯单测累计 **167 个全绿**：+`test_documents_store` 8（SQLite CRUD / 旧 JSON 迁移 / metadata 合并 / 状态形状兼容）、+`test_stage6_lifecycle` 3（PDF 级联删向量+文件+记录 / 表格注销+复位 active / 删除不存在文档抛 KeyError）。
- 文档库 UI 手动验证：上传 PDF → 侧栏显示解析中 → 完成后显示 chunk 数与整页图数 → 删除二次确认 → 刷新后文档消失；上传表格 → 待确认 schema → 确认激活 → 删除后数据源切换器自动移除该项。
- 全量回归（commit 前补跑完成）：`test_tools` / `test_pipelines`（四分支 5 问）/ `test_multi_tool`（3 场景）全绿；eval 意图分类 **100.0%**（50/50，≥94% 达标）、Recall@5 n=25 口径 **76.0%**（19/25）与 Stage 4/5 精排基线逐位持平（同日 n=20 口径为 75.0%，仍是两个口径别混）。注意：回归中途 glm-4.7 免费额度耗尽（403 FreeTierOnly），`.env` 的 `DEEPSEEK_MODEL` 已换为 deepseek-v4-flash，后续场景与 eval 均在该模型下通过——模型槽位口径随 `.env` 为准。
- `test_tools` 的 `semantic_search` 断言按 source 形状兼容化：开发库含用户真实 PDF（莫奈手稿 16 页全图版）时 top-k 会混入 `user_pdf_image` 整页图结果（无 `author` 键），画作形状断言只针对 semart 结果（与 §13 #12 锁 semart 源同理由）。

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

## 4. Stage 3：PDF 解析与入库 ✅（已完成，实施记录见 §1.5；MinerU 接入与 Qwen-VL 读图两项遗留均已于 2026-08-01 闭环）

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

## 5. Stage 4：检索质量层 ✅（已完成，实施记录见 §1.6）

1. **上下文头（context header）**：向量化前给 PDF 文字 chunk 拼接 `[文档 | 章节 | 实体]` 头（只影响向量与展示，不改存储）；SemArt/结构化表不加。
2. **Rerank**：RRF 粗排 → top 15–20 → **qwen3-rerank** 精排 → top 5–8。已核实：DashScope OpenAI 兼容端点 `POST https://dashscope.aliyuncs.com/compatible-api/v1/reranks`，单次 ≤500 文档、单文档 ≤4000 token，支持 `instruct` 参数。~~gte-rerank-v2 已下线勿用~~（**2026-08-01 实测订正：gte-rerank-v2 未下线**——它不在兼容端点、走原生 `services/rerank` 端点但服务正常，免费额度与 qwen3-rerank 各自独立 100 万，已落地为精排后备模型，见 §1.7 末尾接力设计）。彩蛋：`qwen3-vl-rerank`（同原生端点，文本兼容）可给多模态整页图做精排，Phase 2 可选。
3. **结果相关性校正通用化**：把 `recommendation` 的 `rec_filter` 思路提炼为 HybridRetriever 后的通用轻量 LLM 过滤步骤，所有分支受益。
4. 不做（Phase 2 再说）：BM25/关键词混合、结果压缩；永不做：向量数值量化。

**验收**：reranker/relevance_filter 纯单测（mock 候选集）；eval Recall@5 应较 64.0% 基线**提升**（这是本 Stage 的价值度量）。

---

## 6. Stage 5：结构化表格上传 ✅（已完成，实施记录见 §1.7）

依托 Stage 2 的 `StructuredTableRetriever` + `TableSchema`：

1. **文件类型路由**（零模型调用）：.csv/.xlsx/.xls → 表格通道；.pdf → Stage 3 通道。
2. **Schema 推断必须有人工确认**：LLM 看表头+前几行猜列角色 → 用户确认/纠正后生效。原因：猜错 entity_col 会让 recommendation 的排除逻辑**静默出错**（排除了错误的行而不报错），比明显报错更危险。
3. **能力开关真正生效**：无时间/分类列的表，`timeline` 意图降级 general（Stage 2 预留的判断在此启用）。
4. 明确不做：MinerU 表格块升格为结构化表（HTML→DataFrame 可靠性不足，后续阶段再议）。

**验收**：schema 推断单测；准备两份测试表格（字段完整支持 timeline+recommendation 的 vs 不支持的纯列表）验证推断与降级。

---

## 7. Stage 6：文档/数据源生命周期管理 ✅（已完成，实施记录见 §1.8）

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
| 测试表格素材 | ✅ 已定三件套（2026-08-01）：SemArt 分层子集 CSV（正样本，Stage 5 开工时脚本生成：3–5 个跨多时期画家全作品 + 噪声行共 100–300 行，AUTHOR 倒序格式顺带压测 fuzzy_match）+ `tests/fixtures/plain_list_books.csv`（负样本：有实体列、无时间轴、无自由文本描述，已入库）+ 用户的学习计划 xlsx（刁难样本：有"天数/阶段"日期类列但语义无关，专测 schema 人工纠正交互 + 多 sheet 选择，文件在用户桌面，开工时直接读） |
| schema 确认交互形式 | ✅ 已定（2026-08-01）：字段映射下拉 UI——确认弹窗内 4 个角色各一个下拉框（列名+「无」），预填推断值，用户改后确认；显示名可编辑 |
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
12. **eval/parity 必须锁 `sources=["semart"]`**：semantic_search 已融合用户 PDF，不锁源会被开发库里的测试文档污染指标（64.0% 基线只对 semart 源成立）。
13. **带 `source` 键的是用户文档片段**：`collect_artworks`/`_parse_artworks_from_messages` 靠这个键跳过它们防止配图卡片污染；新增工具返回形状时注意画作字典**不要**带 source 键。
14. **Chroma 用户 collection 用 `get_or_create`**：用户文档未上传前 collection 不存在，检索器对空集合短路返回（尤其 user_pdf_image，避免白调 DashScope 编码 API）。
15. **pdfminer 警告已压制在 ERROR**（`pdfplumber_fallback.py` 顶部）：中文字体缺 FontBBox 会逐条刷屏，别再全局调回 WARNING。
16. **视觉/对话是两套模型**：glm-4.7（纯文本大脑，工具决策）与 qwen3.5-omni-plus（唯一"眼睛"）分工——需要看图的场景只能走视觉工具（`image_lookup analyze` / `read_page_image`），别指望对话模型读图；视觉客户端统一用 `utils/llm.py::get_vision_llm`。
17. **`read_page_image` 只放行 `data/uploads/` 下的图片路径**（防路径穿越）：SemArt 图片的分析走 `image_lookup analyze=True`，不要混用。
18. **MinerU quota 按整份文档页数计**：`mineru_parser.parse_pages` 是整份上传、按页过滤——`page_ranges` 参数的 page_idx 语义（原页码 vs 重排序）未实测前勿用（Phase 2 省 quota 优化点）；每日 2000 页高优额度，批量灌库前先算页数。Token 在 `.env` 的 `MINERU_TOKEN`，空着自动降级 pdfplumber。
19. **Recall@5 有两个口径，别混**：`RERANK_ENABLED=0`（粗排）复现 64.0% 基线；Stage 4 起默认开精排是 76.0%（均 n=25 锁 semart 源）。跑 `verify_recall_parity` 必须关精排——开着跑出的"不一致"是精排改序的设计行为，不是 bug；它证明粗排路径无回归的唯一姿势是 `RERANK_ENABLED=0`。
20. **相关性过滤只在编排层，工具里没有**：`semantic_search` 工具与 eval 检索路径不含 LLM 过滤（保确定性）；过滤在 `comparison_retrieve` 与 `general_tools` 节点。改工具返回形状时保住 `title`/`description_snippet` 两键——过滤器靠它们拼候选清单，键没了过滤静默失效（降级回原列表，不报错）。
21. **schema 推断的 entity_col 是"归属主体"不是记录标题**：画作表必须取 AUTHOR 而不是 TITLE——取 TITLE 时 capability 门照常显示"支持"，管线却按画名匹配作者**静默查空**（比报错危险）。`SCHEMA_INFER_PROMPT` 已写明"作品-创作者结构取创作者"规则与反例，改提示词时别把这个区分改没；推断结果必须过人工确认才注册。
22. **精排双端点与主备接力**：qwen3-rerank 走兼容端点（唯一），gte-rerank-v2 / qwen3-vl-rerank 走原生 `services/rerank` 端点（报文不同，**未下线**——2026-08-01 实测，各模型免费额度独立 100 万）。`reranker.py` 按模型名自动选端点，主模型失败自动接力 `RERANK_FALLBACK_MODEL`（默认 gte-rerank-v2），双失败才降级粗排——换模型只改 env（`RERANK_MODEL`/`RERANK_FALLBACK_MODEL`），别把端点 URL 写死在某一个调用里。
23. **用户表格检索是三级兜底**：实体 fuzzy → 描述整串包含 → 词重叠打分。recommendation 的长特征 query（30–60 词）只有第三级能命中——改动时别把第三级砍了，否则推荐管线在用户表上必瘫。表格 Phase 1 无向量索引是刻意设计（Phase 2 才考虑入 BGE 库）。
24. **注册表/Hybrid 是内存单例，active_dataset 三处口径**：重启必须 `restore_active_tables()`（已挂 api lifespan；改 doc_status.json 结构时同步它）。工具读 `hybrid.active_dataset`、节点读 `state.dataset_id`、service 每轮从单例同步——切换数据源只改单例一处；注销表用 `unregister_table()`（两注册表都要清）。
