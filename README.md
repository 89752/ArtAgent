# 🎨 ArtAgent —— 西方艺术史对话 Agent

<p align="right"><a href="README.md">中文</a> · <a href="README.en.md">English</a></p>

ArtAgent 是一个面向西方艺术史的对话式 Agent，基于 LangGraph 编排：内置 **5.5 万+ 条作品记录**的本地艺术库，支持事实查询、风格对比、时间线梳理、偏好推荐、图像分析、文档解读与跨会话长期记忆。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 必填：LLM_API_KEY（对话模型）
# 可选：RERANK_API_KEY（精排）、TAVILY_API_KEY（联网兜底）、MINERU_TOKEN（PDF 精准解析）

# 3. 启动 Web 界面
python api.py
# 打开 http://127.0.0.1:7860

# 前端开发（可选，React + Vite，需要 Node 20+）
cd frontend
npm install
npm run dev
# 打开 http://127.0.0.1:5173（/api 与 /static 自动代理到 7860）

# 前端生产构建：产物输出到 static/dist，由 FastAPI 直接托管
npm run build
```

> 仓库不包含数据资产。运行前需要本地 `data/`（核心库 CSV、Chroma 向量索引、SQLite 记忆库、`data/core/images/` 图片）；只有 CSV 时可执行 `python scripts/06_index_core.py --csv data/core/artworks_core.csv` 重建索引。

## 应用场景

| 场景 | 能做什么 |
|---|---|
| 作品与画家查询 | 按标题、作者、年代、流派检索画作与画家信息 |
| 风格对比与演变 | 对比多位画家或画作的风格差异，梳理某位画家、某个流派的风格演变 |
| 偏好推荐 | 根据你表达的审美偏好，推荐合适的画家与作品 |
| 图像视觉分析 | 从构图、色彩、笔触等角度分析画作 |
| 专家技能 | 画作深度分析、文档总结、展览前期研究 |
| 文档与表格解读 | 上传 PDF / Excel 后解读内容，可定位到具体页 |
| 数据统计 | 统计库内作品的流派、年代、技法分布 |
| 记忆与收藏 | 记住你的偏好并跨会话调用，维护收藏清单 |

## 核心能力

- **本地艺术库优先**：知识来自合并后的核心库（Wikidata + SemArt + AIC），查不到时再联网（Tavily / Wikipedia / Met 馆藏 API），不硬编。
- **意图路由 + ReAct**：寒暄、定义、算术直接回答；实时问题走联网；对比 / 时间线 / 推荐走专用能力工具；信息不足先反问。
- **26 个可调用工具**：语义检索、精确查询、画家知识、图像定位与视觉分析、PDF 整页图读取、风格对比、时间线、偏好推荐、色彩分析、聚合统计、馆藏检索、维基百科、联网搜索、记忆读写删、收藏清单 CRUD，以及 3 个专家技能（画作深度分析 / 文档总结 / 展览研究）。
- **长期记忆（可选增强）**：显式“记住 / 忘记”开箱即用；可开启自动抽取（`MEMORY_AUTO_EXTRACT=1`）、语义冲突合并（`MEMORY_SMART_MERGE=1`）、跨会话用户画像（`MEMORY_PROFILE_REFRESH=1`）；全部存本地 SQLite，记忆面板按条查看 / 删除。
- **文档与表格**：上传 PDF / Excel，MinerU 精准解析（可选）、扫描页视觉读取，回答可引用《文档》第 N 页。
- **Web 界面**：SSE 流式展示思考链、多会话后台生成（回答未结束也能开新对话）、停止生成、侧栏折叠 / 拖拽调宽、深色模式、参考来源卡片、记忆面板、对话反馈。

## 数据

核心库（`dataset_id=core`，运行时默认数据源）由三个来源合并归一化：

| 来源 | 内容 | 数量 |
|---|---|---:|
| Wikidata | 结构化作品 / 画家 / 流派 / 馆藏信息 | 30,041 |
| SemArt | 8–19 世纪欧洲绘画的描述文本 | 19,862 |
| Art Institute of Chicago | 馆藏开放数据 | 3,085 |

> 部分记录存在多源交叉（如同一作品同时命中 Wikidata 与 SemArt）。

合并去重后共 **5.5 万+ 条作品记录**，其中带描述的记录已建立 Chroma 向量索引（BGE-M3 多语言向量，数量随核心库重建更新）。年代以 8–19 世纪欧洲绘画为主（约占 83%），另含少量 20 世纪早期作品，约 7,000 条未标注年份；每条记录都带图片引用（本地图片或馆藏 URL）。

## 技术栈

| 模块 | 选型 |
|---|---|
| Agent 编排 | LangGraph：load_memory → rewrite_split → classify → rag_gate → multi_retrieve → ReAct 工具循环 → reflection → save_memory |
| 检索 | BGE-M3 语义向量 + 词法双通道（core FTS5 / PDF BM25，跨语言按需翻译）→ 加权 RRF 融合 → Jina Reranker v3.5 API 精排 |
| 对话模型 | DeepSeek / Qwen（OpenAI 兼容 API） |
| 视觉模型 | Qwen-Omni（图像分析、PDF 整页图） |
| 记忆与会话 | SQLite（memory_items / memory_events / conversations / conversation_summary） |
| Web 前端 | React 19 + TypeScript + Vite（SSE 流式对话） |
| 测试 | pytest 快档（54 个测试文件，离线，CI 已接入） |

## 评估

```bash
python eval/agent_eval_v2.py --retrieval-n 100   # 离线检索 Recall@5
python eval/agent_eval_v2.py                     # 全量（在线评估）
pytest                                           # 快档离线测试
```

最近基线（2026-08）：

| 维度 | 结果 |
|---|---|
| 答案质量（30 条黄金集） | 平均 4.67/5 · 通过率(≥4) 93% |
| 事实准确率 | 27/31（87%） |
| 多轮对话 | 6/6 |
| 工具选择 | 42/48（88%） |
| 对抗与安全 | 8/10 |
| 意图诊断（软信号） | 36/40（90%） |
| 路由决策 | 13/15 |
| 检索 Recall@5 | 90.0%（core · 语义+词法混合 · Jina API 精排） |

## 路线图

**已完成**

- **混合检索与精排**：语义向量（BGE-M3）与词法（core FTS5 / PDF BM25）双通道召回，查询语言与索引语言不一致时按需翻译，RRF 融合后经 Jina Reranker v3.5 API 精排；
- **成熟工具带**：意图路由 + 26 个工具，覆盖查询、对比、时间线、推荐、视觉分析、统计、记忆与收藏；
- **长期记忆**：显式记忆、自动抽取、语义冲突合并与跨会话画像，记忆可见可控；
- **文档理解**：PDF 文字层解析与整页图像识别双通道、表格上传问答，回答可溯源到具体页；
- **评估体系**：答案质量、事实、工具、多轮、对抗、意图、路由、检索多维评估与测试集；
- **质量闭环**：LLM 自动裁判评分、规则判题与状态断言，用户点赞/点踩反馈入库并可导出为评估候选集。

**规划中**

- **多用户与数据隔离**：用户体系、会话与知识库归属隔离；
- **OpenAI 兼容 API**：对外开放流式对话接口与 OpenAPI 文档；
- **MCP 工具接入**：导入第三方工具扩展 Agent 能力；
- **部署与运维**：Docker 一键启动、CI 门禁、成本与延迟可观测。

## 项目结构

```text
api.py                 # FastAPI 后端（SSE 流式，python api.py 启动）
web/                   # 服务层：LangGraph 推理与渲染
frontend/              # React 前端（Vite + TypeScript；构建产物输出到 static/dist）
static/                # 静态资源（svg）与前端构建产物（dist/）
src/
├─ agent/              # LangGraph 图、意图树、改写、节点、上下文
├─ memory/             # 长期记忆：条目化存储 / 自动抽取 / 冲突 / 画像 / 情景
├─ tools/              # 工具带（检索 / 分析 / 记忆 / 收藏 / 技能）
├─ retrieval/          # 混合检索：语义 + 词法双通道 + RRF 融合 + 精排
├─ ingestion/          # PDF / Excel 解析（MinerU / Qwen-VL / 表格）
├─ tasks/              # 文档解析任务队列
└─ observability/      # 轨迹日志与 /api/metrics
agent_skills/          # 专家技能定义（SKILL.md）
eval/                  # 评估入口与测试集
tests/                 # pytest 快档
```
