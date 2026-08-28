# 🎨 ArtAgent —— 西方艺术史对话 Agent

<p align="right"><a href="README.md">中文</a> · <a href="README.en.md">English</a></p>

ArtAgent 是一个面向西方艺术史的对话式 Agent，基于 LangGraph 编排。它优先使用本地核心艺术库（约 **5.3 万条来源记录**）回答问题，并支持事实查询、风格对比、时间线梳理、审美偏好驱动的检索、图像分析、文档解读与跨会话长期记忆。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 配置环境变量
# macOS / Linux：cp .env.example .env
# Windows PowerShell：Copy-Item .env.example .env
# 必填：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（对话模型，不预设平台）
# 可选：VISION_MODEL / VISION_API_KEY / VISION_BASE_URL（图像分析；缺省回落对话模型配置，LLM_MODEL 需支持图像输入）
# 可选：JUDGE_MODEL / JUDGE_API_KEY / JUDGE_BASE_URL（评估裁判模型，缺省同对话模型）
# 可选：PDF_IMAGE_EMBED_PROVIDER / MODEL / API_KEY / BASE_URL（PDF 整页图嵌入，默认 DashScope，可换 OpenAI 兼容端点）
# 可选：RERANK_API_KEY（精排）、TAVILY_API_KEY（联网兜底）、MINERU_TOKEN（PDF 精准解析）
# config.yaml 已随仓库提供且不含密钥：超时/并发/检索嵌入等非密钥项直接改它，模型与密钥放 .env

# 首次访问：在注册页创建账号后即可使用。
# 共享部署请保持 ARTAGENT_SEED_DEFAULT_ACCOUNT 未设置；它会创建
# user/11111111 这一已知演示账号，仅适合临时本地体验。

# 3. 启动 Web 界面
python api.py
# 打开 http://127.0.0.1:7860

# 前端开发（可选，React + Vite，需要 Node 20+）
cd frontend
npm install
npm run dev
# 打开 http://127.0.0.1:5173（/api 与 /static 自动代理到 7860）

# 前端生产构建：产物输出到 static/dist
npm run build
```

运行轨迹默认仅保存脱敏元数据 30 天，可通过 `TRACE_RETENTION_DAYS` 调整。模型提供商返回 `usage_metadata` 时会优先采用真实 token 用量，否则回退字符估算。

> 仓库不包含数据资产。运行前需要本地 `data/`（核心库 CSV、Chroma 向量索引、SQLite 记忆库、`data/core/images/` 图片）。只有 CSV 时，请在数据准备环境中构建与之匹配的核心索引后再启动服务；不要把数据资产或本地构建缓存提交到仓库。

## 应用场景

| 场景 | 能做什么 |
|---|---|
| 作品与画家查询 | 按标题、作者、年代、流派检索画作与画家信息 |
| 风格对比与演变 | 对比多位画家或画作的风格差异，梳理某位画家、某个流派的风格演变 |
| 审美偏好驱动检索 | 结合当前问题与已记住的偏好，检索并解释匹配的画家、作品或流派 |
| 图像视觉分析 | 从构图、色彩、笔触等角度分析画作 |
| 专家技能 | 风格对比、时间线梳理、画作深度分析、文档总结、展览前期研究 |
| 文档与表格解读 | 上传 PDF / Excel 后解读内容，可定位到具体页 |
| 数据统计 | 统计库内作品的流派、年代、技法分布 |
| 记忆与收藏 | 记住你的偏好并跨会话调用，维护收藏清单 |

## 数据

核心库（`dataset_id=core`，运行时默认数据源）由三个来源合并归一化：

| 来源 | 内容 | 数量 |
|---|---|---:|
| Wikidata | 结构化作品 / 画家 / 流派 / 馆藏信息 | 30,041 |
| SemArt | 8–19 世纪欧洲绘画的描述文本 | 19,862 |
| Art Institute of Chicago | 馆藏开放数据 | 3,085 |

> 部分记录存在多源交叉（如同一作品同时命中 Wikidata 与 SemArt）。

## 技术栈

| 模块 | 选型 |
|---|---|
| Agent 编排 | LangGraph：load_memory → ask_user → ReAct 工具循环 ⇄ tools → reflection → save_memory |
| 检索 | BGE-M3 语义向量 + 词法双通道（core FTS5 / PDF BM25，跨语言按需翻译）→ 加权 RRF 融合 → Jina Reranker v3.5 API 精排 |
| 对话模型 | OpenAI 兼容接口（不预设平台，经 config.yaml / 环境变量配置） |
| 视觉模型 | OpenAI 兼容视觉模型（可独立配置，缺省回落对话模型配置） |
| 用户体系 | 账号注册/登录/API Key 鉴权，会话/记忆/文档/反馈按用户隔离 |
| 记忆与会话 | SQLite（记忆条目/事件、会话、滚动摘要、用户文档、收藏、反馈、抽取指标） |
| Web 前端 | React 19 + TypeScript + Vite（SSE 流式对话） |
| 测试与交付 | pytest 离线回归、确定性评测门禁与前端类型检查/构建均已接入 GitHub Actions |

## 评估

```bash
pytest -q                                      # 先跑不消耗模型额度的离线回归
python eval/memory_reliability_eval.py         # 验证记忆写入、召回、冲突与遗忘
python eval/agent_eval_v2.py --retrieval-n 100 # 测量核心库的离线检索 Recall@5
python eval/agent_eval_v2.py                   # 跑完整核心在线评估
```

## 项目结构

```text
api.py                 # FastAPI 后端（SSE 流式，python api.py 启动）
web/                   # 服务层：LangGraph 推理与渲染、绘画分析 SSE 编排
frontend/              # React 前端（Vite + TypeScript；构建产物输出到 static/dist）
static/                # 静态资源（svg）与前端构建产物（dist/）
src/
├─ agent/              # LangGraph 图、节点、上下文构建（ReAct + 澄清 + 反思）
├─ memory/             # 长期记忆：条目化存储 / 自动抽取 / 冲突合并 / 画像 / 滚动摘要 / 收藏 / 反馈
├─ tools/              # 工具带（检索 / 分析 / 记忆 / 收藏 / 技能）
├─ retrieval/          # 混合检索：语义 + 词法双通道 + RRF 融合 + 精排
├─ ingestion/          # PDF / Excel 解析（MinerU / 扫描页视觉读取 / 表格）
├─ analysis/           # 绘画分析引擎（视觉评估 / 校验 / 报告存储）
├─ data/               # 文档状态存储与 SQLite 连接共享层
├─ platform/           # 用户 / 认证 / API Key
├─ skills/             # 专家技能加载与激活
├─ subagents/          # delegate_task 并行子智能体执行器
├─ utils/              # 配置 / LLM 客户端 / 日志 / 工具执行治理
├─ tasks/              # 文档解析任务队列
└─ observability/      # 轨迹日志与 /api/metrics
agent_skills/          # 专家技能定义（SKILL.md）
eval/                  # 评估入口与测试集
tests/                 # pytest 测试
```
