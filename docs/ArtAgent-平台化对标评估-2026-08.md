# ArtAgent 平台化对标评估报告（2026-08）

> 评估日期：2026-08-03 ｜ 评估人：Codex（项目经理视角）
> 评估范围：代码（src/ 约 6,900 行、web/ 485 行、static/ 约 1,660 行）、测试（346 个测试函数）、eval 报告、规划文档（ArtAgent-2.0 实施方案 / Agent 化规划 / ragent 借鉴分析）、Git 历史（19 个提交，main 单分支）
> 对标对象：Dify、Coze（扣子）、RAGFlow、LangGraph/LangSmith 生态、CrewAI、Microsoft Agent Framework、OpenAI Agents SDK

---

## 0. 结论摘要

**一句话结论：ArtAgent 是一个"单域、单用户、高质量 Agent 应用"，Agent 内核（编排、检索、记忆、上下文工程、评估）已达到市场框架级水准，但距离"平台"还缺整个平台层——身份与多租户、应用定义与发布、公共 API、MCP/插件生态、任务可靠性、可观测性、部署运维、安全合规。**

按 14 个平台维度打分，综合成熟度约 **2.4 / 5**（Agent 内核 4 / 5，平台层 1~2 / 5）。如果把"平台化"理解为 Dify/Coze 那种可承载多 Agent、多用户、多知识源、可发布可运维的产品，当前完成度大约 **30%**。

关键判断：

1. **内核不弱，缺的是"外壳"**。意图树打分、多意图并行、工具三态守卫、上下文预算、滚动摘要、技能注册表、混合检索 + 精排 + 相关性过滤——这些能力很多自研平台都没有。真正的短板不在模型层，而在工程产品层。
2. **最优先补的不是功能，是"身份与隔离"**。全项目 `user_id` 硬编码为 `web_user`，会话、偏好、文档、向量库全部无归属。没有用户模型，后面的一切（API、租户、安全、计费）都无从谈起。
3. **其次补"应用定义层"**。平台 = 能配置多个 Agent。当前意图树、工具集、Prompt、数据源全部写死在代码里，换一个领域就要改代码。抽出一份 Agent 清单（manifest）是"应用 → 平台"的分水岭。
4. **MCP 是 2026 年的事实标准**，本项目明确选择不接（ragent 分析留了薄适配器扩展点）。作为个人项目可接受；对标市场平台，MCP 双向支持（消费第三方工具 + 把 Agent 发布为 MCP Server）是 P0。
5. **评估体系是最大隐藏资产**，但样本量太小（意图 50 条、检索 25 条、事实题 6 条、行为用例重复 2 次），还撑不起"成熟平台"的质量承诺。扩集 + 流水线化（CI 回归门禁）比再加功能更值钱。

---

## 1. 现状能力盘点（证据）

### 1.1 已具备的 Agent 内核

| 能力 | 落地情况 | 证据 |
|---|---|---|
| 混合编排 | 单图：load_memory → rewrite_split → classify → rag_gate → ask_user → multi_retrieve → ReAct → reflection → web_fallback → save_memory | `src/agent/graph.py` |
| 意图路由 | 打分式意图树（capability/tool/system 三类叶子，阈值过滤，失败回落 general） | `src/agent/intent_tree.py`（239 行） |
| 查询改写/拆分 | 指代消解 + 多问题拆分 + 关键实体抽取 + 歧义标记 | `src/agent/rewrite.py`、`state.py` |
| 多意图并行检索 | sub_questions 并行预取证据，按子任务分组注入上下文 | `graph.py`、`nodes/general.py` |
| ReAct + 工具守卫 | 三态校验（SUCCESS / NEED_CLARIFICATION / FAILED），类型/枚举/必填严格校验，未知参数拒绝 | `src/tools/guard.py` |
| 反思与兜底 | reflection PASS/RETRY + web_fallback，retry 上限防死循环 | `nodes/common.py` |
| 上下文工程 | 六块组装（system/profile/summary/session/evidence/history）+ 编号引用 [N] + 预算裁剪 + 证据去重 | `src/agent/context.py`（324 行） |
| 技能系统 | SKILL.md 注册表，技能 = 多步程序化能力，受限工具集 + 步数上限 | `src/skills/loader.py`、`agent_skills/`（3 个技能） |
| 工具带 | 15 个工具：检索 3、图像 2、联网 1、能力管线 3、记忆 3、收藏 3 | `nodes/general.py:43` |
| 长期记忆 | 偏好画像（SQLite 加权）+ 滚动摘要 + 会话台账 + agent 主动 remember/recall/forget + 收藏夹 | `src/memory/` |
| RAG 检索 | 多源混合检索 + 加权 RRF + qwen3-rerank 精排（主备接力）+ LLM 相关性过滤 + 跨模态（BGE 文本 / tongyi-vision 图像） | `src/retrieval/hybrid.py`、`reranker.py`、`relevance.py` |
| 文档理解 | PDF 页级自适应路由（文字/多模态/双路线）+ MinerU 云端解析 + 表格 schema 推断人工确认 + 数据集切换 | `src/ingestion/` |
| 多模态 | 视觉模型读图（image_lookup analyze / read_page_image），与对话模型明确分工 | `src/utils/llm.py` |
| 可观测性（基础） | 节点级结构化日志 + 耗时 + 上下文体积/工具轮次观测 | `src/utils/logging_config.py` |
| 评估（基础） | 意图分类准确率 + Recall@5 + 行为化 eval（8 类用例）+ 事实准确率 | `eval/` |
| 工程纪律 | 346 个测试函数、纯单测与真实 API 测试分层、eval 固定种子、设计文档完整、提交信息带验收数据 | `tests/`、`docs/`、git log |

### 1.2 当前评估数据（如实引用，含口径）

- 意图分类：README 基线 96.0%（50 条）；2026-08-02 最新一次 100.0%（50/50，Macro-F1 1.000）——注意样本小且走 LLM，波动 ±2pp 属正常。
- 检索 Recall@5：粗排 n=25 口径 64.0%；默认开精排后 n=25 为 76.0%（19/25）。
- 行为化 eval：8 类用例 ×2 次 = 14/14 触发通过（rag_gate / clarify / multi_intent / skill / memory_write / grain / collection）。
- 事实准确率：6/6（长尾精确元数据题）。
- 测试：346 个 `test_*` 函数，纯单测占主体（无 LLM/无网络，秒级）；端到端测试走真实 API，全量约 25–35 分钟。

### 1.3 已规划但未落地的（规划文档中的承诺）

- Stage 7 能力层：color_analysis / museum_search（Met/Rijksmuseum）/ nearby_venues / wiki_lookup。
- Phase 2：多租户（kb_id + user_id 隔离）、异步任务深化、可观测性扩展（数据源贡献条数/图片路线生成开销/工具调用分布）、评测集扩展、Docker Compose。
- 模型路由与降级（ragent 分析 P2-②，未实施，仅检索精排做了主备接力）。

---

## 2. 市场对标基准（2026-08）

| 平台 | 定位 | 平台层能力（与 ArtAgent 直接相关的） |
|---|---|---|
| Dify | 生产级 LLM 应用/Agent 开发平台（开源 + 云） | 可视化工作流、插件系统（50+ 官方工具）、**MCP 双向支持**（导入工具 + 应用发布为 MCP Server）、多模型供应商、应用发布（Web/API/嵌入）、监控与日志、企业版多租户/RBAC/SSO。2026-07 v1.16 新增 Agent 沙箱 |
| Coze（扣子） | 零代码 Agent 平台（字节） | 工作流引擎、插件生态、**技能商店**、长短期记忆、定时任务、多渠道发布（飞书/微信/Discord/网页）、Agent World；3.0 定位 Agent 原生操作系统 |
| RAGFlow | 深度文档理解 RAG 引擎 + Agent 模板 | 复杂文档解析、可溯源引用、知识治理、记忆管理（v0.24, 2026-02）、图编排 Agentic RAG |
| LangGraph + LangSmith | 代码级编排框架 + 可观测平台 | 有状态图、持久执行、human-in-the-loop、LangSmith 全链路 trace/eval/成本归因、LangGraph Platform 托管部署 |
| CrewAI | 角色化多 Agent 框架 | **MCP 全传输支持**、crew 协作模式、快速原型 |
| Microsoft Agent Framework | 企业编排框架（AutoGen + Semantic Kernel 合流） | 图工作流、多 Agent 模式（顺序/并发/handoff/group chat）、OpenTelemetry、负责任 AI 护栏（PII、prompt 注入防御） |
| OpenAI Agents SDK | 轻量 Agent SDK | 多 Agent 委派、内置 guardrails、会话管理 |

2026 年平台共性（= ArtAgent 的差距清单来源）：

1. **MCP 双向支持**已是标配（消费第三方工具 + 把自家 Agent 发布为 MCP Server）。
2. **应用 = 可配置资产**：平台都能"新建一个 Agent/Bot"，配置模型、工具、知识、记忆策略后发布，而不是改代码。
3. **身份与多租户**：所有商业平台都有用户、空间/组织、权限，数据按用户隔离。
4. **公共 API + 多渠道发布**：OpenAI 兼容接口、SDK、嵌入、IM 渠道、Webhook。
5. **可观测与运营**：trace、成本、延迟、错误率看板；评估流水线；反馈闭环（点赞/点踩回流）。
6. **任务可靠性**：异步任务队列、持久化状态、重试、断点续跑（Dify 工作流、LangGraph 持久执行）。
7. **安全与合规**：鉴权、限流、prompt 注入防护、审计日志、数据导出/删除。
8. **部署形态**：Docker Compose / Helm / 云托管，配置化而非硬编码路径。

---

## 3. 能力矩阵与差距评分

评分口径：1=缺失，2=雏形，3=可用，4=接近成熟，5=市场领先。

| # | 维度 | 现状 | 市场基准 | 评分 | 最大差距 |
|---|---|---|---|---|---|
| 1 | Agent 编排与推理 | 意图树+多意图+ReAct+反思+上下文预算 | 框架级（LangGraph 同级） | **4.0** | 无可视化/可配置化，图写死在代码 |
| 2 | 知识管理与 RAG | 混合检索+精排+多模态+PDF/表格入库 | RAGFlow/Dify 同级 | **4.0** | 无知识库版本/重解析/权限/管理台 |
| 3 | 记忆体系 | 偏好+摘要+台账+主动记忆工具 | 平台级中上 | **3.5** | 无向量记忆、无显式遗忘策略/隐私删除 |
| 4 | 工具系统与生态 | 15 工具+守卫+技能注册表 | 平台级 | **2.5** | 无 MCP、无插件市场、无第三方工具 |
| 5 | 评估体系 | 意图/检索/行为化/事实 4 套 | 自研平台少见 | **3.5** | 样本小、无 CI 门禁、无 LLM-as-judge 答案质量评估 |
| 6 | 可观测性 | 结构化日志+节点耗时 | LangSmith/商业平台 | **2.5** | 无 trace 视图、无成本/延迟看板、无指标采集 |
| 7 | 人机协同 | ask_user 澄清短路径 | 框架级 HITL | **2.0** | 无审批/中断/恢复，无"任务需确认后执行" |
| 8 | 多 Agent 协作 | 无（单图单 Agent） | CrewAI/MAF 标准能力 | **1.0** | 无 sub-agent/handoff/group chat |
| 9 | 任务可靠性与并发 | FastAPI BackgroundTasks + 全局锁 | Dify/LangGraph 持久执行 | **1.5** | 无队列、无持久化任务、进程崩溃即丢 |
| 10 | 身份/多租户/安全 | 单用户硬编码，无鉴权 | 商业平台标配 | **1.0** | 用户模型缺失，一切隔离无从谈起 |
| 11 | API 与集成面 | 内部 REST（聊天/会话/文档） | OpenAI 兼容 API/发布 SDK | **1.5** | 无对外 API、无密钥管理、无发布渠道 |
| 12 | UI/Studio 产品化 | 自研单用户聊天 UI | Dify/Coze Studio | **2.0** | 无 Agent 配置台/管理台/知识库可视化 |
| 13 | 部署与运维 | `python api.py` 本地跑 | Docker Compose/托管 | **1.0** | 无容器、无 CI、无环境分级 |
| 14 | 成本与性能治理 | 上下文预算+工具轮次上限+体积观测 | 平台看板级 | **2.5** | 无每请求成本核算、无限流、无超时治理策略 |
| — | **综合** | — | — | **2.4** | 平台层整体缺失 |

---

## 4. 差距详析与改进意见（按优先级）

### P0 —— 平台地基（不做则无法称为平台）

#### P0-1 用户与数据隔离模型

现状：`web/service.py` 硬编码 `WEB_USER_ID = "web_user"`；SQLite 表虽带 `user_id` 字段但只有单用户；Chroma 无用户/知识库维度的 metadata filter（规划文档 Phase 2 已承诺）；会话表无 `user_id` 列。

建议：
- 引入最小用户模型：`users` 表 + 会话/偏好/文档/向量集合全部挂 `user_id`（SQLite 迁移即可）。
- Chroma 元数据统一加 `user_id`/`kb_id`，检索时强制 filter（不是可选）。
- API 层加简单鉴权（首版：静态 token / 登录态 cookie），为后续 RBAC 留接口。
- 验收：两个用户的数据完全不可见、不可串；删除用户可级联清空会话/偏好/向量/文件。

#### P0-2 应用定义层（Agent Manifest）——"应用 → 平台"的分水岭

现状：意图树（`intent_tree.py`）、工具集（`GENERAL_TOOLS`）、Prompt、技能、数据源全部硬编码，换领域必须改代码。

建议：
- 定义 `AgentSpec`（YAML/JSON manifest）：`id / name / system_prompt / model_config / tools 白名单 / skills / 绑定的 knowledge base / memory 策略 / guardrails / 意图树`。
- 启动时加载 `agents/` 目录，一个进程可注册多个 Agent（`registry`）。
- 把当前艺术 Agent 整体固化为 `agents/artagent.yaml`，作为平台内置模板——这同时保住现有成果，又让平台具备"新建 Agent"能力。
- 这一步做完，Dify/Coze 的"应用资产化"概念就成立了；后续 Studio 只是这个 manifest 的编辑器。

#### P0-3 公共 API 面与发布

现状：只有聊天 UI 的内部 REST（`/api/chat` 返回 SSE HTML 气泡，业务与渲染耦合）。

建议：
- 提供 OpenAI 兼容 `/v1/chat/completions`（流式 + 工具调用），这是生态接入成本最低的面。
- 业务层与渲染层解耦：`stream_answer` 目前直接产出 HTML 气泡，应拆成"结构化事件流"（node/intent/tools/evidence/answer）与"渲染器"两层，前端和 API 各取所需。
- 会话、文档、偏好接口补 `user_id` 维度并文档化；提供 OpenAPI 文档（当前 `docs_url=None`，对平台产品是负分项）。
- 发布通道（网页嵌入 iframe/JS SDK、API Key 体系）放 P1，但接口设计现在就要留。

#### P0-4 MCP 双向支持（2026 事实标准）

现状：明确决策"不接 MCP"（Agent化-规划.md §2.3），工具全部进程内注册。

建议：
- **消费侧**：加薄 MCP client 适配器（stdio/HTTP），把第三方 MCP Server 的工具导入现有 `GENERAL_TOOLS` 注册表——工具守卫（`guard.py`）已经吸收了 MCP 参数抽取思想，接入成本低。
- **提供侧**：把当前 Agent/技能发布为 MCP Server（FastMCP 即可），让 Cursor/Claude Desktop/Dify 能调用——这是"平台生态位"的最低证明。
- 这与"技能不依赖 MCP"的旧决策不冲突：进程内技能仍是一等公民，MCP 只是互操作层。

#### P0-5 任务化与持久执行

现状：PDF 解析走 `fastapi.BackgroundTasks`，进程重启/崩溃即丢；无重试、无任务表、无并发控制；SQLite 全局锁 + 单进程 uvicorn。

建议：
- 建 `tasks` 表（task_id/type/status/payload/progress/error/created_at），解析、入库、摘要生成全部任务化，状态可查询可恢复（现有 `documents` 表已有雏形，扩展到通用任务）。
- 启动时扫描 `processing` 任务并标记 `interrupted`，提供"重试"入口（文档重解析也是 Stage 6 已规划的 Phase 2 项）。
- 并发治理：解析任务信号量（如同时最多 2 个）、请求级限流中间件（慢速启动：按 IP/session 的令牌桶）。

### P0 —— 可运营性

#### P0-6 可观测性升级

现状：结构化日志 + 节点耗时，无 trace 视图、无指标、无成本归因看板。

建议：
- 每个请求生成 `request_id` 贯穿日志/SSE/数据库，按会话可回放整条 Agent 轨迹（logs 已是 JSON 友好格式，做查询即可）。
- 接 OpenTelemetry（LangChain/LangGraph 有原生支持）或 LangSmith 免费层；自建最小方案：`agent_runs` 表记录 每轮意图/节点/工具/耗时/token/上下文体积，提供 `/api/metrics` 汇总（成本、延迟、工具分布、反思触发率、兜底率）。
- 现有 `context_chars`/`tool_rounds` 观测点直接成为看板字段，成本很低。

#### P0-7 安全加固

现状与风险点：
- 无鉴权、无限流（单机本地可接受，平台化后不可接受）。
- **XSS**：前端 `setBubbleHTML` 用 `marked.parse(box.textContent)` 渲染 LLM 输出，marked 默认不过滤原始 HTML——服务端已 `html.escape`，但 `textContent` 取回的是解码后的原文，恶意模型输出/文档注入可构造 `<img onerror>`。需引入 DOMPurify 或改为纯文本渲染白名单。
- **Prompt 注入**：上传 PDF/表格内容会进上下文，无注入检测/隔离；联网搜索结果直接入证据。建议加"外部内容边界"（证据块与系统指令隔离提示、可疑内容标记，参考 Microsoft Agent Framework 的 guardrail 思路）。
- 上传校验：按扩展名分类 + 50MB 上限，无内容嗅探、无按用户归属；`read_page_image` 已有路径穿越防护（好），文档删除级联已做（好）。
- 密钥管理：`.env` 已 gitignore（好），但无密钥轮换/加密存储；生产建议环境变量注入 + 密钥服务。

#### P0-8 部署产物与 CI

现状：无 Dockerfile/docker-compose/CI 配置；依赖人工跑 `python api.py`。

建议：
- Docker Compose：`api`（uvicorn 多 worker）+ 可选 `worker`（解析任务）+ 卷（data/、SQLite）+ 健康检查；Chroma 先保持嵌入式，预留外部向量库切换。
- GitHub Actions：PR 跑全部纯单测（秒级）+ eval 的离线子集（意图/检索可用 seed 固定、rerank 可关）；真实 API 的端到端放手动/nightly。
- 配置分级：`.env.example` 已具备雏形，补 `ARTAGENT_ENV`、日志级别、限流参数、任务并发数等平台参数。

#### P0-9 评估流水线与反馈闭环

现状：eval 脚本齐备但靠人肉跑；无答案质量评估；无用户反馈采集。

建议：
- 评估集扩容：意图 ≥ 200 条（含跨域负样本）、检索 ≥ 100 条、事实题 ≥ 50 条、工具选择集（Stage 7 已规划）、技能调用集、多轮记忆集。
- 加 LLM-as-judge 答案质量评估（事实性/引用忠实度/拒绝得体性），纳入 `eval/` 与行为化套件并列。
- CI 门禁：意图 ≥ 基线、Recall@5 ≥ 基线、行为化 8/8，波动超 2pp 告警。
- 前端加 👍/👎 + 原因标签，落 `feedback` 表，回流为 eval 候选集——这是平台"自进化"的最小闭环。

### P1 —— 能力扩展（平台化后按需）

1. **多 Agent 协作**：sub-agent（并行研究）、handoff（任务交接）、审批型 HITL（工具执行前人工确认）。LangGraph 原生支持 supervisor/team 模式，工程量可控。
2. **Agent/知识库管理台**：现有前端升级为"用户端 + 管理端"双区；管理台至少覆盖：Agent 清单编辑（manifest）、工具/技能开关、知识库状态与重解析、用户与会话管理、成本/延迟看板。
3. **模型路由与降级**：多供应商 + 健康状态 + TTFT 探测（ragent 分析 P2-②），当前只有检索精排有主备接力，对话/视觉/嵌入都要有降级策略。
4. **知识库增强**：重解析、版本与回滚、chunk 可视化校对、按 KB 检索隔离、多 KB 绑定到 Agent。
5. **记忆增强**：向量记忆/实体记忆（明确不与 SQLite 冲突，互补）；隐私优先：记忆项来源可查、可单项删除（GDPR 类能力）。
6. **定时与触发**：定时任务（知识源刷新、摘要批处理）、Webhook 触发——Coze 已有，平台标配。
7. **多渠道发布**：网页嵌入 SDK、飞书/钉钉/企微适配器（面向国内市场的差异化）。
8. **Stage 7 领域能力**：color_analysis / museum_search / nearby_venues / wiki_lookup，作为"平台内置技能包"而不是平台本身的依赖。

### P2 —— 商业化与规模化

- SaaS 多租户（组织/空间/成员/角色）、SSO（OIDC）、审计日志、用量计费。
- 插件/技能市场（对标 Coze 技能商店、Dify 插件市场），支持社区提交与版本审核。
- 分布式：外部向量库（Milvus/Qdrant/PGVector）、消息队列（Redis/RabbitMQ）、多 worker 横向扩展。
- A2A 协议互操作（2026 年已与 MCP 并立）。

---

## 5. 建议路线图（约 3 期）

### 一期（4–6 周）：平台地基

- P0-1 用户模型与数据隔离（含 SQLite 迁移 + Chroma filter + 会话归属）
- P0-2 Agent Manifest + 多 Agent 注册（ArtAgent 固化为内置模板）
- P0-3 业务/渲染解耦 + OpenAI 兼容 API + OpenAPI 文档
- P0-4 MCP 消费侧接入（先 stdio/HTTP 导入第三方工具）

**验收**：两个用户各建一个 Agent，互不可见；第三方 MCP 工具可被 Agent 调用；API 客户端可流式对话。

### 二期（4–6 周）：可运营

- P0-5 任务表与持久执行 + 限流
- P0-6 可观测性（agent_runs + 成本/延迟看板 + 请求回放）
- P0-7 安全加固（DOMPurify、上传归属与嗅探、注入边界、鉴权）
- P0-8 Docker Compose + CI + 配置分级
- P0-9 评估扩容 + LLM-as-judge + 反馈闭环

**验收**：一键 `docker compose up` 起服务；CI 绿了才能合代码；任意一次对话可回放完整决策链与成本。

### 三期（6–10 周）：生态与差异化

- 管理台（Agent 编辑、知识库管理、看板）
- MCP 提供侧（发布为 MCP Server）+ 多 Agent 协作（sub-agent/handoff）
- 模型路由降级 + 定时任务
- 可选：多渠道发布、技能市场雏形、Stage 7 领域技能包

---

## 6. 定位策略建议

**不要把 ArtAgent 从"艺术 Agent"改造成"空壳平台"——把艺术 Agent 变成平台的第一张王牌模板。**

- 艺术领域问答（对比/时间线/偏好推荐/读图/文档问答）本身在市场上没有强对手，这是差异化资产：平台内置的"种子应用"直接证明平台价值（对标 Coze 的 Agent World 模板、RAGFlow 的预置模板）。
- 平台内核（AgentSpec + 工具注册表 + 技能 + 记忆 + 检索）必须领域无关；领域知识收敛到 manifest、意图树、工具、技能包、知识库——现有代码已部分做到（`structured_retriever` 注册表、`skills/loader`、`IntentLeaf` 都是可配置化雏形），继续这个方向。
- 市场切入建议："垂直 Agent 平台"而不是"通用平台"。Dify/Coze 是通用平台，ArtAgent 如果做通用平台正面竞争没有胜算；做"艺术/文化/教育领域 Agent 平台"（内置艺术知识管线、视觉分析、画册文档理解）才有差异化。这意味着评测、模板、技能市场都围绕领域做深。

---

## 7. 风险与诚实提示

1. **工程量被低估的风险**：平台层不是"再加几个 Stage"的量级。上面 P0 的 9 项，单人开发保守估计 2–3 个月；建议严格按一期验收门槛推进，不要同时铺开。
2. **当前评估数据不足以支撑对外承诺**：50 条意图/25 条检索/6 条事实，样本太小，`100%` 这类数字对外发布会有回旋镖。先扩集，再谈"成熟"。
3. **单点依赖**：LLM/视觉/精排/解析全部依赖 DashScope 一家 + 免费额度模型切换（glm-5→glm-4.7 已发生过一次被迫切换），平台化后必须模型路由与降级（P1-3）前置到二期。
4. **XSS 与注入是当前真实存在的风险**：marked 渲染 + 上传文档进上下文，在"单机自用"场景风险可控，一旦对外发布就是首发事故，必须随安全加固一起处理。
5. **存储耦合 UI**：`conversations` 表存渲染好的 HTML 气泡，schema 与前端强耦合，改版要迁移数据；P0-3 的解耦（结构化消息流 + 渲染器）应尽早做。
6. **并发与锁**：SQLite 全局连接 + 单进程，多用户后会先撞锁再撞性能；一期就要按用户隔离 + 连接池/写队列规划。
7. **"明确不做"清单需要重审**：旧决策（不做 MCP、不做代码沙箱、不做浏览器代理）在平台化语境下要逐条重审——MCP 已升级为 P0；代码沙箱（Dify 已出 Agent 沙箱）可以继续不做，但要在文档里说明安全模型。

---

## 附录：代码证据索引

- 单用户硬编码：`web/service.py` `WEB_USER_ID = "web_user"`
- 工具清单：`src/agent/nodes/general.py:43` `GENERAL_TOOLS`
- 技能注册：`src/skills/loader.py` `register_skills()`
- 工具守卫：`src/tools/guard.py`（validate_args / llm_extract_parameters）
- 意图树：`src/agent/intent_tree.py` `INTENT_LEAVES`
- 上下文预算：`src/agent/context.py` `ContextBudget`
- 混合检索：`src/retrieval/hybrid.py`（RRF + 精排 + 路线感知去重）
- 文档任务：`api.py` `BackgroundTasks`；`src/ingestion/pipeline.py`
- 记忆：`src/memory/store.py`（preferences）、`conversations.py`（会话）、`summary.py`（滚动摘要）、`agent_memory.py`（remember/recall/forget）
- 前端渲染：`static/app.js` `setBubbleHTML`（marked.parse，XSS 风险点）
- 评估：`eval/run_eval.py`、`eval/agentic_eval.py`、`eval/ab_embedding_models.py`
- 规划承诺：`docs/ArtAgent-2.0-实施方案.md` §8/§10；`docs/Agent化-规划.md` §5/§6
