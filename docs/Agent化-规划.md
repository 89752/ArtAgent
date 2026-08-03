# ArtAgent Agent 化架构规划（2026-08-02）

> 定位：把当前"RAG 问答应用"重构为"以 Agent 为中心、RAG 只是其中一个工具"的架构。
> 对齐目标：上下文工程、技能调用、长短期记忆、多意图执行，全部落地为可测试的增量阶段。

---

## 1. 目标架构

```
                    ┌──────────────────────────────────────────┐
                    │              Agent 主循环                   │
                    │  load_memory → rewrite/split → intent      │
                    │  → plan → 选工具/技能 → 执行 → 自评/澄清       │
                    └───────┬─────────────┬────────────┬────────┘
                 RAG 工具组  │     技能注册表  │    记忆工具    │
                    ┌───────▼───┐ ┌─────▼─────┐ ┌───▼────────┐
                    │ semantic/  │ │ 分析/总结/  │ │ remember/  │
                    │ exact/     │ │ 调研 等技能 │ │ recall/    │
                    │ timeline/  │ │ (SKILL.md) │ │ forget     │
                    │ web/…     │ │           │ │            │
                    └───────────┘ └───────────┘ └────────────┘
                    ┌──────────────────────────────────────────┐
                    │     上下文工程 ContextBuilder（每轮组装）       │
                    │ 系统块+画像块+摘要块+会话台账块+证据块+历史窗口     │
                    │ （编号引用 / 去重 / token 预算 / 滚动窗口）       │
                    └──────────────────────────────────────────┘
```

原则：上下文是地基，技能/记忆/多意图都依赖它；RAG 退位为"检索工具组"；
一切新能力都必须有单测和 eval 用例，不能只是"加了 prompt"。

---

## 2. 子系统规划

### 2.1 上下文工程（Context Builder）— 最高优先级

现状：SYSTEM_PROMPT 一段话 + 意图建议 + 裸 JSON 工具结果堆进消息，无画像/摘要/台账/预算。

目标：每轮由 `src/agent/context.py` 的 ContextBuilder 组装结构化上下文：

| 块 | 内容 | 来源 | v1 预算 |
|---|---|---|---|
| system | 角色声明、能力清单、技能索引、引用/澄清规则、输出规范 | prompts.py 重构 | 固定 |
| profile | 用户偏好画像（喜欢画家/风格，带权重） | memory/store.py | ≤ 800 字符 |
| summary | 会话滚动摘要（Phase 4 前为空） | conversation_summary 表 | ≤ 1200 字符 |
| session | 会话台账：本轮已展示画作、已推荐画家、待澄清项 | AgentState 新字段 | ≤ 600 字符 |
| evidence | 编号引用 [N]、按 artwork_id 去重、来源元数据 | 检索工具结果 | ≤ 4500 字符 |
| history | 最近 N 轮用户/助手消息（Phase 4 后缩小窗口） | LangGraph messages | ≤ 8 轮 |

要点：
- 证据去重在 context 层做（同一画作多路命中只出现一次，保留最高分）；
- 引用编号 [N] 贯穿"上下文 → 回答 → UI 展示"（ragent CitationMarkup 模式）；
- 预算不足时按优先级裁剪：先裁 history 旧轮，再裁 evidence 长文，绝不裁 system/profile；
- 工具结果不再裸灌：统一走证据块格式化（去 JSON 噪音，只留 title/author/date/snippet/来源）。

### 2.2 工具带：RAG 退位为工具

现状：11 个工具全部是"检索/读"，agent 的职业被锁死在艺术问答。

目标：
- 检索工具组保持 `@tool` + 工具守卫（已有），对外就是一组 retrieve 类工具；
- 新增非检索动作（按阶段落地）：
  - `ask_user`（澄清短路径，Phase 1）：歧义/信息不足时结束本轮、向用户追问；
  - 记忆工具 `remember` / `recall` / `forget`（Phase 4）：agent 主动存取记忆；
  - 文档动作 `summarize_doc`（Phase 3 并入技能，不单设工具）。
- 工具调用链路不变：general_agent 选工具 → guard 三态校验 → 执行 → 回灌。

### 2.3 技能系统（Skill Registry）

定义：技能 = 可复用的多步程序化能力（有步骤、约束、输出规范），区别于原子工具。

落地：
- 目录约定：`agent_skills/<skill_id>/SKILL.md`；
- SKILL.md 格式：front matter（name / description / when_to_use / version / tools）+
  正文（分步执行流程、约束、输出格式）；
- `src/skills/loader.py`：启动时发现 → 解析 → 把每个技能注册为 `skill_<id>` 工具
  （并入 GENERAL_TOOLS + 工具守卫）；
- 执行模型：技能工具收到任务 → 按 SKILL.md 步骤调用其声明允许的工具（受限工具集）
  → 返回结构化结果；步数上限防失控（v1 每技能 ≤ 6 步）。
- **不依赖 MCP（已确认 2026-08-02）**：技能/工具全部进程内 Python 函数直接注册；
  工具守卫的三态校验已吸收 MCP 参数抽取思想。若将来要接第三方 MCP 生态
  （浏览器/文件系统/外部服务），加一个薄适配器（MCP client 工具）即可，
  现有工具接口不变——此为扩展点，不为此做设计。

首发 3 个技能（待确认）：
1. `artwork_deep_analysis`：定位画作 → 元数据+描述 →（用户要求时）视觉分析 → 结构化评析；
2. `document_summary`：定位用户上传文档 → 读页 → 结构化摘要（含页码引用）；
3. `exhibition_research`：本地检索 + web_search → 输出带来源的研究笔记。

### 2.4 长短期记忆

现状：preferences.db（结构化偏好）+ conversations.db（原始消息），无摘要、无台账、无主动记忆。

目标（分两层）：
- 短期（会话内，LangGraph 状态）：会话台账 shown_artworks / recommended_artists /
  pending_clarification / task_steps；工具执行后自动登记，注入 context.session 块；
- 长期（跨会话）：
  1. 用户画像（已有，扩展：显式点击/收藏等隐式信号，可选）；
  2. 滚动摘要：conversations.db 新增 conversation_summary 表，增量摘要器
     （异步、每会话锁、半窗重叠——ragent JdbcConversationMemorySummaryService 模式），
     摘要注入 context.summary 块；
  3. agent 主动记忆工具 remember/recall/forget（已确认加入，Phase 4）：
     agent 显式决定记什么、查什么、忘什么，是"有记忆主体感"的关键动作。

### 2.5 编排增强

- 多意图并行（P0-A，Phase 2）：`sub_questions` × `intent_scores` → 子任务并行执行
  （ThreadPoolExecutor，chroma 已有线程本地客户端）→ 证据按子任务分组合并；
- 澄清短路径（P0-B，Phase 1，已确认策略）：**仅当信息不足以检索/执行时才追问**
  （guard 的 NEED_CLARIFICATION、检索/执行前信息缺口判定）；一般性歧义不打断，
  带合理假设继续，回答中说明假设即可；
- 规划可见（Phase 5）：agent 先输出一句话计划，UI 决策链展示。

---

## 3. 分阶段实施计划

### Phase 0：收尾（不阻塞）
- 完成 `--force` 重索引；跑 A/B（eval/ab_subpipeline_agent.py）出报告；
- 归档清理：确认稳定后删 `data/_backup-20260802`、移除 semart 通道注册。

### Phase 1：上下文工程 v1 + 澄清短路径（地基）✅ 已完成（2026-08-02）
| 任务 | 落点 | 验收 |
|---|---|---|
| 1.1 ContextBuilder | 新增 `src/agent/context.py` | 六块组装、预算裁剪、证据去重可单测 |
| 1.2 SYSTEM_PROMPT 重构 | `src/agent/prompts.py` | 角色+能力+技能索引+引用/澄清规则结构化 |
| 1.3 证据块格式化 | context.py + `_format_result` 契约不变 | [N] 引用编号贯穿回答 |
| 1.4 会话台账 | AgentState + general_tools 后置钩子 | shown/recommended 自动登记、注入 session 块 |
| 1.5 ask_user 澄清节点 | nodes/common.py + graph | 仅信息不足时短路追问（不打断一般歧义）；guard NEED_CLARIFICATION 汇合 |
| 1.6 UI 决策链 | web/service.py | 展示计划/澄清/台账 |

新增单测 ≥ 12（context 块、去重、预算、台账、澄清路由）。

### Phase 2：多意图并行（P0-A）✅ 已完成（2026-08-02）
| 任务 | 落点 | 验收 |
|---|---|---|
| 2.1 并行编排器 | 新节点 `multi_retrieve` | sub_questions>1 时并行执行、单测覆盖纯逻辑 |
| 2.2 多段 prompt 组装 | context.py 扩展 | 每子任务证据分组清晰 |
| 2.3 复合问题 eval | eval/ 用例 ≥ 5 | "对比X和Y，顺便推荐Z" 并行可见 |

### Phase 3：技能系统 ✅ 已完成（2026-08-02）
| 任务 | 落点 | 验收 |
|---|---|---|
| 3.1 SKILL.md 格式与样例 | `agent_skills/` 目录 | 格式文档 + 3 技能 |
| 3.2 SkillLoader | 新增 `src/skills/loader.py` | 发现→解析→注册 skill_<id> 工具，单测 |
| 3.3 执行模型 | loader + guard | 受限工具集、步数上限、失败回退 |
| 3.4 上下文集成 | context.py | 系统块列出技能、技能调用可观测 |

### Phase 4：长短期记忆 ✅ 已完成（2026-08-02）
| 任务 | 落点 | 验收 |
|---|---|---|
| 4.1 滚动摘要 | conversations.db 新表 + `src/memory/summary.py` | 增量、异步、每会话锁、不阻塞 |
| 4.2 摘要注入 | context.summary 块，history 缩至 6 轮 | 长对话 token 受控 |
| 4.3 记忆工具 | `remember/recall/forget` 入工具带 | agent 主动读写可测 |
| 4.4 save_memory 扩展 | nodes/common.py | 摘要+台账落库 |

### Phase 5：打磨与评测 🔄 进行中（2026-08-03）
- ✅ save_memory 旧逻辑清理（偏好记录改由 agent 用 remember 显式写入）
- ✅ ReAct 工具轮次上限（MAX_TOOL_ROUNDS=5，实测 29 次循环问题）
- ✅ 工具多元化：save_collection / list_collections / list_preferences
- ✅ 轻量成本观测：save_memory 记录 turns / tool_rounds / summary_len
- ✅ eval 行为化套件（eval/agentic_eval.py）：RAG-gate/澄清/多意图/技能/记忆/粒度/收藏 8/8 通过
- ✅ 上下文预算 v2（长上下文治理）：统一 ContextBudget + 优先级裁剪（system/profile 不裁
  → summary → evidence → subtasks → history 保底 2 轮）+ 自适应历史窗口 + 体积触发摘要
  （轮数 OR 体积超限）+ context_chars 体积观测（对齐 Claude/LangGraph 的滚动摘要+窗口模式）
- ⏳ 待办：二次 A/B（Phase 1 前后对比）、成本看板细化

---

## 4. 约束与明确不做

- 保持 LangGraph 单图，不引入多 Agent/规划框架；
- 技能 v1 只做"指令 + 受限工具集"，不做任意代码执行沙箱；
- 不建向量记忆库，SQLite 先验证价值；
- 不做浏览器/爬虫代理，web_search 保持简单；
- 每个 Phase 都有单测与验收，禁止"只加 prompt 不加测试"。

## 5. 风险

| 风险 | 缓解 |
|---|---|
| 上下文预算参数需实测调优 | Phase 5 eval + 成本看板 |
| 并行检索线程安全 | chroma 线程本地客户端（已有）+ ThreadPoolExecutor 收敛 |
| 技能指令污染系统 prompt | 技能只在其执行上下文生效，不注入全局 system |
| 异步摘要锁竞争 | 每会话一把 asyncio 锁 |
| LLM 成本上升 | 预算裁剪 + 澄清/技能步数上限 + 看板监控 |

## 6. 待确认的决策

## 6. 已确认的决策（2026-08-02）

1. 首发技能：artwork_deep_analysis / document_summary / exhibition_research ✅
2. 技能/工具 **不接 MCP**：进程内注册；第三方 MCP 生态留薄适配器扩展点 ✅
3. 记忆范围：偏好 + 滚动摘要 + 会话台账 + agent 主动 remember/recall/forget ✅
4. 澄清策略：仅当信息不足以检索/执行时才追问；一般歧义带假设继续 ✅
5. Phase 顺序：Phase 1 上下文工程先行 ✅
