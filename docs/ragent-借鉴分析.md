# Ragent 可借鉴能力分析（对齐 ArtAgent Stage 7 与检索链路）

> 分析对象：[nageoffer/ragent](https://github.com/nageoffer/ragent)（Apache-2.0，Java/Spring AI 多模块项目，本地克隆：`ragent-ref/`）
> 分析日期：2026-08-02 ｜ 用途：为 ArtAgent 的 Stage 7（Agent 架构 / Tool Calling）和检索链路增强提供借鉴清单

---

## 1. 结论先行

Ragent 与 ArtAgent 在检索主干上高度同构：多通道并行检索 + 加权 RRF 融合 + 去重 + Rerank 候选池 + 意图路由 + 流式输出 + eval/trace。**这些你已经有了，不用照抄。**

真正值得借鉴的，按优先级分三档：

| 优先级 | 能力 | 一句话价值 |
|---|---|---|
| P0（Stage 7 骨架） | 工具参数抽取三态校验 | 工具调用从"模型说了算"升级为"校验后才调用"，必填缺失会追问而不是带垃圾参数执行 |
| P0（Stage 7 骨架） | 意图树 + 多意图并行 | 意图从"4 选 1"升级为"对所有叶子打分 + 阈值过滤 + 多意图并行检索"，节点直接挂工具/KB |
| P0 | 查询改写 + 多问题拆分 | 一次提问拆成多个子问题，配合多意图并行，召回质量上一个台阶 |
| P1（检索小改） | 通道归因日志 + 统一 chunk key | 让你正在做的加权 RRF "是否有效"可以被观测和验证 |
| P1 | 引用编号 + 历史清洗 | 证据编号 [N] 随回答落地，下一轮喂历史前自动清理，防止编号污染 |
| P1 | 编排兜底短路径 | 歧义反问 / 系统直答 / 空检索明说，三处短路不用白烧 token |
| P2（生产化） | 增量会话摘要记忆 | 对话超长时压缩为摘要注入上下文，控 token 成本 |
| P2 | 模型路由与降级 | 多供应商候选 + 健康状态 + 失败自动换路，流式首包超时探测 |
| P2 | 定时刷新远程知识源 | 对应"本地核心库 + 实时 API 增量"设想的调度侧 |

明确**不借鉴**：Redis/RocketMQ/分布式限流（单机 FastAPI 用 asyncio 信号量即可）、知识图谱通道（收益/成本比低，Phase 2 再说）、ES/BM25 通道（Chroma + SQLite FTS 够用）。

---

## 2. 架构对比

Ragent 流式对话管线（`rag/service/pipeline/StreamChatPipeline.java`）：

```
loadMemory → rewriteQuery（改写+拆分）→ resolveIntents（多意图打分）
  → handleGuidance（歧义→直接反问，短路）
  → handleSystemOnly（全系统意图→不检索直答，短路）
  → retrieve（多子问题×多意图并行）
  → handleEmptyRetrieval（无结果→明说，短路）
  → streamRagResponse（来源定稿→引用编号→组装 prompt→流式输出）
```

ArtAgent 现状（`src/agent/graph.py`，LangGraph）：

```
load_memory → contextualize（指代消解）→ classify（4 意图单选路由）
  → comparison / timeline / recommendation / general_agent(tools)
  → reflection（PASS/RETRY）→ web_fallback → save_memory
```

共同点：意图路由、检索后反思/兜底、记忆、eval、trace。差异点即借鉴点：ragent 把"意图"做成可打分的树并支持多意图，把"工具调用"做成显式抽取+校验，把"上下文"做成带引用编号的证据块。

---

## 3. 借鉴项明细

### 3.1 P0-① 工具参数抽取三态校验（Stage 7 直接抄）

**Ragent 的做法**（`rag/core/mcp/LLMMcpParameterExtractor.java`）：

- 用 LLM 按工具的 JSON Schema 抽取参数，结果严格分三类：
  - `SUCCESS`：参数合法，填默认值后调用工具；
  - `NEED_CLARIFICATION`：必填参数缺失（模型省略 key 或显式 null 均视为缺失）→ 不调用，向用户追问；
  - `FAILED`：JSON 解析失败 / 值类型非法 / 枚举值非法 → 一律不调用（杜绝"garbage 进工具"和"静默丢过滤条件"）。
- 抽取请求固定 `temperature=0.1, topP=0.3`，保证低随机；容错 markdown code fence 与 `{results: [...]}` 嵌套包装。
- 每个工具可挂自定义抽取 prompt（挂在意图节点的 `paramPromptTemplate` 上），默认用通用模板。

**ArtAgent 现状**：general_agent 用 LangChain tool calling，schema 由函数签名自动生成，缺"必填缺失→澄清"和"值非法→拒绝调用"两档，模型说什么就是什么。

**落点**：在 `src/tools/` 上加一层 `ToolInvocationGuard`：包装每个工具（semantic_search / image_lookup / web_search / read_page_image / knowledge…）的 schema → LLM 抽取 → 按 schema 校验 → 返回三态。Stage 7 工具设计讨论时直接以这个模型为骨架。

### 3.1 P0-② 意图树 + 多意图并行

**Ragent 的做法**（`rag/core/intent/DefaultIntentClassifier.java`、`IntentNode.java`、`vector/strategy/IntentParallelRetriever.java`）：

- 意图维护成树（DOMAIN/CATEGORY/TOPIC 三级），叶子节点带 `id / path / description / examples`，节点可声明类型：
  - `KB`：挂知识库 collection 列表 → 路由到哪些库检索；
  - `MCP`：挂 `toolId` + 参数抽取 prompt → 路由到哪个工具；
  - `SYSTEM`：挂自定义 prompt → 直答不检索。
- 分类：把所有叶子一次性发给 LLM 打分，输出 `[{id, score, reason}]`，`topKAboveThreshold(topN, minScore)` 过滤；失败/畸形 JSON 一律回落"无意图"。
- 检索：查询向量只 embedding 一次，各意图节点并行检索（各自可配独立 `topK`），失败的单意图跳过并记日志。

**ArtAgent 现状**：`classify` 单选路由到 4 个专用分支（comparison/timeline/recommendation/general），无打分、无多意图、节点不挂工具。

**落点**：Stage 7 把 classify 升级为打分式意图树；工具和知识库都建成叶子节点，意图层自然成为"该调用哪个工具/哪个库"的路由层。本地版本先用静态 JSON 定义树（`src/agent/intent_tree.json`），不用上数据库。

### 3.1 P0-③ 查询改写 + 多问题拆分

**Ragent 的做法**（`rag/core/rewrite/MultiQuestionRewriteService.java`）：

- 两步：先规则归一化（术语映射，可关闭）→ 再 LLM 改写 + 拆分为子问题（`RewriteResult(rewrittenQuestion, subQuestions)`）。
- 只带最近 2 轮历史做改写上下文；用 `Tier.FAST` 档模型；LLM 失败回落归一化原文，永不报错。

**ArtAgent 现状**：`contextualize` 只做指代消解（"这幅"→具体对象），没有多问题拆分；"顺便推荐几幅类似的"这类复合问题只能走一个意图。

**落点**：在 contextualize 之后加 `rewrite_and_split`，子问题 × 意图打分后并行检索（喂给 P0-②）。这是检索质量提升里改动最小、收益最直接的一项。

### 3.2 P1-① 通道归因日志 + 统一 chunk key

**Ragent 的做法**（`rag/core/retrieval/postprocessor/FusionPostProcessor.java`、`ChannelAttribution.java`）：

- 融合/去重/引用反查共用同一把 key：`id` 优先，缺省 SHA-256(文本)（明确不用 `String.hashCode()` 防碰撞）。
- RRF 融合截断后记录"送入 Rerank 的候选按通道分布"（`ChannelAttribution.countByChannel`），观察每个通道真实贡献，新接入的低可信通道权重是否合理一目了然。

**ArtAgent 现状**：加权 RRF 已上线（`src/retrieval/hybrid.py` 的 `CHANNEL_WEIGHTS`），但没有归因统计，权重调没调对只能靠感觉。

**落点**：在 `_rrf_fuse` 与 `_rerank_fused` 之间加一条计数日志（每个 source 进池多少条、进 Rerank 多少条）。这就是你正在做的"加权 RRF 是否有效"测试的直接观测面。

### 3.2 P1-② 引用编号 + 历史清洗

**Ragent 的做法**（`rag/core/source/CitationContextEnricher.java`、`CitationMarkup.java`）：

- 上下文内部用 `data-ragent-doc-id` 标记证据块，来源定稿后统一替换为模型可见的 `ref="N"`——内部 docId 永远不暴露给模型；
- 回答行内引用 `[N](#cite-N)` 随消息落库；下一轮喂历史前 `CitationMarkup.strip()` 清掉，防止上一轮编号污染本轮引用。

**ArtAgent 现状**：检索结果带 source 元数据，prompt 里没有统一编号引用机制。

**落点**：synthesizer / general 组装 prompt 时给证据块编号 [N]；历史清洗在记忆读取时做。改动小、对"答案可追溯"体验提升明显。

### 3.2 P1-③ 编排兜底短路径

**Ragent 的做法**（`StreamChatPipeline.java`）：

- 歧义检测（`IntentGuidanceService` + `AmbiguityLLMChecker`）→ 直接向用户反问，不发检索；
- 全部意图为 system → 跳过检索直接答；
- 检索为空 → 明确回复"未检索到相关内容"，不硬编。

**ArtAgent 现状**：有 reflection / web_fallback；general 分支的空检索文案与歧义反问行为需自查。

**落点**：graph 里补 `ambiguity → ask_user` 节点；与 P0-① 的 NEED_CLARIFICATION 共用同一"追问"通路。

### 3.3 P2-① 增量会话摘要记忆

**Ragent 的做法**（`rag/core/memory/JdbcConversationMemorySummaryService.java`）：

- 第 N 轮起，assistant 回复后异步触发摘要（独立线程池 + 每会话锁防并发）；
- 增量式：只摘要 `afterId → cutoffId` 区间并与旧摘要拼接（约半窗重叠滑出后才再次摘要），不是每轮全量重算；
- 摘要以 system 消息经上下文模板注入。

**ArtAgent 现状**：S5 只有偏好库；conversations.db 存原始消息，无压缩。

**落点**：conversations.db 加 `conversation_summary` 表；FastAPI 用后台任务 + 每会话 asyncio 锁实现（本地单机不需要 Redisson）。

### 3.3 P2-② 模型路由与降级

**Ragent 的做法**（`infra-ai/chat/RoutingLLMService.java`、`LlmFirstPacketProbe.java`）：

- `ModelSelector` 按能力×档位×偏好选候选，`ModelHealthStore` 记健康状态，`executeWithFallback` 依次尝试，全失败给统一文案；
- 流式首包（TTFT）超时探测单独抽成 bean 便于 trace。

**ArtAgent 现状**：`src/utils/llm.py` 固定 DashScope 系（get_llm / get_deterministic_llm），无多供应商降级。

**落点**：若将来有多家 key，抽 `route_llm(request, tier)` 包装失败自动换路；只有一家时跳过。

### 3.3 P2-③ 定时刷新远程知识源

**Ragent 的做法**（`knowledge/schedule/ScheduleRefreshProcessor.java`）：远程文件/网页按计划拉取重入库，任务租约锁 + 心跳防重复执行。

**ArtAgent 落点**：对应你设想的"本地核心库 + 实时 API 增量"——按需查询做成工具（已有 web_search 雏形），curated 源再做"每日定点刷新"；本地单机不需要租约锁。

---

## 4. 落地路线建议

1. **Stage 7 第一步（骨架）**：P0-① 工具三态校验 + P0-② 意图树多意图。两者共同定义"工具该不该调用、参数合不合法、缺了怎么追问"。
2. **同步小改（各 0.5 天）**：P1-① 归因日志（给加权 RRF 测试提供观测面）、P1-② 引用编号。
3. **紧接着**：P0-③ 改写拆分（配合多意图并行检索）、P1-③ 兜底短路径。
4. **按需**：P2-① 记忆摘要（对话变长后再做）、P2-② 模型降级、P2-③ 定时刷新。

## 5. 关键参考文件索引（ragent-ref 内）

- 工具调用：`bootstrap/.../rag/core/mcp/LLMMcpParameterExtractor.java`、`McpToolRegistry.java`、`McpClientToolExecutor.java`
- 意图路由：`bootstrap/.../rag/core/intent/DefaultIntentClassifier.java`、`IntentNode.java`、`IntentResolver.java`
- 检索：`bootstrap/.../rag/core/retrieval/channel/*.java`、`postprocessor/FusionPostProcessor.java`、`ChannelAttribution.java`、`vector/strategy/IntentParallelRetriever.java`
- 编排：`bootstrap/.../rag/service/pipeline/StreamChatPipeline.java`
- 记忆：`bootstrap/.../rag/core/memory/JdbcConversationMemorySummaryService.java`
- 引用：`bootstrap/.../rag/core/source/CitationContextEnricher.java`、`CitationMarkup.java`
- 模型路由：`infra-ai/.../chat/RoutingLLMService.java`、`LlmFirstPacketProbe.java`
- 定时刷新：`bootstrap/.../knowledge/schedule/ScheduleRefreshProcessor.java`
