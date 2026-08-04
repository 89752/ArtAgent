# ArtAgent 工具带升级方案（2026-08）

> 版本：v1.0 ｜ 编制：Codex（资深项目经理视角）｜ 日期：2026-08-03
> 关联文档：[成熟 Agent 应用升级方案](./ArtAgent-成熟Agent应用升级方案-2026-08.md)（总纲）、[Agent 评估报告-2026-08-03](./ArtAgent-Agent评估报告-2026-08-03.md)（差距证据）、[2.0 实施方案](./ArtAgent-2.0-实施方案.md)（Stage 7 预留能力）
> 定位：本方案是升级总纲下的**工具带专项**，只解决"可调用工具"的广度、深度与选择准确率问题。

---

## 0. 摘要

**目标**：把工具带从"查询器"升级为"能力器"——工具类型从纯"读/查"扩展到"算/统计/对比读图/外部知识"，同时把意图路由与工具带对齐，使工具选择通过率从 82.6% 提升到 ≥90%。

**总工时**：约 21.5 人日（P0 约 5 人日 / P1 约 11 人日 / P2 约 5.5 人日），按 3~4 周推进。

**一句话现状**：工具基础设施（守卫、治理、技能、观测、评估）已达成熟应用水准，但工具本身仍单薄——18 项绑定全部是"读/查"型，且意图树 tool 叶子只有 3 个，与工具带严重脱节（这是工具选择 82.6% 且失败模式集中的直接原因）。

---

## 1. 现状盘点（2026-08-03 代码实测）

### 1.1 工具带（18 项绑定）

| 类别 | 工具 | 性质 |
|---|---|---|
| 检索（3） | semantic_search / exact_lookup / query_painter_knowledge | 读 |
| 图像（2） | image_lookup（含 analyze）/ read_page_image | 读 |
| 联网（1） | web_search | 读 |
| 能力管线（3） | compare_subjects / timeline_by_periods / recommend_with_exclusions | 读 + 内部偷藏 LLM |
| 记忆（3） | remember / recall / forget | 写（轻） |
| 收藏（3） | save_collection / list_collections / list_preferences | 写（轻） |
| 技能（3） | skill_artwork_deep_analysis / skill_document_summary / skill_exhibition_research | 组合 |

### 1.2 基础设施（已具备，不动）

| 设施 | 落点 |
|---|---|
| 参数三态守卫 | `src/tools/guard.py`（SUCCESS / NEED_CLARIFICATION / FAILED） |
| 执行治理 | `src/utils/governance.py`（超时 60s / 重试 / 失败结构化返回） |
| 技能执行器 | `src/skills/loader.py`（steps_json + output_schema 校验 + 受限工具集 + 步数上限） |
| 调用观测 | `src/observability/runs.py`（tools_json + 工具分布指标） |
| 工具选择评估 | `eval/sets/tool_testset.json`（30 条）+ `eval/agent_eval_v2.py` |

### 1.3 评估证据（本方案的问题清单来源）

工具选择（2026-08-03 报告）：23 条有效，19 通过 = **82.6%**。失败样本：

| 样本 | 期望工具 | 实际行为 | 根因 |
|---|---|---|---|
| [16] 深度分析《向日葵》 | skill_artwork_deep_analysis | exact_lookup + semantic_search×2 + web_search | 技能无意图叶子，从未被建议 |
| [19] 什么是线性透视 | 零工具 | semantic_search×2 | 知识问答检索开关不稳 |
| [20] 今天北京天气 | web_search | 零工具 | 联网触发未命中 |
| [23] 油画与丙烯区别 | 零工具 | semantic_search×5 + web_search | 知识问答过度检索 |

行为化 multi_intent：真实执行多轮工具但未调用 compare_subjects / recommend_with_exclusions —— 与 [16] 同根因（能力工具无意图叶子）。

### 1.4 最新评估证据（2026-08-03 深夜重跑）

工具选择集 30 条全量重跑后为 **27/30（90%）**。此前修复已确认生效：

- 零工具直答生效：[什么是线性透视] [油画颜料和丙烯颜料有什么区别] [什么是印象派] 均直接回答、零工具 ✅；
- 文档通道生效：[莫奈晚年视力问题] [莫奈在葛列尔画室学习时身边有哪些同学] 均走 `semantic_search` ✅。

剩余 3 条失败**同源**：No-tool 规则是"黑名单式"列举，模型把实时信息、知识事实、结构化比较一并短路。

| 失败用例 | 期望工具 | 实际行为 | 根因 |
|---|---|---|---|
| 今天北京天气怎么样 | web_search | 零工具（reflection RETRY 后由 web_fallback 兜底拉到天气，但工具链无 web_search） | 时效类被零工具规则短路 |
| 巴洛克和洛可可的装饰风格有什么不同 | compare_subjects | 零工具 | classify 已识别 comparison（0.95）但路由未真正生效，general 直接回答 |
| 印象派这个名称是怎么来的 | semantic_search + web_search | 零工具 | 知识事实被零工具规则短路（此前为通过用例，本轮回归） |

---

## 2. 差距诊断：三个"单薄"

### 2.1 广度单薄：全是"读"，没有"算"

18 项绑定中，真正产生"新信息"的只有检索与读图；没有任何本地计算（颜色分析、统计聚合）、对比读图、外部馆藏/百科查询。用户问"印象派哪个时期作品最多""这两幅画的笔触差异"时，Agent 只能用 semantic_search 硬查或凭模型知识硬答。

### 2.2 选择单薄：意图树与工具带脱节

`src/agent/intent_tree.py` 的 tool 叶子只有 3 个（image_lookup / read_page_image / web_search），**compare_subjects、timeline_by_periods、recommend_with_exclusions、skill_*、remember、save_collection 均无意图叶子**。`intent_tool_suggestions` 只能建议这 3 个工具，其余 15 项完全靠模型盲选——这是选择失败三模式（技能未选中、能力工具未选中、知识问答开关不稳）的共同根因。

### 2.3 深度单薄：参数与返回粗糙

- `semantic_search(query, top_k)` 无结构化过滤（author/school/timeframe/source），精确意图只能靠自然语言硬凑，推高过度检索概率；
- `exact_lookup` 的 timeframe 用字符串包含匹配，"1900-1950" 会漏掉边界数据；
- 工具返回无统一截断（skills runner 有 2000 字符限制，general 工具没有），超长 JSON 直接灌上下文；
- 能力工具内嵌 LLM（compare_subjects 内调相关性过滤、recommend_with_exclusions 内调特征提取），违反"工具内不藏 LLM"纪律，调用不可预测、成本不可见。

---

## 3. 升级目标（量化）

| # | 目标 | 当前 | 目标 |
|---|---|---|---|
| T1 | 工具选择通过率（30 条集） | 82.6% → 90%（2026-08-03 深夜 27/30） | 稳定 ≥90%（回归确认） |
| T2 | 意图树 tool 叶子与工具带对齐率 | 3/18 | 100%（自动生成） |
| T3 | 新增工具类型 | 0 | ≥4（计算/统计/对比读图/外部知识） |
| T4 | 知识问答零工具命中（[19][23] 类） | 2/4 失败 | 100% 通过 |
| T5 | 技能选择（[16] 类） | 失败 | 100% 通过 |
| T6 | 新工具纯单测 + 守卫 schema 覆盖 | — | 100% |
| T7 | 检索基线不回归 | Recall@5 = 88% | ≥88% |

---

## 4. 分阶段实施计划

### P0：选择机制修复（第 1 周，约 5 人日）

**目标：先让模型"选得对"，再谈"工具多"。**

| # | 任务 | 关键内容 | 工时 |
|---|---|---|---|
| P0-1 | 意图树叶子自动对齐 | 从 `GENERAL_TOOLS` + `register_skills()` 自动生成 tool 叶子（含每个工具的 description 与触发示例）；`intent_tool_suggestions` 随之覆盖全部工具；保留手工 examples 覆盖 | 2d |
| P0-2 | 知识问答零工具叶子 | 新增 `system_knowledge` 叶子（"什么是 X"、常识性区别如"油画 vs 丙烯" → 不检索直答）；**领域比较（画家/画作/风格）仍走 comparison 路由，不得直答**；SYSTEM_PROMPT 的 Tool Selection Rules 补硬规则与负例 | 1d |
| P0-3 | 技能描述触发友好化 | 3 个 SKILL.md 的 description 补触发词（深度分析/详细解读/评析/总结文档/调研）；技能叶子示例同步 | 0.5d |
| P0-4 | semantic_search 结构化过滤 | 增加可选 `filters={author, school, timeframe, source}`；HybridRetriever 侧按 metadata 过滤；无 filters 行为不变（eval 口径锁定） | 1.5d |

**验收**：tool_testset 重跑通过率 ≥90%；[16][19][20][23] 四条全部通过；multi_intent 行为用例恢复调用 compare_subjects / recommend_with_exclusions。

### P1：工具类型扩展（第 2~3 周，约 11 人日）

**目标：工具带从"查询器"升级为"能力器"。** 新工具设计草案见 §5。

| # | 工具 | 类型 | 关键内容 | 工时 |
|---|---|---|---|---|
| P1-1 | color_analysis | 计算 | 本地 K-means 主色调 + 直方图 + 明度对比 + 构图网格；零 LLM/API | 3d |
| P1-2 | aggregate_stats | 统计 | 按流派/时期/技法分组计数与占比，支持过滤 | 1.5d |
| P1-3 | compare_images | 对比读图 | 两幅画同帧喂视觉模型，结构化对比笔触/色彩/构图 | 2d |
| P1-4 | museum_search | 外部知识 | Met API 免 key（CC0 公共领域），query/artist/title 检索 + 详情 | 2d |
| P1-5 | wiki_lookup | 外部知识 | Wikidata/Wikipedia 摘要 + 关键事实 + 来源 URL | 1.5d |
| P1-6 | 收藏补全 | 管理 | delete_collection / get_collection / rename_collection | 1d |

**验收**：每个新工具带纯单测（无 LLM/网络）+ 守卫 schema 单测；tool_testset 补充对应用例（颜色分析、统计、馆藏、百科各 ≥3 条）；README 工具表同步。

### P2：机制收尾（第 4 周，约 5.5 人日）

| # | 任务 | 关键内容 | 工时 |
|---|---|---|---|
| P2-1 | 能力工具去 LLM 化 | compare_subjects / recommend_with_exclusions 内部 LLM 调用上移编排层节点（可观测、可审计），或返回体显式带 `llm_used` / 估算成本字段；二选一，倾向前者 | 3d |
| P2-2 | 工具输出统一截断 | `governed_invoke` 对超长返回统一压缩（对齐 skills runner 的 2000 字符上限），保留结构化 JSON 形状 | 1d |
| P2-3 | tool_testset 回归门禁 | 纳入 `scripts/regression.ps1` + CI nightly；额度受限时提供离线 mock 路径 | 1.5d |

**验收**：P2-1 后工具调用链在 agent_runs 中可完整归因（含 LLM 子调用）；P2-2 后单条 ToolMessage 有大小上限；P2-3 后工具选择集进回归。

---

## 5. 新工具设计草案

### 5.1 color_analysis（本地计算，零成本）

```text
color_analysis(title?, author?, image_path?, top_k=1)
→ [{title, author, date, image_file,
    dominant_colors: [{hex, ratio}],        # K-means k=5，PIL/numpy 实现，不引 sklearn
    brightness_contrast: "high|medium|low",
    saturation: "vivid|muted",
    composition_grid: "balanced|dynamic|symmetrical"}]   # 九宫格显著性启发式
```

- 实现要点：`Image.quantize` 或手写 K-means；全部确定性输出，可单测；
- 成本：约 1s CPU，免费；**这是"工具带从读到算"的第一块招牌**；
- 边界：算法只输出结构性数值，不做审美判断（审美判断留给对话模型结合分析结果组织）。

### 5.2 aggregate_stats（结构化统计）

```text
aggregate_stats(group_by: "school|timeframe|technique",
                filter?: {author, school, timeframe}, top_k=10)
→ {group_by, groups: [{value, count, ratio}], total, sample_titles: [...]}
```

- 解决"印象派哪个时期作品最多""藏画中哪种技法最常见"类问题，替代 semantic_search 硬查；
- 复用 `StructuredTableRetriever` schema 角色列，与数据集切换兼容。

### 5.3 compare_images（对比读图）

```text
compare_images(title_a, title_b, focus: "brushwork|color|composition|general" = "general")
→ {a: {title, author, date, image_file},
   b: {...},
   comparison: "视觉模型结构化对比文本（按 focus 分节）"}
```

- 实现：一次视觉调用，两图同帧 + focus 提示词（复用 `_ANALYSIS_FOCUS_PROMPTS` 扩展）；
- 成本：每次 1 次视觉 API 调用，description 中显式标注成本规则；
- 定位失败时返回 `{success: false, error}`，不中断整轮。

### 5.4 museum_search（外部馆藏）

```text
museum_search(query?, artist?, title?, has_image=true, top_k=5)
→ [{title, artist, date, medium, department, image_url, object_url, is_public_domain}]
```

- 实现：Met API `GET /public/collection/v1/search` + object 详情（top_k 小，N+1 可接受）；80 req/s 限速，内部令牌桶；
- 降级：超时/失败返回结构化错误，Agent 可转 web_search；
- 价值：把知识边界外扩到 20 世纪与馆藏信息，补 SemArt 8~19 世纪覆盖缺口。

### 5.5 wiki_lookup（百科兜底）

```text
wiki_lookup(entity: "画家/流派/术语英文名")
→ {entity, summary, key_facts: [...], source_url, lang}
```

- 实现：Wikipedia REST API（免费、无需 key），摘要取前 N 句 + 关键事实抽取；
- 与 web_search 分工：wiki_lookup 适合"定义/生平/流派"类，web_search 适合"时效/价格/展览"类，prompt 中写明。

### 5.6 收藏管理补全

```text
delete_collection(name)          → 删除确认
get_collection(name)             → 单清单内容
rename_collection(old, new)      → 重命名确认
```

与 save_collection / list_collections 对称，均走 `src/memory/collections.py`。

---

## 6. 意图树对齐设计

### 6.1 自动生成 vs 手工

- 基础叶子从 `GENERAL_TOOLS` + `register_skills()` 自动生成：`tool_<name>`，description 取工具 docstring 首段；
- 手工层只维护**触发示例**（每个工具 2~3 条中文示例，直接取自 tool_testset 语料）；
- `system_knowledge`（零工具）、`system_greeting`（已有）为手工叶子；
- 生成函数 `build_tool_leaves()` 放 `src/agent/intent_tree.py`，单测断言"工具带 ↔ 叶子 1:1 对齐"（T2 门禁）。

### 6.2 建议注入

- `intent_tool_suggestions` 已按分数取 top_n 注入 system 消息，无需改动逻辑；
- 建议数从 3 提到 5（工具变多后 3 条不够），阈值 min_score 保持 0.3；
- 注意 token 成本：叶子全量进分类 prompt 会变长，提示词按"id + 一句话 + 1 示例"压缩，200 字符/叶子封顶。

### 6.3 路由决策层（Route Decision Layer，2026-08-03 新增）

**问题定位**：`classify` 只做意图诊断（输出进日志/UI，不参与路由），`rag_gate` 只对寒暄短路，"要不要调工具"最终由 general 分支的 ReAct 模型自行决定——SYSTEM_PROMPT 的 No-tool 规则是黑名单式列举，导致 1.4 节三条失败。

**对标结论**（OpenAI Agents SDK triage+handoff、LangGraph supervisor+structured output、AutoGen selector）：成熟 agent 把"直答 vs 检索 vs 联网 vs 专用工具"做成**独立的、受约束的、可观测的路由决策点**，而不是在 ReAct 里靠模型自觉；决策输出用有限枚举 + reason，并用正反例训练提示词，配合 evaluator 兜底。

**设计**：

1. `classify` 升级为"路由决策节点"：
   - 输出结构化枚举 `route ∈ {direct, rag, web, comparison, timeline, recommendation, clarify, tool_<name>}`，附带 `confidence` 与 `reason`；
   - graph 用条件边真正分流（`classify → route`，替代"只看日志"），comparison/timeline/recommendation 直接进对应分支，direct/rag/web 走对应节点；
   - 路由结果写入 trace，纳入 `agent_runs` 与新增 `route_diag` 评估指标（各路由误判率）。
2. 确定性预筛（在 classify 之前，规则命中即短路，不花 LLM）：
   - direct 白名单：寒暄 / 算术 / 常识定义（现有 `_GREETING_RE` 扩展 + "什么是 X / 1+1 类"）；
   - web 强制：时效词（今天 / 天气 / 新闻 / 最新 / 价格 / 几点 / 目前）；
   - comparison 强制：比较动词（对比 / 区别 / 差异 / 差别 / 有什么不同 / 哪个更）；
   - timeline / recommendation 同样保留现有动词命中；
   - 未命中才走 LLM classify。
3. No-tool 规则改**白名单**：
   - 只允许：常识 / 定义 / 算术 / 寒暄 直答；
   - 知识事实（术语来源、画家生平细节）默认 rag（`semantic_search`）；
   - 实时信息强制 web；结构化任务（比较/时间线/推荐/收藏/记忆）强制对应工具。
4. 负例进 prompt（OpenAI triage 实践：tighter trigger + negative examples）：
   - "今天北京天气怎么样" → 必须 web_search，不允许直答；
   - "印象派这个名称是怎么来的" → 必须检索，不允许凭记忆直答；
   - "巴洛克和洛可可的装饰风格有什么不同" → comparison 分支，不允许 general 直答。
5. 兜底泛化：reflection RETRY 不再只升 `web_fallback`，改为"工具升级"节点（按 route 原始意向先补 rag、再补 web），避免 direct 误判后只能联网兜底的单点路径。

**三个失败的对策**：

| 失败用例 | 期望 | 对策 |
|---|---|---|
| 今天北京天气怎么样 | web_search | 时效词预筛强制 web 路由；web 分支调用 `web_search` 工具（`web_fallback` 降级为备份，不作为主路径） |
| 巴洛克和洛可可的装饰风格有什么不同 | compare_subjects | comparison 路由真正生效（条件边 + 分支提示"必须调用 compare_subjects"）；同时扩展 `compare_subjects` 支持风格/流派对象（当前只面向画家/画作） |
| 印象派这个名称是怎么来的 | semantic_search + web_search | 白名单外默认 rag；SYSTEM_PROMPT 与 classify prompt 同时补负例 |

**验收**：tool_testset 三条失败用例重跑全过；新增 route_diag（direct/rag/web/comparison 误判率）；多轮/对抗不回归。

---

## 7. 质量保障与验收

### 任务级 DoD

1. 新工具：纯单测（无 LLM/无网络，秒级）+ 守卫 schema 单测 + 1 条 tool_testset 用例
2. 修改现有工具：eval 口径锁定（sources=["core"]、RERANK 开关不变）复跑无回归
3. 文档同步：README 工具表、prompts 工具说明、本方案勾选

### 回归清单（每次合入前）

```powershell
scripts/regression.ps1          # 纯单测 + Web/API + 渲染
python eval/agent_eval_v2.py --tool-only   # 工具选择集（额度允许时）
python eval/agent_eval_v2.py --retrieval-only  # Recall@5 ≥88%
```

### 期级验收

- P0 末：tool_testset ≥90%，[16][19][20][23] 全过，multi_intent 恢复
- P1 末：6 个新工具全部上线 + 单测全绿 + tool_testset 扩充至 ≥48 条
- P2 末：能力工具调用链可归因、ToolMessage 有上限、工具集进 CI 门禁

---

## 8. 范围护栏（明确不做）

- 不做 MCP 接线 / 插件市场（平台层，已有 `src/platform/mcp_client.py` 但不在本方案范围）
- 不做代码执行沙箱、浏览器自动化、语音/视频工具
- 不做非领域通用工具凑数（天气日历类仅作为 web_search 用例，不单设工具）
- 新工具一律走现有守卫 + `governed_invoke`，不引入旁路

---

## 9. 风险管理

| 风险 | 概率/影响 | 缓解 |
|---|---|---|
| 工具变多导致选择更差 | 中/高 | P0 先修选择机制再扩工具；tool_testset 门禁盯住；意图树叶子 1:1 对齐 |
| color_analysis 算法质量边界 | 中/低 | 明确定位"结构性数值"而非审美判断；输出可单测；质量不足时降级 image_lookup analyze |
| 外部 API（Met/Wikidata）不稳定 | 中/低 | 统一走 governed_invoke（超时/重试/降级）；description 写明失败可转 web_search |
| compare_images 视觉成本 | 中/中 | 每次仅 1 次调用；prompt 与 description 显式标注；top_k 不放开 |
| 意图树 prompt 变长、分类变慢 | 中/中 | 叶子压缩格式 + 字符封顶；分类耗时纳入 agent_runs 观测，若劣化回退手工精简 |
| API 额度中断评估 | 高/中 | 工具集离线 mock 路径（P2-3）；评估分批跑，不依赖单次全量 |

---

## 10. 第一步建议

1. **周一**：P0-1 意图树叶子自动对齐 + P0-4 semantic_search filters（两者独立，可并行）；
2. **周二**：P0-2 / P0-3，然后重跑 tool_testset，先看 [16][19][20][23] 是否全过；
3. 通过后再开工 P1 的 color_analysis（最快见效的新工具）。

---

## 11. 实施状态（2026-08-04 更新）

> 本方案已全量实施（P0 + P1 + P2），代码与单测完成；涉及 DashScope/Jina 额度的
> 在线评估由用户按命令分批实跑确认。

### P0：选择机制与路由决策层 ✅

- P0-1 意图树叶子自动对齐：`src/agent/intent_tree.py` 新增 `get_tool_leaves()` /
  `all_leaves()`，从 `GENERAL_TOOLS` + `register_skills()` 1:1 生成；新增
  `tests/test_intent_tree.py::test_tool_leaves_align_with_general_tools`；
- P0-2 零工具白名单：新增 `system_knowledge` 叶子；SYSTEM_PROMPT 改为
  "白名单式直答 + 负例"（天气/术语来源/领域比较必须走工具）；
- P0-3 技能触发词：3 个 SKILL.md 的 description / when_to_use 补齐触发词；
- P0-4 semantic_search filters：工具层 `filters={author,school,timeframe,source}`，
  HybridRetriever 透传 + 向量结果后置过滤（多取候选防漏召回）；
- §6.3 路由决策层：classify 输出 `route ∈ {direct,rag,web,comparison,timeline,
  recommendation,tool:<name>}` + reason；确定性预筛（寒暄/定义/算术→direct、
  时效词→web、强比较动词→comparison、演变→timeline、推荐→recommendation）；
  graph 条件边直答分流；general 分支注入路由强指令；reflection RETRY →
  `tool_upgrade`（先本地证据再联网）；`route_diag` 用例集 + 报告分节 + `--route`；

### P1：工具类型扩展 ✅

- `color_analysis`（本地 K-means 主色调/明度对比/饱和度/构图网格，零成本）；
- `aggregate_stats`（按 school/timeframe/technique/author 分组计数与占比）；
- `compare_images`（两幅本地画同帧视觉对比，一次视觉调用）；
- `museum_search`（Met 开放馆藏，CC0）；
- `wiki_lookup`（中/英维基摘要，自动选语言）；
- 收藏管理补全：`get_collection` / `delete_collection` / `rename_collection`；
- tool_testset 扩充至 48 条（新工具各 ≥3 条 + 收藏 CRUD 3 条）；

### P2：机制收尾 ✅

- P2-1 能力工具可归因：`compare_subjects` / `recommend_with_exclusions`
  返回体带 `llm_used` / `llm_calls`（对比证据按对象计次、推荐特征提取计次）；
- P2-2 工具输出统一截断：`governed_invoke` 按 `TOOL_OUTPUT_MAX_CHARS`（默认 2000）
  递归压缩，保留 JSON 形状 + `truncated` 标记；
- P2-3 回归门禁：`scripts/regression.ps1` 增加新工具/路由单测与
  "工具带 ↔ 意图树叶子 1:1" 对齐检查；`.github/workflows/test.yml` 同步；

### 待用户实跑验证（不消耗额度优先，按需分批）

```powershell
# 三条旧失败用例 + 新工具用例（小批量，额度有限时单条跑）
python eval/agent_eval_v2.py --tools --limit 1 --offset 15 --append
python eval/agent_eval_v2.py --tools --limit 1 --offset 18 --append
python eval/agent_eval_v2.py --tools --limit 1 --offset 23 --append
python eval/agent_eval_v2.py --tools --limit 18 --offset 30 --append
# 路由决策诊断（15 次轻量分类，不跑整轮对话）
python eval/agent_eval_v2.py --route --append
# 全量工具选择（48 条，额度允许时）
python eval/agent_eval_v2.py --tools --append
```
