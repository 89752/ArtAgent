# 评估（Evaluation）

评价体系设计见 `docs/ArtAgent-评价体系设计-2026-08-03.md`。
统一数据集：**每条用例只跑一次 Agent，多维指标从同一次运行派生**
（答案质量 / 事实命中 / 工具选择 / 行为 / 状态 / 对抗安全）。

## 结构

```text
eval/
├─ agent_eval_v2.py          # 主评估入口（自包含）
├─ agent_eval_report.md      # 最新报告
├─ metrics_history.jsonl     # 历史趋势（每次运行追加一条）
├─ sets/                     # 全部 JSON 测试数据集
│  ├─ cases.json             # 统一单轮用例（60 条，多维标记见下）
│  ├─ multi_turn_golden.json # 10 条多轮轨迹
│  └─ intent_diag.json       # 40 条意图诊断（规则分类器，软信号，非验收）
│  └─ painting_analysis.json # 绘画分析引擎评测（20 条，独立脚本，image_path 待补充）
└─ ArtAgent-ABtest-2026-08-03.md  # A/B 测试汇总
```

每条 cases.json 用例通过字段声明参与哪些维度：

| 字段 | 维度 | 判定方式 |
|---|---|---|
| `judge: true` | 答案质量 1-5 | LLM 裁判（开放题才开，省 token） |
| `gold_facts` | 事实命中 | 答案关键词包含（封闭题不调裁判） |
| `expected_tools` | 工具选择 | 预期工具是否被调用 / 空数组=要求零工具 |
| `behavior` | 行为 | ask / no_tools / web_fallback / tools_any |
| `state_check` | 状态落库 | 记忆偏好表 / 收藏表 |
| `safety_expect` | 对抗与安全 | LLM 裁判按期望行为判定 |

## 用法

```bash
python eval/agent_eval_v2.py                     # 全量：全部单轮用例 + 多轮 + 意图诊断
python eval/agent_eval_v2.py --answers 10        # 只跑质量分用例（配合 --offset 分块）
python eval/agent_eval_v2.py --facts             # 只跑事实命中用例
python eval/agent_eval_v2.py --tools             # 只跑工具选择用例
python eval/agent_eval_v2.py --behavior          # 只跑行为断言用例
python eval/agent_eval_v2.py --adversarial       # 只跑对抗与安全用例
python eval/agent_eval_v2.py --pr                # PR 门禁档：离线检索 20 + 意图诊断
python eval/agent_eval_v2.py --retrieval-n 100   # 只跑检索（离线）
```

## 约定

- 每条用例只跑一次 Agent；开放题（judge）才调 LLM 裁判，封闭事实题用关键词判定；
- 单条用例 API 失败只跳过不中断，报告标注有效样本数；429 速率限制
  自动退避重试（最多 3 次，间隔 5/15/30s），限流不再直接丢用例；
- 状态校验：记忆写 `preferences` 表、收藏写 `collections` 表，跑完自动清理；
- 检索指标：core · seed=42 · n=100 · Jina API 精排 + 词法通道（90% 基线，2026-08-05）；
- 裁判模型：llm-as-judge 默认用对话模型（temperature=0）；可配置 `JUDGE_MODEL /
  JUDGE_API_KEY / JUDGE_BASE_URL`（config.yaml 的 models.judge_*）使用更强/更稳的独立裁判，
  缺省回落对话模型配置；
- 跳过占比高时先恢复 API 可用性再重跑，勿把不完整报告当正式基线。

> 2026-08-17 对齐当前图：行为/工具用例中的 compare_subjects /
> timeline_by_periods / recommend_with_exclusions 已替换为对应技能
> （skill_art_comparison / skill_art_timeline / skill_art_recommendation）；
> 路由决策诊断（route_diag）随旧路由管线移除，意图诊断改为规则
> classify_intent（comparison / timeline / recommendation / general）。
>
> 2026-08-18 整体重建为统一数据集：单一 cases.json（60 条）
> 每条只跑一次
> Agent，多维指标同源派生，不再按维度重复跑；开放题 23 条走裁判，封闭事实题
> 15 条关键词判定，工具/行为/状态/安全维度随用例字段自动计算。事实答案均经
> data/core/artworks_core.csv 核对（旧集《Woman with a Pink》年份等错误已剔除）。
> 意图规则补充"变化/转变/之后/有什么不同"等关键词并排除书目推荐，
> intent_diag 40 条与规则 100% 一致；多轮 10 条、绘画分析 20 条。
> 技能 skill_document_summary 因需上传文档夹具暂未纳入用例。
> 旧 case_cache 与旧基线已作废，不可直接对比。
