# 评估（Evaluation）

评价体系设计见 `docs/ArtAgent-评价体系设计-2026-08-03.md`。
验收口径 = **最终答案质量 + 状态校验**；意图分类仅作诊断，不作主指标。

## 结构

```text
eval/
├─ agent_eval_v2.py          # 主评估入口（自包含）
├─ agent_eval_report.md      # 最新报告
├─ metrics_history.jsonl     # 历史趋势（每次运行追加一条）
├─ sets/                     # 全部 JSON 测试数据集
│  ├─ answer_golden.json     # 30 条分域黄金集（验收主集）
│  ├─ fact_testset.json      # 60 条 core 可验证事实题（build_fact_testset.py 生成）
│  ├─ behavior_testset.json  # 9 类行为用例
│  ├─ tool_testset.json      # 30 条工具选择（含零工具负例）
│  ├─ multi_turn_golden.json # 6 条多轮轨迹
│  ├─ adversarial.json       # 10 条对抗/安全用例
│  └─ intent_diag.json       # 40 条意图诊断（软信号，非验收）
└─ ArtAgent-ABtest-2026-08-03.md  # A/B 测试汇总
```

## 用法

```bash
python eval/agent_eval_v2.py --pr                # PR 门禁档：离线检索 20 + 意图诊断
python eval/agent_eval_v2.py                     # 全量（需 API 额度）
python eval/agent_eval_v2.py --answers 30 --facts --behavior-runs 3 --tools --multi-turn --adversarial --retrieval-n 100 --diag
python eval/agent_eval_v2.py --retrieval-n 100   # 只跑检索（离线）
python eval/agent_eval_v2.py --out eval/agent_eval_report_adversarial.md --adversarial   # 单跑对抗
```

## 约定

- 单条用例 API 失败只跳过不中断，报告标注有效样本数；
- 状态校验：记忆写 `preferences` 表、收藏写 `collections` 表，跑完自动清理；
- 检索指标：core · seed=42 · n=100 · Jina API 精排（88% 基线）；
- 跳过占比高时先恢复 API 额度再重跑，勿把不完整报告当正式基线。

