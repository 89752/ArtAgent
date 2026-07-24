# ArtAgent 评估结果

> 生成时间：2026-07-24 14:21:40

## 1. 意图分类

**准确率 96.0%**（48/50） · Macro-F1 0.962

| 意图 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| comparison | 1.00 | 1.00 | 1.00 | 11 |
| timeline | 1.00 | 1.00 | 1.00 | 12 |
| recommendation | 0.83 | 1.00 | 0.91 | 10 |
| general | 1.00 | 0.88 | 0.94 | 17 |

**混淆矩阵**（行=真实，列=预测）

| gold \ pred | comparison | timeline | recommendation | general |
|---|---|---|---|---|
| **comparison** | 11 | 0 | 0 | 0 |
| **timeline** | 0 | 12 | 0 | 0 |
| **recommendation** | 0 | 0 | 10 | 0 |
| **general** | 0 | 0 | 2 | 15 |

**误分类样本**

- `general` → `recommendation`：推荐一本介绍伦勃朗的书
- `general` → `recommendation`：喜欢梵高的人一般也喜欢莫奈吗

## 2. 已知项检索

**Recall@5 = 64.0%**（16/25）

> 从 SemArt 随机抽画作，用其描述中段片段作 query，检验原画能否命中 semantic_search 的 top-5。全自动标注，衡量向量检索质量。
