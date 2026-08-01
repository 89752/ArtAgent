# ArtAgent 评估结果

> 生成时间：2026-08-01 10:08:59

## 1. 意图分类

**准确率 96.0%**（48/50） · Macro-F1 0.964

| 意图 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| comparison | 0.92 | 1.00 | 0.96 | 11 |
| timeline | 1.00 | 0.92 | 0.96 | 12 |
| recommendation | 1.00 | 1.00 | 1.00 | 10 |
| general | 0.94 | 0.94 | 0.94 | 17 |

**混淆矩阵**（行=真实，列=预测）

| gold \ pred | comparison | timeline | recommendation | general |
|---|---|---|---|---|
| **comparison** | 11 | 0 | 0 | 0 |
| **timeline** | 0 | 11 | 0 | 1 |
| **recommendation** | 0 | 0 | 10 | 0 |
| **general** | 1 | 0 | 0 | 16 |

**误分类样本**

- `timeline` → `general`：莫奈晚年为什么画风变得更抽象
- `general` → `comparison`：喜欢梵高的人一般也喜欢莫奈吗

## 2. 已知项检索

**Recall@5 = 70.0%**（14/20）

> 从 SemArt 随机抽画作，用其描述中段片段作 query，检验原画能否命中 semantic_search 的 top-5。全自动标注，衡量向量检索质量。
