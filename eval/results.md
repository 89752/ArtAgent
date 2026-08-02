# ArtAgent 评估结果

> 生成时间：2026-08-02 11:03:23

## 1. 意图分类

**准确率 100.0%**（50/50） · Macro-F1 1.000

| 意图 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| comparison | 1.00 | 1.00 | 1.00 | 11 |
| timeline | 1.00 | 1.00 | 1.00 | 12 |
| recommendation | 1.00 | 1.00 | 1.00 | 10 |
| general | 1.00 | 1.00 | 1.00 | 17 |

**混淆矩阵**（行=真实，列=预测）

| gold \ pred | comparison | timeline | recommendation | general |
|---|---|---|---|---|
| **comparison** | 11 | 0 | 0 | 0 |
| **timeline** | 0 | 12 | 0 | 0 |
| **recommendation** | 0 | 0 | 10 | 0 |
| **general** | 0 | 0 | 0 | 17 |

## 2. 已知项检索

**Recall@5 = 76.0%**（19/25）

> 从 SemArt 随机抽画作，用其描述中段片段作 query，检验原画能否命中 semantic_search 的 top-5。全自动标注，衡量向量检索质量。
