# Agent 行为化 eval 指标报告

- 行为用例重复次数：2
- 事实题：6 题

## 1. 行为指标（触发率 / 平均耗时 / 平均工具轮次）

| 用例 | 触发率 | 平均耗时(s) | 平均工具轮次 |
|---|---|---|---|
| rag_gate | 100% | 9.7 | 0.0 |
| clarify | 100% | 6.3 | 0.0 |
| multi_intent | 100% | 79.4 | 2.5 |
| skill | 100% | 61.4 | 1.0 |
| memory_write | 100% | 17.6 | 1.5 |
| grain_paintings | 100% | 99.0 | 3.5 |
| collection | 100% | 18.6 | 2.0 |

行为整体通过率：14/14（100%）

## 2. 事实准确率（长尾/精确元数据，完整 Agent 链路）

| 问题 | 命中 |
|---|---|
| 《The Assumption Altarpiece》的作者是谁？ | ✅ |
| MET 收藏的《Cityscape》（2008.359.25）的作者是谁 | ✅ |
| 《Woman with a Pink》是哪一年创作的？ | ✅ |
| 《The Davis Madonna》是用什么材料画的？ | ✅ |
| 《Portrait of a Man with Gloves in Ha | ✅ |
| Gillis van Coninxloo 的森林风景画有什么特点？ | ✅ |

事实准确率：6/6（100%）
