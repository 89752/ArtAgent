# 评估（Evaluation）

用可量化指标衡量 Agent 效果，而非"能跑通"级别的冒烟测试。

```bash
python eval/run_eval.py                 # 跑全部（意图分类 + 检索）
python eval/run_eval.py --no-retrieval  # 只跑意图分类（快，不加载向量库）
python eval/run_eval.py --retrieval-n 30
```

结果打印到控制台并写入 [`results.md`](results.md)。

## 指标 1 · 意图分类准确率

- **数据**：[`intent_testset.json`](intent_testset.json),50 条人工标注 query,四类意图均衡覆盖,含 10+ 条边界/歧义样本(如"推荐一本介绍伦勃朗的书"——含"推荐"但实为知识问答)。
- **方法**:对每条 query 调用真实的 `classify_intent` 节点(确定性 LLM,temperature=0),与金标签比对。
- **产出**:总准确率、各类 Precision/Recall/F1、混淆矩阵、误分类样本清单。
- **当前结果**:**准确率 96.0%,Macro-F1 0.962**。2 处误分类均为 general→recommendation,由偏好类关键词("推荐"/"喜欢")触发,是可解释的边界情况。

## 指标 2 · 已知项检索 Recall@k

- **方法**:从 SemArt 随机抽 N 幅画,取其**描述中段片段**(非标题)作 query,检验原画能否命中 `semantic_search` 的 top-k。
- **为何客观**:标签由数据本身自动生成(query 来自哪幅画,金标就是哪幅),无需人工标注,也无主观性。
- **当前结果**:**Recall@5 = 64%**。片段检索(而非标题精确匹配)下的合理量级,反映真实语义检索质量,而非虚高数字。

## 诚实性说明

- 指标在**真实系统组件**上运行(同一个 `classify_intent`、同一个 `semantic_search`),不是离线仿真。
- 检索评估用固定随机种子(`seed=42`)保证可复现。
- 标注集有意加入歧义样本,让准确率落在可信区间而非刻意的 100%。
