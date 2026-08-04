# ArtAgent A/B 测试汇总（2026-08-03）

本文件汇总项目内全部受控对比实验（A/B 或三路对照），含方法、数据、结果与结论。原始报告位于 `eval/` 下，本文件为合并口径。

## A/B-1 · 文本 embedding 模型：bge-small-en vs bge-m3

**动机**：用户上传中文扫描书后检索近乎失效，怀疑 embedding 通道是瓶颈。

**方法**：
- 语料：莫奈手稿 OCR **40 chunks**（1 份中文 PDF，16 页扫描件）；
- 测试集：20 题（8 条原文片段召回 + 12 条改写提问），gold 人工标注；
- 指标：raw cosine 排序（无重排）Recall@1/3/5、MRR、最佳 gold 排名；另跑一次 qwen3-rerank 全池重排对照。

**结果（raw cosine）**：

| 模型 | R@1 | R@3 | R@5 | MRR | 平均排名 |
|---|---|---|---|---|---|
| bge-small-en | 0.00 | 0.20 | 0.30 | 0.161 | 12.6/40 |
| bge-m3 | 0.80 | 0.95 | 1.00 | 0.879 | 1.4/40 |
| qwen3-rerank（全池重排） | 0.95 | 1.00 | 1.00 | 0.975 | 1.1/40 |

**结论**：bge-small-en 对中文用户文档近乎失效（20 题 R@1 = 0）；bge-m3 全面碾压且耗时几乎无差（9.0s vs 12.5s）。生产靠 reranker 兜底，但候选池一旦超过 40 chunks，粗排 embedding 决定进池内容，small-en 会把池子填满无关块。

**落地**：`user_pdf_text` 通道已切换 bge-m3 并重建索引；真实检索器复跑 20 题：**R@1 0.90 / R@3 1.00 / R@5 1.00 / MRR 0.950**，与 A/B 理论一致。

## A/B-2 · 检索源三路对照：本地向量库 vs Web 检索 vs 纯 LLM

**方法**：同一批问题（exact 精确元数据题 + general 知识题）分别走三条路：local（本地库 + 证据约束）、web（Tavily + 证据约束）、llm（纯模型知识），人工比对答案。

**核心发现**：
- **local 最稳但覆盖率有限**：exact 题能给出可验证答案（如《The Assumption Altarpiece》→ Moretto ✅、MET《Cityscape》→ anonymous ✅），但证据缺失时倾向"无法回答"（诚实但挫败）；
- **web 信息全但噪音高**：对伦勃朗风格演变、提香特点等知识题给出高质量结构化答案；但天气/商品类检索会拉回无关结果；
- **纯 LLM 流畅但有幻觉**：典型错误——《Portrait of a Man with Gloves in Hand》被答成 Frans Hals（实际应为 Rembrandt）、MET《Cityscape》被答成 Michele Oka Doner、祭坛画作者被答成提香（local 正确答 Moretto）。

**结论**：答案质量排序约为 **local ≥ web > llm（可靠性）；web > llm > local（覆盖率）**。生产应采用"本地库优先 + 证据不足时联网兜底 + 纯 LLM 仅作兜底"，且最终答案必须基于检索证据（当前图结构已按此设计：reflection 不通过 → web_fallback）。

## A/B-3 · 精排对照：API / 本地 / 粗排

**口径**：core 检索 Recall@5（n=100，seed=42），仅精排环节不同。

| 精排方案 | Recall@5 | 备注 |
|---|---|---|
| qwen3-rerank（DashScope API，历史） | 88.0% | 需 DashScope 额度 |
| **Jina Reranker v3.5（API，2026-08-03）** | **88.0%** | 单次 ~0.48s，当前生产配置 |
| bge-reranker-v2-m3（本地 CPU） | 85.0% | ~3.7s/查询 |
| 粗排（无精排） | 85.0% | 零额度零延迟 |

**结论**：Jina v3.5 API 与 qwen3-rerank 持平（88%），均高于本地/粗排 85%；精排在该指标上贡献约 +3pp。本地 Jina v3.5（listwise）质量更高但 CPU 上单轮需分钟级，仅适合无额度兜底。

## 附录 · 历史回归冒烟（`eval/ab_report.md`，2026-08-02）

非受控 A/B，属早期"统一 Agent 回归冒烟"：对比/推荐类样例走完整工具链（`compare_subjects` + `query_painter_knowledge` + `semantic_search` + `image_lookup` / `recommend_with_exclusions`）均产出结构化答案。用于确认子管线下沉为工具后链路可跑，不作为指标基线。

---

**一致性说明**：embedding 与精排两项为量化 A/B；检索源三路对照为定性人工比对（12 题级，未做逐题硬计数）；全部实验基于 core 或用户文档真实语料，无模拟数据。

