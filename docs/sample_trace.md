# 示例执行轨迹（可观测性）

每个节点都输出结构化日志(`[节点] key=value`)与耗时(ms)。下面是两条真实查询的完整决策链,展示"走了哪个分支 / 检索到几条 / 反思结论 / 哪个节点慢"一目了然。

配置(见 [`src/utils/logging_config.py`](../src/utils/logging_config.py)):

```bash
ARTAGENT_LOG_LEVEL=INFO   # 默认；DEBUG 更详细
ARTAGENT_LOG_FILE=run.log # 可选，同时写文件
```

---

## 场景①：对比"莫奈 vs 梵高的色彩"

```
[load_memory] user=demo artists=[] styles=[]
[load_memory] done in 2ms → load_memory
[classify] query=对比莫奈和梵高在色彩运用上的差异 intent=comparison
[classify] done in 1892ms → classify→comparison
[decompose] subjects=['Claude Monet', 'Vincent van Gogh'] dimensions=['color use', 'brushwork', 'emotional tone']
[decompose] done in 2145ms → comparison_decompose
[retrieve] hits_per_subject={'Claude Monet': 4, 'Vincent van Gogh': 4} artworks=[8]
[retrieve] done in 15687ms → comparison_retrieve
[comp_synthesize] done in 19560ms → comparison_synthesize
[reflection] verdict=PASS answer_len=1105
[reflection] done in 1245ms → reflection
[save_memory] done in 0ms → save_memory
```

## 场景③：推荐"喜欢梵高浓烈奔放的风格"（核心亮点：推理式检索）

```
[load_memory] user=log_test artists=[] styles=[]
[classify] query=我喜欢梵高浓烈奔放的风格，还会喜欢哪些画家 intent=recommendation
[extract_features] liked=['Vincent van Gogh'] features=Bold vivid color contrasts, thick expressive impasto brus...
[feature_search] raw_hits=12 after_exclude=12 excluded=['Vincent van Gogh']
[relevance_filter] candidates_in=12 recommended=['Paul Cézanne', 'Gian Lorenzo Bernini', 'Jean-Baptiste Greuze']
[reflection] verdict=PASS answer_len=437
[save_memory] user=log_test saved_artists=['Vincent van Gogh']
```

> `[extract_features]` 这行是项目最能讲的一处:用户说"浓烈奔放"(主观),Agent 把它**推理**成 `Bold vivid color contrasts, thick expressive impasto brushwork...`(结构化风格特征),再拿它去向量检索 —— 检索 query 是 Agent 生成的中间产物,而非用户原话。

---

## 从日志能读出什么

- **分支决策**:`classify → intent=comparison`,确认路由正确。
- **检索健康度**:`hits_per_subject`、`raw_hits/after_exclude`,一眼看出是否检索为空或过滤过度。
- **性能瓶颈**:`comp_retrieve done in 15687ms` —— embedding 模型 + 向量查询是最慢环节,是优化的首要目标。
- **反思结论**:`verdict=PASS/RETRY`,判断是否触发了 web 兜底。
- **记忆写入**:`save_memory saved_artists=[...]`,确认跨会话偏好确实落库。
