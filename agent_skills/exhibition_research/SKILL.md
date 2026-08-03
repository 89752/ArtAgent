---
name: exhibition_research
description: 调研一位画家或流派：本地库 + 联网多源交叉核对，输出带来源标注的研究笔记（JSON）
when_to_use: 用户要"调研/研究/全面了解"某画家或流派，或本地资料不足需要补充
version: 2
tools: [query_painter_knowledge, semantic_search, exact_lookup, web_search]
max_steps: 10
steps_json: ["本地统计与代表作品（query_painter_knowledge + exact_lookup）",
             "本地评论证据（semantic_search）",
             "本地不足或需实时信息时 web_search 补充",
             "交叉核对：本地与网页信息冲突时分别标注",
             "输出完整 JSON（含来源标注）"]
output_schema_json: {"overview": "生平与地位", "style": "风格特征", "works": "代表作品（分点）", "sources": "来源标注（[本地]或[web:域名]）"}
---
# 执行纪律
- 本地与网页信息冲突时分别标注，不强行调和。
- 网页信息注明来源域名。

# 领域框架
- 研究分面：生平/地位 → 风格特征 → 代表作品 → 馆藏/影响；
- 同名异人风险：核对画家全名与活跃时期（参考 timeline_by_periods 的 identity_note 做法）；
- 来源分级：本地库 = 结构化事实；维基/博物馆页 = 较高可信；论坛/博客 = 低可信，需标注。
