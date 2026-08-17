---
name: art_timeline
description: 梳理某位画家或某个流派随时间演变的风格与代表作品（触发词：时间线/演变/分期/发展脉络/早期中期晚期）
when_to_use: 用户要求梳理风格演变或时间线
version: 1
tools: [exact_lookup, semantic_search, query_painter_knowledge, image_lookup, web_search, wiki_lookup]
max_steps: 8
steps_json: ["定位研究对象（exact_lookup / query_painter_knowledge）",
             "按年代分组检索代表作品（semantic_search，按 timeframe 过滤）",
             "核对身份：同名异人（如 Turner）时必须排除不匹配作品",
             "本地不足时用 web_search / wiki_lookup 补充",
             "按时间顺序组织分期，每期给出代表作品",
             "输出完整 JSON"]
output_schema_json: {"subject": "研究对象", "identity_note": "同名异人提示或空", "periods": "按时间排列的分期列表", "conclusion": "演变脉络一句话"}
---
# 执行纪律
- 不得编造分期；每个分期的年代必须有作品证据支撑。
- 发现作品归属多个同名画家时，在 identity_note 说明并排除不匹配者。
