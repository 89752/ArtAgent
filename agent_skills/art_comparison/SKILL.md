---
name: art_comparison
description: 对比两位或多位画家/画作/风格的差异，逐对象收集证据后按维度组织（触发词：对比/区别/差异/vs/比较/有什么不同）
when_to_use: 用户要求对比画家、画作或风格
version: 1
tools: [exact_lookup, semantic_search, query_painter_knowledge, image_lookup, web_search, wiki_lookup]
max_steps: 8
steps_json: ["定位所有对比对象（exact_lookup / semantic_search），确保每个对象都有证据",
             "对每个对象分别检索（query_painter_knowledge + semantic_search），不要混用证据",
             "需要量化视觉特征时用 image_lookup / color_analysis",
             "本地证据不足时用 web_search / wiki_lookup 补充",
             "按维度组织对比（风格/色彩/构图/主题），逐对象说明",
             "输出完整 JSON"]
output_schema_json: {"subjects": "对比对象列表", "dimensions": "对比维度", "comparison": "逐维度对比内容", "conclusion": "一句话结论"}
---
# 执行纪律
- 每个对象必须单独检索，禁止用一个对象的证据回答另一个对象。
- 不得编造作品细节；证据不足的维度明确标注"证据不足"。
