---
name: art_recommendation
description: 根据用户审美偏好推荐画家/作品，排除用户已喜欢或已提到的画家（触发词：推荐/类似/喜欢...还有什么/还有谁）
when_to_use: 用户表达偏好并要求推荐
version: 1
tools: [semantic_search, exact_lookup, query_painter_knowledge, web_search, wiki_lookup, list_preferences]
max_steps: 8
steps_json: ["先读取用户偏好与已收藏（list_preferences / 记忆块）",
             "提炼风格特征（颜色/题材/年代/情绪）",
             "用语义检索找候选（semantic_search）",
             "排除用户已喜欢的画家",
             "按匹配度排序，给出 5-8 个候选",
             "输出完整 JSON"]
output_schema_json: {"features": "提炼出的风格特征", "liked_artists": "已喜欢的画家", "candidates": "候选画家/作品列表", "by_artist": "按画家分组的作品", "conclusion": "推荐理由一句话"}
---
# 执行纪律
- 必须排除用户已喜欢/已提到的画家，不得重复推荐。
- 候选必须有检索证据；不硬凑数量。
