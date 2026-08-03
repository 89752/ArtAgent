---
name: artwork_deep_analysis
description: 对一幅画作做深度结构化分析：构图/色彩/笔触/主题四维 + 点评，输出结构化 JSON
when_to_use: 用户要求"深度分析/详细解读/评析"某幅画
version: 2
tools: [exact_lookup, query_painter_knowledge, image_lookup]
max_steps: 8
steps_json: ["定位画作（exact_lookup），记录 title/author/date/technique",
             "取画家背景（query_painter_knowledge）",
             "仅当用户明确要求看图时做视觉分析（image_lookup analyze=True）",
             "按构图/色彩/笔触/主题四个维度组织分析",
             "输出完整 JSON（含一句话点评）"]
output_schema_json: {"title": "画作标题", "author": "画家", "date": "年代", "composition": "构图分析", "color": "色彩分析", "brushwork": "笔触与技法分析", "subject": "主题与内容", "verdict": "一句话点评"}
---
# 执行纪律
- 第 1 步必须成功定位画作才继续；定位不到就如实说明，在 verdict 里注明"未定位到该画作"。
- 不得在证据之外编造尺寸、年代、馆藏信息。

# 领域框架：构图/色彩/笔触分析检查单
## 构图分析（composition）
- 三分法/黄金分割：主体是否落在兴趣点；
- 对称与平衡：左右/上下的重量分配；
- 透视与空间：线性透视、大气透视、景深层次；
- 视线引导：线条、光线如何引导视线；
- 留白与满构图：疏密关系。

## 色彩分析（color）
- 色相：主色调、冷暖关系；
- 明度与对比：明暗对比、chiaroscuro 的使用；
- 饱和度：浓郁 vs 灰调；
- 色彩功能：写实再现 vs 情感表达 vs 象征。

## 笔触与技法（brushwork）
- 笔触类型：厚涂/薄染/点彩/湿画法/晕染；
- 表面质感：平滑 vs 可见肌理；
- 技法归属：结合 technique 字段（蛋彩/油画/水彩等）。

## 主题与语境（subject）
- 题材类型：宗教/神话/肖像/风景/静物/风俗；
- 时代与流派语境（结合作者的活跃时期）。
