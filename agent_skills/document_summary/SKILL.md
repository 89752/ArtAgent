---
name: document_summary
description: 总结用户上传的文档（PDF/表格）：定位 → 读取 → 结构化摘要（JSON）
when_to_use: 用户要求总结/概括自己上传的文档或画册
version: 2
tools: [semantic_search, read_page_image]
max_steps: 8
steps_json: ["用 semantic_search 定位用户文档片段（标题形如《文档名》第N页）",
             "对 source=user_pdf_image 的整页图用 read_page_image 读取内容",
             "归纳文档主题与关键信息（分点）",
             "输出完整 JSON（含页码引用）"]
output_schema_json: {"topic": "文档主题", "key_points": "关键信息（分点列表，用\n分隔）", "page_refs": "页码引用（如 《画册》第3页）"}
---
# 执行纪律
- 只总结检索到的内容，不补充外部知识。
- 每条要点必须标注来源页码。

# 领域框架
- 先判断文档类型：画册（按页逐幅）／文章／表格；
- 画册类按页归纳作品与说明；文章类按段落归纳论点；
- 页码引用统一格式：《文档名》第N页。
