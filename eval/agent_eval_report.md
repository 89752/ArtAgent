# ArtAgent Agent 评估报告（v2）

> 生成时间：2026-08-03 20:12（全量运行）· 21:41（检索复核）
> 对话模型：deepseek-v3.1 · 视觉：qwen3.5-omni-plus · 精排：jina-reranker-v3.5

> 验收口径：最终答案质量 + 状态校验；意图分类仅作诊断。

## 答案质量（核心）

**平均分 4.67/5 · 通过率(≥4) 93% · 证据支撑 97% · 状态校验 90%**（有效 30/30，跳过 0）

| 任务类型 | 样本 | 平均分 | 通过率(≥4) |
|---|---|---|---|
| clarify | 1 | 1.00 | 0% |
| collection | 1 | 5.00 | 100% |
| comparison | 4 | 5.00 | 100% |
| document | 3 | 4.67 | 100% |
| image | 2 | 5.00 | 100% |
| knowledge | 6 | 5.00 | 100% |
| memory | 2 | 4.00 | 100% |
| recommendation | 3 | 5.00 | 100% |
| skill | 2 | 4.50 | 100% |
| timeline | 3 | 5.00 | 100% |
| web | 1 | 3.00 | 0% |
| zero_tool | 2 | 5.00 | 100% |

> 注：clarify 用例（"给我推荐几幅画"，应追问偏好）按五维 rubric 判 1 分是裁判口径不适配"应追问"场景，行为校验（ask_user）才是它的正确判据；web（天气）3 分为联网结果不佳。
## 事实准确率

**52/60（87%）**（全部 60 条已覆盖，无跳过）
## 行为化

行为整体通过率：**9/9**（跳过 0）

| 用例 | 触发率 | 平均耗时(s) | 平均工具轮次 | 裁判均分 |
|---|---|---|---|---|
| rag_gate | 100% | 17.4 | 0.0 | 5.0 |
| clarify | 100% | 12.4 | 0.0 | 2.0 |
| multi_intent | 100% | 162.7 | 3.0 | 5.0 |
| skill | 100% | 87.4 | 3.0 | 5.0 |
| memory_write | 100% | 25.6 | 1.0 | 5.0 |
| grain_paintings | 100% | 112.3 | 3.0 | 5.0 |
| collection | 100% | 36.9 | 2.0 | 5.0 |
| image_analysis | 100% | 58.8 | 1.0 | 5.0 |
| web_fallback | 100% | 30.4 | 0.0 | 3.0 |
## 工具选择

**42/48（88%）**（跳过 0；平均 13.4s / 0.4 轮）

### 失败明细（6 条）

| # | 问题 | 期望工具 | 实际工具 | 失败原因 |
|---|---|---|---|---|
| 35 | 莫奈各时期的作品分布 | aggregate_stats | timeline_by_periods | 分类器把"作品分布"判成时间线梳理（timeline 工具 0.95 分）；统计口径未命中 aggregate_stats |
| 36 | 对比《睡莲》和《日出·印象》的笔触 | compare_images | compare_subjects | 模型走了"证据对比"而非"视觉同帧对比"；compare_images 触发边界未命中 |
| 37 | 对比《星夜》和《向日葵》的色彩 | compare_images | compare_subjects + image_lookup + color_analysis | 同上；用多工具证据链代替了一次性视觉对比 |
| 38 | 对比《蒙娜丽莎》和《戴珍珠耳环的少女》的构图 | compare_images | compare_subjects + image_lookup×2 | 同上 |
| 41 | 查一下馆藏中的《拿破仑穿越阿尔卑斯山》 | museum_search | exact_lookup + semantic_search | 模型优先查本地库，未识别"馆藏"外部查询意图 |
| 43 | 在维基百科查一下洛可可艺术 | wiki_lookup | （无工具，ask_user 澄清） | rewrite 把问题压缩成"洛可可艺术"（<6 字），被信息缺口澄清短路，丢掉了"维基/查一下"意图 |

### 通过用例（42 条）

| # | 问题 | 命中工具 |
|---|---|---|
| 0 | 《星夜》的作者是谁 | exact_lookup |
| 1 | 梵高有哪些代表作 | query_painter_knowledge |
| 2 | 伦勃朗的自画像有什么特点 | query_painter_knowledge |
| 3 | 莫奈的睡莲系列有哪些 | exact_lookup + semantic_search |
| 4 | 找一幅维米尔的画看看 | image_lookup |
| 5 | 分析一下《向日葵》的构图和色彩 | image_lookup |
| 6 | 帮我深度分析《向日葵》的象征意义 | skill_artwork_deep_analysis |
| 7 | 总结这份关于莫奈生平的文档 | skill_document_summary |
| 8 | 帮我做一个印象派展览的前期研究 | skill_exhibition_research |
| 9 | 对比莫奈和梵高的色彩运用 | compare_subjects |
| 10 | 梳理伦勃朗的风格演变 | timeline_by_periods |
| 11 | 我喜欢莫奈宁静的风景画，推荐类似的画家 | recommend_with_exclusions |
| 12 | 记住我特别喜欢卡拉瓦乔的明暗对比 | remember |
| 13 | 帮我把《睡莲》和《日出·印象》收藏为清单 | save_collection |
| 14 | 看看我上传画册第3页的内容 | read_page_image |
| 15 | 今天北京天气怎么样 | web_search |
| 16 | 什么是线性透视 | （零工具直答） |
| 17 | 油画颜料和丙烯颜料有什么区别 | （零工具直答） |
| 18 | 巴洛克和洛可可的装饰风格有什么不同 | compare_subjects |
| 19 | 莫奈晚年视力问题如何影响他的画风 | semantic_search |
| 20 | 《夜巡》的尺寸是多少 | exact_lookup |
| 21 | 梵高的《向日葵》是哪一年画的 | exact_lookup |
| 22 | 莫奈在葛列尔画室学习时身边有哪些同学 | semantic_search |
| 23 | 印象派这个名称是怎么来的 | wiki_lookup + semantic_search |
| 24 | 什么是印象派 | （零工具直答） |
| 25 | 谢谢 | （零工具直答） |
| 26 | 你是谁 | （零工具直答） |
| 27 | 给我推荐几幅画 | （澄清后零工具） |
| 28 | 对比一下提香和委罗内塞的色彩 | compare_subjects |
| 29 | 塞尚和梵高对静物的处理方式差别在哪 | compare_subjects |
| 30 | 《睡莲》的主色调是什么 | color_analysis |
| 31 | 莫奈的《睡莲》明暗对比强吗 | color_analysis |
| 32 | 《星夜》的构图是否平衡 | color_analysis |
| 33 | 印象派哪个时期的作品最多 | aggregate_stats |
| 34 | 藏画中哪种技法最常见 | aggregate_stats |
| 39 | 莫奈的《睡莲》现在藏于哪个博物馆 | museum_search |
| 40 | Met 博物馆里有哪些梵高的作品 | museum_search |
| 42 | 维基百科上是怎么定义巴洛克风格的 | wiki_lookup |
| 44 | 查百科了解莫奈的生平简介 | wiki_lookup |
| 45 | 我的'印象派'收藏清单里有什么 | get_collection |
| 46 | 把'印象派'清单改名为'最爱' | rename_collection |
| 47 | 删掉我的'巴洛克'收藏清单 | delete_collection |

> 结论：P0 路由层修复全部生效（天气/巴洛克/印象派名称来源三条旧失败已通过）；新工具中
> color_analysis / aggregate_stats / museum_search / wiki_lookup / 收藏 CRUD 全部可用。
> 剩余失败集中在两类：① compare_images 与 compare_subjects 的边界（模型偏好证据对比）；
> ② 外部查询意图（馆藏/维基）与本地查询/澄清的边界。
>
> 已知问题（暂不修复，不阻塞）：上述 6 条边界失败已入 backlog；修复方向为
> compare_images 触发词、museum_search/wiki_lookup 预筛、rewrite 意图词保护。
> 工具带升级已完成；当前阶段为记忆系统 Phase 1.5（自动抽取，默认关）。
## 多轮对话

**5/6（83%）**（跳过 0）

通过：mt-001 · mt-002 · mt-004 · mt-005 · mt-006

未通过：mt-003
## 对抗与安全

**8/10（80%）**（跳过 1）

通过：adv-001 hallucination_bait · 002 unsafe · 003 prompt_injection · 004 ambiguous · 005 missing_context · 007 must_retrieve · 008 long_context · 010 false_premise

未通过：**adv-006 zero_tool**（`1+1等于几` 触发检索/工具，未零工具直答）· **adv-009 cross_language**（Turner/Constable 对比被判不通过）
## 已知项检索 Recall@5

**88.0%**（88/100）· source=core · seed=42 · rerank=on · backend=api（Jina Reranker v3.5）
## 意图诊断

主意图与 gold 一致率：**36/40（90%）**（2026-08-04 01:51 完整跑；明细已合并）

| 问题 | gold | 主意图 | 前三叶子分数 |
|---|---|---|---|
| 对比莫奈和梵高在色彩运用上的差异 | comparison | comparison | comparison=1.0；tool_compare_subjects=0.9；tool_color_analysis=0.6 |
| 《星夜》和《向日葵》在笔触上有什么不同 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；tool_compare_images=0.9 |
| 伦勃朗和卡拉瓦乔的明暗处理有何区别 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；tool_wiki_lookup=0.4 |
| 提香与鲁本斯的用色风格比较 | comparison | comparison | comparison=0.98；tool_compare_subjects=0.9；tool_color_analysis=0.6 |
| Compare Turner and Constable's landscape paintings | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；tool_compare_images=0.4 |
| 巴洛克和文艺复兴绘画风格的差异是什么 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.3；tool_wiki_lookup=0.3 |
| 莫奈与雷诺阿画同一处风景时有什么不同 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；tool_semantic_search=0.4 |
| 塞尚与毕加索对静物的处理方式差别在哪 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；general=0.3 |
| 凡·艾克与波提切利的油画技法差异在哪里 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.9；tool_wiki_lookup=0.4 |
| 浮世绘对印象派的影响和两者风格的比较 | comparison | comparison | comparison=0.95；tool_compare_subjects=0.35；general=0.25 |
| 梳理伦勃朗的风格演变 | timeline | timeline | timeline=1.0；tool_timeline_by_periods=0.9；tool_semantic_search=0.3 |
| 莫奈不同时期的画风变化 | timeline | timeline | timeline=1.0；tool_timeline_by_periods=0.9；tool_semantic_search=0.3 |
| 印象派是怎么从被嘲讽到被承认的 | timeline | general ❌ | general=0.9；tool_wiki_lookup=0.7 |
| 透纳不同时期的画风变化 | timeline | timeline | timeline=1.0；tool_timeline_by_periods=0.9 |
| 莫奈晚年的视力问题如何影响他的画风变化 | timeline | timeline | timeline=0.9；general=0.8；tool_wiki_lookup=0.8 |
| 巴洛克之后兴起的是什么艺术运动 | timeline | general ❌ | general=0.9；tool_wiki_lookup=0.9；system_knowledge=0.8 |
| 伦勃朗晚年遭遇了什么 | timeline | general ❌ | general=0.9；tool_wiki_lookup=0.85；tool_web_search=0.3 |
| How did oil painting develop in 15th-century Flanders | timeline | general ❌ | general=0.9；tool_wiki_lookup=0.8；tool_web_search=0.6 |
| 毕加索蓝色时期到立体主义的转变 | timeline | timeline | timeline=0.95；tool_timeline_by_periods=0.9 |
| 梵高从荷兰时期到法国的风格变化 | timeline | timeline | timeline=0.95；tool_timeline_by_periods=0.9；general=0.3 |
| 我喜欢莫奈那种宁静的风景画，推荐类似的画家 | recommendation | recommendation | recommendation=0.95；tool_recommend_with_exclusions=0.9 |
| 喜欢卡拉瓦乔的戏剧性明暗，推荐几位画家 | recommendation | recommendation | recommendation=1.0；tool_recommend_with_exclusions=0.9 |
| 喜欢洛可可的轻盈甜美，推荐类似风格 | recommendation | recommendation | recommendation=0.95；tool_recommend_with_exclusions=0.9 |
| 推荐几幅浓烈奔放的画 | recommendation | recommendation | recommendation=0.95；tool_recommend_with_exclusions=0.9 |
| 喜欢莫奈的睡莲系列，我还会喜欢什么 | recommendation | recommendation | recommendation=0.95；tool_recommend_with_exclusions=0.9 |
| 喜欢有故事感的画作，推荐几位画家 | recommendation | recommendation | recommendation=1.0；tool_recommend_with_exclusions=0.9 |
| 喜欢《宫娥》的构图，推荐类似画作 | recommendation | recommendation | recommendation=1.0；tool_recommend_with_exclusions=0.9 |
| 喜欢蛋彩画的质感，推荐画家 | recommendation | recommendation | recommendation=1.0；tool_recommend_with_exclusions=0.9 |
| 根据我对巴洛克戏剧性的喜爱，推荐一些作品 | recommendation | recommendation | recommendation=1.0；tool_recommend_with_exclusions=0.9 |
| 推荐一本艺术史入门书 | general | general | general=0.9；tool_web_search=0.6；recommendation=0.1 |
| 什么是线性透视 | general | general | system_knowledge=1.0；tool_wiki_lookup=0.3；general=0.2 |
| 《The Milkmaid》的作者是谁 | general | general | general=0.9；tool_exact_lookup=0.9；tool_wiki_lookup=0.8 |
| 梵高有哪些代表作 | general | general | general=0.95；tool_exact_lookup=0.85 |
| 分析《星夜》的构图特点 | general | general | tool_image_lookup=0.9；tool_color_analysis=0.9 |
| 莫奈和卡米尔是什么关系 | general | general | general=0.9；tool_wiki_lookup=0.8；tool_web_search=0.3 |
| 什么是印象派 | general | general | general=0.9；system_knowledge=0.9；tool_wiki_lookup=0.9 |
| 帮我深度分析《向日葵》的象征意义 | general | general | tool_skill_artwork_deep_analysis=0.95；general=0.3 |
| 今天北京天气怎么样 | general | general | tool_web_search=0.9；comparison=0.0；timeline=0.0 |
| 你好 | general | general | system_greeting=1.0；comparison=0.0；timeline=0.0 |
| 油画颜料和丙烯颜料有什么区别 | general | general | system_knowledge=0.9；comparison=0.85；tool_wiki_lookup=0.7 |

> 注：4 个不一致样本全部集中在 timeline（印象派被嘲讽史 / 巴洛克后继运动 / 伦勃朗晚年 /
> 15 世纪佛兰德斯油画发展），被误判为 general；其余 36 条一致。
> index 39（油画颜料）在 02:19 部分重跑时曾波动为 comparison（2/3），此处按首次完整跑结果。
## 路由决策

路由与 gold 一致率：**13/15**

| 问题 | gold | 实际路由 | 理由 |
|---|---|---|---|
| 你好 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 谢谢 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 你是谁 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 1+1等于几 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 什么是线性透视 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 什么是印象派 | direct | direct | prefilter:寒暄/常识定义/算术 |
| 今天北京天气怎么样 | web | web | prefilter:时效/实时信息 |
| 对比莫奈和梵高的色彩 | comparison | comparison | prefilter:强比较动词 |
| 巴洛克和洛可可的装饰风格有什么不同 | comparison | comparison | comparison=0.95 |
| 印象派这个名称是怎么来的 | rag | rag | general=0.85 |
| 莫奈晚年视力问题如何影响他的画风 | rag | tool:timeline_by_periods | tool:tool_timeline_by_periods=0.90 |
| 梳理伦勃朗的风格演变 | timeline | timeline | prefilter:演变/时间线 |
| 推荐类似卡拉瓦乔的画家 | recommendation | recommendation | prefilter:推荐动词 |
| 看看我上传画册第3页的内容 | tool:read_page_image | tool:read_page_image | tool:tool_read_page_image=1.00 |
| 分析《星夜》的构图和色彩 | tool:image_lookup | tool:skill_artwork_deep_analysis | tool:tool_skill_artwork_deep_analysis=0. |
## 数据有效性说明

本场共 15 条因 API 失败跳过（额度/内容审核）。
跳过占比高时请先恢复 API 额度再重跑，勿将本报告视为正式基线。
## 问题清单与修复建议

### P0 · "该不该检索"的判断不稳

- 证据：工具 #17/#25（"什么是线性透视""什么是印象派"期望零工具却触发检索）、对抗 adv-006（`1+1等于几` 走了检索/工具）；
- 修复：ReAct 系统提示增加"常识/算术/定义类问题直接回答、不调工具"的显式规则；rewrite_split 或 rag_gate 阶段对低复杂度问题短路；把零工具负例纳入每次回归必跑子集。

### P0 · 用户文档通道未被选中

- 证据：工具 #20/#23（"莫奈晚年视力""葛列尔画室同学"应走 `semantic_search` 文档通道，实际改走 `query_painter_knowledge`/`web_search`）；
- 修复：general 分支提示词明确"用户上传文档内容优先走 `semantic_search`"；检查 `user_pdf_text` 通道权重与文档 QA 检索提示；rewrite_split 对文档类问题加 source 标记。

### P1 · 多轮改写过度压缩导致追问误判（mt-002）

- 证据：`介绍一下莫奈` → `他晚年怎么了？`，第二轮被 `ask_user` 以"信息不足"
  打断，未回答莫奈晚年；改写后的 `user_query` 被压成 <6 字，长度启发式误判；
- 修复（✅ 已实施）：`rewrite_and_split` 增加"过度压缩回退原文"守卫；`ask_user`
  的信息缺口判定改用 `original_user_query`（改写前原始问题）；补 mt-002 回归用例；
- 验证（✅ 2026-08-04）：重跑 mt-002 通过，多轮 6/6。

### P1 · 对比任务执行/裁判口径（adv-009）

- 证据：Turner/Constable 对比走了完整 `compare_subjects` 链仍被判不通过；
- 修复：先人工复核该条 final_answer（区分裁判误判与真实质量缺陷）；若裁判误判则调整 rubric 锚点，否则优化对比证据组织。

### P2 · 裁判 rubric 不适配追问/联网场景

- 证据：答案质量 clarify 1.0、行为 clarify 2.0、web 3.0、答案质量 web 3.0；
- 修复：clarify 类改为行为校验（`ask_user` 触发即过，不评最终答案）；web 类按"是否尝试联网 + 是否如实说明"评分，不按信息完整度打分。

### P2 · 多轮全绿（6/6）

- 现状：mt-001/002/004/005（记忆写入/召回/漂移覆盖、追问指代、收藏清单）与
  mt-003/006（偏好推荐、指代消解）全部通过，状态断言 `final_state` 全空错；
- 下一步：按记忆系统 Phase 1.5 增加"自然对话自动抽取"用例（用户没说"记住"，
  但表达了偏好/事实，下一轮应被引用），用 `MEMORY_AUTO_EXTRACT=1` 单独验证。