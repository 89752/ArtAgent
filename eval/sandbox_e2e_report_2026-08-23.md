# ArtAgent 沙盒真实用户验收报告

- 日期：2026-08-23
- 环境：Docker 沙盒 `http://localhost:7861`，已登录的本地测试账户
- 范围：不含 MCP / Multi-Agent；覆盖单会话聊天、收藏与长期记忆、文档/表格/图像上传、图像分析、资料库、运行中心、移动端导航、任务恢复和限定证据问答。
- 测试资料：`eval/sandbox_fixtures/` 中的无个人信息 PDF、CSV，以及项目核心库中的公开艺术图像。

## 结论

核心单用户服务链路在沙盒中可用，发现的阻断问题均已修复并在真实界面回归。容器当前状态为 **healthy**。其中，PDF 入库、CSV schema 确认、公开图像上传与一次三层分析、以及“仅根据上传 PDF”的受限问答均已通过。

偏好推荐的卡片/来源一致性已完成真实复验：旧链路一次请求耗时约 87.5 秒、显示 17 个步骤，且曾出现正文与卡片/来源不一致；优化后的候选检索与展示契约能使三幅正文作品、三张卡片和三条来源逐项一致。曾测得的 34.64 秒“直达推荐工作流”已在后续架构复审中撤回，原因是它会截断复合需求；当前推荐重新由统一 ReAct 编排，需以多意图真实任务另行记录新基线。

## 真实用户流程

| 场景 | 结果 | 关键观察 |
| --- | --- | --- |
| 登录后会话恢复、历史与运行中心 | 通过 | 已登录会话能恢复；运行中心可看到运行记录与耗时。 |
| 收藏清单创建、列举、改名 | 通过 | 创建“验收-印象派”、加入两件作品、列举并改名均成功。 |
| 记住偏好并据此推荐 | 通过 | 以“宁静、明亮、水景”为偏好，推荐河道、海景、池畔水景三幅作品；正文、卡片、来源一一对应。当前普通推荐已回归 ReAct 编排。 |
| “遗忘关于巴洛克的一切” | 通过 | 最终审计显示实体级删除成功，相关活动记忆为 0。 |
| 窄屏导航 | 通过 | 768px 左右断点下可通过“打开导航菜单”访问新建对话、历史、运行中心和设置。 |
| PDF 上传和解析 | 通过 | 初次因本地 BGE-M3 缺失失败；修复后同一 PDF 显示“已就绪”，1 个文字 chunk 和 1 个页面图索引完成。 |
| CSV 上传、推断、确认和启用 | 通过 | 推断 `artist` 为实体列；确认后文档状态为 `active`，用户数据源已注册。 |
| 公开艺术图像上传 | 通过 | 修复容器写路径后，图片显示“分析画作”。 |
| 一次真实三层图像分析 | 通过 | 约 50 秒完成；客观技法、风格/情绪、专业建议三层均在界面呈现。 |
| “仅根据上传 PDF”问答 | 通过 | 最终引用只保留《artagent_sample.pdf》第 1 页，答案正确指出 Claude Monet 与 Water Lilies。 |

## 本轮修复

1. 长期记忆删除可靠性
   - 将“遗忘”识别为明确删除授权，避免治理层拒绝操作而模型错误宣称已完成。
   - 对“关于 X 的一切/相关信息”强制执行实体级删除，连同派生用户画像清理。
   - 删除跨会话历史中的 `forget` 工具消息对新请求的错误抑制；只在当前 ReAct 轮次防重复。

2. 文档与表格可用性
   - 新鲜 Docker 环境没有预下载 BGE-M3 时，用户 PDF 文字改入独立的轻量本地 n-gram 向量库；核心数据仍保留 BGE 语义空间。安装 BGE-M3 后仍使用原有高质量路径。
   - 修复表格入库、schema 确认和重启恢复时丢失真实 `user_id` 的问题。
   - 服务重启时把未重新调度的 `pending` 导入任务标为可重试，避免永久卡在排队状态。
   - “仅依据上传 PDF”自动为 `semantic_search` 加上 `user_pdf_text` 证据边界，防止核心库来源混入。

3. 图像与容器路径
   - 用户图像默认跟随 `UPLOADS_DIR`；沙盒显式写入 `/sandbox/uploads/user_images`，解决非 root 容器下 `/app/data/uploads` 无权限的问题。
   - 沙盒取消继承宿主机不可达的回环代理，Jina 精排可按实际网络配置使用或正常降级。

4. 前端可访问性和小屏体验
   - 小屏增加侧边栏打开按钮。
   - 上传入口改为原生关联的 `label` + 文件输入，浏览器文件选择器可正常触发，键盘也能聚焦操作。
   - 表格的“待确认 schema”从不可访问的可点击文本改为语义按钮。

5. 偏好推荐的性能与证据一致性
   - 经架构复核，`recommendation_search` 与 `art_recommendation` Skill 均已移除：它们没有独占能力，只是以较窄的本地词法捷径重复了通用增强检索。
   - 偏好与“推荐”不再触发专用路由、专属澄清或表格能力标记。记忆在澄清前加载，随后统一由 ReAct 结合 `semantic_search` / `agentic_retrieve`、文档、对比和联网工具处理。
   - UI 对所有回答统一执行“正文采用的作品才展示卡片和来源”的证据约束；不再仅把这项规则绑定到推荐请求。
   - 先前基于专用工具的网页性能与候选结果仅为历史诊断，不能作为当前统一检索路径的性能基线；本次离线回归覆盖其新的工具、澄清与卡片契约。

## 自动化验证

以下均在 `artagent` Conda 环境执行并通过：

| 命令范围 | 结果 |
| --- | --- |
| `tests/test_agent.py` | 76 passed |
| `tests/test_memory.py` 与 `tests/test_agent.py` | 161 passed（当时的联合回归） |
| `tests/test_retrieval.py` 与 `tests/test_ingestion.py` | 207 passed |
| `tests/test_multi_user.py`、`tests/test_api.py`、`tests/test_ingestion.py` | 155 passed，1 个第三方弃用警告 |
| `tests/test_analysis.py` 与 `tests/test_tools_unit.py` | 46 passed，1 个第三方弃用警告 |
| 统一检索/服务/技能/卡片联合回归：`tests/test_api.py`、`tests/test_skills.py`、`tests/test_agent.py`、`tests/test_retrieval.py` | 240 passed，1 个第三方弃用警告 |
| 评测集与方案对齐：`tests/test_eval_dataset.py`、`tests/test_upgrade_alignment.py` | 10 passed |
| 其余离线套件：分析、配置、接入、记忆、多用户、工具 | 267 passed、4 deselected，1 个第三方弃用警告 |
| 前端 `typecheck`、`build` | 通过；Vite 保留两条现有静态 SVG 运行时解析提示 |

本次移除后的分组回归合计为 **517 passed、4 deselected**；Docker 沙盒已重建，`/ready` 返回 `{"ok":true,"status":"ready"}`。

## 2026-08-25：BGE-M3 与核心向量索引复验

- 宿主机 Hugging Face 缓存中已存在完整 `BAAI/bge-m3` 快照；沙盒以只读方式挂载到容器用户的缓存目录，并设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。
- Chroma 即便仅读取集合也会尝试更新 SQLite 元数据，不能直接只读挂载生产索引。为避免污染 `data/index/chroma`，已复制 **约 436 MB** 的核心索引到 `.sandbox/core-index/chroma`，由沙盒独立使用。
- 代码新增 `CORE_INDEX_DIR`：核心作品集合从沙盒核心索引副本读取；用户文档、记忆等仍写入原有 `.sandbox/index`，两者不混用。
- 容器内实测：`SentenceTransformer` 成功离线加载 BGE-M3（`max_seq_length=1024`）；核心向量集合记录数 **39,314**；英文通用语义检索日志确认 `core hits=3`。验收时关闭外部精排，且最终查询未触发查询翻译或对话模型。
- 网页 `http://localhost:7861/` 与 `/ready` 均正常加载；未在网页发送会消耗模型额度的聊天请求。

## 仍需关注

- **偏好增强检索的生产语义质量（中优先级）**：应在肖像、静物、宗教画、强烈明暗和“偏好 + 其他任务”等场景各抽样复测，记录完整链路耗时与工具选择；避免为单一指标重新引入按意图分叉。
- **生产语义质量**：轻量本地向量兜底保证离线可用，但不应替代 BGE-M3。生产镜像/部署应预热 BGE-M3 缓存或明确提供远程嵌入服务。
- **未在本轮做真实写操作的范围**：密码修改/登出、批量任务取消、反馈提交、外部 Web/MCP、Multi-Agent。这些不影响上述核心单用户链路结论，但应在对应权限和额度充足时单独验收。
- 沙盒里保留了本轮失败后重传的合成附件，便于复现与审计；它们不含个人数据。

## 相关改动位置

- `src/agent/nodes/general.py`
- `src/agent/nodes/common.py`
- `src/agent/context.py`
- `src/agent/prompts.py`
- `src/tools/retrieval.py`
- `src/utils/governance.py`
- `src/memory/memory_items.py`
- `src/retrieval/userdoc_text_retriever.py`
- `src/ingestion/pipeline.py`
- `src/ingestion/table_pipeline.py`
- `src/tasks/store.py`
- `src/analysis/engine.py`
- `frontend/src/components/Composer.tsx`
- `frontend/src/components/LibraryDrawer.tsx`
- `frontend/src/api/types.ts`
- `frontend/src/components/Topbar.tsx`
- `frontend/src/styles/app.css`
- `docker-compose.sandbox.yml`
