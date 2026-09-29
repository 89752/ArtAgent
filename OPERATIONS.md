# ArtAgent 单机运维与验收

更新：2026-09-19。适用边界：同一台机器、共享本地 SQLite/文件系统的少量 worker。不要将本地文件锁与 SQLite 租约解释为多主机协调能力。

## 安装与执行模式

- 已验证环境：Windows、Python 3.10；精确的已安装依赖闭包见 `requirements-win-py310.lock`。该快照含 `torch==2.13.0+cu132`，重建时须准备对应的 CUDA wheel 来源，再将快照作为 pip constraints 使用。尚未完成全新环境重装；这不是跨平台、带 wheel 哈希的供应链锁，不能保证仅用默认 PyPI 即可安装。Linux 容器仍需单独验证。
- 前端使用 `cd frontend` 后 `npm ci`、`npm run build`、`npm run lint`。
- `ARTAGENT_JOB_EXECUTION=process`：研究任务和文档解析使用独立进程。
- `ARTAGENT_STREAM_EXECUTION=process`：普通对话与图像分析使用独立进程。停止/断开后监护进程终止子进程；强制终止可能不保留尚未落库的最后一段文字，已发生的远端副作用不自动撤销。
- inline 仅用于调试和测试替身。进程模式有每次启动 Python/加载模型的额外延迟，应通过真实负载测量容量。
- 不要在共享部署启用默认测试账号或开发身份头。反向代理应限制请求体大小，并为 SSE 关闭响应缓冲。

## 就绪与诊断

- `/health`：进程存活。
- `/ready`：基础持久层可打开。
- 登录后 `/api/capabilities`：模型配置、持久 checkpoint、两种进程隔离模式及本人资料状态；不返回密钥，不代表供应商网络可达。
- 运行中心查看终态、错误、步骤、验收问题和报告版本。模型连接失败应显示失败及重试入口，不能显示完成。

## 文档版本与维护

解析写入独立版本，验收后发布 `active_index_id`。文字向量、图像向量和词法检索均限制为本人资料的生效版本。PDF 脚本入口也使用相同的发布协议。

后台每 5 分钟检查一次非生效版本，每轮最多回收 20 个；默认保留 24 小时，可用 `INDEX_VERSION_GRACE_SECONDS` 调整，配置最低 1 小时。回收与解析/删除使用同一把锁。失败会保留记录供重试，当前生效版本不会回收。版本图片也随旧版本清理，长期保存的历史报告图片引用可能因此失效；有长期归档要求时需单独导出原始材料。

无持久任务记录且超过 1 小时的历史解析文档标为失败，原始文件保留。任务租约失联的已开始解析不自动重放，用户应显式重试。

表格注册由每个 worker 在取得共享检索器时从 SQLite 同步；重新确认 schema 后刷新本进程缓存，删除后移除旧注册。已开始的一次查询可能仍持有旧快照，不承诺跨存储串行化快照。

## 删除账号

管理员删除账号先禁用身份并撤销凭据，再请求停止任务。若还有有效执行租约、上传预留或图像分析，返回 409，账号保持禁用，稍后重试同一删除操作。

清理覆盖文档文件和版本向量、用户图片与分析结果、研究任务、产物、经验、工具回执、checkpoint、轨迹明细、会话、记忆、收藏、反馈和抽取指标。只有清理成功才删除身份记录。文件/向量后端失败不会被当成删除成功；确认问题后重试。磁盘介质的物理擦除、日志归档和已有备份的保留/删除属于部署运维策略。

## 停机备份与恢复

先停止所有 API worker、任务执行进程和写入脚本。不能用单个数据库的在线备份代替多存储一致性快照。

默认目录示例：

```powershell
python -m src.ops.backup create .sandbox/backup-YYYYMMDD --root data=data --offline
python -m src.ops.backup verify .sandbox/backup-YYYYMMDD
python -m src.ops.backup restore .sandbox/backup-YYYYMMDD --destination .sandbox/restored-YYYYMMDD --offline
```

若 `INDEX_DIR`、`UPLOADS_DIR`、`USER_IMAGES_DIR`、`ARTAGENT_MEMORY_DIR`、`ARTAGENT_PLATFORM_DIR`、`ARTAGENT_HARNESS_DB_PATH`、`ARTAGENT_CHECKPOINT_DB_PATH` 指向 data 之外，须使用额外的 `--root 名称=目录` 包含全部位置。不要把含密钥的整个项目目录当作备份源。密钥配置应由部署方单独安全管理。

备份使用 SQLite backup API 和 SHA-256 清单；恢复前核对全部文件，只写入不存在的新目录，拒绝路径越界及覆盖已有目录。切换到恢复数据之前，验证数据库 integrity_check、任务/文档数量、登录、资料检索、报告下载。确认恢复可用后再由运维切换配置。

实际隔离演练：`.sandbox/acceptance-snapshot-20260919` → `.sandbox/acceptance-restored-20260919`，9 个文件；未操作用户业务数据。

## 发布门禁

```powershell
python -m pytest -q
python eval/agent_eval_v2.py --pr --out .sandbox/pr-eval.md
```

`--pr` 强制关闭远程重排、查询翻译和模型下载，必须跑公开词法样例和规则意图诊断。本地完整检索数据评测另行显式启用。设置 `ARTAGENT_EVAL_HISTORY_PATH` 可将评测历史写入隔离目录。

经验晋升要求当前内容及当前评测集版本通过基线对比，不得出现基线已成功样例的退化，且总 token 增幅不超过 50%。这是工程保护；合成样例通过不能证明真实研究收益。真实供应商端到端、领域标注保留集、用户试用和容量压测是独立的上线验收项，见 `RELEASE_READINESS.md`。
