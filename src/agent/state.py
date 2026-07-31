"""
ArtAgent 状态定义（混合架构版）

一个 AgentState 贯穿所有分支：
  - general       → ReAct 工具循环
  - comparison    → 跨维度风格对比子管线
  - timeline      → 时间线梳理 + 配图子管线
  - recommendation→ 基于偏好的链式推荐子管线

所有分支最终汇聚到 reflection，必要时走 web_fallback。
"""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """LangGraph 状态，贯穿整个 Agent 执行流程。"""

    # ── 对话与输入 ─────────────────────────────────────────────
    # 多轮对话消息历史（LangGraph 托管，自动 append）
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_query: str = ""

    # ── 路由 ───────────────────────────────────────────────────
    # 意图类型：general / comparison / timeline / recommendation
    intent: str = ""
    # 当前执行到的节点（便于 UI 展示 Agent 决策链）
    current_step: str = ""

    # ── 数据源（Stage 2） ───────────────────────────────────────
    # 当前生效的结构化数据源（对应 StructuredTableRetriever 注册表 key）。
    # timeline / recommendation 据此访问数据，路由层据此做能力开关判断；
    # Stage 5 用户上传表格接入后可切换，Stage 2 恒为 "semart"。
    dataset_id: str = "semart"

    # ── 规划 / 拆解 ────────────────────────────────────────────
    # 对比/推荐场景抽取出的对象，如 ["Claude Monet", "Vincent van Gogh"]
    subjects: list[str] = Field(default_factory=list)
    # 拆解后的子查询（对比按对象、时间线按时期）
    sub_queries: list[str] = Field(default_factory=list)
    # 推荐场景：LLM 推理生成的结构化风格特征（本项目核心亮点）
    extracted_features: str = ""

    # ── 检索结果 ───────────────────────────────────────────────
    # 分组检索结果：key 为子查询/对象/时期，value 为画作 dict 列表
    retrieved_docs: dict[str, list[dict]] = Field(default_factory=dict)
    # 扁平化的候选画作（供 UI 卡片、图像佐证消费）
    artworks: list[dict] = Field(default_factory=list)
    # 时间线/推荐场景配图：[{title, author, image_file, image_path}]
    images: list[dict] = Field(default_factory=list)
    # 推荐场景过滤后的候选画家/作品
    candidates: list[dict] = Field(default_factory=list)

    # ── 反思 / 兜底 ────────────────────────────────────────────
    # 反思节点判定：PASS / RETRY
    reflection_notes: str = ""
    # web 兜底检索结果
    web_results: list[dict] = Field(default_factory=list)
    # 兜底重试次数（防止死循环）
    retry_count: int = 0

    # ── 长期记忆（S5） ─────────────────────────────────────────
    # 稳定用户标识，跨会话记忆的 key
    user_id: str = "default_user"
    # 从持久化存储读出的用户偏好：{"artists": [...], "styles": [...]}
    user_preferences: dict[str, list[str]] = Field(default_factory=dict)

    # ── 工具结果（ReAct 分支原始记录，兼容旧代码） ──────────────
    tool_results: list[Any] = Field(default_factory=list)

    # ── 最终输出 ───────────────────────────────────────────────
    final_answer: str = ""
