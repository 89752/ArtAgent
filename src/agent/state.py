"""ArtAgent 状态定义（温和版：纯 ReAct + 记忆 + 澄清 + 反思）。"""

from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """LangGraph 状态，贯穿整个 Agent 执行流程。"""

    # ── 对话与输入 ─────────────────────────────────────────────
    # 多轮对话消息历史（LangGraph 托管，自动 append）
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_query: str = ""
    # 本轮改写前的原始用户问题（供信息缺口判定使用，避免压缩后的短句误判）
    original_user_query: str = ""

    # ── 路由 ───────────────────────────────────────────────────
    # 意图类型：general / comparison / timeline / recommendation
    intent: str = ""
    # 会话滚动摘要（由增量摘要器写入，注入 context.summary 块）
    conversation_summary: str = ""
    # load_memory 检索注入的记忆块文本（token 预算内）
    memory_block: str = ""
    # 自动抽取已推进到第几轮（节流计数，跨轮持久）
    memory_extracted_turns: int = 0
    # 最近一次自动抽取结果（可观测/调试）
    memory_extract_result: dict = Field(default_factory=dict)
    # 最近一次用户画像聚合结果（可观测/调试）
    memory_profile_result: dict = Field(default_factory=dict)
    # 记忆检索原始条目（[{id, kind, content, entity, source, importance, ...}]）
    memory_items: list[dict] = Field(default_factory=list)
    # 当前会话已上传的文档清单（[{doc_name, pages, kind, text_chunks, image_pages}]）
    uploaded_docs: list[dict] = Field(default_factory=list)
    # 当前会话已上传的图片（[{image_id, original_name, width, height, session_id}]）
    uploaded_images: list[dict] = Field(default_factory=list)
    # 当前会话已产生的分析报告（[{image_id, framework, result_path, updated_at}]）
    analysis_reports: list[dict] = Field(default_factory=list)
    # 信息缺口澄清路由信号："ask"=追问用户并短路；"continue"=放行
    ask_user: str = "continue"
    # ReAct 工具轮次计数（防循环失控上限）
    tool_rounds: int = 0
    # 本轮已执行过的工具调用签名（防重复调用烧光预算）
    executed_tool_signatures: list[str] = Field(default_factory=list)
    # 本轮送入 LLM 的上下文体积（字符近似，成本观测用）
    context_chars: int = 0
    # 当前执行到的节点（便于 UI 展示 Agent 决策链）
    current_step: str = ""

    # ── 数据源 ─────────────────────────────────────────────────
    # 当前生效的结构化数据源（对应 StructuredTableRetriever 注册表 key）。
    # timeline / recommendation 据此访问数据，路由层据此做能力开关判断；
    # 用户上传表格接入后可切换，默认核心库。
    dataset_id: str = "core"

    # ── 反思 / 兜底 ────────────────────────────────────────────
    # 反思节点判定：PASS / RETRY
    reflection_notes: str = ""
    # 兜底重试次数（防止死循环）
    retry_count: int = 0

    # ── 长期记忆 ───────────────────────────────────────────────
    # 稳定用户标识，跨会话记忆的 key；空串时由 load_memory/save_memory
    # 回落 get_memory_user_id()（环境变量/ContextVar），服务端显式传入
    user_id: str = ""
    # 会话标识（Web 传 sid；用于滚动摘要按会话存取）
    conversation_id: str = "default"
    # 从持久化存储读出的用户偏好：{"artists": [...], "styles": [...]}
    user_preferences: dict[str, list[str]] = Field(default_factory=dict)
    # 会话台账（P1）：本轮已展示画作 / 已推荐画家 / 待澄清项
    shown_artworks: list[str] = Field(default_factory=list)
    recommended_artists: list[str] = Field(default_factory=list)
    pending_clarification: str = ""

    # ── 最终输出 ───────────────────────────────────────────────
    final_answer: str = ""
