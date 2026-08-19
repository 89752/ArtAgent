"""
ArtAgent 服务层 —— 把 LangGraph 推理与渲染逻辑从任何 UI 框架中解耦。

职责：
  · stream_answer(message, sid)：生成器，逐节点产出「助手气泡 HTML」，
    收尾持久化会话并给出最终 payload。API 层只需把每次产出转成 SSE。
  · 会话/偏好读写：直接透传 src.memory，供 REST 端点调用。

设计要点：
  · 与 app.py 的渲染完全一致（思考链折叠 + 内联配图）。
  · 助手气泡以 HTML 字符串产出：前端 innerHTML 直接挂载（内容由本服务生成，可信）。
"""

from __future__ import annotations

import json
import html
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import get_graph
from src.memory.memory_items import (
    clear_user_memories,
    delete_memory,
    list_memories,
)
from src.observability import runs as runs_store
from src.tasks import store as tasks_store
from src.memory.conversations import (
    save_conversation,
    list_conversations,
    load_conversation,
    delete_conversation,
    remove_attachment_from_all,
    rename_conversation as rename_conversation_db,
    relative_time,
)
from src.utils.logging_config import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_USER_ID = "web_user"  # 稳定用户标识：长期偏好跨会话累积
# 记忆身份对齐：图节点 load_memory / 工具层 remember 统一走 MEMORY_USER_ID，
# 避免 web 端 state.user_id 与工具写入身份不一致（2026-08-04）
os.environ.setdefault("MEMORY_USER_ID", WEB_USER_ID)

logger = get_logger("web.service")
graph = get_graph()

# ── 场景卡：文案 + 代表画作缩略图（点击直接发问）──
SCENE_CARDS = [
    {
        "query": "对比莫奈和梵高在色彩运用上的差异",
        "text": "对比莫奈和梵高在色彩运用上的差异？",
        "image": "28496-early05.jpg",
    },
    {
        "query": "梳理透纳的风格演变",
        "text": "透纳的绘画风格经历了怎样的演变？",
        "image": "40307-110turne.jpg",
    },
    {
        "query": "我喜欢维米尔的室内光线，还有哪些作品也采用了类似的光影处理？",
        "text": "喜欢维米尔的室内光线，还有哪些作品也采用了类似的光影处理？",
        "image": "42649-351seat.jpg",
    },
    {
        "query": "卡拉瓦乔的明暗对照法有什么特点",
        "text": "卡拉瓦乔的明暗对照法有哪些特点？",
        "image": "07480-13fligh.jpg",
    },
]

_NODE_LABELS = {
    "load_memory": "读取长期记忆",
    "ask_user": "澄清信息不足",
    "general_agent": "ReAct 推理",
    "general_tools": "执行工具",
    "reflection": "反思答案质量",
    "save_memory": "写入偏好记忆",
}


# ═══════════════════════════════════════════════════════════════════
# 渲染工具（与 app.py 一致）
# ═══════════════════════════════════════════════════════════════════
def _thumb_url(image_file: str) -> str:
    """本地图片转可缓存 URL（由 /api/images 从 core 或 SemArt 读取，不再内联 base64）。"""
    if not image_file:
        return ""
    # M3：核心库图片是 http(s) URL，直接透传给前端
    if image_file.startswith(("http://", "https://")):
        return image_file
    name = Path(image_file).name
    if not name:
        return ""
    return f"/api/images/{quote(name)}"


def _chain_detail(node: str, out: dict) -> str:
    """把某节点输出压成一句人类可读的进度说明。"""
    if not isinstance(out, dict):
        return ""
    if node == "load_memory":
        prefs = out.get("user_preferences") or {}
        arts = prefs.get("preferences") or prefs.get("artists") or []
        return (
            f"已知偏好 <span class='hl'>{len(arts)}</span> 条"
            if arts
            else "暂无历史偏好"
        )
    if node == "ask_user":
        q = (out.get("pending_clarification") or "").strip()
        return (
            f"追问：<span class='hl'>{html.escape(q[:48])}{'…' if len(q) > 48 else ''}</span>"
            if q
            else "信息充足，继续"
        )
    if node == "general_tools":
        shown = out.get("shown_artworks") or []
        rec = out.get("recommended_artists") or []
        parts = []
        if shown:
            parts.append(f"已展示 <span class='hl'>{len(shown)}</span> 幅")
        if rec:
            parts.append(f"已推荐 <span class='hl'>{len(rec)}</span> 位")
        return "；".join(parts) if parts else ""
    if node == "general_agent":
        msgs = out.get("messages") or []
        if msgs and getattr(msgs[-1], "tool_calls", None):
            return "调用工具：" + "、".join(
                f"<span class='hl'>{t.get('name')}</span>" for t in msgs[-1].tool_calls
            )
        return "直接作答"
    if node == "reflection":
        return (
            "结论：<span class='hl'>通过</span>"
            if out.get("reflection_notes") == "PASS"
            else "结论：<span class='hl'>信息不足，触发兜底</span>"
        )
    if node == "save_memory":
        extract_result = out.get("memory_extract_result") or {}
        profile_result = out.get("memory_profile_result") or {}
        if extract_result.get("scheduled"):
            return "已安排后台记忆抽取"
        if profile_result.get("error"):
            return "画像刷新失败（已降级）"
        if profile_result.get("action"):
            return "用户画像已更新"
        return "偏好已持久化"
    return ""


def _chain_html(steps: list[dict], done: bool) -> str:
    html = ""
    for i, s in enumerate(steps):
        pending = (i == len(steps) - 1) and not done
        dot = "chain-dot pending" if pending else "chain-dot"
        name = _NODE_LABELS.get(s["node"], s["node"])
        detail = (
            f'<div class="chain-detail">{s["detail"]}</div>' if s.get("detail") else ""
        )
        html += (
            f'<div class="chain-step"><span class="{dot}"></span>'
            f'<div class="chain-name">{name}</div>{detail}</div>'
        )
    return html


def _think_box(steps: list[dict], done: bool) -> str:
    if not steps:
        return ""
    open_attr = "" if done else " open"
    label = "思考过程" if done else "正在思考…"
    return (
        f'<details class="think-box"{open_attr}><summary>{label}'
        f"（{len(steps)} 步）</summary>"
        f'<div class="think-body">{_chain_html(steps, done)}</div></details>'
    )


def _artwork_grid(artworks: list[dict], with_thumbs: bool) -> str:
    if not artworks or not with_thumbs:
        return ""
    cells = ""
    for aw in artworks[:4]:
        uri = _thumb_url(aw.get("image_file", ""))
        if not uri:
            continue
        title = html.escape((aw.get("title") or "")[:24])
        author = html.escape((aw.get("author") or "")[:22])
        uri_esc = html.escape(uri, quote=True)
        cells += (
            f'<figure class="aw-card"><img src="{uri_esc}" alt="{title}" loading="lazy"/>'
            f'<figcaption class="aw-cap"><b>{title}</b>{author}</figcaption></figure>'
        )
    return f'<div class="aw-grid">{cells}</div>' if cells else ""


def _answer_block(answer: str) -> str:
    """
    答案是 LLM 的 Markdown 原文：转义后放进 .md-answer，
    前端用 marked.js 就地渲染（不引入 Python markdown 依赖）。
    存库的历史同样带此包裹，重载会话时前端一致解析。
    """
    if not answer:
        return ""
    return f'<div class="md-answer">{html.escape(answer)}</div>'


def _assistant_bubble(steps, answer, artworks, with_thumbs, done) -> str:
    return (
        _think_box(steps, done)
        + _answer_block(answer)
        + _artwork_grid(artworks, with_thumbs)
    )


def _parse_artworks_from_messages(messages: list) -> list[dict]:
    artworks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for item in (data if isinstance(data, list) else [data]):
                # 用户 PDF 片段带 source 键：不进配图卡片（无 SemArt 本地图）
                if isinstance(item, dict):
                    # 画家知识工具的代表作图片（query_painter_knowledge）
                    for aw in item.get("sample_work_images") or []:
                        if isinstance(aw, dict) and aw.get("title"):
                            artworks.append(
                                {
                                    "title": aw.get("title", ""),
                                    "author": item.get("matched_author", ""),
                                    "date": "",
                                    "image_file": aw.get("image_file", ""),
                                }
                            )
                    if item.get("title") and not item.get("source"):
                        artworks.append(
                            {
                                "title": item.get("title", ""),
                                "author": item.get("author", ""),
                                "date": item.get("date", ""),
                                "image_file": item.get("image_file", ""),
                            }
                        )
    return artworks


def _collect_sources(
    tool_messages: list,
    struct_artworks: list[dict],
    evidence: list[dict] | None = None,
) -> list[dict]:
    """从工具消息、结构化输出与检索证据收集参考来源（去重、限量）。"""
    sources: list[dict] = []
    seen: set = set()

    def add(kind: str, label: str) -> None:
        key = (kind, label)
        if key in seen:
            return
        seen.add(key)
        sources.append({"kind": kind, "label": label})

    for msg in tool_messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            src = item.get("source")
            if src in ("user_pdf_text", "user_pdf_image"):
                doc = item.get("doc_name") or "用户文档"
                page = item.get("page")
                label = f"《{doc}》第{page}页" if page else f"《{doc}》"
                add("document", label)
            elif src == "user_table":
                name = item.get("doc_name") or item.get("dataset_id") or "用户表格"
                add("table", f"表格《{name}》")
            elif item.get("title") and not src:
                author = item.get("author") or ""
                label = f"《{item['title']}》" + (f" · {author}" if author else "")
                add("artwork", label)

    for aw in struct_artworks or []:
        if isinstance(aw, dict) and aw.get("title") and not aw.get("source"):
            author = aw.get("author") or ""
            label = f"《{aw['title']}》" + (f" · {author}" if author else "")
            add("artwork", label)
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        src = item.get("source")
        if src in ("user_pdf_text", "user_pdf_image"):
            doc = item.get("doc_name") or "用户文档"
            page = item.get("page")
            label = f"《{doc}》第{page}页" if page else f"《{doc}》"
            add("document", label)
        elif src == "user_table":
            name = item.get("doc_name") or item.get("dataset_id") or "用户表格"
            add("table", f"表格《{name}》")
        elif item.get("title") and not src:
            author = item.get("author") or ""
            label = f"《{item['title']}》" + (f" · {author}" if author else "")
            add("artwork", label)
        elif item.get("url"):
            add("web", item.get("url"))
    return sources[:6]


def _clear_thread_checkpoint(thread_id: str) -> None:
    """尽力清理图检查点，避免中断运行残留导致下一轮续跑状态错乱。"""
    cp = getattr(graph, "checkpointer", None)
    if cp is None:
        return
    for meth in ("adelete_thread", "delete_thread", "adelete", "delete"):
        fn = getattr(cp, meth, None)
        if fn is None:
            continue
        try:
            if meth.startswith("a"):
                asyncio_ = __import__("asyncio")
                try:
                    asyncio_.get_running_loop()
                except RuntimeError:
                    asyncio_.run(fn(thread_id))
                else:
                    return
            else:
                fn(thread_id)
            logger.info("已清理会话 %s 的图检查点（%s）", thread_id, meth)
            return
        except Exception:  # noqa: BLE001
            continue


def memory_count(user_id: str = WEB_USER_ID) -> int:
    """已记住的记忆条目数（memory_items 全 kind）。"""
    return len(list_memories(user_id))


def memory_items_list(user_id: str = WEB_USER_ID) -> list[dict]:
    """记忆面板 v2：全部有效记忆条目（含来源/重要性/时间，供 UI 展示与删除）。"""
    return [
        {
            "id": i.get("id"),
            "kind": i.get("kind"),
            "content": i.get("content"),
            "entity": i.get("entity") or "",
            "source": i.get("source") or "user_explicit",
            "importance": float(i.get("importance") or 0.5),
            "scope": i.get("scope") or "user",
            "updated_at": i.get("updated_at") or "",
        }
        for i in list_memories(user_id)
    ]


def delete_memory_item(item_id: str, user_id: str = WEB_USER_ID) -> bool:
    """记忆面板：按条目 id 软删除（保留审计可追溯）。"""
    return delete_memory(user_id, item_id)


def clear_all_memories(user_id: str = WEB_USER_ID) -> int:
    """记忆面板：清空该用户全部记忆（memory_items + 会话滚动摘要）。"""
    from src.memory.summary import delete_user_summaries

    n = clear_user_memories(user_id)
    delete_user_summaries(user_id)
    return n


# ═══════════════════════════════════════════════════════════════════
# 流式 Agent 调用
# ═══════════════════════════════════════════════════════════════════
def stream_answer(
    message: str,
    sid: str,
    regenerate: bool = False,
    stop_event: threading.Event | None = None,
    request_id: str | None = None,
    user_id: str = WEB_USER_ID,
) -> Iterator[dict]:
    """
    生成器：逐节点产出事件字典，API 层转 SSE。
      · {"type": "delta", "html": <助手气泡 HTML>}           —— 流式刷新
      · {"type": "done",  "html": ..., "session_id": sid,
         "memory": <偏好数>, "sources": [...], "cancelled": bool,
         "request_id": ..., "error": ...} —— 收尾（含可观测轨迹与错误态）

    regenerate=True：丢弃最后一个用户消息及其后的回复，用本轮消息替代
    （编辑 / 重新生成共用）。stop_event：客户端断开/停止时置位，生成器在
    节点边界提前收尾并持久化部分内容。
    """
    request_id = request_id or uuid.uuid4().hex[:12]
    from src.memory.memory_items import set_active_user_id

    set_active_user_id(user_id)
    message = (message or "").strip()
    if not message:
        yield {
            "type": "done",
            "html": "",
            "session_id": sid,
            "memory": memory_count(user_id),
            "sources": [],
            "cancelled": False,
            "request_id": request_id,
            "error": "",
        }
        return

    start_ts = time.time()

    # 历史消息在库中（前端无状态）：读出→追加本轮→回写
    history = load_conversation(sid, user_id)
    if regenerate:
        last_user = -1
        for i, m in enumerate(history):
            if m.get("role") == "user":
                last_user = i
        if last_user >= 0:
            history = history[:last_user]
    history = history + [{"role": "user", "content": message}]
    # 在任何耗时推理开始前先落下用户消息。这样即使用户立即刷新、关闭页面，
    # 新会话也已经存在，不会因最终回答尚未来得及保存而被前端当作空会话丢弃。
    initial_title = next(
        (m.get("content", "") for m in history if m.get("role") == "user"),
        message,
    )
    save_conversation(sid, initial_title or "新对话", history, user_id)
    history = history + [
        {"role": "assistant", "content": _think_box([], done=False)},
    ]
    steps: list[dict] = []
    yield {"type": "delta", "html": history[-1]["content"]}

    # 当前生效数据源由服务端单例持有（前端已合并为统一知识库，不再切换；
    # 该值仅供 exact_lookup 等结构化工具默认使用 core）；每轮读进 state
    from src.platform import users as users_store

    active_dataset = users_store.get_user_dataset(user_id)
    from src.data import documents_store

    uploaded_docs = [
        {
            "doc_name": d.get("doc_name") or "",
            "pages": d.get("pages"),
            "kind": d.get("kind"),
            "text_chunks": d.get("text_chunks") or 0,
            "image_pages": d.get("image_pages") or 0,
        }
        for d in documents_store.list_documents(user_id)
    ]
    from src.analysis.store import (
        list_analysis_by_session,
        list_images_by_session,
    )

    uploaded_images = list_images_by_session(sid, user_id)
    analysis_reports = list_analysis_by_session(sid, user_id)

    intent, final_answer = "", ""
    tool_artworks: list[dict] = []
    tool_msgs: list = []
    evidence: list[dict] = []
    context_chars = 0
    tool_rounds = 0
    tool_names: list[str] = []
    reflection_triggered = False
    error_msg = ""
    cancelled = False

    try:
        for chunk in graph.stream(
            {
                # messages 由 checkpointer 跨轮累积；其余标量每轮重置，
                # 避免上一轮的 intent/retry_count 串味。
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "conversation_id": sid,
                "uploaded_docs": uploaded_docs,
                "uploaded_images": uploaded_images,
                "analysis_reports": analysis_reports,
                "intent": "",
                "tool_rounds": 0,
                "context_chars": 0,
                "executed_tool_signatures": [],
                "ask_user": "",
                "pending_clarification": "",
                "dataset_id": active_dataset,  # 每轮重置当前生效数据源
                "reflection_notes": "",
                "retry_count": 0,
                "final_answer": "",
            },
            config={"configurable": {"thread_id": sid}},
            stream_mode="updates",
        ):
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break
            for node, out in chunk.items():
                if node == "__interrupt__":
                    continue
                steps.append({"node": node, "detail": _chain_detail(node, out)})
                if isinstance(out, dict):
                    if out.get("context_chars"):
                        context_chars = out["context_chars"]
                    if out.get("tool_rounds"):
                        tool_rounds = out["tool_rounds"]
                    if node == "reflection" and out.get("reflection_notes") == "RETRY":
                        reflection_triggered = True
                    if out.get("intent"):
                        intent = out["intent"]
                    if out.get("final_answer"):
                        final_answer = out["final_answer"]
                    if out.get("messages"):
                        msgs = out["messages"]
                        tool_artworks.extend(_parse_artworks_from_messages(msgs))
                        tool_msgs.extend(m for m in msgs if isinstance(m, ToolMessage))
                        for m in msgs:
                            if getattr(m, "tool_calls", None):
                                tool_names.extend(
                                    str(tc.get("name"))
                                    for tc in m.tool_calls
                                    if tc.get("name")
                                )
                history[-1]["content"] = _assistant_bubble(
                    steps, "", [], False, done=False
                )
                yield {"type": "delta", "html": history[-1]["content"]}

        artworks = tool_artworks
        with_thumbs = intent in ("timeline", "recommendation", "general")
        reply = (
            (final_answer or "（未能生成回答，请重试）")
            if not cancelled
            else "（已停止生成，以上为已生成的部分内容）"
        )
        if cancelled:
            with_thumbs = False
        history[-1]["content"] = _assistant_bubble(
            steps, reply, artworks, with_thumbs, done=True
        )
        history[-1]["sources"] = _collect_sources(tool_msgs, tool_artworks, evidence)
        if cancelled:
            _clear_thread_checkpoint(sid)
    except Exception as e:  # noqa: BLE001 — 面向用户兜底，避免整页崩溃
        logger.exception("graph.stream failed: %s", e)
        error_msg = f"{type(e).__name__}: {e}"[:300]
        steps.append(
            {"node": "error", "detail": f"<span class='hl'>{type(e).__name__}</span>"}
        )
        history[-1]["content"] = _assistant_bubble(
            steps,
            "😔 抱歉，处理时出错了。可能是模型接口超时或未配置 API Key，请稍后重试。",
            [],
            False,
            done=True,
        )
        history[-1]["sources"] = []

    title = next((m["content"] for m in history if m["role"] == "user"), message)
    save_conversation(sid, title, history, user_id)
    runs_store.record_run(
        request_id=request_id,
        session_id=sid,
        intent=intent,
        steps=steps,
        tools=list(dict.fromkeys(tool_names)),
        context_chars=context_chars,
        tool_rounds=tool_rounds,
        latency_ms=(time.time() - start_ts) * 1000,
        final_answer_len=len(final_answer or ""),
        reflection_triggered=reflection_triggered,
        web_fallback=False,
        cancelled=cancelled,
        error=error_msg,
    )
    yield {
        "type": "done",
        "html": history[-1]["content"],
        "session_id": sid,
        "memory": memory_count(user_id),
        "sources": history[-1].get("sources", []),
        "cancelled": cancelled,
        "request_id": request_id,
        "error": error_msg,
    }


# ── 会话 / 偏好：透传给 REST 端点 ──
def sessions(
    offset: int = 0,
    limit: int = 50,
    user_id: str = WEB_USER_ID,
) -> tuple[list[dict], int]:
    """侧栏列表（分页）：附带相对时间，返回 (items, total)。"""
    convos, total = list_conversations(limit=limit, offset=offset, user_id=user_id)
    out = [{**c, "relative": relative_time(c["updated_at"])} for c in convos]
    return out, total


def rename_conversation(sid: str, title: str, user_id: str = WEB_USER_ID) -> bool:
    return rename_conversation_db(sid, title, user_id)


def conversation(sid: str, user_id: str = WEB_USER_ID) -> list[dict]:
    return load_conversation(sid, user_id)


def remove_conversation(sid: str, user_id: str = WEB_USER_ID) -> None:
    delete_conversation(sid, user_id)


def record_attachment(
    sid: str,
    doc_id: str,
    doc_name: str,
    kind: str,
    user_id: str = WEB_USER_ID,
) -> dict:
    """把「已上传文档」事件写进会话历史，切换会话/刷新后仍可见。"""
    if not sid or not doc_id:
        return {"ok": False, "error": "缺少会话或文档标识"}
    history = load_conversation(sid, user_id)
    if any(
        m.get("role") == "attachment" and m.get("doc_id") == doc_id for m in history
    ):
        return {"ok": True, "duplicated": True}
    history.append(
        {
            "role": "attachment",
            "content": doc_name or "文档",
            "doc_id": doc_id,
            "doc_name": doc_name or "文档",
            "kind": kind or "pdf",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    title = next((m["content"] for m in history if m["role"] == "user"), None)
    save_conversation(sid, title or "新对话", history, user_id)
    return {"ok": True, "duplicated": False}


def record_analysis_turn(
    sid: str,
    image_id: str,
    user_text: str = "",
    html: str = "",
    title: str = "",
    user_id: str = WEB_USER_ID,
) -> dict:
    """把分析（含拒绝）写入会话历史：用户回合 + assistant 回合，重载可还原。

    幂等：同一 image_id 已存在 assistant 分析回合时直接返回，不重复插入。
    """
    from src.analysis.store import get_analysis

    analysis = get_analysis(image_id)
    report: dict = {}
    if analysis and analysis.get("result_path"):
        path = Path(analysis["result_path"])
        if path.is_file():
            try:
                report = (json.loads(path.read_text(encoding="utf-8")) or {}).get(
                    "report"
                ) or {}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"分析结果读取失败：{e}"}
    history = load_conversation(sid, user_id)
    if any(
        m.get("role") == "assistant" and m.get("image_id") == image_id
        for m in history
    ):
        return {"ok": True, "duplicated": True, "image_id": image_id}
    user_text = (user_text or "").strip()
    if user_text and not any(
        m.get("role") == "user" and m.get("content") == user_text for m in history
    ):
        history.append(
            {
                "role": "user",
                "content": user_text,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    if not html:
        overall = str(report.get("overall_assessment") or "已生成三层分析报告")[:120]
        html = f'<div class="md-answer">分析完成：{html.escape(overall)}</div>'
    history.append(
        {
            "role": "assistant",
            "content": html,
            "report": report or None,
            "image_id": image_id,
            "title": (title or "").strip() or None,
            "analysis": True,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    title = next((m["content"] for m in history if m["role"] == "user"), None)
    save_conversation(sid, title or "新对话", history, user_id)
    return {"ok": True, "image_id": image_id}


# ── 文档上传与入库（PDF / 表格） ──

# 并发治理：解析任务信号量（env TASK_PARSE_CONCURRENCY，默认 2），
# 防止多文档同时跑 MinerU/视觉编码打爆单机资源。
_parse_semaphore = threading.Semaphore(
    max(1, int(os.getenv("TASK_PARSE_CONCURRENCY", "2")))
)


def save_upload(
    filename: str,
    data: bytes,
    kb_id: str = "default",
    user_id: str = WEB_USER_ID,
) -> dict:
    """把上传文件存到 uploads/{kb_id}/{doc_id}/；按类型路由存储名。

    PDF → document.pdf；表格 → table{原扩展名}。
    调用方须先用 classify_upload 判型，本函数不重复校验。
    """
    import uuid

    from src.data import documents_store
    from src.ingestion.pipeline import UPLOADS_DIR
    from src.ingestion.table_loader import classify_upload

    doc_id = uuid.uuid4().hex[:12]
    work_dir = UPLOADS_DIR / kb_id / doc_id
    work_dir.mkdir(parents=True, exist_ok=True)
    kind = classify_upload(filename)
    if kind == "table":
        file_path = work_dir / f"table{Path(filename).suffix.lower()}"
    else:
        file_path = work_dir / "document.pdf"
    file_path.write_bytes(data)

    # 一上传就落库基础记录，后续后台任务补充解析结果
    documents_store.add_document(
        doc_id=doc_id,
        kind=kind or "pdf",
        user_id=user_id,
        doc_name=filename,
        kb_id=kb_id,
        status="processing",
        file_path=str(file_path),
        file_size=len(data),
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    return {
        "doc_id": doc_id,
        "doc_name": filename,
        "kind": kind,
        "file_path": str(file_path),
        "kb_id": kb_id,
    }


def ingest_document(
    doc_id: str,
    doc_name: str,
    pdf_path: str,
    kb_id: str,
    task_id: str | None = None,
    force_pdfplumber: bool = False,
    user_id: str = WEB_USER_ID,
) -> None:
    """后台任务入口（BackgroundTasks）：跑入库流水线，异常已落 failed 状态。

    task_id 提供时同步维护任务表状态，解析并发受信号量约束。
    """
    from src.ingestion.pipeline import ingest_pdf

    with _parse_semaphore:
        if task_id:
            tasks_store.update_task(task_id, status="processing")
        try:
            ingest_pdf(
                pdf_path, doc_id, doc_name=doc_name, kb_id=kb_id,
                force_pdfplumber=force_pdfplumber,
                user_id=user_id,
            )
            if task_id:
                tasks_store.update_task(task_id, status="done", progress=100)
        except Exception as e:
            logger.exception("ingest_document failed: %s", doc_id)
            from src.data import documents_store

            documents_store.update_document(
                doc_id, status="failed", error=str(e)[:300]
            )
            if task_id:
                tasks_store.update_task(task_id, status="failed", error="文档解析失败")


def ingest_table_doc(
    doc_id: str,
    doc_name: str,
    table_path: str,
    kb_id: str,
    task_id: str | None = None,
    user_id: str = WEB_USER_ID,
) -> None:
    """表格后台任务入口：加载 + schema 推断 → 待确认状态。"""
    from src.ingestion.table_pipeline import ingest_table

    with _parse_semaphore:
        if task_id:
            tasks_store.update_task(task_id, status="processing")
        try:
            ingest_table(
                table_path, doc_id, doc_name=doc_name, kb_id=kb_id, user_id=user_id
            )
            if task_id:
                tasks_store.update_task(task_id, status="done", progress=100)
        except Exception:
            logger.exception("ingest_table_doc failed: %s", doc_id)
            if task_id:
                tasks_store.update_task(task_id, status="failed", error="表格解析失败")


def confirm_table(
    doc_id: str,
    roles: dict,
    user_id: str = WEB_USER_ID,
) -> dict:
    """确认/纠正表格 schema：注册生效。"""
    from src.ingestion.table_pipeline import confirm_table_schema

    return confirm_table_schema(doc_id, roles, user_id=user_id)


def restore_tables() -> int:
    """服务启动时恢复已确认的表格数据源。"""
    from src.ingestion.table_pipeline import restore_active_tables

    return restore_active_tables()


def documents(user_id: str = WEB_USER_ID) -> list[dict]:
    """文档库列表（新的在前）。"""
    from src.data import documents_store

    docs = documents_store.list_documents(user_id)
    return [_reconcile_pdf_task_status(doc, user_id) for doc in docs]


def document_status(doc_id: str, user_id: str = WEB_USER_ID) -> dict:
    from src.data import documents_store

    doc = documents_store.get_document(doc_id, user_id) or {}
    return _reconcile_pdf_task_status(doc, user_id) if doc else {}


def _reconcile_pdf_task_status(doc: dict, user_id: str) -> dict:
    """修复旧版本遗留的“任务已结束、文档仍解析中”状态。"""
    from src.data import documents_store

    if doc.get("kind") != "pdf" or doc.get("status") != "processing":
        return doc
    task = tasks_store.get_task(str(doc.get("doc_id") or ""))
    if not task:
        return doc
    payload = task.get("payload") or {}
    if payload.get("user_id") and payload.get("user_id") != user_id:
        return doc
    task_status = task.get("status")
    if task_status == "done":
        patch = {
            "status": "done",
            "finished_at": task.get("finished_at") or "",
            "error": "",
        }
    elif task_status in ("failed", "interrupted"):
        patch = {
            "status": "failed",
            "finished_at": task.get("finished_at") or "",
            "error": task.get("error") or "文档解析未完成，请重试",
        }
    else:
        return doc
    documents_store.update_document(doc["doc_id"], **patch)
    return {**doc, **patch}


def delete_document(doc_id: str, user_id: str = WEB_USER_ID) -> dict:
    """删除文档并级联清理：状态记录、上传文件、向量（PDF）、注册表（Table）。"""
    from src.data import documents_store
    from src.ingestion.pipeline import UPLOADS_DIR, delete_pdf_vectors
    from src.ingestion.table_pipeline import table_dataset_id, unregister_table
    from src.retrieval.hybrid import get_hybrid_retriever

    doc = documents_store.get_document(doc_id, user_id)
    if not doc:
        raise KeyError(f"文档不存在：{doc_id}")

    kind = doc.get("kind", "pdf")
    kb_id = doc.get("kb_id", "default")
    result: dict = {
        "doc_id": doc_id,
        "kind": kind,
        "vectors": {},
        "files_removed": False,
    }

    # 1. PDF：清理两路 Chroma 向量
    if kind == "pdf":
        result["vectors"] = delete_pdf_vectors(doc_id)

    # 2. Table：从内存注册表与 Hybrid 移除；若当前生效则复位
    else:
        dataset_id = doc.get("dataset_id") or table_dataset_id(doc_id)
        unregister_table(dataset_id)
        from src.platform import users as users_store

        from src.retrieval.hybrid import get_hybrid_retriever

        hybrid = get_hybrid_retriever()
        if hybrid.active_dataset == dataset_id:
            hybrid.active_dataset = "core"
            result["active_dataset_reset"] = "core"
        if users_store.get_user_dataset(user_id) == dataset_id:
            users_store.set_user_dataset(user_id, "core")
            result["user_dataset_reset"] = "core"

    # 3. 删除上传文件目录
    work_dir = UPLOADS_DIR / kb_id / doc_id
    if work_dir.exists():
        try:
            import shutil

            shutil.rmtree(work_dir)
            result["files_removed"] = True
        except Exception as e:  # noqa: BLE001
            logger.warning("delete_document 删除文件目录失败 %s: %s", work_dir, e)

    # 4. 删除 SQLite 记录
    deleted = documents_store.delete_document(doc_id, user_id)
    result["db_deleted"] = deleted

    # 5. 从所有会话历史中移除该文档的附件记录
    result["attachment_records_removed"] = remove_attachment_from_all(doc_id, user_id)
    return result
