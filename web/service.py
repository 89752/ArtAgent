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
from src.memory.store import (
    load_preferences,
    clear_preferences,
    list_preferences,
    delete_preference,
)
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

_INTENT_LABELS = {
    "comparison": "🆚 跨维度对比",
    "timeline": "📅 时间线梳理",
    "recommendation": "💡 偏好推荐",
    "general": "💬 综合问答",
}
_NODE_LABELS = {
    "load_memory": "读取长期记忆",
    "rewrite_split": "改写与拆分",
    "classify": "识别意图",
    "rag_gate": "判断是否需要检索",
    "direct_answer": "直接回答",
    "ask_user": "澄清信息不足",
    "multi_retrieve": "并行检索子任务",
    "comp_decompose": "拆解对比对象与维度",
    "comp_retrieve": "分组语义检索",
    "comp_synthesize": "逐维度综合对比",
    "tl_subject": "锁定梳理对象",
    "tl_periods": "按时期收集证据+配图",
    "tl_synthesize": "编织时间线叙事",
    "rec_extract": "推理风格特征",
    "rec_search": "特征向量检索",
    "rec_filter": "相关性筛选",
    "rec_synthesize": "组织推荐理由",
    "general_agent": "ReAct 推理",
    "general_tools": "执行工具",
    "reflection": "反思答案质量",
    "web_fallback": "联网兜底检索",
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
        arts = (out.get("user_preferences") or {}).get("artists") or []
        return (
            f"已知偏好画家 <span class='hl'>{len(arts)}</span> 位"
            if arts
            else "暂无历史偏好"
        )
    if node in ("contextualize", "rewrite_split"):
        q = (out.get("user_query") or "").strip()
        return (
            f"理解为：<span class='hl'>{html.escape(q[:40])}"
            f"{'…' if len(q) > 40 else ''}</span>"
            if q
            else ""
        )
    if node == "classify":
        it = out.get("intent", "")
        return (
            f"意图 = <span class='hl'>{html.escape(_INTENT_LABELS.get(it, it))}</span>"
        )
    if node == "rag_gate":
        return (
            "无需检索，直接回答"
            if out.get("rag_needed") is False
            else "需要检索，进入检索路径"
        )
    if node == "direct_answer":
        a = (out.get("final_answer") or "").strip()
        return (
            f"直接回答：<span class='hl'>{html.escape(a[:48])}"
            f"{'…' if len(a) > 48 else ''}</span>"
            if a
            else ""
        )
    if node == "ask_user":
        q = (out.get("pending_clarification") or "").strip()
        return (
            f"追问：<span class='hl'>{html.escape(q[:48])}{'…' if len(q) > 48 else ''}</span>"
            if q
            else "信息充足，继续"
        )
    if node == "multi_retrieve":
        grouped = out.get("multi_evidence") or {}
        if not grouped:
            return "单一子任务，直接进入 agent"
        return "并行检索 <span class='hl'>%d</span> 个子任务" % len(grouped)
    if node == "general_tools":
        shown = out.get("shown_artworks") or []
        rec = out.get("recommended_artists") or []
        parts = []
        if shown:
            parts.append(f"已展示 <span class='hl'>{len(shown)}</span> 幅")
        if rec:
            parts.append(f"已推荐 <span class='hl'>{len(rec)}</span> 位")
        return "；".join(parts) if parts else ""
    if node == "comp_decompose":
        subs = out.get("subjects") or []
        return "对象：" + "、".join(
            f"<span class='hl'>{html.escape(s)}</span>" for s in subs
        )
    if node == "comp_retrieve":
        docs = out.get("retrieved_docs") or {}
        return f"检索到 <span class='hl'>{sum(len(v) for v in docs.values())}</span> 条评论证据"
    if node == "rec_extract":
        feat = (out.get("extracted_features") or "").strip()
        return (
            f"推理特征：<span class='hl'>{html.escape(feat[:48])}"
            f"{'…' if len(feat) > 48 else ''}</span>"
            if feat
            else ""
        )
    if node == "rec_search":
        return f"匹配候选 <span class='hl'>{len(out.get('artworks') or [])}</span> 幅"
    if node == "rec_filter":
        cands = out.get("candidates") or []
        names = "、".join(html.escape(c.get("author", "")) for c in cands[:4])
        return (
            f"筛出 <span class='hl'>{len(cands)}</span> 位：{names}"
            if cands
            else "未筛出匹配画家"
        )
    if node == "tl_subject":
        subs = out.get("subjects") or []
        return f"对象：<span class='hl'>{html.escape(subs[0])}</span>" if subs else ""
    if node == "tl_periods":
        return f"覆盖 <span class='hl'>{len(out.get('retrieved_docs') or {})}</span> 个时期"
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
    if node == "web_fallback":
        return (
            f"联网补充 <span class='hl'>{len(out.get('web_results') or [])}</span> 条"
        )
    if node == "save_memory":
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


def memory_count() -> int:
    """已记住的记忆条目数（memory_items 全 kind；空则回退旧偏好表）。"""
    items = list_memories(WEB_USER_ID)
    if items:
        return len(items)
    prefs = load_preferences(WEB_USER_ID)
    return len(prefs.get("artists") or []) + len(prefs.get("styles") or [])


def memory_items_list() -> list[dict]:
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
        for i in list_memories(WEB_USER_ID)
    ]


def delete_memory_item(item_id: str) -> bool:
    """记忆面板：按条目 id 软删除（保留审计可追溯）。"""
    return delete_memory(WEB_USER_ID, item_id)


def clear_all_memories() -> int:
    """记忆面板：清空该用户全部记忆（memory_items + 会话滚动摘要）。"""
    from src.memory.summary import delete_user_summaries

    n = clear_user_memories(WEB_USER_ID)
    delete_user_summaries(WEB_USER_ID)
    clear_preferences(WEB_USER_ID)
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
    message = (message or "").strip()
    if not message:
        yield {
            "type": "done",
            "html": "",
            "session_id": sid,
            "memory": memory_count(),
            "sources": [],
            "cancelled": False,
            "request_id": request_id,
            "error": "",
        }
        return

    start_ts = time.time()

    # 历史消息在库中（前端无状态）：读出→追加本轮→回写
    history = load_conversation(sid)
    if regenerate:
        last_user = -1
        for i, m in enumerate(history):
            if m.get("role") == "user":
                last_user = i
        if last_user >= 0:
            history = history[:last_user]
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": _think_box([], done=False)},
    ]
    steps: list[dict] = []
    yield {"type": "delta", "html": history[-1]["content"]}

    # 当前生效数据源由服务端单例持有（前端已合并为统一知识库，不再切换；
    # 该值仅供 exact_lookup 等结构化工具默认使用 core）；每轮读进 state
    from src.retrieval.hybrid import get_hybrid_retriever

    active_dataset = get_hybrid_retriever().active_dataset
    from src.data import documents_store

    uploaded_docs = [
        {
            "doc_name": d.get("doc_name") or "",
            "pages": d.get("pages"),
            "kind": d.get("kind"),
            "text_chunks": d.get("text_chunks") or 0,
            "image_pages": d.get("image_pages") or 0,
        }
        for d in documents_store.list_documents()
    ]

    intent, final_answer = "", ""
    struct_artworks: list[dict] = []
    tool_artworks: list[dict] = []
    tool_msgs: list = []
    evidence: list[dict] = []
    context_chars = 0
    tool_rounds = 0
    tool_names: list[str] = []
    reflection_triggered = False
    web_fallback_used = False
    error_msg = ""
    cancelled = False

    try:
        for chunk in graph.stream(
            {
                # messages 由 checkpointer 跨轮累积；其余标量每轮重置，
                # 避免上一轮的 intent/subjects/检索结果/retry_count 串味。
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": WEB_USER_ID,
                "conversation_id": sid,
                "uploaded_docs": uploaded_docs,
                "intent": "",
                "rag_needed": True,
                "tool_rounds": 0,
                "context_chars": 0,
                "executed_tool_signatures": [],
                "ask_user": "",
                "pending_clarification": "",
                "dataset_id": active_dataset,  # 每轮重置当前生效数据源
                "subjects": [],
                "sub_queries": [],
                "extracted_features": "",
                "retrieved_docs": {},
                "artworks": [],
                "images": [],
                "candidates": [],
                "reflection_notes": "",
                "web_results": [],
                "retry_count": 0,
                "tool_results": [],
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
                    if node == "web_fallback":
                        web_fallback_used = True
                    if out.get("intent"):
                        intent = out["intent"]
                    if out.get("final_answer"):
                        final_answer = out["final_answer"]
                    if out.get("artworks"):
                        struct_artworks = out["artworks"]
                    for key in ("multi_evidence", "retrieved_docs"):
                        groups = out.get(key) or {}
                        if isinstance(groups, dict):
                            for vals in groups.values():
                                if isinstance(vals, list):
                                    evidence.extend(
                                        v for v in vals if isinstance(v, dict)
                                    )
                    if isinstance(out.get("web_results"), list):
                        evidence.extend(
                            v for v in out["web_results"] if isinstance(v, dict)
                        )
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

        artworks = struct_artworks or tool_artworks
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
        history[-1]["sources"] = _collect_sources(tool_msgs, struct_artworks, evidence)
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
    save_conversation(sid, title, history)
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
        web_fallback=web_fallback_used,
        cancelled=cancelled,
        error=error_msg,
    )
    yield {
        "type": "done",
        "html": history[-1]["content"],
        "session_id": sid,
        "memory": memory_count(),
        "sources": history[-1].get("sources", []),
        "cancelled": cancelled,
        "request_id": request_id,
        "error": error_msg,
    }


# ── 会话 / 偏好：透传给 REST 端点 ──
def sessions(offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
    """侧栏列表（分页）：附带相对时间，返回 (items, total)。"""
    convos, total = list_conversations(limit=limit, offset=offset)
    out = [{**c, "relative": relative_time(c["updated_at"])} for c in convos]
    return out, total


def rename_conversation(sid: str, title: str) -> bool:
    return rename_conversation_db(sid, title)


def conversation(sid: str) -> list[dict]:
    return load_conversation(sid)


def remove_conversation(sid: str) -> None:
    delete_conversation(sid)


def record_attachment(sid: str, doc_id: str, doc_name: str, kind: str) -> dict:
    """把「已上传文档」事件写进会话历史，切换会话/刷新后仍可见。"""
    if not sid or not doc_id:
        return {"ok": False, "error": "缺少会话或文档标识"}
    history = load_conversation(sid)
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
    save_conversation(sid, title or "新对话", history)
    return {"ok": True, "duplicated": False}


def preferences() -> dict:
    return load_preferences(WEB_USER_ID)


def reset_preferences() -> None:
    clear_preferences(WEB_USER_ID)


def preferences_items() -> list[dict]:
    """记忆面板：该用户全部偏好分项（kind/value/weight/updated_at）。"""
    return list_preferences(WEB_USER_ID)


def delete_preference_item(kind: str, value: str) -> bool:
    """记忆面板：单项删除偏好。"""
    return delete_preference(WEB_USER_ID, kind, value)


# ── 文档上传与入库（PDF / 表格） ──

# 并发治理：解析任务信号量（env TASK_PARSE_CONCURRENCY，默认 2），
# 防止多文档同时跑 MinerU/视觉编码打爆单机资源。
_parse_semaphore = threading.Semaphore(
    max(1, int(os.getenv("TASK_PARSE_CONCURRENCY", "2")))
)


def save_upload(filename: str, data: bytes, kb_id: str = "default") -> dict:
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
) -> None:
    """表格后台任务入口：加载 + schema 推断 → 待确认状态。"""
    from src.ingestion.table_pipeline import ingest_table

    with _parse_semaphore:
        if task_id:
            tasks_store.update_task(task_id, status="processing")
        try:
            ingest_table(table_path, doc_id, doc_name=doc_name, kb_id=kb_id)
            if task_id:
                tasks_store.update_task(task_id, status="done", progress=100)
        except Exception:
            logger.exception("ingest_table_doc failed: %s", doc_id)
            if task_id:
                tasks_store.update_task(task_id, status="failed", error="表格解析失败")


def confirm_table(doc_id: str, roles: dict) -> dict:
    """确认/纠正表格 schema：注册生效。"""
    from src.ingestion.table_pipeline import confirm_table_schema

    return confirm_table_schema(doc_id, roles)


def datasets() -> dict:
    """数据源清单（前端切换器）：核心库 + 所有 active 表格。"""
    from src.retrieval.hybrid import get_hybrid_retriever

    hybrid = get_hybrid_retriever()
    items = [{"dataset_id": "core", "name": "核心库（默认）", "kind": "builtin"}]
    for st in documents():
        if st.get("kind") == "table" and st.get("status") == "active":
            items.append(
                {
                    "dataset_id": st["dataset_id"],
                    "name": st.get("display_name")
                    or st.get("doc_name")
                    or st["dataset_id"],
                    "kind": "table",
                    "doc_id": st["doc_id"],
                    "rows": st.get("rows", 0),
                    "supports_timeline": st.get("supports_timeline", False),
                    "supports_recommendation": st.get("supports_recommendation", False),
                }
            )
    return {"active": hybrid.active_dataset, "items": items}


def set_active_dataset(dataset_id: str) -> dict:
    """切换当前生效数据源。"""
    from src.retrieval.hybrid import get_hybrid_retriever

    get_hybrid_retriever().set_active_dataset(dataset_id)
    return {"ok": True, "active": dataset_id}


def restore_tables() -> int:
    """服务启动时恢复已确认的表格数据源。"""
    from src.ingestion.table_pipeline import restore_active_tables

    return restore_active_tables()


def documents() -> list[dict]:
    """文档库列表（新的在前）。"""
    from src.data import documents_store

    return documents_store.list_documents()


def document_status(doc_id: str) -> dict:
    from src.data import documents_store

    return documents_store.get_document(doc_id) or {}


def delete_document(doc_id: str) -> dict:
    """删除文档并级联清理：状态记录、上传文件、向量（PDF）、注册表（Table）。"""
    from src.data import documents_store
    from src.ingestion.pipeline import UPLOADS_DIR, delete_pdf_vectors
    from src.ingestion.table_pipeline import table_dataset_id, unregister_table
    from src.retrieval.hybrid import get_hybrid_retriever

    doc = documents_store.get_document(doc_id)
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
        hybrid = get_hybrid_retriever()
        if hybrid.active_dataset == dataset_id:
            hybrid.active_dataset = "core"
            result["active_dataset_reset"] = "core"

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
    deleted = documents_store.delete_document(doc_id)
    result["db_deleted"] = deleted

    # 5. 从所有会话历史中移除该文档的附件记录
    result["attachment_records_removed"] = remove_attachment_from_all(doc_id)
    return result
