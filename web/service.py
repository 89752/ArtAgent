"""
ArtAgent 服务层 —— 把 LangGraph 推理与渲染逻辑从任何 UI 框架中解耦。

职责：
  · stream_answer(message, sid)：生成器，逐节点产出「助手气泡 HTML」，
    收尾持久化会话并给出最终 payload。API 层只需把每次产出转成 SSE。
  · 会话/偏好读写：直接透传 src.memory，供 REST 端点调用。

设计要点：
  · 与 app.py 的渲染完全一致（思考链折叠 + 内联配图），但不含 gr.update。
  · 助手气泡以 HTML 字符串产出：前端 innerHTML 直接挂载（内容由本服务生成，可信）。
"""

from __future__ import annotations

import json
import html
import base64
from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import get_graph
from src.memory.store import load_preferences, clear_preferences
from src.memory.conversations import (
    save_conversation,
    list_conversations,
    load_conversation,
    delete_conversation,
    relative_time,
)
from src.utils.logging_config import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_USER_ID = "web_user"  # 稳定用户标识：长期偏好跨会话累积

logger = get_logger("web.service")
graph = get_graph()

# ── 场景卡：文案 + 代表画作缩略图（点击直接发问）──
SCENE_CARDS = [
    {"query": "对比莫奈和梵高在色彩运用上的差异",
     "text": "对比莫奈和梵高在色彩运用上的差异？", "image": "28496-early05.jpg"},
    {"query": "梳理透纳的风格演变",
     "text": "透纳的绘画风格经历了怎样的演变？", "image": "40307-110turne.jpg"},
    {"query": "我喜欢维米尔的室内光线，还会喜欢谁",
     "text": "喜欢维米尔的室内光线，还会喜欢哪些画家？", "image": "42649-351seat.jpg"},
    {"query": "卡拉瓦乔的明暗对照法有什么特点",
     "text": "卡拉瓦乔的明暗对照法有哪些特点？", "image": "07480-13fligh.jpg"},
]

_INTENT_LABELS = {
    "comparison": "🆚 跨维度对比", "timeline": "📅 时间线梳理",
    "recommendation": "💡 偏好推荐", "general": "💬 综合问答",
}
_NODE_LABELS = {
    "load_memory": "读取长期记忆", "contextualize": "理解上下文", "classify": "识别意图",
    "comp_decompose": "拆解对比对象与维度", "comp_retrieve": "分组语义检索",
    "comp_synthesize": "逐维度综合对比", "tl_subject": "锁定梳理对象",
    "tl_periods": "按时期收集证据+配图", "tl_synthesize": "编织时间线叙事",
    "rec_extract": "推理风格特征", "rec_search": "特征向量检索",
    "rec_filter": "相关性筛选", "rec_synthesize": "组织推荐理由",
    "general_agent": "ReAct 推理", "general_tools": "执行工具",
    "reflection": "反思答案质量", "web_fallback": "联网兜底检索",
    "save_memory": "写入偏好记忆",
}

# ═══════════════════════════════════════════════════════════════════
# 渲染工具（与 app.py 一致，去除 Gradio 依赖）
# ═══════════════════════════════════════════════════════════════════
def _thumb_data_uri(image_file: str) -> str:
    """本地 SemArt 图片转 base64 data URI（内联可靠，无需静态路由）。"""
    if not image_file:
        return ""
    path = BASE_DIR / "SemArt" / "Images" / image_file
    if not path.exists():
        return ""
    try:
        return "data:image/jpeg;base64," + base64.b64encode(
            path.read_bytes()).decode("ascii")
    except OSError:
        return ""


def _chain_detail(node: str, out: dict) -> str:
    """把某节点输出压成一句人类可读的进度说明。"""
    if not isinstance(out, dict):
        return ""
    if node == "load_memory":
        arts = (out.get("user_preferences") or {}).get("artists") or []
        return (f"已知偏好画家 <span class='hl'>{len(arts)}</span> 位"
                if arts else "暂无历史偏好")
    if node == "contextualize":
        q = (out.get("user_query") or "").strip()
        return f"理解为：<span class='hl'>{q[:40]}{'…' if len(q) > 40 else ''}</span>" if q else ""
    if node == "classify":
        it = out.get("intent", "")
        return f"意图 = <span class='hl'>{_INTENT_LABELS.get(it, it)}</span>"
    if node == "comp_decompose":
        subs = out.get("subjects") or []
        return "对象：" + "、".join(f"<span class='hl'>{s}</span>" for s in subs)
    if node == "comp_retrieve":
        docs = out.get("retrieved_docs") or {}
        return f"检索到 <span class='hl'>{sum(len(v) for v in docs.values())}</span> 条评论证据"
    if node == "rec_extract":
        feat = (out.get("extracted_features") or "").strip()
        return (f"推理特征：<span class='hl'>{feat[:48]}{'…' if len(feat) > 48 else ''}</span>"
                if feat else "")
    if node == "rec_search":
        return f"匹配候选 <span class='hl'>{len(out.get('artworks') or [])}</span> 幅"
    if node == "rec_filter":
        cands = out.get("candidates") or []
        names = "、".join(c.get("author", "") for c in cands[:4])
        return (f"筛出 <span class='hl'>{len(cands)}</span> 位：{names}"
                if cands else "未筛出匹配画家")
    if node == "tl_subject":
        subs = out.get("subjects") or []
        return f"对象：<span class='hl'>{subs[0]}</span>" if subs else ""
    if node == "tl_periods":
        return f"覆盖 <span class='hl'>{len(out.get('retrieved_docs') or {})}</span> 个时期"
    if node == "general_agent":
        msgs = out.get("messages") or []
        if msgs and getattr(msgs[-1], "tool_calls", None):
            return "调用工具：" + "、".join(
                f"<span class='hl'>{t.get('name')}</span>" for t in msgs[-1].tool_calls)
        return "直接作答"
    if node == "reflection":
        return ("结论：<span class='hl'>通过</span>"
                if out.get("reflection_notes") == "PASS"
                else "结论：<span class='hl'>信息不足，触发兜底</span>")
    if node == "web_fallback":
        return f"联网补充 <span class='hl'>{len(out.get('web_results') or [])}</span> 条"
    if node == "save_memory":
        return "偏好已持久化"
    return ""


def _chain_html(steps: list[dict], done: bool) -> str:
    html = ""
    for i, s in enumerate(steps):
        pending = (i == len(steps) - 1) and not done
        dot = "chain-dot pending" if pending else "chain-dot"
        name = _NODE_LABELS.get(s["node"], s["node"])
        detail = (f'<div class="chain-detail">{s["detail"]}</div>'
                  if s.get("detail") else "")
        html += (f'<div class="chain-step"><span class="{dot}"></span>'
                 f'<div class="chain-name">{name}</div>{detail}</div>')
    return html


def _think_box(steps: list[dict], done: bool) -> str:
    if not steps:
        return ""
    open_attr = "" if done else " open"
    label = "思考过程" if done else "正在思考…"
    return (f'<details class="think-box"{open_attr}><summary>{label}'
            f"（{len(steps)} 步）</summary>"
            f'<div class="think-body">{_chain_html(steps, done)}</div></details>')


def _artwork_grid(artworks: list[dict], with_thumbs: bool) -> str:
    if not artworks or not with_thumbs:
        return ""
    cells = ""
    for aw in artworks[:4]:
        uri = _thumb_data_uri(aw.get("image_file", ""))
        if not uri:
            continue
        title = (aw.get("title") or "")[:24]
        author = (aw.get("author") or "")[:22]
        cells += (f'<figure class="aw-card"><img src="{uri}" alt="{title}"/>'
                  f'<figcaption class="aw-cap"><b>{title}</b>{author}</figcaption></figure>')
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
    return _think_box(steps, done) + _answer_block(answer) + _artwork_grid(artworks, with_thumbs)


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
                if isinstance(item, dict) and item.get("title") and not item.get("source"):
                    artworks.append({
                        "title": item.get("title", ""), "author": item.get("author", ""),
                        "date": item.get("date", ""), "image_file": item.get("image_file", ""),
                    })
    return artworks


def memory_count() -> int:
    """已记住的偏好项数（画家 + 风格）。"""
    prefs = load_preferences(WEB_USER_ID)
    return len(prefs.get("artists") or []) + len(prefs.get("styles") or [])


# ═══════════════════════════════════════════════════════════════════
# 流式 Agent 调用
# ═══════════════════════════════════════════════════════════════════
def stream_answer(message: str, sid: str) -> Iterator[dict]:
    """
    生成器：逐节点产出事件字典，API 层转 SSE。
      · {"type": "delta", "html": <助手气泡 HTML>}           —— 流式刷新
      · {"type": "done",  "html": ..., "session_id": sid,
         "memory": <偏好数>}                                  —— 收尾
    """
    message = (message or "").strip()
    if not message:
        yield {"type": "done", "html": "", "session_id": sid, "memory": memory_count()}
        return

    # 历史消息在库中（前端无状态）：读出→追加本轮→回写
    history = load_conversation(sid)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": _think_box([], done=False)},
    ]
    steps: list[dict] = []
    yield {"type": "delta", "html": history[-1]["content"]}

    # Stage 5：当前生效数据源由服务端单例持有（前端切换器调 /api/dataset/active
    # 改变它）；每轮从单例读进 state，重置清单纪律不变
    from src.retrieval.hybrid import get_hybrid_retriever

    active_dataset = get_hybrid_retriever().active_dataset

    intent, final_answer = "", ""
    struct_artworks: list[dict] = []
    tool_artworks: list[dict] = []

    try:
        for chunk in graph.stream(
            {
                # messages 由 checkpointer 跨轮累积；其余标量每轮重置，
                # 避免上一轮的 intent/subjects/检索结果/retry_count 串味。
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": WEB_USER_ID,
                "intent": "",
                "dataset_id": active_dataset,  # Stage 2/5：每轮重置当前生效数据源
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
            for node, out in chunk.items():
                if node == "__interrupt__":
                    continue
                steps.append({"node": node, "detail": _chain_detail(node, out)})
                if isinstance(out, dict):
                    if out.get("intent"):
                        intent = out["intent"]
                    if out.get("final_answer"):
                        final_answer = out["final_answer"]
                    if out.get("artworks"):
                        struct_artworks = out["artworks"]
                    if out.get("messages"):
                        tool_artworks.extend(_parse_artworks_from_messages(out["messages"]))
                history[-1]["content"] = _assistant_bubble(steps, "", [], False, done=False)
                yield {"type": "delta", "html": history[-1]["content"]}

        artworks = struct_artworks or tool_artworks
        with_thumbs = intent in ("timeline", "recommendation", "general")
        reply = final_answer or "（未能生成回答，请重试）"
        history[-1]["content"] = _assistant_bubble(steps, reply, artworks, with_thumbs, done=True)
    except Exception as e:  # noqa: BLE001 — 面向用户兜底，避免整页崩溃
        logger.exception("graph.stream failed: %s", e)
        steps.append({"node": "error", "detail": f"<span class='hl'>{type(e).__name__}</span>"})
        history[-1]["content"] = _assistant_bubble(
            steps, "😔 抱歉，处理时出错了。可能是模型接口超时或未配置 API Key，请稍后重试。",
            [], False, done=True)

    title = next((m["content"] for m in history if m["role"] == "user"), message)
    save_conversation(sid, title, history)
    yield {"type": "done", "html": history[-1]["content"],
           "session_id": sid, "memory": memory_count()}


# ── 会话 / 偏好：透传给 REST 端点 ──
def sessions() -> list[dict]:
    """侧栏列表：附带相对时间。"""
    out = []
    for c in list_conversations():
        out.append({**c, "relative": relative_time(c["updated_at"])})
    return out


def conversation(sid: str) -> list[dict]:
    return load_conversation(sid)


def remove_conversation(sid: str) -> None:
    delete_conversation(sid)


def preferences() -> dict:
    return load_preferences(WEB_USER_ID)


def reset_preferences() -> None:
    clear_preferences(WEB_USER_ID)


# ── 文档上传与入库（Stage 3 PDF / Stage 5 表格） ──
def save_upload(filename: str, data: bytes, kb_id: str = "default") -> dict:
    """把上传文件存到 uploads/{kb_id}/{doc_id}/；按类型路由存储名。

    PDF → document.pdf（Stage 3 路径不变）；表格 → table{原扩展名}（Stage 5）。
    调用方须先用 classify_upload 判型，本函数不重复校验。
    """
    import uuid

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
    return {
        "doc_id": doc_id,
        "doc_name": filename,
        "kind": kind,
        "file_path": str(file_path),
        "kb_id": kb_id,
    }


def ingest_document(doc_id: str, doc_name: str, pdf_path: str, kb_id: str) -> None:
    """后台任务入口（BackgroundTasks）：跑入库流水线，异常已落 failed 状态。"""
    from src.ingestion.pipeline import ingest_pdf

    try:
        ingest_pdf(pdf_path, doc_id, doc_name=doc_name, kb_id=kb_id)
    except Exception:
        logger.exception("ingest_document failed: %s", doc_id)


def ingest_table_doc(doc_id: str, doc_name: str, table_path: str, kb_id: str) -> None:
    """表格后台任务入口（Stage 5）：加载 + schema 推断 → 待确认状态。"""
    from src.ingestion.table_pipeline import ingest_table

    try:
        ingest_table(table_path, doc_id, doc_name=doc_name, kb_id=kb_id)
    except Exception:
        logger.exception("ingest_table_doc failed: %s", doc_id)


def confirm_table(doc_id: str, roles: dict) -> dict:
    """确认/纠正表格 schema（Stage 5）：注册生效。"""
    from src.ingestion.table_pipeline import confirm_table_schema

    return confirm_table_schema(doc_id, roles)


def datasets() -> dict:
    """数据源清单（Stage 5 前端切换器）：semart + 所有 active 表格。"""
    from src.retrieval.hybrid import get_hybrid_retriever

    hybrid = get_hybrid_retriever()
    items = [{"dataset_id": "semart", "name": "SemArt 画作库（默认）", "kind": "builtin"}]
    for st in documents():
        if st.get("kind") == "table" and st.get("status") == "active":
            items.append({
                "dataset_id": st["dataset_id"],
                "name": st.get("display_name") or st.get("doc_name") or st["dataset_id"],
                "kind": "table",
                "doc_id": st["doc_id"],
                "rows": st.get("rows", 0),
                "supports_timeline": st.get("supports_timeline", False),
                "supports_recommendation": st.get("supports_recommendation", False),
            })
    return {"active": hybrid.active_dataset, "items": items}


def set_active_dataset(dataset_id: str) -> dict:
    """切换当前生效数据源（Stage 5）。"""
    from src.retrieval.hybrid import get_hybrid_retriever

    get_hybrid_retriever().set_active_dataset(dataset_id)
    return {"ok": True, "active": dataset_id}


def restore_tables() -> int:
    """服务启动时恢复已确认的表格数据源（Stage 5）。"""
    from src.ingestion.table_pipeline import restore_active_tables

    return restore_active_tables()


def documents() -> list[dict]:
    """文档库列表（新的在前）。"""
    from src.ingestion.pipeline import list_doc_status

    return sorted(
        list_doc_status(), key=lambda d: d.get("started_at", ""), reverse=True
    )


def document_status(doc_id: str) -> dict:
    from src.ingestion.pipeline import get_doc_status

    return get_doc_status(doc_id) or {}
