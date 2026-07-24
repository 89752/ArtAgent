"""
ArtAgent Web 界面 v5 —— 双栏布局，让「思考」可见又不喧宾夺主。

设计（复刻参考稿）：
  · 左侧深海蓝侧栏：新建对话（金）+ 真实历史会话列表（可切换）+ 用户卡
  · 右侧米白主区：
      - 欢迎态：徽标 + 标题 + 4 张场景卡（点击直接发问）
      - 对话态：顶部标题栏 + 气泡对话 + 底部输入栏
  · 决策链（意图→拆解→检索→反思）折叠进助手气泡的「🧠 思考过程」
  · 相关画作内联在气泡底部网格展示
样式拆分至 static/style.css。
"""

import base64
import json
import os
import uuid
from pathlib import Path

# 仅对本地地址绕过系统代理：让 Gradio 启动自检（startup-events）不走代理返回 502，
# 又保留代理用于访问外部 API（DeepSeek 等）。不能用 "*"，否则外部 API 也会被断开。
_LOCAL_NOPROXY = "localhost,127.0.0.1,0.0.0.0,::1"
for _k in ("NO_PROXY", "no_proxy"):
    _parts = [p for p in os.environ.get(_k, "").split(",") if p.strip()]
    for _addr in _LOCAL_NOPROXY.split(","):
        if _addr not in _parts:
            _parts.append(_addr)
    os.environ[_k] = ",".join(_parts)

# BGE 向量模型已随索引本地缓存，强制离线加载，避免首次检索时联网自检超时卡顿。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 以本文件位置锚定项目根，图片/样式路径不受启动时 CWD 影响。
BASE_DIR = Path(__file__).resolve().parent

import gradio as gr
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import get_graph
from src.memory.store import load_preferences, clear_preferences
from src.memory.conversations import (
    save_conversation, list_conversations, load_conversation,
    delete_conversation, relative_time,
)
from src.utils.logging_config import get_logger

logger = get_logger("webui")
graph = get_graph()

WEB_USER_ID = "web_user"          # 稳定用户标识：长期偏好记忆跨会话持久化
CSS = (BASE_DIR / "static" / "style.css").read_text(encoding="utf-8")

# ── SVG 徽标：天使像（简笔），复用于欢迎/标题栏/头像 ──
EMBLEM_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke="#b8934a" stroke-width="2">
<path d="M50 6 C68 6 82 20 82 40 C82 62 66 82 50 94 C34 82 18 62 18 40 C18 20 32 6 50 6 Z"/>
<circle cx="50" cy="34" r="12"/>
<path d="M38 52 C38 44 44 40 50 40 C56 40 62 44 62 52 L62 70 L38 70 Z"/>
<path d="M38 54 C28 50 22 54 20 62 C30 60 34 60 38 64"/>
<path d="M62 54 C72 50 78 54 80 62 C70 60 66 60 62 64"/>
</g></svg>"""

USER_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke="#c9a86a" stroke-width="4">
<circle cx="50" cy="36" r="18"/>
<path d="M18 88 C18 64 34 56 50 56 C66 56 82 64 82 88"/>
</g></svg>"""


def _svg_data_uri(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + b64


EMBLEM_URI = _svg_data_uri(EMBLEM_SVG)
USER_URI = _svg_data_uri(USER_SVG)

# 头像走文件路径（Gradio avatar_images 不接受 data URI，会当成文件路径去 hash）。
_STATIC = BASE_DIR / "static"
_STATIC.mkdir(exist_ok=True)
(_STATIC / "emblem.svg").write_text(EMBLEM_SVG, encoding="utf-8")
(_STATIC / "user.svg").write_text(USER_SVG, encoding="utf-8")
EMBLEM_FILE = str(_STATIC / "emblem.svg")
USER_FILE = str(_STATIC / "user.svg")

# ── 场景卡：文案 + 代表画作缩略图（点击直接发问）──
SCENE_CARDS = [
    {"query": "对比莫奈和梵高在色彩运用上的差异",
     "text": "对比莫奈和梵高在\n色彩运用上的差异？", "image": "28496-early05.jpg"},
    {"query": "梳理透纳的风格演变",
     "text": "透纳的绘画风格\n经历了怎样的演变？", "image": "40307-110turne.jpg"},
    {"query": "我喜欢维米尔的室内光线，还会喜欢谁",
     "text": "喜欢维米尔的室内光线，\n还会喜欢哪些画家？", "image": "42649-351seat.jpg"},
    {"query": "卡拉瓦乔的明暗对照法有什么特点",
     "text": "卡拉瓦乔的明暗对照法\n有哪些特点？", "image": "07480-13fligh.jpg"},
]

_INTENT_LABELS = {
    "comparison": "🆚 跨维度对比", "timeline": "📅 时间线梳理",
    "recommendation": "💡 偏好推荐", "general": "💬 综合问答",
}
_NODE_LABELS = {
    "load_memory": "读取长期记忆", "classify": "识别意图",
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
# 渲染工具
# ═══════════════════════════════════════════════════════════════════
def _thumb_data_uri(image_file: str) -> str:
    """本地 SemArt 图片转 base64 data URI（内联可靠，无需 allowed_paths）。"""
    if not image_file:
        return ""
    path = BASE_DIR / "SemArt" / "Images" / image_file
    if not path.exists():
        return ""
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""


def _chain_detail(node: str, out: dict) -> str:
    """把某节点输出压成一句人类可读的进度说明。"""
    if not isinstance(out, dict):
        return ""
    if node == "load_memory":
        arts = (out.get("user_preferences") or {}).get("artists") or []
        return f"已知偏好画家 <span class='hl'>{len(arts)}</span> 位" if arts else "暂无历史偏好"
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
        return f"推理特征：<span class='hl'>{feat[:48]}{'…' if len(feat) > 48 else ''}</span>" if feat else ""
    if node == "rec_search":
        return f"匹配候选 <span class='hl'>{len(out.get('artworks') or [])}</span> 幅"
    if node == "rec_filter":
        cands = out.get("candidates") or []
        names = "、".join(c.get("author", "") for c in cands[:4])
        return f"筛出 <span class='hl'>{len(cands)}</span> 位：{names}" if cands else "未筛出匹配画家"
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
    """决策链步骤序列 → HTML（未完成时末步用脉冲点）。"""
    html = ""
    for i, s in enumerate(steps):
        pending = (i == len(steps) - 1) and not done
        dot = "chain-dot pending" if pending else "chain-dot"
        name = _NODE_LABELS.get(s["node"], s["node"])
        detail = f'<div class="chain-detail">{s["detail"]}</div>' if s.get("detail") else ""
        html += (f'<div class="chain-step"><span class="{dot}"></span>'
                 f'<div class="chain-name">{name}</div>{detail}</div>')
    return html


def _think_box(steps: list[dict], done: bool) -> str:
    """折叠的「思考过程」；进行中默认展开，完成后折叠。"""
    if not steps:
        return ""
    open_attr = "" if done else " open"
    label = "思考过程" if done else "正在思考…"
    return (f'<details class="think-box"{open_attr}><summary>{label}'
            f'（{len(steps)} 步）</summary>'
            f'<div class="think-body">{_chain_html(steps, done)}</div></details>')


def _artwork_grid(artworks: list[dict], with_thumbs: bool) -> str:
    """气泡底部内联画作网格。"""
    if not artworks or not with_thumbs:
        return ""
    cells = ""
    for aw in artworks[:4]:
        uri = _thumb_data_uri(aw.get("image_file", ""))
        if not uri:
            continue
        title = (aw.get("title") or "")[:24]
        author = (aw.get("author") or "")[:22]
        cells += (f'<div class="aw-card"><img src="{uri}"/>'
                  f'<div class="aw-cap"><b>{title}</b>{author}</div></div>')
    return f'<div class="aw-grid">{cells}</div>' if cells else ""


def _assistant_bubble(steps: list[dict], answer: str, artworks: list[dict],
                      with_thumbs: bool, done: bool) -> str:
    """组装助手气泡：思考折叠 + 正文 + 内联配图。"""
    return _think_box(steps, done) + (answer or "") + _artwork_grid(artworks, with_thumbs)


def _parse_artworks_from_messages(messages: list) -> list[dict]:
    """general 分支从 ToolMessage 解析画作（结构化分支直接用 state.artworks）。"""
    artworks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            for item in (data if isinstance(data, list) else [data]):
                if isinstance(item, dict) and item.get("title"):
                    artworks.append({
                        "title": item.get("title", ""), "author": item.get("author", ""),
                        "date": item.get("date", ""), "image_file": item.get("image_file", ""),
                    })
    return artworks


def _memory_line() -> str:
    """用户卡副标题：显示已记住的偏好数（长期记忆 S5）。"""
    prefs = load_preferences(WEB_USER_ID)
    n = len(prefs.get("artists") or []) + len(prefs.get("styles") or [])
    return f"已记住 {n} 项偏好" if n else "Lv.3 学徒"

# ═══════════════════════════════════════════════════════════════════
# 流式 Agent 调用 + 会话管理
# ═══════════════════════════════════════════════════════════════════
def respond(message: str, history: list, sid: str):
    """
    生成器：graph.stream 逐节点产出。每步刷新助手气泡内的折叠思考。
    yield (chatbot, session_id, 历史列表刷新信号)。
    """
    message = (message or "").strip()
    if not message:
        yield history, sid, gr.update()
        return
    if not sid:
        sid = str(uuid.uuid4())

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": _think_box([], done=False)},
    ]
    steps: list[dict] = []
    yield history, sid, gr.update()

    intent, final_answer = "", ""
    struct_artworks: list[dict] = []
    tool_artworks: list[dict] = []

    try:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=message)], "user_query": message,
             "user_id": WEB_USER_ID, "tool_results": [], "final_answer": ""},
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
                yield history, sid, gr.update()

        # 收尾：正文 + 内联配图 + 折叠思考
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

    # 持久化本轮会话（标题取首条用户消息）
    title = next((m["content"] for m in history if m["role"] == "user"), message)
    save_conversation(sid, title, history)
    yield history, sid, gr.update()


def new_session():
    """开新会话：清空对话、生成新 sid、回到欢迎态。"""
    return [], str(uuid.uuid4())


def load_session(sid: str):
    """从历史加载某会话到对话区。

    注意：历史按钮位于 @gr.render 内，输出 session_id 会触发列表重渲染并销毁
    触发按钮，导致其 .then(_view) 无法执行。因此这里一次性返回值 + 显隐更新，
    不再依赖 .then 链。
    """
    hist = load_conversation(sid)
    has = bool(hist)
    return (
        gr.update(value=hist, visible=has),   # chatbot: 值 + 显隐
        sid,                                  # session_id
        gr.update(visible=not has),           # welcome
        gr.update(visible=has),               # top_bar
    )


def clear_current(sid: str):
    """清除当前会话（从库中删除并回到欢迎态）。"""
    if sid:
        delete_conversation(sid)
    return [], str(uuid.uuid4())

# ═══════════════════════════════════════════════════════════════════
# 界面
# ═══════════════════════════════════════════════════════════════════
def _view(history: list):
    """依据是否有对话，切换 欢迎态 / 对话态 的显隐。"""
    has_chat = bool(history)
    return (
        gr.update(visible=not has_chat),  # welcome
        gr.update(visible=has_chat),      # top_bar
        gr.update(visible=has_chat),      # chatbot
    )


# 场景卡缩略图：动态注入 #card-N::before 的 background-image
_card_style = "<style>\n"
for _i, _c in enumerate(SCENE_CARDS):
    _uri = _thumb_data_uri(_c["image"])
    if _uri:
        _card_style += f'#card-{_i}::before {{ background-image: url("{_uri}"); }}\n'
_card_style += "</style>"


with gr.Blocks(title="西方艺术智能助手", analytics_enabled=False) as demo:
    session_id = gr.State(value=str(uuid.uuid4()))
    hist_trigger = gr.State(value=0)      # bump 后触发历史列表重渲染
    gr.HTML(_card_style)

    with gr.Row(elem_id="app-row"):
        # ══════════ 左侧侧栏 ══════════
        with gr.Column(elem_id="sidebar", scale=0, min_width=300):
            btn_new = gr.Button("＋  新建对话", elem_id="btn-new")
            gr.HTML('<div class="side-heading">🕮 历史对话</div>')

            with gr.Column(elem_id="history-list"):
                @gr.render(inputs=[hist_trigger, session_id])
                def _render_history(_trigger, cur_sid):
                    convs = list_conversations()
                    if not convs:
                        gr.HTML('<div style="color:#6d7c94;font-size:0.85em;'
                                'padding:10px 12px;">还没有历史对话</div>')
                        return
                    for c in convs:
                        cls = "hist-btn active" if c["session_id"] == cur_sid else "hist-btn"
                        label = f'💬  {c["title"][:16]}    ·  {relative_time(c["updated_at"])}'
                        b = gr.Button(label, elem_classes=cls.split())
                        b.click(load_session, inputs=gr.State(c["session_id"]),
                                outputs=[chatbot, session_id, welcome, top_bar])

            gr.HTML(
                f'<div id="user-card"><div class="u-avatar">{USER_SVG}</div>'
                f'<div><div class="u-name">学者</div>'
                f'<div class="u-level" id="u-level">{_memory_line()}</div></div></div>')

        # ══════════ 右侧主区 ══════════
        with gr.Column(elem_id="main-col", scale=1):

            # 对话态顶栏（初始隐藏）
            with gr.Row(visible=False) as top_bar:
                gr.HTML(
                    f'<div id="top-bar"><img class="tb-emblem" src="{EMBLEM_URI}"/>'
                    '<div class="tb-titles"><div class="tb-title">'
                    '<span class="flr">❖</span>西方艺术智能助手<span class="flr">❖</span>'
                    '</div><div class="tb-sub">探 索 · 学 习 · 欣 赏</div></div></div>')
                btn_clear = gr.Button("🗑 清除对话", elem_id="btn-clear", scale=0)

            # 欢迎态（初始显示）
            with gr.Column(elem_id="welcome", visible=True) as welcome:
                gr.HTML(
                    f'<img class="w-emblem" src="{EMBLEM_URI}"/>'
                    '<div class="w-title"><span class="flr">❖</span>'
                    '西方艺术智能助手<span class="flr">❖</span></div>'
                    '<div class="w-sub">探 索 · 学 习 · 欣 赏</div>'
                    '<div class="w-divider"></div>'
                    '<div class="w-greeting">您好！我是您的西方艺术智能助手。</div>'
                    '<div class="w-hint">您可以向我提问，关于艺术作品、艺术家、'
                    '风格流派、历史背景等问题。</div>')
                with gr.Row(elem_id="cards-row"):
                    card_btns = []
                    for i, c in enumerate(SCENE_CARDS):
                        b = gr.Button(c["text"], elem_id=f"card-{i}",
                                      elem_classes=["scene-card"])
                        card_btns.append(b)

            # 对话态聊天区（初始隐藏）
            chatbot = gr.Chatbot(
                value=[], show_label=False, elem_id="chatbot",
                render_markdown=True, sanitize_html=False, visible=False,
                avatar_images=(USER_FILE, EMBLEM_FILE),
            )

            # 输入栏（常驻）
            with gr.Row(elem_id="input-bar"):
                msg_box = gr.Textbox(
                    placeholder="输入您的问题或想法…", lines=1, max_lines=4,
                    scale=1, show_label=False, autofocus=True, elem_id="msg-box",
                )
                btn_send = gr.Button("➤", elem_id="btn-send", scale=0)

            gr.HTML('<div id="disclaimer">⟢ 内容由 AI 生成，仅供学习参考，'
                    '请结合权威资料进行深入研究。 ⟣</div>')

    # ══════════ 事件绑定 ══════════
    def _bump(n):
        return (n or 0) + 1

    send_out = [chatbot, session_id, hist_trigger]

    def _send_chain(trigger_component):
        """发送→流式回答→清空输入→切到对话态→刷新历史列表。"""
        ev = trigger_component
        ev = ev.then(_view, inputs=chatbot, outputs=[welcome, top_bar, chatbot])
        ev = ev.then(lambda: "", outputs=msg_box)
        ev = ev.then(_bump, inputs=hist_trigger, outputs=hist_trigger)
        return ev

    _send_chain(btn_send.click(respond, [msg_box, chatbot, session_id], send_out))
    _send_chain(msg_box.submit(respond, [msg_box, chatbot, session_id], send_out))

    # 场景卡：点击直接以该问题发起对话
    for i, c in enumerate(SCENE_CARDS):
        _send_chain(card_btns[i].click(
            respond, [gr.State(c["query"]), chatbot, session_id], send_out))

    # 新建 / 清除
    btn_new.click(new_session, outputs=[chatbot, session_id]).then(
        _view, inputs=chatbot, outputs=[welcome, top_bar, chatbot]).then(
        _bump, inputs=hist_trigger, outputs=hist_trigger)
    btn_clear.click(clear_current, inputs=session_id, outputs=[chatbot, session_id]).then(
        _view, inputs=chatbot, outputs=[welcome, top_bar, chatbot]).then(
        _bump, inputs=hist_trigger, outputs=hist_trigger)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                inbrowser=True, css=CSS, allowed_paths=[str(_STATIC)])




