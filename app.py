"""
ArtAgent Web 界面 v3 — 还原第二版设计
亮色系：米白 + 深海蓝 + 金色
三栏布局，一屏完整展示，无工具调用区
"""

import json
import uuid

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.graph import get_graph

graph = get_graph()

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
CSS = """
/* ── 全局重置 ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { height: 100%; overflow: hidden; }

body {
    height: 100%;
    overflow: hidden;
    background: #f4f0e8 !important;
    font-family: 'Georgia', 'STSong', serif !important;
    color: #1a2d4a !important;
}

/* 撑满视口，禁止整体滚动 */
.gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    max-width: 100vw !important;
    width: 100vw !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #f4f0e8 !important;
    overflow: hidden !important;
}

/* Gradio 内层容器全部撑满 */
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > .padding {
    height: 100% !important;
    max-height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

/* 隐藏多余元素 */
footer { display: none !important; }
.gradio-container .label-wrap { display: none !important; }
.generating { display: none !important; }

/* ── 顶部标题栏 ── */
#header {
    height: 62px;
    background: #1a2d4a;
    border-bottom: 3px solid #c9a84c;
    padding: 0 28px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
}
#header-logo { font-size: 1.6em; line-height: 1; }
#header-title {
    font-size: 1.4em;
    font-weight: bold;
    color: #c9a84c;
    letter-spacing: 0.05em;
}
#header-sub {
    font-size: 0.78em;
    color: #8a9bb5;
    margin-top: 3px;
}

/* ── 主体行：撑满剩余高度 ── */
#main-row {
    height: calc(100vh - 62px) !important;
    display: flex !important;
    overflow: hidden !important;
    flex-wrap: nowrap !important;
}

/* ── 左侧面板 ── */
#left-panel {
    width: 200px !important;
    min-width: 200px !important;
    max-width: 200px !important;
    height: 100%;
    background: #ffffff;
    border-right: 1px solid #ddd6c8;
    overflow-y: auto;
    padding: 16px 10px;
    flex-shrink: 0;
}

.nav-label {
    font-size: 0.66em;
    color: #c9a84c;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0 6px;
    margin-bottom: 6px;
    margin-top: 4px;
    display: block;
}

.nav-btn {
    display: flex !important;
    align-items: center;
    gap: 8px;
    width: 100% !important;
    padding: 8px 10px !important;
    border-radius: 7px !important;
    border: none !important;
    background: transparent !important;
    color: #4a6080 !important;
    font-size: 0.84em !important;
    font-family: 'Georgia', serif !important;
    cursor: pointer;
    text-align: left !important;
    transition: all 0.15s;
    margin-bottom: 2px !important;
    box-shadow: none !important;
}
.nav-btn:hover {
    background: #f0ece3 !important;
    color: #1a2d4a !important;
}
.nav-btn-active {
    background: #eef2f9 !important;
    color: #1a2d4a !important;
    font-weight: bold !important;
    border-left: 3px solid #c9a84c !important;
    padding-left: 7px !important;
}

.nav-divider {
    border: none;
    border-top: 1px solid #e8e0d0;
    margin: 12px 4px;
}

.theme-wrap { display: flex; flex-wrap: wrap; gap: 5px; padding: 2px 2px; }
.theme-chip {
    padding: 3px 9px;
    border-radius: 14px;
    border: 1px solid #ddd6c8;
    background: #faf7f2;
    color: #5a7090;
    font-size: 0.74em;
    cursor: pointer;
    transition: all 0.15s;
    font-family: 'Georgia', serif;
}
.theme-chip:hover { border-color: #c9a84c; color: #a07820; background: #fff8ec; }

/* ── 中间聊天区 ── */
#center-col {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    background: #faf7f2;
}

/* Chatbot 撑满可用空间 */
#chatbot {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    background: #faf7f2 !important;
    border: none !important;
    border-radius: 0 !important;
}
#chatbot > div { height: 100% !important; }

/* 用户气泡 */
.message-wrap .user > div,
[data-testid="user"] .prose {
    background: #1a2d4a !important;
    color: #f4f0e8 !important;
    border-radius: 14px 14px 4px 14px !important;
    border: none !important;
}
/* AI 气泡 */
.message-wrap .bot > div,
[data-testid="bot"] .prose {
    background: #ffffff !important;
    color: #1a2d4a !important;
    border: 1px solid #ddd6c8 !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* 相关画作带 */
#related-band {
    background: #fff8ec;
    border-top: 1px solid #e8dfc8;
    padding: 8px 16px;
    flex-shrink: 0;
    min-height: 0;
}
.related-label {
    font-size: 0.68em;
    color: #c9a84c;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.aw-card {
    display: inline-block;
    width: 118px;
    vertical-align: top;
    background: #ffffff;
    border: 1px solid #ddd6c8;
    border-radius: 7px;
    padding: 7px 9px;
    margin-right: 6px;
    font-size: 0.75em;
    color: #5a7090;
    cursor: pointer;
    transition: border-color 0.15s;
}
.aw-card:hover { border-color: #c9a84c; }
.aw-title {
    font-weight: bold;
    color: #1a2d4a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
    font-size: 0.9em;
}
.aw-date { color: #c9a84c; }
.aw-thumb {
    width: 100%;
    height: 66px;
    object-fit: cover;
    border-radius: 4px;
    margin-bottom: 5px;
    display: block;
}
.intent-badge {
    display: inline-block;
    background: #1a2d4a;
    color: #c9a84c;
    font-size: 0.82em;
    padding: 1px 8px;
    border-radius: 10px;
    margin-left: 8px;
    letter-spacing: normal;
    text-transform: none;
    vertical-align: middle;
}

/* 输入区 */
#input-bar {
    background: #ffffff;
    border-top: 1px solid #ddd6c8;
    padding: 10px 14px;
    flex-shrink: 0;
    display: flex;
    align-items: flex-end;
    gap: 8px;
}
#msg-box textarea {
    background: #faf7f2 !important;
    border: 1px solid #d0c8b8 !important;
    color: #1a2d4a !important;
    border-radius: 8px !important;
    font-size: 0.92em !important;
    padding: 9px 13px !important;
    resize: none !important;
    font-family: 'Georgia', serif !important;
}
#msg-box textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px #c9a84c30 !important;
    outline: none !important;
}
#btn-send {
    background: #c9a84c !important;
    color: #1a2d4a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    font-size: 0.9em !important;
    height: 46px !important;
    min-width: 72px !important;
    box-shadow: none !important;
}
#btn-send:hover { background: #dfc060 !important; }
#btn-new {
    background: transparent !important;
    border: 1px solid #d0c8b8 !important;
    color: #6a7f98 !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    height: 46px !important;
    min-width: 64px !important;
    box-shadow: none !important;
}
#btn-new:hover { border-color: #c9a84c !important; color: #a07820 !important; }

/* ── 右侧面板 ── */
#right-panel {
    width: 220px !important;
    min-width: 220px !important;
    max-width: 220px !important;
    height: 100%;
    background: #ffffff;
    border-left: 1px solid #ddd6c8;
    overflow-y: auto;
    padding: 16px 13px;
    flex-shrink: 0;
}

.panel-label {
    font-size: 0.67em;
    color: #c9a84c;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 7px;
    border-bottom: 1px solid #e8dfc8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 5px;
}

.rec-card {
    background: #faf7f2;
    border: 1px solid #ddd6c8;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 7px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.rec-card:hover { border-color: #c9a84c; }
.rec-title { font-size: 0.86em; color: #1a2d4a; font-weight: bold; margin-bottom: 3px; }
.rec-sub   { font-size: 0.73em; color: #7a8fa8; }

.panel-divider {
    border: none;
    border-top: 1px solid #e8dfc8;
    margin: 13px 0;
}

.eq-btn {
    display: block !important;
    width: 100% !important;
    text-align: left !important;
    background: #faf7f2 !important;
    border: 1px solid #ddd6c8 !important;
    color: #4a6080 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    font-size: 0.79em !important;
    margin-bottom: 5px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    font-family: 'Georgia', serif !important;
    box-shadow: none !important;
}
.eq-btn:hover {
    border-color: #c9a84c !important;
    color: #1a2d4a !important;
    background: #fff8ec !important;
}
"""

# ═══════════════════════════════════════════════════════════════════
# 静态数据
# ═══════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("💬", "对话助手", True),
    ("🖼", "艺术作品库", False),
    ("🎨", "艺术家与工坊", False),
    ("🏛", "风格与流派", False),
    ("📅", "艺术时间线", False),
    ("📚", "学习资源", False),
]

THEMES = [
    "🌅 印象派风景",
    "⛪ 文艺复兴",
    "🌸 花卉静物",
    "⚔️ 战争历史",
    "👤 肖像画",
    "🌙 夜景光影",
    "🏛️ 神话题材",
    "🌊 海景画",
]

RECOMMENDS = [
    {"title": "The Starry Night", "author": "Van Gogh", "period": "后印象派 · 1889"},
    {
        "title": "The Night Watch",
        "author": "Rembrandt",
        "period": "荷兰黄金时代 · 1642",
    },
    {
        "title": "Judith Slaying Holofernes",
        "author": "Caravaggio",
        "period": "巴洛克 · 1599",
    },
]

EXAMPLES = [
    "找几幅梵高的画介绍一下",
    "什么是巴洛克风格？",
    "对比《星夜》和《向日葵》",
    "伦勃朗的绘画特点",
    "分析《星夜》的构图",
    "卡拉瓦乔的光影风格",
    "找几幅印象派风景画",
    "维米尔的艺术成就",
]


# ═══════════════════════════════════════════════════════════════════
# 工具结果解析
# ═══════════════════════════════════════════════════════════════════
import base64
from pathlib import Path

_INTENT_LABELS = {
    "comparison": "🆚 跨维度对比",
    "timeline": "📅 时间线梳理",
    "recommendation": "💡 偏好推荐",
    "general": "💬 综合问答",
}


def _parse_artworks_from_messages(messages: list) -> list[dict]:
    """从 general 分支的 ToolMessage 中解析画作（结构化分支直接用 state.artworks）。"""
    artworks = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and item.get("title"):
                        artworks.append(
                            {
                                "title": item.get("title", ""),
                                "author": item.get("author", ""),
                                "date": item.get("date", ""),
                                "image_file": item.get("image_file", ""),
                            }
                        )
            except Exception:
                pass
    return artworks


def _thumb_data_uri(image_file: str) -> str:
    """把本地 SemArt 图片转成 base64 data URI（小图，内联可靠，无需配置 allowed_paths）。"""
    if not image_file:
        return ""
    path = Path("SemArt") / "Images" / image_file
    if not path.exists():
        return ""
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _cards_html(artworks: list[dict], intent: str = "", with_thumbs: bool = False) -> str:
    if not artworks and not intent:
        return ""
    badge = ""
    if intent:
        badge = f'<span class="intent-badge">{_INTENT_LABELS.get(intent, intent)}</span>'
    if not artworks:
        return f'<div class="related-label">🖼 相关画作 {badge}</div>'

    html = f'<div class="related-label">🖼 相关画作 {badge}</div>'
    for aw in artworks[:5]:
        thumb = _thumb_data_uri(aw.get("image_file", "")) if with_thumbs else ""
        img_html = (
            f'<img src="{thumb}" class="aw-thumb"/>' if thumb else ""
        )
        html += f"""<div class="aw-card">
            {img_html}
            <div class="aw-title">{aw['title'][:20]}</div>
            <div>{aw['author'][:16]}</div>
            <div class="aw-date">{aw.get('date','')[:12]}</div>
        </div>"""
    return html


# ═══════════════════════════════════════════════════════════════════
# Agent 调用
# ═══════════════════════════════════════════════════════════════════
# 稳定的用户标识：让长期偏好记忆（S5）跨会话/刷新持久化。
# 单用户 demo 用固定值即可；多用户场景可换成登录态。
WEB_USER_ID = "web_user"


def chat_fn(message: str, history: list, sid: str):
    if not sid:
        sid = str(uuid.uuid4())
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "user_query": message,
            "user_id": WEB_USER_ID,
            "tool_results": [],
            "final_answer": "",
        },
        config={"configurable": {"thread_id": sid}},
    )
    reply = result.get("final_answer") or result["messages"][-1].content
    intent = result.get("intent", "")

    # 结构化分支把画作放在 state.artworks；general 分支从 ToolMessage 解析
    artworks = result.get("artworks") or []
    if not artworks:
        artworks = _parse_artworks_from_messages(result["messages"][1:])

    # 时间线/推荐分支带配图 → 展示缩略图
    with_thumbs = intent in ("timeline", "recommendation")
    return reply, sid, _cards_html(artworks, intent, with_thumbs)


def new_session():
    return [], str(uuid.uuid4()), ""


# ═══════════════════════════════════════════════════════════════════
# HTML 片段生成
# ═══════════════════════════════════════════════════════════════════
def _nav_html() -> str:
    html = ""
    for icon, label, active in NAV_ITEMS:
        cls = "nav-btn nav-btn-active" if active else "nav-btn"
        html += f'<button class="{cls}">{icon}&nbsp;&nbsp;{label}</button>\n'
    return html


def _themes_html() -> str:
    chips = "".join(f'<span class="theme-chip">{t}</span>' for t in THEMES)
    return f'<div class="theme-wrap">{chips}</div>'


def _recommends_html() -> str:
    html = ""
    for r in RECOMMENDS:
        html += f"""<div class="rec-card">
            <div class="rec-title">《{r['title']}》</div>
            <div class="rec-sub">{r['author']} · {r['period']}</div>
        </div>"""
    return html


# ═══════════════════════════════════════════════════════════════════
# 界面
# ═══════════════════════════════════════════════════════════════════
with gr.Blocks(title="🎨 ArtAgent") as demo:

    # ── 顶部标题栏 ────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
        <span id="header-logo">🎨</span>
        <div>
            <div id="header-title">ArtAgent</div>
            <div id="header-sub">
                基于 SemArt 数据集的西方艺术智能助手
                &nbsp;·&nbsp; 画作检索 &nbsp;·&nbsp; 风格分析
                &nbsp;·&nbsp; 图像解读 &nbsp;·&nbsp; 多轮对话
            </div>
        </div>
    </div>
    """)

    # ── 三栏主体 ─────────────────────────────────────────────────
    with gr.Row(elem_id="main-row"):

        # 左侧
        with gr.Column(elem_id="left-panel", min_width=200, scale=0):
            gr.HTML(f"""
            <span class="nav-label">导航</span>
            {_nav_html()}
            <hr class="nav-divider">
            <span class="nav-label">探索主题</span>
            {_themes_html()}
            """)

        # 中间
        with gr.Column(elem_id="center-col", scale=1):
            session_id = gr.State(value=str(uuid.uuid4()))

            chatbot = gr.Chatbot(
                value=[],
                label=None,
                show_label=False,
                height=None,
                render_markdown=True,
                layout="bubble",
                elem_id="chatbot",
                buttons=[],
            )

            related = gr.HTML(value="", elem_id="related-band")

            with gr.Row(elem_id="input-bar"):
                msg_box = gr.Textbox(
                    placeholder="问我任何关于西方艺术的问题，如：对比一下梵高和莫奈的风格……",
                    lines=2,
                    scale=1,
                    show_label=False,
                    autofocus=True,
                    elem_id="msg-box",
                )
                with gr.Column(min_width=80, scale=0):
                    btn_send = gr.Button("发送", variant="primary", elem_id="btn-send")
                    btn_new = gr.Button("🔄 新建", elem_id="btn-new")

        # 右侧
        with gr.Column(elem_id="right-panel", min_width=220, scale=0):
            gr.HTML(f"""
            <div class="panel-label">⭐ 今日推荐</div>
            {_recommends_html()}
            <hr class="panel-divider">
            <div class="panel-label">💡 试试这些问题</div>
            """)
            for q in EXAMPLES:
                gr.Button(q, elem_classes=["eq-btn"]).click(
                    fn=lambda _, question=q: question,
                    inputs=[msg_box],
                    outputs=[msg_box],
                )

    # ── 事件 ────────────────────────────────────────────────────
    def respond(message, history, sid):
        if not message.strip():
            return history, sid, ""
        reply, sid, cards = chat_fn(message, history, sid)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, sid, cards

    btn_send.click(
        fn=respond,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, related],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, related],
    ).then(fn=lambda: "", outputs=msg_box)

    btn_new.click(
        fn=new_session,
        outputs=[chatbot, session_id, related],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
    )
