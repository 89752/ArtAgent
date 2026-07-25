"""
ArtAgent Web —— FastAPI 后端（替代 Gradio）。

  · GET  /                       前端单页
  · GET  /static/*               静态资源（素材 + 前端 css/js）
  · POST /api/chat               SSE 流式：逐节点推送思考链，收尾给最终答案
  · GET  /api/sessions           历史会话列表
  · GET  /api/sessions/{sid}     单会话完整消息
  · DELETE /api/sessions/{sid}   删除会话
  · GET  /api/bootstrap          启动数据（场景卡 + 偏好数）
  · DELETE /api/preferences      清空长期偏好

样式与逻辑 100% 自控，彻底摆脱 Gradio 的 DOM/CSS 束缚。
"""

import os
import json
from pathlib import Path

# 仅对本地地址绕过系统代理（沿用 app.py：避免启动自检走代理 502，又不断外部 API）。
_LOCAL_NOPROXY = "localhost,127.0.0.1,0.0.0.0,::1"
for _k in ("NO_PROXY", "no_proxy"):
    _parts = [p for p in os.environ.get(_k, "").split(",") if p.strip()]
    for _addr in _LOCAL_NOPROXY.split(","):
        if _addr not in _parts:
            _parts.append(_addr)
    os.environ[_k] = ",".join(_parts)

# BGE 向量模型随索引本地缓存，强制离线加载，避免首检索联网自检超时。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web import service

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="西方艺术智能助手", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/bootstrap")
def bootstrap():
    """首屏数据：场景卡（含缩略图 data URI）+ 已记忆偏好数。"""
    cards = [
        {"query": c["query"], "text": c["text"],
         "thumb": service._thumb_data_uri(c["image"])}
        for c in service.SCENE_CARDS
    ]
    return JSONResponse({"cards": cards, "memory": service.memory_count()})


@app.get("/api/sessions")
def get_sessions():
    return JSONResponse(service.sessions())


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return JSONResponse({"messages": service.conversation(sid)})


@app.delete("/api/sessions/{sid}")
def del_session(sid: str):
    service.remove_conversation(sid)
    return JSONResponse({"ok": True})


@app.delete("/api/preferences")
def del_preferences():
    service.reset_preferences()
    return JSONResponse({"ok": True, "memory": service.memory_count()})


@app.post("/api/chat")
async def chat(request: Request):
    """SSE 流式对话。请求体：{message, session_id}。"""
    body = await request.json()
    message = (body.get("message") or "").strip()
    sid = body.get("session_id") or ""

    def event_stream():
        for evt in service.stream_answer(message, sid):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
