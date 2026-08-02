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
  · POST /api/documents/upload   上传 PDF，后台解析入库（Stage 3）
  · GET  /api/documents          文档库列表（解析状态/路由分布/chunk 数）
  · GET  /api/documents/{doc_id} 单文档状态（前端轮询进度）

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

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from web import service
from src.data import documents_store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动时初始化 SQLite documents 表并迁移旧 JSON；然后恢复已确认表格数据源。"""
    documents_store.init_db()
    service.restore_tables()
    yield


app = FastAPI(title="西方艺术智能助手", docs_url=None, redoc_url=None, lifespan=_lifespan)
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


# ── 文档上传与入库（Stage 3 PDF / Stage 5 表格） ──
_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50MB


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile, background: BackgroundTasks):
    """上传 PDF/表格：保存 → BackgroundTasks 后台处理 → 前端轮询进度。

    文件类型路由（零模型调用）：.pdf → Stage 3 解析入库；
    .csv/.xlsx/.xls → Stage 5 表格通道（加载 + schema 推断 → 待确认）。
    Phase 1 即采用后台任务而非同步阻塞（MinerU/大图文档解析耗时以分钟计，
    浏览器与 uvicorn 都会超时）。
    """
    from src.ingestion.table_loader import classify_upload

    filename = file.filename or ""
    kind = classify_upload(filename)
    if kind is None:
        return JSONResponse(
            {"ok": False, "error": "仅支持 PDF / CSV / XLSX / XLS 文件"},
            status_code=400,
        )
    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    if len(data) > _UPLOAD_MAX_BYTES:
        return JSONResponse(
            {"ok": False, "error": "文件超过 50MB 限制"}, status_code=400
        )

    saved = service.save_upload(filename, data)
    if kind == "table":
        background.add_task(
            service.ingest_table_doc,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
        )
    else:
        background.add_task(
            service.ingest_document,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
        )
    return JSONResponse(
        {"ok": True, "doc_id": saved["doc_id"], "doc_name": saved["doc_name"],
         "kind": kind}
    )


@app.post("/api/documents/{doc_id}/schema")
async def confirm_schema(doc_id: str, request: Request):
    """确认/纠正表格 schema（Stage 5）：用户确认后注册数据源生效。

    请求体：{entity_col, group_axis_col?, description_col?, image_col?, display_name?}
    （空串/null 表示该角色无列；entity_col 必填）
    """
    roles = await request.json()
    try:
        result = service.confirm_table(doc_id, roles or {})
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "doc": result})


@app.get("/api/datasets")
def get_datasets():
    """数据源清单（semart + 已确认表格）+ 当前生效项（Stage 5 切换器用）。"""
    return JSONResponse(service.datasets())


@app.post("/api/dataset/active")
async def switch_dataset(request: Request):
    """切换当前生效数据源（Stage 5）。请求体：{dataset_id}。"""
    body = await request.json()
    dataset_id = (body or {}).get("dataset_id") or ""
    try:
        return JSONResponse(service.set_active_dataset(dataset_id))
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)


@app.get("/api/documents")
def get_documents():
    return JSONResponse(service.documents())


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    return JSONResponse(service.document_status(doc_id))


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除文档并级联清理向量/文件/状态（Stage 6）。"""
    try:
        result = service.delete_document(doc_id)
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "result": result})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
