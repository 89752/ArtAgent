"""
ArtAgent Web —— FastAPI 后端。

  · GET  /                       前端单页
  · GET  /static/*               静态资源（素材 + 前端 css/js）
  · POST /api/chat               SSE 流式：逐节点推送思考链，收尾给最终答案
  · GET  /api/sessions           历史会话列表
  · GET  /api/sessions/{sid}     单会话完整消息
  · DELETE /api/sessions/{sid}   删除会话
  · GET  /api/bootstrap          启动数据（场景卡 + 偏好数）
  · DELETE /api/preferences      清空长期偏好
  · POST /api/documents/upload   上传 PDF，后台解析入库
  · GET  /api/documents          文档库列表（解析状态/路由分布/chunk 数）
  · GET  /api/documents/{doc_id} 单文档状态（前端轮询进度）

样式与逻辑 100% 自控。
"""

import os
import json
import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

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

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from web import service
from src.data import documents_store
from src.memory import feedback as feedback_store
from src.tasks import store as tasks_store
from src.observability import runs as runs_store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
_CORE_IMAGES = BASE_DIR / "data" / "core" / "images"
_SEMART_IMAGES = BASE_DIR / "SemArt" / "Images"  # 回退源（镜像未就绪时）
logger = logging.getLogger("api")


# ── 请求体模型（统一参数校验） ──
class ChatIn(BaseModel):
    message: str = Field(default="", max_length=8000)
    session_id: str = Field(default="", max_length=128)
    regenerate: bool = False


class AttachmentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)


class SchemaIn(BaseModel):
    entity_col: str = Field(min_length=1, max_length=200)
    group_axis_col: str | None = None
    description_col: str | None = None
    image_col: str | None = None
    display_name: str | None = Field(default=None, max_length=60)


class DatasetIn(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=128)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=60)


class FeedbackIn(BaseModel):
    session_id: str = Field(max_length=128)
    rating: Literal[1, -1]
    reason: str = Field(default="", max_length=40)
    comment: str = Field(default="", max_length=500)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动时初始化各存储并恢复：documents/任务表（processing→interrupted）/表格数据源。"""
    documents_store.init_db()
    tasks_store.mark_interrupted_on_startup()  # 进程崩溃恢复：中断任务可重试
    service.restore_tables()
    yield


app = FastAPI(title="西方艺术智能助手", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 请求治理中间件：request_id 贯穿 + 令牌桶限流 ──
_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = {}


def _rate_limited(client_ip: str) -> bool:
    """令牌桶近似：时间窗口内每 IP 最多 burst 次（env 可调；RATE_LIMIT_RPM=0 关闭）。"""
    try:
        rpm = float(os.getenv("RATE_LIMIT_RPM", "60"))
        burst = float(os.getenv("RATE_LIMIT_BURST", "20"))
    except ValueError:
        rpm, burst = 60.0, 20.0
    if rpm <= 0 or burst <= 0:
        return False
    now = time.time()
    window = max(1.0, 60.0 * burst / rpm)  # 桶容量折算窗口
    with _rate_lock:
        stamps = [t for t in _rate_buckets.get(client_ip, []) if now - t < window]
        if len(stamps) >= burst:
            _rate_buckets[client_ip] = stamps
            return True
        stamps.append(now)
        _rate_buckets[client_ip] = stamps
        return False


@app.middleware("http")
async def _platform_guard(request: Request, call_next):
    """给每个请求生成/透传 X-Request-Id，并对 /api/* 做基础限流。"""
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/images"):
        client_ip = request.client.host if request.client else "unknown"
        if _rate_limited(client_ip):
            resp = JSONResponse(
                {"ok": False, "error": "请求过于频繁，请稍后再试"},
                status_code=429,
                headers={"Retry-After": "30", "X-Request-Id": request_id},
            )
            return resp
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def _validation_handler(_request: Request, exc: RequestValidationError):
    errs = exc.errors()
    msg = errs[0].get("msg", "参数错误") if errs else "参数错误"
    return JSONResponse({"ok": False, "error": f"参数错误：{msg}"}, status_code=422)


@app.exception_handler(Exception)
async def _unhandled_handler(_request: Request, exc: Exception):
    logger.exception("unhandled error: %s", exc)
    return JSONResponse({"ok": False, "error": "服务器内部错误"}, status_code=500)


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/images/{file_name}")
def artwork_image(file_name: str):
    """本地画作配图静态服务：优先 data/core/images，回退 SemArt/Images。"""
    name = Path(file_name).name
    for base in (_CORE_IMAGES, _SEMART_IMAGES):
        base = base.resolve()
        path = (base / name).resolve()
        if path.parent == base and path.is_file():
            return FileResponse(
                str(path),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )
    return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)


@app.get("/api/bootstrap")
def bootstrap():
    """首屏数据：场景卡（含缩略图 data URI）+ 已记忆偏好数。"""
    cards = [
        {"query": c["query"], "text": c["text"],
         "thumb": service._thumb_url(c["image"])}
        for c in service.SCENE_CARDS
    ]
    return JSONResponse({"cards": cards, "memory": service.memory_count()})


@app.get("/api/sessions")
def get_sessions(offset: int = 0, limit: int = 50):
    """会话列表：分页返回 {items, total, has_more}。"""
    offset = max(0, offset)
    limit = min(max(1, limit), 100)
    items, total = service.sessions(offset=offset, limit=limit)
    return JSONResponse({
        "items": items,
        "total": total,
        "offset": offset,
        "has_more": offset + len(items) < total,
    })


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, payload: RenameIn):
    """重命名会话。"""
    ok = service.rename_conversation(sid, payload.title)
    if not ok:
        return JSONResponse({"ok": False, "error": "会话不存在"}, status_code=404)
    return JSONResponse({"ok": True, "title": payload.title})


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return JSONResponse({"messages": service.conversation(sid)})


@app.delete("/api/sessions/{sid}")
def del_session(sid: str):
    service.remove_conversation(sid)
    return JSONResponse({"ok": True})


@app.post("/api/sessions/{sid}/attachment")
async def attach_session_document(sid: str, payload: AttachmentIn):
    """把已上传文档记录进会话历史（前端刷新/切换会话后仍可见）。"""
    doc = service.document_status(payload.doc_id)
    if not doc:
        return JSONResponse({"ok": False, "error": "文档不存在"}, status_code=404)
    return JSONResponse(service.record_attachment(
        sid, payload.doc_id, doc.get("doc_name") or "", doc.get("kind") or ""
    ))


@app.delete("/api/preferences")
def del_preferences():
    service.reset_preferences()
    return JSONResponse({"ok": True, "memory": service.memory_count()})


@app.get("/api/preferences")
def get_preferences():
    """记忆面板：全部偏好分项（kind/value/weight/updated_at）。"""
    return JSONResponse({"items": service.preferences_items()})


@app.delete("/api/preferences/{kind}/{value}")
def del_preference_item(kind: str, value: str):
    """记忆面板：单项删除偏好（kind ∈ artist/style，value 需 URL 编码）。"""
    if kind not in ("artist", "style"):
        return JSONResponse(
            {"ok": False, "error": "kind 只支持 artist/style"}, status_code=400
        )
    ok = service.delete_preference_item(kind, value)
    if not ok:
        return JSONResponse({"ok": False, "error": "偏好项不存在"}, status_code=404)
    return JSONResponse({"ok": True, "memory": service.memory_count()})


@app.get("/api/memory")
def get_memory_items():
    """记忆面板 v2：全部记忆条目（含自动抽取/来源/时间，按 id 删除）。"""
    return JSONResponse({"items": service.memory_items_list()})


@app.delete("/api/memory")
def clear_memory_items():
    """记忆面板 v2：清空全部记忆条目（含旧偏好表兼容）。"""
    service.clear_all_memories()
    return JSONResponse({"ok": True, "memory": service.memory_count()})


@app.delete("/api/memory/{item_id}")
def del_memory_item(item_id: str):
    """记忆面板 v2：按条目 id 单项删除。"""
    ok = service.delete_memory_item(item_id)
    if not ok:
        return JSONResponse({"ok": False, "error": "记忆条目不存在"}, status_code=404)
    return JSONResponse({"ok": True, "memory": service.memory_count()})


@app.get("/api/feedback")
def get_feedback(limit: int = 100, offset: int = 0):
    """反馈列表（导出/人工审核用）。"""
    items, total = feedback_store.list_feedback(limit=limit, offset=offset)
    return JSONResponse({"items": items, "total": total})


@app.post("/api/feedback")
def add_feedback(payload: FeedbackIn):
    """用户反馈闭环：{session_id, rating(1/-1), reason?, comment?}。"""
    fid = feedback_store.add_feedback(
        payload.session_id, payload.rating, payload.reason, payload.comment
    )
    return JSONResponse({"ok": True, "id": fid})


@app.get("/api/metrics")
def get_metrics(limit: int = 500):
    """可观测汇总：延迟/成本/工具分布/反思与兜底率。"""
    return JSONResponse(runs_store.metrics(limit=limit))


@app.post("/api/chat")
async def chat(payload: ChatIn, request: Request):
    """SSE 流式对话。请求体：{message, session_id, regenerate}。

    实现：生产者线程驱动 sync 生成器（graph.stream 为阻塞调用），事件经
    asyncio.Queue 交给响应流。客户端断开/停止时 finally 置 stop_event，
    生产者在节点边界检测到后自行收尾（保存部分内容），线程安全退出。
    不依赖 request.is_disconnected()（在部分测试/代理场景会死锁）。
    """
    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def put(evt) -> None:
        """跨线程安全入队（asyncio.Queue 非线程安全，必须经 loop 调度）。"""
        loop.call_soon_threadsafe(queue.put_nowait, evt)

    def producer() -> None:
        it = service.stream_answer(
            payload.message,
            payload.session_id,
            regenerate=payload.regenerate,
            stop_event=stop_event,
            request_id=getattr(request.state, "request_id", None),
        )
        try:
            for evt in it:
                if stop_event.is_set():
                    break
                put(evt)
        except Exception as e:  # noqa: BLE001
            logger.exception("chat producer failed: %s", e)
            put({"type": "error", "message": "服务器内部错误"})
        finally:
            put(None)  # 哨兵：流结束

    threading.Thread(target=producer, daemon=True, name="chat-producer").start()

    async def event_stream():
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            stop_event.set()   # 客户端断开/正常收尾：通知生产者尽早停止

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ── 文档上传与入库（PDF / 表格） ──
_UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_MB", "50")) * 1024 * 1024


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    oversize: str = Form(""),
):
    """上传 PDF/表格：保存 → BackgroundTasks 后台处理 → 前端轮询进度。

    文件类型路由（零模型调用）：.pdf → PDF 解析入库；
    .csv/.xlsx/.xls → 表格通道（加载 + schema 推断 → 待确认）。
    采用后台任务而非同步阻塞（MinerU/大图文档解析耗时以分钟计，
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
        if kind == "pdf" and oversize == "split":
            from src.ingestion.pdf_splitter import split_pdf

            parts = split_pdf(data, _UPLOAD_MAX_BYTES, filename)
            docs = []
            for part_name, part_bytes in parts:
                saved = service.save_upload(part_name, part_bytes)
                tid = tasks_store.create_task(
                    type="ingest_pdf",
                    task_id=saved["doc_id"],
                    payload={
                        "doc_id": saved["doc_id"],
                        "doc_name": saved["doc_name"],
                        "file_path": saved["file_path"],
                        "kb_id": saved["kb_id"],
                        "kind": "pdf",
                    },
                )
                background.add_task(
                    service.ingest_document,
                    saved["doc_id"], saved["doc_name"], saved["file_path"],
                    saved["kb_id"], task_id=tid,
                )
                docs.append({"doc_id": saved["doc_id"], "doc_name": saved["doc_name"]})
            return JSONResponse({
                "ok": True, "split": True,
                "count": len(docs), "documents": docs,
            })
        if kind != "pdf" or oversize != "pdfplumber":
            return JSONResponse(
                {
                    "ok": False,
                    "error": "文件超过 50MB 限制",
                    "code": "oversized",
                    "choices": ["split", "pdfplumber"],
                    "max_bytes": _UPLOAD_MAX_BYTES,
                },
                status_code=400,
            )

    force_pdfplumber = oversize == "pdfplumber"
    saved = service.save_upload(filename, data)
    # 任务化：task_id 复用 doc_id，后台解析全程可查可重试（响应形状不变）
    task_id = tasks_store.create_task(
        type=f"ingest_{kind}",
        task_id=saved["doc_id"],
        payload={
            "doc_id": saved["doc_id"],
            "doc_name": saved["doc_name"],
            "file_path": saved["file_path"],
            "kb_id": saved["kb_id"],
            "kind": kind,
            "force_pdfplumber": force_pdfplumber,
        },
    )
    if kind == "table":
        background.add_task(
            service.ingest_table_doc,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
            task_id=task_id,
        )
    else:
        background.add_task(
            service.ingest_document,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
            task_id=task_id,
            force_pdfplumber=force_pdfplumber,
        )
    return JSONResponse(
        {"ok": True, "doc_id": saved["doc_id"], "doc_name": saved["doc_name"],
         "kind": kind}
    )


@app.get("/api/tasks")
def get_tasks(status: str = ""):
    """任务列表：可按状态过滤 pending/processing/done/failed/interrupted。"""
    return JSONResponse({
        "items": tasks_store.list_tasks(status=status.strip() or None)
    })


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = tasks_store.get_task(task_id)
    if not task:
        return JSONResponse({"ok": False, "error": "任务不存在"}, status_code=404)
    return JSONResponse({"ok": True, "task": task})


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str, background: BackgroundTasks):
    """重试 failed/interrupted 的解析任务（进程崩溃/解析失败后的恢复入口）。"""
    task = tasks_store.get_task(task_id)
    if not task:
        return JSONResponse({"ok": False, "error": "任务不存在"}, status_code=404)
    if not tasks_store.reset_task(task_id):
        return JSONResponse(
            {"ok": False, "error": "只有 failed/interrupted 任务可重试"},
            status_code=400,
        )
    payload = task.get("payload") or {}
    doc_id = payload.get("doc_id") or task_id
    doc_name = payload.get("doc_name") or "文档"
    file_path = payload.get("file_path") or ""
    kb_id = payload.get("kb_id") or "default"
    force_pdfplumber = bool(payload.get("force_pdfplumber"))
    # 同步把文档状态重置为解析中，避免界面一直停留在失败
    if documents_store.get_document(doc_id):
        documents_store.update_document(doc_id, status="processing", error="")
    if payload.get("kind") == "table":
        background.add_task(
            service.ingest_table_doc,
            doc_id, doc_name, file_path, kb_id, task_id=task_id,
        )
    else:
        background.add_task(
            service.ingest_document,
            doc_id, doc_name, file_path, kb_id, task_id=task_id,
            force_pdfplumber=force_pdfplumber,
        )
    return JSONResponse({"ok": True, "task": tasks_store.get_task(task_id)})


@app.post("/api/documents/{doc_id}/schema")
async def confirm_schema(doc_id: str, payload: SchemaIn):
    """确认/纠正表格 schema：用户确认后注册数据源生效。

    请求体：{entity_col, group_axis_col?, description_col?, image_col?, display_name?}
    （空串/null 表示该角色无列；entity_col 必填）
    """
    try:
        result = service.confirm_table(doc_id, payload.model_dump())
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "doc": result})


@app.get("/api/datasets")
def get_datasets():
    """数据源清单（核心库 + 已确认表格）+ 当前生效项（前端切换器用）。"""
    return JSONResponse(service.datasets())


@app.post("/api/dataset/active")
async def switch_dataset(payload: DatasetIn):
    """切换当前生效数据源。请求体：{dataset_id}。"""
    try:
        return JSONResponse(service.set_active_dataset(payload.dataset_id))
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
    """删除文档并级联清理向量/文件/状态。"""
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
