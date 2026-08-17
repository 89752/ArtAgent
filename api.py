"""
ArtAgent Web —— FastAPI 后端。

  · GET  /                       前端单页
  · GET  /static/*               静态资源（素材 + 前端 css/js）
  · POST /api/chat               SSE 流式：逐节点推送思考链，收尾给最终答案
  · GET  /api/sessions           历史会话列表
  · GET  /api/sessions/{sid}     单会话完整消息
  · DELETE /api/sessions/{sid}   删除会话
  · GET  /api/bootstrap          启动数据（场景卡 + 偏好数）
  · GET/DELETE /api/memory        记忆面板（列表/清空，含旧偏好统一存储）
  · POST /api/documents/upload   上传 PDF，后台解析入库
  · GET  /api/documents          文档库列表（解析状态/路由分布/chunk 数）
  · GET  /api/documents/{doc_id} 单文档状态（前端轮询进度）

样式与逻辑 100% 自控。
"""

import os
import json
import asyncio
import logging
import shutil
import threading
import time
import uuid
from io import BytesIO
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

from fastapi import BackgroundTasks, Depends, FastAPI, Form, Header, Request, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from web import service
from src.data import documents_store
from src.memory import feedback as feedback_store
from src.tasks import store as tasks_store
from src.observability import runs as runs_store
from src.platform.auth import current_user, optional_user, require_admin
from src.platform import users as users_store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
logger = logging.getLogger("api")


# ── 请求体模型（统一参数校验） ──
class ChatIn(BaseModel):
    message: str = Field(default="", max_length=8000)
    session_id: str = Field(default="", max_length=128)
    regenerate: bool = False


class AttachmentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)


class UserImageAttachIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class AnalysisMessageIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_text: str = Field(default="", max_length=200)
    html: str = Field(default="", max_length=4000)
    title: str = Field(default="", max_length=200)


class SchemaIn(BaseModel):
    entity_col: str = Field(min_length=1, max_length=200)
    group_axis_col: str | None = None
    description_col: str | None = None
    image_col: str | None = None
    display_name: str | None = Field(default=None, max_length=60)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=60)


class FeedbackIn(BaseModel):
    session_id: str = Field(max_length=128)
    rating: Literal[1, -1]
    reason: str = Field(default="", max_length=40)
    comment: str = Field(default="", max_length=500)


class LoginIn(BaseModel):
    username: str = Field(default="", max_length=40)
    password: str = Field(default="", max_length=128)


class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=128)
    name: str = Field(default="", max_length=60)


class ChangePasswordIn(BaseModel):
    old_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


class AdminCreateUserIn(BaseModel):
    name: str = Field(default="", max_length=60)
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=128)
    is_admin: bool = False


class AdminResetPasswordIn(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class MemoryImportIn(BaseModel):
    items: list[dict] = Field(default_factory=list)
    text: str = Field(default="", max_length=20000)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动时初始化各存储并恢复：documents/任务表（processing→interrupted）/表格数据源。"""
    users_store.init_db()
    documents_store.init_db()
    tasks_store.mark_interrupted_on_startup()  # 进程崩溃恢复：中断任务可重试
    service.restore_tables()
    from src.analysis import store as analysis_store

    analysis_store.init_db()
    try:
        from src.analysis.engine import USER_IMAGE_ROOT

        expired = analysis_store.cleanup_expired(
            int(os.getenv("USER_IMAGE_TTL_DAYS", "30"))
        )
        for image_id in expired:
            shutil.rmtree(USER_IMAGE_ROOT / image_id, ignore_errors=True)
    except Exception:  # noqa: BLE001
        logger.exception("user image TTL cleanup failed")
    yield
    try:
        from src.memory.extract import shutdown_flush

        flushed = shutdown_flush(30.0)
        logger.info("memory extract shutdown flush: %s", "ok" if flushed else "timeout")
    except Exception:  # noqa: BLE001
        logger.exception("memory extract shutdown flush failed")


app = FastAPI(title="西方艺术智能助手", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 请求治理中间件：request_id 贯穿 + 令牌桶限流 ──
_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = {}
_rate_last_cleanup = time.time()


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
        # 定期清理过期 IP 桶，避免长期运行内存微增
        global _rate_last_cleanup
        if now - _rate_last_cleanup > 300:
            for ip in [ip for ip, stamps in _rate_buckets.items() if not stamps]:
                del _rate_buckets[ip]
            _rate_last_cleanup = now
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


# ── 认证：登录 / 登出 / 当前用户 ─────────────────────────────
@app.post("/api/auth/login")
def login(payload: LoginIn):
    """账号密码登录；成功签发会话 token。"""
    user = users_store.verify_login(payload.username, payload.password)
    if user is None:
        return JSONResponse(
            {"ok": False, "error": "用户名或密码错误"}, status_code=401
        )
    token = users_store.issue_session_token(user["user_id"])
    return JSONResponse(
        {"ok": True, "token": token, "user": users_store.public_user(user)}
    )


@app.post("/api/auth/logout")
def logout(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """吊销当前会话 token。"""
    from src.platform.auth import _extract_key

    key = _extract_key(authorization, x_api_key)
    if key:
        users_store.revoke_api_key(key)
    return JSONResponse({"ok": True})


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return JSONResponse({"ok": True, "user": user})


@app.post("/api/auth/register")
def register(payload: RegisterIn):
    """自助注册：校验用户名/密码并建号，成功即签发会话 token（自动登录）。"""
    try:
        result = users_store.register_user(
            payload.username, payload.password, payload.name
        )
    except (KeyError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    token = users_store.issue_session_token(result["user"]["user_id"])
    return JSONResponse(
        {"ok": True, "token": token, "user": users_store.public_user(result["user"])}
    )


@app.post("/api/auth/change-password")
def change_password(
    payload: ChangePasswordIn,
    user: dict = Depends(current_user),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """本人修改密码：校验旧密码后更新，吊销其他会话（当前会话保持有效）。"""
    from src.platform.auth import _extract_key

    key = _extract_key(authorization, x_api_key)
    try:
        users_store.change_password(
            user["user_id"],
            payload.old_password,
            payload.new_password,
            keep_token=key,
        )
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True})


# ── 管理端：建号 / 列表 / 删除 / 重置密码（仅管理员） ─────────
@app.post("/api/admin/users")
def admin_create_user(
    payload: AdminCreateUserIn,
    _admin: dict = Depends(require_admin),
):
    try:
        result = users_store.create_user_with_password(
            payload.name, payload.username, payload.password, is_admin=payload.is_admin
        )
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse(
        {"ok": True, "user": users_store.public_user(result["user"])}
    )


@app.get("/api/admin/users")
def admin_list_users(_admin: dict = Depends(require_admin)):
    users = [users_store.public_user(u) for u in users_store.list_users()]
    return JSONResponse({"ok": True, "users": users})


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: str, admin: dict = Depends(require_admin)):
    if user_id == admin.get("user_id"):
        return JSONResponse(
            {"ok": False, "error": "不能删除当前登录的管理员"}, status_code=400
        )
    try:
        result = users_store.delete_user(user_id)
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    return JSONResponse({"ok": True, "result": result})


@app.post("/api/admin/users/{user_id}/reset-password")
def admin_reset_password(
    user_id: str,
    payload: AdminResetPasswordIn,
    _admin: dict = Depends(require_admin),
):
    if not users_store.reset_password(user_id, payload.password):
        return JSONResponse({"ok": False, "error": "用户不存在"}, status_code=404)
    return JSONResponse({"ok": True})


@app.get("/")
def index():
    # React 构建产物（frontend/ → static/dist）；旧版原生前端已迭代移除
    built = STATIC_DIR / "dist" / "index.html"
    if built.exists():
        return FileResponse(str(built))
    return JSONResponse(
        {"ok": False, "error": "前端未构建，请先运行 cd frontend && npm run build"},
        status_code=404,
    )


@app.get("/api/images/{file_name}")
def artwork_image(file_name: str):
    """本地画作配图静态服务：优先 data/core/images，回退 SemArt/Images。"""
    from src.utils.images import artwork_image_bases

    name = Path(file_name).name
    for base in artwork_image_bases():
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
def bootstrap(user_id: str = Depends(optional_user)):
    """首屏数据：场景卡 + 记忆条数 + 上传大小上限（前端预检用，唯一权威值）。"""
    cards = [
        {"query": c["query"], "text": c["text"],
         "thumb": service._thumb_url(c["image"])}
        for c in service.SCENE_CARDS
    ]
    return JSONResponse({
        "cards": cards,
        "memory": service.memory_count(user_id),
        "upload_max_bytes": _UPLOAD_MAX_BYTES,
    })


@app.get("/api/sessions")
def get_sessions(
    offset: int = 0,
    limit: int = 50,
    user_id: str = Depends(optional_user),
):
    """会话列表：分页返回 {items, total, has_more}。"""
    offset = max(0, offset)
    limit = min(max(1, limit), 100)
    items, total = service.sessions(offset=offset, limit=limit, user_id=user_id)
    return JSONResponse({
        "items": items,
        "total": total,
        "offset": offset,
        "has_more": offset + len(items) < total,
    })


@app.patch("/api/sessions/{sid}")
async def rename_session(
    sid: str,
    payload: RenameIn,
    user_id: str = Depends(optional_user),
):
    """重命名会话。"""
    ok = service.rename_conversation(sid, payload.title, user_id)
    if not ok:
        return JSONResponse({"ok": False, "error": "会话不存在"}, status_code=404)
    return JSONResponse({"ok": True, "title": payload.title})


@app.get("/api/sessions/{sid}")
def get_session(sid: str, user_id: str = Depends(optional_user)):
    return JSONResponse({"messages": service.conversation(sid, user_id)})


@app.delete("/api/sessions/{sid}")
def del_session(sid: str, user_id: str = Depends(optional_user)):
    service.remove_conversation(sid, user_id)
    return JSONResponse({"ok": True})


@app.post("/api/sessions/{sid}/attachment")
async def attach_session_document(
    sid: str,
    payload: AttachmentIn,
    user_id: str = Depends(optional_user),
):
    """把已上传文档记录进会话历史（前端刷新/切换会话后仍可见）。"""
    doc = service.document_status(payload.doc_id)
    if not doc:
        return JSONResponse({"ok": False, "error": "文档不存在"}, status_code=404)
    return JSONResponse(service.record_attachment(
        sid, payload.doc_id, doc.get("doc_name") or "", doc.get("kind") or "",
        user_id,
    ))


@app.get("/api/memory")
def get_memory_items(user_id: str = Depends(optional_user)):
    """记忆面板 v2：全部记忆条目（含自动抽取/来源/时间，按 id 删除）。"""
    return JSONResponse({"items": service.memory_items_list(user_id)})


@app.delete("/api/memory")
def clear_memory_items(user_id: str = Depends(optional_user)):
    """记忆面板 v2：清空全部记忆条目（含旧偏好表兼容）。"""
    service.clear_all_memories(user_id)
    return JSONResponse({"ok": True, "memory": service.memory_count(user_id)})


@app.post("/api/memory/import")
def import_memory_items(
    payload: MemoryImportIn,
    user_id: str = Depends(optional_user),
):
    """批量导入记忆：支持 items(JSON) 或 text（每行一条）。"""
    from src.memory.memory_items import import_memories

    items = list(payload.items or [])
    for line in (payload.text or "").splitlines():
        line = line.strip()
        if line:
            items.append({"content": line, "kind": "preference"})
    if not items:
        return JSONResponse(
            {"ok": False, "error": "没有可导入的内容"}, status_code=400
        )
    stats = import_memories(user_id, items)
    return JSONResponse(
        {"ok": True, "stats": stats, "memory": service.memory_count(user_id)}
    )


@app.post("/api/memory/import-file")
async def import_memory_file(
    file: UploadFile,
    user_id: str = Depends(optional_user),
):
    """记忆导入（文件版）：上传 .txt/.md/.json/.csv，解析后批量写入。"""
    from src.memory.memory_items import import_memories, parse_import_file

    filename = file.filename or ""
    raw = await file.read(2 * 1024 * 1024 + 1)
    try:
        items = parse_import_file(filename, raw)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not items:
        return JSONResponse(
            {"ok": False, "error": "文件里没有可导入的内容"}, status_code=400
        )
    stats = import_memories(user_id, items)
    return JSONResponse(
        {"ok": True, "stats": stats, "memory": service.memory_count(user_id)}
    )


@app.delete("/api/memory/{item_id}")
def del_memory_item(item_id: str, user_id: str = Depends(optional_user)):
    """记忆面板 v2：按条目 id 单项删除。"""
    ok = service.delete_memory_item(item_id, user_id)
    if not ok:
        return JSONResponse({"ok": False, "error": "记忆条目不存在"}, status_code=404)
    return JSONResponse({"ok": True, "memory": service.memory_count(user_id)})


@app.get("/api/feedback")
def get_feedback(
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(optional_user),
):
    """反馈列表（导出/人工审核用）。"""
    items, total = feedback_store.list_feedback(
        limit=limit, offset=offset, user_id=user_id
    )
    return JSONResponse({"items": items, "total": total})


@app.post("/api/feedback")
def add_feedback(payload: FeedbackIn, user_id: str = Depends(optional_user)):
    """用户反馈闭环：{session_id, rating(1/-1), reason?, comment?}。"""
    fid = feedback_store.add_feedback(
        payload.session_id, payload.rating, payload.reason, payload.comment, user_id
    )
    return JSONResponse({"ok": True, "id": fid})


@app.get("/api/metrics")
def get_metrics(limit: int = 500):
    """可观测汇总：延迟/成本/工具分布/反思与兜底率。"""
    return JSONResponse(runs_store.metrics(limit=limit))


@app.get("/api/metrics/memory")
def get_memory_metrics(limit: int = 50):
    """记忆抽取质量：提取数/放行数/各门控拒绝数/拒绝率。"""
    from src.memory.metrics import recent_extraction_metrics

    return JSONResponse({"items": recent_extraction_metrics(limit=limit)})


@app.post("/api/chat")
async def chat(
    payload: ChatIn,
    request: Request,
    user_id: str = Depends(optional_user),
):
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
            user_id=user_id,
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
    user_id: str = Depends(optional_user),
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
    # 分块读取：未选择拆分/直传模式时，超限立即中止，避免超大文件整份读入内存（OOM）
    allow_full_read = kind == "pdf" and oversize in ("split", "pdfplumber")
    max_mb = max(1, _UPLOAD_MAX_BYTES // (1024 * 1024))
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _UPLOAD_MAX_BYTES and not allow_full_read:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"文件超过 {max_mb}MB 限制",
                    "code": "oversized",
                    "choices": ["split", "pdfplumber"],
                    "max_bytes": _UPLOAD_MAX_BYTES,
                },
                status_code=400,
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    if len(data) > _UPLOAD_MAX_BYTES:
        if kind == "pdf" and oversize == "split":
            from src.ingestion.pdf_splitter import split_pdf

            parts = split_pdf(data, _UPLOAD_MAX_BYTES, filename)
            docs = []
            for part_name, part_bytes in parts:
                saved = service.save_upload(part_name, part_bytes, user_id=user_id)
                tid = tasks_store.create_task(
                    type="ingest_pdf",
                    task_id=saved["doc_id"],
                    payload={
                        "doc_id": saved["doc_id"],
                        "doc_name": saved["doc_name"],
                        "file_path": saved["file_path"],
                        "kb_id": saved["kb_id"],
                        "kind": "pdf",
                        "user_id": user_id,
                    },
                )
                background.add_task(
                    service.ingest_document,
                    saved["doc_id"], saved["doc_name"], saved["file_path"],
                    saved["kb_id"], task_id=tid, user_id=user_id,
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
                    "error": f"文件超过 {max_mb}MB 限制",
                    "code": "oversized",
                    "choices": ["split", "pdfplumber"],
                    "max_bytes": _UPLOAD_MAX_BYTES,
                },
                status_code=400,
            )

    force_pdfplumber = oversize == "pdfplumber"
    saved = service.save_upload(filename, data, user_id=user_id)
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
            "user_id": user_id,
        },
    )
    if kind == "table":
        background.add_task(
            service.ingest_table_doc,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
            task_id=task_id, user_id=user_id,
        )
    else:
        background.add_task(
            service.ingest_document,
            saved["doc_id"], saved["doc_name"], saved["file_path"], saved["kb_id"],
            task_id=task_id,
            force_pdfplumber=force_pdfplumber,
            user_id=user_id,
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
async def retry_task(
    task_id: str,
    background: BackgroundTasks,
    user_id: str = Depends(optional_user),
):
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
    task_user_id = payload.get("user_id") or user_id
    # 同步把文档状态重置为解析中，避免界面一直停留在失败
    if documents_store.get_document(doc_id, user_id):
        documents_store.update_document(
            doc_id, user_id, status="processing", error=""
        )
    if payload.get("kind") == "table":
        background.add_task(
            service.ingest_table_doc,
            doc_id, doc_name, file_path, kb_id, task_id=task_id, user_id=task_user_id,
        )
    else:
        background.add_task(
            service.ingest_document,
            doc_id, doc_name, file_path, kb_id, task_id=task_id,
            force_pdfplumber=force_pdfplumber,
            user_id=task_user_id,
        )
    return JSONResponse({"ok": True, "task": tasks_store.get_task(task_id)})


@app.post("/api/documents/{doc_id}/schema")
async def confirm_schema(
    doc_id: str,
    payload: SchemaIn,
    user_id: str = Depends(optional_user),
):
    """确认/纠正表格 schema：用户确认后注册数据源生效。

    请求体：{entity_col, group_axis_col?, description_col?, image_col?, display_name?}
    （空串/null 表示该角色无列；entity_col 必填）
    """
    if not documents_store.get_document(doc_id, user_id):
        return JSONResponse({"ok": False, "error": "文档不存在"}, status_code=404)
    try:
        result = service.confirm_table(doc_id, payload.model_dump(), user_id)
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "doc": result})


@app.get("/api/documents")
def get_documents(user_id: str = Depends(optional_user)):
    return JSONResponse(service.documents(user_id))


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str, user_id: str = Depends(optional_user)):
    return JSONResponse(service.document_status(doc_id, user_id))


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(optional_user)):
    """删除文档并级联清理向量/文件/状态。"""
    try:
        result = service.delete_document(doc_id, user_id)
    except KeyError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:  # noqa: BLE001
        logger.exception("delete_document failed: %s", e)
        return JSONResponse({"ok": False, "error": "服务器内部错误"}, status_code=500)
    return JSONResponse({"ok": True, "result": result})


# ── 用户图片通用层（user_images） ─────────────────────────────
@app.post("/api/user-images/upload")
async def upload_user_image(
    file: UploadFile,
    session_id: str = Form(""),
    user_id: str = Depends(optional_user),
):
    """上传用户图片（jpg/png/webp）：解码校验、EXIF 归一化、重编码去隐私元数据。"""
    from src.analysis.engine import USER_IMAGE_ROOT
    from src.analysis.store import add_image
    from PIL import Image, ImageOps

    filename = file.filename or ""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return JSONResponse(
            {"ok": False, "error": "仅支持 jpg / png / webp 图片"}, status_code=400
        )
    max_bytes = int(os.getenv("USER_IMAGE_MAX_MB", "20")) * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            return JSONResponse(
                {"ok": False, "error": "图片超过大小限制"}, status_code=400
            )
    data = b"".join(chunks)
    if not data:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    try:
        img = Image.open(BytesIO(data))
        img.load()
        fmt = (img.format or "").upper()
        if fmt not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("unsupported image format")
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        width, height = img.size
    except Exception:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": "无法识别的图片文件（支持 jpg/png/webp）"},
            status_code=400,
        )

    image_id = uuid.uuid4().hex[:12]
    target_dir = USER_IMAGE_ROOT / image_id
    target_dir.mkdir(parents=True, exist_ok=True)
    save_ext = "png" if fmt == "PNG" else ("webp" if fmt == "WEBP" else "jpeg")
    out_path = target_dir / f"original.{save_ext}"
    try:
        if fmt == "PNG":
            img.save(out_path, "PNG")
        elif fmt == "WEBP":
            img.save(out_path, "WEBP", quality=90)
        else:
            img.save(out_path, "JPEG", quality=92)
    except Exception as e:  # noqa: BLE001
        logger.exception("save user image failed: %s", e)
        return JSONResponse(
            {"ok": False, "error": "图片保存失败"}, status_code=500
        )
    mime = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[save_ext]
    add_image(
        image_id,
        (session_id or "").strip()[:128],
        filename,
        str(out_path),
        total,
        mime,
        width,
        height,
        user_id=user_id,
    )
    return JSONResponse(
        {
            "ok": True,
            "image_id": image_id,
            "thumb_url": f"/api/user-images/{image_id}/file",
            "width": width,
            "height": height,
        }
    )


@app.get("/api/user-images/{image_id}")
def get_user_image(image_id: str, user_id: str = Depends(optional_user)):
    from src.analysis.store import get_image

    rec = get_image(image_id, user_id)
    if not rec:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    rec["thumb_url"] = f"/api/user-images/{image_id}/file"
    return JSONResponse({"ok": True, "image": rec})


@app.get("/api/user-images/{image_id}/file")
def user_image_file(image_id: str, user_id: str = Depends(optional_user)):
    from src.analysis.store import get_image

    rec = get_image(image_id, user_id)
    if not rec or not rec.get("file_path") or not Path(rec["file_path"]).is_file():
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return FileResponse(
        str(rec["file_path"]),
        media_type=rec.get("mime_type") or "image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.delete("/api/user-images/{image_id}")
def delete_user_image(image_id: str, user_id: str = Depends(optional_user)):
    from src.analysis.engine import USER_IMAGE_ROOT
    from src.analysis.store import delete_image, get_image

    rec = get_image(image_id, user_id)
    if not rec:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    delete_image(image_id, user_id)
    shutil.rmtree(USER_IMAGE_ROOT / image_id, ignore_errors=True)
    return JSONResponse({"ok": True})


@app.post("/api/user-images/{image_id}/attach")
async def attach_user_image(
    image_id: str,
    payload: UserImageAttachIn,
    user_id: str = Depends(optional_user),
):
    """把已上传图片写入会话历史（kind=image），刷新/切会话后仍可见。"""
    from src.analysis.store import get_image

    rec = get_image(image_id, user_id)
    if not rec:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return JSONResponse(
        service.record_attachment(
            payload.session_id, image_id, rec.get("original_name") or "图片", "image",
            user_id,
        )
    )


# ── 绘画分析引擎（功能层） ────────────────────────────────────
def _analysis_sse(
    image_id: str,
    focus: str,
    framework_override: str,
    rerun: bool,
):
    """painting-analysis 的 SSE 事件流（线程 + Queue，与 /api/chat 同模式）。"""
    from web import analysis_service

    stop_event = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def put(evt) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, evt)

    def producer() -> None:
        it = analysis_service.stream_analysis(
            image_id,
            focus=focus,
            framework_override=framework_override.strip() or None,
            rerun=rerun,
            stop_event=stop_event,
        )
        try:
            for evt in it:
                if stop_event.is_set():
                    break
                put(evt)
        except Exception as e:  # noqa: BLE001
            logger.exception("painting analysis producer failed: %s", e)
            put({"type": "error", "message": "服务器内部错误"})
        finally:
            put(None)

    threading.Thread(target=producer, daemon=True, name="painting-analysis").start()

    async def event_stream():
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/painting-analysis/{image_id}")
async def painting_analysis(
    image_id: str,
    focus: str = "all",
    framework_override: str = "",
    user_id: str = Depends(optional_user),
):
    """SSE：stage 进度 + done/rejected/error；已有同参缓存则直接返回。"""
    from src.analysis.store import get_image

    if not get_image(image_id, user_id):
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return _analysis_sse(image_id, focus, framework_override, rerun=False)


@app.post("/api/painting-analysis/{image_id}/rerun")
async def painting_analysis_rerun(
    image_id: str,
    focus: str = "all",
    framework_override: str = "",
    user_id: str = Depends(optional_user),
):
    """以新的 focus / framework_override 强制重新分析。"""
    from src.analysis.store import get_image

    if not get_image(image_id, user_id):
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return _analysis_sse(image_id, focus, framework_override, rerun=True)


@app.get("/api/painting-analysis/{image_id}")
def get_painting_analysis(image_id: str, user_id: str = Depends(optional_user)):
    """取缓存报告（历史重载）。"""
    from src.analysis.store import get_analysis, get_image

    rec = get_image(image_id, user_id)
    if not rec:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    analysis = get_analysis(image_id)
    if not analysis or not analysis.get("result_path"):
        return JSONResponse({"ok": False, "error": "尚未分析"}, status_code=404)
    path = Path(analysis["result_path"])
    if not path.is_file():
        return JSONResponse(
            {"ok": False, "error": "分析结果文件丢失"}, status_code=404
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": f"分析结果读取失败：{e}"}, status_code=500
        )
    return JSONResponse({"ok": True, **payload})


@app.post("/api/painting-analysis/{image_id}/message")
async def persist_analysis_message(
    image_id: str,
    payload: AnalysisMessageIn,
    user_id: str = Depends(optional_user),
):
    """把分析结果作为 assistant 回合写入会话历史（前端完成后调用一次）。"""
    from src.analysis.store import get_image

    if not get_image(image_id, user_id):
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return JSONResponse(
        service.record_analysis_turn(
            payload.session_id,
            image_id,
            user_text=payload.user_text,
            html=payload.html,
            title=payload.title,
            user_id=user_id,
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
