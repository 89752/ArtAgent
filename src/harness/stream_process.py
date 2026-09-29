"""Supervise SSE workers so cancellation stops their model/tool threads."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
from src.harness.process import terminate_worker


def stream(kind, payload, stop_event=None, *, timeout=None, command=None):
    owner = uuid.uuid4().hex
    env = dict(os.environ, ARTAGENT_STREAM_EXECUTION="inline", ARTAGENT_STREAM_OWNER=owner, PYTHONIOENCODING="utf-8")
    deadline = time.monotonic() + (timeout if timeout is not None else (180 if kind == "analysis" else 600))
    events = queue.Queue()
    process = subprocess.Popen(command or [sys.executable, "-m", "src.harness.stream_worker"],
        cwd=Path(__file__).resolve().parents[2], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    ended, terminal = False, False
    last_identity_check = 0
    def read():
        try:
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except ValueError:
                    continue
        finally:
            events.put(None)
    reader = threading.Thread(target=read, daemon=True, name="stream-process-reader")
    reader.start()
    try:
        process.stdin.write(json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False) + "\n")
        process.stdin.close()
        while True:
            if time.monotonic() - last_identity_check >= 1:
                from src.platform.users import get_user
                account = get_user(payload.get("user_id", "")) if kind == "chat" else None
                last_identity_check = time.monotonic()
                if account and account.get("deleting"):
                    yield {"type": "error", "code": "account_deleting", "message": "账号正在清理，执行已停止"}
                    break
            if stop_event is not None and stop_event.is_set():
                break
            if time.monotonic() >= deadline:
                yield {"type": "error", "code": "budget_exhausted", "message": "执行超时，已停止本次任务"}
                break
            try:
                event = events.get(timeout=.1)
            except queue.Empty:
                continue
            if event is None:
                ended = True
                if not terminal:
                    yield {"type": "error", "code": "worker_exit", "message": "执行进程中断，请重试"}
                break
            if not isinstance(event, dict):
                continue
            if event.get("type") in {"done", "error", "rejected"}:
                terminal = True
            yield event
    finally:
        terminate_worker(process)
        reader.join(timeout=1)
        process.stdout.close()
        if kind == "chat":
            from src.memory import conversations
            with conversations._lock:
                conn = conversations._get_conn()
                conn.execute("DELETE FROM conversation_leases WHERE user_id=? AND session_id=? AND owner=?",
                             (payload["user_id"], payload["sid"], owner))
                conn.commit()
        elif kind == "analysis" and not ended:
            from src.analysis.store import get_image, update_image_status
            image = get_image(payload["image_id"])
            if image and image.get("status") == "analyzing":
                update_image_status(payload["image_id"], "failed", "执行进程已停止")
