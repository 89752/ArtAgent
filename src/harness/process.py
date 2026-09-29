"""Own a research worker process so a stopped job cannot retain Python threads."""
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from src.tasks import store


def terminate_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_process(task_id, user_id):
    from src.harness.verification import TaskSpec
    job = store.get_task(task_id)
    if not job or job["payload"].get("user_id") != user_id:
        return
    spec = TaskSpec(**(job["payload"].get("spec") or {"min_sources":0}))
    limit_seconds = spec.max_seconds if job["type"] == "agent_job" else max(10,int(os.getenv("TASK_PARSE_TIMEOUT_SEC","1800")))
    attempt = store.claim_task(task_id, user_id, "process:"+uuid.uuid4().hex)
    if not attempt:
        return
    started = time.monotonic()
    previous = float((job.get("usage") or {}).get("elapsed_seconds", 0))
    deadline = started + max(0, limit_seconds-previous)
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.harness.worker", task_id, user_id, attempt],
            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        last_heartbeat = started
        while process.poll() is None:
            fresh = store.get_task(task_id)
            if not fresh or fresh.get("attempt_id") != attempt:
                terminate_worker(process)
                return
            if fresh["status"] not in {"pending", "processing"}:
                # Child completed its database commit; let it flush and exit.
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminate_worker(process)
                return
            status = ("interrupted" if fresh["cancel_requested"] else
                      "paused" if fresh["pause_requested"] else
                      "budget_exhausted" if time.monotonic() >= deadline else "")
            if status:
                # Allow cooperative cleanup briefly, then stop all worker threads.
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    terminate_worker(process)
                usage = dict((store.get_task(task_id) or fresh).get("usage") or {})
                usage["elapsed_seconds"] = previous + time.monotonic()-started
                store.heartbeat(task_id, attempt, usage)
                store.finish_attempt(task_id, attempt, status, "执行进程已停止")
                return
            if time.monotonic()-last_heartbeat >= 10:
                if not store.heartbeat(task_id, attempt):
                    terminate_worker(process)
                    return
                last_heartbeat = time.monotonic()
            time.sleep(.2)
        fresh = store.get_task(task_id)
        if fresh and fresh["attempt_id"] == attempt and fresh["status"] == "processing":
            store.finish_attempt(task_id, attempt, "interrupted", f"执行进程异常退出（{process.returncode}），可从持久步骤重试")
    except Exception as exc:
        store.finish_attempt(task_id, attempt, "failed", f"执行进程启动或监护失败：{type(exc).__name__}")
        raise
    finally:
        if process is not None:
            terminate_worker(process)
