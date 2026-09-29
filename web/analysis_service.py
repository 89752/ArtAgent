"""绘画分析 SSE 编排（仿 stream_answer 的生成器 + 缓存短路）。"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterator

from src.analysis.engine import run_analysis
from src.analysis.store import get_analysis, get_image

ALLOWED_FOCUS = {"all", "perspective", "composition", "color", "brushwork", "style"}
ALLOWED_OVERRIDE = {"realistic", "abstract", "childlike", "decorative"}


def _cached_payload(image_id: str) -> dict | None:
    analysis = get_analysis(image_id)
    if not analysis or not analysis.get("result_path"):
        return None
    path = Path(analysis["result_path"])
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def stream_analysis(
    image_id: str,
    focus: str = "all",
    framework_override: str | None = None,
    rerun: bool = False,
    stop_event=None,
) -> Iterator[dict]:
    """产出 SSE 事件：stage/metrics → done/rejected/error。"""
    if os.getenv("ARTAGENT_STREAM_EXECUTION", "process") == "process":
        from src.harness.stream_process import stream
        yield from stream("analysis", {"image_id": image_id, "focus": focus,
                                      "framework_override": framework_override, "rerun": rerun}, stop_event)
        return
    rec = get_image(image_id)
    if not rec:
        yield {"type": "error", "message": "图片不存在或已删除"}
        return
    focus = focus if focus in ALLOWED_FOCUS else "all"
    if framework_override and framework_override not in ALLOWED_OVERRIDE:
        framework_override = None
    if not rerun:
        cached = _cached_payload(image_id)
        if cached:
            cached["cached"] = True
            yield {"type": "done", **cached}
            return
    from src.harness.context import RunContext, RunStopped, current_run, run_scope
    context = current_run() or RunContext(
        uuid.uuid4().hex, rec.get("user_id") or "web_user", task_type="analysis",
        max_model_calls=4, max_tokens=30000, deadline=time.monotonic()+180,
        cancelled=lambda: stop_event is not None and stop_event.is_set())
    try:
        with run_scope(context):
            context.check()
            for evt in run_analysis(image_id, focus=focus, framework_override=framework_override):
                context.check()
                yield evt
    except RunStopped as exc:
        from src.analysis.store import update_image_status
        update_image_status(image_id, "failed", exc.status)
        yield {"type": "error", "message": "分析已取消或超出执行预算", "code": exc.status}
