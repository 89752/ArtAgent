"""Shared run budgets and cancellation, propagated through ContextVars."""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable


class RunStopped(RuntimeError):
    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


@dataclass
class RunContext:
    run_id: str
    user_id: str
    task_id: str = ""
    step_id: str = ""
    attempt_id: str = ""
    task_type: str = ""
    max_model_calls: int = 40
    max_tool_calls: int = 60
    max_tokens: int = 80000
    max_output_tokens: int = 4096
    deadline: float = field(default_factory=lambda: time.monotonic() + 600)
    allowed_tools: set[str] | None = None
    document_ids: set[str] | None = None
    cancelled: Callable[[], bool] = field(default=lambda: False, repr=False)
    usage: dict = field(default_factory=lambda: {"model_calls": 0, "tool_calls": 0, "tokens": 0})
    versions: dict = field(default_factory=dict)
    persist: Callable[[dict], None] | None = field(default=None, repr=False)
    stopped_status: str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def check(self):
        if self.stopped_status:
            raise RunStopped(self.stopped_status)
        if self.cancelled():
            self.stopped_status = "cancelled"
            raise RunStopped("cancelled")
        if time.monotonic() >= self.deadline:
            self.stopped_status = "budget_exhausted"
            raise RunStopped("budget_exhausted")

    def charge(self, kind: str, amount: int = 1):
        with self._lock:
            self.check()
            limits = {"model_calls": self.max_model_calls, "tool_calls": self.max_tool_calls, "tokens": self.max_tokens}
            if self.usage.get(kind, 0) + amount > limits[kind]:
                self.stopped_status = "budget_exhausted"
                raise RunStopped("budget_exhausted")
            self.usage[kind] = self.usage.get(kind, 0) + amount
            if self.persist:
                self.persist(dict(self.usage))


_current = contextvars.ContextVar("artagent_run", default=None)


def current_run() -> RunContext | None:
    return _current.get()


@contextmanager
def run_scope(context: RunContext):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def invoke_model(model, messages):
    """Charge all nested callers to one budget; never retry an exhausted run."""
    context = current_run()
    if context is None:
        return model.invoke(messages)
    context.charge("model_calls")
    def sized(value):
        if isinstance(value, dict):
            if value.get("type") in {"image_url", "input_image"}:
                return 4096  # Image cost estimate, never count base64 as text tokens.
            return sum(len(str(k)) + sized(v) for k, v in value.items())
        if isinstance(value, list):
            return sum(sized(v) for v in value)
        return len(str(value))

    request = messages if isinstance(messages, str) else [
        {"content": getattr(m, "content", str(m)), "tool_calls": getattr(m, "tool_calls", [])} for m in messages]
    schema = getattr(model, "kwargs", {})
    estimated = max(1, (sized(request) + sized(schema) + 1) // 2)
    # Refuse oversized complete requests instead of silently cutting a tool pair.
    if estimated > max(1000, int(os.getenv("RUN_MAX_INPUT_TOKENS", "24000"))):
        context.stopped_status = "budget_exhausted"
        raise RunStopped("budget_exhausted")
    with context._lock:
        remaining = context.max_tokens-context.usage.get("tokens", 0)-estimated
        if remaining <= 0:
            context.stopped_status = "budget_exhausted"
            raise RunStopped("budget_exhausted")
        output_limit = min(context.max_output_tokens, remaining)
        reserved = estimated + output_limit
        context.charge("tokens", reserved)
    bounded = model.bind(max_tokens=output_limit) if hasattr(model, "bind") else model
    from src.utils.governance import run_with_timeout
    result = run_with_timeout(lambda: bounded.invoke(messages), max(.01, context.deadline - time.monotonic()))
    context.check()
    usage = getattr(result, "usage_metadata", None) or {}
    total = int(usage.get("total_tokens") or estimated + max(1, len(str(result.content)) // 2))
    with context._lock:
        if total > reserved:
            context.charge("tokens", total-reserved)
        else:
            context.usage["tokens"] -= reserved-max(estimated, total)
            if context.persist:
                context.persist(dict(context.usage))
    return result
