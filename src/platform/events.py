"""结构化事件流（业务/渲染解耦）。

Agent 内核只产出结构化事件；Web 渲染器（web/service.py）与 OpenAI 兼容
API（platform/openai_api.py）各自消费，互不耦合。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentEvent:
    type: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.payload}


def start_event(session_id: str, agent_id: str, user_id: str) -> AgentEvent:
    return AgentEvent("start", {
        "session_id": session_id, "agent_id": agent_id, "user_id": user_id,
    })


def node_event(node: str, label: str = "", detail: str = "") -> AgentEvent:
    return AgentEvent("node", {"node": node, "label": label, "detail": detail})


def intent_event(intent: str) -> AgentEvent:
    return AgentEvent("intent", {"intent": intent})


def tool_event(name: str, args: dict) -> AgentEvent:
    return AgentEvent("tool", {"name": name, "args": args})


def answer_event(content: str) -> AgentEvent:
    return AgentEvent("answer", {"content": content})


def error_event(message: str) -> AgentEvent:
    return AgentEvent("error", {"message": message})


def done_event(
    session_id: str,
    *,
    intent: str,
    steps: list,
    sources: list,
    cancelled: bool,
    error: str = "",
) -> AgentEvent:
    return AgentEvent("done", {
        "session_id": session_id,
        "intent": intent,
        "steps": steps,
        "sources": sources,
        "cancelled": cancelled,
        "error": error,
    })
