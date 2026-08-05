"""Agent 注册表（规划中）：扫描 agents/ 目录，一个进程注册多个 Agent。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.platform.agent_spec import AgentSpec, load_agent_spec
from src.utils.logging_config import get_logger

logger = get_logger("platform.registry")

AGENTS_DIR = Path(os.getenv(
    "ARTAGENT_AGENTS_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "agents"),
))

_REGISTRY: dict[str, AgentSpec] = {}
_LOADED = False


def _discover_manifests(agents_dir: Path) -> list[Path]:
    if not agents_dir.exists():
        return []
    return sorted(
        p for p in agents_dir.iterdir()
        if p.suffix.lower() in (".json", ".yaml", ".yml")
    )


def load_registry(agents_dir: Path | None = None) -> dict[str, AgentSpec]:
    """（重）载 agents/ 目录，返回 {agent_id: AgentSpec}。"""
    global _REGISTRY, _LOADED
    directory = agents_dir or AGENTS_DIR
    found: dict[str, AgentSpec] = {}
    for path in _discover_manifests(directory):
        try:
            spec = load_agent_spec(path)
            found[spec.id] = spec
            logger.info("[registry] 已注册 Agent %s (%s)", spec.id, path.name)
        except Exception as e:  # noqa: BLE001 —— 单份 manifest 损坏不阻断启动
            logger.warning("[registry] 加载 %s 失败：%s", path.name, e)
    if not found:
        logger.warning("[registry] agents/ 目录无可用 manifest（目录：%s）", directory)
    _REGISTRY = found
    _LOADED = True
    return dict(found)


def get_registry() -> dict[str, AgentSpec]:
    if not _LOADED:
        load_registry()
    return dict(_REGISTRY)


def get_agent_spec(agent_id: str) -> AgentSpec:
    """按 id 取 AgentSpec；未知 Agent 抛 KeyError。"""
    spec = get_registry().get(agent_id)
    if spec is None:
        raise KeyError(f"未注册的 Agent：{agent_id}")
    return spec


def list_agent_specs() -> list[AgentSpec]:
    return list(get_registry().values())


def reset_registry() -> None:
    global _REGISTRY, _LOADED
    _REGISTRY = {}
    _LOADED = False
