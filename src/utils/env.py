"""环境变量小工具：布尔/整数开关的统一定义与解析。"""

from __future__ import annotations

import os

_TRUEY = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: str = "0") -> bool:
    """把 1/true/yes/on/y 解析为 True，其余一律 False（拼写错误不会误开功能）。"""
    raw = os.getenv(name, default).strip().lower()
    return raw in _TRUEY


def env_int(name: str, default: int) -> int:
    """读取整数环境变量；非法值回落 default。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
