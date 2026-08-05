"""环境变量小工具：布尔/整数开关的统一定义与解析。"""

from __future__ import annotations

import os

_FALSEY = {"0", "false", "no", "off", "n", ""}


def env_flag(name: str, default: str = "0") -> bool:
    """把 1/true/yes/on/y 解析为 True，其余为 False；default 用字符串形式。"""
    raw = os.getenv(name, default).strip().lower()
    return raw not in _FALSEY


def env_int(name: str, default: int) -> int:
    """读取整数环境变量；非法值回落 default。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
