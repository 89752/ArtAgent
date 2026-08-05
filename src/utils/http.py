"""HTTP 小工具：外部 API 的 GET + JSON 解码（固定 UA 与超时）。"""

from __future__ import annotations

import json
import urllib.request

DEFAULT_TIMEOUT = 10.0
DEFAULT_USER_AGENT = "ArtAgent/1.0 (local art assistant; contact: local)"


def get_json(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    """GET 并解码 JSON；异常由调用方自行捕获与降级。"""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 固定 HTTPS 域名
        return json.loads(resp.read().decode("utf-8"))


def download_bytes(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
    max_bytes: int = 15 * 1024 * 1024,
) -> bytes:
    """下载二进制内容（图片等）；超限或失败抛异常，由调用方降级。"""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 固定 HTTPS/HTTP 图片源
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"下载内容超过 {max_bytes} 字节上限")
    return data
