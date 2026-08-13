"""HTTP 小工具：外部 API 的 GET + JSON 解码（固定 UA 与超时）。"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 10.0
DEFAULT_USER_AGENT = "ArtAgent/1.0 (local art assistant; contact: local)"


def _allowed_image_roots() -> list[Path]:
    """本地图片允许根目录：core 图片 / SemArt / uploads / ARTAGENT_IMAGE_ROOTS 扩展。"""
    roots: list[Path] = []
    try:
        from src.utils.images import artwork_image_bases

        roots.extend(artwork_image_bases())
    except Exception:  # noqa: BLE001 —— 图片目录解析失败不阻断读取
        pass
    roots.append(Path(os.getenv("UPLOADS_DIR", "./data/uploads")).resolve())
    for part in re.split(r"[;,]", os.getenv("ARTAGENT_IMAGE_ROOTS", "")):
        if part.strip():
            roots.append(Path(part.strip()).resolve())
    return roots


def _local_image_path_allowed(path: Path) -> bool:
    """校验本地图片路径位于允许根目录内（防任意文件读取）。"""
    p = path.resolve()
    return any(p == r or r in p.parents for r in _allowed_image_roots())


def _is_private_host(host: str) -> bool:
    """判断主机是否指向内网/本机/保留地址（SSRF 防护）。解析失败时保守拒绝。"""
    host = (host or "").strip().rstrip(".").lower()
    if not host or host in ("localhost", "localhost.localdomain"):
        return True
    try:
        addrs = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return True
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr[4][0])
        except ValueError:
            continue
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            return True
    return False


def _assert_public_http_url(url: str) -> None:
    """仅允许公网 http/https；内网/本机/保留地址一律拒绝。"""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("仅支持 http/https URL")
    if _is_private_host(parts.hostname or ""):
        raise ValueError("不允许访问内网或本机地址")


def get_json(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict:
    """GET 并解码 JSON；异常由调用方自行捕获与降级。"""
    _assert_public_http_url(url)
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
    _assert_public_http_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 固定 HTTPS/HTTP 图片源
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"下载内容超过 {max_bytes} 字节上限")
    return data


def load_image_bytes(
    path_or_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = 15 * 1024 * 1024,
) -> tuple[bytes, str]:
    """读取本地图片或下载网络图片，返回 (bytes, 标准化后缀 jpeg/png/gif/webp)。"""
    if path_or_url.startswith(("http://", "https://")):
        data = download_bytes(path_or_url, timeout=timeout, max_bytes=max_bytes)
        ext = Path(urlparse(path_or_url).path).suffix.lstrip(".").lower()
    else:
        p = Path(path_or_url)
        if not _local_image_path_allowed(p):
            raise ValueError("本地图片路径不在允许目录内")
        data = p.read_bytes()
        ext = p.suffix.lstrip(".").lower()
    if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
        ext = "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return data, ext
