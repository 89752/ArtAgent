"""API 鉴权依赖（静态 API Key）。"""

from __future__ import annotations

from fastapi import Header, HTTPException

from src.platform import users


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip()
    return None


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """校验 API Key 并返回 user_id；失败抛 401。"""
    key = _extract_key(authorization, x_api_key)
    user = users.get_user_by_api_key(key) if key else None
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"ok": False, "error": "无效或缺失 API Key（Authorization: Bearer <key>）"},
        )
    return user["user_id"]


def optional_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> str:
    """本地 UI 路径：优先显式 X-User-Id；否则 API Key；否则默认 web_user。"""
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()[:64]
    key = _extract_key(authorization, x_api_key)
    user = users.get_user_by_api_key(key) if key else None
    if user:
        return user["user_id"]
    return users.DEFAULT_USER_ID
