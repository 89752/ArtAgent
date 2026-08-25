"""API 鉴权依赖（静态 API Key）。"""

from __future__ import annotations

import os

from fastapi import Cookie, Header, HTTPException

from src.platform import users


def _extract_key(
    authorization: str | None,
    x_api_key: str | None,
    cookie: str | None = None,
) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if cookie:
        return cookie.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip()
    return None


def current_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    cookie: str | None = Cookie(default=None, alias="artagent_token"),
) -> dict:
    """要求有效登录态，返回公开用户信息；失败抛 401。"""
    key = _extract_key(authorization, x_api_key, cookie)
    user = users.get_user_by_api_key(key) if key else None
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"ok": False, "error": "未登录或登录已失效"},
        )
    return users.public_user(user) or {}


def require_authenticated_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    cookie: str | None = Cookie(default=None, alias="artagent_token"),
) -> str:
    """Return an identity derived from a server-validated credential."""
    return str(current_user(authorization, x_api_key, cookie)["user_id"])


def dev_user(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> str:
    """Development-only header identity, guarded by an explicit opt-in."""
    if os.getenv("ARTAGENT_ALLOW_HEADER_IDENTITY", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        raise HTTPException(status_code=401, detail={"ok": False, "error": "开发身份头未启用；请登录"})
    value = (x_user_id or "").strip()
    if not value:
        raise HTTPException(status_code=401, detail={"ok": False, "error": "开发身份头缺失"})
    return value[:64]


def require_admin(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    cookie: str | None = Cookie(default=None, alias="artagent_token"),
) -> dict:
    """要求管理员身份；普通用户返回 403。"""
    user = current_user(authorization, x_api_key, cookie)
    if not user.get("is_admin"):
        raise HTTPException(
            status_code=403,
            detail={"ok": False, "error": "需要管理员权限"},
        )
    return user


def optional_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    cookie: str | None = Cookie(default=None, alias="artagent_token"),
) -> str:
    """Deprecated compatibility dependency with safe-by-default behaviour."""
    if os.getenv("ARTAGENT_ALLOW_HEADER_IDENTITY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return dev_user(x_user_id)
    return require_authenticated_user(authorization, x_api_key, cookie)
