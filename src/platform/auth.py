"""API 鉴权依赖（静态 API Key）。"""

from __future__ import annotations

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


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    cookie: str | None = Cookie(default=None, alias="artagent_token"),
) -> str:
    """校验 API Key 并返回 user_id；失败抛 401。"""
    key = _extract_key(authorization, x_api_key, cookie)
    user = users.get_user_by_api_key(key) if key else None
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"ok": False, "error": "无效或缺失 API Key（Authorization: Bearer <key>）"},
        )
    return user["user_id"]


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
    """本地 UI 路径：优先显式 X-User-Id；否则 API Key；否则默认 web_user。"""
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()[:64]
    key = _extract_key(authorization, x_api_key, cookie)
    user = users.get_user_by_api_key(key) if key else None
    if user:
        return user["user_id"]
    return users.DEFAULT_USER_ID
