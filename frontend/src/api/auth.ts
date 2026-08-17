import { getJson, sendJson } from "./client";
import { clearToken, setToken, setUser, type StoredUser } from "../lib/user";

export interface LoginResult {
  ok: boolean;
  token?: string;
  user?: StoredUser;
  error?: string;
}

function applySession(j: LoginResult): LoginResult {
  if (j.ok && j.token && j.user) {
    setToken(j.token);
    setUser(j.user);
    document.cookie = `artagent_token=${encodeURIComponent(j.token)}; path=/; SameSite=Lax`;
  }
  return j;
}

export async function login(
  username: string,
  password: string,
): Promise<LoginResult> {
  try {
    const j = await sendJson<LoginResult>("/api/auth/login", "POST", {
      username,
      password,
    });
    return applySession(j);
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "登录失败",
    };
  }
}

export async function register(
  username: string,
  password: string,
  name = "",
): Promise<LoginResult> {
  try {
    const j = await sendJson<LoginResult>("/api/auth/register", "POST", {
      username,
      password,
      name,
    });
    return applySession(j);
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "注册失败",
    };
  }
}

export async function changePassword(
  oldPassword: string,
  newPassword: string,
): Promise<{ ok: boolean; error?: string }> {
  try {
    const j = await sendJson<{ ok: boolean; error?: string }>(
      "/api/auth/change-password",
      "POST",
      { old_password: oldPassword, new_password: newPassword },
    );
    return j;
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "修改密码失败",
    };
  }
}

export async function logout(): Promise<void> {
  try {
    await sendJson<{ ok: boolean }>("/api/auth/logout", "POST");
  } catch {
    /* 忽略 */
  }
  document.cookie = "artagent_token=; path=/; Max-Age=0";
  clearToken();
}

export async function fetchMe(): Promise<StoredUser | null> {
  try {
    const j = await getJson<{ ok: boolean; user?: StoredUser }>("/api/auth/me");
    return j.user || null;
  } catch {
    return null;
  }
}
