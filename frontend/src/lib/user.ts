const TOKEN_KEY = "artagent.token";
const USER_KEY = "artagent.user";

export interface StoredUser {
  user_id: string;
  name?: string;
  username?: string;
}

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* 忽略 */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  } catch {
    /* 忽略 */
  }
}

export function getUser(): StoredUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    const u = JSON.parse(raw) as StoredUser;
    return u && u.user_id ? u : null;
  } catch {
    return null;
  }
}

export function setUser(user: StoredUser): void {
  try {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  } catch {
    /* 忽略 */
  }
}
