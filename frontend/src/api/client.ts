import { getToken } from "../lib/user";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

async function readError(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: string; detail?: string };
    if (j.error) return String(j.error);
    if (typeof j.detail === "string") return j.detail;
    if (j.detail) return JSON.stringify(j.detail);
  } catch {
    /* 非 JSON 响应走兜底 */
  }
  return `请求失败（HTTP ${res.status}）`;
}

export async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  return (await res.json()) as T;
}

export async function sendJson<T>(
  url: string,
  method: "POST" | "PATCH" | "DELETE",
  body?: unknown,
): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: {
      ...authHeaders(body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const j = (await res.json().catch(() => ({}))) as T & { ok?: boolean };
  if (!res.ok || j.ok === false) {
    throw new ApiError(
      String((j as { error?: string }).error || (await readError(res))),
      res.status,
    );
  }
  return j as T;
}
