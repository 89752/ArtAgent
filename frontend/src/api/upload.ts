import type { UploadResult } from "./types";
import { getToken } from "../lib/user";

/** XHR 上传封装：保留 upload.onprogress + 调用方节流（与原生版一致）。 */
export function uploadWithProgress(
  file: File,
  oversize: string,
  onProgress: (pct: number) => void,
): Promise<UploadResult> {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents/upload");
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    let lastProgRender = 0;
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      const now = Date.now();
      if (now - lastProgRender > 120) {
        lastProgRender = now;
        onProgress(pct);
      }
    };
    xhr.onload = () => {
      let j: UploadResult = { ok: false, error: "上传失败" };
      try {
        j = JSON.parse(xhr.responseText || "{}") as UploadResult;
      } catch {
        /* 忽略 */
      }
      if (xhr.status >= 400 || !j.ok) {
        resolve(j);
        return;
      }
      resolve(j);
    };
    xhr.onerror = () => {
      resolve({ ok: false, error: "上传失败：网络错误" });
    };
    const fd = new FormData();
    fd.append("file", file);
    fd.append("oversize", oversize);
    xhr.send(fd);
  });
}

export interface MemoryImportResult {
  ok: boolean;
  error?: string;
  stats?: { added: number; dup: number; invalid: number };
}

/** 记忆文件导入：直接 multipart 上传，由后端按扩展名解析 txt/json/csv。 */
export async function importMemoryFile(
  file: File,
): Promise<MemoryImportResult> {
  const fd = new FormData();
  fd.append("file", file);
  const token = getToken();
  let res: Response;
  try {
    res = await fetch("/api/memory/import-file", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: fd,
    });
  } catch {
    return { ok: false, error: "导入失败：网络错误" };
  }
  const j = (await res.json().catch(() => ({}))) as MemoryImportResult;
  if (!res.ok || j.ok === false) {
    return { ok: false, error: j.error || `导入失败（HTTP ${res.status}）` };
  }
  return j;
}
