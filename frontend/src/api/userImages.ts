import type { UserImageUploadResult } from "./types";

/** 用户图片上传（XHR 进度，与文档上传同款封装）。 */
export function uploadUserImage(
  file: File,
  sessionId: string,
  onProgress: (pct: number) => void,
): Promise<UserImageUploadResult> {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/user-images/upload");
    let lastRender = 0;
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      const now = Date.now();
      if (now - lastRender > 120) {
        lastRender = now;
        onProgress(pct);
      }
    };
    xhr.onload = () => {
      let j: UserImageUploadResult = { ok: false, error: "上传失败" };
      try {
        j = JSON.parse(xhr.responseText || "{}") as UserImageUploadResult;
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
    fd.append("session_id", sessionId);
    xhr.send(fd);
  });
}

/** 把已上传图片写入会话历史（kind=image）。 */
export async function attachUserImage(
  sid: string,
  imageId: string,
): Promise<void> {
  await fetch(`/api/user-images/${encodeURIComponent(imageId)}/attach`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sid }),
  });
}
