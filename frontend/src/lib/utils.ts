export function genId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

export function relTime(iso?: string): string {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  const days = Math.floor((Date.now() - t.getTime()) / 86400000);
  return days <= 0 ? "今天" : `${days} 天前`;
}

export function escapeHtmlText(s: unknown): string {
  return String(s).replace(/[&<>"']/g, (c) => (
    {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c] ?? c
  ));
}

export function currentSidFromUrl(): string {
  return new URLSearchParams(window.location.search).get("s") || "";
}

export function updateUrl(sid?: string): void {
  try {
    const url = sid
      ? window.location.pathname + "?s=" + encodeURIComponent(sid)
      : window.location.pathname;
    window.history.replaceState(null, "", url);
  } catch {
    /* 忽略：文件协议或受限环境 */
  }
}
