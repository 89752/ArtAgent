import { marked } from "marked";
import { sanitizeHtml } from "./sanitize";

marked.setOptions({ breaks: true, gfm: true });

/** 与原生版一致：marked 输出必须过白名单消毒器。 */
export function renderMarkdown(raw: string): string {
  const parsed = marked.parse(raw, { async: false }) as string;
  const sanitized = sanitizeHtml(parsed);
  // marked 遵循 CommonMark：`**` 紧贴词字符且后接标点时不会开启加粗
  // （如 `而**《画名》**`），会把 `**` 原样漏给用户。这里把残留在
  // 文本里的 `**...**` 兜底转成 <strong>，避免 Markdown 语法外泄。
  return sanitized.replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>");
}
