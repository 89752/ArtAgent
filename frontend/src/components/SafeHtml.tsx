import { memo, useLayoutEffect, useRef } from "react";
import { renderMarkdown } from "../lib/markdown";

/** 渲染服务端 HTML（独立子节点，避免向 React 管理的父节点写 innerHTML），
 *  并对 .md-answer 内的 Markdown 原文做 marked + 白名单渲染。 */
export const SafeHtml = memo(function SafeHtml({
  html,
}: {
  html: string;
}) {
  const innerRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    el.querySelectorAll<HTMLElement>(".md-answer").forEach((box) => {
      if (box.dataset.rendered) return;
      const raw = box.textContent || "";
      box.innerHTML = renderMarkdown(raw);
      box.dataset.rendered = "1";
    });
  }, [html]);

  return <div ref={innerRef} dangerouslySetInnerHTML={{ __html: html }} />;
});
