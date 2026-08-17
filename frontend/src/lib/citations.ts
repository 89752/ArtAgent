import type { Source } from "../api/types";

export function focusSource(root: HTMLElement, n: number): void {
  const chip = root.querySelector<HTMLElement>(`.source-chip[data-i="${n}"]`);
  if (!chip) return;
  chip.scrollIntoView({ behavior: "smooth", block: "nearest" });
  chip.classList.remove("flash");
  void chip.offsetWidth;
  chip.classList.add("flash");
}

/** 遍历 .md-answer 文本节点，把 [n] 替换为可点击引用角标（原生 linkCitations 等价）。 */
export function attachCitations(root: HTMLElement, sources: Source[]): void {
  if (!root || !sources.length) return;
  root.querySelectorAll<HTMLElement>(".md-answer").forEach((answer) => {
    if (answer.dataset.cited) return;
    const walker = document.createTreeWalker(answer, NodeFilter.SHOW_TEXT);
    const textNodes: Text[] = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode as Text);
    const re = /\[(\d{1,2})\]/g;
    for (const node of textNodes) {
      const text = node.nodeValue || "";
      re.lastIndex = 0;
      if (!re.test(text)) continue;
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0;
      let m: RegExpExecArray | null;
      while ((m = re.exec(text))) {
        const n = parseInt(m[1], 10);
        if (n < 1 || n > sources.length) continue;
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        const sup = document.createElement("sup");
        sup.className = "cite";
        sup.textContent = m[0];
        sup.dataset.i = String(n);
        sup.setAttribute("data-tip", sources[n - 1].label || "");
        sup.addEventListener("click", () => focusSource(root, n));
        frag.appendChild(sup);
        last = m.index + m[0].length;
      }
      if (last) {
        frag.appendChild(document.createTextNode(text.slice(last)));
        node.parentNode?.replaceChild(frag, node);
      }
    }
    answer.dataset.cited = "1";
  });
}
