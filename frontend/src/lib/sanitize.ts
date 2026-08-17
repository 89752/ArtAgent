/* 零依赖 HTML 白名单消毒器（与 static/lib/sanitize.js 等价，TS 化） */

const ALLOWED: Record<string, true> = {
  p: true, br: true, hr: true, strong: true, em: true, b: true, i: true,
  u: true, s: true, del: true, code: true, pre: true, blockquote: true,
  ul: true, ol: true, li: true, dl: true, dt: true, dd: true,
  h1: true, h2: true, h3: true, h4: true, h5: true, h6: true,
  table: true, thead: true, tbody: true, tfoot: true, tr: true, th: true, td: true,
  a: true, img: true, span: true, div: true, sup: true, sub: true,
  details: true, summary: true,
};

const URI_ATTRS: Record<string, string[]> = { a: ["href"], img: ["src"] };

const PLAIN_ATTRS: Record<string, string[]> = {
  a: ["target", "rel", "title"],
  img: ["alt", "title", "loading"],
  td: ["colspan", "rowspan"],
  th: ["colspan", "rowspan"],
  code: ["class"],
  span: ["class"],
};

function isSafeUrl(u: string): boolean {
  if (!u) return false;
  u = String(u).trim();
  if (/^(javascript|vbscript|data:text\/html):/i.test(u)) return false;
  if (/^data:image\/(png|jpe?g|gif|webp|svg\+xml);base64,/i.test(u)) return true;
  if (/^(https?:|mailto:|tel:)/i.test(u)) return true;
  if (u.charAt(0) === "/") return true;
  return false;
}

function clean(node: Node, doc: Document): DocumentFragment {
  const frag = doc.createDocumentFragment();
  const children = Array.prototype.slice.call(node.childNodes || []) as ChildNode[];
  for (const n of children) {
    if (n.nodeType === Node.TEXT_NODE) {
      frag.appendChild(doc.createTextNode(n.nodeValue ?? ""));
      continue;
    }
    if (n.nodeType !== Node.ELEMENT_NODE) continue;
    const el = n as Element;
    const tag = el.tagName.toLowerCase();
    if (!ALLOWED[tag]) continue;
    const clone = doc.createElement(tag);
    const attrs = el.attributes ? Array.prototype.slice.call(el.attributes) : [];
    for (const attr of attrs as Attr[]) {
      const name = attr.name.toLowerCase();
      if (/^on/i.test(name) || name === "style" || name === "srcdoc") continue;
      const val = el.getAttribute(attr.name);
      if (val == null) continue;
      if (URI_ATTRS[tag] && URI_ATTRS[tag].indexOf(name) >= 0) {
        if (!isSafeUrl(val)) continue;
      } else if (tag === "a" && name === "target") {
        clone.setAttribute("target", "_blank");
        clone.setAttribute("rel", "noopener noreferrer");
        continue;
      } else if (!PLAIN_ATTRS[tag] || PLAIN_ATTRS[tag].indexOf(name) < 0) {
        continue;
      }
      clone.setAttribute(name, val);
    }
    if (tag === "a") clone.setAttribute("rel", "noopener noreferrer");
    clone.appendChild(clean(n, doc));
    frag.appendChild(clone);
  }
  return frag;
}

export function sanitizeHtml(html: string): string {
  if (typeof DOMParser === "undefined" || typeof document === "undefined") {
    return "";
  }
  const doc = new DOMParser().parseFromString(String(html || ""), "text/html");
  const body = doc.body || doc;
  const cleanFrag = clean(body, doc);
  const out = doc.createElement("div");
  out.appendChild(cleanFrag);
  return out.innerHTML;
}
