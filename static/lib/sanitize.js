/* sanitize.js —— 零依赖 HTML 白名单消毒器（成熟应用 G4/1.4）
 *
 * 背景：marked.parse 默认不过滤原始 HTML，LLM 输出/文档注入可构造
 * <img onerror> 等 XSS。本文件实现 DOMPurify 的最小等价物：
 *   - 只保留安全标签（白名单）；
 *   - 剥离全部 on* 事件属性与 style 属性；
 *   - a/img 的 URL 只允许 http(s)/mailto/tel/站内路径/安全 data:image；
 *   - 其余属性按标签白名单透传。
 * 用法：box.innerHTML = sanitizeHtml(marked.parse(raw))。
 */
(function (global) {
  "use strict";

  var ALLOWED = {
    p: 1, br: 1, hr: 1, strong: 1, em: 1, b: 1, i: 1, u: 1, s: 1, del: 1,
    code: 1, pre: 1, blockquote: 1, ul: 1, ol: 1, li: 1, dl: 1, dt: 1, dd: 1,
    h1: 1, h2: 1, h3: 1, h4: 1, h5: 1, h6: 1,
    table: 1, thead: 1, tbody: 1, tfoot: 1, tr: 1, th: 1, td: 1,
    a: 1, img: 1, span: 1, div: 1, sup: 1, sub: 1, details: 1, summary: 1
  };
  var URI_ATTRS = { a: ["href"], img: ["src"] };
  var PLAIN_ATTRS = {
    a: ["target", "rel", "title"],
    img: ["alt", "title", "loading"],
    td: ["colspan", "rowspan"],
    th: ["colspan", "rowspan"],
    code: ["class"],
    span: ["class"]
  };

  function isSafeUrl(u) {
    if (!u) return false;
    u = String(u).trim();
    if (/^(javascript|vbscript|data:text\/html):/i.test(u)) return false;
    if (/^data:image\/(png|jpe?g|gif|webp|svg\+xml);base64,/i.test(u)) return true;
    if (/^(https?:|mailto:|tel:)/i.test(u)) return true;
    if (u.charAt(0) === "/") return true;   // 站内路径（/api/images/...）
    return false;
  }

  function clean(node, doc) {
    var frag = doc.createDocumentFragment();
    var children = Array.prototype.slice.call(node.childNodes || []);
    for (var i = 0; i < children.length; i++) {
      var n = children[i];
      if (n.nodeType === 3) { frag.appendChild(doc.createTextNode(n.nodeValue)); continue; }
      if (n.nodeType !== 1) continue;
      var tag = n.tagName.toLowerCase();
      if (!ALLOWED[tag]) continue;          // 非白名单标签整块丢弃
      var clone = doc.createElement(tag);
      var attrs = n.attributes ? Array.prototype.slice.call(n.attributes) : [];
      for (var a = 0; a < attrs.length; a++) {
        var name = attrs[a].name.toLowerCase();
        if (/^on/i.test(name) || name === "style" || name === "srcdoc") continue;
        var val = n.getAttribute(attrs[a].name);
        if (val == null) continue;
        if (URI_ATTRS[tag] && URI_ATTRS[tag].indexOf(name) >= 0) {
          if (!isSafeUrl(val)) continue;
        } else if (tag === "a" && name === "target") {
          val = "_blank";
          clone.setAttribute("rel", "noopener noreferrer");
        } else if (!PLAIN_ATTRS[tag] || PLAIN_ATTRS[tag].indexOf(name) < 0) {
          continue;                          // 未列入白名单的属性剥掉
        }
        clone.setAttribute(name, val);
      }
      if (tag === "a") clone.setAttribute("rel", "noopener noreferrer");
      clone.appendChild(clean(n, doc));
      frag.appendChild(clone);
    }
    return frag;
  }

  global.sanitizeHtml = function (html) {
    if (typeof DOMParser === "undefined" || typeof document === "undefined") {
      return "";
    }
    var doc = new DOMParser().parseFromString(String(html || ""), "text/html");
    var body = doc.body || doc;
    var cleanFrag = clean(body, doc);
    var out = doc.createElement("div");
    out.appendChild(cleanFrag);
    return out.innerHTML;
  };
})(window);
