/* ══════════════════════════════════════════════════════════════════
   ArtAgent 前端逻辑（原生 JS，无构建）—— v2 布局重构
   · 侧栏：新建对话 + 历史搜索/管理
   · 资料库抽屉：上传、文档状态、数据源切换
   · 输入区：paperclip 附件 → chip（上传/解析/待确认状态），随消息发送
   · SSE 流式对话（POST + ReadableStream）
   ══════════════════════════════════════════════════════════════════ */
const $  = (s) => document.querySelector(s);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
/* ChatGPT 风格线性图标 */
function actIcon(paths, flip) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  if (flip) svg.style.transform = "scale(-1, -1)";  // 垂直+水平镜像（180°旋转）：手在右、拇指向下
  for (const d of paths) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    svg.appendChild(p);
  }
  return svg;
}
const genId = () => (crypto.randomUUID
  ? crypto.randomUUID()
  : Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10));

const LOGO = "/static/emblem.svg";

const state = {
  sid: genId(),
  title: "新对话",
  chips: new Map(),        // chip.id -> { id, doc_id, name, size, kind, status, error }
  streams: new Map(),      // sid -> { sid, bubble, abortCtrl }（每个会话独立流，支持后台生成）
  pendingRegenerate: false, // 下次发送时替换最后一条用户消息（编辑/重新生成）
};

function sidStreaming(sid) { return state.streams.has(sid); }
function anyStreaming() { return state.streams.size > 0; }

/* 每个会话的待发送附件（切会话不丢失；localStorage 兜底刷新） */
const PENDING_PREFIX = "artagent.pending.v1.";
let pendingChips = new Map();   // sid -> Map(chipId -> entry)
let editing = null;   // { wrap, bubble, afterWraps, text }：原地编辑中的用户消息

/* ── URL 会话路由：?s=<sid>，刷新后回到原对话 ── */
function currentSidFromUrl() {
  return new URLSearchParams(location.search).get("s") || "";
}
function updateUrl(sid) {
  try {
    const url = sid ? location.pathname + "?s=" + encodeURIComponent(sid) : location.pathname;
    history.replaceState(null, "", url);
  } catch (_) { /* 忽略：文件协议或受限环境 */ }
}

const dom = {
  app:        $("#app"),
  sideOverlay: $("#side-overlay"),
  sidebar:    $("#sidebar"),
  resizer:    $("#side-resizer"),
  sideCollapse: $("#btn-side-collapse"),
  history:    $("#history"),
  histFilter: $("#hist-filter"),
  tbTitle:    $("#tb-title"),
  btnLibrary: $("#btn-library"),
  welcome:    $("#welcome"),
  chat:       $("#chat"),
  cards:      $("#cards"),
  chips:      $("#chips"),
  msg:        $("#msg"),
  send:       $("#btn-send"),
  attach:     $("#btn-attach"),
  file:       $("#file-input"),
  jump:       $("#btn-jump"),
  library:    $("#library"),
  docList:    $("#doc-list"),
  docCount:   $("#doc-count"),
  upload:     $("#btn-upload"),
  schemaModal:   $("#schema-modal"),
  schemaMeta:    $("#schema-meta"),
  schemaReason:  $("#schema-reason"),
  schemaError:   $("#schema-error"),
  schemaEntity:  $("#schema-entity"),
  schemaAxis:    $("#schema-axis"),
  schemaDesc:    $("#schema-desc"),
  schemaImage:   $("#schema-image"),
  schemaDisplay: $("#schema-display"),
  schemaOk:      $("#schema-ok"),
  schemaCancel:  $("#schema-cancel"),
  confirmModal:  $("#confirm-modal"),
  confirmTitle:  $("#confirm-title"),
  confirmText:   $("#confirm-text"),
  confirmOk:     $("#confirm-ok"),
  confirmCancel: $("#confirm-cancel"),
  toast:         $("#toast"),
  memoryPanel:   $("#memory-panel"),
  memList:       $("#mem-list"),
  memClear:      $("#mem-clear"),
  memClose:      $("#mem-close"),
  btnMemory:     $("#btn-memory"),
};

/* ══════════════ 视图切换 ══════════════ */
function showWelcome() {
  dom.welcome.classList.remove("is-hidden");
  dom.chat.classList.add("is-hidden");
  dom.chat.innerHTML = "";
  setSendStop(sidStreaming(state.sid));
  syncTitle();
  updateJumpButton();
}
function showChat() {
  dom.welcome.classList.add("is-hidden");
  dom.chat.classList.remove("is-hidden");
  setSendStop(sidStreaming(state.sid));
  updateJumpButton();
}

/* 一键回到最新消息按钮：仅当上翻离开底部且有消息时显示 */
function updateJumpButton() {
  const hidden = dom.chat.classList.contains("is-hidden")
    || stickBottom
    || !dom.chat.querySelector(".turn");
  dom.jump.hidden = hidden;
}

function jumpToBottom() {
  stickBottom = true;
  dom.chat.scrollTo({ top: dom.chat.scrollHeight, behavior: "smooth" });
  dom.jump.hidden = true;
}

/* ══════════════ 消息渲染 ══════════════ */
function assistantAvatar() {
  const a = el("div", "avatar");
  const img = el("img"); img.src = LOGO; img.alt = ""; a.appendChild(img);
  return a;
}
function userAvatar() { const a = el("div", "avatar"); a.textContent = "我"; return a; }

const FILE_STATUS_LABEL = {
  uploading: "上传中…",
  processing: "解析中…",
  pending_confirm: "待确认 schema",
  done: "已就绪",
  active: "已就绪",
  failed: "失败",
};

function fileCard(entry) {
  const c = el("div", "file-card");
  const ico = el("span", "fc-ico"); ico.textContent = entry.kind === "table" ? "📊" : "📄";
  const name = el("span", "fc-name"); name.textContent = entry.name;
  name.setAttribute("data-tip", entry.name);
  const badge = el("span", "fc-badge");
  badge.textContent = FILE_STATUS_LABEL[entry.status] || entry.status || "";
  c.append(ico, name, badge);
  return c;
}

function addUserTurn(text, files) {
  const turn = el("div", "turn user");
  const bubble = el("div", "bubble");
  if (text) {
    const p = el("p"); p.textContent = text;   // 用户输入：纯文本，防注入
    bubble.appendChild(p);
  }
  for (const f of (files || [])) bubble.appendChild(fileCard(f));
  turn.append(userAvatar(), bubble);
  wrapAndAppend(turn);
  addUserActions(turn);
}
function addAssistantTurn(html, opts = {}) {
  const turn = el("div", "turn assistant");
  const bubble = el("div", "bubble");
  bubble.dataset.sid = state.sid;   // 反馈归属当前会话
  setBubbleHTML(bubble, html || "");
  turn.append(assistantAvatar(), bubble);
  wrapAndAppend(turn);
  addAssistantActions(turn, bubble);
  if (opts.pending) {
    const actions = turn.querySelector(".msg-actions");
    if (actions) actions.hidden = true;  // 回答完成前不显示复制/重试等按钮
  }
  return bubble;
}

function showAssistantActions(bubble) {
  const turn = bubble.parentElement;
  const actions = turn && turn.querySelector(".msg-actions");
  if (actions) actions.hidden = false;
}

/* ── 消息操作：复制 / 重新生成 / 编辑（对标成熟平台） ── */
function addAssistantActions(turn, bubble) {
  const actions = el("div", "msg-actions");
  const copy = el("button", "msg-act"); copy.type = "button";
  copy.setAttribute("data-tip", "复制回答"); copy.setAttribute("aria-label", "复制回答");
  copy.appendChild(actIcon([
    "M9 9h11a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V10a1 1 0 0 1 1-1Z",
    "M5 15V5a1 1 0 0 1 1-1h10",
  ]));
  copy.addEventListener("click", async () => {
    const md = bubble.querySelector(".md-answer");
    const raw = md ? md.textContent : bubble.textContent;
    try {
      await navigator.clipboard.writeText(raw || "");
      toast("已复制到剪贴板");
    } catch (e) {
      toast("复制失败：" + e.message, "err");
    }
  });
  const regen = el("button", "msg-act"); regen.type = "button";
  regen.dataset.action = "regenerate";
  regen.setAttribute("data-tip", "重新生成回答");
  regen.setAttribute("aria-label", "重新生成回答");
  regen.appendChild(actIcon([
    "M23 4v6h-6",
    "M20.49 15a9 9 0 1 1-2.12-9.36L23 10",
  ]));
  regen.addEventListener("click", () => regenerateLast(turn));
  /* G1 反馈闭环：👍 / 👎 + 原因标签 */
  const fbUp = el("button", "msg-act fb"); fbUp.type = "button";
  fbUp.setAttribute("data-tip", "回答有帮助"); fbUp.setAttribute("aria-label", "回答有帮助");
  fbUp.appendChild(actIcon([
    "M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3z",
    "M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3",
  ]));
  const fbDown = el("button", "msg-act fb"); fbDown.type = "button";
  fbDown.setAttribute("data-tip", "回答有问题");
  fbDown.setAttribute("aria-label", "回答有问题");
  fbDown.appendChild(actIcon([
    "M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3z",
    "M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3",
  ], true));
  let rated = 0;
  fbUp.addEventListener("click", () => {
    if (sidStreaming(state.sid)) { toast("请等待回答完成后再评价"); return; }
    if (rated) return;
    rated = 1;
    submitFeedback(bubble, 1, "", "");
    markRated(turn, fbUp);
  });
  fbDown.addEventListener("click", () => {
    if (sidStreaming(state.sid)) { toast("请等待回答完成后再评价"); return; }
    if (rated) return;
    rated = -1;
    openFeedbackPanel(bubble, turn, fbDown);
  });
  actions.append(copy, regen, fbUp, fbDown);
  turn.appendChild(actions);
}

/* 评价后：禁用本消息全部反馈按钮并高亮所选 */
function markRated(turn, chosenBtn) {
  turn.querySelectorAll(".msg-act.fb").forEach((b) => {
    b.disabled = true;
    b.classList.toggle("on", b === chosenBtn);
    b.setAttribute("data-tip", "已评价");
  });
}

/* 踩反馈：原因标签 + 可选补充说明 */
const FB_REASONS = ["回答不准确", "引用不充分", "过于冗长", "其他"];
function openFeedbackPanel(bubble, turn, fbDown) {
  // 同一时间只允许一个反馈面板
  dom.chat.querySelectorAll(".fb-panel").forEach((p) => p.remove());
  const panel = el("div", "fb-panel");
  const head = el("div", "fb-head"); head.textContent = "哪里有问题？";
  const opts = el("div", "fb-options");
  let reason = "";
  for (const r of FB_REASONS) {
    const chip = el("button", "fb-opt"); chip.type = "button"; chip.textContent = r;
    chip.addEventListener("click", () => {
      opts.querySelectorAll(".fb-opt").forEach((c) => c.classList.remove("on"));
      chip.classList.add("on");
      reason = r;
    });
    opts.appendChild(chip);
  }
  const comment = el("input", "fb-comment");
  comment.type = "text"; comment.maxLength = 500;
  comment.placeholder = "补充说明（可选）";
  const row = el("div", "fb-submit-row");
  const cancel = el("button", "fb-cancel"); cancel.type = "button"; cancel.textContent = "取消";
  cancel.addEventListener("click", () => panel.remove());
  const submit = el("button", "fb-submit"); submit.type = "button"; submit.textContent = "提交反馈";
  submit.addEventListener("click", () => {
    if (!reason) { toast("请先选择一个原因", "err"); return; }
    submitFeedback(bubble, -1, reason, comment.value.trim());
    markRated(turn, fbDown);
    panel.remove();
  });
  row.append(cancel, submit);
  panel.append(head, opts, comment, row);
  turn.appendChild(panel);
  setTimeout(() => comment.focus(), 50);
}

async function submitFeedback(bubble, rating, reason, comment) {
  const sid = bubble.dataset.sid || state.sid;
  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, rating, reason, comment }),
    });
    const j = await res.json().catch(() => ({}));
    if (res.ok && j.ok) toast(rating === 1 ? "感谢反馈 👍" : "已记录，我们会改进");
    else toast(j.error || "反馈提交失败", "err");
  } catch (e) {
    toast("反馈提交失败：" + e.message, "err");
  }
}

/* G3 引用可交互：把回答里的 [N] 变成可点击角标，联动来源卡 */
function linkCitations(bubble, sources) {
  if (!sources || !sources.length) return;
  const answer = bubble.querySelector(".md-answer");
  if (!answer || answer.dataset.cited) return;
  const walker = document.createTreeWalker(answer, NodeFilter.SHOW_TEXT, null);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  const re = /\[(\d{1,2})\]/g;
  for (const node of textNodes) {
    const text = node.nodeValue || "";
    re.lastIndex = 0;
    if (!re.test(text)) continue;
    re.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0, m;
    while ((m = re.exec(text))) {
      const n = parseInt(m[1], 10);
      if (n < 1 || n > sources.length) continue;
      frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const sup = document.createElement("sup");
      sup.className = "cite";
      sup.textContent = m[0];
      sup.dataset.i = String(n);
      sup.setAttribute("data-tip", sources[n - 1].label || "");
      sup.addEventListener("click", () => focusSource(bubble, n));
      frag.appendChild(sup);
      last = m.index + m[0].length;
    }
    if (last) {
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }
  answer.dataset.cited = "1";
}

function focusSource(bubble, n) {
  const chip = bubble.querySelector('.source-chip[data-i="' + n + '"]');
  if (!chip) return;
  chip.scrollIntoView({ behavior: "smooth", block: "nearest" });
  chip.classList.remove("flash"); void chip.offsetWidth; chip.classList.add("flash");
}

/* 错误态：提供重试按钮（G1.5） */
function addRetryButton(bubble) {
  const turn = bubble.closest(".turn");
  if (!turn || turn.querySelector(".msg-act-retry")) return;
  let actions = turn.querySelector(".msg-actions") || (() => {
    const a = el("div", "msg-actions");
    turn.appendChild(a);
    return a;
  })();
  // 去掉同义的"重新生成"，避免同一消息出现双入口
  [...actions.querySelectorAll(".msg-act")].forEach((b) => {
    if (b.dataset.action === "regenerate") b.remove();
  });
  const retry = el("button", "msg-act msg-act-retry"); retry.type = "button";
  retry.setAttribute("data-tip", "重试"); retry.setAttribute("aria-label", "重试");
  retry.appendChild(actIcon([
    "M23 4v6h-6",
    "M20.49 15a9 9 0 1 1-2.12-9.36L23 10",
  ]));
  retry.addEventListener("click", () => regenerateLast(turn));
  actions.insertBefore(retry, actions.firstChild);
  bubble.classList.add("bubble-error");
}

function addUserActions(turn) {
  const actions = el("div", "msg-actions");
  const edit = el("button", "msg-act"); edit.type = "button";
  edit.setAttribute("data-tip", "编辑这条消息");
  edit.setAttribute("aria-label", "编辑这条消息");
  edit.appendChild(actIcon([
    "M12 20h9",
    "M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z",
  ]));
  edit.addEventListener("click", () => editLast(turn));
  actions.appendChild(edit);
  turn.appendChild(actions);
}

function lastUserMessageText() {
  const users = dom.chat.querySelectorAll(".turn.user .bubble p");
  return users.length ? users[users.length - 1].textContent : "";
}

function regenerateLast(turn) {
  if (sidStreaming(state.sid)) return;
  const assistants = dom.chat.querySelectorAll(".turn.assistant");
  if (!turn || assistants.length === 0 || assistants[assistants.length - 1] !== turn) {
    toast("只能重新生成最后一条回答");
    return;
  }
  const wrap = turn.closest(".chat-wrap");
  const prevWrap = wrap && wrap.previousElementSibling;
  const text = prevWrap
    ? ((prevWrap.querySelector(".turn.user .bubble p") || {}).textContent || "")
    : lastUserMessageText();
  if (!text) return;
  if (wrap) wrap.remove();
  send(text, { regenerate: true });
}

function editLast(turn) {
  if (sidStreaming(state.sid)) return;
  if (editing) { toast("请先完成当前编辑"); return; }
  const users = dom.chat.querySelectorAll(".turn.user");
  if (!turn || users.length === 0 || users[users.length - 1] !== turn) {
    toast("只能编辑最后一条消息");
    return;
  }
  const p = turn.querySelector(".bubble p");
  const text = p ? p.textContent : "";
  // 移除后续的助手回答（重新生成后会补上；取消时再插回）
  const afterWraps = [];
  let node = turn.closest(".chat-wrap").nextElementSibling;
  while (node) {
    const next = node.nextElementSibling;
    afterWraps.push(node);
    node.remove();
    node = next;
  }
  // 消息气泡原地变为编辑卡片
  const bubble = turn.querySelector(".bubble");
  const actions = turn.querySelector(".msg-actions");
  if (actions) actions.style.display = "none";
  turn.classList.add("editing");
  bubble.innerHTML = "";
  const ta = el("textarea", "edit-input");
  ta.value = text; ta.maxLength = 8000;
  ta.setAttribute("aria-label", "编辑消息内容");
  const row = el("div", "edit-row");
  const cancel = el("button", "edit-cancel"); cancel.type = "button"; cancel.textContent = "取消";
  const save = el("button", "edit-save"); save.type = "button"; save.textContent = "保存";
  row.append(cancel, save);
  bubble.append(ta, row);
  editing = { wrap: turn.closest(".chat-wrap"), bubble, afterWraps, text };

  function finish(canceled) {
    if (!editing) return;
    const e = editing;
    editing = null;
    if (actions) actions.style.display = "";
    turn.classList.remove("editing");
    if (canceled) {
      // 恢复原文气泡与后续回答
      const t = e.wrap.querySelector(".turn.user");
      const b = t ? t.querySelector(".bubble") : null;
      if (b) {
        b.innerHTML = "";
        const np = el("p"); np.textContent = e.text;
        b.appendChild(np);
      }
      let ref = e.wrap;
      for (const w of e.afterWraps) {
        ref.parentNode.insertBefore(w, ref.nextSibling);
        ref = w;
      }
    }
  }

  cancel.addEventListener("click", () => finish(true));
  save.addEventListener("click", () => {
    const val = ta.value.trim();
    if (!val) { toast("消息不能为空", "err"); return; }
    const wrap = editing.wrap;
    finish(false);
    send(val, { editInPlace: true, editWrap: wrap });
  });
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); save.click(); }
    else if (e.key === "Escape") { e.preventDefault(); cancel.click(); }
  });
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
  ta.focus();
  const end = ta.value.length;
  ta.setSelectionRange(end, end);
}

/* 会话历史里的「已上传文档」卡片（切会话/刷新后仍渲染） */
function addAttachmentTurn(m) {
  const turn = el("div", "turn system");
  const bubble = el("div", "bubble system-note");
  const card = el("div", "file-card attach-note");
  const ico = el("span", "fc-ico");
  ico.textContent = m.kind === "table" ? "📊" : "📄";
  const name = el("span", "fc-name");
  name.textContent = "已上传《" + (m.doc_name || m.content || "文档") + "》";
  name.setAttribute("data-tip", m.doc_name || "");
  card.dataset.docId = m.doc_id || "";
  card.append(ico, name);
  bubble.appendChild(card);
  turn.appendChild(bubble);
  wrapAndAppend(turn);
}

/* 挂载气泡 HTML，并把 .md-answer 里的 Markdown 原文用 marked 渲染 */
function escapeHtmlText(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function setBubbleHTML(bubble, html) {
  bubble.innerHTML = html;
  if (!window.marked) return;
  if (!window.__markedOpts) {
    marked.setOptions({ breaks: true, gfm: true });
    window.__markedOpts = true;
  }
  bubble.querySelectorAll(".md-answer").forEach((box) => {
    if (box.dataset.rendered) return;
    /* G4/1.4 XSS 修复：marked 输出必须过白名单消毒器 */
    const raw = box.textContent || "";
    const parsed = (window.marked.parse || window.marked)(raw);
    if (!window.sanitizeHtml) {
      /* fail-closed：消毒器缺失时不渲染任何 HTML，降级为纯文本 */
      box.innerHTML = escapeHtmlText(parsed);
      box.dataset.rendered = "1";
      if (!window.__sanitizeWarned) {
        window.__sanitizeWarned = true;
        toast("内容消毒器未加载，Markdown 已降级为纯文本", "err");
      }
      return;
    }
    box.innerHTML = window.sanitizeHtml(parsed);
    box.dataset.rendered = "1";
  });
}

/* 参考来源卡片（done 事件 / 会话历史携带） */
function renderSources(bubble, sources) {
  if (!sources || !sources.length) return;
  if (bubble.querySelector(".sources")) return;
  const box = el("div", "sources");
  const head = el("div", "sources-head");
  head.textContent = "参考来源";
  box.appendChild(head);
  sources.forEach((s, i) => {
    const chip = el("button", "source-chip"); chip.type = "button";
    chip.dataset.i = String(i + 1);
    chip.textContent = (i + 1) + " " + (s.label || "");
    chip.setAttribute("data-tip", s.label || "");
    chip.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(s.label || "");
        toast("已复制来源");
      } catch (_) { /* 剪贴板不可用时忽略 */ }
    });
    box.appendChild(chip);
  });
  bubble.appendChild(box);
  linkCitations(bubble, sources);
}

/* 自动滚动：仅在用户贴近底部时跟随 */
let stickBottom = true;
function scrollChat() { if (stickBottom) dom.chat.scrollTop = dom.chat.scrollHeight; }
function wrapAndAppend(turn) {
  const wrap = el("div", "chat-wrap"); wrap.appendChild(turn);
  dom.chat.appendChild(wrap);
  scrollChat();
}
dom.chat.addEventListener("scroll", () => {
  stickBottom = dom.chat.scrollHeight - dom.chat.scrollTop - dom.chat.clientHeight < 120;
  updateJumpButton();
});
dom.jump.addEventListener("click", jumpToBottom);

/* ══════════════ 发送（SSE 流式） ══════════════ */
async function send(text, opts = {}) {
  text = (text || "").trim();
  const hasFiles = state.chips.size > 0;
  if ((!text && !hasFiles) || sidStreaming(state.sid)) return;
  if (opts.regenerate || opts.editInPlace) state.pendingRegenerate = true;

  const sidAtSend = state.sid;
  const filesAtSend = [...state.chips.values()];
  setSendStop(true);
  if (!sessionsMap.has(state.sid)) {
    state.title = (text || "上传文档").slice(0, 24) || "新对话";
    syncTitle();
  }
  showChat();
  // 新会话发送首条消息时，侧边栏立即出现该对话（先本地占位，收尾再同步服务端）
  if (!sessionsMap.has(state.sid)) {
    sessionsMap = new Map([
      [state.sid, {
        session_id: state.sid,
        title: state.title,
        updated_at: new Date().toISOString(),
        relative: "刚刚",
      }],
      ...sessionsMap.entries(),
    ]);
    renderHistory([...sessionsMap.values()], sessionsMap.size >= SESSIONS_PAGE);
    markActive(state.sid);
  }
  if (opts.editInPlace && opts.editWrap) {
    // 原地编辑：编辑框替换为普通用户消息（新文本）
    const t = opts.editWrap.querySelector(".turn.user");
    const b = t ? t.querySelector(".bubble") : null;
    if (b) {
      b.innerHTML = "";
      const np = el("p"); np.textContent = text;
      b.appendChild(np);
    }
  } else {
    addUserTurn(text, [...state.chips.values()]);
  }
  state.chips.clear(); renderChips(); persistChips(sidAtSend);
  dom.msg.value = ""; autoResize();

  const bubble = addAssistantTurn(
    '<div class="think-box" open><summary>正在思考…</summary></div>',
    { pending: true });
  const abortCtrl = new AbortController();
  const entry = {
    sid: sidAtSend, bubble, wrap: bubble.closest(".chat-wrap"), abortCtrl,
    userText: text || "请查阅我上传的文档并回答。",
    files: filesAtSend,
  };
  state.streams.set(sidAtSend, entry);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortCtrl.signal,
      body: JSON.stringify({
        message: text || "请查阅我上传的文档并回答。",
        session_id: sidAtSend,
        regenerate: state.pendingRegenerate,
      }),
    });
    if (!res.ok) {
      let msg = `服务请求失败（HTTP ${res.status}）`;
      try {
        const j = await res.json();
        if (j.error) msg = String(j.error);
        else if (j.detail) msg = String(j.detail);
      } catch (_) { /* ignore */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let gotDone = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "delta") {
          setBubbleHTML(bubble, evt.html);
          scrollChat();
        } else if (evt.type === "done") {
          gotDone = true;
          setBubbleHTML(bubble, evt.html);
          renderSources(bubble, evt.sources || []);
          showAssistantActions(bubble);
          const ans = bubble.querySelector(".md-answer");
          if (ans) ans.setAttribute("aria-live", "polite");
          if (evt.error) addRetryButton(bubble);
          if (evt.session_id && state.sid === sidAtSend) {
            state.sid = evt.session_id;
            updateUrl(state.sid);
          }
          if (typeof evt.memory === "number") setLevel(evt.memory);
          scrollChat();
        }
      }
    }
    if (!gotDone) throw new Error("连接中断，未收到完整回复，请重试。");
  } catch (err) {
    if (err.name === "AbortError") {
      const note = el("div", "stop-note");
      note.textContent = "⏹ 已停止生成";
      bubble.appendChild(note);
      showAssistantActions(bubble);
      scrollChat();
    } else {
      bubble.innerHTML = "😔 " + (err.message || "网络中断或服务未响应，请稍后重试。");
      addRetryButton(bubble);
      showAssistantActions(bubble);
    }
    console.error(err);
  } finally {
    if (state.streams.get(sidAtSend) === entry) state.streams.delete(sidAtSend);
    // 发送/停止按钮跟随“当前会话”的流状态：后台流继续跑，不阻塞新会话
    setSendStop(sidStreaming(state.sid));
    if (state.sid === sidAtSend) {
      state.pendingRegenerate = false;
      updateJumpButton();
    }
    // 后台流完成后，若用户正停留在该会话（气泡已被重新渲染/分离）→ 重载展示最终答案
    if (gotDone && state.sid === sidAtSend && !bubble.isConnected) {
      openSession(sidAtSend);
    }
    await loadSessions();
  }
}

/* 发送/停止按钮切换 */
function setSendStop(stop) {
  dom.send.classList.toggle("stopping", stop);
  dom.send.setAttribute("aria-label", stop ? "停止生成" : "发送");
  dom.send.disabled = false;
}

function stopGeneration() {
  const entry = state.streams.get(state.sid);
  if (entry) entry.abortCtrl.abort();
}

/* ══════════════ 侧栏：历史会话 ══════════════ */
let sessionsMap = new Map();   // sid -> session
let sessionsTotal = 0;
let sessionsOffset = 0;
const SESSIONS_PAGE = 50;

async function loadSessions() {
  try {
    const data = await (await fetch(`/api/sessions?offset=0&limit=${SESSIONS_PAGE}`)).json();
    sessionsMap = new Map((data.items || []).map((s) => [s.session_id, s]));
    sessionsTotal = data.total || sessionsMap.size;
    sessionsOffset = (data.items || []).length;
    renderHistory([...sessionsMap.values()], (data.has_more) === true);
  } catch (e) { console.error(e); }
}

async function loadMoreSessions() {
  try {
    const data = await (await fetch(
      `/api/sessions?offset=${sessionsOffset}&limit=${SESSIONS_PAGE}`)).json();
    for (const s of (data.items || [])) sessionsMap.set(s.session_id, s);
    sessionsTotal = data.total || sessionsMap.size;
    sessionsOffset += (data.items || []).length;
    renderHistory([...sessionsMap.values()], (data.has_more) === true);
    markActive(state.sid);
  } catch (e) { console.error(e); }
}

function renderHistory(list, hasMore) {
  const prevScroll = dom.history.scrollTop;
  dom.history.innerHTML = "";
  if (!list.length) {
    const e = el("div", "hist-empty"); e.textContent = "暂无历史对话";
    dom.history.appendChild(e); dom.history.scrollTop = prevScroll; syncTitle(); return;
  }
  const q = (dom.histFilter.value || "").trim().toLowerCase();
  const filtered = q ? list.filter((s) => (s.title || "").toLowerCase().includes(q)) : list;
  if (!filtered.length) {
    const e = el("div", "hist-empty"); e.textContent = "无匹配的对话";
    dom.history.appendChild(e); dom.history.scrollTop = prevScroll; syncTitle(); return;
  }
  for (const s of filtered) {
    const b = el("button", "hist-btn"); b.type = "button"; b.dataset.sid = s.session_id;
    if (s.session_id === state.sid) b.classList.add("active");
    const body = el("span", "h-body");
    const t = el("span", "h-title"); t.textContent = s.title || "未命名对话";
    const tm = el("span", "h-time"); tm.textContent = s.relative || "";
    body.append(t, tm);
    const del = el("button", "hist-del"); del.type = "button"; del.textContent = "×";
    del.setAttribute("data-tip", "删除该对话"); del.setAttribute("aria-label", "删除对话");
    del.addEventListener("click", (ev) => { ev.stopPropagation(); deleteSession(s.session_id, s.title); });
    const ren = el("button", "hist-ren"); ren.type = "button"; ren.textContent = "✎";
    ren.setAttribute("data-tip", "重命名"); ren.setAttribute("aria-label", "重命名对话");
    ren.addEventListener("click", (ev) => { ev.stopPropagation(); renameSession(s, b); });
    b.append(body, ren, del);
    b.addEventListener("click", () => openSession(s.session_id));
    dom.history.appendChild(b);
  }
  if (hasMore) {
    const more = el("button", "hist-more"); more.type = "button";
    more.textContent = "加载更多";
    more.addEventListener("click", loadMoreSessions);
    dom.history.appendChild(more);
  }
  dom.history.scrollTop = prevScroll;
  syncTitle();
}

function renameSession(s, btn) {
  const titleEl = btn.querySelector(".h-title");
  if (!titleEl) return;
  const input = el("input", "hist-rename-input");
  input.type = "text"; input.maxLength = 60;
  input.value = s.title || "";
  input.setAttribute("aria-label", "重命名对话");
  titleEl.replaceWith(input);
  input.focus(); input.select();
  let committed = false;
  const commit = async (save) => {
    if (committed) return;
    committed = true;
    const val = input.value.trim();
    if (save && val && val !== (s.title || "")) {
      try {
        const res = await fetch(`/api/sessions/${s.session_id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: val }),
        });
        const j = await res.json().catch(() => ({}));
        if (!res.ok || !j.ok) toast(j.error || "重命名失败", "err");
        else toast("已重命名");
      } catch (e) {
        toast("重命名失败：" + e.message, "err");
      }
    }
    loadSessions();
    if (state.sid === s.session_id) { state.title = val || s.title; syncTitle(); }
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(true); }
    else if (e.key === "Escape") { commit(false); }
  });
  input.addEventListener("blur", () => commit(true));
}

function markActive(sid) {
  dom.history.querySelectorAll(".hist-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.sid === sid);
  });
}

function syncTitle() {
  const s = sessionsMap.get(state.sid);
  dom.tbTitle.textContent = (s && s.title) ? s.title : state.title;
  dom.tbTitle.setAttribute("data-tip", dom.tbTitle.textContent);
}

async function openSession(sid) {
  stashChips();
  editing = null;
  // 后台/正在流式生成的会话还没落库：直接挂回实时气泡，不读空历史
  const live = state.streams.get(sid);
  if (live && live.bubble) {
    state.sid = sid;
    state.title = (sessionsMap.get(sid) || {}).title || state.title;
    showChat();
    dom.chat.innerHTML = "";
    // 已落库的历史（本轮之前的轮次）
    try {
      const data = await (await fetch(`/api/sessions/${sid}`)).json();
      for (const m of (data.messages || [])) {
        if (m.role === "user") addUserTurn(m.content, []);
        else if (m.role === "attachment") addAttachmentTurn(m);
        else {
          const b = addAssistantTurn(m.content);
          if (m.sources) renderSources(b, m.sources);
        }
      }
    } catch (_) { /* 历史读取失败不影响实时气泡 */ }
    // 本轮尚未落库的用户消息 + 实时生成气泡
    if (live.userText) addUserTurn(live.userText, live.files || []);
    dom.chat.appendChild(live.wrap || live.bubble);
    markActive(sid);
    syncTitle();
    updateUrl(sid);
    loadChipsFor(sid);
    stickBottom = true;
    scrollChat();
    updateJumpButton();
    return;
  }
  try {
    const data = await (await fetch(`/api/sessions/${sid}`)).json();
    if (!(data.messages || []).length) { newSession(); return; }
    state.sid = sid;
    state.title = (sessionsMap.get(sid) || {}).title || "未命名对话";
    showChat();
    dom.chat.innerHTML = "";
    for (const m of (data.messages || [])) {
      if (m.role === "user") addUserTurn(m.content, []);
      else if (m.role === "attachment") addAttachmentTurn(m);
      else {
        const bubble = addAssistantTurn(m.content);
        if (m.sources) renderSources(bubble, m.sources);
      }
    }
    state.pendingRegenerate = false;
    markActive(sid);
    syncTitle();
    updateUrl(sid);
    loadChipsFor(sid);
    stickBottom = true;
    scrollChat();
    updateJumpButton();
    restoreFeedbackState(sid);
  } catch (e) { console.error(e); toast("加载会话失败", "err"); }
}

/* 刷新/重开会话后，若该会话已有反馈则禁用全部评价按钮（防重复提交） */
async function restoreFeedbackState(sid) {
  try {
    const data = await (await fetch("/api/feedback?limit=200")).json();
    const rated = (data.items || []).some((f) => f.session_id === sid);
    if (!rated) return;
    dom.chat.querySelectorAll(".msg-act.fb").forEach((b) => {
      b.disabled = true;
      b.setAttribute("data-tip", "本会话已评价过");
    });
  } catch (_) { /* 忽略：反馈服务不可用时不影响主流程 */ }
}

function newSession() {
  stashChips();
  editing = null;
  state.sid = genId();
  state.title = "新对话";
  state.pendingRegenerate = false;
  state.chips = new Map();
  renderChips();
  showWelcome();
  markActive(null);
  syncTitle();
  updateUrl("");
}

async function deleteSession(sid, title) {
  const ok = await confirmAsk({
    title: "删除对话",
    text: `确定删除「${title || "未命名对话"}」？删除后不可恢复。`,
    okText: "删除", danger: true,
  });
  if (!ok) return;
  if (sid === state.sid) { state.chips = new Map(); renderChips(); }
  clearPending(sid);
  try { await fetch(`/api/sessions/${sid}`, { method: "DELETE" }); } catch (_) { /* ignore */ }
  if (sid === state.sid) newSession();
  loadSessions();
}

/* ══════════════ 首屏：场景卡 + 记忆入口提示 ══════════════ */
function setLevel(memory) {
  // 记忆入口保留在头像按钮，不显示等级/条数/文案
}

async function bootstrap() {
  try {
    const data = await (await fetch("/api/bootstrap")).json();
    setLevel(data.memory);
    dom.cards.innerHTML = "";
    for (const c of (data.cards || [])) {
      const card = el("button", "scene-card"); card.type = "button";
      if (c.thumb) {
        const img = el("img", "sc-thumb"); img.src = c.thumb; img.alt = ""; img.loading = "lazy";
        card.appendChild(img);
      }
      const body = el("div", "sc-body");
      const title = el("div", "sc-title"); title.textContent = c.text;
      body.appendChild(title);
      card.appendChild(body);
      card.onclick = () => send(c.query);
      dom.cards.appendChild(card);
    }
  } catch (e) {
    console.error(e);
    const p = el("p"); p.className = "hist-empty"; p.textContent = "场景卡加载失败";
    dom.cards.appendChild(p);
  }
}

/* ══════════════ 记忆面板：记忆可见可控（G2 + Phase 2） ══════════════ */
const KIND_LABEL = {
  preference: "偏好", fact: "事实", profile: "画像", event: "事件",
  artist: "画家", style: "风格",
};
const SOURCE_LABEL = {
  user_explicit: "用户明确", extracted: "自动抽取", eval: "评估数据",
};
function relTime(iso) {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  const days = Math.floor((Date.now() - t.getTime()) / 86400000);
  return days <= 0 ? "今天" : days + " 天前";
}
function openMemoryPanel() { rememberFocus(); dom.memoryPanel.hidden = false; loadMemoryPanel(); }
function closeMemoryPanel() { dom.memoryPanel.hidden = true; restoreFocus(); }

async function loadMemoryPanel() {
  try {
    const data = await (await fetch("/api/memory")).json();
    const items = data.items || [];
    dom.memList.innerHTML = "";
    if (!items.length) {
      const e = el("div", "mem-empty");
      e.textContent = "还没有记忆。多聊聊喜欢的画家、风格和偏好，Agent 会记住（含自动抽取）。";
      dom.memList.appendChild(e);
      return;
    }
    for (const it of items) {
      const row = el("div", "mem-item");
      const main = el("div", "mem-main");
      const kind = el("span", "mem-kind");
      kind.textContent = KIND_LABEL[it.kind] || it.kind || "记忆";
      const val = el("span", "mem-value"); val.textContent = it.content || it.value || "";
      val.setAttribute("data-tip", it.content || it.value || "");
      main.append(kind, val);
      const meta = el("div", "mem-meta");
      const src = SOURCE_LABEL[it.source] || it.source || "用户明确";
      const when = relTime(it.updated_at);
      meta.textContent = (it.entity ? "实体：" + it.entity + " · " : "") + src +
        (when ? " · " + when : "");
      const del = el("button", "mem-del"); del.type = "button"; del.textContent = "×";
      del.setAttribute("data-tip", "删除这条记忆");
      del.setAttribute("aria-label", `删除记忆 ${it.content || it.value}`);
      del.addEventListener("click", async () => {
        const ok = await confirmAsk({
          title: "删除记忆",
          text: `确定删除「${it.content || it.value}」这条记忆？删除后不可恢复。`,
          okText: "删除", danger: true,
        });
        if (!ok) return;
        try {
          const res = await fetch(
            "/api/memory/" + encodeURIComponent(it.id),
            { method: "DELETE" },
          );
          const j = await res.json().catch(() => ({}));
          if (!res.ok || !j.ok) { toast(j.error || "删除失败", "err"); return; }
          toast("已删除这条记忆");
          if (typeof j.memory === "number") setLevel(j.memory);
          loadMemoryPanel();
        } catch (e) { toast("删除失败：" + e.message, "err"); }
      });
      row.append(main, meta, del);
      dom.memList.appendChild(row);
    }
  } catch (e) {
    console.error(e);
    toast("加载记忆失败", "err");
  }
}

dom.btnMemory.addEventListener("click", openMemoryPanel);
dom.memClose.addEventListener("click", closeMemoryPanel);
dom.memoryPanel.addEventListener("click", (e) => {
  if (e.target === dom.memoryPanel) closeMemoryPanel();
});
dom.memClear.addEventListener("click", async () => {
  const ok = await confirmAsk({
    title: "清空全部记忆",
    text: "确定清空 Agent 记住的全部偏好？不可恢复。",
    okText: "清空", danger: true,
  });
  if (!ok) return;
  try {
    const res = await fetch("/api/memory", { method: "DELETE" });
    const j = await res.json().catch(() => ({}));
    if (!res.ok || !j.ok) { toast(j.error || "清空失败", "err"); return; }
    toast("记忆已清空");
    if (typeof j.memory === "number") setLevel(j.memory);
    loadMemoryPanel();
  } catch (e) { toast("清空失败：" + e.message, "err"); }
});

/* ══════════════ 输入框 ══════════════ */
function autoResize() {
  dom.msg.style.height = "auto";
  dom.msg.style.height = Math.min(dom.msg.scrollHeight, 168) + "px";
}
dom.msg.addEventListener("input", autoResize);
dom.msg.addEventListener("keydown", (e) => {
  // isComposing：中文输入法选词确认时不能发送
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send(dom.msg.value);
  }
});
dom.send.addEventListener("click", () => {
  if (sidStreaming(state.sid)) { stopGeneration(); return; }
  send(dom.msg.value);
});
$("#btn-new").addEventListener("click", newSession);

/* ══════════════ 附件：chip 上传（输入区 paperclip） ══════════════ */
let uploadBusy = 0;

/* 待发送附件按会话保存：切会话/刷新后恢复 */
function stashChips() {
  if (!state.sid) return;
  if (state.chips.size) {
    pendingChips.set(state.sid, state.chips);
    persistChips(state.sid);
  } else {
    clearPending(state.sid);
  }
}

function loadChipsFor(sid) {
  const saved = pendingChips.get(sid);
  state.chips = saved ? saved : restoreChipsFromStorage(sid);
  renderChips();
}

function persistChips(sid) {
  const m = (state.sid === sid) ? state.chips : pendingChips.get(sid);
  try {
    const arr = [...(m || []).values()]
      .filter((c) => c.doc_id)
      .map((c) => ({ id: c.id, doc_id: c.doc_id, name: c.name, size: c.size,
                    kind: c.kind, status: c.status, error: c.error }));
    localStorage.setItem(PENDING_PREFIX + sid, JSON.stringify(arr));
  } catch (_) { /* 隐私模式等环境忽略 */ }
}

function restoreChipsFromStorage(sid) {
  try {
    const raw = localStorage.getItem(PENDING_PREFIX + sid);
    if (!raw) return new Map();
    const arr = JSON.parse(raw) || [];
    return new Map(arr.map((c) => [c.id, c]));
  } catch (_) {
    return new Map();
  }
}

function clearPending(sid) {
  pendingChips.delete(sid);
  try { localStorage.removeItem(PENDING_PREFIX + sid); } catch (_) { /* ignore */ }
}

function chipStillPending(entry, sid) {
  if (state.sid === sid && state.chips.has(entry.id)) return true;
  const m = pendingChips.get(sid);
  return !!(m && m.has(entry.id));
}

function chipStatusLabel(c) {
  if (c.status === "uploading" && typeof c.progress === "number") return `${c.progress}%`;
  return FILE_STATUS_LABEL[c.status] || c.status || "";
}

function renderChips() {
  dom.chips.innerHTML = "";
  const entries = [...state.chips.values()];
  dom.chips.hidden = !entries.length;
  for (const c of entries) {
    const chip = el("div", "chip");
    const ico = el("span", "chip-ico"); ico.textContent = c.kind === "table" ? "📊" : "📄";
    const name = el("span", "chip-name"); name.textContent = c.name;
    name.setAttribute("data-tip", c.name);
    const st = el("span", "chip-status");
    st.textContent = chipStatusLabel(c);
    if (c.status === "done" || c.status === "active") st.classList.add("ok");
    else if (c.status === "failed") st.classList.add("err");
    else if (c.status === "pending_confirm") {
      st.classList.add("warn", "action");
      st.setAttribute("data-tip", "点击确认列角色");
      st.addEventListener("click", () => {
        const d = c.doc_id && docsById.get(c.doc_id);
        if (d) openSchemaModal(d);
      });
    } else st.classList.add("warn");
    if (c.error) chip.setAttribute("data-tip", c.error);

    const x = el("button", "chip-x"); x.type = "button"; x.textContent = "×";
    x.setAttribute("data-tip", "移除附件");
    x.setAttribute("aria-label", `移除附件 ${c.name}`);
    x.addEventListener("click", () => removeChip(c.id));
    chip.append(ico, name, st, x);
    dom.chips.appendChild(chip);
  }
}

function uploadFiles(files) {
  const sidAtUpload = state.sid;
  for (const f of files) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    const entry = {
      id: "tmp-" + genId(),
      doc_id: null,
      name: f.name,
      size: f.size,
      kind: [".csv", ".xlsx", ".xls"].includes("." + ext) ? "table" : "pdf",
      status: "uploading",
      error: "",
    };
    state.chips.set(entry.id, entry);
    uploadDoc(f, entry, sidAtUpload);
  }
  renderChips();
}

function uploadDoc(file, entry, sidAtUpload) {
  uploadBusy += 1;
  dom.upload.disabled = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/documents/upload");
  let lastProgRender = 0;
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    entry.progress = Math.round((e.loaded / e.total) * 100);
    const now = Date.now();
    if (now - lastProgRender > 120) {   // 节流：避免高频全量重绘 chips
      lastProgRender = now;
      renderChips();
    }
  };
  xhr.onload = () => {
    let j = {};
    try { j = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* ignore */ }
    if (xhr.status >= 400 || !j.ok) {
      entry.status = "failed";
      entry.error = j.error || `上传失败（HTTP ${xhr.status}）`;
      entry.progress = null;
      renderChips();
      toast(entry.error, "err");
    } else {
      entry.doc_id = j.doc_id;
      entry.kind = j.kind || entry.kind;
      entry.status = "processing";
      entry.progress = null;
      docPollDelay = 3000;   // 新上传：重置轮询退避
      renderChips();
      persistChips(sidAtUpload);
      if (chipStillPending(entry, sidAtUpload)) recordAttachment(entry, sidAtUpload);
      toast("已上传《" + (j.doc_name || entry.name) + "》，正在解析…");
      loadDocuments();
    }
    finishUpload();
  };
  xhr.onerror = () => {
    entry.status = "failed";
    entry.error = "上传失败：网络错误";
    entry.progress = null;
    renderChips();
    toast(entry.error, "err");
    finishUpload();
  };

  function finishUpload() {
    uploadBusy -= 1;
    dom.upload.disabled = uploadBusy > 0;
    dom.file.value = "";
  }

  const fd = new FormData();
  fd.append("file", file);
  xhr.send(fd);
}

/* 把上传事件写进当前会话历史（后端持久化，刷新/切会话不丢） */
async function recordAttachment(entry, sid) {
  try {
    const res = await fetch(`/api/sessions/${sid}/attachment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: entry.doc_id }),
    });
    const j = await res.json().catch(() => ({}));
    if (res.ok && j.ok && state.sid === sid) updateUrl(sid);
  } catch (e) {
    console.error("记录附件失败", e);
  }
}

async function removeChip(id) {
  const c = state.chips.get(id);
  if (!c) return;
  if (c.doc_id && c.status !== "processing" && c.status !== "uploading") {
    const ok = await confirmAsk({
      title: "删除文档",
      text: `确定删除「${c.name}」？将同时删除上传文件和索引向量，不可恢复。`,
      okText: "删除", danger: true,
    });
    if (!ok) return;
    try {
      const res = await fetch(`/api/documents/${c.doc_id}`, { method: "DELETE" });
      const j = await res.json().catch(() => ({}));
      if (!j.ok) { toast(j.error || "删除失败", "err"); return; }
      removeAttachmentTurns(c.doc_id);
      loadDocuments();
    } catch (e) { toast("删除失败：" + e.message, "err"); return; }
  } else if (c.status === "processing") {
    toast("文档仍在后台解析，可在资料库中管理");
  } else {
    toast("已移除附件");
  }
  state.chips.delete(id);
  renderChips();
  persistChips(state.sid);
}

dom.attach.addEventListener("click", () => dom.file.click());
dom.file.addEventListener("change", () => uploadFiles(dom.file.files));

/* 拖拽上传到输入框 */
const composerBox = document.querySelector(".composer-inner");
["dragenter", "dragover"].forEach((evt) => composerBox.addEventListener(evt, (e) => {
  e.preventDefault();
  composerBox.classList.add("drag-over");
}));
["dragleave", "drop"].forEach((evt) => composerBox.addEventListener(evt, (e) => {
  e.preventDefault();
  composerBox.classList.remove("drag-over");
}));
composerBox.addEventListener("drop", (e) => {
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) uploadFiles(files);
});

/* ══════════════ 资料库抽屉 ══════════════ */
function openLibrary(focusDataset) {
  rememberFocus();
  dom.library.hidden = false;
}
function closeLibrary() { dom.library.hidden = true; restoreFocus(); }

dom.btnLibrary.addEventListener("click", () => openLibrary());
dom.upload.addEventListener("click", () => dom.file.click());

/* 移动端侧栏抽屉 */
function openSidebar() {
  dom.app.classList.add("side-open");
  dom.sideOverlay.hidden = false;
}
function closeSidebar() {
  dom.app.classList.remove("side-open");
  dom.sideOverlay.hidden = true;
}
dom.sideOverlay.addEventListener("click", closeSidebar);
window.addEventListener("resize", () => {
  if (window.innerWidth > 900) closeSidebar();
});

/* ══════════════ 文档列表 ══════════════ */
let docsById = new Map();
let docPollTimer = null;
let docPollDelay = 3000;

async function loadDocuments() {
  try {
    const list = await (await fetch("/api/documents")).json();
    docsById = new Map((list || []).map((d) => [d.doc_id, d]));
    renderDocuments(list || []);
    syncChips(list || []);
    const pending = (list || []).some(
      (d) => d.status === "processing" || d.status === "pending");
    if (pending) {
      if (!docPollTimer) docPollTimer = setTimeout(schedulePoll, docPollDelay);
    } else if (docPollTimer) {
      clearTimeout(docPollTimer); docPollTimer = null;
      docPollDelay = 3000;
    }
  } catch (e) { console.error(e); }
}

/* 轮询退避：3s → 5s → 8s → 13s → 21s → 30s 封顶 */
function schedulePoll() {
  docPollTimer = null;
  docPollDelay = Math.min(Math.round(docPollDelay * 1.6), 30000);
  loadDocuments();
}

function syncChips(list) {
  const server = new Map(list.map((d) => [d.doc_id, d]));
  let changed = false;
  for (const c of state.chips.values()) {
    if (!c.doc_id) continue;
    const d = server.get(c.doc_id);
    if (d) {
      const ns = d.status === "done" ? "done" : d.status;
      if (ns !== c.status) { c.status = ns; changed = true; }
      if (d.kind) c.kind = d.kind;
      c.error = d.error || "";
    }
  }
  if (changed) renderChips();
}

function badgeForDoc(d) {
  const badge = el("span", `doc-badge ${d.status || ""}`);
  if (d.kind === "table") {
    if (d.status === "pending_confirm") {
      badge.textContent = "待确认 schema";
    } else if (d.status === "active") {
      const caps = [d.supports_timeline ? "时间线" : null, d.supports_recommendation ? "推荐" : null]
        .filter(Boolean).join("/") || "仅检索";
      badge.textContent = `${d.rows || 0} 行`;
      badge.setAttribute("data-tip", `已启用 · 支持：${caps}`);
    } else if (d.status === "failed") {
      badge.textContent = "失败"; badge.setAttribute("data-tip", d.error || "");
    } else {
      badge.textContent = "解析中…";
    }
  } else if (d.status === "done") {
    const chunks = d.text_chunks || 0;
    const imgs = d.image_pages || 0;
    if (chunks && imgs) badge.textContent = `${chunks} 片段 · ${imgs} 图`;
    else if (imgs) badge.textContent = `${imgs} 整页图`;
    else badge.textContent = `${chunks} 片段`;
    badge.setAttribute(
      "data-tip",
      `${d.pages || 0} 页 · 路由 ${JSON.stringify(d.route_distribution || {})}`);
  } else if (d.status === "failed") {
    badge.textContent = "失败"; badge.setAttribute("data-tip", d.error || "");
  } else {
    badge.textContent = "解析中…";
  }
  return badge;
}

function renderDocuments(list) {
  dom.docCount.textContent = list.length ? `(${list.length})` : "";
  dom.docList.innerHTML = "";
  if (!list.length) {
    const e = el("div", "doc-empty"); e.textContent = "暂无上传文档";
    dom.docList.appendChild(e); return;
  }
  for (const d of list) {
    const item = el("div", "doc-item");
    const ico = el("span", "doc-ico");
    ico.textContent = d.kind === "table" ? "📊" : "📄";
    const name = el("div", "doc-name");
    name.textContent = d.doc_name || "未命名";
    name.setAttribute("data-tip", d.doc_name || "");
    const badge = badgeForDoc(d);
    if (d.kind === "table" && d.status === "pending_confirm") {
      badge.classList.add("actionable");
      badge.onclick = () => openSchemaModal(d);
    }
    const actions = el("div", "doc-actions");
    if (d.kind === "table" && d.status === "pending_confirm") {
      const okBtn = el("button", "doc-confirm"); okBtn.type = "button";
      okBtn.textContent = "确认";
      okBtn.addEventListener("click", () => openSchemaModal(d));
      actions.appendChild(okBtn);
    }
    if (d.status !== "processing" && d.status !== "pending") {
      const delBtn = el("button", "doc-delete"); delBtn.type = "button";
      delBtn.textContent = "×"; delBtn.setAttribute("data-tip", "删除文档及关联向量");
      delBtn.setAttribute("aria-label", `删除文档 ${d.doc_name || ""}`);
      delBtn.addEventListener("click", () => deleteDocument(d));
      actions.appendChild(delBtn);
    }
    item.append(ico, name, badge, actions);
    dom.docList.appendChild(item);
  }
}

async function deleteDocument(d) {
  const ok = await confirmAsk({
    title: "删除文档",
    text: `确定删除「${d.doc_name || "未命名文档"}」？\n将同时删除上传文件和索引向量，不可恢复。`,
    okText: "删除", danger: true,
  });
  if (!ok) return;
  try {
    const res = await fetch(`/api/documents/${d.doc_id}`, { method: "DELETE" });
    const j = await res.json();
    if (!j.ok) { toast(j.error || "删除失败", "err"); return; }
    toast("已删除文档");
    removeAttachmentTurns(d.doc_id);
    for (const [id, c] of state.chips) if (c.doc_id === d.doc_id) state.chips.delete(id);
    renderChips();
    loadDocuments();
  } catch (e) { toast("删除失败：" + e.message, "err"); }
}

/* 删除文档后，把当前对话里对应的附件卡片一并移除 */
function removeAttachmentTurns(docId) {
  dom.chat.querySelectorAll(".file-card.attach-note").forEach((card) => {
    if (card.dataset.docId === docId) {
      const turn = card.closest(".turn");
      if (turn) turn.remove();
    }
  });
}

/* ══════════════ schema 确认弹窗 ══════════════ */
let schemaDocId = null;

function fillRoleSelect(select, columns, proposed, allowEmpty) {
  select.innerHTML = "";
  if (allowEmpty) {
    const none = el("option"); none.value = ""; none.textContent = "（无）";
    select.appendChild(none);
  }
  for (const c of columns) {
    const opt = el("option"); opt.value = c; opt.textContent = c;
    select.appendChild(opt);
  }
  select.value = proposed || "";
  if (select.value !== (proposed || "")) select.selectedIndex = 0;
}

function openSchemaModal(d) {
  rememberFocus();
  schemaDocId = d.doc_id;
  const cols = d.columns || [];
  const p = d.proposed_schema || {};
  dom.schemaMeta.textContent =
    `${d.doc_name} · ${d.sheet_name ? `子表「${d.sheet_name}」 · ` : ""}${d.rows || 0} 行 × ${d.cols || cols.length} 列`;
  fillRoleSelect(dom.schemaEntity, cols, p.entity_col, false);
  fillRoleSelect(dom.schemaAxis, cols, p.group_axis_col, true);
  fillRoleSelect(dom.schemaDesc, cols, p.description_col, true);
  fillRoleSelect(dom.schemaImage, cols, p.image_col, true);
  dom.schemaDisplay.value = p.display_name || (d.doc_name || "").replace(/\.[^.]+$/, "");
  dom.schemaReason.textContent = p.reasoning ? `推断依据：${p.reasoning}` : "";
  dom.schemaError.hidden = true;
  dom.schemaModal.hidden = false;
  setTimeout(() => dom.schemaEntity.focus(), 60);
}

function closeSchemaModal() {
  dom.schemaModal.hidden = true;
  schemaDocId = null;
  restoreFocus();
}

dom.schemaCancel.addEventListener("click", closeSchemaModal);
dom.schemaOk.addEventListener("click", async () => {
  if (!schemaDocId) return;
  dom.schemaOk.disabled = true;
  try {
    const res = await fetch(`/api/documents/${schemaDocId}/schema`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        entity_col: dom.schemaEntity.value,
        group_axis_col: dom.schemaAxis.value || null,
        description_col: dom.schemaDesc.value || null,
        image_col: dom.schemaImage.value || null,
        display_name: dom.schemaDisplay.value,
      }),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok || !j.ok) {
      dom.schemaError.textContent = j.error || `确认失败（HTTP ${res.status}）`;
      dom.schemaError.hidden = false;
      return;
    }
    closeSchemaModal();
    toast("数据源已启用");
    loadDocuments();
  } catch (e) {
    dom.schemaError.textContent = "确认失败：" + e.message;
    dom.schemaError.hidden = false;
  } finally {
    dom.schemaOk.disabled = false;
  }
});

/* ══════════════ 通用确认弹窗 ══════════════ */
let confirmResolve = null;

function confirmAsk({ title, text, okText = "确认", danger = false }) {
  return new Promise((resolve) => {
    rememberFocus();
    confirmResolve = resolve;
    dom.confirmTitle.textContent = title;
    dom.confirmText.textContent = text;
    dom.confirmOk.textContent = okText;
    dom.confirmOk.classList.toggle("danger", danger);
    dom.confirmModal.hidden = false;
    setTimeout(() => dom.confirmOk.focus(), 60);
  });
}

function closeConfirm() {
  dom.confirmModal.hidden = true;
  if (confirmResolve) { confirmResolve(false); confirmResolve = null; }
  restoreFocus();
}
dom.confirmOk.addEventListener("click", () => {
  dom.confirmModal.hidden = true;
  if (confirmResolve) { confirmResolve(true); confirmResolve = null; }
  restoreFocus();
});
dom.confirmCancel.addEventListener("click", closeConfirm);

/* ══════════════ Toast ══════════════ */
let toastTimer = null;
function toast(msg, type = "") {
  dom.toast.textContent = msg;
  dom.toast.className = "toast show" + (type ? " " + type : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { dom.toast.className = "toast"; }, 3400);
}

/* ══════════════ 自定义悬停提示（替代原生 title，风格统一） ══════════════ */
const tip = el("div", "tooltip");
tip.hidden = true;
document.body.appendChild(tip);
let tipTimer = null;
let tipTarget = null;

function showTip(target) {
  const text = target.getAttribute("data-tip") || "";
  if (!text) return;
  tip.textContent = text;
  tip.hidden = false;
  tip.classList.remove("show");
  const r = target.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  let x = r.left + r.width / 2 - t.width / 2;
  let y = r.top - t.height - 8;
  if (y < 6) y = r.bottom + 8;
  x = Math.max(8, Math.min(x, window.innerWidth - t.width - 8));
  tip.style.left = x + "px";
  tip.style.top = y + "px";
  requestAnimationFrame(() => tip.classList.add("show"));
}

function hideTip() {
  clearTimeout(tipTimer);
  tip.hidden = true;
  tip.classList.remove("show");
  tipTarget = null;
}

document.addEventListener("pointerover", (e) => {
  const t = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (!t || t.disabled) { hideTip(); return; }
  if (tipTarget === t && !tip.hidden) return;
  tipTarget = t;
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => showTip(t), 120);
});
document.addEventListener("pointerout", (e) => {
  const from = e.target.closest ? e.target.closest("[data-tip]") : null;
  const to = e.relatedTarget && e.relatedTarget.closest
    ? e.relatedTarget.closest("[data-tip]") : null;
  if (from && from !== to) hideTip();
});
document.addEventListener("focusin", (e) => {
  const t = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (t && !t.disabled) { clearTimeout(tipTimer); showTip(t); }
});
document.addEventListener("focusout", hideTip);
window.addEventListener("scroll", hideTip, true);
window.addEventListener("resize", hideTip);

/* ══════════════ 全局：遮罩关闭 / Esc ══════════════ */
document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => {
  const t = b.dataset.close;
  if (t === "library") closeLibrary();
  else if (t === "schema") closeSchemaModal();
  else if (t === "confirm") closeConfirm();
}));

/* ── 弹窗焦点管理：记忆焦点 + Tab 圈闭（键盘可达性） ── */
let lastFocusEl = null;
function rememberFocus() { lastFocusEl = document.activeElement; }
function restoreFocus() {
  if (lastFocusEl && lastFocusEl.focus) {
    try { lastFocusEl.focus(); } catch (_) { /* ignore */ }
  }
  lastFocusEl = null;
}
function activeModalContainer() {
  if (!dom.confirmModal.hidden) return dom.confirmModal;
  if (!dom.schemaModal.hidden) return dom.schemaModal;
  if (!dom.memoryPanel.hidden) return dom.memoryPanel;
  if (!dom.library.hidden) return dom.library;
  return null;
}
function trapTab(e, container) {
  const focusables = container.querySelectorAll(
    'button:not([disabled]), [href], input:not([disabled]), ' +
    'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])');
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    const fbPanel = dom.chat.querySelector(".fb-panel");
    if (fbPanel) { fbPanel.remove(); return; }
    if (!dom.confirmModal.hidden) closeConfirm();
    else if (!dom.schemaModal.hidden) closeSchemaModal();
    else if (!dom.library.hidden) closeLibrary();
    else if (!dom.memoryPanel.hidden) closeMemoryPanel();
    else if (dom.app.classList.contains("side-open")) closeSidebar();
    return;
  }
  if (e.key === "Tab") {
    const container = activeModalContainer();
    if (container) trapTab(e, container);
  }
});

/* 搜索时拉取全部分页（最多 500 条），避免只搜到已加载的 50 条 */
async function loadAllSessions() {
  let offset = sessionsOffset;
  const cap = 500;
  while (offset < sessionsTotal && offset < cap) {
    const data = await (await fetch(
      `/api/sessions?offset=${offset}&limit=100`)).json();
    for (const s of (data.items || [])) sessionsMap.set(s.session_id, s);
    sessionsTotal = data.total || sessionsMap.size;
    offset += (data.items || []).length;
    if (!data.has_more) break;
  }
}

dom.histFilter.addEventListener("input", async () => {
  const q = (dom.histFilter.value || "").trim().toLowerCase();
  if (!q) { loadSessions(); return; }
  await loadAllSessions();
  const list = [...sessionsMap.values()].filter(
    (s) => (s.title || "").toLowerCase().includes(q));
  renderHistory(list, false);
  markActive(state.sid);
});

/* ══════════════ 深色模式 ══════════════ */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("artagent.theme", theme); } catch (_) { /* ignore */ }
}
function initTheme() {
  let t = null;
  try { t = localStorage.getItem("artagent.theme"); } catch (_) { /* ignore */ }
  if (!t) {
    t = (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
      ? "dark" : "light";
  }
  applyTheme(t);
}
$("#btn-theme").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(cur);
});
initTheme();

/* ══════════════ 侧栏拖拽调宽（桌面端，宽度持久化） ══════════════ */
const SIDEBAR_MIN = 220, SIDEBAR_MAX = 480;
const SIDEBAR_W_KEY = "artagent.sidebar.width";
const SIDEBAR_COLLAPSED_KEY = "artagent.sidebar.collapsed";
function applySidebarWidth(px) {
  const w = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, Math.round(px)));
  dom.sidebar.style.width = w + "px";
  try { localStorage.setItem(SIDEBAR_W_KEY, String(w)); } catch (_) { /* 忽略 */ }
}
(function initSidebarWidth() {
  try {
    const saved = Number(localStorage.getItem(SIDEBAR_W_KEY));
    if (saved >= SIDEBAR_MIN && saved <= SIDEBAR_MAX) dom.sidebar.style.width = saved + "px";
  } catch (_) { /* 忽略 */ }
})();
if (dom.resizer) {
  dom.resizer.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dom.resizer.setPointerCapture(e.pointerId);
    dom.resizer.classList.add("dragging");
    const startX = e.clientX;
    const startW = dom.sidebar.getBoundingClientRect().width;
    const move = (ev) => applySidebarWidth(startW + ev.clientX - startX);
    const up = () => {
      dom.resizer.classList.remove("dragging");
      dom.resizer.removeEventListener("pointermove", move);
    };
    dom.resizer.addEventListener("pointermove", move);
    dom.resizer.addEventListener("pointerup", up, { once: true });
    dom.resizer.addEventListener("pointercancel", up, { once: true });
  });
}
if (dom.sideCollapse) {
  const syncCollapse = () => {
    const collapsed = dom.app.classList.contains("side-collapsed");
    dom.sideCollapse.setAttribute("aria-label", collapsed ? "展开侧边栏" : "收起侧边栏");
    return collapsed;
  };
  dom.sideCollapse.addEventListener("click", () => {
    const collapsed = dom.app.classList.toggle("side-collapsed");
    syncCollapse();
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch (_) { /* 忽略 */ }
  });
  try {
    if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1") {
      dom.app.classList.add("side-collapsed");
    }
  } catch (_) { /* 忽略 */ }
  syncCollapse();
}

/* ══════════════ 启动 ══════════════ */
async function init() {
  if (window.marked) marked.setOptions({ breaks: true, gfm: true });
  bootstrap();
  loadSessions();
  loadDocuments();
  const urlSid = currentSidFromUrl();
  if (urlSid) {
    await openSession(urlSid);   // 刷新后停留在原对话
  } else {
    showWelcome();
  }
}
init();
