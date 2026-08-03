/* ══════════════════════════════════════════════════════════════════
   ArtAgent 前端逻辑（原生 JS，无构建）—— v2 布局重构
   · 侧栏：新建对话 + 历史搜索/管理
   · 资料库抽屉：上传、文档状态、数据源切换
   · 输入区：paperclip 附件 → chip（上传/解析/待确认状态），随消息发送
   · SSE 流式对话（POST + ReadableStream）
   ══════════════════════════════════════════════════════════════════ */
const $  = (s) => document.querySelector(s);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const genId = () => (crypto.randomUUID
  ? crypto.randomUUID()
  : Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10));

const LOGO = "/static/emblem.svg";

const state = {
  sid: genId(),
  streaming: false,
  title: "新对话",
  chips: new Map(),        // chip.id -> { id, doc_id, name, size, kind, status, error }
  abortCtrl: null,         // 当前流的 AbortController（停止生成用）
  pendingRegenerate: false, // 下次发送时替换最后一条用户消息（编辑/重新生成）
};

/* 每个会话的待发送附件（切会话不丢失；localStorage 兜底刷新） */
const PENDING_PREFIX = "artagent.pending.v1.";
let pendingChips = new Map();   // sid -> Map(chipId -> entry)

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
  btnMenu:    $("#btn-menu"),
  history:    $("#history"),
  histFilter: $("#hist-filter"),
  tbTitle:    $("#tb-title"),
  datasetPillLabel: $("#dataset-pill-label"),
  btnLibrary: $("#btn-library"),
  btnDataset: $("#btn-dataset"),
  welcome:    $("#welcome"),
  chat:       $("#chat"),
  cards:      $("#cards"),
  chips:      $("#chips"),
  msg:        $("#msg"),
  send:       $("#btn-send"),
  attach:     $("#btn-attach"),
  file:       $("#file-input"),
  uLevel:     $("#u-level"),
  library:    $("#library"),
  docList:    $("#doc-list"),
  docCount:   $("#doc-count"),
  upload:     $("#btn-upload"),
  datasetSelect: $("#dataset-select"),
  datasetHint:   $("#dataset-hint"),
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
};

/* ══════════════ 视图切换 ══════════════ */
function showWelcome() {
  dom.welcome.classList.remove("is-hidden");
  dom.chat.classList.add("is-hidden");
  dom.chat.innerHTML = "";
  syncTitle();
}
function showChat() {
  dom.welcome.classList.add("is-hidden");
  dom.chat.classList.remove("is-hidden");
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
  const name = el("span", "fc-name"); name.textContent = entry.name; name.title = entry.name;
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
function addAssistantTurn(html) {
  const turn = el("div", "turn assistant");
  const bubble = el("div", "bubble");
  setBubbleHTML(bubble, html || "");
  turn.append(assistantAvatar(), bubble);
  wrapAndAppend(turn);
  addAssistantActions(turn, bubble);
  return bubble;
}

/* ── 消息操作：复制 / 重新生成 / 编辑（对标成熟平台） ── */
function addAssistantActions(turn, bubble) {
  const actions = el("div", "msg-actions");
  const copy = el("button", "msg-act"); copy.type = "button"; copy.textContent = "复制";
  copy.title = "复制回答";
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
  const regen = el("button", "msg-act"); regen.type = "button"; regen.textContent = "重新生成";
  regen.title = "重新生成回答";
  regen.addEventListener("click", regenerateLast);
  actions.append(copy, regen);
  turn.appendChild(actions);
}

function addUserActions(turn) {
  const actions = el("div", "msg-actions");
  const edit = el("button", "msg-act"); edit.type = "button"; edit.textContent = "编辑";
  edit.title = "编辑这条消息";
  edit.addEventListener("click", editLast);
  actions.appendChild(edit);
  turn.appendChild(actions);
}

function lastUserMessageText() {
  const users = dom.chat.querySelectorAll(".turn.user .bubble p");
  return users.length ? users[users.length - 1].textContent : "";
}

function regenerateLast() {
  if (state.streaming) return;
  const text = lastUserMessageText();
  if (!text) return;
  const assistants = dom.chat.querySelectorAll(".turn.assistant");
  const last = assistants[assistants.length - 1];
  if (last) {
    const wrap = last.closest(".chat-wrap");
    if (wrap) wrap.remove();
  }
  send(text, { regenerate: true });
}

function editLast() {
  if (state.streaming) return;
  const users = dom.chat.querySelectorAll(".turn.user");
  const lastUser = users[users.length - 1];
  if (!lastUser) return;
  const p = lastUser.querySelector(".bubble p");
  const text = p ? p.textContent : "";
  let node = lastUser.closest(".chat-wrap");
  while (node) {
    const next = node.nextElementSibling;
    node.remove();
    node = next;
  }
  dom.msg.value = text;
  autoResize();
  dom.msg.focus();
  state.pendingRegenerate = true;   // 发送后由后端替换旧问答
}
function addSystemNote(text) {
  const turn = el("div", "turn system");
  const bubble = el("div", "bubble system-note");
  bubble.textContent = text;
  turn.appendChild(bubble);
  wrapAndAppend(turn);
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
  name.title = m.doc_name || "";
  card.dataset.docId = m.doc_id || "";
  card.append(ico, name);
  bubble.appendChild(card);
  turn.appendChild(bubble);
  wrapAndAppend(turn);
}

/* 挂载气泡 HTML，并把 .md-answer 里的 Markdown 原文用 marked 渲染 */
function setBubbleHTML(bubble, html) {
  bubble.innerHTML = html;
  if (!window.marked) return;
  if (!window.__markedOpts) {
    marked.setOptions({ breaks: true, gfm: true });
    window.__markedOpts = true;
  }
  bubble.querySelectorAll(".md-answer").forEach((box) => {
    if (box.dataset.rendered) return;
    box.innerHTML = marked.parse(box.textContent || "");
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
  for (const s of sources) {
    const chip = el("span", "source-chip");
    chip.textContent = s.label || "";
    chip.title = s.label || "";
    box.appendChild(chip);
  }
  bubble.appendChild(box);
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
});

/* ══════════════ 发送（SSE 流式） ══════════════ */
async function send(text, opts = {}) {
  text = (text || "").trim();
  const hasFiles = state.chips.size > 0;
  if ((!text && !hasFiles) || state.streaming) return;
  if (opts.regenerate) state.pendingRegenerate = true;

  state.streaming = true;
  setSendStop(true);
  if (!sessionsMap.has(state.sid)) {
    state.title = (text || "上传文档").slice(0, 24) || "新对话";
    syncTitle();
  }
  showChat();
  addUserTurn(text, []);
  const sidAtSend = state.sid;
  state.chips.clear(); renderChips(); persistChips(sidAtSend);
  dom.msg.value = ""; autoResize();

  const bubble = addAssistantTurn(
    '<div class="think-box" open><summary>正在思考…</summary></div>');
  const abortCtrl = new AbortController();
  state.abortCtrl = abortCtrl;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortCtrl.signal,
      body: JSON.stringify({
        message: text || "请查阅我上传的文档并回答。",
        session_id: state.sid,
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
          if (evt.session_id) { state.sid = evt.session_id; updateUrl(state.sid); }
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
      scrollChat();
    } else {
      bubble.innerHTML = "😔 " + (err.message || "网络中断或服务未响应，请稍后重试。");
    }
    console.error(err);
  } finally {
    state.streaming = false;
    state.pendingRegenerate = false;
    state.abortCtrl = null;
    setSendStop(false);
    loadSessions();
  }
}

/* 发送/停止按钮切换 */
function setSendStop(stop) {
  dom.send.classList.toggle("stopping", stop);
  dom.send.setAttribute("aria-label", stop ? "停止生成" : "发送");
  dom.send.disabled = false;
}

function stopGeneration() {
  if (state.abortCtrl) state.abortCtrl.abort();
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
  dom.history.innerHTML = "";
  if (!list.length) {
    const e = el("div", "hist-empty"); e.textContent = "暂无历史对话";
    dom.history.appendChild(e); syncTitle(); return;
  }
  const q = (dom.histFilter.value || "").trim().toLowerCase();
  const filtered = q ? list.filter((s) => (s.title || "").toLowerCase().includes(q)) : list;
  if (!filtered.length) {
    const e = el("div", "hist-empty"); e.textContent = "无匹配的对话";
    dom.history.appendChild(e); syncTitle(); return;
  }
  for (const s of filtered) {
    const b = el("button", "hist-btn"); b.type = "button"; b.dataset.sid = s.session_id;
    if (s.session_id === state.sid) b.classList.add("active");
    const body = el("span", "h-body");
    const t = el("span", "h-title"); t.textContent = s.title || "未命名对话";
    const tm = el("span", "h-time"); tm.textContent = s.relative || "";
    body.append(t, tm);
    const del = el("button", "hist-del"); del.type = "button"; del.textContent = "×";
    del.title = "删除该对话"; del.setAttribute("aria-label", "删除对话");
    del.addEventListener("click", (ev) => { ev.stopPropagation(); deleteSession(s.session_id, s.title); });
    const ren = el("button", "hist-ren"); ren.type = "button"; ren.textContent = "✎";
    ren.title = "重命名"; ren.setAttribute("aria-label", "重命名对话");
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
  dom.tbTitle.title = dom.tbTitle.textContent;
}

async function openSession(sid) {
  if (state.streaming) return;
  stashChips();
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
  } catch (e) { console.error(e); toast("加载会话失败", "err"); }
}

function newSession() {
  if (state.streaming) return;
  stashChips();
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

async function clearSession() {
  const ok = await confirmAsk({
    title: "清除对话",
    text: "确定清除当前会话的全部消息？删除后不可恢复。",
    okText: "清除", danger: true,
  });
  if (!ok) return;
  const oldSid = state.sid;
  try { await fetch(`/api/sessions/${oldSid}`, { method: "DELETE" }); } catch (_) { /* ignore */ }
  clearPending(oldSid);
  state.chips = new Map();
  renderChips();
  await loadSessions();
  newSession();
}

/* ══════════════ 首屏：场景卡 + 记忆等级 ══════════════ */
function setLevel(memory) {
  const n = memory || 0;
  const tier = n >= 12 ? "Lv.5 鉴赏家" : n >= 8 ? "Lv.4 行家"
             : n >= 4  ? "Lv.3 学徒"   : n >= 1 ? "Lv.2 访客" : "Lv.1 初见";
  dom.uLevel.textContent = `${tier} · 已记 ${n} 项偏好`;
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
  if (state.streaming) { stopGeneration(); return; }
  send(dom.msg.value);
});
$("#btn-new").addEventListener("click", newSession);
$("#btn-clear").addEventListener("click", clearSession);

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
    const name = el("span", "chip-name"); name.textContent = c.name; name.title = c.name;
    const st = el("span", "chip-status");
    st.textContent = chipStatusLabel(c);
    if (c.status === "done" || c.status === "active") st.classList.add("ok");
    else if (c.status === "failed") st.classList.add("err");
    else if (c.status === "pending_confirm") {
      st.classList.add("warn", "action");
      st.title = "点击确认列角色";
      st.addEventListener("click", () => {
        const d = c.doc_id && docsById.get(c.doc_id);
        if (d) openSchemaModal(d);
      });
    } else st.classList.add("warn");
    if (c.error) chip.title = c.error;

    const x = el("button", "chip-x"); x.type = "button"; x.textContent = "×";
    x.title = "移除附件"; x.setAttribute("aria-label", `移除附件 ${c.name}`);
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
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      entry.progress = Math.round((e.loaded / e.total) * 100);
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
      if (state.sid === sidAtUpload) {
        addSystemNote("📎 已上传《" + (j.doc_name || entry.name) + "》");
      }
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
      loadDocuments(); loadDatasets();
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
  dom.library.hidden = false;
  if (focusDataset) setTimeout(() => dom.datasetSelect.focus(), 60);
}
function closeLibrary() { dom.library.hidden = true; }

dom.btnLibrary.addEventListener("click", () => openLibrary(false));
dom.btnDataset.addEventListener("click", () => openLibrary(true));
dom.upload.addEventListener("click", () => dom.file.click());

/* 移动端侧栏抽屉 */
function openSidebar() {
  dom.app.classList.add("side-open");
  dom.sideOverlay.hidden = false;
  dom.btnMenu.setAttribute("aria-expanded", "true");
}
function closeSidebar() {
  dom.app.classList.remove("side-open");
  dom.sideOverlay.hidden = true;
  dom.btnMenu.setAttribute("aria-expanded", "false");
}
dom.btnMenu.addEventListener("click", () =>
  dom.app.classList.contains("side-open") ? closeSidebar() : openSidebar());
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
      badge.title = `已启用 · 支持：${caps}`;
    } else if (d.status === "failed") {
      badge.textContent = "失败"; badge.title = d.error || "";
    } else {
      badge.textContent = "解析中…";
    }
  } else if (d.status === "done") {
    const chunks = d.text_chunks || 0;
    const imgs = d.image_pages || 0;
    if (chunks && imgs) badge.textContent = `${chunks} 片段 · ${imgs} 图`;
    else if (imgs) badge.textContent = `${imgs} 整页图`;
    else badge.textContent = `${chunks} 片段`;
    badge.title = `${d.pages || 0} 页 · 路由 ${JSON.stringify(d.route_distribution || {})}`;
  } else if (d.status === "failed") {
    badge.textContent = "失败"; badge.title = d.error || "";
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
    name.title = d.doc_name || "";
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
      delBtn.textContent = "×"; delBtn.title = "删除文档及关联向量";
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
    loadDatasets();
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
    loadDatasets();
  } catch (e) {
    dom.schemaError.textContent = "确认失败：" + e.message;
    dom.schemaError.hidden = false;
  } finally {
    dom.schemaOk.disabled = false;
  }
});

/* ══════════════ 数据源切换 ══════════════ */
async function loadDatasets() {
  try {
    const data = await (await fetch("/api/datasets")).json();
    dom.datasetSelect.innerHTML = "";
    for (const item of (data.items || [])) {
      const opt = el("option");
      opt.value = item.dataset_id;
      opt.textContent = item.kind === "table" ? `📊 ${item.name}` : item.name;
      dom.datasetSelect.appendChild(opt);
    }
    dom.datasetSelect.value = data.active;
    updateDatasetHint(data);
    const cur = (data.items || []).find((i) => i.dataset_id === data.active);
    dom.datasetPillLabel.textContent = cur ? cur.name : "核心库";
  } catch (e) { console.error(e); }
}

function updateDatasetHint(data) {
  const cur = (data.items || []).find((i) => i.dataset_id === data.active);
  if (!cur) { dom.datasetHint.textContent = ""; return; }
  dom.datasetHint.textContent = cur.kind === "table"
    ? `${cur.rows || 0} 行 · 支持：${[cur.supports_timeline && "时间线", cur.supports_recommendation && "推荐"]
        .filter(Boolean).join("/") || "仅检索"}`
    : "内置西方艺术核心库 · 覆盖 8–19 世纪画作";
}

dom.datasetSelect.addEventListener("change", async () => {
  try {
    const res = await fetch("/api/dataset/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: dom.datasetSelect.value }),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok || !j.ok) toast(j.error || "切换失败", "err");
    else toast("已切换数据源");
  } catch (e) { toast("切换失败：" + e.message, "err"); }
  loadDatasets();
});

/* ══════════════ 通用确认弹窗 ══════════════ */
let confirmResolve = null;

function confirmAsk({ title, text, okText = "确认", danger = false }) {
  return new Promise((resolve) => {
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
}
dom.confirmOk.addEventListener("click", () => {
  dom.confirmModal.hidden = true;
  if (confirmResolve) { confirmResolve(true); confirmResolve = null; }
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

/* ══════════════ 全局：遮罩关闭 / Esc ══════════════ */
document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => {
  const t = b.dataset.close;
  if (t === "library") closeLibrary();
  else if (t === "schema") closeSchemaModal();
  else if (t === "confirm") closeConfirm();
}));

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!dom.confirmModal.hidden) closeConfirm();
  else if (!dom.schemaModal.hidden) closeSchemaModal();
  else if (!dom.library.hidden) closeLibrary();
  else if (dom.app.classList.contains("side-open")) closeSidebar();
});

dom.histFilter.addEventListener("input", () => {
  const list = [...sessionsMap.values()];
  renderHistory(list, sessionsOffset < sessionsTotal);
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

/* ══════════════ 启动 ══════════════ */
async function init() {
  if (window.marked) marked.setOptions({ breaks: true, gfm: true });
  bootstrap();
  loadSessions();
  loadDocuments();
  loadDatasets();
  const urlSid = currentSidFromUrl();
  if (urlSid) {
    await openSession(urlSid);   // 刷新后停留在原对话
  } else {
    showWelcome();
  }
}
init();
