/* ══════════════════════════════════════════════════════════════════
   ArtAgent 前端逻辑（原生 JS，无构建）
   · SSE 流式对话（fetch + ReadableStream，因为是 POST，不能用 EventSource）
   · 会话无状态：session_id 由前端生成，历史存后端 SQLite
   · 助手气泡 HTML 由服务层生成（可信）→ innerHTML 直挂
   ══════════════════════════════════════════════════════════════════ */
const $  = (s) => document.querySelector(s);
const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };

const LOGO = "/static/emblem.svg";
const state = { sid: crypto.randomUUID(), streaming: false };

const dom = {
  history:  $("#history"),
  welcome:  $("#welcome"),
  topbar:   $("#topbar"),
  chat:     $("#chat"),
  cards:    $("#cards"),
  msg:      $("#msg"),
  send:     $("#btn-send"),
  uLevel:   $("#u-level"),
};

/* ── 视图切换 ── */
function showWelcome() {
  dom.welcome.classList.remove("is-hidden");
  dom.chat.classList.add("is-hidden");
  dom.topbar.classList.add("is-hidden");
  dom.chat.innerHTML = "";
}
function showChat() {
  dom.welcome.classList.add("is-hidden");
  dom.chat.classList.remove("is-hidden");
  dom.topbar.classList.remove("is-hidden");
}

/* ── 消息渲染：把服务层气泡 HTML 包进 turn 结构 ── */
function assistantAvatar() {
  const a = el("div", "avatar");
  const img = el("img"); img.src = LOGO; img.alt = ""; a.appendChild(img);
  return a;
}
function userAvatar() { const a = el("div", "avatar"); a.textContent = "You"; return a; }

function addUserTurn(text) {
  const turn = el("div", "turn user");
  const bubble = el("div", "bubble");
  bubble.textContent = text;                 // 用户输入：纯文本，防注入
  turn.append(userAvatar(), bubble);
  wrapAndAppend(turn);
}
function addAssistantTurn(html) {
  const turn = el("div", "turn assistant");
  const bubble = el("div", "bubble");
  setBubbleHTML(bubble, html || "");         // 服务层生成，可信
  turn.append(assistantAvatar(), bubble);
  wrapAndAppend(turn);
  return bubble;
}

/* 挂载气泡 HTML，并把 .md-answer 里的 Markdown 原文用 marked 渲染 */
function setBubbleHTML(bubble, html) {
  bubble.innerHTML = html;
  if (!window.marked) return;
  bubble.querySelectorAll(".md-answer").forEach((box) => {
    if (box.dataset.rendered) return;
    box.innerHTML = marked.parse(box.textContent || "");
    box.dataset.rendered = "1";
  });
}
function wrapAndAppend(turn) {
  const wrap = el("div", "chat-wrap"); wrap.appendChild(turn);
  dom.chat.appendChild(wrap);
  dom.chat.scrollTop = dom.chat.scrollHeight;
}

/* ── 发送（SSE 流式）── */
async function send(text) {
  text = (text || "").trim();
  if (!text || state.streaming) return;
  state.streaming = true;
  dom.send.disabled = true;
  showChat();
  addUserTurn(text);
  dom.msg.value = ""; autoResize();

  const bubble = addAssistantTurn(
    '<div class="think-box" open><summary>正在思考…</summary></div>');

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: state.sid }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();                       // 末段可能不完整，留到下次
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "delta") {
          setBubbleHTML(bubble, evt.html);
          dom.chat.scrollTop = dom.chat.scrollHeight;
        } else if (evt.type === "done") {
          setBubbleHTML(bubble, evt.html);
          if (evt.session_id) state.sid = evt.session_id;
          if (typeof evt.memory === "number") setLevel(evt.memory);
          dom.chat.scrollTop = dom.chat.scrollHeight;
        }
      }
    }
  } catch (err) {
    bubble.innerHTML = "😔 网络中断或服务未响应，请稍后重试。";
    console.error(err);
  } finally {
    state.streaming = false;
    dom.send.disabled = false;
    loadSessions();                            // 刷新侧栏（新会话入列 / 标题更新）
  }
}

/* ── 侧栏历史 ── */
async function loadSessions() {
  try {
    const list = await (await fetch("/api/sessions")).json();
    dom.history.innerHTML = "";
    if (!list.length) {
      const e = el("div", "hist-empty"); e.textContent = "暂无历史对话";
      dom.history.appendChild(e); return;
    }
    for (const s of list) {
      const b = el("button", "hist-btn");
      if (s.session_id === state.sid) b.classList.add("active");
      const t = el("div", "h-title"); t.textContent = s.title || "未命名对话";
      const tm = el("div", "h-time"); tm.textContent = s.relative || "";
      b.append(t, tm);
      b.onclick = () => openSession(s.session_id);
      dom.history.appendChild(b);
    }
  } catch (e) { console.error(e); }
}

async function openSession(sid) {
  if (state.streaming) return;
  try {
    const data = await (await fetch(`/api/sessions/${sid}`)).json();
    state.sid = sid;
    showChat();
    dom.chat.innerHTML = "";
    for (const m of (data.messages || [])) {
      if (m.role === "user") addUserTurn(m.content);
      else addAssistantTurn(m.content);
    }
    markActive();
  } catch (e) { console.error(e); }
}

function markActive() {
  dom.history.querySelectorAll(".hist-btn").forEach((b) => b.classList.remove("active"));
}

function newSession() {
  if (state.streaming) return;
  state.sid = crypto.randomUUID();
  showWelcome();
  markActive();
}

async function clearSession() {
  if (state.streaming) return;
  try { await fetch(`/api/sessions/${state.sid}`, { method: "DELETE" }); } catch (e) {}
  await loadSessions();
  newSession();
}

/* ── 首屏：场景卡 + 记忆等级 ── */
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
      const card = el("button", "scene-card");
      if (c.thumb) {
        const img = el("img", "sc-thumb"); img.src = c.thumb; img.alt = ""; card.appendChild(img);
      }
      const body = el("div", "sc-body");
      const title = el("div", "sc-title"); title.textContent = c.text;
      body.appendChild(title);
      card.appendChild(body);
      card.onclick = () => send(c.query);
      dom.cards.appendChild(card);
    }
  } catch (e) { console.error(e); }
}

/* ── 输入框：自适应高度 + Enter 发送 ── */
function autoResize() {
  dom.msg.style.height = "auto";
  dom.msg.style.height = Math.min(dom.msg.scrollHeight, 168) + "px";
}
dom.msg.addEventListener("input", autoResize);
dom.msg.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(dom.msg.value); }
});
dom.send.addEventListener("click", () => send(dom.msg.value));
$("#btn-new").addEventListener("click", newSession);
$("#btn-clear").addEventListener("click", clearSession);

/* ── 文档上传与入库进度（Stage 3 PDF / Stage 5 表格）── */
const docDom = { upload: $("#btn-upload"), file: $("#file-input"), list: $("#doc-list") };
let docPollTimer = null;

async function loadDocuments() {
  try {
    const list = await (await fetch("/api/documents")).json();
    renderDocuments(list);
    // 有解析中的文档 → 3s 轮询；全部落定 → 停止（待确认 schema 不算解析中）
    const pending = list.some((d) => d.status === "processing" || d.status === "pending");
    if (pending && !docPollTimer) {
      docPollTimer = setInterval(loadDocuments, 3000);
    } else if (!pending && docPollTimer) {
      clearInterval(docPollTimer); docPollTimer = null;
    }
  } catch (e) { console.error(e); }
}

function renderDocuments(list) {
  docDom.list.innerHTML = "";
  if (!list.length) {
    const e = el("div", "doc-empty"); e.textContent = "暂无上传文档";
    docDom.list.appendChild(e); return;
  }
  for (const d of list) {
    const item = el("div", "doc-item");
    const name = el("div", "doc-name");
    name.textContent = (d.kind === "table" ? "📊 " : "") + (d.doc_name || "未命名");
    name.title = d.doc_name || "";
    const badge = el("span", `doc-badge ${d.status || ""}`);
    if (d.kind === "table") {
      if (d.status === "pending_confirm") {
        badge.textContent = "待确认 schema";
        badge.classList.add("actionable");
        badge.title = "点击确认/纠正列角色后启用";
        badge.onclick = () => openSchemaModal(d);
      } else if (d.status === "active") {
        const caps = [d.supports_timeline ? "时间线" : null, d.supports_recommendation ? "推荐" : null]
          .filter(Boolean).join("/") || "仅检索";
        badge.textContent = `${d.rows || 0} 行`;
        badge.title = `已启用 · 支持：${caps}`;
      } else if (d.status === "failed") {
        badge.textContent = "失败";
        badge.title = d.error || "";
      } else {
        badge.textContent = "解析中…";
      }
    } else if (d.status === "done") {
      const chunks = d.text_chunks || 0;
      const imgs = d.image_pages || 0;
      if (chunks && imgs) {
        badge.textContent = `${chunks} 片段 · ${imgs} 图`;
      } else if (imgs) {
        badge.textContent = `${imgs} 整页图`;
      } else {
        badge.textContent = `${chunks} 片段`;
      }
      badge.title = `${d.pages || 0} 页 · 路由 ${JSON.stringify(d.route_distribution || {})}`;
    } else if (d.status === "failed") {
      badge.textContent = "失败";
      badge.title = d.error || "";
    } else {
      badge.textContent = "解析中…";
    }

    const actions = el("div", "doc-actions");
    // 解析中时不允许删除，避免后台任务与清理冲突
    if (d.status !== "processing") {
      const delBtn = el("button", "doc-delete");
      delBtn.textContent = "×";
      delBtn.title = "删除文档及关联向量";
      delBtn.onclick = () => deleteDocument(d);
      actions.appendChild(delBtn);
    }

    item.append(name, badge, actions);
    docDom.list.appendChild(item);
  }
}

async function deleteDocument(d) {
  const label = d.doc_name || "未命名文档";
  if (!confirm(`确定删除「${label}」？\n将同时删除上传文件和索引向量，不可恢复。`)) return;
  try {
    const res = await fetch(`/api/documents/${d.doc_id}`, { method: "DELETE" });
    const j = await res.json();
    if (!j.ok) { alert(j.error || "删除失败"); return; }
    loadDocuments();
    loadDatasets();            // 表格数据源可能随之移除
  } catch (e) {
    alert("删除失败：" + e.message);
  }
}

async function uploadDoc(file) {
  if (!file) return;
  docDom.upload.disabled = true;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/documents/upload", { method: "POST", body: fd });
    const j = await res.json();
    if (!j.ok) { alert(j.error || "上传失败"); return; }
    loadDocuments();           // 立即出现"解析中"，随后轮询
  } catch (e) {
    alert("上传失败：" + e.message);
  } finally {
    docDom.upload.disabled = false;
    docDom.file.value = "";
  }
}

docDom.upload.addEventListener("click", () => docDom.file.click());
docDom.file.addEventListener("change", () => uploadDoc(docDom.file.files[0]));

/* ── schema 确认弹窗（Stage 5）── */
const schemaDom = {
  modal: $("#schema-modal"), meta: $("#schema-meta"), reason: $("#schema-reason"),
  entity: $("#schema-entity"), axis: $("#schema-axis"), desc: $("#schema-desc"),
  image: $("#schema-image"), display: $("#schema-display"),
  ok: $("#schema-ok"), cancel: $("#schema-cancel"),
};
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
  if (select.value !== (proposed || "")) select.selectedIndex = 0; // 推断列已不存在时兜底
}

function openSchemaModal(d) {
  schemaDocId = d.doc_id;
  const cols = d.columns || [];
  const p = d.proposed_schema || {};
  schemaDom.meta.textContent =
    `${d.doc_name} · ${d.sheet_name ? `子表「${d.sheet_name}」 · ` : ""}${d.rows || 0} 行 × ${d.cols || cols.length} 列`;
  fillRoleSelect(schemaDom.entity, cols, p.entity_col, false);
  fillRoleSelect(schemaDom.axis, cols, p.group_axis_col, true);
  fillRoleSelect(schemaDom.desc, cols, p.description_col, true);
  fillRoleSelect(schemaDom.image, cols, p.image_col, true);
  schemaDom.display.value = p.display_name || (d.doc_name || "").replace(/\.[^.]+$/, "");
  schemaDom.reason.textContent = p.reasoning ? `推断依据：${p.reasoning}` : "";
  schemaDom.modal.classList.remove("is-hidden");
}

function closeSchemaModal() {
  schemaDom.modal.classList.add("is-hidden");
  schemaDocId = null;
}

schemaDom.cancel.addEventListener("click", closeSchemaModal);
schemaDom.ok.addEventListener("click", async () => {
  if (!schemaDocId) return;
  schemaDom.ok.disabled = true;
  try {
    const res = await fetch(`/api/documents/${schemaDocId}/schema`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        entity_col: schemaDom.entity.value,
        group_axis_col: schemaDom.axis.value || null,
        description_col: schemaDom.desc.value || null,
        image_col: schemaDom.image.value || null,
        display_name: schemaDom.display.value,
      }),
    });
    const j = await res.json();
    if (!j.ok) { alert(j.error || "确认失败"); return; }
    closeSchemaModal();
    loadDocuments();
    loadDatasets();            // 新数据源入列
  } catch (e) {
    alert("确认失败：" + e.message);
  } finally {
    schemaDom.ok.disabled = false;
  }
});

/* ── 数据源切换器（Stage 5）── */
const dsDom = { select: $("#dataset-select"), hint: $("#dataset-hint") };

async function loadDatasets() {
  try {
    const data = await (await fetch("/api/datasets")).json();
    dsDom.select.innerHTML = "";
    for (const item of (data.items || [])) {
      const opt = el("option");
      opt.value = item.dataset_id;
      opt.textContent = item.kind === "table" ? `📊 ${item.name}` : item.name;
      dsDom.select.appendChild(opt);
    }
    dsDom.select.value = data.active;
    updateDatasetHint(data);
  } catch (e) { console.error(e); }
}

function updateDatasetHint(data) {
  const cur = (data.items || []).find((i) => i.dataset_id === dsDom.select.value);
  if (!cur) { dsDom.hint.textContent = ""; return; }
  dsDom.hint.textContent = cur.kind === "table"
    ? `${cur.rows || 0} 行 · 支持：${[cur.supports_timeline && "时间线", cur.supports_recommendation && "推荐"].filter(Boolean).join("/") || "仅检索"}`
    : "21,384 幅西方画作（8–19 世纪）";
}

dsDom.select.addEventListener("change", async () => {
  try {
    const res = await fetch("/api/dataset/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: dsDom.select.value }),
    });
    const j = await res.json();
    if (!j.ok) { alert(j.error || "切换失败"); }
  } catch (e) { alert("切换失败：" + e.message); }
  loadDatasets();
});

/* ── 启动 ── */
if (window.marked) marked.setOptions({ breaks: true, gfm: true });
bootstrap();
loadSessions();
loadDocuments();
loadDatasets();
