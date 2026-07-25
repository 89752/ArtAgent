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

/* ── 启动 ── */
if (window.marked) marked.setOptions({ breaks: true, gfm: true });
bootstrap();
loadSessions();
