/* KnowledgeForge front end — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const HISTORY_KEY = "kf-chat-v1";
  const MAX_STORED_MESSAGES = 60;

  const state = {
    sources: [],
    scope: null,            // selected source name or null for all
    history: loadHistory(), // [{role, content, sources?}]
    busy: false,
    file: null,
  };

  /* ─── Utilities ───────────────────────────────────────────────────────── */
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v; // only ever used with sanitized markup
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  const icon = (name) => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  };

  /** Minimal, safe Markdown: HTML is escaped first, then a small subset is rendered. */
  function renderMarkdown(src, { citations = false } = {}) {
    const text = String(src ?? "").replace(/\r\n?/g, "\n");
    const blocks = [];
    const codeStore = [];
    let body = text.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => {
      codeStore.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return `\u0000${codeStore.length - 1}\u0000`;
    });
    body = escapeHtml(body);

    const inline = (s) => {
      let out = s
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
          '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      if (citations) {
        out = out.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (_, nums) =>
          nums.split(/\s*,\s*/).map((n) => `<button type="button" class="cite" data-cite="${n}">${n}</button>`).join(""));
      }
      return out;
    };

    const lines = body.split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const code = line.match(/^\u0000(\d+)\u0000$/);
      if (code) { blocks.push(codeStore[+code[1]]); i++; continue; }
      if (!line.trim()) { i++; continue; }
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { blocks.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
      if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) { blocks.push("<hr>"); i++; continue; }
      if (/^&gt;\s?/.test(line)) {
        const q = [];
        while (i < lines.length && /^&gt;\s?/.test(lines[i])) q.push(lines[i++].replace(/^&gt;\s?/, ""));
        blocks.push(`<blockquote>${inline(q.join("<br>"))}</blockquote>`);
        continue;
      }
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const row = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
        const head = row(line);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(row(lines[i++]));
        blocks.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
          rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      const listRe = /^\s*(?:([-*+])|(\d+)[.)])\s+(.*)$/;
      if (listRe.test(line)) {
        const ordered = !!line.match(listRe)[2];
        const items = [];
        while (i < lines.length && listRe.test(lines[i])) {
          items.push(`<li>${inline(lines[i].match(listRe)[3])}</li>`);
          i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !listRe.test(lines[i])) {
            items[items.length - 1] = items[items.length - 1].replace(/<\/li>$/, ` ${inline(lines[i].trim())}</li>`);
            i++;
          }
        }
        blocks.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
        continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*+]\s|\s*\d+[.)]\s|&gt;|\u0000)/.test(lines[i])) {
        para.push(lines[i++]);
      }
      if (!para.length) para.push(lines[i++]);
      blocks.push(`<p>${inline(para.join("<br>"))}</p>`);
    }
    return blocks.join("");
  }

  /* ─── Toasts ──────────────────────────────────────────────────────────── */
  function toast(message, type = "info", ms = 4200) {
    const t = el("div", { class: `toast ${type}`, role: type === "error" ? "alert" : "status" },
      el("span", { class: "msg-text" }, message),
      el("button", { class: "icon-btn tiny", "aria-label": "Dismiss", onclick: () => dismiss() }, icon("x")));
    $("#toasts").append(t);
    let timer = setTimeout(dismiss, ms);
    function dismiss() {
      clearTimeout(timer);
      t.classList.add("out");
      setTimeout(() => t.remove(), 250);
    }
  }

  /* ─── API ─────────────────────────────────────────────────────────────── */
  async function api(path, { method = "GET", body, form } = {}) {
    const opts = { method, headers: {} };
    if (form) opts.body = form;
    else if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(path, opts);
    } catch {
      throw new Error("Can't reach the server. Is it running?");
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok || (data && data.success === false)) {
      throw new Error((data && data.message) || `Request failed (${res.status}).`);
    }
    return data || {};
  }

  function setLoading(btn, on) {
    if (!btn) return;
    btn.classList.toggle("loading", on);
    btn.disabled = on;
  }

  /* ─── Theme ───────────────────────────────────────────────────────────── */
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("kf-theme", next); } catch { /* storage disabled */ }
    $("#theme-toggle").setAttribute("aria-label", next === "dark" ? "Switch to light theme" : "Switch to dark theme");
    if (typeof initMermaid === "function") initMermaid();
  });

  /* ─── Sidebar (mobile drawer) ─────────────────────────────────────────── */
  const sidebar = $("#sidebar");
  const scrim = $("#scrim");
  const openSidebar = () => { sidebar.classList.add("open"); scrim.hidden = false; };
  const closeSidebar = () => { sidebar.classList.remove("open"); scrim.hidden = true; };
  $("#open-sidebar").addEventListener("click", openSidebar);
  $("#close-sidebar").addEventListener("click", closeSidebar);
  scrim.addEventListener("click", closeSidebar);


  /* ─── Views ───────────────────────────────────────────────────────────── */
  function showView(name) {
    $$(".tab").forEach((t) => {
      const on = t.dataset.view === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on);
    });
    $$("[data-view-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.viewPane !== name));
    if (name === "chat") $("#chat-input").focus();
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

  /* ─── Add-source forms ────────────────────────────────────────────────── */
  $$(".seg").forEach((b) => b.addEventListener("click", () => {
    $$(".seg").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    $$(".add-form").forEach((f) => f.classList.toggle("hidden", f.dataset.pane !== b.dataset.add));
  }));

  const isYouTube = (url) => /^(https?:\/\/)?((www|m|music)\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)\//i.test(url);

  async function addSource(payload, btn) {
    setLoading(btn, true);
    try {
      const data = await api("/api/add-content", { method: "POST", body: payload });
      toast(data.message, data.chunks_added ? "success" : "info");
      await refreshSources();
      return true;
    } catch (err) {
      toast(err.message, "error", 7000);
      return false;
    } finally {
      setLoading(btn, false);
    }
  }

  $("#form-link").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#link-input");
    const url = input.value.trim();
    if (!url) return;
    const ok = await addSource({ source: url, source_type: isYouTube(url) ? "youtube" : "web" }, e.submitter || $("#form-link button"));
    if (ok) input.value = "";
  });

  $("#form-text").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#text-input").value.trim();
    if (text.length < 20) { toast("Please paste at least a couple of sentences.", "error"); return; }
    const ok = await addSource({ source: text, source_type: "text", title: $("#text-title").value.trim() || null },
      e.submitter || $("#form-text button"));
    if (ok) { $("#text-input").value = ""; $("#text-title").value = ""; }
  });

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  function pickFile(file) {
    state.file = file || null;
    $("#dz-sub").textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB` : $("#dz-sub").dataset.default;
    $("#upload-btn").disabled = !file;
  }
  $("#dz-sub").dataset.default = $("#dz-sub").textContent;
  fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

  $("#form-file").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.file) return;
    const btn = $("#upload-btn");
    const form = new FormData();
    form.append("file", state.file);
    setLoading(btn, true);
    toast(`Processing ${state.file.name}… media files can take a minute.`, "info");
    try {
      const data = await api("/api/upload-file", { method: "POST", form });
      toast(data.message, data.chunks_added ? "success" : "info");
      fileInput.value = "";
      pickFile(null);
      await refreshSources();
    } catch (err) {
      toast(err.message, "error", 7000);
    } finally {
      setLoading(btn, false);
      btn.disabled = !state.file;
    }
  });

  /* ─── Sources ─────────────────────────────────────────────────────────── */
  const TYPE_ICON = { web: "globe", youtube: "play", text: "text", pdf: "file", audio: "file", video: "play", image: "file" };

  function displayName(src) {
    try {
      if (/^https?:\/\//.test(src)) {
        const u = new URL(src);
        const host = u.hostname.replace(/^www\./, "");
        if (/youtube\.com$/.test(host) && u.searchParams.get("v")) return `YouTube · ${u.searchParams.get("v")}`;
        return host + (u.pathname !== "/" ? u.pathname : "");
      }
    } catch { /* fall through */ }
    return src;
  }

  function renderSources() {
    const list = $("#source-list");
    list.replaceChildren();
    for (const s of state.sources) {
      const selected = state.scope === s.source;
      const li = el("li", {
        class: `source${selected ? " selected" : ""}`, tabindex: "0", role: "button",
        "aria-pressed": selected, title: s.source,
        onclick: () => setScope(selected ? null : s.source),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setScope(selected ? null : s.source); } },
      },
      el("span", { class: "source-icon" }, icon(TYPE_ICON[s.type] || "file")),
      el("span", { class: "source-meta" },
        el("div", { class: "source-name" }, displayName(s.source)),
        el("div", { class: "source-sub" }, `${s.type} · ${s.chunks} chunk${s.chunks === 1 ? "" : "s"}`)),
      el("button", {
        class: "icon-btn danger", "aria-label": `Remove ${s.source}`, title: "Remove source",
        onclick: (e) => { e.stopPropagation(); deleteSource(s.source); },
      }, icon("trash")));
      list.append(li);
    }
    const n = state.sources.length;
    $("#source-count").textContent = n;
    $("#sources-empty").hidden = n > 0;
    $("#sources-empty").style.display = n ? "none" : "";
    $("#clear-all").hidden = n === 0;
    const scopeText = state.scope ? displayName(state.scope) : null;
    $("#scope-hint").innerHTML = scopeText
      ? `Focused on <strong>${escapeHtml(scopeText)}</strong>. Click it again to use all sources.`
      : "Answers use <strong>all sources</strong>. Select one to focus.";
    $("#scope-pill").hidden = !scopeText;
    $("#scope-pill-text").textContent = scopeText ? `Scope: ${scopeText}` : "";
  }

  function setScope(source) {
    state.scope = source;
    renderSources();
  }
  $("#scope-clear").addEventListener("click", () => setScope(null));

  async function refreshSources() {
    try {
      const data = await api("/api/get-sources");
      state.sources = data.sources || [];
      if (state.scope && !state.sources.some((s) => s.source === state.scope)) state.scope = null;
      const chunks = data.stats?.chunks ?? state.sources.reduce((a, s) => a + s.chunks, 0);
      $("#kb-stats").textContent = `${chunks} chunk${chunks === 1 ? "" : "s"} · ${data.stats?.backend || "local"} store`;
      $("#key-banner").hidden = data.gemini_configured !== false;
      renderSources();
      renderSuggestions();
    } catch (err) {
      toast(err.message, "error");
      $("#sources-empty").hidden = false;
      $("#sources-empty").style.display = "";
    } finally {
      $$(".source-skel").forEach((n) => n.remove());
      $("#source-list").removeAttribute("aria-busy");
    }
  }

  async function deleteSource(source) {
    if (!confirm(`Remove "${displayName(source)}" from your knowledge base?`)) return;
    try {
      const data = await api("/api/delete-source", { method: "POST", body: { source } });
      toast(data.message, "success");
      await refreshSources();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  $("#clear-all").addEventListener("click", async () => {
    if (!confirm("Remove every source and its embeddings? This cannot be undone.")) return;
    try {
      await api("/api/clear-data", { method: "POST" });
      toast("Knowledge base cleared.", "success");
      await refreshSources();
    } catch (err) {
      toast(err.message, "error");
    }
  });

  /* ─── Chat ────────────────────────────────────────────────────────────── */
  function loadHistory() {
    try {
      const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(h) ? h.filter((m) => m && typeof m.content === "string") : [];
    } catch { return []; }
  }
  function saveHistory() {
    try {
      const keep = state.history.filter((m) => !m.pending).slice(-MAX_STORED_MESSAGES);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(keep));
    } catch { /* storage full or disabled */ }
  }

  const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
  const stamp = (ts) => (ts ? el("time", { class: "stamp", datetime: new Date(ts).toISOString() }, timeFmt.format(ts)) : null);

  function citationsBlock(sources) {
    if (!sources || !sources.length) return null;
    const cited = sources.filter((s) => s.cited).length;
    const label = cited
      ? `${cited} cited · ${sources.length} retrieved passage${sources.length === 1 ? "" : "s"}`
      : `${sources.length} retrieved passage${sources.length === 1 ? "" : "s"}`;
    const details = el("details", { class: "citations" }, el("summary", {}, label));
    const grid = el("div", { class: "cite-grid" });
    for (const s of sources) {
      grid.append(el("div", { class: `cite-card${s.cited === false && cited ? " dim" : ""}`, "data-n": s.n },
        el("div", { class: "cite-card-head" },
          el("span", { class: "cite-n" }, s.n),
          el("span", { class: "cite-type" }, icon(TYPE_ICON[s.source_type] || "file")),
          el("span", { class: "cite-src", title: s.source }, displayName(s.source))),
        el("div", { class: "cite-snip" }, s.snippet)));
    }
    details.append(grid);
    return details;
  }

  /* Hover/focus preview for inline [n] citations. */
  const pop = $("#cite-pop");
  let popFor = null;
  function showCitePop(btn, source) {
    if (!source) return;
    popFor = btn;
    pop.replaceChildren(
      el("div", { class: "cite-pop-src" }, `[${source.n}] ${displayName(source.source)}`),
      el("div", { class: "cite-pop-snip" }, source.snippet));
    pop.hidden = false;
    const r = btn.getBoundingClientRect();
    const w = Math.min(340, window.innerWidth - 24);
    pop.style.width = `${w}px`;
    pop.style.left = `${Math.max(12, Math.min(r.left + r.width / 2 - w / 2, window.innerWidth - w - 12))}px`;
    const below = r.bottom + 8 + pop.offsetHeight < window.innerHeight;
    pop.style.top = `${below ? r.bottom + 8 : r.top - pop.offsetHeight - 8}px`;
    btn.setAttribute("aria-describedby", "cite-pop");
  }
  function hideCitePop() {
    if (popFor) popFor.removeAttribute("aria-describedby");
    popFor = null;
    pop.hidden = true;
  }

  function wireCitations(bubble, m, cites) {
    const find = (btn) => (m.sources || []).find((s) => String(s.n) === btn.dataset.cite);
    bubble.addEventListener("mouseover", (e) => { const b = e.target.closest(".cite"); if (b) showCitePop(b, find(b)); });
    bubble.addEventListener("mouseout", (e) => { if (e.target.closest(".cite")) hideCitePop(); });
    bubble.addEventListener("focusin", (e) => { const b = e.target.closest(".cite"); if (b) showCitePop(b, find(b)); });
    bubble.addEventListener("focusout", hideCitePop);
    bubble.addEventListener("click", (e) => {
      const btn = e.target.closest(".cite");
      if (!btn || !cites) return;
      cites.open = true;
      const item = cites.querySelector(`.cite-card[data-n="${btn.dataset.cite}"]`);
      if (item) { item.classList.remove("flash"); void item.offsetWidth; item.classList.add("flash"); item.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
    });
  }

  function copyText(text, what = "Copied to clipboard.") {
    if (!navigator.clipboard) { toast("Clipboard is not available in this browser.", "error"); return; }
    navigator.clipboard.writeText(text).then(() => toast(what, "success", 1600), () => toast("Copy failed.", "error"));
  }

  function messageNode(m, index) {
    if (m.role === "user") {
      return el("div", { class: "msg user" },
        el("div", { class: "bubble" }, el("div", { class: "md", html: renderMarkdown(m.content) })),
        el("div", { class: "meta" }, stamp(m.ts)));
    }
    const body = el("div", { class: "md", html: m.pending && !m.content ? "" : renderMarkdown(m.content, { citations: !!(m.sources && m.sources.length) }) });
    const bubble = el("div", { class: "bubble" }, body);
    if (m.pending && !m.content) {
      bubble.append(el("span", { class: "typing", role: "status", "aria-label": "Thinking" }, el("i"), el("i"), el("i")));
    }
    if (m.pending && m.content) body.classList.add("streaming");
    const cites = m.pending ? null : citationsBlock(m.sources);
    if (cites) bubble.append(cites);
    const isLast = index === state.history.length - 1;
    const meta = el("div", { class: "meta" }, stamp(m.ts),
      !m.pending && !m.error && m.sources && m.sources.length && m.grounded === false
        ? el("span", { class: "badge warn", title: "The answer did not cite any of your passages" }, "Not from your sources") : null,
      m.stopped ? el("span", { class: "badge" }, "Stopped") : null);
    if (!m.pending) {
      const actions = el("div", { class: "bubble-actions" });
      if (!m.error) {
        actions.append(el("button", { class: "icon-btn", title: "Copy answer", "aria-label": "Copy answer", onclick: () => copyText(m.content) }, icon("copy")));
      }
      if (isLast && !state.busy) {
        actions.append(el("button", { class: "icon-btn", title: m.error ? "Retry" : "Regenerate", "aria-label": m.error ? "Retry" : "Regenerate answer", onclick: regenerate }, icon("refresh")));
      }
      meta.append(actions);
    }
    wireCitations(bubble, m, cites);
    return el("div", { class: `msg bot${m.error ? " error" : ""}` },
      el("div", { class: "avatar", "aria-hidden": "true" }, "KF"),
      el("div", { class: "msg-col" }, bubble, meta));
  }

  /* Smart auto-scroll: follow the stream only while the reader is at the bottom. */
  const scroller = $("#chat-scroll");
  let stick = true;
  const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  scroller.addEventListener("scroll", () => {
    stick = nearBottom();
    $("#jump-latest").hidden = stick || !state.history.length;
  }, { passive: true });
  const scrollChat = (force = false) => {
    if (force || stick) { scroller.scrollTop = scroller.scrollHeight; stick = true; $("#jump-latest").hidden = true; }
    else $("#jump-latest").hidden = false;
  };
  $("#jump-latest").addEventListener("click", () => { scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); });

  function renderChat(force = false) {
    const log = $("#chat-log");
    log.replaceChildren(...state.history.map(messageNode));
    $("#chat-empty").classList.toggle("hidden", state.history.length > 0);
    scrollChat(force);
  }

  /* Only the streaming bubble is re-rendered per token (throttled to one paint per frame). */
  let paintQueued = false;
  function paintPending() {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => {
      paintQueued = false;
      const log = $("#chat-log");
      const i = state.history.length - 1;
      if (i < 0 || !log.lastElementChild) return;
      log.lastElementChild.replaceWith(messageNode(state.history[i], i));
      scrollChat();
    });
  }

  async function* sseEvents(res) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        try { yield { event, data: JSON.parse(data.join("\n") || "{}") }; } catch { /* ignore malformed */ }
      }
    }
  }

  let controller = null;
  function setBusy(on) {
    state.busy = on;
    $("#chat-send").hidden = on;
    $("#chat-stop").hidden = !on;
  }

  async function streamAnswer(message, history) {
    const reply = { role: "assistant", content: "", sources: [], pending: true, ts: Date.now() };
    state.history.push(reply);
    renderChat(true);
    controller = new AbortController();
    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message, history, source: state.scope }),
        signal: controller.signal,
      });
      if (!res.ok || !(res.headers.get("content-type") || "").includes("text/event-stream")) {
        let data = null;
        try { data = await res.json(); } catch { /* not JSON */ }
        throw new Error((data && data.message) || `Request failed (${res.status}).`);
      }
      for await (const { event, data } of sseEvents(res)) {
        if (event === "meta") reply.sources = data.sources || [];
        else if (event === "token") { reply.content += data.text || ""; paintPending(); }
        else if (event === "done") { reply.content = data.answer ?? reply.content; reply.sources = data.sources || reply.sources; reply.grounded = data.grounded; }
        else if (event === "error") throw new Error(data.message || "The answer stream failed.");
      }
      if (!reply.content.trim()) throw new Error("The model returned an empty answer. Try rephrasing.");
    } catch (err) {
      if (err.name === "AbortError") {
        reply.stopped = true;
        if (!reply.content) { reply.content = "Stopped before an answer arrived."; reply.error = true; }
      } else {
        const msg = err instanceof TypeError ? "Can't reach the server. Is it running?" : err.message;
        if (reply.content) { reply.content += `\n\n> ${msg}`; reply.stopped = true; } else { reply.content = msg; reply.error = true; }
      }
    } finally {
      controller = null;
      delete reply.pending;
      setBusy(false);
      saveHistory();
      renderChat();
    }
  }

  function historyForApi(upto = state.history.length) {
    return state.history.slice(0, upto).filter((m) => !m.error && !m.pending).map(({ role, content }) => ({ role, content }));
  }

  async function sendMessage(text) {
    text = text.trim();
    if (state.busy || !text) return;
    if (!state.sources.length && !confirm("Your knowledge base is empty, so answers won't be grounded in your sources. Ask anyway?")) return;
    setBusy(true);
    const history = historyForApi();
    state.history.push({ role: "user", content: text, ts: Date.now() });
    saveHistory();
    await streamAnswer(text, history);
  }

  function regenerate() {
    if (state.busy) return;
    let i = state.history.length - 1;
    while (i >= 0 && state.history[i].role !== "user") i--;
    if (i < 0) return;
    const question = state.history[i].content;
    state.history = state.history.slice(0, i + 1);
    setBusy(true);
    streamAnswer(question, historyForApi(i));
  }

  $("#chat-stop").addEventListener("click", () => controller?.abort());

  const chatInput = $("#chat-input");
  const autoGrow = () => { chatInput.style.height = "auto"; chatInput.style.height = `${Math.min(chatInput.scrollHeight, 200)}px`; };
  chatInput.addEventListener("input", autoGrow);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#chat-form").requestSubmit(); }
    if (e.key === "ArrowUp" && !chatInput.value) {
      const last = [...state.history].reverse().find((m) => m.role === "user");
      if (last) { e.preventDefault(); chatInput.value = last.content; autoGrow(); }
    }
  });
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = chatInput.value;
    if (!text.trim() || state.busy) return;
    chatInput.value = "";
    autoGrow();
    sendMessage(text);
  });
  $("#suggestions").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) sendMessage(c.dataset.prompt); });

  function renderSuggestions() {
    const box = $("#suggestions");
    const base = [
      ["Overview of key ideas", "Give me a short overview of the key ideas."],
      ["Explain the hardest concept", "Explain the hardest concept here in simple terms."],
      ["Self-check questions", "What are 5 questions I should be able to answer after studying this?"],
    ];
    const recent = state.sources.slice(-2).reverse().map((s) => {
      const name = displayName(s.source).replace(/^Note: /, "").slice(0, 48);
      return [`Summarise “${name}”`, `Summarise the main points of "${name}".`];
    });
    box.replaceChildren(...[...recent, ...base].slice(0, 4).map(([label, prompt]) =>
      el("button", { class: "chip", type: "button", "data-prompt": prompt }, icon("spark"), label)));
    $("#chat-empty p").textContent = state.sources.length
      ? `Answers are grounded in your ${state.sources.length} source${state.sources.length === 1 ? "" : "s"} and cite the passages they use.`
      : "Add a link, notes or a file in the sidebar, then ask questions. Answers cite the passages they use.";
  }

  $("#chat-clear").addEventListener("click", () => {
    if (!state.history.length || state.busy) return;
    if (!confirm("Clear this conversation?")) return;
    state.history = [];
    saveHistory();
    renderChat(true);
    toast("Conversation cleared.", "info", 1800);
  });

  /* ─── Study tools ─────────────────────────────────────────────────────── */
  const mermaidTheme = () => (document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "default");
  function initMermaid() {
    if (window.mermaid) window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: mermaidTheme() });
  }
  initMermaid();
  let mermaidSeq = 0;
  let lastTool = null;

  function resultShell(title, ...extra) {
    const out = $("#tool-result");
    out.hidden = false;
    out.replaceChildren(el("div", { class: "result-head" }, el("h2", {}, title), el("div", { class: "result-actions" }, ...extra)));
    return out;
  }

  function skeleton(title, tool) {
    const out = resultShell(title);
    if (tool === "flashcards") out.append(el("div", { class: "skeleton", style: "height:220px;max-width:560px;margin:10px auto" }));
    else if (tool === "flowchart" || tool === "mindmap") out.append(el("div", { class: "skeleton", style: "height:300px" }));
    else for (const w of [92, 78, 85, 60, 70]) out.append(el("div", { class: "skeleton", style: `width:${w}%` }));
    out.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function download(name, blob) {
    const a = el("a", { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  const slug = (s) => String(s || "diagram").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "diagram";

  /* Diagram viewer: wheel/pinch zoom, drag to pan, fit, SVG/PNG export. */
  async function renderMermaid(container, code, title) {
    const stage = el("div", { class: "diagram", tabindex: "0", "aria-label": "Diagram. Use plus and minus to zoom, arrow keys to pan." });
    const canvas = el("div", { class: "diagram-canvas" });
    stage.append(canvas);
    const view = { x: 0, y: 0, k: 1 };
    const apply = () => { canvas.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`; zoomLabel.textContent = `${Math.round(view.k * 100)}%`; };
    const zoom = (f, cx = stage.clientWidth / 2, cy = stage.clientHeight / 2) => {
      const k = Math.min(4, Math.max(0.2, view.k * f));
      view.x = cx - ((cx - view.x) * k) / view.k;
      view.y = cy - ((cy - view.y) * k) / view.k;
      view.k = k;
      apply();
    };
    const fit = () => {
      const svg = canvas.querySelector("svg");
      if (!svg) return;
      const w = svg.viewBox?.baseVal?.width || svg.getBBox().width || 1;
      const h = svg.viewBox?.baseVal?.height || svg.getBBox().height || 1;
      svg.setAttribute("width", w); svg.setAttribute("height", h);
      svg.style.maxWidth = "none";
      const k = Math.min((stage.clientWidth - 24) / w, (stage.clientHeight - 24) / h, 1.5);
      view.k = k; view.x = (stage.clientWidth - w * k) / 2; view.y = (stage.clientHeight - h * k) / 2;
      apply();
    };
    const zoomLabel = el("span", { class: "zoom-label", "aria-live": "polite" }, "100%");
    const svgText = () => { const s = canvas.querySelector("svg"); return s ? new XMLSerializer().serializeToString(s) : ""; };
    const toolbar = el("div", { class: "diagram-tools", role: "toolbar", "aria-label": "Diagram controls" },
      el("button", { class: "icon-btn", type: "button", title: "Zoom out", "aria-label": "Zoom out", onclick: () => zoom(1 / 1.2) }, icon("zoom-out")),
      zoomLabel,
      el("button", { class: "icon-btn", type: "button", title: "Zoom in", "aria-label": "Zoom in", onclick: () => zoom(1.2) }, icon("zoom-in")),
      el("button", { class: "icon-btn", type: "button", title: "Fit to view", "aria-label": "Fit to view", onclick: fit }, icon("fit")),
      el("span", { class: "sep" }),
      el("button", { class: "btn small", type: "button", onclick: () => download(`${slug(title)}.svg`, new Blob([svgText()], { type: "image/svg+xml" })) }, icon("download"), "SVG"),
      el("button", { class: "btn small", type: "button", onclick: () => exportPng(canvas.querySelector("svg"), title) }, icon("download"), "PNG"),
      el("button", { class: "btn small", type: "button", onclick: () => copyText(code, "Mermaid code copied.") }, icon("copy"), "Code"));
    container.append(toolbar, stage);
    if (!window.mermaid) { canvas.append(el("pre", {}, code)); return; }
    try {
      const { svg } = await window.mermaid.render(`kf-mmd-${++mermaidSeq}`, code);
      canvas.innerHTML = svg; // produced by mermaid with securityLevel "strict"
      requestAnimationFrame(fit);
    } catch {
      toolbar.remove();
      stage.replaceWith(el("div", { class: "state-box" }, icon("alert"),
        el("p", {}, "The diagram could not be rendered. Try regenerating, or copy the raw definition:"), el("pre", {}, code)));
      return;
    }
    let drag = null;
    stage.addEventListener("pointerdown", (e) => { drag = { x: e.clientX - view.x, y: e.clientY - view.y }; stage.setPointerCapture(e.pointerId); stage.classList.add("grabbing"); });
    stage.addEventListener("pointermove", (e) => { if (drag) { view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; apply(); } });
    const end = () => { drag = null; stage.classList.remove("grabbing"); };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    stage.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    stage.addEventListener("keydown", (e) => {
      const step = 40;
      const moves = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
      if (moves[e.key]) { e.preventDefault(); view.x += moves[e.key][0]; view.y += moves[e.key][1]; apply(); }
      else if (e.key === "+" || e.key === "=") zoom(1.2);
      else if (e.key === "-") zoom(1 / 1.2);
      else if (e.key === "0") fit();
    });
  }

  function exportPng(svg, title) {
    if (!svg) return;
    const w = svg.viewBox?.baseVal?.width || svg.clientWidth;
    const h = svg.viewBox?.baseVal?.height || svg.clientHeight;
    const clone = svg.cloneNode(true);
    clone.setAttribute("width", w); clone.setAttribute("height", h);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const img = new Image();
    img.onload = () => {
      const scale = 2;
      const c = document.createElement("canvas");
      c.width = w * scale; c.height = h * scale;
      const ctx = c.getContext("2d");
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--surface").trim() || "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      c.toBlob((b) => (b ? download(`${slug(title)}.png`, b) : toast("PNG export failed.", "error")), "image/png");
    };
    img.onerror = () => toast("PNG export failed in this browser; try SVG.", "error");
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`;
  }

  /* Flashcards: one card at a time, flip, prev/next, shuffle, "got it" progress. */
  let activeDeck = null;
  function renderFlashcards(container, cards) {
    if (!cards.length) { container.append(emptyResult("No flashcards came back for this topic. Try a broader topic.")); return; }
    const deck = { cards: cards.map((c, i) => ({ ...c, id: i })), i: 0, known: new Set() };
    const bar = el("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(cards.length) }, el("span"));
    const counter = el("span", { class: "deck-count" });
    const card = el("button", { class: "flashcard", type: "button" },
      el("div", { class: "flashcard-inner" },
        el("div", { class: "face front" }, el("small", {}, "Question"), el("div", { class: "face-text q" })),
        el("div", { class: "face back" }, el("small", {}, "Answer"), el("div", { class: "face-text a" }))));
    const knowBtn = el("button", { class: "btn", type: "button" }, icon("check"), "Got it");
    const show = () => {
      const c = deck.cards[deck.i];
      card.classList.remove("flipped");
      $(".q", card).textContent = c.question;
      $(".a", card).textContent = c.answer;
      card.setAttribute("aria-label", `Card ${deck.i + 1} of ${deck.cards.length}. ${c.question}. Press to reveal the answer.`);
      counter.textContent = `Card ${deck.i + 1} / ${deck.cards.length} · ${deck.known.size} known`;
      bar.firstChild.style.width = `${(deck.known.size / deck.cards.length) * 100}%`;
      bar.setAttribute("aria-valuenow", String(deck.known.size));
      bar.setAttribute("aria-label", `${deck.known.size} of ${deck.cards.length} cards known`);
      knowBtn.classList.toggle("on", deck.known.has(c.id));
      knowBtn.lastChild.textContent = deck.known.has(c.id) ? "Known" : "Got it";
    };
    const flip = () => {
      card.classList.toggle("flipped");
      if (card.classList.contains("flipped")) card.setAttribute("aria-label", `Answer: ${deck.cards[deck.i].answer}`);
    };
    const go = (d) => { deck.i = (deck.i + d + deck.cards.length) % deck.cards.length; show(); };
    card.addEventListener("click", flip);
    knowBtn.addEventListener("click", () => {
      const id = deck.cards[deck.i].id;
      if (deck.known.has(id)) deck.known.delete(id); else deck.known.add(id);
      if (deck.known.size === deck.cards.length) toast("Deck complete — every card marked known. 🎉", "success");
      show();
      if (deck.known.has(id) && deck.known.size < deck.cards.length) setTimeout(() => go(1), 250);
    });
    const controls = el("div", { class: "deck-controls" },
      el("button", { class: "icon-btn", type: "button", "aria-label": "Previous card", title: "Previous (←)", onclick: () => go(-1) }, icon("left")),
      el("button", { class: "btn", type: "button", onclick: flip }, "Flip"),
      knowBtn,
      el("button", { class: "icon-btn", type: "button", "aria-label": "Shuffle", title: "Shuffle", onclick: () => {
        for (let j = deck.cards.length - 1; j > 0; j--) { const k = Math.floor(Math.random() * (j + 1)); [deck.cards[j], deck.cards[k]] = [deck.cards[k], deck.cards[j]]; }
        deck.i = 0; show();
      } }, icon("shuffle")),
      el("button", { class: "icon-btn", type: "button", "aria-label": "Next card", title: "Next (→)", onclick: () => go(1) }, icon("right")));
    activeDeck = { flip, go };
    container.append(el("div", { class: "deck" }, el("div", { class: "deck-top" }, counter, bar), card, controls));
    show();
  }

  /* Quiz: answer each question, then score with per-question review. */
  function renderQuiz(container, questions) {
    if (!questions.length) { container.append(emptyResult("No quiz questions came back for this topic. Try a broader topic.")); return; }
    const picks = new Array(questions.length).fill(null);
    let submitted = false;
    const form = el("form", { class: "quiz" });
    const scoreBox = el("div", { class: "quiz-score", role: "status", hidden: true });
    const submit = el("button", { class: "btn primary", type: "submit", disabled: true }, "Check answers");
    const status = el("span", { class: "muted" }, `0 / ${questions.length} answered`);
    const update = () => {
      const n = picks.filter((p) => p !== null).length;
      status.textContent = `${n} / ${questions.length} answered`;
      submit.disabled = n === 0 || submitted;
    };
    questions.forEach((q, qi) => {
      const fs = el("fieldset", { class: "quiz-q" }, el("legend", {}, el("span", { class: "qn" }, `Q${qi + 1}`), q.question));
      q.options.forEach((opt, oi) => {
        const id = `q${qi}-o${oi}`;
        const input = el("input", { type: "radio", name: `q${qi}`, id, value: String(oi) });
        input.addEventListener("change", () => { picks[qi] = oi; update(); });
        fs.append(el("label", { class: "opt", for: id }, input, el("span", { class: "opt-letter" }, "ABCDEFGH"[oi]), el("span", {}, opt)));
      });
      fs.append(el("p", { class: "explain", hidden: true }));
      form.append(fs);
    });
    const reset = el("button", { class: "btn", type: "button", hidden: true }, icon("refresh"), "Try again");
    reset.addEventListener("click", () => {
      submitted = false; picks.fill(null);
      $$("input", form).forEach((i) => { i.checked = false; i.disabled = false; });
      $$(".opt", form).forEach((o) => o.classList.remove("correct", "wrong"));
      $$(".explain", form).forEach((p) => { p.hidden = true; });
      scoreBox.hidden = true; reset.hidden = true; update();
      form.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    form.append(el("div", { class: "quiz-foot" }, status, reset, submit));
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      submitted = true;
      let score = 0;
      questions.forEach((q, qi) => {
        const fs = form.children[qi];
        const right = picks[qi] === q.answer;
        if (right) score++;
        $$(".opt", fs).forEach((o, oi) => {
          o.classList.toggle("correct", oi === q.answer);
          o.classList.toggle("wrong", oi === picks[qi] && !right);
          $("input", o).disabled = true;
        });
        const ex = $(".explain", fs);
        ex.hidden = false;
        ex.textContent = `${right ? "✓ Correct." : picks[qi] === null ? `Skipped. Answer: ${"ABCDEFGH"[q.answer]}.` : `✗ Answer: ${"ABCDEFGH"[q.answer]}.`} ${q.explanation || ""}`;
      });
      const pct = Math.round((score / questions.length) * 100);
      scoreBox.hidden = false;
      scoreBox.className = `quiz-score ${pct >= 80 ? "good" : pct >= 50 ? "ok" : "low"}`;
      scoreBox.replaceChildren(el("strong", {}, `${score} / ${questions.length}`), el("span", {},
        pct >= 80 ? " Great work!" : pct >= 50 ? " Good effort — review the misses below." : " Keep going — the explanations below will help."));
      reset.hidden = false;
      update();
      scoreBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    container.append(scoreBox, form);
  }

  function renderStoryboard(container, scenes) {
    if (!scenes.length) { container.append(emptyResult("No scenes came back for this topic.")); return; }
    const grid = el("ol", { class: "story" });
    scenes.forEach((s, i) => grid.append(el("li", { class: "scene" },
      el("div", { class: "n" }, `Scene ${i + 1}`),
      el("b", {}, String(s.title ?? "")),
      el("div", {}, String(s.scene ?? "")))));
    container.append(grid);
  }

  const emptyResult = (msg) => el("div", { class: "state-box" }, icon("alert"), el("p", {}, msg));

  function sourcesFooter(sources) {
    if (!sources || !sources.length) return null;
    const names = [...new Set(sources.map((s) => displayName(s.source)))];
    return el("p", { class: "result-sources" }, `Based on ${sources.length} passage${sources.length === 1 ? "" : "s"} from: ${names.join(", ")}`);
  }

  async function runTool(tool, card) {
    const topic = $("#tool-topic").value.trim();
    const needsTopic = card.dataset.needsTopic !== "false";
    if (needsTopic && !topic) {
      toast("Enter a topic first.", "error");
      $("#tool-topic").focus();
      $("#tool-topic").classList.add("invalid");
      return;
    }
    if (!state.sources.length) { toast("Add a source first — tools work on your knowledge base.", "error"); return; }
    $$(".tool-card").forEach((c) => { c.disabled = true; c.classList.toggle("selected", c === card); });
    card.classList.add("busy");
    card.setAttribute("aria-busy", "true");
    activeDeck = null;
    lastTool = { tool, card };
    const label = card.querySelector("strong").textContent;
    skeleton(`Generating ${label.toLowerCase()}…`, tool);
    try {
      const data = tool === "summary"
        ? await api("/api/summarize", { method: "POST", body: { source: state.scope } })
        : await api("/api/generate-tool-content", { method: "POST", body: { tool, topic, source: state.scope } });
      const title = data.title || label;
      const extras = [el("button", { class: "btn small", type: "button", onclick: () => runTool(tool, card) }, icon("refresh"), "Regenerate")];
      if (typeof data.content === "string") extras.unshift(el("button", { class: "btn small", type: "button", onclick: () => copyText(data.content) }, icon("copy"), "Copy"));
      const out = resultShell(title, ...extras);
      if (data.mermaid_code) await renderMermaid(out, data.mermaid_code, title);
      else if (tool === "flashcards") renderFlashcards(out, data.content || []);
      else if (tool === "quiz") renderQuiz(out, data.content || []);
      else if (tool === "storyboard") renderStoryboard(out, data.content || []);
      else out.append(el("div", { class: "md", html: renderMarkdown(data.content) }));
      const foot = sourcesFooter(data.sources);
      if (foot) out.append(foot);
    } catch (err) {
      resultShell(label, el("button", { class: "btn small", type: "button", onclick: () => runTool(tool, card) }, icon("refresh"), "Retry"))
        .append(emptyResult(err.message));
    } finally {
      $$(".tool-card").forEach((c) => { c.disabled = false; });
      card.classList.remove("busy");
      card.removeAttribute("aria-busy");
    }
  }
  $$(".tool-card").forEach((c) => c.addEventListener("click", () => runTool(c.dataset.tool, c)));
  $("#tool-topic").addEventListener("input", (e) => e.target.classList.remove("invalid"));
  $("#tool-topic").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const target = lastTool && lastTool.tool !== "summary" ? lastTool.card : $('.tool-card[data-tool="quiz"]');
      runTool(target.dataset.tool, target);
    }
  });

  /* ─── Insights ────────────────────────────────────────────────────────── */
  $("#btn-entities").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const out = $("#entities-out");
    setLoading(btn, true);
    out.replaceChildren(el("div", { class: "skeleton", style: "width:70%" }), el("div", { class: "skeleton", style: "width:50%" }));
    try {
      const data = await api("/api/extract-entities", { method: "POST", body: { source: state.scope } });
      const groups = Object.entries(data.entities || {}).filter(([, v]) => v.length);
      out.replaceChildren(...(groups.length ? groups.map(([cat, items]) => el("div", { class: "ent-group" },
        el("h4", {}, cat), el("div", { class: "tags" }, items.map((x) => el("span", { class: "tag" }, x)))))
        : [el("p", { class: "placeholder" }, "No entities found.")]));
    } catch (err) {
      out.replaceChildren(el("p", { class: "placeholder" }, err.message));
    } finally {
      setLoading(btn, false);
    }
  });

  $("#btn-clusters").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const out = $("#clusters-out");
    setLoading(btn, true);
    out.replaceChildren(el("div", { class: "skeleton", style: "width:100%;height:220px" }));
    try {
      const data = await api("/api/cluster-topics");
      out.replaceChildren(
        el("img", { class: "cluster-img", src: data.image, alt: `Topic clusters: ${(data.labels || []).join(", ")}` }),
        el("div", { class: "tags", style: "margin-top:12px" }, (data.labels || []).map((l) => el("span", { class: "tag" }, l))));
    } catch (err) {
      out.replaceChildren(el("p", { class: "placeholder" }, err.message));
    } finally {
      setLoading(btn, false);
    }
  });

  /* ─── Keyboard shortcuts ──────────────────────────────────────────────── */
  const shortcuts = $("#shortcuts");
  $("#shortcuts-btn").addEventListener("click", () => shortcuts.showModal());
  $("#shortcuts-close").addEventListener("click", () => shortcuts.close());
  shortcuts.addEventListener("click", (e) => { if (e.target === shortcuts) shortcuts.close(); });
  const typing = (t) => t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (controller) controller.abort();
      closeSidebar();
      hideCitePop();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      $('.seg[data-add="link"]')?.click();
      if (window.innerWidth < 900) openSidebar();
      $("#link-input").focus();
      return;
    }
    if (e.altKey && ["1", "2", "3"].includes(e.key)) {
      e.preventDefault();
      showView(["chat", "tools", "insights"][+e.key - 1]);
      return;
    }
    if (typing(e.target) || e.ctrlKey || e.metaKey || e.altKey || shortcuts.open) return;
    if (e.key === "/") { e.preventDefault(); showView("chat"); chatInput.focus(); }
    else if (e.key === "?") { e.preventDefault(); shortcuts.showModal(); }
    else if (activeDeck && !$("#view-tools").classList.contains("hidden")) {
      if (e.key === " " && !e.target.closest("button")) { e.preventDefault(); activeDeck.flip(); }
      else if (e.key === "ArrowRight") activeDeck.go(1);
      else if (e.key === "ArrowLeft") activeDeck.go(-1);
    }
  });

  /* ─── Boot ────────────────────────────────────────────────────────────── */
  window.KF = { renderMarkdown, escapeHtml }; // exposed for debugging/tests
  renderChat();
  refreshSources();
})();
