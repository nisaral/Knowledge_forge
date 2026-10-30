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
    localStorage.setItem("kf-theme", next);
  });

  /* ─── Sidebar (mobile drawer) ─────────────────────────────────────────── */
  const sidebar = $("#sidebar");
  const scrim = $("#scrim");
  const openSidebar = () => { sidebar.classList.add("open"); scrim.hidden = false; };
  const closeSidebar = () => { sidebar.classList.remove("open"); scrim.hidden = true; };
  $("#open-sidebar").addEventListener("click", openSidebar);
  $("#close-sidebar").addEventListener("click", closeSidebar);
  scrim.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });

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
    } catch (err) {
      toast(err.message, "error");
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
      localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(-MAX_STORED_MESSAGES)));
    } catch { /* storage full or disabled */ }
  }

  function citationsBlock(sources) {
    if (!sources || !sources.length) return null;
    const details = el("details", { class: "citations" },
      el("summary", {}, `${sources.length} source passage${sources.length === 1 ? "" : "s"}`));
    for (const s of sources) {
      details.append(el("div", { class: "cite-item", "data-n": s.n },
        el("span", { class: "cite-n" }, `[${s.n}]`),
        el("div", {},
          el("div", { class: "cite-src" }, displayName(s.source)),
          el("div", { class: "cite-snip" }, `${s.snippet}${s.snippet.length >= 320 ? "…" : ""}`))));
    }
    return details;
  }

  function messageNode(m) {
    if (m.role === "user") {
      return el("div", { class: "msg user" }, el("div", { class: "bubble" }, el("div", { class: "md", html: renderMarkdown(m.content) })));
    }
    const bubble = el("div", { class: "bubble" },
      el("div", { class: "md", html: renderMarkdown(m.content, { citations: !!(m.sources && m.sources.length) }) }));
    const cites = citationsBlock(m.sources);
    if (cites) bubble.append(cites);
    if (!m.error) {
      bubble.append(el("div", { class: "bubble-actions" },
        el("button", {
          class: "icon-btn", title: "Copy answer", "aria-label": "Copy answer",
          onclick: () => navigator.clipboard?.writeText(m.content).then(() => toast("Copied to clipboard.", "success", 1800)),
        }, icon("copy"))));
    }
    bubble.addEventListener("click", (e) => {
      const btn = e.target.closest(".cite");
      if (!btn || !cites) return;
      cites.open = true;
      const item = cites.querySelector(`.cite-item[data-n="${btn.dataset.cite}"]`);
      if (item) { item.classList.remove("flash"); void item.offsetWidth; item.classList.add("flash"); item.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
    });
    return el("div", { class: `msg bot${m.error ? " error" : ""}` }, el("div", { class: "avatar", "aria-hidden": "true" }, "KF"), bubble);
  }

  function renderChat() {
    const log = $("#chat-log");
    log.replaceChildren(...state.history.map(messageNode));
    $("#chat-empty").classList.toggle("hidden", state.history.length > 0);
    scrollChat();
  }
  const scrollChat = () => { const s = $("#chat-scroll"); s.scrollTop = s.scrollHeight; };

  async function sendMessage(text) {
    if (state.busy || !text.trim()) return;
    state.busy = true;
    const history = state.history.filter((m) => !m.error).map(({ role, content }) => ({ role, content }));
    state.history.push({ role: "user", content: text.trim() });
    saveHistory();
    renderChat();
    const typing = el("div", { class: "msg bot" }, el("div", { class: "avatar" }, "KF"),
      el("div", { class: "bubble" }, el("span", { class: "typing", "aria-label": "Thinking" }, el("i"), el("i"), el("i"))));
    $("#chat-log").append(typing);
    scrollChat();
    $("#chat-send").disabled = true;
    try {
      const data = await api("/api/chat", { method: "POST", body: { message: text.trim(), history, source: state.scope } });
      state.history.push({ role: "assistant", content: data.answer, sources: data.sources || [] });
    } catch (err) {
      state.history.push({ role: "assistant", content: err.message, error: true });
    } finally {
      state.busy = false;
      $("#chat-send").disabled = false;
      saveHistory();
      renderChat();
    }
  }

  const chatInput = $("#chat-input");
  const autoGrow = () => { chatInput.style.height = "auto"; chatInput.style.height = `${Math.min(chatInput.scrollHeight, 200)}px`; };
  chatInput.addEventListener("input", autoGrow);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = chatInput.value;
    if (!text.trim()) return;
    chatInput.value = "";
    autoGrow();
    sendMessage(text);
  });
  $$("#suggestions .chip").forEach((c) => c.addEventListener("click", () => sendMessage(c.dataset.prompt)));
  $("#chat-clear").addEventListener("click", () => {
    if (!state.history.length) return;
    state.history = [];
    saveHistory();
    renderChat();
    api("/api/chat/clear", { method: "POST" }).catch(() => {});
    toast("Conversation cleared.", "info", 1800);
  });

  /* ─── Study tools ─────────────────────────────────────────────────────── */
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "default" });
  }
  let mermaidSeq = 0;

  function resultShell(title, extra) {
    const out = $("#tool-result");
    out.hidden = false;
    out.replaceChildren(el("div", { class: "result-head" }, el("h2", {}, title), extra || null));
    return out;
  }

  function skeleton(title) {
    const out = resultShell(title);
    for (const w of [92, 78, 85, 60]) out.append(el("div", { class: "skeleton", style: `width:${w}%` }));
    out.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function renderMermaid(container, code) {
    const wrap = el("div", { class: "mermaid-wrap" });
    container.append(wrap);
    if (!window.mermaid) { wrap.append(el("pre", {}, code)); return; }
    try {
      const { svg } = await window.mermaid.render(`kf-mmd-${++mermaidSeq}`, code);
      wrap.innerHTML = svg; // produced by mermaid with securityLevel "strict"
    } catch {
      wrap.replaceChildren(el("p", { class: "placeholder" }, "The diagram could not be rendered. Raw definition:"), el("pre", {}, code));
    }
  }

  function renderFlashcards(container, cards) {
    const grid = el("div", { class: "flash-grid" });
    cards.forEach((c, i) => {
      const card = el("button", { class: "flashcard", type: "button", "aria-label": `Flashcard ${i + 1}, press to flip` },
        el("div", { class: "flashcard-inner" },
          el("div", { class: "face front" }, el("small", {}, `Card ${i + 1} · question`), String(c.front ?? c.question ?? "")),
          el("div", { class: "face back" }, el("small", {}, "Answer"), String(c.back ?? c.answer ?? ""))));
      card.addEventListener("click", () => card.classList.toggle("flipped"));
      grid.append(card);
    });
    container.append(el("p", { class: "muted", style: "margin-bottom:10px" }, "Click a card to flip it."), grid);
  }

  function renderStoryboard(container, scenes) {
    const grid = el("div", { class: "story" });
    scenes.forEach((s, i) => grid.append(el("div", { class: "scene" },
      el("div", { class: "n" }, `Scene ${i + 1}`),
      el("b", {}, String(s.title ?? s.heading ?? "")),
      el("div", {}, String(s.scene ?? s.description ?? s.text ?? "")))));
    container.append(grid);
  }

  async function runTool(tool, card) {
    const topic = $("#tool-topic").value.trim();
    const needsTopic = card.dataset.needsTopic !== "false";
    if (needsTopic && !topic) {
      toast("Enter a topic first.", "error");
      $("#tool-topic").focus();
      return;
    }
    if (!state.sources.length) { toast("Add a source first — tools work on your knowledge base.", "error"); return; }
    $$(".tool-card").forEach((c) => { c.disabled = true; });
    card.classList.add("busy");
    const label = card.querySelector("strong").textContent;
    skeleton(`Generating ${label.toLowerCase()}…`);
    try {
      const data = tool === "summary"
        ? await api("/api/summarize", { method: "POST", body: { source: state.scope } })
        : await api("/api/generate-tool-content", { method: "POST", body: { tool, topic, source: state.scope } });
      const copyText = typeof data.content === "string" ? data.content : data.mermaid_code;
      const out = resultShell(data.title || label, copyText ? el("button", {
        class: "btn", onclick: () => navigator.clipboard?.writeText(copyText).then(() => toast("Copied.", "success", 1500)),
      }, icon("copy"), "Copy") : null);
      if (data.mermaid_code) await renderMermaid(out, data.mermaid_code);
      else if (tool === "flashcards") renderFlashcards(out, data.content || []);
      else if (tool === "storyboard") renderStoryboard(out, data.content || []);
      else out.append(el("div", { class: "md", html: renderMarkdown(data.content) }));
    } catch (err) {
      resultShell(label).append(el("p", { class: "placeholder" }, err.message));
      toast(err.message, "error");
    } finally {
      $$(".tool-card").forEach((c) => { c.disabled = false; });
      card.classList.remove("busy");
    }
  }
  $$(".tool-card").forEach((c) => c.addEventListener("click", () => runTool(c.dataset.tool, c)));
  $("#tool-topic").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); const first = $('.tool-card[data-tool="mock_test"]'); runTool(first.dataset.tool, first); }
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

  /* ─── Boot ────────────────────────────────────────────────────────────── */
  window.KF = { renderMarkdown, escapeHtml }; // exposed for debugging/tests
  renderChat();
  refreshSources();
})();
