/* Create hub: storyboard (timeline, edit, reorder, regenerate, present, export) and
 * flashcards (SM-2 review saved in localStorage, Anki CSV / JSON export, citations). */
(() => {
  "use strict";
  const KF = window.KF;
  const SM2 = window.SM2;
  const { $, $$, el, icon, api, toast, setLoading, download, slug } = KF;
  const NS = "http://www.w3.org/2000/svg";

  /* Frame illustrations: small stroke icons (24x24 paths), one per backend ICONS name. */
  const ART = {
    book: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 21V5M8 7h7",
    lightbulb: "M9 18h6M10 21h4M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z",
    gear: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19 12l2-1-1-3-2 .3-1.3-1.3.3-2-3-1-1 2h-2l-1-2-3 1 .3 2L6 7.3 4 7 3 10l2 1v2l-2 1 1 3 2-.3 1.3 1.3-.3 2 3 1 1-2h2l1 2 3-1-.3-2 1.3-1.3 2 .3 1-3-2-1z",
    chart: "M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-8",
    globe: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM3 12h18M12 3c2.5 2.5 3.5 5.5 3.5 9s-1 6.5-3.5 9c-2.5-2.5-3.5-5.5-3.5-9s1-6.5 3.5-9z",
    flask: "M9 3h6M10 3v6L4.5 19A1.5 1.5 0 0 0 6 21h12a1.5 1.5 0 0 0 1.5-2L14 9V3M7 15h10",
    users: "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM2 21a7 7 0 0 1 14 0M16 3.5a4 4 0 0 1 0 7.5M22 21a7 7 0 0 0-4-6.3",
    clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
    flag: "M5 21V4M5 4h11l-2 4 2 4H5",
    target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 12h.01",
    leaf: "M5 21c0-9 6-15 15-16-1 9-7 15-15 16zM5 21l8-8",
    bolt: "M13 2 4 14h7l-1 8 9-12h-7z",
    code: "M8 8l-4 4 4 4M16 8l4 4-4 4M14 4l-4 16",
    heart: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
    shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
    star: "M12 3l2.8 5.7 6.2.9-4.5 4.4 1 6.2L12 17.3 6.5 20.2l1-6.2L3 9.6l6.2-.9z",
    question: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 1-1 1.7v.5M12 17h.01",
    check: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM8 12l3 3 5-6",
  };
  function art(name, size = 48) {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", size); svg.setAttribute("height", size);
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", "frame-art");
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", ART[name] || ART.book);
    svg.append(p);
    return svg;
  }
  const artMarkup = (name) => `<svg viewBox="0 0 24 24" width="64" height="64" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="${ART[name] || ART.book}"/></svg>`;

  let mode = "storyboard";
  let board = null;    // last storyboard response (frames mutable)
  const esc = KF.escapeHtml;

  /* ─── Scope pickers (groups) shared with the graph tab ─── */
  function groupsOf(sources) {
    const g = new Map();
    for (const s of sources) if (s.group) g.set(s.group, (g.get(s.group) || 0) + 1);
    return [...g.entries()];
  }
  KF.onSources((sources) => {
    for (const sel of [$("#create-group"), $("#graph-group")]) {
      if (!sel) continue;
      const cur = sel.value;
      const opts = [el("option", { value: "" }, "All sources")];
      const groups = groupsOf(sources);
      if (groups.length) opts.push(el("optgroup", { label: "Site groups" }, groups.map(([g, n]) => el("option", { value: `g:${g}` }, `${g} (${n} pages)`))));
      opts.push(el("optgroup", { label: "Single source" }, sources.map((s) => el("option", { value: `s:${s.source}` }, KF.displayName(s.source)))));
      sel.replaceChildren(...opts);
      if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
    }
  });
  KF.scopeBody = (value) => (!value ? {} : value.startsWith("g:") ? { groups: [value.slice(2)] } : { sources: [value.slice(2)] });

  function setMode(m) {
    mode = m;
    $$(".cseg").forEach((b) => { const on = b.dataset.create === m; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); });
    $$("#create-form [data-for]").forEach((l) => l.classList.toggle("hidden", l.dataset.for !== m));
    if (m === "flashcards") renderDecks(); else $("#decks").replaceChildren();
    if (m === "storyboard" && board) renderBoard(); else if (m === "storyboard") $("#create-out").replaceChildren();
    if (m === "flashcards") $("#create-out").replaceChildren();
  }
  $$(".cseg").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.create)));

  $("#create-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("#create-go");
    const body = { topic: $("#create-topic").value.trim() || undefined, ...KF.scopeBody($("#create-group").value) };
    const out = $("#create-out");
    out.replaceChildren(...[88, 70, 80].map((w) => el("div", { class: "skeleton", style: `width:${w}%;height:64px` })));
    setLoading(btn, true);
    try {
      if (mode === "storyboard") {
        board = await api("/api/storyboard", { method: "POST", body: { ...body, frames: +$("#sb-frames").value || 6, style: $("#sb-style").value } });
        board.scope = body;
        renderBoard();
      } else {
        const deck = await api("/api/flashcards", { method: "POST", body: { ...body, count: +$("#fc-count").value || 12, difficulty: $("#fc-difficulty").value } });
        saveDeck(deck);
        out.replaceChildren();
        renderDecks();
        startReview(deckId(deck), true);
      }
    } catch (err) {
      out.replaceChildren(el("div", { class: "state-box" }, icon("alert"), el("p", {}, err.message)));
      toast(err.message, "error", 6000);
    } finally {
      setLoading(btn, false);
    }
  });

  /* ─── Storyboard ─── */
  const total = () => board.frames.reduce((a, f) => a + (f.duration_sec || 0), 0);
  const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  function citeList(sources) {
    if (!sources || !sources.length) return null;
    return el("details", { class: "cites" }, el("summary", {}, `${sources.length} source${sources.length === 1 ? "" : "s"}`),
      el("ol", {}, sources.map((s) => el("li", {}, el("strong", {}, KF.displayName(s.source)), " — ", s.snippet))));
  }

  function frameCard(f, i) {
    const n = board.frames.length;
    const editable = (cls, key, tag = "p") => {
      const node = el(tag, { class: cls, contenteditable: "true", spellcheck: "true", "aria-label": `Frame ${i + 1} ${key}` }, f[key]);
      node.addEventListener("blur", () => { f[key] = node.textContent.trim(); });
      return node;
    };
    const points = el("ul", { class: "kp" }, f.key_points.map((p, j) => {
      const li = el("li", { contenteditable: "true", "aria-label": `Key point ${j + 1}` }, p);
      li.addEventListener("blur", () => { f.key_points[j] = li.textContent.trim(); f.key_points = f.key_points.filter(Boolean); });
      return li;
    }));
    return el("article", { class: "frame", "data-index": String(i), "aria-label": `Frame ${i + 1} of ${n}` },
      el("div", { class: "frame-visual", title: f.visual }, art(f.icon), el("span", { class: "frame-no" }, String(i + 1)), el("span", { class: "frame-dur" }, `${f.duration_sec}s`)),
      el("div", { class: "frame-body" },
        editable("frame-title", "title", "h3"),
        editable("frame-narr", "narration"),
        points,
        el("p", { class: "frame-vis-desc" }, el("strong", {}, "Visual: "), editable("", "visual", "span")),
        citeList(f.sources)),
      el("div", { class: "frame-tools", role: "toolbar", "aria-label": `Frame ${i + 1} actions` },
        el("button", { class: "icon-btn", title: "Move up", "aria-label": "Move frame up", disabled: i === 0, onclick: () => move(i, -1) }, icon("left")),
        el("button", { class: "icon-btn", title: "Move down", "aria-label": "Move frame down", disabled: i === n - 1, onclick: () => move(i, 1) }, icon("right")),
        el("button", { class: "icon-btn", title: "Regenerate this frame", "aria-label": "Regenerate frame", onclick: (e) => regen(i, e.currentTarget) }, icon("refresh")),
        el("button", { class: "icon-btn", title: "Delete frame", "aria-label": "Delete frame", disabled: n <= 1, onclick: () => { board.frames.splice(i, 1); renderBoard(); } }, icon("trash"))));
  }

  function move(i, d) {
    const j = i + d;
    [board.frames[i], board.frames[j]] = [board.frames[j], board.frames[i]];
    renderBoard();
    $(`.frame[data-index="${j}"] .frame-tools button`)?.focus();
  }

  async function regen(i, btn) {
    const instruction = window.prompt("Optional: how should this frame change? (leave blank to just regenerate)", "") ?? null;
    if (instruction === null) return;
    setLoading(btn, true);
    try {
      const res = await api("/api/storyboard/frame", { method: "POST", body: {
        frames: board.frames, index: i, topic: board.topic || undefined, instruction: instruction.trim() || undefined, ...board.scope,
      } });
      board.frames[i] = res.frame;
      renderBoard();
      toast(res.fallback ? "The model was unavailable; used a text-based frame." : `Frame ${i + 1} regenerated.`, res.fallback ? "info" : "success");
    } catch (err) { toast(err.message, "error"); setLoading(btn, false); }
  }

  function renderBoard() {
    const out = $("#create-out");
    const tl = el("div", { class: "timeline-bar", role: "img", "aria-label": `Timeline, total ${fmt(total())}` },
      board.frames.map((f, i) => el("button", { class: "tl-seg", style: `flex:${f.duration_sec || 1}`, title: `${i + 1}. ${f.title} (${f.duration_sec}s)`, onclick: () => $(`.frame[data-index="${i}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" }) }, String(i + 1))));
    out.replaceChildren(
      el("div", { class: "result-head" },
        el("div", {}, el("h2", {}, board.title), el("p", { class: "muted" },
          `${board.frames.length} frames · ${fmt(total())} · ${board.passages} passages used${board.map_reduce ? " · summarised in stages (large source set)" : ""}`)),
        el("div", { class: "result-actions" },
          el("button", { class: "btn small primary", onclick: () => present(0) }, icon("play"), "Present"),
          el("button", { class: "btn small", onclick: () => download(`${slug(board.title)}.md`, new Blob([toMarkdown()], { type: "text/markdown" })) }, icon("download"), "MD"),
          el("button", { class: "btn small", onclick: () => download(`${slug(board.title)}.json`, new Blob([JSON.stringify(exportObj(), null, 2)], { type: "application/json" })) }, icon("download"), "JSON"),
          el("button", { class: "btn small", onclick: printable }, icon("file"), "Print / PDF"))),
      board.fallback ? el("p", { class: "notice" }, "The AI model was unavailable or returned invalid output, so these frames were built directly from your text.") : null,
      tl,
      el("div", { class: "frames" }, board.frames.map(frameCard)));
  }

  const exportObj = () => ({ title: board.title, topic: board.topic, total_duration_sec: total(), frames: board.frames });
  function toMarkdown() {
    const lines = [`# ${board.title}`, "", `Total duration: ${fmt(total())}`, ""];
    board.frames.forEach((f, i) => {
      lines.push(`## ${i + 1}. ${f.title} (${f.duration_sec}s)`, "", f.narration, "");
      f.key_points.forEach((p) => lines.push(`- ${p}`));
      lines.push("", `*Visual:* ${f.visual} (icon: ${f.icon})`);
      if (f.sources?.length) lines.push("", "Sources: " + f.sources.map((s) => `[${s.n}] ${s.source}`).join("; "));
      lines.push("");
    });
    return lines.join("\n");
  }
  function printable() {
    const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(board.title)}</title><style>
      body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#111}h1{font-size:22px}
      .f{display:flex;gap:18px;border:1px solid #ccc;border-radius:10px;padding:14px;margin:0 0 14px;page-break-inside:avoid}
      .v{flex:0 0 110px;display:flex;align-items:center;justify-content:center;background:#eef;border-radius:8px;color:#335}
      h2{margin:0 0 6px;font-size:16px}.m{color:#555;font-size:12px}@media print{button{display:none}}</style></head><body>
      <button onclick="print()">Print / Save as PDF</button><h1>${esc(board.title)}</h1><p class="m">${board.frames.length} frames · ${fmt(total())}</p>
      ${board.frames.map((f, i) => `<div class="f"><div class="v">${artMarkup(f.icon)}</div><div><h2>${i + 1}. ${esc(f.title)} <span class="m">(${f.duration_sec}s)</span></h2>
      <p>${esc(f.narration)}</p><ul>${f.key_points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul><p class="m"><b>Visual:</b> ${esc(f.visual)}</p>
      ${f.sources?.length ? `<p class="m">Sources: ${f.sources.map((s) => `[${s.n}] ${esc(s.source)}`).join("; ")}</p>` : ""}</div></div>`).join("")}</body></html>`;
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    const w = window.open(url, "_blank", "noopener");
    if (!w) download(`${slug(board.title)}.html`, new Blob([html], { type: "text/html" }));
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  /* Present mode: fullscreen overlay; ←/→/space/PageUp/PageDown, Home/End, F fullscreen, Esc exit. */
  function present(start) {
    let i = start;
    const stage = el("div", { class: "present", role: "dialog", "aria-modal": "true", "aria-label": "Storyboard presentation", tabindex: "-1" });
    const live = el("div", { class: "sr-only", "aria-live": "polite" });
    const draw = () => {
      const f = board.frames[i];
      stage.replaceChildren(
        el("div", { class: "present-frame" },
          el("div", { class: "present-art" }, art(f.icon, 120)),
          el("h2", {}, f.title),
          el("p", { class: "present-narr" }, f.narration),
          f.key_points.length ? el("ul", {}, f.key_points.map((p) => el("li", {}, p))) : null),
        el("div", { class: "present-bar" },
          el("button", { class: "icon-btn", "aria-label": "Previous frame", onclick: () => go(-1) }, icon("left")),
          el("span", {}, `${i + 1} / ${board.frames.length}`),
          el("button", { class: "icon-btn", "aria-label": "Next frame", onclick: () => go(1) }, icon("right")),
          el("span", { class: "muted" }, "← → Space · F fullscreen · Esc exit"),
          el("button", { class: "icon-btn", "aria-label": "Exit presentation", onclick: close }, icon("x"))),
        el("div", { class: "present-progress" }, el("span", { style: `width:${((i + 1) / board.frames.length) * 100}%` })),
        live);
      live.textContent = `Frame ${i + 1}: ${f.title}`;
    };
    const go = (d) => { i = Math.max(0, Math.min(board.frames.length - 1, i + d)); draw(); };
    const key = (e) => {
      if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(1); }
      else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); go(-1); }
      else if (e.key === "Home") { i = 0; draw(); } else if (e.key === "End") { i = board.frames.length - 1; draw(); }
      else if (e.key === "f" || e.key === "F") { document.fullscreenElement ? document.exitFullscreen() : stage.requestFullscreen?.().catch(() => {}); }
      else if (e.key === "Escape" && !document.fullscreenElement) close();
    };
    function close() { document.removeEventListener("keydown", key, true); if (document.fullscreenElement) document.exitFullscreen(); stage.remove(); }
    document.addEventListener("keydown", key, true);
    document.body.append(stage);
    draw();
    stage.focus();
  }

  /* ─── Flashcards with SM-2 ─── */
  const deckId = (deck) => deck.id || (deck.id = `d${Date.now().toString(36)}`);
  function saveDeck(deck) {
    const data = SM2.load();
    const id = deckId(deck);
    data.decks[id] = { id, title: deck.title, topic: deck.topic, difficulty: deck.difficulty, created: Date.now(), fallback: deck.fallback, cards: deck.cards };
    if (!SM2.save(data)) toast("Browser storage is full or disabled; this deck won't be kept after reload.", "error");
  }

  function renderDecks() {
    const data = SM2.load();
    const decks = Object.values(data.decks).sort((a, b) => b.created - a.created);
    const box = $("#decks");
    if (!decks.length) { box.replaceChildren(el("p", { class: "placeholder" }, "Generated decks are saved in this browser and scheduled with spaced repetition (SM-2).")); return; }
    box.replaceChildren(el("h3", {}, "Your decks"), ...decks.map((d) => {
      const due = SM2.dueCards(d.cards, data.states).length;
      return el("div", { class: "deck-row" },
        el("div", { class: "source-meta" }, el("div", { class: "source-name" }, d.title),
          el("div", { class: "source-sub" }, `${d.cards.length} cards · ${due} due · ${d.difficulty}${d.fallback ? " · text-based" : ""}`)),
        el("button", { class: "btn small primary", onclick: () => startReview(d.id) }, due ? `Review ${due}` : "Practise"),
        el("button", { class: "btn small", title: "Anki CSV", onclick: () => download(`${slug(d.title)}-anki.csv`, new Blob([SM2.toAnkiCSV(d.cards)], { type: "text/csv" })) }, icon("download"), "Anki"),
        el("button", { class: "btn small", title: "JSON", onclick: () => download(`${slug(d.title)}.json`, new Blob([JSON.stringify(d, null, 2)], { type: "application/json" })) }, icon("download"), "JSON"),
        el("button", { class: "icon-btn", "aria-label": `Delete deck ${d.title}`, onclick: () => {
          if (!confirm(`Delete "${d.title}" and its review history?`)) return;
          const all = SM2.load(); d.cards.forEach((c) => delete all.states[c.id]); delete all.decks[d.id]; SM2.save(all); renderDecks();
        } }, icon("trash")));
    }));
  }

  const GRADES = [["Again", 1], ["Hard", 3], ["Good", 4], ["Easy", 5]];
  let reviewKey = null;
  function startReview(id, fresh) {
    const data = SM2.load();
    const deck = data.decks[id];
    if (!deck) return;
    let queue = SM2.dueCards(deck.cards, data.states);
    if (!queue.length) queue = deck.cards.slice();
    let flipped = false, done = 0;
    const out = $("#create-out");
    const finish = () => {
      document.removeEventListener("keydown", reviewKey); reviewKey = null;
      out.replaceChildren(el("div", { class: "state-box" }, icon("check"), el("p", {}, `Session done: ${done} card${done === 1 ? "" : "s"} reviewed. Due cards come back on schedule.`)));
      renderDecks();
    };
    const grade = (q) => {
      const card = queue.shift();
      const all = SM2.load();
      all.states[card.id] = SM2.review(all.states[card.id], q);
      SM2.save(all);
      done += 1;
      if (q < 3) queue.push(card); // relearn in this session
      flipped = false;
      queue.length ? draw() : finish();
    };
    const draw = () => {
      const c = queue[0];
      const st = SM2.load().states[c.id];
      out.replaceChildren(
        el("div", { class: "result-head" }, el("h2", {}, deck.title), el("p", { class: "muted" }, `${queue.length} left · ${done} reviewed${fresh && deck.fallback ? " · text-based cards (model unavailable)" : ""}`)),
        el("div", { class: `fc ${flipped ? "flipped" : ""}`, tabindex: "0", role: "button", "aria-label": flipped ? "Answer shown" : "Show answer (Space)", onclick: () => { flipped = true; draw(); } },
          el("div", { class: "fc-q" }, c.question),
          flipped ? el("div", { class: "fc-a" }, c.answer) : (c.hint ? el("div", { class: "muted" }, `Hint: ${c.hint}`) : null),
          el("div", { class: "fc-tags" }, [c.difficulty, ...(c.tags || [])].filter(Boolean).map((t) => el("span", { class: "chip" }, t)))),
        flipped ? el("div", { class: "grades", role: "group", "aria-label": "How well did you know it?" },
          GRADES.map(([label, q], k) => el("button", { class: `btn small grade-${label.toLowerCase()}`, onclick: () => grade(q) }, `${k + 1} · ${label}`)))
          : el("p", { class: "hint" }, "Space to flip, then 1–4 to grade."),
        flipped ? citeList(c.sources) : null,
        st ? el("p", { class: "hint" }, `Last interval ${st.interval} day(s), ease ${st.ef}`) : null,
        el("button", { class: "link-btn", onclick: finish }, "End session"));
    };
    if (reviewKey) document.removeEventListener("keydown", reviewKey);
    reviewKey = (e) => {
      if (KF.typing(e.target) || !$("#view-create") || $("#view-create").classList.contains("hidden")) return;
      if (e.key === " " && !flipped) { e.preventDefault(); flipped = true; draw(); }
      else if (flipped && ["1", "2", "3", "4"].includes(e.key)) { e.preventDefault(); grade(GRADES[+e.key - 1][1]); }
    };
    document.addEventListener("keydown", reviewKey);
    draw();
  }

  setMode("storyboard");
})();
