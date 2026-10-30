/* Knowledge graph: SVG force-directed layout (zoom, pan, drag), details panel, search/filter,
 * neighbour highlight, JSON/SVG/PNG export, keyboard + ARIA node list; plus overview, concept map,
 * glossary, timeline, connect-the-dots and related-sources views. */
(() => {
  "use strict";
  const KF = window.KF;
  const { $, $$, el, icon, api, toast, setLoading, download, slug } = KF;
  const NS = "http://www.w3.org/2000/svg";
  const PALETTE = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#db2777", "#65a30d", "#0d9488", "#ea580c"];
  let graph = null;
  let view = "graph";
  let sim = null;

  const svgEl = (tag, attrs = {}) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const scopeBody = () => ({ ...KF.scopeBody($("#graph-group").value), topic: $("#graph-topic").value.trim() || undefined });
  const out = () => $("#graph-out");
  const busy = () => out().replaceChildren(el("div", { class: "skeleton", style: "height:360px" }));
  const fail = (err) => { out().replaceChildren(el("div", { class: "state-box" }, icon("alert"), el("p", {}, err.message))); toast(err.message, "error"); };
  const fallbackNote = (on, what = "This view") => on ? el("p", { class: "notice" }, `${what} was built with a text-based fallback because the AI model was unavailable.`) : null;

  /* ─── Force simulation (velocity Verlet-ish, O(n²) repulsion; fine for ≤150 nodes) ─── */
  function simulate(nodes, edges, w, h, onTick) {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    nodes.forEach((n, i) => {
      const a = (i / nodes.length) * Math.PI * 2;
      n.x ??= w / 2 + Math.cos(a) * w * 0.3 * (0.5 + Math.random() * 0.5);
      n.y ??= h / 2 + Math.sin(a) * h * 0.3 * (0.5 + Math.random() * 0.5);
      n.vx = 0; n.vy = 0;
    });
    const links = edges.map((e) => [byId.get(e.source), byId.get(e.target)]).filter(([a, b]) => a && b);
    let alpha = 1, raf = 0, stopped = false;
    const step = () => {
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          let d2 = dx * dx + dy * dy || 0.01;
          if (d2 > 250000) continue;
          const f = (1800 * alpha) / d2;
          const d = Math.sqrt(d2);
          dx /= d; dy /= d;
          a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
        }
      }
      for (const [a, b] of links) {
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
        const f = ((d - 110) / d) * 0.04 * alpha;
        a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
      }
      for (const n of nodes) {
        n.vx += (w / 2 - n.x) * 0.004 * alpha; n.vy += (h / 2 - n.y) * 0.004 * alpha;
        if (n.fixed) { n.vx = n.vy = 0; continue; }
        n.vx *= 0.6; n.vy *= 0.6;
        n.x += Math.max(-30, Math.min(30, n.vx)); n.y += Math.max(-30, Math.min(30, n.vy));
      }
      alpha = Math.max(0, alpha * 0.985);
      onTick();
      if (alpha > 0.02 && !stopped) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return {
      reheat: () => { if (alpha < 0.3) { alpha = 0.3; cancelAnimationFrame(raf); raf = requestAnimationFrame(step); } },
      stop: () => { stopped = true; cancelAnimationFrame(raf); },
    };
  }

  /* ─── Graph view ─── */
  function renderGraph() {
    if (!graph) { out().replaceChildren(el("p", { class: "placeholder" }, "Build the graph to explore entities and how they relate.")); return; }
    sim?.stop();
    const colour = new Map(graph.types.map((t, i) => [t, PALETTE[i % PALETTE.length]]));
    const radius = (n) => 7 + n.importance * 16;
    const W = 900, H = 560;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "kg-svg", role: "img", "aria-label": `Knowledge graph with ${graph.nodes.length} entities and ${graph.edges.length} relations. Use the list to browse with a keyboard.` });
    const root = svgEl("g");
    svg.append(root);
    const edgeG = svgEl("g", { class: "kg-edges" }), nodeG = svgEl("g", { class: "kg-nodes" });
    root.append(edgeG, nodeG);
    const nbrs = new Map(graph.nodes.map((n) => [n.id, new Set()]));
    graph.edges.forEach((e) => { nbrs.get(e.source)?.add(e.target); nbrs.get(e.target)?.add(e.source); });
    const byId = new Map(graph.nodes.map((n) => [n.id, n]));
    const edgeEls = graph.edges.map((e) => {
      const g = svgEl("g", { class: "kg-edge" });
      const line = svgEl("line", { "stroke-width": String(1 + Math.min(4, e.weight || 1) * 0.6) });
      const label = svgEl("text", { class: "kg-elabel" }); label.textContent = e.label;
      const title = svgEl("title"); title.textContent = `${byId.get(e.source)?.name} — ${e.label} → ${byId.get(e.target)?.name}`;
      g.append(line, label, title); edgeG.append(g);
      return { e, g, line, label };
    });
    const nodeEls = graph.nodes.map((n) => {
      const g = svgEl("g", { class: "kg-node", tabindex: "-1", "data-id": n.id });
      const c = svgEl("circle", { r: String(radius(n)), fill: colour.get(n.type) || PALETTE[0] });
      const t = svgEl("text", { dy: String(radius(n) + 12), "text-anchor": "middle" }); t.textContent = n.name.length > 26 ? `${n.name.slice(0, 25)}…` : n.name;
      const title = svgEl("title"); title.textContent = `${n.name} (${n.type})`;
      g.append(c, t, title); nodeG.append(g);
      return { n, g };
    });
    const tick = () => {
      for (const { e, line, label } of edgeEls) {
        const a = byId.get(e.source), b = byId.get(e.target);
        line.setAttribute("x1", a.x); line.setAttribute("y1", a.y); line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
        label.setAttribute("x", (a.x + b.x) / 2); label.setAttribute("y", (a.y + b.y) / 2);
      }
      for (const { n, g } of nodeEls) g.setAttribute("transform", `translate(${n.x.toFixed(1)},${n.y.toFixed(1)})`);
    };

    // zoom & pan
    const z = { x: 0, y: 0, k: 1 };
    const applyZ = () => root.setAttribute("transform", `translate(${z.x},${z.y}) scale(${z.k})`);
    const toLocal = (ev) => { const r = svg.getBoundingClientRect(); return { x: ((ev.clientX - r.left) / r.width) * W, y: ((ev.clientY - r.top) / r.height) * H }; };
    const zoomAt = (f, p = { x: W / 2, y: H / 2 }) => { const k = Math.max(0.2, Math.min(5, z.k * f)); z.x = p.x - ((p.x - z.x) * k) / z.k; z.y = p.y - ((p.y - z.y) * k) / z.k; z.k = k; applyZ(); };
    svg.addEventListener("wheel", (ev) => { ev.preventDefault(); zoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, toLocal(ev)); }, { passive: false });
    let drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      const g = ev.target.closest(".kg-node");
      const p = toLocal(ev);
      if (g) { const n = byId.get(g.dataset.id); drag = { n, moved: false }; n.fixed = true; }
      else drag = { pan: true, sx: p.x - z.x, sy: p.y - z.y };
      svg.setPointerCapture(ev.pointerId);
    });
    svg.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const p = toLocal(ev);
      if (drag.pan) { z.x = p.x - drag.sx; z.y = p.y - drag.sy; applyZ(); return; }
      drag.moved = true; drag.n.x = (p.x - z.x) / z.k; drag.n.y = (p.y - z.y) / z.k; tick(); sim.reheat();
    });
    svg.addEventListener("pointerup", () => {
      if (drag?.n) { drag.n.fixed = false; if (!drag.moved) select(drag.n.id); }
      else if (drag?.pan === true) { /* click on background clears selection */ }
      drag = null;
    });

    // selection & filter
    let selected = null;
    const panel = el("aside", { class: "kg-panel", "aria-live": "polite" }, el("p", { class: "muted" }, "Click a node, or pick one from the list, to see what the sources say about it."));
    const highlight = () => {
      const q = search.value.trim().toLowerCase();
      const type = typeSel.value;
      const near = selected ? nbrs.get(selected) : null;
      for (const { n, g } of nodeEls) {
        const match = (!type || n.type === type) && (!q || n.name.toLowerCase().includes(q) || n.aliases.some((a) => a.toLowerCase().includes(q)));
        g.classList.toggle("dim", !match || (selected && n.id !== selected && !near.has(n.id)));
        g.classList.toggle("sel", n.id === selected);
        g.style.display = type && n.type !== type ? "none" : "";
      }
      for (const { e, g } of edgeEls) {
        const hidden = type && (byId.get(e.source).type !== type || byId.get(e.target).type !== type);
        g.style.display = hidden ? "none" : "";
        g.classList.toggle("dim", !!selected && e.source !== selected && e.target !== selected);
        g.classList.toggle("hot", !!selected && (e.source === selected || e.target === selected));
      }
      $$(".kg-list li", wrap).forEach((li) => {
        const n = byId.get(li.dataset.id);
        li.hidden = !((!type || n.type === type) && (!q || n.name.toLowerCase().includes(q)));
        li.setAttribute("aria-selected", String(n.id === selected));
      });
    };
    function select(id) {
      selected = selected === id ? null : id;
      highlight();
      if (!selected) { panel.replaceChildren(el("p", { class: "muted" }, "Nothing selected.")); return; }
      const n = byId.get(id);
      const rel = graph.edges.filter((e) => e.source === id || e.target === id);
      panel.replaceChildren(
        el("div", { class: "kg-head" }, el("span", { class: "dot", style: `background:${colour.get(n.type)}` }), el("h3", {}, n.name), el("span", { class: "chip" }, n.type)),
        n.description ? el("p", {}, n.description) : null,
        n.aliases.length ? el("p", { class: "muted" }, `Also: ${n.aliases.slice(0, 5).join(", ")}`) : null,
        el("p", { class: "muted" }, `Importance ${(n.importance * 100).toFixed(0)}% · ${n.mentions} mention(s) · ${n.degree} link(s)`),
        rel.length ? el("div", {}, el("h4", {}, "Relations"), el("ul", { class: "kg-rels" }, rel.slice(0, 12).map((e) => {
          const other = byId.get(e.source === id ? e.target : e.source);
          return el("li", {}, e.source === id ? `→ ${e.label} ` : `← ${e.label} `, el("button", { class: "link-btn", onclick: () => select(other.id) }, other.name));
        }))) : null,
        n.sources.length ? el("div", {}, el("h4", {}, "Passages"), n.sources.map((s) => el("blockquote", { class: "kg-quote" }, s.snippet, el("cite", {}, KF.displayName(s.source))))) : null,
        el("div", { class: "result-actions" },
          el("button", { class: "btn small primary", onclick: () => KF.askAbout(`What do my sources say about ${n.name}${n.type !== "Other" ? ` (${n.type})` : ""}? How does it relate to the other topics?`) }, icon("chat-q"), "Ask about this"),
          el("button", { class: "btn small", onclick: () => { connectFrom = n.name; setView("connect"); } }, icon("graph"), "Connect…")));
    }

    const search = el("input", { type: "search", placeholder: "Find a node…", "aria-label": "Find a node", oninput: highlight });
    const typeSel = el("select", { "aria-label": "Filter by type", onchange: highlight }, el("option", { value: "" }, "All types"), graph.types.map((t) => el("option", { value: t }, t)));
    const legend = el("div", { class: "kg-legend" }, graph.types.map((t) => el("button", { class: "chip", onclick: () => { typeSel.value = typeSel.value === t ? "" : t; highlight(); } }, el("span", { class: "dot", style: `background:${colour.get(t)}` }), t)));
    const list = el("ul", { class: "kg-list", role: "listbox", "aria-label": "Entities (arrow keys to move, Enter to select)", tabindex: "0" },
      [...graph.nodes].sort((a, b) => b.importance - a.importance).map((n) => el("li", { role: "option", "data-id": n.id, "aria-selected": "false", onclick: () => select(n.id) }, el("span", { class: "dot", style: `background:${colour.get(n.type)}` }), n.name)));
    let cursor = -1;
    list.addEventListener("keydown", (e) => {
      const items = $$("li:not([hidden])", list);
      if (!items.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); cursor = Math.max(0, Math.min(items.length - 1, cursor + (e.key === "ArrowDown" ? 1 : -1))); items.forEach((li, i) => li.classList.toggle("cursor", i === cursor)); items[cursor].scrollIntoView({ block: "nearest" }); list.setAttribute("aria-activedescendant", ""); }
      else if ((e.key === "Enter" || e.key === " ") && items[cursor]) { e.preventDefault(); select(items[cursor].dataset.id); }
    });
    const svgString = () => {
      const clone = svg.cloneNode(true);
      clone.setAttribute("xmlns", NS); clone.setAttribute("width", W); clone.setAttribute("height", H);
      const style = svgEl("style");
      const fg = getComputedStyle(document.documentElement).getPropertyValue("--text").trim() || "#111";
      style.textContent = `text{font:11px system-ui,sans-serif;fill:${fg}}.kg-elabel{font-size:9px;opacity:.7}line{stroke:#94a3b8}.dim{opacity:.15}`;
      clone.prepend(style);
      return { clone, text: new XMLSerializer().serializeToString(clone) };
    };
    const toolbar = el("div", { class: "kg-tools", role: "toolbar", "aria-label": "Graph controls" },
      search, typeSel,
      el("button", { class: "icon-btn", "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoomAt(1.2) }, icon("zoom-in")),
      el("button", { class: "icon-btn", "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoomAt(1 / 1.2) }, icon("zoom-out")),
      el("button", { class: "icon-btn", "aria-label": "Reset view", title: "Reset view", onclick: () => { z.x = z.y = 0; z.k = 1; applyZ(); } }, icon("fit")),
      el("span", { class: "sep" }),
      el("button", { class: "btn small", onclick: () => download(`${slug(graphTitle())}.json`, new Blob([JSON.stringify({ nodes: graph.nodes.map(({ x, y, vx, vy, fixed, ...n }) => n), edges: graph.edges }, null, 2)], { type: "application/json" })) }, icon("download"), "JSON"),
      el("button", { class: "btn small", onclick: () => download(`${slug(graphTitle())}.svg`, new Blob([svgString().text], { type: "image/svg+xml" })) }, icon("download"), "SVG"),
      el("button", { class: "btn small", onclick: () => KF.exportPng(svgString().clone, graphTitle()) }, icon("download"), "PNG"));
    const wrap = el("div", { class: "kg" },
      toolbar,
      fallbackNote(graph.fallback, "The graph") || (graph.partial_fallback ? el("p", { class: "notice" }, "Some batches used a text-based fallback.") : null),
      el("div", { class: "kg-main" },
        el("div", { class: "kg-stage", tabindex: "0", "aria-label": "Graph canvas. Plus/minus zoom, arrows pan.", onkeydown: (e) => {
          const step = 40;
          if (e.key === "+" || e.key === "=") zoomAt(1.2); else if (e.key === "-") zoomAt(1 / 1.2);
          else if (e.key === "ArrowLeft") z.x += step; else if (e.key === "ArrowRight") z.x -= step;
          else if (e.key === "ArrowUp") z.y += step; else if (e.key === "ArrowDown") z.y -= step; else return;
          e.preventDefault(); applyZ();
        } }, svg),
        el("div", { class: "kg-side" }, panel, el("details", { class: "kg-list-wrap", open: true }, el("summary", {}, `${graph.nodes.length} entities`), list))),
      legend,
      el("p", { class: "hint" }, `${graph.nodes.length} entities · ${graph.edges.length} relations · ${graph.passages} passages in ${graph.batches} batch(es). Drag nodes, scroll to zoom, drag the background to pan.`));
    out().replaceChildren(wrap);
    sim = simulate(graph.nodes, graph.edges, W, H, tick);
  }
  const graphTitle = () => `knowledge-graph-${$("#graph-topic").value.trim() || "all"}`;

  async function build(refresh = false) {
    const btn = $("#graph-build");
    setLoading(btn, true);
    busy();
    try {
      graph = await api("/api/graph", { method: "POST", body: { ...scopeBody(), refresh } });
      graph.nodes.forEach((n) => { delete n.x; delete n.y; });
      setView("graph");
    } catch (err) { graph = null; fail(err); }
    finally { setLoading(btn, false); }
  }
  $("#graph-build").addEventListener("click", () => build(!!graph));

  /* ─── Other views ─── */
  let connectFrom = "";
  async function renderOverview() {
    busy();
    try {
      const d = await api("/api/overview", { method: "POST", body: scopeBody() });
      out().replaceChildren(
        fallbackNote(d.fallback, "This overview"),
        el("section", { class: "ov-all" }, el("h3", {}, "Across your sources"), el("p", {}, d.overall),
          el("div", { class: "chips" }, d.key_topics.map((t) => el("button", { class: "chip", onclick: () => KF.askAbout(`Tell me about ${t} using my sources.`) }, t)))),
        el("div", { class: "ov-grid" }, d.sources.map((s) => el("article", { class: "ov-card" },
          el("h4", { title: s.source }, KF.displayName(s.source)),
          el("p", {}, s.summary || el("span", { class: "muted" }, "No summary.")),
          el("div", { class: "chips" }, s.key_topics.map((t) => el("span", { class: "chip" }, t))),
          el("div", { class: "result-actions" },
            el("span", { class: "muted" }, `${s.chunks} chunks`),
            el("button", { class: "link-btn", onclick: () => { relatedSource = s.source; setView("related"); } }, "Related sources"))))));
    } catch (err) { fail(err); }
  }

  async function renderMap() {
    busy();
    try {
      const d = await api("/api/concept-map", { method: "POST", body: scopeBody() });
      const box = el("div", {});
      out().replaceChildren(fallbackNote(d.fallback, "This map"), box);
      await KF.renderMermaid(box, d.mermaid_code, "Concept map");
    } catch (err) { fail(err); }
  }

  async function renderGlossary() {
    busy();
    try {
      const d = await api("/api/glossary", { method: "POST", body: scopeBody() });
      const filter = el("input", { type: "search", placeholder: "Filter terms…", "aria-label": "Filter glossary" });
      const dl = el("dl", { class: "glossary" });
      const draw = () => {
        const q = filter.value.trim().toLowerCase();
        dl.replaceChildren(...d.terms.filter((t) => !q || t.term.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q)).flatMap((t) => [
          el("dt", {}, t.term, el("span", { class: "chip" }, t.type)),
          el("dd", {}, t.definition, t.aliases.length ? el("span", { class: "muted" }, ` (also: ${t.aliases.join(", ")})`) : null,
            t.sources[0] ? el("span", { class: "muted" }, ` — ${KF.displayName(t.sources[0].source)}`) : null)]));
      };
      filter.addEventListener("input", draw);
      draw();
      const md = () => d.terms.map((t) => `**${t.term}** (${t.type}): ${t.definition}`).join("\n\n");
      out().replaceChildren(fallbackNote(d.fallback, "This glossary"),
        el("div", { class: "kg-tools" }, filter, el("button", { class: "btn small", onclick: () => download("glossary.md", new Blob([md()], { type: "text/markdown" })) }, icon("download"), "MD")),
        d.terms.length ? dl : el("p", { class: "placeholder" }, "No defined terms were found."));
    } catch (err) { fail(err); }
  }

  async function renderTimeline() {
    busy();
    try {
      const d = await api("/api/timeline", { method: "POST", body: scopeBody() });
      out().replaceChildren(d.events.length
        ? el("ol", { class: "tl" }, d.events.map((e) => el("li", {},
          el("span", { class: "tl-date" }, e.date),
          el("div", {}, el("p", {}, e.text), el("span", { class: "muted" }, KF.displayName(e.source))))))
        : el("p", { class: "placeholder" }, "No dated events (years) were found in these sources."));
    } catch (err) { fail(err); }
  }

  function renderConnect() {
    const names = graph ? graph.nodes.map((n) => n.name) : [];
    const dl = el("datalist", { id: "kg-names" }, names.map((n) => el("option", { value: n })));
    const a = el("input", { type: "text", list: "kg-names", placeholder: "First concept", "aria-label": "First concept", value: connectFrom, required: true });
    const b = el("input", { type: "text", list: "kg-names", placeholder: "Second concept", "aria-label": "Second concept", required: true });
    const res = el("div", { class: "connect-out", "aria-live": "polite" });
    const go = el("button", { class: "btn primary", type: "submit" }, icon("graph"), "Connect the dots");
    const form = el("form", { class: "kg-tools", onsubmit: async (e) => {
      e.preventDefault();
      setLoading(go, true);
      try {
        const d = await api("/api/connect", { method: "POST", body: { ...scopeBody(), a: a.value.trim(), b: b.value.trim() } });
        res.replaceChildren(
          el("p", { class: "connect-expl" }, d.explanation),
          d.fallback && d.paths.length ? el("p", { class: "hint" }, "Explanation is templated from the graph (model unavailable).") : null,
          d.paths.map((p, i) => el("div", { class: "path" }, el("strong", {}, `Path ${i + 1}: `),
            p.names.flatMap((n, j) => [j ? el("span", { class: "arrow" }, ` —${p.edges[j - 1].label}→ `) : null, el("span", { class: "chip" }, n)]))),
          el("button", { class: "btn small", onclick: () => KF.askAbout(`How is ${d.from} connected to ${d.to}? Use my sources.`) }, icon("chat-q"), "Ask about this"));
      } catch (err) { res.replaceChildren(el("p", { class: "crawl-error" }, err.message)); }
      finally { setLoading(go, false); }
    } }, a, b, go, dl);
    out().replaceChildren(el("p", { class: "muted" }, graph ? "Pick two entities to find how they link in your sources." : "Tip: build the graph first to get name suggestions."), form, res);
  }

  let relatedSource = "";
  function renderRelated() {
    const sources = KF.state.sources || [];
    const sel = el("select", { "aria-label": "Source" }, sources.map((s) => el("option", { value: s.source }, KF.displayName(s.source))));
    if (relatedSource) sel.value = relatedSource;
    const res = el("div", { "aria-live": "polite" });
    const run = async () => {
      relatedSource = sel.value;
      if (!relatedSource) return;
      res.replaceChildren(el("div", { class: "skeleton" }));
      try {
        const d = await api("/api/related-sources", { method: "POST", body: { source: relatedSource, k: 8 } });
        res.replaceChildren(d.related.length ? el("ol", { class: "related" }, d.related.map((r) => el("li", {},
          el("strong", {}, KF.displayName(r.source)), el("span", { class: "muted" }, ` similarity ${(r.similarity * 100).toFixed(0)}%`),
          r.shared_terms.length ? el("div", { class: "chips" }, r.shared_terms.map((t) => el("span", { class: "chip" }, t))) : null)))
          : el("p", { class: "placeholder" }, "No other sources to compare with."));
      } catch (err) { res.replaceChildren(el("p", { class: "crawl-error" }, err.message)); }
    };
    sel.addEventListener("change", run);
    out().replaceChildren(sources.length ? el("div", { class: "kg-tools" }, el("label", { class: "mini grow" }, "Find sources similar to", sel)) : el("p", { class: "placeholder" }, "Add sources first."), res);
    if (sources.length) run();
  }

  const RENDER = { graph: renderGraph, overview: renderOverview, map: renderMap, glossary: renderGlossary, timeline: renderTimeline, connect: renderConnect, related: renderRelated };
  function setView(v) {
    view = v;
    if (v !== "graph") sim?.stop();
    $$(".gseg").forEach((b) => { const on = b.dataset.gview === v; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); });
    RENDER[v]();
  }
  $$(".gseg").forEach((b) => b.addEventListener("click", () => setView(b.dataset.gview)));
  $("#graph-group").addEventListener("change", () => { graph = null; if (view !== "graph") setView(view); else renderGraph(); });
  KF.graphView = () => view;
})();
