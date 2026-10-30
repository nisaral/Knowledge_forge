/* Search tab: hybrid search with filters, highlighted snippets grouped by page, passage view, "ask about this". */
(() => {
  "use strict";
  const KF = window.KF;
  const { $, el, icon, api, toast } = KF;
  let last = null;

  function highlighted(text, spans) {
    const out = [];
    let pos = 0;
    for (const [a, b] of spans || []) {
      if (a < pos) continue;
      out.push(text.slice(pos, a), el("mark", {}, text.slice(a, b)));
      pos = b;
    }
    out.push(text.slice(pos));
    return out;
  }

  function spansFor(text, terms) {
    const spans = [];
    const low = text.toLowerCase();
    for (const t of terms || []) {
      let i = low.indexOf(t);
      while (i !== -1) {
        let j = i + t.length;
        while (j < low.length && /\w/.test(low[j])) j++;
        spans.push([i, j]);
        i = low.indexOf(t, j);
      }
    }
    return spans.sort((x, y) => x[0] - y[0]);
  }

  function fillSelect(sel, values, allLabel, label = (v) => v) {
    const cur = sel.value;
    sel.replaceChildren(el("option", { value: "" }, allLabel), ...values.map((v) => el("option", { value: v }, label(v))));
    if (values.includes(cur)) sel.value = cur;
  }

  KF.onSources((sources) => {
    fillSelect($("#filter-type"), [...new Set(sources.map((s) => s.type))].sort(), "All types");
    fillSelect($("#filter-site"), [...new Set(sources.map((s) => s.site).filter(Boolean))].sort(), "All sites");
    fillSelect($("#filter-source"), sources.map((s) => s.source), "All sources", KF.displayName);
  });

  function passageLink(source, text) {
    if (!/^https?:\/\//.test(source)) return null;
    const frag = text.split(/\s+/).slice(0, 6).join(" ");
    return `${source.split("#")[0]}#:~:text=${encodeURIComponent(frag)}`;
  }

  function openPassage(hit, terms) {
    const dlg = el("dialog", { class: "modal passage", "aria-label": `Passage from ${hit.title || hit.source}` },
      el("div", { class: "modal-head" }, el("h2", {}, hit.title || KF.displayName(hit.source)),
        el("button", { class: "icon-btn", "aria-label": "Close", onclick: () => dlg.close() }, icon("x"))),
      el("p", { class: "muted" }, `${hit.source} · score ${hit.score.toFixed(3)} (semantic ${hit.semantic.toFixed(2)}, keyword ${hit.keyword.toFixed(2)})`),
      el("div", { class: "passage-text" }, highlighted(hit.text, spansFor(hit.text, terms))),
      el("div", { class: "result-actions" },
        passageLink(hit.source, hit.text) ? el("a", { class: "btn small", href: passageLink(hit.source, hit.text), target: "_blank", rel: "noopener noreferrer" }, icon("globe"), "Open at passage") : null,
        el("button", { class: "btn small primary", onclick: () => { dlg.close(); ask(hit); } }, icon("chat-q"), "Ask about this")));
    dlg.addEventListener("close", () => dlg.remove());
    document.body.append(dlg);
    dlg.showModal();
  }

  function ask(hit) {
    const quote = hit.text.length > 400 ? `${hit.text.slice(0, 400)}…` : hit.text;
    KF.askAbout(`Explain this passage from "${hit.title || KF.displayName(hit.source)}" in context: "${quote}"`);
  }

  function render(data) {
    const box = $("#search-results");
    if (!data.results.length) {
      box.replaceChildren(el("div", { class: "state-box" }, icon("search"), el("p", {}, "No matching passages. Try other words or loosen the filters.")));
      return;
    }
    box.replaceChildren(
      el("p", { class: "muted" }, `${data.total} passage${data.total === 1 ? "" : "s"} on ${data.pages.length} page${data.pages.length === 1 ? "" : "s"}`),
      ...data.pages.map((p) => el("article", { class: "search-page" },
        el("div", { class: "sp-head" },
          el("div", {},
            el("h3", {}, p.title || KF.displayName(p.source)),
            el("div", { class: "source-sub" }, [p.site || p.source_type, KF.displayName(p.source)].filter(Boolean).join(" · "))),
          el("span", { class: "score", title: "Best fused score on this page" }, p.score.toFixed(3))),
        p.hits.map((h) => el("div", { class: "sp-hit" },
          el("button", { class: "snippet", title: "Open the full passage", onclick: () => openPassage(h, data.terms) }, highlighted(h.snippet, h.highlights)),
          el("div", { class: "sp-meta" },
            el("span", { class: "muted" }, `score ${h.score.toFixed(3)} · semantic ${h.semantic.toFixed(2)} · keyword ${h.keyword.toFixed(2)}`),
            el("button", { class: "link-btn", onclick: () => ask(h) }, icon("chat-q"), "Ask about this")))))));
  }

  $("#search-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("#search-input").value.trim();
    if (!query) return;
    const body = { query, k: 20, per_page: 3 };
    if ($("#filter-type").value) body.types = [$("#filter-type").value];
    if ($("#filter-site").value) body.sites = [$("#filter-site").value];
    if ($("#filter-source").value) body.sources = [$("#filter-source").value];
    const box = $("#search-results");
    box.replaceChildren(...[90, 75, 82].map((w) => el("div", { class: "skeleton", style: `width:${w}%` })));
    try {
      last = await api("/api/search", { method: "POST", body });
      render(last);
    } catch (err) {
      box.replaceChildren(el("div", { class: "state-box" }, icon("alert"), el("p", {}, err.message)));
      toast(err.message, "error");
    }
  });
  ["#filter-type", "#filter-site", "#filter-source"].forEach((s) => $(s).addEventListener("change", () => {
    if ($("#search-input").value.trim()) $("#search-form").requestSubmit();
  }));
  document.addEventListener("kf:view", (e) => { if (e.detail === "search") $("#search-input").focus(); });
})();
