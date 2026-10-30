/* Whole-site crawl: start form, live job progress (polling), per-page status, cancel, retry. */
(() => {
  "use strict";
  const KF = window.KF;
  const { $, el, icon, api, toast, setLoading } = KF;
  const jobs = new Map(); // id -> snapshot
  const expanded = new Set();
  let timer = null;

  const STATUS_LABEL = {
    queued: "Queued", running: "Crawling", cancelling: "Cancelling", cancelled: "Cancelled", done: "Done", error: "Failed",
  };
  const PAGE_CLASS = { ingested: "ok", failed: "err", skipped: "muted", not_fetched: "warn", fetching: "busy", queued: "muted" };

  function jobCard(job) {
    const c = job.counts || {};
    const pct = job.max_pages ? Math.min(100, Math.round(((c.fetched || 0) + (c.failed || 0) + (c.skipped || 0)) / Math.max(1, Math.min(job.max_pages, c.found || 1)) * 100)) : 0;
    const active = ["queued", "running", "cancelling"].includes(job.status);
    const failedPages = (job.pages || []).filter((p) => p.status === "failed" || p.status === "not_fetched");
    const card = el("div", { class: `crawl-job ${job.status}`, "data-job": job.id },
      el("div", { class: "crawl-head" },
        el("span", { class: "source-icon" }, icon("sitemap")),
        el("div", { class: "source-meta" },
          el("div", { class: "source-name", title: job.url }, job.site),
          el("div", { class: "source-sub" }, `${STATUS_LABEL[job.status] || job.status} · depth ≤ ${job.max_depth} · ≤ ${job.max_pages} pages`)),
        active
          ? el("button", { class: "btn small", onclick: (e) => cancel(job.id, e.currentTarget) }, icon("stop"), "Cancel")
          : (failedPages.length ? el("button", { class: "btn small", onclick: (e) => retry(job.id, null, e.currentTarget) }, icon("refresh"), "Retry failed") : null)),
      el("div", { class: "progress slim", role: "progressbar", "aria-label": `Crawl progress for ${job.site}`, "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(active ? pct : 100) },
        el("span", { style: `width:${active ? pct : 100}%` })),
      el("div", { class: "crawl-counts" },
        ["found", "fetched", "ingested", "failed", "skipped"].map((k) => el("span", { class: `cc ${k}` }, el("b", {}, String(c[k] || 0)), ` ${k}`))),
      job.error ? el("p", { class: "crawl-error", role: "alert" }, job.error) : null,
      job.crawl_delay ? el("p", { class: "hint" }, `Polite delay ${job.crawl_delay}s between requests${job.sitemap_urls ? ` · ${job.sitemap_urls} URLs from sitemap` : ""}.`) : null);
    if (job.pages && job.pages.length) {
      const open = expanded.has(job.id);
      card.append(el("button", { class: "link-btn", "aria-expanded": String(open), onclick: () => { open ? expanded.delete(job.id) : expanded.add(job.id); render(); } },
        open ? "Hide pages" : `Show ${job.pages.length} pages`));
      if (open) {
        card.append(el("ul", { class: "crawl-pages" }, job.pages.map((p) => el("li", { class: `cp ${PAGE_CLASS[p.status] || ""}` },
          el("span", { class: "cp-status" }, p.status.replace("_", " ")),
          el("a", { href: p.url, target: "_blank", rel: "noopener noreferrer", title: p.url }, p.title || KF.displayName(p.url)),
          p.error ? el("span", { class: "cp-err", title: p.error }, p.error) : null,
          (p.status === "failed" || p.status === "not_fetched") && !active
            ? el("button", { class: "icon-btn tiny", title: "Retry this page", "aria-label": `Retry ${p.url}`, onclick: (e) => retry(job.id, [p.url], e.currentTarget) }, icon("refresh"))
            : null))));
      }
    }
    return card;
  }

  function render() {
    const box = $("#crawl-jobs");
    const list = [...jobs.values()].sort((a, b) => b.created - a.created).slice(0, 5);
    $("#crawl-panel").hidden = list.length === 0;
    box.replaceChildren(...list.map(jobCard));
  }

  async function poll() {
    clearTimeout(timer);
    const isActive = (j) => ["queued", "running", "cancelling"].includes(j.status);
    const todo = [...jobs.values()].filter((j) => isActive(j) || !j.pages);
    for (const j of todo) {
      const wasActive = isActive(j);
      try {
        const { job } = await api(`/api/crawl/${encodeURIComponent(j.id)}`);
        const before = j.counts?.ingested || 0;
        jobs.set(job.id, job);
        if (!wasActive) continue;
        if ((job.counts?.ingested || 0) !== before || !isActive(job)) KF.refreshSources();
        if (job.status === "done") toast(`Crawl of ${job.site} finished: ${job.counts.ingested} pages added.`, "success");
        if (job.status === "error") toast(`Crawl of ${job.site} failed: ${job.error || "unknown error"}`, "error", 8000);
      } catch (err) {
        if (/not found/i.test(err.message)) jobs.delete(j.id);
      }
    }
    render();
    if ([...jobs.values()].some(isActive)) timer = setTimeout(poll, 1500);
  }

  async function cancel(id, btn) {
    setLoading(btn, true);
    try { const { job } = await api(`/api/crawl/${encodeURIComponent(id)}/cancel`, { method: "POST" }); jobs.set(id, job); }
    catch (err) { toast(err.message, "error"); }
    poll();
  }

  async function retry(id, urls, btn) {
    setLoading(btn, true);
    try {
      const { job } = await api(`/api/crawl/${encodeURIComponent(id)}/retry`, { method: "POST", body: urls ? { urls } : {} });
      jobs.set(id, job);
      toast("Retrying failed pages…", "info");
    } catch (err) { toast(err.message, "error"); setLoading(btn, false); }
    poll();
  }

  $("#form-site").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = e.submitter || $("#form-site button");
    const url = $("#site-input").value.trim();
    if (!url) return;
    const body = {
      url,
      max_pages: +$("#site-pages").value || undefined,
      max_depth: +$("#site-depth").value,
      include_subdomains: $("#site-subdomains").checked,
      use_sitemap: $("#site-sitemap").checked,
      render_js: KF.state.renderJs?.available ? ($("#site-js").checked ? "auto" : "never") : undefined,
      path_prefix: $("#site-prefix").value.trim() || undefined,
    };
    setLoading(btn, true);
    try {
      const { job } = await api("/api/crawl", { method: "POST", body });
      jobs.set(job.id, job);
      expanded.add(job.id);
      toast(`Crawling ${job.site}… pages appear in Sources as they are added.`, "info");
      $("#site-input").value = "";
      poll();
    } catch (err) {
      toast(err.message, "error", 7000);
    } finally {
      setLoading(btn, false);
    }
  });

  KF.onSources(() => {
    const avail = KF.state.renderJs?.available;
    $("#site-js").disabled = !avail;
    $("#site-js-wrap").classList.toggle("disabled", !avail);
  });

  // Resume jobs that were running before a reload.
  api("/api/crawl").then(({ jobs: list }) => {
    (list || []).forEach((j) => jobs.set(j.id, j));
    render();
    poll();
  }).catch(() => {});
})();
