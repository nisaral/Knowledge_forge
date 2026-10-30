# Testing

## Automated tests

```bash
pip install -r requirements-dev.txt
pytest -q                                   # offline; Gemini and the network are mocked
pip install -r requirements-js.txt && playwright install --with-deps chromium
pytest -q tests/test_js_render.py           # real headless Chromium against a local test server
```

CI runs ruff, pytest on Python 3.10/3.11/3.12, a JS-render job with real Chromium, and a Docker build.

## Live end-to-end run (30 Oct 2026)

Real public sites and real PDFs went through the Flask API (`/api/add-content`, `/api/upload-file`) and the UI.

**Read these caveats first:**

- **No Gemini key was available.** All generation used a **fake model** that returns canned text, and embeddings were deterministic hash vectors. The crawling, extraction, chunking, storage, routing and error handling were real. Answer quality, OCR and retrieval relevance were **not** tested.
- The test box resolves every public hostname to a 198.18.0.0/15 address behind a transparent proxy. The SSRF guard correctly treats that range as non-public, so the run used `ALLOW_PRIVATE_URLS=true`. SSRF blocking is covered separately by the unit tests and the real-Chromium tests.
- Politeness: the run sent one request per URL with 2 s or more between requests, used the crawler's identifying User-Agent, and checked robots.txt before every fetch except the robots test target. There, the app's own check refused the URL after fetching only robots.txt.
- Honest note: the first harness run fetched robots.txt without our User-Agent. Wikipedia answered 403, which the harness wrongly treated as "no rules", so **one request was made to a robots-disallowed page** (`/wiki/Special:Search`) before the bug was caught. The harness was fixed, and the app itself now enforces robots.txt.

Settings: `RENDER_JS=auto`, default limits (`CRAWL_MAX_BYTES` 5 MB, `CRAWL_TIMEOUT_SECONDS` 15, except 5 for the slow target).

| Target | Result | Chunks | Time (s) | Bug found → fix |
|---|---|---|---|---|
| Static docs: docs.python.org tutorial | ✅ added | 10 | 0.1 | – |
| Blog index: blog.python.org | ✅ added (listing page has little text) | 1 | 1.8 | – |
| Blog post: blog.python.org (3.16.0a1) | ✅ added | 2 | 0.2 | – |
| Wikipedia section: Photosynthesis#Light-dependent_reactions | ✅ added, **whole article** | 42 | 0.4 | Not fixed: fragments are ignored and the whole page is ingested (see below) |
| JS SPA quotes.toscrape.com/js, rendering off | ❌ clear error suggesting the toggle | 0 | 2.8 | – |
| JS SPA quotes.toscrape.com/js, rendering on | ✅ added via headless Chromium | 2 | 4.4 | – |
| robots-disallowed: Wikipedia Special:Search | ❌ "robots.txt does not allow crawling this page" | 0 | 0.0 | **Crawler ignored robots.txt** → `CRAWL_RESPECT_ROBOTS` (default on) |
| Sitemap: sitemaps.org/sitemap.xml | ❌ "This is an XML sitemap…" | 0 | 0.5 | **Sitemap ingested as junk text**; later **HTTP 406** because the Accept header had no `*/*` → both fixed |
| Site with a sitemap: sitemaps.org/protocol.html | ✅ added | 9 | 0.1 | – |
| Slow: httpbin.org/delay/10 (5 s timeout) | ❌ "The site took too long to respond." | 0 | 5.4 | – |
| Large: html.spec.whatwg.org (>5 MB) | ❌ "larger than the 5120 KB crawl limit" | 0 | 2.4 | **auto mode launched the browser after the size error and showed a vague "could not load"** → fixed |
| Text PDF: W3C dummy.pdf (1 line) | ✅ added | 1 | 3.7 | **Rejected: short text was sent to OCR, and OCR failure discarded it** → fixed |
| arXiv 1706.03762 (15 pages, 2.2 MB) | ✅ added | 18 | 5.4 | – |
| Image-only PDF (generated locally; no real scan used) | ❌ "no text layer … OCR did not return any text" (expected without a key) | 0 | 0.0 | Error was generic → clearer message |
| Password-protected PDF (generated locally) | ❌ "This PDF is password protected…" | 0 | 0.0 | – (already handled) |
| Truncated PDF (first 2 KB of the arXiv file) | ❌ "may be damaged or truncated" | 0 | 2.3 | Error was generic → fixed |

Tool flows on the ingested content (all on the **fake model**): ask-question, chat/stream, summarize, extract-entities, storyboard, flashcards, quiz, mindmap → 200; cluster-topics → 200 (1.5 s, real KMeans and matplotlib). This shows routing, retrieval plumbing, citations and response shapes work. It says nothing about output quality.

### Not fixed

- **URL fragments** (`#section`) are ignored; the whole page is ingested. Extracting just one section is a feature decision (heading ids vary by site), not a crash.
- **OCR of scanned PDFs** was not exercised (no key), and no real scanned PDF was used; the image-only PDF was generated locally.
- The upload toast says "media files can take a minute" for PDFs too (cosmetic).

### Screenshots (real runs, fake model)

`docs/e2e/01-static-docs.png`, `02-spa-without-js-error.png`, `03-spa-rendered.png`, `04-robots-refused.png`, `05-large-page-limit.png`, `06-arxiv-pdf.png`, `07-password-pdf-error.png`, `08-chat-over-real-sources-fake-model.png`.

## Live UI run with the real Gemini key (30 Oct 2026, evening)

A real key was supplied through `.env`, which is git-ignored. The key was never printed, and server logs were read through a redaction filter. A check confirmed the key string is absent from the working tree, from `HEAD` and from every screenshot source.

**What the real model actually did:**

- **Embeddings: real.** There were 13 `gemini-embedding-001:batchEmbedContents` calls, and all returned HTTP 200. Search ranking and retrieval in this run therefore used real Gemini vectors.
- **Generation: failed and fell back.** All 195 `generateContent` calls returned **HTTP 429**: "Quota exceeded … GenerateContent request limit per minute for a region". The models tried were `gemini-2.0-flash`, `2.5-flash`, `2.5-flash-lite` and `flash-latest`. The 429 persisted after more than 90 s idle, so the key's project has no usable generation quota. As designed, the app fell back to its text-based heuristics, and the UI shows a "built with a text-based fallback" banner. **The storyboard, flashcards and knowledge graph in the screenshots below are fallback output, not model output.** LLM output quality remains untested.

Run with Playwright and headless Chromium against `flask run`, using `ALLOW_PRIVATE_URLS=true` (see the DNS caveat above), `RENDER_JS=auto` and `CRAWL_SITE_DELAY_SECONDS=1.5`:

| Step | Result |
|---|---|
| Site crawl: flask.palletsprojects.com/en/stable/, 6 pages, depth 1 | ✅ done. 67 links found, 6 fetched and ingested, 0 failed, about 10 s. robots.txt honoured; 1.5 s between requests; no sitemap URLs listed |
| PDF by link: arxiv.org/pdf/1706.03762 | ❌ at first: **HTTP 422 "Unsupported content type 'application/pdf'"**. The Link box rejected PDF URLs; only the crawler handled them. → **fixed**: single-URL ingest now detects PDFs (content type or `%PDF-` magic), extracts them with the PDF pipeline under the upload size cap, and has tests. On re-run: ✅ 18 chunks |
| Search | ✅ 7 page groups with highlighted snippets; the passage dialog opens |
| Storyboard | ✅ 5 frames, timeline and present mode (**fallback**) |
| Flashcards | ✅ 6 cards, SM-2 review (**fallback, "text-based"**) |
| Knowledge graph and extras | ✅ 80 nodes; overview, concept map, glossary, timeline, connect and related all render (**fallback**) |
| Browser console errors | 0 on the final run |

Screenshots: `docs/e2e/11-crawl-done-site-group.png` to `25-mobile.png`.
