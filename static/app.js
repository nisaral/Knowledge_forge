/* KnowledgeForge V2 — app.js */
mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });

const API = '/api';
let chatSrc = '';

// ── Utilities ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const toast = (msg, isErr = false) => {
  const el = $('toast');
  el.innerHTML = `<i class="fas fa-${isErr ? 'exclamation-circle' : 'check-circle'}"></i> ${msg}`;
  el.className = `toast show ${isErr ? 'error' : 'success'}`;
  setTimeout(() => el.classList.remove('show'), 3500);
};
const loader = (show, text = 'Processing...') => {
  $('overlay').classList.toggle('active', show);
  $('overlay-text').textContent = text;
};

async function api(endpoint, opts = {}) {
  const r = await fetch(`${API}/${endpoint}`, opts);
  return r.json();
}

// ── Sources ────────────────────────────────────────────────────────────────
const TYPE_ICONS = { web:'fa-globe', youtube:'fa-youtube fab', pdf:'fa-file-pdf',
  audio:'fa-microphone-alt', video:'fa-video', image:'fa-image', text:'fa-file-alt' };
const TYPE_CLS   = { web:'chip-web', youtube:'chip-youtube', pdf:'chip-pdf',
  audio:'chip-audio', video:'chip-video', image:'chip-image', text:'chip-text' };

async function refreshSources() {
  const data = await api('get-sources');
  if (!data.success) return;
  const chips = $('source-chips');
  const filters = document.querySelectorAll('.src-filter');

  // Save current selections
  const saved = {};
  filters.forEach(s => { saved[s.id] = s.value; });

  if (!data.sources || data.sources.length === 0) {
    chips.innerHTML = '<div style="font-size:12px;color:var(--muted);text-align:center;padding:10px 0">No sources added yet</div>';
    filters.forEach(s => { s.innerHTML = '<option value="">All Sources</option>'; });
    return;
  }

  chips.innerHTML = data.sources.map(s => `
    <div class="source-chip">
      <div class="chip-icon ${TYPE_CLS[s.type] || 'chip-text'}">
        <i class="fas ${TYPE_ICONS[s.type] || 'fa-file'}"></i>
      </div>
      <span class="chip-label" title="${s.source}">${s.source.length > 42 ? s.source.substring(0,39)+'...' : s.source}</span>
    </div>`).join('');

  const opts = '<option value="">All Sources</option>' +
    data.sources.map(s => `<option value="${s.source}">${s.source.length>40?s.source.substring(0,37)+'...':s.source}</option>`).join('');
  filters.forEach(s => { s.innerHTML = opts; s.value = saved[s.id] || ''; });
}

// ── Tab Switcher ─────────────────────────────────────────────────────────
function setupTabs(barId, panelPrefix, onChange) {
  const bar = $(barId);
  if (!bar) return;
  bar.addEventListener('click', e => {
    const tab = e.target.closest('.tab, .tool-tab');
    if (!tab) return;
    bar.querySelectorAll('.tab, .tool-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const key = tab.dataset.tab || tab.dataset.tool;
    document.querySelectorAll(`[id^="${panelPrefix}"]`).forEach(p => {
      p.classList.toggle('active', p.id === `${panelPrefix}${key}`);
    });
    if (onChange) onChange(key);
  });
}

// ── Text/URL Content ──────────────────────────────────────────────────────
function setupTextBtns() {
  document.querySelectorAll('.add-text-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const type = btn.dataset.type;
      const inputId = type === 'text' ? 'custom-text' : (type === 'youtube' ? 'yt-url' : 'web-url');
      const val = $(inputId).value.trim();
      if (!val) return toast('Please enter a value.', true);
      loader(true, `Ingesting ${type} content…`);
      try {
        const d = await api('add-content', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ source: val, source_type: type })
        });
        toast(d.message, !d.success);
        if (d.success) { $(inputId).value = ''; refreshSources(); }
      } catch { toast('Network error.', true); }
      finally { loader(false); }
    });
  });
}

// ── File Uploads ──────────────────────────────────────────────────────────
function setupDropZone(dropId, inputId, progressId, fillId, label) {
  const zone  = $(dropId);
  const input = $(inputId);
  const prog  = $(progressId);
  const fill  = $(fillId);
  if (!zone) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); uploadFile(e.dataTransfer.files[0]); });
  input.addEventListener('change', () => { if (input.files[0]) uploadFile(input.files[0]); });

  async function uploadFile(file) {
    if (!file) return;
    prog.style.display = 'block';
    fill.style.width = '10%';

    const fd = new FormData();
    fd.append('file', file);

    // Fake progress while waiting
    let pct = 10;
    const tick = setInterval(() => { pct = Math.min(pct + 8, 85); fill.style.width = pct + '%'; }, 600);

    loader(true, `Processing ${label}… (${file.name})`);
    try {
      const r = await fetch(`${API}/upload-file`, { method: 'POST', body: fd });
      const d = await r.json();
      clearInterval(tick);
      fill.style.width = '100%';
      toast(d.message, !d.success);
      if (d.success) { input.value = ''; refreshSources(); }
    } catch { toast('Upload failed.', true); }
    finally {
      loader(false);
      setTimeout(() => { prog.style.display = 'none'; fill.style.width = '0%'; }, 1000);
    }
  }
}

// ── Results ───────────────────────────────────────────────────────────────
function addCard(title, icon, bodyFn) {
  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <div class="result-card-header">
      <div class="result-card-title"><i class="fas ${icon}"></i> ${title}</div>
      <div class="result-card-actions">
        <button class="icon-btn copy-btn" title="Copy"><i class="fas fa-copy"></i></button>
        <button class="icon-btn dl-btn" title="Download"><i class="fas fa-download"></i></button>
      </div>
    </div>
    <div class="result-card-body"></div>`;
  bodyFn(card.querySelector('.result-card-body'));

  // Copy plain text
  card.querySelector('.copy-btn').addEventListener('click', () => {
    const txt = card.querySelector('.result-card-body').innerText;
    navigator.clipboard.writeText(txt).then(() => toast('Copied!'));
  });
  // Download markdown
  card.querySelector('.dl-btn').addEventListener('click', () => {
    const txt = card.querySelector('.result-card-body').innerText;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([txt], {type:'text/markdown'}));
    a.download = title.replace(/\s+/g,'_') + '.md';
    a.click();
  });

  $('results').prepend(card);
  return card;
}

function renderMarkdown(text) {
  // Very lightweight markdown renderer
  return text
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>')
    .replace(/^# (.+)$/gm,'<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/`(.+?)`/g,'<code>$1</code>')
    .replace(/\n- (.+)/g,'\n<li>$1</li>')
    .replace(/\n/g,'<br>');
}

function textCard(title, icon, content) {
  addCard(title, icon, body => {
    body.innerHTML = `<div class="md-content">${renderMarkdown(content)}</div>`;
  });
}

async function mermaidCard(title, icon, code) {
  addCard(title, icon, body => {
    const wrap = document.createElement('div');
    wrap.className = 'mermaid-wrap';
    body.appendChild(wrap);
    (async () => {
      try {
        const { svg } = await mermaid.render(`mg-${Date.now()}`, code);
        wrap.innerHTML = svg;
      } catch {
        wrap.innerHTML = `<p style="color:var(--error);font-size:13px">Diagram render failed.</p><pre style="font-size:11px;overflow:auto">${code}</pre>`;
      }
    })();
  });
}

function flashcardsCard(title, cards) {
  addCard(title, 'fa-clone', body => {
    const grid = document.createElement('div');
    grid.className = 'flashcard-grid';
    cards.forEach(fc => {
      const card = document.createElement('div');
      card.className = 'flashcard';
      card.innerHTML = `
        <div class="flashcard-inner">
          <div class="flashcard-front">${fc.question}<span class="fc-hint">Click to reveal</span></div>
          <div class="flashcard-back">${fc.answer}</div>
        </div>`;
      card.addEventListener('click', () => card.classList.toggle('flipped'));
      grid.appendChild(card);
    });
    body.appendChild(grid);
  });
}

function storyboardCard(title, panels) {
  addCard(title, 'fa-film', body => {
    const grid = document.createElement('div');
    grid.className = 'storyboard-grid';
    panels.forEach((p, i) => {
      const div = document.createElement('div');
      div.className = 'storyboard-panel';
      div.innerHTML = `<div class="panel-num">Panel ${i+1}</div><i class="${p.icon||'fas fa-image'}"></i><p>${p.scene}</p>`;
      grid.appendChild(div);
    });
    body.appendChild(grid);
  });
}

function entitiesCard(entities) {
  addCard('Extracted Entities', 'fa-tags', body => {
    let hasAny = false;
    for (const [cat, list] of Object.entries(entities)) {
      if (list && list.length > 0) {
        hasAny = true;
        const g = document.createElement('div');
        g.className = 'entity-group';
        g.innerHTML = `<h4>${cat}</h4>` + list.map(e => `<span class="entity-pill">${e}</span>`).join('');
        body.appendChild(g);
      }
    }
    if (!hasAny) body.innerHTML = '<p style="font-size:13px;color:var(--muted)">No entities found.</p>';
  });
}

// ── Tool Actions ─────────────────────────────────────────────────────────
async function runQnA() {
  const q = $('qna-q').value.trim();
  const src = $('qna-src').value;
  if (!q) return toast('Enter a question.', true);
  loader(true, 'Searching knowledge base…');
  try {
    const d = await api('ask-question', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, source:src||null})
    });
    if (d.success) textCard(`Q: ${q}`, 'fa-question-circle', d.answer);
    else toast(d.message, true);
  } catch { toast('Error.', true); }
  finally { loader(false); }
}

async function runSummarize() {
  const src = $('sum-src').value;
  loader(true, 'Generating summary…');
  try {
    const d = await api('summarize', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({source:src||null})
    });
    if (d.success) textCard(d.title, 'fa-align-left', d.content);
    else toast(d.message, true);
  } catch { toast('Error.', true); }
  finally { loader(false); }
}

async function runTool(tool) {
  const topicEl = $(`${tool}-topic`);
  const srcEl = $(`${tool}-src`);
  const topic = topicEl ? topicEl.value.trim() : '';
  const src = srcEl ? srcEl.value : '';
  if (topicEl && !topic) return toast('Enter a topic.', true);

  const loaderMsgs = {flashcards:'Generating flashcards…', study_plan:'Building study plan…',
    mock_test:'Creating mock test…', mindmap:'Creating mind map…',
    flowchart:'Creating flowchart…', storyboard:'Creating storyboard…'};
  loader(true, loaderMsgs[tool] || 'Processing…');
  try {
    const d = await api('generate-tool-content', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({tool, topic, source:src||null})
    });
    if (!d.success) { toast(d.message, true); return; }

    if (tool === 'flashcards') flashcardsCard(d.title, d.content);
    else if (tool === 'storyboard') storyboardCard(d.title, d.content);
    else if (tool === 'mindmap' || tool === 'flowchart') mermaidCard(d.title, 'fa-project-diagram', d.mermaid_code);
    else textCard(d.title, tool==='study_plan'?'fa-calendar-check':'fa-clipboard-list', d.content);
  } catch { toast('Error.', true); }
  finally { loader(false); }
}

async function runEntities() {
  const src = $('entities-src').value;
  loader(true, 'Extracting entities…');
  try {
    const d = await api('extract-entities', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({source:src||null})
    });
    if (d.success) entitiesCard(d.entities);
    else toast(d.message, true);
  } catch { toast('Error.', true); }
  finally { loader(false); }
}

async function runTopics() {
  loader(true, 'Clustering topics…');
  try {
    const d = await api('cluster-topics');
    if (d.success) {
      addCard('Topic Cluster Visualization', 'fa-chart-pie', body => {
        body.innerHTML = `<img class="plot-img" src="${d.image}" alt="Topic clusters">`;
      });
    } else toast(d.message, true);
  } catch { toast('Error.', true); }
  finally { loader(false); }
}

// ── Chat ──────────────────────────────────────────────────────────────────
function appendChatMsg(role, text) {
  const msgs = $('chat-messages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = renderMarkdown(text);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function sendChat() {
  const input = $('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  const src = $('chat-src').value;
  input.value = '';
  appendChatMsg('user', msg);
  loader(true, 'Thinking…');
  try {
    const d = await api('chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:msg, source:src||null})
    });
    if (d.success) appendChatMsg('assistant', d.answer);
    else toast(d.message, true);
  } catch { toast('Network error.', true); }
  finally { loader(false); }
}

// ── Wire Up ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

  // Source tabs
  setupTabs('src-tabs', '', key => {
    document.querySelectorAll('[id$="-panel"]').forEach(p => {
      p.classList.toggle('active', p.id === `${key}-panel`);
    });
  });

  // Tool tabs — show/hide chat UI + tool panels
  setupTabs('tool-tabs', 'tp-', key => {
    const isChatTab = key === 'chat';
    $('chat-ui').style.display = isChatTab ? 'block' : 'none';
  });

  // Text add buttons
  setupTextBtns();

  // File drop zones
  setupDropZone('pdf-drop',   'pdf-input',   'pdf-progress',   'pdf-fill',   'PDF');
  setupDropZone('audio-drop', 'audio-input', 'audio-progress', 'audio-fill', 'Audio');
  setupDropZone('image-drop', 'image-input', 'image-progress', 'image-fill', 'Image');
  setupDropZone('video-drop', 'video-input', 'video-progress', 'video-fill', 'Video');

  // Tool buttons (data-action)
  document.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'qna')       runQnA();
      else if (action === 'summarize') runSummarize();
      else if (action === 'entities')  runEntities();
      else if (action === 'topics')    runTopics();
      else if (action === 'tool')  runTool(btn.dataset.tool);
    });
  });

  // Chat
  $('chat-send').addEventListener('click', sendChat);
  $('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) sendChat(); });
  $('clearChatBtn').addEventListener('click', async () => {
    const d = await api('chat/clear', {method:'POST'});
    $('chat-messages').innerHTML = '<div style="text-align:center;font-size:12px;color:var(--muted);padding:20px 0">Chat history cleared.</div>';
    toast(d.message);
  });

  // Clear all data
  $('clearBtn').addEventListener('click', async () => {
    if (!confirm('Clear ALL knowledge data? This cannot be undone.')) return;
    loader(true, 'Clearing…');
    const d = await api('clear-data', {method:'POST'});
    loader(false);
    toast(d.message, !d.success);
    if (d.success) refreshSources();
  });

  // Enter key for inline inputs
  ['qna-q','fc-topic','study_plan-topic','mock_test-topic','mindmap-topic','flowchart-topic','storyboard-topic'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('keydown', e => {
      if (e.key !== 'Enter') return;
      const btn = el.closest('.tool-panel')?.querySelector('[data-action]');
      if (btn) btn.click();
    });
  });

  refreshSources();
});
