import os
import re
import json
import uuid
import io
import base64
import time
import tempfile
import mimetypes
import numpy as np
import google.generativeai as genai

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import requests
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import faiss
from dotenv import load_dotenv

# ─── App Init ────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("FLASK_SECRET", "kf-v2-secret-change-me")
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# ─── Model & Config ──────────────────────────────────────────────────────────
EMBED_MODEL    = "models/text-embedding-004"   # Gemini 2.0 embedding, 768-dim
GEN_MODEL      = "gemini-2.0-flash"            # Latest flash model
EMBED_DIM      = 768
CHUNK_SIZE     = 400   # words per chunk
CHUNK_OVERLAP  = 50    # overlap words
TOP_K_COARSE   = 12    # retrieve this many before re-rank
TOP_K_FINAL    = 5     # keep this many after re-rank

KB_FILE    = 'knowledge_base.json'
INDEX_FILE = 'faiss_index.bin'


# ─── Gemini Embedding Helper ──────────────────────────────────────────────────
def gemini_embed(texts: list[str]) -> np.ndarray:
    """Batch embed texts using Gemini text-embedding-004."""
    vectors = []
    # API allows up to 100 texts per call
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="retrieval_document"
        )
        vectors.extend(result['embedding'])
    return np.array(vectors, dtype='float32')


def gemini_embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'], dtype='float32').reshape(1, -1)


# ─── Knowledge Base ───────────────────────────────────────────────────────────
class KnowledgeBase:
    def __init__(self):
        self.dim = EMBED_DIM
        self.chunks: list[dict] = []   # {id, text, source, source_type, metadata}
        self.index = faiss.IndexFlatIP(self.dim)  # Inner product (cosine-like with normalized vecs)
        self.load()

    def load(self):
        try:
            if os.path.exists(KB_FILE) and os.path.exists(INDEX_FILE):
                with open(KB_FILE, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                self.index = faiss.read_index(INDEX_FILE)
                print(f"✓ Loaded {len(self.chunks)} chunks from disk.")
        except Exception as e:
            print(f"Warn: Could not load existing KB ({e}). Starting fresh.")
            self.clear(save=False)

    def save(self):
        try:
            with open(KB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            if self.index.ntotal > 0:
                faiss.write_index(self.index, INDEX_FILE)
        except Exception as e:
            print(f"Error saving KB: {e}")

    def clear(self, save=True):
        self.chunks = []
        self.index = faiss.IndexFlatIP(self.dim)
        if save:
            if os.path.exists(KB_FILE):   os.remove(KB_FILE)
            if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
        print("Knowledge base cleared.")

    def add_chunks(self, chunks: list[str], source: str, source_type: str, metadata: dict = None):
        if not chunks:
            return False
        vectors = gemini_embed(chunks)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        for chunk, vec in zip(chunks, vectors):
            self.chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": source,
                "source_type": source_type,
                "metadata": metadata or {}
            })
        self.index.add(vectors)
        self.save()
        return True

    def hybrid_search(self, query: str, source_filter: str = None, k: int = TOP_K_FINAL) -> list[str]:
        """Hybrid BM25 + semantic search with optional source filter, then re-rank."""
        if self.index.ntotal == 0:
            return []

        # Filter chunks by source if requested
        pool = self.chunks if not source_filter else [
            c for c in self.chunks if c["source"] == source_filter
        ]
        if not pool:
            return []

        pool_texts = [c["text"] for c in pool]
        pool_indices = [self.chunks.index(c) for c in pool]  # original indices

        # ── Semantic search ───────────────────────────────────────────────────
        q_vec = gemini_embed_query(query)
        faiss.normalize_L2(q_vec)

        if source_filter and pool_indices:
            # Build a sub-index for just the filtered chunks
            sub_vecs = np.array([self.index.reconstruct(i) for i in pool_indices], dtype='float32')
            sub_idx = faiss.IndexFlatIP(self.dim)
            sub_idx.add(sub_vecs)
            sem_k = min(TOP_K_COARSE, len(pool_indices))
            D, I = sub_idx.search(q_vec, sem_k)
            sem_indices = [pool_indices[i] for i in I[0]]
            sem_scores  = {pool_indices[i]: float(D[0][j]) for j, i in enumerate(I[0])}
        else:
            sem_k = min(TOP_K_COARSE, self.index.ntotal)
            D, I = self.index.search(q_vec, sem_k)
            sem_indices = list(I[0])
            sem_scores  = {int(I[0][j]): float(D[0][j]) for j in range(len(I[0]))}

        # ── BM25 search ───────────────────────────────────────────────────────
        tokenized = [t.lower().split() for t in pool_texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores_raw = bm25.get_scores(query.lower().split())
        bm25_max = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        # Map pool index → original chunk index
        bm25_scores = {pool_indices[i]: bm25_scores_raw[i] / bm25_max for i in range(len(pool_indices))}

        # ── Combine (RRF-style) ───────────────────────────────────────────────
        # Collect all candidates
        candidate_idxs = set(sem_indices) | set(bm25_scores.keys())
        combined = {}
        for idx in candidate_idxs:
            s = sem_scores.get(idx, 0.0) * 0.7 + bm25_scores.get(idx, 0.0) * 0.3
            combined[idx] = s

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:TOP_K_COARSE]
        top_chunks = [self.chunks[i]["text"] for i, _ in ranked if i < len(self.chunks)]

        # ── LLM Re-ranking ────────────────────────────────────────────────────
        if len(top_chunks) > TOP_K_FINAL:
            numbered = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(top_chunks))
            rerank_prompt = (
                f"Given the query: '{query}'\n\n"
                f"Here are {len(top_chunks)} candidate passages:\n{numbered}\n\n"
                f"Return ONLY a JSON array of the {TOP_K_FINAL} passage numbers (1-indexed) "
                f"most relevant to the query, ordered by relevance. Example: [2,5,1,3,4]"
            )
            try:
                model = genai.GenerativeModel(GEN_MODEL)
                resp  = model.generate_content(rerank_prompt)
                nums  = json.loads(clean_llm_output(resp.text, 'json'))
                top_chunks = [top_chunks[n - 1] for n in nums if 1 <= n <= len(top_chunks)]
            except Exception:
                top_chunks = top_chunks[:TOP_K_FINAL]

        return top_chunks[:TOP_K_FINAL]


kb = KnowledgeBase()


# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window word chunker with overlap."""
    words = re.split(r'\s+', text.strip())
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─── Content Ingestion Helpers ────────────────────────────────────────────────
def crawl_url(url: str) -> str:
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return None


def get_youtube_transcript(video_url: str) -> str:
    try:
        match = re.search(r'(?<=v=)[^&]+|(?<=be/)[^?]+', video_url)
        vid_id = match.group(0) if match else None
        if not vid_id:
            return None
        transcript = YouTubeTranscriptApi.get_transcript(vid_id)
        return " ".join(d['text'] for d in transcript)
    except Exception as e:
        print(f"Error fetching YouTube transcript: {e}")
        return None


def ingest_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        raw = "\n".join(texts).strip()
        if len(raw) < 100:
            # Likely a scanned PDF — use Gemini vision on each page
            return ingest_pdf_via_gemini(doc)
        return raw
    except ImportError:
        return ingest_pdf_via_gemini_bytes(data)
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None


def ingest_pdf_via_gemini_bytes(data: bytes) -> str:
    """Fallback: send raw PDF bytes to Gemini if PyMuPDF unavailable."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        uploaded = genai.upload_file(tmp_path, mime_type='application/pdf')
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content([
            uploaded,
            "Extract and return all the text content from this PDF document. Preserve structure."
        ])
        os.unlink(tmp_path)
        genai.delete_file(uploaded.name)
        return resp.text
    except Exception as e:
        print(f"Gemini PDF fallback error: {e}")
        return None


def ingest_file_via_gemini(data: bytes, mime_type: str, prompt: str) -> str:
    """Upload any file to Gemini Files API and extract content."""
    ext_map = {
        'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/x-m4a': '.m4a',
        'audio/ogg': '.ogg', 'video/mp4': '.mp4', 'video/webm': '.webm',
        'image/jpeg': '.jpg', 'image/png': '.png', 'image/gif': '.gif',
        'image/webp': '.webp', 'application/pdf': '.pdf'
    }
    suffix = ext_map.get(mime_type, '.bin')
    tmp_path = None
    uploaded = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        uploaded = genai.upload_file(tmp_path, mime_type=mime_type)
        # Wait for processing
        for _ in range(20):
            f = genai.get_file(uploaded.name)
            if f.state.name == "ACTIVE":
                break
            elif f.state.name == "FAILED":
                raise RuntimeError("Gemini file processing failed.")
            time.sleep(3)
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content([uploaded, prompt])
        return resp.text
    except Exception as e:
        print(f"Gemini file ingestion error ({mime_type}): {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if uploaded:
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                pass


# ─── Output Cleaning ──────────────────────────────────────────────────────────
def clean_llm_output(raw: str, output_type: str) -> str:
    if output_type == 'mermaid':
        m = re.search(r'```mermaid(.*?)```', raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()
    if output_type == 'json':
        m = re.search(r'```json(.*?)```', raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'(\[.*\]|\{.*\})', raw, re.DOTALL)
        return m.group(0) if m else "{}"
    return raw


def sanitize_mermaid(code: str) -> str:
    sanitized = []
    node_pat = re.compile(r'(\w+)(\[.+?\]|\(.+?\)|{.+?})')
    for line in code.split('\n'):
        m = node_pat.search(line)
        if m:
            before = line[:m.start(2)]
            text   = m.group(2)[1:-1]
            if re.search(r'[()\[\]{}:+]', text) and not (text.startswith('"') and text.endswith('"')):
                text = text.replace('"', '&quot;')
                sanitized.append(f'{before}["{text}"]')
            else:
                sanitized.append(line)
        else:
            sanitized.append(line)
    return "\n".join(sanitized)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/add-content', methods=['POST'])
def add_content():
    data   = request.get_json()
    source = data.get('source', '').strip()
    stype  = data.get('source_type', '')
    text   = ""

    if stype == "youtube":
        text = get_youtube_transcript(source)
    elif stype == "text":
        text = source
    elif stype == "web":
        text = crawl_url(source)
    else:
        return jsonify({"success": False, "message": f"Unknown source type: {stype}"})

    if not text:
        return jsonify({"success": False, "message": f"Failed to extract content from {stype}."})

    chunks  = chunk_text(text)
    success = kb.add_chunks(chunks, source, stype)
    return jsonify({
        "success": success,
        "message": f"Added {len(chunks)} chunks from {stype}." if success else "Failed to process content."
    })


@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Handle multimodal file uploads: PDF, audio, video, image."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file provided."})

    f         = request.files['file']
    filename  = f.filename or "upload"
    data      = f.read()
    mime_type = f.mimetype or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    source    = filename

    text = None
    source_type = "file"

    if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
        source_type = "pdf"
        text = ingest_pdf_bytes(data)

    elif mime_type.startswith('audio/') or any(filename.lower().endswith(x) for x in ['.mp3', '.wav', '.m4a', '.ogg']):
        source_type = "audio"
        text = ingest_file_via_gemini(
            data, mime_type,
            "Transcribe this audio lecture completely and accurately. Preserve all key concepts and details."
        )

    elif mime_type.startswith('video/') or any(filename.lower().endswith(x) for x in ['.mp4', '.webm', '.mov']):
        source_type = "video"
        text = ingest_file_via_gemini(
            data, mime_type,
            "Analyze this educational video. Transcribe all speech and describe key visual content, diagrams, and concepts shown."
        )

    elif mime_type.startswith('image/') or any(filename.lower().endswith(x) for x in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
        source_type = "image"
        text = ingest_file_via_gemini(
            data, mime_type,
            "Extract all text (OCR) from this image. Also describe any diagrams, charts, or visual content in detail."
        )

    elif filename.lower().endswith('.txt') or mime_type == 'text/plain':
        source_type = "text"
        text = data.decode('utf-8', errors='replace')

    else:
        return jsonify({"success": False, "message": f"Unsupported file type: {mime_type}"})

    if not text:
        return jsonify({"success": False, "message": "Failed to extract content from file."})

    chunks  = chunk_text(text)
    success = kb.add_chunks(chunks, source, source_type, metadata={"filename": filename})
    return jsonify({
        "success": success,
        "message": f"Added {len(chunks)} chunks from '{filename}'." if success else "Failed to process file."
    })


@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    data     = request.get_json()
    question = data.get('question', '').strip()
    source   = data.get('source') or None

    chunks = kb.hybrid_search(question, source_filter=source)
    if not chunks:
        return jsonify({"success": True, "question": question,
                        "answer": "I don't have enough information to answer that. Please add relevant content first."})

    context = "\n\n---\n\n".join(chunks)
    prompt  = (
        f"You are a helpful educational AI assistant. Use ONLY the provided context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a clear, well-structured answer in markdown. "
        f"If the context doesn't contain the answer, say so honestly."
    )
    model = genai.GenerativeModel(GEN_MODEL)
    resp  = model.generate_content(prompt)
    return jsonify({"success": True, "question": question, "answer": resp.text.strip()})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Multi-turn conversational Q&A with session history."""
    data     = request.get_json()
    message  = data.get('message', '').strip()
    source   = data.get('source') or None

    if 'chat_history' not in session:
        session['chat_history'] = []

    chunks  = kb.hybrid_search(message, source_filter=source)
    context = "\n\n---\n\n".join(chunks) if chunks else "No specific context retrieved."

    history_text = "\n".join(
        f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}"
        for h in session['chat_history'][-6:]  # keep last 3 turns
    )

    prompt = (
        f"You are a knowledgeable educational AI assistant. Use the context below to help the user.\n\n"
        f"Knowledge Context:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User: {message}\n\n"
        f"Respond clearly and helpfully. Use markdown where appropriate."
    )

    model = genai.GenerativeModel(GEN_MODEL)
    resp  = model.generate_content(prompt)
    answer = resp.text.strip()

    session['chat_history'].append({"role": "user",      "content": message})
    session['chat_history'].append({"role": "assistant", "content": answer})
    session.modified = True

    return jsonify({"success": True, "message": message, "answer": answer,
                    "history": session['chat_history']})


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    session.pop('chat_history', None)
    return jsonify({"success": True, "message": "Chat history cleared."})


@app.route('/api/summarize', methods=['POST'])
def summarize():
    data   = request.get_json()
    source = data.get('source') or None

    pool = [c for c in kb.chunks if not source or c['source'] == source]
    if not pool:
        return jsonify({"success": False, "message": "No content to summarize."})

    # Summarize top representative chunks
    texts = " ".join(c['text'] for c in pool[:30])
    prompt = (
        f"Create a comprehensive, well-structured summary of the following educational content. "
        f"Use headings, bullet points, and key takeaways. Format in markdown.\n\n{texts}"
    )
    model = genai.GenerativeModel(GEN_MODEL)
    resp  = model.generate_content(prompt)
    return jsonify({"success": True, "type": "summary",
                    "title": f"Summary{' of ' + source if source else ''}",
                    "content": resp.text.strip()})


@app.route('/api/generate-tool-content', methods=['POST'])
def generate_tool_content():
    data   = request.get_json()
    tool   = data.get('tool')
    topic  = data.get('topic', '').strip()
    source = data.get('source') or None

    if not topic:
        return jsonify({"success": False, "message": "A topic is required."})

    chunks = kb.hybrid_search(topic, source_filter=source)
    if not chunks:
        return jsonify({"success": False, "message": "No relevant content found for this topic."})

    context = "\n\n".join(chunks)

    mermaid_rules = (
        "MERMAID RULES: 1) Use simple single-word IDs (A, B, C1). "
        "2) Wrap node text with special chars in double quotes: A[\"Text (with parens)\"]."
    )

    prompts = {
        "mock_test": (
            f"Based on the context, generate a 5-question multiple-choice test on '{topic}'. "
            f"Provide 4 options (A-D) per question. End with an 'Answers:' section.\n\nContext:\n{context}"
        ),
        "mindmap": (
            f"Create a Mermaid.js 'mindmap' syntax about '{topic}' from the context.\n"
            f"{mermaid_rules}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
        ),
        "flowchart": (
            f"Create a Mermaid.js 'graph TD' flowchart for '{topic}' from the context.\n"
            f"{mermaid_rules}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
        ),
        "storyboard": (
            f"Create a 3-panel storyboard about '{topic}'. For each panel give 'scene' (description) "
            f"and 'icon' (Font Awesome class like 'fas fa-lightbulb'). "
            f"Return ONLY a valid JSON array inside ```json ... ```.\n\nContext:\n{context}"
        ),
        "flashcards": (
            f"Generate 8 flashcards about '{topic}' from the context. "
            f"Each flashcard has a 'question' (the concept/term) and 'answer' (explanation). "
            f"Return ONLY a JSON array: [{{\"question\":\"...\",\"answer\":\"...\"}}] inside ```json ... ```.\n\nContext:\n{context}"
        ),
        "study_plan": (
            f"Create a structured 4-week study plan to master '{topic}' based on the context. "
            f"Format as markdown with week headings, daily goals, and resources. "
            f"Be specific and actionable.\n\nContext:\n{context}"
        ),
    }

    if tool not in prompts:
        return jsonify({"success": False, "message": "Invalid tool specified."})

    model = genai.GenerativeModel(GEN_MODEL)
    resp  = model.generate_content(prompts[tool])

    if tool in ('mindmap', 'flowchart'):
        raw  = clean_llm_output(resp.text, 'mermaid')
        safe = sanitize_mermaid(raw)
        return jsonify({"success": True, "type": tool,
                        "title": f"{tool.replace('_',' ').title()} — {topic}",
                        "mermaid_code": safe})

    if tool in ('storyboard', 'flashcards'):
        try:
            content = json.loads(clean_llm_output(resp.text, 'json'))
            return jsonify({"success": True, "type": tool,
                            "title": f"{tool.replace('_',' ').title()} — {topic}",
                            "content": content})
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "AI returned invalid JSON. Please try again."})

    # mock_test, study_plan
    return jsonify({"success": True, "type": tool,
                    "title": f"{tool.replace('_',' ').title()} — {topic}",
                    "content": resp.text.strip()})


@app.route('/api/extract-entities', methods=['POST'])
def extract_entities():
    data   = request.get_json()
    source = data.get('source') or None
    text   = " ".join(c['text'] for c in kb.chunks if not source or c['source'] == source)
    if not text:
        return jsonify({"success": False, "message": "No content available."})

    prompt = (
        f"Extract named entities from the text below. "
        f"Return a JSON object with keys 'Person', 'Organization', 'Location', 'Concept' — "
        f"each containing a list of unique names. Use ```json ... ```.\n\nText:\n{text[:4000]}"
    )
    model = genai.GenerativeModel(GEN_MODEL)
    resp  = model.generate_content(prompt)
    try:
        entities = json.loads(clean_llm_output(resp.text, 'json'))
        return jsonify({"success": True, "entities": entities})
    except json.JSONDecodeError:
        return jsonify({"success": False, "message": "Failed to parse entities."})


@app.route('/api/cluster-topics', methods=['GET'])
def cluster_topics():
    if kb.index.ntotal < 5:
        return jsonify({"success": False, "message": "Need at least 5 content chunks to cluster."})

    embeddings = np.array([kb.index.reconstruct(i) for i in range(kb.index.ntotal)], dtype='float32')
    num_clusters = min(5, kb.index.ntotal // 2)
    if num_clusters < 2:
        return jsonify({"success": False, "message": "Not enough diverse content."})

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    cluster_labels = []
    model = genai.GenerativeModel(GEN_MODEL)
    for i in range(num_clusters):
        sample = " ".join(kb.chunks[int(j)]["text"] for j in np.where(labels == i)[0][:3])
        resp = model.generate_content(f"Give a concise 2-3 word topic label for: {sample[:500]}")
        cluster_labels.append(re.sub(r'[\"*]', '', resp.text.strip()))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    palette = ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"]
    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=[cluster_labels[l] for l in labels],
        palette=palette[:num_clusters], s=120, alpha=0.85, ax=ax,
        edgecolor='white', linewidth=0.5
    )
    ax.set_title("Content Topic Clusters", fontsize=16, fontweight='bold', color='#1e293b')
    ax.set_xlabel("Component 1", color='#475569')
    ax.set_ylabel("Component 2", color='#475569')
    legend = ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return jsonify({"success": True, "image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"})


@app.route('/api/get-sources', methods=['GET'])
def get_sources():
    sources = []
    seen = set()
    for c in kb.chunks:
        if c['source'] not in seen:
            seen.add(c['source'])
            sources.append({"source": c['source'], "type": c.get('source_type', 'text')})
    return jsonify({"success": True, "sources": sources})


@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    kb.clear()
    return jsonify({"success": True, "message": "Knowledge base cleared."})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)