import os
import re
import json
import uuid
import io
import base64
import numpy as np
import google.generativeai as genai
import faiss

# IMPORTANT: Set the matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup

# --- App Initialization & Configuration (Unchanged) ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
# Load API Key from environment variable. For local dev, you can replace the value.
# IMPORTANT: Remove hardcoded key before pushing to a public repository.
GEMINI_API_KEY =("AIzaSyAj9qY_NbobH5D7wkF1QjWWJ0Wb-agLEgI")
genai.configure(api_key=GEMINI_API_KEY)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
MODEL_DIMENSION = 384
KNOWLEDGE_BASE_FILE = 'knowledge_base.json'
INDEX_FILE = 'faiss_index.bin'

# --- KnowledgeBase Class (Unchanged) ---
class KnowledgeBase:
    def __init__(self):
        self.dimension = MODEL_DIMENSION
        self.knowledge_base = []
        self.index = faiss.IndexFlatL2(self.dimension)
        self.load()

    def load(self):
        try:
            if os.path.exists(KNOWLEDGE_BASE_FILE) and os.path.exists(INDEX_FILE):
                with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                self.index = faiss.read_index(INDEX_FILE)
        except Exception as e:
            print(f"Error loading knowledge base: {e}.")
            self.clear()

    def save(self):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2)
            if self.index.ntotal > 0:
                faiss.write_index(self.index, INDEX_FILE)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

    def clear(self):
        self.knowledge_base = []
        self.index = faiss.IndexFlatL2(self.dimension)
        if os.path.exists(KNOWLEDGE_BASE_FILE): os.remove(KNOWLEDGE_BASE_FILE)
        if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
        print("Knowledge base cleared.")

kb = KnowledgeBase()

# --- Helper Functions (Unchanged) ---
def crawl_url(url: str) -> str:
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        return " ".join(soup.stripped_strings)
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return None

def get_youtube_transcript(video_url: str) -> str:
    try:
        video_id_match = re.search(r'(?<=v=)[^&]+|(?<=be/)[^?]+', video_url)
        video_id = video_id_match.group(0) if video_id_match else None
        if not video_id: return None
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d['text'] for d in transcript_list])
    except Exception as e:
        print(f"Error fetching YouTube transcript: {e}")
        return None

def chunk_text(text: str, max_length: int = 400) -> list:
    words = re.split(r'\s+', text)
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def update_knowledge_base(text: str, source: str, source_type: str):
    if not text: return False
    chunks = chunk_text(text)
    if not chunks: return False
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    for chunk in chunks:
        kb.knowledge_base.append({"id": str(uuid.uuid4()), "text": chunk, "source": source, "type": source_type})
    kb.index.add(embeddings)
    kb.save()
    return True

def retrieve_chunks(query: str, source: str = None, k: int = 5) -> list:
    if kb.index.ntotal == 0: return []
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    indices_to_search = list(range(kb.index.ntotal))
    if source:
        indices_to_search = [i for i, item in enumerate(kb.knowledge_base) if item["source"] == source]
        if not indices_to_search: return []
    search_index = kb.index
    if source and len(indices_to_search) > 0:
        source_embeddings = np.array([kb.index.reconstruct(i) for i in indices_to_search]).astype('float32')
        search_index = faiss.IndexFlatL2(kb.dimension)
        search_index.add(source_embeddings)
    distances, indices = search_index.search(query_embedding.astype('float32'), min(k, search_index.ntotal))
    original_indices = [indices_to_search[i] for i in indices[0]] if source else indices[0]
    return [kb.knowledge_base[i]["text"] for i in original_indices]

def clean_llm_output(raw_text: str, output_type: str) -> str:
    if output_type == 'mermaid':
        match = re.search(r'```mermaid(.*?)```', raw_text, re.DOTALL)
        return match.group(1).strip() if match else raw_text.strip()
    if output_type == 'json':
        match = re.search(r'```json(.*?)```', raw_text, re.DOTALL)
        if match: return match.group(1).strip()
        match = re.search(r'(\[.*\]|\{.*\})', raw_text, re.DOTALL)
        return match.group(0) if match else "{}"
    return raw_text

# --- NEW: Mermaid Syntax Sanitizer ---
def validate_and_sanitize_mermaid(code: str) -> str:
    """
    Automatically quotes node text containing special characters to prevent Mermaid syntax errors.
    """
    sanitized_lines = []
    # This regex finds a node ID and the text within its brackets, e.g., E[text here]
    node_pattern = re.compile(r'(\w+)(\[.+?\]|\(.+?\)|{.+?})')

    for line in code.split('\n'):
        match = node_pattern.search(line)
        if match:
            node_id_part = line[:match.start(2)]
            node_content = match.group(2)
            node_text = node_content[1:-1] # Get text inside brackets
            
            # If text has special characters and is not already quoted, quote it
            if re.search(r'[()\[\]{}:+]', node_text) and not (node_text.startswith('"') and node_text.endswith('"')):
                # Escape existing quotes inside the text before wrapping
                escaped_text = node_text.replace('"', '&quot;')
                sanitized_line = f'{node_id_part}["{escaped_text}"]'
                sanitized_lines.append(sanitized_line)
            else:
                sanitized_lines.append(line)
        else:
            sanitized_lines.append(line)
            
    return "\n".join(sanitized_lines)


# --- API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/add-content', methods=['POST'])
def add_content():
    # ... (endpoint remains the same) ...
    data = request.get_json()
    source, source_type = data.get('source'), data.get('source_type')
    text = ""
    if source_type == "youtube": text = get_youtube_transcript(source)
    elif source_type == "text": text = source
    elif source_type == "web": text = crawl_url(source)
    if not text: return jsonify({"success": False, "message": f"Failed to extract content from {source_type}."})
    success = update_knowledge_base(text, source, source_type)
    return jsonify({"success": success, "message": "Content added successfully." if success else "Failed to process content."})

@app.route('/api/ask-question', methods=['POST'])
def ask_question_route():
    # ... (endpoint remains the same) ...
    data = request.get_json()
    question, source = data.get('question'), data.get('source')
    chunks = retrieve_chunks(question, source)
    if not chunks: return jsonify({"success": True, "question": question, "answer": "I don't have enough information to answer that. Please add relevant content first."})
    context = "\n\n".join(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based *only* on the provided context. Format the answer clearly using markdown."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return jsonify({"success": True, "question": question, "answer": response.text.strip()})

@app.route('/api/generate-tool-content', methods=['POST'])
def generate_tool_content_route():
    data = request.get_json()
    tool, topic, source = data.get('tool'), data.get('topic'), data.get('source')
    if not topic: return jsonify({"success": False, "message": "A topic is required."})

    chunks = retrieve_chunks(topic, source)
    if not chunks: return jsonify({"success": False, "message": "Not enough content for this topic."})
    context = "\n\n".join(chunks)
    
    # --- FIX: Stronger Prompts for Mermaid ---
    mermaid_prompt_injection = """
    IMPORTANT RULES:
    1. Use simple, single-word node IDs (e.g., A, B, C1, C2).
    2. CRITICAL: If any node text contains special characters like (), [], {}, :, +, or others, you MUST enclose the entire text in double quotes. For example: `A["Node with (parentheses)"]`.
    """

    prompts = {
        "mock_test": f"Based on the context, generate a {5} question multiple-choice mock test on '{topic}'. Provide 4 options (A, B, C, D) per question and list the correct answers in a separate 'Answers' section at the end.\n\nContext:\n{context}",
        "mindmap": f"Create a Mermaid.js 'mindmap' syntax about '{topic}' based on the context.\n\n{mermaid_prompt_injection}\n\nContext:\n{context}\n\nProvide only the Mermaid code inside a ```mermaid block.",
        "flowchart": f"Create a Mermaid.js 'graph TD' syntax for a flowchart illustrating '{topic}' based on the context.\n\n{mermaid_prompt_injection}\n\nContext:\n{context}\n\nProvide only the Mermaid code inside a ```mermaid block.",
        "storyboard": f"Create a 3-panel storyboard about '{topic}'. For each panel, provide a 'scene' description and a relevant 'icon' from Font Awesome (e.g., 'fas fa-lightbulb'). Return ONLY a valid JSON array of objects with 'scene' and 'icon' keys, inside a ```json block.\n\nContext:\n{context}"
    }
    
    if tool not in prompts:
        return jsonify({"success": False, "message": "Invalid tool specified."})

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompts[tool])
    
    if tool in ["mindmap", "flowchart"]:
        raw_mermaid = clean_llm_output(response.text, 'mermaid')
        # --- FIX: Sanitize Mermaid output before sending to frontend ---
        sanitized_mermaid = validate_and_sanitize_mermaid(raw_mermaid)
        return jsonify({"success": True, "type": tool, "title": f"{tool.capitalize()} for {topic}", "mermaid_code": sanitized_mermaid})
    elif tool == "storyboard":
        try:
            storyboard_data = json.loads(clean_llm_output(response.text, 'json'))
            return jsonify({"success": True, "type": "storyboard", "title": f"Storyboard for {topic}", "content": storyboard_data})
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "Failed to generate a valid storyboard from the AI."})
    else: # mock_test
        return jsonify({"success": True, "type": "mock_test", "title": f"Mock Test on {topic}", "content": response.text.strip()})


# ... (All other endpoints like /api/extract-entities, /api/cluster-topics, etc., remain unchanged) ...
@app.route('/api/extract-entities', methods=['POST'])
def extract_entities_route():
    data = request.get_json()
    source = data.get('source')
    text = " ".join(item['text'] for item in kb.knowledge_base if not source or item['source'] == source)
    if not text: return jsonify({"success": False, "message": "No content available to analyze."})
    prompt = f"From the text, extract 'Person', 'Organization', and 'Location' named entities. Return a JSON object with these keys, each with a list of unique names. If empty, return an empty list.\n\nText:\n{text}\n\nProvide ONLY the valid JSON object inside a ```json block."
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    try:
        entities = json.loads(clean_llm_output(response.text, 'json'))
        return jsonify({"success": True, "entities": entities})
    except json.JSONDecodeError:
        return jsonify({"success": False, "message": "Failed to parse entities from the AI response."})

@app.route('/api/cluster-topics', methods=['GET'])
def cluster_topics_route():
    if kb.index.ntotal < 5:
        return jsonify({"success": False, "message": "Need at least 5 content chunks to create topics."})
    embeddings = np.array([kb.index.reconstruct(i) for i in range(kb.index.ntotal)]).astype('float32')
    num_clusters = min(5, kb.index.ntotal // 2)
    if num_clusters < 2:
        return jsonify({"success": False, "message": "Not enough diverse content to form clusters."})
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    cluster_labels = []
    for i in range(num_clusters):
        sample_texts = " ".join([kb.knowledge_base[int(j)]["text"] for j in np.where(labels == i)[0][:3]])
        prompt = f"Based on this text, provide a short, 2-3 word topic label: {sample_texts}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cluster_labels.append(re.sub(r'["*]', '', response.text.strip()))
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=[cluster_labels[l] for l in labels], palette="viridis", s=120, alpha=0.8, ax=ax, edgecolor='w', linewidth=0.5)
    ax.set_title("2D Visualization of Content Topics", fontsize=16, color='white')
    ax.set_xlabel("Principal Component 1", color='white')
    ax.set_ylabel("Principal Component 2", color='white')
    ax.tick_params(colors='white', which='both')
    legend = ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({"success": True, "image": f"data:image/png;base64,{img_base64}"})

@app.route('/api/get-sources', methods=['GET'])
def get_sources_route():
    return jsonify({"success": True, "sources": sorted(list(set(item['source'] for item in kb.knowledge_base)))})

@app.route('/api/clear-data', methods=['POST'])
def clear_data_route():
    kb.clear()
    return jsonify({"success": True, "message": "Knowledge base cleared."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)