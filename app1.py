from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import json
import os
import uuid
import tempfile

app = Flask(__name__, template_folder='templates')
# Update CORS to match frontend origin
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBLi6kgtmc3M9SzZ6aA15tDMmmLcKokQ6s"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Persistent knowledge base storage
KNOWLEDGE_BASE_FILE = 'knowledge_base.json'
INDEX_FILE = 'faiss_index.bin'

class KnowledgeBase:
    def __init__(self):
        self.dimension = 384
        self.knowledge_base = []
        self.index = None
        self.load()

    def load(self):
        try:
            if os.path.exists(KNOWLEDGE_BASE_FILE):
                with open(KNOWLEDGE_BASE_FILE, 'r') as f:
                    self.knowledge_base = json.load(f)
            if self.knowledge_base and os.path.exists(INDEX_FILE):
                self.index = faiss.read_index(INDEX_FILE)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                if self.knowledge_base:
                    embeddings = embedder.encode([item["text"] for item in self.knowledge_base], convert_to_numpy=True)
                    self.index.add(embeddings)
                    self.save()
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)

    def save(self):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            if self.index.ntotal > 0:
                faiss.write_index(self.index, INDEX_FILE)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")

kb = KnowledgeBase()

# Initialize Crawler for YouTube
crawler = WebCrawler()
crawler.warmup()
extraction_strategy = JsonCssExtractionStrategy(
    schema={"text_content": {"css": "p", "type": "string"}}
)

def crawl_url(url: str) -> str:
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        driver.quit()
        return text_content if text_content else None
    except Exception as e:
        print(f"Error crawling {url}: {str(e)}")
        return None

def crawl_youtube(video_url: str) -> str:
    try:
        result = crawler.run(url=video_url, extraction_strategy=extraction_strategy, bypass_cache=True)
        if result.success and result.extracted_content:
            return " ".join([item["text_content"] for item in result.extracted_content if "text_content" in item])
        return None
    except Exception as e:
        print(f"Error crawling YouTube {video_url}: {str(e)}")
        return None

def get_youtube_transcript(video_url: str) -> str:
    video_id = video_url.split("v=")[-1].split("&")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def chunk_text(text: str, max_length: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def update_knowledge_base(text: str, source: str, source_type: str):
    if not text:
        return False
    
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    for i, chunk in enumerate(chunks):
        kb.knowledge_base.append({
            "text": chunk,
            "source": source,
            "type": source_type,
            "timestamp": str(uuid.uuid4())
        })
    
    if embeddings.size > 0:
        kb.index.add(embeddings)
        kb.save()
        return True
    return False

def process_input(source: str, source_type: str = "web"):
    if source_type == "web":
        text = crawl_url(source)
        if text:
            return update_knowledge_base(text, source, "web")
    elif source_type == "video":
        page_text = crawl_youtube(source)
        transcript = get_youtube_transcript(source)
        text = f"{page_text or ''} {transcript or ''}".strip()
        if text:
            return update_knowledge_base(text, source, "video")
    elif source_type == "user":
        return update_knowledge_base(source, "User Input", "text")
    return False

def retrieve_chunks(query: str, source: str = None) -> list:
    if not kb.knowledge_base or kb.index.ntotal == 0:
        return []
    
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    k = min(5, kb.index.ntotal)
    
    if source:
        source_chunks = [item for item in kb.knowledge_base if item["source"] == source]
        if not source_chunks:
            return []
        embeddings = embedder.encode([item["text"] for item in source_chunks], convert_to_numpy=True)
        temp_index = faiss.IndexFlatL2(kb.dimension)
        temp_index.add(embeddings)
        distances, indices = temp_index.search(query_embedding, min(k, len(source_chunks)))
        return [source_chunks[i]["text"] for i in indices[0] if i < len(source_chunks)]
    else:
        distances, indices = kb.index.search(query_embedding, k)
        return [kb.knowledge_base[i]["text"] for i in indices[0] if i < len(kb.knowledge_base)]

def generate_answer(query: str, source: str = None) -> str:
    chunks = retrieve_chunks(query, source)
    if not chunks:
        return "I don't have enough information to answer that question."
    
    context = "\n".join(chunks)
    prompt = f"""
    You are an expert educational assistant. Based on the following context, provide a clear, concise, and accurate answer to the question in pointers and in a presentable format,don't use astericks..use minimal emojis.

    Context: {context}
    Question: {query}
    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_summary(source: str = None) -> dict:
    if not kb.knowledge_base:
        return {"title": "Summary", "content": "No content available to summarize."}
    
    if source:
        chunks = [item["text"] for item in kb.knowledge_base if item["source"] == source][:3]
        title = f"Summary for {source}"
    else:
        chunks = [item["text"] for item in kb.knowledge_base][:3]
        title = "Summary of All Content"
    
    context = "\n".join(chunks)
    prompt = f"""
    You are a skilled summarizer. Provide a concise summary of the following content in 2-3 sentences.

    Content: {context}
    Summary:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return {"title": title, "content": response.text.strip()}

def generate_mock_test(topic: str, source: str = None, num_questions: int = 5) -> dict:
    chunks = retrieve_chunks(topic, source)
    if not chunks:
        return {"title": f"Mock Test on {topic}", "content": "Not enough content to generate a mock test."}
    
    context = "\n".join(chunks)
    prompt = f"""
    You are an educational agent tasked with creating a mock test. Based on the following context, generate a mock test with {num_questions} multiple-choice questions on the topic '{topic}'. Each question should have 4 options (A, B, C, D) and give the correct answers in the end after user attempts the test.

    Context: {context}
    Mock Test:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return {"title": f"Mock Test on {topic}", "content": response.text.strip()}

def generate_mind_map(topic: str, source: str = None) -> dict:
    chunks = retrieve_chunks(topic, source)
    if not chunks:
        return {"success": False, "message": "Not enough content to generate a mind map."}
    
    context = "\n".join(chunks)
    prompt = f"""
    You are a learning expert. Based on the following context, create a Mermaid.js syntax for a mind map on the topic '{topic}'. The mind map should have a central node (the topic), main branches, and sub-branches. Use the 'graph TD' syntax.

    Context: {context}
    Mermaid.js Syntax:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    mermaid_content = response.text.strip()
    
    if '```mermaid' in mermaid_content:
        mermaid_content = mermaid_content.split('```mermaid')[1].split('```')[0].strip()
    elif '```' in mermaid_content:
        mermaid_content = mermaid_content.split('```')[1].split('```')[0].strip()
    
    return {
        "success": True,
        "mermaid_code": mermaid_content,
        "title": f"Mind Map for {topic}",
        "type": "mindmap"
    }

def generate_flowchart(topic: str, source: str = None) -> dict:
    chunks = retrieve_chunks(topic, source)
    if not chunks:
        return {"success": False, "message": "Not enough content to generate a flowchart."}
    
    context = "\n".join(chunks)
    prompt = f"""
    You are a process design expert. Based on the following context, create a Mermaid.js syntax for a flowchart on the topic '{topic}'. The flowchart should represent a sequence of steps or decisions in a logical order. Use the 'graph TD' syntax and appropriate shapes (e.g., '[ ]' for steps, '{{}}' for decisions).

    Context: {context}
    Mermaid.js Syntax:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    mermaid_content = response.text.strip()
    
    if '```mermaid' in mermaid_content:
        mermaid_content = mermaid_content.split('```mermaid')[1].split('```')[0].strip()
    elif '```' in mermaid_content:
        mermaid_content = mermaid_content.split('```')[1].split('```')[0].strip()
    
    return {
        "success": True,
        "mermaid_code": mermaid_content,
        "title": f"Flowchart for {topic}",
        "type": "flowchart"
    }

# Routes
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/api/add-web-content', methods=['POST', 'OPTIONS'])
def add_web_content():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    url = data.get('url')  # Match frontend key
    if url:
        success = process_input(url, "web")
        return jsonify({"success": success, "message": f"Added content from {url}" if success else "Failed to extract content"})
    return jsonify({"success": False, "message": "No URL provided"}), 400

@app.route('/api/add-video', methods=['POST', 'OPTIONS'])
def add_video():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    url = data.get('url')
    if url:
        success = process_input(url, "video")
        return jsonify({"success": success, "message": f"Added video content from {url}" if success else "Failed to extract content"})
    return jsonify({"success": False, "message": "No URL provided"}), 400

@app.route('/api/add-text', methods=['POST', 'OPTIONS'])
def add_text():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    text = data.get('text')
    if text:
        success = process_input(text, "user")
        return jsonify({"success": success, "message": "Added user text" if success else "Failed to add text"})
    return jsonify({"success": False, "message": "No text provided"}), 400

@app.route('/api/ask-question', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    question = data.get('question')
    source = data.get('source')
    if question:
        answer = generate_answer(question, source)
        return jsonify({"success": True, "question": question, "answer": answer})
    return jsonify({"success": False, "message": "No question provided"}), 400

@app.route('/api/get-summary', methods=['POST', 'OPTIONS'])
def get_summary():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    source = data.get('source')
    summary = generate_summary(source)
    return jsonify({"success": True, "summary": summary})

@app.route('/api/get-sources', methods=['GET', 'OPTIONS'])
def get_sources():
    if request.method == 'OPTIONS':
        return '', 200
    sources = list(set(item['source'] for item in kb.knowledge_base))
    return jsonify({"success": True, "sources": sources})

@app.route('/api/generate-mock-test', methods=['POST', 'OPTIONS'])
def generate_mock_test_route():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if topic:
        mock_test = generate_mock_test(topic, source)
        return jsonify({"success": True, "mock_test": mock_test})
    return jsonify({"success": False, "message": "No topic provided"}), 400

@app.route('/api/generate-mind-map', methods=['POST', 'OPTIONS'])
def generate_mind_map_route():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if topic:
        return jsonify(generate_mind_map(topic, source))
    return jsonify({"success": False, "message": "No topic provided"}), 400

@app.route('/api/generate-flowchart', methods=['POST', 'OPTIONS'])
def generate_flowchart_route():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    topic = data.get('topic')
    source = data.get('source')
    if topic:
        return jsonify(generate_flowchart(topic, source))
    return jsonify({"success": False, "message": "No topic provided"}), 400

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')