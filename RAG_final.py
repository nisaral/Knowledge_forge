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
import re
import json
from typing import List, Dict
from graphviz import Digraph  # For mind maps and flowcharts
import os

# Configure Gemini API (replace with your API key)
GEMINI_API_KEY = "AIzaSyDq3W6bcmtED-s0vDKmSBZr8uIwy4Gc1Io"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
knowledge_base: List[Dict] = []  # Extended with metadata: {'text': str, 'source': str, 'type': str}
vector_base = []

# Initialize Crawler for YouTube
crawler = WebCrawler()
crawler.warmup()
extraction_strategy = JsonCssExtractionStrategy(
    schema={"text_content": {"css": "p", "type": "string"}}
)

def crawl_url(url: str, output_file: str = "crawled_data.json") -> str:
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        driver.implicitly_wait(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        with open(output_file, "w") as f:
            json.dump({"url": url, "text": text_content}, f)
        print(f"Crawled {url} and saved to {output_file}")
        driver.quit()
        return text_content if text_content else None
    except Exception as e:
        print(f"Error crawling {url} with Selenium: {str(e)}")
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
        print(f"Error fetching transcript for {video_url}: {e}")
        return None

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

def update_knowledge_base(text: str, source: str, source_type: str):
    if not text:
        return
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)
    for chunk in chunks:
        knowledge_base.append({"text": chunk, "source": source, "type": source_type})
    vector_base.extend(embeddings)
    print(f"Added {len(chunks)} chunks from {source} ({source_type}) to knowledge base.")

def process_input(source: str, source_type: str = "web"):
    if source_type == "web":
        text = crawl_url(source)
        if text:
            update_knowledge_base(text, source, "web")
    elif source_type == "video":
        page_text = crawl_youtube(source)
        transcript = get_youtube_transcript(source)
        text = f"{page_text or ''} {transcript or ''}".strip()
        if text:
            update_knowledge_base(text, source, "video")
    elif source_type == "user":
        update_knowledge_base(source, "User Input", "text")

def retrieve_chunks(query: str, source: str = None) -> List[str]:
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=3)
    if source:
        chunks = [kb["text"] for kb in knowledge_base if kb["source"] == source][:3]
    else:
        chunks = [knowledge_base[i]["text"] for i in indices[0]]
    return chunks

def generate_answer(query: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(query, source))
    prompt = f"""
    You are an expert educational assistant. Based on the following context, provide a clear, concise, and accurate answer to the question. Use the context provided and avoid speculation beyond it.

    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_summary(source: str = None) -> str:
    if not knowledge_base:
        return "No content available to summarize."
    if source:
        context = "\n".join([kb["text"] for kb in knowledge_base if kb["source"] == source][:3])
        title = f"Summary for {source}"
    else:
        context = "\n".join([kb["text"] for kb in knowledge_base][:3])
        title = "Summary of All Content"
    prompt = f"""
    You are a skilled summarizer. Provide a concise and informative summary of the following content in 2-3 sentences.

    Content: {context}
    
    Summary:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return f"### {title}\n{response.text.strip()}"

def generate_mock_test(topic: str, source: str = None, num_questions: int = 5) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are an educational agent tasked with creating a mock test. Based on the following context, generate a mock test with {num_questions} multiple-choice questions on the topic '{topic}'. Each question should have 4 options (A, B, C, D) and the correct answer clearly marked.

    Context: {context}
    
    Mock Test:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return f"### Mock Test on {topic}\n{response.text.strip()}"

def generate_mind_map(topic: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are a learning expert. Based on the following context, create a Graphviz DOT language script for a mind map on the topic '{topic}'. The mind map should have a central node (the topic), main branches, and sub-branches. Keep it concise and structured.

    Context: {context}
    
    Graphviz DOT Script:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    dot_content = response.text.strip()

    # Confirmation for image generation
    print(f"\n### Mind Map DOT Script for {topic}\n{dot_content}")
    confirm = input("Generate mind map image (PNG) from this script? (y/n): ").strip().lower()
    if confirm == 'y':
        dot = Digraph(comment=f'Mind Map for {topic}', format='png')
        # Extract DOT content (assuming Gemini wraps it in ```dot ... ```)
        if '```dot' in dot_content:
            dot_content = dot_content.split('```dot')[1].split('```')[0].strip()
        dot.body.append(dot_content)
        output_file = f"mind_map_{topic.replace(' ', '_')}"
        dot.render(output_file, view=True)
        return f"Generated mind map image: {output_file}.png"
    return "Image generation skipped."

def generate_flowchart(topic: str, source: str = None) -> str:
    context = "\n".join(retrieve_chunks(topic, source))
    prompt = f"""
    You are a process design expert. Based on the following context, create a Graphviz DOT language script for a flowchart on the topic '{topic}'. The flowchart should represent a sequence of steps or decisions in a logical order. Use shapes like 'box' for steps and 'diamond' for decisions.

    Context: {context}
    
    Graphviz DOT Script:
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    dot_content = response.text.strip()

    # Confirmation for image generation
    print(f"\n### Flowchart DOT Script for {topic}\n{dot_content}")
    confirm = input("Generate flowchart image (PNG) from this script? (y/n): ").strip().lower()
    if confirm == 'y':
        dot = Digraph(comment=f'Flowchart for {topic}', format='png')
        # Extract DOT content
        if '```dot' in dot_content:
            dot_content = dot_content.split('```dot')[1].split('```')[0].strip()
        dot.body.append(dot_content)
        output_file = f"flowchart_{topic.replace(' ', '_')}"
        dot.render(output_file, view=True)
        return f"Generated flowchart image: {output_file}.png"
    return "Image generation skipped."

def main():
    print("=== Welcome to the AI-Powered Learning Assistant ===")
    while True:
        print("\nOptions:")
        print("1. Add web content")
        print("2. Add video")
        print("3. Add text")
        print("4. Ask a question")
        print("5. Get summary")
        print("6. Generate mock test")
        print("7. Generate mind map")
        print("8. Generate flowchart")
        print("9. Exit")
        action = input("Choose an option (1-9): ").strip()

        if action == "1":
            url = input("Enter webpage URL: ")
            process_input(url, "web")
        elif action == "2":
            url = input("Enter YouTube video URL: ")
            process_input(url, "video")
        elif action == "3":
            text = input("Enter text: ")
            process_input(text, "user")
        elif action == "4":
            query = input("Ask your question: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            response = generate_answer(query, source)
            print(f"\n### Question\n{query}\n### Answer\n{response}")
        elif action == "5":
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            summary = generate_summary(source)
            print(f"\n{summary}")
        elif action == "6":
            topic = input("Enter the topic for the mock test: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            test = generate_mock_test(topic, source)
            print(f"\n{test}")
        elif action == "7":
            topic = input("Enter the topic for the mind map: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            result = generate_mind_map(topic, source)
            print(f"\n{result}")
        elif action == "8":
            topic = input("Enter the topic for the flowchart: ")
            source = input("Optional: Specify a source URL (or press Enter for all content): ").strip() or None
            result = generate_flowchart(topic, source)
            print(f"\n{result}")
        elif action == "9":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1-9.")

if __name__ == "__main__":
    main()