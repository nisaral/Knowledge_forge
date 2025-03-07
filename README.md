# Knowledge_forge
# KnowledgeForge

*An AI-powered educational tool for content analysis and visualization.*

KnowledgeForge is a Retrieval-Augmented Generation (RAG) application that empowers users to extract, analyze, and interact with knowledge from diverse sources like web pages, YouTube videos, and custom text. Featuring a modern frontend with Three.js 3D particles and glassmorphism design, it integrates a Flask backend with FAISS for vector search and Google’s Gemini API for natural language processing. Whether you’re summarizing articles, generating mock tests, or visualizing concepts with mind maps and flowcharts, KnowledgeForge is your all-in-one learning companion.

---

## Features
- **Content Ingestion**: Extract text from web URLs, YouTube videos (transcripts + page content), and user-provided text.
- **Question Answering**: Ask questions based on ingested content with context-aware responses.
- **Summarization**: Generate concise summaries of your content.
- **Mock Tests**: Create multiple-choice tests on specific topics.
- **Visualizations**: Produce mind maps and flowcharts using Graphviz.
- **Stunning UI**: 3D particle background with Three.js and glassmorphism design for an immersive experience.
- **Scalable Backend**: Built with Flask, FAISS for efficient vector retrieval, and Gemini for generation.

---

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript, Three.js
- **Backend**: Flask (Python), Flask-CORS
- **AI/ML**: SentenceTransformers (all-MiniLM-L6-v2), FAISS, Google Gemini API
- **Crawling**: Selenium, Crawl4AI, YouTube Transcript API
- **Visualization**: Graphviz
- **Dependencies**: Playwright, BeautifulSoup4

---

## Prerequisites
- Python 3.11+
- Google Gemini API Key (sign up at [Google AI Studio](https://aistudio.google.com/))
- Chrome browser (for Selenium compatibility)
- Git

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/nisaral/knowledge_forge.git
cd knowledge_forge



