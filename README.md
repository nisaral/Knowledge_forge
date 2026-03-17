# KnowledgeForge V2

<div align="center">

![KnowledgeForge V2](https://img.shields.io/badge/KnowledgeForge-V2%20AI%20Learning%20Companion-2563eb?style=for-the-badge)

*A multimodal AI-powered learning platform — ingest anything, learn everything*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Gemini 2.0](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)](https://aistudio.google.com/)
[![CI](https://github.com/nisaral/knowledge_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/nisaral/knowledge_forge/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Overview

KnowledgeForge V2 is a production-grade multimodal RAG (Retrieval-Augmented Generation) application that lets you **ingest knowledge from any format** — PDFs, audio lectures, YouTube videos, images, web pages, and plain text — and interact with it through powerful AI tools.

Built with **Gemini 2.0** embeddings and generation, a **hybrid BM25 + semantic search** pipeline with LLM re-ranking, and a clean professional UI.

## What's New in V2

| Feature | V1 | V2 |
|---|---|---|
| **Embeddings** | SentenceTransformers (384-dim) | Gemini `text-embedding-004` (768-dim) |
| **Generation** | gemini-1.5-flash | **gemini-2.0-flash** |
| **RAG Search** | FAISS semantic only | **Hybrid BM25 + Semantic + LLM Re-ranking** |
| **Chunking** | Fixed-size | **Sliding window with overlap** |
| **Multimodal** | Web, YouTube, Text | + **PDF, Audio, Video, Images** |
| **Tools** | Q&A, Mock Test, Mind Map, Flowchart, Storyboard, Entities, Topics | + **Chat (multi-turn), Flashcards, Study Plan, Summarize** |
| **UI** | Dark glassmorphism | **Professional white/blue theme** |
| **MLOps** | Dockerfile only | **GitHub Actions CI/CD → GHCR** |

## Features

- **📄 Multimodal Ingestion**: PDF (text + scanned), MP3/WAV audio lectures, MP4/WEBM videos, screenshots/images, web URLs, YouTube transcripts, plain text
- **🔍 Hybrid RAG**: BM25 keyword + FAISS semantic vector search, followed by Gemini LLM re-ranking for maximum relevance
- **💬 Multi-turn Chat**: Session-based conversation with full history awareness
- **🃏 Flashcards**: Auto-generated spaced-repetition flip cards
- **📅 Study Plan**: Structured multi-week study schedules from your content
- **🧪 Mock Tests**: Generated MCQ assessments with answer keys
- **🧠 Mind Maps & Flowcharts**: Mermaid.js visualizations
- **🏷 Entity Extraction**: People, organizations, locations, concepts
- **📊 Topic Clustering**: PCA + K-Means visualization of content clusters
- **📋 Copy & Download**: Every result card has copy-to-clipboard and markdown download

## Tech Stack

| Layer | Technology |
|---|---|
| **AI / Embeddings** | Google Gemini 2.0 Flash + gemini-embedding-2-preview |
| **RAG** | FAISS (vector store) + BM25Okapi (keyword) + LLM re-ranking |
| **Multimodal** | Gemini Files API (audio, video, image, PDF) + PyMuPDF |
| **Backend** | Flask, Flask-CORS, Gunicorn |
| **Visualization** | Mermaid.js, Matplotlib, Seaborn |
| **Crawling** | BeautifulSoup4, YouTube Transcript API |
| **MLOps** | GitHub Actions CI/CD, Docker, GHCR |
| **Frontend** | HTML5, CSS3, Vanilla JS, Inter font |

## Prerequisites

- Python 3.11+
- Google Gemini API Key — [Get one free at Google AI Studio](https://aistudio.google.com/)
- Git

## Installation

```bash
git clone https://github.com/nisaral/knowledge_forge.git
cd knowledge_forge

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Create .env file
echo GEMINI_API_KEY=your_api_key_here > .env

python app.py
```

Visit `http://localhost:5000`

## API Key Management in Production

| Platform | Method |
|---|---|
| **GitHub Actions** | `Settings → Secrets → GEMINI_API_KEY` → use `${{ secrets.GEMINI_API_KEY }}` |
| **Docker** | `docker run -e GEMINI_API_KEY=xxx knowledge-forge` |
| **Render / Railway** | Dashboard → Environment Variables |
| **Google Cloud Run** | Secret Manager → mount as env var |
| **HuggingFace Spaces** | Settings → Repository Secrets |

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

## GitHub Actions (MLOps)

**CI** — runs on every push/PR to `main`:
1. `flake8` lint
2. `pytest` unit tests (mocked, no real API cost)
3. Docker image build verification

**CD** — runs on push to `main` or version tags:
1. Builds Docker image
2. Pushes to GitHub Container Registry (GHCR)
3. Optional: trigger Render/HuggingFace deploy webhook

```bash
# Set your GitHub secret:
# Repository → Settings → Secrets → Actions → New secret
# Name: GEMINI_API_KEY
# Value: your_actual_api_key
```

## Docker

```bash
# Build
docker build -t knowledge-forge:v2 .

# Run
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e FLASK_SECRET=your_secret \
  knowledge-forge:v2

# Or pull from GHCR (after CD runs):
docker pull ghcr.io/nisaral/knowledge_forge:main
```

## Usage

### Add Knowledge
- **Web URL** — paste any article/documentation URL
- **YouTube** — paste video URL for auto-transcription
- **PDF** — drag & drop (text or scanned PDFs supported)
- **Audio** — upload `.mp3`, `.wav`, `.m4a` lecture recordings
- **Video** — upload `.mp4`, `.webm` lecture videos
- **Image** — upload screenshots, diagrams, handwritten notes

### Use AI Tools
| Tool | Description |
|---|---|
| Q&A | One-shot question answering with citation-aware retrieval |
| Chat | Multi-turn conversations with session memory |
| Summarize | Comprehensive structured summary of all or filtered content |
| Flashcards | Interactive flip cards for spaced repetition |
| Study Plan | 4-week structured learning schedule |
| Mock Test | 5-question MCQ with answer key |
| Mind Map | Mermaid.js radial concept map |
| Flowchart | Mermaid.js process diagram |
| Storyboard | 3-panel visual story of key concepts |
| Entities | Extract people, orgs, locations, concepts |
| Topics | K-Means visual cluster of knowledge base content |

## Architecture

```
Input Sources
  │  Web URL / YouTube / Text / PDF / Audio / Video / Image
  ▼
Ingestion Pipeline
  │  crawl_url() / get_youtube_transcript() / ingest_pdf_bytes()
  │  ingest_file_via_gemini()  ← Gemini Files API for audio/video/image
  ▼
Chunking: Sliding Window (400 words, 50-word overlap)
  ▼
Gemini text-embedding-004 (768-dim) → FAISS IndexFlatIP
  ▼
Hybrid Retrieval
  │  Semantic (FAISS) × 0.7  +  BM25 × 0.3
  │  → LLM Re-ranking (top 12 → top 5)
  ▼
Gemini 2.0 Flash Generation
  ▼
Response (Q&A / Chat / Flashcards / Mindmap / etc.)
```

## Running Tests

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
  <sub>Built with ❤️ by Keyush  — KnowledgeForge V2</sub>
</div>
