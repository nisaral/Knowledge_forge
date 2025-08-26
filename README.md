# KnowledgeForge

<div align="center">

![KnowledgeForge Logo](https://img.shields.io/badge/KnowledgeForge-AI%20Learning%20Companion-blue?style=for-the-badge)

*An AI-powered educational tool for content analysis and visualization*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Gemini-API-orange.svg)](https://aistudio.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

##  Overview

KnowledgeForge is a Retrieval-Augmented Generation (RAG) application that empowers users to extract, analyze, and interact with knowledge from diverse sources like web pages, YouTube videos, and custom text. 

Featuring a modern frontend with Three.js 3D particles and glassmorphism design, it integrates a Flask backend with FAISS for vector search and Google's Gemini API for natural language processing. Whether you're summarizing articles, generating mock tests, or visualizing concepts with mind maps and flowcharts, KnowledgeForge is your all-in-one learning Companion.

##  Features

- **🌐 Content Ingestion**: Extract text from web URLs, YouTube videos (transcripts + page content), and user-provided text
- **❓ Question Answering**: Ask questions based on ingested content with context-aware responses
- **📝 Summarization**: Generate concise summaries of your content
- **📚 Mock Tests**: Create multiple-choice tests on specific topics
- **🧠 Visualizations**: Produce mind maps and flowcharts using Graphviz
- **🎨 Stunning UI**: 3D particle background with Three.js and glassmorphism design for an immersive experience
- **⚡ Scalable Backend**: Built with Flask, FAISS for efficient vector retrieval, and Gemini for generation

<img width="1919" height="1029" alt="Screenshot 2025-08-03 123849" src="https://github.com/user-attachments/assets/45cd69e1-33b3-47a9-9ebd-d4e9ce744a22" />
<img width="1919" height="1031" alt="Screenshot 2025-08-03 123914" src="https://github.com/user-attachments/assets/0c9397cf-1ebb-4ea2-85d9-c0437240c1c1" />


## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | HTML, CSS, JavaScript, Three.js |
| **Backend** | Flask (Python), Flask-CORS |
| **AI/ML** | SentenceTransformers (all-MiniLM-L6-v2), FAISS, Google Gemini API |
| **Crawling** | Selenium, Crawl4AI, YouTube Transcript API |
| **Visualization** | Graphviz |
| **Dependencies** | Playwright, BeautifulSoup4 |

## 📋 Prerequisites

- Python 3.11+
- Google Gemini API Key (sign up at [Google AI Studio](https://aistudio.google.com/))
- Chrome browser (for Selenium compatibility)
- Git

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/nisaral/knowledge_forge.git
cd knowledge_forge

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your Gemini API key
# Create a .env file and add: GEMINI_API_KEY=your_api_key_here

# Run the application
python app.py
```

## 📊 Usage

### Add Content
- Use the **Web URL** tab to input a webpage URL
- Use the **YouTube** tab for video URLs (transcripts + metadata)
- Use the **Custom Text** tab to paste your own text

### Interact
- **Ask Questions**: Type a question and optionally select a source
- **Summarize**: Generate a summary of all or specific content
- **Mock Test**: Create a test by entering a topic
- **Mind Map/Flowchart**: Visualize concepts with a topic input
- **Clear Data**: Reset the session using the "Clear Data" button

## 📷 Screenshots

<div align="center">
  <em>Screenshots coming soon!</em>
</div>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Google Gemini API](https://aistudio.google.com/) for powerful AI capabilities
- [Three.js](https://threejs.org/) for the immersive 3D background
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search

---

<div align="center">
  <sub>Built with ❤️ by the Keyush Nisar</sub>
</div>
