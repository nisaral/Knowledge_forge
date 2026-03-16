FROM python:3.11-slim

# Install system deps for PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exclude local data files from image
RUN rm -f knowledge_base.json faiss_index.bin

EXPOSE 8000

# GEMINI_API_KEY and FLASK_SECRET must be passed as environment variables at runtime
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]