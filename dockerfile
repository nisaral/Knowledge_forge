FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    mupdf-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -f knowledge_base.json faiss_index.bin .env \
    && mkdir -p /app/data

ENV DATA_DIR=/app/data
ENV PORT=8000
ENV VECTOR_STORE=qdrant
ENV QDRANT_URL=http://qdrant:6333
ENV REDIS_URL=redis://redis:6379/0
ENV SESSION_TYPE=redis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]