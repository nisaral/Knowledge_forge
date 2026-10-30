FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py gunicorn.conf.py pyproject.toml ./
COPY knowledge_forge ./knowledge_forge
COPY templates ./templates
COPY static ./static

RUN mkdir -p /app/data && useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV DATA_DIR=/app/data
ENV PORT=8000
ENV VECTOR_STORE=qdrant
ENV QDRANT_URL=http://qdrant:6333
ENV REDIS_URL=redis://redis:6379/0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
