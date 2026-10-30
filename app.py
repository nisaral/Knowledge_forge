"""WSGI entry point: ``gunicorn app:app`` or ``python app.py`` for local development."""
from knowledge_forge.config import get_settings
from knowledge_forge.web import create_app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    app.run(host=settings.host, port=settings.port, debug=settings.debug, threaded=True)
