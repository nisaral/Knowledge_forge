"""WSGI entry point: ``gunicorn app:app`` or ``python app.py`` for local development."""
import os

from knowledge_forge.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
