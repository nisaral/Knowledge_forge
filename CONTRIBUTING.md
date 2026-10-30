# Contributing

1. Create a virtualenv and run `pip install -r requirements-dev.txt`.
2. Copy `.env.example` to `.env`. **Never commit `.env` or any key.**
3. Keep changes small and focused, and add or adjust tests in `tests/`. Tests must not hit the network unless they are marked `@pytest.mark.network`.
4. Before pushing, make sure `ruff check .` and `pytest -q` both pass.
5. The frontend is plain HTML, CSS and JS in `templates/` and `static/`, with no build step. Keep it dependency-free.

Please report security issues privately to the maintainer rather than opening a public issue.
