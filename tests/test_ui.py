"""The single-page UI renders with every feature view and serves its CSP-safe static scripts."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from knowledge_forge.web import create_app

STATIC = Path(__file__).resolve().parent.parent / "static"
FEATURE_JS = ("app.js", "sm2.js", "crawl.js", "search.js", "create.js", "graph.js")


@pytest.fixture
def client(kb):
    return create_app(kb).test_client()


def test_index_has_all_views_and_crawl_limits(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    for view in ("chat", "search", "create", "graph", "tools", "insights"):
        assert f'data-view-pane="{view}"' in html
    for ident in ("form-site", "crawl-jobs", "search-form", "create-form", "graph-out", "sb-frames", "fc-count"):
        assert f'id="{ident}"' in html
    assert re.search(r'id="site-pages"[^>]*max="\d+"', html)
    for js in FEATURE_JS:
        assert f"/static/{js}" in html
    assert "<script>" not in html  # CSP: script-src 'self', no inline scripts
    assert "script-src 'self'" in res.headers["Content-Security-Policy"]


@pytest.mark.parametrize("name", FEATURE_JS)
def test_static_scripts_served(client, name):
    res = client.get(f"/static/{name}")
    assert res.status_code == 200
    assert len(res.data) > 500


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("name", FEATURE_JS)
def test_scripts_parse(name):
    subprocess.run(["node", "--check", str(STATIC / name)], check=True)
