"""Hermetic tests for the web search providers and the guarded fetcher.

No test opens a socket except the fetcher tests, which use an ``http.server``
bound to loopback inside :func:`loopback_test_scope` — the loopback exemption is
scoped to the fixture, not a production bypass.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.model import web_teacher
from src.model.web_teacher import (
    DuckDuckGoHTMLProvider,
    JsonApiSearchProvider,
    NullSearchProvider,
    fetch_page,
    html_to_text,
    loopback_test_scope,
    select_provider,
)

DDG_FIXTURE = """
<html><body>
  <div class="result">
    <a rel="nofollow" class="result__a" href="https://example.test/one">First result</a>
    <a class="result__snippet" href="https://example.test/one">First snippet text</a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Ftwo&amp;rut=abc">Second result</a>
    <a class="result__snippet" href="#">Second snippet</a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="https://example.test/three">Third result</a>
    <a class="result__snippet" href="#">Third snippet</a>
  </div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def test_null_provider_returns_no_results_and_raises_nothing():
    provider = NullSearchProvider()
    assert provider.search("anything", limit=5) == []


def test_key_free_html_provider_parses_a_checked_in_listing_without_a_socket():
    calls = []

    def transport(url, headers, timeout):
        calls.append((url, headers, timeout))
        return DDG_FIXTURE.encode("utf-8")

    provider = DuckDuckGoHTMLProvider(transport=transport)
    results = provider.search("remote sensing", limit=3)
    assert len(results) == 3
    assert [result.url for result in results] == [
        "https://example.test/one",
        "https://example.test/two",
        "https://example.test/three",
    ]
    assert results[0].title == "First result"
    assert results[0].snippet == "First snippet text"
    assert calls and "q=remote+sensing" in calls[0][0]
    assert "User-Agent" in calls[0][1]


def test_json_provider_sends_the_credential_and_maps_the_response(monkeypatch):
    monkeypatch.setenv("TMT_SEARCH_API_KEY", "sekret")
    seen = {}

    def transport(url, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        return json.dumps(
            {
                "web": {
                    "results": [
                        {"title": "Radar", "url": "https://example.test/radar",
                         "description": "radio detection"},
                        {"title": "Lidar", "url": "https://example.test/lidar",
                         "description": "light detection"},
                    ]
                }
            }
        ).encode("utf-8")

    provider = JsonApiSearchProvider(
        "https://api.example.test/search", api_key_env="TMT_SEARCH_API_KEY",
        transport=transport,
    )
    results = provider.search("radar", limit=2)
    assert seen["headers"]["Authorization"] == "Bearer sekret"
    assert "q=radar" in seen["url"]
    assert [result.url for result in results] == [
        "https://example.test/radar",
        "https://example.test/lidar",
    ]
    assert results[0].snippet == "radio detection"


def test_provider_selection_by_configuration_name():
    provider, error = select_provider({"provider": "duckduckgo"})
    assert isinstance(provider, DuckDuckGoHTMLProvider)
    assert error is None

    provider, error = select_provider({"provider": None})
    assert provider is None
    assert error["available"] is False
    assert "no search provider" in error["reason"]

    provider, error = select_provider({"provider": "carrier-pigeon"})
    assert provider is None
    assert "carrier-pigeon" in error["reason"]
    assert error["provider"] == "carrier-pigeon"


# --------------------------------------------------------------------------- #
# Fetcher guards
# --------------------------------------------------------------------------- #
def test_loopback_and_private_targets_are_refused_outside_the_test_scope():
    for url in ("http://127.0.0.1:9/page", "http://10.1.2.3/page", "http://169.254.1.1/"):
        result = fetch_page(url, timeout=0.2)
        assert result.ok is False
        assert "refuses" in result.reason or "refused" in result.reason
        assert result.bytes_read == 0


def test_non_http_scheme_is_refused():
    result = fetch_page("file:///etc/passwd")
    assert result.ok is False
    assert "scheme" in result.reason


def test_intranet_and_reserved_names_are_refused():
    result = fetch_page("http://localhost:9/page")
    assert result.ok is False
    assert "localhost" in result.reason


# --------------------------------------------------------------------------- #
# Bounded fetching against a loopback fixture
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.startswith("/slow"):
            time.sleep(0.8)
            body = b"<html><body><p>late</p></body></html>"
        elif self.path.startswith("/huge"):
            body = ("<html><body><p>" + ("word " * 2000) + "</p></body></html>").encode()
        elif self.path.startswith("/redirect-private"):
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.7/secret")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        elif self.path.startswith("/entities"):
            body = (
                "<html><head><script>alert('x')</script><style>.a{}</style></head>"
                "<body><nav>menu</nav><h1>Title &amp; more</h1>"
                "<p>Paragraph text</p><footer>footer</footer>"
                "<ul><li>item one</li></ul></body></html>"
            ).encode()
        else:
            body = b"<html><body><p>ok</p></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the fixture's stderr
        return


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_loopback_fixture_is_reachable_only_inside_the_scope(local_server):
    with loopback_test_scope():
        result = fetch_page(local_server + "/ok", timeout=2.0)
        assert result.ok is True
    outside = fetch_page(local_server + "/ok", timeout=0.2)
    assert outside.ok is False


def test_response_size_cap_truncates_and_the_text_stays_usable(local_server):
    with loopback_test_scope():
        result = fetch_page(local_server + "/huge", timeout=2.0, max_bytes=256)
    assert result.ok is True
    assert result.truncated is True
    assert result.bytes_read == 256
    assert "word" in result.text


def test_timeout_returns_a_structured_timeout_failure(local_server):
    with loopback_test_scope():
        started = time.perf_counter()
        result = fetch_page(local_server + "/slow", timeout=0.2)
        elapsed = time.perf_counter() - started
    assert result.ok is False
    assert "timeout" in result.reason.lower()
    assert elapsed < 0.7


def test_unreachable_host_returns_a_failure_naming_the_host():
    with loopback_test_scope():
        result = fetch_page("http://127.0.0.1:9/nowhere", timeout=0.5)
    assert result.ok is False
    assert "127.0.0.1" in result.reason


def test_redirect_to_a_private_address_is_refused(local_server):
    with loopback_test_scope():
        result = fetch_page(local_server + "/redirect-private", timeout=1.0)
    assert result.ok is False
    assert "redirect refused" in result.reason
    assert "10.0.0.7" in result.reason


# --------------------------------------------------------------------------- #
# HTML to text
# --------------------------------------------------------------------------- #
def test_html_to_text_drops_chrome_keeps_prose_and_decodes_entities(local_server):
    with loopback_test_scope():
        result = fetch_page(local_server + "/entities", timeout=2.0)
    assert result.ok is True
    assert "alert" not in result.text
    assert ".a{}" not in result.text
    assert "menu" not in result.text
    assert "footer" not in result.text
    assert "Paragraph text" in result.text
    assert "item one" in result.text
    assert "Title & more" in result.text


def test_html_to_text_collapses_whitespace_and_stops_at_the_budget():
    text = html_to_text("<p>" + ("alpha   beta\n\n") * 100 + "</p>", max_chars=40)
    assert len(text) <= 40
    assert "  " not in text
