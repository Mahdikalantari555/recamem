"""Pluggable web search providers and a stdlib-only page fetcher.

The Droid can only learn what it is given until something goes and looks. This
module is that something, kept deliberately small and boring: one provider
interface, one key-free HTML provider, one key-configurable JSON provider, and
a fetcher that refuses to be turned into a server-side request forgery tool.

Rules the fetcher enforces (``adaptation/web-grounded-learning``):

* only ``http`` and ``https``;
* every hop after a redirect re-applies the host check;
* loopback, private, link-local, reserved, unspecified and multicast targets are
  refused — the loopback exemption exists only inside
  :func:`loopback_test_scope`, which the tests use;
* at most two redirects, a connect/read timeout, a response byte cap, and a
  descriptive user agent;
* every failure is returned as a structured dict, never raised out of a fetch.

No third-party HTTP, HTML, or search SDK is used.
"""

from __future__ import annotations

import contextlib
import contextvars
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

__all__ = [
    "SearchResult",
    "SearchProvider",
    "NullSearchProvider",
    "DuckDuckGoHTMLProvider",
    "JsonApiSearchProvider",
    "FetchResult",
    "select_provider",
    "fetch_page",
    "html_to_text",
    "loopback_test_scope",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BYTES",
    "MAX_REDIRECTS",
    "USER_AGENT",
    "PROVIDER_NAMES",
]

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_BYTES = 512 * 1024
MAX_REDIRECTS = 2
USER_AGENT = (
    "recamem/1.0"
)

PROVIDER_NAMES = ("none", "duckduckgo", "json_api")

_LOOPBACK_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tmt_allow_loopback_fetch", default=False
)


@contextlib.contextmanager
def loopback_test_scope():
    """Allow loopback targets for an in-process test server.

    This is a context-scoped exemption, not a configuration switch: production
    code never enters it, and leaving the block restores the refusal.
    """
    token = _LOOPBACK_ALLOWED.set(True)
    try:
        yield
    finally:
        _LOOPBACK_ALLOWED.reset(token)


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    url: str
    text: str = ""
    status: int = 0
    bytes_read: int = 0
    truncated: bool = False
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "url": self.url,
            "text": self.text,
            "status": self.status,
            "bytes_read": self.bytes_read,
            "truncated": self.truncated,
            "reason": self.reason,
        }


def failure(reason: str, url: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """A structured failure: always a dict, never an exception."""
    payload: Dict[str, Any] = {"ok": False, "reason": str(reason), "url": url}
    payload.update(extra)
    return payload


def unavailable(reason: str = "no search provider configured") -> Dict[str, Any]:
    """The documented 'external acquisition is unavailable' result."""
    return {
        "ok": False,
        "available": False,
        "unavailable": True,
        "reason": reason,
    }


def _transport(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class SearchProvider:
    """One method: a query and a limit in, results out."""

    name = "abstract"

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class NullSearchProvider(SearchProvider):
    """The provider that never touches the network."""

    name = "none"

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        return []


class _ResultListingParser(HTMLParser):
    """Extract result titles, links and snippets from a listing page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current: Optional[Dict[str, Any]] = None
        self.entries: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag != "a":
            return
        attributes = {key: (value or "") for key, value in attrs}
        classes = attributes.get("class", "")
        href = attributes.get("href", "")
        if "result__a" in classes:
            self._current = {"kind": "title", "href": href, "text": []}
            self.entries.append(self._current)
        elif "result__snippet" in classes:
            self._current = {"kind": "snippet", "href": href, "text": []}
            self.entries.append(self._current)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current["text"].append(data)


def _resolve_listing_url(href: str, base: str = "https://html.duckduckgo.com/") -> str:
    """Resolve a listing link, unwrapping a redirector's ``uddg`` parameter."""
    if not href:
        return ""
    target = urljoin(base, href)
    parsed = urlparse(target)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return target


class DuckDuckGoHTMLProvider(SearchProvider):
    """The key-free provider: parse the public HTML result listing.

    A key-free public endpoint is unofficial and can change shape or rate-limit;
    a failure here is returned as a structured error by the caller rather than
    raised into chat.
    """

    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, endpoint: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 transport: Optional[Callable[..., bytes]] = None, fetcher: Optional[Callable[..., Any]] = None):
        self.endpoint = endpoint or self.endpoint
        self.timeout = timeout
        self._transport = transport or _transport
        # The provider can borrow the guarded fetcher so its own request gets
        # the same scheme and host checks.
        self._fetcher = fetcher

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        url = f"{self.endpoint}?{urllib.parse.urlencode({'q': query})}"
        if self._fetcher is not None:
            fetched = self._fetcher(url, timeout=self.timeout)
            body = fetched.text.encode("utf-8") if getattr(fetched, "ok", False) else b""
        else:
            body = self._transport(
                url, {"User-Agent": USER_AGENT, "Accept": "text/html"}, self.timeout
            )
        parser = _ResultListingParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        results: List[SearchResult] = []
        pending_title: Optional[Dict[str, Any]] = None
        for entry in parser.entries:
            text = " ".join("".join(entry["text"]).split())
            if entry["kind"] == "title":
                pending_title = entry
                results.append(
                    SearchResult(
                        title=text,
                        url=_resolve_listing_url(entry["href"], self.endpoint),
                        snippet="",
                        provider=self.name,
                    )
                )
            elif entry["kind"] == "snippet" and results and pending_title is not None:
                last = results[-1]
                results[-1] = SearchResult(
                    title=last.title, url=last.url, snippet=text, provider=self.name
                )
        return [result for result in results if result.url][:limit]


class JsonApiSearchProvider(SearchProvider):
    """The key-configurable provider: a JSON search API with a bearer token.

    The credential is read from the environment and never stored in the
    profile; the transport is injectable so the tests never open a socket.
    """

    name = "json_api"

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 api_key_env: str = "TMT_SEARCH_API_KEY",
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 transport: Optional[Callable[..., bytes]] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.timeout = timeout
        self._transport = transport or _transport

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        url = f"{self.base_url}?{urllib.parse.urlencode({'q': query, 'count': limit})}"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = self._transport(url, headers, self.timeout)
        payload = json.loads(body.decode("utf-8", errors="replace"))
        raw_results = payload.get("results")
        if raw_results is None and isinstance(payload.get("web"), dict):
            raw_results = payload["web"].get("results")
        results: List[SearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            url_value = item.get("url") or item.get("link") or ""
            results.append(
                SearchResult(
                    title=str(item.get("title") or item.get("name") or ""),
                    url=str(url_value),
                    snippet=str(item.get("snippet") or item.get("description") or ""),
                    provider=self.name,
                )
            )
        return [result for result in results if result.url][:limit]


def select_provider(config: Optional[Mapping[str, Any]] = None,
                    transport: Optional[Callable[..., bytes]] = None
                    ) -> Tuple[Optional[SearchProvider], Optional[Dict[str, Any]]]:
    """Build the configured provider, or return the unavailable failure.

    Returns ``(provider, failure)``: exactly one is non-``None``. An unknown
    provider name is a structured failure naming the value; a missing one is
    the documented unavailable result; neither raises.
    """
    config = dict(config or {})
    name = str(config.get("provider") or "").strip().lower()
    timeout = float(config.get("request_timeout", DEFAULT_TIMEOUT_SECONDS))
    if name in ("", "none", "null", "off"):
        return None, unavailable("no search provider configured")
    if name in ("duckduckgo", "ddg", "html"):
        return DuckDuckGoHTMLProvider(timeout=timeout, transport=transport), None
    if name in ("json_api", "json", "api"):
        base_url = str(config.get("base_url") or config.get("search_url") or "").strip()
        if not base_url:
            return None, unavailable(
                "the json_api provider needs a base_url in the research configuration"
            )
        env_name = str(config.get("api_key_env") or "TMT_SEARCH_API_KEY")
        provider = JsonApiSearchProvider(
            base_url=base_url,
            api_key=config.get("api_key"),
            api_key_env=env_name,
            timeout=timeout,
            transport=transport,
        )
        if not provider.api_key:
            return None, failure(
                f"the json_api provider has no credential (set {env_name})",
                provider=name,
            )
        return provider, None
    return None, failure(f"unknown search provider {name!r}", provider=name)


# --------------------------------------------------------------------------- #
# Fetcher
# --------------------------------------------------------------------------- #
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _host_is_public(hostname: str, port: int) -> Tuple[bool, str]:
    """Resolve ``hostname`` and refuse every non-public address class."""
    if not hostname:
        return False, "the URL has no host"
    if _LOOPBACK_ALLOWED.get():
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(hostname, port)}
        except OSError as exc:
            return False, f"{hostname} did not resolve: {exc}"
        if all(ipaddress.ip_address(address).is_loopback for address in addresses):
            return True, ""
        return False, f"{hostname} is not loopback inside the test scope"
    try:
        infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return False, f"{hostname} did not resolve: {exc}"
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False, f"{hostname} did not resolve"
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False, f"{hostname} resolved to an unparseable address {address!r}"
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, f"{hostname} resolves to the refused address {address}"
    return True, ""


def _check_url(url: str) -> Tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme {parsed.scheme!r} is not http or https"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _host_is_public(parsed.hostname or "", port)


def fetch_page(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
               max_bytes: int = DEFAULT_MAX_BYTES,
               max_redirects: int = MAX_REDIRECTS,
               max_chars: int = 20000) -> FetchResult:
    """Fetch one page and return its extracted text or a structured failure."""
    allowed, reason = _check_url(url)
    if not allowed:
        return FetchResult(ok=False, url=url, reason=reason)

    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    redirects = 0
    while True:
        request = urllib.request.Request(
            current, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"}
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                status = getattr(response, "status", 200)
            truncated = len(body) > max_bytes
            body = body[:max_bytes]
            charset = "utf-8"
            text = html_to_text(body.decode(charset, errors="replace"), max_chars=max_chars)
            return FetchResult(
                ok=True,
                url=current,
                text=text,
                status=int(status),
                bytes_read=len(body),
                truncated=truncated,
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and redirects < max_redirects:
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    return FetchResult(ok=False, url=current, status=exc.code,
                                       reason=f"HTTP {exc.code} without a Location header")
                target = urljoin(current, location)
                allowed, reason = _check_url(target)
                if not allowed:
                    return FetchResult(
                        ok=False, url=target,
                        reason=f"redirect refused: {reason}",
                    )
                current = target
                redirects += 1
                continue
            return FetchResult(ok=False, url=current, status=int(exc.code),
                               reason=f"HTTP {exc.code}")
        except (socket.timeout, TimeoutError) as exc:
            return FetchResult(ok=False, url=current,
                               reason=f"timeout after {timeout}s ({exc})")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                return FetchResult(ok=False, url=current,
                                   reason=f"timeout after {timeout}s")
            return FetchResult(ok=False, url=current,
                               reason=f"connection failed for {urlparse(current).hostname}: {exc.reason}")
        except Exception as exc:  # every failure is structured
            return FetchResult(ok=False, url=current,
                               reason=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# HTML to text
# --------------------------------------------------------------------------- #
_DROP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript"}
_KEEP_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "pre", "br"}


class _TextExtractor(HTMLParser):
    def __init__(self, max_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_chars = max_chars
        self._drop_depth = 0
        self._chunks: List[str] = []
        self._length = 0
        self.done = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _DROP_TAGS:
            self._drop_depth += 1
        elif tag in _KEEP_TAGS and self._drop_depth == 0:
            self._append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_TAGS and self._drop_depth:
            self._drop_depth -= 1
        elif tag in _KEEP_TAGS and self._drop_depth == 0:
            self._append(" ")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0 and not self.done:
            self._append(data)

    def _append(self, data: str) -> None:
        if self.done:
            return
        self._chunks.append(data)
        self._length += len(data)
        if self._length >= self.max_chars:
            self.done = True

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()[: self.max_chars]


def html_to_text(markup: str, *, max_chars: int = 20000) -> str:
    """Readable text from HTML: drop chrome and scripts, keep prose."""
    parser = _TextExtractor(max_chars=max_chars)
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        # A malformed page still yields whatever was extracted before the error.
        pass
    return parser.text()
