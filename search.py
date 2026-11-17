"""Stage 3 search tool. Tries backends in order until one returns results:

1. ddgs (metasearch; good snippets). Its bundled HTTP client uses its own DNS
   resolver, which fails intermittently on some networks.
2. Google News RSS (system DNS; real publication dates; respects quotes/OR).
   Links are news.google.com redirects to the article.
3. Bing web RSS (system DNS). Its pubDate is a crawl date, so it is discarded.

Each result: url, title, snippet, published_date (ISO or None), query, origin, backend,
fallbacks (backends tried first and skipped; the first result also carries their errors).
"""

import html
import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests

import config as C

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
_TAG = re.compile(r"<[^>]+>")


def _rec(query, url, title, snippet, published, backend):
    return {"url": url, "title": (title or "").strip(), "snippet": (snippet or "").strip(),
            "published_date": published, "query": query, "origin": "search", "backend": backend}


def _ddgs(query, news):
    from ddgs import DDGS
    d = DDGS(timeout=C.SEARCH_TIMEOUT_S)
    if news:
        rows = d.news(query, max_results=C.RESULTS_PER_QUERY) or []
        return [_rec(query, r.get("url"), r.get("title"), r.get("body"), r.get("date"), "ddgs-news") for r in rows]
    rows = d.text(query, max_results=C.RESULTS_PER_QUERY) or []
    return [_rec(query, r.get("href"), r.get("title"), r.get("body"), None, "ddgs-web") for r in rows]


def _rss_items(url, params):
    r = requests.get(url, params=params, headers=_UA, timeout=C.SEARCH_TIMEOUT_S)
    r.raise_for_status()
    return list(ET.fromstring(r.content).iter("item"))


def _iso(pub):
    try:
        return parsedate_to_datetime(pub).date().isoformat() if pub else None
    except (TypeError, ValueError):
        return None


def _google_news(query):
    items = _rss_items("https://news.google.com/rss/search",
                       {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    out = []
    for it in items[:C.RESULTS_PER_QUERY]:
        title = it.findtext("title") or ""
        src = it.find("source")
        outlet = src.text if src is not None else ""
        desc = html.unescape(_TAG.sub(" ", it.findtext("description") or "")).strip()
        snippet = desc if desc and desc not in title else f"{title} ({outlet})"
        out.append(_rec(query, it.findtext("link"), title, snippet, _iso(it.findtext("pubDate")), "google-news-rss"))
    return out


def _bing_web(query):
    items = _rss_items("https://www.bing.com/search", {"q": query, "format": "rss", "count": C.RESULTS_PER_QUERY})
    return [_rec(query, it.findtext("link"), it.findtext("title"), it.findtext("description"), None, "bing-rss")
            for it in items[:C.RESULTS_PER_QUERY]]


# ---------------------------------------------------------------------------
# Full-text fetch (Stage 3): the article text of a result, as paragraphs.
# ---------------------------------------------------------------------------
class _Paragraphs(HTMLParser):
    """Collects visible paragraph-like text; skips scripts, menus, headers, footers."""
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button",
            "template", "iframe"}
    BLOCK = {"p", "h1", "h2", "h3", "li", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip, self.cur, self.paras = 0, None, []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK and not self.skip:
            self._flush()
            self.cur = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.cur is not None and not self.skip:
            self.cur.append(data)

    def _flush(self):
        if self.cur:
            t = " ".join("".join(self.cur).split())
            if len(t) >= 40:  # drop buttons, bylines, one-word list items
                self.paras.append(t)
        self.cur = None


def _check_fetch_url(url):
    """Only public http(s) hosts that are not on the skip list. Checked on every
    redirect hop, so a result cannot bounce the fetch to localhost or the LAN."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        raise ValueError(f"not an http(s) URL: {url[:100]}")
    bare = host.removeprefix("www.")
    if any(bare == h or bare.endswith("." + h) for h in C.FETCH_SKIP_HOSTS):
        raise ValueError(f"skipped host {host}")
    for info in socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80)):
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError(f"non-public address {ip}")


def fetch_paragraphs(url):
    """GET a page and return its paragraphs. Raises on HTTP errors, non-HTML,
    non-public hosts or more than 5 redirects."""
    for _ in range(6):
        _check_fetch_url(url)
        r = requests.get(url, headers=_UA, timeout=C.FETCH_TIMEOUT_S, stream=True, allow_redirects=False)
        if not r.is_redirect:
            break
        url = urljoin(url, r.headers.get("location", ""))
        r.close()
    else:
        raise ValueError("too many redirects")
    with r:
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "").lower():
            raise ValueError(f"not HTML ({r.headers.get('content-type')})")
        raw = r.raw.read(C.FETCH_MAX_BYTES, decode_content=True)
        enc = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
    parser = _Paragraphs()
    parser.feed(raw.decode(enc, errors="replace"))
    parser.close()
    parser._flush()
    return parser.paras


def search(q):
    """q = {"bucket": "company"|"person", "query": str}. Returns a list of results.
    Raises RuntimeError with every backend's error if all of them fail."""
    query = q["query"]
    news_first = q["bucket"] == "company" and not query.startswith("site:")
    chain = [("ddgs", lambda: _ddgs(query, news=news_first))]
    if news_first:
        chain.append(("ddgs-web", lambda: _ddgs(query, news=False)))
    chain += [("google-news-rss", lambda: _google_news(query)), ("bing-rss", lambda: _bing_web(query))]

    errors = []
    for name, fn in chain:
        try:
            rows = [r for r in fn() if r["url"]]
        except Exception as e:  # network, rate limit, parse error, "No results found"
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if rows:
            for r in rows:  # backends tried first and skipped before this one answered (observability)
                r["fallbacks"] = len(errors)
            rows[0]["backend_errors"] = " | ".join(errors)
            return rows
        errors.append(f"{name}: no results")
    if all(e.endswith("no results") or "No results" in e for e in errors):
        return []
    raise RuntimeError(" | ".join(errors))
