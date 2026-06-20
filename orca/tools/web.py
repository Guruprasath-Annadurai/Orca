"""
Orca Web Tool — search the web without any API key.
Uses DuckDuckGo's HTML interface and direct page fetching.
100% local, no account required.
"""
from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import quote_plus, urljoin

import httpx

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

DDG_URL = "https://html.duckduckgo.com/html/?q={query}&kl=us-en"


class SearchResult(NamedTuple):
    title: str
    url: str
    snippet: str


def search(query: str, n: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo, return top N results."""
    url = DDG_URL.format(query=quote_plus(query))
    try:
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        r.raise_for_status()
        return _parse_ddg(r.text, n)
    except Exception as e:
        return [SearchResult(title="Search failed", url="", snippet=str(e))]


def _parse_ddg(html: str, n: int) -> list[SearchResult]:
    results = []
    # Extract result blocks
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    for url, title, snippet in blocks[:n]:
        title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        url = url.strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def fetch_page(url: str, max_chars: int = 8000) -> str:
    """Fetch and clean a webpage, returning readable text."""
    try:
        r = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        text = r.text

        # Strip scripts, styles, nav
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:max_chars]
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def search_and_fetch(query: str, n: int = 3) -> str:
    """Search + fetch top results. Returns formatted context string."""
    results = search(query, n=n)
    if not results:
        return f"No results for: {query}"

    lines = [f"Search: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    {r.url}")
        lines.append(f"    {r.snippet}\n")

    return "\n".join(lines)
