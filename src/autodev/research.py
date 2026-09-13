"""Bounded, cacheable official-document research for repair evidence."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Callable
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class ResearchResult:
    query: str
    url: str
    text: str = ""
    official: bool = False
    cache_hit: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    def to_dict(self) -> dict[str, str | bool]:
        return {"query": self.query, "url": self.url, "text": self.text, "official": self.official, "cache_hit": self.cache_hit, "error": self.error}


class WebResearch:
    """Fetch a short official-doc page only; never receives workspace source."""

    OFFICIAL = {
        "sqlalchemy": "https://docs.sqlalchemy.org/en/20/",
        "flask": "https://flask.palletsprojects.com/en/latest/",
        "flask-sqlalchemy": "https://flask-sqlalchemy.palletsprojects.com/en/latest/",
    }

    def __init__(self, fetch: Callable[[str], str] | None = None, cache: dict[str, dict[str, str | bool]] | None = None) -> None:
        self.fetch = fetch or self._fetch
        self.cache = cache if cache is not None else {}

    def official_docs_search(self, package: str, symbol: str, version: str, error: str) -> ResearchResult:
        normalized = package.strip().lower()
        query = " ".join(part for part in (package.strip(), version.strip(), symbol.strip(), error.strip()) if part)
        key = query.lower()
        cached = self.cache.get(key)
        if cached:
            return ResearchResult(query, str(cached["url"]), str(cached.get("text", "")), bool(cached.get("official")), True, str(cached.get("error", "")))
        url = self.OFFICIAL.get(normalized, f"https://pypi.org/project/{normalized}/")
        try:
            text = self._compact(self.fetch(url))
            result = ResearchResult(query, url, text, normalized in self.OFFICIAL)
        except Exception as error_value:
            result = ResearchResult(query, url, error=str(error_value))
        self.cache[key] = result.to_dict()
        return result

    @staticmethod
    def _fetch(url: str) -> str:
        with urlopen(url, timeout=8) as response:
            return response.read(256_000).decode("utf-8", errors="replace")

    @staticmethod
    def _compact(value: str) -> str:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", value, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", html.unescape(text)).strip()[:6000]
