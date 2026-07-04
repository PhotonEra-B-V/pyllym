"""Citable tool results."""

from __future__ import annotations

from typing import Any

from .content import Content


class SearchResults(Content):
    """Tool results the model can cite.

    Providers with native citation support receive citable blocks; others get
    plain text. Return one from a tool's ``execute``::

        def execute(self, query):
            return SearchResults(title="Q4 Report", url="...", text=report_text)
    """

    def __init__(self, *results: dict[str, Any], **result: Any) -> None:
        entries = list(results)
        if result:
            entries.append(result)
        self.results = [self._normalize(entry) for entry in entries]
        if not self.results:
            raise ValueError("SearchResults requires at least one result")
        super().__init__("\n\n".join(self._format_result(e) for e in self.results))

    def format(self) -> SearchResults:
        # Stay structured so citation-capable providers can format natively.
        return self

    @staticmethod
    def _normalize(entry: dict[str, Any]) -> dict[str, Any]:
        data = dict(entry)
        if not (data.get("title") and data.get("text")):
            raise ValueError("Search results require 'title' and 'text'")
        return {k: data[k] for k in ("title", "url", "text") if k in data}

    @staticmethod
    def _format_result(entry: dict[str, Any]) -> str:
        attributes = f"title='{entry['title']}'"
        if entry.get("url"):
            attributes += f" url='{entry['url']}'"
        return f"<search_result {attributes}>\n{entry['text']}\n</search_result>"
