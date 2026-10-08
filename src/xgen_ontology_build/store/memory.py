"""An in-memory graph sink: keeps the uploaded Turtle, for tests and dry runs."""
from __future__ import annotations


class InMemoryGraphSink:
    """A GraphSink that just keeps the uploaded Turtle (for tests / dry runs)."""

    def __init__(self):
        self.graphs: dict[str | None, str] = {}

    def upload_turtle(self, ttl: str, *, graph: str | None = None, clear: bool = False) -> None:
        if clear or graph not in self.graphs:
            self.graphs[graph] = ttl
        else:
            self.graphs[graph] += "\n" + ttl
