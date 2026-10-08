"""Pluggable backend interfaces — structural typing, no inheritance required.

The build pipeline and the search engine only ever talk to these Protocols, never
to a concrete DB / LLM / analyzer. Ship the in-memory backends (zero infra) by
default; swap in any SPARQL store, any LLM, any morphological analyzer or embedder
by passing objects that match these shapes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from xgen_ontology_build.models import Node


@runtime_checkable
class LLM(Protocol):
    """A language model. ``generate`` is used for search synthesis; the build stages
    additionally expect JSON back (parsed leniently from ``generate``'s output)."""

    def generate(self, prompt: str, *, system: str = "", timeout: float | None = None) -> str:
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Read-side knowledge-graph operations the search algorithm needs."""

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        """Full-text search over node labels. Returns (node, score) desc."""

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        """Enumerate all instances of a class (the structural 'count/list' power)."""

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        """1-hop (or N-hop) relations around a node as (subj_label, predicate, obj_label)."""

    def count_class(self, class_id: str) -> int:
        ...

    def get_node(self, node_id: str) -> Node | None:
        ...


@runtime_checkable
class GraphSink(Protocol):
    """Write-side: load a built ontology (as Turtle) into any triple store."""

    def upload_turtle(self, ttl: str, *, graph: str | None = None, clear: bool = False) -> None:
        ...


@runtime_checkable
class Morphology(Protocol):
    """Optional morphological analyzer used to normalize names for dedup/governance.

    English-neutral default is *no* morphology (a cleaned-lowercase key). Plug a
    Korean (Kiwi) / Japanese / etc. analyzer to fold inflected forms together."""

    def nouns(self, text: str) -> list[str]:
        """Return the content morphemes (noun-like tokens) of ``text``."""


@runtime_checkable
class Embedder(Protocol):
    """Optional embedding provider for semantic (meaning-level) dedup + vector search."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
