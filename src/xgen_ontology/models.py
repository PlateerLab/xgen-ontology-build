"""Core data models for xgen-ontology — plain dataclasses, zero deps.

Two families:

* **Build schema** — ``Class`` / ``ObjectProperty`` / ``DataProperty`` / ``Concepts``
  (the T-Box) and ``Instance`` / ``Relation`` / ``DataValue`` (the A-Box). These are
  what the extraction + cleaning pipeline produces and mutates.
* **Graph / search** — ``Node`` / ``Chunk`` / ``RDFTriple`` / ``SearchResult`` used by
  the in-memory graph, the RDF emitters and the one-shot GraphRAG search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ───────────────────────── build schema (T-Box) ─────────────────────────


@dataclass
class Class:
    """An ontology class (``owl:Class``)."""

    name: str
    description: str = ""
    parent: str | None = None
    source_chunks: list[str] = field(default_factory=list)


@dataclass
class ObjectProperty:
    """A relation between classes (``owl:ObjectProperty``)."""

    name: str
    domain: str = ""
    range: str = ""


@dataclass
class DataProperty:
    """A literal-valued attribute (``owl:DatatypeProperty``); ``range`` is an xsd type."""

    name: str
    domain: str = ""
    range: str = "xsd:string"
    display_name: str = ""


@dataclass
class Concepts:
    """The ontology schema: classes, properties and the is-a hierarchy."""

    classes: list[Class] = field(default_factory=list)
    object_properties: list[ObjectProperty] = field(default_factory=list)
    datatype_properties: list[DataProperty] = field(default_factory=list)
    class_hierarchy: list[tuple[str, str]] = field(default_factory=list)  # (parent, child)


# ───────────────────────── build instances (A-Box) ─────────────────────────


@dataclass
class Instance:
    """A concrete individual of a class (``owl:NamedIndividual``)."""

    name: str
    class_name: str = ""
    source_chunks: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """An asserted edge between two individuals (or to a literal)."""

    subject: str
    predicate: str
    object: str
    predicate_type: str = "ObjectProperty"  # or "DatatypeProperty"
    source_chunks: list[str] = field(default_factory=list)


@dataclass
class DataValue:
    """A literal attribute value on an individual."""

    entity: str
    property: str
    value: Any
    value_type: str = "xsd:string"
    source_chunks: list[str] = field(default_factory=list)


# ───────────────────────── graph / search ─────────────────────────


@dataclass
class Node:
    """A graph node. ``kind`` is one of class | instance | property."""

    id: str
    label: str
    kind: str = "instance"


@dataclass
class Chunk:
    """A source text passage; ``entities`` are the node ids it mentions."""

    id: str
    text: str
    entities: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RDFTriple:
    """A serializable RDF statement (subject/predicate are IRIs; object may be a literal)."""

    s: str
    p: str
    o: str
    o_is_literal: bool = False
    datatype: str = ""  # e.g. "xsd:integer" — only for literals
    lang: str = ""      # e.g. "ko" — only for literals


@dataclass
class SearchResult:
    """Result of an :meth:`Ontology.search`."""

    answer: str
    question: str
    chunks: list[Chunk] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)       # "s → p → o" strings used
    evidence_nodes: list[str] = field(default_factory=list)  # node labels the answer cites
    class_seed: str = ""


@dataclass
class BuildReport:
    """Counts + diagnostics emitted by a build."""

    classes: int = 0
    object_properties: int = 0
    datatype_properties: int = 0
    instances: int = 0
    relations: int = 0
    data_values: int = 0
    renamed: int = 0           # entities/classes/props merged by dedup
    predicates_merged: int = 0
    folded: int = 0            # name fragments folded back into the name they came from
    pruned: int = 0            # unlinked common-word entities dropped
    llm_calls: int = 0
    mode: str = ""             # "basic" | "enrich" | "llm"
    quality: dict = field(default_factory=dict)   # review_quality() of the finished build
    notes: list[str] = field(default_factory=list)
