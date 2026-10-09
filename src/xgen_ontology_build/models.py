"""Core data models for xgen-ontology — plain dataclasses, zero deps.

Two families:

* **Build schema** — ``Class`` / ``ObjectProperty`` / ``DataProperty`` / ``Concepts``
  (the T-Box) and ``Instance`` / ``Relation`` / ``DataValue`` (the A-Box). These are
  what the extraction + cleaning pipeline produces and mutates.
* **Graph** — ``Node`` / ``Chunk`` / ``RDFTriple`` used by the stores, the chunk
  bookkeeping and the RDF emitters.
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
    source: str = ""   # "table" for a class built from a table's schema: a deterministic identity


@dataclass
class ObjectProperty:
    """A relation between classes (``owl:ObjectProperty``). ``description`` defines the relation
    name, so a vocabulary of relation names can be shown to a model with its meanings."""

    name: str
    domain: str = ""
    range: str = ""
    description: str = ""


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
    # A database row's identity: the key the production loader derives from the table's source
    # id, its key column and the row's key value (``row_identity_key``). It names the row even
    # when its label is a bare number, keeps it apart from a document entity spelled the same,
    # makes a row loaded again the same individual, and is its URI in a store ("dbrow_…").
    # Empty for anything that is not a row.
    identity: str = ""


@dataclass
class Relation:
    """An asserted edge between two individuals (or to a literal).

    ``label`` is how the document itself names the relation (a table's column name,
    the original spelling of a name the graph could not keep as a predicate); the
    predicate stays an identifier. ``weight`` is how many times the relation was
    asserted, and ``source_chunks`` are the chunks that assert it.
    """

    subject: str
    predicate: str
    object: str
    predicate_type: str = "ObjectProperty"  # or "DatatypeProperty"
    source_chunks: list[str] = field(default_factory=list)
    label: str | None = None
    weight: float = 1.0


@dataclass
class DataValue:
    """A literal attribute value on an individual."""

    entity: str
    property: str
    value: Any
    value_type: str = "xsd:string"
    source_chunks: list[str] = field(default_factory=list)


# ───────────────────────── graph ─────────────────────────


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
    chunks: int = 0            # chunks the ontology has seen (grows with extend())
    mode: str = ""             # "basic" | "enrich" | "llm"
    normalized: dict = field(default_factory=dict)   # normalize_graph() counts (rerouted / dropped)
    relation_stats: dict = field(default_factory=dict)   # relation formation: units, calls, vocabulary, lost
    retracted: dict = field(default_factory=dict)    # retract_chunks() counts of the last retraction
    quality: dict = field(default_factory=dict)   # review_quality() of the finished build
    notes: list[str] = field(default_factory=list)
