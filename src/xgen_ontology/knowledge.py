"""knowledge/v1: portable, dependency-free records shared by build and retrieval.

Generated from the contract source in xgen-ontology-build. Do not edit a consumer
copy. IDs are opaque, labels/paths are display data, and revisions are immutable.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.0"


class ContractError(ValueError):
    """Invalid knowledge data; never equivalent to an empty search result."""


class SnapshotConflict(ContractError):
    """A writer or reader used a different snapshot than the one required."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"not portable JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(kind: str, *parts: str) -> str:
    return f"{kind}:{digest(list(parts))}"


def _text(value: Any, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ContractError(f"{name} must be {'a' if empty else 'a nonempty'} string")


def _unique(records: tuple, name: str) -> dict:
    out = {}
    for record in records:
        _text(record.id, f"{name}.id")
        if record.id in out:
            raise ContractError(f"duplicate {name} id: {record.id}")
        out[record.id] = record
    return out


@dataclass(frozen=True)
class Resource:
    id: str
    revision: str
    name: str
    kind: str = "file"
    parent_id: str | None = None
    media_type: str = "text/plain"
    content_digest: str = ""
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceChunk:
    id: str
    resource_id: str
    revision: str
    extraction_id: str
    text: str
    ordinal: int = 0
    title: str = ""
    locator: dict = field(default_factory=dict)
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingProfile:
    id: str
    model: str
    model_revision: str
    dimension: int
    preprocessing: str = "identity"
    metric: str = "cosine"
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkEmbedding:
    chunk_id: str
    profile_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class Evidence:
    id: str
    chunk_id: str
    # Optional refinement inside the chunk; {} means the whole chunk, not a
    # fabricated exact span. SourceChunk carries the original source locator.
    locator: dict = field(default_factory=dict)
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEntity:
    id: str
    label: str
    kind: str = "instance"
    evidence_ids: tuple[str, ...] = ()
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphFact:
    id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    literal: Any = None
    datatype: str | None = None
    evidence_ids: tuple[str, ...] = ()
    assertion: str = "extracted"
    extensions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeBundle:
    corpus_id: str
    snapshot_id: str
    resources: tuple[Resource, ...] = ()
    chunks: tuple[SourceChunk, ...] = ()
    profiles: tuple[EmbeddingProfile, ...] = ()
    embeddings: tuple[ChunkEmbedding, ...] = ()
    entities: tuple[GraphEntity, ...] = ()
    facts: tuple[GraphFact, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    # Presence is explicit: an empty graph can be complete, or not built at all.
    components: tuple[str, ...] = ("hierarchy", "content")
    contract_version: str = CONTRACT_VERSION
    extensions: dict = field(default_factory=dict)

    def validate(self) -> KnowledgeBundle:
        _text(self.corpus_id, "corpus_id")
        _text(self.snapshot_id, "snapshot_id")
        if self.contract_version != CONTRACT_VERSION:
            raise ContractError(f"unsupported contract: {self.contract_version}")
        if len(set(self.components)) != len(self.components) or set(self.components) - {
            "hierarchy", "content", "embeddings", "graph"
        }:
            raise ContractError("invalid components")
        canonical_json(asdict(self))
        resources = _unique(self.resources, "resource")
        chunks = _unique(self.chunks, "chunk")
        profiles = _unique(self.profiles, "profile")
        entities = _unique(self.entities, "entity")
        evidence = _unique(self.evidence, "evidence")
        _unique(self.facts, "fact")
        if (self.entities or self.facts or self.evidence) and "graph" not in self.components:
            raise ContractError("graph data requires graph component")
        if (self.embeddings or self.profiles) and "embeddings" not in self.components:
            raise ContractError("embedding data requires embeddings component")
        if self.chunks and "content" not in self.components:
            raise ContractError("chunks require content component")
        for r in self.resources:
            _text(r.revision, "resource.revision")
            _text(r.name, "resource.name")
            if r.kind not in {"directory", "file", "document", "record"}:
                raise ContractError(f"unsupported resource kind: {r.kind}")
            if r.parent_id is not None:
                if "hierarchy" not in self.components:
                    raise ContractError("parent_id requires hierarchy component")
                if r.parent_id not in resources or resources[r.parent_id].kind != "directory":
                    raise ContractError(f"missing/non-directory parent: {r.parent_id}")
        # Linear-time cycle detection, including very deep directory trees.
        done = set()
        for rid in resources:
            visiting = set()
            while rid is not None and rid not in done:
                if rid in visiting:
                    raise ContractError("hierarchy cycle")
                visiting.add(rid)
                rid = resources[rid].parent_id
            done.update(visiting)
        for c in self.chunks:
            r = resources.get(c.resource_id)
            if r is None or r.kind == "directory" or r.revision != c.revision:
                raise ContractError(f"chunk source revision mismatch: {c.id}")
            _text(c.extraction_id, "extraction_id")
            _text(c.text, "chunk.text", empty=True)
            _text(c.title, "chunk.title", empty=True)
            if type(c.ordinal) is not int or c.ordinal < 0:
                raise ContractError("ordinal must be a nonnegative integer")
            validate_locator(c.locator)
        for p in self.profiles:
            for name in ("model", "model_revision", "preprocessing"):
                _text(getattr(p, name), f"profile.{name}")
            if type(p.dimension) is not int or p.dimension < 1 or p.metric not in {"cosine", "dot", "euclidean"}:
                raise ContractError("invalid embedding profile")
        pairs = set()
        for e in self.embeddings:
            if e.chunk_id not in chunks or e.profile_id not in profiles:
                raise ContractError("embedding references missing chunk/profile")
            pair = (e.chunk_id, e.profile_id)
            if pair in pairs:
                raise ContractError("duplicate chunk embedding profile")
            pairs.add(pair)
            validate_vector(e.values, profiles[e.profile_id])
        for e in self.evidence:
            if e.chunk_id not in chunks:
                raise ContractError(f"missing evidence chunk: {e.chunk_id}")
            validate_locator(e.locator, text=chunks[e.chunk_id].text)
        for e in self.entities:
            _text(e.label, "entity.label")
            if e.kind not in {"instance", "class", "property"}:
                raise ContractError("invalid entity kind")
            if set(e.evidence_ids) - evidence.keys():
                raise ContractError(f"missing entity evidence: {e.id}")
        for f in self.facts:
            if f.subject_id not in entities or (f.object_id is not None and f.object_id not in entities):
                raise ContractError(f"missing fact endpoint: {f.id}")
            _text(f.predicate, "fact.predicate")
            if f.object_id is None:
                _text(f.datatype, "literal.datatype")
            elif f.literal is not None or f.datatype is not None:
                raise ContractError("fact must have either an entity object or a typed literal")
            if f.assertion not in {"extracted", "inferred", "manual"}:
                raise ContractError("invalid assertion kind")
            if not f.evidence_ids or set(f.evidence_ids) - evidence.keys():
                raise ContractError(f"fact requires resolvable evidence: {f.id}")
        return self

    def to_dict(self) -> dict:
        self.validate()
        # The JSON boundary deliberately avoids the two packages' class identity.
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict) -> KnowledgeBundle:
        if not isinstance(value, dict):
            raise ContractError("bundle must be an object")
        value = json.loads(canonical_json(value))
        classes = {
            "resources": Resource, "chunks": SourceChunk, "profiles": EmbeddingProfile,
            "embeddings": ChunkEmbedding, "entities": GraphEntity, "facts": GraphFact, "evidence": Evidence,
        }
        try:
            for key, record_type in classes.items():
                rows = []
                for row in value.get(key, []):
                    for field_name in ("evidence_ids", "values"):
                        if field_name in row:
                            row[field_name] = tuple(row[field_name])
                    rows.append(record_type(**row))
                value[key] = tuple(rows)
            if "components" in value:
                value["components"] = tuple(value["components"])
            return cls(**value).validate()
        except (TypeError, KeyError, AttributeError) as exc:
            raise ContractError(f"malformed bundle: {exc}") from exc

    @property
    def content_digest(self) -> str:
        return digest(self.to_dict())

    def dump(self, path: str | Path) -> None:
        """Write a checksummed JSON exchange file (no executable deserialization)."""
        import os
        import tempfile
        data = self.to_dict()
        destination = Path(path)
        payload = canonical_json({"digest": digest(data), "bundle": data})
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeBundle:
        try:
            envelope = json.loads(Path(path).read_text(encoding="utf-8"))
            if set(envelope) != {"digest", "bundle"} or digest(envelope["bundle"]) != envelope["digest"]:
                raise ContractError("bundle digest mismatch")
            return cls.from_dict(envelope["bundle"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError(f"invalid bundle file: {exc}") from exc


def validate_vector(values, profile: EmbeddingProfile) -> None:
    try:
        invalid = len(values) != profile.dimension or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values
        )
    except (TypeError, OverflowError) as exc:
        raise ContractError("invalid numeric vector") from exc
    if invalid:
        raise ContractError("embedding dimension/non-finite value mismatch")
    if profile.metric == "cosine" and not any(values):
        raise ContractError("zero vector has no cosine direction")


def validate_locator(locator: dict, *, text: str | None = None) -> None:
    if not isinstance(locator, dict):
        raise ContractError("locator must be an object")
    if not locator:
        return
    kind = locator.get("kind")
    if kind == "text_span":
        start, end = locator.get("start"), locator.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ContractError("invalid Unicode code-point [start,end) locator")
        if text is not None and end > len(text):
            raise ContractError("evidence span exceeds chunk text")
    elif kind == "page":
        if type(locator.get("page")) is not int or locator["page"] < 1:
            raise ContractError("page is one-based")
    elif kind == "json_pointer":
        pointer = locator.get("pointer")
        if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
            raise ContractError("invalid JSON pointer")
    elif kind == "table_cell":
        if any(type(locator.get(k)) is not int or locator[k] < 0 for k in ("row", "column")):
            raise ContractError("table coordinates are zero-based")
    elif kind == "media_time":
        start, end = locator.get("start_ms"), locator.get("end_ms")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ContractError("invalid millisecond interval")
    else:
        raise ContractError(f"unsupported locator kind: {kind}")


class OperationCancelled(RuntimeError):
    """Cooperative cancellation; partial work must not be published."""


class BudgetExceeded(RuntimeError):
    """The shared operation budget was exhausted."""


class OperationContext:
    """Thread-safe cancellation, deadline and call budget for either library.

    A deadline is measured with time.monotonic(). Third-party blocking calls must
    also configure transport timeouts; cancellation is checked between calls.
    """
    def __init__(self, *, deadline: float | None = None, max_calls: int = 1000):
        from threading import Event, Lock
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.cancelled = Event()
        self.deadline = deadline
        self.max_calls = max_calls
        self._calls = 0
        self._lock = Lock()

    def check(self) -> None:
        from time import monotonic
        if self.cancelled.is_set():
            raise OperationCancelled("knowledge operation cancelled")
        if self.deadline is not None and monotonic() >= self.deadline:
            raise TimeoutError("knowledge operation deadline exceeded")

    def consume(self) -> None:
        self.check()
        with self._lock:
            if self._calls >= self.max_calls:
                raise BudgetExceeded("knowledge provider call budget exceeded")
            self._calls += 1
