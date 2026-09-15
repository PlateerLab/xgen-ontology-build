"""Build portable knowledge with the production pipeline, without retrieval imports."""
from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import asdict, replace
from pathlib import PurePosixPath

from .build.pipeline import OntologyBuilder
from .knowledge import (
    ContractError,
    Evidence,
    GraphEntity,
    GraphFact,
    KnowledgeBundle,
    OperationContext,
    canonical_json,
    stable_id,
)


def build_knowledge(
    source: KnowledgeBundle | dict, *, snapshot_id: str, builder: OntologyBuilder | None = None,
    context=None,
) -> KnowledgeBundle:
    """Build a complete replacement graph for an immutable source snapshot.

    Source resources/chunks/embeddings survive unchanged. Replacing or deleting a
    source is handled by building its *complete current* corpus again; the legacy
    additive ``extend`` API is intentionally not used for retractions. Publication
    is the host's CAS transaction, not a side effect of this function.
    """
    source = KnowledgeBundle.from_dict(source.to_dict() if hasattr(source, "to_dict") else source)
    context = context or OperationContext()
    engine = copy(builder) if builder is not None else OntologyBuilder()
    if engine.mode != "basic" and engine.llm is None:
        raise ContractError(f"mode {engine.mode!r} requires an LLM provider")
    if context is not None:
        tick = engine._tick

        def checked_tick(stage, **detail):
            context.check()
            tick(stage, **detail)
            context.check()

        engine._tick = checked_tick
        context.check()
    resources = {r.id: r for r in source.resources}
    documents = defaultdict(list)
    for chunk in sorted(source.chunks, key=lambda c: (c.resource_id, c.ordinal, c.id)):
        # Stable source ID is part of the key, retaining the extension used for
        # table routing. A displayed filename can occur in multiple directories.
        name = resources[chunk.resource_id].name
        suffix = PurePosixPath(name).suffix
        key = f"{stable_id('source', chunk.resource_id)}/{name}" if suffix else f"{chunk.resource_id}.txt"
        documents[key].append({"chunk_id": chunk.id, "chunk_text": chunk.text, "chunk_index": chunk.ordinal})
    ontology = engine.build(dict(documents))
    if context is not None:
        context.check()
    return export_knowledge(ontology, source, snapshot_id=snapshot_id)


def export_knowledge(ontology, source: KnowledgeBundle | dict, *, snapshot_id: str) -> KnowledgeBundle:
    """Convert a built ontology using exact caller-owned source identities.

    No guessed revision, page, embedding or evidence is manufactured. Legacy
    ontology data lacking support is diagnosed and omitted from published facts.
    The lossless build schema/report remain available in namespaced extensions.
    """
    source = KnowledgeBundle.from_dict(source.to_dict() if hasattr(source, "to_dict") else source)
    chunks = {c.id: c for c in source.chunks}
    for chunk in ontology.chunks:
        if chunk.id not in chunks or chunks[chunk.id].text != chunk.text:
            raise ContractError(f"ontology chunk is absent/different in source snapshot: {chunk.id}")
    evidence = {cid: Evidence(stable_id("evidence", source.corpus_id, cid), cid) for cid in chunks}
    supports: dict[tuple[str, str], set[str]] = defaultdict(set)
    labels: dict[tuple[str, str], str] = {}
    class_names = {c.name for c in ontology.concepts.classes}
    class_names.update(x for pair in ontology.concepts.class_hierarchy for x in pair)
    diagnostics = []

    def refs(ids):
        unknown = set(ids) - chunks.keys()
        if unknown:
            raise ContractError(f"build returned unknown evidence chunks: {sorted(unknown)}")
        return {evidence[c].id for c in ids}

    def entity(name, kind="instance", ids=()):
        key = (kind, name)
        labels[key] = name
        supports[key].update(refs(ids))
        return key

    for c in ontology.concepts.classes:
        entity(c.name, "class", c.source_chunks)
    for name in class_names:
        entity(name, "class")
    for inst in ontology.instances:
        entity(inst.name, ids=inst.source_chunks)
        if inst.class_name:
            entity(inst.class_name, "class", inst.source_chunks)
    for relation in ontology.relations:
        entity(relation.subject, ids=relation.source_chunks)
        if relation.predicate_type != "DatatypeProperty":
            entity(relation.object, ids=relation.source_chunks)
    for value in ontology.data_values:
        entity(value.entity, ids=value.source_chunks)

    facts = {}

    def add(subject, predicate, obj=None, *, literal=None, datatype=None, ids=(), inferred=False):
        support = refs(ids)
        if not support:
            # These are dependencies of a derived statement, not an exact quote.
            support = supports[subject] | (supports[obj] if obj else set())
            inferred = True
        if not support:
            diagnostics.append({"subject": subject[1], "predicate": predicate, "reason": "missing_support"})
            return
        sid = stable_id("entity", source.corpus_id, *subject)
        oid = stable_id("entity", source.corpus_id, *obj) if obj else None
        fid = stable_id("fact", source.corpus_id, sid, predicate, oid or canonical_json([datatype, literal]))
        if fid in facts:
            support |= set(facts[fid].evidence_ids)
        facts[fid] = GraphFact(fid, sid, predicate, oid, literal, datatype, tuple(sorted(support)),
                               "inferred" if inferred else "extracted")
        supports[subject].update(support)
        if obj:
            supports[obj].update(support)

    for inst in ontology.instances:
        if inst.class_name:
            add(("instance", inst.name), "instanceOf", ("class", inst.class_name), ids=inst.source_chunks)
    for parent, child in ontology.concepts.class_hierarchy:
        add(("class", child), "subClassOf", ("class", parent), inferred=True)
    for relation in ontology.relations:
        if relation.predicate_type == "DatatypeProperty":
            add(("instance", relation.subject), relation.predicate, literal=relation.object,
                datatype="xsd:string", ids=relation.source_chunks)
        else:
            add(("instance", relation.subject), relation.predicate, ("instance", relation.object),
                ids=relation.source_chunks)
    for value in ontology.data_values:
        add(("instance", value.entity), value.property, literal=value.value,
            datatype=value.value_type, ids=value.source_chunks)
    entities = tuple(GraphEntity(stable_id("entity", source.corpus_id, *key), label, key[0],
                                 tuple(sorted(supports[key]))) for key, label in sorted(labels.items()))
    extensions = dict(source.extensions)
    extensions["xgen_ontology.build"] = {
        "source_snapshot": source.snapshot_id, "source_digest": source.content_digest,
        "report": asdict(ontology.report), "schema": asdict(ontology.concepts),
        "unsupported_facts": diagnostics, "graph_complete": not diagnostics,
        "update_mode": "full_replacement",
    }
    return replace(source, snapshot_id=snapshot_id, entities=entities,
                   facts=tuple(sorted(facts.values(), key=lambda f: f.id)), evidence=tuple(evidence.values()),
                   components=tuple(sorted(set(source.components) | {"graph"})), extensions=extensions).validate()
