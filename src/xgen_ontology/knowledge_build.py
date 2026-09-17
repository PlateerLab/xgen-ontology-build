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


def build_resource_fragment(
    source: KnowledgeBundle | dict, *, snapshot_id: str, builder: OntologyBuilder | None = None,
    context=None,
) -> KnowledgeBundle:
    """Build the graph projection for one immutable resource revision.

    A fragment is deliberately free of embeddings.  It can therefore be built,
    retried and discarded independently from the vector projection.  Directory
    ancestors may be present to keep the resource path meaningful, but exactly
    one non-directory resource revision must own all source chunks.
    """
    source = KnowledgeBundle.from_dict(source.to_dict() if hasattr(source, "to_dict") else source)
    files = [resource for resource in source.resources if resource.kind != "directory"]
    if len(files) != 1:
        raise ContractError("a resource fragment requires exactly one non-directory resource")
    resource = files[0]
    if any(chunk.resource_id != resource.id or chunk.revision != resource.revision for chunk in source.chunks):
        raise ContractError("fragment chunks must belong to the single resource revision")
    graph_source = replace(
        source,
        profiles=(),
        embeddings=(),
        components=tuple(component for component in source.components if component != "embeddings"),
    )
    built = build_knowledge(graph_source, snapshot_id=snapshot_id, builder=builder, context=context)
    extensions = dict(built.extensions)
    extensions["xgen_ontology.fragment"] = {
        "resource_id": resource.id,
        "revision": resource.revision,
        "source_snapshot": source.snapshot_id,
    }
    return replace(built, extensions=extensions).validate()


def assemble_resource_fragments(
    source: KnowledgeBundle | dict, fragments, *, snapshot_id: str,
) -> KnowledgeBundle:
    """Assemble current resource fragments into one deterministic graph snapshot.

    ``source`` owns hierarchy, chunks and optional embeddings.  Fragments own
    only graph records.  Omitting a deleted or replaced resource's fragment is
    the retraction mechanism, so assembly never needs the deleted file bytes and
    never rebuilds unaffected resources.
    """
    source = KnowledgeBundle.from_dict(source.to_dict() if hasattr(source, "to_dict") else source)
    resources = {resource.id: resource for resource in source.resources}
    chunks = {chunk.id: chunk for chunk in source.chunks}
    evidence = {}
    entities = {}
    facts = {}
    seen_revisions = set()

    def compatible(existing, incoming, fields, name):
        if any(getattr(existing, field) != getattr(incoming, field) for field in fields):
            raise ContractError(f"conflicting {name} record: {incoming.id}")

    for raw in fragments:
        fragment = KnowledgeBundle.from_dict(raw.to_dict() if hasattr(raw, "to_dict") else raw)
        if fragment.corpus_id != source.corpus_id:
            raise ContractError("fragment corpus does not match source corpus")
        metadata = fragment.extensions.get("xgen_ontology.fragment")
        if not isinstance(metadata, dict):
            raise ContractError("bundle is not a resource fragment")
        resource_id, revision = metadata.get("resource_id"), metadata.get("revision")
        current = resources.get(resource_id)
        if current is None or current.revision != revision:
            raise ContractError(f"fragment does not reference a current resource revision: {resource_id}")
        key = (resource_id, revision)
        if key in seen_revisions:
            raise ContractError(f"duplicate resource fragment: {resource_id}@{revision}")
        seen_revisions.add(key)
        if any(item.chunk_id not in chunks for item in fragment.evidence):
            raise ContractError(f"fragment evidence is absent from source: {resource_id}@{revision}")

        for item in fragment.evidence:
            if item.id in evidence:
                compatible(evidence[item.id], item, ("chunk_id", "locator", "extensions"), "evidence")
            else:
                evidence[item.id] = item
        for item in fragment.entities:
            if item.id in entities:
                existing = entities[item.id]
                compatible(existing, item, ("label", "kind", "extensions"), "entity")
                entities[item.id] = replace(
                    existing, evidence_ids=tuple(sorted(set(existing.evidence_ids) | set(item.evidence_ids)))
                )
            else:
                entities[item.id] = item
        for item in fragment.facts:
            if item.id in facts:
                existing = facts[item.id]
                compatible(existing, item, (
                    "subject_id", "predicate", "object_id", "literal", "datatype", "assertion", "extensions",
                ), "fact")
                facts[item.id] = replace(
                    existing, evidence_ids=tuple(sorted(set(existing.evidence_ids) | set(item.evidence_ids)))
                )
            else:
                facts[item.id] = item

    extensions = dict(source.extensions)
    extensions["xgen_ontology.assembly"] = {
        "fragment_count": len(seen_revisions),
        "resource_revisions": [list(key) for key in sorted(seen_revisions)],
    }
    components = set(source.components)
    components.add("graph")
    return replace(
        source,
        snapshot_id=snapshot_id,
        evidence=tuple(evidence[key] for key in sorted(evidence)),
        entities=tuple(entities[key] for key in sorted(entities)),
        facts=tuple(facts[key] for key in sorted(facts)),
        components=tuple(sorted(components)),
        extensions=extensions,
    ).validate()


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
