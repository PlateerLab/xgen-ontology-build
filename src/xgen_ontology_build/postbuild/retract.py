"""Retraction: take deleted chunks out of a built graph without rebuilding it.

The production store applies a document deletion as a delta (``prune_chunks``):
the chunk links go, and with them whatever they were the only evidence for. Ported
here over the in-memory build models, so :meth:`OntologyBuilder.extend
<xgen_ontology.OntologyBuilder.extend>` can retract before it extracts and the
post-build sees the current corpus. The rules, in the store's order:

1. an entity's *evidence* is every chunk that links it: its own source chunks, the
   chunks of the relations it takes part in (both ends) and the chunks of its
   attributes. ``touched`` = the names that lose at least one chunk
2. a relation with source chunks of its own is judged by them: it is dropped when
   every one of them was removed, and keeps the rest otherwise. A relation with no
   source chunks between two touched names whose ends no longer share a surviving
   chunk has lost its evidence and is dropped, except the links made from name
   structure (``structural_predicates``: the "related" link, ``sameAs``), which are
   not chunk evidence and stay. (The production store judges every relation by its
   ends' co-occurrence; a relation's own sources are the stricter evidence.)
3. a touched individual with no evidence left is an orphan
4. a class with no evidence left is an orphan unless something still refers to it:
   a surviving individual typed by it, a subclass, a property that ranges over it,
   a relation. Its own declarations (its properties, its own parent link) are not
   references. Evaluated to a fixpoint: a class whose last reference was an orphan
   (its only individual, its only subclass) is an orphan too
5. a property declared only by orphan classes goes with them; relations and
   attributes touching an orphan go with it
"""
from __future__ import annotations

from xgen_ontology_build.models import Concepts, DataValue, Instance, Relation


def retract_chunks(concepts: Concepts, instances: list[Instance], relations: list[Relation],
                   data_values: list[DataValue], chunk_ids, *, structural_predicates=()) -> dict:
    """Remove the traces of ``chunk_ids`` from the build models in place.

    Returns the counts: ``chunks`` asked for, ``links`` (chunk references removed),
    ``instances`` / ``classes`` / ``properties`` (declarations dropped), ``relations``
    and ``data_values`` dropped. Surviving elements keep only their surviving chunks.
    """
    gone_chunks = {str(c) for c in (chunk_ids or []) if c}
    stats = {"chunks": len(gone_chunks), "links": 0, "instances": 0, "classes": 0, "properties": 0,
             "relations": 0, "data_values": 0}
    if not gone_chunks:
        return stats
    structural = set(structural_predicates or ())
    class_names = {c.name for c in concepts.classes if c.name}

    evidence: dict[str, set[str]] = {}

    def link(name: str, cids) -> None:
        if name:
            evidence.setdefault(name, set()).update(c for c in (cids or []) if c)

    for i in instances:
        link(i.name, i.source_chunks)
    for c in concepts.classes:
        link(c.name, c.source_chunks)
    for r in relations:
        if r.predicate_type != "DatatypeProperty":
            link(r.subject, r.source_chunks)
            link(r.object, r.source_chunks)
    for d in data_values:
        link(d.entity, d.source_chunks)

    touched = {n for n, cids in evidence.items() if cids & gone_chunks}
    if not touched:
        return stats
    stats["links"] = sum(len(cids & gone_chunks) for cids in evidence.values())
    remaining = {n: cids - gone_chunks for n, cids in evidence.items()}

    def keep_chunks(cids) -> list[str]:
        return [c for c in (cids or []) if c not in gone_chunks]

    # 2. relations whose only co-occurrence was in the removed chunks
    kept_relations: list[Relation] = []
    for r in relations:
        own = {c for c in r.source_chunks if c}
        if r.predicate_type != "DatatypeProperty" and own:
            if own <= gone_chunks:
                stats["relations"] += 1          # every chunk that asserted it is gone
                continue
            r.source_chunks = keep_chunks(r.source_chunks)
        elif (r.predicate_type != "DatatypeProperty" and r.predicate not in structural
                and r.subject in touched and r.object in touched
                and not (remaining.get(r.subject, set()) & remaining.get(r.object, set()))):
            stats["relations"] += 1
            continue
        kept_relations.append(r)

    # 3. individuals (and bare relation endpoints) with no evidence left
    orphan_inst = {n for n in touched if not remaining[n] and n not in class_names}
    gone = set(orphan_inst)
    stats["instances"] = len({i.name for i in instances if i.name in orphan_inst})

    # 4./5. classes: those that lost a chunk, then those that lost their last reference
    candidates = {n for n in touched if n in class_names} | {
        i.class_name for i in instances if i.name in orphan_inst and i.class_name}
    n_props = len(concepts.object_properties) + len(concepts.datatype_properties)
    while candidates:
        referenced: set[str] = set()
        for i in instances:
            if i.class_name and i.name not in gone:
                referenced.add(i.class_name)
        for parent, child in concepts.class_hierarchy:
            if child != parent and child not in gone:
                referenced.add(parent)
        for c in concepts.classes:
            if c.parent and c.parent != c.name and c.name not in gone:
                referenced.add(c.parent)
        for op in concepts.object_properties:
            if op.range and op.range != op.domain and op.domain not in gone:
                referenced.add(op.range)
        for r in kept_relations:
            if r.predicate_type != "DatatypeProperty" and r.subject not in gone and r.object not in gone:
                referenced.update((r.subject, r.object))
        orphan_cls = {n for n in candidates
                      if n in class_names and n not in gone and not remaining.get(n) and n not in referenced}
        if not orphan_cls:
            break
        gone |= orphan_cls
        stats["classes"] += len(orphan_cls)
        candidates = {p for p, ch in concepts.class_hierarchy if ch in orphan_cls}
        candidates |= {c.parent for c in concepts.classes if c.name in orphan_cls and c.parent}
        candidates |= {op.range for op in concepts.object_properties if op.domain in orphan_cls and op.range}
        used_predicates = {r.predicate for r in kept_relations if r.subject not in gone and r.object not in gone}
        used_attributes = {d.property for d in data_values if d.entity not in gone}
        concepts.classes = [c for c in concepts.classes if c.name not in orphan_cls]
        for c in concepts.classes:
            if c.parent in orphan_cls:
                c.parent = None
        concepts.class_hierarchy = [(p, ch) for p, ch in concepts.class_hierarchy
                                    if p not in orphan_cls and ch not in orphan_cls]
        concepts.object_properties = [
            op for op in concepts.object_properties
            if not ((op.domain in orphan_cls or op.range in orphan_cls) and op.name not in used_predicates)]
        concepts.datatype_properties = [
            dp for dp in concepts.datatype_properties
            if not (dp.domain in orphan_cls and dp.name not in used_attributes)]
        candidates -= gone
    stats["properties"] = n_props - len(concepts.object_properties) - len(concepts.datatype_properties)

    for c in concepts.classes:
        if c.name in touched:
            c.source_chunks = keep_chunks(c.source_chunks)

    instances[:] = [i for i in instances if i.name not in gone]
    for i in instances:
        if i.name in touched:
            i.source_chunks = keep_chunks(i.source_chunks)

    survivors: list[Relation] = []
    for r in kept_relations:
        if r.subject in gone or r.object in gone:
            stats["relations"] += 1
            continue
        if r.subject in touched or r.object in touched:
            own = keep_chunks(r.source_chunks)
            r.source_chunks = own or sorted(remaining.get(r.subject, set()) & remaining.get(r.object, set()))
        survivors.append(r)
    relations[:] = survivors

    kept_values: list[DataValue] = []
    for d in data_values:
        if d.entity in gone:
            stats["data_values"] += 1
            continue
        if d.entity in touched:
            d.source_chunks = keep_chunks(d.source_chunks)
        kept_values.append(d)
    data_values[:] = kept_values
    return stats
