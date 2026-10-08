"""Build-quality review — a graph-reviewer that scores a finished ontology.

Four weighted dimensions (knowledge-graph usefulness order):

* **completeness** (0.40) — fraction of classes that actually have instances.
* **integrity**    (0.25) — no dangling edges (an end that is not a node).
* **grounding**    (0.20) — relations that state their source chunks or whose endpoints
  co-occur in one (cheap anti-hallucination check; skipped/full marks when no markers).
* **shape**        (0.15) — SHACL-like hygiene: untyped instances, ungoverned
  predicates, domain violations (penalized by *ratio*, not raw count).

Also counted, not scored: ``orphan_class_count`` (classes nothing instantiates and no
object property declares), the production review's first warning. ``relation_count`` and
the ungoverned predicates are judged by declaration here; the production review reads
them off schema edges, which exist only for properties with a domain and a range.

All in-memory over the build models — no SPARQL, no infra.
"""
from __future__ import annotations

import re

from ..models import Concepts, DataValue, Instance, Relation

QUALITY_WEIGHTS = {"completeness": 0.40, "integrity": 0.25, "grounding": 0.20, "shape": 0.15}
SHAPE_SUBWEIGHTS = {"untyped": 0.4, "ungoverned": 0.3, "domain": 0.3}
WARN_COMPLETENESS_PCT = 50.0
WARN_GROUNDING_PCT = 80.0


def review_quality(
    concepts: Concepts,
    instances: list[Instance],
    relations: list[Relation],
    data_values: list[DataValue],
) -> dict:
    class_names = {c.name for c in concepts.classes if c.name}
    class_count = len(class_names)
    inst_names = {i.name for i in instances if i.name}
    instance_count = len(inst_names)
    relation_count = len({op.name for op in concepts.object_properties if op.name})

    inst_class = {i.name: i.class_name for i in instances if i.name}
    classes_with_instances = {i.class_name for i in instances if i.class_name}
    parents = {p for p, _ in concepts.class_hierarchy}
    prop_domains = ({op.domain for op in concepts.object_properties if op.domain}
                    | {dp.domain for dp in concepts.datatype_properties if dp.domain})
    classes_without_instance = sum(
        1 for c in class_names
        if c not in classes_with_instances and c not in parents and c not in prop_domains
    )
    completeness_pct = round(100.0 * (class_count - classes_without_instance) / class_count, 1) if class_count else 0.0

    obj_rels = [r for r in relations if r.predicate_type != "DatatypeProperty"]
    nodes = inst_names | class_names
    dangling = sum(1 for r in obj_rels if (r.subject and r.subject not in nodes) or (r.object and r.object not in nodes))
    integrity_ok = dangling == 0
    # a class nothing instantiates and no object property declares a domain or range of
    # (the production review's first warning)
    op_ends = ({op.domain for op in concepts.object_properties if op.domain}
               | {op.range for op in concepts.object_properties if op.range})
    orphan_classes = sum(1 for c in class_names if c not in classes_with_instances and c not in op_ends)

    # shape
    untyped = sum(1 for i in instances if i.name and (not i.class_name or i.class_name not in class_names))
    declared_props = {op.name for op in concepts.object_properties if op.name}
    used_preds = {r.predicate for r in obj_rels if r.predicate}
    ungoverned = len(used_preds - declared_props)
    op_domain = {op.name: op.domain for op in concepts.object_properties if op.name and op.domain}
    domain_violation = 0
    for r in obj_rels:
        dom = op_domain.get(r.predicate)
        if dom and r.subject in inst_class and inst_class[r.subject] != dom:
            domain_violation += 1
    shape_ok = (untyped + ungoverned + domain_violation) == 0

    # grounding: the relation states its own chunks, or its two ends share one (the stored
    # graph links a relation's chunks to both ends, so a stated relation is grounded there)
    chunk_of: dict[str, set] = {i.name: set(i.source_chunks or []) for i in instances if i.name}
    has_markers = any(chunk_of.get(n) for n in inst_names) or any(r.source_chunks for r in obj_rels)
    total_inst_relations = sum(1 for r in obj_rels if r.subject in inst_names and r.object in inst_names)
    if total_inst_relations and has_markers:
        ungrounded = sum(
            1 for r in obj_rels
            if r.subject in inst_names and r.object in inst_names and not r.source_chunks
            and not (chunk_of.get(r.subject, set()) & chunk_of.get(r.object, set()))
        )
        grounding_pct = round(100.0 * (total_inst_relations - ungrounded) / total_inst_relations, 1)
    else:
        ungrounded = 0
        grounding_pct = 100.0

    # scores (ratio-based penalties so larger graphs aren't unfairly punished)
    ref_total = total_inst_relations + dangling
    integrity_score = 100.0 * (1 - dangling / ref_total) if ref_total else 100.0
    untyped_ratio = untyped / instance_count if instance_count else 0.0
    pred_total = relation_count + ungoverned
    ungoverned_ratio = ungoverned / pred_total if pred_total else 0.0
    domain_ratio = min(1.0, domain_violation / total_inst_relations) if total_inst_relations else 0.0
    shape_score = 100.0 * (1 - min(1.0,
                                   untyped_ratio * SHAPE_SUBWEIGHTS["untyped"]
                                   + ungoverned_ratio * SHAPE_SUBWEIGHTS["ungoverned"]
                                   + domain_ratio * SHAPE_SUBWEIGHTS["domain"]))
    score = round(
        completeness_pct * QUALITY_WEIGHTS["completeness"]
        + integrity_score * QUALITY_WEIGHTS["integrity"]
        + shape_score * QUALITY_WEIGHTS["shape"]
        + grounding_pct * QUALITY_WEIGHTS["grounding"], 1)

    warnings: list[str] = []
    if dangling:
        warnings.append(f"{dangling} dangling reference(s)")
    if class_count and completeness_pct < WARN_COMPLETENESS_PCT:
        warnings.append(f"low completeness: {completeness_pct}% of classes have instances")
    if untyped:
        warnings.append(f"{untyped} untyped instance(s)")
    if ungoverned:
        warnings.append(f"{ungoverned} ungoverned predicate(s)")
    if domain_violation:
        warnings.append(f"{domain_violation} domain violation(s)")
    if total_inst_relations and grounding_pct < WARN_GROUNDING_PCT:
        warnings.append(f"weak grounding: only {grounding_pct}% of relations co-occur in source")
    if orphan_classes:
        warnings.append(f"{orphan_classes} orphan class(es)")

    return {
        "class_count": class_count,
        "instance_count": instance_count,
        "relation_count": relation_count,
        "classes_without_instance": classes_without_instance,
        "orphan_class_count": orphan_classes,
        "completeness_pct": completeness_pct,
        "dangling_edge_count": dangling,
        "integrity_ok": integrity_ok,
        "shape_violations": {"untyped_instance": untyped, "ungoverned_predicate": ungoverned,
                             "domain_violation": domain_violation},
        "shape_ok": shape_ok,
        "grounding_pct": grounding_pct,
        "ungrounded_relations": ungrounded,
        "score": score,
        "warnings": warnings,
    }


_NUMERIC_LABEL = re.compile(r"^[\d\-.\/\s,]+$")


def is_value_like(label: str) -> bool:
    """True for labels that are pure date/number/separator or a single char (not entities)."""
    s = (label or "").strip()
    return bool(s) and (bool(_NUMERIC_LABEL.fullmatch(s)) or len(s) < 2)
