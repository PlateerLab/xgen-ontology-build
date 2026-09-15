"""Graph normalization: the store-loading rules of the production build, applied in memory.

Before a build is loaded into a store, the production pipeline decides what each
extracted item *is*: a class must look like a class name, a relation whose name is
graph vocabulary (``type``, ``instanceOf``, ``subClassOf``, ``sameAs``) is a typing
statement rather than a relation, a relation whose object is a value is an
attribute, a subject that is a value is nothing at all, and anything referenced
must exist (no dangling endpoints). :func:`normalize_graph` applies the same rules
to the build models so that what a caller emits or searches is what the production
graph would hold.
"""
from __future__ import annotations

import re

from ..korean import normalize_label
from ..models import Class, Concepts, DataValue, Instance, Relation
from .deterministic import is_class_name

# Relation names the graph uses as structure. A relation carrying one is a typing statement.
RESERVED_PREDICATES = {
    "instanceOf": "instanceOf", "instance_of": "instanceOf", "isA": "instanceOf",
    "is_a": "instanceOf", "type": "instanceOf", "rdf:type": "instanceOf",
    "subClassOf": "subClassOf", "subclass_of": "subClassOf",
    "sameAs": "sameAs", "same_as": "sameAs",
}
_DATATYPES = {"string", "integer", "decimal", "date", "boolean", "number"}


# A value even with a unit, currency sign or bracket attached.
_VALUE_SHAPE = re.compile(r"^[\s(\[]*[+\-]?[\d,.]+\s*[^\d]{0,4}[)\]]*$")


def is_value_like(s) -> bool:
    """A value (no letters, or a number with at most a short unit) rather than an entity."""
    t = ("" if s is None else str(s)).strip()
    if not t:
        return False
    if not any(ch.isalpha() for ch in t):
        return True
    return bool(_VALUE_SHAPE.match(t))


def _is_datatype(name: str) -> bool:
    n = str(name or "").strip().lower()
    return n.startswith("xsd:") or n in _DATATYPES


def normalize_graph(concepts: Concepts, instances: list[Instance], relations: list[Relation],
                    data_values: list[DataValue], *, structural_predicates=()) -> dict:
    """Apply the store-loading rules in place. Returns counts of what was rerouted or dropped.

    * class, property and hierarchy declarations whose names are not class-shaped
      (a bare number, a sentence, an enumerator) are dropped; classes that are only
      referenced (as a type, domain, range or parent) are declared
    * a relation named with graph vocabulary becomes what it says: ``instanceOf``
      types the subject, ``subClassOf`` links two classes, ``sameAs`` folds the
      subject into the object
    * a relation whose predicate is a declared datatype property, or whose object is
      a value, becomes a data value; a relation or data value whose subject is a
      value is dropped
    * a name that occurs only as a relation endpoint or an attribute's subject becomes an
      individual, even when a class of that name exists (OWL punning, as the store does);
      ``structural_predicates`` (e.g. the "related" neighbour link) do not spawn individuals
    * names are normalized to one spelling and duplicate records merged
    """
    stats = {"typed": 0, "subclassed": 0, "folded": 0, "to_data_value": 0, "dropped": 0,
             "classes_dropped": 0, "classes_declared": 0}

    # ── names to one spelling ──
    def nm(s) -> str:
        return normalize_label("" if s is None else str(s))

    for c in concepts.classes:
        c.name = nm(c.name)
        c.parent = nm(c.parent) or None
    for op in concepts.object_properties:
        op.name, op.domain, op.range = nm(op.name), nm(op.domain), nm(op.range)
    for dp in concepts.datatype_properties:
        dp.name, dp.domain = nm(dp.name), nm(dp.domain)
    concepts.class_hierarchy = [(nm(p), nm(c)) for p, c in concepts.class_hierarchy]
    for i in instances:
        i.name, i.class_name = nm(i.name), nm(i.class_name)
    for r in relations:
        r.subject, r.predicate, r.object = nm(r.subject), str(r.predicate or "").strip(), nm(r.object)
    for d in data_values:
        d.entity, d.property = nm(d.entity), str(d.property or "").strip()

    # ── classes: only class-shaped names survive ──
    kept: list[Class] = []
    declared: dict[str, Class] = {}
    for c in concepts.classes:
        if not c.name or not is_class_name(c.name):
            stats["classes_dropped"] += 1
            continue
        if c.name in declared:
            have = declared[c.name]
            have.parent = have.parent or c.parent
            have.description = have.description or c.description
            for cid in c.source_chunks:
                if cid not in have.source_chunks:
                    have.source_chunks.append(cid)
            continue
        declared[c.name] = c
        kept.append(c)
    concepts.classes = kept

    def class_endpoint(name: str) -> str | None:
        """The class for ``name``, declaring it when it is only referenced; None if not class-shaped."""
        if not name or not is_class_name(name):
            return None
        if name not in declared:
            c = Class(name=name)
            declared[name] = c
            concepts.classes.append(c)
            stats["classes_declared"] += 1
        return name

    # ── declarations ──
    dps = []
    seen_dp: set[tuple[str, str]] = set()
    for dp in concepts.datatype_properties:
        if not dp.name:
            continue
        if dp.domain and class_endpoint(dp.domain) is None:
            stats["dropped"] += 1
            continue
        if (dp.name, dp.domain) in seen_dp:
            continue
        seen_dp.add((dp.name, dp.domain))
        dps.append(dp)
    concepts.datatype_properties = dps
    ops = []
    seen_op: set[tuple[str, str, str]] = set()
    for op in concepts.object_properties:
        if not op.name:
            continue
        if op.range and _is_datatype(op.range):
            stats["dropped"] += 1        # a datatype range: not an object property
            continue
        dom = class_endpoint(op.domain) if op.domain else ""
        rng = class_endpoint(op.range) if op.range else ""
        if dom is None or rng is None:
            stats["dropped"] += 1
            continue
        if (op.name, op.domain, op.range) in seen_op:
            continue
        seen_op.add((op.name, op.domain, op.range))
        ops.append(op)
    concepts.object_properties = ops
    declared_dtp = {dp.name for dp in concepts.datatype_properties}
    declared_obp = {op.name for op in concepts.object_properties}

    # ── typing statements hiding in relations ──
    rename: dict[str, str] = {}
    plain: list[Relation] = []
    typings: list[Instance] = []
    for r in relations:
        if not (r.subject and r.predicate and r.object):
            stats["dropped"] += 1
            continue
        if is_value_like(r.subject):
            stats["dropped"] += 1
            continue
        reserved = RESERVED_PREDICATES.get(r.predicate)
        if reserved is None:
            plain.append(r)
            continue
        if reserved == "subClassOf":
            if class_endpoint(r.object) is None or class_endpoint(r.subject) is None or r.subject == r.object:
                stats["dropped"] += 1
                continue
            concepts.class_hierarchy.append((r.object, r.subject))
            stats["subclassed"] += 1
        elif reserved == "instanceOf":
            if class_endpoint(r.object) is None:
                stats["dropped"] += 1
                continue
            typings.append(Instance(name=r.subject, class_name=r.object, source_chunks=list(r.source_chunks)))
            stats["typed"] += 1
        else:  # sameAs: two spellings of one thing
            if r.subject != r.object:
                rename[r.subject] = r.object
                stats["folded"] += 1
    relations[:] = plain

    # ── relation vs attribute: the declaration decides, else the shape of the object ──
    kept_r: list[Relation] = []
    for r in relations:
        if r.predicate_type == "DatatypeProperty":
            data_values.append(DataValue(entity=r.subject, property=r.predicate, value=r.object,
                                         source_chunks=list(r.source_chunks)))
            stats["to_data_value"] += 1
            continue
        if r.predicate in declared_dtp and r.predicate not in declared_obp:
            is_attr = True
        elif r.predicate in declared_obp and r.predicate not in declared_dtp:
            is_attr = False
        else:
            is_attr = is_value_like(r.object)
        if is_attr:
            data_values.append(DataValue(entity=r.subject, property=r.predicate, value=r.object,
                                         source_chunks=list(r.source_chunks)))
            stats["to_data_value"] += 1
            continue
        kept_r.append(r)
    seen_r: set[tuple[str, str, str]] = set()
    relations[:] = [r for r in kept_r
                    if (r.subject, r.predicate, r.object) not in seen_r
                    and not seen_r.add((r.subject, r.predicate, r.object))]

    # ── instances: class-shaped types only; typings from relations; sameAs folds ──
    for i in instances:
        if i.class_name and class_endpoint(i.class_name) is None:
            i.class_name = ""
    instances.extend(typings)
    if rename:
        for i in instances:
            i.name = rename.get(i.name, i.name)
        for r in relations:
            r.subject, r.object = rename.get(r.subject, r.subject), rename.get(r.object, r.object)
        for d in data_values:
            d.entity = rename.get(d.entity, d.entity)
    merged: dict[tuple[str, str], Instance] = {}
    out_i: list[Instance] = []
    for i in instances:
        if not i.name:
            stats["dropped"] += 1
            continue
        key = (i.name, i.class_name)
        if key in merged:
            have = merged[key]
            for cid in i.source_chunks:
                if cid not in have.source_chunks:
                    have.source_chunks.append(cid)
            continue
        merged[key] = i
        out_i.append(i)
    # an untyped record is redundant next to a typed one of the same name
    typed_names = {n for n, c in merged if c}
    instances[:] = [i for i in out_i if i.class_name or i.name not in typed_names]

    # ── data values: a value is not a subject; one record per (entity, property, value) ──
    seen_dv: set[tuple[str, str, str]] = set()
    kept_dv: list[DataValue] = []
    for d in data_values:
        if not (d.entity and d.property) or d.value is None or is_value_like(d.entity):
            stats["dropped"] += 1
            continue
        key = (d.entity, d.property, str(d.value))
        if key in seen_dv:
            continue
        seen_dv.add(key)
        kept_dv.append(d)
    data_values[:] = kept_dv

    # ── names that exist only as a relation endpoint or an attribute's subject become instances ──
    known = {i.name for i in instances}
    structural = set(structural_predicates or ())
    chunks_of: dict[str, list[str]] = {}
    ends = [(r.subject, r.source_chunks) for r in relations if r.predicate not in structural]
    ends += [(r.object, r.source_chunks) for r in relations if r.predicate not in structural]
    ends += [(d.entity, d.source_chunks) for d in data_values]
    for end, cids in ends:
        if end and end not in known:
            chunks_of.setdefault(end, [])
            for cid in cids:
                if cid not in chunks_of[end]:
                    chunks_of[end].append(cid)
    for end, cids in chunks_of.items():
        instances.append(Instance(name=end, source_chunks=cids))
        known.add(end)

    # ── hierarchy: class-shaped, no self loops, no duplicates ──
    seen_h: set[tuple[str, str]] = set()
    hier = []
    for p, c in concepts.class_hierarchy:
        if not (is_class_name(p) and is_class_name(c)) or p == c:
            stats["dropped"] += 1
            continue
        class_endpoint(p)
        class_endpoint(c)
        if (p, c) in seen_h:
            continue
        seen_h.add((p, c))
        hier.append((p, c))
    concepts.class_hierarchy = hier
    return stats
