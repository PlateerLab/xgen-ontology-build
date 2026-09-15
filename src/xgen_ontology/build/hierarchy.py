"""Class hierarchy cleaning, typing repairs and property inheritance.

* ``clean_hierarchy`` — keep only genuine is-a edges: a parent must be a *class*,
  not a property/relation name an extractor mislabeled as a parent ("being linked
  is not being a subclass"). Also drops self-loops and cycles.
* ``fix_self_typed_instances`` — an individual typed by a class of its own name is
  not typed at all; move it to that class's parent when there is one.
* ``materialize_property_inheritance`` — copy a parent's declared properties onto
  its subclasses, so a query needs no reasoner to see them.
"""
from __future__ import annotations

from collections import defaultdict

from ..models import Concepts, DataProperty, Instance, ObjectProperty


def clean_hierarchy(concepts: Concepts) -> int:
    """Drop hierarchy edges whose parent isn't a real class (in place). Returns dropped count."""
    class_names = {c.name for c in concepts.classes if c.name}

    kept: list[tuple[str, str]] = []
    seen: set = set()
    dropped = 0
    # collect edges from both class_hierarchy and class.parent
    edges = list(concepts.class_hierarchy)
    for c in concepts.classes:
        if c.parent:
            edges.append((c.parent, c.name))
    for parent, child in edges:
        if not parent or not child or parent == child:
            dropped += 1
            continue
        if parent not in class_names:
            # a property masquerading as a parent, or an undefined node -> not is-a
            dropped += 1
            continue
        if (parent, child) in seen:
            continue
        seen.add((parent, child))
        kept.append((parent, child))

    # break cycles deterministically (drop the edge that closes a cycle)
    parent_of: dict[str, str] = {}
    final: list[tuple[str, str]] = []
    for parent, child in kept:
        if _creates_cycle(parent_of, child, parent):
            dropped += 1
            continue
        parent_of[child] = parent
        final.append((parent, child))

    concepts.class_hierarchy = final
    # normalize class.parent to match the cleaned hierarchy
    for c in concepts.classes:
        c.parent = parent_of.get(c.name)
    return dropped


def fix_self_typed_instances(instances: list[Instance], concepts: Concepts) -> dict:
    """Repair ``instanceOf`` edges that point at a class spelled like the instance itself.

    "Acme is an Acme" carries no type. When that class has a parent, the
    instance is retyped by the parent (the only type the graph knows for it);
    otherwise the typing is dropped. A duplicate typing record that would result
    is removed. Returns ``{"moved", "dropped"}``.
    """
    parent_of: dict[str, str] = {child: parent for parent, child in concepts.class_hierarchy}
    for c in concepts.classes:
        if c.parent and c.name not in parent_of:
            parent_of[c.name] = c.parent
    typed: dict[str, set] = defaultdict(set)
    for i in instances:
        if i.name and i.class_name and i.class_name != i.name:
            typed[i.name].add(i.class_name)
    moved = dropped = 0
    kept: list[Instance] = []
    for i in instances:
        if not (i.name and i.class_name == i.name):
            kept.append(i)
            continue
        parent = parent_of.get(i.name)
        if parent and parent != i.name:
            if parent in typed[i.name]:
                dropped += 1          # the parent typing already exists on another record
                continue
            i.class_name = parent
            typed[i.name].add(parent)
            moved += 1
            kept.append(i)
            continue
        if typed[i.name]:
            dropped += 1
            continue
        i.class_name = ""
        dropped += 1
        kept.append(i)
    instances[:] = kept
    return {"moved": moved, "dropped": dropped}


def materialize_property_inheritance(concepts: Concepts) -> int:
    """Declare each parent's object/datatype properties on its subclasses too, in place.

    Follows the whole is-a chain (grandparents included). A property is added to a
    subclass only when that subclass has no declaration of the same name yet.
    Returns the number of declarations added.
    """
    class_names = {c.name for c in concepts.classes if c.name}
    parent_of: dict[str, str] = {}
    for parent, child in concepts.class_hierarchy:
        if parent in class_names and child in class_names and parent != child:
            parent_of.setdefault(child, parent)

    def ancestors(name: str) -> list[str]:
        out, cur, seen = [], parent_of.get(name), {name}
        while cur and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = parent_of.get(cur)
        return out

    op_by_domain: dict[str, list[ObjectProperty]] = defaultdict(list)
    for op in concepts.object_properties:
        if op.name and op.domain:
            op_by_domain[op.domain].append(op)
    dp_by_domain: dict[str, list[DataProperty]] = defaultdict(list)
    for dp in concepts.datatype_properties:
        if dp.name and dp.domain:
            dp_by_domain[dp.domain].append(dp)

    added = 0
    for child in list(parent_of):
        have_op = {op.name for op in op_by_domain.get(child, [])}
        have_dp = {dp.name for dp in dp_by_domain.get(child, [])}
        for anc in ancestors(child):
            for op in op_by_domain.get(anc, []):
                if op.name not in have_op:
                    new = ObjectProperty(name=op.name, domain=child, range=op.range)
                    concepts.object_properties.append(new)
                    have_op.add(op.name)
                    added += 1
            for dp in dp_by_domain.get(anc, []):
                if dp.name not in have_dp:
                    new = DataProperty(name=dp.name, domain=child, range=dp.range,
                                       display_name=dp.display_name)
                    concepts.datatype_properties.append(new)
                    have_dp.add(dp.name)
                    added += 1
    return added


def _creates_cycle(parent_of: dict[str, str], child: str, parent: str) -> bool:
    cur = parent
    seen = {child}
    while cur is not None:
        if cur in seen:
            return True
        seen.add(cur)
        cur = parent_of.get(cur)
    return False
