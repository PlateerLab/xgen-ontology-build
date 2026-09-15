"""Class hierarchy cleaning, typing repairs, inheritance + Semantic Context Synthesis (SCS).

* ``clean_hierarchy`` — keep only genuine is-a edges: a parent must be a *class*,
  not a property/relation name an extractor mislabeled as a parent ("being linked
  is not being a subclass"). Also drops self-loops and cycles.
* ``fix_self_typed_instances`` — an individual typed by a class of its own name is
  not typed at all; move it to that class's parent when there is one.
* ``materialize_property_inheritance`` — copy a parent's declared properties onto
  its subclasses, so a query needs no reasoner to see them.
* ``SCSGenerator`` — per class, aggregate **direct** properties (horizontal) and
  **inherited** properties down the is-a chain (vertical), and synthesize a short
  natural-language context summary (LLM if available, rule-based fallback).
"""
from __future__ import annotations

from collections import defaultdict

from ..llm import invoke_json
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


class SCSGenerator:
    """Generate SCS context profiles (depth, direct + inherited properties, summary)."""

    def __init__(self, llm=None):
        self.llm = llm
        self.llm_calls = 0

    def generate_profiles(self, concepts: Concepts) -> list[dict]:
        classes = concepts.classes
        parent_map = self._parent_map(concepts)
        depth_map = self._depths(classes, parent_map)
        depth_groups: dict[int, list[str]] = defaultdict(list)
        for name, d in depth_map.items():
            depth_groups[d].append(name)

        domain_props = self._property_map(concepts)
        related = self._related_map(concepts)
        cache: dict[str, dict] = {}

        max_depth = max(depth_groups) if depth_groups else 0
        for depth in range(max_depth + 1):
            batch = []
            for name in depth_groups.get(depth, []):
                batch.append({
                    "class_name": name,
                    "depth": depth,
                    "direct_properties": domain_props.get(name, []),
                    "inherited_properties": self._inherited(name, parent_map, cache),
                    "related_classes": related.get(name, []),
                    "context_summary": "",
                })
            for profile, summary in zip(batch, self._summarize(batch)):
                profile["context_summary"] = summary
                cache[profile["class_name"]] = profile
        return list(cache.values())

    def _parent_map(self, concepts: Concepts) -> dict[str, str | None]:
        pm: dict[str, str | None] = {}
        for parent, child in concepts.class_hierarchy:
            pm[child] = parent
        for c in concepts.classes:
            if c.name and c.parent and c.name not in pm:
                pm[c.name] = c.parent
        for c in concepts.classes:
            pm.setdefault(c.name, None)
        return pm

    def _depths(self, classes, parent_map) -> dict[str, int]:
        depth: dict[str, int] = {}

        def get(name, visited=None):
            if name in depth:
                return depth[name]
            visited = visited or set()
            if name in visited:
                return 0
            visited.add(name)
            parent = parent_map.get(name)
            depth[name] = 0 if not parent else get(parent, visited) + 1
            return depth[name]

        for c in classes:
            if c.name:
                get(c.name)
        return depth

    def _property_map(self, concepts: Concepts) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for p in concepts.object_properties:
            if p.domain:
                out[p.domain].append({"name": p.name, "type": "ObjectProperty", "range": p.range})
        for p in concepts.datatype_properties:
            if p.domain:
                out[p.domain].append({"name": p.name, "type": "DatatypeProperty", "range": p.range})
        return out

    def _related_map(self, concepts: Concepts) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for p in concepts.object_properties:
            if p.domain and p.range:
                if p.range not in out[p.domain]:
                    out[p.domain].append(p.range)
                if p.domain not in out[p.range]:
                    out[p.range].append(p.domain)
        return out

    def _inherited(self, name, parent_map, cache) -> list[dict]:
        inherited: list[dict] = []
        visited: set = set()
        current = parent_map.get(name)
        depth_count = 1
        while current and current not in visited:
            visited.add(current)
            pp = cache.get(current)
            if pp:
                for prop in pp.get("direct_properties", []):
                    inherited.append({"name": prop["name"], "from": current, "depth": depth_count})
                for prop in pp.get("inherited_properties", []):
                    inherited.append({"name": prop["name"], "from": prop["from"],
                                      "depth": prop["depth"] + depth_count})
            current = parent_map.get(current)
            depth_count += 1
        return inherited

    def _summarize(self, profiles: list[dict]) -> list[str]:
        if not profiles:
            return []
        if self.llm is not None:
            desc = []
            for p in profiles:
                direct = ", ".join(d["name"] for d in p["direct_properties"]) or "none"
                inh = ", ".join(f"{i['name']}(from {i['from']})" for i in p["inherited_properties"]) or "none"
                rel = ", ".join(p["related_classes"]) or "none"
                desc.append(f"[{p['class_name']}] direct: {direct} / inherited: {inh} / related: {rel}")
            user = (
                "Summarize each ontology class below in 1-2 sentences "
                "(direct properties, inherited 'from X', related classes).\n\n"
                "## Class profiles\n" + "\n".join(desc) + "\n\n"
                '## Output (JSON only, same order)\n{"summaries": ["...", "..."]}'
            )
            self.llm_calls += 1
            result = invoke_json(self.llm, "You summarize ontology class context.", user)
            summaries = result.get("summaries")
            if isinstance(summaries, list) and len(summaries) == len(profiles):
                return [str(s) for s in summaries]
        return [self._rule_summary(p) for p in profiles]

    @staticmethod
    def _rule_summary(p: dict) -> str:
        name = p["class_name"]
        parts = []
        if p["direct_properties"]:
            parts.append(f"{name} has properties: " + ", ".join(d["name"] for d in p["direct_properties"]))
        if p["inherited_properties"]:
            src: dict[str, list[str]] = {}
            for i in p["inherited_properties"]:
                src.setdefault(i["from"], []).append(i["name"])
            for s, props in src.items():
                parts.append(f"inherits {', '.join(props)} from {s}")
        if p["related_classes"]:
            parts.append("related to " + ", ".join(p["related_classes"]))
        return (". ".join(parts) + ".") if parts else f"{name} class."
