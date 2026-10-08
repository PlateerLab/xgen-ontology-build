"""Entity resolution — normalize individual names and merge near-duplicates.

Runs after extraction, before downstream cleaning. Folds together case/whitespace/
unicode variants and similar surface forms (containment + ``SequenceMatcher``),
while *guarding* date / id / numeric values and number-conflicting names
("800W" vs "600W") from being merged. Returns a canonical map you apply to
instances, relations and data values.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from xgen_ontology_build.models import DataValue, Instance, Relation
from xgen_ontology_build.text.tokens import normalize_name

SIMILARITY_THRESHOLD = 0.85

_SKIP = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{4}/\d{2}/\d{2}"),
    re.compile(r"^[A-Z]\d{4,}$"),
    re.compile(r"^\d+(\.\d+)?$"),
]


def resolve_entities(
    instances: list[Instance],
    relations: list[Relation],
    data_values: list[DataValue],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Compute a ``{original_name -> canonical_name}`` map and apply it in place.

    Returns the (transitively resolved) canonical map for logging."""
    by_class: dict[str, list[str]] = {}
    for inst in instances:
        by_class.setdefault(inst.class_name, []).append(inst.name)

    canonical: dict[str, str] = {}
    for _cls, names in by_class.items():
        normalized: list[str] = []
        for n in names:
            norm = normalize_name(n)
            if not norm:
                continue
            if norm != n:
                canonical[n] = norm
            normalized.append(norm)

        # case-fold (keep the longer surface form)
        seen_lower: dict[str, str] = {}
        deduped: list[str] = []
        for name in normalized:
            low = name.lower()
            if low in seen_lower:
                existing = seen_lower[low]
                if len(name) > len(existing):
                    canonical[existing] = name
                    seen_lower[low] = name
                    deduped = [name if x == existing else x for x in deduped]
                else:
                    canonical[name] = existing
            else:
                seen_lower[low] = name
                deduped.append(name)

        for group in _group_similar(deduped, threshold):
            canon = max(group, key=len)
            for m in group:
                if m != canon:
                    canonical[m] = canon

    canonical = _resolve_transitive(canonical)
    _apply(canonical, instances, relations, data_values)
    return canonical


def _apply(cmap: dict[str, str], instances, relations, data_values) -> None:
    if not cmap:
        return
    for inst in instances:
        inst.name = cmap.get(inst.name, inst.name)
    for rel in relations:
        rel.subject = cmap.get(rel.subject, rel.subject)
        rel.object = cmap.get(rel.object, rel.object)
    for dv in data_values:
        dv.entity = cmap.get(dv.entity, dv.entity)


def _is_skip(name: str) -> bool:
    return any(p.match(name.strip()) for p in _SKIP)


def _numbers(text: str) -> set[str]:
    units = re.findall(r"(\d+)\s*(?:W|w|MHz|GHz|GB|TB|MB|V|A|mm|mg|kg|g|ml|L)\b", text)
    kr = re.findall(r"(\d+)\s*(?:원|개|건|명|회|점|위|배)", text)
    big = re.findall(r"(?:^|(?<=\D))(\d{3,})(?=\D|$)", text)
    return set(units) | set(kr) | set(big)


def _conflicting_numbers(a: str, b: str) -> bool:
    na, nb = _numbers(a), _numbers(b)
    if not na or not nb:
        return False
    return len(na & nb) == 0


def _group_similar(names: list[str], threshold: float) -> list[list[str]]:
    if not names:
        return []
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _is_skip(names[i]) or _is_skip(names[j]) or _conflicting_numbers(names[i], names[j]):
                continue
            a, b = names[i].lower(), names[j].lower()
            if a in b or b in a:
                lo, hi = min(len(a), len(b)), max(len(a), len(b))
                if lo >= 3 and lo / hi >= 0.4:
                    union(i, j)
                    continue
            if SequenceMatcher(None, a, b).ratio() >= threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return list(groups.values())


def _resolve_transitive(cmap: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in cmap.items():
        final = v
        visited = {k}
        while final in cmap and final not in visited:
            visited.add(final)
            final = cmap[final]
        if k != final:
            out[k] = final
    return out
