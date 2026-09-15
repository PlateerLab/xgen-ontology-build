"""Deduplication — fold synonymous classes, properties and instances into one.

Four complementary passes, each optional and degrading gracefully:

1. **instance normalization** (rule) — same content morphemes -> one name. Uses a
   :class:`~xgen_ontology.protocols.Morphology` analyzer if given, else a cleaned
   lowercase key (English-neutral).
2. **object-property normalization** (rule) — group by (domain, range, form-key),
   then by (domain, range), keeping the shortest canonical name.
3. **LLM synonymy** — classes, then object-properties: ask an LLM for merge groups.
4. **vector dedup** (embedding) — cosine-cluster names whose *meaning* matches even
   though the surface form differs.
"""
from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache

from ..korean import tokenize
from ..llm import invoke_json
from ..models import Concepts, DataValue, Instance, Relation

_SEP = re.compile(r"[\s_\-·•/\\()（）「」『』【】\[\]]+")
# Content-morpheme tags used for the normalization key when the bundled Korean analyzer
# is available: nouns, foreign script, Hanja, symbols, digits, and XPN (negation/degree
# prefixes -- dropping those flips the meaning).
_KEY_TAGS = ("SL", "SH", "SW", "XPN", "SN")
_QUOTE_CATEGORIES = ("Pi", "Pf")
_QUOTE_NAME_PARTS = ("QUOTATION MARK", "APOSTROPHE", "PRIME", "GRAVE ACCENT")
_EN_NOISE = re.compile(
    r"(to|from|of|by|with|for|the|and|or|is|are|has|have|belongs|connection|link|relation|mapping|reference)"
)


class Deduplicator:
    """Synonym/duplicate folding. All backends optional."""

    def __init__(self, llm=None, morphology=None, embedder=None, *, vector_threshold: float = 0.7):
        self.llm = llm
        self.morph = morphology
        self.embedder = embedder
        self.vector_threshold = vector_threshold
        self.llm_calls = 0

    def deduplicate(
        self,
        concepts: Concepts,
        instances: list[Instance],
        relations: list[Relation],
        data_values: list[DataValue],
    ) -> int:
        """Run all passes in place. Returns the number of names merged."""
        merged = 0

        rename = self._normalize_instances(instances)
        if rename:
            self._apply_instance(rename, instances, relations, data_values)
            merged += len(rename)

        if len(concepts.classes) >= 3:
            rename = self._llm_synonyms(
                [f"- {c.name}: {c.description}" for c in concepts.classes if c.name],
                system="You identify synonymous classes that denote the same concept.",
                rules=("Merge only different names for the *same* concept.\n"
                       "Treat a term and its translation as the same (keep the original-language name).\n"
                       "Never merge a parent with its subclass."),
                label="class",
            )
            if rename:
                self._apply_class(rename, concepts, instances)
                merged += len(rename)

        if len(concepts.object_properties) >= 2:
            rename = self._normalize_object_properties(concepts.object_properties)
            if rename:
                self._apply_property(rename, concepts, relations, data_values)
                merged += len(rename)

        if len(concepts.object_properties) >= 3:
            rename = self._llm_synonyms(
                [f"- {p.name} (domain: {p.domain or '?'}, range: {p.range or '?'})"
                 for p in concepts.object_properties if p.name],
                system="You identify synonymous relations (object properties).",
                rules=("Merge only different names for the *same* relation.\n"
                       "Treat a term and its translation as the same (keep the original-language name).\n"
                       "Merge more readily when domain and range match."),
                label="relation",
            )
            if rename:
                self._apply_property(rename, concepts, relations, data_values)
                merged += len(rename)

        rename = self._vector_dedup([c.name for c in concepts.classes if c.name])
        if rename:
            self._apply_class(rename, concepts, instances)
            merged += len(rename)

        return merged

    def compute_rename_map(self, concepts: Concepts) -> dict[str, str]:
        """Schema synonym map only (classes + object properties), nothing applied.

        The same LLM and rule passes as :meth:`deduplicate`, flattened
        (``a->b, b->c`` becomes ``a->c``) with self-maps removed. Used by the
        enrich build, which folds synonyms after the base build rather than during it.
        """
        rename: dict[str, str] = {}
        classes = [c for c in concepts.classes if c.name]
        if len(classes) >= 3:
            rename.update(self._llm_synonyms(
                [f"- {c.name}: {c.description}" for c in classes],
                system="You identify synonymous classes that denote the same concept.",
                rules=("Merge only different names for the *same* concept.\n"
                       "Treat a term and its translation as the same (keep the original-language name).\n"
                       "Never merge a parent with its subclass."),
                label="class"))
        props = concepts.object_properties
        if len(props) >= 2:
            rename.update(self._normalize_object_properties(props))
        named = [p for p in props if p.name]
        if len(named) >= 3:
            rename.update(self._llm_synonyms(
                [f"- {p.name} (domain: {p.domain or '?'}, range: {p.range or '?'})" for p in named],
                system="You identify synonymous relations (object properties).",
                rules=("Merge only different names for the *same* relation.\n"
                       "Treat a term and its translation as the same (keep the original-language name).\n"
                       "Merge more readily when domain and range match."),
                label="relation"))
        flat: dict[str, str] = {}
        for k, v in rename.items():
            if not k or not v:
                continue
            seen = {k}
            while v in rename and rename[v] != v and rename[v] not in seen:
                seen.add(v)
                v = rename[v]
            if k != v:
                flat[k] = v
        return flat

    # ── rule passes ──

    def _norm_key(self, name: str) -> str:
        """Content-morpheme key: particles/endings/spacing variants fold together.

        Leading quote characters are not part of a name. With an injected
        :class:`~xgen_ontology.protocols.Morphology` its nouns form the key; else
        the bundled Korean analyzer (when installed); else a cleaned lowercase form.
        """
        name = (name or "").strip()
        while name and _is_quote_char(name[0]):
            name = name[1:].lstrip()
        cleaned = _SEP.sub("", name)
        if not cleaned:
            return ""
        if self.morph is not None:
            try:
                nouns = self.morph.nouns(cleaned)
                if nouns:
                    return "".join(nouns).lower()
            except Exception:
                pass
        else:
            toks = tokenize(cleaned)
            if toks:
                nouns = [t.form for t in toks if t.tag.startswith("N") or t.tag in _KEY_TAGS]
                if nouns:
                    return "".join(nouns).lower()
        return cleaned.lower()

    def _norm_prop_key(self, name: str) -> str:
        key = self._norm_key(name)
        return _EN_NOISE.sub("", key)

    def _normalize_instances(self, instances: list[Instance]) -> dict[str, str]:
        """Same key -> one name; the canonical is the shortest spelling (closest to the base form)."""
        groups: dict[str, list[str]] = {}
        for inst in instances:
            name = inst.name
            if not name:
                continue
            key = self._norm_key(name)
            if key and name not in groups.setdefault(key, []):
                groups[key].append(name)
        rename: dict[str, str] = {}
        for names in groups.values():
            if len(names) < 2:
                continue
            canon = min(names, key=lambda s: (len(s), s))
            for n in names:
                if n != canon:
                    rename[n] = canon
        return rename

    def _normalize_object_properties(self, obj_props) -> dict[str, str]:
        rename: dict[str, str] = {}
        groups: dict[tuple, list[str]] = {}
        for p in obj_props:
            if not p.name:
                continue
            groups.setdefault((p.domain, p.range, self._norm_prop_key(p.name)), []).append(p.name)
        for names in groups.values():
            if len(names) <= 1:
                continue
            canon = _canonical(names)
            for n in names:
                if n != canon:
                    rename[n] = canon
        dr: dict[tuple, set] = {}
        for p in obj_props:
            name = rename.get(p.name, p.name)
            if p.domain and p.range:
                dr.setdefault((p.domain, p.range), set()).add(name)
        for names_set in dr.values():
            names = list(names_set)
            if len(names) <= 1:
                continue
            canon = _canonical(names)
            for n in names:
                if n != canon:
                    rename[n] = canon
        return rename

    # ── LLM pass ──

    def _llm_synonyms(self, items: list[str], system: str, rules: str, label: str) -> dict[str, str]:
        if self.llm is None or len(items) < 3:
            return {}
        user = (
            f"Find synonym groups (same concept, different name) among the {label} list below.\n\n"
            f"## {label} list\n" + "\n".join(items) + "\n\n"
            f"## Rules\n{rules}\n- Return an empty array if there are no duplicates.\n"
            "- Pick the most general, intuitive canonical name.\n\n"
            '## Output (JSON only)\n{"merge_groups": [{"canonical": "...", "synonyms": ["...", "..."]}]}'
        )
        self.llm_calls += 1
        result = invoke_json(self.llm, system, user)
        rename: dict[str, str] = {}
        for group in result.get("merge_groups", []) or []:
            canon = group.get("canonical", "")
            for syn in group.get("synonyms", []) or []:
                if syn and syn != canon:
                    rename[syn] = canon
        return rename

    # ── vector pass ──

    def _vector_dedup(self, names: list[str], max_names: int = 600) -> dict[str, str]:
        uniq = sorted({(n or "").strip() for n in names if (n or "").strip()})
        if self.embedder is None or len(uniq) < 2 or len(uniq) > max_names:
            return {}
        try:
            vectors = self.embedder.embed(uniq)
        except Exception:
            return {}
        if not vectors or len(vectors) != len(uniq):
            return {}
        return cluster_by_cosine(uniq, vectors, self.vector_threshold)

    # ── apply ──

    @staticmethod
    def _apply_instance(rename, instances, relations, data_values) -> None:
        for inst in instances:
            inst.name = rename.get(inst.name, inst.name)
        for rel in relations:
            rel.subject = rename.get(rel.subject, rel.subject)
            rel.object = rename.get(rel.object, rel.object)
        for dv in data_values:
            dv.entity = rename.get(dv.entity, dv.entity)

    @staticmethod
    def _apply_class(rename, concepts: Concepts, instances) -> None:
        seen: set[str] = set()
        new_classes = []
        for c in concepts.classes:
            c.name = rename.get(c.name, c.name)
            if c.parent:
                c.parent = rename.get(c.parent, c.parent)
            if c.name not in seen:
                new_classes.append(c)
                seen.add(c.name)
        concepts.classes = new_classes
        seen_h: set = set()
        new_h = []
        for parent, child in concepts.class_hierarchy:
            p = rename.get(parent, parent)
            ch = rename.get(child, child)
            if p != ch and (p, ch) not in seen_h:
                new_h.append((p, ch))
                seen_h.add((p, ch))
        concepts.class_hierarchy = new_h
        for op in concepts.object_properties:
            op.domain = rename.get(op.domain, op.domain)
            op.range = rename.get(op.range, op.range)
        for dp in concepts.datatype_properties:
            dp.domain = rename.get(dp.domain, dp.domain)
        for inst in instances:
            inst.class_name = rename.get(inst.class_name, inst.class_name)

    @staticmethod
    def _apply_property(rename, concepts: Concepts, relations, data_values) -> None:
        seen: set[str] = set()
        deduped = []
        for op in concepts.object_properties:
            op.name = rename.get(op.name, op.name)
            if op.name not in seen:
                deduped.append(op)
                seen.add(op.name)
        concepts.object_properties = deduped
        for rel in relations:
            rel.predicate = rename.get(rel.predicate, rel.predicate)
        for dv in data_values:
            dv.property = rename.get(dv.property, dv.property)


def _canonical(names: list[str]) -> str:
    """Prefer a Korean (Hangul) name, else shortest; ties broken lexicographically."""
    ko = [n for n in names if any(0xAC00 <= ord(c) <= 0xD7A3 for c in n)]
    pool = ko or names
    return min(pool, key=lambda s: (len(s), s))


def cluster_by_cosine(names: list[str], vectors, threshold: float = 0.7) -> dict[str, str]:
    """Cosine >= threshold -> same cluster (union-find); canonical = shortest name.

    Pure function, unit-testable without an embedder."""
    n = len(names)
    if n != len(vectors) or n < 2:
        return {}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    rename: dict[str, str] = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        canon = min((names[k] for k in members), key=lambda s: (len(s), s))
        for k in members:
            if names[k] != canon:
                rename[names[k]] = canon
    return rename


def _cosine(a, b) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@lru_cache(maxsize=512)
def _is_quote_char(ch: str) -> bool:
    if unicodedata.category(ch) in _QUOTE_CATEGORIES:
        return True
    name = unicodedata.name(ch, "")
    return any(part in name for part in _QUOTE_NAME_PARTS)


_NAME_TAGS = ("NNG", "NNP", "NNB", "XSN", "SL", "SH", "SN", "XPN", "XR")


def shorten_entity_name(name: str) -> str:
    """Keep only the name in a sentence fragment an extractor returned as an entity.

    Takes the leading run of noun-family morphemes and stops at the first particle,
    ending or verb ("Korea Racing Authority-NOM" -> "Korea Racing Authority"). A
    name that is already all nouns is untouched; when nothing can be kept, or no
    analyzer is installed, the original is returned. Losing a name loses knowledge,
    so the safe side is to keep it.
    """
    t = (name or "").strip()
    if not t:
        return t
    toks = tokenize(t)
    if not toks:
        return t
    if all(x.tag in _NAME_TAGS for x in toks):
        return t
    kept = []
    for x in toks:
        if x.tag in _NAME_TAGS:
            kept.append(x)
            continue
        break
    if not kept:
        return t
    head = t[kept[0].start:kept[-1].start + kept[-1].len].strip()
    return head or t
