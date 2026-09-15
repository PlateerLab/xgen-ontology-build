"""Term dictionary: alias -> canonical term, applied at query time, at indexing time and to a build.

The production system keeps a per-collection dictionary (an acronym and its full
form, a house spelling and the official one) and applies it in three places, all
rule-based and all reproduced here without the database:

* :meth:`TermDictionary.normalize_query` -- **expand** (default) appends the
  canonical term when an alias occurs in the query, so a passage using either
  spelling matches; **replace** substitutes it. Word-boundary matching that
  understands Hangul, so "DSR" never fires inside "DSRA".
* :meth:`TermDictionary.normalize_text` -- before chunking/embedding, writes
  ``alias(canonical)`` so an embedding carries both spellings. Idempotent.
* :meth:`TermDictionary.apply_to_build` -- renames aliases to their canonical
  term across an ontology's classes, instances, relations and data values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Concepts, DataValue, Instance, Relation

_BOUNDARY_L = r"(?<![\w가-힣])"
_BOUNDARY_R = r"(?![\w가-힣])"


@dataclass
class Term:
    alias: str
    canonical: str
    type: str = ""
    full_form: str = ""
    category: str = ""
    definition: str = ""
    notes: str = ""
    confidence: float = 1.0
    source: str = "manual"
    active: bool = True
    canonical_element: str = ""   # the ontology element the canonical names, if linked

    @property
    def key(self) -> str:
        return TermDictionary.normalize_alias(self.alias)


@dataclass
class TermDictionary:
    """An alias table with the production system's matching rules. Keyed by the normalized alias."""

    name: str = ""
    terms: dict[str, Term] = field(default_factory=dict)

    @staticmethod
    def normalize_alias(alias: str) -> str:
        return (alias or "").strip().lower()

    # ── maintenance ──

    def add(self, alias: str, canonical: str, **meta) -> Term | None:
        """Upsert one alias (case-insensitive). Returns the term, or None when alias/canonical is empty."""
        alias, canonical = (alias or "").strip(), (canonical or "").strip()
        if not alias or not canonical:
            return None
        term = Term(alias=alias, canonical=canonical, **{k: v for k, v in meta.items() if v is not None})
        self.terms[term.key] = term
        return term

    def lookup(self, alias: str) -> Term | None:
        return self.terms.get(self.normalize_alias(alias))

    def remove(self, alias: str) -> bool:
        return self.terms.pop(self.normalize_alias(alias), None) is not None

    def set_active(self, alias: str, active: bool) -> bool:
        t = self.lookup(alias)
        if t is None:
            return False
        t.active = active
        return True

    def link(self, alias: str, element: str) -> bool:
        """Record which ontology element the alias's canonical term names."""
        t = self.lookup(alias)
        if t is None:
            return False
        t.canonical_element = element
        return True

    def bulk_import(self, rows: list[dict]) -> dict[str, int]:
        """Import spreadsheet rows (``alias`` and ``canonical`` required; the rest optional)."""
        stats = {"inserted": 0, "updated": 0, "skipped": 0}
        for row in rows or []:
            alias, canonical = (row.get("alias") or "").strip(), (row.get("canonical") or "").strip()
            if not alias or not canonical:
                stats["skipped"] += 1
                continue
            existed = self.normalize_alias(alias) in self.terms
            self.add(alias, canonical, type=row.get("type") or "", full_form=row.get("full_form") or "",
                     category=row.get("category") or "", definition=row.get("definition") or "",
                     notes=row.get("notes") or "", confidence=float(row.get("confidence", 1.0) or 1.0),
                     source=row.get("source") or "import")
            stats["updated" if existed else "inserted"] += 1
        return stats

    def active_terms(self, *, category: str = "", type: str = "") -> list[Term]:
        return [t for t in self.terms.values() if t.active
                and (not category or t.category == category) and (not type or t.type == type)]

    # ── application ──

    def normalize_query(self, query: str, *, mode: str = "expand") -> tuple[str, list[dict]]:
        """``(normalized query, applied)``; ``applied`` lists ``{alias, canonical, mode}`` per hit."""
        if not query or not self.terms:
            return query, []
        query_lower = query.lower()
        appended: list[str] = []
        applied: list[dict] = []
        normalized = query
        for t in self.active_terms():
            if not re.search(_BOUNDARY_L + re.escape(t.key) + _BOUNDARY_R, query_lower):
                continue
            applied.append({"alias": t.alias, "canonical": t.canonical, "mode": mode})
            if mode == "replace":
                normalized = re.sub(_BOUNDARY_L + re.escape(t.alias) + _BOUNDARY_R, t.canonical,
                                    normalized, flags=re.IGNORECASE)
            else:
                appended.append(t.canonical)
        if mode != "replace" and appended:
            extra = [c for c in dict.fromkeys(appended) if c.lower() not in query_lower]
            if extra:
                normalized = f"{query} {' '.join(extra)}"
        return normalized, applied

    def normalize_text(self, text: str) -> str:
        """Write ``alias(canonical)`` at each alias occurrence, once (idempotent); for indexing."""
        if not text or not self.terms:
            return text
        out = text
        # longest alias first so a short alias never breaks a longer one it sits inside
        for t in sorted(self.active_terms(), key=lambda t: len(t.alias), reverse=True):
            pattern = (_BOUNDARY_L + re.escape(t.alias) + _BOUNDARY_R
                       + r"(?!\(" + re.escape(t.canonical) + r"\))")
            out = re.sub(pattern, f"{t.alias}({t.canonical})", out, flags=re.IGNORECASE)
        return out

    def apply_to_build(self, concepts: Concepts, instances: list[Instance], relations: list[Relation],
                       data_values: list[DataValue]) -> int:
        """Rename aliases to canonical terms across the build models in place. Returns renames applied."""
        if not self.terms:
            return 0
        rename = {t.alias: t.canonical for t in self.active_terms() if t.alias != t.canonical}
        by_key = {self.normalize_alias(a): c for a, c in rename.items()}

        def canon(name: str) -> str:
            return by_key.get(self.normalize_alias(name), name) if name else name

        n = 0
        for c in concepts.classes:
            new = canon(c.name)
            if new != c.name:
                c.name, n = new, n + 1
            if c.parent:
                c.parent = canon(c.parent)
        for p in [*concepts.object_properties, *concepts.datatype_properties]:
            p.domain = canon(p.domain)
            if hasattr(p, "range") and p.range and not str(p.range).startswith("xsd:"):
                p.range = canon(p.range)
        concepts.class_hierarchy = [(canon(a), canon(b)) for a, b in concepts.class_hierarchy]
        for i in instances:
            new = canon(i.name)
            if new != i.name:
                i.name, n = new, n + 1
            i.class_name = canon(i.class_name)
        for r in relations:
            for attr in ("subject", "object"):
                new = canon(getattr(r, attr))
                if new != getattr(r, attr):
                    setattr(r, attr, new)
                    n += 1
        for d in data_values:
            new = canon(d.entity)
            if new != d.entity:
                d.entity, n = new, n + 1
        return n
