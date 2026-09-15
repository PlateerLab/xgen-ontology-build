"""Predicate governance: keep the relation vocabulary from sprawling, without a word list.

Free-form extraction sprays near-synonym predicates ("belongs", "is part of",
"part-of") and sometimes bakes an argument into the name ("supervises-system"
for *supervises*). Four rule passes, all data-driven:

* :func:`govern_predicates` folds surface variants (separators, repeated Korean
  relational endings) to one canonical name, strips a subject/object noun glued
  into the predicate, and anchors to the schema's declared vocabulary and to any
  ``seed_predicates`` already in use.
* :func:`vote_relation_direction` flips relations the extractor wrote backwards,
  by majority vote of the (subject type, object type) pairs each predicate joins.
* :func:`merge_predicates` merges predicates that share a stem, and predicates
  whose (subject, object) extension is contained in another's (same relation,
  two names).

Meaning-level synonymy that survives surface normalization is handled separately
by the embedding dedup in :mod:`.dedup`.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from ..models import ObjectProperty, Relation

# Korean relational endings: removed only to merge surface variants of one form.
_DEFAULT_SUFFIX = re.compile(
    r"(되어있는것|되어있음|되어있다|되어있는|되어진|되어야|되어|되는|된다|되며|된|됨|"
    r"하는것|하는|하다|한다|하며|하고|"
    r"이다|이며|있는|있다|있음)$"
)
_SEP = re.compile(r"[\s_\-/]+")
_MIN_PRED_TAIL = 2      # what remains after removing an argument noun must be this long to mean anything


def normalize_predicate(pred: str, *, suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX) -> str:
    """Lowercase, strip separators, peel repeated relational endings -> a form key."""
    if not pred:
        return ""
    p = _SEP.sub("", pred.strip().lower())
    if suffix_pattern is not None:
        for _ in range(3):
            new = suffix_pattern.sub("", p)
            if new == p or not new:
                break
            p = new
    return p


def strip_argument_noun(pred: str, subject: str, obj: str) -> str:
    """Remove a subject/object noun the extractor glued into the predicate.

    "system-operation-supervision" between *operations office* and *system
    operation* is really *supervision*. Only fires when what remains is long
    enough to mean something; otherwise the predicate is returned unchanged.
    """
    p = _SEP.sub("", pred or "")
    if not p:
        return pred
    for arg in (obj, subject):
        a = _SEP.sub("", str(arg or ""))
        if not a or len(a) < _MIN_PRED_TAIL or a == p:
            continue
        if p.startswith(a) and len(p) - len(a) >= _MIN_PRED_TAIL:
            return p[len(a):]
        if p.endswith(a) and len(p) - len(a) >= _MIN_PRED_TAIL:
            return p[: len(p) - len(a)]
    return pred


def govern_predicates(
    relations: list[Relation],
    object_properties: list[ObjectProperty] | None,
    seed_predicates: list[str] | None = None,
    *,
    suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX,
) -> dict:
    """Fold predicate surface variants to one canonical name, in place.

    Anchors, in order: ``seed_predicates`` (names already in use, e.g. from an
    earlier build), then the schema's object properties, then the shortest
    spelling seen. Returns stats ``{total, argument_noun_stripped, distinct_before,
    distinct_after, merged, anchored_to_schema, canonical_map}``; ``canonical_map``
    (form key -> canonical name) lets a caller apply the same vocabulary to
    property declarations and data values.
    """
    empty = {"total": 0, "argument_noun_stripped": 0, "distinct_before": 0, "distinct_after": 0,
             "merged": 0, "anchored_to_schema": 0, "canonical_map": {}}
    if not relations:
        return empty

    canon_by_norm: dict[str, str] = {}
    schema_norms: set[str] = set()
    for name in (seed_predicates or []):
        nm = str(name or "").strip()
        norm = normalize_predicate(nm, suffix_pattern=suffix_pattern) if nm else ""
        if norm and norm not in canon_by_norm:
            canon_by_norm[norm] = nm
            schema_norms.add(norm)
    for op in (object_properties or []):
        name = (op.name or "").strip()
        norm = normalize_predicate(name, suffix_pattern=suffix_pattern) if name else ""
        if norm and norm not in canon_by_norm:
            canon_by_norm[norm] = name
            schema_norms.add(norm)

    distinct_before = {(r.predicate or "").strip() for r in relations if (r.predicate or "").strip()}
    stripped = 0
    for rel in relations:
        pred = (rel.predicate or "").strip()
        lean = strip_argument_noun(pred, rel.subject or "", rel.object or "")
        if lean != pred:
            stripped += 1
            rel.predicate = lean

    for rel in relations:
        pred = (rel.predicate or "").strip()
        norm = normalize_predicate(pred, suffix_pattern=suffix_pattern)
        if not norm:
            continue
        if norm not in canon_by_norm:
            canon_by_norm[norm] = pred
        elif norm not in schema_norms and len(pred) < len(canon_by_norm[norm]):
            canon_by_norm[norm] = pred   # not a schema anchor: prefer the shorter (base) spelling

    used: set[str] = set()
    anchored = 0
    for rel in relations:
        pred = (rel.predicate or "").strip()
        norm = normalize_predicate(pred, suffix_pattern=suffix_pattern)
        if not norm:
            continue
        canon = canon_by_norm.get(norm, pred)
        if canon != pred:
            rel.predicate = canon
        if norm in schema_norms:
            anchored += 1
        used.add(canon)

    return {
        "total": len(relations),
        "argument_noun_stripped": stripped,
        "distinct_before": len(distinct_before),
        "distinct_after": len(used),
        "merged": max(0, len(distinct_before) - len(used)),
        "anchored_to_schema": anchored,
        "canonical_map": dict(canon_by_norm),
    }


def apply_canonical_predicates(canonical_map: dict[str, str], names: list[str], *,
                               suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX) -> list[str]:
    """Map each name through ``canonical_map`` by its form key (unknown names pass through)."""
    out = []
    for n in names:
        norm = normalize_predicate(str(n or ""), suffix_pattern=suffix_pattern)
        out.append(canonical_map.get(norm, n) if norm else n)
    return out


# ───────────────────────── direction vote ─────────────────────────

_DIR_MIN_SAMPLES = 4      # fewer occurrences than this is not evidence
_DIR_MIN_RATIO = 0.75     # the majority direction must be at least this dominant


def vote_relation_direction(
    relations: list[Relation],
    entity_class: dict[str, str],
    *,
    suffix_pattern: re.Pattern | None = _DEFAULT_SUFFIX,
) -> dict:
    """Flip relations written against the majority direction of their predicate, in place.

    Extractors swap subject and object regardless of predicate, and which
    predicates appear depends on the domain, so no rule per predicate can be
    written down. Instead the data decides: for predicate P joining types A and
    B, if (A->B) outnumbers (B->A) by ``_DIR_MIN_RATIO`` over at least
    ``_DIR_MIN_SAMPLES`` samples, the minority is flipped. Most entities carry no
    type, so a second pass votes by *role*: an entity that is always an object of
    P sitting in subject position (and vice versa) marks a flipped relation. Weak
    or ambiguous evidence leaves the relation alone.

    Returns ``{"flipped", "predicates"}``.
    """
    if not relations:
        return {"flipped": 0, "predicates": 0}

    def _type_of(name: str) -> str:
        n = (name or "").strip()
        return entity_class.get(n) or n

    def _norm(p: str) -> str:
        return normalize_predicate(p or "", suffix_pattern=suffix_pattern)

    tally: dict[str, dict[tuple[str, str], list[int]]] = {}
    for r in relations:
        p = _norm(r.predicate)
        s, o = _type_of(r.subject), _type_of(r.object)
        if not p or not s or not o or s == o:
            continue          # same type on both sides: no basis for a direction
        key = (s, o) if s <= o else (o, s)
        slot = tally.setdefault(p, {}).setdefault(key, [0, 0])
        slot[0 if (s, o) == key else 1] += 1
    majority: dict[tuple[str, tuple[str, str]], int] = {}
    for p, pairs in tally.items():
        for key, (fwd, rev) in pairs.items():
            total = fwd + rev
            if total < _DIR_MIN_SAMPLES:
                continue
            if max(fwd, rev) / total < _DIR_MIN_RATIO:
                continue
            majority[(p, key)] = 0 if fwd >= rev else 1

    role: dict[str, dict[str, list[int]]] = {}
    for r in relations:
        p = _norm(r.predicate)
        s_, o_ = (r.subject or "").strip(), (r.object or "").strip()
        if not p or not s_ or not o_ or s_ == o_:
            continue
        role.setdefault(p, {}).setdefault(s_, [0, 0])[0] += 1
        role.setdefault(p, {}).setdefault(o_, [0, 0])[1] += 1

    def _role_says_flipped(p: str, s_: str, o_: str) -> bool:
        rp = role.get(p) or {}
        rs, ro = rp.get(s_), rp.get(o_)
        if not rs or not ro:
            return False
        # This relation's own contribution is excluded: it cannot be its own evidence.
        s_as_subj, s_as_obj = rs[0] - 1, rs[1]
        o_as_subj, o_as_obj = ro[0], ro[1] - 1
        if s_as_subj + s_as_obj < _DIR_MIN_SAMPLES or o_as_subj + o_as_obj < _DIR_MIN_SAMPLES:
            return False
        s_obj_ratio = s_as_obj / max(1, s_as_subj + s_as_obj)
        o_subj_ratio = o_as_subj / max(1, o_as_subj + o_as_obj)
        return s_obj_ratio >= _DIR_MIN_RATIO and o_subj_ratio >= _DIR_MIN_RATIO

    flipped = 0
    for r in relations:
        p = _norm(r.predicate)
        s, o = _type_of(r.subject), _type_of(r.object)
        key = (s, o) if s <= o else (o, s)
        want = majority.get((p, key))
        if want is not None:
            do_flip = (0 if (s, o) == key else 1) != want
        else:
            do_flip = bool(p) and _role_says_flipped(p, (r.subject or "").strip(), (r.object or "").strip())
        if do_flip:
            r.subject, r.object = r.object, r.subject
            flipped += 1
    return {"flipped": flipped, "predicates": len(majority)}


# ───────────────────────── stem / co-extension merge ─────────────────────────

_RESERVED = {"instanceOf", "subClassOf"}
_COEXT_SAMPLE = 500     # (subject, object) pairs per predicate compared for containment


def _stem_is_usable(stem: str, group: list[tuple[str, int]]) -> bool:
    """May the stem itself be shown as the name?"""
    if not any("가" <= ch <= "힣" for p, _ in group for ch in p):
        return False
    longest = max(len(re.sub(r"[\s_\-·/]+", "", p)) for p, _ in group)
    return len(stem) >= 2 and len(stem) * 2 >= longest


def _digits(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.isdigit())


def merge_predicates(
    relations: list[Relation],
    normalize_key: Callable[[str], str],
) -> dict:
    """Merge predicates that are one relation under several names, in place.

    Pass 1 groups predicates by ``normalize_key`` (a content-morpheme key; digits
    are put back so numbered variants stay apart) and keeps the most frequent
    spelling, or the stem itself when it is a usable Korean name. Pass 2 merges
    two remaining predicates when the smaller one's (subject, object) set is
    wholly contained in the other's: same extension, same relation. Duplicate
    triples produced by a merge are removed. Ported from the production
    ``normalize_predicates``. Returns ``{"merged_predicates": n}``.
    """
    counts: dict[str, int] = {}
    for r in relations:
        p = (r.predicate or "").strip()
        if p and p not in _RESERVED:
            counts[p] = counts.get(p, 0) + 1
    preds = list(counts.items())
    if len(preds) < 2:
        return {"merged_predicates": 0}

    by_stem: dict[str, list[tuple[str, int]]] = {}
    for p, n in preds:
        try:
            k = normalize_key(p)
        except Exception:
            k = p
        if k:
            by_stem.setdefault(k + _digits(p), []).append((p, n))
    rename: dict[str, str] = {}
    for stem, group in by_stem.items():
        canon = max(group, key=lambda t: (t[1], -len(t[0])))[0]
        has_num = any(_digits(p) for p, _ in group)
        target = (stem if stem and stem != canon and not has_num
                  and _stem_is_usable(stem, group) else canon)
        for p, _ in group:
            if p != target:
                rename[p] = target

    remain = [p for p, _ in preds if p not in rename]
    sig: dict[str, frozenset] = {}
    for p in remain:
        pairs = [(r.subject, r.object) for r in relations if r.predicate == p][:_COEXT_SAMPLE]
        sig[p] = frozenset(pairs)
    for i, a in enumerate(remain):
        if a in rename or not sig[a]:
            continue
        for b in remain[i + 1:]:
            if b in rename or not sig[b]:
                continue
            inter = len(sig[a] & sig[b])
            if inter and inter == min(len(sig[a]), len(sig[b])):
                keep, drop = (a, b) if counts.get(a, 0) >= counts.get(b, 0) else (b, a)
                rename[drop] = keep

    if not rename:
        return {"merged_predicates": 0}
    seen: set[tuple[str, str, str]] = set()
    kept: list[Relation] = []
    for r in relations:
        r.predicate = rename.get(r.predicate, r.predicate)
        key = (r.subject, r.predicate, r.object)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    relations[:] = kept
    return {"merged_predicates": len(rename)}
