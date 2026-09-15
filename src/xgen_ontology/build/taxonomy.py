"""Hierarchy induction from raw text and from the names themselves -- zero LLM calls.

Ported from the XGEN production build, where the base extraction alone produced
a *flat* schema. Three passes, all name/structure based, no domain word lists:

* **Hearst patterns** (:func:`hearst_hierarchy`) read is-a straight out of prose:
  "blood, hair root, etc. bio-samples" says *bio-sample* is a hypernym of *blood*
  and *hair root*. Classic 1992 method -- high precision, low recall by nature,
  so it picks up sure things rather than everything.
* **Name hygiene** before structure is read: :func:`fold_name_fragments` folds a
  short name that never stood alone in the source back into the name it was cut
  from; :func:`prune_common_words` drops an unlinked everyday word that got
  extracted as an entity.
* **Head-noun decomposition** (:func:`induce_head_noun_hierarchy`) reads is-a out
  of class *and instance* names: in a compound the head (the general category)
  comes last, in Korean and English alike -- "standing audit office" is a kind of
  "audit office". A name used as a head is promoted to a class; a code-prefixed
  spelling folds into its canonical name; a shared leading word links neighbours.

Korean-tuned (:mod:`xgen_ontology.korean` supplies the tokenizer) but degrades to
word-boundary-only behavior with no morphological analyzer installed.
"""
from __future__ import annotations

from collections import defaultdict

from ..korean import tokenize
from ..models import Class, Concepts, Instance, Relation
from .deterministic import is_common_word, prose_only  # noqa: F401  (re-exported)

# ───────────────────────── Hearst patterns ─────────────────────────

# Tags accepted as "noun". SL = foreign-script token, SN = digit run -- both let
# through so mixed-script terms ("TimeCheck", a screening item's own code number)
# survive as hyponyms/hypernyms.
_NOUN_TAGS = ("NNG", "NNP", "SL", "SN")
# List separators inside an enumeration: SP = comma/middle-dot, MAG = a coordinating
# adverb ("and", "or"). Particles all start with "J" and are matched by prefix, not
# enumerated, so the anchor's own inflected forms are caught automatically.
_SEP_TAGS = ("SP", "MAG")
_PARTICLE_PREFIX = "J"
# The anchor word. A Hearst pattern is defined by anchoring on a fixed lexical item
# (English implementations anchor on "such as" the same way); this is the one word
# this module hard-codes, and it is a dependent noun meaning roughly "etc./and so on".
_ANCHOR_FORM, _ANCHOR_TAG = "등", "NNB"

_MIN_NAME_LEN, _MAX_NAME_LEN = 2, 20
# Discriminativeness cap, same value the entity-extraction side uses: a name that
# shows up in more than this fraction of chunks isn't telling you anything about
# what it's the hypernym *of* (the IDF intuition -- holds regardless of domain).
_DEFAULT_MAX_COVERAGE = 0.30
# A hypernym only counts once it has this many distinct hyponyms. This is Hearst's
# standard filter: a pair seen only once is usually a mis-parsed sentence, not a
# real is-a relationship (observed on real data: "purchase-cap-compliance is-a
# essence" from a single, oddly-phrased sentence).
_DEFAULT_MIN_HYPONYMS = 2


def extract_hearst_pairs(text: str) -> list[tuple[str, str]]:
    """One chunk of prose -> ``[(hyponym, hypernym), ...]``, pre-filtering (raw, noisy).

    Reads morpheme tags directly rather than slicing the string with a regex,
    which buys two things a regex can't: the modifying clause in front of a noun
    doesn't get swept in ("a center established to support horse-industry
    startups and jobs" would otherwise become the *entire* hyponym), and
    particles don't need to be enumerated by hand (spelling every case-marked
    form of the anchor word by hand would still miss some; matching on the tag
    catches all of them for free). Where a noun run ends is where a name ends,
    and that boundary is a tag change, not a character class.
    """
    if _ANCHOR_FORM not in (text or ""):
        return []  # skip tokenizing entirely when the anchor isn't even present
    toks = tokenize(text)
    if not toks:
        return []
    toks = list(toks)
    out: list[tuple[str, str]] = []
    for i, tok in enumerate(toks):
        if tok.form != _ANCHOR_FORM or tok.tag != _ANCHOR_TAG:
            continue

        # Hypernym: the noun phrase right after the anchor, skipping its particles.
        j = i + 1
        while j < len(toks) and toks[j].tag.startswith(_PARTICLE_PREFIX):
            j += 1
        hyper = _noun_run(toks, j, +1)
        if not _in_name_range(hyper):
            continue
        # If a derivational suffix follows ("various", "confirming"), that noun is
        # the stem of an adjective/verb, not a category -- promoting it to a class
        # produces nonsense like "various <- glamping, English village" (observed).
        # This is judged by tag (XSA/XSV), not by a word list.
        end = j
        while end < len(toks) and toks[end].tag in _NOUN_TAGS:
            end += 1
        if end < len(toks) and toks[end].tag in ("XSA", "XSV"):
            continue

        # Hyponyms: walk backward from the anchor collecting the enumeration.
        # A verb, ending or sentence break marks where the list starts, which is
        # exactly why no modifying clause gets pulled in here either.
        items: list[str] = []
        current: list[str] = []
        k = i - 1
        while k >= 0:
            tag = toks[k].tag
            if tag in _NOUN_TAGS:
                current.insert(0, toks[k].form)
            elif tag in _SEP_TAGS:
                if current:
                    items.insert(0, "".join(current))
                    current = []
            else:
                break
            k -= 1
        if current:
            items.insert(0, "".join(current))

        for hypo in items:
            if _in_name_range(hypo) and hypo != hyper:
                out.append((hypo, hyper))
    return out


def _noun_run(toks, start: int, step: int) -> str:
    buf: list[str] = []
    i = start
    while 0 <= i < len(toks) and toks[i].tag in _NOUN_TAGS:
        buf.append(toks[i].form)
        i += step
    return "".join(buf if step > 0 else reversed(buf))


def _in_name_range(name: str) -> bool:
    return bool(name) and _MIN_NAME_LEN <= len(name) <= _MAX_NAME_LEN


def _canonical_parents(by_parent: dict[str, set]) -> dict[str, set]:
    """Fold hypernym spellings that are really the same concept, using corpus evidence.

    If one hypernym's spelling is a prefix of another ("bio-sample" vs.
    "bio-sample-collection"), the short spelling only absorbs the long one when
    the short spelling *also* stands as a hypernym on its own (has its own
    hyponyms). That's the evidence that a word boundary actually falls there --
    without it a name like "Korea Racing Authority" would get incorrectly split
    just because "Korea" happens to be a shorter prefix.
    """
    merged: dict[str, set] = defaultdict(set)
    names = sorted(by_parent, key=len)  # shortest first so a prefix claims its spot early
    for name in names:
        target = name
        for shorter in names:
            if shorter == name:
                break  # only consider names strictly shorter than this one
            if name.startswith(shorter) and by_parent.get(shorter):
                target = shorter
                break
        merged[target] |= by_parent[name]
    return merged


def hearst_hierarchy(
    texts: list[str],
    *,
    min_hyponyms: int = _DEFAULT_MIN_HYPONYMS,
    max_coverage: float = _DEFAULT_MAX_COVERAGE,
    header_patterns=(),
) -> list[tuple[str, str]]:
    """Chunks of prose -> ``[(parent, child), ...]``. Zero LLM calls.

    Each text is passed through :func:`prose_only` first -- callers do not need
    to pre-filter table content themselves; see its docstring for why that
    matters (measured: 80% of pairs found without this guard were wrong).

    ``min_hyponyms``: how many distinct hyponyms a hypernym needs before it's
    trusted (see module docstring -- this is what buys precision).
    ``max_coverage``: names appearing in more than this fraction of chunks are
    dropped as non-discriminative (generic words like "etc." or "results" show up
    everywhere and say nothing about what they're the hypernym of).
    """
    prose_texts = [prose_only(t, header_patterns) for t in texts]
    by_parent: dict[str, set] = defaultdict(set)
    for text in prose_texts:
        for hypo, hyper in extract_hearst_pairs(text):
            by_parent[hyper].add(hypo)

    n = max(1, len(prose_texts))
    for name in list(by_parent):
        if sum(1 for t in prose_texts if name in t) / n > max_coverage:
            del by_parent[name]
    by_parent = _canonical_parents(by_parent)

    # One parent per child: the schema's `parent` field is single-valued, and
    # there's no principled way to pick among several without one, so the
    # candidate with more supporting hyponyms wins.
    best: dict[str, tuple[str, int]] = {}
    for parent, children in by_parent.items():
        if len(children) < min_hyponyms:
            continue
        for child in children:
            if child in parent or parent in child:
                continue  # substring containment means "compound", not "enumeration"
            if len(children) > best.get(child, ("", 0))[1]:
                best[child] = (parent, len(children))
    return [(parent, child) for child, (parent, _count) in sorted(best.items())]


# ───────────────────────── name-structure induction ─────────────────────────

_HEAD_MIN_TAIL = 2      # a head noun needs at least this many characters to mean anything
_HEAD_MIN_MOD = 2       # same for the modifier in front of it
_VARIANT_MAX_LEN = 20   # names longer than this are not considered a fragment of another
_MORPH_BUDGET = 30000   # above this many names, morpheme boundaries are skipped (build time)
_COMMON_MAX_LEN = 8     # a common-word node is a short single word
_HANGUL = ("가", "힣")
DEFAULT_RELATED_PREDICATE = "관련"


def _morph_starts(text: str) -> set[int] | None:
    toks = tokenize(text)
    return {t.start for t in toks} if toks else None


def _has_hangul(text: str) -> bool:
    return any(_HANGUL[0] <= ch <= _HANGUL[1] for ch in text)


def _is_code(text: str) -> bool:
    """A classification code or serial number: has a digit, no Hangul."""
    return bool(text) and any(ch.isdigit() for ch in text) and not _has_hangul(text)


def _boundary_parts(label: str, use_morph: bool = True) -> tuple[list[str], list[str]]:
    """A name cut at its boundaries -> ``(leading pieces, trailing pieces)``."""
    heads: list[str] = []
    tails: list[str] = []
    parts = label.split()
    if len(parts) > 1:
        for cut in range(1, len(parts)):
            heads.append(" ".join(parts[:cut]))
            tails.append(" ".join(parts[cut:]))
    # Spaced names already have word boundaries; morphology only runs on unspaced compounds.
    starts = _morph_starts(label) if (use_morph and len(parts) == 1) else None
    for off in sorted(starts or ()):
        if 0 < off < len(label):
            heads.append(label[:off])
            tails.append(label[off:])
    return heads, tails


def _variant_holders(weak: list[str], labels: list[str]) -> dict[str, list[str]]:
    """For each fragment candidate, the longer names containing it (longest first).

    A candidate that stands as a whole word inside the longer name is not a
    fragment of it but a place to attach to, so it is not counted.
    """
    weak_labels = set(weak)
    hits: dict[str, list[str]] = {}
    for ll in labels:
        if len(ll) <= _HEAD_MIN_TAIL:
            continue
        words = set(ll.split())
        for cut in range(1, min(len(ll), _VARIANT_MAX_LEN + 1)):
            for piece in (ll[cut:], ll[:cut]):
                if len(piece) >= _HEAD_MIN_TAIL and piece in weak_labels and piece not in words:
                    hits.setdefault(piece, []).append(ll)
    for k in hits:
        hits[k] = sorted(set(hits[k]), key=lambda h: (-len(h), h))   # longest first, ties by spelling
    return {lb: hits[lb] for lb in weak if hits.get(lb)}


def _resolve_fold_chain(fold: dict[str, str]) -> dict[str, str]:
    """Follow chained folds to the surviving name."""
    out = dict(fold)
    for su in list(out):
        seen = {su}
        tu = out[su]
        while tu in out and tu not in seen:
            seen.add(tu)
            tu = out[tu]
        out[su] = tu
    return out


def fold_name_fragments(
    names: list[str],
    chunks_of: dict[str, set],
    linked: set[str],
) -> dict[str, str]:
    """Fold name fragments that never stood alone in the source back into the name they came from.

    A fragment is a short name (2..20 chars) with no edge of any kind and at least
    one source chunk; it is folded into a longer name that contains it (not as a
    whole word) when every chunk the fragment appears in also contains the longer
    name. Returns ``{fragment: holder}``; the caller removes the fragment and moves
    its chunks. Ported from the production ``consolidate_name_variants``.
    """
    weak = sorted(
        {n for n in names if n and _HEAD_MIN_TAIL <= len(n) <= _VARIANT_MAX_LEN
         and n not in linked and chunks_of.get(n)},
        key=lambda s: (len(s), s))
    if not weak:
        return {}
    holders = _variant_holders(weak, [n for n in dict.fromkeys(names) if n])
    if not holders:
        return {}
    fold: dict[str, str] = {}
    for sl in weak:
        sc = chunks_of.get(sl) or set()
        if not sc:
            continue
        for ll in holders.get(sl, ()):
            if ll != sl and sc <= (chunks_of.get(ll) or set()):
                fold[sl] = ll
                break
    return _resolve_fold_chain(fold)


def prune_common_words(
    instance_names: list[str], linked: set[str], *, rank_max: int = 5000,
) -> set[str]:
    """Instance names that are a verbal habit rather than a name.

    A short (2..8 chars) single common noun with no edge of any kind: not typed,
    in no relation. See :func:`~xgen_ontology.build.deterministic.is_common_word`
    for how "common" is judged (analyzer dictionary rank, no word list).
    """
    return {n for n in dict.fromkeys(instance_names)
            if n and 2 <= len(n) <= _COMMON_MAX_LEN and n not in linked
            and is_common_word(n, rank_max)}


def induce_head_noun_hierarchy(
    labels: list[str], *, class_labels: set[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, str], list[tuple[str, str]]]:
    """Names -> ``(subclass edges, sameAs renames, related pairs)``. Zero LLM calls.

    In a Korean (or English) compound the **head sits last**: if ``B`` is a
    boundary-aligned suffix of ``A``, ``A`` is a kind of ``B`` ("standing audit
    office" is-a "audit office"). Only tokenizer-recognized boundaries count. A
    space before the tail makes two words, not a compound: when what precedes the
    space is only a code ("NA162000.23 Jeju Ranch Business") the two spellings
    name the same thing (sameAs rename); otherwise nothing is produced. A leading
    word that is itself a known name is a same-topic neighbour ("related").

    ``class_labels`` orders the lookup so a class wins over an instance with the
    same spelling. Returns ``edges`` as ``[(parent, child)]``, ``rename`` as
    ``{spelling: canonical}``, ``related`` as ``[(name, neighbour)]``.
    """
    class_labels = class_labels or set()
    uniq = [lb for lb in dict.fromkeys(labels) if lb]
    if len(uniq) < 2:
        return [], {}, []
    # Classes first, then longer names first: the order the production store uses.
    uniq.sort(key=lambda lb: (lb not in class_labels, -len(lb)))
    by_label = set(uniq)
    use_morph = len(uniq) <= _MORPH_BUDGET
    best: dict[str, str] = {}
    same: dict[str, str] = {}
    related: list[tuple[str, str]] = []
    for cl in uniq:
        heads, tails = _boundary_parts(cl, use_morph)
        for tail in tails:
            if tail not in by_label or tail == cl or len(tail) < _HEAD_MIN_TAIL or not cl.endswith(tail):
                continue
            mod = cl[:len(cl) - len(tail)]
            if len(mod) < _HEAD_MIN_MOD:
                continue
            if mod[-1].isspace():
                if _is_code(mod.strip()):
                    same.setdefault(cl, tail)
                continue
            if cl not in best:
                best[cl] = tail
            break
        for head in heads:
            if head in by_label and head != cl and head in cl.split():
                related.append((cl, head))
                break
    edges = [(parent, child) for child, parent in best.items()]
    return edges, same, related


def induce_hierarchy(
    concepts: Concepts,
    texts: list[str] | None = None,
    instances: list[Instance] | None = None,
    relations: list[Relation] | None = None,
    *,
    min_hyponyms: int = _DEFAULT_MIN_HYPONYMS,
    max_coverage: float = _DEFAULT_MAX_COVERAGE,
    related_predicate: str | None = DEFAULT_RELATED_PREDICATE,
    header_patterns=(),
) -> dict[str, int]:
    """Induce hierarchy in place: Hearst patterns over ``texts``, then name structure over classes *and* instances.

    Name-structure results are applied the way the production store applies them:
    a name used as another name's head becomes a class (an instance is promoted);
    a class child gets a ``subClassOf`` edge; an instance child is typed by its
    head class (an extra ``rdf:type`` when it already has one); a code-prefixed
    spelling is folded into its canonical name; a leading-word neighbour becomes a
    ``related_predicate`` relation (``None`` disables that).

    Returns ``{"hearst_classes_added", "hearst_edges_added", "compound_edges_added",
    "promoted", "typed", "renamed", "related_added"}``.
    """
    instances = instances if instances is not None else []
    relations = relations if relations is not None else []
    existing = {c.name for c in concepts.classes if c.name}
    counts = {"hearst_classes_added": 0, "hearst_edges_added": 0, "compound_edges_added": 0,
              "promoted": 0, "typed": 0, "renamed": 0, "related_added": 0}

    if texts:
        parent_of: dict[str, str] = {}
        for parent, child in hearst_hierarchy(texts, min_hyponyms=min_hyponyms,
                                              max_coverage=max_coverage,
                                              header_patterns=header_patterns):
            parent_of.setdefault(child, parent)
            for name in (parent, child):
                if name not in existing:
                    concepts.classes.append(Class(name=name, parent=parent_of.get(name)))
                    existing.add(name)
                    counts["hearst_classes_added"] += 1
            edge = (parent, child)
            if edge not in concepts.class_hierarchy:
                concepts.class_hierarchy.append(edge)
                counts["hearst_edges_added"] += 1

    class_labels = {c.name for c in concepts.classes if c.name}
    inst_labels = [i.name for i in instances if i.name]
    edges, rename, related = induce_head_noun_hierarchy(
        [*class_labels, *inst_labels], class_labels=class_labels)

    # A head is a concept: an instance used as a head becomes a class (its chunks come along).
    parents = {p for p, _ in edges}
    promoted = sorted(parents - class_labels)
    if promoted:
        chunks: dict[str, list[str]] = {p: [] for p in promoted}
        for i in instances:
            if i.name in chunks:
                chunks[i.name].extend(c for c in i.source_chunks if c not in chunks[i.name])
        instances[:] = [i for i in instances if i.name not in chunks]
        for p in promoted:
            concepts.classes.append(Class(name=p, source_chunks=chunks[p]))
            class_labels.add(p)
            counts["promoted"] += 1
    typed_by: dict[str, set] = {}
    for i in instances:
        if i.name:
            typed_by.setdefault(i.name, set()).add(i.class_name or "")
    hier = set(concepts.class_hierarchy)
    for parent, child in edges:
        if child in class_labels:
            if (parent, child) not in hier:
                concepts.class_hierarchy.append((parent, child))
                hier.add((parent, child))
                counts["compound_edges_added"] += 1
            continue
        classes_of = typed_by.get(child, set())
        if parent in classes_of:
            continue
        if not classes_of - {""}:
            for i in instances:
                if i.name == child and not i.class_name:
                    i.class_name = parent
        else:
            src = next(i for i in instances if i.name == child)
            instances.append(Instance(name=child, class_name=parent,
                                      source_chunks=list(src.source_chunks)))
        typed_by.setdefault(child, set()).add(parent)
        counts["typed"] += 1

    if rename:
        from .dedup import Deduplicator  # local import: avoid a hard import cycle

        Deduplicator._apply_class(rename, concepts, instances)
        Deduplicator._apply_instance(rename, instances, relations, [])
        counts["renamed"] = len(rename)

    if related_predicate and related:
        have = {(r.subject, r.predicate, r.object) for r in relations}
        for name, head in related:
            name, head = rename.get(name, name), rename.get(head, head)
            key = (name, related_predicate, head)
            if name != head and key not in have:
                relations.append(Relation(subject=name, predicate=related_predicate, object=head))
                have.add(key)
                counts["related_added"] += 1
    return counts
