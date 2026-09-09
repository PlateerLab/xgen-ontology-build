"""Is-a hierarchy induction from raw text and from class names -- zero LLM calls.

Two independent, complementary techniques, both ported from a Korean-language
production ontology pipeline where the base build (headers + LLM-free table
analysis) alone produced a *flat* schema -- classes with no hierarchy at all.

* **Hearst patterns** (:func:`hearst_hierarchy`) read is-a straight out of prose:
  "blood, hair root, etc. bio-samples" says *bio-sample* is a hypernym of *blood*
  and *hair root*. This is the classic 1992 method -- high precision, low recall
  by nature, so it's used to pick up sure things, not to find everything.
* **Head-noun compound decomposition** (:func:`induce_head_noun_hierarchy`) reads
  is-a out of the *class names themselves*: in a compound noun the head (the
  general category) comes last, in both Korean and English -- "corporate racing
  business" is a kind of "business", "standing audit office" is a kind of "audit
  office". This is a morphological fact, not a domain word list, so it holds
  across any corpus.

Both are Korean-tuned (:mod:`xgen_ontology.korean` supplies the tokenizer) but
degrade to word-boundary-only behavior with no morphological analyzer installed,
rather than raising.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..korean import tokenize
from ..models import Class, Concepts, Instance

# ───────────────────────── prose guard ─────────────────────────

_HTML_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.S | re.I)
_BRACKET_LINE = re.compile(r"^[\[(（].*[\])）]$")
_PIPE_RULE_LINE = re.compile(r"^[\s|:\-]+$")


def _looks_like_pipe_table(text: str) -> bool:
    """At least a few lines with 2+ interior pipes -- a markdown/plain-text grid."""
    hits = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or _PIPE_RULE_LINE.match(line):
            continue
        if line.count("|") >= 2:
            hits += 1
            if hits >= 2:
                return True
    return False


def prose_only(text: str) -> str:
    """The prose portion of ``text``, with any embedded table stripped.

    A Hearst pattern must not fire inside a table cell: a cell is an
    enumeration, not a sentence, so the noun phrase after the anchor word is
    not actually a hypernym of what came before it -- it's a shared column
    value, not a category. Measured on a real corpus: of the raw pairs Hearst
    found before this guard existed, 80% came from HTML tables embedded in
    otherwise-prose chunks, and nearly all of them were wrong -- a materials
    line item titled "production cost" ended up as the induced is-a parent of
    "plaque" and "signboard", just because a table cell happened to list them
    together.

    This only strips the two generic, format-level table shapes a plain-text
    extraction commonly leaves behind (literal ``<table>`` HTML, and a
    pipe-delimited grid). It intentionally does not try to detect a table that
    has been flattened to whitespace-aligned columns with no delimiter at all
    -- that needs column-shape heuristics tuned to a specific document corpus,
    which belongs in a caller's own preprocessing, not in a generic library.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if "<table" in t.lower():
        return _HTML_TABLE_BLOCK.sub(" ", t)
    if _looks_like_pipe_table(t):
        return "\n".join(line for line in t.splitlines()
                         if "|" not in line and not _BRACKET_LINE.match(line.strip()))
    return t


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
    prose_texts = [prose_only(t) for t in texts]
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


# ───────────────────────── head-noun compound decomposition ─────────────────────────

_HEAD_MIN_LEN = 2  # a head noun needs at least this many characters to mean anything
_MODIFIER_MIN_LEN = 2  # same for the modifier in front of it
_VARIANT_MAX_LEN = 20  # names longer than this aren't considered as a "piece" of another
# Above this many class names, skip morpheme boundaries and use word boundaries only.
# Trades a few missed compound links for build time that doesn't grow with corpus size
# (empirically: ~4s for 20K labels with morphology, ~30s for 150K).
_MORPHOLOGY_BUDGET = 30000
_HANGUL_RANGE = ("가", "힣")


def _has_hangul(text: str) -> bool:
    return any(_HANGUL_RANGE[0] <= ch <= _HANGUL_RANGE[1] for ch in text)


def _is_code(text: str) -> bool:
    """A classification code or serial number, not a name (has a digit, no Hangul)."""
    return bool(text) and any(ch.isdigit() for ch in text) and not _has_hangul(text)


def _morpheme_starts(text: str) -> set[int] | None:
    toks = tokenize(text)
    return {t.start for t in toks} if toks else None


def _boundary_parts(label: str, use_morphology: bool = True) -> tuple[list[str], list[str]]:
    """A name cut at its boundaries -> ``(leading pieces, trailing pieces)``.

    Both word boundaries and morpheme boundaries are considered, which keeps the
    number of pieces proportional to the name's length -- pair-matching against a
    lookup table stays a dictionary lookup instead of an all-pairs comparison.
    """
    heads: list[str] = []
    tails: list[str] = []
    words = label.split()
    if len(words) > 1:
        for cut in range(1, len(words)):
            heads.append(" ".join(words[:cut]))
            tails.append(" ".join(words[cut:]))
    # A name with spaces already has word boundaries to work with; morphology is
    # only worth running on names written as one unbroken compound (it costs time).
    starts = _morpheme_starts(label) if (use_morphology and len(words) == 1) else None
    for offset in sorted(starts or ()):
        if 0 < offset < len(label):
            heads.append(label[:offset])
            tails.append(label[offset:])
    return heads, tails


def induce_head_noun_hierarchy(
    labels: list[str],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Class names -> ``(subclass edges, sameAs-style rename map)``. Zero LLM calls.

    In both Korean and English, a compound noun's **head sits last**: if ``B`` is
    a suffix of ``A``, then ``A`` is a kind of ``B`` -- "corporate racing
    business" is-a "business", "standing audit office" is-a "audit office". This
    is a rule about word structure, not a domain word list, so it holds no matter
    what the corpus is about. A modifier in *front* is not evidence of anything:
    "product catalog" is a kind of catalog, not a kind of product.

    Two failure modes this filters out (found by running it on real data):

    * An accidental suffix match that starts mid-morpheme is not a real head
      noun. Only boundaries the tokenizer actually recognizes count.
    * **A name with a space in it is two words, not one compound.** "Department
      Chairman's Office" is not a subclass of "Chairman's Office" -- it's a
      table-header prefix stuck in front of a spelling that already existed. When
      the part in front of the space is *only* a code ("NA162000.23 Jeju Ranch
      Business"), that's the same concept under a different spelling, so it comes
      back as a rename (sameAs) instead of a hierarchy edge; otherwise it's
      neither and nothing is produced.

    Returns ``(edges, rename)`` where ``edges`` is ``[(parent, child), ...]`` and
    ``rename`` maps a name that should be folded away to the name it becomes.
    """
    uniq = [lb for lb in dict.fromkeys(labels) if lb]
    if len(uniq) < 2:
        return [], {}

    use_morphology = len(uniq) <= _MORPHOLOGY_BUDGET
    by_label = set(uniq)
    best: dict[str, tuple[str, str]] = {}  # child -> (parent, matched tail)
    rename: dict[str, str] = {}

    for name in uniq:
        heads, tails = _boundary_parts(name, use_morphology)
        for tail in tails:
            if tail not in by_label or tail == name or len(tail) < _HEAD_MIN_LEN:
                continue
            if not name.endswith(tail):
                continue
            modifier = name[: len(name) - len(tail)]
            if len(modifier) < _MODIFIER_MIN_LEN:
                continue
            if modifier[-1].isspace():
                # Space before the tail -> two words, not a compound (see docstring).
                if _is_code(modifier.strip()):
                    rename[name] = tail
                continue
            if name not in best:
                best[name] = (tail, tail)
            break

    edges = [(parent, child) for child, (parent, _tail) in best.items()]
    return edges, rename


# ───────────────────────── orchestrator ─────────────────────────


def induce_hierarchy(
    concepts: Concepts,
    texts: list[str] | None = None,
    instances: list[Instance] | None = None,
    *,
    min_hyponyms: int = _DEFAULT_MIN_HYPONYMS,
    max_coverage: float = _DEFAULT_MAX_COVERAGE,
) -> dict[str, int]:
    """Run both techniques against ``concepts`` (and its source ``texts``), in place.

    Order: Hearst first (it can *mint new classes* a table-header-only build never
    saw), then head-noun decomposition over the resulting full class list (it only
    *connects* existing classes, so it benefits from Hearst having run first).
    Safe to call with ``texts=None`` (head-noun decomposition only, e.g. for a
    pure-CSV build) or on a language with no morphological analyzer installed
    (Hearst finds nothing without ``texts``' tokenizer; head-noun falls back to
    word-boundaries-only and still works on space-separated compounds).

    Returns ``{"hearst_classes_added", "hearst_edges_added", "compound_edges_added",
    "renamed"}``.
    """
    existing = {c.name for c in concepts.classes if c.name}
    counts = {"hearst_classes_added": 0, "hearst_edges_added": 0,
              "compound_edges_added": 0, "renamed": 0}

    if texts:
        parent_of: dict[str, str] = {}
        for parent, child in hearst_hierarchy(texts, min_hyponyms=min_hyponyms,
                                              max_coverage=max_coverage):
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

    labels = [c.name for c in concepts.classes if c.name]
    compound_edges, rename = induce_head_noun_hierarchy(labels)
    for edge in compound_edges:
        if edge not in concepts.class_hierarchy:
            concepts.class_hierarchy.append(edge)
            counts["compound_edges_added"] += 1
    if rename:
        from .dedup import Deduplicator  # local import: avoid a hard import cycle

        Deduplicator._apply_class(rename, concepts, instances or [])
        counts["renamed"] = len(rename)

    return counts
