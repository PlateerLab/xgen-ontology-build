"""Korean text utilities — label cleaning + morphology, optional dependency.

Everything here degrades gracefully with no ``kiwipiepy`` installed: label
cleaning still runs (regex + Unicode only), only the morpheme-aware parts
(sentence-fragment detection, morpheme boundaries used by
:mod:`xgen_ontology.build.taxonomy`) become no-ops. Install the ``korean``
extra (``pip install "xgen-ontology[korean]"``) to get the full behavior.

Nothing here is English-only-language-specific in its *architecture* — the
label-cleaning regexes are the one genuinely Korean-specific piece (list
markers, letter-spacing); the sentence-fragment and morpheme-boundary helpers
are thin wrappers around a pluggable tokenizer and would work with any
morphological analyzer exposing the same ``(form, tag, start)`` shape.
"""
from __future__ import annotations

import html
import re
import unicodedata

_kiwi = None
_kiwi_unavailable = False


def get_kiwi():
    """The process-wide Kiwi instance, or ``None`` if ``kiwipiepy`` isn't installed."""
    global _kiwi, _kiwi_unavailable
    if _kiwi_unavailable:
        return None
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
        except Exception:
            _kiwi_unavailable = True
            return None
    return _kiwi


def tokenize(text: str):
    """Morpheme tokens for ``text``, or ``None`` if no analyzer is available."""
    kiwi = get_kiwi()
    if kiwi is None:
        return None
    try:
        return kiwi.tokenize(text or "")
    except Exception:
        return None


# Leading list markers: 1-2 digit numbers, circled digits and roman numerals may be
# glued to the text ("6.경마"), but a single Hangul syllable ("가." "나.") only counts
# as a marker when followed by whitespace -- otherwise "내.외부" and "일.숙직비" would
# be mangled. Bullet glyphs only count when followed by whitespace too.
_LEAD = re.compile(
    r"^(?:(?:\d{1,2}|[①-⑳]|[IVX]{1,4}|[Ⅰ-Ⅻ])[.)]\s*(?=[가-힣A-Za-z(])"
    # A numbered heading is only stripped when followed by whitespace. Glued to a
    # digit it's not a number, it's a value (3.14). This branch also catches
    # multi-part numbering (1-1. 12-2.) and headings that start with a digit
    # (e.g. "01. Chapter 4 revolution") that the branch above misses because it
    # requires a letter right after.
    r"|\d{1,3}(?:-\d{1,3})*[.)]\s+(?=\S)"
    # Parenthesized numbers: (1) (12). Hangul-lettered parens are a separate branch
    # below so "(주)마장" (a company-name abbreviation) is left untouched.
    r"|\(\s*\d{1,3}\s*\)\s*(?=\S)"
    r"|[①-⑳]\s*"
    r"|[가-하][.)]\s+(?=\S)"
    r"|\([가-하]\)\s+"
    r"|[□■○●◦▣▶※•․ᄋㅇ*†]\s*)+"
)
# Trailing markers: footnote-style (1) / circled digits / asterisks and daggers.
_TAIL = re.compile(r"(?:\s*\(\s*(?:\d{1,2}|[①-⑳])\s*\)|\s*[①-⑳]|[*※†]+)+$")
_WRAP = "'‘’\"“” "
# Sentence-final punctuation. A name or a table cell does not end like a sentence.
SENT_END = re.compile(r"[.!?。！？]\Z")
# Escapes left in text (a literal backslash-n, an escaped quote, a \uXXXX code).
# A backslash never means anything inside a name.
_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|[nrt]|.)")


def _unescape(m: "re.Match[str]") -> str:
    e = m.group(1)
    if e[0] == "u" and len(e) == 5:
        return chr(int(e[1:], 16))
    return " " if e in ("n", "r", "t") else e


def _opens(ch: str) -> bool:
    """An opening bracket or quote (Unicode categories Ps / Pi, or a straight double quote)."""
    return ch == '"' or unicodedata.category(ch) in ("Ps", "Pi")


def _closes(ch: str) -> bool:
    return ch == '"' or unicodedata.category(ch) in ("Pe", "Pf")


def _balanced(t: str) -> bool:
    """Brackets pair up in order (the open count never goes negative and ends at zero), and so do quotes."""
    depth = 0
    for ch in t:
        cat = unicodedata.category(ch)
        depth += (cat == "Ps") - (cat == "Pe")
        if depth < 0:
            return False
    cats = [unicodedata.category(ch) for ch in t]
    return depth == 0 and cats.count("Pi") == cats.count("Pf") and t.count('"') % 2 == 0


def strip_closers(s: str) -> str:
    """Drop trailing commas, semicolons, spaces and closing brackets or quotes, so a full stop before them shows."""
    t = (s or "").rstrip(" ,;")
    while t and _closes(t[-1]):
        t = t[:-1].rstrip(" ,;")
    return t


def _trailing_mark(t: str) -> bool:
    """Does ``t`` end in a mark that is not part of a name: a phrase separator (, ; :) or a symbol
    (a table border, an arrow, an operator, a check box)? A unit such as % (punctuation, not a
    symbol) stays. Known limit: a grade suffix glued to a letter ("AA+") goes too, since + shares
    its Unicode category with × = → and shape alone cannot tell a suffix from a connector."""
    ch = t[-1]
    return ch in ",;:" or unicodedata.category(ch)[0] == "S"


def is_marker_char(ch: str) -> bool:
    """A bullet: punctuation or a symbol that is not a letter, digit, bracket or quote."""
    return (bool(ch) and not ch.isalnum() and not _opens(ch) and not _closes(ch)
            and unicodedata.category(ch)[0] in "PS")


def starts_with_list_marker(s: str) -> bool:
    """Is this line a list item (numbered, lettered, circled or bulleted)?"""
    t = (s or "").lstrip()
    return bool(t) and (bool(_LEAD.match(t)) or is_marker_char(t[0]))


def _decode(s: str) -> str:
    """Resolve HTML entities and escapes. Runs before bullet detection, or the & of an entity reads as a bullet."""
    return _ESCAPE.sub(_unescape, html.unescape(s or ""))


def strip_list_markers(s: str) -> str:
    """Drop a leading/trailing outline marker (``"01. Foo"`` -> ``"Foo"``).

    Bullets the marker pattern does not list and a trailing separator or symbol are
    not part of the name either. A sentence-final mark is left for
    :func:`is_name_shape` to reject. A bracket or quote pair wrapping the whole name
    is removed when the inside is balanced on its own.
    """
    t = " ".join(_decode(s).split())
    core = _TAIL.sub("", _LEAD.sub("", t)).strip()
    while core and is_marker_char(core[0]):
        core = core[1:].lstrip()
    while core and _trailing_mark(core):
        core = core[:-1].rstrip()
    while len(core) > 2 and _opens(core[0]) and _closes(core[-1]) and _balanced(core[1:-1]):
        core = core[1:-1].strip()
    return core if core else t


def _despace_letterspaced(s: str) -> str:
    """Undo letter-spacing (every token a single character) by joining with no space."""
    parts = s.split(" ")
    if len(parts) < 2 or any(len(p) != 1 for p in parts):
        return s
    return "".join(parts)


def normalize_text(s: str) -> str:
    """Normalize form only: resolve entities and escapes, NFKC, single spaces, strip wrapping
    quotes, undo letter-spacing.

    List markers are left in place -- callers that need to detect a table header
    (which cares whether a marker was present) run before :func:`normalize_label`.
    """
    t = unicodedata.normalize("NFKC", _decode(s))
    return _despace_letterspaced(" ".join(t.split()).strip(_WRAP))


def normalize_label(s: str) -> str:
    """The canonical spelling of a name: strip list markers, then :func:`normalize_text`."""
    return normalize_text(strip_list_markers(s or ""))


_SENTENCE_ENDINGS = ("EF", "EC")
_VERBAL_TAIL_TAGS = ("VV", "VA", "VX", "VCP", "VCN", "ETN", "EP")
_PUNCT_TAGS = ("SF", "SP", "SS", "SE", "SO", "SW", "SB")


def is_sentence_like(name: str) -> bool:
    """Is this a sentence fragment rather than a name?

    True when it has a sentence-final or connective ending, or its last content
    morpheme is a verb/adjective stem. Single-token labels are skipped -- a short
    label (e.g. a Korean noun meaning "note" or "report") is easily mistaken for a
    verb form by the tagger, so this only fires on multi-token text. An
    adnominal-form morpheme alone (roughly "-the one that is X") is common inside
    real names too, so it alone doesn't decide anything.
    """
    t = (name or "").strip()
    if len(t.split()) < 2 or len(t) < 6:
        return False
    toks = tokenize(t)
    if not toks:
        return False
    tags = [x.tag for x in toks if x.tag not in _PUNCT_TAGS]
    return any(tag in _SENTENCE_ENDINGS for tag in tags) or (bool(tags) and tags[-1] in _VERBAL_TAIL_TAGS)


def is_name_shape(s: str) -> bool:
    """Does this have the shape of a name?

    It starts with a letter, a digit, an opening bracket or a quote; its brackets and
    quotes pair up; it does not end in a symbol, separator or sentence punctuation; it
    holds no line break or backslash. Decided by Unicode categories, with no word list.
    """
    t = (s or "").strip()
    if not t or "\n" in t or "\\" in t:
        return False
    if not (t[0].isalnum() or t[0] == '"' or _opens(t[0])):
        return False
    if _trailing_mark(t) or SENT_END.search(t):
        return False
    return _balanced(t)


def clean_name(name: str) -> str | None:
    """The normalized name, or ``None`` if it's a sentence fragment or not shaped like a name."""
    t = normalize_label(name)
    if not t or is_sentence_like(t) or not is_name_shape(t):
        return None
    return t
