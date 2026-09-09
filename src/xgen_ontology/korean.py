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
    r"|[가-하][.)]\s+(?=[가-힣A-Za-z(])"
    r"|\([가-하]\)\s+"
    r"|[□■○●◦▣▶※•․ㅇㅇ*†]\s*)+"
)
# Trailing markers: footnote-style (1) / circled digits / asterisks and daggers.
_TAIL = re.compile(r"(?:\s*\(\s*(?:\d{1,2}|[①-⑳])\s*\)|\s*[①-⑳]|[*※†]+)+$")
_WRAP = "'‘’\"“” "


def strip_list_markers(s: str) -> str:
    """Drop a leading/trailing outline marker (``"01. Foo"`` -> ``"Foo"``)."""
    t = " ".join((s or "").split())
    core = _TAIL.sub("", _LEAD.sub("", t)).strip()
    return core if core else t


def _despace_letterspaced(s: str) -> str:
    """Undo letter-spacing (every token a single character) by joining with no space."""
    parts = s.split(" ")
    if len(parts) < 2 or any(len(p) != 1 for p in parts):
        return s
    return "".join(parts)


def normalize_text(s: str) -> str:
    """Normalize form only: NFKC, single spaces, strip wrapping quotes, undo letter-spacing.

    List markers are left in place -- callers that need to detect a table header
    (which cares whether a marker was present) run before :func:`normalize_label`.
    """
    t = unicodedata.normalize("NFKC", s or "")
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


def clean_name(name: str) -> str | None:
    """The normalized name, or ``None`` if it's a sentence fragment rather than a name."""
    t = normalize_label(name)
    if not t or is_sentence_like(t):
        return None
    return t
