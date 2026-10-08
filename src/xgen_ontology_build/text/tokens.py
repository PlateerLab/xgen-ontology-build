"""Tokens and IRIs — pure Python, zero deps.

CJK is tokenized into character bi-grams (like Lucene's CJK analyzer) so Korean /
Chinese / Japanese labels match with no morphological analyzer; Latin text falls back
to word tokens. ``safe_uri`` makes an IRI local name, ``normalize_name`` a comparable
spelling.
"""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[가-힣぀-ヿ一-鿿]+")
# Characters not allowed unescaped in a Turtle IRIREF local name (plus whitespace).
_IRI_BAD = re.compile(r"[\s<>\"{}|^`\\()\[\],;:/?#%&*!+=~'·•«»‘’“”]")


def tokenize(text: str) -> list[str]:
    """Latin word tokens + CJK character bi-grams (unigram if length 1)."""
    text = (text or "").lower()
    toks = _WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def safe_uri(name: str) -> str:
    """A Turtle-safe IRI local name. Keeps Unicode letters/digits (incl. Korean),
    collapses everything else to ``_``. Returns ``"Unknown"`` if nothing survives."""
    name = unicodedata.normalize("NFC", (name or "").strip())
    if not name:
        return "Unknown"
    out = _IRI_BAD.sub("_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "Unknown"


def normalize_name(name: str) -> str:
    """NFC + whitespace collapse + strip stray quote/punctuation artefacts."""
    if not name:
        return ""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.strip("\"'.,;:")
