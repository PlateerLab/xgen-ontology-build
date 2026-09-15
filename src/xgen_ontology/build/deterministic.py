"""Deterministic document -> ontology extraction: zero LLM calls.

Reads entities, classes and row facts out of the *structure* of a chunk (HTML
tables, whitespace row dumps, pipe grids, prose noun phrases) and returns the
same four-part result the LLM extractor does. This is the base build path of the
XGEN production pipeline, ported as-is; the LLM is reserved for a later
"enrich" pass (relations only).

Everything is position-and-shape based: no domain word lists. Korean morphology
(:mod:`xgen_ontology.korean`, optional ``kiwipiepy``) sharpens noun-phrase and
sentence-fragment detection; without it, table parsing still works and prose
extraction degrades to nothing rather than raising.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache
from typing import Any, NamedTuple

from ..korean import is_sentence_like, normalize_label, normalize_text, tokenize
from ..models import Class, Concepts, DataProperty, DataValue, Instance, Relation

_CELL_TAG = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.S)      # (attrs, body)
_SPAN = re.compile(r"(row|col)span\s*=\s*['\"]?(\d+)", re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_META = re.compile(r"<Document-Metadata>.*?</Document-Metadata>", re.S)
_MARKER = re.compile(r"\[(?:Image|Page Number)[^\]]*\]")
# Sentence-final punctuation. A table cell does not end like a sentence.
_SENT_END = re.compile(r"[.!?。！？]\Z")
# Affixes that only change the form, not the meaning, are not glued onto a name.
_AFFIX_SKIP = {"적", "들"}
# A number followed by at most one short unit and one short qualifier is a value.
_VALUE = re.compile(r"^[\s(\[]*[+\-]?[\d,.]+\s*[^\d\s]{0,4}(?:\s+[^\d\s]{1,3})?[)\]]*$")
_DATE = re.compile(r"^\d{4}[-./]\d{1,2}([-./]\d{1,2})?$")

_MIN_NAME, _MAX_NAME = 2, 40
_COVERAGE_MIN_CHUNKS = 20   # below this the discriminativeness ratio is not applied
_HEAD_MIN_CHILDREN = 2      # a head with this many names under it is a concept (class)
_MAX_ROW_LABEL = 80         # cap on a row label built from its identifying cells
_MAX_HEAD = 40              # cap on a column name; longer is a cell value
_CLASS_MAX_TOKENS = 6       # cap on class-name words; longer is a sentence
_PAIR_HEAD, _PAIR_TAIL = 4, 6      # entities per row to pair up
_HEADER_SCAN = 3                   # title / subtitle lines above a table
_ROW_DUMP_MIN_COLS = 4             # minimum columns for a line to count as a row dump
# Cells that start with an enumerator: (가) (1) [A] and bullet glyphs.
_ENUMERATOR = re.compile(r"^\s*(?:[\(\[（]\s*[가-힣a-zA-Z0-9]{1,3}\s*[\)\]）]|[□■○●◦▣▶※•])")

DEFAULT_COMMON_WORD_RANK = 5000


def strip_headers(text: str, patterns=()) -> str:
    """Remove ingestion-time preambles (retrieval headers a chunker prepends)."""
    t = text or ""
    for pat in patterns or ():
        t = pat.sub("", t)
    return t


def _clean(s: str) -> str:
    return normalize_text(html.unescape(_TAG.sub("", s or "")))


def is_class_name(s: str) -> bool:
    """Can this stand as a class name: a short noun phrase."""
    t = (s or "").strip()
    return (len(t) >= _MIN_NAME and len(t.split()) <= _CLASS_MAX_TOKENS
            and looks_like_header(t) and not is_sentence_like(t))


def is_value(s: str) -> bool:
    """A value (number, date, measurement) rather than an entity."""
    t = (s or "").strip().lstrip("'‘’\"")
    if not t:
        return True
    if _DATE.match(t) or not any(ch.isalpha() for ch in t):
        return True
    return bool(_VALUE.match(t))


def classify_columns(rows: list[list[str]]) -> list[str]:
    """Per column: ``"entity"`` or ``"value"``; a column is a value column when most cells look like values."""
    if not rows:
        return []
    kinds = []
    for c in range(max(len(r) for r in rows)):
        col = [r[c] for r in rows if c < len(r) and r[c].strip()]
        if not col:
            kinds.append("value")
            continue
        kinds.append("value" if sum(1 for v in col if is_value(v)) * 2 > len(col) else "entity")
    return kinds


def parse_html_table(text: str) -> list[list[str]]:
    """``<table>`` -> grid, expanding rowspan/colspan so every row's columns line up."""
    out: list[list[str]] = []
    carry: dict[int, list[Any]] = {}          # col -> [rows remaining, body] (rowspan carry-over)

    def _flush(cells: list[str], col: int) -> int:
        while col in carry:
            cells.append(carry[col][1])
            carry[col][0] -= 1
            if carry[col][0] <= 0:
                del carry[col]
            col += 1
        return col

    for row in _ROW.findall(text):
        cells: list[str] = []
        col = 0
        for m in _CELL_TAG.finditer(row):
            col = _flush(cells, col)
            txt = _clean(m.group(2))
            rs = cs = 1
            for kind, n in _SPAN.findall(m.group(1)):
                if kind.lower() == "row":
                    rs = max(1, int(n))
                else:
                    cs = max(1, int(n))
            for _ in range(cs):
                cells.append(txt)
                if rs > 1:
                    carry[col] = [rs - 1, txt]
                col += 1
        _flush(cells, col)
        if any(cells):
            out.append(cells)
    return out


def parse_row_dump(text: str, min_rows: int = 4) -> list[list[str]]:
    """Whitespace-separated rows with no table markup. The column count must be steady across lines."""
    lines = [ln.strip() for ln in (text or "").split("\n")]
    cand = [ln for ln in lines if len(ln.split()) >= _ROW_DUMP_MIN_COLS]
    if len(cand) < min_rows:
        return []
    # If most lines end like sentences, this is prose, not a table.
    if sum(1 for ln in cand if _SENT_END.search(ln)) * 2 > len(cand):
        return []
    rows = [ln.split() for ln in cand]
    widths = Counter(len(r) for r in rows)
    modal, n_modal = widths.most_common(1)[0]
    aligned = sum(c for w, c in widths.items() if abs(w - modal) <= 1)
    return rows if aligned * 2 > len(rows) and n_modal >= 2 else []


_PIPE_RULE = re.compile(r"^[\s|:\-]+$")


def parse_pipe_table(text: str) -> list[list[str]]:
    """Pipe-drawn (markdown-style) table -> grid. Rule lines dropped, outer pipes stripped."""
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        if "|" not in line or _PIPE_RULE.match(line):
            continue
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(rows) < 2:
        return []
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    return [r for r in rows if len(r) == width] if width >= 2 else []


def detect_header(rows: list[list[str]], kinds: list[str]) -> int | None:
    """Index of the header row: the row whose value-column cells are all text. ``<th>`` is a title, not trusted."""
    val_cols = [i for i, k in enumerate(kinds) if k == "value"]
    if not val_cols or len(rows) < 3:
        return None
    for ri in range(min(_HEADER_SCAN, len(rows) - 1)):
        r = rows[ri]
        filled = [i for i in val_cols if i < len(r) and r[i].strip()]
        nonempty = [c for c in r if c.strip()]
        # The same text in several cells is a title row.
        if not filled or len(set(nonempty)) < 2:
            continue
        if all(not is_value(r[i]) for i in filled) and all(looks_like_header(c) for c in nonempty):
            return ri
    return None


def merge_header_rows(rows: list[list[str]], kinds: list[str], hi: int,
                      max_extra: int = 2) -> tuple[list[str], int]:
    """Fold header rows following ``hi`` (value columns empty, filled cells header-shaped). Returns (names, first data row)."""
    heads = [c.strip() for c in rows[hi]]
    nxt = hi + 1
    val_cols = [i for i, k in enumerate(kinds) if k == "value"]
    if not val_cols:
        return heads, nxt
    while nxt < len(rows) - 1 and nxt - hi <= max_extra:
        r = rows[nxt]
        filled = [c for c in r if c.strip()]
        if not filled or any(i < len(r) and r[i].strip() and is_value(r[i]) for i in val_cols):
            break
        if not all(looks_like_header(c) for c in filled):
            break
        for i, c in enumerate(r):
            c = c.strip()
            if not c:
                continue
            if i >= len(heads):
                heads.extend([""] * (i + 1 - len(heads)))
            heads[i] = c if not heads[i] or heads[i] == c else f"{heads[i]} {c}"
        nxt += 1
    return heads, nxt


_MAX_UNIT = 6              # cap on a unit notation (e.g. "bn USD", "%", "bp")
_UNIT_VALUE_RATIO = 0.75   # cells below a unit row must be values at this rate


def unit_row(rows: list[list[str]], idx: int) -> bool:
    """Is this row the *units* of the numbers below it rather than data?"""
    if idx >= len(rows) - 1:
        return False
    filled = [(i, c.strip()) for i, c in enumerate(rows[idx]) if c.strip()]
    if len(filled) < 2 or any(len(c) > _MAX_UNIT for _i, c in filled):
        return False
    if any(ch.isdigit() for _i, c in filled for ch in c):
        return False
    cols = [i for i, _c in filled if i > 0]
    below = [b[i].strip() for b in rows[idx + 1:] for i in cols if i < len(b) and b[i].strip()]
    return bool(below) and sum(1 for v in below if is_value(v)) >= len(below) * _UNIT_VALUE_RATIO


def looks_like_header(cell: str) -> bool:
    """Column-name shaped: not a value, enumerator, sentence end, or a long digit-laden phrase."""
    t = (cell or "").strip()
    if not t:
        return True
    if is_value(t) or len(t) > _MAX_HEAD:
        return False
    if _ENUMERATOR.match(t) or _SENT_END.search(t):
        return False
    if t.count("(") != t.count(")") or t.count("[") != t.count("]"):
        return False                    # unbalanced brackets: a cell that was split
    toks = t.split()
    if len(toks) >= 5 and any(ch.isdigit() for ch in t):
        return False
    return True


def table_caption(text: str) -> str:
    """A short noun phrase within ``_HEADER_SCAN`` lines above ``<table>``; bracketed notes and sentences skipped."""
    i = text.find("<table")
    if i < 0:
        return ""
    seen = 0
    for line in reversed(text[:i].split("\n")):
        raw = " ".join(line.split())
        if not raw:
            continue
        if raw[0] in "([（" and raw[-1] in ")]）":
            continue
        seen += 1
        if seen > _HEADER_SCAN:
            break
        if _SENT_END.search(raw):
            continue
        t = normalize_label(raw)
        if (t and _MIN_NAME <= len(t) <= _MAX_HEAD and looks_like_header(t)
                and len(t.split()) <= 8 and not is_sentence_like(t)):
            return t
    return ""


_NOUN_TAGS = ("NNG", "NNP", "SL")   # digits (SN) are values, not names
_AFFIX_TAGS = ("XR", "XSN")         # roots/suffixes attach only after a noun


def _phrase_pieces(phrase: str) -> list[str]:
    """A noun phrase and its word pieces. Pieces give long names a place to attach to."""
    out = [phrase]
    parts = phrase.split()
    if len(parts) > 1:
        out.extend(parts)
    return out


def _acceptable(name: str) -> bool:
    n = name.strip()
    return _MIN_NAME <= len(n) <= _MAX_NAME and any(ch.isalpha() for ch in n)


def _morph_tails(name: str) -> list[str]:
    """Trailing pieces of an unspaced compound, cut at morpheme boundaries only."""
    toks = tokenize(name or "")
    if not toks:
        return []
    out = []
    for t in toks[1:]:
        tail = name[t.start:]
        if _MIN_NAME <= len(tail) < len(name):
            out.append(tail)
    return out


@lru_cache(maxsize=20000)
def _is_proper(name: str) -> bool:
    """A proper noun. An individual is not a type, so it is never promoted to a concept even when used as a head."""
    toks = tokenize(name or "")
    if not toks:
        return False
    tags = [t.tag for t in toks if t.tag not in ("SP", "SF", "SS")]
    return bool(tags) and all(t == "NNP" for t in tags)


_SHEET_MARK = re.compile(r"^\[\s*Sheet\s*:\s*(.+?)\s*\]$")
_BRACKET_MARK = re.compile(r"^[\[(（].*[\])）]$")


def _pipe_caption(text: str) -> str:
    """The title line above a pipe table (markdown heading included) is the table's name."""
    head = ""
    for line in (text or "").splitlines():
        if "|" in line:
            break
        raw = line.lstrip("#").strip()
        m = _SHEET_MARK.match(raw)
        if m:
            raw = m.group(1)
        elif _BRACKET_MARK.match(raw):
            continue
        t = normalize_label(raw)
        if t and _MIN_NAME <= len(t) <= _MAX_HEAD and not is_sentence_like(t):
            head = t
    return head


def _noun_phrases(text: str, toks=None) -> list[tuple[str, bool]]:
    """Prose noun phrases -> ``[(name, is_piece)]``."""
    src = (text or "")[:4000]
    toks = toks if toks is not None else tokenize(src)
    if not toks:
        return []
    out: list[tuple[str, bool]] = []
    start: int | None = None            # source offset where the current name starts
    end = 0
    prev_end = 0
    n_tokens = 0
    prev_tag = ""
    run_prev_tag = ""                   # tag right before the current name began

    def _flush(next_tag: str = "") -> None:
        nonlocal start, n_tokens
        # A single word attached to a quantity is an adverbial, not a name.
        if n_tokens == 1 and (next_tag == "SN" or run_prev_tag == "NNB"):
            start, n_tokens = None, 0
            return
        if start is not None:
            phrase = normalize_text(src[start:end])
            if _acceptable(phrase):
                out.append((phrase, False))
            for piece in _phrase_pieces(phrase)[1:]:
                if _acceptable(piece):
                    out.append((piece, True))
        start, n_tokens = None, 0

    affix = False
    for tok in list(toks) + [None]:
        joins = (tok is not None and tok.tag in _AFFIX_TAGS
                 and start is not None and tok.form not in _AFFIX_SKIP)
        if tok is not None and (tok.tag in _NOUN_TAGS or joins):
            gap = src[prev_end:tok.start] if start is not None else ""
            # A name ends at a line break, or at a space after an affix.
            if start is not None and ("\n" in gap or (affix and gap.strip() != gap)):
                _flush()
                if tok.tag not in _NOUN_TAGS:
                    affix = False
                    continue            # an affix at line start has nothing to attach to
            if start is None:
                start, run_prev_tag = tok.start, prev_tag
            end = tok.start + tok.len
            n_tokens += 1
            affix = tok.tag in _AFFIX_TAGS
            prev_end = end
            prev_tag = tok.tag
            continue
        if start is not None:
            # A verbalizing suffix right after a single morpheme: used as a verb, dropped.
            if tok is not None and tok.tag == "XSV" and n_tokens == 1:
                start, n_tokens = None, 0
            else:
                _flush(tok.tag if tok is not None else "")
        affix = False
        prev_tag = tok.tag if tok is not None else ""
    return out


_VALUE_TAGS = ("SN", "NR", "SW")            # numbers and symbols; one unit token may follow
_VALUE_CASE = ("JKO", "JKS", "JKC", "VCP")  # a value that is a sentence constituent takes one of these


def _prose_values(text: str, toks=None) -> list[tuple[str, str, str]]:
    """Prose ``(entity, attribute, value)`` triples."""
    sent = text or ""
    toks = toks if toks is not None else tokenize(sent)
    out: list[tuple[str, str, str]] = []
    if not toks:
        return out
    subject = ""
    first_np = ""
    last_name = ""
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.tag in ("SF", "SE"):          # sentence over: pick the subject again
            subject = first_np = last_name = ""
            i += 1
            continue
        if tok.tag in ("NNG", "NNP"):
            j = i
            while j + 1 < len(toks) and toks[j + 1].tag in ("NNG", "NNP", "XSN"):
                j += 1
            phrase = normalize_text(sent[toks[i].start:toks[j].start + toks[j].len])
            if not subject and (_is_proper(phrase) or not first_np):
                subject = phrase if _is_proper(phrase) else ""
            if not first_np:
                first_np = phrase
            last_name = phrase if len(phrase) >= _MIN_NAME else ""
            i = j + 1
            continue
        subj = subject or first_np      # no proper noun: the sentence's first name is the subject
        if tok.tag == "SN" and last_name and subj and last_name != subj:
            j = i
            while j + 1 < len(toks) and toks[j + 1].tag in _VALUE_TAGS:
                j += 1
            # At most one unit token. A common noun of 2+ characters is the next phrase, not a unit.
            if j + 1 < len(toks) and (toks[j + 1].tag == "NNB"
                                      or (toks[j + 1].tag == "NNG" and toks[j + 1].len == 1)):
                j += 1
            nxt = toks[j + 1].tag if j + 1 < len(toks) else ""
            value = normalize_text(sent[toks[i].start:toks[j].start + toks[j].len])
            if value and len(value) <= _MAX_HEAD and nxt in _VALUE_CASE:
                out.append((subj, last_name, value))
            last_name = ""
            i = j + 1
            continue
        if tok.tag.startswith("J"):
            i += 1
            continue
        last_name = ""
        i += 1
    return out


class ChunkFacts(NamedTuple):
    """What one chunk yielded. ``relations`` / ``props`` only come from tables with a header."""
    ents: list[tuple[str, str]]                 # (entity, class)
    pairs: list[tuple[str, str]]                # entities that shared a row (co-occurrence)
    relations: list[tuple[str, str, str]]       # (row entity, column name, entity cell)
    props: list[tuple[str, str, str]]           # (row entity, column name, value cell)
    heads: list[str]                            # this table's column names (carried to the next chunk)
    pieces: list[str] = []                      # word pieces of long names (attachment candidates)


def _width(rows: list[list[str]]) -> int:
    return Counter(len(r) for r in rows).most_common(1)[0][0] if rows else 0


# A leading classification code: an alphanumeric token with a digit, 3+ chars, followed by whitespace.
_CODE_HEAD = re.compile(r"^(?=[0-9A-Za-z\-.]*\d)[0-9A-Za-z][0-9A-Za-z\-.]{2,}\s+(?=\S)")
# A trailing description: what follows a separator is a gloss when it is long.
_DESC_TAIL = re.compile(r"\s[-–—:]\s.{10,}$")
# A trailing parenthetical: the name ends before it; the bracket is a cue about the name.
_PAREN_TAIL = re.compile(r"\s*[(\[（][^()\[\]（）]*[)\]）]\s*$")


def name_core(s: str, drop_paren: bool = True) -> str:
    """Keep only the name in a name cell."""
    t = (s or "").strip()
    if not t:
        return t
    cut = _DESC_TAIL.sub("", t).strip()
    cut = _CODE_HEAD.sub("", cut).strip()
    if drop_paren:
        core = _PAREN_TAIL.sub("", cut).strip()
        if _MIN_NAME <= len(core) and any(ch.isalpha() for ch in core):
            cut = core
    return cut if _MIN_NAME <= len(cut) and any(ch.isalpha() for ch in cut) else t


def _paren_distinguishes(values: list[str]) -> bool:
    """Do trailing parentheticals tell these names apart?"""
    keep, drop = set(), set()
    for v in values:
        base = name_core(normalize_label(v), False)
        if not base:
            continue
        keep.add(base)
        drop.add(name_core(normalize_label(v), True) or base)
    return len(drop) < len(keep)


def _cooccurrence(rows: list[list[str]], kinds: list[str],
                  heads: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Headerless table (row dump): entity cells and their same-row pairs only."""
    keep_paren = {
        c for c in range(max((len(r) for r in rows), default=0))
        if _paren_distinguishes([r[c] for r in rows if c < len(r)])
    }
    ents: list[tuple[str, str]] = []
    pairs: list[tuple[str, str]] = []
    for r in rows:
        names = []
        for i, cell in enumerate(r):
            if i >= len(kinds) or kinds[i] != "entity":
                continue
            v = name_core(normalize_label(cell), i not in keep_paren)
            if not v or is_value(v) or not (_MIN_NAME <= len(v) <= _MAX_NAME) or is_sentence_like(v):
                continue
            cls = heads[i].strip() if i < len(heads) else ""
            if is_value(cls) or not (0 < len(cls) <= _MAX_NAME):
                cls = ""
            names.append((v, cls))
        ents.extend(names)
        for a in range(min(len(names), _PAIR_HEAD)):
            for b in range(a + 1, min(len(names), _PAIR_TAIL)):
                if names[a][0] != names[b][0]:
                    pairs.append((names[a][0], names[b][0]))
    return ents, pairs


def _valueish_column(rows: list[list[str]], col: int) -> bool:
    """Are this column's cells values (years, figures) rather than names? More than half decides."""
    vals = [r[col].strip() for r in rows if col < len(r) and r[col].strip()]
    return bool(vals) and sum(1 for v in vals if is_value(v)) * 2 > len(vals)


# A string the dictionary cannot know, used to learn the out-of-vocabulary id.
_OOV_PROBE = "뷁" * 4


@lru_cache(maxsize=1)
def _oov_rank() -> int | None:
    """The id the analyzer gives to a word not in its dictionary."""
    toks = tokenize(_OOV_PROBE)
    if not toks or len(toks) != 1:
        return None
    rank = getattr(toks[0], "id", None)
    return rank if isinstance(rank, int) else None


@lru_cache(maxsize=20000)
def is_common_word(name: str, rank_max: int = DEFAULT_COMMON_WORD_RANK) -> bool:
    """A single common noun of everyday Korean.

    The analyzer's dictionary is numbered by corpus frequency, so a low id means
    a common word; no separate reference corpus is needed. Words *missing* from
    the dictionary all share one id and are the opposite of common, so that id
    is learned by probing (:func:`_oov_rank`) rather than hard-coded.
    """
    toks = tokenize(name or "")
    if not toks or len(toks) != 1:
        return False
    t = toks[0]
    if t.tag != "NNG" or t.form != (name or "").strip():
        return False
    rank = getattr(t, "id", None)
    if not isinstance(rank, int) or rank >= rank_max:
        return False
    oov = _oov_rank()
    return oov is not None and rank != oov


def _code_column(rows: list[list[str]], col: int) -> bool:
    """Is this column an identifier code: digits with at most one letter, in most cells?"""
    vals = [r[col].strip() for r in rows if col < len(r) and r[col].strip()]
    if not vals:
        return False
    code = sum(1 for v in vals
               if any(ch.isdigit() for ch in v) and len([c for c in v if c.isalpha()]) <= 1)
    return code * 2 > len(vals)


def _unique_ratio(rows: list[list[str]], col: int) -> float:
    """How distinct this column's values are: the subject-column criterion of table interpretation."""
    vals = [r[col].strip() for r in rows if col < len(r) and r[col].strip()]
    return (len(set(vals)) / len(vals)) if vals else 0.0


def _row_key(r: list[str], id_cols: list[int], keep_paren: set | None = None) -> str:
    """Row name from its identifying cells."""
    idc: list[str] = []
    for i in id_cols:
        v = (name_core(normalize_label(r[i]), i not in (keep_paren or ()))
             if i < len(r) else "")
        if v and v not in idc:
            idc.append(v)
    return " ".join(idc)


def _table_facts(rows: list[list[str]], kinds: list[str], heads: list[str],
                 caption: str) -> ChunkFacts:
    """Table with a header: one entity per row; (row -> column -> cell) is a property for value columns and a relation for entity columns."""
    heads = [(h if looks_like_header(h) and _MIN_NAME <= len(h) <= _MAX_HEAD else "") for h in heads]
    ent_cols = [i for i, k in enumerate(kinds) if k == "entity" and i < len(heads)]
    prefix = ""
    if ent_cols:
        # The subject column is the entity column with the most distinct values.
        first = max(ent_cols, key=lambda i: (_unique_ratio(rows, i),
                                             not _code_column(rows, i), -i))
        id_cols = [first]
        # If the identifying cells alone do not separate rows, add neighbouring entity columns.
        for i in range(first + 1, len(heads)):
            keys = [_row_key(r, id_cols) for r in rows]
            keys = [k for k in keys if k]
            if len(set(keys)) == len(keys) or kinds[i] != "entity":
                break
            id_cols.append(i)
    else:
        id_cols = [0]
        # No entity column (chart data whose first column is years): row names are values, so prefix something.
        prefix = caption or (heads[0] if heads else "")
        if not prefix:
            ents, pairs = _cooccurrence(rows, kinds, heads)
            return ChunkFacts(ents, pairs, [], [], heads)
    # What a row *is* comes from the subject column's header.
    _subject = heads[id_cols[0]] if id_cols[0] < len(heads) else ""
    if _subject and caption and _valueish_column(rows, id_cols[0]):
        _subject = ""     # only dropped when a caption can replace it
    row_class = _subject or caption

    # A parenthetical that separates a column's names is part of the name, not decoration.
    keep_paren = {
        c for c in range(len(heads))
        if _paren_distinguishes([r[c] for r in rows if c < len(r)])
    }

    ents: list[tuple[str, str]] = []
    pairs: list[tuple[str, str]] = []
    rels: list[tuple[str, str, str]] = []
    props: list[tuple[str, str, str]] = []
    # In a column that keeps parentheticals, the part before them is the shared head.
    pieces: list[str] = []
    for c in keep_paren:
        for r in rows:
            if c < len(r) and r[c].strip():
                head = name_core(normalize_label(r[c]), True)
                if head and _MIN_NAME <= len(head) <= _MAX_NAME:
                    pieces.append(head)
    for r in rows:
        key = _row_key(r, id_cols, keep_paren)
        if not key:
            continue
        label = f"{prefix} {key}" if prefix else key
        if len(label) > _MAX_ROW_LABEL:
            label = _row_key(r, id_cols[:1])
        if (not (_MIN_NAME <= len(label) <= _MAX_ROW_LABEL) or (not prefix and is_value(label))
                or is_sentence_like(label)):
            continue
        ents.append((label, row_class))
        names = [label]
        if len(id_cols) > 1:
            # Identifying cells become entities too, but are not typed by their column name.
            for i in id_cols:
                v = (name_core(normalize_label(r[i]), i not in keep_paren)
                     if i < len(r) else "")
                if v and v != label and not is_value(v) and _MIN_NAME <= len(v) <= _MAX_NAME:
                    ents.append((v, ""))
                    if heads[i]:
                        rels.append((label, heads[i], v))
                    names.append(v)
        for j, cell in enumerate(r):
            if j in id_cols or j >= len(heads) or not heads[j]:
                continue
            cell = cell.strip()
            if not cell:
                continue
            if (j < len(kinds) and kinds[j] == "value") or is_value(cell):
                props.append((label, heads[j], cell))
                continue
            # Only cells that become entities are normalized to name form; values stay verbatim.
            v = name_core(normalize_label(cell), j not in keep_paren)
            if not v or not (_MIN_NAME <= len(v) <= _MAX_NAME) or is_sentence_like(v):
                props.append((label, heads[j], cell))   # long text / sentence cells become values
                continue
            ents.append((v, heads[j]))
            rels.append((label, heads[j], v))
            names.append(v)
        for a in range(min(len(names), _PAIR_HEAD)):
            for b in range(a + 1, min(len(names), _PAIR_TAIL)):
                if names[a] != names[b]:
                    pairs.append((names[a], names[b]))
    return ChunkFacts(ents, pairs, rels, props, heads, pieces)


def prose_only(text: str, header_patterns=()) -> str:
    """The prose part of a chunk, with tables removed.

    A Hearst pattern must not fire inside a table cell: a cell is an enumeration,
    not a sentence, so the noun after the anchor word is a shared column value,
    not a hypernym (measured on a real corpus: 80% of raw pairs came from tables
    and nearly all were wrong). Strips HTML tables, pipe grids and whitespace
    row dumps, plus ingestion markers.
    """
    t = _MARKER.sub("", _META.sub("", strip_headers(text or "", header_patterns))).strip()
    if not t:
        return ""
    if "<table" in t:
        return _TABLE_BLOCK.sub(" ", t)
    if parse_pipe_table(t):
        return "\n".join(ln for ln in t.splitlines()
                         if "|" not in ln and not _BRACKET_MARK.match(ln.strip()))
    return "" if parse_row_dump(t) else t


def extract_chunk(text: str, prev_heads: list[str] | None = None,
                  header_patterns=()) -> ChunkFacts:
    """One chunk -> :class:`ChunkFacts`. ``prev_heads``: column names of the previous chunk of the same document (a table split across chunks)."""
    t = _MARKER.sub("", _META.sub("", strip_headers(text, header_patterns))).strip()
    if not t:
        return ChunkFacts([], [], [], [], [])
    is_html = "<table" in t
    pipe_rows = [] if is_html else parse_pipe_table(t)
    rows = parse_html_table(t) if is_html else (pipe_rows or parse_row_dump(t))
    if not rows:
        body = t[:4000]
        toks = tokenize(body)
        np = _noun_phrases(body, toks)
        ents = [(w, "") for w, piece in np if not piece]
        return ChunkFacts(ents, [], [],
                          _prose_values(body, toks), [],
                          [w for w, piece in np if piece])

    caption = table_caption(t) if is_html else (_pipe_caption(t) if pipe_rows else "")
    if is_html and len(rows) > 1 and len({c for c in rows[0] if c.strip()}) == 1:
        # A first row with a single text is a caption, not data.
        caption = caption or next(c for c in rows[0] if c.strip())
        rows = rows[1:]
    kinds = classify_columns(rows)
    # Row dumps have no header. A pipe table's first row is the header (the rule line guarantees it).
    hi = detect_header(rows, kinds) if is_html else (0 if pipe_rows else None)
    heads: list[str] = []
    if hi is not None:
        heads, first_data = merge_header_rows(rows, kinds, hi)
        # A unit row right after the header: attach the units to the column names, drop the row.
        if first_data < len(rows) and unit_row(rows, first_data):
            for i, c in enumerate(rows[first_data]):
                c = c.strip()
                if not c or i == 0 or i >= len(heads) or not heads[i]:
                    continue
                heads[i] = heads[i] if heads[i].endswith(c) else f"{heads[i]} {c}"
            first_data += 1
        rows = rows[first_data:]
        kinds = classify_columns(rows) or kinds
    elif is_html and prev_heads and _width(rows) == len(prev_heads):
        heads = list(prev_heads)

    if heads and any(heads):
        facts = _table_facts(rows, kinds, heads, caption)
    else:
        ents, pairs = _cooccurrence(rows, kinds, heads)
        facts = ChunkFacts(ents, pairs, [], [], [])
    # Text outside the table is read too.
    rest = ""
    if is_html:
        rest = _TABLE_BLOCK.sub(" ", t)
    elif pipe_rows:
        rest = "\n".join(ln for ln in t.splitlines()
                         if "|" not in ln and not _BRACKET_MARK.match(ln.strip()))
    if rest.strip():
        np = _noun_phrases(rest)
        facts = facts._replace(
            ents=facts.ents + [(w, "") for w, piece in np if not piece],
            pieces=list(facts.pieces) + [w for w, piece in np if piece])
    return facts


def extract_from_chunk(text: str, header_patterns=()) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """One chunk -> ``([(entity, class)], [same-row entity pairs])``."""
    f = extract_chunk(text, header_patterns=header_patterns)
    return f.ents, f.pairs


def build_facts(chunks: list[dict[str, Any]], *, min_freq: int = 1,
                max_coverage: float = 0.30, denominator: int | None = None,
                header_patterns=()) -> dict[str, Any]:
    """Chunks -> entities, classes, co-occurrence pairs and row facts.

    Names appearing in more than ``max_coverage`` of ``denominator`` chunks are
    dropped as non-discriminative (the IDF intuition; no stop-word list).
    """
    n = max(1, denominator or len(chunks))
    ent_chunks: dict[str, set] = {}
    piece_chunks: dict[str, set] = {}
    ent_class: dict[str, Counter] = {}
    pair_count: Counter = Counter()
    rel_chunks: dict[tuple[str, str, str], set] = {}
    prop_chunks: dict[tuple[str, str, str], set] = {}
    last_heads: dict[Any, list[str]] = {}          # per document: the previous table's column names
    for ch in chunks:
        cid = ch.get("chunk_id")
        doc = ch.get("_doc")
        f = extract_chunk(ch.get("chunk_text") or "", prev_heads=last_heads.get(doc),
                          header_patterns=header_patterns)
        if f.heads:
            last_heads[doc] = f.heads
        for name in f.pieces:
            piece_chunks.setdefault(name, set()).add(cid)
        for name, cls in f.ents:
            ent_chunks.setdefault(name, set()).add(cid)
            if cls:
                ent_class.setdefault(name, Counter())[cls] += 1
        for p in f.pairs:
            pair_count[tuple(sorted(p))] += 1
        for t in f.relations:
            rel_chunks.setdefault(t, set()).add(cid)
        for t in f.props:
            prop_chunks.setdefault(t, set()).add(cid)
    # Discriminativeness only means something once the corpus has some size.
    coverage_on = n >= _COVERAGE_MIN_CHUNKS
    keep = {e: cs for e, cs in ent_chunks.items()
            if len(cs) >= min_freq
            and (not coverage_on or len(cs) <= 1 or len(cs) / n <= max_coverage)}
    # The same name under different columns in several tables takes the most frequent column name.
    classes = {e: ent_class[e].most_common(1)[0][0] for e in keep if e in ent_class}
    pairs = [(a, b, c) for (a, b), c in pair_count.items() if a in keep and b in keep]
    # A fact survives with its subject (row); the object may be common.
    relations = [(t, cs) for t, cs in rel_chunks.items() if t[0] in keep]
    props = [(t, cs) for t, cs in prop_chunks.items() if t[0] in keep]
    pieces = {p: cs for p, cs in piece_chunks.items() if p not in keep}
    return {"entities": keep, "pieces": pieces, "classes": classes, "pairs": pairs, "n_chunks": n,
            "relations": relations, "props": props}


def extract_as_dicts(
    documents: dict[str, list[dict[str, Any]]],
    *,
    min_freq: int = 1,
    max_coverage: float = 0.30,
    corpus_chunks: int | None = None,
    header_patterns=(),
    common_word_rank: int = DEFAULT_COMMON_WORD_RANK,
    hearst: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict]], list[dict], list[dict]]:
    """Zero-LLM extraction as plain dicts: ``(concepts, entities_by_doc, relations, data_values)``.

    Same shape as the LLM extractor's result. ``corpus_chunks`` is the
    denominator of the discriminativeness test (the whole corpus when the
    documents passed are only a part of it). ``hearst=False`` skips the
    prose hierarchy pass.
    """
    from .taxonomy import hearst_hierarchy  # local import: taxonomy imports prose_only from here

    concepts: dict[str, Any] = {"classes": [], "object_properties": [],
                                "datatype_properties": [], "class_hierarchy": []}
    ner: dict[str, list[dict]] = {}
    seen_class: set = set()

    # Discriminativeness is a corpus property, so it is not counted per document.
    flat: list[dict[str, Any]] = []
    owner: dict[Any, str] = {}
    for doc_name, chunks in (documents or {}).items():
        chunks = list(chunks or [])
        if all(isinstance(c.get("chunk_index"), int) for c in chunks):
            chunks.sort(key=lambda c: c["chunk_index"])     # header carry-over follows document order
        for ch in chunks:
            flat.append({**ch, "_doc": doc_name})
            owner[ch.get("chunk_id")] = doc_name
    if not flat:
        return concepts, ner, [], []

    # Hypernym / hyponym sentences (Hearst) read as hierarchy, on prose only.
    _hier: list[dict] = []
    if hearst:
        try:
            _hier = [{"child": c, "parent": p}
                     for p, c in hearst_hierarchy([_c.get("chunk_text") or "" for _c in flat],
                                                  header_patterns=header_patterns)]
        except Exception:
            _hier = []
    concepts["class_hierarchy"] = _hier
    # Names in the hierarchy are classes too.
    _parent_of = {h["child"]: h["parent"] for h in _hier if h.get("child") and h.get("parent")}
    _seen_h: set = set()
    for _h in _hier:
        for _n in (_h.get("child"), _h.get("parent")):
            if _n and _n not in _seen_h:
                _seen_h.add(_n)
                concepts["classes"].append(
                    {"name": _n, "description": "", "parent": _parent_of.get(_n)})
    seen_class |= _seen_h

    res = build_facts(flat, min_freq=min_freq, max_coverage=max_coverage,
                      denominator=corpus_chunks or len(flat), header_patterns=header_patterns)
    # === Layering: concept / entity / attribute ===
    _names = set(res["entities"])
    _all = _names | set(res.get("pieces") or {})
    _children: dict[str, set] = {}
    _head_of: dict[str, str] = {}
    for _n in _names:
        _parts = _n.split()
        _tails = ([" ".join(_parts[c:]) for c in range(1, len(_parts))] if len(_parts) > 1
                  else _morph_tails(_n))
        # A name with a parenthetical cue: the head is what comes before the bracket.
        _front = _PAREN_TAIL.sub("", _n).strip()
        if _front and _front != _n:
            _tails = [_front] + _tails
        for _head in _tails:
            if _head in _all and _head != _n:
                _children.setdefault(_head, set()).add(_n)
                _head_of[_n] = _head
                break
    _head_classes = {h for h, ch in _children.items()
                     if (len(ch) >= _HEAD_MIN_CHILDREN or (h in _names and ch))
                     and not _is_proper(h)}
    _head_of = {c: h for c, h in _head_of.items() if h in _head_classes}
    # Names whose head came from before a parenthetical.
    _paren_heads = {
        _head_of[_n] for _n in _head_of
        if _PAREN_TAIL.sub("", _n).strip() == _head_of[_n]
    }
    for _h in sorted(_head_classes):
        if _h not in seen_class:
            seen_class.add(_h)
            _parent = _head_of.get(_h)
            if not _parent and _h in _paren_heads:
                # The column name the names under this head share is the head's parent.
                _cols = {res["classes"].get(_c) for _c in _children.get(_h, ())}
                _cols.discard(None)
                _cols.discard("")
                if len(_cols) == 1:
                    _parent = _cols.pop()
            concepts["classes"].append(
                {"name": _h, "description": "", "parent": _parent})

    # A single common noun that came from no table, heads no other name and sits in no
    # hierarchy is a verbal habit ("result", "environment"), not a name.
    _common_drop = {
        _n for _n in _names
        if _n not in _head_classes and _n not in _children and _n not in _head_of
        and not res["classes"].get(_n) and _n not in _parent_of
        and is_common_word(_n, common_word_rank)
    }

    for name, cids in res["entities"].items():
        if name in _head_classes:
            continue                      # already placed in the concept layer
        if name in _common_drop:
            continue
        # Class: parenthetical head > column header > Hearst hypernym > name head > none.
        _ph = _head_of.get(name) if _head_of.get(name) in _paren_heads else ""
        cls = _ph or res["classes"].get(name) or _parent_of.get(name) or _head_of.get(name) or ""
        if cls and cls not in seen_class:
            seen_class.add(cls)
            # description stays empty: a sentence not in the document is the LLM's to write
            concepts["classes"].append({"name": cls, "description": "", "parent": None})
        # source_chunks must match their document, so split per document.
        per_doc: dict[str, list[str]] = {}
        for c in cids:
            if c:
                per_doc.setdefault(owner.get(c, ""), []).append(c)
        for doc_name, chunk_ids in per_doc.items():
            ner.setdefault(doc_name, []).append(
                {"entity": name, "class": cls, "type": "INSTANCE",
                 "source_chunks": chunk_ids})

    # Row facts, in the LLM extractor's dict shape.
    relations: list[dict] = []
    data_props: list[dict] = []
    for (subj, pred, obj), cids in res.get("relations", []):
        relations.append({"subject": subj, "predicate": pred, "object": obj,
                          "predicate_type": "ObjectProperty",
                          "source_chunks": sorted(c for c in cids if c)})
    props_seen: set = set()
    for (subj, prop, val), cids in res.get("props", []):
        data_props.append({"entity": subj, "property": prop, "value": val,
                           "source_chunks": sorted(c for c in cids if c)})
        props_seen.add(prop)
    # One declaration per column name, with no domain: attaching it to a class would make
    # inheritance materialization copy it into every subclass.
    concepts["datatype_properties"] = [
        {"name": p, "domain": None, "range": "xsd:string"} for p in sorted(props_seen)]
    return concepts, ner, relations, data_props


def extract_deterministic(
    documents: dict[str, list[dict[str, Any]]],
    *,
    min_freq: int = 1,
    max_coverage: float = 0.30,
    corpus_chunks: int | None = None,
    header_patterns=(),
    common_word_rank: int = DEFAULT_COMMON_WORD_RANK,
    hearst: bool = True,
) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
    """Zero-LLM extraction into the build models. See :func:`extract_as_dicts`."""
    c, ents, rels, dvs = extract_as_dicts(
        documents, min_freq=min_freq, max_coverage=max_coverage, corpus_chunks=corpus_chunks,
        header_patterns=header_patterns, common_word_rank=common_word_rank, hearst=hearst)
    concepts = Concepts(
        classes=[Class(name=x["name"], description=x.get("description", ""), parent=x.get("parent"))
                 for x in c["classes"]],
        datatype_properties=[DataProperty(name=x["name"], domain=x.get("domain") or "",
                                          range=x.get("range", "xsd:string"))
                             for x in c["datatype_properties"]],
        class_hierarchy=[(h["parent"], h["child"]) for h in c["class_hierarchy"]],
    )
    instances = [Instance(name=e["entity"], class_name=e.get("class") or "",
                          source_chunks=list(e.get("source_chunks") or []))
                 for doc_ents in ents.values() for e in doc_ents]
    relations = [Relation(subject=r["subject"], predicate=r["predicate"], object=r["object"],
                          source_chunks=list(r.get("source_chunks") or []))
                 for r in rels]
    data_values = [DataValue(entity=d["entity"], property=d["property"], value=d["value"],
                             source_chunks=list(d.get("source_chunks") or []))
                   for d in dvs]
    return concepts, instances, relations, data_values
