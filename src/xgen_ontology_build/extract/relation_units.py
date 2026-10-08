"""The deterministic layer of relation formation: units of evidence and the mentions in them.

A document is cut into units (a sentence or clause of prose, a row of a table) and every
mention of a graph node inside a unit is located. A model then only names the relation
between two numbered mentions of the same unit, so a relation always has a place in the
text that states it, and the number of units and mentions (hence calls and tokens) is
known before any model is called. Ported from the production ``relation_units``; the
ingestion header the product strips is a ``header_patterns`` argument here.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from xgen_ontology_build.extract.deterministic import (
    _BRACKET_MARK,
    _MARKER,
    _META,
    _TABLE_BLOCK,
    _pipe_caption,
    _width,
    classify_columns,
    detect_header,
    is_value,
    merge_header_rows,
    parse_html_table,
    parse_pipe_table,
    parse_row_dump,
    strip_headers,
    table_caption,
    unit_row,
)

# A unit of prose ends at a sentence end, a line break, or before a circled clause number.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+|(?=[①-⑳㉠-㉿])")
_NUMBER_HEAD = re.compile(r"^[+\-]?[\d,.]+")
_TOKEN_EDGE = " \t\r\n|"


class Unit(NamedTuple):
    """One unit of evidence: a row of a table or a piece of prose, with the nodes it mentions."""

    chunk_id: str
    doc: str
    caption: str                        # the table's title (context for the model only)
    row: bool                           # a table row (else a piece of prose)
    text: str
    mentions: list[str]                 # node labels, in order of appearance
    spans: list[tuple[int, int]]        # their places in ``text``
    heads: dict[str, str]               # for a row: mention -> its column name

    def prompt_text(self) -> str:
        return f"[{self.caption}] {self.text}" if self.caption else self.text


class _Cell(NamedTuple):
    hstart: int                         # where the column name starts (``start`` when none)
    start: int
    end: int
    text: str
    head: str
    value: bool


def norm_key(s: str) -> str:
    """Matching key of a label: NFKC, lower case, no white space."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", s or "").lower() if not ch.isspace())


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _clean(text: str, header_patterns=()) -> str:
    return _MARKER.sub("", _META.sub("", strip_headers(text or "", header_patterns))).strip()


def _rows_with_heads(block: str, prev_heads: list[str] | None, pipe: bool = False) -> tuple[list[list[str]], list[str]]:
    """A table grid -> (data rows, column names). Without a header row, the previous chunk's column names carry over."""
    rows = parse_pipe_table(block) if pipe else parse_html_table(block)
    if not rows:
        return [], []
    if not pipe and len(rows) > 1 and len({c for c in rows[0] if c.strip()}) == 1:
        rows = rows[1:]
    kinds = classify_columns(rows)
    hi = 0 if pipe else detect_header(rows, kinds)
    heads: list[str] = []
    if hi is not None:
        heads, first = merge_header_rows(rows, kinds, hi)
        if first < len(rows) and unit_row(rows, first):
            first += 1
        rows = rows[first:]
    elif prev_heads and _width(rows) == len(prev_heads):
        heads = list(prev_heads)
    return rows, heads


def _row_text(cells: Sequence[str], heads: Sequence[str]) -> tuple[str, list[_Cell]]:
    parts: list[str] = []
    meta: list[_Cell] = []
    at = 0
    for i, c in enumerate(cells):
        c = " ".join(_nfkc(c).split())
        if not c:
            continue
        h = " ".join(_nfkc(heads[i]).split()) if i < len(heads) and heads[i] else ""
        part = f"{h}: {c}" if h else c
        if parts:
            at += 3
        start = at + (len(h) + 2 if h else 0)
        meta.append(_Cell(at, start, start + len(c), c, h, is_value(c)))
        parts.append(part)
        at += len(part)
    return " | ".join(parts), meta


def split_units(text: str, prev_heads: list[str] | None = None,
                header_patterns=()) -> tuple[list[tuple[str, str, list[_Cell]]], list[str]]:
    """A chunk -> ``[(table title, unit text, cells)]`` and the column names of its last table.

    A table gives one unit per row, prose one per sentence or clause. Only rows have cells."""
    t = _clean(text, header_patterns)
    if not t:
        return [], []
    out: list[tuple[str, str, list[_Cell]]] = []
    last_heads: list[str] = []
    rest = t
    if "<table" in t:
        caption = table_caption(t)
        for block in _TABLE_BLOCK.findall(t):
            rows, heads = _rows_with_heads(block, prev_heads)
            if heads:
                last_heads = heads
            for cells in rows:
                txt, meta = _row_text(cells, heads)
                if txt:
                    out.append((caption, txt, meta))
        rest = _TABLE_BLOCK.sub(" ", t)
    elif parse_pipe_table(t):
        rows, heads = _rows_with_heads(t, prev_heads, pipe=True)
        last_heads = heads or last_heads
        caption = _pipe_caption(t)
        for cells in rows:
            txt, meta = _row_text(cells, heads)
            if txt:
                out.append((caption, txt, meta))
        rest = "\n".join(ln for ln in t.splitlines() if "|" not in ln and not _BRACKET_MARK.match(ln.strip()))
    elif parse_row_dump(t):
        for cells in parse_row_dump(t):
            txt, meta = _row_text(cells, [])
            if txt:
                out.append(("", txt, meta))
        rest = ""
    for s in _SENT_SPLIT.split(rest):
        s = " ".join(_nfkc(s).split())
        if s:
            out.append(("", s, []))
    return out, last_heads


def _in_number_token(text: str, start: int, end: int) -> bool:
    """Is the mention the tail of a number ("36개월이하", "0.1억이상")?"""
    a = start
    while a > 0 and text[a - 1] not in _TOKEN_EDGE:
        a -= 1
    b = end
    while b < len(text) and text[b] not in _TOKEN_EDGE:
        b += 1
    return bool(_NUMBER_HEAD.match(text[a:b]))


def find_mentions(text: str, labels: Iterable[str],
                  blocked: Sequence[tuple[int, int]] = ()) -> tuple[list[str], list[tuple[int, int]]]:
    """The node labels mentioned in a unit's text, in order. A longer label claims its place
    first, a label counts once, and white space is ignored. A mention inside a value cell or
    a column name (``blocked``), or glued to a number, is not an entity mention."""
    text = _nfkc(text)
    pos: list[int] = []
    chars: list[str] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch.lower())
        pos.append(i)
    hay = "".join(chars)
    if not hay:
        return [], []
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, str, tuple[int, int]]] = []
    for label in sorted({lb for lb in labels if lb and any(ch.isalpha() for ch in lb)},
                        key=lambda lb: (-len(norm_key(lb)), lb)):
        key = norm_key(label)
        if not key:
            continue
        start = 0
        first = True
        while True:
            at = hay.find(key, start)
            if at < 0:
                break
            end = at + len(key)
            start = end
            if any(at < b and end > a for a, b in claimed):
                continue
            claimed.append((at, end))
            span = (pos[at], pos[end - 1] + 1)
            if any(span[0] >= a and span[1] <= b for a, b in blocked) or _in_number_token(text, *span):
                continue
            if first:
                found.append((at, label, span))
                first = False
    found.sort(key=lambda f: f[0])
    return [f[1] for f in found], [f[2] for f in found]


def build_units(chunk_id: str, text: str, labels: Iterable[str], prev_heads: list[str] | None = None,
                doc: str = "", header_patterns=()) -> tuple[list[Unit], list[str]]:
    """One chunk -> its units with at least two mentions, and the column names to carry to the next chunk."""
    labels = list(labels or [])
    pieces, heads = split_units(text, prev_heads, header_patterns)
    units: list[Unit] = []
    if not labels:
        return units, heads
    for caption, txt, meta in pieces:
        blocked = [(c.start, c.end) for c in meta if c.value] + [(c.hstart, c.start) for c in meta if c.head]
        mentions, spans = find_mentions(txt, labels, blocked)
        if len(mentions) < 2:
            continue
        head_of: dict[str, str] = {}
        for m, (s0, s1) in zip(mentions, spans):
            for c in meta:
                if c.head and s0 >= c.start and s1 <= c.end:
                    head_of[m] = c.head
                    break
        units.append(Unit(str(chunk_id), doc, caption, bool(meta), txt, mentions, spans, head_of))
    return units, heads


def units_for_documents(documents: dict[str, list[dict]], labels_by_chunk: dict[str, Sequence[str]],
                        header_patterns=()) -> list[Unit]:
    """Documents -> units. A document's chunks are read in order so column names carry over."""
    out: list[Unit] = []
    for doc, chunks in (documents or {}).items():
        heads: list[str] = []
        for c in sorted(chunks or [], key=lambda c: c.get("chunk_index", 0)):
            cid = str(c.get("chunk_id") or "")
            if not cid:
                continue
            units, heads = build_units(cid, c.get("chunk_text") or "", labels_by_chunk.get(cid, ()), heads,
                                       doc=str(doc), header_patterns=header_patterns)
            out.extend(units)
    return out


def relation_label(unit: Unit, i: int, j: int) -> str | None:
    """How the document names the relation between mentions ``i`` and ``j``: in a table row, the
    object cell's column name. Prose gets none (the predicate is not between the two mentions)."""
    if not unit.row:
        return None
    return unit.heads.get(unit.mentions[j]) or unit.heads.get(unit.mentions[i]) or None
