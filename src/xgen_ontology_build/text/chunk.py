"""Boundary-aware text chunking — pure Python, zero deps.

Pack a document into ~``max_chars`` windows, preferring natural boundaries
(paragraphs, then sentences), with a small character overlap so a fact split
across a boundary still lands whole in at least one chunk. Each chunk keeps a
stable ``chunk_id`` so build provenance (``source_chunks``) and search passages
line up.
"""
from __future__ import annotations

import re

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n")


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping, boundary-aware chunks of ~``max_chars``."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    units = _split_units(text, max_chars)
    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > max_chars:
            chunks.append(cur)
            cur = (_tail(cur, overlap) + "\n" + u) if overlap else u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(name: str, text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[dict]:
    """Chunk a document into ``[{chunk_id, chunk_text, chunk_index}, ...]``."""
    return [
        {"chunk_id": f"{name}#{i}", "chunk_text": c, "chunk_index": i}
        for i, c in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap))
    ]


def _split_units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, then over-long paragraphs into sentences, then hard char windows."""
    units: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _SENT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                units.extend(sent[i:i + max_chars] for i in range(0, len(sent), max_chars))
    return units


def _tail(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s if len(s) <= n else ""
    cut = s[-n:]
    sp = cut.find(" ")
    return cut[sp + 1:] if 0 <= sp < n // 2 else cut
