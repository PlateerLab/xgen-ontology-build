"""One-call helpers — the friendly entry points.

* ``build_from_documents(docs, llm)`` — LLM extraction + cleaning -> Ontology.
* ``build_from_csv(tables)`` — deterministic table -> Ontology (no LLM).
* ``build_from_triples(triples)`` — searchable ontology from loose (s, p, o).
"""
from __future__ import annotations

import csv as _csv
import io
import os

from .build.parse import load_documents
from .build.pipeline import OntologyBuilder
from .ontology import Ontology


def build_from_documents(documents, llm=None, *, morphology=None, embedder=None,
                         domain: str = "", dedup: bool = True, scs: bool = False,
                         hierarchy: bool = True, chunk: bool = True,
                         chunk_size: int = 1200, chunk_overlap: int = 150) -> Ontology:
    """Build an ontology from text (and/or table) documents.

    ``documents`` = ``{name: text}`` or ``{name: [chunk, ...]}``. Pass an ``llm`` to
    extract from prose; table files (``.csv``/``.tsv``/``.xlsx``) build with no LLM.
    Raw prose strings are auto-chunked (boundary-aware) unless ``chunk=False``.
    ``hierarchy`` (default on, zero LLM calls) induces is-a edges from Hearst
    patterns in the text and from head-noun decomposition of class names --
    set ``False`` to keep the flat, headers/LLM-only schema."""
    return OntologyBuilder(llm, morphology=morphology, embedder=embedder, domain=domain,
                           dedup=dedup, scs=scs, hierarchy=hierarchy, chunk=chunk,
                           chunk_size=chunk_size, chunk_overlap=chunk_overlap).build(documents)


def build_from_text(text: str, *, name: str = "document.txt", llm=None, **kwargs) -> Ontology:
    """Build from a single raw document string (parsed already to text). It is chunked,
    extracted (if ``llm`` given) and cleaned end-to-end."""
    return build_from_documents({name: text}, llm=llm, **kwargs)


def build_from_files(paths: list[str], llm=None, **kwargs) -> Ontology:
    """Parse files (txt/md/html/csv built-in; pdf/docx/xlsx via the ``[files]`` extra),
    then build end-to-end. Table files route to the no-LLM tabular path automatically."""
    return build_from_documents(load_documents(paths), llm=llm, **kwargs)


def build_from_csv(tables: dict[str, str], *, embedder=None, dedup: bool = True,
                   hierarchy: bool = True) -> Ontology:
    """Build deterministically from CSV content. ``tables`` maps a (file) name to its
    CSV text; names without a table extension get ``.csv`` appended."""
    docs = {}
    for name, content in tables.items():
        key = name if _has_table_ext(name) else f"{name}.csv"
        docs[key] = content
    return OntologyBuilder(None, embedder=embedder, dedup=dedup, hierarchy=hierarchy).build(docs)


def build_from_csv_files(paths: list[str], *, embedder=None, dedup: bool = True) -> Ontology:
    """Build from CSV/TSV files on disk."""
    tables = {}
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            tables[os.path.basename(p)] = f.read()
    return build_from_csv(tables, embedder=embedder, dedup=dedup)


def build_from_triples(triples, chunks=None) -> Ontology:
    """Searchable ontology from loose ``(s, p, o)`` triples (tuples / dicts) — no LLM."""
    return Ontology.from_triples(triples, chunks=chunks)


def rows_to_csv(rows: list[dict], *, columns: list[str] | None = None) -> str:
    """Helper: turn a list of row dicts into CSV text for :func:`build_from_csv`."""
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _has_table_ext(name: str) -> bool:
    i = name.rfind(".")
    return i >= 0 and name[i:].lower() in {".csv", ".tsv", ".xlsx", ".xls"}
