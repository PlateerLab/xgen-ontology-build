"""Deterministic table -> ontology (no LLM).

Two stages, both domain-general:

1. ``analyze_tables`` — infer a relational schema from CSV/TSV/Markdown/HTML table
   chunks: columns, xsd column types, primary-key candidates, and foreign-key
   relations (same-name, normalized-name, *and* value-overlap detection).
2. ``build_from_tables`` — turn that schema into an ontology by the star-schema
   rule: table -> Class, FK -> ObjectProperty, column -> DataProperty, dimension
   rows -> instances. Fact / event tables (FK-source-only or junctions, when large)
   are kept as schema only (their rows belong in a SQL store, not the graph).

``build_from_rows`` is the same build for the rows of one database table (a SELECT
result): the schema is declared by the caller (primary key, label column, column
types, foreign keys) instead of inferred, every row carries its own source id, and a
value's type comes from its Python type.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from xgen_ontology_build.extract.deterministic import (
    _CELL_TAG,
    _META,
    _ROW,
    _SPAN,
    _clean,
    is_value,
    parse_pipe_table,
    strip_headers,
)
from xgen_ontology_build.models import (
    Class,
    Concepts,
    DataProperty,
    DataValue,
    Instance,
    ObjectProperty,
    Relation,
)
from xgen_ontology_build.text.korean import normalize_text
from xgen_ontology_build.text.tokens import safe_uri

TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls"}
_REF_TABLE_MAX_ROWS = 200  # at/below this a table is treated as a dimension (instantiated)

# A database value's Python type -> xsd type; subclasses first (bool < int, datetime < date).
_PY_TO_XSD: tuple[tuple[type, str], ...] = (
    (bool, "xsd:boolean"),
    (int, "xsd:integer"),
    (Decimal, "xsd:decimal"),
    (float, "xsd:decimal"),
    (datetime, "xsd:dateTime"),
    (date, "xsd:date"),
)


def xsd_type_of(value: Any) -> str:
    """The xsd type of a Python value (``xsd:string`` for anything else)."""
    for py_type, xsd in _PY_TO_XSD:
        if isinstance(value, py_type):
            return xsd
    return "xsd:string"


def value_text(value: Any) -> str:
    """A database value as a graph literal: ``None`` is empty (not recorded), booleans are
    ``true`` / ``false``, dates are ISO 8601, bytes are decoded, JSON columns are JSON."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)


def _infer_types_from_values(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, str]:
    """A column's xsd type from the Python type of its first recorded value."""
    types: dict[str, str] = {}
    for col in columns:
        for row in rows:
            v = row.get(col)
            if v is not None and v != "":
                types[col] = xsd_type_of(v)
                break
        else:
            types[col] = "xsd:string"
    return types


# ───────────────────────── schema inference ─────────────────────────


def analyze_tables(documents: dict[str, list[dict]], *, header_patterns=()) -> dict[str, Any]:
    """Infer table schemas from chunked table documents.

    ``documents`` = ``{file_name: [{"chunk_id","chunk_text","chunk_index"}, ...]}``.
    A file with several sheets (chunks led by ``[Sheet: name]``) is one table per
    sheet, keyed by :func:`split_sheets`; the sheet name is the table name (the file
    name is prefixed when two files share a sheet name). ``header_patterns`` strips
    ingestion headers before the sheet marker is read.
    Returns ``{"tables": {...}, "fk_relations": [...], "is_table_collection": bool}``.
    """
    tables: dict[str, dict] = {}
    table_file_count = 0

    for file_name, file_chunks in documents.items():
        if _ext(file_name) not in TABLE_EXTENSIONS:
            continue
        table_file_count += 1
        for key, chunks in split_sheets(file_name, file_chunks, header_patterns).items():
            header, sample_rows, total_rows = _header_and_samples(chunks)
            if not header:
                continue
            stem = _table_name(file_name)
            name = sheet_of(key) or stem
            if any(t["table_name"] == name for t in tables.values()):
                name = f"{stem}_{name}"   # "Sheet1" of another file must not become the same class
            tables[key] = {
                "table_name": name,
                "file_name": file_name,
                "columns": header,
                "column_types": _infer_column_types(header, sample_rows),
                "pk_candidates": _pk_candidates(header, sample_rows),
                "sample_values": _sample_values(header, sample_rows),
                "row_count_estimate": total_rows,
            }

    return {
        "tables": tables,
        "fk_relations": _fk_relations(tables),
        "is_table_collection": table_file_count > 0 and table_file_count >= len(documents) * 0.5,
    }


def build_from_tables(
    schema: dict[str, Any],
    documents: dict[str, list[dict]],
    *,
    header_patterns=(),
) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
    """Schema + rows -> ontology. LLM-free, deterministic."""
    tables = schema.get("tables", {})
    fk_relations = schema.get("fk_relations", [])
    if not tables:
        return Concepts(), [], [], []
    # One table per sheet, keyed the way analyze_tables keyed them.
    documents = {key: chunks for file_name, file_chunks in documents.items()
                 for key, chunks in split_sheets(file_name, file_chunks, header_patterns).items()}
    table_rows: dict[str, list[dict[str, str]]] = {}
    row_sources: dict[str, list[list[str]]] = {}
    for fn, chunks in documents.items():
        t = tables.get(fn)
        if t:
            table_rows[fn], row_sources[fn] = _rows_from_chunks(chunks, t.get("columns", []))
    return _build_graph(tables, fk_relations, table_rows, row_sources)


def build_from_rows(
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    column_types: dict[str, str] | None = None,
    pk_candidates: list[str] | None = None,
    fk_relations: list[dict] | None = None,
    label_column: str | None = None,
) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
    """The rows of one database table (a SELECT result) -> ontology, no LLM.

    The schema is declared, not inferred: ``pk_candidates`` (the first is the key),
    ``label_column`` (the column that names a row; judged from the data when absent),
    ``column_types`` (xsd types; else the Python type of each column's first recorded value)
    and ``fk_relations`` (``{"from_column", "to_table", "to_column", "to_pk_column"}``, each
    checked by :func:`normalize_fk_relations`). Values are rendered by :func:`value_text`.
    Every row is its own source: ``"{source_id}:{pk}"`` (``"{source_id}:row{i}"`` without a
    key), so a row can later be retracted on its own. A single table is never a fact table
    here, and a foreign-key cell that is SQL NULL makes no relation while an empty string
    still points at ``"{to_table}_"``, as the production loader does.
    """
    columns = [str(c) for c in columns]
    if not columns or not rows:
        return Concepts(), [], [], []
    text_rows = [{c: value_text(r.get(c)) for c in columns} for r in rows]
    null_cells = {(table_name, i, c) for i, r in enumerate(rows) for c in columns if r.get(c) is None}
    types = dict(column_types) if column_types else _infer_types_from_values(columns, rows)
    pks = [str(p) for p in (pk_candidates or []) if p]
    table = {"table_name": table_name, "columns": columns, "column_types": types, "pk_candidates": pks,
             "label_column": label_column, "row_count_estimate": len(text_rows)}
    sources = {table_name: row_source_ids(source_id, text_rows, pks[0] if pks else None)}
    return _build_graph({table_name: table}, normalize_fk_relations(fk_relations, table_name, columns),
                        {table_name: text_rows}, sources, skip_fact_tables=False, null_cells=null_cells,
                        identity_source_id=source_id)


def normalize_fk_relations(fk_relations: list[dict] | None, table_name: str, columns: list[str]) -> list[dict]:
    """Check the foreign keys declared for one table's rows: each names ``from_column`` (a column
    of the rows), ``to_table``, ``to_column`` and ``to_pk_column`` (the same column: a key
    points at the target's primary key); ``from_table`` defaults to the table."""
    out: list[dict] = []
    for i, fk in enumerate(fk_relations or []):
        rel = {**fk}
        rel.setdefault("from_table", table_name)
        missing = [k for k in ("from_column", "to_table", "to_column", "to_pk_column") if not rel.get(k)]
        if missing:
            raise ValueError(f"fk_relations[{i}] lacks {', '.join(missing)}")
        if rel["from_table"] != table_name:
            raise ValueError(f"fk_relations[{i}].from_table must be the loaded table {table_name!r}")
        if rel["from_column"] not in columns:
            raise ValueError(f"fk_relations[{i}].from_column {rel['from_column']!r} is not a column of the rows")
        if rel["to_column"] != rel["to_pk_column"]:
            raise ValueError(f"fk_relations[{i}].to_column must be the target's primary key column (to_pk_column)")
        out.append(rel)
    return out


def row_source_ids(source_id: str, rows: list[dict[str, str]], pk_col: str | None) -> list[list[str]]:
    """One stable source id per row: the key's value, or the row number when there is no key."""
    if pk_col is None:
        return [[f"{source_id}:row{i}"] for i in range(len(rows))]
    return [[f"{source_id}:{row.get(pk_col, '')}"] for row in rows]


def row_chunks(table_name: str, columns: list[str], rows: list[dict[str, Any]], *, source_id: str,
               pk_candidates: list[str] | None = None) -> list[dict]:
    """The rows as chunks for the build's bookkeeping: each row under its source id
    (:func:`row_source_ids`, the ids :func:`build_from_rows` gives) with its ``column: value``
    lines as text, so a row is searched and retracted like any chunk."""
    del table_name
    columns = [str(c) for c in columns]
    text_rows = [{c: value_text(r.get(c)) for c in columns} for r in rows]
    pks = [str(p) for p in (pk_candidates or []) if p]
    ids = row_source_ids(source_id, text_rows, pks[0] if pks else None)
    return [{"chunk_id": ids[i][0], "chunk_text": "\n".join(f"{c}: {v}" for c, v in row.items() if v),
             "chunk_index": i} for i, row in enumerate(text_rows)]


def _build_graph(
    tables: dict[str, dict[str, Any]],
    fk_relations: list[dict],
    table_rows: dict[str, list[dict[str, str]]],
    row_sources: dict[str, list[list[str]]],
    *,
    skip_fact_tables: bool = True,
    null_cells: set | None = None,
    identity_source_id: str | None = None,
) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
    """Table schemas + their text rows (with each row's source ids) -> ontology.

    ``identity_source_id`` (database rows) gives every keyed row an identity
    (:func:`row_identity_key`) and a foreign key that points at a row this load does not hold a
    placeholder individual with the target row's identity, named ``"{table}_{key}"``: when that
    row is loaded it is the same individual and takes its own name.
    """
    table_class = {t.get("table_name", fn): _camel(t.get("table_name", fn)) for fn, t in tables.items()}

    classes = [
        Class(name=table_class[t.get("table_name", fn)], source="table",
              description=f"{t.get('table_name', fn)} table ({t.get('row_count_estimate', 0)} rows)")
        for fn, t in tables.items()
    ]

    object_properties: list[ObjectProperty] = []
    seen_props: set = set()
    for fk in fk_relations:
        from_cls = table_class.get(fk["from_table"], _camel(fk["from_table"]))
        to_cls = table_class.get(fk["to_table"], _camel(fk["to_table"]))
        key = (from_cls, fk["from_column"], to_cls)
        if key in seen_props:
            continue
        seen_props.add(key)
        object_properties.append(ObjectProperty(name=f"{from_cls}_{fk['from_column']}",
                                                domain=from_cls, range=to_cls))

    fk_cols_by_table: dict[str, set] = defaultdict(set)
    for fk in fk_relations:
        fk_cols_by_table[fk["from_table"]].add(fk["from_column"])

    datatype_properties: list[DataProperty] = []
    for fn, t in tables.items():
        raw = t.get("table_name", fn)
        cls_name = table_class[raw]
        for col in t.get("columns", []):
            if col in fk_cols_by_table.get(raw, set()):
                continue
            datatype_properties.append(DataProperty(
                name=col, display_name=col, domain=cls_name,
                range=t.get("column_types", {}).get(col, "xsd:string")))

    concepts = Concepts(classes=classes, object_properties=object_properties,
                        datatype_properties=datatype_properties)

    instances: list[Instance] = []
    relations: list[Relation] = []
    data_values: list[DataValue] = []

    fk_index: dict[str, list[tuple]] = defaultdict(list)
    for fk in fk_relations:
        fk_index[fk["from_table"]].append((fk["from_column"], fk["to_table"], fk["to_column"]))

    table_pk: dict[str, str] = {}
    for fn, t in tables.items():
        pk = t.get("pk_candidates", [])
        if pk:
            table_pk[t["table_name"]] = pk[0]

    # star-schema fact/dimension split (structure, not a magic row cap)
    fk_targets = {fk["to_table"] for fk in fk_relations}
    fk_source_only = {fk["from_table"] for fk in fk_relations} - fk_targets
    fk_col_count: dict[str, int] = defaultdict(int)
    for fk in fk_relations:
        fk_col_count[fk["from_table"]] += 1

    def _is_fact(raw: str, n_rows: int) -> bool:
        if not skip_fact_tables or n_rows <= _REF_TABLE_MAX_ROWS:
            return False
        return raw in fk_source_only or fk_col_count.get(raw, 0) >= 2

    # Pass 1: rows, one instance label per row (duplicates split by identity), PK -> label.
    # A table with no name column gets no instances: naming rows by their PK turns numbers
    # into entities. Fact tables get none either.
    table_labels: dict[str, list[str]] = {}
    pk_lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for fn, rows in table_rows.items():
        t = tables.get(fn)
        if not t:
            continue
        raw = t.get("table_name", fn)
        if _is_fact(raw, len(rows)):
            continue
        label_col = _label_col(t, rows)
        if not label_col:
            continue
        pk_col = table_pk.get(raw)
        labels = _instance_labels(rows, label_col, pk_col, raw)
        table_labels[fn] = labels
        if not pk_col:
            continue
        for i, row in enumerate(rows):
            pk_val = row.get(pk_col, "")
            if not pk_val.strip():
                continue
            pk_lookup[raw][pk_val] = labels[i]                 # the raw key is the identity
            pk_lookup[raw].setdefault(pk_val.strip(), labels[i])

    # Pass 2: instances + data values + FK relations (resolved through the lookup).
    for fn, rows in table_rows.items():
        t = tables.get(fn)
        if not t or fn not in table_labels:
            continue
        raw = t.get("table_name", fn)
        cls_name = table_class[raw]
        pk_col = table_pk.get(raw)
        label_col = _label_col(t, rows)
        fk_cols = {fc for fc, _, _ in fk_index.get(raw, [])}
        labels = table_labels[fn]
        src = row_sources.get(fn, [])
        placeholders: dict[str, Instance] = {}
        for i, row in enumerate(rows):
            name = labels[i]
            chunk_ids = src[i] if i < len(src) else []
            identity = ""
            if identity_source_id and pk_col and row.get(pk_col, "").strip():
                identity = row_identity_key(identity_source_id, pk_col, row.get(pk_col, ""))
            instances.append(Instance(name=name, class_name=cls_name, source_chunks=list(chunk_ids),
                                      identity=identity))

            for col, val in row.items():
                if not val or not val.strip() or col in (pk_col, label_col) or col in fk_cols:
                    continue
                data_values.append(DataValue(
                    entity=name, property=col, value=val.strip(),
                    value_type=t.get("column_types", {}).get(col, "xsd:string"),
                    source_chunks=list(chunk_ids)))

            for from_col, to_table, _to_col in fk_index.get(raw, []):
                fk_raw = row.get(from_col, "")
                fk_val = fk_raw.strip()
                # Rows from a database say which cells were NULL: only those make no relation.
                # Rows from text have no NULL, so a blank cell is one.
                if null_cells is not None:
                    if (fn, i, from_col) in null_cells:
                        continue
                elif not fk_val:
                    continue
                to_cls = table_class.get(to_table, _camel(to_table))
                prop = f"{cls_name}_{from_col}"
                lookup = pk_lookup.get(to_table, {})
                target = lookup.get(fk_raw) or lookup.get(fk_val)
                if target is None:
                    target = f"{to_table}_{fk_val}"
                    # the row the key points at is not in this load: a placeholder with its identity
                    target_src = related_table_source_id(identity_source_id, raw, to_table) if identity_source_id else None
                    if target_src and target not in placeholders:
                        placeholders[target] = Instance(name=target, identity=row_identity_key(target_src, _to_col, fk_raw))
                relations.append(Relation(subject=name, predicate=prop, object=target,
                                          predicate_type="ObjectProperty", source_chunks=list(chunk_ids)))
        instances.extend(placeholders.values())

    return concepts, instances, relations, data_values


# ───────────────────────── helpers ─────────────────────────


def _ext(file_name: str) -> str:
    i = file_name.rfind(".")
    return file_name[i:].lower() if i >= 0 else ""


def _table_name(file_name: str) -> str:
    name = file_name.rsplit("/", 1)[-1]
    i = name.rfind(".")
    return name[:i] if i > 0 else name


def _camel(name: str) -> str:
    if not name:
        return name
    return "".join(p.capitalize() for p in name.split("_") if p)


def row_identity_key(table_source_id: str, key_column: str, key_value: str) -> str:
    """A database row's identity, the production loader's key: a hash of the table's source id,
    its key column and the row's key value (as text). Stable across loads, the same whatever the
    row is named, and the local part of the row's URI in a store."""
    payload = json.dumps(["xgen-db-row-v1", table_source_id, key_column, key_value],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "dbrow_" + hashlib.sha256(payload).hexdigest()


def related_table_source_id(source_id: str, from_table: str, to_table: str) -> str | None:
    """The source id of the table a foreign key points at, from the pointing table's: the same
    prefix with the table name swapped (``"db1:orders"`` -> ``"db1:customers"``). None when the
    source id does not end in the table name, so no identity can be derived."""
    suffix = f":{from_table}"
    if not source_id.endswith(suffix):
        return None
    return f"{source_id[:-len(suffix)]}:{to_table}"


_MAX_LABEL_CHARS = 40       # names are short
_LABEL_MIN_TEXT = 0.7       # share of values that look like names (not numbers/dates)
_LABEL_MIN_DISTINCT = 0.6   # share of distinct values; a repeated category (region, dept) is not a name


def _is_code_value(v: str) -> bool:
    """An identifier code rather than a name: digits with at most a one-letter prefix."""
    t = (v or "").strip()
    if not t or not any(ch.isdigit() for ch in t):
        return False
    return len([ch for ch in t if ch.isalpha()]) <= 1


def _label_col(table: dict, rows: list[dict[str, str]] | None = None) -> str:
    """The column that names a row, judged from the data.

    A declared ``label_column`` wins. Otherwise the first column whose values are
    mostly text, mostly distinct and not identifier codes; a code-like column is
    kept only as a fallback. Empty when no column names the rows.
    """
    declared = table.get("label_column")
    if declared and declared in table.get("columns", []):
        return declared

    def _ok(col: str) -> list[str] | None:
        vals = [str(r.get(col, "")).strip() for r in (rows or [])]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            return None
        text = sum(1 for v in vals if not is_value(v) and len(v) <= _MAX_LABEL_CHARS)
        if text / len(vals) < _LABEL_MIN_TEXT or len(set(vals)) / len(vals) < _LABEL_MIN_DISTINCT:
            return None
        return vals

    fallback = ""
    for col in table.get("columns", []):
        vals = _ok(col)
        if vals is None:
            continue
        if not fallback:
            fallback = col
        if sum(1 for v in vals if _is_code_value(v)) * 2 <= len(vals):
            return col
    return fallback


def _instance_labels(rows: list[dict[str, str]], label_col: str, pk_col: str | None, raw: str) -> list[str]:
    """One label per row; rows whose labels would collide as IRIs are told apart by their PK (or index)."""
    base: list[str] = []
    for idx, row in enumerate(rows):
        if label_col and row.get(label_col, "").strip():
            base.append(row[label_col].strip())
        elif pk_col and row.get(pk_col, "").strip():
            base.append(row[pk_col].strip())
        else:
            base.append(f"{raw}_{idx}")
    dup = {u for u, n in Counter(safe_uri(b) for b in base).items() if n > 1}
    if not dup:
        return base
    taken = {safe_uri(b) for b in base}
    next_n: dict[str, int] = defaultdict(int)
    out: list[str] = []
    for idx, (row, name) in enumerate(zip(rows, base)):
        if safe_uri(name) not in dup:
            out.append(name)
            continue
        ident = (row.get(pk_col, "").strip() if pk_col else "") or str(idx)
        prefix = f"{name}#{ident}"
        cand = prefix
        while safe_uri(cand) in taken:
            next_n[prefix] += 1
            cand = f"{prefix}#{next_n[prefix]}"
        taken.add(safe_uri(cand))
        out.append(cand)
    return out


_SHEET = re.compile(r"^\s*\[Sheet:\s*([^\]]+)\]", re.I)
_SHEET_SEP = "#sheet="   # a '#' in the file name must not read as a sheet key


def _html_rows(text: str) -> list[list[str]]:
    """HTML table rows with spans expanded back to one value and empty cells, the way a
    converter folds empty cells into the previous cell's span."""
    out: list[list[str]] = []
    carry: dict[int, int] = {}   # column -> rows still covered by a rowspan

    def _fill(cells: list[str], col: int) -> int:
        while col in carry:
            cells.append("")
            carry[col] -= 1
            if carry[col] <= 0:
                del carry[col]
            col += 1
        return col

    for row in _ROW.findall(text):
        cells: list[str] = []
        col = 0
        for m in _CELL_TAG.finditer(row):
            col = _fill(cells, col)
            rs = cs = 1
            for kind, n in _SPAN.findall(m.group(1)):
                if kind.lower() == "row":
                    rs = max(1, int(n))
                else:
                    cs = max(1, int(n))
            cells.append(_clean(m.group(2)))
            cells.extend([""] * (cs - 1))
            if rs > 1:
                for k in range(cs):
                    carry[col + k] = rs - 1
            col += cs
        _fill(cells, col)
        if cells:
            out.append(cells)
    return out


def _delimited_rows(text: str) -> list[list[str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    sep = "\t" if lines and "\t" in lines[0] else ","
    return [[c.strip() for c in cells] for cells in csv.reader(lines, delimiter=sep)]


def table_cell_rows(text: str) -> list[list[str]]:
    """The cell rows of a table chunk (HTML, pipe grid, CSV or TSV), header row included."""
    text = (text or "").lstrip("\ufeff")
    if not text.strip():
        return []
    low = text.lower()
    if "<table" in low or "<tr" in low:
        rows = _html_rows(text)
        if rows:
            return rows
    if text.count("|") >= 3:
        return parse_pipe_table(text)
    return _delimited_rows(text)


def split_sheets(file_name: str, chunks: list[dict], header_patterns=()) -> dict[str, list[dict]]:
    """A file with several sheets is one table per sheet: ``{key: chunks}``. One sheet (or
    no sheet marker) keeps the file name as the key."""
    by_sheet: dict[str, list[dict]] = {}
    for chunk in chunks:
        # the sheet marker comes first once ingestion headers and metadata are gone
        m = _SHEET.match(_META.sub("", strip_headers(chunk.get("chunk_text", "") or "", header_patterns)))
        by_sheet.setdefault(m.group(1).strip() if m else "", []).append(chunk)
    if len(by_sheet) <= 1:
        return {file_name: chunks}
    return {f"{file_name}{_SHEET_SEP}{sheet}" if sheet else file_name: part for sheet, part in by_sheet.items()}


def sheet_of(key: str) -> str:
    """The sheet name in a :func:`split_sheets` key, or ``""``."""
    return key.rsplit(_SHEET_SEP, 1)[1] if _SHEET_SEP in key else ""


def unique_names(names: list[str]) -> list[str]:
    """Number repeated names (_2, _3): two equal column names would collapse into one dict key."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def header_names(cells: list[str]) -> list[str]:
    """A cell row as column names: normalized spelling, repeats numbered. Header rows are compared in this form."""
    return unique_names([normalize_text(c) for c in cells])


def _header_and_samples(chunks: list[dict]) -> tuple[list[str], list[list[str]], int]:
    if not chunks:
        return [], [], 0
    all_rows: list[list[str]] = []
    for ch in sorted(chunks, key=lambda c: c.get("chunk_index", 0)):
        all_rows.extend(table_cell_rows(ch.get("chunk_text", "")))
    if not all_rows:
        return [], [], 0
    header = header_names(all_rows[0])
    # a table split over chunks repeats its header in every chunk: those are not data
    data_rows = [r for r in all_rows[1:] if header_names(r) != header]
    return header, data_rows, len(data_rows)


def _rows_from_chunks(chunks: list[dict], columns: list[str]) -> tuple[list[dict[str, str]], list[list[str]]]:
    """Cell rows as column dicts, each with the id of the chunk it came from."""
    n = len(columns)
    rows: list[dict[str, str]] = []
    src: list[list[str]] = []
    for ch in chunks:
        cid = ch.get("chunk_id", "")
        for r in table_cell_rows(ch.get("chunk_text", "")):
            if len(r) >= n and header_names(r[:n]) != columns:
                rows.append(dict(zip(columns, r[:n])))
                src.append([cid] if cid else [])
    return rows, src


def _infer_column_types(header: list[str], rows: list[list[str]]) -> dict[str, str]:
    types: dict[str, str] = {}
    for i, col in enumerate(header):
        vals = [r[i].strip() for r in rows[:20] if i < len(r) and r[i].strip()]
        types[col] = _value_type(vals) if vals else "xsd:string"
    return types


def _value_type(values: list[str]) -> str:
    if not values:
        return "xsd:string"
    ip = re.compile(r"^-?\d+$")
    dp = re.compile(r"^-?\d+\.\d+$")
    datep = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")
    bools = {"true", "false", "yes", "no", "0", "1"}
    ic = dc = dtc = bc = 0
    for v in values:
        v = v.strip()
        if ip.match(v):
            ic += 1
        elif dp.match(v):
            dc += 1
        elif datep.match(v):
            dtc += 1
        elif v.lower() in bools:
            bc += 1
    total = len(values)
    if (ic + dc) / total >= 0.7:
        return "xsd:decimal" if dc > 0 else "xsd:integer"
    if dtc / total >= 0.7:
        return "xsd:date"
    if bc / total >= 0.7:
        return "xsd:boolean"
    return "xsd:string"


def _pk_candidates(header: list[str], rows: list[list[str]]) -> list[str]:
    out = []
    suffixes = ("_id", "_code", "_no", "id")
    for i, col in enumerate(header):
        cl = col.lower()
        if not (any(cl.endswith(s) for s in suffixes) or cl.startswith("id") or cl in ("id", "code", "no")):
            continue
        vals = [r[i].strip() for r in rows[:30] if i < len(r)]
        if vals and len(set(vals)) / len(vals) >= 0.8:
            out.append(col)
    if not out and header:
        out.append(header[0])
    return out


def _sample_values(header: list[str], rows: list[list[str]], max_samples: int = 30) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i, col in enumerate(header):
        vals: list[str] = []
        for r in rows[:max_samples]:
            if i < len(r):
                v = r[i].strip()
                if v and v not in vals:
                    vals.append(v)
        out[col] = vals
    return out


def _norm_col(name: str) -> str:
    n = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for s in ("id", "code", "no", "key", "num", "number"):
        if n.endswith(s) and len(n) > len(s):
            return n[:-len(s)]
    return n


def _is_numeric_data_col(col: str, ctype: str, samples: list[str]) -> bool:
    if ctype not in ("xsd:integer", "xsd:decimal"):
        return False
    if any(col.lower().endswith(s) for s in ("_id", "_code", "_no", "id")):
        return False
    return bool(samples) and len(set(samples)) / max(len(samples), 1) > 0.5


def _fk_relations(tables: dict[str, dict]) -> list[dict[str, str]]:
    relations: list[dict] = []
    seen: set = set()
    if len(tables) < 2:
        return relations

    numeric_cols: dict[str, set] = {}
    for fn, info in tables.items():
        numeric_cols[fn] = {
            col for col in info["columns"]
            if _is_numeric_data_col(col, info.get("column_types", {}).get(col, "xsd:string"),
                                    info.get("sample_values", {}).get(col, []))
        }

    # same-name columns
    col_to_tables: dict[str, list[str]] = defaultdict(list)
    for fn, info in tables.items():
        for col in info["columns"]:
            if col not in numeric_cols.get(fn, set()):
                col_to_tables[col].append(fn)
    for col, fns in col_to_tables.items():
        if len(fns) < 2:
            continue
        # the referenced (PK) table is the dimension this column points at: prefer the
        # table whose name matches the column (color_id -> colors), then the smaller one.
        pk_table = None
        best = None
        norm = _norm_col(col)
        for fn in fns:
            if col not in tables[fn].get("pk_candidates", []):
                continue
            tname = tables[fn]["table_name"].lower()
            score = (bool(norm) and (norm in tname or tname in norm), -len(tables[fn]["columns"]))
            if best is None or score > best:
                best, pk_table = score, fn
        if not pk_table:
            continue
        pk_vals = set(tables[pk_table].get("sample_values", {}).get(col, []))
        for fn in fns:
            if fn == pk_table:
                continue
            from_vals = set(tables[fn].get("sample_values", {}).get(col, []))
            if from_vals and pk_vals and len(from_vals & pk_vals) / max(len(from_vals), 1) < 0.3:
                continue
            key = (tables[fn]["table_name"], col, tables[pk_table]["table_name"], col)
            if key not in seen:
                seen.add(key)
                relations.append({"from_table": key[0], "from_column": key[1],
                                  "to_table": key[2], "to_column": key[3]})

    # different-name columns: normalized name + value overlap
    items = list(tables.items())
    for i, (fn_a, a) in enumerate(items):
        for fn_b, b in items[i + 1:]:
            for col_a in a["columns"]:
                if col_a in numeric_cols.get(fn_a, set()):
                    continue
                vals_a = set(a.get("sample_values", {}).get(col_a, []))
                if len(vals_a) < 2:
                    continue
                for col_b in b["columns"]:
                    if col_a == col_b or col_b in numeric_cols.get(fn_b, set()):
                        continue
                    vals_b = set(b.get("sample_values", {}).get(col_b, []))
                    if len(vals_b) < 2:
                        continue
                    overlap = vals_a & vals_b
                    if not overlap:
                        continue
                    na, nb = _norm_col(col_a), _norm_col(col_b)
                    name_match = na and nb and (na == nb or na in nb or nb in na)
                    # Serial ids of two tables overlap by construction: numbers alone are no evidence.
                    if not name_match and all(v.strip().lstrip("+-").replace(".", "", 1).isdigit() for v in overlap):
                        continue
                    threshold = 0.3 if name_match else 0.8
                    if len(overlap) / len(vals_a) >= threshold or len(overlap) / len(vals_b) >= threshold:
                        pk_b = set(b.get("pk_candidates", []))
                        pk_a = set(a.get("pk_candidates", []))
                        # The direction comes from the definitions, not the sizes: the side whose
                        # column is a key is referenced. Both keys (a 1:1 identity) or neither (a
                        # shared code value) is not a foreign key.
                        if col_b in pk_b and col_a not in pk_a:
                            ft, fc, tt, tc = a["table_name"], col_a, b["table_name"], col_b
                        elif col_a in pk_a and col_b not in pk_b:
                            ft, fc, tt, tc = b["table_name"], col_b, a["table_name"], col_a
                        else:
                            continue
                        key = (ft, fc, tt, tc)
                        if key not in seen:
                            seen.add(key)
                            relations.append({"from_table": ft, "from_column": fc,
                                              "to_table": tt, "to_column": tc})
    return relations
