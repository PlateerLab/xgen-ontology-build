"""Deterministic table -> ontology (no LLM).

Two stages, both domain-general:

1. ``analyze_tables`` — infer a relational schema from CSV/TSV/Markdown/HTML table
   chunks: columns, xsd column types, primary-key candidates, and foreign-key
   relations (same-name, normalized-name, *and* value-overlap detection).
2. ``build_from_tables`` — turn that schema into an ontology by the star-schema
   rule: table -> Class, FK -> ObjectProperty, column -> DataProperty, dimension
   rows -> instances. Fact / event tables (FK-source-only or junctions, when large)
   are kept as schema only (their rows belong in a SQL store, not the graph).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from typing import Any

from ..models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation
from ..text import safe_uri
from .deterministic import is_value

TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
_REF_TABLE_MAX_ROWS = 200  # at/below this a table is treated as a dimension (instantiated)


# ───────────────────────── schema inference ─────────────────────────


def analyze_tables(documents: dict[str, list[dict]]) -> dict[str, Any]:
    """Infer table schemas from chunked table documents.

    ``documents`` = ``{file_name: [{"chunk_id","chunk_text","chunk_index"}, ...]}``.
    Returns ``{"tables": {...}, "fk_relations": [...], "is_table_collection": bool}``.
    """
    tables: dict[str, dict] = {}
    table_file_count = 0

    for file_name, chunks in documents.items():
        if _ext(file_name) not in TABLE_EXTENSIONS:
            continue
        table_file_count += 1
        header, sample_rows, total_rows = _header_and_samples(chunks)
        if not header:
            continue
        tables[file_name] = {
            "table_name": _table_name(file_name),
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
) -> tuple[Concepts, list[Instance], list[Relation], list[DataValue]]:
    """Schema + rows -> ontology. LLM-free, deterministic."""
    tables = schema.get("tables", {})
    fk_relations = schema.get("fk_relations", [])
    if not tables:
        return Concepts(), [], [], []

    table_class = {t.get("table_name", fn): _camel(t.get("table_name", fn)) for fn, t in tables.items()}

    classes = [
        Class(name=table_class[t.get("table_name", fn)],
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
        if n_rows <= _REF_TABLE_MAX_ROWS:
            return False
        return raw in fk_source_only or fk_col_count.get(raw, 0) >= 2

    # Pass 1: rows, one instance label per row (duplicates split by identity), PK -> label.
    # A table with no name column gets no instances: naming rows by their PK turns numbers
    # into entities. Fact tables get none either.
    table_rows: dict[str, list[dict[str, str]]] = {}
    row_chunks: dict[str, list[list[str]]] = {}
    table_labels: dict[str, list[str]] = {}
    pk_lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for fn, chunks in documents.items():
        t = tables.get(fn)
        if not t:
            continue
        raw = t["table_name"]
        rows, src = _rows_from_chunks(chunks, t.get("columns", []))
        table_rows[fn], row_chunks[fn] = rows, src
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
    for fn in documents:
        t = tables.get(fn)
        if not t or fn not in table_labels:
            continue
        raw = t["table_name"]
        cls_name = table_class[raw]
        pk_col = table_pk.get(raw)
        rows = table_rows.get(fn, [])
        label_col = _label_col(t, rows)
        fk_cols = {fc for fc, _, _ in fk_index.get(raw, [])}
        labels = table_labels[fn]
        src = row_chunks.get(fn, [])
        for i, row in enumerate(rows):
            name = labels[i]
            chunk_ids = src[i] if i < len(src) else []
            instances.append(Instance(name=name, class_name=cls_name, source_chunks=list(chunk_ids)))

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
                if not fk_val:
                    continue
                to_cls = table_class.get(to_table, _camel(to_table))
                prop = f"{cls_name}_{from_col}"
                lookup = pk_lookup.get(to_table, {})
                target = lookup.get(fk_raw) or lookup.get(fk_val) or f"{to_table}_{fk_val}"
                relations.append(Relation(subject=name, predicate=prop, object=target,
                                          predicate_type="ObjectProperty", source_chunks=list(chunk_ids)))

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


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell = ""
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell, self._cell = True, ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._row.append(self._cell.strip())
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell += data


def _parse_rows(text: str) -> list[list[str]]:
    if "<table" in text.lower() or "<tr" in text.lower():
        p = _TableHTMLParser()
        try:
            p.feed(text)
            if p.rows:
                return p.rows
        except Exception:
            pass
    lines = text.splitlines()
    pipe_rows = []
    for line in lines:
        line = line.strip()
        if "|" in line and line.count("|") >= 2:
            if re.match(r"^[\s|:-]+$", line):
                continue
            cells = [c.strip() for c in line.split("|")]
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if cells:
                pipe_rows.append(cells)
    if pipe_rows:
        return pipe_rows
    csv_rows = []
    for line in lines:
        line = line.strip()
        if line and "," in line:
            cells = _csv_split(line)
            if len(cells) >= 2:
                csv_rows.append(cells)
    return csv_rows


def _csv_split(line: str) -> list[str]:
    cells, cur, q = [], "", False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            cells.append(cur.strip().strip('"'))
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip().strip('"'))
    return cells


def _header_and_samples(chunks: list[dict]) -> tuple[list[str], list[list[str]], int]:
    if not chunks:
        return [], [], 0
    all_rows: list[list[str]] = []
    for ch in sorted(chunks, key=lambda c: c.get("chunk_index", 0)):
        all_rows.extend(_parse_rows(ch.get("chunk_text", "")))
    if not all_rows:
        return [], [], 0
    total = sum(max(0, len(_parse_rows(ch.get("chunk_text", ""))) - 1) for ch in chunks)
    return all_rows[0], all_rows[1:], max(total, 1)


def _rows_from_chunks(chunks: list[dict], columns: list[str]) -> tuple[list[dict[str, str]], list[list[str]]]:
    """Cell rows as column dicts, each with the id of the chunk it came from."""
    rows: list[dict[str, str]] = []
    src: list[list[str]] = []
    for ch in chunks:
        cid = ch.get("chunk_id", "")
        for r in _parse_rows(ch.get("chunk_text", "")):
            if len(r) >= len(columns) and r[:len(columns)] != columns:
                rows.append(dict(zip(columns, r[:len(columns)])))
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
            key = (_table_name(fn), col, _table_name(pk_table), col)
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
                    threshold = 0.3 if name_match else 0.8
                    if len(overlap) / len(vals_a) >= threshold or len(overlap) / len(vals_b) >= threshold:
                        pk_b = set(b.get("pk_candidates", []))
                        pk_a = set(a.get("pk_candidates", []))
                        if col_b in pk_b or (len(vals_b) >= len(vals_a) and col_a not in pk_a):
                            ft, fc, tt, tc = _table_name(fn_a), col_a, _table_name(fn_b), col_b
                        else:
                            ft, fc, tt, tc = _table_name(fn_b), col_b, _table_name(fn_a), col_a
                        key = (ft, fc, tt, tc)
                        if key not in seen:
                            seen.add(key)
                            relations.append({"from_table": ft, "from_column": fc,
                                              "to_table": tt, "to_column": tc})
    return relations
