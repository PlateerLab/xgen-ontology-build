"""Which rows of a table changed since they were last loaded.

A table is loaded again from a full read in key order; every row gets a fingerprint (the
mapping it is loaded under plus its values) and is compared with the fingerprint recorded
when it was last loaded. A key without a record is a new row, a key whose fingerprint differs
a changed one, the same fingerprint an unchanged one, and a recorded key the read did not
reach (when the read was complete) a row that left the table. No timestamp or sequence column
is needed, so a value that changed without its stamp, a stamp set back in time and a deleted
row are all seen. This is the production synchronization's judgement
(``service/ontology_db/row_state.py``) without its storage: keep the fingerprints wherever the
application keeps state and hand them back on the next read.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from xgen_ontology_build.extract.tabular import value_text

__all__ = ["RowDiff", "diff_rows", "key_text", "mapping_signature", "row_fingerprint"]


def mapping_signature(columns: Iterable[str], pk_column: str, label_column: str | None = None,
                      fk_relations: Iterable[dict] | None = None) -> str:
    """How the rows are turned into graph: the columns read, the key, the name column and the
    foreign keys. When this changes every row is a changed row even if its values are the same,
    because what the graph holds for it is different."""
    payload = {
        "columns": [str(c) for c in columns],
        "pk": pk_column,
        "label": label_column or None,
        "fk": sorted(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in (fk_relations or [])),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def row_fingerprint(signature: str, columns: Iterable[str], row: Mapping[str, Any]) -> str:
    """The fingerprint of one row under a mapping: the values of ``columns`` as the driver gave
    them (``None`` stays ``None``, so a value becoming NULL is a change)."""
    values = [row.get(c) for c in columns]
    body = json.dumps([signature, values], ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def key_text(value: Any) -> str:
    """A key value as the text the row's source id carries (``"{source_id}:{key}"``), the same
    rendering :func:`~xgen_ontology_build.extract.tabular.build_from_rows` uses, so a recorded
    key finds the row's chunk when the row has to be retracted."""
    return value_text(value)


@dataclass
class RowDiff:
    """What one read of a table found against the recorded fingerprints."""

    added: list[dict] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    #: recorded keys the (complete) read did not reach: rows that left the table
    missing: list[str] = field(default_factory=list)
    #: the fingerprints to record for the next read: every row read, by key
    fingerprints: dict[str, str] = field(default_factory=dict)
    complete: bool = True

    @property
    def to_load(self) -> list[dict]:
        """The rows the graph has to take again: new and changed, in read order."""
        return self.added + self.changed

    def summary(self) -> dict:
        return {"scanned": len(self.added) + len(self.changed) + len(self.unchanged),
                "added": len(self.added), "changed": len(self.changed), "unchanged": len(self.unchanged),
                "deleted": len(self.missing), "complete": self.complete}


def diff_rows(rows: Iterable[Mapping[str, Any]], columns: Iterable[str], pk_column: str, *,
              known: Mapping[str, str] | None = None, signature: str | None = None,
              complete: bool = True) -> RowDiff:
    """Compare a read of a table with the fingerprints recorded at the last load.

    ``known`` maps a key (``key_text``) to the fingerprint recorded when the row was last loaded
    into the graph; ``signature`` is :func:`mapping_signature` (a bare column list when not
    given). ``complete`` says the read reached the end of the table: only then is a recorded key
    the read did not meet a deleted row. A row whose key is empty is skipped: it cannot be told
    apart between reads.
    """
    columns = [str(c) for c in columns]
    sig = signature or mapping_signature(columns, pk_column)
    known = dict(known or {})
    out = RowDiff(complete=complete)
    seen: set[str] = set()
    for row in rows:
        key = key_text(row.get(pk_column))
        if not key.strip():
            continue
        seen.add(key)
        digest = row_fingerprint(sig, columns, row)
        out.fingerprints[key] = digest
        have = known.get(key)
        if have is None:
            out.added.append(dict(row))
        elif have != digest:
            out.changed.append(dict(row))
        else:
            out.unchanged.append(key)
    if complete:
        out.missing = sorted(k for k in known if k not in seen)
    return out
