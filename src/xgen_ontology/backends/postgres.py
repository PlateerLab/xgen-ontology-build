"""PostgreSQL adapter: the XGEN graph tables as a sink, a search store and a source for incremental builds.

The production system keeps its graph in three tables (``ontology_nodes``,
``ontology_edges``, ``ontology_node_chunks``) plus ``ontology_enriched_chunks``
for the LLM relation pass. :class:`PgGraph` speaks that schema over any DB-API
connection (psycopg 2/3; sqlite for tests), so a library build can be loaded where
the product reads it, a stored graph can be searched through the
:class:`~xgen_ontology.protocols.GraphStore` protocol, and it can be loaded back
into an :class:`~xgen_ontology.Ontology` to be :meth:`extended
<xgen_ontology.OntologyBuilder.extend>`.

No driver is imported here: pass a connection. Job / session bookkeeping stays
with the application; this module only knows the graph.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..korean import normalize_label
from ..models import Chunk, Class, Concepts, DataProperty, DataValue, Instance, Node, ObjectProperty, Relation
from ..text import tokenize

DOMAIN_NS = "https://w3id.org/xgen-domain#"
INSTANCE_NS = "https://w3id.org/xgen-instance#"
_URI_LOCAL_MAX = 40
_INSERT_BATCH = 2000
_KIND_OF = {"concept": "class", "instance": "instance", "property": "property", "literal": "instance"}

_DDL = {
    "postgres": [
        "CREATE TABLE IF NOT EXISTS ontology_nodes (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, kind TEXT NOT NULL, label TEXT, attrs JSONB, nkey TEXT,"
        " UNIQUE (collection_id, uri))",
        "CREATE TABLE IF NOT EXISTS ontology_edges (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " subject_uri TEXT NOT NULL, predicate TEXT, object_uri TEXT, predicate_uri TEXT, edge_kind TEXT,"
        " edge_weight REAL, UNIQUE (collection_id, subject_uri, predicate, object_uri))",
        "CREATE TABLE IF NOT EXISTS ontology_node_chunks (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, chunk_id TEXT NOT NULL, UNIQUE (collection_id, uri, chunk_id))",
        "CREATE TABLE IF NOT EXISTS ontology_enriched_chunks (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " chunk_id TEXT NOT NULL, UNIQUE (collection_id, chunk_id))",
        "CREATE INDEX IF NOT EXISTS ix_onodes_coll_label ON ontology_nodes (collection_id, label)",
        "CREATE INDEX IF NOT EXISTS ix_oedges_o ON ontology_edges (collection_id, object_uri)",
        "CREATE INDEX IF NOT EXISTS ix_oedges_po ON ontology_edges (collection_id, predicate, object_uri)",
        "CREATE INDEX IF NOT EXISTS ix_onc_coll_chunk ON ontology_node_chunks (collection_id, chunk_id)",
    ],
    "sqlite": [
        "CREATE TABLE IF NOT EXISTS ontology_nodes (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, kind TEXT NOT NULL, label TEXT, attrs TEXT, nkey TEXT, UNIQUE (collection_id, uri))",
        "CREATE TABLE IF NOT EXISTS ontology_edges (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " subject_uri TEXT NOT NULL, predicate TEXT, object_uri TEXT, predicate_uri TEXT, edge_kind TEXT,"
        " edge_weight REAL, UNIQUE (collection_id, subject_uri, predicate, object_uri))",
        "CREATE TABLE IF NOT EXISTS ontology_node_chunks (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, chunk_id TEXT NOT NULL, UNIQUE (collection_id, uri, chunk_id))",
        "CREATE TABLE IF NOT EXISTS ontology_enriched_chunks (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " chunk_id TEXT NOT NULL, UNIQUE (collection_id, chunk_id))",
    ],
}


def stable_local(name: str, prefix: str = "") -> str:
    """IRI local part: the name itself, or a truncation plus a hash when it is too long."""
    local = prefix + name
    if len(local) <= _URI_LOCAL_MAX:
        return local
    return local[:_URI_LOCAL_MAX] + "~" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def class_uri(name: str) -> str:
    return DOMAIN_NS + name


def instance_uri(name: str) -> str:
    return INSTANCE_NS + stable_local(normalize_label(name))


def graph_rows(ontology, *, translations: dict[str, str] | None = None
               ) -> tuple[list[tuple], list[tuple], list[tuple[str, str]]]:
    """An ontology as the store's rows: ``(nodes, edges, node_chunks)``.

    ``nodes`` are ``(uri, kind, label, attrs)`` with ``kind`` in concept / property /
    instance and ``attrs`` the node's data values ``{property: [values]}``; ``edges``
    are ``(subject_uri, predicate, object_uri, predicate_uri, edge_kind, weight)`` with
    the production edge kinds; ``node_chunks`` are ``(uri, chunk_id)``. Same shape the
    production ``serialize_graph`` produces from an equivalent build.
    """
    translations = translations or {}
    c = ontology.concepts

    def pred_uri(name: str) -> str:
        eng = translations.get(name) or translations.get(normalize_label(name))
        if eng:
            eng = re.sub(r"[^a-zA-Z0-9]", "", eng)
            if eng:
                return DOMAIN_NS + eng[0].lower() + eng[1:]
        return DOMAIN_NS + name

    nodes: list[tuple] = []
    edges: list[tuple] = []
    chunks: list[tuple[str, str]] = []
    class_names = {cl.name for cl in c.classes if cl.name}
    for cl in c.classes:
        if not cl.name:
            continue
        nodes.append((class_uri(cl.name), "concept", cl.name, None))
        chunks += [(class_uri(cl.name), cid) for cid in cl.source_chunks if cid]
    for dp in c.datatype_properties:
        if not dp.name:
            continue
        puri = DOMAIN_NS + (f"{dp.domain}_{dp.name}" if dp.domain else dp.name)
        nodes.append((puri, "property", dp.name, None))
        if dp.domain and dp.domain in class_names:
            edges.append((class_uri(dp.domain), dp.name, puri, pred_uri(dp.name), "datatypeProperty_schema", 1.0))
    for op in c.object_properties:
        if op.name and op.domain in class_names and op.range in class_names:
            edges.append((class_uri(op.domain), op.name, class_uri(op.range), pred_uri(op.name),
                          "objectProperty_schema", 1.0))
    for parent, child in c.class_hierarchy:
        if parent in class_names and child in class_names and parent != child:
            edges.append((class_uri(child), "subClassOf", class_uri(parent), DOMAIN_NS + "subClassOf",
                          "subClassOf", 1.0))

    inst_names: set[str] = set()
    for i in ontology.instances:
        if not i.name:
            continue
        u = instance_uri(i.name)
        if i.name not in inst_names:
            nodes.append((u, "instance", i.name, None))
            inst_names.add(i.name)
        if i.class_name and i.class_name in class_names:
            edges.append((u, "instanceOf", class_uri(i.class_name), DOMAIN_NS + "instanceOf", "instanceOf", 1.0))
        chunks += [(u, cid) for cid in i.source_chunks if cid]

    def endpoint(name: str) -> str:
        return instance_uri(name) if name in inst_names or name not in class_names else class_uri(name)

    for r in ontology.relations:
        if not (r.subject and r.predicate and r.object) or r.predicate_type == "DatatypeProperty":
            continue
        for end in (r.subject, r.object):
            if end not in inst_names and end not in class_names:
                nodes.append((instance_uri(end), "instance", end, None))
                inst_names.add(end)
        edges.append((endpoint(r.subject), r.predicate, endpoint(r.object), pred_uri(r.predicate),
                      "objectProperty", 1.0))
        chunks += [(endpoint(r.subject), cid) for cid in r.source_chunks if cid]
        chunks += [(endpoint(r.object), cid) for cid in r.source_chunks if cid]

    attrs: dict[str, dict[str, list[str]]] = {}
    for d in ontology.data_values:
        if not (d.entity and d.property) or d.value is None:
            continue
        if d.entity not in inst_names and d.entity not in class_names:
            nodes.append((instance_uri(d.entity), "instance", d.entity, None))
            inst_names.add(d.entity)
        u = endpoint(d.entity)
        vals = attrs.setdefault(u, {}).setdefault(str(d.property), [])
        if str(d.value) not in vals:
            vals.append(str(d.value))
        chunks += [(u, cid) for cid in d.source_chunks if cid]

    seen: set[str] = set()
    uniq_nodes = []
    for u, k, lb, _a in nodes:
        if u in seen:
            continue
        seen.add(u)
        uniq_nodes.append((u, k, lb, attrs.get(u) or None))
    seen_e: set[tuple] = set()
    uniq_edges = []
    for e in edges:
        if e[:3] in seen_e:
            continue
        seen_e.add(e[:3])
        uniq_edges.append(e)
    return uniq_nodes, uniq_edges, sorted(set(chunks))


class PgGraph:
    """The XGEN graph tables for one collection over a DB-API connection.

    ``paramstyle`` is ``"format"`` (``%s``, psycopg) or ``"qmark"`` (``?``, sqlite);
    ``dialect`` selects the DDL :meth:`ensure_schema` emits. The connection is used
    as given: commit yourself, or pass one in autocommit mode.
    """

    def __init__(self, conn, collection_id: str, *, paramstyle: str = "format", dialect: str = "postgres"):
        self.conn = conn
        self.collection_id = collection_id
        self._ph = "%s" if paramstyle == "format" else "?"
        self.dialect = dialect

    # ── plumbing ──

    def _q(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self.conn.cursor()
        cur.execute(sql.replace("?", self._ph), params)
        try:
            return cur.fetchall() if cur.description else []
        finally:
            cur.close()

    def _x(self, sql: str, params: tuple = ()) -> None:
        cur = self.conn.cursor()
        cur.execute(sql.replace("?", self._ph), params)
        cur.close()

    def _insert(self, prefix: str, n_cols: int, rows: list[tuple], suffix: str) -> None:
        batch = max(1, min(_INSERT_BATCH, 999 // max(1, n_cols)))
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            ph = ",".join(["(" + ",".join(["?"] * n_cols) + ")"] * len(part))
            params: list[Any] = []
            for row in part:
                params.extend(row)
            self._x(f"{prefix} VALUES {ph} {suffix}", tuple(params))

    def ensure_schema(self) -> None:
        """Create the four tables (and the read indexes) when they do not exist."""
        for ddl in _DDL[self.dialect]:
            self._x(ddl)

    # ── sink ──

    def write(self, ontology, *, replace: bool = True, translations: dict[str, str] | None = None) -> dict:
        """Store ``ontology``. ``replace`` clears the collection first; otherwise rows are added and a
        node seen again merges its attributes (the incremental path)."""
        nodes, edges, chunks = graph_rows(ontology, translations=translations or ontology.translations)
        C = self.collection_id
        if replace:
            self.clear()
            merged = {u: a for u, _k, _l, a in nodes if a}
        else:
            merged = {}
            uris = [u for u, _k, _l, a in nodes if a]
            for i in range(0, len(uris), 500):
                part = uris[i:i + 500]
                rows = self._q("SELECT uri, attrs FROM ontology_nodes WHERE collection_id=? AND uri IN ("
                               + ",".join(["?"] * len(part)) + ")", (C, *part))
                for u, a in rows:
                    merged[u] = _attrs(a)
            for u, _k, _l, a in nodes:
                if a:
                    cur = merged.setdefault(u, {})
                    for k, vs in a.items():
                        have = cur.setdefault(k, [])
                        have.extend(v for v in vs if v not in have)
        self._insert("INSERT INTO ontology_nodes(collection_id, uri, kind, label, attrs)", 5,
                     [(C, u, k, lb, json.dumps(merged.get(u), ensure_ascii=False) if merged.get(u) else None)
                      for u, k, lb, _a in nodes],
                     "ON CONFLICT (collection_id, uri) DO UPDATE SET attrs = COALESCE(excluded.attrs, "
                     "ontology_nodes.attrs)")
        self._insert("INSERT INTO ontology_edges(collection_id, subject_uri, predicate, object_uri, predicate_uri,"
                     " edge_kind, edge_weight)", 7, [(C, *e) for e in edges],
                     "ON CONFLICT (collection_id, subject_uri, predicate, object_uri) DO NOTHING")
        self._insert("INSERT INTO ontology_node_chunks(collection_id, uri, chunk_id)", 3,
                     [(C, u, cid) for u, cid in chunks], "ON CONFLICT (collection_id, uri, chunk_id) DO NOTHING")
        if ontology.enriched_chunks:
            self.mark_enriched(ontology.enriched_chunks)
        return {"nodes": len(nodes), "edges": len(edges), "node_chunks": len(chunks)}

    def clear(self) -> None:
        for t in ("ontology_edges", "ontology_nodes", "ontology_node_chunks", "ontology_enriched_chunks"):
            self._x(f"DELETE FROM {t} WHERE collection_id=?", (self.collection_id,))

    def mark_enriched(self, chunk_ids) -> int:
        ids = sorted({str(c) for c in chunk_ids if c})
        if ids:
            self._insert("INSERT INTO ontology_enriched_chunks(collection_id, chunk_id)", 2,
                         [(self.collection_id, c) for c in ids], "ON CONFLICT (collection_id, chunk_id) DO NOTHING")
        return len(ids)

    def enriched_chunk_ids(self) -> set[str]:
        return {r[0] for r in self._q("SELECT chunk_id FROM ontology_enriched_chunks WHERE collection_id=?",
                                      (self.collection_id,))}

    def counts(self) -> dict:
        C = self.collection_id
        out = {}
        for k in ("concept", "instance", "property"):
            out[k] = self._q("SELECT count(*) FROM ontology_nodes WHERE collection_id=? AND kind=?", (C, k))[0][0]
        out["edges"] = self._q("SELECT count(*) FROM ontology_edges WHERE collection_id=?", (C,))[0][0]
        out["chunks"] = self._q("SELECT count(DISTINCT chunk_id) FROM ontology_node_chunks WHERE collection_id=?",
                                (C,))[0][0]
        return out

    # ── source: back into build models ──

    def load(self):
        """The stored collection as an :class:`~xgen_ontology.Ontology` (chunk texts are not stored: ids only)."""
        from ..ontology import Ontology

        C = self.collection_id
        rows = self._q("SELECT uri, kind, label, attrs FROM ontology_nodes WHERE collection_id=? ORDER BY id", (C,))
        label: dict[str, str] = {}
        kind: dict[str, str] = {}
        attrs: dict[str, dict] = {}
        for u, k, lb, a in rows:
            label[u], kind[u] = lb or "", k
            if a:
                attrs[u] = _attrs(a)
        chunks_of: dict[str, list[str]] = {}
        for u, cid in self._q("SELECT uri, chunk_id FROM ontology_node_chunks WHERE collection_id=? ORDER BY id", (C,)):
            chunks_of.setdefault(u, []).append(cid)
        concepts = Concepts(classes=[Class(name=label[u], source_chunks=chunks_of.get(u, []))
                                     for u in label if kind[u] == "concept" and label[u]])
        instances: list[Instance] = []
        typed: set[str] = set()
        relations: list[Relation] = []
        edges = self._q("SELECT subject_uri, predicate, object_uri, edge_kind FROM ontology_edges"
                        " WHERE collection_id=? ORDER BY id", (C,))
        for s, p, o, ek in edges:
            if ek == "subClassOf":
                concepts.class_hierarchy.append((label.get(o, ""), label.get(s, "")))
            elif ek == "instanceOf":
                instances.append(Instance(name=label.get(s, ""), class_name=label.get(o, ""),
                                          source_chunks=chunks_of.get(s, [])))
                typed.add(s)
            elif ek == "objectProperty_schema":
                concepts.object_properties.append(ObjectProperty(name=p, domain=label.get(s, ""), range=label.get(o, "")))
            elif ek == "datatypeProperty_schema":
                concepts.datatype_properties.append(DataProperty(name=p, domain=label.get(s, "")))
            elif ek == "objectProperty":
                relations.append(Relation(subject=label.get(s, ""), predicate=p, object=label.get(o, ""),
                                          source_chunks=sorted(set(chunks_of.get(s, [])) & set(chunks_of.get(o, [])))))
        for u in label:
            if kind[u] == "instance" and u not in typed and label[u]:
                instances.append(Instance(name=label[u], source_chunks=chunks_of.get(u, [])))
        declared_dp = {dp.name for dp in concepts.datatype_properties}
        data_values: list[DataValue] = []
        for u, a in attrs.items():
            for prop, vals in a.items():
                for v in vals:
                    data_values.append(DataValue(entity=label.get(u, ""), property=prop, value=v,
                                                 source_chunks=chunks_of.get(u, [])))
                if prop not in declared_dp:
                    concepts.datatype_properties.append(DataProperty(name=prop))
                    declared_dp.add(prop)
        all_chunks = sorted({cid for cids in chunks_of.values() for cid in cids})
        onto = Ontology(concepts=concepts, instances=instances, relations=relations, data_values=data_values,
                        chunks=[Chunk(id=cid, text="") for cid in all_chunks],
                        enriched_chunks=sorted(self.enriched_chunk_ids()))
        return onto

    # ── GraphStore protocol (search) ──

    def _uri(self, node_id: str) -> str | None:
        if node_id.startswith("http"):
            return node_id
        rows = self._q("SELECT uri FROM ontology_nodes WHERE collection_id=? AND label=?"
                       " ORDER BY CASE kind WHEN 'concept' THEN 0 WHEN 'instance' THEN 1 ELSE 2 END, id",
                       (self.collection_id, node_id))
        return rows[0][0] if rows else None

    def get_node(self, node_id: str) -> Node | None:
        u = self._uri(node_id)
        if u is None:
            return None
        rows = self._q("SELECT uri, kind, label FROM ontology_nodes WHERE collection_id=? AND uri=?",
                       (self.collection_id, u))
        return Node(rows[0][0], rows[0][2] or "", _KIND_OF.get(rows[0][1], "instance")) if rows else None

    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Node, float]]:
        toks = [t for t in dict.fromkeys(tokenize(query)) if len(t) >= 2][:12]
        if not toks:
            return []
        conds = " OR ".join(["LOWER(label) LIKE ?"] * len(toks))
        rows = self._q(f"SELECT uri, kind, label FROM ontology_nodes WHERE collection_id=? AND kind <> 'literal'"
                       f" AND ({conds})", (self.collection_id, *[f"%{t.lower()}%" for t in toks]))
        scored = []
        for u, k, lb in rows:
            low = (lb or "").lower()
            hits = sum(1 for t in toks if t.lower() in low)
            scored.append((Node(u, lb or "", _KIND_OF.get(k, "instance")), hits / (1 + len(low) / 40.0)))
        scored.sort(key=lambda x: (-x[1], x[0].label))
        return scored[:limit]

    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Node]:
        u = self._uri(class_id)
        if u is None:
            return []
        rows = self._q("SELECT n.uri, n.kind, n.label FROM ontology_edges e JOIN ontology_nodes n"
                       " ON n.collection_id=e.collection_id AND n.uri=e.subject_uri"
                       " WHERE e.collection_id=? AND e.edge_kind='instanceOf' AND e.object_uri=? ORDER BY n.id LIMIT ?",
                       (self.collection_id, u, int(limit)))
        return [Node(r[0], r[2] or "", _KIND_OF.get(r[1], "instance")) for r in rows]

    def count_class(self, class_id: str) -> int:
        u = self._uri(class_id)
        if u is None:
            return 0
        return self._q("SELECT count(*) FROM ontology_edges WHERE collection_id=? AND edge_kind='instanceOf'"
                       " AND object_uri=?", (self.collection_id, u))[0][0]

    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]:
        u = self._uri(node_id)
        if u is None:
            return []
        out: list[tuple[str, str, str]] = []
        seen = {u}
        frontier = [u]
        for _ in range(max(1, hops)):
            if not frontier or len(out) >= limit:
                break
            ph = ",".join(["?"] * len(frontier))
            rows = self._q("SELECT s.label, e.predicate, o.label, e.subject_uri, e.object_uri FROM ontology_edges e"
                           " JOIN ontology_nodes s ON s.collection_id=e.collection_id AND s.uri=e.subject_uri"
                           " JOIN ontology_nodes o ON o.collection_id=e.collection_id AND o.uri=e.object_uri"
                           f" WHERE e.collection_id=? AND (e.subject_uri IN ({ph}) OR e.object_uri IN ({ph}))"
                           " ORDER BY e.id LIMIT ?",
                           (self.collection_id, *frontier, *frontier, int(limit)))
            nxt = []
            for sl, p, ol, su, ou in rows:
                out.append((sl or "", p or "", ol or ""))
                for x in (su, ou):
                    if x not in seen:
                        seen.add(x)
                        nxt.append(x)
            frontier = nxt
        return out[:limit]

    def entity_labels(self, limit: int = 300) -> list[str]:
        """Instance and class labels, best-connected first (the enrich pass's known-entity list)."""
        rows = self._q("SELECT n.label, (SELECT count(*) FROM ontology_edges e WHERE e.collection_id=n.collection_id"
                       " AND (e.subject_uri=n.uri OR e.object_uri=n.uri)) AS d FROM ontology_nodes n"
                       " WHERE n.collection_id=? AND n.kind <> 'literal' AND COALESCE(n.label,'') <> ''"
                       " ORDER BY d DESC, n.id LIMIT ?", (self.collection_id, int(limit)))
        return [r[0] for r in rows]

    def predicates(self, limit: int = 120) -> list[str]:
        rows = self._q("SELECT predicate, count(*) AS n FROM ontology_edges WHERE collection_id=?"
                       " AND COALESCE(predicate,'') <> '' GROUP BY predicate ORDER BY n DESC LIMIT ?",
                       (self.collection_id, int(limit)))
        return [r[0] for r in rows]


def _attrs(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) or {}
    except Exception:
        return {}
