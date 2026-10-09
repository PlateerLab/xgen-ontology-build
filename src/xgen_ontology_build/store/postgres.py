"""PostgreSQL adapter: the XGEN graph tables as a sink, a search store and a source for incremental builds.

The production system keeps its graph in three tables (``ontology_nodes``,
``ontology_edges``, ``ontology_node_chunks``), the chunks each relation was stated
in (``ontology_edge_sources``) and ``ontology_enriched_chunks`` for the LLM
relation pass. An edge carries its weight (how many times it was asserted) and its
label (how the document names the relation). The relation vocabulary (names with
definitions) is kept where the product keeps it: ``ObjectProperty`` rows of
``ontology_schema``. :class:`PgGraph` speaks that schema over any DB-API
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
from typing import Any

from xgen_ontology_build.models import (
    Chunk,
    Class,
    Concepts,
    DataProperty,
    DataValue,
    Instance,
    Node,
    ObjectProperty,
    Relation,
)
from xgen_ontology_build.postbuild.finalize import LEGACY_RELATED_PREDICATE
from xgen_ontology_build.postbuild.taxonomy import DEFAULT_RELATED_PREDICATE
from xgen_ontology_build.text.korean import normalize_label
from xgen_ontology_build.text.tokens import tokenize

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
        " edge_weight REAL, label TEXT, UNIQUE (collection_id, subject_uri, predicate, object_uri))",
        "ALTER TABLE ontology_edges ADD COLUMN IF NOT EXISTS label TEXT",
        "CREATE TABLE IF NOT EXISTS ontology_edge_sources (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " subject_uri TEXT NOT NULL, predicate TEXT NOT NULL, object_uri TEXT NOT NULL, source_type TEXT NOT NULL,"
        " source_id TEXT NOT NULL,"
        " UNIQUE (collection_id, subject_uri, predicate, object_uri, source_type, source_id))",
        "CREATE TABLE IF NOT EXISTS ontology_node_chunks (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, chunk_id TEXT NOT NULL, UNIQUE (collection_id, uri, chunk_id))",
        "CREATE TABLE IF NOT EXISTS ontology_enriched_chunks (id BIGSERIAL PRIMARY KEY, collection_id TEXT NOT NULL,"
        " chunk_id TEXT NOT NULL, UNIQUE (collection_id, chunk_id))",
        "CREATE INDEX IF NOT EXISTS ix_onodes_coll_label ON ontology_nodes (collection_id, label)",
        "CREATE INDEX IF NOT EXISTS ix_oedges_o ON ontology_edges (collection_id, object_uri)",
        "CREATE INDEX IF NOT EXISTS ix_oedges_po ON ontology_edges (collection_id, predicate, object_uri)",
        "CREATE INDEX IF NOT EXISTS ix_onc_coll_chunk ON ontology_node_chunks (collection_id, chunk_id)",
        "CREATE INDEX IF NOT EXISTS ix_onc_coll_uri ON ontology_node_chunks (collection_id, uri)",
        "CREATE INDEX IF NOT EXISTS ix_oec_coll ON ontology_enriched_chunks (collection_id)",
        "CREATE INDEX IF NOT EXISTS ix_oes_coll_source ON ontology_edge_sources (collection_id, source_id)",
        "CREATE TABLE IF NOT EXISTS ontology_schema (id SERIAL PRIMARY KEY, collection_id VARCHAR(255) NOT NULL,"
        " element_name VARCHAR(500) NOT NULL, element_type VARCHAR(50) NOT NULL, description TEXT,"
        " domain_class VARCHAR(500), range_value VARCHAR(500), parent_class VARCHAR(500), source VARCHAR(20),"
        " is_auto_generated BOOLEAN DEFAULT TRUE, is_confirmed BOOLEAN DEFAULT FALSE)",
        "ALTER TABLE ontology_schema ADD COLUMN IF NOT EXISTS is_auto_generated BOOLEAN DEFAULT TRUE",
        "ALTER TABLE ontology_schema ADD COLUMN IF NOT EXISTS is_confirmed BOOLEAN DEFAULT FALSE",
    ],
    "sqlite": [
        "CREATE TABLE IF NOT EXISTS ontology_nodes (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " uri TEXT NOT NULL, kind TEXT NOT NULL, label TEXT, attrs TEXT, nkey TEXT, UNIQUE (collection_id, uri))",
        "CREATE TABLE IF NOT EXISTS ontology_edges (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " subject_uri TEXT NOT NULL, predicate TEXT, object_uri TEXT, predicate_uri TEXT, edge_kind TEXT,"
        " edge_weight REAL, label TEXT, UNIQUE (collection_id, subject_uri, predicate, object_uri))",
        "CREATE TABLE IF NOT EXISTS ontology_edge_sources (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " subject_uri TEXT NOT NULL, predicate TEXT NOT NULL, object_uri TEXT NOT NULL, source_type TEXT NOT NULL,"
        " source_id TEXT NOT NULL,"
        " UNIQUE (collection_id, subject_uri, predicate, object_uri, source_type, source_id))",
        "CREATE TABLE IF NOT EXISTS ontology_schema (id INTEGER PRIMARY KEY, collection_id TEXT NOT NULL,"
        " element_name TEXT NOT NULL, element_type TEXT NOT NULL, description TEXT, domain_class TEXT,"
        " range_value TEXT, parent_class TEXT, source TEXT, is_auto_generated INTEGER DEFAULT 1,"
        " is_confirmed INTEGER DEFAULT 0)",
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


#: the URI prefix of a database row: its identity (``Instance.identity``, the production loader's key)
IDENTITY_PREFIX = INSTANCE_NS + "dbrow_"


def _identity_of(uri: str) -> str:
    return uri[len(INSTANCE_NS):] if uri.startswith(IDENTITY_PREFIX) else ""


def _uri_fns(ontology, uris: dict | None = None):
    """URI makers for an ontology's nodes: a node the store already holds keeps its URI
    (``ontology.uris`` and ``uris``, ``(kind, label) -> uri``), a head promoted to a class keeps
    the instance URI it was stored under (the product's promotion changes the kind, not the
    URI), and only a node the store has not seen gets a URI from its name."""
    known: dict = dict(getattr(ontology, "uris", None) or {})
    if uris:
        known.update(uris)
    inst_names = {i.name for i in ontology.instances if i.name}
    # a row's URI is its identity, whatever it is named now and whatever a store held it under
    identity = {i.name: i.identity for i in ontology.instances if i.name and i.identity}

    def cu(name: str) -> str:
        u = known.get(("concept", name))
        if u is None and name not in inst_names:
            u = known.get(("instance", name))
        return u or class_uri(name)

    def iu(name: str) -> str:
        ident = identity.get(name)
        if ident:
            return INSTANCE_NS + ident
        return known.get(("instance", name)) or instance_uri(name)

    return cu, iu


def graph_rows(ontology, *, translations: dict[str, str] | None = None, uris: dict | None = None
               ) -> tuple[list[tuple], list[tuple], list[tuple[str, str]]]:
    """An ontology as the store's rows: ``(nodes, edges, node_chunks)``.

    ``nodes`` are ``(uri, kind, label, attrs)`` with ``kind`` in concept / property /
    instance and ``attrs`` the node's data values ``{property: [values]}``; ``edges``
    are ``(subject_uri, predicate, object_uri, predicate_uri, edge_kind, weight, label)``
    with the production edge kinds; ``node_chunks`` are ``(uri, chunk_id)``. Same shape
    the production ``serialize_graph`` produces from an equivalent build. The chunks
    each relation was stated in are :func:`edge_source_rows`.

    A node keeps the URI a store holds it under (``ontology.uris``, filled by
    :meth:`PgGraph.load`, and ``uris``); a node the store has not seen gets one from its
    name. A predicate IRI is the predicate itself: relation names are English identifiers
    (``normalize_graph``). ``translations`` is accepted for compatibility and not used
    here; :meth:`Ontology.translate <xgen_ontology.Ontology.translate>` still names the
    RDF export.
    """
    del translations
    c = ontology.concepts
    cu, iu = _uri_fns(ontology, uris)

    def pred_uri(name: str) -> str:
        return DOMAIN_NS + name

    nodes: list[tuple] = []
    edges: list[tuple] = []
    chunks: list[tuple[str, str]] = []
    class_names = {cl.name for cl in c.classes if cl.name}
    for cl in c.classes:
        if not cl.name:
            continue
        nodes.append((cu(cl.name), "concept", cl.name, None))
        chunks += [(cu(cl.name), cid) for cid in cl.source_chunks if cid]
    for dp in c.datatype_properties:
        if not dp.name:
            continue
        puri = DOMAIN_NS + (f"{dp.domain}_{dp.name}" if dp.domain else dp.name)
        nodes.append((puri, "property", dp.name, None))
        if dp.domain and dp.domain in class_names:
            edges.append((cu(dp.domain), dp.name, puri, pred_uri(dp.name), "datatypeProperty_schema", 1.0,
                          None))
    for op in c.object_properties:
        if op.name and op.domain in class_names and op.range in class_names:
            edges.append((cu(op.domain), op.name, cu(op.range), pred_uri(op.name),
                          "objectProperty_schema", 1.0, None))
    for parent, child in c.class_hierarchy:
        if parent in class_names and child in class_names and parent != child:
            edges.append((cu(child), "subClassOf", cu(parent), DOMAIN_NS + "subClassOf",
                          "subClassOf", 1.0, None))

    inst_names: set[str] = set()
    for i in ontology.instances:
        if not i.name:
            continue
        u = iu(i.name)
        if i.name not in inst_names:
            nodes.append((u, "instance", i.name, None))
            inst_names.add(i.name)
        if i.class_name and i.class_name in class_names:
            edges.append((u, "instanceOf", cu(i.class_name), DOMAIN_NS + "instanceOf", "instanceOf", 1.0,
                          None))
        chunks += [(u, cid) for cid in i.source_chunks if cid]

    def endpoint(name: str) -> str:
        return iu(name) if name in inst_names or name not in class_names else cu(name)

    for r in ontology.relations:
        if not (r.subject and r.predicate and r.object) or r.predicate_type == "DatatypeProperty":
            continue
        for end in (r.subject, r.object):
            if end not in inst_names and end not in class_names:
                nodes.append((iu(end), "instance", end, None))
                inst_names.add(end)
        edges.append((endpoint(r.subject), r.predicate, endpoint(r.object), pred_uri(r.predicate),
                      "objectProperty", float(r.weight or 1.0), r.label or None))
        chunks += [(endpoint(r.subject), cid) for cid in r.source_chunks if cid]
        chunks += [(endpoint(r.object), cid) for cid in r.source_chunks if cid]

    attrs: dict[str, dict[str, list[str]]] = {}
    for d in ontology.data_values:
        if not (d.entity and d.property) or d.value is None:
            continue
        if d.entity not in inst_names and d.entity not in class_names:
            nodes.append((iu(d.entity), "instance", d.entity, None))
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


def edge_source_rows(ontology, *, uris: dict | None = None) -> list[tuple[str, str, str, str]]:
    """The chunks each relation was stated in, as the store keeps them: ``(subject_uri, predicate,
    object_uri, chunk_id)``. Same endpoints as :func:`graph_rows`."""
    class_names = {cl.name for cl in ontology.concepts.classes if cl.name}
    inst_names = {i.name for i in ontology.instances if i.name}
    cu, iu = _uri_fns(ontology, uris)

    def endpoint(name: str) -> str:
        return iu(name) if name in inst_names or name not in class_names else cu(name)

    out: set[tuple[str, str, str, str]] = set()
    for r in ontology.relations:
        if not (r.subject and r.predicate and r.object) or r.predicate_type == "DatatypeProperty":
            continue
        su, ou = endpoint(r.subject), endpoint(r.object)
        out.update((su, r.predicate, ou, str(cid)) for cid in r.source_chunks if cid)
    return sorted(out)


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

    def _x(self, sql: str, params: tuple = ()) -> int:
        cur = self.conn.cursor()
        cur.execute(sql.replace("?", self._ph), params)
        try:
            return max(0, int(cur.rowcount or 0))
        finally:
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
        """Create the tables (and the read indexes) when they do not exist. Tables made by an
        earlier version get the edge ``label`` column and the relation-sources table."""
        for ddl in _DDL[self.dialect]:
            self._x(ddl)
        if self.dialect == "sqlite":
            if "label" not in {r[1] for r in self._q("PRAGMA table_info(ontology_edges)")}:
                self._x("ALTER TABLE ontology_edges ADD COLUMN label TEXT")
            schema_cols = {r[1] for r in self._q("PRAGMA table_info(ontology_schema)")}
            if "is_auto_generated" not in schema_cols:
                self._x("ALTER TABLE ontology_schema ADD COLUMN is_auto_generated INTEGER DEFAULT 1")
            if "is_confirmed" not in schema_cols:
                self._x("ALTER TABLE ontology_schema ADD COLUMN is_confirmed INTEGER DEFAULT 0")

    # ── sink ──

    def write(self, ontology, *, replace: bool = True, translations: dict[str, str] | None = None) -> dict:
        """Store ``ontology``. ``replace`` clears the collection first; otherwise rows are added and a
        node seen again merges its attributes (the incremental path).

        ``ontology`` is the whole graph (an incremental build extends a loaded one), so an edge
        seen again takes the model's weight, not one more, and keeps its first label. Relations a
        named relation superseded (``normalize_graph``) leave the table too, and links still under
        the former neighbour name (``관련``) are renamed to the current one. A node the store
        already holds keeps its URI (``ontology.uris`` from :meth:`load`, and on an appending
        write the rows as they are now), so a name whose kind or spelling the build changed is
        the same row, not a second one. A database row (URI under :data:`IDENTITY_PREFIX`) is
        written as the table has it now: its attributes and label are replaced, not merged, so a
        value that became NULL leaves the node. Its former relations are taken out by
        :meth:`detach_chunks` before the write.
        """
        del translations
        known_uris = dict(getattr(ontology, "uris", None) or {})
        if not replace:
            known_uris.update(self.label_uris())
        nodes, edges, chunks = graph_rows(ontology, uris=known_uris)
        sources = edge_source_rows(ontology, uris=known_uris)
        C = self.collection_id
        is_row = {u for u, _k, _l, _a in nodes if u.startswith(IDENTITY_PREFIX)}
        if replace:
            self.clear()
            merged = {u: a for u, _k, _l, a in nodes if a}
        else:
            merged = {}
            with_attrs = [u for u, _k, _l, a in nodes if a and u not in is_row]
            for i in range(0, len(with_attrs), 500):
                part = with_attrs[i:i + 500]
                rows = self._q("SELECT uri, attrs FROM ontology_nodes WHERE collection_id=? AND uri IN ("
                               + ",".join(["?"] * len(part)) + ")", (C, *part))
                for u, a in rows:
                    merged[u] = _attrs(a)
            for u, _k, _l, a in nodes:
                if u in is_row:
                    merged[u] = a or {}
                elif a:
                    cur = merged.setdefault(u, {})
                    for k, vs in a.items():
                        have = cur.setdefault(k, [])
                        have.extend(v for v in vs if v not in have)
        node_rows = [(C, u, k, lb, json.dumps(merged.get(u), ensure_ascii=False) if merged.get(u) else None)
                     for u, k, lb, _a in nodes]
        self._insert("INSERT INTO ontology_nodes(collection_id, uri, kind, label, attrs)", 5,
                     [r for r in node_rows if r[1] not in is_row],
                     "ON CONFLICT (collection_id, uri) DO UPDATE SET kind = excluded.kind,"
                     " attrs = COALESCE(excluded.attrs, ontology_nodes.attrs)")
        self._insert("INSERT INTO ontology_nodes(collection_id, uri, kind, label, attrs)", 5,
                     [r for r in node_rows if r[1] in is_row],
                     "ON CONFLICT (collection_id, uri) DO UPDATE SET kind = excluded.kind,"
                     " label = excluded.label, attrs = excluded.attrs")
        if not replace:
            self._rename_legacy_related()
        self._insert("INSERT INTO ontology_edges(collection_id, subject_uri, predicate, object_uri, predicate_uri,"
                     " edge_kind, edge_weight, label)", 8, [(C, *e) for e in edges],
                     "ON CONFLICT (collection_id, subject_uri, predicate, object_uri) DO UPDATE SET"
                     " edge_weight = excluded.edge_weight, label = COALESCE(ontology_edges.label, excluded.label)")
        self._insert("INSERT INTO ontology_node_chunks(collection_id, uri, chunk_id)", 3,
                     [(C, u, cid) for u, cid in chunks], "ON CONFLICT (collection_id, uri, chunk_id) DO NOTHING")
        self._insert("INSERT INTO ontology_edge_sources(collection_id, subject_uri, predicate, object_uri,"
                     " source_type, source_id)", 6, [(C, su, p, ou, "chunk", cid) for su, p, ou, cid in sources],
                     "ON CONFLICT (collection_id, subject_uri, predicate, object_uri, source_type, source_id)"
                     " DO NOTHING")
        if not replace:
            self._drop_superseded(ontology, known_uris)
        self.write_schema(ontology.concepts)
        if ontology.enriched_chunks:
            self.mark_enriched(ontology.enriched_chunks)
        return {"nodes": len(nodes), "edges": len(edges), "node_chunks": len(chunks), "edge_sources": len(sources)}

    def write_schema(self, concepts) -> int:
        """Keep the schema where the product keeps it, in ``ontology_schema``: every class (its
        description, parent and, for a class built from a table, ``source='csv'``, which the
        product's dedupe reads as a deterministic identity), every object property (the relation
        vocabulary with its definitions, and the table and document relation names) and every
        datatype property with its range. So the next build, the product's or the library's,
        sees the same schema. A row already there gets a description or source it lacks;
        nothing is removed. Returns rows added."""
        C = self.collection_id
        parent_of = {child: parent for parent, child in concepts.class_hierarchy
                     if parent and child and parent != child}
        rows: dict[tuple[str, str], tuple] = {}
        for cl in concepts.classes:
            if cl.name and ("Class", cl.name) not in rows:
                rows[("Class", cl.name)] = (cl.description or "", None, None, cl.parent or parent_of.get(cl.name),
                                            "csv" if cl.source == "table" else None)
        for op in concepts.object_properties:
            if op.name and ("ObjectProperty", op.name) not in rows:
                rows[("ObjectProperty", op.name)] = (op.description or "", op.domain or None, op.range or None,
                                                     None, None)
        for dp in concepts.datatype_properties:
            if dp.name and ("DatatypeProperty", dp.name) not in rows:
                rows[("DatatypeProperty", dp.name)] = ("", dp.domain or None, dp.range or "xsd:string", None, None)
        if not rows:
            return 0
        have = {(t, n): (d, s) for n, t, d, s in self._q(
            "SELECT element_name, element_type, description, source FROM ontology_schema WHERE collection_id=?", (C,))}
        new = [(C, n, t, d, dom, rng, par, src) for (t, n), (d, dom, rng, par, src) in rows.items()
               if (t, n) not in have]
        if new:
            self._insert("INSERT INTO ontology_schema(collection_id, element_name, element_type, description,"
                         " domain_class, range_value, parent_class, source)", 8, new, "")
        for (t, n), (d, _dom, _rng, _par, src) in rows.items():
            if (t, n) not in have:
                continue
            had_desc, had_src = have[(t, n)]
            if d and not had_desc:
                self._x("UPDATE ontology_schema SET description=? WHERE collection_id=? AND element_type=?"
                        " AND element_name=?", (d, C, t, n))
            if src and not had_src:
                self._x("UPDATE ontology_schema SET source=? WHERE collection_id=? AND element_type=?"
                        " AND element_name=?", (src, C, t, n))
        return len(new)

    def write_vocabulary(self, object_properties) -> int:
        """The relation vocabulary alone, as ``ObjectProperty`` rows (:meth:`write_schema` writes it
        with the rest of the schema). Returns rows added."""
        return self.write_schema(Concepts(object_properties=[op for op in (object_properties or []) if op.name]))

    def label_uris(self) -> dict[tuple[str, str], str]:
        """The stored nodes' URIs by ``(kind, label)``, concepts and instances (the first row when
        two share a kind and label): what :func:`graph_rows` keeps on an appending write."""
        out: dict[tuple[str, str], str] = {}
        for u, k, lb in self._q("SELECT uri, kind, label FROM ontology_nodes WHERE collection_id=?"
                                " AND kind IN ('concept', 'instance') ORDER BY id", (self.collection_id,)):
            if lb and (k, lb) not in out:
                out[(k, lb)] = u
        return out

    def _rename_legacy_related(self) -> None:
        """Links stored under the former neighbour name take the current one (a duplicate goes first)."""
        C = self.collection_id
        self._x("DELETE FROM ontology_edges WHERE collection_id=? AND predicate=? AND edge_kind='objectProperty'"
                " AND EXISTS (SELECT 1 FROM ontology_edges c WHERE c.collection_id=ontology_edges.collection_id"
                "   AND c.predicate=? AND c.subject_uri=ontology_edges.subject_uri"
                "   AND c.object_uri=ontology_edges.object_uri)",
                (C, LEGACY_RELATED_PREDICATE, DEFAULT_RELATED_PREDICATE))
        self._x("UPDATE ontology_edges SET predicate=?, predicate_uri=? WHERE collection_id=? AND predicate=?"
                " AND edge_kind='objectProperty'",
                (DEFAULT_RELATED_PREDICATE, DOMAIN_NS + DEFAULT_RELATED_PREDICATE, C, LEGACY_RELATED_PREDICATE))

    def _drop_superseded(self, ontology, uris: dict | None = None) -> None:
        """On an incremental write: a stored relatedTo between two nodes the model links with a named
        relation was superseded; it leaves with its sources (the model no longer holds it)."""
        held = {(r.subject, r.object) for r in ontology.relations if r.predicate == DEFAULT_RELATED_PREDICATE}
        _n, edges, _c = graph_rows(ontology, uris=uris)
        named = sorted({(su, ou) for su, p, ou, _pu, ek, _w, _lb in edges
                        if ek == "objectProperty" and p != DEFAULT_RELATED_PREDICATE})
        if not named:
            return
        cu, iu = _uri_fns(ontology, uris)
        label_of = {cu(cl.name): cl.name for cl in ontology.concepts.classes if cl.name}
        label_of.update({iu(i.name): i.name for i in ontology.instances if i.name})
        C = self.collection_id
        for su, ou in named:
            for a, b in ((su, ou), (ou, su)):
                if (label_of.get(a), label_of.get(b)) in held:
                    continue
                for t in ("ontology_edge_sources", "ontology_edges"):
                    self._x(f"DELETE FROM {t} WHERE collection_id=? AND subject_uri=? AND predicate=?"
                            " AND object_uri=?", (C, a, DEFAULT_RELATED_PREDICATE, b))

    def clear(self) -> None:
        """Remove the collection's graph, its relation sources and enrich markers, and its schema
        rows (a rebuild starts its vocabulary over, as the product's does)."""
        for t in ("ontology_edge_sources", "ontology_edges", "ontology_nodes", "ontology_node_chunks",
                  "ontology_enriched_chunks", "ontology_schema"):
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

    def detach_chunks(self, chunk_ids, *, attr_keys=()) -> dict:
        """Take the traces of rows that are about to be written again out of the tables: the
        relations those chunks stated (gone when no other chunk states them, kept with their
        other sources otherwise) and, on the nodes linked to the chunks, the attribute keys in
        ``attr_keys`` (a row's columns). The nodes and their links stay: :meth:`write` with
        ``replace=False`` then puts the rows back as they are now. A row node keeps nothing from
        the write anyway (its attributes are replaced); ``attr_keys`` is for nodes that are not
        rows. Returns ``chunks / sources / edges / nodes`` counts. Several statements: one
        transaction, or an autocommit connection (an interrupted call leaves nothing the next
        one cannot redo)."""
        ids = sorted({str(c) for c in (chunk_ids or []) if c})
        keys = sorted({str(k) for k in (attr_keys or []) if k})
        stats = {"chunks": len(ids), "sources": 0, "edges": 0, "nodes": 0}
        if not ids:
            return stats
        C = self.collection_id
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            ph = ",".join(["?"] * len(part))
            triples = self._q("SELECT DISTINCT subject_uri, predicate, object_uri FROM ontology_edge_sources"
                              f" WHERE collection_id=? AND source_type='chunk' AND source_id IN ({ph})", (C, *part))
            stats["sources"] += self._x("DELETE FROM ontology_edge_sources WHERE collection_id=?"
                                        f" AND source_type='chunk' AND source_id IN ({ph})", (C, *part))
            for s, p, o in triples:
                stats["edges"] += self._x(
                    "DELETE FROM ontology_edges WHERE collection_id=? AND subject_uri=? AND predicate=? AND object_uri=?"
                    " AND edge_kind='objectProperty' AND NOT EXISTS (SELECT 1 FROM ontology_edge_sources x"
                    "   WHERE x.collection_id=ontology_edges.collection_id AND x.subject_uri=ontology_edges.subject_uri"
                    "     AND x.predicate=ontology_edges.predicate AND x.object_uri=ontology_edges.object_uri)",
                    (C, s, p, o))
            if keys:
                uris = [r[0] for r in self._q("SELECT DISTINCT uri FROM ontology_node_chunks"
                                              f" WHERE collection_id=? AND chunk_id IN ({ph})", (C, *part))]
                for j in range(0, len(uris), 500):
                    upart = uris[j:j + 500]
                    uph = ",".join(["?"] * len(upart))
                    if self.dialect == "sqlite":
                        paths = ",".join(["?"] * len(keys))
                        stats["nodes"] += self._x(
                            f"UPDATE ontology_nodes SET attrs = json_remove(attrs, {paths})"
                            f" WHERE collection_id=? AND uri IN ({uph}) AND attrs IS NOT NULL",
                            (*[f'$."{k}"' for k in keys], C, *upart))
                    else:
                        stats["nodes"] += self._x(
                            "UPDATE ontology_nodes SET attrs = attrs - ?::text[]"
                            f" WHERE collection_id=? AND uri IN ({uph}) AND attrs IS NOT NULL",
                            (keys, C, *upart))
        return stats

    def remove_rows(self, chunk_ids, *, attr_keys=(), **prune) -> dict:
        """Rows that left the table: their traces go (:meth:`detach_chunks`) and then their chunks
        (:meth:`prune_chunks`), so a row node nothing else links to leaves with its relations,
        while a node a document also links to stays with that evidence only. ``prune`` is passed
        to :meth:`prune_chunks`. Returns the two counts dicts merged."""
        out = self.detach_chunks(chunk_ids, attr_keys=attr_keys)
        pruned = self.prune_chunks(chunk_ids, **prune)
        for k, v in pruned.items():
            out[k] = out.get(k, 0) + v if k != "chunks" else v
        return out

    def prune_chunks(self, chunk_ids, *,
                     structural_predicates=(DEFAULT_RELATED_PREDICATE, LEGACY_RELATED_PREDICATE, "sameAs")) -> dict:
        """Take deleted chunks out of the stored graph: the production deletion delta, on the tables.

        Removes the chunks' links and the nodes they were the only evidence for: individuals
        with no link left, classes with no link left that nothing refers to any more (their
        own property declarations and parent link are not references), the property nodes
        only those classes declared, and every edge touching them. A relation with stated
        source chunks goes when all of them are removed and otherwise keeps the rest, their
        count as its weight; a
        relation with none between two surviving nodes that no longer share a chunk has lost
        its evidence and goes too, except relations made from name structure
        (``structural_predicates``). The chunks' enrich markers go so a later relation pass
        does not wait on them. Same rules as
        :func:`~xgen_ontology.build.retract.retract_chunks` on the build models.

        Several statements; run them in one transaction (a non-autocommit connection, then
        commit). Their order still makes a retry safe on an autocommit connection: the links
        are removed last, so an interrupted run leaves them in place and the next call finds
        the same nodes again. Returns ``chunks / links / nodes / edges / enriched`` counts.
        """
        ids = sorted({str(c) for c in (chunk_ids or []) if c})
        stats = {"chunks": len(ids), "links": 0, "nodes": 0, "edges": 0, "enriched": 0, "sources": 0}
        if not ids:
            return stats
        C = self.collection_id
        gone = "(SELECT chunk_id FROM xo_prune_chunks)"
        cand = "(SELECT uri FROM xo_prune_cand)"
        gone_uris = "(SELECT uri FROM xo_prune_gone)"
        for t, col in (("xo_prune_chunks", "chunk_id"), ("xo_prune_cand", "uri"), ("xo_prune_gone", "uri")):
            self._x(f"CREATE TEMP TABLE IF NOT EXISTS {t} ({col} TEXT PRIMARY KEY)")
            self._x(f"DELETE FROM {t}")
        try:
            self._insert("INSERT INTO xo_prune_chunks(chunk_id)", 1, [(c,) for c in ids],
                         "ON CONFLICT (chunk_id) DO NOTHING")
            stats["links"] = self._q(f"SELECT count(*) FROM ontology_node_chunks WHERE collection_id=?"
                                     f" AND chunk_id IN {gone}", (C,))[0][0]
            stats["enriched"] = self._x(
                f"DELETE FROM ontology_enriched_chunks WHERE collection_id=? AND chunk_id IN {gone}", (C,))
            if not stats["links"]:
                stats["sources"] = self._x(f"DELETE FROM ontology_edge_sources WHERE collection_id=?"
                                           f" AND source_type='chunk' AND source_id IN {gone}", (C,))
                return stats
            # the nodes that lose a link are the candidates of the first round
            self._x("INSERT INTO xo_prune_cand(uri) SELECT DISTINCT uri FROM ontology_node_chunks"
                    f" WHERE collection_id=? AND chunk_id IN {gone}", (C,))
            # a relation with stated sources: gone when every one of them is removed
            same_edge = ("s.collection_id=ontology_edges.collection_id AND s.subject_uri=ontology_edges.subject_uri"
                         " AND s.predicate=ontology_edges.predicate AND s.object_uri=ontology_edges.object_uri")
            stats["edges"] = self._x(
                "DELETE FROM ontology_edges WHERE collection_id=? AND edge_kind='objectProperty'"
                f" AND EXISTS (SELECT 1 FROM ontology_edge_sources s WHERE {same_edge})"
                f" AND NOT EXISTS (SELECT 1 FROM ontology_edge_sources s WHERE {same_edge}"
                f"                  AND NOT (s.source_type='chunk' AND s.source_id IN {gone}))", (C,))
            # a relation without: its two ends survive but no longer share a chunk outside the removed ones
            ph = ",".join(["?"] * len(structural_predicates)) or "''"
            stats["edges"] += self._x(
                "DELETE FROM ontology_edges WHERE collection_id=? AND edge_kind='objectProperty'"
                f" AND predicate NOT IN ({ph}) AND subject_uri IN {cand} AND object_uri IN {cand}"
                f" AND NOT EXISTS (SELECT 1 FROM ontology_edge_sources s WHERE {same_edge})"
                " AND NOT EXISTS (SELECT 1 FROM ontology_node_chunks a"
                "                  JOIN ontology_node_chunks b ON b.collection_id=a.collection_id"
                "                   AND b.chunk_id=a.chunk_id AND b.uri=ontology_edges.object_uri"
                "                  WHERE a.collection_id=ontology_edges.collection_id"
                f"                   AND a.uri=ontology_edges.subject_uri AND a.chunk_id NOT IN {gone})",
                (C, *structural_predicates))
            # a relation that keeps some of its sources: its weight is the evidence left
            self._x("UPDATE ontology_edges SET edge_weight = (SELECT count(*) FROM ontology_edge_sources s"
                    f"   WHERE {same_edge} AND NOT (s.source_type='chunk' AND s.source_id IN {gone}))"
                    " WHERE collection_id=? AND edge_kind='objectProperty'"
                    f" AND EXISTS (SELECT 1 FROM ontology_edge_sources s WHERE {same_edge}"
                    f"             AND s.source_type='chunk' AND s.source_id IN {gone})", (C,))
            # the removed chunks are nobody's source any more (after the judgement above, which needs them)
            stats["sources"] = self._x(f"DELETE FROM ontology_edge_sources WHERE collection_id=?"
                                       f" AND source_type='chunk' AND source_id IN {gone}", (C,))
            # nodes the removed chunks were the only evidence for; then, round by round, the classes
            # whose last reference was one of them (a class referenced by nothing, linked to no chunk)
            no_link_left = ("NOT EXISTS (SELECT 1 FROM ontology_node_chunks c WHERE c.collection_id=n.collection_id"
                            f" AND c.uri=n.uri AND c.chunk_id NOT IN {gone})")
            orphan_sql = (
                "WITH orphan_inst AS (SELECT n.uri FROM ontology_nodes n WHERE n.collection_id=? AND n.kind='instance'"
                f"   AND n.uri IN {cand} AND {no_link_left}),"
                " orphan_cls AS (SELECT n.uri FROM ontology_nodes n WHERE n.collection_id=? AND n.kind='concept'"
                f"   AND n.uri IN {cand} AND {no_link_left}"
                "   AND NOT EXISTS (SELECT 1 FROM ontology_edges e WHERE e.collection_id=n.collection_id"
                "        AND (e.subject_uri=n.uri OR e.object_uri=n.uri)"
                "        AND NOT (e.subject_uri=n.uri AND e.edge_kind IN"
                "                 ('datatypeProperty_schema', 'objectProperty_schema', 'subClassOf'))"
                "        AND e.subject_uri NOT IN (SELECT uri FROM orphan_inst)"
                "        AND e.object_uri NOT IN (SELECT uri FROM orphan_inst))),"
                " orphan_prop AS (SELECT n.uri FROM ontology_nodes n WHERE n.collection_id=? AND n.kind='property'"
                "   AND EXISTS (SELECT 1 FROM ontology_edges e WHERE e.collection_id=n.collection_id"
                "        AND e.object_uri=n.uri AND e.subject_uri IN (SELECT uri FROM orphan_cls))"
                "   AND NOT EXISTS (SELECT 1 FROM ontology_edges e WHERE e.collection_id=n.collection_id"
                "        AND e.object_uri=n.uri AND e.subject_uri NOT IN (SELECT uri FROM orphan_cls)))"
                " SELECT uri FROM orphan_inst UNION SELECT uri FROM orphan_cls UNION SELECT uri FROM orphan_prop")
            while True:
                orphans = sorted({r[0] for r in self._q(orphan_sql, (C, C, C))})
                if not orphans:
                    break
                self._x("DELETE FROM xo_prune_gone")
                self._insert("INSERT INTO xo_prune_gone(uri)", 1, [(u,) for u in orphans],
                             "ON CONFLICT (uri) DO NOTHING")
                # the classes these nodes pointed at are the next round's candidates
                self._x("DELETE FROM xo_prune_cand")
                self._x("INSERT INTO xo_prune_cand(uri) SELECT DISTINCT e.object_uri FROM ontology_edges e"
                        " JOIN ontology_nodes n ON n.collection_id=e.collection_id AND n.uri=e.object_uri"
                        f" AND n.kind='concept' WHERE e.collection_id=? AND e.subject_uri IN {gone_uris}"
                        f" AND e.object_uri NOT IN {gone_uris}", (C,))
                stats["edges"] += self._x(
                    f"DELETE FROM ontology_edges WHERE collection_id=? AND (subject_uri IN {gone_uris}"
                    f" OR object_uri IN {gone_uris})", (C,))
                stats["nodes"] += self._x(
                    f"DELETE FROM ontology_nodes WHERE collection_id=? AND uri IN {gone_uris}", (C,))
            self._x(f"DELETE FROM ontology_node_chunks WHERE collection_id=? AND chunk_id IN {gone}", (C,))
            # sources of the edges that went with an orphan node
            self._x("DELETE FROM ontology_edge_sources WHERE collection_id=? AND NOT EXISTS"
                    " (SELECT 1 FROM ontology_edges e WHERE e.collection_id=ontology_edge_sources.collection_id"
                    "   AND e.subject_uri=ontology_edge_sources.subject_uri AND e.predicate=ontology_edge_sources.predicate"
                    "   AND e.object_uri=ontology_edge_sources.object_uri)", (C,))
        finally:
            for t in ("xo_prune_chunks", "xo_prune_cand", "xo_prune_gone"):
                try:
                    self._x(f"DROP TABLE IF EXISTS {t}")
                except Exception:
                    pass   # an aborted transaction refuses it; a temp table dies with the session anyway
        return stats

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
        from xgen_ontology_build.ontology import Ontology

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
        sources: dict[tuple[str, str, str], list[str]] = {}
        for su, p, ou, cid in self._q("SELECT subject_uri, predicate, object_uri, source_id FROM ontology_edge_sources"
                                      " WHERE collection_id=? AND source_type='chunk' ORDER BY id", (C,)):
            sources.setdefault((su, p, ou), []).append(cid)
        edges = self._q("SELECT subject_uri, predicate, object_uri, edge_kind, edge_weight, label FROM ontology_edges"
                        " WHERE collection_id=? ORDER BY id", (C,))
        for s, p, o, ek, w, lb in edges:
            if ek == "subClassOf":
                concepts.class_hierarchy.append((label.get(o, ""), label.get(s, "")))
            elif ek == "instanceOf":
                instances.append(Instance(name=label.get(s, ""), class_name=label.get(o, ""),
                                          source_chunks=chunks_of.get(s, []), identity=_identity_of(s)))
                typed.add(s)
            elif ek == "objectProperty_schema":
                concepts.object_properties.append(ObjectProperty(name=p, domain=label.get(s, ""), range=label.get(o, "")))
            elif ek == "datatypeProperty_schema":
                concepts.datatype_properties.append(DataProperty(name=p, domain=label.get(s, "")))
            elif ek == "objectProperty":
                relations.append(Relation(subject=label.get(s, ""),
                                          predicate=DEFAULT_RELATED_PREDICATE if p == LEGACY_RELATED_PREDICATE else p,
                                          object=label.get(o, ""), source_chunks=sources.get((s, p, o), []),
                                          label=lb or None, weight=float(w) if w is not None else 1.0))
        for u in label:
            if kind[u] == "instance" and u not in typed and label[u]:
                instances.append(Instance(name=label[u], source_chunks=chunks_of.get(u, []), identity=_identity_of(u)))
        # the relation vocabulary: names with their definitions
        declared_op = {op.name: op for op in concepts.object_properties}
        for name, desc, dom, rng in self._q("SELECT element_name, description, domain_class, range_value"
                                            " FROM ontology_schema WHERE collection_id=?"
                                            " AND element_type='ObjectProperty' ORDER BY id", (C,)):
            if name in declared_op:
                declared_op[name].description = declared_op[name].description or (desc or "")
            elif name:
                op = ObjectProperty(name=name, domain=dom or "", range=rng or "", description=desc or "")
                concepts.object_properties.append(op)
                declared_op[name] = op
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
        # the rest of the schema rows: a class built from a table keeps its deterministic identity
        # (source csv / db) and its description; a datatype property its range
        cls_rows = {n: (d, s) for n, d, s in self._q("SELECT element_name, description, source FROM ontology_schema"
                                                     " WHERE collection_id=? AND element_type='Class' ORDER BY id",
                                                     (C,))}
        for cl in concepts.classes:
            desc, src = cls_rows.get(cl.name, ("", None))
            if src in ("csv", "db"):
                cl.source = "table"
            cl.description = cl.description or (desc or "")
        ranges = {n: r for n, r in self._q("SELECT element_name, range_value FROM ontology_schema"
                                           " WHERE collection_id=? AND element_type='DatatypeProperty' ORDER BY id",
                                           (C,)) if r}
        for dp in concepts.datatype_properties:
            dp.range = ranges.get(dp.name, dp.range)
        all_chunks = sorted({cid for cids in chunks_of.values() for cid in cids})
        uris = {}
        for u in label:
            if kind[u] in ("concept", "instance") and label[u] and (kind[u], label[u]) not in uris:
                uris[(kind[u], label[u])] = u
        onto = Ontology(concepts=concepts, instances=instances, relations=relations, data_values=data_values,
                        chunks=[Chunk(id=cid, text="") for cid in all_chunks],
                        enriched_chunks=sorted(self.enriched_chunk_ids()), uris=uris)
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
