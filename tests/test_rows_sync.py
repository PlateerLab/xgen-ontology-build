"""Database rows kept in step with their table: identities, fingerprints, replaced and retracted rows,
in memory and on the graph tables."""
import sqlite3

from xgen_ontology_build import (
    IDENTITY_PREFIX,
    OntologyBuilder,
    PgGraph,
    diff_rows,
    graph_rows,
    mapping_signature,
    row_identity_key,
)

_CONTRACT_FK = [{"from_column": "cust_id", "to_table": "CUSTOMER", "to_column": "cust_id", "to_pk_column": "cust_id"}]
_COLL_FK = [{"from_column": "contract_no", "to_table": "LOAN_CONTRACT", "to_column": "contract_no",
             "to_pk_column": "contract_no"}]
_CONTRACTS = [{"contract_no": "L1", "cust_id": 100, "amt": 10}, {"contract_no": "L2", "cust_id": 101, "amt": 20}]
_COLLATERAL = [{"coll_id": 500000, "contract_no": "L1", "coll_type": "아파트"},
               {"coll_id": 500001, "contract_no": "L2", "coll_type": "토지"}]
_CUSTOMERS = [{"cust_id": 100, "cust_nm": "김민준"}, {"cust_id": 101, "cust_nm": "이서연"}]


def _two_tables():
    b = OntologyBuilder()
    o = b.build_rows("LOAN_CONTRACT", ["contract_no", "cust_id", "amt"], _CONTRACTS, source_id="db:1:LOAN_CONTRACT",
                     pk_candidates=["contract_no"], label_column="contract_no", fk_relations=_CONTRACT_FK)
    b.extend_rows(o, "COLLATERAL", ["coll_id", "contract_no", "coll_type"], _COLLATERAL, source_id="db:1:COLLATERAL",
                  pk_candidates=["coll_id"], label_column="coll_id", fk_relations=_COLL_FK)
    return b, o


def _rels(o):
    return {(r.subject, r.predicate, r.object) for r in o.relations}


def _vals(o, entity):
    return {(d.property, str(d.value)) for d in o.data_values if d.entity == entity}


def test_rows_named_by_a_numeric_key_keep_their_relations_and_values():
    b, o = _two_tables()
    assert ("500000", "collateralContractNo", "L1") in _rels(o)
    assert _vals(o, "500000") == {("coll_type", "아파트")}
    ident = {i.name: i.identity for i in o.instances}
    assert ident["500000"] == row_identity_key("db:1:COLLATERAL", "coll_id", "500000")
    # a key pointing at a row this load does not hold: a placeholder with the target row's identity
    assert ident["CUSTOMER_100"] == row_identity_key("db:1:CUSTOMER", "cust_id", "100")


def test_placeholder_becomes_the_row_when_its_table_arrives():
    b, o = _two_tables()
    b.extend_rows(o, "CUSTOMER", ["cust_id", "cust_nm"], _CUSTOMERS, source_id="db:1:CUSTOMER",
                  pk_candidates=["cust_id"], label_column="cust_nm")
    names = {i.name for i in o.instances}
    assert "김민준" in names and "CUSTOMER_100" not in names
    assert ("L1", "loanContractCustId", "김민준") in _rels(o)
    assert sum(1 for i in o.instances if i.identity == row_identity_key("db:1:CUSTOMER", "cust_id", "100")) == 1


def test_a_renamed_row_keeps_the_relations_other_rows_hold_to_it():
    b, o = _two_tables()
    b.extend_rows(o, "CUSTOMER", ["cust_id", "cust_nm"], _CUSTOMERS, source_id="db:1:CUSTOMER",
                  pk_candidates=["cust_id"], label_column="cust_nm")
    b.extend_rows(o, "CUSTOMER", ["cust_id", "cust_nm"], [{"cust_id": 100, "cust_nm": "김민준(개명)"}],
                  source_id="db:1:CUSTOMER", pk_candidates=["cust_id"], label_column="cust_nm")
    names = {i.name for i in o.instances}
    assert "김민준(개명)" in names and "김민준" not in names
    assert ("L1", "loanContractCustId", "김민준(개명)") in _rels(o)


def test_diff_rows_sees_values_keys_and_deletions():
    cols = ["id", "status", "note"]
    sig = mapping_signature(cols, "id", "status")
    first = diff_rows([{"id": 1, "status": "a", "note": "x"}, {"id": 2, "status": "b", "note": None}], cols, "id",
                      known={}, signature=sig)
    assert [r["id"] for r in first.added] == [1, 2] and not first.changed and not first.missing
    later = diff_rows([{"id": 1, "status": "a", "note": "x"}, {"id": 3, "status": "c", "note": None}], cols, "id",
                      known=first.fingerprints, signature=sig)
    assert later.unchanged == ["1"] and [r["id"] for r in later.added] == [3] and later.missing == ["2"]
    assert later.summary() == {"scanned": 2, "added": 1, "changed": 0, "unchanged": 1, "deleted": 1, "complete": True}
    # a value that became NULL is a change; an incomplete read judges no deletion
    nulled = diff_rows([{"id": 1, "status": "a", "note": None}], cols, "id", known=first.fingerprints, signature=sig,
                       complete=False)
    assert [r["id"] for r in nulled.changed] == [1] and nulled.missing == []
    # a different mapping: every row is a changed row
    remapped = diff_rows([{"id": 1, "status": "a", "note": "x"}], cols, "id", known=first.fingerprints,
                         signature=mapping_signature(cols, "id", "note"))
    assert [r["id"] for r in remapped.changed] == [1]


def test_sync_rows_replaces_changed_rows_and_retracts_missing_ones():
    b, o = _two_tables()
    schema = dict(pk_candidates=["coll_id"], label_column="coll_id", fk_relations=_COLL_FK)
    cols = ["coll_id", "contract_no", "coll_type"]
    first = b.sync_rows(o, "COLLATERAL", cols, _COLLATERAL, source_id="db:1:COLLATERAL", **schema)
    assert first.summary()["added"] == 2
    # a value changed, a key moved, a value NULLed, a row gone, a row new
    now = [{"coll_id": 500000, "contract_no": "L2", "coll_type": None},
           {"coll_id": 500002, "contract_no": "L1", "coll_type": "선박"}]
    diff = b.sync_rows(o, "COLLATERAL", cols, now, source_id="db:1:COLLATERAL",
                       fingerprints=first.fingerprints, **schema)
    assert diff.summary() == {"scanned": 2, "added": 1, "changed": 1, "unchanged": 0, "deleted": 1, "complete": True}
    rels = _rels(o)
    assert ("500000", "collateralContractNo", "L2") in rels and ("500000", "collateralContractNo", "L1") not in rels
    assert _vals(o, "500000") == set()                       # the NULLed value left
    assert ("500002", "collateralContractNo", "L1") in rels and _vals(o, "500002") == {("coll_type", "선박")}
    assert "500001" not in {i.name for i in o.instances}     # the deleted row left with its relation
    assert not [r for r in o.relations if r.subject == "500001"]
    # unchanged on the next read: nothing to load, nothing to retract
    again = b.sync_rows(o, "COLLATERAL", cols, now, source_id="db:1:COLLATERAL", fingerprints=diff.fingerprints, **schema)
    assert again.summary()["unchanged"] == 2 and not again.to_load and not again.missing


def _pg():
    conn = sqlite3.connect(":memory:")
    pg = PgGraph(conn, "col-1", paramstyle="qmark", dialect="sqlite")
    pg.ensure_schema()
    return pg


def _node(pg, label):
    rows = pg._q("SELECT uri, attrs FROM ontology_nodes WHERE collection_id=? AND label=?", ("col-1", label))
    return rows[0] if rows else None


def test_store_writes_rows_under_their_identity_and_replaces_them():
    pg = _pg()
    b, o = _two_tables()
    pg.write(o)
    uri, _attrs = _node(pg, "500000")
    assert uri == IDENTITY_PREFIX + row_identity_key("db:1:COLLATERAL", "coll_id", "500000")[len("dbrow_"):]
    nodes, edges, _chunks = graph_rows(o)
    assert any(e[0] == uri and e[1] == "collateralContractNo" for e in edges)
    assert all(u.startswith(IDENTITY_PREFIX) for u, k, _l, _a in nodes if k == "instance")

    # the row changed (key moved, value NULLed): its traces go, then it is written as it is now
    b.extend_rows(o, "COLLATERAL", ["coll_id", "contract_no", "coll_type"],
                  [{"coll_id": 500000, "contract_no": "L2", "coll_type": None}],
                  source_id="db:1:COLLATERAL", pk_candidates=["coll_id"], label_column="coll_id", fk_relations=_COLL_FK)
    detached = pg.detach_chunks(["db:1:COLLATERAL:500000"])
    assert detached["sources"] == 1 and detached["edges"] == 1
    pg.write(o, replace=False)
    uri2, attrs = _node(pg, "500000")
    assert uri2 == uri and not attrs                                   # same node, old value gone
    objs = {o_ for s, _p, o_ in pg._q("SELECT subject_uri, predicate, object_uri FROM ontology_edges"
                                      " WHERE collection_id=? AND edge_kind='objectProperty' AND subject_uri=?",
                                      ("col-1", uri))}
    assert objs == {_node(pg, "L2")[0]}

    # a loaded graph knows its rows again
    back = pg.load()
    assert {i.name: bool(i.identity) for i in back.instances if i.name in ("500000", "L1")} == {"500000": True, "L1": True}

    # the row left the table: node, link and relation go
    removed = pg.remove_rows(["db:1:COLLATERAL:500000"])
    assert removed["nodes"] >= 1 and _node(pg, "500000") is None
    assert not pg._q("SELECT 1 FROM ontology_edges WHERE collection_id=? AND subject_uri=?", ("col-1", uri))
    assert _node(pg, "L2") is not None
