"""The XGEN graph-table adapter, driven through sqlite (same SQL, ``?`` placeholders), and the progress callback."""
import sqlite3

from xgen_ontology_build import Ontology, OntologyBuilder, PgGraph, graph_rows

_T1 = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국마사회</td><td>말산업연구소</td><td>12억원</td></tr>
<tr><td>농림축산식품부</td><td>축산정책과</td><td>30억원</td></tr>
<tr><td>제주특별자치도</td><td>축산과</td><td>5억원</td></tr>
</table>"""
_T2 = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>렛츠런재단</td><td>사업팀</td><td>3억원</td></tr>
<tr><td>서울경마공원</td><td>경주팀</td><td>9억원</td></tr>
<tr><td>부산경남경마공원</td><td>운영팀</td><td>7억원</td></tr>
</table>"""


def _pg():
    conn = sqlite3.connect(":memory:")
    pg = PgGraph(conn, "col-1", paramstyle="qmark", dialect="sqlite")
    pg.ensure_schema()
    return pg


def test_graph_rows_have_the_store_shape():
    onto = OntologyBuilder(chunk=False).build({"a.md": _T1})
    nodes, edges, chunks = graph_rows(onto)
    kinds = {k for _u, k, _l, _a in nodes}
    assert kinds == {"concept", "property", "instance"}
    by_label = {lb: (u, k, a) for u, k, lb, a in nodes}
    assert by_label["기관명"][0].startswith("https://w3id.org/xgen-domain#")
    assert by_label["한국마사회"][0].startswith("https://w3id.org/xgen-instance#")
    assert by_label["한국마사회"][2] == {"예산": ["12억원"]}          # data values ride on the node
    edge_kinds = {e[4] for e in edges}
    assert edge_kinds == {"instanceOf", "objectProperty"}
    assert all(len(e) == 7 for e in edges) and chunks and all(c[1] == "a.md#0" for c in chunks)


def test_write_load_roundtrip_and_search_protocol():
    pg = _pg()
    onto = OntologyBuilder(chunk=False).build({"a.md": _T1})
    counts = pg.write(onto)
    assert counts["nodes"] == pg.counts()["concept"] + pg.counts()["instance"] + pg.counts()["property"]

    back = pg.load()
    assert {i.name for i in back.instances} == {i.name for i in onto.instances}
    assert {(i.name, i.class_name) for i in back.instances} == {(i.name, i.class_name) for i in onto.instances}
    assert {(r.subject, r.predicate, r.object) for r in back.relations} == {
        (r.subject, r.predicate, r.object) for r in onto.relations}
    assert {(d.entity, d.property, str(d.value)) for d in back.data_values} == {
        (d.entity, d.property, str(d.value)) for d in onto.data_values}
    assert [c.id for c in back.chunks] == ["a.md#0"]

    # GraphStore protocol over the tables
    hits = pg.search_labels("한국마사회 예산")
    assert hits and hits[0][0].label == "한국마사회"
    node = pg.get_node("기관명")
    assert node is not None and node.kind == "class"
    assert {n.label for n in pg.class_instances("기관명")} == {"한국마사회", "농림축산식품부", "제주특별자치도"}
    assert pg.count_class(node.id) == 3
    trip = pg.neighbors("한국마사회")
    assert ("한국마사회", "relatedTo", "말산업연구소") in trip and ("한국마사회", "instanceOf", "기관명") in trip
    labels = pg.entity_labels()
    assert labels[0] == "기관명" and "한국마사회" in labels and "relatedTo" in pg.predicates()   # best-connected first



def test_incremental_build_on_a_stored_graph():
    pg = _pg()
    builder = OntologyBuilder(chunk=False)
    pg.write(builder.build({"a.md": _T1}))

    stored = pg.load()                                   # chunk ids only: enough to know what is new
    builder.extend(stored, {"a.md": _T1, "b.md": _T2})
    assert stored.report.chunks == 2
    pg.write(stored, replace=False)                      # append: existing rows kept, attrs merged
    names = {n.label for n in pg.class_instances("기관명")}
    assert names == {"한국마사회", "농림축산식품부", "제주특별자치도", "렛츠런재단", "서울경마공원", "부산경남경마공원"}
    assert pg.counts()["chunks"] == 2
    again = pg.load()
    assert ("렛츠런재단", "예산", "3억원") in {(d.entity, d.property, str(d.value)) for d in again.data_values}
    assert ("한국마사회", "예산", "12억원") in {(d.entity, d.property, str(d.value)) for d in again.data_values}


def test_enriched_markers_round_trip():
    pg = _pg()
    onto = OntologyBuilder(chunk=False).build({"a.md": _T1})
    onto.enriched_chunks = ["a.md#0"]
    pg.write(onto)
    assert pg.enriched_chunk_ids() == {"a.md#0"} and pg.load().enriched_chunks == ["a.md#0"]
    pg.clear()
    assert pg.counts()["edges"] == 0 and pg.enriched_chunk_ids() == set()


def test_progress_callback_reports_each_stage():
    seen = []
    OntologyBuilder(chunk=False, progress=lambda stage, detail: seen.append((stage, detail))).build({"a.md": _T1})
    stages = [s for s, _ in seen]
    assert stages[0] == "start" and stages[-1] == "done"
    assert {"extract", "dedup", "hierarchy", "finalize"} <= set(stages)
    assert seen[0][1]["mode"] == "basic" and seen[-1][1]["instances"] >= 6


def test_label_weight_and_sources_make_the_round_trip():
    from xgen_ontology_build.models import Instance, Relation

    pg = _pg()
    onto = Ontology(instances=[Instance("한국마사회", source_chunks=["c1", "c2"]),
                               Instance("말산업연구소", source_chunks=["c1"])],
                    relations=[Relation("한국마사회", "relatedTo", "말산업연구소", label="담당부서",
                                        source_chunks=["c1"], weight=2.0)])
    pg.write(onto)
    back = {(r.subject, r.predicate, r.object): r for r in pg.load().relations}
    r = back[("한국마사회", "relatedTo", "말산업연구소")]
    assert r.label == "담당부서" and r.weight == 2.0 and r.source_chunks == ["c1"]


def test_incremental_write_supersedes_and_renames_the_old_neighbour_link():
    from xgen_ontology_build.models import Instance, Relation
    from xgen_ontology_build.store.postgres import DOMAIN_NS, instance_uri

    pg = _pg()
    a, b = instance_uri("A"), instance_uri("B")
    base = Ontology(instances=[Instance("A", source_chunks=["c1"]), Instance("B", source_chunks=["c1"])],
                    relations=[Relation("A", "relatedTo", "B", label="담당", source_chunks=["c1"])])
    pg.write(base)
    pg._x("INSERT INTO ontology_edges(collection_id, subject_uri, predicate, object_uri, predicate_uri, edge_kind,"
          " edge_weight) VALUES (?,?,?,?,?,?,?)", ("col-1", b, "관련", a, DOMAIN_NS + "관련", "objectProperty", 1.0))
    loaded = pg.load()
    assert ("B", "relatedTo", "A") in {(r.subject, r.predicate, r.object) for r in loaded.relations}
    # relation formation names the pair: the unnamed links go, in the table too
    loaded.relations = [Relation("A", "manages", "B", source_chunks=["c1"])]
    pg.write(loaded, replace=False)
    stored = {tuple(r) for r in pg._q("SELECT subject_uri, predicate, object_uri FROM ontology_edges"
                                      " WHERE edge_kind='objectProperty'")}
    assert stored == {(a, "manages", b)}
    assert pg._q("SELECT count(*) FROM ontology_edge_sources WHERE predicate='relatedTo'")[0][0] == 0


def test_the_relation_vocabulary_is_kept_in_the_schema_table():
    from xgen_ontology_build import vocabulary_of
    from xgen_ontology_build.models import Concepts, ObjectProperty

    pg = _pg()
    onto = Ontology(concepts=Concepts(object_properties=[
        ObjectProperty("hasDepartment", description="the organization has the department"),
        ObjectProperty("담당", description="not an identifier: not vocabulary")]))
    pg.write(onto)
    pg.write(onto, replace=False)                                     # written twice, kept once
    rows = pg._q("SELECT element_name, description FROM ontology_schema WHERE element_type='ObjectProperty'")
    # every declared relation name is a schema row (the product keeps them all); the vocabulary
    # a model is shown is the identifiers among them
    assert rows == [("hasDepartment", "the organization has the department"),
                    ("담당", "not an identifier: not vocabulary")]
    back = pg.load().concepts
    assert {op.name: op.description for op in back.object_properties} == {
        "hasDepartment": "the organization has the department", "담당": "not an identifier: not vocabulary"}
    assert vocabulary_of(back) == [{"name": "hasDepartment", "definition": "the organization has the department"}]


# ── 0.14: the store boundary (develop schema rows, URIs kept across load and write) ──


def test_a_stored_node_keeps_its_uri_across_load_extend_and_append():
    from xgen_ontology_build.models import Class, Concepts, Instance

    pg = _pg()
    # a node the product stored under a URI the name would not give now (a promoted head)
    onto = Ontology(concepts=Concepts(classes=[Class(name="기관")]),
                    instances=[Instance(name="한국마사회", class_name="기관", source_chunks=["c1"])],
                    uris={("concept", "기관"): "https://w3id.org/xgen-instance#기관"})
    pg.write(onto)
    back = pg.load()
    assert back.uris[("concept", "기관")] == "https://w3id.org/xgen-instance#기관"
    back.instances.append(Instance(name="렛츠런재단", class_name="기관", source_chunks=["c2"]))
    pg.write(back, replace=False)
    rows = pg._q("SELECT uri FROM ontology_nodes WHERE kind='concept' AND label='기관'")
    assert rows == [("https://w3id.org/xgen-instance#기관",)]               # one row, the stored URI
    assert pg.counts()["concept"] == 1 and {n.label for n in pg.class_instances("기관")} == {"한국마사회", "렛츠런재단"}
    # a fresh build appended to the store finds the rows as they are: no second node for a known name
    fresh = Ontology(concepts=Concepts(classes=[Class(name="기관")]),
                     instances=[Instance(name="서울경마공원", class_name="기관", source_chunks=["c3"])])
    pg.write(fresh, replace=False)
    assert pg.counts()["concept"] == 1 and pg.count_class("기관") == 3


def test_a_head_promoted_to_a_class_keeps_the_instance_uri_and_changes_kind():
    from xgen_ontology_build.models import Class, Concepts, Instance

    pg = _pg()
    pg.write(Ontology(instances=[Instance(name="감사실", source_chunks=["c1"])]))
    (uri,) = pg._q("SELECT uri FROM ontology_nodes WHERE label='감사실'")[0]
    pg.write(Ontology(concepts=Concepts(classes=[Class(name="감사실")]),
                      instances=[Instance(name="상임감사실", class_name="감사실", source_chunks=["c2"])]), replace=False)
    assert pg._q("SELECT uri, kind FROM ontology_nodes WHERE label='감사실'") == [(uri, "concept")]


def test_the_whole_schema_is_kept_in_the_schema_table():
    from xgen_ontology_build import build_from_csv

    pg = _pg()
    onto = build_from_csv({"colors": "color_id,name,price\n10,Red,1.5\n20,Blue,2.0"})
    pg.write(onto)
    rows = {(t, n): (src, rng) for t, n, src, rng in pg._q(
        "SELECT element_type, element_name, source, range_value FROM ontology_schema ORDER BY id")}
    assert rows[("Class", "Colors")] == ("csv", None)                       # a table's class, marked as the product marks it
    assert rows[("DatatypeProperty", "price")] == (None, "xsd:decimal")
    back = pg.load()
    assert next(c for c in back.concepts.classes if c.name == "Colors").source == "table"
    assert next(d for d in back.concepts.datatype_properties if d.name == "price").range == "xsd:decimal"
    assert pg._q("SELECT is_auto_generated, is_confirmed FROM ontology_schema LIMIT 1") == [(1, 0)]
