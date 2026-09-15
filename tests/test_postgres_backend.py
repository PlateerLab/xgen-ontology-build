"""The XGEN graph-table adapter, driven through sqlite (same SQL, ``?`` placeholders), and the progress callback."""
import sqlite3

from xgen_ontology import GraphRAG, InMemoryVector, OntologyBuilder, PgGraph, graph_rows

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
    assert all(len(e) == 6 for e in edges) and chunks and all(c[1] == "a.md#0" for c in chunks)


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
    assert ("한국마사회", "담당부서", "말산업연구소") in trip and ("한국마사회", "instanceOf", "기관명") in trip
    labels = pg.entity_labels()
    assert labels[0] == "기관명" and "한국마사회" in labels and "담당부서" in pg.predicates()   # best-connected first

    # and the one-shot search runs on it end to end (EchoLLM: the fused evidence comes back)
    res = GraphRAG(pg, InMemoryVector(onto.chunks), None).search("한국마사회 담당부서")
    assert any("말산업연구소" in r for r in res.relations)


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
