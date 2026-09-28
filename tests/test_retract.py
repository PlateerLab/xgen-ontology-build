"""Retraction: deleted chunks leave the graph as a delta, in memory and on the graph tables."""
import sqlite3

from xgen_ontology import Ontology, OntologyBuilder, PgGraph, removed_chunks, retract_chunks
from xgen_ontology.models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation

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
_T3 = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국농어촌공사</td><td>농지관리처</td><td>20억원</td></tr>
<tr><td>축산물품질평가원</td><td>등급판정과</td><td>4억원</td></tr>
</table>"""


def _shape(onto):
    return ({i.name for i in onto.instances},
            {(r.subject, r.predicate, r.object) for r in onto.relations},
            {(d.entity, d.property, str(d.value)) for d in onto.data_values})


def _hand_made():
    """K{X, Y, Z} with X-p-Y stated in c1 but X and Y also sharing c2; Z lives in c1 only.

    K2{W} is declared with its own property and a property ranging over K; P is the parent
    of K with no chunk of its own."""
    concepts = Concepts(
        classes=[Class(name="K", source_chunks=["c1", "c2"]), Class(name="K2", source_chunks=["c9"]),
                 Class(name="P", source_chunks=["c9"])],
        object_properties=[ObjectProperty(name="p", domain="K", range="K"),
                           ObjectProperty(name="wp", domain="K2", range="K")],
        datatype_properties=[DataProperty(name="attr", domain="K"), DataProperty(name="w_attr", domain="K2")],
        class_hierarchy=[("P", "K")])
    instances = [Instance(name="X", class_name="K", source_chunks=["c1", "c2"]),
                 Instance(name="Y", class_name="K", source_chunks=["c1", "c2"]),
                 Instance(name="Z", class_name="K", source_chunks=["c1"]),
                 Instance(name="W", class_name="K2", source_chunks=["c9"])]
    relations = [Relation(subject="X", predicate="p", object="Y", source_chunks=["c1"]),
                 Relation(subject="X", predicate="q", object="Z", source_chunks=["c1"]),
                 Relation(subject="X", predicate="관련", object="Y"),
                 Relation(subject="W", predicate="wp", object="X", source_chunks=["c9"])]
    data_values = [DataValue(entity="X", property="attr", value="1", source_chunks=["c1"]),
                   DataValue(entity="Z", property="attr", value="2", source_chunks=["c1"]),
                   DataValue(entity="W", property="w_attr", value="3", source_chunks=["c9"])]
    return concepts, instances, relations, data_values


# ───────────────────────── in memory ─────────────────────────


def test_retract_leaves_what_a_build_of_the_surviving_corpus_would():
    stages = []
    builder = OntologyBuilder(chunk=False, progress=lambda s, d: stages.append(s))
    onto = builder.build({"a.md": _T1, "b.md": _T2})
    builder.retract(onto, ["b.md#0"])
    assert "retract" in stages
    assert _shape(onto) == _shape(OntologyBuilder(chunk=False).build({"a.md": _T1}))
    assert [c.id for c in onto.chunks] == ["a.md#0"] and onto.report.chunks == 1
    assert onto.report.retracted["chunks"] == 1 and onto.report.retracted["instances"] >= 3
    # nothing left to take out: a second call is a no-op
    builder.retract(onto, ["b.md#0"])
    assert onto.report.retracted["links"] == 0 and _shape(onto) == _shape(OntologyBuilder(chunk=False).build({"a.md": _T1}))


def test_extend_with_retract_missing_applies_the_corpus_delta():
    builder = OntologyBuilder(chunk=False)
    onto = builder.build({"a.md": _T1, "b.md": _T2})
    now = {"a.md": _T1, "c.md": _T3}                     # b deleted, c added
    assert removed_chunks(onto, now) == ["b.md#0"]
    builder.extend(onto, now, retract_missing=True)
    assert _shape(onto) == _shape(OntologyBuilder(chunk=False).build(now))
    assert sorted(c.id for c in onto.chunks) == ["a.md#0", "c.md#0"]
    # without the flag, extend stays additive (documents may be only the new ones)
    additive = OntologyBuilder(chunk=False).build({"a.md": _T1, "b.md": _T2})
    OntologyBuilder(chunk=False).extend(additive, {"c.md": _T3})
    assert "렛츠런재단" in {i.name for i in additive.instances}


def test_retract_chunks_rules_evidence_orphans_and_structural_links():
    concepts, instances, relations, data_values = _hand_made()
    stats = retract_chunks(concepts, instances, relations, data_values, ["c1"], structural_predicates=("관련",))
    names = {i.name for i in instances}
    assert names == {"X", "Y", "W"}                                   # Z had no chunk left
    rels = {(r.subject, r.predicate, r.object): r.source_chunks for r in relations}
    assert ("X", "p", "Y") in rels and rels[("X", "p", "Y")] == ["c2"]   # X and Y still share c2
    assert ("X", "q", "Z") not in rels                                   # went with Z
    assert ("X", "관련", "Y") in rels                                     # name structure is not chunk evidence
    assert {(d.entity, d.property) for d in data_values} == {("X", "attr"), ("W", "w_attr")}
    assert {c.name for c in concepts.classes} == {"K", "K2", "P"}     # K is still referenced
    assert stats["instances"] == 1 and stats["relations"] == 1 and stats["data_values"] == 1
    assert stats["links"] == 4                                        # X, Y, Z and K each lost c1


def test_retract_chunks_drops_an_unreferenced_class_with_its_properties_but_keeps_a_parent():
    concepts, instances, relations, data_values = _hand_made()
    stats = retract_chunks(concepts, instances, relations, data_values, ["c9"])
    assert {i.name for i in instances} == {"X", "Y", "Z"}
    assert {c.name for c in concepts.classes} == {"K", "P"}           # K2 lost W and nothing refers to it; P is K's parent
    assert concepts.class_hierarchy == [("P", "K")]
    assert {op.name for op in concepts.object_properties} == {"p"}     # wp was declared by K2 only
    assert {dp.name for dp in concepts.datatype_properties} == {"attr"}
    assert ("W", "wp", "X") not in {(r.subject, r.predicate, r.object) for r in relations}
    assert stats["classes"] == 1 and stats["properties"] == 2 and stats["instances"] == 1
    # everything gone: an empty graph, as when a collection's last document is deleted
    concepts, instances, relations, data_values = _hand_made()
    retract_chunks(concepts, instances, relations, data_values, ["c1", "c2", "c9"], structural_predicates=("관련",))
    assert not instances and not relations and not data_values and not concepts.classes


# ───────────────────────── graph tables ─────────────────────────


def _pg(cid="col-1"):
    conn = sqlite3.connect(":memory:")
    pg = PgGraph(conn, cid, paramstyle="qmark", dialect="sqlite")
    pg.ensure_schema()
    return pg


def test_pg_prune_chunks_leaves_the_rows_of_the_surviving_corpus():
    pg = _pg()
    onto = OntologyBuilder(chunk=False).build({"a.md": _T1, "b.md": _T2})
    onto.enriched_chunks = ["a.md#0", "b.md#0"]
    pg.write(onto)
    stats = pg.prune_chunks(["b.md#0"])
    assert stats["chunks"] == 1 and stats["links"] > 0 and stats["nodes"] >= 6 and stats["enriched"] == 1
    assert _shape(pg.load()) == _shape(OntologyBuilder(chunk=False).build({"a.md": _T1}))
    ref = _pg("ref")
    ref.write(OntologyBuilder(chunk=False).build({"a.md": _T1}))
    assert pg.counts() == ref.counts() and pg.enriched_chunk_ids() == {"a.md#0"}
    # a retry finds nothing left to do
    again = pg.prune_chunks(["b.md#0"])
    assert again["links"] == 0 and again["nodes"] == 0 and pg.counts() == ref.counts()


def test_pg_prune_chunks_evidence_orphans_and_structural_edges_on_the_tables():
    pg = _pg()
    concepts, instances, relations, data_values = _hand_made()
    pg.write(Ontology(concepts=concepts, instances=instances, relations=relations, data_values=data_values))
    pg.prune_chunks(["c1"])
    back = pg.load()
    assert {i.name for i in back.instances} == {"X", "Y", "W"}
    rels = {(r.subject, r.predicate, r.object) for r in back.relations}
    assert ("X", "p", "Y") in rels and ("X", "관련", "Y") in rels and ("X", "q", "Z") not in rels
    assert {c.name for c in back.concepts.classes} == {"K", "K2", "P"}
    pg.prune_chunks(["c9"])
    back = pg.load()
    assert {c.name for c in back.concepts.classes} == {"K", "P"} and {i.name for i in back.instances} == {"X", "Y"}
    assert pg.counts()["property"] == 1                                # K2's own property node went with it
    pg.prune_chunks(["c2"])
    assert pg.counts() == {"concept": 0, "instance": 0, "property": 0, "edges": 0, "chunks": 0}
