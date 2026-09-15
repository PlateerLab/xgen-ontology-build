"""Graph normalization, incremental builds, IRI translation, the term dictionary and communities."""
import json

from xgen_ontology import (
    CallableLLM,
    OntologyBuilder,
    TermDictionary,
    build_from_documents,
    normalize_graph,
    translate_names,
    unbuilt_chunks,
)
from xgen_ontology.build.translate import english_local_name
from xgen_ontology.models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation

# ───────────────────────── normalize_graph ─────────────────────────


def test_normalize_graph_applies_the_store_loading_rules():
    concepts = Concepts(
        classes=[Class(name="기관"), Class(name="'18년"), Class(name="S"), Class(name="부서 "), Class(name="기관")],
        object_properties=[ObjectProperty(name="소속", domain="직원", range="부서"),
                           ObjectProperty(name="이름", domain="직원", range="xsd:string")],
        datatype_properties=[DataProperty(name="예산", domain="기관"), DataProperty(name="면적", domain="'18년")],
        class_hierarchy=[("기관", "29"), ("기관", "공공기관"), ("기관", "기관")])
    instances = [Instance(name="한국마사회", class_name="기관", source_chunks=["c1"]),
                 Instance(name="한국마사회", source_chunks=["c2"]), Instance(name="연혁", class_name="'18년")]
    relations = [Relation("김철수", "소속", "말산업연구소"), Relation("김철수", "type", "직원"),
                 Relation("말산업연구소", "instanceOf", "부서"), Relation("공기업", "subClassOf", "기관"),
                 Relation("한국마사회", "예산", "12억원"), Relation("한국마사회", "설립", "1949"),
                 Relation("2020", "연도", "2020"), Relation("마사회", "sameAs", "한국마사회"),
                 Relation("한국마사회", "소속", "농림축산식품부"), Relation("한국마사회", "소속", "농림축산식품부")]
    data_values = [DataValue("마사회", "자본금", "500억원"), DataValue("2020", "연도", "2020")]
    stats = normalize_graph(concepts, instances, relations, data_values)

    names = {c.name for c in concepts.classes}
    assert names == {"기관", "부서", "직원", "공공기관", "공기업"}          # junk dropped, referenced ones declared
    assert ("기관", "공공기관") in concepts.class_hierarchy and ("기관", "공기업") in concepts.class_hierarchy
    assert [op.name for op in concepts.object_properties] == ["소속"]      # a datatype range is not an object property
    assert [dp.name for dp in concepts.datatype_properties] == ["예산"]
    typed = {(i.name, i.class_name) for i in instances}
    assert ("김철수", "직원") in typed and ("말산업연구소", "부서") in typed   # typing statements became types
    assert ("한국마사회", "기관") in typed and ("한국마사회", "") not in typed  # merged, untyped record redundant
    assert ("연혁", "") in typed                                          # its class was junk: untyped, kept
    rel = {(r.subject, r.predicate, r.object) for r in relations}
    assert rel == {("김철수", "소속", "말산업연구소"), ("한국마사회", "소속", "농림축산식품부")}
    dv = {(d.entity, d.property, str(d.value)) for d in data_values}
    assert ("한국마사회", "예산", "12억원") in dv and ("한국마사회", "설립", "1949") in dv
    assert ("한국마사회", "자본금", "500억원") in dv                       # sameAs folded 마사회 into 한국마사회
    assert not any(d.entity == "2020" for d in data_values)             # a value is not a subject
    assert "농림축산식품부" in {i.name for i in instances}                 # no dangling endpoints
    assert stats["typed"] == 2 and stats["subclassed"] == 1 and stats["folded"] == 1


# ───────────────────────── incremental build ─────────────────────────

_T1 = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국마사회</td><td>말산업연구소</td><td>12억원</td></tr>
<tr><td>농림축산식품부</td><td>축산정책과</td><td>30억원</td></tr>
<tr><td>제주특별자치도</td><td>축산과</td><td>5억원</td></tr>
</table>"""
_T2 = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>렛츠런재단</td><td>사업팀</td><td>3억원</td></tr>
<tr><td>한국마사회 제주본부</td><td>말산업팀</td><td>2억원</td></tr>
<tr><td>서울경마공원</td><td>경주팀</td><td>9억원</td></tr>
</table>"""


def test_extend_builds_only_the_new_chunks_and_matches_a_full_build():
    builder = OntologyBuilder(chunk=False)
    onto = builder.build({"a.md": _T1})
    assert onto.report.chunks == 1
    assert unbuilt_chunks(onto, {"a.md": _T1, "b.md": _T2}) == {
        "b.md": [{"chunk_id": "b.md#0", "chunk_text": _T2, "chunk_index": 0}]}
    builder.extend(onto, {"a.md": _T1, "b.md": _T2})
    assert onto.report.chunks == 2
    names = {i.name for i in onto.instances}
    assert {"한국마사회", "렛츠런재단", "서울경마공원"} <= names
    assert ("렛츠런재단", "예산", "3억원") in {(d.entity, d.property, d.value) for d in onto.data_values}
    # an incremental build lands where a full build would
    full = OntologyBuilder(chunk=False).build({"a.md": _T1, "b.md": _T2})
    assert {i.name for i in full.instances} == names
    assert {(r.subject, r.predicate, r.object) for r in full.relations} == {
        (r.subject, r.predicate, r.object) for r in onto.relations}
    # nothing new: nothing extracted, state unchanged
    before = len(onto.instances)
    builder.extend(onto, {"a.md": _T1, "b.md": _T2})
    assert len(onto.instances) == before and any("no new chunks" in n for n in onto.report.notes)


def test_extend_in_enrich_mode_only_asks_for_unenriched_chunks():
    asked = []

    def llm(prompt, system=""):
        if "Known entities" in prompt:
            asked.append(prompt)
            return json.dumps({"relations": []})
        return json.dumps({"merge_groups": []})

    builder = OntologyBuilder(CallableLLM(llm), mode="enrich", chunk=False)
    onto = builder.build({"a.md": _T1})
    assert onto.enriched_chunks == ["a.md#0"] and len(asked) == 1
    builder.extend(onto, {"a.md": _T1, "b.md": _T2})
    assert onto.enriched_chunks == ["a.md#0", "b.md#0"] and len(asked) == 2
    assert "[CHUNK:c0]" in asked[1] and "렛츠런재단" in asked[1] and "한국마사회 제주본부" not in asked[0]


# ───────────────────────── IRI translation ─────────────────────────


def test_translate_names_batches_caches_and_shapes_identifiers():
    calls = []

    def llm(prompt, system=""):
        calls.append(prompt)
        terms = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
        return json.dumps({"translations": {t: {"신용등급": "credit rating", "대출한도": "Loan-Limit"}.get(t, "") for t in terms}})

    got = translate_names(["신용등급", "대출한도", "신용등급"], CallableLLM(llm), cache={"기관": "Organization"})
    assert got == {"기관": "Organization", "신용등급": "credit rating", "대출한도": "Loan-Limit"} and len(calls) == 1
    translate_names(["신용등급"], CallableLLM(llm), cache=got)
    assert len(calls) == 1                                               # cached: not sent again
    assert english_local_name("신용등급", got, is_class=True) == "Creditrating"
    assert english_local_name("대출한도", got, is_class=False) == "loanLimit"
    assert english_local_name("미번역", got) == "미번역"


def test_ontology_translate_feeds_the_emitter():
    onto = build_from_documents({"a.md": _T1}, chunk=False)
    onto.translate(CallableLLM(lambda p, system="": json.dumps(
        {"translations": {"기관명": "OrganizationName", "예산": "Budget", "담당부서": "Department"}})))
    ttl = onto.to_turtle()
    assert ":OrganizationName a owl:Class" in ttl and ":budget a owl:DatatypeProperty" in ttl


# ───────────────────────── term dictionary ─────────────────────────


def test_term_dictionary_query_expansion_indexing_and_build_canonicalization():
    d = TermDictionary("finance")
    d.bulk_import([{"alias": "DSR", "canonical": "총부채원리금상환비율", "type": "acronym"},
                   {"alias": "LTV", "canonical": "담보인정비율"}, {"alias": "", "canonical": "x"}])
    assert d.lookup("dsr").canonical == "총부채원리금상환비율" and d.lookup("nope") is None
    q, applied = d.normalize_query("DSR 60% 고객의 DSRA 확인")
    assert q == "DSR 60% 고객의 DSRA 확인 총부채원리금상환비율" and len(applied) == 1   # not inside DSRA
    q, _ = d.normalize_query("DSR 60%", mode="replace")
    assert q == "총부채원리금상환비율 60%"
    text = d.normalize_text("고객의 LTV 60%, DSR 50%")
    assert text == "고객의 LTV(담보인정비율) 60%, DSR(총부채원리금상환비율) 50%"
    assert d.normalize_text(text) == text                              # idempotent
    d.set_active("LTV", False)
    assert d.normalize_text("LTV 60%") == "LTV 60%"

    concepts = Concepts(classes=[Class(name="DSR")])
    instances = [Instance(name="dsr", class_name="지표")]
    relations = [Relation("고객A", "적용", "DSR")]
    n = d.apply_to_build(concepts, instances, relations, [])
    assert n == 3 and concepts.classes[0].name == "총부채원리금상환비율" and relations[0].object == "총부채원리금상환비율"


def test_builder_applies_a_dictionary_to_the_build():
    d = TermDictionary()
    d.add("한국마사회", "한국마사회(KRA)")
    onto = OntologyBuilder(chunk=False, dictionary=d).build({"a.md": _T1})
    assert "한국마사회(KRA)" in {i.name for i in onto.instances}
    assert "한국마사회" not in {i.name for i in onto.instances}


# ───────────────────────── communities ─────────────────────────


def test_community_of_partitions_the_instance_graph():
    onto = build_from_documents({"a.md": _T1, "b.md": _T2}, chunk=False)
    comm = onto.community_of()
    assert comm and set(comm) <= {i.name for i in onto.instances}
    assert comm["한국마사회"] == comm["말산업연구소"]                      # linked by a row relation
    assert onto.communities()[0]["size"] >= 2
