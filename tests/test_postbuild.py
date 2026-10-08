"""Rule post-build (name hygiene, hierarchy from names, predicate governance, typing repairs) and the pipeline modes."""
import json
import re

import pytest

from xgen_ontology import (
    CallableLLM,
    Deduplicator,
    build_from_csv,
    build_from_documents,
    fix_self_typed_instances,
    fold_name_fragments,
    govern_predicates,
    induce_hierarchy,
    materialize_property_inheritance,
    merge_predicates,
    prune_common_words,
    strip_argument_noun,
    vote_relation_direction,
)
from xgen_ontology.korean import get_kiwi
from xgen_ontology.models import Class, Concepts, DataProperty, Instance, ObjectProperty, Relation

kiwi_required = pytest.mark.skipif(get_kiwi() is None, reason="kiwipiepy not installed")


# ───────────────────────── name hygiene ─────────────────────────


def test_fold_name_fragments_folds_a_cut_piece_into_its_source_name():
    chunks = {"신용보증기금": {"c1", "c2"}, "증기금": {"c1"}, "기술보증기금": {"c3"}}
    assert fold_name_fragments(list(chunks), chunks, linked=set()) == {"증기금": "신용보증기금"}


def test_fold_name_fragments_keeps_whole_words_linked_names_and_unsupported_pieces():
    chunks = {"보증 기금": {"c1"}, "기금": {"c1"}, "증기금": {"c9"}, "신용보증기금": {"c2"}}
    # "기금" stands as a whole word inside "보증 기금" -> a place to attach to, not a fragment
    # "증기금" appears in a chunk none of the longer names covers -> stays
    assert fold_name_fragments(list(chunks), chunks, linked=set()) == {}
    chunks = {"신용보증기금": {"c1"}, "증기금": {"c1"}}
    assert fold_name_fragments(list(chunks), chunks, linked={"증기금"}) == {}


def test_fold_name_fragments_follows_chains():
    chunks = {"신용보증기금": {"c1"}, "보증기금": {"c1"}, "증기금": {"c1"}}
    fold = fold_name_fragments(list(chunks), chunks, linked=set())
    assert fold["증기금"] == "신용보증기금" and fold["보증기금"] == "신용보증기금"


@kiwi_required
def test_prune_common_words_drops_unlinked_everyday_words_only():
    assert prune_common_words(["결과", "환경", "마사회", "포토레지스트"], linked=set()) == {"결과", "환경"}
    assert prune_common_words(["결과"], linked={"결과"}) == set()


def test_dedup_key_ignores_leading_quotes_and_prefers_the_shortest_spelling():
    d = Deduplicator()
    insts = [Instance(name="'노동생산성"), Instance(name="노동생산성"), Instance(name="노동 생산성")]
    assert d._normalize_instances(insts) == {"'노동생산성": "노동생산성", "노동 생산성": "노동생산성"}


# ───────────────────────── hierarchy from names ─────────────────────────


@kiwi_required
def test_induce_hierarchy_types_instances_by_their_head_and_promotes_heads():
    concepts = Concepts(classes=[Class(name="감사실")])
    instances = [Instance(name="상임감사실", source_chunks=["c1"]),
                 Instance(name="준법감사실", class_name="부서", source_chunks=["c2"]),
                 Instance(name="전사경마사업", source_chunks=["c3"]), Instance(name="사업", source_chunks=["c4"])]
    relations: list[Relation] = []
    counts = induce_hierarchy(concepts, None, instances, relations)
    typed = {(i.name, i.class_name) for i in instances}
    assert ("상임감사실", "감사실") in typed                       # untyped -> typed by its head
    assert ("준법감사실", "부서") in typed and ("준법감사실", "감사실") in typed   # keeps its type, gains the head
    assert "사업" in {c.name for c in concepts.classes}           # an instance used as a head is a class
    assert "사업" not in {i.name for i in instances} and ("전사경마사업", "사업") in typed   # converted, not duplicated
    assert counts["promoted"] >= 1 and counts["typed"] >= 3


def test_induce_hierarchy_folds_code_prefixed_spellings_and_links_leading_words():
    concepts = Concepts(classes=[Class(name="제주목장사업"), Class(name="한국마사회")])
    instances = [Instance(name="NA162000.23 제주목장사업", source_chunks=["c1"]),
                 Instance(name="한국마사회 제주본부", source_chunks=["c2"])]
    relations: list[Relation] = []
    counts = induce_hierarchy(concepts, None, instances, relations, related_predicate="관련")
    assert "NA162000.23 제주목장사업" not in {i.name for i in instances}   # sameAs -> one spelling
    assert counts["renamed"] == 1
    assert ("한국마사회 제주본부", "관련", "한국마사회") in {(r.subject, r.predicate, r.object) for r in relations}
    assert "부서명 회장실" not in {c.name for c in concepts.classes}


def test_induce_hierarchy_related_can_be_switched_off():
    concepts = Concepts(classes=[Class(name="한국마사회")])
    instances = [Instance(name="한국마사회 제주본부")]
    relations: list[Relation] = []
    induce_hierarchy(concepts, None, instances, relations, related_predicate=None)
    assert relations == []


# ───────────────────────── predicate governance ─────────────────────────


def test_strip_argument_noun_removes_a_glued_object_but_keeps_short_remainders():
    assert strip_argument_noun("시스템가동감독", "전산운영실", "시스템가동") == "감독"
    assert strip_argument_noun("소속", "직원", "부서") == "소속"
    assert strip_argument_noun("부서장", "김철수", "부서") == "부서장"   # what remains is too short to mean anything


def test_govern_predicates_anchors_to_seeds_and_reports_a_canonical_map():
    rels = [Relation("A", "소속되어있다", "B"), Relation("C", "소속", "D"), Relation("E", "시스템가동감독", "시스템가동")]
    stats = govern_predicates(rels, [ObjectProperty(name="소속됨")], seed_predicates=["감독"])
    assert {r.predicate for r in rels} == {"소속됨", "감독"}
    assert stats["argument_noun_stripped"] == 1 and stats["anchored_to_schema"] == 3
    assert stats["canonical_map"]["소속"] == "소속됨"


def test_vote_relation_direction_flips_the_minority_by_type_pair_then_by_role():
    rels = [Relation(f"부서{i}", "담당", f"사람{i}") for i in range(4)] + [Relation("사람9", "담당", "부서9")]
    types = {**{f"부서{i}": "부서" for i in range(10)}, **{f"사람{i}": "사람" for i in range(10)}}
    stats = vote_relation_direction(rels, types)
    assert stats["flipped"] == 1 and (rels[-1].subject, rels[-1].object) == ("부서9", "사람9")
    # no types at all: an entity that is always the object sitting in subject position is flipped
    rels = [Relation(f"s{i}", "감독", "본부") for i in range(4)] + [Relation("본부", "감독", "s0")]
    rels[-1] = Relation("본부", "감독", "s5")
    stats = vote_relation_direction(rels, {})
    assert stats["flipped"] == 0            # s5 has no history of its own: not enough evidence


def test_merge_predicates_by_stem_and_by_contained_extension():
    rels = ([Relation("a", "소속", "x"), Relation("b", "소속", "y"), Relation("c", "소속", "z")]
            + [Relation("a", "소속됨", "x")]
            + [Relation("p", "담당", "q"), Relation("r", "담당", "s"), Relation("p", "담당함", "q")])
    stats = merge_predicates(rels, lambda p: p.replace("됨", "").replace("함", ""))
    assert stats["merged_predicates"] == 2
    assert {r.predicate for r in rels} == {"소속", "담당"}
    assert len(rels) == 5                    # duplicate triples produced by the merge are removed


# ───────────────────────── typing repairs ─────────────────────────


def test_fix_self_typed_instances_moves_to_the_parent_or_drops_the_typing():
    concepts = Concepts(classes=[Class(name="기관"), Class(name="마사회", parent="기관"), Class(name="농협")],
                        class_hierarchy=[("기관", "마사회")])
    instances = [Instance(name="마사회", class_name="마사회"), Instance(name="농협", class_name="농협"),
                 Instance(name="제주은행", class_name="기관")]
    stats = fix_self_typed_instances(instances, concepts)
    typed = {i.name: i.class_name for i in instances}
    assert typed["마사회"] == "기관" and typed["농협"] == "" and typed["제주은행"] == "기관"
    assert stats == {"moved": 1, "dropped": 1}


def test_materialize_property_inheritance_copies_declarations_down_the_chain():
    concepts = Concepts(
        classes=[Class(name="기관"), Class(name="공공기관"), Class(name="지방공기업")],
        class_hierarchy=[("기관", "공공기관"), ("공공기관", "지방공기업")],
        object_properties=[ObjectProperty(name="소재지", domain="기관", range="지역")],
        datatype_properties=[DataProperty(name="예산", domain="기관"), DataProperty(name="예산", domain="지방공기업")])
    added = materialize_property_inheritance(concepts)
    assert added == 3
    assert {(dp.name, dp.domain) for dp in concepts.datatype_properties} == {("예산", "기관"), ("예산", "공공기관"), ("예산", "지방공기업")}
    assert {(op.name, op.domain) for op in concepts.object_properties} >= {("소재지", "공공기관"), ("소재지", "지방공기업")}


# ───────────────────────── pipeline modes ─────────────────────────

_TABLE = """예산 배정표
<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국마사회</td><td>말산업연구소</td><td>12억원</td></tr>
<tr><td>농림축산식품부</td><td>축산정책과</td><td>30억원</td></tr>
<tr><td>제주특별자치도</td><td>축산과</td><td>5억원</td></tr>
</table>"""


_UNIT_ENTITY = re.compile(r"(\d+)=(.+?)(?= \d+=|$)")


def relation_model(pairs, *, definitions=None, log=None):
    """A fake model for relation formation, shaped by the real prompts: a vocabulary request gets
    ``definitions``; a classification request names ``pairs`` ({(subject, object): predicate})
    by unit and mention numbers. ``log`` collects (kind, prompt)."""
    def llm(prompt, system=""):
        if '"predicates"' in system:      # defining a vocabulary, or adding to it
            if log is not None:
                log.append(("vocabulary" if "Make the list" in system else "extend", prompt))
            return json.dumps({"predicates": [{"name": n, "definition": d}
                                              for n, d in (definitions or {}).items()]})
        if '"units_with_relations"' in system:
            if log is not None:
                log.append(("classify", prompt))
            rows, unit = [], 0
            for line in prompt.splitlines():
                if line.startswith("["):
                    unit += 1
                elif line.strip().startswith("entities:") and unit:
                    num = {name: int(n) for n, name in _UNIT_ENTITY.findall(line.split(":", 1)[1].strip())}
                    for (s, o), p in pairs.items():
                        if s in num and o in num:
                            rows.append({"u": unit, "s": num[s], "p": p, "o": num[o]})
            return json.dumps({"units_with_relations": sorted({r["u"] for r in rows}), "relations": rows})
        if '"mapping"' in system:
            return json.dumps({"mapping": []})
        return json.dumps({"merge_groups": []})
    return llm


def test_basic_mode_builds_from_documents_with_zero_llm_calls():
    onto = build_from_documents({"budget.md": _TABLE}, chunk=False)
    assert onto.report.mode == "basic" and onto.report.llm_calls == 0
    names = {i.name: i.class_name for i in onto.instances}
    assert names["한국마사회"] == "기관명"
    assert ("한국마사회", "예산", "12억원") in {(d.entity, d.property, d.value) for d in onto.data_values}
    assert ("한국마사회", "relatedTo", "말산업연구소", "담당부서") in {
        (r.subject, r.predicate, r.object, r.label) for r in onto.relations}
    assert onto.report.quality["score"] > 0 and "class_count" in onto.report.quality
    assert onto.search("한국마사회 예산").chunks       # searchable end to end, still no LLM


def test_basic_mode_ignores_a_provided_llm():
    calls = []

    def spy(prompt, system=""):
        calls.append(prompt)
        return "{}"

    onto = build_from_documents({"budget.md": _TABLE}, llm=CallableLLM(spy), chunk=False)
    assert calls == [] and onto.report.llm_calls == 0


def test_enrich_mode_names_relations_unit_by_unit():
    # no vocabulary yet: one is defined from the units first, then each unit's pairs are named
    log = []
    llm = relation_model({("한국마사회", "말산업연구소"): "hasDepartment", ("농림축산식품부", "축산정책과"): "hasDepartment"},
                         definitions={"hasDepartment": "the organization has the department"}, log=log)
    onto = build_from_documents({"budget.md": _TABLE}, llm=CallableLLM(llm), mode="enrich", chunk=False)
    assert onto.report.mode == "enrich" and [k for k, _p in log] == ["vocabulary", "classify"]
    rel = {(r.subject, r.predicate, r.object): r for r in onto.relations}
    r = rel[("한국마사회", "hasDepartment", "말산업연구소")]
    assert r.label == "담당부서" and r.source_chunks == ["budget.md#0"]     # the column name, the stating chunk
    assert ("한국마사회", "relatedTo", "말산업연구소") not in rel             # the named relation supersedes it
    assert ("제주특별자치도", "relatedTo", "축산과") in rel                   # an unnamed pair stays a neighbour
    ops = {op.name: op.description for op in onto.concepts.object_properties}
    assert ops["hasDepartment"] == "the organization has the department"  # the vocabulary is declared
    assert onto.enriched_chunks == ["budget.md#0"] and onto.report.relation_stats["units"] == 3


def test_enrich_mode_drops_rows_that_do_not_hold_together():
    # the graph's own vocabulary, a subject that is its object, a number out of range
    def llm(prompt, system=""):
        if '"units_with_relations"' in system:
            return json.dumps({"units_with_relations": [1], "relations": [
                {"u": 1, "s": 1, "p": "instanceOf", "o": 2}, {"u": 1, "s": 1, "p": "manages", "o": 1},
                {"u": 9, "s": 1, "p": "manages", "o": 2}, {"u": 1, "s": 1, "p": "has department", "o": 2}]})
        if '"predicates"' in system:
            return json.dumps({"predicates": [{"name": "manages", "definition": "x"}]})
        return "{}"

    onto = build_from_documents({"budget.md": _TABLE}, llm=CallableLLM(llm), mode="enrich", chunk=False)
    assert not any(r.predicate in ("manages", "instanceOf", "hasDepartment") for r in onto.relations)
    assert onto.enriched_chunks == ["budget.md#0"]        # answered, even if with nothing usable


def test_enrich_and_llm_modes_fall_back_to_basic_without_an_llm():
    onto = build_from_documents({"budget.md": _TABLE}, mode="enrich", chunk=False)
    assert onto.report.mode == "basic" and any("no LLM" in n for n in onto.report.notes)


def test_builder_rejects_unknown_modes():
    with pytest.raises(ValueError):
        build_from_documents({"a.md": "x"}, mode="magic")


# ───────────────────────── tables ─────────────────────────


def test_table_label_column_is_judged_from_the_data():
    onto = build_from_csv({"staff": "id,name,region\n1,김철수,제주\n2,이영희,제주\n3,박민수,서울"})
    assert {i.name for i in onto.instances} == {"김철수", "이영희", "박민수"}
    onto = build_from_csv({"codes": "id,code,amount\n1,A-10,100\n2,A-11,200\n3,A-12,300"})
    assert {i.name for i in onto.instances} == {"A-10", "A-11", "A-12"}   # a code column is the fallback name
    onto = build_from_csv({"figures": "id,amount,qty\n1,100,5\n2,200,6\n3,300,7"})
    assert onto.instances == [] and {c.name for c in onto.concepts.classes} == {"Figures"}   # no name column: schema only


def test_table_rows_with_the_same_label_are_told_apart_by_their_key():
    onto = build_from_csv({"staff": "id,name,dept\n1,홍길동,영업\n2,홍길동,기획\n3,성춘향,감사"})
    names = sorted(i.name for i in onto.instances)
    assert names == ["성춘향", "홍길동#1", "홍길동#2"]
