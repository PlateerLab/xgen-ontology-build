"""The ontology-learning layers as gates: terms (row dumps are not prose), synonyms (aliases from evidence),
concepts (a generic head is not a type), taxonomy (Hearst reads the sentence, not a quoted title)."""
import pytest

from xgen_ontology.build.deterministic import parse_row_dump
from xgen_ontology.build.taxonomy import (
    _MORPH_CACHE,
    DEFAULT_RELATED_PREDICATE,
    _morph_starts,
    _rotation_key,
    extract_hearst_pairs,
    induce_aliases,
    induce_head_noun_hierarchy,
)
from xgen_ontology.korean import get_kiwi

kiwi_required = pytest.mark.skipif(get_kiwi() is None, reason="needs kiwipiepy")

_REGULATION = """제1조 (목적) 이 지침은 당사를 위한 각종 대출 모집 및 중개 등의 위탁업무를 수행하고 있는 Agency의 운용에 관한
제2조 (적용범위) Agency의 등록, 계약, 관리 및 해지 등 운용에 관한 사항은 이 지침에 따른다
제3조 (정의) 이 지침에서 사용하는 용어의 정의는 다음과 같다
제4조 (등록) Agency 로 등록하려는 자는 다음 각 호의 요건을 갖추어야 한다
제5조 (계약) 당사와 Agency 는 위수탁계약을 체결한다"""
_TABLE = """한국마사회 말산업연구소 12억원 2024
농림축산식품부 축산정책과 30억원 2024
제주특별자치도 축산과 5억원 2023
렛츠런재단 사업팀 3억원 2023"""


@kiwi_required
def test_row_dump_rejects_regulation_prose_and_keeps_a_table():
    assert parse_row_dump(_REGULATION) == []           # words end in particles and verbs: sentences, not cells
    assert len(parse_row_dump(_TABLE)) == 4            # nouns and numbers in steady columns: a table


@kiwi_required
def test_hearst_skips_quoted_titles_and_modifying_hypernyms():
    assert extract_hearst_pairs("「사행행위 등 규제 및 처벌특례법」에 따른 사행행위의 이용 대가") == []
    assert extract_hearst_pairs("오락, 도박, 쇼핑, 채팅, 음란사이트 등 업무와 관련이 없는 인터넷 사이트를 이용하는 행위") == []
    assert extract_hearst_pairs("도박, 사행행위 등 불법행위를 하는 자") == [("도박", "불법행위"), ("사행행위", "불법행위")]
    assert extract_hearst_pairs("한국마사회, 렛츠런재단 등 공공기관은 매년 보고한다.") == [
        ("한국마사회", "공공기관"), ("렛츠런재단", "공공기관")]


@kiwi_required
def test_aliases_need_a_proper_name_the_same_parts_and_a_shared_chunk():
    chunks = {"호텔신라": {"c1", "c2"}, "신라호텔": {"c2"}, "신라 호텔": {"c9"}, "신라호텔식": {"c2"},
              "신용평가정보": {"c3"}, "정보신용평가": {"c3"}, "정보보호": {"c4"}, "보호정보": {"c4"}}
    alias = induce_aliases(list(chunks), chunks)
    assert alias == {"신라호텔": "호텔신라"}      # canonical = the spelling used in more chunks, then shorter
    assert "신라 호텔" not in alias                 # same parts, never seen together: no evidence
    assert "신라호텔식" not in alias                # a suffix is a part, not another spelling
    assert "정보신용평가" not in alias and "보호정보" not in alias   # common-noun compounds: the order is the head


@kiwi_required
def test_generic_head_becomes_a_neighbour_link_not_a_type():
    # 40-chunk corpus: "-여부" names sit in 30 chunks (spread everywhere), "-공원" names in 3.
    names = ["감사여부", "승인여부", "등록여부", "여부", "서울경마공원", "부산경마공원", "공원"]
    chunks_of = {"여부": set(), "공원": set(),
                 "감사여부": {f"c{i}" for i in range(10)}, "승인여부": {f"c{i}" for i in range(10, 20)},
                 "등록여부": {f"c{i}" for i in range(20, 30)},
                 "서울경마공원": {"c1", "c2"}, "부산경마공원": {"c3"}}
    edges, _same, related = induce_head_noun_hierarchy(names, class_labels={"여부", "공원"},
                                                       chunks_of=chunks_of, corpus_chunks=40, max_coverage=0.30)
    assert ("공원", "서울경마공원") in edges and ("공원", "부산경마공원") in edges
    assert not any(p == "여부" for p, _c in edges)
    assert {("감사여부", "여부"), ("승인여부", "여부"), ("등록여부", "여부")} <= set(related)
    # below the corpus-size floor the ratio is not applied: same names, small corpus, plain is-a
    edges_small, _s, _r = induce_head_noun_hierarchy(names, class_labels={"여부", "공원"},
                                                     chunks_of=chunks_of, corpus_chunks=10)
    assert ("여부", "감사여부") in edges_small


@kiwi_required
def test_spread_is_measured_over_documents_when_known():
    # 200 chunks in 20 documents. "-여부" names: 24 chunks (12% of chunks) but spread over 12 documents (60%).
    names = ["감사여부", "승인여부", "여부", "서울경마공원", "부산경마공원", "공원"]
    doc_of = {f"c{i}": f"d{i % 20}" for i in range(200)}
    chunks_of = {"여부": set(), "공원": set(),
                 "감사여부": {f"c{i}" for i in range(0, 24, 2)}, "승인여부": {f"c{i}" for i in range(1, 24, 2)},
                 "서울경마공원": {"c40"}, "부산경마공원": {"c41"}}
    edges, _s, related = induce_head_noun_hierarchy(names, class_labels={"여부", "공원"}, chunks_of=chunks_of,
                                                    corpus_chunks=200, doc_of=doc_of, corpus_docs=20)
    assert not any(p == "여부" for p, _c in edges) and ("감사여부", "여부") in related
    assert ("공원", "서울경마공원") in edges
    edges_by_chunk, _s, _r = induce_head_noun_hierarchy(names, class_labels={"여부", "공원"}, chunks_of=chunks_of,
                                                        corpus_chunks=200)
    assert ("여부", "감사여부") in edges_by_chunk        # by chunk share alone (12%) it would have passed


@kiwi_required
def test_alias_bag_keeps_suffixes_as_parts():
    chunks = {"금융기관": {"c1"}, "금융기관별": {"c1"}}
    assert induce_aliases(list(chunks), chunks) == {}     # "-별" is a part of the name, not a permutation


def test_whole_name_key_is_a_rotation_not_a_permutation():
    # A name the analyzer keeps whole is keyed by its rotation: the two blocks swapped
    # ("호텔신라" / "신라호텔") share a key, letters merely rearranged do not.
    assert _rotation_key("호텔신라") == _rotation_key("신라호텔")
    assert _rotation_key("김민수정") != _rotation_key("김수민정")
    assert _rotation_key("삼성전자") != _rotation_key("전자삼상")
    assert len(_rotation_key("호텔신라")) == 1 and _rotation_key("호텔신라")[0].startswith("\x00")


def test_morph_cache_is_bounded():
    assert _morph_starts.cache_parameters()["maxsize"] == _MORPH_CACHE and _MORPH_CACHE > 0


def test_related_predicate_is_graph_vocabulary():
    assert DEFAULT_RELATED_PREDICATE == "relatedTo"
