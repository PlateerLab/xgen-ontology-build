"""Zero-LLM structural extraction: tables (no analyzer needed) and prose (Korean analyzer)."""
import re

import pytest

from xgen_ontology.build import deterministic as dx
from xgen_ontology.korean import get_kiwi

kiwi_required = pytest.mark.skipif(get_kiwi() is None, reason="kiwipiepy not installed")

# The retrieval header an ingestion pipeline may prepend to every chunk. Content, not an entity.
HEADER_RE = re.compile(r"\A\s*이는 '[^\n']*' 콜렉션에 존재하는 [^\n]* 파일의 내용입니다\.[ \t\r]*\n"
                       r"(?:(?:[^\n]*\n){0,2}?\{'keywords':[^\n]*\}[ \t\r]*\n)?")
UPLOAD_HEAD = (
    "이는 'zz_devcheck_01' 콜렉션에 존재하는 A_yeosin.txt 파일의 내용입니다.\n\n"
    "{'keywords': [], 'topics': [], 'entities': [], 'sentiment': '', "
    "'document_type': '', 'complexity_level': '', 'main_concepts': []}\n\n"
)
PROSE = UPLOAD_HEAD + (
    "여신관리규정\r\n"
    "제1조 목적. 이 규정은 여신심사위원회의 심사 절차를 정한다.\r\n"
    "여신심사위원회는 대출 신청을 심사한다. 담보평가부는 담보물건을 평가한다.\r\n"
    "개인신용대출과 기업운전자금대출은 여신상품에 속한다.\n"
)
ROW_DUMP = (
    "부서 담당자 예산 기준일\n"
    "여신심사부 김철수 1200 2026-01-01\n"
    "담보평가부 이영희 800 2026-01-01\n"
    "리스크관리부 박민수 1500 2026-01-01\n"
    "준법감시실 최지훈 600 2026-01-01\n"
)
JEJU = """▣ 주거용 소액임차보증금 지역별 소액보증금 금액
<table border='1'>
  <tr><th colspan="2">구 분</th><th>최우선변제금액</th><th>최우선변제를 받을 수 있는 임차인의 범위</th></tr>
  <tr><td rowspan="2">1987.11.30일 이전 담보취득분</td><td>특별시, 직할시</td><td>100만원 이하</td><td></td></tr>
  <tr><td>기타지역</td><td>50만원 이하</td><td></td></tr>
  <tr><td rowspan="2">2010.07.26일 부터 담보취득분</td><td>서울특별시</td><td>700만원</td><td>5,000만원 이하</td></tr>
  <tr><td>기타지역</td><td>400만원</td><td>4,000만원 이하</td></tr>
</table>
"""
JEONKYUL = """여신 전결권 한도
다음 내용은 여신신규를 취급할 때, 적용하는 여신전결기준표이다.
(단위 : 억원)
<table border='1'>
<tr><th></th><th></th><th>심사역 전결</th><th>여신 협의회 전결</th><th>비고</th></tr>
<tr><td>신용등급</td><td>여신종류</td><td></td><td></td><td></td></tr>
<tr><td rowspan="2">AAA ~ AA+</td><td>신용대출</td><td>3 이하</td><td>25 이하</td><td rowspan="4">제1호 여신 제외</td></tr>
<tr><td>담보대출 (신용대출 금액 포함)</td><td>20 이하</td><td>60 이하</td></tr>
<tr><td rowspan="2">AA ~ A+</td><td>신용대출</td><td>2 이하</td><td>20 이하</td></tr>
<tr><td>담보대출 (신용대출 금액 포함)</td><td>15 이하</td><td>40 이하</td></tr>
</table>
"""


def _names(text):
    return [e for e, _cls in dx.extract_from_chunk(text, header_patterns=[HEADER_RE])[0]]


def _phrases(text):
    return [w for w, piece in dx._noun_phrases(text) if not piece]


# ───────────────────────── table shapes (no analyzer needed) ─────────────────────────


def test_prose_is_not_mistaken_for_a_row_dump():
    assert dx.parse_row_dump(PROSE) == []


def test_row_dump_is_a_table_and_its_entities_survive():
    rows = dx.parse_row_dump(ROW_DUMP)
    assert len(rows) == 5 and rows[1][0] == "여신심사부"
    names = _names(ROW_DUMP)
    assert "여신심사부" in names and "리스크관리부" in names


def test_rowspan_colspan_expand_to_grid():
    rows = dx.parse_html_table(JEJU)
    assert all(len(r) == 4 for r in rows)
    assert rows[0] == ["구분", "구분", "최우선변제금액", "최우선변제를 받을 수 있는 임차인의 범위"]
    assert rows[2][0] == "1987.11.30일 이전 담보취득분" and rows[2][1] == "기타지역"


def test_letter_spaced_cells_are_joined_but_names_are_not():
    for spaced, joined in (("구 분", "구분"), ("비 고", "비고"), ("N C S", "NCS")):
        assert dx._clean(spaced) == joined
    for keep in ("김 철수", "서울 특별시", "2024 년 1 월"):
        assert dx._clean(keep) == keep


def test_table_rows_become_facts():
    f = dx.extract_chunk(JEJU)
    # The subject column is the entity column with the most distinct values (the region),
    # its header types the rows, and the other identifying cell links to it by that header.
    assert ("서울특별시", "최우선변제금액", "700만원") in f.props
    assert ("서울특별시", "최우선변제를 받을 수 있는 임차인의 범위", "5,000만원 이하") in f.props
    assert ("서울특별시", "구분", "2010.07.26일 부터 담보취득분") in f.relations
    classes = {e: c for e, c in f.ents}
    assert classes["서울특별시"] == "구분"
    assert f.heads[2] == "최우선변제금액"
    assert all(len(r) == 4 for r in dx.parse_html_table(JEJU))


def test_value_keyed_table_prefixes_rows_with_the_header():
    html = """<table>
<tr><td>연도</td><td>매출</td><td>영업이익</td></tr>
<tr><td>2019</td><td>1,000억원</td><td>80억원</td></tr>
<tr><td>2020</td><td>1,200억원</td><td>95억원</td></tr>
<tr><td>2021</td><td>1,500억원</td><td>120억원</td></tr>
</table>"""
    f = dx.extract_chunk(html)
    assert ("연도 2020", "매출", "1,200억원") in f.props
    assert ("연도 2020", "연도") in f.ents


def test_data_row_is_not_mistaken_for_a_header():
    html = """<table>
<tr><td>'18년</td><td>실적</td><td>미집계</td></tr>
<tr><td>'19년</td><td>실적</td><td>120</td></tr>
<tr><td>'20년</td><td>실적</td><td>130</td></tr>
<tr><td>'21년</td><td>실적</td><td>140</td></tr>
</table>"""
    f = dx.extract_chunk(html)
    assert "'18년" not in {c for _, c in f.ents}
    assert not f.heads


def test_two_row_header_is_merged_and_identity_uses_both_columns():
    f = dx.extract_chunk(JEONKYUL)
    assert f.heads[:2] == ["신용등급", "여신종류"]
    assert ("AAA ~ AA+ 담보대출", "여신 협의회 전결", "60 이하") in f.props
    assert not any(p[0] == "AAA ~ AA+" for p in f.props)          # rows are not lumped by grade alone
    # a parenthetical that does not tell names apart is decoration and is dropped
    assert not any("(" in p[0] for p in f.props)
    assert "단위 : 억원)" not in {c for _, c in f.ents}           # the bracketed note is not a caption


def test_unit_row_under_the_header_annotates_the_columns():
    html = """<table>
<tr><td>항목</td><td>전년</td><td>당년</td></tr>
<tr><td></td><td>억원</td><td>억원</td></tr>
<tr><td>매출액</td><td>120</td><td>150</td></tr>
<tr><td>영업이익</td><td>12</td><td>18</td></tr>
<tr><td>순이익</td><td>9</td><td>14</td></tr>
</table>"""
    f = dx.extract_chunk(html)
    assert f.heads == ["항목", "전년 억원", "당년 억원"]
    assert ("매출액", "당년 억원", "150") in f.props
    assert not any(e == "억원" for e, _ in f.ents)


def test_bullet_cell_is_not_a_header():
    html = """<table>
<tr><td>구분</td><td>□ 예산 집행지침 I. 일반지침</td><td>금액</td></tr>
<tr><td>인건비</td><td>기본급</td><td>10억원</td></tr>
<tr><td>경비</td><td>여비</td><td>2억원</td></tr>
<tr><td>사업비</td><td>위탁</td><td>7억원</td></tr>
</table>"""
    f = dx.extract_chunk(html)
    assert not f.heads


def test_name_core_strips_codes_glosses_and_decorative_parens():
    assert dx.name_core("NA162000.23 제주목장사업") == "제주목장사업"
    assert dx.name_core("경영주 (관리직)") == "경영주"
    assert dx.name_core("경영주 (관리직)", drop_paren=False) == "경영주 (관리직)"
    assert dx.name_core("감사 - 회계 처리의 적정성을 검토하는 절차") == "감사"
    assert dx.name_core("A1") == "A1"     # nothing usable remains -> unchanged


def test_is_value_and_class_name_gates():
    for v in ("1,200", "2024-01-01", "15%", "3 이하", "'18년", ""):
        assert dx.is_value(v), v
    for n in ("여신심사위원회", "5,000만원 이하 임차인"):
        assert not dx.is_value(n), n
    for junk in ("'18년", "29", "S", "(가) 당일 단속금액 1억원 미만인 경우", ""):
        assert not dx.is_class_name(dx.normalize_label(junk)), junk
    for ok in ("내용", "주거용 소액임차보증금 지역별 소액보증금 금액", "Thing"):
        assert dx.is_class_name(ok), ok


# ───────────────────────── whole-document extraction ─────────────────────────


def test_header_carries_over_to_the_next_chunk_of_the_same_table():
    first = """<table>
<tr><td>기관명</td><td>담당부서</td><td>예산</td></tr>
<tr><td>한국마사회</td><td>말산업연구소</td><td>12억원</td></tr>
<tr><td>농림축산식품부</td><td>축산정책과</td><td>30억원</td></tr>
<tr><td>제주특별자치도</td><td>축산과</td><td>5억원</td></tr>
</table>"""
    second = """<table>
<tr><td>렛츠런재단</td><td>사업팀</td><td>3억원</td></tr>
<tr><td>한국마사회 제주본부</td><td>말산업팀</td><td>2억원</td></tr>
</table>"""
    docs = {"a.docx": [{"chunk_id": "c1", "chunk_text": first, "chunk_index": 0},
                       {"chunk_id": "c2", "chunk_text": second, "chunk_index": 1}]}
    concepts, ner, relations, props = dx.extract_as_dicts(docs, corpus_chunks=50)
    got = {(d["entity"], d["property"], d["value"]) for d in props}
    assert ("렛츠런재단", "예산", "3억원") in got and ("한국마사회", "예산", "12억원") in got
    rel = {(r["subject"], r["predicate"], r["object"]) for r in relations}
    assert ("렛츠런재단", "담당부서", "사업팀") in rel
    assert all(r["source_chunks"] for r in relations) and all(d["source_chunks"] for d in props)
    # one declaration per column name, no domain: shared across the collection
    assert [(p["name"], p["domain"]) for p in concepts["datatype_properties"]] == [("예산", None)]
    assert concepts["object_properties"] == []
    ents = {e["entity"]: e["class"] for es in ner.values() for e in es}
    assert ents["렛츠런재단"] == "기관명"      # the subject column's header types the rows


def test_models_variant_mirrors_the_dict_variant():
    docs = {"a": [{"chunk_id": "c1", "chunk_text": JEONKYUL, "chunk_index": 0}],
            "b": [{"chunk_id": "c2", "chunk_text": JEJU, "chunk_index": 0}]}
    concepts, instances, relations, data_values = dx.extract_deterministic(docs, corpus_chunks=50)
    assert {dp.name for dp in concepts.datatype_properties} >= {"여신 협의회 전결", "최우선변제금액"}
    assert all(dp.domain == "" for dp in concepts.datatype_properties)
    assert ("서울특별시", "최우선변제금액", "700만원") in {
        (d.entity, d.property, d.value) for d in data_values}
    assert any(r.predicate == "구분" for r in relations)
    assert all(i.source_chunks for i in instances)


def test_discriminativeness_cap_only_applies_to_a_sizeable_corpus():
    row = "<table><tr><td>항목</td><td>금액</td></tr><tr><td>공통비</td><td>1억원</td></tr><tr><td>기타</td><td>2억원</td></tr><tr><td>예비비</td><td>3억원</td></tr></table>"
    docs = {"a": [{"chunk_id": f"c{i}", "chunk_text": row, "chunk_index": i} for i in range(25)]}
    _c, ner, _r, _p = dx.extract_as_dicts(docs, corpus_chunks=25)
    names = {e["entity"] for es in ner.values() for e in es}
    assert "공통비" not in names          # in every chunk of a 25-chunk corpus -> says nothing
    _c, ner, _r, _p = dx.extract_as_dicts({"a": docs["a"][:5]}, corpus_chunks=5)
    assert "공통비" in {e["entity"] for es in ner.values() for e in es}


# ───────────────────────── prose (Korean analyzer) ─────────────────────────


@kiwi_required
def test_upload_header_and_particles_do_not_become_entities():
    names = _names(PROSE)
    for junk in ("이는", "파일의", "콜렉션", "keywords", "document"):
        assert not any(junk in n for n in names), names
    for bad in ("여신심사위원회는", "신청을", "정한다.", "규정은"):
        assert bad not in names
    for want in ("여신심사위원회", "담보평가부", "여신상품"):
        assert want in names, names


@kiwi_required
def test_noun_phrases_respect_line_breaks_affixes_and_verbal_use():
    got = _phrases("내부감사규정\n감사위원회는 내부통제 점검을 수행한다.")
    assert "내부감사규정감사위원회" not in got and "내부감사규정" in got and "감사위원회" in got
    assert "수행" not in got                                    # used as a verb, not an entity
    got = _phrases("현물환거래와 선물환거래는 외환상품에 속한다.")
    assert "현물환거래" in got and "선물환거래" in got and "현물" not in got
    assert "준법감시인" in _phrases("준법감시인은 법규 준수를 감시한다.")
    assert "경제적" not in _phrases("경제적 효과가 크다.") and "경제" in _phrases("경제적 효과가 크다.")
    assert "구조경량화" in _phrases("구조경량화 설계")
    assert "감사" in _phrases("감사를 실시한다.")
    assert "품질관리" in _phrases("품질관리한다.")


@kiwi_required
def test_prose_values_read_subject_attribute_value():
    vals = dx._prose_values("한국마사회의 자본금은 500억원이다.")
    assert ("한국마사회", "자본금", "500억원") in vals


@kiwi_required
def test_head_noun_layering_promotes_shared_heads_to_classes():
    text = ("상임감사실은 내부감사를 맡는다. 준법감사실은 법규 준수를 본다. "
            "감사실은 두 부서를 아우른다.")
    docs = {"a": [{"chunk_id": "c1", "chunk_text": text, "chunk_index": 0}]}
    concepts, ner, _r, _p = dx.extract_as_dicts(docs)
    classes = {c["name"]: c for c in concepts["classes"]}
    assert "감사실" in classes                                   # head of two names -> a concept
    ents = {e["entity"]: e["class"] for es in ner.values() for e in es}
    assert ents.get("상임감사실") == "감사실" and ents.get("준법감사실") == "감사실"


@kiwi_required
def test_common_word_detection_uses_the_analyzer_dictionary():
    assert dx.is_common_word("결과") and dx.is_common_word("환경")
    assert not dx.is_common_word("마사회") and not dx.is_common_word("포토레지스트")
    assert not dx.is_common_word("결과 보고")                    # a phrase is never a single common word


@kiwi_required
def test_prose_only_strips_every_table_shape_and_the_header():
    assert "콜렉션" not in dx.prose_only(PROSE, [HEADER_RE]) and "여신관리규정" in dx.prose_only(PROSE, [HEADER_RE])
    assert dx.prose_only(ROW_DUMP) == ""
    assert "<table" not in dx.prose_only(JEJU) and "주거용" in dx.prose_only(JEJU)
    pipe = "| 구분 | 금액 |\n|---|---|\n| 인건비 | 1억 |\n| 경비 | 2억 |\n본문 문장이다."
    assert dx.prose_only(pipe).strip() == "본문 문장이다."
