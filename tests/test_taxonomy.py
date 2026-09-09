import pytest

from xgen_ontology import build_from_documents
from xgen_ontology.build.taxonomy import (
    extract_hearst_pairs,
    hearst_hierarchy,
    induce_head_noun_hierarchy,
    induce_hierarchy,
    prose_only,
)
from xgen_ontology.korean import clean_name, get_kiwi, is_sentence_like, strip_list_markers

kiwi_required = pytest.mark.skipif(get_kiwi() is None, reason="kiwipiepy not installed")


# ───────────────────────── korean.py ─────────────────────────


def test_strip_list_markers_leading_and_trailing():
    assert strip_list_markers("01. 말산업 활성화를 위한 연구") == "말산업 활성화를 위한 연구"
    assert strip_list_markers("(1) 경영주 (관리직)") == "경영주 (관리직)"
    assert strip_list_markers("*수료증번호") == "수료증번호"
    assert strip_list_markers("경영지표(1)") == "경영지표"


def test_strip_list_markers_does_not_touch_real_content():
    # a single Hangul syllable glued to a period is not a list marker
    assert strip_list_markers("내.외부") == "내.외부"
    assert strip_list_markers("일.숙직비") == "일.숙직비"
    # a company-name abbreviation in parens
    assert strip_list_markers("(주)마장") == "(주)마장"


@kiwi_required
def test_clean_name_rejects_sentence_fragments():
    assert clean_name("노동생산성") == "노동생산성"
    assert clean_name("확인하는 절차를 진행한다") is None
    assert is_sentence_like("본사에 보고하여야 한다")


# ───────────────────────── Hearst patterns ─────────────────────────

_HEARST_TEXT = (
    "혈액, 모근 등의 생체시료를 채취한다. "
    "모바일, 인터넷뱅킹 등 대체채널을 통한 거래가 늘고 있다."
)
# Unrelated filler so a repeated hypernym doesn't trip max_coverage (30%) in a
# fixture this small -- real corpora are large enough that this isn't an issue,
# but a handful of test sentences needs padding to stay realistic.
_FILLER_TEXTS = [
    "오늘 회의는 예정대로 진행되었다.",
    "다음 분기 예산안을 검토해야 한다.",
    "고객 문의는 접수 후 이틀 내 처리한다.",
    "신규 지점 개설을 위한 부지를 선정했다.",
    "직원 교육은 매월 첫째 주에 실시한다.",
    "계약서 검토는 법무팀이 담당한다.",
]


@kiwi_required
def test_extract_hearst_pairs_reads_hyponym_hypernym():
    pairs = extract_hearst_pairs(_HEARST_TEXT)
    assert ("혈액", "생체시료") in pairs
    assert ("모근", "생체시료") in pairs
    assert ("모바일", "대체채널") in pairs


@kiwi_required
def test_hearst_hierarchy_filters_single_occurrence_pairs():
    # "생체시료" has 2 distinct hyponyms (passes min_hyponyms=2); "보호구" has
    # only one and should be filtered regardless of coverage.
    texts = [_HEARST_TEXT, "장갑 등의 보호구를 착용한다.", *_FILLER_TEXTS]
    edges = hearst_hierarchy(texts, min_hyponyms=2)
    assert ("생체시료", "혈액") in edges
    assert ("생체시료", "모근") in edges
    assert not any(parent == "보호구" for parent, _ in edges)  # only 1 hyponym


@kiwi_required
def test_hearst_ignores_table_markup():
    # the exact failure mode this guards against: a materials-cost line item
    # in an HTML table cell must not become the is-a parent of its neighbors.
    table_text = "<table><tr><td>감사패, 상패 등의 제작비</td></tr></table>"
    assert extract_hearst_pairs(prose_only(table_text)) == []
    # same source folded into a larger chunk alongside real prose
    mixed = "예산 편성 기준은 다음과 같다. " + table_text + " 세부 항목은 별도 공지한다."
    pairs = extract_hearst_pairs(prose_only(mixed))
    assert not any(hyper == "제작비" for _hypo, hyper in pairs)


def test_prose_only_strips_pipe_tables():
    text = "설명 문단입니다.\n| a | b |\n| - | - |\n| 1 | 2 |\n결론 문단입니다."
    cleaned = prose_only(text)
    assert "|" not in cleaned
    assert "설명 문단입니다." in cleaned
    assert "결론 문단입니다." in cleaned


def test_prose_only_leaves_plain_prose_untouched():
    text = "이것은 표가 아닌 평범한 문단입니다."
    assert prose_only(text) == text


# ───────────────────────── head-noun decomposition ─────────────────────────


@kiwi_required
def test_induce_head_noun_hierarchy_finds_compound_suffix():
    edges, rename = induce_head_noun_hierarchy(["전사경마사업", "사업", "상임감사실", "감사실"])
    assert ("사업", "전사경마사업") in edges
    assert ("감사실", "상임감사실") in edges
    assert rename == {}


def test_induce_head_noun_hierarchy_keeps_spaced_multiword_names_intact():
    # "Korea Racing Authority" must not be split just because "Korea" is a
    # shorter existing name -- there is no modifier/head relationship here.
    edges, _rename = induce_head_noun_hierarchy(["한국 마사회", "마사회", "한국"])
    assert edges == []


def test_induce_head_noun_hierarchy_folds_code_prefixed_duplicates():
    edges, rename = induce_head_noun_hierarchy(
        ["NA162000.23 제주목장사업", "제주목장사업"]
    )
    assert rename == {"NA162000.23 제주목장사업": "제주목장사업"}
    assert edges == []  # a rename, not a hierarchy edge


def test_induce_head_noun_hierarchy_rejects_prefix_modifiers():
    # a modifier in *front* is not evidence of anything ("product catalog" is
    # a kind of catalog, not a kind of product) -- only suffix matches count.
    edges, _rename = induce_head_noun_hierarchy(["상품목록", "상품"])
    assert edges == []


# ───────────────────────── orchestrator ─────────────────────────


@kiwi_required
def test_induce_hierarchy_mints_new_classes_from_hearst():
    from xgen_ontology.models import Class, Concepts

    concepts = Concepts(classes=[Class(name="혈액"), Class(name="모근")])
    counts = induce_hierarchy(concepts, texts=[_HEARST_TEXT, *_FILLER_TEXTS])
    names = {c.name for c in concepts.classes}
    assert "생체시료" in names  # minted, wasn't a pre-existing class
    assert ("생체시료", "혈액") in concepts.class_hierarchy
    assert counts["hearst_classes_added"] >= 1


def test_induce_hierarchy_disabled_by_flag_produces_flat_schema():
    docs = {f"doc{i}.txt": t for i, t in enumerate([_HEARST_TEXT, *_FILLER_TEXTS])}
    onto = build_from_documents(docs, llm=None, hierarchy=False)
    assert onto.concepts.class_hierarchy == []


@kiwi_required
def test_build_from_documents_induces_hierarchy_end_to_end():
    docs = {f"doc{i}.txt": t for i, t in enumerate([_HEARST_TEXT, *_FILLER_TEXTS])}
    onto = build_from_documents(docs, llm=None)  # zero LLM calls
    assert ("생체시료", "혈액") in onto.concepts.class_hierarchy
    assert any("induced" in note for note in onto.report.notes)
