"""Term layer: what may stand as a name at all."""
from xgen_ontology_build.extract.deterministic import (
    _acceptable,
    _phrase_pieces,
    _pipe_caption,
    looks_like_header,
)


def test_a_name_needs_a_letter_and_short_latin_terms_are_names():
    assert not _acceptable("5.") and not _acceptable("12-3") and not looks_like_header("5.")
    for short in ("id", "db", "ip", "Go", "js", "ml", "IT", "DB"):
        assert _acceptable(short) and looks_like_header(short)
    assert _acceptable("취득일") and _acceptable("Letter of Credit")


def test_phrase_pieces_are_hangul_words_only():
    # A Korean word cut from a noun run is a noun; a Latin word cut from a phrase is not
    # a name on its own ("of" from a title), while the whole phrase still is.
    assert _phrase_pieces("여신 심사 기준") == ["여신 심사 기준", "여신", "심사", "기준"]
    assert _phrase_pieces("Letter of Credit") == ["Letter of Credit"]
    assert _phrase_pieces("IT 거버넌스") == ["IT 거버넌스", "거버넌스"]


def test_pipe_table_caption_needs_a_name_shape():
    # A bare list number above a pipe table is not the table's name.
    assert _pipe_caption("5.\n| 구분 | 값 |\n| a | 1 |") == ""
    assert _pipe_caption("렌탈 품목\n| 구분 | 값 |\n| a | 1 |") == "렌탈 품목"
    assert _pipe_caption("## 심의 기준\n| 구분 | 값 |\n| a | 1 |") == "심의 기준"


# ── 0.13: name shape, record dumps, row dumps, table titles (develop !644, f4afbf0) ──

import pytest  # noqa: E402

from xgen_ontology_build.extract.deterministic import (  # noqa: E402
    _is_table_title,
    extract_chunk,
    parse_pipe_table,
    parse_row_dump,
    prose_only,
)
from xgen_ontology_build.postbuild.taxonomy import extract_hearst_pairs  # noqa: E402
from xgen_ontology_build.text.korean import get_kiwi, is_name_shape, normalize_label  # noqa: E402

kiwi_required = pytest.mark.skipif(get_kiwi() is None, reason="kiwipiepy not installed")


def test_name_shape_by_unicode_category():
    # bullets, separators and wrapping brackets are not part of a name
    assert normalize_label("- 대출 심사") == "대출 심사"
    assert normalize_label("「상법」") == "상법" and normalize_label("[상품별 사항]") == "상품별 사항"
    assert normalize_label("사업:") == "사업" and normalize_label("비율 ×") == "비율"
    assert normalize_label("R&amp;D 투자") == "R&D 투자"            # entities resolved first
    # a trailing symbol is not part of a name: a table border, an arrow, an operator. Known limit
    # (as in the production store): a grade suffix goes too, AA+ reads AA.
    assert normalize_label("주민등│") == "주민등" and normalize_label("거래반영→") == "거래반영"
    assert normalize_label("<별표 1>") == "별표 1"
    # cut brackets, sentence ends and line breaks are not names
    for bad in ("제1조(목적", "포함한다.", "여신\n심사", "정보가,"):
        assert not is_name_shape(bad)


def test_record_dump_reads_as_its_text_values():
    dump = '{\n"조문번호": "제3조",\n"항내용": "회사는 내부통제기준을 마련한다",\n"시행일": 20250101,\n"구분": "Y"\n}'
    assert prose_only(dump) == "제3조\n회사는 내부통제기준을 마련한다"
    names = {e for e, _ in extract_chunk(dump).ents}
    assert "항내용" not in names and "조문번호" not in names    # keys are not entities


def test_row_dump_needs_one_width_on_most_lines():
    # A regulation's revision history: varied widths, so it is prose, not a table.
    history = "\n".join(["개정 2016. 02. 24 대표이사", "개정 2018. 05. 11 이사회 결의 사항",
                         "개정 2019. 01. 10", "개정 2020. 01. 01 전면 개정 시행", "부칙 이 규정은 시행한다 당일부터"])
    assert parse_row_dump(history) == []
    assert prose_only(history) == history
    table = "\n".join(["구분 금액 비율 비고", "가산금리 100 10 연", "기준금리 200 20 월", "우대금리 50 5 연", "합계 350 35 -"])
    assert len(parse_row_dump(table)) == 5
    # list items are prose too
    items = "\n".join(["1. 대출 한도 기준 설정", "2. 심사 절차 및 승인", "3. 사후 관리 및 점검", "4. 기타 필요한 사항 처리"])
    assert parse_row_dump(items) == []


def test_escaped_pipe_stays_inside_its_cell():
    rows = parse_pipe_table("| 구분 | 내용 |\n|---|---|\n| A | x \| y |\n| B | z |")
    assert rows[1] == ["A", "x | y"]


def test_a_table_title_names_something():
    assert _is_table_title("3분기 예산") and _is_table_title("렌탈 품목")
    assert not _is_table_title("5.")
    # a converter's "[Table N]" marker is not part of the title
    grid = "\n| 구분 | 값 |\n| a | 1 |"
    assert _pipe_caption("[Table 2] 렌탈 품목" + grid) == "렌탈 품목"
    assert _pipe_caption("[Table 3]" + grid) == ""


@kiwi_required
def test_a_word_glued_to_a_number_does_not_start_a_name():
    names = {e for e, _ in extract_chunk("제2장 내부통제체제를 정비한다").ents}
    assert "장 내부통제체제" not in names and "장" not in names


@kiwi_required
def test_hearst_lists_do_not_cross_a_line():
    pairs = extract_hearst_pairs("은행, 보험사 등 금융기관에 적용한다")
    assert ("은행", "금융기관") in pairs and ("보험사", "금융기관") in pairs
    across = extract_hearst_pairs("은행\n보험사 등 금융기관\n20260101 시행")
    assert ("은행", "금융기관") not in across and all("20260101" not in h for _, h in across)
