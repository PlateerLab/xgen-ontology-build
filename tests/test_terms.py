"""Term layer: what may stand as a name at all."""
from xgen_ontology.build.deterministic import _acceptable, _phrase_pieces, _pipe_caption, looks_like_header


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
