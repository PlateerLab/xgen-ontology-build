"""Term layer: what may stand as a name at all. A list number, a value or a two-letter
Latin scrap of a wrapped line is not a term; an acronym is."""
from xgen_ontology.build.deterministic import _acceptable, _pipe_caption, looks_like_header


def test_latin_scraps_and_numbers_are_not_names():
    assert not _acceptable("of") and not _acceptable("mm") and not _acceptable("5.")
    assert _acceptable("IT") and _acceptable("DB") and _acceptable("취득일") and _acceptable("Letter of Credit")
    assert not looks_like_header("5.") and not looks_like_header("of") and looks_like_header("구분") and looks_like_header("PF")


def test_pipe_table_caption_needs_a_name_shape():
    # A bare list number above a pipe table is not the table's name (on a real corpus "5."
    # had become a class with 574 rows as its instances).
    assert _pipe_caption("5.\n| 구분 | 값 |\n| a | 1 |") == ""
    assert _pipe_caption("렌탈 품목\n| 구분 | 값 |\n| a | 1 |") == "렌탈 품목"
    assert _pipe_caption("## 심의 기준\n| 구분 | 값 |\n| a | 1 |") == "심의 기준"
