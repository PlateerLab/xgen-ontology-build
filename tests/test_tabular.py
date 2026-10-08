from xgen_ontology import build_from_csv
from xgen_ontology.build.tabular import analyze_tables


def _docs(csv_map):
    return {n: [{"chunk_id": f"{n}#0", "chunk_text": t, "chunk_index": 0}] for n, t in csv_map.items()}


def test_star_schema_fk_and_instances():
    onto = build_from_csv({
        "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20",
        "colors": "color_id,name\n10,Red\n20,Blue",
    })
    assert {c.name for c in onto.concepts.classes} == {"Products", "Colors"}
    # FK products.color_id -> colors becomes an object property
    assert any(p.domain == "Products" and p.range == "Colors" for p in onto.concepts.object_properties)
    # FK value is resolved to the target instance label, not the raw id
    rels = {(r.subject, r.object) for r in onto.relations}
    assert ("Widget", "Red") in rels and ("Gadget", "Blue") in rels


def test_column_type_inference():
    schema = analyze_tables(_docs({"t.csv": "id,price,when\n1,9.5,2020-01-01\n2,8.0,2020-02-02"}))
    types = schema["tables"]["t.csv"]["column_types"]
    assert types["price"] == "xsd:decimal"
    assert types["when"] == "xsd:date"


def test_fact_table_kept_as_schema_only():
    # a junction/fact table (2 FKs, many rows) should not be instantiated
    rows = "\n".join(f"{i},{i%3+1},{i%2+1}" for i in range(300))
    onto = build_from_csv({
        "sales": "sale_id,product_id,color_id\n" + rows,
        "products": "product_id,name\n1,A\n2,B\n3,C",
        "colors": "color_id,name\n1,Red\n2,Blue",
    })
    inst_classes = {i.class_name for i in onto.instances}
    assert "Sales" not in inst_classes      # fact table: schema only
    assert "Products" in inst_classes and "Colors" in inst_classes
    assert any(c.name == "Sales" for c in onto.concepts.classes)  # class still declared


# ── 0.13: sheets, spans, TSV, foreign-key evidence (develop table_text / csv_schema_analyzer) ──

from xgen_ontology.build.tabular import table_cell_rows  # noqa: E402


def test_each_sheet_of_a_workbook_is_a_table():
    docs = {"book.xlsx": [
        {"chunk_id": "s1", "chunk_text": "[Sheet: 부서]\ndept_id,부서명\nD1,경영지원\nD2,영업", "chunk_index": 0},
        {"chunk_id": "s2", "chunk_text": "[Sheet: 직원]\nemp_id,이름,dept_id\nE1,김철수,D1\nE2,이영희,D2", "chunk_index": 1},
    ]}
    tables = analyze_tables(docs)["tables"]
    assert sorted(t["table_name"] for t in tables.values()) == ["부서", "직원"]


def test_tsv_and_spans_are_read():
    assert table_cell_rows("a\tb\n1\t2") == [["a", "b"], ["1", "2"]]
    rows = table_cell_rows("<table><tr><td rowspan='2'>A</td><td>x</td></tr><tr><td>y</td></tr></table>")
    assert rows == [["A", "x"], ["", "y"]]


def test_repeated_header_rows_are_not_data():
    docs = {"t.csv": [{"chunk_id": "c0", "chunk_text": "id,name\n1,Alpha", "chunk_index": 0},
                      {"chunk_id": "c1", "chunk_text": "id,name\n2,Beta", "chunk_index": 1}]}
    t = analyze_tables(docs)["tables"]["t.csv"]
    assert t["row_count_estimate"] == 2 and t["sample_values"]["id"] == ["1", "2"]


def test_numbers_alone_do_not_make_a_foreign_key():
    # serial ids of two unrelated tables overlap by construction
    schema = analyze_tables(_docs({
        "orders.csv": "seq,item\n1,pen\n2,ink\n3,cap",
        "visits.csv": "no,place\n1,Seoul\n2,Busan\n3,Jeju",
    }))
    assert schema["fk_relations"] == []


# ── 0.14: the rows of a database table, with the schema declared (develop build_from_table_rows) ──

from datetime import datetime  # noqa: E402

import pytest  # noqa: E402

from xgen_ontology import (  # noqa: E402
    OntologyBuilder,
    build_from_db_rows,
    build_from_rows,
    normalize_fk_relations,
)

_ROWS = [
    {"id": 1, "name": "Red", "hex": "#f00", "active": True, "since": datetime(2020, 1, 1, 9, 0),
     "group_id": 7, "meta": {"a": 1}},
    {"id": 2, "name": "Blue", "hex": None, "active": False, "since": None, "group_id": None, "meta": None},
    {"id": 3, "name": "Green", "hex": "", "active": True, "since": None, "group_id": "", "meta": None},
]
_FK = [{"from_column": "group_id", "to_table": "color_groups", "to_column": "id", "to_pk_column": "id"}]


def test_database_rows_build_from_the_declared_schema_and_python_types():
    c, i, r, dv = build_from_rows("colors", list(_ROWS[0]), _ROWS, source_id="db1:colors", pk_candidates=["id"],
                                  label_column="name", fk_relations=_FK)
    assert [x.name for x in i] == ["Red", "Blue", "Green"]
    assert [x.source_chunks for x in i] == [["db1:colors:1"], ["db1:colors:2"], ["db1:colors:3"]]  # a row is its own source
    types = {d.name: d.range for d in c.datatype_properties}
    assert types == {"id": "xsd:integer", "name": "xsd:string", "hex": "xsd:string", "active": "xsd:boolean",
                     "since": "xsd:dateTime", "meta": "xsd:string"}            # the FK column is no attribute
    vals = {(d.entity, d.property): d.value for d in dv}
    assert vals[("Red", "active")] == "true" and vals[("Red", "since")] == "2020-01-01T09:00:00"
    assert vals[("Red", "meta")] == '{"a": 1}' and ("Blue", "hex") not in vals   # NULL is not recorded
    rel = {(x.subject, x.predicate, x.object) for x in r}
    assert ("Red", "Colors_group_id", "color_groups_7") in rel
    assert not any(s == "Blue" for s, _p, _o in rel)                            # a NULL key points nowhere
    assert ("Green", "Colors_group_id", "color_groups_") in rel                  # an empty string still points
    assert any(op.name == "Colors_group_id" and op.range == "ColorGroups" for op in c.object_properties)


def test_a_declared_foreign_key_is_checked():
    with pytest.raises(ValueError, match="lacks"):
        normalize_fk_relations([{"from_column": "group_id"}], "colors", ["id", "group_id"])
    with pytest.raises(ValueError, match="not a column"):
        normalize_fk_relations([{**_FK[0], "from_column": "nope"}], "colors", ["id", "group_id"])
    with pytest.raises(ValueError, match="primary key"):
        normalize_fk_relations([{**_FK[0], "to_column": "name"}], "colors", ["id", "group_id"])
    assert normalize_fk_relations(_FK, "colors", ["id", "group_id"])[0]["from_table"] == "colors"


def test_rows_without_a_key_are_numbered_and_types_can_be_declared():
    c, i, _r, _dv = build_from_rows("t", ["code", "name"], [{"code": "A", "name": "Alpha"}, {"code": "B", "name": "Beta"}],
                                    source_id="db:t", column_types={"code": "xsd:string", "name": "xsd:string"})
    assert [x.source_chunks for x in i] == [["db:t:row0"], ["db:t:row1"]]
    assert {d.range for d in c.datatype_properties} == {"xsd:string"}


def _colors(onto):
    return {i.name for i in onto.instances if i.class_name == "Colors"}


def test_the_builder_loads_rows_and_replaces_a_row_loaded_before():
    builder = OntologyBuilder(chunk=False)
    onto = builder.build_rows("colors", list(_ROWS[0]), _ROWS, source_id="db1:colors", pk_candidates=["id"],
                              label_column="name", fk_relations=_FK)
    assert _colors(onto) == {"Red", "Blue", "Green"} and onto.report.llm_calls == 0
    assert sorted(c.id for c in onto.chunks) == ["db1:colors:1", "db1:colors:2", "db1:colors:3"]
    assert "hex: #f00" in next(c.text for c in onto.chunks if c.id == "db1:colors:1")
    changed = [{**_ROWS[0], "hex": "#ff0000"}]
    builder.extend_rows(onto, "colors", list(_ROWS[0]), changed, source_id="db1:colors", pk_candidates=["id"],
                        label_column="name", fk_relations=_FK)
    hexes = sorted(d.value for d in onto.data_values if d.entity == "Red" and d.property == "hex")
    assert hexes == ["#ff0000"] and len(onto.chunks) == 3                  # replaced, not added next to the old
    builder.retract(onto, ["db1:colors:2"])
    assert _colors(onto) == {"Red", "Green"}                               # a row that left the table leaves
    assert _colors(build_from_db_rows("colors", list(_ROWS[0]), _ROWS, source_id="x", pk_candidates=["id"],
                                      label_column="name")) == {"Red", "Blue", "Green"}
