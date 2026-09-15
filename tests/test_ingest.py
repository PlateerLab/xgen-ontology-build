import json

from xgen_ontology import (
    CallableLLM,
    build_from_files,
    build_from_text,
    chunk_document,
    chunk_text,
    html_to_text,
    load_documents,
)


def test_chunk_text_respects_max_and_overlaps():
    para = "Sentence one is here. Sentence two follows. " * 60   # ~2600 chars, one paragraph
    chunks = chunk_text(para, max_chars=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 80 + 40 for c in chunks)   # window + overlap slack


def test_chunk_text_short_is_single():
    assert chunk_text("short doc") == ["short doc"]


def test_chunk_document_ids():
    parts = chunk_document("doc", "alpha\n\nbeta\n\ngamma", max_chars=5, overlap=0)
    assert len(parts) == 3
    assert [p["chunk_id"] for p in parts] == ["doc#0", "doc#1", "doc#2"]
    assert [p["chunk_index"] for p in parts] == [0, 1, 2]


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>x{}</style></head><body><h1>Title</h1><p>Hello <b>world</b></p>" \
           "<script>evil()</script></body></html>"
    txt = html_to_text(html)
    assert "Title" in txt and "Hello" in txt and "world" in txt
    assert "evil" not in txt and "<" not in txt


def _stub(prompt, system=""):
    if "merge_groups" in prompt:
        return json.dumps({"merge_groups": []})
    return json.dumps({
        "classes": [{"name": "Regulation", "description": "rule"}],
        "entities": [{"entity": "Rule A", "class": "Regulation", "source_chunks": []}],
        "relations": [], "object_properties": [], "datatype_properties": [], "data_values": [],
    })


def test_build_from_text_chunks_and_extracts():
    text = ("Rule A is a key regulation.\n\n" * 40)   # long enough to chunk
    onto = build_from_text(text, llm=CallableLLM(_stub), mode="llm", chunk_size=400, chunk_overlap=50)
    assert len(onto.chunks) > 1                       # it got chunked
    assert any(c.name == "Regulation" for c in onto.concepts.classes)


def test_build_from_files_text_and_table(tmp_path):
    (tmp_path / "policy.txt").write_text("Rule A applies to Acme Bank.", encoding="utf-8")
    (tmp_path / "colors.csv").write_text("color_id,name\n10,Red\n20,Blue", encoding="utf-8")
    onto = build_from_files([str(tmp_path / "policy.txt"), str(tmp_path / "colors.csv")],
                            llm=CallableLLM(_stub), mode="llm")
    names = {c.name for c in onto.concepts.classes}
    assert "Regulation" in names and "Colors" in names

    docs = load_documents([str(tmp_path / "colors.csv")])
    assert "color_id" in docs["colors.csv"]          # csv kept as raw text
