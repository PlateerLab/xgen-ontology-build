import json

from xgen_ontology import CallableLLM, build_from_documents
from xgen_ontology.build.extract import DocumentExtractor

_EXTRACTION = {
    "classes": [{"name": "Regulation", "description": "a rule"},
                {"name": "Bank", "description": "an institution"}],
    "object_properties": [{"name": "appliesTo", "domain": "Regulation", "range": "Bank"}],
    "datatype_properties": [{"name": "year", "domain": "Regulation", "range": "xsd:integer"}],
    "entities": [{"entity": "Rule A", "class": "Regulation", "source_chunks": ["k1"]},
                 {"entity": "Acme Bank", "class": "Bank", "source_chunks": ["k1"]}],
    "relations": [{"subject": "Rule A", "predicate": "appliesTo", "object": "Acme Bank",
                   "predicate_type": "ObjectProperty", "source_chunks": ["k1"]}],
    "data_values": [{"entity": "Rule A", "property": "year", "value": "2020",
                     "value_type": "xsd:integer", "source_chunks": ["k1"]}],
}


def _stub(prompt, system=""):
    if "merge_groups" in prompt:
        return json.dumps({"merge_groups": []})
    if "summaries" in prompt:
        return json.dumps({"summaries": []})
    return json.dumps(_EXTRACTION)


def test_document_extractor_typed_output():
    ex = DocumentExtractor(CallableLLM(_stub))
    docs = {"doc1": [{"chunk_id": "k1", "chunk_text": "Rule A applies to Acme Bank since 2020.",
                      "chunk_index": 0}]}
    concepts, instances, relations, data_values = ex.extract(docs)
    assert {c.name for c in concepts.classes} == {"Regulation", "Bank"}
    assert any(i.name == "Rule A" and i.class_name == "Regulation" for i in instances)
    assert relations and relations[0].subject == "Rule A"
    assert data_values and data_values[0].value == "2020"


def test_pipeline_text_plus_table():
    docs = {
        "doc1": "Rule A applies to Acme Bank since 2020.",
        "colors.csv": "color_id,name\n10,Red\n20,Blue",
    }
    onto = build_from_documents(docs, llm=CallableLLM(_stub), mode="llm")
    names = {c.name for c in onto.concepts.classes}
    assert "Regulation" in names and "Colors" in names      # text + table merged
    assert onto.report.llm_calls >= 1
    r = onto.search("which regulation applies to Acme Bank", llm=CallableLLM(_stub))
    assert isinstance(r.answer, str)
