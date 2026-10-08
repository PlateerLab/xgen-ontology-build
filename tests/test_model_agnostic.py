"""Model-agnostic hardening: lenient parse / salvage / key aliases /
self-typed repair / unit verification / split-on-failure."""
import json

from xgen_ontology_build.extract.llm_extract import (
    KO_UNIT_SCALES,
    DocumentExtractor,
    _normalize_keys,
    verify_numeric_units,
)
from xgen_ontology_build.llm import CallableLLM, parse_json_lenient, salvage_truncated
from xgen_ontology_build.models import DataValue


def test_parse_lenient_variants():
    good = {"entities": [{"entity": "A", "class": "Org"}]}
    variants = [
        json.dumps(good),
        "Here is the result:\n```json\n" + json.dumps(good) + "\n```\nDone.",
        '{"entities": [{"entity": "A", "class": "Org"},]}',            # trailing comma
        '// note\n{"entities": [{"entity": "A", "class": "Org"}]}',    # comment
        '{"entities": [{"entity": "A", "class": "Org"}]} trailing prose',
    ]
    for v in variants:
        assert parse_json_lenient(v) == good, v
    # top-level array wraps as entities
    assert parse_json_lenient('[{"entity": "A"}]') == {"entities": [{"entity": "A"}]}


def test_salvage_truncated_keeps_complete_elements():
    cut = '{"classes": [{"name": "Org"}], "entities": [{"entity": "A", "class": "Org"}, {"entity": "B", "cla'
    obj = salvage_truncated(cut)
    assert obj and obj["classes"] == [{"name": "Org"}]
    assert {"entity": "A", "class": "Org"} in obj["entities"]


def test_normalize_keys_aliases():
    out = _normalize_keys({"클래스": [{"name": "Org"}], "instance": [{"entity": "A"}]})
    assert out["classes"] == [{"name": "Org"}]
    assert out["entities"] == [{"entity": "A"}]


def test_verify_numeric_units_grounded():
    dvs = [DataValue(entity="E", property="budget", value="150000000"),   # 15억 off by 10x
           DataValue(entity="E", property="count", value="42")]          # unrelated -> untouched
    fixed = verify_numeric_units(dvs, "예산은 15억원이다", KO_UNIT_SCALES)
    assert fixed == 1
    assert dvs[0].value == "1500000000"
    assert dvs[1].value == "42"


def _chunks(n):
    return [{"chunk_id": f"c{i}", "chunk_text": f"Alpha Corp acquired Beta Inc in 202{i}."}
            for i in range(n)]


def test_self_typed_repair_retypes():
    calls = []

    def fake(prompt, system=""):
        calls.append(prompt)
        if "wrongly typed as themselves" in prompt:
            return json.dumps({"types": [{"entity": "Alpha Corp", "class": "Company"}]})
        return json.dumps({
            "classes": [{"name": "Alpha Corp"}],
            "entities": [{"entity": "Alpha Corp", "class": "Alpha Corp"}],
            "relations": [], "data_values": [],
        })

    ex = DocumentExtractor(CallableLLM(fake))
    concepts, instances, _, _ = ex.extract({"d": _chunks(1)})
    assert instances[0].class_name == "Company"
    names = {c.name for c in concepts.classes}
    assert "Company" in names and "Alpha Corp" not in names
    assert len(calls) == 2  # one extract + one repair


def test_split_on_failure_recovers_half():
    def fake(prompt, system=""):
        # Fails whenever both chunks are in one prompt; succeeds on single-chunk batches.
        if "c0" in prompt and "c1" in prompt:
            return "not json at all"
        return json.dumps({"classes": [], "entities": [{"entity": "A", "class": "Org"}],
                           "relations": [], "data_values": []})

    ex = DocumentExtractor(CallableLLM(fake))
    _, instances, _, _ = ex.extract({"d": _chunks(2)})
    assert len(instances) == 2  # both halves recovered after the combined batch failed
