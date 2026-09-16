"""Contract copies must stay byte-identical to their published source."""
import hashlib
import importlib.resources
import json
import subprocess
import sys
from pathlib import Path

from xgen_ontology.knowledge import KnowledgeBundle, Resource, SourceChunk, canonical_json, digest


def test_packaged_contract_digests():
    package = importlib.resources.files("xgen_ontology")
    lock = json.loads(package.joinpath("knowledge.lock.json").read_text())
    for filename, key in (("knowledge.py", "records_sha256"), ("knowledge.schema.json", "schema_sha256")):
        assert hashlib.sha256(package.joinpath(filename).read_bytes()).hexdigest() == lock[key]


def test_contract_source_matches_generated_files():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/sync_knowledge_contract.py"), "--check"], check=True)


def test_build_import_does_not_import_search_or_other_package():
    root = Path(__file__).resolve().parents[1]
    code = f"import sys; sys.path.insert(0, {str(root / 'src')!r}); import xgen_ontology; " \
           "assert 'omnifuse' not in sys.modules; assert 'xgen_ontology.search.oneshot' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-S", "-c", code], check=True)


def test_consumer_fixture():
    root = Path(__file__).resolve().parents[1]
    bundle = KnowledgeBundle.load(root / "tests/fixtures/knowledge-v1.json")
    assert bundle.facts and bundle.chunks[0].revision == "v1"


def test_producer_fixture_matches_current_builder(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "fresh.json"
    subprocess.run([sys.executable, str(root / "examples/knowledge_build.py"), str(output)], check=True)
    assert KnowledgeBundle.load(output).to_dict() == KnowledgeBundle.load(root / "tests/fixtures/knowledge-v1.json").to_dict()


def test_portable_json_composes_pairs_and_replaces_isolated_surrogates():
    raw_pair = "\ud83d\ude00"
    actual_scalar = "\U0001f600"
    value = {"nested": [{raw_pair: raw_pair + "-ok"}], "isolated": "a\ud800b\udc00c"}

    encoded = canonical_json(value).encode("utf-8")
    decoded = json.loads(encoded)

    assert decoded["nested"] == [{actual_scalar: actual_scalar + "-ok"}]
    assert decoded["isolated"] == "a\ufffdb\ufffdc"
    assert digest(raw_pair) == digest(actual_scalar)


def test_bundle_serialization_normalizes_parser_surrogates():
    bundle = KnowledgeBundle(
        "corpus", "snapshot",
        resources=(Resource("resource", "v1", "document.txt"),),
        chunks=(SourceChunk("chunk", "resource", "v1", "parser", "title \ud83d\ude00"),),
    )

    data = bundle.to_dict()

    assert data["chunks"][0]["text"] == "title \U0001f600"
    json.dumps(data, ensure_ascii=False).encode("utf-8")
