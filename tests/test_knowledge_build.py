"""Producer tests: portable IDs, real pipeline, isolated independent consumers."""
from dataclasses import replace

import pytest

from xgen_ontology import (
    OntologyBuilder,
    assemble_resource_fragments,
    build_knowledge,
    build_resource_fragment,
    export_knowledge,
)
from xgen_ontology.knowledge import (
    ContractError,
    KnowledgeBundle,
    OperationCancelled,
    OperationContext,
    Resource,
    SourceChunk,
)


def source(*, revision="v1", text="id,name\n1,Red\n2,Blue", snapshot="input"):
    return KnowledgeBundle("demo", snapshot, resources=(Resource("root", "v1", "root", "directory"),
                           Resource("colors", revision, "colors.csv", parent_id="root")),
                           chunks=(SourceChunk(f"colors:{revision}", "colors", revision, f"parser:{revision}",
                                               text, locator={"kind": "page", "page": 2}),))


def test_real_build_round_trip_preserves_source(tmp_path):
    original = source()
    built = build_knowledge(original, snapshot_id="built-v1")
    assert built.facts and built.entities
    assert built.resources == original.resources and built.chunks == original.chunks
    assert "graph" in built.components
    assert all(f.evidence_ids for f in built.facts)
    assert {e.chunk_id for e in built.evidence} == {"colors:v1"}
    path = tmp_path / "knowledge.json"
    built.dump(path)
    assert KnowledgeBundle.load(path).to_dict() == built.to_dict()
    assert original.facts == ()


def test_replacement_and_deletion_do_not_use_additive_extend():
    first = build_knowledge(source(), snapshot_id="g1")
    second = build_knowledge(source(revision="v2", text="id,name\n1,Green\n2,Yellow"), snapshot_id="g2")
    assert "Red" in {e.label for e in first.entities}
    assert "Red" not in {e.label for e in second.entities}
    assert all(e.chunk_id == "colors:v2" for e in second.evidence)
    empty = build_knowledge(KnowledgeBundle("demo", "deleted"), snapshot_id="g3")
    assert not empty.facts and not empty.entities and not empty.evidence


def test_duplicate_display_names_keep_source_identity():
    a = source()
    b = replace(a, resources=a.resources + (Resource("other", "v1", "colors.csv", parent_id="root"),),
                chunks=a.chunks + (SourceChunk("other:v1", "other", "v1", "parser:other", "id,name\n3,Orange\n4,Pink"),))
    built = build_knowledge(b, snapshot_id="g")
    assert {c.resource_id for c in built.chunks} == {"colors", "other"}
    assert {e.chunk_id for e in built.evidence} == {"colors:v1", "other:v1"}
    assert {"Red", "Orange"} <= {e.label for e in built.entities}


def test_explicit_enrichment_never_silently_falls_back():
    with pytest.raises(ContractError, match="requires an LLM"):
        build_knowledge(source(), snapshot_id="g", builder=OntologyBuilder(mode="enrich"))


def test_cancelled_build_has_no_artifact_and_does_not_mutate_builder():
    context = OperationContext()
    context.cancelled.set()
    builder = OntologyBuilder()
    with pytest.raises(OperationCancelled):
        build_knowledge(source(), snapshot_id="g", builder=builder, context=context)
    assert "_tick" not in builder.__dict__


def test_cancellation_from_progress_is_not_swallowed():
    context = OperationContext()
    def progress(stage, detail):
        context.cancelled.set()
    with pytest.raises(OperationCancelled):
        build_knowledge(source(), snapshot_id="g", builder=OntologyBuilder(progress=progress), context=context)


def test_export_rejects_wrong_source_version():
    onto = OntologyBuilder().build({"colors.csv": [{"chunk_id": "colors:v1", "chunk_text": "changed"}]})
    with pytest.raises(ContractError, match="absent/different"):
        export_knowledge(onto, source(), snapshot_id="g")


def test_resource_fragments_assemble_and_retract_without_embeddings():
    first_source = source()
    second_source = replace(
        first_source,
        resources=(first_source.resources[0], Resource("other", "v7", "other.csv", parent_id="root")),
        chunks=(SourceChunk("other:v7", "other", "v7", "parser:other", "id,name\n3,Orange\n4,Pink"),),
    )
    first = build_resource_fragment(first_source, snapshot_id="fragment:first")
    second = build_resource_fragment(second_source, snapshot_id="fragment:second")
    assert not first.embeddings and "embeddings" not in first.components

    combined_source = replace(
        first_source,
        resources=first_source.resources + (second_source.resources[1],),
        chunks=first_source.chunks + second_source.chunks,
        snapshot_id="source:combined",
    )
    combined = assemble_resource_fragments(combined_source, (second, first), snapshot_id="graph:combined")
    assert {"Red", "Orange"} <= {entity.label for entity in combined.entities}
    assert combined.extensions["xgen_ontology.assembly"]["fragment_count"] == 2

    retracted_source = replace(combined_source, resources=first_source.resources, chunks=first_source.chunks)
    retracted = assemble_resource_fragments(retracted_source, (first,), snapshot_id="graph:retracted")
    assert "Orange" not in {entity.label for entity in retracted.entities}


def test_resource_fragment_rejects_stale_and_duplicate_revisions():
    fragment = build_resource_fragment(source(), snapshot_id="fragment")
    with pytest.raises(ContractError, match="duplicate resource fragment"):
        assemble_resource_fragments(source(), (fragment, fragment), snapshot_id="graph")
    with pytest.raises(ContractError, match="current resource revision"):
        assemble_resource_fragments(source(revision="v2"), (fragment,), snapshot_id="graph")
