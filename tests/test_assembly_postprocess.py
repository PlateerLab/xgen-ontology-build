"""Assembly post-build: parts built one document at a time, judged over the whole corpus (0.17.0)."""
from xgen_ontology_build import (
    OntologyBuilder,
    assemble_resource_fragments,
    build_knowledge,
    build_resource_fragment,
    export_knowledge,
    import_knowledge,
    table_bound_names,
)
from xgen_ontology_build.exchange.knowledge import KnowledgeBundle, Resource, SourceChunk

# A pipe table per chunk: the entities come from cells (no morphology needed), the column header
# types them. "공통부서" sits in every chunk of every file: a flood over the corpus, but never
# over one file's chunks alone, which are too few for the coverage judgement.
_ROW = "| 부서 | 담당자 |\n|---|---|\n| 공통부서 | 김{n} |\n| {dept} | 이{n} |"


def _corpus(files=3, chunks_per_file=8):
    resources = [Resource("root", "v1", "root", "directory")]
    chunks = []
    for f in range(files):
        rid = f"doc{f}"
        resources.append(Resource(rid, "v1", f"doc{f}.txt", parent_id="root"))
        for n in range(chunks_per_file):
            k = f * chunks_per_file + n
            chunks.append(SourceChunk(f"{rid}:{n}", rid, "v1", "parser:v1",
                                      _ROW.format(n=k, dept=f"영업{f}팀"), ordinal=n))
    return KnowledgeBundle("corp", "src", resources=tuple(resources), chunks=tuple(chunks))


def _fragment_of(source, rid):
    part = KnowledgeBundle("corp", "src", resources=tuple(r for r in source.resources if r.id in ("root", rid)),
                           chunks=tuple(c for c in source.chunks if c.resource_id == rid))
    return build_resource_fragment(part, snapshot_id=f"frag:{rid}")


def _labels(bundle):
    return {e.label for e in bundle.entities if e.kind == "instance"}


def test_import_knowledge_round_trips_a_built_graph():
    source = _corpus(files=1, chunks_per_file=3)
    built = build_knowledge(source, snapshot_id="g1")
    again = export_knowledge(import_knowledge(built), source, snapshot_id="g1")
    assert {(e.id, e.kind, e.label) for e in again.entities} == {(e.id, e.kind, e.label) for e in built.entities}
    assert {(f.id, f.assertion) for f in again.facts} == {(f.id, f.assertion) for f in built.facts}
    assert {e.id: set(e.evidence_ids) for e in again.entities} == {e.id: set(e.evidence_ids) for e in built.entities}


def test_assembly_without_postprocess_keeps_each_part_as_built():
    source = _corpus()
    fragments = [_fragment_of(source, f"doc{f}") for f in range(3)]
    assembled = assemble_resource_fragments(source, fragments, snapshot_id="asm")
    # each file has 8 chunks: below the coverage minimum, so the flood survives in every fragment
    assert "공통부서" in _labels(assembled)
    assert assembled.extensions["xgen_ontology.assembly"]["fragment_count"] == 3


def test_assembly_postprocess_judges_the_flood_over_the_whole_corpus():
    source = _corpus()
    fragments = [_fragment_of(source, f"doc{f}") for f in range(3)]
    whole = build_knowledge(source, snapshot_id="whole")
    assembled = assemble_resource_fragments(source, fragments, snapshot_id="asm", postprocess=OntologyBuilder())
    assert "공통부서" not in _labels(whole)          # 24 chunks: the extractor dropped the flood
    assert "공통부서" not in _labels(assembled)      # the assembly post-build made the same judgement
    assert _labels(assembled) == _labels(whole)
    meta = assembled.extensions["xgen_ontology.assembly"]
    assert meta["postprocessed"] is True and meta["fragment_count"] == 3 and meta["mode"] == "basic"
    assert assembled.extensions["xgen_ontology.build"]["report"]["chunks"] == 24
    assert all(f.evidence_ids for f in assembled.facts)
    assert {e.chunk_id for e in assembled.evidence} == {c.id for c in source.chunks}


def test_table_bound_names_are_protected_from_the_post_build():
    source = KnowledgeBundle("corp", "src", resources=(
        Resource("root", "v1", "root", "directory"),
        Resource("t", "v1", "codes.csv", parent_id="root"),
        Resource("d", "v1", "memo.txt", parent_id="root"),
    ), chunks=(
        SourceChunk("t:0", "t", "v1", "p", "code,name\nA1,alpha\nA2,beta", ordinal=0),
        SourceChunk("d:0", "d", "v1", "p", "| 부서 | 담당자 |\n|---|---|\n| 총무 | 박 |", ordinal=0),
    ))
    fragments = [_fragment_of(source, "t"), _fragment_of(source, "d")]
    assembled = assemble_resource_fragments(source, fragments, snapshot_id="asm")
    bound = table_bound_names(assembled)
    assert "alpha" in bound and "총무" not in bound
    kept = assemble_resource_fragments(source, fragments, snapshot_id="asm2", postprocess=OntologyBuilder())
    assert {"alpha", "beta"} <= {e.label for e in kept.entities}


def test_enrich_fragment_is_seeded_with_the_corpus_vocabulary():
    class Silent:
        def generate(self, prompt, *, system="", timeout=None):
            return "{}"

        def generate_json_meta(self, prompt, *, system="", schema=None, max_tokens=None, timeout=None):
            return "{}", {"truncated": False, "failed": False, "out_tokens": 0}

    source = _corpus(files=1, chunks_per_file=2)
    part = KnowledgeBundle("corp", "src", resources=source.resources, chunks=source.chunks)
    built = build_resource_fragment(part, snapshot_id="f", builder=OntologyBuilder(Silent(), mode="enrich"),
                                    vocabulary=["hasManager", "belongsTo", "hasManager"])
    declared = [op["name"] for op in built.extensions["xgen_ontology.build"]["schema"]["object_properties"]]
    assert declared[:2] == ["hasManager", "belongsTo"]
    plain = build_resource_fragment(part, snapshot_id="f2", builder=OntologyBuilder())
    assert "hasManager" not in [op["name"] for op in plain.extensions["xgen_ontology.build"]["schema"]["object_properties"]]


def test_inferred_facts_carry_the_subject_evidence_not_the_hub():
    """A derived edge into a hub class carries the subject's chunks only: the hub's whole spread is not
    replicated on every edge (167 MiB over a 55-file corpus) and the edge stays visible when one hub chunk is out."""
    from xgen_ontology_build.models import Class, Instance
    from xgen_ontology_build.ontology import Ontology

    chunks = tuple(SourceChunk(f"d:{n}", "d", "v1", "p", f"text {n}", ordinal=n) for n in range(40))
    source = KnowledgeBundle("corp", "src", resources=(Resource("d", "v1", "d.txt"),), chunks=chunks)
    onto = Ontology()
    onto.concepts.classes += [Class(name="법률"), Class(name="특별법", parent="법률")]
    onto.concepts.class_hierarchy.append(("법률", "특별법"))
    onto.instances += [Instance(name=f"법률{n}", class_name="법률", source_chunks=[f"d:{n}"]) for n in range(39)]
    onto.instances.append(Instance(name="특별법1", class_name="특별법", source_chunks=["d:39"]))
    built = export_knowledge(onto, source, snapshot_id="x")
    by_label = {e.label: e for e in built.entities}
    edge = next(f for f in built.facts if f.predicate == "subClassOf")
    assert edge.assertion == "inferred"
    assert set(edge.evidence_ids) == set(by_label["특별법"].evidence_ids)
    assert len(edge.evidence_ids) == 1 and len(by_label["법률"].evidence_ids) == 40   # the edge adds its chunk to both ends
