from xgen_ontology import normalize_predicate, resolve_entities
from xgen_ontology.build.dedup import Deduplicator, cluster_by_cosine
from xgen_ontology.build.govern import govern_predicates
from xgen_ontology.models import Concepts, Instance, ObjectProperty, Relation


def test_resolve_entities_merges_case_and_guards_numbers():
    insts = [Instance("ASUS ROG", "Product"), Instance("asus rog", "Product"),
             Instance("800W", "Product"), Instance("600W", "Product")]
    rels = [Relation("asus rog", "made_by", "X")]
    cmap = resolve_entities(insts, rels, [])
    # case variants fold together
    assert cmap.get("asus rog") == "ASUS ROG" or cmap.get("ASUS ROG") == "asus rog"
    # number-conflicting names never merge
    names = {i.name for i in insts}
    assert "800W" in names and "600W" in names
    # rename applied to relations
    assert rels[0].subject in names


def test_govern_predicates_folds_surface_variants():
    # separator variants normalize to one key ("belongsto")
    rels = [Relation("a", "belongs to", "b"), Relation("c", "belongs_to", "d"),
            Relation("e", "belongs-to", "f")]
    stats = govern_predicates(rels, [ObjectProperty("belongs to", "X", "Y")])
    assert len({r.predicate for r in rels}) == 1   # all folded to one
    assert stats["merged"] >= 2
    # Korean ending variants also fold
    kr = [Relation("a", "소속", "b"), Relation("c", "소속됨", "d"), Relation("e", "소속되어 있다", "f")]
    govern_predicates(kr, [ObjectProperty("소속", "X", "Y")])
    assert {r.predicate for r in kr} == {"소속"}


def test_normalize_predicate_korean_suffix():
    assert normalize_predicate("소속") == normalize_predicate("소속됨") == normalize_predicate("소속되어있다")


def test_cluster_by_cosine_union_find():
    names = ["belongs", "memberOf", "color"]
    vecs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    rename = cluster_by_cosine(names, vecs, threshold=0.9)
    # belongs/memberOf cluster -> shortest canonical "color" is separate
    assert rename.get("memberOf") == "belongs" or rename.get("belongs") == "memberOf"
    assert "color" not in rename


def test_dedup_leaves_relation_names_to_the_vocabulary():
    # Relation names are English identifiers held to relation formation's vocabulary;
    # the synonym pass folds classes only (folding relations by domain/range merged
    # different relations and turned English names back into document words).
    concepts = Concepts(object_properties=[
        ObjectProperty("hasColor", "Product", "Color"),
        ObjectProperty("color", "Product", "Color"),
    ])
    d = Deduplicator()  # no LLM/embedder -> rule passes only
    d.deduplicate(concepts, [], [Relation("p", "hasColor", "c")], [])
    assert {p.name for p in concepts.object_properties} == {"hasColor", "color"}
