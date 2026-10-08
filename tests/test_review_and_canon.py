"""0.14: the quality review's production counts, and canonicalization examples by weight."""
from xgen_ontology import canonicalize_predicates, review_quality
from xgen_ontology.models import Class, Concepts, Instance, ObjectProperty, Relation


def test_review_counts_orphan_classes_dangling_ends_and_stated_sources():
    concepts = Concepts(classes=[Class(name="기관"), Class(name="부서"), Class(name="고립")],
                        object_properties=[ObjectProperty(name="hasDepartment", domain="기관", range="부서")])
    instances = [Instance(name="한국마사회", class_name="기관", source_chunks=["c1"]),
                 Instance(name="말산업연구소", class_name="부서", source_chunks=["c2"])]
    relations = [
        Relation("한국마사회", "hasDepartment", "말산업연구소", source_chunks=["c3"]),   # stated: grounded by its own chunk
        Relation("유령", "hasDepartment", "말산업연구소"),                             # a subject that is no node
    ]
    q = review_quality(concepts, instances, relations, [])
    assert q["orphan_class_count"] == 1 and "1 orphan class(es)" in q["warnings"]
    assert q["dangling_edge_count"] == 1                                            # counted on the subject end too
    assert q["grounding_pct"] == 100.0 and q["ungrounded_relations"] == 0


class _Canon:
    def __init__(self):
        self.asked = None
        self.llm_calls = 0

    def canonicalize_names(self, names, vocab):
        self.asked = names
        return {n: "hasDepartment" for n in names}

    def define_from_names(self, names, vocab):
        return []


def test_canonicalization_shows_the_best_evidenced_examples_first():
    concepts = Concepts(object_properties=[ObjectProperty(name="hasDepartment", description="has the department")])
    relations = [Relation("A", "runs", "x", weight=1.0), Relation("B", "runs", "y", weight=3.0),
                 Relation("C", "runs", "z", weight=1.0), Relation("D", "runs", "w", weight=2.0)]
    former = _Canon()
    stats = canonicalize_predicates(concepts, relations, former)
    assert former.asked == {"runs": [("B", "y"), ("D", "w")]}       # weight desc, then subject
    assert stats["renamed"] == 4 and {r.predicate for r in relations} == {"hasDepartment"}
