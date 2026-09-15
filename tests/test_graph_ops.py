from xgen_ontology import clean_hierarchy, louvain_communities, review_quality, to_turtle
from xgen_ontology.build.emit import to_rdf_triples
from xgen_ontology.models import Class, Concepts, DataProperty, DataValue, Instance, ObjectProperty, Relation


def test_louvain_two_communities():
    # two triangles joined by a single bridge edge
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = [("a", "b"), ("b", "c"), ("a", "c"),
             ("d", "e"), ("e", "f"), ("d", "f"),
             ("c", "d")]
    comm = louvain_communities(nodes, edges)
    assert comm["a"] == comm["b"] == comm["c"]
    assert comm["d"] == comm["e"] == comm["f"]
    assert comm["a"] != comm["d"]


def test_clean_hierarchy_drops_property_parent_and_cycles():
    concepts = Concepts(
        classes=[Class("Animal"), Class("Dog", parent="Animal"), Class("Cat", parent="livesIn")],
        object_properties=[ObjectProperty("livesIn", "Animal", "Place")],
        class_hierarchy=[("Animal", "Dog"), ("livesIn", "Cat"), ("Dog", "Animal")],
    )
    clean_hierarchy(concepts)
    h = set(concepts.class_hierarchy)
    assert ("Animal", "Dog") in h          # real is-a kept
    assert ("livesIn", "Cat") not in h     # property-as-parent dropped
    assert ("Dog", "Animal") not in h      # cycle edge dropped


def test_quality_flags_dangling_and_completeness():
    concepts = Concepts(
        classes=[Class("Person"), Class("Empty")],
        object_properties=[ObjectProperty("knows", "Person", "Person")],
    )
    instances = [Instance("Alice", "Person"), Instance("Bob", "Person")]
    relations = [Relation("Alice", "knows", "Bob"), Relation("Alice", "knows", "Ghost")]
    q = review_quality(concepts, instances, relations, [])
    assert q["dangling_edge_count"] == 1     # Ghost is not a node
    assert q["classes_without_instance"] == 1  # Empty
    assert 0 <= q["score"] <= 100


def test_emit_turtle_round_fields():
    concepts = Concepts(
        classes=[Class("Person")],
        datatype_properties=[DataProperty("age", "Person", "xsd:integer")],
    )
    instances = [Instance("Alice", "Person")]
    data_values = [DataValue("Alice", "age", "30", "xsd:integer")]
    ttl = to_turtle(to_rdf_triples(concepts, instances, [], data_values))
    assert "@prefix owl:" in ttl
    assert "a owl:Class" in ttl
    assert '"30"^^xsd:integer' in ttl
