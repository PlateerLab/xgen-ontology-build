"""Deterministic table -> ontology -> RDF, with no LLM and no infrastructure."""
import sys

sys.path.insert(0, "src")  # run from repo root without installing

from xgen_ontology_build import build_from_csv  # noqa: E402

onto = build_from_csv({
    "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20\n3,Gizmo,10",
    "colors":   "color_id,name\n10,Red\n20,Blue",
})

print("stats     :", onto.stats())
print("classes   :", [c.name for c in onto.concepts.classes])
print("relations :", [(r.subject, r.predicate, r.object) for r in onto.relations])
print("quality   :", onto.quality()["score"])
print("communities:", onto.communities())
print("\n--- Turtle ---")
print(onto.to_turtle())
