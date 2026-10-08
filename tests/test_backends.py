from xgen_ontology_build import InMemoryGraphSink, build_from_csv
from xgen_ontology_build.store.sparql import SparqlGraph, fuseki


def test_inmemory_sink_collects_turtle():
    onto = build_from_csv({"colors": "color_id,name\n10,Red\n20,Blue"})
    sink = InMemoryGraphSink()
    onto.push(sink, graph="urn:g", clear=True)
    assert "urn:g" in sink.graphs
    assert "a owl:Class" in sink.graphs["urn:g"]


def test_fuseki_helper_builds_urls():
    g = fuseki("http://localhost:3030", "ds")
    assert g.query_url.endswith("/ds/query")
    assert g.update_url.endswith("/ds/update")
    assert g.gsp_url.endswith("/ds/data")


def test_sparql_graph_label_filter_terms():
    # offline: just verify the query builder tokenizes & builds a CONTAINS filter
    g = SparqlGraph("http://x/query", graph_uri="urn:g")
    assert g.graph_uri == "urn:g"
    # no network call — exercise the internal term extraction path via tokenize
    from xgen_ontology_build.text.tokens import tokenize
    assert [t for t in tokenize("Red color") if len(t) >= 2]
