"""xgen-ontology — backend-agnostic ontology / knowledge-graph toolkit.

Build a provenance-bearing knowledge graph from documents or tables. Zero infra (pure-Python in-memory), zero lock-in (any SPARQL
store), zero hard deps in the core.

Quickstart — deterministic table -> ontology, no LLM, no infra (documents build
the same way: ``build_from_documents({"policy.md": text})`` needs no LLM either)::

    from xgen_ontology import build_from_csv
    onto = build_from_csv({
        "products": "id,name,color_id\\n1,Widget,10\\n2,Gadget,20",
        "colors":   "color_id,name\\n10,Red\\n20,Blue",
    })
    print(onto.stats())
    onto.to_turtle()                                      # serialize to RDF
"""
from .backends.memory import InMemoryGraph, InMemoryGraphSink, InMemoryVector
from .backends.postgres import PgGraph, graph_rows
from .backends.sparql import SparqlGraph, fuseki
from .build.chunk import chunk_document, chunk_text
from .build.community import detect_communities, louvain_communities
from .build.dedup import Deduplicator, cluster_by_cosine, shorten_entity_name
from .build.deterministic import (
                                  extract_as_dicts,
                                  extract_chunk,
                                  extract_deterministic,
                                  is_class_name,
                                  is_common_word,
                                  is_value,
)
from .build.dictionary import Term, TermDictionary
from .build.emit import to_owl_xml, to_rdf_triples, to_turtle
from .build.extract import DocumentExtractor, extraction_schema
from .build.finalize import normalize_graph
from .build.govern import (
                                  govern_predicates,
                                  merge_predicates,
                                  normalize_predicate,
                                  strip_argument_noun,
                                  vote_relation_direction,
)
from .build.hierarchy import (
    clean_hierarchy,
    fix_self_typed_instances,
    materialize_property_inheritance,
)
from .build.parse import extract_text, html_to_text, load_documents
from .build.pipeline import OntologyBuilder, unbuilt_chunks
from .build.quality import review_quality
from .build.resolve import resolve_entities
from .build.tabular import analyze_tables, build_from_tables
from .build.taxonomy import (
                                  extract_hearst_pairs,
                                  fold_name_fragments,
                                  hearst_hierarchy,
                                  induce_head_noun_hierarchy,
                                  induce_hierarchy,
                                  prose_only,
                                  prune_common_words,
)
from .build.translate import clean_korean_name, translate_names
from .facade import (
                                  build_from_csv,
                                  build_from_csv_files,
                                  build_from_documents,
                                  build_from_files,
                                  build_from_text,
                                  build_from_triples,
                                  rows_to_csv,
)
from .knowledge_build import build_knowledge, export_knowledge
from .korean import clean_name, is_sentence_like, normalize_label, strip_list_markers
from .llm import CallableLLM, EchoLLM
from .models import (
                                  BuildReport,
                                  Chunk,
                                  Class,
                                  Concepts,
                                  DataProperty,
                                  DataValue,
                                  Instance,
                                  Node,
                                  ObjectProperty,
                                  RDFTriple,
                                  Relation,
                                  SearchResult,
)
from .ontology import Ontology
from .protocols import LLM, Embedder, GraphSink, GraphStore, Morphology, VectorStore
from .text import BM25, safe_uri, tokenize

__version__ = "0.9.0"

__all__ = [
    # portable knowledge exchange
    "build_knowledge", "export_knowledge",
    # facade
    "build_from_documents", "build_from_text", "build_from_files", "build_from_csv",
    "build_from_csv_files", "build_from_triples", "rows_to_csv", "OntologyBuilder", "Ontology",
    "unbuilt_chunks",
    # search
    "GraphRAG",
    # ingest
    "chunk_text", "chunk_document", "extract_text", "html_to_text", "load_documents",
    # build stages
    "analyze_tables", "build_from_tables", "extract_deterministic", "extract_as_dicts",
    "extract_chunk", "is_value", "is_class_name", "is_common_word", "DocumentExtractor",
    "extraction_schema", "resolve_entities", "Deduplicator", "cluster_by_cosine",
    "shorten_entity_name", "govern_predicates", "normalize_predicate", "strip_argument_noun",
    "vote_relation_direction", "merge_predicates", "clean_hierarchy", "fix_self_typed_instances",
    "materialize_property_inheritance", "review_quality", "detect_communities",
    "louvain_communities", "normalize_graph", "translate_names", "clean_korean_name",
    "TermDictionary", "Term", "to_rdf_triples", "to_turtle", "to_owl_xml",
    # hierarchy induction (Hearst patterns + name structure, zero LLM calls)
    "induce_hierarchy", "hearst_hierarchy", "extract_hearst_pairs",
    "induce_head_noun_hierarchy", "fold_name_fragments", "prune_common_words", "prose_only",
    # Korean text utilities (degrade gracefully with no morphological analyzer)
    "clean_name", "is_sentence_like", "normalize_label", "strip_list_markers",
    # backends
    "InMemoryGraph", "InMemoryVector", "InMemoryGraphSink", "SparqlGraph", "fuseki", "PgGraph", "graph_rows",
    # llm
    "EchoLLM", "CallableLLM",
    # models
    "Class", "ObjectProperty", "DataProperty", "Concepts", "Instance", "Relation",
    "DataValue", "Node", "Chunk", "RDFTriple", "SearchResult", "BuildReport",
    # protocols
    "LLM", "GraphStore", "VectorStore", "GraphSink", "Morphology", "Embedder",
    # text
    "BM25", "tokenize", "safe_uri",
    "__version__",
]


def __getattr__(name):
    # Preserve old imports while keeping the independent build path free of the
    # retired search engine. New retrieval integrations use xgen-omnifuse.
    if name == "GraphRAG":
        import warnings

        from .search.oneshot import GraphRAG
        warnings.warn("xgen_ontology.GraphRAG is legacy; use xgen-omnifuse for retrieval",
                      DeprecationWarning, stacklevel=2)
        return GraphRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
