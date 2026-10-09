"""xgen-ontology-build — the ontology build, as a library of stages.

Documents or tables go in; a clean knowledge graph with provenance comes out, the same
way the XGEN document service builds its collections. The package is laid out by stage:

* ``text``      — what every stage shares: Korean name shapes, tokens and IRIs, chunking,
                  file parsing, the term dictionary
* ``extract``   — stage 1: documents and tables -> classes, individuals, relations, values.
                  Deterministic by default (no model); relation formation takes a model
* ``postbuild`` — stage 2, over the whole graph: name merging, hierarchy induction, the
                  store-loading rules, retraction, the quality review
* ``store``     — stage 3: the XGEN PostgreSQL tables, any SPARQL 1.1 store, a memory sink
* ``exchange``  — what leaves the build: the knowledge/v1 contract and RDF
* ``pipeline``  — :class:`OntologyBuilder`, the stages in the production build's order;
                  ``facade`` gives the one-call entry points

Search is not here: export a build with ``export_knowledge`` and search it with
xgen-omnifuse.

Quickstart — deterministic table -> ontology, no LLM, no infra (documents build the
same way: ``build_from_documents({"policy.md": text})`` needs no LLM either)::

    from xgen_ontology_build import build_from_csv
    onto = build_from_csv({
        "products": "id,name,color_id\\n1,Widget,10\\n2,Gadget,20",
        "colors":   "color_id,name\\n10,Red\\n20,Blue",
    })
    print(onto.stats())
    onto.to_turtle()                                      # serialize to RDF
"""
from .exchange.emit import to_owl_xml, to_rdf_triples, to_turtle
from .exchange.knowledge_build import (
    assemble_resource_fragments,
    build_knowledge,
    build_resource_fragment,
    export_knowledge,
)
from .extract.deterministic import (
    MAX_COVERAGE,
    RELATED_PREDICATE,
    XGEN_UPLOAD_HEADER_RE,
    extract_as_dicts,
    extract_chunk,
    extract_deterministic,
    is_class_name,
    is_common_word,
    is_value,
)
from .extract.llm_extract import DocumentExtractor, extraction_schema
from .extract.relation_formation import (
    EN_RELATION_PROMPTS,
    KO_RELATION_PROMPTS,
    RelationFormer,
    RelationPrompts,
    canonicalize_predicates,
    labels_by_chunk,
    vocab_name,
    vocabulary_of,
)
from .extract.relation_units import Unit, build_units, find_mentions, relation_label, units_for_documents
from .extract.rowsync import RowDiff, diff_rows, key_text, mapping_signature, row_fingerprint
from .extract.tabular import (
    analyze_tables,
    build_from_rows,
    build_from_tables,
    normalize_fk_relations,
    related_table_source_id,
    row_chunks,
    row_identity_key,
    split_sheets,
    table_cell_rows,
    value_text,
    xsd_type_of,
)
from .facade import (
    build_from_csv,
    build_from_csv_files,
    build_from_db_rows,
    build_from_documents,
    build_from_files,
    build_from_text,
    build_from_triples,
    rows_to_csv,
)
from .llm import CallableLLM, invoke_json_meta
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
)
from .ontology import Ontology
from .pipeline import OntologyBuilder, removed_chunks, unbuilt_chunks
from .postbuild.community import detect_communities, louvain_communities
from .postbuild.dedup import Deduplicator, cluster_by_cosine, shorten_entity_name
from .postbuild.finalize import (
    GRAPH_VOCABULARY,
    RESERVED_PREDICATE_NAMES,
    canon_predicate,
    is_entity_shape,
    normalize_graph,
)
from .postbuild.govern import (
    govern_predicates,
    merge_predicates,
    normalize_predicate,
    strip_argument_noun,
    vote_relation_direction,
)
from .postbuild.hierarchy import (
    clean_hierarchy,
    fix_self_typed_instances,
    materialize_property_inheritance,
)
from .postbuild.quality import review_quality
from .postbuild.resolve import resolve_entities
from .postbuild.retract import retract_chunks
from .postbuild.taxonomy import (
    extract_hearst_pairs,
    fold_name_fragments,
    hearst_hierarchy,
    induce_head_noun_hierarchy,
    induce_hierarchy,
    is_name_child,
    prose_only,
    prune_common_words,
)
from .protocols import LLM, Embedder, GraphSink, GraphStore, Morphology
from .store.memory import InMemoryGraphSink
from .store.postgres import IDENTITY_PREFIX, PgGraph, edge_source_rows, graph_rows
from .store.sparql import SparqlGraph, fuseki
from .text.chunk import chunk_document, chunk_text
from .text.dictionary import Term, TermDictionary
from .text.korean import clean_name, is_name_shape, is_sentence_like, normalize_label, strip_list_markers
from .text.parse import extract_text, html_to_text, load_documents
from .text.tokens import safe_uri, tokenize
from .text.translate import clean_korean_name, translate_names

__version__ = "0.16.0"

__all__ = [
    # one-call entry points and the builder
    "build_from_documents", "build_from_text", "build_from_files", "build_from_csv",
    "build_from_csv_files", "build_from_db_rows", "build_from_triples", "rows_to_csv", "OntologyBuilder",
    "Ontology", "unbuilt_chunks", "removed_chunks",
    # exchange: knowledge/v1 and RDF
    "build_knowledge", "export_knowledge", "build_resource_fragment", "assemble_resource_fragments",
    "to_rdf_triples", "to_turtle", "to_owl_xml",
    # text
    "chunk_text", "chunk_document", "extract_text", "html_to_text", "load_documents",
    "TermDictionary", "Term", "translate_names", "clean_korean_name",
    "clean_name", "is_name_shape", "is_sentence_like", "normalize_label", "strip_list_markers",
    "tokenize", "safe_uri",
    # extract
    "analyze_tables", "build_from_tables", "build_from_rows", "row_chunks", "normalize_fk_relations",
    "row_identity_key", "related_table_source_id", "value_text", "xsd_type_of", "split_sheets", "table_cell_rows",
    "RowDiff", "diff_rows", "row_fingerprint", "mapping_signature", "key_text",
    "extract_deterministic", "extract_as_dicts", "extract_chunk", "is_value", "is_class_name", "is_common_word",
    "MAX_COVERAGE", "RELATED_PREDICATE", "XGEN_UPLOAD_HEADER_RE",
    "DocumentExtractor", "extraction_schema",
    "RelationFormer", "RelationPrompts", "EN_RELATION_PROMPTS", "KO_RELATION_PROMPTS", "canonicalize_predicates",
    "labels_by_chunk", "vocab_name", "vocabulary_of", "Unit", "build_units", "find_mentions", "relation_label",
    "units_for_documents",
    # post-build
    "induce_hierarchy", "hearst_hierarchy", "extract_hearst_pairs", "induce_head_noun_hierarchy", "is_name_child",
    "fold_name_fragments", "prune_common_words", "prose_only",
    "Deduplicator", "cluster_by_cosine", "shorten_entity_name", "resolve_entities",
    "govern_predicates", "normalize_predicate", "strip_argument_noun", "vote_relation_direction", "merge_predicates",
    "clean_hierarchy", "fix_self_typed_instances", "materialize_property_inheritance",
    "normalize_graph", "GRAPH_VOCABULARY", "RESERVED_PREDICATE_NAMES", "canon_predicate", "is_entity_shape",
    "retract_chunks", "review_quality", "detect_communities", "louvain_communities",
    # store
    "PgGraph", "IDENTITY_PREFIX", "graph_rows", "edge_source_rows", "SparqlGraph", "fuseki", "InMemoryGraphSink",
    # llm
    "CallableLLM", "invoke_json_meta",
    # models and protocols
    "Class", "ObjectProperty", "DataProperty", "Concepts", "Instance", "Relation",
    "DataValue", "Node", "Chunk", "RDFTriple", "BuildReport",
    "LLM", "GraphStore", "GraphSink", "Morphology", "Embedder",
    "__version__",
]
