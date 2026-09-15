"""The :class:`Ontology` — the hub that ties build output to search, emit and review.

A built ontology (schema + instances + relations + data values, plus optional source
chunks) that you can:

* ``search(question)`` — one-shot GraphRAG over an in-memory index (or any backend);
* ``to_turtle()`` / ``to_owl()`` — serialize to RDF;
* ``push(sink)`` — load into any SPARQL store;
* ``quality()`` / ``communities()`` — review + cluster.

``Ontology.from_triples(...)`` builds a searchable ontology straight from loose
``(s, p, o)`` triples (no LLM, no build) for the search-only use case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .backends.memory import InMemoryGraph, InMemoryVector
from .build import emit as _emit
from .build.community import detect_communities
from .build.quality import review_quality
from .models import BuildReport, Chunk, Concepts, DataValue, Instance, Node, RDFTriple, Relation, SearchResult

_ISA_PREDICATES = {"instanceof", "instance_of", "type", "rdf:type", "a", "subclassof", "subclass_of", "is-a", "isa"}


@dataclass
class Ontology:
    concepts: Concepts = field(default_factory=Concepts)
    instances: list[Instance] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    data_values: list[DataValue] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    enriched_chunks: list[str] = field(default_factory=list)   # chunk ids the LLM relation pass covered
    translations: dict[str, str] = field(default_factory=dict)
    report: BuildReport = field(default_factory=BuildReport)

    # ── search ──

    def nodes(self) -> list[Node]:
        out: list[Node] = []
        seen: set[str] = set()
        for c in self.concepts.classes:
            if c.name and c.name not in seen:
                out.append(Node(c.name, c.name, "class"))
                seen.add(c.name)
        for parent, child in self.concepts.class_hierarchy:
            for name in (parent, child):
                if name and name not in seen:
                    out.append(Node(name, name, "class"))
                    seen.add(name)
        for inst in self.instances:
            if inst.name and inst.name not in seen:
                out.append(Node(inst.name, inst.name, "instance"))
                seen.add(inst.name)
        for r in self.relations:
            for name in (r.subject, r.object):
                if name and name not in seen:
                    out.append(Node(name, name, "instance"))
                    seen.add(name)
        return out

    def edges(self) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for inst in self.instances:
            if inst.name and inst.class_name:
                triples.append((inst.name, "instanceOf", inst.class_name))
        for parent, child in self.concepts.class_hierarchy:
            if parent and child:
                triples.append((child, "subClassOf", parent))
        for r in self.relations:
            if r.subject and r.predicate and r.object and r.predicate_type != "DatatypeProperty":
                triples.append((r.subject, r.predicate, r.object))
        return triples

    def graph(self) -> InMemoryGraph:
        return InMemoryGraph(self.nodes(), self.edges())

    def vector(self, embedder=None) -> InMemoryVector:
        return InMemoryVector(self.chunks, embedder=embedder)

    def search(self, question: str, *, llm=None, embedder=None, **kwargs) -> SearchResult:
        import warnings

        from .search.oneshot import GraphRAG
        warnings.warn("Ontology.search is legacy; export knowledge to xgen-omnifuse",
                      DeprecationWarning, stacklevel=2)
        engine = GraphRAG(self.graph(), self.vector(embedder=embedder), llm, **kwargs)
        return engine.search(question)

    # ── identifiers ──

    def translate(self, llm, *, batch_size: int = 50) -> dict[str, str]:
        """Give every class and property an English IRI local name via ``llm`` (cached in ``translations``).

        Only names not yet translated are sent, so calling it after an
        :meth:`~xgen_ontology.OntologyBuilder.extend` costs one batch per 50 new names.
        """
        from .build.translate import collect_terms, translate_names

        self.translations = translate_names(collect_terms(self.concepts), llm,
                                            cache=self.translations, batch_size=batch_size)
        return self.translations

    # ── emit ──

    def to_rdf_triples(self) -> list[RDFTriple]:
        return _emit.to_rdf_triples(self.concepts, self.instances, self.relations,
                                    self.data_values, translations=self.translations)

    def to_turtle(self) -> str:
        return _emit.to_turtle(self.to_rdf_triples())

    def to_owl(self) -> str:
        return _emit.to_owl_xml(self.to_rdf_triples())

    def push(self, sink, *, graph: str | None = None, clear: bool = True) -> None:
        sink.upload_turtle(self.to_turtle(), graph=graph, clear=clear)

    # ── review ──

    def quality(self) -> dict:
        return review_quality(self.concepts, self.instances, self.relations, self.data_values)

    def communities(self) -> list[dict]:
        return detect_communities(self.instances, self.relations)

    def community_of(self) -> dict[str, int]:
        """Instance name -> Louvain community id over the instance-relation graph (largest community is 0)."""
        from .build.community import louvain_communities

        names = {i.name for i in self.instances if i.name}
        edges = [(r.subject, r.object) for r in self.relations
                 if r.predicate_type != "DatatypeProperty" and r.subject in names
                 and r.object in names and r.subject != r.object]
        nodes = sorted({n for e in edges for n in e})
        return louvain_communities(nodes, edges) if edges else {}

    def stats(self) -> dict:
        return {
            "classes": len(self.concepts.classes),
            "object_properties": len(self.concepts.object_properties),
            "datatype_properties": len(self.concepts.datatype_properties),
            "instances": len({i.name for i in self.instances if i.name}),
            "relations": len(self.relations),
            "data_values": len(self.data_values),
            "chunks": len(self.chunks),
        }

    # ── search-only convenience ──

    @classmethod
    def from_triples(cls, triples, chunks=None) -> Ontology:
        """Build a searchable ontology from loose ``(s, p, o)`` triples (no LLM).

        is-a edges (``instanceOf`` / ``type`` / ``subClassOf`` / ...) seed classes
        and class membership; every other edge becomes a relation."""
        concepts = Concepts()
        instances: list[Instance] = []
        relations: list[Relation] = []
        class_names: set[str] = set()
        inst_seen: set[str] = set()

        norm = []
        for t in triples:
            if isinstance(t, dict):
                s, p, o = t.get("s") or t.get("subject"), t.get("p") or t.get("predicate"), t.get("o") or t.get("object")
            else:
                s, p, o = t
            if s and p and o:
                norm.append((str(s), str(p), str(o)))

        for s, p, o in norm:
            if p.lower().replace(" ", "") in _ISA_PREDICATES:
                class_names.add(o)
                if s not in inst_seen:
                    instances.append(Instance(name=s, class_name=o))
                    inst_seen.add(s)
            else:
                relations.append(Relation(subject=s, predicate=p, object=o))

        concepts.classes = [_cls(n) for n in sorted(class_names)]
        ont = cls(concepts=concepts, instances=instances, relations=relations)
        if chunks:
            ont.chunks = _coerce_chunks(chunks)
        return ont


def _cls(name: str):
    from .models import Class
    return Class(name=name)


def _coerce_chunks(chunks) -> list[Chunk]:
    out: list[Chunk] = []
    for i, c in enumerate(chunks):
        if isinstance(c, Chunk):
            out.append(c)
        elif isinstance(c, dict):
            out.append(Chunk(id=c.get("id") or c.get("chunk_id") or f"c{i}",
                             text=c.get("text") or c.get("chunk_text") or "",
                             entities=c.get("entities", [])))
        elif isinstance(c, (tuple, list)):
            out.append(Chunk(id=str(c[0]), text=str(c[1]),
                             entities=list(c[2]) if len(c) > 2 else []))
        else:
            out.append(Chunk(id=f"c{i}", text=str(c)))
    return out
