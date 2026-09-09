"""OntologyBuilder — orchestrate documents/tables into a clean :class:`Ontology`.

Stages: input split (table vs text) -> deterministic table build + LLM document
extraction -> merge -> entity resolution -> hierarchy induction (Hearst patterns +
head-noun decomposition, zero LLM calls) -> hierarchy clean -> dedup -> hierarchy
re-clean -> (optional) SCS context profiles. Each stage is independently importable;
the orchestrator just wires them with injected backends (LLM / morphology / embedder),
all optional. The CSV path needs no LLM at all.
"""
from __future__ import annotations

from ..models import BuildReport, Chunk, Concepts, DataValue, Instance, Relation
from .chunk import chunk_document
from .dedup import Deduplicator
from .extract import DocumentExtractor
from .hierarchy import SCSGenerator, clean_hierarchy
from .resolve import resolve_entities
from .tabular import TABLE_EXTENSIONS, analyze_tables, build_from_tables
from .taxonomy import induce_hierarchy


class OntologyBuilder:
    def __init__(self, llm=None, *, morphology=None, embedder=None, domain: str = "",
                 dedup: bool = True, scs: bool = False, hierarchy: bool = True,
                 chunk: bool = True, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.llm = llm
        self.morphology = morphology
        self.embedder = embedder
        self.domain = domain
        self.dedup = dedup
        self.scs = scs
        self.hierarchy = hierarchy
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build(self, documents: dict[str, list[dict]]):
        from ..ontology import Ontology  # local import (Ontology imports build.*)

        documents = _normalize_documents(documents, chunk=self.chunk,
                                         size=self.chunk_size, overlap=self.chunk_overlap)
        table_docs = {n: c for n, c in documents.items() if _ext(n) in TABLE_EXTENSIONS}
        text_docs = {n: c for n, c in documents.items() if _ext(n) not in TABLE_EXTENSIONS}

        concepts = Concepts()
        instances: list[Instance] = []
        relations: list[Relation] = []
        data_values: list[DataValue] = []
        report = BuildReport()

        if table_docs:
            schema = analyze_tables(table_docs)
            c, i, r, dv = build_from_tables(schema, table_docs)
            _merge(concepts, c)
            instances += i
            relations += r
            data_values += dv

        if text_docs and self.llm is not None:
            extractor = DocumentExtractor(self.llm, domain=self.domain)
            c, i, r, dv = extractor.extract(text_docs)
            _merge(concepts, c)
            instances += i
            relations += r
            data_values += dv
            report.llm_calls += extractor.llm_calls

        resolve_entities(instances, relations, data_values)

        if self.hierarchy:
            prose_texts = [ch.get("chunk_text", "") for chs in text_docs.values() for ch in chs]
            induced = induce_hierarchy(concepts, prose_texts, instances)
            total_edges = induced["hearst_edges_added"] + induced["compound_edges_added"]
            if total_edges:
                report.notes.append(
                    f"induced {total_edges} hierarchy edge(s) "
                    f"({induced['hearst_classes_added']} new class(es) from Hearst patterns)"
                )
            report.renamed += induced["renamed"]

        clean_hierarchy(concepts)

        if self.dedup:
            deduper = Deduplicator(self.llm, self.morphology, self.embedder)
            report.renamed = deduper.deduplicate(concepts, instances, relations, data_values)
            report.llm_calls += deduper.llm_calls
            clean_hierarchy(concepts)

        scs_profiles: list[dict] = []
        if self.scs:
            gen = SCSGenerator(self.llm)
            scs_profiles = gen.generate_profiles(concepts)
            report.llm_calls += gen.llm_calls

        chunks = _build_chunks(documents, instances)

        report.classes = len(concepts.classes)
        report.object_properties = len(concepts.object_properties)
        report.datatype_properties = len(concepts.datatype_properties)
        report.instances = len({i.name for i in instances if i.name})
        report.relations = len(relations)
        report.data_values = len(data_values)

        return Ontology(concepts=concepts, instances=instances, relations=relations,
                        data_values=data_values, chunks=chunks, scs_profiles=scs_profiles, report=report)


def _ext(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i >= 0 else ""


def _normalize_documents(documents: dict, *, chunk: bool = False,
                         size: int = 1200, overlap: int = 150) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name, value in documents.items():
        if isinstance(value, str):
            # tables are never chunked (the whole table must stay together); prose is
            if chunk and _ext(name) not in TABLE_EXTENSIONS:
                out[name] = chunk_document(name, value, max_chars=size, overlap=overlap) \
                    or [{"chunk_id": f"{name}#0", "chunk_text": value, "chunk_index": 0}]
            else:
                out[name] = [{"chunk_id": f"{name}#0", "chunk_text": value, "chunk_index": 0}]
        elif isinstance(value, list):
            norm = []
            for j, ch in enumerate(value):
                if isinstance(ch, dict):
                    norm.append({"chunk_id": ch.get("chunk_id") or f"{name}#{j}",
                                 "chunk_text": ch.get("chunk_text") or ch.get("text") or "",
                                 "chunk_index": ch.get("chunk_index", j)})
                else:
                    norm.append({"chunk_id": f"{name}#{j}", "chunk_text": str(ch), "chunk_index": j})
            out[name] = norm
        else:
            out[name] = [{"chunk_id": f"{name}#0", "chunk_text": str(value), "chunk_index": 0}]
    return out


def _merge(into: Concepts, new: Concepts) -> None:
    cn = {c.name for c in into.classes}
    for c in new.classes:
        if c.name and c.name not in cn:
            into.classes.append(c)
            cn.add(c.name)
    opn = {p.name for p in into.object_properties}
    for p in new.object_properties:
        if p.name and p.name not in opn:
            into.object_properties.append(p)
            opn.add(p.name)
    dpn = {p.name for p in into.datatype_properties}
    for p in new.datatype_properties:
        if p.name and p.name not in dpn:
            into.datatype_properties.append(p)
            dpn.add(p.name)
    h = set(into.class_hierarchy)
    for edge in new.class_hierarchy:
        if edge not in h:
            into.class_hierarchy.append(edge)
            h.add(edge)


def _build_chunks(documents: dict[str, list[dict]], instances: list[Instance]) -> list[Chunk]:
    chunks: dict[str, Chunk] = {}
    for _name, chs in documents.items():
        for ch in chs:
            cid = ch["chunk_id"]
            chunks[cid] = Chunk(id=cid, text=ch.get("chunk_text", ""))
    for inst in instances:
        for cid in inst.source_chunks or []:
            if cid in chunks and inst.name not in chunks[cid].entities:
                chunks[cid].entities.append(inst.name)
    return list(chunks.values())
