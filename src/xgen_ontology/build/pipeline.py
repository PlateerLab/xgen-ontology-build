"""OntologyBuilder: orchestrate documents/tables into a clean :class:`Ontology`.

The stage order is the production build's, ported without its collection /
database plumbing:

1. input split: table files -> deterministic schema build; text -> extraction
2. extraction by ``mode``:
   * ``"basic"`` (default): :mod:`.deterministic`, **zero LLM calls** -- entities,
     classes and row facts from document structure, Hearst hierarchy from prose
   * ``"enrich"``: basic, then relation formation (:mod:`.relation_formation`): the
     documents are cut into units of evidence and a model names the relation between
     two nodes a unit mentions, held to a vocabulary of relation names; then LLM synonym
     folding of the classes. The production system calls this build "full"
   * ``"llm"``: full LLM extraction of schema and instances (the pre-0.6 path)
3. rule post-build: instance-key merge -> name-fragment folding -> common-word
   pruning -> hierarchy from name structure (classes *and* instances; a head too
   common to classify is judged over the whole collection) -> class synonyms (LLM)
   and vector dedup of classes (if an embedder is given) -> self-typed repair ->
   hierarchy clean -> property inheritance -> graph normalization (the
   store-loading rules: relation names as English identifiers with the document's
   name as the label, weights and sources merged)
4. a quality review recorded on the report

Pass ``progress=callable`` to be told each stage (``progress(stage, detail)``); the
stages are start / retract / tables / extract / enrich / llm / dedup / hierarchy /
finalize / done. Job and session bookkeeping is the application's: drive it from
that callback.

:meth:`OntologyBuilder.extend` runs the same pipeline incrementally over the
chunks an existing ontology has not seen; with ``retract_missing=True`` the chunks
that left the corpus are taken out first (:mod:`.retract`), so a deleted document
is a delta too, not a rebuild. :meth:`OntologyBuilder.retract` does only that and
the post-build. Each stage is independently importable; the orchestrator only
wires them with injected backends (LLM / morphology / embedder / term dictionary),
all optional.
"""
from __future__ import annotations

from ..models import BuildReport, Chunk, Concepts, DataValue, Instance, ObjectProperty, Relation
from .chunk import chunk_document
from .dedup import Deduplicator
from .deterministic import DEFAULT_COMMON_WORD_RANK, MAX_COVERAGE, extract_deterministic
from .dictionary import TermDictionary
from .extract import DocumentExtractor
from .finalize import GRAPH_VOCABULARY, LEGACY_RELATED_PREDICATE, canon_predicate, normalize_graph
from .hierarchy import clean_hierarchy, fix_self_typed_instances, materialize_property_inheritance
from .quality import review_quality
from .relation_formation import (
    EN_RELATION_PROMPTS,
    RelationFormer,
    RelationPrompts,
    canonicalize_predicates,
    labels_by_chunk,
    vocabulary_of,
)
from .resolve import resolve_entities
from .retract import retract_chunks
from .tabular import TABLE_EXTENSIONS, analyze_tables, build_from_tables
from .taxonomy import (
    DEFAULT_RELATED_PREDICATE,
    fold_name_fragments,
    induce_hierarchy,
    prune_common_words,
)

MODES = ("basic", "enrich", "llm")


class OntologyBuilder:
    def __init__(self, llm=None, *, mode: str = "basic", morphology=None, embedder=None,
                 domain: str = "", dedup: bool = True, hierarchy: bool = True,
                 resolve: bool = False, chunk: bool = True, chunk_size: int = 1200,
                 chunk_overlap: int = 150, header_patterns=(), max_coverage: float = MAX_COVERAGE,
                 min_freq: int = 1, common_word_rank: int = DEFAULT_COMMON_WORD_RANK,
                 related_predicate: str | None = DEFAULT_RELATED_PREDICATE,
                 unit_scales: dict | None = None, dictionary: TermDictionary | None = None,
                 progress=None, relation_prompts: RelationPrompts = EN_RELATION_PROMPTS,
                 char_budget: int = 10000, max_output_tokens: int = 0, llm_timeout: float | None = None,
                 max_workers: int = 1, retry_backoff: float = 5.0, should_stop=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.llm = llm
        self.mode = mode
        self.morphology = morphology
        self.embedder = embedder
        self.domain = domain
        self.dedup = dedup
        self.hierarchy = hierarchy
        self.resolve = resolve
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_patterns = tuple(header_patterns or ())
        self.max_coverage = max_coverage
        self.min_freq = min_freq
        self.common_word_rank = common_word_rank
        self.related_predicate = related_predicate
        self.unit_scales = unit_scales
        self.dictionary = dictionary
        self.progress = progress
        # relation formation: the prompts, one call's input budget, the answer cap (0: unknown),
        # a call's timeout, parallel calls, the pause before asking an empty answer again, and a
        # check asked before every call (once true, nothing more is sent)
        self.relation_prompts = relation_prompts
        self.char_budget = char_budget
        self.max_output_tokens = max_output_tokens
        self.llm_timeout = llm_timeout
        self.max_workers = max_workers
        self.retry_backoff = retry_backoff
        self.should_stop = should_stop

    def _relation_former(self) -> RelationFormer:
        return RelationFormer(
            self.llm, prompts=self.relation_prompts, char_budget=self.char_budget,
            max_output_tokens=self.max_output_tokens, timeout=self.llm_timeout, max_workers=self.max_workers,
            retry_backoff=self.retry_backoff, should_stop=self.should_stop, header_patterns=self.header_patterns,
            progress=lambda done, total: self._tick("enrich", done=done, total=total))

    def _tick(self, stage: str, **detail) -> None:
        """Report a pipeline stage to ``progress(stage, detail)``; an application drives its own job state from it."""
        if self.progress is not None:
            try:
                self.progress(stage, detail)
            except Exception:
                pass

    # ── entry points ──

    def build(self, documents: dict[str, list[dict]]):
        """Build from scratch."""
        from ..ontology import Ontology  # local import (Ontology imports build.*)

        return self._run(Ontology(), _normalize_documents(documents, chunk=self.chunk,
                                                          size=self.chunk_size, overlap=self.chunk_overlap))

    def extend(self, ontology, documents: dict[str, list[dict]], *, rebuild: bool = False,
               retract_missing: bool = False):
        """Incremental build: extract only the chunks ``ontology`` has not seen, then re-run the post-build.

        Chunks are recognized by id, so pass the same ids the original build saw
        (raw strings get deterministic ``name#index`` ids). The discriminativeness
        denominator is the whole corpus (old chunks + new). Hierarchy induction is
        restricted to the names this extension introduced and the names they can
        touch, the way the production store does it. In ``enrich`` mode the LLM
        pass covers only chunks whose relations were not asked for yet.
        ``rebuild=True`` treats every chunk as new.

        With ``retract_missing=True``, ``documents`` is the *whole* current corpus:
        chunks the ontology has that are not in it were deleted and are retracted
        before extraction (the store's ``baseline - snapshot`` delta), so the
        post-build runs over the current corpus. Leave it off when you pass only the
        new documents. Returns the same ``ontology``.
        """
        documents = _normalize_documents(documents, chunk=self.chunk, size=self.chunk_size,
                                         overlap=self.chunk_overlap)
        if retract_missing:
            present = {ch["chunk_id"] for chs in documents.values() for ch in chs}
            self._retract(ontology, [c.id for c in ontology.chunks if c.id not in present])
        if rebuild:
            ontology.chunks = [c for c in ontology.chunks
                               if c.id not in {ch["chunk_id"] for chs in documents.values() for ch in chs}]
            ontology.enriched_chunks = [c for c in ontology.enriched_chunks
                                        if c not in {ch["chunk_id"] for chs in documents.values() for ch in chs}]
        return self._run(ontology, documents, incremental=True)

    def retract(self, ontology, chunk_ids):
        """Take deleted chunks out of ``ontology`` and re-run the post-build over what is left.

        Nothing is extracted; the merge, hierarchy and normalization passes run so the
        graph is what a build of the surviving corpus would leave. The counts land on
        ``ontology.report.retracted``. Returns the same ``ontology``.
        """
        self._retract(ontology, chunk_ids)
        return self._run(ontology, {}, incremental=True)

    def _retract(self, onto, chunk_ids) -> dict:
        ids = {str(c) for c in (chunk_ids or []) if c}
        stats = {"chunks": 0}
        if ids:
            self._tick("retract", chunks=len(ids))
            # name-structure links are not chunk evidence; the former spelling counts too
            structural = tuple(x for x in (self.related_predicate, LEGACY_RELATED_PREDICATE, "sameAs") if x)
            stats = retract_chunks(onto.concepts, onto.instances, onto.relations, onto.data_values, ids,
                                   structural_predicates=structural)
            onto.chunks = [c for c in onto.chunks if c.id not in ids]
            onto.enriched_chunks = [c for c in onto.enriched_chunks if c not in ids]
        onto.report.retracted = stats
        return stats

    # ── the pipeline ──

    def _run(self, onto, documents: dict[str, list[dict]], *, incremental: bool = False):
        concepts, instances, relations, data_values = (onto.concepts, onto.instances,
                                                       onto.relations, onto.data_values)
        report = onto.report if incremental else BuildReport()
        mode = self.mode if (self.llm is not None or self.mode == "basic") else "basic"
        if mode != self.mode:
            report.notes.append(f"no LLM given: mode {self.mode!r} ran as 'basic'")
        report.mode = mode

        known = {c.id for c in onto.chunks}
        new_docs = {n: [ch for ch in chs if ch["chunk_id"] not in known] for n, chs in documents.items()}
        new_docs = {n: chs for n, chs in new_docs.items() if chs}
        corpus_total = len(known | {ch["chunk_id"] for chs in documents.values() for ch in chs})
        table_docs = {n: c for n, c in new_docs.items() if _ext(n) in TABLE_EXTENSIONS}
        text_docs = {n: c for n, c in new_docs.items() if _ext(n) not in TABLE_EXTENSIONS}
        before = {c.name for c in concepts.classes} | {i.name for i in instances}
        # Which document each chunk came from (this call's documents plus what earlier builds
        # recorded on the chunks): the spread of a head noun is measured over documents.
        doc_of = {c.id: c.meta["doc"] for c in onto.chunks if c.meta.get("doc")}
        for name, chs in documents.items():
            for ch in chs:
                doc_of[ch["chunk_id"]] = name
        corpus_docs = len(set(doc_of.values()))
        if incremental and not new_docs and mode != "enrich":
            report.notes.append("no new chunks: nothing to extract")

        self._tick("start", mode=mode, incremental=incremental, new_chunks=sum(len(v) for v in new_docs.values()),
                   corpus_chunks=corpus_total)
        protected: set[str] = set()   # table-built names: deterministic identities, never renamed away
        if table_docs:
            self._tick("tables", documents=len(table_docs))
            schema = analyze_tables(table_docs, header_patterns=self.header_patterns)
            c, i, r, dv = build_from_tables(schema, table_docs, header_patterns=self.header_patterns)
            _merge(concepts, c)
            instances += i
            relations += r
            data_values += dv
            protected = {cl.name for cl in c.classes if cl.name} | {x.name for x in i if x.name}

        if text_docs and mode in ("basic", "enrich"):
            self._tick("extract", documents=len(text_docs), chunks=sum(len(v) for v in text_docs.values()))
            c, i, r, dv = extract_deterministic(
                text_docs, min_freq=self.min_freq, max_coverage=self.max_coverage,
                corpus_chunks=corpus_total, corpus_docs=corpus_docs, header_patterns=self.header_patterns,
                common_word_rank=self.common_word_rank, hearst=self.hierarchy)
            _merge(concepts, c)
            instances += i
            relations += r
            data_values += dv
        if mode == "enrich":
            # the enrich baseline is "chunks whose relations were asked for", not "chunks built"
            done = set(onto.enriched_chunks)
            todo = {n: [ch for ch in chs if ch["chunk_id"] not in done] for n, chs in documents.items()
                    if _ext(n) not in TABLE_EXTENSIONS}
            todo = {n: chs for n, chs in todo.items() if chs}
            if todo:
                self._tick("enrich", chunks=sum(len(v) for v in todo.values()))
                former = self._relation_former()
                # The vocabulary is the declared relation names with their definitions; without
                # one, the names already in use (without definitions); without those, the
                # former defines one from the documents.
                vocab = vocabulary_of(concepts) or [{"name": p, "definition": ""}
                                                    for p in _known_predicates(relations)]
                found, rstats = former.extract(todo, labels_by_chunk(concepts, instances),
                                               vocabulary=vocab, domain=self.domain)
                relations += found
                _declare_relations(concepts, rstats.get("new_vocabulary") or [], found)
                report.llm_calls += former.llm_calls
                report.relation_stats = {k: v for k, v in rstats.items() if k != "new_vocabulary"}
                report.relation_stats["new_vocabulary"] = len(rstats.get("new_vocabulary") or [])
                report.relation_stats["lost_chunks"] = len(former.lost_chunk_ids)
                # a chunk no answer came back for is asked again next time
                asked = {ch["chunk_id"] for chs in todo.values() for ch in chs} - former.lost_chunk_ids
                onto.enriched_chunks = sorted(done | asked)
        if text_docs and mode == "llm":
            self._tick("llm", chunks=sum(len(v) for v in text_docs.values()))
            extractor = DocumentExtractor(self.llm, domain=self.domain,
                                          header_patterns=self.header_patterns,
                                          unit_scales=self.unit_scales)
            c, i, r, dv = extractor.extract(text_docs)
            _merge(concepts, c)
            instances += i
            relations += r
            data_values += dv
            report.llm_calls += extractor.llm_calls

        if self.dictionary is not None:
            report.renamed += self.dictionary.apply_to_build(concepts, instances, relations, data_values)
        if self.resolve:
            resolve_entities(instances, relations, data_values)

        deduper = Deduplicator(self.llm if mode == "enrich" else None, self.morphology, self.embedder)
        if self.dedup:
            self._tick("dedup", instances=len(instances), relations=len(relations))
            # Key merge is for extracted names; a table row's label is its identity.
            rename = {o: n for o, n in deduper._normalize_instances(instances).items()
                      if o not in protected} if (text_docs or incremental) else {}
            if rename:
                Deduplicator._apply_instance(rename, instances, relations, data_values)
                report.renamed += len(rename)
            report.folded += _apply_fold(concepts, instances, data_values, relations)
            report.pruned += _apply_prune(instances, relations, data_values, self.common_word_rank)

        if self.hierarchy:
            self._tick("hierarchy", classes=len(concepts.classes), instances=len(instances))
            prose_texts = ([ch.get("chunk_text", "") for chs in text_docs.values() for ch in chs]
                           if mode == "llm" else None)   # basic/enrich ran Hearst during extraction
            new_names = None
            if incremental:
                new_names = ({c.name for c in concepts.classes} | {i.name for i in instances}) - before
            induced = induce_hierarchy(concepts, prose_texts, instances, relations,
                                       related_predicate=self.related_predicate,
                                       header_patterns=self.header_patterns, new_names=new_names,
                                       corpus_chunks=corpus_total, max_coverage=self.max_coverage,
                                       doc_of=doc_of, corpus_docs=corpus_docs, protected=protected)
            edges = induced["hearst_edges_added"] + induced["compound_edges_added"]
            if edges or induced["typed"] or induced["promoted"] or induced["retyped"]:
                report.notes.append(
                    f"name structure: {edges} hierarchy edge(s), {induced['typed']} instance typing(s), "
                    f"{induced['promoted']} head(s) promoted to class, "
                    f"{induced['hearst_classes_added']} class(es) from Hearst patterns, "
                    f"{induced['retyped']} typing(s) under too common a head made neighbour links")
            report.renamed += induced["renamed"]

        if self.dedup:
            rename = deduper.compute_rename_map(concepts) if mode == "enrich" else {}
            report.llm_calls += deduper.llm_calls
            vmap = deduper._vector_dedup([c.name for c in concepts.classes if c.name])
            rename = {**vmap, **rename}          # the LLM map wins on conflict
            rename = {o: n for o, n in rename.items() if o not in protected and o != n}
            if rename:
                Deduplicator._apply_class(rename, concepts, instances)
                Deduplicator._apply_instance(rename, instances, relations, data_values)
                report.renamed += len(rename)

        self._tick("finalize")
        fix_self_typed_instances(instances, concepts)
        clean_hierarchy(concepts)
        materialize_property_inheritance(concepts)
        structural = (self.related_predicate,) if self.related_predicate else ()
        report.normalized = normalize_graph(concepts, instances, relations, data_values,
                                            structural_predicates=structural)
        if mode == "enrich" and relations:
            # Hold the relation names to the vocabulary (the production end-of-build step), then
            # merge the triples that renaming made equal.
            canon_former = self._relation_former()
            canon = canonicalize_predicates(concepts, relations, canon_former)
            report.llm_calls += canon_former.llm_calls
            if canon:
                report.relation_stats["canonicalized"] = canon
                if canon.get("renamed"):
                    normalize_graph(concepts, instances, relations, data_values, structural_predicates=structural)
        clean_hierarchy(concepts)

        onto.chunks = _extend_chunks(onto.chunks, documents, instances)

        report.classes = len(concepts.classes)
        report.object_properties = len(concepts.object_properties)
        report.datatype_properties = len(concepts.datatype_properties)
        report.instances = len({i.name for i in instances if i.name})
        report.relations = len(relations)
        report.data_values = len(data_values)
        report.chunks = len(onto.chunks)
        report.quality = review_quality(concepts, instances, relations, data_values)
        onto.report = report
        self._tick("done", classes=report.classes, instances=report.instances, relations=report.relations,
                   quality=report.quality.get("score"))
        return onto


def _known_predicates(relations: list[Relation], limit: int = 120) -> list[str]:
    """The relation names in use, most used first, without the graph's own vocabulary."""
    counts: dict[str, int] = {}
    for r in relations:
        if r.predicate and r.predicate not in GRAPH_VOCABULARY and r.predicate_type != "DatatypeProperty":
            counts[r.predicate] = counts.get(r.predicate, 0) + 1
    return [p for p, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]


def _declare_relations(concepts: Concepts, vocabulary: list[dict], relations: list[Relation]) -> None:
    """Declare relation names as object properties: the vocabulary defined on the way, with its
    definitions, then every other name the model used. A declared name stays a relation when
    the store loads it (an undeclared one with a value-shaped object would become an attribute)."""
    known = {op.name for op in concepts.object_properties if op.name}
    for v in vocabulary:
        name = v.get("name")
        if name and name not in known:
            concepts.object_properties.append(ObjectProperty(name=name, description=v.get("definition") or ""))
            known.add(name)
    for r in relations:
        name = canon_predicate(r.predicate)[0]
        if name and name not in known and name not in GRAPH_VOCABULARY:
            concepts.object_properties.append(ObjectProperty(name=name))
            known.add(name)


def unbuilt_chunks(ontology, documents: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The chunks of ``documents`` that ``ontology`` has not built yet (by chunk id), per document."""
    docs = _normalize_documents(documents)
    known = {c.id for c in ontology.chunks}
    out = {n: [ch for ch in chs if ch["chunk_id"] not in known] for n, chs in docs.items()}
    return {n: chs for n, chs in out.items() if chs}


def removed_chunks(ontology, documents: dict[str, list[dict]]) -> list[str]:
    """The chunk ids ``ontology`` has built that ``documents`` (the whole current corpus) no longer contain.

    What ``extend(..., retract_missing=True)`` would retract: the deleted documents' chunks."""
    present = {ch["chunk_id"] for chs in _normalize_documents(documents).values() for ch in chs}
    return sorted(c.id for c in ontology.chunks if c.id not in present)


def _linked_names(concepts: Concepts, instances: list[Instance], relations: list[Relation]) -> set[str]:
    """Names that take part in any edge: typing, hierarchy, property declaration or relation."""
    linked: set[str] = set()
    for i in instances:
        if i.name and i.class_name:
            linked.add(i.name)
            linked.add(i.class_name)
    for parent, child in concepts.class_hierarchy:
        linked.update((parent, child))
    for c in concepts.classes:
        if c.parent:
            linked.update((c.name, c.parent))
    for op in concepts.object_properties:
        linked.update(x for x in (op.domain, op.range) if x)
    for dp in concepts.datatype_properties:
        if dp.domain:
            linked.add(dp.domain)
    for r in relations:
        linked.update(x for x in (r.subject, r.object) if x)
    linked.discard("")
    return linked


def _apply_fold(concepts: Concepts, instances: list[Instance], data_values: list[DataValue],
                relations: list[Relation]) -> int:
    """Run :func:`fold_name_fragments` and apply it: drop the fragment, move its chunks."""
    chunks_of: dict[str, set] = {}
    for i in instances:
        if i.name:
            chunks_of.setdefault(i.name, set()).update(c for c in i.source_chunks if c)
    for c in concepts.classes:
        if c.name:
            chunks_of.setdefault(c.name, set()).update(x for x in c.source_chunks if x)
    names = [c.name for c in concepts.classes if c.name] + [i.name for i in instances if i.name]
    fold = fold_name_fragments(names, chunks_of, _linked_names(concepts, instances, relations))
    if not fold:
        return 0
    by_inst = {}
    for i in instances:
        by_inst.setdefault(i.name, i)
    by_cls = {c.name: c for c in concepts.classes if c.name}
    for frag, holder in fold.items():
        moved = sorted(chunks_of.get(frag, set()))
        target = by_inst.get(holder) or by_cls.get(holder)
        if target is not None:
            for cid in moved:
                if cid not in target.source_chunks:
                    target.source_chunks.append(cid)
    instances[:] = [i for i in instances if i.name not in fold]
    concepts.classes = [c for c in concepts.classes if c.name not in fold]
    data_values[:] = [d for d in data_values if d.entity not in fold]
    return len(fold)


def _apply_prune(instances: list[Instance], relations: list[Relation], data_values: list[DataValue],
                 rank_max: int) -> int:
    linked = {i.name for i in instances if i.name and i.class_name}
    for r in relations:
        linked.update(x for x in (r.subject, r.object) if x)
    victims = prune_common_words([i.name for i in instances], linked, rank_max=rank_max)
    if not victims:
        return 0
    instances[:] = [i for i in instances if i.name not in victims]
    data_values[:] = [d for d in data_values if d.entity not in victims]
    return len(victims)


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


def _extend_chunks(existing: list[Chunk], documents: dict[str, list[dict]],
                   instances: list[Instance]) -> list[Chunk]:
    """The chunk list with ``documents`` added, entity mentions recomputed from ``instances``."""
    chunks: dict[str, Chunk] = {c.id: c for c in existing}
    for name, chs in documents.items():
        for ch in chs:
            cid = ch["chunk_id"]
            if cid not in chunks:
                chunks[cid] = Chunk(id=cid, text=ch.get("chunk_text", ""), meta={"doc": name})
    for c in chunks.values():
        c.entities = []
    for inst in instances:
        for cid in inst.source_chunks or []:
            if cid in chunks and inst.name not in chunks[cid].entities:
                chunks[cid].entities.append(inst.name)
    return list(chunks.values())
