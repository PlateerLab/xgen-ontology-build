# 0.5.0 (2026-09-09)

**Hierarchy induction: is-a edges from text, zero LLM calls.** `build_from_documents` /
`build_from_csv` gain a `hierarchy: bool = True` flag (`OntologyBuilder(hierarchy=...)`),
ported from the XGEN production ontology build path and verified byte-for-byte against
it on a real 87-chunk corpus before merging.

- `build.taxonomy.hearst_hierarchy` — Korean Hearst-pattern extraction ("X, Y 등의 Z" ->
  Z is-a X, Z is-a Y): noun-run matching around the "등" (NNB) anchor, `min_hyponyms` /
  `max_coverage` filtering so a hypernym only counts once it generalizes >=2 distinct
  things without swallowing the corpus. Mints new `Class` objects for hypernyms that
  weren't already modeled.
- `build.taxonomy.prose_only` — strips HTML `<table>` blocks and pipe-delimited grid
  lines before Hearst runs on them. Table rows read as false Hearst hits ("감사패, 상패
  등의 제작비" -> "제작비" is-a "감사패" is-a "상패", a cost line item misread as a
  category); on the verification corpus this cut raw pairs 133 -> 27, all genuine.
- `build.taxonomy.induce_head_noun_hierarchy` — decomposes compound class names by their
  Korean/English head noun ("상임감사실" -> "감사실"), and folds code-prefixed spelling
  variants ("NA162000.23 제주목장사업" -> "제주목장사업") into a rename instead of a
  hierarchy edge. Guards against splitting spaced multi-word names ("한국 마사회" stays
  intact) and against firing on a leading modifier ("상품목록" is not made a child of
  "상품" — only suffix matches count as evidence).
- `korean.py` — `clean_name` / `is_sentence_like` / `normalize_label` / `strip_list_markers`,
  ported from the production name-cleanup pass; degrades gracefully to word-boundary-only
  matching with no crash when `kiwipiepy` isn't installed (the `korean` extra).

Set `hierarchy=False` to keep the old flat, headers/LLM-only schema.

# 0.4.0 (2026-09-09)

**License changed to source-available, all rights reserved (Jinsoo Kim).** Repository
renamed `jinsoo96/xgen-ontology` → `jinsoo96/js-ontology-build` (GitHub keeps the old
name as a redirect); the published PyPI package name stays `xgen-ontology`. A new
organization mirror, `PlateerLab/xgen-ontology-build`, replaces the previous
`PlateerLab/xgen-ontology-build` placeholder and fast-forward-syncs from
`js-ontology-build:main` every 15 minutes, matching the `js-omnifuse` /
`PlateerLab/xgen-omnifuse` setup.

Releases 0.1.0-0.3.0 remain under the MIT License for that exact code — this is not
retroactive. From this release on: reading, personal evaluation and citation are
allowed; use in a product or service, copying, modification, redistribution,
derivative works and training ML models on the code require the Owner's prior written
permission. `pyproject.toml` now declares `license = { file = "LICENSE" }` and the
`License :: Other/Proprietary License` classifier so the terms surface on PyPI and in
the built wheel/sdist. See `LICENSE` for the full text and the NOTICE explaining the
version cutover.

# 0.3.0 (2026-08-26)

Model-agnostic extraction hardening, back-ported from the XGEN production build
path after real-corpus A/B runs (cloud vs local models on identical input).

- `parse_json_lenient` / `salvage_truncated` (llm.py): accept fenced / commented /
  trailing-comma / smart-quote / top-array replies, and close replies cut off by
  the output cap so complete elements survive instead of the whole batch dying.
- Top-level key aliases (`KEY_ALIASES`): singular/plural and non-English key
  spellings no longer silently produce 0 items.
- Self-typed repair: entities typed as themselves (proper noun promoted to class)
  are re-typed by the model, ratio-gated so well-behaved replies cost no extra call.
- `verify_numeric_units` + `KO_UNIT_SCALES`: source-grounded magnitude check for
  unit notations ("15억" written as 150000000); pluggable per language, off by default.
- Split-on-failure: a batch whose reply cannot be parsed is halved and retried
  (depth-capped) instead of being silently dropped.

# Changelog

## 0.2.0

- **Ingestion** so it works end-to-end from raw documents:
  - `parse` — `extract_text` / `load_documents`: txt/md/rst/json/html (zero-dep), csv/tsv kept
    as raw table text; pdf/docx/xlsx via the new `[files]` extra.
  - `chunk` — `chunk_text` / `chunk_document`: boundary-aware (paragraph→sentence→char) windows
    with overlap and stable chunk ids for provenance/search.
  - `build_from_files(paths)` and `build_from_text(text)`; raw prose is auto-chunked in the
    pipeline (tables are never chunked). `OntologyBuilder(chunk=, chunk_size=, chunk_overlap=)`.
- **License: MIT © jinsoo96** (was unset).

## 0.1.0

Initial extraction of the production ontology build + search logic as a
backend-agnostic library.

**Build** (documents/tables → a clean knowledge graph):
- `build_from_csv` / `build_from_csv_files` — deterministic table → ontology, no LLM:
  table→Class, FK→ObjectProperty (same-name / normalized-name / value-overlap), column→DataProperty,
  dimension rows→instances, large fact/junction tables kept schema-only.
- `build_from_documents` — LLM extraction (schema + instances per chunk batch, source-tagged),
  with a junk filter; mixes table + text inputs.
- Cleaning stages, each independently importable: `resolve_entities` (entity resolution),
  `govern_predicates` / `normalize_predicate` (predicate governance), `Deduplicator` +
  `cluster_by_cosine` (rule + LLM + embedding dedup), `clean_hierarchy` (genuine is-a only,
  cycle-breaking), `SCSGenerator` (property inheritance + context profiles).
- `review_quality` — completeness / integrity / grounding / shape score (in-memory, no SPARQL).
- `louvain_communities` / `detect_communities` — pure-Python Louvain clustering.
- `to_turtle` (zero-dep) and `to_owl_xml` (optional rdflib) emit.

**Search** (one-shot GraphRAG):
- `Ontology.search` / `GraphRAG` — fuse vector/lexical + graph label-linking + class
  enumeration + HippoRAG 1-hop with MMR diversity and adaptive top-k; single synthesis;
  honest `evidence_nodes`. Language-neutral default prompt (overridable).

**Backends**:
- Zero-infra `InMemoryGraph` / `InMemoryVector` / `InMemoryGraphSink` (BM25, CJK n-grams).
- `SparqlGraph` — read + write any SPARQL 1.1 store (Fuseki/GraphDB/Blazegraph/Virtuoso),
  stdlib-only; `fuseki(base, dataset)` convenience.

`dependencies = []` core; rdflib/kiwipiepy/qdrant-client are optional extras behind protocols.
`Ontology.from_triples` for the search-only path. pytest suite; build + search examples.
