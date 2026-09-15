# 0.7.0 (2026-09-15)

**Everything that was still only in the product is now in the library.** Incremental
builds, the store-loading rules, IRI translation, the term dictionary and community
membership, each without its database plumbing.

- **Incremental build** -- `OntologyBuilder.extend(onto, documents, rebuild=False)`
  extracts only the chunks the ontology has not seen (by chunk id), uses the whole
  corpus as the discriminativeness denominator, re-runs the post-build and restricts
  hierarchy induction to the new names and the existing names they can touch (the
  production `only_uris` delta, `_touched_by_new`). In `enrich` mode the LLM relation
  pass covers only chunks not yet asked about (`Ontology.enriched_chunks`, the
  production enrich baseline). `unbuilt_chunks(onto, documents)` shows what an
  extension would do -- the check the production auto-backfill runs per collection.
  `BuildReport.chunks` counts what the ontology has seen.
- **`build.finalize.normalize_graph`** (new, runs at the end of every build) -- the
  production store-loading rules (`serialize_graph`): class-shaped names only, with
  referenced classes declared; relations named with graph vocabulary (`type`,
  `instanceOf`, `subClassOf`, `sameAs`) become typing statements, hierarchy edges or
  folds; a relation whose predicate is a declared datatype property or whose object
  is a value becomes an attribute; a value is never a subject; a name that occurs
  only as an endpoint or an attribute's subject becomes an individual (punning, as
  the store does); duplicate records merge. Verified against the production function
  on a 16,400-node build: identical node, edge and attribute sets.
  `BuildReport.normalized` carries the counts.
- **`build.translate`** (new) -- `translate_names` (LLM, batches of 50, cached),
  `english_local_name` (classes UpperCamelCase, properties lowerCamelCase),
  `clean_korean_name`; `Ontology.translate(llm)` fills `Ontology.translations`, which
  the emitters already consumed; the emitter now applies the case rule. The
  production OWL generator's identifier logic, without rdflib.
- **`build.dictionary`** (new) -- `TermDictionary` / `Term`: upsert by normalized
  alias, `bulk_import`, activation, element linking; `normalize_query` (expand /
  replace, Hangul-aware word boundaries), `normalize_text` (idempotent
  `alias(canonical)` for indexing), `apply_to_build` (canonicalize the build models).
  `OntologyBuilder(dictionary=...)` applies it in the pipeline. The production
  dictionary service and its query / indexing normalizers, minus the tables.
- `Ontology.community_of()` -- instance -> Louvain community id (the production
  community tag), next to the existing `communities()` summary.

Not ported, by design: PG dual-write and the `nkey` column (storage of the key the
library computes on the fly), job/session bookkeeping, OWL serialization via rdflib
(the library emits Turtle / OWL itself).

# 0.6.0 (2026-09-15)

**The document build no longer needs an LLM.** The whole production build path of
XGEN as of this date is ported: structure-based extraction, the rule post-build, and
the LLM reserved for an explicit enrich pass. Every ported function was checked
against the production code on a real 14,924-chunk corpus (same input, byte-identical
output; see the parity notes below). Collection / database plumbing was left behind;
everything runs on the in-memory build models.

- **`build.deterministic`** (new) -- `extract_deterministic` / `extract_as_dicts`:
  zero-LLM extraction from document structure. HTML tables (rowspan/colspan expanded),
  whitespace row dumps and pipe grids become row entities typed by the subject
  column's header (the entity column with the most distinct values), value cells
  become attributes, entity cells relations; a caption or the header prefixes rows
  keyed by a value; headers carry across chunk boundaries of one document; a unit
  row annotates its columns; codes, glosses and decorative parentheses are stripped
  from names (a parenthetical that tells names apart is kept); the words before a
  table are read as prose. Prose yields noun-phrase entities (morpheme-aware, with
  affix and line-break rules) and (entity, attribute, value) facts. A shared head
  noun becomes a class; a single unlinked everyday word is dropped; names in more
  than `max_coverage` of the corpus are dropped as non-discriminative once the corpus
  is large enough to judge. Hearst hierarchy runs inside it, on prose only.
- **`OntologyBuilder(mode=...)`** -- `"basic"` (default, zero LLM calls), `"enrich"`
  (basic + LLM relations between the extracted entities + LLM schema synonym
  folding), `"llm"` (full LLM extraction, the pre-0.6 path). Without an `llm`,
  `"enrich"`/`"llm"` run as `"basic"` and say so in `report.notes`. The stage order
  is the production one: extraction -> instance-key merge -> predicate merge ->
  fragment folding -> common-word pruning -> hierarchy from names -> vector/LLM dedup
  -> self-typed repair -> hierarchy clean -> property inheritance -> quality review.
  `BuildReport` gains `mode`, `folded`, `pruned`, `quality`.
- **`build.taxonomy`** -- `induce_head_noun_hierarchy` now runs over class *and*
  instance names and returns `(edges, renames, related)`: a name used as a head is
  promoted to a class, an instance is typed by its head (an extra `rdf:type` when it
  already has one), a code-prefixed spelling folds into its canonical name, a shared
  leading word becomes a `related_predicate` relation (default `"관련"`, `None`
  disables). New `fold_name_fragments` (a short name that never stood alone in the
  source folds back into the longer name that covers all its chunks) and
  `prune_common_words` (unlinked everyday words, judged by analyzer dictionary rank,
  no word list). `prose_only` now also strips whitespace row dumps and ingestion
  markers; `hearst_hierarchy` / `induce_hierarchy` take `header_patterns`.
- **`build.govern`** -- `strip_argument_noun` (a subject/object noun glued into the
  predicate), `seed_predicates` and a returned `canonical_map` in `govern_predicates`,
  `vote_relation_direction` (flip the minority direction of a predicate by type-pair
  majority, then by entity role), `merge_predicates` (same stem, or one predicate's
  (subject, object) extension contained in another's; duplicate triples removed).
- **`build.hierarchy`** -- `fix_self_typed_instances` (an instance typed by a class of
  its own name moves to that class's parent or loses the typing),
  `materialize_property_inheritance` (a parent's declared properties are declared on
  its subclasses).
- **`build.dedup`** -- the instance key ignores leading quote characters and, with
  no injected morphology, uses the bundled Korean analyzer's content morphemes; the
  canonical spelling is the shortest one. New `compute_rename_map` (schema synonym
  map only) and `shorten_entity_name` (a sentence fragment returned as an entity
  name is cut to its leading noun run).
- **`build.extract`** -- `DocumentExtractor.extract_relations` (the enrich pass: the
  known entities of each batch and the predicates already in use are offered; chunk
  ids are aliased and unknown ids never kept), `include_relations`, an existing-class
  context, row-aware splitting of oversized chunks, and rule post-processing: a data
  value whose value is an entity becomes a relation and the reverse, sentence-fragment
  entity names are shortened, direction is voted, predicates governed and the same
  vocabulary applied to declarations and data values. `verify_numeric_units` only
  corrects a power-of-ten error and leaves a value that also occurs bare in the source;
  `KO_UNIT_SCALES` covers the compound units. `extraction_schema` (JSON schema for
  structured output) is public; `invoke_json` uses an LLM's `generate_json(prompt,
  system, schema)` when it has one.
- **`build.tabular`** -- the name column is judged from the data (mostly text, mostly
  distinct, not identifier codes; a code column only as fallback), a table with no
  name column stays schema-only, rows whose labels would collide as IRIs are told
  apart by their key, FK properties are named per column, and each row keeps the id of
  the chunk it came from.
- `resolve_entities` (fuzzy surface-form merging) is **off by default**
  (`resolve=True` to keep it); the production build does not fuzzy-merge.
- `korean.strip_list_markers` accepts both Hangul ieung glyphs as bullets.

Parity notes: the deterministic extractor was run side by side with the production
module on all 14,924 chunks of a customer corpus -- chunk facts, Hearst pairs,
`prose_only` and the final four-part result were identical. The name-structure,
fragment-folding, common-word and predicate-merge passes were run against the
production store functions over a fake store fed the same 16K-node graph: identical
edge sets (4,146 subclass, 5,285 related), fold and merge sets. Where the production
code is order-dependent on ties (which longer name a fragment folds into), the port
picks the spelling-first candidate deterministically.

Also in this release: **repository model reversed.** `PlateerLab/xgen-ontology-build` is
now the origin and publishes the package; `jinsoo96/js-ontology-build` is a read-only
mirror that fast-forwards from it with its own `GITHUB_TOKEN` (no personal credential,
nothing to expire). LICENSE gains §4, a written grant letting PlateerLab organization
members use, modify, build and ship the Software as part of Plateer products. Not open
source; copyright unchanged; the 0.1.0-0.3.0 MIT carve-out unchanged.

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
