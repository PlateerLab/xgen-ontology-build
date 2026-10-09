# 0.16.0 (2026-10-09)

Database rows kept in step with their table. The production build (xgen-documents · xgen-workflow,
2026-10-09) replaced the timestamp cursor it loaded tables with by a full read compared against
row fingerprints, and fixed what the loader did with rows along the way; this release carries
that into the library.

## Rows are individuals by declaration

- `Instance.identity`: a database row's identity, the production loader's key
  (`row_identity_key(table_source_id, key_column, key_value)` → `dbrow_…`). `build_from_rows`
  sets it for every keyed row and makes a placeholder individual, carrying the target row's
  identity, for a foreign key that points at a row the load does not hold
  (`related_table_source_id`). The post-build reconciles identities: a row loaded again or a
  placeholder its row arrived for is the same individual, the row's name wins, and the
  relations other rows hold to it follow (`extend_rows` renames before it retracts, so a renamed
  row keeps its foreign keys).
- `normalize_graph` keeps a row's relations and values whatever the shape of its label: a row
  named by a numeric key (a collateral id, a card number) no longer loses its foreign keys and
  attributes at the store-loading step. The production serializer had the same defect, hidden
  until now by merge-on-write.
- `PgGraph` writes a row under its identity (`IDENTITY_PREFIX`), the URI the product uses, and
  `load()` gives the rows their identity back. On an appending write a row's attributes and
  label are replaced, not merged: a value that became NULL leaves the node.

## Synchronization

- `extract/rowsync`: `mapping_signature`, `row_fingerprint`, `diff_rows(rows, columns, pk, known=,
  signature=, complete=)` → `RowDiff` (added · changed · unchanged · missing · fingerprints ·
  `summary()`). A full read in key order against the fingerprints of the last load tells new,
  changed and unchanged rows and, when the read was complete, the rows that left the table. No
  timestamp column: a value changed without its stamp, a stamp set back and a deletion are all
  seen. The fingerprints are the application's to keep (the product keeps them in
  `db_ontology_row_states`).
- `OntologyBuilder.sync_rows(ontology, table, columns, rows, source_id=, fingerprints=, complete=,
  **schema)`: the diff applied to the build models (`extend_rows` for new and changed rows,
  `retract` for the missing ones). Returns the `RowDiff`.
- `PgGraph.detach_chunks(chunk_ids, attr_keys=)`: the traces of rows about to be written again
  leave the tables first (the relations those chunks stated, gone when no other chunk states
  them; the attribute keys on nodes that are not rows). `PgGraph.remove_rows(chunk_ids)`: rows
  that left the table (`detach_chunks` then `prune_chunks`).

The old import name `xgen_ontology` (the `compat` package) follows to 0.16.0, its last release.

# 0.15.0 (2026-10-08)

The repository has been named for the build since 0.4 (`xgen-ontology-build`) while the
package and the distribution kept the toolkit name (`xgen_ontology`, `xgen-ontology`) and
the toolkit shape: build functions in one flat folder next to a search half that
xgen-omnifuse had already replaced. This release makes the package read as what it is.

## Renamed and laid out by stage

- The import name is **`xgen_ontology_build`** and the distribution **`xgen-ontology-build`**.
  The package is laid out by build stage, in the production build's order:
  `text/` (what every stage shares: Korean name shapes, tokens and IRIs, chunking, file
  parsing, the term dictionary, IRI translation), `extract/` (stage 1: deterministic
  extraction, tables and database rows, relation units and relation formation, full LLM
  extraction as `llm_extract`), `postbuild/` (stage 2: taxonomy, dedup, hierarchy, the
  store-loading rules `finalize`, governance, resolution, quality, communities, retract),
  `store/` (stage 3: PostgreSQL, SPARQL, the in-memory sink), `exchange/` (knowledge/v1,
  RDF emit), with `pipeline.py` (`OntologyBuilder`) and `facade.py` at the top. Every
  public name of 0.14 is still exported from the package root; module paths moved:
  `xgen_ontology.build.deterministic` is `xgen_ontology_build.extract.deterministic`,
  `xgen_ontology.backends.postgres` is `xgen_ontology_build.store.postgres`,
  `xgen_ontology.knowledge` is `xgen_ontology_build.exchange.knowledge`, `xgen_ontology.korean`
  is `xgen_ontology_build.text.korean`, and so on (the full map is in the compatibility
  package). Intra-package imports are absolute.
- **The search half is gone**: `GraphRAG`, `Ontology.search` / `graph` / `vector`,
  `InMemoryGraph`, `InMemoryVector`, `BM25`, `SearchResult`, `VectorStore`, `EchoLLM`. A build is
  searched by exporting it (`export_knowledge`) to xgen-omnifuse, which owns search.
  `PgGraph` keeps its read methods (the `GraphStore` protocol), `InMemoryGraphSink` stays
  for tests and dry runs.
- The knowledge/v1 contract is unchanged (byte-identical `exchange/knowledge.py`, same
  lock), and bundles keep their extension key `xgen_ontology.build`, so consumers of
  the contract need no change.

## Compatibility

- `compat/` publishes **`xgen-ontology` 0.15.0** as a shim: it depends on
  `xgen-ontology-build==0.15.0` and re-exports it under the old name, aliasing the old
  module paths (`xgen_ontology.knowledge`, `xgen_ontology.build.*`, `xgen_ontology.backends.*`,
  `xgen_ontology.korean` ...) with a `DeprecationWarning`. It is published with 0.15 and
  0.16 and then stops. `pip install xgen-ontology` keeps working until then; the search
  names are not in the shim.

## Repository

- `mirror.yml`: a push to main or a release tag on PlateerLab/xgen-ontology-build is
  pushed on to the mirror jinsoo96/js-ontology-build at once, with a deploy key that can
  write only that repository; the mirror's own scheduled sync drops from every 15 minutes
  to once a day as a safety net.
- CI and Publish build both distributions; `tools/sync_knowledge_contract.py` writes the
  contract copy into `src/xgen_ontology_build/exchange`.

# 0.14.0 (2026-10-08)

A second pass over the production build (xgen-documents, develop 58429bd), area by area:
the LLM-free extraction, the table build, the relation pass and the post-build were read
against the library end to end, and what still stood apart is brought over here. The
deterministic extraction was unchanged and still gives the production code's result on the
regulation corpus (60 documents, 1,532 chunks re-checked: chunk facts, Hearst pairs,
concepts, entities, relations and data values identical).

## Rows of a database table

- `build_from_rows`, `OntologyBuilder.build_rows` / `extend_rows` and the facade
  `build_from_db_rows`: the production loader's path for a SELECT result, which the
  library only approximated through CSV text. The schema is declared, not guessed:
  `pk_candidates`, `label_column`, `column_types`, `fk_relations` (each checked by
  `normalize_fk_relations`: the key column must be a column of the rows and must point at
  the target's primary key).
- A value's xsd type comes from its Python type (`xsd_type_of`: bool, int, Decimal and
  float, datetime, date) and its literal from `value_text` (`true` / `false`, ISO 8601
  dates, decoded bytes, JSON for dict and list columns). A column's type is its first
  recorded value's.
- Every row is its own source, `"{source_id}:{pk}"` (`"{source_id}:row{i}"` without a key),
  kept as a chunk whose text is the row's `column: value` lines (`row_chunks`), so one row
  can be retracted, searched and loaded again on its own. `extend_rows` retracts a row
  loaded before and loads it again, so a changed attribute does not pile up next to the
  old one.
- A foreign-key cell that is SQL NULL makes no relation; an empty string still points at
  `"{to_table}_"`, as in production. A single table is never a fact table.
- `build_from_csv` takes `header_patterns`; `.xlsm` is a table extension in the facade too.

## Relation formation in groups

- The pass runs in groups of about `group_size` chunks (default 300, the production
  `ONTOLOGY_BUILD_GROUP_SIZE`), a document never split. The vocabulary a group defines or
  grows is declared before the next group, which classifies against it, and the next
  group's batches are planned with what the calls so far taught about relations per unit
  and tokens per item. Before, one plan and one pass covered the whole corpus, so a
  vocabulary grown on the way never reached a classification prompt within the build.
  `BuildReport.relation_stats` gains `groups`.
- `should_stop`: a stop inside a group discards that group whole (its relations, its
  vocabulary, its enriched marks), as the production cancel does; the groups before are
  kept, the end-of-build canonicalization and the LLM class synonyms are not run, and the
  report says so (`relation_stats["stopped"]`, a note).
- A provider that reports `length_error` from `generate_json_meta` is read as a cut-off
  answer: the batch is halved as for a truncated one.
- Canonicalization shows the model each name's two best-evidenced examples (weight, then
  subject), as the production store query does.

## Post-build

- Class synonym folding (LLM) and vector class dedup run when text was extracted in this
  build, as in production (a table-only or rows-only build has none to find); the LLM pass
  sees at most `max_classes` (500) classes, and names it folds are not offered to the
  vector pass, so a name is not folded and a cluster's canonical at once.
- `extend` and `retract` take `doc_of` (chunk id -> document) for chunks a loaded graph
  holds without a document, so the spread of a head noun is measured over the right number
  of documents; the document is remembered on the chunk afterwards.
- `review_quality` counts `orphan_class_count` (classes nothing instantiates and no object
  property declares) with the production warning, counts a dangling subject as well as a
  dangling object, and takes a relation's own source chunks as grounding.

## PgGraph

- A node keeps the URI the store holds it under: `load` fills `Ontology.uris`
  (`(kind, label) -> uri`), `graph_rows` and `write` use it, and an appending write reads
  the rows as they are now (`label_uris`). A head promoted to a class keeps the instance
  URI it was stored under and changes kind, as the production promotion does. Before, a
  promoted head came back from the store as a second node under a new URI.
- `write_schema`: `ontology_schema` gets every class (description, parent, `source='csv'`
  for a class built from a table, which the product's dedupe reads as a deterministic
  identity), every object property and every datatype property with its range, not only
  the relation vocabulary. `load` reads them back (`Class.source`, `DataProperty.range`,
  descriptions). The DDL carries the product's `is_auto_generated` / `is_confirmed`
  columns, so a table the library created accepts the product's upsert.
- `XGEN_UPLOAD_HEADER_RE`: the retrieval preamble the XGEN document service prepends to
  every chunk, exported for `header_patterns`; the library still strips nothing by itself.

## Where the library differs from the product, on purpose (with 0.13's list)

- `sameAs` folds the two names into one node; the product keeps two nodes and a `sameAs` edge.
- Orphan classes are pruned to a fixpoint; the product prunes one round.
- subClassOf cycles are broken; the product keeps both edges. Inherited properties are
  materialized down the whole chain in one build; the product does one level per build.
- An appending write unions a node's attribute values; the product keeps the last write's list.
- A table row's label is never key-merged; the product merges table instances in a hybrid build.
- A lost chunk (no answer came back) stays in the built set and only its relation pass is
  asked again; the product re-extracts it as well.
- The rebuild that keeps the previous graph until success is the store's transaction
  (`write(replace=True)` in one transaction), not a library step.
- `relation_count` and the ungoverned predicates of the quality review follow declarations;
  the product reads them off schema edges, which exist only for properties with a domain
  and a range.

# 0.13.0 (2026-10-08)

The build catches up with the production build (xgen-documents, develop): its relation
pass, its store-loading rules and its term layer. Each part was checked against the
production code on a 443-document, 14,924-chunk regulation corpus and gives the same
result: chunk facts, Hearst pairs (5,426), concepts (24,457), entities (178,845),
relations (2,480) and data values (2,709) with no difference; the name functions on
153,233 inputs with no difference; units and mentions (15,907 and 66,640 on 60
documents), batch plans (11,388 units, 198 batches) and every prompt, schema and parse of
the relation pass identical with the product's prompts. The few places the library
differs on purpose are listed below.

## Relation formation (`mode="enrich"`)

- New `relation_units`: a document is cut into units of evidence (a sentence or clause,
  a table row) and the graph nodes each unit mentions are located and numbered, with no
  model. A unit with fewer than two mentions is never sent, so the number of calls is
  known before the first one.
- New `relation_formation.RelationFormer`, the product's relation classifier. Units are
  batched by an input budget (`char_budget`) and by the expected size of the answer,
  learnt as calls come back. Relation names come from a vocabulary of English
  identifiers with definitions; with none, one is defined first from units sampled
  evenly across the corpus. The model answers with numbers (unit, subject, relation,
  object). A cut-off answer keeps its complete part and the batch is halved; an empty
  answer is retried once after a pause; an answer with no relation is asked once more.
  A row that does not hold together (a number out of range, a subject that is its own
  object, a name that is not an English identifier or is the graph's own vocabulary) is
  dropped. Names used often enough outside the vocabulary (a natural break in their
  frequencies, no threshold) are defined and join it.
- `canonicalize_predicates`, run at the end of an enrich build: every relation name is
  held to the vocabulary. A name the model maps to a vocabulary name takes it, a frequent
  unmapped name joins the vocabulary, the rest become `relatedTo` with the old name as
  label.
- A relation keeps the chunk that states it as its source and, for a table row, the
  object cell's column name as its label. The vocabulary is declared as object
  properties with their definitions (`ObjectProperty.description`).
- A chunk no answer came back for (a failed call, or no vocabulary to classify against)
  is not marked enriched, so the next `extend` asks it again.
  `BuildReport.relation_stats` reports units, calls, relations, vocabulary size, new
  names, lost chunks and canonicalization.
- Prompts are a `RelationPrompts`. `KO_RELATION_PROMPTS` are the product's prompts
  verbatim; `EN_RELATION_PROMPTS` (the default) are the same instructions in English.
- `OntologyBuilder` takes `relation_prompts`, `char_budget`, `max_output_tokens`,
  `llm_timeout`, `max_workers`, `retry_backoff` and `should_stop`. Model limits are
  passed in, never assumed.
- `invoke_json_meta` returns the parsed answer with whether it was cut off, whether it
  failed and the output tokens. An LLM with `generate_json_meta` reports these exactly;
  otherwise the lenient reader flags a salvaged answer as cut off.
- `DocumentExtractor.extract_relations` delegates to `RelationFormer` and takes
  `labels_by_chunk` and `vocabulary`.

## Store-loading rules (`normalize_graph`)

- A relation name is an English identifier. `canon_predicate`: an ASCII name is
  camelCased (`reports to` -> `reportsTo`); anything else becomes `relatedTo` with the
  original as its label. Structural names (`instanceOf`, `subClassOf`, `sameAs`, ...)
  are left as they are; `RESERVED_PREDICATE_NAMES` and `GRAPH_VOCABULARY` name them.
- An endpoint is an existing instance or a declared class, else a new instance only
  when it has the shape of an entity (`is_entity_shape`). A declared class is never
  spawned as an instance.
- The same relation stated twice is one relation: its sources are the union and its
  weight is the number of distinct chunks that state it.
- A pair that has a named relation drops its `relatedTo`; that relation's chunks move
  onto the two nodes so no chunk link is lost.
- The legacy related predicate `관련` reads as `relatedTo`. The default
  `related_predicate` of the facades is now `relatedTo`.
- `BuildReport.normalized` counts `relabeled` and `superseded`.

## Term layer

- Whether a string can be a name is judged by its shape in Unicode categories
  (`is_name_shape`: it has a letter, it is not a sentence, its brackets are balanced),
  not by a word list. HTML entities and escapes are decoded first.
- Record dumps (`key: value` lines) read as one row. A whitespace row dump is a table
  only when every row is cells; a list-marker line is prose.
- A pipe table's title is told apart from a sentence by morphology. Escaped pipes and
  `[Table N]` captions are read.
- A word glued to a number does not start a name; a Hearst run does not cross a line
  break.
- An entity cell becomes a `relatedTo` relation labelled with its column name, for the
  enrich pass to name. `MAX_COVERAGE` (0.30) and the document count behind it live in
  one place.
- `extract_from_chunk` is removed (the product removed it); `extract_chunk(...).ents`
  gives the same names.

## Hierarchy

- A head whose compounds are spread over the corpus is not a type (0.12): names an
  earlier step had already typed by that head now count toward its spread and are
  retyped as `relatedTo` neighbours too. A table's own class and protected names are
  left as they are. `is_name_child` is exported; the build notes count the retyped names.

## Tables

- Each sheet of a workbook is a table of its own (`split_sheets`); `.xlsm` is read.
  HTML tables with row and column spans and TSV are read (`table_cell_rows`).
- A header row repeated down a table is skipped.
- Serial numbers that merely overlap are not a foreign key, and an FK's direction comes
  from which side is a primary key.
- A class made from a table is marked `Class.source = "table"`.

## Retract

- A relation with recorded sources goes when all of them are gone and keeps the rest
  when some survive. A relation with no recorded sources keeps the co-occurrence rule
  (its two ends must still share a chunk).

## PgGraph

- `ontology_edges` has a `label` column (added to existing tables); new tables
  `ontology_edge_sources` (the chunks that state each relation) and `ontology_schema`
  (the relation vocabulary as `ObjectProperty` rows, where the product keeps it).
- `write` stores sources and the vocabulary, and an edge written again takes the new
  weight and keeps a label it had. An appending write (`replace=False`) renames legacy
  `관련` edges to `relatedTo` and drops a `relatedTo` edge whose pair now has a named
  relation.
- `prune_chunks` drops a relation with its last source; a relation that keeps some
  sources takes their count as its weight. Relations with no sources fall back to
  co-occurrence, and orphan source rows are cleaned. `clear` removes
  the collection's source and schema rows too.
- `load` reads labels, weights, sources and the vocabulary back.
- `graph_rows` returns edges as 7-tuples `(subject, predicate, object, predicate_uri,
  kind, weight, label)`; `edge_source_rows` is new.

## Where the library differs from the product, on purpose

- Weight is the number of distinct chunks that state a relation. The product adds 1 on
  every load, so the same chunk loaded twice counts twice.
- Retract judges a relation by its own sources. The product judges it by whether its
  two ends still share a chunk.
- A `relatedTo` is superseded on every build, not only when it is written to the store.
- A column of the same name in two tables still points at the table chosen by the
  library's rule (the table whose name matches the column, then the smaller one). For
  differently named columns the product's rules are ported.
- When a relation is dropped, its chunk is not linked to the subject node. The product
  still links it.

## Known limit (shared with the product)

- A trailing symbol is stripped from a name, so "AA+" reads "AA". Unicode categories
  cannot tell `+` from `×`, `=` or `→`, which are junk in the same position; a word
  list would be the only way, and the term layer does not use one.

## Removed and deprecated

- `Deduplicator` no longer merges relation names (the property passes and
  `_apply_property` are gone): relation names are held to a vocabulary instead, and the
  "keep the original-language name" rule folded English names back into document
  words. The pipeline no longer calls `merge_predicates`, which is deprecated;
  `BuildReport.predicates_merged` stays for compatibility and is 0.
- The old relations-only prompt of `DocumentExtractor` is replaced by relation
  formation. `mode="llm"` (full extraction) is unchanged.

# 0.12.6 (2026-09-29)

- Hearst pattern, corrected after review: the hypernym is no longer dropped when a
  particle follows it. The 0.12.x gate on adverbial, adnominal and conjunctive
  particles also removed correct pairs ("은행, 보험사 등 금융기관에", "금융기관의",
  "관계기관과"). Only the construction "X와/과 관련" (related to X) is skipped now: a
  list that is about a relation to X is not a list of kinds of X.
- Brackets are matched as pairs. A bracket left open by a chunk cut opens nothing, so
  one "제3조(적용범위" no longer silences every later "등" in the chunk. A quoted title
  inside a closed pair is still skipped.
- Row dumps: one verbal cell ("있음", "해당함") no longer turns a table into prose. A row is
  prose when a third of its words are case-marked or more than half are verbal.

# 0.12.5 (2026-09-29)

- Term layer, corrected after review: a short Latin term ("id", "db", "js", "Go") is a
  name again; only a string with no letter at all is excluded, which is what removes
  the list number "5.". The scrap words ("of", "mm") are stopped at their source
  instead: a word cut from a multi-word phrase is a name of its own only when it is
  Hangul, because a Korean word cut from a noun run is a noun while a Latin word cut
  from a phrase may be anything. The whole phrase is unaffected.

# 0.12.4 (2026-09-29)

- Term layer: a pipe-table caption must have the shape of a name, the same test an HTML
  caption already passed. A bare list number ("5.") above a table had become a class
  with 574 rows as its instances on a 764-document corpus. A Latin-script scrap shorter
  than an acronym ("of", "mm": lower-case, under three letters) is no longer a name
  anywhere; "IT", "DB", "PC" still are. Found in the viewer of a production graph.

# 0.12.3 (2026-09-29)

- `induce_aliases` is removed, and with it the synonym-by-reordering rule of 0.12.0-0.12.2.
  Review of the production port showed the rule was not safe as an automatic merge: a
  whole name has no morpheme boundary to cut at (0.12.2's rotation key still cut at
  every letter), a shared chunk is weak evidence when tables and lists put different
  things side by side, and even proper names of several parts can mean different things
  in another order. Measured on 764 regulation documents it also bought nothing: 2 of
  1,646 whole names shared a rotation key, both different things ("금융리스" / "리스금융").
  Synonyms still come from code-prefix `sameAs`, key merging and fragment folding; the
  next evidence-based source is the definition pattern in the text itself
  ("여신전문금융회사(이하 '여전사')"). `BuildReport.aliased` is gone.

# 0.12.2 (2026-09-29)

- `induce_aliases`: a name the analyzer keeps whole is keyed by its rotation, not by its
  sorted letters. "호텔신라" and "신라호텔" are the same string cut once and swapped and
  still fold; two different names that merely share letters ("김민수정" / "김수민정") no
  longer can. Raised in review of the production port.
- `_morph_starts` cache is bounded (LRU, 262,144 names) so a long-lived process does not
  grow without limit.

# 0.12.1 (2026-09-29)

- `induce_aliases` folds proper names only. A common-noun compound has its head last,
  so another order is another thing ("정보보호" is not "보호정보"); every part of a
  candidate must be a proper noun (NNP), and a suffix stays a part ("신라호텔식" is not
  a spelling of "신라호텔"). Raised in review of the production port.

# 0.12.0 (2026-09-29)

**The build follows the ontology-learning layers, with evidence at each one and no word
lists or new size caps.** Found by reading a production graph (521 regulation documents,
174,540 nodes, 24,969 classes): particle-bearing "entities" (`정보가`, `제1항에`), is-a edges
read out of quoted law titles (`사행행위 ⊂ 규제` from 「사행행위 등 규제 및 처벌특례법」) and
negated clauses (`도박 ⊂ 업무` from "gambling, ... etc. *unrelated to* work"), and everyday
words (`여부`, `기준`, `업무`, `사항`) as classes with hundreds of members each: 57% of all
subClassOf edges had such a parent. All reproduced by rebuilding the same 764 documents here.

- **terms** -- `parse_row_dump` reads the words of a candidate row by morpheme tag: an
  inflected verb, or a third of the words ending in a particle, makes the line a sentence,
  not a row. Article-numbered regulation paragraphs no longer pass as tables.
- **synonyms** -- `induce_aliases` (new, in the pipeline before predicate merge): two names
  made of the same parts in another order ("호텔신라" / "신라호텔"; parts = every morpheme but
  particles, endings, copulas and punctuation, or the syllables of a name the analyzer keeps
  whole) that the corpus uses together in a chunk are one name; the spelling used in more
  chunks wins. Grouped by part bag, so linear in the number of names. `BuildReport.aliased`.
- **concepts** -- a head noun is a class only when its compounds are not spread over the
  corpus: the same discriminativeness ratio that filters entities (30%), measured over
  *documents* (a corpus of many documents dilutes any name's share of chunks; "whether"-names
  still appear in most documents). Otherwise its names become `relatedTo` neighbours. Applied
  in the extractor's head promotion and in `induce_head_noun_hierarchy` (`doc_of`,
  `corpus_docs`; chunk share when documents are unknown).
- **taxonomy** -- `extract_hearst_pairs` skips an anchor inside brackets (the analyzer's
  SSO/SSC tags) and a hypernym followed by an adverbial, adnominal or conjunctive particle
  (JKB/JKG/JC: it modifies a later noun). Reads tags, not word lists.
- **relations** -- the neighbour link is `relatedTo` (graph vocabulary), configurable as
  before through `related_predicate`.
- The 30,000-name morphology budget is gone: `_morph_starts` is cached per name, so an
  incremental build pays only for its new names.

764 documents, 51,006 chunks, before -> after: Hearst pairs inside a quoted title 2,003 ->
9, raw Hearst pairs 51,314 -> 18,056, particle-ending entity names 7.6% -> 3.5% (most of the
rest are regular names such as 정의, 결과), chunks misread as row dumps 2,247 -> 983, classes
36,511 -> 30,560, subClassOf edges with an everyday-word parent 16,239 -> 9,770, hierarchy
edges 26,988 -> 20,303, build time unchanged (1,321 s -> 1,313 s). The former top classes
(여부 594, 기준 665, 업무 595, 포함 641) are neighbour groups now.

# 0.11.0 (2026-09-28)

**A deleted document leaves the graph as a delta.** The production deletion path
(`prune_chunks`, in the product since 2026-09-17) ported over the build models and the
graph tables; every other build stage was checked against the production source and is
unchanged since the 0.6-0.8 port.

- **`build.retract.retract_chunks`** (new) -- take chunks out of a built graph in place:
  their links go, then whatever they were the only evidence for. An entity's evidence is
  every chunk that links it (its own, its relations', its attributes'). An individual with
  no chunk left is an orphan; a class with no chunk left is an orphan unless something still
  refers to it (an individual typed by it, a subclass, a property ranging over it, a
  relation; its own declarations do not count), evaluated to a fixpoint so a class whose
  only individual or subclass went goes too; properties declared only by orphan classes go
  with them; a relation between two survivors that no longer share a chunk has lost its
  evidence and goes, except relations made from name structure (`structural_predicates`:
  the "related" neighbour link, `sameAs`). Survivors keep only their surviving chunks.
- **`OntologyBuilder.retract(onto, chunk_ids)`** -- retraction followed by the post-build
  (merge, hierarchy, normalization) over what is left, the production removal-only build.
  **`OntologyBuilder.extend(..., retract_missing=True)`** treats `documents` as the whole
  current corpus and retracts the chunks that left it before extracting (the store's
  `baseline - snapshot`); off by default because `extend` also accepts only the new
  documents. `removed_chunks(onto, documents)` shows what that would retract, next to
  `unbuilt_chunks`. `BuildReport.retracted` carries the counts; the progress callback gets
  a `retract` stage.
- **`PgGraph.prune_chunks(chunk_ids)`** -- the same rules on the tables, several statements
  in the production order (links last, so an interrupted run on an autocommit connection is
  retried from the same state), over temp tables so a large deletion needs no long `IN`
  lists; portable across PostgreSQL and sqlite. Verified on both with identical results.
  `ensure_schema()` adds the two read indexes the production migration has that the DDL
  here lacked (`ontology_node_chunks (collection_id, uri)`,
  `ontology_enriched_chunks (collection_id)`).
- The producer fixture `tests/fixtures/knowledge-v1.json` regenerated: the embedded build
  report gained the `retracted` key, nothing else changed.

# 0.9.1 (2026-09-17)

- Normalize UTF-16 surrogate pairs from document parsers and model JSON into
  valid Unicode scalar values throughout the portable knowledge contract.
- Replace isolated malformed surrogates with U+FFFD so one damaged character
  cannot abort an otherwise valid knowledge build or snapshot publication.

# 0.9.0 (2026-09-15)

- Add `build_knowledge` and `export_knowledge` using the current production build pipeline with explicit immutable source identities.
- Preserve directory hierarchy, embedding profiles/vectors, chunk revisions and source locators through graph construction.
- Export typed facts with resolvable evidence, corpus-qualified entity IDs, schema and build diagnostics through `knowledge/v1`.
- Add checksummed atomic JSON exchange, structural schema, semantic validation and digest-pinned standalone consumer records.
- Use complete replacement builds for source modification/deletion; keep publication and job transactions in the host.
- Require an LLM for explicitly requested enrichment on the new API and propagate cooperative cancellation between stages.
- Lazy-load/deprecate legacy search entry points while preserving their compatibility.
- Add independent build/import tests and a producer fixture for retrieval conformance.

# Unreleased

- Mirror sync (`sync-from-xgen-ontology-build.yml`) pushes with the mirror's `SYNC_TOKEN`
  secret when set: `GITHUB_TOKEN` cannot push a commit that touches `.github/workflows/`,
  so the mirror had stalled at the first origin commit that changed CI.

# 0.8.0 (2026-09-15)

- **`backends.postgres.PgGraph`** (new) -- the production graph tables
  (`ontology_nodes`, `ontology_edges`, `ontology_node_chunks`,
  `ontology_enriched_chunks`) over any DB-API connection, no driver imported.
  `write(onto, replace=)` stores a build (rows identical to the production loader's
  on a 16,418-node build: URIs, kinds, edge kinds, attributes, chunk links),
  `load()` brings a stored collection back into an `Ontology` for
  `OntologyBuilder.extend`, `mark_enriched` / `enriched_chunk_ids` carry the enrich
  baseline, and the `GraphStore` protocol (`search_labels`, `class_instances`,
  `neighbors`, `count_class`, `get_node`) runs the one-shot search straight on the
  tables. `ensure_schema()` creates the tables where the product has not
  (PostgreSQL DDL; `dialect="sqlite"` for tests). `graph_rows(onto)` exposes the row
  shape. Optional extra `postgres` (psycopg).
- **`OntologyBuilder(progress=fn)`** -- `fn(stage, detail)` per pipeline stage
  (`start / tables / extract / enrich / llm / dedup / hierarchy / finalize / done`),
  the hook an application drives its job table from. Job and session bookkeeping
  itself stays out of the library.
- **Removed `SCSGenerator`**, the `scs=` flag and `Ontology.scs_profiles`: the
  context-profile generator was retired in the product (no caller left) and was off
  by default here since 0.1.0.

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

## 0.10.0

- Add `build_resource_fragment` for bounded, per-resource ontology builds.
- Add `assemble_resource_fragments` for deterministic graph publication and
  retraction by omission of deleted or replaced resource revisions.
- Keep graph fragments independent from embedding vectors while preserving
  exact source evidence and immutable revision identities.

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
