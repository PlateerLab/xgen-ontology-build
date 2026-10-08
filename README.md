# xgen-ontology-build

**The ontology build, as a library of stages.** Documents or tables go in; a *clean*
knowledge graph with provenance comes out, built the way the XGEN document service
builds its collections: extract (no LLM unless you ask for one), post-build (merge names,
induce the hierarchy, apply the store-loading rules, retract what left), store (the XGEN
PostgreSQL tables or any SPARQL 1.1 store), exchange (knowledge/v1, RDF). **Zero hard
deps** in the core, **zero infra** to try it. Search is not here: export a build and search
it with [xgen-omnifuse](https://github.com/PlateerLab/xgen-omnifuse).

```python
from xgen_ontology_build import build_from_csv

onto = build_from_csv({                       # no LLM, no DB, no API key
    "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20",
    "colors":   "color_id,name\n10,Red\n20,Blue",
})
print(onto.stats())          # {'classes': 2, 'instances': 4, 'relations': 2, ...}
print(onto.to_turtle())      # standards RDF/Turtle
```

A knowledge graph built with it (degree-sized nodes, community clusters), shown in a
graph explorer:

<p align="center">
  <img src="assets/ontology-graph.png" alt="An ontology knowledge graph built with xgen-ontology-build" width="760">
</p>

Documents build **without an LLM too**. The base build reads document *structure*:
tables become row entities with their column values as attributes and relations,
prose yields noun-phrase entities, and the hierarchy comes from Hearst patterns
and from the structure of the names themselves. Mix tables and text freely; raw
documents are **parsed and chunked** for you:

```python
from xgen_ontology_build import build_from_files, build_from_text, CallableLLM

# zero LLM calls (the default, mode="basic"); pdf/docx/xlsx via the [files] extra
onto = build_from_files(["policy.pdf", "products.csv"])
onto.report.llm_calls            # 0
onto.report.quality["score"]     # graph-reviewer score of the finished build

# an LLM is optional: mode="enrich" names the relations between the entities the base
# build found, sentence by sentence and row by row, from one vocabulary of relation names;
# mode="llm" is the older full LLM extraction of schema and instances
llm = CallableLLM(lambda p, system="": my_model(system, p))     # OpenAI / Anthropic / vLLM / …
onto = build_from_text("Rule A applies to Acme Bank since 2020. ...", llm=llm, mode="enrich")
```

The enrich pass sends only what can hold a relation: a unit of evidence (a sentence, a
clause, a table row) that mentions two or more of the graph's nodes, with the mentions
numbered. The model answers with numbers and a relation name from the vocabulary, so a
relation always points at the chunk that states it, and the number of calls is known
before the first one. The pass runs in groups of about `group_size` chunks (300, a
document never split): what a group defines or grows is the vocabulary the next group
classifies against.

```python
from xgen_ontology_build import OntologyBuilder, KO_RELATION_PROMPTS

builder = OntologyBuilder(llm, mode="enrich",
                          relation_prompts=KO_RELATION_PROMPTS,   # the product's prompts; English by default
                          char_budget=10000, max_output_tokens=8192, max_workers=4, group_size=300)
onto = builder.build(docs)
onto.report.relation_stats      # units, calls, groups, relations, vocabulary, new_vocabulary, lost_chunks, ...
[(op.name, op.description) for op in onto.concepts.object_properties]   # the vocabulary
```

## The stages

The package is laid out in the production build's order. Each stage is a module of
functions you can import on its own; `OntologyBuilder` (`pipeline.py`) only wires them.

| Package | Stage | What it does |
|---|---|---|
| `text/` | shared | `korean`: whether a string can be a name (Unicode categories, no word list), morphology when `kiwipiepy` is installed · `tokens`: CJK bi-gram tokens, IRI local names · `chunk`: boundary-aware chunking with stable ids · `parse`: txt/md/html/csv built in, pdf/docx/xlsx via `[files]` · `dictionary`: alias → canonical at build, query and indexing time · `translate`: English IRI names via an LLM |
| `extract/` | 1 | `deterministic`: documents → ontology with **no LLM**: HTML tables, whitespace row dumps, pipe grids and record dumps become row entities typed by the subject column's header, with value cells as attributes and entity cells as `relatedTo` relations labelled with the column name; prose yields noun-phrase entities and (entity, attribute, value) facts; a shared head noun promotes to a class; names in more than 30% of the chunks are dropped as non-discriminative. The XGEN retrieval preamble is `XGEN_UPLOAD_HEADER_RE` for `header_patterns` · `tabular`: table files → ontology by the star-schema rule (table → Class, FK → ObjectProperty by same-name / normalized-name / value-overlap evidence with the direction from the primary key, column → DataProperty, dimension rows → instances; large fact tables and tables with no name column stay schema-only; one table per sheet, spans, TSV, repeated headers), and **the rows of a database table** (`build_from_rows`) with the schema declared by the caller, values typed by their Python type and each row its own source · `relation_units`: units of evidence and the numbered mentions in them (no LLM) · `relation_formation`: the enrich pass, batches planned by an input budget and the expected answer size, a cut-off answer halved, names held to a vocabulary that grows by a natural break in frequencies · `llm_extract`: the older full LLM extraction |
| `postbuild/` | 2 | `taxonomy`: hierarchy without an LLM, Hearst patterns in prose and the structure of names (a boundary-aligned head noun is the parent; a head spread over the corpus is not a type) · `dedup`: content-morpheme keys for instances, LLM class synonyms (`mode="enrich"`, at most 500 classes), embedding clusters when an embedder is given · `hierarchy`: genuine is-a only, cycles broken, self-typed repair, property inheritance · `finalize`: the store-loading rules (a relation name is an English identifier or `relatedTo` with the original as label; endpoints resolved; duplicates merged with weight = distinct stating chunks; a named relation supersedes `relatedTo`) · `govern`: predicate governance for `mode="llm"` · `resolve`: fuzzy entity resolution (off by default) · `retract`: a deleted document out of the graph as a delta · `quality`: the graph-reviewer score and the production review's counts · `community`: Louvain |
| `store/` | 3 | `postgres`: `PgGraph`, the XGEN graph tables over any DB-API connection: write, append, load for an incremental build, prune, schema rows, URIs kept across round trips · `sparql`: any SPARQL 1.1 store, read and write · `memory`: an in-memory sink for tests |
| `exchange/` | out | `knowledge`: the knowledge/v1 contract (byte-identical copy of `contracts/knowledge/v1`), `knowledge_build`: `build_knowledge`, `export_knowledge`, resource fragments · `emit`: Turtle (zero-dep) and OWL/RDF-XML (rdflib) |

### Incremental resource graphs

Large file stores should build one immutable resource revision at a time. The
fragment contract keeps graph construction independent from hierarchy and
embedding projections, while assembly retracts deleted revisions without
rebuilding unchanged files ([docs/knowledge.md](docs/knowledge.md)).

```python
from xgen_ontology_build import build_resource_fragment, assemble_resource_fragments

fragment = build_resource_fragment(one_file_bundle, snapshot_id="fragment:file-7:r3")
snapshot = assemble_resource_fragments(
    current_source_bundle,
    [fragment, *unchanged_fragments],
    snapshot_id="graph:42",
)
```

### Keep building: incremental, rows, dictionary, identifiers

```python
from xgen_ontology_build import OntologyBuilder, TermDictionary, removed_chunks, unbuilt_chunks

builder = OntologyBuilder()                      # or mode="enrich" with an llm
onto = builder.build({"2024.md": [...chunks...]})

# later: only the chunks the ontology has not seen are extracted (by chunk id); the
# post-build re-runs over the whole graph, hierarchy induction only where new names can
# reach. In enrich mode the LLM pass covers only chunks it has not asked about yet.
unbuilt_chunks(onto, {"2024.md": [...], "2025.md": [...]})     # what extend() would do
builder.extend(onto, {"2024.md": [...], "2025.md": [...]})

# a deleted document is a delta too: pass the whole current corpus and the chunks that
# left it are retracted before extraction (their links, the nodes and relations they were
# the only evidence for), then the post-build runs over what is left
removed_chunks(onto, {"2025.md": [...]})                       # -> ["2024.md#0", ...]
builder.extend(onto, {"2025.md": [...]}, retract_missing=True)
builder.retract(onto, ["2024.md#0"])                           # or by chunk id, no extraction

# the rows of a database table: the schema is declared, every row is its own source
builder.build_rows("colors", ["id", "name", "group_id"], rows, source_id="db1:colors",
                   pk_candidates=["id"], label_column="name",
                   fk_relations=[{"from_column": "group_id", "to_table": "color_groups",
                                  "to_column": "id", "to_pk_column": "id"}])
builder.extend_rows(onto, "colors", cols, changed_rows, source_id="db1:colors", pk_candidates=["id"])
builder.retract(onto, ["db1:colors:2"])                        # a row that left the table

# a term dictionary (acronym -> full form, house spelling -> official one) applies at
# build time, at query time and at indexing time
d = TermDictionary("finance")
d.bulk_import([{"alias": "DSR", "canonical": "총부채원리금상환비율"}])
OntologyBuilder(dictionary=d).build(docs)                    # aliases canonicalized in the graph
d.normalize_query("DSR 60% 고객")                             # -> "DSR 60% 고객 총부채원리금상환비율"

# English IRI local names for a non-ASCII ontology, translated once per name and cached
onto.translate(llm)          # classes UpperCamelCase, properties lowerCamelCase
onto.to_turtle()             # :CreditRating a owl:Class ; rdfs:label "신용등급"
```

`OntologyBuilder(progress=fn)` reports each stage (`start / retract / tables / extract /
enrich / llm / dedup / hierarchy / finalize / done`) so a job table can be driven from it;
`should_stop=fn` stops the relation pass between calls (a group under way is discarded).
Job and session bookkeeping is the application's.

## Stores

The algorithms talk to small protocols (`LLM`, `Embedder`, `Morphology`, `GraphSink`,
`GraphStore`), never to a database.

The production system keeps its graph in PostgreSQL (`ontology_nodes` / `ontology_edges` /
`ontology_node_chunks`, `ontology_edge_sources` for the chunks that state each relation,
`ontology_schema` for classes, properties and the relation vocabulary). `PgGraph` speaks
that schema over any DB-API connection (psycopg 2/3; sqlite in the tests), so a library
build lands where the product reads it and a stored graph can be extended:

```python
import psycopg
from xgen_ontology_build import OntologyBuilder, PgGraph

pg = PgGraph(psycopg.connect(DSN, autocommit=True), collection_id="col-1")
pg.ensure_schema()                        # no-op where the product already created the tables
pg.write(OntologyBuilder().build(docs))   # rows identical to the product's own loader

onto = pg.load()                          # back into build models (chunk ids only; onto.uris keeps
                                          # each node's stored URI, so it is written back as itself)
OntologyBuilder().extend(onto, more_docs, doc_of=chunk_to_document)   # incremental
pg.write(onto, replace=False)             # append; a node seen again merges its attributes,
                                          # ontology_schema gets classes, properties and the vocabulary
pg.prune_chunks(deleted_chunk_ids)        # a deleted document, applied as a delta on the tables
                                          # (a relation goes with the last chunk that states it)
```

Any SPARQL 1.1 store takes a build as Turtle:

```python
from xgen_ontology_build import fuseki
store = fuseki("http://localhost:3030", "ds", user="admin", password="…")
onto.push(store, graph="urn:my-graph")
```

To search a build, export it: `export_knowledge(onto, ...)` gives a knowledge/v1 bundle
(hierarchy, embeddings and graph as one portable JSON) that xgen-omnifuse consumes.

## Install

```bash
pip install xgen-ontology-build                 # core, zero deps
pip install "xgen-ontology-build[files]"        # + pypdf / python-docx / openpyxl (parse pdf/docx/xlsx)
pip install "xgen-ontology-build[rdf]"          # + rdflib (OWL / RDF-XML emit & parse)
pip install "xgen-ontology-build[korean]"       # + kiwipiepy (Korean morphology)
pip install "xgen-ontology-build[vector]"       # + qdrant-client (embedding adapters)
pip install "xgen-ontology-build[postgres]"     # + psycopg (PgGraph takes any DB-API connection)
```

`xgen-ontology` is the former name. Until 0.16 it is published as a shim that installs this
package and re-exports it under `xgen_ontology` (old module paths aliased, with a
`DeprecationWarning`); see [compat/](compat).

Run the demos with no install:

```bash
python examples/build_csv.py
python examples/build_documents.py
python examples/knowledge_build.py out.json
```

## Design — algorithms as a library

- **`dependencies = []`** — the core needs nothing but the standard library; the Turtle
  writer is hand-rolled.
- **No LLM in the loop unless you ask for one** — the base build and the whole
  post-build are rules over document structure and name structure, so a build is
  reproducible and costs nothing; `mode="enrich"` / `"llm"` bring a model in for the
  parts only a model can do (relations stated in prose, free-form schema).
- **Same result as the production build** — the stages are ported from the XGEN document
  service and checked against it on its own corpus (see CHANGELOG); the places the
  library differs on purpose are listed there.
- **English-neutral architecture, Korean-tuned defaults** — the morphology-aware parts
  (noun phrases, sentence-fragment detection, head nouns, common words) use the optional
  `korean` extra and degrade gracefully without it; prompts, name→URI translation and
  the tokenizer are pluggable.
- **Bring your own everything** — LLM (`generate(prompt, system)`, optionally
  `generate_json_meta(...)` for structured output with finish reasons), embedder,
  morphology, graph store.

```
src/xgen_ontology_build/
  models.py        # Class/Property/Concepts (T-Box), Instance/Relation/DataValue (A-Box), Node/Chunk
  protocols.py     # LLM / GraphStore / GraphSink / Morphology / Embedder
  llm.py           # CallableLLM, the lenient JSON reader, invoke_json_meta
  ontology.py      # Ontology — one build's result: schema, individuals, relations, values, chunks, uris
  pipeline.py      # OntologyBuilder — the stages in the production order; build / extend / retract / rows
  facade.py        # build_from_csv / build_from_documents / build_from_db_rows / build_from_triples
  text/            # korean · tokens · chunk · parse · dictionary · translate
  extract/         # deterministic · tabular · relation_units · relation_formation · llm_extract
  postbuild/       # taxonomy · dedup · hierarchy · finalize · govern · resolve · retract · quality · community
  store/           # postgres (PgGraph) · sparql (SparqlGraph) · memory (InMemoryGraphSink)
  exchange/        # knowledge (the v1 contract) · knowledge_build · emit (Turtle / OWL)
compat/                 # the former name xgen_ontology, re-exporting this package (until 0.16)
contracts/knowledge/v1  # the exchange contract's source: records.py, schema.json, lock.json
examples/  tests/  tools/
```

## Roadmap

- async relation pass (parallel groups)
- Neo4j / property-graph store adapter
- RDF-star / qualified statements (n-ary relations, provenance) in emit

## Repository model

[`PlateerLab/xgen-ontology-build`](https://github.com/PlateerLab/xgen-ontology-build) is the
source of truth and publishes the Python package `xgen-ontology-build` (and, until 0.16, the
shim `xgen-ontology`). Members of the PlateerLab organization commit there directly and cut
releases there (a GitHub Release runs [`publish.yml`](.github/workflows/publish.yml)).

[`jinsoo96/js-ontology-build`](https://github.com/jinsoo96/js-ontology-build) is a read-only
mirror. Every push to `main` and every release tag on the origin is pushed on to it at once by
[`mirror.yml`](.github/workflows/mirror.yml), with a deploy key that can write only the mirror
(secret `MIRROR_DEPLOY_KEY`). The mirror's own
[`sync-from-xgen-ontology-build.yml`](.github/workflows/sync-from-xgen-ontology-build.yml)
fast-forwards once a day and on manual dispatch as a safety net (it pushes with the mirror's
`SYNC_TOKEN`, since a repository's own `GITHUB_TOKEN` cannot push a commit that touches
`.github/workflows/`). The origin never pulls from the mirror.

## License

Source-available. Copyright (c) 2026 Jinsoo Kim.

Members of the PlateerLab organization may use, modify and ship it as part of Plateer products
(LICENSE §4). For anyone else, reading and citing are fine; any other use needs written
permission. Releases 0.1.0 through 0.3.0 stay MIT for that specific code. See [`LICENSE`](LICENSE).
