# xgen-ontology

**Independent graph building:** [portable hierarchy, embeddings and graph exchange](docs/knowledge.md).

**Backend-agnostic ontology / knowledge-graph toolkit.** Turn documents or tables
into a *clean* knowledge graph — extract, dedup, induce the is-a hierarchy, govern
predicates, score quality — then **search it with one-shot GraphRAG**. **Zero LLM
calls by default** (structure-based extraction, rule post-build), **zero infra** (the
whole thing runs on a pure-Python in-memory backend), **zero lock-in** (load into
*any* SPARQL 1.1 store), **zero hard deps** in the core.

```python
from xgen_ontology import build_from_csv

onto = build_from_csv({                       # no LLM, no DB, no API key
    "products": "product_id,name,color_id\n1,Widget,10\n2,Gadget,20",
    "colors":   "color_id,name\n10,Red\n20,Blue",
})
print(onto.stats())          # {'classes': 2, 'instances': 4, 'relations': 2, ...}
print(onto.to_turtle())      # standards RDF/Turtle
print(onto.search("what color is Widget").answer)
```

A knowledge graph built with xgen-ontology (degree-sized nodes, community clusters),
shown in a graph explorer:

<p align="center">
  <img src="assets/ontology-graph.png" alt="An ontology knowledge graph built with xgen-ontology" width="760">

### Incremental resource graphs

Large file stores should build one immutable resource revision at a time. The
fragment contract keeps graph construction independent from hierarchy and
embedding projections, while assembly retracts deleted revisions without
rebuilding unchanged files.

```python
from xgen_ontology import build_resource_fragment, assemble_resource_fragments

fragment = build_resource_fragment(one_file_bundle, snapshot_id="fragment:file-7:r3")
snapshot = assemble_resource_fragments(
    current_source_bundle,
    [fragment, *unchanged_fragments],
    snapshot_id="graph:42",
)
```
</p>

Documents build **without an LLM too**. The base build reads document *structure*:
tables become row entities with their column values as attributes and relations,
prose yields noun-phrase entities, and the hierarchy comes from Hearst patterns
and from the structure of the names themselves. Mix tables and text freely; raw
documents are **parsed and chunked** for you:

```python
from xgen_ontology import build_from_files, build_from_text, CallableLLM

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
before the first one:

```python
from xgen_ontology import OntologyBuilder, KO_RELATION_PROMPTS

builder = OntologyBuilder(llm, mode="enrich",
                          relation_prompts=KO_RELATION_PROMPTS,   # the product's prompts; English by default
                          char_budget=10000, max_output_tokens=8192, max_workers=4)
onto = builder.build(docs)
onto.report.relation_stats      # units, calls, relations, vocabulary, new_vocabulary, lost_chunks, ...
[(op.name, op.description) for op in onto.concepts.object_properties]   # the vocabulary
```

## Two halves of the lifecycle

### Build — documents/tables → a clean graph

The pipeline is a sequence of independently-importable, backend-agnostic stages:

| Stage | What it does |
|------|--------------|
| **parse** | extract text from files — txt/md/html/csv built-in (zero-dep), pdf/docx/xlsx via `[files]` |
| **chunk** | boundary-aware chunking (paragraph→sentence→char) with overlap, stable chunk ids for provenance |
| **tabular** | table files → ontology with **no LLM**: table→Class, FK→ObjectProperty (same-name / normalized-name / value-overlap detection; serial numbers that merely overlap are not a key, and the direction comes from which side is a primary key), column→DataProperty, dimension rows→instances (the name column is judged from the data; rows with the same name are told apart by their key); large fact/junction tables and tables with no name column stay schema-only. Each sheet of a workbook is a table of its own; HTML tables (row and column spans expanded), CSV and TSV are read; a header row repeated down the table is skipped |
| **deterministic** | documents → ontology with **no LLM** (`mode="basic"`, the default): HTML tables, whitespace row dumps and pipe grids become row entities typed by the subject column's header, with value cells as attributes and entity cells as relations (headers carry across chunk boundaries, unit rows annotate columns, codes/glosses/decorative parentheses are stripped from names); prose yields noun-phrase entities and (entity, attribute, value) facts; a shared head noun promotes to a class; names appearing in more than 30% of chunks are dropped as non-discriminative. A whitespace block only counts as a row dump when its words are cells (nouns, numbers) rather than sentence constituents, judged by morpheme tag, so an article-numbered regulation paragraph never turns into a table of particle-bearing "entities". Record dumps (`key: value` lines) read as one row. Whether a string can be a name is judged by its shape in Unicode categories (it has a letter, it is not a sentence, its brackets are balanced), never by a word list; HTML entities and escaped pipes are decoded first. An entity cell becomes a `relatedTo` relation labelled with its column name, for the enrich pass to name |
| **relation formation** | `mode="enrich"`, the product's relation pass: documents are cut into units of evidence and the nodes each unit mentions are numbered (deterministic); units are batched by an input budget and by the expected size of the answer, learnt as calls come back; relation names come from a vocabulary of English identifiers with definitions, defined first from units sampled across the corpus when the graph has none. An answer is rows of numbers (unit, subject, relation, object); a cut-off answer keeps its complete part and the batch is halved, an empty one is retried, a row that does not hold together is dropped. Names used often enough outside the vocabulary (a natural break in their frequencies) join it; at the end of the build every name is held to the vocabulary, and the rest become `relatedTo` with the old name as label. A chunk no answer came back for is asked again by the next `extend` |
| **extract** | `mode="llm"`, the older path: full extraction of schema *and* instances per chunk batch, tagged to source chunks. Junk (base64/degenerate) filtered first; unit-notation magnitudes verified against the source; structured-output JSON schema available |
| **taxonomy** | hierarchy **without an LLM**: Hearst patterns in prose ("X, Y 등의 Z" → Z is-a X, Z is-a Y), read from the sentence: an anchor inside a quoted title is skipped, and only a hypernym that heads its own phrase counts (after "… etc. work-*with* unrelated sites" the list is about the sites, not the work). Then name structure over classes *and* instances: a boundary-aligned head noun is the parent ("상임감사실" → is-a "감사실"; an instance is typed by its head, a name used as a head becomes a class), a code-prefixed spelling folds into its canonical name, a shared leading word links neighbours. A head whose compounds are spread over the corpus ("whether", "matter", "standard" head hundreds of names in every document) is not a type: its names become `relatedTo` neighbours instead, by the same discriminativeness ratio that filters entities. Names an earlier step had already typed by that head count toward its spread and are retyped as neighbours too; a table's own class is left as it is. Before that, name fragments that never stood alone are folded back into their source name and unlinked everyday words are dropped. Off with `hierarchy=False` |
| **govern** | predicate governance for `mode="llm"`: strip a subject/object noun glued into the predicate, fold surface variants, anchor to the schema and to predicates already in use; vote relation direction by (subject type, object type) majority. `merge_predicates` is deprecated: relation names are held to a vocabulary instead |
| **dedup** | merge synonymous names — content-morpheme keys for instances (shortest spelling wins), LLM class synonym groups (`mode="enrich"` only), embedding cosine clusters of class names when an embedder is given. Relation names are not merged here |
| **hierarchy** | keep only genuine is-a edges ("being linked is not being a subclass"), break cycles, repair instances typed by a class of their own name, materialize inherited properties onto subclasses |
| **normalize** | the store-loading rules: a class must look like a class name; a relation named with graph vocabulary (`type`, `instanceOf`, `subClassOf`, `sameAs`) is a typing statement; a relation whose predicate is a declared datatype property or whose object is a value is an attribute; a subject that is a value is dropped. A relation name is an English identifier: an ASCII name is camelCased, anything else becomes `relatedTo` with the original as its label. An endpoint is an existing instance or a declared class, else a new instance only when it has the shape of an entity; the same relation stated twice is one relation whose sources are the union and whose weight is the number of chunks that state it; a pair that has a named relation drops its `relatedTo` |
| **retract** | a deleted document leaves as a delta, not a rebuild: its chunks' links go, then whatever they were the only evidence for. An individual with no chunk left; a class with no chunk left that nothing refers to any more (its own declarations do not count), evaluated to a fixpoint so a class whose only individual or subclass went goes too; the properties only those classes declared; a relation whose sources (the chunks that state it) are all gone, keeping the rest when some survive; a relation with no recorded sources goes when its two ends no longer share a chunk (links made from name structure, the `relatedTo` neighbour and `sameAs`, are not chunk evidence and stay). Same rules on the build models (`retract_chunks`) and on the graph tables (`PgGraph.prune_chunks`); the post-build then runs over the surviving corpus |
| **quality** | a graph-reviewer score: completeness · integrity · grounding · shape, recorded on `onto.report.quality` |
| **resolve** | (off by default) fuzzy entity resolution: fold similar surface forms, *guarding* dates/ids and number-conflicting names |
| **community** | Louvain modularity clustering (pure Python) |
| **emit** | Turtle (zero-dep) or OWL/RDF-XML (rdflib) |

### Keep building: incremental, dictionary, identifiers

```python
from xgen_ontology import OntologyBuilder, TermDictionary, removed_chunks, unbuilt_chunks

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

# a term dictionary (acronym -> full form, house spelling -> official one) applies at
# build time, at query time and at indexing time
d = TermDictionary("finance")
d.bulk_import([{"alias": "DSR", "canonical": "총부채원리금상환비율"}])
OntologyBuilder(dictionary=d).build(docs)                    # aliases canonicalized in the graph
d.normalize_query("DSR 60% 고객")                             # -> "DSR 60% 고객 총부채원리금상환비율"
d.normalize_text("고객의 DSR 50%")                            # -> "고객의 DSR(총부채원리금상환비율) 50%"

# English IRI local names for a non-ASCII ontology, translated once per name and cached
onto.translate(llm)          # classes UpperCamelCase, properties lowerCamelCase
onto.to_turtle()             # :CreditRating a owl:Class ; rdfs:label "신용등급"
```

### Search — one-shot GraphRAG

Not an iterative ReAct loop — fire several retrieval strategies at once and **fuse**:

1. vector / lexical passages (what it says)
2. graph label-linking → 1-hop relations (how entities connect)
3. **class enumeration** — the complete "list/count" a vector index can't give
4. HippoRAG: entities of the retrieved chunks → 1-hop expansion
5. evidence assembled with **MMR diversity** + **adaptive top-k** (the decisive
   minority — a warning/exception — survives instead of being crowded out)
6. one LLM synthesis; honest `evidence_nodes` = only the nodes the answer cites

```python
res = onto.search("which regulation applies to Acme Bank", llm=llm)
res.answer          # the synthesis
res.relations       # graph relations used
res.evidence_nodes  # nodes the answer actually cites (honest highlight)
```

## Any graph DB, or none

The algorithms only ever talk to small protocols (`GraphStore`, `VectorStore`,
`LLM`, `GraphSink`, `Morphology`, `Embedder`), never to a database:

```python
# zero infra — pure-Python in-memory (default)
onto.search("…")

# load into any SPARQL 1.1 store (Fuseki, GraphDB, Blazegraph, Virtuoso, …)
from xgen_ontology import fuseki
store = fuseki("http://localhost:3030", "ds", user="admin", password="…")
onto.push(store, graph="urn:my-graph")           # write
onto.search("…")                                  # or search a remote store via SparqlGraph
```

`SparqlGraph` is stdlib-only (urllib) and uses portable `FILTER(CONTAINS(...))`, so
it works on **any** SPARQL 1.1 endpoint — not just jena-text.

The production system keeps its graph in PostgreSQL (`ontology_nodes` /
`ontology_edges` / `ontology_node_chunks`, with `ontology_edge_sources` for the chunks
that state each relation and `ontology_schema` for the relation vocabulary). `PgGraph`
speaks that schema over any DB-API connection (psycopg 2/3; sqlite in the tests), so a
library build lands where the product reads it, and a stored graph can be searched or
extended. An edge carries its weight and its label (the column name or the document's
own word for the relation); an appending write renames the legacy `관련` edges to
`relatedTo` and drops a `relatedTo` edge whose pair now has a named relation:

```python
import psycopg
from xgen_ontology import OntologyBuilder, PgGraph

pg = PgGraph(psycopg.connect(DSN, autocommit=True), collection_id="col-1")
pg.ensure_schema()                        # no-op where the product already created the tables
pg.write(OntologyBuilder().build(docs))   # rows identical to the product's own loader

onto = pg.load()                          # back into build models (chunk ids only)
OntologyBuilder().extend(onto, more_docs) # incremental
pg.write(onto, replace=False)             # append; a node seen again merges its attributes
pg.prune_chunks(deleted_chunk_ids)        # a deleted document, applied as a delta on the tables
                                          # (a relation goes with the last chunk that states it)

onto.search(...)                          # or GraphRAG(pg, vector_store, llm) straight on the tables
```

Job and session bookkeeping is the application's; `OntologyBuilder(progress=fn)`
reports each stage (`start / retract / tables / extract / enrich / llm / dedup /
hierarchy / finalize / done`) so a job table can be driven from it.

## Install

```bash
pip install xgen-ontology                 # core, zero deps
pip install "xgen-ontology[files]"        # + pypdf / python-docx / openpyxl (parse pdf/docx/xlsx)
pip install "xgen-ontology[rdf]"          # + rdflib (OWL / RDF-XML emit & parse)
pip install "xgen-ontology[korean]"       # + kiwipiepy (Korean morphological dedup)
pip install "xgen-ontology[vector]"       # + qdrant-client (embedding adapters)
pip install "xgen-ontology[postgres]"     # + psycopg (PgGraph takes any DB-API connection)
```

Run the demos with no install:

```bash
python examples/build_csv.py
python examples/build_documents.py
python examples/build_and_search.py
```

## Design — algorithms as a library

- **`dependencies = []`** — the core needs nothing but the standard library. The
  in-memory graph indexes labels with **BM25** (CJK character n-grams, so Korean/CJK
  search works with no morphological analyzer); the Turtle writer is hand-rolled.
- **No LLM in the loop unless you ask for one** — the base build and the whole
  post-build are rules over document structure and name structure, so a build is
  reproducible and costs nothing; `mode="enrich"` / `"llm"` bring a model in for the
  parts only a model can do (relations stated in prose, free-form schema).
- **English-neutral architecture, Korean-tuned defaults** — the morphology-aware parts
  (noun phrases, sentence-fragment detection, head nouns, common words) use the optional
  `korean` extra and degrade gracefully without it; prompts, name→URI translation and
  the tokenizer are pluggable.
- **Bring your own everything** — LLM (`generate(prompt, system)`, optionally
  `generate_json(prompt, system, schema)` for structured output), embedder, morphology,
  graph store. The bundled `EchoLLM` lets search run with no API key.

```
src/xgen_ontology/
  models.py        # Class/Property/Concepts (T-Box), Instance/Relation/DataValue (A-Box), Node/Chunk
  protocols.py     # LLM / GraphStore / VectorStore / GraphSink / Morphology / Embedder
  text.py          # tokenizer + BM25 (CJK n-grams), IRI-safe slugging
  korean.py        # label cleanup + morphology (optional kiwipiepy; degrades gracefully)
  build/
    parse.py         # file -> text (txt/md/html/csv; pdf/docx/xlsx optional)
    chunk.py         # boundary-aware chunking
    tabular.py       # table file -> ontology (no LLM)
    deterministic.py # document -> ontology from structure (no LLM): tables, row dumps, prose
    relation_units.py     # units of evidence and the numbered mentions in them (no LLM)
    relation_formation.py # enrich: relations named unit by unit from one vocabulary
    extract.py       # document -> ontology with an LLM (full extraction; enrich delegates)
    resolve.py       # fuzzy entity resolution (off by default)
    taxonomy.py      # hierarchy without an LLM: Hearst patterns + name structure, fragment folding
    govern.py        # predicate governance and direction vote (mode="llm")
    dedup.py         # rule + LLM + vector dedup of instances and classes
    hierarchy.py     # is-a cleaning, self-typed repair, property inheritance
    finalize.py      # graph normalization (the store-loading rules)
    retract.py       # deleted chunks out of the graph as a delta (evidence, orphans, structural links)
    dictionary.py    # TermDictionary: alias -> canonical at build / query / indexing time
    translate.py     # LLM translation of names to English IRI local names
    quality.py     # graph-reviewer score
    community.py   # Louvain
    emit.py        # Turtle / OWL
    pipeline.py    # OntologyBuilder (wires the stages)
  backends/
    memory.py      # InMemoryGraph / InMemoryVector / InMemoryGraphSink (zero infra)
    sparql.py      # SparqlGraph — any SPARQL 1.1 store (read + write)
    postgres.py    # PgGraph — the XGEN graph tables: write, search, load for incremental builds
  search/          # fusion + one-shot GraphRAG
  ontology.py      # Ontology — the hub (search / emit / push / quality / communities)
  facade.py        # build_from_csv / build_from_documents / build_from_triples
examples/  tests/
```

## Roadmap

- async pipeline (parallel chunk extraction + parallel search seeds)
- Neo4j / property-graph `GraphStore` adapter; Qdrant `VectorStore` adapter
- RDF-star / qualified statements (n-ary relations, provenance) in emit
- reranker / cross-encoder hook for search

## Repository model

[`PlateerLab/xgen-ontology-build`](https://github.com/PlateerLab/xgen-ontology-build) is the
source of truth and publishes the Python package `xgen-ontology`. Members of the PlateerLab
organization commit there directly and cut releases there.

[`jinsoo96/js-ontology-build`](https://github.com/jinsoo96/js-ontology-build) is a read-only
mirror. It runs [`sync-from-xgen-ontology-build.yml`](.github/workflows/sync-from-xgen-ontology-build.yml)
every 15 minutes and on manual dispatch, fast-forwarding from the organization repository. The
push uses the mirror's `SYNC_TOKEN` secret (the mirror owner's personal access token, `repo` +
`workflow` scopes) because GitHub does not let a repository's own `GITHUB_TOKEN` push a commit
that touches `.github/workflows/`; with no secret it falls back to `GITHUB_TOKEN`, which is
enough until the origin next changes its CI. The organization side has no such dependency: the
origin never pulls from the mirror.

## License

Source-available. Copyright (c) 2026 Jinsoo Kim.

Members of the PlateerLab organization may use, modify and ship it as part of Plateer products
(LICENSE §4). For anyone else, reading and citing are fine; any other use needs written
permission. Releases 0.1.0 through 0.3.0 stay MIT for that specific code. See [`LICENSE`](LICENSE).
