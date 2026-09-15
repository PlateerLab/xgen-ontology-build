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

# an LLM is optional: mode="enrich" adds relations between the extracted entities,
# mode="llm" is full LLM extraction of schema and instances
llm = CallableLLM(lambda p, system="": my_model(system, p))     # OpenAI / Anthropic / vLLM / …
onto = build_from_text("Rule A applies to Acme Bank since 2020. ...", llm=llm, mode="enrich")
```

## Two halves of the lifecycle

### Build — documents/tables → a clean graph

The pipeline is a sequence of independently-importable, backend-agnostic stages:

| Stage | What it does |
|------|--------------|
| **parse** | extract text from files — txt/md/html/csv built-in (zero-dep), pdf/docx/xlsx via `[files]` |
| **chunk** | boundary-aware chunking (paragraph→sentence→char) with overlap, stable chunk ids for provenance |
| **tabular** | table files → ontology with **no LLM**: table→Class, FK→ObjectProperty (same-name / normalized-name / value-overlap detection), column→DataProperty, dimension rows→instances (the name column is judged from the data; rows with the same name are told apart by their key); large fact/junction tables and tables with no name column stay schema-only |
| **deterministic** | documents → ontology with **no LLM** (`mode="basic"`, the default): HTML tables, whitespace row dumps and pipe grids become row entities typed by the subject column's header, with value cells as attributes and entity cells as relations (headers carry across chunk boundaries, unit rows annotate columns, codes/glosses/decorative parentheses are stripped from names); prose yields noun-phrase entities and (entity, attribute, value) facts; a shared head noun promotes to a class; names appearing in more than 30% of chunks are dropped as non-discriminative |
| **extract** | the LLM paths: `mode="enrich"` asks only for relations between the entities the base build found (choosing from the predicates already in use); `mode="llm"` is full extraction of schema *and* instances per chunk batch, tagged to source chunks. Junk (base64/degenerate) filtered first; unit-notation magnitudes verified against the source; structured-output JSON schema available |
| **taxonomy** | hierarchy **without an LLM**: Hearst patterns in prose ("X, Y 등의 Z" → Z is-a X, Z is-a Y); then name structure over classes *and* instances: a boundary-aligned head noun is the parent ("상임감사실" → is-a "감사실"; an instance is typed by its head, a name used as a head becomes a class), a code-prefixed spelling folds into its canonical name, a shared leading word links neighbours. Before that, name fragments that never stood alone are folded back into their source name and unlinked everyday words are dropped. Off with `hierarchy=False` |
| **govern** | predicate governance: strip a subject/object noun glued into the predicate, fold surface variants, anchor to the schema and to predicates already in use; vote relation direction by (subject type, object type) majority; merge predicates that share a stem or whose extension is contained in another's |
| **dedup** | merge synonymous names — content-morpheme keys for instances (shortest spelling wins), (domain, range, key) groups for properties, LLM synonym groups (`mode="enrich"` only), embedding cosine clusters when an embedder is given |
| **hierarchy** | keep only genuine is-a edges ("being linked is not being a subclass"), break cycles, repair instances typed by a class of their own name, materialize inherited properties onto subclasses |
| **normalize** | the store-loading rules: a class must look like a class name; a relation named with graph vocabulary (`type`, `instanceOf`, `subClassOf`, `sameAs`) is a typing statement; a relation whose predicate is a declared datatype property or whose object is a value is an attribute; a subject that is a value is dropped; anything referenced is declared (no dangling endpoints) |
| **quality** | a graph-reviewer score: completeness · integrity · grounding · shape, recorded on `onto.report.quality` |
| **resolve** | (off by default) fuzzy entity resolution: fold similar surface forms, *guarding* dates/ids and number-conflicting names |
| **community** | Louvain modularity clustering (pure Python) |
| **emit** | Turtle (zero-dep) or OWL/RDF-XML (rdflib) |

### Keep building: incremental, dictionary, identifiers

```python
from xgen_ontology import OntologyBuilder, TermDictionary, unbuilt_chunks

builder = OntologyBuilder()                      # or mode="enrich" with an llm
onto = builder.build({"2024.md": [...chunks...]})

# later: only the chunks the ontology has not seen are extracted (by chunk id); the
# post-build re-runs over the whole graph, hierarchy induction only where new names can
# reach. In enrich mode the LLM pass covers only chunks it has not asked about yet.
unbuilt_chunks(onto, {"2024.md": [...], "2025.md": [...]})     # what extend() would do
builder.extend(onto, {"2024.md": [...], "2025.md": [...]})

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
`ontology_edges` / `ontology_node_chunks`). `PgGraph` speaks that schema over any
DB-API connection (psycopg 2/3; sqlite in the tests), so a library build lands where
the product reads it, and a stored graph can be searched or extended:

```python
import psycopg
from xgen_ontology import OntologyBuilder, PgGraph

pg = PgGraph(psycopg.connect(DSN, autocommit=True), collection_id="col-1")
pg.ensure_schema()                        # no-op where the product already created the tables
pg.write(OntologyBuilder().build(docs))   # rows identical to the product's own loader

onto = pg.load()                          # back into build models (chunk ids only)
OntologyBuilder().extend(onto, more_docs) # incremental
pg.write(onto, replace=False)             # append; a node seen again merges its attributes

onto.search(...)                          # or GraphRAG(pg, vector_store, llm) straight on the tables
```

Job and session bookkeeping is the application's; `OntologyBuilder(progress=fn)`
reports each stage (`start / tables / extract / enrich / llm / dedup / hierarchy /
finalize / done`) so a job table can be driven from it.

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
    extract.py       # document -> ontology with an LLM (relations-only enrich / full)
    resolve.py       # fuzzy entity resolution (off by default)
    taxonomy.py      # hierarchy without an LLM: Hearst patterns + name structure, fragment folding
    govern.py        # predicate governance, direction vote, stem / co-extension merge
    dedup.py         # rule + LLM + vector dedup
    hierarchy.py     # is-a cleaning, self-typed repair, property inheritance
    finalize.py      # graph normalization (the store-loading rules)
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
every 15 minutes and on manual dispatch, fast-forwarding from the organization repository with
the repository's own `GITHUB_TOKEN`; no personal credential is involved, so there is nothing to
expire.

## License

Source-available. Copyright (c) 2026 Jinsoo Kim.

Members of the PlateerLab organization may use, modify and ship it as part of Plateer products
(LICENSE §4). For anyone else, reading and citing are fine; any other use needs written
permission. Releases 0.1.0 through 0.3.0 stay MIT for that specific code. See [`LICENSE`](LICENSE).
