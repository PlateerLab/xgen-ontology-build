# xgen-ontology

**Backend-agnostic ontology / knowledge-graph toolkit.** Turn documents or tables
into a *clean* knowledge graph — extract, resolve, dedup, induce the is-a hierarchy,
govern predicates, score quality — then **search it with one-shot GraphRAG**.
**Zero infra** (the whole thing runs on a pure-Python in-memory backend), **zero
lock-in** (load into *any* SPARQL 1.1 store), **zero hard deps** in the core.

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

Build from prose with any LLM, mix tables and text freely — raw documents are
**parsed and chunked** for you:

```python
from xgen_ontology import build_from_files, build_from_text, CallableLLM

llm = CallableLLM(lambda p, system="": my_model(system, p))     # OpenAI / Anthropic / vLLM / …

# from files on disk (txt/md/html/csv built-in; pdf/docx/xlsx via the [files] extra)
onto = build_from_files(["policy.pdf", "products.csv"], llm=llm)

# or from a single raw string (auto boundary-aware chunking)
onto = build_from_text("Rule A applies to Acme Bank since 2020. ...", llm=llm)
```

## Two halves of the lifecycle

### Build — documents/tables → a clean graph

The pipeline is a sequence of independently-importable, backend-agnostic stages:

| Stage | What it does |
|------|--------------|
| **parse** | extract text from files — txt/md/html/csv built-in (zero-dep), pdf/docx/xlsx via `[files]` |
| **chunk** | boundary-aware chunking (paragraph→sentence→char) with overlap, stable chunk ids for provenance |
| **tabular** | table → ontology with **no LLM**: table→Class, FK→ObjectProperty (same-name / normalized-name / value-overlap detection), column→DataProperty, dimension rows→instances; large fact/junction tables stay schema-only |
| **extract** | one LLM call per chunk batch → schema *and* instances, tagged to source chunks; junk (base64/degenerate) filtered first |
| **resolve** | entity resolution: fold case/whitespace/unicode + similar surface forms, *guarding* dates/ids and number-conflicting names |
| **taxonomy** | is-a edges from the text itself, **zero LLM calls**: Hearst patterns ("X, Y 등의 Z" → Z is-a X, Z is-a Y) plus head-noun decomposition of compound class names ("상임감사실" → is-a "감사실"). Off with `hierarchy=False` |
| **govern** | predicate governance: fold surface variants of a relation, anchor to the schema vocabulary |
| **dedup** | merge synonymous classes/properties/instances — rule keys, LLM synonym groups, and embedding cosine clusters |
| **hierarchy** | keep only genuine is-a edges ("being linked is not being a subclass"), break cycles, then SCS context profiles with property inheritance |
| **quality** | a graph-reviewer score: completeness · integrity · grounding · shape |
| **community** | Louvain modularity clustering (pure Python) |
| **emit** | Turtle (zero-dep) or OWL/RDF-XML (rdflib) |

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

## Install

```bash
pip install xgen-ontology                 # core, zero deps
pip install "xgen-ontology[files]"        # + pypdf / python-docx / openpyxl (parse pdf/docx/xlsx)
pip install "xgen-ontology[rdf]"          # + rdflib (OWL / RDF-XML emit & parse)
pip install "xgen-ontology[korean]"       # + kiwipiepy (Korean morphological dedup)
pip install "xgen-ontology[vector]"       # + qdrant-client (embedding adapters)
```

Run the demos with no install:

```bash
python examples/build_csv.py
python examples/build_and_search.py
```

## Design — algorithms as a library

- **`dependencies = []`** — the core needs nothing but the standard library. The
  in-memory graph indexes labels with **BM25** (CJK character n-grams, so Korean/CJK
  search works with no morphological analyzer); the Turtle writer is hand-rolled.
- **English-neutral by default** — no hardcoded language. Korean morphology, name→URI
  translation and the extraction/synthesis prompts are all pluggable; the defaults
  assume nothing about your domain or language.
- **Bring your own everything** — LLM (`generate(prompt, system)`), embedder, morphology,
  graph store. The bundled `EchoLLM` lets the whole pipeline run with no API key.

```
src/xgen_ontology/
  models.py        # Class/Property/Concepts (T-Box), Instance/Relation/DataValue (A-Box), Node/Chunk
  protocols.py     # LLM / GraphStore / VectorStore / GraphSink / Morphology / Embedder
  text.py          # tokenizer + BM25 (CJK n-grams), IRI-safe slugging
  korean.py        # label cleanup + morphology (optional kiwipiepy; degrades gracefully)
  build/
    parse.py       # file -> text (txt/md/html/csv; pdf/docx/xlsx optional)
    chunk.py       # boundary-aware chunking
    tabular.py     # table -> ontology (no LLM)
    extract.py     # document -> ontology (LLM)
    resolve.py     # entity resolution
    taxonomy.py    # is-a induction: Hearst patterns + head-noun decomposition (no LLM)
    govern.py      # predicate governance
    dedup.py       # rule + LLM + vector dedup
    hierarchy.py   # is-a cleaning + SCS inheritance
    quality.py     # graph-reviewer score
    community.py   # Louvain
    emit.py        # Turtle / OWL
    pipeline.py    # OntologyBuilder (wires the stages)
  backends/
    memory.py      # InMemoryGraph / InMemoryVector / InMemoryGraphSink (zero infra)
    sparql.py      # SparqlGraph — any SPARQL 1.1 store (read + write)
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
