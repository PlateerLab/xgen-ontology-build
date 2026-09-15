# Independent graph building and knowledge/v1 exchange

`xgen-ontology` builds graphs. The new `build_knowledge` API accepts explicit source
identities and returns portable data for any consumer, including `xgen-omnifuse`.
It does not require or import OmniFuse, Xgen, a database or a server.

```python
from xgen_ontology import build_knowledge
from xgen_ontology.knowledge import KnowledgeBundle, Resource, SourceChunk

source = KnowledgeBundle(
    corpus_id="company", snapshot_id="source-v1",
    resources=(
        Resource("root", "v1", "policies", "directory"),
        Resource("file-17", "v1", "colors.csv", parent_id="root"),
    ),
    chunks=(SourceChunk(
        "file-17:v1:chunk-0", "file-17", "v1", "csv-parser-v1",
        "id,name\n1,Red\n2,Blue",
    ),),
)
result = build_knowledge(source, snapshot_id="graph-v1")
result.dump("knowledge.json")
```

`examples/knowledge_build.py` is a runnable build-only example. In a **different**
environment, a consumer can load the JSON without this package installed. The
producer can also export to another database or graph product using the same records.

## Three independent data components

| Component | Contract records | Semantics |
| --- | --- | --- |
| Directory hierarchy | `Resource.id`, `parent_id`, kind, name, revision | Stable source identity; names are display data. Parent is a directory; cycles and dangling parents are rejected. |
| File embeddings | `EmbeddingProfile`, `ChunkEmbedding` | Embedding per source chunk; model/revision/preprocessing/dimension/metric are explicit. Multiple profiles may coexist. |
| Graph | `GraphEntity`, `GraphFact`, `Evidence` | Entity IDs differ from chunk IDs; typed literals differ from entity references; facts carry source support. |

`SourceChunk` joins these components through immutable resource revision, extraction
ID and source locator. A chunk can be an entire file for small inputs or one parsed
passage/page/row for larger inputs. The library accepts caller-owned parsed chunks;
it does not reparse or rechunk them. Existing file parsers remain available separately.

`components` explicitly declares presence, including an empty completed graph.
Embeddings and hierarchy are preserved through graph construction, not recomputed.
To attach embeddings, include profiles and vectors in the input and declare the
`embeddings` component. Vector dimension, finite values and profile references are
validated. A zero cosine vector is rejected.

The same structure represents non-file knowledge: use resource kind `document` or
`record`, omit `parent_id` and `hierarchy` when not meaningful, and supply a source
locator such as a JSON pointer. This version uses source chunks as evidence anchors;
structured rows can use an empty text plus their row locator.

## Identity and evidence

- Resource IDs are supplied by the host and remain stable across rename/move.
- Resource revisions refer to immutable bytes. Supply `content_digest` when available;
  it must describe the original source, not an invented hash of parsed content.
- Chunk IDs must change when text/revision/extraction output changes. Ordinal is
  display order, not the identifier. Extraction identity is interpreted together
  with resource and revision.
- Builder routing uses resource identity plus filename; equal filenames in different
  folders do not overwrite each other's input. The existing builder's semantic
  entity resolution policy is retained. The exchange does not claim that two equal
  labels from arbitrary external graphs represent the same entity.
- Graph IDs are corpus- and kind-qualified. Explicit externally created entities can
  carry entirely different IDs. Labels never serve as foreign keys in the contract.
- Facts include source evidence. Statements derived from endpoint support are marked
  `inferred`; unsupported facts are omitted with diagnostics, not presented as proven.
- Exact caller locators are preserved. `{}` means whole chunk/unspecified location,
  not a fabricated page or quote. Text offsets are Unicode code points `[start,end)`;
  pages are one-based; table row/column indexes are zero-based; media uses milliseconds.

`export_knowledge(ontology, source, snapshot_id=...)` adapts an already-built ontology
using an explicit source snapshot. Unknown chunk IDs or changed chunk text are
rejected. Build schema, report, source digest and unsupported-fact diagnostics appear
under `extensions["xgen_ontology.build"]`.

## Updated build engine and existing APIs

This entry point uses the current 0.8 production pipeline: structural extraction,
post-build normalization, dictionary, hierarchy, optional enrichment, and injected
morphology/embedding/model providers. Pass `builder=OntologyBuilder(...)` to configure
it. `enrich`/`llm` on the new API requires an LLM; it never silently reports basic
mode as requested enrichment.

The legacy `Ontology.search`/`GraphRAG` entry points remain available with a
DeprecationWarning. They are loaded only on use, so graph building does not import
a retrieval engine. New integrations use the separate retrieval library. Existing
Turtle/OWL/PostgreSQL APIs continue to work; a Turtle export alone is not the complete
versioned knowledge exchange artifact.

## Modification, deletion, cancellation

`build_knowledge` builds a **complete replacement** from the current source corpus.
Pass the updated resource revisions/chunks/embeddings; omit removed sources. The old
additive `extend()` is not a safe deletion primitive and is not used for this path.
Rebuilding naturally retracts unsupported facts while retaining knowledge supported
by surviving input. This first version prioritizes correctness over delta efficiency.

Building returns data only. The host stages and validates it, then publishes all
three components with a compare-and-swap against the expected current snapshot.
A late build must never overwrite a newer source state. The OmniFuse reference
provider supplies such a CAS operation for offline use; application databases must
implement their own atomic publication and durable job handling.

`OperationContext` from `xgen_ontology.knowledge` supplies cancellation/deadline checks.
The new entry point checks before/after each pipeline stage without swallowing
cancellation as a progress callback error. It copies the supplied builder's settings
rather than attaching mutable callbacks to the caller's builder. A blocking external
SDK still needs its own timeout; interruption is cooperative between stages/calls.

## Contract governance and validation

The canonical source is `contracts/knowledge/v1/records.py` with the semantic rules
in this document. The generator produces a language-neutral JSON Schema and a
version/digest lock, then vendors a standalone record implementation into both wheels:

```bash
python tools/sync_knowledge_contract.py --consumer ../xgen-omnifuse
python tools/sync_knowledge_contract.py --check --consumer ../xgen-omnifuse
```

Never manually edit the consumer copy. Both packages test their embedded code/schema
hashes; producer tests also compare the canonical source. The consumer tests an actual
producer-generated fixture. Cross-process wheel tests verify that no counterpart
library is installed in either environment.

The structural schema is complemented by zero-dependency semantic checks: references,
source revisions, hierarchy cycles, duplicate IDs, vectors and typed graph objects.
Custom optional data belongs in namespaced `extensions` and round-trips unchanged.
Unsupported contract versions are rejected. `1.0` is the initial supported version;
future compatibility ranges must be added deliberately with conformance fixtures.

JSON exchange uses a content checksum and atomic file replacement. Checksums detect
corruption, not authenticity. No pickle or runtime network schema loading is involved.
This release implements library interfaces and reference adapters; Xgen filestore DB,
job/UI integration and its next search-node rollout remain separate work.
