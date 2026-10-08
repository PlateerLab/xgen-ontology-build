# xgen-ontology (compatibility package)

`xgen-ontology` was renamed **`xgen-ontology-build`** in 0.15 and its import name is now
`xgen_ontology_build`, laid out by build stage (`text`, `extract`, `postbuild`, `store`,
`exchange`). This package only installs `xgen-ontology-build` of the same version and
re-exports it under the old name, so

```python
from xgen_ontology import OntologyBuilder                  # still works, with a DeprecationWarning
from xgen_ontology.knowledge import KnowledgeBundle        # old module paths are aliased
from xgen_ontology.build.deterministic import extract_chunk
```

keep working while callers move to

```python
from xgen_ontology_build import OntologyBuilder
from xgen_ontology_build.exchange.knowledge import KnowledgeBundle
from xgen_ontology_build.extract.deterministic import extract_chunk
```

The search half of the old toolkit (`GraphRAG`, `Ontology.search`, the in-memory graph
and vector stores) is not re-exported: it was superseded by xgen-omnifuse. This package
is published alongside 0.15 and 0.16 and then stops.
