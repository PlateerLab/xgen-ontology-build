"""``xgen_ontology.backends.*`` -> :mod:`xgen_ontology_build.store`."""
import importlib as _importlib
import sys as _sys

from xgen_ontology_build.store.memory import InMemoryGraphSink  # noqa: F401
from xgen_ontology_build.store.postgres import PgGraph, edge_source_rows, graph_rows  # noqa: F401
from xgen_ontology_build.store.sparql import SparqlGraph  # noqa: F401

for _old, _new in {"postgres": "xgen_ontology_build.store.postgres", "sparql": "xgen_ontology_build.store.sparql",
                   "memory": "xgen_ontology_build.store.memory"}.items():
    _mod = _importlib.import_module(_new)
    _sys.modules[f"{__name__}.{_old}"] = _mod
    globals()[_old] = _mod
