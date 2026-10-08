"""``xgen_ontology.build.*`` -> the stage packages of :mod:`xgen_ontology_build`."""
import importlib as _importlib
import sys as _sys

_MODULES = {
    "chunk": "xgen_ontology_build.text.chunk",
    "parse": "xgen_ontology_build.text.parse",
    "dictionary": "xgen_ontology_build.text.dictionary",
    "translate": "xgen_ontology_build.text.translate",
    "deterministic": "xgen_ontology_build.extract.deterministic",
    "tabular": "xgen_ontology_build.extract.tabular",
    "relation_units": "xgen_ontology_build.extract.relation_units",
    "relation_formation": "xgen_ontology_build.extract.relation_formation",
    "extract": "xgen_ontology_build.extract.llm_extract",
    "taxonomy": "xgen_ontology_build.postbuild.taxonomy",
    "dedup": "xgen_ontology_build.postbuild.dedup",
    "hierarchy": "xgen_ontology_build.postbuild.hierarchy",
    "finalize": "xgen_ontology_build.postbuild.finalize",
    "govern": "xgen_ontology_build.postbuild.govern",
    "resolve": "xgen_ontology_build.postbuild.resolve",
    "quality": "xgen_ontology_build.postbuild.quality",
    "community": "xgen_ontology_build.postbuild.community",
    "retract": "xgen_ontology_build.postbuild.retract",
    "pipeline": "xgen_ontology_build.pipeline",
    "emit": "xgen_ontology_build.exchange.emit",
}
for _old, _new in _MODULES.items():
    _mod = _importlib.import_module(_new)
    _sys.modules[f"{__name__}.{_old}"] = _mod
    globals()[_old] = _mod
