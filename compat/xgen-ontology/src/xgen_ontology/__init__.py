"""The former name of :mod:`xgen_ontology_build`.

Everything public is re-exported from ``xgen_ontology_build`` and the old module paths are
aliased (``xgen_ontology.knowledge``, ``xgen_ontology.build.deterministic``,
``xgen_ontology.backends.postgres`` ...), so existing imports keep working while callers
move to the new name. The search half of the old toolkit is gone (xgen-omnifuse searches).
This package stops after 0.16.
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

from xgen_ontology_build import *  # noqa: F401,F403
from xgen_ontology_build import __all__, __version__  # noqa: F401

_warnings.warn("xgen_ontology is the former name of xgen_ontology_build; this compatibility package "
               "stops after 0.16. Import xgen_ontology_build.", DeprecationWarning, stacklevel=2)

# old top-level module -> where it lives now
_MODULES = {
    "models": "xgen_ontology_build.models",
    "protocols": "xgen_ontology_build.protocols",
    "llm": "xgen_ontology_build.llm",
    "ontology": "xgen_ontology_build.ontology",
    "facade": "xgen_ontology_build.facade",
    "korean": "xgen_ontology_build.text.korean",
    "text": "xgen_ontology_build.text.tokens",
    "knowledge": "xgen_ontology_build.exchange.knowledge",
    "knowledge_build": "xgen_ontology_build.exchange.knowledge_build",
}
for _old, _new in _MODULES.items():
    _mod = _importlib.import_module(_new)
    _sys.modules[f"{__name__}.{_old}"] = _mod
    globals()[_old] = _mod
