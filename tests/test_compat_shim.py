"""The former import name keeps working through the compatibility package (compat/)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "compat"


def _run(code: str) -> str:
    return subprocess.run([sys.executable, "-W", "error::DeprecationWarning:xgen_ontology", "-c", code],
                          check=False, capture_output=True, text=True, encoding="utf-8",
                          env={"PYTHONPATH": f"{SHIM}{';' if sys.platform == 'win32' else ':'}{ROOT / 'src'}",
                               "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": __import__('os').environ.get("SYSTEMROOT", "")}).stderr


def test_old_name_warns_and_aliases_the_old_module_paths():
    err = _run("import xgen_ontology")
    assert "xgen_ontology is the former name of xgen_ontology_build" in err     # a DeprecationWarning, raised here
    code = (
        "import warnings; warnings.simplefilter('ignore')\n"
        "import xgen_ontology, xgen_ontology_build\n"
        "from xgen_ontology import OntologyBuilder, build_from_csv\n"
        "from xgen_ontology.knowledge import KnowledgeBundle\n"
        "from xgen_ontology.build.deterministic import extract_chunk\n"
        "from xgen_ontology.build import pipeline\n"
        "from xgen_ontology.backends.postgres import PgGraph\n"
        "from xgen_ontology.korean import is_name_shape\n"
        "assert OntologyBuilder is xgen_ontology_build.OntologyBuilder\n"
        "assert extract_chunk is xgen_ontology_build.extract.deterministic.extract_chunk\n"
        "assert pipeline is xgen_ontology_build.pipeline and PgGraph is xgen_ontology_build.PgGraph\n"
        "assert xgen_ontology.__version__ == xgen_ontology_build.__version__\n"
        "assert build_from_csv({'t': 'id,name\\n1,A\\n2,B'}).stats()['instances'] == 2\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, encoding="utf-8",
                         env={"PYTHONPATH": f"{SHIM}{';' if sys.platform == 'win32' else ':'}{ROOT / 'src'}",
                              "PYTHONIOENCODING": "utf-8",
                              "SYSTEMROOT": __import__('os').environ.get("SYSTEMROOT", "")}).stdout
    assert out.strip() == "ok"
