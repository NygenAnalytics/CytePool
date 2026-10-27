"""The names an analysis starts from, imported from the top level of the package.

``cytepool.__all__`` must list the names that the package's docstring documents, and
``__version__``; each must be the object that its module implements; and ``import
cytepool`` must load none of the optional libraries, which only the functions that
need them import.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import cytepool

# The module that implements each name of the top level.
MODULES = {
    "cytepool.activity": ("Activities", "mlm", "ulm", "weighted_aggregate"),
    "cytepool.composition": ("CompositionResult", "propeller"),
    "cytepool.cube.collection": ("CubeCollection",),
    "cytepool.cube.cube": ("Cube",),
    "cytepool.cube.pool": ("from_bulk", "from_cells"),
    "cytepool.design.design": ("Design",),
    "cytepool.differential_expression": (
        "DifferentialExpressionResult",
        "GroupFit",
        "dream_voom",
        "limma_trend",
        "limma_voom",
    ),
    "cytepool.gene_set_enrichment": ("gsea",),
    "cytepool.gene_set_tests": ("camera", "fry"),
    "cytepool.gene_sets": ("GeneSets", "read_gmt", "write_gmt"),
    "cytepool.held_out_studies": ("HeldOutPrediction", "held_out_prediction"),
    "cytepool.io.anndata": ("from_anndata", "from_mudata", "to_anndata", "to_mudata"),
    "cytepool.io.cube_files": ("read_cube", "write_cube"),
    "cytepool.meta_analysis_of_studies": ("MetaAnalysisResult", "meta_analysis"),
    "cytepool.networks": ("Network", "collectri", "progeny"),
    "cytepool.over_representation_analysis": ("over_representation",),
    "cytepool.pathway_scores": (
        "PathwayScores",
        "combined_z_scores",
        "gsva",
        "plage",
        "ssgsea",
    ),
    "cytepool.principal_component_analysis": (
        "PrincipalComponents",
        "principal_components",
    ),
    "cytepool.replication_across_studies": ("ReplicationResult", "replication"),
    "cytepool.rotation_gene_set_tests": ("mroast", "roast", "romer"),
    "cytepool.score_tests": ("limma",),
    "cytepool.study_results": ("StudyResults", "analyse_each_study"),
    "cytepool.sumrank_across_studies": ("SumRankResult", "sumrank"),
    "cytepool.table": ("Table",),
    "cytepool.variance_components": (
        "VariancePartitioningResult",
        "variance_partitioning",
    ),
}
# Libraries that some functions use, which importing the package must not load.
OPTIONAL = ("anndata", "matplotlib", "mudata", "pandas")


def test_the_top_level_lists_the_names_of_its_docstring() -> None:
    documented = re.findall(r":(?:class|func):`(\w+)`", cytepool.__doc__ or "")
    assert len(documented) == len(set(documented)) == 56
    assert len(cytepool.__all__) == len(set(cytepool.__all__)) == 57
    assert set(cytepool.__all__) == {*documented, "__version__"}
    assert isinstance(cytepool.__version__, str) and cytepool.__version__


def test_each_name_is_the_object_its_module_implements() -> None:
    listed = [name for names in MODULES.values() for name in names]
    assert sorted(listed) == sorted(set(cytepool.__all__) - {"__version__"})
    for module_name, names in MODULES.items():
        module = importlib.import_module(module_name)
        for name in names:
            found = getattr(cytepool, name)
            assert found is getattr(module, name), name
            assert found.__module__ == module_name, name


def test_importing_the_package_loads_no_optional_library() -> None:
    # A fresh interpreter, started where the package under test is found first.
    code = "\n".join(
        [
            "import sys",
            "import cytepool",
            "print(cytepool.__file__)",
            f"print([name for name in {OPTIONAL!r} if name in sys.modules])",
        ]
    )
    root = Path(cytepool.__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    location, loaded = done.stdout.splitlines()
    assert Path(location).resolve() == Path(cytepool.__file__).resolve()
    assert loaded == "[]"
