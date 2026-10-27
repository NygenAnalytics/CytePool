"""Where shared code lives.

A name with a leading underscore belongs to its module. Code that two or more modules
share lives in a private module (one whose name starts with an underscore), under
names without one, and code that one module uses stays in that module. The names
listed below are the exceptions, each with the reason it stays where it is.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "cytepool"

_DIGESTS = "cube files read and set the digests that a Cube keeps in its own state"
_ENCODING = (
    "the encoding that cytepool.cube.fingerprints documents, computed next to its "
    "documentation"
)
_CHECKS = (
    "parts of Design.check, which build the public DesignIssue of design.checks with "
    "its wording and tolerances"
)
_VARIABLES = (
    "the reading of a design's variables and the building of the public ModelMatrix, "
    "which model_matrix, Design and Design.check share"
)
_F_DISTRIBUTION = (
    "compiled with the public norm_ppf_scalar and norm_isf_scalar of distributions, "
    "which they call"
)
_QUALITY_WEIGHTS = (
    "the REML of sample quality weights raises _SingularInformation, the type of an "
    "error that sample_quality_weights gives its callers"
)
_VOOM_RESULT = "builds the public VoomResult of cytepool.stats.voom"
_SUMRANK = (
    "parts of the public fit_sumrank of cytepool.stats.sumrank, which the analyses of "
    "several studies run on their own statistics and permutations"
)
_VARIANCE_FRACTIONS = (
    "parts of the public variance_fractions of cytepool.stats.variance_fractions, "
    "which variance_partitioning runs on each group of a cube"
)

# (module, name): why other modules of the package import it from that module.
SHARED_FROM_THEIR_MODULE = {
    ("cytepool.cube.cube", "_digest_of_read_group"): _DIGESTS,
    ("cytepool.cube.cube", "_digest_record_of"): _DIGESTS,
    ("cytepool.cube.cube", "_take_stored_digests"): _DIGESTS,
    ("cytepool.cube.fingerprints", "_block_digest"): _ENCODING,
    ("cytepool.cube.fingerprints", "_cells_digest"): _ENCODING,
    ("cytepool.cube.fingerprints", "_check_record"): _ENCODING,
    ("cytepool.cube.fingerprints", "_column_type"): _ENCODING,
    ("cytepool.cube.fingerprints", "_digest_record"): _ENCODING,
    ("cytepool.cube.fingerprints", "_layer_type"): _ENCODING,
    ("cytepool.cube.fingerprints", "_normalization_digest"): _ENCODING,
    ("cytepool.cube.fingerprints", "_table_digest"): _ENCODING,
    ("cytepool.design.checks", "_aliasing"): _CHECKS,
    ("cytepool.design.checks", "_contrast_violations"): _CHECKS,
    ("cytepool.design.checks", "_random_checks"): _CHECKS,
    ("cytepool.design.checks", "_relation"): _CHECKS,
    ("cytepool.design.checks", "_same_partition"): _CHECKS,
    ("cytepool.design.model_matrix", "_Covariate"): _VARIABLES,
    ("cytepool.design.model_matrix", "_Factor"): _VARIABLES,
    ("cytepool.design.model_matrix", "_build_model_matrix"): _VARIABLES,
    ("cytepool.design.model_matrix", "_describe_rows"): _VARIABLES,
    ("cytepool.design.model_matrix", "_normalize_levels"): _VARIABLES,
    ("cytepool.design.model_matrix", "_read_samples"): _VARIABLES,
    ("cytepool.stats.distributions", "_f_log_x_density"): _F_DISTRIBUTION,
    ("cytepool.stats.distributions", "_f_tail_quantile"): _F_DISTRIBUTION,
    ("cytepool.stats.distributions", "_f_tails"): _F_DISTRIBUTION,
    ("cytepool.stats.dream", "_dream_weights"): _VOOM_RESULT,
    ("cytepool.stats.dream", "_moderated_tests"): (
        "builds the public ModeratedFixedEffectTests of cytepool.stats.dream from the "
        "parts of tests that dream_voom chooses feature by feature"
    ),
    ("cytepool.stats.meta_analysis", "_checked_options"): (
        "the options of the public fit_meta_analysis, which meta_analysis checks "
        "alike before it reads the studies"
    ),
    ("cytepool.stats.mixed_models", "_fit"): (
        "builds the public MixedModelFit of cytepool.stats.mixed_models"
    ),
    ("cytepool.stats.sample_weights", "_SingularInformation"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_VarianceModel"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_estimate"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_prior_weights"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_too_few_df_message"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_variance_model"): _QUALITY_WEIGHTS,
    ("cytepool.stats.sample_weights", "_weights_of"): _QUALITY_WEIGHTS,
    ("cytepool.stats.smoothing", "_quantile_knots"): (
        "the knots of the public natural_spline_basis, placed as R's ns() places them"
    ),
    ("cytepool.stats.smoothing", "_smoothing_spline_fit"): (
        "R's smooth.spline, built from the B-splines of the public "
        "natural_spline_basis next to them, with which multiple_testing smooths "
        "qvalue's estimates of pi0"
    ),
    ("cytepool.stats.smoothing", "_weighted_lowess"): (
        "tests and validation/empirical_bayes.py import it from smoothing"
    ),
    ("cytepool.stats.sumrank", "_NullCounts"): _SUMRANK,
    ("cytepool.stats.sumrank", "_capped_sums"): _SUMRANK,
    ("cytepool.stats.sumrank", "_studies_summed"): _SUMRANK,
    ("cytepool.stats.variance_fractions", "_Terms"): _VARIANCE_FRACTIONS,
    ("cytepool.stats.variance_fractions", "_checked_terms"): _VARIANCE_FRACTIONS,
    ("cytepool.stats.variance_fractions", "_linear_fractions"): _VARIANCE_FRACTIONS,
    ("cytepool.stats.variance_fractions", "_mixed_fractions"): _VARIANCE_FRACTIONS,
    ("cytepool.stats.voom", "_quality_weighted_voom"): _VOOM_RESULT,
}


def _modules() -> dict[str, Path]:
    """Every module of the package, by its import name."""
    modules = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def _imports() -> list[tuple[str, str, str]]:
    """(importing module, imported module, name) for every name that a module of the
    package imports from another module of the package."""
    modules = _modules()
    found = []
    for module, path in modules.items():
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            if source in modules and source != module:
                found.extend((module, source, alias.name) for alias in node.names)
    return found


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def test_modules_take_no_private_name_from_another_module_but_the_listed_ones() -> None:
    taken = {(source, name) for _, source, name in _imports() if _is_private(name)}
    assert sorted(taken - set(SHARED_FROM_THEIR_MODULE)) == []
    # Every name listed is still taken by another module.
    assert sorted(set(SHARED_FROM_THEIR_MODULE) - taken) == []


def test_every_private_module_is_shared_by_at_least_two_modules() -> None:
    importers: dict[str, set[str]] = {}
    for module, source, _ in _imports():
        importers.setdefault(source, set()).add(module)
    private = [
        module
        for module in _modules()
        if _is_private(module.rpartition(".")[2]) and module != "cytepool._version"
    ]
    assert private
    assert [module for module in private if len(importers.get(module, ())) < 2] == []
