"""Every public entry point refuses numpy masked arrays, with one standard message.

numpy drops the mask of a ``numpy.ma.MaskedArray`` when it converts one, so a value
marked as missing would be read as data. The inventory below puts every public
callable in one of three groups:

- ``ENTRY_POINTS``: callables that read array input, with a call that succeeds, each
  array argument, and how that argument marks missing values;
- ``NO_ARRAY_INPUT``: callables whose arguments are names, paths, options or
  objects that were checked when they were built, each with its arguments and the
  reason;
- ``RECORDS``: result containers, which store their fields as given and so cannot
  lose a mask.

The public callables are found from the public names of each module, by the rule
tests/test_documentation.py uses (a module's ``__all__``, or else the functions and
classes it defines): each function; each class, as its constructor; every method
of a class, its CytePool base classes included, that takes arguments and is public
or a protocol such as ``__getitem__``; and the methods of an object that a public
property returns, when its class is CytePool's own but not public, as an indexer
is. A public callable added later fails ``test_every_public_callable_is_in_one_group``
until it is classified. Every array argument of every entry point is then given a
masked array and ``numpy.ma.masked`` in place of its valid value, and must raise the
standard message, which the tests build from its parts.
"""

import dataclasses
import functools
import importlib
import inspect
import itertools
import pkgutil
import re
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numba.core.errors import TypingError
from numpy.testing import assert_array_equal

import cytepool
from cytepool._masked_arrays import reject_masked
from cytepool.activity import ulm
from cytepool.composition import propeller
from cytepool.cube import Cube, CubeCollection, from_bulk, from_cells
from cytepool.design import Design, make_contrasts
from cytepool.differential_expression import dream_voom, limma_voom
from cytepool.gene_sets import GeneSets
from cytepool.gsea import gsea
from cytepool.held_out_prediction import held_out_prediction
from cytepool.io.anndata import to_anndata, to_mudata
from cytepool.io.zarr import open_array, write_array, write_group
from cytepool.meta_analysis import differential_expression_meta_analysis
from cytepool.networks import Network
from cytepool.normalization import cpm, filter_by_expression, log_cpm
from cytepool.over_representation import over_representation
from cytepool.pathway_scores import gsva
from cytepool.replication import differential_expression_replication
from cytepool.stats import distributions, special
from cytepool.stats.deseq2 import deseq2_dispersions, deseq2_wald_test, fit_deseq2
from cytepool.stats.distributions import norm_cdf, t_sf_scalar
from cytepool.stats.empirical_bayes import moderated_statistics, squeeze_variances
from cytepool.stats.gene_set_tests import camera
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.mixed_models import fit_mixed_model, fixed_effect_tests
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.quasi_likelihood import fit_quasi_likelihood
from cytepool.stats.smoothing import lowess
from cytepool.stats.quasi_likelihood import quasi_likelihood_f_test
from cytepool.stats.voom import voom, voom_with_quality_weights
from cytepool.sumrank import differential_expression_sumrank
from cytepool.table import Table
from cytepool.variance_partitioning import variance_partitioning

# anndata and mudata, which the conversions of cubes call, warn about their own use
# of pandas options that pandas 4 removes.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:anndata"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning:mudata"),
]

# The standard message: what was found, why it is refused, and the remedy for how
# the argument marks missing values (None when it accepts no missing values). The
# remedy shows how to apply it to a masked array; numpy.ma.masked needs only the
# first part, and a masked array with nothing masked needs only its data.
REASON = (
    "CytePool does not accept numpy masked arrays, because numpy drops the mask when "
    "it converts one, and the masked entries would be read as data."
)
REMEDIES = {
    None: ("Remove the masked entries, or fill them with valid values, first", ""),
    "NaN": (
        "Mark missing values with NaN instead",
        ", for example with numpy.ma.filled(values.astype(float), numpy.nan)",
    ),
    "None": (
        "Mark missing values with None instead",
        "; values.tolist() gives None for each masked entry",
    ),
    "None or NaN": (
        "Mark missing values with None, or NaN in numbers, instead",
        "; values.tolist() gives None for each masked entry",
    ),
}
NOTHING_MASKED = "No entry is masked: pass numpy.ma.getdata(values) instead."
# What a message says was found.
FOUND = {
    "array": "a numpy masked array",
    "constant": "numpy.ma.masked",
    "unmasked": "a numpy masked array",
    "converted": "an array-like that converts to a numpy masked array",
}


def message(subject: str, found: str = "array", missing: str | None = None) -> str:
    """The standard message, as a pattern that matches it exactly.

    ``found`` is "array" (a masked array with a masked entry), "constant"
    (``numpy.ma.masked``), "unmasked" (a masked array with nothing masked) or
    "converted" (an array-like that converts to a masked array with a masked entry).
    """
    if found == "unmasked":
        remedy = NOTHING_MASKED
    elif found == "constant":
        remedy = REMEDIES[missing][0] + "."
    else:
        remedy = "".join(REMEDIES[missing]) + "."
    return re.escape(f"{subject} is {FOUND[found]}; {REASON} {remedy}") + "$"


# Small valid inputs: 4 samples x 8 genes, and a 2-column design.
COUNTS = np.random.default_rng(1).poisson(np.linspace(5, 200, 8), (4, 8))
DESIGN = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
SAMPLE_TABLE = {
    "sample_id": ["S1", "S2", "S3", "S4"],
    "condition": ["ctrl", "stim", "ctrl", "stim"],
    "donor": ["D1", "D1", "D2", "D2"],
}
LEVELS = {"condition": ["ctrl", "stim"]}
COLUMNS = ("(Intercept)", "conditionstim")
# Statistics of 8 genes by gene id, for gene set enrichment analysis.
RANKED = dict(
    zip([f"g{j}" for j in range(8)], [0.5, -1.0, 2.0, 0.1, 3.0, -0.2, 1.1, 0.4])
)
# Cells for from_cells: 4 cells x 3 features, two samples and two groups.
CELL_COUNTS = np.array([[1, 0, 2], [3, 1, 0], [0, 2, 5], [4, 0, 1]])
CELL_SAMPLES = ["S1", "S1", "S2", "S2"]
CELL_GROUPS = ["B", "T", "B", "T"]
CUBE_AXES = {
    "samples": {"sample_id": ["S1", "S2"]},
    "groups": {"group_id": ["B", "T"]},
    "features": {"feature_id": ["g1", "g2", "g3"]},
}


@dataclass
class Context:
    """Objects that valid calls need, built once for the module."""

    directory: Path
    expression: np.ndarray  # log-CPM of COUNTS, with one missing value
    fit: Any  # a LinearModelFit of the expression on DESIGN
    contrast_fit: Any  # with contrasts and per-feature covariance matrices
    cube: Cube  # pooled from the cells, with every layer
    design: Design
    zarr_array: Any
    writes: Iterator[int] = field(default_factory=itertools.count)

    def new_path(self) -> Path:
        """A path that does not exist yet, for a call that writes."""
        return self.directory / f"write_{next(self.writes)}"


@pytest.fixture(scope="module")
def context(tmp_path_factory: pytest.TempPathFactory) -> Context:
    directory = tmp_path_factory.mktemp("masked_inputs")
    expression = log_cpm(COUNTS)
    expression[0, 0] = np.nan
    fit = fit_linear_model(expression, DESIGN)
    write_array(directory / "array", np.arange(12.0).reshape(3, 4), chunks=(2, 2))
    return Context(
        directory=directory,
        expression=expression,
        fit=fit,
        contrast_fit=contrasts_fit(fit, np.eye(2), names=["first", "second"]),
        cube=from_cells(CELL_COUNTS, samples=CELL_SAMPLES, groups=CELL_GROUPS),
        design=Design(SAMPLE_TABLE, "~ condition"),
        zarr_array=open_array(directory / "array"),
    )


@dataclass(frozen=True)
class Entry:
    """An entry point and a call of it that succeeds.

    Attributes:
        valid: The keyword arguments of the call, from the context.
        arrays: Each array argument, with how it marks missing values: "NaN",
            "None or NaN" (table columns), or None.
        options: The arguments that take a number, a bool or a string.
        objects: The other arguments that take no array, such as a fit or a path.
        instance: For a method, the object it is called on.
        keys: For a mapping argument, the key whose value is replaced; messages
            name it as ``argument['key']``.
        names: How messages name an argument, where not by its parameter name.
    """

    valid: Callable[[Context], dict[str, Any]]
    arrays: Mapping[str, str | None]
    options: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    instance: Callable[[Context], Any] | None = None
    keys: Mapping[str, str] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)


def _fields(value: Any) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}


def _cube_arguments(context: Context) -> dict[str, Any]:
    return {
        "layers": {"counts": np.arange(12, dtype=np.uint32).reshape(2, 2, 3)},
        **CUBE_AXES,
        "n_cells": np.ones((2, 2), dtype=np.int64),
        "normalization": {"method": "library_size", "target_sum": 1e4},
        "provenance": {"source": "test", "values": [1, 2]},
    }


def _collection_arguments(context: Context) -> dict[str, Any]:
    """Two studies' cubes, with a donor column, and a value for every option."""
    cube = Cube(
        **_cube_arguments(context)
        | {"samples": {"sample_id": ["S1", "S2"], "donor": ["D1", "D2"]}}
    )
    return {
        "cubes": {"A": cube, "B": cube},
        "align_groups_by": "group_id",
        "group_mapping": {"B": {"T": "T"}},
        "group_names": {"T": "T cells"},
        "within_study": ["donor"],
    }


@functools.cache
def _collection() -> CubeCollection:
    cube = Cube(
        {"counts": np.arange(12, dtype=np.uint32).reshape(2, 2, 3)},
        **CUBE_AXES,
        n_cells=np.ones((2, 2), dtype=np.int64),
    )
    return CubeCollection({"A": cube, "B": cube})


def _from_cells_arguments(context: Context) -> dict[str, Any]:
    return {
        "counts": CELL_COUNTS,
        "samples": CELL_SAMPLES,
        "groups": CELL_GROUPS,
        "sample_table": {"sample_id": ["S1", "S2"]},
        "group_table": {"group_id": ["B", "T"]},
        "features": {"feature_id": ["g1", "g2", "g3"]},
        "library_sizes": [3.0, 4.0, 7.0, 5.0],
        "layers": ["counts", "detected"],
        "n_threads": 1,
        "source": {"dataset": "test", "values": [1, 2]},
    }


def _normalization(**options: Any) -> Callable[[Context], dict[str, Any]]:
    return lambda context: {"counts": COUNTS, **options}


def _differential_expression(context: Context) -> dict[str, Any]:
    """A bulk cube of COUNTS, a design matched to its samples, and options."""
    return {
        "cube": from_bulk(COUNTS, samples={"sample_id": SAMPLE_TABLE["sample_id"]}),
        "design": Design(SAMPLE_TABLE, "~ condition", levels=LEVELS),
        "groups": ["bulk"],
        "contrasts": {"stim": [0.0, 1.0]},
        "min_cells": 10,
        "min_counts": 100,
        "expression_filter": {"min_count": 5},
        "quality_weights": {"prior_n": 10.0},
        "adjust_across_groups": True,
    }


def _weighted_by_cells(context: Context) -> dict[str, Any]:
    """As _differential_expression, with a cube that records cell numbers, and prior
    weights from them instead of quality weights."""
    arguments = _differential_expression(context)
    del arguments["quality_weights"]
    return arguments | {
        "cube": Cube(
            {"counts": COUNTS[np.newaxis]},
            samples={"sample_id": SAMPLE_TABLE["sample_id"]},
            groups={"group_id": ["bulk"]},
            features={"feature_id": [f"g{j}" for j in range(COUNTS.shape[1])]},
            n_cells=np.array([[20], [30], [40], [50]]),
        ),
        "prior_weights": "n_cells",
    }


def _dream(context: Context) -> dict[str, Any]:
    """As _differential_expression, with a random donor intercept and no quality
    weights, which dream_voom does not estimate, and an F-test."""
    arguments = _differential_expression(context)
    del arguments["quality_weights"]
    return arguments | {
        "design": Design(
            SAMPLE_TABLE, "~ condition", random="(1 | donor)", levels=LEVELS
        ),
        "f_tests": {"both": ["stim"]},
        "span": 0.6,
        "df_method": "kenward_roger",
        "kenward_roger_weights": "precision",
        "n_threads": 1,
    }


def _dream_by_cells(context: Context) -> dict[str, Any]:
    """As _dream, with prior weights from the cell numbers of _weighted_by_cells."""
    arguments = _weighted_by_cells(context)
    return arguments | {"design": _dream(context)["design"]}


def _variance_partitioning(context: Context) -> dict[str, Any]:
    """The bulk cube of _differential_expression with a random donor intercept, and a
    value for every option of variance_partitioning."""
    arguments = _differential_expression(context)
    for name in ("contrasts", "quality_weights", "adjust_across_groups"):
        del arguments[name]
    return arguments | {
        "design": Design(SAMPLE_TABLE, "~ 1", random="(1 | donor)"),
        "span": 0.6,
        "method": "reml",
        "n_threads": 1,
    }


def _variance_partitioning_by_cells(context: Context) -> dict[str, Any]:
    """As _variance_partitioning, with prior weights from the cell numbers of
    _weighted_by_cells."""
    arguments = _weighted_by_cells(context)
    for name in ("contrasts", "adjust_across_groups"):
        del arguments[name]
    return arguments | {"design": _variance_partitioning(context)["design"]}


@functools.cache
def _variance_partition() -> Any:
    """The fractions of variance of the bulk cube of COUNTS with a random donor
    intercept, every gene fitted."""
    cube = from_bulk(COUNTS, samples={"sample_id": SAMPLE_TABLE["sample_id"]})
    design = Design(SAMPLE_TABLE, "~ 1", random="(1 | donor)")
    return variance_partitioning(cube, design, expression_filter=False, span=0.9)


def _scverse_arguments(context: Context, container: str) -> dict[str, Any]:
    """The cube as a long AnnData or a MuData, and a value for every argument of the
    import. Skipped without anndata, or mudata."""
    pytest.importorskip("anndata")
    pandas = pytest.importorskip("pandas")
    cube = context.cube
    if container == "mudata":
        pytest.importorskip("mudata")
        data = {"mdata": to_mudata(cube)}
    else:
        data = {"adata": to_anndata(cube), "sample": "sample_id", "group": "group_id"}
    return {
        **data,
        "layers": {"counts": "counts"},
        "n_cells": pandas.DataFrame(
            cube.n_cells,
            index=cube.samples["sample_id"].tolist(),
            columns=cube.groups["group_id"].tolist(),
        ),
        "sample_table": {"sample_id": ["S1", "S2"]},
        "group_table": {"group_id": ["B", "T"]},
        "normalization": {"method": "library_size", "target_sum": 1e4},
        "source": {"dataset": "test", "values": [1, 2]},
    }


# The arguments of from_anndata and from_mudata that take arrays: the id columns of
# the tables accept no missing values, and JSON metadata marks them with None.
_SCVERSE_ARRAYS: dict[str, str | None] = {
    "layers": None,
    "n_cells": None,
    "sample_table": None,
    "group_table": None,
    "normalization": "None",
    "source": "None",
}
_SCVERSE_KEYS = {
    "layers": "counts",
    "sample_table": "sample_id",
    "group_table": "group_id",
    "normalization": "target_sum",
    "source": "values",
}


@functools.cache
def _deseq2_fit() -> Any:
    """A DESeq2 fit of COUNTS on DESIGN at given dispersions and unit size factors."""
    return fit_deseq2(
        COUNTS, DESIGN, size_factors=np.ones(4), dispersion=np.full(8, 0.1)
    )


def _gene_set_arguments() -> dict[str, Any]:
    """log-CPM of COUNTS (complete), the design, a contrast and one set of two ids."""
    return {
        "y": log_cpm(COUNTS),
        "design": DESIGN,
        "contrast": [0.0, 1.0],
        "gene_sets": {"set": ["g1", "g2"]},
        "gene_ids": [f"g{i}" for i in range(COUNTS.shape[1])],
        "weights": np.full(COUNTS.shape, 2.0),
    }


def _mixed_model(context: Context) -> dict[str, Any]:
    """Complete log-CPM values, and a design with a random intercept per donor."""
    return {
        "y": log_cpm(COUNTS),
        "design": Design(
            SAMPLE_TABLE, "~ condition", random="(1 | donor)", levels=LEVELS
        ),
        "weights": np.full(COUNTS.shape, 2.0),
        "n_threads": 1,
    }


def _meta_analysis_arguments(context: Context) -> dict[str, Any]:
    """Two studies, the same differential expression of the bulk cube under two
    names, and a value for every argument that can hold an array."""
    arguments = _differential_expression(context)
    result = limma_voom(
        arguments["cube"],
        arguments["design"],
        contrasts=arguments["contrasts"],
        expression_filter=False,
    )
    return {
        "results": {"first": result, "second": result},
        "groups": ["bulk"],
        "contrasts": ["stim"],
        "min_studies": 2,
    }


@functools.cache
def _replication_studies() -> tuple[CubeCollection, dict[str, Any]]:
    """Two studies of the bulk cube of COUNTS, each tested on its aligned cube."""
    collection = CubeCollection({"A": _bulk_cube(), "B": _bulk_cube()})
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        results[study] = limma_voom(
            cube,
            Design(cube.samples, "~ condition", levels=LEVELS),
            contrasts={"stim": [0.0, 1.0]},
            expression_filter=False,
        )
    return collection, results


def _replication_arguments(context: Context | None) -> dict[str, Any]:
    """The replication of the two studies, with a value for every argument."""
    collection, results = _replication_studies()
    return {
        "collection": collection,
        "results": results,
        "group": "bulk",
        "contrast": "stim",
        "alpha": 0.5,
        "method": "stouffer",
        "at_least": 2,
        "conjunction_method": "fisher",
        "weights": {"A": 1.0, "B": 2.0},
        "meta_method": "dl",
        "null_proportion": {"lambdas": [0.2, 0.4, 0.6, 0.8]},
    }


@functools.cache
def _replication() -> Any:
    return differential_expression_replication(**_replication_arguments(None))


def _axes() -> Any:
    """An Axes of a figure that pyplot does not keep; skipped without matplotlib."""
    return pytest.importorskip("matplotlib.figure").Figure().subplots()


@functools.cache
def _bulk_cube() -> Cube:
    """COUNTS as a bulk cube with the sample table, for the MDS of a cube."""
    return from_bulk(COUNTS, samples=SAMPLE_TABLE)


@functools.cache
def _pathway_scores() -> Any:
    """GSVA scores of two sets in the four samples of COUNTS."""
    genes = [f"g{j}" for j in range(COUNTS.shape[1])]
    return gsva(log_cpm(COUNTS), {"set": genes[:3], "other": genes[3:]}, gene_ids=genes)


@functools.cache
def _activities() -> Any:
    """ULM activities of two sources in the four samples of COUNTS."""
    genes = [f"g{j}" for j in range(COUNTS.shape[1])]
    network = {"A": {"g1": 1.0, "g2": -1.0, "g5": 0.5}, "B": ["g0", "g3", "g4"]}
    return ulm(log_cpm(COUNTS), network, gene_ids=genes,
               sample_ids=SAMPLE_TABLE["sample_id"], min_targets=2)  # fmt: skip


# Two sources of a network, over the genes of COUNTS or the features of a cube.
NETWORK = {"A": {"g1": 1.0, "g2": -1.0, "g5": 0.5}, "B": ["g0", "g3", "g4"]}
FEATURE_NETWORK = {
    "A": ["feature_1", "feature_2", "feature_5"],
    "B": ["feature_0", "feature_3", "feature_4"],
}


@functools.cache
def _composition_result() -> Any:
    """propeller of the cell numbers of four samples, with one contrast."""
    cells = {"sample_id": SAMPLE_TABLE["sample_id"], "B": [10, 22, 15, 31]}
    cells["T"] = [40, 35, 52, 20]
    design = Design(SAMPLE_TABLE, "~ condition", levels=LEVELS)
    return propeller(cells, design, contrasts={"stim": [0.0, 1.0]})


def _tested(context: Context, **options: Any) -> Any:
    """limma_voom of the bulk cube of _differential_expression, every gene tested."""
    arguments = _differential_expression(context)
    return limma_voom(
        arguments["cube"],
        arguments["design"],
        contrasts=arguments["contrasts"],
        expression_filter=False,
        **options,
    )


def _forest_arguments(context: Context) -> dict[str, Any]:
    """The meta-analysis of _meta_analysis_arguments, its studies and a gene."""
    results = _meta_analysis_arguments(context)["results"]
    meta = differential_expression_meta_analysis(results)
    feature = meta.table("bulk", "stim")["feature_id"][0]
    return {"meta": meta, "studies": results, "feature": feature}


@functools.cache
def _studies() -> tuple[CubeCollection, dict[str, Any]]:
    """Two studies, the bulk cube of COUNTS twice, each tested by limma_voom."""
    cube = from_bulk(COUNTS, samples=SAMPLE_TABLE)
    collection = CubeCollection({"first": cube, "second": cube})
    design = Design(SAMPLE_TABLE, "~ condition", levels=LEVELS)
    results = {
        study: limma_voom(
            collection.aligned_cube(study),
            design,
            contrasts={"stim": [0.0, 1.0]},
            expression_filter=False,
        )
        for study in collection.studies
    }
    return collection, results


def _sumrank_arguments(context: Context) -> dict[str, Any]:
    """SumRank of the two studies, with permutations and a value for every option."""
    collection, results = _studies()
    return {
        "collection": collection,
        "results": results,
        "group": "bulk",
        "contrast": "stim",
        "proportion_of_studies": 1.0,
        "n_permutations": 2,
        "permute": "condition",
        "rng": 1,
    }


def _held_out_arguments(context: Context) -> dict[str, Any]:
    """Held-out prediction of the two studies, with a value for every option."""
    collection, results = _studies()
    return {
        "collection": collection,
        "results": results,
        "contrast": "stim",
        "variable": "condition",
        "case": "stim",
        "control": "ctrl",
        "groups": ["bulk"],
        "n_genes": 2,
        "proportion_of_studies": 1.0,
    }


TABLE = Table({"x": [1.0, 2.0], "s": ["a", "b"]})
NORM_FACTORS = [1.0, 1.1, 0.9, 1.0]
LIBRARY_SIZES = COUNTS.sum(axis=1).astype(float)
# Counts that vary more than Poisson counts between replicates, so that a common
# dispersion has a maximum inside its search interval.
OVERDISPERSED = COUNTS * np.array([[1.0], [2.0], [1.0], [3.0]])

ENTRY_POINTS: dict[str, Entry] = {
    # Tables.
    "cytepool.table.Table": Entry(
        lambda c: {"data": {"x": [1.0, 2.0], "s": ["a", "b"]}},
        {"data": "None or NaN"},
        keys={"data": "x"},
    ),
    "cytepool.table.Table.take": Entry(
        lambda c: {"indices": [1, 0]}, {"indices": None}, instance=lambda c: TABLE
    ),
    # A column named by its keyword takes one value, which a masked array is not.
    "cytepool.table.Table.filter": Entry(
        lambda c: {"mask": [True, False], "x": 1.0},
        {"mask": None, "x": None},
        instance=lambda c: TABLE,
    ),
    "cytepool.table.Table.select": Entry(
        lambda c: {"names": ["s", "x"]}, {"names": None}, instance=lambda c: TABLE
    ),
    # A keyword column is named by its keyword.
    "cytepool.table.Table.with_columns": Entry(
        lambda c: {"columns": {"y": [3.0, 4.0]}, "z": ["c", None]},
        {"columns": "None or NaN", "z": "None or NaN"},
        instance=lambda c: TABLE,
        keys={"columns": "y"},
    ),
    # Cubes.
    "cytepool.cube.cube.Cube": Entry(
        _cube_arguments,
        # The keys below are id columns, which accept no missing values; the
        # metadata of normalization and provenance is JSON, which marks them None.
        {
            "layers": None,
            "samples": None,
            "groups": None,
            "features": None,
            "n_cells": None,
            "normalization": "None",
            "provenance": "None",
        },
        options=("copy",),
        keys={
            "layers": "counts",
            "samples": "sample_id",
            "groups": "group_id",
            "features": "feature_id",
            "normalization": "target_sum",
            "provenance": "values",
        },
    ),
    # A group is an id or a position; operator.index reads a 0-d masked integer.
    "cytepool.cube.cube.Cube.layer": Entry(
        lambda c: {"name": "counts", "group": 1},
        {"group": None},
        options=("name",),
        instance=lambda c: c.cube,
    ),
    "cytepool.cube.cube.Cube.mean": Entry(
        lambda c: {"group": 1, "scale": "linear"},
        {"group": None},
        options=("scale",),
        instance=lambda c: c.cube,
    ),
    **{
        f"cytepool.cube.cube.Cube.{method}": Entry(
            lambda c: {"group": "T"}, {"group": None}, instance=lambda c: c.cube
        )
        for method in ("variance", "fraction_detected", "library_sizes")
    },
    "cytepool.cube.cube.Cube.select": Entry(
        lambda c: {"samples": ["S2"], "groups": [True, False], "features": [2, 0]},
        {"samples": None, "groups": None, "features": None},
        instance=lambda c: c.cube,
    ),
    # The mapping of each axis is read column by column, as a table reads it.
    "cytepool.cube.cube.Cube.with_columns": Entry(
        lambda c: {
            "samples": {"pool": ["a", "a"]},
            "groups": {"kind": ["x", None]},
            "features": {"length": [1.0, 2.0, np.nan]},
        },
        {"samples": "None or NaN", "groups": "None or NaN", "features": "None or NaN"},
        instance=lambda c: c.cube,
        keys={"samples": "pool", "groups": "kind", "features": "length"},
    ),
    "cytepool.cube.cube.Cube.collapse_samples": Entry(
        lambda c: {"by": ["sample_id"]}, {"by": None}, instance=lambda c: c.cube
    ),
    "cytepool.cube.cube.Cube.regroup": Entry(
        lambda c: {"mapping": {"all": ["B", "T"]}},
        {"mapping": None},
        instance=lambda c: c.cube,
        keys={"mapping": "all"},
    ),
    # The cubes were checked when they were built; the mapping of each study's
    # groups and the column names are read value by value.
    "cytepool.cube.collection.CubeCollection": Entry(
        _collection_arguments,
        {"group_mapping": None, "group_names": None, "within_study": None},
        options=("align_groups_by",),
        objects=("cubes",),
        keys={"group_mapping": "B", "group_names": "T"},
    ),
    "cytepool.cube.collection.CubeCollection.select": Entry(
        lambda c: {"studies": ["B"], "groups": [True, False], "features": [2, 0]},
        {"studies": None, "groups": None, "features": None},
        instance=lambda c: _collection(),
    ),
    "cytepool.cube.collection.CubeCollection.stack": Entry(
        lambda c: {"groups": ["T"], "features": [1, 0], "layers": ["counts"]},
        {"groups": None, "features": None, "layers": None},
        instance=lambda c: _collection(),
    ),
    "cytepool.cube.pool.from_cells": Entry(
        _from_cells_arguments,
        {
            "counts": None,
            "samples": None,
            "groups": None,
            "sample_table": None,
            "group_table": None,
            "features": None,
            "library_sizes": None,
            "layers": None,
            "n_threads": None,
            # A source record is JSON, which marks missing values with None.
            "source": "None",
        },
        options=("target_sum",),
        keys={
            "sample_table": "sample_id",
            "group_table": "group_id",
            "features": "feature_id",
            "source": "values",
        },
    ),
    # Feature ids rather than a feature table.
    "cytepool.cube.pool.from_cells[feature ids]": Entry(
        lambda c: {
            "counts": CELL_COUNTS,
            "samples": CELL_SAMPLES,
            "groups": CELL_GROUPS,
            "features": ["g1", "g2", "g3"],
        },
        {"features": None},
    ),
    "cytepool.cube.pool.from_bulk": Entry(
        lambda c: {
            "counts": COUNTS,
            "samples": {"sample_id": SAMPLE_TABLE["sample_id"]},
            "features": {"feature_id": [f"g{i}" for i in range(8)]},
            "source": {"dataset": "test", "values": [1, 2]},
        },
        {"counts": None, "samples": None, "features": None, "source": "None"},
        keys={"samples": "sample_id", "features": "feature_id", "source": "values"},
    ),
    "cytepool.cube.pool.from_bulk[ids]": Entry(
        lambda c: {
            "counts": COUNTS,
            "samples": SAMPLE_TABLE["sample_id"],
            "features": [f"g{i}" for i in range(8)],
        },
        {"samples": None, "features": None},
    ),
    # Provenance records are JSON, which marks missing values with None.
    "cytepool.cube.provenance.describe_provenance": Entry(
        lambda c: {"provenance": {"source": {"dataset": "test"}, "values": [1, 2]}},
        {"provenance": "None"},
        keys={"provenance": "values"},
    ),
    # Saved arrays. operator.index reads a 0-d masked integer, and so does a slice.
    "cytepool.io.zarr.ZarrArray.__getitem__": Entry(
        lambda c: {"key": (1, slice(0, 3))},
        {"key": None},
        instance=lambda c: c.zarr_array,
        names={"key": "index"},
    ),
    # Orthogonal indexing also takes integer arrays and masks along each axis.
    "cytepool.io.zarr.ZarrArray.oindex.__getitem__": Entry(
        lambda c: {"key": ([0, 2], slice(0, 3))},
        {"key": None},
        instance=lambda c: c.zarr_array.oindex,
        names={"key": "index"},
    ),
    # Cubes from scverse's data structures.
    "cytepool.io.anndata.from_anndata": Entry(
        lambda c: _scverse_arguments(c, "anndata"),
        _SCVERSE_ARRAYS,
        options=("sample", "group"),
        objects=("adata",),
        keys=_SCVERSE_KEYS,
    ),
    "cytepool.io.anndata.from_mudata": Entry(
        lambda c: _scverse_arguments(c, "mudata"),
        _SCVERSE_ARRAYS,
        objects=("mdata",),
        keys=_SCVERSE_KEYS,
    ),
    "cytepool.io.zarr.write_array": Entry(
        lambda c: {
            "path": c.new_path(),
            "data": np.arange(6.0).reshape(2, 3),
            "chunks": (1, 2),
            "level": 4,
            "dimension_names": ["row", "column"],
        },
        {"data": None, "chunks": None, "dimension_names": None},
        options=("compression", "level"),
        objects=("path", "attributes"),
    ),
    # Designs and contrasts.
    "cytepool.design.design.Design": Entry(
        lambda c: {
            "samples": SAMPLE_TABLE,
            "formula": "~ condition",
            "random": ["(1 | donor)"],
            "contrasts": {"stim": [0.0, 1.0]},
            "levels": LEVELS,
        },
        {
            "samples": "None or NaN",
            "random": None,
            "contrasts": None,
            "levels": None,
        },
        options=("formula",),
        keys={"samples": "condition", "contrasts": "stim", "levels": "condition"},
    ),
    "cytepool.design.design.Design.with_contrasts": Entry(
        lambda c: {"contrasts": {"stim": [0.0, 1.0]}},
        {"contrasts": None},
        instance=lambda c: c.design,
        keys={"contrasts": "stim"},
    ),
    # Contrasts given as expressions.
    "cytepool.design.design.Design.with_contrasts[expressions]": Entry(
        lambda c: {"contrasts": ["conditionstim"]},
        {"contrasts": None},
        instance=lambda c: c.design,
    ),
    "cytepool.design.model_matrix.model_matrix": Entry(
        lambda c: {"data": SAMPLE_TABLE, "formula": "~ condition", "levels": LEVELS},
        {"data": "None or NaN", "levels": None},
        options=("formula",),
        keys={"data": "condition", "levels": "condition"},
    ),
    "cytepool.design.model_matrix.ModelMatrix": Entry(
        lambda c: _fields(c.design.model_matrix),
        {
            "matrix": None,
            "columns": None,
            "assign": None,
            "terms": None,
            "levels": None,
            "variables": None,
            "coding": None,
        },
        options=("intercept", "formula"),
        keys={"levels": "condition"},
    ),
    "cytepool.design.contrasts.make_contrasts": Entry(
        lambda c: {"contrasts": {"stim": [0.0, 1.0]}, "columns": COLUMNS},
        {"contrasts": None, "columns": None},
        keys={"contrasts": "stim"},
    ),
    "cytepool.design.contrasts.Contrasts": Entry(
        lambda c: _fields(c.design.with_contrasts(["conditionstim"]).contrasts),
        {"matrix": None, "names": None, "columns": None},
    ),
    "cytepool.design.contrasts.Contrasts.select": Entry(
        lambda c: {"names": ["conditionstim"]},
        {"names": None},
        instance=lambda c: c.design.with_contrasts(["conditionstim"]).contrasts,
    ),
    # Normalization and filtering.
    "cytepool.normalization.library_sizes": Entry(_normalization(), {"counts": None}),
    "cytepool.normalization.tmm_factors": Entry(
        _normalization(library_sizes=LIBRARY_SIZES, reference=0),
        {"counts": None, "library_sizes": None},
        options=("reference", "log_ratio_trim", "sum_trim", "weighted", "a_cutoff"),
    ),
    "cytepool.normalization.upper_quartile_factors": Entry(
        _normalization(library_sizes=LIBRARY_SIZES),
        {"counts": None, "library_sizes": None},
        options=("p",),
    ),
    "cytepool.normalization.rle_factors": Entry(
        _normalization(library_sizes=LIBRARY_SIZES),
        {"counts": None, "library_sizes": None},
    ),
    "cytepool.normalization.median_ratio_size_factors": Entry(
        _normalization(), {"counts": None}, options=("method",)
    ),
    "cytepool.normalization.cpm": Entry(
        _normalization(library_sizes=LIBRARY_SIZES, norm_factors=NORM_FACTORS),
        {"counts": None, "library_sizes": None, "norm_factors": None},
    ),
    "cytepool.normalization.log_cpm": Entry(
        _normalization(library_sizes=LIBRARY_SIZES, norm_factors=NORM_FACTORS),
        {"counts": None, "library_sizes": None, "norm_factors": None},
        options=("prior_count",),
    ),
    "cytepool.normalization.filter_by_expression": Entry(
        _normalization(
            group=["a", "b", "a", "b"],
            library_sizes=LIBRARY_SIZES,
            norm_factors=NORM_FACTORS,
        ),
        {"counts": None, "group": None, "library_sizes": None, "norm_factors": None},
        options=("min_count", "min_total_count", "large_n", "min_prop"),
    ),
    # A design instead of groups; edgeR refuses both at once.
    "cytepool.normalization.filter_by_expression[design]": Entry(
        _normalization(design=DESIGN), {"design": None}
    ),
    # Linear models, voom and moderation.
    "cytepool.stats.linear_models.fit_linear_model": Entry(
        lambda c: {
            "y": c.expression,
            "design": DESIGN,
            "weights": np.full(c.expression.shape, 2.0),
            "block": SAMPLE_TABLE["donor"],
            "correlation": 0.2,
            "coefficient_names": ["a", "b"],
            "n_threads": 1,
        },
        {
            "y": "NaN",
            "design": None,
            "weights": None,
            "block": None,
            "coefficient_names": None,
            "n_threads": None,
        },
        options=("correlation",),
    ),
    # NaN marks what a fit cannot estimate in five of the fields.
    "cytepool.stats.linear_models.LinearModelFit": Entry(
        lambda c: _fields(c.contrast_fit),
        {
            "coefficients": "NaN",
            "stdev_unscaled": "NaN",
            "sigma": "NaN",
            "df_residual": None,
            "cov_coefficients": None,
            "mean_expression": "NaN",
            "coefficient_names": None,
            "design": None,
            "cov_coefficients_per_feature": "NaN",
            "contrasts": None,
        },
        options=("exact_standard_errors",),
    ),
    "cytepool.stats.linear_models.contrasts_fit": Entry(
        lambda c: {"fit": c.fit, "contrasts": np.eye(2), "names": ["u", "v"]},
        {"contrasts": None, "names": None},
        options=("method",),
        objects=("fit",),
    ),
    # Mixed models take complete values; n_threads goes through operator.index,
    # which reads a 0-d masked integer.
    "cytepool.stats.mixed_models.fit_mixed_model": Entry(
        _mixed_model,
        {"y": None, "weights": None, "n_threads": None},
        options=("method",),
        objects=("design",),
    ),
    "cytepool.stats.mixed_models.fixed_effect_tests": Entry(
        lambda c: {
            "fit": fit_mixed_model(**_mixed_model(c)),
            "contrasts": np.eye(2),
            "names": ["u", "v"],
        },
        {"contrasts": None, "names": None},
        options=("df_method", "kenward_roger_weights"),
        objects=("fit",),
    ),
    "cytepool.stats.mixed_models.fixed_effect_f_test": Entry(
        lambda c: {
            "fit": fit_mixed_model(**_mixed_model(c)),
            "contrasts": np.eye(2),
            "names": ["u", "v"],
            "df_method": "kenward_roger",
            "kenward_roger_weights": "precision",
        },
        {"contrasts": None, "names": None},
        options=("df_method", "kenward_roger_weights"),
        objects=("fit",),
    ),
    # dream: voom weights from mixed models, and their moderated tests.
    "cytepool.stats.dream.voom_with_dream_weights": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": _mixed_model(c)["design"],
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "span": 1.0,
            "weights": np.full(COUNTS.shape, 2.0),
            "method": "reml",
            "n_threads": 1,
        },
        {
            "counts": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
            "n_threads": None,
        },
        options=("span", "method"),
        objects=("design",),
    ),
    "cytepool.stats.dream.moderated_fixed_effect_tests": Entry(
        lambda c: {
            "tests": fixed_effect_tests(fit_mixed_model(**_mixed_model(c))),
            "stdev_coef_lim": [0.1, 4.0],
            "trend": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
            "prior_estimator": "likelihood",
        },
        {"stdev_coef_lim": None, "trend": "NaN", "winsor_tail_p": None},
        options=("proportion", "robust", "prior_estimator"),
        objects=("tests",),
    ),
    # A span given as a number; None, the default, chooses it.
    "cytepool.stats.voom.voom": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "span": 0.6,
            "weights": np.full(COUNTS.shape, 2.0),
        },
        {
            "counts": None,
            "design": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
        },
        options=("span",),
    ),
    # A variance design, here; variance groups below, as the two cannot be combined.
    "cytepool.stats.voom.voom_with_quality_weights": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "span": 0.6,
            "variance_design": [[0.0], [0.0], [1.0], [1.0]],
            "method": "reml",
            "prior_n": 5.0,
            "max_iterations": 20,
            "tolerance": 1e-6,
        },
        {
            "counts": None,
            "design": None,
            "library_sizes": None,
            "norm_factors": None,
            "variance_design": None,
        },
        options=("span", "method", "prior_n", "max_iterations", "tolerance"),
    ),
    "cytepool.stats.voom.voom_with_quality_weights[groups]": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "variance_groups": ["a", "a", "b", "b"],
        },
        {"variance_groups": None},
    ),
    # NaN marks the features that the trend leaves out.
    "cytepool.stats.voom.VoomResult": Entry(
        lambda c: _fields(voom_with_quality_weights(COUNTS, DESIGN)),
        {
            "log_cpm": None,
            "weights": None,
            "effective_library_sizes": None,
            "mean_log_count": "NaN",
            "sqrt_sd": "NaN",
            "sqrt_sd_trend": "NaN",
            "sample_weights": None,
        },
        options=("span",),
    ),
    # Missing values of y marked with NaN; observation weights and a variance
    # design here, and variance groups below.
    "cytepool.stats.sample_weights.sample_quality_weights": Entry(
        lambda c: {
            "y": c.expression,
            "design": DESIGN,
            "weights": np.full(c.expression.shape, 2.0),
            "variance_design": [[0.0], [0.0], [1.0], [1.0]],
            "prior_n": 5.0,
            "method": "gene_by_gene",
            "max_iterations": 20,
            "tolerance": 1e-6,
        },
        {"y": "NaN", "design": None, "weights": None, "variance_design": None},
        options=("prior_n", "method", "max_iterations", "tolerance"),
    ),
    "cytepool.stats.sample_weights.sample_quality_weights[groups]": Entry(
        lambda c: {
            "y": c.expression,
            "design": DESIGN,
            "variance_groups": ["a", "a", "b", "b"],
        },
        {"variance_groups": None},
    ),
    # Negative binomial fits: library sizes and normalization factors, or offsets.
    "cytepool.stats.negative_binomial_models.fit_negative_binomial_model": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "dispersion": np.full(8, 0.1),
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "weights": np.full((4, 8), 2.0),
            "coefficient_names": ["a", "b"],
            "prior_count": 0.5,
            "max_iterations": 50,
            "n_threads": 1,
        },
        {
            "counts": None,
            "design": None,
            "dispersion": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
            "coefficient_names": None,
            "n_threads": None,
        },
        options=("prior_count", "max_iterations"),
    ),
    "cytepool.stats.negative_binomial_models.fit_negative_binomial_model[offsets]": (
        Entry(
            lambda c: {
                "counts": COUNTS,
                "design": DESIGN,
                "dispersion": 0.1,
                "offsets": np.log(LIBRARY_SIZES),
            },
            {"offsets": None},
        )
    ),
    "cytepool.stats.negative_binomial_models.negative_binomial_deviance": Entry(
        lambda c: {
            "counts": COUNTS,
            "means": COUNTS + 1.0,
            "dispersion": np.full(8, 0.1),
            "weights": np.full(4, 2.0),
        },
        {"counts": None, "means": None, "dispersion": None, "weights": None},
    ),
    "cytepool.stats.negative_binomial_models.negative_binomial_unit_deviance": Entry(
        lambda c: {"counts": COUNTS, "means": COUNTS + 1.0, "dispersion": 0.1},
        {"counts": None, "means": None, "dispersion": None},
    ),
    # DESeq2: dispersions with every stage given or estimated, fits, and tests.
    "cytepool.stats.deseq2.deseq2_dispersions": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "size_factors": np.ones(4),
            "trend": "mean",
            "genewise_dispersions": np.full(8, 0.1),
            "trended_dispersions": np.full(8, 0.1),
            "prior_variance": 1.0,
            "min_mean": 0.5,
            "n_threads": 1,
        },
        {
            "counts": None,
            "design": None,
            "size_factors": None,
            "genewise_dispersions": "NaN",
            "trended_dispersions": "NaN",
            "n_threads": None,
        },
        options=("trend", "prior_variance", "min_mean"),
    ),
    "cytepool.stats.deseq2.fit_deseq2": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "size_factors": np.ones(4),
            "dispersion": np.full(8, 0.1),
            "trend": "mean",
            "min_mean": 0.5,
            "n_threads": 1,
        },
        {
            "counts": None,
            "design": None,
            "size_factors": None,
            "dispersion": "NaN",
            "n_threads": None,
        },
        options=("trend", "min_mean"),
    ),
    "cytepool.stats.deseq2.deseq2_wald_test": Entry(
        lambda c: {
            "fit": _deseq2_fit(),
            "contrast": [0.0, 1.0],
            "alpha": 0.1,
            "independent_filtering": True,
            "cooks_cutoff": 10.0,
        },
        {"contrast": None},
        options=("alpha", "independent_filtering", "cooks_cutoff"),
        objects=("fit",),
    ),
    "cytepool.stats.deseq2.deseq2_likelihood_ratio_test": Entry(
        lambda c: {
            "fit": _deseq2_fit(),
            "reduced": DESIGN[:, :1],
            "coefficient": "x2",
            "alpha": 0.1,
            "independent_filtering": True,
            "cooks_cutoff": 10.0,
            "n_threads": 1,
        },
        {"reduced": None, "n_threads": None},
        options=("coefficient", "alpha", "independent_filtering", "cooks_cutoff"),
        objects=("fit",),
    ),
    # Quasi-likelihood fits and tests (edgeR's glmQLFit and glmQLFTest).
    "cytepool.stats.quasi_likelihood.average_log_cpm": Entry(
        lambda c: {
            "counts": COUNTS,
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "weights": np.full((4, 8), 2.0),
            "dispersion": np.full(8, 0.1),
            "prior_count": 1.0,
        },
        {
            "counts": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
            "dispersion": None,
        },
        options=("prior_count",),
    ),
    "cytepool.stats.quasi_likelihood.average_log_cpm[offsets]": Entry(
        lambda c: {"counts": COUNTS, "offsets": np.log(LIBRARY_SIZES)},
        {"offsets": None},
    ),
    "cytepool.stats.quasi_likelihood.common_dispersion": Entry(
        lambda c: {
            "counts": OVERDISPERSED,
            "design": DESIGN,
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "weights": np.full((4, 8), 2.0),
            "interval": [0.01, 2.0],
            "min_row_sum": 1.0,
            "subset": 100,
            "coefficient_names": ["a", "b"],
            "max_iterations": 50,
            "n_threads": 1,
        },
        {
            "counts": None,
            "design": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
            "interval": None,
            "coefficient_names": None,
            "n_threads": None,
        },
        options=("min_row_sum", "subset", "max_iterations"),
    ),
    "cytepool.stats.quasi_likelihood.common_dispersion[offsets]": Entry(
        lambda c: {
            "counts": OVERDISPERSED,
            "design": DESIGN,
            "offsets": np.log(LIBRARY_SIZES),
            "interval": [0.01, 2.0],
        },
        {"offsets": None},
    ),
    "cytepool.stats.quasi_likelihood.fit_quasi_likelihood": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "dispersion": np.full(8, 0.1),
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
            "weights": np.full((4, 8), 2.0),
            "trend": np.linspace(1.0, 2.0, 8),
            "robust": False,
            "prior_count": 0.5,
            "coefficient_names": ["a", "b"],
            "max_iterations": 50,
            "n_threads": 1,
        },
        {
            "counts": None,
            "design": None,
            "dispersion": None,
            "library_sizes": None,
            "norm_factors": None,
            "weights": None,
            "trend": "NaN",
            "coefficient_names": None,
            "n_threads": None,
        },
        options=("robust", "prior_count", "max_iterations"),
    ),
    "cytepool.stats.quasi_likelihood.fit_quasi_likelihood[offsets]": Entry(
        lambda c: {
            "counts": COUNTS + np.array([[0], [0], [0], [40]]),
            "design": DESIGN,
            "offsets": np.log(LIBRARY_SIZES),
            "top_proportion": 1.0,
        },
        {"offsets": None},
        options=("top_proportion",),
    ),
    "cytepool.stats.quasi_likelihood.negative_binomial_unit_deviance_moments": Entry(
        lambda c: {"means": COUNTS + 1.0, "dispersion": 0.1},
        {"means": None, "dispersion": None},
    ),
    # Tables of the tests of fits on arrays: one id per gene.
    "cytepool.stats.empirical_bayes.ModeratedFit.table": Entry(
        lambda c: {"coefficient": "second", "feature_ids": [f"g{j}" for j in range(8)]},
        {"feature_ids": None},
        options=("coefficient",),
        instance=lambda c: moderated_statistics(c.contrast_fit),
    ),
    "cytepool.stats.quasi_likelihood.QuasiLikelihoodTest.table": Entry(
        lambda c: {"feature_ids": [f"g{j}" for j in range(8)]},
        {"feature_ids": None},
        instance=lambda c: quasi_likelihood_f_test(
            fit_quasi_likelihood(COUNTS, DESIGN, dispersion=0.1), coefficients=[1]
        ),
    ),
    "cytepool.stats.deseq2.DESeq2Results.table": Entry(
        lambda c: {"feature_ids": [f"g{j}" for j in range(8)]},
        {"feature_ids": None},
        instance=lambda c: deseq2_wald_test(_deseq2_fit(), [0.0, 1.0]),
    ),
    "cytepool.stats.quasi_likelihood.quasi_likelihood_f_test": Entry(
        lambda c: {
            "fit": fit_quasi_likelihood(COUNTS, DESIGN, dispersion=0.1),
            "contrasts": np.array([0.0, 1.0]),
            "names": ["b"],
        },
        {"contrasts": None, "names": None},
        objects=("fit",),
    ),
    "cytepool.stats.quasi_likelihood.quasi_likelihood_f_test[coefficients]": Entry(
        lambda c: {
            "fit": fit_quasi_likelihood(COUNTS, DESIGN, dispersion=0.1),
            "coefficients": [1],
        },
        {"coefficients": None},
        objects=("fit",),
    ),
    "cytepool.stats.empirical_bayes.fit_f_distribution": Entry(
        lambda c: {
            "variances": c.fit.sigma**2,
            "df": c.fit.df_residual,
            "covariate": np.linspace(1.0, 2.0, 8),
        },
        {"variances": "NaN", "df": None, "covariate": "NaN"},
    ),
    "cytepool.stats.empirical_bayes.fit_f_distribution_robust": Entry(
        lambda c: {
            "variances": c.fit.sigma**2,
            "df": c.fit.df_residual,
            "covariate": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
        },
        {"variances": "NaN", "df": None, "covariate": "NaN", "winsor_tail_p": None},
    ),
    "cytepool.stats.empirical_bayes.fit_f_distribution_likelihood": Entry(
        lambda c: {
            "variances": c.fit.sigma**2,
            "df": c.fit.df_residual,
            "covariate": np.linspace(1.0, 2.0, 8),
            "prior_weights": np.linspace(0.5, 1.5, 8),
        },
        {"variances": "NaN", "df": None, "covariate": "NaN", "prior_weights": None},
        options=("robust",),
    ),
    # The likelihood estimator, as the default chooses it for unequal df.
    "cytepool.stats.empirical_bayes.squeeze_variances": Entry(
        lambda c: {
            "variances": c.fit.sigma**2,
            "df": c.fit.df_residual,
            "covariate": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
            "prior_estimator": "likelihood",
        },
        {"variances": "NaN", "df": None, "covariate": "NaN", "winsor_tail_p": None},
        options=("robust", "prior_estimator"),
    ),
    # winsor_tail_p is read even when robust is False.
    "cytepool.stats.empirical_bayes.moderated_statistics": Entry(
        lambda c: {
            "fit": c.fit,
            "stdev_coef_lim": [0.1, 4.0],
            "trend": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
            "f_statistic": "exact",
            "prior_estimator": "likelihood",
        },
        {"stdev_coef_lim": None, "trend": "NaN", "winsor_tail_p": None},
        options=("proportion", "robust", "f_statistic", "prior_estimator"),
        objects=("fit",),
    ),
    "cytepool.stats.empirical_bayes.treat": Entry(
        lambda c: {
            "fit": c.fit,
            "lfc": 0.5,
            "trend": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
            "prior_estimator": "moments",
        },
        {"trend": "NaN", "winsor_tail_p": None},
        options=("lfc", "robust", "prior_estimator"),
        objects=("fit",),
    ),
    # A fold change instead of its logarithm; treat refuses both at once.
    "cytepool.stats.empirical_bayes.treat[fc]": Entry(
        lambda c: {"fit": c.fit, "fc": 1.5}, {}, options=("fc",), objects=("fit",)
    ),
    # Gene-set tests take complete values: none may be missing.
    "cytepool.stats.gene_set_tests.camera": Entry(
        lambda c: {**_gene_set_arguments(), "inter_gene_correlation": 0.05},
        dict.fromkeys(
            ("y", "design", "contrast", "gene_sets", "gene_ids", "weights"), None
        ),
        options=(
            "inter_gene_correlation",
            "allow_negative_correlation",
            "use_ranks",
            "trend",
            "robust",
        ),
        keys={"gene_sets": "set"},
    ),
    "cytepool.stats.gene_set_tests.fry": Entry(
        lambda c: _gene_set_arguments(),
        dict.fromkeys(
            ("y", "design", "contrast", "gene_sets", "gene_ids", "weights"), None
        ),
        options=("standardize", "trend", "robust"),
        keys={"gene_sets": "set"},
    ),
    # Rotation tests take complete values too; contribution weights are finite.
    "cytepool.stats.rotation_gene_set_tests.roast": Entry(
        lambda c: (
            {
                key: value
                for key, value in _gene_set_arguments().items()
                if key != "gene_sets"
            }
            | {
                "gene_set": ["g1", "g2"],
                "gene_weights": np.linspace(-1.0, 1.0, COUNTS.shape[1]),
                "n_rotations": 9,
                "rng": 1,
                "n_threads": 1,
            }
        ),
        dict.fromkeys(
            (
                "y",
                "design",
                "contrast",
                "gene_set",
                "gene_ids",
                "weights",
                "gene_weights",
            ),
            None,
        ),
        options=(
            "set_statistic",
            "n_rotations",
            "trend",
            "robust",
            "rng",
            "n_threads",
        ),
    ),
    "cytepool.stats.rotation_gene_set_tests.mroast": Entry(
        lambda c: {
            **_gene_set_arguments(),
            "gene_weights": np.linspace(-1.0, 1.0, COUNTS.shape[1]),
            "n_rotations": 9,
            "rng": 1,
            "n_threads": 1,
        },
        dict.fromkeys(
            (
                "y",
                "design",
                "contrast",
                "gene_sets",
                "gene_ids",
                "weights",
                "gene_weights",
            ),
            None,
        ),
        options=(
            "set_statistic",
            "n_rotations",
            "trend",
            "robust",
            "mid_p",
            "rng",
            "n_threads",
        ),
        keys={"gene_sets": "set"},
    ),
    # romer takes sample weights only.
    "cytepool.stats.rotation_gene_set_tests.romer": Entry(
        lambda c: {
            **_gene_set_arguments(),
            "weights": np.full(COUNTS.shape[0], 2.0),
            "n_rotations": 9,
            "rng": 1,
            "n_threads": 1,
        },
        dict.fromkeys(
            ("y", "design", "contrast", "gene_sets", "gene_ids", "weights"), None
        ),
        options=(
            "set_statistic",
            "n_rotations",
            "shrink_contrast_effects",
            "rng",
            "n_threads",
        ),
        keys={"gene_sets": "set"},
    ),
    # Gene-set collections: no gene of a set may be missing; the genes of an
    # analysis, a selection and a universe mark missing ids with None.
    "cytepool.gene_sets.GeneSets": Entry(
        lambda c: {
            "sets": {"set": ["g1", "g2"]},
            "descriptions": {"set": "two genes"},
            "source": "a curated list",
            "version": "1",
            "licence": "CC0",
            "file": "sets.gmt",
            "sha256": "0" * 64,
        },
        {"sets": None},
        options=("source", "version", "licence", "file", "sha256"),
        objects=("descriptions",),
        keys={"sets": "set"},
    ),
    "cytepool.gene_sets.GeneSets.select": Entry(
        lambda c: {"names": ["other", "set"]},
        {"names": None},
        instance=lambda c: GeneSets({"set": ["g1", "g2"], "other": ["g3"]}),
    ),
    "cytepool.gene_sets.GeneSets.restrict_to": Entry(
        lambda c: {"genes": ["g1", None, "g3"], "keep_empty": True},
        {"genes": "None"},
        options=("keep_empty",),
        instance=lambda c: GeneSets({"set": ["g1", "g2"], "other": ["g3"]}),
    ),
    "cytepool.gene_sets.GeneSets.filter_by_size": Entry(
        lambda c: {"min_size": 1, "max_size": 2},
        {},
        options=("min_size", "max_size"),
        instance=lambda c: GeneSets({"set": ["g1", "g2"], "other": ["g3"]}),
    ),
    "cytepool.gene_sets.write_gmt": Entry(
        lambda c: {
            "gene_sets": {"set": ["g1", "g2"]},
            "path": c.new_path(),
            "overwrite": False,
        },
        {"gene_sets": None},
        options=("overwrite",),
        objects=("path",),
        keys={"gene_sets": "set"},
    ),
    # A list of selected genes with a universe; the options that read a table are
    # checked first.
    "cytepool.over_representation.over_representation": Entry(
        lambda c: {
            "genes": ["g1", "g2", None],
            "gene_sets": {"set": ["g1", "g3"]},
            "universe": [f"g{j}" for j in range(COUNTS.shape[1])],
            "alpha": 0.1,
            "id_column": "feature_id",
            "p_value_column": "adjusted_p_value",
            "fold_change_column": "log2_fold_change",
        },
        {"genes": "None", "gene_sets": None, "universe": "None"},
        options=("alpha", "id_column", "p_value_column", "fold_change_column"),
        keys={"gene_sets": "set"},
    ),
    # A differential-expression result, of which a group and a contrast are tested.
    "cytepool.over_representation.over_representation[result]": Entry(
        lambda c: {
            "genes": _tested(c),
            "gene_sets": {"set": ["feature_1", "feature_2"]},
            "group": "bulk",
            "contrast": "stim",
        },
        {},
        options=("group", "contrast"),
        objects=("genes", "gene_sets"),
    ),
    # Statistics by gene id, ranked; the options that read a table are checked too.
    "cytepool.gsea.gsea": Entry(
        lambda c: {
            "statistics": RANKED,
            "gene_sets": {"set": ["g1", "g2", "g5"]},
            "id_column": "feature_id",
            "statistic_column": "t",
            "min_size": 1,
            "max_size": 5,
            "exponent": 1.0,
            "score_type": "std",
            "method": "permutation",
            "n_permutations": 20,
            "sample_size": 11,
            "eps": 1e-10,
            "rng": 1,
            "n_threads": 1,
        },
        {"statistics": "NaN", "gene_sets": None},
        options=(
            "id_column",
            "statistic_column",
            "min_size",
            "max_size",
            "exponent",
            "score_type",
            "method",
            "n_permutations",
            "sample_size",
            "eps",
            "rng",
            "n_threads",
        ),
        keys={"statistics": "g0", "gene_sets": "set"},
    ),
    # A differential-expression result, of which a group and a contrast are ranked.
    "cytepool.gsea.gsea[result]": Entry(
        lambda c: {
            "statistics": _tested(c),
            "gene_sets": {"set": ["feature_1", "feature_2"]},
            "group": "bulk",
            "contrast": "stim",
            "method": "permutation",
            "n_permutations": 200,
        },
        {},
        options=("group", "contrast"),
        objects=("statistics", "gene_sets"),
    ),
    # Values, gene ids and sample ids of an array; a cube's options are refused with
    # one.
    **{
        f"cytepool.pathway_scores.{name}": Entry(
            lambda c, options=options: {
                "data": c.expression if missing else log_cpm(COUNTS),
                "gene_sets": {"set": ["g1", "g2", "g5"], "other": ["g0", "g3"]},
                "gene_ids": [f"g{j}" for j in range(7)] + [None],
                "sample_ids": ["S1", "S2", "S3", "S4"],
                **options,
            },
            {
                "data": missing,
                "gene_sets": None,
                "gene_ids": "None",
                "sample_ids": None,
            },
            options=tuple(options),
            keys={"gene_sets": "set"},
        )
        for name, options, missing in (
            (
                "gsva",
                {
                    "kernel": "gaussian",
                    "min_samples_without_kernel": 200,
                    "exponent": 1.0,
                    "score_type": "difference",
                    "min_size": 1,
                    "max_size": 5,
                    "n_threads": 1,
                },
                "NaN",
            ),
            (
                "ssgsea",
                {
                    "exponent": 0.25,
                    "normalize": True,
                    "min_size": 1,
                    "max_size": 5,
                    "n_threads": 1,
                },
                "NaN",
            ),
            ("combined_z_scores", {"min_size": 1, "max_size": 5}, "NaN"),
            ("plage", {"min_size": 1, "max_size": 5}, None),
        )
    },
    # A cube's group, its genes named by a column of its features table.
    **{
        f"cytepool.pathway_scores.{name}[cube]": Entry(
            lambda c: {
                "data": _bulk_cube(),
                "gene_sets": {
                    "set": ["feature_1", "feature_2", "feature_5"],
                    "other": ["feature_0", "feature_3"],
                },
                "gene_ids": "feature_id",
                "group": "bulk",
                "values": "counts",
                "min_cells": 1,
                "expression_filter": False,
            },
            {},
            options=("gene_ids", "group", "values", "min_cells", "expression_filter"),
            objects=("data", "gene_sets"),
        )
        for name in ("gsva", "ssgsea", "combined_z_scores", "plage")
    },
    # Values, gene ids and sample ids of an array, and a network as a mapping, which
    # Network reads as its edges; a cube's options and a result's are refused.
    **{
        f"cytepool.activity.{name}": Entry(
            lambda c, options=options: {
                "data": c.expression,
                "network": NETWORK,
                "gene_ids": [f"g{j}" for j in range(7)] + [None],
                "sample_ids": ["S1", "S2", "S3", "S4"],
                "min_targets": 2,
                **options,
            },
            {
                "data": "NaN",
                "network": "None or NaN",
                "gene_ids": "None",
                "sample_ids": None,
            },
            options=("min_targets", *options),
            keys={"network": "A"},
            names={"network": "edges"},
        )
        for name, options in (
            ("ulm", {"n_threads": 1}),
            ("mlm", {}),
            (
                "weighted_aggregate",
                {"statistic": "sum", "n_permutations": 10, "rng": 1, "n_threads": 1},
            ),
        )
    },
    # A cube's group, its genes named by a column of its features table.
    **{
        f"cytepool.activity.{name}[cube]": Entry(
            lambda c: {
                "data": _bulk_cube(),
                "network": FEATURE_NETWORK,
                "gene_ids": "feature_id",
                "group": "bulk",
                "min_cells": 1,
                "expression_filter": False,
                "min_targets": 2,
            },
            {},
            options=("gene_ids", "group", "min_cells", "expression_filter"),
            objects=("data", "network"),
        )
        for name in ("ulm", "mlm", "weighted_aggregate")
    },
    # The statistics of a group and contrast of a differential-expression result.
    **{
        f"cytepool.activity.{name}[result]": Entry(
            lambda c: {
                "data": _tested(c),
                "network": FEATURE_NETWORK,
                "group": "bulk",
                "contrast": "stim",
                "id_column": "feature_id",
                "statistic_column": "t",
                "min_targets": 2,
            },
            {},
            options=("group", "contrast", "id_column", "statistic_column"),
            objects=("data", "network"),
        )
        for name in ("ulm", "mlm", "weighted_aggregate")
    },
    # Edges as a mapping of source to targets, and text that the network keeps.
    "cytepool.networks.Network": Entry(
        lambda c: {
            "edges": {"A": {"g1": 1.0, "g2": -1.0}, "B": ["g3"]},
            "name": "mine",
            "version": "1",
            "url": "https://example.org/network.csv",
            "licence": "CC0",
            "citation": "a paper",
            "file": "network.csv",
            "sha256": "0" * 64,
        },
        {"edges": "None or NaN"},
        options=("name", "version", "url", "licence", "citation", "file", "sha256"),
        keys={"edges": "A"},
    ),
    "cytepool.networks.Network.select": Entry(
        lambda c: {"sources": ["B", "A"]},
        {"sources": None},
        instance=lambda c: Network({"A": ["g1", "g2"], "B": ["g3"]}),
    ),
    # n and axis go through operator.index, which reads a 0-d masked integer.
    "cytepool.stats.multiple_testing.adjust_p_values": Entry(
        lambda c: {"p": [0.01, 0.2, 0.5], "n": 5, "axis": 0},
        {"p": "NaN", "n": None, "axis": None},
        options=("method",),
    ),
    # Numbers of points and the settings of the rule, each a number.
    "cytepool.stats.smoothing.choose_lowess_span": Entry(
        lambda c: {"n": 400, "small_n": 10, "min_span": 0.2, "power": 0.5},
        {},
        options=("n", "small_n", "min_span", "power"),
    ),
    "cytepool.stats.smoothing.lowess": Entry(
        lambda c: {
            "x": np.linspace(0.0, 1.0, 10),
            "y": np.linspace(1.0, 3.0, 10),
            "delta": 0.01,
        },
        {"x": None, "y": None},
        options=("span", "iterations", "delta"),
    ),
    "cytepool.stats.smoothing.natural_spline_basis": Entry(
        lambda c: {
            "x": np.linspace(0.0, 1.0, 10),
            "knots": [0.5],
            "boundary_knots": [0.0, 1.0],
        },
        {"x": "NaN", "knots": None, "boundary_knots": None},
        options=("intercept",),
    ),
    # Degrees of freedom instead of knots.
    "cytepool.stats.smoothing.natural_spline_basis[df]": Entry(
        lambda c: {"x": np.linspace(0.0, 1.0, 10), "df": 3},
        {"x": "NaN"},
        options=("df",),
    ),
    # Differential expression. min_cells and n_threads go through operator.index,
    # which reads a 0-d masked integer.
    "cytepool.differential_expression.limma_voom": Entry(
        lambda c: {**_differential_expression(c), "span": 0.6, "n_threads": 1},
        {
            "contrasts": None,
            "groups": None,
            "min_cells": None,
            "expression_filter": None,
            "quality_weights": None,
            "n_threads": None,
        },
        options=(
            "min_counts",
            "normalization",
            "span",
            "robust",
            "contrast_standard_errors",
            "adjust_across_groups",
        ),
        objects=("cube", "design"),
        keys={
            "contrasts": "stim",
            "expression_filter": "min_count",
            "quality_weights": "prior_n",
        },
    ),
    # Contrasts given as expressions.
    "cytepool.differential_expression.limma_voom[expressions]": Entry(
        lambda c: {**_differential_expression(c), "contrasts": ["conditionstim"]},
        {"contrasts": None},
    ),
    # Prior weights from cell numbers, which need a cube that records them and
    # exclude quality weights.
    "cytepool.differential_expression.limma_voom[prior_weights]": Entry(
        _weighted_by_cells, {}, options=("prior_weights",)
    ),
    "cytepool.differential_expression.dream_voom": Entry(
        _dream,
        {
            "contrasts": None,
            "f_tests": None,
            "groups": None,
            "min_cells": None,
            "expression_filter": None,
            "n_threads": None,
        },
        options=(
            "min_counts",
            "normalization",
            "span",
            "df_method",
            "kenward_roger_weights",
            "robust",
            "trend",
            "adjust_across_groups",
        ),
        objects=("cube", "design"),
        keys={"contrasts": "stim", "f_tests": "both", "expression_filter": "min_count"},
    ),
    "cytepool.differential_expression.dream_voom[prior_weights]": Entry(
        _dream_by_cells, {}, options=("prior_weights",)
    ),
    "cytepool.differential_expression.limma_trend": Entry(
        _differential_expression,
        {
            "contrasts": None,
            "groups": None,
            "min_cells": None,
            "expression_filter": None,
            "quality_weights": None,
        },
        options=(
            "min_counts",
            "normalization",
            "prior_count",
            "robust",
            "adjust_across_groups",
        ),
        objects=("cube", "design"),
        keys={
            "contrasts": "stim",
            "expression_filter": "min_count",
            "quality_weights": "prior_n",
        },
    ),
    "cytepool.differential_expression.limma_trend[prior_weights]": Entry(
        _weighted_by_cells, {}, options=("prior_weights",)
    ),
    # Composition. A table of cell numbers takes no missing values; a cube would be
    # an object, built and checked before.
    "cytepool.composition.propeller": Entry(
        lambda c: {
            "data": {
                "sample_id": SAMPLE_TABLE["sample_id"],
                "B": [10, 22, 15, 31],
                "T": [40, 35, 52, 20],
            },
            "design": Design(SAMPLE_TABLE, "~ condition", levels=LEVELS),
            "contrasts": {"stim": [0.0, 1.0]},
            "f_test": "condition",
        },
        {"data": None, "contrasts": None},
        options=("f_test", "transform", "robust", "trend"),
        objects=("design",),
        keys={"data": "B", "contrasts": "stim"},
    ),
    # Contrasts given as expressions.
    "cytepool.composition.propeller[expressions]": Entry(
        lambda c: {
            "data": {
                "sample_id": SAMPLE_TABLE["sample_id"],
                "B": [1, 2, 3, 4],
                "T": [6, 5, 7, 4],
            },
            "design": Design(SAMPLE_TABLE, "~ condition", levels=LEVELS),
            "contrasts": ["conditionstim"],
        },
        {"contrasts": None},
    ),
    # Principal components: of an array, or of a cube's group (an object).
    "cytepool.principal_components.principal_components": Entry(
        lambda c: {"data": log_cpm(COUNTS), "top": 4},
        {"data": None},
        options=("top",),
    ),
    "cytepool.principal_components.principal_components[cube]": Entry(
        lambda c: {"data": _bulk_cube(), "group": "bulk", "min_cells": 1},
        {},
        options=("group", "min_cells"),
        objects=("data",),
    ),
    # Meta-analysis. NaN marks a study that did not measure a feature.
    "cytepool.meta_analysis.meta_analysis": Entry(
        lambda c: {
            "effects": [[0.1, 0.4], [0.3, np.nan], [0.2, 0.5]],
            "standard_errors": [[0.1, 0.2], [0.2, 0.3], [0.15, 0.1]],
            "level": 0.9,
        },
        {"effects": "NaN", "standard_errors": "NaN"},
        options=("method", "test", "level"),
    ),
    # min_studies goes through operator.index, which reads a 0-d masked integer;
    # the results were checked when they were built.
    "cytepool.meta_analysis.differential_expression_meta_analysis": Entry(
        _meta_analysis_arguments,
        {"groups": None, "contrasts": None, "min_studies": None},
        options=("method", "test", "level"),
        objects=("results",),
    ),
    # Combined p-values and pi0: NaN marks a study that did not test a feature, or a
    # missing p-value.
    "cytepool.stats.combined_p_values.combine_p_values": Entry(
        lambda c: {
            "p_values": [[0.01, 0.4], [0.3, np.nan], [0.02, 0.5]],
            "method": "stouffer",
            "weights": [1.0, 2.0, 3.0],
            "directions": [[1.0, -1.0], [1.0, np.nan], [-1.0, 1.0]],
        },
        {"p_values": "NaN", "weights": "NaN", "directions": "NaN"},
        options=("method",),
    ),
    "cytepool.stats.combined_p_values.combine_p_values[wilkinson]": Entry(
        lambda c: {"p_values": [0.01, 0.3, 0.02], "method": "wilkinson", "rank": 2},
        {},
        options=("rank",),
    ),
    "cytepool.stats.combined_p_values.partial_conjunction": Entry(
        lambda c: {
            "p_values": [[0.01, 0.4], [0.3, np.nan], [0.02, 0.5]],
            "at_least": 2,
            "method": "fisher",
            "directions": [[1.0, -1.0], [1.0, np.nan], [-1.0, 1.0]],
        },
        {"p_values": "NaN", "directions": "NaN"},
        options=("at_least", "method"),
    ),
    "cytepool.stats.multiple_testing.proportion_of_true_nulls": Entry(
        lambda c: {
            "p": np.linspace(0.0, 1.0, 21),
            "lambdas": [0.2, 0.4, 0.6, 0.8],
            "method": "bootstrap",
            "smooth_df": 3.0,
            "smooth_log": False,
        },
        {"p": "NaN", "lambdas": None},
        options=("method", "smooth_df", "smooth_log"),
    ),
    # Replication: a mapping of weights by study, and options of
    # proportion_of_true_nulls; the collection and results were checked when they
    # were built.
    "cytepool.replication.differential_expression_replication": Entry(
        _replication_arguments,
        {"weights": None, "null_proportion": None},
        options=(
            "group",
            "contrast",
            "alpha",
            "method",
            "at_least",
            "conjunction_method",
            "meta_method",
        ),
        objects=("collection", "results"),
        keys={"weights": "A", "null_proportion": "lambdas"},
    ),
    # SumRank: NaN marks a gene a study did not test. n_permutations and n_genes go
    # through operator.index, which reads a 0-d masked integer.
    "cytepool.sumrank.sumrank": Entry(
        lambda c: {
            "statistics": [[1.0, 2.0, np.nan, 0.5], [2.0, 1.0, 3.0, -1.0]],
            "proportion_of_studies": 0.5,
            "null_statistics": [[[0.5, 1.0, 2.0, 1.5], [1.0, 2.0, 0.0, np.nan]]],
        },
        {"statistics": "NaN", "null_statistics": "NaN"},
        options=("proportion_of_studies",),
    ),
    "cytepool.sumrank.differential_expression_sumrank": Entry(
        _sumrank_arguments,
        {"n_permutations": None},
        options=("group", "contrast", "proportion_of_studies", "permute", "rng"),
        objects=("collection", "results"),
    ),
    "cytepool.held_out_prediction.held_out_prediction": Entry(
        _held_out_arguments,
        {"groups": None, "n_genes": None},
        options=("contrast", "variable", "case", "control", "proportion_of_studies"),
        objects=("collection", "results"),
    ),
    # Figures, each drawn into an Axes that pyplot does not keep. A variable of the
    # samples is an array of labels, or a name; the other inputs are objects.
    "cytepool.plotting.plot_mds": Entry(
        lambda c: {
            "data": log_cpm(COUNTS),
            "color_by": ["a", "b", "a", "b"],
            "marker_by": ["x", "x", "y", "y"],
            "labels": ["S1", "S2", "S3", "S4"],
            "top": 4,
            "gene_selection": "common",
            "dimensions": (1, 2),
            "ax": _axes(),
        },
        {
            "data": None,
            "color_by": None,
            "marker_by": None,
            "labels": None,
            "dimensions": None,
        },
        options=("top", "gene_selection"),
        objects=("ax",),
    ),
    "cytepool.plotting.plot_mds[cube]": Entry(
        lambda c: {
            "data": _bulk_cube(),
            "group": "bulk",
            "color_by": "condition",
            "min_cells": 1,
            "ax": _axes(),
        },
        {},
        options=("group", "color_by", "min_cells"),
        objects=("data", "ax"),
    ),
    "cytepool.plotting.plot_pca": Entry(
        lambda c: {
            "data": log_cpm(COUNTS),
            "color_by": ["a", "b", "a", "b"],
            "marker_by": ["x", "x", "y", "y"],
            "labels": ["S1", "S2", "S3", "S4"],
            "top": 4,
            "components": (1, 2),
            "ax": _axes(),
        },
        {
            "data": None,
            "color_by": None,
            "marker_by": None,
            "labels": None,
            "components": None,
        },
        options=("top",),
        objects=("ax",),
    ),
    "cytepool.plotting.plot_pca[cube]": Entry(
        lambda c: {
            "data": _bulk_cube(),
            "group": "bulk",
            "color_by": "condition",
            "min_cells": 1,
            "ax": _axes(),
        },
        {},
        options=("group", "color_by", "min_cells"),
        objects=("data", "ax"),
    ),
    # Genes, groups and samples named by id; the cube is an object.
    "cytepool.plotting.plot_dotplot": Entry(
        lambda c: {
            "cube": c.cube,
            "genes": ["feature_0", "feature_2"],
            "groups": ["T", "B"],
            "samples": ["S1", "S2"],
            "standard_scale": "gene",
            "mean_only_expressed": True,
            "vmin": 0.0,
            "vmax": 2.0,
            "ax": _axes(),
        },
        {"genes": None, "groups": None, "samples": None},
        options=("standard_scale", "mean_only_expressed", "vmin", "vmax"),
        objects=("cube", "ax"),
    ),
    "cytepool.plotting.plot_cell_numbers": Entry(
        lambda c: {"cube": c.cube, "min_cells": 1, "numbers": True, "ax": _axes()},
        {},
        options=("min_cells", "numbers"),
        objects=("cube", "ax"),
    ),
    "cytepool.plotting.plot_composition": Entry(
        lambda c: {"data": c.cube, "by": ["x", "y"], "ax": _axes()},
        {"by": None},
        objects=("data", "ax"),
    ),
    "cytepool.plotting.plot_composition[variable]": Entry(
        lambda c: {"data": c.cube, "by": "sample_id", "ax": _axes()},
        {},
        options=("by",),
    ),
    "cytepool.plotting.plot_proportions": Entry(
        lambda c: {
            "result": _composition_result(),
            "by": ["a", "b", "a", "b"],
            "contrast": "stim",
            "ax": _axes(),
        },
        {"by": None},
        options=("contrast",),
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_proportions[variable]": Entry(
        lambda c: {"result": _composition_result(), "by": "condition", "ax": _axes()},
        {},
        options=("by",),
    ),
    "cytepool.plotting.plot_mean_variance": Entry(
        lambda c: {"result": voom(COUNTS, DESIGN), "ax": _axes()},
        {},
        objects=("result", "ax"),
    ),
    # A group of a differential-expression result, by id.
    "cytepool.plotting.plot_mean_variance[result]": Entry(
        lambda c: {"result": _tested(c), "group": "bulk", "ax": _axes()},
        {},
        options=("group",),
    ),
    "cytepool.plotting.plot_residual_variances": Entry(
        lambda c: {"fit": moderated_statistics(c.fit), "ax": _axes()},
        {},
        objects=("fit", "ax"),
    ),
    "cytepool.plotting.plot_residual_variances[result]": Entry(
        lambda c: {"fit": _tested(c), "group": "bulk", "ax": _axes()},
        {},
        options=("group",),
    ),
    "cytepool.plotting.plot_sample_weights": Entry(
        lambda c: {
            "weights": [1.0, 0.5, 2.0, 1.0],
            "labels": ["a", "b", "c", "d"],
            "ax": _axes(),
        },
        {"weights": None, "labels": None},
        objects=("ax",),
    ),
    "cytepool.plotting.plot_sample_weights[result]": Entry(
        lambda c: {
            "weights": _tested(c, quality_weights=True),
            "group": "bulk",
            "ax": _axes(),
        },
        {},
        options=("group",),
        objects=("weights",),
    ),
    "cytepool.plotting.plot_md": Entry(
        lambda c: {
            "result": moderated_statistics(c.contrast_fit),
            "coefficient": "first",
            "alpha": 0.1,
            "ax": _axes(),
        },
        {},
        options=("coefficient", "alpha"),
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_volcano": Entry(
        lambda c: {
            "result": _tested(c).table("bulk", "stim"),
            "alpha": 0.1,
            "top": 3,
            "labels": [f"g{j}" for j in range(COUNTS.shape[1])],
            "ax": _axes(),
        },
        {"labels": None},
        options=("alpha", "top"),
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_volcano[fit]": Entry(
        lambda c: {
            "result": moderated_statistics(c.contrast_fit),
            "coefficient": 1,
            "ax": _axes(),
        },
        {},
        options=("coefficient",),
    ),
    "cytepool.plotting.plot_significant_genes": Entry(
        lambda c: {
            "result": _tested(c, adjust_across_groups=True),
            "contrast": "stim",
            "alpha": 0.1,
            "across_groups": True,
            "ax": _axes(),
        },
        {},
        options=("contrast", "alpha", "across_groups"),
        objects=("result", "ax"),
    ),
    # NaN marks a missing p-value.
    "cytepool.plotting.plot_p_values": Entry(
        lambda c: {"result": [0.1, 0.5, np.nan], "bins": 5, "ax": _axes()},
        {"result": "NaN"},
        options=("bins",),
        objects=("ax",),
    ),
    "cytepool.plotting.plot_p_values[fit]": Entry(
        lambda c: {
            "result": moderated_statistics(c.contrast_fit),
            "coefficient": "second",
            "ax": _axes(),
        },
        {},
        options=("coefficient",),
        objects=("result",),
    ),
    "cytepool.plotting.plot_barcode": Entry(
        lambda c: {
            "statistics": [0.5, -1.0, 2.0, 0.1, 3.0, -0.2, 1.1, 0.0],
            "gene_set": ["g1", "g2"],
            "gene_ids": [f"g{j}" for j in range(8)],
            "span": 0.5,
            "ax": _axes(),
        },
        {"statistics": None, "gene_set": None, "gene_ids": None},
        options=("span",),
        objects=("ax",),
    ),
    "cytepool.plotting.plot_gene_sets": Entry(
        lambda c: {
            "table": camera(**_gene_set_arguments()),
            "top": 5,
            "alpha": 0.1,
            "ax": _axes(),
        },
        {},
        options=("top", "alpha"),
        objects=("table", "ax"),
    ),
    "cytepool.plotting.plot_over_representation": Entry(
        lambda c: {
            "table": over_representation(
                ["g1", "g2"],
                {"set": ["g1", "g3"], "other": ["g4"]},
                universe=[f"g{j}" for j in range(COUNTS.shape[1])],
            ),
            "top": 5,
            "alpha": 0.1,
            "ax": _axes(),
        },
        {},
        options=("top", "alpha"),
        objects=("table", "ax"),
    ),
    "cytepool.plotting.plot_enrichment": Entry(
        lambda c: {
            "statistics": RANKED,
            "gene_set": ["g1", "g2"],
            "id_column": "feature_id",
            "statistic_column": "t",
            "exponent": 1.0,
            "ax": _axes(),
        },
        {"statistics": "NaN", "gene_set": None},
        options=("id_column", "statistic_column", "exponent"),
        objects=("ax",),
        keys={"statistics": "g0"},
    ),
    "cytepool.plotting.plot_enrichment[result]": Entry(
        lambda c: {
            "statistics": _tested(c),
            "gene_set": ["feature_1", "feature_2"],
            "group": "bulk",
            "contrast": "stim",
        },
        {},
        options=("group", "contrast"),
        objects=("statistics", "gene_set"),
    ),
    "cytepool.plotting.plot_gsea_table": Entry(
        lambda c: {
            "statistics": RANKED,
            "gene_sets": {"set": ["g1", "g2", "g5"], "other": ["g4"]},
            "table": gsea(
                RANKED,
                {"set": ["g1", "g2", "g5"], "other": ["g4"]},
                method="permutation",
                n_permutations=20,
                rng=1,
            ),
            "id_column": "feature_id",
            "statistic_column": "t",
            "top": 2,
            "exponent": 0.5,
            "ax": _axes(),
        },
        {"statistics": "NaN", "gene_sets": None},
        options=("id_column", "statistic_column", "top", "exponent"),
        objects=("table", "ax"),
        keys={"statistics": "g0", "gene_sets": "set"},
    ),
    "cytepool.plotting.plot_gsea_table[result]": Entry(
        lambda c: {
            "statistics": _tested(c),
            "gene_sets": {"set": ["feature_1", "feature_2"]},
            "table": gsea(
                _tested(c),
                {"set": ["feature_1", "feature_2"]},
                method="permutation",
                n_permutations=20,
                rng=1,
            ),
            "group": "bulk",
            "contrast": "stim",
        },
        {},
        options=("group", "contrast"),
        objects=("statistics", "gene_sets"),
    ),
    # The sets shown and the samples' labels.
    "cytepool.plotting.plot_pathway_scores": Entry(
        lambda c: {
            "scores": _pathway_scores(),
            "sets": ["other", "set"],
            "color_by": ["a", "b", "a", "b"],
            "standardize": True,
            "ax": _axes(),
        },
        {"sets": None, "color_by": None},
        options=("standardize",),
        objects=("scores", "ax"),
    ),
    # The sources shown and the profiles' labels.
    "cytepool.plotting.plot_activities": Entry(
        lambda c: {
            "activities": _activities(),
            "sources": ["B", "A"],
            "top": 2,
            "color_by": ["a", "b", "a", "b"],
            "standardize": True,
            "ax": _axes(),
        },
        {"sources": None, "color_by": None},
        options=("top", "standardize"),
        objects=("activities", "ax"),
    ),
    "cytepool.plotting.plot_top_activities": Entry(
        lambda c: {
            "activities": _activities(),
            "profile": "S2",
            "top": 2,
            "ax": _axes(),
        },
        {},
        options=("profile", "top"),
        objects=("activities", "ax"),
    ),
    "cytepool.plotting.plot_dispersions": Entry(
        lambda c: {
            "dispersions": deseq2_dispersions(
                COUNTS,
                DESIGN,
                size_factors=np.ones(4),
                trend="mean",
                genewise_dispersions=np.full(8, 0.1),
                trended_dispersions=np.full(8, 0.1),
                prior_variance=1.0,
            ),
            "ax": _axes(),
        },
        {},
        objects=("dispersions", "ax"),
    ),
    "cytepool.plotting.plot_quasi_dispersions": Entry(
        lambda c: {
            "fit": fit_quasi_likelihood(COUNTS, DESIGN, dispersion=0.1),
            "ax": _axes(),
        },
        {},
        objects=("fit", "ax"),
    ),
    "cytepool.plotting.plot_forest": Entry(
        lambda c: (
            _forest_arguments(c) | {"group": "bulk", "contrast": "stim", "ax": _axes()}
        ),
        {},
        options=("feature", "group", "contrast"),
        objects=("meta", "studies", "ax"),
    ),
    "cytepool.plotting.plot_study_agreement": Entry(
        lambda c: {
            "result": _replication(),
            "measure": "sign_agreement",
            "numbers": True,
            "ax": _axes(),
        },
        {},
        options=("measure", "numbers"),
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_replicated_genes": Entry(
        lambda c: {"result": _replication(), "ax": _axes()},
        {},
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_sumrank": Entry(
        lambda c: {
            "result": differential_expression_sumrank(*_studies(), "bulk", "stim"),
            "direction": "down",
            "alpha": 0.1,
            "n_labels": 2,
            "ax": _axes(),
        },
        {},
        options=("direction", "alpha", "n_labels"),
        objects=("result", "ax"),
    ),
    "cytepool.plotting.plot_held_out_auc": Entry(
        lambda c: {
            "prediction": held_out_prediction(
                *_studies(), "stim", "condition", "stim", "ctrl", n_genes=2
            ),
            "ax": _axes(),
        },
        {},
        objects=("prediction", "ax"),
    ),
    "cytepool.plotting.plot_variance_fractions": Entry(
        lambda c: {
            "data": _variance_partition(),
            "group": "bulk",
            "sort": False,
            "ax": _axes(),
        },
        {},
        options=("group", "sort"),
        objects=("data", "ax"),
    ),
    # Genes named by labels, one per gene.
    "cytepool.plotting.plot_gene_variance_fractions": Entry(
        lambda c: {
            "data": _variance_partition(),
            "genes": ["x1", "x2"],
            "group": "bulk",
            "labels": [f"x{j}" for j in range(COUNTS.shape[1])],
            "sort": False,
            "ax": _axes(),
        },
        {"genes": None, "labels": None},
        options=("group", "sort"),
        objects=("data", "ax"),
    ),
    # Scores of samples tested with a design: an array of values, with one id per
    # column and per row, weights per sample and a covariate of the trend.
    "cytepool.score_tests.limma": Entry(
        lambda c: {
            "data": log_cpm(COUNTS),
            "design": Design(
                SAMPLE_TABLE,
                "~ condition",
                levels=LEVELS,
                contrasts={"stim": [0.0, 1.0]},
            ),
            "contrast": "stim",
            "feature_ids": [f"g{j}" for j in range(COUNTS.shape[1])],
            "sample_ids": SAMPLE_TABLE["sample_id"],
            "weights": [1.0, 2.0, 1.0, 2.0],
            "trend": np.arange(COUNTS.shape[1], dtype=float),
            "robust": False,
        },
        {
            "data": "NaN",
            "feature_ids": None,
            "sample_ids": None,
            "weights": None,
            "trend": "NaN",
        },
        options=("contrast", "robust"),
        objects=("design",),
    ),
    # Every study of a collection analysed alike: the design's parts, as Design reads
    # them, and the analysis's options, as dream_voom reads them.
    "cytepool.study_results.analyse_each_study": Entry(
        lambda c: {
            "collection": _studies()[0],
            "formula": "~ condition",
            "analysis": dream_voom,
            "random": ["(1 | donor)"],
            "levels": LEVELS,
            "contrasts": {"stim": [0.0, 1.0]},
            "groups": ["bulk"],
        },
        {"random": None, "levels": None, "contrasts": None, "groups": None},
        options=("formula",),
        objects=("collection", "analysis"),
        keys={"levels": "condition", "contrasts": "stim"},
    ),
    # Variance partitioning: a mixed model, a linear model, and every group of a cube.
    "cytepool.variance_partitioning.variance_fractions": Entry(
        lambda c: {
            "y": log_cpm(COUNTS),
            "design": Design(SAMPLE_TABLE, "~ 1", random="(1 | donor)"),
            "weights": np.full(COUNTS.shape, 2.0),
            "method": "reml",
            "n_threads": 1,
        },
        {"y": None, "weights": None, "n_threads": None},
        options=("method",),
        objects=("design",),
    ),
    "cytepool.variance_partitioning.variance_fractions[linear]": Entry(
        lambda c: {
            "y": log_cpm(COUNTS),
            "design": Design(SAMPLE_TABLE, "~ condition", levels=LEVELS),
            "weights": np.full(4, 2.0),
        },
        {"y": None, "weights": None},
    ),
    "cytepool.variance_partitioning.variance_partitioning": Entry(
        _variance_partitioning,
        {
            "groups": None,
            "min_cells": None,
            "expression_filter": None,
            "n_threads": None,
        },
        options=("min_counts", "normalization", "span", "method"),
        objects=("cube", "design"),
        keys={"expression_filter": "min_count"},
    ),
    "cytepool.variance_partitioning.variance_partitioning[prior_weights]": Entry(
        _variance_partitioning_by_cells, {}, options=("prior_weights",)
    ),
}
# Every argument of the distribution and special functions is an array, with NaN
# for missing values; values outside a domain give NaN, so any numbers are valid.
for _module in (distributions, special):
    for _name in _module.__all__:
        _parameters = inspect.signature(getattr(_module, _name)).parameters
        ENTRY_POINTS[f"{_module.__name__}.{_name}"] = Entry(
            lambda c, names=tuple(_parameters): dict.fromkeys(names, [0.5, 0.25]),
            dict.fromkeys(_parameters, "NaN"),
        )

# Each with the arguments it takes, so that an argument added later is classified.
NO_ARRAY_INPUT = {
    "cytepool.table.Table.__getitem__": (("name",), "a column name, a str"),
    "cytepool.table.Table.__contains__": (("name",), "a column name, looked up"),
    "cytepool.table.Table.equals": (("other",), "a table, compared as it is"),
    "cytepool.table.Table.rename": (("mapping",), "column names, which are str"),
    "cytepool.cube.collection.CubeCollection.aligned_cube": (
        ("study",),
        "a study name, looked up",
    ),
    "cytepool.design.contrasts.Contrasts.__getitem__": (("name",), "a contrast name"),
    "cytepool.design.contrasts.Contrasts.__contains__": (("name",), "a contrast name"),
    "cytepool.design.contrasts.Contrasts.get": (
        ("name", "default"),
        "a contrast name, and a default returned as given",
    ),
    "cytepool.design.formula.parse_formula": (("formula",), "a formula, a str"),
    "cytepool.design.formula.parse_random_effects": (("random",), "a str"),
    "cytepool.design.formula.quote_name": (("name",), "a variable name, a str"),
    "cytepool.io.cube_files.read_cube": (("path",), "a path"),
    "cytepool.io.cube_files.write_cube": (
        ("cube", "path", "compression", "level", "overwrite"),
        "a Cube, whose arrays were checked when it was built, a path and options",
    ),
    "cytepool.io.anndata.to_anndata": (
        ("cube", "x_layer"),
        "a Cube, whose arrays were checked when it was built, and a layer name",
    ),
    "cytepool.io.anndata.to_mudata": (
        ("cube",),
        "a Cube, whose arrays were checked when it was built",
    ),
    "cytepool.io.zarr.ZarrArray": (
        ("path", "metadata", "stamp"),
        "built by open_array from a file's metadata",
    ),
    "cytepool.io.zarr.ZarrArray.__array__": (("dtype", "copy"), "a dtype and a flag"),
    "cytepool.io.zarr.check_compression": (
        ("compression", "level"),
        "a codec name and an integer level",
    ),
    "cytepool.io.zarr.open_array": (("path",), "a path"),
    "cytepool.io.zarr.read_group_attributes": (("path",), "a path"),
    "cytepool.io.zarr.write_group": (
        ("path", "attributes"),
        "a path, and attributes written as JSON, which cannot hold an array",
    ),
    "cytepool.differential_expression.DifferentialExpressionResult": (
        (
            "method",
            "contrasts",
            "summary",
            "samples",
            "tables",
            "provenance",
            "cube",
            "design",
        ),
        "a method name, Contrasts, Tables, a Cube and a Design, which were checked "
        "when they were built, and a record kept as JSON, which cannot hold an array",
    ),
    "cytepool.differential_expression.DifferentialExpressionResult.fit": (
        ("group",),
        "a group id, looked up",
    ),
    "cytepool.table.Table.sort": (
        ("by", "descending"),
        "a column name, looked up, and a flag",
    ),
    "cytepool.study_results.StudyResults": (
        ("collection", "results"),
        "a CubeCollection and results, which were checked when they were built",
    ),
    "cytepool.study_results.StudyResults.__getitem__": (
        ("study",),
        "a study name, looked up",
    ),
    "cytepool.study_results.StudyResults.table": (
        ("group", "contrast"),
        "a group id and a contrast name, looked up",
    ),
    "cytepool.study_results.StudyResults.matrix": (
        ("column", "group", "contrast"),
        "a column name, a group id and a contrast name, looked up",
    ),
    "cytepool.differential_expression.DifferentialExpressionResult.table": (
        ("group", "contrast"),
        "a group id and a contrast name, looked up",
    ),
    "cytepool.variance_partitioning.VariancePartitioningResult": (
        ("variables", "summary", "samples", "tables", "provenance"),
        "variable names, Tables, which were checked when they were built, and a "
        "record kept as JSON, which cannot hold an array",
    ),
    "cytepool.variance_partitioning.VariancePartitioningResult.table": (
        ("group",),
        "a group id, looked up",
    ),
    "cytepool.composition.CompositionResult": (
        (
            "contrasts",
            "tables",
            "f_test",
            "summary",
            "samples",
            "proportions",
            "provenance",
        ),
        "Contrasts and Tables, which were checked when they were built, and a record "
        "kept as JSON, which cannot hold an array",
    ),
    "cytepool.composition.CompositionResult.table": (
        ("contrast",),
        "a contrast name, looked up",
    ),
    "cytepool.meta_analysis.MetaAnalysisResult": (
        ("summary", "studies", "tables", "provenance"),
        "Tables, which were checked when they were built, and a record kept as JSON, "
        "which cannot hold an array",
    ),
    "cytepool.meta_analysis.MetaAnalysisResult.table": (
        ("group", "contrast"),
        "a group id and a contrast name, looked up",
    ),
    "cytepool.replication.ReplicationResult": (
        ("genes", "study_pairs", "found_in", "studies", "provenance"),
        "Tables, which were checked when they were built, and a record kept as JSON, "
        "which cannot hold an array",
    ),
    "cytepool.sumrank.SumRankResult": (
        ("table", "studies", "provenance"),
        "Tables, which were checked when they were built, and a record kept as JSON, "
        "which cannot hold an array",
    ),
    "cytepool.held_out_prediction.HeldOutPrediction": (
        ("auc", "scores", "genes", "provenance"),
        "Tables, which were checked when they were built, and a record kept as JSON, "
        "which cannot hold an array",
    ),
    "cytepool.meta_analysis.align_study_results": (
        ("collection", "results", "group", "contrast"),
        "a CubeCollection and results, which were checked when they were built, and a "
        "group id and a contrast name, looked up",
    ),
    "cytepool.gene_sets.GeneSets.__getitem__": (("name",), "a set name, looked up"),
    "cytepool.gene_sets.GeneSets.__contains__": (("name",), "a set name, looked up"),
    "cytepool.gene_sets.read_gmt": (
        ("path", "source", "version", "licence"),
        "a path, and text that the collection keeps",
    ),
    "cytepool.networks.Network.__getitem__": (("source",), "a source, looked up"),
    "cytepool.networks.Network.__contains__": (("source",), "a source, looked up"),
    "cytepool.networks.collectri": (
        ("licence", "cache_dir"),
        "a licence level and a directory",
    ),
    "cytepool.networks.progeny": (
        ("organism", "top", "max_p_value", "cache_dir"),
        "an organism, a number of genes, a p-value and a directory",
    ),
}

# Result containers: they store their fields as given, so a masked field keeps its
# mask. One that starts converting its fields must become an entry point.
RECORDS = {
    "cytepool.design.checks.DesignIssue",
    "cytepool.design.checks.DesignReport",
    "cytepool.design.formula.FormulaTerms",
    "cytepool.design.formula.RandomEffect",
    "cytepool.stats.empirical_bayes.FDistributionFit",
    "cytepool.stats.empirical_bayes.LikelihoodFDistributionFit",
    "cytepool.stats.empirical_bayes.RobustFDistributionFit",
    "cytepool.stats.empirical_bayes.SqueezedVariances",
    "cytepool.stats.empirical_bayes.ModeratedFit",
    "cytepool.stats.empirical_bayes.TreatFit",
    "cytepool.stats.negative_binomial_models.NegativeBinomialFit",
    "cytepool.stats.deseq2.DESeq2Dispersions",
    "cytepool.stats.deseq2.DESeq2Fit",
    "cytepool.stats.deseq2.DESeq2Results",
    "cytepool.stats.mixed_models.FixedEffectFTest",
    "cytepool.stats.mixed_models.FixedEffectTests",
    "cytepool.stats.mixed_models.MixedModelFit",
    "cytepool.meta_analysis.MetaAnalysisFit",
    "cytepool.stats.combined_p_values.CombinedPValues",
    "cytepool.stats.multiple_testing.TrueNullProportion",
    "cytepool.sumrank.SumRankFit",
    "cytepool.principal_components.PrincipalComponents",
    "cytepool.stats.dream.ModeratedFixedEffectTests",
    "cytepool.stats.quasi_likelihood.QuasiLikelihoodFit",
    "cytepool.stats.quasi_likelihood.QuasiLikelihoodTest",
    "cytepool.stats.quasi_likelihood.UnitDevianceMoments",
    "cytepool.variance_partitioning.VarianceFractions",
    "cytepool.differential_expression.GroupFit",
    "cytepool.pathway_scores.PathwayScores",
    "cytepool.activity.Activities",
}

# Callables that modules with __all__ define but do not export.
NOT_EXPORTED = {
    "cytepool.design.checks.check_design": "takes a Design; Design.check is public",
}
COMPILED_SCALAR = "_scalar"  # the compiled routine behind each distribution function

# Methods that take arguments but need no entry of their own.
NOT_INPUT_METHODS = {
    "__init__": "the class stands for its constructor",
    "__new__": "the class stands for its constructor",
    "__post_init__": "part of the constructor",
    "__replace__": "builds a new object through the constructor",
    "__setattr__": "refuses every change: the objects are immutable",
    "__delattr__": "refuses every change: the objects are immutable",
    "__deepcopy__": "returns the object, whose values were checked",
    "__eq__": "compares, converting nothing",
}


# The inventory.


def _public_names(module: ModuleType) -> list[str]:
    """``__all__``, or else the public functions and classes a module defines.

    This is the rule tests/test_documentation.py uses for the API reference.
    """
    if hasattr(module, "__all__"):
        return list(module.__all__)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and value.__module__ == module.__name__
    ]


def _modules() -> Iterator[ModuleType]:
    """Every module whose name has no private component."""
    for info in pkgutil.walk_packages(cytepool.__path__, "cytepool."):
        if not any(part.startswith("_") for part in info.name.split(".")):
            yield importlib.import_module(info.name)


def _own(value: Any) -> bool:
    return getattr(value, "__module__", "").startswith("cytepool.")


@dataclass(frozen=True)
class Public:
    """A public callable: a function, a class, or the function of a method.

    Attributes:
        function: The function, or the class.
        parameters: Its parameters as a caller passes them, without ``self`` or
            ``cls``.
    """

    function: Any
    parameters: tuple[inspect.Parameter, ...]


def _methods(cls: type) -> dict[str, Public]:
    """The methods of a class, and of its CytePool bases, that take arguments.

    A method counts when its name is public or it is a protocol (a dunder) other
    than those in ``NOT_INPUT_METHODS``.
    """
    found: dict[str, Public] = {}
    seen: set[str] = set()
    for base in cls.__mro__:
        if not _own(base):
            continue
        for attribute, member in vars(base).items():
            if attribute in seen:
                continue  # a subclass defines it again, which is what callers get
            seen.add(attribute)
            dunder = attribute.startswith("__") and attribute.endswith("__")
            if (attribute.startswith("_") and not dunder) or (
                attribute in NOT_INPUT_METHODS
            ):
                continue
            function = getattr(member, "__func__", member)
            if not inspect.isfunction(function):
                continue
            parameters = tuple(inspect.signature(function).parameters.values())
            if not isinstance(member, staticmethod):
                parameters = parameters[1:]  # self, or cls
            if parameters:
                found[attribute] = Public(function, parameters)
    return found


def _public_callables() -> dict[str, Public]:
    """Every public function, class and method that takes arguments, by full name."""
    found: dict[str, Public] = {}
    classes = []
    for module in _modules():
        for name in _public_names(module):
            value = getattr(module, name)
            if not callable(value):
                continue  # a constant
            full = f"{value.__module__}.{value.__qualname__}"
            if inspect.isclass(value):
                classes.append((full, value))
                parameters = tuple(inspect.signature(value).parameters.values())
                if parameters:
                    found[full] = Public(value, parameters)
                for method, public in _methods(value).items():
                    found[f"{full}.{method}"] = public
            else:
                parameters = tuple(inspect.signature(value).parameters.values())
                found[full] = Public(value, parameters)
    public_classes = {cls for _, cls in classes}
    for full, cls in classes:
        for name, public in _indexers(cls, public_classes).items():
            found[f"{full}.{name}"] = public
    return found


def _indexers(cls: type, public_classes: set[type]) -> dict[str, Public]:
    """The methods of the objects that public properties of a class return.

    An object of a CytePool class that is not public, as an indexer is, takes its
    arguments through its own methods, named here as ``property.method``. The
    class is read from the return annotation of the property.
    """
    found: dict[str, Public] = {}
    for attribute in dir(cls):
        member = inspect.getattr_static(cls, attribute)
        if attribute.startswith("_") or not isinstance(member, property):
            continue
        returned = typing.get_type_hints(member.fget).get("return")
        if inspect.isclass(returned) and _own(returned):
            if returned not in public_classes:
                for method, public in _methods(returned).items():
                    found[f"{attribute}.{method}"] = public
    return found


PUBLIC = _public_callables()


def _callable_name(entry_name: str) -> str:
    """The callable an entry calls: its name without a variant such as ``[ids]``."""
    return entry_name.partition("[")[0]


def _classified(callable_name: str) -> set[str]:
    """The arguments the entries of a callable classify, over all its variants."""
    return {
        argument
        for name, entry in ENTRY_POINTS.items()
        if _callable_name(name) == callable_name
        for argument in (*entry.arrays, *entry.options, *entry.objects)
    }


def _call(name: str, context: Context, arguments: dict[str, Any]) -> Any:
    """Call an entry point with keyword arguments, positional-only ones by position."""
    entry = ENTRY_POINTS[name]
    target = PUBLIC[_callable_name(name)].function
    if entry.instance is not None:
        target = getattr(entry.instance(context), target.__name__)
    arguments = dict(arguments)
    positional = [
        arguments.pop(parameter.name)
        for parameter in inspect.signature(target).parameters.values()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        and parameter.name in arguments
    ]
    return target(*positional, **arguments)


def _masked(value: Any) -> np.ma.MaskedArray:
    """``value`` as a masked array with its first entry masked."""
    try:
        array = np.asarray(value)
    except ValueError:  # ragged
        array = np.empty(len(value), dtype=object)
        array[:] = value
    mask = np.zeros(array.shape, dtype=bool)
    mask.flat[0] = True
    return np.ma.array(array, mask=mask)


def _with_masked_constant(value: Any) -> tuple[Any, str]:
    """``value`` with ``numpy.ma.masked`` as its first leaf, and where that sits.

    Arrays become nested lists and tuples stay tuples; a scalar or a string is
    replaced as a whole.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple) and value:
        first, where = _with_masked_constant(value[0])
        changed = [first, *value[1:]]
        return tuple(changed) if isinstance(value, tuple) else changed, f"[0]{where}"
    return np.ma.masked, ""


def _replaced(
    entry: Entry, arguments: dict[str, Any], argument: str, change: Callable
) -> tuple[dict[str, Any], str]:
    """The arguments with one argument changed, and the subject messages name.

    For a mapping argument, the value of its key is changed.
    """
    arguments = dict(arguments)
    subject = entry.names.get(argument, argument)
    if argument in entry.keys:
        key = entry.keys[argument]
        changed, where = change(arguments[argument][key])
        arguments[argument] = {**arguments[argument], key: changed}
        return arguments, f"{subject}[{key!r}]{where}"
    arguments[argument], where = change(arguments[argument])
    return arguments, f"{subject}{where}"


ARRAY_ARGUMENTS = [
    pytest.param(name, argument, id=f"{name}:{argument}")
    for name, entry in ENTRY_POINTS.items()
    for argument in entry.arrays
]


# The inventory is complete.


def test_every_public_callable_is_in_one_group() -> None:
    entry_points = {_callable_name(name) for name in ENTRY_POINTS}
    groups = [entry_points, set(NO_ARRAY_INPUT), RECORDS]

    for first, second in itertools.combinations(groups, 2):
        assert first & second == set()
    assert set(PUBLIC) - set().union(*groups) == set(), "classify these callables"
    assert set().union(*groups) - set(PUBLIC) == set(), "these are not public"
    # The rule finds methods, protocols and constructors, and leaves out what takes
    # no arguments.
    assert {
        "cytepool.table.Table.with_columns",
        "cytepool.io.zarr.ZarrArray.__getitem__",
        "cytepool.stats.voom.VoomResult",
    } <= set(PUBLIC)
    assert not {"cytepool.cube.cube.Cube.to_memory", "cytepool.table.Table.to_dict"} & (
        set(PUBLIC)
    )


class _Indexer:
    """Reads an array by positions, as an object that a property returns can."""

    def __getitem__(self, key: Any) -> None: ...

    @classmethod
    def from_positions(cls, positions: Any) -> "_Indexer": ...


class _Base:
    def scale(self, factor: Any) -> None: ...

    def shift(self, offset: Any) -> None: ...


class _Array(_Base):
    @property
    def by_position(self) -> _Indexer: ...

    @property
    def size(self) -> int: ...

    @staticmethod
    def zeros(shape: Any) -> None: ...

    def __add__(self, other: Any) -> None: ...

    def __eq__(self, other: object) -> bool: ...

    def copy(self) -> "_Array": ...

    def scale(self) -> None: ...  # hides the method of the base, which takes one


def test_the_rule_finds_every_way_a_class_takes_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kinds of methods a class added later may have: inherited, static, class
    # and protocol methods, and an indexer that a property returns.
    for cls in (_Indexer, _Base, _Array):
        monkeypatch.setattr(cls, "__module__", "cytepool.example")

    methods = _methods(_Array)
    assert set(methods) == {"shift", "zeros", "__add__"}
    assert set(_methods(_Base)) == {"scale", "shift"}
    assert [p.name for p in methods["zeros"].parameters] == ["shape"]
    indexers = _indexers(_Array, public_classes=set())
    assert set(indexers) == {"by_position.__getitem__", "by_position.from_positions"}
    # Neither self nor cls is an argument.
    assert [p.name for p in indexers["by_position.from_positions"].parameters] == [
        "positions"
    ]
    # A public class that a property returns is in the inventory on its own.
    assert _indexers(_Array, public_classes={_Indexer}) == {}


def test_callables_without_array_input_take_the_arguments_listed() -> None:
    for name, (arguments, _) in NO_ARRAY_INPUT.items():
        assert tuple(p.name for p in PUBLIC[name].parameters) == arguments, name


def test_every_callable_a_module_defines_is_exported_or_listed() -> None:
    # A public function that a module with __all__ leaves out would escape both the
    # API reference and this inventory.
    unexported = {}
    for module in _modules():
        if not hasattr(module, "__all__"):
            continue
        for name, value in vars(module).items():
            if (
                not name.startswith("_")
                and callable(value)
                and getattr(value, "__module__", None) == module.__name__
                and name not in module.__all__
            ):
                unexported[f"{module.__name__}.{name}"] = value
    # Compiled routines, which numba refuses masked arrays for (tested below).
    compiled = {name for name, value in unexported.items() if hasattr(value, "py_func")}

    assert set(unexported) - compiled == set(NOT_EXPORTED)
    assert all(name.endswith(COMPILED_SCALAR) for name in compiled)


def test_compiled_scalar_routines_refuse_masked_arguments() -> None:
    # They are called from compiled code; numba cannot type a masked array.
    for value in (np.ma.array(1.0, mask=True), np.ma.masked):
        with pytest.raises(TypingError, match="Unsupported array type"):
            t_sf_scalar(value, 3.0)


@pytest.mark.parametrize("name", sorted({_callable_name(n) for n in ENTRY_POINTS}))
def test_every_argument_of_an_entry_point_is_classified(name: str) -> None:
    parameters = PUBLIC[name].parameters
    named = {p.name for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD}
    for entry_name, entry in ENTRY_POINTS.items():
        if _callable_name(entry_name) == name:
            given = [*entry.arrays, *entry.options, *entry.objects]
            assert len(given) == len(set(given)), "an argument is classified twice"
            assert set(entry.keys) | set(entry.names) <= set(entry.arrays)
    classified = _classified(name)

    assert named <= classified
    extra = classified - named
    # Only keyword arguments of a **parameter are named by their keys.
    takes_keywords = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    assert bool(extra) == takes_keywords, extra


# Every array argument refuses masked arrays.


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_the_valid_call_of_each_entry_point_succeeds(
    name: str, context: Context
) -> None:
    entry = ENTRY_POINTS[name]
    arguments = entry.valid(context)

    assert set(arguments) <= _classified(_callable_name(name))
    # The masked calls below differ from this call in one argument only.
    _call(name, context, arguments)


@pytest.mark.parametrize(("name", "argument"), ARRAY_ARGUMENTS)
def test_every_array_argument_refuses_masked_arrays(
    name: str, argument: str, context: Context
) -> None:
    entry = ENTRY_POINTS[name]
    missing = entry.arrays[argument]
    # The data under the mask are the valid values, so a call that dropped the mask
    # would succeed.
    arguments, subject = _replaced(
        entry, entry.valid(context), argument, lambda value: (_masked(value), "")
    )
    with pytest.raises(TypeError, match=message(subject, "array", missing)):
        _call(name, context, arguments)

    arguments, subject = _replaced(
        entry, entry.valid(context), argument, _with_masked_constant
    )
    with pytest.raises(TypeError, match=message(subject, "constant", missing)):
        _call(name, context, arguments)


NON_ARRAY_ARGUMENTS = [
    pytest.param(name, argument, id=f"{name}:{argument}")
    for name, entry in ENTRY_POINTS.items()
    for argument in (*entry.options, *entry.objects)
]


@pytest.mark.parametrize(("name", "argument"), NON_ARRAY_ARGUMENTS)
def test_other_arguments_refuse_masked_values_or_keep_them_as_given(
    name: str, argument: str, context: Context
) -> None:
    # A number, bool or string read through its truth value, a comparison or
    # operator.index would give a masked value's hidden data, or none at all.
    entry = ENTRY_POINTS[name]
    arguments = entry.valid(context)
    masked_values: list[Any] = [np.ma.masked]
    if argument in entry.options:
        default = {p.name: p.default for p in PUBLIC[_callable_name(name)].parameters}
        value = arguments.get(argument, default[argument])
        assert isinstance(value, bool | int | float | str), "give the option a value"
        masked_values.append(np.ma.array(value, mask=True))

    for masked in masked_values:
        try:
            result = _call(name, context, {**arguments, argument: masked})
        except (TypeError, ValueError, KeyError):
            continue
        # Only a constructor that stores the argument untouched may accept it.
        assert getattr(result, argument) is masked


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_records_store_masked_fields_as_given(name: str) -> None:
    fields = {
        parameter.name: np.ma.array([1.0, 2.0], mask=[False, True])
        for parameter in PUBLIC[name].parameters
    }
    record = PUBLIC[name].function(**fields)

    for parameter, value in fields.items():
        assert getattr(record, parameter) is value


# The two defects this check was written for, and their remedies.


def test_a_masked_sample_label_is_refused_and_dropping_its_cells_works() -> None:
    samples = np.ma.array(["s1", "s1", "hidden_sample", "s2"], mask=[0, 0, 1, 0])
    with pytest.raises(TypeError, match=message("samples")):
        from_cells(CELL_COUNTS, samples=samples, groups=CELL_GROUPS)

    keep = ~np.ma.getmaskarray(samples)
    cube = from_cells(
        CELL_COUNTS[keep],
        samples=samples.compressed(),
        groups=np.array(CELL_GROUPS)[keep],
    )
    assert cube.samples["sample_id"].tolist() == ["s1", "s2"]
    # Cells 0 and 1 are s1 (groups B and T), cell 3 is s2 (group T).
    assert_array_equal(cube.layer("counts", "B"), [[1, 0, 2], [0, 0, 0]])
    assert_array_equal(cube.layer("counts", "T"), [[3, 1, 0], [4, 0, 1]])


def test_a_masked_table_column_is_refused_and_tolist_marks_it_missing() -> None:
    values = np.ma.array([1.0, 2.0], mask=[0, 1])
    with pytest.raises(TypeError, match=message("data['x']", missing="None or NaN")):
        Table({"x": values})

    assert_array_equal(Table({"x": values.tolist()})["x"], [1.0, np.nan])
    text = np.ma.array(["a", "b"], mask=[1, 0])
    assert Table({"t": text.tolist()})["t"].tolist() == [None, "b"]


def test_the_nan_remedy_marks_masked_p_values_missing() -> None:
    p = np.ma.array([0.01, 0.04, 0.03, 0.2], mask=[0, 1, 0, 0])
    filled = np.ma.filled(p.astype(float), np.nan)

    adjusted = adjust_p_values(filled)
    assert np.isnan(adjusted[1])
    # Missing p-values leave the family: the others are adjusted as three.
    assert_array_equal(adjusted[~p.mask], adjust_p_values(p.compressed()))


def test_ids_take_no_missing_values_and_other_columns_take_none_or_nan() -> None:
    layers = {"counts": np.ones((2, 2, 3), dtype=np.uint32)}
    axes = {"groups": CUBE_AXES["groups"], "features": CUBE_AXES["features"]}
    ids = ["S1", "S2"]
    with pytest.raises(TypeError, match=message("samples['sample_id']")):
        Cube(layers, samples={"sample_id": _masked(ids), "age": [1.0, 2.0]}, **axes)
    age = _masked([30.0, 40.0])
    with pytest.raises(
        TypeError, match=message("samples['age']", missing="None or NaN")
    ):
        Cube(layers, samples={"sample_id": ids, "age": age}, **axes)

    cube = Cube(layers, samples={"sample_id": ids, "age": age.tolist()}, **axes)
    assert_array_equal(cube.samples["age"], [np.nan, 40.0])


def test_provenance_marks_missing_values_with_none() -> None:
    values = _masked([1.0, 2.0])
    with pytest.raises(TypeError, match=message("provenance['x']", missing="None")):
        Cube(
            {"counts": np.ones((2, 2, 3), dtype=np.uint32)},
            **CUBE_AXES,
            provenance={"x": values},
        )

    cube = Cube(
        {"counts": np.ones((2, 2, 3), dtype=np.uint32)},
        **CUBE_AXES,
        provenance={"x": values.tolist()},
    )
    assert cube.provenance["x"] == [None, 2.0]


def test_an_array_with_nothing_masked_is_refused_and_getdata_reads_it() -> None:
    # netCDF4 returns such arrays by default.
    values = np.ma.array(COUNTS)
    with pytest.raises(TypeError, match=message("counts", "unmasked")):
        from_bulk(values)

    cube = from_bulk(np.ma.getdata(values))
    assert_array_equal(cube.layer("counts", "bulk"), COUNTS)


# Forms of input that the inventory's calls do not reach.


class _ConvertsToMasked:
    """An array-like that converts to a masked array, as a netCDF4 variable does."""

    def __init__(self, values: Any) -> None:
        self.values = _masked(values)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return self.values


ARRAY_LIKE_CALLS = [
    pytest.param(adjust_p_values, [0.01, 0.5, 0.02], "p", "NaN", id="p-values"),
    pytest.param(norm_cdf, [0.5, 1.5], "x", "NaN", id="distribution"),
    pytest.param(cpm, COUNTS, "counts", None, id="normalization counts"),
    pytest.param(
        lambda v: cpm(COUNTS, library_sizes=v),
        LIBRARY_SIZES,
        "library_sizes",
        None,
        id="library sizes",
    ),
    pytest.param(
        lambda v: fit_linear_model(v, DESIGN), np.ones((4, 3)), "y", "NaN", id="y"
    ),
    pytest.param(
        lambda v: fit_linear_model(np.ones((4, 3)), v),
        DESIGN,
        "design",
        None,
        id="design",
    ),
    pytest.param(from_bulk, COUNTS, "counts", None, id="bulk counts"),
    pytest.param(
        lambda v: from_cells(v, samples=CELL_SAMPLES, groups=CELL_GROUPS),
        CELL_COUNTS,
        "counts",
        None,
        id="cell counts",
    ),
    pytest.param(
        lambda v: from_cells(CELL_COUNTS, samples=v, groups=CELL_GROUPS),
        CELL_SAMPLES,
        "samples",
        None,
        id="cell labels",
    ),
    pytest.param(
        lambda v: squeeze_variances(v, 3.0),
        [1.0, 2.0, 0.5],
        "variances",
        "NaN",
        id="variances",
    ),
    pytest.param(
        lambda v: lowess(v, np.arange(5.0)), np.arange(5.0), "x", None, id="lowess"
    ),
    pytest.param(
        lambda v: Table({"x": v}),
        [1.0, 2.0],
        "column 'x'",
        "None or NaN",
        id="table column",
    ),
    pytest.param(
        lambda v: Cube({"counts": v}, **CUBE_AXES),
        np.ones((2, 2, 3), dtype=np.uint32),
        "layers['counts']",
        None,
        id="cube layer",
    ),
]


@pytest.mark.parametrize(("call", "valid", "subject", "missing"), ARRAY_LIKE_CALLS)
def test_array_likes_that_convert_to_masked_arrays_are_refused(
    call: Callable[[Any], Any], valid: Any, subject: str, missing: str | None
) -> None:
    call(valid)
    with pytest.raises(TypeError, match=message(subject, "converted", missing)):
        call(_ConvertsToMasked(valid))


def test_pandas_inputs_keep_their_missing_values() -> None:
    pandas = pytest.importorskip("pandas")
    # pandas turns a masked column into NaN, so the mask is not lost.
    frame = pandas.DataFrame({"x": np.ma.array([1.0, 2.0], mask=[0, 1])})
    assert_array_equal(Table(frame)["x"], [1.0, np.nan])
    labels = pandas.Categorical(CELL_SAMPLES, categories=["S2", "S1"])
    cube = from_cells(CELL_COUNTS, samples=labels, groups=pandas.Series(CELL_GROUPS))
    assert cube.samples["sample_id"].tolist() == ["S1", "S2"]
    design = Design(pandas.DataFrame(SAMPLE_TABLE), "~ condition")
    assert design.columns == COLUMNS


class _LazyRows:
    """An array-like that reads its rows, or groups, through ``wrap``."""

    def __init__(self, values: np.ndarray, wrap: Callable[[np.ndarray], Any]) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.wrap = wrap

    def __getitem__(self, key: Any) -> Any:
        return self.wrap(self.values[key])


@pytest.mark.parametrize(
    ("wrap", "found"),
    [(_masked, "array"), (np.ma.array, "unmasked"), (_ConvertsToMasked, "converted")],
)
def test_values_read_from_a_lazy_source_are_checked(
    wrap: Callable[[np.ndarray], Any], found: str, tmp_path: Path
) -> None:
    counts = _LazyRows(np.ones((2, 2, 3), dtype=np.uint32), wrap)
    cube = Cube({"counts": counts}, **CUBE_AXES)
    with pytest.raises(
        TypeError, match=message("group 'T' of the lazy layer 'counts'", found)
    ):
        cube.layer("counts", "T")

    rows = _LazyRows(np.ones((2, 3)), wrap)
    with pytest.raises(TypeError, match=message("data[0:1]", found)):
        write_array(tmp_path / "rows", rows, chunks=(1, 3))
    # The array is not written, since its metadata comes last.
    with pytest.raises(FileNotFoundError):
        open_array(tmp_path / "rows")


def test_masked_attributes_of_duck_typed_arguments_are_refused(
    context: Context,
) -> None:
    masked = _masked
    csr = {
        "data": CELL_COUNTS[CELL_COUNTS > 0],
        "indices": np.nonzero(CELL_COUNTS)[1],
        "indptr": np.concatenate([[0], np.cumsum((CELL_COUNTS > 0).sum(axis=1))]),
        "shape": CELL_COUNTS.shape,
    }
    named = {"matrix": np.eye(2), "names": ("u", "v"), "columns": ("a", "b")}
    calls: list[tuple[str, Callable[[], Any]]] = []
    for part in ("data", "indices", "indptr"):
        counts = SimpleNamespace(**{**csr, part: masked(csr[part])})
        calls.append(
            (
                f"counts.{part}",
                lambda counts=counts: from_cells(
                    counts, samples=CELL_SAMPLES, groups=CELL_GROUPS
                ),
            )
        )
    for part in ("matrix", "columns"):
        design = SimpleNamespace(matrix=DESIGN, columns=["a", "b"])
        setattr(design, part, masked(getattr(design, part)))
        calls.append(
            (
                f"design.{part}",
                lambda design=design: fit_linear_model(context.expression, design),
            )
        )
    calls.append(
        (
            "design.matrix",
            lambda: filter_by_expression(
                COUNTS, design=SimpleNamespace(matrix=masked(DESIGN))
            ),
        )
    )
    for part in named:
        contrasts = SimpleNamespace(**{**named, part: masked(named[part])})
        calls.append(
            (
                f"contrasts.{part}",
                lambda contrasts=contrasts: make_contrasts(contrasts, ("a", "b")),
            )
        )
        calls.append(
            (
                f"contrasts.{part}",
                lambda contrasts=contrasts: contrasts_fit(
                    fit_linear_model(
                        context.expression, DESIGN, coefficient_names=["a", "b"]
                    ),
                    contrasts,
                ),
            )
        )
    calls.append(
        (
            "columns.columns",
            lambda: make_contrasts(
                ["b"], SimpleNamespace(columns=masked(np.array(["a", "b"])))
            ),
        )
    )

    for subject, call in calls:
        with pytest.raises(TypeError, match=message(subject)):
            call()


def _object_array_holding(value: Any) -> np.ndarray:
    array = np.empty((), dtype=object)
    array[()] = value
    return array


@pytest.mark.parametrize(
    ("value", "subject", "found"),
    [
        # An object array, searched by position.
        (np.array(["a", np.ma.masked], dtype=object), "x[1]", "constant"),
        (
            np.array([[1, 2], [3, np.ma.array([4], mask=[True])]], dtype=object),
            "x[1, 1]",
            "array",
        ),
        (_object_array_holding(np.ma.masked), "x[()]", "constant"),
        # Nested sequences and mappings; a numpy string key is named as its text.
        ((1, (2, [3, np.ma.masked])), "x[1][1][1]", "constant"),
        ([{"a": 1}, {np.str_("b"): np.ma.masked}], "x[1]['b']", "constant"),
        # A bound of a slice, which Python reads through __index__.
        ((0, slice(1, np.ma.array(3, mask=True))), "x[1].stop", "array"),
        # A 0-d masked array, and one with nothing masked.
        (np.ma.array(5, mask=True), "x", "array"),
        (np.ma.array([1.0, 2.0]), "x", "unmasked"),
    ],
)
def test_masked_arrays_are_found_inside_containers(
    value: Any, subject: str, found: str
) -> None:
    with pytest.raises(TypeError, match=message(subject, found, "NaN")):
        reject_masked(value, "x", missing="NaN")


def test_containers_without_masked_arrays_pass() -> None:
    nested = [1, (2.0, "a", None), {"b": [np.arange(3), slice(1, 4, 2)]}, np.array([5])]
    reject_masked(nested, "x")
    reject_masked(np.array([np.arange(2), "c"], dtype=object), "x")


def test_a_masked_entry_anywhere_in_a_long_list_is_found() -> None:
    labels = [f"sample{i}" for i in range(1000)] + [np.ma.masked]
    with pytest.raises(TypeError, match=message("samples[1000]", "constant")):
        from_bulk(np.ones((1001, 2)), samples=labels)
    cube = from_bulk(np.ones((1000, 2)), samples=labels[:-1])
    assert cube.n_samples == 1000


def test_attributes_hold_no_arrays_so_no_masked_ones(tmp_path: Path) -> None:
    # Attributes are JSON, which holds no arrays, masked or not.
    for value in (np.ma.array([1.0], mask=[True]), np.array([1.0])):
        with pytest.raises(TypeError, match="not JSON serializable"):
            write_group(tmp_path / "group", {"x": value})


def test_orthogonal_selections_refuse_masked_positions_and_masks(
    context: Context,
) -> None:
    # numpy.asarray would read the hidden position 2, and the hidden True of the mask.
    positions = np.ma.array([0, 2], mask=[False, True])
    mask = np.ma.array([True, False, True, False], mask=[False, False, True, False])
    for key, subject, found in [
        ((positions, slice(None)), "index[0]", "array"),
        ((0, mask), "index[1]", "array"),
        (([0, np.ma.masked], 1), "index[0][1]", "constant"),
        ((..., slice(np.ma.array(1, mask=True), None)), "index[1].start", "array"),
    ]:
        with pytest.raises(TypeError, match=message(subject, found)):
            context.zarr_array.oindex[key]
    # The same selections without masks are read.
    assert_array_equal(
        context.zarr_array.oindex[[0], [True, False, False, True]], [[0.0, 3.0]]
    )
