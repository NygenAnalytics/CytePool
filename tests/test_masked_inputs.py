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
from cytepool.cube import Cube, from_bulk, from_cells
from cytepool.design import Design, make_contrasts
from cytepool.io.zarr import open_array, write_array, write_group
from cytepool.normalization import cpm, filter_by_expression, log_cpm
from cytepool.stats import distributions, special
from cytepool.stats.distributions import norm_cdf, t_sf_scalar
from cytepool.stats.empirical_bayes import squeeze_variances
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.smoothing import lowess
from cytepool.stats.voom import voom
from cytepool.table import Table

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
    }


def _normalization(**options: Any) -> Callable[[Context], dict[str, Any]]:
    return lambda context: {"counts": COUNTS, **options}


TABLE = Table({"x": [1.0, 2.0], "s": ["a", "b"]})
NORM_FACTORS = [1.0, 1.1, 0.9, 1.0]
LIBRARY_SIZES = COUNTS.sum(axis=1).astype(float)

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
    "cytepool.table.Table.filter": Entry(
        lambda c: {"mask": [True, False]}, {"mask": None}, instance=lambda c: TABLE
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
    "cytepool.cube.cube.Cube.regroup": Entry(
        lambda c: {"mapping": {"all": ["B", "T"]}},
        {"mapping": None},
        instance=lambda c: c.cube,
        keys={"mapping": "all"},
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
        },
        options=("target_sum",),
        keys={
            "sample_table": "sample_id",
            "group_table": "group_id",
            "features": "feature_id",
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
        },
        {"counts": None, "samples": None, "features": None},
        keys={"samples": "sample_id", "features": "feature_id"},
    ),
    "cytepool.cube.pool.from_bulk[ids]": Entry(
        lambda c: {
            "counts": COUNTS,
            "samples": SAMPLE_TABLE["sample_id"],
            "features": [f"g{i}" for i in range(8)],
        },
        {"samples": None, "features": None},
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
        },
        {
            "y": "NaN",
            "design": None,
            "weights": None,
            "block": None,
            "coefficient_names": None,
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
    "cytepool.stats.voom.voom": Entry(
        lambda c: {
            "counts": COUNTS,
            "design": DESIGN,
            "library_sizes": LIBRARY_SIZES,
            "norm_factors": NORM_FACTORS,
        },
        {"counts": None, "design": None, "library_sizes": None, "norm_factors": None},
        options=("span",),
    ),
    # NaN marks the features that the trend leaves out.
    "cytepool.stats.voom.VoomResult": Entry(
        lambda c: _fields(voom(COUNTS, DESIGN)),
        {
            "log_cpm": None,
            "weights": None,
            "effective_library_sizes": None,
            "mean_log_count": "NaN",
            "sqrt_sd": "NaN",
            "sqrt_sd_trend": "NaN",
        },
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
    "cytepool.stats.empirical_bayes.squeeze_variances": Entry(
        lambda c: {
            "variances": c.fit.sigma**2,
            "df": c.fit.df_residual,
            "covariate": np.linspace(1.0, 2.0, 8),
            "winsor_tail_p": [0.05, 0.1],
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
        },
        {"trend": "NaN", "winsor_tail_p": None},
        options=("lfc", "robust", "prior_estimator"),
        objects=("fit",),
    ),
    # A fold change instead of its logarithm; treat refuses both at once.
    "cytepool.stats.empirical_bayes.treat[fc]": Entry(
        lambda c: {"fit": c.fit, "fc": 1.5}, {}, options=("fc",), objects=("fit",)
    ),
    # n and axis go through operator.index, which reads a 0-d masked integer.
    "cytepool.stats.multiple_testing.adjust_p_values": Entry(
        lambda c: {"p": [0.01, 0.2, 0.5], "n": 5, "axis": 0},
        {"p": "NaN", "n": None, "axis": None},
        options=("method",),
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
    "cytepool.cube.cube.Cube.collapse_samples": (("by",), "a column name, a str"),
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
}

# Result containers: they store their fields as given, so a masked field keeps its
# mask. One that starts converting its fields must become an entry point.
RECORDS = {
    "cytepool.design.checks.DesignIssue",
    "cytepool.design.checks.DesignReport",
    "cytepool.design.formula.FormulaTerms",
    "cytepool.design.formula.RandomEffect",
    "cytepool.stats.empirical_bayes.FDistributionFit",
    "cytepool.stats.empirical_bayes.RobustFDistributionFit",
    "cytepool.stats.empirical_bayes.SqueezedVariances",
    "cytepool.stats.empirical_bayes.ModeratedFit",
    "cytepool.stats.empirical_bayes.TreatFit",
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
