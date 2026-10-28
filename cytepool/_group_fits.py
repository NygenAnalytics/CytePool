"""Fits of every group of a cube: the samples, design, features and counts of each.

The workflows that fit every group (cell type) of a cube one at a time, such as
``limma_voom``, ``dream_voom`` and ``variance_partitioning``, prepare a group alike:

1. Samples. A design sample is left out of a group's fit when it has fewer than
   ``min_cells`` cells there (when the cube records cell numbers), no counts, or
   fewer than ``min_counts`` counts in all; the record keeps why.
2. Design. The design matrix of the group is the rows of the full design matrix for
   the samples used, so that every column keeps its meaning. Columns that are
   combinations of earlier ones on these samples (aliased, by R's ``qr()`` rule) are
   recorded. With contrasts to test, the group is fitted on the columns that span
   its rows, with the contrasts its samples can estimate; without contrasts, every
   column must be estimable. With random effects, those of the design are checked
   on the samples used.
3. Features. Features are filtered by expression with the group's design, by edgeR's
   ``filterByExpr`` (Chen, Lun and Smyth 2016), or with the groups of samples that a
   column of the design's samples gives (``filterByExpr(y, group = ...)``), and the
   library sizes are taken over the features kept.
4. Normalization. TMM normalization factors (Robinson and Oshlack 2010), when asked.

This module also checks the options these workflows share, and builds the table of
the samples every group used.

References:
    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2

    Robinson MD, Oshlack A (2010). A scaling normalization method for differential
    expression analysis of RNA-seq data. Genome Biology 11, R25.
    doi:10.1186/gb-2010-11-3-r25
"""

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Self, TypeVar

import numpy as np

from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import listing
from ._options import index_integer, non_negative_number
from .cube import Cube
from .design import Contrasts, Design
from .design.checks import _aliasing, _contrast_violations, _relation
from .normalization import filter_by_expression, library_sizes, tmm_factors
from .stats._mixed_model_designs import MixedDesign, mixed_design_of_samples
from .table import Table

FITTED = "fitted"
"""The status of a group that was fitted."""

COUNTS_LAYER = "counts"
"""The layer of the cube that the fits read."""

FILTER_DEFAULTS = {
    "min_count": 10.0,
    "min_total_count": 15.0,
    "large_n": 10.0,
    "min_prop": 0.7,
}
"""edgeR's ``filterByExpr`` defaults."""

SAMPLE_GROUPS = "sample_groups"
"""The setting of ``expression_filter`` that names a column of the samples whose
values group them for ``filterByExpr``, in place of the design."""


@dataclass
class GroupRecord:
    """What happened in one group: the samples used, and why it stopped if it did.

    The workflows extend it with what their fits produce.
    """

    group_id: str
    used: np.ndarray  # bool, per design sample in cube order
    reasons: list[str]  # why each design sample was left out; "" when used
    status: str = FITTED
    reason: str = ""
    aliased: tuple[str, ...] = ()
    not_estimable: dict[str, str] = field(default_factory=dict)
    df_residual: float = float("nan")
    tested: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    library_sizes: np.ndarray | None = None  # per used sample, when fitted
    norm_factors: np.ndarray | None = None
    prior_weights: np.ndarray | None = None  # per used sample, when fitted with them
    voom_span: float | None = None  # the span of voom's trend, once voom has run
    features_not_fitted: int = 0  # mixed models: features whose model has no minimum
    features_not_converged: int = 0  # mixed models: fits that did not converge
    warnings: list[str] = field(default_factory=list)  # given as the group is fitted

    def stop(self, status: str, reason: str) -> Self:
        """Record that the group is not fitted, and why; returns the record."""
        self.status, self.reason = status, reason
        return self


Record = TypeVar("Record", bound=GroupRecord)


@dataclass(frozen=True)
class GroupInputs:
    """The checked inputs shared by every group."""

    method: str
    cube: Cube
    model_rows: np.ndarray  # the design's rows in cube order
    cube_rows: np.ndarray  # the cube position of each, increasing
    contrasts: Contrasts | None  # the contrasts to test, or None to fit every column
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class GroupData:
    """The checked data of one group, ready for the fits of a method."""

    x: np.ndarray  # the rows of the full design for the samples used, kept columns
    columns: tuple[str, ...]  # the names of the kept columns
    kept: list[int]  # the kept columns among the design's
    estimable: list[int]  # the contrasts that the samples can estimate
    counts: np.ndarray  # samples used x features tested
    sizes: np.ndarray  # library sizes over the features tested
    factors: np.ndarray | None  # normalization factors, or None
    prior: np.ndarray | None  # prior weights of the samples used, or None
    mixed: MixedDesign | None  # the mixed model's design of the samples used


def prepare_group(
    record_type: type[Record],
    inputs: GroupInputs,
    design: Design,
    g: int,
    *,
    mixed: bool = False,
) -> tuple[Record, GroupData | None]:
    """The samples, design, features and normalization of one group.

    Returns the group's record, a new ``record_type``, and its data, or no data when
    the group cannot be fitted, with the record saying why. Without contrasts in
    ``inputs``, a group whose design matrix is not of full column rank on its
    samples is not fitted. With ``mixed``, the random effects of the design are also
    checked on the samples used.
    """
    cube, settings = inputs.cube, inputs.settings
    counts = cube.layer(COUNTS_LAYER, g)[inputs.cube_rows]  # design samples, cube order
    cells = None if cube.n_cells is None else cube.n_cells[inputs.cube_rows, g]
    min_counts = settings["min_counts"]
    reasons = left_out(counts, cells, settings["min_cells"], min_counts)
    used = np.array([not reason for reason in reasons], dtype=bool)
    fit = record_type(str(cube.groups["group_id"][g]), used, reasons)
    if counts.shape[1] == 0:
        return fit.stop("no_features", "the cube has no features"), None
    n_used = int(np.count_nonzero(used))
    if n_used == 0:
        # No counts is fewer than any positive minimum.
        rules = f"fewer than {amount(min_counts)} counts" if min_counts else "no counts"
        if cells is not None:
            rules = f"fewer than {settings['min_cells']} cells, or {rules}"
        reason = f"every one of the {used.size} samples was left out ({rules})"
        return fit.stop("no_samples", reason), None

    # The rows of the full design, and the columns that span them.
    x = design.matrix[inputs.model_rows][used]
    aliasing = _aliasing(x)
    kept = list(aliasing.kept)
    fit.aliased = tuple(design.columns[j] for j in aliasing.aliased)
    fit.df_residual = float(n_used - len(kept))
    if n_used == len(kept):
        return fit.stop(
            "no_residual_df",
            f"no residual degrees of freedom: the design has rank {len(kept)} on the "
            f"{n_used} sample{'' if n_used == 1 else 's'} left, so it fits "
            f"{'it' if n_used == 1 else 'them'} exactly and no variance can be "
            "estimated",
        ), None
    contrasts = inputs.contrasts
    model = replace(design.model_matrix, matrix=x)
    if contrasts is None:
        estimable: list[int] = []
        if aliasing.aliased:
            relations = [_relation(model, aliasing, j) for j in aliasing.aliased]
            return fit.stop(
                "fixed_effects_not_estimable",
                "the samples of this group cannot separate the fixed effects: "
                + listing(relations, separator="; "),
            ), None
    else:
        # limma's contrasts.fit refuses an estimable contrast that weighs an aliased
        # column, and every contrast passed with it (found by probing); it is
        # computed here, from the fit on the kept columns.
        violations = _contrast_violations(model, aliasing, contrasts)
        for position, relations in violations.items():
            fit.not_estimable[contrasts.names[position]] = (
                "not estimable from the samples of this group, which cannot separate "
                "the effects it depends on: " + listing(relations, separator="; ")
            )
        estimable = [p for p in range(len(contrasts)) if p not in violations]
        if not estimable:
            return fit.stop(
                "no_estimable_contrast",
                "no contrast is estimable from the samples of this group",
            ), None
    x_kept = np.ascontiguousarray(x[:, kept])
    columns = tuple(design.columns[j] for j in kept)
    parts = None
    if mixed:
        try:
            parts = mixed_design_of_samples(
                design, inputs.model_rows[used], x_kept, columns
            )
        except ValueError as error:
            return fit.stop("random_effects_not_estimable", str(error)), None

    y = np.asarray(counts[used], dtype=np.float64)
    filter_settings = settings["expression_filter"]
    if filter_settings is None:
        keep = np.ones(y.shape[1], dtype=bool)
    elif SAMPLE_GROUPS in filter_settings:
        numbers = {
            name: value
            for name, value in filter_settings.items()
            if name != SAMPLE_GROUPS
        }
        labels = design.samples[filter_settings[SAMPLE_GROUPS]]
        keep = filter_by_expression(
            y, sample_groups=labels[inputs.model_rows[used]], **numbers
        )
    else:
        keep = filter_by_expression(y, design=x_kept, **filter_settings)
    fit.tested = np.flatnonzero(keep)
    n_tested = fit.tested.size
    if n_tested == 0:
        return fit.stop("no_features", "no feature passed the expression filter"), None
    y = np.ascontiguousarray(y[:, keep])
    sizes = library_sizes(y)
    if not np.all(sizes > 0.0):
        sample_ids = cube.samples["sample_id"][inputs.cube_rows][used]
        empty = sample_ids[sizes == 0.0].tolist()
        return fit.stop(
            "empty_library",
            f"{listing(empty)} {'has' if len(empty) == 1 else 'have'} no counts in "
            f"the {n_tested} features that passed the expression filter",
        ), None
    factors = tmm_factors(y) if settings["normalization"] == "tmm" else None
    # Prior weights are positive: a used sample has counts, so it has cells.
    prior = None
    if settings["prior_weights"] == "n_cells" and cells is not None:
        prior = np.asarray(cells[used], dtype=np.float64)
    data = GroupData(x_kept, columns, kept, estimable, y, sizes, factors, prior, parts)
    return fit, data


def left_out(
    counts: np.ndarray, cells: np.ndarray | None, min_cells: int, min_counts: float
) -> list[str]:
    """Why each sample is left out of a group's fit, or "" when it is used."""
    # A sample with exactly min_cells cells or min_counts counts is kept, as
    # decoupler's filter_samples keeps it (found by probing).
    has_counts = counts.any(axis=1)
    # Totals over every feature, exact for whole counts, as R's colSums gives them.
    totals = library_sizes(counts) if min_counts > 0.0 and counts.size else None
    reasons = []
    for sample in range(counts.shape[0]):
        if cells is not None and cells[sample] < min_cells:
            n = int(cells[sample])
            reasons.append(
                f"{n} cell{'' if n == 1 else 's'}, fewer than min_cells ({min_cells})"
            )
        elif not has_counts[sample]:
            reasons.append("no counts")
        elif totals is not None and totals[sample] < min_counts:
            total = float(totals[sample])
            reasons.append(
                f"{amount(total)} count{'' if total == 1.0 else 's'}, fewer than "
                f"min_counts ({amount(min_counts)})"
            )
        else:
            reasons.append("")
    return reasons


def group_positions(cube: Cube, groups: str | Sequence[str] | None) -> list[int]:
    """The positions of the groups asked for, in the order given; all by default."""
    ids = cube.groups["group_id"].tolist()
    if groups is None:
        return list(range(len(ids)))
    if isinstance(groups, str):
        chosen: list[Any] = [groups]
    elif isinstance(groups, Sequence | np.ndarray):
        chosen = list(groups)
    else:
        raise TypeError(
            f"groups must be a group id or a list of group ids, got "
            f"{type(groups).__name__}"
        )
    if not chosen:
        raise ValueError(
            "groups is empty; give at least one group id, or None for every group"
        )
    index = {value: i for i, value in enumerate(ids)}
    positions = []
    for value in chosen:
        if not isinstance(value, str):
            raise TypeError(
                f"groups must be group ids (str), got {type(value).__name__} {value!r}"
            )
        if value not in index:
            raise KeyError(
                f"no group with id {value!r}; groups are: {listing(ids) or '(none)'}"
            )
        if index[value] in positions:
            raise ValueError(f"group {value!r} is listed more than once")
        positions.append(index[value])
    return positions


# Checks of the options the workflows share.


def min_cells_setting(value: Any) -> int:
    """The fewest cells a sample needs in a group, a whole number of at least 0."""
    if isinstance(value, bool | np.bool_):
        raise TypeError("min_cells must be a whole number of cells, got bool")
    number = index_integer(value, "min_cells", expected="a whole number of cells")
    if number < 0:
        raise ValueError(f"min_cells must be non-negative, got {number}")
    return number


def min_counts_setting(value: Any) -> float:
    """The fewest counts a sample needs in a group, a number of at least 0."""
    # real_number would refuse a masked value too, but without saying how to fix it.
    reject_masked(value, "min_counts")
    return non_negative_number(value, "min_counts")


def prior_weight_setting(prior_weights: Any) -> str | None:
    """The checked prior weights: "n_cells" or None."""
    if prior_weights is not None and not (
        isinstance(prior_weights, str) and prior_weights == "n_cells"
    ):
        error = ValueError if isinstance(prior_weights, str) else TypeError
        raise error(
            "prior_weights must be 'n_cells', for the number of cells of each sample "
            f"in the group, or None, got {prior_weights!r}"
        )
    return None if prior_weights is None else "n_cells"


def span_setting(value: Any) -> float | None:
    """The LOWESS span of voom's trend, in (0, 1], or None to choose it."""
    if value is None:
        return None
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(
            "span must be a real number in (0, 1], or None to choose it in each group "
            f"from the number of features tested, got {type(value).__name__}"
        )
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ValueError(
            "span must be in (0, 1], or None to choose it in each group from the "
            f"number of features tested, got {number}"
        )
    return number


def threads_setting(value: Any) -> int | None:
    """The most threads for the fits, a positive integer, or None."""
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        raise TypeError("n_threads must be an integer or None, got bool")
    number = index_integer(value, "n_threads", expected="an integer or None")
    if number < 1:
        raise ValueError(f"n_threads must be at least 1, got {number}")
    return number


def filter_setting(value: Any) -> dict[str, Any] | None:
    """The settings of ``filterByExpr``: its defaults, changed ones, or None.

    A mapping may also name, as ``sample_groups``, a column of the samples whose
    values group them; ``check_sample_groups`` checks it against the samples.
    """
    reject_masked(value, "expression_filter")
    if isinstance(value, bool | np.bool_):
        return dict(FILTER_DEFAULTS) if value else None
    if not isinstance(value, Mapping):
        raise TypeError(
            "expression_filter must be True, False or a mapping of filterByExpr "
            f"settings such as {{'min_count': 5}} or {{'{SAMPLE_GROUPS}': "
            f"'condition'}}, got {type(value).__name__}"
        )
    settings: dict[str, Any] = dict(FILTER_DEFAULTS)
    for name, number in value.items():
        if name == SAMPLE_GROUPS:
            if not isinstance(number, str):
                raise TypeError(
                    f"expression_filter[{SAMPLE_GROUPS!r}] must name a column of the "
                    f"samples (str), got {type(number).__name__}"
                )
            settings[name] = number
            continue
        if name not in FILTER_DEFAULTS:
            raise ValueError(
                f"expression_filter has the unknown setting {name!r}; the settings are "
                f"{', '.join([*FILTER_DEFAULTS, SAMPLE_GROUPS])}"
            )
        settings[name] = non_negative_number(number, f"expression_filter[{name!r}]")
    if settings["min_prop"] > 1.0:
        raise ValueError(
            f"expression_filter['min_prop'] must be in [0, 1], got {settings['min_prop']}"
        )
    return settings


def check_sample_groups(settings: Mapping[str, Any], samples: Table) -> None:
    """Refuse a column of ``sample_groups`` that the samples do not have, or in which
    a sample has no value.

    Raises:
        KeyError: If ``samples`` has no such column.
        ValueError: If a value of the column is missing.
    """
    filter_settings = settings["expression_filter"]
    if filter_settings is None or SAMPLE_GROUPS not in filter_settings:
        return
    name = filter_settings[SAMPLE_GROUPS]
    if name not in samples:
        raise KeyError(
            f"expression_filter[{SAMPLE_GROUPS!r}] names {name!r}, which is not a "
            f"column of the samples; their columns are: {listing(list(samples.columns))}"
        )
    values = samples[name].tolist()
    missing = [
        position
        for position, value in enumerate(values)
        if value is None or value != value
    ]
    if missing:
        where = (
            f"sample {samples['sample_id'][missing[0]]!r}"
            if "sample_id" in samples
            else f"the sample in row {missing[0]}"
        )
        raise ValueError(
            f"expression_filter[{SAMPLE_GROUPS!r}] names {name!r}, which is missing for "
            f"{where}; every sample needs a group for the filter"
        )


def normalization_setting(value: Any) -> str | None:
    """The normalization: "tmm" or None."""
    if value is None or (isinstance(value, str) and value == "tmm"):
        return value
    error = ValueError if isinstance(value, str) else TypeError
    raise error(
        "normalization must be 'tmm' (edgeR's TMM factors) or None (library sizes "
        f"alone), got {value!r}"
    )


# Results.


def sample_table(
    cube: Cube,
    cube_rows: np.ndarray,
    sample_ids: np.ndarray,
    positions: list[int],
    fits: Sequence[GroupRecord],
    settings: Mapping[str, Any],
) -> Table:
    """Every design sample of every group: used or left out and why, and the
    library size, normalization factor and sample weights that entered its fit."""
    n = cube_rows.size
    numbers = {
        "library_size": "library_sizes",
        "norm_factor": "norm_factors",
    }
    if settings["prior_weights"] is not None:
        numbers["prior_weight"] = "prior_weights"
    if settings.get("quality_weights") is not None:
        numbers["quality_weight"] = "quality_weights"
    # Per group and design sample; NaN where the number did not enter a fit.
    values = {column: np.full((len(fits), n), np.nan) for column in numbers}
    for row, fit in enumerate(fits):
        if fit.status != FITTED:
            continue
        for column, attribute in numbers.items():
            values[column][row, fit.used] = getattr(fit, attribute)
    columns: dict[str, Any] = {
        "group_id": text_column([fit.group_id for fit in fits for _ in range(n)]),
        "sample_id": text_column(np.tile(sample_ids, len(fits)).tolist()),
    }
    if cube.n_cells is not None:
        cells = cube.n_cells[np.ix_(cube_rows, positions)]  # samples x groups
        columns["n_cells"] = np.ascontiguousarray(cells.T).reshape(-1)
    columns |= {
        "used": np.concatenate([fit.used for fit in fits] or [np.zeros(0, bool)]),
        "reason": text_column([reason for fit in fits for reason in fit.reasons]),
    }
    columns |= {column: array.reshape(-1) for column, array in values.items()}
    return Table(columns)


def amount(value: float) -> str:
    """A number of counts as text, without a decimal point when it is whole."""
    return f"{value:.15g}"
