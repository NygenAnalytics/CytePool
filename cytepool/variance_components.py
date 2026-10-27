"""Variance partitioning of every group (cell type) of a cube.

:func:`~cytepool.variance_partitioning` takes, in every group of a cube, the steps of
:func:`~cytepool.dream_voom` up to voom's precision weights (Law et al. 2014), the
mean-variance trend of the design's mixed model (Hoffman and Roussos 2021), and
partitions the variance of every feature among the variables of the design, as
variancePartition's ``fitExtractVarPartModel`` does (Hoffman and Schadt 2016):
:mod:`cytepool.stats.variance_fractions` gives the model, the fractions and the
conventions found by probing variancePartition, and
:func:`~cytepool.stats.variance_fractions.variance_fractions` partitions the variance
of a samples x features matrix.

References:
    Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of variation
    in complex gene expression studies. BMC Bioinformatics 17, 483.
    doi:10.1186/s12859-016-1323-z

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ._analysis_records import (
    cube_record,
    matched_samples,
    mixed_design_record,
    only_key,
    software_versions,
)
from ._columns import text_column
from ._counts import effective_library_sizes
from ._group_fits import (
    COUNTS_LAYER,
    FITTED,
    GroupInputs,
    GroupRecord,
    check_sample_groups,
    filter_setting,
    group_positions,
    min_cells_setting,
    min_counts_setting,
    normalization_setting,
    prepare_group,
    prior_weight_setting,
    sample_table,
    span_setting,
    threads_setting,
)
from ._jit import thread_limit
from ._masked_arrays import reject_masked
from ._messages import fitted_lines, listing
from .cube import Cube
from .design import Contrasts, Design
from .stats._mean_variance_trend import scaled_prior_weights, trend_span
from .stats._mixed_model_designs import maximum_likelihood
from .stats.dream import _dream_weights
from .stats.linear_models import fit_linear_model
from .stats.variance_fractions import (
    _checked_terms,
    _linear_fractions,
    _mixed_fractions,
    _Terms,
)
from .stats.voom import voom
from .table import Table

__all__ = ["VariancePartitioningResult", "variance_partitioning"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Columns of the tables of a cube's groups other than the fractions.
_IDENTITY = ("feature_id", "symbol")
_METHOD = "variance_partitioning"
# The fits behind the precision weights, recorded in the provenance: maximum
# likelihood, as variancePartition's voomWithDreamWeights fits.
_FIXED_SETTINGS = {"weights_fit": "ml"}


# Every group of a cube.


class VariancePartitioningResult:
    """The fractions of variance of every group of a cube, from
    :func:`variance_partitioning`.

    Tables hold, for every group that was fitted, one row per feature that passed the
    expression filter, in the cube's feature order: ``feature_id`` and, when the
    cube's feature table has one, ``symbol``; then the fraction of the feature's
    variance that each variable explains, one column per variable, named as in
    ``variables`` (NaN for a feature whose mixed model has no maximum, which the fixed
    and random effects reproduce).

    The summary has one row per group: ``group_id``; ``status``, ``"fitted"`` or why
    the group was not fitted (``"no_samples"``, ``"no_residual_df"``,
    ``"fixed_effects_not_estimable"``, ``"random_effects_not_estimable"``,
    ``"no_features"``, ``"too_few_features"``, ``"empty_library"`` or
    ``"failed"``); ``reason``, the explanation (empty when fitted);
    ``n_samples_used`` and ``n_samples_left_out``; and ``n_features``, the number of
    features fitted (0 where the group was not fitted).

    The sample table has one row per group and design sample, as for
    :class:`~cytepool.DifferentialExpressionResult`:
    ``group_id``, ``sample_id``, ``n_cells`` (when the cube records cell numbers),
    ``used``, ``reason``, and the ``library_size`` and ``norm_factor`` that entered the
    fit; with ``prior_weights``, the ``prior_weight`` of each sample.

    Built by :func:`variance_partitioning`. Immutable.

    Raises:
        TypeError: If ``variables`` is not a sequence of names, ``summary`` or
            ``samples`` not a ``Table``, ``tables`` not a mapping of group id to
            ``Table``, or ``provenance`` not a mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_variables", "_summary", "_samples", "_tables", "_record")

    _variables: tuple[str, ...]
    _summary: Table
    _samples: Table
    _tables: Mapping[str, Table]
    _record: str

    def __init__(
        self,
        variables: Sequence[str],
        summary: Table,
        samples: Table,
        tables: Mapping[str, Table],
        provenance: Mapping[str, Any],
    ) -> None:
        if isinstance(variables, str) or not all(
            isinstance(name, str) for name in variables
        ):
            raise TypeError("variables must be a sequence of variable names (str)")
        for name, table in (("summary", summary), ("samples", samples)):
            if not isinstance(table, Table):
                raise TypeError(f"{name} must be a Table, got {type(table).__name__}")
        if not isinstance(tables, Mapping):
            raise TypeError(
                f"tables must be a mapping of group id to Table, got "
                f"{type(tables).__name__}"
            )
        for key, table in tables.items():
            if not isinstance(key, str) or not isinstance(table, Table):
                raise TypeError(
                    f"tables must map group ids (str) to Tables, got {key!r}: "
                    f"{type(table).__name__}"
                )
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_variables": tuple(variables),
            "_summary": summary,
            "_samples": samples,
            "_tables": MappingProxyType(dict(tables)),
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("VariancePartitioningResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("VariancePartitioningResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            VariancePartitioningResult,
            (
                self._variables,
                self._summary,
                self._samples,
                dict(self._tables),
                json.loads(self._record),
            ),
        )

    @property
    def variables(self) -> tuple[str, ...]:
        """The variables, in the order of the columns of the tables, residuals last."""
        return self._variables

    @property
    def groups(self) -> tuple[str, ...]:
        """The ids of the groups analysed, in order."""
        return tuple(self._summary["group_id"].tolist())

    @property
    def summary(self) -> Table:
        """The status, samples and features of every group."""
        return self._summary

    @property
    def samples(self) -> Table:
        """Every design sample of every group, used in its fit or left out, and why."""
        return self._samples

    @property
    def tables(self) -> Mapping[str, Table]:
        """The table of fractions of every group that was fitted, by group id."""
        return self._tables

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the analysis, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the method; the CytePool, numpy, numba and Python versions; the
        cube, by its fingerprint, digests and provenance; the layer read; the design
        (formula, columns, levels, random effects, and the values of the variables of
        both for every sample); the settings; and, for every group, its status, the
        samples that entered its fit in the order they were fitted, those left out
        with the reason, the number of features fitted, the span of its trend
        (``voom_span``), the number of features whose mixed model has no maximum
        (``features_not_fitted``) and of those whose fit stopped without converging
        (``features_not_converged``), and with ``prior_weights`` the prior weights of
        its samples. The cost of the cube's record is that of
        :func:`~cytepool.limma_voom`'s ("Provenance").
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None) -> Table:
        """Return the table of fractions of one group.

        Args:
            group: A group id; it may be left out when the result has one group.

        Raises:
            ValueError: If ``group`` is left out and the result has several groups.
            KeyError: If the group was not analysed or not fitted; the message says
                why.
        """
        group = only_key(group, list(self.groups), "group", "groups")
        found = self._tables.get(group)
        if found is not None:
            return found
        record = {entry["group_id"]: entry for entry in self.provenance["groups"]}
        if group not in record:
            raise KeyError(
                f"no group {group!r} was analysed; groups are: "
                f"{listing(list(record)) or '(none)'}"
            )
        entry = record[group]
        raise KeyError(
            f"group {group!r} was not fitted ({entry['status']}): {entry['reason']}"
        )

    def __repr__(self) -> str:
        statuses = self._summary["status"].tolist()
        fitted = statuses.count(FITTED)
        lines = [
            f"VariancePartitioningResult: {len(statuses):,} group"
            f"{'' if len(statuses) == 1 else 's'}, {fitted:,} fitted; variables: "
            + ", ".join(self._variables),
        ]
        for row in range(len(statuses)):
            group = self._summary["group_id"][row]
            if statuses[row] != FITTED:
                lines.append(f"  {group}: not fitted ({statuses[row]})")
                continue
            used = int(self._summary["n_samples_used"][row])
            features = int(self._summary["n_features"][row])
            lines.append(f"  {group}: {used:,} samples, {features:,} features")
        return fitted_lines("\n".join(lines))


@dataclass
class _GroupFit(GroupRecord):
    """What happened in one group, and its table of fractions."""

    table: Table | None = None


def variance_partitioning(
    cube: Cube,
    design: Design,
    *,
    groups: str | Sequence[str] | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    expression_filter: bool | Mapping[str, Any] = True,
    normalization: Literal["tmm"] | None = "tmm",
    span: float | None = None,
    prior_weights: Literal["n_cells"] | None = None,
    method: Literal["ml", "reml"] = "ml",
    n_threads: int | None = None,
) -> VariancePartitioningResult:
    """Partition the variance of every feature in every group of a cube.

    What drives the differences between samples in each cell type: for every group of
    the cube, the fraction of each feature's variance that each variable of the design
    explains, as dreamlet runs variancePartition's ``fitExtractVarPartModel`` on the
    ``voomWithDreamWeights`` output of every cell type. For each group, the samples are
    left out, the features filtered and the counts normalized as for
    :func:`~cytepool.dream_voom`; voom's precision weights come from the mean-variance
    trend of the design's mixed model, fitted by maximum likelihood
    (:func:`~cytepool.stats.dream.voom_with_dream_weights`), or from limma's voom for a
    design without random effects; and the fractions of every feature are those of
    :func:`~cytepool.stats.variance_fractions.variance_fractions` of its log2 counts per
    million with these weights.

    Every group's mixed model uses the random effects of the samples left in it, as
    in ``dream_voom``. A group is not fitted, and its status says why, when no sample
    is left, its samples cannot estimate a random effect (one donor left, or one
    sample per donor), they cannot separate the fixed effects (a covariate constant
    over them), fewer than two features with counts pass the filter, or the trend
    cannot give weights. A run never stops for one group.

    Args:
        cube: A cube with a ``counts`` layer.
        design: A :class:`~cytepool.Design` with an intercept, matched to the
            cube by ``sample_id`` as for :func:`~cytepool.limma_voom`, whose categorical
            variables are random intercepts and continuous variables fixed effects, such
            as ``Design(samples, "~ age", random="(1 | donor) + (1 | condition)")``; or
            a design without random effects.
        groups: A group id or a list of them; all groups by default.
        min_cells: The fewest cells a sample needs in a group to enter its fit;
            ignored for cubes without cell numbers.
        min_counts: The fewest counts a sample needs in a group, over every feature
            of the cube, to enter its fit; 0 by default.
        expression_filter: True for edgeR's ``filterByExpr`` defaults with the
            group's fixed effects, a mapping of changed settings or of the column of
            ``"sample_groups"``, as for :func:`~cytepool.limma_voom`, or False. With a
            random effect that separates groups whose expression may differ, such as the
            tissues of each donor, ``{"sample_groups": "tissue"}`` keeps the features
            expressed in any one of them.
        normalization: ``"tmm"`` for TMM normalization factors, or None.
        span: The LOWESS span of voom's trend, in (0, 1], for every group; None
            chooses it in each group from the number of features, as ``voom``
            does. variancePartition's ``voomWithDreamWeights`` uses 0.5.
        prior_weights: ``"n_cells"`` to give each sample its number of cells in the
            group as a prior weight, which the precision weights multiply, as
            dreamlet weights samples; or None. Needs a cube that records cell
            numbers.
        method: ``"ml"`` (maximum likelihood, variancePartition's default) or
            ``"reml"``, for the fits behind the fractions; the fits behind the
            precision weights are by maximum likelihood, as
            ``voomWithDreamWeights`` fits them.
        n_threads: The most threads for the fits of the features, at least 1; None
            keeps numba's setting. The results do not depend on it.

    Returns:
        The tables of fractions of every fitted group, a summary of every group, the
        samples used and left out, the settings and a provenance record.

    Raises:
        TypeError: If ``cube`` is not a Cube, ``design`` not a Design, an option has
            the wrong type, or ``groups``, ``min_cells``, ``expression_filter`` or
            ``n_threads`` is or holds a numpy masked array.
        ValueError: If the design cannot be partitioned (as for
            :func:`~cytepool.stats.variance_fractions.variance_fractions`), has no
            ``sample_id`` column, names samples the cube does not have or names a sample
            twice, a variable is named ``"feature_id"`` or ``"symbol"``, an option is
            invalid, or ``prior_weights`` is ``"n_cells"`` and the cube does not record
            cell numbers.
        KeyError: If a group is not in the cube.
    """
    reject_masked(groups, "groups")
    if not isinstance(cube, Cube):
        raise TypeError(f"cube must be a Cube, got {type(cube).__name__}")
    terms = _checked_terms(design)
    clashing = [name for name in terms.variables if name in _IDENTITY]
    if clashing:
        raise ValueError(
            f"a variable is named {clashing[0]!r}, a column of the tables of features; "
            "rename it in the sample table"
        )
    settings = {
        "min_cells": min_cells_setting(min_cells),
        "min_counts": min_counts_setting(min_counts),
        "expression_filter": filter_setting(expression_filter),
        "normalization": normalization_setting(normalization),
        "span": span_setting(span),
        "prior_weights": prior_weight_setting(prior_weights),
        "method": "ml" if maximum_likelihood(method) else "reml",
        "n_threads": threads_setting(n_threads),
    }
    if settings["prior_weights"] == "n_cells" and cube.n_cells is None:
        raise ValueError(
            "prior_weights='n_cells' weights each sample by its number of cells, which "
            "this cube does not record (a cube of bulk samples has none); leave "
            "prior_weights at None"
        )
    check_sample_groups(settings, design.samples)
    cube_rows, order = matched_samples(
        cube,
        design,
        no_ids="the design's sample table has no 'sample_id' column, so its rows cannot "
        "be matched to the cube's samples; build the Design from cube.samples, or from "
        "a table whose 'sample_id' column holds cube sample ids",
        text_ids="the cube's does",
        data="the cube",
        data_ids="the cube's sample ids",
    )
    positions = group_positions(cube, groups)
    inputs = GroupInputs(_METHOD, cube, order, cube_rows, None, settings)
    fits = [_fit_group(inputs, design, terms, g) for g in positions]
    sample_ids = cube.samples["sample_id"][cube_rows]
    no_contrasts = Contrasts(np.empty((len(design.columns), 0)), (), design.columns)
    return VariancePartitioningResult(
        terms.variables,
        _summary_table(fits),
        sample_table(cube, cube_rows, sample_ids, positions, fits, settings),
        {fit.group_id: fit.table for fit in fits if fit.table is not None},
        {
            "method": _METHOD,
            "versions": software_versions(),
            "cube": cube_record(cube),
            "layer": COUNTS_LAYER,
            "design": mixed_design_record(design, no_contrasts),
            "settings": {**settings, **_FIXED_SETTINGS},
            "groups": [_group_record(fit, sample_ids, settings) for fit in fits],
        },
    )


def _fit_group(inputs: GroupInputs, design: Design, terms: _Terms, g: int) -> _GroupFit:
    """Partition the variance of one group, or record why it cannot be done."""
    cube, settings = inputs.cube, inputs.settings
    mixed = bool(design.random)
    fit, data = prepare_group(_GroupFit, inputs, design, g, mixed=mixed)
    if data is None:
        return fit
    y = data.counts
    n_used, n_tested = y.shape
    with_counts = int(np.count_nonzero(y.any(axis=0)))
    if with_counts < 2:
        return fit.stop(
            "too_few_features",
            "the mean-variance trend of the precision weights needs at least two "
            f"features with counts, and {with_counts} of the {n_tested} features "
            f"{'has' if with_counts == 1 else 'have'} counts",
        )
    ml = settings["method"] == "ml"
    with thread_limit(settings["n_threads"]):
        try:
            if data.mixed is not None:
                sizes = np.array(effective_library_sizes(y, None, data.factors))
                fraction = trend_span(settings["span"], n_tested)
                prior = (
                    None
                    if data.prior is None
                    else scaled_prior_weights(data.prior, y.shape)
                )
                weighted = _dream_weights(y, sizes, fraction, data.mixed, prior, True)
            else:
                weighted = voom(
                    y,
                    data.x,
                    norm_factors=data.factors,
                    span=settings["span"],
                    weights=data.prior,
                )
        except ValueError as error:  # the trend cannot give weights for these data
            return fit.stop("failed", f"precision weights: {error}")
        if data.mixed is not None:
            fractions = _mixed_fractions(
                weighted.log2_cpm, weighted.weights, ml, data.mixed, terms
            )
        else:
            rows = np.ascontiguousarray(weighted.weights.T)
            linear = fit_linear_model(
                weighted.log2_cpm,
                data.x,
                weights=weighted.weights,
                coefficient_names=data.columns,
                n_threads=settings["n_threads"],
            )
            fractions = _linear_fractions(
                weighted.log2_cpm, data.x, rows, linear, ml, terms
            )
    fit.voom_span = weighted.span
    fitted = np.all(np.isfinite(fractions.fractions), axis=1)
    fit.features_not_fitted = int(np.count_nonzero(~fitted))
    fit.features_not_converged = int(np.count_nonzero(fitted & ~fractions.converged))
    fit.library_sizes = data.sizes
    fit.norm_factors = np.ones(n_used) if data.factors is None else data.factors
    fit.prior_weights = data.prior
    features = cube.features
    columns: dict[str, Any] = {"feature_id": features["feature_id"][fit.tested]}
    if "symbol" in features:
        columns["symbol"] = features["symbol"][fit.tested]
    for position, name in enumerate(fractions.variables):
        columns[name] = fractions.fractions[:, position]
    fit.table = Table(columns)
    return fit


def _summary_table(fits: list[_GroupFit]) -> Table:
    return Table(
        {
            "group_id": text_column([fit.group_id for fit in fits]),
            "status": text_column([fit.status for fit in fits]),
            "reason": text_column([fit.reason for fit in fits]),
            "n_samples_used": np.array(
                [np.count_nonzero(fit.used) for fit in fits], dtype=np.int64
            ),
            "n_samples_left_out": np.array(
                [np.count_nonzero(~fit.used) for fit in fits], dtype=np.int64
            ),
            "n_features": np.array(
                [fit.tested.size if fit.status == FITTED else 0 for fit in fits],
                dtype=np.int64,
            ),
        }
    )


def _group_record(
    fit: _GroupFit, sample_ids: np.ndarray, settings: Mapping[str, Any]
) -> dict[str, Any]:
    fitted = fit.status == FITTED
    record: dict[str, Any] = {
        "group_id": fit.group_id,
        "status": fit.status,
        "reason": fit.reason,
        "samples": sample_ids[fit.used].tolist(),
        "left_out": {
            sample_id: reason
            for sample_id, reason in zip(sample_ids.tolist(), fit.reasons, strict=True)
            if reason
        },
        "aliased_columns": list(fit.aliased),
        "n_features": int(fit.tested.size) if fitted else 0,
        "voom_span": fit.voom_span,
        "features_not_fitted": fit.features_not_fitted if fitted else None,
        "features_not_converged": fit.features_not_converged if fitted else None,
    }
    if settings["prior_weights"] is not None:
        weights = fit.prior_weights
        record["prior_weights"] = (
            weights.tolist() if fitted and weights is not None else None
        )
    return record
