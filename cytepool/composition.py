"""Tests of differences in cell-type composition between conditions, by propeller.

:func:`propeller` tests, for every group (cell type) of a cube, whether the share of a
sample's cells that belong to the group differs between conditions, as speckle's
``propeller`` does (Phipson et al. 2022). It reads only the number of cells of every
sample in every group (``Cube.n_cells``), or a table of them, and fits one linear
model to all groups at once:

1. Proportions. Sample j, with x_ij cells of group i and N_j cells in all, has the
   proportion p_ij = x_ij / N_j of group i.
2. Transform. The logit transform, log(q_ij / (1 - q_ij)), takes the proportions
   after half a cell is added to every count, q_ij = (x_ij + 0.5) / sum_i (x_ij +
   0.5), so that a group absent from a sample has a finite value; the arcsine
   square-root transform, arcsin(sqrt(p_ij)), takes the proportions as they are
   (Phipson et al. 2022, supplementary section 1).
3. Linear model. The transformed proportions of every group are fitted with the
   design by least squares (limma's ``lmFit``), and the contrasts are formed
   (``contrasts.fit``).
4. Moderation. The residual variances of the groups are moderated by empirical
   Bayes, robustly by default, as speckle calls limma's ``eBayes(robust = TRUE)``
   (Smyth 2004; Phipson et al. 2016).
5. Tests. Each contrast is tested by its moderated t-statistic, as
   ``propeller.ttest`` does. ``f_test`` names a factor whose levels are tested for
   equal means by the moderated F-statistic, as ``propeller.anova`` does, and
   ``propeller`` with more than two groups. The p-values of a test are adjusted by
   the method of Benjamini and Hochberg over the groups (speckle's ``FDR``).

Samples and groups:
    The design's sample table must have a ``sample_id`` column; its samples are
    matched to those of the data by ``sample_id``, never by position, and may be some
    of them, in any order. A sample of the design without cells in any group has no
    proportions and is left out, and the result records why. Proportions are taken
    over the groups of the data, so to test the composition of one lineage, select its
    groups first (``Cube.select(groups=...)``). A group without cells in any sample
    used is left out too: it is not observed in the experiment, and speckle, which
    reads the label of every cell, never sees it. At least two groups must remain.

Descriptive columns:
    Each table also gives speckle's descriptive columns, whose definitions were found
    by probing speckle 1.6.0. The baseline proportion of a group is its share of all
    the cells of the samples used (speckle's ``BaselineProp``). The mean proportion
    of a level of a factor is the mean of p_ij over the samples at that level
    (``PropMean``), taken over the samples as they are, whatever the other terms of
    the design. A contrast *compares the levels* of a factor when it is a
    combination, with weights d_l that sum to zero, of the differences between the
    levels of a factor that enters the formula as a main effect only; for example
    ``conditionstim`` in ``~ donor + condition``, ``conditionstim -
    conditionctrl`` in ``~ 0 + condition``, or ``(groupA + groupB)/2 - groupC``,
    but not ``groupA + groupB``. The proportion ratio of such a contrast is the
    product of the mean proportions m_l of its levels raised to their weights,
    prod_l m_l^d_l (``PropRatio``): m_stim / m_ctrl for stim against ctrl, and
    sqrt(m_A m_B) / m_C for ``(groupA + groupB)/2 - groupC``. The table of such a
    contrast gives the mean proportion of each level with a weight other than zero,
    and the ratio; the table of another contrast, such as a covariate's slope, gives
    neither.

Deliberate differences from speckle:
    ``validation/composition_tests.py`` reports each of them.

    - speckle's ``PropMean`` are the coefficients of the proportions regressed on
      the columns of the design matrix that the contrast weighs, and its
      ``PropRatio`` is their product raised to the contrast's weights (found by
      probing). Where those columns indicate groups of samples, as in a design
      without an intercept whose first columns are the groups, the coefficients
      are the groups' mean proportions and the ratio is the one above. Otherwise
      they are not: for ``groupA + groupB``, which compares nothing, speckle
      reports the product m_A m_B as a ratio; for ``(Intercept) + groupB`` in ``~
      group + age``, the mean of the samples outside B and a difference of means;
      and for a contrast that weighs a single column, such as ``conditionstim`` in
      ``~ condition`` or a covariate, it stops with an error. Here the levels a
      contrast compares are found from the design's coding, so every
      parametrisation of the same comparison gives the same table, and a contrast
      that compares no levels is tested without descriptive columns.
    - A group without cells in any sample is left out. speckle keeps an unused level
      of a factor of cell labels, tests it, and with the logit transform its half
      cells change the transformed proportions of every other group (found by
      probing).
    - A sample without cells is left out. speckle keeps an unused level of a factor
      of sample labels: with the arcsine transform its proportions are NaN and limma
      leaves it out too, but with the logit transform every group of that sample has
      the proportion 1 / (number of groups), which enters the fit as data (found by
      probing).
    - Data that the design fits exactly in every group, up to rounding, as when the
      samples of each condition are copies of one another, raise an error: no
      variance can be estimated. speckle returns statistics of the rounding errors
      (t-statistics near 1e15, for example) or NaN.

Accuracy:
    The transforms are computed as log((x_ij + 0.5) / (N_j - x_ij + 0.5 (I - 1)))
    for I groups and as arctan2(sqrt(x_ij), sqrt(N_j - x_ij)): the formulas above,
    without the cancellation in 1 - q_ij and the loss of digits of arcsin near 1.
    They agree with speckle's values to rounding, and stay within a few units in
    the last place where those lose digits, for a group that holds nearly all the
    cells of a sample. Cell numbers are summed exactly, in float64 up to 2**53.

References:
    Phipson B, Sim CB, Porrello ER, Hewitt AW, Powell J, Oshlack A (2022). propeller:
    testing for differences in cell type proportions in single cell data. Bioinformatics
    38, 4720-4726. doi:10.1093/bioinformatics/btac582

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._analysis_records import (
    cube_record,
    design_record,
    matched_samples,
    software_versions,
)
from ._columns import is_data_frame, text_column
from ._masked_arrays import reject_masked, unmasked_array
from ._messages import listing, quoted
from ._options import checked_choice, checked_flag
from ._summation import accurate_mean
from .cube import Cube
from .design import Contrasts, Design, ModelMatrix, make_contrasts
from .design.checks import _aliasing, _relation
from .stats.empirical_bayes import ModeratedFit, moderated_statistics
from .stats.linear_models import LinearModelFit, contrasts_fit, fit_linear_model
from .stats.multiple_testing import adjust_p_values
from .table import Table
from .design._design_matrices import refuse_random_effects

__all__ = ["CompositionResult", "propeller"]

_METHOD = "propeller"
_SAMPLE_ID = "sample_id"
_LOGIT_OFFSET = 0.5  # cells added to every count before the logit transform
# limma's defaults that speckle keeps, recorded in the provenance.
_FIXED_SETTINGS = {
    "logit_offset": _LOGIT_OFFSET,
    "winsor_tail_p": [0.05, 0.1],
    "p_value_adjustment": "BH",
}
# A weight of a contrast, or the sum of the weights of its levels, below this fraction
# of its largest weight counts as zero (it is rounding in weights such as 1/3).
_ZERO_WEIGHT = 1e-12
# Cell numbers are summed in float64, which is exact up to 2**53.
_EXACT_TOTAL = 2.0**53
_INT64_MAX = 2**63 - 1  # the most cells a cube holds in an entry
# A residual standard deviation at most this fraction of the largest transformed
# proportion of a group (or of 1) is rounding: the design fits the group exactly.
_ROUNDING = 1e-12


class CompositionResult:
    """Tests of differences in cell-type composition, from :func:`propeller`.

    Tables hold one row per group (cell type) tested, in the order of the data's
    groups; speckle sorts them by p-value, as ``table.take(numpy.argsort(
    table["p_value"], kind="stable"))`` does. The table of a contrast (``tables``,
    ``table``) has the columns

    - ``group_id``;
    - ``baseline_proportion``: the group's share of all the cells of the samples
      used (speckle's ``BaselineProp``);
    - for a contrast that compares the levels of a factor (see
      :mod:`cytepool.composition`), ``mean_proportion_<level>`` for each level it
      weighs, in level order: the mean proportion of the group over the samples at
      that level (``PropMean``), and ``proportion_ratio``: the product of these
      means raised to the weights of their levels, such as m_stim / m_ctrl
      (``PropRatio``);
    - ``t`` and ``p_value``: the moderated t-statistic of the contrast on the
      transformed proportions and its two-sided p-value (``Tstatistic``,
      ``P.Value``);
    - ``adjusted_p_value``: Benjamini-Hochberg over the groups (``FDR``).

    The F-test table (``f_test``) has ``group_id``, ``baseline_proportion``,
    ``mean_proportion_<level>`` for every level of the factor, ``f_statistic``,
    ``p_value`` and ``adjusted_p_value`` (speckle's ``Fstatistic``, ``P.Value`` and
    ``FDR``).

    The summary has one row per group of the data: ``group_id``; ``n_cells``, its
    cells in the samples used; ``tested``; ``reason``, why it was not tested (empty
    when tested); ``df_residual``; and ``prior_df``, the prior degrees of freedom of
    the moderation (each group has its own with robust moderation; NaN for a group
    not tested). The sample table has one row per design sample, in the data's
    sample order: ``sample_id``, ``n_cells`` (in all groups), ``used`` and
    ``reason``. The proportions table has one row per group tested and sample used:
    ``group_id``, ``sample_id``, ``n_cells``, ``proportion`` and
    ``transformed_proportion``, the value the linear model was fitted to.

    Built by :func:`propeller` from the tables and record it computed. Immutable.

    Raises:
        TypeError: If ``contrasts`` is not a ``Contrasts``, ``tables`` not a mapping
            of contrast name to ``Table``, ``f_test`` not a ``Table`` or None, or
            ``summary``, ``samples`` or ``proportions`` not a ``Table``, or
            ``provenance`` not a mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = (
        "_contrasts",
        "_tables",
        "_f_test",
        "_summary",
        "_samples",
        "_proportions",
        "_record",
    )

    _contrasts: Contrasts
    _tables: Mapping[str, Table]
    _f_test: Table | None
    _summary: Table
    _samples: Table
    _proportions: Table
    _record: str

    def __init__(
        self,
        contrasts: Contrasts,
        tables: Mapping[str, Table],
        f_test: Table | None,
        summary: Table,
        samples: Table,
        proportions: Table,
        provenance: Mapping[str, Any],
    ) -> None:
        if not isinstance(contrasts, Contrasts):
            raise TypeError(
                f"contrasts must be a Contrasts, got {type(contrasts).__name__}"
            )
        if not isinstance(tables, Mapping):
            raise TypeError(
                "tables must be a mapping of contrast name to Table, got "
                f"{type(tables).__name__}"
            )
        for name, table in tables.items():
            if not isinstance(name, str) or not isinstance(table, Table):
                raise TypeError(
                    "tables must map contrast names (str) to Tables, got "
                    f"{type(name).__name__} {name!r} to {type(table).__name__}"
                )
        if f_test is not None and not isinstance(f_test, Table):
            raise TypeError(
                f"f_test must be a Table or None, got {type(f_test).__name__}"
            )
        for label, table in (
            ("summary", summary),
            ("samples", samples),
            ("proportions", proportions),
        ):
            if not isinstance(table, Table):
                raise TypeError(f"{label} must be a Table, got {type(table).__name__}")
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_contrasts": contrasts,
            "_tables": MappingProxyType(dict(tables)),
            "_f_test": f_test,
            "_summary": summary,
            "_samples": samples,
            "_proportions": proportions,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("CompositionResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CompositionResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            CompositionResult,
            (
                self._contrasts,
                dict(self._tables),
                self._f_test,
                self._summary,
                self._samples,
                self._proportions,
                json.loads(self._record),
            ),
        )

    @property
    def contrasts(self) -> Contrasts:
        """The contrasts tested by t-tests, over the columns of the design."""
        return self._contrasts

    @property
    def tables(self) -> Mapping[str, Table]:
        """The table of every contrast, by contrast name."""
        return self._tables

    @property
    def f_test(self) -> Table | None:
        """The table of the F-test across the levels of ``f_test``, or None."""
        return self._f_test

    @property
    def summary(self) -> Table:
        """Every group of the data, with its cells, whether it was tested, and its df."""
        return self._summary

    @property
    def samples(self) -> Table:
        """Every design sample, used or left out, and why."""
        return self._samples

    @property
    def proportions(self) -> Table:
        """The proportion and transformed proportion of every group in every sample."""
        return self._proportions

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the analysis, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the method; the CytePool, numpy, numba and Python versions; what
        the data were (``"input"``: ``"cube"`` or ``"table"``); the cube
        (``"cube"``): its fingerprint, the digests it is built from and its own
        provenance, where a table of cell numbers is recorded as the cube without
        features that holds them, whose ``samples``, ``groups`` and ``n_cells``
        digests identify the table; the design (formula, columns, levels, the
        values of its variables for every sample, and the weights of every
        contrast); the settings; the hypothesis of the F-test (the factor, its
        levels and the contrasts between them), or null; the samples used, in the
        order fitted, and those left out with the reason; and the groups tested
        and those left out with the reason. The fingerprint, and the digest of
        ``n_cells``, say whether a cube at hand holds the cell numbers behind the
        tables (``Cube.verify_fingerprint``).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, contrast: str) -> Table:
        """Return the table of one contrast.

        Args:
            contrast: A contrast name.

        Raises:
            KeyError: If the contrast was not tested; the message lists those that
                were.
        """
        found = self._tables.get(contrast)
        if found is None:
            raise KeyError(
                f"no contrast {contrast!r} was tested; contrasts are: "
                f"{listing(list(self._contrasts.names)) or '(none)'}"
            )
        return found

    def __repr__(self) -> str:
        record = json.loads(self._record)
        tested = len(record["groups"])
        used = len(record["samples"])
        lines = [
            f"CompositionResult (propeller, {record['settings']['transform']}): "
            f"{tested} of {len(self._summary)} groups tested, {used} of "
            f"{len(self._samples)} samples used"
        ]
        if len(self._contrasts):
            lines.append("  t-tests: " + ", ".join(self._contrasts.names))
        if record["f_test"] is not None:
            factor = record["f_test"]["factor"]
            levels = record["f_test"]["levels"]
            lines.append(f"  F-test across the {len(levels)} levels of {factor}")
        return "\n".join(lines)


def propeller(
    data: Cube | Any,
    design: Design,
    *,
    contrasts: Mapping[str, Any] | Sequence[str] | str | None = None,
    f_test: str | None = None,
    transform: Literal["logit", "asin"] = "logit",
    robust: bool = True,
    trend: bool = False,
) -> CompositionResult:
    """Test whether the cell-type composition of samples differs between conditions.

    For every group (cell type), the proportion of each sample's cells in the
    group is transformed, a linear model is fitted to the transformed proportions
    with the design, and its contrasts are tested by moderated t-statistics, and the
    levels of a factor by a moderated F-statistic, after empirical Bayes moderation
    of the residual variances across groups: speckle's ``propeller``,
    ``propeller.ttest`` and ``propeller.anova``. The module documentation describes
    the steps, the descriptive columns and the deliberate differences from speckle.

    As in speckle, ``propeller(clusters, sample, group)`` with two groups tests the
    first level against the second: ``Design(samples, "~ 0 + group",
    contrasts={"A_vs_B": "groupA - groupB"})``; with more groups it is
    ``f_test="group"``. ``propeller.ttest`` with a design matrix of group columns
    followed by covariates is a design such as ``~ 0 + group + age`` with a
    contrast between groups, and ``propeller.anova`` the F-test of ``group`` in
    it. The same comparisons in another parametrisation, such as ``~ group + age``
    with the contrast ``groupB``, give the same tables.

    Provenance:
        The result records the cube's fingerprint, the digests it is built from and
        its own provenance, as :func:`~cytepool.differential_expression.limma_voom`
        does, although the test reads only the cell numbers. For a cube opened with
        :func:`~cytepool.io.read_cube` this reads nothing, as the file stores the
        digests; a cube in memory hashes its layers once, at about 1 s per GB of
        values where the processor has SHA instructions, and keeps the digests.

    Args:
        data: The number of cells of every sample in every group: a
            :class:`~cytepool.cube.Cube` that records them (``n_cells``), or a
            table with one row per sample, a text column ``sample_id`` and one
            column of cell numbers (integers) per group, named by its group id: a
            ``Table``, a mapping of columns, or a pandas DataFrame. A DataFrame
            without a ``sample_id`` column is read with its index as the sample
            ids, as ``pandas.crosstab(cells["sample_id"], cells["cell_type"])``
            gives it.
        design: A :class:`~cytepool.design.Design` whose sample table has a
            ``sample_id`` column with sample ids of the data; its rows may be some
            of the samples, in any order. A design with random effects is refused:
            the fit has fixed effects only.
        contrasts: The contrasts to test by t-tests, as accepted by
            :func:`~cytepool.design.make_contrasts` and matched to the design's
            columns by name, such as ``{"stim_vs_ctrl": "conditionstim"}``. By
            default the design's named contrasts; an empty list tests none.
        f_test: The name of a factor of the design, entering its formula as a main
            effect only, whose levels are tested for equal means by an F-test
            (speckle's ANOVA), or None for no F-test. To compare some of its levels
            only, test their differences as contrasts.
        transform: ``"logit"`` (speckle's default), with half a cell added to
            every count, or ``"asin"`` for the arcsine square root.
        robust: Whether to estimate the prior of the moderation robustly, as
            speckle does by default (``eBayes(robust = TRUE)``), which protects
            against groups whose variance is unusually large.
        trend: Whether the prior variance follows the mean transformed proportion
            of the groups (``eBayes(trend = TRUE)``).

    Returns:
        The table of every contrast and of the F-test, a summary of every group,
        the samples used and left out, the proportions, the settings and a
        provenance record that identifies the data by their fingerprint.

    Raises:
        TypeError: If ``data`` is not a Cube or a table of cell numbers, a column of
            cell numbers does not hold integers, an id of the data or the design's
            ``sample_id`` is not text, ``design`` is not a Design, an option has the
            wrong type, or ``data`` or ``contrasts`` is or holds a numpy masked
            array.
        ValueError: If the cube records no cell numbers; the table has no rows or
            no group columns, or an id that is missing, empty or repeated; a cell
            number is negative or a sample has 2**53 cells or more; the design has
            random effects or no ``sample_id`` column, names samples the data do not
            have or names a sample twice; a contrast is invalid, or there is nothing
            to test; ``transform`` is neither ``"logit"`` nor ``"asin"``; ``f_test``
            does not name a factor that enters the formula as a main effect only;
            fewer than two groups have cells; the design is not of full column rank
            or leaves no residual degrees of freedom on the samples used; or it fits
            the transformed proportions of every group exactly.
    """
    cube, given = _as_cube(data)
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design whose sample table has a sample_id column with "
            f"the sample ids of the data, got {type(design).__name__}"
        )
    refuse_random_effects(design, _METHOD)
    settings: dict[str, Any] = {
        "transform": checked_choice(transform, "transform", ("logit", "asin")),
        "robust": checked_flag(robust, "robust"),
        "trend": checked_flag(trend, "trend"),
        "f_test": _factor_name(f_test),
    }
    named = "the cube" if given == "cube" else "the table of cell numbers"
    cube_rows, design_rows = matched_samples(
        cube,
        design,
        no_ids="the design's sample table has no 'sample_id' column, so its rows cannot "
        f"be matched to the samples of {named}; build the Design from a table whose "
        f"'sample_id' column holds the sample ids of {named}",
        text_ids=f"the sample ids of {named} are",
        data=named,
        data_ids=f"sample ids of {named}",
    )
    # make_contrasts refuses contrasts that are or hold a numpy masked array.
    tested = (
        design.contrasts if contrasts is None else make_contrasts(contrasts, design)
    )
    factor = None if settings["f_test"] is None else _factor(design, settings["f_test"])
    if not len(tested) and factor is None:
        raise ValueError(
            "propeller has nothing to test: the design has no named contrasts, none "
            "were given, and f_test is None. Name the contrasts in the design, as in "
            "Design(samples, '~ 0 + group', contrasts={'B_vs_A': 'groupB - "
            "groupA'}), or pass contrasts= in the same form, or name a factor of "
            "the design in f_test= to test all its levels at once; the design's "
            f"columns are: {', '.join(design.columns)}"
        )

    assert cube.n_cells is not None  # _as_cube refuses a cube without them
    sample_ids = cube.samples[_SAMPLE_ID][cube_rows]
    group_ids = cube.groups["group_id"]
    fit = _fit(
        cube.n_cells[cube_rows],  # the design's samples, in the data's order
        design.matrix[design_rows],
        design,
        settings["transform"],
    )

    tables: dict[str, Table] = {}
    moderated: ModeratedFit | None = None
    if len(tested):
        moderated = _moderate(contrasts_fit(fit.model, tested), settings)
        adjusted = adjust_p_values(moderated.p_value, "bh", axis=0)
        for k, name in enumerate(tested.names):
            compared = _compared_levels(design.model_matrix, tested.matrix[:, k])
            tables[name] = Table(
                {
                    **_descriptive(fit, group_ids, compared),
                    "t": moderated.t[:, k],
                    "p_value": moderated.p_value[:, k],
                    "adjusted_p_value": adjusted[:, k],
                }
            )
    f_table, hypothesis = None, None
    if factor is not None:
        differences = factor.differences(len(design.columns))
        names = [f"{level}_vs_{factor.levels[0]}" for level in factor.levels[1:]]
        f_fit = _moderate(contrasts_fit(fit.model, differences, names=names), settings)
        if moderated is None:
            moderated = f_fit
        f_table = Table(
            {
                **_descriptive(fit, group_ids, (factor, None)),
                "f_statistic": f_fit.f_statistic,
                "p_value": f_fit.f_p_value,
                "adjusted_p_value": adjust_p_values(f_fit.f_p_value, "bh"),
            }
        )
        hypothesis = {
            "factor": factor.name,
            "levels": list(factor.levels),
            "contrasts": {
                name: differences[:, k].tolist() for k, name in enumerate(names)
            },
        }
    assert moderated is not None  # there is a contrast or an F-test
    prior_df = np.broadcast_to(moderated.prior_df, (fit.cells.shape[1],))

    used_ids = sample_ids[fit.used]
    return CompositionResult(
        tested,
        tables,
        f_table,
        _summary_table(group_ids, fit, prior_df),
        _sample_table(sample_ids, fit),
        _proportion_table(fit, group_ids[fit.tested], used_ids),
        {
            "method": _METHOD,
            "versions": software_versions(),
            "input": given,
            "cube": cube_record(cube),
            "design": design_record(design, tested),
            "settings": {**settings, **_FIXED_SETTINGS},
            "f_test": hypothesis,
            "samples": used_ids.tolist(),
            "left_out": dict.fromkeys(
                sample_ids[~fit.used].tolist(), "no cells in any group"
            ),
            "groups": group_ids[fit.tested].tolist(),
            "groups_left_out": dict.fromkeys(
                group_ids[~fit.tested].tolist(), "no cells in the samples used"
            ),
        },
    )


# The data.


def _as_cube(data: Any) -> tuple[Cube, str]:
    """The cube that holds the cell numbers of ``data``, and what ``data`` was.

    A table of cell numbers becomes a cube without features, which checks its ids
    and gives it a fingerprint.
    """
    if isinstance(data, Cube):
        if data.n_cells is None:
            raise ValueError(
                "the cube records no cell numbers (its n_cells is None, as for bulk "
                "data), and propeller tests the proportions of each sample's cells "
                "in each group; build the cube from cells with from_cells, or pass "
                "a table of cell numbers"
            )
        return data, "cube"
    reject_masked(data, "data")
    sample_ids, group_ids, columns = _read_table(data)
    n_cells = np.empty((len(sample_ids), len(group_ids)), dtype=np.int64)
    for g, (group, values) in enumerate(zip(group_ids, columns, strict=True)):
        n_cells[:, g] = _cell_numbers(values, group, sample_ids)
    cube = Cube(
        {"counts": np.zeros((len(group_ids), len(sample_ids), 0), dtype=np.uint32)},
        samples={_SAMPLE_ID: text_column(sample_ids)},
        groups={"group_id": text_column(group_ids)},
        features={"feature_id": text_column([])},
        n_cells=n_cells,
    )
    return cube, "table"


def _read_table(data: Any) -> tuple[list[Any], list[Any], list[Any]]:
    """The sample ids, group ids and cell-number columns of a table."""
    if is_data_frame(data):
        if _SAMPLE_ID in data.columns:
            ids = data[_SAMPLE_ID].tolist()
            names = [name for name in data.columns if name != _SAMPLE_ID]
            where = "its column 'sample_id'"
        else:
            ids, names, where = list(data.index), list(data.columns), "its index"
        columns = [data[name].to_numpy() for name in names]
    elif isinstance(data, Table | Mapping):
        table = data if isinstance(data, Table) else Table(data)
        if _SAMPLE_ID not in table:
            raise ValueError(
                "the table of cell numbers has no 'sample_id' column; give one row "
                "per sample, with the sample ids in a column 'sample_id' and the "
                "cell numbers of each group in a column named by its group id"
            )
        ids = table[_SAMPLE_ID].tolist()
        names = [name for name in table.columns if name != _SAMPLE_ID]
        columns = [table[name] for name in names]
        where = "its column 'sample_id'"
    else:
        raise TypeError(
            "data must be a Cube with cell numbers, or a table of cell numbers (a "
            "Table, a mapping of columns or a pandas DataFrame) with a 'sample_id' "
            "column and one column of cell numbers per group, got "
            f"{type(data).__name__}"
        )
    if not ids:
        raise ValueError(
            "the table of cell numbers has no rows; give one row per sample"
        )
    for kind, values, place in (
        ("sample", ids, where),
        ("group", names, "its column names"),
    ):
        _check_ids(kind, values, place)
    if not names:
        raise ValueError(
            "the table of cell numbers has no group columns; give one column of cell "
            "numbers per group, named by its group id"
        )
    return ids, names, columns


def _check_ids(kind: str, values: list[Any], place: str) -> None:
    """Refuse ids of data that are missing, not text, empty or repeated."""
    for row, value in enumerate(values):
        if value is None or (isinstance(value, float) and value != value):
            raise ValueError(
                f"data has no {kind} id in position {row} of {place}; every {kind} "
                "needs an id"
            )
        if not isinstance(value, str):
            raise TypeError(
                f"the {kind} ids of data, in {place}, must be text, got {value!r} "
                f"({type(value).__name__})"
            )
        if not value:
            raise ValueError(
                f"data has an empty {kind} id in position {row} of {place}"
            )
    repeated = sorted(value for value, count in Counter(values).items() if count > 1)
    if repeated:
        raise ValueError(
            f"data has the {kind} id(s) {quoted(repeated)} more than once, in "
            f"{place}; each {kind} must appear once"
        )


def _cell_numbers(values: Any, group: str, sample_ids: list[str]) -> np.ndarray:
    """One column of cell numbers, as int64."""
    where = f"data[{group!r}]"
    array = unmasked_array(values, where)
    items = array.tolist()
    if array.dtype == object:  # a pandas column of nullable integers, for example
        if not all(
            isinstance(item, int | np.integer) and not isinstance(item, bool)
            for item in items
        ):
            raise TypeError(
                f"{where} must hold whole numbers of cells, as integers, and holds a "
                "missing or non-integer value"
            )
    elif array.dtype.kind not in "iu":
        raise TypeError(
            f"{where} must hold whole numbers of cells, as integers, got dtype "
            f"{array.dtype}"
        )
    if array.shape != (len(sample_ids),):
        raise ValueError(
            f"{where} must hold one cell number per sample ({len(sample_ids)}), got "
            f"shape {array.shape}"
        )
    for row, value in enumerate(items):
        if not 0 <= int(value) <= _INT64_MAX:
            raise ValueError(
                f"{where} has {value} cells for sample {sample_ids[row]!r}; cell "
                "numbers must be from 0 to 2**63 - 1"
            )
    return np.array(items, dtype=np.int64)


# The fit.


@dataclass(frozen=True)
class _Fit:
    """The proportions of the groups tested in the samples used, and their fit."""

    used: np.ndarray  # bool, for each design sample in the data's order
    tested: np.ndarray  # bool, for each group of the data
    sample_cells: np.ndarray  # int64, the cells of each design sample
    group_cells: np.ndarray  # int64, the cells of each group in the samples used
    cells: np.ndarray  # int64, samples used x groups tested
    proportions: np.ndarray  # float64, samples used x groups tested
    transformed: np.ndarray  # float64, samples used x groups tested
    design: np.ndarray  # the design's rows of the samples used
    model: LinearModelFit


def _fit(cells: np.ndarray, matrix: np.ndarray, design: Design, transform: str) -> _Fit:
    """Leave out the samples and groups without cells, transform, and fit."""
    totals = cells.sum(axis=1, dtype=np.float64)
    if np.any(totals >= _EXACT_TOTAL):
        raise ValueError(
            "a sample has 2**53 cells or more, more than float64 counts exactly; "
            "propeller needs exact proportions"
        )
    used = totals > 0.0
    group_cells = cells[used].sum(axis=0)
    tested = group_cells > 0
    n_tested = int(np.count_nonzero(tested))
    if n_tested < 2:
        raise ValueError(
            f"{n_tested} group{'' if n_tested == 1 else 's'} of the data "
            f"{'has' if n_tested == 1 else 'have'} cells in the samples of the "
            "design, and a test of composition compares the proportions of at least "
            "two groups"
        )
    counts = np.ascontiguousarray(cells[np.ix_(used, tested)])
    x = counts.astype(np.float64)
    n = totals[used][:, np.newaxis]
    if transform == "logit":
        # log(q / (1 - q)) for q = (x + 1/2) / (n + I/2), without forming 1 - q.
        others = (n - x) + _LOGIT_OFFSET * (n_tested - 1)
        transformed = np.log((x + _LOGIT_OFFSET) / others)
    else:
        # arcsin(sqrt(x / n)), without the loss of digits of arcsin near 1.
        transformed = np.arctan2(np.sqrt(x), np.sqrt(n - x))
    rows = np.ascontiguousarray(matrix[used])
    _check_rank(rows, design, int(np.count_nonzero(~used)))
    model = fit_linear_model(transformed, rows, coefficient_names=design.columns)
    # Residuals of a fit are exact up to rounding, a few units in the last place of
    # the values; real variation in cell numbers is many orders of magnitude larger.
    rounding = _ROUNDING * np.maximum(np.max(np.abs(transformed), axis=0), 1.0)
    if np.all(model.sigma <= rounding):
        raise ValueError(
            "the design fits the transformed proportions of every group exactly, up "
            f"to rounding (the largest residual standard deviation is "
            f"{float(np.max(model.sigma)):.1e}): the composition of the samples varies "
            "only as the design does, as when the samples of a condition are copies "
            "of one another, so no variance can be estimated and no test is possible"
        )
    return _Fit(
        used=used,
        tested=tested,
        sample_cells=cells.sum(axis=1),
        group_cells=group_cells,
        cells=counts,
        proportions=x / n,
        transformed=transformed,
        design=rows,
        model=model,
    )


def _check_rank(rows: np.ndarray, design: Design, n_left_out: int) -> None:
    """Refuse a design that is rank deficient, or leaves no residual df, on the samples."""
    n_used, n_columns = rows.shape
    left_out = (
        f", once the {n_left_out} sample{'' if n_left_out == 1 else 's'} without "
        "cells are left out"
        if n_left_out
        else ""
    )
    aliasing = _aliasing(rows)
    if aliasing.aliased:
        model = replace(design.model_matrix, matrix=rows)
        relations = "; ".join(_relation(model, aliasing, j) for j in aliasing.aliased)
        raise ValueError(
            f"the design is not of full column rank on the {n_used} samples used"
            f"{left_out}: {relations}. Remove these columns or reparametrise the "
            "design, for example by leaving out levels without samples"
        )
    if n_used <= n_columns:
        raise ValueError(
            f"the design leaves no residual degrees of freedom: it has {n_columns} "
            f"columns and {n_used} samples used{left_out}, so the variance of the "
            "transformed proportions cannot be estimated"
        )


def _moderate(fit: LinearModelFit, settings: Mapping[str, Any]) -> ModeratedFit:
    return moderated_statistics(fit, robust=settings["robust"], trend=settings["trend"])


# Factors and the levels a contrast compares.


@dataclass(frozen=True)
class _Factor:
    """A factor that enters the formula as a main effect only, and its columns.

    ``columns`` holds the design column of each level, in level order, and None for
    the first level when the factor is coded by treatment contrasts (its reference
    level); without an intercept the first factor has a column for every level.
    """

    name: str
    levels: tuple[str, ...]
    columns: tuple[int | None, ...]

    def differences(self, n_columns: int) -> np.ndarray:
        """Contrasts of every level but the first against the first, as columns."""
        matrix = np.zeros((n_columns, len(self.levels) - 1))
        reference = self.columns[0]
        for k, column in enumerate(self.columns[1:]):
            assert column is not None  # only the first level can lack a column
            matrix[column, k] = 1.0
            if reference is not None:
                matrix[reference, k] = -1.0
        return matrix

    def level_of(self, rows: np.ndarray) -> np.ndarray:
        """The level of each sample, by position, read from its row of the design."""
        level = np.zeros(rows.shape[0], dtype=np.intp)
        for position, column in enumerate(self.columns):
            if column is not None:
                level[rows[:, column] == 1.0] = position
        return level


def _main_effect(model: ModelMatrix, name: str) -> _Factor | None:
    """A factor's columns, if it enters the formula as a main effect only."""
    levels = model.levels.get(name)
    if levels is None:
        return None
    columns: dict[str | None, int] = {}
    for j, parts in enumerate(model.coding):
        if any(variable == name for variable, _ in parts):
            if len(parts) != 1:
                return None  # an interaction
            columns[parts[0][1]] = j
    if set(columns) == set(levels):
        return _Factor(name, levels, tuple(columns[level] for level in levels))
    if set(columns) == set(levels[1:]):
        return _Factor(name, levels, (None, *(columns[lv] for lv in levels[1:])))
    return None


def _factor_name(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(
        "f_test must be the name of a factor of the design (a str) or None, got "
        f"{type(value).__name__}"
    )


def _factor(design: Design, name: str) -> _Factor:
    """The factor that f_test names, checked."""
    model = design.model_matrix
    factors = [variable for variable in model.variables if variable in model.levels]
    if name not in model.levels:
        kind = (
            "a covariate, whose slope a contrast tests"
            if name in model.variables
            else f"not a variable of the formula {design.formula!r}"
        )
        raise ValueError(
            f"f_test names {name!r}, {kind}; the F-test compares the levels of a "
            f"factor, and the factors of the design are: {quoted(factors) or '(none)'}"
        )
    factor = _main_effect(model, name)
    if factor is None:
        terms = sorted(
            {
                model.terms[model.assign[j] - 1]
                for j, parts in enumerate(model.coding)
                if len(parts) > 1 and any(variable == name for variable, _ in parts)
            }
        )
        raise ValueError(
            f"f_test names {name!r}, which enters the formula in the interaction "
            f"{listing(terms)}: the differences between its levels then depend on "
            "the other variables of the interaction, so equal means across its levels "
            "is not one hypothesis. Test the contrasts of interest with contrasts=, "
            "or use a formula in which the factor is a main effect only"
        )
    return factor


def _compared_levels(
    model: ModelMatrix, contrast: np.ndarray
) -> tuple[_Factor, np.ndarray] | None:
    """The factor whose levels a contrast compares, and the weight of each level.

    None when the contrast is not a combination, with weights that sum to zero, of
    differences between the levels of a factor that is a main effect only.
    """
    support = np.flatnonzero(contrast)
    first = model.coding[support[0]]
    if len(first) != 1 or first[0][1] is None:
        return None  # the intercept, a covariate or an interaction
    factor = _main_effect(model, first[0][0])
    if factor is None or not set(support.tolist()) <= set(factor.columns):
        return None
    weights = np.array(
        [0.0 if column is None else contrast[column] for column in factor.columns]
    )
    scale = float(np.max(np.abs(contrast)))
    if factor.columns[0] is None:
        # Treatment coding: each column is its level minus the reference level.
        weights[0] = -np.sum(weights[1:])
    elif abs(np.sum(weights)) > _ZERO_WEIGHT * scale:
        return None  # a sum of level means, which compares nothing
    weights[np.abs(weights) <= _ZERO_WEIGHT * scale] = 0.0
    return factor, weights


# Results.


def _descriptive(
    fit: _Fit,
    group_ids: np.ndarray,
    compared: tuple[_Factor, np.ndarray | None] | None,
) -> dict[str, np.ndarray]:
    """The group ids, baseline proportions, mean proportions and ratio of a test.

    ``compared`` is the factor and level weights of a contrast that compares levels
    (only the levels it weighs are shown, with the ratio), the factor of an F-test
    with weights None (every level is shown), or None.
    """
    columns = {
        "group_id": group_ids[fit.tested],
        "baseline_proportion": fit.cells.sum(axis=0) / float(fit.cells.sum()),
    }
    if compared is None:
        return columns
    factor, weights = compared
    level = factor.level_of(fit.design)
    shown = (
        np.arange(len(factor.levels)) if weights is None else np.flatnonzero(weights)
    )
    means = np.full((shown.size, fit.cells.shape[1]), np.nan)
    for row, position in enumerate(shown):
        at_level = fit.proportions[level == position]
        if at_level.shape[0]:
            means[row] = [accurate_mean(np.ascontiguousarray(c)) for c in at_level.T]
        columns[f"mean_proportion_{factor.levels[position]}"] = means[row]
    if weights is not None:
        # A mean of zero gives a ratio of 0 or infinity, and both give NaN, as in R.
        with np.errstate(divide="ignore", invalid="ignore"):
            powers = np.power(means, weights[shown][:, np.newaxis])
            columns["proportion_ratio"] = np.prod(powers, axis=0)
    return columns


def _summary_table(group_ids: np.ndarray, fit: _Fit, prior_df: np.ndarray) -> Table:
    n_groups = group_ids.size
    df = np.full(n_groups, np.nan)
    df[fit.tested] = fit.model.df_residual[0]
    prior = np.full(n_groups, np.nan)
    prior[fit.tested] = prior_df
    return Table(
        {
            "group_id": group_ids,
            "n_cells": fit.group_cells,
            "tested": fit.tested,
            "reason": text_column(
                [
                    "" if tested else "no cells in the samples used"
                    for tested in fit.tested
                ]
            ),
            "df_residual": df,
            "prior_df": prior,
        }
    )


def _sample_table(sample_ids: np.ndarray, fit: _Fit) -> Table:
    return Table(
        {
            _SAMPLE_ID: sample_ids,
            "n_cells": fit.sample_cells,
            "used": fit.used,
            "reason": text_column(
                ["" if used else "no cells in any group" for used in fit.used]
            ),
        }
    )


def _proportion_table(
    fit: _Fit, group_ids: np.ndarray, sample_ids: np.ndarray
) -> Table:
    """One row per group tested and sample used, group by group."""
    n_samples, n_groups = fit.cells.shape
    return Table(
        {
            "group_id": np.repeat(group_ids, n_samples),
            _SAMPLE_ID: np.tile(sample_ids, n_groups),
            "n_cells": fit.cells.T.reshape(-1),
            "proportion": fit.proportions.T.reshape(-1),
            "transformed_proportion": fit.transformed.T.reshape(-1),
        }
    )
