"""Which genes respond to a dose in each cell type, by moderated tests of one model of
the dose groups.

:func:`~cytepool.dose_response` tests a dose series in every cell type (group) of a
cube that supports tests, in one call
(https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-tests):

1. The check. :func:`~cytepool.check_dose_design` says which cell types support
   tests, which only description and which nothing, and which dose groups each
   keeps; only cell types that support tests are fitted.
2. One model of the dose groups per cell type. The design holds the terms other than
   dose, such as ``"~ 1"`` or ``"~ sex"``; an intercept and an indicator of each
   treated dose join them, so that the coefficient theta_k of dose d_k is the
   difference of its mean from the vehicle's at the other terms. The model is
   fitted by :func:`~cytepool.limma_voom` (Law et al. 2014), the default,
   :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom` (Hoffman and
   Roussos 2021), with their expression filter, normalization and empirical Bayes
   moderation (Smyth 2004).
3. The tests, as contrasts of that one fit, so that every test of a gene uses its
   one moderated variance: ``spline``, whether the gene responds at all, along a
   natural cubic spline of dose (Desquilbet and Mariotti 2010; Harrell 2015);
   ``nonlinearity``, whether the spline departs from a straight line along the dose
   axis; ``slope``, the slope of a straight line along the axis; and
   ``dose_groups``, whether any dose group differs from the vehicle, the one-way
   analysis of variance of the dose groups with moderated variances, which NTP's
   genomic dose-response approach runs as a prefilter (NTP 2018).

The spline:
    In each cell type the spline is fitted by least squares to the means of the dose
    groups that the cell type compares, each weighted by its samples: with Z the
    columns at those doses, an intercept first, and N the samples of each group, the
    coefficients are A mu, A = (Z'NZ)^-1 Z'N. The rows of A after the intercept's
    sum to zero, so each coefficient is a contrast of the theta_k. Without
    observation weights or other terms, these are the coefficients of the spline
    fitted to the samples themselves, which the limma User's Guide fits for a time
    course (section 9.6.2); with voom's weights or other terms, the means are those
    of the weighted fit, and the spline passes through them by least squares. The
    basis is placed once, on the doses of the design's samples, so that knots and
    curves compare across cell types.

Adjustment for multiple testing:
    ``adjusted_p_value`` adjusts the p-values of one cell type and test by
    Benjamini-Hochberg (Benjamini and Hochberg 1995), as limma's ``topTable`` does,
    and ``adjusted_p_value_across_groups`` adjusts them over every cell type that ran
    the test, stacked into one family. The tests of each dose against the vehicle
    are adjusted within the cell type and dose only.

References:
    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Desquilbet L, Mariotti F (2010). Dose-response analyses using restricted cubic
    spline functions in public health research. Statistics in Medicine 29,
    1037-1057. doi:10.1002/sim.3841

    Harrell FE (2015). Regression Modeling Strategies, 2nd edition. Springer, Cham.
    doi:10.1007/978-3-319-19425-7

    NTP (2018). NTP research report on National Toxicology Program approach to
    genomic dose-response modeling. NTP Research Report 5. National Institute of
    Environmental Health Sciences. doi:10.22427/ntp-rr-5

    Tukey JW, Ciminera JL, Heyse JF (1985). Testing the statistical certainty of a
    response to increasing doses of a drug. Biometrics 41, 295-301.
    doi:10.2307/2530666

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._analysis_records import (
    cube_record,
    design_record,
    mixed_design_record,
    only_key,
    software_versions,
)
from ._columns import text_column
from ._doses import SAME_DOSE, checked_doses, dose_group_column, dose_text
from ._group_fits import COUNTS_LAYER, FITTED
from ._linear_algebra import column_space
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import positive_integer
from .cube import Cube
from .design import Contrasts, Design
from .design.formula import quote_name
from .differential_expression import (
    DifferentialExpressionResult,
    GroupFit,
    dream_voom,
    limma_trend,
    limma_voom,
)
from .dose_axes import DoseAxis
from .dose_designs import INFERENTIAL, DoseDesignReport, check_dose_design
from .stats.distributions import t_ppf
from .stats.multiple_testing import adjust_p_values
from .stats.smoothing import NaturalSplineBasis
from .table import Table

__all__ = ["DoseResponseResult", "dose_response"]

FloatArray = npt.NDArray[np.float64]

SPLINE, NONLINEARITY, SLOPE, DOSE_GROUPS = (
    "spline",
    "nonlinearity",
    "slope",
    "dose_groups",
)
TESTS = (SPLINE, NONLINEARITY, SLOPE, DOSE_GROUPS)
"""The tests, in the order of tables and printouts; ``spline`` is read first."""
EACH_DOSE = "each_dose"
"""The table of each dose against the vehicle."""
# Of two tests of the same hypothesis in a cell type, the first of this order runs
# when the tests are chosen by default.
_PRIORITY = (SPLINE, DOSE_GROUPS, SLOPE, NONLINEARITY)
_ANALYSES: dict[str, Callable[..., DifferentialExpressionResult]] = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
    "dream_voom": dream_voom,
}
# The options of the analysis that dose_response sets itself.
_SET_HERE = ("contrasts", "f_tests", "adjust_across_groups")
_ACROSS_GROUPS = "adjusted_p_value_across_groups"
# The level of the calls a printout counts, and of the confidence bands.
_FDR = 0.05
_LEVEL = 0.95
# The names of the contrasts of the spline's coefficients, after the column of the
# dose axis they weigh, such as "spline_dose_s1".
_SPLINE_PREFIX = "spline_"


class _Estimates(NamedTuple):
    """The estimates of a cell type's dose groups against the vehicle, gene by gene,
    from the moderated fit of its dose-group model.

    ``theta`` holds each gene's estimates, ``(n_genes, n_doses)``; ``covariance``
    their unscaled covariance, ``(n_genes, n_doses, n_doses)``, which times
    ``posterior_variance`` is their covariance; and ``df_total`` the degrees of
    freedom of the moderated t-tests.
    """

    theta: FloatArray
    covariance: FloatArray
    posterior_variance: FloatArray
    df_total: FloatArray


class DoseResponseResult:
    """The dose tests of every cell type of a cube, from :func:`dose_response`.

    Tables, by cell type and test (:meth:`table`), one row per gene tested, in the
    cube's feature order:

    - ``"spline"``, ``"nonlinearity"`` and ``"dose_groups"``: F-tests, with
      ``feature_id``, ``symbol`` (when the cube's feature table has one),
      ``average_log2_cpm``, ``f_statistic``, ``df`` (its denominator degrees of
      freedom), ``p_value``, ``adjusted_p_value`` (Benjamini-Hochberg over the cell
      type's genes), ``adjusted_p_value_across_groups`` (over every cell type that
      ran the test) and ``standardized_f_statistic``, the F-statistic on infinite
      denominator degrees of freedom with the same p-value. The numerator degrees of
      freedom are in the cell type's entry of ``provenance["groups"]``
      (``f_test_df1``).
    - ``"slope"``: the analysis's table of a contrast, whose ``log2_fold_change`` is
      the slope, the change in log2 expression per log10 unit of dose on a log10
      axis or per unit of dose on a linear one, with ``standard_error`` (not for
      dream), ``average_log2_cpm``, ``t``, ``p_value``, ``adjusted_p_value``,
      ``adjusted_p_value_across_groups`` and ``log_odds``; dream adds ``df`` and
      ``z_statistic``.
    - ``"each_dose"``: each treated dose the cell type compares against the vehicle,
      the rows of a dose together, doses ascending, each dose's genes in the cube's
      feature order: ``feature_id``, ``symbol``, ``dose`` (in the axis's unit), then
      the columns of the analysis's table of a contrast, adjusted within the cell
      type and dose.

    ``summary`` has one row per cell type checked: ``group_id``; ``status``, the
    check's (``"inferential"``, ``"descriptive"`` or ``"refused"``); ``tested``;
    ``reason``, why a cell type was not tested, the check's reason or the
    analysis's when its fit stopped, empty when tested; ``n_dose_groups_tested``,
    the vehicle and the treated doses compared; ``n_samples_used``;
    ``n_features_tested``; ``df_residual`` and ``df_prior`` of the fit (medians for
    dream), NaN where not tested; and ``tests_not_run``, the tests of the design
    that a tested cell type did not run, whose reasons :meth:`table` gives.

    Built by :func:`dose_response`, which keeps each cell type's fit
    (``fits``, the analysis's results without their tables, from which
    :meth:`curve` computes the fit again) and the design check
    (``design_report``). A pickled result carries the cube: a cube read from a cube
    file refers to its file, and a cube in memory carries its arrays. Immutable.

    Raises:
        TypeError: If ``summary`` is not a ``Table``, ``tables`` not a mapping of
            (cell type, test) to ``Table``, ``fits`` not a mapping of cell type to
            ``DifferentialExpressionResult``, ``design_report`` not a
            ``DoseDesignReport``, or ``provenance`` not a mapping that JSON can
            write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_summary", "_tables", "_fits", "_report", "_record")

    _summary: Table
    _tables: Mapping[tuple[str, str], Table]
    _fits: Mapping[str, DifferentialExpressionResult]
    _report: DoseDesignReport
    _record: str

    def __init__(
        self,
        summary: Table,
        tables: Mapping[tuple[str, str], Table],
        fits: Mapping[str, DifferentialExpressionResult],
        design_report: DoseDesignReport,
        provenance: Mapping[str, Any],
    ) -> None:
        if not isinstance(summary, Table):
            raise TypeError(f"summary must be a Table, got {type(summary).__name__}")
        if not isinstance(tables, Mapping):
            raise TypeError(
                "tables must be a mapping of (cell type, test) to Table, got "
                f"{type(tables).__name__}"
            )
        for key, table in tables.items():
            pair = isinstance(key, tuple) and len(key) == 2
            if not (pair and all(isinstance(part, str) for part in key)):
                raise TypeError(
                    f"tables must be keyed by (cell type, test), got the key {key!r}"
                )
            if not isinstance(table, Table):
                raise TypeError(
                    f"tables[{key!r}] must be a Table, got {type(table).__name__}"
                )
        if not isinstance(fits, Mapping):
            raise TypeError(
                "fits must be a mapping of cell type to DifferentialExpressionResult, "
                f"got {type(fits).__name__}"
            )
        for group, fit in fits.items():
            if not isinstance(fit, DifferentialExpressionResult):
                raise TypeError(
                    f"fits[{group!r}] must be a DifferentialExpressionResult, got "
                    f"{type(fit).__name__}"
                )
        if not isinstance(design_report, DoseDesignReport):
            raise TypeError(
                "design_report must be a DoseDesignReport, got "
                f"{type(design_report).__name__}"
            )
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_summary": summary,
            "_tables": MappingProxyType(dict(tables)),
            "_fits": MappingProxyType(dict(fits)),
            "_report": design_report,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DoseResponseResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DoseResponseResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            DoseResponseResult,
            (
                self._summary,
                dict(self._tables),
                dict(self._fits),
                self._report,
                self.provenance,
            ),
        )

    @property
    def summary(self) -> Table:
        """One row per cell type checked: its status, whether it was tested and why
        not, its dose groups, samples, genes and degrees of freedom."""
        return self._summary

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """Every table, by (cell type, test) and (cell type, ``"each_dose"``)."""
        return self._tables

    @property
    def fits(self) -> Mapping[str, DifferentialExpressionResult]:
        """The analysis's result of each cell type tested, without its tables:
        :meth:`DifferentialExpressionResult.fit` computes the cell type's fit
        again."""
        return self._fits

    @property
    def design_report(self) -> DoseDesignReport:
        """The check of the design, from :func:`~cytepool.check_dose_design`."""
        return self._report

    @property
    def groups(self) -> tuple[str, ...]:
        """The cell types checked, in order."""
        return tuple(self._summary["group_id"].tolist())

    @property
    def tests(self) -> tuple[str, ...]:
        """The tests of the design: those named, or those it supports when none
        were, in the order of :data:`TESTS`."""
        return tuple(json.loads(self._record)["tests"])

    @property
    def method(self) -> str:
        """The analysis: ``"limma_voom"``, ``"limma_trend"`` or ``"dream_voom"``."""
        return str(json.loads(self._record)["method"])

    @property
    def dose(self) -> DoseAxis:
        """The dose axis."""
        return DoseAxis(**self.settings["dose"])

    @property
    def spline_basis(self) -> NaturalSplineBasis | None:
        """The basis of the spline of dose, placed on the doses of the design's
        samples; None when fewer than two doses are on the axis."""
        basis = self.settings["spline_basis"]
        return None if basis is None else NaturalSplineBasis(**basis)

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the analysis, as a new dict (also in the provenance): the
        axis (``dose``, as :meth:`DoseAxis.to_dict` gives it), the spline's basis
        (``spline_basis``, as :meth:`NaturalSplineBasis.to_dict` gives it), the
        tests and ``spline_df`` as given, the analysis, the options given to it, and
        ``batch``, ``groups``, ``min_cells`` and ``min_counts``."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the analysis (``method``) and the versions of the software; the cube
        (``cube``): its fingerprint, the digests it is built from and its own
        provenance; the layer read; the model of the dose groups (``design``): its
        formula, columns, levels and the values of its variables for every sample;
        the settings, and the analysis's settings in full (``analysis_settings``,
        null when no cell type was fitted); the tests of the design; and, for every
        cell type (``groups``), its status, whether it was tested and why not, the
        doses it compares with their samples, the tests it did not run with the
        reasons, the numerator degrees of freedom of each test it ran
        (``f_test_df1``), the weights of every contrast of its fit on the model's
        columns, and the analysis's record of its fit (``fit``: the samples used and
        left out, the features tested, the span of voom's trend and the warnings).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None, name: str | None = None) -> Table:
        """Return the table of one cell type and test, or of each dose.

        Args:
            group: A cell type; it may be left out when one cell type was checked.
            name: A test the cell type ran, ``"spline"``, ``"nonlinearity"``,
                ``"slope"`` or ``"dose_groups"``, or ``"each_dose"`` for each dose
                against the vehicle. The class documentation gives the columns.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, or ``name`` is left out.
            KeyError: If the cell type was not checked or not tested, or the test was
                not run there; the message says why.
        """
        group = only_key(group, list(self.groups), "group", "cell types")
        name = only_key(name, [*self.tests, EACH_DOSE], "name", "tables")
        found = self._tables.get((group, name))
        if found is not None:
            return found
        entry = self._entry(group)
        if not entry["tested"]:
            raise KeyError(
                f"cell type {group!r} was not tested ({entry['status']}): "
                f"{entry['reason']}"
            )
        if name in entry["tests_not_run"]:
            raise KeyError(
                f"{name!r} was not run in {group!r}: {entry['tests_not_run'][name]}"
            )
        raise KeyError(
            f"no table {name!r}; the tables are: "
            f"{listing([*self.tests, EACH_DOSE])}{close_matches(str(name), TESTS)}"
        )

    def curve(
        self,
        group: str | None = None,
        genes: str | Sequence[str] | None = None,
        *,
        at: npt.ArrayLike,
    ) -> Table:
        """Return the fitted spline of genes against the vehicle, with a 95% band.

        The curve at dose x is the fitted spline minus the vehicle's mean,
        f(x) - mu_0, a contrast a(x)'theta of the cell type's dose groups: with the
        vehicle in a group of its own, 0 at the vehicle with a standard error of 0;
        with the vehicle on the axis, the spline need not pass through the
        vehicle's mean, and its value at the vehicle shows how far it misses. The
        standard error is (a(x)' V a(x))^(1/2) times the gene's posterior standard
        deviation, with V the unscaled covariance of the dose groups' estimates,
        each gene's own under voom's weights; the band is pointwise, the estimate
        plus and minus the 0.975 quantile of t on the degrees of freedom of the
        gene's moderated t-tests, as limma's ``topTable(confint = TRUE)`` forms
        intervals (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-curves).
        The fit of the cell type is computed again
        (:meth:`DifferentialExpressionResult.fit`), which takes the time of its
        analysis.

        Args:
            group: A cell type that was tested; it may be left out when one cell
                type was checked.
            genes: A feature id or a list of them, among the genes tested there;
                None for every gene tested.
            at: Doses in the axis's unit: the vehicle's 0, and doses from the lowest
                to the highest dose the cell type compares, or from 0 when the
                vehicle sits on the axis. The curve is not extrapolated.

        Returns:
            One row per gene and dose, the genes in the order given and each gene's
            doses in the order of ``at``: ``feature_id``, ``dose``,
            ``log2_fold_change``, ``standard_error``, ``ci_lower`` and ``ci_upper``.

        Raises:
            TypeError: If ``genes`` is not a feature id or a list of them, or ``at``
                is not real-valued, or either is or holds a numpy masked array.
            ValueError: If ``group`` is left out and several cell types were
                checked; the result is of ``dream_voom``, whose fits do not keep
                the degrees of freedom a band needs; the cell type's doses cannot
                estimate the spline, or no spline is placed; or a dose of ``at`` is
                not finite, is negative, or lies outside the doses the cell type
                compares.
            KeyError: If the cell type was not tested, or a gene was not tested
                there.
        """
        reject_masked(genes, "genes")
        reject_masked(at, "at")
        group = only_key(group, list(self.groups), "group", "cell types")
        entry = self._entry(group)
        if not entry["tested"]:
            raise KeyError(
                f"cell type {group!r} was not tested ({entry['status']}): "
                f"{entry['reason']}"
            )
        if self.method == "dream_voom":
            raise ValueError(
                "curve() needs each curve point's degrees of freedom, which dream's "
                "fits do not keep; the tests of a dream_voom result are complete, "
                "and limma_voom gives curves with bands"
            )
        axis, basis = self.dose, self.spline_basis
        doses = np.array(entry["doses"], dtype=np.float64)
        samples = np.array(entry["samples"], dtype=np.float64)
        weights = (
            None if basis is None else _spline_weights(axis, basis, doses, samples)
        )
        if basis is None or weights is None:
            raise ValueError(
                f"no curve in {group!r}: {_no_spline_reason(axis, basis, doses)}"
            )
        at_doses = _curve_doses(at, axis, doses, group)
        fit = self._fits[group].fit(group)
        names = [dose_group_column(axis.column, dose) for dose in doses[1:]]
        estimates = _dose_group_estimates(fit, names)
        rows = _gene_rows(genes, fit.features["feature_id"].tolist(), group)
        columns = axis.design_columns({axis.column: at_doses}, basis)
        z = np.column_stack([np.ones(at_doses.size), *columns.values()])
        # The curve's weights on the treated doses: f(x) - mu_0 = c(x)'theta. With
        # the vehicle in a group of its own, the spline passes through the vehicle's
        # mean, so the curve is 0 there, exactly.
        c = (z @ weights)[:, 1:]
        if axis.vehicle == "own_group":
            c[at_doses == 0.0] = 0.0
        theta = estimates.theta[rows]
        estimate = theta @ c.T
        variance = (
            np.einsum("ak,gkl,al->ga", c, estimates.covariance[rows], c)
            * estimates.posterior_variance[rows, None]
        )
        error = np.sqrt(np.maximum(variance, 0.0))
        quantile = np.asarray(
            t_ppf(0.5 + _LEVEL / 2.0, estimates.df_total[rows]), dtype=np.float64
        )
        margin = quantile[:, None] * error
        ids = np.asarray(fit.features["feature_id"])[rows]
        return Table(
            {
                "feature_id": np.repeat(ids, at_doses.size),
                "dose": np.tile(at_doses, rows.size),
                "log2_fold_change": estimate.ravel(),
                "standard_error": error.ravel(),
                "ci_lower": (estimate - margin).ravel(),
                "ci_upper": (estimate + margin).ravel(),
            }
        )

    def _entry(self, group: str) -> dict[str, Any]:
        """The record of a cell type."""
        for entry in self.provenance["groups"]:
            if entry["group_id"] == group:
                found: dict[str, Any] = entry
                return found
        raise KeyError(
            f"no cell type {group!r} was checked; cell types are: "
            f"{listing(list(self.groups)) or '(none)'}"
        )

    def __repr__(self) -> str:
        record = self.provenance
        settings = record["settings"]
        axis = self.dose
        tested = int(np.count_nonzero(self._summary["tested"]))
        n_groups = len(self._summary)
        names = [f"{t} (read first)" if t == SPLINE else t for t in self.tests]
        lines = [
            f"DoseResponseResult ({record['method']}): {n_groups:,} cell "
            f"type{'' if n_groups == 1 else 's'}, {tested:,} tested; tests: "
            + ", ".join(names),
            "  dose: " + _axis_text(axis, settings["spline_basis"]),
        ]
        for entry, row in zip(record["groups"], range(n_groups), strict=True):
            group = entry["group_id"]
            if not entry["tested"]:
                lines.append(
                    f"  {group}: not tested, {entry['status']}: {entry['reason']}"
                )
                continue
            genes = int(self._summary["n_features_tested"][row])
            calls = [
                f"{test} {self._calls(group, test):,}"
                + (f" ({_slope_unit(axis)})" if test == SLOPE else "")
                for test in self.tests
                if (group, test) in self._tables
            ]
            text = (
                f"  {group}: {len(entry['doses'])} dose groups, "
                f"{sum(entry['samples']):g} samples, {genes:,} genes; calls at "
                f"{_FDR:.0%} FDR: " + ", ".join(calls)
            )
            if entry["tests_not_run"]:
                text += "; not run: " + ", ".join(entry["tests_not_run"])
            lines.append(text)
        return fitted_lines("\n".join(lines))

    def _calls(self, group: str, test: str) -> int:
        """The genes whose adjusted p-value of a test is below the printout's FDR."""
        adjusted = self._tables[(group, test)]["adjusted_p_value"]
        return int(np.count_nonzero(adjusted < _FDR))


def dose_response(
    cube: Cube,
    design: Design,
    *,
    dose: DoseAxis,
    tests: str | Sequence[str] | None = None,
    spline_df: int | None = None,
    analysis: Callable[..., DifferentialExpressionResult] = limma_voom,
    batch: str | None = None,
    groups: str | Sequence[str] | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    **options: Any,
) -> DoseResponseResult:
    """Test which genes respond to a dose in every cell type of a cube.

    For every cell type (group) of the cube that supports tests: one moderated fit
    of the dose groups, the vehicle and the doses that keep samples there, by
    ``analysis``, and four tests as contrasts of it (the module documentation gives
    the model):

    - ``"spline"``: whether the gene responds at all, by the F-test of a natural
      cubic spline of dose fitted to the means of the dose groups, with
      ``{column}_treated`` when the vehicle is a group of its own. Read it first.
    - ``"nonlinearity"``: whether the spline departs from a straight line along the
      axis, by the F-test of its restricted cubic columns after the first; with the
      vehicle in its own group, the line runs through the treated doses.
    - ``"slope"``: the slope of a straight line fitted to the means along the axis,
      by a moderated t-test, as a change in log2 expression per log10 unit of dose
      on a log10 axis or per unit of dose on a linear one. With the vehicle in a
      group of its own, the line runs through the treated doses only, so a gene that
      jumps at the lowest dose and then stays flat has a slope near 0, and the
      vehicle's difference is tested by the other two; with the vehicle on the axis,
      it runs through every dose, the regression on log dose of Tukey's trend test
      (Tukey, Ciminera and Heyse 1985).
    - ``"dose_groups"``: whether any dose group differs from the vehicle, by the
      F-test of every dose against the vehicle, the one-way analysis of variance of
      the dose groups with moderated variances: the counterpart of the ANOVA
      prefilter of NTP's genomic dose-response approach (NTP 2018).

    Each dose against the vehicle is also tested, in the table ``"each_dose"``.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-tests
    explains the tests and how to read them.

    The design is checked first by :func:`~cytepool.check_dose_design`, with
    ``batch``, ``groups``, ``min_cells`` and ``min_counts``, and the check is kept as
    the result's ``design_report``: cell types that it calls descriptive or refused
    are listed with its reason and not fitted. A cell type tests the dose groups
    that keep a sample there and that the design can compare with the vehicle; a
    test that its dose groups cannot estimate, such as a spline of more columns than
    its doses support, is not run there, with the reason.

    Args:
        cube: A cube with a ``counts`` layer, as :func:`~cytepool.limma_voom` takes.
        design: A :class:`~cytepool.Design` of the terms other than dose, such as
            ``Design(cube.samples, "~ 1")`` or ``"~ sex"``, with random intercepts
            for ``dream_voom``, whose sample table has a ``sample_id`` column with
            cube sample ids and the dose column; its rows may be some of the cube's
            samples, in any order. Its terms may not use the dose column or the
            axis's columns, and it may not name contrasts: dose_response adds the
            dose groups and tests them itself.
        dose: The :class:`~cytepool.DoseAxis` of the series. The dose groups are the
            distinct doses of the design's samples.
        tests: A test name, or a list of them, among ``"spline"``,
            ``"nonlinearity"``, ``"slope"`` and ``"dose_groups"``; None runs every
            test the design supports, leaving out in each cell type a test of the
            same hypothesis as another, which happens when the dose groups are few.
            ``"spline"`` and ``"slope"`` need two doses on the axis (the treated
            doses when the vehicle is a group of its own, every dose when it is on
            the axis), and ``"nonlinearity"`` a spline of two columns or more.
        spline_df: The number of spline columns, at least 1 and at most the doses on
            the axis minus 1; None gives 2 when the axis holds 4 doses or more, and
            1 otherwise. The knots are placed by :meth:`DoseAxis.spline_basis` with
            its rule ``"levels"``, in the restricted cubic form.
        analysis: :func:`~cytepool.limma_voom` (by default),
            :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom`; a design
            with random intercepts needs ``dream_voom``.
        batch: A column of the design's samples that names each sample's processing
            batch, which the check uses to find dose groups confounded with it. It
            is checked, not modelled: a batch to adjust for belongs in the design.
        groups: A group id or a list of them; all groups by default.
        min_cells: The fewest cells a sample needs in a cell type, as in
            :func:`~cytepool.limma_voom`; for the check and the fits.
        min_counts: The fewest counts, summed over all features, a sample needs in a
            cell type, as in :func:`~cytepool.limma_voom`; for the check and the
            fits.
        **options: The analysis's other options, such as ``prior_weights``,
            ``quality_weights``, ``robust``, ``expression_filter``,
            ``contrast_standard_errors``, ``n_threads``, ``prior_count``
            (``limma_trend``) or ``df_method`` (``dream_voom``); their values are
            checked by the analysis.

    Returns:
        The tables of every cell type tested, the summary, the design check, each
        cell type's fit, the settings and a provenance record.

    Warns:
        RuntimeWarning: As the analysis warns, such as when sample quality weights
            cannot be estimated in a cell type.

    Raises:
        TypeError: If ``analysis`` is not one of the three, an option is one the
            analysis does not take or one that dose_response sets
            (``contrasts``, ``f_tests``, ``adjust_across_groups``), ``tests`` or
            ``spline_df`` has the wrong type, or an argument is of a type that
            :func:`~cytepool.check_dose_design` or the analysis refuses.
        ValueError: If ``tests`` names no test or a test twice, or names a test
            that the design cannot support; ``spline_df`` is below 1 or above the
            doses on the axis minus 1, or given without two doses on the axis; the
            design names contrasts, has random intercepts with an analysis of
            limma or none with ``dream_voom``; the sample table already holds a
            column of the name of a dose group's indicator; or as
            :func:`~cytepool.check_dose_design` and the analysis raise.
        KeyError: If a group is not in the cube.
        RuntimeError: If a fit used other samples than the check, or cannot estimate
            a dose group that the check calls estimable.
    """
    given = _tests_setting(tests)
    if spline_df is not None:
        spline_df = positive_integer(spline_df, "spline_df")
    method = _analysis_name(analysis, design)
    _check_options(method, options)
    if isinstance(design, Design) and len(design.contrasts):
        raise ValueError(
            f"the design names contrasts ({quoted(list(design.contrasts.names))}), "
            "which dose_response does not test: it adds the dose groups and tests "
            "them itself; build the design without contrasts, and test other "
            "contrasts with limma_voom"
        )
    report = check_dose_design(
        cube,
        design,
        dose=dose,
        batch=batch,
        min_cells=min_cells,
        min_counts=min_counts,
        groups=groups,
    )
    levels = np.unique(report.doses["dose"])
    splines = _splines(dose, design.samples, levels, given, spline_df)
    statuses = report.summary["status"].tolist()
    reasons = report.summary["reason"].tolist()
    model = _dose_group_model(design, dose, levels) if INFERENTIAL in statuses else None
    fits: list[_CellTypeFit] = []
    groups_checked = report.summary["group_id"].tolist()
    for group, status, reason in zip(groups_checked, statuses, reasons, strict=True):
        if status != INFERENTIAL:
            fits.append(_CellTypeFit(group, status, reason))
            continue
        assert model is not None  # built when a cell type supports tests
        comparison = _comparison(
            report.doses.filter(group_id=group), dose, splines, given is None
        )
        fits.append(
            _fit_cell_type(
                cube,
                model,
                group,
                comparison,
                analysis,
                min_cells=min_cells,
                min_counts=min_counts,
                options=options,
            )
        )
    _adjust_across_groups(fits, splines.tests)
    return _result(
        cube, report, model, splines, fits, given, spline_df, method, options
    )


# The checks of the arguments.


def _tests_setting(value: Any) -> list[str] | None:
    """The tests named, in the order given, or None for every test the design
    supports."""
    reject_masked(value, "tests")
    if value is None:
        return None
    named = [value] if isinstance(value, str) else value
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            f"tests must be a test name or a list of them, among {quoted(TESTS)}, or "
            f"None for every test the design supports, got {type(value).__name__}"
        )
    listed = list(named)
    if not listed:
        raise ValueError(
            "tests is empty; name at least one test, or pass None for every test the "
            "design supports"
        )
    for name in listed:
        if not isinstance(name, str):
            raise TypeError(
                f"tests must be test names (str), got {type(name).__name__} {name!r}"
            )
        if name not in TESTS:
            raise ValueError(
                f"tests names {name!r}, which is not a test"
                f"{close_matches(name, TESTS)}; the tests are {quoted(TESTS)}"
            )
    repeated = sorted({name for name in listed if listed.count(name) > 1})
    if repeated:
        raise ValueError(f"tests names {quoted(repeated)} more than once")
    return listed


def _analysis_name(analysis: Any, design: Any) -> str:
    """The name of the analysis, which must be one of the three and suit the
    design's random effects."""
    method = next(
        (name for name, known in _ANALYSES.items() if analysis is known), None
    )
    if method is None:
        raise TypeError(
            f"analysis must be limma_voom, limma_trend or dream_voom, got {analysis!r}"
        )
    if not isinstance(design, Design):
        return method  # check_dose_design refuses it, with its own message
    if design.random and method != "dream_voom":
        effects = ", ".join(str(effect) for effect in design.random)
        raise ValueError(
            f"the design has random effects, {effects}, which {method} cannot fit; "
            "pass analysis=dream_voom, which fits random intercepts"
        )
    if not design.random and method == "dream_voom":
        raise ValueError(
            "dream_voom fits random effects, and the design has none; give the "
            "design random intercepts such as random='(1 | donor)', or use the "
            "default analysis, limma_voom"
        )
    return method


def _check_options(method: str, options: Mapping[str, Any]) -> None:
    """Refuse the options the analysis does not take and those dose_response sets,
    before anything runs."""
    for name, value in options.items():
        reject_masked(value, name)
    # The record keeps the options as given, so they must be values JSON can write.
    json.dumps(dict(options), default=_plain)
    accepted = [
        name
        for name, parameter in inspect.signature(_ANALYSES[method]).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    for name in options:
        if name in _SET_HERE:
            raise TypeError(
                f"dose_response sets {name} itself, from the dose groups of each cell "
                "type; leave it out"
            )
        if name not in accepted:
            # The options to name are those dose_response neither sets nor takes.
            taken = (*_SET_HERE, "groups", "min_cells", "min_counts")
            others = [option for option in accepted if option not in taken]
            raise TypeError(
                f"{method} has no option {name!r}{close_matches(name, others)}; its "
                f"options are: {listing(others, limit=len(others))}"
            )


# The splines of the design.


@dataclass(frozen=True)
class _Splines:
    """The bases placed on the design's doses, and the tests of the design."""

    basis: NaturalSplineBasis | None  # None with fewer than two doses on the axis
    slope_basis: NaturalSplineBasis | None  # one column, the position minus the lowest
    tests: tuple[str, ...]


def _splines(
    axis: DoseAxis,
    samples: Table,
    levels: npt.NDArray[np.float64],
    given: list[str] | None,
    spline_df: int | None,
) -> _Splines:
    """The spline bases on the design's doses, and the tests the design supports:
    those given, refused with the reason when it cannot support them, or every test
    it supports."""
    on_axis = levels if axis.vehicle == "on_axis" else levels[levels > 0.0]
    m = on_axis.size
    where = (
        " (the treated doses: the vehicle is a group of its own)"
        if axis.vehicle == "own_group"
        else ""
    )
    doses = f"{dose_text(on_axis, axis.unit)}" if m else "none"
    if m < 2:
        if spline_df is not None:
            raise ValueError(
                f"a spline needs two doses on the axis, and the design has {m}, "
                f"{doses}{where}; leave spline_df out"
            )
        along = [test for test in (given or []) if test != DOSE_GROUPS]
        if along:
            raise ValueError(
                f"{quoted(along)} run along the dose axis and need two doses on it, "
                f"and the design has {m}, {doses}{where}; test dose_groups, or place "
                "the vehicle on the axis with DoseAxis(..., vehicle='on_axis')"
            )
        return _Splines(None, None, (DOSE_GROUPS,))
    df = spline_df if spline_df is not None else (2 if m >= 4 else 1)
    if df > m - 1:
        raise ValueError(
            f"spline_df={df} needs {df + 1} distinct doses on the axis, and the design "
            f"has {m}, {doses}{where}; lower spline_df to at most {m - 1}"
        )
    if given is not None and NONLINEARITY in given and df < 2:
        fix = (
            "pass spline_df=2"
            if m >= 3
            else "it needs three doses on the axis, and the design has two"
        )
        raise ValueError(
            "nonlinearity tests the spline's columns after the first, so it needs a "
            f"spline of two columns or more, and the spline has {df}; {fix}"
        )
    basis = axis.spline_basis(samples, df=df)
    slope_basis = axis.spline_basis(samples, df=1)
    supported = [test for test in TESTS if test != NONLINEARITY or df >= 2]
    chosen = supported if given is None else [t for t in TESTS if t in given]
    return _Splines(basis, slope_basis, tuple(chosen))


# The model of the dose groups.


@dataclass(frozen=True)
class _Model:
    """The design of the dose groups, with the names of the treated doses'
    indicators."""

    design: Design
    column: str  # the dose column
    unit: str  # the doses' unit
    indicators: dict[float, str]  # dose -> the name of its indicator column


def _dose_group_model(
    design: Design, axis: DoseAxis, levels: npt.NDArray[np.float64]
) -> _Model:
    """The design's terms, an intercept, and an indicator of each treated dose."""
    samples = design.samples
    doses = np.asarray(samples[axis.column], dtype=np.float64)
    indicators = {
        float(level): dose_group_column(axis.column, level)
        for level in levels
        if level > 0.0
    }
    taken = [name for name in indicators.values() if name in samples]
    if taken:
        raise ValueError(
            f"the sample table already has the column(s) {quoted(taken)}, the names "
            "dose_response gives the indicators of the dose groups; rename them"
        )
    columns = {
        name: (doses == level).astype(np.float64) for level, name in indicators.items()
    }
    # "+ 1" keeps the vehicle's mean as the intercept, even under "~ 0".
    formula = (
        f"{design.formula} + 1 + "
        f"{' + '.join(quote_name(name) for name in indicators.values())}"
    )
    random = [str(effect) for effect in design.random] or None
    model = Design(
        samples.with_columns(columns),
        formula,
        random=random,
        levels={name: list(values) for name, values in design.levels.items()},
    )
    return _Model(model, axis.column, axis.unit, indicators)


# One cell type: what it compares, its contrasts, and its fit.


@dataclass(frozen=True)
class _Comparison:
    """The dose groups a cell type compares, and its tests as contrasts of them."""

    doses: npt.NDArray[np.float64]  # the vehicle first, then the treated doses
    samples: npt.NDArray[np.float64]  # the samples of each
    # Each contrast's weights on the treated doses, by name: each dose against the
    # vehicle, named after its indicator; the spline's coefficients; the slope.
    contrasts: dict[str, npt.NDArray[np.float64]]
    tests: dict[str, list[str]]  # each test run, in the order of TESTS: its contrasts
    not_run: dict[str, str]  # each test of the design not run here: why


def _comparison(
    rows: Table, axis: DoseAxis, splines: _Splines, by_default: bool
) -> _Comparison:
    """The dose groups that a cell type compares, from its rows of the check's doses
    table, and its tests as contrasts of the treated doses against the vehicle.

    Each test's contrasts span its hypothesis: every dose equal to the vehicle, or a
    set of the spline's coefficients equal to zero, as linear functions of the
    doses' differences from the vehicle.
    """
    doses = np.asarray(rows["dose"], dtype=np.float64)
    compared = (doses == 0.0) | np.asarray(rows["estimable"], dtype=bool)
    levels = doses[compared]
    samples = np.asarray(rows["n_samples_used"], dtype=np.float64)[compared]
    names = [dose_group_column(axis.column, dose) for dose in levels[1:]]
    contrasts = dict(zip(names, np.eye(len(names)), strict=True))
    tests: dict[str, list[str]] = {}
    not_run: dict[str, str] = {}
    wanted = splines.tests
    if SPLINE in wanted or NONLINEARITY in wanted:
        assert splines.basis is not None  # the design supports both tests
        spline = _spline_weights(axis, splines.basis, levels, samples)
        if spline is None:
            reason = _no_spline_reason(axis, splines.basis, levels)
            not_run |= {
                test: reason for test in (SPLINE, NONLINEARITY) if test in wanted
            }
        else:
            # The spline's coefficients after the intercept, named after the
            # columns of the axis they weigh, such as "spline_dose_s1".
            columns = axis.design_columns({axis.column: levels[:1]}, splines.basis)
            labels = [_SPLINE_PREFIX + column for column in columns]
            contrasts |= dict(zip(labels, spline[1:, 1:], strict=True))
            if SPLINE in wanted:
                tests[SPLINE] = labels
            if NONLINEARITY in wanted:
                # The restricted cubic columns after the first.
                first = 2 if axis.vehicle == "own_group" else 1
                tests[NONLINEARITY] = labels[first:]
    if SLOPE in wanted:
        assert splines.slope_basis is not None  # the design supports the slope
        line = _spline_weights(axis, splines.slope_basis, levels, samples)
        if line is None:
            not_run[SLOPE] = (
                "with the vehicle in a group of its own, the slope runs along the "
                "treated doses, and this cell type compares one, "
                f"{dose_text(levels[1:], axis.unit)}"
            )
        else:
            contrasts[SLOPE] = line[-1, 1:]
            tests[SLOPE] = [SLOPE]
    if DOSE_GROUPS in wanted:
        tests[DOSE_GROUPS] = names
    if by_default:
        _leave_out_repeats(tests, contrasts, not_run)
    ordered = {test: tests[test] for test in TESTS if test in tests}
    reasons = {test: not_run[test] for test in TESTS if test in not_run}
    return _Comparison(levels, samples, contrasts, ordered, reasons)


def _leave_out_repeats(
    tests: dict[str, list[str]],
    contrasts: Mapping[str, npt.NDArray[np.float64]],
    not_run: dict[str, str],
) -> None:
    """Leave out each test of the same hypothesis as one before it in ``_PRIORITY``,
    with the reason: with few dose groups, the spline can pass through every group's
    mean, or be the straight line itself."""
    kept: list[str] = []
    for test in _PRIORITY:
        if test not in tests:
            continue
        rows = np.array([contrasts[name] for name in tests[test]])
        same = next(
            (
                other
                for other in kept
                if _same_hypothesis(
                    np.array([contrasts[name] for name in tests[other]]), rows
                )
            ),
            None,
        )
        if same is None:
            kept.append(test)
            continue
        del tests[test]
        not_run[test] = (
            f"it tests the same hypothesis as {same} in this cell type, which runs "
            "in its place"
        )


def _spline_weights(
    axis: DoseAxis,
    basis: NaturalSplineBasis,
    doses: npt.NDArray[np.float64],
    samples: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """The weights A = (Z'NZ)^-1 Z'N that give the spline's coefficients, the
    intercept first, from the means of the dose groups at ``doses`` with ``samples``
    each; None when these dose groups cannot estimate the spline."""
    columns = axis.design_columns({axis.column: doses}, basis)
    z = np.column_stack([np.ones(doses.size), *columns.values()])
    if column_space(z).aliased:
        return None
    root = np.sqrt(samples)
    q, r = np.linalg.qr(z * root[:, None])
    weights: npt.NDArray[np.float64] = np.linalg.solve(r, q.T * root)
    return weights


def _no_spline_reason(
    axis: DoseAxis,
    basis: NaturalSplineBasis | None,
    doses: npt.NDArray[np.float64],
) -> str:
    """Why the dose groups of a cell type cannot estimate the spline."""
    if basis is None:
        return (
            "a spline needs two doses on the axis, and the design has fewer (the "
            "vehicle is a group of its own)"
        )
    width = basis.df + (2 if axis.vehicle == "own_group" else 1)
    text = dose_text(doses, axis.unit)
    if doses.size < width:
        return (
            f"a spline of {basis.df} column{'' if basis.df == 1 else 's'} needs "
            f"{width} dose groups, the vehicle included, and this cell type compares "
            f"{doses.size}, {text}"
        )
    return (
        f"the spline's columns are linearly dependent at the doses this cell type "
        f"compares, {text}"
    )


def _same_hypothesis(
    first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
) -> bool:
    """Whether two tests' contrasts span the same space, so that they test the same
    hypothesis and give the same F-statistic."""
    rank = len(column_space(first.T).kept)
    return (
        rank
        == len(column_space(second.T).kept)
        == len(column_space(np.vstack([first, second]).T).kept)
    )


@dataclass(frozen=True)
class _CellTypeFit:
    """What happened in one cell type: its status in the check and, when it supports
    tests, what it compares, its fit and the tables of its tests."""

    group: str
    status: str  # the check's
    reason: str  # why it was not tested: the check's reason or the analysis's
    comparison: _Comparison | None = None
    result: DifferentialExpressionResult | None = None  # the analysis's, with tables
    tables: dict[str, Table] = field(default_factory=dict)


def _fit_cell_type(
    cube: Cube,
    model: _Model,
    group: str,
    comparison: _Comparison,
    analysis: Callable[..., DifferentialExpressionResult],
    *,
    min_cells: Any,
    min_counts: Any,
    options: Mapping[str, Any],
) -> _CellTypeFit:
    """Fit a cell type's dose groups and run its tests, as contrasts of the fit."""
    contrasts, f_tests = _contrasts(model, comparison)
    result = analysis(
        cube,
        model.design,
        groups=[group],
        contrasts=contrasts,
        f_tests=f_tests or None,
        min_cells=min_cells,
        min_counts=min_counts,
        adjust_across_groups=False,
        **options,
    )
    (entry,) = result.provenance["groups"]
    if entry["status"] != FITTED:
        return _CellTypeFit(group, INFERENTIAL, entry["reason"], comparison)
    _check_agreement(entry, model, comparison, group)
    tables: dict[str, Table] = {}
    for test in comparison.tests:
        table = result.table(group, test)
        if test != SLOPE:
            table = table.select(
                [c for c in table.columns if not c.startswith("log2_fold_change_")]
            )
        tables[test] = table
    tables[EACH_DOSE] = _each_dose_table(result, group, model, comparison)
    return _CellTypeFit(group, INFERENTIAL, "", comparison, result, tables)


def _contrasts(
    model: _Model, comparison: _Comparison
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, list[str]]]:
    """A cell type's contrasts on the columns of the model of the dose groups, by
    name, and its F-tests, the tests of more than one contrast."""
    columns = list(model.design.columns)
    positions = [
        columns.index(model.indicators[float(dose)]) for dose in comparison.doses[1:]
    ]
    contrasts = {}
    for name, weights in comparison.contrasts.items():
        vector = np.zeros(len(columns))
        vector[positions] = weights
        contrasts[name] = vector
    f_tests = {
        test: members for test, members in comparison.tests.items() if test != SLOPE
    }
    return contrasts, f_tests


def _check_agreement(
    entry: Mapping[str, Any], model: _Model, comparison: _Comparison, group: str
) -> None:
    """Raise unless the fit used as many samples of each dose group as the check
    counted, which set the weights of the spline, and estimated every contrast."""
    doses = dict(
        zip(
            model.design.samples["sample_id"].tolist(),
            np.asarray(model.design.samples[model.column], dtype=np.float64).tolist(),
            strict=True,
        )
    )
    used = [doses[sample] for sample in entry["samples"]]
    counted = [used.count(float(dose)) for dose in comparison.doses]
    expected = comparison.samples.astype(int).tolist()
    if counted != expected or entry["not_estimable"]:
        missing = list(entry["not_estimable"])
        raise RuntimeError(
            f"the fit of {group!r} used {counted} samples of the dose groups "
            f"{dose_text(comparison.doses, model.unit)}, where the check counted "
            f"{expected}"
            + (f", and could not estimate {quoted(missing)}" if missing else "")
            + ": the check and the analysis disagree, as they can when a lazy cube's "
            "source changes between them; analyse the cube again"
        )


def _each_dose_table(
    result: DifferentialExpressionResult,
    group: str,
    model: _Model,
    comparison: _Comparison,
) -> Table:
    """The table of each treated dose against the vehicle: the analysis's tables of
    its indicators, stacked in dose order, with each row's dose after the gene."""
    tables = [
        result.table(group, model.indicators[float(dose)])
        for dose in comparison.doses[1:]
    ]
    first = tables[0]
    after = "symbol" if "symbol" in first else "feature_id"
    columns: dict[str, Any] = {}
    for name in first.columns:
        columns[name] = np.concatenate([table[name] for table in tables])
        if name == after:
            columns["dose"] = np.repeat(comparison.doses[1:], len(first))
    return Table(columns)


def _adjust_across_groups(fits: list["_CellTypeFit"], tests: Sequence[str]) -> None:
    """Add to each table of a test the Benjamini-Hochberg adjustment of its p-values
    over every cell type that ran it, after its ``adjusted_p_value``."""
    for test in tests:
        running = [fit for fit in fits if test in fit.tables]
        if not running:
            continue
        adjusted = adjust_p_values(
            np.concatenate([fit.tables[test]["p_value"] for fit in running]), "bh"
        )
        start = 0
        for fit in running:
            table = fit.tables[test]
            stop = start + len(table)
            columns: dict[str, Any] = {}
            for name in table.columns:
                columns[name] = table[name]
                if name == "adjusted_p_value":
                    columns[_ACROSS_GROUPS] = adjusted[start:stop]
            fit.tables[test] = Table(columns)
            start = stop


# The result.


def _result(
    cube: Cube,
    report: DoseDesignReport,
    model: _Model | None,
    splines: _Splines,
    fits: list[_CellTypeFit],
    given: list[str] | None,
    spline_df: int | None,
    method: str,
    options: Mapping[str, Any],
) -> DoseResponseResult:
    check = report.settings
    first = next((fit.result for fit in fits if fit.result is not None), None)
    design = None
    if model is not None:
        columns = model.design.columns
        record = (mixed_design_record if method == "dream_voom" else design_record)(
            model.design, Contrasts(np.empty((len(columns), 0)), (), columns)
        )
        design = {key: value for key, value in record.items() if key != "contrasts"}
    settings = {
        "dose": check["dose"],
        "spline_basis": None if splines.basis is None else splines.basis.to_dict(),
        "tests": given,
        "spline_df": spline_df,
        "analysis": method,
        "options": json.loads(json.dumps(dict(options), default=_plain)),
        "batch": check["batch"],
        "groups": check["groups"],
        "min_cells": check["min_cells"],
        "min_counts": check["min_counts"],
    }
    provenance = {
        "method": method,
        "versions": software_versions(),
        "cube": cube_record(cube),
        "layer": COUNTS_LAYER,
        "design": design,
        "settings": settings,
        "analysis_settings": None if first is None else first.settings,
        "tests": list(splines.tests),
        "groups": [_group_record(fit) for fit in fits],
    }
    tables = {
        (fit.group, name): table for fit in fits for name, table in fit.tables.items()
    }
    kept = {
        fit.group: _without_tables(fit.result) for fit in fits if fit.result is not None
    }
    return DoseResponseResult(_summary(fits, report), tables, kept, report, provenance)


def _plain(value: Any) -> Any:
    """A numpy scalar among the options, as the Python number JSON writes."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        "the options must hold numbers, strings, lists and mappings, which the record "
        f"keeps, got {type(value).__name__}"
    )


def _without_tables(
    result: DifferentialExpressionResult,
) -> DifferentialExpressionResult:
    """A cell type's result of the analysis without its tables, which the dose
    tables replace, and without the record of the cube, which the dose result keeps
    once; its fit can still be computed again."""
    record = result.provenance
    del record["cube"]
    return DifferentialExpressionResult(
        result.method,
        result.contrasts,
        result.summary,
        result.samples,
        {},
        record,
        cube=result.cube,
        design=result.design,
    )


def _group_record(fit: _CellTypeFit) -> dict[str, Any]:
    """The record of a cell type: what it compares, what it ran and its fit."""
    comparison, result = fit.comparison, fit.result
    tested = result is not None
    entry: dict[str, Any] = {
        "group_id": fit.group,
        "status": fit.status,
        "tested": tested,
        "reason": fit.reason,
        "doses": [] if comparison is None else comparison.doses.tolist(),
        "samples": [] if comparison is None else comparison.samples.tolist(),
        "tests_not_run": dict(comparison.not_run) if tested and comparison else {},
        "f_test_df1": {},
        "contrasts": {},
        "fit": None,
    }
    if result is not None:
        record = result.provenance
        (group,) = record["groups"]
        df1 = {**group["f_test_df1"], **({SLOPE: 1} if SLOPE in fit.tables else {})}
        entry["f_test_df1"] = {test: df1[test] for test in TESTS if test in df1}
        entry["contrasts"] = record["design"]["contrasts"]
        entry["fit"] = group
    return entry


def _summary(fits: list[_CellTypeFit], report: DoseDesignReport) -> Table:
    summary = report.summary
    tested = [fit.result for fit in fits]

    def of_fit(column: str) -> FloatArray:
        return np.array(
            [np.nan if r is None else float(r.summary[column][0]) for r in tested],
            dtype=np.float64,
        )

    return Table(
        {
            "group_id": summary["group_id"],
            "status": summary["status"],
            "tested": np.array([r is not None for r in tested], dtype=bool),
            "reason": text_column([fit.reason for fit in fits]),
            "n_dose_groups_tested": np.array(
                [
                    0
                    if r is None or fit.comparison is None
                    else fit.comparison.doses.size
                    for fit, r in zip(fits, tested, strict=True)
                ],
                dtype=np.int64,
            ),
            "n_samples_used": summary["n_samples_used"],
            "n_features_tested": np.array(
                [
                    0 if r is None else int(r.summary["n_features_tested"][0])
                    for r in tested
                ],
                dtype=np.int64,
            ),
            "df_residual": of_fit("df_residual"),
            "df_prior": of_fit("df_prior"),
            "tests_not_run": text_column(
                [
                    ", ".join(fit.comparison.not_run)
                    if r is not None and fit.comparison is not None
                    else ""
                    for fit, r in zip(fits, tested, strict=True)
                ]
            ),
        }
    )


# Curves.


def _dose_group_estimates(fit: GroupFit, names: Sequence[str]) -> _Estimates:
    """The estimates of a cell type's dose groups against the vehicle, gene by gene,
    from its fit by limma-voom or limma-trend.

    ``names`` are the contrasts of the dose groups, as the fit names them. Their
    unscaled covariance is each gene's own under voom's weights, the one shared by
    every gene without weights or with sample weights, and, with limma's standard
    errors (``contrast_standard_errors="limma"``), each gene's unscaled standard
    errors with the correlation of the design without weights, as limma's
    ``contrasts.fit`` combines them.
    """
    moderated = fit.moderated
    model = moderated.fit
    index = [model.coefficient_names.index(name) for name in names]
    theta = model.coefficients[:, index]
    n_genes = theta.shape[0]
    per_feature = model.cov_coefficients_per_feature
    shared = model.cov_coefficients[np.ix_(index, index)]
    if per_feature is not None:
        covariance = per_feature[:, index][:, :, index]
    elif model.exact_standard_errors:
        covariance = np.broadcast_to(shared, (n_genes, *shared.shape))
    else:
        scale = np.sqrt(np.diag(shared))
        correlation = shared / np.outer(scale, scale)
        unscaled = model.stdev_unscaled[:, index]
        covariance = unscaled[:, :, None] * correlation * unscaled[:, None, :]
    return _Estimates(
        np.asarray(theta, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        np.asarray(moderated.posterior_variance, dtype=np.float64),
        np.asarray(moderated.df_total, dtype=np.float64),
    )


def _curve_doses(
    at: Any, axis: DoseAxis, doses: npt.NDArray[np.float64], group: str
) -> npt.NDArray[np.float64]:
    """The doses of a curve, checked to lie within the doses a cell type compares:
    the vehicle's 0, and the treated doses' range, from 0 when the vehicle sits on
    the axis."""
    values = checked_doses(np.atleast_1d(np.asarray(at)), "at")
    low, high = float(doses[1]), float(doses[-1])
    inside = values <= high * (1.0 + SAME_DOSE)
    if axis.vehicle == "own_group":
        inside &= (values == 0.0) | (values >= low * (1.0 - SAME_DOSE))
    if not inside.all():
        span = (
            f"0, the vehicle, and {low:g} to {high:g} {axis.unit}"
            if axis.vehicle == "own_group"
            else f"0 to {high:g} {axis.unit}"
        )
        raise ValueError(
            f"at holds {dose_text(np.unique(values[~inside]), axis.unit)}, outside the "
            f"doses that {group!r} compares, {span}; curve() does not extrapolate"
        )
    return values


def _gene_rows(genes: Any, tested: list[str], group: str) -> npt.NDArray[np.intp]:
    """The rows of the genes asked for among those tested, in the order given."""
    if genes is None:
        return np.arange(len(tested))
    named = [genes] if isinstance(genes, str) else genes
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            "genes must be a feature id or a list of them, or None for every gene "
            f"tested, got {type(genes).__name__}"
        )
    index = {feature: row for row, feature in enumerate(tested)}
    rows = []
    for gene in list(named):
        if not isinstance(gene, str):
            raise TypeError(
                f"genes must be feature ids (str), got {type(gene).__name__} {gene!r}"
            )
        if gene not in index:
            raise KeyError(
                f"gene {gene!r} was not tested in {group!r}: it is not a feature of "
                "the cube, or did not pass the expression filter there"
            )
        rows.append(index[gene])
    return np.array(rows, dtype=np.intp)


# Printouts.


def _axis_text(axis: DoseAxis, basis: Mapping[str, Any] | None) -> str:
    """The axis and the spline's knots, in doses, for the printout."""
    scale = "log10" if axis.scale == "log10" else "linear"
    vehicle = (
        "the vehicle in a group of its own"
        if axis.vehicle == "own_group"
        else "the vehicle on the axis"
        + (f" at an offset of {axis.offset:g}" if axis.offset is not None else "")
    )
    text = f"{axis.unit} ({axis.metric}) on a {scale} axis, {vehicle}"
    if basis is None:
        return text + "; no spline (one dose on the axis)"
    spline = NaturalSplineBasis(**basis)
    # A knot at the vehicle on the axis comes back as 0 up to rounding.
    knots = np.maximum(axis.doses(np.array(spline.knots, dtype=np.float64)), 0.0)
    bounds = np.maximum(
        axis.doses(np.array(spline.boundary_knots, dtype=np.float64)), 0.0
    )
    interior = (
        f"knots at {listing([f'{k:.3g}' for k in knots])} {axis.unit}, "
        if knots.size
        else ""
    )
    return (
        f"{text}; spline of {spline.df} column{'' if spline.df == 1 else 's'}, "
        f"{interior}boundary knots at {bounds[0]:.3g} and {bounds[1]:.3g} {axis.unit}"
    )


def _slope_unit(axis: DoseAxis) -> str:
    """The unit of the slope, for the printout."""
    return "per log10 dose" if axis.scale == "log10" else f"per {axis.unit}"
