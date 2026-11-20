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
   axis; ``linear_trend``, the slope of a straight line along the axis; and
   ``dose_groups``, whether any dose group differs from the vehicle, the one-way
   analysis of variance of the dose groups with moderated variances: the moderated
   counterpart of the ANOVA gate of NTP's genomic dose-response approach, which at
   least one gene must pass for a data set to be analysed further (NTP 2018). When
   named, three multiple contrast tests refer the largest of several correlated
   moderated t-statistics to the multivariate t distribution: ``dunnett``, each dose
   against the vehicle (Dunnett 1955); ``williams``, the Williams-type test of the
   vehicle against the pooled top doses (Bretz 2006); and ``mcp``, the MCP step, the
   multiple comparison procedure, of MCP-Mod (multiple comparison procedure and
   modelling): the contrasts that best detect each of a few candidate shapes (Bretz,
   Pinheiro and Branson 2005; Pinheiro et al. 2014).
4. On demand, the lowest effective dose of each gene: the lowest dose called by
   stage-wise testing after a screen (Heller et al. 2009; Van den Berge et al. 2017).

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
    are adjusted within the cell type and dose only. A multiple contrast test's
    ``p_value`` is already adjusted for the gene's contrasts (single-step, Hothorn,
    Bretz and Westfall 2008), and its adjusted p-values adjust it across genes.

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

    Dunnett CW (1955). A multiple comparison procedure for comparing several
    treatments with a control. Journal of the American Statistical Association 50,
    1096-1121. doi:10.1080/01621459.1955.10501294

    Bretz F (2006). An extension of the Williams trend test to general unbalanced
    linear models. Computational Statistics and Data Analysis 50, 1735-1748.
    doi:10.1016/j.csda.2005.02.005

    Bretz F, Pinheiro JC, Branson M (2005). Combining multiple comparisons and
    modeling techniques in dose-response studies. Biometrics 61, 738-748.
    doi:10.1111/j.1541-0420.2005.00344.x

    Pinheiro J, Bornkamp B, Glimm E, Bretz F (2014). Model-based dose finding under
    model uncertainty using general parametric models. Statistics in Medicine 33,
    1646-1661. doi:10.1002/sim.6052

    Hothorn T, Bretz F, Westfall P (2008). Simultaneous inference in general
    parametric models. Biometrical Journal 50, 346-363. doi:10.1002/bimj.200810425

    Heller R, Manduchi E, Grant GR, Ewens WJ (2009). A flexible two-stage procedure
    for identifying gene sets that are differentially expressed. Bioinformatics 25,
    1019-1025. doi:10.1093/bioinformatics/btp076

    Van den Berge K, Soneson C, Robinson MD, Clement L (2017). stageR: a general
    stage-wise method for controlling the gene-level false discovery rate in
    differential expression and differential transcript usage. Genome Biology 18,
    151. doi:10.1186/s13059-017-1277-0
"""

import inspect
import json
import numbers
import operator
import zlib
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
from ._doses import SAME_DOSE, checked_doses, dose_group_column, dose_label, dose_text
from ._group_fits import COUNTS_LAYER, FITTED
from ._linear_algebra import column_space
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import positive_integer, random_generator, real_number
from .candidate_shapes import BETA_SCALE_FACTOR, CandidateShapes
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
from .stats.dream import ModeratedFixedEffectTests
from .stats.multiple_contrasts import (
    dunnett_contrasts,
    multiple_contrast_test,
    optimal_contrasts,
    stage_wise_adjusted_p_values,
    williams_contrasts,
)
from .stats.multiple_testing import adjust_p_values
from .stats.smoothing import NaturalSplineBasis
from .table import Table

__all__ = ["DoseResponseResult", "dose_response"]

FloatArray = npt.NDArray[np.float64]

SPLINE, NONLINEARITY, LINEAR_TREND, DOSE_GROUPS = (
    "spline",
    "nonlinearity",
    "linear_trend",
    "dose_groups",
)
TESTS = (SPLINE, NONLINEARITY, LINEAR_TREND, DOSE_GROUPS)
"""The tests, in the order of tables and printouts; ``spline`` is read first."""
DUNNETT, WILLIAMS, MCP = "dunnett", "williams", "mcp"
CONTRAST_TESTS = (DUNNETT, WILLIAMS, MCP)
"""The multiple contrast tests, run only when named, in the order of tables and
printouts."""
_ALL_TESTS = (*TESTS, *CONTRAST_TESTS)
SCREENS = (SPLINE, DOSE_GROUPS, *CONTRAST_TESTS)
"""The tests of any response, which can screen the genes whose doses are called."""
EACH_DOSE = "each_dose"
"""The table of each dose against the vehicle."""
# Of two tests of the same hypothesis in a cell type, the first of this order runs
# when the tests are chosen by default.
_PRIORITY = (SPLINE, DOSE_GROUPS, LINEAR_TREND, NONLINEARITY)
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
# The doses of a curve by default, evenly spaced along the dose axis.
_CURVE_POINTS = 100
# The names of the contrasts of the spline's coefficients, after the column of the
# dose axis they weigh, such as "spline_dose_s1".
_SPLINE_PREFIX = "spline_"
# The vehicle's name in the names of the multiple contrast tests' contrasts.
_VEHICLE = "vehicle"
# A candidate shape whose largest change at a cell type's treated doses is below the
# smallest normal double does not change there, to double precision.
_FLAT = float(np.finfo(np.float64).tiny)


class _Estimates(NamedTuple):
    """The estimates of a cell type's dose groups against the vehicle, gene by gene,
    from the moderated fit of its dose-group model.

    ``theta`` holds each gene's estimates, ``(n_genes, n_doses)``; ``covariance``
    their unscaled covariance, ``(n_genes, n_doses, n_doses)``, which times
    ``posterior_variance`` is their covariance; and ``df_total`` the degrees of
    freedom of the moderated t-tests: one per gene, ``(n_genes,)``, under limma, and
    one per gene and dose, ``(n_genes, n_doses)``, under dream.
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
    - ``"linear_trend"``: the analysis's table of a contrast, whose
      ``log2_fold_change`` is the slope of the straight line, the change in log2
      expression per log10 unit of dose on a log10 axis (of dose plus the offset,
      with the vehicle on the axis) or per unit of dose on a linear one, with
      ``standard_error`` (not for dream), ``average_log2_cpm``, ``t``, ``p_value``,
      ``adjusted_p_value``, ``adjusted_p_value_across_groups`` and ``log_odds``;
      dream adds ``df`` and ``z_statistic``.
    - ``"dunnett"``, ``"williams"`` and ``"mcp"``: multiple contrast tests, each
      gene as a whole, with ``feature_id``, ``symbol``, ``average_log2_cpm``,
      ``contrast`` (the contrast with the largest \\|t\\|, such as ``"10 - vehicle"``,
      ``"3 to 30 - vehicle"`` or a candidate shape's name: for the MCP step, the
      candidate whose contrast best matches the observed changes, not a fitted
      shape, since the shapes' contrasts are strongly correlated and fitting a shape
      is MCP-Mod's modelling step, not implemented yet), its ``t``, ``df`` (of
      the multivariate t), ``p_value`` (adjusted for the gene's contrasts, the
      single-step max-t test) and ``p_value_error`` (its Monte Carlo standard
      error), ``adjusted_p_value`` (Benjamini-Hochberg over the cell type's genes)
      and ``adjusted_p_value_across_groups``. Each contrast's weights on the treated
      doses are in the cell type's entry of ``provenance["groups"]``
      (``contrast_tests``).
    - ``"each_dose"``: each treated dose the cell type compares against the vehicle,
      the rows of a dose together, doses ascending, each dose's genes in the cube's
      feature order: ``feature_id``, ``symbol``, ``dose`` (in the axis's unit), then
      the columns of the analysis's table of a contrast, adjusted within the cell
      type and dose.

    :meth:`lowest_effective_doses` calls each dose against the vehicle in the genes
    that a screen calls, and gives each gene's lowest effective dose.

    ``summary`` has one row per cell type checked: ``group_id``; ``status``, the
    check's (``"inferential"``, ``"descriptive"`` or ``"refused"``); ``tested``;
    ``reason``, why a cell type was not tested, the check's reason or the
    analysis's when its fit stopped, empty when tested; ``n_dose_groups_tested``,
    the vehicle and the treated doses compared; ``n_samples_used``;
    ``n_features_tested``; ``df_residual`` and ``df_prior`` of the fit (medians for
    dream), NaN where not tested; and ``tests_not_run``, the tests of the design
    that a tested cell type did not run, whose reasons :meth:`table` gives.

    Built by :func:`dose_response` from the summary, the tables, the design check
    (``design_report``), the provenance and, as ``fits``, each tested cell type's
    result of the analysis without its tables, from which :meth:`fit` and
    :meth:`curve` compute the cell type's fit again. A pickled result carries the
    cube: a cube read from a cube file refers to its file, and a cube in memory
    carries its arrays. Immutable.

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
        """The status of each cell type checked, whether it was tested and why not,
        and its dose groups, samples, genes and degrees of freedom."""
        return self._summary

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """Every table, by (cell type, test) and (cell type, ``"each_dose"``)."""
        return self._tables

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
        """The tests of the design, those named or, when none were, those it
        supports, in the order spline, nonlinearity, linear_trend and dose_groups,
        then the multiple contrast tests named, in the order dunnett, williams and
        mcp."""
        return tuple(json.loads(self._record)["tests"])

    @property
    def method(self) -> str:
        """The analysis, ``"limma_voom"``, ``"limma_trend"`` or ``"dream_voom"``."""
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
    def candidates(self) -> CandidateShapes | None:
        """The candidate shapes of the MCP step, given or spread over the design's
        doses; None when ``"mcp"`` was not named."""
        record = self.settings["candidates"]
        return None if record is None else CandidateShapes(**record)

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the analysis, as a new dict, also in the provenance.

        They hold the axis (``dose``, as :meth:`DoseAxis.to_dict` gives it), the
        spline's basis (``spline_basis``, as
        :meth:`~cytepool.stats.smoothing.NaturalSplineBasis.to_dict` gives it), the
        tests and ``spline_df`` as given, the candidate shapes of ``"mcp"``
        (``candidates``, as :meth:`CandidateShapes.to_dict` gives them; None without
        ``"mcp"``), the seed of the multiple contrast tests (``seed``; None when none
        ran), the analysis, and ``batch``, ``groups``, ``min_cells`` and
        ``min_counts``. The analysis's other options are in the provenance's
        ``analysis_settings``, as the analysis records them.
        """
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
        doses it compares with their samples, the contrasts of each test it ran
        (``tests``), the tests it did not run with the reasons, the numerator
        degrees of freedom of each test it ran (``f_test_df1``), the weights of
        every contrast of its fit on the model's columns (``contrasts``), the
        weights on the treated doses of each multiple contrast test's contrasts, with
        the dimensions the MCP step's contrasts span and the candidate shapes it left
        out (``contrast_tests``), and the analysis's record of its fit (``fit``: the
        samples used and left out, the features tested, the span of voom's trend and
        the warnings).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None, name: str | None = None) -> Table:
        """Return the table of one cell type and test, or of each dose.

        Args:
            group: A cell type; it may be left out when one cell type was checked.
            name: A test the cell type ran, ``"spline"``, ``"nonlinearity"``,
                ``"linear_trend"``, ``"dose_groups"``, ``"dunnett"``, ``"williams"``
                or ``"mcp"``, or ``"each_dose"`` for each dose against the vehicle.
                The class documentation gives the columns.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, or ``name`` is left out.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked or not tested, or the test was
                not run there; the message says why.
        """
        group = self._group(group)
        name = only_key(name, [*self.tests, EACH_DOSE], "name", "tables")
        found = self._tables.get((group, name))
        if found is not None:
            return found
        entry = self._tested(group)
        if name in entry["tests_not_run"]:
            raise KeyError(
                f"{name!r} was not run in {group!r}: {entry['tests_not_run'][name]}"
            )
        raise KeyError(
            f"no table {name!r}; the tables are: "
            f"{listing([*self.tests, EACH_DOSE])}"
            f"{close_matches(str(name), _ALL_TESTS)}"
        )

    def fit(self, group: str | None = None) -> GroupFit:
        """Compute the fit of a tested cell type again: its values, weights and
        moderated tests.

        The fit of the cell type's model of the dose groups, as
        :meth:`DifferentialExpressionResult.fit` computes it again from the cube and
        settings the result keeps; it takes the time of the cell type's analysis.
        Its moderated tests are those of the contrasts in the cell type's entry of
        ``provenance["groups"]``: each dose against the vehicle, named after its
        indicator such as ``dose_0.1``, the spline's coefficients and the linear
        trend.

        Args:
            group: A cell type that was tested; it may be left out when one cell
                type was checked.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, or the fit no longer gives what the result records (a lazy
                cube whose source changed).
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked or not tested.
        """
        group = self._group(group)
        self._tested(group)
        return self._fits[group].fit(group)

    def curve(
        self,
        group: str | None = None,
        genes: str | Sequence[str] | None = None,
        *,
        at: npt.ArrayLike | None = None,
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
            genes: A feature id or symbol, or a list of them, among the genes tested
                there; None for every gene tested.
            at: Doses in the axis's unit: the vehicle's 0, and doses from the lowest
                to the highest dose the cell type compares, or from 0 when the
                vehicle sits on the axis. The curve is not extrapolated. None gives
                100 doses evenly spaced along the dose axis, from the lowest dose
                the cell type compares, or from the vehicle when it sits on the
                axis, to the highest; a vehicle in a group of its own, where the
                curve is 0, is left out.

        Returns:
            One row per gene and dose, the genes in the order given and each gene's
            doses in the order of ``at``: ``feature_id``, ``symbol`` (when the
            cube's feature table has one), ``dose``, ``log2_fold_change``,
            ``standard_error``, ``ci_lower`` and ``ci_upper``.

        Raises:
            TypeError: If ``group`` is not a string, ``genes`` is not a feature id
                or symbol or a list of them, or ``at`` is not real-valued, or either
                is or holds a numpy masked array.
            ValueError: If ``group`` is left out and several cell types were
                checked; the result is of ``dream_voom``, whose fits do not keep
                the degrees of freedom a band needs; the cell type's doses cannot
                estimate the spline, or no spline is placed; a symbol names several
                genes tested there; or a dose of ``at`` is not finite, is negative,
                or lies outside the doses the cell type compares.
            KeyError: If the cell type was not checked or not tested, or a gene was
                not tested there.
        """
        reject_masked(genes, "genes")
        reject_masked(at, "at")
        group = self._group(group)
        entry = self._tested(group)
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
        at_doses = (
            _curve_grid(axis, doses)
            if at is None
            else _curve_doses(at, axis, doses, group)
        )
        fit = self._fits[group].fit(group)
        names = [dose_group_column(axis.column, dose) for dose in doses[1:]]
        estimates = _dose_group_estimates(fit, names)
        rows = _gene_rows(genes, fit.features, group)
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
        # The genes' columns repeated by position, which keeps them as they were
        # checked.
        genes_of_rows = fit.features.take(np.repeat(rows, at_doses.size))
        return genes_of_rows.select(
            [name for name in ("feature_id", "symbol") if name in genes_of_rows]
        ).with_columns(
            {
                "dose": np.tile(at_doses, rows.size),
                "log2_fold_change": estimate.ravel(),
                "standard_error": error.ravel(),
                "ci_lower": (estimate - margin).ravel(),
                "ci_upper": (estimate + margin).ravel(),
            }
        )

    def lowest_effective_doses(
        self,
        group: str | None = None,
        *,
        screen: str | None = None,
        alpha: float = 0.05,
    ) -> Table:
        """Return the lowest dose from which each gene of a cell type responds, by
        stage-wise calls of each dose against the vehicle.

        The calls of the stage-wise procedure of stageR (Heller et al. 2009; Van den
        Berge et al. 2017): ``screen``, a test of any response of the gene, calls the
        genes whose Benjamini-Hochberg adjusted p-value over the cell type's genes is
        at most ``alpha``, R of the G genes it tested; in each gene called, the
        moderated t-tests of each dose against the vehicle, those of the table
        ``"each_dose"``, are called by Holm's step-down procedure with Shaffer's
        first step at the level R alpha / G, through
        :func:`~cytepool.stats.multiple_contrasts.stage_wise_adjusted_p_values`. The
        lowest effective dose is the lowest dose called. The calls control the
        overall false discovery rate (OFDR) at ``alpha``: the expected share of the
        genes called with at least one false call, the screen's or a dose's, proved
        under independence between genes (Heller et al. 2009) and expected wherever
        Benjamini-Hochberg controls the false discovery rate (Van den Berge et al.
        2017).

        An effective dose here is a dose whose change is detectable at the stage-wise
        level, with no threshold on its size: toxicology's lowest observed effect
        level (LOEL), and the ICH E4 sense of a dose that shows an effect; the
        clinical minimum effective dose usually adds a clinically relevant size. It
        is a tested dose, not a potency between doses, and it moves with the
        replicates, the spacing of the doses and ``alpha``. A gene can respond with
        no dose called. The values hold only at the ``alpha`` given; a call is a
        value at most ``alpha``, as stageR counts
        (https://docs.nygen.io/CytePool/statistics/dose-response#lowest-effective-dose).

        Args:
            group: A cell type that was tested; it may be left out when one cell type
                was checked.
            screen: The test of any response that calls the genes, one the cell type
                ran: None, the default, means ``"spline"``; or ``"dose_groups"``,
                ``"dunnett"``, ``"williams"`` or ``"mcp"``. ``"linear_trend"`` and
                ``"nonlinearity"`` test a straight line or a departure from one, not
                any response.
            alpha: The level of the calls, between 0 and 1.

        Returns:
            A table of one row per gene tested in the cell type, in the cube's
            feature order, with ``feature_id``, ``symbol`` (when the cube's feature
            table has one), ``responds`` (the screen calls the gene),
            ``lowest_effective_dose`` (in the axis's unit; NaN when no dose is
            called), ``log2_fold_change`` and ``stage_wise_adjusted_p_value`` (that
            dose's change from the vehicle and its stage-wise adjusted p-value; NaN
            with the dose), and ``n_doses_called``.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, ``screen`` is not a test of any response, or ``alpha`` is
                not between 0 and 1.
            KeyError: If the cell type was not checked or not tested, or the screen
                was not run there; the message says why and which tests ran.
            TypeError: If ``group`` or ``screen`` is not a string, or ``alpha`` not
                a real number.
        """
        group = self._group(group)
        level = real_number(alpha, "alpha", finite=True)
        if not 0.0 < level < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {level}")
        test = SPLINE if screen is None else screen
        if not isinstance(test, str):
            raise TypeError(f"screen must be a test name, got {type(test).__name__}")
        if test not in SCREENS:
            reason = (
                "tests a straight line or a departure from one, not any response"
                if test in (LINEAR_TREND, NONLINEARITY)
                else "is not a test" + close_matches(test, SCREENS)
            )
            raise ValueError(
                f"screen {test!r} {reason}; the screens are {quoted(SCREENS)}"
            )
        entry = self._tested(group)
        found = self._tables.get((group, test))
        if found is None:
            ran = [name for name in SCREENS if (group, name) in self._tables]
            why = entry["tests_not_run"].get(test, "tests did not name it")
            raise KeyError(
                f"the screen {test!r} was not run in {group!r}: {why}; the tests of any "
                f"response run there are {quoted(ran) or 'none'}: pass one as screen, "
                f"or name {test!r} in dose_response's tests"
            )
        doses = np.array(entry["doses"][1:], dtype=np.float64)
        each = self._tables[(group, EACH_DOSE)]
        genes = len(found)

        def by_dose(column: str) -> FloatArray:
            values = np.asarray(each[column], dtype=np.float64)
            return values.reshape(doses.size, genes).T

        screened = np.asarray(found["p_value"], dtype=np.float64)
        adjusted = stage_wise_adjusted_p_values(
            screened, by_dose("p_value"), alpha=level, method="holm"
        )
        called = adjusted <= level
        lowest = np.argmax(called, axis=1)
        rows = np.arange(genes)
        some = called.any(axis=1)
        columns = [name for name in ("feature_id", "symbol") if name in found]
        return Table(
            {
                **{name: found[name] for name in columns},
                "responds": np.asarray(found["adjusted_p_value"]) <= level,
                "lowest_effective_dose": np.where(some, doses[lowest], np.nan),
                "log2_fold_change": np.where(
                    some, by_dose("log2_fold_change")[rows, lowest], np.nan
                ),
                "stage_wise_adjusted_p_value": np.where(
                    some, adjusted[rows, lowest], np.nan
                ),
                "n_doses_called": called.sum(axis=1).astype(np.int64),
            }
        )

    def _group(self, group: Any) -> str:
        """A cell type that was checked; the only one when ``group`` is left out."""
        groups = list(self.groups)
        name = only_key(group, groups, "group", "cell types")
        if not isinstance(name, str):
            reject_masked(name, "group")
            raise TypeError(
                f"group must be a cell type (str), got {type(name).__name__}"
            )
        if name in groups:
            return name
        if name in (*_ALL_TESTS, EACH_DOSE):
            raise KeyError(
                f"{name!r} is a table, not a cell type; pass name={name!r}, with the "
                "cell type as group, which may be left out when one was checked"
            )
        raise KeyError(
            f"no cell type {name!r} was checked; cell types are: "
            f"{listing(groups) or '(none)'}"
        )

    def _tested(self, group: str) -> dict[str, Any]:
        """The record of a checked cell type, which must have been tested."""
        entry: dict[str, Any] = next(
            e for e in self.provenance["groups"] if e["group_id"] == group
        )
        if not entry["tested"]:
            raise KeyError(
                f"cell type {group!r} was not tested ({entry['status']}): "
                f"{entry['reason']}"
            )
        return entry

    def __repr__(self) -> str:
        record = self.provenance
        settings = record["settings"]
        axis = self.dose
        tested = int(np.count_nonzero(self._summary["tested"]))
        n_groups = len(self._summary)
        names = [f"{t} (read first)" if t == SPLINE else t for t in self.tests]
        dose = "  dose: " + _axis_text(axis, settings["spline_basis"])
        if LINEAR_TREND in self.tests:
            dose += f"; linear_trend slopes {_slope_unit(axis)}"
        lines = [
            f"DoseResponseResult ({record['method']}): {n_groups:,} cell "
            f"type{'' if n_groups == 1 else 's'}, {tested:,} tested; tests: "
            + ", ".join(names),
            dose,
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
            mcp = entry["contrast_tests"].get(MCP)
            notes = [] if mcp is None else _mcp_notes(mcp, len(entry["doses"]) - 1)
            if notes:
                text += "; mcp: " + ", and ".join(notes)
            lines.append(text)
        return fitted_lines("\n".join(lines))

    def _calls(self, group: str, test: str) -> int:
        """The genes whose adjusted p-value of a test is below the printout's FDR."""
        adjusted = self._tables[(group, test)]["adjusted_p_value"]
        return int(np.count_nonzero(adjusted < _FDR))


def _mcp_notes(record: Mapping[str, Any], treated: int) -> list[str]:
    """What a printout says of the MCP step in a cell type: the shapes it left out,
    and more shapes than the dimensions their contrasts span."""
    notes = []
    if record["shapes_left_out"]:
        left_out = listing(list(record["shapes_left_out"]))
        notes.append(f"{left_out} left out, flat at the doses compared")
    if len(record["contrasts"]) > record["dimensions"]:
        notes.append(
            f"{len(record['contrasts'])} candidate shapes on {treated} treated doses, "
            f"whose contrasts span {record['dimensions']} dimensions"
        )
    return notes


def dose_response(
    cube: Cube,
    design: Design,
    *,
    dose: DoseAxis,
    tests: str | Sequence[str] | None = None,
    spline_df: int | None = None,
    candidates: CandidateShapes | None = None,
    analysis: Callable[..., DifferentialExpressionResult] = limma_voom,
    batch: str | None = None,
    groups: str | Sequence[str] | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    rng: Any = None,
    **options: Any,
) -> DoseResponseResult:
    """Test which genes respond to a dose in every cell type of a cube.

    For every cell type (group) of the cube that supports tests: one moderated fit
    of the dose groups, the vehicle and the doses that keep samples there, by
    ``analysis``, and four tests as contrasts of it (the module documentation gives
    the model):

    - ``"spline"``: whether the gene responds at all, by the F-test of a natural
      cubic spline of dose fitted to the means of the dose groups, with a jump from
      the vehicle to the lowest dose (the axis's ``dose_treated`` column, for a dose
      column named ``dose``) when the vehicle is a group of its own. Read it first.
    - ``"nonlinearity"``: whether the spline departs from a straight line along the
      axis, by the F-test of its restricted cubic columns after the first; with the
      vehicle in its own group, the line runs through the treated doses.
    - ``"linear_trend"``: the slope of a straight line fitted to the means along the
      axis, by a moderated t-test, as a change in log2 expression per log10 unit of
      dose on a log10 axis (of dose plus the offset, with the vehicle on the axis)
      or per unit of dose on a linear one. With the vehicle in a group of its own,
      the line runs through the treated doses only, so a gene that jumps at the
      lowest dose and then stays flat has a slope near 0, and the vehicle's
      difference is tested by ``spline`` and ``dose_groups``; with the vehicle on
      the axis, it runs through every dose, the regression on log dose of Tukey's
      trend test (Tukey, Ciminera and Heyse 1985).
    - ``"dose_groups"``: whether any dose group differs from the vehicle, by the
      F-test of every dose against the vehicle, the one-way analysis of variance of
      the dose groups with moderated variances: the moderated counterpart of the
      ANOVA gate of NTP's genomic dose-response approach (NTP 2018).

    Each dose against the vehicle is also tested, in the table ``"each_dose"``.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-tests
    explains the tests and how to read them.

    Three multiple contrast tests run when named, each two-sided, with one error
    rate for the gene's contrasts (the single-step max-t test, from the multivariate
    t distribution on the gene's df) and Benjamini-Hochberg across genes:

    - ``"dunnett"``: each dose against the vehicle (Dunnett 1955).
    - ``"williams"``: the Williams-type test (Bretz 2006), the vehicle against the
      mean of the top one, two or more doses, weighted by the cell type's samples,
      which detects changes that grow in dose order. It is not Williams' (1971)
      trend test of BMDExpress and NTP's rules, which compares each dose with the
      vehicle after fitting means that change in dose order.
    - ``"mcp"``: the MCP step of MCP-Mod, multiple comparison procedure and
      modelling (Bretz, Pinheiro and Branson 2005; Pinheiro et al. 2014): the
      optimal contrasts of the ``candidates`` for the cell type's doses and samples.

    None runs by default: on 20,000 genes of the vehicle and 7 doses in three
    replicates, on two threads, Dunnett's test and the MCP step each added about
    10 s to the 0.9 s of the default tests and the Williams-type test 23 s; with 3
    doses the MCP step took 0.6 ms per gene and the other two under 0.1 ms. Each
    gene's covariance of its doses' estimates, voom's weights included, sets the
    contrasts' correlation, so limma's approximate standard errors
    (``contrast_standard_errors="limma"``) are refused with them. Under dream the
    multivariate t takes the smallest df of the gene's doses, an approximation:
    contrasts that combine doses can have fewer.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-contrasts
    explains them. :meth:`DoseResponseResult.lowest_effective_doses` then gives the
    lowest dose from which each gene responds.

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
            ``"nonlinearity"``, ``"linear_trend"``, ``"dose_groups"``, ``"dunnett"``,
            ``"williams"`` and ``"mcp"``; None runs the first four that the design
            supports, leaving out in each cell type a test of the same hypothesis as
            another, which happens when the dose groups are few. ``"spline"`` and
            ``"linear_trend"`` need two doses on the axis (the treated doses when the
            vehicle is a group of its own, every dose when it is on the axis),
            ``"nonlinearity"`` a spline of two columns or more, and ``"mcp"`` two
            treated doses; a cell type that compares one treated dose does not run
            ``"mcp"``, with the reason.
        spline_df: The number of spline columns, at least 1 and at most the doses on
            the axis minus 1; None gives 2 when the axis holds 4 doses or more, and
            1 otherwise. The knots are placed by :meth:`DoseAxis.spline_basis` with
            its rule ``"levels"``, in the restricted cubic form.
        candidates: The :class:`~cytepool.CandidateShapes` of ``"mcp"``, with a
            ``beta_scale`` of at least the top dose; None spreads the default shapes
            over the design's doses, ``CandidateShapes.spread(dose,
            design.samples)``. Given only with ``"mcp"``. A shape that does not
            change at a cell type's doses is left out there, with the reason in the
            record.
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
        rng: A numpy random generator, an integer seed, or None, for the Monte Carlo
            error of the multiple contrast tests' p-values. dose_response takes one
            seed, the integer given or one drawn from the generator, and records it
            in ``settings["seed"]``; each cell type and test draws from a generator
            seeded by it, the cell type's name and the test, so that
            ``rng=<recorded seed>`` gives the same p-values whatever the other cell
            types, the other tests named and the threads. Tests of two contrasts or
            fewer use no random numbers.
        **options: The analysis's other options, such as ``prior_weights``,
            ``quality_weights``, ``robust``, ``expression_filter``,
            ``contrast_standard_errors``, ``n_threads`` (which the multiple contrast
            tests also follow), ``prior_count`` (``limma_trend``) or ``df_method``
            (``dream_voom``); their values are checked by the analysis.

    Returns:
        The tables of every cell type tested, the summary, the design check, each
        cell type's fit, the settings and a provenance record.

    Warns:
        RuntimeWarning: As the analysis warns, such as when sample quality weights
            cannot be estimated in a cell type.

    Raises:
        TypeError: If ``analysis`` is not one of the three functions, an option is
            one the analysis does not take or one that dose_response sets
            (``contrasts``, ``f_tests``, ``adjust_across_groups``), ``tests``,
            ``spline_df``, ``candidates`` or ``rng`` has the wrong type, or an
            argument is of a type that :func:`~cytepool.check_dose_design` or the
            analysis refuses.
        ValueError: If ``tests`` names no test or a test twice, or names a test
            that the design cannot support; ``spline_df`` is below 1 or above the
            doses on the axis minus 1, or given without two doses on the axis;
            ``candidates`` are given without ``"mcp"`` or with a ``beta_scale``
            below the top dose; a multiple contrast test is named with
            ``contrast_standard_errors="limma"``; ``rng`` is a negative seed; the
            design names
            contrasts, has random intercepts with an analysis of limma or none with
            ``dream_voom``; the sample table already holds a column of the name of a
            dose group's indicator; or as :func:`~cytepool.check_dose_design` and the
            analysis raise.
        KeyError: If a group is not in the cube.
    """
    given = _tests_setting(tests)
    linear = None if given is None else [test for test in given if test in TESTS]
    named = [test for test in CONTRAST_TESTS if given is not None and test in given]
    if spline_df is not None:
        spline_df = positive_integer(spline_df, "spline_df")
    if candidates is not None and not isinstance(candidates, CandidateShapes):
        raise TypeError(
            "candidates must be a CandidateShapes, such as CandidateShapes.spread(axis, "
            f"design.samples) gives, got {type(candidates).__name__}"
        )
    if candidates is not None and MCP not in named:
        raise ValueError(
            "candidates are the shapes of the MCP step, and tests does not name 'mcp'; "
            "add 'mcp' to tests, or leave candidates out"
        )
    # The seed is checked whatever the tests, and recorded when one draws on it.
    seed = _seed(rng)
    method = _analysis_name(analysis, design)
    _check_options(method, options)
    if named and options.get("contrast_standard_errors") == "limma":
        raise ValueError(
            f"the multiple contrast tests, {quoted(named)}, need each gene's covariance "
            "of its doses' estimates, which limma's approximate standard errors "
            "(contrast_standard_errors='limma') do not give, so multiple_contrast_test "
            "refuses them too; leave contrast_standard_errors out, for the exact default"
        )
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
    splines = _splines(dose, design.samples, levels, linear, spline_df)
    shapes = _candidates(candidates, dose, design, levels) if MCP in named else None
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
            report.doses.filter(group_id=group),
            dose,
            splines,
            given is None,
            named,
            shapes,
        )
        fit = _fit_cell_type(
            cube,
            model,
            group,
            comparison,
            analysis,
            min_cells=min_cells,
            min_counts=min_counts,
            options=options,
        )
        if fit.result is not None and comparison.contrast_tests:
            _run_contrast_tests(fit, seed, options.get("n_threads"))
        fits.append(fit)
    _adjust_across_groups(fits, (*splines.tests, *named))
    return _result(
        cube,
        report,
        model,
        splines,
        fits,
        given,
        spline_df,
        method,
        shapes,
        seed if named else None,
        named,
    )


# The checks of the arguments.


def _tests_setting(value: Any) -> list[str] | None:
    """The tests named, in the order given, or None for every test the design
    supports."""
    reject_masked(value, "tests")
    if value is None:
        return None
    named = [value] if isinstance(value, str) else value
    every = _ALL_TESTS
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            f"tests must be a test name or a list of them, among {quoted(every)}, or "
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
        if name not in every:
            raise ValueError(
                f"tests names {name!r}, which is not a test"
                f"{close_matches(name, every)}; the tests are {quoted(every)}"
            )
    repeated = sorted({name for name in listed if listed.count(name) > 1})
    if repeated:
        raise ValueError(f"tests names {quoted(repeated)} more than once")
    return listed


def _seed(rng: Any) -> int:
    """The seed of the multiple contrast tests: an integer seed as given, or one drawn
    from a generator, or from fresh entropy for None, so that the record reproduces
    the p-values."""
    generator = random_generator(rng)
    if isinstance(rng, numbers.Integral) and not isinstance(rng, bool | np.bool_):
        return operator.index(rng)
    return int(generator.integers(0, 2**63))


def _candidates(
    candidates: CandidateShapes | None,
    axis: DoseAxis,
    design: Design,
    levels: npt.NDArray[np.float64],
) -> CandidateShapes:
    """The candidate shapes of the MCP step: those given, or spread over the doses of
    the design's samples; the design needs two treated doses, and beta shapes a scale
    of at least the top dose."""
    treated = levels[levels > 0.0]
    if treated.size < 2:
        raise ValueError(
            "the MCP step compares candidate shapes along the treated doses, and the "
            f"design has {treated.size}: {dose_text(treated, axis.unit)}; leave 'mcp' "
            "out of tests, or test each dose with 'dunnett'"
        )
    if candidates is None:
        return CandidateShapes.spread(axis, design.samples)
    top = float(levels[-1])
    if candidates.beta_scale is not None and candidates.beta_scale < top:
        raise ValueError(
            f"the candidates' beta_scale, {candidates.beta_scale:g} {axis.unit}, is "
            f"below the design's top dose, {top:g} {axis.unit}, where the beta shapes "
            "are not defined; give a beta_scale of at least the top dose, such as "
            f"{BETA_SCALE_FACTOR:g} times it ({BETA_SCALE_FACTOR * top:g})"
        )
    return candidates


def _analysis_name(analysis: Any, design: Any) -> str:
    """The name of the analysis, which must be one of the three and suit the
    design's random effects."""
    method = next(
        (name for name, known in _ANALYSES.items() if analysis is known), None
    )
    if method is None:
        given = (
            f"the name {analysis!r}" if isinstance(analysis, str) else repr(analysis)
        )
        raise TypeError(
            "analysis must be one of the functions limma_voom, limma_trend and "
            f"dream_voom, such as analysis=cytepool.limma_trend, got {given}"
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
    own_group = axis.vehicle == "own_group"
    m = np.count_nonzero(levels > 0.0) if own_group else levels.size
    if m < 2 and spline_df is None:
        along = [test for test in (given or []) if test != DOSE_GROUPS]
        if along:
            one = len(along) == 1
            where, fix = "", "a dose series needs the vehicle and a treated dose"
            if levels.size >= 2:  # the vehicle in a group of its own, and one dose
                where = " (the treated doses: the vehicle is a group of its own)"
                fix = (
                    "test dose_groups, or place the vehicle on the axis with "
                    "DoseAxis(..., vehicle='on_axis')"
                )
            raise ValueError(
                f"{quoted(along)} {'runs' if one else 'run'} along the dose axis and "
                f"{'needs' if one else 'need'} two doses on it, and the design has "
                f"{'one' if m else 'none'}{where}; {fix}"
            )
        wanted = given is None or DOSE_GROUPS in given
        return _Splines(None, None, (DOSE_GROUPS,) if wanted else ())
    df = spline_df if spline_df is not None else (2 if m >= 4 else 1)
    try:
        basis = axis.spline_basis(samples, df=df)
    except ValueError as error:
        error.add_note(
            "(df is dose_response's spline_df; left out, it is chosen from the doses "
            "on the axis)"
        )
        raise
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
    slope_basis = axis.spline_basis(samples, df=1)
    supported = [test for test in TESTS if test != NONLINEARITY or df >= 2]
    chosen = supported if given is None else [t for t in TESTS if t in given]
    return _Splines(basis, slope_basis, tuple(chosen))


# The model of the dose groups.


def _dose_group_model(
    design: Design, axis: DoseAxis, levels: npt.NDArray[np.float64]
) -> Design:
    """The design's terms, an intercept, and an indicator of each treated dose."""
    samples = design.samples
    doses = np.asarray(samples[axis.column], dtype=np.float64)
    treated = levels[levels > 0.0]
    indicators = [dose_group_column(axis.column, level) for level in treated]
    taken = [name for name in indicators if name in samples]
    if taken:
        raise ValueError(
            f"the sample table already has the column(s) {quoted(taken)}, the names "
            "dose_response gives the indicators of the dose groups; rename them"
        )
    columns = {
        name: (doses == level).astype(np.float64)
        for level, name in zip(treated, indicators, strict=True)
    }
    # "+ 1" keeps the vehicle's mean as the intercept, even under "~ 0".
    formula = (
        f"{design.formula} + 1 + {' + '.join(quote_name(name) for name in indicators)}"
    )
    random = [str(effect) for effect in design.random] or None
    return Design(
        samples.with_columns(columns),
        formula,
        random=random,
        levels={name: list(values) for name, values in design.levels.items()},
    )


# One cell type: what it compares, its contrasts, and its fit.


@dataclass(frozen=True)
class _Comparison:
    """The dose groups a cell type compares, and its tests as contrasts of them."""

    doses: npt.NDArray[np.float64]  # the vehicle first, then the treated doses
    samples: npt.NDArray[np.float64]  # the samples of each
    indicators: list[str]  # the names of the treated doses' indicator columns
    # Each contrast's weights on the treated doses, by name: each dose against the
    # vehicle, named after its indicator; the spline's coefficients; the linear
    # trend.
    contrasts: dict[str, npt.NDArray[np.float64]]
    tests: dict[str, list[str]]  # each test run, in the order of TESTS: its contrasts
    not_run: dict[str, str]  # each test of the design not run here: why
    # Each multiple contrast test run here, in the order of CONTRAST_TESTS.
    contrast_tests: dict[str, "_ContrastSet"]


@dataclass(frozen=True)
class _ContrastSet:
    """The contrasts of a multiple contrast test in one cell type: their names, their
    weights on the treated doses' differences from the vehicle, ``(n_doses,
    n_contrasts)``, and what the record keeps besides the weights: for the MCP step,
    the dimensions its contrasts span and the shapes it left out."""

    names: tuple[str, ...]
    weights: npt.NDArray[np.float64]
    extras: dict[str, Any]


def _comparison(
    rows: Table,
    axis: DoseAxis,
    splines: _Splines,
    by_default: bool,
    named: Sequence[str],
    shapes: CandidateShapes | None,
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
    if LINEAR_TREND in wanted:
        assert splines.slope_basis is not None  # the design supports the trend
        line = _spline_weights(axis, splines.slope_basis, levels, samples)
        if line is None:
            not_run[LINEAR_TREND] = (
                "with the vehicle in a group of its own, the linear trend runs along "
                "the treated doses, and this cell type compares one, "
                f"{dose_text(levels[1:], axis.unit)}"
            )
        else:
            contrasts[LINEAR_TREND] = line[-1, 1:]
            tests[LINEAR_TREND] = [LINEAR_TREND]
    if DOSE_GROUPS in wanted:
        tests[DOSE_GROUPS] = names
    if by_default:
        _leave_out_repeats(tests, contrasts, not_run)
    sets: dict[str, _ContrastSet] = {}
    for test in named:
        found = _contrast_set(test, levels, samples, shapes)
        if isinstance(found, str):
            not_run[test] = found
        else:
            sets[test] = found
    ordered = {test: tests[test] for test in TESTS if test in tests}
    reasons = {test: not_run[test] for test in _ALL_TESTS if test in not_run}
    return _Comparison(levels, samples, names, contrasts, ordered, reasons, sets)


def _contrast_set(
    test: str,
    doses: npt.NDArray[np.float64],
    samples: npt.NDArray[np.float64],
    shapes: CandidateShapes | None,
) -> "_ContrastSet | str":
    """The contrasts of a multiple contrast test on a cell type's treated doses, from
    the doses it compares and their samples, or why the test does not run there.

    Dunnett's contrasts are the doses themselves; Williams-type contrasts weigh the
    top doses by their samples (Bretz 2006).
    """
    groups = [_VEHICLE, *(dose_label(dose) for dose in doses[1:])]
    if test == MCP:
        assert shapes is not None  # dose_response places them when mcp is named
        return _mcp_contrast_set(doses, samples, shapes, groups)
    contrasts = (
        dunnett_contrasts(groups)
        if test == DUNNETT
        else williams_contrasts(
            dict(zip(groups, (int(n) for n in samples), strict=True))
        )
    )
    return _ContrastSet(contrasts.names, np.ascontiguousarray(contrasts.matrix[1:]), {})


def _mcp_contrast_set(
    doses: npt.NDArray[np.float64],
    samples: npt.NDArray[np.float64],
    shapes: CandidateShapes,
    groups: list[str],
) -> "_ContrastSet | str":
    """The MCP step's contrasts on a cell type's treated doses, or why it does not run
    there: the optimal contrasts of the candidate shapes that change at its doses,
    for groups of its samples, c proportional to n_k (mu_k - the size-weighted mean
    of mu), with the vehicle's weight dropped and each scaled to unit length, as
    DoseFinding's placebo-adjusted contrasts (Pinheiro et al. 2014)."""
    if doses.size < 3:
        return (
            "the MCP step compares candidate shapes along the treated doses, and "
            f"this cell type compares one, {groups[1]}"
        )
    means = shapes.means(doses)
    flat = np.abs(means[:, 1:]).max(axis=1) < _FLAT
    if flat.all():
        return (
            "no candidate shape changes at the doses this cell type compares, "
            f"{listing(groups[1:])}"
        )
    contrasts = optimal_contrasts(
        {
            name: values
            for name, values, gone in zip(shapes.names, means, flat, strict=True)
            if not gone
        },
        np.diag(1.0 / samples),
        groups=groups,
    )
    weights = contrasts.matrix[1:] / np.linalg.norm(contrasts.matrix[1:], axis=0)
    left_out = {
        name: "it does not change at the doses this cell type compares"
        for name, gone in zip(shapes.names, flat, strict=True)
        if gone
    }
    return _ContrastSet(
        contrasts.names,
        weights,
        {"dimensions": len(column_space(weights).kept), "shapes_left_out": left_out},
    )


def _weights_record(
    names: Sequence[str], weights: npt.NDArray[np.float64]
) -> dict[str, list[float]]:
    """Each contrast's weights on the treated doses, by name, for the record."""
    return {name: weights[:, j].tolist() for j, name in enumerate(names)}


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
    model: Design,
    group: str,
    comparison: _Comparison,
    analysis: Callable[..., DifferentialExpressionResult],
    *,
    min_cells: Any,
    min_counts: Any,
    options: Mapping[str, Any],
) -> _CellTypeFit:
    """Fit a cell type's dose groups and run its tests, as contrasts of the fit.

    The check and the fit choose a cell type's samples alike, so the fit compares
    the dose groups the check counted; a contrast the fit could not estimate would
    make ``result.table`` raise, with the reason.
    """
    contrasts, f_tests = _contrasts(model, comparison)
    result = analysis(
        cube,
        model,
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
    tables: dict[str, Table] = {}
    for test in comparison.tests:
        table = result.table(group, test)
        if test != LINEAR_TREND:
            table = table.select(
                [c for c in table.columns if not c.startswith("log2_fold_change_")]
            )
        tables[test] = table
    tables[EACH_DOSE] = _each_dose_table(result, group, comparison)
    return _CellTypeFit(group, INFERENTIAL, "", comparison, result, tables)


def _run_contrast_tests(fit: _CellTypeFit, seed: int, n_threads: Any) -> None:
    """Run the multiple contrast tests of a fitted cell type and add their tables.

    Each gene's dose-group estimates and their moderated covariance come from the
    cell type's fit, computed again; the multivariate t takes one df per gene, the
    smallest of its doses' under dream. Each test draws its random numbers from
    ``numpy.random.default_rng([seed, crc32 of the cell type's name, its position in
    CONTRAST_TESTS])``, so that its p-values depend neither on the other cell types
    nor on the order of the run.
    """
    comparison, result = fit.comparison, fit.result
    assert comparison is not None and result is not None
    group_fit = result.fit(fit.group)
    estimates = _dose_group_estimates(group_fit, comparison.indicators)
    df = estimates.df_total
    if df.ndim == 2:
        df = df.min(axis=1)
    covariances = estimates.covariance * estimates.posterior_variance[:, None, None]
    genes = fit.tables[EACH_DOSE].take(np.arange(estimates.theta.shape[0]))
    columns = [name for name in ("feature_id", "symbol") if name in genes]
    for position, test in enumerate(CONTRAST_TESTS):
        found = comparison.contrast_tests.get(test)
        if found is None:
            continue
        generator = np.random.default_rng(
            [seed, zlib.crc32(fit.group.encode("utf-8")), position]
        )
        outcome = multiple_contrast_test(
            estimates.theta,
            covariances,
            dict(zip(found.names, found.weights.T, strict=True)),
            df=df,
            alternative="two-sided",
            rng=generator,
            n_threads=n_threads,
        )
        table = outcome.table()
        fit.tables[test] = Table(
            {
                **{name: genes[name] for name in columns},
                "average_log2_cpm": genes["average_log2_cpm"],
                "contrast": table["contrast"],
                "t": table["t"],
                "df": outcome.df,
                "p_value": table["p_value"],
                "p_value_error": table["p_value_error"],
                "adjusted_p_value": table["adjusted_p_value"],
            }
        )


def _contrasts(
    model: Design, comparison: _Comparison
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, list[str]]]:
    """A cell type's contrasts on the columns of the model of the dose groups, by
    name, and its F-tests, the tests of more than one contrast."""
    columns = list(model.columns)
    # The design names a column as its formula does, quoting a name such as
    # "dose_1e-09" in backticks.
    positions = [columns.index(quote_name(name)) for name in comparison.indicators]
    contrasts = {}
    for name, weights in comparison.contrasts.items():
        vector = np.zeros(len(columns))
        vector[positions] = weights
        contrasts[name] = vector
    f_tests = {
        test: members
        for test, members in comparison.tests.items()
        if test != LINEAR_TREND
    }
    return contrasts, f_tests


def _each_dose_table(
    result: DifferentialExpressionResult, group: str, comparison: _Comparison
) -> Table:
    """The table of each treated dose against the vehicle: the analysis's tables of
    its indicators, stacked in dose order, with each row's dose after the gene."""
    doses = comparison.doses[1:]
    tables = [result.table(group, name) for name in comparison.indicators]
    first = tables[0]
    genes = [name for name in ("feature_id", "symbol") if name in first]
    # The genes' columns repeated by position, which keeps them as they were checked;
    # the statistics stacked, and each row's dose.
    stacked = first.take(np.tile(np.arange(len(first)), len(tables))).with_columns(
        {
            **{
                name: np.concatenate([table[name] for table in tables])
                for name in first.columns
                if name not in genes
            },
            "dose": np.repeat(doses, len(first)),
        }
    )
    order = [*genes, "dose", *(name for name in first.columns if name not in genes)]
    return stacked.select(order)


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
            names = list(table.columns)
            at = names.index("adjusted_p_value") + 1
            fit.tables[test] = table.with_columns(
                {_ACROSS_GROUPS: adjusted[start:stop]}
            ).select([*names[:at], _ACROSS_GROUPS, *names[at:]])
            start = stop


# The result.


def _result(
    cube: Cube,
    report: DoseDesignReport,
    model: Design | None,
    splines: _Splines,
    fits: list[_CellTypeFit],
    given: list[str] | None,
    spline_df: int | None,
    method: str,
    shapes: CandidateShapes | None,
    seed: int | None,
    named: Sequence[str],
) -> DoseResponseResult:
    check = report.settings
    first = next((fit.result for fit in fits if fit.result is not None), None)
    design = None
    if model is not None:
        columns = model.columns
        record = (mixed_design_record if method == "dream_voom" else design_record)(
            model, Contrasts(np.empty((len(columns), 0)), (), columns)
        )
        design = {key: value for key, value in record.items() if key != "contrasts"}
    settings = {
        "dose": check["dose"],
        "spline_basis": None if splines.basis is None else splines.basis.to_dict(),
        "tests": given,
        "spline_df": spline_df,
        "candidates": None if shapes is None else shapes.to_dict(),
        "seed": seed,
        "analysis": method,
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
        "tests": [*splines.tests, *named],
        "groups": [_group_record(fit) for fit in fits],
    }
    tables = {
        (fit.group, name): table for fit in fits for name, table in fit.tables.items()
    }
    kept = {
        fit.group: _without_tables(fit.result) for fit in fits if fit.result is not None
    }
    return DoseResponseResult(_summary(fits, report), tables, kept, report, provenance)


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
        "tests": {},
        "tests_not_run": {},
        "f_test_df1": {},
        "contrasts": {},
        "contrast_tests": {},
        "fit": None,
    }
    if result is not None:
        assert comparison is not None  # a tested cell type compares dose groups
        entry["tests"] = {test: list(names) for test, names in comparison.tests.items()}
        entry["tests_not_run"] = dict(comparison.not_run)
        record = result.provenance
        (group,) = record["groups"]
        trend = {LINEAR_TREND: 1} if LINEAR_TREND in fit.tables else {}
        df1 = {**group["f_test_df1"], **trend}
        entry["f_test_df1"] = {test: df1[test] for test in TESTS if test in df1}
        entry["contrasts"] = record["design"]["contrasts"]
        entry["contrast_tests"] = {
            test: {
                "contrasts": _weights_record(found.names, found.weights),
                **found.extras,
            }
            for test, found in comparison.contrast_tests.items()
        }
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
    from its fit.

    ``names`` are the contrasts of the dose groups, as the fit names them. Their
    unscaled covariance is each gene's own under voom's weights, the one shared by
    every gene without weights or with sample weights, and, with limma's standard
    errors (``contrast_standard_errors="limma"``), each gene's unscaled standard
    errors with the correlation of the design without weights, as limma's
    ``contrasts.fit`` combines them. Under dream, each gene's covariance of the
    contrasts over the variance it moderates (dream's sigma^2), so that times the
    posterior variance it is the covariance of the moderated t-tests.
    """
    moderated = fit.moderated
    if isinstance(moderated, ModeratedFixedEffectTests):
        tests = moderated.tests
        columns = [tests.contrast_names.index(name) for name in names]
        scale = moderated.residual_variance_at_mean_weight[:, None, None]
        return _Estimates(
            np.asarray(tests.estimate[:, columns], dtype=np.float64),
            np.asarray(tests.covariance[:, columns][:, :, columns] / scale),
            np.asarray(moderated.posterior_variance, dtype=np.float64),
            np.asarray(moderated.df_total[:, columns], dtype=np.float64),
        )
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


def _curve_grid(axis: DoseAxis, doses: npt.NDArray[np.float64]) -> FloatArray:
    """The doses of a curve by default: evenly spaced along the axis over the doses a
    cell type compares, from the lowest treated dose, or from the vehicle when it
    sits on the axis, to the highest."""
    low = float(doses[1]) if axis.vehicle == "own_group" else 0.0
    high = float(doses[-1])
    ends = axis.positions(np.array([low, high]))
    grid = axis.doses(np.linspace(ends[0], ends[1], _CURVE_POINTS))
    # The ends exactly, which reading positions back as doses can round.
    grid[0], grid[-1] = low, high
    return grid


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


def _gene_rows(genes: Any, features: Table, group: str) -> npt.NDArray[np.intp]:
    """The rows of the genes asked for among those tested, by feature id or, when the
    features have one, symbol, in the order given."""
    ids = features["feature_id"].tolist()
    if genes is None:
        return np.arange(len(ids))
    named = [genes] if isinstance(genes, str) else genes
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            "genes must be a feature id or symbol, or a list of them, or None for "
            f"every gene tested, got {type(genes).__name__}"
        )
    by_id = {feature: row for row, feature in enumerate(ids)}
    by_symbol: dict[Any, list[int]] = {}
    if "symbol" in features:
        for row, symbol in enumerate(features["symbol"].tolist()):
            by_symbol.setdefault(symbol, []).append(row)
    rows = []
    for gene in list(named):
        if not isinstance(gene, str):
            raise TypeError(
                "genes must be feature ids or symbols (str), got "
                f"{type(gene).__name__} {gene!r}"
            )
        found = [by_id[gene]] if gene in by_id else by_symbol.get(gene, [])
        if len(found) > 1:
            raise ValueError(
                f"the symbol {gene!r} names {len(found)} genes tested in {group!r}, "
                f"{quoted([ids[row] for row in found])}; give their feature ids"
            )
        if not found:
            kind = "feature id or symbol" if "symbol" in features else "feature id"
            raise KeyError(
                f"gene {gene!r} was not tested in {group!r}: no gene tested there has "
                f"this {kind}; it is not in the cube, or did not pass the expression "
                "filter there"
            )
        rows.append(found[0])
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
    """The unit of the linear trend's slope, for the printout: per unit of the axis's
    position."""
    if axis.scale == "linear":
        return f"per {axis.unit}"
    if axis.offset is None:
        return "per log10 unit of dose"
    return f"per log10 unit of dose plus {axis.offset:g} {axis.unit}"
