"""Which genes respond to a dose in each cell type, by moderated tests of one model of
the dose groups.

:func:`~cytepool.dose_response` tests a dose series in every cell type (group) of a
cube that supports tests, in one call, or the scores of one cell type's samples
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
4. The gate, which decides which genes count as responding: one test of any
   response at a stated false discovery rate, after the dose-blind expression
   filter (Bourgon, Gentleman and Huber 2010), with any fold change in the null
   hypothesis of each dose against the vehicle by TREAT (McCarthy and Smyth 2009).
5. On demand, the lowest effective dose of each gene the gate passes: the lowest
   dose called by stage-wise testing (Heller et al. 2009; Van den Berge et al.
   2017).

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

Curves of named models:
    With ``models``, each cell type's genes also get dose-response curves of a set of
    named models (:mod:`cytepool.stats.dose_curves`): under ``variance="constant"``,
    each gene's values in the cell type's fit, refitted against dose as BMDS refits a
    gene's samples; under ``"weighted"`` or ``"moderated"``, the dose-group means of
    the fit by generalised least squares with their covariance, which carries voom's
    weights and the design's other terms, and the residual or moderated variance
    (Pinheiro et al. 2014). Cell types that support only description get curves under
    constant variance, without lack-of-fit p-values.

Scores of samples:
    Pathway scores (:class:`~cytepool.PathwayScores`) and activities of samples
    (:class:`~cytepool.Activities`) of one cell type are tested on the same model and
    contrasts, fitted as :func:`~cytepool.limma` fits scores: limma's linear model
    without observation weights, a score with missing values on the samples it has,
    and empirical Bayes moderation across the scores (Smyth 2004), as GSVA's guide
    tests its scores (Haenzelmann, Castelo and Guinney 2013). limma does not moderate
    one or two scores, so a signature tested alone is an ordinary linear model
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-signatures).

Tests in one direction:
    With ``direction``, each test with a sign is one-sided: the linear trend's
    p-value is P(T >= t) for a rise and P(T <= t) for a fall, p / 2 or 1 - p / 2 of
    its two-sided p, on any distribution of t symmetric about 0; the multiple
    contrast tests refer the largest of their t-statistics in the stated direction
    to the multivariate t distribution (Hothorn, Bretz and Westfall 2008). The
    contrasts rise with dose, as DoseFinding's increasing models do, and a fall is
    their mirror image. The gate's test is then the Williams-type test, which
    compares the top doses with the vehicle. The F-tests have no direction, and
    each dose against the vehicle stays two-sided
    (https://docs.nygen.io/CytePool/statistics/dose-response#one-sided-dose-tests).

Benchmark doses:
    With ``bmr`` as well, each gene's best curve gives its benchmark dose at each
    benchmark response, and at 1 SD beside them, with one-sided 95% profile-likelihood
    bounds given the chosen model in cell types that support tests and without bounds
    in those that support only description (:mod:`cytepool.stats.potency`).

Adjustment for multiple testing:
    ``adjusted_p_value`` adjusts the p-values of one cell type and test by
    Benjamini-Hochberg (Benjamini and Hochberg 1995), as limma's ``topTable`` does,
    and ``adjusted_p_value_across_groups`` adjusts them over every cell type that ran
    the test, stacked into one family. The tests of each dose against the vehicle
    are adjusted within the cell type and dose only. A multiple contrast test's
    ``p_value`` is already adjusted for the gene's contrasts (single-step, Hothorn,
    Bretz and Westfall 2008), and its adjusted p-values adjust it across genes. The
    gate adjusts its p-values by Benjamini-Hochberg within each cell type; a gate
    with a fold change first takes, for each gene, the smallest of its doses' TREAT
    p-values times the doses (Bonferroni), so that a gene counts once.

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

    Haenzelmann S, Castelo R, Guinney J (2013). GSVA: gene set variation analysis for
    microarray and RNA-seq data. BMC Bioinformatics 14, 7.
    doi:10.1186/1471-2105-14-7

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

    Bourgon R, Gentleman R, Huber W (2010). Independent filtering increases detection
    power for high-throughput experiments. Proceedings of the National Academy of
    Sciences 107, 9546-9551. doi:10.1073/pnas.0914005107

    McCarthy DJ, Smyth GK (2009). Testing significance relative to a fold-change
    threshold is a TREAT. Bioinformatics 25, 765-771. doi:10.1093/bioinformatics/btp053
"""

import inspect
import json
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, NamedTuple, cast

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
from ._counts import effective_library_sizes
from ._doses import SAME_DOSE, checked_doses, dose_group_column, dose_label, dose_text
from ._expression_values import MIN_CELLS
from ._group_fits import (
    COUNTS_LAYER,
    FITTED,
    check_sample_groups,
    expressed_features,
    filter_setting,
    group_positions,
    threads_setting,
)
from ._linear_algebra import column_space
from ._masked_arrays import read_only, reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import checked_choice, positive_integer, real_number, recorded_seed
from ._sample_scores import SampleScores, contrast_table, fit_scores, sample_scores
from .activity import Activities
from .benchmark_responses import BenchmarkResponse
from .candidate_shapes import BETA_SCALE_FACTOR, CandidateShapes
from .cube import Cube
from .design import Contrasts, Design
from .design.checks import _aliasing
from .design.formula import quote_name
from .differential_expression import (
    DifferentialExpressionResult,
    GroupFit,
    _limma_f_test,
    dream_voom,
    limma_trend,
    limma_voom,
)
from .dose_axes import DoseAxis
from .dose_designs import DESCRIPTIVE, INFERENTIAL, DoseDesignReport, check_dose_design
from .normalization import library_sizes, log2_cpm, tmm_factors
from .pathway_scores import PathwayScores
from .stats._benchmark_dose_flags import checked_flag_sets
from .stats._benchmark_dose_search import search_settings
from .stats._dose_curve_models import curve_values, model_of
from .stats._dose_curve_search import fitter_settings
from .stats.distributions import t_ppf
from .stats._mean_variance_trend import voom_log_cpm
from .stats.dose_curves import (
    CONSTANT,
    VARIANCES,
    DoseCurveFits,
    _best_counts,
    _checked_model_set,
    _constant_variance_data,
    _CurveData,
    _fit_means,
    _no_model_fits,
    _turn_text,
    _unchecked_turns,
)
from .stats.dream import ModeratedFixedEffectTests
from .stats.empirical_bayes import ModeratedFit, treat
from .stats.multiple_contrasts import (
    dunnett_contrasts,
    multiple_contrast_test,
    optimal_contrasts,
    stage_wise_adjusted_p_values,
    williams_contrasts,
)
from .stats.multiple_testing import adjust_p_values
from .stats.potency import (
    BenchmarkDoses,
    _benchmark_record,
    _checked_responses,
    _status_counts,
    benchmark_doses,
)
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
EACH_DOSE = "each_dose"
"""The table of each dose against the vehicle, and the test of a gate with a fold
change."""
GATE = "gate"
"""The table of the gate's calls."""
GATE_TESTS = (SPLINE, DOSE_GROUPS, *CONTRAST_TESTS)
"""The tests of any response that a gate can take without a fold change."""
UP, DOWN = "up", "down"
DIRECTIONS = (UP, DOWN)
"""The directions of one-sided tests: a rise or a fall with dose, DoseFinding's
``"increasing"`` and ``"decreasing"``."""
SIGNED_TESTS = (LINEAR_TREND, *CONTRAST_TESTS)
"""The tests of a change with a sign, which ``direction`` makes one-sided."""
# The alternative of the multiple contrast tests in each direction: the contrasts rise
# with dose, so a fall is a contrast below 0.
_ALTERNATIVES = {None: "two-sided", UP: "greater", DOWN: "less"}
# The options of the fit of scores, as limma takes them, and the threads of the
# multiple contrast tests and curves.
_SCORE_OPTIONS = ("trend", "robust", "n_threads")
# The analysis of scores, whose name the result records.
_LIMMA = "limma"
CURVES = "curves"
"""The table of the curves of named models, built from the fits on demand."""
BENCHMARK_DOSES = "benchmark_doses"
"""The table of the benchmark doses of the curves, built on demand."""
# The options of the analyses that change the values a fit would use, which curves of
# cell types that support only description use too.
_VALUE_OPTIONS = ("expression_filter", "normalization", "prior_count")
_BEST = "best"
# The columns that name the features of a table: a cube's genes, or the gene sets or
# sources of scores.
_ID_COLUMNS = ("feature_id", "symbol", "gene_set", "source")
# What the id column of a table names, in messages and printouts.
_NOUNS = {"feature_id": "gene", "gene_set": "gene set", "source": "source"}
# The numbers of a cell type's fit that the summary reports.
_FIT_NUMBERS = ("n_features_tested", "df_residual", "df_prior")
# Of two tests of the same hypothesis in a cell type, the first of this order runs
# when the tests are chosen by default.
_PRIORITY = (SPLINE, DOSE_GROUPS, LINEAR_TREND, NONLINEARITY)
ANALYSES: dict[str, Callable[..., DifferentialExpressionResult]] = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
    "dream_voom": dream_voom,
}
"""The analyses of dose_response, by the name its results record."""
# The options of the analysis that dose_response sets itself.
_SET_HERE = ("contrasts", "f_tests", "adjust_across_groups")
SET_BY_DOSE_RESPONSE = (*_SET_HERE, "groups", "min_cells", "min_counts")
"""The options of the analysis that dose_response sets, or takes as arguments of its
own, and so does not pass on from its other options."""
_ACROSS_GROUPS = "adjusted_p_value_across_groups"
# The settings of a gate.
_GATE_SETTINGS = ("test", "fdr", "fold_change")
# The level of the calls a printout counts and of the default gate, and of the
# confidence bands.
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
# A score whose residual standard deviation is at most this share of its largest value
# has residuals of rounding alone.
_ROUNDING = 64.0 * float(np.finfo(np.float64).eps)


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
    """The dose tests of every cell type of a cube, or of one cell type's scores, from
    :func:`dose_response`.

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
      ``contrast`` (the contrast of the most extreme t: the largest \\|t\\|, or with a
      direction the largest t for ``"up"`` and the smallest for ``"down"``; such as
      ``"10 - vehicle"``, ``"3 to 30 - vehicle"`` or a candidate shape's name: for
      the MCP step, the candidate whose contrast best matches the observed changes,
      not a fitted shape, since the shapes' contrasts are strongly correlated and
      fitting a shape is MCP-Mod's modelling step, not implemented yet), its ``t``,
      ``df`` (of the multivariate t), ``p_value`` (adjusted for the gene's
      contrasts, the single-step max-t test) and ``p_value_error`` (its Monte Carlo
      standard error), ``adjusted_p_value`` (Benjamini-Hochberg over the cell
      type's genes) and ``adjusted_p_value_across_groups``. Each contrast's weights
      on the treated doses are in the cell type's entry of ``provenance["groups"]``
      (``contrast_tests``).
    - ``"each_dose"``: each treated dose the cell type compares against the vehicle,
      the rows of a dose together, doses ascending, each dose's genes in the cube's
      feature order: ``feature_id``, ``symbol``, ``dose`` (in the axis's unit), then
      the columns of the analysis's table of a contrast, adjusted within the cell
      type and dose.
    - ``"gate"``: the gate's call of each gene (:attr:`gate`), with ``feature_id``,
      ``symbol``, ``p_value`` (the p-value of the gate's test, or with a fold change
      the smallest of the gene's TREAT p-values times the doses, capped at 1),
      ``adjusted_p_value`` (Benjamini-Hochberg over the cell type's genes) and
      ``passes_gate``, whether ``adjusted_p_value`` is at most the gate's FDR.

    - ``"curves"``, with ``models``: one row per gene and model fitted, the columns of
      :meth:`DoseCurveFits.table <cytepool.stats.dose_curves.DoseCurveFits.table>`
      with ``symbol`` after ``feature_id`` and, where the gate ran, ``passes_gate``
      after them, as in the table ``"gate"``, built from :meth:`curve_fits` when
      asked for; also for cell types that support only description, which have no
      gate.
    - ``"benchmark_doses"``, with ``bmr``: one row per BMR and gene with curves, the
      columns of :meth:`BenchmarkDoses.table <cytepool.stats.potency.BenchmarkDoses.table>`
      with ``symbol`` and ``passes_gate`` (as in the table ``"gate"``, in cell types
      where the gate ran) after ``feature_id``, and a column of failed checks per set
      of ``flags``, built from :meth:`benchmark_doses` when asked for; for cell types
      that support only description, without bounds or ``passes_gate``.

    The tables of pathway scores or activities have one row per gene set or source,
    in the scores' order, with the columns :func:`~cytepool.limma` gives scores:
    ``gene_set`` or ``source`` in place of ``feature_id`` and ``symbol``;
    ``estimate``, the change in the scores' units, in place of ``log2_fold_change``;
    and ``average``, the mean score over the samples fitted, in place of
    ``average_log2_cpm``
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-signatures).

    :meth:`lowest_effective_doses` calls each dose against the vehicle in the genes
    that pass the gate, and gives each gene's lowest effective dose.

    ``summary`` has one row per cell type checked: ``group_id``; ``status``, the
    check's (``"inferential"``, ``"descriptive"`` or ``"refused"``); ``tested``;
    ``reason``, why a cell type was not tested, the check's reason or the
    analysis's when its fit stopped, empty when tested; ``n_dose_groups_tested``,
    the vehicle and the treated doses compared; ``n_samples_used``;
    ``n_features_tested``; ``df_residual`` and ``df_prior`` of the fit (medians for
    dream), NaN where not tested; ``tests_not_run``, the tests of the design that a
    tested cell type did not run, whose reasons :meth:`table` gives; and
    ``n_genes_passing_gate``, NaN where the gate did not run, which is not a cell
    type without responding genes.

    Built by :func:`dose_response` from the summary, the tables, the design check
    (``design_report``), the provenance, as ``fits``, each tested cell type's result
    of the analysis without its tables, from which :meth:`fit` and :meth:`curve`
    compute the cell type's fit again, or the moderated fit of scores
    (:class:`~cytepool.stats.empirical_bayes.ModeratedFit`), which :meth:`curve`
    reads; as ``curves``, each cell type's genes fitted (``feature_id``, and
    ``symbol`` when the cube has one) with their
    :class:`~cytepool.stats.dose_curves.DoseCurveFits`; as ``benchmark_doses``, the
    :class:`~cytepool.stats.potency.BenchmarkDoses` of each cell type's curves; and
    with the ``cube`` and ``design`` analysed, from which
    :func:`~cytepool.permutation_check` runs the analysis again; a result of scores,
    or one built without them, cannot be checked. A pickled result carries the cube:
    a cube read from a cube file refers to its file, and a cube in memory carries its
    arrays. Immutable.

    Raises:
        TypeError: If ``summary`` is not a ``Table``, ``tables`` not a mapping of
            (cell type, test) to ``Table``, ``fits`` not a mapping of cell type to
            ``DifferentialExpressionResult`` or ``ModeratedFit``, ``design_report``
            not a ``DoseDesignReport``, ``provenance`` not a mapping that JSON can
            write, ``curves`` not a mapping of cell type to a pair of a ``Table``
            and a ``DoseCurveFits``, ``benchmark_doses`` not a mapping of cell type
            to ``BenchmarkDoses``, ``cube`` not a ``Cube`` or ``design`` not a
            ``Design``.
        ValueError: If ``provenance`` holds a NaN or an infinity, or one of
            ``cube`` and ``design`` is given without the other.
    """

    __slots__ = (
        "_summary",
        "_tables",
        "_fits",
        "_report",
        "_record",
        "_curves",
        "_benchmark_doses",
        "_cube",
        "_design",
    )

    _summary: Table
    _tables: Mapping[tuple[str, str], Table]
    _fits: Mapping[str, DifferentialExpressionResult | ModeratedFit]
    _report: DoseDesignReport
    _record: str
    _curves: Mapping[str, tuple[Table, DoseCurveFits]]
    _benchmark_doses: Mapping[str, BenchmarkDoses]
    _cube: Cube | None
    _design: Design | None

    def __init__(
        self,
        summary: Table,
        tables: Mapping[tuple[str, str], Table],
        fits: Mapping[str, DifferentialExpressionResult | ModeratedFit],
        design_report: DoseDesignReport,
        provenance: Mapping[str, Any],
        *,
        curves: Mapping[str, tuple[Table, DoseCurveFits]] | None = None,
        benchmark_doses: Mapping[str, BenchmarkDoses] | None = None,
        cube: Cube | None = None,
        design: Design | None = None,
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
                f"or to the ModeratedFit of scores, got {type(fits).__name__}"
            )
        for group, fit in fits.items():
            if not isinstance(fit, DifferentialExpressionResult | ModeratedFit):
                raise TypeError(
                    f"fits[{group!r}] must be a DifferentialExpressionResult, or the "
                    f"ModeratedFit of scores, got {type(fit).__name__}"
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
        curves = {} if curves is None else curves
        if not isinstance(curves, Mapping):
            raise TypeError(
                "curves must be a mapping of cell type to a pair of a Table of genes "
                f"and their DoseCurveFits, got {type(curves).__name__}"
            )
        for group, item in curves.items():
            if not (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], Table)
                and isinstance(item[1], DoseCurveFits)
            ):
                raise TypeError(
                    f"curves[{group!r}] must be a pair of a Table of genes and their "
                    f"DoseCurveFits, got {type(item).__name__}"
                )
        benchmark_doses = {} if benchmark_doses is None else benchmark_doses
        if not isinstance(benchmark_doses, Mapping):
            raise TypeError(
                "benchmark_doses must be a mapping of cell type to BenchmarkDoses, got "
                f"{type(benchmark_doses).__name__}"
            )
        for group, found in benchmark_doses.items():
            if not isinstance(found, BenchmarkDoses):
                raise TypeError(
                    f"benchmark_doses[{group!r}] must be a BenchmarkDoses, got "
                    f"{type(found).__name__}"
                )
        if cube is not None and not isinstance(cube, Cube):
            raise TypeError(f"cube must be a Cube or None, got {type(cube).__name__}")
        if design is not None and not isinstance(design, Design):
            raise TypeError(
                f"design must be a Design or None, got {type(design).__name__}"
            )
        if (cube is None) != (design is None):
            raise ValueError(
                "give both the cube and the design that dose_response analysed, or "
                "neither: permutation_check needs both to run the analysis again"
            )
        state = {
            "_summary": summary,
            "_tables": MappingProxyType(dict(tables)),
            "_fits": MappingProxyType(dict(fits)),
            "_report": design_report,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
            "_curves": MappingProxyType(dict(curves)),
            "_benchmark_doses": MappingProxyType(dict(benchmark_doses)),
            "_cube": cube,
            "_design": design,
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DoseResponseResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DoseResponseResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_result,
            (
                self._summary,
                dict(self._tables),
                dict(self._fits),
                self._report,
                self.provenance,
                dict(self._curves),
                dict(self._benchmark_doses),
                self._cube,
                self._design,
            ),
        )

    @property
    def summary(self) -> Table:
        """The status of each cell type checked, whether it was tested and why not,
        its dose groups, samples, genes and degrees of freedom, and the genes that
        pass the gate."""
        return self._summary

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """The tables of the tests, by (cell type, test), (cell type, ``"each_dose"``)
        and (cell type, ``"gate"``); the tables of curves and benchmark doses are built
        by :meth:`table` when asked for."""
        return self._tables

    @property
    def cube(self) -> Cube | None:
        """The cube analysed; None for a result built without it."""
        return self._cube

    @property
    def design(self) -> Design | None:
        """The design given, of the terms other than dose; None for a result built
        without it."""
        return self._design

    @property
    def gate(self) -> dict[str, Any]:
        """The gate's ``test``, ``fdr`` and ``fold_change``, as a new dict that
        ``dose_response(..., gate=result.gate)`` takes back."""
        gate: dict[str, Any] = json.loads(self._record)["settings"]["gate"]
        return gate

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
        mcp; the gate's test is among them, named or not."""
        return tuple(json.loads(self._record)["tests"])

    @property
    def method(self) -> str:
        """The analysis, ``"limma_voom"``, ``"limma_trend"`` or ``"dream_voom"``, or
        ``"limma"`` for scores."""
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
        ran), the gate (``gate``, as :attr:`gate` gives it), the ``direction`` of
        one-sided tests (None for two-sided ones), the curves' ``models`` and
        ``variance``, the BMRs given (``bmr``, as :meth:`BenchmarkResponse.to_dict`
        gives each; None without) and the sets of checks (``flags``), the analysis,
        and ``batch``, ``groups``, ``min_cells`` and ``min_counts`` (None for
        scores). The analysis's other options are in the provenance's
        ``analysis_settings``, as the analysis records them, and for scores
        ``trend`` and ``robust``.
        """
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the analysis (``method``) and the versions of the software; the cube
        (``cube``): its fingerprint, the digests it is built from and its own
        provenance, for scores the record of the cube they were computed from, None
        for scores of an array; the layer read (None for scores); for scores
        (``scores``, None for a cube) their type, method, settings and names; the
        model of the dose groups (``design``): its
        formula, columns, levels and the values of its variables for every sample;
        the settings, and the analysis's settings in full (``analysis_settings``,
        null when no cell type was fitted); the gate's adjustment for multiple testing
        (``gate_adjustment``); the tests of the design; and, for every
        cell type (``groups``), its status, whether it was tested and why not, the
        doses it compares with their samples, the contrasts of each test it ran
        (``tests``), the tests it did not run with the reasons, the numerator
        degrees of freedom of each test it ran (``f_test_df1``), the weights of
        every contrast of its fit on the model's columns (``contrasts``), the
        weights on the treated doses of each multiple contrast test's contrasts, with
        the dimensions the MCP step's contrasts span and the candidate shapes it left
        out (``contrast_tests``), the analysis's record of its fit (``fit``: the
        samples used and left out, the features tested, the span of voom's trend and
        the warnings; for scores the samples used and left out, the columns aliased
        and the scores tested), and, with ``models``, its curves (``curves``: the
        variance mode, the data they were fitted to, the models fitted and not fitted
        with the reasons, why turns could not be checked for a dose group that drives
        them (``turns_unchecked``, empty when they could), the genes, the scores left
        out with their reasons (``left_out``, empty for a cube's genes), or why the
        cell type has none), and, with ``bmr``, its benchmark doses
        (``benchmark_doses``: the BMRs with the 1 SD comparison added, the bounds or
        why there are none, the threshold of each BMR's bounds, the range of doses
        searched and the count of BMDs whose bounds could not be computed; None
        without curves). With ``models``, ``curves`` also holds each model's mean
        and parameters, the sizes of the grids and the refinement's settings, and
        with ``bmr``, ``benchmark_doses`` how BMDs and their bounds are searched.
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None, name: str | None = None) -> Table:
        """Return the table of one cell type and test, or of each dose.

        Args:
            group: A cell type; it may be left out when one cell type was checked.
            name: A test the cell type ran, ``"spline"``, ``"nonlinearity"``,
                ``"linear_trend"``, ``"dose_groups"``, ``"dunnett"``, ``"williams"``
                or ``"mcp"``; ``"each_dose"`` for each dose against the vehicle;
                ``"gate"`` for the gate's calls; ``"curves"`` for the curves of
                named models; or ``"benchmark_doses"`` for their benchmark doses. The
                class documentation gives the columns.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, or ``name`` is left out.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked or not tested, or the test, or
                the gate's test, was not run there, or it has no curves; the message
                says why.
        """
        group = self._group(group)
        name = only_key(name, self._table_names(), "name", "tables")
        if name == CURVES:
            genes, fits = self._curve_pair(group)
            rows = np.repeat(np.arange(len(genes)), len(fits.models))
            return _with_gene_columns(
                fits.table(feature_ids=genes[genes.columns[0]]),
                genes,
                rows,
                self._tables.get((group, GATE)),
            )
        if name == BENCHMARK_DOSES:
            genes, _ = self._curve_pair(group)
            doses = self.benchmark_doses(group)
            table = doses.table(flags=self.settings["flags"])
            rows = np.tile(np.arange(len(genes)), len(doses.benchmark_responses))
            return _with_gene_columns(
                table, genes, rows, self._tables.get((group, GATE))
            )
        found = self._tables.get((group, name))
        if found is not None:
            return found
        entry = self._tested(group)
        if name == GATE:
            raise KeyError(f"the gate was not run in {group!r}: {self._no_gate(entry)}")
        if name in entry["tests_not_run"]:
            raise KeyError(
                f"{name!r} was not run in {group!r}: {entry['tests_not_run'][name]}"
            )
        known = [*_ALL_TESTS, EACH_DOSE, GATE, CURVES, BENCHMARK_DOSES]
        raise KeyError(
            f"no table {name!r}; the tables are: "
            f"{listing(self._table_names())}{close_matches(str(name), known)}"
        )

    def _table_names(self) -> list[str]:
        """The names of the tables: the tests, each dose, the gate and, with models,
        curves, and with bmr, benchmark doses."""
        settings = self.settings
        curves = [CURVES] if settings.get("models") is not None else []
        doses = [BENCHMARK_DOSES] if settings.get("bmr") is not None else []
        return [*self.tests, EACH_DOSE, GATE, *curves, *doses]

    def fit(self, group: str | None = None) -> GroupFit:
        """Compute the fit of a tested cell type again: its values, weights and
        moderated tests.

        The fit of the cell type's model of the dose groups, as
        :meth:`DifferentialExpressionResult.fit` computes it again from the cube and
        settings the result keeps; it takes the time of the cell type's analysis.
        Its moderated tests are those of the contrasts in the cell type's entry of
        ``provenance["groups"]``: each dose against the vehicle, named after its
        indicator such as ``dose_0.1``, the spline's coefficients, the linear trend
        and, for curves of weighted or moderated variance, the vehicle's mean at the
        samples' average of the design's other terms (``vehicle_mean``).

        Args:
            group: A cell type that was tested; it may be left out when one cell
                type was checked.

        A result of scores has no such fit: its tables and :meth:`curve` give the
        tests of the scores, and :func:`~cytepool.limma` tests a contrast of them.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, the fit no longer gives what the result records (a lazy
                cube whose source changed), or the result tests scores.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked or not tested.
        """
        group = self._group(group)
        self._tested(group)
        found = self._fits[group]
        if isinstance(found, ModeratedFit):
            raise ValueError(
                "fit() computes the fit of a cube's cell type again, with voom's "
                "values and weights, and this result tests scores, whose values are "
                "their own: the tables, curve() and limma give their tests"
            )
        return found.fit(group)

    def curve(
        self,
        group: str | None = None,
        genes: str | Sequence[str] | None = None,
        *,
        at: npt.ArrayLike | None = None,
        model: str | None = None,
    ) -> Table:
        """Return the fitted spline of genes against the vehicle, with a 95% band, or
        their curves of a named model.

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
        The fit of a cube's cell type is computed again
        (:meth:`DifferentialExpressionResult.fit`), which takes the time of its
        analysis; a result of scores keeps theirs.

        With ``model``, the curve at dose x is the fitted model minus the mean of the
        vehicle that the curves were fitted to, f(x) - ybar_0, from the parameters of
        :meth:`curve_fits`, without a band; its value at the vehicle shows how far
        the model misses the vehicle's mean
        (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-model-curves).

        Args:
            group: A cell type that was tested, or with ``model`` one with curves; it
                may be left out when one cell type was checked.
            genes: A feature id or symbol, or a list of them, among the genes tested
                there, or fitted with ``model``; for scores, gene sets or sources by
                name; None for every gene.
            at: Doses in the axis's unit: the vehicle's 0, and doses from the lowest
                to the highest dose the cell type compares, or from 0 when the
                vehicle sits on the axis; with ``model``, any dose from 0 to the
                highest, as every model is fitted with the vehicle at dose 0. The
                curve is not extrapolated. None gives 100 doses evenly spaced along
                the dose axis, from the lowest dose the cell type compares, or from
                the vehicle when it sits on the axis, to the highest; a vehicle in a
                group of its own, where the curve is 0, is left out.
            model: None for the spline; a model of the cell type's curves, such as
                ``"hill"``, or ``"best"`` for each gene's best model, where genes
                without one are left out when ``genes`` is None.

        Returns:
            One row per gene and dose, the genes in the order given and each gene's
            doses in the order of ``at``: ``feature_id``, ``symbol`` (when the
            cube's feature table has one), ``dose``, ``log2_fold_change``,
            ``standard_error``, ``ci_lower`` and ``ci_upper``; with ``model``,
            ``feature_id``, ``symbol``, ``model``, ``dose`` and
            ``log2_fold_change``. For scores, ``gene_set`` or ``source`` and
            ``estimate`` in place of the genes' columns and ``log2_fold_change``.

        Raises:
            TypeError: If ``group`` is not a string, ``genes`` is not a feature id
                or symbol or a list of them, or ``at`` is not real-valued, or either
                is or holds a numpy masked array.
            ValueError: If ``group`` is left out and several cell types were
                checked; the result is of ``dream_voom``, whose fits do not keep
                the degrees of freedom a band needs; the cell type's doses cannot
                estimate the spline, or no spline is placed; a symbol names several
                genes tested there; or a dose of ``at`` is not finite, is negative,
                or lies outside the doses the cell type compares; ``model`` is not
                one of the cell type's models or ``"best"``, or a gene named has no
                best model.
            KeyError: If the cell type was not checked or not tested, or has no
                curves with ``model``, or a gene was not tested or fitted there.
        """
        reject_masked(genes, "genes")
        reject_masked(at, "at")
        reject_masked(model, "model")
        if model is not None:
            return self._model_curve(group, genes, at, model)
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
        features, moderated = self._moderated(group)
        names = [dose_group_column(axis.column, dose) for dose in doses[1:]]
        estimates = _dose_group_estimates(moderated, names)
        rows = _gene_rows(genes, features, group)
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
        # The features' columns repeated by position, which keeps them as they were
        # checked.
        genes_of_rows = features.take(np.repeat(rows, at_doses.size))
        return genes_of_rows.select(
            [name for name in _ID_COLUMNS if name in genes_of_rows]
        ).with_columns(
            {
                "dose": np.tile(at_doses, rows.size),
                self._change(): estimate.ravel(),
                "standard_error": error.ravel(),
                "ci_lower": (estimate - margin).ravel(),
                "ci_upper": (estimate + margin).ravel(),
            }
        )

    def lowest_effective_doses(self, group: str | None = None) -> Table:
        """Return the lowest dose from which each gene of a cell type responds, by
        stage-wise calls of each dose against the vehicle in the genes that pass the
        gate.

        The calls of the stage-wise procedure of stageR (Heller et al. 2009; Van den
        Berge et al. 2017), with the gate (:attr:`gate`) as its screen: the gate
        passes R of the G genes it tested at its FDR, alpha; in each gene it passes,
        each dose against the vehicle is called by Holm's step-down procedure with
        Shaffer's first step at the level R alpha / G, through
        :func:`~cytepool.stats.multiple_contrasts.stage_wise_adjusted_p_values`, on
        the moderated t-tests of the table ``"each_dose"``, or, under a gate with a
        fold change, on each dose's TREAT p-value against that fold change, from the
        cell type's fit computed again, as :meth:`fit` computes it. The lowest
        effective dose is the lowest dose called: the lowest whose change is called
        or, with a fold change, the lowest whose change beyond it is called. The
        calls control the overall false discovery rate (OFDR) at alpha: the
        expected share of the genes passed with at least one false call, the gate's
        or a dose's, proved under independence between genes (Heller et al. 2009)
        and expected wherever Benjamini-Hochberg controls the false discovery rate
        (Van den Berge et al. 2017). With ``direction``, the gate is one-sided, a
        test that every dose equals the vehicle as valid as a two-sided one, and the
        doses stay two-sided, so that a dose called can move either way
        (https://docs.nygen.io/CytePool/statistics/dose-response#one-sided-dose-tests).

        An effective dose here is a dose whose change is detectable at the stage-wise
        level, with no threshold on its size but the gate's fold change: toxicology's
        lowest observed effect level (LOEL), and the ICH E4 sense of a dose that
        shows an effect; the clinical minimum effective dose usually adds a
        clinically relevant size. It is a tested dose, not a potency between doses,
        and it moves with the replicates, the spacing of the doses and the gate's
        FDR. A gene can pass the gate with no dose called. The values hold only at the
        gate's FDR, and a call is a value at most it, as stageR counts; another
        level or another test is another gate, such as ``gate={"test": "williams",
        "fdr": 0.1}`` in :func:`dose_response`
        (https://docs.nygen.io/CytePool/statistics/dose-response#lowest-effective-dose).

        Args:
            group: A cell type that was tested; it may be left out when one cell type
                was checked.

        Returns:
            A table of one row per gene tested in the cell type, in the cube's
            feature order, with ``feature_id``, ``symbol`` (when the cube's feature
            table has one), ``passes_gate`` (as in the table ``"gate"``),
            ``lowest_effective_dose`` (in the axis's unit; NaN when no dose is
            called), ``log2_fold_change`` and ``stage_wise_adjusted_p_value`` (that
            dose's change from the vehicle and its stage-wise adjusted p-value; NaN
            with the dose), and ``n_doses_called``; for scores, ``gene_set`` or
            ``source`` and ``estimate`` in place of the genes' columns and
            ``log2_fold_change``.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked, or, under a gate with a fold change, the fit computed again
                no longer gives what the result records.
            KeyError: If the cell type was not checked or not tested, or the gate was
                not run there; the message says why.
            TypeError: If ``group`` is not a string.
        """
        group = self._group(group)
        entry = self._tested(group)
        gate = self._tables.get((group, GATE))
        if gate is None:
            raise KeyError(f"the gate was not run in {group!r}: {self._no_gate(entry)}")
        doses = np.array(entry["doses"][1:], dtype=np.float64)
        each = self._tables[(group, EACH_DOSE)]
        genes = len(gate)

        def by_dose(column: str) -> FloatArray:
            values = np.asarray(each[column], dtype=np.float64)
            return values.reshape(doses.size, genes).T

        setting = self.gate
        if setting["fold_change"] is None:
            per_dose = by_dose("p_value")
        else:
            axis = self.dose
            per_dose = _treat_p_values(
                self.fit(group),
                [dose_group_column(axis.column, dose) for dose in doses],
                setting["fold_change"],
                self.method,
                self.provenance["analysis_settings"]["robust"],
            )
        level = setting["fdr"]
        adjusted = stage_wise_adjusted_p_values(
            np.asarray(gate["p_value"], dtype=np.float64),
            per_dose,
            alpha=level,
            method="holm",
        )
        called = adjusted <= level
        lowest = np.argmax(called, axis=1)
        rows = np.arange(genes)
        some = called.any(axis=1)
        columns = [name for name in _ID_COLUMNS if name in gate]
        change = self._change()
        return Table(
            {
                **{name: gate[name] for name in columns},
                "passes_gate": gate["passes_gate"],
                "lowest_effective_dose": np.where(some, doses[lowest], np.nan),
                change: np.where(some, by_dose(change)[rows, lowest], np.nan),
                "stage_wise_adjusted_p_value": np.where(
                    some, adjusted[rows, lowest], np.nan
                ),
                "n_doses_called": called.sum(axis=1).astype(np.int64),
            }
        )

    def _no_gate(self, entry: Mapping[str, Any]) -> str:
        """Why the gate did not run in a tested cell type: its test did not."""
        test = self.gate["test"]
        why = entry["tests_not_run"].get(test, "it did not run there")
        return f"its test, {test!r}, was not run: {why}"

    def curve_fits(self, group: str | None = None) -> DoseCurveFits:
        """Return the curves of named models of a cell type, with the dose-group
        means and covariance they were fitted to.

        Args:
            group: A cell type with curves; it may be left out when one cell type
                was checked.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked, or has no curves; the message
                says why.
        """
        return self._curve_pair(self._group(group))[1]

    def benchmark_doses(self, group: str | None = None) -> BenchmarkDoses:
        """Return the benchmark doses of a cell type's curves, with their bounds, the
        statistics that the checks of BMDS, NTP, ETAP and EFSA read, and the genes'
        ids.

        Their table with other sets of checks needs no new bounds:
        ``result.benchmark_doses(group).table(flags="ntp_2018")``, whose rows carry the
        genes' ids, as the table ``"benchmark_doses"`` does, without ``symbol`` and
        ``passes_gate``.

        Args:
            group: A cell type with curves; it may be left out when one cell type
                was checked.

        Raises:
            ValueError: If ``group`` is left out and several cell types were
                checked.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not checked or has no curves, or
                dose_response ran without ``bmr``; the message says why.
        """
        group = self._group(group)
        if self.settings.get("bmr") is None:
            raise KeyError(
                "no benchmark doses were computed: dose_response ran without bmr; "
                "pass bmr=BenchmarkResponse.sd(1), for instance, with models"
            )
        self._curve_pair(group)
        return self._benchmark_doses[group]

    def _curve_pair(self, group: str) -> tuple[Table, DoseCurveFits]:
        """The genes and curves of a checked cell type, which must have curves."""
        found = self._curves.get(group)
        if found is not None:
            return found
        if self.settings.get("models") is None:
            raise KeyError(
                "no curves were fitted: dose_response ran without models; pass "
                "models='epa', 'll4' or 'dromics'"
            )
        entry = next(e for e in self.provenance["groups"] if e["group_id"] == group)
        raise KeyError(f"no curves in {group!r}: {entry['curves']['reason']}")

    def _model_curve(self, group: Any, genes: Any, at: Any, model: Any) -> Table:
        """The curves of a named model, or of each gene's best, minus the vehicle's
        mean that they were fitted to."""
        group = self._group(group)
        found, fits = self._curve_pair(group)
        if not isinstance(model, str):
            raise TypeError(
                f"model must be a model name (str), got {type(model).__name__}"
            )
        if model != _BEST and model not in fits.models:
            raise ValueError(
                f"model {model!r} is not one of the curves of {group!r}, "
                f"{quoted(list(fits.models))}, or 'best'"
                + close_matches(model, [*fits.models, _BEST])
            )
        axis = self.dose
        doses = np.asarray(fits.doses, dtype=np.float64)
        at_doses = (
            _curve_grid(axis, doses)
            if at is None
            else _model_curve_doses(at, axis, doses, group)
        )
        rows = _gene_rows(genes, found, group, "fitted")
        if model == _BEST:
            chosen = np.asarray(fits.best[rows], dtype=np.int64)
            missing = chosen < 0
            if missing.any():
                if genes is not None:
                    names = found[found.columns[0]][rows[missing]].tolist()
                    raise ValueError(
                        f"{quoted(names)} {'has' if len(names) == 1 else 'have'} no "
                        f"best model in {group!r}: no model was fitted to it there, "
                        "as the not_fitted_reason column of its curves table says"
                    )
                rows, chosen = rows[~missing], chosen[~missing]
        else:
            chosen = np.full(rows.size, fits.models.index(model), dtype=np.int64)
        values = np.empty((rows.size, at_doses.size))
        for m in np.unique(chosen):
            mine = chosen == m
            spec = model_of(fits.model_set, fits.models[m])
            positions = [fits.parameter_names.index(n) for n in spec.parameters]
            values[mine] = curve_values(
                spec, fits.parameters[rows[mine], m][:, positions], at_doses
            )
        change = values - fits.means[rows, 0][:, None]
        genes_of_rows = found.take(np.repeat(rows, at_doses.size))
        return genes_of_rows.with_columns(
            {
                "model": text_column(
                    [fits.models[m] for m in np.repeat(chosen, at_doses.size)]
                ),
                "dose": np.tile(at_doses, rows.size),
                self._change(): change.ravel(),
            }
        )

    def _moderated(self, group: str) -> tuple[Table, Any]:
        """The features tested in a cell type and their moderated tests: the cube's
        cell type's fit computed again, or the moderated fit of scores the result
        keeps."""
        found = self._fits[group]
        if isinstance(found, ModeratedFit):
            record = self.provenance["scores"]
            column = "gene_set" if record["type"] == "PathwayScores" else "source"
            return Table({column: text_column(record["names"])}), found
        fit = found.fit(group)
        return fit.features, fit.moderated

    def _change(self) -> str:
        """The column of a change from the vehicle: the log2 fold change of a gene,
        or the estimate of a score, as limma names it."""
        scores = json.loads(self._record).get("scores")
        return "log2_fold_change" if scores is None else "estimate"

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
        if name in (*_ALL_TESTS, EACH_DOSE, GATE, CURVES, BENCHMARK_DOSES):
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
        dose = "  dose: " + _axis_text(axis, settings["spline_basis"])
        if LINEAR_TREND in self.tests:
            dose += f"; linear_trend slopes {_slope_unit(axis)}"
        direction = settings.get("direction")
        side = "" if direction is None else f"; one-sided, {direction}"
        scores = record.get("scores")
        noun = "gene"
        if scores is not None:
            noun = "gene set" if scores["type"] == "PathwayScores" else "source"
        lines = [
            f"DoseResponseResult ({record['method']}): {n_groups:,} cell "
            f"type{'' if n_groups == 1 else 's'}, {tested:,} tested; tests: "
            + ", ".join(self.tests)
            + side,
            dose,
            "  gate: " + gate_text(settings["gate"]),
        ]
        if settings.get("models") is not None:
            lines.append(
                f"  curves: {settings['models']}, {settings['variance']} variance"
            )
        if settings.get("bmr") is not None:
            labels = [_bmr_text(BenchmarkResponse(**bmr)) for bmr in settings["bmr"]]
            text = "  benchmark doses at " + ", ".join(labels)
            if settings["flags"]:
                text += "; flags: " + ", ".join(settings["flags"])
            lines.append(text)
        for entry, row in zip(record["groups"], range(n_groups), strict=True):
            group = entry["group_id"]
            if not entry["tested"]:
                lines.append(
                    f"  {group}: not tested, {entry['status']}: {entry['reason']}"
                    + self._curve_text(group)
                    + self._benchmark_text(group)
                )
                continue
            genes = int(self._summary["n_features_tested"][row])
            calls = [
                f"{test} {self._calls(group, test):,}"
                for test in self.tests
                if (group, test) in self._tables
            ]
            gate = self._tables.get((group, GATE))
            passed = (
                "gate not run"
                if gate is None
                else f"{int(np.count_nonzero(gate['passes_gate'])):,} pass the gate"
            )
            text = (
                f"  {group}: {len(entry['doses'])} dose groups, "
                f"{sum(entry['samples']):g} samples, {genes:,} "
                f"{noun}{'' if genes == 1 else 's'}; {passed}; calls at {_FDR:.0%} "
                "FDR: " + ", ".join(calls)
            )
            if entry["tests_not_run"]:
                text += "; not run: " + ", ".join(entry["tests_not_run"])
            mcp = entry["contrast_tests"].get(MCP)
            notes = [] if mcp is None else _mcp_notes(mcp, len(entry["doses"]) - 1)
            if notes:
                text += "; mcp: " + ", and ".join(notes)
            lines.append(text + self._curve_text(group) + self._benchmark_text(group))
        return fitted_lines("\n".join(lines))

    def _curve_text(self, group: str) -> str:
        """The best models of a cell type's curves and its turning best fits, for
        the printout; empty without curves."""
        found = self._curves.get(group)
        if found is None:
            return ""
        fits = found[1]
        text = f"; curves ({fits.variance} variance): best {_best_counts(fits)}"
        turns = _turn_text(fits)
        return f"{text}; {turns}" if turns else text

    def _benchmark_text(self, group: str) -> str:
        """Where a cell type's BMDs lie, for each BMR, for the printout; empty
        without them."""
        found = self._benchmark_doses.get(group)
        if found is None:
            return ""
        parts = [
            f"BMDs at {_bmr_text(bmr)}: {_status_counts(found.status[i])}"
            for i, bmr in enumerate(found.benchmark_responses)
        ]
        bounds = "" if found.bounds else " (no bounds)"
        return "; " + "; ".join(parts) + bounds

    def _calls(self, group: str, test: str) -> int:
        """The genes whose adjusted p-value of a test is at most the printout's FDR,
        as the gate calls them."""
        adjusted = self._tables[(group, test)]["adjusted_p_value"]
        return int(np.count_nonzero(adjusted <= _FDR))


def _bmr_text(bmr: BenchmarkResponse) -> str:
    """A BMR as printouts write it after "at": "1.349 SD", "a change of 0.584963"."""
    return bmr.label if bmr.kind == "sd" else f"a {bmr.label}"


def _with_gene_columns(
    table: Table, genes: Table, rows: np.ndarray, gate: Table | None
) -> Table:
    """A table of a cell type's genes, the gene of each row at ``rows`` of ``genes``:
    its ``feature_id`` named as in ``genes`` (``gene_set`` or ``source`` for scores),
    with ``symbol`` after it when the cube has symbols, and the gate's call of each
    gene, ``passes_gate``, after them where the gate ran."""
    column = genes.columns[0]
    if column != "feature_id":
        table = table.rename({"feature_id": column})
    added: dict[str, Any] = {}
    if "symbol" in genes:
        added["symbol"] = genes["symbol"][rows]
    if gate is not None:
        # The gate's call of each gene or score, matched by its id.
        row_of = {gene: i for i, gene in enumerate(gate[column].tolist())}
        calls = np.asarray(gate["passes_gate"], dtype=bool)
        positions = [row_of[gene] for gene in genes[column].tolist()]
        added["passes_gate"] = calls[positions][rows]
    columns = list(table.columns)
    return table.with_columns(added).select([columns[0], *added, *columns[1:]])


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
    data: Cube | PathwayScores | Activities,
    design: Design,
    *,
    dose: DoseAxis,
    tests: str | Sequence[str] | None = None,
    direction: str | None = None,
    spline_df: int | None = None,
    candidates: CandidateShapes | None = None,
    gate: Mapping[str, Any] | None = None,
    models: str | None = None,
    variance: str = CONSTANT,
    bmr: BenchmarkResponse | Sequence[BenchmarkResponse] | None = None,
    flags: str | Sequence[str] | None = None,
    analysis: Callable[..., DifferentialExpressionResult] = limma_voom,
    batch: str | None = None,
    groups: str | Sequence[str] | None = None,
    min_cells: int = MIN_CELLS,
    min_counts: float = 0,
    rng: Any = None,
    **options: Any,
) -> DoseResponseResult:
    """Test which genes respond to a dose in every cell type of a cube, or which
    scores of a cell type's samples do.

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

    Three multiple contrast tests run when named, each two-sided, or one-sided with
    ``direction``, with one error rate for the gene's contrasts (the single-step
    max-t test, from the multivariate t distribution on the gene's df) and
    Benjamini-Hochberg across genes:

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
    explains them.

    The gate decides which genes count as responding: one test of any response at a
    stated false discovery rate, Benjamini-Hochberg within each cell type, after the
    expression filter, which does not read the doses (Bourgon, Gentleman and Huber
    2010). Its calls are the table ``"gate"``, with ``passes_gate``; the genes it
    passes are those whose doses
    :meth:`DoseResponseResult.lowest_effective_doses` calls. A fold change enters
    the null hypothesis of each dose against the vehicle by TREAT (McCarthy and
    Smyth 2009), and a gene passes when its change beyond the fold change at one dose
    or more is called: the smallest of its doses' TREAT p-values times the doses
    (Bonferroni), then Benjamini-Hochberg. The gate takes the place of BMDExpress's
    prefilter, the step that picks the genes that go on to curves: BMDExpress's
    default prefilter takes an unadjusted Williams p-value and cuts the estimated
    fold change, where the gate adjusts its p-values and puts any fold change inside
    the test. It is not the ANOVA gate of NTP's and EPA's rules, a test of the data
    set that at least one gene must pass before the set is analysed further. With 2
    replicates per dose, a cell type of 1,000 null genes had a gene pass the default
    gate at 5% FDR in 10% to 16% of simulated series, against 7% with 3 replicates
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-gate):
    :func:`~cytepool.permutation_check` runs the gate again with the doses permuted
    and tests whether the true doses pass more genes.

    With ``models``, every gene tested in a cell type, and every gene that passes the
    expression filter in a cell type that supports only description, also gets the
    curves of a set of named models, with their AICc, Akaike weights, lack-of-fit
    tests and best model (:mod:`cytepool.stats.dose_curves`), in the table
    ``"curves"`` and in :meth:`DoseResponseResult.curve_fits`.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-model-curves
    explains the models, the variance modes and how to read the curves. With ``bmr``
    as well, each gene's best curve gives its benchmark dose (BMD), the dose at which
    the curve departs from its level at dose 0 by the benchmark response (BMR), with
    one-sided 95% profile-likelihood bounds given each gene's chosen model, BMDL and
    BMDU, in cell types that support tests
    (:func:`~cytepool.stats.potency.benchmark_doses`), in the table
    ``"benchmark_doses"`` and in :meth:`DoseResponseResult.benchmark_doses`;
    https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses explains them.

    The pathway scores or activities of one cell type's samples, such as those of
    ``gsva(cube, sets, group="pDC")``, are tested in place of the genes, with every
    test, table and curve: fitted as :func:`~cytepool.limma` fits scores, by limma's
    linear model without observation weights and empirical Bayes moderation across
    the scores, which limma leaves out for one or two scores, so that a signature
    tested alone is an ordinary linear model. Their tables name the gene sets or
    sources, with the change in the scores' units as ``estimate``
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-signatures).

    With ``direction``, the tests with a sign are one-sided: ``"dunnett"`` and
    ``"williams"`` test a change from the vehicle that way, ``"linear_trend"`` a
    slope, and ``"mcp"`` agreement with the candidate shapes or their mirror
    images, which a response that moves the other way at some doses can show. The
    F-tests have no direction and are refused with it; each dose against the
    vehicle, the curves and the lowest effective doses stay two-sided. Use a
    direction fixed before the data, such as the expected effect of a drug on a
    pharmacodynamic signature
    (https://docs.nygen.io/CytePool/statistics/dose-response#one-sided-dose-tests).

    The design is checked first by :func:`~cytepool.check_dose_design`, with
    ``batch``, ``groups``, ``min_cells`` and ``min_counts``, and the check is kept as
    the result's ``design_report``: cell types that it calls descriptive or refused
    are listed with its reason and not fitted. A cell type tests the dose groups
    that keep a sample there and that the design can compare with the vehicle; a
    test that its dose groups cannot estimate, such as a spline of more columns than
    its doses support, is not run there, with the reason.

    Args:
        data: A cube with a ``counts`` layer, as :func:`~cytepool.limma_voom` takes;
            or the :class:`~cytepool.PathwayScores` or :class:`~cytepool.Activities`
            of the samples of one cell type, such as :func:`~cytepool.gsva` and
            :func:`~cytepool.ulm` give for a cube's group or an array, whose
            profiles are matched to the design's rows by ``sample_id``. Scores of an
            array, which name no cell type, form the cell type ``"bulk"``.
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
            supports (not with ``direction``), leaving out in each cell type a test
            of the same hypothesis as another, which happens when the dose groups
            are few. ``"spline"`` and ``"linear_trend"`` need two doses on the axis
            (the treated doses when the vehicle is a group of its own, every dose
            when it is on the axis), ``"nonlinearity"`` a spline of two columns or
            more, and ``"mcp"`` two treated doses; a cell type that compares one
            treated dose does not run ``"mcp"``, with the reason. The gate's test,
            the spline by default (the Williams-type test with ``direction``), also
            runs when ``tests`` does not name it, and decides which genes pass the
            gate and whose lowest effective doses are called: ``tests="williams"``
            runs the spline too, and ``gate={"test": "williams"}`` makes the
            Williams-type test decide.
        direction: ``"up"`` or ``"down"`` for tests of a rise or a fall with dose
            only, DoseFinding's ``"increasing"`` and ``"decreasing"``, of the tests
            with a sign that ``tests`` names; None (default) for two-sided tests.
        spline_df: The number of spline columns, at least 1 and at most the doses on
            the axis minus 1; None gives 2 when the axis holds 4 doses or more, and
            1 otherwise. The knots are placed by :meth:`DoseAxis.spline_basis` with
            its rule ``"levels"``, in the restricted cubic form.
        candidates: The :class:`~cytepool.CandidateShapes` of ``"mcp"``, with a
            ``beta_scale`` of at least the top dose; None spreads the default shapes
            over the design's doses, ``CandidateShapes.spread(dose,
            design.samples)``. Given only with ``"mcp"``, in ``tests`` or as the
            gate's test. A shape that does not change at a cell type's doses is left
            out there, with the reason in the record.
        gate: None for the default gate, the spline test at 5% FDR (with
            ``direction``, the one-sided Williams-type test), or a mapping of its
            settings: ``"test"``, a test of any response, ``"spline"`` (the
            default), ``"dose_groups"``, ``"dunnett"``, ``"williams"`` or
            ``"mcp"``, which runs whether ``tests`` names it or not, with
            ``"dose_groups"`` the default when fewer than two doses are on the axis,
            where it asks the spline's question; ``"fdr"``, the false discovery rate
            of the calls, between 0 and 1 (0.05); and ``"fold_change"``, a fold
            change above 1, up or down, that a gene's change from the vehicle must
            exceed at one dose or more, by TREAT, which the gate records as the test
            ``"each_dose"``. ``gate=result.gate`` gives a result's gate back. With
            ``direction``, the gate's test is ``"dunnett"``, ``"williams"`` or
            ``"mcp"``, one-sided, ``"williams"`` by default, which compares the top
            doses with the vehicle: the MCP step's one-sided test calls agreement
            with the mirror image of a rising shape, which a response that rises at
            the top doses only, or rises and returns, also shows. Scores take no
            fold change.
        models: A set of dose-response models to fit to each cell type's genes,
            ``"epa"`` (the eight models of NTP and ETAP in BMDS's forms), ``"ll4"``
            (drc's four-parameter log-logistic curve) or ``"dromics"`` (DRomics'
            models, with curves that rise and then fall); None (default) for no
            curves.
        variance: How the curves weigh each cell type's dose groups:
            ``"constant"`` (default), each gene's values in the cell type's fit
            refitted against dose under constant variance, as BMDS refits a gene's
            samples, with a design of no terms but dose; ``"weighted"``, the
            dose-group means of the cell type's moderated fit by generalised least
            squares with their covariance, which carries voom's weights and the
            design's other terms, and the gene's residual variance; or
            ``"moderated"``, the same with the gene's moderated variance. Cell types
            that support only description are fitted under constant variance.
        bmr: The benchmark response, a :class:`~cytepool.BenchmarkResponse` such as
            ``BenchmarkResponse.sd(1.349)`` or ``BenchmarkResponse.absolute(0.5)``,
            or a list of them, for the BMDs of each gene's best curve, with the 1 SD
            BMD beside them; None (default) for none. Needs ``models``.
        flags: The checks of regulatory guidance whose failures the table
            ``"benchmark_doses"`` lists: those of BMDS (``"bmds_2026"``), NTP 2018
            (``"ntp_2018"``), EPA's ETAP 2024 (``"etap_2024"``) or EFSA 2022
            (``"efsa_2022"``), a set's name or a list of names; None (default) for
            none. Needs ``bmr``.
        analysis: :func:`~cytepool.limma_voom` (by default),
            :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom`; a design
            with random intercepts needs ``dream_voom``. Left at its default for
            scores, which are fitted as :func:`~cytepool.limma` fits them.
        batch: A column of the design's samples that names each sample's processing
            batch, which the check uses to find dose groups confounded with it. It
            is checked, not modelled: a batch to adjust for belongs in the design.
        groups: A group id or a list of them; all groups by default.
        min_cells: The fewest cells a sample needs in a cell type, as in
            :func:`~cytepool.limma_voom`; for the check and the fits.
        min_counts: The fewest counts, summed over all features, a sample needs in a
            cell type, as in :func:`~cytepool.limma_voom`; for the check and the
            fits. ``groups``, ``min_cells`` and ``min_counts`` stay at their defaults
            for scores, which chose their samples when they were computed.
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
            (``dream_voom``); their values are checked by the analysis. For scores,
            ``trend`` and ``robust``, as :func:`~cytepool.limma` takes them, and
            ``n_threads``. Moderating 30 sets or more of different sizes, pass
            ``trend=scores.n_genes``, which lets the prior variance follow the set
            sizes, or ``trend=activities.n_targets.mean(axis=0)``, the sources'
            targets: on null series it held the sets' error near 5%, where the
            default, as in limma, called the smallest third of the sets in up to
            15%. With fewer sets, test each set alone
            (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-signatures).

    Returns:
        The tables of every cell type tested, the summary, the design check, each
        cell type's fit and curves, the settings and a provenance record.

    Warns:
        RuntimeWarning: As the analysis warns, such as when sample quality weights
            cannot be estimated in a cell type.

    Raises:
        TypeError: If ``analysis`` is not one of the three functions, an option is
            one the analysis does not take or one that dose_response sets
            (``contrasts``, ``f_tests``, ``adjust_across_groups``), or for scores
            other than ``trend``, ``robust`` and ``n_threads``; ``tests``, ``direction``,
            ``spline_df``, ``candidates``, ``gate`` or one of its settings,
            ``models``, ``variance``, ``bmr``, ``flags`` or ``rng`` has the wrong
            type, or an argument is of a type that
            :func:`~cytepool.check_dose_design` or the analysis refuses.
        ValueError: If ``tests`` names no test or a test twice, or names a test
            that the design cannot support; ``spline_df`` is below 1 or above the
            doses on the axis minus 1, or given without two doses on the axis;
            ``candidates`` are given without ``"mcp"`` or with a ``beta_scale``
            below the top dose; a multiple contrast test is named with
            ``contrast_standard_errors="limma"``; ``gate`` has a setting it does not
            know, a test that is not a test of any response or that the design
            cannot support, an FDR outside (0, 1), or a fold change that is not
            finite and above 1, given with another test than ``"each_dose"``, with
            ``dream_voom`` or with scores, or ``"each_dose"`` without one; ``rng``
            is a negative seed; the design names contrasts, has random intercepts
            with an analysis of limma or none with ``dream_voom``; the sample table
            already holds a column of the name of a dose group's indicator;
            ``models`` or ``variance`` names no set or mode, ``variance`` is other
            than ``"constant"`` without ``models``, ``models`` is given with
            ``dream_voom``, or ``variance="constant"`` with a design of other terms
            than dose; ``direction`` names no direction, or is given with
            ``tests=None``, with an F-test or with a gate of a test without a sign;
            with scores, ``analysis`` is not the default; ``bmr`` is empty, repeats
            a BMR or is given without ``models``; ``flags`` names an unknown set or
            one twice, or is given without ``bmr``; or as
            :func:`~cytepool.check_dose_design` and the analysis raise.
        KeyError: If a group is not in the cube.
    """
    given = _tests_setting(tests)
    side = _direction_setting(direction, given)
    asked = _signed_gate(_gate_setting(gate), side)
    linear = None if given is None else [test for test in given if test in TESTS]
    # The gate's multiple contrast test runs whether tests names it or not.
    named = [
        test
        for test in CONTRAST_TESTS
        if (given is not None and test in given) or test == asked["test"]
    ]
    if spline_df is not None:
        spline_df = positive_integer(spline_df, "spline_df")
    if candidates is not None and not isinstance(candidates, CandidateShapes):
        raise TypeError(
            "candidates must be a CandidateShapes, such as CandidateShapes.spread(axis, "
            f"design.samples) gives, got {type(candidates).__name__}"
        )
    if candidates is not None and MCP not in named:
        raise ValueError(
            "candidates are the shapes of the MCP step, and neither tests nor the gate "
            "names 'mcp'; add 'mcp' to tests, or leave candidates out"
        )
    # The seed is checked whatever the tests, and recorded when one draws on it.
    seed = recorded_seed(rng)
    scores = (
        sample_scores(data, "dose_response")
        if isinstance(data, PathwayScores | Activities)
        else None
    )
    if scores is None:
        method = _analysis_name(analysis, design)
        _check_options(method, options)
    else:
        _score_options(analysis, options, asked)
        method = _LIMMA
    if named and options.get("contrast_standard_errors") == "limma":
        raise ValueError(
            f"the multiple contrast tests, {quoted(named)}, need each gene's covariance "
            "of its doses' estimates, which limma's approximate standard errors "
            "(contrast_standard_errors='limma') do not give, so multiple_contrast_test "
            "refuses them too; leave contrast_standard_errors out, for the exact default"
        )
    model_set, variance = _curve_settings(models, variance, method, design)
    responses, flag_sets = _benchmark_settings(bmr, flags, model_set)
    if isinstance(design, Design) and "expression_filter" in options:
        # The column of sample groups, if any, is checked here for every cell type,
        # also those that only describe and so run no analysis.
        check_sample_groups(
            {"expression_filter": filter_setting(options["expression_filter"])},
            design.samples,
        )
    if isinstance(design, Design) and len(design.contrasts):
        raise ValueError(
            f"the design names contrasts ({quoted(list(design.contrasts.names))}), "
            "which dose_response does not test: it adds the dose groups and tests "
            "them itself; build the design without contrasts, and test other "
            "contrasts with limma_voom"
        )
    report = check_dose_design(
        data,
        design,
        dose=dose,
        batch=batch,
        min_cells=min_cells,
        min_counts=min_counts,
        groups=groups,
    )
    # The check refused data that are neither a cube nor scores.
    source = cast(Cube, data) if scores is None else scores
    levels = np.unique(report.doses["dose"])
    chosen = _gate(asked, dose, levels, method, side)
    if linear is not None and chosen["test"] in TESTS and chosen["test"] not in linear:
        linear.append(chosen["test"])
    splines = _splines(dose, design.samples, levels, linear, spline_df)
    shapes = _candidates(candidates, dose, design, levels) if MCP in named else None
    statuses = report.summary["status"].tolist()
    reasons = report.summary["reason"].tolist()
    needs_model = INFERENTIAL in statuses or (
        model_set is not None and DESCRIPTIVE in statuses
    )
    model = _dose_group_model(design, dose, levels) if needs_model else None
    fits: list[_CellTypeFit] = []
    curves: dict[str, tuple[Table, DoseCurveFits]] = {}
    curve_records: dict[str, dict[str, Any]] = {}
    doses_of: dict[str, BenchmarkDoses] = {}
    groups_checked = report.summary["group_id"].tolist()
    for group, status, reason in zip(groups_checked, statuses, reasons, strict=True):
        if status != INFERENTIAL:
            fits.append(_CellTypeFit(group, status, reason))
        else:
            assert model is not None  # built when a cell type supports tests
            comparison = _comparison(
                report.doses.filter(group_id=group),
                dose,
                splines,
                given is None,
                named,
                shapes,
                chosen["test"],
            )
            extra = {}
            if model_set is not None and variance != CONSTANT:
                rows = report.samples.filter(group_id=group)
                used = rows["sample_id"][np.asarray(rows["used"], dtype=bool)]
                extra[VEHICLE_MEAN] = _vehicle_mean_weights(
                    model, used.tolist(), _indicator_columns(dose, levels)
                )
            if isinstance(source, Cube):
                fit = _fit_cell_type(
                    source,
                    model,
                    group,
                    comparison,
                    analysis,
                    min_cells=min_cells,
                    min_counts=min_counts,
                    options=options,
                    extra_contrasts=extra,
                )
            else:
                fit = _fit_scores(
                    source,
                    model,
                    report.samples.filter(group_id=group),
                    comparison,
                    options,
                    extra,
                )
            if fit.result is not None:
                if side is not None and LINEAR_TREND in fit.tables:
                    fit.tables[LINEAR_TREND] = _one_sided(
                        fit.tables[LINEAR_TREND], side
                    )
                # The first stage, computed once for the tests and the curves.
                needed = comparison.contrast_tests or model_set is not None
                if needed and isinstance(fit.result, DifferentialExpressionResult):
                    fit = replace(fit, stage=_first_stage(fit.result.fit(group), dose))
                if comparison.contrast_tests:
                    _run_contrast_tests(fit, seed, options.get("n_threads"), side)
                # The gate reads the tests in their direction.
                _run_gate(fit, chosen, method)
            fits.append(fit)
        if model_set is not None:
            found, curve_records[group] = _cell_type_curves(
                source,
                design,
                model,
                report,
                fits[-1],
                dose,
                model_set,
                variance,
                analysis,
                options,
            )
            if found is not None:
                curves[group] = found
                if responses is not None:
                    genes = found[0]
                    doses_of[group] = replace(
                        benchmark_doses(
                            found[1],
                            responses,
                            bounds=status == INFERENTIAL,
                            n_threads=options.get("n_threads"),
                        ),
                        feature_ids=read_only(
                            np.asarray(genes[genes.columns[0]].tolist(), dtype=object)
                        ),
                    )
    _adjust_across_groups(fits, (*splines.tests, *named))
    return _result(
        data,
        design,
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
        chosen,
        (model_set, variance, curves, curve_records),
        side,
        (responses, flag_sets, doses_of),
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


def _direction_setting(direction: Any, given: list[str] | None) -> str | None:
    """The direction of one-sided tests, or None for two-sided ones; refused with
    the tests that have no direction."""
    if direction is None:
        return None
    if not isinstance(direction, str):
        reject_masked(direction, "direction")
        raise TypeError(
            "direction must be 'up' or 'down', or None for two-sided tests, got "
            f"{type(direction).__name__}"
        )
    if direction not in DIRECTIONS:
        raise ValueError(
            "direction must be 'up' (DoseFinding's 'increasing') or 'down' "
            f"('decreasing'), or None for two-sided tests, got {direction!r}"
        )
    signed = quoted(SIGNED_TESTS)
    if given is None:
        raise ValueError(
            f"direction={direction!r} makes the tests with a sign one-sided, and "
            "tests=None runs the default tests, whose F-tests (spline, nonlinearity, "
            f"dose_groups) test a change either way; name tests among {signed}, such "
            "as tests=('linear_trend', 'mcp')"
        )
    unsigned = [test for test in given if test not in SIGNED_TESTS]
    if unsigned:
        one = len(unsigned) == 1
        raise ValueError(
            f"direction={direction!r} makes the tests with a sign one-sided, and "
            f"{quoted(unsigned)} {'is an F-test' if one else 'are F-tests'} of a "
            f"change either way, with no one-sided form; leave "
            f"{'it' if one else 'them'} out of tests, or leave direction out for "
            "two-sided tests"
        )
    return direction


def _score_options(
    analysis: Any, options: Mapping[str, Any], gate: Mapping[str, Any]
) -> None:
    """Refuse an analysis, options and a gate that scores do not take: scores are
    fitted as limma fits them, with its options ``trend`` and ``robust``, and have no
    fold change; ``n_threads`` is checked here, since the fit does not read it."""
    if gate["test"] == EACH_DOSE:
        raise ValueError(
            "a gate with a fold change tests each dose's log2 fold change against it "
            "by TREAT, and scores have no fold change: their changes are in the "
            "scores' own units; leave the fold change out"
        )
    if analysis is not limma_voom:
        raise ValueError(
            "scores are fitted as limma fits them, by its linear model and empirical "
            "Bayes moderation across the scores, without voom's weights, which are "
            "for counts; leave analysis out"
        )
    for name, value in options.items():
        reject_masked(value, name)
        if name not in _SCORE_OPTIONS:
            raise TypeError(
                f"the fit of scores has no option {name!r}"
                f"{close_matches(name, _SCORE_OPTIONS)}; its options are trend and "
                "robust, as limma takes them, and n_threads"
            )
    threads_setting(options.get("n_threads"))


def _gate_setting(value: Any) -> dict[str, Any]:
    """The gate's settings as given, checked: its test (None for the default), its
    FDR and its fold change; a fold change makes the test ``"each_dose"``."""
    reject_masked(value, "gate")
    if value is None:
        return {"test": None, "fdr": _FDR, "fold_change": None}
    if not isinstance(value, Mapping):
        raise TypeError(
            "gate must be a mapping of its settings, such as {'test': 'spline', 'fdr': "
            "0.05} or {'fold_change': 1.5}, or None for the spline test at 5% FDR, got "
            f"{type(value).__name__}"
        )
    for key in value:
        if key not in _GATE_SETTINGS:
            raise ValueError(
                f"gate has no setting {key!r}{close_matches(str(key), _GATE_SETTINGS)}; "
                f"its settings are {quoted(_GATE_SETTINGS)}"
            )
    test = value.get("test")
    if test is not None and not isinstance(test, str):
        raise TypeError(f"gate['test'] must be a test name, got {type(test).__name__}")
    if test in (LINEAR_TREND, NONLINEARITY):
        raise ValueError(
            f"gate['test'] is {test!r}, which tests a straight line or a departure from "
            f"one, not any response; the gate's tests are {quoted(GATE_TESTS)}"
        )
    if test is not None and test not in (*GATE_TESTS, EACH_DOSE):
        hint = (
            " (the one-way ANOVA of the dose groups is 'dose_groups')"
            if "anova" in test.lower()
            else close_matches(test, (*GATE_TESTS, EACH_DOSE))
        )
        raise ValueError(
            f"gate['test'] is {test!r}, which is not one of the gate's tests{hint}; "
            f"they are {quoted(GATE_TESTS)}, and 'each_dose' with a fold change"
        )
    fdr = real_number(value.get("fdr", _FDR), "gate['fdr']", finite=True)
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"gate['fdr'] must be between 0 and 1, got {fdr}")
    fold_change = value.get("fold_change")
    if fold_change is None:
        if test == EACH_DOSE:
            raise ValueError(
                "the gate 'each_dose' tests each dose against a fold change by TREAT and "
                "needs gate['fold_change'], such as 1.5; without one, 'dunnett' tests "
                "each dose against the vehicle with control over the gene's doses"
            )
        return {"test": test, "fdr": fdr, "fold_change": None}
    change = real_number(fold_change, "gate['fold_change']", finite=True)
    if not change > 1.0:
        raise ValueError(
            "gate['fold_change'] must be above 1, the change from the vehicle, up or "
            f"down, that a gene must exceed at one dose or more, such as 1.5; got {change}"
        )
    if test not in (None, EACH_DOSE):
        raise ValueError(
            f"the gate's test {test!r} takes no fold change; pass "
            f"gate={{'fold_change': {change:g}}} alone, which tests each dose against "
            "the vehicle beyond that fold change by TREAT"
        )
    return {"test": EACH_DOSE, "fdr": fdr, "fold_change": change}


def _gate(
    asked: Mapping[str, Any],
    axis: DoseAxis,
    levels: npt.NDArray[np.float64],
    method: str,
    direction: str | None,
) -> dict[str, Any]:
    """The gate of the design: its test as given, or the spline, or the dose groups
    when fewer than two doses are on the axis, where no spline is placed and the
    dose-group test asks the same question (with a direction, a multiple contrast
    test, chosen already); refused where the design cannot run it."""
    on_axis = _doses_on_axis(axis, levels)
    test = asked["test"]
    if test is None:
        test = SPLINE if on_axis >= 2 else DOSE_GROUPS
    treated = levels[levels > 0.0]
    if test == SPLINE and on_axis < 2:
        raise ValueError(
            "the gate's test 'spline' runs along the dose axis and needs two doses on "
            f"it, and the design has {'one' if on_axis else 'none'}; pass "
            "gate={'test': 'dose_groups'}, which asks the same question, or leave the "
            "gate's test out"
        )
    if test == MCP and treated.size < 2:
        other = "dunnett" if direction is not None else "dose_groups"
        raise ValueError(
            "the gate's test 'mcp' compares candidate shapes along the treated doses, "
            f"and the design has {treated.size}; pass gate={{'test': {other!r}}}"
        )
    if test == EACH_DOSE and method == "dream_voom":
        raise ValueError(
            "a gate with a fold change tests each dose by TREAT on limma's moderated "
            "t-tests, which dream_voom's tests are not; leave the fold change out. "
            "TREAT runs under limma_voom and limma_trend, which fit designs without "
            "random effects"
        )
    return {**asked, "test": test}


def _signed_gate(asked: dict[str, Any], direction: str | None) -> dict[str, Any]:
    """The gate's settings with a direction: its test a multiple contrast test,
    which has a sign, as given, or by default the Williams-type test, which then runs
    too; as given without a direction.

    The Williams-type test compares the pooled top doses with the vehicle, as
    Dunnett's test compares each dose, so a gate for a fall passes no response that
    stays above the vehicle; the MCP step's contrasts compare the dose groups among
    themselves, and a response that rises only at the top doses or rises and returns
    follows the mirror image of a rising shape."""
    if direction is None:
        return asked
    test = asked["test"]
    if test is None:
        return {**asked, "test": WILLIAMS}
    if test not in CONTRAST_TESTS:
        what = (
            "each dose against a fold change either way, by TREAT"
            if test == EACH_DOSE
            else "a change either way"
        )
        raise ValueError(
            f"direction={direction!r} makes the gate one-sided, and the gate's test "
            f"{test!r} tests {what}; gate on one of {quoted(CONTRAST_TESTS)}, or "
            "leave direction out"
        )
    return asked


def _doses_on_axis(axis: DoseAxis, levels: npt.NDArray[np.float64]) -> int:
    """The doses on the axis: the treated ones when the vehicle is a group of its
    own, all of them when it sits on the axis."""
    if axis.vehicle == "own_group":
        return int(np.count_nonzero(levels > 0.0))
    return int(levels.size)


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
    method = next((name for name, known in ANALYSES.items() if analysis is known), None)
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
        for name, parameter in inspect.signature(ANALYSES[method]).parameters.items()
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
            others = [o for o in accepted if o not in SET_BY_DOSE_RESPONSE]
            raise TypeError(
                f"{method} has no option {name!r}{close_matches(name, others)}; its "
                f"options are: {listing(others, limit=len(others))}"
            )


def _curve_settings(
    models: Any, variance: Any, method: str, design: Any
) -> tuple[str | None, str]:
    """The model set of the curves and their variance mode, refused before anything
    runs when they do not suit each other, the analysis or the design."""
    reject_masked(models, "models")
    reject_masked(variance, "variance")
    mode = checked_choice(variance, "variance", VARIANCES)
    if models is None:
        if mode != CONSTANT:
            raise ValueError(
                f"variance={mode!r} sets how curves weigh the dose groups, and models "
                "is None, so no curves are fitted; name a model set, such as "
                "models='epa', or leave variance out"
            )
        return None, mode
    model_set = _checked_model_set(models)
    if method == "dream_voom":
        raise ValueError(
            "dose_response fits curves to the fits of limma_voom and limma_trend, "
            "which give each cell type's dose-group means and their covariance; "
            "dream_voom's tests run without models"
        )
    others = list(design.terms) if isinstance(design, Design) else []
    if mode == CONSTANT and others:
        raise ValueError(
            "variance='constant' refits each gene's values against dose alone, as "
            f"BMDS does, which would leave out the design's other terms "
            f"({quoted(others)}); use variance='weighted' or 'moderated', which fit "
            "the dose-group means of each cell type's fit, adjusted for them"
        )
    return model_set, mode


def _benchmark_settings(
    bmr: Any, flags: Any, model_set: str | None
) -> tuple[list[BenchmarkResponse] | None, tuple[str, ...] | None]:
    """The BMRs and the sets of checks of the benchmark doses, refused before anything
    runs when they do not suit the curves."""
    if bmr is None:
        if flags is not None:
            raise ValueError(
                "flags name the checks of BMDS, NTP, ETAP or EFSA of benchmark doses, "
                "and bmr is None, so no benchmark doses are computed; give bmr, such "
                "as BenchmarkResponse.sd(1), or leave flags out"
            )
        return None, None
    responses = _checked_responses(bmr)
    flag_sets = checked_flag_sets(flags)
    if model_set is None:
        raise ValueError(
            "bmr sets the change that defines each gene's benchmark dose on its curve, "
            "and models is None, so no curves are fitted; name a model set, such as "
            "models='epa'"
        )
    return responses, flag_sets


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
    m = _doses_on_axis(axis, levels)
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
    gate: str,
) -> _Comparison:
    """The dose groups that a cell type compares, from its rows of the check's doses
    table, and its tests as contrasts of the treated doses against the vehicle; of
    two tests of the same hypothesis, the gate's test runs.

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
        _leave_out_repeats(tests, contrasts, not_run, gate)
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
    gate: str,
) -> None:
    """Leave out each test of the same hypothesis as one before it in ``_PRIORITY``,
    the gate's test first, with the reason: with few dose groups, the spline can pass
    through every group's mean, or be the straight line itself."""
    kept: list[str] = []
    for test in sorted(_PRIORITY, key=lambda name: name != gate):
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
    # The analysis's result, with tables, or the moderated fit of scores.
    result: DifferentialExpressionResult | ModeratedFit | None = None
    tables: dict[str, Table] = field(default_factory=dict)
    # The fit as the multiple contrast tests and the curves read it, built when they
    # need it.
    stage: "_FirstStage | None" = None
    # The record of a tested cell type's fit: its entry, its contrasts' weights, the
    # summary's numbers and the analysis settings; empty when it was not tested.
    record: dict[str, Any] = field(default_factory=dict)


class _FirstStage(NamedTuple):
    """A tested cell type's fit of its dose groups, as its multiple contrast tests
    and curves read it."""

    features: Table  # feature_id and symbol, or the scores' gene_set or source
    doses: FloatArray  # the dose of each sample fitted
    values: FloatArray  # the values fitted, samples x features
    moderated: Any  # its moderated tests, a ModeratedFit, or dream's


def _first_stage(fit: GroupFit, axis: DoseAxis) -> _FirstStage:
    """The first stage of a cube's cell type, from its fit computed again."""
    return _FirstStage(
        fit.features,
        np.asarray(fit.samples[axis.column], dtype=np.float64),
        np.asarray(fit.log2_cpm, dtype=np.float64),
        fit.moderated,
    )


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
    extra_contrasts: Mapping[str, npt.NDArray[np.float64]],
) -> _CellTypeFit:
    """Fit a cell type's dose groups and run its tests, as contrasts of the fit, with
    the contrasts that its curves need besides.

    The check and the fit choose a cell type's samples alike, so the fit compares
    the dose groups the check counted; a contrast the fit could not estimate would
    make ``result.table`` raise, with the reason.
    """
    contrasts, f_tests = _contrasts(model, comparison)
    contrasts |= extra_contrasts
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
    tables[EACH_DOSE] = _each_dose_table(
        [result.table(group, name) for name in comparison.indicators],
        comparison.doses[1:],
    )
    record = {
        "fit": entry,
        "contrasts": result.provenance["design"]["contrasts"],
        "summary": {name: float(result.summary[name][0]) for name in _FIT_NUMBERS},
        "analysis": result.settings,
    }
    return _CellTypeFit(
        group, INFERENTIAL, "", comparison, result, tables, None, record
    )


def _fit_scores(
    scores: SampleScores,
    model: Design,
    samples: Table,
    comparison: _Comparison,
    options: Mapping[str, Any],
    extra_contrasts: Mapping[str, npt.NDArray[np.float64]],
) -> _CellTypeFit:
    """Fit the scores of a cell type to its dose groups as limma fits scores, and run
    its tests, as contrasts of the fit, with the contrasts its curves need besides.

    ``samples`` are the cell type's rows of the check's sample table, whose used
    samples are fitted: the columns of the model of the dose groups that R's
    ``qr()`` keeps on their rows, without weights, moderated across the scores.

    A score whose residuals are rounding errors, as those of a score that does not
    vary within its dose groups, is fitted as a missing score, with the reason in
    the record: limma would test its rounding noise, and its variance near 0 would
    pull the prior of every other score towards 0.
    """
    group = str(samples["group_id"][0])
    sample_ids, values, doses = _used_scores(scores, samples)
    row_of_design = {
        sample: row for row, sample in enumerate(model.samples["sample_id"])
    }
    x = model.matrix[[row_of_design[sample] for sample in sample_ids]]
    aliasing = _aliasing(x)
    kept = list(aliasing.kept)
    contrasts, f_tests = _contrasts(model, comparison)
    contrasts |= extra_contrasts
    names = tuple(contrasts)
    matrix = np.column_stack([contrasts[name] for name in names])
    trend, robust = options.get("trend", False), options.get("robust", False)
    tested = Contrasts(matrix[kept], names, tuple(model.columns[j] for j in kept))
    moderated = fit_scores(values, x[:, kept], tested, trend=trend, robust=robust)
    largest = np.abs(np.where(np.isnan(values), 0.0, values)).max(axis=0)
    flat = moderated.fit.sigma <= _ROUNDING * largest
    if flat.any():
        if np.all(flat | np.isnan(moderated.fit.sigma)):
            return _CellTypeFit(
                group,
                INFERENTIAL,
                "no score varies beyond what the model of the dose groups fits: "
                "their residuals are rounding errors, so none can be tested",
                comparison,
            )
        values = np.where(flat, np.nan, values)
        moderated = fit_scores(values, x[:, kept], tested, trend=trend, robust=robust)
    ids = Table({scores.id_column: scores.ids})
    tables: dict[str, Table] = {}
    f_test_df1: dict[str, int] = {}
    for test in comparison.tests:
        if test == LINEAR_TREND:
            tables[test] = contrast_table(
                scores.id_column, scores.ids, moderated, names.index(test)
            )
            continue
        members = f_tests[test]
        found = _limma_f_test(moderated, [names.index(name) for name in members])
        f_test_df1[test] = found.df1
        tables[test] = ids.with_columns(
            {
                "average": moderated.fit.mean_expression,
                "f_statistic": found.f_statistic,
                "df": found.df2,
                "p_value": found.p_value,
                "adjusted_p_value": adjust_p_values(found.p_value, "bh"),
                "standardized_f_statistic": found.standardized_f_statistic,
            }
        )
    tables[EACH_DOSE] = _each_dose_table(
        [
            contrast_table(scores.id_column, scores.ids, moderated, names.index(i))
            for i in comparison.indicators
        ],
        comparison.doses[1:],
    )
    record = {
        "fit": {
            "group_id": group,
            "status": FITTED,
            "reason": "",
            "samples": sample_ids,
            "left_out": {
                sample: reason
                for sample, reason in zip(
                    samples["sample_id"].tolist(),
                    samples["reason"].tolist(),
                    strict=True,
                )
                if reason
            },
            "aliased_columns": [model.columns[j] for j in aliasing.aliased],
            "n_features_tested": int(values.shape[1]),
            "scores_not_tested": {
                name: "its residuals are rounding errors: it does not vary beyond "
                "what the model of the dose groups fits, so its tests would be "
                "rounding noise"
                for name in scores.ids[flat].tolist()
            },
            "f_test_df1": f_test_df1,
        },
        "contrasts": {name: contrasts[name].tolist() for name in names},
        # The summary's numbers; the prior df can be infinite, which JSON cannot hold.
        "summary": {
            "n_features_tested": float(values.shape[1]),
            "df_residual": float(len(sample_ids) - len(kept)),
            "df_prior": float(np.median(moderated.df_prior)),
        },
        # The settings the fit has checked, as the result records an analysis's.
        "analysis": {
            "trend": trend
            if isinstance(trend, bool | np.bool_)
            else np.asarray(trend, dtype=np.float64).tolist(),
            "robust": bool(robust),
        },
    }
    stage = _FirstStage(ids, doses, values, moderated)
    return _CellTypeFit(
        group, INFERENTIAL, "", comparison, moderated, tables, stage, record
    )


def _used_scores(
    scores: SampleScores, samples: Table
) -> tuple[list[str], FloatArray, FloatArray]:
    """The samples a cell type of scores uses, by its rows of the check's sample
    table, with their scores, samples x scores, and their doses."""
    used = np.asarray(samples["used"], dtype=bool)
    sample_ids = samples["sample_id"][used].tolist()
    row_of = {sample: row for row, sample in enumerate(scores.sample_ids)}
    values = scores.values[[row_of[sample] for sample in sample_ids]]
    return sample_ids, values, np.asarray(samples["dose"], dtype=np.float64)[used]


def _one_sided(table: Table, direction: str) -> Table:
    """A table of a t-test with its p-values in one direction: for the two-sided p
    of t, P(T >= t) is p / 2 when t >= 0 and 1 - p / 2 when t < 0, on any
    distribution symmetric about 0, and P(T <= t) the mirror image; adjusted again
    within the cell type."""
    t = np.asarray(table["t"], dtype=np.float64)
    p = np.asarray(table["p_value"], dtype=np.float64)
    toward = t >= 0.0 if direction == UP else t <= 0.0
    one_sided = np.where(toward, p / 2.0, 1.0 - p / 2.0)
    return table.with_columns(
        {"p_value": one_sided, "adjusted_p_value": adjust_p_values(one_sided, "bh")}
    )


def _run_contrast_tests(
    fit: _CellTypeFit, seed: int, n_threads: Any, direction: str | None
) -> None:
    """Run the multiple contrast tests of a fitted cell type and add their tables.

    Each gene's dose-group estimates and their moderated covariance come from the
    cell type's first stage; the multivariate t takes one df per gene, the smallest
    of its doses' under dream; with a direction, each test is one-sided. Each test
    draws its random numbers from ``numpy.random.default_rng([seed, crc32 of the cell
    type's name, its position in CONTRAST_TESTS])``, so that its p-values depend
    neither on the other cell types nor on the order of the run.
    """
    comparison, stage = fit.comparison, fit.stage
    assert comparison is not None and stage is not None
    estimates = _dose_group_estimates(stage.moderated, comparison.indicators)
    df = estimates.df_total
    if df.ndim == 2:
        df = df.min(axis=1)
    covariances = estimates.covariance * estimates.posterior_variance[:, None, None]
    each = fit.tables[EACH_DOSE]
    genes = each.take(np.arange(estimates.theta.shape[0]))
    # The features' columns, those before the dose; a cube's mean log-CPM, or the
    # scores' mean.
    columns = [name for name in _ID_COLUMNS if name in each]
    average = "average_log2_cpm" if "average_log2_cpm" in each else "average"
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
            alternative=_ALTERNATIVES[direction],
            rng=generator,
            n_threads=n_threads,
        )
        table = outcome.table()
        fit.tables[test] = Table(
            {
                **{name: genes[name] for name in columns},
                average: genes[average],
                "contrast": table["contrast"],
                "t": table["t"],
                "df": outcome.df,
                "p_value": table["p_value"],
                "p_value_error": table["p_value_error"],
                "adjusted_p_value": table["adjusted_p_value"],
            }
        )


def _run_gate(fit: _CellTypeFit, gate: Mapping[str, Any], method: str) -> None:
    """Add the gate's calls to a fitted cell type, where the gate's test ran.

    The gate's p-values are those of its test, or, with a fold change, the smallest
    of each gene's TREAT p-values of the doses times the doses, capped at 1; then
    Benjamini-Hochberg over the cell type's genes, and a gene passes at an adjusted
    p-value at most the gate's FDR.
    """
    test = gate["test"]
    if test == EACH_DOSE:
        comparison, result = fit.comparison, fit.result
        # A cube's: dose_response refuses a fold change for scores.
        assert comparison is not None
        assert isinstance(result, DifferentialExpressionResult)
        # The analysis's result keeps the cell type it fitted last, so the fit that
        # the multiple contrast tests computed, when they ran, serves here again.
        per_dose = _treat_p_values(
            result.fit(fit.group),
            comparison.indicators,
            gate["fold_change"],
            method,
            result.settings["robust"],
        )
        p_values = _smallest_times_doses(per_dose)
        genes = fit.tables[EACH_DOSE].take(np.arange(per_dose.shape[0]))
    elif test in fit.tables:
        genes = fit.tables[test]
        p_values = np.asarray(genes["p_value"], dtype=np.float64)
    else:
        return  # the test did not run here; tests_not_run says why
    adjusted = np.asarray(adjust_p_values(p_values, "bh"), dtype=np.float64)
    columns = [name for name in _ID_COLUMNS if name in genes]
    fit.tables[GATE] = Table(
        {
            **{name: genes[name] for name in columns},
            "p_value": p_values,
            "adjusted_p_value": adjusted,
            "passes_gate": adjusted <= gate["fdr"],
        }
    )


def _treat_p_values(
    fit: GroupFit,
    names: Sequence[str],
    fold_change: float,
    method: str,
    robust: bool,
) -> FloatArray:
    """Each dose's TREAT p-value against a fold change, ``(n_genes, n_doses)``: the
    tests of |theta_k| <= log2(fold change) (McCarthy and Smyth 2009), on the
    variances moderated as the cell type's fit moderates them."""
    contrast_fit = fit.moderated.fit
    tested = treat(
        contrast_fit,
        fc=fold_change,
        trend=method == "limma_trend",
        robust=robust,
    )
    columns = [contrast_fit.coefficient_names.index(name) for name in names]
    return np.asarray(tested.p_value[:, columns], dtype=np.float64)


def _smallest_times_doses(per_dose: FloatArray) -> FloatArray:
    """Each gene's smallest p-value of its doses times the doses compared, capped at
    1 (Bonferroni), so that a gene counts once; NaN where every dose is NaN."""
    present = ~np.isnan(per_dose)
    smallest = np.min(np.where(present, per_dose, np.inf), axis=1)
    combined = np.minimum(1.0, per_dose.shape[1] * smallest)
    return np.where(present.any(axis=1), combined, np.nan)


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


def _indicator_columns(axis: DoseAxis, levels: npt.NDArray[np.float64]) -> list[str]:
    """The columns of the model of the dose groups that indicate a treated dose, as
    the design names them."""
    return [
        quote_name(dose_group_column(axis.column, level))
        for level in levels[levels > 0]
    ]


# Curves of named models.


def _cell_type_curves(
    data: Cube | SampleScores,
    design: Design,
    model: Design | None,
    report: DoseDesignReport,
    fit: "_CellTypeFit",
    axis: DoseAxis,
    model_set: str,
    variance: str,
    analysis: Callable[..., DifferentialExpressionResult],
    options: Mapping[str, Any],
) -> tuple[tuple[Table, DoseCurveFits] | None, dict[str, Any]]:
    """A cell type's genes and curves, with their record; no curves, with the reason
    in the record, when the cell type has none."""
    record: dict[str, Any] = {
        "variance": None,
        "data": "",
        "models": [],
        "models_not_fitted": {},
        "turns_unchecked": "",
        "n_genes": 0,
        "left_out": {},
        "reason": "",
    }
    if fit.status == INFERENTIAL and fit.result is None:
        record["reason"] = f"its fit stopped: {fit.reason}"
        return None, record
    if fit.status not in (INFERENTIAL, DESCRIPTIVE):
        record["reason"] = f"the cell type was {fit.status}: {fit.reason}"
        return None, record
    if fit.status == DESCRIPTIVE and design.terms:
        record["reason"] = (
            "a cell type that supports only description is fitted under constant "
            "variance, against dose alone, and the design has other terms, "
            f"{quoted(list(design.terms))}"
        )
        return None, record
    found = _curve_data(
        data, design, model, report, fit, axis, variance, analysis, options
    )
    if isinstance(found, str):
        record["reason"] = found
        return None, record
    curve_data, genes, source = found
    reason = _no_model_fits(model_set, curve_data.doses.size)
    if reason:
        doses = dose_text(curve_data.doses, axis.unit)
        record["reason"] = f"{reason}; its doses are {doses}"
        return None, record
    # Scores with a missing value under constant variance, or whose dose groups the
    # samples they have cannot estimate, have no curve.
    complete = _complete(curve_data)
    column = _id_column(genes)
    if not complete.all():
        incomplete = "its data are incomplete: " + (
            "a sample has no value"
            if curve_data.variance == CONSTANT
            else "the samples it has cannot estimate every dose group"
        )
        # Scores fitted as missing, whose residuals were rounding errors, say so.
        untested = fit.record.get("fit", {}).get("scores_not_tested", {})
        record["left_out"] = {
            name: untested.get(name, incomplete)
            for name in genes[column][~complete].tolist()
        }
        if not complete.any():
            record["reason"] = f"no {_NOUNS[column]} has complete data"
            return None, record
        curve_data = _subset(curve_data, complete)
        genes = genes.filter(complete)
    fits = _fit_means(curve_data, model_set, n_threads=options.get("n_threads"))
    if fit.status == DESCRIPTIVE:
        # Without replicates at its doses, a cell type gets no lack-of-fit test (R2).
        fits = replace(
            fits, lack_of_fit_p_value=read_only(np.full(fits.aicc.shape, np.nan))
        )
    record |= {
        "variance": fits.variance,
        "data": source,
        "models": list(fits.models),
        "models_not_fitted": dict(fits.models_not_fitted),
        "turns_unchecked": _unchecked_turns(model_set, curve_data.doses.size),
        "n_genes": int(fits.means.shape[0]),
    }
    kept = [name for name in _ID_COLUMNS if name in genes]
    return (genes.select(kept), fits), record


def _complete(data: _CurveData) -> np.ndarray:
    """The features whose curve data are finite: every mean, the variance that scales
    their covariance where their samples leave residual degrees of freedom (a cell
    type without replicates has none, for every gene) and, under constant variance,
    every within-group sum of squares."""
    complete = np.isfinite(data.means).all(axis=1)
    complete &= np.isfinite(data.covariance_scale) | (data.df == 0)
    if data.within is not None:
        complete &= np.isfinite(data.within).all(axis=1)
    return complete


def _subset(data: _CurveData, rows: np.ndarray) -> _CurveData:
    """The curve data of some features."""
    return replace(
        data,
        means=np.ascontiguousarray(data.means[rows]),
        covariance=np.ascontiguousarray(data.covariance[rows]),
        covariance_scale=np.ascontiguousarray(data.covariance_scale[rows]),
        df=np.ascontiguousarray(data.df[rows]),
        within=None if data.within is None else np.ascontiguousarray(data.within[rows]),
    )


def _curve_data(
    data: Cube | SampleScores,
    design: Design,
    model: Design | None,
    report: DoseDesignReport,
    fit: "_CellTypeFit",
    axis: DoseAxis,
    variance: str,
    analysis: Callable[..., DifferentialExpressionResult],
    options: Mapping[str, Any],
) -> tuple[_CurveData, Table, str] | str:
    """The data that a cell type's curves are fitted to, its genes and where the data
    come from: the values of its fit under constant variance, the dose-group
    estimates of its fit otherwise, or for a cell type that supports only
    description the values a fit would compute, or the scores of its samples; or
    why it has none."""
    if fit.status == INFERENTIAL:
        assert fit.stage is not None and fit.comparison is not None
        found = fit.stage
        if variance == CONSTANT:
            return (
                _constant_variance_data(found.doses, found.values),
                found.features,
                "each gene's values in the cell type's fit, against the samples' doses",
            )
        comparison = fit.comparison
        estimates = _dose_group_estimates(
            found.moderated, [VEHICLE_MEAN, *comparison.indicators]
        )
        moderated = found.moderated
        weighted = variance == "weighted"
        curve_data = _means_of_estimates(
            estimates.theta,
            estimates.covariance,
            comparison.doses,
            comparison.samples,
            moderated.fit.sigma**2 if weighted else estimates.posterior_variance,
            moderated.fit.df_residual + (0.0 if weighted else moderated.df_prior),
            variance,
        )
        return (
            curve_data,
            found.features,
            "the dose-group means of the cell type's fit and their covariance, "
            f"with the {'residual' if weighted else 'moderated'} variance",
        )
    if isinstance(data, SampleScores):
        # A cell type of scores that supports only description: the scores of its
        # samples, under constant variance.
        _, held, doses = _used_scores(data, report.samples.filter(group_id=fit.group))
        return (
            _constant_variance_data(doses, held),
            Table({data.id_column: data.ids}),
            "each score of the samples, without lack-of-fit tests",
        )
    assert model is not None  # built when a cell type supports description
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(analysis).parameters.items()
        if name in _VALUE_OPTIONS
    }
    values = _descriptive_values(
        data,
        model,
        report.samples.filter(group_id=fit.group),
        group_positions(data, fit.group)[0],
        _analysis_name(analysis, design),
        options,
        defaults,
    )
    if isinstance(values, str):
        return values
    doses, matrix, rows = values
    return (
        _constant_variance_data(doses, matrix),
        data.features.take(rows),
        "each gene's values, as a fit would compute them, against the samples' "
        "doses, without lack-of-fit tests",
    )


def _each_dose_table(tables: list[Table], doses: FloatArray) -> Table:
    """The table of each treated dose against the vehicle: the tables of the
    indicators of the treated doses, stacked in dose order, with each row's dose
    after the feature's columns."""
    first = tables[0]
    genes = [name for name in _ID_COLUMNS if name in first]
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


# The data that the curves of each cell type are fitted to.

VEHICLE_MEAN = "vehicle_mean"
"""The contrast of the vehicle's mean that a cell type's fit adds for curves of
weighted or moderated variance."""


def _vehicle_mean_weights(
    model: Design, sample_ids: Sequence[str], indicators: Sequence[str]
) -> FloatArray:
    """The contrast of the vehicle's mean at the samples' average of the design's
    other terms, on the columns of the model of the dose groups: 1 on the intercept,
    the average of each other term's column over the cell type's samples, and 0 on
    the indicators of the treated doses. Under ``"~ 1"`` it is the intercept."""
    row_of = {sample: row for row, sample in enumerate(model.samples["sample_id"])}
    rows = np.array([row_of[sample] for sample in sample_ids], dtype=np.intp)
    averages = model.matrix[rows].mean(axis=0)
    columns = list(model.columns)
    weights = np.where(
        [column not in indicators for column in columns], averages, 0.0
    ).astype(np.float64)
    return weights


def _means_of_estimates(
    estimates: FloatArray,
    covariance: FloatArray,
    doses: FloatArray,
    n_samples: FloatArray,
    scale: FloatArray,
    df: FloatArray,
    variance: str,
) -> _CurveData:
    """The dose-group means and their unscaled covariance from the vehicle's mean
    and the doses' differences from it, ``(n_genes, K)``, with their covariance."""
    k = doses.size
    to_means = np.eye(k)
    to_means[:, 0] = 1.0
    means = estimates @ to_means.T
    unscaled = np.einsum("ab,gbc,dc->gad", to_means, covariance, to_means)
    return _CurveData(
        variance,
        doses,
        n_samples,
        np.ascontiguousarray(means),
        np.ascontiguousarray(unscaled),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(df, dtype=np.float64),
        None,
    )


def _descriptive_values(
    cube: Cube,
    model: Design,
    samples: Table,
    g: int,
    method: str,
    options: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[FloatArray, FloatArray, np.ndarray] | str:
    """The doses and values of a cell type that supports only description, samples x
    genes, with the cube rows of the genes; or why there are none.

    ``samples`` are the cell type's rows of the check's sample table, and
    ``defaults`` the analysis's defaults of the options that change values."""
    used = np.asarray(samples["used"], dtype=bool)
    sample_ids = samples["sample_id"][used].tolist()
    position = {sample: i for i, sample in enumerate(cube.samples["sample_id"])}
    cube_rows = np.array([position[sample] for sample in sample_ids], dtype=np.intp)
    order = np.argsort(cube_rows, kind="stable")
    cube_rows, sample_ids = cube_rows[order], [sample_ids[i] for i in order]
    counts = np.asarray(cube.layer(COUNTS_LAYER, g)[cube_rows], dtype=np.float64)
    row_of = {sample: row for row, sample in enumerate(model.samples["sample_id"])}
    design_rows = np.array([row_of[sample] for sample in sample_ids], dtype=np.intp)
    x = model.matrix[design_rows]
    x_kept = np.ascontiguousarray(x[:, list(column_space(x).kept)])
    settings = {name: options.get(name, value) for name, value in defaults.items()}
    keep = expressed_features(
        counts,
        x_kept,
        model.samples,
        design_rows,
        filter_setting(settings["expression_filter"]),
    )
    if not keep.any():
        return "no gene passed the expression filter"
    y = np.ascontiguousarray(counts[:, keep])
    if not np.all(library_sizes(y) > 0.0):
        return "a sample has no counts in the genes that passed the expression filter"
    factors = tmm_factors(y) if settings["normalization"] == "tmm" else None
    if method == "limma_voom":
        values = voom_log_cpm(y, effective_library_sizes(y, None, factors))
    else:
        values = log2_cpm(y, norm_factors=factors, prior_count=settings["prior_count"])
    doses = np.asarray(samples["dose"], dtype=np.float64)[used][order]
    return doses, np.asarray(values), np.flatnonzero(keep)


# The result.


def _result(
    data: Cube | PathwayScores | Activities,
    design: Design,
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
    gate: dict[str, Any],
    curve_parts: tuple[
        str | None, str, dict[str, tuple[Table, DoseCurveFits]], dict[str, Any]
    ],
    direction: str | None,
    benchmark_parts: tuple[
        list[BenchmarkResponse] | None,
        tuple[str, ...] | None,
        dict[str, BenchmarkDoses],
    ],
) -> DoseResponseResult:
    model_set, variance, curves, curve_records = curve_parts
    responses, flag_sets, doses_of = benchmark_parts
    check = report.settings
    analysis = next((fit.record["analysis"] for fit in fits if fit.record), None)
    cube: dict[str, Any] | None
    if isinstance(data, Cube):
        cube, scores, layer = cube_record(data), None, COUNTS_LAYER
    else:
        found = data.provenance
        cube, layer = (None if found is None else found["cube"]), None
        names = (
            list(data.gene_sets) if isinstance(data, PathwayScores) else data.sources
        )
        scores = {
            "type": type(data).__name__,
            "method": data.method,
            "settings": dict(data.settings),
            "names": list(names),
        }
    model_record = None
    if model is not None:
        columns = model.columns
        record = (mixed_design_record if method == "dream_voom" else design_record)(
            model, Contrasts(np.empty((len(columns), 0)), (), columns)
        )
        model_record = {k: value for k, value in record.items() if k != "contrasts"}
    settings = {
        "dose": check["dose"],
        "spline_basis": None if splines.basis is None else splines.basis.to_dict(),
        "tests": given,
        "spline_df": spline_df,
        "candidates": None if shapes is None else shapes.to_dict(),
        "seed": seed,
        "gate": gate,
        "direction": direction,
        "models": model_set,
        "variance": variance,
        "bmr": None if responses is None else [bmr.to_dict() for bmr in responses],
        "flags": None if flag_sets is None else list(flag_sets),
        "analysis": method,
        "batch": check["batch"],
        "groups": check["groups"],
        "min_cells": check["min_cells"],
        "min_counts": check["min_counts"],
    }
    provenance = {
        "method": method,
        "versions": software_versions(),
        "cube": cube,
        "scores": scores,
        "layer": layer,
        "design": model_record,
        "settings": settings,
        "analysis_settings": analysis,
        "gate_adjustment": _gate_adjustment(gate),
        "tests": [*splines.tests, *named],
        "curves": None if model_set is None else fitter_settings(model_set),
        "benchmark_doses": None if responses is None else search_settings(),
        "groups": [
            _group_record(fit)
            | {
                "curves": curve_records.get(fit.group),
                "benchmark_doses": _benchmark_record(doses_of.get(fit.group)),
            }
            for fit in fits
        ],
    }
    tables = {
        (fit.group, name): table for fit in fits for name, table in fit.tables.items()
    }
    kept = {
        fit.group: fit.result
        if isinstance(fit.result, ModeratedFit)
        else _without_tables(fit.result)
        for fit in fits
        if fit.result is not None
    }
    return DoseResponseResult(
        _summary(fits, report),
        tables,
        kept,
        report,
        provenance,
        curves=curves,
        benchmark_doses=doses_of,
        cube=data if isinstance(data, Cube) else None,
        design=design if isinstance(data, Cube) else None,
    )


def _restore_result(
    summary: Table,
    tables: Mapping[tuple[str, str], Table],
    fits: Mapping[str, DifferentialExpressionResult | ModeratedFit],
    design_report: DoseDesignReport,
    provenance: Mapping[str, Any],
    curves: Mapping[str, tuple[Table, DoseCurveFits]],
    benchmark_doses: Mapping[str, BenchmarkDoses],
    cube: Cube | None,
    design: Design | None,
) -> DoseResponseResult:
    return DoseResponseResult(
        summary,
        tables,
        fits,
        design_report,
        provenance,
        curves=curves,
        benchmark_doses=benchmark_doses,
        cube=cube,
        design=design,
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
        group, contrasts = fit.record["fit"], fit.record["contrasts"]
        trend = {LINEAR_TREND: 1} if LINEAR_TREND in fit.tables else {}
        df1 = {**group["f_test_df1"], **trend}
        entry["f_test_df1"] = {test: df1[test] for test in TESTS if test in df1}
        entry["contrasts"] = contrasts
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
            [
                np.nan if fit.result is None else fit.record["summary"][column]
                for fit in fits
            ],
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
                    0
                    if fit.result is None
                    else int(fit.record["summary"]["n_features_tested"])
                    for fit in fits
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
            "n_genes_passing_gate": np.array(
                [
                    float(np.count_nonzero(fit.tables[GATE]["passes_gate"]))
                    if GATE in fit.tables
                    else np.nan
                    for fit in fits
                ],
                dtype=np.float64,
            ),
        }
    )


# Curves.


def _dose_group_estimates(moderated: Any, names: Sequence[str]) -> _Estimates:
    """The estimates of a cell type's dose groups against the vehicle, gene by gene,
    from the moderated tests of its fit.

    ``names`` are the contrasts of the dose groups, as the fit names them. Their
    unscaled covariance is each gene's own under voom's weights or for a score with
    missing values, the one shared by every gene without weights or with sample
    weights, and, with limma's standard errors (``contrast_standard_errors="limma"``),
    each gene's unscaled standard errors with the correlation of the design without
    weights, as limma's ``contrasts.fit`` combines them. Under dream, each gene's
    covariance of the contrasts over the variance it moderates (dream's sigma^2), so
    that times the posterior variance it is the covariance of the moderated t-tests.
    """
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


def _model_curve_doses(
    at: Any, axis: DoseAxis, doses: npt.NDArray[np.float64], group: str
) -> npt.NDArray[np.float64]:
    """The doses of a model's curve, checked to lie from 0 to the highest dose the
    cell type compares: every model is fitted with the vehicle at dose 0, whether or
    not the vehicle sits on the axis."""
    values = checked_doses(np.atleast_1d(np.asarray(at)), "at")
    high = float(doses[-1])
    beyond = values > high * (1.0 + SAME_DOSE)
    if beyond.any():
        raise ValueError(
            f"at holds {dose_text(np.unique(values[beyond]), axis.unit)}, beyond the "
            f"doses that {group!r} compares, 0 to {high:g} {axis.unit}; curve() does "
            "not extrapolate"
        )
    return values


def _gene_rows(
    genes: Any, features: Table, group: str, verb: str = "tested"
) -> npt.NDArray[np.intp]:
    """The rows of the genes asked for among those tested, or fitted with curves
    (``verb``), by feature id or, when the features have one, symbol, in the order
    given; for scores, the gene sets or sources by name."""
    column = _id_column(features)
    noun = _NOUNS[column]
    ids = features[column].tolist()
    if genes is None:
        return np.arange(len(ids))
    one, many = (
        ("feature id or symbol", "feature ids or symbols")
        if column == "feature_id"
        else (f"{noun}'s name", f"names of {noun}s")
    )
    named = [genes] if isinstance(genes, str) else genes
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            f"genes must be a {one}, or a list of them, or None for every {noun} "
            f"tested, got {type(genes).__name__}"
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
                f"genes must be {many} (str), got {type(gene).__name__} {gene!r}"
            )
        found = [by_id[gene]] if gene in by_id else by_symbol.get(gene, [])
        if len(found) > 1:
            raise ValueError(
                f"the symbol {gene!r} names {len(found)} genes {verb} in {group!r}, "
                f"{quoted([ids[row] for row in found])}; give their feature ids"
            )
        if not found:
            if column == "feature_id":
                kind = "feature id or symbol" if "symbol" in features else "feature id"
                hint = (
                    f": no gene {verb} there has this {kind}; it is not in the cube, or "
                    "did not pass the expression filter there"
                )
            else:
                hint = (
                    f"; the {noun}s {verb} there are {quoted(ids)}"
                    f"{close_matches(gene, ids)}"
                )
            raise KeyError(f"{noun} {gene!r} was not {verb} in {group!r}{hint}")
        rows.append(found[0])
    return np.array(rows, dtype=np.intp)


def _id_column(table: Table) -> str:
    """The column that names a table's rows: feature_id, gene_set or source."""
    return next(name for name in _NOUNS if name in table)


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


def gate_text(gate: Mapping[str, Any]) -> str:
    """The gate in words, as printouts give it: its test, its adjustments for
    multiple testing and its level."""
    if gate["fold_change"] is None:
        test = f"{gate['test']} test"
    else:
        test = f"TREAT beyond {gate['fold_change']:g}-fold at each dose"
    return f"{test}, {_gate_adjustment(gate)}, at {100.0 * gate['fdr']:g}% FDR"


def _gate_adjustment(gate: Mapping[str, Any]) -> str:
    """The gate's adjustments for multiple testing, in words."""
    within = "Benjamini-Hochberg within each cell type"
    if gate["fold_change"] is None:
        return within
    return f"Bonferroni over the cell type's doses, {within}"


def _slope_unit(axis: DoseAxis) -> str:
    """The unit of the linear trend's slope, for the printout: per unit of the axis's
    position."""
    if axis.scale == "linear":
        return f"per {axis.unit}"
    if axis.offset is None:
        return "per log10 unit of dose"
    return f"per log10 unit of dose plus {axis.offset:g} {axis.unit}"
