"""Differential expression between samples, one group (cell type) of a cube at a time.

:func:`limma_voom` and :func:`limma_trend` run the limma workflows for RNA-seq counts
(Ritchie et al. 2015) on the pseudobulk counts of each group of a cube: limma-voom,
which gives every observation a precision weight from the mean-variance trend of the
log counts per million (Law et al. 2014), and limma-trend, which lets the prior
variance of the empirical Bayes moderation follow that trend instead.
:func:`dream_voom` runs variancePartition's dream (Hoffman and Roussos 2021),
limma-voom with a linear mixed model in place of the linear model, for designs with
random effects such as several samples of each donor. Each tests the named contrasts
of a :class:`~cytepool.design.Design` and returns, for every group and contrast, a
table of the tested features with ``topTable``'s statistics, together with a summary,
the samples behind every fit and a provenance record.

Samples:
    The design's sample table must have a ``sample_id`` column. Its samples are matched
    to the cube's by ``sample_id``, never by position, and they may be a subset of the
    cube's samples; the order of the design's rows does not matter, as every group is
    fitted with its samples in the cube's order. In each group, a sample is left out
    of the fit when it has fewer than ``min_cells`` cells (when the cube records cell
    numbers), no counts, or fewer than ``min_counts`` counts in all (its total over
    every feature of the cube), and the result records why. ``min_counts`` is 0 by
    default; the scverse single-cell best practices (Heumos et al. 2023) keep samples
    with at least 10 cells and 1,000 counts (``min_cells=10, min_counts=1000``), as
    decoupler's ``filter_samples`` does, which keeps a sample with exactly as many
    (found by probing).

The design of a group:
    The design matrix of a group is the rows of the full design matrix for the
    samples left in it, so every column keeps its meaning: factor levels, reference
    levels and covariate values are those of the full design. With fewer samples that
    matrix can be rank deficient, for example when every sample of one donor is left
    out. Such a group is fitted on the column space of its design matrix, keeping the
    columns that R's ``qr()`` keeps, as limma's ``lmFit`` does; each other column
    (aliased) equals a combination of the kept columns before it. A contrast ``c`` is
    estimable from the group's samples when it is orthogonal to the null space of the
    group's design matrix (Searle 1971): when its weight on every aliased column is the
    sum of its weights on the kept columns, weighted by that column's combination.
    This is the rule of :meth:`Design.check <cytepool.design.Design.check>`. ``c``
    applied to the coefficients then equals its weights on the kept columns applied to
    the fit on the kept columns, whatever the coefficients that the samples cannot
    separate, so its estimate and its exact standard error are those of any full-rank
    parametrisation of the same model. A contrast that is not estimable is reported,
    with the relations it violates, and not computed.

Groups that are not fitted:
    A group is not fitted, and its status in the summary says why, when no sample is
    left, the design leaves no residual degrees of freedom, no contrast is estimable,
    for dream the random effects cannot be estimated from the group's samples, no
    feature passes the expression filter (or, for voom, fewer than two), a sample has
    no counts in the features that pass, or voom, the sample quality weights or the
    moderation fail on its data. A run never stops for one group.

Steps:
    For each group, with the R functions they match in the limma and edgeR User's
    Guides:

    1. Leave out samples, as above; the design matrix is the rows of the rest.
    2. Filter features by expression with the group's design, by edgeR's
       ``filterByExpr(y, design)`` (Chen, Lun and Smyth 2016) with its defaults
       (``expression_filter``), and recompute the library sizes over the features
       kept (``y[keep, , keep.lib.sizes = FALSE]``).
    3. Normalization factors by TMM (Robinson and Oshlack 2010), edgeR's
       ``calcNormFactors`` (``normalization``).
    4. limma-voom: ``voom`` on the library sizes times the normalization factors,
       with its LOWESS span chosen from the number of features tested, as limma's
       ``voom`` chooses it by default since limma 3.66 (``span``), then ``lmFit``
       with its weights. limma-trend: log2 counts per million with a prior count of
       3, ``cpm(y, log = TRUE, prior.count = 3)`` as the limma User's Guide uses for
       limma-trend, then ``lmFit``. Sample weights, when asked, enter here (below).
    5. ``contrasts.fit``, ``eBayes`` (Smyth 2004), with ``trend = TRUE`` for
       limma-trend and ``robust`` (Phipson et al. 2016) when asked, and
       ``topTable(sort.by = "none")``: the Benjamini-Hochberg adjustment (Benjamini
       and Hochberg 1995) runs over the tested features of one group and contrast,
       and with ``adjust_across_groups`` also over every group (below).

Sample weights:
    Both kinds are off by default, and they are alternatives: limma's
    ``voomWithQualityWeights`` takes no prior weights, and edgeR's ``voomLmFit``
    refuses prior weights together with estimated sample weights (found by probing),
    so CytePool refuses the two together too. The sample table and the record of
    each fitted group hold the sample weights of its fit. With limma-voom, the
    quality weights recorded are the final ones; ``voomWithQualityWeights`` estimates
    them twice, and its first estimate only weights the fit of voom's second trend.
    :func:`~cytepool.stats.voom.voom_with_quality_weights`, given the group's counts,
    design rows and normalization factors, reproduces both.

    - ``quality_weights``: sample quality weights, estimated in each group from how
      closely each sample's values follow the linear model (Ritchie et al. 2006; Liu
      et al. 2015), which a limma author recommends for pseudobulk data (Smyth 2025,
      on the Bioconductor support site). limma-voom runs limma's
      ``voomWithQualityWeights``
      (:func:`~cytepool.stats.voom.voom_with_quality_weights`), gene by gene by
      default, as limma does. limma-trend estimates them from the log2 counts per
      million as limma's ``arrayWeights`` does
      (:func:`~cytepool.stats.sample_weights.sample_quality_weights`), by REML by
      default, the method that ``arrayWeights`` chooses for such data, and passes
      them to ``lmFit`` as sample weights. Where a donor has two samples, as in
      ``~ donor + condition``, their residuals are equal and opposite (up to voom's
      weights), so both get the low weight of an outlying sample. Where the weights
      cannot be estimated, as with fewer than two residual degrees of freedom, they
      are one, as limma's are, with a warning that names the group and that its
      record keeps; where the estimate fails, as when a sample carries no
      information about its variance and ``prior_n`` is 0, the group is not fitted.
    - ``prior_weights="n_cells"``: each sample's number of cells in the group, as
      prior weights. A pseudobulk profile sums the counts of its cells, so its log2
      counts per million estimate the mean expression of those cells, whose
      variance falls as one over their number: the precision of a profile is
      proportional to its number of cells, as the dreamlet paper argues (Hoffman et
      al. 2026, Methods), and dreamlet's ``processAssays`` weights its fits by the
      number of cells of each sample, as its documentation describes.
      limma-voom passes them to ``voom`` as ``weights``: they weight the fits
      behind voom's trend, and the precision weights are the trend's times them,
      as limma's help page says voom's weights modify prior weights, and as edgeR's
      ``voomLmFit(prior.weights = ...)`` forms them (found by probing). limma-trend
      passes them to ``lmFit`` as sample weights. Only their ratios within a group
      matter: weights that are all ten times larger give the same results.

dream:
    :func:`dream_voom` takes the steps of limma-voom up to the normalization, then
    those of variancePartition's ``voomWithDreamWeights``, ``dream(ddf =
    "Satterthwaite")``, ``eBayes`` and ``topTable``: precision weights from the
    mean-variance trend of the design's mixed model, fitted to every feature by
    maximum likelihood (:func:`~cytepool.stats.dream.voom_with_dream_weights`); the
    mixed model of every feature fitted by REML with those weights
    (:func:`~cytepool.stats.mixed_models.fit_mixed_model`); Satterthwaite's (1946)
    t-tests of the estimable contrasts
    (:func:`~cytepool.stats.mixed_models.fixed_effect_tests`);
    and their empirical Bayes moderation
    (:func:`~cytepool.stats.dream.moderated_fixed_effect_tests`). The random effects of
    a group are those of its samples: a grouping factor's levels without samples in
    the group drop out. :mod:`cytepool.stats.dream` documents the conventions found by
    probing variancePartition and the deliberate differences from it. Prior weights
    from the number of cells are divided by their mean in each group, as
    variancePartition divides prior weights, which changes no test.

P-values adjusted across groups:
    ``adjusted_p_value`` adjusts the p-values of one group and contrast, as limma's
    ``topTable`` does. With ``adjust_across_groups``, the column
    ``adjusted_p_value_across_groups`` adjusts each contrast's p-values by
    Benjamini-Hochberg over every test of that contrast in the analysis: every
    feature tested in every group that was fitted and can estimate the contrast,
    stacked into one family. Each contrast is a family of its own, and groups left
    out of ``groups`` are not in it. muscat reports both adjustments (Crowell et
    al. 2020), whose comparison found the adjustment over all groups very
    conservative. It is not larger for every test: a family with more small
    p-values can lower an adjusted value.

Deliberate differences from limma:
    ``validation/differential_expression.py`` reports each of them.

    - In a rank-deficient group, a contrast that is estimable but weighs an aliased
      column is computed. limma's ``contrasts.fit`` stops with an error for such a
      contrast, and with it for every contrast passed together with it (found by
      probing), so in limma it needs a full-rank parametrisation.
    - A contrast that is not estimable, but weighs no aliased column, is reported as
      not estimable. limma computes it from the kept columns, whose meaning has then
      changed: with every sample of the reference donor D1 left out of
      ``~ donor + condition``, limma reports for the coefficient ``donorD2`` (donor D2
      against D1) the difference between donor D2 and the last donor.
    - With ``contrast_standard_errors="exact"``, the default of
      :func:`limma_voom`, the standard errors of contrasts that combine several
      coefficients are exact; limma's ``contrasts.fit`` approximates them when
      weights differ between features, and ``"limma"`` reproduces it. limma's
      approximation, unlike the exact value, depends on the parametrisation, so only
      exact standard errors are the same in every parametrisation of a group's
      design.

Memory:
    One group is in memory at a time: its counts layer, read from the cube when the
    group is fitted, and the fit of its features. A cube opened with
    :func:`~cytepool.io.read_cube` therefore stays on disk except for the group
    being fitted. Results hold one table per group and contrast, and a record that
    identifies the cube by its fingerprint, which costs no read for a cube file
    ("Provenance" in :func:`limma_voom`).

References:
    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK (2015). limma powers
    differential expression analyses for RNA-sequencing and microarray studies. Nucleic
    Acids Research 43, e47. doi:10.1093/nar/gkv007

    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    Searle SR (1971). Linear Models. Wiley, New York.

    Ritchie ME, Diyagama D, Neilson J, van Laar R, Dobrovic A, Holloway A, Smyth GK
    (2006). Empirical array quality weights in the analysis of microarray data. BMC
    Bioinformatics 7, 261. doi:10.1186/1471-2105-7-261

    Liu R, Holik AZ, Su S, Jansz N, Chen K, Leong HS, Blewitt ME, Asselin-Labat ML,
    Smyth GK, Ritchie ME (2015). Why weight? Modelling sample and observational level
    variability improves power in RNA-seq analyses. Nucleic Acids Research 43, e97.
    doi:10.1093/nar/gkv412

    Hoffman GE, Lee D, Bendl J, Prashant NM, Hong A, Casey C, Alvia M, Shao Z,
    Argyriou S, Therrien K, Venkatesh S, Voloudakis G, Haroutunian V, Fullard JF,
    Roussos P (2026). Efficient differential expression analysis of large-scale
    single-cell transcriptomics data using Dreamlet. Nature Communications 17, 9597.
    doi:10.1038/s41467-026-75680-8

    Crowell HL, Soneson C, Germain PL, Calini D, Collin L, Raposo C, Malhotra D,
    Robinson MD (2020). muscat detects subpopulation-specific state transitions from
    multi-sample multi-condition single-cell transcriptomics data. Nature Communications
    11, 6077. doi:10.1038/s41467-020-19894-4

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Robinson MD, Oshlack A (2010). A scaling normalization method for differential
    expression analysis of RNA-seq data. Genome Biology 11, R25.
    doi:10.1186/gb-2010-11-3-r25

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Satterthwaite FE (1946). An approximate distribution of estimates of variance
    components. Biometrics Bulletin 2, 110-114. doi:10.2307/3002019

    Heumos L, Schaar AC, Lance C, Litinetskaya A, Drost F, Zappia L, Luecken MD,
    Strobl DC, Henao J, Curion F, Single-cell Best Practices Consortium, Schiller HB,
    Theis FJ (2023). Best practices for single-cell analysis across modalities. Nature
    Reviews Genetics 24, 550-572. doi:10.1038/s41576-023-00586-w

    Smyth GK (2025). Answer to "Guidance on limma for pseudobulk scRNA-seq with
    multi-factor design", 11 September 2025. Bioconductor support site,
    https://support.bioconductor.org/p/9162569/
"""

import json
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._analysis_records import (
    cube_record,
    design_record,
    matched_samples,
    software_versions,
)
from ._columns import text_column
from ._counts import effective_library_sizes
from ._jit import thread_limit
from ._masked_arrays import reject_masked
from ._messages import PACKAGE_DIRECTORY, listing
from ._options import (
    checked_choice,
    checked_flag,
    index_integer,
    non_negative_number,
    number_at_least,
    positive_finite_number,
    positive_integer,
    real_number,
)
from .cube import Cube
from .design import Contrasts, Design, make_contrasts
from .design._design_matrices import is_random_intercept, refuse_random_effects
from .design.checks import _aliasing, _contrast_violations
from .normalization import (
    filter_by_expression,
    library_sizes,
    log_cpm,
    tmm_factors,
)
from .stats.dream import (
    _dream_weights,
    _scaled_prior_weights,
    moderated_fixed_effect_tests,
)
from .stats.empirical_bayes import moderated_statistics
from .stats.linear_models import contrasts_fit, fit_linear_model
from .stats.mixed_models import _fit as _fit_mixed_model
from .stats.mixed_models import (
    _mixed_design_of_samples,
    _MixedDesign,
    _random_intercepts,
    fixed_effect_tests,
)
from .stats.multiple_testing import adjust_p_values
from .stats.sample_weights import (
    _estimate,
    _SingularInformation,
    _variance_model,
    _weights_of,
)
from .stats.voom import (
    VoomResult,
    _quality_weighted_voom,
    _trend_span,
    _voom_inputs,
    voom,
)
from .table import Table

__all__ = ["DifferentialExpressionResult", "dream_voom", "limma_trend", "limma_voom"]

_FITTED = "fitted"  # the status of a group whose contrasts were tested
_LAYER = "counts"
_ACROSS_GROUPS = "adjusted_p_value_across_groups"
# edgeR's filterByExpr defaults.
_FILTER_DEFAULTS = {
    "min_count": 10.0,
    "min_total_count": 15.0,
    "large_n": 10.0,
    "min_prop": 0.7,
}
# The settings of sample quality weights: limma's defaults (arrayWeights and
# voomWithQualityWeights), and the method by default. voomWithQualityWeights
# estimates the weights gene by gene; arrayWeights' "auto" chooses REML for values
# without prior weights or missing values, such as the log2 CPM of limma-trend.
_QUALITY_DEFAULTS = {"prior_n": 10.0, "max_iterations": 50, "tolerance": 1e-5}
_QUALITY_METHODS = {"limma_voom": "gene_by_gene", "limma_trend": "reml"}
# limma's defaults that the workflow keeps, recorded in the provenance.
_FIXED_SETTINGS = {
    "proportion": 0.01,
    "stdev_coef_lim": [0.1, 4.0],
    "winsor_tail_p": [0.05, 0.1],
    "p_value_adjustment": "BH",
}
# dream's fits, recorded in the provenance: maximum likelihood for the weights, as
# variancePartition's voomWithDreamWeights fits, REML and Satterthwaite's degrees of
# freedom for the tests (dream's REML = TRUE, ddf = "Satterthwaite").
_DREAM_SETTINGS = {
    "weights_fit": "ml",
    "fit": "reml",
    "df_method": "satterthwaite",
}
_METHODS = ("limma_voom", "limma_trend", "dream_voom")


class DifferentialExpressionResult:
    """Differential expression of every group of a cube, from limma or dream.

    Tables hold, for every group and estimable contrast, one row per tested feature,
    in the cube's feature order (``topTable(sort.by = "none")``):

    - ``feature_id`` and, when the cube's feature table has one, ``symbol``;
    - ``log2_fold_change``: the contrast's estimate (limma's ``logFC``);
    - ``standard_error``: its moderated standard error, the contrast's unscaled
      standard deviation times the feature's posterior standard deviation (limma's
      ``stdev.unscaled * sqrt(s2.post)``), so that ``t`` is
      ``log2_fold_change / standard_error``; it is what a meta-analysis of studies
      reads (:func:`~cytepool.meta_analysis.differential_expression_meta_analysis`);
    - ``average_log2_cpm``: the mean log2 count per million over the group's samples
      (``AveExpr``);
    - ``t`` and ``p_value``: the moderated t-statistic and its two-sided p-value;
    - ``adjusted_p_value``: Benjamini-Hochberg over the group's tested features
      (``adj.P.Val``);
    - ``adjusted_p_value_across_groups``, with ``adjust_across_groups`` only:
      Benjamini-Hochberg over the tested features of every group that tests the
      contrast;
    - ``b_statistic``: the log-odds that the feature is differentially expressed
      (``B``).

    Tables from :func:`dream_voom` also give ``df``, the degrees of freedom of each
    test (Satterthwaite's plus the prior's, ``df.total``), after ``t``, and
    ``z_statistic``, the z-statistic with the test's p-value (``topTable``'s
    ``z.std``), after ``b_statistic``; a feature whose mixed model has no minimum has
    NaN statistics.

    The summary has one row per group: ``group_id``; ``status``, ``"fitted"`` or why
    the group was not fitted (``"no_samples"``, ``"no_residual_df"``,
    ``"no_estimable_contrast"``, ``"random_effects_not_estimable"``,
    ``"no_features"``, ``"too_few_features"``, ``"empty_library"`` or
    ``"failed"``); ``reason``, the explanation (empty when fitted);
    ``n_samples_used`` and ``n_samples_left_out``; ``n_features_tested``;
    ``df_residual``, for dream the median over the features of their approximate
    residual degrees of freedom (in a group not fitted, the residual degrees of
    freedom of the fixed effects, as for limma); ``prior_df``, the prior degrees of freedom (their
    median over features with robust moderation, where each feature has its own); and
    ``not_estimable``, the contrasts the group's samples cannot estimate, separated
    by ``", "``. Numbers are NaN, or 0, where a group was not fitted that far.

    The sample table has one row per group and design sample, in the cube's sample
    order: ``group_id``, ``sample_id``, ``n_cells`` (when the cube records cell
    numbers), ``used`` (in the group's design), ``reason`` (why a sample was left out;
    empty when used), and the ``library_size`` (over the tested features) and
    ``norm_factor`` that entered the fit; with ``prior_weights``, the
    ``prior_weight`` of each sample, and with ``quality_weights``, its
    ``quality_weight``. Numbers that did not enter a fit are NaN: those of samples
    left out, and of groups not fitted.

    Built by :func:`limma_voom`, :func:`limma_trend` and :func:`dream_voom`, from the
    contrasts, tables and record they computed. Immutable.

    Raises:
        TypeError: If ``method`` is not a string, ``contrasts`` not a ``Contrasts``,
            ``summary`` or ``samples`` not a ``Table``, ``tables`` not a mapping of
            (group id, contrast name) to ``Table``, or ``provenance`` not a mapping
            that JSON can write.
        ValueError: If ``method`` is not ``"limma_voom"``, ``"limma_trend"`` or
            ``"dream_voom"``, or ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_method", "_contrasts", "_summary", "_samples", "_tables", "_record")

    _method: str
    _contrasts: Contrasts
    _summary: Table
    _samples: Table
    _tables: Mapping[tuple[str, str], Table]
    _record: str

    def __init__(
        self,
        method: str,
        contrasts: Contrasts,
        summary: Table,
        samples: Table,
        tables: Mapping[tuple[str, str], Table],
        provenance: Mapping[str, Any],
    ) -> None:
        if not isinstance(method, str):
            raise TypeError(f"method must be a string, got {type(method).__name__}")
        if method not in _METHODS:
            raise ValueError(
                "method must be 'limma_voom', 'limma_trend' or 'dream_voom', got "
                f"{method!r}"
            )
        if not isinstance(contrasts, Contrasts):
            raise TypeError(
                f"contrasts must be a Contrasts, got {type(contrasts).__name__}"
            )
        for name, table in (("summary", summary), ("samples", samples)):
            if not isinstance(table, Table):
                raise TypeError(f"{name} must be a Table, got {type(table).__name__}")
        if not isinstance(tables, Mapping):
            raise TypeError(
                f"tables must be a mapping of (group id, contrast name) to Table, got "
                f"{type(tables).__name__}"
            )
        for key, table in tables.items():
            pair = isinstance(key, tuple) and len(key) == 2
            if not (pair and all(isinstance(part, str) for part in key)):
                raise TypeError(
                    "tables must be keyed by (group id, contrast name), got the key "
                    f"{key!r}"
                )
            if not isinstance(table, Table):
                raise TypeError(
                    f"tables[{key!r}] must be a Table, got {type(table).__name__}"
                )
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_method": method,
            "_contrasts": contrasts,
            "_summary": summary,
            "_samples": samples,
            "_tables": MappingProxyType(dict(tables)),
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DifferentialExpressionResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DifferentialExpressionResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            DifferentialExpressionResult,
            (
                self._method,
                self._contrasts,
                self._summary,
                self._samples,
                dict(self._tables),
                json.loads(self._record),
            ),
        )

    @property
    def method(self) -> str:
        """``"limma_voom"``, ``"limma_trend"`` or ``"dream_voom"``."""
        return self._method

    @property
    def groups(self) -> tuple[str, ...]:
        """The ids of the groups analysed, in order."""
        return tuple(self._summary["group_id"].tolist())

    @property
    def contrasts(self) -> Contrasts:
        """The contrasts tested, over the columns of the full design."""
        return self._contrasts

    @property
    def summary(self) -> Table:
        """The status, samples, features and degrees of freedom of every group."""
        return self._summary

    @property
    def samples(self) -> Table:
        """Every design sample of every group, used in its fit or left out, and why."""
        return self._samples

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """The table of every group and contrast that was tested, by (group, contrast)."""
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
        cube (``"cube"``): its fingerprint, the digests it is built from and its own
        provenance; the layer read; the design (formula, columns, levels, the values
        of its variables for every sample, and the weights of every contrast); the
        settings; and, for every group, its status, the samples that entered its
        design, in the order they were fitted, those left out with the reason, the
        design columns its samples cannot separate, the contrasts it cannot
        estimate with the reason, the number of features tested, for limma-voom and
        dream the span of its trend (``voom_span``, null where voom did not run), for
        dream the number of features whose mixed model has no minimum
        (``features_not_fitted``) and of those whose REML fit stopped without
        converging (``features_not_converged``; their tests use the estimates where
        it stopped), the
        sample weights its fit used, in the order of its samples
        (``prior_weights`` with ``prior_weights``, ``quality_weights`` with
        ``quality_weights``; null where the group was not fitted), and the warnings
        that the estimation of its sample quality weights gave (``warnings``, such as
        weights that could not be estimated and are one). Each number of a table
        comes from the profiles of its group and of those samples in the cube with
        that fingerprint: the counts of every feature of these profiles enter its
        filter, normalization and trend. The fingerprint, and the digest of the
        group's counts, say whether a cube at hand holds those profiles
        (``Cube.verify_fingerprint``).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str, contrast: str) -> Table:
        """Return the table of one group and contrast.

        Args:
            group: A group id.
            contrast: A contrast name.

        Raises:
            KeyError: If the group or contrast was not analysed, or the group was
                not fitted or cannot estimate the contrast; the message says why.
        """
        found = self._tables.get((group, contrast))
        if found is not None:
            return found
        record = {entry["group_id"]: entry for entry in self.provenance["groups"]}
        if group not in record:
            raise KeyError(
                f"no group {group!r} was analysed; groups are: "
                f"{listing(list(record)) or '(none)'}"
            )
        if contrast not in self._contrasts.names:
            raise KeyError(
                f"no contrast {contrast!r} was tested; contrasts are: "
                f"{listing(list(self._contrasts.names))}"
            )
        entry = record[group]
        if entry["status"] != _FITTED:
            raise KeyError(
                f"group {group!r} was not fitted ({entry['status']}): {entry['reason']}"
            )
        raise KeyError(
            f"contrast {contrast!r} was not tested in group {group!r}: "
            f"{entry['not_estimable'][contrast]}"
        )

    def __repr__(self) -> str:
        statuses = self._summary["status"].tolist()
        fitted = statuses.count(_FITTED)
        lines = [
            f"DifferentialExpressionResult ({self._method}): {len(statuses)} "
            f"group{'' if len(statuses) == 1 else 's'}, {fitted} fitted; "
            f"{len(self._contrasts)} contrast"
            f"{'' if len(self._contrasts) == 1 else 's'}: "
            + ", ".join(self._contrasts.names),
        ]
        for row in range(len(statuses)):
            group = self._summary["group_id"][row]
            if statuses[row] != _FITTED:
                lines.append(f"  {group}: not fitted ({statuses[row]})")
                continue
            used = int(self._summary["n_samples_used"][row])
            tested = int(self._summary["n_features_tested"][row])
            text = f"  {group}: {used} samples, {tested} features"
            missing = self._summary["not_estimable"][row]
            if missing:
                text += f"; not estimable: {missing}"
            lines.append(text)
        return "\n".join(lines)


def limma_voom(
    cube: Cube,
    design: Design,
    *,
    groups: str | Sequence[str] | None = None,
    contrasts: Mapping[str, Any] | Sequence[str] | str | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    expression_filter: bool | Mapping[str, float] = True,
    normalization: Literal["tmm"] | None = "tmm",
    span: float | None = None,
    prior_weights: Literal["n_cells"] | None = None,
    quality_weights: bool | Mapping[str, Any] | None = False,
    robust: bool = False,
    contrast_standard_errors: Literal["exact", "limma"] = "exact",
    adjust_across_groups: bool = False,
    n_threads: int | None = None,
) -> DifferentialExpressionResult:
    """Test contrasts in every group of a cube by limma-voom.

    For each group, the samples with fewer than ``min_cells`` cells or no counts are
    left out, features are filtered by expression, the counts are normalized, voom
    gives every observation a precision weight from the mean-variance trend of the
    log2 counts per million, a weighted linear model is fitted to every feature, and
    its contrasts are moderated by empirical Bayes: the steps of the limma User's
    Guide for RNA-seq (``voom``, ``lmFit``, ``contrasts.fit``, ``eBayes``,
    ``topTable``) after edgeR's ``filterByExpr`` and ``calcNormFactors``. The module
    documentation describes the steps, rank-deficient groups and the deliberate
    differences from limma.

    voom smooths its mean-variance trend with a LOWESS span that, by default, it
    chooses in each group from the number of features tested there, by limma's
    ``chooseLowessSpan`` rule (:func:`~cytepool.stats.smoothing.choose_lowess_span`),
    as limma's ``voom`` does by default since limma 3.66 (``adaptive.span = TRUE``).
    ``span`` fixes it instead; limma's default before 3.66 was 0.5. The record of
    each group gives the span its trend used, as ``voom_span``.

    Options for pseudobulk data, each off by default so that the results are
    limma's standard workflow unless asked: a minimum total count per sample
    (``min_counts``), sample quality weights (``quality_weights``, limma's
    ``voomWithQualityWeights``), prior weights from the number of cells behind each
    profile (``prior_weights="n_cells"``, as dreamlet weights samples), and p-values
    adjusted across groups as well as within each (``adjust_across_groups``). The
    module documentation gives the rules and their sources.

    Provenance:
        The result records the cube's fingerprint, the digests it is built from and
        the cube's provenance (``Cube.fingerprint``, ``Cube.digests``,
        ``Cube.provenance``), computed after the fits. For a cube opened with
        :func:`~cytepool.io.read_cube` this reads nothing: the file stores the
        digests, and each group the fits read whole is checked against its digest.
        A cube in memory hashes its layers once, at about 1 s per GB of values
        where the processor has SHA instructions. A lazy cube derived from a cube
        file with ``select``, ``regroup`` or ``collapse_samples`` reads, once, the
        groups whose values its operations changed and that the fits did not read
        whole: those of its other layers, and those of groups not analysed. A lazy
        layer of another array-like, which can change, is read whole once more.

    Args:
        cube: A cube with a ``counts`` layer, such as one from
            :func:`~cytepool.cube.from_cells` or :func:`~cytepool.io.read_cube`.
        design: A :class:`~cytepool.design.Design` whose sample table has a
            ``sample_id`` column with cube sample ids; its rows may be some of the
            cube's samples, in any order. A design with random effects is refused:
            these fits have fixed effects only, and :func:`dream_voom` fits random
            intercepts.
        groups: A group id or a list of them; all groups by default.
        contrasts: The contrasts to test, as accepted by
            :func:`~cytepool.design.make_contrasts` and matched to the design's
            columns by name, such as ``{"stim_vs_ctrl": "conditionstim"}``. By
            default, the design's named contrasts.
        min_cells: The fewest cells a sample needs in a group to enter its fit.
            Ignored when the cube does not record cell numbers (bulk data), where
            only samples without counts are left out.
        min_counts: The fewest counts a sample needs in a group, summed over every
            feature of the cube (its library size before the expression filter),
            to enter its fit; at least 0. The default, 0, leaves out only samples
            without counts.
        expression_filter: True for edgeR's ``filterByExpr`` with its defaults
            (``min_count=10``, ``min_total_count=15``, ``large_n=10``,
            ``min_prop=0.7``), using the group's design, as the edgeR and limma
            User's Guides do; a mapping with some of these settings to change them
            (see :func:`~cytepool.normalization.filter_by_expression`); or False to
            test every feature.
        normalization: ``"tmm"`` for edgeR's TMM normalization factors, the
            default of ``calcNormFactors``, or None for library sizes alone.
        span: The LOWESS span of voom's trend, in (0, 1], for every group; None
            chooses it in each group from the number of features tested, as
            :func:`~cytepool.stats.voom.voom` does.
        prior_weights: ``"n_cells"`` to give each sample its number of cells in
            the group as a prior weight, which voom's precision weights multiply,
            or None for none. Needs a cube that records cell numbers.
        quality_weights: True to estimate sample quality weights in each group and
            multiply them into voom's precision weights, as limma's
            ``voomWithQualityWeights`` does, with its settings
            (``method="gene_by_gene"``, ``prior_n=10``, ``max_iterations=50``,
            ``tolerance=1e-5``; see
            :func:`~cytepool.stats.voom.voom_with_quality_weights`); a mapping with
            some of these settings to change them, such as ``{"method": "reml"}``;
            or False (or None) for none. Give at most one of ``prior_weights`` and
            ``quality_weights``.
        robust: Whether to estimate the prior of the moderation robustly, as
            ``eBayes(robust = TRUE)`` does, which protects against hypervariable
            features.
        contrast_standard_errors: ``"exact"`` computes the standard error of every
            contrast from each feature's own weighted fit; ``"limma"`` reproduces
            the approximation of limma's ``contrasts.fit``. They differ only for
            contrasts that combine correlated coefficients (see
            :func:`~cytepool.stats.linear_models.contrasts_fit`).
        adjust_across_groups: Whether to add the column
            ``adjusted_p_value_across_groups`` to every table: each contrast's
            p-values adjusted by Benjamini-Hochberg over every group that tests
            it, besides ``adjusted_p_value``, which adjusts them within the group.
        n_threads: The most threads for the weighted least-squares fits of the
            features, which run in parallel (numba's threads; see
            ``numba.set_num_threads`` and
            :func:`~cytepool.stats.linear_models.fit_linear_model`), at least 1;
            None keeps numba's setting. The results do not depend on it. With 1, no
            parallel region runs, so that forked processes and several Python
            threads can fit at once.

    Returns:
        The tables of every group and estimable contrast, a summary of every group,
        the samples used and left out, the settings and a provenance record.

    Warns:
        RuntimeWarning: If sample quality weights cannot be estimated in a group,
            such as with fewer than two residual degrees of freedom (they are then
            one), or their REML stops without converging; the message names the
            group, and its record keeps it.

    Raises:
        TypeError: If ``cube`` is not a Cube, ``design`` not a Design, an option has
            the wrong type, or ``contrasts``, ``groups``, ``min_cells``,
            ``expression_filter``, ``quality_weights`` or ``n_threads`` is or holds
            a numpy masked array.
        ValueError: If the design has random effects or no ``sample_id`` column,
            names samples the cube does not have or names a sample twice, if there
            are no contrasts to test, an option is invalid, ``prior_weights`` and
            ``quality_weights`` are both given, or ``prior_weights`` is
            ``"n_cells"`` and the cube does not record cell numbers.
        KeyError: If a group is not in the cube.
    """
    standard_errors = checked_choice(
        contrast_standard_errors, "contrast_standard_errors", ("exact", "limma")
    )
    return _analyse(
        "limma_voom",
        cube,
        design,
        groups=groups,
        contrasts=contrasts,
        settings={
            "min_cells": _min_cells(min_cells),
            "min_counts": _min_counts(min_counts),
            "expression_filter": _filter_settings(expression_filter),
            "normalization": _normalization(normalization),
            "span": _span(span),
            **_sample_weight_settings(prior_weights, quality_weights, "limma_voom"),
            "robust": checked_flag(robust, "robust"),
            "contrast_standard_errors": standard_errors,
            "adjust_across_groups": checked_flag(
                adjust_across_groups, "adjust_across_groups"
            ),
            "n_threads": _threads(n_threads),
        },
    )


def limma_trend(
    cube: Cube,
    design: Design,
    *,
    groups: str | Sequence[str] | None = None,
    contrasts: Mapping[str, Any] | Sequence[str] | str | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    expression_filter: bool | Mapping[str, float] = True,
    normalization: Literal["tmm"] | None = "tmm",
    prior_count: float = 3.0,
    prior_weights: Literal["n_cells"] | None = None,
    quality_weights: bool | Mapping[str, Any] | None = False,
    robust: bool = False,
    adjust_across_groups: bool = False,
) -> DifferentialExpressionResult:
    """Test contrasts in every group of a cube by limma-trend.

    As :func:`limma_voom`, but the counts become log2 counts per million with a
    prior count (edgeR's ``cpm(y, log = TRUE, prior.count = 3)``, the limma User's
    Guide's choice for limma-trend), every feature is fitted without observation
    weights, and the prior variance of the empirical Bayes moderation follows the
    mean log2 CPM of the features (``eBayes(trend = TRUE)``). The standard errors of
    contrasts are exact, as limma's are, also with sample weights. The result
    records the cube's fingerprint and digests, at the cost "Provenance" in
    :func:`limma_voom` gives.

    The options for pseudobulk data are those of :func:`limma_voom`; sample weights
    (prior or quality weights) enter ``lmFit`` as sample weights.

    Args:
        cube: A cube with a ``counts`` layer.
        design: A :class:`~cytepool.design.Design` matched to the cube by
            ``sample_id``, as for :func:`limma_voom`.
        groups: A group id or a list of them; all groups by default.
        contrasts: The contrasts to test; by default the design's named contrasts.
        min_cells: The fewest cells a sample needs in a group to enter its fit;
            ignored for cubes without cell numbers.
        min_counts: The fewest counts a sample needs in a group, over every
            feature of the cube, to enter its fit; 0 by default.
        expression_filter: True for edgeR's ``filterByExpr`` defaults with the
            group's design, a mapping of changed settings, or False.
        normalization: ``"tmm"`` for TMM normalization factors, or None.
        prior_count: The average count added to every observation before taking
            logs, scaled by library size as edgeR does; positive. 3 by default, as
            in the limma User's Guide.
        prior_weights: ``"n_cells"`` to weight each sample by its number of cells
            in the group, or None for none. Needs a cube that records cell
            numbers.
        quality_weights: True to estimate sample quality weights from the log2
            counts per million of each group, as limma's ``arrayWeights`` does,
            with its settings for such data (``method="reml"``, ``prior_n=10``,
            ``max_iterations=50``, ``tolerance=1e-5``; see
            :func:`~cytepool.stats.sample_weights.sample_quality_weights`); a
            mapping with some of these settings to change them, such as
            ``{"method": "gene_by_gene"}``; or False (or None) for none. Give at
            most one of ``prior_weights`` and ``quality_weights``.
        robust: Whether to estimate the prior of the moderation robustly.
        adjust_across_groups: Whether to add the column
            ``adjusted_p_value_across_groups``, as for :func:`limma_voom`.

    Returns:
        The tables of every group and estimable contrast, a summary of every group,
        the samples used and left out, the settings and a provenance record.

    Warns:
        RuntimeWarning: As for :func:`limma_voom`.

    Raises:
        TypeError: If ``cube`` is not a Cube, ``design`` not a Design, an option has
            the wrong type, or ``contrasts``, ``groups``, ``min_cells``,
            ``expression_filter`` or ``quality_weights`` is or holds a numpy masked
            array.
        ValueError: As for :func:`limma_voom`, or if ``prior_count`` is not
            positive.
        KeyError: If a group is not in the cube.
    """
    count = real_number(prior_count, "prior_count", finite=True)
    if not count > 0.0:
        raise ValueError(
            f"prior_count must be positive, so that zero counts have finite logs; got "
            f"{count}"
        )
    return _analyse(
        "limma_trend",
        cube,
        design,
        groups=groups,
        contrasts=contrasts,
        settings={
            "min_cells": _min_cells(min_cells),
            "min_counts": _min_counts(min_counts),
            "expression_filter": _filter_settings(expression_filter),
            "normalization": _normalization(normalization),
            "prior_count": count,
            **_sample_weight_settings(prior_weights, quality_weights, "limma_trend"),
            "robust": checked_flag(robust, "robust"),
            "adjust_across_groups": checked_flag(
                adjust_across_groups, "adjust_across_groups"
            ),
        },
    )


def dream_voom(
    cube: Cube,
    design: Design,
    *,
    groups: str | Sequence[str] | None = None,
    contrasts: Mapping[str, Any] | Sequence[str] | str | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    expression_filter: bool | Mapping[str, float] = True,
    normalization: Literal["tmm"] | None = "tmm",
    span: float | None = None,
    prior_weights: Literal["n_cells"] | None = None,
    robust: bool = False,
    trend: bool = False,
    adjust_across_groups: bool = False,
    n_threads: int | None = None,
) -> DifferentialExpressionResult:
    """Test contrasts in every group of a cube by dream, with random effects.

    variancePartition's dream workflow for repeated measures (Hoffman and Roussos
    2021), such as several samples of each donor: as :func:`limma_voom`, but the
    design's random intercepts, such as ``(1 | donor)``, enter every fit. For each
    group, the samples are left out, the features filtered and the counts normalized
    as for ``limma_voom``; voom's precision weights come from the mean-variance trend
    of mixed models fitted by maximum likelihood (``voomWithDreamWeights``,
    :func:`~cytepool.stats.dream.voom_with_dream_weights`); every feature's mixed
    model is fitted by REML with those weights (``dream``,
    :func:`~cytepool.stats.mixed_models.fit_mixed_model`); its estimable contrasts
    are tested with Satterthwaite's degrees of freedom
    (:func:`~cytepool.stats.mixed_models.fixed_effect_tests`); and the tests are
    moderated by empirical Bayes as variancePartition's ``eBayes`` moderates them
    (:func:`~cytepool.stats.dream.moderated_fixed_effect_tests`). The module
    documentation gives the steps and the differences from variancePartition.

    Every group's mixed model uses the random effects of the samples left in it: the
    levels of a grouping factor without samples there drop out, and a group whose
    samples cannot estimate a random effect, such as one donor left, or one sample
    per donor, is not fitted (status ``"random_effects_not_estimable"``). A feature
    whose mixed model has no minimum, which the fixed and random effects reproduce,
    is tested with NaN statistics; its group's record counts them.

    Satterthwaite's degrees of freedom are used for every group, as ``dream(...,
    ddf = "Satterthwaite")``; dream's default, ``ddf = "adaptive"``, uses
    Kenward-Roger's for 20 samples or fewer, which are not available yet.

    Args:
        cube: A cube with a ``counts`` layer.
        design: A :class:`~cytepool.design.Design` whose random effects are random
            intercepts, such as ``random="(1 | donor)"``, matched to the cube by
            ``sample_id`` as for :func:`limma_voom`.
        groups: A group id or a list of them; all groups by default.
        contrasts: The contrasts to test, matched to the design's columns by name;
            by default the design's named contrasts.
        min_cells: The fewest cells a sample needs in a group to enter its fit;
            ignored for cubes without cell numbers.
        min_counts: The fewest counts a sample needs in a group, over every
            feature of the cube, to enter its fit; 0 by default.
        expression_filter: True for edgeR's ``filterByExpr`` defaults with the
            group's fixed effects, a mapping of changed settings, or False.
        normalization: ``"tmm"`` for TMM normalization factors, or None.
        span: The LOWESS span of the trend, in (0, 1], for every group; None
            chooses it in each group from the number of features tested, as
            ``limma_voom`` does. variancePartition's ``voomWithDreamWeights`` uses
            0.5.
        prior_weights: ``"n_cells"`` to give each sample its number of cells in
            the group as a prior weight, as dreamlet weights samples, or None for
            none. Needs a cube that records cell numbers.
        robust: Whether to estimate the prior of the moderation robustly, as
            ``eBayes(robust = TRUE)`` does.
        trend: Whether the prior variance of the moderation follows the mean log2
            CPM of the features, as ``eBayes(trend = TRUE)`` does. dream moderates
            the residual variance of an observation of average weight, which keeps
            much of the mean-variance trend that voom's weights remove (see
            :mod:`cytepool.stats.dream`); variancePartition's default is False.
        adjust_across_groups: Whether to add the column
            ``adjusted_p_value_across_groups``, as for :func:`limma_voom`.
        n_threads: The most threads for the mixed model fits of the features (see
            :func:`~cytepool.stats.mixed_models.fit_mixed_model`), at least 1; None
            keeps numba's setting. The results do not depend on it.

    Returns:
        The tables of every group and estimable contrast, a summary of every group,
        the samples used and left out, the settings and a provenance record, as for
        :func:`limma_voom`; the tables also give each test's degrees of freedom
        (``df``) and z-statistic (``z_statistic``).

    Raises:
        TypeError: If ``cube`` is not a Cube, ``design`` not a Design, an option has
            the wrong type, or ``contrasts``, ``groups``, ``min_cells``,
            ``expression_filter`` or ``n_threads`` is or holds a numpy masked array.
        ValueError: If the design has no random effects or one that is not a
            random intercept, has no ``sample_id`` column, names samples the cube
            does not have or names a sample twice, if there are no contrasts to
            test, an option is invalid, or ``prior_weights`` is ``"n_cells"`` and
            the cube does not record cell numbers.
        KeyError: If a group is not in the cube.
    """
    return _analyse(
        "dream_voom",
        cube,
        design,
        groups=groups,
        contrasts=contrasts,
        settings={
            "min_cells": _min_cells(min_cells),
            "min_counts": _min_counts(min_counts),
            "expression_filter": _filter_settings(expression_filter),
            "normalization": _normalization(normalization),
            "span": _span(span),
            "prior_weights": _prior_weight_setting(prior_weights),
            "robust": checked_flag(robust, "robust"),
            "trend": checked_flag(trend, "trend"),
            "adjust_across_groups": checked_flag(
                adjust_across_groups, "adjust_across_groups"
            ),
            "n_threads": _threads(n_threads),
        },
    )


# The analysis.


@dataclass
class _GroupFit:
    """What happened in one group, and its tables."""

    group_id: str
    used: np.ndarray  # bool, per design sample in cube order
    reasons: list[str]  # why each design sample was left out; "" when used
    status: str = _FITTED
    reason: str = ""
    aliased: tuple[str, ...] = ()
    not_estimable: dict[str, str] = field(default_factory=dict)
    df_residual: float = float("nan")
    prior_df: float = float("nan")
    tested: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    library_sizes: np.ndarray | None = None  # per used sample, when fitted
    norm_factors: np.ndarray | None = None
    prior_weights: np.ndarray | None = None  # per used sample, when fitted with them
    quality_weights: np.ndarray | None = None
    voom_span: float | None = None  # the span of voom's trend, once voom has run
    features_not_fitted: int = 0  # dream: features whose mixed model has no minimum
    features_not_converged: int = 0  # dream: fitted features whose fit did not converge
    warnings: list[str] = field(default_factory=list)  # given as the group is fitted
    tables: dict[str, Table] = field(default_factory=dict)

    def stop(self, status: str, reason: str) -> "_GroupFit":
        self.status, self.reason = status, reason
        return self


@dataclass(frozen=True)
class _Inputs:
    """The checked inputs shared by every group."""

    method: str
    cube: Cube
    model_rows: np.ndarray  # the design's rows in cube order
    cube_rows: np.ndarray  # the cube position of each, increasing
    contrasts: Contrasts
    settings: Mapping[str, Any]


def _analyse(
    method: str,
    cube: Cube,
    design: Design,
    *,
    groups: str | Sequence[str] | None,
    contrasts: Any,
    settings: dict[str, Any],
) -> DifferentialExpressionResult:
    reject_masked(groups, "groups")
    if not isinstance(cube, Cube):
        raise TypeError(f"cube must be a Cube, got {type(cube).__name__}")
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design whose sample table has a sample_id column with "
            f"the cube's sample ids, got {type(design).__name__}"
        )
    if method == "dream_voom":
        if not design.random:
            raise ValueError(
                "dream_voom fits a mixed model to every feature, but the design has "
                "no random effects; build it with random intercepts such as "
                "random='(1 | donor)', or test a design without random effects with "
                "limma_voom"
            )
        _random_intercepts(design)
    else:
        try:
            refuse_random_effects(design, method)
        except ValueError as error:
            if not all(is_random_intercept(effect) for effect in design.random):
                raise
            raise ValueError(
                f"{error} On a cube, dream_voom fits these random intercepts in every "
                "group, as variancePartition's dream does."
            ) from None
    if settings["prior_weights"] == "n_cells" and cube.n_cells is None:
        raise ValueError(
            "prior_weights='n_cells' weights each sample by its number of cells, which "
            "this cube does not record (a cube of bulk samples has none); leave "
            "prior_weights at None"
        )
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
    tested = _contrasts_to_test(design, contrasts, method)
    positions = _group_positions(cube, groups)
    inputs = _Inputs(method, cube, order, cube_rows, tested, settings)
    fit_group = _fit_dream_group if method == "dream_voom" else _fit_group
    fits = []
    for g in positions:
        fit = fit_group(inputs, design, g)
        for message in fit.warnings:
            warnings.warn(
                f"group {fit.group_id!r}: {message}",
                RuntimeWarning,
                skip_file_prefixes=(PACKAGE_DIRECTORY,),
            )
        fits.append(fit)
    if settings["adjust_across_groups"]:
        _adjust_across_groups(fits, tested.names)
    sample_ids = cube.samples["sample_id"][cube_rows]
    return DifferentialExpressionResult(
        method,
        tested,
        _summary_table(fits),
        _sample_table(cube, cube_rows, sample_ids, positions, fits, settings),
        {
            (fit.group_id, contrast): table
            for fit in fits
            for contrast, table in fit.tables.items()
        },
        {
            "method": method,
            "versions": software_versions(),
            "cube": cube_record(cube),
            "layer": _LAYER,
            "design": design_record(design, tested),
            "settings": {
                **settings,
                **_FIXED_SETTINGS,
                **(_DREAM_SETTINGS if method == "dream_voom" else {}),
            },
            "groups": [
                _group_record(fit, sample_ids, method, settings) for fit in fits
            ],
        },
    )


@dataclass(frozen=True)
class _GroupData:
    """The checked data of one group, ready for the fits of a method."""

    x: np.ndarray  # the rows of the full design for the samples used, kept columns
    columns: tuple[str, ...]  # the names of the kept columns
    kept: list[int]  # the kept columns among the design's
    estimable: list[int]  # the contrasts that the samples can estimate
    counts: np.ndarray  # samples used x features tested
    sizes: np.ndarray  # library sizes over the features tested
    factors: np.ndarray | None  # normalization factors, or None
    prior: np.ndarray | None  # prior weights of the samples used, or None
    mixed: _MixedDesign | None  # the mixed model's design of the samples used


def _prepare_group(
    inputs: _Inputs, design: Design, g: int, *, mixed: bool = False
) -> tuple[_GroupFit, _GroupData | None]:
    """The samples, design, features and normalization of one group.

    Returns the group's record and its data, or no data when the group cannot be
    fitted, with the record saying why. With ``mixed``, the random effects of the
    design are also checked on the samples used.
    """
    cube, settings = inputs.cube, inputs.settings
    counts = cube.layer(_LAYER, g)[inputs.cube_rows]  # design samples, cube order
    cells = None if cube.n_cells is None else cube.n_cells[inputs.cube_rows, g]
    min_counts = settings["min_counts"]
    reasons = _left_out(counts, cells, settings["min_cells"], min_counts)
    used = np.array([not reason for reason in reasons], dtype=bool)
    fit = _GroupFit(str(cube.groups["group_id"][g]), used, reasons)
    if counts.shape[1] == 0:
        return fit.stop("no_features", "the cube has no features"), None
    n_used = int(np.count_nonzero(used))
    if n_used == 0:
        # No counts is fewer than any positive minimum.
        rules = (
            f"fewer than {_amount(min_counts)} counts" if min_counts else "no counts"
        )
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
    violations = _contrast_violations(
        replace(design.model_matrix, matrix=x), aliasing, contrasts
    )
    for position, relations in violations.items():
        fit.not_estimable[contrasts.names[position]] = (
            "not estimable from the samples of this group, which cannot separate the "
            "effects it depends on: " + "; ".join(relations)
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
            parts = _mixed_design_of_samples(
                design, inputs.model_rows[used], x_kept, columns
            )
        except ValueError as error:
            return fit.stop("random_effects_not_estimable", str(error)), None

    y = np.asarray(counts[used], dtype=np.float64)
    filter_settings = settings["expression_filter"]
    if filter_settings is None:
        keep = np.ones(y.shape[1], dtype=bool)
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
    data = _GroupData(x_kept, columns, kept, estimable, y, sizes, factors, prior, parts)
    return fit, data


def _fit_group(inputs: _Inputs, design: Design, g: int) -> _GroupFit:
    """Fit one group by limma, or record why it cannot be fitted."""
    cube, settings = inputs.cube, inputs.settings
    fit, data = _prepare_group(inputs, design, g)
    if data is None:
        return fit
    x_kept, columns, kept, estimable = data.x, data.columns, data.kept, data.estimable
    y, sizes, factors, prior = data.counts, data.sizes, data.factors, data.prior
    n_used, n_tested = y.shape
    contrasts = inputs.contrasts
    quality_settings = settings["quality_weights"]

    if inputs.method == "limma_voom":
        with_counts = int(np.count_nonzero(y.any(axis=0)))
        if with_counts < 2:
            return fit.stop(
                "too_few_features",
                "voom fits its mean-variance trend to at least two features with "
                f"counts, and {with_counts} of the {n_tested} features tested "
                f"{'has' if with_counts == 1 else 'have'} counts",
            )
        step = "voom" if quality_settings is None else "voom with quality weights"
        try:
            weighted = _voom_weights(y, x_kept, factors, prior, settings, fit.warnings)
        except _SingularInformation:
            return fit.stop("failed", f"{step}: {_singular(quality_settings)}")
        except ValueError as error:  # the trend cannot give weights for these data
            return fit.stop("failed", f"{step}: {error}")
        fit.voom_span = weighted.span
        quality = weighted.sample_weights
        model = fit_linear_model(
            weighted.log_cpm,
            x_kept,
            weights=weighted.weights,
            coefficient_names=columns,
            n_threads=settings["n_threads"],
        )
        method = settings["contrast_standard_errors"]
    else:
        values = log_cpm(y, norm_factors=factors, prior_count=settings["prior_count"])
        quality = None
        if quality_settings is not None:
            try:
                quality = _sample_quality(
                    values, x_kept, quality_settings, fit.warnings
                )
            except _SingularInformation:
                reason = _singular(quality_settings)
                return fit.stop("failed", f"sample quality weights: {reason}")
        # Sample weights, when given, are shared by every feature: one weighted fit
        # serves them all, and the standard errors of contrasts are exact, as
        # limma's are then.
        model = fit_linear_model(
            values,
            x_kept,
            weights=quality if prior is None else prior,
            coefficient_names=columns,
        )
        method = "exact"
    # An estimable contrast is its kept part applied to the fit on the kept columns.
    contrast_fit = contrasts_fit(
        model,
        Contrasts(
            contrasts.matrix[np.ix_(kept, estimable)],
            tuple(contrasts.names[p] for p in estimable),
            columns,
        ),
        method=method,
    )
    try:
        moderated = moderated_statistics(
            contrast_fit,
            trend=inputs.method == "limma_trend",
            robust=settings["robust"],
        )
    except ValueError as error:  # for example, mostly zero variances (robust)
        return fit.stop("failed", f"empirical Bayes moderation: {error}")
    adjusted = adjust_p_values(moderated.p_value, "bh", axis=0)

    fit.df_residual = float(model.df_residual[0])
    fit.prior_df = float(np.median(moderated.prior_df))
    fit.library_sizes = sizes
    fit.norm_factors = np.ones(n_used) if factors is None else factors
    fit.prior_weights = prior
    fit.quality_weights = quality
    features = cube.features
    identity = {"feature_id": features["feature_id"][fit.tested]}
    if "symbol" in features:
        identity["symbol"] = features["symbol"][fit.tested]
    # The moderated standard errors, as they enter the t-statistics.
    errors = (
        contrast_fit.stdev_unscaled * np.sqrt(moderated.posterior_variance)[:, None]
    )
    for column, name in enumerate(contrast_fit.coefficient_names):
        fit.tables[name] = Table(
            {
                **identity,
                "log2_fold_change": contrast_fit.coefficients[:, column],
                "standard_error": errors[:, column],
                "average_log2_cpm": model.mean_expression,
                "t": moderated.t[:, column],
                "p_value": moderated.p_value[:, column],
                "adjusted_p_value": adjusted[:, column],
                "b_statistic": moderated.lods[:, column],
            }
        )
    return fit


def _fit_dream_group(inputs: _Inputs, design: Design, g: int) -> _GroupFit:
    """Fit one group by dream, or record why it cannot be fitted."""
    cube, settings = inputs.cube, inputs.settings
    fit, data = _prepare_group(inputs, design, g, mixed=True)
    if data is None or data.mixed is None:
        return fit
    y = data.counts
    n_used, n_tested = y.shape
    with_counts = int(np.count_nonzero(y.any(axis=0)))
    if with_counts < 2:
        return fit.stop(
            "too_few_features",
            "the mean-variance trend of the precision weights needs at least two "
            f"features with counts, and {with_counts} of the {n_tested} features "
            f"tested {'has' if with_counts == 1 else 'have'} counts",
        )
    sizes = np.array(effective_library_sizes(y, None, data.factors))
    fraction = _trend_span(settings["span"], n_tested)
    prior = None if data.prior is None else _scaled_prior_weights(data.prior, y.shape)
    with thread_limit(settings["n_threads"]):
        try:
            weighted = _dream_weights(y, sizes, fraction, data.mixed, prior, True)
        except ValueError as error:  # the trend cannot give weights for these data
            return fit.stop("failed", f"voom with dream weights: {error}")
        model = _fit_mixed_model(weighted.log_cpm, weighted.weights, False, data.mixed)
    fit.voom_span = weighted.span
    contrasts = inputs.contrasts
    names = tuple(contrasts.names[p] for p in data.estimable)
    # An estimable contrast is its kept part applied to the fit on the kept columns.
    tests = fixed_effect_tests(
        model,
        Contrasts(
            contrasts.matrix[np.ix_(data.kept, data.estimable)], names, data.columns
        ),
    )
    try:
        moderated = moderated_fixed_effect_tests(
            tests, robust=settings["robust"], trend=settings["trend"]
        )
    except ValueError as error:  # for example, no feature has a fit
        return fit.stop("failed", f"empirical Bayes moderation: {error}")
    adjusted = adjust_p_values(moderated.p_value, "bh", axis=0)

    fitted = np.isfinite(model.residual_variance)
    fit.features_not_fitted = int(np.count_nonzero(~fitted))
    fit.features_not_converged = int(np.count_nonzero(fitted & ~model.converged))
    fit.df_residual = float(np.median(model.residual_df[fitted]))
    fit.prior_df = float(np.median(moderated.prior_df))
    fit.library_sizes = data.sizes
    fit.norm_factors = np.ones(n_used) if data.factors is None else data.factors
    fit.prior_weights = data.prior
    features = cube.features
    identity = {"feature_id": features["feature_id"][fit.tested]}
    if "symbol" in features:
        identity["symbol"] = features["symbol"][fit.tested]
    for column, name in enumerate(names):
        fit.tables[name] = Table(
            {
                **identity,
                "log2_fold_change": tests.estimate[:, column],
                "average_log2_cpm": model.mean_expression,
                "t": moderated.t[:, column],
                "df": moderated.df_total[:, column],
                "p_value": moderated.p_value[:, column],
                "adjusted_p_value": adjusted[:, column],
                "b_statistic": moderated.lods[:, column],
                "z_statistic": moderated.z[:, column],
            }
        )
    return fit


def _left_out(
    counts: np.ndarray, cells: np.ndarray | None, min_cells: int, min_counts: float
) -> list[str]:
    """Why each sample is left out of a group's fit, or "" when it is used."""
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
                f"{_amount(total)} count{'' if total == 1.0 else 's'}, fewer than "
                f"min_counts ({_amount(min_counts)})"
            )
        else:
            reasons.append("")
    return reasons


def _voom_weights(
    y: np.ndarray,
    x: np.ndarray,
    factors: np.ndarray | None,
    prior: np.ndarray | None,
    settings: Mapping[str, Any],
    notes: list[str],
) -> VoomResult:
    """voom, with prior weights or sample quality weights when asked.

    The warnings of the quality weights are appended to ``notes``; voom alone gives
    none here, as the group has residual degrees of freedom and positive weights.
    """
    quality = settings["quality_weights"]
    if quality is None:
        return voom(y, x, norm_factors=factors, span=settings["span"], weights=prior)
    counts, sizes, fraction = _voom_inputs(
        y, x, None, factors, settings["span"], "limma_voom"
    )
    model = _variance_model(
        None, None, counts.shape[0], methods=("gene_by_gene", "reml"), **quality
    )
    weighted, given = _quality_weighted_voom(counts, x, sizes, fraction, model)
    notes.extend(given)
    return weighted


def _sample_quality(
    values: np.ndarray, x: np.ndarray, settings: Mapping[str, Any], notes: list[str]
) -> np.ndarray:
    """Sample quality weights of log-expression values, as limma's arrayWeights.

    Their warnings are appended to ``notes``.
    """
    model = _variance_model(
        None, None, values.shape[0], methods=("gene_by_gene", "reml"), **settings
    )
    log_variances, given = _estimate(values, x, None, model)
    notes.extend(given)
    weights: np.ndarray = _weights_of(log_variances)
    return weights


def _singular(settings: Mapping[str, Any]) -> str:
    """Why sample quality weights could not be estimated, and what to change."""
    return (
        "the information about the sample weights is singular: some samples carry no "
        "information about their variance, such as a sample that the design fits "
        "exactly, and quality_weights['prior_n'] "
        f"({settings['prior_n']:g}) is too small to make up for it; a larger prior_n "
        "keeps their weights near 1"
    )


def _adjust_across_groups(fits: list[_GroupFit], contrasts: Sequence[str]) -> None:
    """Add to each table the Benjamini-Hochberg adjustment of its contrast's p-values
    over every group that tests the contrast, after its ``adjusted_p_value``."""
    for contrast in contrasts:
        testing = [fit for fit in fits if contrast in fit.tables]
        if not testing:
            continue
        p_values = np.concatenate([fit.tables[contrast]["p_value"] for fit in testing])
        adjusted = adjust_p_values(p_values, "bh")
        start = 0
        for fit in testing:
            table = fit.tables[contrast]
            stop = start + len(table)
            columns: dict[str, Any] = {}
            for name in table.columns:
                columns[name] = table[name]
                if name == "adjusted_p_value":
                    columns[_ACROSS_GROUPS] = adjusted[start:stop]
            fit.tables[contrast] = Table(columns)
            start = stop


# Checks of the arguments.


def _contrasts_to_test(design: Design, contrasts: Any, function: str) -> Contrasts:
    # make_contrasts refuses contrasts that are or hold a numpy masked array.
    chosen = (
        design.contrasts if contrasts is None else make_contrasts(contrasts, design)
    )
    if not len(chosen):
        raise ValueError(
            f"{function} has no contrasts to test: the design has no named contrasts "
            "and none were given. Name them in the design, as in Design(samples, "
            "formula, contrasts={'stim_vs_ctrl': 'conditionstim'}), or pass "
            f"contrasts= in the same form; the design's columns are: "
            f"{', '.join(design.columns)}"
        )
    return chosen


def _group_positions(cube: Cube, groups: str | Sequence[str] | None) -> list[int]:
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


def _min_cells(value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError("min_cells must be a whole number of cells, not bool")
    number = index_integer(value, "min_cells", expected="a whole number of cells")
    if number < 0:
        raise ValueError(f"min_cells must be at least 0, got {number}")
    return number


def _min_counts(value: Any) -> float:
    # real_number would refuse a masked value too, but without saying how to fix it.
    reject_masked(value, "min_counts")
    return non_negative_number(value, "min_counts")


def _prior_weight_setting(prior_weights: Any) -> str | None:
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


def _sample_weight_settings(
    prior_weights: Any, quality_weights: Any, method: str
) -> dict[str, Any]:
    """The checked settings of the sample weights, which are alternatives."""
    _prior_weight_setting(prior_weights)
    quality = _quality_settings(quality_weights, _QUALITY_METHODS[method])
    if prior_weights is not None and quality is not None:
        raise ValueError(
            "give prior_weights or quality_weights, not both: quality weights "
            "estimate the precision of each sample from the data, and prior weights "
            "set it from the number of cells. limma's voomWithQualityWeights takes no "
            "prior weights, and edgeR's voomLmFit refuses the two together"
        )
    cells = None if prior_weights is None else "n_cells"
    return {"prior_weights": cells, "quality_weights": quality}


def _quality_settings(value: Any, method: str) -> dict[str, Any] | None:
    """The settings of sample quality weights: None, or limma's with changes."""
    reject_masked(value, "quality_weights")
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return {"method": method, **_QUALITY_DEFAULTS} if value else None
    if not isinstance(value, Mapping):
        raise TypeError(
            "quality_weights must be True, a mapping of settings such as "
            f"{{'method': 'reml'}}, or False or None for none, got "
            f"{type(value).__name__}"
        )
    settings = {"method": method, **_QUALITY_DEFAULTS}
    for name in value:
        if name not in settings:
            raise ValueError(
                f"quality_weights has the unknown setting {name!r}; the settings are "
                f"{', '.join(settings)}"
            )
    settings |= dict(value)
    return {
        "method": checked_choice(
            settings["method"], "quality_weights['method']", ("gene_by_gene", "reml")
        ),
        "prior_n": number_at_least(
            settings["prior_n"], "quality_weights['prior_n']", 0.0
        ),
        "max_iterations": positive_integer(
            settings["max_iterations"], "quality_weights['max_iterations']"
        ),
        "tolerance": positive_finite_number(
            settings["tolerance"], "quality_weights['tolerance']"
        ),
    }


def _span(value: Any) -> float | None:
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


def _threads(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        raise TypeError("n_threads must be a positive integer or None, not bool")
    number = index_integer(value, "n_threads", expected="a positive integer or None")
    if number < 1:
        raise ValueError(f"n_threads must be at least 1, got {number}")
    return number


def _filter_settings(value: Any) -> dict[str, float] | None:
    reject_masked(value, "expression_filter")
    if isinstance(value, bool | np.bool_):
        return dict(_FILTER_DEFAULTS) if value else None
    if not isinstance(value, Mapping):
        raise TypeError(
            "expression_filter must be True, False or a mapping of filterByExpr "
            f"settings such as {{'min_count': 5}}, got {type(value).__name__}"
        )
    settings = dict(_FILTER_DEFAULTS)
    for name, number in value.items():
        if name not in _FILTER_DEFAULTS:
            raise ValueError(
                f"expression_filter has the unknown setting {name!r}; the settings are "
                f"{', '.join(_FILTER_DEFAULTS)}"
            )
        settings[name] = non_negative_number(number, f"expression_filter[{name!r}]")
    if settings["min_prop"] > 1.0:
        raise ValueError(
            f"expression_filter['min_prop'] must be in [0, 1], got {settings['min_prop']}"
        )
    return settings


def _normalization(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and value == "tmm"):
        return value
    error = ValueError if isinstance(value, str) else TypeError
    raise error(
        "normalization must be 'tmm' (edgeR's TMM factors) or None (library sizes "
        f"alone), got {value!r}"
    )


# Results.


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
            "n_features_tested": np.array(
                [fit.tested.size if fit.status == _FITTED else 0 for fit in fits],
                dtype=np.int64,
            ),
            "df_residual": np.array([fit.df_residual for fit in fits], dtype=float),
            "prior_df": np.array([fit.prior_df for fit in fits], dtype=float),
            "not_estimable": text_column(
                [", ".join(fit.not_estimable) for fit in fits]
            ),
        }
    )


def _sample_table(
    cube: Cube,
    cube_rows: np.ndarray,
    sample_ids: np.ndarray,
    positions: list[int],
    fits: list[_GroupFit],
    settings: Mapping[str, Any],
) -> Table:
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
        if fit.status != _FITTED:
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


# The provenance record.


def _group_record(
    fit: _GroupFit, sample_ids: np.ndarray, method: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    fitted = fit.status == _FITTED
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
        "not_estimable": dict(fit.not_estimable),
        "n_features_tested": int(fit.tested.size) if fitted else 0,
    }
    if method in ("limma_voom", "dream_voom"):
        # The span of the group's voom trend; null when voom did not run.
        record["voom_span"] = fit.voom_span
    if method == "dream_voom":
        # The tested features whose mixed model has no minimum, and those whose fit
        # stopped without converging; null when the group was not fitted.
        record["features_not_fitted"] = fit.features_not_fitted if fitted else None
        record["features_not_converged"] = (
            fit.features_not_converged if fitted else None
        )
    # The sample weights of the fit, in the order of "samples"; null when not fitted.
    for key in ("prior_weights", "quality_weights"):
        if settings.get(key) is not None:
            weights = getattr(fit, key)
            record[key] = weights.tolist() if fitted and weights is not None else None
    record["warnings"] = list(fit.warnings)
    return record


# Messages.


def _amount(value: float) -> str:
    """A number of counts as text, without a decimal point when it is whole."""
    return f"{value:.15g}"
