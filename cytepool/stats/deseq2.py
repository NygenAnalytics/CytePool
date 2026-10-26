"""DESeq2's dispersion estimates and its Wald and likelihood-ratio tests.

DESeq2 (Love, Huber and Anders 2014) models the count y_gi of gene g in sample i as

    y_gi ~ NB(mu_gi, alpha_g),    var(y_gi) = mu_gi + alpha_g mu_gi^2,
    log2(mu_gi) = log2(s_i) + x_i' beta_g,

with s_i the size factor of sample i (by default the median-of-ratios factors of
Anders and Huber (2010), :func:`~cytepool.normalization.median_ratio_size_factors`),
x_i its row of the design
and beta_g the coefficients of gene g on the log2 scale. :func:`fit_deseq2` estimates
the dispersions and fits every gene, as ``DESeq()`` does; :func:`deseq2_wald_test`
tests a coefficient or contrast and :func:`deseq2_likelihood_ratio_test` a reduced
design against the full one, with the columns of ``results()``. They follow DESeq2's
documentation and paper, with DESeq2's defaults; conventions the documentation leaves
open were found by running DESeq2 1.46.0 as a black box and are marked "found by
probing". Genes without counts are left out of every step and get NaN.

Dispersions (:func:`deseq2_dispersions`, as ``estimateDispersions()``):

1. The means of the likelihood. For a one-way layout, whose design rows take as many
   distinct values as it has columns, the linear model of the normalized counts
   y / s: s_i times the mean normalized count of the samples with sample i's row.
   For other designs the fit described below, at a rough dispersion: the smaller of
   a moment estimate from that linear model's residuals, its fitted values raised to
   at least 1, and one from the mean and variance of the normalized counts (found by
   probing, as PyDESeq2 computes them). Means below ``min_mean`` (0.5) are raised to
   it.
2. Gene-wise estimates: the maximum of the Cox-Reid adjusted profile log-likelihood
   (Cox and Reid 1987; McCarthy, Chen and Smyth 2012), conditional on those means,

       l(alpha) - 1/2 log det(X' W X),    W = diag(mu / (1 + alpha mu)),

   over [1e-8, max(10, n_samples)], the bounds of DESeq2 (found by probing). A gene
   whose rough dispersion is at the lower bound keeps the bound as its estimate, as in
   DESeq2, even where the maximum lies above it (found by probing).
3. The trend of the dispersions on the mean normalized count m, alpha_tr(m) = a0 +
   a1 / m: a gamma-family generalized linear model with identity link, fitted to the
   genes whose gene-wise estimate is above 1e-6 (found by probing) and lies within
   (1e-4, 15) times the current trend, from a0 = 0.1 and a1 = 1, and fitted again until
   the squared logarithmic changes of a0 and a1 add up to less than 1e-6 after a fit
   that converged, in at most 11 fits. Each fit is that of R's ``glm()`` with its
   default control, as DESeq2 makes it: iteratively reweighted least squares from the
   coefficients of the fit before, stopped when the deviance changes by less than 1e-8
   of it or after 25 steps, with a step that makes a mean non-positive halved towards
   the coefficients before it (found by probing; with these, the coefficients agree
   with DESeq2's to rounding). The trend is then evaluated at every gene. With
   ``trend="mean"``, the trend is the mean of the gene-wise estimates above 1e-7,
   trimmed by 0.1% at each end, instead.
4. The prior log(alpha_g) ~ N(log(alpha_tr(m_g)), sigma_d^2). With s_lr the median
   absolute deviation, scaled as R's ``mad``, of the log residuals log(alpha_gw /
   alpha_tr) over the genes whose gene-wise estimate is at least 1e-6 (found by
   probing), and d the residual degrees of freedom, sigma_d^2 = max(s_lr^2 -
   trigamma(d / 2), 0.25). For d of 3 or fewer the paper matches the density of the
   log residuals to that of simulated ones, log(chi^2_d / d) + N(0, sigma_d^2); here
   sigma_d^2 is the maximum likelihood estimate in that model instead, over [0, 8] (the
   range of DESeq2's grid), and at least 0.25.
5. Maximum a posteriori estimates: the maximum of the adjusted log-likelihood plus the
   logarithm of the prior density, over the same interval and with the same means.
6. A gene whose log gene-wise estimate exceeds the log trend by more than 2 s_lr is a
   dispersion outlier: it keeps its gene-wise estimate, and the others take their
   maximum a posteriori estimate.

Each maximum is found from the sign changes of the derivative on a grid of 30 log
dispersions, refined to 1e-12 by the Illinois variant of false position (Dowell and
Jarratt 1971), the largest of several being kept. The derivative is
computed without cancellation where the dispersion is small: as series in the
dispersion and from logarithms of ratios.

Fits (:func:`fit_deseq2`, as ``nbinomWaldTest()``): DESeq2's iteratively reweighted
least squares raises fitted values below ``min_mean`` to it in the weights and working
values, and adds a ridge penalty of 1e-6 beta^2 / 2 on the log2 scale (found by
probing). Its fixed point maximizes a concave function: the log-likelihood in which the
term of each observation is continued, below a fitted value of ``min_mean``, by its
quadratic expansion there in the linear predictor, with the expected information as
curvature, less the penalty. Each gene is fitted by Newton's method on that function,
to steps of 1e-10 in the log fitted values; a design that is an intercept alone is
fitted by the log of the mean normalized count, as DESeq2 does (found by probing). The
covariance of the coefficients is (I + L)^-1 I (I + L)^-1, with I = X' W X, W =
mu / (1 + alpha mu) at the raised fitted values and L the penalty; the hat values are
the diagonal of W^(1/2) X (I + L)^-1 X' W^(1/2). The fitted values reported are not
raised.

Cook's distances are r^2 / p h / (1 - h)^2, with h the hat value, p the number of
coefficients and r the Pearson residual (y - mu) / sqrt(mu + a mu^2) at a robust moment
estimate a of the dispersion (found by probing, with the constants of PyDESeq2): a = (v
- m) / m^2, at least 0.04, with m the mean normalized count and v a robust variance of
the normalized counts, the largest over the groups of three or more samples that share
a design row of a trimmed mean of squared deviations from the group's trimmed mean (at
each end 1/3 of the samples of a group of 3, 1/4 below 24 and 1/8 from 24, scaled by
2.04, 1.86 or 1.51), or, without such groups, the same over all samples with 1/8 and
1.51.

Tests and ``results()``:

- Wald test: the log2 fold change c' beta of a contrast c, its standard error
  sqrt(c' Sigma c), and the statistic c' beta / sqrt(c' Sigma c) with two-sided normal
  p-values. When a contrast has weights of both signs and every count is zero in the
  samples with a nonzero entry in a column of positive weight, and in those with one in
  a column of negative weight, the log2 fold change and the statistic are 0 and the
  p-value 1 (found by probing; ``results()`` documents this for a comparison of two
  groups). The standard error is kept.
- Likelihood ratio test: the difference of the deviances of the reduced and the full
  fit, against the chi-squared distribution with as many degrees of freedom as the full
  design has more columns. Its log2 fold change and standard error are those of a
  coefficient of the full fit.
- A gene's p-value is removed (NaN) when its largest Cook's distance over the samples
  whose design row occurs three or more times exceeds the 0.99 quantile of F(p, n - p);
  in a comparison of two groups (one factor of two levels, with or without an
  intercept) not when three or more samples have larger counts than the sample with
  the largest distance over all samples (found by probing). A numeric covariate of two
  values is not such a comparison; a design matrix that is not a ``Design`` is, when
  its rows take two distinct values of zeros and ones.
- Adjusted p-values are Benjamini-Hochberg's (Benjamini and Hochberg 1995), after
  independent filtering on the mean normalized count (Bourgon, Gentleman and Huber
  2010): at 50 quantiles of the mean counts, from the fraction of genes without counts
  to 0.95, the adjusted p-values of the genes at or above the quantile, and the number
  of them below ``alpha``. The quantile kept is the lowest whose number exceeds the
  maximum of a LOWESS fit (span 1/5) less the root mean square of its residuals, or
  failing that 90% and then 80% of that maximum (found by probing), or the lowest
  quantile; and the lowest quantile when no number exceeds 10. Genes below it get NaN.

Deliberate differences from DESeq2, found by probing:

- DESeq2 maximizes the dispersions by a line search that stops when an iteration gains
  less than 1e-6, and that can stall where it starts, returning its starting value: its
  gene-wise estimates are often about 1e-4 from the maximum on the log scale and a few
  up to 1 or more, and its maximum a posteriori estimates up to about 1e-2 from theirs,
  relative. Here each is the maximum, but for the genes of step 2 that keep the lower
  bound. DESeq2's own maximum a posteriori routine, given a nearly flat prior, finds
  the same gene-wise maxima.
- DESeq2's fits stop when the deviance changes by less than 1e-8 of it: its
  coefficients up to about 3e-3 from the fixed point, relative to the larger of the
  coefficient and 1, and the means of its first fit, at the rough dispersion, up to
  about 1e-3. Here every fit converges.
- For three or fewer residual degrees of freedom DESeq2 chooses sigma_d^2 from a grid of
  1,000 values on [0, 8] by comparing densities of simulated residuals; its choice moves
  by several percent with the sample of residuals, and is high for small variances (0.55
  to 0.58 for 0.5). The maximum likelihood estimate used here is the limit of its
  criterion as those densities become exact.
- Where the parametric trend fails, a coefficient not positive or no convergence in 11
  fits, DESeq2 substitutes a local regression, which CytePool does not have; the mean
  trend is used instead, with a warning, as PyDESeq2 does.

Not implemented: the replacement of outlier counts that ``DESeq()`` makes for design
rows that occur seven times or more (the results here are those of ``DESeq(...,
minReplicatesForReplace = Inf)``), shrunken fold changes (``lfcShrink``), the local
trend and glmGamPoi, observation weights, normalization factors that differ between
genes, thresholds on the fold change (``lfcThreshold``) and t-distributed Wald tests
(``useT``).

The constants and the structure taken from PyDESeq2 (Muzellec et al. 2023), in the
robust dispersions of Cook's distances, the rough dispersions and the fallback to the
mean trend, are recorded, with PyDESeq2's license, in ``THIRD_PARTY_NOTICES.md``.

References:
    Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and
    dispersion for RNA-seq data with DESeq2. Genome Biology 15, 550.
    doi:10.1186/s13059-014-0550-8

    McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation. Nucleic Acids
    Research 40, 4288-4297. doi:10.1093/nar/gks042

    Bourgon R, Gentleman R, Huber W (2010). Independent filtering increases detection
    power for high-throughput experiments. Proceedings of the National Academy of
    Sciences 107, 9546-9551. doi:10.1073/pnas.0914005107

    Cook RD (1977). Detection of influential observation in linear regression.
    Technometrics 19, 15-18. doi:10.1080/00401706.1977.10489493

    Muzellec B, Telenczuk M, Cabeli V, Andreux M (2023). PyDESeq2: a python package for
    bulk RNA-seq differential expression analysis. Bioinformatics 39, btad547.
    doi:10.1093/bioinformatics/btad547

    Anders S, Huber W (2010). Differential expression analysis for sequence count data.
    Genome Biology 11, R106. doi:10.1186/gb-2010-11-10-r106

    Cox DR, Reid N (1987). Parameter orthogonality and approximate conditional
    inference. Journal of the Royal Statistical Society B 49, 1-18.
    doi:10.1111/j.2517-6161.1987.tb01422.x

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Dowell M, Jarratt P (1971). A modified regula falsi method for computing the root of
    an equation. BIT 11, 168-174. doi:10.1007/BF01934364

    Olver FWJ, Olde Daalhuis AB, Lozier DW, Schneider BI, Boisvert RF, Clark CW,
    Miller BR, Saunders BV, Cohl HS, McClain MA, eds (2026). NIST Digital Library of
    Mathematical Functions, release 1.2.8 of 2026-09-15. https://dlmf.nist.gov/
"""

import math
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._jit import jit, thread_limit
from cytepool._linear_algebra import RANK_TOLERANCE, column_space
from cytepool._masked_arrays import reject_masked, unmasked_array
from cytepool._summation import accurate_sums
from cytepool.design.contrasts import make_contrasts
from cytepool.design.design import Design, _refuse_random_effects
from cytepool.normalization import _as_counts, median_ratio_size_factors
from cytepool.stats.distributions import chi2_sf, f_ppf, norm_sf
from cytepool.stats.linear_models import (
    _aliased_columns,
    _contrast_matrix,
    _design_matrix,
)
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.negative_binomial_models import (
    _log1p_ratio,
    negative_binomial_deviance,
)
from cytepool.stats.smoothing import lowess
from cytepool.stats.special import (
    _log_gamma_increment,
    _stirling_error,
    digamma_scalar,
    lgamma_scalar,
    trigamma,
)

__all__ = [
    "DESeq2Dispersions",
    "DESeq2Fit",
    "DESeq2Results",
    "deseq2_dispersions",
    "deseq2_likelihood_ratio_test",
    "deseq2_wald_test",
    "fit_deseq2",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Dispersions lie in [_MIN_DISPERSION, max(_MAX_DISPERSION, n_samples)] (found by
# probing). The parametric trend is fitted to the genes whose gene-wise estimate is
# above _TREND_GENES times the lower bound, the prior's width estimated from those at
# least _PRIOR_GENES times it, and the mean trend from those above _MEAN_GENES times
# it (all found by probing).
_MIN_DISPERSION = 1e-8
_MAX_DISPERSION = 10.0
_TREND_GENES = 100.0
_PRIOR_GENES = 100.0
_MEAN_GENES = 10.0
_MEAN_TRIM = 0.001
# R's mad() scales the median absolute deviation by this constant.
_MAD_SCALE = 1.4826
_PRIOR_FLOOR = 0.25
_FEW_DF = 3
_LARGEST_PRIOR_VARIANCE = 8.0
_OUTLIER_SDS = 2.0
# The parametric trend (found by probing): its starting coefficients, the range of
# the ratios of gene-wise estimate to trend of the genes it is fitted to, its
# convergence criterion and the most fits.
_TREND_START = (0.1, 1.0)
_TREND_RATIOS = (1e-4, 15.0)
_TREND_TOLERANCE = 1e-6
_TREND_FITS = 11
# Each gamma fit is that of R's glm() with its default control (glm.control()): the
# convergence tolerance on the relative change of the deviance and the most steps; and
# the most halvings of a step whose means are not positive.
_GLM_TOLERANCE = 1e-8
_GLM_STEPS = 25
_GLM_HALVINGS = 60
# The ridge penalty of DESeq2's fits, 1e-6 on the log2 scale (found by probing), and
# its value for coefficients on the natural log scale.
_RIDGE = 1e-6
_LOG2 = math.log(2.0)
_NATURAL_RIDGE = _RIDGE / (_LOG2 * _LOG2)
# The robust dispersion of Cook's distances: the trimmed fraction and scale for groups
# of fewer than 3.5, fewer than 23.5 and more samples, and its lower bound.
_TRIMMED_FRACTIONS = (1.0 / 3.0, 1.0 / 4.0, 1.0 / 8.0)
_TRIMMED_SCALES = (2.04, 1.86, 1.51)
_ROBUST_DISPERSION_FLOOR = 0.04
_REPLICATES = 3
_COOKS_QUANTILE = 0.99
# Independent filtering: the number of quantiles, the highest one, the LOWESS span,
# the fewest rejections for which a threshold is chosen, and the fallback fractions
# of the fitted maximum.
_FILTER_QUANTILES = 50
_UPPER_QUANTILE = 0.95
_FILTER_SPAN = 1.0 / 5.0
_FEWEST_REJECTIONS = 10
_FALLBACK_FRACTIONS = (0.9, 0.8)
# Searches for the maxima of the dispersions: grid points over the log interval, and
# the width of the bracket at which the root of the derivative is accepted.
_GRID_POINTS = 30
_ROOT_TOLERANCE = 1e-12
_ROOT_ITERATIONS = 200
# Newton's method for the fits: steps that change some log fitted value by more than
# _LARGEST_STEP are shortened to it; a fit has converged when a full step changes no
# log fitted value by more than _STEP_TOLERANCE.
_LARGEST_STEP = 5.0
_STEP_TOLERANCE = 1e-10
_FIT_ITERATIONS = 200
_ROUNDING_SLACK = 1e-12
# Coefficients c_k = B_2k / (2k) of the asymptotic series of digamma,
# psi(x) = log(x) - 1 / (2x) - sum_k c_k / x^(2k) (DLMF 5.11.2).
_DIGAMMA_SERIES = np.array(
    [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        1.0 / 132.0,
        -691.0 / 32760.0,
        1.0 / 12.0,
        -3617.0 / 8160.0,
    ]
)


@dataclass(frozen=True, eq=False, repr=False)
class DESeq2Dispersions:
    """DESeq2's dispersion estimates of many genes, from :func:`deseq2_dispersions`.

    Arrays with a gene axis have one value per gene, in the order of the columns of
    the counts, and NaN (or False) for genes without counts; arrays with a sample axis
    are samples x genes. DESeq2's names are given in parentheses. The arrays are
    read-only.

    Attributes:
        dispersion: ``(n_features,)`` final estimates (``dispersion``): the maximum a
            posteriori estimate, or the gene-wise estimate of an outlier.
        genewise: ``(n_features,)`` gene-wise estimates (``dispGeneEst``): the
            maxima of the adjusted profile likelihood, or 1e-8 for genes whose rough
            dispersion is at that lower bound.
        trended: ``(n_features,)`` the trend at each gene's mean normalized count
            (``dispFit``).
        map_estimate: ``(n_features,)`` maximum a posteriori estimates
            (``dispMAP``).
        outlier: ``(n_features,)`` dispersion outliers (``dispOutlier``), which keep
            their gene-wise estimate.
        all_zero: ``(n_features,)`` genes without counts (``allZero``).
        base_mean: ``(n_features,)`` mean normalized counts (``baseMean``).
        base_variance: ``(n_features,)`` variances of the normalized counts
            (``baseVar``).
        means: ``(n_samples, n_features)`` the fitted means on which the gene-wise
            and maximum a posteriori estimates are conditional, raised to
            ``min_mean`` (DESeq2's ``mu`` after ``estimateDispersionsGeneEst``).
        size_factors: ``(n_samples,)`` size factors.
        trend: ``"parametric"``, ``"mean"``, or ``"given"`` for values of the trend
            given.
        asymptotic_dispersion: a0 of the trend a0 + a1 / mean (``asymptDisp``); the
            mean for the mean trend; NaN for a given trend.
        extra_poisson: a1 (``extraPois``); 0 for the mean trend, NaN for a given one.
        log_residual_variance: s_lr^2, the squared scaled median absolute deviation
            of the log residuals (``varLogDispEsts``); NaN when no gene has a gene-wise
            estimate of at least 1e-6, and then no gene is a dispersion outlier.
        prior_variance: sigma_d^2, the variance of the prior of the log dispersions
            (``dispPriorVar``).
        df_residual: Residual degrees of freedom, the number of samples less the
            number of coefficients.
        min_mean: The lower bound on the means.
    """

    dispersion: FloatArray
    genewise: FloatArray
    trended: FloatArray
    map_estimate: FloatArray
    outlier: BoolArray
    all_zero: BoolArray
    base_mean: FloatArray
    base_variance: FloatArray
    means: FloatArray
    size_factors: FloatArray
    trend: str
    asymptotic_dispersion: float
    extra_poisson: float
    log_residual_variance: float
    prior_variance: float
    df_residual: int
    min_mean: float

    def __repr__(self) -> str:
        n_samples, n_features = self.means.shape
        return (
            f"DESeq2Dispersions(n_features={n_features}, n_samples={n_samples}, "
            f"trend={self.trend!r}, prior_variance={self.prior_variance:.4g}, "
            f"outliers={int(np.count_nonzero(self.outlier))})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class DESeq2Fit:
    """DESeq2's negative binomial fits of many genes, from :func:`fit_deseq2`.

    Coefficients, their standard errors and covariances are on the log2 scale, as
    DESeq2 reports them. Arrays with a gene axis have one row per gene, in the order of
    the columns of the counts, and NaN for genes without counts; arrays with a sample
    axis are samples x genes. DESeq2's names are given in parentheses. The arrays are
    read-only.

    Attributes:
        coefficients: ``(n_features, n_coefficients)`` log2 coefficients, one column
            per design column.
        standard_errors: ``(n_features, n_coefficients)`` their standard errors
            (``SE_...``).
        covariance: ``(n_features, n_coefficients, n_coefficients)`` covariance
            matrices of the coefficients.
        fitted_values: ``(n_samples, n_features)`` fitted means (the ``mu`` assay).
        minus_twice_log_likelihood: ``(n_features,)`` -2 times the log-likelihood at
            the fitted means (``deviance``).
        hat_values: ``(n_samples, n_features)`` diagonals of the hat matrix (the
            ``H`` assay).
        cooks_distances: ``(n_samples, n_features)`` Cook's distances (the ``cooks``
            assay).
        max_cooks: ``(n_features,)`` the largest Cook's distance over the samples
            whose design row occurs three or more times (``maxCooks``); NaN when there
            are none.
        converged: ``(n_features,)`` whether each fit converged; True for genes
            without counts.
        iterations: ``(n_features,)`` Newton steps of each fit.
        dispersion: ``(n_features,)`` the dispersions of the fits.
        dispersion_estimates: The dispersion estimates of every stage, or None when
            the dispersions were given.
        size_factors: ``(n_samples,)`` size factors.
        base_mean: ``(n_features,)`` mean normalized counts (``baseMean``).
        counts: ``(n_samples, n_features)`` the counts.
        design: ``(n_samples, n_coefficients)`` design matrix.
        coefficient_names: Names of the coefficients, the design's columns.
        min_mean: The lower bound on the fitted values in the fits.
        replicated: ``(n_samples,)`` samples whose design row occurs three or more
            times, which count for ``max_cooks``.
        two_groups: Whether the design compares two groups of one factor, where a
            gene is not flagged by Cook's distance when three or more samples have
            larger counts than the sample with the largest distance.
    """

    coefficients: FloatArray
    standard_errors: FloatArray
    covariance: FloatArray
    fitted_values: FloatArray
    minus_twice_log_likelihood: FloatArray
    hat_values: FloatArray
    cooks_distances: FloatArray
    max_cooks: FloatArray
    converged: BoolArray
    iterations: IntArray
    dispersion: FloatArray
    dispersion_estimates: DESeq2Dispersions | None
    size_factors: FloatArray
    base_mean: FloatArray
    counts: FloatArray
    design: FloatArray
    coefficient_names: tuple[str, ...]
    min_mean: float
    replicated: BoolArray
    two_groups: bool

    def __repr__(self) -> str:
        n_samples, n_features = self.counts.shape
        n_converged = int(np.count_nonzero(self.converged))
        return (
            f"DESeq2Fit(n_features={n_features}, n_samples={n_samples}, "
            f"coefficient_names={self.coefficient_names!r}, "
            f"converged={n_converged} of {n_features})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class DESeq2Results:
    """One test of every gene, with the columns of DESeq2's ``results()``.

    Per-gene arrays have one value per gene, in the order of the columns of the
    counts; genes without counts have NaN in all of them. The arrays are read-only.

    Attributes:
        base_mean: ``(n_features,)`` mean normalized counts (``baseMean``).
        log2_fold_change: ``(n_features,)`` the contrast's log2 fold change
            (``log2FoldChange``); for the likelihood ratio test that of
            ``name``, a coefficient of the full design.
        standard_error: ``(n_features,)`` its standard error (``lfcSE``).
        statistic: ``(n_features,)`` the Wald statistic, or the likelihood ratio
            statistic (``stat``).
        p_value: ``(n_features,)`` p-values (``pvalue``); NaN for genes flagged by
            Cook's distance.
        adjusted_p_value: ``(n_features,)`` Benjamini-Hochberg adjusted p-values
            after independent filtering (``padj``); NaN for genes filtered out.
        test: ``"Wald"`` or ``"likelihood ratio"``.
        name: The name of the contrast or coefficient of ``log2_fold_change``.
        contrast: ``(n_coefficients,)`` its weights over the coefficients.
        df: Degrees of freedom of the likelihood ratio test; None for a Wald test.
        cooks_outlier: ``(n_features,)`` genes whose p-value Cook's distance removed.
        converged: ``(n_features,)`` whether the fit of each gene converged
            (``betaConv``, or ``fullBetaConv`` for the likelihood ratio test); True for
            genes without counts.
        reduced_converged: ``(n_features,)`` for the likelihood ratio test, whether
            the fit of the reduced design converged (``reducedBetaConv``); None for a
            Wald test.
        cooks_cutoff: The cutoff on Cook's distance; infinite when none is applied.
        alpha: The false discovery rate for which the filter was chosen.
        filter_threshold: The mean normalized count below which genes are filtered
            out (``filterThreshold``); NaN without independent filtering.
        filter_quantile: Its quantile among the genes (``filterTheta``); NaN
            without independent filtering.
        filter_quantiles: ``(50,)`` the quantiles tried; empty without independent
            filtering.
        rejections: ``(50,)`` the number of adjusted p-values below ``alpha`` at each
            (``filterNumRej``).
        rejections_fit: ``(50,)`` their LOWESS fit (``lo.fit``).
    """

    base_mean: FloatArray
    log2_fold_change: FloatArray
    standard_error: FloatArray
    statistic: FloatArray
    p_value: FloatArray
    adjusted_p_value: FloatArray
    test: str
    name: str
    contrast: FloatArray
    df: int | None
    cooks_outlier: BoolArray
    converged: BoolArray
    reduced_converged: BoolArray | None
    cooks_cutoff: float
    alpha: float
    filter_threshold: float
    filter_quantile: float
    filter_quantiles: FloatArray
    rejections: FloatArray
    rejections_fit: FloatArray

    def __repr__(self) -> str:
        significant = int(np.count_nonzero(self.adjusted_p_value < self.alpha))
        return (
            f"DESeq2Results(test={self.test!r}, name={self.name!r}, "
            f"n_features={self.p_value.size}, "
            f"adjusted_p_value_below_{self.alpha:g}={significant})"
        )


def deseq2_dispersions(
    counts: npt.ArrayLike,
    design: Any,
    *,
    size_factors: npt.ArrayLike | None = None,
    trend: str = "parametric",
    genewise_dispersions: npt.ArrayLike | None = None,
    trended_dispersions: npt.ArrayLike | None = None,
    prior_variance: float | None = None,
    min_mean: float = 0.5,
    n_threads: int | None = None,
) -> DESeq2Dispersions:
    """Estimate every gene's dispersion as DESeq2's ``estimateDispersions()``.

    The module description gives the steps: gene-wise estimates, a trend, a prior on
    the log dispersions around it, and maximum a posteriori estimates. Gene-wise
    estimates, values of the trend and the prior's variance may be given in
    place of their estimates, as DESeq2's lower-level functions allow
    (``estimateDispersionsGeneEst``, a custom ``dispersionFunction`` and the
    ``dispPriorVar`` of ``estimateDispersionsMAP``).

    Args:
        counts: ``(n_samples, n_features)`` counts: whole numbers, finite and
            non-negative, as DESeq2 requires.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank
            with fewer columns than samples, an object with ``matrix`` and
            ``columns`` attributes such as a ``Design``, or a table with
            ``columns``. A ``Design`` with random effects is refused.
        size_factors: ``(n_samples,)`` positive size factors. Defaults to the
            median-of-ratios factors of
            :func:`~cytepool.normalization.median_ratio_size_factors`, as
            ``estimateSizeFactors()`` gives them.
        trend: ``"parametric"`` (DESeq2's default) or ``"mean"``.
        genewise_dispersions: ``(n_features,)`` gene-wise estimates to use instead of
            estimating them: positive and finite for genes with counts, any value or
            NaN for the others.
        trended_dispersions: ``(n_features,)`` values of the trend to use instead of
            fitting one, positive and finite for genes with counts.
        prior_variance: The variance of the prior of the log dispersions, positive and
            finite, instead of its estimate.
        min_mean: The lower bound on the means of the likelihood, positive (DESeq2's
            ``minmu``).
        n_threads: The most threads for estimating genes in parallel. ``None`` uses
            numba's setting for the calling thread. The result does not depend on it,
            and with one thread no parallel region runs.

    Returns:
        The estimates of every stage, with the trend and the prior.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If shapes disagree; counts are negative, not finite or not whole
            numbers; no gene has counts; the design is not finite, has random effects,
            is not of full column rank or leaves no residual degrees of freedom; a
            size factor is not positive and finite, or every gene has a zero count when
            size factors are to be estimated; a given dispersion or prior variance is
            not positive and finite; ``trend`` is unknown; or, without
            ``prior_variance``, no gene has a gene-wise estimate of at least 1e-6 to
            estimate the prior from.
    """
    reject_masked(trend, "trend")
    reject_masked(prior_variance, "prior_variance")
    with thread_limit(n_threads):
        y = _whole_counts(counts)
        x, _ = _full_rank_design(design, y.shape[0], "deseq2_dispersions")
        s = _size_factors(size_factors, y)
        floor = _positive(min_mean, "min_mean")
        kind = _trend_kind(trend)
        given_genewise = _optional_per_gene(
            genewise_dispersions, "genewise_dispersions", y
        )
        given_trended = _optional_per_gene(
            trended_dispersions, "trended_dispersions", y
        )
        given_prior = (
            None
            if prior_variance is None
            else _positive(prior_variance, "prior_variance")
        )
        return _dispersions(
            y, x, s, floor, kind, given_genewise, given_trended, given_prior
        )


def fit_deseq2(
    counts: npt.ArrayLike,
    design: Any,
    *,
    size_factors: npt.ArrayLike | None = None,
    dispersion: npt.ArrayLike | DESeq2Dispersions | None = None,
    trend: str = "parametric",
    min_mean: float = 0.5,
    n_threads: int | None = None,
) -> DESeq2Fit:
    """Fit DESeq2's negative binomial model to every gene, as ``DESeq()`` does.

    Estimates the size factors and dispersions unless they are given, fits every gene
    as ``nbinomWaldTest()`` does, and computes the standard errors, hat values and
    Cook's distances; the module description gives the methods. Test the fit with
    :func:`deseq2_wald_test` or :func:`deseq2_likelihood_ratio_test`. Outlier counts
    are not replaced: the fit is that of ``DESeq(..., minReplicatesForReplace =
    Inf)``, which is DESeq2's default where no design row occurs seven or more times.

    Args:
        counts: ``(n_samples, n_features)`` counts: whole numbers, finite and
            non-negative.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank
            with fewer columns than samples, an object with ``matrix`` and
            ``columns`` attributes such as a ``Design``, or a table with
            ``columns``. A ``Design`` with random effects is refused.
        size_factors: ``(n_samples,)`` positive size factors. Defaults to those of
            ``dispersion`` when it is a :class:`DESeq2Dispersions`, else to the
            median-of-ratios factors.
        dispersion: The dispersions of the fits: those of
            :func:`deseq2_dispersions`, or ``(n_features,)`` values, positive and
            finite for genes with counts (NaN or any value for the others). Estimated
            with :func:`deseq2_dispersions` by default.
        trend: The trend of the dispersions when they are estimated,
            ``"parametric"`` or ``"mean"``.
        min_mean: The lower bound on the fitted values in the weights of the fits and
            on the means of the dispersion estimates, positive (``minmu``).
        n_threads: The most threads for fitting genes in parallel. ``None`` uses
            numba's setting for the calling thread. The result does not depend on it.

    Returns:
        The fit of every gene, with the dispersions it used.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As :func:`deseq2_dispersions`, and if given dispersions do not
            have one value per gene or are not positive and finite for genes with
            counts.
    """
    reject_masked(dispersion, "dispersion", missing="NaN")
    reject_masked(trend, "trend")
    with thread_limit(n_threads):
        y = _whole_counts(counts)
        n_samples, n_features = y.shape
        x, names = _full_rank_design(design, n_samples, "fit_deseq2")
        floor = _positive(min_mean, "min_mean")
        kind = _trend_kind(trend)
        estimates: DESeq2Dispersions | None = None
        if isinstance(dispersion, DESeq2Dispersions):
            estimates = dispersion
            if estimates.dispersion.shape != (n_features,):
                raise ValueError(
                    f"dispersion holds estimates of {estimates.dispersion.size} genes "
                    f"but the counts have {n_features}"
                )
            if size_factors is None:
                size_factors = estimates.size_factors
        s = _size_factors(size_factors, y)
        if dispersion is None:
            estimates = _dispersions(y, x, s, floor, kind, None, None, None)
            alpha = estimates.dispersion
        elif estimates is not None:
            alpha = _per_gene(estimates.dispersion, "dispersion", y)
        else:
            alpha = _per_gene(dispersion, "dispersion", y)
        return _fit(y, x, names, s, alpha, floor, estimates, design)


def deseq2_wald_test(
    fit: DESeq2Fit,
    contrast: Any = None,
    *,
    alpha: float = 0.1,
    independent_filtering: bool = True,
    cooks_cutoff: float | bool | None = None,
) -> DESeq2Results:
    """Wald tests of a coefficient or contrast, as DESeq2's ``results()`` gives them.

    Args:
        fit: The fit, from :func:`fit_deseq2`.
        contrast: What to test: a coefficient name (DESeq2's ``name``), an expression
            in coefficient names such as ``"groupC - groupB"``, weights over the
            coefficients (a vector in their order, or a mapping of one contrast name to
            such weights or an expression), or a ``Contrasts`` holding one contrast,
            such as ``design.contrasts.select("stim_vs_ctrl")``. Defaults to the last
            coefficient, which ``results()`` tests by default when the design ends with
            a factor or a covariate.
        alpha: The false discovery rate for which the independent filter is chosen,
            in (0, 1).
        independent_filtering: Whether to filter genes on their mean normalized count
            before adjusting the p-values.
        cooks_cutoff: The cutoff on Cook's distance above which a gene's p-value is
            removed. ``None`` uses the 0.99 quantile of F(p, n - p); ``False`` or
            infinity applies none.

    Returns:
        The test of every gene.

    Raises:
        TypeError: If ``fit`` is not a :class:`DESeq2Fit`, an argument has the wrong
            type, or is or holds a numpy masked array.
        ValueError: If ``contrast`` names no coefficient, does not hold exactly one
            contrast, has the wrong length or is zero on every coefficient, or an
            option is out of range.
    """
    _check_fit(fit)
    weights, name = _contrast(contrast, fit.coefficient_names)
    settings = _results_settings(alpha, independent_filtering, cooks_cutoff, fit)
    lfc = fit.coefficients @ weights
    se = np.sqrt(np.einsum("j,gjk,k->g", weights, fit.covariance, weights))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = lfc / se
    p_value = np.asarray(2.0 * norm_sf(np.abs(statistic)), dtype=np.float64)
    zero = _zero_contrast(fit, weights)
    lfc[zero] = 0.0
    statistic[zero] = 0.0
    p_value[zero] = 1.0
    return _results(
        fit, lfc, se, statistic, p_value, "Wald", name, weights, None, settings, None
    )


def deseq2_likelihood_ratio_test(
    fit: DESeq2Fit,
    reduced: Any,
    *,
    coefficient: str | None = None,
    alpha: float = 0.1,
    independent_filtering: bool = True,
    cooks_cutoff: float | bool | None = None,
    n_threads: int | None = None,
) -> DESeq2Results:
    """Likelihood ratio tests of a reduced design, as ``nbinomLRT()`` and ``results()``.

    The reduced design is fitted with the dispersions and size factors of the full fit,
    as ``nbinomLRT()`` fits it. The statistic, the difference of the deviances, tests
    all the coefficients the reduced design leaves out at once, such as every level of
    a factor or every interaction.

    Args:
        fit: The fit of the full design, from :func:`fit_deseq2`.
        reduced: The reduced design: a matrix of full column rank whose columns lie in
            the column space of the full design, with fewer columns, or an object with
            ``matrix`` and ``columns`` attributes such as a ``Design`` built from the
            same samples (``Design(samples, "~ batch")``). A formula alone is refused,
            as the fit keeps the design matrix and not the samples.
        coefficient: The coefficient of the full design whose log2 fold change is
            reported; defaults to the last.
        alpha: The false discovery rate for which the independent filter is chosen,
            in (0, 1).
        independent_filtering: Whether to filter genes on their mean normalized count
            before adjusting the p-values.
        cooks_cutoff: The cutoff on Cook's distance, as for :func:`deseq2_wald_test`.
        n_threads: The most threads for fitting the reduced design. ``None`` uses
            numba's setting for the calling thread.

    Returns:
        The test of every gene.

    Raises:
        TypeError: If ``fit`` is not a :class:`DESeq2Fit`, ``reduced`` is a formula,
            an argument has the wrong type, or is or holds a numpy masked array.
        ValueError: If the reduced design has the wrong number of rows, is not finite,
            not of full column rank, not nested in the full design or not smaller than
            it, has random effects, or ``coefficient`` is not a coefficient name.
    """
    _check_fit(fit)
    reject_masked(reduced, "reduced")
    reject_masked(coefficient, "coefficient")
    with thread_limit(n_threads):
        x_reduced = _reduced_design(reduced, fit)
        if coefficient is None:
            position = len(fit.coefficient_names) - 1
        elif isinstance(coefficient, str) and coefficient in fit.coefficient_names:
            position = fit.coefficient_names.index(coefficient)
        else:
            raise ValueError(
                f"coefficient must be one of the full design's coefficients "
                f"{fit.coefficient_names}, got {coefficient!r}"
            )
        settings = _results_settings(alpha, independent_filtering, cooks_cutoff, fit)
        counted = ~np.isnan(fit.dispersion)
        reduced_values = np.full(fit.fitted_values.shape, np.nan)
        reduced_converged = np.ones(fit.dispersion.size, dtype=np.bool_)
        _, reduced_values[:, counted], reduced_converged[counted], _ = _fitted_means(
            fit.counts[:, counted],
            x_reduced,
            fit.size_factors,
            fit.dispersion[counted],
            fit.min_mean,
        )
    statistic = np.full(fit.dispersion.size, np.nan)
    phi = fit.dispersion[counted]
    y = fit.counts[:, counted]
    statistic[counted] = negative_binomial_deviance(
        y, reduced_values[:, counted], phi
    ) - negative_binomial_deviance(y, fit.fitted_values[:, counted], phi)
    df = fit.design.shape[1] - x_reduced.shape[1]
    p_value = np.full(statistic.size, np.nan)
    p_value[counted] = chi2_sf(statistic[counted], float(df))
    weights = np.zeros(len(fit.coefficient_names))
    weights[position] = 1.0
    lfc = fit.coefficients[:, position].copy()
    se = fit.standard_errors[:, position].copy()
    name = fit.coefficient_names[position]
    return _results(
        fit,
        lfc,
        se,
        statistic,
        p_value,
        "likelihood ratio",
        name,
        weights,
        df,
        settings,
        reduced_converged,
    )


# Dispersions.


def _dispersions(
    y: FloatArray,
    x: FloatArray,
    s: FloatArray,
    floor: float,
    kind: str,
    given_genewise: FloatArray | None,
    given_trended: FloatArray | None,
    given_prior: float | None,
) -> DESeq2Dispersions:
    """Every stage of the dispersion estimates; the arguments have been checked."""
    n_samples, n_features = y.shape
    p = x.shape[1]
    df = n_samples - p
    counted = (y > 0.0).any(axis=0)
    if not counted.any():
        raise ValueError("no gene has a positive count, so no dispersion is estimated")
    normalized = y / s[:, np.newaxis]
    base_mean, base_variance = _moments(normalized)
    low, high = _MIN_DISPERSION, max(_MAX_DISPERSION, float(n_samples))
    kept = y[:, counted]
    rough = _rough_dispersions(normalized[:, counted], x, s, high)
    means = _likelihood_means(kept, normalized[:, counted], x, s, floor, rough)
    means_by_gene = np.ascontiguousarray(means.T)
    counts_by_gene = np.ascontiguousarray(kept.T)
    zeros = np.zeros(counts_by_gene.shape[0])
    if given_genewise is None:
        genewise = _maximize(counts_by_gene, means_by_gene, x, low, high, zeros, 0.0)
        # A gene whose moment estimates show no overdispersion keeps the lower bound,
        # as in DESeq2, whose search starts there (found by probing).
        genewise[rough <= low] = low
    else:
        genewise = given_genewise[counted]
    mean_counts = base_mean[counted]
    if given_trended is not None:
        trended = given_trended[counted]
        used, a0, a1 = "given", math.nan, math.nan
    else:
        used, a0, a1 = _trend(genewise, mean_counts, kind)
        trended = a0 + a1 / mean_counts
    residuals = np.log(genewise) - np.log(trended)
    usable = genewise >= _PRIOR_GENES * _MIN_DISPERSION
    if usable.any():
        centred = residuals[usable] - np.median(residuals[usable])
        log_residual_variance = float((_MAD_SCALE * np.median(np.abs(centred))) ** 2)
    elif given_prior is None:
        raise ValueError(
            "no gene has a gene-wise dispersion estimate of at least "
            f"{_PRIOR_GENES * _MIN_DISPERSION:g}, so the prior's width cannot be "
            "estimated; pass prior_variance"
        )
    else:
        # Without the spread of the log residuals no gene is a dispersion outlier.
        log_residual_variance = math.nan
    if given_prior is not None:
        prior = given_prior
    elif df > _FEW_DF:
        sampling = float(trigamma(df / 2.0))
        prior = max(log_residual_variance - sampling, _PRIOR_FLOOR)
    else:
        prior = _few_df_prior_variance(residuals[usable], df)
    prior_means = np.log(trended)
    posterior = _maximize(
        counts_by_gene, means_by_gene, x, low, high, prior_means, 1.0 / prior
    )
    outlier = residuals > _OUTLIER_SDS * math.sqrt(log_residual_variance)
    final = np.where(outlier, genewise, posterior)

    def spread(values: npt.NDArray[Any], fill: Any) -> npt.NDArray[Any]:
        full = np.full(n_features, fill, dtype=values.dtype)
        full[counted] = values
        return _read_only(full)

    full_means = np.full((n_samples, n_features), np.nan)
    full_means[:, counted] = means
    return DESeq2Dispersions(
        dispersion=spread(final, np.nan),
        genewise=spread(np.asarray(genewise, dtype=np.float64), np.nan),
        trended=spread(np.asarray(trended, dtype=np.float64), np.nan),
        map_estimate=spread(posterior, np.nan),
        outlier=spread(outlier, False),
        all_zero=_read_only(~counted),
        base_mean=_read_only(base_mean),
        base_variance=_read_only(base_variance),
        means=_read_only(full_means),
        size_factors=_read_only(s.copy()),
        trend=used,
        asymptotic_dispersion=float(a0),
        extra_poisson=float(a1),
        log_residual_variance=log_residual_variance,
        prior_variance=float(prior),
        df_residual=df,
        min_mean=floor,
    )


def _moments(normalized: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean and variance (n - 1 in the denominator) of every column."""
    mean = _column_means(normalized)
    variance = np.sum((normalized - mean) ** 2, axis=0) / (normalized.shape[0] - 1)
    return mean, variance


def _column_means(values: FloatArray) -> FloatArray:
    """Means of the columns from sums accumulated with error compensation, as R's
    rowMeans() accumulates them in extended precision."""
    means: FloatArray = accurate_sums(np.ascontiguousarray(values), 0) / values.shape[0]
    return means


def _likelihood_means(
    y: FloatArray,
    normalized: FloatArray,
    x: FloatArray,
    s: FloatArray,
    floor: float,
    rough: FloatArray,
) -> FloatArray:
    """The means of the dispersion likelihood, samples x genes, raised to ``floor``."""
    distinct, groups = np.unique(x, axis=0, return_inverse=True)
    groups = groups.reshape(-1)
    if distinct.shape[0] == x.shape[1]:
        # A one-way layout: the linear model's fitted values are the group means of
        # the normalized counts.
        means = np.empty_like(normalized)
        for h in range(distinct.shape[0]):
            rows = groups == h
            means[rows] = normalized[rows].mean(axis=0)
        means *= s[:, np.newaxis]
    else:
        means = _fitted_means(y, x, s, rough, floor)[1]
    result: FloatArray = np.maximum(means, floor)
    return result


def _rough_dispersions(
    normalized: FloatArray, x: FloatArray, s: FloatArray, high: float
) -> FloatArray:
    """DESeq2's starting dispersions: the smaller of two moment estimates (found by
    probing), within [1e-8, high]."""
    n_samples, p = x.shape
    coefficients, *_ = np.linalg.lstsq(x, normalized, rcond=None)
    fitted = np.maximum(x @ coefficients, 1.0)
    from_residuals = np.maximum(
        np.sum(((normalized - fitted) ** 2 - fitted) / fitted**2, axis=0)
        / (n_samples - p),
        0.0,
    )
    mean = normalized.mean(axis=0)
    variance = normalized.var(axis=0, ddof=1)
    from_moments = (variance - np.mean(1.0 / s) * mean) / mean**2
    rough: FloatArray = np.clip(
        np.minimum(from_residuals, from_moments), _MIN_DISPERSION, high
    )
    return rough


def _maximize(
    counts: FloatArray,
    means: FloatArray,
    x: FloatArray,
    low: float,
    high: float,
    prior_means: FloatArray,
    precision: float,
) -> FloatArray:
    """The dispersion that maximizes each gene's adjusted log-likelihood plus, with a
    positive precision, its log prior; genes are rows of counts and means."""
    arguments = (
        counts,
        means,
        np.ascontiguousarray(x),
        low,
        high,
        np.ascontiguousarray(prior_means, dtype=np.float64),
        float(precision),
    )
    threads = numba.get_num_threads()
    if threads > 1 and counts.shape[0] > 1:
        n_chunks = max(1, min(counts.shape[0], 4 * threads))
        result: FloatArray = _maximize_in_parallel(*arguments, n_chunks)
    else:
        result = _maximize_in_order(*arguments)
    return result


def _trend(
    genewise: FloatArray, mean_counts: FloatArray, kind: str
) -> tuple[str, float, float]:
    """The kind of trend used and its coefficients a0 and a1."""
    if kind == "parametric":
        use = genewise > _TREND_GENES * _MIN_DISPERSION
        coefficients = _parametric_trend(genewise[use], mean_counts[use])
        if coefficients is not None:
            return "parametric", coefficients[0], coefficients[1]
        warnings.warn(
            "the parametric dispersion trend a0 + a1 / mean did not converge with "
            "positive coefficients; DESeq2 substitutes a local regression, which "
            "CytePool does not implement, so the mean of the gene-wise dispersions is "
            "used as the trend; pass trend='mean' to choose it, or "
            "trended_dispersions",
            RuntimeWarning,
            stacklevel=4,
        )
    use = genewise > _MEAN_GENES * _MIN_DISPERSION
    if not use.any():
        raise ValueError(
            "no gene has a gene-wise dispersion estimate above "
            f"{_MEAN_GENES * _MIN_DISPERSION:g}, so no trend can be fitted; pass "
            "trended_dispersions"
        )
    return "mean", _trimmed_mean(genewise[use], _MEAN_TRIM), 0.0


def _parametric_trend(
    genewise: FloatArray, mean_counts: FloatArray
) -> tuple[float, float] | None:
    """DESeq2's iteration of gamma fits for a0 + a1 / mean, to the genes given; None
    where it fails.

    Each fit starts from the coefficients of the one before, and the iteration stops
    when the squared logarithmic changes of the coefficients add up to less than 1e-6
    after a fit that converged; it fails after 11 fits, or when a fit fails or gives a
    coefficient that is not positive (found by probing).
    """
    x = 1.0 / mean_counts
    coefficients = np.array(_TREND_START)
    low, high = _TREND_RATIOS
    for _ in range(_TREND_FITS):
        ratio = genewise / (coefficients[0] + coefficients[1] * x)
        use = (ratio > low) & (ratio < high)
        if np.count_nonzero(use) < 2:
            return None
        fitted = _gamma_identity_fit(genewise[use], x[use], coefficients)
        if fitted is None:
            return None
        updated, converged = fitted
        if not np.all(updated > 0.0):
            return None
        change = float(np.sum(np.log(updated / coefficients) ** 2))
        coefficients = updated
        if change < _TREND_TOLERANCE and converged:
            return float(coefficients[0]), float(coefficients[1])
    return None


def _gamma_identity_fit(
    y: FloatArray, x: FloatArray, start: FloatArray
) -> tuple[FloatArray, bool] | None:
    """A gamma-family fit of y on b0 + b1 x with identity link, as R's ``glm()`` fits
    it from ``start``: and whether it converged; None if no step keeps the means
    positive.

    Iteratively reweighted least squares, the working response y and the weights 1 /
    mu^2, stopped when the deviance changes by less than 1e-8 of it (plus 0.1) or after
    25 steps, the convergence criterion and step limit of ``glm.control()``. A step
    that makes a mean non-positive, where the deviance is not finite, is halved
    towards the coefficients before it until it does not (found by probing with
    ``glm.control(trace = TRUE)``).
    """
    design = np.column_stack([np.ones_like(x), x])
    beta = start.astype(np.float64)
    deviance = _gamma_deviance(y, design @ beta)
    if not math.isfinite(deviance):
        return None
    for _ in range(_GLM_STEPS):
        mean = design @ beta
        root = 1.0 / mean
        q, r = np.linalg.qr(design * root[:, np.newaxis])
        updated = np.linalg.solve(r, q.T @ (y * root))
        updated_deviance = _gamma_deviance(y, design @ updated)
        for _ in range(_GLM_HALVINGS):
            if math.isfinite(updated_deviance):
                break
            updated = 0.5 * (updated + beta)
            updated_deviance = _gamma_deviance(y, design @ updated)
        else:
            return None
        converged = (
            abs(updated_deviance - deviance) / (abs(updated_deviance) + 0.1)
            < _GLM_TOLERANCE
        )
        beta, deviance = updated, updated_deviance
        if converged:
            return beta, True
    return beta, False


def _gamma_deviance(y: FloatArray, mean: FloatArray) -> float:
    """The gamma deviance, 2 sum(-log(y / mu) + (y - mu) / mu); infinite where a mean
    is not positive."""
    if not np.all(mean > 0.0):
        return math.inf
    return float(np.sum(2.0 * (-np.log(y / mean) + (y - mean) / mean)))


def _few_df_prior_variance(residuals: FloatArray, df: int) -> float:
    """The maximum likelihood estimate of sigma^2 in residual = log(chi^2_df / df) +
    N(0, sigma^2), within [0, 8] and at least 0.25."""
    shape = df / 2.0
    log_gamma = float(lgamma_scalar(shape))
    values = np.ascontiguousarray(residuals, dtype=np.float64)

    def score(variance: float) -> float:
        result: float = _variance_score(values, variance, shape, log_gamma)
        return result

    low, high = _PRIOR_FLOOR, _LARGEST_PRIOR_VARIANCE
    at_low, at_high = score(low), score(high)
    # The score of the variance is E[(r - u)^2 | r] - sigma^2 summed over residuals;
    # the likelihood rises up to its root.
    if at_low <= 0.0:
        return low
    if at_high >= 0.0:
        return high
    return _illinois(score, low, high, at_low, at_high, 1e-13)


def _illinois(
    function: Any, a: float, b: float, fa: float, fb: float, tolerance: float
) -> float:
    """The root of a decreasing function in [a, b], fa > 0 > fb, by the Illinois
    variant of false position."""
    side = 0
    for _ in range(_ROOT_ITERATIONS):
        if b - a <= tolerance * max(1.0, abs(a)):
            break
        c = b - fb * (b - a) / (fb - fa)
        if not a < c < b:
            c = 0.5 * (a + b)
        fc = function(c)
        if fc > 0.0:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
        elif fc < 0.0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            return c
    return b - fb * (b - a) / (fb - fa)


def _trimmed_mean(values: FloatArray, trim: float) -> float:
    """R's mean(values, trim = trim): floor(n trim) values dropped at each end, the
    rest summed exactly."""
    ordered = np.sort(values)
    k = math.floor(ordered.size * trim)
    kept = ordered[k : ordered.size - k]
    return math.fsum(kept.tolist()) / kept.size


# Fits.


def _fit(
    y: FloatArray,
    x: FloatArray,
    names: tuple[str, ...],
    s: FloatArray,
    alpha: FloatArray,
    floor: float,
    estimates: DESeq2Dispersions | None,
    design: Any,
) -> DESeq2Fit:
    """Fit every gene with counts, and the quantities derived from the fits."""
    n_samples, n_features = y.shape
    p = x.shape[1]
    counted = (y > 0.0).any(axis=0)
    natural = np.full((n_features, p), np.nan)
    fitted = np.full((n_samples, n_features), np.nan)
    converged = np.ones(n_features, dtype=np.bool_)
    iterations = np.zeros(n_features, dtype=np.int64)
    if counted.any():
        (
            natural[counted],
            fitted[:, counted],
            converged[counted],
            iterations[counted],
        ) = _fitted_means(y[:, counted], x, s, alpha[counted], floor)
    covariance, hat = _covariance_and_hat(fitted, x, alpha, floor)
    scale = 1.0 / _LOG2
    normalized = y / s[:, np.newaxis]
    base_mean = _column_means(normalized)
    replicated = _replicated(x)
    cooks = _cooks_distances(y, normalized, fitted, hat, x, counted)
    max_cooks = np.full(n_features, np.nan)
    if replicated.any():
        max_cooks = np.max(cooks[replicated], axis=0)
    log2_covariance = covariance * scale * scale
    standard_errors = np.sqrt(np.diagonal(log2_covariance, axis1=1, axis2=2))
    return DESeq2Fit(
        coefficients=_read_only(natural * scale),
        standard_errors=_read_only(np.ascontiguousarray(standard_errors)),
        covariance=_read_only(log2_covariance),
        fitted_values=_read_only(fitted),
        minus_twice_log_likelihood=_read_only(
            _minus_twice_log_likelihood(y, fitted, alpha, counted)
        ),
        hat_values=_read_only(hat),
        cooks_distances=_read_only(cooks),
        max_cooks=_read_only(max_cooks),
        converged=_read_only(converged),
        iterations=_read_only(iterations),
        dispersion=_read_only(np.where(counted, alpha, np.nan)),
        dispersion_estimates=estimates,
        size_factors=_read_only(s.copy()),
        base_mean=_read_only(base_mean),
        counts=_read_only(y),
        design=_read_only(x),
        coefficient_names=names,
        min_mean=floor,
        replicated=_read_only(replicated),
        two_groups=_two_groups(x, design),
    )


def _fitted_means(
    y: FloatArray, x: FloatArray, s: FloatArray, alpha: FloatArray, floor: float
) -> tuple[FloatArray, FloatArray, BoolArray, IntArray]:
    """DESeq2's fit of every gene (columns of y): natural log coefficients (genes x
    coefficients), fitted values (samples x genes), convergence and steps."""
    n_samples, n_genes = y.shape
    normalized = y / s[:, np.newaxis]
    if x.shape[1] == 1 and np.all(x == 1.0):
        # An intercept alone: the log mean normalized count (found by probing).
        level = np.log(_column_means(normalized))
        coefficients = level[:, np.newaxis]
        fitted = s[:, np.newaxis] * np.exp(level)[np.newaxis, :]
        return (
            coefficients,
            fitted,
            np.ones(n_genes, dtype=np.bool_),
            np.zeros(n_genes, dtype=np.int64),
        )
    start, *_ = np.linalg.lstsq(x, np.log(normalized + 0.1), rcond=None)
    arguments = (
        np.ascontiguousarray(y.T),
        np.log(s),
        np.ascontiguousarray(x),
        np.ascontiguousarray(alpha, dtype=np.float64),
        float(floor),
        np.full(x.shape[1], _NATURAL_RIDGE),
        np.ascontiguousarray(start.T),
    )
    threads = numba.get_num_threads()
    if threads > 1 and n_genes > 1:
        n_chunks = max(1, min(n_genes, 4 * threads))
        coefficients, fitted_by_gene, converged, iterations = _fit_in_parallel(
            *arguments, n_chunks
        )
    else:
        coefficients, fitted_by_gene, converged, iterations = _fit_in_order(*arguments)
    return coefficients, np.ascontiguousarray(fitted_by_gene.T), converged, iterations


def _covariance_and_hat(
    fitted: FloatArray, x: FloatArray, alpha: FloatArray, floor: float
) -> tuple[FloatArray, FloatArray]:
    """Covariance of the natural log coefficients and hat values; NaN for genes
    without fits."""
    p = x.shape[1]
    raised = np.maximum(fitted, floor).T  # genes x samples
    with np.errstate(invalid="ignore"):
        weights = raised / (1.0 + alpha[:, np.newaxis] * raised)
    good = np.isfinite(weights).all(axis=1)
    covariance = np.full((weights.shape[0], p, p), np.nan)
    hat = np.full(fitted.shape, np.nan)
    if good.any():
        w = weights[good]
        information = np.einsum("gi,ij,ik->gjk", w, x, x)
        inverse = np.linalg.inv(information + _NATURAL_RIDGE * np.eye(p))
        covariance[good] = inverse @ information @ inverse
        hat[:, good] = (w * np.einsum("ij,gjk,ik->gi", x, inverse, x)).T
    return covariance, hat


def _minus_twice_log_likelihood(
    y: FloatArray, fitted: FloatArray, alpha: FloatArray, counted: BoolArray
) -> FloatArray:
    """-2 log-likelihood of each gene with counts at its fitted values; NaN otherwise."""
    result = np.full(y.shape[1], np.nan)
    if counted.any():
        result[counted] = _log_likelihoods(
            np.ascontiguousarray(y[:, counted].T),
            np.ascontiguousarray(fitted[:, counted].T),
            np.ascontiguousarray(alpha[counted]),
        )
    return result


def _replicated(x: FloatArray) -> BoolArray:
    """Samples whose design row occurs at least three times."""
    _, groups, sizes = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    result: BoolArray = sizes[groups.reshape(-1)] >= _REPLICATES
    return result


def _two_groups(x: FloatArray, design: Any) -> bool:
    """Whether a design compares two groups of one factor: two distinct rows of zeros
    and ones and, for a ``Design``, a single term that is a factor (found by probing:
    DESeq2 does not count a numeric covariate with two values)."""
    distinct = np.unique(x, axis=0)
    if distinct.shape[0] != 2 or not np.all((x == 0.0) | (x == 1.0)):
        return False
    if isinstance(design, Design):
        return len(design.terms) == 1 and design.terms[0] in design.levels
    return True


def _cooks_distances(
    y: FloatArray,
    normalized: FloatArray,
    fitted: FloatArray,
    hat: FloatArray,
    x: FloatArray,
    counted: BoolArray,
) -> FloatArray:
    """Cook's distances, samples x genes, at DESeq2's robust dispersions."""
    n_samples, n_features = y.shape
    cooks = np.full((n_samples, n_features), np.nan)
    if not counted.any():
        return cooks
    robust = _robust_dispersions(normalized[:, counted], x)
    mu = fitted[:, counted]
    pearson = (y[:, counted] - mu) ** 2 / (mu + robust * mu**2)
    h = hat[:, counted]
    cooks[:, counted] = pearson / x.shape[1] * h / (1.0 - h) ** 2
    return cooks


def _robust_dispersions(normalized: FloatArray, x: FloatArray) -> FloatArray:
    """DESeq2's method-of-moments dispersions from trimmed variances (found by
    probing, with PyDESeq2's constants)."""
    _, groups, sizes = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    groups = groups.reshape(-1)
    replicated = np.flatnonzero(sizes >= _REPLICATES)
    if replicated.size:
        variance = np.full(normalized.shape[1], -np.inf)
        for h in replicated:
            cell = normalized[groups == h]
            variance = np.maximum(variance, _trimmed_variance(cell))
    else:
        variance = _trimmed_variance(normalized, index=2)
    mean = normalized.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        robust = (variance - mean) / mean**2
    result: FloatArray = np.maximum(robust, _ROBUST_DISPERSION_FLOOR)
    return result


def _trimmed_variance(values: FloatArray, index: int | None = None) -> FloatArray:
    """A scaled trimmed mean of squared deviations from the trimmed mean, by column."""
    n = values.shape[0]
    if index is None:
        index = 2 if n >= 23.5 else 1 if n >= 3.5 else 0
    trim = _TRIMMED_FRACTIONS[index]
    center = _trimmed_column_means(values, trim)
    squares = (values - center) ** 2
    result: FloatArray = _TRIMMED_SCALES[index] * _trimmed_column_means(squares, trim)
    return result


def _trimmed_column_means(values: FloatArray, trim: float) -> FloatArray:
    n = values.shape[0]
    k = math.floor(n * trim)
    ordered = np.sort(values, axis=0)[k : n - k]
    result: FloatArray = ordered.mean(axis=0)
    return result


# Tests.


def _contrast(contrast: Any, names: tuple[str, ...]) -> tuple[FloatArray, str]:
    """The weights of the contrast to test over the coefficients, and its name."""
    reject_masked(contrast, "contrast")
    if contrast is None:
        weights = np.zeros(len(names))
        weights[-1] = 1.0
        return weights, names[-1]
    if isinstance(contrast, str):
        if contrast in names:
            weights = np.zeros(len(names))
            weights[names.index(contrast)] = 1.0
            return weights, contrast
        contrast = make_contrasts({contrast: contrast}, names)
    elif isinstance(contrast, Mapping):
        contrast = make_contrasts(contrast, names)
    matrix, contrast_names = _contrast_matrix(contrast, None, names)
    if matrix.shape[1] != 1:
        raise ValueError(
            "contrast must hold one contrast, as results() tests one at a time; got "
            f"{matrix.shape[1]}: {contrast_names}"
        )
    if not np.any(matrix[:, 0] != 0.0):
        raise ValueError(
            f"contrast {contrast_names[0]!r} is zero on every coefficient, so it "
            "compares nothing"
        )
    return np.ascontiguousarray(matrix[:, 0]), contrast_names[0]


def _zero_contrast(fit: DESeq2Fit, weights: FloatArray) -> BoolArray:
    """Genes whose counts are all zero in the samples a two-sided contrast compares
    (found by probing)."""
    positive, negative = weights > 0.0, weights < 0.0
    if not (positive.any() and negative.any()):
        return np.zeros(fit.counts.shape[1], dtype=np.bool_)
    design = fit.design
    samples = (design[:, positive] != 0.0).any(axis=1) | (
        design[:, negative] != 0.0
    ).any(axis=1)
    result: BoolArray = ~(fit.counts[samples] > 0.0).any(axis=0) & ~np.isnan(
        fit.dispersion
    )
    return result


@dataclass(frozen=True)
class _Settings:
    alpha: float
    filtering: bool
    cutoff: float


def _results_settings(
    alpha: Any, filtering: Any, cutoff: Any, fit: DESeq2Fit
) -> _Settings:
    reject_masked(alpha, "alpha")
    reject_masked(filtering, "independent_filtering")
    reject_masked(cutoff, "cooks_cutoff")
    level = _real(alpha, "alpha")
    if not 0.0 < level < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {level}")
    if not isinstance(filtering, (bool, np.bool_)):
        raise TypeError(
            "independent_filtering must be True or False, got "
            f"{type(filtering).__name__}"
        )
    n_samples, p = fit.design.shape
    if cutoff is None:
        value = float(f_ppf(_COOKS_QUANTILE, float(p), float(n_samples - p)))
    elif cutoff is False:
        value = math.inf
    elif cutoff is True:
        raise TypeError(
            "cooks_cutoff must be None (the default cutoff), False (none) or a number, "
            "got True"
        )
    else:
        value = _real(cutoff, "cooks_cutoff", finite=False)
        if not value > 0.0:
            raise ValueError(f"cooks_cutoff must be positive, got {value}")
    return _Settings(level, bool(filtering), value)


def _results(
    fit: DESeq2Fit,
    lfc: FloatArray,
    se: FloatArray,
    statistic: FloatArray,
    p_value: FloatArray,
    test: str,
    name: str,
    weights: FloatArray,
    df: int | None,
    settings: _Settings,
    reduced_converged: BoolArray | None,
) -> DESeq2Results:
    """results(): Cook's distance flags, independent filtering and adjustment."""
    empty = np.isnan(fit.dispersion)
    for values in (lfc, se, statistic, p_value):
        values[empty] = np.nan
    outlier = _cooks_outliers(fit, settings.cutoff)
    p_value = np.where(outlier, np.nan, p_value)
    base_mean = np.asarray(fit.base_mean, dtype=np.float64)
    if settings.filtering:
        adjusted, threshold, quantile, quantiles, rejections, fitted = (
            _independent_filtering(p_value, base_mean, settings.alpha)
        )
    else:
        adjusted = np.full(p_value.size, np.nan)
        tested = ~np.isnan(p_value)
        if tested.any():
            adjusted[tested] = adjust_p_values(p_value[tested], "bh")
        threshold = quantile = math.nan
        quantiles = rejections = fitted = np.empty(0)
    return DESeq2Results(
        base_mean=_read_only(base_mean.copy()),
        log2_fold_change=_read_only(lfc),
        standard_error=_read_only(se),
        statistic=_read_only(statistic),
        p_value=_read_only(p_value),
        adjusted_p_value=_read_only(adjusted),
        test=test,
        name=name,
        contrast=_read_only(weights),
        df=df,
        cooks_outlier=_read_only(outlier),
        converged=fit.converged,
        reduced_converged=(
            None if reduced_converged is None else _read_only(reduced_converged)
        ),
        cooks_cutoff=settings.cutoff,
        alpha=settings.alpha,
        filter_threshold=threshold,
        filter_quantile=quantile,
        filter_quantiles=_read_only(quantiles),
        rejections=_read_only(rejections),
        rejections_fit=_read_only(fitted),
    )


def _cooks_outliers(fit: DESeq2Fit, cutoff: float) -> BoolArray:
    """Genes whose largest Cook's distance over replicated samples exceeds the cutoff,
    less those spared in a comparison of two groups."""
    with np.errstate(invalid="ignore"):
        outlier: BoolArray = np.asarray(fit.max_cooks > cutoff)
    if fit.two_groups and outlier.any():
        genes = np.flatnonzero(outlier)
        cooks = fit.cooks_distances[:, genes]
        largest = np.argmax(cooks, axis=0)
        counts = fit.counts[:, genes]
        higher = np.sum(counts > counts[largest, np.arange(genes.size)], axis=0)
        outlier = outlier.copy()
        outlier[genes] = higher < _REPLICATES
    return outlier


def _independent_filtering(
    p_value: FloatArray, base_mean: FloatArray, alpha: float
) -> tuple[FloatArray, float, float, FloatArray, FloatArray, FloatArray]:
    """results()'s filter on the mean normalized count (found by probing)."""
    lower = float(np.mean(base_mean == 0.0))
    upper = _UPPER_QUANTILE if lower < _UPPER_QUANTILE else 1.0
    quantiles = np.linspace(lower, upper, _FILTER_QUANTILES)
    cutoffs = np.quantile(base_mean, quantiles)
    tested = ~np.isnan(p_value)
    rejections = np.zeros(_FILTER_QUANTILES)
    adjusted_at: list[FloatArray] = []
    for k, cutoff in enumerate(cutoffs):
        use = tested & (base_mean >= cutoff)
        adjusted = np.full(p_value.size, np.nan)
        if use.any():
            adjusted[use] = adjust_p_values(p_value[use], "bh")
        adjusted_at.append(adjusted)
        rejections[k] = np.count_nonzero(adjusted < alpha)
    fitted = lowess(quantiles, rejections, span=_FILTER_SPAN)
    chosen = 0
    if rejections.max() > _FEWEST_REJECTIONS:
        positive = rejections > 0
        residuals = rejections[positive] - fitted[positive]
        top = float(fitted.max())
        thresholds = [top - math.sqrt(float(np.mean(residuals**2)))]
        thresholds += [fraction * top for fraction in _FALLBACK_FRACTIONS]
        for threshold in thresholds:
            above = np.flatnonzero(rejections > threshold)
            if above.size:
                chosen = int(above[0])
                break
    return (
        adjusted_at[chosen],
        float(cutoffs[chosen]),
        float(quantiles[chosen]),
        quantiles,
        rejections,
        fitted,
    )


def _reduced_design(reduced: Any, fit: DESeq2Fit) -> FloatArray:
    """The reduced design, checked to be nested in the full design and smaller."""
    if isinstance(reduced, str):
        raise TypeError(
            "reduced must be a design matrix or a Design of the fit's samples, such as "
            f"Design(samples, {reduced!r}); a formula alone does not give the samples"
        )
    x, _ = _design_matrix(reduced, None)
    _refuse_random_effects(reduced, "deseq2_likelihood_ratio_test")
    n_samples = fit.design.shape[0]
    if x.shape[0] != n_samples:
        raise ValueError(
            f"the reduced design has {x.shape[0]} rows but the fit has {n_samples} "
            "samples; both need one row per sample"
        )
    aliased = _aliased_columns(x)
    if aliased:
        raise ValueError(
            "the reduced design is not of full column rank: column(s) "
            f"{aliased} are linear combinations of the preceding columns"
        )
    basis = column_space(fit.design).basis
    remainder = x - basis @ (basis.T @ x)
    sizes = np.linalg.norm(x, axis=0)
    outside = np.flatnonzero(
        np.linalg.norm(remainder, axis=0) > RANK_TOLERANCE * np.maximum(sizes, 1e-300)
    )
    if outside.size:
        raise ValueError(
            "the reduced design must be nested in the full design, but its column(s) "
            f"{outside.tolist()} are not combinations of the full design's columns"
        )
    if x.shape[1] >= fit.design.shape[1]:
        raise ValueError(
            f"the reduced design must have fewer columns than the full design "
            f"({fit.design.shape[1]}), got {x.shape[1]}"
        )
    return x


# Arguments.


def _whole_counts(counts: npt.ArrayLike) -> FloatArray:
    """The counts as a new float64 array, so that a fit does not change with the
    caller's array; whole numbers are required."""
    y = _as_counts(counts)
    if isinstance(counts, np.ndarray) and np.shares_memory(y, counts):
        y = y.copy()
    bad = np.argwhere(y != np.floor(y))
    if bad.size:
        i, j = bad[0]
        raise ValueError(
            "counts must be whole numbers, as DESeq2 requires; sample "
            f"{i}, feature {j} is {y[i, j]} (round estimated counts first)"
        )
    return y


def _full_rank_design(
    design: Any, n_samples: int, function: str
) -> tuple[FloatArray, tuple[str, ...]]:
    x, names = _design_matrix(design, None)
    _refuse_random_effects(design, function)
    if x.shape[0] != n_samples:
        raise ValueError(
            f"counts have {n_samples} rows (samples) but the design has "
            f"{x.shape[0]}; both need one row per sample"
        )
    aliased = _aliased_columns(x)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            f"design is not of full column rank: column(s) {listed} are linear "
            "combinations of the preceding columns; remove them or reparametrise the "
            "design"
        )
    if x.shape[1] >= n_samples:
        raise ValueError(
            f"the design has {x.shape[1]} columns for {n_samples} samples and leaves no "
            "residual degrees of freedom, so dispersions cannot be estimated; DESeq2 "
            "needs replicates"
        )
    return x, names


def _size_factors(size_factors: npt.ArrayLike | None, y: FloatArray) -> FloatArray:
    n_samples = y.shape[0]
    if size_factors is None:
        try:
            factors: FloatArray = median_ratio_size_factors(y)
        except ValueError as error:
            raise ValueError(
                f"{error}; pass size_factors, for example "
                "median_ratio_size_factors(counts, method='poscounts')"
            ) from None
        return factors
    values = unmasked_array(size_factors, "size_factors")
    if values.dtype.kind not in "iuf":
        raise TypeError(
            f"size_factors must hold real numbers, got dtype {values.dtype}"
        )
    if values.shape != (n_samples,):
        raise ValueError(
            f"size_factors must have one value per sample ({n_samples},), got shape "
            f"{values.shape}"
        )
    factors = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(factors) & (factors > 0.0)):
        raise ValueError("size_factors must be positive and finite")
    return factors


def _per_gene(values: Any, name: str, y: FloatArray) -> FloatArray:
    """Positive, finite values per gene; genes without counts may have any value, and
    get NaN."""
    array = unmasked_array(values, name, missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    n_features = y.shape[1]
    if array.shape != (n_features,):
        raise ValueError(
            f"{name} must have one value per gene ({n_features},), got shape "
            f"{array.shape}"
        )
    result = np.array(array, dtype=np.float64)
    counted = (y > 0.0).any(axis=0)
    with np.errstate(invalid="ignore"):
        bad = counted & ~(np.isfinite(result) & (result > 0.0))
    if bad.any():
        raise ValueError(
            f"{name} must be positive and finite for genes with counts; "
            f"{int(np.count_nonzero(bad))} are not, such as gene "
            f"{int(np.flatnonzero(bad)[0])}"
        )
    result[~counted] = np.nan
    return result


def _optional_per_gene(values: Any, name: str, y: FloatArray) -> FloatArray | None:
    return None if values is None else _per_gene(values, name, y)


def _trend_kind(trend: Any) -> str:
    if not isinstance(trend, str):
        raise TypeError(f"trend must be a string, got {type(trend).__name__}")
    if trend not in ("parametric", "mean"):
        raise ValueError(f'trend must be "parametric" or "mean", got {trend!r}')
    return trend


def _real(value: Any, name: str, *, finite: bool = True) -> float:
    reject_masked(value, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or (finite and math.isinf(number)):
        raise ValueError(
            f"{name} must be {'finite' if finite else 'a number'}, got {number}"
        )
    return number


def _positive(value: Any, name: str) -> float:
    number = _real(value, name)
    if not number > 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _check_fit(fit: Any) -> None:
    if not isinstance(fit, DESeq2Fit):
        raise TypeError(
            f"fit must be a DESeq2Fit from fit_deseq2, got {type(fit).__name__}"
        )


def _read_only(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    view = np.asarray(values).view()
    view.flags.writeable = False
    return view


# Compiled routines: the dispersions.


@jit
def _log_gamma_shift(y: float, r: float) -> float:
    """log Gamma(y + r) - log Gamma(r) - y log(r), without cancellation for large r."""
    if y == 0.0:
        return 0.0
    if r >= 10.0:
        shift: float = (
            (r + y - 0.5) * math.log1p(y / r)
            - y
            + _stirling_error(r + y)
            - _stirling_error(r)
        )
        return shift
    value: float = _log_gamma_increment(r, y) - y * math.log(r)
    return value


@jit
def _log1p_remainder(u: float) -> float:
    """(u - log1p(u)) / u^2, which is 1/2 at u = 0, summed as a series for small u."""
    if abs(u) < 0.1:
        total = 0.0
        power = 1.0
        n = 0.0
        while True:
            updated = total + power / (n + 2.0)
            if updated == total:
                return total
            total = updated
            power *= -u
            n += 1.0
    return (u - math.log1p(u)) / (u * u)


@jit
def _count_score(y: float, mu: float, alpha: float) -> float:
    """The derivative with respect to log(alpha) of one count's log-likelihood.

    With r = 1 / alpha it is r [log1p(alpha mu) - (psi(y + r) - psi(r))] + (y - mu) /
    (1 + alpha mu). For r >= 10, psi(y + r) - psi(r) is log1p(alpha y) plus the
    differences of the terms of its asymptotic series, and the first part is formed
    from u = alpha (mu - y) / (1 + alpha y): as (mu - y) u [1 / (1 + alpha mu) - g(u) /
    (1 + alpha y)] with g(u) = (u - log1p(u)) / u^2 for small u, which keeps the
    cancellation of the terms of order mu out of it.
    """
    r = 1.0 / alpha
    if r < 10.0:
        shift = digamma_scalar(y + r) - digamma_scalar(r)
        return r * (math.log1p(alpha * mu) - shift) + (y - mu) / (1.0 + alpha * mu)
    u = alpha * (mu - y) / (1.0 + alpha * y)
    first: float
    if abs(u) < 0.1:
        first = (
            (mu - y)
            * u
            * (1.0 / (1.0 + alpha * mu) - _log1p_remainder(u) / (1.0 + alpha * y))
        )
    else:
        first = r * math.log1p(u) + (y - mu) / (1.0 + alpha * mu)
    if y == 0.0:
        return first
    log_ratio = math.log1p(alpha * y)
    series = 0.0
    power = alpha
    for k in range(_DIGAMMA_SERIES.size):
        series += _DIGAMMA_SERIES[k] * power * -math.expm1(-2.0 * (k + 1) * log_ratio)
        power *= alpha * alpha
    score: float = first - 0.5 * alpha * y / (1.0 + alpha * y) - series
    return score


@jit
def _cox_reid(
    alpha: float,
    mu: FloatArray,
    x: FloatArray,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> tuple[float, float]:
    """log det(X' W X), W = mu / (1 + alpha mu), and the derivative of -1/2 of it
    with respect to log(alpha), 1/2 sum_i h_i alpha W_i with h the hat values."""
    n, p = x.shape
    for a in range(p):
        for b in range(a + 1):
            total = 0.0
            for i in range(n):
                total += mu[i] / (1.0 + alpha * mu[i]) * x[i, a] * x[i, b]
            cross[a, b] = total
    log_det = 0.0
    for a in range(p):
        total = cross[a, a]
        for k in range(a):
            total -= factor[a, k] * factor[a, k]
        if not total > 0.0:
            return math.nan, math.nan
        diagonal = math.sqrt(total)
        factor[a, a] = diagonal
        log_det += 2.0 * math.log(diagonal)
        for b in range(a + 1, p):
            total = cross[b, a]
            for k in range(a):
                total -= factor[b, k] * factor[a, k]
            factor[b, a] = total / diagonal
    derivative = 0.0
    for i in range(n):
        weight = mu[i] / (1.0 + alpha * mu[i])
        norm = 0.0
        for a in range(p):
            total = x[i, a]
            for k in range(a):
                total -= factor[a, k] * solved[k]
            solved[a] = total / factor[a, a]
            norm += solved[a] * solved[a]
        derivative += weight * norm * alpha * weight
    return log_det, 0.5 * derivative


@jit
def _posterior_score(
    t: float,
    y: FloatArray,
    mu: FloatArray,
    x: FloatArray,
    prior_mean: float,
    precision: float,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> float:
    """The derivative with respect to t = log(alpha) of the adjusted log-likelihood
    plus the log prior."""
    alpha = math.exp(t)
    total: float = 0.0
    for i in range(y.size):
        total += _count_score(y[i], mu[i], alpha)
    total += _cox_reid(alpha, mu, x, cross, factor, solved)[1]
    if precision > 0.0:
        total -= precision * (t - prior_mean)
    return total


@jit
def _posterior_value(
    t: float,
    y: FloatArray,
    mu: FloatArray,
    x: FloatArray,
    prior_mean: float,
    precision: float,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> float:
    """The adjusted log-likelihood plus the log prior, up to terms free of alpha."""
    alpha = math.exp(t)
    r = 1.0 / alpha
    total: float = 0.0
    for i in range(y.size):
        total += _log_gamma_shift(y[i], r) - (y[i] + r) * math.log1p(alpha * mu[i])
    total -= 0.5 * _cox_reid(alpha, mu, x, cross, factor, solved)[0]
    if precision > 0.0:
        total -= 0.5 * precision * (t - prior_mean) ** 2
    return total


@jit
def _score_root(
    a: float,
    b: float,
    fa: float,
    fb: float,
    y: FloatArray,
    mu: FloatArray,
    x: FloatArray,
    prior_mean: float,
    precision: float,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> float:
    """The root of the derivative in [a, b], where fa > 0 >= fb, by the Illinois
    variant of false position, to a bracket of width 1e-12."""
    if fb == 0.0:
        return b
    side = 0
    for _ in range(_ROOT_ITERATIONS):
        if b - a <= _ROOT_TOLERANCE:
            break
        c = b - fb * (b - a) / (fb - fa)
        if not a < c < b:
            c = 0.5 * (a + b)
        fc = _posterior_score(c, y, mu, x, prior_mean, precision, cross, factor, solved)
        if fc > 0.0:
            a = c
            fa = fc
            if side == 1:
                fb *= 0.5
            side = 1
        elif fc < 0.0:
            b = c
            fb = fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            return c
    root: float = b - fb * (b - a) / (fb - fa)
    return root


@jit
def _maximize_one(
    y: FloatArray,
    mu: FloatArray,
    x: FloatArray,
    low: float,
    high: float,
    prior_mean: float,
    precision: float,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> tuple[float, int]:
    """The log dispersion of the largest maximum over [log(low), log(high)], and
    whether it is at the lower bound (-1), upper bound (1) or inside (0)."""
    t_low = math.log(low)
    t_high = math.log(high)
    width = (t_high - t_low) / (_GRID_POINTS - 1)
    best = math.nan
    best_value = -math.inf
    best_bound = 0
    n_candidates = 0
    t_previous = t_low
    s_previous = _posterior_score(
        t_low, y, mu, x, prior_mean, precision, cross, factor, solved
    )
    if not s_previous > 0.0:
        # The objective falls from the lower bound.
        best = t_low
        best_bound = -1
        n_candidates = 1
    for k in range(1, _GRID_POINTS):
        t = t_high if k == _GRID_POINTS - 1 else t_low + k * width
        s = _posterior_score(t, y, mu, x, prior_mean, precision, cross, factor, solved)
        if s_previous > 0.0 and not s > 0.0:
            root = _score_root(
                t_previous,
                t,
                s_previous,
                s,
                y,
                mu,
                x,
                prior_mean,
                precision,
                cross,
                factor,
                solved,
            )
            best, best_value, best_bound, n_candidates = _better(
                root,
                0,
                best,
                best_value,
                best_bound,
                n_candidates,
                y,
                mu,
                x,
                prior_mean,
                precision,
                cross,
                factor,
                solved,
            )
        t_previous = t
        s_previous = s
    if s_previous > 0.0:
        # The objective rises to the upper bound.
        best, best_value, best_bound, n_candidates = _better(
            t_high,
            1,
            best,
            best_value,
            best_bound,
            n_candidates,
            y,
            mu,
            x,
            prior_mean,
            precision,
            cross,
            factor,
            solved,
        )
    return best, best_bound


@jit
def _better(
    candidate: float,
    bound: int,
    best: float,
    best_value: float,
    best_bound: int,
    n_candidates: int,
    y: FloatArray,
    mu: FloatArray,
    x: FloatArray,
    prior_mean: float,
    precision: float,
    cross: FloatArray,
    factor: FloatArray,
    solved: FloatArray,
) -> tuple[float, float, int, int]:
    """The better of a new local maximum and the best so far; values are computed
    only once there are two."""
    if n_candidates == 0:
        return candidate, best_value, bound, 1
    if n_candidates == 1:
        best_value = _posterior_value(
            best, y, mu, x, prior_mean, precision, cross, factor, solved
        )
    value = _posterior_value(
        candidate, y, mu, x, prior_mean, precision, cross, factor, solved
    )
    if value > best_value:
        return candidate, value, bound, n_candidates + 1
    return best, best_value, best_bound, n_candidates + 1


@jit
def _maximize_genes(
    counts: FloatArray,
    means: FloatArray,
    x: FloatArray,
    low: float,
    high: float,
    prior_means: FloatArray,
    precision: float,
    first: int,
    last: int,
    out: FloatArray,
) -> None:
    """The dispersions of genes first..last-1; bounds are returned exactly."""
    p = x.shape[1]
    cross = np.zeros((p, p))
    factor = np.zeros((p, p))
    solved = np.zeros(p)
    for g in range(first, last):
        t, bound = _maximize_one(
            counts[g],
            means[g],
            x,
            low,
            high,
            prior_means[g],
            precision,
            cross,
            factor,
            solved,
        )
        if bound == -1:
            out[g] = low
        elif bound == 1:
            out[g] = high
        else:
            out[g] = math.exp(t)


@jit(parallel=True)
def _maximize_in_parallel(
    counts: FloatArray,
    means: FloatArray,
    x: FloatArray,
    low: float,
    high: float,
    prior_means: FloatArray,
    precision: float,
    n_chunks: int,
) -> FloatArray:
    """The maxima of :func:`_maximize_genes`, in runs of consecutive genes."""
    n_genes = counts.shape[0]
    out = np.empty(n_genes)
    for chunk in prange(n_chunks):
        _maximize_genes(
            counts,
            means,
            x,
            low,
            high,
            prior_means,
            precision,
            chunk * n_genes // n_chunks,
            (chunk + 1) * n_genes // n_chunks,
            out,
        )
    return out


@jit
def _maximize_in_order(
    counts: FloatArray,
    means: FloatArray,
    x: FloatArray,
    low: float,
    high: float,
    prior_means: FloatArray,
    precision: float,
) -> FloatArray:
    """The maxima of :func:`_maximize_genes` on the calling thread alone."""
    out = np.empty(counts.shape[0])
    _maximize_genes(
        counts, means, x, low, high, prior_means, precision, 0, counts.shape[0], out
    )
    return out


@jit
def _deconvolution_moment(residual: float, variance: float, shape: float) -> float:
    """E[(residual - u)^2 | residual] for residual = u + e, u = log(G / shape) with G
    gamma-distributed of this shape, e ~ N(0, variance).

    The log density of u given the residual is phi(u) = shape u - shape e^u -
    (residual - u)^2 / (2 variance) up to a constant, concave with its mode between 0
    and the residual. The posterior is integrated by the trapezoidal rule at steps of
    half its standard deviation at the mode, far beyond the steps where the error of
    the rule matters for such smooth functions, outwards until the terms fall below
    1e-17 of the mode's.
    """
    low = min(residual, 0.0)
    high = max(residual, 0.0)
    u = 0.5 * (low + high)
    for _ in range(200):
        e = math.exp(u)
        slope = shape - shape * e + (residual - u) / variance
        if slope > 0.0:
            low = u
        else:
            high = u
        updated = u + slope / (shape * e + 1.0 / variance)
        if not low <= updated <= high:
            updated = 0.5 * (low + high)
        if abs(updated - u) <= 1e-14 * max(1.0, abs(u)):
            u = updated
            break
        u = updated
    step = 0.5 / math.sqrt(shape * math.exp(u) + 1.0 / variance)
    peak = shape * u - shape * math.exp(u) - (residual - u) ** 2 / (2.0 * variance)
    total = 1.0
    moment = (residual - u) ** 2
    for direction in (-1.0, 1.0):
        for i in range(1, 100_000):
            v = u + direction * i * step
            weight = math.exp(
                shape * v
                - shape * math.exp(v)
                - (residual - v) ** 2 / (2.0 * variance)
                - peak
            )
            total += weight
            moment += weight * (residual - v) ** 2
            if weight < 1e-17:
                break
    result: float = moment / total
    return result


@jit
def _variance_score(
    residuals: FloatArray, variance: float, shape: float, log_gamma: float
) -> float:
    """Twice the variance squared times the derivative of the log-likelihood of the
    residuals with respect to the variance: sum E[(r - u)^2 | r] - variance."""
    total = 0.0
    for r in residuals:
        total += _deconvolution_moment(r, variance, shape) - variance
    return total


# Compiled routines: the fits.


@jit
def _observation_value(
    y: float, eta: float, alpha: float, floor: float, log_floor: float
) -> float:
    """One observation's term of the function the fits maximize, up to a constant:
    its log-likelihood at the linear predictor eta, continued below log(floor) by its
    quadratic expansion there with the expected information as curvature."""
    if eta >= log_floor:
        mu = math.exp(eta)
        above: float = (
            y * eta - y * math.log1p(alpha * mu) - mu * _log1p_ratio(alpha * mu)
        )
        return above
    base = (
        y * log_floor
        - y * math.log1p(alpha * floor)
        - floor * _log1p_ratio(alpha * floor)
    )
    slope = (y - floor) / (1.0 + alpha * floor)
    curvature = floor / (1.0 + alpha * floor)
    d = eta - log_floor
    below: float = base + slope * d - 0.5 * curvature * d * d
    return below


@jit
def _penalized_value(
    y: FloatArray,
    eta: FloatArray,
    alpha: float,
    floor: float,
    beta: FloatArray,
    ridge: FloatArray,
) -> tuple[float, float]:
    """The function the fits maximize, and the sum of the absolute values of its
    terms, which bounds its rounding error in units of epsilon."""
    log_floor = math.log(floor)
    total = 0.0
    size = 0.0
    for i in range(y.size):
        term = _observation_value(y[i], eta[i], alpha, floor, log_floor)
        total += term
        size += abs(term)
    for k in range(beta.size):
        penalty = 0.5 * ridge[k] * beta[k] * beta[k]
        total -= penalty
        size += penalty
    return total, size


@jit
def _linear_predictors(
    x: FloatArray, beta: FloatArray, offsets: FloatArray, eta: FloatArray
) -> None:
    n, p = x.shape
    for i in range(n):
        total = offsets[i]
        for k in range(p):
            total += x[i, k] * beta[k]
        eta[i] = total


@jit
def _solve_positive_definite(
    matrix: FloatArray, factor: FloatArray, rhs: FloatArray, out: FloatArray
) -> bool:
    """Solve matrix out = rhs by the Cholesky factor of the lower triangle of matrix;
    False if it is not positive definite."""
    p = rhs.size
    for a in range(p):
        total = matrix[a, a]
        for k in range(a):
            total -= factor[a, k] * factor[a, k]
        if not total > 0.0:
            return False
        diagonal = math.sqrt(total)
        factor[a, a] = diagonal
        for b in range(a + 1, p):
            total = matrix[b, a]
            for k in range(a):
                total -= factor[b, k] * factor[a, k]
            factor[b, a] = total / diagonal
    for a in range(p):
        total = rhs[a]
        for k in range(a):
            total -= factor[a, k] * out[k]
        out[a] = total / factor[a, a]
    for a in range(p - 1, -1, -1):
        total = out[a]
        for k in range(a + 1, p):
            total -= factor[k, a] * out[k]
        out[a] = total / factor[a, a]
    return True


@jit
def _fit_one(
    y: FloatArray,
    offsets: FloatArray,
    x: FloatArray,
    alpha: float,
    floor: float,
    ridge: FloatArray,
    beta: FloatArray,
    eta: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
    factor: FloatArray,
    step: FloatArray,
    trial: FloatArray,
    trial_eta: FloatArray,
) -> tuple[int, bool]:
    """Newton's method with step halving for one gene, from the coefficients in
    ``beta``, which it overwrites; ``eta`` gets the linear predictors."""
    n, p = x.shape
    log_floor = math.log(floor)
    floor_weight = floor / (1.0 + alpha * floor)
    floor_slope_offset = 1.0 + alpha * floor
    _linear_predictors(x, beta, offsets, eta)
    value, size = _penalized_value(y, eta, alpha, floor, beta, ridge)
    for iteration in range(1, _FIT_ITERATIONS + 1):
        for a in range(p):
            gradient[a] = -ridge[a] * beta[a]
            for b in range(p):
                hessian[a, b] = 0.0
        for i in range(n):
            if eta[i] >= log_floor:
                mu = math.exp(eta[i])
                denominator = 1.0 + alpha * mu
                score = (y[i] - mu) / denominator
                weight = mu * (1.0 + alpha * y[i]) / (denominator * denominator)
            else:
                score = (y[i] - floor) / floor_slope_offset - floor_weight * (
                    eta[i] - log_floor
                )
                weight = floor_weight
            for a in range(p):
                gradient[a] += x[i, a] * score
                for b in range(a + 1):
                    hessian[a, b] += weight * x[i, a] * x[i, b]
        for a in range(p):
            hessian[a, a] += ridge[a]
        if not _solve_positive_definite(hessian, factor, gradient, step):
            return iteration, False
        change = 0.0
        for i in range(n):
            total = 0.0
            for k in range(p):
                total += x[i, k] * step[k]
            change = max(change, abs(total))
        if not change < math.inf:
            return iteration, False
        length = 1.0
        if change > _LARGEST_STEP:
            length = _LARGEST_STEP / change
        while True:
            for k in range(p):
                trial[k] = beta[k] + length * step[k]
            _linear_predictors(x, trial, offsets, trial_eta)
            trial_value, trial_size = _penalized_value(
                y, trial_eta, alpha, floor, trial, ridge
            )
            slack = _ROUNDING_SLACK * max(size, trial_size)
            if trial_value >= value - slack:
                break
            length *= 0.5
            if length < 1e-12:
                return iteration, change <= _STEP_TOLERANCE
        for k in range(p):
            beta[k] = trial[k]
        for i in range(n):
            eta[i] = trial_eta[i]
        value = trial_value
        size = trial_size
        if change <= _STEP_TOLERANCE:
            return iteration, True
    return _FIT_ITERATIONS, False


@jit
def _fit_genes(
    counts: FloatArray,
    offsets: FloatArray,
    x: FloatArray,
    alpha: FloatArray,
    floor: float,
    ridge: FloatArray,
    starts: FloatArray,
    first: int,
    last: int,
    coefficients: FloatArray,
    fitted: FloatArray,
    converged: BoolArray,
    iterations: IntArray,
) -> None:
    """Fit genes first..last-1 (rows of counts) into the output arrays."""
    n, p = x.shape
    beta = np.empty(p)
    eta = np.empty(n)
    gradient = np.empty(p)
    hessian = np.zeros((p, p))
    factor = np.zeros((p, p))
    step = np.empty(p)
    trial = np.empty(p)
    trial_eta = np.empty(n)
    for g in range(first, last):
        for k in range(p):
            beta[k] = starts[g, k]
        steps, done = _fit_one(
            counts[g],
            offsets,
            x,
            alpha[g],
            floor,
            ridge,
            beta,
            eta,
            gradient,
            hessian,
            factor,
            step,
            trial,
            trial_eta,
        )
        for k in range(p):
            coefficients[g, k] = beta[k]
        for i in range(n):
            fitted[g, i] = math.exp(eta[i])
        converged[g] = done
        iterations[g] = steps


@jit(parallel=True)
def _fit_in_parallel(
    counts: FloatArray,
    offsets: FloatArray,
    x: FloatArray,
    alpha: FloatArray,
    floor: float,
    ridge: FloatArray,
    starts: FloatArray,
    n_chunks: int,
) -> tuple[FloatArray, FloatArray, BoolArray, IntArray]:
    """The fits of :func:`_fit_genes`, in runs of consecutive genes."""
    n_genes, n = counts.shape
    p = x.shape[1]
    coefficients = np.empty((n_genes, p))
    fitted = np.empty((n_genes, n))
    converged = np.empty(n_genes, dtype=np.bool_)
    iterations = np.empty(n_genes, dtype=np.int64)
    for chunk in prange(n_chunks):
        _fit_genes(
            counts,
            offsets,
            x,
            alpha,
            floor,
            ridge,
            starts,
            chunk * n_genes // n_chunks,
            (chunk + 1) * n_genes // n_chunks,
            coefficients,
            fitted,
            converged,
            iterations,
        )
    return coefficients, fitted, converged, iterations


@jit
def _fit_in_order(
    counts: FloatArray,
    offsets: FloatArray,
    x: FloatArray,
    alpha: FloatArray,
    floor: float,
    ridge: FloatArray,
    starts: FloatArray,
) -> tuple[FloatArray, FloatArray, BoolArray, IntArray]:
    """The fits of :func:`_fit_in_parallel` on the calling thread alone."""
    n_genes, n = counts.shape
    p = x.shape[1]
    coefficients = np.empty((n_genes, p))
    fitted = np.empty((n_genes, n))
    converged = np.empty(n_genes, dtype=np.bool_)
    iterations = np.empty(n_genes, dtype=np.int64)
    _fit_genes(
        counts,
        offsets,
        x,
        alpha,
        floor,
        ridge,
        starts,
        0,
        n_genes,
        coefficients,
        fitted,
        converged,
        iterations,
    )
    return coefficients, fitted, converged, iterations


@jit
def _log_likelihoods(
    counts: FloatArray, means: FloatArray, alpha: FloatArray
) -> FloatArray:
    """-2 times the negative binomial log-likelihood of each gene (rows)."""
    n_genes, n = counts.shape
    out = np.empty(n_genes)
    for g in range(n_genes):
        a = alpha[g]
        total = 0.0
        for i in range(n):
            y = counts[g, i]
            mu = means[g, i]
            term = y * math.log(mu) - lgamma_scalar(y + 1.0) if y > 0.0 else 0.0
            if a > 0.0:
                r = 1.0 / a
                term += _log_gamma_shift(y, r) - (y + r) * math.log1p(a * mu)
            else:
                term -= mu
            total += term
        out[g] = -2.0 * total
    return out
