"""Empirical Bayes moderation of feature-wise variances, as in limma's ``eBayes``.

A linear model fit gives each feature g a residual variance s_g^2 on d_g degrees of
freedom. The hierarchical model of Smyth (2004) assumes

    s_g^2 | sigma_g^2 ~ sigma_g^2 chi^2(d_g) / d_g,
    1 / sigma_g^2 ~ chi^2(d0) / (d0 s0^2),

so that s_g^2 ~ s0^2 F(d_g, d0) marginally. The prior (s0^2, d0) is estimated from all
features, and each variance is replaced by its posterior value
s~_g^2 = (d0 s0^2 + d_g s_g^2) / (d0 + d_g), the inverse of the posterior mean of
1 / sigma_g^2 (Smyth 2004). A moderated t-statistic, which uses s~_g in
place of s_g, follows a t distribution on d0 + d_g degrees of freedom under the null
hypothesis; an infinite d0 gives every feature the prior variance.

Estimation of the prior, by one of two estimators (``prior_estimator``):
    - The method of moments (``"moments"``, limma's ``legacy=TRUE``):
      :func:`fit_f_distribution` matches the mean and variance of e_g = log s_g^2 -
      psi(d_g / 2) + log(d_g / 2), whose variance is psi'(d0 / 2) plus the mean of
      psi'(d_g / 2) (Smyth 2004, section 6.2). When the variance of the e_g is no larger
      than chi-squared sampling explains, d0 is infinite and s0^2 is the mean variance.
      With a covariate (limma-trend uses the mean log-expression) the moments are taken
      about a natural cubic spline fit of e_g, and s0^2 becomes a trend (Phipson et al.
      2016, section 10.1). :func:`fit_f_distribution_robust` matches the moments of the
      log variances Winsorized at their 5% and 90% quantiles to those of a Winsorized
      log F variable, computed by Gauss-Legendre quadrature, so that hypervariable
      features do not lower d0. Each feature then gets its own prior df, smaller for
      outliers (Phipson et al. 2016). With a covariate, the log variances are first
      detrended by LOWESS.
    - Weighted moments and profile likelihood (``"likelihood"``, limma's
      ``fitFDistUnequalDF1``): :func:`fit_f_distribution_likelihood` estimates log s0^2
      by a weighted mean of the e_g, which gives less weight to variances on fewer
      degrees of freedom, and d0 by maximising the likelihood of the variances under
      s0^2 F(d_g, d0) with that scale (Chen et al. 2025). It suits degrees of freedom
      that differ between features, or are small or fractional. With a covariate the
      weighted mean becomes a weighted LOWESS trend; its robust variant downweights
      outliers by prior weights instead of Winsorizing.

    By default (``prior_estimator=None``) the likelihood estimator is used when the
    positive degrees of freedom differ between features, as with missing values or
    zero observation weights, and the method of moments otherwise: limma's default,
    ``legacy=NULL``, since limma 3.61.8.

Statistics: :func:`moderated_statistics` computes moderated t and F statistics, their
p-values and the log-odds of differential expression (the B-statistic of Loennstedt
and Speed 2002, in the form of Smyth 2004, section 5); :func:`treat` tests whether
coefficients exceed a threshold in absolute value (McCarthy and Smyth 2009).

Results match limma 3.62.1 (``fitFDist``, ``fitFDistRobustly``, ``fitFDistUnequalDF1``,
``squeezeVar``, ``eBayes`` and ``treat``, with ``legacy`` NULL, TRUE or FALSE).
Conventions that limma's documentation leaves open were settled by running it as a
black box; comments mark them "found by probing". The robust likelihood estimator
deliberately differs from limma's where some features have no usable variance or
prior weights are given (see :func:`fit_f_distribution_likelihood`).

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    McCarthy DJ, Smyth GK (2009). Testing significance relative to a fold-change
    threshold is a TREAT. Bioinformatics 25, 765-771. doi:10.1093/bioinformatics/btp053

    Loennstedt I, Speed TP (2002). Replicated microarray data. Statistica Sinica 12,
    31-46.

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. Nucleic Acids Research 53, gkaf018.
    doi:10.1093/nar/gkaf018

    Brent RP (1973). Algorithms for Minimization without Derivatives. Prentice-Hall,
    Englewood Cliffs.

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Olver FWJ, Olde Daalhuis AB, Lozier DW, Schneider BI, Boisvert RF, Clark CW,
    Miller BR, Saunders BV, Cohl HS, McClain MA, eds (2026). NIST Digital Library of
    Mathematical Functions, release 1.2.8 of 2026-09-15. https://dlmf.nist.gov/
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import column_space, correlation_matrix
from cytepool._masked_arrays import real_array, unmasked_array
from cytepool._messages import PACKAGE_DIRECTORY, value_range
from cytepool._options import checked_flag, real_number
from cytepool._ranks import average_ranks
from cytepool._summation import accurate_sum, two_sum
from cytepool.stats._interpolation import approx
from cytepool.stats._moderation import (
    LIKELIHOOD_MIN_DF,
    coefficient_prior_variance,
    log_odds,
    posterior_variances,
)
from cytepool.stats._optimize import brent_minimum
from cytepool.stats._result_tables import coefficient_position, feature_id_column
from cytepool.stats.distributions import (
    _f_log_x_density,
    _f_tail_quantile,
    _f_tails,
    f_isf_scalar,
    f_logsf,
    f_logsf_scalar,
    f_ppf_scalar,
    f_sf,
    t_sf,
)
from cytepool.stats.linear_models import LinearModelFit
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.smoothing import (
    _quantile_knots,
    _weighted_lowess,
    choose_lowess_span,
    lowess,
    natural_spline_basis,
)
from cytepool.stats.special import digamma, trigamma, trigamma_inverse
from cytepool.table import Table

__all__ = [
    "FDistributionFit",
    "LikelihoodFDistributionFit",
    "ModeratedFit",
    "RobustFDistributionFit",
    "SqueezedVariances",
    "TreatFit",
    "fit_f_distribution",
    "fit_f_distribution_likelihood",
    "fit_f_distribution_robust",
    "moderated_statistics",
    "squeeze_variances",
    "treat",
]

FloatArray = npt.NDArray[np.float64]

# Degrees of freedom below this count as zero (limma documents the coercion).
_ZERO_DF = 1e-15
# Variances are bounded below by these fractions of their median before taking logs,
# in the moment and the robust estimators respectively; in the moment estimator a
# zero median counts as one (found by probing), and the robust estimator does not
# accept one.
_VARIANCE_FLOOR = 1e-5
_ROBUST_VARIANCE_FLOOR = 1e-12
# The spline trend has 1 + (n >= 3) + (n >= 6) + (n >= 30) degrees of freedom,
# intercept included, for n usable features (limma 3.45.8 and later; found by
# probing), and no more than the covariate has distinct values.
_SPLINE_DF_STEPS = (3, 6, 30)
# Infinite covariates are moved one unit beyond the finite range (found by probing).
_INFINITE_COVARIATE_OFFSET = 1.0
# LOWESS settings of the robust trend (found by probing).
_TREND_SPAN = 0.4
_TREND_ITERATIONS = 3
# Iterations of the fixed point for the prior df of the most extreme outlier
# (Phipson et al. 2016, section 10.5; the count was found by probing).
_OUTLIER_ITERATIONS = 2
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(128)
# Eigenvalues of the coefficient correlation matrix at or below this fraction of the
# largest do not count towards the rank of the F-test (found by probing).
_F_RANK_TOLERANCE = 1e-8
_EPS = float(np.finfo(np.float64).eps)
# The likelihood estimator (limma's fitFDistUnequalDF1), all found by probing limma
# 3.62.1: variances on fewer degrees of freedom than LIKELIHOOD_MIN_DF take no part in
# the fit; variances at or below zero are raised to this fraction of the median of the
# positive usable variances;
_LIKELIHOOD_FLOOR = 1e-12
# d0 minimises minus twice the log-likelihood over v = d0 / (d0 + 2) in this interval
# (d0 from 2 to 9998), by Brent's method with the tolerance of R's optimize;
_DF2_SEARCH = (0.5, 0.9998)
_SEARCH_TOLERANCE = _EPS**0.25
# the trend's weights are divided by their upper quartile and kept in this range;
_TREND_WEIGHT_RANGE = (1e-8, 100.0)
# the span of the trend is choose_lowess_span(n, small_n=500);
_TREND_SMALL_N = 500
# with at most this many usable variances above zero the location is a constant and
# a robust fit is not attempted;
_CONSTANT_TREND_MAX = 2
# a straight-line trend leaves out its slope as R's QR decomposition does, with
# this tolerance;
_LINE_RANK_TOLERANCE = 1e-7
# robust fits flag the features whose two-sided tail probabilities, adjusted by
# Benjamini and Hochberg's method, are below this.
_OUTLIER_LEVEL = 0.3
# What makes a variance unusable, for error messages.
_MOMENT_UNUSABLE = "is NaN or has zero degrees of freedom"
_LIKELIHOOD_UNUSABLE = (
    f"is NaN, has fewer than {LIKELIHOOD_MIN_DF} degrees of freedom or has a zero "
    "prior weight"
)
_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)
_LARGEST = float(np.finfo(np.float64).max)
_LOG_HALF = math.log(0.5)


class FDistributionFit(NamedTuple):
    """A scaled F distribution fitted to variances: s_g^2 ~ scale F(df_g, df2).

    Attributes:
        scale: The prior variance s0^2: a float, or one value per feature when the
            fit has a covariate.
        df2: The prior degrees of freedom d0, possibly infinite.
    """

    scale: float | FloatArray
    df2: float


class RobustFDistributionFit(NamedTuple):
    """A scaled F distribution fitted robustly, with prior df for each feature.

    Attributes:
        scale: The prior variance s0^2: a float, or one value per feature when the
            fit has a covariate.
        df2: Prior degrees of freedom of the features that are not outliers.
        df2_shrunk: ``(n,)`` prior degrees of freedom of each feature: ``df2``,
            except for outliers, which get smaller values.
        df2_outlier: Prior degrees of freedom of the most extreme outlier; NaN
            when ``df2`` is infinite or the moment estimate is returned.
        tail_p_value: ``(n,)`` P(F > s_g^2 / s0^2), the probability of a variance
            as large; NaN for unusable features, and when the moment estimate is
            returned.
        prob_outlier: ``(n,)`` estimated probability that each feature is an
            outlier, before it is made monotonic in ``tail_p_value``; NaN where
            ``tail_p_value`` is.
    """

    scale: float | FloatArray
    df2: float
    df2_shrunk: FloatArray
    df2_outlier: float
    tail_p_value: FloatArray
    prob_outlier: FloatArray


class LikelihoodFDistributionFit(NamedTuple):
    """A scaled F distribution fitted by weighted moments and profile likelihood.

    Attributes:
        scale: The prior variance s0^2: a float, or one value per feature when it
            follows a covariate (NaN where the covariate is NaN).
        df2: The prior degrees of freedom d0, between 2 and 9998.
        df2_shrunk: ``(n,)`` prior degrees of freedom of each feature: ``df2``,
            except for the outliers of a robust fit, which get smaller values.
        df2_outlier: Prior degrees of freedom of the most extreme outlier of a
            robust fit; NaN when the fit is not robust or finds no outlier.
    """

    scale: float | FloatArray
    df2: float
    df2_shrunk: FloatArray
    df2_outlier: float


class SqueezedVariances(NamedTuple):
    """Posterior variances and the prior they were shrunk towards.

    Attributes:
        posterior_variance: ``(n,)`` posterior values of the variances: the
            inverses of the posterior means of their reciprocals.
        prior_variance: The prior variance: one value per feature when it follows a
            covariate or with at most two variances, a float otherwise.
        df_prior: The prior degrees of freedom: a float, or one value per feature
            from a robust fit (from the likelihood estimator only when it finds
            outliers, as in limma).
    """

    posterior_variance: FloatArray
    prior_variance: float | FloatArray
    df_prior: float | FloatArray


@dataclass(frozen=True, eq=False, repr=False)
class ModeratedFit:
    """Moderated t and F statistics and log-odds of a linear model fit.

    Rows of the per-feature arrays are features and columns are the coefficients
    (or contrasts) of ``fit``. The arrays are read-only.

    Attributes:
        fit: The linear model fit.
        prior_variance: The prior variance s0^2 (limma's ``s2.prior``): a float, or
            one value per feature with a trend or with at most two features.
        df_prior: The prior degrees of freedom d0 (``df.prior``): a float, possibly
            infinite, or one value per feature from a robust fit (from the
            likelihood estimator only when it finds outliers, as in limma).
        posterior_variance: ``(n_features,)`` posterior variances (``s2.post``).
        df_total: ``(n_features,)`` degrees of freedom of the moderated t-statistics,
            ``df_residual + df_prior`` capped at the residual degrees of freedom
            summed over all features (``df.total``).
        t: ``(n_features, n_coefficients)`` moderated t-statistics.
        p_value: Two-sided p-values of ``t`` on ``df_total`` degrees of freedom.
        log_odds: Log-odds that each coefficient is non-zero (limma's B-statistic,
            ``lods``), on the natural log scale.
        coefficient_prior_variance: ``(n_coefficients,)`` unscaled prior variance v0
            of a non-zero coefficient, which is N(0, v0 sigma_g^2) (``var.prior``).
        f_statistic: ``(n_features,)`` moderated F-statistics testing all
            coefficients equal to zero.
        f_p_value: P-values of ``f_statistic`` on ``f_df1`` and
            ``df_residual + df_prior`` degrees of freedom; limma does not cap the
            denominator degrees of freedom here (found by probing).
        f_df1: Numerator degrees of freedom of the F-tests: the rank of the
            correlation matrix of the coefficients.
        proportion: The assumed proportion of non-zero coefficients behind
            ``log_odds``.
        prior_estimator: The estimator of the prior: ``"moments"`` or
            ``"likelihood"`` (see :func:`squeeze_variances`); with at most two
            features neither is used.
    """

    fit: LinearModelFit
    prior_variance: float | FloatArray
    df_prior: float | FloatArray
    posterior_variance: FloatArray
    df_total: FloatArray
    t: FloatArray
    p_value: FloatArray
    log_odds: FloatArray
    coefficient_prior_variance: FloatArray
    f_statistic: FloatArray
    f_p_value: FloatArray
    f_df1: int
    proportion: float
    prior_estimator: str

    def table(
        self, coefficient: str | int | None = None, *, feature_ids: Any = None
    ) -> Table:
        """Return the tests of one coefficient or contrast as a table.

        limma's ``topTable(sort.by = "none")``, with the columns of the tables of
        :func:`~cytepool.limma_voom`, one row per feature in the order of the fit:
        ``feature_id``; ``log2_fold_change``, the coefficient (``logFC``);
        ``standard_error``, its moderated standard error, ``stdev_unscaled`` times the
        posterior standard deviation; ``average_log2_cpm``, the mean of the values
        fitted (``AveExpr``); ``t`` and ``p_value``; ``adjusted_p_value``,
        Benjamini-Hochberg over the features (``adj.P.Val``); and ``log_odds``, the
        B-statistic (``B``). The names of the effect and of the mean are those of log2
        counts per million, such as voom's, as limma names them ``logFC`` and
        ``AveExpr`` whatever the values.

        Args:
            coefficient: The coefficient or contrast, by name or position; it may be
                left out when the fit has one.
            feature_ids: One id per feature, as text, such as the gene ids of the
                columns of the counts; by default ``feature_0``, ``feature_1``, ...

        Raises:
            ValueError: If ``coefficient`` is left out and the fit has several, or
                ``feature_ids`` does not give one id per feature.
            KeyError, IndexError: If ``coefficient`` names or points to none.
            TypeError: If an argument has the wrong type, or is or holds a numpy
                masked array.
        """
        j = coefficient_position(coefficient, self.fit.coefficient_names)
        ids = feature_id_column(feature_ids, self.t.shape[0])
        p_value = self.p_value[:, j]
        return Table(
            {
                "feature_id": ids,
                "log2_fold_change": self.fit.coefficients[:, j],
                "standard_error": self.fit.stdev_unscaled[:, j]
                * np.sqrt(self.posterior_variance),
                "average_log2_cpm": self.fit.mean_expression,
                "t": self.t[:, j],
                "p_value": p_value,
                "adjusted_p_value": adjust_p_values(p_value, "bh"),
                "log_odds": self.log_odds[:, j],
            }
        )

    def __repr__(self) -> str:
        return (
            f"ModeratedFit(n_features={self.t.shape[0]}, "
            f"coefficient_names={self.fit.coefficient_names!r}, "
            f"df_prior={value_range(self.df_prior)}, "
            f"prior_variance={value_range(self.prior_variance)}, "
            f"prior_estimator={self.prior_estimator!r})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class TreatFit:
    """Moderated t-tests relative to a threshold on the absolute coefficients.

    Attributes:
        fit: The linear model fit.
        lfc: The threshold tau of the null hypothesis |beta| <= tau, on the scale of
            the coefficients (log2 fold changes for log2 expression).
        prior_variance: The prior variance, as in :class:`ModeratedFit`.
        df_prior: The prior degrees of freedom, as in :class:`ModeratedFit`.
        posterior_variance: ``(n_features,)`` posterior variances.
        df_total: ``(n_features,)`` degrees of freedom of the tests.
        t: ``(n_features, n_coefficients)`` moderated t-statistics shrunk towards
            zero by the threshold: sign(t) max(|t| - tau / se, 0), with se the
            moderated standard error.
        p_value: P-values of the tests of |beta| <= tau.
        prior_estimator: The estimator of the prior, as in :class:`ModeratedFit`.
    """

    fit: LinearModelFit
    lfc: float
    prior_variance: float | FloatArray
    df_prior: float | FloatArray
    posterior_variance: FloatArray
    df_total: FloatArray
    t: FloatArray
    p_value: FloatArray
    prior_estimator: str

    def __repr__(self) -> str:
        return (
            f"TreatFit(n_features={self.t.shape[0]}, "
            f"coefficient_names={self.fit.coefficient_names!r}, lfc={self.lfc!r})"
        )


def fit_f_distribution(
    variances: npt.ArrayLike,
    df: npt.ArrayLike,
    *,
    covariate: npt.ArrayLike | None = None,
) -> FDistributionFit:
    """Fit a scaled F distribution to variances by the method of moments.

    Each variance is modelled as ``scale * F(df_g, df2)``. Matches limma's
    ``fitFDist``. The moments are those of the logs of the variances, each bounded
    below by 1e-5 times their median (limma does the same). With a covariate, the
    scale follows a natural cubic spline in it: with n usable features, the spline
    has 4 degrees of freedom including the intercept for n >= 30, 3 for n from 6 to
    29, 2 for n from 3 to 5 and 1 (a constant) below, and never more than the
    covariate has distinct values; its interior knots are at quantiles of the
    covariate.

    A variance is usable when it is finite and its df is positive. Unusable
    features do not enter the estimate; with a covariate, their scale is the spline
    evaluated at their covariate (linear beyond the range of the usable features).
    A single usable variance gives ``df2 = 0`` and itself as the scale.

    Args:
        variances: ``(n,)`` non-negative variances; NaN marks a missing value.
        df: Degrees of freedom of the variances: one value, or ``(n,)``. Values
            below 1e-15 count as zero.
        covariate: ``(n,)`` covariate for a trended scale, such as the mean
            log-expression. Infinite values are moved one unit beyond the finite
            range, as limma does; NaN is allowed only for unusable features.

    Returns:
        The scale and ``df2``, which is infinite when the variances vary no more
        than chi-squared sampling explains; the scale is then their mean, or the
        exponentiated trend.

    Raises:
        TypeError: If an input is not real-valued, or is a numpy masked array.
        ValueError: If shapes disagree, a variance is negative or infinite, a df is
            negative or not finite, a usable feature has a NaN covariate, or no
            variance is usable.
    """
    x = _variance_vector(variances)
    d = _df_vector(df, x.size)
    usable = _usable(x, d)
    z = _covariate_vector(covariate, usable)
    _check_zero_variances(x, usable, robust=False)
    return _fit_moments(x, d, z, usable)


def fit_f_distribution_robust(
    variances: npt.ArrayLike,
    df: npt.ArrayLike,
    *,
    covariate: npt.ArrayLike | None = None,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
) -> RobustFDistributionFit:
    """Fit a scaled F distribution robustly, with prior df for each feature.

    The robust empirical Bayes estimator of Phipson et al. (2016), matching limma's
    ``fitFDistRobustly``:

    1. Variances are bounded below by 1e-12 times their median. When their df
       differ, each is mapped to the largest df d by matching tail probabilities
       under the moment estimate (s0^2, d0): s0^2 Q(P(s^2 / s0^2; d_g, d0); d, d0),
       with Q the F quantile (section 10.2).
    2. With a covariate, the log variances are detrended by LOWESS (span 0.4, three
       robustness iterations), and the scale is the exponentiated trend times the
       estimated scale of the detrended values.
    3. The log variances are Winsorized at their ``winsor_tail_p`` quantiles (R's
       type 7), and ``df2`` solves var(Winsorized log variances) = var(log win(F)),
       with the Winsorized moments of log F computed by 128-point Gauss-Legendre
       quadrature (sections 3 and 10.3). It is never below the moment estimate (a
       rule found by probing). log s0^2 is the Winsorized mean minus the mean of the
       Winsorized log F.
    4. Each feature gets a tail probability p_g = P(F > s_g^2 / s0^2), and
       pi_g = min(1, p_g / ((r_g - 0.5) / n)), with r_g the rank of s_g^2 / s0^2
       from the largest, is its probability of not being an outlier;
       ``prob_outlier`` is 1 - pi_g. The pi_g are made non-decreasing in p_g: in
       increasing order of p_g, the leading values up to the first minimum of
       their cumulative mean are set to that minimum, and a cumulative maximum
       follows. ``df2_shrunk`` is pi_g df2 + (1 - pi_g) d_outlier, where
       s0^2 F(d, d_outlier) has the largest ratio as median (two fixed-point
       iterations, as in limma). When ``df2`` is infinite, the cumulative maximum
       alone makes the pi_g non-decreasing, and ``df2_shrunk`` is infinite where
       pi_g = 1 and pi_g n d elsewhere, the df of all n variances on the common
       df d (both found by probing).

    Unusable features (NaN variance or zero df) are left out of the fit and get
    ``df2``; with a covariate, their scale is interpolated linearly in the
    covariate (constant beyond the range), as limma does. The moment estimate
    itself is returned, with ``df2_shrunk`` equal to its ``df2``, when the number
    of usable features times the larger tail proportion is below one (fewer than
    10 features with the default proportions), and when the moment estimate of
    ``df2`` is infinite but the Winsorized one is not (both rules found by
    probing). With more than two usable features, more than half of which have a
    zero variance, the estimator is undefined and raises an error, as limma does.

    limma solves the equation of step 3 only to a tolerance, so its ``df2`` agrees
    with this exact solution to about 1e-7 relative. Two limitations are shared
    with limma, whose results agree with these. The quadrature of step 3 is
    accurate to better than 1e-12 when both df are at least 1 and the tail
    proportions are the defaults, but not when a df is well below 1 (at df = 0.3
    the Winsorized variance of log F is off by 14% to 32%) or, near df = 1, with
    much smaller tail proportions (2e-4 at 0.01). And with about 20 or fewer
    features and a covariate, LOWESS passes exactly through some points, whose
    detrended log variances then tie up to rounding, so that their ranks, and with
    them ``prob_outlier`` and ``df2_shrunk``, can depend on the last bits of the
    input.

    Args:
        variances: ``(n,)`` non-negative variances; NaN marks a missing value.
        df: Degrees of freedom of the variances: one value, or ``(n,)``.
        covariate: ``(n,)`` covariate for a trended scale; see
            :func:`fit_f_distribution`.
        winsor_tail_p: Proportions of the variances Winsorized in the lower and
            upper tails, each strictly between 0 and 1/2; one value applies to both.

    Returns:
        The robust fit.

    Raises:
        TypeError: If an input is not real-valued, or is a numpy masked array.
        ValueError: As :func:`fit_f_distribution`, if ``winsor_tail_p`` is
            invalid, or if more than half of three or more usable variances are
            zero.
    """
    x = _variance_vector(variances)
    d = _df_vector(df, x.size)
    usable = _usable(x, d)
    z = _covariate_vector(covariate, usable)
    lower, upper = _tail_proportions(winsor_tail_p)
    _check_zero_variances(x, usable, robust=True)
    return _fit_robust(x, d, z, usable, lower, upper)


def fit_f_distribution_likelihood(
    variances: npt.ArrayLike,
    df: npt.ArrayLike,
    *,
    covariate: npt.ArrayLike | None = None,
    prior_weights: npt.ArrayLike | None = None,
    robust: bool = False,
) -> LikelihoodFDistributionFit:
    """Fit a scaled F distribution by weighted moments and profile likelihood.

    limma's ``fitFDistUnequalDF1`` (limma 3.61.8 and later), which ``squeezeVar``,
    ``eBayes`` and ``treat`` use by default when the degrees of freedom of the
    variances differ. Each variance is modelled as ``scale * F(df_g, df2)``. The
    mean of e_g = log s_g^2 - psi(d_g / 2) + log(d_g / 2) is log s0^2 - psi(d0 / 2) +
    log(d0 / 2) whatever d_g, so for each d0 the scale is estimated from the weighted
    mean of the e_g with weights pw_g / psi'(d_g / 2): psi'(d_g / 2) is the sampling
    variance of log s_g^2, so variances on fewer degrees of freedom count less (pw_g
    are the prior weights, 1 by default). d0 then maximises the likelihood of the
    variances, each term weighted by pw_g, under the scaled F distribution with that
    scale: a profile likelihood (Chen et al. 2025). With a covariate, the weighted
    mean becomes a weighted trend of the e_g on the covariate.

    A variance is usable when it is finite, its df is at least 0.01 and its prior
    weight is positive; at least two usable variances must be above zero. The
    conventions of limma 3.62.1 that its documentation leaves open were found by
    probing it:

    - Variances at or below zero are raised to 1e-12 times the median of the
      positive usable variances.
    - d0 minimises minus twice the log-likelihood by Brent's method, as R's
      ``optimize`` with its default tolerance, over v = d0 / (d0 + 2) in
      [1/2, 0.9998]. d0 therefore lies between 2 and 9998: variances that vary no
      more than chi-squared sampling explains give d0 near 8,135, not infinity, and
      the scale is that of this d0.
    - The trend is limma's ``loessFit(e, covariate, weights, span =
      chooseLowessSpan(n, small.n = 500), iterations = 1)`` with the weights divided
      by their upper quartile and kept in [1e-8, 100]. Every feature with a covariate
      enters it, an unusable one with the smallest weight, 1e-8, its variance (zero
      when missing) raised as above and a df of 1 when below 0.01. Equal weights give
      R's ``lowess``. Unequal weights give a weighted least squares line through at
      most four features, and through more limma's weighted LOWESS, whose windows
      hold a share of the total weight.
    - With at most two usable variances above zero the scale is one value whatever
      the covariate, and a robust fit is the first fit. When some variances are not
      usable as well, limma returns NaN; here they are left out.
    - A feature whose covariate is NaN (allowed only when its variance is not usable)
      is left out of the trend and gets a NaN scale, where limma stops; infinite
      covariates are moved one unit beyond the finite range, as for
      :func:`fit_f_distribution`, where limma returns NaN.

    Robust fits (``robust=True``) downweight outliers by prior weights instead of
    Winsorizing, as limma does:

    1. Each usable variance gets the two-sided tail probability of s_g^2 / s0^2
       under F(d_g, d0) from the fit above, adjusted by Benjamini and Hochberg's
       (1995) method; those below 0.3 are outliers (large or small variances).
    2. The fit is repeated with each outlier's prior weight multiplied by its
       adjusted probability.
    3. pi_g = min(1, p_g / ((r_g - 0.5) / n)), with p_g the upper tail
       probability of the first fit and r_g the rank of s_g^2 / s0^2 from the
       largest among the n usable variances, is made non-decreasing in p_g as in
       :func:`fit_f_distribution_robust` and taken as the probability of not being
       an outlier. ``df2_outlier`` takes two steps of d <- d log(1/2) / log P(F > r)
       (Phipson et al. 2016, section 10.5) for the variance with the smallest p_g,
       from d = ``df2`` of the second fit with its first p_g, then with its ratio
       r to the first fit's scale. ``df2_shrunk`` is pi_g ``df2`` + (1 - pi_g)
       ``df2_outlier``.

    Without outliers the first fit is returned, and when every pi_g is 1 the second
    one, with ``df2_shrunk`` equal to ``df2``. This deliberately differs from limma
    3.62.1 in two respects. Only the usable variances enter steps 1 and 3, where
    limma also counts the others, a missing variance as a zero one: it becomes the
    most extreme outlier and lowers the adjusted probabilities of the others, so
    missing values change which features are outliers. In a simulation where 40 of
    2,000 genes had no residual df, limma's robust prior df was 11.5% larger with a
    constant prior, where it also gave 45 genes a smaller prior df as outliers and
    this function gives every gene the same, and 6% to 10% larger with a trend.
    Without a covariate the result is limma's fit of the usable variances alone.
    And the second fit keeps the prior weights, which limma's drops, bringing back
    the features they leave out. The two agree when every variance is usable and
    there are no prior weights.

    With df of 1e7 and more, limma's likelihood loses precision: its d0 differed
    from this function's by up to 1e-5 relative at 1e7 and 5e-2 at 1e12, where this
    function's agrees with the likelihood computed to 60 digits.

    Args:
        variances: ``(n,)`` non-negative variances; NaN marks a missing value.
        df: Degrees of freedom of the variances: one value, or ``(n,)``. They may
            be fractional or below 1.
        covariate: ``(n,)`` covariate for a trended scale, such as the mean
            log-expression; NaN is allowed only where the variance is not usable.
        prior_weights: ``(n,)`` non-negative weights of the features; 0 leaves a
            feature out.
        robust: Whether to downweight outliers.

    Returns:
        The scale, ``df2``, and with ``robust`` the prior df of each feature and of
        the most extreme outlier.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array, or
            ``robust`` is not a bool.
        ValueError: As :func:`fit_f_distribution`, if ``prior_weights`` is
            invalid, or if fewer than two usable variances are above zero (limma
            returns NA), in the fit or, with ``robust``, in its second fit.
    """
    x = _variance_vector(variances)
    d = _df_vector(df, x.size)
    weights = _prior_weight_vector(prior_weights, x.size)
    usable = _likelihood_usable(x, d, weights)
    z = _covariate_vector(covariate, usable, _LIKELIHOOD_UNUSABLE)
    checked_flag(robust, "robust", expected="a bool")
    _check_zero_variances(x, usable, robust=False)
    return _fit_likelihood(x, d, z, weights, bool(robust))


def squeeze_variances(
    variances: npt.ArrayLike,
    df: npt.ArrayLike,
    *,
    covariate: npt.ArrayLike | None = None,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments", "likelihood"] | None = None,
) -> SqueezedVariances:
    """Shrink variances towards a prior by empirical Bayes.

    Matches limma's ``squeezeVar``. The prior is estimated by the method of
    moments, :func:`fit_f_distribution` or, when ``robust``,
    :func:`fit_f_distribution_robust`; or by weighted moments and profile
    likelihood, :func:`fit_f_distribution_likelihood`, robustly when ``robust``.
    Each variance is replaced by its posterior value (d0 s0^2 + d_g s_g^2) /
    (d0 + d_g), the inverse of the posterior mean of its reciprocal (Smyth 2004).
    A feature with zero df, or an infinite prior df, gets the prior variance; a NaN
    variance with positive df stays NaN. With at most two variances there is no
    moderation: the prior df is 0 and each variance is its own prior, as in limma
    3.61.12 and later. A robust fit gives each feature its own prior df: always with
    the method of moments, and with the likelihood estimator when it finds
    outliers, as in limma. The likelihood estimator counts a variance on zero df as
    missing, as limma's ``squeezeVar`` does.

    Args:
        variances: ``(n,)`` non-negative variances; NaN marks a missing value.
        df: Degrees of freedom of the variances: one value, or ``(n,)``.
        covariate: ``(n,)`` covariate for a trended prior variance.
        robust: Whether to estimate the prior robustly.
        winsor_tail_p: Winsorized tail proportions of the robust moment estimator;
            the likelihood estimator does not use them, as in limma.
        prior_estimator: ``"moments"`` (limma's ``legacy=TRUE``), ``"likelihood"``
            (``legacy=FALSE``), or None (the default) for limma's default
            ``legacy=NULL``: the likelihood estimator when the positive degrees of
            freedom are not all equal, and the method of moments otherwise.

    Returns:
        The posterior variances, the prior variance and the prior df.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array, or
            ``robust`` is not a bool.
        ValueError: As :func:`fit_f_distribution` or
            :func:`fit_f_distribution_likelihood`, or if an option is invalid.
    """
    x = _variance_vector(variances)
    given = _df_values(df, x.size)
    d = _zero_tiny_df(given)
    usable = _usable(x, d)
    z = _covariate_vector(covariate, usable)
    lower, upper = _tail_proportions(winsor_tail_p)
    checked_flag(robust, "robust", expected="a bool")
    estimator = _prior_estimator(prior_estimator, given)
    return _squeeze(x, d, z, usable, bool(robust), lower, upper, estimator)


def moderated_statistics(
    fit: LinearModelFit,
    *,
    proportion: float = 0.01,
    stdev_coef_lim: Sequence[float] = (0.1, 4.0),
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    f_statistic: Literal["exact", "limma"] | None = None,
    prior_estimator: Literal["moments", "likelihood"] | None = None,
) -> ModeratedFit:
    """Empirical Bayes moderated t and F statistics and log-odds of a fit.

    Matches limma's ``eBayes(fit)``, except that by default the F-statistics of
    fits with missing values or observation weights are exact rather than limma's
    approximation (see below and ``f_statistic``). The residual variances of the
    fit are squeezed by :func:`squeeze_variances`; the moderated t-statistics divide
    the coefficients by ``stdev_unscaled`` times the posterior standard deviations
    and have ``df_total`` degrees of freedom. By default the prior is estimated as
    limma's ``eBayes`` estimates it (``legacy=NULL``): by the likelihood estimator
    when the residual degrees of freedom differ between the features that have
    some, as with missing values or zero observation weights, and by the method
    of moments otherwise.

    The log-odds (B-statistic) of coefficient j of feature g is (Smyth 2004)

        log(p / (1 - p)) + log(r) / 2 + (1 + d) / 2 log((t^2 + d) / (t^2 r + d)),

    with p = ``proportion``, d = ``df_total``, r = v / (v + v0), v the squared
    unscaled standard error and v0 the coefficient's prior variance; with an
    infinite prior df the last term is its limit t^2 (1 - r) / 2 (found by
    probing). v0 is estimated from the largest |t| (Smyth 2004, section 6.3): with
    G the number of finite t-statistics, each of the ceiling(p G / 2) largest gives
    an estimate, clipped to [lim1^2, lim2^2] / median(prior variance) from
    ``stdev_coef_lim``, and v0 is their mean. The proportion used there is at least
    the share of the largest values, and t-statistics on unequal df are first
    mapped to the largest df by matching tail probabilities (found by probing).

    The moderated F-statistic tests all coefficients of a feature equal to zero:
    t' R^+ t / r, with R the correlation matrix of the coefficients and r its rank,
    counting eigenvalues above 1e-8 times the largest. ``f_statistic="limma"`` uses
    the correlation of ``fit.cov_coefficients`` for every feature, as limma does;
    ``"exact"`` uses each feature's own covariance matrix when the fit holds them
    (missing values or observation weights), which is the correct F-statistic
    (Smyth 2004, section 7). The two agree when one covariance matrix serves every
    feature. A feature with a NaN coefficient has a NaN F-statistic.

    Args:
        fit: A fit from :func:`~cytepool.stats.linear_models.fit_linear_model` or
            :func:`~cytepool.stats.linear_models.contrasts_fit`.
        proportion: Assumed proportion of non-zero coefficients, in (0, 1).
        stdev_coef_lim: Lower and upper limits for the standard deviation of a
            non-zero coefficient, used to bound v0; 0 <= lower <= upper, and the
            upper limit may be infinite.
        trend: Whether the prior variance follows the mean expression of the
            features (limma-trend), or an ``(n_features,)`` covariate to use
            instead. Features without observed values may have a NaN covariate.
        robust: Whether to estimate the prior robustly.
        winsor_tail_p: Winsorized tail proportions of the robust estimator.
        f_statistic: "exact" or "limma"; by default "exact" unless the fit's
            standard errors are limma's approximation from
            ``contrasts_fit(method="limma")``, whose F-statistics can only be
            limma's.
        prior_estimator: ``"moments"``, ``"likelihood"`` or None, as in
            :func:`squeeze_variances`; the result records the estimator used.

    Returns:
        The moderated statistics.

    Raises:
        TypeError: If ``fit`` is not a LinearModelFit, or an option has the wrong
            type or is a numpy masked array.
        ValueError: If an option is invalid, no feature has residual degrees of
            freedom, or "exact" F-statistics are asked of a fit whose standard
            errors are limma's approximation.
    """
    _check_fit(fit)
    p = _open_unit_interval(proportion, "proportion")
    limits = _coefficient_limits(stdev_coef_lim)
    covariate = _trend_covariate(trend, fit)
    exact = _f_method(f_statistic, fit)
    squeezed, df_total, estimator = _moderate(
        fit, covariate, robust, winsor_tail_p, prior_estimator
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        t = fit.coefficients / (
            fit.stdev_unscaled * np.sqrt(squeezed.posterior_variance)[:, np.newaxis]
        )
    p_value = 2.0 * _as_array(t_sf(np.abs(t), df_total[:, np.newaxis]))
    df_prior = np.broadcast_to(squeezed.df_prior, df_total.shape)
    infinite = np.isinf(df_prior)
    prior_variance = np.broadcast_to(squeezed.prior_variance, df_total.shape)
    median_prior = float(np.nanmedian(prior_variance))
    v0 = np.array(
        [
            coefficient_prior_variance(
                t[:, j], fit.stdev_unscaled[:, j], df_total, p, limits, median_prior
            )
            for j in range(t.shape[1])
        ]
    )
    # Without a finite t-statistic (an all-zero contrast, or features whose
    # variances are zero and not moderated) there is nothing to estimate v0 from;
    # limma then takes the reciprocal of the first prior variance, infinite when it
    # is zero, a value that the log-odds, all NaN, never use (found by probing).
    failed = ~np.isfinite(v0)
    if failed.any():
        with np.errstate(divide="ignore"):
            v0[failed] = np.reciprocal(prior_variance[0])
    lods = log_odds(t, fit.stdev_unscaled, df_total, infinite, v0, p)
    f_value, f_df1 = _f_statistics(t, fit, exact)
    f_df2 = df_prior + fit.df_residual
    f_p_value = _as_array(f_sf(f_value, float(f_df1), f_df2))
    return ModeratedFit(
        fit=fit,
        prior_variance=_frozen(squeezed.prior_variance),
        df_prior=_frozen(squeezed.df_prior),
        posterior_variance=_frozen(squeezed.posterior_variance),
        df_total=_frozen(df_total),
        t=_frozen(t),
        p_value=_frozen(p_value),
        log_odds=_frozen(lods),
        coefficient_prior_variance=_frozen(v0),
        f_statistic=_frozen(f_value),
        f_p_value=_frozen(f_p_value),
        f_df1=f_df1,
        proportion=p,
        prior_estimator=estimator,
    )


def treat(
    fit: LinearModelFit,
    *,
    lfc: float | None = None,
    fc: float | None = None,
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments", "likelihood"] | None = None,
) -> TreatFit:
    """Moderated t-tests that coefficients exceed a threshold in absolute value.

    Tests H0: |beta| <= tau against |beta| > tau (McCarthy and Smyth 2009), matching
    limma's ``treat(fit)``. With the moderated t-statistic t, its
    standard error se and delta = tau / se, the p-value is the worst case over the
    null interval, P(T > |t| - delta) + P(T > |t| + delta) with T on ``df_total``
    degrees of freedom; it is at least the ordinary moderated p-value and equals it
    for tau = 0. The reported t-statistics are sign(t) max(|t| - delta, 0). The
    variances are moderated as by :func:`moderated_statistics`. Unlike limma, which
    reports t = 0 for a coefficient that is NaN, and for an all-zero contrast
    tested with a zero threshold, the t-statistic is NaN there.

    Give the threshold as a log2 fold change ``lfc`` or as a fold change ``fc``,
    with ``lfc = log2(fc)``; the default is ``fc = 1.2``, as in limma. The
    threshold is |lfc|, so ``fc = 0.8`` and ``fc = 1.25`` are the same test.
    Typical values are fc of 1.1, 1.2 or 1.5: estimated fold changes must be
    clearly above the threshold to be significant.

    Args:
        fit: A fit from :func:`~cytepool.stats.linear_models.fit_linear_model` or
            :func:`~cytepool.stats.linear_models.contrasts_fit` of log2 values.
        lfc: Threshold as a log2 fold change, finite.
        fc: Threshold as a fold change, positive and finite.
        trend: As in :func:`moderated_statistics`.
        robust: Whether to estimate the prior robustly.
        winsor_tail_p: Winsorized tail proportions of the robust estimator.
        prior_estimator: ``"moments"``, ``"likelihood"`` or None, as in
            :func:`squeeze_variances`; the result records the estimator used.

    Returns:
        The tests relative to the threshold.

    Raises:
        TypeError: If ``fit`` is not a LinearModelFit, or an option has the wrong
            type or is a numpy masked array.
        ValueError: If both ``lfc`` and ``fc`` are given, either is invalid, or no
            feature has residual degrees of freedom.
    """
    _check_fit(fit)
    threshold = _treat_threshold(lfc, fc)
    covariate = _trend_covariate(trend, fit)
    squeezed, df_total, estimator = _moderate(
        fit, covariate, robust, winsor_tail_p, prior_estimator
    )
    # (|beta| - tau) / se and (|beta| + tau) / se, which are |t| - delta and
    # |t| + delta. A coefficient that is zero with zero standard error (an all-zero
    # contrast) gets -inf and +inf for tau > 0, so t = 0 and p = 1, as in limma;
    # for tau = 0 both are 0 / 0, so t and p are NaN (limma reports t = 0 there).
    with np.errstate(divide="ignore", invalid="ignore"):
        se = fit.stdev_unscaled * np.sqrt(squeezed.posterior_variance)[:, np.newaxis]
        size = np.abs(fit.coefficients)
        lower = (size - threshold) / se
        upper = (size + threshold) / se
        shrunk = np.sign(fit.coefficients) * np.maximum(lower, 0.0)
    df = df_total[:, np.newaxis]
    p_value = _as_array(t_sf(lower, df)) + _as_array(t_sf(upper, df))
    return TreatFit(
        fit=fit,
        lfc=threshold,
        prior_variance=_frozen(squeezed.prior_variance),
        df_prior=_frozen(squeezed.df_prior),
        posterior_variance=_frozen(squeezed.posterior_variance),
        df_total=_frozen(df_total),
        t=_frozen(shrunk),
        p_value=_frozen(p_value),
        prior_estimator=estimator,
    )


# ---------------------------------------------------------------------------------------
# Moment estimator
# ---------------------------------------------------------------------------------------


def _fit_moments(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    usable: npt.NDArray[np.bool_],
) -> FDistributionFit:
    """Moment fit of the usable variances; see :func:`fit_f_distribution`."""
    values = x[usable]
    dfs = d[usable]
    n = values.size
    if n == 1:
        scale = float(values[0])
        if covariate is None:
            return FDistributionFit(scale, 0.0)
        return FDistributionFit(np.full(x.size, scale), 0.0)
    values = np.maximum(values, _VARIANCE_FLOOR * _median_or_one(values))
    e = np.log(values) + _log_minus_digamma(dfs / 2.0)
    mean_trigamma = float(np.mean(_as_array(trigamma(dfs / 2.0))))
    if covariate is None:
        e_mean = float(np.mean(e))
        e_variance = float(np.sum((e - e_mean) ** 2)) / (n - 1) - mean_trigamma
        if e_variance > 0.0:
            df2 = 2.0 * float(trigamma_inverse(e_variance))
            scale = math.exp(
                e_mean - float(_log_minus_digamma(np.array([df2 / 2.0]))[0])
            )
        else:
            df2 = math.inf
            scale = float(np.mean(values))
        return FDistributionFit(scale, df2)
    fitted, df_residual = _spline_trend(covariate, usable, e)
    if fitted is None:
        # A trend of one degree of freedom is a constant: limma then fits no trend,
        # which matters for an infinite df2, where the scale is the mean variance
        # (found by probing).
        constant = _fit_moments(x, d, None, usable)
        return FDistributionFit(np.full(x.size, constant.scale), constant.df2)
    residuals = e - fitted[usable]
    e_variance = float(np.sum(residuals**2)) / df_residual - mean_trigamma
    if e_variance > 0.0:
        df2 = 2.0 * float(trigamma_inverse(e_variance))
        shift = float(_log_minus_digamma(np.array([df2 / 2.0]))[0])
        return FDistributionFit(np.exp(fitted - shift), df2)
    return FDistributionFit(np.exp(fitted), math.inf)


def _spline_trend(
    covariate: FloatArray, usable: npt.NDArray[np.bool_], e: FloatArray
) -> tuple[FloatArray | None, int]:
    """Least squares natural spline fit of e on the covariate of the usable features.

    Returns the fit evaluated at every feature (NaN where the covariate is NaN) and
    its residual degrees of freedom, or None when the spline has one degree of
    freedom (fewer than three usable features, or a constant covariate).
    """
    known = covariate[usable]
    n = known.size
    df = 1 + sum(n >= step for step in _SPLINE_DF_STEPS)
    df = min(df, np.unique(known).size)
    if df == 1:
        return None, n - 1
    lower, upper = float(known.min()), float(known.max())
    try:
        knots = _quantile_knots(known, df - 2, lower, upper) if df > 2 else None
    except ValueError as error:
        # R's ns() fails here too, and so does limma's fitFDist.
        raise ValueError(
            f"the spline trend needs {df} degrees of freedom, but the covariate "
            "has too many ties at its extremes to place the knots at its "
            f"quantiles ({error})"
        ) from None
    basis = natural_spline_basis(
        covariate, knots=knots, intercept=True, boundary_knots=(lower, upper)
    )
    design = basis[usable]
    kept = list(column_space(design).kept)
    coefficients, *_ = np.linalg.lstsq(design[:, kept], e, rcond=None)
    fitted: FloatArray = basis[:, kept] @ coefficients
    return fitted, n - len(kept)


def _log_minus_digamma(x: FloatArray) -> FloatArray:
    """log(x) - psi(x) for x > 0, without cancellation for large x."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape)
    large = x >= 10.0
    small = ~large
    out[small] = np.log(x[small]) - _as_array(digamma(x[small]))
    # log(x) - psi(x) = 1/(2x) + sum_k B_2k / (2k x^2k) (DLMF 5.11.2); eight terms
    # are below 1e-16 relative for x >= 10.
    r = 1.0 / x[large]
    r2 = r * r
    series = r2 * (
        1.0 / 12.0
        - r2
        * (
            1.0 / 120.0
            - r2
            * (
                1.0 / 252.0
                - r2
                * (
                    1.0 / 240.0
                    - r2
                    * (
                        1.0 / 132.0
                        - r2
                        * (691.0 / 32760.0 - r2 * (1.0 / 12.0 - r2 * (3617.0 / 8160.0)))
                    )
                )
            )
        )
    )
    out[large] = 0.5 * r + series
    return out


# ---------------------------------------------------------------------------------------
# Robust estimator
# ---------------------------------------------------------------------------------------


def _fit_robust(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    usable: npt.NDArray[np.bool_],
    p_lower: float,
    p_upper: float,
) -> RobustFDistributionFit:
    """Robust fit; see :func:`fit_f_distribution_robust`."""
    n = x.size
    if not usable.all():
        part = _fit_robust(
            x[usable],
            d[usable],
            None if covariate is None else covariate[usable],
            np.ones(int(usable.sum()), dtype=bool),
            p_lower,
            p_upper,
        )
        shrunk = np.full(n, part.df2)
        shrunk[usable] = part.df2_shrunk
        tail = np.full(n, np.nan)
        tail[usable] = part.tail_p_value
        outlier = np.full(n, np.nan)
        outlier[usable] = part.prob_outlier
        scale: float | FloatArray = part.scale
        if covariate is not None:
            full = np.empty(n)
            full[usable] = part.scale
            full[~usable] = np.exp(
                approx(covariate[usable], np.log(part.scale), covariate[~usable])
            )
            scale = full
        return RobustFDistributionFit(
            scale, part.df2, shrunk, part.df2_outlier, tail, outlier
        )
    if n * max(p_lower, p_upper) < 1.0:
        # Too few features to Winsorize: limma returns the moment estimate (found
        # by probing).
        return _moment_fallback(_fit_moments(x, d, covariate, usable), n)

    values = np.maximum(x, _ROBUST_VARIANCE_FLOOR * _median_or_one(x))
    moments = _fit_moments(values, d, covariate, usable)
    df1 = float(d.max())
    if float(d.min()) < df1:
        values = _equivalent_variances(values, d, df1, moments.scale, moments.df2)
        # An extreme outlier can map beyond the doubles; its log stays finite.
        values = np.clip(values, _SMALLEST_NORMAL, _LARGEST)
    log_values = np.log(values)
    trend = (
        np.zeros(n)
        if covariate is None
        else lowess(
            covariate, log_values, span=_TREND_SPAN, iterations=_TREND_ITERATIONS
        )
    )
    detrended = log_values - trend
    low, high = np.quantile(detrended, [p_lower, 1.0 - p_upper])
    winsorized = np.clip(detrended, low, high)
    target = float(np.var(winsorized, ddof=1))
    # The Winsorized estimate is never below the moment estimate. When that is
    # infinite but the Winsorized one is not, limma returns the moment estimate
    # (found by probing).
    if _winsorized_moments(df1, math.inf, p_lower, p_upper)[1] >= target:
        df2 = math.inf
    elif math.isinf(moments.df2):
        return _moment_fallback(moments, n)
    else:
        df2 = _winsorized_df2(df1, target, p_lower, p_upper, moments.df2)
    log_scale = (
        float(np.mean(winsorized)) - _winsorized_moments(df1, df2, p_lower, p_upper)[0]
    )
    scale_values = np.exp(log_scale + trend)
    ratio = values / scale_values
    tail = _as_array(f_sf(ratio, df1, df2))
    not_outlier = np.minimum(1.0, tail / ((average_ranks(-ratio) - 0.5) / n))
    if math.isinf(df2):
        # A cumulative maximum alone, and pi_g times the df of all n variances on
        # the common df (found by probing).
        monotone = _monotone_in(tail, not_outlier, pool_leading=False)
        df2_outlier = math.nan
        shrunk = np.where(monotone < 1.0, monotone * (n * df1), math.inf)
    else:
        monotone = _monotone_in(tail, not_outlier, pool_leading=True)
        largest = float(ratio.max())
        df2_outlier = df2
        for _ in range(_OUTLIER_ITERATIONS):
            log_p = float(f_logsf(largest, df1, df2_outlier))
            df2_outlier = df2_outlier * _LOG_HALF / log_p
        shrunk = monotone * df2 + (1.0 - monotone) * df2_outlier
    final_scale: float | FloatArray = (
        math.exp(log_scale) if covariate is None else scale_values
    )
    return RobustFDistributionFit(
        final_scale, df2, shrunk, df2_outlier, tail, 1.0 - not_outlier
    )


def _equivalent_variances(
    values: FloatArray,
    df: FloatArray,
    df_common: float,
    scale: float | FloatArray,
    df2: float,
) -> FloatArray:
    """Variances on ``df`` mapped to ``df_common`` by matching F tail probabilities."""
    scales = np.ascontiguousarray(
        np.broadcast_to(scale, values.shape), dtype=np.float64
    )
    mapped: FloatArray = _matched_quantiles(
        np.ascontiguousarray(values, dtype=np.float64),
        np.ascontiguousarray(df, dtype=np.float64),
        float(df_common),
        scales,
        float(df2),
    )
    return mapped


@jit
def _matched_quantiles(
    values: FloatArray,
    df: FloatArray,
    df_common: float,
    scales: FloatArray,
    df2: float,
) -> FloatArray:
    """scale Q(P(value / scale; df, df2); df_common, df2) in the smaller tail."""
    out = values.copy()
    for i in range(values.size):
        if df[i] == df_common:
            continue
        log_sf, log_cdf = _f_tails(values[i] / scales[i], df[i], df2)
        if log_sf <= log_cdf:
            quantile = _f_tail_quantile(log_sf, True, df_common, df2)
        else:
            quantile = _f_tail_quantile(log_cdf, False, df_common, df2)
        out[i] = scales[i] * quantile
    return out


def _moment_fallback(moments: FDistributionFit, n: int) -> RobustFDistributionFit:
    return RobustFDistributionFit(
        moments.scale,
        moments.df2,
        np.full(n, moments.df2),
        math.nan,
        np.full(n, np.nan),
        np.full(n, np.nan),
    )


def _winsorized_df2(
    df1: float, target: float, p_lower: float, p_upper: float, floor: float
) -> float:
    """The df2 at least the finite ``floor`` whose Winsorized log F variance is
    ``target``, for a target above that of df2 = inf.

    The variance falls as df2 grows, so it is solved for s = 1 / df2 in
    (0, 1 / floor] by Brent's method.
    """
    at_floor = _winsorized_moments(df1, floor, p_lower, p_upper)[1]
    if not at_floor > target:
        return floor

    def excess(s: float) -> float:
        df2 = math.inf if s == 0.0 else 1.0 / s
        variance = _winsorized_moments(df1, df2, p_lower, p_upper)[1]
        return min(variance - target, 1e300)

    s = _brent_root(excess, 0.0, 1.0 / floor)
    return math.inf if s == 0.0 else 1.0 / s


def _winsorized_moments(
    df1: float, df2: float, p_lower: float, p_upper: float
) -> tuple[float, float]:
    """Mean and variance of log win(F), F ~ F(df1, df2), Winsorized at quantiles."""
    mean, variance = _winsorized_log_f_moments(
        df1, df2, p_lower, p_upper, _QUADRATURE_NODES, _QUADRATURE_WEIGHTS
    )
    return float(mean), float(variance)


@jit
def _winsorized_log_f_moments(
    df1: float,
    df2: float,
    p_lower: float,
    p_upper: float,
    nodes: FloatArray,
    weights: FloatArray,
) -> tuple[float, float]:
    """Gauss-Legendre moments of log win(F) (Phipson et al. 2016, section 10.3).

    With q_l and q_u the quantiles of F at p_lower and 1 - p_upper, the middle part
    of each moment is an integral over u = f / (1 + f) from a = q_l / (1 + q_l) to
    b = q_u / (1 + q_u), where the density of f becomes x f(x) / (u (1 - u)).
    Returns NaN and an infinite variance if a quantile is not finite and positive.
    """
    q_lower = f_ppf_scalar(p_lower, df1, df2)
    q_upper = f_isf_scalar(p_upper, df1, df2)
    if not (0.0 < q_lower and q_upper < np.inf):
        return np.nan, np.inf
    log_lower = math.log(q_lower)
    log_upper = math.log(q_upper)
    c_lower = 1.0 / (1.0 + q_lower)  # 1 - a
    c_upper = 1.0 / (1.0 + q_upper)  # 1 - b
    a = q_lower * c_lower
    half = 0.5 * (q_upper - q_lower) * c_lower * c_upper  # (b - a) / 2
    centre = a + half
    complement = c_lower - half  # 1 - centre
    log_f = np.empty(nodes.size)
    mass = np.empty(nodes.size)
    total = 0.0
    for i in range(nodes.size):
        u = centre + half * nodes[i]
        v = complement - half * nodes[i]
        log_f[i] = math.log(u) - math.log(v)
        f = u / v
        mass[i] = weights[i] * math.exp(_f_log_x_density(f, df1, df2)) / (u * v)
        total += mass[i] * log_f[i]
    mean = p_lower * log_lower + half * total + p_upper * log_upper
    total = 0.0
    for i in range(nodes.size):
        deviation = log_f[i] - mean
        total += mass[i] * deviation * deviation
    deviation_lower = log_lower - mean
    deviation_upper = log_upper - mean
    variance = (
        p_lower * deviation_lower * deviation_lower
        + half * total
        + p_upper * deviation_upper * deviation_upper
    )
    return mean, variance


def _brent_root(function: Callable[[float], float], low: float, high: float) -> float:
    """Root of an increasing function on [low, high], with f(low) < 0 < f(high).

    Brent's method (Brent 1973, chapter 4): inverse quadratic or linear
    interpolation, with bisection whenever interpolation does not shrink the
    bracket fast enough; it stops when the bracket is a few units in the last place
    wide.
    """
    a, b = low, high
    fa, fb = function(a), function(b)
    c, fc = a, fa
    step = previous = b - a
    for _ in range(200):
        if (fb > 0.0) == (fc > 0.0):
            c, fc = a, fa
            step = previous = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance = 2.0 * _EPS * abs(b) + 1e-300
        middle = 0.5 * (c - b)
        if abs(middle) <= tolerance or fb == 0.0:
            return b
        if abs(previous) >= tolerance and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * middle * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * middle * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * middle * q - abs(tolerance * q), abs(previous * q)):
                previous, step = step, p / q
            else:
                previous = step = middle
        else:
            previous = step = middle
        a, fa = b, fb
        b += step if abs(step) > tolerance else math.copysign(tolerance, middle)
        fb = function(b)
    return b


def _monotone_in(
    key: FloatArray, values: FloatArray, *, pool_leading: bool
) -> FloatArray:
    """values made non-decreasing in key by a cumulative maximum in increasing key.

    With ``pool_leading`` (Phipson et al. 2016, section 4), the leading values up to
    the first minimum of their cumulative mean are first set to that minimum.
    """
    order = np.argsort(key, kind="stable")
    ordered = values[order]
    if pool_leading:
        cumulative_mean = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
        first_minimum = int(np.argmin(cumulative_mean))
        ordered[: first_minimum + 1] = cumulative_mean[first_minimum]
    result = np.empty(values.size)
    result[order] = np.maximum.accumulate(ordered)
    return result


# ---------------------------------------------------------------------------------------
# Likelihood estimator
# ---------------------------------------------------------------------------------------


def _fit_likelihood(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    prior_weights: FloatArray,
    robust: bool,
) -> LikelihoodFDistributionFit:
    """Weighted moments and profile likelihood; see
    :func:`fit_f_distribution_likelihood`."""
    usable = _likelihood_usable(x, d, prior_weights)
    positive = int(np.count_nonzero(x[usable] > 0.0))
    if positive < 2:
        raise ValueError(
            "the likelihood estimator needs at least two usable variances above zero "
            f"(finite, on at least {LIKELIHOOD_MIN_DF} degrees of freedom and with a "
            f"positive prior weight), got {positive}; limma's fitFDistUnequalDF1 "
            "returns NA and squeezeVar stops here (found by probing). The method of "
            "moments (prior_estimator='moments') does not need two"
        )
    weights = np.where(usable, prior_weights, 0.0)
    location, df2, trended = _likelihood_location_and_df2(
        x, d, covariate, weights, usable
    )
    scale = np.exp(location - _log_minus_digamma_scalar(0.5 * df2))
    first = LikelihoodFDistributionFit(
        _scale_value(scale, trended), df2, np.full(x.size, df2), math.nan
    )
    # With two variances above zero there is no robust stage (found by probing).
    if not robust or positive <= _CONSTANT_TREND_MAX:
        return first
    return _robust_likelihood(x, d, covariate, weights, usable, scale, first)


def _likelihood_usable(
    x: FloatArray, d: FloatArray, prior_weights: FloatArray
) -> npt.NDArray[np.bool_]:
    """The variances that take part in a likelihood fit."""
    result: npt.NDArray[np.bool_] = (
        np.isfinite(x) & (d >= LIKELIHOOD_MIN_DF) & (prior_weights > 0.0)
    )
    return result


def _likelihood_location_and_df2(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    weights: FloatArray,
    usable: npt.NDArray[np.bool_],
) -> tuple[FloatArray, float, bool]:
    """The location of the e_g, d0, and whether the location is a trend.

    ``weights`` are the prior weights, zero where a variance is not usable; at least
    two usable variances are above zero.
    """
    positive = usable & (x > 0.0)
    floor = _LIKELIHOOD_FLOOR * float(np.median(x[positive]))
    # A missing variance counts as zero, so it is raised to the floor, and a df below
    # the minimum counts as 1 (found by probing); they enter only the trend.
    raised = np.where(np.isnan(x), floor, np.maximum(x, floor))
    shape = 0.5 * np.where(d >= LIKELIHOOD_MIN_DF, d, 1.0)
    e = np.log(raised) + _log_minus_digamma(shape)
    precision = weights / _as_array(trigamma(shape))
    # With at most two usable variances above zero the location is a constant, and
    # the scale one value, as limma's are (found by probing); limma returns NaN when
    # some variances are not usable as well.
    trended = False
    if covariate is not None and int(positive.sum()) > _CONSTANT_TREND_MAX:
        trended = True
        location = _likelihood_trend(e, covariate, precision)
    else:
        mean = accurate_sum(precision * e)[0] / accurate_sum(precision)[0]
        location = np.full(x.size, mean)
    df2 = _profile_likelihood_df2(
        raised[usable], d[usable], location[usable], weights[usable]
    )
    return location, df2, trended


def _likelihood_trend(
    e: FloatArray, covariate: FloatArray, precision: FloatArray
) -> FloatArray:
    """limma's loessFit trend of e on the covariate, NaN where it is NaN.

    The weights are divided by their upper quartile (R's type 7, over the features in
    the trend, zeros included) and kept within ``_TREND_WEIGHT_RANGE``, as limma
    calls loessFit (found by probing). loessFit fits R's lowess for equal weights;
    for unequal ones it fits a weighted least squares line when 1 <= span n and
    span (n - 4) < 1, which with the span of 1 used up to 500 features means at
    most four features (found by probing; its help page says that it fits a line
    when there are too few observations), and limma's weighted LOWESS otherwise.
    """
    known = ~np.isnan(covariate)
    z = np.ascontiguousarray(covariate[known])
    values = np.ascontiguousarray(e[known])
    w = precision[known]
    n = int(z.size)
    span = choose_lowess_span(n, small_n=_TREND_SMALL_N)
    upper_quartile = float(np.quantile(w, 0.75))
    low, high = _TREND_WEIGHT_RANGE
    if upper_quartile > 0.0:
        relative = np.clip(w / upper_quartile, low, high)
    else:
        # 0 / 0 counts as 0 and w / 0 as inf before the clipping (found by probing).
        relative = np.where(w > 0.0, high, low)
    if (relative == relative[0]).all():
        fitted = lowess(z, values, span=span, iterations=0)
    elif span * n >= 1.0 and span * (n - 4) < 1.0:
        fitted = _weighted_line(z, values, relative)
    else:
        fitted = _weighted_lowess(z, values, relative, span)
    trend = np.full(e.size, np.nan)
    trend[known] = fitted
    return trend


def _weighted_line(x: FloatArray, y: FloatArray, weights: FloatArray) -> FloatArray:
    """Fitted values of the weighted least squares line of y on x, as R's lm.wfit.

    As R's QR decomposition with its tolerance of 1e-7, the slope is left out, and
    the fit is the weighted mean, when the weighted sum of squares of x about its
    mean is below 1e-14 times that about zero.
    """
    total = float(np.sum(weights))
    mean_x = float(np.sum(weights * x)) / total
    mean_y = float(np.sum(weights * y)) / total
    centred = x - mean_x
    sxx = float(np.sum(weights * centred * centred))
    if sxx == 0.0 or sxx < _LINE_RANK_TOLERANCE**2 * float(np.sum(weights * x * x)):
        return np.full(x.size, mean_y)
    slope = float(np.sum(weights * centred * (y - mean_y))) / sxx
    fitted: FloatArray = mean_y + slope * centred
    return fitted


def _profile_likelihood_df2(
    x: FloatArray, d: FloatArray, location: FloatArray, weights: FloatArray
) -> float:
    """The d0 minimising minus twice the log-likelihood, searched as R's optimize."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64)
    location = np.ascontiguousarray(location, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    def deviance(v: float) -> float:
        df2 = 2.0 * v / (1.0 - v)
        shift = _log_minus_digamma_scalar(0.5 * df2)
        return float(_minus_twice_log_likelihood(x, d, location, weights, df2, shift))

    v = brent_minimum(deviance, _DF2_SEARCH[0], _DF2_SEARCH[1], _SEARCH_TOLERANCE)
    return 2.0 * v / (1.0 - v)


@jit
def _minus_twice_log_likelihood(
    x: FloatArray,
    d: FloatArray,
    location: FloatArray,
    weights: FloatArray,
    df2: float,
    shift: float,
) -> float:
    """-2 sum_g w_g log f(s_g^2), with s_g^2 / s0_g^2 ~ F(d_g, df2) and
    log s0_g^2 = location_g - shift, summed with compensation."""
    total = 0.0
    error = 0.0
    for i in range(x.size):
        log_scale = location[i] - shift
        ratio = x[i] / math.exp(log_scale)
        log_density = _f_log_x_density(ratio, d[i], df2) - math.log(ratio)
        total, rounding = two_sum(total, weights[i] * (log_density - log_scale))
        error += rounding
    return -2.0 * (total + error)


def _robust_likelihood(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    weights: FloatArray,
    usable: npt.NDArray[np.bool_],
    scale: FloatArray,
    first: LikelihoodFDistributionFit,
) -> LikelihoodFDistributionFit:
    """The robust stage of the likelihood estimator; see
    :func:`fit_f_distribution_likelihood`."""
    index = np.flatnonzero(usable)
    ratio = np.ascontiguousarray(x[index] / scale[index])
    df1 = np.ascontiguousarray(d[index])
    log_upper, log_lower = _f_tail_logs(ratio, df1, first.df2)
    # Two-sided tail probabilities of the raw variances (zeros included), adjusted by
    # Benjamini and Hochberg's method; below 0.3 is an outlier (found by probing).
    two_sided = np.minimum(1.0, 2.0 * np.exp(np.minimum(log_upper, log_lower)))
    adjusted = _as_array(adjust_p_values(two_sided, "bh"))
    outlier = adjusted < _OUTLIER_LEVEL
    if not outlier.any():
        return first
    refit_weights = weights.copy()
    refit_weights[index] *= np.where(outlier, adjusted, 1.0)
    refit_usable = usable & (refit_weights > 0.0)
    if int(np.count_nonzero(x[refit_usable] > 0.0)) < 2:
        # Only when the adjusted probabilities of all but one of the variances above
        # zero are 0, which takes tail probabilities below the smallest double.
        raise ValueError(
            "the second fit of the robust likelihood estimator keeps fewer than two "
            "variances above zero: the outlier probabilities of the others are 0; "
            "use robust=False"
        )
    location, df2, trended = _likelihood_location_and_df2(
        x, d, covariate, refit_weights, refit_usable
    )
    robust_scale = _scale_value(
        np.exp(location - _log_minus_digamma_scalar(0.5 * df2)), trended
    )
    shrunk = np.full(x.size, df2)
    # The probability of not being an outlier, from the first fit's upper tail.
    upper = np.exp(log_upper)
    n = index.size
    naive = np.minimum(1.0, upper / ((average_ranks(-ratio) - 0.5) / n))
    not_outlier = _monotone_in(log_upper, naive, pool_leading=True)
    if (not_outlier == 1.0).all():
        return LikelihoodFDistributionFit(robust_scale, df2, shrunk, math.nan)
    # Two fixed-point steps for the most extreme variance (found by probing).
    most = int(np.argmin(log_upper))
    df2_outlier = df2 * _LOG_HALF / float(log_upper[most])
    df2_outlier *= _LOG_HALF / f_logsf_scalar(
        float(ratio[most]), float(df1[most]), df2_outlier
    )
    shrunk[index] = not_outlier * df2 + (1.0 - not_outlier) * df2_outlier
    return LikelihoodFDistributionFit(robust_scale, df2, shrunk, df2_outlier)


@jit
def _f_tail_logs(
    ratio: FloatArray, df1: FloatArray, df2: float
) -> tuple[FloatArray, FloatArray]:
    """log P(F > ratio) and log P(F <= ratio) for F ~ F(df1, df2), element-wise."""
    n = ratio.size
    upper = np.empty(n)
    lower = np.empty(n)
    for i in range(n):
        upper[i], lower[i] = _f_tails(ratio[i], df1[i], df2)
    return upper, lower


def _scale_value(scale: FloatArray, trended: bool) -> float | FloatArray:
    """A constant scale as a float, a trended one as an array."""
    return scale if trended else float(scale[0])


def _log_minus_digamma_scalar(a: float) -> float:
    return float(_log_minus_digamma(np.array([a]))[0])


# ---------------------------------------------------------------------------------------
# Squeezing and moderated statistics
# ---------------------------------------------------------------------------------------


def _squeeze(
    x: FloatArray,
    d: FloatArray,
    covariate: FloatArray | None,
    usable: npt.NDArray[np.bool_],
    robust: bool,
    p_lower: float,
    p_upper: float,
    estimator: str,
) -> SqueezedVariances:
    if x.size <= 2:
        return SqueezedVariances(x.copy(), x.copy(), 0.0)
    prior_variance: float | FloatArray
    df_prior: float | FloatArray
    if estimator == "likelihood":
        # limma's likelihood estimator does not stop on mostly zero variances.
        _check_zero_variances(x, usable, robust=False)
        # A variance on zero df counts as missing, as in limma's squeezeVar; it
        # enters only the trend (found by probing).
        missing = np.where(d > 0.0, x, np.nan)
        fitted = _fit_likelihood(missing, d, covariate, np.ones(x.size), robust)
        prior_variance = fitted.scale
        # A robust fit gives each feature its prior df once it finds outliers.
        df_prior = fitted.df2 if math.isnan(fitted.df2_outlier) else fitted.df2_shrunk
    elif robust:
        _check_zero_variances(x, usable, robust)
        robust_fit = _fit_robust(x, d, covariate, usable, p_lower, p_upper)
        prior_variance, df_prior = robust_fit.scale, robust_fit.df2_shrunk
    else:
        _check_zero_variances(x, usable, robust)
        moments = _fit_moments(x, d, covariate, usable)
        prior_variance, df_prior = moments.scale, moments.df2
    return SqueezedVariances(
        posterior_variances(x, d, prior_variance, df_prior), prior_variance, df_prior
    )


def _moderate(
    fit: LinearModelFit,
    covariate: FloatArray | None,
    robust: bool,
    winsor_tail_p: float | Sequence[float],
    prior_estimator: Any,
) -> tuple[SqueezedVariances, FloatArray, str]:
    """Squeezed variances of a fit, the total df of its tests and the estimator."""
    checked_flag(robust, "robust", expected="a bool")
    lower, upper = _tail_proportions(winsor_tail_p)
    variances = fit.sigma**2
    df = _zero_tiny_df(fit.df_residual)
    estimator = _prior_estimator(prior_estimator, fit.df_residual)
    usable = np.isfinite(variances) & (df > 0.0)
    if not usable.any():
        raise ValueError(
            "no feature has residual degrees of freedom and a finite residual "
            "variance, so the prior cannot be estimated; the design leaves no "
            "residual degrees of freedom"
        )
    if covariate is not None:
        _check_covariate_nan(covariate, usable)
        covariate = _finite_covariate(covariate)
    squeezed = _squeeze(
        variances, df, covariate, usable, robust, lower, upper, estimator
    )
    df_prior = np.broadcast_to(squeezed.df_prior, df.shape)
    df_total = np.minimum(df + df_prior, float(df.sum()))
    return squeezed, df_total, estimator


def _f_statistics(
    t: FloatArray, fit: LinearModelFit, exact: bool
) -> tuple[FloatArray, int]:
    """Moderated F-statistics and their numerator degrees of freedom."""
    n_features, k = t.shape
    correlation = correlation_matrix(fit.cov_coefficients, unit_diagonal=True)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    keep = eigenvalues > _F_RANK_TOLERANCE * eigenvalues.max()
    rank = int(keep.sum())
    per_feature = fit.cov_coefficients_per_feature
    if not exact or per_feature is None:
        with np.errstate(invalid="ignore"):
            q = (t @ eigenvectors[:, keep]) / np.sqrt(eigenvalues[keep])
            f_value: FloatArray = np.sum(q * q, axis=1) / rank
        return f_value, rank
    f_value = np.full(n_features, np.nan)
    complete = np.isfinite(t).all(axis=1)
    if complete.any():
        covariances = per_feature[complete]
        scale = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = covariances / (
                scale[:, :, np.newaxis] * scale[:, np.newaxis, :]
            )
        values, vectors = np.linalg.eigh(correlations)
        # The rank largest eigenpairs; eigh sorts eigenvalues in increasing order.
        values, vectors = values[:, k - rank :], vectors[:, :, k - rank :]
        q = np.einsum("gk,gkr->gr", t[complete], vectors)
        with np.errstate(invalid="ignore", divide="ignore"):
            statistic = np.sum(q * q / values, axis=1) / rank
        degenerate = values[:, 0] <= _F_RANK_TOLERANCE * values[:, -1]
        f_value[complete] = np.where(degenerate, np.nan, statistic)
    return f_value, rank


# ---------------------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------------------


def _variance_vector(values: Any) -> FloatArray:
    x = real_array(
        unmasked_array(values, "variances", missing="NaN"),
        "variances",
        boolean=False,
        objects=False,
    )
    if x.ndim != 1:
        raise ValueError(f"variances must be a 1-D array, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("variances must hold at least one value")
    with np.errstate(invalid="ignore"):
        invalid = (x < 0.0) | np.isinf(x)
    if invalid.any():
        raise ValueError(
            "variances must be non-negative and finite (NaN marks a missing value); "
            f"{int(invalid.sum())} are negative or infinite"
        )
    return x


def _df_vector(df: Any, n: int) -> FloatArray:
    """The degrees of freedom, those below 1e-15 set to zero."""
    return _zero_tiny_df(_df_values(df, n))


def _df_values(df: Any, n: int) -> FloatArray:
    """The degrees of freedom as given, checked and one per variance."""
    d = real_array(unmasked_array(df, "df"), "df", boolean=False, objects=False)
    if d.size == 1 and d.ndim <= 1:
        d = np.full(n, float(d.reshape(-1)[0]))
    elif d.shape != (n,):
        raise ValueError(
            f"df must be a single value or have one value per variance ({n}), "
            f"got shape {d.shape}"
        )
    if not (np.isfinite(d).all() and (d >= 0.0).all()):
        raise ValueError("df must be finite and non-negative")
    return d


def _zero_tiny_df(d: FloatArray) -> FloatArray:
    result: FloatArray = np.where(d < _ZERO_DF, 0.0, d)
    return result


def _usable(x: FloatArray, d: FloatArray) -> npt.NDArray[np.bool_]:
    usable = np.isfinite(x) & (d > 0.0)
    if not usable.any():
        raise ValueError(
            "no variance is usable: every one is NaN or has zero degrees of freedom"
        )
    return usable


def _check_zero_variances(
    x: FloatArray, usable: npt.NDArray[np.bool_], robust: bool
) -> None:
    """Warn when most usable variances are zero; the robust estimator raises.

    With at most two usable variances the robust estimator returns the moment
    estimate, so it only warns, as limma does (found by probing).
    """
    n = int(usable.sum())
    zeros = int(np.count_nonzero(x[usable] == 0.0))
    if 2 * zeros <= n:
        return
    if robust and n > 2:
        raise ValueError(
            f"{zeros} of the {n} usable variances are zero, and the robust estimator "
            "needs more than half of them to be positive (limma stops here too); "
            "use robust=False, or leave out the features without variation"
        )
    warnings.warn(
        "more than half of the variances are exactly zero, so the estimated prior "
        "is unreliable",
        RuntimeWarning,
        skip_file_prefixes=(PACKAGE_DIRECTORY,),
    )


def _covariate_vector(
    covariate: Any, usable: npt.NDArray[np.bool_], unusable: str = _MOMENT_UNUSABLE
) -> FloatArray | None:
    if covariate is None:
        return None
    z = real_array(
        unmasked_array(covariate, "covariate", missing="NaN"),
        "covariate",
        boolean=False,
        objects=False,
    )
    if z.shape != usable.shape:
        raise ValueError(
            f"covariate must have one value per variance ({usable.size}), "
            f"got shape {z.shape}"
        )
    _check_covariate_nan(z, usable, unusable)
    return _finite_covariate(z)


def _check_covariate_nan(
    z: FloatArray, usable: npt.NDArray[np.bool_], unusable: str = _MOMENT_UNUSABLE
) -> None:
    missing = np.isnan(z) & usable
    if missing.any():
        raise ValueError(
            f"the covariate is NaN for {int(missing.sum())} features with a usable "
            f"variance; it may be NaN only where the variance {unusable}"
        )


def _finite_covariate(z: FloatArray) -> FloatArray:
    """Infinite values moved one unit beyond the finite range, as limma does."""
    infinite = np.isinf(z)
    if not infinite.any():
        return z
    finite = np.isfinite(z)
    if not finite.any():
        raise ValueError("the covariate must have at least one finite value")
    z = z.copy()
    z[infinite & (z < 0.0)] = float(z[finite].min()) - _INFINITE_COVARIATE_OFFSET
    z[infinite & (z > 0.0)] = float(z[finite].max()) + _INFINITE_COVARIATE_OFFSET
    return z


def _tail_proportions(winsor_tail_p: Any) -> tuple[float, float]:
    proportions = unmasked_array(winsor_tail_p, "winsor_tail_p")
    values = real_array(
        proportions, "winsor_tail_p", boolean=False, objects=False
    ).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2:
        raise ValueError(
            "winsor_tail_p must hold one or two proportions (lower and upper tail), "
            f"got {values.size}"
        )
    if not ((values > 0.0) & (values < 0.5)).all():
        raise ValueError(
            f"winsor_tail_p must lie strictly between 0 and 0.5, got {values.tolist()}"
        )
    return float(values[0]), float(values[1])


def _trend_covariate(trend: Any, fit: LinearModelFit) -> FloatArray | None:
    if isinstance(trend, (bool, np.bool_)):
        return np.asarray(fit.mean_expression, dtype=np.float64) if trend else None
    z = real_array(
        unmasked_array(trend, "trend", missing="NaN"),
        "trend",
        boolean=False,
        objects=False,
    )
    if z.shape != fit.sigma.shape:
        raise ValueError(
            "trend must be a bool or one covariate value per feature "
            f"({fit.sigma.size}), got shape {z.shape}"
        )
    return z


def _f_method(f_statistic: Any, fit: LinearModelFit) -> bool:
    if f_statistic is None:
        return fit.exact_standard_errors
    if f_statistic not in ("exact", "limma"):
        raise ValueError(
            f"f_statistic must be 'exact', 'limma' or None, got {f_statistic!r}"
        )
    if f_statistic == "exact" and not fit.exact_standard_errors:
        raise ValueError(
            "the standard errors of this fit are limma's approximation (from "
            "contrasts_fit with method='limma'), so its F-statistics cannot be "
            "exact; use f_statistic='limma', or contrast with method='exact'"
        )
    return bool(f_statistic == "exact")


def _treat_threshold(lfc: Any, fc: Any) -> float:
    if lfc is not None and fc is not None:
        raise ValueError("give the threshold as lfc or as fc, not both")
    if lfc is not None:
        value = real_number(lfc, "lfc")
        if not math.isfinite(value):
            raise ValueError(f"lfc must be finite, got {value}")
        return abs(value)
    ratio = 1.2 if fc is None else real_number(fc, "fc")
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f"fc must be positive and finite, got {ratio}")
    return abs(math.log2(ratio))


def _coefficient_limits(limits: Any) -> tuple[float, float]:
    values = real_array(
        unmasked_array(limits, "stdev_coef_lim"),
        "stdev_coef_lim",
        boolean=False,
        objects=False,
    )
    values = values.reshape(-1)
    if values.size != 2:
        raise ValueError(
            f"stdev_coef_lim must hold two values (lower, upper), got {values.size}"
        )
    low, high = float(values[0]), float(values[1])
    if not (0.0 <= low <= high):
        raise ValueError(
            "stdev_coef_lim must satisfy 0 <= lower <= upper (upper may be "
            f"infinite), got {values.tolist()}"
        )
    return low, high


def _open_unit_interval(value: Any, name: str) -> float:
    number = real_number(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {number}")
    return number


def _check_fit(fit: Any) -> None:
    if not isinstance(fit, LinearModelFit):
        raise TypeError(f"fit must be a LinearModelFit, got {type(fit).__name__}")


def _prior_estimator(prior_estimator: Any, d: FloatArray) -> str:
    """The estimator to use: limma's legacy=NULL rule for None.

    ``d`` are the degrees of freedom as given, before those below 1e-15 are set to
    zero.
    """
    if prior_estimator is None:
        # The positive df, those of missing variances and those below 1e-15
        # included, compared exactly (found by probing): df of zero do not count.
        positive = d[d > 0.0]
        unequal = positive.size > 0 and positive.min() < positive.max()
        return "likelihood" if unequal else "moments"
    # A masked value compares as masked, which is false, so test the type first.
    if isinstance(prior_estimator, str) and prior_estimator in (
        "moments",
        "likelihood",
    ):
        return prior_estimator
    raise ValueError(
        "prior_estimator must be 'moments' (the method of moments, limma's "
        "legacy=TRUE), 'likelihood' (weighted moments and profile likelihood, "
        "limma's fitFDistUnequalDF1 and legacy=FALSE) or None (limma's default, "
        f"legacy=NULL); got {prior_estimator!r}"
    )


def _prior_weight_vector(weights: Any, n: int) -> FloatArray:
    if weights is None:
        return np.ones(n)
    w = real_array(
        unmasked_array(weights, "prior_weights"),
        "prior_weights",
        boolean=False,
        objects=False,
    )
    if w.shape != (n,):
        raise ValueError(
            f"prior_weights must have one value per variance ({n}), got shape {w.shape}"
        )
    if not (np.isfinite(w).all() and (w >= 0.0).all()):
        raise ValueError(
            "prior_weights must be finite and non-negative (0 leaves a feature out)"
        )
    return w


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------


def _median_or_one(values: FloatArray) -> float:
    median = float(np.median(values))
    return median if median > 0.0 else 1.0


def _as_array(values: Any) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _frozen(values: Any) -> Any:
    if isinstance(values, float):
        return values
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array
