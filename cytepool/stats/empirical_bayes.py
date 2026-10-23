"""Empirical Bayes moderation of feature-wise variances, as in limma's ``eBayes``.

A linear model fit gives each feature g a residual variance s_g^2 on d_g degrees of
freedom. The hierarchical model of Smyth (2004) assumes

    s_g^2 | sigma_g^2 ~ sigma_g^2 chi^2(d_g) / d_g,
    1 / sigma_g^2 ~ chi^2(d0) / (d0 s0^2),

so that s_g^2 ~ s0^2 F(d_g, d0) marginally. The prior (s0^2, d0) is estimated from all
features, and each variance is replaced by its posterior mean
s~_g^2 = (d0 s0^2 + d_g s_g^2) / (d0 + d_g). A moderated t-statistic, which uses s~_g in
place of s_g, follows a t distribution on d0 + d_g degrees of freedom under the null
hypothesis; an infinite d0 gives every feature the prior variance.

Estimation of the prior:
    - :func:`fit_f_distribution`: the method of moments on e_g = log s_g^2 -
      psi(d_g / 2) + log(d_g / 2), whose variance is psi'(d0 / 2) plus the mean of
      psi'(d_g / 2) (Smyth 2004, section 6.2). When the variance of the e_g is no larger
      than chi-squared sampling explains, d0 is infinite and s0^2 is the mean variance.
      With a covariate (limma-trend uses the mean log-expression) the moments are taken
      about a natural cubic spline fit of e_g, and s0^2 becomes a trend (Phipson et al.
      2016, section 10.1).
    - :func:`fit_f_distribution_robust`: the moments of the log variances Winsorized at
      their 5% and 90% quantiles are matched to those of a Winsorized log F variable,
      computed by Gauss-Legendre quadrature, so that hypervariable features do not lower
      d0. Each feature then gets its own prior df, smaller for outliers (Phipson et al.
      2016). With a covariate, the log variances are first detrended by LOWESS.

Statistics: :func:`moderated_statistics` computes moderated t and F statistics, their
p-values and the log-odds of differential expression (the B-statistic of Loennstedt
and Speed 2002, in the form of Smyth 2004, section 5); :func:`treat` tests whether
coefficients exceed a threshold in absolute value (McCarthy and Smyth 2009).

Results match limma 3.62.1 (``fitFDist``, ``fitFDistRobustly``, ``squeezeVar``,
``eBayes`` and ``treat`` with ``legacy=TRUE``). Conventions that limma's documentation
leaves open were settled by running it as a black box; comments mark them "found by
probing". One estimator is implemented: the method of moments above, which limma calls
the legacy method; the ``prior_estimator`` argument names it. When the residual degrees
of freedom differ between features (with missing values or zero weights), limma 3.61.8
and later estimate the prior differently by default (``fitFDistUnequalDF1``, which
downweights features with fewer degrees of freedom and estimates d0 by profile maximum
likelihood; see Chen et al. 2024), so compare with ``eBayes(fit, legacy=TRUE)``.

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3.

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust
    hyperparameter estimation protects against hypervariable genes and improves power
    to detect differential expression. Annals of Applied Statistics 10, 946-963.

    McCarthy DJ, Smyth GK (2009). Testing significance relative to a fold-change
    threshold is a TREAT. Bioinformatics 25, 765-771.

    Loennstedt I, Speed TP (2002). Replicated microarray data. Statistica Sinica 12,
    31-46.

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2024). edgeR 4.0: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. bioRxiv 2024.01.21.576131.
"""

import math
import numbers
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import column_space
from cytepool.stats.distributions import (
    _f_log_x_density,
    _f_tail_quantile,
    _f_tails,
    f_isf_scalar,
    f_logsf,
    f_ppf_scalar,
    f_sf,
    t_isf,
    t_sf,
)
from cytepool.stats.linear_models import LinearModelFit
from cytepool.stats.smoothing import _quantile_knots, lowess, natural_spline_basis
from cytepool.stats.special import digamma, trigamma, trigamma_inverse

__all__ = [
    "FDistributionFit",
    "ModeratedFit",
    "RobustFDistributionFit",
    "SqueezedVariances",
    "TreatFit",
    "fit_f_distribution",
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
# Warnings are attributed to the first caller outside the package.
_PACKAGE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""
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


class SqueezedVariances(NamedTuple):
    """Posterior variances and the prior they were shrunk towards.

    Attributes:
        posterior_variance: ``(n,)`` posterior means of the variances.
        prior_variance: The prior variance: a float, or one value per feature with a
            covariate or with at most two variances.
        prior_df: The prior degrees of freedom: a float, or one value per feature
            when estimated robustly.
    """

    posterior_variance: FloatArray
    prior_variance: float | FloatArray
    prior_df: float | FloatArray


@dataclass(frozen=True, eq=False, repr=False)
class ModeratedFit:
    """Moderated t and F statistics and log-odds of a linear model fit.

    Rows of the per-feature arrays are features and columns are the coefficients
    (or contrasts) of ``fit``. The arrays are read-only.

    Attributes:
        fit: The linear model fit.
        prior_variance: The prior variance s0^2 (limma's ``s2.prior``): a float, or
            one value per feature with a trend or with at most two features.
        prior_df: The prior degrees of freedom d0 (``df.prior``): a float, possibly
            infinite, or one value per feature when estimated robustly.
        posterior_variance: ``(n_features,)`` posterior variances (``s2.post``).
        df_total: ``(n_features,)`` degrees of freedom of the moderated t-statistics,
            ``df_residual + prior_df`` capped at the residual degrees of freedom
            summed over all features (``df.total``).
        t: ``(n_features, n_coefficients)`` moderated t-statistics.
        p_value: Two-sided p-values of ``t`` on ``df_total`` degrees of freedom.
        lods: Log-odds that each coefficient is non-zero (the B-statistic), on the
            natural log scale.
        coefficient_prior_variance: ``(n_coefficients,)`` unscaled prior variance v0
            of a non-zero coefficient, which is N(0, v0 sigma_g^2) (``var.prior``).
        f_statistic: ``(n_features,)`` moderated F-statistics testing all
            coefficients equal to zero.
        f_p_value: P-values of ``f_statistic`` on ``f_df1`` and
            ``df_residual + prior_df`` degrees of freedom; limma does not cap the
            denominator degrees of freedom here (found by probing).
        f_df1: Numerator degrees of freedom of the F-tests: the rank of the
            correlation matrix of the coefficients.
        proportion: The assumed proportion of non-zero coefficients behind ``lods``.
    """

    fit: LinearModelFit
    prior_variance: float | FloatArray
    prior_df: float | FloatArray
    posterior_variance: FloatArray
    df_total: FloatArray
    t: FloatArray
    p_value: FloatArray
    lods: FloatArray
    coefficient_prior_variance: FloatArray
    f_statistic: FloatArray
    f_p_value: FloatArray
    f_df1: int
    proportion: float

    def __repr__(self) -> str:
        return (
            f"ModeratedFit(n_features={self.t.shape[0]}, "
            f"coefficient_names={self.fit.coefficient_names!r}, "
            f"prior_df={_summary(self.prior_df)}, "
            f"prior_variance={_summary(self.prior_variance)})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class TreatFit:
    """Moderated t-tests relative to a threshold on the absolute coefficients.

    Attributes:
        fit: The linear model fit.
        lfc: The threshold tau of the null hypothesis |beta| <= tau, on the scale of
            the coefficients (log2 fold changes for log2 expression).
        prior_variance: The prior variance, as in :class:`ModeratedFit`.
        prior_df: The prior degrees of freedom, as in :class:`ModeratedFit`.
        posterior_variance: ``(n_features,)`` posterior variances.
        df_total: ``(n_features,)`` degrees of freedom of the tests.
        t: ``(n_features, n_coefficients)`` moderated t-statistics shrunk towards
            zero by the threshold: sign(t) max(|t| - tau / se, 0), with se the
            moderated standard error.
        p_value: P-values of the tests of |beta| <= tau.
    """

    fit: LinearModelFit
    lfc: float
    prior_variance: float | FloatArray
    prior_df: float | FloatArray
    posterior_variance: FloatArray
    df_total: FloatArray
    t: FloatArray
    p_value: FloatArray

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
        TypeError: If an input is not real-valued.
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
        TypeError: If an input is not real-valued.
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


def squeeze_variances(
    variances: npt.ArrayLike,
    df: npt.ArrayLike,
    *,
    covariate: npt.ArrayLike | None = None,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments"] = "moments",
) -> SqueezedVariances:
    """Shrink variances towards a prior by empirical Bayes.

    Matches limma's ``squeezeVar``. The prior is estimated by
    :func:`fit_f_distribution`, or :func:`fit_f_distribution_robust` when
    ``robust``, and each variance is replaced by its posterior mean
    (d0 s0^2 + d_g s_g^2) / (d0 + d_g). A feature with zero df, or an infinite
    prior df, gets the prior variance; a NaN variance with positive df stays NaN.
    With at most two variances there is no moderation: the prior df is 0 and each
    variance is its own prior, as in limma 3.61.12 and later.

    Args:
        variances: ``(n,)`` non-negative variances; NaN marks a missing value.
        df: Degrees of freedom of the variances: one value, or ``(n,)``.
        covariate: ``(n,)`` covariate for a trended prior variance.
        robust: Whether to estimate the prior robustly.
        winsor_tail_p: Winsorized tail proportions of the robust estimator.
        prior_estimator: How the prior is estimated. The only option, "moments",
            is the method of moments of Smyth (2004), Winsorized when ``robust``
            (Phipson et al. 2016): limma's ``legacy=TRUE``. When the df differ
            between features, limma 3.61.8 and later use another estimator by
            default (``fitFDistUnequalDF1``).

    Returns:
        The posterior variances, the prior variance and the prior df.

    Raises:
        TypeError: If an input is not real-valued, or ``robust`` not a bool.
        ValueError: As :func:`fit_f_distribution`, or if an option is invalid.
    """
    x = _variance_vector(variances)
    d = _df_vector(df, x.size)
    usable = _usable(x, d)
    z = _covariate_vector(covariate, usable)
    lower, upper = _tail_proportions(winsor_tail_p)
    _check_flag(robust, "robust")
    _check_prior_estimator(prior_estimator)
    return _squeeze(x, d, z, usable, bool(robust), lower, upper)


def moderated_statistics(
    fit: LinearModelFit,
    *,
    proportion: float = 0.01,
    stdev_coef_lim: Sequence[float] = (0.1, 4.0),
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    f_statistic: Literal["exact", "limma"] | None = None,
    prior_estimator: Literal["moments"] = "moments",
) -> ModeratedFit:
    """Empirical Bayes moderated t and F statistics and log-odds of a fit.

    Matches limma's ``eBayes(fit, legacy=TRUE)``, except that by default the
    F-statistics of fits with missing values or observation weights are exact
    rather than limma's approximation (see below and ``f_statistic``). The residual
    variances of the fit are squeezed by :func:`squeeze_variances`; the moderated
    t-statistics divide the coefficients by ``stdev_unscaled`` times the posterior
    standard deviations and have ``df_total`` degrees of freedom.

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
        prior_estimator: See :func:`squeeze_variances`.

    Returns:
        The moderated statistics.

    Raises:
        TypeError: If ``fit`` is not a LinearModelFit or an option has the wrong
            type.
        ValueError: If an option is invalid, no feature has residual degrees of
            freedom, or "exact" F-statistics are asked of a fit whose standard
            errors are limma's approximation.
    """
    _check_fit(fit)
    p = _open_unit_interval(proportion, "proportion")
    limits = _coefficient_limits(stdev_coef_lim)
    covariate = _trend_covariate(trend, fit)
    exact = _f_method(f_statistic, fit)
    squeezed, df_total = _moderate(
        fit, covariate, robust, winsor_tail_p, prior_estimator
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        t = fit.coefficients / (
            fit.stdev_unscaled * np.sqrt(squeezed.posterior_variance)[:, np.newaxis]
        )
    p_value = 2.0 * _as_array(t_sf(np.abs(t), df_total[:, np.newaxis]))
    prior_df = np.broadcast_to(squeezed.prior_df, df_total.shape)
    infinite = np.isinf(prior_df)
    prior_variance = np.broadcast_to(squeezed.prior_variance, df_total.shape)
    median_prior = float(np.nanmedian(prior_variance))
    v0 = np.array(
        [
            _coefficient_prior_variance(
                t[:, j], fit.stdev_unscaled[:, j], df_total, p, limits, median_prior
            )
            for j in range(t.shape[1])
        ]
    )
    # Without a finite t-statistic (an all-zero contrast) there is nothing to
    # estimate v0 from; limma then takes the reciprocal of the first prior
    # variance, a value that the log-odds, all NaN, never use (found by probing).
    v0[~np.isfinite(v0)] = 1.0 / float(prior_variance[0])
    lods = _log_odds(t, fit.stdev_unscaled, df_total, infinite, v0, p)
    f_value, f_df1 = _f_statistics(t, fit, exact)
    f_df2 = prior_df + fit.df_residual
    f_p_value = _as_array(f_sf(f_value, float(f_df1), f_df2))
    return ModeratedFit(
        fit=fit,
        prior_variance=_frozen(squeezed.prior_variance),
        prior_df=_frozen(squeezed.prior_df),
        posterior_variance=_frozen(squeezed.posterior_variance),
        df_total=_frozen(df_total),
        t=_frozen(t),
        p_value=_frozen(p_value),
        lods=_frozen(lods),
        coefficient_prior_variance=_frozen(v0),
        f_statistic=_frozen(f_value),
        f_p_value=_frozen(f_p_value),
        f_df1=f_df1,
        proportion=p,
    )


def treat(
    fit: LinearModelFit,
    *,
    lfc: float | None = None,
    fc: float | None = None,
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments"] = "moments",
) -> TreatFit:
    """Moderated t-tests that coefficients exceed a threshold in absolute value.

    Tests H0: |beta| <= tau against |beta| > tau (McCarthy and Smyth 2009), matching
    limma's ``treat(fit, legacy=TRUE)``. With the moderated t-statistic t, its
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
        prior_estimator: See :func:`squeeze_variances`.

    Returns:
        The tests relative to the threshold.

    Raises:
        TypeError: If ``fit`` is not a LinearModelFit or an option has the wrong
            type.
        ValueError: If both ``lfc`` and ``fc`` are given, either is invalid, or no
            feature has residual degrees of freedom.
    """
    _check_fit(fit)
    threshold = _treat_threshold(lfc, fc)
    covariate = _trend_covariate(trend, fit)
    squeezed, df_total = _moderate(
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
        prior_df=_frozen(squeezed.prior_df),
        posterior_variance=_frozen(squeezed.posterior_variance),
        df_total=_frozen(df_total),
        t=_frozen(shrunk),
        p_value=_frozen(p_value),
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
    fitted, residual_df = _spline_trend(covariate, usable, e)
    if fitted is None:
        # A trend of one degree of freedom is a constant: limma then fits no trend,
        # which matters for an infinite df2, where the scale is the mean variance
        # (found by probing).
        constant = _fit_moments(x, d, None, usable)
        return FDistributionFit(np.full(x.size, constant.scale), constant.df2)
    residuals = e - fitted[usable]
    e_variance = float(np.sum(residuals**2)) / residual_df - mean_trigamma
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
                _interpolate(covariate[usable], np.log(part.scale), covariate[~usable])
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
    not_outlier = np.minimum(1.0, tail / ((_average_ranks(-ratio) - 0.5) / n))
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


def _average_ranks(values: FloatArray) -> FloatArray:
    """Ranks from 1 for the smallest value, with ties given their average rank."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], ordered.size]
    average = 0.5 * (starts + ends - 1) + 1.0
    ranks = np.empty(values.size)
    ranks[order] = np.repeat(average, ends - starts)
    return ranks


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


def _interpolate(x: FloatArray, y: FloatArray, at: FloatArray) -> FloatArray:
    """Linear interpolation of y over x at ``at``, constant beyond the range.

    Values of y at tied x are averaged first (R's ``approx`` with ``ties=mean``).
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    unique_x = xs[starts]
    mean_y = np.add.reduceat(ys, starts) / np.diff(np.r_[starts, xs.size])
    result: FloatArray = np.interp(at, unique_x, mean_y)
    return result


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
) -> SqueezedVariances:
    if x.size <= 2:
        return SqueezedVariances(x.copy(), x.copy(), 0.0)
    _check_zero_variances(x, usable, robust)
    prior_variance: float | FloatArray
    prior_df: float | FloatArray
    if robust:
        fitted = _fit_robust(x, d, covariate, usable, p_lower, p_upper)
        prior_variance, prior_df = fitted.scale, fitted.df2_shrunk
    else:
        moments = _fit_moments(x, d, covariate, usable)
        prior_variance, prior_df = moments.scale, moments.df2
    return SqueezedVariances(
        _posterior_variances(x, d, prior_variance, prior_df), prior_variance, prior_df
    )


def _posterior_variances(
    x: FloatArray,
    d: FloatArray,
    prior_variance: float | FloatArray,
    prior_df: float | FloatArray,
) -> FloatArray:
    s0 = np.broadcast_to(prior_variance, x.shape)
    d0 = np.broadcast_to(prior_df, x.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = np.where(d > 0.0, d * x, 0.0)
        posterior = (weighted + np.where(np.isinf(d0), 0.0, d0 * s0)) / (d + d0)
    # An infinite prior df, or zero residual df, leaves the prior variance.
    result: FloatArray = np.where(np.isinf(d0) | (d == 0.0), s0, posterior)
    return result


def _moderate(
    fit: LinearModelFit,
    covariate: FloatArray | None,
    robust: bool,
    winsor_tail_p: float | Sequence[float],
    prior_estimator: str,
) -> tuple[SqueezedVariances, FloatArray]:
    """Squeezed variances of a fit and the total degrees of freedom of its tests."""
    _check_flag(robust, "robust")
    _check_prior_estimator(prior_estimator)
    lower, upper = _tail_proportions(winsor_tail_p)
    variances = fit.sigma**2
    df = np.where(fit.df_residual < _ZERO_DF, 0.0, fit.df_residual)
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
    squeezed = _squeeze(variances, df, covariate, usable, robust, lower, upper)
    prior_df = np.broadcast_to(squeezed.prior_df, df.shape)
    df_total = np.minimum(df + prior_df, float(df.sum()))
    return squeezed, df_total


def _coefficient_prior_variance(
    t: FloatArray,
    stdev_unscaled: FloatArray,
    df: FloatArray,
    proportion: float,
    limits: tuple[float, float],
    median_prior: float,
) -> float:
    """limma's var.prior for one coefficient (Smyth 2004, section 6.3)."""
    bounds = (limits[0] ** 2 / median_prior, limits[1] ** 2 / median_prior)
    finite = np.isfinite(t)
    t, stdev_unscaled, df = t[finite], stdev_unscaled[finite], df[finite]
    n = t.size
    n_target = math.ceil(proportion / 2.0 * n)
    if n_target < 1:
        return math.nan
    size = np.abs(t)
    if df.min() < df.max():
        # t-statistics on fewer df are mapped to the largest df by matching tail
        # probabilities (found by probing).
        df_common = float(df.max())
        size = _as_array(t_isf(_as_array(t_sf(size, df)), df_common))
        df = np.full(n, df_common)
    share = max(n_target / n, proportion)
    top = np.argsort(-size, kind="stable")[:n_target]
    size, v1, df = size[top], stdev_unscaled[top] ** 2, df[top]
    p0 = 2.0 * _as_array(t_sf(size, df))
    target = ((np.arange(1, n_target + 1) - 0.5) / n - (1.0 - share) * p0) / share
    v0 = np.zeros(n_target)
    above = target > p0
    if above.any():
        quantile = _as_array(t_isf(target[above] / 2.0, df[above]))
        v0[above] = v1[above] * ((size[above] / quantile) ** 2 - 1.0)
    v0 = np.clip(v0, bounds[0], bounds[1])
    return float(np.mean(v0))


def _log_odds(
    t: FloatArray,
    stdev_unscaled: FloatArray,
    df_total: FloatArray,
    infinite_prior: npt.NDArray[np.bool_],
    v0: FloatArray,
    proportion: float,
) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        v = stdev_unscaled**2
        r = v / (v + v0)
        t2 = t * t
        df = df_total[:, np.newaxis]
        finite = (1.0 + df) / 2.0 * np.log1p(t2 * (1.0 - r) / (t2 * r + df))
        limit = t2 * (1.0 - r) / 2.0
        log_ratio = np.where(infinite_prior[:, np.newaxis], limit, finite)
        lods: FloatArray = (
            math.log(proportion / (1.0 - proportion)) + 0.5 * np.log(r) + log_ratio
        )
    return lods


def _f_statistics(
    t: FloatArray, fit: LinearModelFit, exact: bool
) -> tuple[FloatArray, int]:
    """Moderated F-statistics and their numerator degrees of freedom."""
    n_features, k = t.shape
    correlation = _unit_correlation(fit.cov_coefficients)
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


def _unit_correlation(covariance: FloatArray) -> FloatArray:
    """Correlation matrix, with a coefficient of zero variance uncorrelated.

    limma gives such a coefficient (an all-zero contrast) correlations of zero and
    keeps it in the rank of the F-test (found by probing).
    """
    scale = np.sqrt(np.diag(covariance))
    constant = scale == 0.0
    safe = np.where(constant, 1.0, scale)
    correlation: FloatArray = covariance / np.outer(safe, safe)
    correlation[constant, :] = 0.0
    correlation[:, constant] = 0.0
    correlation[constant, constant] = 1.0
    return correlation


# ---------------------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------------------


def _variance_vector(values: Any) -> FloatArray:
    x = _real_array(values, "variances")
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
    d = _real_array(df, "df")
    if d.size == 1 and d.ndim <= 1:
        d = np.full(n, float(d.reshape(-1)[0]))
    elif d.shape != (n,):
        raise ValueError(
            f"df must be a single value or have one value per variance ({n}), "
            f"got shape {d.shape}"
        )
    if not (np.isfinite(d).all() and (d >= 0.0).all()):
        raise ValueError("df must be finite and non-negative")
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
        skip_file_prefixes=(_PACKAGE_DIRECTORY,),
    )


def _covariate_vector(
    covariate: Any, usable: npt.NDArray[np.bool_]
) -> FloatArray | None:
    if covariate is None:
        return None
    z = _real_array(covariate, "covariate")
    if z.shape != usable.shape:
        raise ValueError(
            f"covariate must have one value per variance ({usable.size}), "
            f"got shape {z.shape}"
        )
    _check_covariate_nan(z, usable)
    return _finite_covariate(z)


def _check_covariate_nan(z: FloatArray, usable: npt.NDArray[np.bool_]) -> None:
    missing = np.isnan(z) & usable
    if missing.any():
        raise ValueError(
            f"the covariate is NaN for {int(missing.sum())} features with a usable "
            "variance; it may be NaN only where the variance is NaN or has zero "
            "degrees of freedom"
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
    values = _real_array(winsor_tail_p, "winsor_tail_p").reshape(-1)
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
    z = _real_array(trend, "trend")
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
        value = _real_scalar(lfc, "lfc")
        if not math.isfinite(value):
            raise ValueError(f"lfc must be finite, got {value}")
        return abs(value)
    ratio = 1.2 if fc is None else _real_scalar(fc, "fc")
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f"fc must be positive and finite, got {ratio}")
    return abs(math.log2(ratio))


def _coefficient_limits(limits: Any) -> tuple[float, float]:
    values = _real_array(limits, "stdev_coef_lim").reshape(-1)
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
    number = _real_scalar(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {number}")
    return number


def _check_fit(fit: Any) -> None:
    if not isinstance(fit, LinearModelFit):
        raise TypeError(f"fit must be a LinearModelFit, got {type(fit).__name__}")


def _check_flag(value: Any, name: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")


def _check_prior_estimator(prior_estimator: Any) -> None:
    if prior_estimator != "moments":
        raise ValueError(
            "prior_estimator must be 'moments', the method of moments (limma's "
            f"legacy=TRUE); got {prior_estimator!r}. limma's estimator for unequal "
            "degrees of freedom (fitFDistUnequalDF1) is not implemented"
        )


def _real_array(values: Any, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    return array.astype(np.float64, copy=False)


def _real_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


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


def _summary(values: float | FloatArray) -> str:
    if isinstance(values, float):
        return f"{values:.6g}"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "[non-finite]"
    return f"[{finite.min():.6g} to {finite.max():.6g}]"
