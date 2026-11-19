"""Empirical Bayes moderation: posterior variances, prior variances of coefficients,
the log-odds of differential expression and moderated F-tests.

The parts of limma's moderated statistics (Smyth 2004) that the moderated t-tests,
the gene-set tests and the rotation tests share: the posterior variance of each
feature, which combines its residual variance with the prior variance, the prior
variance of a coefficient's true value among the features that are differentially
expressed, and the log-odds that a feature is. limma's moderated F-tests of several
coefficients or contrasts, which ``eBayes`` computes for all the columns of a fit and
the differential-expression functions for any set of them, and the F-statistic with
the same p-value on infinite denominator degrees of freedom, which dream's and
limma's F-tests report.

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. Nucleic Acids Research 53, gkaf018.
    doi:10.1093/nar/gkaf018
"""

import math

import numpy as np
import numpy.typing as npt

from cytepool._linear_algebra import correlation_matrix
from cytepool.stats.distributions import (
    _f_tail_quantile,
    f_isf,
    f_logsf,
    f_sf,
    t_isf,
    t_sf,
)

FloatArray = npt.NDArray[np.float64]

LIKELIHOOD_MIN_DF = 0.01
"""The fewest degrees of freedom of a variance that takes part in the likelihood
estimate of the prior (limma's fitFDistUnequalDF1, the estimator of Chen et al. 2025;
found by probing limma 3.62.1)."""

# Eigenvalues of the correlation matrix of the coefficients at or below this fraction
# of the largest do not count towards the rank of an F-test (found by probing).
_F_RANK_TOLERANCE = 1e-8
# Below this p-value, the standardized F-statistic comes from the logarithm of the
# tail probability, which stays exact where the p-value underflows to zero.
_LOG_TAIL_BELOW = 1e-300


def posterior_variances(
    x: FloatArray,
    d: float | FloatArray,
    prior_variance: float | FloatArray,
    df_prior: float | FloatArray,
) -> FloatArray:
    """Posterior variances (d0 s0^2 + d s^2) / (d0 + d), as in Smyth (2004).

    The prior variance s0^2 where the prior df d0 is infinite, and where the residual
    df d is zero.
    """
    s0 = np.broadcast_to(prior_variance, x.shape)
    d0 = np.broadcast_to(df_prior, x.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = np.where(d > 0.0, d * x, 0.0)
        posterior = (weighted + np.where(np.isinf(d0), 0.0, d0 * s0)) / (d + d0)
    # An infinite prior df, or zero residual df, leaves the prior variance.
    result: FloatArray = np.where(np.isinf(d0) | (d == 0.0), s0, posterior)
    return result


def coefficient_prior_variance(
    t: FloatArray,
    stdev_unscaled: FloatArray,
    df: FloatArray,
    proportion: float,
    limits: tuple[float, float],
    median_prior: float,
) -> float:
    """limma's var.prior for one coefficient (Smyth 2004, section 6.3)."""
    finite = np.isfinite(t)
    t, stdev_unscaled, df = t[finite], stdev_unscaled[finite], df[finite]
    n = t.size
    n_target = math.ceil(proportion / 2.0 * n)
    if n_target < 1:
        return math.nan
    # The median prior variance is positive here: a finite t-statistic needs a
    # positive posterior variance, which with at most two features, not moderated,
    # is the feature's own prior variance.
    bounds = np.square(np.asarray(limits, dtype=np.float64)) / median_prior
    size = np.abs(t)
    if df.min() < df.max():
        # t-statistics on fewer df are mapped to the largest df by matching tail
        # probabilities (found by probing).
        df_common = float(df.max())
        size = np.asarray(
            t_isf(np.asarray(t_sf(size, df), dtype=np.float64), df_common),
            dtype=np.float64,
        )
        df = np.full(n, df_common)
    share = max(n_target / n, proportion)
    top = np.argsort(-size, kind="stable")[:n_target]
    size, v1, df = size[top], stdev_unscaled[top] ** 2, df[top]
    p0 = 2.0 * np.asarray(t_sf(size, df), dtype=np.float64)
    target = ((np.arange(1, n_target + 1) - 0.5) / n - (1.0 - share) * p0) / share
    v0 = np.zeros(n_target)
    above = target > p0
    if above.any():
        quantile = np.asarray(t_isf(target[above] / 2.0, df[above]), dtype=np.float64)
        v0[above] = v1[above] * ((size[above] / quantile) ** 2 - 1.0)
    v0 = np.clip(v0, bounds[0], bounds[1])
    return float(np.mean(v0))


def log_odds(
    t: FloatArray,
    stdev_unscaled: FloatArray,
    df_total: FloatArray,
    infinite_prior: npt.NDArray[np.bool_],
    v0: FloatArray,
    proportion: float,
) -> FloatArray:
    """The log-odds of differential expression, limma's B-statistic (Smyth 2004)."""
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


def moderated_f_tests(
    t: FloatArray,
    covariance: FloatArray,
    covariance_per_feature: FloatArray | None,
    df_prior: float | FloatArray,
    df_residual: FloatArray,
) -> tuple[FloatArray, int, FloatArray, FloatArray]:
    """limma's moderated F-tests that the coefficients of each feature are all zero.

    F = t' R^+ t / r (Smyth 2004), for the moderated t-statistics t of the
    coefficients (or contrasts) in the columns of ``t``, the correlation matrix R of
    their estimates and its rank r, which counts the eigenvalues of the correlation of
    ``covariance`` above 1e-8 times the largest. R is each feature's own, from
    ``covariance_per_feature``, when given, which is the exact F-statistic under
    weights that differ between features (Smyth 2004, section 7); otherwise the
    correlation of ``covariance`` serves every feature, as in limma. A feature whose
    own R has a smaller rank, or whose t-statistics are not all finite, has a NaN
    F-statistic under the first rule. The denominator degrees of freedom are each
    feature's ``df_residual + df_prior``, without the cap of the moderated t-tests,
    and the p-value is the upper tail of F(r, df2), as limma 3.62.1 computes them in
    ``eBayes``'s ``F.p.value``, ``classifyTestsF`` and ``topTable`` of all of a fit's
    columns (found by probing). ``topTable`` of some of a fit's columns does too,
    except after a fit whose first feature has infinite prior degrees of freedom,
    when it refers every feature's F-statistic to infinite degrees of freedom.

    Returns:
        The F-statistics, r, the denominator degrees of freedom and the p-values.
    """
    n_features, k = t.shape
    correlation = correlation_matrix(covariance, unit_diagonal=True)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    keep = eigenvalues > _F_RANK_TOLERANCE * eigenvalues.max()
    rank = int(keep.sum())
    if covariance_per_feature is None:
        with np.errstate(invalid="ignore"):
            q = (t @ eigenvectors[:, keep]) / np.sqrt(eigenvalues[keep])
            f_value: FloatArray = np.sum(q * q, axis=1) / rank
    else:
        f_value = np.full(n_features, np.nan)
        complete = np.isfinite(t).all(axis=1)
        if complete.any():
            covariances = covariance_per_feature[complete]
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
    df2: FloatArray = df_prior + df_residual
    p_value = np.asarray(f_sf(f_value, float(rank), df2), dtype=np.float64)
    return f_value, rank, df2, p_value


def standardized_f_statistics(
    f_value: FloatArray, df1: int, df2: FloatArray, p_value: FloatArray
) -> FloatArray:
    """The F-statistics on ``df1`` and infinite degrees of freedom with the same
    p-values, so that F-tests on different denominator degrees of freedom compare.

    variancePartition's ``F.std``: the upper quantile of F(df1, infinity) at the
    p-value. Below p-values of 1e-300 it comes from the logarithm of the tail
    probability of ``f_value`` on (df1, df2), which stays exact where the p-value
    underflows to zero: an F-statistic on infinite denominator degrees of freedom is
    its own standardized F-statistic, however large.
    """
    with np.errstate(invalid="ignore"):
        standardized = np.asarray(f_isf(p_value, float(df1), np.inf), dtype=np.float64)
        tiny = p_value < _LOG_TAIL_BELOW
    if tiny.any():
        df = np.broadcast_to(df2, np.shape(f_value))
        log_tails = np.asarray(
            f_logsf(f_value[tiny], float(df1), df[tiny]), dtype=np.float64
        )
        standardized[tiny] = [_upper_quantile(value, float(df1)) for value in log_tails]
    return standardized


def _upper_quantile(log_tail: float, df1: float) -> float:
    """The x with log P(F > x) = ``log_tail`` for F on (df1, infinity) degrees of
    freedom: infinite for a tail of zero."""
    if log_tail == -math.inf:
        return math.inf
    return float(_f_tail_quantile(log_tail, True, df1, math.inf))
