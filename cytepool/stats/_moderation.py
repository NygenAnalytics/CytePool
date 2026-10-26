"""Empirical Bayes moderation: posterior variances, prior variances of coefficients
and the log-odds of differential expression.

The parts of limma's moderated statistics (Smyth 2004) that the moderated t-tests,
the gene-set tests and the rotation tests share: the posterior variance of each
feature, which combines its residual variance with the prior variance, the prior
variance of a coefficient's true value among the features that are differentially
expressed, and the log-odds that a feature is.

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

from cytepool.stats.distributions import t_isf, t_sf

FloatArray = npt.NDArray[np.float64]

LIKELIHOOD_MIN_DF = 0.01
"""The fewest degrees of freedom of a variance that takes part in the likelihood
estimate of the prior (limma's fitFDistUnequalDF1, the estimator of Chen et al. 2025;
found by probing limma 3.62.1)."""


def posterior_variances(
    x: FloatArray,
    d: float | FloatArray,
    prior_variance: float | FloatArray,
    prior_df: float | FloatArray,
) -> FloatArray:
    """Posterior variances (d0 s0^2 + d s^2) / (d0 + d), as in Smyth (2004).

    The prior variance s0^2 where the prior df d0 is infinite, and where the residual
    df d is zero.
    """
    s0 = np.broadcast_to(prior_variance, x.shape)
    d0 = np.broadcast_to(prior_df, x.shape)
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
