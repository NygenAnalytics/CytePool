"""edgeR's quasi-likelihood F-tests: glmQLFit and glmQLFTest as edgeR 4 runs them.

The quasi-likelihood model of Lund, Nettleton, McCarthy and Smyth (2012) gives each gene
g a negative binomial model for its counts with a second, gene-wise dispersion:

    var(y_gi) = sigma_g^2 (mu_gi + phi mu_gi^2),

with phi the negative binomial (NB) dispersion, shared by the genes, and sigma_g^2
the quasi-dispersion. The quasi-dispersions are estimated from the deviances of the
NB fits, moderated by empirical Bayes as limma moderates residual variances, and
genes are tested with F-tests whose denominator is the posterior quasi-dispersion.
:func:`fit_quasi_likelihood` and :func:`quasi_likelihood_f_test` follow edgeR 4.4's
``glmQLFit`` and ``glmQLFTest`` with their defaults (``legacy = FALSE``), the method of
Chen, Chen, Lun, Baldoni and Smyth (2025), as documented and, where the documentation
is silent, as found by running edgeR 4.4.0 as a black box ("found by probing").

The steps:

1. Abundance. :func:`average_log_cpm` is edgeR's ``aveLogCPM``: the log2 of the mean
   count per million of a gene, fitted as one group of samples by a NB model after a
   prior count of 2, scaled by library size, is added to every count (edgeR's
   ``addPriorCount``). ``glmQLFit`` computes it once, at the given NB dispersion, or
   at 0.05 when it estimates the dispersion (found by probing); it ranks the genes
   and is the covariate of both trends below. The tests report it at the NB
   dispersion.
2. NB dispersion. Unless given, it is the Cox-Reid common dispersion
   (:func:`common_dispersion`, edgeR's ``estimateGLMCommonDisp``) of the most highly
   expressed genes: the ``ceiling(p G)`` of the G genes with the largest average log
   CPM, with ``p = chooseLowessSpan(G sqrt(n - k), small.n = 20, min.span = 0.02)``
   for n samples and k coefficients, as edgeR's help page says (the count of genes
   is a ceiling, found by probing).
3. Fits. Every gene gets the NB fit of
   :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model` at
   the working dispersion phi / A, where A is the average quasi-dispersion below.
4. Bias-adjusted deviances. The unit deviance d_i of an observation is approximated
   by a scaled chi-squared variable s_i chi^2(k_i) whose mean and variance are those
   of the unit deviance (Chen et al. 2025): s_i = V_i / (2 E_i) and k_i = 2 E_i^2 /
   V_i, with E_i and V_i from :func:`negative_binomial_unit_deviance_moments`. The
   adjusted unit deviance is d_i / s_i and its degrees of freedom are k_i (1 - h_i),
   with h_i the leverage of the observation. An observation of leverage 1, which the
   fit reproduces, has a residual deviance and degrees of freedom of zero, as in
   exact arithmetic, not the rounding of its fitted value. A gene's adjusted
   residual deviance and degrees of freedom are their sums over its samples, and its
   quasi-dispersion is their ratio, on degrees of freedom that are fractional and
   differ between genes.
   The moments are those of the divided counts (Baldoni et al. 2024): with
   quasi-dispersion A, var(y) = A (mu + (phi / A) mu^2), and y / A follows a NB
   distribution with mean mu / A and dispersion phi. E_i and V_i are taken at the
   mean mu_i / A and the dispersion phi, and d_i at the working dispersion (found by
   probing).
5. Average quasi-dispersion (found by probing). A starts at 1 and is updated twice:
   each time the genes are fitted at phi / A and their quasi-dispersions formed as in
   step 4; the fourth roots of the quasi-dispersions of the genes with residual
   degrees of freedom are smoothed against the abundance by R's ``lowess`` with a
   span of 0.5 (three robustness iterations), and A becomes the fourth power of the
   0.9 quantile of the smoothed values, or 1 if that is smaller. The fits of step 3
   use the final A, which is edgeR's ``average.ql.dispersion``: its help page says
   that it "scales the NB dispersion".
6. Moderation. The quasi-dispersions are squeezed by limma's ``squeezeVar`` with its
   estimator for unequal degrees of freedom (``fitFDistUnequalDF1``, ``legacy =
   FALSE`` whatever the degrees of freedom; found by probing), through
   :func:`~cytepool.stats.empirical_bayes.squeeze_variances` with
   ``prior_estimator="likelihood"``: towards a trend in the abundance by default, and
   robustly when asked. ``winsor.tail.p`` plays no part, as in edgeR (found by
   probing).
7. Tests. :func:`quasi_likelihood_f_test` fits the null model at the working
   dispersion, without the tested coefficients or with the tested contrasts set to
   zero, and divides the likelihood ratio statistic by its degrees of freedom r (the
   rank of the contrasts) and by the posterior quasi-dispersion. The F-statistic is
   referred to the F distribution on r and min(d0 + d_g, sum of n - k over the genes)
   degrees of freedom, with d0 the prior and d_g the adjusted residual degrees of
   freedom (found by probing). Log fold changes are the coefficients shrunk with a
   prior count of 0.125 (Phipson 2013), divided by log 2, and adjusted p-values are
   Benjamini and Hochberg's (1995), as edgeR's ``topTags`` gives them.

Prior weights (found by probing). edgeR uses them in the fits as prior weights, and
in the Cox-Reid likelihood as the number of independent counts each count averages
(edgeR's ``adjustedProfileLik``). In the adjusted deviances it treats an observation
of weight w as w technical replicates: its moments are taken at the mean w mu / A, its
unit deviance at the dispersion w phi / A, and its leverage is that of the weight mu /
(1 + w phi mu / A) rather than of the weight of the fit; its adjusted unit deviance is
per unit of weight, so that the gene's adjusted deviance is the sum of the unit values
times the weights, and its degrees of freedom the plain sum. This module follows edgeR
in each. The weights are therefore not relative: multiplying all of them by a constant
changes the quasi-dispersions and the F-statistics, in both.

Deliberate differences from edgeR 4.4.0:

- The mean and variance of the unit deviance are exact, to about 1e-14. edgeR
  approximates them (found by probing): the scale implied by its adjusted unit
  deviances is off by up to about 1.2e-3 at dispersions up to 0.2 and 1.3e-2 at a
  dispersion of 1, and the degrees of freedom by up to about 2e-4 and 4e-3. The
  adjusted deviances, quasi-dispersions and F-statistics inherit differences of up
  to about 3e-3.
- The Cox-Reid dispersion is the maximum of the adjusted profile likelihood, located
  to about 1e-7 relative; edgeR's search stops within its tolerance of 1e-5. Where
  edgeR approximates the deviance, below a dispersion of 1e-4, its likelihood is
  noisy (found by probing), so that for counts close to Poisson the two estimates
  differ by more than that, both being near zero.
- The fits converge to the maximum, where edgeR's Levenberg fits stop short of it by
  up to about 3e-4 in the coefficients (see
  :mod:`cytepool.stats.negative_binomial_models`), so that the log fold changes of
  designs that are not one-way layouts differ by up to about that. Fitted values
  that tend to zero, as in a group without counts, are zero in both for one-way
  layouts; otherwise they stop near 1e-8 in edgeR's fits and far lower here, and
  both give such observations the leverages of the limit. Either way they carry
  almost no degrees of freedom. A gene with no others, such as one counted in only
  one condition of a paired design, has the ratio of their almost-zero deviances
  and degrees of freedom as its quasi-dispersion, which therefore differs (0.053 in
  edgeR and 0.021 here, for such a gene in eight paired samples). It enters the
  average quasi-dispersion in both.
- The robust prior tests only the genes with residual degrees of freedom for
  outliers, as :func:`~cytepool.stats.empirical_bayes.fit_f_distribution_likelihood`
  does; limma counts a gene without them, such as one without counts, as a zero
  variance, which then lowers the outlier probabilities of the others.
- With offsets per observation, edgeR's ``aveLogCPM`` gives the first gene a value
  that does not depend on its offsets (found by probing: a copy of the gene in
  another row gets the value computed here), which then enters the ranking, the
  trends and the log CPM of that gene.

Not implemented yet: edgeR's ``legacy = TRUE`` methods (the deviances of
``glmQLFit`` in edgeR 3, with the degrees of freedom adjusted for groups without
counts, and the Poisson bound of ``glmQLFTest``). This module follows edgeR's method
for a matrix of counts. edgeR's method for a DGEList agrees with it, except when the
DGEList holds a trended dispersion from ``estimateDisp``: it then takes one
dispersion from that trend in place of the Cox-Reid estimate (found by probing).

References:
    Lund SP, Nettleton D, McCarthy DJ, Smyth GK (2012). Detecting differential
    expression in RNA-sequence data using quasi-likelihood with shrunken dispersion
    estimates. Statistical Applications in Genetics and Molecular Biology 11(5),
    article 8. doi:10.1515/1544-6115.1826

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. Nucleic Acids Research 53, gkaf018.
    doi:10.1093/nar/gkaf018

    Baldoni PL, Chen L, Smyth GK (2024). Faster and more accurate assessment of
    differential transcript expression with Gibbs sampling and edgeR v4. NAR Genomics
    and Bioinformatics 6, lqae151. doi:10.1093/nargab/lqae151

    McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation. Nucleic Acids
    Research 40, 4288-4297. doi:10.1093/nar/gks042

    Cox DR, Reid N (1987). Parameter orthogonality and approximate conditional
    inference. Journal of the Royal Statistical Society B 49, 1-18.
    doi:10.1111/j.2517-6161.1987.tb01422.x

    Loader C (2000). Fast and accurate computation of binomial probabilities. Technical
    note, Lucent Technologies, Murray Hill; the version of 5 February 2002 is available
    at https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    Robinson MD, Smyth GK (2008). Small-sample estimation of negative binomial
    dispersion, with applications to SAGE data. Biostatistics 9, 321-332.
    doi:10.1093/biostatistics/kxm030

    Brent RP (1973). Algorithms for Minimization without Derivatives. Prentice-Hall,
    Englewood Cliffs.

    Neumaier A (1974). Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
    Summen. Zeitschrift fuer Angewandte Mathematik und Mechanik 54, 39-51.
    doi:10.1002/zamm.19740540106

    Giraud L, Langou J, Rozloznik M (2005). The loss of orthogonality in the
    Gram-Schmidt orthogonalization process. Computers & Mathematics with Applications
    50, 1069-1075. doi:10.1016/j.camwa.2005.08.009

    Berrut JP, Trefethen LN (2004). Barycentric Lagrange interpolation. SIAM Review 46,
    501-517. doi:10.1137/S0036144502417715

    Phipson B (2013). Empirical Bayes modelling of expression profiles and their
    associations. PhD thesis, University of Melbourne.
    https://hdl.handle.net/11343/38162

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import math
import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from cytepool._counts import as_counts, dispersion_matrix, offset_matrix, weight_matrix
from cytepool._jit import jit, thread_limit
from cytepool._linear_algebra import orthogonalise
from cytepool._masked_arrays import reject_masked, unmasked_array
from cytepool._messages import value_range
from cytepool._options import (
    checked_flag,
    non_negative_number,
    positive_integer,
    real_number,
)
from cytepool.design._contrast_weights import contrast_matrix
from cytepool.design._design_matrices import design_matrix, refuse_random_effects
from cytepool.stats._moderation import LIKELIHOOD_MIN_DF
from cytepool.stats._negative_binomial import (
    SERIES_LIMIT,
    unit_deviance,
    unit_deviances,
)
from cytepool.stats._optimize import brent_minimum
from cytepool.stats._scalar_special_functions import stirling_error
from cytepool.stats.distributions import f_sf
from cytepool.stats.empirical_bayes import squeeze_variances
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.negative_binomial_models import (
    NegativeBinomialFit,
    _read_only,
    fit_negative_binomial_model,
)
from cytepool.stats.smoothing import lowess

__all__ = [
    "QuasiLikelihoodFit",
    "QuasiLikelihoodTest",
    "UnitDevianceMoments",
    "average_log_cpm",
    "common_dispersion",
    "fit_quasi_likelihood",
    "negative_binomial_unit_deviance_moments",
    "quasi_likelihood_f_test",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_MILLION = math.log(1e6)
_LOG_TWO = math.log(2.0)

# Moments of the unit deviance. A sum over the counts 0, 1, 2, ... runs while it needs
# at most _DIRECT_TERMS terms; otherwise the counts below _SPLIT are summed and the rest
# of the sum is the integral of its smooth continuation over [_SPLIT, inf), with
# Gregory's end correction from the forward differences at _SPLIT, computed by
# Gauss-Legendre panels on the log of the count. Terms below _NEGLIGIBLE times the
# sums are left out.
_DIRECT_TERMS = 600.0
_SPLIT = 64
_NEGLIGIBLE = 1e-22
# Gregory's coefficients G_1, G_2, ...: sum_{k >= 0} f(a + k) = integral_a^inf f +
# sum_j G_{j+1} (Delta^j f)(a) for f smooth on the scale of 1 (Gregory's formula).
_GREGORY = np.array(
    [
        1.0 / 2.0,
        -1.0 / 12.0,
        1.0 / 24.0,
        -19.0 / 720.0,
        3.0 / 160.0,
        -863.0 / 60480.0,
        275.0 / 24192.0,
        -33953.0 / 3628800.0,
        8183.0 / 1036800.0,
        -3250433.0 / 479001600.0,
        4671.0 / 788480.0,
        -13695779093.0 / 2615348736000.0,
        2224234463.0 / 475517952000.0,
    ]
)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)
# A dispersion shared by every observation gets its moments, as functions of the log
# of the mean, from Chebyshev interpolation on panels of _PANEL in the log mean, with
# _PANEL_NODES nodes each, for means from _TABLE_LEAST; smaller means are summed.
_PANEL = 1.0
_PANEL_NODES = 24
_TABLE_LEAST = 1e-12
# The largest mean and dispersion of the moments: beyond about 1e150 for the mean,
# products in the unit deviance overflow.
_LARGEST_MEAN = 1e100
_LARGEST_DISPERSION = 1e4
_CHEBYSHEV = np.cos(
    (2.0 * np.arange(_PANEL_NODES) + 1.0) * np.pi / (2.0 * _PANEL_NODES)
)
_CHEBYSHEV_WEIGHTS = (-1.0) ** np.arange(_PANEL_NODES) * np.sin(
    (2.0 * np.arange(_PANEL_NODES) + 1.0) * np.pi / (2.0 * _PANEL_NODES)
)
# A column of the weighted design whose part orthogonal to the columns before it has a
# norm of at most this times its own carries no information: a leverage is spread over
# the columns that do (the rule of the negative binomial fits).
_ALIASING = 1e-11
# Fitted values, or weights of the leverages, below this fraction of a gene's largest
# tend to zero (the fits leave such fitted values below about 1e-15): they carry no
# information on the dispersion, and their leverages are those of the limit.
_VANISHING = 1e-12
_EPSILON = 2.220446049250313e-16
# edgeR's settings (found by probing or from its help pages): the dispersion of the
# average log CPM, its prior count, the search interval and row-sum filter of the
# Cox-Reid dispersion, the most genes it uses, and the span rule of the top genes.
_ABUNDANCE_DISPERSION = 0.05
_ABUNDANCE_PRIOR_COUNT = 2.0
_DISPERSION_INTERVAL = (0.0, 4.0)
_MIN_ROW_SUM = 5.0
_SUBSET = 10000
_TOP_SMALL_N = 20
_TOP_MIN_SPAN = 0.02
# The tolerance of the Cox-Reid search, absolute, as R's optimize takes it; its
# stopping rule adds sqrt(eps) times the dispersion.
_DISPERSION_TOLERANCE = 1e-10
# Passes of the average quasi-dispersion, the power of the quasi-dispersions it
# smooths, and the quantile of the smoothed values it takes (found by probing).
_AVERAGE_PASSES = 2
_AVERAGE_POWER = 0.25
_AVERAGE_SPAN = 0.5
_AVERAGE_QUANTILE = 0.9


class UnitDevianceMoments(NamedTuple):
    """Mean and variance of the negative binomial unit deviance.

    Attributes:
        mean: E[d(Y, mu, phi)] for Y ~ NB(mu, phi).
        variance: var[d(Y, mu, phi)].
    """

    mean: Any
    variance: Any


def negative_binomial_unit_deviance_moments(
    means: npt.ArrayLike, dispersion: npt.ArrayLike
) -> UnitDevianceMoments:
    """Mean and variance of the unit deviance of a negative binomial count.

    For Y ~ NB(mu, phi), with var(Y) = mu + phi mu^2 (Poisson at phi = 0), the unit
    deviance d(Y, mu, phi) of
    :func:`~cytepool.stats.negative_binomial_models.negative_binomial_unit_deviance`
    has mean E = sum_y P(Y = y) d(y, mu, phi) and variance V = sum_y P(Y = y)
    (d(y, mu, phi) - E)^2. edgeR 4 approximates each unit deviance by the scaled
    chi-squared variable with these two moments, s chi^2(k) with s = V / (2 E) and k
    = 2 E^2 / V (Chen et al. 2025): k is near 1 for large means, larger for means
    near 0.5, and tends to 0 with the mean.

    The probabilities are formed in Loader's (2000) deviance form, P(Y = y) = sqrt(r
    / (2 pi y (y + r))) exp(-d / 2 + delta(y + r) - delta(r) - delta(y)) with r = 1 /
    phi and delta Stirling's error, which has no cancellation for large counts or
    large r. Counts are summed one by one while that takes at most 600 terms;
    otherwise the counts below 64 are summed and the rest of the sum is the integral
    of its continuation, with Gregory's end correction from the forward differences
    at 64, by Gauss-Legendre panels on the log of the count. The moments agree to
    about 1e-14 relative with sums at 30 digits for means up to 300, with the gamma
    limit of large means and with the large-mean expansion of the Poisson moments.
    edgeR's approximations of them differ by up to about 1e-2 (see the module
    description).

    Args:
        means: Means, finite and non-negative; at most 1e100.
        dispersion: Dispersions, finite and non-negative; at most 1e4.

    Returns:
        The means and variances, as float64 arrays of the broadcast shape, or floats
        when both arguments are scalars. A mean of zero gives zero for both.

    Raises:
        TypeError: If an argument is not real-valued, or is a numpy masked array.
        ValueError: If an argument is negative or not finite, a mean exceeds 1e100
            or a dispersion 1e4, or the shapes do not broadcast.
    """
    mu = _non_negative_values(means, "means")
    phi = _non_negative_values(dispersion, "dispersion")
    _check_moment_range(mu, "means", _LARGEST_MEAN)
    _check_moment_range(phi, "dispersion", _LARGEST_DISPERSION)
    try:
        shape = np.broadcast_shapes(mu.shape, phi.shape)
    except ValueError:
        raise ValueError(
            f"means {mu.shape} and dispersion {phi.shape} do not broadcast to one shape"
        ) from None
    flat_mu = np.ascontiguousarray(np.broadcast_to(mu, shape)).ravel()
    flat_phi = np.ascontiguousarray(np.broadcast_to(phi, shape)).ravel()
    mean, variance = _moments_of(flat_mu, flat_phi)
    if len(shape) == 0:
        return UnitDevianceMoments(float(mean[0]), float(variance[0]))
    return UnitDevianceMoments(mean.reshape(shape), variance.reshape(shape))


def _check_moment_range(values: FloatArray, name: str, largest: float) -> None:
    if values.size and float(values.max()) > largest:
        raise ValueError(
            f"{name} must be at most {largest:g} for the moments of the unit "
            f"deviance, beyond which they are not checked; got {float(values.max()):g}"
        )


@jit
def _log_probability(y: float, phi: float, deviance: float) -> float:
    """log P(Y = y) for y > 0, Y ~ NB(mu, phi), in Loader's deviance form."""
    if phi == 0.0:
        return -0.5 * deviance - 0.5 * (_LOG_2PI + math.log(y)) - stirling_error(y)
    r = 1.0 / phi
    return (
        -0.5 * deviance
        - 0.5 * (_LOG_2PI + math.log(y) + math.log1p(phi * y))
        + (stirling_error(y + r) - stirling_error(r))
        - stirling_error(y)
    )


@jit
def _probability_and_deviance(y: float, mu: float, phi: float) -> tuple[float, float]:
    """P(Y = y) and the unit deviance d(y, mu, phi), for y >= 0 and mu > 0."""
    deviance = unit_deviance(y, mu, phi)
    if y == 0.0:
        return math.exp(-0.5 * deviance), deviance
    return math.exp(_log_probability(y, phi, deviance)), deviance


@jit
def _direct_sums(mu: float, phi: float) -> tuple[float, float, float]:
    """sum_y p, sum_y p d and sum_y p d^2 over the counts, term by term.

    Past the mean, the terms fall by at most the factor rho = p(y + 1) / p(y), which
    for r < 1 rises towards its limit q = mu / (mu + r), so the rest of the sum is
    below the last term times 1 / (1 - max(rho, q)), up to the slow growth of d.
    """
    q = 0.0 if phi == 0.0 else mu / (mu + 1.0 / phi)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    y = 0.0
    while True:
        p, d = _probability_and_deviance(y, mu, phi)
        s0 += p
        s1 += p * d
        s2 += p * d * d
        if y > mu:
            size = p * (1.0 + d * d)
            if size < _NEGLIGIBLE * (s0 + s2):
                if phi == 0.0:
                    ratio = mu / (y + 1.0)
                else:
                    ratio = max((y + 1.0 / phi) / (y + 1.0) * q, q)
                if ratio < 1.0 and size < _NEGLIGIBLE * (s0 + s2) * (1.0 - ratio):
                    break
        y += 1.0
    return s0, s1, s2


@jit
def _deviance_from_offset(y: float, s: float, mu: float, phi: float) -> float:
    """d(y, mu, phi) at the count y = mu (1 + s), from the relative offset s.

    Within 1/8 of the mean it is the series of ``unit_deviance`` in s itself, so
    that it is the deviance at mu (1 + s) whatever the rounding of y; farther out it
    is ``unit_deviance`` of the count.
    """
    if abs(s) >= SERIES_LIMIT:
        far: float = unit_deviance(y, mu, phi)
        return far
    product = phi * mu
    share = product / (1.0 + product) if product <= 1.0 else 1.0 / (1.0 + 1.0 / product)
    t = s * share
    total = 0.5
    term = 1.0
    power = 1.0
    n = 1.0
    while n < 64.0:
        power *= -s
        term = power - t * term
        updated = total + term / ((n + 1.0) * (n + 2.0))
        if updated == total:
            break
        total = updated
        n += 1.0
    return 2.0 * (s * s / (1.0 / mu + phi)) * total


@jit
def _integrand(v: float, mu: float, phi: float) -> tuple[float, float]:
    """y p(y) at y = mu exp(v), the integrand in v = log(y / mu), and d(y).

    Near the mean the integrand varies on the scale of the spread of the counts,
    sqrt(mu) for nearly Poisson counts, while the count, rounded to a float, lies off
    the node by up to about 1e-16 mu: a share 1e-16 sqrt(mu) of that scale, which
    would move the deviance. The deviance is therefore taken from the offset s =
    exp(v) - 1 of the node; the rest of the integrand is smooth on the scale of the
    count.
    """
    s = math.expm1(v)
    y = mu * math.exp(v)
    deviance = _deviance_from_offset(y, s, mu, phi)
    return y * math.exp(_log_probability(y, phi, deviance)), deviance


@jit
def _split_sums(mu: float, phi: float) -> tuple[float, float, float]:
    """The three sums: counts below _SPLIT one by one, the rest as an integral.

    The integral over [_SPLIT, inf) in v = log(y / mu) is taken on Gauss-Legendre
    panels no wider than the spread of the log count, from the mean outwards until
    the integrand is negligible on both sides; Gregory's correction adds the
    difference between the sum over the counts from _SPLIT and that integral.
    """
    sd_log = math.sqrt(math.log1p(1.0 / mu + phi))
    width = min(1.0, 2.0 * sd_log)
    start = math.log(_SPLIT / mu)
    centre = max(0.0, start)
    peak, d = _integrand(centre, mu, phi)
    peak *= 1.0 + d * d
    # The bounds of the integral: outwards from the centre until negligible.
    upper = centre
    while True:
        upper += width
        value, d = _integrand(upper, mu, phi)
        size = value * (1.0 + d * d)
        peak = max(peak, size)
        if size < _NEGLIGIBLE * peak:
            break
    lower = centre
    while lower > start:
        lower = max(lower - width, start)
        value, d = _integrand(lower, mu, phi)
        size = value * (1.0 + d * d)
        peak = max(peak, size)
        if size < _NEGLIGIBLE * peak:
            break
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    n_panels = max(1, int(math.ceil((upper - lower) / width)))
    step = (upper - lower) / n_panels
    for panel in range(n_panels):
        a = lower + panel * step
        for i in range(_GAUSS_NODES.size):
            value, d = _integrand(a + 0.5 * step * (_GAUSS_NODES[i] + 1.0), mu, phi)
            weight = 0.5 * step * _GAUSS_WEIGHTS[i] * value
            s0 += weight
            s1 += weight * d
            s2 += weight * d * d
    if lower > start:
        # The counts below the integral hold none of the mass.
        return s0, s1, s2
    n_terms = _GREGORY.size
    f0 = np.empty(n_terms)
    f1 = np.empty(n_terms)
    f2 = np.empty(n_terms)
    for k in range(n_terms):
        p, d = _probability_and_deviance(float(_SPLIT + k), mu, phi)
        f0[k] = p
        f1[k] = p * d
        f2[k] = p * d * d
    # Gregory's correction from the forward differences at _SPLIT.
    for j in range(n_terms):
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        binomial = 1.0
        for k in range(j + 1):
            sign = -1.0 if (j - k) % 2 == 1 else 1.0
            c0 += sign * binomial * f0[k]
            c1 += sign * binomial * f1[k]
            c2 += sign * binomial * f2[k]
            binomial = binomial * (j - k) / (k + 1.0)
        s0 += _GREGORY[j] * c0
        s1 += _GREGORY[j] * c1
        s2 += _GREGORY[j] * c2
    for y in range(_SPLIT):
        p, d = _probability_and_deviance(float(y), mu, phi)
        s0 += p
        s1 += p * d
        s2 += p * d * d
    return s0, s1, s2


@jit
def _deviance_moments(mu: float, phi: float) -> tuple[float, float]:
    """Mean and variance of d(Y, mu, phi) for Y ~ NB(mu, phi); zero for mu = 0."""
    if mu <= 0.0:
        return 0.0, 0.0
    sd = math.sqrt(mu + phi * mu * mu)
    # Terms a sum needs: up to about ten standard deviations past the mean, and as
    # many again as the upper tail, which falls by q per count, needs to fall by
    # about e^-50.
    terms = mu + 10.0 * sd + 50.0
    if phi > 0.0:
        terms += 50.0 / math.log1p(1.0 / (phi * mu))
    if terms <= _DIRECT_TERMS:
        s0, s1, s2 = _direct_sums(mu, phi)
    else:
        s0, s1, s2 = _split_sums(mu, phi)
    mean = s1 / s0
    variance = max(s2 / s0 - mean * mean, 0.0)
    return mean, variance


@jit
def _moments_of(mu: FloatArray, phi: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.empty(mu.size)
    variance = np.empty(mu.size)
    for i in range(mu.size):
        mean[i], variance[i] = _deviance_moments(mu[i], phi[i])
    return mean, variance


@jit
def _scale_and_df(mean: float, variance: float) -> tuple[float, float]:
    """The scale V / (2 E) and degrees of freedom 2 E^2 / V of the chi-squared match.

    A zero mean (a fitted value of zero) gives an infinite scale and no degrees of
    freedom: its unit deviance is zero.
    """
    if mean <= 0.0 or variance <= 0.0:
        return math.inf, 0.0
    # 2 E (E / V) rather than 2 E^2 / V, whose square underflows for tiny means.
    return variance / (2.0 * mean), 2.0 * mean * (mean / variance)


@jit
def _scales_one_dispersion(mu: FloatArray, phi: float) -> tuple[FloatArray, FloatArray]:
    """Scales and degrees of freedom of the chi-squared match at the means ``mu``, all
    with dispersion ``phi``.

    The logs of both, as functions of t = log(mu), are interpolated on panels of width
    _PANEL by the barycentric formula on _PANEL_NODES Chebyshev points (Berrut and
    Trefethen 2004); a panel is computed when a mean falls in it. Means below
    _TABLE_LEAST are summed. Both are smooth in t, and the interpolation agrees with the
    sums to about 3e-14 relative for dispersions from 0 to 1e4.
    """
    n = mu.size
    scale = np.empty(n)
    df = np.empty(n)
    lowest = math.inf
    highest = -math.inf
    for i in range(n):
        if mu[i] >= _TABLE_LEAST:
            t = math.log(mu[i])
            lowest = min(lowest, t)
            highest = max(highest, t)
    if lowest <= highest:
        origin = math.floor(lowest / _PANEL) * _PANEL
        n_panels = int(math.floor((highest - origin) / _PANEL)) + 1
    else:
        origin = 0.0
        n_panels = 0
    log_scale = np.empty((n_panels, _PANEL_NODES))
    log_df = np.empty((n_panels, _PANEL_NODES))
    filled = np.zeros(n_panels, dtype=np.bool_)
    for i in range(n):
        if mu[i] <= 0.0:
            scale[i] = math.inf
            df[i] = 0.0
            continue
        if mu[i] < _TABLE_LEAST:
            mean, variance = _deviance_moments(mu[i], phi)
            scale[i], df[i] = _scale_and_df(mean, variance)
            continue
        t = math.log(mu[i])
        panel = min(int(math.floor((t - origin) / _PANEL)), n_panels - 1)
        a = origin + panel * _PANEL
        if not filled[panel]:
            for k in range(_PANEL_NODES):
                node = math.exp(a + 0.5 * _PANEL * (_CHEBYSHEV[k] + 1.0))
                mean, variance = _deviance_moments(node, phi)
                s, f = _scale_and_df(mean, variance)
                log_scale[panel, k] = math.log(s)
                log_df[panel, k] = math.log(f)
            filled[panel] = True
        x = 2.0 * (t - a) / _PANEL - 1.0
        numerator_scale = 0.0
        numerator_df = 0.0
        denominator = 0.0
        exact = -1
        for k in range(_PANEL_NODES):
            difference = x - _CHEBYSHEV[k]
            if difference == 0.0:
                exact = k
                break
            weight = _CHEBYSHEV_WEIGHTS[k] / difference
            numerator_scale += weight * log_scale[panel, k]
            numerator_df += weight * log_df[panel, k]
            denominator += weight
        if exact >= 0:
            scale[i] = math.exp(log_scale[panel, exact])
            df[i] = math.exp(log_df[panel, exact])
        else:
            scale[i] = math.exp(numerator_scale / denominator)
            df[i] = math.exp(numerator_df / denominator)
    return scale, df


@jit
def _scales_each(mu: FloatArray, phi: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Scales and degrees of freedom with a dispersion for each mean."""
    scale = np.empty(mu.size)
    df = np.empty(mu.size)
    for i in range(mu.size):
        mean, variance = _deviance_moments(mu[i], phi[i])
        scale[i], df[i] = _scale_and_df(mean, variance)
    return scale, df


def _chi_squared_match(
    means: FloatArray, dispersion: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Scales and degrees of freedom at ``means`` (any shape) and ``dispersion`` (a
    single value, or one per mean)."""
    if means.size and float(means.max()) > _LARGEST_MEAN:
        raise ValueError(
            "the fitted values times the prior weights reach "
            f"{float(means.max()):g}, beyond {_LARGEST_MEAN:g}, the largest mean for "
            "the moments of the unit deviance"
        )
    flat = np.ascontiguousarray(means, dtype=np.float64).ravel()
    if dispersion.size == 1:
        scale, df = _scales_one_dispersion(flat, float(dispersion.reshape(-1)[0]))
    else:
        phi = np.ascontiguousarray(np.broadcast_to(dispersion, means.shape)).ravel()
        scale, df = _scales_each(flat, phi)
    return scale.reshape(means.shape), df.reshape(means.shape)


def _non_negative_values(values: Any, name: str) -> FloatArray:
    array = unmasked_array(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    result = np.array(array, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(result) & (result >= 0.0))
    if bad.any():
        raise ValueError(
            f"{name} must be finite and non-negative; "
            f"{int(np.count_nonzero(bad))} values are not"
        )
    return result


# ---------------------------------------------------------------------------------------
# Leverages and adjusted deviances
# ---------------------------------------------------------------------------------------


@jit
def _orthonormal_columns(
    design: FloatArray, root_weights: FloatArray, basis: FloatArray
) -> tuple[int, float]:
    """Orthonormal columns spanning W^(1/2) X, by Gram-Schmidt orthogonalised twice.

    Twice is enough for orthogonality to rounding (Giraud, Langou and Rozloznik 2005).
    ``basis`` gets the kept columns in its first rows. A column whose part orthogonal to
    the columns before it has a norm of at most _ALIASING times its own carries no
    information and is left out.

    Returns:
        The rank, and the log of the determinant of X' W X over the kept columns,
        the sum of the logs of the squared norms of the orthogonal parts.
    """
    n_samples, n_coefficients = design.shape
    rank = 0
    log_determinant = 0.0
    for j in range(n_coefficients):
        total = 0.0
        for i in range(n_samples):
            value = root_weights[i] * design[i, j]
            basis[rank, i] = value
            total += value * value
        if total == 0.0:
            continue
        orthogonalise(basis[rank], basis, rank, n_samples)
        remaining = 0.0
        for i in range(n_samples):
            remaining += basis[rank, i] * basis[rank, i]
        if remaining <= _ALIASING * _ALIASING * total:
            continue
        norm = math.sqrt(remaining)
        for i in range(n_samples):
            basis[rank, i] /= norm
        log_determinant += math.log(remaining)
        rank += 1
    return rank, log_determinant


@jit
def _leverages(design: FloatArray, weights: FloatArray) -> FloatArray:
    """Leverages of every gene (row of ``weights``, genes x samples): the diagonal of
    the projection on the column space of W^(1/2) X, which is zero for an observation
    of weight zero and sums to the rank of the weighted design.

    Observations with weights below _VANISHING times the gene's largest, such as
    those whose fitted values tend to zero, get the limit of their leverages as these
    weights tend to zero together. Rounding would lose it where the design spans
    their direction only as a difference of columns, as for a group without counts
    that is the baseline. The columns are therefore orthogonalised, by Gram-Schmidt
    twice, on the other observations first and, where they are aliased there, on these
    (an inner product taken in that order): the other observations get their
    leverages from the columns of the first kind, these from the second.
    """
    n_genes, n_samples = weights.shape
    n_coefficients = design.shape[1]
    out = np.zeros((n_genes, n_samples))
    basis = np.empty((n_coefficients, n_samples))
    second = np.zeros(n_coefficients, dtype=np.bool_)
    root = np.empty(n_samples)
    small = np.empty(n_samples, dtype=np.bool_)
    for g in range(n_genes):
        largest = 0.0
        for i in range(n_samples):
            largest = max(largest, weights[g, i])
        for i in range(n_samples):
            root[i] = math.sqrt(weights[g, i])
            small[i] = weights[g, i] < _VANISHING * largest
        rank = 0
        for j in range(n_coefficients):
            total_large = 0.0
            total_small = 0.0
            for i in range(n_samples):
                value = root[i] * design[i, j]
                basis[rank, i] = value
                if small[i]:
                    total_small += value * value
                else:
                    total_large += value * value
            for _ in range(2):
                # The columns of the first kind, then those of the second.
                for kind in (False, True):
                    for k in range(rank):
                        if second[k] != kind:
                            continue
                        product = 0.0
                        for i in range(n_samples):
                            if small[i] == kind:
                                product += basis[k, i] * basis[rank, i]
                        for i in range(n_samples):
                            basis[rank, i] -= product * basis[k, i]
            remaining_large = 0.0
            remaining_small = 0.0
            for i in range(n_samples):
                if small[i]:
                    remaining_small += basis[rank, i] * basis[rank, i]
                else:
                    remaining_large += basis[rank, i] * basis[rank, i]
            if remaining_large > _ALIASING * _ALIASING * total_large:
                second[rank] = False
                norm = math.sqrt(remaining_large)
            elif remaining_small > _ALIASING * _ALIASING * total_small:
                second[rank] = True
                norm = math.sqrt(remaining_small)
                for i in range(n_samples):
                    if not small[i]:
                        basis[rank, i] = 0.0
            else:
                continue
            for i in range(n_samples):
                basis[rank, i] /= norm
            rank += 1
        for i in range(n_samples):
            total = 0.0
            for k in range(rank):
                if second[k] == small[i]:
                    total += basis[k, i] * basis[k, i]
            out[g, i] = total
    return out


@dataclass(frozen=True)
class _AdjustedDeviances:
    """The bias-adjusted deviances of one pass, genes x samples where per observation."""

    unit_deviance: FloatArray
    unit_df: FloatArray
    leverage: FloatArray
    deviance: FloatArray
    df_residual: FloatArray


def _adjusted_deviances(
    counts: FloatArray,
    fit: NegativeBinomialFit,
    dispersion: FloatArray,
    average: float,
    weights: FloatArray,
) -> _AdjustedDeviances:
    """edgeR's bias-adjusted unit deviances and degrees of freedom of a fit.

    ``counts`` and the arrays of ``fit`` are samples x genes; ``dispersion`` is the
    NB dispersion phi (before division by ``average``, the average quasi-dispersion
    A), as one value, one per gene or one per observation, and ``weights`` the prior
    weights, broadcast to samples x genes. The fit's dispersion is phi / A.
    """
    mu = np.asarray(fit.fitted_values)
    phi = np.broadcast_to(dispersion, mu.shape)
    working = phi / average
    means = weights * mu / average
    scale, df = _chi_squared_match(means, dispersion)
    deviance = _unit_deviances_at(counts, mu, weights * working)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_deviance = np.where(df > 0.0, deviance / scale, 0.0)
        information = mu / (1.0 + weights * working * mu)
    leverage = _leverages(
        np.ascontiguousarray(fit.design), np.ascontiguousarray(information.T)
    ).T
    # An observation of leverage 1 is reproduced by the fit: it has no residual, so
    # its deviance and degrees of freedom are zero. Such leverages come out within a
    # few roundings of 1; one within _VANISHING of 1 would need other weights below
    # _VANISHING times its own, which get the leverages of the limit. Computed from
    # the fitted value, the deviance would hold only the rounding of the fit, which
    # is all the deviance of a gene whose other fitted values tend to zero.
    exact = leverage > 1.0 - _VANISHING
    unit_deviance = np.where(exact, 0.0, unit_deviance)
    unit_df = np.where(exact, 0.0, df * (1.0 - leverage))
    # The unit deviances are per unit of prior weight: a gene's adjusted deviance
    # weights them, and its degrees of freedom do not (found by probing).
    return _AdjustedDeviances(
        unit_deviance=unit_deviance,
        unit_df=unit_df,
        leverage=leverage,
        deviance=(weights * unit_deviance).sum(axis=0),
        df_residual=unit_df.sum(axis=0),
    )


def _unit_deviances_at(
    counts: FloatArray, means: FloatArray, dispersion: FloatArray
) -> FloatArray:
    shape = np.broadcast_shapes(counts.shape, means.shape, dispersion.shape)
    flat = [
        np.ascontiguousarray(np.broadcast_to(a, shape), dtype=np.float64).ravel()
        for a in (counts, means, dispersion)
    ]
    values: FloatArray = unit_deviances(*flat).reshape(shape)
    return values


def _check_moderation(quasi: FloatArray, df: FloatArray) -> None:
    """Refuse quasi-dispersions that limma's prior for unequal degrees of freedom
    cannot be estimated from (``squeeze_variances`` moderates no more than two)."""
    if quasi.size <= 2:
        return
    usable = np.isfinite(quasi) & (df >= LIKELIHOOD_MIN_DF)
    positive = int(np.count_nonzero(quasi[usable] > 0.0))
    if positive < 2:
        raise ValueError(
            "the quasi-dispersions cannot be moderated: limma's prior for unequal "
            "degrees of freedom needs at least two genes with a quasi-dispersion above "
            f"zero on at least {LIKELIHOOD_MIN_DF:g} adjusted residual degrees of "
            f"freedom, and {positive} have one"
        )


def _average_quasi_dispersion(
    deviance: FloatArray, df: FloatArray, abundance: FloatArray
) -> float:
    """edgeR's average quasi-dispersion (found by probing).

    The fourth roots of the quasi-dispersions of the genes with residual degrees of
    freedom are smoothed against their average log CPM by R's ``lowess`` with a span
    of 0.5 and its three robustness iterations; the average is the fourth power of
    the 0.9 quantile of the smoothed values (R's default quantile, type 7), and at
    least 1.
    """
    usable = df > 0.0
    if not usable.any():
        raise ValueError(
            "no gene has adjusted residual degrees of freedom, so no quasi-dispersion "
            "can be estimated: every gene's fitted values are zero, or the design fits "
            "every sample exactly"
        )
    roots = (deviance[usable] / df[usable]) ** _AVERAGE_POWER
    trend = lowess(abundance[usable], roots, span=_AVERAGE_SPAN)
    average: float = float(np.quantile(trend, _AVERAGE_QUANTILE)) ** (
        1.0 / _AVERAGE_POWER
    )
    return max(1.0, average)


# ---------------------------------------------------------------------------------------
# Average log counts per million
# ---------------------------------------------------------------------------------------


def average_log_cpm(
    counts: npt.ArrayLike,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    offsets: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    prior_count: float = 2.0,
    dispersion: float | npt.ArrayLike = 0.05,
) -> FloatArray:
    """Average log2 counts per million of each gene, as edgeR's aveLogCPM.

    Each gene's samples are fitted as one group by the negative binomial model of
    :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model`,
    after ``prior_count`` times its library size over the mean library size is added
    to each count and twice that to its library size (edgeR's ``addPriorCount``).
    The result is log2 of the fitted count per million: log2(1e6 exp(b)) with b the
    fitted log mean per unit of library size. It is similar to the log2 of the mean
    of the counts per million, with larger libraries given more weight, and equal to
    it for a large dispersion (edgeR's help page). The prior count keeps genes
    without counts finite.

    Args:
        counts: ``(n_samples, n_features)`` counts, finite and non-negative.
        library_sizes: Library size of each sample, positive; defaults to the total
            counts.
        norm_factors: Normalization factor of each sample, positive; defaults to
            ones. The library sizes are multiplied by them.
        offsets: Log library sizes, in place of library sizes and normalization
            factors: one value, one per sample, or one per observation.
        weights: Prior weights, positive: one value, one per sample, or one per
            observation.
        prior_count: Average count added to each observation, at least 0.
        dispersion: Negative binomial dispersion of the fits: one value, one per gene,
            or one per observation; 0.05 by default, as in edgeR.

    Returns:
        ``(n_features,)`` average log2 counts per million.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: As
            :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model`,
            or if ``prior_count`` is negative.
    """
    y = as_counts(counts)
    prior = non_negative_number(prior_count, "prior_count")
    fit = fit_negative_binomial_model(
        y,
        np.ones((y.shape[0], 1)),
        dispersion=dispersion,
        library_sizes=library_sizes,
        norm_factors=norm_factors,
        offsets=offsets,
        weights=weights,
        prior_count=prior,
    )
    result: FloatArray = (np.asarray(fit.shrunk_coefficients)[:, 0] + _LOG_MILLION) / (
        _LOG_TWO
    )
    return result


# ---------------------------------------------------------------------------------------
# Common dispersion (Cox-Reid adjusted profile likelihood)
# ---------------------------------------------------------------------------------------


def common_dispersion(
    counts: npt.ArrayLike,
    design: Any,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    offsets: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    interval: Sequence[float] = _DISPERSION_INTERVAL,
    min_row_sum: float = _MIN_ROW_SUM,
    subset: int = _SUBSET,
    coefficient_names: Sequence[str] | None = None,
    max_iterations: int = 200,
    n_threads: int | None = None,
) -> float:
    """Common negative binomial dispersion of all genes by Cox-Reid adjusted profile
    likelihood, as edgeR's estimateGLMCommonDisp.

    The dispersion phi maximizes the sum over genes of the adjusted profile
    log-likelihoods (McCarthy, Chen and Smyth 2012; Cox and Reid 1987)

        APL_g(phi) = l_g(phi, beta_g(phi)) - 1/2 log det(X' W_g X),

    with beta_g(phi) the maximum likelihood fit at phi of
    :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model` and
    W_g the expected information of the fitted values, w mu / (1 + phi mu) with prior
    weights w. With prior weights each count is the average of w independent negative
    binomial counts, as for edgeR's ``adjustedProfileLik``: its log-likelihood is
    that of w y under NB(w mu, phi / w). The adjustment, which a REML criterion
    parallels, removes most of the downward bias of the maximum likelihood estimate
    (Robinson and Smyth 2008). Observations whose fitted values tend to zero, below
    1e-12 of the gene's largest, carry no information and are left out of the
    determinant, whose remaining factor does not depend on phi.

    As edgeR does (its help pages): genes with fewer than ``min_row_sum`` counts in all
    are left out; when more than ``subset`` genes remain, a systematic subset of them,
    evenly spaced in average log CPM, is used (``systematicSubset``: every s-th gene in
    increasing average log CPM, s = floor(G / subset), from gene floor(s / 2) + 1; found
    by probing), which keeps all G genes while G < 2 ``subset`` and from ``subset`` to
    about 1.5 ``subset`` of them otherwise; and the maximum is searched over
    ``interval``. The search, Brent's (1973) method as R's ``optimize``, locates the
    maximum to about 1e-7 relative, as closely as the rounding of the likelihood allows;
    edgeR's stops within its tolerance of 1e-5. A maximum at an end of the interval is
    returned with a warning.

    Args:
        counts: ``(n_samples, n_features)`` counts, finite and non-negative.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank, an
            object with ``matrix`` and ``columns`` such as a ``Design``, or a table.
        library_sizes: Library size of each sample, positive; defaults to the total
            counts.
        norm_factors: Normalization factor of each sample, positive; defaults to ones.
        offsets: Log effective library sizes, in place of library sizes and
            normalization factors: one value, one per sample, or one per observation.
        weights: Prior weights, positive: one value, one per sample, or one per
            observation.
        interval: The smallest and largest dispersion searched, 0 <= low < high.
        min_row_sum: Genes with fewer counts in all are left out; at least 0.
        subset: The number of genes beyond which a systematic subset of them is
            used, as above; at least 1.
        coefficient_names: Names of the design columns.
        max_iterations: The most Newton steps of each fit.
        n_threads: The most threads of the fits; ``None`` uses numba's setting.

    Returns:
        The common dispersion.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: As
            :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model`,
            if an option is out of range, or if no gene has ``min_row_sum`` counts.
    """
    with thread_limit(n_threads):
        y = as_counts(counts)
        n_samples, n_features = y.shape
        x, names = design_matrix(design, coefficient_names)
        refuse_random_effects(design, "common_dispersion")
        low, high = _search_interval(interval)
        least = non_negative_number(min_row_sum, "min_row_sum")
        reject_masked(subset, "subset")
        most = positive_integer(subset, "subset", expected="a positive integer")
        reject_masked(max_iterations, "max_iterations")
        limit = positive_integer(
            max_iterations, "max_iterations", expected="a positive integer"
        )
        log_sizes = offset_matrix(offsets, library_sizes, norm_factors, y)
        w = weight_matrix(weights, n_samples, n_features)
        return _common_dispersion(
            y, x, names, log_sizes, w, low, high, least, most, limit
        )


def _common_dispersion(
    y: FloatArray,
    x: FloatArray,
    names: tuple[str, ...],
    log_sizes: FloatArray,
    w: FloatArray,
    low: float,
    high: float,
    least: float,
    most: int,
    limit: int,
) -> float:
    """The Cox-Reid common dispersion; arrays as the negative binomial fits hold them:
    counts samples x genes, offsets and weights genes x samples (1 or G rows)."""
    n_samples, n_features = y.shape
    keep = np.flatnonzero(y.sum(axis=0) >= least)
    if keep.size == 0:
        raise ValueError(
            f"no gene has at least {least:g} counts in all (min_row_sum), so no "
            "dispersion can be estimated"
        )
    if keep.size > most:
        abundance = average_log_cpm(
            y[:, keep],
            offsets=_by_samples(log_sizes, keep),
            weights=_by_samples(w, keep),
        )
        keep = keep[_systematic_subset(most, abundance)]
    counts = np.ascontiguousarray(y[:, keep])
    offsets = _by_samples(log_sizes, keep)
    weights = _by_samples(w, keep)
    counts_by_gene = np.ascontiguousarray(counts.T)
    weights_by_gene = _full_by_gene(w, keep, n_samples)

    def minus_likelihood(phi: float) -> float:
        fit = fit_negative_binomial_model(
            counts,
            x,
            dispersion=phi,
            offsets=offsets,
            weights=weights,
            prior_count=0.0,
            coefficient_names=names,
            max_iterations=limit,
        )
        means = np.ascontiguousarray(np.asarray(fit.fitted_values).T)
        value: float = _adjusted_profile_likelihood(
            counts_by_gene, means, weights_by_gene, phi, x
        )
        return -value

    phi = brent_minimum(minus_likelihood, low, high, _DISPERSION_TOLERANCE)
    reach = 2.0 * (math.sqrt(_EPSILON) * abs(phi) + _DISPERSION_TOLERANCE)
    if phi - low <= reach or high - phi <= reach:
        end = "lower" if phi - low <= reach else "upper"
        warnings.warn(
            f"the common dispersion reached the {end} end of its search interval "
            f"[{low:g}, {high:g}]: {phi:.6g}",
            RuntimeWarning,
            stacklevel=3,
        )
    return float(phi)


@jit
def _adjusted_profile_likelihood(
    counts: FloatArray,
    means: FloatArray,
    weights: FloatArray,
    phi: float,
    design: FloatArray,
) -> float:
    """Sum over genes (rows) of the Cox-Reid adjusted profile log-likelihood.

    The log-likelihood of a count y with weight w is that of w y under NB(w mu,
    phi / w), in Loader's deviance form, whose unit deviance is w d(y, mu, phi); the
    adjustment is -1/2 log det(X' W X) over the columns that the weights inform, W =
    w mu / (1 + phi mu). The sum is compensated (Neumaier 1974).
    """
    n_genes, n_samples = counts.shape
    n_coefficients = design.shape[1]
    basis = np.empty((n_coefficients, n_samples))
    root = np.empty(n_samples)
    total = 0.0
    error = 0.0
    for g in range(n_genes):
        largest = 0.0
        for i in range(n_samples):
            largest = max(largest, means[g, i])
        floor = _VANISHING * largest
        log_likelihood = 0.0
        for i in range(n_samples):
            y = counts[g, i]
            mu = means[g, i]
            w = weights[g, i]
            root[i] = math.sqrt(w * mu / (1.0 + phi * mu)) if mu > floor else 0.0
            if mu <= 0.0:
                continue
            deviance = w * unit_deviance(y, mu, phi)
            if y == 0.0:
                log_likelihood -= 0.5 * deviance
                continue
            z = w * y
            term = (
                -0.5 * deviance
                - 0.5 * (_LOG_2PI + math.log(z) + math.log1p(phi * y))
                - stirling_error(z)
            )
            if phi > 0.0:
                size = w / phi
                term += stirling_error(z + size) - stirling_error(size)
            log_likelihood += term
        _, log_determinant = _orthonormal_columns(design, root, basis)
        value = log_likelihood - 0.5 * log_determinant
        updated = total + value
        if abs(total) >= abs(value):
            error += (total - updated) + value
        else:
            error += (value - updated) + total
        total = updated
    return total + error


def _systematic_subset(n: int, order_by: FloatArray) -> npt.NDArray[np.int64]:
    """edgeR's ``systematicSubset(n, order.by)``: every s-th index in increasing
    ``order_by``, s = floor(N / n), from the (floor(s / 2) + 1)-th (found by probing)."""
    total = order_by.size
    step = total // n
    if step <= 1:
        return np.arange(total, dtype=np.int64)
    order = np.argsort(order_by, kind="stable")
    positions = np.arange(step // 2, total, step)
    return np.sort(order[positions]).astype(np.int64)


def _top_genes(abundance: FloatArray, proportion: float) -> npt.NDArray[np.int64]:
    """The ceiling(p G) genes with the largest average log CPM, in index order."""
    n_top = min(abundance.size, math.ceil(proportion * abundance.size))
    order = np.argsort(-abundance, kind="stable")
    return np.sort(order[:n_top]).astype(np.int64)


def _top_proportion(n_features: int, df_residual: int) -> float:
    """edgeR's default share of top genes: limma's chooseLowessSpan(G sqrt(n - k),
    small.n = 20, min.span = 0.02), whose number of points is not a whole number."""
    points = n_features * math.sqrt(df_residual)
    if points <= _TOP_SMALL_N:
        return 1.0
    share: float = _TOP_MIN_SPAN + (1.0 - _TOP_MIN_SPAN) * (_TOP_SMALL_N / points) ** (
        1.0 / 3.0
    )
    return share


def _by_samples(matrix: FloatArray, genes: npt.NDArray[np.int64]) -> Any:
    """A genes x samples matrix of offsets or weights (1 or G rows, 1 or n columns)
    for the given genes, as an argument of the fits: one value, one per sample, or
    samples x genes."""
    if matrix.shape[0] == 1:
        return float(matrix[0, 0]) if matrix.shape[1] == 1 else matrix[0].copy()
    return np.ascontiguousarray(matrix[genes].T)


def _full_by_gene(
    matrix: FloatArray, genes: npt.NDArray[np.int64], n: int
) -> FloatArray:
    """A genes x samples matrix of offsets or weights for the given genes, with every
    row and column."""
    rows = matrix if matrix.shape[0] == 1 else matrix[genes]
    return np.ascontiguousarray(
        np.broadcast_to(rows, (genes.size, n)), dtype=np.float64
    )


# ---------------------------------------------------------------------------------------
# Quasi-likelihood fits
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class QuasiLikelihoodFit:
    """Negative binomial fits of every gene with moderated quasi-dispersions.

    Rows of the per-gene arrays are genes, in the order of the columns of the counts;
    arrays with a sample axis are samples x genes. The arrays are read-only. The
    names in parentheses are edgeR's.

    Attributes:
        fit: The negative binomial fits at the working dispersion ``dispersion /
            average_quasi_dispersion``: coefficients, shrunk coefficients
            (``coefficients``), fitted values, deviances (``deviance``) and the
            offsets, dispersions and weights of the fits.
        dispersion: The negative binomial dispersion (``dispersion``), as given, or
            estimated from the top genes: a float, one value per gene, or one per
            observation.
        top_proportion: The share of genes the dispersion was estimated from
            (``top.proportion``); None when it was given.
        average_quasi_dispersion: The average quasi-dispersion A
            (``average.ql.dispersion``), at least 1.
        unit_deviance_adjusted: ``(n_samples, n_features)`` adjusted unit deviances
            (``unit.deviance.adj``).
        unit_df_adjusted: ``(n_samples, n_features)`` their degrees of freedom
            (``unit.df.adj``).
        leverage: ``(n_samples, n_features)`` leverages of the fits (``leverage``);
            those of fitted values that tend to zero are the limits.
        deviance_adjusted: ``(n_features,)`` adjusted residual deviances, the sums of
            the adjusted unit deviances.
        df_residual_adjusted: ``(n_features,)`` adjusted residual degrees of freedom
            (``df.residual.adj``), fractional; zero for a gene without counts.
        quasi_dispersion: ``(n_features,)`` quasi-dispersions, the ratio of the two;
            NaN where the degrees of freedom are zero.
        prior_df: The prior degrees of freedom (``df.prior``): a float, or one value
            per gene from a robust fit that finds outliers.
        prior_quasi_dispersion: The prior quasi-dispersion (``s2.prior``): one value
            per gene with a trend, a float otherwise.
        posterior_quasi_dispersion: ``(n_features,)`` posterior quasi-dispersions
            (``s2.post``).
        covariate: ``(n_features,)`` covariate of the trend of the prior: by
            default the average log CPM that also ranks the genes, at the given NB
            dispersion or at 0.05 when the dispersion is estimated (edgeR's
            abundance); None without a trend.
        average_log_cpm: ``(n_features,)`` average log2 counts per million at the
            negative binomial dispersion, the ``logCPM`` of the tests.
        robust: Whether the prior was estimated robustly.
        counts: ``(n_samples, n_features)`` the counts fitted (``counts``).
        max_iterations: The most Newton steps of each fit, as given.
    """

    fit: NegativeBinomialFit
    dispersion: float | FloatArray
    top_proportion: float | None
    average_quasi_dispersion: float
    unit_deviance_adjusted: FloatArray
    unit_df_adjusted: FloatArray
    leverage: FloatArray
    deviance_adjusted: FloatArray
    df_residual_adjusted: FloatArray
    quasi_dispersion: FloatArray
    prior_df: float | FloatArray
    prior_quasi_dispersion: float | FloatArray
    posterior_quasi_dispersion: FloatArray
    covariate: FloatArray | None
    average_log_cpm: FloatArray
    robust: bool
    counts: FloatArray
    max_iterations: int

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Names of the coefficients, the design's columns."""
        return self.fit.coefficient_names

    def __repr__(self) -> str:
        n_samples, n_features = self.fit.fitted_values.shape
        return (
            f"QuasiLikelihoodFit(n_features={n_features}, n_samples={n_samples}, "
            f"coefficient_names={self.coefficient_names!r}, "
            f"dispersion={value_range(self.dispersion)}, "
            f"average_quasi_dispersion={self.average_quasi_dispersion:.6g}, "
            f"prior_df={value_range(self.prior_df)}, robust={self.robust})"
        )


def fit_quasi_likelihood(
    counts: npt.ArrayLike,
    design: Any,
    *,
    dispersion: float | npt.ArrayLike | None = None,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    offsets: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    trend: bool | npt.ArrayLike = True,
    robust: bool = False,
    prior_count: float = 0.125,
    top_proportion: float | None = None,
    coefficient_names: Sequence[str] | None = None,
    max_iterations: int = 200,
    n_threads: int | None = None,
) -> QuasiLikelihoodFit:
    """Fit negative binomial models with empirical Bayes quasi-dispersions, as edgeR's
    glmQLFit.

    edgeR 4's default method (``legacy = FALSE``), described in the module
    description: the negative binomial dispersion phi is given or estimated as the
    Cox-Reid common dispersion of the most highly expressed genes; every gene is
    fitted at the working dispersion phi / A, with A the average quasi-dispersion;
    its quasi-dispersion is its bias-adjusted deviance over its adjusted residual
    degrees of freedom; and the quasi-dispersions are squeezed towards a prior that
    follows the average log CPM, by limma's estimator for unequal degrees of
    freedom. edgeR's help page says that a dispersion that is given should come from
    a global model, common or trended, not from each gene on its own: the
    quasi-dispersions carry the variation between genes.

    Args:
        counts: ``(n_samples, n_features)`` counts, finite and non-negative, integer or
            not.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank with
            fewer columns than samples, an object with ``matrix`` and ``columns``
            such as a ``Design``, or a table. A ``Design`` with random effects is
            refused.
        dispersion: Negative binomial dispersion, finite, from 0 to 1e4: one value,
            one per gene ``(n_features,)`` or one per observation ``(n_samples,
            n_features)``. By default it is estimated as the common dispersion
            (:func:`common_dispersion`) of the top ``top_proportion`` of the genes by
            average log CPM.
        library_sizes: Library size of each sample, positive; defaults to the total
            counts.
        norm_factors: Normalization factor of each sample, such as TMM factors from
            :func:`~cytepool.normalization.tmm_factors`, positive; defaults to ones.
        offsets: Log effective library sizes, in place of library sizes and
            normalization factors: one value, one per sample, or one per
            observation.
        weights: Prior weights, positive and finite: one value, one per sample, or one
            per observation; see the module description for how edgeR uses them.
        trend: Whether the prior quasi-dispersion follows the average log CPM
            (edgeR's ``abundance.trend``), or ``(n_features,)`` covariate values for
            it to follow instead (``covariate.trend``).
        robust: Whether to estimate the prior robustly, so that genes whose
            quasi-dispersions are outliers count less (Phipson et al. 2016), as
            edgeR's User's Guide recommends.
        prior_count: Average count added to each observation for the shrunk
            coefficients, whose differences are the log fold changes of the tests;
            at least 0.
        top_proportion: Share of the genes, by average log CPM, that the dispersion
            is estimated from, in (0, 1]. By default ``chooseLowessSpan(G sqrt(n -
            k), small.n = 20, min.span = 0.02)`` for G genes, n samples and k
            coefficients, which falls as genes and residual degrees of freedom are
            added, as in edgeR. Only for an estimated dispersion.
        coefficient_names: Names of the design columns.
        max_iterations: The most Newton steps of each fit, at least 1.
        n_threads: The most threads of the fits; ``None`` uses numba's setting. The
            result does not depend on it.

    Returns:
        The fits, the dispersions and the quasi-dispersions.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As
            :func:`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model`,
            if the design leaves no residual degrees of freedom, an option is out of
            range, ``top_proportion`` is given with a dispersion, no gene has
            adjusted residual degrees of freedom, the fitted values times the prior
            weights exceed 1e100, or fewer than two genes have a positive
            quasi-dispersion on at least 0.01 adjusted residual degrees of freedom,
            so that limma's prior cannot be estimated.
    """
    with thread_limit(n_threads):
        y = as_counts(counts)
        n_samples, n_features = y.shape
        x, names = design_matrix(design, coefficient_names)
        refuse_random_effects(design, "fit_quasi_likelihood")
        if x.shape[0] != n_samples:
            raise ValueError(
                f"counts have {n_samples} rows (samples) but the design has "
                f"{x.shape[0]}; both need one row per sample"
            )
        df_residual = n_samples - x.shape[1]
        if df_residual < 1:
            raise ValueError(
                f"the design has {x.shape[1]} columns for {n_samples} samples, so it "
                "leaves no residual degrees of freedom to estimate quasi-dispersions"
            )
        reject_masked(robust, "robust")
        checked_flag(robust, "robust", expected="a bool")
        prior = non_negative_number(prior_count, "prior_count")
        reject_masked(max_iterations, "max_iterations")
        limit = positive_integer(
            max_iterations, "max_iterations", expected="a positive integer"
        )
        log_sizes = offset_matrix(offsets, library_sizes, norm_factors, y)
        w = weight_matrix(weights, n_samples, n_features)
        covariate = _trend_values(trend, n_features)
        if dispersion is not None and top_proportion is not None:
            raise ValueError(
                "top_proportion chooses the genes the dispersion is estimated from; "
                "give it or the dispersion, not both"
            )
        share = None if top_proportion is None else _proportion(top_proportion)
        given = {"offsets": _argument(log_sizes), "weights": _argument(w)}
        if dispersion is None:
            # The genes are ranked, and the trends follow, by the average log CPM at
            # aveLogCPM's default dispersion (found by probing).
            abundance = average_log_cpm(
                y,
                **given,
                prior_count=_ABUNDANCE_PRIOR_COUNT,
                dispersion=_ABUNDANCE_DISPERSION,
            )
            if share is None:
                share = _top_proportion(n_features, df_residual)
            top = _top_genes(abundance, share)
            low, high = _DISPERSION_INTERVAL
            value = _common_dispersion(
                np.ascontiguousarray(y[:, top]),
                x,
                names,
                _rows(log_sizes, top),
                _rows(w, top),
                low,
                high,
                _MIN_ROW_SUM,
                _SUBSET,
                limit,
            )
            phi = np.array([[value]])
            reported: float | FloatArray = value
            # The logCPM of the tests is at the dispersion of the fit.
            log_cpm = average_log_cpm(
                y,
                **given,
                prior_count=_ABUNDANCE_PRIOR_COUNT,
                dispersion=value,
            )
        else:
            phi = dispersion_matrix(dispersion, n_samples, n_features)
            _check_moment_range(phi, "dispersion", _LARGEST_DISPERSION)
            reported = _as_given(_argument(phi))
            # With a dispersion given, the trends follow the average log CPM at that
            # dispersion, which is also the logCPM of the tests (found by probing).
            abundance = average_log_cpm(
                y,
                **given,
                prior_count=_ABUNDANCE_PRIOR_COUNT,
                dispersion=_argument(phi),
            )
            log_cpm = abundance
        weights_by_sample = np.broadcast_to(w.T, y.shape)
        average = 1.0
        for _ in range(_AVERAGE_PASSES):
            fit = fit_negative_binomial_model(
                y,
                x,
                dispersion=_argument(phi / average),
                **given,
                prior_count=0.0,
                coefficient_names=names,
                max_iterations=limit,
            )
            adjusted = _adjusted_deviances(y, fit, phi.T, average, weights_by_sample)
            average = _average_quasi_dispersion(
                adjusted.deviance, adjusted.df_residual, abundance
            )
        fit = fit_negative_binomial_model(
            y,
            x,
            dispersion=_argument(phi / average),
            **given,
            prior_count=prior,
            coefficient_names=names,
            max_iterations=limit,
        )
        adjusted = _adjusted_deviances(y, fit, phi.T, average, weights_by_sample)
        df = adjusted.df_residual
        with np.errstate(divide="ignore", invalid="ignore"):
            quasi = np.where(df > 0.0, adjusted.deviance / df, np.nan)
        _check_moderation(quasi, df)
        trend_values = abundance if covariate is True else covariate
        squeezed = squeeze_variances(
            quasi,
            df,
            covariate=None if trend_values is False else trend_values,
            robust=robust,
            prior_estimator="likelihood",
        )
    return QuasiLikelihoodFit(
        fit=fit,
        dispersion=reported,
        top_proportion=None if dispersion is not None else share,
        average_quasi_dispersion=float(average),
        unit_deviance_adjusted=_read_only(adjusted.unit_deviance),
        unit_df_adjusted=_read_only(adjusted.unit_df),
        leverage=_read_only(adjusted.leverage),
        deviance_adjusted=_read_only(adjusted.deviance),
        df_residual_adjusted=_read_only(df),
        quasi_dispersion=_read_only(quasi),
        prior_df=_frozen(squeezed.prior_df),
        prior_quasi_dispersion=_frozen(squeezed.prior_variance),
        posterior_quasi_dispersion=_read_only(
            np.asarray(squeezed.posterior_variance, dtype=np.float64)
        ),
        covariate=None
        if trend_values is False
        else _read_only(np.asarray(trend_values, dtype=np.float64)),
        average_log_cpm=_read_only(log_cpm),
        robust=bool(robust),
        # A copy: the tests refit the null model from these counts.
        counts=_read_only(np.array(y)),
        max_iterations=limit,
    )


# ---------------------------------------------------------------------------------------
# Quasi-likelihood F-tests
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class QuasiLikelihoodTest:
    """Quasi-likelihood F-tests of coefficients or contrasts, one per gene.

    The columns of edgeR's ``topTags`` table are ``log_fold_change`` (``logFC``),
    ``average_log_cpm`` (``logCPM``), ``f_statistic`` (``F``), ``p_value``
    (``PValue``) and ``adjusted_p_value`` (``FDR``). The arrays are read-only.

    Attributes:
        fit: The quasi-likelihood fit tested.
        names: Names of the coefficients or contrasts tested together.
        contrasts: ``(n_coefficients, n_tested)`` weights of each tested contrast on
            the coefficients; the columns of the identity for coefficients.
        log_fold_change: ``(n_features, n_tested)`` log2 fold changes, each contrast
            of the shrunk coefficients divided by log 2.
        average_log_cpm: ``(n_features,)`` average log2 counts per million.
        likelihood_ratio: ``(n_features,)`` likelihood ratio statistics: the
            deviance of the null fit less that of the full fit, both at the working
            dispersion; at least 0.
        f_statistic: ``(n_features,)`` F-statistics, the likelihood ratios over
            ``df_test`` and the posterior quasi-dispersions.
        p_value: ``(n_features,)`` P(F > f) on ``df_test`` and ``df_total`` degrees
            of freedom.
        adjusted_p_value: ``(n_features,)`` Benjamini and Hochberg's adjusted
            p-values over the genes.
        df_test: Numerator degrees of freedom, the rank of ``contrasts``.
        df_total: ``(n_features,)`` denominator degrees of freedom (``df.total``):
            the prior plus the adjusted residual degrees of freedom, at most the
            residual degrees of freedom summed over the genes.
    """

    fit: QuasiLikelihoodFit
    names: tuple[str, ...]
    contrasts: FloatArray
    log_fold_change: FloatArray
    average_log_cpm: FloatArray
    likelihood_ratio: FloatArray
    f_statistic: FloatArray
    p_value: FloatArray
    adjusted_p_value: FloatArray
    df_test: int
    df_total: FloatArray

    def __repr__(self) -> str:
        n_significant = int(np.count_nonzero(self.adjusted_p_value < 0.05))
        return (
            f"QuasiLikelihoodTest(n_features={self.f_statistic.size}, "
            f"names={self.names!r}, df_test={self.df_test}, "
            f"adjusted_p_value_below_0.05={n_significant})"
        )


def quasi_likelihood_f_test(
    fit: QuasiLikelihoodFit,
    *,
    coefficients: str | int | Sequence[str | int] | None = None,
    contrasts: Any = None,
    names: Sequence[str] | None = None,
) -> QuasiLikelihoodTest:
    """Quasi-likelihood F-tests that coefficients or contrasts are zero, as edgeR's
    glmQLFTest.

    The null model is fitted at the working dispersion of ``fit``, without the
    tested coefficients, or on the design X N whose coefficients are those that the
    tested contrasts C leave free (N spans the vectors b with C' b = 0, so that every
    full-rank parametrisation of the same null model gives the same deviance). The
    likelihood ratio statistic, the null deviance less the full one, over the rank r
    of the contrasts and the posterior quasi-dispersion, is referred to the F
    distribution on r and ``df_total`` degrees of freedom (see the module
    description). Several coefficients or contrasts are tested together, as edgeR's
    matrix of contrasts is; test each named contrast in its own call for a table per
    contrast. A gene without counts gets an F-statistic of 0 and a p-value of 1 (NaN
    when the fit has at most two genes, whose quasi-dispersions are not moderated).

    Contrasts that name their coefficients are matched to them by name, as in
    :func:`~cytepool.stats.linear_models.contrasts_fit`: a
    :class:`~cytepool.design.Contrasts` (``make_contrasts``, ``Design.contrasts`` or
    its ``select``), a pandas DataFrame or Series, or a mapping of name to weights;
    other weights are taken in the order of the coefficients.

    Args:
        fit: A fit from :func:`fit_quasi_likelihood`.
        coefficients: The coefficients to test, by name or position; by default the
            last one, as in edgeR. Not with ``contrasts``.
        contrasts: Weights on the coefficients of the contrasts to test: a vector, a
            ``(n_coefficients, n_contrasts)`` matrix, or named contrasts as above.
        names: Names of the contrasts when ``contrasts`` carries none; by default
            ``contrast1, contrast2, ...``.

    Returns:
        The tests, with the columns of edgeR's ``topTags`` table.

    Raises:
        TypeError: If ``fit`` is not a ``QuasiLikelihoodFit``, or an argument has
            the wrong type or is a numpy masked array.
        ValueError: If both ``coefficients`` and ``contrasts`` are given, a
            coefficient is unknown or repeated, the contrasts are not finite, have
            the wrong number of rows or are all zero, the tests would leave the
            null model without coefficients (as edgeR's), or names are invalid.
    """
    if not isinstance(fit, QuasiLikelihoodFit):
        raise TypeError(f"fit must be a QuasiLikelihoodFit, got {type(fit).__name__}")
    model = fit.fit
    coefficient_names = model.coefficient_names
    n_coefficients = len(coefficient_names)
    x = np.asarray(model.design)
    if contrasts is not None:
        if coefficients is not None:
            raise ValueError("give coefficients or contrasts to test, not both")
        matrix, tested = contrast_matrix(contrasts, names, coefficient_names)
        if not np.isfinite(matrix).all():
            raise ValueError("contrasts must be finite")
        null_design, rank = _null_design(x, matrix)
        if rank == 0:
            raise ValueError("the contrasts are zero, so they test nothing")
    else:
        if names is not None:
            raise ValueError(
                "names name contrasts; they are not used with coefficients"
            )
        positions = _coefficient_positions(coefficients, coefficient_names)
        matrix = np.zeros((n_coefficients, len(positions)))
        matrix[positions, np.arange(len(positions))] = 1.0
        tested = tuple(coefficient_names[j] for j in positions)
        free = [j for j in range(n_coefficients) if j not in positions]
        null_design = np.ascontiguousarray(x[:, free])
        rank = len(positions)
    if null_design.shape[1] == 0:
        # edgeR refuses this too: its null model needs a coefficient (found by
        # probing).
        raise ValueError(
            "the tests leave the null model without coefficients: test fewer "
            "coefficients or contrasts than the design has columns"
        )
    null_deviance = _null_deviance(fit, null_design)
    likelihood_ratio = np.maximum(null_deviance - np.asarray(model.deviance), 0.0)
    posterior = np.asarray(fit.posterior_quasi_dispersion)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = likelihood_ratio / rank / posterior
    n_samples, n_features = model.fitted_values.shape
    cap = float(n_features * (n_samples - n_coefficients))
    df_total = np.minimum(
        np.asarray(fit.prior_df) + np.asarray(fit.df_residual_adjusted), cap
    )
    df_total = np.broadcast_to(df_total, (n_features,)).astype(np.float64)
    p_value = np.asarray(f_sf(f_statistic, float(rank), df_total), dtype=np.float64)
    log_fold_change = np.asarray(model.shrunk_coefficients) @ matrix / _LOG_TWO
    return QuasiLikelihoodTest(
        fit=fit,
        names=tuple(tested),
        contrasts=_read_only(np.array(matrix, dtype=np.float64)),
        log_fold_change=_read_only(np.ascontiguousarray(log_fold_change)),
        average_log_cpm=fit.average_log_cpm,
        likelihood_ratio=_read_only(likelihood_ratio),
        f_statistic=_read_only(np.asarray(f_statistic, dtype=np.float64)),
        p_value=_read_only(p_value),
        adjusted_p_value=_read_only(
            np.asarray(adjust_p_values(p_value, "bh"), dtype=np.float64)
        ),
        df_test=int(rank),
        df_total=_read_only(df_total),
    )


def _null_design(x: FloatArray, contrasts: FloatArray) -> tuple[FloatArray, int]:
    """The design X N of the null model C' b = 0, and the rank of C.

    N holds the right singular vectors of C' beyond its rank, by R's ``qr()`` rule
    on the singular values (relative to the largest, 1e-7).
    """
    _, values, vectors = np.linalg.svd(contrasts.T)
    if values.size == 0 or values[0] == 0.0:
        return np.ascontiguousarray(x), 0
    rank = int(np.count_nonzero(values > _CONTRAST_RANK * values[0]))
    null_space = vectors[rank:].T
    return np.ascontiguousarray(x @ null_space), rank


_CONTRAST_RANK = 1e-7


def _null_deviance(fit: QuasiLikelihoodFit, null_design: FloatArray) -> FloatArray:
    """Deviances of the null fits, at the dispersions, offsets and weights of the
    full ones."""
    model = fit.fit
    counts = np.asarray(fit.counts)
    null = fit_negative_binomial_model(
        counts,
        null_design,
        dispersion=np.asarray(model.dispersion),
        offsets=np.asarray(model.offsets),
        weights=np.asarray(model.weights),
        prior_count=0.0,
        max_iterations=fit.max_iterations,
    )
    return np.asarray(null.deviance)


# ---------------------------------------------------------------------------------------
# Arguments and small helpers
# ---------------------------------------------------------------------------------------


def _argument(matrix: FloatArray) -> Any:
    """A genes x samples matrix as the fits hold it (1 or G rows, 1 or n columns),
    as their argument: one value, one per sample, one per gene, or samples x genes."""
    rows, columns = matrix.shape
    if rows == 1 and columns == 1:
        return float(matrix[0, 0])
    if rows == 1:
        return np.array(matrix[0])
    if columns == 1:
        return np.array(matrix[:, 0])
    return np.ascontiguousarray(matrix.T)


def _rows(matrix: FloatArray, genes: npt.NDArray[np.int64]) -> FloatArray:
    """The rows of the given genes of a genes x samples matrix of the fits."""
    return matrix if matrix.shape[0] == 1 else np.ascontiguousarray(matrix[genes])


def _as_given(value: Any) -> float | FloatArray:
    if isinstance(value, float):
        return value
    return _read_only(np.asarray(value, dtype=np.float64))


def _proportion(value: Any) -> float:
    reject_masked(value, "top_proportion")
    share = real_number(value, "top_proportion")
    if not 0.0 < share <= 1.0:
        raise ValueError(f"top_proportion must be in (0, 1], got {share}")
    return share


def _trend_values(trend: Any, n_features: int) -> Any:
    """True, False, or the covariate of the trend as a float64 array."""
    if isinstance(trend, (bool, np.bool_)):
        return bool(trend)
    values = unmasked_array(trend, "trend", missing="NaN")
    if values.dtype.kind not in "iuf":
        raise TypeError(
            f"trend must be a bool or real covariate values, got dtype {values.dtype}"
        )
    if values.shape != (n_features,):
        raise ValueError(
            "trend must be a bool or one covariate value per feature "
            f"({n_features},), got shape {values.shape}"
        )
    return np.array(values, dtype=np.float64)


def _search_interval(interval: Any) -> tuple[float, float]:
    values = unmasked_array(interval, "interval")
    if values.dtype.kind not in "iuf" or values.shape != (2,):
        raise ValueError(
            "interval must hold two real numbers, the smallest and largest "
            f"dispersion searched; got {values!r}"
        )
    low, high = float(values[0]), float(values[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
        raise ValueError(
            f"interval must satisfy 0 <= low < high, both finite; got ({low}, {high})"
        )
    return low, high


def _coefficient_positions(
    coefficients: Any, coefficient_names: tuple[str, ...]
) -> list[int]:
    """Positions of the coefficients to test, by name or position; the last by
    default."""
    reject_masked(coefficients, "coefficients")
    n = len(coefficient_names)
    if coefficients is None:
        return [n - 1]
    if isinstance(coefficients, (str, numbers.Integral)):
        chosen = [coefficients]
    elif isinstance(coefficients, (Sequence, np.ndarray)):
        chosen = list(coefficients)
    else:
        raise TypeError(
            "coefficients must be a name, an integer position, or a sequence of them; "
            f"got {type(coefficients).__name__}"
        )
    if not chosen:
        raise ValueError("coefficients must name at least one coefficient")
    positions = []
    for item in chosen:
        if isinstance(item, str):
            if item not in coefficient_names:
                raise ValueError(
                    f"no coefficient named {item!r}; the coefficients are "
                    + ", ".join(repr(name) for name in coefficient_names)
                )
            positions.append(coefficient_names.index(item))
        elif isinstance(item, numbers.Integral) and not isinstance(
            item, (bool, np.bool_)
        ):
            position = int(item)
            if not -n <= position < n:
                raise ValueError(
                    f"coefficient position {position} is out of range for {n} "
                    "coefficients"
                )
            positions.append(position % n)
        else:
            raise TypeError(
                "coefficients must be names or integer positions, got "
                f"{type(item).__name__}"
            )
    if len(set(positions)) < len(positions):
        raise ValueError("a coefficient is listed more than once")
    return positions


def _frozen(values: Any) -> Any:
    if isinstance(values, float):
        return values
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return _read_only(np.array(array))
