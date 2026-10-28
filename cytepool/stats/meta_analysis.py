"""Meta-analysis of the effects of many features across studies, as metafor computes it.

:func:`fit_meta_analysis` meta-analyses, for every feature (gene), the estimates of
one effect in several studies, each with its standard error, by the fixed-effect or a
random-effects model, with the estimate of the between-study variance tau^2 and the
heterogeneity statistics that metafor's ``rma.uni`` reports for a model without
moderators (Viechtbauer 2010), into a :class:`MetaAnalysisFit`;
:func:`cytepool.meta_analysis` does it for the differential-expression results of
several studies. https://docs.nygen.io/CytePool/statistics/meta-analysis explains the
model, the estimators and the tests, and
https://docs.nygen.io/CytePool/statistics/meta-analysis#meta-analysis-limits how they
differ from metafor's.

Studies that did not measure a feature:
    NaN marks them, in the effect or the standard error. Each feature is analysed over
    the k studies that measured it, as metafor leaves out studies with missing values.
    A feature measured in one study has that study's estimate and standard error and
    its z-test, also with ``"knapp_hartung"``, which needs two studies; tau^2 = 0,
    Q = 0 with a p-value of 1, I^2 = 0 and H^2 = 1: metafor's results for one study. A
    feature measured in no study has NaN.

The REML estimate:
    The restricted log-likelihood, -1/2 sum log(v_i + tau^2) - 1/2 log(sum w_i) -
    1/2 sum w_i (y_i - mu_hat)^2 with w_i = 1 / (v_i + tau^2), can have more than one
    maximum. Its maxima are zero, where the score is not positive there, and the roots
    where the score, sum w_i^2 (y_i - mu_hat)^2 - tr(P), with P = W - w w' / sum w_i
    for the diagonal matrix W of the weights, falls from positive to negative. The
    score is negative beyond sum (y_i - y_bar)^2 / (k - 1) + 2 max(v_i). The roots are
    bracketed by the signs of the score at 8 points per decade of tau^2, from
    1e-3 min(v_i) to twice that bound, so that rounding cannot hide one near the bound;
    each is found by Newton's method with the observed information, kept in its
    bracket by bisection, and the estimate is the maximum with the largest restricted
    likelihood. The Paule-Mandel statistic falls with tau^2 and is convex, and its root
    lies between sum (y_i - y_bar)^2 / (k - 1) minus the largest and minus the smallest
    v_i, so Newton's method from the lower end converges to it from below.

Matching metafor:
    The results match metafor 5.0.1 run to convergence, with
    ``control = list(threshold = 1e-13)`` for REML and
    ``control = list(tol = 1e-14)`` for Paule-Mandel: by default metafor stops REML
    when tau^2 changes by less than 1e-5 from one step to the next, and Paule-Mandel
    at the tolerance of R's ``uniroot``, about 1.2e-4 on tau^2. Its documentation
    gives the methods, and running it on chosen inputs settled the conventions it
    leaves open, such as its results for one study above. I^2 and H^2 follow the
    estimate of tau^2, as metafor reports them (Higgins and Thompson 2002, equation 9).
    The Knapp-Hartung adjustment of the fixed-effect model scales the standard error
    by sqrt(Q / (k - 1)), which metafor computes with a warning that the method is
    not meant for that model.

References:
    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Cochran WG (1954). The combination of estimates from different experiments.
    Biometrics 10, 101-129. doi:10.2307/3001666

    DerSimonian R, Laird N (1986). Meta-analysis in clinical trials. Controlled Clinical
    Trials 7, 177-188. doi:10.1016/0197-2456(86)90046-2

    Hartung J (1999). An alternative method for meta-analysis. Biometrical Journal 41,
    901-916. doi:10.1002/(SICI)1521-4036(199912)41:8<901::AID-BIMJ901>3.0.CO;2-W

    Higgins JPT, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis.
    Statistics in Medicine 21, 1539-1558. doi:10.1002/sim.1186

    Knapp G, Hartung J (2003). Improved tests for a random effects meta-regression with
    a single covariate. Statistics in Medicine 22, 2693-2710. doi:10.1002/sim.1482

    Paule RC, Mandel J (1982). Consensus values and weighting factors. Journal of
    Research of the National Bureau of Standards 87, 377-385. doi:10.6028/jres.087.022

    Sidik K, Jonkman JN (2002). A simple confidence interval for meta-analysis.
    Statistics in Medicine 21, 3153-3159. doi:10.1002/sim.1262

    Viechtbauer W (2005). Bias and efficiency of meta-analytic variance estimators in
    the random-effects model. Journal of Educational and Behavioral Statistics 30,
    261-293. doi:10.3102/10769986030003261

    Viechtbauer W (2010). Conducting meta-analyses in R with the metafor package.
    Journal of Statistical Software 36(3), 1-48. doi:10.18637/jss.v036.i03
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .._jit import jit
from .._masked_arrays import read_only, reject_masked, unmasked_array
from .._options import checked_choice, real_number
from .distributions import (
    chi2_sf_scalar,
    norm_isf,
    norm_sf_scalar,
    t_isf,
    t_sf_scalar,
)

__all__ = ["MetaAnalysisFit", "fit_meta_analysis"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_METHODS = {"reml": 0, "dl": 1, "pm": 2, "fixed": 3}
_TESTS = ("z", "knapp_hartung")
_REML, _DERSIMONIAN_LAIRD, _PAULE_MANDEL, _FIXED = 0, 1, 2, 3
# The rows of the compiled routine's output, in order.
_OUTPUTS = (
    "estimate",
    "standard_error",
    "statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "tau_squared",
    "q_statistic",
    "q_p_value",
    "i_squared",
    "h_squared",
)
# The iterations of REML and Paule-Mandel stop when a step changes tau^2 by at most
# _TOLERANCE times tau^2 + s^2, s^2 the typical within-study variance, or after
# _MAX_ITERATIONS steps.
_TOLERANCE = 1e-13
_MAX_ITERATIONS = 1000
# The signs of the restricted score are read on a grid of this many points per decade
# of tau^2, to bracket every maximum of the restricted likelihood.
_GRID_PER_DECADE = 8


@dataclass(frozen=True, eq=False, repr=False)
class MetaAnalysisFit:
    """Meta-analyses of many features, from :func:`fit_meta_analysis`.

    The per-feature arrays have the shape of the effects without their first (study)
    axis: one value per feature, or 0-d arrays for the effects of one feature given
    as a vector. They are read-only.

    Attributes:
        estimate: The estimate of the common effect mu (metafor's ``beta``): the
            weighted mean of the studies' effects, with the weights
            1 / (v_i + tau^2).
        standard_error: Its standard error (``se``), adjusted by Knapp and Hartung
            with ``test="knapp_hartung"``.
        statistic: ``estimate / standard_error`` (``zval``): a z-statistic, or with
            ``test="knapp_hartung"`` a t-statistic on ``n_studies - 1`` degrees of
            freedom.
        p_value: Its two-sided p-value (``pval``).
        ci_lower: The lower limit of the confidence interval of the estimate at
            ``confidence_level`` (``ci.lb``).
        ci_upper: The upper limit (``ci.ub``).
        tau_squared: The estimate of the between-study variance tau^2 (``tau2``); 0
            for the fixed-effect model.
        q_statistic: Cochran's Q, the heterogeneity statistic of the fixed-effect
            fit (``QE``).
        q_p_value: Its p-value, from the chi-squared distribution with
            ``n_studies - 1`` degrees of freedom (``QEp``).
        i_squared: I^2, as a proportion in [0, 1] (metafor's ``I2`` is in percent).
        h_squared: H^2 (``H2``).
        n_studies: The number of studies that measured each feature (``k``).
        converged: Whether the iterations of REML or Paule-Mandel met their
            convergence criterion; True for the other methods.
        method: ``"reml"``, ``"dl"``, ``"pm"`` or ``"fixed"``.
        test: ``"z"`` or ``"knapp_hartung"``.
        confidence_level: The confidence level of the intervals.
    """

    estimate: FloatArray
    standard_error: FloatArray
    statistic: FloatArray
    p_value: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    tau_squared: FloatArray
    q_statistic: FloatArray
    q_p_value: FloatArray
    i_squared: FloatArray
    h_squared: FloatArray
    n_studies: IntArray
    converged: BoolArray
    method: str
    test: str
    confidence_level: float

    def __repr__(self) -> str:
        studies = self.n_studies[self.n_studies > 0]
        span = f"{studies.min()} to {studies.max()}" if studies.size else "none"
        return (
            f"MetaAnalysisFit(n_features={self.estimate.size}, method={self.method!r}, "
            f"test={self.test!r}, studies per feature: {span})"
        )


def fit_meta_analysis(
    effects: npt.ArrayLike,
    standard_errors: npt.ArrayLike,
    *,
    method: Literal["reml", "dl", "pm", "fixed"] = "reml",
    test: Literal["z", "knapp_hartung"] = "z",
    confidence_level: float = 0.95,
) -> MetaAnalysisFit:
    """Meta-analyse the effects of every feature across studies, as metafor's rma.uni.

    For every feature, the effects of the studies that measured it are combined into
    an estimate of a common effect by the fixed-effect or a random-effects model, with
    the between-study variance tau^2, Cochran's Q and its p-value, I^2 and H^2, as
    ``rma.uni(yi, sei = ..., method = ..., test = ...)`` gives them. Study i estimates
    the effect by y_i with the sampling variance v_i = s_i^2, taken as known; in the
    random-effects model y_i = mu + u_i + e_i, with u_i ~ N(0, tau^2) and
    e_i ~ N(0, v_i). With the weights w_i = 1 / (v_i + tau^2), the estimate is
    mu_hat = sum w_i y_i / sum w_i, with standard error 1 / sqrt(sum w_i).

    The estimators of tau^2 (``method``), over the k studies that measured a feature:
    ``"reml"``, the restricted maximum likelihood estimate (Viechtbauer 2005), the
    largest maximum of the restricted likelihood (see the module docstring);
    ``"dl"``, DerSimonian and Laird's, max(0, (Q - (k - 1)) / (sum w_i - sum w_i^2 /
    sum w_i)) with w_i = 1 / v_i; ``"pm"``, Paule and Mandel's (1982), at which the
    generalized Q statistic sum w_i (y_i - mu_hat)^2 equals its expectation k - 1, or
    0 where it is at most k - 1; and ``"fixed"``, tau^2 = 0.

    The tests (``test``): ``"z"``, z = mu_hat / SE against the standard normal
    distribution, with the interval mu_hat +/- z_(1 - alpha / 2) SE; or
    ``"knapp_hartung"``, the standard error times sqrt(q), with q = sum w_i (y_i -
    mu_hat)^2 / (k - 1), against a t distribution with k - 1 degrees of freedom
    (Hartung 1999; Sidik and Jonkman 2002; Knapp and Hartung 2003). Where q < 1 the
    adjusted standard error is the smaller.

    Heterogeneity: Cochran's (1954) Q = sum (y_i - mu_fixed)^2 / v_i, with the
    fixed-effect estimate mu_fixed, and its p-value from the chi-squared distribution
    with k - 1 degrees of freedom. I^2 = tau^2 / (tau^2 + s^2) and H^2 = (tau^2 + s^2)
    / s^2, where s^2 = (k - 1) sum w_i / ((sum w_i)^2 - sum w_i^2), with w_i = 1 / v_i,
    is the typical within-study variance (Higgins and Thompson 2002). For the
    fixed-effect model they are I^2 = max(0, (Q - (k - 1)) / Q) and H^2 = Q / (k - 1),
    which equal the others for DerSimonian-Laird where Q > k - 1; where Q <= k - 1, I^2
    is 0 for both, and H^2 is Q / (k - 1) for the fixed effect and 1 for
    DerSimonian-Laird. https://docs.nygen.io/CytePool/statistics/meta-analysis
    explains the model.

    Every estimate is computed for the deviations of the effects from their mean and
    the standard errors divided, exactly, by a power of two near the largest of them,
    so that the results do not depend on the scale of the data, and no square
    overflows or underflows however much larger the spread of the effects is than
    their standard errors. Sums that cancel when one study is far more precise than
    the others, such as sum w_i - sum w_i^2 / sum w_i, are formed from the sum of the
    weights of the other studies for each study, which does not cancel.

    Args:
        effects: ``(n_studies, n_features)`` estimates of the effect of every feature
            in every study, such as log fold changes, or ``(n_studies,)`` for one
            feature. Finite, or NaN where a study did not measure a feature.
        standard_errors: Their standard errors, of the same shape: positive and
            finite, or NaN where a study did not measure a feature. A study is left
            out of a feature's analysis where its effect or its standard error is NaN.
        method: The estimator of the between-study variance: ``"reml"`` (restricted
            maximum likelihood, metafor's default), ``"dl"`` (DerSimonian-Laird),
            ``"pm"`` (Paule-Mandel), or ``"fixed"`` for the fixed-effect model.
        test: ``"z"`` for z-tests and normal confidence intervals, or
            ``"knapp_hartung"`` for the Knapp-Hartung adjustment, with t-tests on
            ``n_studies - 1`` degrees of freedom.
        confidence_level: The confidence level of the intervals, in (0, 1). metafor
            takes it in percent as ``level`` (``level = 95``).

    Returns:
        The estimate, standard error, statistic, p-value and confidence interval of
        every feature, with tau^2, Q and its p-value, I^2, H^2 and the number of
        studies.

    Raises:
        TypeError: If ``effects`` or ``standard_errors`` does not hold real numbers or
            is a numpy masked array, or an option has the wrong type.
        ValueError: If the shapes differ or are not 1-D or 2-D, there is no study, an
            effect or a standard error is infinite, a standard error is not positive,
            ``method`` or ``test`` is unknown, or ``confidence_level`` is not in
            (0, 1).
    """
    chosen_method, chosen_test, confidence = _checked_options(
        method, test, confidence_level
    )
    y = _study_values(effects, "effects")
    se = _study_values(standard_errors, "standard_errors")
    if y.shape != se.shape:
        raise ValueError(
            f"effects and standard_errors must have the same shape, got {y.shape} and "
            f"{se.shape}; both are (n_studies, n_features)"
        )
    if y.ndim not in (1, 2):
        raise ValueError(
            "effects must be (n_studies, n_features), or (n_studies,) for one "
            f"feature, got {y.ndim} dimensions"
        )
    if y.shape[0] == 0:
        raise ValueError("there are no studies: effects has no rows")
    _check_values(y, se)
    n_studies = y.shape[0]
    by_feature = np.ascontiguousarray(y.reshape(n_studies, -1).T)
    errors = np.ascontiguousarray(se.reshape(n_studies, -1).T)
    n_features = by_feature.shape[0]

    alpha = 1.0 - confidence
    z_critical = float(norm_isf(alpha / 2.0))
    t_critical = np.full(n_studies, np.nan)
    if n_studies > 1:
        t_critical[1:] = t_isf(alpha / 2.0, np.arange(1.0, n_studies))
    results = np.empty((len(_OUTPUTS), n_features))
    counts = np.empty(n_features, dtype=np.int64)
    converged = np.empty(n_features, dtype=np.bool_)
    _meta_analyse(
        by_feature,
        errors,
        _METHODS[chosen_method],
        chosen_test == "knapp_hartung",
        z_critical,
        t_critical,
        results,
        counts,
        converged,
    )
    shape = y.shape[1:]
    arrays = {
        name: read_only(results[row].reshape(shape))
        for row, name in enumerate(_OUTPUTS)
    }
    return MetaAnalysisFit(
        **arrays,
        n_studies=read_only(counts.reshape(shape)),
        converged=read_only(converged.reshape(shape)),
        method=chosen_method,
        test=chosen_test,
        confidence_level=confidence,
    )


# Checks of the arguments.


def _study_values(values: Any, name: str) -> FloatArray:
    array = unmasked_array(values, name, missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    converted: FloatArray = np.array(array, dtype=np.float64)
    return converted


def _check_values(y: FloatArray, se: FloatArray) -> None:
    infinite = np.isinf(y)
    if infinite.any():
        position = tuple(int(i) for i in np.argwhere(infinite)[0])
        raise ValueError(
            f"effects must be finite, or NaN where a study did not measure a feature; "
            f"found {np.count_nonzero(infinite)} infinite, such as {y[position]} at "
            f"{position}"
        )
    invalid = ~np.isnan(se) & ~((se > 0.0) & np.isfinite(se))
    if invalid.any():
        position = tuple(int(i) for i in np.argwhere(invalid)[0])
        raise ValueError(
            "standard_errors must be positive and finite, or NaN where a study did "
            f"not measure a feature; found {np.count_nonzero(invalid)} that are not, "
            f"such as {se[position]} at {position}"
        )


def _checked_options(
    method: Any, test: Any, confidence_level: Any
) -> tuple[str, str, float]:
    """The estimator, the test and the confidence level, checked in that order."""
    return (
        checked_choice(method, "method", tuple(_METHODS)),
        checked_choice(test, "test", _TESTS),
        _confidence_level(confidence_level),
    )


def _confidence_level(value: Any) -> float:
    # real_number would refuse a masked value too, but without saying how to fix it.
    reject_masked(value, "confidence_level")
    number = real_number(value, "confidence_level", finite=True)
    if not 0.0 < number < 1.0:
        raise ValueError(
            "confidence_level must be in (0, 1), such as 0.95 for 95% confidence "
            f"intervals, got {number}"
        )
    return number


# The meta-analyses, compiled. Each routine works on the k studies that measured a
# feature, held in the first k entries of its arrays: the deviations of the effects
# from their unweighted mean (d) and the sampling variances (v), at unit scale.


@jit(error_model="numpy")
def _sums_without_each(w: FloatArray, k: int, without: FloatArray) -> float:
    """Set ``without[i]`` to the sum of ``w`` over the studies other than i, and
    return the sum over all. The partial sums before and after i hold no term larger
    than their total, so nothing cancels, as it would in ``total - w[i]``."""
    total = 0.0
    for i in range(k):
        without[i] = total
        total += w[i]
    after = 0.0
    for i in range(k - 1, -1, -1):
        without[i] += after
        after += w[i]
    return total


@jit(error_model="numpy")
def _fixed_effect(
    d: FloatArray, v: FloatArray, k: int, w: FloatArray, without: FloatArray
) -> tuple[float, float]:
    """Cochran's Q and c = sum w - sum w^2 / sum w, for the weights w = 1 / v."""
    for i in range(k):
        w[i] = 1.0 / v[i]
    total = _sums_without_each(w, k, without)
    centre = 0.0
    for i in range(k):
        centre += w[i] * d[i]
    centre /= total
    q = 0.0
    c = 0.0
    for i in range(k):
        r = d[i] - centre
        q += w[i] * r * r
        c += w[i] * (without[i] / total)  # the ratio is at most 1: no overflow
    return q, c


@jit(error_model="numpy")
def _restricted_score(
    d: FloatArray,
    v: FloatArray,
    k: int,
    tau2: float,
    w: FloatArray,
    w2: FloatArray,
    without: FloatArray,
    without2: FloatArray,
) -> tuple[float, float]:
    """Twice the score of the restricted log-likelihood at tau2, sum w^2 r^2 - tr(P),
    and twice its observed information, 2 sum w^3 r^2 - 2 (sum w^2 r)^2 / sum w -
    tr(P^2), the derivative of the score with its sign changed."""
    for i in range(k):
        w[i] = 1.0 / (v[i] + tau2)
        w2[i] = w[i] * w[i]
    total = _sums_without_each(w, k, without)
    _sums_without_each(w2, k, without2)
    centre = 0.0
    for i in range(k):
        centre += w[i] * d[i]
    centre /= total
    squares = 0.0
    cubes = 0.0
    linear = 0.0
    trace = 0.0
    trace_squared = 0.0
    for i in range(k):
        weighted = w[i] * (d[i] - centre)  # w r, squared without underflow
        squares += weighted * weighted
        cubes += w[i] * weighted * weighted
        linear += w[i] * weighted
        # P_ii = w_i (sum of the other weights) / total, P_ij = -w_i w_j / total.
        diagonal = w[i] * (without[i] / total)
        trace += diagonal
        trace_squared += diagonal * diagonal + (w[i] / total) ** 2 * without2[i]
    observed = 2.0 * cubes - 2.0 * linear * linear / total - trace_squared
    return squares - trace, observed


@jit(error_model="numpy")
def _restricted_log_likelihood(
    d: FloatArray, v: FloatArray, k: int, tau2: float
) -> float:
    """The restricted log-likelihood at tau2, without its constant."""
    total = 0.0
    centre = 0.0
    logs = 0.0
    for i in range(k):
        weight = 1.0 / (v[i] + tau2)
        total += weight
        centre += weight * d[i]
        logs += math.log(v[i] + tau2)
    centre /= total
    q = 0.0
    for i in range(k):
        r = d[i] - centre
        q += r * r / (v[i] + tau2)
    return -0.5 * (logs + math.log(total) + q)


@jit(error_model="numpy")
def _score_root(
    d: FloatArray,
    v: FloatArray,
    k: int,
    lower: float,
    upper: float,
    typical: float,
    w: FloatArray,
    w2: FloatArray,
    without: FloatArray,
    without2: FloatArray,
) -> tuple[float, bool]:
    """The root of the restricted score in (lower, upper), where it falls from positive
    to negative: Newton's method with the observed information, kept in the bracket
    and replaced by bisection where its step leaves it or does not halve every two
    steps."""
    tau2 = 0.5 * (lower + upper)
    last = upper - lower
    before_last = last
    for _ in range(_MAX_ITERATIONS):
        score, observed = _restricted_score(d, v, k, tau2, w, w2, without, without2)
        if score > 0.0:
            lower = tau2
        elif score < 0.0:
            upper = tau2
        else:
            return tau2, True
        step = score / observed if observed > 0.0 else math.inf
        if not (lower < tau2 + step < upper and abs(step) < 0.5 * before_last):
            step = 0.5 * (lower + upper) - tau2
        before_last, last = last, abs(step)
        tau2 += step
        if abs(step) <= _TOLERANCE * (tau2 + typical):
            return tau2, True
    return tau2, False


@jit(error_model="numpy")
def _reml(
    d: FloatArray,
    v: FloatArray,
    k: int,
    typical: float,
    w: FloatArray,
    w2: FloatArray,
    without: FloatArray,
    without2: FloatArray,
) -> tuple[float, bool]:
    """The REML estimate of tau^2, the largest maximum of the restricted likelihood
    over tau2 >= 0, and whether its iteration converged; NaN and False where no
    maximum is found, which the bounds below exclude in exact arithmetic.

    The maxima are zero, where the score there is not positive, and the roots where
    the score falls from positive to negative. The score is negative beyond
    sum d^2 / (k - 1) + 2 max(v), where w_max^2 sum d^2 bounds sum w^2 r^2 and
    (k - 1) w_min^2 / w_max bounds tr(P) from below. The roots are bracketed by the
    signs of the score on a grid with _GRID_PER_DECADE points per decade from
    1e-3 min(v) to twice that bound, so that rounding cannot hide the change of sign
    of a root near the bound (as when the standard errors are far smaller than the
    spread of the effects), and each is found by ``_score_root``.
    """
    squares = 0.0
    smallest = math.inf
    largest = 0.0
    for i in range(k):
        squares += d[i] * d[i]
        smallest = min(smallest, v[i])
        largest = max(largest, v[i])
    bottom = max(1e-3 * smallest, 1e-300)  # positive, should a variance underflow
    top = 2.0 * (squares / (k - 1) + 2.0 * largest)
    if not top < math.inf:
        return math.nan, False
    n = max(2, int(math.ceil(_GRID_PER_DECADE * math.log10(top / bottom))) + 1)
    ratio = (top / bottom) ** (1.0 / (n - 1))
    best = math.nan
    best_likelihood = -math.inf
    converged = True
    left = 0.0
    left_score = _restricted_score(d, v, k, 0.0, w, w2, without, without2)[0]
    if left_score <= 0.0:
        best = 0.0
        best_likelihood = _restricted_log_likelihood(d, v, k, 0.0)
    for j in range(n):
        right = top if j == n - 1 else bottom * ratio**j
        right_score = _restricted_score(d, v, k, right, w, w2, without, without2)[0]
        if left_score > 0.0 and right_score <= 0.0:
            if right_score == 0.0:
                root, ok = right, True
            else:
                root, ok = _score_root(
                    d, v, k, left, right, typical, w, w2, without, without2
                )
            converged = converged and ok
            likelihood = _restricted_log_likelihood(d, v, k, root)
            if likelihood > best_likelihood or math.isnan(best):
                best, best_likelihood = root, likelihood
        left, left_score = right, right_score
    return best, converged and not math.isnan(best)


@jit(error_model="numpy")
def _generalized_q(
    d: FloatArray, v: FloatArray, k: int, tau2: float
) -> tuple[float, float]:
    """The generalized Q statistic at tau2 and minus its derivative, sum w^2 r^2."""
    total = 0.0
    centre = 0.0
    for i in range(k):
        weight = 1.0 / (v[i] + tau2)
        total += weight
        centre += weight * d[i]
    centre /= total
    q = 0.0
    slope = 0.0
    for i in range(k):
        weight = 1.0 / (v[i] + tau2)
        r = d[i] - centre
        q += weight * r * r
        slope += (weight * r) ** 2
    return q, slope


@jit(error_model="numpy")
def _paule_mandel(
    d: FloatArray, v: FloatArray, k: int, q_fixed: float, typical: float
) -> tuple[float, bool]:
    """The Paule-Mandel estimate of tau^2 by Newton's method, and whether it
    converged."""
    target = k - 1.0
    if not q_fixed > target:
        return 0.0, True
    squares = 0.0
    smallest = math.inf
    largest = 0.0
    for i in range(k):
        squares += d[i] * d[i]
        smallest = min(smallest, v[i])
        largest = max(largest, v[i])
    # 1 / (largest + tau2) <= w_i <= 1 / (smallest + tau2) brackets Q(tau2) between
    # sum d^2 / (largest + tau2) and sum d^2 / (smallest + tau2).
    tau2 = max(0.0, squares / target - largest)
    upper = squares / target - smallest
    for _ in range(_MAX_ITERATIONS):
        q, slope = _generalized_q(d, v, k, tau2)
        excess = q - target
        if not excess > 0.0:
            return tau2, True
        updated = min(tau2 + excess / slope, upper)
        if updated - tau2 <= _TOLERANCE * (updated + typical):
            return updated, True
        tau2 = updated
    return tau2, False


@jit(error_model="numpy")
def _meta_analyse(
    effects: FloatArray,
    standard_errors: FloatArray,
    method: int,
    knapp_hartung: bool,
    z_critical: float,
    t_critical: FloatArray,
    results: FloatArray,
    n_studies: IntArray,
    converged: BoolArray,
) -> None:
    """Meta-analyse every feature: rows of ``effects`` and ``standard_errors`` are
    features, columns studies, NaN where a study did not measure a feature. Fills the
    rows of ``results`` in the order of ``_OUTPUTS``."""
    n_features, m = effects.shape
    d = np.empty(m)
    v = np.empty(m)
    w = np.empty(m)
    w2 = np.empty(m)
    without = np.empty(m)
    without2 = np.empty(m)
    for g in range(n_features):
        k = 0
        mean = 0.0
        for i in range(m):
            y = effects[g, i]
            s = standard_errors[g, i]
            if math.isnan(y) or math.isnan(s):
                continue
            d[k] = y
            v[k] = s
            mean += y
            k += 1
        n_studies[g] = k
        converged[g] = True
        if k == 0:
            results[:, g] = math.nan
            continue
        mean /= k
        largest = 0.0
        for i in range(k):
            d[i] -= mean
            largest = max(largest, v[i], abs(d[i]))
        # Divided, exactly, by a power of two near the largest standard error or
        # deviation of the effects from their mean: every number below is then at most
        # of order one.
        scale = math.ldexp(1.0, math.frexp(largest)[1])
        for i in range(k):
            d[i] /= scale
            v[i] /= scale
        single = v[0]  # the standard error of a feature of one study, as given
        for i in range(k):
            v[i] *= v[i]

        tau2 = 0.0
        if k == 1:
            # metafor's results for one study, the z-test of its estimate also with
            # the Knapp-Hartung adjustment (found by probing).
            q = 0.0
            q_p_value = 1.0
            i_squared = 0.0
            h_squared = 1.0
        else:
            q, c = _fixed_effect(d, v, k, w, without)
            typical = (k - 1) / c
            q_p_value = chi2_sf_scalar(q, k - 1.0)
            ok = True
            if method == _REML:
                tau2, ok = _reml(d, v, k, typical, w, w2, without, without2)
            elif method == _DERSIMONIAN_LAIRD:
                tau2 = max(0.0, (q - (k - 1)) / c)
            elif method == _PAULE_MANDEL:
                tau2, ok = _paule_mandel(d, v, k, q, typical)
            converged[g] = ok
            if method == _FIXED:
                i_squared = (q - (k - 1)) / q if q > k - 1 else 0.0
                h_squared = q / (k - 1)
            else:
                i_squared = tau2 / (tau2 + typical)
                h_squared = (tau2 + typical) / typical

        total = 0.0
        centre = 0.0
        for i in range(k):
            weight = 1.0 / (v[i] + tau2)
            total += weight
            centre += weight * d[i]
        centre /= total
        error = single if k == 1 else math.sqrt(1.0 / total)
        if knapp_hartung and k > 1:
            spread = 0.0
            for i in range(k):
                r = d[i] - centre
                spread += r * r / (v[i] + tau2)
            error = math.sqrt(spread / (k - 1) / total)
            critical = t_critical[k - 1]
        else:
            critical = z_critical
        estimate = mean + centre * scale
        error *= scale
        statistic = estimate / error
        if knapp_hartung and k > 1:
            p_value = 2.0 * t_sf_scalar(abs(statistic), k - 1.0)
        else:
            p_value = 2.0 * norm_sf_scalar(abs(statistic))
        results[0, g] = estimate
        results[1, g] = error
        results[2, g] = statistic
        results[3, g] = p_value
        results[4, g] = estimate - critical * error
        results[5, g] = estimate + critical * error
        results[6, g] = tau2 * scale * scale
        results[7, g] = q
        results[8, g] = q_p_value
        results[9, g] = i_squared
        results[10, g] = h_squared
