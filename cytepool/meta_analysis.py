"""Meta-analysis of the effects of many features across studies, as metafor computes it.

:func:`meta_analysis` combines, for every feature (gene), the estimates of one effect
in several studies, each with its standard error, into an estimate of a common effect,
with the between-study variance and the heterogeneity statistics, as metafor's
``rma.uni`` does for one feature without moderators (Viechtbauer 2010).
:func:`differential_expression_meta_analysis` does it for the differential-expression
results of several cubes, such as the same cell types in several cohorts, tissues or
platforms, matched by group, contrast and feature id.

The model:
    Study i estimates the effect of a feature by y_i, with standard error s_i, and its
    sampling variance v_i = s_i^2 is taken as known. In the random-effects model the
    true effects of the studies vary around a mean mu with the between-study variance
    tau^2: y_i = mu + u_i + e_i, with u_i ~ N(0, tau^2) and e_i ~ N(0, v_i)
    (DerSimonian and Laird 1986). With the weights w_i = 1 / (v_i + tau^2), mu is
    estimated by mu_hat = sum w_i y_i / sum w_i, with standard error
    1 / sqrt(sum w_i). The fixed-effect model, which assumes one true effect (metafor's
    equal-effects model, ``method = "EE"``), is the case tau^2 = 0: the
    inverse-variance weighted mean.

The between-study variance (``method``), over the k studies that measured a feature:
    - ``"reml"``, metafor's default: the restricted maximum likelihood estimate
      (Viechtbauer 2005), the tau^2 >= 0 that maximizes
      -1/2 sum log(v_i + tau^2) - 1/2 log(sum w_i) - 1/2 sum w_i (y_i - mu_hat)^2.
      This function can have more than one maximum, as when an imprecise study lies
      far from precise ones that agree. Its maxima are zero, where the score is not
      positive there, and the roots where the score, sum w_i^2 (y_i - mu_hat)^2 -
      tr(P), with P = W - w w' / sum w_i for the diagonal matrix W of the weights,
      falls from positive to negative. The score is negative beyond
      sum (y_i - y_bar)^2 / (k - 1) + 2 max(v_i). The roots are bracketed by the
      signs of the score at 8 points per decade of tau^2, from 1e-3 min(v_i) to twice
      that bound, so that rounding cannot hide one near the bound; each is found by
      Newton's method with the observed information, kept in its bracket by
      bisection, and the estimate is the maximum with the largest restricted
      likelihood.
    - ``"dl"``: the DerSimonian and Laird (1986) estimate, the method of moments
      with the weights 1 / v_i: max(0, (Q - (k - 1)) / (sum w_i - sum w_i^2 /
      sum w_i)), with Q below.
    - ``"pm"``: the Paule and Mandel (1982) estimate, at which the generalized Q
      statistic sum w_i (y_i - mu_hat)^2 equals its expectation k - 1, or 0 where Q
      is at most k - 1. The statistic falls with tau^2 and is convex, and the root
      lies between sum (y_i - y_bar)^2 / (k - 1) minus the largest and minus the
      smallest v_i, so Newton's method from the lower end converges to it from below.
    - ``"fixed"``: tau^2 = 0.

Tests and confidence intervals (``test``):
    - ``"z"``: z = mu_hat / SE against the standard normal distribution, and the
      interval mu_hat +/- z_(1 - alpha / 2) SE, as metafor's default ``test = "z"``.
    - ``"knapp_hartung"``: the standard error is multiplied by sqrt(q), with
      q = sum w_i (y_i - mu_hat)^2 / (k - 1), and the statistic is referred to a t
      distribution with k - 1 degrees of freedom (Hartung 1999; Sidik and Jonkman
      2002; Knapp and Hartung 2003; metafor's ``test = "knha"``), which metafor's
      documentation recommends for few studies. Where q < 1, as when the studies
      agree more closely than their standard errors imply, the adjusted standard
      error is smaller than the unadjusted one; with the fixed-effect model it is
      the standard error scaled by sqrt(Q / (k - 1)), which metafor computes with a
      warning that the method is not meant for that model.

Heterogeneity:
    Cochran's (1954) Q = sum (y_i - mu_fixed)^2 / v_i, with the fixed-effect estimate
    mu_fixed, and its p-value from the chi-squared distribution with k - 1 degrees of
    freedom. I^2 and H^2 are those metafor reports: I^2 = tau^2 / (tau^2 + s^2) and
    H^2 = (tau^2 + s^2) / s^2, where s^2 = (k - 1) sum w_i / ((sum w_i)^2 -
    sum w_i^2), with w_i = 1 / v_i, is the typical within-study variance (Higgins and
    Thompson 2002, equation 9), so that they follow the estimate of tau^2. For the
    fixed-effect model they are I^2 = max(0, (Q - (k - 1)) / Q) and H^2 = Q / (k - 1),
    the conventional definitions, which equal the others for DerSimonian-Laird where
    Q > k - 1; where Q <= k - 1, I^2 is 0 for both, and H^2 is Q / (k - 1) for the
    fixed effect and 1 for DerSimonian-Laird. I^2 is a proportion here; metafor
    reports it in percent.

Studies that did not measure a feature:
    NaN marks them, in the effect or the standard error. Each feature is analysed
    over the k studies that measured it, as metafor leaves out studies with missing
    values. A feature measured in one study has that study's estimate and standard
    error and its z-test, also with ``"knapp_hartung"``, which needs two studies;
    tau^2 = 0, Q = 0 with a p-value of 1, I^2 = 0 and H^2 = 1: metafor's results for
    one study (found by probing). A feature measured in no study has NaN.

Accuracy:
    Every estimate is computed for the deviations of the effects from their mean and
    the standard errors divided, exactly, by a power of two near the largest of them,
    so that the results do not depend on the scale of the data, and no square
    overflows or underflows however much larger the spread of the effects is than
    their standard errors. The sums sum w_i - sum w_i^2 / sum w_i and tr(P^2), which
    cancel when one study is far more precise than the others, are formed from the
    sum of the weights of the other studies for each study, which does not cancel.

Deliberate differences from metafor (``validation/meta_analysis.py`` reports them):
    - The REML estimate is the largest maximum of the restricted likelihood. metafor
      finds a maximum by Fisher scoring from the Hedges estimate,
      max(0, sum (y_i - y_bar)^2 / (k - 1) - mean(v_i)), and replaces it by zero, with
      a warning that Fisher scoring "may have gotten stuck at a local maximum", where
      the restricted likelihood is larger at zero (found by probing). Where the
      likelihood has another, larger maximum, metafor's estimate is the smaller one or
      zero. Fisher scoring also oscillates without converging where the observed
      information exceeds twice the expected one: metafor then stops with an error,
      or, at its default threshold, within the oscillation, and its help page's remedy
      ``control = list(stepadj = 0.5)`` converges to the estimate here.
    - The iterative estimates converge to rounding. metafor stops REML when tau^2
      changes by less than 1e-5 from one step to the next, and Paule-Mandel at the
      tolerance of R's ``uniroot``, about 1.2e-4 on tau^2; ``control = list(threshold
      = 1e-13)`` and ``control = list(tol = 1e-14)`` make it converge to the values
      here.
    - Where the restricted likelihood is largest at zero although the Hedges estimate
      is positive, the REML estimate is zero; metafor's step halving approaches zero
      and stops near its threshold, at about 1e-6 by default (found by probing).
    - metafor's Paule-Mandel search stops with an error when the root lies above
      max(100, 10 mad(y)^2); here it has no upper limit.
    - Standard errors must be positive and finite where they are given. For a zero
      standard error metafor stops (fixed effect, DerSimonian-Laird, Paule-Mandel) or
      fits without Q, I^2 and H^2 (REML); for an infinite one it stops.

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

import json
import math
import operator
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ._analysis_records import software_versions
from ._columns import text_column
from ._jit import jit
from ._masked_arrays import reject_masked, unmasked_array
from ._messages import listing, quoted
from ._options import checked_choice
from .differential_expression import DifferentialExpressionResult
from .normalization import _real
from .stats.distributions import (
    chi2_sf_scalar,
    norm_isf,
    norm_sf_scalar,
    t_isf,
    t_sf_scalar,
)
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = [
    "MetaAnalysisFit",
    "MetaAnalysisResult",
    "differential_expression_meta_analysis",
    "meta_analysis",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_METHODS = {"reml": 0, "dl": 1, "pm": 2, "fixed": 3}
_REML, _DERSIMONIAN_LAIRD, _PAULE_MANDEL, _FIXED = 0, 1, 2, 3
_TESTS = ("z", "knapp_hartung")
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
_ANALYSED = "analysed"  # the status of a group and contrast that was meta-analysed
# Warnings are attributed to the first caller outside the package.
_PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")


@dataclass(frozen=True, eq=False, repr=False)
class MetaAnalysisFit:
    """Meta-analyses of many features, from :func:`meta_analysis`.

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
            ``level`` (``ci.lb``).
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
        level: The confidence level of the intervals.
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
    level: float

    def __repr__(self) -> str:
        studies = self.n_studies[self.n_studies > 0]
        span = f"{studies.min()} to {studies.max()}" if studies.size else "none"
        return (
            f"MetaAnalysisFit(n_features={self.estimate.size}, method={self.method!r}, "
            f"test={self.test!r}, studies per feature: {span})"
        )


def meta_analysis(
    effects: npt.ArrayLike,
    standard_errors: npt.ArrayLike,
    *,
    method: Literal["reml", "dl", "pm", "fixed"] = "reml",
    test: Literal["z", "knapp_hartung"] = "z",
    level: float = 0.95,
) -> MetaAnalysisFit:
    """Meta-analyse the effects of every feature across studies, as metafor's rma.uni.

    For every feature, the effects of the studies that measured it are combined into
    an estimate of a common effect by the fixed-effect or a random-effects model, with
    the between-study variance tau^2, Cochran's Q and its p-value, I^2 and H^2, as
    ``rma.uni(yi, sei = ..., method = ..., test = ...)`` gives them; the module
    description gives the model, the estimators of tau^2 and the tests.

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
        level: The confidence level of the intervals, in (0, 1). metafor takes it in
            percent (``level = 95``).

    Returns:
        The estimate, standard error, statistic, p-value and confidence interval of
        every feature, with tau^2, Q and its p-value, I^2, H^2 and the number of
        studies.

    Raises:
        TypeError: If ``effects`` or ``standard_errors`` does not hold real numbers or
            is a numpy masked array, or an option has the wrong type.
        ValueError: If the shapes differ or are not 1-D or 2-D, there is no study, an
            effect or a standard error is infinite, a standard error is not positive,
            ``method`` or ``test`` is unknown, or ``level`` is not in (0, 1).
    """
    chosen_method = checked_choice(method, "method", tuple(_METHODS))
    chosen_test = checked_choice(test, "test", _TESTS)
    confidence = _level(level)
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
        name: _read_only(results[row].reshape(shape))
        for row, name in enumerate(_OUTPUTS)
    }
    return MetaAnalysisFit(
        **arrays,
        n_studies=_read_only(counts.reshape(shape)),
        converged=_read_only(converged.reshape(shape)),
        method=chosen_method,
        test=chosen_test,
        level=confidence,
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


def _level(value: Any) -> float:
    # _real would refuse a masked value too, but without saying how to fix it.
    reject_masked(value, "level")
    number = _real(value, "level")
    if not 0.0 < number < 1.0:
        raise ValueError(
            f"level must be in (0, 1), such as 0.95 for 95% confidence intervals, got "
            f"{number}"
        )
    return number


def _read_only(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    view = values.view()
    view.flags.writeable = False
    return view


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


# Meta-analysis of differential-expression results.


class MetaAnalysisResult:
    """Meta-analyses of the differential expression of several studies.

    Built by :func:`differential_expression_meta_analysis`. Tables hold, for every
    group and contrast meta-analysed, one row per feature that at least
    ``min_studies`` studies tested, in the order in which the studies, taken in the
    order given, first list them:

    - ``feature_id`` and, when a study's tables have one, ``symbol``: the first symbol
      that a study gives for the feature, taking the studies in the order given;
    - ``n_studies``: the number of studies that tested the feature;
    - ``log2_fold_change`` and ``standard_error``: the estimate of the common log2
      fold change and its standard error;
    - ``statistic`` and ``p_value``: the z-statistic, or the Knapp-Hartung
      t-statistic on ``n_studies - 1`` degrees of freedom, and its two-sided p-value.
      A feature that one study tested, kept with ``min_studies=1``, has the z-test of
      that study's log2 fold change against its standard error, also with
      ``"knapp_hartung"``;
    - ``adjusted_p_value``: Benjamini-Hochberg over the features of the table
      (Benjamini and Hochberg 1995);
    - ``ci_lower`` and ``ci_upper``: the confidence interval of the log2 fold change;
    - ``tau_squared``: the between-study variance;
    - ``i_squared`` and ``h_squared``: I^2, as a proportion, and H^2;
    - ``q_statistic`` and ``q_p_value``: Cochran's Q and its p-value.

    The summary has one row per group and contrast: ``group_id``, ``contrast``,
    ``status`` (``"analysed"``; ``"too_few_studies"`` when fewer than ``min_studies``
    studies tested it; ``"no_features"`` when no feature was tested by that many),
    ``reason`` (empty when analysed), ``n_studies`` (the studies that tested it),
    ``n_features`` (in its table) and ``n_features_left_out`` (tested by fewer than
    ``min_studies`` studies). The table of studies has one row per group, contrast and
    study: ``group_id``, ``contrast``, ``study``, ``tested`` (whether the study's
    results have a table of the group and contrast), ``reason`` (why not; empty when
    tested) and ``n_features`` (in the study's table; 0 when not tested).

    Immutable.

    Raises:
        TypeError: If ``summary`` or ``studies`` is not a ``Table``, ``tables`` not a
            mapping of (group id, contrast name) to ``Table``, or ``provenance`` not a
            mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_summary", "_studies", "_tables", "_record")

    _summary: Table
    _studies: Table
    _tables: Mapping[tuple[str, str], Table]
    _record: str

    def __init__(
        self,
        summary: Table,
        studies: Table,
        tables: Mapping[tuple[str, str], Table],
        provenance: Mapping[str, Any],
    ) -> None:
        for name, table in (("summary", summary), ("studies", studies)):
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
            "_summary": summary,
            "_studies": studies,
            "_tables": MappingProxyType(dict(tables)),
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MetaAnalysisResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MetaAnalysisResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            MetaAnalysisResult,
            (
                self._summary,
                self._studies,
                dict(self._tables),
                json.loads(self._record),
            ),
        )

    @property
    def summary(self) -> Table:
        """The status, studies and features of every group and contrast."""
        return self._summary

    @property
    def studies(self) -> Table:
        """Every study of every group and contrast, whether it tested them, and why not."""
        return self._studies

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """The table of every group and contrast meta-analysed, by (group, contrast)."""
        return self._tables

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the meta-analysis, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions; the settings; the
        provenance of every study's differential-expression result, by study name,
        which identifies its cube by its fingerprint and gives its design and
        settings and the samples of every group's fit; and, for every group and
        contrast, its status, the studies that tested it and those that did not with
        the reason, the number of features in its table and left out of it, and the
        features whose estimate of tau^2 did not converge (``not_converged``).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str, contrast: str) -> Table:
        """Return the table of one group and contrast.

        Args:
            group: A group id.
            contrast: A contrast name.

        Raises:
            KeyError: If the group or contrast was not analysed, fewer than
                ``min_studies`` studies tested it, or none of its features; the message
                says why.
        """
        found = self._tables.get((group, contrast))
        if found is not None:
            return found
        groups = list(dict.fromkeys(self._summary["group_id"].tolist()))
        contrasts = list(dict.fromkeys(self._summary["contrast"].tolist()))
        if group not in groups:
            raise KeyError(
                f"no group {group!r} was analysed; groups are: "
                f"{listing(groups) or '(none)'}"
            )
        if contrast not in contrasts:
            raise KeyError(
                f"no contrast {contrast!r} was analysed; contrasts are: "
                f"{listing(contrasts) or '(none)'}"
            )
        row = np.flatnonzero(
            (self._summary["group_id"] == group)
            & (self._summary["contrast"] == contrast)
        )[0]
        raise KeyError(
            f"group {group!r} and contrast {contrast!r} were not meta-analysed: "
            f"{self._summary['reason'][row]}"
        )

    def __repr__(self) -> str:
        record = json.loads(self._record)
        settings = record["settings"]
        summary = self._summary
        n = len(summary)
        analysed = summary["status"].tolist().count(_ANALYSED)
        lines = [
            f"MetaAnalysisResult ({settings['method']}, test {settings['test']}): "
            f"{len(record['studies'])} studies; {n} group and contrast "
            f"pair{'' if n == 1 else 's'}, {analysed} analysed"
        ]
        for row in range(n):
            label = f"  {summary['group_id'][row]}, {summary['contrast'][row]}: "
            used = int(summary["n_studies"][row])
            if summary["status"][row] == _ANALYSED:
                features = int(summary["n_features"][row])
                lines.append(f"{label}{used} studies, {features} features")
            else:
                lines.append(f"{label}not analysed ({summary['status'][row]})")
        return "\n".join(lines)


def differential_expression_meta_analysis(
    results: Mapping[str, DifferentialExpressionResult],
    *,
    groups: str | Sequence[str] | None = None,
    contrasts: str | Sequence[str] | None = None,
    method: Literal["reml", "dl", "pm", "fixed"] = "reml",
    test: Literal["z", "knapp_hartung"] = "z",
    level: float = 0.95,
    min_studies: int = 2,
) -> MetaAnalysisResult:
    """Meta-analyse the differential expression of several studies, feature by feature.

    Each study is the result of :func:`~cytepool.differential_expression.limma_voom`
    or :func:`~cytepool.differential_expression.limma_trend` on its own cube, such as
    one cohort, tissue or platform. For every group (cell type) and contrast, the log2
    fold changes of the studies that tested it, with their standard errors (the
    tables' ``log2_fold_change`` and ``standard_error``), are matched by feature id
    and meta-analysed by :func:`meta_analysis`. Groups are matched by group id and
    contrasts by name, so the studies must name them alike, and a contrast should
    mean the same comparison in every study. A feature that some studies did not test,
    because their expression filter left it out or their cube does not have it, is
    analysed over the studies that tested it.

    The standard errors are the moderated ones of limma's empirical Bayes moderation,
    with which the tables' t-statistics are formed; the meta-analysis takes them as
    known, as it does any sampling variance, and so does not use their degrees of
    freedom.

    Args:
        results: The differential-expression result of every study, by study name;
            at least two.
        groups: A group id or a list of them; by default every group of any study,
            in the order in which the studies first list them.
        contrasts: A contrast name or a list of them; by default every contrast of
            any study, in the order in which the studies first list them.
        method: The estimator of the between-study variance, as for
            :func:`meta_analysis`: ``"reml"``, ``"dl"``, ``"pm"`` or ``"fixed"``.
        test: ``"z"`` or ``"knapp_hartung"``, as for :func:`meta_analysis`.
        level: The confidence level of the intervals, in (0, 1).
        min_studies: The fewest studies that must test a feature for it to be
            meta-analysed, from 1 to the number of studies; 2 by default, since one
            study gives no information about the between-study variance. A group and
            contrast that fewer studies test is not analysed. With 1, a feature that
            one study tested has the z-test of that study's log2 fold change against
            its standard error, taken as known, as metafor gives it for one study: its
            p-value is smaller than that of the study's own moderated t-test, whose
            degrees of freedom are finite.

    Returns:
        The table of every group and contrast meta-analysed, a summary of every group
        and contrast, the studies that tested them, and a provenance record.

    Warns:
        RuntimeWarning: If the estimate of tau^2 of a feature did not converge; the
            message names the group, the contrast and the features, which the record
            lists too.

    Raises:
        TypeError: If ``results`` is not a mapping of study names to
            ``DifferentialExpressionResult``, an option has the wrong type, or
            ``groups``, ``contrasts`` or ``min_studies`` is or holds a numpy masked
            array.
        ValueError: If there are fewer than two studies, a group or contrast is
            listed twice, an option is invalid, ``min_studies`` exceeds the number of
            studies, or a study's table has no ``standard_error`` column, a log2 fold
            change that is infinite, or a standard error that is not positive and
            finite (NaN marks a feature the study did not test).
        KeyError: If a group or contrast is in no study's results.
    """
    reject_masked(groups, "groups")
    reject_masked(contrasts, "contrasts")
    studies = _studies(results)
    estimator = checked_choice(method, "method", tuple(_METHODS))
    chosen_test = checked_choice(test, "test", _TESTS)
    confidence = _level(level)
    fewest = _min_studies(min_studies)
    if fewest > len(studies):
        raise ValueError(
            f"min_studies ({fewest}) is larger than the number of studies "
            f"({len(studies)}), so no feature could be meta-analysed"
        )
    group_ids = _chosen(
        "group", groups, [list(result.groups) for result in studies.values()]
    )
    names = _chosen(
        "contrast",
        contrasts,
        [list(result.contrasts.names) for result in studies.values()],
    )
    settings = {
        "method": estimator,
        "test": chosen_test,
        "level": confidence,
        "min_studies": fewest,
        "p_value_adjustment": "BH",
    }
    rows: list[dict[str, Any]] = []
    study_rows: list[dict[str, Any]] = []
    tables: dict[tuple[str, str], Table] = {}
    for group in group_ids:
        for contrast in names:
            tested: dict[str, Table] = {}
            left_out: dict[str, str] = {}
            for name, result in studies.items():
                table = result.tables.get((group, contrast))
                if table is None:
                    left_out[name] = _absence(result, group, contrast)
                else:
                    _check_table(table, name, group, contrast)
                    tested[name] = table
                study_rows.append(
                    {
                        "group_id": group,
                        "contrast": contrast,
                        "study": name,
                        "tested": table is not None,
                        "reason": left_out.get(name, ""),
                        "n_features": 0 if table is None else len(table),
                    }
                )
            row: dict[str, Any] = {
                "group_id": group,
                "contrast": contrast,
                "status": _ANALYSED,
                "reason": "",
                "studies": list(tested),
                "left_out": left_out,
                "n_features": 0,
                "n_features_left_out": 0,
                "not_converged": [],
            }
            if len(tested) < fewest:
                row["status"] = "too_few_studies"
                row["reason"] = (
                    f"{len(tested)} of the {len(studies)} studies tested it, fewer "
                    f"than min_studies ({fewest})"
                )
                rows.append(row)
                continue
            table, left, failed = _meta_analysed_table(
                tested, estimator, chosen_test, confidence, fewest
            )
            row["n_features"], row["n_features_left_out"] = len(table), left
            row["not_converged"] = failed
            if len(table):
                tables[(group, contrast)] = table
            else:
                row["status"] = "no_features"
                row["reason"] = (
                    f"none of the {left} features that the {len(tested)} studies "
                    f"tested was tested by min_studies ({fewest}) of them"
                )
            if failed:
                warnings.warn(
                    f"group {group!r}, contrast {contrast!r}: the estimate of tau^2 "
                    f"did not converge for {len(failed)} feature"
                    f"{'' if len(failed) == 1 else 's'}, {quoted(failed)}; the record "
                    "lists them under not_converged",
                    RuntimeWarning,
                    skip_file_prefixes=(_PACKAGE_DIRECTORY,),
                )
            rows.append(row)
    return MetaAnalysisResult(
        _summary_table(rows),
        Table(
            {
                column: (
                    text_column([row[column] for row in study_rows])
                    if column in ("group_id", "contrast", "study", "reason")
                    else np.array(
                        [row[column] for row in study_rows],
                        dtype=bool if column == "tested" else np.int64,
                    )
                )
                for column in (
                    "group_id",
                    "contrast",
                    "study",
                    "tested",
                    "reason",
                    "n_features",
                )
            }
        ),
        tables,
        {
            "analysis": "differential_expression_meta_analysis",
            "versions": software_versions(),
            "settings": settings,
            "studies": {name: result.provenance for name, result in studies.items()},
            "analyses": rows,
        },
    )


def _meta_analysed_table(
    tested: Mapping[str, Table], method: str, test: str, level: float, fewest: int
) -> tuple[Table, int, list[str]]:
    """The meta-analysis of one group and contrast, the number of features that fewer
    than ``fewest`` studies tested, and the ids of those in the table whose estimate of
    tau^2 did not converge."""
    index: dict[str, int] = {}
    positions = []
    for table in tested.values():
        ids = table["feature_id"].tolist()
        for feature in ids:
            index.setdefault(feature, len(index))
        positions.append(np.array([index[feature] for feature in ids], dtype=np.intp))
    effects = np.full((len(tested), len(index)), np.nan)
    errors = np.full((len(tested), len(index)), np.nan)
    symbols: list[str | None] | None = None
    for row, (table, where) in enumerate(zip(tested.values(), positions, strict=True)):
        effects[row, where] = table["log2_fold_change"]
        errors[row, where] = table["standard_error"]
        if "symbol" in table:
            if symbols is None:
                symbols = [None] * len(index)
            for position, symbol in zip(
                where.tolist(), table["symbol"].tolist(), strict=True
            ):
                if symbols[position] is None:
                    symbols[position] = symbol
    fit = meta_analysis(
        effects,
        errors,
        method=method,  # type: ignore[arg-type]
        test=test,  # type: ignore[arg-type]
        level=level,
    )
    keep = fit.n_studies >= fewest
    ids = list(index)
    columns: dict[str, Any] = {
        "feature_id": text_column([f for f, k in zip(ids, keep, strict=True) if k])
    }
    if symbols is not None:
        # A text column whose missing symbols are None.
        columns["symbol"] = np.array(
            [s for s, k in zip(symbols, keep, strict=True) if k], dtype=object
        )
    columns |= {
        "n_studies": fit.n_studies[keep],
        "log2_fold_change": fit.estimate[keep],
        "standard_error": fit.standard_error[keep],
        "statistic": fit.statistic[keep],
        "p_value": fit.p_value[keep],
        "adjusted_p_value": adjust_p_values(fit.p_value[keep], "bh"),
        "ci_lower": fit.ci_lower[keep],
        "ci_upper": fit.ci_upper[keep],
        "tau_squared": fit.tau_squared[keep],
        "i_squared": fit.i_squared[keep],
        "h_squared": fit.h_squared[keep],
        "q_statistic": fit.q_statistic[keep],
        "q_p_value": fit.q_p_value[keep],
    }
    failed = [ids[i] for i in np.flatnonzero(keep & ~fit.converged)]
    return Table(columns), int(np.count_nonzero(~keep)), failed


def _studies(results: Any) -> dict[str, DifferentialExpressionResult]:
    if not isinstance(results, Mapping):
        raise TypeError(
            "results must be a mapping of study names to DifferentialExpressionResult, "
            f"such as {{'cohort_a': limma_voom(cube_a, design_a), ...}}, got "
            f"{type(results).__name__}"
        )
    for name, result in results.items():
        if not isinstance(name, str):
            raise TypeError(
                f"results must be keyed by study names (str), got {type(name).__name__} "
                f"{name!r}"
            )
        if not isinstance(result, DifferentialExpressionResult):
            raise TypeError(
                f"results[{name!r}] must be a DifferentialExpressionResult, from "
                f"limma_voom or limma_trend, got {type(result).__name__}"
            )
    if len(results) < 2:
        raise ValueError(
            f"a meta-analysis combines at least two studies, got {len(results)}"
        )
    return dict(results)


def _chosen(kind: str, value: Any, listed: list[list[str]]) -> list[str]:
    """The groups or contrasts to analyse: those given, or every one of any study."""
    available = list(dict.fromkeys(name for names in listed for name in names))
    if value is None:
        return available
    if isinstance(value, str):
        chosen: list[Any] = [value]
    elif isinstance(value, Sequence | np.ndarray):
        chosen = list(value)
    else:
        raise TypeError(
            f"{kind}s must be a {kind} name or a list of them, got "
            f"{type(value).__name__}"
        )
    if not chosen:
        raise ValueError(
            f"{kind}s is empty; give at least one {kind}, or None for every {kind}"
        )
    for name in chosen:
        if not isinstance(name, str):
            raise TypeError(
                f"{kind}s must be {kind} names (str), got {type(name).__name__} "
                f"{name!r}"
            )
    repeated = [name for name in dict.fromkeys(chosen) if chosen.count(name) > 1]
    if repeated:
        raise ValueError(f"{kind}s lists {quoted(repeated)} more than once")
    missing = [name for name in chosen if name not in available]
    if missing:
        raise KeyError(
            f"no study analysed the {kind} {quoted(missing)}; the studies' {kind}s "
            f"are: {listing(available) or '(none)'}"
        )
    return chosen


def _absence(result: DifferentialExpressionResult, group: str, contrast: str) -> str:
    """Why a study's result has no table of a group and contrast: its own message."""
    try:
        result.table(group, contrast)
    except KeyError as error:
        return str(error.args[0])
    raise AssertionError("a table that is not in result.tables")  # pragma: no cover


def _check_table(table: Table, study: str, group: str, contrast: str) -> None:
    """Refuse a study's table without the columns the meta-analysis reads, or with a
    log2 fold change or standard error it cannot use; the message names the feature."""
    where = (
        f"the table of study {study!r} for group {group!r} and contrast {contrast!r}"
    )
    missing = [
        column
        for column in ("feature_id", "log2_fold_change", "standard_error")
        if column not in table
    ]
    if missing:
        raise ValueError(
            f"{where} has no column {quoted(missing)}; the meta-analysis reads the "
            "log2 fold changes and standard errors of limma_voom and limma_trend "
            "results"
        )
    effects = np.asarray(table["log2_fold_change"], dtype=np.float64)
    errors = np.asarray(table["standard_error"], dtype=np.float64)
    for values, wrong, expected in (
        (effects, np.isinf(effects), "finite"),
        (
            errors,
            ~np.isnan(errors) & ~((errors > 0.0) & np.isfinite(errors)),
            "positive and finite",
        ),
    ):
        if wrong.any():
            row = int(np.flatnonzero(wrong)[0])
            column = "log2_fold_change" if values is effects else "standard_error"
            raise ValueError(
                f"{where} has the {column} {values[row]} for feature "
                f"{table['feature_id'][row]!r}; it must be {expected}, or NaN for a "
                "feature the study did not test"
            )


def _min_studies(value: Any) -> int:
    # operator.index accepts a 0-d masked integer and ignores its mask.
    reject_masked(value, "min_studies")
    if isinstance(value, bool | np.bool_):
        raise TypeError("min_studies must be a whole number of studies, not bool")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(
            f"min_studies must be a whole number of studies, got {type(value).__name__}"
        ) from None
    if number < 1:
        raise ValueError(f"min_studies must be at least 1, got {number}")
    return number


def _summary_table(rows: list[dict[str, Any]]) -> Table:
    columns: dict[str, Any] = {
        name: text_column([row[name] for row in rows])
        for name in ("group_id", "contrast", "status", "reason")
    }
    columns["n_studies"] = np.array([len(row["studies"]) for row in rows], np.int64)
    for name in ("n_features", "n_features_left_out"):
        columns[name] = np.array([row[name] for row in rows], dtype=np.int64)
    return Table(columns)
