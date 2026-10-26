"""Linear mixed models with random intercepts, fitted to many features by REML or ML.

Every feature (gene) of a samples x features matrix is fitted with the same model,

    y = X beta + Z_1 b_1 + ... + Z_K b_K + e,
    b_k ~ N(0, sigma_k^2 I),    e ~ N(0, sigma^2 W^-1),

where X is the fixed-effects design, Z_k assigns each sample to a level of the k-th
grouping factor (one random intercept per level, as ``(1 | donor)`` in lme4), and W
holds the feature's precision weights (the identity without weights). The grouping
factors may be nested, such as ``(1 | donor/sample)``, or crossed, such as
``(1 | donor) + (1 | batch)``. The variance parameters are estimated by restricted
maximum likelihood (REML; Patterson and Thompson 1971), or by maximum likelihood
(``method="ml"``), and the fixed effects by generalised least squares given them, as
lme4's ``lmer`` with ``REML = TRUE`` (or ``FALSE``) does (Bates et al. 2015). Tests of
contrasts of the fixed effects use Satterthwaite's degrees of freedom, as lmerTest
computes them (Kuznetsova et al. 2017), which variancePartition's ``dream`` uses for
repeated measures in gene expression (Hoffman and Roussos 2021);
:mod:`cytepool.stats.dream` builds dream's precision weights and moderated tests on
these fits.

The REML criterion:
    With theta_k = sigma_k / sigma, the relative standard deviations that lme4
    estimates, and V = W^-1 + sum_k theta_k^2 Z_k Z_k', the fixed effects and sigma^2
    are profiled out: beta(theta) is the generalised least squares estimate,
    r^2(theta) = (y - X beta)' V^-1 (y - X beta), and sigma^2(theta) = r^2 / (n - p).
    The criterion minimised is lme4's profiled REML deviance (Bates et al. 2015,
    section 3.4),

        log det V + log det(X' V^-1 X) + (n - p) (1 + log(2 pi r^2 / (n - p))),

    whose value at the estimate is lme4's ``REMLcrit``. With weights, log det V
    includes -sum(log w), so the criterion, the fixed effects and the variance
    components do not change when every weight is multiplied by the same constant;
    only sigma^2 scales with it. Maximum likelihood minimises the profiled deviance
    log det V + n (1 + log(2 pi r^2 / n)), lme4's ``deviance`` of a fit with ``REML =
    FALSE``, and estimates sigma^2 by r^2 / n.

Computation:
    The criterion, its gradient and its Hessian are computed from weighted cross-
    products of the samples, without forming any n x n matrix. The random effects of
    the grouping factor with the most levels are absorbed in closed form: the
    covariance matrix of the samples of one of its levels is a diagonal matrix plus a
    matrix of rank one. The random effects of the other factors, and the fixed
    effects, enter a dense Cholesky factorisation whose size is their number of
    levels plus the number of coefficients, as in the penalised least squares
    formulation of lme4 (Bates et al. 2015, section 3). With one grouping factor each
    evaluation costs O(levels x coefficients^2). The fixed effects are handled in an
    orthonormal basis of the columns of X, which keeps the cross-products as well
    conditioned as X itself, and the estimates are transformed back.

    The derivatives of the criterion with respect to gamma_k = theta_k^2 are the
    classical ones of REML (Harville 1977): with P = V^-1 - V^-1 X (X' V^-1 X)^-1 X'
    V^-1, they need only Z_k' P Z_l and Z_k' P y; those of maximum likelihood take the
    traces and norms of Z_k' V^-1 Z_l in place of those of Z_k' P Z_l, which the same
    factorisation gives before its fixed-effect pivots. Newton's method minimises the
    criterion over u_k = log(1 + c_k gamma_k), where c_k is the mean total weight of a
    level of factor k: in a balanced one-way design the criterion is convex in u, and
    the boundary gamma_k = 0 is u_k = 0. A component whose gradient points outward at
    the boundary is held there (projected Newton method, Bertsekas 1982), so variance
    components at zero are exactly zero. An indefinite Hessian is made positive definite
    by taking the absolute values of its eigenvalues, a step changes no u_k by more than
    5, and a backtracking line search ensures descent.
    The fit converges when a step moves no u_k by more than 1e-10, as a full
    Newton step or where the full step promises no decrease beyond rounding; as
    Newton's method converges quadratically, the estimates are then as accurate as
    the cross-products allow. Those of the absorbed factor are taken about the means
    of its levels, which keeps them accurate however much the levels differ; those
    of the other factors enter the Cholesky factorisation scaled by their theta, as
    in lme4, and lose about as many digits as theta^2 times their levels' weights
    has. Every fit starts at theta_k = 1 relative to the residual standard deviation
    of an observation of average weight: lme4's default start (theta = 1) when there
    are no weights, and one that keeps the fit independent of the scale of the
    weights. Like lme4's optimiser, Newton's method finds a local minimum: when the
    criterion has several, the fit can stop at one that is not the lowest. When the
    fixed and random effects together reproduce a feature's values, the criterion
    has no minimum (it decreases without bound as sigma^2 goes to zero), so such a
    feature is left unfitted; see :class:`MixedModelFit`.

Fitted values and residual degrees of freedom:
    The random effects are predicted at the estimates by their best linear unbiased
    predictors, b_k = gamma_k Z_k' P y (Henderson 1975), and the fitted values are
    X beta + sum_k Z_k b_k, lme4's ``fitted``. Their hat matrix H, on the scale of the
    weighted values, gives the approximate residual degrees of freedom tr((I - H)'
    (I - H)), variancePartition's ``rdf.merMod``, which ``dream`` moderates on: with
    I - H = W^-1/2 P W^-1/2 and P V P = P,

        tr((I - H)' (I - H)) = n - p - 2 sum_k gamma_k tr(Z_k' P Z_k)
                               + sum_kl gamma_k gamma_l |Z_k' P Z_l|^2,

    which is n - p with every component at zero, and in a balanced one-way design
    q (m - 1) + (q - 1) / (1 + m gamma)^2 for q groups of m samples.

Satterthwaite's degrees of freedom:
    For a contrast c of the fixed effects with estimated variance v = c' C c, where
    C is the covariance matrix of the estimates, the degrees of freedom are
    2 v^2 / (g' A g) (Satterthwaite 1946; Giesbrecht and Burns 1985), where g holds
    the derivatives of v with respect to the variance parameters and A is their
    asymptotic covariance matrix: twice the inverse of the Hessian of the REML
    deviance (the observed information, as lmerTest computes it), or of the deviance
    of maximum likelihood for such a fit. The degrees of
    freedom do not depend on how the variance parameters are parametrised; here
    they are the variances (sigma_1^2, ..., sigma_K^2, sigma^2). lmerTest
    parametrises them by (theta, sigma), in which the deviance is an even function
    of each theta_k, so a component estimated at zero contributes nothing to g' A g;
    such components are left out here in the same way, and with every component at
    zero the degrees of freedom are n - p, those of the fixed-effects model (found
    by probing lmerTest). lmerTest computes the derivatives numerically; here they
    are exact.

Limits of this version:
    Random slopes, Kenward-Roger degrees of freedom (dream's default for 20 samples
    or fewer), F-tests of several contrasts at once and variance partitioning are
    not implemented yet.
    Every feature needs a value for every sample, with a positive weight. Designs
    whose fixed and random effects together span every sample, and so reproduce
    every feature exactly, are refused, although their criterion can have a minimum
    (lme4 fits them).

References:
    Bates D, Maechler M, Bolker B, Walker S (2015). Fitting linear mixed-effects models
    using lme4. Journal of Statistical Software 67(1), 1-48. doi:10.18637/jss.v067.i01

    Bertsekas DP (1982). Projected Newton methods for optimization problems with simple
    constraints. SIAM Journal on Control and Optimization 20, 221-246.
    doi:10.1137/0320018

    Giesbrecht FG, Burns JC (1985). Two-stage analysis based on a mixed model:
    large-sample asymptotic theory and small-sample simulation results. Biometrics 41,
    477-486. doi:10.2307/2530872

    Golub GH, Van Loan CF (2013). Matrix Computations, 4th edition. Johns Hopkins
    University Press, Baltimore.

    Harville DA (1977). Maximum likelihood approaches to variance component estimation
    and to related problems. Journal of the American Statistical Association 72,
    320-338. doi:10.1080/01621459.1977.10480998

    Henderson CR (1975). Best linear unbiased estimation and prediction under a
    selection model. Biometrics 31, 423-447. doi:10.2307/2529430

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Kuznetsova A, Brockhoff PB, Christensen RHB (2017). lmerTest package: tests in
    linear mixed effects models. Journal of Statistical Software 82(13), 1-26.
    doi:10.18637/jss.v082.i13

    Patterson HD, Thompson R (1971). Recovery of inter-block information when block
    sizes are unequal. Biometrika 58, 545-554. doi:10.1093/biomet/58.3.545

    Satterthwaite FE (1946). An approximate distribution of estimates of variance
    components. Biometrics Bulletin 2, 110-114. doi:10.2307/3002019
"""

import math
from collections import namedtuple
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, cast

import numba
import numpy as np
import numpy.typing as npt

from cytepool._jit import jit, thread_limit
from cytepool._linear_algebra import RANK_TOLERANCE, cholesky, column_space, symmetric
from cytepool._masked_arrays import real_array, reject_masked, unmasked_array
from cytepool._summation import observed_means
from cytepool.design.checks import _aliasing, _random_checks
from cytepool.design.design import Design
from cytepool.stats.distributions import t_sf
from cytepool.design._contrast_weights import contrast_matrix
from cytepool.design._design_matrices import is_random_intercept

__all__ = ["FixedEffectTests", "MixedModelFit", "fit_mixed_model", "fixed_effect_tests"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Newton's method; see the module docstring.
_MAX_ITERATIONS = 100
_STEP_TOLERANCE = 1e-10
_MAX_STEP = 5.0
_ARMIJO = 1e-4
_HALVINGS = 50
# Increases of the criterion up to this fraction of its size count as rounding in the
# line search, so that steps at the minimum are accepted.
_ROUNDING = 1e-13
# Eigenvalues of the Hessian are kept at least this fraction of the largest.
_EIGENVALUE_FLOOR = 1e-8
# A feature whose residuals from the fixed effects are at most this fraction of its
# values (in the norm the weights define) is fitted exactly by the fixed effects.
_EXACT_FIT = 1e-12
# A feature whose residuals from the fixed and random effects together are at most
# this fraction of its residuals from the fixed effects is reproduced by them.
_REPRODUCED = 1e-6
_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, eq=False, repr=False)
class MixedModelFit:
    """REML or maximum likelihood fits of one linear mixed model to many features.

    Rows of the per-feature arrays are features, in the order of the columns of
    ``y``. Variance components are in the order of the design's random effects. The
    arrays are read-only. A feature whose criterion has no minimum is not
    fitted: its values are NaN and it is not converged. That is a feature the fixed
    effects fit exactly, such as a constant feature (to within 1e-12 of its
    values), or one the fixed and random effects together reproduce, such as a
    feature constant within every level of a grouping factor (to within 1e-6 of its
    residuals from the fixed effects): the criterion then decreases without bound as
    the residual variance goes to zero, as lme4's fit fails to converge.

    Attributes:
        coefficients: ``(n_features, n_coefficients)`` estimates of the fixed
            effects (lme4's ``fixef``).
        cov_coefficients: ``(n_features, n_coefficients, n_coefficients)``
            covariance matrix of the estimates (lme4's ``vcov``).
        variance_components: ``(n_features, n_components)`` variance of each
            random intercept, sigma_k^2 (lme4's ``VarCorr``).
        residual_variance: ``(n_features,)`` residual variance sigma^2 of an
            observation of weight one (``sigma(fit)^2``); the REML estimate r^2 /
            (n - p), or for a maximum likelihood fit r^2 / n.
        reml_criterion: ``(n_features,)`` REML criterion at the estimates, minus
            twice the restricted log-likelihood (lme4's ``REMLcrit``); NaN for a
            maximum likelihood fit.
        boundary: ``(n_features, n_components)`` True where a variance component is
            estimated at zero, which lme4 reports as a singular fit.
        converged: ``(n_features,)`` False where Newton's method did not converge
            within 100 iterations or could not decrease the criterion further, or
            the feature was not fitted.
        iterations: ``(n_features,)`` Newton iterations of each fit.
        variance_parameter_covariance: ``(n_features, n_components + 1,
            n_components + 1)`` asymptotic covariance matrix of the estimates of the
            variance components and the residual variance, in that order: twice the
            inverse of the Hessian of the REML deviance, or of the deviance of a
            maximum likelihood fit. Rows and columns of
            components at the boundary are zero; NaN where the Hessian is singular.
        cov_coefficients_derivatives: ``(n_features, n_components + 1,
            n_coefficients, n_coefficients)`` derivatives of ``cov_coefficients``
            with respect to each variance component and the residual variance, at
            the estimates.
        coefficient_names: Names of the fixed effects.
        component_names: Names of the grouping factors of the variance components,
            as lme4 names them, such as ``"donor"`` or ``"sample:donor"``.
        design: ``(n_samples, n_coefficients)`` fixed-effects design matrix.
        method: ``"reml"`` or ``"ml"``, the likelihood the fit maximised.
        log_likelihood: ``(n_features,)`` the maximised log-likelihood: the
            restricted one, minus half the REML criterion, for a REML fit, and the
            likelihood itself for a maximum likelihood fit (lme4's ``logLik``).
        fitted_values: ``(n_samples, n_features)`` fitted values X beta + Z b, with
            the random effects b at their best linear unbiased predictions,
            gamma_k Z_k' P y (lme4's ``fitted``, which includes them).
        residual_df: ``(n_features,)`` approximate residual degrees of freedom,
            tr((I - H)' (I - H)) for the matrix H that maps the weighted values to
            the weighted fitted values (variancePartition's ``rdf.merMod``). It is
            n - p when every variance component is zero, and falls towards n minus
            the rank of the fixed and random effects together as they grow.
        mean_weights: ``(n_features,)`` mean of each feature's weights (one without
            weights): ``residual_variance / mean_weights`` is the residual variance
            of an observation of the feature's average weight.
        mean_expression: ``(n_features,)`` unweighted mean of each feature's values
            (limma's ``Amean``, R's ``rowMeans`` to the last bit).
    """

    coefficients: FloatArray
    cov_coefficients: FloatArray
    variance_components: FloatArray
    residual_variance: FloatArray
    reml_criterion: FloatArray
    boundary: BoolArray
    converged: BoolArray
    iterations: IntArray
    variance_parameter_covariance: FloatArray
    cov_coefficients_derivatives: FloatArray
    coefficient_names: tuple[str, ...]
    component_names: tuple[str, ...]
    design: FloatArray
    method: str
    log_likelihood: FloatArray
    fitted_values: FloatArray
    residual_df: FloatArray
    mean_weights: FloatArray
    mean_expression: FloatArray

    def __repr__(self) -> str:
        return (
            f"MixedModelFit(n_features={self.coefficients.shape[0]}, "
            f"n_samples={self.design.shape[0]}, "
            f"coefficient_names={self.coefficient_names!r}, "
            f"component_names={self.component_names!r}, method={self.method!r})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class FixedEffectTests:
    """t-tests of contrasts of the fixed effects of a mixed model fit.

    Rows of the arrays are features and columns are contrasts. The arrays are
    read-only.

    Attributes:
        fit: The mixed model fit.
        contrasts: ``(n_coefficients, n_contrasts)`` weights of each contrast on the
            fit's coefficients.
        contrast_names: Names of the contrasts.
        df_method: How the degrees of freedom were computed: ``"satterthwaite"``.
        estimate: ``(n_features, n_contrasts)`` estimates c' beta.
        standard_error: ``(n_features, n_contrasts)`` standard errors
            (c' C c)^(1/2), with C the covariance matrix of the coefficients.
        t: ``(n_features, n_contrasts)`` t-statistics, the estimates over their
            standard errors.
        df: ``(n_features, n_contrasts)`` Satterthwaite's degrees of freedom; NaN
            where the asymptotic covariance of the variance parameters is not
            defined.
        p_value: ``(n_features, n_contrasts)`` two-sided p-values of ``t`` on ``df``
            degrees of freedom.
    """

    fit: MixedModelFit
    contrasts: FloatArray
    contrast_names: tuple[str, ...]
    df_method: str
    estimate: FloatArray
    standard_error: FloatArray
    t: FloatArray
    df: FloatArray
    p_value: FloatArray

    def __repr__(self) -> str:
        return (
            f"FixedEffectTests(n_features={self.t.shape[0]}, "
            f"contrast_names={self.contrast_names!r}, df_method={self.df_method!r})"
        )


def fit_mixed_model(
    y: npt.ArrayLike,
    design: Design,
    *,
    weights: npt.ArrayLike | None = None,
    method: Literal["reml", "ml"] = "reml",
    n_threads: int | None = None,
) -> MixedModelFit:
    """Fit a linear mixed model with random intercepts to every feature by REML or ML.

    Each column of ``y`` is fitted with the fixed effects and the random intercepts
    of ``design``, as lme4's ``lmer(y ~ ... + (1 | donor), REML = TRUE, weights =
    w)`` fits one feature: see the module docstring for the model, the criterion
    and the algorithm. Test contrasts of the fixed effects with
    :func:`fixed_effect_tests`.

    Args:
        y: ``(n_samples, n_features)`` expression values on a log scale, such as
            the log-CPM values of voom. Every value must be finite: there are no
            missing values in this version.
        design: A ``Design`` whose random effects are random intercepts of one or
            more grouping factors, such as ``random="(1 | donor)"``,
            ``"(1 | donor) + (1 | batch)"`` or ``"(1 | donor/sample)"``. Its fixed
            effects must have full column rank, and every random effect must pass
            ``design.check()``.
        weights: Precision weights, ``(n_samples,)`` for sample weights or
            ``(n_samples, n_features)`` for observation weights (as from voom):
            the residual variance of an observation is sigma^2 / weight. They must
            be finite and positive.
        method: ``"reml"`` for restricted maximum likelihood, lme4's default
            (``REML = TRUE``), or ``"ml"`` for maximum likelihood (``REML = FALSE``),
            which variancePartition's ``voomWithDreamWeights`` fits to find its
            precision weights.
        n_threads: The most threads for fitting features, as for
            ``fit_linear_model``. ``None`` uses numba's setting for the calling
            thread (see ``numba.set_num_threads``). The result does not depend on
            it. The features are fitted in runs on Python threads, in compiled code
            that releases the GIL, so no numba threading layer is involved; with one
            thread they are fitted on the calling thread.

    Returns:
        The fits. They store, for every feature, the derivatives of the covariance
        matrix of the coefficients with respect to each variance parameter:
        ``n_features * (n_components + 1) * n_coefficients**2`` floats, and the
        fitted values, as many floats as ``y``.

    Raises:
        TypeError: If ``design`` is not a ``Design``, an input is not real-valued,
            ``method`` is not a string, ``n_threads`` is not an integer, or an input
            is or holds a numpy masked array.
        ValueError: If shapes disagree, ``y`` is not finite, a weight is not finite
            and positive, ``method`` is unknown, ``n_threads`` is below 1, the
            design has no random effects or one that is not a random intercept, its
            fixed effects are not of full column rank, a random effect fails the
            design checks, or the fixed and random effects together span every
            sample.

    Example:
        >>> import numpy as np
        >>> from cytepool.design import Design
        >>> samples = {"condition": ["ctrl", "stim"] * 4,
        ...            "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"]}
        >>> design = Design(samples, "~ condition", random="(1 | donor)")
        >>> y = np.random.default_rng(1).normal(5.0, 1.0, (8, 3))
        >>> fit = fit_mixed_model(y, design)
        >>> fit.component_names
        ('donor',)
    """
    ml = _maximum_likelihood(method)
    with thread_limit(n_threads):
        parts = _mixed_design(design)
        return _fit(y, weights, ml, parts)


def _fit(
    y: npt.ArrayLike, weights: npt.ArrayLike | None, ml: bool, parts: "_MixedDesign"
) -> MixedModelFit:
    """Fit every feature of ``y`` with checked design parts, as fit_mixed_model does.

    Call it within ``thread_limit``, which sets the threads it uses.
    """
    x, names, codes, n_levels, components = parts
    n_samples, n_coefficients = x.shape
    values = _response_matrix(y, n_samples)
    weight_rows = _positive_weights(weights, values.shape)
    n_features = values.shape[1]

    # The fixed effects in an orthonormal basis of the columns of x, x = q r: the
    # cross-products of q are as well conditioned as x itself. Projecting each
    # feature onto the columns of q first, and fitting what remains, changes no
    # variance estimate (the REML criterion depends on y only through its part
    # orthogonal to the columns of x, and so does the profiled likelihood) and
    # shifts the coefficients by the projection, which is added back.
    q, r = np.linalg.qr(x)
    projection = q.T @ values
    residuals = values - q @ projection
    # A feature that the fixed effects fit exactly keeps residuals of rounding
    # size; they are set to zero, which leaves it unfitted.
    weight_columns = weight_rows.T
    residual_norm = np.sum(weight_columns * residuals**2, axis=0)
    value_norm = np.sum(weight_columns * values**2, axis=0)
    residuals[:, residual_norm <= _EXACT_FIT**2 * value_norm] = 0.0
    absorbed = int(np.argmax(n_levels))
    offsets = np.full(n_levels.size, -1, dtype=np.int64)
    others = [k for k in range(n_levels.size) if k != absorbed]
    offsets[others] = np.cumsum([0, *n_levels[others]])[:-1]
    level_starts = np.cumsum([0, *n_levels])[:-1].astype(np.int64)
    outputs = _empty_outputs(
        n_features, n_coefficients, n_levels.size, int(n_levels.sum())
    )
    arguments = (
        _writable_c_array(residuals.T, np.float64),
        weight_rows,
        _writable_c_array(q, np.float64),
        _writable_c_array(codes, np.int64),
        _writable_c_array(n_levels, np.int64),
        offsets,
        absorbed,
        int(n_levels[others].sum()),
        level_starts,
        ml,
    )
    threads = min(numba.get_num_threads(), n_features)
    if threads > 1:
        # Runs of consecutive features on Python threads, several per thread to
        # balance the work; the compiled fits release the GIL. Each feature is
        # fitted by the same operations whatever its run, so the result does not
        # depend on the number of threads.
        n_runs = min(n_features, 4 * threads)
        bounds = [
            (run * n_features // n_runs, (run + 1) * n_features // n_runs)
            for run in range(n_runs)
        ]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(
                lambda run: _fit_features(*arguments, *run, *outputs), bounds
            ):
                pass
    else:
        _fit_features(*arguments, 0, n_features, *outputs)
    (
        coefficients,
        unscaled_covariance,
        residual_variance,
        ratios,
        criterion,
        derivatives,
        parameter_covariance,
        converged,
        iterations,
        random_effects,
        residual_df,
    ) = outputs

    # Back to the coefficients of x: beta = r^-1 (beta_q + projection).
    inverse = np.linalg.solve(r, np.eye(n_coefficients))
    coefficients = (coefficients + projection.T) @ inverse.T
    covariance = np.einsum(
        "ij,gjk,lk->gil", inverse, unscaled_covariance, inverse, optimize=True
    )
    covariance = covariance * residual_variance[:, None, None]
    derivatives = np.einsum(
        "ij,gcjk,lk->gcil", inverse, derivatives, inverse, optimize=True
    )
    if not ml:
        # log det(x' V^-1 x) = log det(q' V^-1 q) + 2 log |det r|.
        criterion = criterion + 2.0 * float(np.sum(np.log(np.abs(np.diag(r)))))
    # The fitted values X beta + sum_k Z_k b_k, with the predicted random effects.
    fitted = x @ coefficients.T
    for k in range(n_levels.size):
        fitted += random_effects[:, level_starts[k] + codes[k]].T
    mean_weights = np.broadcast_to(np.mean(weight_rows, axis=1), n_features)
    return MixedModelFit(
        coefficients=_frozen(coefficients),
        cov_coefficients=_frozen(symmetric(covariance)),
        variance_components=_frozen(ratios * residual_variance[:, None]),
        residual_variance=_frozen(residual_variance),
        reml_criterion=_frozen(np.full(n_features, np.nan) if ml else criterion),
        boundary=_frozen(ratios == 0.0, np.bool_),
        converged=_frozen(converged, np.bool_),
        iterations=_frozen(iterations, np.int64),
        variance_parameter_covariance=_frozen(parameter_covariance),
        cov_coefficients_derivatives=_frozen(symmetric(derivatives)),
        coefficient_names=names,
        component_names=components,
        design=_frozen(x),
        method="ml" if ml else "reml",
        log_likelihood=_frozen(-0.5 * criterion),
        fitted_values=_frozen(fitted),
        residual_df=_frozen(residual_df),
        mean_weights=_frozen(mean_weights),
        mean_expression=_frozen(observed_means(values)),
    )


def fixed_effect_tests(
    fit: MixedModelFit,
    contrasts: Any = None,
    *,
    names: Sequence[str] | None = None,
    df_method: Literal["satterthwaite"] = "satterthwaite",
) -> FixedEffectTests:
    """t-tests of contrasts of the fixed effects with Satterthwaite's degrees of freedom.

    For a contrast c, the t-statistic is c' beta / (c' C c)^(1/2) and its degrees of
    freedom are Satterthwaite's, as lmerTest's ``contest(fit, L, joint = FALSE)``
    and ``summary`` compute them; see the module docstring. Without contrasts, each
    coefficient is tested, as in lmerTest's coefficient table. Kenward-Roger's
    degrees of freedom, and F-tests of several contrasts at once, are planned.

    Contrasts that name the coefficients they weigh are matched to
    ``fit.coefficient_names`` by name, as by
    :func:`~cytepool.stats.linear_models.contrasts_fit`: a
    :class:`~cytepool.design.Contrasts` (from ``make_contrasts`` or
    ``Design.contrasts``) by its ``columns``, and a pandas DataFrame or Series by
    its index. Weights without names are taken in the order of the coefficients.

    Args:
        fit: A fit from :func:`fit_mixed_model`.
        contrasts: The contrasts, one per column of a ``(n_coefficients,
            n_contrasts)`` matrix: a ``Contrasts``, a pandas DataFrame, a mapping of
            contrast name to weights, a matrix or a single vector. None tests every
            coefficient.
        names: Contrast names, when ``contrasts`` is not a mapping or a
            ``Contrasts``; see ``contrasts_fit``.
        df_method: ``"satterthwaite"``, the only method so far.

    Returns:
        The estimates, standard errors, t-statistics, degrees of freedom and
        p-values of every feature and contrast.

    Raises:
        TypeError: If ``fit`` is not a ``MixedModelFit``, or ``contrasts`` or
            ``names`` is not valid or is or holds a numpy masked array.
        ValueError: If the contrasts do not match the coefficients, are not finite,
            or ``df_method`` is unknown.
    """
    if not isinstance(fit, MixedModelFit):
        raise TypeError(f"fit must be a MixedModelFit, got {type(fit).__name__}")
    if df_method != "satterthwaite":
        raise ValueError(
            f"df_method must be 'satterthwaite', got {df_method!r}; Kenward-Roger's "
            "degrees of freedom are not available yet"
        )
    if contrasts is None:
        reject_masked(names, "names")
        if names is not None:
            raise ValueError("names are given only with contrasts")
        matrix = np.eye(len(fit.coefficient_names))
        contrast_names = fit.coefficient_names
    else:
        matrix, contrast_names = contrast_matrix(
            contrasts, names, fit.coefficient_names
        )
    estimate = fit.coefficients @ matrix
    variance = np.einsum("jc,gjk,kc->gc", matrix, fit.cov_coefficients, matrix)
    gradient = np.einsum(
        "jc,gajk,kc->gca", matrix, fit.cov_coefficients_derivatives, matrix
    )
    spread = np.einsum(
        "gca,gab,gcb->gc", gradient, fit.variance_parameter_covariance, gradient
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(variance)
        t = estimate / standard_error
        df = np.where(spread > 0.0, 2.0 * variance**2 / spread, np.nan)
    p_value = 2.0 * np.asarray(t_sf(np.abs(t), df))
    return FixedEffectTests(
        fit=fit,
        contrasts=_frozen(matrix),
        contrast_names=tuple(contrast_names),
        df_method=df_method,
        estimate=_frozen(estimate),
        standard_error=_frozen(standard_error),
        t=_frozen(t),
        df=_frozen(df),
        p_value=_frozen(p_value),
    )


# Arguments.


class _MixedDesign(NamedTuple):
    """The parts of a mixed model's design that the fits read."""

    x: FloatArray  # (n_samples, n_coefficients) fixed effects, of full column rank
    names: tuple[str, ...]  # the names of the fixed effects
    codes: IntArray  # (n_components, n_samples) level of each sample, from 0
    n_levels: IntArray  # the number of levels of each grouping factor
    components: tuple[str, ...]  # the names of the grouping factors


class _SomeSamples:
    """What the random-effect checks of a design read, for some of its samples."""

    def __init__(self, n_samples: int, random_parts: tuple[Any, ...]) -> None:
        self.n_samples = n_samples
        self._random_parts = random_parts


def _mixed_design(design: Any) -> _MixedDesign:
    """The fixed-effects matrix, its column names, and the random intercepts.

    Returns:
        The design matrix, its column names, the level of each sample for every
        grouping factor (``(n_components, n_samples)``, levels numbered from 0), the
        number of levels of each factor, and the factors' names.
    """
    _random_intercepts(design)
    report = design.check()
    for issue in report.errors:
        if issue.kind.startswith("random_"):
            raise ValueError(f"the random effects cannot be estimated: {issue}")
    if report.aliased:
        raise ValueError(
            "the fixed effects are not of full column rank: column(s) "
            + ", ".join(repr(name) for name in report.aliased)
            + " are linear combinations of the preceding columns; remove them or "
            "reparametrise the design (see design.check())"
        )
    x = np.array(design.matrix, dtype=np.float64, order="C")
    parts = design._random_parts
    codes = np.array([part[1] for part in parts], dtype=np.int64).reshape(
        len(parts), x.shape[0]
    )
    return _spanning_checked(x, tuple(design.columns), codes, design)


def _mixed_design_of_samples(
    design: Design, rows: npt.NDArray[np.intp], x: FloatArray, names: tuple[str, ...]
) -> _MixedDesign:
    """The design parts of some samples of a design, checked as for the design.

    For a fit to a subset of a design's samples, such as those a workflow keeps in
    one group of a cube: the fixed effects are ``x``, rows of the design's matrix
    for those samples restricted to columns of full rank, and the random intercepts
    are the design's, with the levels those samples have, numbered from 0 in order
    of appearance. The random effects are checked on these samples as
    ``design.check()`` checks them on all: a factor with one level left, levels with
    one sample each, levels that the fixed effects separate, or two factors that
    group the samples alike cannot be estimated.

    Args:
        design: A ``Design`` with random intercepts, as fit_mixed_model takes.
        rows: The positions of the samples among the design's, in fitting order.
        x: ``(len(rows), n_coefficients)`` fixed effects of the samples.
        names: The names of the columns of ``x``.

    Raises:
        ValueError: As fit_mixed_model, for these samples; the message says why.
    """
    _random_intercepts(design)
    x = np.array(x, dtype=np.float64, order="C")
    aliasing = _aliasing(x)
    if aliasing.aliased:
        raise ValueError(
            "the fixed effects are not of full column rank: column(s) "
            + ", ".join(repr(names[j]) for j in aliasing.aliased)
            + " are linear combinations of the preceding columns"
        )
    parts = []
    for effect, codes, effects in design._random_parts:
        index: dict[int, int] = {}
        recoded = np.array(
            [index.setdefault(int(code), len(index)) for code in codes[rows]],
            dtype=np.intp,
        )
        parts.append((effect, recoded, replace(effects, matrix=effects.matrix[rows])))
    errors: list[Any] = []
    # The checks read the number of samples and the random effects of a Design.
    _random_checks(
        cast(Design, _SomeSamples(len(rows), tuple(parts))), aliasing, errors
    )
    for issue in errors:
        raise ValueError(f"the random effects cannot be estimated: {issue}")
    codes = np.array([part[1] for part in parts], dtype=np.int64).reshape(
        len(parts), len(rows)
    )
    return _spanning_checked(x, tuple(names), codes, design)


def _random_intercepts(design: Any) -> None:
    """Refuse anything but a ``Design`` whose random effects are random intercepts."""
    reject_masked(design, "design")
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design with random effects, such as Design(samples, "
            f"'~ condition', random='(1 | donor)'); got {type(design).__name__}"
        )
    if not design.random:
        raise ValueError(
            "the design has no random effects, so there is no mixed model to fit; "
            "fit it with fit_linear_model, or build it with random effects such as "
            "random='(1 | donor)'"
        )
    for effect in design.random:
        if not is_random_intercept(effect):
            raise ValueError(
                f"fit_mixed_model fits random intercepts, such as (1 | "
                f"{effect.group}), but the random effect {effect} has random slopes, "
                "which are not available yet"
            )


def _spanning_checked(
    x: FloatArray, names: tuple[str, ...], codes: IntArray, design: Design
) -> _MixedDesign:
    """The parts, after refusing fixed and random effects that span every sample."""
    n_samples = x.shape[0]
    n_levels = codes.max(axis=1) + 1
    columns = [x]
    for row, count in zip(codes, n_levels):
        indicator = np.zeros((n_samples, int(count)))
        indicator[np.arange(n_samples), row] = 1.0
        columns.append(indicator)
    rank = column_space(np.column_stack(columns)).rank
    if rank >= n_samples:
        raise ValueError(
            f"the fixed and random effects together span every sample ({rank} "
            f"dimensions for {n_samples} samples), so they reproduce every feature "
            "exactly; fit_mixed_model does not fit such designs yet: add samples, or "
            "remove a random effect or a covariate"
        )
    components = tuple(effect.group for effect in design.random)
    return _MixedDesign(x, names, codes, n_levels, components)


def _maximum_likelihood(method: Any) -> bool:
    """Whether ``method`` asks for maximum likelihood ("ml") rather than REML."""
    if isinstance(method, str) and method in ("reml", "ml"):
        return method == "ml"
    error = ValueError if isinstance(method, str) else TypeError
    raise error(
        "method must be 'reml' (restricted maximum likelihood, lme4's REML = TRUE) "
        f"or 'ml' (maximum likelihood, REML = FALSE), got {method!r}"
    )


def _response_matrix(y: npt.ArrayLike, n_samples: int) -> FloatArray:
    values = real_array(unmasked_array(y, "y"), "y", boolean=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_features), "
            f"got shape {values.shape}"
        )
    if values.shape[0] != n_samples:
        raise ValueError(
            f"y has {values.shape[0]} rows but the design has {n_samples} samples; "
            "rows of both are samples"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one feature (column)")
    if not np.all(np.isfinite(values)):
        missing = int(np.count_nonzero(np.isnan(values)))
        raise ValueError(
            f"y must be finite, but it has {missing} missing (NaN) and "
            f"{int(np.count_nonzero(np.isinf(values)))} infinite values; a mixed "
            "model is fitted to complete data, so remove the features with missing "
            "values or fit them with fit_linear_model"
        )
    return np.asarray(values, dtype=np.float64)


def _positive_weights(
    weights: npt.ArrayLike | None, shape: tuple[int, int]
) -> FloatArray:
    """The weights as rows of samples, one per feature or one for all."""
    if weights is None:
        return np.ones((1, shape[0]))
    w = real_array(unmasked_array(weights, "weights"), "weights")
    if w.shape == (shape[0],):
        rows = w[np.newaxis, :]
        kind = "sample weights"
    elif w.shape == shape:
        rows = w.T
        kind = "observation weights"
    else:
        raise ValueError(
            f"weights must have shape ({shape[0]},) for sample weights or {shape} "
            f"for observation weights, got {w.shape}"
        )
    with np.errstate(invalid="ignore"):
        invalid = int(np.count_nonzero(~(np.isfinite(w) & (w > 0.0))))
    if invalid:
        raise ValueError(
            f"{kind} must be finite and positive; {invalid} are zero, negative or "
            "not finite. A mixed model is fitted to every sample of a feature: lme4 "
            "gives an infinite REML criterion for a zero weight"
        )
    return _writable_c_array(rows, np.float64)


def _writable_c_array(values: npt.NDArray[Any], dtype: type) -> npt.NDArray[Any]:
    return np.require(values, dtype=dtype, requirements=["C", "W", "A"])


def _frozen(values: Any, dtype: type = np.float64) -> Any:
    array: npt.NDArray[Any] = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


# The fits, compiled.


def _empty_outputs(
    n_features: int, p: int, n_components: int, n_levels: int
) -> tuple[Any, ...]:
    """The outputs of the compiled fits, NaN (or not converged) until written.

    For each feature, in the orthonormal basis q of the fixed effects: the
    coefficients, the unscaled covariance matrix (q' V^-1 q)^-1, the residual
    variance, the variance ratios gamma_k = sigma_k^2 / sigma^2, the criterion (for
    REML with log det(q' V^-1 q) in place of log det(x' V^-1 x)), the derivatives of
    the covariance matrix of the coefficients with respect to each variance
    parameter, the asymptotic covariance matrix of the variance parameters, whether
    the fit converged, its iterations, the predicted random effects of every level
    of every grouping factor, and the approximate residual degrees of freedom.
    """
    parameters = n_components + 1
    return (
        np.full((n_features, p), np.nan),
        np.full((n_features, p, p), np.nan),
        np.full(n_features, np.nan),
        np.full((n_features, n_components), np.nan),
        np.full(n_features, np.nan),
        np.full((n_features, parameters, p, p), np.nan),
        np.full((n_features, parameters, parameters), np.nan),
        np.zeros(n_features, dtype=np.bool_),
        np.zeros(n_features, dtype=np.int64),
        np.full((n_features, n_levels), np.nan),
        np.full(n_features, np.nan),
    )


# Work arrays of one run of fits; see _work_arrays. The routines that take them are
# called only from compiled code, so they are compiled without the wrapper that
# would convert the tuple from Python objects.
_Work = namedtuple(
    "_Work",
    [
        "level_weights",
        "cross",
        "within",
        "column_totals",
        "deviation",
        "sizes",
        "ratio",
        "shrink",
        "gam",
        "scale",
        "chol",
        "border",
        "tmat",
        "absorbed_border",
        "absorbed_scores",
        "absorbed_cross",
        "absorbed_gram",
        "column_norms",
        "trace",
        "squared_scores",
        "frobenius",
        "score_products",
        "tmat_v",
        "absorbed_cross_v",
        "column_norms_v",
        "trace_v",
        "frobenius_v",
        "gradient",
        "hessian",
        "residual_sum",
        "u",
        "trial_u",
        "trial_ratio",
        "current_gradient",
        "current_hessian",
        "step",
        "u_gradient",
        "free",
        "eigenvalues",
        "eigenvectors",
        "sub_hessian",
        "inverse",
        "projected",
        "parameter_hessian",
        "parameter_inverse",
        "parameter_product",
        "jacobian",
    ],
)


@jit(no_cpython_wrapper=True)
def _work_arrays(n_components: int, q_absorbed: int, q_rest: int, p: int) -> Any:
    """Work arrays for the fits of features with this layout.

    The cross-product matrix of a feature has ``m = q_rest + p + 1`` rows: the
    levels of the grouping factors other than the absorbed one, the fixed effects
    and the response. The first ``q_rest + p`` of them are the pivots of a Cholesky
    factorisation; the queries are the levels of the other factors, unscaled, and
    the response.
    """
    m = q_rest + p + 1
    pivots = q_rest + p
    k = n_components
    parameters = n_components + 1
    return _Work(
        np.empty(q_absorbed),  # level_weights: total weight of each absorbed level
        np.empty((q_absorbed, m)),  # cross: weighted sums of rows, by absorbed level
        np.empty((m, m)),  # within: cross-products about the absorbed levels' means
        np.empty(m),  # column_totals: weighted sums of squares of the columns
        np.empty(m),  # deviation: a sample's row minus its level's mean
        np.empty(k),  # sizes: the mean total weight of a level of each factor
        np.empty(k),  # ratio: the variance ratios gamma_k
        np.empty(q_absorbed),  # shrink: 1 / (1 + gamma_a w_j) for each level j
        np.empty((m, m)),  # gam: cross-products under the absorbed covariance
        np.empty(pivots),  # scale: theta for random effects, 1 for fixed effects
        np.empty((pivots, pivots)),  # chol: Cholesky factor of the pivots
        np.empty((pivots, q_rest + 1)),  # border: L^-1 times the queries
        np.empty((q_rest + 1, q_rest + 1)),  # tmat: Z_rest' P Z_rest, Z' P y, y' P y
        np.empty((pivots, q_absorbed)),  # absorbed_border: L^-1 times Z_a queries
        np.empty(q_absorbed),  # absorbed_scores: Z_a' P y
        np.empty((q_absorbed, q_rest)),  # absorbed_cross: Z_a' P Z_rest
        np.empty((pivots, pivots)),  # absorbed_gram: absorbed_border products
        np.empty(q_absorbed),  # column_norms: squared columns of absorbed_border
        np.empty(k),  # trace: tr(Z_k' P Z_k)
        np.empty(k),  # squared_scores: |Z_k' P y|^2
        np.empty((k, k)),  # frobenius: |Z_k' P Z_l|^2
        np.empty((k, k)),  # score_products: (Z_k' P y)' Z_k' P Z_l (Z_l' P y)
        np.empty((q_rest, q_rest)),  # tmat_v: Z_rest' V^-1 Z_rest (maximum likelihood)
        np.empty((q_absorbed, q_rest)),  # absorbed_cross_v: Z_a' V^-1 Z_rest
        np.empty(q_absorbed),  # column_norms_v: the random-effect rows' part
        np.empty(k),  # trace_v: tr(Z_k' V^-1 Z_k)
        np.empty((k, k)),  # frobenius_v: |Z_k' V^-1 Z_l|^2
        np.empty(k),  # gradient of the criterion in gamma
        np.empty((k, k)),  # hessian of the criterion in gamma
        np.empty(1),  # residual_sum: r^2 = y' P y
        np.empty(k),  # u: the parameters, log(1 + c_k gamma_k)
        np.empty(k),  # trial_u
        np.empty(k),  # trial_ratio
        np.empty(k),  # current_gradient: at the current parameters
        np.empty((k, k)),  # current_hessian
        np.empty(k),  # step
        np.empty(k),  # u_gradient
        np.empty(k, dtype=np.int64),  # free: the parameters not held at zero
        np.empty(k),  # eigenvalues
        np.empty((k, k)),  # eigenvectors
        np.empty((k, k)),  # sub_hessian
        np.empty((p, p)),  # inverse: the inverse of the fixed-effects Cholesky block
        np.empty((p, max(q_absorbed, q_rest, 1))),  # projected: inverse' times queries
        np.empty((parameters, parameters)),  # parameter_hessian
        np.empty((parameters, parameters)),  # parameter_inverse
        np.empty((parameters, parameters)),  # parameter_product
        np.empty((parameters, parameters)),  # jacobian
    )


@jit(nogil=True)
def _fit_features(
    residuals: FloatArray,
    weight_rows: FloatArray,
    basis: FloatArray,
    codes: IntArray,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    q_rest: int,
    level_starts: IntArray,
    ml: bool,
    start: int,
    stop: int,
    coefficients: FloatArray,
    unscaled_covariance: FloatArray,
    residual_variance: FloatArray,
    ratios: FloatArray,
    criterion: FloatArray,
    derivatives: FloatArray,
    parameter_covariance: FloatArray,
    converged: BoolArray,
    iterations: IntArray,
    random_effects: FloatArray,
    residual_df: FloatArray,
) -> None:
    """Fit features ``start:stop`` into the outputs, with work arrays of their own.

    Only the rows of these features are written, so runs of features can be fitted
    on several threads at once; the routine releases the GIL.

    Args:
        residuals: ``(n_features, n_samples)`` values, orthogonal to the basis.
        weight_rows: ``(n_features, n_samples)`` precision weights, or one row
            shared by all features.
        basis: ``(n_samples, n_coefficients)`` orthonormal basis of the fixed
            effects.
        codes: ``(n_components, n_samples)`` level of each sample for every grouping
            factor.
        n_levels: Number of levels of each grouping factor.
        offsets: For each grouping factor other than the absorbed one, the position
            of its first level among the levels of those factors; -1 for the
            absorbed one.
        absorbed: The grouping factor whose random effects are absorbed in closed
            form, the one with the most levels.
        q_rest: The number of levels of the other grouping factors.
        level_starts: For each grouping factor, the position of its first level in
            the rows of ``random_effects``, the factors in order.
        ml: Whether to maximise the likelihood rather than the restricted one.
        start, stop: The run of features to fit.
        coefficients, ...: The outputs, from :func:`_empty_outputs`.
    """
    n_samples, p = basis.shape
    n_components = codes.shape[0]
    q_absorbed = n_levels[absorbed]
    work = _work_arrays(n_components, q_absorbed, q_rest, p)
    shared_weights = weight_rows.shape[0] == 1
    ratio = work.ratio
    for g in range(start, stop):
        w = 0 if shared_weights else g
        sum_log_weights, mean_weight = _cross_products(
            residuals[g],
            weight_rows[w],
            basis,
            codes,
            n_levels,
            offsets,
            absorbed,
            work,
        )
        if _reproduced_by_effects(work, q_rest + p):
            # The criterion has no minimum: nothing to estimate.
            continue
        done, count, value = _minimize(
            n_samples,
            p,
            n_levels,
            offsets,
            absorbed,
            q_rest,
            sum_log_weights,
            1.0 / mean_weight,
            ml,
            work,
        )
        if count < 0:
            # The criterion is not defined where it would start.
            continue
        iterations[g] = count
        converged[g] = done
        criterion[g] = value
        for k in range(n_components):
            ratios[g, k] = ratio[k]
        variance = work.residual_sum[0] / (n_samples if ml else n_samples - p)
        residual_variance[g] = variance
        _estimates(
            ratio,
            variance,
            n_samples,
            p,
            n_levels,
            offsets,
            absorbed,
            q_rest,
            ml,
            work,
            coefficients[g],
            unscaled_covariance[g],
            derivatives[g],
            parameter_covariance[g],
        )
        _predictions(
            ratio,
            n_levels,
            offsets,
            absorbed,
            q_rest,
            level_starts,
            work,
            random_effects[g],
        )
        residual_df[g] = _approximate_residual_df(ratio, n_samples, p, work)


@jit(no_cpython_wrapper=True)
def _predictions(
    ratio: FloatArray,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    q_rest: int,
    level_starts: IntArray,
    work: Any,
    out: FloatArray,
) -> None:
    """The predictions of the random effects (BLUPs) at the optimum, into ``out``.

    The best linear unbiased predictor of the effects of factor k is
    sigma_k^2 Z_k' Sigma^-1 (y - X beta) = gamma_k Z_k' P y, with Sigma = sigma^2 V
    (Henderson 1975): the scores Z_k' P y of the optimum's evaluation, times its
    ratio. A factor at the boundary predicts zero.
    """
    for k in range(ratio.size):
        first = level_starts[k]
        if k == absorbed:
            for j in range(n_levels[k]):
                out[first + j] = ratio[k] * work.absorbed_scores[j]
        else:
            for i in range(n_levels[k]):
                out[first + i] = ratio[k] * work.tmat[offsets[k] + i, q_rest]


@jit(no_cpython_wrapper=True)
def _approximate_residual_df(
    ratio: FloatArray, n_samples: int, p: int, work: Any
) -> float:
    """tr((I - H)' (I - H)) for the hat matrix H of the fitted values, weighted.

    H maps the weighted values W^1/2 y to the weighted fitted values W^1/2 (X beta +
    Z b), so that I - H = W^-1/2 P W^-1/2 =: Q. From P V P = P, Q^2 = Q - sum_k
    gamma_k Q Z~_k Z~_k' Q with Z~_k = W^1/2 Z_k; with tr(Q) = n - p - sum_k
    gamma_k tr(Z_k' P Z_k), this gives

        tr(Q^2) = n - p - 2 sum_k gamma_k tr(Z_k' P Z_k)
                  + sum_kl gamma_k gamma_l |Z_k' P Z_l|^2,

    from the traces and norms of the optimum's evaluation. It is n - p when every
    component is at zero, a linear model's residual df.
    """
    total = float(n_samples - p)
    for k in range(ratio.size):
        total -= 2.0 * ratio[k] * work.trace[k]
        for other in range(ratio.size):
            total += ratio[k] * ratio[other] * work.frobenius[k, other]
    return total


@jit(no_cpython_wrapper=True)
def _cross_products(
    values: FloatArray,
    weights: FloatArray,
    basis: FloatArray,
    codes: IntArray,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    work: Any,
) -> tuple[float, float]:
    """The weighted cross-products of one feature's samples.

    Each sample's row is ``e = [z_rest, q, y]``: the indicators of its levels of the
    grouping factors other than the absorbed one, its row of the basis and its value.
    Fills ``work.level_weights[j]``, the total weight of level j of the absorbed
    factor; ``work.cross[j]``, the weighted sum of the rows of its samples;
    ``work.within``, the weighted cross-products of the rows about the weighted mean
    of their level, summed over the levels; ``work.column_totals``, the weighted sum
    of squares of each column; and ``work.sizes``, the mean total weight of a level
    of each factor. The second pass, about the means, gives the cross-products within
    levels without the difference of nearly equal sums that the sums of squares and
    products would leave when the levels differ much more than their samples do.

    Returns:
        The sum of the logarithms of the weights, and their mean.
    """
    n_samples, p = basis.shape
    n_components = codes.shape[0]
    level_weights = work.level_weights
    cross = work.cross
    within = work.within
    totals = work.column_totals
    deviation = work.deviation
    q_absorbed, m = cross.shape
    dense = m - p - 1  # the first column of the basis, after the other levels
    for j in range(q_absorbed):
        level_weights[j] = 0.0
        for t in range(m):
            cross[j, t] = 0.0
    for s in range(m):
        totals[s] = 0.0
        for t in range(m):
            within[s, t] = 0.0
    sum_log = 0.0
    total = 0.0
    for i in range(n_samples):
        w = weights[i]
        sum_log += math.log(w)
        total += w
        j = codes[absorbed, i]
        level_weights[j] += w
        for t in range(p):
            value = basis[i, t]
            cross[j, dense + t] += w * value
            totals[dense + t] += w * value * value
        cross[j, m - 1] += w * values[i]
        totals[m - 1] += w * values[i] * values[i]
        for k in range(n_components):
            if k != absorbed:
                s = offsets[k] + codes[k, i]
                cross[j, s] += w
                totals[s] += w
    for i in range(n_samples):
        w = weights[i]
        j = codes[absorbed, i]
        for t in range(m):
            deviation[t] = -cross[j, t] / level_weights[j]
        for k in range(n_components):
            if k != absorbed:
                deviation[offsets[k] + codes[k, i]] += 1.0
        for t in range(p):
            deviation[dense + t] += basis[i, t]
        deviation[m - 1] += values[i]
        for s in range(m):
            v = w * deviation[s]
            for t in range(s + 1):
                within[s, t] += v * deviation[t]
    for s in range(m):
        for t in range(s):
            within[t, s] = within[s, t]
    for k in range(n_components):
        level_total = 0.0
        if k == absorbed:
            for j in range(q_absorbed):
                level_total += level_weights[j]
        else:
            for j in range(offsets[k], offsets[k] + n_levels[k]):
                level_total += totals[j]
        work.sizes[k] = level_total / n_levels[k]
    return sum_log, total / n_samples


@jit(no_cpython_wrapper=True)
def _reproduced_by_effects(work: Any, pivots: int) -> bool:
    """Whether the fixed and random effects reproduce the feature's values.

    The weighted residual sum of squares of the values after projecting them on
    every column of the fixed and random effects is the limit of r^2 as the
    variances of the random effects grow without bound. When it is zero, the REML
    criterion has no minimum: it decreases without bound as sigma^2 goes to zero. It
    is what remains of the values' cross-product within the levels of the absorbed
    factor once the other columns are eliminated from it, each in turn unless what
    remains of it is at most ``RANK_TOLERANCE`` of its norm (R's ``qr()`` rule, which
    also skips the columns the absorbed factor's levels already span). The values
    are reproduced when what remains of them is at most ``_REPRODUCED`` of their
    norm.
    """
    within = work.within
    schur = work.gam
    totals = work.column_totals
    m = within.shape[0]
    for s in range(m):
        for t in range(s + 1):
            schur[s, t] = within[s, t]
    aliased = RANK_TOLERANCE * RANK_TOLERANCE
    for t in range(pivots):
        d = schur[t, t]
        if not d > aliased * totals[t]:
            continue
        for r in range(t + 1, m):
            factor = schur[r, t] / d
            if factor != 0.0:
                for c in range(t + 1, r + 1):
                    schur[r, c] -= factor * schur[c, t]
    return not schur[m - 1, m - 1] > _REPRODUCED * _REPRODUCED * totals[m - 1]


@jit(no_cpython_wrapper=True)
def _criterion(
    ratio: FloatArray,
    n_samples: int,
    p: int,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    q_rest: int,
    sum_log_weights: float,
    ml: bool,
    work: Any,
) -> float:
    """The profiled REML (or ML) criterion at the variance ratios, and its derivatives.

    Also fills ``work.gradient`` and ``work.hessian`` with the derivatives of the
    criterion with respect to the ratios gamma_k, ``work.residual_sum`` with r^2,
    and the traces, norms and products of Z_k' P Z_l and Z_k' P y that the
    derivatives are made of. With ``ml``, the criterion is the profiled deviance of
    maximum likelihood, log det V + n (1 + log(2 pi r^2 / n)), whose derivatives
    take their traces and norms from Z_k' V^-1 Z_l instead (``work.trace_v`` and
    ``work.frobenius_v``): after the random-effect pivots of the Cholesky
    factorisation, the Schur complement of the queries is their cross-product
    under V^-1, and after the fixed-effect pivots too, under P. Returns infinity
    where the criterion is not defined: where the fixed effects fit the feature
    exactly (r^2 = 0), or the covariance matrix is numerically singular.
    """
    n_components = ratio.size
    level_weights = work.level_weights
    cross = work.cross
    within = work.within
    q_absorbed, m = cross.shape
    pivots = q_rest + p
    last = m - 1
    shrink = work.shrink
    gam = work.gam
    scale = work.scale
    chol = work.chol
    border = work.border
    tmat = work.tmat
    tmat_v = work.tmat_v
    ga = ratio[absorbed]

    # The cross-products under the covariance of the absorbed factor alone: within
    # level j it is W^-1 + ga 11', whose inverse is W - ga w w' / (1 + ga w_j). For
    # the rows of level j that gives their cross-products about the level's weighted
    # mean, plus the mean's outer product times w_j / (1 + ga w_j): a sum of positive
    # semidefinite terms, so no cancellation arises however large ga is.
    log_absorbed = 0.0
    for j in range(q_absorbed):
        shrink[j] = 1.0 / (1.0 + ga * level_weights[j])
        log_absorbed += math.log1p(ga * level_weights[j])
    for s in range(m):
        for t in range(s + 1):
            gam[s, t] = within[s, t]
    for j in range(q_absorbed):
        factor = shrink[j] / level_weights[j]
        for s in range(m):
            v = factor * cross[j, s]
            if v != 0.0:
                for t in range(s + 1):
                    gam[s, t] += v * cross[j, t]
    for s in range(m):
        for t in range(s):
            gam[t, s] = gam[s, t]

    # Pivots: the other random effects scaled by theta, plus the identity, and the
    # fixed effects. Queries: the other random effects unscaled, and y.
    for k in range(n_components):
        if k != absorbed:
            theta = math.sqrt(ratio[k])
            for j in range(offsets[k], offsets[k] + n_levels[k]):
                scale[j] = theta
    for t in range(q_rest, pivots):
        scale[t] = 1.0
    for s in range(pivots):
        for t in range(s + 1):
            chol[s, t] = scale[s] * gam[s, t] * scale[t]
        if s < q_rest:
            chol[s, s] += 1.0
    if not cholesky(chol, pivots, 0.0):
        return np.inf
    log_random = 0.0
    for t in range(q_rest):
        log_random += 2.0 * math.log(chol[t, t])
    log_fixed = 0.0
    for t in range(q_rest, pivots):
        log_fixed += 2.0 * math.log(chol[t, t])
    for s in range(pivots):
        for j in range(q_rest):
            border[s, j] = scale[s] * gam[s, j]
        border[s, q_rest] = scale[s] * gam[s, last]
    _forward_substitution(chol, pivots, border, q_rest + 1)
    # tmat = the queries' cross-products minus border' border: Z_rest' P Z_rest,
    # Z_rest' P y and y' P y. Its random-effect block after the random pivots alone
    # is Z_rest' V^-1 Z_rest, which maximum likelihood needs.
    for i in range(q_rest + 1):
        ci = i if i < q_rest else last
        for j in range(i + 1):
            cj = j if j < q_rest else last
            tmat[i, j] = gam[ci, cj]
    _subtract_cross_products(border, 0, q_rest, tmat, q_rest + 1)
    if ml:
        for i in range(q_rest):
            for j in range(i + 1):
                tmat_v[i, j] = tmat[i, j]
                tmat_v[j, i] = tmat[i, j]
    _subtract_cross_products(border, q_rest, pivots, tmat, q_rest + 1)
    for i in range(q_rest + 1):
        for j in range(i):
            tmat[j, i] = tmat[i, j]
    r2 = tmat[q_rest, q_rest]
    work.residual_sum[0] = r2
    if not r2 > 0.0:
        return np.inf
    if ml:
        df = n_samples
        value = (
            -sum_log_weights
            + log_absorbed
            + log_random
            + df * (1.0 + _LOG_TWO_PI + math.log(r2 / df))
        )
    else:
        df = n_samples - p
        value = (
            -sum_log_weights
            + log_absorbed
            + log_random
            + log_fixed
            + df * (1.0 + _LOG_TWO_PI + math.log(r2 / df))
        )

    # Queries of the absorbed factor's levels, with the same pivots.
    gb = work.absorbed_border
    scores = work.absorbed_scores
    gc = work.absorbed_cross
    gc_v = work.absorbed_cross_v
    for s in range(pivots):
        for j in range(q_absorbed):
            gb[s, j] = scale[s] * shrink[j] * cross[j, s]
    _forward_substitution(chol, pivots, gb, q_absorbed)
    for j in range(q_absorbed):
        scores[j] = shrink[j] * cross[j, last]
        for i in range(q_rest):
            gc[j, i] = shrink[j] * cross[j, i]
    for s in range(pivots):
        if ml and s == q_rest:
            # Z_a' V^-1 Z_rest, before the fixed-effect pivots.
            for j in range(q_absorbed):
                for i in range(q_rest):
                    gc_v[j, i] = gc[j, i]
        y_part = border[s, q_rest]
        for j in range(q_absorbed):
            v = gb[s, j]
            if v != 0.0:
                scores[j] -= v * y_part
                for i in range(q_rest):
                    gc[j, i] -= v * border[s, i]
    if ml and pivots == q_rest:
        for j in range(q_absorbed):
            for i in range(q_rest):
                gc_v[j, i] = gc[j, i]
    norms = work.column_norms
    norms_v = work.column_norms_v
    for j in range(q_absorbed):
        total = 0.0
        for s in range(q_rest):
            total += gb[s, j] * gb[s, j]
        norms_v[j] = total
        for s in range(q_rest, pivots):
            total += gb[s, j] * gb[s, j]
        norms[j] = total
    products = work.absorbed_gram
    for s in range(pivots):
        for t in range(s + 1):
            total = 0.0
            for j in range(q_absorbed):
                total += gb[s, j] * gb[t, j]
            products[s, t] = total

    trace = work.trace
    squared = work.squared_scores
    frobenius = work.frobenius
    score_products = work.score_products
    trace_v = work.trace_v
    frobenius_v = work.frobenius_v
    for k in range(n_components):
        if k == absorbed:
            # Z_a' P Z_a = diag(w_j shrink_j) - gb' gb, never formed; Z_a' V^-1 Z_a
            # takes only the random-effect rows of gb.
            t_total = 0.0
            s_total = 0.0
            f_total = 0.0
            p_total = 0.0
            for j in range(q_absorbed):
                d = level_weights[j] * shrink[j]
                t_total += d - norms[j]
                s_total += scores[j] * scores[j]
                f_total += d * d - 2.0 * d * norms[j]
                p_total += d * scores[j] * scores[j]
            for s in range(pivots):
                f_total += products[s, s] * products[s, s]
                for t in range(s):
                    f_total += 2.0 * products[s, t] * products[s, t]
                projection = 0.0
                for j in range(q_absorbed):
                    projection += gb[s, j] * scores[j]
                p_total -= projection * projection
            trace[k] = t_total
            squared[k] = s_total
            frobenius[k, k] = f_total
            score_products[k, k] = p_total
            if ml:
                t_total = 0.0
                f_total = 0.0
                for j in range(q_absorbed):
                    d = level_weights[j] * shrink[j]
                    t_total += d - norms_v[j]
                    f_total += d * d - 2.0 * d * norms_v[j]
                for s in range(q_rest):
                    f_total += products[s, s] * products[s, s]
                    for t in range(s):
                        f_total += 2.0 * products[s, t] * products[s, t]
                trace_v[k] = t_total
                frobenius_v[k, k] = f_total
            continue
        first = offsets[k]
        stop = first + n_levels[k]
        t_total = 0.0
        s_total = 0.0
        for i in range(first, stop):
            t_total += tmat[i, i]
            s_total += tmat[i, q_rest] * tmat[i, q_rest]
        trace[k] = t_total
        squared[k] = s_total
        if ml:
            t_total = 0.0
            for i in range(first, stop):
                t_total += tmat_v[i, i]
            trace_v[k] = t_total
        for other in range(k + 1):
            if other == absorbed:
                continue
            f_total = 0.0
            p_total = 0.0
            for i in range(first, stop):
                for v in range(offsets[other], offsets[other] + n_levels[other]):
                    f_total += tmat[i, v] * tmat[i, v]
                    p_total += tmat[i, q_rest] * tmat[i, v] * tmat[v, q_rest]
            frobenius[k, other] = f_total
            frobenius[other, k] = f_total
            score_products[k, other] = p_total
            score_products[other, k] = p_total
            if ml:
                f_total = 0.0
                for i in range(first, stop):
                    for v in range(offsets[other], offsets[other] + n_levels[other]):
                        f_total += tmat_v[i, v] * tmat_v[i, v]
                frobenius_v[k, other] = f_total
                frobenius_v[other, k] = f_total
        # The pair with the absorbed factor: Z_a' P Z_k is a block of gc.
        f_total = 0.0
        p_total = 0.0
        for j in range(q_absorbed):
            for i in range(first, stop):
                f_total += gc[j, i] * gc[j, i]
                p_total += scores[j] * gc[j, i] * tmat[i, q_rest]
        frobenius[k, absorbed] = f_total
        frobenius[absorbed, k] = f_total
        score_products[k, absorbed] = p_total
        score_products[absorbed, k] = p_total
        if ml:
            f_total = 0.0
            for j in range(q_absorbed):
                for i in range(first, stop):
                    f_total += gc_v[j, i] * gc_v[j, i]
            frobenius_v[k, absorbed] = f_total
            frobenius_v[absorbed, k] = f_total

    # The traces and norms of the criterion's derivatives: of Z' P Z for REML, of
    # Z' V^-1 Z for maximum likelihood; the scores are of P y in both.
    traces = trace_v if ml else trace
    norms_of_pairs = frobenius_v if ml else frobenius
    for k in range(n_components):
        work.gradient[k] = traces[k] - df * squared[k] / r2
        for other in range(n_components):
            work.hessian[k, other] = -norms_of_pairs[k, other] + df * (
                2.0 * score_products[k, other] / r2
                - squared[k] * squared[other] / (r2 * r2)
            )
    return value


@jit(no_cpython_wrapper=True)
def _subtract_cross_products(
    border: FloatArray, first: int, stop: int, target: FloatArray, size: int
) -> None:
    """Subtract the cross-products of rows ``first:stop`` of ``border`` from ``target``.

    Only the lower triangle of ``target[:size, :size]`` is updated, row by row of
    ``border`` in order.
    """
    for s in range(first, stop):
        for i in range(size):
            v = border[s, i]
            if v != 0.0:
                for j in range(i + 1):
                    target[i, j] -= v * border[s, j]


@jit(no_cpython_wrapper=True)
def _minimize(
    n_samples: int,
    p: int,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    q_rest: int,
    sum_log_weights: float,
    start: float,
    ml: bool,
    work: Any,
) -> tuple[bool, int, float]:
    """Minimise the REML (or ML) criterion over u_k = log(1 + c_k gamma_k) >= 0.

    Projected Newton's method from gamma_k = ``start`` for every k, with a
    backtracking line search; see the module docstring. Every evaluation of the
    criterion is a trial point; the loop decides what to do with each.

    Returns:
        Whether the fit converged, the number of iterations, and the criterion at the
        estimates, which are left in ``work.ratio`` with the work arrays of their
        evaluation; -1 iterations when the criterion is not defined at the start,
        as when the fixed effects fit the feature exactly.
    """
    ratio = work.ratio
    sizes = work.sizes
    k_total = ratio.size
    u = work.u
    trial_u = work.trial_u
    trial_ratio = work.trial_ratio
    for k in range(k_total):
        trial_u[k] = math.log1p(sizes[k] * start)
        trial_ratio[k] = math.expm1(trial_u[k]) / sizes[k]
    value = np.inf
    descent = 0.0
    predicted = 0.0
    alpha = 1.0
    halvings = 0
    iteration = 0
    started = False
    finishing = False
    while True:
        trial = _criterion(
            trial_ratio,
            n_samples,
            p,
            n_levels,
            offsets,
            absorbed,
            q_rest,
            sum_log_weights,
            ml,
            work,
        )
        if finishing:
            # The current point, evaluated again so that the work arrays hold it.
            return False, iteration, trial
        if not started:
            if not math.isfinite(trial):
                return False, -1, trial
            started = True
            change = np.inf
        elif math.isfinite(trial) and (
            trial <= value + _ARMIJO * descent + _ROUNDING * abs(value)
        ):
            iteration += 1
            change = 0.0
            for k in range(k_total):
                change = max(change, abs(trial_u[k] - u[k]))
        else:
            halvings += 1
            if halvings < _HALVINGS:
                alpha *= 0.5
                descent = _trial_point(work, alpha)
                continue
            # No step decreases the criterion: stop at the current point.
            iteration += 1
            finishing = True
            for k in range(k_total):
                trial_u[k] = u[k]
                trial_ratio[k] = ratio[k]
            continue
        # Accept the trial point.
        for k in range(k_total):
            u[k] = trial_u[k]
            ratio[k] = trial_ratio[k]
            work.current_gradient[k] = work.gradient[k]
            for other in range(k_total):
                work.current_hessian[k, other] = work.hessian[k, other]
        value = trial
        if change <= _STEP_TOLERANCE:
            # A step this small converges when it was a full Newton step, or when the
            # full step promised no decrease beyond the criterion's rounding (which
            # then decides the line search); otherwise the fit makes no progress.
            return (
                alpha == 1.0 or predicted <= _ROUNDING * abs(value),
                iteration,
                value,
            )
        if iteration == _MAX_ITERATIONS:
            return False, iteration, value
        _newton_direction(work)
        alpha = 1.0
        halvings = 0
        descent = _trial_point(work, alpha)
        predicted = -descent


@jit(no_cpython_wrapper=True)
def _newton_direction(work: Any) -> None:
    """The projected Newton step at the current point, in ``work.step``.

    The derivatives of the criterion in gamma are turned into derivatives in u,
    with gamma = (e^u - 1) / c and so d gamma / du = e^u / c. A parameter at zero
    whose gradient points outward is held there; the step of the others uses their
    Hessian with the absolute values of its eigenvalues, each at least
    ``_EIGENVALUE_FLOOR`` times the largest, and no parameter moves by more than
    ``_MAX_STEP``.
    """
    u = work.u
    sizes = work.sizes
    gradient = work.current_gradient
    hessian = work.current_hessian
    g_u = work.u_gradient
    step = work.step
    free = work.free
    sub = work.sub_hessian
    values = work.eigenvalues
    vectors = work.eigenvectors
    k_total = u.size
    n_free = np.int64(0)
    for k in range(k_total):
        jacobian = math.exp(u[k]) / sizes[k]
        g_u[k] = gradient[k] * jacobian
        step[k] = 0.0
        if not (u[k] <= 0.0 and g_u[k] > 0.0):
            free[n_free] = k
            n_free += 1
    if n_free == 0:
        return
    for a in range(n_free):
        k = free[a]
        jacobian = math.exp(u[k]) / sizes[k]
        for b in range(n_free):
            other = free[b]
            sub[a, b] = hessian[k, other] * jacobian * math.exp(u[other]) / sizes[other]
        sub[a, a] += gradient[k] * jacobian
    _symmetric_eigen(sub, n_free, values, vectors)
    largest = 0.0
    for a in range(n_free):
        largest = max(largest, abs(values[a]))
    floor = _EIGENVALUE_FLOOR * largest
    if not floor > 0.0:
        floor = 1.0
    for a in range(n_free):
        projection = 0.0
        for b in range(n_free):
            projection += vectors[b, a] * g_u[free[b]]
        projection /= max(abs(values[a]), floor)
        for b in range(n_free):
            step[free[b]] -= vectors[b, a] * projection
    longest = 0.0
    for k in range(k_total):
        longest = max(longest, abs(step[k]))
    if longest > _MAX_STEP:
        for k in range(k_total):
            step[k] *= _MAX_STEP / longest


@jit(no_cpython_wrapper=True)
def _trial_point(work: Any, alpha: float) -> float:
    """Set the trial point u + alpha step, projected onto u >= 0.

    Returns:
        The decrease that the gradient predicts for it (negative).
    """
    u = work.u
    sizes = work.sizes
    descent = 0.0
    for k in range(u.size):
        work.trial_u[k] = max(u[k] + alpha * work.step[k], 0.0)
        work.trial_ratio[k] = math.expm1(work.trial_u[k]) / sizes[k]
        descent += work.u_gradient[k] * (work.trial_u[k] - u[k])
    return descent


@jit(no_cpython_wrapper=True)
def _estimates(
    ratio: FloatArray,
    variance: float,
    n_samples: int,
    p: int,
    n_levels: IntArray,
    offsets: IntArray,
    absorbed: int,
    q_rest: int,
    ml: bool,
    work: Any,
    coefficients: FloatArray,
    unscaled_covariance: FloatArray,
    derivatives: FloatArray,
    parameter_covariance: FloatArray,
) -> None:
    """The estimates at the optimum, from the work arrays of its evaluation.

    Writes the coefficients in the orthonormal basis, (q' V^-1 q)^-1, the derivatives
    of the covariance matrix of the coefficients with respect to (sigma_1^2, ...,
    sigma_K^2, sigma^2), and the asymptotic covariance matrix of these parameters:
    twice the inverse of the Hessian of the REML deviance, or with ``ml`` of the
    deviance of maximum likelihood, whose Hessian takes the norms of Z_k' V^-1 Z_l in
    place of those of Z_k' P Z_l and n in place of n - p.
    """
    n_components = ratio.size
    chol = work.chol
    border = work.border
    inverse = work.inverse
    projected = work.projected
    # The fixed-effects block L_X of the Cholesky factor: L_X L_X' = q' V^-1 q, the
    # coefficients solve L_X' beta = (L^-1 b)_X, and (q' V^-1 q)^-1 = L_X^-T L_X^-1.
    for t in range(p - 1, -1, -1):
        total = border[q_rest + t, q_rest]
        for v in range(t + 1, p):
            total -= chol[q_rest + v, q_rest + t] * coefficients[v]
        coefficients[t] = total / chol[q_rest + t, q_rest + t]
    for c in range(p):
        for t in range(c):
            inverse[t, c] = 0.0
        inverse[c, c] = 1.0 / chol[q_rest + c, q_rest + c]
        for t in range(c + 1, p):
            total = 0.0
            for v in range(c, t):
                total += chol[q_rest + t, q_rest + v] * inverse[v, c]
            inverse[t, c] = -total / chol[q_rest + t, q_rest + t]
    for i in range(p):
        for j in range(i + 1):
            total = 0.0
            for t in range(i, p):
                total += inverse[t, i] * inverse[t, j]
            unscaled_covariance[i, j] = total
            unscaled_covariance[j, i] = total
    # The derivative of C = sigma^2 (q' V^-1 q)^-1 with respect to sigma_k^2 is
    # C_V q' V^-1 Z_k Z_k' V^-1 q C_V = L_X^-T (G_k G_k') L_X^-1, where G_k holds
    # the fixed-effects rows of L^-1 times the queries of factor k. That with respect
    # to sigma^2 follows from C = sum over the parameters of each times the
    # derivative (C is homogeneous of degree one in them).
    residual = derivatives[n_components]
    for i in range(p):
        for j in range(p):
            residual[i, j] = unscaled_covariance[i, j]
    for k in range(n_components):
        if k == absorbed:
            source = work.absorbed_border
            first = 0
        else:
            source = border
            first = offsets[k]
        count = n_levels[k]
        for i in range(p):
            for j in range(count):
                total = 0.0
                for t in range(i, p):
                    total += inverse[t, i] * source[q_rest + t, first + j]
                projected[i, j] = total
        target = derivatives[k]
        for i in range(p):
            for j in range(i + 1):
                total = 0.0
                for v in range(count):
                    total += projected[i, v] * projected[j, v]
                target[i, j] = total
                target[j, i] = total
        for i in range(p):
            for j in range(p):
                residual[i, j] -= ratio[k] * target[i, j]

    # The Hessian of the REML (or ML) deviance in (gamma_k of the free components,
    # sigma^2), twice its inverse, then mapped to (sigma_1^2, ..., sigma_K^2, sigma^2)
    # by the Jacobian of sigma_k^2 = gamma_k sigma^2.
    norms_of_pairs = work.frobenius_v if ml else work.frobenius
    free = work.free
    n_free = np.int64(0)
    for k in range(n_components):
        if ratio[k] > 0.0:
            free[n_free] = k
            n_free += 1
    size = n_free + 1
    hessian = work.parameter_hessian
    covariance = work.parameter_inverse
    jacobian = work.jacobian
    squared_variance = variance * variance
    for a in range(n_free):
        k = free[a]
        for b in range(a + 1):
            other = free[b]
            value = (
                -norms_of_pairs[k, other]
                + 2.0 * work.score_products[k, other] / variance
            )
            hessian[a, b] = value
            hessian[b, a] = value
        hessian[a, n_free] = work.squared_scores[k] / squared_variance
        hessian[n_free, a] = hessian[a, n_free]
    df = n_samples if ml else n_samples - p
    hessian[n_free, n_free] = df / squared_variance
    if not _inverse_positive_definite(
        hessian, size, covariance, work.parameter_product
    ):
        for i in range(n_components + 1):
            for j in range(n_components + 1):
                parameter_covariance[i, j] = np.nan
        return
    for i in range(n_components + 1):
        for a in range(size):
            jacobian[i, a] = 0.0
    for a in range(n_free):
        k = free[a]
        jacobian[k, a] = variance
        jacobian[k, n_free] = ratio[k]
    jacobian[n_components, n_free] = 1.0
    for i in range(n_components + 1):
        for j in range(n_components + 1):
            total = 0.0
            for a in range(size):
                for b in range(size):
                    total += jacobian[i, a] * covariance[a, b] * jacobian[j, b]
            parameter_covariance[i, j] = 2.0 * total


@jit
def _inverse_positive_definite(
    matrix: FloatArray, size: int, inverse: FloatArray, work: FloatArray
) -> bool:
    """The inverse of the symmetric positive definite ``matrix[:size, :size]``.

    By Cholesky: L^-1 is formed in ``work``, then ``inverse[:size, :size]`` =
    L^-T L^-1. ``matrix`` is overwritten. Returns False if the matrix is not
    positive definite.
    """
    if not cholesky(matrix, size, 0.0):
        return False
    for i in range(size):
        for j in range(size):
            work[i, j] = 1.0 if i == j else 0.0
    _forward_substitution(matrix, size, work, size)
    for i in range(size):
        for j in range(size):
            total = 0.0
            for t in range(max(i, j), size):
                total += work[t, i] * work[t, j]
            inverse[i, j] = total
    return True


@jit
def _forward_substitution(
    lower: FloatArray, size: int, rhs: FloatArray, n_rhs: int
) -> None:
    """Solve L X = B in place for the first ``n_rhs`` columns of ``rhs``."""
    for s in range(size):
        for t in range(s):
            factor = lower[s, t]
            if factor != 0.0:
                for j in range(n_rhs):
                    rhs[s, j] -= factor * rhs[t, j]
        inverse = 1.0 / lower[s, s]
        for j in range(n_rhs):
            rhs[s, j] *= inverse


@jit
def _symmetric_eigen(
    matrix: FloatArray, size: int, values: FloatArray, vectors: FloatArray
) -> None:
    """Eigenvalues and eigenvectors of ``matrix[:size, :size]`` by Jacobi rotations.

    The cyclic Jacobi method (Golub and Van Loan 2013, section 8.5) for the few
    variance parameters of a fit. ``matrix`` is overwritten; column a of ``vectors``
    is the eigenvector of ``values[a]``.
    """
    for i in range(size):
        for j in range(size):
            vectors[i, j] = 1.0 if i == j else 0.0
    for _ in range(64):
        off = 0.0
        total = 0.0
        for i in range(size):
            for j in range(size):
                total += matrix[i, j] * matrix[i, j]
                if i != j:
                    off += matrix[i, j] * matrix[i, j]
        if off <= 1e-30 * total:
            break
        for a in range(size - 1):
            for b in range(a + 1, size):
                if matrix[a, b] == 0.0:
                    continue
                angle = (matrix[b, b] - matrix[a, a]) / (2.0 * matrix[a, b])
                tangent = 1.0 / (abs(angle) + math.sqrt(angle * angle + 1.0))
                if angle < 0.0:
                    tangent = -tangent
                cosine = 1.0 / math.sqrt(tangent * tangent + 1.0)
                sine = tangent * cosine
                for k in range(size):
                    first = matrix[k, a]
                    second = matrix[k, b]
                    matrix[k, a] = cosine * first - sine * second
                    matrix[k, b] = sine * first + cosine * second
                for k in range(size):
                    first = matrix[a, k]
                    second = matrix[b, k]
                    matrix[a, k] = cosine * first - sine * second
                    matrix[b, k] = sine * first + cosine * second
                for k in range(size):
                    first = vectors[k, a]
                    second = vectors[k, b]
                    vectors[k, a] = cosine * first - sine * second
                    vectors[k, b] = sine * first + cosine * second
    for a in range(size):
        values[a] = matrix[a, a]
