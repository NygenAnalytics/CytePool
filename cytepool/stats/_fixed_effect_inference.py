"""Degrees of freedom and F-tests of the fixed effects of mixed model fits.

The tests of :mod:`cytepool.stats.mixed_models`, dream's moderated tests
(:mod:`cytepool.stats.dream`) and ``dream_voom`` share:

- ``InferenceParts``, made by ``satterthwaite_parts`` and ``kenward_roger_parts``: per
  feature, the covariance matrix of the fixed effects that the statistics use, the
  unadjusted one, its derivatives with respect to the variance parameters, and the
  covariance matrix of those parameters, from which the degrees of freedom of any
  contrast follow. ``chosen_parts`` takes each feature's parts from one of two sets.
- ``contrast_tests``: the t-tests of contrasts.
- ``f_test``: the F-test that several contrasts are all zero, by Satterthwaite's
  method as lmerTest's ``contestMD`` computes it or by Kenward-Roger's as
  pbkrtest's ``KRmodcomp`` does.
- ``combined_df``: the denominator degrees of freedom of an F-statistic that is the
  mean of the squares of independent t-statistics, from theirs.
- ``contrast_rank``: the number of linearly independent contrasts.

Kenward and Roger (1997):
    The covariance matrix of the observations is linear in the variance parameters
    phi = (sigma_1^2, ..., sigma_K^2, s^2), Sigma = sum_i phi_i G_i, with G_k =
    Z_k Z_k' for the random effects and G = R, a diagonal matrix, for the residuals.
    With Phi the covariance matrix of the fixed effects, P_i = -X' Sigma^-1 G_i
    Sigma^-1 X, Q_ij = X' Sigma^-1 G_i Sigma^-1 G_j Sigma^-1 X and W the inverse of the
    expected information of phi under REML, (1/2) tr(P G_i P G_j) with P = Sigma^-1 -
    Sigma^-1 X Phi X' Sigma^-1, the adjusted covariance matrix is Phi + 2 Lambda with

        Lambda = Phi (sum_ij W_ij (Q_ij - P_i Phi P_j)) Phi

    (their R_ij vanish, the covariance being linear in phi). For a contrast c the
    degrees of freedom are 2 (c' Phi c)^2 / sum_ij W_ij (c' E_i c)(c' E_j c) with E_i =
    -Phi P_i Phi, those of Satterthwaite's formula with the expected information; for
    l contrasts they come with a scaling of the F-statistic (Kenward and Roger 1997,
    section 4; Halekoh and Hojsgaard 2014, section 3).

    pbkrtest builds Sigma for an lme4 fit without its prior weights, R = I with s^2
    the fit's residual variance, while Phi is the fit's own covariance matrix, which
    the weights enter (found by probing pbkrtest 0.5.3). dream divides each feature's
    weights by their mean before it fits, so that s^2 is the residual variance of an
    observation of average weight: ``kenward_roger_parts`` computes this with
    ``average_weight=True``. With ``average_weight=False``, R = diag(w_mean / w), and
    Sigma is the covariance matrix of the weighted model, whose Phi it is: Kenward and
    Roger's method for that model. The two agree when the weights are equal.

    Each feature takes O(n^3) operations for n samples, as in pbkrtest, whose
    ``vcovAdj`` also inverts the n x n covariance matrix Sigma. The fixed effects
    enter in an orthonormal basis of their columns, so that badly scaled covariates do
    not cost digits, and the results are transformed back.

References:
    Fai AH-T, Cornelius PL (1996). Approximate F-tests of multiple degree of freedom
    hypotheses in generalized least squares analyses of unbalanced split-plot
    experiments. Journal of Statistical Computation and Simulation 54, 363-378.
    doi:10.1080/00949659608811740

    Halekoh U, Hojsgaard S (2014). A Kenward-Roger approximation and parametric
    bootstrap methods for tests in linear mixed models: the R package pbkrtest.
    Journal of Statistical Software 59(9), 1-32. doi:10.18637/jss.v059.i09

    Kenward MG, Roger JH (1997). Small sample inference for fixed effects from
    restricted maximum likelihood. Biometrics 53, 983-997. doi:10.2307/2533558

    Kuznetsova A, Brockhoff PB, Christensen RHB (2017). lmerTest package: tests in
    linear mixed effects models. Journal of Statistical Software 82(13), 1-26.
    doi:10.18637/jss.v082.i13
"""

import math
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from .._jit import by_runs, jit
from .._linear_algebra import cholesky, column_space, symmetric
from .distributions import f_sf, t_sf

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# lmerTest's get_Fstat_ddf: degrees of freedom within this of each other are averaged.
_EQUAL_DF = 1e-8


class InferenceParts(NamedTuple):
    """What the tests of contrasts of a mixed model fit compute their results from.

    Rows are features; the variance parameters are the variance components and the
    residual variance, in that order.
    """

    covariance: FloatArray  # (G, p, p) covariance of the fixed effects for the tests
    unadjusted: FloatArray  # (G, p, p) the fit's covariance matrix
    derivatives: FloatArray  # (G, K + 1, p, p) of the covariance, or -Phi P_i Phi
    parameter_covariance: FloatArray  # (G, K + 1, K + 1) of the variance parameters
    method: str  # "satterthwaite" or "kenward_roger"


class FTests(NamedTuple):
    """F-tests that several contrasts are all zero, one per feature."""

    statistic: FloatArray  # (G,) F-statistics
    df1: int  # numerator degrees of freedom: the number of independent contrasts
    df2: FloatArray  # (G,) denominator degrees of freedom
    p_value: FloatArray  # (G,)
    scaling: FloatArray  # (G,) Kenward-Roger's scaling of the statistic (or 1)


def satterthwaite_parts(fit: Any) -> InferenceParts:
    """The parts of Satterthwaite's tests: the fit's covariance matrix, its exact
    derivatives and the inverse of the observed information of the variance
    parameters, from which lmerTest computes them."""
    return InferenceParts(
        fit.cov_coefficients,
        fit.cov_coefficients,
        fit.cov_coefficients_derivatives,
        fit.variance_parameter_covariance,
        "satterthwaite",
    )


def kenward_roger_parts(fit: Any, average_weight: bool) -> InferenceParts:
    """The parts of Kenward-Roger's tests of a REML fit; see the module docstring.

    Args:
        fit: A ``MixedModelFit`` by REML, with its weights and the levels of its
            grouping factors.
        average_weight: Whether the adjustment and the degrees of freedom take every
            observation at its feature's average weight, as pbkrtest and dream do,
            rather than at its own.

    Returns:
        The adjusted covariance matrix of the fixed effects, the fit's, the matrices
        -Phi P_i Phi and the inverse of the expected information, for every feature;
        NaN for a feature that has no fit, or whose expected information is singular.
    """
    x = np.asarray(fit.design, dtype=np.float64)
    n_samples, p = x.shape
    n_features = fit.coefficients.shape[0]
    levels = np.ascontiguousarray(fit.component_levels, dtype=np.int64)
    n_components = levels.shape[0]
    n_levels = (levels.max(axis=1) + 1).astype(np.int64)
    # The fixed effects in an orthonormal basis: x = q r.
    q, r = np.linalg.qr(x)
    inverse = np.linalg.solve(r, np.eye(p))
    covariance = np.ascontiguousarray(
        np.einsum("ij,gjk,lk->gil", r, fit.cov_coefficients, r, optimize=True)
    )
    weights = fit.weights
    if weights is None or average_weight:
        scales = np.ones((1, n_samples))
    elif np.ndim(weights) == 1:
        scales = (np.mean(weights) / np.asarray(weights))[np.newaxis, :]
    else:
        scales = (np.mean(weights, axis=0) / np.asarray(weights)).T
    scales = np.ascontiguousarray(scales, dtype=np.float64)
    # The residual variance of an observation of average weight, the s^2 of both.
    residual = np.ascontiguousarray(
        fit.residual_variance / fit.mean_weights, dtype=np.float64
    )
    components = np.ascontiguousarray(fit.variance_components, dtype=np.float64)
    parameters = n_components + 1
    adjusted = np.full((n_features, p, p), np.nan)
    derivatives = np.full((n_features, parameters, p, p), np.nan)
    parameter_covariance = np.full((n_features, parameters, parameters), np.nan)
    arguments = (
        np.ascontiguousarray(q),
        levels,
        n_levels,
        scales,
        components,
        residual,
        covariance,
    )
    outputs = (adjusted, derivatives, parameter_covariance)
    # Runs of consecutive features on Python threads; the compiled routine releases
    # the GIL, and each feature is computed by the same operations.
    by_runs(
        lambda start, stop: _kenward_roger_features(*arguments, start, stop, *outputs),
        n_features,
    )
    # Back to the coefficients of x: beta = r^-1 beta_q.
    adjusted = np.einsum("ij,gjk,lk->gil", inverse, adjusted, inverse, optimize=True)
    derivatives = np.einsum(
        "ij,gajk,lk->gail", inverse, derivatives, inverse, optimize=True
    )
    return InferenceParts(
        symmetric(adjusted),
        fit.cov_coefficients,
        symmetric(derivatives),
        parameter_covariance,
        "kenward_roger",
    )


def chosen_parts(
    first: InferenceParts, second: InferenceParts, use_second: npt.NDArray[np.bool_]
) -> InferenceParts:
    """Each feature's parts from ``second`` where ``use_second``, else ``first``,
    under the method of ``first``."""
    pick = use_second.reshape(-1, 1, 1)
    return InferenceParts(
        np.where(pick, second.covariance, first.covariance),
        np.where(pick, second.unadjusted, first.unadjusted),
        np.where(pick[..., np.newaxis], second.derivatives, first.derivatives),
        np.where(pick, second.parameter_covariance, first.parameter_covariance),
        first.method,
    )


def contrast_tests(
    coefficients: FloatArray, parts: InferenceParts, matrix: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """t-tests of the contrasts in the columns of ``matrix``.

    Returns:
        The estimates, standard errors, t-statistics, degrees of freedom and two-sided
        p-values, ``(G, n_contrasts)`` each.
    """
    estimate = coefficients @ matrix
    variance = np.einsum("jc,gjk,kc->gc", matrix, parts.covariance, matrix)
    df = contrast_df(parts, matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(variance)
        t = estimate / standard_error
    p_value = 2.0 * np.asarray(t_sf(np.abs(t), df), dtype=np.float64)
    return estimate, standard_error, t, df, p_value


def contrast_df(parts: InferenceParts, matrix: FloatArray) -> FloatArray:
    """The degrees of freedom of the contrasts in the columns of ``matrix``.

    2 v^2 / (g' A g), with v = c' Phi c for the unadjusted Phi, g_i = c' D_i c and A
    the covariance matrix of the variance parameters: Satterthwaite's, or Kenward and
    Roger's for a single contrast. NaN where g' A g is not positive.
    """
    variance = np.einsum("jc,gjk,kc->gc", matrix, parts.unadjusted, matrix)
    gradient = np.einsum("jc,gajk,kc->gca", matrix, parts.derivatives, matrix)
    spread = np.einsum(
        "gca,gab,gcb->gc", gradient, parts.parameter_covariance, gradient
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        df: FloatArray = np.where(spread > 0.0, 2.0 * variance**2 / spread, np.nan)
    return df


def contrast_rank(matrix: FloatArray) -> int:
    """The number of linearly independent contrasts among the columns of ``matrix``,
    by the rank rule of R's ``qr()``."""
    return column_space(matrix).rank


def f_test(
    coefficients: FloatArray,
    parts: InferenceParts,
    matrix: FloatArray,
    rank: int,
) -> FTests:
    """The F-test that the contrasts in the columns of ``matrix`` are all zero.

    The contrasts span ``rank`` dimensions. Each feature's are replaced by the
    ``rank`` eigenvectors of their unadjusted covariance matrix with the largest
    eigenvalues, c_m = L p_m, which span the same hypothesis; an exactly dependent
    contrast adds an eigenvalue of zero.

    - Satterthwaite's method, as lmerTest's ``contestMD``: the estimates of c_m are
      uncorrelated, with variances the eigenvalues d_m; F is the mean of their
      squared t-statistics, and its denominator degrees of freedom combine their
      Satterthwaite degrees of freedom by ``combined_df`` (Fai and Cornelius 1996;
      Kuznetsova et al. 2017).
    - Kenward-Roger's, as pbkrtest's ``KRmodcomp``: the Wald statistic with the
      adjusted covariance matrix, divided by the rank and scaled by lambda, on
      (rank, m) degrees of freedom (Kenward and Roger 1997, section 4). It does not
      depend on how the hypothesis is spanned.
    """
    covariance = np.einsum("jc,gjk,kd->gcd", matrix, parts.unadjusted, matrix)
    values, vectors = _largest_eigenpairs(covariance, rank)
    directions = np.einsum("jc,gcm->gjm", matrix, vectors)
    estimates = np.einsum("gj,gjm->gm", coefficients, directions)
    n_features = coefficients.shape[0]
    if parts.method == "satterthwaite":
        gradient = np.einsum(
            "gjm,gajk,gkm->gma", directions, parts.derivatives, directions
        )
        spread = np.einsum(
            "gma,gab,gmb->gm", gradient, parts.parameter_covariance, gradient
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = np.sum(estimates**2 / values, axis=1) / rank
            df = np.where(spread > 0.0, 2.0 * values**2 / spread, np.nan)
        df2 = combined_df(df)
        scaling = np.ones(n_features)
    else:
        statistic, df2, scaling = _kenward_roger_f(
            estimates, directions, values, parts, rank
        )
    p_value = np.asarray(f_sf(statistic, float(rank), df2), dtype=np.float64)
    return FTests(statistic, rank, df2, p_value, scaling)


def correlation_directions(
    parts: InferenceParts, matrix: FloatArray, covariance: FloatArray, rank: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The independent directions of t-statistics of contrasts, and their df.

    For the t-statistics of the contrasts in the columns of ``matrix``, whose
    estimates have the covariance matrix ``covariance`` (``(G, c, c)``), the
    eigenvectors v_m of their correlation matrix R with the ``rank`` largest
    eigenvalues lambda_m give uncorrelated statistics v_m' t / lambda_m^(1/2), the
    t-statistics of the contrasts L D^-1/2 v_m (D the variances of the estimates),
    as limma's F-statistic t' R^-1 t / rank decomposes. Their degrees of freedom are
    those of these contrasts by the method of ``parts``.

    Returns:
        The eigenvalues ``(G, rank)``, the eigenvectors ``(G, c, rank)`` and the
        degrees of freedom ``(G, rank)``, the largest eigenvalue first.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        correlation = covariance / (scale[:, :, np.newaxis] * scale[:, np.newaxis, :])
    values, vectors = _largest_eigenpairs(correlation, rank)
    with np.errstate(divide="ignore", invalid="ignore"):
        directions = np.einsum(
            "jc,gc,gcm->gjm", matrix, 1.0 / scale, vectors, optimize=True
        )
    variance = np.einsum("gjm,gjk,gkm->gm", directions, parts.unadjusted, directions)
    gradient = np.einsum("gjm,gajk,gkm->gma", directions, parts.derivatives, directions)
    spread = np.einsum(
        "gma,gab,gmb->gm", gradient, parts.parameter_covariance, gradient
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        df = np.where(spread > 0.0, 2.0 * variance**2 / spread, np.nan)
    return values, vectors, df


def _largest_eigenpairs(
    matrices: FloatArray, rank: int
) -> tuple[FloatArray, FloatArray]:
    """The ``rank`` largest eigenvalues of each symmetric matrix, the largest first as
    R's ``eigen`` orders them, and their eigenvectors; NaN for a matrix that is not
    finite, such as that of a feature without a fit."""
    values = np.full((matrices.shape[0], rank), np.nan)
    vectors = np.full((matrices.shape[0], matrices.shape[1], rank), np.nan)
    complete = np.all(np.isfinite(matrices), axis=(1, 2))
    if complete.any():
        found, basis = np.linalg.eigh(matrices[complete])
        values[complete] = found[:, ::-1][:, :rank]
        vectors[complete] = basis[:, :, ::-1][:, :, :rank]
    return values, vectors


def combined_df(df: FloatArray) -> FloatArray:
    """The denominator degrees of freedom of the mean of m squared independent
    t-statistics, from their degrees of freedom ``df`` (``(G, m)``), as lmerTest's
    ``get_Fstat_ddf`` documents them.

    One keeps its own, so that the F-test of one contrast is its t-test, as lmerTest's
    ``contestMD`` gives it also below 2 (found by running it). For several: 2 when any
    is at most 2; their mean when consecutive ones differ by less than 1e-8; otherwise
    2 E / (E - m) with E = sum nu / (nu - 2), which matches the mean of the
    F-statistic (Fai and Cornelius 1996). An infinite df counts 1 in E; all infinite
    give infinity. NaN when any is NaN.
    """
    m = df.shape[1]
    if m == 1:
        own: FloatArray = df[:, 0].copy()
        return own
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isinf(df), 1.0, df / (df - 2.0))
        expected = np.sum(ratio, axis=1)
        combined = 2.0 * expected / (expected - m)
    combined = np.where(expected == m, np.inf, combined)
    with np.errstate(invalid="ignore"):  # infinity minus infinity
        steps = np.abs(np.diff(df, axis=1))
    equal = np.all(steps < _EQUAL_DF, axis=1)
    combined = np.where(equal, np.mean(df, axis=1), combined)
    combined = np.where(np.any(df <= 2.0, axis=1), 2.0, combined)
    result: FloatArray = np.where(np.isnan(df).any(axis=1), np.nan, combined)
    return result


def _kenward_roger_f(
    estimates: FloatArray,
    directions: FloatArray,
    values: FloatArray,
    parts: InferenceParts,
    rank: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Kenward-Roger's F-statistic, denominator df and scaling, per feature.

    With Theta = L' (L Phi L')^-1 L for the unadjusted Phi, A1 = sum_ij W_ij
    tr(Theta E_i) tr(Theta E_j) and A2 = sum_ij W_ij tr(Theta E_i Theta E_j), where the
    rows of L are the directions, whose L Phi L' is diag(values).
    """
    adjusted = np.einsum("gjm,gjk,gkn->gmn", directions, parts.covariance, directions)
    terms = np.einsum("gjm,gajk,gkn->gamn", directions, parts.derivatives, directions)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = terms / np.sqrt(values[:, np.newaxis, :, np.newaxis])
        scaled = scaled / np.sqrt(values[:, np.newaxis, np.newaxis, :])
        traces = np.einsum("gamm->ga", scaled)
        pairs = np.einsum("gamn,gbnm->gab", scaled, scaled)
        w = parts.parameter_covariance
        a1 = np.einsum("gab,ga,gb->g", w, traces, traces)
        a2 = np.einsum("gab,gab->g", w, pairs)
        n = float(rank)
        b = (a1 + 6.0 * a2) / (2.0 * n)
        g = ((n + 1.0) * a1 - (n + 4.0) * a2) / ((n + 2.0) * a2)
        denominator = 3.0 * n + 2.0 * (1.0 - g)
        c1 = g / denominator
        c2 = (n - g) / denominator
        c3 = (n + 2.0 - g) / denominator
        expected = 1.0 / (1.0 - a2 / n)
        variance = (2.0 / n) * (1.0 + c1 * b) / ((1.0 - c2 * b) ** 2 * (1.0 - c3 * b))
        rho = variance / (2.0 * expected**2)
        df2 = 4.0 + (n + 2.0) / (n * rho - 1.0)
        scaling = df2 / (expected * (df2 - 2.0))
        statistic = scaling * _wald(estimates, adjusted) / n
    return statistic, df2, scaling


def _wald(estimates: FloatArray, covariance: FloatArray) -> FloatArray:
    """The Wald statistics e' C^-1 e of the features, NaN where C is singular or not
    finite."""
    statistic = np.full(estimates.shape[0], np.nan)
    complete = np.all(np.isfinite(covariance), axis=(1, 2)) & np.all(
        np.isfinite(estimates), axis=1
    )
    rows = np.flatnonzero(complete)
    try:
        solved = np.linalg.solve(covariance[rows], estimates[rows][..., np.newaxis])
        statistic[rows] = np.sum(estimates[rows] * solved[..., 0], axis=1)
    except np.linalg.LinAlgError:  # one is singular: feature by feature
        for row in rows:
            try:
                solved = np.linalg.solve(covariance[row], estimates[row])
            except np.linalg.LinAlgError:
                continue
            statistic[row] = float(estimates[row] @ solved)
    return statistic


# The adjustment, compiled.


@jit(nogil=True)
def _kenward_roger_features(
    basis: FloatArray,
    codes: IntArray,
    n_levels: IntArray,
    residual_scales: FloatArray,
    components: FloatArray,
    residual_variance: FloatArray,
    covariance: FloatArray,
    start: int,
    stop: int,
    adjusted: FloatArray,
    derivatives: FloatArray,
    parameter_covariance: FloatArray,
) -> None:
    """Kenward-Roger's parts of features ``start:stop``, in the basis.

    Everything is computed with Sigma / s^2 = R + sum_k gamma_k Z_k Z_k' (gamma_k =
    sigma_k^2 / s^2), whose inverse is M, and U = M q: then X' Sigma^-1 G_i Sigma^-1 X
    = U' G_i U / s^4 =: H_i / s^4, the matrices of Q_ij are U' G_i M G_j U / s^6, and
    the expected information is (1/2) (tr(M G_i M G_j) - 2 tr(Phi_V U' G_i M G_j U) +
    tr(Phi_V H_i Phi_V H_j)) / s^4 with Phi_V = Phi / s^2; Lambda is s^2 Phi_V
    (sum_ij W_V,ij (U' G_i M G_j U - H_i Phi_V H_j)) Phi_V for W_V the inverse of the
    information without its factor 1 / s^4. Only the rows of these features are
    written, so runs of features can be computed on several threads at once.

    Args:
        basis: ``(n, p)`` orthonormal basis of the fixed effects.
        codes: ``(K, n)`` level of each sample for every grouping factor.
        n_levels: The number of levels of each grouping factor.
        residual_scales: ``(G, n)`` the diagonal of R, or one row for every feature.
        components: ``(G, K)`` variance components.
        residual_variance: ``(G,)`` s^2.
        covariance: ``(G, p, p)`` Phi in the basis.
        start, stop: The run of features.
        adjusted, derivatives, parameter_covariance: The outputs, NaN until written:
            Phi + 2 Lambda, -Phi P_i Phi = Phi_V H_i Phi_V, and W.
    """
    n, p = basis.shape
    n_components = codes.shape[0]
    m = n_components + 1
    largest = 1
    for k in range(n_components):
        largest = max(largest, n_levels[k])
    v = np.empty((n, n))
    lower_inverse = np.empty((n, n))
    inverse = np.empty((n, n))
    u = np.empty((n, p))
    sums = np.empty((largest, p))
    b = np.empty((m, n, p))
    c = np.empty((m, n, p))
    h = np.empty((m, p, p))
    scaled = np.empty((m, p, p))
    products = np.empty((m, m, p, p))
    level_columns = np.empty((max(n_components, 1), n, largest))
    blocks = np.empty((largest, largest))
    traces = np.empty((m, m))
    information = np.empty((m, m))
    information_inverse = np.empty((m, m))
    elimination = np.empty((m, 2 * m))
    phi = np.empty((p, p))
    inner = np.empty((p, p))
    half = np.empty((p, p))
    shared = residual_scales.shape[0] == 1
    for g in range(start, stop):
        s2 = residual_variance[g]
        if not (s2 > 0.0 and math.isfinite(s2)):
            continue
        usable = True
        for k in range(n_components):
            if not (components[g, k] >= 0.0 and math.isfinite(components[g, k])):
                usable = False
        for a in range(p):
            for e in range(p):
                if not math.isfinite(covariance[g, a, e]):
                    usable = False
        if not usable:
            continue
        r = residual_scales[0] if shared else residual_scales[g]
        # V = R + sum_k gamma_k Z_k Z_k', its Cholesky factor and its inverse M.
        for i in range(n):
            for j in range(i + 1):
                value = r[i] if i == j else 0.0
                for k in range(n_components):
                    if codes[k, i] == codes[k, j]:
                        value += components[g, k] / s2
                v[i, j] = value
        if not cholesky(v, n, 0.0):
            continue
        for column in range(n):
            for row in range(column):
                lower_inverse[row, column] = 0.0
            lower_inverse[column, column] = 1.0 / v[column, column]
            for row in range(column + 1, n):
                total = 0.0
                for t in range(column, row):
                    total += v[row, t] * lower_inverse[t, column]
                lower_inverse[row, column] = -total / v[row, row]
        for i in range(n):
            for j in range(i + 1):
                total = 0.0
                for t in range(i, n):
                    total += lower_inverse[t, i] * lower_inverse[t, j]
                inverse[i, j] = total
                inverse[j, i] = total
        for i in range(n):
            for a in range(p):
                total = 0.0
                for t in range(n):
                    total += inverse[i, t] * basis[t, a]
                u[i, a] = total
        for a in range(p):
            for e in range(p):
                phi[a, e] = covariance[g, a, e] / s2
        # B_i = G_i U and H_i = U' G_i U: level sums for the random effects.
        for k in range(n_components):
            for level in range(n_levels[k]):
                for a in range(p):
                    sums[level, a] = 0.0
            for t in range(n):
                for a in range(p):
                    sums[codes[k, t], a] += u[t, a]
            for t in range(n):
                for a in range(p):
                    b[k, t, a] = sums[codes[k, t], a]
            for a in range(p):
                for e in range(a + 1):
                    total = 0.0
                    for level in range(n_levels[k]):
                        total += sums[level, a] * sums[level, e]
                    h[k, a, e] = total
                    h[k, e, a] = total
        for t in range(n):
            for a in range(p):
                b[n_components, t, a] = r[t] * u[t, a]
        for a in range(p):
            for e in range(a + 1):
                total = 0.0
                for t in range(n):
                    total += r[t] * u[t, a] * u[t, e]
                h[n_components, a, e] = total
                h[n_components, e, a] = total
        # C_i = M B_i, and the products B_i' C_j = U' G_i M G_j U.
        for i in range(m):
            for s in range(n):
                for a in range(p):
                    total = 0.0
                    for t in range(n):
                        total += inverse[s, t] * b[i, t, a]
                    c[i, s, a] = total
        for i in range(m):
            for j in range(m):
                for a in range(p):
                    for e in range(p):
                        total = 0.0
                        for t in range(n):
                            total += b[i, t, a] * c[j, t, e]
                        products[i, j, a, e] = total
        # tr(M G_i M G_j), from M Z_k, the column sums of M by level.
        for k in range(n_components):
            for s in range(n):
                for level in range(n_levels[k]):
                    level_columns[k, s, level] = 0.0
                for t in range(n):
                    level_columns[k, s, codes[k, t]] += inverse[s, t]
        for k in range(n_components):
            for other in range(k + 1):
                for level in range(n_levels[k]):
                    for level_other in range(n_levels[other]):
                        blocks[level, level_other] = 0.0
                for s in range(n):
                    for level_other in range(n_levels[other]):
                        blocks[codes[k, s], level_other] += level_columns[
                            other, s, level_other
                        ]
                total = 0.0
                for level in range(n_levels[k]):
                    for level_other in range(n_levels[other]):
                        total += blocks[level, level_other] ** 2
                traces[k, other] = total
                traces[other, k] = total
            total = 0.0
            for s in range(n):
                squares = 0.0
                for level in range(n_levels[k]):
                    squares += level_columns[k, s, level] ** 2
                total += r[s] * squares
            traces[k, n_components] = total
            traces[n_components, k] = total
        total = 0.0
        for s in range(n):
            for t in range(n):
                total += r[s] * r[t] * inverse[s, t] ** 2
        traces[n_components, n_components] = total
        # The expected information, without its factor 1 / s^4.
        for i in range(m):
            for a in range(p):
                for e in range(p):
                    total = 0.0
                    for f in range(p):
                        total += phi[a, f] * h[i, f, e]
                    scaled[i, a, e] = total
        for i in range(m):
            for j in range(i + 1):
                cross = 0.0
                square = 0.0
                for a in range(p):
                    for e in range(p):
                        cross += phi[a, e] * products[i, j, e, a]
                        square += scaled[i, a, e] * scaled[j, e, a]
                value = 0.5 * (traces[i, j] - 2.0 * cross + square)
                information[i, j] = value
                information[j, i] = value
        if not _invert(information, m, information_inverse, elimination):
            continue
        # Lambda = Phi_V (sum_ij W_ij (B_i' C_j - H_i Phi_V H_j)) Phi_V, times s^2.
        for a in range(p):
            for e in range(p):
                total = 0.0
                for i in range(m):
                    for j in range(m):
                        correction = 0.0
                        for f in range(p):
                            correction += h[i, a, f] * scaled[j, f, e]
                        total += information_inverse[i, j] * (
                            products[i, j, a, e] - correction
                        )
                inner[a, e] = total
        for a in range(p):
            for e in range(p):
                total = 0.0
                for f in range(p):
                    total += phi[a, f] * inner[f, e]
                half[a, e] = total
        for a in range(p):
            for e in range(p):
                total = 0.0
                for f in range(p):
                    total += half[a, f] * phi[f, e]
                adjusted[g, a, e] = s2 * phi[a, e] + 2.0 * s2 * total
        for i in range(m):
            for a in range(p):
                for e in range(p):
                    total = 0.0
                    for f in range(p):
                        total += scaled[i, a, f] * phi[f, e]
                    derivatives[g, i, a, e] = total
        for i in range(m):
            for j in range(m):
                parameter_covariance[g, i, j] = s2 * s2 * information_inverse[i, j]


@jit(nogil=True)
def _invert(
    matrix: FloatArray, size: int, inverse: FloatArray, work: FloatArray
) -> bool:
    """The inverse of the small matrix ``matrix[:size, :size]`` into ``inverse``.

    Gauss-Jordan elimination with partial pivoting, which serves the symmetric
    matrices that need not be positive definite. Returns False if a pivot is zero or
    not finite.
    """
    for i in range(size):
        for j in range(size):
            work[i, j] = matrix[i, j]
            work[i, size + j] = 1.0 if i == j else 0.0
    for column in range(size):
        pivot = column
        for row in range(column + 1, size):
            if abs(work[row, column]) > abs(work[pivot, column]):
                pivot = row
        value = work[pivot, column]
        if not (value != 0.0 and math.isfinite(value)):
            return False
        if pivot != column:
            for j in range(2 * size):
                swap = work[column, j]
                work[column, j] = work[pivot, j]
                work[pivot, j] = swap
        for j in range(2 * size):
            work[column, j] /= value
        for row in range(size):
            if row != column:
                factor = work[row, column]
                if factor != 0.0:
                    for j in range(2 * size):
                        work[row, j] -= factor * work[column, j]
    for i in range(size):
        for j in range(size):
            inverse[i, j] = work[i, size + j]
    return True
