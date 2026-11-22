"""The search for each gene's maximum of a dose-response model: a grid of the
nonlinear parameters at which the linear ones are solved exactly, then a bounded
Levenberg-Marquardt refinement of the nonlinear ones.

Every fit minimises Q(theta) = (ybar - f)' W (ybar - f) over the model's box, with ybar
the K dose-group means and W = R'R their weight matrix, given as its factor R, so that
the residuals R (ybar - f) are whitened. At each point of the grid the linear
parameters minimise Q exactly (variable projection; Golub and Pereyra 1973), within
their constraints: a >= 0 for exponential 2 and 3, and for exponential 4 and 5 the cone
a >= 0 with c = 1 + alpha_2 / a in [0, C_MAX], whose faces c = 0 and c = C_MAX are
fitted when the unconstrained solution lies outside it. The lowest local minima of the
grid start refinements in Kaufman's form (Kaufman 1975), which move the nonlinear
parameters within their bounds and solve the linear ones exactly again at every point
tried: a parameter at a bound is held there while the gradient points out of the box,
and the other nonlinear ones take the damped Gauss-Newton step along their Jacobian
columns N, with the span of the free linear parameters' columns projected out, and
with Marquardt's scaling by the diagonal of N'N, which makes the step independent of
each parameter's unit. The refinement stops when the decrease that a full Gauss-Newton
step predicts, r'N (N'N)^-1 N'r, is at most 1e-12 of Q plus the scale of the
objective, or when no step lowers Q at the largest damping, 1e16, nor any step of one
parameter alone; after 500 iterations it stops unconverged.
The straight line and the restricted quadratic have no nonlinear parameters, so their
linear solve is the fit: the quadratic's coefficients b1 and b2 share a sign, so its
maximum is the best of the unrestricted fit, when its coefficients share a sign, and
the fits with b1 = 0 and with b2 = 0.

Genes run on numba's threads, each with the same operations whatever the threads, so
results do not depend on them.

References:
    Golub GH, Pereyra V (1973). The differentiation of pseudo-inverses and nonlinear
    least squares problems whose variables separate. SIAM Journal on Numerical
    Analysis 10, 413-432. doi:10.1137/0710036

    Kaufman L (1975). A variable projection method for solving separable nonlinear
    least squares problems. BIT 15, 49-57. doi:10.1007/BF01932995

    Levenberg K (1944). A method for the solution of certain non-linear problems in
    least squares. Quarterly of Applied Mathematics 2, 164-168. doi:10.1090/qam/10666

    Marquardt DW (1963). An algorithm for least-squares estimation of nonlinear
    parameters. Journal of the Society for Industrial and Applied Mathematics 11,
    431-441. doi:10.1137/0111030
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._jit import jit, jit_numpy_errors
from cytepool.stats._dose_curve_models import (
    C_MAX,
    CONE,
    DROMICS_EXPONENTIAL,
    GRID_SIZES,
    LINE,
    NON_NEGATIVE,
    POLYNOMIAL,
    SAME_SIGN,
    SETS,
    assemble,
    basis_row,
    is_linear,
    mean_gradient,
    mean_value,
    nonlinear_of,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MAX_ITERATIONS = 500
STARTS = 3  # the lowest local minima of the grid that start a refinement
STOP_DECREASE = 1e-12
_DAMPING_START, _DAMPING_MIN, _DAMPING_MAX = 1e-3, 1e-15, 1e16
_PREDICTION_DAMPING = 1e-9
_DEPENDENT = 1e-12  # a column within this of the span of the earlier ones is dropped


@jit_numpy_errors
def grid_bases(
    kind: int, q: int, points: FloatArray, x: FloatArray
) -> tuple[FloatArray, BoolArray]:
    """The linear columns at every grid point and dose, ``(n_points, K, q)``, and
    whether each point's columns are finite."""
    n_points, k = points.shape[0], x.size
    bases = np.zeros((n_points, k, q))
    usable = np.ones(n_points, dtype=np.bool_)
    row = np.zeros(q)
    for p in range(n_points):
        phi1 = points[p, 1] if points.shape[1] > 1 else 0.0
        for i in range(k):
            basis_row(kind, points[p, 0], phi1, x[i], row)
            for j in range(q):
                if not math.isfinite(row[j]):
                    usable[p] = False
                bases[p, i, j] = row[j]
    return bases, usable


@jit_numpy_errors
def _whiten(root: FloatArray, columns: FloatArray, out: FloatArray) -> None:
    """out = R columns for the lower-triangular factor R; a vector v is whitened as
    the one column ``v.reshape((k, 1))``."""
    k, q = columns.shape
    for i in range(k):
        for j in range(q):
            total = 0.0
            for m in range(i + 1):
                total += root[i, m] * columns[m, j]
            out[i, j] = total


@jit_numpy_errors
def _solve_normal_equations(
    gram: FloatArray, rhs: FloatArray, size: int, out: FloatArray
) -> None:
    """The solution of gram out = rhs, of ``size`` unknowns, by the Cholesky factor of
    the symmetric gram matrix, of which the lower triangle is read; a column within
    1e-12 of the span of the columns before it, or of zeros, is dropped and gets 0."""
    factor = np.zeros((size, size))
    kept = np.zeros(size, dtype=np.bool_)
    for j in range(size):
        d = gram[j, j]
        for m in range(j):
            d -= factor[j, m] * factor[j, m]
        if gram[j, j] <= 0.0 or d <= _DEPENDENT * gram[j, j]:
            continue
        kept[j] = True
        factor[j, j] = math.sqrt(d)
        for i in range(j + 1, size):
            total = gram[i, j]
            for m in range(j):
                total -= factor[i, m] * factor[j, m]
            factor[i, j] = total / factor[j, j]
    w = np.zeros(size)
    for j in range(size):
        if kept[j]:
            total = rhs[j]
            for m in range(j):
                total -= factor[j, m] * w[m]
            w[j] = total / factor[j, j]
    for j in range(size - 1, -1, -1):
        if kept[j]:
            total = w[j]
            for m in range(j + 1, size):
                total -= factor[m, j] * out[m]
            out[j] = total / factor[j, j]
        else:
            out[j] = 0.0


@jit_numpy_errors
def _least_squares(
    columns: FloatArray, z: FloatArray, q: int, alpha: FloatArray
) -> None:
    """The least-squares coefficients of z on the first q columns, from the normal
    equations; a column within 1e-12 of the span of the columns before it, or of
    zeros, gets the coefficient 0."""
    k = z.size
    gram = np.zeros((q, q))
    rhs = np.zeros(q)
    for a in range(q):
        for i in range(k):
            rhs[a] += columns[i, a] * z[i]
        for b in range(a + 1):
            total = 0.0
            for i in range(k):
                total += columns[i, a] * columns[i, b]
            gram[a, b] = total
    _solve_normal_equations(gram, rhs, q, alpha)


@jit_numpy_errors
def _sum_squares(
    columns: FloatArray, z: FloatArray, q: int, alpha: FloatArray
) -> float:
    """||z - columns alpha||^2."""
    total = 0.0
    for i in range(z.size):
        r = z[i]
        for j in range(q):
            r -= columns[i, j] * alpha[j]
        total += r * r
    return total


@jit_numpy_errors
def _ray_fit(
    columns: FloatArray, z: FloatArray, slope: float, alpha: FloatArray
) -> float:
    """The best alpha = (a, slope a) with a >= 0, and its sum of squares."""
    uu, uz = 0.0, 0.0
    for i in range(z.size):
        u = columns[i, 0] + slope * columns[i, 1]
        uu += u * u
        uz += u * z[i]
    a = uz / uu if uu > 0.0 and uz > 0.0 else 0.0
    alpha[0], alpha[1] = a, slope * a
    return _sum_squares(columns, z, 2, alpha)


@jit_numpy_errors
def point_fit(
    constraint: int, columns: FloatArray, z: FloatArray, q: int, alpha: FloatArray
) -> float:
    """The linear parameters at one point of the nonlinear ones that minimise
    ||z - columns alpha||^2 within the kind's constraint, written to ``alpha``; returns
    the minimum."""
    if constraint == SAME_SIGN:
        return _same_sign_fit(columns, z, alpha)
    if constraint == NON_NEGATIVE:
        uu, uz = 0.0, 0.0
        for i in range(z.size):
            uu += columns[i, 0] * columns[i, 0]
            uz += columns[i, 0] * z[i]
        alpha[0] = uz / uu if uu > 0.0 and uz > 0.0 else 0.0
        return _sum_squares(columns, z, 1, alpha)
    _least_squares(columns, z, q, alpha)
    if constraint == CONE:
        a, rest = alpha[0], alpha[1]
        if not (a >= 0.0 and -a <= rest <= (C_MAX - 1.0) * a):
            # Outside the cone, the minimum lies on one of its faces, c = 0 or 20.
            other = np.zeros(2)
            low = _ray_fit(columns, z, -1.0, alpha)
            high = _ray_fit(columns, z, C_MAX - 1.0, other)
            if high < low:
                alpha[0], alpha[1] = other[0], other[1]
                return high
            return low
    return _sum_squares(columns, z, q, alpha)


@jit_numpy_errors
def _same_sign_fit(columns: FloatArray, z: FloatArray, alpha: FloatArray) -> float:
    """The restricted quadratic: the least squares of z on three columns with the last
    two coefficients of one sign, written to ``alpha``. The best of the unrestricted
    fit, when feasible, and the fits with the third and with the second coefficient at
    0; returns the minimum."""
    free = np.zeros(3)
    _least_squares(columns, z, 3, free)
    best = math.inf
    if free[1] * free[2] >= 0.0:
        best = _sum_squares(columns, z, 3, free)
        alpha[0], alpha[1], alpha[2] = free[0], free[1], free[2]
    pair = np.zeros((z.size, 2))
    two = np.zeros(2)
    for dropped in (2, 1):
        kept = 1 if dropped == 2 else 2
        for i in range(z.size):
            pair[i, 0] = columns[i, 0]
            pair[i, 1] = columns[i, kept]
        _least_squares(pair, z, 2, two)
        value = _sum_squares(pair, z, 2, two)
        if value < best:
            best = value
            alpha[0], alpha[kept], alpha[dropped] = two[0], two[1], 0.0
    return best


@jit_numpy_errors
def _residuals(
    kind: int,
    theta: FloatArray,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    fitted: FloatArray,
    raw: FloatArray,
    out: FloatArray,
) -> float:
    """The whitened residuals R (ybar - f) at theta, written to ``out``; returns
    their sum of squares, infinite when a mean is not finite."""
    k = x.size
    for i in range(k):
        fitted[i] = mean_value(kind, theta, x[i])
        raw[i] = ybar[i] - fitted[i]
    _whiten(root, raw.reshape((k, 1)), out.reshape((k, 1)))
    total = 0.0
    for i in range(k):
        total += out[i] * out[i]
    if not math.isfinite(total):
        return math.inf
    return total


@jit_numpy_errors
def _jacobian(
    kind: int,
    theta: FloatArray,
    x: FloatArray,
    root: FloatArray,
    gradient: FloatArray,
    raw: FloatArray,
    out: FloatArray,
) -> None:
    """The Jacobian of the whitened residuals, -R df/dtheta, ``(K, p)``."""
    k, p = out.shape
    for i in range(k):
        mean_gradient(kind, theta, x[i], gradient)
        for j in range(p):
            raw[i, j] = gradient[j]
    _whiten(root, raw, out)
    for i in range(k):
        for j in range(p):
            out[i, j] = -out[i, j]


@jit_numpy_errors
def _solve_damped(
    jac: FloatArray,
    r: FloatArray,
    free: BoolArray,
    damping: float,
    step: FloatArray,
) -> None:
    """The free parameters' step of (J'J + damping diag(J'J)) step = -J'r; the
    others 0."""
    k, p = jac.shape
    index = np.zeros(p, dtype=np.int64)
    m = 0
    for j in range(p):
        step[j] = 0.0
        if free[j]:
            index[m] = j
            m += 1
    gram = np.zeros((m, m))
    rhs = np.zeros(m)
    for a in range(m):
        ja = index[a]
        for i in range(k):
            rhs[a] -= jac[i, ja] * r[i]
        for b in range(a + 1):
            jb = index[b]
            total = 0.0
            for i in range(k):
                total += jac[i, ja] * jac[i, jb]
            gram[a, b] = total
        gram[a, a] *= 1.0 + damping
    # The damped matrix is positive definite while each free column of J is nonzero;
    # a pivot that is not positive leaves its parameter in place.
    solution = np.zeros(m)
    _solve_normal_equations(gram, rhs, m, solution)
    for a in range(m):
        step[index[a]] = solution[a]


@jit_numpy_errors
def _resolve_linear(
    kind: int,
    linear: int,
    q: int,
    theta: FloatArray,
    x: FloatArray,
    z: FloatArray,
    root: FloatArray,
    columns: FloatArray,
    whitened: FloatArray,
    alpha: FloatArray,
    row: FloatArray,
) -> float:
    """Solve the linear parameters exactly at theta's nonlinear ones, within their
    constraints, writing them into theta; returns the minimum of Q there."""
    phi0, phi1 = nonlinear_of(kind, theta)
    for i in range(x.size):
        basis_row(kind, phi0, phi1, x[i], row)
        for j in range(q):
            columns[i, j] = row[j]
    _whiten(root, columns, whitened)
    value = point_fit(linear, whitened, z, q, alpha)
    assemble(kind, phi0, phi1, alpha, theta)
    return value


@jit_numpy_errors
def _projected_columns(
    jac: FloatArray,
    linear_free: BoolArray,
    moving: BoolArray,
    out: FloatArray,
) -> None:
    """The Jacobian columns of the moving nonlinear parameters with their projection
    on the span of the free linear parameters' columns removed (Kaufman's step of
    variable projection); the other columns 0."""
    k, p = jac.shape
    n_linear = 0
    for j in range(p):
        if linear_free[j]:
            n_linear += 1
    span = np.zeros((k, n_linear))
    c = 0
    for j in range(p):
        if linear_free[j]:
            for i in range(k):
                span[i, c] = jac[i, j]
            c += 1
    coefficients = np.zeros(n_linear)
    column = np.zeros(k)
    for j in range(p):
        if not moving[j]:
            for i in range(k):
                out[i, j] = 0.0
            continue
        for i in range(k):
            column[i] = jac[i, j]
        _least_squares(span, column, n_linear, coefficients)
        for i in range(k):
            total = column[i]
            for m in range(n_linear):
                total -= span[i, m] * coefficients[m]
            out[i, j] = total


@jit_numpy_errors
def _single_steps(
    kind: int,
    linear: int,
    q: int,
    theta: FloatArray,
    trial: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    projected: FloatArray,
    r: FloatArray,
    moving: BoolArray,
    single: BoolArray,
    step: FloatArray,
    x: FloatArray,
    ybar: FloatArray,
    z: FloatArray,
    root: FloatArray,
    columns: FloatArray,
    whitened: FloatArray,
    alpha: FloatArray,
    row: FloatArray,
    fitted: FloatArray,
    raw: FloatArray,
    r_trial: FloatArray,
    value: float,
) -> float:
    """Damped Gauss-Newton steps of each moving parameter alone, the others held:
    the first that lowers Q is written to ``trial``; returns its Q, or ``value``
    when none does."""
    p = theta.size
    for j in range(p):
        if not moving[j]:
            continue
        for m in range(p):
            single[m] = m == j
        damping = _DAMPING_START
        while damping <= _DAMPING_MAX:
            _solve_damped(projected, r, single, damping, step)
            for m in range(p):
                trial[m] = theta[m]
            trial[j] = min(max(theta[j] + step[j], lower[j]), upper[j])
            if trial[j] != theta[j]:
                _resolve_linear(
                    kind, linear, q, trial, x, z, root, columns, whitened, alpha, row
                )
                new_value = _residuals(kind, trial, x, ybar, root, fitted, raw, r_trial)
                if new_value < value:
                    return new_value
            damping *= 10.0
    return value


@jit_numpy_errors
def refine(
    kind: int,
    linear: int,
    q: int,
    theta: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    x: FloatArray,
    ybar: FloatArray,
    z: FloatArray,
    root: FloatArray,
    scale: float,
    max_iterations: int,
) -> tuple[float, bool]:
    """Refine theta in place within [lower, upper]; returns the minimum of Q and
    whether the refinement converged within ``max_iterations``. ``z`` is R ybar, the
    whitened means.

    Variable projection (Golub and Pereyra 1973) in Kaufman's form (Kaufman 1975):
    the linear parameters are solved exactly, within their constraints (``linear``),
    at every nonlinear point tried, and the nonlinear ones take damped Gauss-Newton
    steps along the Jacobian columns with the span of the free linear columns
    projected out, so that a step accounts for how the levels follow the location,
    width or rate."""
    k, p = x.size, theta.size
    fitted, raw = np.zeros(k), np.zeros(k)
    r, r_trial = np.zeros(k), np.zeros(k)
    jac, projected = np.zeros((k, p)), np.zeros((k, p))
    raw_jac = np.zeros((k, p))
    gradient = np.zeros(p)
    step, trial = np.zeros(p), np.zeros(p)
    linear_free = np.zeros(p, dtype=np.bool_)
    moving, single = np.zeros(p, dtype=np.bool_), np.zeros(p, dtype=np.bool_)
    columns, whitened = np.zeros((k, q)), np.zeros((k, q))
    alpha, row = np.zeros(q), np.zeros(q)
    _resolve_linear(kind, linear, q, theta, x, z, root, columns, whitened, alpha, row)
    value = _residuals(kind, theta, x, ybar, root, fitted, raw, r)
    if not math.isfinite(value):
        return value, False
    damping = _DAMPING_START
    for _ in range(max_iterations):
        _jacobian(kind, theta, x, root, gradient, raw_jac, jac)
        # Hold a parameter at a bound while the descent direction -J'r points out.
        any_moving = False
        for j in range(p):
            g = 0.0
            norm = 0.0
            for i in range(k):
                g += jac[i, j] * r[i]
                norm += jac[i, j] * jac[i, j]
            outward = (theta[j] <= lower[j] and g > 0.0) or (
                theta[j] >= upper[j] and g < 0.0
            )
            free = norm > 0.0 and not outward
            linear_free[j] = free and is_linear(kind, j)
            moving[j] = free and not is_linear(kind, j)
            any_moving = any_moving or moving[j]
        if not any_moving:
            return value, True
        _projected_columns(jac, linear_free, moving, projected)
        # The decrease that a Gauss-Newton step predicts, r'N (N'N)^-1 N'r: when it
        # is negligible, so is what any step can gain. A damping of 1e-9 keeps the
        # direction of a flat valley, whose columns are nearly dependent, which the
        # undamped solve would drop.
        _solve_damped(projected, r, moving, _PREDICTION_DAMPING, step)
        predicted = 0.0
        for j in range(p):
            if moving[j]:
                g = 0.0
                for i in range(k):
                    g += projected[i, j] * r[i]
                predicted -= step[j] * g
        if predicted <= STOP_DECREASE * (value + scale):
            return value, True
        while True:
            _solve_damped(projected, r, moving, damping, step)
            moved = False
            for j in range(p):
                trial[j] = theta[j]
                if moving[j]:
                    trial[j] = min(max(theta[j] + step[j], lower[j]), upper[j])
                    moved = moved or trial[j] != theta[j]
            new_value = math.inf
            if moved:
                _resolve_linear(
                    kind, linear, q, trial, x, z, root, columns, whitened, alpha, row
                )
                new_value = _residuals(kind, trial, x, ybar, root, fitted, raw, r_trial)
            if new_value < value:
                break
            damping *= 10.0
            if damping > _DAMPING_MAX:
                # No joint step lowers Q. Where a parameter's column has nearly
                # vanished, as a Gauss-probit location's does when the bell's height
                # is near 0, its huge steps spoil the joint ones: try each parameter
                # alone before calling the point a minimum.
                new_value = _single_steps(
                    kind,
                    linear,
                    q,
                    theta,
                    trial,
                    lower,
                    upper,
                    projected,
                    r,
                    moving,
                    single,
                    step,
                    x,
                    ybar,
                    z,
                    root,
                    columns,
                    whitened,
                    alpha,
                    row,
                    fitted,
                    raw,
                    r_trial,
                    value,
                )
                if new_value >= value:
                    return value, True
                damping = _DAMPING_START
                break
        for j in range(p):
            theta[j] = trial[j]
        value = _residuals(kind, theta, x, ybar, root, fitted, raw, r)
        damping = max(damping / 10.0, _DAMPING_MIN)
    return value, False


@jit_numpy_errors
def fit_gene(
    kind: int,
    constraint: int,
    q: int,
    points: FloatArray,
    inner: int,
    bases: FloatArray,
    usable: BoolArray,
    shared: bool,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    sign_gap: float,
    scale: float,
    theta: FloatArray,
    fitted: FloatArray,
) -> tuple[float, bool]:
    """Fit one gene: the lowest local minima of the grid, each refined. ``bases`` are
    the columns at the grid points, whitened already when ``shared``. Writes theta and
    the fitted means; returns the minimum of Q and whether the fit converged. The
    line and the quadratic, which have no nonlinear parameters, are their linear
    solve."""
    k = x.size
    z = np.zeros(k)
    _whiten(root, ybar.reshape((k, 1)), z.reshape((k, 1)))
    alpha = np.zeros(q)
    whitened = np.zeros((k, q))
    if kind == LINE or kind == POLYNOMIAL:
        columns, row = np.zeros((k, q)), np.zeros(q)
        value = _resolve_linear(
            kind, constraint, q, theta, x, z, root, columns, whitened, alpha, row
        )
        for i in range(k):
            fitted[i] = mean_value(kind, theta, x[i])
        return value, True
    n_points = points.shape[0]
    grid_values = np.full(n_points, math.inf)
    for point in range(n_points):
        if not usable[point]:
            continue
        if shared:
            grid_values[point] = point_fit(constraint, bases[point], z, q, alpha)
        else:
            _whiten(root, bases[point], whitened)
            grid_values[point] = point_fit(constraint, whitened, z, q, alpha)
    starts = _grid_minima(grid_values, inner)
    low, high = lower.copy(), upper.copy()
    candidate = np.zeros(theta.size)
    best, best_converged = math.inf, False
    for start in starts:
        if start < 0:
            continue
        if shared:
            point_fit(constraint, bases[start], z, q, alpha)
        else:
            _whiten(root, bases[start], whitened)
            point_fit(constraint, whitened, z, q, alpha)
        phi1 = points[start, 1] if points.shape[1] > 1 else 0.0
        assemble(kind, points[start, 0], phi1, alpha, candidate)
        for j in range(theta.size):
            low[j], high[j] = lower[j], upper[j]
        if kind == DROMICS_EXPONENTIAL:
            # s keeps the sign of its grid point, away from 0, where b and e diverge.
            if candidate[2] > 0.0:
                low[2] = sign_gap
            else:
                high[2] = -sign_gap
        for j in range(theta.size):
            candidate[j] = min(max(candidate[j], low[j]), high[j])
        value, converged = refine(
            kind,
            constraint,
            q,
            candidate,
            low,
            high,
            x,
            ybar,
            z,
            root,
            scale,
            MAX_ITERATIONS,
        )
        if value < best:
            best, best_converged = value, converged
            for j in range(theta.size):
                theta[j] = candidate[j]
    for i in range(k):
        fitted[i] = mean_value(kind, theta, x[i])
    return best, best_converged


@jit_numpy_errors
def _grid_minima(values: FloatArray, inner: int) -> npt.NDArray[np.int64]:
    """The positions of the lowest local minima of a grid's values, at most
    ``STARTS``, lowest first, -1 where there are fewer; a point is a local minimum
    when no neighbour along either axis is lower. The second axis has ``inner``
    points and varies fastest."""
    n = values.size
    outer = n // inner
    starts = np.full(STARTS, -1, dtype=np.int64)
    found = np.full(STARTS, math.inf)
    for point in range(n):
        v = values[point]
        if not math.isfinite(v):
            continue
        i, j = point // inner, point % inner
        if (
            (i > 0 and values[point - inner] < v)
            or (i < outer - 1 and values[point + inner] < v)
            or (j > 0 and values[point - 1] < v)
            or (j < inner - 1 and values[point + 1] < v)
        ):
            continue
        for slot in range(STARTS):
            if v < found[slot]:
                for later in range(STARTS - 1, slot, -1):
                    found[later], starts[later] = found[later - 1], starts[later - 1]
                found[slot], starts[slot] = v, point
                break
    return starts


@jit_numpy_errors
def _fit_genes_range(
    kind: int,
    constraint: int,
    q: int,
    points: FloatArray,
    inner: int,
    bases: FloatArray,
    usable: BoolArray,
    shared: bool,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    sign_gap: float,
    scales: FloatArray,
    active: BoolArray,
    theta: FloatArray,
    fitted: FloatArray,
    values: FloatArray,
    converged: BoolArray,
    start: int,
    stop: int,
) -> None:
    for g in range(start, stop):
        if not active[g]:
            continue
        root = roots[0] if roots.shape[0] == 1 else roots[g]
        values[g], converged[g] = fit_gene(
            kind,
            constraint,
            q,
            points,
            inner,
            bases,
            usable,
            shared,
            x,
            means[g],
            root,
            lower,
            upper,
            sign_gap,
            scales[g],
            theta[g],
            fitted[g],
        )


@jit(parallel=True)
def fit_genes_parallel(
    kind: int,
    constraint: int,
    q: int,
    points: FloatArray,
    inner: int,
    bases: FloatArray,
    usable: BoolArray,
    shared: bool,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    sign_gap: float,
    scales: FloatArray,
    active: BoolArray,
    theta: FloatArray,
    fitted: FloatArray,
    values: FloatArray,
    converged: BoolArray,
    n_runs: int,
) -> None:
    n = means.shape[0]
    for run in prange(n_runs):
        _fit_genes_range(
            kind,
            constraint,
            q,
            points,
            inner,
            bases,
            usable,
            shared,
            x,
            means,
            roots,
            lower,
            upper,
            sign_gap,
            scales,
            active,
            theta,
            fitted,
            values,
            converged,
            run * n // n_runs,
            (run + 1) * n // n_runs,
        )


@jit
def fit_genes_serial(
    kind: int,
    constraint: int,
    q: int,
    points: FloatArray,
    inner: int,
    bases: FloatArray,
    usable: BoolArray,
    shared: bool,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    sign_gap: float,
    scales: FloatArray,
    active: BoolArray,
    theta: FloatArray,
    fitted: FloatArray,
    values: FloatArray,
    converged: BoolArray,
) -> None:
    _fit_genes_range(
        kind,
        constraint,
        q,
        points,
        inner,
        bases,
        usable,
        shared,
        x,
        means,
        roots,
        lower,
        upper,
        sign_gap,
        scales,
        active,
        theta,
        fitted,
        values,
        converged,
        0,
        means.shape[0],
    )


@jit_numpy_errors
def whiten_bases(bases: FloatArray, root: FloatArray) -> FloatArray:
    """The columns of every grid point whitened by one factor R."""
    out = np.empty_like(bases)
    for p in range(bases.shape[0]):
        _whiten(root, bases[p], out[p])
    return out


def fitter_settings(model_set: str) -> dict[str, Any]:
    """How a set's curves are fitted, as a record: each model's mean and parameters,
    the sizes of the grids of nonlinear parameters and the refinement's settings."""
    return {
        "models": {
            m.name: {"mean": m.formula, "parameters": list(m.parameters)}
            for m in SETS[model_set]
        },
        "grid_sizes": dict(GRID_SIZES),
        "refinement_starts": STARTS,
        "max_iterations": MAX_ITERATIONS,
        "stop_decrease": STOP_DECREASE,
    }
