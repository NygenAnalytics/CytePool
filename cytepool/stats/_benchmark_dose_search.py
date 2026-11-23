"""The search for benchmark doses and their profile-likelihood bounds.

A gene's curve f(x; theta) changes from dose 0 by c(x; theta) = f(x) - f(0). Its BMD is
the lowest dose x in (0, U] with |c(x)| = R, R the benchmark response's size. Its
bounds are the ends of the profile likelihood's confidence region {b : deviance(b) <=
threshold}, where deviance(b) = 2 (LL_max - LL_p(b)) and LL_p(b) is the largest
log-likelihood of a curve whose change at b is s R, s the direction of the change at
the BMD. Where the BMR is a multiple of the SD, R = v sigma moves with the SD sigma of
the likelihood, which the profile maximises too.

The curve's parameters split into the linear ones alpha, at most 3, and the nonlinear
ones phi, at most 2: f = B(x; phi) alpha. At a fixed phi the change at b is linear in
alpha, c(b) = a'alpha with a = B(b) - B(0), so the constrained least squares has a
closed form: with alpha_hat, Q_hat and M = (B'WB)^-1 the unconstrained fit, D = a'
alpha_hat and V = a'Ma, the smallest weighted sum of squares with a'alpha = t is Q_hat
+ (D - t)^2 / V. Linear parameters with constraints of their own (a >= 0 of exponential
2 and 3; the cone of exponential 4 and 5; the shared sign of the quadratic's
coefficients) are solved on each face of their constraint as well, keeping the best
feasible solution. Where the SD enters the BMR, t = s v sigma, and the largest
log-likelihood over sigma is the positive root of a quadratic.

The profile at a dose, :func:`profile_at`, refines the nonlinear parameters from
several starts: at each dose of the bounds' scan, which runs outward from the BMD,
the lowest local minima of the grid that the fits search and the optima found at the
neighbouring dose of the scan, where the fit itself is the optimum at the BMD. The
refinement is a projected quasi-Newton (BFGS) method on the closed-form objective,
with its exact gradient by the envelope theorem: -2 r'(dB/dphi) alpha + 2 mu
dc(b)/dphi, mu the multiplier of the constraint on the change.

A model's fixed description travels as one tuple, ``model``: its kind, the
constraint of its linear parameters, their number q, the positions of its nonlinear
parameters in theta, their bounds and the gap that keeps the sign of DRomics'
exponential rate. Each gene's numbers travel as one row of the columns ``FORM`` to
``PIECE``.

Every gene's search is the same sequence of operations whatever the threads, so
results do not depend on them.

References:
    Brent RP (1973). Algorithms for Minimization without Derivatives. Prentice-Hall,
    Englewood Cliffs.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._jit import jit, jit_numpy_errors
from cytepool._linear_algebra import householder_qr
from cytepool.stats._dose_curve_models import (
    C_MAX,
    CONE,
    DROMICS_EXPONENTIAL,
    NON_NEGATIVE,
    SAME_SIGN,
    assemble,
    basis_row,
    mean_gradient,
    mean_value,
)
from cytepool.stats._dose_curve_search import (
    STARTS,
    grid_minima,
    whiten,
    whitened_jacobian,
    whitened_residuals,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

# How the likelihood is profiled: a fixed change with the variance profiled with the
# curve (constant variance); a fixed change with a first stage's variance s^2 held
# (the F threshold); a change of v sigma, with sigma profiled with the curve.
CHANGE_PROFILED_VARIANCE, CHANGE_GIVEN_VARIANCE, CHANGE_IN_SD = 0, 1, 2

# The columns of each gene's numbers: how the likelihood is profiled, the BMR's value
# in the search (R, or v), the direction s, the likelihood's constants A0, N and s^2,
# the BMD, the objective at the fit, the threshold, and 1 to bound the BMD by the piece
# of the region that holds it, 0 for the region's far ends.
FORM, VALUE, DIRECTION, A0, N_EFF, S2, BMD, OBJECTIVE_FIT, THRESHOLD, PIECE = range(10)

# The range of doses searched: from the lowest positive dose / 100 (DRomics' minBMD)
# to 50 times the top dose, scanned at 8 doses per decade for the bounds.
SEARCH_BELOW = 100.0
SEARCH_ABOVE = 50.0
SCAN_PER_DECADE = 8

MAX_ITERATIONS = 200
MAX_CARRIED = 2  # optima carried from one dose of the scan to the next
GRID_MARGIN = 20.0  # starts this far above the threshold in deviance are not refined
ROOT_TOLERANCE = 1e-10  # on ln dose
ABOVE_FIT = 1e-6  # a profile this far below the fit's objective: the fit was no maximum
STOP = 1e-13
HELD = 1e-8  # of a parameter's range: this near a bound, it can be held on it
_ARMIJO = 1e-4
_FEASIBLE = 1e-12  # relative slack of the linear parameters' constraints


def search_range(lowest: float, top: float) -> tuple[float, float]:
    """The doses searched for BMDs and bounds: from the lowest positive dose / 100 to
    50 times the top dose."""
    return lowest / SEARCH_BELOW, SEARCH_ABOVE * top


def scan_doses(lowest: float, top: float) -> FloatArray:
    """The doses at which the bounds' search evaluates the profile: the search range
    at 8 doses per decade, both ends included."""
    low, high = search_range(lowest, top)
    n = int(math.ceil(math.log10(high / low) * SCAN_PER_DECADE)) + 1
    doses: FloatArray = np.exp(np.linspace(math.log(low), math.log(high), n))
    doses[0], doses[-1] = low, high
    return doses


def search_settings() -> dict[str, Any]:
    """How BMDs and their bounds are searched, as a record: the range of doses, the
    scan's doses per decade, how many distinct optima at one dose of the scan start
    the refinement at the next, how far above the threshold, in deviance, a start may
    lie and still be refined, the tolerance of the bounds' roots on log dose, and the
    most iterations of a refinement."""
    return {
        "range": (
            f"from the lowest positive dose / {SEARCH_BELOW:g} to {SEARCH_ABOVE:g} "
            "times the top dose"
        ),
        "scan_doses_per_decade": SCAN_PER_DECADE,
        "optima_carried_to_the_next_scan_dose": MAX_CARRIED,
        "starts_refined_within_deviance_above_threshold": GRID_MARGIN,
        "root_tolerance_on_log_dose": ROOT_TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
    }


# ---------------------------------------------------------------------------------------
# The constrained least squares at one point of the nonlinear parameters
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _feasible(linear: int, alpha: FloatArray) -> bool:
    """Whether linear parameters keep their kind's constraint."""
    if linear == NON_NEGATIVE:
        return bool(alpha[0] >= 0.0)
    if linear == CONE:
        a, rest = alpha[0], alpha[1]
        slack = _FEASIBLE * abs(a)
        return bool(a >= 0.0 and -a - slack <= rest <= (C_MAX - 1.0) * a + slack)
    if linear == SAME_SIGN:
        return bool(alpha[1] * alpha[2] >= 0.0)
    return True


@jit_numpy_errors
def _subspace(linear: int, face: int, q: int, basis_out: FloatArray) -> int:
    """The columns of the face-th subspace of a kind's linear parameters, as a q x m
    matrix P (alpha = P beta); returns m, or 0 when the kind has no such face.
    Face 0 is the whole space; the cone of exponential 4 and 5 has its two rays
    c = 0 and c = C_MAX, and the restricted quadratic its faces b2 = 0 and b1 = 0."""
    for i in range(3):
        for j in range(3):
            basis_out[i, j] = 0.0
    if face == 0:
        for j in range(q):
            basis_out[j, j] = 1.0
        return q
    if linear == CONE and face <= 2:
        basis_out[0, 0] = 1.0
        basis_out[1, 0] = -1.0 if face == 1 else C_MAX - 1.0
        return 1
    if linear == SAME_SIGN and face <= 2:
        basis_out[0, 0] = 1.0
        basis_out[1 if face == 1 else 2, 1] = 1.0
        return 2
    return 0


@jit_numpy_errors
def face_fits(
    linear: int,
    q: int,
    columns: FloatArray,
    z: FloatArray,
    inverses: FloatArray,
    betas: FloatArray,
    sums: FloatArray,
    sizes: IntArray,
) -> None:
    """The unconstrained least squares of z on each face of a kind's linear
    parameters, which do not depend on the dose: for face f, alpha = P_f beta, the
    inverse of the triangular factor R of the QR decomposition of the whitened
    columns ``columns`` P_f (``inverses[f]``, so that (P'B'WBP)^-1 = R^-1 R^-T),
    beta_hat (``betas[f]``) and its residual sum of squares (``sums[f]``); ``sizes[f]``
    is the face's dimension, 0 when the kind has no such face or its columns are not
    finite or linearly dependent: one whose part outside the span of the columns
    before it is within 1e-12 of its own norm, the rule of :func:`householder_qr`."""
    k = z.size
    p = np.zeros((3, 3))
    work = np.zeros((k, 4))
    norms = np.zeros(3)
    aliased = np.zeros(3, dtype=np.bool_)
    kept = np.zeros(3, dtype=np.int64)
    products = np.zeros(4)
    for face in range(3):
        m = _subspace(linear, face, q, p)
        sizes[face] = 0
        if m == 0:
            continue
        finite = True
        for i in range(k):
            for j in range(m):
                total = 0.0
                for c in range(q):
                    total += columns[i, c] * p[c, j]
                work[i, j] = total
                finite = finite and math.isfinite(total)
            work[i, m] = z[i]
        if not finite:
            continue
        # [P_f columns | z] = Q [R  Q'z]: R in qr[:m, :m], Q'z in qr[:, m].
        qr = work[:, : m + 1]
        rank = householder_qr(
            qr, k, m, 1e-12, True, 0.0, norms, aliased, kept, products[: m + 1]
        )
        if rank < m:
            continue
        # R^-1 (upper triangular) and beta_hat = R^-1 (Q'z)[:m]
        for i in range(3):
            for j in range(3):
                inverses[face, i, j] = 0.0
        for j in range(m):
            inverses[face, j, j] = 1.0 / qr[j, j]
            for i in range(j - 1, -1, -1):
                total = 0.0
                for c in range(i + 1, j + 1):
                    total += qr[i, c] * inverses[face, c, j]
                inverses[face, i, j] = -total / qr[i, i]
        for i in range(m):
            total = 0.0
            for j in range(i, m):
                total += inverses[face, i, j] * qr[j, m]
            betas[face, i] = total
        residual = 0.0
        for i in range(m, k):
            residual += qr[i, m] * qr[i, m]
        sums[face] = residual
        sizes[face] = m


@jit_numpy_errors
def constrained_minimum(
    linear: int,
    q: int,
    inverses: FloatArray,
    betas: FloatArray,
    sums: FloatArray,
    sizes: IntArray,
    a: FloatArray,
    row: FloatArray,
    alpha: FloatArray,
    extra: FloatArray,
    work: FloatArray,
) -> float:
    """The profile objective at one point of the nonlinear parameters: the smallest
    -2 log-likelihood (up to a constant), or Q / s^2 under ``CHANGE_GIVEN_VARIANCE``, of
    the linear parameters whose change at the dose is s R (``row[VALUE]`` is R, or v
    for ``CHANGE_IN_SD``), within the linear parameters' constraint; writes the best
    alpha. ``row`` holds a gene's numbers, of which this reads ``FORM`` to ``S2``.

    The faces' fits come from :func:`face_fits`; ``a`` = B(b) - B(0). ``extra``
    receives the multiplier of the constraint on the change, the squared SD of the
    solution (``CHANGE_IN_SD``) and the derivative of the objective in Q; ``work`` is a 6 x 3
    scratch array. Returns infinity when no feasible solution exists."""
    form, value, s = int(row[FORM]), row[VALUE], row[DIRECTION]
    a0, n_eff, s2 = row[A0], row[N_EFF], row[S2]
    best = math.inf
    p = work[0:3]
    sub_a = work[3]
    w = work[4]
    trial = work[5]
    for face in range(3):
        m = sizes[face]
        if m == 0:
            continue
        if face == 0:
            for i in range(m):
                sub_a[i] = a[i]
        else:
            _subspace(linear, face, q, p)
            for i in range(m):
                total = 0.0
                for k in range(q):
                    total += p[k, i] * a[k]
                sub_a[i] = total
        d = 0.0
        v = 0.0
        for i in range(m):
            d += sub_a[i] * betas[face, i]
            # w = R^-T a', the i-th entry: sum over rows j <= i of (R^-1)[j, i] a'_j
            total = 0.0
            for j in range(i + 1):
                total += inverses[face, j, i] * sub_a[j]
            w[i] = total
            v += total * total
        if not v > 0.0:
            continue
        sub_q = sums[face]
        sigma2 = math.nan
        if form == CHANGE_IN_SD:
            # N sigma^2 + s v (D / V) sigma - (A0 + Q' + D^2 / V) = 0
            k1 = s * value * d / v
            constant = a0 + sub_q + d * d / v
            root = math.sqrt(k1 * k1 + 4.0 * n_eff * constant)
            if k1 > 0.0:
                sigma = 2.0 * constant / (k1 + root)
            else:
                sigma = (root - k1) / (2.0 * n_eff)
            if not sigma > 0.0:
                continue
            t = s * value * sigma
            sigma2 = sigma * sigma
        else:
            t = s * value
        rss = sub_q + (d - t) * (d - t) / v
        if form == CHANGE_PROFILED_VARIANCE:
            objective = n_eff * math.log(a0 + rss)
            slope = n_eff / (a0 + rss)
        elif form == CHANGE_GIVEN_VARIANCE:
            objective = rss / s2
            slope = 1.0 / s2
        else:
            objective = n_eff * math.log(sigma2) + (a0 + rss) / sigma2
            slope = 1.0 / sigma2
        if not objective < best:
            continue
        # beta = beta_hat - M' a' (D - t) / V, alpha = P beta
        for k in range(q):
            trial[k] = 0.0
        for i in range(m):
            total = 0.0  # (M a')_i = (R^-1 w)_i
            for j in range(i, m):
                total += inverses[face, i, j] * w[j]
            beta_i = betas[face, i] - total * (d - t) / v
            if face == 0:
                trial[i] = beta_i
            else:
                for k in range(q):
                    trial[k] += p[k, i] * beta_i
        if not _feasible(linear, trial):
            continue
        best = objective
        for k in range(q):
            alpha[k] = trial[k]
        extra[0] = (d - t) / v  # the multiplier mu of a'alpha = t
        extra[1] = sigma2
        extra[2] = slope
    return best


@jit_numpy_errors
def grid_faces(
    linear: int,
    q: int,
    bases: FloatArray,
    usable: BoolArray,
    root: FloatArray,
    z: FloatArray,
    inverses: FloatArray,
    betas: FloatArray,
    sums: FloatArray,
    sizes: IntArray,
) -> None:
    """The faces' fits (:func:`face_fits`) of the whitened columns of every usable
    grid point, whose columns at the doses are ``bases``, ``(n_points, K, q)``."""
    columns = np.zeros((z.size, q))
    for point in range(bases.shape[0]):
        if usable[point]:
            whiten(root, bases[point], columns)
            face_fits(
                linear,
                q,
                columns,
                z,
                inverses[point],
                betas[point],
                sums[point],
                sizes[point],
            )


@jit_numpy_errors
def grid_values(
    linear: int,
    q: int,
    usable: BoolArray,
    inverses: FloatArray,
    betas: FloatArray,
    sums: FloatArray,
    sizes: IntArray,
    change: FloatArray,
    row: FloatArray,
    values: FloatArray,
) -> None:
    """The profile objective at every grid point at one dose, whose columns' change
    from dose 0 is ``change``, ``(n_points, q)``; infinite at unusable points."""
    alpha = np.zeros(3)
    extra = np.zeros(3)
    work = np.zeros((6, 3))
    for point in range(values.size):
        values[point] = math.inf
        if usable[point]:
            values[point] = constrained_minimum(
                linear,
                q,
                inverses[point],
                betas[point],
                sums[point],
                sizes[point],
                change[point],
                row,
                alpha,
                extra,
                work,
            )


# ---------------------------------------------------------------------------------------
# The objective and its gradient along the nonlinear parameters
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def workspace(
    k: int,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    IntArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """The scratch arrays of the objective and its gradient at K doses: the raw and
    whitened columns, the faces' fits, alpha, a, a row of up to 5 parameters, the
    work of ``constrained_minimum``, and the gradient's fitted means, raw and
    whitened residuals, raw and whitened Jacobian, and change."""
    return (
        np.zeros((k, 3)),
        np.zeros((k, 3)),
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
        np.zeros(3),
        np.zeros(3, dtype=np.int64),
        np.zeros(3),
        np.zeros(3),
        np.zeros(5),
        np.zeros((6, 3)),
        np.zeros(k),
        np.zeros(k),
        np.zeros(k),
        np.zeros((k, 5)),
        np.zeros((k, 5)),
        np.zeros(2),
    )


@jit_numpy_errors
def profile_point(
    model: tuple,  # type: ignore[type-arg]
    row: FloatArray,
    phi: FloatArray,
    b: float,
    x: FloatArray,
    root: FloatArray,
    z: FloatArray,
    theta: FloatArray,
    extra: FloatArray,
    space: tuple,  # type: ignore[type-arg]
) -> float:
    """The profile objective at nonlinear parameters phi and dose b, computing the
    columns afresh; writes the curve's parameters theta. ``space`` is a
    :func:`workspace`. Columns or a change at b that are not finite, as an
    exponential's far beyond the doses, give an infinite objective."""
    kind, linear, q = model[0], model[1], model[2]
    raw, columns, inverses, betas, sums, sizes = (
        space[0],
        space[1],
        space[2],
        space[3],
        space[4],
        space[5],
    )
    alpha, a, basis, solve_work = space[6], space[7], space[8], space[9]
    phi1 = phi[1] if phi.size > 1 else 0.0
    for i in range(x.size):
        basis_row(kind, phi[0], phi1, x[i], basis)
        for j in range(q):
            raw[i, j] = basis[j]
    whiten(root, raw[:, :q], columns[:, :q])
    basis_row(kind, phi[0], phi1, b, basis)
    for j in range(q):
        a[j] = basis[j]
    basis_row(kind, phi[0], phi1, 0.0, basis)
    for j in range(q):
        a[j] -= basis[j]
    face_fits(linear, q, columns, z, inverses, betas, sums, sizes)
    objective = constrained_minimum(
        linear, q, inverses, betas, sums, sizes, a, row, alpha, extra, solve_work
    )
    if math.isfinite(objective):
        assemble(kind, phi[0], phi1, alpha, theta)
    return objective


@jit_numpy_errors
def profile_gradient(
    kind: int,
    theta: FloatArray,
    b: float,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    index: IntArray,
    extra: FloatArray,
    gradient: FloatArray,
    jacobian: FloatArray,
    space: tuple,  # type: ignore[type-arg]
) -> None:
    """The gradient of the profile objective in the nonlinear parameters, by the
    envelope theorem, and the whitened Jacobian of the residuals in them."""
    k, r, p = x.size, index.size, theta.size
    row, change = space[8], space[15]
    fitted, raw, residual = space[10], space[11], space[12]
    raw_jacobian, full = space[13][:, :p], space[14][:, :p]
    whitened_residuals(kind, theta, x, ybar, root, fitted, raw, residual)
    whitened_jacobian(kind, theta, x, root, row, raw_jacobian, full)
    for j in range(r):
        gradient[j] = 0.0
    for i in range(k):
        for j in range(r):
            jacobian[i, j] = full[i, index[j]]
            gradient[j] += 2.0 * residual[i] * jacobian[i, j]
    mean_gradient(kind, theta, b, row)
    for j in range(r):
        change[j] = row[index[j]]
    mean_gradient(kind, theta, 0.0, row)
    for j in range(r):
        change[j] -= row[index[j]]
        gradient[j] = extra[2] * (gradient[j] + 2.0 * extra[0] * change[j])


@jit_numpy_errors
def _gauss_newton(jacobian: FloatArray, slope: float, hessian: FloatArray) -> None:
    """The Gauss-Newton matrix 2 slope J'J of the whitened means' Jacobian in the
    nonlinear parameters, with a ridge of 1e-12 of its trace."""
    k, r = jacobian.shape
    trace = 0.0
    for i in range(r):
        for j in range(r):
            total = 0.0
            for m in range(k):
                total += jacobian[m, i] * jacobian[m, j]
            hessian[i, j] = 2.0 * slope * total
        trace += hessian[i, i]
    for i in range(r):
        hessian[i, i] += 1e-12 * trace + 1e-300


@jit_numpy_errors
def refine_profile(
    model: tuple,  # type: ignore[type-arg]
    row: FloatArray,
    start: FloatArray,
    b: float,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    z: FloatArray,
    phi: FloatArray,
    theta: FloatArray,
    space: tuple,  # type: ignore[type-arg]
) -> float:
    """Refine the nonlinear parameters from ``start`` to the nearest minimum of the
    profile objective at dose b, within their bounds, writing them to ``phi`` and the
    curve's parameters to ``theta``; returns the objective. DRomics' exponential keeps
    the sign of its start's rate, away from 0, as its fits do.

    A projected quasi-Newton (BFGS) method on the closed-form objective, with its
    exact gradient: a parameter at a bound is held there while the gradient points
    outward, and one within ``HELD`` of its range from a bound, with the gradient
    pointing outward, moves onto it when that does not raise the objective, so that
    the step of the others does not keep pressing it towards the bound; the matrix
    starts as the Gauss-Newton matrix of the whitened means, and
    starts again from it whenever the parameters held change or a step finds no
    positive curvature; a step moves no parameter by more than half its value or a
    hundredth of its range, and backtracks by quadratic interpolation until the
    objective falls by 1e-4 of the decrease the step predicts. It stops when that
    predicted decrease is at most 1e-13 of the objective, when no step lowers it, or
    after ``MAX_ITERATIONS``."""
    kind, index, sign_gap = model[0], model[3], model[6]
    lower, upper = model[4].copy(), model[5].copy()
    if kind == DROMICS_EXPONENTIAL:
        if start[0] > 0.0:
            lower[0] = sign_gap
        else:
            upper[0] = -sign_gap
    r = index.size
    k = x.size
    extra = np.zeros(3)
    trial_extra = np.zeros(3)
    for j in range(r):
        phi[j] = min(max(start[j], lower[j]), upper[j])
    value_now = profile_point(model, row, phi, b, x, root, z, theta, extra, space)
    if not math.isfinite(value_now):
        return value_now
    gradient = np.zeros(r)
    jacobian = np.zeros((k, r))
    profile_gradient(
        kind, theta, b, x, ybar, root, index, extra, gradient, jacobian, space
    )
    hessian = np.zeros((r, r))
    _gauss_newton(jacobian, extra[2], hessian)
    trial = np.zeros(r)
    trial_theta = np.zeros(theta.size)
    new_gradient = np.zeros(r)
    step = np.zeros(r)
    free = np.zeros(r, dtype=np.bool_)
    was_free = np.zeros(r, dtype=np.bool_)
    hs = np.zeros(r)
    for iteration in range(MAX_ITERATIONS):
        snapped = False
        for j in range(r):
            reach = HELD * (upper[j] - lower[j])
            trial[j] = phi[j]
            if gradient[j] > 0.0 and lower[j] < phi[j] <= lower[j] + reach:
                trial[j] = lower[j]
                snapped = True
            elif gradient[j] < 0.0 and upper[j] - reach <= phi[j] < upper[j]:
                trial[j] = upper[j]
                snapped = True
        if snapped:
            new_value = profile_point(
                model, row, trial, b, x, root, z, trial_theta, trial_extra, space
            )
            if new_value <= value_now:
                for j in range(r):
                    phi[j] = trial[j]
                for j in range(theta.size):
                    theta[j] = trial_theta[j]
                for j in range(3):
                    extra[j] = trial_extra[j]
                value_now = new_value
                profile_gradient(
                    kind,
                    theta,
                    b,
                    x,
                    ybar,
                    root,
                    index,
                    extra,
                    gradient,
                    jacobian,
                    space,
                )
        n_free = 0
        changed = False
        for j in range(r):
            outward = (phi[j] <= lower[j] and gradient[j] > 0.0) or (
                phi[j] >= upper[j] and gradient[j] < 0.0
            )
            free[j] = not outward
            n_free += 1 if free[j] else 0
            changed = changed or (iteration > 0 and free[j] != was_free[j])
            was_free[j] = free[j]
        if n_free == 0:
            break
        if changed:
            _gauss_newton(jacobian, extra[2], hessian)
        # step = -H_ff^-1 g_f, r <= 2; a matrix that is not positive definite on the
        # free parameters starts again from the Gauss-Newton matrix.
        for j in range(r):
            step[j] = 0.0
        for attempt in range(2):
            if n_free == 1:
                j = 0 if free[0] else 1
                if hessian[j, j] > 0.0:  # pragma: no branch - the ridge keeps it > 0
                    step[j] = -gradient[j] / hessian[j, j]
                    break
            else:
                det = hessian[0, 0] * hessian[1, 1] - hessian[0, 1] * hessian[1, 0]
                if det > 0.0 and hessian[0, 0] > 0.0:
                    step[0] = (
                        -(hessian[1, 1] * gradient[0] - hessian[0, 1] * gradient[1])
                        / det
                    )
                    step[1] = (
                        -(hessian[0, 0] * gradient[1] - hessian[1, 0] * gradient[0])
                        / det
                    )
                    break
            _gauss_newton(jacobian, extra[2], hessian)
        # No parameter moves by more than half its value or a hundredth of its range.
        factor = 1.0
        for j in range(r):
            cap = max(0.5 * abs(phi[j]), 0.01 * (upper[j] - lower[j]))
            if abs(step[j]) * factor > cap:
                factor = cap / abs(step[j])
        predicted = 0.0
        for j in range(r):
            step[j] *= factor
            predicted -= gradient[j] * step[j]
        if not predicted > STOP * (abs(value_now) + 1.0):
            break
        size = 1.0
        accepted = False
        new_value = value_now
        while size > 1e-12:
            moved = False
            for j in range(r):
                trial[j] = phi[j]
                if free[j]:
                    trial[j] = min(max(phi[j] + size * step[j], lower[j]), upper[j])
                    moved = moved or trial[j] != phi[j]
            if not moved:
                break
            new_value = profile_point(
                model, row, trial, b, x, root, z, trial_theta, trial_extra, space
            )
            if new_value <= value_now - _ARMIJO * size * predicted:
                accepted = True
                break
            # the minimum of the parabola through f(0), f'(0) = -predicted and f(size)
            shrink = 0.5
            excess = new_value - value_now + size * predicted
            if math.isfinite(new_value) and excess > 0.0:
                shrink = min(max(0.5 * predicted * size / excess, 0.1), 0.5)
            size *= shrink
        if not accepted:
            break
        profile_gradient(
            kind,
            trial_theta,
            b,
            x,
            ybar,
            root,
            index,
            trial_extra,
            new_gradient,
            jacobian,
            space,
        )
        # BFGS update with s_k = trial - phi, y_k = new_gradient - gradient; without
        # positive curvature along the step, the Gauss-Newton matrix instead.
        sy = 0.0
        shs = 0.0
        for i in range(r):
            total = 0.0
            for j in range(r):
                total += hessian[i, j] * (trial[j] - phi[j])
            hs[i] = total
            shs += (trial[i] - phi[i]) * total
            sy += (trial[i] - phi[i]) * (new_gradient[i] - gradient[i])
        if sy > 0.0 and shs > 0.0:
            for i in range(r):
                for j in range(r):
                    yi = new_gradient[i] - gradient[i]
                    yj = new_gradient[j] - gradient[j]
                    hessian[i, j] += yi * yj / sy - hs[i] * hs[j] / shs
        else:
            _gauss_newton(jacobian, trial_extra[2], hessian)
        for j in range(r):
            phi[j] = trial[j]
            gradient[j] = new_gradient[j]
        for j in range(theta.size):
            theta[j] = trial_theta[j]
        for j in range(3):
            extra[j] = trial_extra[j]
        value_now = new_value
    return value_now


@jit_numpy_errors
def _same(phi: FloatArray, other: FloatArray, r: int) -> bool:
    for j in range(r):
        if abs(phi[j] - other[j]) > 1e-6 * max(abs(phi[j]), abs(other[j])) + 1e-300:
            return False
    return True


@jit_numpy_errors
def _keep_optimum(
    objective: float,
    phi: FloatArray,
    r: int,
    kept: FloatArray,
    kept_values: FloatArray,
    n_kept: int,
) -> int:
    """Insert a refined optimum among the kept ones, lowest first, unless it repeats
    one; returns how many are kept."""
    if not math.isfinite(objective):
        return n_kept
    for i in range(n_kept):
        if _same(phi, kept[i], r):
            if objective < kept_values[i]:
                kept_values[i] = objective
            return n_kept
    slot = n_kept
    while slot > 0 and kept_values[slot - 1] > objective:
        slot -= 1
    if slot >= MAX_CARRIED:
        return n_kept
    last = min(n_kept, MAX_CARRIED - 1)
    for i in range(last, slot, -1):
        kept_values[i] = kept_values[i - 1]
        for j in range(r):
            kept[i, j] = kept[i - 1, j]
    kept_values[slot] = objective
    for j in range(r):
        kept[slot, j] = phi[j]
    return min(n_kept + 1, MAX_CARRIED)


@jit_numpy_errors
def profile_at(
    model: tuple,  # type: ignore[type-arg]
    row: FloatArray,
    b: float,
    starts: FloatArray,
    n_starts: int,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    z: FloatArray,
    kept: FloatArray,
    kept_values: FloatArray,
    space: tuple,  # type: ignore[type-arg]
) -> tuple[float, int]:
    """The profile objective at dose b, the lowest found by refining the nonlinear
    parameters from each of the first ``n_starts`` rows of ``starts``, in order, with
    the distinct optima found, lowest first, at most ``MAX_CARRIED`` of them, in
    ``kept`` and ``kept_values``; returns the objective and how many are kept. A
    curve without nonlinear parameters, the straight line or the quadratic, has its
    objective in closed form, and keeps none."""
    index = model[3]
    r = index.size
    theta = np.zeros(model[2] + r)
    phi = np.zeros(max(r, 1))
    if r == 0:
        extra = np.zeros(3)
        return profile_point(model, row, phi, b, x, root, z, theta, extra, space), 0
    count = 0
    for i in range(n_starts):
        objective = refine_profile(
            model, row, starts[i], b, x, ybar, root, z, phi, theta, space
        )
        count = _keep_optimum(objective, phi, r, kept, kept_values, count)
    return (kept_values[0] if count else math.inf), count


# ---------------------------------------------------------------------------------------
# Roots by Brent's method, in reverse communication
# ---------------------------------------------------------------------------------------

_EPS = 2.220446049250313e-16


@jit_numpy_errors
def root_start(state: FloatArray, a: float, fa: float, b: float, fb: float) -> None:
    """Start a search for the root of f between a and b, f(a) and f(b) of opposite
    signs (Brent 1973, chapter 4). The state holds a, b, c, f(a), f(b), f(c), the
    last step and the one before."""
    state[0], state[1], state[2] = a, b, a
    state[3], state[4], state[5] = fa, fb, fa
    state[6] = state[7] = b - a


@jit_numpy_errors
def root_point(state: FloatArray, tolerance: float) -> tuple[float, bool]:
    """The next point at which to evaluate f, and whether the search has ended,
    with the root in ``state[1]``: when the bracket is narrower than ``tolerance``
    plus a few units in the last place, or f is 0 there."""
    a, b, c = state[0], state[1], state[2]
    fa, fb, fc = state[3], state[4], state[5]
    step, previous = state[6], state[7]
    if (fb > 0.0) == (fc > 0.0):
        c, fc = a, fa
        step = previous = b - a
    if abs(fc) < abs(fb):
        a, b, c = b, c, b
        fa, fb, fc = fb, fc, fb
    tol = 2.0 * _EPS * max(abs(b), 1.0) + 0.5 * tolerance
    middle = 0.5 * (c - b)
    state[0], state[1], state[2] = a, b, c
    state[3], state[4], state[5] = fa, fb, fc
    if abs(middle) <= tol or fb == 0.0:
        return b, True
    if abs(previous) >= tol and abs(fa) > abs(fb):
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
        if 2.0 * p < min(3.0 * middle * q - abs(tol * q), abs(previous * q)):
            previous, step = step, p / q
        else:
            previous = step = middle
    else:
        previous = step = middle
    state[6], state[7] = step, previous
    return b + (step if abs(step) > tol else math.copysign(tol, middle)), False


@jit_numpy_errors
def root_update(state: FloatArray, x: float, fx: float) -> None:
    """Record f(x) at the point that ``root_point`` gave."""
    state[0], state[3] = state[1], state[4]
    state[1], state[4] = x, fx


# ---------------------------------------------------------------------------------------
# The BMD: the first crossing of the BMR
# ---------------------------------------------------------------------------------------

CROSSED, CROSSED_BELOW_RANGE, NEVER_CROSSED = 0, 1, 2


_LARGEST = 1e300  # a change beyond it, an overflow far beyond the doses, counts as it


@jit_numpy_errors
def _change(kind: int, theta: FloatArray, x: float, level: float) -> float:
    """f(x) - f(0), within -1e300 and 1e300, so that a rising exponential's
    overflow beyond the doses still brackets a root; NaN stays NaN."""
    change = mean_value(kind, theta, x) - level
    if change > _LARGEST:
        return _LARGEST
    if change < -_LARGEST:
        return -_LARGEST
    return change


@jit_numpy_errors
def _crossing_on(
    kind: int, theta: FloatArray, level: float, target: float, low: float, high: float
) -> float:
    """The dose in [low, high] at which the change reaches ``target``, where the
    change is monotone and brackets it; found on ln dose."""
    state = np.zeros(8)
    fa = _change(kind, theta, low, level) - target
    fb = _change(kind, theta, high, level) - target
    if fa == 0.0:
        return low
    if fb == 0.0:
        return high
    root_start(state, math.log(low), fa, math.log(high), fb)
    for _ in range(200):  # pragma: no branch - Brent's method stops far sooner
        x, done = root_point(state, 0.0)
        if done:
            break
        root_update(state, x, _change(kind, theta, math.exp(x), level) - target)
    return math.exp(state[1])


@jit_numpy_errors
def first_crossing(
    kind: int, theta: FloatArray, size: float, turn: float, low: float, high: float
) -> tuple[float, float, int]:
    """The lowest dose in (0, high] at which |f(x) - f(0)| reaches ``size``, its
    direction (+1 or -1) and how it was found: ``CROSSED`` in [low, high], ``CROSSED_BELOW_RANGE``
    when it lies below ``low`` (given as ``low``), ``NEVER_CROSSED`` otherwise.

    The change is monotone up to ``turn``, the dose at which the slope of a curve
    that rises and falls changes sign (NaN for monotone curves), and monotone again
    beyond it, so that |f(x) - f(0)| reaches the size first before the turn, or
    beyond it in the other direction. ``size`` is finite and above 0, and the
    curve's level at dose 0 finite."""
    level = mean_value(kind, theta, 0.0)
    end = turn if (math.isfinite(turn) and turn < high) else high
    at_end = _change(kind, theta, end, level)
    if abs(at_end) >= size:
        s = 1.0 if at_end > 0.0 else -1.0
        if end <= low or s * _change(kind, theta, low, level) > size:
            return low, s, CROSSED_BELOW_RANGE
        return _crossing_on(kind, theta, level, s * size, low, end), s, CROSSED
    if end < high:
        # Beyond the turn the change moves monotonically to its value at ``high``,
        # whose sign gives the direction, also where the change at the turn is 0.
        at_high = _change(kind, theta, high, level)
        if abs(at_high) >= size:
            s = 1.0 if at_high > 0.0 else -1.0
            start = max(end, low)
            if s * _change(kind, theta, start, level) > size:
                return low, s, CROSSED_BELOW_RANGE
            return _crossing_on(kind, theta, level, s * size, start, high), s, CROSSED
    return math.nan, 0.0, NEVER_CROSSED


@jit
def first_crossings(
    kind: int,
    theta: FloatArray,
    sizes: FloatArray,
    turns: FloatArray,
    low: float,
    high: float,
    bmd: FloatArray,
    direction: FloatArray,
    how: IntArray,
) -> None:
    """:func:`first_crossing` of every gene of one kind of model."""
    for g in range(theta.shape[0]):
        bmd[g], direction[g], how[g] = first_crossing(
            kind, theta[g], sizes[g], turns[g], low, high
        )


# ---------------------------------------------------------------------------------------
# The bounds: the ends of the profile likelihood's confidence region
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def gene_bounds(
    model: tuple,  # type: ignore[type-arg]
    points: FloatArray,
    inner: int,
    usable: BoolArray,
    bases: FloatArray,
    scan: FloatArray,
    change: FloatArray,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    row: FloatArray,
    phi_fit: FloatArray,
) -> tuple[float, float]:
    """The BMDL and BMDU of one gene: the lowest and highest doses of the scan's range
    [scan[0], scan[-1]] whose profile deviance is within the threshold, or, where the
    gene's numbers ask for the piece (``PIECE``), the ends of the piece of that region
    that holds the BMD, as BMDS 2 bounds a BMD; 0 when the region reaches the lowest
    dose of the scan, infinity when it reaches the highest. NaN when the profile at the
    BMD or at a dose of the scan lies below the fit's objective by more than
    ``ABOVE_FIT``, so that the fit was not the maximum.

    ``bases`` are the linear columns at the doses ``x`` of every grid point, ``change``
    those at each dose of the scan minus at dose 0, ``(n_points, n_scan, q)``; for
    the straight line and the quadratic, which have no nonlinear parameters, one
    point. ``row`` holds the gene's numbers; ``phi_fit`` the fit's nonlinear
    parameters."""
    linear, q, index = model[1], model[2], model[3]
    k = x.size
    r = index.size
    n_points = bases.shape[0]
    n_scan = scan.size
    bmd, objective_fit, threshold = row[BMD], row[OBJECTIVE_FIT], row[THRESHOLD]
    piece = row[PIECE] != 0.0
    z = np.zeros(k)
    whiten(root, ybar.reshape((k, 1)), z.reshape((k, 1)))
    space = workspace(k)
    kept = np.zeros((n_scan, MAX_CARRIED, max(r, 1)))
    kept_values = np.full((n_scan, MAX_CARRIED), math.inf)
    n_kept = np.zeros(n_scan, dtype=np.int64)
    starts = np.zeros((MAX_CARRIED + STARTS, max(r, 1)))
    # The fit itself must be the profile's maximum at the BMD.
    for c in range(phi_fit.size):
        starts[0, c] = phi_fit[c]
    at_bmd, _ = profile_at(
        model,
        row,
        bmd,
        starts,
        1,
        x,
        ybar,
        root,
        z,
        np.zeros((MAX_CARRIED, max(r, 1))),
        np.zeros(MAX_CARRIED),
        space,
    )
    if at_bmd - objective_fit < -ABOVE_FIT:
        return math.nan, math.nan
    inverses = np.zeros((n_points, 3, 3, 3))
    betas = np.zeros((n_points, 3, 3))
    sums = np.zeros((n_points, 3))
    sizes = np.zeros((n_points, 3), dtype=np.int64)
    if r > 0:
        grid_faces(linear, q, bases, usable, root, z, inverses, betas, sums, sizes)
    deviance = np.full(n_scan, math.inf)
    values = np.full(n_points, math.inf)
    below = -1
    for j in range(n_scan):
        if scan[j] < bmd:
            below = j
    for side in range(2):
        first = below if side == 0 else below + 1
        stop = -1 if side == 0 else n_scan
        step = -1 if side == 0 else 1
        previous = -1
        for j in range(first, stop, step):
            if r == 0:
                lowest, _ = profile_at(
                    model,
                    row,
                    scan[j],
                    starts,
                    0,
                    x,
                    ybar,
                    root,
                    z,
                    kept[j],
                    kept_values[j],
                    space,
                )
                deviance[j] = lowest - objective_fit
                if deviance[j] < -ABOVE_FIT:
                    return math.nan, math.nan
                continue
            grid_values(
                linear,
                q,
                usable,
                inverses,
                betas,
                sums,
                sizes,
                change[:, j],
                row,
                values,
            )
            minima = grid_minima(values, inner)
            # The neighbouring dose's optima, or the fit at the first dose, while the
            # profile stays within the margin of the threshold; beyond, only the
            # grid's minima within the margin are refined, and they start the
            # continuation again.
            n_from = n_kept[previous] if previous >= 0 else 1
            if previous >= 0 and deviance[previous] > threshold + GRID_MARGIN:
                n_from = 0
            n_starts = 0
            for i in range(n_from):
                start = kept[previous, i] if previous >= 0 else phi_fit
                for c in range(r):
                    starts[n_starts, c] = start[c]
                n_starts += 1
            for i in range(minima.size):
                point = minima[i]
                if point < 0:
                    continue
                if values[point] - objective_fit > threshold + GRID_MARGIN:
                    continue
                for c in range(r):
                    starts[n_starts, c] = points[point, c]
                n_starts += 1
            lowest, n_kept[j] = profile_at(
                model,
                row,
                scan[j],
                starts,
                n_starts,
                x,
                ybar,
                root,
                z,
                kept[j],
                kept_values[j],
                space,
            )
            for point in range(n_points):
                lowest = min(lowest, values[point])
            deviance[j] = lowest - objective_fit
            if deviance[j] < -ABOVE_FIT:
                return math.nan, math.nan
            previous = j
    bounds = np.full(2, math.nan)
    for side in range(2):
        # The region's farthest dose of the scan on this side, or with piece the last
        # dose inside it before the first outside, and the dose beyond it.
        inside = -1
        if side == 0 and piece:
            for j in range(below, -1, -1):
                if deviance[j] > threshold:
                    break
                inside = j
        elif side == 0:
            for j in range(0, below + 1):
                if deviance[j] <= threshold:
                    inside = j
                    break
        elif piece:
            for j in range(below + 1, n_scan):
                if deviance[j] > threshold:
                    break
                inside = j
        else:
            for j in range(n_scan - 1, below, -1):
                if deviance[j] <= threshold:
                    inside = j
                    break
        if side == 0:
            if inside == 0:
                bounds[0] = 0.0
                continue
            outside = inside - 1 if inside > 0 else below
        else:
            if inside == n_scan - 1:
                bounds[1] = math.inf
                continue
            outside = inside + 1 if inside >= 0 else below + 1
        if outside < 0 or outside >= n_scan:
            bounds[side] = 0.0 if side == 0 else math.inf
            continue
        bounds[side] = _bound_root(
            model,
            row,
            x,
            ybar,
            root,
            z,
            phi_fit,
            scan,
            deviance,
            kept,
            n_kept,
            inside,
            outside,
            space,
        )
    return bounds[0], bounds[1]


@jit_numpy_errors
def _bound_root(
    model: tuple,  # type: ignore[type-arg]
    row: FloatArray,
    x: FloatArray,
    ybar: FloatArray,
    root: FloatArray,
    z: FloatArray,
    phi_fit: FloatArray,
    scan: FloatArray,
    deviance: FloatArray,
    kept: FloatArray,
    n_kept: IntArray,
    inside: int,
    outside: int,
    space: tuple,  # type: ignore[type-arg]
) -> float:
    """The dose between a dose of the scan outside the region and the one inside it
    (the BMD when ``inside`` is -1) at which the deviance crosses the threshold, by
    Brent's method on ln dose, the profile refined from the optima of both ends."""
    r = model[3].size
    objective_fit, threshold = row[OBJECTIVE_FIT], row[THRESHOLD]
    b_in = row[BMD] if inside < 0 else scan[inside]
    d_in = 0.0 if inside < 0 else deviance[inside]
    state = np.zeros(8)
    root_start(
        state,
        math.log(scan[outside]),
        deviance[outside] - threshold,
        math.log(b_in),
        d_in - threshold,
    )
    starts = np.zeros((2 * MAX_CARRIED, max(r, 1)))
    n_starts = 0
    for j in (outside, inside):
        if j >= 0:
            for i in range(n_kept[j]):
                for c in range(r):
                    starts[n_starts, c] = kept[j, i, c]
                n_starts += 1
        else:
            for c in range(r):
                starts[n_starts, c] = phi_fit[c]
            n_starts += 1
    found = np.zeros((MAX_CARRIED, max(r, 1)))
    found_values = np.zeros(MAX_CARRIED)
    for _ in range(100):  # pragma: no branch - Brent's method stops far sooner
        position, done = root_point(state, ROOT_TOLERANCE)
        if done:
            break
        best, _ = profile_at(
            model,
            row,
            math.exp(position),
            starts,
            n_starts,
            x,
            ybar,
            root,
            z,
            found,
            found_values,
            space,
        )
        root_update(state, position, best - objective_fit - threshold)
    return math.exp(state[1])


@jit_numpy_errors
def _bounds_range(
    model: tuple,  # type: ignore[type-arg]
    points: FloatArray,
    inner: int,
    usable: BoolArray,
    bases: FloatArray,
    scan: FloatArray,
    change: FloatArray,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    numbers: FloatArray,
    phi_fit: FloatArray,
    bounds: FloatArray,
    start: int,
    stop: int,
) -> None:
    for g in range(start, stop):
        root = roots[0] if roots.shape[0] == 1 else roots[g]
        bounds[g, 0], bounds[g, 1] = gene_bounds(
            model,
            points,
            inner,
            usable,
            bases,
            scan,
            change,
            x,
            means[g],
            root,
            numbers[g],
            phi_fit[g],
        )


@jit(parallel=True)
def bounds_parallel(
    model: tuple,  # type: ignore[type-arg]
    points: FloatArray,
    inner: int,
    usable: BoolArray,
    bases: FloatArray,
    scan: FloatArray,
    change: FloatArray,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    numbers: FloatArray,
    phi_fit: FloatArray,
    bounds: FloatArray,
    n_runs: int,
) -> None:
    """The bounds of every gene, ``(n_genes, 2)``, in ``n_runs`` runs of genes on
    numba's threads."""
    n = means.shape[0]
    for run in prange(n_runs):
        _bounds_range(
            model,
            points,
            inner,
            usable,
            bases,
            scan,
            change,
            x,
            means,
            roots,
            numbers,
            phi_fit,
            bounds,
            run * n // n_runs,
            (run + 1) * n // n_runs,
        )


@jit
def bounds_serial(
    model: tuple,  # type: ignore[type-arg]
    points: FloatArray,
    inner: int,
    usable: BoolArray,
    bases: FloatArray,
    scan: FloatArray,
    change: FloatArray,
    x: FloatArray,
    means: FloatArray,
    roots: FloatArray,
    numbers: FloatArray,
    phi_fit: FloatArray,
    bounds: FloatArray,
) -> None:
    """The bounds of every gene on the calling thread."""
    _bounds_range(
        model,
        points,
        inner,
        usable,
        bases,
        scan,
        change,
        x,
        means,
        roots,
        numbers,
        phi_fit,
        bounds,
        0,
        means.shape[0],
    )
