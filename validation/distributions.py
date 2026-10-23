"""Validate the special functions and distributions against mpmath at 50 digits.

Checks every public function of ``cytepool.stats.special`` and
``cytepool.stats.distributions`` against mpmath evaluated at the exact binary values of
the inputs, with 50 significant digits (70 for intermediate steps); quantile references
are roots of the 70-digit tail probabilities. The grids cover deep
tails (probabilities down to 1e-300, and log probabilities far below), shapes from 1e-3
to 1e6 (incomplete gamma shapes to 1e18 above the center), degrees of freedom from 0.5
to 1e12 and infinity, hypergeometric populations up to 1e13, and the hard cases where
textbook methods and scipy lose accuracy. scipy, when installed, is evaluated on the
same points for information only.

Requirements:
    Python: numpy and numba (for CytePool) and mpmath. scipy is optional.
    R: none.

Run from the repository root, against the working tree::

    PYTHONPATH=. python validation/distributions.py
    PYTHONPATH=. python validation/distributions.py --write-fixtures

The reference values are computed by a separate process, this script run as
``python validation/distributions.py --reference DIRECTORY``, which reads
``inputs.npz`` from the directory and writes ``reference.npz``, ``scipy.npz``,
``versions.json`` and ``notes.json`` back; it imports mpmath and scipy but not CytePool.
The incomplete beta reference is DLMF 8.17.22 evaluated in 70-digit arithmetic, checked
against mpmath's own ``betainc`` where that converges quickly; the hypergeometric
reference sums exact ratios in fixed-point integer arithmetic. The script prints
the reference versions and a markdown table, and exits with status 1 when any check
exceeds its tolerance. With ``--write-fixtures`` it writes a few hundred points per
family, the hardest included, with their reference values to tests/data/distributions/.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "distributions"
)
SEED = 20261022
FIXTURE_POINTS = 240

# Tolerances on the relative error. The distribution functions and the incomplete gamma
# and beta functions carry an accuracy contract of 1e-12, tighter than the "exact" tier
# (1e-10). The gamma-function family is expected to be within a few units in the last
# place.
CONTRACT = 1e-12
NEAR_MACHINE = 1e-15
# Probabilities below this are checked through their logarithm.
SMALLEST_CHECKED = 1e-300
# The tests check these quantiles through the tails, which have fixtures, and closed
# forms; their families get no fixtures of their own.
UNFIXTURED = {"f_quantile", "chi2_quantile"}


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------


def _log_grid(low: float, high: float, count: int) -> np.ndarray:
    return 10.0 ** np.linspace(low, high, count)


def build_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the input arrays by family, and the indices of each family's hard cases."""
    rng = np.random.default_rng(SEED)
    inputs: dict[str, np.ndarray] = {}
    hard: dict[str, np.ndarray] = {}

    # Gamma-function family. Points near the zeros of lgamma (1, 2 and, for x < 0, where
    # |Gamma(x)| = 1), near the positive root of digamma and its negative roots, and near
    # the poles.
    near = 10.0 ** -np.linspace(1, 15, 15)
    positive = np.concatenate(
        [
            _log_grid(-300, 300, 121),
            np.linspace(0.01, 12.0, 240),
            1.0 + near,
            1.0 - near,
            2.0 + near,
            2.0 - near,
            10.0 ** rng.uniform(-6, 6, 200),
        ]
    )
    digamma_root = 1.4616321449683622
    root_points = np.concatenate(
        [digamma_root * (1.0 + near), digamma_root * (1.0 - near), [digamma_root]]
    )
    negative = -np.linspace(0.01, 30.0, 300)
    poles = np.concatenate([-np.arange(0.0, 6.0) - 1e-9, -np.arange(1.0, 6.0) + 1e-9])
    lgamma_zeros = np.array(
        [-2.457024738220800, -2.747682646727412, -3.143632561447428]
    )
    digamma_zeros = np.array(
        [-0.5040830082644554, -1.5734984731623904, -2.6107208684441446]
    )
    negative = np.concatenate(
        [
            negative,
            poles,
            lgamma_zeros * (1 + 1e-12),
            lgamma_zeros,
            digamma_zeros,
            digamma_zeros * (1 + 1e-12),
        ]
    )
    negative = negative[negative != np.floor(negative)]
    inputs["lgamma__x"] = np.concatenate([positive, negative])
    inputs["digamma__x"] = np.concatenate([positive, root_points, negative])
    inputs["trigamma__x"] = np.concatenate([positive, negative])

    inputs["trigamma_inverse__y"] = np.concatenate(
        [
            _log_grid(-12, 16, 281),
            [1e-300, 1e-200, 1e-100, 1e-20, 9.99e-5, 1e-4, 1.001e-4, 9.99, 10.0, 10.01],
            [9.99e31, 1e32, 1.001e32, 1e100, 1e200, 1e300],
        ]
    )

    grid = _log_grid(-3, 9, 25)
    a, b = np.meshgrid(np.concatenate([grid, [9.99, 10.0, 10.01]]), grid)
    random_ab = 10.0 ** rng.uniform(-3, 7, (200, 2))
    inputs["log_beta__a"] = np.concatenate([a.ravel(), random_ab[:, 0]])
    inputs["log_beta__b"] = np.concatenate([b.ravel(), random_ab[:, 1]])

    # Incomplete gamma: for each shape, arguments around the mean in units of the standard
    # deviation, a log grid down to 1e-300, the branch points and the deep upper tail.
    shapes = np.unique(
        np.concatenate(
            [_log_grid(-3, 6, 19), [0.5, 0.999, 1.0, 1.5, 9.99, 10.0, 10.01]]
        )
    )
    points = [(1e4, 6737.69), (1e6, 1.03751e6), (1e-3, 1.0), (1e-3, 1.5), (0.5, 1e-300)]
    for shape in shapes:
        sd = math.sqrt(shape)
        xs = {shape + z * sd for z in (-38, -20, -8, -2, -0.5, 0, 0.5, 2, 8, 20, 38)}
        xs |= set(_log_grid(-300, 3, 12))
        xs |= {shape + 1.0, shape + 0.999, 1.5, 1.5001, 700.0}
        xs.add(1.5 * shape + 30 * sd + 800)  # deep upper tail
        points += [(shape, x) for x in sorted(xs) if x > 0]
    # Huge shapes, where the uniform expansion serves the center: the upper side up to
    # 1e18 and both sides at 1e8.
    first_huge = len(points)
    for shape in [1e8, 1e10, 1e12, 1e15, 1e18]:
        sd = math.sqrt(shape)
        points += [(shape, shape + z * sd) for z in (0.3, 2.0, 8.0, 30.0, 40.0)]
    points += [(1e8, 1e8 + z * 1e4) for z in (-30.0, -8.0, -2.0, -0.3)]
    gamma_points = np.array(points)
    inputs["gammainc__a"], inputs["gammainc__x"] = gamma_points.T
    hard["gammainc"] = np.r_[np.arange(5), np.arange(first_huge, len(points))]

    # Incomplete beta: a grid of shapes, with arguments around the mean in units of the
    # standard deviation, near 0 and 1, and on both sides of the branch point
    # (a + 1) / (a + b + 2).
    values = [1e-3, 0.02, 0.5, 1.0, 3.0, 9.99, 10.0, 30.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    points = [
        (1e6, 1e6, 0.489322),
        (670.479, 26.2854, 0.328878),
        (1e5, 1e6, 0.0994142),
        (5e5, 0.5, 1.0 - 1e-5),
        (1e-3, 1e-3, 0.3),
    ]
    for a_ in values:
        for b_ in values:
            mean = a_ / (a_ + b_)
            sd = math.sqrt(a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1.0)))
            xs = {mean + z * sd for z in (-37, -20, -6, -1, 0, 1, 6, 20, 37)}
            xs |= {1e-300, 1e-100, 1e-20, 1e-5, 0.3, 0.7}
            xs |= {1.0 - 1e-16, 1.0 - 1e-10, 1.0 - 1e-5}
            branch = (a_ + 1.0) / (a_ + b_ + 2.0)
            xs |= {branch * (1.0 - 1e-9), branch * (1.0 + 1e-9)}
            points += [(a_, b_, x) for x in sorted(xs) if 0.0 < x < 1.0]
    beta_points = np.array(points)
    inputs["betainc__a"], inputs["betainc__b"], inputs["betainc__x"] = beta_points.T
    hard["betainc"] = np.arange(5)

    # Normal distribution and quantiles.
    tails = np.concatenate([np.linspace(36.5, 38.6, 43), _log_grid(1, 150, 60)])
    inputs["norm__x"] = np.concatenate([np.linspace(-40.0, 40.0, 801), tails, -tails])
    p_low = np.concatenate(
        [10.0 ** -np.linspace(0.31, 300, 300), [1e-305, 1e-310, 1e-320, 5e-324]]
    )
    inputs["norm_quantile__p"] = np.concatenate(
        [
            p_low,
            1.0 - 10.0 ** -np.linspace(0.31, 16, 60),
            [0.5, 0.075, 0.07499999999999, 0.925, 0.92500000000001, 0.4999999],
            rng.uniform(0.0, 1.0, 100),
        ]
    )

    # Student t, including non-integer and infinite degrees of freedom, x^2 overflowing,
    # and the hard cases.
    dfs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 7.3, 10.0, 19.99, 20.0, 30.0, 50.0]
    dfs += [100.0, 1e3, 1e4, 1e5, 1e6, 3e7, 1e12, math.inf]
    xs_t = [0.0, 1e-300, 1e-10, 0.1, 0.5, 0.99, 1.0, 1.01, 1.5, 1.7320508, 2.0, 3.0]
    xs_t += [5.0, 10.0, 20.0, 37.0, 38.5, 50.0, 100.0, 1e3, 1e5, 1e10, 1e50, 1e100]
    xs_t += [1e154, 1e160, 1e300]
    points = [(3.756e6, 50.0), (2.0, 3e7), (-38.0, math.inf), (1e200, 1.5)]
    points += [
        (s * x, df) for df in dfs for x in xs_t for s in (1.0, -1.0) if x > 0 or s > 0
    ]
    t_points = np.array(points)
    inputs["t__x"], inputs["t__df"] = t_points.T
    hard["t"] = np.arange(4)

    dfs_q = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4]
    dfs_q += [1e5, 1e6, 3e7, 1e8, 1e12, math.inf]
    ps = [1e-300, 1.13e-277, 1e-250, 1e-150, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5]
    ps += [1e-3, 0.01, 0.05, 0.1, 0.25, 0.4, 0.49, 0.499999, 0.5, 0.6, 0.9, 0.99]
    ps += [1.0 - 1e-10]
    points = [(1e-300, 4.5), (1.13e-277, 1.5)]
    points += [(p, df) for df in dfs_q for p in ps]
    # small degrees of freedom deep in the tail, where the quantile nears the largest double
    first_small = len(points)
    for df in [0.5, 0.55, 0.6, 0.75, 0.9, 1.25, 1.5, 1.75]:
        points += [(10.0**-k, df) for k in range(60, 301, 20)]
    points += [(1e-111, 0.5), (3.901252767619372e-186, 0.6), (0.26, 0.001)]
    small_block = np.arange(first_small, len(points))
    # random pairs, log-uniform in p (down to 1e-300) and in df (0.5 to 1e12)
    random_p = 10.0 ** rng.uniform(-300, np.log10(0.5), 200)
    random_df = 10.0 ** rng.uniform(np.log10(0.5), 12, 200)
    points += list(zip(random_p, random_df))
    tq_points = np.array(points)
    inputs["t_quantile__p"], inputs["t_quantile__df"] = tq_points.T
    # the reported cases, and one point in four of the small-df block
    hard["t_quantile"] = np.r_[np.arange(2), small_block[::4], small_block[-3:]]

    # F, including infinite denominator (and numerator) degrees of freedom.
    dfs_f = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e3, 1e5, 1e6, math.inf]
    xs_f = [1e-300, 1e-10, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0, 1e5, 1e16]
    xs_f += [1e100]
    points = [(2.894e16, 3.0, 30.0), (1.02, 30.0, 1e5), (0.98, 1e6, 1e6)]
    points += [(1e300, 3.0, 1e6)]
    points += [(x, d1, d2) for d1 in dfs_f for d2 in dfs_f for x in xs_f]
    first_large = len(points)
    large = [1e9, 1e12]
    points += [
        (x, d1, d2) for d1 in large for d2 in [3.0, 30.0, 1e6, 1e12] for x in xs_f
    ]
    points += [(x, d1, d2) for d1 in [3.0, 30.0] for d2 in large for x in xs_f]
    points += [(1e-300, 0.5, 1e12)]
    f_points = np.array(points)
    inputs["f__x"], inputs["f__dfn"], inputs["f__dfd"] = f_points.T
    hard["f"] = np.r_[
        np.arange(4), np.arange(first_large, len(points), 7), len(points) - 1
    ]

    # Chi-squared.
    points = [(1264.0, 1.0), (3.84, 1.0), (1e-300, 0.5)]
    for df in [0.5, 1.0, 2.0, 3.0, 3.7, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4, 1e6]:
        sd = math.sqrt(2.0 * df)
        xs = {df + z * sd for z in (-30, -10, -3, -1, 0, 1, 3, 10, 30, 60)}
        xs |= set(_log_grid(-300, 4, 9))
        points += [(x, df) for x in sorted(xs) if x > 0]
    first_large = len(points)
    for df in [1e9, 1e12]:
        sd = math.sqrt(2.0 * df)
        points += [(df + z * sd, df) for z in (0.3, 3.0, 30.0)]
    chi2_points = np.array(points)
    inputs["chi2__x"], inputs["chi2__df"] = chi2_points.T
    hard["chi2"] = np.r_[np.arange(3), np.arange(first_large, len(points))]

    # Gamma distribution with a scale.
    points = []
    for shape in [1e-3, 0.5, 2.0, 37.0, 1e4]:
        for scale in [1e-3, 0.37, 1.0, 25.0, 1e6]:
            mean = shape * scale
            sd = math.sqrt(shape) * scale
            xs = {mean + z * sd for z in (-30, -5, -1, 0, 1, 5, 30)}
            xs |= {scale * 1e-200, scale * 1e-5, scale * 700.0}
            points += [(x, shape, scale) for x in sorted(xs) if x > 0]
    gamma_dist_points = np.array(points)
    inputs["gamma__x"], inputs["gamma__shape"], inputs["gamma__scale"] = (
        gamma_dist_points.T
    )

    # Beta distribution, including arguments outside [0, 1].
    xs_b = [-1.0, 0.0, 1e-10, 0.1, 0.5, 0.9, 1.0 - 1e-10, 1.0, 2.0]
    shapes_b = [0.5, 2.0, 37.0, 1e4]
    beta_dist = np.array(
        [(x, a_, b_) for a_ in shapes_b for b_ in shapes_b for x in xs_b]
    )
    inputs["beta__x"], inputs["beta__a"], inputs["beta__b"] = beta_dist.T

    # Hypergeometric: random tables up to 30,000 items and boundary cases.
    points = [
        (5, 30000, 1000, 50),
        (200, 30000, 1000, 500),
        (9, 30000, 15000, 10),
        (7499, 30000, 15000, 15000),
        (0, 1, 0, 0),
        (0, 1, 1, 1),
        (-1, 0, 0, 0),
        (3, 20, 7, 12),
    ]
    for _ in range(600):
        M = int(10 ** rng.uniform(0, 4.5))
        n = int(rng.integers(0, M + 1))
        N = int(rng.integers(0, M + 1))
        low, high = max(0, N - (M - n)), min(n, N)
        points.append((int(rng.integers(low - 1, high + 1)), M, n, N))
    # Large populations: k M and n N exceed 2^53, and near the mode the tails need up to
    # millions of terms whose ratios are close to 1. Points at z standard deviations
    # from the mode; for populations up to 1e12, those at z = -30, 0, 10 and 35 go
    # into the fixtures.
    hard_hyper = list(range(4))
    for M in (10**9, 10**11, 10**12, 10**13):
        for n, N in ((M // 2, M // 2), (3 * M // 10, M // 10), (M // 100, M // 2)):
            if M == 10**13 and n != M // 2:
                continue
            sd = math.sqrt(N * (n / M) * (1 - n / M) * (M - N) / (M - 1))
            mode = (N + 1) * (n + 1) // (M + 2)
            for z in (-30.0, -2.0, 0.0, 1.0, 10.0, 35.0):
                if M <= 10**12 and z not in (-2.0, 1.0):
                    hard_hyper.append(len(points))
                points.append((mode - 1 + int(z * sd), M, n, N))
    for _ in range(60):
        M = int(10 ** rng.uniform(4.5, 12.0))
        n = int(rng.integers(1, M))
        N = int(rng.integers(1, M))
        sd = math.sqrt(N * (n / M) * (1 - n / M) * (M - N) / (M - 1))
        mode = (N + 1) * (n + 1) // (M + 2)
        low, high = max(0, N - (M - n)), min(n, N)
        k = mode + int(rng.uniform(-40.0, 40.0) * sd)
        points.append((min(max(k, low - 1), high), M, n, N))
    hyper = np.array(points, dtype=float)
    keys = ("hypergeom__k", "hypergeom__M", "hypergeom__n", "hypergeom__N")
    for key, column in zip(keys, hyper.T):
        inputs[key] = column
    hard["hypergeom"] = np.array(hard_hyper)

    # Realistic batches: 2,000 genes with moderated t statistics on 8.17 degrees of freedom
    # and quasi-likelihood F statistics on (2, 37.4) degrees of freedom; one gene in ten
    # carries a strong effect.
    effect = np.where(rng.uniform(size=2000) < 0.1, rng.uniform(4, 30, 2000), 0.0)
    inputs["t_genes__x"] = rng.standard_t(8.17, 2000) + effect * rng.choice(
        [-1, 1], 2000
    )
    inputs["t_genes__df"] = np.full(2000, 8.17)
    inputs["f_genes__x"] = rng.f(2.0, 37.4, 2000) * (1.0 + effect**2)
    inputs["f_genes__dfn"] = np.full(2000, 2.0)
    inputs["f_genes__dfd"] = np.full(2000, 37.4)

    # F and chi-squared quantiles, at probabilities whose quantiles are normal doubles:
    # a tail holding p falls like p^(2 / df), so p is kept above 10^(-140 min(df, 2)).
    ps_q = [1e-300, 1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3]
    ps_q += [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.9, 0.99, 1.0 - 1e-10]
    dfs_q = [0.5, 1.0, 3.0, 10.0, 100.0, 1e5, math.inf]

    def representable(p: float, *dfs: float) -> bool:
        return p >= 10.0 ** (-140.0 * min(*dfs, 2.0))

    points = [
        (p, d1, d2)
        for d1 in dfs_q
        for d2 in dfs_q
        for p in ps_q
        if not (math.isinf(d1) and math.isinf(d2)) and representable(p, d1, d2)
    ]
    random_p = 10.0 ** rng.uniform(-150, np.log10(0.5), 150)
    random_dfs = 10.0 ** rng.uniform(0, 6, (150, 2))
    points += [(p, d1, d2) for p, (d1, d2) in zip(random_p, random_dfs)]
    keys = ("f_quantile__p", "f_quantile__dfn", "f_quantile__dfd")
    for key, column in zip(keys, np.array(points).T):
        inputs[key] = column
    points = [
        (p, df)
        for df in [0.5, 1.0, 2.0, 3.7, 10.0, 100.0, 1e4, 1e6]
        for p in ps_q
        if representable(p, df)
    ]
    points += list(
        zip(10.0 ** rng.uniform(-150, np.log10(0.5), 50), 10.0 ** rng.uniform(0, 6, 50))
    )
    inputs["chi2_quantile__p"], inputs["chi2_quantile__df"] = np.array(points).T
    return inputs, hard


# ---------------------------------------------------------------------------------------
# Reference program (run in a separate process; imports mpmath and scipy, not CytePool)
# ---------------------------------------------------------------------------------------


def reference_main(directory: Path) -> None:
    """Compute the mpmath references (and scipy, if installed) for the inputs in directory."""
    import mpmath as mp

    mp.mp.dps = 50
    work = 70
    no_convergence = mp.libmp.libhyper.NoConvergence
    inputs = dict(np.load(directory / "inputs.npz"))

    def ibeta_fraction(A: Any, B: Any, X: Any, Y: Any) -> Any:
        """I_X(A, B) for X < (A + 1) / (A + B + 2) from DLMF 8.17.22.

        The plain continued fraction is evaluated backward in 70-digit arithmetic, with
        the depth doubled until two values agree to 55 digits.
        """

        def value(depth: int) -> Any:
            t = mp.mpf(1)
            for m in range(depth, 0, -1):
                k = m // 2
                if m % 2 == 0:
                    d = k * (B - k) * X / ((A + 2 * k - 1) * (A + 2 * k))
                else:
                    d = -(A + k) * (A + B + k) * X / ((A + 2 * k) * (A + 2 * k + 1))
                t = 1 + d / t
            return 1 / t

        depth = 32
        previous = value(depth)
        while True:
            depth *= 2
            current = value(depth)
            if abs(current - previous) <= mp.mpf(10) ** -55 * abs(current):
                break
            previous = current
        log_beta = mp.loggamma(A) + mp.loggamma(B) - mp.loggamma(A + B)
        log_pre = A * mp.log(X) + B * mp.log(Y) - mp.log(A) - log_beta
        return mp.exp(log_pre) * current

    def ibeta_pair(A: Any, B: Any, X: Any, Y: Any) -> tuple[Any, Any]:
        """I_X(A, B) and 1 - I_X(A, B) for Y = 1 - X.

        The tail on the side where the fraction converges comes from it directly and the
        other as its complement, which at 70 digits keeps far more than 50.
        """
        if X == 0:
            return mp.mpf(0), mp.mpf(1)
        if Y == 0:
            return mp.mpf(1), mp.mpf(0)
        with mp.workdps(work):
            if X * (A + B + 2) < A + 1:
                lower = ibeta_fraction(A, B, X, Y)
                return lower, 1 - lower
            upper = ibeta_fraction(B, A, Y, X)
            return 1 - upper, upper

    def gamma_upper_fraction(A: Any, X: Any) -> Any:
        # Legendre's continued fraction (DLMF 8.9.2), evaluated backward with the depth
        # doubled until two values agree to 55 digits.
        def value(depth: int) -> Any:
            t = X + 2 * depth + 1 - A
            for j in range(depth, 0, -1):
                t = X + 2 * j - 1 - A - j * (j - A) / t
            return t

        depth = 64
        previous = value(depth)
        while True:
            depth *= 2
            current = value(depth)
            if abs(current / previous - 1) < mp.mpf(10) ** -55:
                break
            previous = current
        return mp.exp(A * mp.log(X) - X - mp.loggamma(A)) / current

    def gamma_pair(A: Any, X: Any) -> tuple[Any, Any]:
        """P(A, X) and Q(A, X), each computed directly."""
        if X == 0:
            return mp.mpf(0), mp.mpf(1)
        with mp.workdps(work):
            if A >= 1e9:
                # Shapes beyond 1e9 are checked on the upper side only, where Legendre's
                # fraction converges quickly (the lower series would need ~8 sqrt(A) terms).
                assert X > A + 1, (
                    "the reference covers huge shapes above the center only"
                )
                upper = gamma_upper_fraction(A, X)
                return 1 - upper, upper
            near_center = A >= 1e4 and abs(X - A) < 60 * mp.sqrt(A)
            if not near_center:
                try:
                    lower = mp.gammainc(A, 0, X, regularized=True)
                    upper = mp.gammainc(A, X, mp.inf, regularized=True)
                    return lower, upper
                except (no_convergence, ValueError):  # mpmath gave up
                    pass
            if X < A + 1:
                # P = X^A e^-X / Gamma(A + 1) 1F1(1; A + 1; X) (DLMF 8.7.1)
                log_pre = A * mp.log(X) - X - mp.loggamma(A + 1)
                lower = mp.exp(log_pre) * mp.hyp1f1(1, A + 1, X, maxterms=10**7)
                return lower, 1 - lower
            upper = gamma_upper_fraction(A, X)
            return 1 - upper, upper

    def logs(lower: Any, upper: Any) -> tuple[float, float]:
        """Logarithms of two complementary probabilities, each through the smaller one."""
        with mp.workdps(work):
            minus_inf = mp.mpf("-inf")
            log_lower = (
                mp.log1p(-upper)
                if upper < lower
                else (mp.log(lower) if lower > 0 else minus_inf)
            )
            log_upper = (
                mp.log1p(-lower)
                if lower < upper
                else (mp.log(upper) if upper > 0 else minus_inf)
            )
        return float(log_lower), float(log_upper)

    def normal_upper(T: Any) -> Any:
        """P(Z > T) for T >= 0; mpmath's ncdf overflows internally beyond about 1e150."""
        if T < 1e100:
            return mp.ncdf(-T)
        # Asymptotic series of Mills' ratio (DLMF 7.12.1); the omitted terms are far
        # below 1e-70 relative at this size.
        T2 = T * T
        return mp.npdf(T) / T * (1 - 1 / T2 + 3 / T2**2 - 15 / T2**3)

    def normal_tails(T: Any) -> tuple[Any, Any]:
        """P(Z <= T) and P(Z > T), each computed directly."""
        if T >= 0:
            upper = normal_upper(T)
            return (1 - upper if T >= 1e100 else mp.ncdf(T)), upper
        lower = normal_upper(-T)
        return lower, (1 - lower if -T >= 1e100 else mp.ncdf(-T))

    def t_tails(x: float, df: float) -> tuple[Any, Any]:
        T = mp.mpf(x)
        with mp.workdps(work):
            if math.isinf(df):
                return normal_tails(T)
            NU = mp.mpf(df)
            W = NU / (NU + T * T)
            V = T * T / (NU + T * T)
            both, inner = ibeta_pair(NU / 2, mp.mpf(1) / 2, W, V)
            if T >= 0:
                return (1 + inner) / 2, both / 2
            return both / 2, (1 + inner) / 2

    def f_tails(x: float, dfn: float, dfd: float) -> tuple[Any, Any]:
        with mp.workdps(work):
            F = mp.mpf(x)
            if F <= 0:
                return mp.mpf(0), mp.mpf(1)
            if math.isinf(dfn) and math.isinf(dfd):
                return (mp.mpf(0), mp.mpf(1)) if x < 1 else (mp.mpf(1), mp.mpf(0))
            if math.isinf(dfd):
                A = mp.mpf(dfn) / 2
                return gamma_pair(A, A * F)
            if math.isinf(dfn):
                A = mp.mpf(dfd) / 2
                lower, upper = gamma_pair(A, A / F)
                return upper, lower
            D1, D2 = mp.mpf(dfn), mp.mpf(dfd)
            W = D2 / (D2 + D1 * F)
            V = D1 * F / (D2 + D1 * F)
            sf, cdf = ibeta_pair(D2 / 2, D1 / 2, W, V)
            return cdf, sf

    def gamma_mass(A: Any, X: Any) -> Any:
        """X g(X) for the density g of the gamma distribution with shape A, scale 1."""
        return mp.exp(A * mp.log(X) - X - mp.loggamma(A))

    def f_mass(x: Any, dfn: float, dfd: float) -> Any:
        """x f(x) for the density f of the F distribution."""
        with mp.workdps(work):
            if math.isinf(dfd):
                return gamma_mass(mp.mpf(dfn) / 2, mp.mpf(dfn) / 2 * x)
            if math.isinf(dfn):
                return gamma_mass(mp.mpf(dfd) / 2, mp.mpf(dfd) / 2 / x)
            A, B = mp.mpf(dfn) / 2, mp.mpf(dfd) / 2
            log_beta = mp.loggamma(A) + mp.loggamma(B) - mp.loggamma(A + B)
            V = A * x / (B + A * x)
            return mp.exp(A * mp.log(V) + B * mp.log(B / (B + A * x)) - log_beta)

    def tail_quantile(
        tails: Callable[[Any], tuple[Any, Any, Any]], p: float, upper: bool
    ) -> float:
        """The x > 0 with P(X > x) = p if upper, else P(X <= x) = p.

        tails(x) returns P(X <= x), P(X > x) and x f(x) for the density f. The tail
        holding min(p, 1 - p) is solved for u = log x by Newton's method on its
        logarithm, which is monotone in u, inside a bracket found by steps that double
        from u = 0; a step that leaves the bracket is replaced by bisection.
        """
        with mp.workdps(work):
            P = mp.mpf(p)
            if P > mp.mpf(1) / 2:
                P, upper = 1 - P, not upper
            target = mp.log(P)

            def h_and_slope(u: Any) -> tuple[Any, Any]:
                lower_tail, upper_tail, mass = tails(mp.exp(u))
                tail = upper_tail if upper else lower_tail
                return mp.log(tail) - target, (-mass if upper else mass) / tail

            # h decreases in u for the upper tail and increases for the lower one
            a, ha = mp.mpf(0), h_and_slope(mp.mpf(0))[0]
            direction = 1 if (ha > 0) == upper else -1
            step = mp.mpf(1)
            b, hb = a + direction * step, h_and_slope(a + direction * step)[0]
            while (ha > 0) == (hb > 0):
                a, ha = b, hb
                step *= 2
                b, hb = a + direction * step, h_and_slope(a + direction * step)[0]
            low, high = min(a, b), max(a, b)
            u = (low + high) / 2
            for _ in range(500):
                value, slope = h_and_slope(u)
                if (value > 0) == upper:
                    low = u
                else:
                    high = u
                u_new = u - value / slope
                if not (low < u_new < high):
                    u_new = (low + high) / 2
                if abs(u_new - u) < mp.mpf(10) ** -55 * max(1, abs(u)):
                    return float(mp.exp(u_new))
                u = u_new
            raise RuntimeError(f"quantile reference did not converge for p={p}")

    def t_upper_quantile(p: float, df: float) -> float:
        """The t > 0 with P(T > t) = p < 1/2: Newton's method in u = log t, kept inside a
        bracket and replaced by bisection when a step leaves it."""
        with mp.workdps(work):
            target = mp.log(mp.mpf(p))
            NU = mp.mpf(df)
            if not math.isinf(df):
                log_norm = (
                    mp.loggamma((NU + 1) / 2)
                    - mp.loggamma(NU / 2)
                    - mp.log(NU * mp.pi) / 2
                )

            def h_and_slope(u: Any) -> tuple[Any, Any]:
                # h(u) = log P(T > e^u) - log p and its derivative -t f(t) / P(T > t)
                t = mp.exp(u)
                if math.isinf(df):
                    sf, density = normal_upper(t), mp.npdf(t)
                else:
                    W = NU / (NU + t * t)
                    V = t * t / (NU + t * t)
                    sf = ibeta_pair(NU / 2, mp.mpf(1) / 2, W, V)[0] / 2
                    density = mp.exp(log_norm - (NU + 1) / 2 * mp.log1p(t * t / NU))
                return mp.log(sf) - target, -t * density / sf

            low, high = mp.mpf(-60), mp.mpf(2)
            while h_and_slope(high)[0] > 0:
                low, high = high, 2 * high
            while h_and_slope(low)[0] < 0:
                high, low = low, 2 * low
            u = (low + high) / 2
            for _ in range(500):
                value, slope = h_and_slope(u)
                if value > 0:
                    low = u
                else:
                    high = u
                u_new = u - value / slope
                if not (low < u_new < high):
                    u_new = (low + high) / 2
                if abs(u_new - u) < mp.mpf(10) ** -55 * max(1, abs(u)):
                    u = u_new
                    break
                u = u_new
            else:
                raise RuntimeError(f"t quantile did not converge for p={p}, df={df}")
            return float(mp.exp(u))

    def norm_quantile(p: float) -> float:
        P = mp.mpf(p)
        if P == mp.mpf(1) / 2:
            return 0.0
        with mp.workdps(work):
            upper = P > mp.mpf(1) / 2
            Q = 1 - P if upper else P
            target = mp.log(Q)
            z = mp.sqrt(-2 * target)
            for _ in range(200):
                c = normal_upper(z)
                step = -(mp.log(c) - target) * c / mp.npdf(z)
                z -= step
                if abs(step) < mp.mpf(10) ** -55 * max(1, abs(z)):
                    break
            else:
                raise RuntimeError(f"normal quantile reference did not converge, p={p}")
            return float(z if upper else -z)

    hypergeom_agreement: list[float] = []

    def log_binomial(a: int, b: int) -> Any:
        return mp.loggamma(a + 1) - mp.loggamma(b + 1) - mp.loggamma(a - b + 1)

    def hypergeom_tail(k: float, M: float, n: float, N: float) -> tuple[float, float]:
        """P(X > k) and its log.

        The first probability of the tail on the far side of the mode comes from exact
        binomial coefficients for populations up to 1e6, and from 70-digit log-gamma
        values above (also computed, as a check, up to 1e6). The tail is summed by the
        exact ratios of successive probabilities in 256-bit fixed point, until the rest,
        which log-concavity bounds by term r / (1 - r) for the last ratio r, is below
        2^-200 of the total; each step truncates by less than 2^-256 of the first term.
        """
        k_, M_, n_, N_ = int(k), int(M), int(n), int(N)
        low, high = max(0, N_ - (M_ - n_)), min(n_, N_)
        if k_ >= high:
            return 0.0, -math.inf
        if k_ < low:
            return 1.0, 0.0
        mode = (N_ + 1) * (n_ + 1) // (M_ + 2)
        upper_tail = k_ + 1 >= mode
        j = k_ + 1 if upper_tail else k_
        with mp.workdps(work):
            log_first = (
                log_binomial(n_, j)
                + log_binomial(M_ - n_, N_ - j)
                - log_binomial(M_, N_)
            )
            if M_ <= 10**6:
                exact = math.comb(n_, j) * math.comb(M_ - n_, N_ - j)
                log_exact = mp.log(mp.mpf(exact) / math.comb(M_, N_))
                difference = abs(log_first - log_exact) / max(1, abs(log_exact))
                hypergeom_agreement.append(float(difference))
                log_first = log_exact
            shift = 256
            term = total = 1 << shift
            while (j < high) if upper_tail else (j > low):
                if upper_tail:
                    num = (n_ - j) * (N_ - j)
                    den = (j + 1) * (M_ - n_ - N_ + j + 1)
                    j += 1
                else:
                    num = j * (M_ - n_ - N_ + j)
                    den = (n_ - j + 1) * (N_ - j + 1)
                    j -= 1
                term = term * num // den
                total += term
                # the stopping test needs no exactness, only the sums do
                if den > num and float(term) * num < 2.0**-200 * (den - num) * float(
                    total
                ):
                    break
            tail = mp.exp(log_first) * mp.ldexp(mp.mpf(total), -shift)
            sf = tail if upper_tail else 1 - tail
            if upper_tail:
                log_sf = log_first + mp.log(mp.ldexp(mp.mpf(total), -shift))
            else:
                log_sf = mp.log1p(-tail)
        return float(sf), float(log_sf)

    out: dict[str, np.ndarray] = {}
    clock = [time.perf_counter()]

    def store(
        family: str, names: tuple[str, ...], rows: list[tuple[float, ...]]
    ) -> None:
        columns = np.array(rows, dtype=float).reshape(len(rows), len(names))
        for name, column in zip(names, columns.T):
            out[f"{family}__{name}"] = column
        now = time.perf_counter()
        print(f"  {family}: {len(rows)} points in {now - clock[0]:.1f} s", flush=True)
        clock[0] = now

    store(
        "lgamma",
        ("value",),
        [(float(mp.re(mp.loggamma(x))),) for x in inputs["lgamma__x"]],
    )
    store(
        "digamma", ("value",), [(float(mp.digamma(x)),) for x in inputs["digamma__x"]]
    )
    store(
        "trigamma", ("value",), [(float(mp.psi(1, x)),) for x in inputs["trigamma__x"]]
    )

    rows = []
    for y in inputs["trigamma_inverse__y"]:
        with mp.workdps(work):
            Y = mp.mpf(y)
            x = 1 / Y + mp.mpf(1) / 2 if Y < 1 else 1 / mp.sqrt(Y)
            for _ in range(200):  # Newton's method on psi'(x) = y
                step = (mp.psi(1, x) - Y) / mp.psi(2, x)
                if x - step <= 0:
                    step = x / 2
                x -= step
                if abs(step) < mp.mpf(10) ** -55 * x:
                    break
            else:
                raise RuntimeError(
                    f"trigamma inverse reference did not converge, y={y}"
                )
        rows.append((float(x),))
    store("trigamma_inverse", ("value",), rows)

    rows = []
    for a_, b_ in zip(inputs["log_beta__a"], inputs["log_beta__b"]):
        with mp.workdps(work):
            A, B = mp.mpf(a_), mp.mpf(b_)
            rows.append((float(mp.loggamma(A) + mp.loggamma(B) - mp.loggamma(A + B)),))
    store("log_beta", ("value",), rows)

    rows = []
    for a_, x in zip(inputs["gammainc__a"], inputs["gammainc__x"]):
        lower, upper = gamma_pair(mp.mpf(a_), mp.mpf(x))
        rows.append((float(lower), float(upper), *logs(lower, upper)))
    store("gammainc", ("P", "Q", "logP", "logQ"), rows)

    rows = []
    agreement = []  # the continued fraction against mpmath's betainc, where it is fast
    for a_, b_, x in zip(
        inputs["betainc__a"], inputs["betainc__b"], inputs["betainc__x"]
    ):
        A, B, X = mp.mpf(a_), mp.mpf(b_), mp.mpf(x)
        with mp.workdps(work):
            lower, upper = ibeta_pair(A, B, X, 1 - X)
            if max(a_, b_) <= 1e3:
                # Compare on the side the fraction computes, where its argument is exact
                # (1 - X itself can round to 1 when X is tiny).
                if X * (A + B + 2) < A + 1:
                    other, mine = mp.betainc(A, B, 0, X, regularized=True), lower
                else:
                    other, mine = mp.betainc(B, A, 0, 1 - X, regularized=True), upper
                agreement.append(float(abs(other / mine - 1)) if mine else 0.0)
        rows.append((float(lower), float(upper), *logs(lower, upper)))
    store("betainc", ("I", "C", "logI", "logC"), rows)

    rows = []
    for x in inputs["norm__x"]:
        with mp.workdps(work):
            lower, upper = normal_tails(mp.mpf(x))
        rows.append((float(lower), float(upper), *logs(lower, upper)))
    store("norm", ("cdf", "sf", "logcdf", "logsf"), rows)

    store(
        "norm_quantile",
        ("ppf",),
        [(norm_quantile(p),) for p in inputs["norm_quantile__p"]],
    )

    def two_tail_rows(pairs: list[tuple[Any, Any]]) -> list[tuple[float, ...]]:
        return [(float(lo), float(up), *logs(lo, up)) for lo, up in pairs]

    pairs = [t_tails(x, df) for x, df in zip(inputs["t__x"], inputs["t__df"])]
    store("t", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))
    pairs = [
        t_tails(x, df) for x, df in zip(inputs["t_genes__x"], inputs["t_genes__df"])
    ]
    store("t_genes", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))

    rows = []
    for p, df in zip(inputs["t_quantile__p"], inputs["t_quantile__df"]):
        if p == 0.5:
            rows.append((0.0,))
        elif p < 0.5:
            rows.append((t_upper_quantile(p, df),))
        else:
            rows.append((-t_upper_quantile(1.0 - p, df),))  # 1 - p is exact here
    store("t_quantile", ("isf",), rows)

    pairs = [
        f_tails(*v) for v in zip(inputs["f__x"], inputs["f__dfn"], inputs["f__dfd"])
    ]
    store("f", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))
    genes = zip(inputs["f_genes__x"], inputs["f_genes__dfn"], inputs["f_genes__dfd"])
    store(
        "f_genes",
        ("cdf", "sf", "logcdf", "logsf"),
        two_tail_rows([f_tails(*v) for v in genes]),
    )

    pairs = []
    for x, df in zip(inputs["chi2__x"], inputs["chi2__df"]):
        with mp.workdps(work):
            pairs.append(gamma_pair(mp.mpf(df) / 2, mp.mpf(x) / 2))
    store("chi2", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))

    rows = []
    for p, dfn, dfd in zip(
        inputs["f_quantile__p"], inputs["f_quantile__dfn"], inputs["f_quantile__dfd"]
    ):

        def f_quantile_tails(x: Any, dfn: float = dfn, dfd: float = dfd) -> Any:
            return (*f_tails(x, dfn, dfd), f_mass(x, dfn, dfd))

        rows.append(
            (
                tail_quantile(f_quantile_tails, p, True),
                tail_quantile(f_quantile_tails, p, False),
            )
        )
    store("f_quantile", ("isf", "ppf"), rows)
    rows = []
    for p, df in zip(inputs["chi2_quantile__p"], inputs["chi2_quantile__df"]):

        def chi2_quantile_tails(x: Any, df: float = df) -> Any:
            A = mp.mpf(df) / 2
            return (*gamma_pair(A, x / 2), gamma_mass(A, x / 2))

        rows.append(
            (
                tail_quantile(chi2_quantile_tails, p, True),
                tail_quantile(chi2_quantile_tails, p, False),
            )
        )
    store("chi2_quantile", ("isf", "ppf"), rows)

    pairs = []
    for x, shape, scale in zip(
        inputs["gamma__x"], inputs["gamma__shape"], inputs["gamma__scale"]
    ):
        with mp.workdps(work):
            pairs.append(gamma_pair(mp.mpf(shape), mp.mpf(x) / mp.mpf(scale)))
    store("gamma", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))

    pairs = []
    for x, a_, b_ in zip(inputs["beta__x"], inputs["beta__a"], inputs["beta__b"]):
        X = mp.mpf(min(max(x, 0.0), 1.0))
        with mp.workdps(work):
            pairs.append(ibeta_pair(mp.mpf(a_), mp.mpf(b_), X, 1 - X))
    store("beta", ("cdf", "sf", "logcdf", "logsf"), two_tail_rows(pairs))

    columns = (inputs[f"hypergeom__{c}"] for c in "kMnN")
    store("hypergeom", ("sf", "logsf"), [hypergeom_tail(*v) for v in zip(*columns)])

    np.savez(directory / "reference.npz", **out)
    versions = {
        "python": sys.version.split()[0],
        "mpmath": mp.__version__,
        "numpy": np.__version__,
    }
    notes = {
        "betainc_agreement_points": len(agreement),
        "betainc_agreement_max": max(agreement, default=0.0),
        "hypergeom_agreement_points": len(hypergeom_agreement),
        "hypergeom_agreement_max": max(hypergeom_agreement, default=0.0),
    }
    (directory / "notes.json").write_text(json.dumps(notes))
    try:
        versions["scipy"] = _scipy_values(inputs, directory)
    except ImportError:
        pass
    (directory / "versions.json").write_text(json.dumps(versions))


def _scipy_values(inputs: dict[str, np.ndarray], directory: Path) -> str:
    """Evaluate scipy on the same inputs (information only); return its version."""
    import warnings

    import scipy
    from scipy import special, stats

    warnings.simplefilter("ignore")
    i = inputs
    # scipy's hypergeometric tails take time (and, for logsf, memory) in proportion to
    # the population, so they are evaluated for populations up to 1e6 only.
    hyper_evaluated = i["hypergeom__M"] <= 1e6
    hyper = [i[f"hypergeom__{c}"][hyper_evaluated] for c in "kMnN"]
    hyper_sf = np.full(hyper_evaluated.size, np.nan)
    hyper_logsf = np.full(hyper_evaluated.size, np.nan)
    hyper_sf[hyper_evaluated] = stats.hypergeom.sf(*hyper)
    hyper_logsf[hyper_evaluated] = stats.hypergeom.logsf(*hyper)
    fq = (i["f_quantile__p"], i["f_quantile__dfn"], i["f_quantile__dfd"])
    cq = (i["chi2_quantile__p"], i["chi2_quantile__df"])
    values = {
        "lgamma__value": special.gammaln(i["lgamma__x"]),
        "digamma__value": special.psi(i["digamma__x"]),
        "trigamma__value": special.polygamma(1, i["trigamma__x"]),
        "log_beta__value": special.betaln(i["log_beta__a"], i["log_beta__b"]),
        "gammainc__P": special.gammainc(i["gammainc__a"], i["gammainc__x"]),
        "gammainc__Q": special.gammaincc(i["gammainc__a"], i["gammainc__x"]),
        "betainc__I": special.betainc(
            i["betainc__a"], i["betainc__b"], i["betainc__x"]
        ),
        "betainc__C": special.betaincc(
            i["betainc__a"], i["betainc__b"], i["betainc__x"]
        ),
        "norm__cdf": stats.norm.cdf(i["norm__x"]),
        "norm__sf": stats.norm.sf(i["norm__x"]),
        "norm__logcdf": stats.norm.logcdf(i["norm__x"]),
        "norm__logsf": stats.norm.logsf(i["norm__x"]),
        "norm_quantile__ppf": stats.norm.ppf(i["norm_quantile__p"]),
        "t__cdf": stats.t.cdf(i["t__x"], i["t__df"]),
        "t__sf": stats.t.sf(i["t__x"], i["t__df"]),
        "t__logsf": stats.t.logsf(i["t__x"], i["t__df"]),
        "t_genes__sf": stats.t.sf(i["t_genes__x"], i["t_genes__df"]),
        "t_genes__logsf": stats.t.logsf(i["t_genes__x"], i["t_genes__df"]),
        "t_quantile__isf": stats.t.isf(i["t_quantile__p"], i["t_quantile__df"]),
        "f__cdf": stats.f.cdf(i["f__x"], i["f__dfn"], i["f__dfd"]),
        "f__sf": stats.f.sf(i["f__x"], i["f__dfn"], i["f__dfd"]),
        "f__logsf": stats.f.logsf(i["f__x"], i["f__dfn"], i["f__dfd"]),
        "f_genes__sf": stats.f.sf(
            i["f_genes__x"], i["f_genes__dfn"], i["f_genes__dfd"]
        ),
        "f_genes__logsf": stats.f.logsf(
            i["f_genes__x"], i["f_genes__dfn"], i["f_genes__dfd"]
        ),
        "chi2__cdf": stats.chi2.cdf(i["chi2__x"], i["chi2__df"]),
        "chi2__sf": stats.chi2.sf(i["chi2__x"], i["chi2__df"]),
        "chi2__logsf": stats.chi2.logsf(i["chi2__x"], i["chi2__df"]),
        "f_quantile__isf": stats.f.isf(*fq),
        "f_quantile__ppf": stats.f.ppf(*fq),
        "chi2_quantile__isf": stats.chi2.isf(*cq),
        "chi2_quantile__ppf": stats.chi2.ppf(*cq),
        "gamma__cdf": stats.gamma.cdf(
            i["gamma__x"], i["gamma__shape"], scale=i["gamma__scale"]
        ),
        "gamma__sf": stats.gamma.sf(
            i["gamma__x"], i["gamma__shape"], scale=i["gamma__scale"]
        ),
        "beta__cdf": stats.beta.cdf(i["beta__x"], i["beta__a"], i["beta__b"]),
        "beta__sf": stats.beta.sf(i["beta__x"], i["beta__a"], i["beta__b"]),
        "hypergeom__sf": hyper_sf,
        "hypergeom__logsf": hyper_logsf,
        "hypergeom__evaluated": hyper_evaluated,
    }
    np.savez(
        directory / "scipy.npz",
        **{k: np.asarray(v, dtype=float) for k, v in values.items()},
    )
    return str(scipy.__version__)


# ---------------------------------------------------------------------------------------
# CytePool and the comparison
# ---------------------------------------------------------------------------------------


def cytepool_values(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate CytePool on the inputs, keyed like the reference."""
    from cytepool.stats import distributions as d
    from cytepool.stats import special as s

    i = inputs
    hyper = [i[f"hypergeom__{c}"] for c in "kMnN"]
    a_g, x_g = i["gammainc__a"], i["gammainc__x"]
    abx = (i["betainc__a"], i["betainc__b"], i["betainc__x"])
    t_args = (i["t__x"], i["t__df"])
    tg_args = (i["t_genes__x"], i["t_genes__df"])
    f_args = (i["f__x"], i["f__dfn"], i["f__dfd"])
    fg_args = (i["f_genes__x"], i["f_genes__dfn"], i["f_genes__dfd"])
    c_args = (i["chi2__x"], i["chi2__df"])
    g_args = (i["gamma__x"], i["gamma__shape"], i["gamma__scale"])
    b_args = (i["beta__x"], i["beta__a"], i["beta__b"])
    q_args = (i["t_quantile__p"], i["t_quantile__df"])
    fq_args = (i["f_quantile__p"], i["f_quantile__dfn"], i["f_quantile__dfd"])
    cq_args = (i["chi2_quantile__p"], i["chi2_quantile__df"])
    values: dict[str, Any] = {
        "lgamma__value": s.lgamma(i["lgamma__x"]),
        "digamma__value": s.digamma(i["digamma__x"]),
        "trigamma__value": s.trigamma(i["trigamma__x"]),
        "trigamma_inverse__value": s.trigamma_inverse(i["trigamma_inverse__y"]),
        "log_beta__value": s.log_beta(i["log_beta__a"], i["log_beta__b"]),
        "gammainc__P": s.gammainc(a_g, x_g),
        "gammainc__Q": s.gammaincc(a_g, x_g),
        "gammainc__logP": s.log_gammainc(a_g, x_g),
        "gammainc__logQ": s.log_gammaincc(a_g, x_g),
        "betainc__I": s.betainc(*abx),
        "betainc__C": s.betaincc(*abx),
        "betainc__logI": s.log_betainc(*abx),
        "betainc__logC": s.log_betaincc(*abx),
        "norm__cdf": d.norm_cdf(i["norm__x"]),
        "norm__sf": d.norm_sf(i["norm__x"]),
        "norm__logcdf": d.norm_logcdf(i["norm__x"]),
        "norm__logsf": d.norm_logsf(i["norm__x"]),
        "norm_quantile__ppf": d.norm_ppf(i["norm_quantile__p"]),
        "norm_quantile__isf": d.norm_isf(i["norm_quantile__p"]),
        "t__cdf": d.t_cdf(*t_args),
        "t__sf": d.t_sf(*t_args),
        "t__logsf": d.t_logsf(*t_args),
        "t_genes__sf": d.t_sf(*tg_args),
        "t_genes__logsf": d.t_logsf(*tg_args),
        "t_quantile__isf": d.t_isf(*q_args),
        "t_quantile__ppf": d.t_ppf(*q_args),
        "f__cdf": d.f_cdf(*f_args),
        "f__sf": d.f_sf(*f_args),
        "f__logsf": d.f_logsf(*f_args),
        "f_genes__sf": d.f_sf(*fg_args),
        "f_genes__logsf": d.f_logsf(*fg_args),
        "chi2__cdf": d.chi2_cdf(*c_args),
        "chi2__sf": d.chi2_sf(*c_args),
        "chi2__logsf": d.chi2_logsf(*c_args),
        "f_quantile__isf": d.f_isf(*fq_args),
        "f_quantile__ppf": d.f_ppf(*fq_args),
        "chi2_quantile__isf": d.chi2_isf(*cq_args),
        "chi2_quantile__ppf": d.chi2_ppf(*cq_args),
        "gamma__cdf": d.gamma_cdf(*g_args),
        "gamma__sf": d.gamma_sf(*g_args),
        "beta__cdf": d.beta_cdf(*b_args),
        "beta__sf": d.beta_sf(*b_args),
        "hypergeom__sf": d.hypergeom_sf(*hyper),
        "hypergeom__logsf": d.hypergeom_logsf(*hyper),
    }
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def _relative_errors(got: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    errors = np.zeros_like(ref)
    errors[mask] = np.abs(got[mask] - ref[mask]) / np.abs(ref[mask])
    return errors


def _special_values_agree(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Mask of points outside the relative check whose values agree closely enough.

    Infinities and NaN must match exactly; values (and log values) smaller than 1e-300
    in magnitude, which as doubles carry fewer bits, must agree to 1e-312.
    """
    agree = (got == ref) | (np.isnan(got) & np.isnan(ref))
    small = np.isfinite(ref) & (np.abs(ref) < SMALLEST_CHECKED)
    with np.errstate(invalid="ignore"):
        return agree | (small & (np.abs(got - ref) <= 1e-312))


Metric = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _probability(got: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ref) & (np.abs(ref) >= SMALLEST_CHECKED)
    return _relative_errors(got, ref, mask), mask


def _log_probability(got: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ref) & (np.abs(ref) >= SMALLEST_CHECKED)
    return _relative_errors(got, ref, mask), mask


def _scaled(got: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ref)
    errors = np.zeros_like(ref)
    errors[mask] = np.abs(got[mask] - ref[mask]) / np.maximum(1.0, np.abs(ref[mask]))
    return errors, mask


def _relative(got: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ref) & (ref != 0)
    return _relative_errors(got, ref, mask), mask


METRICS: dict[str, tuple[Metric, str]] = {
    "scaled": (_scaled, "max error / max(1, abs(f))"),
    "relative": (_relative, "max rel error"),
    "probability": (_probability, "max rel error, values >= 1e-300"),
    "log": (_log_probability, "max rel error of the log, abs(log) >= 1e-300"),
}

Check = tuple[str, str, str, str, float, str]


def checks() -> list[Check]:
    """The table rows: name, reference key, CytePool key, metric, tolerance, subset.

    The subset is "positive" or "negative" (the sign of the first argument), "negate"
    (compare with minus the reference), or "" for all points of the family.
    """
    rows: list[Check] = []

    def add(
        name: str, key: str, metric: str, tol: float, subset: str = "", ours: str = ""
    ) -> None:
        rows.append((name, key, ours or key, metric, tol, subset))

    add("lgamma, x > 0", "lgamma__value", "scaled", NEAR_MACHINE, "positive")
    add("lgamma, x < 0", "lgamma__value", "scaled", NEAR_MACHINE, "negative")
    add("digamma, x > 0", "digamma__value", "relative", NEAR_MACHINE, "positive")
    add("digamma, x < 0", "digamma__value", "scaled", 2 * NEAR_MACHINE, "negative")
    add("trigamma", "trigamma__value", "relative", NEAR_MACHINE)
    add("trigamma_inverse", "trigamma_inverse__value", "relative", NEAR_MACHINE)
    add("log_beta", "log_beta__value", "scaled", 2 * NEAR_MACHINE)
    for name, key in [
        ("gammainc", "gammainc__P"),
        ("gammaincc", "gammainc__Q"),
        ("betainc", "betainc__I"),
        ("betaincc", "betainc__C"),
    ]:
        add(name, key, "probability", CONTRACT)
        add(f"log_{name}", key.replace("__", "__log"), "log", CONTRACT)
    distributions = [
        ("norm", ("cdf", "sf", "logcdf", "logsf")),
        ("t", ("cdf", "sf", "logsf")),
        ("f", ("cdf", "sf", "logsf")),
        ("chi2", ("cdf", "sf", "logsf")),
        ("gamma", ("cdf", "sf")),
        ("beta", ("cdf", "sf")),
        ("hypergeom", ("sf", "logsf")),
    ]
    for distribution, kinds in distributions:
        for kind in kinds:
            metric = "log" if kind.startswith("log") else "probability"
            add(f"{distribution}_{kind}", f"{distribution}__{kind}", metric, CONTRACT)
        if distribution == "norm":
            add("norm_ppf", "norm_quantile__ppf", "relative", CONTRACT)
            add(
                "norm_isf",
                "norm_quantile__ppf",
                "relative",
                CONTRACT,
                "negate",
                "norm_quantile__isf",
            )
        if distribution in ("t", "f"):
            label = "2,000 moderated t" if distribution == "t" else "2,000 QL F"
            add(
                f"{distribution}_sf, {label}",
                f"{distribution}_genes__sf",
                "probability",
                CONTRACT,
            )
            add(
                f"{distribution}_logsf, {label}",
                f"{distribution}_genes__logsf",
                "log",
                CONTRACT,
            )
        if distribution in ("f", "chi2"):
            add(
                f"{distribution}_isf",
                f"{distribution}_quantile__isf",
                "relative",
                CONTRACT,
            )
            add(
                f"{distribution}_ppf",
                f"{distribution}_quantile__ppf",
                "relative",
                CONTRACT,
            )
        if distribution == "t":
            add("t_isf", "t_quantile__isf", "relative", CONTRACT)
            add(
                "t_ppf",
                "t_quantile__isf",
                "relative",
                CONTRACT,
                "negate",
                "t_quantile__ppf",
            )
    return rows


def _family(key: str) -> str:
    return key.split("__")[0]


def _arguments(inputs: dict[str, np.ndarray], family: str, index: int) -> str:
    keys = [k for k in inputs if _family(k) == family]
    return ", ".join(f"{k.split('__')[1]}={inputs[k][index]:.17g}" for k in keys)


def _selection(inputs: dict[str, np.ndarray], family: str, subset: str) -> np.ndarray:
    first = next(v for k, v in inputs.items() if _family(k) == family)
    if subset == "positive":
        return first > 0
    if subset == "negative":
        return first < 0
    return np.ones(first.shape, dtype=bool)


def compare(
    inputs: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    ours: dict[str, np.ndarray],
    scipy_values: dict[str, np.ndarray],
) -> tuple[
    list[list[str]], list[list[str]], list[list[str]], bool, dict[str, np.ndarray]
]:
    """Return the main table, the worst cases, the scipy table, success and the errors."""
    table, worst, scipy_table = [], [], []
    passed = True
    errors_by_family: dict[str, np.ndarray] = {}
    for name, ref_key, our_key, metric_name, tolerance, subset in checks():
        metric, label = METRICS[metric_name]
        family = _family(ref_key)
        ref = -reference[ref_key] if subset == "negate" else reference[ref_key]
        got = ours[our_key]
        selected = _selection(inputs, family, subset)
        if np.isnan(ref[selected]).any():
            raise RuntimeError(f"the reference for {name} contains NaN")
        errors, mask = metric(got, ref)
        mask &= selected
        errors = np.where(mask, errors, 0.0)
        special = ~_special_values_agree(got, ref)
        disagree = selected & ~mask & special
        value = float(errors.max()) if mask.any() else 0.0
        ok = value <= tolerance and not disagree.any()
        passed &= ok
        result = "pass" if ok else "FAIL"
        if disagree.any():
            result += f" ({int(disagree.sum())} special values differ)"
        reference_label = "mpmath, 50 digits" if family != "hypergeom" else "exact sums"
        table.append(
            [name, reference_label, label, f"{value:.2e}", f"{tolerance:.0e}", result]
        )
        if mask.any():
            index = int(np.argmax(errors))
            worst.append(
                [
                    name,
                    _arguments(inputs, family, index),
                    f"{ref[index]:.10g}",
                    f"{value:.2e}",
                ]
            )
        previous = errors_by_family.get(family, np.zeros_like(errors))
        errors_by_family[family] = np.maximum(previous, errors)
        theirs = scipy_values.get(our_key)
        if theirs is not None and subset != "negate":
            scipy_errors, scipy_mask = metric(theirs, ref)
            scipy_mask &= selected
            notes = []
            evaluated = scipy_values.get(f"{family}__evaluated")
            if evaluated is not None:
                evaluated = evaluated.astype(bool)
                notes.append(f"evaluated at {int((selected & evaluated).sum())} points")
                scipy_mask &= evaluated
            finite = scipy_mask & np.isfinite(theirs)
            scipy_value = float(scipy_errors[finite].max()) if finite.any() else 0.0
            missing = int((scipy_mask & ~np.isfinite(theirs)).sum())
            if missing:
                notes.append(f"{missing} non-finite")
            scipy_table.append(
                [name, f"{scipy_value:.2e}", f"{value:.2e}", "; ".join(notes)]
            )
    return table, worst, scipy_table, passed, errors_by_family


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_fixtures(
    inputs: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    hard: dict[str, np.ndarray],
    errors_by_family: dict[str, np.ndarray],
) -> None:
    """Write a few hundred points per family: the hard cases, the worst and a random sample."""
    rng = np.random.default_rng(SEED + 1)
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    families = sorted({_family(k) for k in inputs} - UNFIXTURED)
    for family in families:
        keys_in = [k for k in inputs if _family(k) == family]
        keys_ref = [k for k in reference if _family(k) == family]
        size = len(inputs[keys_in[0]])
        chosen = set(hard.get(family, np.array([], dtype=int)).tolist())
        errors = errors_by_family.get(family, np.zeros(size))
        chosen |= set(np.argsort(-errors)[:40].tolist())
        remaining = np.setdiff1d(np.arange(size), np.fromiter(chosen, dtype=int))
        extra = max(0, min(FIXTURE_POINTS - len(chosen), remaining.size))
        chosen |= set(rng.choice(remaining, extra, replace=False).tolist())
        index = np.array(sorted(chosen))
        arrays = {k.split("__")[1]: inputs[k][index] for k in keys_in}
        arrays |= {k.split("__")[1]: reference[k][index] for k in keys_ref}
        np.savez_compressed(FIXTURE_DIRECTORY / f"{family}.npz", **arrays)
        print(f"wrote {FIXTURE_DIRECTORY / (family + '.npz')} ({index.size} points)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference is not None:
        reference_main(args.reference)
        return 0

    inputs, hard = build_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        np.savez(directory / "inputs.npz", **inputs)
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, __file__, "--reference", str(directory)], check=True
        )
        print(f"reference computed in {time.perf_counter() - start:.0f} s")
        reference = dict(np.load(directory / "reference.npz"))
        versions = json.loads((directory / "versions.json").read_text())
        notes = json.loads((directory / "notes.json").read_text())
        scipy_file = directory / "scipy.npz"
        scipy_values = dict(np.load(scipy_file)) if scipy_file.exists() else {}

    start = time.perf_counter()
    ours = cytepool_values(inputs)
    print(
        f"CytePool evaluated in {time.perf_counter() - start:.1f} s (including compilation)"
    )
    table, worst, scipy_table, passed, errors_by_family = compare(
        inputs, reference, ours, scipy_values
    )

    print("\nReference versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    print(
        "Reference self-check: the 70-digit continued fraction for the incomplete beta "
        f"agrees with mpmath's betainc to {notes['betainc_agreement_max']:.1e} "
        f"(relative) on the {notes['betainc_agreement_points']} points with both shapes "
        "at most 1e3; the log-gamma form of the first hypergeometric probability agrees "
        f"with exact binomial coefficients to {notes['hypergeom_agreement_max']:.1e} "
        f"(relative to max(1, abs(log p))) on the {notes['hypergeom_agreement_points']} "
        "points with populations up to 1e6"
    )
    if max(notes["betainc_agreement_max"], notes["hypergeom_agreement_max"]) > 1e-45:
        print("REFERENCE SELF-CHECK FAILED")
        passed = False
    sizes: dict[str, int] = {}
    for key, values in inputs.items():
        sizes.setdefault(_family(key), len(values))
    print(
        "Points per family: " + ", ".join(f"{k} {v}" for k, v in sizes.items()) + "\n"
    )
    print(
        _markdown(
            ["check", "reference", "metric", "value", "tolerance", "result"], table
        )
    )
    print("\nWorst case of each check:\n")
    print(_markdown(["check", "arguments", "reference value", "relative error"], worst))
    if scipy_table:
        print(f"\nscipy {versions.get('scipy')} on the same points, for information:\n")
        print(_markdown(["check", "scipy", "CytePool", "note"], scipy_table))
    if args.write_fixtures:
        print()
        write_fixtures(inputs, reference, hard, errors_by_family)
    print("\nall checks passed" if passed else "\nSOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
