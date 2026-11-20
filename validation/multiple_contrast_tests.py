"""Check multivariate t probabilities and quantiles, contrast matrices and multiple
contrast tests against mpmath, mvtnorm, scipy and multcomp.

Every input comes from fixed seeds (``SEED`` and the seeds derived from it), and every
problem that CytePool computes by Monte Carlo has its own seed, so that the checks are
independent. CytePool runs at ``relative_error=1e-3`` for the calibrated checks.

References for the probabilities of boxes, P(T in box) and its complement:

- Two coordinates: the conditional distribution of a bivariate t (Ding 2016): given
  T_2 = t, T_1 is t-distributed on df + 1 degrees of freedom with location rho t and
  scale sqrt((df + t^2) (1 - rho^2) / (df + 1)); the probability and the complement are
  one-dimensional integrals, computed with mpmath at 30 digits, the t distribution
  through the incomplete beta function. CytePool averages over directions instead,
  without random numbers, so these rows are in the close tier (1e-6, relative to the
  smaller of the probability and the complement).
- Product correlations, R_ij = lambda_i lambda_j, as Dunnett's contrasts of groups of
  unequal sizes have, with one limit for every coordinate or a limit per coordinate:
  Dunnett's (1955) integral over the shared normal factor and the chi variable, in
  double precision with tail-accurate normal functions (scipy's ``ndtr``), the factor
  by Gauss-Legendre panels around the peaks of the integrand and the chi variable by
  adaptive quadrature (scipy's ``quad``) to 1e-12 relative.
  Two-dimensional quadrature in mpmath took over two minutes per value, so the script
  checks this reference against Ding's at 30 digits on two-coordinate cases, and
  against mpmath's one-dimensional integral over the factor for the normal. In that
  check, scipy's ``quad`` warns of rounding in 4 of the 48 two-coordinate cases, tail
  orthants, whose values still agree with Ding's to 2.7e-14; it warns too for 9 of the
  96 cases with a limit per coordinate, seven of which agree with Gauss-Legendre in S
  to 5e-10 and two, tails near 7e-10 at df 2.5, with CytePool within its errors.
- Rank-two correlations, such as all pairwise contrasts of three groups (Tukey's), with
  mixed signs: R_ij = cos(theta_i - theta_j), so that T = Z / S with Z = A u for the
  unit vectors a_i of the angles theta_i and u standard normal in the plane; the
  probability of the box is the average over the direction of u of G(r_in) - G(r_out),
  G(r) = (1 + r^2 / df)^(-df / 2), the tail of the F distribution on 2 and df degrees of
  freedom at r^2 / 2. Computed with mpmath at 30 digits; CytePool computes these boxes
  by Monte Carlo, with singular coordinates.
- Correlations of rank 3 and more without a product structure, deep in the tail
  (Williams-type contrasts of 5, 7 and 9 groups, Tukey's of 4 groups and random
  contrasts, complements of 1e-4, 1e-7 and 1e-10): ALOE (Owen, Maximov and Chertkov
  2019), plain Monte Carlo from the mixture of the distributions given each
  exceedance, each draw weighted by the sum of the marginal tails over the number of
  coordinates beyond their limits, 1e6 draws per problem, in numpy and scipy, with
  the standard error of its draws. It differs from CytePool's method, which
  integrates the chance of each first exceedance with lattice points.
- General correlations of three to ten coordinates (Dunnett's, Williams' and Tukey's
  contrasts of groups of unequal sizes, and random contrasts): mvtnorm's ``pmvt`` with
  ``GenzBretz(maxpts = 1e6, abseps = 1e-5, releps = 0)`` for integer df and the normal,
  and scipy's ``multivariate_t.cdf`` for df 4.5, 10 and 20.5, 16 calls with different
  random states. mvtnorm's standard error is taken as its error attribute over 3 (in
  200 repeated calls of a probe, the attribute was 3.0 and 3.4 times the spread of the
  estimates), scipy's from the spread of its 16 estimates.

Tiers: exact (1e-10, relative or relative to max(|reference|, 1) for statistics); close
(1e-6 relative); calibrated: the difference from the reference within 5.5 combined
standard errors, the 99.994% point of the t distribution on 15 degrees of freedom, on
which CytePool's errors are estimated from 16 random shifts (the two-sided tail of 4
normal standard errors). With e the standard error of each value, the check is
|ours - reference| <= 5.5 sqrt(e_ours^2 + e_reference^2).

Checks:

- Probabilities: the rows above; one coordinate against mpmath at df 4.5 (exact); the
  stored error of the two-coordinate quadrature against its actual error (it must
  bound it); the calibration of CytePool's standard errors, the root-mean-square of
  (estimate - reference) / error over the product and rank-two problems at
  ``relative_error`` 1e-3 and, with fresh seeds, at the default 0.01, which must lie
  within 4 simulation standard errors of sqrt(15 / 13), its value for honest errors on
  15 degrees of freedom; mvtnorm with ``abseps = 1e-12`` in the tail of a product
  correlation (information); and mvtnorm's refusal of non-integer df (a deliberate
  difference).
- Quantiles, one- and two-sided, at p 0.9, 0.95 and 0.99: against the roots of
  Ding's formula for two coordinates (close); against the roots of Dunnett's integral
  for product correlations (calibrated); for general correlations, mvtnorm's ``pmvt`` at
  CytePool's quantile against p (calibrated, with the slope of ``pmvt`` around the
  quantile turning its standard error into one of the probability); the round trip
  through ``multivariate_t_cdf``; and mvtnorm's ``qmvt`` (information).
- Contrast matrices of ``dunnett_contrasts`` and ``williams_contrasts`` against
  multcomp's ``contrMat(n, type = "Dunnett", base)`` and ``contrMat(n, type =
  "Williams")`` (exact; Dunnett's names identical); Williams' names and two groups
  (deliberate differences).
- ``multiple_contrast_test`` on 40 seeded features per family (Dunnett's contrasts of 5
  groups, Williams' of 5 groups one-sided, Tukey's of 4 groups, singular, and 5 random
  contrasts of 6 estimates), with df 5, 12, 40 and the normal and every alternative,
  against ``summary(glht(parm(b, V, df), linfct = K))``: estimates, standard errors and
  t (exact) and the single-step p-values from 1e-3 to 1 (calibrated, multcomp's
  standard error its error attribute over 3); below 1e-3, multcomp's default absolute
  error of 1e-3 cannot resolve them, and their ratios to CytePool's are information;
  and, as controls, Bonferroni's p-values
  (``adjusted("bonferroni")``) of the correlated families, which must exceed
  CytePool's by more than 5.5 standard errors for p-values from 1e-3 to 0.5.

Requirements:
    Python 3.13 with numpy and numba, mpmath and scipy, run from the repository root so
    that ``cytepool`` imports from the working tree. R 4.4 or newer with the CRAN
    packages mvtnorm and multcomp; set ``RSCRIPT`` to the Rscript executable when it is
    not on PATH.

The references in Python run in another process, this script run as ``python
validation/multiple_contrast_tests.py --reference DIRECTORY``, which reads
``inputs.npz`` and writes ``reference.npz``; it imports mpmath and scipy, not
CytePool.

Usage:
    PYTHONPATH=. python validation/multiple_contrast_tests.py
    PYTHONPATH=. python validation/multiple_contrast_tests.py --write-fixtures
    PYTHONPATH=. python validation/multiple_contrast_tests.py --benchmark

``--benchmark`` times ``multiple_contrast_test`` on 20,000 genes with 9 dose groups (8
Dunnett contrasts, two-sided), gene-specific variances per group and df 20.5, 90% of
the genes without an effect, against the target of 1 ms per gene on 2 threads; with 3
groups (two contrasts, computed without random numbers), 4 groups, Williams' contrasts
one-sided and ``relative_error=1e-3`` for information; and the weighted fit of
``linear_models.py --benchmark`` on 20,000 genes x 200 samples as the baseline.

References:
    Ding P (2016). On the conditional distribution of the multivariate t
    distribution. The American Statistician 70, 293-295.
    doi:10.1080/00031305.2016.1164756

    Dunnett CW (1955). A multiple comparison procedure for comparing several
    treatments with a control. Journal of the American Statistical Association 50,
    1096-1121. doi:10.1080/01621459.1955.10501294

    Owen AB, Maximov Y, Chertkov M (2019). Importance sampling the union of rare events
    with an application to power systems analysis. Electronic Journal of Statistics
    13, 231-254. doi:10.1214/18-EJS1527
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "multiple_contrast_tests"
)
SEED = 20261120
EXACT = 1e-10
CLOSE = 1e-6
# Standard errors: the 99.994% point of the t distribution on 15 degrees of freedom,
# whose two-sided tail is that of 4 normal standard errors.
CALIBRATED = 5.5
TIGHT = 1e-3  # relative_error of CytePool in the calibrated checks
INF = math.inf
SCIPY_CALLS = 16
# Moments of t^2 for t on 15 degrees of freedom: honest standard errors estimated from
# 16 shifts give z-scores with these.
T15_MEAN = 15.0 / 13.0
T15_VARIANCE = 3.0 * 15.0**2 / (13.0 * 11.0) - T15_MEAN**2
WORKERS = max(1, min(3, len(os.sched_getaffinity(0)) - 1))

PLANE_DF = (1.0, 2.5, 4.0, 9.7, 30.0, 200.0, INF)
PLANE_RHO = (-0.95, -0.5, 0.0, 0.3, 0.7, 0.99)
PRODUCT_K = (3, 4, 6, 8, 10)
PRODUCT_DF = (2.5, 7.0, 20.5, INF)
RANK_TWO_DF = (3.0, 12.5, INF)
MVTNORM_DF = (3.0, 10.0, 30.0, INF)
SCIPY_DF = (4.5, 10.0, 20.5)
DISTINCT_K = (3, 5, 8)
ALOE_DF = (3.0, 8.0, INF)
ALOE_TARGETS = (1e-4, 1e-7, 1e-10)
ALOE_DRAWS = 1_000_000
# The kinds of reference the reference process computes, in the order it returns them.
REFERENCE_KINDS = (
    "plane",
    "plane_product",
    "product",
    "product_mp",
    "distinct",
    "rank_two",
    "product_quantile",
    "plane_quantile",
    "scipy",
    "aloe",
)


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------


def _isf(p: float, df: float) -> float:
    from cytepool.stats.distributions import t_isf

    return float(t_isf(p, df))


def _t_sf(x: np.ndarray, df: float) -> np.ndarray:
    from cytepool.stats.distributions import t_sf

    return np.asarray(t_sf(x, df), dtype=float)


def plane_cases() -> list[dict[str, Any]]:
    """Two coordinates: each correlation and df with eight boxes, from complements of
    about 0.5 to 1e-14, a small box and a tail orthant."""
    cases = []
    for df in PLANE_DF:
        boxes: list[tuple[bool, list[float]]] = []
        for target in (0.5, 1e-3, 1e-8, 1e-14):
            boxes.append((True, [_isf(target / 4, df)] * 2))
        boxes.append((False, [_isf(1e-6 / 2, df)] * 2))
        boxes.append((True, [_isf(1e-4 / 2, df), _isf(1e-8 / 2, df)]))
        boxes.append((True, [_isf(0.495, df)] * 2))  # P(|T| <= c) = 0.01 each
        boxes.append((False, [-_isf(1e-3, df)] * 2))  # P(T <= b) = 1e-3 each
        for rho in PLANE_RHO:
            for two_sided, upper in boxes:
                cases.append(
                    {"rho": rho, "df": df, "two_sided": two_sided, "upper": upper}
                )
    return cases


def product_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Product correlations of Dunnett's contrasts of groups of unequal sizes: each k
    and df with nine boxes, complements from 0.3 to 1e-12 and small boxes."""
    cases = []
    for k in PRODUCT_K:
        sizes = rng.integers(2, 7, k)
        control = int(rng.integers(2, 9))
        lam = np.sqrt(sizes / (sizes + control))
        for df in PRODUCT_DF:
            boxes: list[tuple[bool, float]] = []
            for target in (0.3, 1e-3, 1e-6, 1e-12):
                boxes.append((True, _isf(target / (2 * k), df)))
            for target in (0.05, 1e-6):
                boxes.append((False, _isf(target / k, df)))
            for target in (1e-2, 1e-6):
                boxes.append((True, _isf((1.0 - target ** (1.0 / k)) / 2.0, df)))
            boxes.append((False, -_isf(1e-4 ** (1.0 / k), df)))
            for two_sided, limit in boxes:
                cases.append(
                    {
                        "lam": lam,
                        "df": df,
                        "two_sided": two_sided,
                        "upper": np.full(k, limit),
                    }
                )
    return cases


def distinct_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Product correlations with a limit per coordinate: for each k and df, complements
    of 1e-2, 1e-5 and 1e-9 split unevenly over the coordinates, the largest share
    last, one- and two-sided; and small boxes of probability near 1e-3 split unevenly,
    the coordinate of smallest probability last."""
    cases = []
    for k in DISTINCT_K:
        sizes = rng.integers(2, 7, k)
        lam = np.sqrt(sizes / (sizes + int(rng.integers(2, 9))))
        for df in PRODUCT_DF:
            boxes: list[tuple[bool, list[float]]] = []
            for target in (1e-2, 1e-5, 1e-9):
                shares = np.sort(rng.dirichlet(np.ones(k)))
                boxes.append((True, [_isf(target * s / 2.0, df) for s in shares]))
                boxes.append((False, [_isf(target * s, df) for s in shares]))
            inside = np.sort(1e-3 ** rng.dirichlet(np.ones(k)))[::-1]
            boxes.append((True, [_isf((1.0 - p) / 2.0, df) for p in inside]))
            boxes.append((False, [-_isf(p, df) for p in inside]))
            for two_sided, upper in boxes:
                cases.append(
                    {
                        "lam": lam,
                        "df": df,
                        "two_sided": two_sided,
                        "upper": np.array(upper),
                    }
                )
    return cases


def aloe_families(rng: np.random.Generator) -> list[tuple[str, np.ndarray, bool]]:
    """Correlations of rank 3 and more without a product structure, each with the side
    its test uses: Williams-type contrasts of 5, 7 and 9 groups, one-sided; Tukey's
    contrasts of 4 groups, two-sided; 8 random contrasts of 9 groups, two-sided, and 5
    of 6, one-sided."""
    families = []
    for sizes in ([4, 3, 3, 4, 3], [3] * 7, [6, 2, 3, 2, 4, 2, 3, 2, 5]):
        n = np.array(sizes, float)
        families.append(
            (
                f"Williams' contrasts of {n.size} groups, one-sided",
                _correlation(williams_matrix(n), np.diag(1.0 / n)),
                False,
            )
        )
    n = np.array([3.0, 4.0, 2.0, 5.0])
    families.append(
        (
            "Tukey's contrasts of 4 groups (rank 3), two-sided",
            _correlation(tukey_matrix(4), np.diag(1.0 / n)),
            True,
        )
    )
    for k, two_sided in ((8, True), (5, False)):
        weights = rng.normal(size=(k + 1, k))
        weights -= weights.mean(axis=0)
        n = rng.integers(2, 6, k + 1).astype(float)
        side = "two-sided" if two_sided else "one-sided"
        families.append(
            (
                f"{k} random contrasts of {k + 1} groups, {side}",
                _correlation(weights, np.diag(1.0 / n)),
                two_sided,
            )
        )
    return families


def aloe_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Complements of 1e-4, 1e-7 and 1e-10 by Bonferroni's split, at df 3, 8 and the
    normal, each with its own seed for ALOE."""
    cases = []
    for label, correlation, two_sided in aloe_families(rng):
        k = correlation.shape[0]
        for df in ALOE_DF:
            for target in ALOE_TARGETS:
                share = target / (2 * k) if two_sided else target / k
                cases.append(
                    {
                        "label": label,
                        "correlation": correlation,
                        "df": df,
                        "two_sided": two_sided,
                        "upper": np.full(k, _isf(share, df)),
                        "seed": SEED + 13000 + len(cases),
                    }
                )
    return cases


def _rank_two_angles(contrasts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The angles of unit vectors a_i with a_i . a_j the correlation of contrasts of
    three group means of the given sizes."""
    covariance = contrasts.T @ np.diag(1.0 / sizes) @ contrasts
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    values, vectors = np.linalg.eigh(correlation)
    loadings = vectors[:, -2:] * np.sqrt(values[-2:])
    return np.arctan2(loadings[:, 1], loadings[:, 0])


def rank_two_sets(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Rank-two correlations: contrasts among three groups, and random angles."""
    tukey = np.array([[-1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    williams_sizes = np.array([5.0, 3.0, 4.0])
    williams = np.array(
        [[-1.0, -1.0, 0.0], [0.0, 3.0 / 7.0, -1.0], [1.0, 4.0 / 7.0, 1.0]]
    )
    return [
        ("Tukey's contrasts of 3 equal groups", _rank_two_angles(tukey, np.ones(3))),
        (
            "Tukey's contrasts of 3 groups of 4, 2 and 6",
            _rank_two_angles(tukey, np.array([4.0, 2.0, 6.0])),
        ),
        (
            "Williams' contrasts of 3 groups of 5, 3 and 4, and the doses' difference",
            _rank_two_angles(williams, williams_sizes),
        ),
        ("4 random directions", rng.uniform(0.0, math.pi, 4)),
        ("6 random directions", rng.uniform(0.0, math.pi, 6)),
    ]


def rank_two_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases = []
    for label, angles in rank_two_sets(rng):
        k = angles.size
        for df in RANK_TWO_DF:
            boxes: list[tuple[bool, float]] = []
            for target in (0.2, 1e-4, 1e-10):
                boxes.append((True, _isf(target / (2 * k), df)))
            for target in (1e-3, 1e-8):
                boxes.append((False, _isf(target / k, df)))
            boxes.append((True, _isf((1.0 - 1e-3**0.5) / 2.0, df)))
            for two_sided, limit in boxes:
                cases.append(
                    {
                        "label": label,
                        "angles": angles,
                        "df": df,
                        "two_sided": two_sided,
                        "upper": np.full(k, limit),
                    }
                )
    return cases


def _correlation(contrasts: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    v = contrasts.T @ covariance @ contrasts
    scale = np.sqrt(np.diag(v))
    out = v / np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return 0.5 * (out + out.T)


def dunnett_matrix(k: int, control: int = 0) -> np.ndarray:
    """(k, k - 1): each group minus the control, by hand."""
    columns = []
    for j in range(k):
        if j != control:
            column = np.zeros(k)
            column[j], column[control] = 1.0, -1.0
            columns.append(column)
    return np.column_stack(columns)


def williams_matrix(sizes: np.ndarray) -> np.ndarray:
    """(k, k - 1): the size-weighted mean of the j highest doses minus the control."""
    k = sizes.size
    out = np.zeros((k, k - 1))
    for j in range(1, k):
        pooled = np.arange(k - j, k)
        out[0, j - 1] = -1.0
        out[pooled, j - 1] = sizes[pooled] / sizes[pooled].sum()
    return out


def tukey_matrix(k: int) -> np.ndarray:
    columns = []
    for a in range(k):
        for b in range(a + 1, k):
            column = np.zeros(k)
            column[b], column[a] = 1.0, -1.0
            columns.append(column)
    return np.column_stack(columns)


def general_families(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Correlations of 4 to 10 contrasts, singular ones included."""
    families = []
    sizes = np.array([6.0, 3.0, 4.0, 2.0, 5.0])
    families.append(
        (
            "Dunnett's contrasts of 5 groups",
            _correlation(dunnett_matrix(5), np.diag(1.0 / sizes)),
        )
    )
    sizes = np.array([5.0, 2.0, 3.0, 4.0, 2.0, 6.0])
    families.append(
        (
            "Williams' contrasts of 6 groups",
            _correlation(williams_matrix(sizes), np.diag(1.0 / sizes)),
        )
    )
    sizes = np.array([3.0, 5.0, 2.0, 4.0])
    families.append(
        (
            "Tukey's contrasts of 4 groups (rank 3)",
            _correlation(tukey_matrix(4), np.diag(1.0 / sizes)),
        )
    )
    for k in (8, 10):
        weights = rng.normal(size=(k + 1, k))
        weights -= weights.mean(axis=0)
        sizes = rng.integers(2, 8, k + 1).astype(float)
        families.append(
            (
                f"{k} random contrasts of {k + 1} groups",
                _correlation(weights, np.diag(1.0 / sizes)),
            )
        )
    return families


def general_cases(rng: np.random.Generator, dfs: tuple[float, ...]) -> list[dict]:
    cases = []
    for label, correlation in general_families(rng):
        k = correlation.shape[0]
        for df in dfs:
            boxes: list[tuple[bool, float]] = []
            for target in (0.5, 0.05, 1e-3):
                boxes.append((True, _isf(target / (2 * k), df)))
            for target in (0.05, 1e-3):
                boxes.append((False, _isf(target / k, df)))
            for two_sided, limit in boxes:
                cases.append(
                    {
                        "label": label,
                        "correlation": correlation,
                        "df": df,
                        "two_sided": two_sided,
                        "upper": np.full(k, limit),
                    }
                )
    return cases


def product_quantile_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases = []
    for k in (3, 5, 8):
        sizes = rng.integers(2, 7, k)
        lam = np.sqrt(sizes / (sizes + int(rng.integers(2, 9))))
        for df in (7.0, INF):
            for p in (0.9, 0.95, 0.99):
                for two_sided in (False, True):
                    cases.append({"lam": lam, "df": df, "p": p, "two_sided": two_sided})
    return cases


def plane_quantile_cases() -> list[dict[str, Any]]:
    return [
        {"rho": rho, "df": df, "p": p, "two_sided": two_sided}
        for rho in (-0.5, 0.5, 0.9)
        for df in (4.5, INF)
        for p in (0.9, 0.95, 0.99)
        for two_sided in (False, True)
    ]


def general_quantile_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "correlation": correlation,
            "df": df,
            "p": p,
            "two_sided": two_sided,
        }
        for label, correlation in general_families(rng)
        for df in (10.0, INF)
        for p in (0.95, 0.99)
        for two_sided in (False, True)
    ]


def one_dimension_points() -> dict[str, np.ndarray]:
    return {
        "upper": np.array([-30.0, -3.0, -1.0, 0.0, 0.5, 2.0, 5.0, 20.0, 1e3, 1e8]),
        "within": np.array([1e-12, 1e-6, 1e-3, 0.1, 0.5, 2.0, 10.0, 1e4]),
    }


def contrast_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """contrMat's inputs: Dunnett's contrasts with the control first, in the middle
    and last; Williams' with equal and unequal sizes; Williams' with two groups."""
    cases = []
    for k in range(2, 10):
        names = [f"G{j + 1}" for j in range(k)]
        sizes = rng.integers(2, 8, k)
        for base in sorted({0, k // 2, k - 1}):
            cases.append(
                {"type": "Dunnett", "base": base, "sizes": sizes, "names": names}
            )
    for k in range(3, 10):
        names = [f"D{j}" for j in range(k)]
        for sizes in (np.full(k, 3), rng.integers(2, 8, k)):
            cases.append(
                {"type": "Williams", "base": 0, "sizes": sizes, "names": names}
            )
    cases.append(
        {
            "type": "Williams",
            "base": 0,
            "sizes": np.array([4, 3]),
            "names": ["vehicle", "dose"],
        }
    )
    return cases


MULTCOMP_ALTERNATIVES = ("two-sided", "greater", "less")
MULTCOMP_DF = (5.0, 12.0, 40.0, INF)
# The absolute error of multcomp's default, GenzBretz(maxpts = 25000, abseps = 0.001):
# its p-values below it are not resolved.
MULTCOMP_ABSEPS = 1e-3


def multcomp_families(rng: np.random.Generator) -> list[dict[str, Any]]:
    """40 features per family: estimates, covariances, df and alternatives."""
    families = []
    specifications = [
        ("Dunnett's contrasts of 5 groups", dunnett_matrix(5), [6, 3, 4, 2, 5]),
        (
            "Williams' contrasts of 5 groups",
            williams_matrix(np.array([4.0, 2, 3, 3, 5])),
            [4, 2, 3, 3, 5],
        ),  # fmt: skip
        ("Tukey's contrasts of 4 groups", tukey_matrix(4), [3, 5, 2, 4]),
        ("5 random contrasts of 6 estimates", rng.normal(size=(6, 5)), None),
    ]
    for label, contrasts, sizes in specifications:
        n = contrasts.shape[0]
        alternatives = (
            ("greater", "less")
            if label.startswith("Williams")
            else MULTCOMP_ALTERNATIVES
        )
        features = []
        for i in range(40):
            if sizes is None:
                factor = rng.normal(size=(n, n))
                covariance = factor @ factor.T / n + 0.1 * np.eye(n)
            else:
                variance = 0.3 * 4.0 / rng.chisquare(4.0)
                covariance = np.diag(variance / np.asarray(sizes, float))
            effect = rng.choice([0.0, 0.3, 0.8, 1.6]) * rng.normal(size=n)
            means = 5.0 + effect + rng.multivariate_normal(np.zeros(n), covariance)
            features.append(
                {
                    "estimates": means,
                    "covariance": 0.5 * (covariance + covariance.T),
                    "df": MULTCOMP_DF[i % len(MULTCOMP_DF)],
                    "alternative": alternatives[i % len(alternatives)],
                }
            )
        families.append({"label": label, "contrasts": contrasts, "features": features})
    return families


# ---------------------------------------------------------------------------------------
# R: mvtnorm and multcomp
# ---------------------------------------------------------------------------------------

R_SCRIPT = r"""
suppressPackageStartupMessages({library(mvtnorm); library(multcomp)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_all <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8,
                                   endian = "little")
write_doubles <- function(x, path) writeBin(as.double(x), path, size = 8,
                                            endian = "little")
writeLines(c(R.version.string,
             paste("mvtnorm", format(packageVersion("mvtnorm"))),
             paste("multcomp", format(packageVersion("multcomp")))),
           file.path(root, "versions.txt"))
mvt_df <- function(df) if (is.infinite(df)) 0 else df
inputs <- function(task) sort(list.files(file.path(root, task), pattern = "[.]in$",
                                         full.names = TRUE))
# Boxes: k, df, lower (k), upper (k), correlation (k * k).
for (task in c("general", "tail")) {
  files <- inputs(task)
  for (i in seq_along(files)) {
    v <- read_all(files[[i]]); k <- as.integer(v[1]); df <- v[2]
    lower <- v[3:(2 + k)]; upper <- v[(3 + k):(2 + 2 * k)]
    corr <- matrix(v[(3 + 2 * k):(2 + 2 * k + k * k)], k, k)
    algorithm <- if (task == "tail") GenzBretz(maxpts = 1e7, abseps = 1e-12, releps = 0)
                 else GenzBretz(maxpts = 1e6, abseps = 1e-5, releps = 0)
    set.seed(i)
    started <- proc.time()[["elapsed"]]
    r <- pmvt(lower = lower, upper = upper, corr = corr, df = mvt_df(df),
              algorithm = algorithm)
    write_doubles(c(r, attr(r, "error"), proc.time()[["elapsed"]] - started),
                  sub("[.]in$", ".out", files[[i]]))
  }
}
refusal <- tryCatch({
  pmvt(lower = rep(-2, 3), upper = rep(2, 3), corr = diag(3), df = 4.5)
  "accepted"
}, error = function(e) conditionMessage(e))
writeLines(refusal, file.path(root, "refusal.txt"))
# Quantiles: k, df, p, two-sided, h, CytePool's quantile, correlation (k * k).
files <- inputs("quantiles")
for (i in seq_along(files)) {
  v <- read_all(files[[i]]); k <- as.integer(v[1]); df <- v[2]; p <- v[3]
  two <- v[4] > 0.5; h <- v[5]; q <- v[6]
  corr <- matrix(v[7:(6 + k * k)], k, k)
  set.seed(i)
  quantile <- tryCatch(
    qmvt(p, tail = if (two) "both.tails" else "lower.tail", corr = corr,
         df = mvt_df(df))$quantile,
    error = function(e) NA_real_)
  at <- function(x) {
    set.seed(i)
    lower <- if (two) rep(-x, k) else rep(-Inf, k)
    r <- pmvt(lower = lower, upper = rep(x, k), corr = corr, df = mvt_df(df),
              algorithm = GenzBretz(maxpts = 1e6, abseps = 1e-6, releps = 0))
    c(r, attr(r, "error"))
  }
  write_doubles(c(quantile, at(q), at(q - h)[1], at(q + h)[1]),
                sub("[.]in$", ".out", files[[i]]))
}
# multcomp: n, m, df, alternative, estimates (n), covariance (n * n), contrasts (m * n,
# by row).
alternatives <- c("two.sided", "greater", "less")
files <- inputs("multcomp")
for (i in seq_along(files)) {
  v <- read_all(files[[i]]); n <- as.integer(v[1]); m <- as.integer(v[2]); df <- v[3]
  alternative <- alternatives[as.integer(v[4]) + 1]
  b <- v[5:(4 + n)]; names(b) <- paste0("b", seq_len(n))
  V <- matrix(v[(5 + n):(4 + n + n * n)], n, n)
  K <- matrix(v[(5 + n + n * n):(4 + n + n * n + m * n)], m, n, byrow = TRUE)
  colnames(K) <- names(b); rownames(K) <- paste0("c", seq_len(m))
  g <- glht(parm(b, V, df = mvt_df(df)), linfct = K, alternative = alternative)
  set.seed(i)
  # A failure of multcomp leaves NaN, which the comparison counts as a failure.
  out <- tryCatch({
    s <- summary(g)
    bonferroni <- summary(g, test = adjusted("bonferroni"))$test$pvalues
    c(s$test$coefficients, s$test$sigma, s$test$tstat, s$test$pvalues,
      attr(s$test$pvalues, "error"), bonferroni)
  }, error = function(e) rep(NaN, 5 * m + 1))
  write_doubles(out, sub("[.]in$", ".out", files[[i]]))
}
# Contrast matrices: type, base (1-based), sizes, names.
lines <- readLines(file.path(root, "contrasts.txt"))
dir.create(file.path(root, "contrasts"))
for (i in seq_along(lines)) {
  parts <- strsplit(lines[[i]], " ", fixed = TRUE)[[1]]
  n <- as.numeric(strsplit(parts[3], ",", fixed = TRUE)[[1]])
  names(n) <- strsplit(parts[4], ",", fixed = TRUE)[[1]]
  K <- tryCatch(contrMat(n, type = parts[1], base = as.integer(parts[2])),
                error = function(e) conditionMessage(e))
  out <- file.path(root, "contrasts", sprintf("%03d", i))
  if (is.character(K)) {
    writeLines(c("ERROR", K), paste0(out, ".names"))
    next
  }
  write_doubles(t(unclass(K)), paste0(out, ".bin"))
  writeLines(rownames(K), paste0(out, ".names"))
}
"""


def _write(path: Path, values: list[Any]) -> None:
    np.asarray(
        np.concatenate([np.ravel(np.asarray(v, float)) for v in values]), dtype="<f8"
    ).tofile(path)


def _read(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f8")


def run_r(
    general: list[dict],
    tail: list[dict],
    quantiles: list[dict],
    multcomp: list[dict],
    contrasts: list[dict],
    directory: Path,
) -> dict[str, Any]:
    """Run mvtnorm and multcomp on every input; returns their outputs and versions."""
    for task in ("general", "tail", "quantiles", "multcomp"):
        (directory / task).mkdir()
    for i, case in enumerate(general + tail):
        task = "general" if i < len(general) else "tail"
        index = i if i < len(general) else i - len(general)
        lower = (
            -case["upper"] if case["two_sided"] else np.full(case["upper"].size, -INF)
        )
        _write(
            directory / task / f"{index:04d}.in",
            [case["upper"].size, case["df"], lower, case["upper"], case["correlation"]],
        )
    for i, case in enumerate(quantiles):
        _write(
            directory / "quantiles" / f"{i:04d}.in",
            [
                case["correlation"].shape[0],
                case["df"],
                case["p"],
                float(case["two_sided"]),
                case["h"],
                case["ours"],
                case["correlation"],
            ],
        )
    alternative_codes = {name: j for j, name in enumerate(MULTCOMP_ALTERNATIVES)}
    for i, feature in enumerate(multcomp):
        n, m = feature["contrasts"].shape
        _write(
            directory / "multcomp" / f"{i:04d}.in",
            [
                n,
                m,
                feature["df"],
                alternative_codes[feature["alternative"]],
                feature["estimates"],
                feature["covariance"],
                feature["contrasts"].T,
            ],
        )
    lines = [
        f"{c['type']} {c['base'] + 1} {','.join(map(str, c['sizes']))} "
        f"{','.join(c['names'])}"
        for c in contrasts
    ]
    (directory / "contrasts.txt").write_text("\n".join(lines) + "\n")
    script = directory / "references.R"
    script.write_text(R_SCRIPT)
    rscript = os.environ.get("RSCRIPT", "Rscript")
    subprocess.run([rscript, "--vanilla", str(script), str(directory)], check=True)
    out: dict[str, Any] = {
        "versions": (directory / "versions.txt").read_text().splitlines(),
        "refusal": (directory / "refusal.txt").read_text().strip(),
    }
    for task, count in (
        ("general", len(general)),
        ("tail", len(tail)),
        ("quantiles", len(quantiles)),
        ("multcomp", len(multcomp)),
    ):
        out[task] = [_read(directory / task / f"{i:04d}.out") for i in range(count)]
    matrices = []
    for i in range(len(contrasts)):
        base = directory / "contrasts" / f"{i + 1:03d}"
        names = (base.with_suffix(".names")).read_text().splitlines()
        if names[0] == "ERROR":
            matrices.append({"error": names[1]})
            continue
        values = _read(base.with_suffix(".bin"))
        k = len(contrasts[i]["sizes"])
        matrices.append({"matrix": values.reshape(len(names), k), "names": names})
    out["contrasts"] = matrices
    return out


# ---------------------------------------------------------------------------------------
# References in Python: mpmath and scipy, in a process of their own
# ---------------------------------------------------------------------------------------


def _mp() -> Any:
    import mpmath

    mpmath.mp.dps = 30
    return mpmath


def _t_sf_mp(mp: Any, x: Any, nu: Any) -> Any:
    """P(T > x) for T ~ t(nu); nu None for the normal."""
    if nu is None:
        return mp.ncdf(-x)
    if x >= 0:
        return (
            mp.betainc(nu / 2, mp.mpf(1) / 2, 0, nu / (nu + x * x), regularized=True)
            / 2
        )
    return 1 - _t_sf_mp(mp, -x, nu)


def _t_pdf_mp(mp: Any, x: Any, nu: Any) -> Any:
    if nu is None:
        return mp.npdf(x)
    return mp.exp(
        mp.loggamma((nu + 1) / 2)
        - mp.loggamma(nu / 2)
        - mp.log(nu * mp.pi) / 2
        - (nu + 1) / 2 * mp.log1p(x * x / nu)
    )


def _quad(mp: Any, f: Any, points: list[Any]) -> tuple[Any, Any]:
    """mp.quad with a relative tolerance, and its error estimate. mpmath stops refining
    a piece once its error estimate falls below 10^-dps in absolute terms, which leaves
    an integral of 1e-87 with 9 correct digits, so the integrand is first divided by
    its largest value at the points and the middles of the finite pieces."""
    probes = [x for x in points if mp.isfinite(x)]
    probes += [
        (a + b) / 2
        for a, b in zip(points[:-1], points[1:])
        if mp.isfinite(a) and mp.isfinite(b)
    ]
    scale = max((abs(f(x)) for x in probes), default=mp.mpf(0)) or mp.mpf(1)
    value, error = mp.quad(lambda x: f(x) / scale, points, error=True, maxdegree=10)
    return value * scale, error * scale


def plane_reference(
    rho: float, upper: list[float], df: float, two_sided: bool
) -> tuple[float, float, float]:
    """Probability and complement of a box of two coordinates, by Ding's conditional
    distribution, at 30 digits, and mpmath's error estimate of the smaller."""
    mp = _mp()
    nu = None if df == INF else mp.mpf(df)
    nu1 = None if nu is None else nu + 1
    r = mp.mpf(rho)
    b1, b2 = mp.mpf(upper[0]), mp.mpf(upper[1])
    lo1, lo2 = (-b1, -b2) if two_sided else (-mp.inf, -mp.inf)

    def scale(t: Any) -> Any:
        if nu is None:
            return mp.sqrt(1 - r * r)
        return mp.sqrt((nu + t * t) * (1 - r * r) / (nu + 1))

    def outside(t: Any) -> Any:
        s = scale(t)
        out = _t_sf_mp(mp, (b1 - r * t) / s, nu1)
        if lo1 != -mp.inf:
            out += _t_sf_mp(mp, (r * t - lo1) / s, nu1)
        return out

    def inside(t: Any) -> Any:
        s = scale(t)
        hi, lo = (b1 - r * t) / s, (lo1 - r * t) / s
        if lo == -mp.inf:
            return _t_sf_mp(mp, -hi, nu1)  # P(U <= hi) without cancellation
        if lo >= 0:
            return _t_sf_mp(mp, lo, nu1) - _t_sf_mp(mp, hi, nu1)
        if hi <= 0:
            return _t_sf_mp(mp, -hi, nu1) - _t_sf_mp(mp, -lo, nu1)
        return 1 - _t_sf_mp(mp, hi, nu1) - _t_sf_mp(mp, -lo, nu1)

    points = {lo2, b2}
    if lo2 < 0 < b2:
        points.add(mp.mpf(0))
    for j in range(1, 4):  # the scale of the limits, for wide ranges of t
        for edge in (b2, lo2):
            if edge not in (mp.inf, -mp.inf) and edge != 0:
                point = edge / mp.mpf(10) ** j
                if lo2 < point < b2:
                    points.add(point)
    # Near a finite end, where the mass of a small box or a tail orthant sits.
    for edge, inward in ((b2, -1), (lo2, 1)):
        if edge not in (mp.inf, -mp.inf) and abs(edge) < 1e6:
            for step in (
                "1",
                "0.3",
                "0.1",
                "0.03",
                "0.01",
                "0.003",
                "0.001",
                "1e-4",
            ):
                point = edge + inward * mp.mpf(step)
                if lo2 < point < b2:
                    points.add(point)
    if r != 0:
        for edge in (b1, lo1):
            if edge != -mp.inf:
                t = edge / r
                if lo2 < t < b2:
                    points.add(t)
    ordered = sorted(points)
    marginal = _t_sf_mp(mp, b2, nu)
    if lo2 != -mp.inf:
        marginal += _t_sf_mp(mp, -lo2, nu)
    complement, complement_error = _quad(
        mp, lambda t: _t_pdf_mp(mp, t, nu) * outside(t), ordered
    )
    probability, probability_error = _quad(
        mp, lambda t: _t_pdf_mp(mp, t, nu) * inside(t), ordered
    )
    complement += marginal
    smaller_error = probability_error if probability <= complement else complement_error
    return float(probability), float(complement), float(smaller_error)


def rank_two_reference(
    angles: np.ndarray, upper: np.ndarray, df: float, two_sided: bool
) -> tuple[float, float]:
    """Probability and complement of a box for R_ij = cos(theta_i - theta_j), by the
    average over directions, at 30 digits."""
    mp = _mp()
    nu = None if df == INF else mp.mpf(df)
    a = [(mp.cos(mp.mpf(float(x))), mp.sin(mp.mpf(float(x)))) for x in angles]
    up = [mp.mpf(float(u)) for u in upper]
    lo = [-u for u in up] if two_sided else [-mp.inf] * len(up)

    def tail(radius: Any) -> Any:
        if radius == mp.inf:
            return mp.mpf(0)
        if nu is None:
            return mp.exp(-radius * radius / 2)
        return mp.exp(-nu / 2 * mp.log1p(radius * radius / nu))

    def ray(theta: Any, complement: bool) -> Any:
        c, s = mp.cos(theta), mp.sin(theta)
        r_in, r_out = mp.mpf(0), mp.inf
        for (x, y), low, high in zip(a, lo, up):
            slope = x * c + y * s
            if slope > 0:
                r_out = min(r_out, high / slope)
                r_in = max(r_in, low / slope)
            elif slope < 0:
                r_out = min(r_out, low / slope)
                r_in = max(r_in, high / slope)
        if r_in >= r_out:
            return mp.mpf(1) if complement else mp.mpf(0)
        inside = tail(r_in) - tail(r_out)
        return 1 - inside if complement else inside

    cuts = set()
    for i, (x, y) in enumerate(a):
        phi = mp.atan2(y, x)
        for turn in (0, mp.mpf(1) / 2, 1, mp.mpf(3) / 2):
            cuts.add(phi + turn * mp.pi)
        for j in range(i + 1, len(a)):
            for li in (lo[i], up[i]):
                for lj in (lo[j], up[j]):
                    if mp.isinf(li) or mp.isinf(lj):
                        continue
                    wx, wy = li * a[j][0] - lj * x, li * a[j][1] - lj * y
                    if wx == 0 and wy == 0:
                        continue
                    psi = mp.atan2(wy, wx)
                    cuts.add(psi + mp.pi / 2)
                    cuts.add(psi + 3 * mp.pi / 2)
    ordered = sorted({c % (2 * mp.pi) for c in cuts} | {mp.mpf(0), 2 * mp.pi})
    probability = _quad(mp, lambda t: ray(t, False), ordered)[0] / (2 * mp.pi)
    complement = _quad(mp, lambda t: ray(t, True), ordered)[0] / (2 * mp.pi)
    return float(probability), float(complement)


GAUSS_LEGENDRE = np.polynomial.legendre.leggauss(24)


def _factor_integral(
    s: float,
    lam: np.ndarray,
    sig: np.ndarray,
    up: np.ndarray,
    two_sided: bool,
    panel: float,
) -> tuple[float, float]:
    """The probability and complement of the box given the chi variable s, integrated
    over the shared normal factor by Gauss-Legendre panels around the peaks."""
    from scipy import special

    signs = (1.0, -1.0) if two_sided else (1.0,)
    peaks = [0.0] + [sign * a * u * s for a, u in zip(lam, up) for sign in signs]
    spans = sorted((p - 14.0, p + 14.0) for p in peaks)
    merged = [list(spans[0])]
    for low, high in spans[1:]:
        if low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    nodes, weights = [], []
    for low, high in merged:
        n = max(1, math.ceil((high - low) / panel))
        edges = np.linspace(low, high, n + 1)
        half = 0.5 * np.diff(edges)
        middle = 0.5 * (edges[1:] + edges[:-1])
        nodes.append((middle[:, None] + half[:, None] * GAUSS_LEGENDRE[0]).ravel())
        weights.append((half[:, None] * GAUSS_LEGENDRE[1]).ravel())
    w = np.concatenate(nodes)
    mass = np.concatenate(weights) * np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
    log_inside = np.zeros_like(w)
    for a, sg, u in zip(lam, sig, up):
        high = (u * s - a * w) / sg
        out = special.ndtr(-high)
        if two_sided:
            out = out + special.ndtr((-u * s - a * w) / sg)
        with np.errstate(divide="ignore"):
            log_inside += np.log1p(-np.minimum(out, 1.0))
    complement = -np.expm1(log_inside)
    return float(np.sum(mass * np.exp(log_inside))), float(np.sum(mass * complement))


def product_reference(
    lam: np.ndarray, upper: np.ndarray, df: float, two_sided: bool
) -> tuple[float, float]:
    """Probability and complement of a box for R_ij = lambda_i lambda_j, by Dunnett's
    integral in double precision."""
    from scipy import integrate

    lam = np.asarray(lam, float)
    sig = np.sqrt(1.0 - lam * lam)
    up = np.asarray(upper, float)
    panel = 0.25 * float(sig.min())
    if df == INF:
        return _factor_integral(1.0, lam, sig, up, two_sided, panel)
    log_constant = math.log(2.0) + df / 2 * math.log(df / 2) - math.lgamma(df / 2)

    def density(s: float) -> float:
        if s <= 0.0:
            return 0.0
        return math.exp(log_constant + (df - 1) * math.log(s) - df * s * s / 2)

    b = float(np.abs(up).max())
    star = math.sqrt(df / (df + b * b))
    edges = [0.0] + [star * f for f in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 30.0)]
    out = []
    for which in (0, 1):

        def integrand(s: float) -> float:
            return (
                density(s) * _factor_integral(s, lam, sig, up, two_sided, panel)[which]
            )

        total = 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            total += integrate.quad(
                integrand, low, high, epsabs=0, epsrel=1e-12, limit=200
            )[0]
        total += integrate.quad(
            integrand, edges[-1], INF, epsabs=0, epsrel=1e-12, limit=200
        )[0]
        out.append(total)
    return out[0], out[1]


def product_reference_mp(
    lam: np.ndarray, upper: np.ndarray, two_sided: bool
) -> tuple[float, float]:
    """The same for the normal, a one-dimensional integral over the factor, at 30
    digits."""
    mp = _mp()
    lam_mp = [mp.mpf(float(x)) for x in lam]
    sig = [mp.sqrt(1 - x * x) for x in lam_mp]
    up = [mp.mpf(float(u)) for u in upper]

    def inside(w: Any) -> Any:
        product = mp.mpf(1)
        for a, s, u in zip(lam_mp, sig, up):
            high = (u - a * w) / s
            product *= mp.ncdf(high) - (mp.ncdf((-u - a * w) / s) if two_sided else 0)
        return product

    signs = (1, -1) if two_sided else (1,)
    peaks = sorted(
        {mp.mpf(0)} | {sign * a * u for a, u in zip(lam_mp, up) for sign in signs}
    )
    points = [-mp.inf] + peaks + [mp.inf]
    probability = _quad(mp, lambda w: mp.npdf(w) * inside(w), points)[0]
    complement = _quad(mp, lambda w: mp.npdf(w) * (1 - inside(w)), points)[0]
    return float(probability), float(complement)


def product_quantile(
    lam: np.ndarray, df: float, p: float, two_sided: bool
) -> tuple[float, float]:
    """The equicoordinate quantile of a product correlation, as the root of Dunnett's
    integral, and the slope of the complement there."""
    from scipy import optimize, stats

    k = lam.size
    tail = 1.0 - p
    isf = (lambda x: stats.norm.isf(x)) if df == INF else (lambda x: stats.t.isf(x, df))
    if two_sided:
        low, high = isf(tail / 2), isf(tail / (2 * k))
    else:
        low, high = isf(tail), isf(tail / k)

    def gap(c: float) -> float:
        complement = product_reference(lam, np.full(k, c), df, two_sided)[1]
        return math.log(complement) - math.log(tail)

    root = optimize.brentq(gap, low - 1e-3, high + 1e-3, xtol=1e-13, rtol=1e-14)
    h = 1e-4
    slope = (
        product_reference(lam, np.full(k, root - h), df, two_sided)[1]
        - product_reference(lam, np.full(k, root + h), df, two_sided)[1]
    ) / (2 * h)
    return root, slope


def plane_quantile(rho: float, df: float, p: float, two_sided: bool) -> float:
    """The equicoordinate quantile of two coordinates, as the root of Ding's formula."""
    from scipy import optimize, stats

    tail = 1.0 - p
    isf = (lambda x: stats.norm.isf(x)) if df == INF else (lambda x: stats.t.isf(x, df))
    low, high = (
        (isf(tail / 2), isf(tail / 4)) if two_sided else (isf(tail), isf(tail / 2))
    )

    def gap(c: float) -> float:
        complement = plane_reference(rho, [c, c], df, two_sided)[1]
        return math.log(complement) - math.log(tail)

    return float(optimize.brentq(gap, low - 1e-3, high + 1e-3, xtol=1e-14, rtol=1e-15))


def scipy_reference(
    correlation: np.ndarray, upper: np.ndarray, df: float, two_sided: bool
) -> tuple[float, float]:
    """scipy's complement of the box: the mean of 16 calls with different random
    states, and its standard error."""
    from scipy import stats

    lower = -upper if two_sided else None
    values = [
        1.0
        - stats.multivariate_t.cdf(
            upper,
            shape=correlation,
            df=df,
            allow_singular=True,
            lower_limit=lower,
            random_state=seed,
        )
        for seed in range(SCIPY_CALLS)
    ]
    return float(np.mean(values)), float(
        np.std(values, ddof=1) / math.sqrt(SCIPY_CALLS)
    )


def _chi_given_exceedance(c: float, df: float) -> tuple[np.ndarray, np.ndarray]:
    """The distribution function of log S given Z > c S, on a grid of log S: the
    density of S, df S^2 chi-squared on df degrees of freedom, times P(Z > c S), with
    the Jacobian of log S; for drawing S by inversion."""
    from scipy import special, stats

    u = np.linspace(-14.0, 5.0, 400_001)
    s = np.exp(u)
    log_density = (
        math.log(2.0 * df)
        + 2.0 * u
        + stats.chi2.logpdf(df * s * s, df)
        + special.log_ndtr(-c * s)
    )
    density = np.exp(log_density - log_density.max())
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(u))]
    )
    return u, cumulative / cumulative[-1]


def aloe_reference(
    correlation: np.ndarray, upper: np.ndarray, df: float, two_sided: bool, seed: int
) -> tuple[float, float]:
    """P(T_i > c_i for some i), or P(|T_i| > c_i for some i), by ALOE (Owen, Maximov
    and Chertkov 2019): ALOE_DRAWS draws from the mixture of the distributions of T
    given each exceedance, coordinate i with weight P_i / sum P, each draw weighted by
    sum P over the number of coordinates beyond their limits there. Given T_i > c_i, S
    is drawn by inversion of its distribution, Z_i above c_i S and the other normals
    given Z_i; two-sided, the exceedance below -c_i is the mirror image of this one,
    which gives the same count of |T_j| > c_j. Returns the estimate and its standard
    error. Plain Monte Carlo, with numpy and scipy."""
    from scipy import special

    rng = np.random.default_rng(seed)
    upper = np.asarray(upper, float)
    k = upper.size
    factor = 2.0 if two_sided else 1.0
    if df == INF:
        marginal = factor * special.ndtr(-upper)
    else:
        from scipy import stats

        marginal = factor * stats.t.sf(upper, df)
    total = float(marginal.sum())
    counts = rng.multinomial(ALOE_DRAWS, marginal / total)
    inverse_weights = []
    for i in range(k):
        others = [j for j in range(k) if j != i]
        rho = correlation[others, i]
        values, vectors = np.linalg.eigh(
            correlation[np.ix_(others, others)] - np.outer(rho, rho)
        )
        loadings = vectors * np.sqrt(np.clip(values, 0.0, None))
        grid = None if df == INF else _chi_given_exceedance(upper[i], df)
        done = 0
        while done < counts[i]:
            n = min(100_000, counts[i] - done)
            if grid is None:
                s = np.ones(n)
            else:
                s = np.exp(np.interp(rng.random(n), grid[1], grid[0]))
            z_i = -special.ndtri(rng.random(n) * special.ndtr(-upper[i] * s))
            z = np.empty((n, k))
            z[:, i] = z_i
            z[:, others] = (
                z_i[:, None] * rho + rng.standard_normal((n, k - 1)) @ loadings.T
            )
            t = z / s[:, None]
            if two_sided:
                t = np.abs(t)  # the sign of the whole vector does not matter
            # The draw's own coordinate is beyond its limit by construction.
            beyond = 1 + (t[:, others] > upper[others]).sum(axis=1)
            inverse_weights.append(1.0 / beyond)
            done += n
    values = np.concatenate(inverse_weights)
    return total * float(values.mean()), total * float(values.std(ddof=1)) / math.sqrt(
        values.size
    )


def one_dimension_reference(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mp = _mp()
    nu = mp.mpf(4.5)
    upper = [float(_t_sf_mp(mp, mp.mpf(float(x)), nu)) for x in points["upper"]]
    below = [float(_t_sf_mp(mp, -mp.mpf(float(x)), nu)) for x in points["upper"]]
    within = [
        float(
            mp.betainc(mp.mpf(1) / 2, nu / 2, 0, x * x / (nu + x * x), regularized=True)
        )
        for x in (mp.mpf(float(v)) for v in points["within"])
    ]
    return {
        "above": np.array(upper),
        "below": np.array(below),
        "within": np.array(within),
    }


def _task(job: tuple[str, dict[str, Any]]) -> Any:
    kind, case = job
    if kind == "plane":
        return plane_reference(
            case["rho"], case["upper"], case["df"], case["two_sided"]
        )
    if kind == "plane_product":
        rho = case["rho"]
        lam = np.array([math.sqrt(abs(rho)), math.copysign(math.sqrt(abs(rho)), rho)])
        return product_reference(
            lam, np.asarray(case["upper"]), case["df"], case["two_sided"]
        )
    if kind == "product":
        return product_reference(
            case["lam"], case["upper"], case["df"], case["two_sided"]
        )
    if kind == "product_mp":
        return product_reference_mp(case["lam"], case["upper"], case["two_sided"])
    if kind == "distinct":
        return product_reference(
            case["lam"], case["upper"], case["df"], case["two_sided"]
        )
    if kind == "aloe":
        return aloe_reference(
            case["correlation"],
            case["upper"],
            case["df"],
            case["two_sided"],
            int(case["seed"]),
        )
    if kind == "rank_two":
        return rank_two_reference(
            case["angles"], case["upper"], case["df"], case["two_sided"]
        )
    if kind == "product_quantile":
        return product_quantile(case["lam"], case["df"], case["p"], case["two_sided"])
    if kind == "plane_quantile":
        return plane_quantile(case["rho"], case["df"], case["p"], case["two_sided"])
    if kind == "scipy":
        return scipy_reference(
            case["correlation"], case["upper"], case["df"], case["two_sided"]
        )
    raise ValueError(kind)


def reference_main(directory: Path) -> None:
    """Compute the mpmath and scipy references of the cases in ``directory``."""
    import mpmath
    import scipy

    cases = json.loads((directory / "cases.json").read_text(), object_hook=_from_json)
    jobs: list[tuple[str, dict]] = []
    for kind in REFERENCE_KINDS:
        jobs += [(kind, case) for case in cases[kind]]
    # The slow jobs first, so that the workers finish together.
    order = sorted(
        range(len(jobs)),
        key=lambda j: jobs[j][0] not in ("product_quantile", "plane_product"),
    )
    results: list[Any] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for j, value in zip(order, pool.map(_task, [jobs[j] for j in order])):
            results[j] = value
    out: dict[str, Any] = {}
    start = 0
    for kind in REFERENCE_KINDS:
        count = len(cases[kind])
        out[kind] = results[start : start + count]
        start += count
    out["one_dimension"] = one_dimension_reference(cases["one_dimension"])
    out["versions"] = {"mpmath": mpmath.__version__, "scipy": scipy.__version__}
    (directory / "reference.json").write_text(json.dumps(out, default=_to_json))


def _to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def _from_json(value: dict) -> Any:
    if "__array__" in value:
        return np.array(value["__array__"], dtype=float)
    return value


# ---------------------------------------------------------------------------------------
# CytePool's values
# ---------------------------------------------------------------------------------------


def product_correlation(lam: np.ndarray) -> np.ndarray:
    out = np.outer(lam, lam)
    np.fill_diagonal(out, 1.0)
    return out


def rank_two_correlation(angles: np.ndarray) -> np.ndarray:
    return np.cos(angles[:, None] - angles[None, :])


def cytepool_box(
    correlation: np.ndarray, case: dict, seed: int, relative_error: float = TIGHT
) -> tuple[float, float, float]:
    from cytepool.stats.multivariate_t import multivariate_t_cdf

    result = multivariate_t_cdf(
        case["upper"],
        correlation,
        case["df"],
        two_sided=case["two_sided"],
        relative_error=relative_error,
        rng=seed,
    )
    return float(result.probability), float(result.complement), float(result.error)


def cytepool_planes(cases: list[dict]) -> np.ndarray:
    """(n, 3): probability, complement and error of the two-coordinate cases."""
    from cytepool.stats.multivariate_t import multivariate_t_cdf

    out = np.empty((len(cases), 3))
    for two_sided in (False, True):
        chosen = [i for i, c in enumerate(cases) if c["two_sided"] == two_sided]
        correlations = np.array(
            [[[1.0, cases[i]["rho"]], [cases[i]["rho"], 1.0]] for i in chosen]
        )
        result = multivariate_t_cdf(
            np.array([cases[i]["upper"] for i in chosen]),
            correlations,
            np.array([cases[i]["df"] for i in chosen]),
            two_sided=two_sided,
        )
        out[chosen, 0] = result.probability
        out[chosen, 1] = result.complement
        out[chosen, 2] = result.error
    return out


def cytepool_quantile(
    correlation: np.ndarray, case: dict, seed: int
) -> tuple[float, float]:
    from cytepool.stats.multivariate_t import multivariate_t_quantile

    result = multivariate_t_quantile(
        case["p"],
        correlation,
        case["df"],
        two_sided=case["two_sided"],
        relative_error=TIGHT,
        rng=seed,
    )
    return float(result.quantile), float(result.error)


def cytepool_contrast_matrix(case: dict) -> tuple[np.ndarray, tuple[str, ...]] | str:
    from cytepool.stats.multiple_contrasts import dunnett_contrasts, williams_contrasts

    sizes = dict(zip(case["names"], (int(n) for n in case["sizes"])))
    try:
        if case["type"] == "Dunnett":
            contrasts = dunnett_contrasts(sizes, control=case["base"])
        else:
            contrasts = williams_contrasts(sizes)
    except ValueError as error:
        return str(error)
    return np.asarray(contrasts.matrix).T, contrasts.names


def cytepool_multcomp(family: dict, seed: int) -> list[dict[str, np.ndarray]]:
    """multiple_contrast_test of the family's features, grouped by alternative."""
    from cytepool.stats.multiple_contrasts import multiple_contrast_test

    features = family["features"]
    out: list[dict[str, np.ndarray]] = [{} for _ in features]
    for j, alternative in enumerate(MULTCOMP_ALTERNATIVES):
        chosen = [i for i, f in enumerate(features) if f["alternative"] == alternative]
        if not chosen:
            continue
        test = multiple_contrast_test(
            np.array([features[i]["estimates"] for i in chosen]),
            np.array([features[i]["covariance"] for i in chosen]),
            family["contrasts"],
            df=np.array([features[i]["df"] for i in chosen]),
            alternative=alternative,
            relative_error=TIGHT,
            rng=seed + j,
        )
        for row, i in enumerate(chosen):
            out[i] = {
                "estimate": np.asarray(test.estimate[row]),
                "standard_error": np.asarray(test.standard_error[row]),
                "t": np.asarray(test.t[row]),
                "p_value": np.asarray(test.p_value[row]),
                "p_value_error": np.asarray(test.p_value_error[row]),
            }
    return out


# ---------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------


class Rows:
    """The printed checks: check, reference, value, tolerance and outcome."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, str]] = []
        self.failures = 0

    def bound(
        self,
        check: str,
        reference: str,
        value: float,
        tolerance: float,
        fmt: str = ".1e",
    ) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                format(value, fmt),
                f"{tolerance:g}",
                "pass" if passed else "FAIL",
            )
        )

    def calibrated(
        self, check: str, reference: str, z: np.ndarray, extra: str = ""
    ) -> None:
        largest = float(np.max(np.abs(z))) if z.size else 0.0
        passed = bool(largest <= CALIBRATED)
        self.failures += not passed
        value = f"{largest:.2f} standard errors" + (f"; {extra}" if extra else "")
        self.rows.append(
            (
                check,
                reference,
                value,
                f"{CALIBRATED:g} standard errors",
                "pass" if passed else "FAIL",
            )
        )

    def within(
        self, check: str, reference: str, value: float, low: float, high: float
    ) -> None:
        passed = bool(low <= value <= high)
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{value:.3f}",
                f"{low:.2f} to {high:.2f}",
                "pass" if passed else "FAIL",
            )
        )

    def control(
        self, check: str, reference: str, value: float, tolerance: float
    ) -> None:
        passed = bool(value > tolerance)
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{value:.1f} standard errors",
                f"above {tolerance:g}",
                "control" if passed else "FAIL",
            )
        )

    def information(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "information"))

    def deliberate(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "deliberate difference"))


def _z(ours: float, reference: float, error: float) -> float:
    """(ours - reference) / error, with the error at least the rounding of the
    reference: an estimator of zero variance states an error of 0."""
    return (ours - reference) / max(error, 1e-14 * abs(reference))


def _relative(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(ours) - np.asarray(reference)) / np.abs(reference)


def _statistic_difference(ours: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference relative to max(|reference|, 1); infinite where either
    is NaN, as a failed reference call leaves."""
    reference = np.asarray(reference)
    difference = np.abs(np.asarray(ours) - reference) / np.maximum(
        np.abs(reference), 1.0
    )
    return float(np.inf if np.isnan(difference).any() else difference.max())


def rms_bounds(n: int) -> tuple[float, float]:
    """Where the root-mean-square of n honest z-scores on 15 degrees of freedom lies,
    within 4 simulation standard errors."""
    spread = 4.0 * math.sqrt(T15_VARIANCE / n)
    return math.sqrt(T15_MEAN - spread), math.sqrt(T15_MEAN + spread)


def _smaller(probability: float, complement: float) -> float:
    return min(probability, complement)


def compare_probabilities(
    rows: Rows, inputs: dict, reference: dict, ours: dict
) -> dict:
    found: dict[str, Any] = {}
    # One coordinate.
    one = inputs["one_dimension"]
    expected = reference["one_dimension"]
    from cytepool.stats.multivariate_t import multivariate_t_cdf

    above = multivariate_t_cdf(one["upper"][:, None], [[1.0]], 4.5)
    within = multivariate_t_cdf(one["within"][:, None], [[1.0]], 4.5, two_sided=True)
    difference = max(
        float(np.max(_relative(above.complement, expected["above"]))),
        float(np.max(_relative(above.probability, expected["below"]))),
        float(np.max(_relative(within.probability, expected["within"]))),
    )
    rows.bound(
        "One coordinate, df 4.5: probability and complement of one-sided limits from -30 "
        "to 1e8 and of two-sided limits from 1e-12 to 1e4, 28 values",
        "mpmath, 30 digits",
        difference,
        EXACT,
    )
    # Two coordinates.
    planes = inputs["plane"]
    plane_values = ours["plane"]
    plane_reference_values = np.array(reference["plane"])[:, :2]
    plane_reference_errors = np.array(reference["plane"])[:, 2]
    errors = []
    for df in PLANE_DF:
        chosen = [i for i, c in enumerate(planes) if c["df"] == df]
        differences = []
        for i in chosen:
            p_ref, q_ref = plane_reference_values[i]
            p, q, error = plane_values[i]
            error = error + plane_reference_errors[i]
            if p_ref <= q_ref:
                differences.append(abs(p - p_ref) / p_ref)
                errors.append((abs(p - p_ref), error, p_ref))
            else:
                differences.append(abs(q - q_ref) / q_ref)
                errors.append((abs(q - q_ref), error, q_ref))
        label = "the normal" if df == INF else f"df {df:g}"
        rows.bound(
            f"Two coordinates, {label}: the smaller of the probability and the complement, "
            f"6 correlations from -0.95 to 0.99, 8 boxes each ({len(chosen)} cases)",
            "Ding's conditional distribution, mpmath, 30 digits",
            max(differences),
            CLOSE,
        )
    ratio = max(actual / (error + 4e-16 * value) for actual, error, value in errors)
    rows.bound(
        f"Two coordinates: the actual error over the stored error bound and mpmath's "
        f"error estimate, {len(errors)} cases",
        "Ding's conditional distribution",
        ratio,
        1.0,
        ".2f",
    )
    agreement = max(
        abs(float(np.min(a)) - float(np.min(b))) / float(np.min(b))
        for a, b in zip(
            reference["plane_product"],
            [reference["plane"][i][:2] for i in inputs["plane_product_index"]],
        )
    )
    rows.bound(
        f"Reference self-check: Dunnett's integral in double precision against Ding's "
        f"formula at 30 digits, {len(inputs['plane_product_index'])} two-coordinate cases",
        "Ding's conditional distribution",
        agreement,
        EXACT,
    )
    # Product correlations.
    products = inputs["product"]
    product_values = np.array(reference["product"])
    z_tight = []
    for k in PRODUCT_K:
        for df in PRODUCT_DF:
            chosen = [
                i
                for i, c in enumerate(products)
                if c["lam"].size == k and c["df"] == df
            ]
            z = np.array(
                [
                    _z(
                        ours["product"][i][1],
                        product_values[i][1],
                        ours["product"][i][2],
                    )
                    for i in chosen
                ]
            )
            relative = max(
                abs(ours["product"][i][1] - product_values[i][1])
                / _smaller(*product_values[i])
                for i in chosen
            )
            z_tight += list(z)
            label = "the normal" if df == INF else f"df {df:g}"
            smallest = min(_smaller(*product_values[i]) for i in chosen)
            rows.calibrated(
                f"Product correlations, k = {k}, {label}: 9 boxes, smaller side down to "
                f"{smallest:.1e}",
                "Dunnett's integral",
                z,
                f"largest relative difference {relative:.1e}",
            )
    mp_agreement = max(
        abs(float(np.min(a)) - float(np.min(product_values[i])))
        / float(np.min(product_values[i]))
        for a, i in zip(reference["product_mp"], inputs["product_mp_index"])
    )
    rows.bound(
        f"Reference self-check: Dunnett's integral in double precision against mpmath at "
        f"30 digits, the normal, {len(inputs['product_mp_index'])} product cases",
        "mpmath, 30 digits",
        mp_agreement,
        EXACT,
    )
    # Rank two.
    rank_two = inputs["rank_two"]
    rank_values = np.array(reference["rank_two"])
    for label in dict.fromkeys(c["label"] for c in rank_two):
        chosen = [i for i, c in enumerate(rank_two) if c["label"] == label]
        z = np.array(
            [
                _z(ours["rank_two"][i][1], rank_values[i][1], ours["rank_two"][i][2])
                for i in chosen
            ]
        )
        z_tight += list(z)
        relative = max(
            abs(ours["rank_two"][i][1] - rank_values[i][1]) / _smaller(*rank_values[i])
            for i in chosen
        )
        smallest = min(_smaller(*rank_values[i]) for i in chosen)
        rows.calibrated(
            f"Rank-two correlations, {label}: df 3, 12.5 and the normal, 6 boxes each, "
            f"smaller side down to {smallest:.1e}",
            "average over directions, mpmath, 30 digits",
            z,
            f"largest relative difference {relative:.1e}",
        )
    # A limit per coordinate.
    distinct = inputs["distinct"]
    distinct_values = np.array(reference["distinct"])
    for k in DISTINCT_K:
        chosen = [i for i, c in enumerate(distinct) if c["lam"].size == k]
        z = np.array(
            [
                _z(
                    ours["distinct"][i][1],
                    distinct_values[i][1],
                    ours["distinct"][i][2],
                )
                for i in chosen
            ]
        )
        z_tight += list(z)
        relative = max(
            abs(ours["distinct"][i][1] - distinct_values[i][1])
            / _smaller(*distinct_values[i])
            for i in chosen
        )
        smallest = min(_smaller(*distinct_values[i]) for i in chosen)
        rows.calibrated(
            f"Product correlations with a limit per coordinate, k = {k}: df 2.5, 7, 20.5 "
            f"and the normal, 6 tails split unevenly and 2 small boxes each, smaller "
            f"side down to {smallest:.1e}",
            "Dunnett's integral",
            z,
            f"largest relative difference {relative:.1e}",
        )
    z_tight_array = np.array(z_tight)
    low, high = rms_bounds(z_tight_array.size)
    rows.within(
        f"Calibration at relative_error 1e-3: root-mean-square of (estimate - reference) "
        f"/ error, {z_tight_array.size} problems of the product, rank-two and "
        "per-coordinate rows",
        "sqrt(15/13) = 1.07 for honest errors",
        float(np.sqrt(np.mean(z_tight_array**2))),
        low,
        high,
    )
    z_default = np.concatenate(
        [
            [
                _z(
                    ours["product_default"][i][1],
                    product_values[i][1],
                    ours["product_default"][i][2],
                )
                for i in range(len(products))
            ],
            [
                _z(
                    ours["rank_two_default"][i][1],
                    rank_values[i][1],
                    ours["rank_two_default"][i][2],
                )
                for i in range(len(rank_two))
            ],
            [
                _z(
                    ours["distinct_default"][i][1],
                    distinct_values[i][1],
                    ours["distinct_default"][i][2],
                )
                for i in range(len(distinct))
            ],
        ]
    )
    low, high = rms_bounds(z_default.size)
    rows.within(
        f"Calibration at the default relative_error 0.01, fresh seeds: root-mean-square "
        f"of (estimate - reference) / error, {z_default.size} problems",
        "sqrt(15/13) = 1.07 for honest errors",
        float(np.sqrt(np.mean(z_default**2))),
        low,
        high,
    )
    found["z_default"] = z_default
    found["z_tight"] = z_tight_array
    return found


def compare_general(
    rows: Rows, inputs: dict, reference: dict, r: dict, ours: dict
) -> None:
    general = inputs["general"]
    for label in dict.fromkeys(c["label"] for c in general):
        for df in MVTNORM_DF:
            chosen = [
                i
                for i, c in enumerate(general)
                if c["label"] == label and c["df"] == df
            ]
            z = []
            for i in chosen:
                value, error = r["general"][i][:2]
                p, q, se = ours["general"][i]
                z.append((q - (1.0 - value)) / math.hypot(se, error / 3.0))
            name = "the normal" if df == INF else f"df {df:g}"
            rows.calibrated(
                f"{label}, {name}: complements of 3 two-sided and 2 one-sided boxes from "
                "0.5 to 1e-3",
                "mvtnorm pmvt",
                np.array(z),
            )
    aloe = inputs["aloe"]
    for label in dict.fromkeys(c["label"] for c in aloe):
        chosen = [i for i, c in enumerate(aloe) if c["label"] == label]
        z = []
        for i in chosen:
            value, error = reference["aloe"][i]
            p, q, se = ours["aloe"][i]
            z.append((q - value) / math.hypot(se, error))
        smallest = min(reference["aloe"][i][0] for i in chosen)
        rows.calibrated(
            f"{label}: df 3, 8 and the normal, complements down to {smallest:.1e}",
            f"ALOE, {ALOE_DRAWS:.0e} draws",
            np.array(z),
        )
    scipy_cases = inputs["scipy"]
    singular_ratio = []
    for label in dict.fromkeys(c["label"] for c in scipy_cases):
        chosen = [i for i, c in enumerate(scipy_cases) if c["label"] == label]
        correlation = scipy_cases[chosen[0]]["correlation"]
        if np.linalg.matrix_rank(correlation, tol=1e-8) < correlation.shape[0]:
            # scipy's values for a singular correlation exceed Bonferroni's bound.
            for i in chosen:
                case = scipy_cases[i]
                tails = (2.0 if case["two_sided"] else 1.0) * np.asarray(
                    _t_sf(case["upper"], case["df"])
                )
                singular_ratio.append(reference["scipy"][i][0] / min(1.0, tails.sum()))
            continue
        z = []
        for i in chosen:
            value, error = reference["scipy"][i]
            p, q, se = ours["scipy"][i]
            z.append((q - value) / math.hypot(se, error))
        rows.calibrated(
            f"{label}, df 4.5, 10 and 20.5: complements of 5 boxes each",
            "scipy multivariate_t.cdf, 16 calls",
            np.array(z),
        )
    rows.information(
        "scipy with a singular correlation, Tukey's contrasts of 4 groups (rank 3 of "
        f"6), {len(singular_ratio)} boxes: its complement over Bonferroni's upper bound",
        "scipy multivariate_t.cdf, allow_singular=True",
        f"up to {max(singular_ratio):.2f}; CytePool's agree with mvtnorm's",
    )
    rows.deliberate(
        "Non-integer df, such as the moderated df of limma: multivariate_t_cdf takes them",
        "mvtnorm pmvt",
        f"pmvt refuses df = 4.5: {r['refusal']!r}",
    )
    for i, case in enumerate(inputs["tail"]):
        value, error, seconds = r["tail"][i]
        exact = reference["product"][inputs["tail_index"][i]][1]
        rows.information(
            f"mvtnorm in the tail: k = 3, rho = 0.5, df 10, two-sided limit "
            f"{case['upper'][0]:g}, with abseps = 1e-12 and maxpts = 1e7",
            "Dunnett's integral",
            f"pmvt {1.0 - value:.4g} (stated error {error:.1e}, {seconds:.1f} s) "
            f"against {exact:.4g}",
        )


def compare_quantiles(
    rows: Rows, inputs: dict, reference: dict, r: dict, ours: dict
) -> None:
    from cytepool.stats.multivariate_t import multivariate_t_cdf

    planes = inputs["plane_quantile"]
    differences = [
        abs(ours["plane_quantile"][i][0] - reference["plane_quantile"][i])
        / abs(reference["plane_quantile"][i])
        for i in range(len(planes))
    ]
    rows.bound(
        f"Quantiles of two coordinates: rho -0.5, 0.5 and 0.9, df 4.5 and the normal, p "
        f"0.9, 0.95 and 0.99, one- and two-sided ({len(planes)} cases)",
        "roots of Ding's formula",
        max(differences),
        CLOSE,
    )
    products = inputs["product_quantile"]
    trips = []
    for k in (3, 5, 8):
        chosen = [i for i, c in enumerate(products) if c["lam"].size == k]
        z = []
        for i in chosen:
            q, error = ours["product_quantile"][i]
            root, slope = reference["product_quantile"][i]
            z.append((q - root) / error)
            case = products[i]
            trip = multivariate_t_cdf(
                np.full(k, q), product_correlation(case["lam"]), case["df"],
                two_sided=case["two_sided"], relative_error=TIGHT, rng=SEED + 900 + i,
            )  # fmt: skip
            trips.append(
                (float(trip.probability) - case["p"])
                / math.hypot(float(trip.error), slope * error)
            )
        rows.calibrated(
            f"Quantiles of product correlations, k = {k}: df 7 and the normal, p 0.9, "
            "0.95 and 0.99, one- and two-sided",
            "roots of Dunnett's integral",
            np.array(z),
        )
    general = inputs["general_quantile"]
    qmvt_differences = []
    for label in dict.fromkeys(c["label"] for c in general):
        chosen = [i for i, c in enumerate(general) if c["label"] == label]
        z = []
        for i in chosen:
            case = general[i]
            quantile, value, error, below, above = r["quantiles"][i]
            q, q_error = case["ours"], case["ours_error"]
            slope = (above - below) / (2.0 * case["h"])
            z.append((value - case["p"]) / math.hypot(error / 3.0, slope * q_error))
            if not math.isnan(quantile):
                qmvt_differences.append(abs(quantile - q) / q)
            trip = multivariate_t_cdf(
                np.full(case["correlation"].shape[0], q), case["correlation"], case["df"],
                two_sided=case["two_sided"], relative_error=TIGHT, rng=SEED + 950 + i,
            )  # fmt: skip
            trips.append(
                (float(trip.probability) - case["p"])
                / math.hypot(float(trip.error), slope * q_error)
            )
        rows.calibrated(
            f"Quantiles, {label}: df 10 and the normal, p 0.95 and 0.99, one- and "
            "two-sided; the probability at CytePool's quantile",
            "mvtnorm pmvt",
            np.array(z),
        )
    rows.calibrated(
        f"Quantiles: multivariate_t_cdf at the quantile against p, {len(trips)} cases",
        "the round trip",
        np.array(trips),
    )
    rows.information(
        f"Quantiles: mvtnorm's qmvt against CytePool's, {len(qmvt_differences)} cases",
        "mvtnorm qmvt",
        f"largest relative difference {max(qmvt_differences):.1e}",
    )


def compare_contrasts(rows: Rows, inputs: dict, r: dict) -> None:
    cases = inputs["contrasts"]
    dunnett = [i for i, c in enumerate(cases) if c["type"] == "Dunnett"]
    williams = [
        i
        for i, c in enumerate(cases)
        if c["type"] == "Williams" and len(c["sizes"]) > 2
    ]
    differences, names_differ = [], 0
    for i in dunnett:
        matrix, names = cytepool_contrast_matrix(cases[i])
        differences.append(float(np.max(np.abs(matrix - r["contrasts"][i]["matrix"]))))
        names_differ += list(names) != r["contrasts"][i]["names"]
    rows.bound(
        f"dunnett_contrasts: matrices, 2 to 9 groups, the control first, in the middle and "
        f"last ({len(dunnett)} cases)",
        'contrMat(n, type = "Dunnett", base)',
        max(differences),
        0.0,
        "g",
    )
    rows.bound(
        f"dunnett_contrasts: names ({len(dunnett)} cases)",
        'contrMat(n, type = "Dunnett", base)',
        float(names_differ),
        0.0,
        "g",
    )
    differences = []
    for i in williams:
        matrix, _ = cytepool_contrast_matrix(cases[i])
        differences.append(float(np.max(np.abs(matrix - r["contrasts"][i]["matrix"]))))
    rows.bound(
        f"williams_contrasts: matrices, 3 to 9 groups of equal and unequal sizes "
        f"({len(williams)} cases)",
        'contrMat(n, type = "Williams")',
        max(differences),
        EXACT,
        ".1e",
    )
    example = williams[-1]
    _, names = cytepool_contrast_matrix(cases[example])
    rows.deliberate(
        "williams_contrasts: names that say which doses are pooled",
        'contrMat(n, type = "Williams")',
        f"{names[1]!r} where contrMat writes {r['contrasts'][example]['names'][1]!r}",
    )
    two = [
        i
        for i, c in enumerate(cases)
        if c["type"] == "Williams" and len(c["sizes"]) == 2
    ][0]
    _, names = cytepool_contrast_matrix(cases[two])
    rows.deliberate(
        "williams_contrasts with two groups: Dunnett's one contrast",
        'contrMat(n, type = "Williams")',
        f"{names[0]!r}; contrMat refuses: {r['contrasts'][two]['error'].strip()!r}",
    )


def compare_multcomp(rows: Rows, inputs: dict, r: dict, ours: dict) -> None:
    offset = 0
    # multcomp's p-values below its default absolute error, 1e-3, which it cannot
    # resolve: (CytePool's, multcomp's).
    unresolved: list[tuple[float, float]] = []
    for f, family in enumerate(inputs["multcomp"]):
        features = family["features"]
        m = family["contrasts"].shape[1]
        alternatives = [
            a
            for a in MULTCOMP_ALTERNATIVES
            if any(x["alternative"] == a for x in features)
        ]
        gaps = []
        for alternative in alternatives:
            chosen = [
                i for i, x in enumerate(features) if x["alternative"] == alternative
            ]
            statistics, z, compared = 0.0, [], 0
            for i in chosen:
                values = r["multcomp"][offset + i]
                estimate, se, t, p = (values[j * m : (j + 1) * m] for j in range(4))
                error = values[4 * m]
                bonferroni = values[4 * m + 1 : 5 * m + 1]
                mine = ours["multcomp"][f][i]
                statistics = max(
                    statistics,
                    _statistic_difference(mine["estimate"], estimate),
                    _statistic_difference(mine["standard_error"], se),
                    _statistic_difference(mine["t"], t),
                )
                combined = np.hypot(mine["p_value_error"], error / 3.0)
                resolved = p >= MULTCOMP_ABSEPS
                z += list(((mine["p_value"] - p) / combined)[resolved])
                compared += int(resolved.sum())
                unresolved += list(zip(mine["p_value"][~resolved], p[~resolved]))
                for j in range(m):
                    if 1e-3 <= mine["p_value"][j] <= 0.5:
                        gaps.append(
                            (bonferroni[j] - mine["p_value"][j])
                            / mine["p_value_error"][j]
                        )
            rows.bound(
                f"multiple_contrast_test, {family['label']}, {alternative}: estimates, "
                f"standard errors and t of {len(chosen)} features, df 5, 12, 40 and the "
                "normal",
                "multcomp glht(parm(b, V, df))",
                statistics,
                EXACT,
            )
            rows.calibrated(
                f"multiple_contrast_test, {family['label']}, {alternative}: the "
                f"{compared} single-step p-values of {len(chosen)} features from 1e-3 "
                "to 1",
                "multcomp summary(glht)",
                np.array(z),
            )
        if "random" not in family["label"] and gaps:
            rows.control(
                f"Bonferroni's p-values against the single-step ones, {family['label']}, "
                f"{len(gaps)} p-values from 1e-3 to 0.5: the smallest gap",
                'multcomp adjusted("bonferroni")',
                float(np.min(gaps)),
                CALIBRATED,
            )
        offset += len(features)
    ratios = np.array([ours_p / theirs for ours_p, theirs in unresolved if theirs > 0])
    zeros = sum(theirs == 0 for _, theirs in unresolved)
    rows.information(
        f"multiple_contrast_test: the {len(unresolved)} single-step p-values that "
        "multcomp gives below 1e-3, its default absolute error, which cannot resolve "
        "them",
        "multcomp summary(glht)",
        f"CytePool's over multcomp's: median {np.median(ratios):.3f}, from "
        f"{ratios.min():.3g} to {ratios.max():.3g}; multcomp gives 0 for {zeros}",
    )


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


def write_fixtures(inputs: dict, reference: dict, r: dict) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    planes = inputs["plane"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "two_dimensions.npz",
        rho=np.array([c["rho"] for c in planes]),
        df=np.array([c["df"] for c in planes]),
        two_sided=np.array([c["two_sided"] for c in planes]),
        upper=np.array([c["upper"] for c in planes]),
        reference=np.array(reference["plane"])[:, :2],
    )
    products = inputs["product"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "product_correlations.npz",
        k=np.array([c["lam"].size for c in products]),
        lam=np.array(
            [
                np.pad(c["lam"], (0, 10 - c["lam"].size), constant_values=np.nan)
                for c in products
            ]
        ),
        df=np.array([c["df"] for c in products]),
        two_sided=np.array([c["two_sided"] for c in products]),
        upper=np.array([c["upper"][0] for c in products]),
        reference=np.array(reference["product"]),
    )
    rank_two = inputs["rank_two"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "rank_two.npz",
        k=np.array([c["angles"].size for c in rank_two]),
        angles=np.array(
            [
                np.pad(c["angles"], (0, 6 - c["angles"].size), constant_values=np.nan)
                for c in rank_two
            ]
        ),
        df=np.array([c["df"] for c in rank_two]),
        two_sided=np.array([c["two_sided"] for c in rank_two]),
        upper=np.array([c["upper"][0] for c in rank_two]),
        reference=np.array(reference["rank_two"]),
    )
    general = inputs["general"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "general.npz",
        k=np.array([c["correlation"].shape[0] for c in general]),
        correlation=np.array(
            [
                np.pad(
                    c["correlation"],
                    (0, 10 - c["correlation"].shape[0]),
                    constant_values=np.nan,
                )
                for c in general
            ]
        ),
        df=np.array([c["df"] for c in general]),
        two_sided=np.array([c["two_sided"] for c in general]),
        upper=np.array([c["upper"][0] for c in general]),
        mvtnorm=np.array([values[:2] for values in r["general"]]),
    )
    products = inputs["product_quantile"]
    planes = inputs["plane_quantile"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "quantiles.npz",
        product_k=np.array([c["lam"].size for c in products]),
        product_lam=np.array(
            [
                np.pad(c["lam"], (0, 8 - c["lam"].size), constant_values=np.nan)
                for c in products
            ]
        ),
        product_df=np.array([c["df"] for c in products]),
        product_p=np.array([c["p"] for c in products]),
        product_two_sided=np.array([c["two_sided"] for c in products]),
        product_root=np.array([v[0] for v in reference["product_quantile"]]),
        plane_rho=np.array([c["rho"] for c in planes]),
        plane_df=np.array([c["df"] for c in planes]),
        plane_p=np.array([c["p"] for c in planes]),
        plane_two_sided=np.array([c["two_sided"] for c in planes]),
        plane_root=np.array(reference["plane_quantile"]),
    )
    one = inputs["one_dimension"]
    expected = reference["one_dimension"]
    np.savez_compressed(
        FIXTURE_DIRECTORY / "one_dimension.npz",
        upper=one["upper"],
        above=expected["above"],
        below=expected["below"],
        within_limit=one["within"],
        within=expected["within"],
    )
    contrasts = []
    for case, values in zip(inputs["contrasts"], r["contrasts"]):
        record = {"type": case["type"], "control": case["base"], "names": case["names"],
                  "sizes": [int(n) for n in case["sizes"]]}  # fmt: skip
        if "error" in values:
            record["error"] = values["error"]
        else:
            record["matrix"] = values["matrix"].tolist()
            record["contrast_names"] = values["names"]
        contrasts.append(record)
    (FIXTURE_DIRECTORY / "contrasts.json").write_text(
        json.dumps(contrasts, indent=1) + "\n"
    )
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for f, family in enumerate(inputs["multcomp"]):
        m = family["contrasts"].shape[1]
        arrays[f"family{f}__label"] = np.array(family["label"])
        arrays[f"family{f}__contrasts"] = family["contrasts"]
        chosen = list(range(12))  # every df and alternative of the family
        features = [family["features"][i] for i in chosen]
        values = np.array([r["multcomp"][offset + i] for i in chosen])
        arrays[f"family{f}__estimates"] = np.array([x["estimates"] for x in features])
        arrays[f"family{f}__covariance"] = np.array([x["covariance"] for x in features])
        arrays[f"family{f}__df"] = np.array([x["df"] for x in features])
        arrays[f"family{f}__alternative"] = np.array(
            [x["alternative"] for x in features]
        )
        arrays[f"family{f}__estimate"] = values[:, :m]
        arrays[f"family{f}__standard_error"] = values[:, m : 2 * m]
        arrays[f"family{f}__t"] = values[:, 2 * m : 3 * m]
        arrays[f"family{f}__p_value"] = values[:, 3 * m : 4 * m]
        arrays[f"family{f}__p_value_error"] = values[:, 4 * m]
        arrays[f"family{f}__bonferroni"] = values[:, 4 * m + 1 :]
        offset += len(family["features"])
    np.savez_compressed(FIXTURE_DIRECTORY / "multcomp.npz", **arrays)
    print(f"wrote 8 fixture files to {FIXTURE_DIRECTORY}")


# ---------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------


def build_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    inputs: dict[str, Any] = {
        "plane": plane_cases(),
        "product": product_cases(rng),
        "rank_two": rank_two_cases(rng),
        "one_dimension": one_dimension_points(),
        "product_quantile": product_quantile_cases(rng),
        "plane_quantile": plane_quantile_cases(),
        "contrasts": contrast_cases(rng),
        "multcomp": multcomp_families(rng),
    }
    general_rng = np.random.default_rng(SEED + 1)
    inputs["general"] = general_cases(general_rng, MVTNORM_DF)
    general_rng = np.random.default_rng(SEED + 1)
    inputs["scipy"] = general_cases(general_rng, SCIPY_DF)
    general_rng = np.random.default_rng(SEED + 1)
    inputs["general_quantile"] = general_quantile_cases(general_rng)
    # Their own generators, so that adding them left the other cases as they were.
    inputs["distinct"] = distinct_cases(np.random.default_rng(SEED + 2))
    inputs["aloe"] = aloe_cases(np.random.default_rng(SEED + 3))
    # Every seventh two-coordinate case, for the self-check of Dunnett's integral, and
    # the product cases of the normal, for the check against mpmath.
    inputs["plane_product_index"] = list(range(0, len(inputs["plane"]), 7))
    inputs["product_mp_index"] = [
        i for i, c in enumerate(inputs["product"]) if c["df"] == INF
    ]
    # mvtnorm's tail, the design probe's case: k = 3, rho = 0.5, df 10.
    lam = np.full(3, math.sqrt(0.5))
    tail = []
    for limit in (6.0, 10.0, 20.0):
        inputs["product"].append(
            {
                "lam": lam,
                "df": 10.0,
                "two_sided": True,
                "upper": np.full(3, limit),
                "extra": True,
            }
        )
        tail.append({"correlation": product_correlation(lam), "df": 10.0, "two_sided": True,
                     "upper": np.full(3, limit)})  # fmt: skip
    inputs["tail"] = tail
    inputs["tail_index"] = list(
        range(len(inputs["product"]) - 3, len(inputs["product"]))
    )
    return inputs


def reference_cases(inputs: dict) -> dict[str, Any]:
    """The cases the reference process computes, as JSON."""

    def plain(case: dict) -> dict:
        return {key: value for key, value in case.items() if key != "label"}

    def upper_list(case: dict) -> dict:
        out = plain(case)
        out["upper"] = [float(x) for x in np.ravel(case["upper"])]
        return out

    return {
        "plane": [upper_list(c) for c in inputs["plane"]],
        "plane_product": [
            upper_list(inputs["plane"][i]) for i in inputs["plane_product_index"]
        ],
        "product": [plain(c) for c in inputs["product"]],
        "product_mp": [plain(inputs["product"][i]) for i in inputs["product_mp_index"]],
        "distinct": [plain(c) for c in inputs["distinct"]],
        "rank_two": [plain(c) for c in inputs["rank_two"]],
        "product_quantile": [plain(c) for c in inputs["product_quantile"]],
        "plane_quantile": [plain(c) for c in inputs["plane_quantile"]],
        "scipy": [plain(c) for c in inputs["scipy"]],
        "aloe": [plain(c) for c in inputs["aloe"]],
        "one_dimension": inputs["one_dimension"],
    }


def cytepool_values(inputs: dict) -> dict[str, Any]:
    ours: dict[str, Any] = {"plane": cytepool_planes(inputs["plane"])}
    ours["product"] = [
        cytepool_box(product_correlation(c["lam"]), c, SEED + 1000 + i)
        for i, c in enumerate(inputs["product"])
    ]
    ours["product_default"] = [
        cytepool_box(
            product_correlation(c["lam"]), c, SEED + 2000 + i, relative_error=0.01
        )
        for i, c in enumerate(inputs["product"])
    ]
    ours["distinct"] = [
        cytepool_box(product_correlation(c["lam"]), c, SEED + 10000 + i)
        for i, c in enumerate(inputs["distinct"])
    ]
    ours["distinct_default"] = [
        cytepool_box(
            product_correlation(c["lam"]), c, SEED + 11000 + i, relative_error=0.01
        )
        for i, c in enumerate(inputs["distinct"])
    ]
    ours["aloe"] = [
        cytepool_box(c["correlation"], c, SEED + 12000 + i)
        for i, c in enumerate(inputs["aloe"])
    ]
    ours["rank_two"] = [
        cytepool_box(rank_two_correlation(c["angles"]), c, SEED + 3000 + i)
        for i, c in enumerate(inputs["rank_two"])
    ]
    ours["rank_two_default"] = [
        cytepool_box(
            rank_two_correlation(c["angles"]), c, SEED + 4000 + i, relative_error=0.01
        )
        for i, c in enumerate(inputs["rank_two"])
    ]
    ours["general"] = [
        cytepool_box(c["correlation"], c, SEED + 5000 + i)
        for i, c in enumerate(inputs["general"])
    ]
    ours["scipy"] = [
        cytepool_box(c["correlation"], c, SEED + 6000 + i)
        for i, c in enumerate(inputs["scipy"])
    ]
    ours["plane_quantile"] = [
        cytepool_quantile(
            np.array([[1.0, c["rho"]], [c["rho"], 1.0]]), c, SEED + 7000 + i
        )
        for i, c in enumerate(inputs["plane_quantile"])
    ]
    ours["product_quantile"] = [
        cytepool_quantile(product_correlation(c["lam"]), c, SEED + 8000 + i)
        for i, c in enumerate(inputs["product_quantile"])
    ]
    ours["multcomp"] = [
        cytepool_multcomp(family, SEED + 9000 + 10 * f)
        for f, family in enumerate(inputs["multcomp"])
    ]
    return ours


def general_quantiles(inputs: dict) -> None:
    """CytePool's quantiles of the general cases, which mvtnorm then checks."""
    for i, case in enumerate(inputs["general_quantile"]):
        q, error = cytepool_quantile(case["correlation"], case, SEED + 8500 + i)
        case["ours"], case["h"] = q, 0.02
        case["ours_error"] = error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    parser.add_argument("--benchmark", action="store_true", help="time 20,000 genes")
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.reference is not None:
        reference_main(arguments.reference)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0
    started = time.perf_counter()
    inputs = build_inputs()
    general_quantiles(inputs)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "python").mkdir()
        (directory / "python" / "cases.json").write_text(
            json.dumps(reference_cases(inputs), default=_to_json)
        )
        process = subprocess.Popen(
            [sys.executable, __file__, "--reference", str(directory / "python")]
        )
        (directory / "r").mkdir()
        r = run_r(
            inputs["general"],
            inputs["tail"],
            inputs["general_quantile"],
            [
                dict(x, contrasts=family["contrasts"])
                for family in inputs["multcomp"]
                for x in family["features"]
            ],
            inputs["contrasts"],
            directory / "r",
        )
        print(f"R finished after {time.perf_counter() - started:.0f} s")
        ours = cytepool_values(inputs)
        print(f"CytePool finished after {time.perf_counter() - started:.0f} s")
        if process.wait() != 0:
            raise RuntimeError("the reference process failed")
        reference = json.loads(
            (directory / "python" / "reference.json").read_text(),
            object_hook=_from_json,
        )
        print(f"references finished after {time.perf_counter() - started:.0f} s")
    print(
        "Reference: "
        + ", ".join(r["versions"])
        + f", mpmath {reference['versions']['mpmath']}, scipy {reference['versions']['scipy']}"
    )
    rows = Rows()
    compare_probabilities(rows, inputs, reference, ours)
    compare_general(rows, inputs, reference, r, ours)
    compare_quantiles(rows, inputs, reference, r, ours)
    compare_contrasts(rows, inputs, r)
    compare_multcomp(rows, inputs, r, ours)
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows.rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if rows.failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(inputs, reference, r)
    outcomes: dict[str, int] = {}
    for row in rows.rows:
        outcomes[row[4]] = outcomes.get(row[4], 0) + 1
    print(
        f"{len(rows.rows)} checks: "
        + ", ".join(f"{n} {k}" for k, n in outcomes.items())
    )
    print(f"{rows.failures} failed")
    return 1 if rows.failures else 0


# ---------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------


def _dose_genes(
    rng: np.random.Generator, n_genes: int, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Group means and covariances of genes with gene-specific variances per group, 90%
    without an effect and 10% with effects that give p-values from 1e-2 to 1e-12."""
    k = sizes.size
    variances = 0.2 * rng.chisquare(8.0, (n_genes, k)) / 8.0
    covariances = np.zeros((n_genes, k, k))
    covariances[:, np.arange(k), np.arange(k)] = variances / sizes
    means = rng.normal(0.0, 1.0, (n_genes, k)) * np.sqrt(variances / sizes)
    changed = rng.random(n_genes) < 0.1
    scale = np.sqrt(variances[:, 1:] / sizes[1:] + variances[:, :1] / sizes[0])
    effects = rng.uniform(3.0, 14.0, changed.sum())[:, None] * scale[changed]
    means[changed, 1:] += effects * rng.random((changed.sum(), k - 1)) ** 2
    return means, covariances


def benchmark(n_genes: int = 20_000) -> None:
    """Time multiple_contrast_test against the target of 1 ms per gene on 2 threads."""
    import numba

    from cytepool.stats import multivariate_t
    from cytepool.stats.linear_models import fit_linear_model
    from cytepool.stats.multiple_contrasts import (
        dunnett_contrasts,
        multiple_contrast_test,
        williams_contrasts,
    )

    rng = np.random.default_rng(SEED + 99)
    n_samples = 200
    groups = np.repeat(np.arange(4), n_samples // 4)
    design = np.column_stack(
        [np.ones(n_samples)] + [(groups == g).astype(float) for g in (1, 2, 3)]
        + [rng.normal(50.0, 10.0, n_samples), rng.normal(0.0, 1.0, n_samples)]
    )  # fmt: skip
    y = rng.normal(6.0, 1.0, (n_samples, n_genes))
    weights = np.exp(rng.normal(0.0, 0.5, y.shape))

    def median_seconds(call: Any) -> float:
        call()
        seconds = []
        for _ in range(3):
            start = time.perf_counter()
            call()
            seconds.append(time.perf_counter() - start)
        return float(np.median(seconds))

    baseline = median_seconds(lambda: fit_linear_model(y, design, weights=weights))
    print(f"Benchmark on {numba.get_num_threads()} threads, {n_genes:,} genes")
    print(f"Baseline: weighted fit_linear_model, 200 samples: {baseline:.3f} s, "
          f"{baseline / n_genes * 1e6:.1f} us per gene")  # fmt: skip
    print(
        "| scenario | seconds | ms per gene | baselines | p-values short of the target |"
    )
    print("| --- | --- | --- | --- | --- |")
    scenarios = [
        ("9 groups, 8 Dunnett contrasts, two-sided", 9, "dunnett", "two-sided", 0.01),
        ("3 groups, 2 Dunnett contrasts, two-sided", 3, "dunnett", "two-sided", 0.01),
        ("4 groups, 3 Dunnett contrasts, two-sided", 4, "dunnett", "two-sided", 0.01),
        ("9 groups, 8 Williams contrasts, greater", 9, "williams", "greater", 0.01),
        (
            "9 groups, 8 Dunnett contrasts, relative_error 1e-3",
            9,
            "dunnett",
            "two-sided",
            1e-3,
        ),
    ]
    for label, k, family, alternative, relative_error in scenarios:
        sizes = np.array([4] + [3] * (k - 1))
        names = ["vehicle"] + [f"dose{j}" for j in range(1, k)]
        means, covariances = _dose_genes(rng, n_genes, sizes.astype(float))
        contrasts = (
            dunnett_contrasts(names)
            if family == "dunnett"
            else williams_contrasts(dict(zip(names, (int(n) for n in sizes))))
        )
        result = {}

        def run() -> None:
            result["test"] = multiple_contrast_test(
                means, covariances, contrasts, df=20.5, estimate_names=names,
                alternative=alternative, relative_error=relative_error, rng=1,
            )  # fmt: skip

        seconds = median_seconds(run)
        test = result["test"]
        smaller = np.minimum(test.p_value, 1.0 - test.p_value)
        allowed = relative_error * smaller + np.finfo(float).eps * test.p_value
        short = int(np.count_nonzero(test.p_value_error > allowed))
        print(f"| {label} | {seconds:.2f} | {seconds / n_genes * 1e3:.3f} | "
              f"{seconds / baseline:.0f} | {short} |")  # fmt: skip
    # Points per problem in the first scenario, on 1,000 genes: a problem stops at the
    # smallest limit of points at which its error meets the target, as the rounds up
    # to a limit do not depend on it.
    sizes = np.array([4] + [3] * 8)
    names = ["vehicle"] + [f"dose{j}" for j in range(1, 9)]
    means, covariances = _dose_genes(
        np.random.default_rng(SEED + 98), 1000, sizes.astype(float)
    )
    contrasts = dunnett_contrasts(names)
    stopped = np.full((1000, 8), np.nan)
    default_limit = multivariate_t._MOST_POINTS
    try:
        limit = 4
        while limit <= default_limit and np.isnan(stopped).any():
            multivariate_t._MOST_POINTS = limit
            test = multiple_contrast_test(
                means, covariances, contrasts, df=20.5, estimate_names=names, rng=1
            )
            smaller = np.minimum(test.p_value, 1.0 - test.p_value)
            allowed = 0.01 * smaller + np.finfo(float).eps * test.p_value
            done = test.p_value_error <= allowed
            stopped[np.isnan(stopped) & done] = 16 * limit
            limit *= 2
    finally:
        multivariate_t._MOST_POINTS = default_limit
    print(f"Points per problem, 9 groups, 1,000 genes: mean {np.nanmean(stopped):.0f}, "
          f"median {np.nanmedian(stopped):.0f}, 90th percentile "
          f"{np.nanpercentile(stopped, 90):.0f}, largest {np.nanmax(stopped):.0f}")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
