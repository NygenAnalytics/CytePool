"""Adjustment of p-values for multiple testing, and the proportion of true nulls.

:func:`adjust_p_values` follows R's ``p.adjust`` exactly (R Core Team 2025),
including its conventions for missing values and for families declared larger than
the p-values observed in them, so adjusted p-values can be compared with R output
directly.

:func:`proportion_of_true_nulls` estimates pi0, the proportion of the tests whose null
hypothesis is true, from their p-values, as qvalue's ``pi0est`` (Storey and
Tibshirani 2003); 1 - pi0, pi1, is the proportion with an effect, which the p-values
of one study among the genes that another study finds measure as a replication rate.

The proportion of true nulls:
    Null p-values are uniform, so the m p-values at least lambda hold about
    pi0 m (1 - lambda) null ones, and pi0(lambda) = #{p >= lambda} / (m (1 - lambda))
    estimates pi0 with a bias that falls, and a variance that grows, as lambda rises
    (Storey 2002). qvalue counts the p-values at least lambda, where the paper counts
    those above it (found by probing). With one lambda this is the estimate; with
    several, in increasing order:

    - ``"smoother"`` (qvalue's default): the value at the largest lambda of a cubic
      smoothing spline with ``smooth_df`` equivalent degrees of freedom through
      pi0(lambda), or with ``smooth_log`` through log pi0(lambda), back-transformed
      (Storey and Tibshirani 2003), as R's ``smooth.spline`` fits it (found by
      probing; see ``cytepool.stats.smoothing``).
    - ``"bootstrap"``: pi0(lambda) at the lambda that minimizes a bootstrap estimate
      of its mean squared error (Storey, Taylor and Siegmund 2004), in closed form:
      the variance of a binomial count, W (1 - W / m) / (m (1 - lambda))^2 for the
      W = #{p >= lambda}, plus the square of pi0(lambda) minus a reference value,
      which qvalue takes as the 10% quantile (R's type 7) of the pi0(lambda) (found by
      probing); the smallest lambda where several minimize it.

    The estimate is capped at 1.

    Deliberate differences from qvalue 2.38.0:
        - qvalue stops with an error where no p-value reaches the largest lambda, with
          the smoother and the bootstrap alike (found by probing). Here a count of zero
          enters the smoother as it is, and has no variance in the bootstrap; the
          smoother of logarithms, which a count of zero leaves without a value,
          refuses it.
        - qvalue stops where the estimate is at most 0; here it is 0.
        - With more than 49 values of lambda, R's smoother keeps a knot at some of
          them only, a regression spline; here every lambda is a knot, as for fewer.
          Duplicated values of lambda are refused.

References:
    Holm S (1979). A simple sequentially rejective multiple test procedure. Scandinavian
    Journal of Statistics 6, 65-70.

    Hochberg Y (1988). A sharper Bonferroni procedure for multiple tests of
    significance. Biometrika 75, 800-802. doi:10.1093/biomet/75.4.800

    Hommel G (1988). A stagewise rejective multiple test procedure based on a modified
    Bonferroni test. Biometrika 75, 383-386. doi:10.1093/biomet/75.2.383

    Simes RJ (1986). An improved Bonferroni procedure for multiple tests of
    significance. Biometrika 73, 751-754. doi:10.1093/biomet/73.3.751

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Benjamini Y, Yekutieli D (2001). The control of the false discovery rate in multiple
    testing under dependency. Annals of Statistics 29, 1165-1188.
    doi:10.1214/aos/1013699998

    R Core Team (2025). R: A Language and Environment for Statistical Computing. R
    Foundation for Statistical Computing, Vienna. https://www.R-project.org/

    Storey JD (2002). A direct approach to false discovery rates. Journal of the Royal
    Statistical Society B 64, 479-498. doi:10.1111/1467-9868.00346

    Storey JD, Tibshirani R (2003). Statistical significance for genomewide studies.
    Proceedings of the National Academy of Sciences 100, 9440-9445.
    doi:10.1073/pnas.1530509100

    Storey JD, Taylor JE, Siegmund D (2004). Strong control, conservative point
    estimation and simultaneous conservative consistency of false discovery rates: a
    unified approach. Journal of the Royal Statistical Society B 66, 187-205.
    doi:10.1111/j.1467-9868.2004.00439.x
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cytepool._jit import jit
from cytepool._masked_arrays import read_only, unmasked_array
from cytepool._options import checked_choice, checked_flag, index_integer, real_number
from cytepool.stats.smoothing import _smoothing_spline_fit

_EULER_GAMMA = 0.5772156649015329
# qvalue's lambda, seq(0.05, 0.95, 0.05) in R: 0.05 + 0.05 k for k = 0, ..., 17, and
# 0.95 itself, where 0.05 + 0.05 * 18 rounds to 0.9500000000000001 (found by probing).
_DEFAULT_LAMBDAS = np.append(0.05 + 0.05 * np.arange(18.0), 0.95)
_PI0_METHODS = ("smoother", "bootstrap")
_FEWEST_LAMBDAS = 4  # several lambdas must be at least this many, as for qvalue
_BOOTSTRAP_QUANTILE = 0.1  # of the pi0(lambda), the reference of the bootstrap

# Accepted method names (lower case) and the procedure each one selects.
_PROCEDURES = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "hochberg": "hochberg",
    "hommel": "hommel",
    "bh": "bh",
    "fdr": "bh",
    "benjamini-hochberg": "bh",
    "by": "by",
    "benjamini-yekutieli": "by",
    "none": "none",
}


def adjust_p_values(
    p: ArrayLike,
    method: str = "bh",
    *,
    n: int | None = None,
    axis: int | None = None,
) -> np.ndarray:
    """Adjust p-values for multiple testing, as R's ``p.adjust`` does.

    Bonferroni, Holm (1979), Hochberg (1988) and Hommel (1988) control the
    family-wise error rate; Benjamini and Hochberg (1995) and Benjamini and
    Yekutieli (2001) control the false discovery rate. With ``n`` tests in a
    family and sorted p-values ``p_(1) <= ... <= p_(n)``, the adjusted value of
    ``p_(i)`` is the following, capped at 1:

    - ``"bonferroni"``: ``n p_(i)``.
    - ``"holm"``: ``max over j <= i of (n - j + 1) p_(j)``.
    - ``"hochberg"``: ``min over j >= i of (n - j + 1) p_(j)``.
    - ``"hommel"``: the largest Simes (1986) p-value, ``min over j of k p_(j:S) / j``,
      over all sets ``S`` of ``k`` tests that contain test ``i``, where
      ``p_(j:S)`` is the j-th smallest p-value in ``S``.
    - ``"bh"``: ``min over j >= i of n p_(j) / j``.
    - ``"by"``: the ``"bh"`` value times ``1 + 1/2 + ... + 1/n``.
    - ``"none"``: ``p_(i)``.

    Every method takes O(m log m) time for m p-values, Hommel's included.

    Args:
        p: p-values in [0, 1]. NaN marks a missing value; it stays NaN and is
            not counted as a test.
        method: The procedure, case-insensitive: ``"bonferroni"``, ``"holm"``,
            ``"hochberg"``, ``"hommel"``, ``"bh"`` (also ``"fdr"`` and
            ``"benjamini-hochberg"``), ``"by"`` (also ``"benjamini-yekutieli"``)
            or ``"none"``. The default is ``"bh"``; R's default is ``"holm"``.
        n: Number of tests in each family, when larger than the number of
            non-missing p-values. As in R, the unobserved p-values are taken to
            be larger than all observed ones for Bonferroni and Holm, and equal
            to 1 for the other methods. Defaults to the number of non-missing
            p-values in each family.
        axis: None adjusts all p-values together as one family. An integer
            adjusts each slice along that axis as its own family; for a matrix
            of features by contrasts, ``axis=0`` adjusts each column.

    Returns:
        The adjusted p-values as float64, with the shape of ``p``.

    Raises:
        TypeError: If ``p`` is not real-valued, ``method`` is not a string, ``n``
            or ``axis`` is not an integer, or ``p``, ``n`` or ``axis`` is a numpy
            masked array.
        ValueError: If a p-value is outside [0, 1], ``method`` is unknown,
            ``n`` is smaller than the number of non-missing p-values in a
            family or larger than 2**53, or ``axis`` is out of range.
    """
    procedure = _procedure(method)
    values = _p_value_array(p)
    if n is not None:
        n = _integer(n, "n")
    if axis is None:
        rows = values.reshape(1, values.size)
        return _adjust_rows(rows, procedure, n).reshape(values.shape)
    axis = _axis_index(axis, values.ndim)
    moved = np.moveaxis(values, axis, -1)
    rows = moved.reshape(math.prod(moved.shape[:-1]), moved.shape[-1])
    adjusted = _adjust_rows(rows, procedure, n).reshape(moved.shape)
    return np.ascontiguousarray(np.moveaxis(adjusted, -1, axis))


def _procedure(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}")
    procedure = _PROCEDURES.get(method.lower())
    if procedure is None:
        names = ", ".join(repr(name) for name in _PROCEDURES)
        raise ValueError(f"unknown method {method!r}; expected one of {names}")
    return procedure


def _p_value_array(p: ArrayLike) -> np.ndarray:
    values = unmasked_array(p, "p", missing="NaN")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"p must hold real numbers, got dtype {values.dtype}")
    values = values.astype(np.float64, copy=False)
    outside = ~((values >= 0.0) & (values <= 1.0)) & ~np.isnan(values)
    if outside.any():
        raise ValueError(
            "p-values must lie in [0, 1] or be NaN; found "
            f"{np.count_nonzero(outside)} outside, such as {float(values[outside][0])}"
        )
    return values


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got a boolean")
    return index_integer(value, name, expected="an integer")


def _axis_index(axis: int, ndim: int) -> int:
    index = _integer(axis, "axis")
    if not -ndim <= index < ndim:
        raise ValueError(f"axis {index} is out of range for a {ndim}-dimensional array")
    return index % ndim


def _adjust_rows(rows: np.ndarray, procedure: str, n: int | None) -> np.ndarray:
    """Adjust each row of a 2-D array as one family of tests."""
    # Found by probing R: p.adjust leaves NA out of the default n and checks
    # an explicit n against the non-missing values only, although its help
    # page asks for n >= length(p).
    observed = np.count_nonzero(~np.isnan(rows), axis=1).astype(np.int64)
    if n is None:
        total = observed
    else:
        largest = int(observed.max(initial=0))
        if n < largest:
            raise ValueError(
                "n must be at least the number of non-missing p-values in each "
                f"family ({largest}), got {n}"
            )
        if n > 2**53:
            raise ValueError(f"n must be at most 2**53, got {n}")
        total = np.full(rows.shape[0], n, dtype=np.int64)
    if procedure == "none":
        return rows.copy()
    n_tests = total[:, np.newaxis].astype(np.float64)
    if procedure == "bonferroni":
        bonferroni: np.ndarray = np.minimum(n_tests * rows, 1.0)
        return bonferroni

    # Sorting moves NaN to the end of each row, after the observed p-values.
    order = np.argsort(rows, axis=1)
    ranked = np.take_along_axis(rows, order, axis=1)
    rank = np.arange(1.0, rows.shape[1] + 1.0)
    if procedure == "holm":
        adjusted = np.maximum.accumulate((n_tests - rank + 1.0) * ranked, axis=1)
    elif procedure == "hochberg":
        adjusted = _running_min_from_end((n_tests - rank + 1.0) * ranked)
    elif procedure == "bh":
        # This order of operations, and the BY sum rounded once, reproduce
        # R's results to the last bit (found by probing).
        adjusted = _running_min_from_end(n_tests / rank * ranked)
    elif procedure == "by":
        harmonic = _harmonic_numbers(total)[:, np.newaxis]
        adjusted = _running_min_from_end(harmonic * n_tests / rank * ranked)
    else:
        adjusted = _hommel_sorted(ranked, observed, total)
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(rows)
    np.put_along_axis(result, order, adjusted, axis=1)
    return result


def _running_min_from_end(values: np.ndarray) -> np.ndarray:
    """Running minimum of each row, taken from its last entry backwards.

    The NaN that end a row stay NaN, and ``fmin`` keeps them from spreading.
    """
    return np.fmin.accumulate(values[:, ::-1], axis=1)[:, ::-1]


def _harmonic_numbers(counts: np.ndarray) -> np.ndarray:
    """Return ``1 + 1/2 + ... + 1/k`` for each count ``k``."""
    distinct, positions = np.unique(counts, return_inverse=True)
    sums = [_harmonic_number(int(k)) for k in distinct]
    return np.asarray(sums, dtype=np.float64)[positions]


def _harmonic_number(k: int) -> float:
    """Return ``1 + 1/2 + ... + 1/k``, to within one unit in the last place.

    Up to 2**20 terms are summed with a single rounding. Beyond that, the
    expansion ``log k + gamma + 1/(2k) - 1/(12k^2)`` is used: its first
    omitted term, ``1/(120k^4)``, is below 1e-25, and its cost does not grow
    with ``k``.
    """
    if k <= 2**20:
        return math.fsum(1.0 / np.arange(1.0, k + 1.0))
    inverse = 1.0 / k
    return math.log(k) + (_EULER_GAMMA + (inverse / 2 - inverse * inverse / 12))


@jit
def _hommel_sorted(
    ranked: np.ndarray, observed: np.ndarray, total: np.ndarray
) -> np.ndarray:
    """Hommel adjusted p-values, before the cap at 1, of rows of sorted values.

    Row ``f`` holds ``m = observed[f]`` sorted p-values followed by NaN, in a
    family of ``n = total[f]`` tests whose unobserved p-values equal 1. The
    result keeps the sorted order and the NaN. A row takes O(m) time.

    For sorted ``p_(1) <= ... <= p_(n)``, among the sets of ``k`` tests that
    contain a test with p-value ``x``, the largest Simes p-value comes from
    the set whose other ``k - 1`` tests have the largest p-values. It equals
    ``min(k x, C_k)``, where ``C_k = min over j of k p_(n-k+j) / j`` is the
    Simes p-value of the ``k`` largest p-values, and the adjusted value of
    ``x`` is its maximum over ``k``. Adding a p-value below the others
    multiplies each term of ``C_k`` by ``(k + 1) j / (k (j + 1)) <= 1``, so
    ``C_k`` does not increase with ``k`` while ``k x`` grows: ``C_k >= k x``
    holds for ``k = 1, ..., K`` and fails above ``K``, and the adjusted value
    is ``max(K x, C_(K+1))``. ``K`` falls as ``x`` rises, so one sweep over
    the sorted p-values finds it.

    ``C_k / k`` is the smallest slope from the point ``(n - k, 0)`` to the
    points ``(s, p_(s))`` with ``s > n - k``, reached at a vertex of their
    lower convex hull. As ``k`` grows the hull gains points on the left and
    the touching vertex moves left, so another sweep finds every ``C_k``.

    The unobserved p-values sit at ranks ``m + 1`` to ``n``. For
    ``k <= n - m`` the ``k`` largest are all unobserved and ``C_k = 1``; for
    larger ``k``, ``C_k = min(D_k, 1)``, where ``D_k`` is the Simes p-value
    from the observed values alone (their points ``(s, 1)`` have smallest
    slope ``1 / k``). The adjusted value is therefore the cap at 1 of
    ``max((n - m) x, max over k > n - m of min(k x, D_k))``. The sweeps
    compute this over the ``m`` observed values, with ``K >= n - m``, and the
    cap at 1 on the result completes it.
    """
    families, width = ranked.shape
    adjusted = np.full((families, width), np.nan)
    hull_x = np.empty(width)
    hull_y = np.empty(width)
    simes = np.empty(width)
    largest_before = np.empty(width + 1)
    for f in range(families):
        m = observed[f]
        n = float(total[f])
        p = ranked[f]

        # Lower hull of the points (s, p_(s)) right of the offset a = n - k,
        # stored right to left with the leftmost vertex at index `top`.
        top = -1
        touch = 0
        for a in range(m - 1, -1, -1):
            x = a + 1.0
            y = p[a]
            while top >= 1:
                # Drop the leftmost vertex while it lies on or above the
                # segment from the new point to the next vertex.
                rise = (hull_y[top] - y) * (hull_x[top - 1] - x)
                if rise < (hull_y[top - 1] - y) * (hull_x[top] - x):
                    break
                top -= 1
            top += 1
            hull_x[top] = x
            hull_y[top] = y
            # Pops never reach the touching vertex in exact arithmetic; this
            # keeps the index on the hull should rounding say otherwise.
            touch = min(touch, top)
            while touch < top:
                # Move left while the slope from (a, 0) keeps decreasing.
                left = hull_y[touch + 1] * (hull_x[touch] - a)
                if left >= hull_y[touch] * (hull_x[touch + 1] - a):
                    break
                touch += 1
            # D_k for k = n - a: C_k from the observed p-values alone.
            simes[a] = (n - a) * hull_y[touch] / (hull_x[touch] - a)

        # The largest D_k over k > K is D_(K+1) in exact arithmetic; a running
        # maximum also reproduces a direct maximum over k to the last bit.
        largest_before[0] = 0.0
        for a in range(m):
            largest_before[a + 1] = max(largest_before[a], simes[a])
        start = 0
        for r in range(m):
            x = p[r]
            while start < m and simes[start] < (n - start) * x:
                start += 1
            adjusted[f, r] = max((n - start) * x, largest_before[start])
    return adjusted


@dataclass(frozen=True, eq=False, repr=False)
class TrueNullProportion:
    """Storey's estimate of the proportion of true null hypotheses, from
    :func:`proportion_of_true_nulls`, with qvalue's ``pi0est`` names.

    Attributes:
        pi0: The estimate, in [0, 1]; 1 - pi0 is pi1, the proportion of the tests with
            an effect.
        lambdas: The values of lambda, in increasing order (``lambda``).
        pi0_lambda: The estimate at each lambda, #{p >= lambda} / (m (1 - lambda)),
            which can exceed 1 (``pi0.lambda``).
        pi0_smooth: The smoother's values at each lambda, with ``"smoother"`` and
            several lambdas; None otherwise (``pi0.smooth``).
        method: ``"smoother"`` or ``"bootstrap"``.
        n_p_values: m, the number of p-values (NaN left out).
    """

    pi0: float
    lambdas: NDArray[np.float64]
    pi0_lambda: NDArray[np.float64]
    pi0_smooth: NDArray[np.float64] | None
    method: str
    n_p_values: int

    @property
    def pi1(self) -> float:
        """1 - pi0, the estimated proportion of the tests with an effect."""
        return 1.0 - self.pi0

    def __repr__(self) -> str:
        return (
            f"TrueNullProportion(pi0={self.pi0:.6g}, method={self.method!r}, "
            f"{self.lambdas.size} lambda{'' if self.lambdas.size == 1 else 's'}, "
            f"{self.n_p_values} p-values)"
        )


def proportion_of_true_nulls(
    p: ArrayLike,
    *,
    lambdas: ArrayLike | None = None,
    method: str = "smoother",
    smooth_df: float = 3.0,
    smooth_log: bool = False,
) -> TrueNullProportion:
    """Estimate the proportion of true null hypotheses among tests, as qvalue's pi0est.

    Storey's (2002) estimate pi0(lambda) = #{p >= lambda} / (m (1 - lambda)), at one
    lambda or smoothed over several (Storey and Tibshirani 2003) or chosen by a
    bootstrap estimate of its mean squared error (Storey, Taylor and Siegmund 2004),
    with the defaults and options of qvalue's ``pi0est``; the module description
    gives the method and the deliberate differences from qvalue. 1 - pi0 (``pi1``) of
    one study's p-values among the genes another study finds is the proportion of
    those genes with an effect in it (Storey and Tibshirani 2003).

    Args:
        p: p-values in [0, 1]; NaN marks a missing one, which is left out, as qvalue
            leaves out NA.
        lambdas: One value of lambda in [0, 1), or at least four distinct values, in
            any order; by default 0.05, 0.10, ..., 0.95 (qvalue's ``lambda``).
        method: ``"smoother"`` (qvalue's default) or ``"bootstrap"``
            (``pi0.method``), for several lambdas.
        smooth_df: The equivalent degrees of freedom of the smoother, above 1 and at
            most the number of lambdas (``smooth.df``).
        smooth_log: Whether the smoother fits log pi0(lambda) (``smooth.log.pi0``).

    Returns:
        The estimate, with pi0(lambda) at every lambda and the smoother's values.

    Raises:
        TypeError: If ``p`` or ``lambdas`` does not hold real numbers or is a numpy
            masked array, or an option has the wrong type.
        ValueError: If a p-value lies outside [0, 1], there is none, ``lambdas`` are
            outside [0, 1), repeated, or two or three, ``method`` is unknown,
            ``smooth_df`` is out of range, or ``smooth_log`` meets a lambda that no
            p-value reaches.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> p = np.concatenate([rng.uniform(size=800), rng.beta(0.2, 3.0, size=200)])
        >>> estimate = proportion_of_true_nulls(p, lambdas=0.5)
        >>> bool(estimate.pi0 == np.mean(p >= 0.5) / 0.5)
        True
    """
    chosen = checked_choice(method, "method", _PI0_METHODS)
    log_scale = checked_flag(smooth_log, "smooth_log")
    values = _p_value_array(p)
    values = np.sort(values[~np.isnan(values)].reshape(-1))
    m = values.size
    if m == 0:
        raise ValueError("there are no p-values: p is empty or every value is NaN")
    grid = _lambdas(lambdas)
    df = real_number(smooth_df, "smooth_df", finite=True)
    if grid.size > 1 and chosen == "smoother" and not 1.0 < df <= grid.size:
        raise ValueError(
            f"smooth_df must be above 1 and at most the number of lambdas "
            f"({grid.size}), got {df}"
        )
    # W(lambda), the p-values at least lambda.
    counts = m - np.searchsorted(values, grid, side="left").astype(np.float64)
    pi0_lambda = counts / (m * (1.0 - grid))
    smooth = None
    if grid.size == 1:
        estimate = float(pi0_lambda[0])
    elif chosen == "smoother":
        if log_scale:
            if not (counts > 0.0).all():
                lowest = float(grid[np.flatnonzero(counts == 0.0)[0]])
                raise ValueError(
                    f"no p-value is at least lambda = {lowest:g}, so log pi0(lambda) "
                    "has no value there: smooth_log=False smooths it, or choose "
                    "smaller lambdas"
                )
            smooth = np.exp(_smoothing_spline_fit(grid, np.log(pi0_lambda), df))
        else:
            smooth = _smoothing_spline_fit(grid, pi0_lambda, df)
        estimate = float(smooth[-1])
    else:
        quantile = _type7_quantile(pi0_lambda, _BOOTSTRAP_QUANTILE)
        error = (
            counts / (m**2 * (1.0 - grid) ** 2) * (1.0 - counts / m)
            + (pi0_lambda - quantile) ** 2
        )
        estimate = float(pi0_lambda[int(np.argmin(error))])
    return TrueNullProportion(
        pi0=min(max(estimate, 0.0), 1.0),
        lambdas=read_only(grid),
        pi0_lambda=read_only(pi0_lambda),
        pi0_smooth=None if smooth is None else read_only(smooth),
        method=chosen,
        n_p_values=m,
    )


def _lambdas(value: Any) -> NDArray[np.float64]:
    if value is None:
        return _DEFAULT_LAMBDAS.copy()
    array = unmasked_array(value, "lambdas")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"lambdas must hold real numbers, got dtype {array.dtype}")
    grid = np.sort(np.array(array, dtype=np.float64).reshape(-1))
    if grid.size == 0:
        raise ValueError("lambdas is empty; give one value or at least four")
    if not ((grid >= 0.0) & (grid < 1.0)).all():
        raise ValueError(f"lambdas must lie in [0, 1), got {grid.tolist()}")
    if 1 < grid.size < _FEWEST_LAMBDAS:
        raise ValueError(
            f"lambdas must be one value or at least {_FEWEST_LAMBDAS}, got "
            f"{grid.size}, as for qvalue"
        )
    if (np.diff(grid) == 0.0).any():
        raise ValueError(f"lambdas must be distinct, got {grid.tolist()}")
    return grid


def _type7_quantile(values: NDArray[np.float64], probability: float) -> float:
    """R's quantile of type 7, (1 - g) x_(j) + g x_(j + 1), as its help page gives
    it, with j and g the whole and fractional parts of 1 + (n - 1) probability."""
    ordered = np.sort(values)
    position = 1.0 + (ordered.size - 1) * probability
    j = int(math.floor(position))
    g = position - j
    if g == 0.0 or j >= ordered.size:
        return float(ordered[j - 1])
    return float((1.0 - g) * ordered[j - 1] + g * ordered[j])
