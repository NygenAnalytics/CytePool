import math
from fractions import Fraction
from pathlib import Path

import numba
import numpy as np
import pytest

from cytepool.stats import distributions as dist
from cytepool.stats import special

DATA = Path(__file__).parent / "data" / "distributions"
EPS = np.finfo(float).eps
# Accuracy contract of every distribution function against mpmath (50 digits).
CONTRACT = 1e-12
INF = math.inf


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz") as data:
        return dict(data)


def assert_probability_close(
    got: np.ndarray, ref: np.ndarray, rtol: float = CONTRACT
) -> None:
    """Relative error at most rtol down to 1e-300; below that, agreement to 1e-312."""
    got, ref = np.asarray(got), np.asarray(ref)
    checked = ref >= 1e-300
    np.testing.assert_allclose(got[checked], ref[checked], rtol=rtol, atol=0)
    np.testing.assert_allclose(got[~checked], ref[~checked], rtol=0, atol=1e-312)


def assert_log_close(got: np.ndarray, ref: np.ndarray) -> None:
    """Relative error at most 1e-12 where abs(ref) >= 1e-300; below, agreement to 1e-312."""
    got, ref = np.asarray(got), np.asarray(ref)
    checked = np.abs(ref) >= 1e-300
    np.testing.assert_allclose(got[checked], ref[checked], rtol=CONTRACT, atol=0)
    np.testing.assert_allclose(got[~checked], ref[~checked], rtol=0, atol=1e-312)


def assert_close_to_floor(got: np.ndarray, ref: np.ndarray, ulps: float = 8.0) -> None:
    """Relative error at most a few units of eps * max(1, abs(log p)).

    A probability evaluated through its logarithm carries a relative error of order
    eps abs(log p), because the logarithm itself is rounded; closed forms are matched to
    that floor (2e-13 at p = 1e-300) rather than to a fixed number of digits.
    """
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    scale = EPS * np.maximum(1.0, np.abs(np.log(ref))) * ref
    ratio = np.abs(got - ref) / scale
    worst = int(np.argmax(ratio))
    assert ratio[worst] <= ulps, (
        f"{ratio[worst]:.1f} x eps|log p| at {ref[worst]!r}: {got[worst]!r}"
    )


# ---------------------------------------------------------------------------------------
# Agreement with mpmath (50 digits) on the validation fixtures
# ---------------------------------------------------------------------------------------


def test_normal_matches_high_precision_values() -> None:
    data = _load("norm")
    x = data["x"]
    assert_probability_close(dist.norm_cdf(x), data["cdf"])
    assert_probability_close(dist.norm_sf(x), data["sf"])
    assert_log_close(dist.norm_logcdf(x), data["logcdf"])
    assert_log_close(dist.norm_logsf(x), data["logsf"])


def test_normal_quantiles_match_high_precision_values() -> None:
    data = _load("norm_quantile")
    np.testing.assert_allclose(
        dist.norm_ppf(data["p"]), data["ppf"], rtol=CONTRACT, atol=0
    )
    np.testing.assert_allclose(
        dist.norm_isf(data["p"]), -data["ppf"], rtol=CONTRACT, atol=0
    )


@pytest.mark.parametrize("family", ["t", "t_genes"])
def test_t_matches_high_precision_values(family: str) -> None:
    data = _load(family)
    x, df = data["x"], data["df"]
    if family == "t":
        assert np.isinf(df).any() and (df != np.floor(df)).any()
    assert_probability_close(dist.t_cdf(x, df), data["cdf"])
    assert_probability_close(dist.t_sf(x, df), data["sf"])
    assert_log_close(dist.t_logsf(x, df), data["logsf"])


def test_t_quantiles_match_high_precision_values() -> None:
    data = _load("t_quantile")
    p, df = data["p"], data["df"]
    assert np.isinf(df).any() and p.min() <= 1e-300
    # quantiles near the largest double with df near 1/2, and the central case df = 0.001
    assert np.any((df <= 0.6) & (data["isf"] > 1e200)) and np.any(df == 0.001)
    np.testing.assert_allclose(dist.t_isf(p, df), data["isf"], rtol=CONTRACT, atol=0)
    np.testing.assert_allclose(dist.t_ppf(p, df), -data["isf"], rtol=CONTRACT, atol=0)


@pytest.mark.parametrize("family", ["f", "f_genes"])
def test_f_matches_high_precision_values(family: str) -> None:
    data = _load(family)
    x, dfn, dfd = data["x"], data["dfn"], data["dfd"]
    if family == "f":
        assert np.isinf(dfd).any()
    assert_probability_close(dist.f_cdf(x, dfn, dfd), data["cdf"])
    assert_probability_close(dist.f_sf(x, dfn, dfd), data["sf"])
    assert_log_close(dist.f_logsf(x, dfn, dfd), data["logsf"])


def test_chi2_matches_high_precision_values() -> None:
    data = _load("chi2")
    x, df = data["x"], data["df"]
    assert_probability_close(dist.chi2_cdf(x, df), data["cdf"])
    assert_probability_close(dist.chi2_sf(x, df), data["sf"])
    assert_log_close(dist.chi2_logsf(x, df), data["logsf"])


def test_gamma_matches_high_precision_values() -> None:
    data = _load("gamma")
    args = data["x"], data["shape"], data["scale"]
    assert_probability_close(dist.gamma_cdf(*args), data["cdf"])
    assert_probability_close(dist.gamma_sf(*args), data["sf"])


def test_beta_matches_high_precision_values() -> None:
    data = _load("beta")
    args = data["x"], data["a"], data["b"]
    assert_probability_close(dist.beta_cdf(*args), data["cdf"])
    assert_probability_close(dist.beta_sf(*args), data["sf"])


def test_hypergeometric_matches_exact_values() -> None:
    data = _load("hypergeom")
    args = data["k"], data["M"], data["n"], data["N"]
    assert_probability_close(dist.hypergeom_sf(*args), data["sf"])
    assert_log_close(dist.hypergeom_logsf(*args), data["logsf"])


def test_hypergeometric_tails_of_large_populations() -> None:
    # In populations of 1e9 to 1e12, k M and n N exceed 2^53, and near the mode the tail
    # sums run over millions of terms whose ratios are close to 1. Rounding the
    # deviation from the mean, stopping when a term falls below eps times the total, or
    # dropping terms below the last digit of the total each cost 1e-12 to 1e-9 there.
    data = _load("hypergeom")
    large = data["M"] >= 1e9
    assert np.count_nonzero(large) >= 30
    args = data["k"][large], data["M"][large], data["n"][large], data["N"][large]
    got, ref = dist.hypergeom_sf(*args), data["sf"][large]
    checked = ref >= 1e-300
    error = np.abs(got[checked] / ref[checked] - 1.0)
    floor = 8.0 * EPS * np.abs(np.log(ref[checked]))
    assert np.all(error <= np.maximum(1e-13, floor))
    # Below the mode, P(X > k) is close to 1 and carries no information; there
    # log P(X > k) is close to -P(X <= k), so its relative error is that of the lower
    # tail sum, of order eps |log P(X <= k)|.
    got, ref = dist.hypergeom_logsf(*args), data["logsf"][large]
    checked = np.abs(ref) >= 1e-300
    error = np.abs(got[checked] / ref[checked] - 1.0)
    floor = 8.0 * EPS * np.abs(np.log(np.abs(ref[checked])))
    assert np.all(error <= np.maximum(1e-13, floor))


# ---------------------------------------------------------------------------------------
# Closed forms and limits
# ---------------------------------------------------------------------------------------


def test_t_closed_forms_for_one_and_two_degrees_of_freedom() -> None:
    x = np.array([1e-300, 1e-8, 0.3, 1.0, 2.5, 40.0, 1e6, 1e150, 1e300])
    # Cauchy: P(T > x) = atan(1/x) / pi for x > 0
    assert_close_to_floor(dist.t_sf(x, 1.0), np.arctan(1.0 / x) / math.pi)
    # df = 2: P(T > x) = 1 / (s (s + x)) with s = sqrt(x^2 + 2), for x > 0
    x2 = x[x < 1e150]
    s = np.sqrt(x2**2 + 2.0)
    assert_close_to_floor(dist.t_sf(x2, 2.0), 1.0 / (s * (s + x2)))
    # Quantiles: the closed-form survival functions at the computed quantiles return p.
    p = np.array([1e-300, 1e-100, 1e-8, 0.01, 0.2, 0.45])
    t = dist.t_isf(p, 1.0)
    assert_close_to_floor(np.arctan(1.0 / t) / math.pi, p)
    t = dist.t_isf(p, 2.0)
    s = np.sqrt(t**2 + 2.0)
    assert_close_to_floor(1.0 / (s * (s + t)), p)


def test_chi2_f_gamma_beta_closed_forms() -> None:
    x = np.array([1e-300, 1e-6, 0.4, 3.0, 50.0, 600.0, 1400.0])
    # chi-squared on 2 df and gamma with shape 1 are exponential
    assert_close_to_floor(dist.chi2_sf(x, 2.0), np.exp(-x / 2))
    assert_close_to_floor(dist.gamma_sf(x, 1.0, 2.5), np.exp(-x / 2.5))
    small = x[x <= 100]
    erfc = np.array([math.erfc(math.sqrt(v / 2)) for v in small])
    assert_close_to_floor(dist.chi2_sf(small, 1.0), erfc)
    # F(2, d): P(F > x) = (1 + 2 x / d)^(-d / 2)
    for d in (1.0, 3.5, 40.0, 1e6):
        expected = np.exp(-d / 2 * np.log1p(2 * x / d))
        kept = expected > 0
        assert_close_to_floor(dist.f_sf(x[kept], 2.0, d), expected[kept])
    u = np.array([1e-300, 1e-9, 0.1, 0.5, 0.9, 1.0 - 1e-9])
    assert_close_to_floor(dist.beta_cdf(u, 1.0, 1.0), u)
    assert_close_to_floor(dist.beta_cdf(u[1:], 3.5, 1.0), u[1:] ** 3.5)
    assert_close_to_floor(dist.beta_sf(u, 1.0, 7.0), np.exp(7.0 * np.log1p(-u)))


def test_hypergeometric_small_tables_by_exact_counting() -> None:
    for M, n, N in [(1, 1, 1), (10, 3, 4), (20, 7, 12), (60, 30, 25), (60, 2, 59)]:
        total = math.comb(M, N)
        for k in range(-1, min(n, N) + 1):
            upper = sum(
                math.comb(n, j) * math.comb(M - n, N - j)
                for j in range(k + 1, min(n, N) + 1)
            )
            expected = float(Fraction(upper, total))
            assert dist.hypergeom_sf(k, M, n, N) == pytest.approx(
                expected, rel=1e-14, abs=0
            )
            if upper:
                log_expected = math.log(Fraction(upper, total))
                assert dist.hypergeom_logsf(k + 0.5, M, n, N) == pytest.approx(
                    log_expected, rel=1e-14, abs=1e-16
                )


def test_infinite_degrees_of_freedom_are_the_limits() -> None:
    # df = inf dispatches to the normal (and dfd = inf to the chi-squared) by design; the
    # independent checks are the comparisons at df = 1e300, which go through the
    # incomplete beta function.
    x = np.linspace(-37.0, 37.0, 75)
    assert np.array_equal(dist.t_sf(x, INF), dist.norm_sf(x))
    assert np.array_equal(
        dist.t_isf(np.array([1e-300, 0.01, 0.7]), INF),
        dist.norm_isf([1e-300, 0.01, 0.7]),
    )
    # Very large finite degrees of freedom go through the incomplete beta function and
    # must approach the limits smoothly; the t tail differs from the normal tail by a
    # relative (x^4 + 2 x^2) / (4 df), negligible at df = 1e300. The tolerance follows the
    # eps |log p| floor, so an error of 1e-13 at moderate probabilities fails.
    assert_close_to_floor(dist.t_sf(x, 1e300), dist.norm_sf(x), ulps=16.0)
    np.testing.assert_allclose(
        dist.t_logsf(x, 1e300), dist.norm_logsf(x), rtol=CONTRACT, atol=0
    )
    f = np.array([1e-3, 0.5, 1.0, 2.0, 10.0, 100.0])
    # dfn F and dfd / F are formed with one rounding here and another in the library.
    np.testing.assert_allclose(
        dist.f_sf(f, 3.0, INF), dist.chi2_sf(3.0 * f, 3.0), rtol=1e-13
    )
    np.testing.assert_allclose(
        dist.f_sf(f, 3.0, 1e300), dist.f_sf(f, 3.0, INF), rtol=CONTRACT
    )
    np.testing.assert_allclose(
        dist.f_sf(f, INF, 4.0), dist.chi2_cdf(4.0 / f, 4.0), rtol=1e-13
    )
    assert dist.f_sf(0.999, INF, INF) == 1.0 and dist.f_sf(1.0, INF, INF) == 0.0


# ---------------------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------------------


DFS = np.array([0.5, 1.0, 1.5, 3.0, 7.3, 30.0, 1e4, 1e8, INF])


def test_t_quantiles_for_few_degrees_of_freedom() -> None:
    # With df near 1/2 the quantiles of small p approach and pass the largest double; every
    # finite quantile must reproduce p, and an infinite one must lie beyond the largest
    # double (P(T > largest) > p). Also df far below the checked range, where the central
    # quantiles are astronomically large.
    p = 10.0 ** -np.arange(1.0, 301.0, 7.0)
    for df in (0.5, 0.6, 0.75, 1.5):
        t = dist.t_isf(p, df)
        finite = np.isfinite(t)
        np.testing.assert_allclose(
            dist.t_sf(t[finite], df), p[finite], rtol=CONTRACT, atol=0
        )
        assert np.all(dist.t_sf(1.7976931348623157e308, df) > p[~finite])
        assert np.isinf(t).sum() == (~finite).sum()  # never NaN
    for p_central in (0.26, 0.4, 0.4999):
        t = dist.t_isf(p_central, 0.001)
        assert math.isfinite(t)
        assert dist.t_sf(t, 0.001) == pytest.approx(p_central, rel=CONTRACT)


def test_each_array_loop_compiles_once() -> None:
    # Arguments of every layout (Python and numpy scalars, integers, float32, broadcast
    # and strided views, read-only and Fortran-ordered arrays) reach the compiled loops
    # in one form, so each loop ends with exactly one signature.
    x = np.linspace(0.5, 3.0, 6)
    read_only = x.copy()
    read_only.flags.writeable = False
    layouts = [
        x,
        read_only,
        x[::2],
        x[:, None],
        np.asfortranarray(np.outer(x, [1.0, 2.0])),
        x.astype(np.float32),
        np.arange(1, 4),
        np.float64(1.5),
        2,
        2.5,
    ]
    calls = [
        (special._gamma_family_array, lambda v: special.digamma(v)),
        (special._log_beta_array, lambda v: special.log_beta(v, 2.0)),
        (special._gammainc_array, lambda v: special.gammaincc(v, x[:, None])),
        (special._betainc_array, lambda v: special.betainc(v, 2.0, 0.3)),
        (dist._norm_array, lambda v: dist.norm_sf(v)),
        (dist._t_array, lambda v: dist.t_sf(v, 5.0)),
        (dist._f_array, lambda v: dist.f_sf(v, 3.0, 7.0)),
        (dist._chi2_array, lambda v: dist.chi2_sf(v, 4.0)),
        (dist._gamma_array, lambda v: dist.gamma_sf(v, 2.0, scale=v)),
        (dist._beta_array, lambda v: dist.beta_cdf(0.3, v, 2.0)),
        (dist._hypergeom_array, lambda v: dist.hypergeom_sf(v, 20, 7, 12)),
    ]
    for loop, call in calls:
        for value in layouts:
            call(value)
        assert len(loop.signatures) == 1, loop


def test_hypergeometric_beyond_exact_integers_gives_nan() -> None:
    # Integers above 2^53 are not all doubles; counts there are not exact.
    assert math.isnan(dist.hypergeom_sf(2.5e16, 1e17, 5e16, 5e16))
    assert math.isnan(dist.hypergeom_logsf(1.0, 2.0**53, 10.0, 10.0))
    assert dist.hypergeom_sf(0.0, 2.0**53 - 1, 1.0, 1.0) == pytest.approx(
        2.0**-53, rel=1e-12
    )


def test_t_lower_tail_mirrors_upper_tail_exactly() -> None:
    # t_cdf is computed as t_sf(-x) (and norm_sf as norm_cdf(-x)), so this guards that
    # wiring; the task requires the mirror to hold to the last bit.
    x = np.concatenate(
        [
            -(10.0 ** np.linspace(-300, 300, 61)),
            np.linspace(-40, 40, 161),
            10.0 ** np.linspace(-300, 300, 61),
        ]
    )
    x, df = np.meshgrid(x, DFS)
    assert np.array_equal(dist.t_cdf(-x, df), dist.t_sf(x, df))
    assert np.array_equal(dist.norm_cdf(-x[0]), dist.norm_sf(x[0]))
    assert np.array_equal(dist.norm_logcdf(-x[0]), dist.norm_logsf(x[0]))


def test_cdf_and_sf_sum_to_one() -> None:
    # Both tails come from one evaluation, one directly and the other as its complement
    # in log space; this guards that pairing on every branch.
    x = np.linspace(-30.0, 30.0, 241)[:, None]
    np.testing.assert_allclose(
        dist.t_cdf(x, DFS) + dist.t_sf(x, DFS), 1.0, rtol=0, atol=4 * EPS
    )
    np.testing.assert_allclose(
        dist.norm_cdf(x) + dist.norm_sf(x), 1.0, rtol=0, atol=4 * EPS
    )
    q = np.abs(x) * 3
    dfs = np.array([0.5, 2.0, 7.3, 300.0, 1e5, INF])
    total = dist.f_cdf(q, 3.0, dfs) + dist.f_sf(q, 3.0, dfs)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)
    total = dist.chi2_cdf(q, dfs[:-1]) + dist.chi2_sf(q, dfs[:-1])
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)
    total = dist.gamma_cdf(q, dfs[:-1], 0.7) + dist.gamma_sf(q, dfs[:-1], 0.7)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)
    u = np.linspace(0.0, 1.0, 101)[:, None]
    total = dist.beta_cdf(u, dfs[:-1], 2.5) + dist.beta_sf(u, dfs[:-1], 2.5)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)


def test_quantile_round_trips_at_extreme_probabilities() -> None:
    # The round trip checks that the root finder converges for every (p, df); the
    # accuracy of the survival functions themselves is checked against mpmath above.
    p = np.concatenate([10.0 ** -np.linspace(1, 300, 49), [0.2, 0.45]])
    np.testing.assert_allclose(
        dist.norm_cdf(dist.norm_ppf(p)), p, rtol=CONTRACT, atol=0
    )
    np.testing.assert_allclose(dist.norm_sf(dist.norm_isf(p)), p, rtol=CONTRACT, atol=0)
    # Subnormal probabilities carry few bits; compare their logarithms.
    tiny = np.array([1e-305, 1e-310, 1e-320, 5e-324])
    np.testing.assert_allclose(
        dist.norm_logcdf(dist.norm_ppf(tiny)), np.log(tiny), rtol=CONTRACT, atol=0
    )
    for df in DFS:
        t = dist.t_isf(p, df)
        finite = np.isfinite(t)
        # Only df = 1/2 overflows, where the quantile exceeds the largest double.
        assert finite.all() or (
            df == 0.5 and np.all(dist.t_sf(1.7976931348623157e308, df) > p[~finite])
        )
        np.testing.assert_allclose(
            dist.t_sf(t[finite], df), p[finite], rtol=CONTRACT, atol=0
        )
        np.testing.assert_allclose(
            dist.t_cdf(dist.t_ppf(p, df)[finite], df), p[finite], rtol=CONTRACT, atol=0
        )


def test_monotone_on_fine_grids() -> None:
    x = np.linspace(-60.0, 60.0, 2401)
    for df in DFS:
        assert np.all(np.diff(dist.t_sf(x, df)) <= 0)
        assert np.all(np.diff(dist.t_logsf(x, df)) <= 0)
    assert np.all(np.diff(dist.norm_cdf(x)) >= 0) and np.all(
        np.diff(dist.norm_logcdf(x)) >= 0
    )
    p = np.linspace(0.0, 1.0, 2001)
    assert np.all(np.diff(dist.norm_ppf(p)) > 0)
    for df in (0.5, 1.5, 3.0, 7.3, 1e4):
        assert np.all(np.diff(dist.t_isf(p, df)) < 0)
    q = np.linspace(0.0, 80.0, 2001)
    for dfn, dfd in [(1.0, 1.0), (3.0, 7.3), (30.0, 1e5), (2.0, INF)]:
        assert np.all(np.diff(dist.f_sf(q, dfn, dfd)) <= 0)
    for df in (0.5, 3.0, 40.0):
        assert np.all(np.diff(dist.chi2_logsf(q, df)) <= 0)
        assert np.all(np.diff(dist.gamma_cdf(q, df, 1.3)) >= 0)
    u = np.linspace(-0.1, 1.1, 2001)
    assert np.all(np.diff(dist.beta_cdf(u, 0.3, 4.0)) >= 0)
    k = np.arange(-2.0, 60.0)
    assert np.all(np.diff(dist.hypergeom_sf(k, 500.0, 120.0, 57.0)) <= 0)


# ---------------------------------------------------------------------------------------
# Special values, domains and argument handling
# ---------------------------------------------------------------------------------------


def test_values_at_the_ends_of_the_support() -> None:
    assert (
        dist.norm_ppf(0.0) == -INF
        and dist.norm_ppf(1.0) == INF
        and dist.norm_ppf(0.5) == 0.0
    )
    assert (
        dist.norm_isf(0.0) == INF
        and dist.norm_logsf(INF) == -INF
        and dist.norm_logcdf(INF) == 0.0
    )
    assert (
        dist.t_ppf(0.0, 3.0) == -INF
        and dist.t_isf(0.0, 3.0) == INF
        and dist.t_isf(0.5, 3.0) == 0.0
    )
    assert (
        dist.t_sf(INF, 3.0) == 0.0
        and dist.t_sf(-INF, 3.0) == 1.0
        and dist.t_sf(0.0, 3.0) == 0.5
    )
    assert (
        dist.f_sf(0.0, 2.0, 3.0) == 1.0
        and dist.f_sf(-1.0, 2.0, 3.0) == 1.0
        and dist.f_sf(INF, 2.0, 3.0) == 0.0
    )
    assert dist.f_logsf(0.0, 2.0, 3.0) == 0.0 and dist.f_logsf(INF, 2.0, 3.0) == -INF
    assert (
        dist.chi2_cdf(0.0, 1.0) == 0.0
        and dist.chi2_sf(-1.0, 1.0) == 1.0
        and dist.chi2_logsf(INF, 1.0) == -INF
    )
    assert dist.gamma_cdf(-1.0, 2.0) == 0.0 and dist.gamma_sf(INF, 2.0, 3.0) == 0.0
    assert dist.beta_cdf(-1.0, 2.0, 3.0) == 0.0 and dist.beta_cdf(2.0, 2.0, 3.0) == 1.0
    assert (
        dist.hypergeom_sf(-1.0, 20.0, 7.0, 12.0) == 1.0
        and dist.hypergeom_sf(7.0, 20.0, 7.0, 12.0) == 0.0
    )
    assert (
        dist.hypergeom_logsf(7.0, 20.0, 7.0, 12.0) == -INF
        and dist.hypergeom_logsf(-INF, 20.0, 7.0, 12.0) == 0.0
    )
    assert (
        dist.hypergeom_sf(0.0, 0.0, 0.0, 0.0) == 0.0
        and dist.hypergeom_sf(-1.0, 0.0, 0.0, 0.0) == 1.0
    )


def test_logarithms_stay_finite_where_probabilities_underflow() -> None:
    assert dist.norm_sf(40.0) == 0.0 and dist.norm_logsf(40.0) == pytest.approx(
        -804.6084420137538, rel=1e-14
    )
    assert dist.t_sf(1e300, 1.5) == 0.0 and math.isfinite(dist.t_logsf(1e300, 1.5))
    assert dist.f_sf(1e300, 3.0, 1e6) == 0.0 and math.isfinite(
        dist.f_logsf(1e300, 3.0, 1e6)
    )
    assert dist.chi2_sf(1e5, 3.0) == 0.0 and math.isfinite(dist.chi2_logsf(1e5, 3.0))
    # P(X > 2999) = P(X = 3000) = C(27000, 12000) / C(30000, 15000) = exp(-2246.39...)
    assert dist.hypergeom_sf(2999.0, 30000.0, 3000.0, 15000.0) == 0.0
    log_sf = dist.hypergeom_logsf(2999.0, 30000.0, 3000.0, 15000.0)
    assert log_sf == pytest.approx(-2246.3939582330477, rel=1e-12)


@pytest.mark.parametrize(
    "function, arguments",
    [
        (dist.norm_ppf, (-0.1,)),
        (dist.norm_isf, (1.5,)),
        (dist.t_sf, (1.0, 0.0)),
        (dist.t_cdf, (1.0, -2.0)),
        (dist.t_logsf, (1.0, -INF)),
        (dist.t_ppf, (1.2, 3.0)),
        (dist.t_isf, (0.1, 0.0)),
        (dist.f_sf, (1.0, 0.0, 3.0)),
        (dist.f_cdf, (1.0, 2.0, -1.0)),
        (dist.chi2_sf, (1.0, 0.0)),
        (dist.chi2_logsf, (1.0, INF)),
        (dist.gamma_sf, (1.0, 0.0, 1.0)),
        (dist.gamma_cdf, (1.0, 2.0, 0.0)),
        (dist.gamma_cdf, (1.0, 2.0, INF)),
        (dist.beta_cdf, (0.5, 0.0, 1.0)),
        (dist.beta_sf, (0.5, 1.0, INF)),
        (dist.hypergeom_sf, (1.0, 20.5, 7.0, 5.0)),
        (dist.hypergeom_sf, (1.0, 20.0, 21.0, 5.0)),
        (dist.hypergeom_logsf, (1.0, 20.0, 7.0, 21.0)),
        (dist.hypergeom_logsf, (1.0, 20.0, -1.0, 5.0)),
        (dist.hypergeom_sf, (1.0, INF, 7.0, 5.0)),
    ],
)
def test_parameters_outside_the_domain_give_nan(function, arguments) -> None:
    assert math.isnan(function(*arguments))


@pytest.mark.parametrize(
    "function, arguments",
    [
        (dist.norm_cdf, (0.3,)),
        (dist.norm_sf, (0.3,)),
        (dist.norm_logcdf, (0.3,)),
        (dist.norm_logsf, (0.3,)),
        (dist.norm_ppf, (0.3,)),
        (dist.norm_isf, (0.3,)),
        (dist.t_cdf, (0.3, 4.0)),
        (dist.t_sf, (0.3, 4.0)),
        (dist.t_logsf, (0.3, 4.0)),
        (dist.t_ppf, (0.3, 4.0)),
        (dist.t_isf, (0.3, 4.0)),
        (dist.f_cdf, (0.3, 4.0, 5.0)),
        (dist.f_sf, (0.3, 4.0, 5.0)),
        (dist.f_logsf, (0.3, 4.0, 5.0)),
        (dist.chi2_cdf, (0.3, 4.0)),
        (dist.chi2_sf, (0.3, 4.0)),
        (dist.chi2_logsf, (0.3, 4.0)),
        (dist.gamma_cdf, (0.3, 4.0, 5.0)),
        (dist.gamma_sf, (0.3, 4.0, 5.0)),
        (dist.beta_cdf, (0.3, 4.0, 5.0)),
        (dist.beta_sf, (0.3, 4.0, 5.0)),
        (dist.hypergeom_sf, (3.0, 20.0, 7.0, 12.0)),
        (dist.hypergeom_logsf, (3.0, 20.0, 7.0, 12.0)),
    ],
)
def test_nan_in_any_argument_gives_nan(function, arguments) -> None:
    for position in range(len(arguments)):
        changed = list(arguments)
        changed[position] = math.nan
        assert math.isnan(function(*changed))


def test_scalars_give_floats_and_arrays_broadcast() -> None:
    assert type(dist.t_sf(2.0, 5)) is float
    assert type(dist.norm_ppf(np.float32(0.25))) is float
    assert type(dist.chi2_sf(np.array(3.0), 2)) is float
    out = dist.t_sf(np.array([[0.5], [1.0], [2.0]]), [1.0, 4.5, 30.0, INF])
    assert (
        isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (3, 4)
    )
    assert dist.t_sf(1.0, 30.0) == out[1, 2]
    assert dist.gamma_sf([1.0, 2.0], 3.0).shape == (2,)
    assert dist.hypergeom_sf(np.arange(4), 20, 7, 12).shape == (4,)
    assert dist.norm_cdf(np.empty((0, 3))).shape == (0, 3)


@pytest.mark.parametrize("bad", ["0.5", None, 0.5 + 0.0j, [0.1, "a"]])
def test_non_numeric_arguments_raise_type_error(bad) -> None:
    with pytest.raises(TypeError, match="df must contain real numbers"):
        dist.t_sf(1.0, bad)
    with pytest.raises(TypeError, match="p must contain real numbers"):
        dist.norm_ppf(bad)


def test_scalar_routines_can_be_called_from_compiled_code() -> None:
    @numba.njit
    def two_sided_p_values(t: np.ndarray, df: float) -> np.ndarray:
        out = np.empty(t.size)
        for i in range(t.size):
            out[i] = 2.0 * dist.t_sf_scalar(abs(t[i]), df)
        return out

    t = np.array([0.1, 2.0, 7.5, 40.0])
    assert np.array_equal(two_sided_p_values(t, 6.5), 2.0 * dist.t_sf(np.abs(t), 6.5))


# ---------------------------------------------------------------------------------------
# F and chi-squared quantiles
# ---------------------------------------------------------------------------------------

F_DFS = [(0.5, 3.0), (1.0, 1.0), (2.0, 7.3), (4.0, 0.5), (7.3, 30.0), (30.0, 1e5)]
F_DFS += [(1e4, 2.0), (3.0, INF), (INF, 4.0), (1e6, 1e6)]
QUANTILE_P = np.concatenate([10.0 ** -np.linspace(0.31, 300, 40), [0.2, 0.45, 0.5]])


def test_f_and_chi2_quantiles_invert_the_tails() -> None:
    # The survival functions are checked against mpmath above, and the quantiles
    # against mpmath in the validation script; the round trips check that the root
    # finder converges in both tails for every (p, df), including quantiles below
    # the smallest double (0) and above the largest (inf).
    log_p = np.log(QUANTILE_P)
    for dfn, dfd in F_DFS:
        upper = dist.f_isf(QUANTILE_P, dfn, dfd)
        lower = dist.f_ppf(QUANTILE_P, dfn, dfd)
        assert not np.isnan(upper).any() and not np.isnan(lower).any()
        big = np.isinf(upper)
        assert np.all(dist.f_sf(1.7976931348623157e308, dfn, dfd) > QUANTILE_P[big])
        np.testing.assert_allclose(
            dist.f_logsf(upper[~big], dfn, dfd), log_p[~big], rtol=CONTRACT, atol=0
        )
        normal = lower >= 2.2250738585072014e-308
        assert np.all(dist.f_cdf(5e-324, dfn, dfd) >= QUANTILE_P[lower == 0.0])
        np.testing.assert_allclose(
            np.log(dist.f_cdf(lower[normal], dfn, dfd)),
            log_p[normal],
            rtol=CONTRACT,
            atol=0,
        )
    for df in (0.5, 1.0, 4.0, 30.0, 1e5):
        upper = dist.chi2_isf(QUANTILE_P, df)
        lower = dist.chi2_ppf(QUANTILE_P, df)
        np.testing.assert_allclose(
            dist.chi2_logsf(upper, df), log_p, rtol=CONTRACT, atol=0
        )
        normal = lower >= 2.2250738585072014e-308
        np.testing.assert_allclose(
            np.log(dist.chi2_cdf(lower[normal], df)), log_p[normal], rtol=CONTRACT
        )


def test_f_and_chi2_quantile_closed_forms() -> None:
    p = QUANTILE_P
    # A quantile from a probability evaluated through its logarithm has a relative
    # error of order eps |log p| times |d log x / d log P|, at most about 1e-12 here.
    rtol = 1e-12
    # F(2, d): P(F > x) = (1 + 2x / d)^(-d / 2)
    for d in (2.0, 5.0, 40.0):
        np.testing.assert_allclose(
            dist.f_isf(p, 2.0, d), d / 2.0 * np.expm1(-2.0 / d * np.log(p)), rtol=rtol
        )
    # chi-squared(2): P(X > x) = exp(-x / 2)
    np.testing.assert_allclose(dist.chi2_isf(p, 2.0), -2.0 * np.log(p), rtol=rtol)
    np.testing.assert_allclose(dist.chi2_ppf(p, 2.0), -2.0 * np.log1p(-p), rtol=rtol)
    # F(1, d) is the square of t(d); 1 / F(a, b) ~ F(b, a); the infinite limits
    np.testing.assert_allclose(
        dist.f_isf(p, 1.0, 7.5), dist.t_isf(p / 2.0, 7.5) ** 2, rtol=rtol
    )
    np.testing.assert_allclose(
        dist.f_isf(p, 3.0, 11.0), 1.0 / dist.f_ppf(p, 11.0, 3.0), rtol=rtol
    )
    np.testing.assert_allclose(
        dist.f_isf(p, 3.5, INF), dist.chi2_isf(p, 3.5) / 3.5, rtol=rtol
    )
    np.testing.assert_allclose(
        dist.f_isf(p, INF, 4.0), 4.0 / dist.chi2_ppf(p, 4.0), rtol=rtol
    )
    # above 1/2 the other tail, 1 - p, is solved
    upper = np.linspace(0.5, 0.999, 50)
    np.testing.assert_allclose(
        dist.f_sf(dist.f_isf(upper, 3.0, 7.0), 3.0, 7.0), upper, rtol=1e-13
    )
    np.testing.assert_allclose(
        dist.f_cdf(dist.f_ppf(upper, 3.0, 7.0), 3.0, 7.0), upper, rtol=1e-13
    )


def test_f_and_chi2_quantiles_at_the_ends_and_outside_the_domain() -> None:
    assert dist.f_isf(0.0, 2.0, 3.0) == INF and dist.f_isf(1.0, 2.0, 3.0) == 0.0
    assert dist.f_ppf(0.0, 2.0, 3.0) == 0.0 and dist.f_ppf(1.0, 2.0, 3.0) == INF
    assert dist.f_isf(0.3, INF, INF) == 1.0 and dist.f_ppf(0.3, INF, INF) == 1.0
    # quantiles beyond the largest double: P(F > max) = 0.667 for F(0.02, 0.001)
    largest = 1.7976931348623157e308
    assert (
        dist.f_sf(largest, 0.02, 0.001) > 0.53 and dist.f_isf(0.53, 0.02, 0.001) == INF
    )
    assert (
        dist.f_cdf(largest, 0.35, 0.001) < 0.31 and dist.f_ppf(0.31, 0.35, 0.001) == INF
    )
    assert dist.chi2_isf(0.0, 3.0) == INF and dist.chi2_ppf(1.0, 3.0) == INF
    for function, arguments in [
        (dist.f_isf, (1.2, 3.0, 4.0)),
        (dist.f_ppf, (-0.1, 3.0, 4.0)),
        (dist.f_isf, (0.1, 0.0, 4.0)),
        (dist.f_ppf, (0.1, 3.0, -1.0)),
        (dist.chi2_isf, (0.5, 0.0)),
        (dist.chi2_ppf, (0.5, INF)),
    ]:
        assert math.isnan(function(*arguments))
    for function, arguments in [
        (dist.f_isf, (0.3, 4.0, 5.0)),
        (dist.f_ppf, (0.3, 4.0, 5.0)),
        (dist.chi2_isf, (0.3, 4.0)),
        (dist.chi2_ppf, (0.3, 4.0)),
    ]:
        for position in range(len(arguments)):
            changed = list(arguments)
            changed[position] = math.nan
            assert math.isnan(function(*changed))
    p = np.linspace(0.0, 1.0, 1001)
    for dfn, dfd in [(1.0, 1.0), (3.0, 7.3), (30.0, 1e5), (2.0, INF)]:
        assert np.all(np.diff(dist.f_isf(p, dfn, dfd)) < 0)
        assert np.all(np.diff(dist.f_ppf(p, dfn, dfd)) > 0)
    with pytest.raises(TypeError, match="p must contain real numbers"):
        dist.f_isf("0.5", 2.0, 3.0)
