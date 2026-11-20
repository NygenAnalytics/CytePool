"""Multivariate t probabilities and quantiles against independent references.

The references are closed forms, Dunnett's (1955) integral computed here in plain Python,
and the fixtures of ``validation/multiple_contrast_tests.py``: mpmath at 30 digits for
one and two coordinates and for rank-two correlations, and Dunnett's integral for
product correlations. Monte Carlo values must lie within 5.5 of their stored standard
errors, the 99.994% point of the t distribution on the 15 degrees of freedom of those
errors.
"""

import doctest
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

from cytepool._jit import thread_limit
from cytepool.stats import multivariate_t
from cytepool.stats.multivariate_t import (
    MultivariateTProbability,
    MultivariateTQuantile,
    multivariate_t_cdf,
    multivariate_t_quantile,
)

FIXTURES = Path(__file__).parent / "data" / "multiple_contrast_tests"
CALIBRATED = 5.5


def _fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name) as data:
        return {key: data[key] for key in data.files}


def equicorrelated(k: int, rho: float) -> np.ndarray:
    return np.full((k, k), rho) + (1.0 - rho) * np.eye(k)


def product_correlation(lam: np.ndarray) -> np.ndarray:
    out = np.outer(lam, lam)
    np.fill_diagonal(out, 1.0)
    return out


def normal_tail(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ---------------------------------------------------------------------------------------
# One coordinate: the t distribution
# ---------------------------------------------------------------------------------------


def cauchy_tail(x: float) -> float:
    """P(T > x) for df = 1: atan(1/x) / pi for x > 0."""
    return math.atan2(1.0, x) / math.pi


def df2_tail(x: float) -> float:
    """P(T > x) for df = 2: 1/2 - x / (2 sqrt(2 + x^2)), written without cancellation."""
    root = math.sqrt(2.0 + x * x)
    return 1.0 / (root * (root + x)) if x >= 0 else 1.0 - df2_tail(-x)


@pytest.mark.parametrize(
    ("df", "tail"), [(1.0, cauchy_tail), (2.0, df2_tail), (math.inf, normal_tail)]
)
def test_one_coordinate_is_the_t_distribution(df, tail) -> None:
    for x in (-7.0, -1.0, 0.0, 0.3, 2.0, 9.0, 1e4):
        one_sided = multivariate_t_cdf([x], [[1.0]], df)
        assert one_sided.complement == pytest.approx(tail(x), rel=1e-12)
        assert one_sided.probability == pytest.approx(tail(-x), rel=1e-12)
        assert one_sided.error == 0.0
    for x in (1e-9, 0.2, 2.0, 30.0):
        two_sided = multivariate_t_cdf([x], [[1.0]], df, two_sided=True)
        assert two_sided.complement == pytest.approx(2.0 * tail(x), rel=1e-12)
        # The probability keeps its relative accuracy when it is small.
        assert two_sided.probability == pytest.approx(1.0 - 2.0 * tail(x), rel=1e-6)
        assert two_sided.error == 0.0
    small = multivariate_t_cdf([1e-9], [[1.0]], df, two_sided=True).probability
    density = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    density = 1.0 / math.sqrt(2.0 * math.pi) if math.isinf(df) else density
    assert small == pytest.approx(2e-9 * density, rel=1e-8)


def test_one_coordinate_at_non_integer_df_matches_mpmath() -> None:
    data = _fixture("one_dimension.npz")
    result = multivariate_t_cdf(data["upper"][:, None], [[1.0]], 4.5)
    np.testing.assert_allclose(result.complement, data["above"], rtol=1e-11)
    np.testing.assert_allclose(result.probability, data["below"], rtol=1e-11)
    within = multivariate_t_cdf(
        data["within_limit"][:, None], [[1.0]], 4.5, two_sided=True
    )
    np.testing.assert_allclose(within.probability, data["within"], rtol=1e-11)


# ---------------------------------------------------------------------------------------
# Two coordinates: quadrature without random numbers
# ---------------------------------------------------------------------------------------


def _planes() -> tuple[dict[str, np.ndarray], list[MultivariateTProbability]]:
    data = _fixture("two_dimensions.npz")
    results = []
    for i in range(data["rho"].size):
        rho = data["rho"][i]
        results.append(
            multivariate_t_cdf(
                data["upper"][i],
                [[1.0, rho], [rho, 1.0]],
                data["df"][i],
                two_sided=bool(data["two_sided"][i]),
            )
        )
    return data, results


def test_two_coordinates_match_mpmath() -> None:
    data, results = _planes()
    for (probability, complement), result in zip(data["reference"], results):
        if probability <= complement:
            assert result.probability == pytest.approx(probability, rel=1e-9)
            actual = abs(result.probability - probability)
            assert result.error <= 1e-11 * probability
        else:
            assert result.complement == pytest.approx(complement, rel=1e-9)
            actual = abs(result.complement - complement)
            assert result.error <= 1e-11 * complement
        # The stored error bounds the actual one.
        assert actual <= result.error + 4e-16 * min(probability, complement)
    # Complements down to 1e-14 and small boxes are in the fixture.
    assert data["reference"].min() < 1e-13


def test_two_coordinates_use_no_random_numbers() -> None:
    correlation = [[1.0, -0.4], [-0.4, 1.0]]
    first = multivariate_t_cdf([2.0, 3.0], correlation, 6.5, rng=1)
    second = multivariate_t_cdf([2.0, 3.0], correlation, 6.5, rng=2)
    assert (first.probability, first.complement, first.error) == (
        second.probability,
        second.complement,
        second.error,
    )


def test_two_coordinates_of_correlation_one_are_one_coordinate() -> None:
    for df in (3.0, math.inf):
        both = multivariate_t_cdf([2.5, 2.5], np.ones((2, 2)), df, two_sided=True)
        one = multivariate_t_cdf([2.5], [[1.0]], df, two_sided=True)
        assert both.complement == pytest.approx(one.complement, rel=1e-10)
        opposite = multivariate_t_cdf([2.0, 3.0], [[1.0, -1.0], [-1.0, 1.0]], df)
        # T_2 = -T_1: the box is -3 <= T_1 <= 2.
        expected = (
            multivariate_t_cdf([2.0], [[1.0]], df).probability
            - multivariate_t_cdf([-3.0], [[1.0]], df).probability
        )
        assert opposite.probability == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------------------
# Three or more coordinates: Monte Carlo with stored errors
# ---------------------------------------------------------------------------------------


def within_errors(estimate, reference, error) -> None:
    """Within 5.5 standard errors, plus rounding: an estimator of zero variance, such
    as that of independent normal coordinates, has an error of 0."""
    estimate, reference, error = map(np.asarray, (estimate, reference, error))
    allowed = CALIBRATED * error + 1e-13 * np.abs(reference)
    assert np.all(np.abs(estimate - reference) <= allowed), np.max(
        np.abs(estimate - reference) / allowed
    )


@pytest.mark.parametrize("k", [3, 4, 6])
def test_independent_normal_coordinates_have_a_closed_form(k) -> None:
    limits = np.array([0.8, 2.5, 4.0, 6.0, 7.2])
    result = multivariate_t_cdf(
        np.repeat(limits[:, None], k, axis=1),
        np.eye(k),
        math.inf,
        two_sided=True,
        rng=k,
    )
    expected = [-math.expm1(k * math.log1p(-2.0 * normal_tail(c))) for c in limits]
    within_errors(result.complement, expected, result.error)
    assert min(expected) < 1e-10


def dunnett_complement(k: int, rho: float, c: float) -> float:
    """P(max |Z_i| > c) for equicorrelated normal Z, by Dunnett's integral over the
    shared factor with 400-node Gauss-Legendre, the complement integrated directly."""
    nodes, weights = np.polynomial.legendre.leggauss(400)
    half = 12.0 + math.sqrt(rho) * c
    total = 0.0
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)
    for x, w in zip(nodes * half, weights * half):
        upper = normal_tail((c - a * x) / b)
        lower = normal_tail((c + a * x) / b)
        out = min(upper + lower, 1.0)
        value = 1.0 if out >= 1.0 else -math.expm1(k * math.log1p(-out))
        total += w * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) * value
    return total


@pytest.mark.parametrize("rho", [0.5, 0.9])
@pytest.mark.parametrize("k", [3, 5])
def test_equicorrelated_normal_matches_dunnetts_integral(k, rho) -> None:
    limits = np.array([1.5, 3.0, 5.0, 7.0])
    result = multivariate_t_cdf(
        np.repeat(limits[:, None], k, axis=1),
        equicorrelated(k, rho),
        math.inf,
        two_sided=True,
        rng=11,
    )
    expected = [dunnett_complement(k, rho, c) for c in limits]
    within_errors(result.complement, expected, result.error)


def _products(
    relative_error: float, seed: int
) -> tuple[dict, list[MultivariateTProbability]]:
    data = _fixture("product_correlations.npz")
    results = []
    for i in range(data["k"].size):
        lam = data["lam"][i, : data["k"][i]]
        results.append(
            multivariate_t_cdf(
                np.full(lam.size, data["upper"][i]),
                product_correlation(lam),
                data["df"][i],
                two_sided=bool(data["two_sided"][i]),
                relative_error=relative_error,
                rng=seed + i,
            )
        )
    return data, results


def test_product_correlations_match_dunnetts_integral_into_the_tail() -> None:
    data, results = _products(0.01, 100)
    for (probability, complement), result in zip(data["reference"], results):
        within_errors(result.complement, complement, result.error)
        # The error meets the target on the estimate's smaller side, also for small
        # boxes, which the complement's absolute error would not serve.
        smaller_estimate = min(result.probability, result.complement)
        assert result.error <= 0.01 * smaller_estimate * (1 + 1e-12)
    smaller = data["reference"].min(axis=1)
    assert smaller.min() < 1e-11  # complements to 1e-12
    assert (data["reference"][:, 0] < 1e-5).any()  # small boxes


def test_stated_errors_are_honest() -> None:
    """The root-mean-square of (estimate - reference) / error is near sqrt(15/13) for
    honest errors on 15 degrees of freedom: 200 seeds of one product problem and 200
    of a small box, each within 4 simulation standard errors, which a doubled or a
    halved error leaves."""
    data = _fixture("product_correlations.npz")
    k = data["k"]
    for case in (int(np.flatnonzero(k == 6)[1]), int(np.flatnonzero(k == 4)[6])):
        lam = data["lam"][case, : k[case]]
        reference = data["reference"][case, 1]
        z = []
        for seed in range(200):
            result = multivariate_t_cdf(
                np.full(lam.size, data["upper"][case]),
                product_correlation(lam),
                data["df"][case],
                two_sided=bool(data["two_sided"][case]),
                rng=seed,
            )
            z.append((result.complement - reference) / result.error)
        low, high = _rms_bounds(200)
        assert low <= math.sqrt(np.mean(np.square(z))) <= high


def _rms_bounds(n: int) -> tuple[float, float]:
    """Where the root-mean-square of n honest z-scores on 15 degrees of freedom lies,
    within 4 simulation standard errors: E t^2 = 15/13, Var t^2 = 675/143 - (15/13)^2."""
    spread = 4.0 * math.sqrt((675.0 / 143.0 - (15.0 / 13.0) ** 2) / n)
    return math.sqrt(15.0 / 13.0 - spread), math.sqrt(15.0 / 13.0 + spread)


def test_errors_stay_honest_deep_in_the_normal_tail() -> None:
    """Eight coordinates of the normal at a complement of 1e-12: overlaps of two
    exceedances are rare events there, which the draws must reach for the stated
    error to hold."""
    data = _fixture("product_correlations.npz")
    case = int(
        np.flatnonzero(
            (data["k"] == 8)
            & np.isinf(data["df"])
            & data["two_sided"]
            & (data["reference"][:, 1] < 1e-11)
        )[0]
    )
    lam = data["lam"][case, :8]
    reference = data["reference"][case, 1]
    z = []
    for seed in range(60):
        result = multivariate_t_cdf(
            np.full(8, data["upper"][case]),
            product_correlation(lam),
            math.inf,
            two_sided=True,
            rng=seed,
        )
        z.append((result.complement - reference) / result.error)
    low, high = _rms_bounds(60)
    assert low <= math.sqrt(np.mean(np.square(z))) <= high


def test_errors_stay_honest_with_nearly_collinear_coordinates() -> None:
    """A small central box of six coordinates of rank two, two of them correlated by
    0.99998: the near duplicate cuts the box near the ends of the conditioning
    interval, a feature that four points per shift miss together."""
    data = _fixture("rank_two.npz")
    small = np.flatnonzero((data["k"] == 6) & (data["reference"][:, 0] < 0.01))
    case = int(small[np.isinf(data["df"][small])][0])
    angles = data["angles"][case, :6]
    correlation = np.cos(angles[:, None] - angles[None, :])
    off_diagonal = correlation[~np.eye(6, dtype=bool)]
    assert off_diagonal.max() > 0.999
    reference = data["reference"][case, 0]
    z = []
    for seed in range(60):
        result = multivariate_t_cdf(
            np.full(6, data["upper"][case]),
            correlation,
            math.inf,
            two_sided=True,
            rng=seed,
        )
        z.append((result.probability - reference) / result.error)
    low, high = _rms_bounds(60)
    assert low <= math.sqrt(np.mean(np.square(z))) <= high


def test_rank_two_correlations_match_the_average_over_directions() -> None:
    """Singular correlations with mixed signs, such as all pairs of three groups."""
    data = _fixture("rank_two.npz")
    for i in range(data["k"].size):
        angles = data["angles"][i, : data["k"][i]]
        result = multivariate_t_cdf(
            np.full(angles.size, data["upper"][i]),
            np.cos(angles[:, None] - angles[None, :]),
            data["df"][i],
            two_sided=bool(data["two_sided"][i]),
            rng=200 + i,
        )
        within_errors(result.complement, data["reference"][i, 1], result.error)


def test_general_correlations_match_mvtnorm() -> None:
    data = _fixture("general.npz")
    for i in range(data["k"].size):
        k = data["k"][i]
        result = multivariate_t_cdf(
            np.full(k, data["upper"][i]),
            data["correlation"][i, :k, :k],
            data["df"][i],
            two_sided=bool(data["two_sided"][i]),
            rng=300 + i,
        )
        value, stated = data["mvtnorm"][i]
        combined = math.hypot(result.error, stated / 3.0)
        assert abs(result.complement - (1.0 - value)) <= CALIBRATED * combined


def test_repeated_coordinates_count_once() -> None:
    # Identical coordinates: the complement is the univariate one, with no error.
    for df in (4.0, math.inf):
        result = multivariate_t_cdf(
            [2.0] * 4, np.ones((4, 4)), df, two_sided=True, rng=1
        )
        one = multivariate_t_cdf([2.0], [[1.0]], df, two_sided=True)
        assert result.complement == pytest.approx(one.complement, rel=1e-12)
        assert result.error <= 1e-15 * result.complement  # rounding alone
    # A coordinate repeated among others leaves the probability as it was.
    base = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    repeated = base[np.ix_([0, 1, 1, 2], [0, 1, 1, 2])]
    a = multivariate_t_cdf([2.2, 2.2, 2.2], base, 9.0, two_sided=True, rng=3)
    b = multivariate_t_cdf([2.2] * 4, repeated, 9.0, two_sided=True, rng=4)
    within_errors(a.complement, b.complement, math.hypot(a.error, b.error))


def test_singular_correlations_with_rounding_noise_are_accepted() -> None:
    """Tukey's contrasts of four groups, whose correlation has rank 3, computed from a
    covariance so that its zero eigenvalues come out about -1e-16."""
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    contrasts = np.zeros((4, 6))
    for j, (a, b) in enumerate(pairs):
        contrasts[b, j], contrasts[a, j] = 1.0, -1.0
    covariance = contrasts.T @ np.diag(1.0 / np.array([3.0, 5.0, 2.0, 4.0])) @ contrasts
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    assert np.linalg.eigvalsh(correlation).min() < 0.0
    result = multivariate_t_cdf([2.8] * 6, correlation, 12.0, two_sided=True, rng=5)
    assert 0.0 < result.complement < 6 * 2 * 0.008  # below Bonferroni's bound
    assert np.isfinite(result.error) and result.error > 0.0


def test_bounds_hold() -> None:
    """max marginal <= complement <= min(1, sum of marginals), and for two-sided boxes
    complement <= 1 - prod(1 - marginals), which holds for the t given S (Sidak's
    inequality) and so for the t."""
    from cytepool.stats.distributions import t_sf

    rng = np.random.default_rng(7)
    for trial in range(12):
        k = int(rng.integers(3, 7))
        factor = rng.normal(size=(k, k + 2))
        covariance = factor @ factor.T
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        df = float(rng.choice([2.5, 8.0, math.inf]))
        upper = rng.uniform(1.0, 4.0, k)
        two_sided = bool(trial % 2)
        result = multivariate_t_cdf(
            upper, correlation, df, two_sided=two_sided, rng=trial
        )
        marginals = (2.0 if two_sided else 1.0) * np.asarray(t_sf(upper, df))
        slack = CALIBRATED * result.error
        assert result.complement >= marginals.max() - slack
        assert result.complement <= min(1.0, marginals.sum()) + slack
        if two_sided:
            assert result.complement <= 1.0 - np.prod(1.0 - marginals) + slack


def test_one_sided_and_two_sided_complements_are_ordered() -> None:
    """P(max T > c) <= P(max |T| > c) <= 2 P(max T > c)."""
    correlation = np.array(
        [
            [1.0, 0.4, -0.3, 0.1],
            [0.4, 1.0, 0.2, 0.5],
            [-0.3, 0.2, 1.0, -0.4],
            [0.1, 0.5, -0.4, 1.0],
        ]
    )
    for c in (1.5, 3.0, 5.0):
        one = multivariate_t_cdf([c] * 4, correlation, 6.0, rng=1)
        two = multivariate_t_cdf([c] * 4, correlation, 6.0, two_sided=True, rng=2)
        slack = CALIBRATED * math.hypot(one.error, two.error)
        assert one.complement <= two.complement + slack
        assert two.complement <= 2.0 * one.complement + 2.0 * slack


def test_two_sided_boxes_do_not_depend_on_signs_or_order() -> None:
    correlation = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.3], [0.2, -0.3, 1.0]])
    upper = np.array([2.0, 2.5, 3.0])
    base = multivariate_t_cdf(upper, correlation, 7.0, two_sided=True, rng=1)
    signs = np.array([1.0, -1.0, 1.0])
    flipped = multivariate_t_cdf(
        upper, correlation * np.outer(signs, signs), 7.0, two_sided=True, rng=2
    )
    order = [2, 0, 1]
    permuted = multivariate_t_cdf(
        upper[order], correlation[np.ix_(order, order)], 7.0, two_sided=True, rng=3
    )
    for other in (flipped, permuted):
        within_errors(
            base.complement, other.complement, math.hypot(base.error, other.error)
        )


def test_results_do_not_depend_on_order_threads_or_batches() -> None:
    rng = np.random.default_rng(3)
    upper = rng.uniform(1.0, 3.5, (6, 4))
    correlation = equicorrelated(4, 0.3)
    df = np.array([3.0, 3.0, 9.5, math.inf, 9.5, 3.0])
    with thread_limit(1):
        one = multivariate_t_cdf(upper, correlation, df, two_sided=True, rng=9)
    with thread_limit(4):
        many = multivariate_t_cdf(upper, correlation, df, two_sided=True, rng=9)
    reverse = multivariate_t_cdf(
        upper[::-1], correlation, df[::-1], two_sided=True, rng=9
    )
    np.testing.assert_array_equal(one.complement, many.complement)
    np.testing.assert_array_equal(one.error, many.error)
    np.testing.assert_array_equal(one.complement, np.asarray(reverse.complement)[::-1])
    for i in range(6):
        single = multivariate_t_cdf(upper[i], correlation, df[i], two_sided=True, rng=9)
        assert single.complement == one.complement[i]
        assert single.probability == one.probability[i]
        assert single.error == one.error[i]
    other_seed = multivariate_t_cdf(upper, correlation, df, two_sided=True, rng=10)
    assert not np.array_equal(one.complement, other_seed.complement)


def test_arguments_broadcast_over_leading_dimensions() -> None:
    correlations = np.stack([equicorrelated(3, 0.2), equicorrelated(3, 0.7)])
    upper = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    result = multivariate_t_cdf(
        upper[:, None, :], correlations[None], np.array([5.0, math.inf]), rng=2
    )
    assert np.shape(result.complement) == (3, 2)
    assert not np.asarray(result.complement).flags.writeable
    single = multivariate_t_cdf(upper[1], correlations[0], 5.0, rng=2)
    assert isinstance(single.complement, float)
    assert single.complement == result.complement[1, 0]


def test_missing_values_give_nan_for_their_problem_only() -> None:
    upper = np.array([[2.0, 2.0, 2.0], [np.nan, 2.0, 2.0], [2.0, 2.0, 2.0]])
    df = np.array([5.0, 5.0, np.nan])
    result = multivariate_t_cdf(upper, equicorrelated(3, 0.5), df, rng=1)
    assert np.isnan(result.complement[1:]).all() and np.isnan(result.error[1:]).all()
    assert np.isfinite(result.complement[0])
    nan_correlation = multivariate_t_cdf(
        [2.0, 2.0, 2.0], np.full((3, 3), np.nan), 5.0, rng=1
    )
    assert math.isnan(nan_correlation.probability)


def test_infinite_limits() -> None:
    correlation = equicorrelated(3, 0.5)
    # An infinite limit drops its coordinate.
    dropped = multivariate_t_cdf([2.0, np.inf, 3.0], correlation, 6.0, two_sided=True)
    pair = multivariate_t_cdf([2.0, 3.0], equicorrelated(2, 0.5), 6.0, two_sided=True)
    assert dropped.complement == pair.complement
    everything = multivariate_t_cdf([np.inf] * 3, correlation, 6.0)
    assert (everything.probability, everything.complement, everything.error) == (
        1.0,
        0.0,
        0.0,
    )
    nothing = multivariate_t_cdf([1.0, -np.inf, 2.0], correlation, 6.0)
    assert (nothing.probability, nothing.complement, nothing.error) == (0.0, 1.0, 0.0)
    # With three coordinates left, the dropped one leaves the others' problem.
    four = multivariate_t_cdf(
        [2.0, np.inf, 2.5, 3.0], equicorrelated(4, 0.5), 6.0, two_sided=True, rng=2
    )
    three = multivariate_t_cdf([2.0, 2.5, 3.0], correlation, 6.0, two_sided=True, rng=3)
    within_errors(four.complement, three.complement, np.hypot(four.error, three.error))
    assert four.error > 0.0


def test_a_two_sided_limit_of_zero_leaves_no_box() -> None:
    """|T_i| <= 0 has probability 0, as the p-values of a t-statistic of 0 need: the
    complement is exactly 1, without error, with 0 for one limit or for all."""
    correlation = equicorrelated(4, 0.5)
    for upper in ([0.0, 1.0, 2.0, 3.0], [0.0] * 4):
        result = multivariate_t_cdf(upper, correlation, 6.0, two_sided=True, rng=1)
        assert (result.probability, result.complement, result.error) == (0.0, 1.0, 0.0)


def test_extreme_tails_underflow_to_zero_without_nan() -> None:
    correlation = equicorrelated(3, 0.5)
    gone = multivariate_t_cdf([40.0] * 3, correlation, math.inf, two_sided=True, rng=1)
    assert (gone.complement, gone.error, gone.probability) == (0.0, 0.0, 1.0)
    # Near the normal's tail at 37.5, Q(37.5) = 4.6e-308, close to the smallest double.
    near_normal = multivariate_t_cdf(
        [37.5] * 3, correlation, 1e6, two_sided=True, rng=1
    )
    assert np.isfinite(near_normal.complement) and np.isfinite(near_normal.error)
    assert 0.0 < near_normal.complement < 1e-300


def test_small_boxes_keep_their_relative_accuracy() -> None:
    """The probability, not only the complement, carries the requested relative error
    when it is the smaller side: a central box and a tail orthant."""
    correlation = equicorrelated(5, 0.5)
    central = multivariate_t_cdf([0.1] * 5, correlation, 4.0, two_sided=True, rng=1)
    assert central.probability < 1e-4
    assert central.error <= 0.01 * central.probability
    orthant = multivariate_t_cdf([-3.0] * 5, correlation, 4.0, rng=1)
    assert orthant.probability < 0.01
    assert orthant.error <= 0.01 * orthant.probability
    # P(all T_i <= -3) <= P(T_1 <= -3), and at least the independent product.
    from cytepool.stats.distributions import t_cdf

    single = float(t_cdf(-3.0, 4.0))
    assert single**5 <= orthant.probability <= single


def test_the_point_limit_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(multivariate_t, "_MOST_POINTS", 16)
    result = multivariate_t_cdf(
        [2.5] * 4,
        equicorrelated(4, 0.5),
        8.0,
        two_sided=True,
        relative_error=1e-9,
        rng=1,
    )
    assert np.isfinite(result.complement)
    assert result.error > 1e-9 * result.complement


# ---------------------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------------------


def test_one_coordinate_quantiles_have_closed_forms() -> None:
    for p in (0.2, 0.9, 0.99):
        cauchy = multivariate_t_quantile(p, [[1.0]], 1.0)
        assert cauchy.quantile == pytest.approx(
            math.tan(math.pi * (p - 0.5)), rel=1e-11
        )
        assert cauchy.error == 0.0
        two = multivariate_t_quantile(p, [[1.0]], 2.0)
        assert two.quantile == pytest.approx(
            (2 * p - 1) / math.sqrt(2 * p * (1 - p)), rel=1e-11
        )
        normal = multivariate_t_quantile(p, [[1.0]], math.inf, two_sided=True)
        expected = statistics.NormalDist().inv_cdf(0.5 + p / 2)
        assert normal.quantile == pytest.approx(expected, rel=1e-11)


def test_two_coordinate_quantiles_match_mpmath() -> None:
    data = _fixture("quantiles.npz")
    for i in range(data["plane_rho"].size):
        rho = data["plane_rho"][i]
        result = multivariate_t_quantile(
            data["plane_p"][i],
            [[1.0, rho], [rho, 1.0]],
            data["plane_df"][i],
            two_sided=bool(data["plane_two_sided"][i]),
        )
        assert result.quantile == pytest.approx(data["plane_root"][i], rel=1e-9)
        assert result.error < 1e-10


def test_product_correlation_quantiles_match_dunnetts_integral() -> None:
    data = _fixture("quantiles.npz")
    for i in range(data["product_k"].size):
        lam = data["product_lam"][i, : data["product_k"][i]]
        result = multivariate_t_quantile(
            data["product_p"][i],
            product_correlation(lam),
            data["product_df"][i],
            two_sided=bool(data["product_two_sided"][i]),
            rng=400 + i,
        )
        within_errors(result.quantile, data["product_root"][i], result.error)


def test_quantiles_round_trip_through_the_distribution_function() -> None:
    correlation = np.array(
        [
            [1.0, 0.5, -0.2, 0.3],
            [0.5, 1.0, 0.1, 0.6],
            [-0.2, 0.1, 1.0, -0.1],
            [0.3, 0.6, -0.1, 1.0],
        ]
    )
    for p, two_sided in ((0.95, True), (0.99, False), (0.25, True), (0.1, False)):
        q = multivariate_t_quantile(p, correlation, 11.0, two_sided=two_sided, rng=1)
        back = multivariate_t_cdf(
            [q.quantile] * 4, correlation, 11.0, two_sided=two_sided, rng=2
        )
        # The slope of the probability at q turns the error of q into one of p.
        h = 1e-3
        above = multivariate_t_cdf(
            [q.quantile + h] * 4, correlation, 11.0, two_sided=two_sided, rng=2
        )
        below = multivariate_t_cdf(
            [q.quantile - h] * 4, correlation, 11.0, two_sided=two_sided, rng=2
        )
        slope = (above.probability - below.probability) / (2 * h)
        assert abs(back.probability - p) <= CALIBRATED * math.hypot(
            back.error, slope * q.error
        )


def test_quantile_of_identical_coordinates_is_the_univariate_one() -> None:
    from cytepool.stats.distributions import t_isf

    q = multivariate_t_quantile(0.95, np.ones((3, 3)), 6.0, two_sided=True, rng=1)
    assert q.quantile == pytest.approx(float(t_isf(0.025, 6.0)), rel=1e-9)


def test_quantiles_broadcast_and_give_nan() -> None:
    result = multivariate_t_quantile(
        np.array([0.9, 0.95, np.nan]),
        equicorrelated(3, 0.5),
        10.0,
        two_sided=True,
        rng=1,
    )
    assert np.shape(result.quantile) == (3,)
    assert result.quantile[0] < result.quantile[1]
    assert math.isnan(result.quantile[2])
    assert isinstance(result, MultivariateTQuantile)
    correlation = equicorrelated(3, 0.5)
    correlation[0, 1] = correlation[1, 0] = np.nan
    missing = multivariate_t_quantile(0.95, correlation, 10.0, rng=1)
    assert math.isnan(missing.quantile) and math.isnan(missing.error)


# ---------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "options", "error", "message"),
    [
        (([2.0, 2.0], [[1.0, 0.5], [0.4, 1.0]], 5.0), {}, ValueError, "symmetric"),
        (([2.0, 2.0], [[2.0, 0.5], [0.5, 1.0]], 5.0), {}, ValueError, "unit diagonal"),
        (
            ([2.0] * 3, [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]], 5.0),
            {},
            ValueError,
            "positive semidefinite",
        ),
        (([2.0, 2.0], [[1.0, np.inf], [np.inf, 1.0]], 5.0), {}, ValueError, "finite"),
        (([2.0, 2.0], [1.0, 0.5], 5.0), {}, ValueError, "square matrix"),
        (([], np.zeros((0, 0)), 5.0), {}, ValueError, "at least one coordinate"),
        (([2.0, 2.0, 2.0], np.eye(2), 5.0), {}, ValueError, "one limit per coordinate"),
        (([2.0, -1.0], np.eye(2), 5.0), {"two_sided": True}, ValueError, "at least 0"),
        (([2.0, 2.0], np.eye(2), 0.0), {}, ValueError, "df must be positive"),
        (([2.0, 2.0], np.eye(2), -3.0), {}, ValueError, "df must be positive"),
        (
            ([2.0, 2.0], np.eye(2), 5.0),
            {"relative_error": 0.0},
            ValueError,
            "relative_error",
        ),
        (
            ([2.0, 2.0], np.eye(2), 5.0),
            {"relative_error": 1.0},
            ValueError,
            "relative_error",
        ),
        (([2.0, 2.0], np.eye(2), 5.0), {"two_sided": 1}, TypeError, "two_sided"),
        (([2.0, 2.0], np.eye(2), 5.0), {"rng": "seed"}, TypeError, "rng"),
        (([[2.0, 2.0]] * 2, np.eye(2), [5.0, 6.0, 7.0]), {}, ValueError, "broadcast"),
        ((["a", "b"], np.eye(2), 5.0), {}, TypeError, "real numbers"),
    ],
)
def test_arguments_are_refused_with_a_fix(arguments, options, error, message) -> None:
    with pytest.raises(error, match=message):
        multivariate_t_cdf(*arguments, **options)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.5, 2.0])
def test_quantile_probabilities_outside_zero_and_one_are_refused(probability) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        multivariate_t_quantile(probability, np.eye(3), 5.0)


def test_rng_is_checked_even_without_random_numbers() -> None:
    with pytest.raises(ValueError, match="non-negative seed"):
        multivariate_t_cdf([2.0, 2.0], np.eye(2), 5.0, rng=-1)


def test_the_examples_run() -> None:
    from cytepool.stats import multivariate_t

    assert doctest.testmod(multivariate_t, raise_on_error=False).failed == 0
