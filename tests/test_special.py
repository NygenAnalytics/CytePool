import itertools
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.stats import special

DATA = Path(__file__).parent / "data" / "distributions"
EPS = np.finfo(float).eps
# Accuracy contract of the incomplete gamma and beta functions against mpmath.
CONTRACT = 1e-12
# The gamma-function family is within a few units in the last place.
NEAR_MACHINE = 1e-15


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz") as data:
        return dict(data)


def _scaled_error(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Absolute error near zeros of the function, relative error elsewhere."""
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


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


def assert_log_close(got: np.ndarray, ref: np.ndarray) -> None:
    """Relative error at most 1e-12 where abs(ref) >= 1e-300; below, agreement to 1e-312."""
    got, ref = np.asarray(got), np.asarray(ref)
    checked = np.abs(ref) >= 1e-300
    np.testing.assert_allclose(got[checked], ref[checked], rtol=CONTRACT, atol=0)
    np.testing.assert_allclose(got[~checked], ref[~checked], rtol=0, atol=1e-312)


def assert_probability_close(
    got: np.ndarray, ref: np.ndarray, rtol: float = CONTRACT
) -> None:
    """Relative error at most rtol down to 1e-300; below that, agreement to 1e-312."""
    got, ref = np.asarray(got), np.asarray(ref)
    checked = ref >= 1e-300
    np.testing.assert_allclose(got[checked], ref[checked], rtol=rtol, atol=0)
    np.testing.assert_allclose(got[~checked], ref[~checked], rtol=0, atol=1e-312)


# ---------------------------------------------------------------------------------------
# Agreement with mpmath (50 digits) on the validation fixtures
# ---------------------------------------------------------------------------------------


def test_lgamma_matches_high_precision_values() -> None:
    data = _load("lgamma")
    error = _scaled_error(special.lgamma(data["x"]), data["value"])
    assert error.max() <= NEAR_MACHINE


def test_digamma_matches_high_precision_values() -> None:
    data = _load("digamma")
    got = special.digamma(data["x"])
    positive = data["x"] > 0
    # Relative accuracy holds on x > 0, including next to the root at 1.4616...
    np.testing.assert_allclose(
        got[positive], data["value"][positive], rtol=NEAR_MACHINE, atol=0
    )
    # For x < 0 the reflection formula cancels near the negative roots, so the error is
    # measured relative to max(1, |psi|).
    assert (
        _scaled_error(got[~positive], data["value"][~positive]).max()
        <= 2 * NEAR_MACHINE
    )


def test_digamma_keeps_relative_accuracy_at_its_positive_root() -> None:
    root = 1.4616321449683622  # the double nearest the root; psi there is -9.2412655217e-17
    assert special.digamma(root) == pytest.approx(-9.2412655217294275e-17, rel=4 * EPS)
    assert special.digamma(np.nextafter(root, 2.0)) > 0 > special.digamma(root)


def test_trigamma_matches_high_precision_values() -> None:
    data = _load("trigamma")
    np.testing.assert_allclose(
        special.trigamma(data["x"]), data["value"], rtol=NEAR_MACHINE, atol=0
    )


def test_trigamma_inverse_matches_high_precision_values() -> None:
    data = _load("trigamma_inverse")
    got = special.trigamma_inverse(data["y"])
    np.testing.assert_allclose(got, data["value"], rtol=NEAR_MACHINE, atol=0)


def test_log_beta_matches_high_precision_values() -> None:
    data = _load("log_beta")
    error = _scaled_error(special.log_beta(data["a"], data["b"]), data["value"])
    assert error.max() <= 2 * NEAR_MACHINE


def test_incomplete_gamma_matches_high_precision_values() -> None:
    data = _load("gammainc")
    a, x = data["a"], data["x"]
    # shapes from 1e-3 to 1e18, the largest served by the uniform expansion
    assert a.min() <= 1e-3 and a.max() >= 1e18
    assert_probability_close(special.gammainc(a, x), data["P"])
    assert_probability_close(special.gammaincc(a, x), data["Q"])
    assert_log_close(special.log_gammainc(a, x), data["logP"])
    assert_log_close(special.log_gammaincc(a, x), data["logQ"])


def test_incomplete_beta_matches_high_precision_values() -> None:
    data = _load("betainc")
    a, b, x = data["a"], data["b"], data["x"]
    assert_probability_close(special.betainc(a, b, x), data["I"])
    assert_probability_close(special.betaincc(a, b, x), data["C"])
    assert_log_close(special.log_betainc(a, b, x), data["logI"])
    assert_log_close(special.log_betaincc(a, b, x), data["logC"])


def test_incomplete_beta_matches_high_precision_values_at_extreme_arguments() -> None:
    # Shapes from the smallest subnormal to the largest double, and points near the
    # center of shapes from 1e10, where the uniform expansion serves them.
    data = _load("betainc_extreme")
    a, b, x = data["a"], data["b"], data["x"]
    assert_probability_close(special.betainc(a, b, x), data["I"])
    assert_probability_close(special.betaincc(a, b, x), data["C"])
    assert_log_close(special.log_betainc(a, b, x), data["logI"])
    assert_log_close(special.log_betaincc(a, b, x), data["logC"])


LARGEST = float(np.finfo(np.float64).max)
EDGES = [0.0, 5e-324, 1e-300, 1e-8, 0.5, 1.0 - 2.0**-53, 1.0, 2.0, 30.0, 1e8, 1e300]
EDGES.append(LARGEST)


def test_incomplete_beta_is_fast_at_every_extreme_argument() -> None:
    # Some of these once ran 10^7 steps of the continued fraction (0.15 s a value),
    # and with a + b beyond the double range a deviance term never returned.
    a, b, x = (np.array(v) for v in zip(*itertools.product(EDGES, repeat=3)))
    special.log_betainc(2.0, 3.0, 0.5)  # compiled
    start = time.perf_counter()
    log_i = special.log_betainc(a, b, x)
    log_c = special.log_betaincc(a, b, x)
    elapsed = time.perf_counter() - start

    # 3,456 values take about 0.01 s; the bound leaves room for slow machines.
    assert elapsed < 2.0
    domain = (a > 0.0) & (b > 0.0) & (x <= 1.0)
    assert not np.isnan(log_i[domain]).any() and not np.isnan(log_c[domain]).any()
    assert np.isnan(log_i[~domain]).all() and np.isnan(log_c[~domain]).all()
    # One tail is the complement of the other.
    both = domain & (log_i > -700.0) & (log_c > -700.0)
    assert_allclose(np.exp(log_i[both]) + np.exp(log_c[both]), 1.0, rtol=4 * EPS)


def test_incomplete_beta_of_equal_shapes_is_one_half_at_one_half() -> None:
    shapes = np.array([1e10, 3e15, 1e20, 1e300, LARGEST])
    assert_array_equal(special.betainc(shapes, shapes, 0.5), 0.5)
    assert_array_equal(special.betaincc(shapes, shapes, 0.5), 0.5)


def test_incomplete_beta_near_the_center_of_large_shapes_is_its_reflection() -> None:
    # I_x(a, b) = 1 - I_(1-x)(b, a), with x and 1 - x exact, in the band of the uniform
    # expansion (about one standard deviation, 2.2e-7, of the center) and beyond it,
    # where the continued fraction takes over.
    a, b = 1e12, 3e12
    sd = math.sqrt(3.0 / 16.0 / (a + b + 1.0))
    x = 0.25 + np.arange(-6, 7) * 2.0**-23
    assert np.abs(x - 0.25).max() > 3.0 * sd
    total = special.betainc(a, b, x) + special.betainc(b, a, 1.0 - x)
    assert_allclose(total, 1.0, rtol=2 * EPS)
    assert np.all(np.diff(special.betainc(a, b, x)) > 0.0)


def test_incomplete_gamma_returns_where_the_shape_and_the_argument_overflow() -> None:
    # Loader's deviance term of a + x beyond the double range once never returned.
    # mpmath, from Legendre's continued fraction at 40 digits, gives
    # log Q(1e300, LARGEST) = -1.7976929347904657564e308.
    assert_allclose(special.log_gammaincc(1e300, LARGEST), -1.7976929347904657564e308)
    assert special.gammainc(LARGEST, LARGEST) == 0.5  # 1/2 + 1/(3 sqrt(2 pi a))
    assert special.log_gammainc(LARGEST, 1e300) == -np.inf  # below -LARGEST


@pytest.mark.parametrize("a", [1e-8, 1e-20, 1e-300, 5e-324])
def test_incomplete_beta_of_a_tiny_shape_against_its_closed_form(a: float) -> None:
    # 1 - I_x(a, 1) = 1 - x^a = -expm1(t) with t = a log x, whose logarithm is
    # log(a) + log(-log x) + log(expm1(t) / t); t is subnormal for the smallest a.
    x = np.array([1e-300, 1e-8, 0.5, 0.99])
    t = a * np.log(x)
    ratio = np.where(np.abs(t) > 1e-300, np.expm1(t) / np.where(t == 0.0, 1.0, t), 1.0)
    expected = np.log(a) + np.log(-np.log(x)) + np.log(ratio)
    assert_allclose(special.log_betaincc(a, 1.0, x), expected, rtol=4 * EPS)
    # 1 - x is exact from x = 1/2.
    assert_allclose(
        special.log_betainc(1.0, a, 1.0 - x[2:]), expected[2:], rtol=4 * EPS
    )


# ---------------------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------------------


def test_lgamma_at_integers_and_one_half() -> None:
    n = np.arange(1, 171)
    expected = np.array([math.log(math.factorial(k - 1)) if k > 2 else 0.0 for k in n])
    error = _scaled_error(special.lgamma(n.astype(float)), expected)
    assert error.max() <= NEAR_MACHINE
    assert special.lgamma(0.5) == pytest.approx(0.5 * math.log(math.pi), rel=EPS)


def test_digamma_and_trigamma_at_integers_and_one_half() -> None:
    euler = 0.5772156649015329
    n = np.arange(1, 60)
    harmonic = [float(sum(Fraction(1, k) for k in range(1, m))) for m in n]
    np.testing.assert_allclose(
        special.digamma(n.astype(float)), np.array(harmonic) - euler, rtol=4 * EPS
    )
    assert special.digamma(0.5) == pytest.approx(-euler - 2 * math.log(2), rel=2 * EPS)
    # pi^2/6 - sum_{k<n} 1/k^2 cancels as n grows; n <= 10 keeps the expected values
    # accurate to a few units in the last place.
    small = n[:10]
    squares = [float(sum(Fraction(1, k * k) for k in range(1, m))) for m in small]
    expected = math.pi**2 / 6 - np.array(squares)
    np.testing.assert_allclose(
        special.trigamma(small.astype(float)), expected, rtol=1e-14
    )
    assert special.trigamma(0.5) == pytest.approx(math.pi**2 / 2, rel=2 * EPS)


def test_incomplete_gamma_closed_forms() -> None:
    x = np.concatenate([[1e-300, 1e-10, 0.3], np.linspace(0.5, 700.0, 50)])
    # a = 1: Q = exp(-x); a = 1/2: Q = erfc(sqrt x); integer a: a Poisson sum.
    assert_close_to_floor(special.gammaincc(1.0, x), np.exp(-x))
    assert_close_to_floor(special.gammainc(1.0, x), -np.expm1(-x))
    x_moderate = x[x < 100]
    erfc = np.array([math.erfc(math.sqrt(v)) for v in x_moderate])
    assert_close_to_floor(special.gammaincc(0.5, x_moderate), erfc)
    for a in (3, 7):
        terms = [[v**k / math.factorial(k) for k in range(a)] for v in x]
        poisson = np.exp(-x) * np.array([math.fsum(row) for row in terms])
        assert_close_to_floor(special.gammaincc(float(a), x), poisson)


def test_incomplete_beta_closed_forms() -> None:
    x = np.array([1e-300, 1e-12, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-12])
    assert_close_to_floor(special.betainc(1.0, 1.0, x), x)
    for a in (1e-3, 0.5, 3.0, 40.0):
        # I_x(a, 1) = x^a and 1 - I_x(1, a) = (1 - x)^a
        power = x**a
        kept = power > 0
        assert_close_to_floor(special.betainc(a, 1.0, x)[kept], power[kept])
        complement = np.exp(a * np.log1p(-x))
        kept = complement > 0
        assert_close_to_floor(special.betaincc(1.0, a, x)[kept], complement[kept])


def test_log_beta_closed_forms() -> None:
    b = np.array([1e-3, 0.5, 2.0, 37.0, 1e5, 1e300])
    np.testing.assert_allclose(special.log_beta(1.0, b), -np.log(b), rtol=4 * EPS)
    assert special.log_beta(0.5, 0.5) == pytest.approx(math.log(math.pi), rel=4 * EPS)
    # B(m, n) = (m - 1)! (n - 1)! / (m + n - 1)! for integers
    for m, n in [(3, 4), (12, 30), (50, 51)]:
        exact = math.log(
            Fraction(
                math.factorial(m - 1) * math.factorial(n - 1), math.factorial(m + n - 1)
            )
        )
        assert special.log_beta(float(m), float(n)) == pytest.approx(exact, rel=1e-15)


# ---------------------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------------------


def test_lower_and_upper_incomplete_functions_sum_to_one() -> None:
    a = np.array([1e-3, 0.3, 1.0, 4.5, 30.0, 1e4])[:, None]
    x = np.array([1e-5, 0.1, 0.9, 1.5, 3.0, 30.0, 1e4 - 100, 1e4 + 100])[None, :]
    total = special.gammainc(a, x) + special.gammaincc(a, x)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)
    b = np.array([1e-3, 0.5, 2.0, 40.0, 1e5])[None, :, None]
    u = np.array([1e-6, 0.2, 0.5, 0.8, 1.0 - 1e-6])[None, None, :]
    total = special.betainc(a[:, :, None], b, u) + special.betaincc(a[:, :, None], b, u)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=4 * EPS)


def test_recurrence_and_reflection_identities() -> None:
    # Below 10 digamma is itself computed by this recurrence, so the identity is checked
    # where at least one side comes from the asymptotic expansion.
    x = np.linspace(9.0, 40.0, 621)
    lhs = special.digamma(x + 1.0) - special.digamma(x)
    scale = np.abs(special.digamma(x + 1.0)) + np.abs(special.digamma(x))
    assert np.all(np.abs(lhs - 1.0 / x) <= 8 * EPS * (scale + 1.0 / x))
    # u <= 1/2 keeps sin(pi u) in the test accurate to a few units in the last place.
    u = np.linspace(0.01, 0.5, 50)
    reflected = special.trigamma(u) + special.trigamma(1.0 - u)
    np.testing.assert_allclose(
        reflected, math.pi**2 / np.sin(math.pi * u) ** 2, rtol=1e-14
    )


def test_digamma_at_half_integers() -> None:
    # psi(1/2 + n) = psi(1/2 - n) exactly (the reflection term pi cot(pi x) vanishes),
    # and psi(3/2) = 2 - euler - 2 log 2.
    n = np.arange(1.0, 12.0)
    assert np.array_equal(special.digamma(0.5 - n), special.digamma(0.5 + n))
    assert special.digamma(1.5) == pytest.approx(0.036489973978576520559, rel=4 * EPS)
    assert special.digamma(-0.5) == pytest.approx(0.036489973978576520559, rel=4 * EPS)


def test_log_beta_does_not_overflow_for_huge_arguments() -> None:
    # a + b overflows; log B(a, a) = -2 a log 2 + O(log a)
    huge = special.log_beta(1e308, 1e308)
    assert huge == pytest.approx(-1.3862943611198906341e308, rel=4 * EPS)


def test_trigamma_inverse_round_trip() -> None:
    x = 10.0 ** np.linspace(-8, 8, 161)
    np.testing.assert_allclose(
        special.trigamma_inverse(special.trigamma(x)), x, rtol=4e-15
    )


def test_monotone_in_the_argument() -> None:
    x = np.linspace(1e-3, 60.0, 4001)
    assert np.all(np.diff(special.digamma(x)) > 0)
    assert np.all(np.diff(special.trigamma(x)) < 0)
    for a in (1e-3, 0.7, 5.0, 25.0):
        assert np.all(np.diff(special.gammainc(a, x)) >= 0)
        assert np.all(np.diff(special.log_gammaincc(a, x)) <= 0)
    u = np.linspace(0.0, 1.0, 4001)
    for a, b in [(1e-3, 2.0), (0.5, 0.5), (3.0, 40.0), (200.0, 150.0)]:
        assert np.all(np.diff(special.betainc(a, b, u)) >= 0)
        assert np.all(np.diff(special.log_betaincc(a, b, u)) <= 0)


# ---------------------------------------------------------------------------------------
# Special values, domains and argument handling
# ---------------------------------------------------------------------------------------


def test_values_at_poles_and_infinities() -> None:
    inf = math.inf
    assert special.lgamma(0.0) == inf and special.lgamma(-3.0) == inf
    assert special.lgamma(inf) == inf and math.isnan(special.lgamma(-inf))
    assert math.isnan(special.digamma(0.0)) and math.isnan(special.digamma(-2.0))
    assert special.digamma(inf) == inf and math.isnan(special.digamma(-inf))
    assert special.trigamma(0.0) == inf and special.trigamma(-4.0) == inf
    assert special.trigamma(inf) == 0.0 and math.isnan(special.trigamma(-inf))
    assert special.trigamma_inverse(0.0) == inf and special.trigamma_inverse(inf) == 0.0
    assert special.log_beta(0.0, 2.0) == inf and special.log_beta(inf, 2.0) == -inf
    assert special.gammainc(2.0, 0.0) == 0.0 and special.gammainc(2.0, inf) == 1.0
    assert (
        special.log_gammainc(2.0, 0.0) == -inf
        and special.log_gammaincc(2.0, inf) == -inf
    )
    assert (
        special.betainc(2.0, 3.0, 0.0) == 0.0 and special.betainc(2.0, 3.0, 1.0) == 1.0
    )
    assert (
        special.log_betainc(2.0, 3.0, 0.0) == -inf
        and special.log_betaincc(2.0, 3.0, 1.0) == -inf
    )


@pytest.mark.parametrize(
    "function, arguments",
    [
        (special.trigamma_inverse, (-1.0,)),
        (special.log_beta, (-1.0, 2.0)),
        (special.log_beta, (0.0, math.inf)),
        (special.gammainc, (0.0, 1.0)),
        (special.gammainc, (-1.0, 1.0)),
        (special.gammaincc, (math.inf, 1.0)),
        (special.log_gammainc, (1.0, -1.0)),
        (special.betainc, (0.0, 1.0, 0.5)),
        (special.betaincc, (1.0, -2.0, 0.5)),
        (special.log_betainc, (1.0, math.inf, 0.5)),
        (special.log_betaincc, (1.0, 1.0, 1.5)),
        (special.betainc, (1.0, 1.0, -0.1)),
    ],
)
def test_arguments_outside_the_domain_give_nan(function, arguments) -> None:
    assert math.isnan(function(*arguments))


@pytest.mark.parametrize(
    "function, arity",
    [
        (special.lgamma, 1),
        (special.digamma, 1),
        (special.trigamma, 1),
        (special.trigamma_inverse, 1),
        (special.log_beta, 2),
        (special.gammainc, 2),
        (special.gammaincc, 2),
        (special.log_gammainc, 2),
        (special.log_gammaincc, 2),
        (special.betainc, 3),
        (special.betaincc, 3),
        (special.log_betainc, 3),
        (special.log_betaincc, 3),
    ],
)
def test_nan_in_any_argument_gives_nan(function, arity) -> None:
    for position in range(arity):
        arguments = [0.5] * arity
        arguments[position] = math.nan
        assert math.isnan(function(*arguments))


def test_scalars_give_floats_and_arrays_broadcast() -> None:
    assert type(special.gammainc(2.0, 1.5)) is float
    assert type(special.gammainc(2, 1)) is float
    assert type(special.lgamma(np.float32(3.0))) is float
    assert type(special.lgamma(np.array(3.0))) is float
    out = special.betainc(np.array([[1.0], [2.0], [3.0]]), [0.5, 1.0, 2.0, 4.0], 0.3)
    assert (
        isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (3, 4)
    )
    assert special.betainc(2.0, 2.0, 0.3) == out[1, 2]
    assert special.digamma([1.0, 2.0]).shape == (2,)
    assert special.digamma(np.empty(0)).shape == (0,)
    assert special.lgamma(np.array([True, False]))[0] == 0.0


@pytest.mark.parametrize(
    "bad", ["1.5", None, 1.0 + 2.0j, [1.0, "a"], np.array([object()])]
)
def test_non_numeric_arguments_raise_type_error(bad) -> None:
    with pytest.raises(TypeError, match="must contain real numbers"):
        special.gammainc(bad, 1.0)
    with pytest.raises(TypeError, match="x must contain real numbers"):
        special.digamma(bad)


def test_incompatible_shapes_raise() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        special.gammainc(np.ones(3), np.ones(4))
