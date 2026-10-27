import itertools
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.stats.multiple_testing import adjust_p_values

METHODS = ("bonferroni", "holm", "hochberg", "hommel", "bh", "by", "none")
R_METHODS = ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none")
FIXTURES = Path(__file__).parent / "data" / "multiple_testing" / "p_adjust.npz"


@pytest.fixture(scope="module")
def r_output() -> dict[str, np.ndarray]:
    """Inputs and R's p.adjust output, written by validation/multiple_testing.py."""
    with np.load(FIXTURES) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def p_values() -> np.ndarray:
    """454 p-values with many ties, exact 0, 0.5 and 1, and tiny values.

    The five tied values 1e-4 make Hochberg's step-up beat Holm's step-down.
    """
    rng = np.random.default_rng(7)
    parts = (
        np.round(rng.uniform(size=200), 2),
        rng.uniform(size=200),
        rng.uniform(size=40) ** 30,
        [0.0, 0.5, 1.0] * 3,
        [1e-4] * 5,
    )
    return rng.permutation(np.concatenate(parts))


@pytest.mark.parametrize(
    ("case", "axis"),
    [("mixed", None), ("subset", None), ("columns_small", 0), ("small_rows", 1)],
)
@pytest.mark.parametrize("method", R_METHODS)
def test_matches_r_p_adjust(
    r_output: dict[str, np.ndarray], case: str, axis: int | None, method: str
) -> None:
    n = int(r_output[f"{case}/n"])
    adjusted = adjust_p_values(
        r_output[f"{case}/p"], method, n=None if n < 0 else n, axis=axis
    )
    # NaN must sit where R puts NA (assert_allclose compares their positions).
    assert_allclose(adjusted, r_output[f"{case}/{method}"], rtol=1e-10, atol=0)


# Worked by hand from the definitions, each family in shuffled order. Sorted,
# A is (0.01, 0.02, 0.03, 0.9) and B is (0.01, 0.02, 0.03, 0.04); C is
# (0.01, 0.02, 0.03) in a family of n = 5 tests, two of them unobserved.
# - Holm and Hochberg use (n - j + 1) p_(j): (0.04, 0.06, 0.06, 0.9) for A and
#   (0.04, 0.06, 0.06, 0.04) for B, where the running maximum (Holm) and the
#   running minimum from the end (Hochberg) differ.
# - Hommel, A: the set {0.02, 0.03, 0.9} has the Simes p-value
#   min(3 x 0.02, 3 x 0.03 / 2, 0.9) = 0.045, the largest over sets with 0.02.
#   C: {0.02, 1, 1} gives min(3 x 0.02, 3 / 2, 3 / 3) = 0.06.
# - BH uses n p_(j) / j: (0.04, 0.04, 0.04, 0.9) for A; BY multiplies by
#   1 + 1/2 + 1/3 + 1/4 = 25/12, or 137/60 for n = 5, and caps at 1.
FAMILY_A = [0.03, 0.9, 0.01, 0.02]
FAMILY_B = [0.04, 0.01, 0.03, 0.02]
FAMILY_C = [0.03, 0.01, 0.02]
BY_C = 0.05 * 137 / 60
WORKED = {
    "bonferroni": ([0.12, 1, 0.04, 0.08], [0.16, 0.04, 0.12, 0.08], [0.15, 0.05, 0.1]),
    "holm": ([0.06, 0.9, 0.04, 0.06], [0.06, 0.04, 0.06, 0.06], [0.09, 0.05, 0.08]),
    "hochberg": ([0.06, 0.9, 0.04, 0.06], [0.04] * 4, [0.09, 0.05, 0.08]),
    "hommel": ([0.06, 0.9, 0.04, 0.045], [0.04] * 4, [0.09, 0.05, 0.06]),
    "bh": ([0.04, 0.9, 0.04, 0.04], [0.04] * 4, [0.05] * 3),
    "by": ([1 / 12, 1, 1 / 12, 1 / 12], [1 / 12] * 4, [BY_C] * 3),
    "none": (FAMILY_A, FAMILY_B, FAMILY_C),
}


@pytest.mark.parametrize("method", METHODS)
def test_matches_values_worked_by_hand(method: str) -> None:
    a, b, c = WORKED[method]
    assert_allclose(adjust_p_values(FAMILY_A, method), a, rtol=1e-12)
    assert_allclose(adjust_p_values(FAMILY_B, method), b, rtol=1e-12)
    assert_allclose(adjust_p_values(FAMILY_C, method, n=5), c, rtol=1e-12)


def test_default_method_is_benjamini_hochberg() -> None:
    assert_allclose(adjust_p_values(FAMILY_A), WORKED["bh"][0], rtol=1e-12)


def _simes(values: list[float]) -> float:
    values = sorted(values)
    return min(len(values) * v / j for j, v in enumerate(values, start=1))


def _closed_testing(p: list[float], count: int) -> list[float]:
    """Hommel's adjusted values of the first `count` tests, by definition: the
    largest Simes p-value over every set of tests that contains the test."""
    adjusted = []
    for i, value in enumerate(p[:count]):
        others = p[:i] + p[i + 1 :]
        adjusted.append(
            max(
                _simes([value, *subset])
                for size in range(len(others) + 1)
                for subset in itertools.combinations(others, size)
            )
        )
    return adjusted


def test_hommel_equals_closed_testing_of_simes_tests() -> None:
    rng = np.random.default_rng(11)
    grid = np.array([0.0, 0.01, 0.02, 0.05, 0.3, 0.6, 1.0])
    for trial in range(300):
        size = int(rng.integers(1, 7))
        p = rng.choice(grid, size) if trial % 2 else rng.uniform(size=size) ** 3
        unobserved = int(rng.integers(0, 3))
        expected = _closed_testing([*p.tolist(), *[1.0] * unobserved], size)
        adjusted = adjust_p_values(p, "hommel", n=size + unobserved)
        assert_allclose(adjusted, expected, rtol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_adjusted_values_keep_the_order_of_the_p_values(
    p_values: np.ndarray, method: str
) -> None:
    adjusted = adjust_p_values(p_values, method)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= p_values) & (adjusted <= 1))
    for value in (0.0, 0.5, 1.0):
        tied = p_values == value
        assert tied.sum() > 1 and np.ptp(adjusted[tied]) == 0


def test_methods_are_ordered_as_theory_guarantees(p_values: np.ndarray) -> None:
    adjusted = {method: adjust_p_values(p_values, method) for method in METHODS}
    # Adjusted values never decrease along this chain, and BY is at least BH.
    # Equal values reached by different formulas may differ in the last bit,
    # hence the slack of a few units of rounding.
    chain = ("none", "bh", "hommel", "hochberg", "holm", "bonferroni")
    for lower, upper in [*itertools.pairwise(chain), ("bh", "by")]:
        assert np.all(adjusted[lower] <= adjusted[upper] * (1 + 1e-15)), (lower, upper)
    assert np.any(adjusted["hommel"] < adjusted["hochberg"] * (1 - 1e-6))
    assert np.any(adjusted["hochberg"] < adjusted["holm"] * (1 - 1e-6))


def test_hommel_scales_to_100000_p_values() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(size=100_000)
    p[:5000] **= 20
    adjust_p_values(p[:10], "hommel")  # compile before timing
    start = time.perf_counter()
    hommel = adjust_p_values(p, "hommel")
    # The sweeps take milliseconds here; a quadratic algorithm takes seconds
    # to minutes, which this loose bound still catches.
    assert time.perf_counter() - start < 1.0
    order = np.argsort(p)
    assert np.all(np.diff(hommel[order]) >= 0)
    assert np.all(adjust_p_values(p, "bh") <= hommel * (1 + 1e-15))
    assert np.all(hommel <= adjust_p_values(p, "hochberg") * (1 + 1e-15))


def test_by_in_very_large_families() -> None:
    # Harmonic numbers from mpmath at 40 digits. Past 2**20 tests the sum is
    # replaced by its asymptotic expansion, which keeps the cost constant. The
    # tolerance leaves two units in the last place for the platform's log.
    harmonic = {
        2**20 + 1: 14.44016070661092837,
        10**9: 21.30048150234794402,
        2**53: 37.31401623457863432,
    }
    for n, value in harmonic.items():
        adjusted = adjust_p_values([1e-300], "by", n=n)
        assert_allclose(adjusted, [value * n * 1e-300], rtol=5e-16)


def test_missing_values_stay_missing_and_are_not_counted() -> None:
    # Two tests, not three: R's p.adjust also leaves NA out of n.
    assert_allclose(
        adjust_p_values([0.01, np.nan, 0.02], "bonferroni"), [0.02, np.nan, 0.04]
    )
    p = np.array([0.04, np.nan, 0.01, 0.3, np.nan, 0.02])
    for method in METHODS:
        adjusted = adjust_p_values(p, method)
        assert_array_equal(np.isnan(adjusted), np.isnan(p))
        assert_array_equal(
            adjusted[[0, 2, 3, 5]], adjust_p_values(p[[0, 2, 3, 5]], method)
        )


@pytest.mark.parametrize("method", METHODS)
def test_unobserved_tests_act_as_p_values_of_one(
    p_values: np.ndarray, method: str
) -> None:
    observed = p_values[:60]
    padded = np.concatenate([observed, np.ones(140)])
    # The two inputs take different paths through Hommel's procedure, so the
    # last bits may differ.
    assert_allclose(
        adjust_p_values(observed, method, n=200),
        adjust_p_values(padded, method)[:60],
        rtol=1e-14,
    )


def test_axis_selects_the_families() -> None:
    p = np.array([[0.01, 0.02, 0.03], [0.04, np.nan, 0.06]])
    # Bonferroni multiplies by the number of non-missing p-values per family.
    expected = {
        None: [[0.05, 0.1, 0.15], [0.2, np.nan, 0.3]],
        0: [[0.02, 0.02, 0.06], [0.08, np.nan, 0.12]],
        1: [[0.03, 0.06, 0.09], [0.08, np.nan, 0.12]],
        -1: [[0.03, 0.06, 0.09], [0.08, np.nan, 0.12]],
    }
    for axis, values in expected.items():
        assert_allclose(adjust_p_values(p, "bonferroni", axis=axis), values, rtol=1e-15)


def test_axis_of_a_three_dimensional_array(p_values: np.ndarray) -> None:
    p = p_values[:120].reshape(4, 5, 6).copy()
    p[1, :, 2] = np.nan
    adjusted = adjust_p_values(p, "hommel", axis=1, n=7)
    assert adjusted.shape == p.shape and adjusted.flags.c_contiguous
    for i, k in itertools.product(range(4), range(6)):
        assert_array_equal(
            adjusted[i, :, k], adjust_p_values(p[i, :, k], "hommel", n=7)
        )


@pytest.mark.parametrize("method", METHODS)
def test_empty_scalar_and_all_missing_inputs(method: str) -> None:
    assert adjust_p_values([], method).shape == (0,)
    assert adjust_p_values(np.empty((0, 3)), method, axis=0).shape == (0, 3)
    assert adjust_p_values(np.empty((0, 3)), method, axis=1).shape == (0, 3)
    single = adjust_p_values(0.3, method)
    assert single.shape == () and single == 0.3
    assert np.isnan(adjust_p_values([np.nan, np.nan], method, n=4)).all()


def test_input_types_and_copies() -> None:
    assert_array_equal(adjust_p_values([0, 1], "holm"), [0.0, 1.0])
    float32 = adjust_p_values(np.array([0.25, 0.5], dtype=np.float32), "bonferroni")
    assert float32.dtype == np.float64
    assert_array_equal(float32, [0.5, 1.0])
    p = np.array([0.2, 0.4])
    adjust_p_values(p, "none")[0] = 0.9
    assert p[0] == 0.2


def test_method_names_are_case_insensitive_with_aliases(p_values: np.ndarray) -> None:
    aliases = {
        "bh": ("BH", "fdr", "Benjamini-Hochberg"),
        "by": ("BY", "benjamini-yekutieli"),
        "hommel": ("Hommel",),
    }
    for name, others in aliases.items():
        for alias in others:
            assert_array_equal(
                adjust_p_values(p_values, alias), adjust_p_values(p_values, name)
            )


def test_n_may_exceed_the_values_but_not_fall_below_the_observed_count() -> None:
    # R: p.adjust(c(0.1, 0.2, NA), n = 2) gives 0.2, 0.2, NA.
    assert_allclose(
        adjust_p_values([0.1, 0.2, np.nan], "holm", n=2), [0.2, 0.2, np.nan]
    )
    with pytest.raises(
        ValueError, match=r"non-missing p-values in each family \(3\), got 2"
    ):
        adjust_p_values([0.1, 0.2, 0.3], n=2)
    with pytest.raises(ValueError, match=r"in each family \(2\), got 1"):
        adjust_p_values([[0.1, np.nan], [0.2, 0.3]], n=1, axis=1)
    with pytest.raises(ValueError, match=r"at most 2\*\*53, got 9223372036854775808"):
        adjust_p_values([0.1], n=2**63)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(
        ValueError, match="unknown method 'sidak'; expected one of 'bonferroni'"
    ):
        adjust_p_values([0.1], "sidak")
    with pytest.raises(TypeError, match="method must be a string, got NoneType"):
        adjust_p_values([0.1], None)  # type: ignore[arg-type]
    for bad in (1.5, -0.1, np.inf):
        with pytest.raises(
            ValueError,
            match=rf"lie in \[0, 1\] or be NaN; found 1 outside, such as {bad}",
        ):
            adjust_p_values([0.1, bad])
    for values in (["0.1"], [0.1 + 0j], [True, False]):
        with pytest.raises(
            TypeError, match="p_values must hold real numbers, got dtype"
        ):
            adjust_p_values(values)
    with pytest.raises(TypeError, match="n must be an integer, got float"):
        adjust_p_values([0.1], n=2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="n must be an integer, got a boolean"):
        adjust_p_values([0.1], n=True)
    with pytest.raises(ValueError, match="axis 2 is out of range for a 2-dimensional"):
        adjust_p_values([[0.1]], axis=2)
    with pytest.raises(ValueError, match="axis 0 is out of range for a 0-dimensional"):
        adjust_p_values(0.1, axis=0)
    with pytest.raises(TypeError, match="axis must be an integer, got float"):
        adjust_p_values([0.1], axis=0.0)  # type: ignore[arg-type]
