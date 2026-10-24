from fractions import Fraction

import numpy as np

from cytepool._summation import accurate_mean, accurate_sums, rounded_quotient, two_sum


def test_two_sum_returns_the_exact_rounding_error() -> None:
    for a, b in ((1e16, 1.0), (0.1, 0.2), (-3.5, 1e-300), (2.0**60, -1.0)):
        total, error = two_sum(a, b)
        assert total == a + b
        assert Fraction(total) + Fraction(error) == Fraction(a) + Fraction(b)


def test_accurate_mean_is_the_correctly_rounded_mean() -> None:
    rng = np.random.default_rng(6)
    for values in (
        rng.normal(size=7),
        rng.uniform(0, 1, 1000) * 10.0 ** rng.integers(-5, 5, 1000),
        np.array([1e16, 1.0, -1e16, 3.0, 0.1]),
    ):
        exact = sum(Fraction(v) for v in values) / len(values)
        assert accurate_mean(values) == float(exact)


def test_rounded_quotient_rounds_the_exact_quotient_once() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        total, remainder = two_sum(*rng.normal(size=2) * 10.0 ** rng.integers(-8, 8, 2))
        n = float(rng.integers(1, 1000))
        exact = (Fraction(total) + Fraction(remainder)) / Fraction(n)
        assert rounded_quotient(total, remainder, n) == float(exact)


def test_accurate_sums_round_each_row_and_column_sum_once() -> None:
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(30, 4)) * 10.0 ** rng.integers(-6, 6, (30, 4))
    matrix[0, 0], matrix[1, 0] = 1e16, -1e16  # cancellation that plain sums lose
    columns = accurate_sums(matrix, 0)
    rows = accurate_sums(matrix, 1)
    for j in range(matrix.shape[1]):
        assert columns[j] == float(sum(Fraction(v) for v in matrix[:, j]))
    for i in range(matrix.shape[0]):
        assert rows[i] == float(sum(Fraction(v) for v in matrix[i]))
