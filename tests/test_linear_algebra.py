import numpy as np
import pytest
from numpy.testing import assert_allclose

from cytepool._linear_algebra import RANK_TOLERANCE, column_space


def test_a_combination_of_earlier_columns_is_aliased() -> None:
    x = np.array([0.5, 1.0, 2.5, 4.0, 7.0])
    matrix = np.column_stack([np.ones(5), x, 3.0 * x - 2.0, x**2])
    space = column_space(matrix)
    assert space.kept == (0, 1, 3)
    assert space.aliased == (2,)
    assert space.rank == 3
    assert space.remainders[2] < 1e-15


def test_the_later_of_two_collinear_columns_is_the_aliased_one() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert column_space(np.column_stack([x, 2.0 * x])).aliased == (1,)
    assert column_space(np.column_stack([2.0 * x, x])).aliased == (1,)


@pytest.mark.parametrize(("relative_size", "aliased"), [(1e-6, ()), (1e-8, (1,))])
def test_the_tolerance_is_relative_to_the_column_norm(
    relative_size: float, aliased: tuple[int, ...]
) -> None:
    ones = np.ones(4)
    # A unit vector orthogonal to the first column.
    direction = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
    nearly = 1e5 * (ones + relative_size * np.linalg.norm(ones) * direction)
    space = column_space(np.column_stack([ones, nearly]))
    assert space.aliased == aliased
    assert_allclose(space.remainders[1], relative_size, rtol=1e-6)
    assert (relative_size <= RANK_TOLERANCE) == bool(aliased)


def test_zero_columns_and_columns_beyond_the_row_count_are_aliased() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(3, 5))
    matrix[:, 1] = 0.0
    space = column_space(matrix)
    assert space.kept == (0, 2, 3)
    assert space.aliased == (1, 4)
    assert space.remainders[1] == 0.0
    assert space.scales[1] == 1.0


def test_the_basis_is_orthonormal_and_spans_the_kept_columns() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(9, 4))
    matrix[:, 3] = matrix[:, 0] - 2.0 * matrix[:, 2]
    space = column_space(matrix)
    basis = space.basis
    assert basis.shape == (9, 3)
    assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    projected = basis @ (basis.T @ matrix)
    assert_allclose(projected, matrix, atol=1e-13)


def test_extreme_column_scales_neither_overflow_nor_change_decisions() -> None:
    x = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    matrix = np.column_stack([np.ones(5), x, 2.0 * x])
    huge = matrix * np.array([1.0, 1e300, 1e-300])
    plain, scaled = column_space(matrix), column_space(huge)
    assert scaled.kept == plain.kept == (0, 1)
    assert scaled.aliased == plain.aliased == (2,)
    assert np.all(np.isfinite(scaled.basis))
    assert_allclose(np.abs(scaled.basis), np.abs(plain.basis), atol=1e-14)


def test_leverages_of_a_one_way_layout_are_one_over_group_size() -> None:
    groups = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    indicators = (groups[:, np.newaxis] == np.arange(3)).astype(float)
    with_intercept = np.column_stack([np.ones(9), indicators])  # one column aliased
    space = column_space(with_intercept)
    assert space.rank == 3
    assert_allclose(space.leverages(), 1.0 / np.bincount(groups)[groups], rtol=1e-14)
    assert_allclose(space.leverages().sum(), 3.0, rtol=1e-14)


def test_invalid_matrices_are_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        column_space(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        column_space(np.array([[1.0, np.nan], [2.0, 3.0]]))
