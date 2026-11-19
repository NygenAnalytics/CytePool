"""Tests for the F-tests that the moderated statistics share.

``moderated_f_tests`` is checked against F-statistics written out by hand for chosen
t-statistics and correlations. ``standardized_f_statistics`` is checked against the
closed form of F(2, d), against the F-statistic itself on infinite denominator degrees
of freedom, by the round trip of its tail probability over F-statistics up to 1e300,
where the p-values underflow, and against scipy for a single value.
"""

import numpy as np
import pytest
from scipy import stats

from cytepool.stats._moderation import moderated_f_tests, standardized_f_statistics
from cytepool.stats.distributions import f_logsf, f_sf


def test_each_features_own_correlation_gives_its_f_statistic() -> None:
    t = np.array([[1.5, -2.0], [3.0, 1.0], [0.5, 0.25]])
    # Feature 0's own covariance has the correlation 0.5 (standard deviations 2 and
    # 1), feature 1's is singular and feature 2's is the identity, which every
    # feature shares otherwise.
    own = np.array([[[4.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], np.eye(2)])
    f_value, df1, df2, p_value = moderated_f_tests(
        t, np.eye(2), own, 4.0, np.full(3, 6.0)
    )
    # t' R^-1 t / 2 with R = [[1, 0.5], [0.5, 1]] is (t1^2 - t1 t2 + t2^2) / 1.5.
    t1, t2 = t[0]
    np.testing.assert_allclose(
        f_value[0], (t1 * t1 - t1 * t2 + t2 * t2) / 1.5, rtol=1e-13
    )
    assert np.isnan(f_value[1]) and np.isnan(p_value[1])
    np.testing.assert_allclose(f_value[2], (0.5**2 + 0.25**2) / 2, rtol=1e-13)
    assert df1 == 2
    np.testing.assert_array_equal(df2, 10.0)
    np.testing.assert_allclose(
        p_value[[0, 2]], stats.f.sf(f_value[[0, 2]], 2, 10), rtol=1e-12
    )
    # Without the features' own covariances, the identity serves all three: t't / 2.
    shared, *_ = moderated_f_tests(t, np.eye(2), None, 4.0, np.full(3, 6.0))
    np.testing.assert_allclose(shared, np.sum(t * t, axis=1) / 2, rtol=1e-13)


def test_features_without_finite_t_statistics_have_no_f_statistic() -> None:
    t = np.full((3, 2), np.nan)
    t[1, 0] = 2.0  # one of two
    own = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    for per_feature in (own, None):
        f_value, df1, df2, p_value = moderated_f_tests(
            t, np.eye(2), per_feature, np.inf, np.full(3, 5.0)
        )
        assert np.isnan(f_value).all() and np.isnan(p_value).all()
        assert df1 == 2 and np.isinf(df2).all()


def test_the_standardized_f_statistic_stays_exact_where_p_underflows() -> None:
    # F on (2, d) has the upper tail (1 + 2x / d)^(-d / 2), and F on (2, infinity)
    # exp(-x), so the standardized F-statistic is (d / 2) log(1 + 2x / d).
    f_value = np.array([3.0, 40.0, 1e9, 1e70, 740.0])
    df2 = np.array([10.0, 10.0, 100.0, 10.0, 1e6])
    p_value = np.asarray(f_sf(f_value, 2.0, df2))
    # 1e9 and 1e70 underflow to zero, and 740 on 10^6 df gives a subnormal p-value,
    # about 8e-322, whose few significant bits the standardized F-statistic must not
    # inherit.
    assert (p_value == 0.0).sum() == 2
    assert 0.0 < p_value[4] < np.finfo(np.float64).smallest_normal
    expected = df2 / 2.0 * np.log1p(2.0 * f_value / df2)
    np.testing.assert_allclose(
        standardized_f_statistics(f_value, 2, df2, p_value), expected, rtol=1e-12
    )


@pytest.mark.parametrize("df1", range(1, 13))
def test_on_infinite_df_the_standardized_f_statistic_is_the_f_statistic(
    df1: int,
) -> None:
    # Up to 1e300, where the p-values underflow, and around the subnormal ones.
    f_value = np.concatenate([np.geomspace(1e-3, 1e300, 61), [740.0, 744.0]])
    df2 = np.full(f_value.size, np.inf)
    p_value = np.asarray(f_sf(f_value, float(df1), df2))
    assert (p_value == 0.0).any()
    np.testing.assert_allclose(
        standardized_f_statistics(f_value, df1, df2, p_value), f_value, rtol=1e-12
    )


@pytest.mark.parametrize("df1", [1, 2, 3, 4, 8, 12])
def test_the_standardized_f_statistic_has_the_tail_probability_of_the_f_statistic(
    df1: int,
) -> None:
    # log P(F(df1, infinity) > standardized) = log P(F(df1, df2) > F), also where the
    # p-value underflows; prior degrees of freedom can be very large and finite.
    f_value = np.geomspace(1e-2, 1e300, 80)
    for df in (3.0, 30.0, 1e3, 1e6, 1e10, 1e12, 1e15):
        df2 = np.full(f_value.size, df)
        p_value = np.asarray(f_sf(f_value, float(df1), df2))
        standardized = standardized_f_statistics(f_value, df1, df2, p_value)
        target = np.asarray(f_logsf(f_value, float(df1), df2))
        found = np.asarray(
            f_logsf(standardized, float(df1), np.full(f_value.size, np.inf))
        )
        assert (p_value == 0.0).any()
        error = np.abs(found - target) / np.maximum(np.abs(target), 1.0)
        assert error.max() <= 1e-12, (df, f_value[np.argmax(error)], error.max())


def test_the_standardized_f_statistic_of_a_missing_or_infinite_f() -> None:
    f_value = np.array([np.nan, np.inf, 2.0, 2.0, np.nan])
    df2 = np.array([5.0, 5.0, 5.0, np.nan, np.inf])
    p_value = np.asarray(f_sf(f_value, 3.0, df2))
    standardized = standardized_f_statistics(f_value, 3, df2, p_value)
    assert np.isnan(standardized[[0, 3, 4]]).all() and standardized[1] == np.inf
    np.testing.assert_allclose(
        standardized[2], stats.chi2.isf(stats.f.sf(2.0, 3, 5), 3) / 3, rtol=1e-12
    )
