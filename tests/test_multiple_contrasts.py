"""Dunnett's and Williams-type contrasts and multiple contrast tests.

The references are multcomp's ``contrMat`` and ``summary(glht(parm(b, V, df)))``, saved
by ``validation/multiple_contrast_tests.py``, weights derived by hand, limma's moderated
t-test (for one contrast), and invariants of the single-step max-t test.
"""

import doctest
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from cytepool._jit import thread_limit
from cytepool.design import Contrasts
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.multiple_contrasts import (
    MultipleContrastTest,
    dunnett_contrasts,
    multiple_contrast_test,
    williams_contrasts,
)

FIXTURES = Path(__file__).parent / "data" / "multiple_contrast_tests"
CALIBRATED = 5.5


# ---------------------------------------------------------------------------------------
# Contrast matrices
# ---------------------------------------------------------------------------------------


def test_contrast_matrices_match_multcomp() -> None:
    cases = json.loads((FIXTURES / "contrasts.json").read_text())
    checked = {"Dunnett": 0, "Williams": 0}
    for case in cases:
        sizes = dict(zip(case["names"], case["sizes"]))
        if "error" in case:
            # multcomp refuses Williams' contrasts of two groups; here they are
            # Dunnett's one contrast.
            assert case["type"] == "Williams" and len(sizes) == 2
            two = williams_contrasts(sizes)
            np.testing.assert_array_equal(two.matrix[:, 0], [-1.0, 1.0])
            continue
        if case["type"] == "Dunnett":
            contrasts = dunnett_contrasts(sizes, control=case["control"])
            np.testing.assert_array_equal(contrasts.matrix.T, case["matrix"])
            assert list(contrasts.names) == case["contrast_names"]
        else:
            contrasts = williams_contrasts(sizes)
            np.testing.assert_allclose(
                contrasts.matrix.T, case["matrix"], rtol=0, atol=1e-15
            )
        assert contrasts.columns == tuple(case["names"])
        checked[case["type"]] += 1
    assert checked["Dunnett"] >= 20 and checked["Williams"] >= 14


def test_williams_weights_by_hand() -> None:
    contrasts = williams_contrasts({"vehicle": 5, "low": 4, "mid": 3, "high": 6})
    expected = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 3 / 9, 6 / 9],
            [-1.0, 4 / 13, 3 / 13, 6 / 13],
        ]
    )
    np.testing.assert_allclose(contrasts.matrix.T, expected, rtol=1e-15)
    assert contrasts.names == (
        "high - vehicle",
        "mid to high - vehicle",
        "low to high - vehicle",
    )
    np.testing.assert_allclose(contrasts.matrix.sum(axis=0), 0.0, atol=1e-15)


def test_dunnett_contrasts_take_groups_and_a_control() -> None:
    by_position = dunnett_contrasts(["a", "b", "c"], control=1)
    by_name = dunnett_contrasts({"a": 3, "b": 9, "c": 1}, control="b")
    from_end = dunnett_contrasts(("a", "b", "c"), control=-2)
    for contrasts in (by_name, from_end):
        np.testing.assert_array_equal(contrasts.matrix, by_position.matrix)
        assert contrasts.names == by_position.names == ("a - b", "c - b")
    np.testing.assert_array_equal(
        by_position.matrix, [[1.0, 0.0], [-1.0, -1.0], [0.0, 1.0]]
    )


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: dunnett_contrasts(["only"]), ValueError, "at least two groups"),
        (lambda: dunnett_contrasts("abc"), TypeError, "single string"),
        (lambda: dunnett_contrasts(["a", "a", "b"]), ValueError, "repeated: 'a'"),
        (lambda: dunnett_contrasts(["a", ""]), ValueError, "empty group name"),
        (lambda: dunnett_contrasts(["a", 2]), TypeError, "text"),
        (
            lambda: dunnett_contrasts(["a", "b"], control="c"),
            ValueError,
            "not one of the groups",
        ),
        (lambda: dunnett_contrasts(["a", "b"], control=2), ValueError, "out of range"),
        (lambda: dunnett_contrasts(["a", "b"], control=1.0), TypeError, "control"),
        (lambda: williams_contrasts({"v": 3}), ValueError, "at least two groups"),
        (lambda: williams_contrasts({"v": 3, "d": 0}), ValueError, "positive integer"),
        (lambda: williams_contrasts({"v": 3, "d": 2.5}), TypeError, "integer"),
        (lambda: williams_contrasts("34"), TypeError, "map each group name"),
        # Counts alone name no groups, which the estimates' names would have to match.
        (lambda: williams_contrasts([3, 3, 3]), TypeError, "map each group name"),
        # A set has no order, so the control, the first group, would be any of them.
        (
            lambda: dunnett_contrasts({"vehicle", "low", "high"}),
            TypeError,
            "a set has no order",
        ),
        (lambda: dunnett_contrasts(frozenset({"a", "b"})), TypeError, "no order"),
        (lambda: williams_contrasts({4, 3}), TypeError, "in dose order"),
        # The coefficients of a model with an intercept are differences already.
        (
            lambda: dunnett_contrasts(["(Intercept)", "dose_groupd1"]),
            ValueError,
            r"already their differences from the reference group.*'~ 0 \+ dose'",
        ),
        (
            lambda: williams_contrasts({"(Intercept)": 3, "d1": 3, "d2": 3}),
            ValueError,
            "intercept of a model",
        ),
    ],
)
def test_contrast_builders_refuse(call, error, message) -> None:
    with pytest.raises(error, match=message):
        call()


# ---------------------------------------------------------------------------------------
# Multiple contrast tests
# ---------------------------------------------------------------------------------------


def _multcomp() -> list[dict[str, np.ndarray]]:
    with np.load(FIXTURES / "multcomp.npz") as data:
        arrays = {key: data[key] for key in data.files}
    families = []
    for f in range(4):
        families.append(
            {
                key.split("__")[1]: value
                for key, value in arrays.items()
                if key.startswith(f"family{f}__")
            }
        )
    return families


def test_multiple_contrast_test_matches_multcomp() -> None:
    for f, family in enumerate(_multcomp()):
        for alternative in np.unique(family["alternative"]):
            chosen = np.flatnonzero(family["alternative"] == alternative)
            test = multiple_contrast_test(
                family["estimates"][chosen],
                family["covariance"][chosen],
                family["contrasts"],
                df=family["df"][chosen],
                alternative=str(alternative),
                relative_error=1e-3,
                rng=f,
            )
            for name in ("estimate", "standard_error", "t"):
                reference = family[name][chosen]
                difference = np.abs(getattr(test, name) - reference) / np.maximum(
                    np.abs(reference), 1.0
                )
                assert difference.max() <= 1e-10, name
            # multcomp's default absolute error of 1e-3 cannot resolve smaller
            # p-values, which mvtnorm computes as 1 minus the inside of the box.
            reference = family["p_value"][chosen]
            resolved = reference >= 1e-3
            combined = np.hypot(
                test.p_value_error, family["p_value_error"][chosen, None] / 3.0
            )
            assert resolved.mean() > 0.5
            assert np.all(
                np.abs(test.p_value - reference)[resolved]
                <= CALIBRATED * combined[resolved]
            )
            # Bonferroni's adjustment, the union bound, is never below the
            # single-step p-value, at any magnitude; scipy gives the t tails, as
            # multcomp's own Bonferroni p-values are 0 beyond |t| of about 9.
            df = family["df"][chosen, None]
            if alternative == "two-sided":
                tails = 2.0 * stats.t.sf(np.abs(test.t), df)
            else:
                tails = stats.t.sf(test.t if alternative == "greater" else -test.t, df)
            union = np.minimum(1.0, test.t.shape[1] * tails)
            assert np.all(union >= test.p_value - CALIBRATED * test.p_value_error)


def test_one_contrast_is_the_moderated_t_test() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 1.0, (8, 30))
    y[4:, :5] += 2.0
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    moderated = moderated_statistics(fit_linear_model(y, design))
    covariances = (
        moderated.posterior_variance[:, None, None] * moderated.fit.cov_coefficients
    )
    test = multiple_contrast_test(
        moderated.fit.coefficients, covariances, [0.0, 1.0], df=moderated.df_total
    )
    np.testing.assert_allclose(test.t[:, 0], moderated.t[:, 1], rtol=1e-12)
    np.testing.assert_allclose(test.p_value[:, 0], moderated.p_value[:, 1], rtol=1e-12)
    assert (test.p_value_error == 0.0).all()


def _groups(
    n_features: int = 6, seed: int = 2
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    names = ["vehicle", "low", "mid", "high"]
    sizes = np.array([4.0, 3.0, 3.0, 4.0])
    variances = rng.uniform(0.05, 0.3, (n_features, 1)) / sizes
    means = rng.normal(5.0, 0.3, (n_features, 4))
    means[:, 3] += np.linspace(0.0, 2.0, n_features)
    covariances = np.zeros((n_features, 4, 4))
    covariances[:, np.arange(4), np.arange(4)] = variances
    return means, covariances, names


def test_contrasts_are_matched_to_estimates_by_name() -> None:
    means, covariances, names = _groups()
    contrasts = dunnett_contrasts(names)
    named = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names, rng=1
    )
    # The same contrasts over the groups in another order give the same tests.
    order = [2, 0, 3, 1]
    reordered = Contrasts(
        contrasts.matrix[order], contrasts.names, tuple(names[j] for j in order)
    )
    shuffled = multiple_contrast_test(
        means, covariances, reordered, df=9.0, estimate_names=names, rng=1
    )
    np.testing.assert_array_equal(named.p_value, shuffled.p_value)
    plain = multiple_contrast_test(means, covariances, contrasts.matrix, df=9.0, rng=1)
    np.testing.assert_array_equal(named.p_value, plain.p_value)
    assert plain.names == ("contrast1", "contrast2", "contrast3")
    np.testing.assert_array_equal(named.contrasts, contrasts.matrix)
    assert named.estimate_names == tuple(names)
    with pytest.raises(
        ValueError,
        match="'vehicle', 'low', 'mid' and 'high' are not among them; name the "
        "estimates with estimate_names, in the order of their columns",
    ):
        multiple_contrast_test(means, covariances, contrasts, df=9.0)
    with pytest.raises(
        ValueError,
        match="'middle' is missing and 'mid' .did you mean 'middle'.. is not among",
    ):
        multiple_contrast_test(
            means,
            covariances,
            contrasts,
            df=9.0,
            estimate_names=["vehicle", "low", "middle", "high"],
        )


def test_mismatched_names_are_listed() -> None:
    means, covariances, names = _groups()
    three = dunnett_contrasts(names[:3])
    with pytest.raises(
        ValueError, match=r"the estimates \('vehicle', .*'high' is miss"
    ):
        multiple_contrast_test(means, covariances, three, df=9.0, estimate_names=names)
    five = dunnett_contrasts([*names, "extra"])
    with pytest.raises(ValueError, match=r"but 'extra' is not among them; name the"):
        multiple_contrast_test(means, covariances, five, df=9.0, estimate_names=names)
    # Weights in the order of the estimates, of the wrong length or orientation.
    with pytest.raises(ValueError, match=r"one row for each of the estimates \(4\)"):
        multiple_contrast_test(means, covariances, np.ones((3, 4)), df=9.0)
    with pytest.raises(ValueError, match="one value for each of the estimates"):
        multiple_contrast_test(means, covariances, {"x": [1.0, -1.0]}, df=9.0)


def test_pandas_weights_are_matched_by_their_index() -> None:
    import pandas as pd

    means, covariances, names = _groups()
    contrasts = dunnett_contrasts(names)
    reference = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names, rng=1
    )
    order = [3, 1, 0, 2]
    frame = pd.DataFrame(
        np.asarray(contrasts.matrix)[order],
        index=[names[j] for j in order],
        columns=list(contrasts.names),
    )
    by_name = multiple_contrast_test(
        means, covariances, frame, df=9.0, estimate_names=names, rng=1
    )
    np.testing.assert_array_equal(by_name.p_value, reference.p_value)
    assert by_name.names == contrasts.names
    # Without names in the index, the rows are read in the order of the estimates.
    in_order = pd.DataFrame(np.asarray(contrasts.matrix), columns=list(contrasts.names))
    plain = multiple_contrast_test(means, covariances, in_order, df=9.0, rng=1)
    np.testing.assert_array_equal(plain.p_value, reference.p_value)
    # An index that is neither names nor pandas' default positions is refused.
    shifted = pd.DataFrame({"x": [-1.0, 0.0, 0.0, 1.0]}, index=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="must hold the names of the estimates"):
        multiple_contrast_test(means, covariances, shifted, df=9.0)


def test_without_contrasts_the_estimates_are_tested() -> None:
    means, covariances, names = _groups()
    differences = means[:, 1:] - means[:, :1]
    variance = covariances[:, 1:, 1:] + covariances[:, :1, :1]
    direct = multiple_contrast_test(
        differences, variance, df=9.0, estimate_names=names[1:], rng=3
    )
    assert direct.names == tuple(names[1:])
    np.testing.assert_allclose(direct.contrasts, np.eye(3))
    via = multiple_contrast_test(
        means,
        covariances,
        dunnett_contrasts(names),
        df=9.0,
        estimate_names=names,
        rng=3,
    )
    np.testing.assert_allclose(direct.t, via.t, rtol=1e-12)
    np.testing.assert_allclose(direct.p_value, via.p_value, rtol=1e-12)
    assert multiple_contrast_test(differences, variance, df=9.0).names[0] == "estimate1"


def test_less_at_t_is_greater_at_minus_t() -> None:
    means, covariances, names = _groups()
    contrasts = williams_contrasts(dict(zip(names, (4, 3, 3, 4))))
    greater = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names,
        alternative="greater", rng=5,
    )  # fmt: skip
    less = multiple_contrast_test(
        -means, covariances, contrasts, df=9.0, estimate_names=names,
        alternative="less", rng=5,
    )  # fmt: skip
    np.testing.assert_array_equal(greater.p_value, less.p_value)
    np.testing.assert_array_equal(greater.t, -less.t)


def test_p_values_never_decrease_as_t_becomes_less_extreme() -> None:
    means, covariances, names = _groups(n_features=40)
    test = multiple_contrast_test(
        means,
        covariances,
        dunnett_contrasts(names),
        df=6.0,
        estimate_names=names,
        rng=2,
    )
    for t, p in zip(np.abs(test.t), test.p_value):
        order = np.argsort(-t, kind="stable")
        assert np.all(np.diff(p[order]) >= 0.0)
    # And each lies between the unadjusted p-value and Bonferroni's.
    from cytepool.stats.distributions import t_sf

    unadjusted = 2.0 * np.asarray(t_sf(np.abs(test.t), 6.0))
    slack = CALIBRATED * test.p_value_error
    assert np.all(test.p_value >= unadjusted - slack)
    assert np.all(test.p_value <= np.minimum(1.0, 3.0 * unadjusted) + slack)


def test_a_less_extreme_contrast_never_gets_a_smaller_p_value() -> None:
    """Monte Carlo noise can put the estimate of a less extreme contrast below that of a
    more extreme one, which the data above rarely show; the running maximum raises it
    to that p-value, with that p-value's error."""
    from cytepool.stats.multiple_contrasts import _monotone

    p_value = np.array([[0.010, 0.009, 0.5], [0.2, 0.3, 0.1]])
    error = np.array([[1e-4, 2e-4, 3e-3], [1e-3, 2e-3, 5e-4]])
    extremeness = np.array([[2.0, 1.9, 0.3], [1.0, 0.5, 2.0]])
    raised, raised_error = _monotone(p_value, error, extremeness)
    np.testing.assert_array_equal(raised, [[0.010, 0.010, 0.5], [0.2, 0.3, 0.1]])
    np.testing.assert_array_equal(
        raised_error, [[1e-4, 1e-4, 3e-3], [1e-3, 2e-3, 5e-4]]
    )


def test_two_contrasts_need_no_random_numbers() -> None:
    means, covariances, names = _groups()
    contrasts = dunnett_contrasts(names[:3])
    first = multiple_contrast_test(
        means[:, :3],
        covariances[:, :3, :3],
        contrasts,
        df=7.5,
        estimate_names=names[:3],
        rng=1,
    )
    second = multiple_contrast_test(
        means[:, :3],
        covariances[:, :3, :3],
        contrasts,
        df=7.5,
        estimate_names=names[:3],
        rng=2,
    )
    np.testing.assert_array_equal(first.p_value, second.p_value)
    assert np.all(
        first.p_value_error <= 1e-11 * np.minimum(first.p_value, 1 - first.p_value)
    )


def test_missing_and_degenerate_features() -> None:
    means, covariances, names = _groups()
    contrasts = dunnett_contrasts(names)
    clean = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names, rng=4
    )
    means[1, 2] = np.nan
    covariances[2, 0, 0] = np.inf
    covariances[3] = 0.0  # every contrast of zero variance
    dirty = multiple_contrast_test(
        means, covariances, contrasts, df=np.array([9.0, 9.0, 9.0, 9.0, np.nan, 9.0]),
        estimate_names=names, rng=4,
    )  # fmt: skip
    for row in (1, 2, 3, 4):
        for name in ("estimate", "standard_error", "t", "p_value", "p_value_error"):
            assert np.isnan(getattr(dirty, name)[row]).all(), (row, name)
    np.testing.assert_array_equal(dirty.p_value[[0, 5]], clean.p_value[[0, 5]])
    table = dirty.table()
    assert table["contrast"].tolist()[1] is None
    assert np.isnan(table["p_value"][1])
    # With no complete feature, every result is NaN.
    means[:] = np.nan
    empty = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names
    )
    assert np.isnan(empty.p_value).all() and np.isnan(empty.t).all()


def test_covariances_that_are_not_positive_semidefinite_are_refused() -> None:
    means, covariances, names = _groups()
    covariances[4] = np.array(
        [
            [1.0, 0.9, 0.9, -0.9],
            [0.9, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.9],
            [-0.9, 0.9, 0.9, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="covariance of feature 4 is not positive"):
        multiple_contrast_test(
            means, covariances, dunnett_contrasts(names), df=9.0, estimate_names=names
        )


def test_the_tables() -> None:
    means = np.array(
        [
            [0.0, 0.1, 0.2, 3.0],
            [0.0, -2.5, 0.1, 0.3],
            [0.0, 0.2, 0.1, 0.1],
            [0.0, 1.0, 1.1, 1.2],
        ]
    )
    covariances = np.diag([0.1, 0.1, 0.1, 0.1])
    names = ["vehicle", "low", "mid", "high"]
    test = multiple_contrast_test(
        means,
        covariances,
        dunnett_contrasts(names),
        df=12.0,
        estimate_names=names,
        rng=1,
    )
    overall = test.table(feature_ids=["g1", "g2", "g3", "g4"])
    assert overall.columns == (
        "feature_id", "contrast", "estimate", "standard_error", "t", "p_value",
        "p_value_error", "adjusted_p_value",
    )  # fmt: skip
    assert overall["feature_id"].tolist() == ["g1", "g2", "g3", "g4"]
    assert overall["contrast"].tolist() == [
        "high - vehicle", "low - vehicle", "low - vehicle", "high - vehicle"
    ]  # fmt: skip
    np.testing.assert_array_equal(overall["p_value"], test.p_value.min(axis=1))
    # The chosen contrasts' estimates are the mean differences, with standard errors
    # sqrt(0.1 + 0.1).
    np.testing.assert_allclose(overall["estimate"], [3.0, -2.5, 0.2, 1.2], rtol=1e-12)
    np.testing.assert_allclose(overall["standard_error"], np.sqrt(0.2), rtol=1e-12)
    # Benjamini and Hochberg's adjustment, by hand: p_(i) * n / i, then the running
    # minimum from the largest.
    p = np.asarray(overall["p_value"])
    order = np.argsort(p)
    scaled = p[order] * 4 / np.arange(1, 5)
    expected = np.empty(4)
    expected[order] = np.minimum.accumulate(scaled[::-1])[::-1]
    np.testing.assert_allclose(
        overall["adjusted_p_value"], np.minimum(expected, 1.0), rtol=1e-15
    )
    one = test.table("mid - vehicle")
    assert one.columns == (
        "feature_id",
        "estimate",
        "standard_error",
        "t",
        "p_value",
        "p_value_error",
    )
    np.testing.assert_array_equal(one["estimate"], means[:, 2])
    np.testing.assert_array_equal(one["p_value"], test.table(1)["p_value"])
    with pytest.raises(KeyError, match="no coefficient or contrast 'top'"):
        test.table("top")
    with pytest.raises(IndexError, match="out of range for 3 coefficients or contra"):
        test.table(5)
    with pytest.raises(ValueError, match="one id per feature"):
        test.table(feature_ids=["g1"])
    text = repr(test)
    assert text.startswith("MultipleContrastTest(n_features=4, ")
    assert "adjusted_p_value_below_0.05=2" in text
    assert "p_values_short_of_relative_error=0" in text
    assert isinstance(test, MultipleContrastTest)


def test_repr_counts_p_values_short_of_their_target(monkeypatch) -> None:
    from cytepool.stats import multivariate_t

    monkeypatch.setattr(multivariate_t, "_MOST_POINTS", 4)
    means, covariances, names = _groups()
    test = multiple_contrast_test(
        means, covariances, dunnett_contrasts(names), df=9.0, estimate_names=names,
        relative_error=1e-6, rng=1,
    )  # fmt: skip
    assert f"p_values_short_of_relative_error={test.p_value.size}" in repr(test)


def test_results_do_not_depend_on_threads() -> None:
    means, covariances, names = _groups(n_features=20)
    contrasts = dunnett_contrasts(names)
    with thread_limit(1):
        one = multiple_contrast_test(
            means, covariances, contrasts, df=9.0, estimate_names=names, rng=8
        )
    many = multiple_contrast_test(
        means, covariances, contrasts, df=9.0, estimate_names=names, rng=8, n_threads=4
    )
    np.testing.assert_array_equal(one.p_value, many.p_value)
    np.testing.assert_array_equal(one.p_value_error, many.p_value_error)


@pytest.mark.parametrize("weighted", [False, True])
def test_a_moderated_fit_is_tested_with_its_own_covariances(weighted) -> None:
    """A cell-means fit, with weights that differ by gene (one covariance per gene)
    or without (one for every gene): its Dunnett contrasts give the moderated t of
    contrasts_fit and moderated_statistics, and a fit of the contrasts tested as it
    is gives the same tests."""
    rng = np.random.default_rng(5)
    groups = np.repeat([0, 1, 2, 3], 3)
    design = np.eye(4)[groups]
    y = rng.normal(5.0, 1.0, (12, 25))
    y[groups == 3, :5] += 4.0
    weights = rng.uniform(0.5, 2.0, y.shape) if weighted else None
    names = ["vehicle", "low", "mid", "high"]
    fit = fit_linear_model(y, design, weights=weights, coefficient_names=names)
    assert (fit.cov_coefficients_per_feature is not None) == weighted
    moderated = moderated_statistics(fit)
    contrasts = dunnett_contrasts(fit.coefficient_names)
    test = multiple_contrast_test(moderated, contrasts=contrasts, rng=1)
    reference = moderated_statistics(contrasts_fit(fit, contrasts))
    np.testing.assert_allclose(test.t, reference.t, rtol=1e-12)
    np.testing.assert_array_equal(test.df, moderated.df_total)
    as_fitted = multiple_contrast_test(reference, rng=1)
    assert as_fitted.names == contrasts.names
    np.testing.assert_allclose(as_fitted.t, test.t, rtol=1e-12)
    np.testing.assert_allclose(as_fitted.p_value, test.p_value, rtol=1e-12)
    assert (test.table()["p_value"][:5] < 0.01).all()
    # One contrast of the fit is its moderated t-test.
    one = multiple_contrast_test(moderated, contrasts={"high": [0.0, 0.0, 0.0, 1.0]})
    np.testing.assert_allclose(one.p_value[:, 0], moderated.p_value[:, 3], rtol=1e-12)


def test_a_fit_without_exact_standard_errors_and_missing_arguments_are_refused() -> (
    None
):
    rng = np.random.default_rng(6)
    design = np.eye(3)[np.repeat([0, 1, 2], 3)]
    y = rng.normal(5.0, 1.0, (9, 12))
    names = ["vehicle", "low", "high"]
    fit = fit_linear_model(
        y, design, weights=rng.uniform(0.5, 2.0, y.shape), coefficient_names=names
    )
    approximate = contrasts_fit(fit, dunnett_contrasts(names), method="limma")
    assert not approximate.exact_standard_errors
    with pytest.raises(ValueError, match="limma's approximate standard errors"):
        multiple_contrast_test(moderated_statistics(approximate))
    moderated = moderated_statistics(fit)
    with pytest.raises(ValueError, match="so leave out covariances and df; pass the"):
        multiple_contrast_test(moderated, np.eye(3), df=4.0)
    with pytest.raises(ValueError, match="so leave out estimate_names"):
        multiple_contrast_test(moderated, estimate_names=names)
    # Contrasts of other names point to the fit's own coefficient names.
    with pytest.raises(
        ValueError,
        match=r"exactly the fit's coefficients \('vehicle', 'low' and 'high'\).*"
        r"fit\.coefficient_names",
    ):
        multiple_contrast_test(moderated, contrasts=dunnett_contrasts(["v", "l", "h"]))
    with pytest.raises(TypeError, match="needs covariances and df with estimates"):
        multiple_contrast_test(fit.coefficients)
    with pytest.raises(TypeError, match="needs df with estimates"):
        multiple_contrast_test(fit.coefficients, np.eye(3))


def test_the_correlation_of_dunnett_contrasts_and_simultaneous_intervals() -> None:
    """Dunnett's contrasts of independent group means with variances s^2 / n_i are
    correlated by lambda_i lambda_j, lambda_i = sqrt(n_i / (n_i + n_0)) (Dunnett
    1955); the stored correlation gives simultaneous intervals, which scipy's
    dunnett computes from the samples themselves."""
    rng = np.random.default_rng(12)
    sizes = np.array([6, 3, 4, 5])
    samples = [rng.normal(mean, 1.0, n) for mean, n in zip([0, 0.4, 1.0, 2.0], sizes)]
    means = np.array([[sample.mean() for sample in samples]])
    residual_df = int(sizes.sum() - sizes.size)
    pooled = sum(((s - s.mean()) ** 2).sum() for s in samples) / residual_df
    names = ["vehicle", "low", "mid", "high"]
    test = multiple_contrast_test(
        means, pooled * np.diag(1.0 / sizes), dunnett_contrasts(names),
        df=residual_df, estimate_names=names, relative_error=1e-3, rng=3,
    )  # fmt: skip
    lam = np.sqrt(sizes[1:] / (sizes[1:] + sizes[0]))
    expected = np.outer(lam, lam)
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(test.correlation[0], expected, rtol=1e-12)
    from cytepool.stats.multivariate_t import multivariate_t_quantile

    critical = multivariate_t_quantile(
        0.95, test.correlation, test.df, two_sided=True, relative_error=1e-3, rng=4
    )
    half_width = critical.quantile[:, None] * test.standard_error
    reference = stats.dunnett(*samples[1:], control=samples[0], rng=5)
    interval = reference.confidence_interval(confidence_level=0.95)
    # scipy integrates by randomized quasi-Monte Carlo too: its interval ends move by
    # up to 1.4e-3 between seeds.
    np.testing.assert_allclose(
        test.estimate[0] - half_width[0], interval.low, rtol=0, atol=3e-3
    )
    np.testing.assert_allclose(
        test.estimate[0] + half_width[0], interval.high, rtol=0, atol=3e-3
    )
    np.testing.assert_allclose(test.p_value[0], reference.pvalue, rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"df": np.ones((6, 3))}, ValueError, "one df per feature"),
        ({"df": np.ones(5)}, ValueError, r"one per feature \(6\)"),
        ({"df": 0.0}, ValueError, "df must be positive"),
        ({"df": True}, TypeError, "bool"),
        ({"df": 9.0, "alternative": "both"}, ValueError, "alternative"),
        ({"df": 9.0, "relative_error": 2.0}, ValueError, "relative_error"),
    ],
)
def test_arguments_are_refused(options, error, message) -> None:
    means, covariances, names = _groups()
    with pytest.raises(error, match=message):
        multiple_contrast_test(
            means,
            covariances,
            dunnett_contrasts(names),
            estimate_names=names,
            **options,
        )


def test_arrays_of_text_are_refused() -> None:
    means, covariances, names = _groups()
    with pytest.raises(TypeError, match="estimates must hold real numbers"):
        multiple_contrast_test(means.astype(str), covariances, df=9.0)
    with pytest.raises(TypeError, match="covariances must hold real numbers"):
        multiple_contrast_test(means, covariances.astype(str), df=9.0)
    with pytest.raises(TypeError, match="df must hold real numbers"):
        multiple_contrast_test(means, covariances, df="nine")


def test_shapes_and_zero_contrasts_are_refused() -> None:
    means, covariances, names = _groups()
    with pytest.raises(ValueError, match="covariances must have shape"):
        multiple_contrast_test(means, covariances[:, :3, :3], df=9.0)
    with pytest.raises(ValueError, match="estimates must have shape"):
        multiple_contrast_test(np.ones((2, 2, 2)), covariances, df=9.0)
    with pytest.raises(ValueError, match="'nothing' has only zero weights"):
        multiple_contrast_test(
            means, covariances, {"nothing": [0.0, 0.0, 0.0, 0.0]}, df=9.0
        )
    single = multiple_contrast_test(
        means[0], covariances[0], [-1.0, 0.0, 0.0, 1.0], df=9.0
    )
    assert single.p_value.shape == (1, 1)
    assert math.isfinite(single.p_value[0, 0])


def test_the_examples_run() -> None:
    from cytepool.stats import multiple_contrasts

    assert doctest.testmod(multiple_contrasts, raise_on_error=False).failed == 0


def test_the_overall_table_of_a_one_sided_test() -> None:
    """The feature's contrast is the largest t for "greater" and the smallest for
    "less", even when another contrast has a larger |t|, and its p-value is the
    smallest of the feature's."""
    means = np.array([[0.0, -3.0, 0.5, 1.0], [0.0, 0.2, -2.0, 0.4]])
    names = ["vehicle", "low", "mid", "high"]
    expected = {
        "greater": ["high - vehicle", "high - vehicle"],
        "less": ["low - vehicle", "mid - vehicle"],
    }
    for alternative, chosen in expected.items():
        test = multiple_contrast_test(
            means, np.diag([0.1] * 4), dunnett_contrasts(names), df=10.0,
            estimate_names=names, alternative=alternative, rng=1,
        )  # fmt: skip
        table = test.table()
        assert table["contrast"].tolist() == chosen
        np.testing.assert_array_equal(table["p_value"], test.p_value.min(axis=1))


def test_a_contrast_of_zero_variance_makes_its_feature_missing() -> None:
    """One contrast of zero variance among contrasts of positive variance: the
    vehicle and the low dose are known exactly."""
    names = ["vehicle", "low", "mid", "high"]
    test = multiple_contrast_test(
        np.array([5.0, 5.0, 6.0, 7.0]), np.diag([0.0, 0.0, 0.04, 0.04]),
        dunnett_contrasts(names), df=9.0, estimate_names=names, rng=1,
    )  # fmt: skip
    for name in ("estimate", "standard_error", "t", "p_value", "p_value_error"):
        assert np.isnan(getattr(test, name)).all(), name
    assert np.isnan(test.correlation).all()


def test_a_negative_variance_is_refused() -> None:
    names = ["vehicle", "low", "mid", "high"]
    with pytest.raises(
        ValueError,
        match="feature 0 is not positive semidefinite: it gives contrast 'low - "
        "vehicle' the variance -0.4",
    ):
        multiple_contrast_test(
            np.zeros(4), np.diag([0.1, -0.5, 0.1, 0.1]), dunnett_contrasts(names),
            df=9.0, estimate_names=names,
        )  # fmt: skip
    # Rounding below 0 in a contrast of zero variance leaves its feature untested.
    covariance = np.full((2, 2), 0.3)
    covariance[0, 1] = covariance[1, 0] = 0.3 * (1 + 4e-16)
    rounded = multiple_contrast_test(np.zeros(2), covariance, [-1.0, 1.0], df=9.0)
    assert np.isnan(rounded.p_value).all()


def test_asymmetric_covariances_are_refused() -> None:
    means, covariances, names = _groups()
    covariances[2, 0, 1] = 0.01
    with pytest.raises(
        ValueError, match="covariances must be symmetric.*the matrix of feature 2"
    ):
        multiple_contrast_test(
            means, covariances, dunnett_contrasts(names), df=9.0, estimate_names=names
        )
    one = np.diag([0.1, 0.1, 0.1, 0.1])
    one[3, 0] = 0.02
    with pytest.raises(ValueError, match="symmetric.*the matrix differs"):
        multiple_contrast_test(means, one, df=9.0)
    # Rounding of a product of symmetric matrices is accepted.
    rounded = one.copy()
    rounded[3, 0] = 0.0
    rounded[0, 1], rounded[1, 0] = 0.01, 0.01 * (1 + 1e-15)
    assert np.isfinite(multiple_contrast_test(means, rounded, df=9.0).p_value).all()
