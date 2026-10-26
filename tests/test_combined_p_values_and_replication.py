"""Tests for combined p-values, partial conjunctions, pi0 and replication across studies.

Reference values in ``tests/data/combined_p_values_and_replication/`` come from
``validation/combined_p_values_and_replication.py``: ``combined.npz`` holds p-values
of five studies with scipy's ``combine_pvalues`` and metap's outputs for them, the
partial conjunction p-values from their definition, and mpmath's values (400 digits)
for features of extreme p-values; ``pi0.npz`` holds sets of p-values with qvalue's
``pi0est`` for every option; ``replication.npz`` holds the pseudobulk counts of three
small cohorts and what R (metap, qvalue, metafor, ``p.adjust``) computes from their
``limma_voom`` tables. The other tests compare with closed forms derived by hand (two
studies, identical studies, opposite signs, a p-value of 1, one study, the count of
p-values at least lambda) and check what a wrong implementation would break: the order
of the studies and of the features, the directions turned around, weights scaled
together, the monotonicity of Simes's partial conjunction in u, and how the public
functions fit together in a replication.
"""

import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.cube import Cube, CubeCollection
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.meta_analysis import align_study_results, meta_analysis
from cytepool.replication import (
    ReplicationResult,
    differential_expression_replication,
)
from cytepool.stats.combined_p_values import (
    CombinedPValues,
    combine_p_values,
    partial_conjunction,
)
from cytepool.stats.multiple_testing import (
    TrueNullProportion,
    proportion_of_true_nulls,
)

DATA = Path(__file__).parent / "data" / "combined_p_values_and_replication"
EXACT = 1e-10
CLOSE = 1e-6  # REML is iterative; metafor converged to 1e-13 on tau^2
SCIPY = ("fisher", "pearson", "tippett", "stouffer", "mudholkar_george")
# metap's methods, in the order of the fixture, as CytePool's (rank for Wilkinson's).
METAP = (
    ("fisher", None),
    ("stouffer", None),
    ("tippett", None),
    ("wilkinson", 2),
    ("wilkinson", "k"),  # metap's maximump
    ("mudholkar_george", None),
    ("simes", None),
)
EXTREME = ("fisher", "pearson", "stouffer", "mudholkar_george", "tippett")
EXTREME += ("wilkinson", "simes")
PI0_OPTIONS = {
    "defaults": {},
    "smooth.df = 5": {"smooth_df": 5},
    "smooth.log.pi0": {"smooth_log": True},
    "bootstrap": {"method": "bootstrap"},
    "lambda = 0.5": {"lambdas": 0.5},
    "lambda = 0.1, ..., 0.9": {"lambdas": [0.1 * j for j in range(1, 10)]},
    "lambda unsorted": {"lambdas": [0.5, 0.2, 0.8, 0.35, 0.65]},
    "lambda = 0, ..., 0.9, bootstrap": {
        "lambdas": [0.05 * j for j in range(19)],
        "method": "bootstrap",
    },
}
# The settings of the replications of the fixture, as the validation script runs them.
SETTINGS = {
    "T0": {"method": "stouffer", "conjunction_method": "simes", "at_least": 2},
    "T1": {"method": "fisher", "conjunction_method": "fisher", "at_least": 3},
}
SETTINGS["T0"] |= {"alpha": 0.05, "meta_method": "reml", "null_proportion": {}}
SETTINGS["T1"] |= {"alpha": 0.25, "meta_method": "dl"}
SETTINGS["T1"]["null_proportion"] = {"method": "bootstrap"}
# Phi-bar(2) and Phi-bar(4), the upper tails of the standard normal distribution.
TAIL_2 = 0.022750131948179207
TAIL_4 = 3.1671241833119863e-05


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def scaled_difference(ours: Any, reference: Any, p_value: bool = False) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|); equal values, NaN included, differ by 0."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    same = (ours == reference) | (np.isnan(ours) & np.isnan(reference))
    with np.errstate(invalid="ignore", divide="ignore"):
        if p_value:
            scale = np.maximum(reference, 1e-300)
            scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
        else:
            scale = np.maximum(np.abs(reference), 1.0)
        scaled = np.where(same, 0.0, np.abs(ours - reference) / scale)
    return float(np.max(np.where(np.isnan(scaled), np.inf, scaled), initial=0.0))


def combined(p: Any, method: str, rank: Any = None, directions: Any = None) -> Any:
    """CytePool's (p-value, statistic) of every feature; rank "k" is Wilkinson's
    r = k, metap's maximump, for the features of two studies or more."""
    data = fixture("combined")
    p = np.asarray(p)
    weights = data["weights"] if method == "stouffer" else None
    if rank == "k":
        k = np.count_nonzero(~np.isnan(p), axis=0)
        values = np.full((2, p.shape[1]), np.nan)
        for size in np.unique(k[k >= 2]):
            cut = k == size
            signs = None if directions is None else directions[:, cut]
            values[:, cut] = combined(p[:, cut], method, int(size), signs)
        return values
    r = combine_p_values(
        p, method=method, weights=weights, directions=directions, rank=rank
    )
    return np.array([r.p_value, r.statistic])


# Against scipy, metap and mpmath.


@pytest.mark.parametrize("method", SCIPY)
def test_combinations_match_scipy(method: str) -> None:
    data = fixture("combined")
    p_value, statistic = combined(data["p"], method)
    k = np.count_nonzero(~np.isnan(data["p"]), axis=0)
    if method == "pearson":  # scipy reports -X
        statistic = -statistic
    if method == "mudholkar_george":  # and the sum of the logits, before scaling
        with np.errstate(divide="ignore", invalid="ignore"):  # features of no study
            statistic = statistic * np.pi / np.sqrt(3 / k * (5 * k + 4) / (5 * k + 2))
    theirs = data[f"scipy/{method}"]

    assert scaled_difference(p_value, theirs[1], p_value=True) <= EXACT
    assert scaled_difference(statistic, theirs[0]) <= EXACT


def test_combinations_match_metap_with_and_without_directions() -> None:
    data = fixture("combined")
    p, signs, theirs = data["p"], data["signs"], data["metap"]
    several = np.count_nonzero(~np.isnan(p), axis=0) >= 2
    # metap's two2one rounds 1 - p / 2, which loses digits for p below about 1e-5.
    large = several & (np.nanmin(np.where(np.isnan(p), 1.0, p), axis=0) >= 1e-5)
    assert large.sum() >= 10
    for m, (method, rank) in enumerate(METAP):
        p_value, statistic = combined(p, method, rank)
        signed = combined(p, method, rank, directions=signs)[0]

        assert scaled_difference(statistic[several], theirs[3 * m][several]) <= EXACT
        p_values = (p_value[several], theirs[3 * m + 1][several])
        assert scaled_difference(*p_values, p_value=True) <= EXACT
        signed_values = (signed[large], theirs[3 * m + 2][large])
        assert scaled_difference(*signed_values, p_value=True) <= EXACT


@pytest.mark.parametrize("u", [1, 2, 3])
@pytest.mark.parametrize("method", ["simes", "fisher", "stouffer"])
def test_partial_conjunctions_match_their_definition(u: int, method: str) -> None:
    data = fixture("combined")
    p, signs = data["p"], data["signs"]
    row = 3 * len(METAP) + 2 * (3 * (u - 1) + ("simes", "fisher", "stouffer").index(method))  # fmt: skip
    several = np.count_nonzero(~np.isnan(p), axis=0) >= 1
    large = several & (np.nanmin(np.where(np.isnan(p), 1.0, p), axis=0) >= 1e-5)
    plain = partial_conjunction(p, u, method=method)
    signed = partial_conjunction(p, u, method=method, directions=signs)

    assert scaled_difference(plain.p_value, data["metap"][row], p_value=True) <= EXACT
    reference = data["metap"][row + 1][large]
    assert scaled_difference(signed.p_value[large], reference, p_value=True) <= EXACT
    # A feature that fewer than u studies tested has no p-value.
    assert np.isnan(plain.p_value[plain.n_studies < u]).all()


@pytest.mark.parametrize("method", EXTREME)
def test_extreme_p_values_match_mpmath(method: str) -> None:
    data = fixture("combined")
    p, signs, weights = (data[f"extreme/{part}"] for part in ("p", "signs", "weights"))
    theirs = data[f"extreme/{method}"]
    rank = 2 if method == "wilkinson" else None
    w = weights if method == "stouffer" else None
    plain = combine_p_values(p, method=method, weights=w, rank=rank)
    signed = combine_p_values(p, method=method, weights=w, rank=rank, directions=signs)
    # Half of a subnormal p-value, below 2.2e-308, loses digits.
    normal = np.nanmin(p, axis=0) >= np.finfo(np.float64).tiny

    assert scaled_difference(plain.p_value, theirs[0], p_value=True) <= EXACT
    assert scaled_difference(plain.statistic, theirs[1]) <= EXACT
    signed_values = (signed.p_value[normal], theirs[2][normal])
    assert scaled_difference(*signed_values, p_value=True) <= EXACT
    assert scaled_difference(signed.statistic[normal], theirs[3][normal]) <= EXACT


def test_p_values_near_1e_300_keep_their_digits() -> None:
    # Tippett's 1 - (1 - p)^k would round to 0 here.
    tiny = np.array([1e-300, 0.5, 0.3])
    assert_allclose(
        combine_p_values(tiny, method="tippett").p_value, 3e-300, rtol=1e-14
    )
    # Fisher's for two studies, q (1 - log q) for the product q of the p-values.
    q = 1e-300 * 0.5
    fisher = combine_p_values(np.array([1e-300, 0.5]), method="fisher").p_value
    assert_allclose(fisher, q * (1.0 - np.log(q)), rtol=1e-13)
    # A p-value below the smallest double is 0, not a NaN.
    assert float(combine_p_values(np.full(3, 1e-300), method="stouffer").p_value) == 0.0
    # Two-sided p-values of 1e-300 halve exactly; opposite directions cancel.
    signed = combine_p_values(
        np.array([1e-300, 1e-300]), method="stouffer", directions=[1.0, -1.0]
    )
    assert float(signed.p_value) == 1.0
    assert float(signed.direction) == 0.0
    one = combine_p_values(np.array([1e-300]), method="stouffer", directions=[-3.0])
    # Through z = 37, p comes back within |log p| = 690 times the rounding of z.
    assert_allclose(one.p_value, 1e-300, rtol=1e-11)
    assert float(one.direction) == -1.0


# Values derived by hand.


def test_two_studies_give_the_closed_forms() -> None:
    p1, p2 = 0.01, 0.2
    p = np.array([p1, p2])
    product = p1 * p2
    # chi-squared with 4 degrees of freedom: P(X > x) = exp(-x/2) (1 + x/2).
    fisher = product * (1.0 - np.log(product))
    rest = (1 - p1) * (1 - p2)
    pearson = 1.0 - rest * (1.0 - np.log(rest))
    expected = {
        "fisher": fisher,
        "pearson": pearson,
        "tippett": 1.0 - (1.0 - p1) ** 2,
        "simes": min(2 * p1, p2),
    }
    for method, value in expected.items():
        assert_allclose(combine_p_values(p, method=method).p_value, value, rtol=1e-13)
    # The larger p-value to the power k: metap's maximump.
    wilkinson = combine_p_values(p, method="wilkinson", rank=2)
    assert_allclose(wilkinson.p_value, p2**2, rtol=1e-14)
    assert float(wilkinson.statistic) == p2


def test_identical_studies_give_the_closed_forms() -> None:
    # Four studies with p = Phi-bar(2): Stouffer's Z = 2 sqrt(4) = 4.
    p = np.full(4, TAIL_2)
    stouffer = combine_p_values(p, method="stouffer")
    assert_allclose(stouffer.statistic, 4.0, rtol=1e-14)
    assert_allclose(stouffer.p_value, TAIL_4, rtol=1e-13)
    assert float(combine_p_values(p, method="simes").p_value) == TAIL_2
    assert_allclose(
        combine_p_values(p, method="tippett").p_value,
        1.0 - (1.0 - TAIL_2) ** 4,
        rtol=1e-14,
    )
    # Simes's partial conjunction of equal p-values is that p-value, for every u.
    for u in range(1, 5):
        assert_allclose(partial_conjunction(p, u).p_value, TAIL_2, rtol=1e-15)


def test_effects_in_opposite_directions_cancel_in_stouffers_method() -> None:
    p = np.array([0.04, 0.04])
    signed = combine_p_values(p, method="stouffer", directions=[1.0, -1.0])

    assert float(signed.statistic) == 0.0
    assert float(signed.p_value) == 1.0
    assert float(signed.direction) == 0.0
    assert float(signed.p_value_up) == float(signed.p_value_down) == 0.5
    # Without directions the two-sided p-values add up, whatever their signs.
    assert float(combine_p_values(p, method="stouffer").p_value) < 0.01
    # In the same direction, the two-sided p-value of Z = 2 z(0.02) / sqrt(2).
    same = combine_p_values(p, method="stouffer", directions=[2.5, 0.1])
    assert float(same.direction) == 1.0
    assert_allclose(same.p_value, 2.0 * same.p_value_up, rtol=1e-15)


@pytest.mark.parametrize(
    "method", ["fisher", "stouffer", "tippett", "pearson", "mudholkar_george", "simes"]
)
def test_p_values_of_1_combine_to_1(method: str) -> None:
    result = combine_p_values(np.ones((3, 2)), method=method)

    assert_array_equal(result.p_value, [1.0, 1.0])


def test_p_values_of_0_and_1_are_limits() -> None:
    p = np.array([[0.0, 0.0, 0.3], [0.4, 1.0, 1.0]])
    fisher = combine_p_values(p).p_value
    # Two studies: q (1 - log q) for q = 0.3 * 1.
    assert_allclose(fisher, [0.0, 0.0, 0.3 * (1.0 - np.log(0.3))], rtol=1e-14)
    stouffer = combine_p_values(p, method="stouffer").p_value
    # 0 and 1 together have no limit in Stouffer's method.
    assert stouffer[0] == 0.0 and np.isnan(stouffer[1]) and stouffer[2] == 1.0
    assert combine_p_values(p, method="pearson").p_value[2] == 1.0


@pytest.mark.parametrize("method", ["fisher", "stouffer", "tippett", "pearson", "simes"])  # fmt: skip
def test_one_study_is_its_own_combination(method: str) -> None:
    p = np.array([[0.3, np.nan], [np.nan, 0.004]])
    result = combine_p_values(p, method=method)
    signed = combine_p_values(p, method=method, directions=[[-1.0, np.nan], [np.nan, 2.0]])  # fmt: skip

    assert_allclose(result.p_value, [0.3, 0.004], rtol=1e-14)
    # With its direction, one study gives back its two-sided p-value.
    assert_allclose(signed.p_value, [0.3, 0.004], rtol=1e-14)
    assert_array_equal(signed.direction, [-1.0, 1.0])
    assert_array_equal(result.n_studies, [1, 1])


def test_the_logit_method_approximates_also_for_one_study() -> None:
    # T = log((1 - p) / p) is logistic; its t approximation on 9 degrees of freedom
    # gives 0.30457 for p = 0.3 (scipy's mudholkar_george gives the same).
    result = combine_p_values([0.3], method="mudholkar_george")
    assert 0.3 < float(result.p_value) < 0.31


def test_a_feature_that_no_study_tested_has_no_p_value() -> None:
    p = np.array([[0.1, np.nan], [0.2, np.nan]])
    for method in ("fisher", "stouffer", "simes"):
        result = combine_p_values(p, method=method, directions=np.ones((2, 2)))
        assert np.isnan(result.p_value[1]) and np.isnan(result.direction[1])
        assert_array_equal(result.n_studies, [2, 0])
    assert np.isnan(partial_conjunction(p, 1).p_value[1])


def test_partial_conjunction_of_every_study_is_the_largest_p_value() -> None:
    p = np.array([[0.01, 0.2], [0.03, 0.001], [0.004, 0.05]])
    for method in ("simes", "fisher", "stouffer"):
        result = partial_conjunction(p, 3, method=method)
        assert_allclose(result.p_value, [0.03, 0.2], rtol=1e-14)
    # One study's effect does not make the replication p-value small.
    found_once = np.array([1e-12, 0.6, 0.9])
    assert float(partial_conjunction(found_once, 1).p_value) == 3e-12
    assert float(partial_conjunction(found_once, 2).p_value) == 0.9


# Invariants.


def test_the_order_of_the_studies_does_not_matter() -> None:
    data = fixture("combined")
    p, signs, w = data["p"], data["signs"], data["weights"]
    order = [3, 0, 4, 2, 1]
    for method in ("fisher", "stouffer", "tippett", "pearson", "mudholkar_george", "simes"):  # fmt: skip
        weights = w if method == "stouffer" else None
        first = combine_p_values(p, method=method, weights=weights, directions=signs)
        other = combine_p_values(
            p[order],
            method=method,
            weights=None if weights is None else weights[order],
            directions=signs[order],
        )
        # Sums in another order round alike to about 1e-16.
        assert_allclose(other.p_value, first.p_value, rtol=1e-12)
        assert_allclose(other.statistic, first.statistic, rtol=1e-12, atol=1e-14)
    first = partial_conjunction(p, 2, directions=signs)
    assert_array_equal(partial_conjunction(p[order], 2, directions=signs[order]).p_value, first.p_value)  # fmt: skip


def test_the_order_of_the_features_does_not_matter() -> None:
    data = fixture("combined")
    p, signs = data["p"], data["signs"]
    order = np.random.default_rng(3).permutation(p.shape[1])
    for method in ("fisher", "stouffer", "simes"):
        first = combine_p_values(p, method=method, directions=signs)
        other = combine_p_values(p[:, order], method=method, directions=signs[:, order])
        assert_array_equal(other.p_value, first.p_value[order])
        assert_array_equal(other.direction, first.direction[order])


def test_turning_every_direction_around_swaps_the_one_sided_p_values() -> None:
    data = fixture("combined")
    p, signs = data["p"], data["signs"]
    for method in ("fisher", "stouffer", "tippett", "simes"):
        up = combine_p_values(p, method=method, directions=signs)
        down = combine_p_values(p, method=method, directions=-signs)
        assert_array_equal(down.p_value, up.p_value)
        assert_array_equal(down.direction, -up.direction)
        assert_array_equal(down.p_value_up, up.p_value_down)
        assert_array_equal(down.p_value_down, up.p_value_up)


def test_a_direction_is_the_one_sided_p_value_of_its_sign() -> None:
    # Two-sided p-values with the signs of their effects give, for effects above
    # zero, the combination of the one-sided p-values p / 2 and 1 - p / 2, which
    # metap's two2one writes; 1 - p / 2 loses digits for a small p, so the features
    # compared have p-values of at least 1e-5.
    data = fixture("combined")
    p, signs = data["p"], data["signs"]
    one_sided = np.where(signs > 0, p / 2, 1 - p / 2)
    large = np.nanmin(np.where(np.isnan(p), 1.0, p), axis=0) >= 1e-5
    for method in ("fisher", "stouffer", "pearson", "mudholkar_george", "simes"):
        signed = combine_p_values(
            p[:, large], method=method, directions=signs[:, large]
        )
        plain = combine_p_values(one_sided[:, large], method=method)
        assert_allclose(signed.p_value_up, plain.p_value, rtol=1e-11)
        reversed_ = combine_p_values(1 - one_sided[:, large], method=method)
        assert_allclose(signed.p_value_down, reversed_.p_value, rtol=1e-11)


def test_weights_scaled_together_change_nothing() -> None:
    data = fixture("combined")
    p, w = data["p"], data["weights"]
    first = combine_p_values(p, method="stouffer", weights=w)
    # A power of two scales exactly; another factor to rounding.
    assert_array_equal(combine_p_values(p, method="stouffer", weights=4 * w).p_value, first.p_value)  # fmt: skip
    third = combine_p_values(p, method="stouffer", weights=w / 3)
    assert_allclose(third.p_value, first.p_value, rtol=1e-13)
    # Equal weights are Stouffer's unweighted method.
    equal = combine_p_values(p, method="stouffer", weights=np.full(5, 7.0))
    assert_allclose(equal.p_value, combine_p_values(p, method="stouffer").p_value, rtol=1e-13)  # fmt: skip


def test_simes_partial_conjunction_grows_with_u() -> None:
    p = fixture("combined")["p"]
    values = [partial_conjunction(p, u).p_value for u in range(1, 6)]
    for smaller, larger in zip(values, values[1:]):
        tested = ~np.isnan(larger)
        # Each term (n - u) p_(i) / (i - u) grows with u, and fewer are taken.
        assert (larger[tested] >= smaller[tested]).all()


# Arguments.


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: combine_p_values([0.1, 1.2]), ValueError, r"lie in \[0, 1\]"),
        (
            lambda: combine_p_values([0.1, 0.2], method="edgington"),
            ValueError,
            "method must be one of",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], weights=[1.0, 2.0]),
            ValueError,
            "Stouffer's method only",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], method="stouffer", weights=[1.0, 0.0]),
            ValueError,
            "positive and finite",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], method="stouffer", weights=[1.0]),
            ValueError,
            "weights must be",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], method="wilkinson"),
            ValueError,
            "needs rank",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], rank=2),
            ValueError,
            "for method='wilkinson'",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], method="wilkinson", rank=0),
            ValueError,
            "at least 1",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], directions=[1.0, np.nan]),
            ValueError,
            "directions must be given",
        ),  # fmt: skip
        (
            lambda: combine_p_values([0.1, 0.2], directions=[1.0]),
            ValueError,
            "shape of p_values",
        ),  # fmt: skip
        (lambda: combine_p_values(np.empty((0, 3))), ValueError, "no studies"),
        (lambda: combine_p_values(np.ones((2, 2, 2))), ValueError, "dimensions"),
        (lambda: combine_p_values(["a", "b"]), TypeError, "real numbers"),
        (lambda: partial_conjunction([0.1, 0.2], 0), ValueError, "at least 1"),
        (lambda: partial_conjunction([0.1, 0.2], 1.5), TypeError, "whole number"),
        (
            lambda: partial_conjunction([0.1, 0.2], 1, method="tippett"),
            ValueError,
            "method must be one of",
        ),  # fmt: skip
    ],
)
def test_invalid_arguments_are_refused(call: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


def test_results_are_read_only_and_say_what_they_hold() -> None:
    result = combine_p_values(fixture("combined")["p"], method="stouffer", directions=fixture("combined")["signs"])  # fmt: skip

    assert isinstance(result, CombinedPValues)
    assert result.method == "stouffer" and result.at_least == 1 and result.rank is None
    with pytest.raises(ValueError, match="read-only"):
        result.p_value[0] = 0.0
    assert "with directions" in repr(result) and "method='stouffer'" in repr(result)
    one = combine_p_values([0.2, 0.3])
    assert one.p_value.shape == () and one.direction is None


# The proportion of true nulls.


@pytest.mark.parametrize("label", list(PI0_OPTIONS))
def test_pi0_matches_qvalue(label: str) -> None:
    data = fixture("pi0")
    sets = sorted({key.split("/")[0] for key in data})
    compared = 0
    for name in sets:
        if f"{name}/{label}/pi0" not in data:
            continue  # qvalue stops: a count of zero
        ours = proportion_of_true_nulls(data[f"{name}/p"], **PI0_OPTIONS[label])
        assert abs(ours.pi0 - float(data[f"{name}/{label}/pi0"])) <= EXACT
        smooth = data.get(f"{name}/{label}/pi0_smooth")
        if smooth is None:
            assert ours.pi0_smooth is None
        else:
            assert_allclose(ours.pi0_smooth, smooth, rtol=0.0, atol=EXACT)
        compared += 1
    assert compared >= 2


def test_pi0_at_one_lambda_counts_the_p_values_at_least_lambda() -> None:
    p = np.array([0.1, 0.2, 0.5, 0.7, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35])
    # Two of ten p-values are at least 0.5, as qvalue counts them (not above it).
    estimate = proportion_of_true_nulls(p, lambdas=0.5)
    assert estimate.pi0 == 2 / (10 * 0.5)
    assert estimate.pi1 == 1.0 - estimate.pi0
    assert estimate.pi0_smooth is None and estimate.n_p_values == 10
    # Capped at 1, and 0 where no p-value reaches lambda.
    assert proportion_of_true_nulls(np.full(5, 0.9), lambdas=0.5).pi0 == 1.0
    assert proportion_of_true_nulls(np.full(5, 0.1), lambdas=0.5).pi0 == 0.0


def test_pi0_leaves_out_missing_p_values_and_their_order() -> None:
    p = fixture("pi0")["set0/p"]
    shuffled = np.random.default_rng(4).permutation(p)
    first = proportion_of_true_nulls(p)
    for other in (np.append(shuffled, [np.nan, np.nan]), shuffled):
        again = proportion_of_true_nulls(other)
        assert again.pi0 == first.pi0
        assert again.n_p_values == p.size


def test_pi0_lambdas_by_default_are_qvalues() -> None:
    lambdas = proportion_of_true_nulls(np.linspace(0, 1, 50)).lambdas
    # R's seq(0.05, 0.95, 0.05): the last is 0.95 itself, not 0.05 + 18 * 0.05.
    assert lambdas.size == 19 and lambdas[-1] == 0.95 and lambdas[2] == 0.05 + 0.1
    assert isinstance(proportion_of_true_nulls([0.5, 0.7]), TrueNullProportion)


def test_pi0_smooths_counts_of_zero_that_qvalue_refuses() -> None:
    p = np.random.default_rng(1).uniform(size=500) * 0.9
    estimate = proportion_of_true_nulls(p)
    assert 0.0 <= estimate.pi0 < 0.5 and estimate.pi0_lambda[-1] == 0.0
    with pytest.raises(ValueError, match="no p-value is at least lambda"):
        proportion_of_true_nulls(p, smooth_log=True)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"lambdas": [0.2, 0.5]}, "one value or at least 4"),
        ({"lambdas": 1.0}, r"lie in \[0, 1\)"),
        ({"lambdas": [0.1, 0.2, 0.2, 0.4]}, "distinct"),
        ({"smooth_df": 1.0}, "smooth_df must be above 1"),
        ({"smooth_df": 25.0}, "at most the number of lambdas"),
        ({"method": "spline"}, "method must be one of"),
    ],
)
def test_pi0_refuses_invalid_options(options: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        proportion_of_true_nulls(np.linspace(0.0, 1.0, 101), **options)


def test_pi0_refuses_p_values_outside_0_and_1_and_none_at_all() -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        proportion_of_true_nulls([0.2, 1.5])
    with pytest.raises(ValueError, match="no p-values"):
        proportion_of_true_nulls([np.nan])


# Replication across the studies of a collection.


@cache
def cohorts() -> tuple[CubeCollection, dict[str, Any]]:
    """The three small cohorts of the fixture, aligned and tested by limma_voom."""
    data = fixture("replication")
    cubes = {}
    for study in ("A", "B", "C"):
        samples = {column: data[f"{study}/{column}"].tolist() for column in ("sample_id", "donor", "condition")}  # fmt: skip
        cubes[study] = Cube(
            {"counts": data[f"{study}/counts"]},
            samples=samples,
            groups={"group_id": ["T", "B"]},
            features={"feature_id": data[f"{study}/feature_id"].tolist()},
            n_cells=data[f"{study}/n_cells"],
        )
    collection = CubeCollection(cubes, align_groups_by="group_id")
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        design = Design(
            cube.samples,
            "~ donor + condition",
            levels={"condition": ["ctrl", "stim"]},
            contrasts={"stim_vs_ctrl": "conditionstim"},
        )
        results[study] = limma_voom(cube, design)
    return collection, results


@cache
def replication(name: str) -> ReplicationResult:
    collection, results = cohorts()
    return differential_expression_replication(
        collection, results, "T", "stim_vs_ctrl", **SETTINGS[name]
    )


@pytest.mark.parametrize("name", ["T0", "T1"])
def test_replication_matches_r(name: str) -> None:
    data = fixture("replication")
    result = replication(name)
    genes, g = result.genes, data[f"{name}/genes"]
    counts = ("n_studies", "n_up", "n_down", "n_found", "n_found_up", "n_found_down", "direction")  # fmt: skip
    for i, column in enumerate(counts):
        assert_array_equal(genes[column], g[i], err_msg=column)
    for i, column in enumerate(("sign_agreement", "p_value", "replication_p_value", "adjusted_p_value", "replication_adjusted_p_value"), 7):  # fmt: skip
        assert scaled_difference(genes[column], g[i], i > 7) <= EXACT, column
    tier = CLOSE if SETTINGS[name]["meta_method"] == "reml" else EXACT
    for i, column in enumerate(("q_statistic", "q_p_value", "i_squared", "tau_squared")):  # fmt: skip
        assert scaled_difference(genes[column], data[f"{name}/heterogeneity"][i], i == 1) <= tier, column  # fmt: skip
    pairs, r = result.study_pairs, data[f"{name}/pairs"]
    assert_array_equal(pairs["n_found"], r[2])
    assert_array_equal(pairs["n_found_in_both"], r[3])
    assert scaled_difference(pairs["sign_agreement"], r[4]) <= EXACT
    known = ~np.isnan(r[5])  # qvalue stops where no p-value reaches its largest lambda
    assert scaled_difference(pairs["pi1"][known], r[5][known]) <= EXACT
    assert not np.isnan(pairs["pi1"]).any()
    f = data[f"{name}/found_in"]
    assert_array_equal(result.found_in["n_genes"], f[1])
    assert_array_equal(result.found_in["n_genes_same_direction"], f[2])


def test_replication_combines_the_aligned_results_as_the_array_functions_do() -> None:
    collection, results = cohorts()
    result = replication("T0")
    aligned = align_study_results(collection, results, "T", "stim_vs_ctrl")
    k, n = collection.n_studies, collection.n_features
    tested = np.array(aligned["status"].tolist()).reshape(k, n) == "tested"
    genes = tested.any(axis=0)

    def column(name: str) -> np.ndarray:
        values = np.asarray(aligned[name]).reshape(k, n)
        return np.where(tested, values, np.nan)[:, genes]

    p, lfc, se = column("p_value"), column("log2_fold_change"), column("standard_error")
    weights = np.sqrt(result.studies["n_samples"].astype(float))
    combined_p = combine_p_values(
        p,
        method="stouffer",
        weights=np.where(np.isnan(p), np.nan, weights[:, None]),
        directions=lfc,
    )
    assert_array_equal(result.genes["p_value"], combined_p.p_value)
    assert_array_equal(result.genes["direction"], combined_p.direction)
    replicated = partial_conjunction(p, 2, directions=lfc)
    assert_array_equal(result.genes["replication_p_value"], replicated.p_value)
    fit = meta_analysis(lfc, se)
    several = combined_p.n_studies >= 2
    assert_array_equal(result.genes["tau_squared"][several], fit.tau_squared[several])
    assert np.isnan(result.genes["tau_squared"][~several]).all()
    # pi1 of the first pair, from the p-values of the genes the first study finds.
    found = tested[0, genes] & (column("adjusted_p_value")[0] < 0.05)
    chosen = found & ~np.isnan(p[1])
    assert result.study_pairs["pi1"][0] == proportion_of_true_nulls(p[1, chosen]).pi1
    # The genes found in at least k studies, counted from the genes table.
    for k_studies, count in zip(
        result.found_in["n_studies"], result.found_in["n_genes"]
    ):
        assert count == np.count_nonzero(result.genes["n_found"] >= k_studies)


def test_replication_records_its_settings_studies_and_provenance() -> None:
    result = replication("T1")
    settings = result.settings
    assert settings["group"] == "T" and settings["contrast"] == "stim_vs_ctrl"
    assert settings["method"] == "fisher" and settings["weights"] is None
    assert result.provenance["studies"].keys() == {"A", "B", "C"}
    assert "alignment" in result.provenance["collection"]
    assert_array_equal(result.studies["analysed"], [True, True, True])
    assert np.isnan(result.studies["weight"]).all()
    assert_array_equal(replication("T0").studies["weight"], np.sqrt(result.studies["n_samples"]))  # fmt: skip
    copy = pickle.loads(pickle.dumps(result))
    assert copy.genes.equals(result.genes) and copy.provenance == result.provenance
    with pytest.raises(AttributeError, match="immutable"):
        result._genes = None  # type: ignore[misc]
    assert "ReplicationResult of group 'T'" in repr(result)


def test_replication_refuses_invalid_settings() -> None:
    collection, results = cohorts()

    def run(**options: Any) -> None:
        differential_expression_replication(collection, results, "T", "stim_vs_ctrl", **options)  # fmt: skip

    with pytest.raises(ValueError, match="at_least must be from 1 to the 3"):
        run(at_least=4)
    with pytest.raises(ValueError, match="Stouffer's method only"):
        run(method="fisher", weights={"A": 1.0, "B": 1.0, "C": 1.0})
    with pytest.raises(ValueError, match="'C' has none"):
        run(weights={"A": 1.0, "B": 2.0})
    with pytest.raises(ValueError, match="alpha must be in"):
        run(alpha=1.0)
    with pytest.raises(ValueError, match="no option 'smooth'"):
        run(null_proportion={"smooth": 3})
    with pytest.raises(ValueError, match="one value or at least 4"):
        run(null_proportion={"lambdas": [0.2, 0.5]})
    with pytest.raises(KeyError, match="no group 'NK'"):
        differential_expression_replication(collection, results, "NK", "stim_vs_ctrl")
    one = CubeCollection({"A": collection.cubes["A"]}, align_groups_by="group_id")
    with pytest.raises(ValueError, match="replication needs at least two"):
        differential_expression_replication(
            one, {"A": results["A"]}, "T", "stim_vs_ctrl"
        )


# Figures.


def test_the_agreement_heatmap_draws_pi1_of_every_pair() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from cytepool.plotting import plot_study_agreement

    result = replication("T1")
    ax = plot_study_agreement(result)
    drawn = np.ma.filled(ax.get_images()[0].get_array().astype(float), np.nan)
    pairs = result.study_pairs
    expected = np.full((3, 3), np.nan)
    index = {"A": 0, "B": 1, "C": 2}
    for a, b, value in zip(pairs["discovery"], pairs["replication"], pairs["pi1"]):
        expected[index[a], index[b]] = value
    assert_array_equal(drawn, expected)
    numbers = [t.get_text() for t in ax.texts if t.get_gid() == "numbers"]
    assert len(numbers) == 6 and numbers[0].startswith(f"{expected[0, 1]:.2f}")
    agreement = plot_study_agreement(result, measure="sign_agreement", numbers=False)
    assert not [t for t in agreement.texts if t.get_gid() == "numbers"]
    with pytest.raises(ValueError, match="measure must be one of"):
        plot_study_agreement(result, measure="kappa")
    with pytest.raises(TypeError, match="ReplicationResult"):
        plot_study_agreement(result.genes)  # type: ignore[arg-type]
    pyplot.close("all")


def test_the_bars_count_the_genes_each_number_of_studies_finds() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from cytepool.plotting import plot_replicated_genes

    result = replication("T0")
    ax = plot_replicated_genes(result)
    genes = result.genes
    found = np.asarray(genes["n_found"])
    one_way = np.maximum(genes["n_found_up"], genes["n_found_down"]) == found
    for gid, mask in (("same direction", one_way), ("both directions", ~one_way)):
        heights = [bar.get_height() for bar in ax.patches if bar.get_gid() == gid]
        assert heights == [np.count_nonzero((found == k) & mask) for k in (1, 2, 3)]
    pyplot.close("all")
