"""Tests of williams_trend_test against isotonic regression by scipy, statistics derived
by hand, exact enumeration of every relabelling of small designs, and invariants."""

import itertools
import json
import math
import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import isotonic_regression
from scipy.stats import norm

from cytepool.stats.trend_tests import WilliamsTrendTest, williams_trend_test

# Genes of the 3, 2, 2, 2 design whose exact p-value, conditional on the side, lies
# 3 to 16 tolerances away from the larger-of-two-sides p-value (found by enumeration
# of random genes); values rounded to 3 decimals.
UNEQUAL_DOSES = np.repeat([0.0, 1.0, 3.0, 10.0], (3, 2, 2, 2))


def _end_statistics(groups: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """t_up and t_down of every row of group labels, by the minimax formula of
    isotonic regression: the non-decreasing fit is min_j mean(0..j) at the first group
    and max_i mean(i..K-1) at the last; the non-increasing fit swaps min and max."""
    one_hot = groups[:, :, None] == np.arange(k)[None, None, :]
    n = one_hot.sum(axis=1).astype(float)
    sums = np.einsum("ank,n->ak", one_hot, y)
    within = (y**2).sum() - (sums**2 / n).sum(axis=1)
    s = np.sqrt(within / (y.size - k))
    scale = s * np.sqrt(1.0 / n[:, -1] + 1.0 / n[:, 0])
    head = np.cumsum(sums, axis=1) / np.cumsum(n, axis=1)  # mean(0..j)
    tail = (
        np.cumsum(sums[:, ::-1], axis=1)[:, ::-1]
        / np.cumsum(n[:, ::-1], axis=1)[:, ::-1]
    )  # mean(i..K-1)
    up = (tail.max(axis=1) - head.min(axis=1)) / scale
    down = (head.max(axis=1) - tail.min(axis=1)) / scale
    return np.column_stack([up, down])


def _exact(doses: np.ndarray, y: np.ndarray) -> tuple[float, float, float, str]:
    """The exact p-value conditional on the side, the larger-of-two-sides p-value, the
    share of relabellings on the side and the side, over every distinct relabelling."""
    levels = np.unique(doses)
    labels = np.searchsorted(levels, doses)
    table = np.array(sorted(set(itertools.permutations(labels.tolist()))))
    observed = _end_statistics(labels[None, :], y, levels.size)[0]
    up_side = observed[0] >= observed[1]
    statistic = observed.max()
    every = _end_statistics(table, y, levels.size)
    same = every[:, 0] >= every[:, 1] if up_side else every[:, 1] > every[:, 0]
    reach = (every[:, 0] if up_side else every[:, 1]) >= statistic * (1.0 - 1e-9)
    conditional = np.count_nonzero(same & reach) / np.count_nonzero(same)
    larger = np.count_nonzero(every.max(axis=1) >= statistic * (1.0 - 1e-9)) / len(
        table
    )
    share = np.count_nonzero(same) / len(table)
    return conditional, larger, share, "up" if up_side else "down"


UNEQUAL_GENES = [
    # conditional 0.0020, larger of two sides 0.0312
    [1.886, 1.347, 0.662, -0.403, -0.247, -2.919, -4.879, 2.051, 2.157],
    # conditional 0.0474, larger of two sides 0.1231
    [0.705, 0.197, -0.051, -4.053, -0.889, -0.525, 0.624, 0.975, 0.889],
    # conditional 0.0743, larger of two sides 0.1205
    [-0.756, 1.081, -2.026, 0.787, -1.425, 2.462, 1.794, 2.01, 1.525],
]


def _exact_without_spread(y: np.ndarray, labels: np.ndarray) -> Fraction:
    """The exact conditional p-value of a gene whose groups have no spread, in exact
    arithmetic: a side's statistic is infinite when its fit changes and the
    relabelling leaves no spread within groups, 0 when its fit is flat."""
    k = int(labels.max()) + 1
    on_side = reaching = 0
    for row in sorted(set(itertools.permutations(labels.tolist()))):
        groups = [[Fraction(v) for v, r in zip(y, row) if r == g] for g in range(k)]
        means = [sum(group) / len(group) for group in groups]
        sizes = [len(group) for group in groups]
        within = sum((v - m) ** 2 for group, m in zip(groups, means) for v in group)
        head = [
            sum(m * n for m, n in zip(means[: j + 1], sizes)) / sum(sizes[: j + 1])
            for j in range(k)
        ]
        tail = [
            sum(m * n for m, n in zip(means[i:], sizes)) / sum(sizes[i:])
            for i in range(k)
        ]

        def side(change: Fraction) -> float:
            if change <= 0:
                return 0.0
            return math.inf if within == 0 else float(change) / math.sqrt(within)

        up, down = side(max(tail) - min(head)), side(max(head) - min(tail))
        if up >= down:
            on_side += 1
            reaching += up == math.inf
    return Fraction(reaching, on_side)


def test_isotonic_means_match_scipy() -> None:
    rng = np.random.default_rng(3)
    doses = np.repeat([0.0, 0.1, 1.0, 3.0, 10.0], (4, 2, 3, 3, 2))
    values = (
        rng.normal(size=(doses.size, 40))
        + rng.normal(size=(5, 40))[np.searchsorted(np.unique(doses), doses)]
    )
    values[:, 0] = np.repeat([1.0, 2.0, 2.0, 1.0, 3.0], (4, 2, 3, 3, 2))  # ties
    test = williams_trend_test(doses, values, n_permutations=10, rng=1)
    labels = np.searchsorted(test.doses, doses)
    for g in range(values.shape[1]):
        means = np.bincount(labels, weights=values[:, g]) / test.n_samples
        fit = isotonic_regression(
            means, weights=test.n_samples, increasing=test.direction[g] == "up"
        ).x
        np.testing.assert_allclose(test.isotonic_means[g], fit, rtol=0, atol=1e-12)


def test_statistic_by_hand() -> None:
    # Equal sizes: means (1, 3, 2, 4); the rising fit pools 3 and 2 into 2.5, the
    # falling fit pools everything into 2.5; within sum of squares 0.30 on 4 df.
    up_gene = [0.9, 1.1, 2.8, 3.2, 1.9, 2.1, 3.7, 4.3]
    # Sizes (2, 3, 2, 2): means (2, 1.1, 1.6, 0.5); the falling fit pools 1.1 (3
    # samples) and 1.6 (2) into 1.3, not the 1.35 of equal weights; within sum of
    # squares 0.08 on 5 df.
    down_gene = [1.9, 2.1, 1.0, 1.1, 1.2, 1.5, 1.7, 0.4, 0.6]
    test = williams_trend_test(
        np.repeat([0.0, 1, 3, 10], 2), np.array(up_gene)[:, None]
    )
    assert test.direction[0] == "up"
    assert test.statistic[0] == pytest.approx(3.0 / math.sqrt(0.30 / 4), rel=1e-12)
    np.testing.assert_allclose(test.isotonic_means[0], [1.0, 2.5, 2.5, 4.0], atol=1e-12)
    doses = np.repeat([0.0, 1, 3, 10], (2, 3, 2, 2))
    test = williams_trend_test(doses, np.array(down_gene)[:, None])
    assert test.direction[0] == "down"
    assert test.statistic[0] == pytest.approx(1.5 / math.sqrt(0.08 / 5), rel=1e-12)
    np.testing.assert_allclose(test.isotonic_means[0], [2.0, 1.3, 1.3, 0.5], atol=1e-12)
    # A vehicle of 3 samples and a top dose of 2: means (2.0, 2.6, 3.2), within sum
    # of squares 0.12 on 4 df, and sqrt(1/2 + 1/3) in the denominator.
    rising = [1.9, 2.0, 2.1, 2.5, 2.7, 3.0, 3.4]
    test = williams_trend_test(
        np.repeat([0.0, 1, 3], (3, 2, 2)), np.array(rising)[:, None]
    )
    expected = 1.2 / (math.sqrt(0.12 / 4) * math.sqrt(1 / 2 + 1 / 3))
    assert test.statistic[0] == pytest.approx(expected, rel=1e-12)
    # Equal means with spread: both sides' fits are flat, 0 each, and the tie points
    # up; every relabelling reaches 0.
    flat = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    test = williams_trend_test(np.repeat([0.0, 1, 3, 10], 2), np.array(flat)[:, None])
    assert test.statistic[0] == 0.0 and test.direction[0] == "up"
    assert test.p_value[0] == 1.0


@pytest.mark.parametrize("design", ["equal", "unequal"])
def test_p_values_match_exact_enumeration(design: str) -> None:
    """Within z Monte Carlo SDs plus 2 / (B + 1) of the exact conditional p-value,
    z the two-sided 0.001 / n_genes quantile; on the unequal design, the genes lie
    far enough from the larger-of-two-sides p-value that that estimator fails."""
    b = 20_000
    if design == "equal":
        doses = np.repeat([0.0, 1.0, 3.0, 10.0], 2)
        rng = np.random.default_rng(7)
        trend = np.array([0.0, 0.5, 1.0, 1.5])[np.searchsorted([0, 1, 3, 10], doses)]
        genes = [rng.normal(size=8) + scale * trend for scale in (0, 0, 1, 1, 2, -2)]
    else:
        doses = UNEQUAL_DOSES
        genes = [np.array(gene) for gene in UNEQUAL_GENES]
    values = np.column_stack(genes)
    test = williams_trend_test(doses, values, n_permutations=b, rng=11)
    z = norm.isf(0.0005 / len(genes))
    for g, gene in enumerate(genes):
        conditional, larger, share, side = _exact(doses, gene)
        assert test.direction[g] == side
        allowed = z * math.sqrt(conditional * (1 - conditional) / (share * b)) + 2 / (
            b + 1
        )
        assert abs(test.p_value[g] - conditional) <= allowed
        if design == "unequal":
            assert abs(larger - conditional) >= 3 * allowed


def test_one_permutation_gives_the_formula_exactly() -> None:
    """With one relabelling of 40 samples, which cannot repeat the observed labels
    of strong genes: p is 1/2 where it lies on the gene's side short of its
    statistic and 1 where it lies on the other side, which b / n (0 or NaN) and
    (b + 1) / n (1 or infinity) do not give."""
    doses = np.repeat([0.0, 1.0, 3.0, 10.0, 30.0], 8)
    rng = np.random.default_rng(5)
    level = np.searchsorted([0, 1, 3, 10, 30], doses)
    # Strong genes of many shapes, so that the one relabelling falls on the side of
    # some genes and not of others.
    patterns = rng.normal(0.0, 3.0, (5, 100))
    values = rng.normal(0, 0.2, (40, 100)) + patterns[level]
    test = williams_trend_test(doses, values, n_permutations=1, rng=2)
    assert set(np.unique(test.p_value).tolist()) == {0.5, 1.0}


def test_genes_without_spread() -> None:
    doses = np.repeat([0.0, 1.0, 3.0, 10.0], 2)
    values = np.column_stack(
        [
            np.full(8, 5.0),  # one value: no test
            np.repeat([1.0, 2.0, 2.0, 3.0], 2),  # no spread within groups
        ]
    )
    test = williams_trend_test(doses, values, n_permutations=20_000, rng=4)
    assert math.isnan(test.statistic[0]) and math.isnan(test.p_value[0])
    assert test.direction[0] == ""
    np.testing.assert_array_equal(test.isotonic_means[0], np.full(4, 5.0))
    # No spread within groups: the rising side's statistic is infinite, or as large
    # as the rounding of the sums of squares leaves it.
    assert test.statistic[1] > 1e6 and test.direction[1] == "up"
    # The relabellings that leave every group without spread and the fit rising reach
    # it: 36 of the 1,428 distinct relabellings on the rising side, counted in exact
    # arithmetic, the tied 2s of the middle groups among them.
    exact = _exact_without_spread(values[:, 1], np.repeat(np.arange(4), 2))
    assert exact == Fraction(36, 1428)
    sd = math.sqrt(float(exact) * (1 - float(exact)) / (20_000 * 1428 / 2520))
    assert abs(test.p_value[1] - float(exact)) <= 4 * sd + 2 / 20_001
    assert "no test for 1 gene whose samples all have one value" in repr(test)


def test_invariants() -> None:
    rng = np.random.default_rng(9)
    doses = np.repeat([0.0, 0.3, 1.0, 3.0, 10.0], (3, 3, 4, 3, 3))
    values = (
        rng.normal(0, 0.05, (16, 30))
        + np.linspace(-0.1, 0.1, 30)[None, :]
        * (np.searchsorted([0, 0.3, 1, 3, 10], doses)[:, None])
    )
    base = williams_trend_test(doses, values, n_permutations=3_000, rng=21)
    # A change of scale and a large shift: an uncentred sum of squares loses the
    # digits that tell relabellings apart.
    shifted = williams_trend_test(doses, values * 3 + 1e4, n_permutations=3_000, rng=21)
    np.testing.assert_allclose(shifted.statistic, base.statistic, rtol=1e-9)
    np.testing.assert_array_equal(shifted.p_value, base.p_value)
    np.testing.assert_array_equal(shifted.direction, base.direction)
    flipped = williams_trend_test(doses, -values, n_permutations=3_000, rng=21)
    swap = {"up": "down", "down": "up"}
    assert [swap[d] for d in base.direction] == flipped.direction.tolist()
    np.testing.assert_array_equal(flipped.p_value, base.p_value)
    order = rng.permutation(30)
    reordered = williams_trend_test(
        doses, values[:, order], n_permutations=3_000, rng=21
    )
    np.testing.assert_array_equal(reordered.p_value, base.p_value[order])
    one = williams_trend_test(doses, values, n_permutations=3_000, rng=21, n_threads=1)
    four = williams_trend_test(doses, values, n_permutations=3_000, rng=21, n_threads=4)
    np.testing.assert_array_equal(one.p_value, four.p_value)
    np.testing.assert_array_equal(one.p_value, base.p_value)
    other = williams_trend_test(doses, values, n_permutations=3_000, rng=22)
    assert not np.array_equal(other.p_value, base.p_value)
    # Samples in another order, with their doses: the same statistics.
    samples = rng.permutation(16)
    moved = williams_trend_test(
        doses[samples], values[samples], n_permutations=10, rng=1
    )
    np.testing.assert_allclose(moved.statistic, base.statistic, rtol=1e-12)


def test_seed_is_recorded_and_draws_the_same_relabellings() -> None:
    doses = np.repeat([0.0, 1.0, 3.0], 3)
    values = np.random.default_rng(1).normal(size=(9, 5))
    first = williams_trend_test(doses, values, n_permutations=25_000, rng=None)
    again = williams_trend_test(doses, values, n_permutations=25_000, rng=first.seed)
    np.testing.assert_array_equal(first.p_value, again.p_value)
    from_generator = williams_trend_test(
        doses, values, n_permutations=100, rng=np.random.default_rng(3)
    )
    replay = williams_trend_test(
        doses, values, n_permutations=100, rng=from_generator.seed
    )
    np.testing.assert_array_equal(from_generator.p_value, replay.p_value)
    assert first.n_permutations == 25_000


def test_relabellings_beyond_one_block_are_new() -> None:
    """Relabellings are drawn in blocks of 10,000 from one generator: at B = 20,000
    the second block is new, so the p-values move from those of the first block by
    Monte Carlo error, where a block drawn again would move each by less than
    1 / (2n + 1), n the relabellings on the gene's side."""
    doses = np.repeat([0.0, 1.0, 3.0, 10.0], 3)
    values = np.random.default_rng(12).normal(size=(12, 60))
    first = williams_trend_test(doses, values, n_permutations=10_000, rng=3)
    both = williams_trend_test(doses, values, n_permutations=20_000, rng=3)
    moved = np.abs(both.p_value - first.p_value) > 2 / 10_001
    assert np.mean(moved) > 0.5


def test_result_reads_back() -> None:
    doses = np.repeat([0.0, 1.0, 3.0], 3)
    values = np.random.default_rng(2).normal(size=(9, 3))
    values[:, 0] += [0, 0, 0, 1, 1, 1, 2, 2, 2]
    test = williams_trend_test(doses, values, n_permutations=500, rng=8)
    table = test.table(feature_ids=["a", "b", "c"])
    assert table.columns == ("feature_id", "direction", "statistic", "p_value")
    assert table["feature_id"].tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(table["p_value"], test.p_value)
    assert test.table()["feature_id"].tolist() == [
        "feature_0",
        "feature_1",
        "feature_2",
    ]
    text = repr(test)
    assert text.startswith(
        "WilliamsTrendTest: 3 genes; 3 dose groups (the vehicle and 2"
    )
    assert "500 permutations, seed 8" in text
    up = int(np.sum((test.p_value < 0.05) & (test.direction == "up")))
    assert "p < 0.05: " in text and f"({up} up" in text
    copy = pickle.loads(pickle.dumps(test))
    assert isinstance(copy, WilliamsTrendTest)
    np.testing.assert_array_equal(copy.p_value, test.p_value)
    for array in (copy.p_value, copy.isotonic_means, test.statistic, test.doses):
        with pytest.raises(ValueError):
            array[0] = 1.0
    assert np.array_equal(test.doses, [0.0, 1.0, 3.0])
    assert test.n_samples.tolist() == [3, 3, 3]


def test_vehicle_given_as_negative_zero() -> None:
    doses = np.array([-0.0, -0.0, 0.0, 1.0, 1.0, 1.0])
    values = np.random.default_rng(4).normal(size=(6, 2))
    test = williams_trend_test(doses, values, n_permutations=50, rng=1)
    assert test.doses.tolist() == [0.0, 1.0]


def test_refusals() -> None:
    doses = np.repeat([0.0, 1.0, 3.0], 2)
    values = np.ones((6, 2)) + np.arange(6)[:, None]
    with pytest.raises(ValueError, match="no dose is 0"):
        williams_trend_test(doses + 1, values)
    with pytest.raises(ValueError, match="needs a dose group besides the vehicle"):
        williams_trend_test(np.zeros(6), values)
    with pytest.raises(ValueError, match="every dose group has one sample"):
        williams_trend_test([0.0, 1.0, 3.0], values[:3])
    with pytest.raises(ValueError, match="must be finite"):
        williams_trend_test(doses, np.where(values > 5, np.nan, values))
    with pytest.raises(ValueError, match="at most 1e"):
        williams_trend_test(doses, values * 1e101)
    with pytest.raises(ValueError, match="2-D"):
        williams_trend_test(doses, values[:, 0])
    with pytest.raises(ValueError, match="rows"):
        williams_trend_test(doses[:4], values)
    with pytest.raises(ValueError, match="at least one gene"):
        williams_trend_test(doses, values[:, :0])
    with pytest.raises(ValueError, match="finite and at least 0"):
        williams_trend_test(doses - 0.5, values)
    with pytest.raises(ValueError, match="differ by less than 1e-8"):
        williams_trend_test([0.0, 0.0, 1.0, 1.0 + 1e-12, 3.0, 3.0], values)
    with pytest.raises(ValueError, match="at least 1"):
        williams_trend_test(doses, values, n_permutations=0)
    with pytest.raises(TypeError, match="n_permutations"):
        williams_trend_test(doses, values, n_permutations=10.5)
    with pytest.raises(TypeError, match="rng"):
        williams_trend_test(doses, values, rng="seed")
    with pytest.raises(ValueError, match="non-negative"):
        williams_trend_test(doses, values, rng=-1)
    with pytest.raises(TypeError, match="masked"):
        williams_trend_test(doses, np.ma.masked_array(values))
    with pytest.raises(TypeError, match="masked"):
        williams_trend_test(np.ma.masked_array(doses), values)
    with pytest.raises(TypeError, match="real numbers"):
        williams_trend_test(doses, values.astype(bool))
    with pytest.raises(TypeError, match="real numbers"):
        williams_trend_test(doses, np.full((6, 2), "a"))


def test_p_values_match_bmdexpress() -> None:
    """BMDExpress 2.30.520's Williams p-values of 100 acrylamide probes of GSE147072
    (validation/toxicology_rule_sets.py), each from 10,000 of its permutations, within
    z SDs plus 2 / (B + 1) of CytePool's from as many, z the two-sided 0.001 / 100
    quantile and the SD from at least 0.45 B relabellings per side."""
    found = json.loads(
        (
            Path(__file__).parent / "data" / "toxicology_rule_sets" / "williams.json"
        ).read_text()
    )
    b = found["permutations"]
    test = williams_trend_test(
        np.array(found["doses"]), np.array(found["values"]), n_permutations=b, rng=3
    )
    theirs = np.array(found["bmdexpress_p_values"])
    mean = (test.p_value + theirs) / 2
    sd = np.sqrt(mean * (1 - mean) * 2 / (0.45 * b))
    z = norm.isf(0.0005 / theirs.size)
    assert np.all(np.abs(test.p_value - theirs) <= z * sd + 2 / (b + 1))
