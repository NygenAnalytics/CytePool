"""Tests for the rotation gene-set tests roast, mroast and romer.

Reference values in ``tests/data/rotation_gene_set_tests`` come from limma's ``roast``
and ``romer``, with the standard normal draws R gave limma for its rotations, which
the tests replay to CytePool so that the p-values must agree exactly;
``validation/rotation_gene_set_tests.py`` writes them. The other tests check
properties that a wrong implementation would break: invariances, symmetries, the
limit that fry computes, closed forms and the documented edge cases. A result of
limma-voom or limma-trend is tested, bit for bit, as the values of its group's fit,
with the same rotations.
"""

import json
import math
import re
from functools import cache
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_gene_set_tests import SYMBOL_SETS, fitted, results

from cytepool import Design, Table, fry, mroast, roast, romer
from cytepool.rotation_gene_set_tests import _bailey_z_scores, _proportion_true_null
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.voom import voom

DATA = Path(__file__).parent / "data" / "rotation_gene_set_tests"
EXACT = 1e-12
FLOOR = 0.6744897501960817  # floormean's mixed floor
CASES = (
    "groups",
    "groups_romer",
    "groups_weights",
    "groups_trend_robust",
    "paired_sample_weights",
    "equal_variances",
    "gene_weights",
)


class Replay(np.random.Generator):
    """A generator that gives, batch after batch, the standard normal draws R gave
    limma."""

    def __init__(self, draws: np.ndarray) -> None:
        super().__init__(np.random.PCG64())
        self.draws, self.used = np.asarray(draws, dtype=float).ravel(), 0

    def standard_normal(
        self, size: Any = None, dtype: Any = None, out: Any = None
    ) -> Any:
        n = int(np.prod(size))
        self.used += n
        return self.draws[self.used - n : self.used].reshape(size)


@cache
def fixture(case: str) -> dict[str, Any]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def runs(case: str) -> list[dict[str, Any]]:
    return list(json.loads(str(fixture(case)["runs"])))


def gene_sets(case: str) -> dict[str, Any]:
    """Each set as CytePool takes it: positions, or a mapping to its weights."""
    data = fixture(case)
    sets: dict[str, Any] = {}
    for key, rows in data.items():
        if key.startswith("set ") and not key.startswith("set weights "):
            name = key[4:]
            weights = data.get(f"set weights {name}")
            sets[name] = (
                rows.tolist()
                if weights is None
                else dict(zip(rows.tolist(), weights.tolist(), strict=True))
            )
    return sets


def arguments(case: str) -> dict[str, Any]:
    data = fixture(case)
    if "contrast weights" in data:
        contrast: Any = data["contrast weights"]
    else:
        contrast = data["columns"].tolist().index(str(data["contrast"]))
    return {
        "data": data["y"],
        "design": data["design"],
        "contrast": contrast,
        "weights": data.get("weights"),
    }


def limma(case: str, run: str) -> Any:
    return json.loads(str(fixture(case)[f"limma {run}"]))


def roast_p_values(table: Table) -> np.ndarray:
    """The p-values of roast's row in the order of limma's alternatives: down, up,
    up or down (the two-sided p-value) and mixed."""
    columns = ("down_p_value", "up_p_value", "p_value", "mixed_p_value")
    return np.array([table[column][0] for column in columns])


def active_proportions(table: Table) -> np.ndarray:
    """The proportions of active genes of roast's row in the order of limma's
    alternatives: down, up, up or down (the larger of the two) and mixed."""
    down, up = table["proportion_down"][0], table["proportion_up"][0]
    return np.array([down, up, max(down, up), table["proportion_mixed"][0]])


def replay(case: str, run: str) -> Replay:
    return Replay(fixture(case)[f"draws {run}"])


def compared_runs(test: str) -> list[Any]:
    return [
        pytest.param(case, run, id=f"{case}-{run['name']}")
        for case in CASES
        for run in runs(case)
        if run["test"] == test
    ]


# Agreement with limma, given limma's rotations.


def expected_p_values(
    statistic: str, z: np.ndarray, genes: Any, reference: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """limma's p-values, changed where a deliberate difference explains them, and
    which are compared; ``z`` are limma's z-scores of every gene."""
    rows = np.array(list(genes), dtype=int)
    a = np.array(list(genes.values()) if isinstance(genes, dict) else [1.0] * rows.size)
    expected = np.array(reference["p"], dtype=float)
    compared = np.ones(4, dtype=bool)
    if statistic == "floormean" and np.any(a < 0):
        expected[3] = reference["abs_weights"][3]  # limma's mixed statistic with |a|
    if statistic == "msq" and np.any(a < 0) and np.any(a > 0):
        compared[:3] = False  # limma's directional msq with weights of both signs
    if statistic == "mean50" and np.any(a == 0):
        compared[:] = False  # limma keeps genes of weight zero
    if statistic in ("floormean", "msq"):
        # The observed statistic at its smallest value (by limma's z-scores):
        # CytePool counts rotated statistics equal to it, limma does not.
        x, zs = a[a != 0] * z[rows[a != 0]], z[rows[a != 0]]
        minimum = np.array(
            [
                np.all(x >= 0),
                np.all(x <= 0),
                False,
                statistic == "floormean" and np.all(np.abs(zs) <= FLOOR),
            ]
        )
        expected[minimum & (expected < 1.0)] = 1.0
    return expected, compared


@pytest.mark.parametrize(("case", "run"), compared_runs("roast"))
def test_roast_matches_limma_given_its_rotations(
    case: str, run: dict[str, Any]
) -> None:
    reference_run = limma(case, run["name"])
    z = np.array(reference_run["z"], dtype=float)
    options = {
        "set_statistic": run["stat"],
        "n_rotations": run["nrot"],
        "trend": run.get("trend", False),
        "robust": run.get("robust", False),
    }
    for (name, genes), reference in zip(
        gene_sets(case).items(), reference_run["sets"], strict=True
    ):
        table = roast(
            **arguments(case), gene_set=genes, rng=replay(case, run["name"]), **options
        )
        expected, compared = expected_p_values(run["stat"], z, genes, reference)
        assert_allclose(
            roast_p_values(table)[compared], expected[compared], rtol=0, atol=EXACT
        )
        assert_allclose(active_proportions(table), reference["active"], atol=EXACT)
        zero = isinstance(genes, dict) and 0.0 in genes.values()
        n_expected = sum(w != 0 for w in genes.values()) if zero else reference["n"][0]
        assert table["n_genes"][0] == n_expected, name


@pytest.mark.parametrize(("case", "run"), compared_runs("romer"))
def test_romer_matches_limma_given_its_rotations(
    case: str, run: dict[str, Any]
) -> None:
    reference = limma(case, run["name"])
    table = romer(
        **arguments(case),
        gene_sets=gene_sets(case),
        set_statistic=run["stat"],
        n_rotations=run["nrot"],
        shrink_contrast_effects=run.get("shrink", True),
        rng=replay(case, run["name"]),
    )
    assert_array_equal(table["n_genes"], reference["n"])
    for column in ("up", "down", "mixed"):
        assert_allclose(
            table[f"{column}_p_value"], reference[column], rtol=0, atol=EXACT
        )


def test_mroast_shares_the_rotations_of_each_roast_and_gets_limmas_directions() -> None:
    # limma tests each set of the fixture with the same seed, so the replayed
    # rotations are those of every set, as mroast shares them.
    case, run = "groups", "roast_mean"
    table = mroast(
        **arguments(case),
        gene_sets=gene_sets(case),
        n_rotations=runs(case)[0]["nrot"],
        rng=replay(case, run),
    )
    for i, reference in enumerate(limma(case, run)["sets"]):
        if table["gene_set"][i] != "partly_zero":  # limma counts genes of weight 0
            assert table["n_genes"][i] == reference["n"][0]
        assert_allclose(table["p_value"][i], reference["two_sided"], atol=EXACT)
        assert_allclose(table["mixed_p_value"][i], reference["p"][3], atol=EXACT)
        assert_allclose(table["proportion_down"][i], reference["active"][0], atol=EXACT)
        assert_allclose(table["proportion_up"][i], reference["active"][1], atol=EXACT)
        assert table["direction"][i] == ("up" if reference["up"][0] else "down")


# Rotations and their generator.


def three_groups(seed: int, n_genes: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(9), np.repeat([0, 1, 0], 3), np.repeat([0, 0, 1], 3)])
    y = rng.normal(size=(9, n_genes)) * rng.uniform(0.5, 2.0, n_genes)
    y[3:6, :10] += 1.5
    y[3:6, 10:15] -= 1.5
    return y, x


SETS = {"up": list(range(10)), "mixed": list(range(5, 15)), "null": list(range(30, 50))}


def assert_tables_equal(first: Table, second: Table) -> None:
    assert first.columns == second.columns
    for column in first.columns:
        assert_array_equal(first[column], second[column])


def test_a_seed_or_a_generator_gives_the_same_tables_every_time() -> None:
    y, x = three_groups(1)
    for test, options in ((mroast, {}), (romer, {"n_rotations": 199})):
        first = test(y, SETS, design=x, contrast=1, rng=7, **options)
        again = test(
            y, SETS, design=x, contrast=1, rng=np.random.default_rng(7), **options
        )
        assert_tables_equal(first, again)
        other = test(y, SETS, design=x, contrast=1, rng=8, **options)
        assert not np.array_equal(first["mixed_p_value"], other["mixed_p_value"])


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs numba to start two threads"
)
def test_the_number_of_threads_does_not_change_the_result() -> None:
    y, x = three_groups(2)
    for test, statistic in ((mroast, "mean50"), (romer, "floormean")):
        options = {"set_statistic": statistic, "n_rotations": 101, "rng": 3}
        one = test(y, SETS, design=x, contrast=1, n_threads=1, **options)
        two = test(y, SETS, design=x, contrast=1, n_threads=2, **options)
        assert_tables_equal(one, two)


def test_rotations_drawn_in_batches_are_those_of_one_draw() -> None:
    # 5,000 rotations take two batches; numpy's stream is the same as for one draw
    # of 5,000, which a generator giving that draw at once confirms.
    y, x = three_groups(17)
    one_draw = np.random.default_rng(8).standard_normal((5000, 7))
    for test, options in (
        (roast, {"gene_set": [0, 1, 2, 3, 20]}),
        (romer, {"gene_sets": SETS}),
    ):
        batched = test(y, design=x, contrast=1, n_rotations=5000, rng=8, **options)
        at_once = test(
            y, design=x, contrast=1, n_rotations=5000, rng=Replay(one_draw), **options
        )
        assert_tables_equal(batched, at_once)


def test_mroast_tests_every_set_with_the_rotations_of_roast() -> None:
    y, x = three_groups(3)
    table = mroast(y, SETS, design=x, contrast=1, set_statistic="floormean", rng=5)
    for i, genes in enumerate(SETS.values()):
        alone = roast(y, genes, design=x, contrast=1, set_statistic="floormean", rng=5)
        assert table["p_value"][i] == roast_p_values(alone)[2]
        assert table["mixed_p_value"][i] == roast_p_values(alone)[3]


# Invariances and symmetries.


@pytest.mark.parametrize("statistic", ["mean", "floormean", "mean50", "msq"])
def test_negating_the_contrast_and_the_rotations_swaps_up_and_down(
    statistic: str,
) -> None:
    # c and -c have the same residual basis when c_0 is not zero, and only the
    # contrast effect changes sign. Negating the first coordinate of every rotation
    # then gives -c the rotated z-scores of c, while its observed z-scores are those
    # of c negated, so the up and down tests swap exactly and the others stay.
    y, x = three_groups(4)
    contrast = np.array([0.25, 1.0, -0.5])
    draws = np.random.default_rng(6).standard_normal((499, 7))
    flipped = draws.copy()
    flipped[:, 0] *= -1.0
    genes = {g: (1.0 if g < 12 else -2.0) for g in range(0, 20)}
    for gene_set in (list(range(10)), list(range(5, 25)), genes):
        plus = roast(
            y,
            gene_set,
            design=x,
            contrast=contrast,
            set_statistic=statistic,
            n_rotations=499,
            rng=Replay(draws),
        )
        minus = roast(
            y,
            gene_set,
            design=x,
            contrast=-contrast,
            set_statistic=statistic,
            n_rotations=499,
            rng=Replay(flipped),
        )
        assert_allclose(
            roast_p_values(minus), roast_p_values(plus)[[1, 0, 2, 3]], atol=EXACT
        )
        assert_allclose(
            active_proportions(minus), active_proportions(plus)[[1, 0, 2, 3]]
        )


def test_the_order_of_the_genes_and_their_scale_change_nothing() -> None:
    # romer's shrinkage bounds the prior variance of the contrast in log-expression
    # units, as limma's B-statistics do, so only mroast is invariant to the scale.
    y, x = three_groups(5)
    genes = np.array([f"g{i}" for i in range(60)])
    sets = {"a": genes[:12], "b": genes[[3, 30, 31, 40, 50]], "c": genes[20:45]}
    order = np.random.default_rng(3).permutation(60)
    for test, options, rescaled in (
        (mroast, {"set_statistic": "msq"}, 10.0 * y + np.arange(60.0)),
        (romer, {}, y + np.arange(60.0)),
    ):
        original = test(
            y,
            sets,
            design=x,
            contrast=2,
            feature_ids=genes,
            rng=4,
            n_rotations=301,
            **options,
        )
        shuffled = test(
            y[:, order],
            {k: v[::-1] for k, v in sets.items()},
            design=x,
            contrast=2,
            feature_ids=genes[order],
            rng=4,
            n_rotations=301,
            **options,
        )
        scaled = test(
            rescaled,
            sets,
            design=x,
            contrast=2,
            feature_ids=genes,
            rng=4,
            n_rotations=301,
            **options,
        )
        for other in (shuffled, scaled):
            for column in original.columns:
                if original[column].dtype.kind == "f":
                    assert_allclose(other[column], original[column], rtol=1e-12)
                else:
                    assert_array_equal(other[column], original[column])


@pytest.mark.parametrize("statistic", ["mean", "floormean", "mean50", "msq"])
def test_only_the_relative_size_and_sign_of_contribution_weights_matter(
    statistic: str,
) -> None:
    # Multiplying every weight by 10 changes no statistic; limma's msq is the
    # exception (a deliberate difference). Weights of +1 and -1 equal a set without
    # weights whose second half is the negated genes: -1 on the genes 10 to 14, which
    # go down, makes them count as up.
    y, x = three_groups(6)
    weights = {g: w for g, w in zip(range(15), [1.0] * 10 + [-1.0] * 5, strict=True)}
    sizes = {g: w for g, w in zip(range(8), [0.5, 2, 1, 3, 1, 0.2, 1, 4], strict=True)}
    for gene_set in (weights, sizes):
        table = roast(y, gene_set, design=x, contrast=1, set_statistic=statistic, rng=9)
        scaled = {g: 10.0 * w for g, w in gene_set.items()}
        again = roast(y, scaled, design=x, contrast=1, set_statistic=statistic, rng=9)
        assert_allclose(roast_p_values(again), roast_p_values(table), atol=EXACT)
    signed = roast(y, weights, design=x, contrast=1, set_statistic=statistic, rng=9)
    negated = y.copy()
    negated[:, 10:15] = -negated[:, 10:15]
    plain = roast(
        negated, list(range(15)), design=x, contrast=1, set_statistic=statistic, rng=9
    )
    assert_allclose(roast_p_values(signed), roast_p_values(plain), atol=EXACT)
    assert_array_equal(active_proportions(signed), active_proportions(plain))
    assert active_proportions(signed)[0] == 0.0  # the down genes now count as up


def test_genome_wide_weights_equal_the_same_weights_given_by_set() -> None:
    y, x = three_groups(7)
    weights = np.random.default_rng(2).choice([-1.0, 0.5, 2.0], 60)
    sets = {"a": list(range(12)), "b": list(range(20, 40))}
    by_gene = mroast(y, sets, design=x, contrast=1, gene_weights=weights, rng=3)
    by_set = {name: {g: weights[g] for g in rows} for name, rows in sets.items()}
    assert_tables_equal(mroast(y, by_set, design=x, contrast=1, rng=3), by_gene)


# Limits and closed forms.


def test_mroast_with_many_rotations_approaches_fry_with_equal_variances() -> None:
    # With equal residual variances (an infinite prior df) fry's p-value is the
    # limit of the mean statistic's two-sided p-value over infinitely many rotations
    # (Giner and Smyth 2016); 20,000 rotations give a standard error below 0.0036.
    rng = np.random.default_rng(10)
    x = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    q = np.linalg.qr(x, mode="complete")[0]
    noise = rng.normal(size=(8, 40))
    residual = q[:, 2:].T @ noise
    residual *= np.sqrt(6.0 / np.sum(residual**2, axis=0))
    y = q[:, :2] @ (q[:, :2].T @ noise) + q[:, 2:] @ residual
    y[4:, :6] += 0.6
    sets = {"changed": list(range(8)), "null": list(range(10, 30)), "pair": [3, 20]}
    limit = fry(y, sets, design=x, contrast=1, standardize="none")
    rotated = mroast(y, sets, design=x, contrast=1, n_rotations=20_000, rng=11)
    p = limit["p_value"]
    bound = 4 * np.sqrt(p * (1 - p) / 20_000) + 2 / 20_001  # the resolution of p
    assert np.all(np.abs(rotated["p_value"] - p) <= bound)
    assert list(rotated["direction"]) == list(limit["direction"])


def test_the_smallest_p_values_and_one_rotation() -> None:
    # A set of strongly changed genes beats every rotation: 1 / (B + 1), and
    # 1 / (2B + 1) for the up alternative.
    y, x = three_groups(8)
    y[3:6, :10] += 20.0
    table = roast(y, list(range(10)), design=x, contrast=1, n_rotations=99, rng=1)
    p_values = roast_p_values(table)
    assert_allclose(p_values[[1, 2, 3]], [1 / 199, 1 / 100, 1 / 100])
    assert p_values[0] == 1.0  # no rotated mean is lower than its negative
    one = roast(y, list(range(10)), design=x, contrast=1, n_rotations=1, rng=1)
    assert set(roast_p_values(one)[2:].tolist()) <= {0.5, 1.0}


def test_a_set_of_every_gene_has_romer_p_values_of_one() -> None:
    # The ranks of all genes always sum to G (G + 1) / 2.
    y, x = three_groups(9)
    for statistic in ("mean", "floormean"):
        table = romer(
            y,
            {"all": list(range(60))},
            design=x,
            contrast=1,
            set_statistic=statistic,
            n_rotations=99,
            rng=1,
        )
        assert_array_equal(
            [table["up_p_value"], table["down_p_value"], table["mixed_p_value"]], 1.0
        )


def test_romer_up_and_down_p_values_of_mean_ranks_add_to_just_over_one() -> None:
    # The down rank of a gene is G + 1 minus its up rank, so a rotation at least as
    # extreme up is at most as extreme down: p_up + p_down = 1 + (1 + ties) / (B + 1).
    y, x = three_groups(10)
    table = romer(y, SETS, design=x, contrast=1, n_rotations=199, rng=2)
    total = table["up_p_value"] + table["down_p_value"]
    assert np.all(total >= 1.0 + 1.0 / 200 - 1e-12)
    assert np.all(total <= 1.0 + 10.0 / 200)


def test_bailey_z_scores_match_limmas_zscoreT() -> None:
    # limma's zscoreT(t, df, approx = TRUE) on a grid, from R (Bailey 1980).
    t = np.array([2.0, 5.0, 0.5, 20.0, -1.0, 0.1])
    df = np.array([1.0, 10.0, 4.0, 50.0, 2.5, 1e5])
    expected = [
        1.05798120585186894,
        3.458737089579013713,
        0.463613084798031316,
        10.434380377039330412,
        -0.837035600640890332,
        0.099999747500320993,
    ]
    assert_allclose(_bailey_z_scores(t, df), expected, rtol=1e-14)
    # Infinite df gives t, and |t| beyond the overflow of t^2 stays finite.
    assert _bailey_z_scores(np.array([3.0]), np.array([np.inf]))[0] == 3.0
    huge = _bailey_z_scores(np.array([1e200]), np.array([5.0]))[0]
    expected_huge = (
        41 / 49 * math.sqrt((5 + 19 / 12) * (400 * math.log(10) - math.log(5 + 1 / 12)))
    )
    assert_allclose(huge, expected_huge, rtol=1e-14)


def test_the_proportion_of_true_null_hypotheses() -> None:
    # Values of limma's propTrueNull(p, method = "lfdr"), from R; evenly spaced
    # p-values i / (n + 1) give n / (n + 1).
    for p, expected in (
        ([0.5], 0.5),
        ([0.2, 0.8], 2 / 3),
        ([0.99, 0.99], 0.99333333333333329),
        ([0.1, 0.5, 0.9], 0.75),
        ([0.001, 0.01, 0.2, 0.9, 0.95, 0.99], 0.77171428571428569),
        (np.arange(1, 20) / 20, 19 / 20),
    ):
        assert_allclose(_proportion_true_null(np.asarray(p)), expected, rtol=1e-15)


def test_adjusted_p_values_use_mid_p_values_and_never_fall_below_the_p_values() -> None:
    y, x = three_groups(11)
    sets = {f"s{i}": list(range(i, i + 6)) for i in range(0, 48, 4)}
    plain = mroast(y, sets, design=x, contrast=1, n_rotations=99, rng=4, mid_p=False)
    assert_allclose(plain["adjusted_p_value"], adjust_p_values(plain["p_value"], "bh"))
    mid = mroast(y, sets, design=x, contrast=1, n_rotations=99, rng=4)
    assert np.all(mid["adjusted_p_value"] >= mid["p_value"])
    assert np.all(mid["adjusted_p_value"] <= plain["adjusted_p_value"])


# Edge cases.


def test_empty_sets_and_sets_of_weight_zero_have_no_p_value() -> None:
    y, x = three_groups(12)
    genes = np.array([f"g{i}" for i in range(60)])
    sets = {
        "empty": ["absent"],
        "a": genes[:10],
        "zero": {"g20": 0.0},
        "b": genes[30:40],
    }
    table = mroast(y, sets, design=x, contrast=1, feature_ids=genes, rng=1)
    assert_array_equal(table["n_genes"], [0, 10, 0, 10])
    assert np.isnan(table["p_value"][[0, 2]]).all()
    assert np.isnan(table["proportion_up"][[0, 2]]).all()
    assert table["direction"][0] is None and table["direction"][2] is None
    alone = mroast(
        y,
        {"a": genes[:10], "b": genes[30:40]},
        design=x,
        contrast=1,
        feature_ids=genes,
        rng=1,
    )
    assert_allclose(table["adjusted_p_value"][[1, 3]], alone["adjusted_p_value"])
    assert np.isnan(
        romer(y, {"e": []}, design=x, contrast=1, rng=1, n_rotations=9)["up_p_value"][0]
    )
    empty = roast(y, [], design=x, contrast=1, rng=1)
    assert np.isnan(roast_p_values(empty)).all() and empty["n_genes"][0] == 0


@pytest.mark.parametrize("statistic", ["mean", "floormean", "mean50", "msq"])
def test_genes_listed_twice_count_once_and_weight_zero_leaves_the_set(
    statistic: str,
) -> None:
    # A gene of weight zero does not count among mean50's largest values either
    # (limma keeps it there and in its number of genes).
    y, x = three_groups(13)
    once = roast(y, [1, 2, 3, 8], design=x, contrast=1, set_statistic=statistic, rng=2)
    twice = roast(
        y, [3, 1, 2, 3, 8, 1], design=x, contrast=1, set_statistic=statistic, rng=2
    )
    assert_tables_equal(once, twice)
    weights = {1: 1.0, 2: 1.0, 3: 1.0, 8: 1.0, 9: 0.0, 40: 0.0}
    dropped = roast(y, weights, design=x, contrast=1, set_statistic=statistic, rng=2)
    assert_tables_equal(once, dropped)


def test_a_gene_without_a_change_has_equal_one_sided_p_values_and_goes_down() -> None:
    # With a zero contrast effect every rotation and its negative split evenly; limma
    # then reports "Down" (found by probing).
    x = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])
    y = np.random.default_rng(14).normal(size=(6, 30))
    y[:, 0] = [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]
    table = roast(y, [0], design=x, contrast=1, rng=3, n_rotations=99)
    p_values = roast_p_values(table)
    assert p_values[0] == p_values[1] == 100 / 199
    assert p_values[2] == 1.0
    assert (
        mroast(y, {"zero": [0]}, design=x, contrast=1, rng=3)["direction"][0] == "down"
    )


def test_a_voom_result_and_a_design_with_named_contrasts_work_together() -> None:
    rng = np.random.default_rng(15)
    counts = rng.poisson(rng.uniform(20, 500, 80), size=(8, 80))
    counts[4:, :10] *= 3
    design = Design(
        {"group": np.repeat(["a", "b"], 4)}, "~ group", contrasts={"b_vs_a": "groupb"}
    )
    result = voom(counts, design)
    sets = {"up": list(range(10)), "null": list(range(40, 60))}
    direct = mroast(result, sets, design=design, contrast="b_vs_a", rng=1)
    explicit = mroast(
        result.log2_cpm,
        sets,
        design=design,
        contrast="groupb",
        weights=result.weights,
        rng=1,
    )
    assert_tables_equal(direct, explicit)
    assert direct["direction"][0] == "up" and direct["p_value"][0] < 0.01
    with pytest.raises(ValueError, match="sample weights only"):
        romer(result, sets, design=design, contrast="b_vs_a")


# Arguments.


def test_invalid_arguments_are_refused_with_a_reason() -> None:
    y, x = three_groups(16)
    sets = {"a": [0, 1, 2]}
    cases: list[tuple[Any, dict[str, Any], type[Exception], str]] = [
        (mroast, {"set_statistic": "median"}, ValueError, "set_statistic must be one"),
        (romer, {"set_statistic": "msq"}, ValueError, "set_statistic must be one"),
        (mroast, {"n_rotations": 0}, ValueError, "at least 1"),
        (mroast, {"n_rotations": 9.5}, TypeError, "positive integer"),
        (mroast, {"n_rotations": True}, TypeError, "positive integer"),
        (mroast, {"rng": -1}, ValueError, "non-negative seed"),
        (mroast, {"rng": "seed"}, TypeError, "numpy.random.Generator"),
        (mroast, {"gene_weights": np.ones(5)}, ValueError, "one weight per gene"),
        (mroast, {"gene_weights": np.full(60, np.nan)}, ValueError, "finite"),
        (mroast, {"gene_weights": np.array(["a"] * 60)}, TypeError, "real numbers"),
        (
            mroast,
            {"gene_sets": {"a": {0: 1.0}}, "gene_weights": np.ones(60)},
            ValueError,
            "cannot be given too",
        ),  # fmt: skip
        (mroast, {"gene_sets": {"a": {0: "up"}}}, TypeError, "real numbers"),
        (mroast, {"gene_sets": {"a": {0: math.inf}}}, ValueError, "must be finite"),
        (mroast, {"gene_sets": {"a": {60: 1.0}}}, ValueError, "position 60"),
        (mroast, {"mid_p": "yes"}, TypeError, "mid_p must be a bool"),
        (mroast, {"trend": 1}, TypeError, "trend must be a bool"),
        (romer, {"shrink_contrast_effects": None}, TypeError, "must be a bool"),
        (romer, {"weights": np.ones((9, 60))}, ValueError, "sample weights only"),
        (romer, {"gene_sets": {"a": {0: 1.0}}}, TypeError, "romer does not take"),
        (romer, {"n_threads": 0}, ValueError, "n_threads must be at least 1"),
        (mroast, {"gene_sets": {}}, ValueError, "gene_sets is empty"),
        (mroast, {"design": np.eye(9)}, ValueError, "no residual degrees of freedom"),
    ]
    for test, changes, error, message in cases:
        options = {"data": y, "design": x, "contrast": 1, "gene_sets": sets} | changes
        with pytest.raises(error, match=message):
            test(**options)
    with pytest.raises(ValueError, match="no gene of gene_set is among feature_ids"):
        roast(
            y,
            ["absent"],
            design=x,
            contrast=1,
            feature_ids=[f"g{i}" for i in range(60)],
        )


# Results of differential expression.

RESULT_TESTS = ((roast, SYMBOL_SETS["up"]), (mroast, SYMBOL_SETS), (romer, SYMBOL_SETS))


@pytest.mark.parametrize("contrast", ["stim_vs_ctrl", "D2_vs_D1"])
def test_roast_and_mroast_of_a_limma_voom_result_test_the_voom_values_of_its_fit(
    contrast: str,
) -> None:
    result = results()["limma_voom"]
    fit, read = fitted(result, "B", contrast)
    moderation = {"trend": False, "robust": result.settings["robust"]}
    tested = {"group": "B", "contrast": contrast, "id_column": "symbol"}
    rotations = {"n_rotations": 99, "rng": 5}
    for name, genes in SYMBOL_SETS.items():
        table = roast(result, genes, **tested, **rotations)
        expected = roast(fit.voom, genes, **read, **moderation, **rotations)
        assert table.equals(expected), name
    table = mroast(result, SYMBOL_SETS, **tested, **rotations)
    expected = mroast(fit.voom, SYMBOL_SETS, **read, **moderation, **rotations)
    assert table.equals(expected)


@pytest.mark.parametrize("contrast", ["stim_vs_ctrl", "D2_vs_D1"])
def test_romer_of_a_limma_trend_result_tests_its_fit_with_its_sample_weights(
    contrast: str,
) -> None:
    result = results()["limma_trend"]
    fit, read = fitted(result, "B", contrast)
    assert fit.weights is not None and fit.weights.shape == (len(fit.samples),)
    rotations = {"n_rotations": 99, "rng": 5}
    table = romer(
        result,
        SYMBOL_SETS,
        group="B",
        contrast=contrast,
        id_column="symbol",
        **rotations,
    )
    expected = romer(
        fit.log2_cpm, SYMBOL_SETS, **read, weights=fit.weights, **rotations
    )
    assert table.equals(expected)


def test_romer_refuses_the_observation_weights_of_voom() -> None:
    result = results()["limma_voom"]
    fit, read = fitted(result, "B", "stim_vs_ctrl")
    message = re.escape(
        "romer takes sample weights only, of shape (n_samples,); limma's romer has no "
        "observation weights, so a VoomResult, a result of limma_voom or a matrix of "
        "weights cannot be used"
    )
    with pytest.raises(ValueError, match=message):
        romer(
            result, SYMBOL_SETS, group="B", contrast="stim_vs_ctrl", id_column="symbol"
        )
    with pytest.raises(ValueError, match=message):
        romer(fit.voom, SYMBOL_SETS, **read)


def test_a_result_gives_the_design_weights_and_ids_of_its_fit() -> None:
    result = results()["limma_voom"]
    fit, read = fitted(result, "B", "stim_vs_ctrl")
    tested = {"group": "B", "contrast": "stim_vs_ctrl", "id_column": "symbol"}
    for test, sets in RESULT_TESTS:
        for name, value, owner in (
            ("design", read["design"], "the design of its group's fit"),
            ("feature_ids", read["feature_ids"], "the ids of the column id_column"),
            ("weights", fit.weights, "the weights of its group's fit"),
        ):
            # romer takes sample weights only, so no VoomResult.
            inputs = "an array" if test is romer else "an array or a VoomResult"
            message = (
                f"{name} applies to {inputs}; a differential-expression result "
                f"gives {owner}"
            )
            with pytest.raises(ValueError, match=message):
                test(result, sets, **tested, **{name: value})
        # A group names a group of a result, not of an array.
        with pytest.raises(
            ValueError,
            match="group names a group of a differential-expression result, and data "
            "is a ndarray",
        ):
            test(fit.log2_cpm, sets, group="B", **read)


def test_a_result_of_dream_voom_is_refused() -> None:
    result = results()["dream_voom"]
    for test, sets in RESULT_TESTS:
        with pytest.raises(
            ValueError,
            match=f"{test.__name__} tests the contrast and residual effects of a "
            "linear model, and this result is from dream_voom",
        ):
            test(result, sets, contrast="stim_vs_ctrl", id_column="symbol")
