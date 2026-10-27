"""Tests of the activities of regulators and pathways: ULM, MLM and the weighted
aggregate.

The activities are compared with decoupler 2.2.0's, which validation/activity.py wrote
to tests/data/activity, its permutations replayed; with values derived by hand on
small networks; with the exact distribution of a weighted mean over every permutation
of a few genes; and with invariants that a wrong implementation would break. The bar
plot is compared with decoupler's and the heatmap read back from what it draws.
"""

import dataclasses
import doctest
import itertools
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import (
    Activities,
    Cube,
    Design,
    GeneSets,
    Network,
    Table,
    activity,
    limma_voom,
    mlm,
    ulm,
    weighted_aggregate,
)
from cytepool.normalization import filter_by_expression, log2_cpm, tmm_factors
from cytepool.plotting import plot_activities, plot_top_activities
from cytepool.stats.linear_models import fit_linear_model

DATA = Path(__file__).parent / "data" / "activity"
FIXTURE = json.loads((DATA / "decoupler.json").read_text())
ARRAYS = np.load(DATA / "decoupler.npz")
EXACT = 1e-10  # least squares and sums in closed form
CASES = {case["name"]: case for case in FIXTURE["cases"]}


def _network(name: str) -> Network:
    return Network(Table(FIXTURE["networks"][name]))


def _data(name: str) -> tuple[np.ndarray, list[str], list[str]]:
    described = FIXTURE["data"][name]
    return ARRAYS[name], described["profiles"], described["genes"]


def _missing() -> np.ndarray:
    """The samples with the values that decoupler did not see missing: those of the
    first two samples outside the genes of their own profiles."""
    values, _, genes = _data("samples")
    missing = values.copy()
    for row in range(2):
        observed = set(_data(f"missing_{row}")[2])
        missing[row, [gene not in observed for gene in genes]] = np.nan
    return missing


def _scaled(ours: np.ndarray, theirs: np.ndarray) -> float:
    assert_array_equal(np.isnan(ours), np.isnan(theirs))
    return float(np.nanmax(np.abs(ours - theirs)) / np.nanmax(np.abs(theirs)))


def _relative(ours: np.ndarray, theirs: np.ndarray) -> float:
    assert_array_equal(np.isnan(ours), np.isnan(theirs))
    return float(np.nanmax(np.abs(ours - theirs) / np.abs(theirs)))


def _run(case: dict) -> tuple[Activities, list[int], list[int]]:
    """CytePool's activities of a case, the rows decoupler scored, and the columns of
    decoupler's sources."""
    function = {"ulm": ulm, "mlm": mlm}[case["method"]]
    values, profiles, genes = _data(case["data"])
    rows = list(range(len(profiles)))
    if case["data"].startswith("missing_"):
        values, profiles, genes = _missing(), *_data("samples")[1:]
        rows = [int(case["data"][-1])]
    result = function(
        values,
        _network(case["network"]),
        feature_ids=genes,
        sample_ids=profiles,
        min_targets=case["tmin"],
    )
    found = FIXTURE["found"][case["name"]]
    columns = [result.sources.index(source) for source in found["sources"]]
    return result, rows, columns


LINEAR = [
    pytest.param(case, id=case["name"])
    for case in FIXTURE["cases"]
    if case["method"] in ("ulm", "mlm")
]


# Against decoupler.


@pytest.mark.parametrize("case", LINEAR)
def test_the_linear_models_are_those_of_decoupler(case: dict) -> None:
    result, rows, columns = _run(case)
    name = case["name"]
    pick = np.ix_(rows, columns)

    # The sources scored in these profiles are decoupler's, after its tmin.
    scored = ~np.all(np.isnan(result.scores[rows]), axis=0)
    assert sorted(np.array(result.sources)[scored]) == sorted(
        FIXTURE["found"][name]["sources"]
    )
    assert _scaled(result.scores[pick], ARRAYS[f"{name}/raw_scores"]) < EXACT
    assert _scaled(result.estimates[pick], ARRAYS[f"{name}/estimates"]) < EXACT
    if case["method"] == "ulm":
        assert _relative(result.p_value[pick], ARRAYS[f"{name}/raw_p"]) < EXACT
        # Benjamini and Hochberg's over the sources of each profile, by scipy.
        bh = ARRAYS[f"{name}/bh"]
        assert _relative(result.adjusted_p_value[pick], bh) < EXACT
        # decoupler keeps them in single precision.
        single = result.adjusted_p_value[pick].astype(np.float32)
        assert_array_equal(single, ARRAYS[f"{name}/padj"].astype(np.float32))
    else:
        # decoupler's own are 1 - cdf; scipy's upper tail of its t-statistics.
        assert _relative(result.p_value[pick], ARRAYS[f"{name}/t_sf"]) < EXACT


@pytest.mark.parametrize("name", ["weighted mean", "weighted sum"])
def test_the_weighted_aggregate_is_that_of_decoupler(name: str) -> None:
    case = CASES[name]
    values, profiles, genes = _data(case["data"])
    statistic = "sum" if case["options"].get("fun") == "wsum" else "mean"
    result = weighted_aggregate(
        values,
        _network(case["network"]),
        feature_ids=genes,
        statistic=statistic,
        n_permutations=0,
    )
    columns = [result.sources.index(s) for s in FIXTURE["found"][name]["sources"]]
    assert _scaled(result.scores[:, columns], ARRAYS[f"{name}/raw_scores"]) < EXACT
    assert_array_equal(result.scores, result.estimates)
    assert np.isnan(result.p_value).all()


class _Replay(np.random.Generator):
    """A generator whose shuffles give the permutations decoupler drew."""

    def __init__(self, orders: np.ndarray) -> None:
        super().__init__(np.random.PCG64(0))
        self.orders = iter(orders)

    def shuffle(self, x: object, axis: int = 0) -> None:
        x[:] = next(self.orders)  # type: ignore[index]


def test_normalized_scores_are_decouplers_given_its_permutations() -> None:
    name = "weighted mean, 100 permutations"
    case = CASES[name]
    values, profiles, genes = _data(case["data"])
    found = FIXTURE["found"][name]
    # decoupler's permutations are of the genes in its own order, the zero genes out.
    order = [genes.index(gene) for gene in found["genes"]]
    permutations = ARRAYS[f"{name}/permutations"].astype(np.int64)
    result = weighted_aggregate(
        values[:, order],
        _network(case["network"]),
        feature_ids=found["genes"],
        n_permutations=permutations.shape[0],
        rng=_Replay(permutations),
    )
    columns = [result.sources.index(s) for s in found["sources"]]
    assert _scaled(result.scores[:, columns], ARRAYS[f"{name}/raw_scores"]) < EXACT
    # The p-values differ from decoupler's on purpose (see the module docstring).
    assert np.all((result.p_value > 0.0) & (result.p_value <= 1.0))


def test_the_bar_plot_is_decouplers() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    result, _, _ = _run(CASES["ulm"])
    ax = plot_top_activities(
        result, "s4", top=15, ax=matplotlib.figure.Figure().subplots()
    )
    names = [label.get_text() for label in ax.get_yticklabels()]
    assert names == FIXTURE["found"]["ulm"]["bar_names"]
    widths = np.array([bar.get_width() for bar in ax.patches])
    assert _scaled(widths, ARRAYS["ulm/bar_widths"]) < EXACT


# Values derived by hand.

# Four genes, one source of two targets, A: g1 (weight 1) and g2 (weight 2). With
# y = (1, 2, 3, 5): x = (1, 2, 0, 0), mean 3/4; y's mean 11/4. Sxx = 11/4, Sxy =
# -13/4, Syy = 35/4; slope -13/11, residual sum of squares 35/4 - 169/44 = 54/11 on 2
# degrees of freedom, t = slope / sqrt(54/11 / (2 * 11/4)) = -13 / sqrt(108). With 2
# degrees of freedom P(|T| > t) = 1 - t / sqrt(t^2 + 2) = 1 - 13 / sqrt(385).
FOUR = ["g1", "g2", "g3", "g4"]
Y = np.array([1.0, 2.0, 3.0, 5.0])
PAIR = Network({"A": {"g1": 1.0, "g2": 2.0}})


def test_ulm_of_a_source_with_two_targets_on_three_samples() -> None:
    # The second sample is 2y + 7, which doubles the slope and keeps t; the third,
    # -y, negates both.
    values = np.array([Y, 2.0 * Y + 7.0, -Y])
    result = ulm(values, PAIR, feature_ids=FOUR, min_targets=2)

    t = -13.0 / math.sqrt(108.0)
    assert_allclose(result.scores[:, 0], [t, t, -t], rtol=1e-14)
    assert_allclose(result.estimates[:, 0], [-13 / 11, -26 / 11, 13 / 11], rtol=1e-14)
    p = 1.0 - 13.0 / math.sqrt(385.0)
    assert_allclose(result.p_value[:, 0], [p, p, p], rtol=1e-13)
    # One source per profile: nothing to adjust.
    assert_allclose(result.adjusted_p_value, result.p_value, rtol=0)
    assert_array_equal(result.n_targets, [[2], [2], [2]])


def test_mlm_of_a_network_of_one_source_is_ulm() -> None:
    values = np.array([Y, 2.0 * Y + 7.0, -Y])
    single = mlm(values, PAIR, feature_ids=FOUR, min_targets=2)
    assert_allclose(single.scores[:, 0], -13.0 / math.sqrt(108.0) * np.array(
        [1.0, 1.0, -1.0]), rtol=1e-13)  # fmt: skip
    assert_allclose(single.estimates[:, 0], [-13 / 11, -26 / 11, 13 / 11], rtol=1e-13)


def test_the_weighted_mean_and_sum_by_hand() -> None:
    values = np.array([Y])
    mean = weighted_aggregate(values, PAIR, feature_ids=FOUR, min_targets=2,
                              n_permutations=0)  # fmt: skip
    total = weighted_aggregate(values, PAIR, feature_ids=FOUR, min_targets=2,
                               statistic="sum", n_permutations=0)  # fmt: skip
    # (1 * 1 + 2 * 2) / (|1| + |2|) and 1 * 1 + 2 * 2.
    assert_allclose(mean.scores, [[5.0 / 3.0]], rtol=1e-15)
    assert_allclose(total.scores, [[5.0]], rtol=1e-15)


def test_ulm_with_weights_of_one_sign_is_the_two_sample_t_test() -> None:
    # Regressing on a 0/1 indicator gives Student's pooled two-sample t-statistic of
    # the targets against the other genes.
    rng = np.random.default_rng(4)
    y = rng.normal(size=(2, 30))
    genes = [f"g{j}" for j in range(30)]
    targets = genes[:7]
    result = ulm(y, {"set": targets}, feature_ids=genes)
    inside, outside = y[:, :7], y[:, 7:]
    pooled = np.sqrt(
        (((inside - inside.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
         + ((outside - outside.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)) / 28
    )  # fmt: skip
    t = (inside.mean(axis=1) - outside.mean(axis=1)) / (
        pooled * math.sqrt(1 / 7 + 1 / 23)
    )
    assert_allclose(result.scores[:, 0], t, rtol=1e-12)
    # A GeneSets is a network of the same sets.
    as_sets = ulm(y, GeneSets({"set": targets}), feature_ids=genes)
    assert_array_equal(as_sets.scores, result.scores)


def test_permutations_estimate_the_exact_null_distribution() -> None:
    # Five genes, a source on two of them: over the 120 permutations of the genes,
    # each equally likely, the weighted mean has an exact mean and standard deviation,
    # which 5,000 random permutations estimate within Monte Carlo error.
    y = np.array([[0.3, 2.0, -1.0, 4.0, 1.5]])
    genes = [f"g{j}" for j in range(5)]
    network = {"A": {"g0": 1.0, "g1": -0.5}}
    nulls = np.array(
        [(y[0, p[0]] * 1.0 - 0.5 * y[0, p[1]]) / 1.5
         for p in itertools.permutations(range(5))]
    )  # fmt: skip
    observed = (0.3 - 1.0) / 1.5
    exact = (observed - nulls.mean()) / nulls.std(ddof=0)
    tails = min(np.mean(nulls >= observed), np.mean(nulls <= observed))
    result = weighted_aggregate(y, network, feature_ids=genes, min_targets=2,
                                n_permutations=5000, rng=1)  # fmt: skip
    assert_allclose(result.estimates, [[observed]], rtol=1e-15)
    # Monte Carlo error: the mean and sd of 5,000 draws, about 2% of the score.
    assert_allclose(result.scores[0, 0], exact, rtol=0.05)
    assert abs(result.p_value[0, 0] - min(1.0, 2 * tails)) < 0.03


# Invariants.

RNG = np.random.default_rng(11)
GENES = [f"G{j}" for j in range(60)]
VALUES = RNG.normal(size=(4, 60))
NETWORK = Network(
    {
        f"S{k}": {gene: float(RNG.choice([-2.0, -1.0, 0.5, 1.0])) for gene in
                  RNG.choice(GENES, 8, replace=False)}
        for k in range(6)
    }
)  # fmt: skip
FUNCTIONS = [
    pytest.param(ulm, id="ulm"),
    pytest.param(mlm, id="mlm"),
    pytest.param(
        lambda *a, **k: weighted_aggregate(*a, n_permutations=0, **k),
        id="weighted_aggregate",
    ),
]


@pytest.mark.parametrize("function", FUNCTIONS)
def test_the_order_of_genes_sources_and_samples_does_not_matter(function) -> None:
    base = function(VALUES, NETWORK, feature_ids=GENES)
    genes = RNG.permutation(60)
    shuffled = function(
        VALUES[:, genes], NETWORK, feature_ids=[GENES[j] for j in genes]
    )
    assert_allclose(shuffled.scores, base.scores, rtol=1e-12, atol=1e-13)
    # The sources keep the order of the network.
    reversed_network = Network(NETWORK.edges.take(np.arange(len(NETWORK.edges))[::-1]))
    flipped = function(VALUES, reversed_network, feature_ids=GENES)
    assert flipped.sources == base.sources[::-1]
    assert_allclose(flipped.scores[:, ::-1], base.scores, rtol=1e-12, atol=1e-13)
    rows = [2, 0, 3, 1]
    by_rows = function(VALUES[rows], NETWORK, feature_ids=GENES)
    assert_allclose(by_rows.scores, base.scores[rows], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_negating_a_sources_weights_negates_its_activity(function) -> None:
    edges = NETWORK.edges
    weights = np.where(edges["source"] == "S2", -edges["weight"], edges["weight"])
    negated = Network(edges.with_columns({"weight": weights}))
    base = function(VALUES, NETWORK, feature_ids=GENES)
    flipped = function(VALUES, negated, feature_ids=GENES)
    column = base.sources.index("S2")
    assert_allclose(flipped.scores[:, column], -base.scores[:, column], rtol=1e-12)
    assert_allclose(flipped.estimates[:, column], -base.estimates[:, column],
                    rtol=1e-12)  # fmt: skip
    others = [j for j in range(len(base.sources)) if j != column]
    assert_allclose(flipped.scores[:, others], base.scores[:, others], rtol=1e-12,
                    atol=1e-13)  # fmt: skip


def test_scaling_a_profile_scales_ulms_estimates_and_keeps_its_t() -> None:
    base = ulm(VALUES, NETWORK, feature_ids=GENES)
    scaled = ulm(3.0 * VALUES - 5.0, NETWORK, feature_ids=GENES)
    assert_allclose(scaled.estimates, 3.0 * base.estimates, rtol=1e-12)
    assert_allclose(scaled.scores, base.scores, rtol=1e-11)
    assert_allclose(scaled.p_value, base.p_value, rtol=1e-10)


def test_min_targets_and_missing_values() -> None:
    # S0 has 8 targets; with 3 of them missing in the first sample, it has 5 there.
    targets = list(NETWORK["S0"])
    values = VALUES.copy()
    values[0, [GENES.index(gene) for gene in targets[:3]]] = np.nan
    result = ulm(values, NETWORK, feature_ids=GENES, min_targets=6)
    column = result.sources.index("S0")
    assert result.n_targets[0, column] == 5 and result.n_targets[1, column] == 8
    assert np.isnan(result.scores[0, column]) and np.isfinite(result.scores[1, column])
    # The first sample scored alone over its observed genes gives the same score.
    alone = ulm(values[:1, ~np.isnan(values[0])], NETWORK,
                feature_ids=[g for g, v in zip(GENES, values[0], strict=True)
                          if not np.isnan(v)], min_targets=5)  # fmt: skip
    again = ulm(values, NETWORK, feature_ids=GENES, min_targets=5)
    assert_allclose(again.scores[0], alone.scores[0], rtol=1e-12)
    with pytest.raises(ValueError, match="no source of the network has 9 targets"):
        ulm(values, NETWORK, feature_ids=GENES, min_targets=9)


def test_genes_zero_everywhere_and_without_ids_are_left_out() -> None:
    values = np.hstack([VALUES, np.zeros((4, 3))])
    genes = [*GENES, "Z1", "Z2", None]
    zeros = ulm(values, Network({**NETWORK, "Z": {"Z1": 1.0, "Z2": 1.0}}),
                feature_ids=genes)  # fmt: skip
    base = ulm(VALUES, NETWORK, feature_ids=GENES)
    assert "Z" not in zeros.sources
    assert_allclose(zeros.scores, base.scores, rtol=1e-12)


def test_edges_of_weight_zero_are_not_targets() -> None:
    edges = NETWORK.edges
    padded = Network(
        Table(
            {
                "source": [*edges["source"].tolist(), "S0", "S0"],
                "target": [*edges["target"].tolist(), "G59", "G58"],
                "weight": [*edges["weight"].tolist(), 0.0, 0.0],
            }
        )
    )
    base = ulm(VALUES, NETWORK, feature_ids=GENES)
    padded_result = ulm(VALUES, padded, feature_ids=GENES)
    assert_allclose(padded_result.scores, base.scores, rtol=1e-12)
    assert_array_equal(padded_result.n_targets, base.n_targets)


# Profiles of cubes and of differential-expression results.


def _cube() -> Cube:
    rng = np.random.default_rng(2)
    means = np.exp(rng.uniform(2.0, 6.0, 60))
    counts = rng.negative_binomial(8, 8 / (8 + means), (2, 6, 60)).astype(np.uint32)
    counts[1, :, 50:] = 0  # not expressed in the second group
    return Cube(
        {"counts": counts},
        samples={"sample_id": [f"S{i}" for i in range(6)],
                 "condition": ["ctrl", "stim"] * 3},
        groups={"group_id": ["B", "T"]},
        features={"feature_id": GENES},
        n_cells=np.array([[30, 30]] * 5 + [[5, 30]]),
    )  # fmt: skip


def test_a_cubes_group_is_scored_as_its_log2_cpm() -> None:
    cube = _cube()
    result = ulm(cube, NETWORK, group="B")
    # Its samples with at least 10 cells, over the genes filterByExpr keeps.
    counts = cube.layer("counts", "B")[:5]
    keep = filter_by_expression(counts, sample_groups=np.zeros(5))
    values = log2_cpm(counts[:, keep], norm_factors=tmm_factors(counts[:, keep]),
                     prior_count=2.0)  # fmt: skip
    expected = ulm(values, NETWORK, feature_ids=np.array(GENES)[keep].tolist())
    assert_allclose(result.scores, expected.scores, rtol=1e-12)
    assert result.profiles["profile_id"].tolist() == [f"S{i}" for i in range(5)]
    assert result.profiles["condition"].tolist() == ["ctrl", "stim"] * 2 + ["ctrl"]
    assert result.settings["group"] == "B" and result.settings["min_targets"] == 5
    # The scores take a design built from the profiles, as pathway scores do.
    fit = fit_linear_model(result.scores, Design(result.profiles, "~ condition"))
    assert fit.coefficients.shape == (len(result.sources), 2)


def assert_pickles(record: Any) -> None:
    """A record comes back from pickle with equal fields, read-only arrays and
    settings, and the same provenance, or none."""
    copy = pickle.loads(pickle.dumps(record))
    for field in dataclasses.fields(record):
        mine, theirs = getattr(record, field.name), getattr(copy, field.name)
        if isinstance(mine, np.ndarray):
            assert_array_equal(theirs, mine)
            assert theirs.dtype == mine.dtype and not theirs.flags.writeable
        elif isinstance(mine, Table):
            assert theirs.equals(mine)
        elif field.name == "settings":
            assert dict(theirs) == dict(mine)
            with pytest.raises(TypeError):
                theirs["x"] = 1  # type: ignore[index]
        elif hasattr(mine, "edges"):  # a Network
            assert theirs.edges.equals(mine.edges)
        elif hasattr(mine, "items"):  # GeneSets
            assert dict(theirs.items()) == dict(mine.items())
        else:
            assert theirs == mine, field.name
    assert copy.provenance == record.provenance
    if record.provenance is None:
        assert "_provenance" not in vars(copy)


def test_activities_of_a_cube_record_the_cube_they_read() -> None:
    cube = _cube()
    result = ulm(cube, NETWORK, group="B")
    record = result.provenance
    assert record is not None
    assert record["method"] == "ulm" and record["layer"] == "counts"
    assert record["cube"]["fingerprint"] == cube.fingerprint()
    assert record["settings"] == result.settings
    assert json.loads(json.dumps(record)) == record
    assert mlm(cube, NETWORK, group="B").provenance["method"] == "mlm"
    # Activities of values or of statistics name no cube.
    of_values = ulm(VALUES, NETWORK, feature_ids=GENES)
    assert of_values.provenance is None
    tested = limma_voom(cube, Design(cube.samples, "~ condition"),
                        contrasts={"stim": "conditionstim"}, min_cells=0)  # fmt: skip
    assert ulm(tested, NETWORK, min_targets=3).provenance is None
    # They pickle, the provenance with them.
    assert_pickles(result)
    assert_pickles(of_values)
    assert_pickles(weighted_aggregate(cube, NETWORK, group="B", rng=1))


def test_each_group_and_contrast_of_a_result_is_a_profile() -> None:
    cube = _cube()
    result = limma_voom(cube, Design(cube.samples, "~ condition"),
                        contrasts={"stim": "conditionstim"}, min_cells=0)  # fmt: skip
    found = ulm(result, NETWORK, min_targets=3)
    assert found.profiles["profile_id"].tolist() == ["B", "T"]
    assert found.profiles["contrast"].tolist() == ["stim", "stim"]
    for row, group in enumerate(["B", "T"]):
        table = result.table(group, "stim")
        alone = ulm(table, NETWORK, min_targets=3)
        assert alone.profiles["profile_id"].tolist() == ["t"]
        columns = [found.sources.index(s) for s in alone.sources]
        assert_allclose(found.scores[row, columns], alone.scores[0], rtol=1e-12)
    # T tests fewer genes, so its sources have fewer targets.
    assert (found.n_targets[1] <= found.n_targets[0]).all()
    assert (found.n_targets[1] < found.n_targets[0]).any()
    only = ulm(result, NETWORK, group="T", min_targets=3)
    assert only.profiles["profile_id"].tolist() == ["T"]
    # A mapping of gene id to statistic is one profile too.
    table = result.table("B", "stim")
    mapping = dict(zip(table["feature_id"].tolist(), table["t"].tolist(), strict=True))
    as_mapping = ulm(mapping, NETWORK, min_targets=3)
    assert_allclose(as_mapping.scores[0], found.scores[0], rtol=1e-12)


# The record and its table.


def test_the_record_and_its_table() -> None:
    result = ulm(VALUES, NETWORK, feature_ids=GENES, sample_ids=["a", "b", "c", "d"])
    assert repr(result) == "Activities(method='ulm', n_profiles=4, n_sources=6)"
    assert result.sources == NETWORK.sources
    for array in (result.scores, result.estimates, result.p_value,
                  result.adjusted_p_value, result.n_targets):  # fmt: skip
        assert not array.flags.writeable
    table = result.table()
    assert table.columns == ("source", "profile_id", "n_targets", "score",
                             "estimate", "p_value", "adjusted_p_value")  # fmt: skip
    assert table["source"].tolist()[:5] == ["S0"] * 4 + ["S1"]
    assert table["profile_id"].tolist()[:5] == ["a", "b", "c", "d", "a"]
    assert_array_equal(table["score"], result.scores.T.ravel())
    # The network kept: the sources scored and their targets among the genes.
    assert result.network.edges.equals(NETWORK.edges)


def test_mlm_refuses_sources_it_cannot_tell_apart() -> None:
    twins = Network({**{s: dict(NETWORK[s]) for s in NETWORK},
                     "TWIN": dict(NETWORK["S1"])})  # fmt: skip
    with pytest.raises(ValueError, match="cannot tell the sources 'TWIN' apart"):
        mlm(VALUES, twins, feature_ids=GENES)
    # Three sources of one target each and an intercept use up four genes.
    with pytest.raises(ValueError, match="more genes than sources"):
        mlm(np.array([Y]), {"A": ["g1"], "B": ["g2"], "C": ["g3"]}, feature_ids=FOUR,
            min_targets=1)  # fmt: skip


def test_messages() -> None:
    with pytest.raises(ValueError, match="no target of the network is among the genes"):
        ulm(VALUES, {"A": ["X1", "X2"]}, feature_ids=GENES)
    with pytest.raises(TypeError, match="network must be a Network"):
        ulm(VALUES, 3, feature_ids=GENES)
    with pytest.raises(ValueError, match="feature_ids is needed with an array"):
        ulm(VALUES, NETWORK)
    with pytest.raises(ValueError, match="contrast applies to the statistics"):
        ulm(VALUES, NETWORK, feature_ids=GENES, contrast="stim")
    with pytest.raises(ValueError, match="feature_ids applies to an array or a cube"):
        ulm({"G1": 1.0, "G2": 2.0}, NETWORK, feature_ids=GENES)
    with pytest.raises(ValueError, match="min_targets must be at least 1"):
        ulm(VALUES, NETWORK, feature_ids=GENES, min_targets=0)
    with pytest.raises(ValueError, match="n_permutations must be 0, for no"):
        weighted_aggregate(VALUES, NETWORK, feature_ids=GENES, n_permutations=1)
    with pytest.raises(ValueError, match="statistic must be one of 'mean', 'sum'"):
        weighted_aggregate(VALUES, NETWORK, feature_ids=GENES, statistic="median")
    with pytest.raises(ValueError, match="To score a samples x genes table"):
        ulm(Table({"G1": [1.0, 2.0], "G2": [3.0, 4.0]}), NETWORK)
    with pytest.raises(ValueError, match="data holds infinite values"):
        ulm(np.where(VALUES > 2, np.inf, VALUES), NETWORK, feature_ids=GENES)


# The heatmap.


def test_the_heatmap_draws_the_scores_of_the_top_sources() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.figure

    result = ulm(VALUES, NETWORK, feature_ids=GENES, sample_ids=["a", "b", "c", "d"])
    ax = plot_activities(result, top=3, color_by=["x", "y", "x", "y"],
                         ax=matplotlib.figure.Figure().subplots())  # fmt: skip
    cells = next(image for image in ax.get_images() if image.get_gid() == "scores")
    drawn = np.ma.filled(cells.get_array().astype(float), np.nan)
    largest = np.max(np.abs(result.scores), axis=0)
    shown = np.sort(np.argsort(-largest)[:3])
    # The profiles grouped by label, in the order the labels first appear.
    assert_array_equal(drawn, result.scores[[0, 2, 1, 3]][:, shown].T)
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == [result.sources[j] for j in shown]
    with pytest.raises(TypeError, match="activities must be an Activities"):
        plot_activities(result.scores)
    with pytest.raises(ValueError, match="no source 'X' among the activities"):
        plot_activities(result, sources=["X"])
    with pytest.raises(ValueError, match="the activities have 4 profiles"):
        plot_top_activities(result)


def test_the_examples_run() -> None:
    assert doctest.testmod(activity, raise_on_error=False).failed == 0
