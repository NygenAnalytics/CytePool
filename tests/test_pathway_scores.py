"""Tests of the per-sample scores of gene sets: GSVA, ssGSEA, combined z-scores and
PLAGE.

The scores are compared with GSVA 2.0.0's, which validation/pathway_scores.py wrote to
tests/data/pathway_scores; with values derived by hand on small matrices; and with
invariants that a wrong implementation would break. The heatmap is read back from the
artists it draws.
"""

import dataclasses
import doctest
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import (
    Cube,
    Design,
    GeneSets,
    PathwayScores,
    Table,
    combined_z_scores,
    gsva,
    pathway_scores,
    plage,
    ssgsea,
)
from cytepool.normalization import filter_by_expression, log2_cpm, tmm_factors
from cytepool.plotting import plot_pathway_scores
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import fit_linear_model

DATA = Path(__file__).parent / "data" / "pathway_scores"
FIXTURE = json.loads((DATA / "gsva.json").read_text())
ARRAYS = np.load(DATA / "gsva.npz")
FUNCTIONS = {
    "gsva": gsva,
    "ssgsea": ssgsea,
    "zscore": combined_z_scores,
    "plage": plage,
}
EXACT = 1e-10  # relative to the largest absolute score: closed forms and ranks


def _options(case: dict) -> dict:
    """CytePool's options for GSVA's, as validation/pathway_scores.py maps them."""
    names = {"tau": "exponent", "alpha": "exponent", "normalize": "normalize"}
    kernels = {"Gaussian": "gaussian", "Poisson": "poisson", "none": "none"}
    options = {}
    for key, value in case["options"].items():
        if key == "kcdf":
            options["kernel"] = kernels[value]
        elif key == "maxDiff":
            options["score_type"] = "max_deviation"
        elif key == "absRanking":
            options["score_type"] = "kuiper"
        elif key in names:
            options[names[key]] = value
    return options


def _run(case: dict) -> tuple[PathwayScores, np.ndarray]:
    values = ARRAYS[case["data"]]
    if case["data"] == "counts":
        values = values.astype(np.int64)  # stored as integers, as GSVA was given them
    function = FUNCTIONS[case["method"]]
    result = function(
        values,
        FIXTURE["sets"][case["sets"]],
        feature_ids=FIXTURE["genes"],
        **_options(case),
    )
    return result, ARRAYS[case["name"]]


def _difference(ours: np.ndarray, theirs: np.ndarray) -> float:
    assert_array_equal(np.isnan(ours), np.isnan(theirs))
    return float(np.nanmax(np.abs(ours - theirs)) / np.nanmax(np.abs(theirs)))


CASES = [pytest.param(case, id=case["name"]) for case in FIXTURE["cases"]]


# Against GSVA.


@pytest.mark.parametrize("case", CASES)
def test_the_scores_are_those_of_gsva(case: dict) -> None:
    result, theirs = _run(case)

    assert list(result.gene_sets) == FIXTURE["scored"][case["name"]]
    ours = result.scores
    if case["method"] == "plage":
        # GSVA's sign is that of R's svd; CytePool's makes the loadings sum to a
        # positive number. Sets of one gene differ on purpose (GSVA gives 1).
        several = result.n_genes > 1
        signs = np.sign(np.sum(ours * theirs, axis=0))[several]
        assert np.all(signs != 0)
        assert _difference(ours[:, several] * signs, theirs[:, several]) < EXACT
        assert_array_equal(theirs[:, ~several], 1.0)
    elif case["name"] == "gsva, missing values":
        # The samples without missing values agree; the others differ on purpose.
        complete = ~np.isnan(ARRAYS[case["data"]]).any(axis=1)
        assert complete.sum() == 5
        assert _difference(ours[complete], theirs[complete]) < EXACT
        assert np.nanmax(np.abs(ours[~complete] - theirs[~complete])) > 1e-3
    else:
        assert _difference(ours, theirs) < EXACT


def test_the_options_that_auto_chose_are_recorded() -> None:
    counts = next(case for case in FIXTURE["cases"] if case["data"] == "counts")
    result, _ = _run(counts)
    assert result.settings["kernel"] == "poisson"  # integers: GSVA's Poisson kernel
    as_decimals = gsva(
        ARRAYS["counts"], FIXTURE["sets"]["main"], feature_ids=FIXTURE["genes"]
    )
    assert as_decimals.settings["kernel"] == "gaussian"
    many = gsva(
        ARRAYS["log_expression"],
        FIXTURE["sets"]["main"],
        feature_ids=FIXTURE["genes"],
        min_samples_without_kernel=10,
    )
    assert many.settings["kernel"] == "none"
    assert dict(many.settings) == {
        "kernel": "none",
        "min_samples_without_kernel": 10,
        "exponent": 1.0,
        "score_type": "difference",
        "min_size": 1,
        "max_size": None,
    }


# Values derived by hand.


# Four genes in three samples. Without a kernel, each gene's estimate is the share of
# its samples at or below its value: g1 (1/3, 2/3, 1), g2 (1, 2/3, 1/3),
# g3 (1/3, 1, 2/3), g4 (2/3, 1/3, 1). Ranked within each sample, the later of equal
# estimates lower, and weighted by |p/2 - r| = |2 - r|, the walks of {g1, g2} give,
# with their highest and lowest values:
# sample 1, ranks g3 1, g1 2, g4 3, g2 4, walk 1, 1/2, 1/2, 0: 1 and 0;
# sample 2, ranks g4 1, g2 2, g1 3, g3 4, walk -1/2, 1/2, 1/2, 0: 1/2 and -1/2;
# sample 3, ranks g2 1, g3 2, g4 3, g1 4, walk 2/3, 1/6, -1/3, 0: 2/3 and -1/3.
FOUR_GENES = np.array(
    [[1.0, 3.0, 1.0, 2.0], [2.0, 2.0, 3.0, 1.0], [3.0, 1.0, 2.0, 3.0]]
)
GENES = ["g1", "g2", "g3", "g4"]


@pytest.mark.parametrize(
    ("score_type", "expected"),
    [
        ("difference", [1.0, 0.0, 1.0 / 3.0]),
        ("kuiper", [1.0, 1.0, 1.0]),
        # Equal deviations give the negative one.
        ("max_deviation", [1.0, -0.5, 2.0 / 3.0]),
    ],
)
def test_gsva_of_a_two_gene_set_on_three_samples(
    score_type: str, expected: list[float]
) -> None:
    result = gsva(
        FOUR_GENES,
        {"pair": ["g1", "g2"], "every gene": GENES},
        feature_ids=GENES,
        kernel="none",
        score_type=score_type,
    )
    assert_allclose(result.scores[:, 0], expected, rtol=0, atol=1e-15)
    # The walk of a set of every gene has no other gene to fall at.
    assert np.isnan(result.scores[:, 1]).all()


def test_a_gene_at_the_middle_rank_weighs_nothing() -> None:
    # In sample 1, g1 has rank 2 = p/2: a set of g1 alone has no weight there.
    result = gsva(FOUR_GENES, {"g1": ["g1"]}, feature_ids=GENES, kernel="none")
    assert np.isnan(result.scores[0, 0])
    assert np.isfinite(result.scores[1:, 0]).all()


def test_ssgsea_by_hand_with_ties() -> None:
    # One sample: g2 and g3 tie. Ranks from the bottom, the later of equal values
    # lower: g1 1, g2 3, g3 2, g4 4, g5 5, g6 6; weights: the average ranks
    # truncated, 1, 2, 2, 4, 5, 6. For {g1, g2} with alpha 1:
    # (1 * 1 + 2 * 3) / (1 + 2) - (21 - 1 - 3) / 4 = 7/3 - 17/4 (GSVA: -1.9166...).
    values = np.array([[1.0, 2.0, 2.0, 3.0, 4.0, 5.0]])
    genes = [f"g{i}" for i in range(1, 7)]
    sets = {"g12": ["g1", "g2"], "g23": ["g2", "g3"], "g34": ["g3", "g4"]}
    result = ssgsea(values, sets, feature_ids=genes, exponent=1.0, normalize=False)
    # {g2, g3}: (2 * 3 + 2 * 2) / 4 - (21 - 5) / 4; {g3, g4}: (2 * 2 + 4 * 4) / 6 -
    # (21 - 6) / 4.
    expected = [7.0 / 3.0 - 17.0 / 4.0, 2.5 - 4.0, 20.0 / 6.0 - 15.0 / 4.0]
    assert_allclose(result.scores[0], expected, rtol=1e-15)
    normalized = ssgsea(values, sets, feature_ids=genes, exponent=1.0)
    spread = max(expected) - min(expected)
    assert_allclose(normalized.scores[0], np.array(expected) / spread, rtol=1e-15)


def test_combined_z_scores_of_a_two_gene_set() -> None:
    # z of (1, 2, 3) is (-1, 0, 1); z of (2, 4, 9) is (-3, -1, 4) / sqrt(13).
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 9.0]])
    result = combined_z_scores(values, {"pair": ["a", "b"]}, feature_ids=["a", "b"])
    expected = np.array([-1.0, 0.0, 1.0]) + np.array([-3.0, -1.0, 4.0]) / 13**0.5
    assert_allclose(result.scores[:, 0], expected / 2**0.5, rtol=1e-14, atol=1e-15)


def test_plage_of_correlated_anticorrelated_and_single_genes() -> None:
    # Genes a and b rise together, c falls as a rises. The first right singular
    # vector of their standardized values is z / |z| for z = (-1, 0, 1).
    values = np.array([[1.0, 10.0, 7.0], [2.0, 30.0, 6.0], [3.0, 50.0, 5.0]])
    sets = {"together": ["a", "b"], "opposed": ["a", "c"], "alone": ["c"]}
    result = plage(values, sets, feature_ids=["a", "b", "c"])
    rising = np.array([-1.0, 0.0, 1.0]) / 2**0.5
    # Together: the loadings sum to a positive number for the rising scores.
    assert_allclose(result.scores[:, 0], rising, atol=1e-15)
    # Opposed: the loadings sum to zero; the first of the two largest is a's,
    # positive.
    assert_allclose(result.scores[:, 1], rising, atol=1e-15)
    # Alone: the gene's standardized values, of length one; GSVA gives 1.
    assert_allclose(result.scores[:, 2], -rising, atol=1e-15)


def test_plage_scores_rise_with_the_mean_standardized_expression() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(12, 30))
    genes = [f"g{j}" for j in range(30)]
    sets = {f"s{i}": genes[i : i + 6] for i in range(20)}
    scores = plage(values, sets, feature_ids=genes).scores
    z = combined_z_scores(values, sets, feature_ids=genes).scores
    assert np.all(np.sum(scores * z, axis=0) > 0.0)
    assert_allclose(np.sum(scores**2, axis=0), 1.0, rtol=1e-14)
    assert_allclose(scores.sum(axis=0), 0.0, atol=1e-14)


# Invariants.


def _continuous() -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """The log-expression values of the fixtures without gene 7, equal to gene 6, and
    gene 8, whose whole numbers can equal the constant gene 3: genes with equal
    values or estimates take their ranks in the order of the genes, by design."""
    keep = [j for j in range(ARRAYS["log_expression"].shape[1]) if j not in (7, 8)]
    genes = [FIXTURE["genes"][j] for j in keep]
    return ARRAYS["log_expression"][:, keep], genes, FIXTURE["sets"]["ranked"]


@pytest.mark.parametrize("method", ["gsva", "ssgsea", "zscore", "plage"])
def test_the_order_of_the_sets_and_of_the_genes_does_not_matter(method: str) -> None:
    values, genes, sets = _continuous()
    function = FUNCTIONS[method]
    result = function(values, sets, feature_ids=genes)
    names = list(sets)[::-1]
    reordered = function(
        values, {name: sets[name] for name in names}, feature_ids=genes
    )
    order = [list(result.gene_sets).index(name) for name in reordered.gene_sets]
    assert_array_equal(reordered.scores, result.scores[:, order])
    # The genes in another order; the data have no tied estimates of two genes.
    permutation = np.random.default_rng(1).permutation(len(genes))
    permuted = function(
        values[:, permutation], sets, feature_ids=[genes[j] for j in permutation]
    )
    assert_allclose(permuted.scores, result.scores, rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", ["gsva", "ssgsea", "zscore", "plage"])
def test_relabelled_samples_permute_the_scores(method: str) -> None:
    values, genes, sets = _continuous()
    function = FUNCTIONS[method]
    result = function(values, sets, feature_ids=genes)
    order = np.array([3, 7, 0, 9, 1, 5, 8, 2, 6, 4])
    ids = [f"S{i}" for i in range(values.shape[0])]
    moved = function(
        values[order], sets, feature_ids=genes, sample_ids=[ids[i] for i in order]
    )
    assert moved.profiles["sample_id"].tolist() == [ids[i] for i in order]
    # Kernel estimates summed in another order can differ in the last bit only.
    assert_allclose(moved.scores, result.scores[order], rtol=0, atol=1e-14)


def test_the_kernels_follow_shifts_and_scales_of_a_gene() -> None:
    # Values on a grid of 2^-6, so that shifted differences are exact.
    values = np.round(ARRAYS["log_expression"] * 64.0) / 64.0
    genes, sets = FIXTURE["genes"], FIXTURE["sets"]["main"]
    shifted = values.copy()
    shifted[:, ::3] += 8.0
    # Scaled by a power of two, so that the bandwidth scales exactly too.
    scaled = values.copy()
    scaled[:, 1::4] *= 4.0
    for kernel in ("gaussian", "none"):
        base = gsva(values, sets, feature_ids=genes, kernel=kernel)
        for moved in (shifted, scaled):
            assert_array_equal(
                gsva(moved, sets, feature_ids=genes, kernel=kernel).scores, base.scores
            )
    # Only the order of a gene's values matters without a kernel.
    bent = values.copy()
    bent[:, 1::2] = np.exp(bent[:, 1::2])
    assert_array_equal(
        gsva(bent, sets, feature_ids=genes, kernel="none").scores,
        gsva(values, sets, feature_ids=genes, kernel="none").scores,
    )


def test_ssgsea_depends_on_the_order_of_values_within_samples_only() -> None:
    values, genes, sets = _continuous()
    result = ssgsea(values, sets, feature_ids=genes)
    # A monotone transform of every value, and different shifts of the samples.
    bent = np.exp(values) + np.arange(values.shape[0])[:, np.newaxis]
    assert_array_equal(ssgsea(bent, sets, feature_ids=genes).scores, result.scores)


def test_combined_z_scores_are_affine_invariant_and_odd() -> None:
    values, genes, sets = _continuous()
    result = combined_z_scores(values, sets, feature_ids=genes)
    moved = combined_z_scores(3.0 * values - 1.0, sets, feature_ids=genes)
    assert_allclose(moved.scores, result.scores, rtol=1e-12, atol=1e-13)
    negated = combined_z_scores(-values, sets, feature_ids=genes)
    assert_allclose(negated.scores, -result.scores, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("function", [gsva, ssgsea])
def test_the_number_of_threads_does_not_change_the_scores(function) -> None:
    values, genes, sets = _continuous()
    one = function(values, sets, feature_ids=genes, n_threads=1)
    two = function(values, sets, feature_ids=genes, n_threads=2)
    assert_array_equal(one.scores, two.scores)


# The genes, the sets and their sizes.


def test_sets_are_matched_after_constant_genes_are_left_out() -> None:
    values, genes, _ = _continuous()
    # Genes 3 and 4 are constant: GSVA, z-scores and PLAGE leave them out, ssGSEA
    # ranks them.
    sets = {"constant": [genes[3], genes[4], genes[10], "OUTSIDE"], "only": [genes[3]]}
    for function in (gsva, combined_z_scores, plage):
        result = function(values, sets, feature_ids=genes)
        assert list(result.gene_sets) == ["constant"]
        assert result.gene_sets["constant"] == (genes[10],)
        assert result.n_genes.tolist() == [1]
    ranked = ssgsea(values, sets, feature_ids=genes)
    assert ranked.gene_sets["constant"] == (genes[3], genes[4], genes[10])
    assert ranked.n_genes.tolist() == [3, 1]


def test_sizes_are_counted_among_the_genes_scored() -> None:
    values, genes, _ = _continuous()
    sets = GeneSets(
        {"small": genes[10:12], "medium": genes[10:15] + ["X", "Y"], "large": genes},
        descriptions={"medium": "five genes here"},
        name="tests",
        version="1",
        licence="CC0",
    )
    result = gsva(values, sets, feature_ids=genes, min_size=3, max_size=10)
    assert list(result.gene_sets) == ["medium"]
    assert result.n_genes.tolist() == [5]
    assert result.gene_sets.descriptions["medium"] == "five genes here"
    assert (result.gene_sets.name, result.gene_sets.licence) == ("tests", "CC0")
    with pytest.raises(ValueError, match=r"no gene set has between min_size \(20\)"):
        gsva(values, sets, feature_ids=genes, min_size=20, max_size=30)
    with pytest.raises(ValueError, match="no gene of any set is among the genes"):
        gsva(values, {"other": ["A", "B"]}, feature_ids=genes)


def test_genes_without_an_id_are_ranked_but_in_no_set() -> None:
    values, genes, sets = _continuous()
    ids = list(genes)
    ids[100] = None
    ids[101] = ""
    result = ssgsea(values, sets, feature_ids=ids)
    # The universe keeps every gene; the sets lose those two.
    assert result.scores.shape == (10, len(sets))
    assert all(genes[100] not in members for members in result.gene_sets.values())


def test_the_table_lists_every_set_and_sample() -> None:
    values, genes, sets = _continuous()
    result = gsva(
        values, sets, feature_ids=genes, sample_ids=[f"S{i}" for i in range(10)]
    )
    table = result.table()
    assert len(table) == result.scores.size
    assert table["profile_id"].tolist()[:3] == ["S0", "S1", "S2"]
    assert table["gene_set"].tolist()[10] == list(result.gene_sets)[1]
    assert_array_equal(table["score"], result.scores.T.ravel())
    assert_array_equal(table["n_genes"], np.repeat(result.n_genes, 10))
    assert repr(result) == "PathwayScores(method='gsva', n_profiles=10, n_sets=20)"
    with pytest.raises(ValueError, match="read-only"):
        result.scores[0, 0] = 1.0


# Missing values.


def test_missing_values_are_left_out_or_make_the_score_missing() -> None:
    values, genes, sets = _continuous()
    # Sets of two genes or more: one gene can weigh nothing in GSVA (rank p/2).
    present = set(genes)
    sets = {
        name: members
        for name, members in sets.items()
        if len(present & set(members)) > 1
    }
    gappy = values.copy()
    gappy[2, :] = np.nan  # a sample without values
    gappy[5, 7] = np.nan
    for function in (gsva, ssgsea):
        result = function(gappy, sets, feature_ids=genes)
        assert np.isnan(result.scores[2]).all()
        assert np.isfinite(np.delete(result.scores, 2, axis=0)).all()
    z = combined_z_scores(gappy, sets, feature_ids=genes)
    holds = np.array([genes[7] in members for members in z.gene_sets.values()])
    assert np.isnan(z.scores[5, holds]).all() and np.isfinite(z.scores[5, ~holds]).all()
    with pytest.raises(ValueError, match="PLAGE's singular value decomposition"):
        plage(gappy, sets, feature_ids=genes)


def test_ssgsea_scores_a_set_of_every_gene_as_missing() -> None:
    values, genes, _ = _continuous()
    sets = {"every": genes, "some": genes[:20], "more": genes[20:60]}
    result = ssgsea(values, sets, feature_ids=genes)
    assert np.isnan(result.scores[:, 0]).all()
    raw = ssgsea(values, sets, feature_ids=genes, normalize=False).scores[:, 1:]
    assert_allclose(result.scores[:, 1:], raw / (raw.max() - raw.min()), rtol=1e-15)


# A cube's group, and the linear models.


def _cube() -> tuple[Cube, np.ndarray]:
    rng = np.random.default_rng(8)
    n_samples, n_genes = 12, 400
    means = np.exp(rng.uniform(2.0, 7.0, n_genes))
    counts = rng.negative_binomial(10, 10 / (10 + means), (n_samples, n_genes))
    condition = np.array(["control", "treated"] * (n_samples // 2))
    counts[condition == "treated", :30] *= 3
    cells = np.full((n_samples, 1), 40)
    cells[3] = 5  # too few cells: left out
    cube = Cube(
        {"counts": counts[np.newaxis]},
        samples={
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "condition": condition,
        },
        groups={"group_id": ["B cells"]},
        features={
            "feature_id": [f"ENSG{j}" for j in range(n_genes)],
            "symbol": [f"G{j}" for j in range(n_genes)],
        },
        n_cells=cells,
    )
    return cube, counts


def test_a_cube_group_is_scored_as_its_log2_cpm() -> None:
    cube, counts = _cube()
    sets = {
        "treated": [f"G{j}" for j in range(30)],
        "other": [f"G{j}" for j in range(50, 90)],
    }
    result = gsva(cube, sets, id_column="symbol")
    used = np.flatnonzero(np.arange(12) != 3)
    kept = counts[used]
    keep = filter_by_expression(kept, sample_groups=np.zeros(used.size))
    values = log2_cpm(kept[:, keep], norm_factors=tmm_factors(kept[:, keep]))
    symbols = [f"G{j}" for j in np.flatnonzero(keep)]
    expected = gsva(values, sets, feature_ids=symbols)
    assert_array_equal(result.scores, expected.scores)
    assert result.profiles["sample_id"].tolist() == [f"S{i}" for i in used]
    assert result.profiles["condition"].tolist()[:2] == ["control", "treated"]
    assert (
        result.settings["values"] == "log2_cpm" and result.settings["min_cells"] == 10
    )
    as_counts = gsva(cube, sets, id_column="symbol", values="counts")
    assert as_counts.settings["kernel"] == "poisson"


def test_a_cube_group_filtered_by_sample_groups_keeps_the_genes_of_each_group() -> None:
    cube, counts = _cube()
    sets = {
        "treated": [f"G{j}" for j in range(30)],
        "other": [f"G{j}" for j in range(50, 90)],
    }
    by_condition = {"sample_groups": "condition"}
    result = gsva(cube, sets, id_column="symbol", expression_filter=by_condition)
    used = np.flatnonzero(np.arange(12) != 3)
    kept = counts[used]
    keep = filter_by_expression(kept, sample_groups=cube.samples["condition"][used])
    values = log2_cpm(kept[:, keep], norm_factors=tmm_factors(kept[:, keep]))
    symbols = [f"G{j}" for j in np.flatnonzero(keep)]
    assert_array_equal(result.scores, gsva(values, sets, feature_ids=symbols).scores)
    assert result.settings["expression_filter"]["sample_groups"] == "condition"
    with pytest.raises(KeyError, match="names 'site', which is not a column"):
        gsva(
            cube, sets, id_column="symbol", expression_filter={"sample_groups": "site"}
        )


def test_scores_of_a_cube_go_into_a_linear_model() -> None:
    cube, _ = _cube()
    sets = {
        "treated": [f"G{j}" for j in range(30)],
        "other": [f"G{j}" for j in range(50, 90)],
    }
    result = gsva(cube, sets, id_column="symbol")
    design = Design(result.profiles, "~ condition")
    moderated = moderated_statistics(fit_linear_model(result.scores, design))
    # The set of the genes that the treatment triples is higher in treated samples.
    assert moderated.t[0, 1] > 5.0 and moderated.p_value[0, 1] < 1e-4
    assert abs(moderated.t[1, 1]) < 3.0


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


def test_scores_of_a_cube_record_the_cube_they_read() -> None:
    cube, counts = _cube()
    sets = {
        "treated": [f"G{j}" for j in range(30)],
        "other": [f"G{j}" for j in range(50, 90)],
    }
    result = gsva(cube, sets, id_column="symbol")
    record = result.provenance
    assert record is not None
    assert list(record) == ["method", "versions", "cube", "layer", "settings"]
    assert record["method"] == "gsva" and record["layer"] == "counts"
    assert record["cube"]["fingerprint"] == cube.fingerprint()
    assert record["cube"]["provenance"] == cube.provenance
    assert record["settings"] == result.settings
    assert record["versions"]["cytepool"] == cytepool.__version__
    assert json.loads(json.dumps(record)) == record
    record["method"] = "changed"  # a new dict each time
    assert result.provenance["method"] == "gsva"
    assert repr(result) == "PathwayScores(method='gsva', n_profiles=11, n_sets=2)"
    # Scores of values given as an array name no cube.
    values = log2_cpm(counts[:4], norm_factors=tmm_factors(counts[:4]))
    symbols = [f"G{j}" for j in range(counts.shape[1])]
    of_values = gsva(values, sets, feature_ids=symbols)
    assert of_values.provenance is None
    # Both pickle, the provenance with them.
    assert_pickles(result)
    assert_pickles(of_values)


# Messages.


def test_arguments_are_checked() -> None:
    values, genes, sets = _continuous()
    cube, _ = _cube()
    with pytest.raises(ValueError, match="feature_ids is needed with an array"):
        gsva(values, sets)
    with pytest.raises(ValueError, match=r"one id per gene \(298\), got 3"):
        gsva(values, sets, feature_ids=["a", "b", "c"])
    with pytest.raises(ValueError, match="names two of the genes scored 'G00000'"):
        gsva(values, sets, feature_ids=[genes[0]] * 298)
    with pytest.raises(TypeError, match="gene ids as strings, got int 7"):
        gsva(values, sets, feature_ids=[7] + genes[1:])
    with pytest.raises(ValueError, match="data holds infinite values"):
        gsva(np.where(values > 7.5, np.inf, values), sets, feature_ids=genes)
    with pytest.raises(ValueError, match=r"2-D array of shape \(n_samples, n_genes\)"):
        gsva(values[0], sets, feature_ids=genes)
    with pytest.raises(TypeError, match="data must hold real numbers"):
        gsva(values.astype(str), sets, feature_ids=genes)
    with pytest.raises(ValueError, match="group applies to a cube"):
        gsva(values, sets, feature_ids=genes, group="B cells")
    with pytest.raises(ValueError, match="sample_ids must be unique"):
        gsva(values, sets, feature_ids=genes, sample_ids=["S"] * 10)
    with pytest.raises(ValueError, match="kernel must be one of"):
        gsva(values, sets, feature_ids=genes, kernel="Gaussian")
    with pytest.raises(ValueError, match="the Poisson kernel takes counts"):
        gsva(values, sets, feature_ids=genes, kernel="poisson")
    with pytest.raises(ValueError, match="exponent must be finite and at least 0"):
        ssgsea(values, sets, feature_ids=genes, exponent=-1.0)
    with pytest.raises(ValueError, match=r"max_size \(2\) is smaller than min_size"):
        plage(values, sets, feature_ids=genes, min_size=3, max_size=2)
    with pytest.raises(ValueError, match="every gene has equal values"):
        gsva(np.ones((4, 3)), {"s": ["a"]}, feature_ids=["a", "b", "c"])
    with pytest.raises(ValueError, match="range, which is zero here"):
        ssgsea(values[:1], {"s": genes[:5]}, feature_ids=genes)
    with pytest.raises(KeyError, match="no column 'gene_name'"):
        gsva(cube, sets, id_column="gene_name")
    with pytest.raises(ValueError, match="sample_ids applies to an array"):
        gsva(cube, sets, sample_ids=["S1"])
    with pytest.raises(ValueError, match="values must be one of 'log2_cpm', 'counts'"):
        gsva(cube, sets, values="cpm")
    with pytest.raises(TypeError, match="gene_sets must be a GeneSets or a mapping"):
        gsva(values, ["G00001"], feature_ids=genes)


def test_the_docstring_example_runs_as_shown() -> None:
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in finder.find(pathway_scores.gsva, "gsva", globs={"gsva": gsva}):
        runner.run(test)
    assert runner.failures == 0 and runner.tries == 8


def test_scores_lie_in_their_ranges() -> None:
    values, genes, sets = _continuous()
    for score_type, low, high in (("difference", -1, 1), ("kuiper", 0, 2)):
        scores = gsva(values, sets, feature_ids=genes, score_type=score_type).scores
        assert low <= np.nanmin(scores) and np.nanmax(scores) <= high
    assert math.isclose(
        np.ptp(ssgsea(values, sets, feature_ids=genes).scores), 1.0, rel_tol=1e-15
    )


# The heatmap.


def pathway_scores_figure(scores: PathwayScores, **options):
    """plot_pathway_scores into the Axes of a figure that pyplot does not keep;
    skipped without matplotlib."""
    ax = pytest.importorskip("matplotlib.figure").Figure().subplots()
    return plot_pathway_scores(scores, ax=ax, **options)


def _cells(ax, gid: str = "scores") -> np.ndarray:
    image = next(image for image in ax.get_images() if image.get_gid() == gid)
    return np.ma.filled(np.ma.asarray(image.get_array(), dtype=float), np.nan)


def test_the_heatmap_shows_the_scores_grouped_by_samples() -> None:
    values, genes, sets = _continuous()
    labels = ["b", "a", "b", "c", "a", "a", "b", "c", "c", "a"]
    result = gsva(values, sets, feature_ids=genes)
    ax = pathway_scores_figure(result, color_by=labels)
    # Samples in the order their groups first appear, each group in its order.
    order = [0, 2, 6, 1, 4, 5, 9, 3, 7, 8]
    assert_array_equal(_cells(ax), result.scores[order].T)
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(i + 1) for i in order]
    assert [t.get_text() for t in ax.get_yticklabels()] == list(result.gene_sets)
    strip = ax.child_axes[0]
    codes = _cells(strip, "groups")[0]
    assert codes.tolist() == [0.0] * 3 + [1.0] * 4 + [2.0] * 3
    hues = strip.get_images()[0].get_cmap().colors
    assert list(hues) == ["#2a78d6", "#eb6834", "#1baf7a"]  # the first three hues
    lines = next(c for c in ax.collections if c.get_gid() == "group boundaries")
    assert [segment[0, 0] for segment in lines.get_segments()] == [2.5, 6.5]
    # The scale is symmetric about zero, from blue to red through grey.
    norm = ax.get_images()[0].norm
    largest = np.nanmax(np.abs(result.scores))
    assert (norm.vmin, norm.vmax) == (-largest, largest)
    assert ax.figure.axes[-1].get_ylabel() == "GSVA score"


def test_the_heatmap_standardizes_the_sets_it_shows() -> None:
    values, genes, sets = _continuous()
    result = combined_z_scores(values, sets, feature_ids=genes)
    chosen = [list(sets)[3], list(sets)[0]]
    ax = pathway_scores_figure(result, sets=chosen, standardize=True)
    columns = [list(result.gene_sets).index(name) for name in chosen]
    shown = result.scores[:, columns].T
    expected = (shown - shown.mean(axis=1, keepdims=True)) / shown.std(
        axis=1, ddof=1, keepdims=True
    )
    assert_allclose(_cells(ax), expected, rtol=1e-13, atol=1e-14)
    assert [t.get_text() for t in ax.get_yticklabels()] == chosen
    assert ax.figure.axes[-1].get_ylabel() == "Standardized score"
    assert not ax.child_axes  # no groups, no strip


def test_the_heatmap_of_a_cube_groups_by_a_sample_variable() -> None:
    cube, _ = _cube()
    sets = {"treated": [f"G{j}" for j in range(30)]}
    result = ssgsea(cube, sets, id_column="symbol")
    ax = pathway_scores_figure(result, color_by="condition")
    condition = result.profiles["condition"].tolist()
    order = np.argsort([value != condition[0] for value in condition], kind="stable")
    assert_array_equal(_cells(ax), result.scores[order].T)
    # A missing score is masked, and drawn in the colour for missing values.
    gappy = PathwayScores(
        np.where(np.arange(11)[:, None] == 4, np.nan, result.scores),
        result.profiles,
        result.gene_sets,
        result.method,
        result.settings,
    )
    assert np.isnan(_cells(pathway_scores_figure(gappy))[0, 4])


def test_the_heatmap_refuses_what_it_cannot_show() -> None:
    values, genes, sets = _continuous()
    result = gsva(values, sets, feature_ids=genes)
    with pytest.raises(TypeError, match="scores must be a PathwayScores"):
        pathway_scores_figure(result.scores)
    with pytest.raises(ValueError, match="no set 'unknown' among the scores"):
        pathway_scores_figure(result, sets=["unknown"])
    with pytest.raises(ValueError, match="sets names a set twice"):
        pathway_scores_figure(result, sets=[list(sets)[0]] * 2)
    with pytest.raises(ValueError, match="color_by has 10 levels, more than the 8"):
        pathway_scores_figure(result, color_by=[str(i) for i in range(10)])
    with pytest.raises(KeyError, match="no variable 'condition'"):
        pathway_scores_figure(result, color_by="condition")
    with pytest.raises(ValueError, match=r"one label per sample \(10\)"):
        pathway_scores_figure(result, color_by=["a", "b"])
