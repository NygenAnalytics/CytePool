"""Tests of SumRank.

The reference values come from validation/sumrank.py --write-fixtures: the authors' R
code (SumRank and CalibratePValueswithPermutations, under the Apache-2.0 licence) run on
4 studies of 60 genes with tied p-values, and on 4 studies of 50 genes with 20
permutations. Other values are worked out by hand below, or are closed forms and
invariants that a wrong ranking, sum or count would break.
"""

import io
import math
import pickle
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import (
    Cube,
    CubeCollection,
    Design,
    StudyResults,
    SumRankResult,
    Table,
    dream_voom,
    limma_trend,
    limma_voom,
    sumrank,
)
from cytepool._jit import thread_limit
from cytepool.stats.sumrank import _irwin_hall_cdf, _studies_summed, fit_sumrank

DATA = Path(__file__).parent / "data" / "sumrank"
CONTRAST = "case_vs_ctrl"


@cache
def authors() -> dict[str, np.ndarray]:
    return dict(np.load(DATA / "authors.npz"))


def signed(p: np.ndarray, change: np.ndarray) -> np.ndarray:
    """-log10 p, negative where the log2 fold change is negative."""
    return np.where(change < 0, np.log10(p), -np.log10(p))


# Against the authors' code.


@pytest.mark.parametrize("proportion", [1, 0.5])
def test_statistics_match_the_authors_code(proportion: float) -> None:
    reference = authors()
    fit = fit_sumrank(
        signed(reference["ties_p"], reference["ties_change"]),
        proportion_of_studies=proportion,
    )
    for direction in ("up", "down"):
        # The authors' tables hold 15 significant digits.
        assert_allclose(
            getattr(fit, f"{direction}_statistic"),
            reference[f"ties_{proportion}_{direction}_sums"],
            rtol=1e-14,
        )


def test_permutation_p_values_count_the_authors_null_statistics() -> None:
    reference = authors()
    statistics = signed(reference["small_p"], reference["small_sign"])
    fit = fit_sumrank(
        statistics[0], proportion_of_studies=0.75, null_statistics=statistics[1:]
    )
    assert fit.n_studies_summed == 3
    assert fit.n_permutations == 20
    assert fit.n_null_values == 20 * 50
    scale = 2 * (50 - 1)  # sums are multiples of 1 / (2 (G - 1)) for G genes
    for direction in ("up", "down"):
        observed = np.rint(reference[f"small_{direction}_sums"] * scale)
        null = np.sort(np.rint(reference[f"small_{direction}_null_sums"] * scale))
        at_most = np.searchsorted(null, observed, side="right")
        below = np.searchsorted(null, observed, side="left")
        ours = getattr(fit, f"{direction}_permutation_p_value")
        assert_array_equal(ours, (1 + at_most) / (1 + null.size))
        # The authors' p-value counts the null statistics strictly below (and, by the
        # rounding of their floating-point sums, perhaps some equal ones), without
        # adding one: never larger than this one.
        theirs = reference[f"small_{direction}_calibrated"] * null.size
        assert np.all(theirs >= below - 1e-6)
        assert np.all(theirs <= at_most + 1e-6)
        assert np.all(ours > theirs / null.size)


# Worked out by hand.


def test_two_studies_by_hand() -> None:
    # Study 1 ties genes b and c; study 2 did not test gene d, which is left out.
    fit = fit_sumrank([[3.0, 1.0, 1.0, 2.0, -2.0], [2.0, -1.0, 0.5, np.nan, 1.0]])
    # Over a, b, c and e (G = 4), study 1 ranks a 1, b and c 2.5, e 4, so u = (0,
    # 1/2, 1/2, 1); study 2 ranks a 1, e 2, c 3, b 4, so u = (0, 1, 2/3, 1/3).
    assert_array_equal(fit.ranked, [True, True, True, False, True])
    assert_allclose(
        fit.normalized_ranks,
        [[0, 0.5, 0.5, np.nan, 1], [0, 1, 2 / 3, np.nan, 1 / 3]],
        rtol=1e-15,
    )
    # Two ranks are summed and capped at 1; genes down rank 1 - u.
    assert_allclose(fit.up_statistic, [0, 1, 1, np.nan, 1], rtol=1e-15)
    assert_allclose(fit.down_statistic, [1, 0.5, 5 / 6, np.nan, 2 / 3], rtol=1e-15)
    # F_2(x) = x^2 / 2 for x <= 1; 1 at the cap.
    assert_allclose(fit.up_irwin_hall_p_value, [0, 1, 1, np.nan, 1], rtol=1e-15)
    assert_allclose(
        fit.down_irwin_hall_p_value, [1, 1 / 8, 25 / 72, np.nan, 2 / 9], rtol=1e-14
    )
    assert fit.up_permutation_p_value is None
    assert (fit.n_studies, fit.n_studies_summed, fit.n_null_values) == (2, 2, 0)


def test_the_best_of_two_studies_by_hand() -> None:
    fit = fit_sumrank(
        [[3.0, 1.0, 1.0, 2.0, -2.0], [2.0, -1.0, 0.5, np.nan, 1.0]],
        proportion_of_studies=0.5,
    )
    # One rank, the smaller, capped at 1/2: up min(u), down min(1 - u).
    assert_allclose(fit.up_statistic, [0, 0.5, 0.5, np.nan, 1 / 3], rtol=1e-15)
    assert_allclose(fit.down_statistic, [0.5, 0, 1 / 3, np.nan, 0], rtol=1e-15)
    # F_1(x) = x.
    assert_allclose(fit.up_irwin_hall_p_value, [0, 1, 1, np.nan, 1 / 3], rtol=1e-15)
    assert_allclose(fit.down_irwin_hall_p_value, [1, 0, 1 / 3, np.nan, 0], rtol=1e-15)


def test_permutation_p_values_pool_every_gene_of_every_permutation() -> None:
    # Each study ranks three genes as (3, 2, 1) or (1, 2, 3); the four combinations
    # are the permutations. Their capped sums, up: (0, 1, 1), (1, 1, 1) twice and
    # (1, 1, 0); so of the 12 null statistics two are 0 and ten are 1.
    forward, backward = [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]
    null = [[a, b] for a in (forward, backward) for b in (forward, backward)]
    fit = fit_sumrank([forward, forward], null_statistics=null)
    assert_allclose(fit.up_statistic, [0, 1, 1])
    assert_allclose(fit.up_permutation_p_value, [3 / 13, 1, 1], rtol=1e-15)
    # Down, the data's statistics are (1, 1, 0): gene c ties the two null zeros.
    assert_allclose(fit.down_permutation_p_value, [1, 1, 3 / 13], rtol=1e-15)
    assert fit.n_null_values == 12


def test_each_permutation_ranks_the_genes_its_studies_tested() -> None:
    # The second permutation did not test gene c: it ranks two genes, and adds two
    # null statistics.
    statistics = [[3.0, 2.0, 1.0, 0.0], [2.0, 3.0, 1.0, 0.0]]
    null = np.array([statistics, statistics])
    null[1, 0, 2] = np.nan
    fit = fit_sumrank(statistics, null_statistics=null)
    assert fit.n_null_values == 4 + 3
    # Genes a and b rank 1 and 2 of 4 in one study each: they sum 0 + 1/3. The first
    # permutation is the data; in the second, over a, b and d, they sum 0 + 1/2. So
    # two of the 7 null statistics are at most 1/3: (1 + 2) / (1 + 7).
    assert_allclose(fit.up_statistic[:2], [1 / 3, 1 / 3], rtol=1e-15)
    assert_allclose(fit.up_permutation_p_value[:2], [3 / 8, 3 / 8], rtol=1e-15)


def test_irwin_hall_closed_forms() -> None:
    x = np.linspace(0.0, 1.0, 11)
    for n in (1, 2, 5, 10, 30):
        # Below 1, F_n(x) = x^n / n!.
        expected = x**n / math.factorial(n)
        assert_allclose(_irwin_hall_cdf(x, n), expected, rtol=1e-13, atol=0)
        # The distribution is symmetric about n / 2.
        y = np.linspace(0.0, n, 37)
        assert_allclose(
            _irwin_hall_cdf(y, n) + _irwin_hall_cdf(n - y, n), 1.0, rtol=1e-13
        )
    for n in range(1, 61):
        assert _irwin_hall_cdf(np.array([n / 2]), n)[0] == pytest.approx(0.5, abs=1e-14)
    # Two and three uniforms at integers and halves.
    assert_allclose(_irwin_hall_cdf(np.array([1.5]), 2), [1 - 0.5**2 / 2], rtol=1e-15)
    assert_allclose(
        _irwin_hall_cdf(np.array([1.0, 2.0]), 3), [1 / 6, 5 / 6], rtol=1e-15
    )
    assert_array_equal(
        _irwin_hall_cdf(np.array([-1.0, 0.0, 3.0, 4.0]), 3), [0, 0, 1, 1]
    )


@pytest.mark.parametrize(
    ("proportion", "n_studies", "summed"),
    [
        (1, 7, 7),
        (0.65, 17, 11),  # the paper's 17 studies of Alzheimer's disease
        (0.55, 16, 8),  # and 16 of COVID-19, as the authors' code takes them
        (0.57, 100, 57),  # 57 although 0.57 * 100 is 56.99999999999999
        (0.1, 3, 1),  # at least one
    ],
)
def test_the_number_of_studies_summed(
    proportion: float, n_studies: int, summed: int
) -> None:
    assert _studies_summed(None, proportion, n_studies) == (summed, proportion)


@pytest.mark.parametrize(("summed", "proportion"), [(1, 0.25), (3, 0.6), (5, 1.0)])
def test_a_number_of_studies_summed_is_the_proportion_that_gives_it(
    summed: int, proportion: float
) -> None:
    # Of 5 studies, floor(0.25 * 5) = 1, floor(0.6 * 5) = 3 and 1.0 sums all 5.
    statistics = random_statistics(14, (5, 30))
    null = random_statistics(15, (3, 5, 30))
    by_number = fit_sumrank(statistics, n_studies_summed=summed, null_statistics=null)
    by_proportion = fit_sumrank(
        statistics, proportion_of_studies=proportion, null_statistics=null
    )
    assert_fits_equal(by_number, by_proportion)
    assert_array_equal(by_number.normalized_ranks, by_proportion.normalized_ranks)
    assert by_number.n_studies_summed == by_proportion.n_studies_summed == summed
    assert by_number.proportion_of_studies is None
    assert by_proportion.proportion_of_studies == proportion


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        (
            {"n_studies_summed": 1, "proportion_of_studies": 0.5},
            ValueError,
            r"\(n_studies_summed\) or their proportion \(proportion_of_studies\), "
            "not both",
        ),
        (
            {"n_studies_summed": 0},
            ValueError,
            "n_studies_summed must be from 1 to the number of studies, 2, got 0",
        ),
        ({"n_studies_summed": 3}, ValueError, "number of studies, 2, got 3"),
        ({"n_studies_summed": True}, TypeError, "whole number of studies, not bool"),
        ({"n_studies_summed": 1.5}, TypeError, "whole number of studies, got float"),
    ],
)
def test_a_number_of_studies_summed_out_of_range_is_refused(
    options: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        fit_sumrank([[1.0, 2.0], [3.0, 4.0]], **options)  # type: ignore[arg-type]


# Invariants.


def random_statistics(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    statistics = np.random.default_rng(seed).normal(size=shape)
    statistics[..., 0, 5] = np.nan  # a gene the first study did not test
    statistics[..., 2, :3] = statistics[..., 2, 3, None]  # ties
    return statistics


def assert_fits_equal(first: object, second: object) -> None:
    for name in (
        "up_statistic",
        "down_statistic",
        "up_irwin_hall_p_value",
        "down_irwin_hall_p_value",
        "up_permutation_p_value",
        "down_permutation_p_value",
    ):
        assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)


@pytest.mark.parametrize("proportion", [1, 0.6])
def test_the_order_of_studies_and_genes_does_not_matter(proportion: float) -> None:
    statistics = random_statistics(1, (5, 40))
    null = random_statistics(2, (6, 5, 40))
    fit = fit_sumrank(
        statistics, proportion_of_studies=proportion, null_statistics=null
    )
    studies = [3, 0, 4, 1, 2]
    genes = np.random.default_rng(3).permutation(40)
    shuffled = fit_sumrank(
        statistics[studies][:, genes],
        proportion_of_studies=proportion,
        null_statistics=null[:, studies][:, :, genes],
    )
    assert_fits_equal(shuffled, _taken(fit, genes))


def _taken(fit: object, genes: np.ndarray) -> object:
    class Taken:
        pass

    taken = Taken()
    for name in (
        "up_statistic",
        "down_statistic",
        "up_irwin_hall_p_value",
        "down_irwin_hall_p_value",
        "up_permutation_p_value",
        "down_permutation_p_value",
    ):
        setattr(taken, name, getattr(fit, name)[genes])
    return taken


def test_negated_statistics_swap_up_and_down() -> None:
    # Testing controls against cases negates every statistic.
    statistics = random_statistics(4, (4, 30))
    null = random_statistics(5, (3, 4, 30))
    fit = fit_sumrank(statistics, proportion_of_studies=0.75, null_statistics=null)
    negated = fit_sumrank(
        -statistics, proportion_of_studies=0.75, null_statistics=-null
    )
    for name in ("statistic", "irwin_hall_p_value", "permutation_p_value"):
        assert_array_equal(getattr(negated, f"up_{name}"), getattr(fit, f"down_{name}"))
        assert_array_equal(getattr(negated, f"down_{name}"), getattr(fit, f"up_{name}"))


def test_a_monotone_transform_of_a_study_changes_nothing() -> None:
    statistics = random_statistics(6, (3, 25))
    changed = statistics.copy()
    changed[1] = np.exp(changed[1]) + changed[1] ** 3  # increasing
    changed[2] = 10.0 * changed[2] - 4.0
    assert_fits_equal(fit_sumrank(changed), fit_sumrank(statistics))


def test_infinite_statistics_rank_first_and_tie() -> None:
    # p-values of 0 give infinite statistics: genes a and b tie at the top.
    fit = fit_sumrank([[np.inf, np.inf, 1.0, -np.inf], [5.0, 4.0, 3.0, 2.0]])
    assert_allclose(fit.normalized_ranks[0], [1 / 6, 1 / 6, 2 / 3, 1], rtol=1e-15)
    assert_allclose(fit.up_statistic, [1 / 6, 1 / 2, 1, 1], rtol=1e-15)


def test_one_study_is_its_own_ranking() -> None:
    fit = fit_sumrank([[2.0, 0.5, -1.0, 3.0]])
    assert_allclose(fit.up_statistic, [1 / 3, 0.5, 0.5, 0.0], rtol=1e-15)
    assert_allclose(fit.up_irwin_hall_p_value, [1 / 3, 1, 1, 0], rtol=1e-15)


def test_statistics_are_read_only() -> None:
    fit = fit_sumrank(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], null_statistics=np.ones((1, 2, 3))
    )
    for name in (
        "up_statistic",
        "normalized_ranks",
        "ranked",
        "up_permutation_p_value",
    ):
        with pytest.raises(ValueError, match="read-only"):
            getattr(fit, name)[0] = 0
    assert repr(fit) == (
        "SumRankFit(n_features=3, ranked=3, n_studies=2, n_studies_summed=2, "
        "n_permutations=1)"
    )


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"statistics": [1.0, 2.0]}, ValueError, r"\(n_studies, n_features\)"),
        ({"statistics": np.empty((0, 3))}, ValueError, "has no study"),
        ({"statistics": [["a", "b"], ["c", "d"]]}, TypeError, "real numbers"),
        ({"statistics": [[1.0, 2.0], [np.nan, np.nan]]}, ValueError, "study 1 tested"),
        ({"statistics": [[1.0, np.nan], [np.nan, 2.0]]}, ValueError, "0 genes were"),
        ({"proportion_of_studies": 0}, ValueError, r"in \(0, 1\]"),
        ({"proportion_of_studies": 1.5}, ValueError, r"in \(0, 1\]"),
        ({"proportion_of_studies": "all"}, TypeError, "real number"),
        ({"null_statistics": np.ones((2, 2, 3))}, ValueError, "studies and features"),
        ({"null_statistics": np.ones((2, 3))}, ValueError, "n_permutations"),
        ({"null_statistics": np.ones((0, 2, 2))}, ValueError, "no permutation"),
        (
            {"null_statistics": [[[1.0, np.nan], [np.nan, 1.0]]]},
            ValueError,
            "permutation 0: 0 genes",
        ),
    ],
)
def test_invalid_arguments_are_refused(
    arguments: dict[str, object], error: type[Exception], match: str
) -> None:
    given = {"statistics": [[1.0, 2.0], [3.0, 4.0]]} | arguments
    with pytest.raises(error, match=match):
        fit_sumrank(**given)  # type: ignore[arg-type]


# Several studies of a collection.


def simulated_cube(
    rng: np.random.Generator, study: str, n: int, groups: list[str]
) -> Cube:
    genes = [f"g{j:03d}" for j in range(60)]
    condition = np.array(["ctrl", "case"] * (n // 2))
    change = np.zeros(60)
    change[:6], change[6:10] = 1.5, -1.5
    mean = np.exp(rng.normal(4.0, 1.0, size=(len(groups), 1, 60)))
    mean = mean * np.where(condition[None, :, None] == "case", 2.0**change, 1.0)
    counts = rng.negative_binomial(10, 10 / (10 + mean))
    return Cube(
        {"counts": counts},
        samples={
            "sample_id": [f"{study}{i}" for i in range(n)],
            "condition": condition,
        },
        groups={"group_id": groups},
        features={"feature_id": genes},
        n_cells=np.full((n, len(groups)), 30),
    )


def design(samples: object) -> Design:
    return Design(
        samples,
        "~ condition",
        levels={"condition": ["ctrl", "case"]},
        contrasts={CONTRAST: "conditioncase"},
    )


@cache
def studies() -> StudyResults:
    """Three studies, the third without group B; the second analysed by limma-trend."""
    rng = np.random.default_rng(20261030)
    cubes = {
        "first": simulated_cube(rng, "a", 8, ["T", "B"]),
        "second": simulated_cube(rng, "b", 10, ["T", "B"]),
        "third": simulated_cube(rng, "c", 8, ["T"]),
    }
    collection = CubeCollection(cubes)
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        analysis = limma_trend if study == "second" else limma_voom
        results[study] = analysis(cube, design(cube.samples))
    return StudyResults(collection, results)


def test_studies_of_a_collection_are_ranked_as_fit_sumrank_ranks_their_results() -> (
    None
):
    results = studies()
    collection = results.collection
    # The warning's lines are at most 80 characters.
    with pytest.warns(
        RuntimeWarning, match=r"best 2 of 3 studies' ranks are too\ssmall"
    ):
        result = sumrank(results, "T", CONTRAST, proportion_of_studies=0.7)
    # The signed -log10 p-values of the aligned results, as plain arrays.
    table = results.table("T", CONTRAST)
    shape = (collection.n_studies, collection.n_features)
    p = table["p_value"].reshape(shape)
    change = table["log2_fold_change"].reshape(shape)
    fit = fit_sumrank(signed(p, change), proportion_of_studies=0.7)
    ranked = fit.ranked
    out = result.features
    assert_array_equal(out["feature_id"], collection.features["feature_id"][ranked])
    for direction in ("up", "down"):
        assert_array_equal(
            out[f"{direction}_statistic"],
            getattr(fit, f"{direction}_statistic")[ranked],
        )
        assert_array_equal(
            out[f"{direction}_p_value"],
            getattr(fit, f"{direction}_irwin_hall_p_value")[ranked],
        )
    assert result.settings["n_studies_summed"] == 2
    assert result.settings["p_values"] == "irwin_hall"
    # The genes that every study tested: those measured but left out are counted.
    tested = ~np.isnan(p).any(axis=0)
    assert len(out) == int(tested.sum())
    assert result.provenance["n_features_left_out"] == collection.n_features - len(out)


def test_a_study_without_the_group_is_left_out() -> None:
    result = sumrank(studies(), "B", CONTRAST)
    studies_table = result.studies
    assert studies_table["analysed"].tolist() == [True, True, False]
    assert "has no group 'B'" in studies_table["reason"][2]
    assert studies_table["n_features_tested"][2] == 0
    assert result.settings["n_studies"] == 2


def test_the_studies_summed_may_be_given_as_a_number_or_a_proportion() -> None:
    results = studies()
    warning = r"best 2 of 3 studies' ranks are too\ssmall"
    with pytest.warns(RuntimeWarning, match=warning):
        by_number = sumrank(results, "T", CONTRAST, n_studies_summed=2)
    with pytest.warns(RuntimeWarning, match=warning):
        by_proportion = sumrank(results, "T", CONTRAST, proportion_of_studies=0.7)
    assert by_number.features.equals(by_proportion.features)
    assert by_number.studies.equals(by_proportion.studies)
    settings = by_number.settings, by_proportion.settings
    assert [s["n_studies_summed"] for s in settings] == [2, 2]
    assert [s["proportion_of_studies"] for s in settings] == [None, 0.7]
    with pytest.raises(ValueError, match="not both"):
        sumrank(results, "T", CONTRAST, n_studies_summed=2, proportion_of_studies=0.7)
    # The number is that of the studies that analysed the group: two have group B.
    with pytest.raises(ValueError, match="from 1 to the number of studies, 3, got 4"):
        sumrank(results, "T", CONTRAST, n_studies_summed=4)
    with pytest.raises(ValueError, match="from 1 to the number of studies, 2, got 3"):
        sumrank(results, "B", CONTRAST, n_studies_summed=3)


def test_the_group_and_contrast_of_studies_that_have_one_may_be_left_out() -> None:
    # The studies tested one contrast, and analysed two groups.
    results = studies()
    named = sumrank(results, "T", CONTRAST)
    assert sumrank(results, "T").features.equals(named.features)
    with pytest.raises(ValueError, match="the studies analysed 2 groups; name one"):
        sumrank(results)
    # The same studies with group T alone.
    collection = results.collection
    only_t = StudyResults(
        collection,
        {
            study: (limma_trend if study == "second" else limma_voom)(
                collection.aligned_cube(study), result.design, groups=["T"]
            )
            for study, result in results.items()
        },
    )
    alone = sumrank(only_t)
    assert alone.features.equals(named.features) and alone.studies.equals(named.studies)
    assert (alone.settings["group"], alone.settings["contrast"]) == ("T", CONTRAST)


def explicit_permutations(
    results: StudyResults, n_permutations: int, seed: int
) -> np.ndarray:
    """The statistics of group T of the permutations that the seed draws, each
    study's analysis run by hand on a design with the condition permuted among the
    samples of its fit of T."""
    collection = results.collection
    generator = np.random.default_rng(seed)
    index = {f: i for i, f in enumerate(collection.features["feature_id"].tolist())}
    null = np.full(
        (n_permutations, collection.n_studies, collection.n_features), np.nan
    )
    for b in range(n_permutations):
        for row, study in enumerate(collection.studies):
            cube, samples = collection.aligned_cube(study), results[study].samples
            used = samples["sample_id"][(samples["group_id"] == "T") & samples["used"]]
            ids = cube.samples["sample_id"].tolist()
            condition = dict(zip(ids, cube.samples["condition"].tolist(), strict=True))
            new = dict(condition)
            for i, source in enumerate(generator.permutation(len(used)).tolist()):
                new[used[i]] = condition[used[source]]
            analysis = limma_trend if study == "second" else limma_voom
            table = analysis(
                cube,
                design({"sample_id": ids, "condition": [new[s] for s in ids]}),
                groups=["T"],
            ).table("T", CONTRAST)
            positions = [index[f] for f in table["feature_id"].tolist()]
            null[b, row, positions] = signed(
                table["p_value"], table["log2_fold_change"]
            )
    return null


def test_permutations_rerun_each_studys_analysis() -> None:
    results = studies()
    collection = results.collection
    result = sumrank(
        results, "T", CONTRAST, n_permutations=4, permute="condition", rng=7
    )
    table = results.table("T", CONTRAST)
    shape = (collection.n_studies, collection.n_features)
    observed = signed(
        table["p_value"].reshape(shape), table["log2_fold_change"].reshape(shape)
    )
    fit = fit_sumrank(observed, null_statistics=explicit_permutations(results, 4, 7))
    for direction in ("up", "down"):
        assert_array_equal(
            result.features[f"{direction}_p_value"],
            getattr(fit, f"{direction}_permutation_p_value")[fit.ranked],
        )
    assert result.settings["p_values"] == "permutation"
    assert result.settings["n_null_values"] == fit.n_null_values
    assert result.settings["seed"] == 7


def test_permutations_follow_the_seed_and_not_the_threads() -> None:
    results = studies()

    def p_values(rng: object, threads: int | None = None) -> np.ndarray:
        with thread_limit(threads):
            out = sumrank(
                results, "T", CONTRAST, n_permutations=3, permute="condition", rng=rng
            ).features
        return np.concatenate([out["up_p_value"], out["down_p_value"]])

    first = p_values(11)
    assert_array_equal(p_values(11, threads=1), first)
    assert_array_equal(p_values(np.random.default_rng(11)), first)
    assert not np.array_equal(p_values(12), first)


def test_the_result_records_its_studies_and_is_immutable() -> None:
    results = studies()
    collection = results.collection
    result = sumrank(
        results, "T", CONTRAST, n_permutations=2, permute="condition", rng=1
    )
    record = result.provenance
    assert record["analysis"] == "sumrank"
    assert record["settings"]["permute"] == "condition"
    assert set(record["studies"]) == set(collection.studies)
    assert record["studies"]["first"] == results["first"].provenance
    with pytest.raises(AttributeError, match="immutable"):
        result.features = result.studies  # type: ignore[misc]
    copy = pickle.loads(pickle.dumps(result))
    assert copy.features.equals(result.features) and copy.provenance == record
    text = repr(result)
    assert "3 studies, the best 3 summed" in " ".join(text.split())
    assert "2 permutations of 'condition'" in text
    assert max(map(len, text.splitlines())) <= 80


@cache
def studies_with_a_covariate() -> StudyResults:
    """The three studies with the sequencing depth of each sample in their designs,
    and a contrast of it, which compares no two levels of a factor."""
    collection = studies().collection
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        samples = cube.samples.with_columns(
            {"depth": np.linspace(1.0, 2.0, cube.n_samples)}
        )
        with_depth = Design(
            samples,
            "~ condition + depth",
            levels={"condition": ["ctrl", "case"]},
            contrasts={CONTRAST: "conditioncase", "depth": "depth"},
        )
        analysis = limma_trend if study == "second" else limma_voom
        results[study] = analysis(cube, with_depth)
    return StudyResults(collection, results)


def test_permutations_permute_the_factor_whose_levels_the_contrast_compares() -> None:
    # CONTRAST, conditioncase in ~ condition, compares case with ctrl, the reference.
    results = studies()
    read = sumrank(results, "T", CONTRAST, n_permutations=2, rng=1)
    given = sumrank(
        results, "T", CONTRAST, n_permutations=2, permute="condition", rng=1
    )
    assert read.features.equals(given.features) and read.studies.equals(given.studies)
    assert read.settings == given.settings and read.settings["permute"] == "condition"
    # A covariate beside the factor leaves the comparison of its levels as it is.
    beside = sumrank(studies_with_a_covariate(), "T", CONTRAST, n_permutations=1, rng=1)
    assert beside.settings["permute"] == "condition"


def test_permutations_need_a_variable_of_each_design() -> None:
    results = studies()
    # A contrast of a covariate compares no two levels: the variable must be named.
    with pytest.raises(
        ValueError,
        match="contrast 'depth' does not compare two levels of one factor in every "
        "study's design; give the variable's name as permute",
    ):
        sumrank(studies_with_a_covariate(), "T", "depth", n_permutations=2)
    with pytest.raises(ValueError, match="n_permutations is 0"):
        sumrank(results, "T", CONTRAST, permute="condition")
    with pytest.raises(ValueError, match="not a variable of the design of study"):
        sumrank(results, "T", CONTRAST, n_permutations=1, permute="donor")
    with pytest.raises(ValueError, match="at least 0"):
        sumrank(results, "T", CONTRAST, n_permutations=-1)
    with pytest.raises(TypeError, match="not bool"):
        sumrank(results, "T", CONTRAST, n_permutations=True)


def test_permutations_refuse_a_result_of_another_cube() -> None:
    results = studies()
    collection = results.collection
    # The first study's result from counts other than those of its aligned cube, with
    # the same genes and groups: the line-up accepts it, a re-run would not be its own.
    cube = collection.aligned_cube("first")
    changed = Cube(
        {"counts": cube.layer("counts") + 1},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    other = dict(results)
    other["first"] = limma_voom(changed, design(changed.samples))
    other_studies = StudyResults(collection, other)
    assert len(sumrank(other_studies, "T", CONTRAST).features)
    with pytest.raises(ValueError, match="fingerprint"):
        sumrank(other_studies, "T", CONTRAST, n_permutations=1, permute="condition")


def test_a_mixed_model_is_not_permuted_sample_by_sample() -> None:
    rng = np.random.default_rng(3)
    cube = simulated_cube(rng, "d", 8, ["T"])
    samples = cube.samples.with_columns({"donor": ["D1", "D1", "D2", "D2"] * 2})
    cube = Cube(
        {"counts": cube.layer("counts")},
        samples=samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    collection = CubeCollection({"one": cube, "two": cube})
    mixed = Design(
        samples,
        "~ condition",
        random="(1 | donor)",
        levels={"condition": ["ctrl", "case"]},
        contrasts={CONTRAST: "conditioncase"},
    )
    results = StudyResults(
        collection,
        {
            study: dream_voom(collection.aligned_cube(study), mixed)
            for study in collection.studies
        },
    )
    # Without permutations dream's results are ranked; with them they are refused.
    assert len(sumrank(results, "T", CONTRAST).features)
    with pytest.raises(ValueError, match="random effects"):
        sumrank(results, "T", CONTRAST, n_permutations=1, permute="condition")


def test_invalid_studies_and_contrasts_are_refused() -> None:
    results = studies()
    with pytest.raises(KeyError, match="no group 'NK'"):
        sumrank(results, "NK", CONTRAST)
    with pytest.raises(KeyError, match="no study tested the contrast"):
        sumrank(results, "T", "other")
    with pytest.raises(ValueError, match="every study of the collection"):
        sumrank(
            StudyResults(results.collection, {"first": results["first"]}),
            "T",
            CONTRAST,
        )
    with pytest.raises(TypeError, match="sumrank takes the studies' results"):
        sumrank(dict(results), "T", CONTRAST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="permute must be"):
        sumrank(
            results,
            "T",
            CONTRAST,
            n_permutations=1,
            permute=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-negative seed"):
        sumrank(results, "T", CONTRAST, n_permutations=1, permute="condition", rng=-1)


def test_the_figure_shows_the_p_values_in_alphabetical_order() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from cytepool.plotting import plot_sumrank

    result = sumrank(studies(), "T", CONTRAST)
    ax = plot_sumrank(result, direction="down", top=3, ax=Figure().subplots())
    table = result.features
    ids = table["feature_id"].tolist()
    order = np.argsort(ids, kind="stable")
    # A p-value of 0 is drawn at the smallest positive float.
    tiny = np.finfo(np.float64).tiny
    heights = -np.log10(np.maximum(table["down_p_value"][order], tiny))
    adjusted = table["down_adjusted_p_value"][order]
    shown = {
        artist.get_gid(): artist.get_offsets()
        for artist in ax.collections
        if artist.get_gid() in ("genes", "significant")
    }
    passed = adjusted < 0.05
    assert passed.any() and (~passed).any()
    assert_allclose(shown["significant"][:, 1], heights[passed])
    assert_allclose(shown["genes"][:, 1], heights[~passed])
    assert_allclose(shown["genes"][:, 0], np.flatnonzero(~passed))
    (line,) = [line for line in ax.lines if line.get_gid() == "threshold"]
    assert line.get_ydata()[0] == pytest.approx(heights[~passed].max())
    labels = [text.get_text() for text in ax.texts if text.get_gid() == "labels"]
    best = np.argsort(table["down_p_value"], kind="stable")[:3]
    assert sorted(labels) == sorted(ids[i] for i in best)
    with pytest.raises(ValueError, match="direction"):
        plot_sumrank(result, direction="both")
    with pytest.raises(TypeError, match="SumRankResult"):
        plot_sumrank(table)  # type: ignore[arg-type]


def tied_result() -> SumRankResult:
    """200 genes, of which 30 next to each other in alphabetical order share the
    smallest p-value of 100 permutations, with statistics in a shuffled order."""
    n = 200
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 1.0, n)
    tied = np.arange(50, 80)
    p[tied] = 1.0 / 101.0
    statistic = rng.permutation(n).astype(np.float64)
    adjusted = np.where(np.isin(np.arange(n), tied), 0.01, 0.9)
    table = Table(
        {
            "feature_id": [f"g{j:03d}" for j in range(n)],
            "symbol": [f"GENE{j:03d}" for j in range(n)],
            "up_statistic": statistic,
            "up_p_value": p,
            "up_adjusted_p_value": adjusted,
            "down_statistic": statistic,
            "down_p_value": p,
            "down_adjusted_p_value": adjusted,
        }
    )
    settings = {
        "group": "T",
        "contrast": CONTRAST,
        "n_studies": 3,
        "n_studies_summed": 3,
        "p_values": "permutation",
        "n_permutations": 100,
        "permute": "condition",
    }
    return SumRankResult(
        table, Table({"study": ["a", "b", "c"]}), {"settings": settings}
    )


def test_tied_p_values_are_labelled_by_statistic_and_the_labels_kept_apart() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from cytepool.plotting import plot_sumrank

    result = tied_result()
    figure = Figure()
    FigureCanvasAgg(figure)
    ax = plot_sumrank(result, top=10, ax=figure.subplots())
    labels = [text for text in ax.texts if text.get_gid() == "labels"]
    # Of the 30 genes tied at the smallest p-value, the 10 with the smallest
    # statistics, the best ranks.
    statistic = result.features["up_statistic"]
    tied = np.arange(50, 80)
    best = tied[np.argsort(statistic[tied], kind="stable")[:10]]
    assert sorted(text.get_text() for text in labels) == sorted(
        f"GENE{j:03d}" for j in best
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    # The extent of each name, without the line that joins a moved name to its gene.
    boxes = [Text.get_window_extent(text, renderer) for text in labels]
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            assert not first.overlaps(second)
    # Labels moved above others point at their genes; all are inside the frame.
    frame = ax.get_window_extent()
    assert all(box.y1 <= frame.y1 for box in boxes)
    moved = [text for text in labels if text.xyann[1] > labels[0].xyann[1]]
    assert moved and all(text.arrow_patch is not None for text in moved)
    assert labels[0].arrow_patch is None


def test_labels_and_legends_stay_inside_two_panels_after_the_final_draw() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from cytepool.plotting import plot_sumrank

    # The genes with the smallest p-values are the first and last in alphabetical
    # order, where a label centred on its gene would cross the frame.
    n = 400
    rng = np.random.default_rng(4)
    p = rng.uniform(0.2, 1.0, n)
    ends = np.array([0, 1, n - 2, n - 1])
    p[ends] = [1e-9, 2e-9, 3e-9, 4e-9]
    adjusted = np.where(np.isin(np.arange(n), ends), 1e-6, 0.9)
    table = Table(
        {
            "feature_id": [f"gene_with_a_long_name_{j:04d}" for j in range(n)],
            "up_statistic": np.arange(n, dtype=np.float64),
            "up_p_value": p,
            "up_adjusted_p_value": adjusted,
            "down_statistic": np.arange(n, dtype=np.float64),
            "down_p_value": p,
            "down_adjusted_p_value": adjusted,
        }
    )
    settings = {
        "group": "T",
        "contrast": CONTRAST,
        "n_studies": 3,
        "n_studies_summed": 3,
        "p_values": "irwin_hall",
    }
    result = SumRankResult(
        table, Table({"study": ["a", "b", "c"]}), {"settings": settings}
    )
    # As the replication guide draws genes up and down: two panels, constrained.
    figure = Figure(figsize=(8, 6.5), layout="constrained")
    FigureCanvasAgg(figure)
    upper, lower = figure.subplots(2, 1)
    plot_sumrank(result, top=4, ax=upper)
    plot_sumrank(result, direction="down", top=4, ax=lower)
    figure.savefig(io.BytesIO(), format="png")
    renderer = figure.canvas.get_renderer()
    for ax in (upper, lower):
        frame = ax.get_window_extent()
        labels = [text for text in ax.texts if text.get_gid() == "labels"]
        assert len(labels) == 4
        for text in labels:
            box = Text.get_window_extent(text, renderer)  # the name, without its line
            assert frame.x0 <= box.x0 and box.x1 <= frame.x1, text.get_text()
            assert box.y1 <= frame.y1, text.get_text()
        # The legend lies beside the plot, clear of the labels along its top.
        legend = ax.get_legend().get_window_extent(renderer)
        assert legend.x0 >= frame.x1


def test_a_result_is_built_from_tables_and_a_record() -> None:
    result = sumrank(studies(), "T", CONTRAST)
    with pytest.raises(TypeError, match="features must be a Table"):
        SumRankResult({}, result.studies, {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Out of range float values"):
        SumRankResult(result.features, result.studies, {"x": float("nan")})
