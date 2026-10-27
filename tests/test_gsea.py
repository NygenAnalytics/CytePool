"""Tests for gene set enrichment analysis (GSEA) and its figures.

Reference values in ``tests/data/gsea`` come from fgsea 1.32.2, which
``validation/gsea.py`` runs on 300 genes and 14 sets, recording the seeds it draws:
given them through ``rng``, ``gsea`` must give fgsea's tables, its choice of the
multilevel method for every count, and the data of ``plotEnrichmentData``. The other
tests check enrichment scores and leading edges derived by hand, the Mersenne
Twister against the C++ standard and numpy, the scores of all the first genes of a
random sample against a walk of each, invariants that a wrong implementation would
break (negated statistics, tied genes outside a set, the order of the sets and of
the genes, seeds and threads), sizes, the forms the statistics take, warnings,
messages and the figures.
"""

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import gsea as module
from cytepool._ranked_genes import running_sum
from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.gsea import gsea
from cytepool.plotting import plot_enrichment, plot_gsea_table
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "gsea"
EXACT = 1e-10
# fgsea's arguments and gsea's.
OPTIONS = {
    "gseaParam": "exponent",
    "scoreType": "score_type",
    "minSize": "min_size",
    "maxSize": "max_size",
    "eps": "eps",
    "sampleSize": "sample_size",
    "nPermSimple": "n_permutations",
    "nperm": "n_permutations",
}
COLUMNS = (
    "gene_set",
    "n_genes",
    "enrichment_score",
    "normalized_enrichment_score",
    "p_value",
    "adjusted_p_value",
    "log2_error",
    "leading_edge",
)
# A ranking derived by hand: weights |r| of 3, 2, 1, 1, 2, 3.
SIX = {"a": 3.0, "b": 2.0, "c": 1.0, "d": -1.0, "e": -2.0, "f": -3.0}


class Replay(np.random.Generator):
    """A generator that draws the given seeds, fgsea's, as gsea draws its seeds."""

    def __init__(self, seeds: list[int]) -> None:
        super().__init__(np.random.PCG64())
        self.seeds = np.asarray(seeds, dtype=np.int64)

    def choice(self, a: Any, size: Any = None, *args: Any, **kwargs: Any) -> Any:
        return self.seeds[:size] - 1  # gsea's seeds run from 1, numpy's from 0


@cache
def fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA / "fgsea.json").read_text())
    return data


def case_named(name: str) -> dict[str, Any]:
    return next(case for case in fixture()["cases"] if case["name"] == name)


def run_case(case: dict[str, Any], rng: Any, **extra: Any) -> Table:
    data = fixture()
    values = data["statistics"][case["statistics"]]
    options = {OPTIONS[key]: value for key, value in case["options"].items()}
    if "nperm" in case["options"]:
        options["method"] = "permutation"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return gsea(
            dict(zip(data["genes"], values, strict=True)),
            data["sets"][case["sets"]],
            rng=rng,
            **options,
            **extra,
        )


def fgsea_seeds(name: str) -> list[int]:
    reference = fixture()["fgsea"][name]
    seeds = list(reference["simple"])
    if "nperm" not in case_named(name)["options"]:
        seeds.append((reference["multilevel"] or [1])[0])
    return seeds


def new_axes() -> Any:
    """The Axes of a figure that pyplot does not keep; skipped without matplotlib."""
    return pytest.importorskip("matplotlib.figure").Figure().subplots()


def numbers(values: list[Any]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def scores(statistics: dict[str, float], sets: dict[str, Any], **options: Any) -> Table:
    """gsea with few permutations, for its deterministic columns."""
    options = {"method": "permutation", "n_permutations": 10, "rng": 1} | options
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return gsea(statistics, sets, **options)


# fgsea, replayed.


@pytest.mark.parametrize(
    "name",
    [
        "defaults",
        "gseaParam 0",
        "gseaParam 0.5",
        "scoreType neg",
        "scoreType pos, all positive",
        "sizes 15 to 500, eps 0, sampleSize 51, nPermSimple 2000",
        "permutations, nperm 2500",
        "mostly positive statistics",
    ],
)
def test_gsea_given_fgsea_seeds_gives_fgsea_table(name: str) -> None:
    reference = fixture()["fgsea"][name]
    table = run_case(case_named(name), Replay(fgsea_seeds(name)))
    row = {gene_set: i for i, gene_set in enumerate(table["gene_set"].tolist())}
    assert sorted(row) == sorted(reference["pathway"])
    at = [row[gene_set] for gene_set in reference["pathway"]]
    assert table["n_genes"][at].tolist() == reference["size"]
    edges = [table["leading_edge"][i] for i in at]
    assert edges == [", ".join(genes) for genes in reference["leadingEdge"]]
    assert_allclose(table["enrichment_score"][at], reference["ES"], rtol=EXACT)
    pairs = [("normalized_enrichment_score", "NES"), ("p_value", "pval")]
    if reference["log2err"] is not None:
        pairs.append(("log2_error", "log2err"))
    for column, key in pairs:
        assert_allclose(table[column][at], numbers(reference[key]), rtol=EXACT)
    if name != "mostly positive statistics":  # the deliberate difference
        assert_allclose(
            table["adjusted_p_value"][at], numbers(reference["padj"]), rtol=EXACT
        )


def test_a_single_set_has_fgsea_enrichment_score_and_leading_edge() -> None:
    # fgsea draws the random sets of a single set with R's sampler.
    reference = fixture()["fgsea"]["one set"]
    table = run_case(case_named("one set"), 1)
    assert table["gene_set"].tolist() == reference["pathway"]
    assert_allclose(table["enrichment_score"], reference["ES"], rtol=EXACT)
    assert table["leading_edge"][0] == ", ".join(reference["leadingEdge"][0])
    # Within Monte Carlo error: 4 standard deviations of the difference of the log2
    # p-values, as each program estimates its own; and of the NES, whose means of
    # about 500 random scores each vary by about 1.5%.
    errors = np.hypot(table["log2_error"][0], reference["log2err"][0])
    difference = abs(np.log2(table["p_value"][0] / reference["pval"][0]))
    assert difference < 4.0 * errors
    assert_allclose(table["normalized_enrichment_score"], reference["NES"], rtol=0.1)


def test_adjusted_p_values_are_over_the_sets_with_a_p_value() -> None:
    # fgsea adjusts over the sets before it removes those with too few random scores
    # of their sign; CytePool adjusts the p-values it reports.
    name = "mostly positive statistics"
    table = run_case(case_named(name), Replay(fgsea_seeds(name)))
    p = table["p_value"]
    assert np.isnan(p).sum() == 3
    assert_array_equal(table["adjusted_p_value"], adjust_p_values(p, "bh"))
    assert np.all(np.isnan(table["normalized_enrichment_score"][np.isnan(p)]))
    assert np.all(np.isnan(table["log2_error"][np.isnan(p)]))


def test_the_choice_of_the_multilevel_method_is_fgsea() -> None:
    choices = fixture()["multilevel"]
    for key, n, size in (("n1000_size101", 1000, 101), ("n2000_size51", 2000, 51)):
        counts = np.arange(n + 1, dtype=np.float64)
        assert module._needs_multilevel(counts, n, size).tolist() == choices[key]


# Enrichment scores and leading edges derived by hand.


def test_enrichment_score_and_leading_edge_of_a_set_at_the_top() -> None:
    # Weights 3 and 1 of 4, at ranks 1 and 3 of 6: the walk reaches 3/4 at a and
    # 1 - 1/4 at c, and never falls below 0 before them.
    weighted = scores(SIX, {"ac": ["a", "c"]})
    assert weighted["enrichment_score"][0] == 0.75
    assert weighted["leading_edge"][0] == "a"
    # Unweighted: 1/2 at a and 1 - 1/4 at c, the maximum.
    unweighted = scores(SIX, {"ac": ["a", "c"]}, exponent=0.0)
    assert unweighted["enrichment_score"][0] == 0.75
    assert unweighted["leading_edge"][0] == "a, c"


def test_enrichment_score_and_leading_edge_of_a_set_at_the_bottom() -> None:
    # Weights 1 and 3 at ranks 4 and 6: just before d the walk is -3/4, just before f
    # 1/4 - 4/4 = -3/4 too; the first minimum, at d, starts the leading edge, which
    # lists the genes from the bottom.
    table = scores(SIX, {"df": ["d", "f"]})
    assert table["enrichment_score"][0] == -0.75
    assert table["leading_edge"][0] == "f, d"


def test_equal_deviations_give_no_score_unless_one_sided() -> None:
    # Weights 3 and 3 at ranks 1 and 6: the walk reaches 1/2 and falls to -1/2.
    sets = {"af": ["a", "f"]}
    both = scores(SIX, sets)
    assert both["enrichment_score"][0] == 0.0 and both["leading_edge"][0] == ""
    up = scores(SIX, sets, score_type="pos")
    assert up["enrichment_score"][0] == 0.5 and up["leading_edge"][0] == "a"
    down = scores(SIX, sets, score_type="neg")
    assert down["enrichment_score"][0] == -0.5 and down["leading_edge"][0] == "f"


def test_tied_genes_keep_the_order_given() -> None:
    # b ties with a and c: second as given, it is passed after one miss.
    statistics = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.5, "e": -1.0}
    assert scores(statistics, {"b": ["b"]})["enrichment_score"][0] == 0.75
    first = {"b": 1.0, "a": 1.0, "c": 1.0, "d": 0.5, "e": -1.0}
    assert scores(first, {"b": ["b"]})["enrichment_score"][0] == 1.0


# Invariants.


@cache
def ranking() -> tuple[dict[str, float], dict[str, list[str]]]:
    """500 genes with distinct statistics, and sets up, down and spread."""
    rng = np.random.default_rng(3)
    genes = [f"g{i}" for i in range(500)]
    values = rng.normal(size=500)
    values[:40] += 0.8
    values[40:70] -= 0.8
    sets = {
        "up": genes[:30] + genes[200:210],
        "down": genes[40:65],
        "spread": genes[::17],
        "small": genes[100:103],
    }
    return dict(zip(genes, values.tolist(), strict=True)), sets


def test_negated_statistics_negate_the_scores() -> None:
    statistics, sets = ranking()
    table = scores(statistics, sets)
    negated = scores({gene: -value for gene, value in statistics.items()}, sets)
    assert_allclose(negated["enrichment_score"], -table["enrichment_score"], rtol=1e-12)
    # The leading edge holds the same genes, from the most extreme statistic.
    assert negated["leading_edge"].tolist() == table["leading_edge"].tolist()


def test_tied_genes_outside_a_set_leave_its_results_unchanged() -> None:
    statistics, sets = ranking()
    tied = dict(statistics)
    block = [f"g{i}" for i in range(300, 340)]
    for gene in block:
        tied[gene] = 0.25
    reordered = {gene: tied[gene] for gene in reversed(block)}
    reordered |= {gene: value for gene, value in tied.items() if gene not in reordered}
    outside = {
        name: genes for name, genes in sets.items() if not set(genes) & set(block)
    }
    with pytest.warns(RuntimeWarning, match="39 of the 500 statistics"):
        first = gsea(tied, outside, rng=2, n_permutations=200)
    with pytest.warns(RuntimeWarning, match="tie with another one"):
        second = gsea(reordered, outside, rng=2, n_permutations=200)
    assert list(outside) == ["up", "down", "small"]
    assert first.equals(second)


def test_the_order_of_genes_with_distinct_statistics_does_not_matter() -> None:
    statistics, sets = ranking()
    shuffled = dict(sorted(statistics.items(), key=lambda item: item[0][::-1]))
    assert gsea(statistics, sets, rng=4).equals(gsea(shuffled, sets, rng=4))


def test_each_set_has_the_same_results_in_any_order_of_the_sets() -> None:
    # The random sets of each size, and the multilevel method of each size, do not
    # depend on the other sets.
    statistics, sets = ranking()
    forward = gsea(statistics, sets, rng=5)
    backward = gsea(statistics, dict(reversed(list(sets.items()))), rng=5)
    order = [backward["gene_set"].tolist().index(name) for name in sets]
    assert backward.take(order).equals(forward)


def test_a_seed_reproduces_results_and_threads_do_not_change_them() -> None:
    statistics, sets = ranking()
    first = gsea(statistics, sets, rng=7, n_threads=1)
    assert first.equals(gsea(statistics, sets, rng=np.random.default_rng(7)))
    assert first.equals(gsea(statistics, sets, rng=7, n_threads=2))
    other = gsea(statistics, sets, rng=8)
    assert other["enrichment_score"].tolist() == first["enrichment_score"].tolist()
    assert other["p_value"].tolist() != first["p_value"].tolist()
    # Permutations in three chunks of their own seeds.
    chunks = gsea(statistics, sets, rng=9, method="permutation", n_permutations=2500)
    assert chunks.equals(
        gsea(
            statistics,
            sets,
            rng=9,
            method="permutation",
            n_permutations=2500,
            n_threads=2,
        )
    )


def test_sets_are_sized_and_matched_by_the_ranked_genes() -> None:
    statistics, sets = ranking()
    genes = list(statistics)
    padded = {
        "up": [*sets["up"], "NOT_RANKED_1", *sets["up"][:5], "NOT_RANKED_2"],
        "every": genes,
        "one": [genes[0]],
    }
    table = gsea(statistics, padded, rng=1, n_permutations=100)
    # A set of every ranked gene is never tested; genes listed twice count once.
    assert table["gene_set"].tolist() == ["up", "one"]
    assert table["n_genes"].tolist() == [40, 1]
    plain = gsea(
        statistics, {"up": sets["up"], "one": [genes[0]]}, rng=1, n_permutations=100
    )
    assert table.equals(plain)
    assert gsea(statistics, padded, rng=1, n_permutations=100, max_size=10**6).equals(
        plain
    )
    limited = gsea(
        statistics, padded, rng=1, n_permutations=100, min_size=2, max_size=40
    )
    assert limited["gene_set"].tolist() == ["up"]
    empty = gsea(statistics, padded, rng=1, min_size=41, max_size=50)
    assert empty.columns == COLUMNS and len(empty["gene_set"]) == 0


# Random numbers and the compiled parts.


def test_mersenne_twister_is_cxx_std_mt19937() -> None:
    # The C++ standard requires the 10000th output of a default std::mt19937, seeded
    # with 5489, to be 4123659995.
    state = module._mt_seeded(5489)
    outputs = [module._mt_next(state) for _ in range(10_000)]
    assert outputs[-1] == 4123659995
    # numpy's Mersenne Twister, seeded as the reference generator seeds it.
    for seed in (1, 123_456_789, 10**9):
        state = module._mt_seeded(seed)
        reference = np.random.MT19937(0)
        reference._legacy_seeding(seed)
        expected = reference.random_raw(2000).astype(np.int64).tolist()
        assert [module._mt_next(state) for _ in range(2000)] == expected


def test_scores_of_all_first_genes_are_the_scores_of_each() -> None:
    # The convex hulls give, for each number of the first genes drawn, the score of
    # those genes, as a walk over them gives it.
    rng = np.random.default_rng(11)
    weights = np.ascontiguousarray(np.sort(np.abs(rng.normal(size=400)))[::-1])
    weights[150:170] = 0.0  # genes of no weight
    selected = rng.choice(400, 120, replace=False).astype(np.int64) + 1
    # fgsea raises the weights of the sample to a floor below its smallest nonzero one.
    drawn = weights[selected - 1]
    floored = np.maximum(weights, min(1e-5, drawn[drawn > 0].min()) / 1024)
    for score_type in (0, 1, 2):
        prefix = np.empty(selected.size)
        module._prefix_scores(weights, selected, score_type, prefix)
        expected = []
        for i in range(1, selected.size + 1):
            positions = np.sort(selected[:i] - 1)
            tops, bottoms = running_sum(floored, positions)
            up, down = tops.max(), bottoms.min()
            expected.append(
                [0.0 if up == -down else up if up > -down else down, up, down][
                    score_type
                ]
            )
        assert_allclose(prefix, expected, rtol=1e-12, atol=1e-15)


def test_random_samples_hold_distinct_genes_in_range() -> None:
    state = module._mt_seeded(3)
    used = np.zeros(50, dtype=np.bool_)
    for k in (1, 10, 24, 25, 40, 49):  # below half the genes, and at least half
        out = np.empty(k, dtype=np.int64)
        module._random_sample(state, 1, 50, k, used, out)
        assert len(set(out.tolist())) == k and 1 <= out.min() and out.max() <= 50
        assert not used.any()


# The forms of the statistics.


@cache
def de_result() -> Any:
    """limma-voom on simulated counts of 200 genes in 8 samples and two groups."""
    rng = np.random.default_rng(5)
    mean = np.exp(rng.uniform(2.0, 7.0, 200))
    counts = rng.poisson(np.tile(mean, (2, 8, 1)))
    counts[:, 4:, :30] = rng.poisson(4.0 * np.tile(mean[:30], (2, 4, 1)))
    counts[:, 4:, 30:50] = rng.poisson(0.25 * np.tile(mean[30:50], (2, 4, 1)))
    samples = {
        "sample_id": [f"S{i}" for i in range(8)],
        "condition": ["ctrl"] * 4 + ["stim"] * 4,
    }
    cube = Cube(
        {"counts": counts.astype(np.uint32)},
        samples=samples,
        groups={"group_id": ["B", "T"]},
        features={"feature_id": [f"g{i}" for i in range(200)]},
    )
    design = Design(samples, "~ condition", contrasts={"stim": "conditionstim"})
    return limma_voom(cube, design)


def test_a_result_its_table_and_a_mapping_rank_genes_alike() -> None:
    result = de_result()
    sets = {
        "up": [f"g{i}" for i in range(0, 30)],
        "down": [f"g{i}" for i in range(30, 50)],
    }
    quick = {"method": "permutation", "n_permutations": 100, "rng": 1}
    tested = gsea(result, sets, group="T", contrast="stim", **quick)
    table = result.table("T", "stim")
    assert tested.equals(gsea(table, sets, **quick))
    mapping = dict(zip(table["feature_id"].tolist(), table["t"].tolist(), strict=True))
    assert tested.equals(gsea(mapping, sets, **quick))
    assert tested["enrichment_score"][0] > 0.5 > -0.5 > tested["enrichment_score"][1]
    by_change = gsea(table, sets, statistic_column="log2_fold_change", **quick)
    assert by_change["enrichment_score"][0] > 0.5
    with pytest.raises(
        ValueError, match="the result has 2 groups; name one with group"
    ):
        gsea(result, sets)


def test_a_pandas_series_ranks_by_its_index() -> None:
    pandas = pytest.importorskip("pandas")
    statistics, sets = ranking()
    series = pandas.Series(statistics, name="t")
    assert gsea(series, sets, rng=1).equals(gsea(statistics, sets, rng=1))
    frame = pandas.DataFrame({"feature_id": list(statistics), "t": series.to_numpy()})
    assert gsea(frame, sets, rng=1).equals(gsea(statistics, sets, rng=1))


def test_genes_without_an_id_or_a_statistic_are_left_out() -> None:
    statistics, sets = ranking()
    table = Table(
        {
            "feature_id": [*statistics, None, "", "extra"],
            "t": [*statistics.values(), 5.0, 6.0, np.nan],
        }
    )
    assert gsea(table, sets, rng=1).equals(gsea(statistics, sets, rng=1))


# Warnings.


@pytest.mark.filterwarnings("ignore:the p-values of .* likely too large:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:3 of the 300 statistics:RuntimeWarning")
def test_warnings_about_ties_positive_statistics_and_p_values() -> None:
    statistics, sets = ranking()
    tied = dict(statistics) | {"g300": 0.5, "g301": 0.5, "g302": 0.0, "g303": 0.0}
    with pytest.warns(RuntimeWarning, match=r"1 of the 500 statistics \(0.20%\) tie"):
        gsea(tied, sets, rng=1, n_permutations=50, method="permutation")
    positive = {gene: abs(value) + 0.01 for gene, value in statistics.items()}
    with pytest.warns(RuntimeWarning, match="all statistics are positive"):
        gsea(positive, sets, rng=1, n_permutations=50, method="permutation")
    data, case = fixture(), case_named("mostly positive statistics")
    unbalanced = dict(
        zip(data["genes"], data["statistics"][case["statistics"]], strict=True)
    )
    with pytest.warns(RuntimeWarning, match="3 gene set.s. have fewer than 10 random"):
        gsea(unbalanced, data["sets"][case["sets"]], rng=1)
    with pytest.warns(RuntimeWarning, match="are below eps=0.001 and are reported"):
        table = gsea(statistics, sets, rng=1, eps=1e-3)
    floored = table["p_value"] == 1e-3
    assert floored.any() and np.all(np.isnan(table["log2_error"][floored]))


def test_more_permutations_are_suggested_only_where_they_can_help() -> None:
    # Statistics of one sign but for the last three genes, the largest far above
    # the others: random sets of 40 genes almost never score below zero, while
    # some of those of 12 genes do.
    rng = np.random.default_rng(5)
    values = np.sort(np.exp(2.5 * rng.normal(size=500)))[::-1] + 0.01
    values[-3:] = [-0.01, -0.02, -0.03]
    genes = [f"g{j}" for j in range(500)]
    statistics = dict(zip(genes, values, strict=True))
    sets = {"twelve": genes[-12:], "forty": genes[-40:], "top": genes[:15]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = gsea(statistics, sets, rng=1)
    messages = [" ".join(str(w.message).split()) for w in caught]
    assert messages == [
        "1 gene set(s) have fewer than 10 random scores of the sign of their "
        "enrichment score among 1000 permutations, but some; their normalized "
        "scores and p-values are NaN. More permutations (n_permutations=10000) may "
        "give them enough",
        "1 gene set(s) have no random score of the sign of their enrichment score "
        "among 1000 permutations: random sets of their size almost always score "
        "with the other sign, because most statistics share one sign, so more "
        "permutations are unlikely to give them any; their normalized scores and "
        "p-values are NaN",
    ]
    assert all(len(line) <= 80 for w in caught for line in str(w.message).split("\n"))
    assert np.isnan(table["p_value"][:2]).all() and table["p_value"][2] < 1e-20
    # Ten times the permutations give the set of 12 genes enough, and not the other.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bottom = {name: sets[name] for name in ("twelve", "forty")}
        more = gsea(statistics, bottom, rng=1, n_permutations=10_000)
    assert np.isfinite(more["p_value"][0]) and np.isnan(more["p_value"][1])
    assert any("have no random score" in str(w.message) for w in caught)
    assert not any("fewer than 10 random" in str(w.message) for w in caught)


# Messages.


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"min_size": 0}, ValueError, "min_size must be at least 1"),
        ({"min_size": 1.5}, TypeError, "min_size must be an integer"),
        ({"min_size": 5, "max_size": 4}, ValueError, r"max_size \(4\) is smaller"),
        ({"exponent": -1.0}, ValueError, "exponent must be finite and at least 0"),
        ({"score_type": "up"}, ValueError, "score_type must be one of 'std'"),
        ({"method": "simple"}, ValueError, "method must be one of 'multilevel'"),
        ({"n_permutations": 0}, ValueError, "n_permutations must be at least 1"),
        ({"sample_size": 2}, ValueError, "sample_size must be at least 3"),
        ({"eps": 1.0}, ValueError, r"eps must be in \[0, 1\)"),
        ({"rng": -1}, ValueError, "rng must be a non-negative seed"),
        ({"rng": "seed"}, TypeError, "rng must be a numpy.random.Generator"),
        ({"group": "T"}, ValueError, "group and contrast choose a table"),
        ({"n_threads": 0}, ValueError, "n_threads must be at least 1"),
    ],
)
def test_options_are_checked(options: dict[str, Any], error: type, match: str) -> None:
    statistics, sets = ranking()
    with pytest.raises(error, match=match):
        gsea(statistics, sets, **options)


@pytest.mark.filterwarnings("ignore:all statistics are positive:RuntimeWarning")
@pytest.mark.parametrize(
    ("statistics", "sets", "error", "match"),
    [
        (
            {"a": 1.0, "b": np.inf},
            {"s": ["a"]},
            ValueError,
            "the statistic of 'b' is inf",
        ),
        ({"a": 1.0}, {"s": ["a"]}, ValueError, "a ranking needs at least two"),
        ({"a": 0.0, "b": -0.0}, {"s": ["a"]}, ValueError, "every statistic is zero"),
        ({"a": 1.0, "b": 2.0}, {"s": ["x"]}, ValueError, "no gene of any set is among"),
        ({"a": 1.0, "b": 2.0}, {}, ValueError, "gene_sets is empty"),
        ({"a": 1.0, "b": 2.0}, ["a"], TypeError, "gene_sets must be a GeneSets"),
        ({1: 1.0, "b": 2.0}, {"s": ["b"]}, TypeError, "gene ids must be strings"),
        ({"a": "x", "b": 2.0}, {"s": ["b"]}, TypeError, "statistics must map gene ids"),
        ([1.0, 2.0], {"s": ["b"]}, TypeError, "statistics must be a DifferentialExpr"),
    ],
)
def test_statistics_and_sets_are_checked(
    statistics: Any, sets: Any, error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        gsea(statistics, sets)


def test_tables_need_their_columns_and_one_row_per_gene() -> None:
    table = Table({"feature_id": ["a", "b", "a"], "t": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="the gene id 'a' is on more than one row"):
        gsea(table, {"s": ["a"]})
    with pytest.raises(ValueError, match="the table has no column 'stat'"):
        gsea(table, {"s": ["a"]}, statistic_column="stat")
    numeric = Table({"feature_id": [1, 2], "t": [1.0, 2.0]})
    with pytest.raises(TypeError, match="must hold gene ids as text"):
        gsea(numeric, {"s": ["1"]})


# The figures.


def test_plot_enrichment_draws_fgsea_enrichment_data() -> None:
    data = fixture()
    statistics = dict(zip(data["genes"], data["statistics"]["main"], strict=True))
    for figure in data["figures"]:
        ax = plot_enrichment(
            statistics,
            data["sets"]["main"][figure["set"][0]],
            exponent=figure["gseaParam"][0],
            ax=new_axes(),
        )
        walk = next(line for line in ax.lines if line.get_gid() == "running sum")
        assert_array_equal(walk.get_xdata(), figure["rank"])
        assert_allclose(walk.get_ydata(), figure["ES"], rtol=EXACT, atol=1e-15)
        ticks = next(c for c in ax.collections if c.get_gid() == "set")
        assert [s[0, 0] for s in ticks.get_segments()] == figure["ticks"]
        extremes = [ln.get_ydata()[0] for ln in ax.lines if ln.get_gid() == "extremes"]
        assert_allclose(extremes, figure["posES"] + figure["negES"], rtol=EXACT)


def test_plot_enrichment_refuses_sets_without_ranked_genes() -> None:
    with pytest.raises(ValueError, match="no gene of the set is among the ranked"):
        plot_enrichment(SIX, ["x"])
    with pytest.raises(ValueError, match="holds every ranked gene"):
        plot_enrichment(SIX, list(SIX))
    with pytest.raises(TypeError, match="gene_set must be a list of gene ids"):
        plot_enrichment(SIX, "a")


def test_plot_gsea_table_shows_the_top_sets_of_each_sign_with_their_numbers() -> None:
    statistics, sets = ranking()
    table = Table(
        {
            "gene_set": ["up", "down", "spread", "small"],
            "enrichment_score": [0.6, -0.5, 0.2, -0.3],
            "normalized_enrichment_score": [2.345, -1.5, 0.71, -1.0],
            "p_value": [1.23e-5, 0.0004, 0.5, 0.02],
            "adjusted_p_value": [4.9e-5, 0.0008, 0.5, 0.027],
        }
    )
    ax = plot_gsea_table(statistics, sets, table, top=1, ax=new_axes())
    # The most significant set up, then the most significant down, last.
    assert [label.get_text() for label in ax.get_yticklabels()] == ["up", "down"]
    numbers = [text.get_text() for text in ax.texts if text.get_gid() == "numbers"]
    assert numbers == [
        "2.35",
        "-1.50",
        r"$1.2 \cdot 10^{-5}$",
        r"$4.0 \cdot 10^{-4}$",
        r"$4.9 \cdot 10^{-5}$",
        r"$8.0 \cdot 10^{-4}$",
    ]
    bars = next(c for c in ax.collections if c.get_gid() == "gene ranks")
    ranked = sorted(statistics, key=lambda gene: -statistics[gene])
    top_row = sorted(s[0, 0] for s in bars.get_segments() if s[0, 1] == 1.0)
    assert top_row == sorted(ranked.index(gene) + 1.0 for gene in sets["up"])
    ax = plot_gsea_table(statistics, sets, table, ax=new_axes())
    names = [label.get_text() for label in ax.get_yticklabels()]
    assert names == ["up", "spread", "small", "down"]
    with pytest.raises(ValueError, match="gene_sets has no set 'small'"):
        plot_gsea_table(
            statistics, {k: v for k, v in sets.items() if k != "small"}, table
        )
    with pytest.raises(ValueError, match="the table lacks the column"):
        plot_gsea_table(statistics, sets, table.select(["gene_set", "p_value"]))
