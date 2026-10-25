"""Tests for limma-voom and limma-trend on the groups of a cube.

Reference values in ``tests/data/differential_expression`` come from limma and edgeR run
on the pseudobulk counts of the cells in ``cells.npz``: ``filterByExpr``,
``calcNormFactors``, ``voom`` (or ``cpm(log = TRUE, prior.count = 3)``), ``lmFit``,
``contrasts.fit``, ``eBayes`` and ``topTable``, per cell type, after the samples with
fewer than 10 cells are left out; ``validation/differential_expression.py`` writes both
files. In the NK cells both samples of donor D1, the reference level, have fewer than
10 cells, so the design of that group is rank deficient. The other tests check what a
wrong implementation would break: identical results for any order of the design's
rows, for a design covering some samples and for a cube without the samples left out;
equal estimates of an estimable contrast in two parametrisations; identities that
follow the cube; and the statuses of groups that cannot be fitted.
"""

import json
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest

from cytepool.cube import Cube, from_bulk, from_cells
from cytepool.design import Design
from cytepool.differential_expression import (
    DifferentialExpressionResult,
    limma_trend,
    limma_voom,
)
from cytepool.io import read_cube, write_cube
from cytepool.io.zarr import ZarrArray
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "differential_expression"
EXACT = 1e-10
# limma solves the robust estimate of the prior only to a tolerance.
CLOSE = 1e-6
STATISTICS = {
    "log2_fold_change": 0,
    "average_log2_cpm": 1,
    "t": 2,
    "p_value": 3,
    "adjusted_p_value": 4,
    "b_statistic": 5,
}
CONTRASTS = {
    "stim_vs_ctrl": "conditionstim",
    "D2_vs_D1": "donorD2",
    "D2_vs_D4": "donorD2 - donorD4",
}
LIMMA_ERRORS = {"contrast_standard_errors": "limma"}
# The analyses of the fixture, as the validation script ran them, with the tier of
# each. voom chooses its span from the number of features in the first, as limma's
# adaptive.span = TRUE does, and has it fixed at 0.5 in the second. The robust prior
# of the second has finite df, which limma solves for only to a tolerance; in the
# trend fit they are infinite for nearly every feature, where the robust estimate is
# closed-form.
ANALYSES = {
    "voom": (limma_voom, LIMMA_ERRORS, EXACT),
    "robust": (limma_voom, LIMMA_ERRORS | {"robust": True, "span": 0.5}, CLOSE),
    "trend": (
        limma_trend,
        {"expression_filter": False, "normalization": None, "robust": True},
        EXACT,
    ),
}


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def cells_cube() -> Cube:
    """The cube that from_cells pools from the fixture's cells."""
    data = fixture("cells")
    table = {
        key.removeprefix("sample_table "): data[key]
        for key in data
        if key.startswith("sample_table ")
    }
    return from_cells(
        data["counts"],
        samples=data["samples"],
        groups=data["groups"],
        sample_table=table,
        layers="counts",
    )


def paired_design(samples: Any, **options: Any) -> Design:
    options.setdefault("levels", {"condition": ["ctrl", "stim"]})
    options.setdefault("contrasts", CONTRASTS)
    return Design(samples, "~ donor + condition", **options)


@cache
def voom_result() -> DifferentialExpressionResult:
    cube = cells_cube()
    return limma_voom(cube, paired_design(cube.samples), **LIMMA_ERRORS)


def scaled_difference(ours: np.ndarray, reference: np.ndarray, p_value: bool) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|): about the error in log p relative to |log p|."""
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale))


def assert_matches_limma(table: Table, reference: np.ndarray, tolerance: float) -> None:
    for column, row in STATISTICS.items():
        p_value = column.endswith("p_value")
        difference = scaled_difference(table[column], reference[row], p_value)
        assert difference <= tolerance, (column, difference)


def assert_tables_agree(first: Table, second: Table, tolerance: float) -> None:
    assert first["feature_id"].tolist() == second["feature_id"].tolist()
    for column in STATISTICS:
        p_value = column.endswith("p_value")
        difference = scaled_difference(first[column], second[column], p_value)
        assert difference <= tolerance, (column, difference)


def simulated_cube(cells: dict[str, list[int]], n_features: int = 80) -> Cube:
    """Negative binomial counts of 4 donors x 2 conditions, with these cell numbers."""
    samples = {
        "sample_id": [f"D{d}_{c}" for d in range(1, 5) for c in ("ctrl", "stim")],
        "donor": [f"D{d}" for d in range(1, 5) for _ in range(2)],
        "condition": ["ctrl", "stim"] * 4,
    }
    rng = np.random.default_rng(7)
    n_cells = np.array(list(cells.values()), dtype=np.int64)  # groups x samples
    mean = rng.lognormal(4.0, 1.0, n_features) * n_cells[:, :, np.newaxis] / 20
    counts = rng.negative_binomial(5, 5 / (5 + mean))
    return Cube(
        {"counts": counts},
        samples=samples,
        groups={"group_id": list(cells)},
        features={"feature_id": [f"g{j}" for j in range(n_features)]},
        n_cells=n_cells.T,
    )


def group_rows(result: DifferentialExpressionResult, group: str) -> Table:
    return result.samples.filter(result.samples["group_id"] == group)


# Against limma, from cells to tables.


@pytest.mark.parametrize("analysis", ANALYSES)
def test_tables_from_cells_match_limma(analysis: str) -> None:
    function, options, tolerance = ANALYSES[analysis]
    cube = cells_cube()
    result = function(cube, paired_design(cube.samples), **options)
    limma = fixture("limma")
    for group in ("NK", "T"):
        key = f"{analysis} {group}"
        assert (
            group_rows(result, group)["used"].tolist() == limma[f"{key} used"].tolist()
        )
        record = next(g for g in result.provenance["groups"] if g["group_id"] == group)
        assert record["aliased_columns"] == limma[f"{key} aliased"].tolist()
        # The span of voom's trend, which limma reports: chosen, or 0.5 as given.
        span = float(limma[f"{key} span"][0]) if f"{key} span" in limma else None
        assert record.get("voom_span") == span
        row = result.groups.index(group)
        prior = float(np.median(limma[f"{key} df_prior"]))
        assert result.summary["prior_df"][row] == pytest.approx(prior, rel=tolerance)
        tested = cube.features["feature_id"][limma[f"{key} keep"]]
        computed = {c for c in CONTRASTS if f"{key} {c}" in limma}
        ours = {c for g, c in result.tables if g == group}
        if group == "NK":
            # The deliberate differences of the next test: limma refuses D2_vs_D4 and
            # computes D2_vs_D1, which CytePool reports as not estimable.
            assert computed == {"stim_vs_ctrl", "D2_vs_D1"}
            assert ours == {"stim_vs_ctrl", "D2_vs_D4"}
        else:
            assert ours == computed == set(CONTRASTS)
        for contrast in ours & computed:
            table = result.table(group, contrast)
            assert table["feature_id"].tolist() == tested.tolist()
            assert_matches_limma(table, limma[f"{key} {contrast}"], tolerance)
    if analysis == "robust":
        # Each feature has its own prior df: the robust estimator did its work.
        assert np.ptp(limma["robust T df_prior"]) > 0


@pytest.mark.parametrize("analysis", ANALYSES)
def test_rank_deficient_group_differs_from_limma_only_where_limma_is_wrong(
    analysis: str,
) -> None:
    # Without donor D1, the reference level, donorD4 = (Intercept) - donorD2 -
    # donorD3 in the NK cells. limma's contrasts.fit refuses D2 vs D4, which weighs
    # the aliased column, and computes donorD2 from the kept columns, where it means
    # D2 vs D4: the values limma reports under the name D2_vs_D1.
    function, options, tolerance = ANALYSES[analysis]
    cube = cells_cube()
    result = function(cube, paired_design(cube.samples), **options)
    limma = fixture("limma")
    assert "non-estimable" in str(limma[f"{analysis} NK D2_vs_D4 error"])
    computed = result.table("NK", "D2_vs_D4")
    assert_matches_limma(computed, limma[f"{analysis} NK D2_vs_D1"], tolerance)
    assert ("NK", "D2_vs_D1") not in result.tables
    row = result.groups.index("NK")
    assert result.summary["not_estimable"][row] == "D2_vs_D1"
    with pytest.raises(KeyError, match=r"donorD4 = \(Intercept\) - donorD2 - donorD3"):
        result.table("NK", "D2_vs_D1")


def test_exact_standard_errors_are_limmas_for_a_single_coefficient() -> None:
    cube = cells_cube()
    result = limma_voom(cube, paired_design(cube.samples))
    limma = fixture("limma")
    for group in ("NK", "T"):
        table = result.table(group, "stim_vs_ctrl")
        assert_matches_limma(table, limma[f"voom {group} stim_vs_ctrl"], EXACT)


# Sample identity.


def test_the_order_of_the_designs_rows_does_not_matter() -> None:
    cube = cells_cube()
    order = np.random.default_rng(3).permutation(cube.n_samples)
    shuffled = paired_design(cube.samples.take(order))
    result = limma_voom(cube, shuffled, contrast_standard_errors="limma")
    expected = voom_result()
    assert result.tables.keys() == expected.tables.keys()
    for key, table in expected.tables.items():
        assert result.tables[key].equals(table)
    assert result.summary.equals(expected.summary)
    assert result.samples.equals(expected.samples)
    assert result.provenance["groups"] == expected.provenance["groups"]


def test_a_design_of_some_samples_equals_a_cube_of_those_samples() -> None:
    cube = cells_cube()
    some = [sid for sid in cube.samples["sample_id"] if not sid.startswith("D3")]
    subset = cube.samples.filter(np.isin(cube.samples["sample_id"], some))
    design = paired_design(subset)
    on_full = limma_voom(cube, design)
    on_subset = limma_voom(cube.select(samples=some), design)
    for key, table in on_subset.tables.items():
        assert on_full.tables[key].equals(table)
    assert on_full.samples.equals(on_subset.samples)
    assert on_full.samples["sample_id"].tolist() == 2 * some


def test_leaving_samples_out_equals_a_cube_without_them() -> None:
    # In the T cells, D3_stim has 4 cells. Without it, D3 keeps its control sample,
    # so a design built on the other samples has the same columns.
    cube = cells_cube()
    rows = group_rows(voom_result(), "T")
    assert rows["reason"][rows["sample_id"] == "D3_stim"][0].startswith("4 cells")
    others = [sid for sid in cube.samples["sample_id"] if sid != "D3_stim"]
    smaller = cube.select(samples=others, groups="T")
    design = paired_design(smaller.samples)
    assert design.columns == paired_design(cube.samples).columns
    without = limma_voom(smaller, design, contrast_standard_errors="limma")
    for contrast in CONTRASTS:
        assert without.table("T", contrast).equals(voom_result().table("T", contrast))


def test_identities_follow_the_cube() -> None:
    cube = cells_cube()
    reversed_features = cube.select(features=cube.features["feature_id"][::-1])
    with_symbols = Cube(
        {"counts": reversed_features.layer("counts")},
        samples=cube.samples,
        groups=cube.groups,
        features={
            "feature_id": reversed_features.features["feature_id"],
            "symbol": [
                f"SYM{f.removeprefix('feature_')}"
                for f in reversed_features.features["feature_id"]
            ],
        },
        n_cells=cube.n_cells,
    )
    result = limma_voom(
        with_symbols, paired_design(cube.samples), contrast_standard_errors="limma"
    )
    for (group, contrast), expected in voom_result().tables.items():
        table = result.table(group, contrast)
        # Rows follow the cube's feature order, here reversed.
        position = {f: i for i, f in enumerate(with_symbols.features["feature_id"])}
        rows = [position[f] for f in table["feature_id"]]
        assert rows == sorted(rows)
        assert table["symbol"].tolist() == [
            f"SYM{f.removeprefix('feature_')}" for f in table["feature_id"]
        ]
        row_of = {f: i for i, f in enumerate(table["feature_id"])}
        restored = table.take([row_of[f] for f in expected["feature_id"]])
        assert_tables_agree(restored, expected, EXACT)
    assert "symbol" not in voom_result().table("T", "stim_vs_ctrl")


def test_samples_and_their_cells_are_the_cubes() -> None:
    cube = cells_cube()
    samples = voom_result().samples
    for group in ("NK", "T"):
        rows = group_rows(voom_result(), group)
        assert rows["sample_id"].tolist() == cube.samples["sample_id"].tolist()
        g = cube.groups["group_id"].tolist().index(group)
        assert rows["n_cells"].tolist() == cube.n_cells[:, g].tolist()
        assert rows["used"].tolist() == (cube.n_cells[:, g] >= 10).tolist()
    used = samples["used"]
    # Library sizes are the counts of the tested features, as edgeR recomputes them.
    nk = group_rows(voom_result(), "NK")
    tested = np.isin(
        cube.features["feature_id"],
        voom_result().table("NK", "stim_vs_ctrl")["feature_id"],
    )
    counts = cube.layer("counts", "NK")[:, tested].sum(axis=1)
    assert np.array_equal(nk["library_size"][nk["used"]], counts[nk["used"]])
    assert np.all(np.isnan(samples["library_size"][~used]))
    # TMM factors have geometric mean one over the samples of a fit.
    factors = nk["norm_factor"][nk["used"]]
    assert np.exp(np.mean(np.log(factors))) == pytest.approx(1.0, rel=1e-12)


# Rank-deficient groups.


def test_an_estimable_contrast_is_the_same_in_a_full_rank_parametrisation() -> None:
    # The NK design without donor D1 is rank deficient, and its kept columns are
    # (Intercept), donorD2, donorD3 and conditionstim. On a cube without D1, the
    # design ~ 0 + donor + condition has full rank and other columns, a mean for each
    # donor, and D2 vs D4 combines two of them. Exact standard errors do not depend
    # on the parametrisation, so every statistic agrees up to rounding.
    cube = cells_cube()
    others = [sid for sid in cube.samples["sample_id"] if not sid.startswith("D1")]
    smaller = cube.select(samples=others, groups="NK")
    full_rank = Design(
        smaller.samples,
        "~ 0 + donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim", "D2_vs_D4": "donorD2 - donorD4"},
    )
    assert full_rank.columns == ("donorD2", "donorD3", "donorD4", "conditionstim")
    assert full_rank.check().rank == 4
    for function in (limma_voom, limma_trend):
        expected = function(smaller, full_rank)
        result = function(cube, paired_design(cube.samples), groups="NK")
        assert result.provenance["groups"][0]["aliased_columns"] == ["donorD4"]
        for contrast in ("stim_vs_ctrl", "D2_vs_D4"):
            assert_tables_agree(
                result.table("NK", contrast), expected.table("NK", contrast), EXACT
            )


def test_limmas_approximation_depends_on_the_parametrisation() -> None:
    # In the T cells D2 vs D4 combines two coefficients, donorD2 - donorD4, or is the
    # single coefficient donorD2 with D4 as the reference. Exact standard errors agree;
    # limma's approximation with voom weights does not.
    cube = cells_cube()
    releveled = paired_design(
        cube.samples,
        levels={"condition": ["ctrl", "stim"], "donor": ["D4", "D1", "D2", "D3"]},
        contrasts={"D2_vs_D4": "donorD2"},
    )
    t_values = {}
    for method in ("exact", "limma"):
        for name, design in (
            ("original", paired_design(cube.samples)),
            ("releveled", releveled),
        ):
            result = limma_voom(
                cube, design, groups="T", contrast_standard_errors=method
            )
            t_values[method, name] = result.table("T", "D2_vs_D4")["t"]
    exact = scaled_difference(
        t_values["exact", "original"], t_values["exact", "releveled"], False
    )
    limma = scaled_difference(
        t_values["limma", "original"], t_values["limma", "releveled"], False
    )
    assert exact <= EXACT
    assert limma > 1e-6


# Groups that cannot be fitted.


def test_a_group_without_residual_df_is_reported_and_the_others_fitted() -> None:
    cube = simulated_cube({"common": [20] * 8, "rare": [20, 20, 20, 0, 3, 0, 0, 5]})
    design = paired_design(cube.samples)
    for function in (limma_voom, limma_trend):
        result = function(cube, design)
        summary = result.summary
        assert summary["status"].tolist() == ["fitted", "no_residual_df"]
        assert "no residual degrees of freedom" in summary["reason"][1]
        assert summary["n_samples_used"].tolist() == [8, 3]
        assert summary["df_residual"].tolist() == [3.0, 0.0]
        assert [key[0] for key in result.tables] == ["common"] * 3
        with pytest.raises(KeyError, match=r"'rare' was not fitted \(no_residual_df\)"):
            result.table("rare", "stim_vs_ctrl")
        rare = group_rows(result, "rare")
        assert rare["reason"].tolist()[3:5] == [
            "0 cells, fewer than min_cells (10)",
            "3 cells, fewer than min_cells (10)",
        ]


def test_groups_without_samples_features_or_estimable_contrasts() -> None:
    cube = simulated_cube({"empty": [0, 2, 9, 0, 1, 0, 3, 4], "full": [30] * 8})
    design = paired_design(cube.samples)
    result = limma_voom(cube, design)
    assert result.summary["status"].tolist() == ["no_samples", "fitted"]
    strict = limma_voom(
        cube, design, groups="full", expression_filter={"min_count": 1e9}
    )
    assert strict.summary["status"].tolist() == ["no_features"]
    assert strict.summary["reason"][0] == "no feature passed the expression filter"
    d1_gone = simulated_cube({"g": [5, 5, 20, 20, 20, 20, 20, 20]})
    only = paired_design(d1_gone.samples, contrasts={"D2_vs_D1": "donorD2"})
    result = limma_trend(d1_gone, only)
    assert result.summary["status"].tolist() == ["no_estimable_contrast"]
    assert result.summary["not_estimable"].tolist() == ["D2_vs_D1"]
    assert result.tables == {}


def test_a_sample_without_counts_is_left_out() -> None:
    cube = from_bulk(
        np.random.default_rng(5).poisson(50.0, (6, 40)) * (np.arange(6) != 2)[:, None],
        samples={
            "sample_id": [f"S{i}" for i in range(6)],
            "condition": ["ctrl", "stim"] * 3,
        },
    )
    design = Design(cube.samples, "~ condition", contrasts={"stim": "conditionstim"})
    result = limma_voom(cube, design)  # min_cells: no cell numbers to compare
    rows = group_rows(result, "bulk")
    assert "n_cells" not in rows
    assert rows["used"].tolist() == [True, True, False, True, True, True]
    assert rows["reason"][2] == "no counts"
    assert result.summary["df_residual"].tolist() == [3.0]


def bulk_study(counts: Any) -> tuple[Cube, Design]:
    """A bulk cube of these counts, two groups of samples, and its design."""
    counts = np.asarray(counts)
    n = counts.shape[0]
    groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    samples = {"sample_id": [f"S{i}" for i in range(n)], "group": groups}
    cube = from_bulk(counts, samples=samples)
    return cube, Design(cube.samples, "~ group", contrasts={"B_vs_A": "groupB"})


def test_groups_that_cannot_be_fitted_say_why_and_the_run_goes_on() -> None:
    rng = np.random.default_rng(9)
    # A sample whose only counts are in a feature that the filter removes.
    counts = np.zeros((6, 30), dtype=np.int64)
    counts[:5, :29] = rng.poisson(200.0, (5, 29))
    counts[5, 29] = 3
    cube, design = bulk_study(counts)
    result = limma_voom(cube, design)
    assert result.summary["status"].tolist() == ["empty_library"]
    assert result.summary["reason"][0] == (
        "S5 has no counts in the 29 features that passed the expression filter"
    )
    # voom fits its trend to two features with counts or more.
    counts = np.zeros((6, 3), dtype=np.int64)
    counts[:, 0] = rng.poisson(100.0, 6)
    cube, design = bulk_study(counts)
    result = limma_voom(cube, design, expression_filter=False)
    assert result.summary["status"].tolist() == ["too_few_features"]
    assert result.summary["reason"][0].endswith("1 of the 3 features tested has counts")
    assert limma_trend(cube, design, expression_filter=False).summary["status"][0] == (
        "fitted"
    )
    # Counts of five genes whose voom trend, with the span of 1 that voom chooses
    # for so few genes, falls to -1.04 where weights are read from it; limma would
    # take its inverse fourth power anyway, a weight without meaning.
    trend = [
        [36, 130, 103, 652, 25],
        [15, 121, 161, 669, 43],
        [14, 134, 106, 680, 8],
        [49, 107, 123, 462, 39],
    ]
    cube, design = bulk_study(trend)
    result = limma_voom(cube, design, expression_filter=False, normalization=None)
    assert result.summary["status"].tolist() == ["failed"]
    assert result.summary["reason"][0].startswith(
        "voom: the mean-variance trend falls to -1.04"
    )
    assert result.provenance["groups"][0]["voom_span"] is None
    with pytest.raises(KeyError, match=r"not fitted \(failed\): voom: the mean"):
        result.table("bulk", "B_vs_A")


def test_a_cube_without_features_or_counts() -> None:
    cube, design = bulk_study(np.zeros((4, 0)))
    result = limma_voom(cube, design)
    assert result.summary["status"].tolist() == ["no_features"]
    assert result.summary["reason"][0] == "the cube has no features"
    cube, design = bulk_study(np.zeros((4, 5)))
    result = limma_trend(cube, design)
    assert result.summary["status"].tolist() == ["no_samples"]
    assert result.summary["reason"][0] == (
        "every one of the 4 samples was left out (no counts)"
    )


# The settings.


def test_a_filter_that_keeps_everything_equals_no_filter() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    everything = {"min_count": 0, "min_total_count": 0}
    kept = limma_voom(cube, design, expression_filter=everything)
    unfiltered = limma_voom(cube, design, expression_filter=False)
    for key, table in unfiltered.tables.items():
        assert kept.tables[key].equals(table)
        assert len(table) == cube.n_features
    assert kept.settings["expression_filter"] == {
        "min_count": 0.0,
        "min_total_count": 0.0,
        "large_n": 10.0,
        "min_prop": 0.7,
    }
    assert unfiltered.settings["expression_filter"] is None


def test_without_normalization_factors_they_are_one() -> None:
    cube = cells_cube()
    result = limma_trend(cube, paired_design(cube.samples), normalization=None)
    used = result.samples["used"]
    assert np.all(result.samples["norm_factor"][used] == 1.0)
    assert result.settings["normalization"] is None
    assert result.settings["prior_count"] == 3.0


def test_the_number_of_threads_does_not_change_the_results() -> None:
    # One thread fits without a parallel region; more than numba starts use them all.
    cube = cells_cube()
    design = paired_design(cube.samples)
    many = numba.config.NUMBA_NUM_THREADS + 3
    one, every = (
        limma_voom(cube, design, n_threads=threads, contrast_standard_errors="limma")
        for threads in (1, many)
    )
    for key, table in one.tables.items():
        assert every.tables[key].equals(table)
        assert voom_result().tables[key].equals(table)
    # The settings record n_threads as given; limma-trend has no parallel step.
    assert [one.settings["n_threads"], every.settings["n_threads"]] == [1, many]
    assert "n_threads" not in limma_trend(cube, design).settings


def test_min_cells_decides_which_samples_enter() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    lenient = limma_voom(cube, design, min_cells=3)
    assert group_rows(lenient, "NK")["used"].all()
    assert lenient.provenance["groups"][0]["aliased_columns"] == []
    assert lenient.settings["min_cells"] == 3


# Saved cubes and memory.


class CountingLayer:
    """A lazy layer that records which groups are read."""

    def __init__(self, array: np.ndarray) -> None:
        self.array, self.shape, self.dtype = array, array.shape, array.dtype
        self.reads: list[Any] = []

    def __getitem__(self, index: Any) -> np.ndarray:
        self.reads.append(index)
        return self.array[index]


def test_a_saved_cube_is_read_one_group_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cube = cells_cube()
    write_cube(cube, tmp_path / "cells.cube.zarr")
    opened = read_cube(tmp_path / "cells.cube.zarr")
    design = paired_design(cube.samples)
    reads: list[Any] = []
    read = ZarrArray.__getitem__
    monkeypatch.setattr(
        ZarrArray, "__getitem__", lambda self, key: reads.append(key) or read(self, key)
    )
    result = limma_voom(opened, design, contrast_standard_errors="limma")
    # One whole read per group; the fingerprint and digests the record holds are
    # those stored in the file, which those reads checked, and cost no read.
    assert reads == [0, 1]
    for key, table in voom_result().tables.items():
        assert result.tables[key].equals(table)
    assert result.provenance["cube"]["fingerprint"] == cube.fingerprint()
    # A lazy layer of another array-like can change, so its digests are not kept:
    # the record reads each group once more, after the fits.
    layer = CountingLayer(np.asarray(cube.layer("counts")))
    lazy = Cube(
        {"counts": layer},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    limma_trend(lazy, design)
    assert layer.reads == [0, 1, 0, 1]


# The record.


def test_the_provenance_record_traces_every_number_to_its_profiles() -> None:
    cube = cells_cube()
    result = voom_result()
    record = result.provenance
    text = json.dumps(record, allow_nan=False)
    assert json.loads(text) == record
    # The cube: its fingerprint, the digest of every part, and how it was made.
    assert record["cube"]["fingerprint"] == cube.fingerprint()
    assert record["cube"]["digests"] == cube.digests()
    assert len(record["cube"]["digests"]["layers"]["counts"]["groups"]) == 2
    assert record["cube"]["provenance"] == cube.provenance
    assert record["cube"]["provenance"]["construction"]["function"] == "from_cells"
    assert record["layer"] == "counts"
    assert set(record["versions"]) == {"cytepool", "numpy", "numba", "python"}
    design = record["design"]
    assert design["formula"] == "~ donor + condition"
    assert design["levels"] == {
        "donor": ["D1", "D2", "D3", "D4"],
        "condition": ["ctrl", "stim"],
    }
    weights = dict(zip(design["columns"], design["contrasts"]["D2_vs_D4"]))
    assert weights == {
        "(Intercept)": 0.0,
        "donorD2": 1.0,
        "donorD3": 0.0,
        "donorD4": -1.0,
        "conditionstim": 0.0,
    }
    assert design["samples"]["sample_id"] == cube.samples["sample_id"].tolist()
    for entry in record["groups"]:
        rows = group_rows(result, entry["group_id"])
        assert entry["samples"] == rows["sample_id"][rows["used"]].tolist()
        assert entry["left_out"] == {
            sid: reason
            for sid, reason in zip(rows["sample_id"], rows["reason"])
            if reason
        }
        table = result.table(entry["group_id"], "stim_vs_ctrl")
        assert entry["n_features_tested"] == len(table)
        assert 0.0 < entry["voom_span"] <= 1.0
    assert record["settings"] == result.settings
    assert result.settings["span"] is None
    assert result.settings["contrast_standard_errors"] == "limma"


def test_the_result_is_immutable_and_survives_pickling() -> None:
    result = voom_result()
    with pytest.raises(AttributeError, match="immutable"):
        result.summary = Table()  # type: ignore[misc]
    restored = pickle.loads(pickle.dumps(result))
    assert restored.provenance == result.provenance
    for key, table in result.tables.items():
        assert restored.tables[key].equals(table)
    assert "NK: 6 samples" in repr(result)
    assert "not estimable: D2_vs_D1" in repr(result)


def test_a_result_holds_checked_objects_and_a_json_record() -> None:
    result = voom_result()
    parts = {
        "method": result.method,
        "contrasts": result.contrasts,
        "summary": result.summary,
        "samples": result.samples,
        "tables": dict(result.tables),
        "provenance": result.provenance,
    }
    rebuilt = DifferentialExpressionResult(**parts)
    assert rebuilt.provenance == result.provenance
    key = next(iter(result.tables))
    for changes, error, pattern in [
        ({"method": "voom"}, ValueError, "'limma_voom' or 'limma_trend'"),
        ({"contrasts": result.contrasts.matrix}, TypeError, "must be a Contrasts"),
        ({"summary": result.summary.to_dict()}, TypeError, "summary must be a Table"),
        ({"tables": {key[0]: result.tables[key]}}, TypeError, "keyed by"),
        ({"tables": {key: result.tables[key].to_dict()}}, TypeError, "be a Table"),
        ({"provenance": {"x": np.ones(2)}}, TypeError, "not JSON serializable"),
        ({"provenance": {"x": float("nan")}}, ValueError, "Out of range float"),
    ]:
        with pytest.raises(error, match=pattern):
            DifferentialExpressionResult(**(parts | changes))


# Errors.


def test_random_effects_are_refused_with_what_can_be_fitted() -> None:
    cube = cells_cube()
    design = Design(
        cube.samples,
        "~ condition",
        random="(1 | donor)",
        contrasts={"stim": "conditionstim"},
    )
    with pytest.raises(
        ValueError,
        match=r"limma_voom fits fixed effects only, so it "
        r"would drop the random effect \(1 \| donor\)",
    ) as error:
        limma_voom(cube, design)
    assert "Cube.collapse_samples('donor')" in str(error.value)
    assert "block=" not in str(error.value)


def test_samples_are_matched_by_id_or_the_mismatch_is_named() -> None:
    cube = cells_cube()
    table = cube.samples.to_dict()
    unknown = dict(
        table,
        sample_id=np.array(
            [*table["sample_id"][:-2], "D9_ctrl", "D9_stim"], dtype=object
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"2 of the design's 8 samples are not samples "
        r"of the cube: 'D9_ctrl' and 'D9_stim'",
    ):
        limma_voom(cube, paired_design(unknown))
    twice = dict(
        table,
        sample_id=np.array(
            [*table["sample_id"][:-1], table["sample_id"][0]], dtype=object
        ),
    )
    with pytest.raises(ValueError, match="repeats the sample id"):
        limma_voom(cube, paired_design(twice))
    nameless = {k: v for k, v in table.items() if k != "sample_id"}
    with pytest.raises(ValueError, match="no 'sample_id' column"):
        limma_voom(cube, paired_design(nameless))
    with pytest.raises(TypeError, match="design must be a Design"):
        limma_voom(cube, paired_design(table).matrix)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cube must be a Cube"):
        limma_trend(cube.layer("counts"), paired_design(table))  # type: ignore[arg-type]


def test_contrasts_are_required_and_matched_by_name() -> None:
    cube = cells_cube()
    plain = paired_design(cube.samples, contrasts=None)
    with pytest.raises(ValueError, match="no contrasts to test.*contrasts="):
        limma_voom(cube, plain)
    given = limma_voom(
        cube,
        plain,
        contrasts={"D2_vs_D4": "donorD2 - donorD4"},
        contrast_standard_errors="limma",
        groups="T",
    )
    assert given.table("T", "D2_vs_D4").equals(voom_result().table("T", "D2_vs_D4"))
    with pytest.raises(ValueError, match="not a column of the design"):
        limma_voom(cube, plain, contrasts={"x": "donorD9"})


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"groups": "B"}, KeyError, "no group with id 'B'; groups are: NK and T"),
        ({"groups": ["T", "T"]}, ValueError, "'T' is listed more than once"),
        ({"groups": 3}, TypeError, "groups must be a group id"),
        ({"groups": []}, ValueError, "groups is empty"),
        ({"min_cells": -1}, ValueError, "min_cells must be at least 0"),
        ({"min_cells": 2.5}, TypeError, "min_cells must be a whole number"),
        ({"min_cells": True}, TypeError, "not bool"),
        (
            {"expression_filter": {"min_counts": 5}},
            ValueError,
            "unknown setting 'min_counts'",
        ),
        ({"expression_filter": {"min_prop": 2}}, ValueError, r"must be in \[0, 1\]"),
        ({"expression_filter": {"min_count": -1}}, ValueError, "non-negative"),
        ({"expression_filter": "edgeR"}, TypeError, "True, False or a mapping"),
        ({"normalization": "upper_quartile"}, ValueError, "'tmm'.*or None"),
        ({"normalization": True}, TypeError, "'tmm'.*or None"),
        ({"span": 0}, ValueError, r"span must be in \(0, 1\]"),
        ({"span": 1.5}, ValueError, r"span must be in \(0, 1\]"),
        ({"span": "0.5"}, TypeError, "span must be a real number"),
        ({"span": True}, TypeError, "span must be a real number"),
        ({"robust": 1}, TypeError, "robust must be True or False"),
        ({"contrast_standard_errors": "approximate"}, ValueError, "'exact', 'limma'"),
        ({"contrast_standard_errors": 1}, TypeError, "'exact', 'limma'"),
        ({"n_threads": 0}, ValueError, "n_threads must be at least 1"),
    ],
)
def test_invalid_options_are_refused(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    cube = cells_cube()
    with pytest.raises(error, match=message):
        limma_voom(cube, paired_design(cube.samples), **options)


def test_the_prior_count_must_be_positive() -> None:
    cube = cells_cube()
    with pytest.raises(ValueError, match="prior_count must be positive"):
        limma_trend(cube, paired_design(cube.samples), prior_count=0)
