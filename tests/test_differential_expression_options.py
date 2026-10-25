"""Tests for the options of limma_voom and limma_trend for pseudobulk data.

Reference values in ``tests/data/differential_expression_options`` come from limma and
edgeR run on the pseudobulk counts of the cells in ``cells.npz``, with each option:
``voomWithQualityWeights``, ``arrayWeights``, voom and ``lmFit`` with the number of
cells of each sample as prior weights, samples below a minimum total count left out,
and ``p.adjust`` over the p-values of every cell type stacked;
``validation/differential_expression_options.py`` writes both files. In that study,
D3_stim is an outlier in every cell type, D2_ctrl is sequenced shallowly in the T
cells, and both samples of D1 have fewer than 10 NK cells. The other tests check what
a wrong implementation would break: results that only the ratios of prior weights
decide, a cube without the samples that a minimum count leaves out, an adjustment
across one group that is the adjustment within it, families that hold only the
groups that test a contrast, weights in the record that reproduce a table through
the public functions, and options that change nothing else.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.cube import Cube, from_bulk, from_cells
from cytepool.design import Contrasts, Design
from cytepool.differential_expression import (
    DifferentialExpressionResult,
    limma_trend,
    limma_voom,
)
from cytepool.normalization import log_cpm
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.voom import voom, voom_with_quality_weights
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "differential_expression_options"
EXACT = 1e-10
# Sample quality weights are iterative estimates, and so is everything computed
# from them.
CLOSE = 1e-6
STATISTICS = (
    "log2_fold_change",
    "average_log2_cpm",
    "t",
    "p_value",
    "adjusted_p_value",
    "b_statistic",
)
ACROSS = "adjusted_p_value_across_groups"
CONTRASTS = {"stim_vs_ctrl": "conditionstim", "D3_vs_D2": "donorD3 - donorD2"}
LIMMA_ERRORS = {"contrast_standard_errors": "limma"}
MIN_COUNTS = 10_000  # leaves out D2_ctrl in the T cells, with 5,342 counts
# The analyses of the fixture, as the validation script ran them, with the tier of
# each and the weights they record.
Analysis = Callable[..., DifferentialExpressionResult]
ANALYSES: dict[str, tuple[Analysis, dict[str, Any], float]] = {
    "voom quality": (
        limma_voom,
        LIMMA_ERRORS | {"quality_weights": True, "adjust_across_groups": True},
        CLOSE,
    ),
    "voom cells": (
        limma_voom,
        LIMMA_ERRORS | {"prior_weights": "n_cells", "adjust_across_groups": True},
        EXACT,
    ),
    "trend quality": (
        limma_trend,
        {"quality_weights": True, "adjust_across_groups": True},
        CLOSE,
    ),
    "trend cells": (
        limma_trend,
        {
            "prior_weights": "n_cells",
            "min_counts": MIN_COUNTS,
            "adjust_across_groups": True,
        },
        EXACT,
    ),
}


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def cells_cube() -> Cube:
    """The cube that from_cells pools from the fixture's cells: groups NK and T."""
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


def paired_design(samples: Any) -> Design:
    return Design(
        samples,
        "~ donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts=CONTRASTS,
    )


@cache
def analysis(name: str) -> DifferentialExpressionResult:
    function, options, _ = ANALYSES[name]
    cube = cells_cube()
    return function(cube, paired_design(cube.samples), **options)


def scaled_difference(ours: Any, reference: Any, p_value: bool) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|): about the error in log p relative to |log p|."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale))


def assert_tables_agree(first: Table, second: Table, tolerance: float) -> None:
    assert first["feature_id"].tolist() == second["feature_id"].tolist()
    for column in STATISTICS:
        difference = scaled_difference(
            first[column], second[column], column.endswith("p_value")
        )
        assert difference <= tolerance, (column, difference)


def group_rows(result: DifferentialExpressionResult, group: str) -> Table:
    return result.samples.filter(result.samples["group_id"] == group)


def record_of(result: DifferentialExpressionResult, group: str) -> dict[str, Any]:
    return next(g for g in result.provenance["groups"] if g["group_id"] == group)


# Against limma.


@pytest.mark.parametrize("name", ANALYSES)
def test_options_match_limma(name: str) -> None:
    _, options, tolerance = ANALYSES[name]
    result = analysis(name)
    cube = cells_cube()
    limma = fixture("limma")
    for k, group in enumerate(result.groups):
        key = f"{name} {k}"
        rows = group_rows(result, group)
        assert rows["used"].tolist() == limma[f"{key} used"].astype(bool).tolist()
        record = record_of(result, group)
        assert record.get("voom_span") == (
            float(limma[f"{key} span"][0]) if f"{key} span" in limma else None
        )
        assert result.summary["df_residual"][k] == limma[f"{key} df_residual"][0]
        prior = float(np.median(limma[f"{key} df_prior"]))
        assert result.summary["prior_df"][k] == pytest.approx(prior, rel=tolerance)
        # The sample weights: limma's quality weights, or the number of cells.
        weights = record.get("quality_weights") or record["prior_weights"]
        np.testing.assert_allclose(weights, limma[f"{key} weights"], rtol=tolerance)
        tested = cube.features["feature_id"][limma[f"{key} keep"].astype(bool)]
        for contrast in CONTRASTS:
            table = result.table(group, contrast)
            assert table["feature_id"].tolist() == tested.tolist()
            reference = limma[f"{key} {contrast}"]
            for row, column in enumerate(STATISTICS):
                difference = scaled_difference(
                    table[column], reference[row], column.endswith("p_value")
                )
                assert difference <= tolerance, (group, contrast, column, difference)
            across = scaled_difference(
                table[ACROSS], limma[f"{key} {contrast} across"], True
            )
            assert across <= tolerance, (group, contrast, across)
    if options.get("min_counts"):
        # The shallow sample is left out of the T cells; the NK cells keep it.
        assert group_rows(result, "T")["reason"][2] == (
            "5342 counts, fewer than min_counts (10000)"
        )
        assert group_rows(result, "NK")["reason"][2] == ""


def test_quality_weights_find_the_outlying_donor() -> None:
    # D3_stim is an outlier in both cell types. In a paired design the two samples
    # of a donor have opposite residuals, so both of D3's samples get the lowest
    # weights. Weights multiply to one, as limma's do.
    for name in ("voom quality", "trend quality"):
        result = analysis(name)
        for group in result.groups:
            rows = group_rows(result, group)
            used = rows["used"]
            weights = rows["quality_weight"][used]
            lowest = set(rows["sample_id"][used][np.argsort(weights)[:2]])
            assert lowest == {"D3_ctrl", "D3_stim"}, (name, group)
            assert np.prod(weights) == pytest.approx(1.0, rel=1e-12)
            assert np.all(np.isnan(rows["quality_weight"][~used]))


# Every weight is in the result.


def test_the_record_holds_the_weights_that_reproduce_a_table() -> None:
    # The T-cell table of each analysis follows from the cube's counts, the
    # recorded library sizes, normalization factors and sample weights, through
    # the public functions, as the workflow runs them. voom estimates its quality
    # weights twice; voom_with_quality_weights reproduces the recorded ones.
    cube = cells_cube()
    design = paired_design(cube.samples)
    t = cube.groups["group_id"].tolist().index("T")
    for name in ANALYSES:
        result = analysis(name)
        record, rows = record_of(result, "T"), group_rows(result, "T")
        used = rows["used"]
        assert record["samples"] == rows["sample_id"][used].tolist()
        table = result.table("T", "stim_vs_ctrl")
        tested = np.isin(cube.features["feature_id"], table["feature_id"])
        counts = cube.layer("counts", t)[used][:, tested]
        assert np.array_equal(counts.sum(axis=1), rows["library_size"][used])
        factors = rows["norm_factor"][used]
        column = "prior_weight" if "cells" in name else "quality_weight"
        weights = np.array(record[f"{column}s"])
        assert rows[column][used].tolist() == weights.tolist()
        assert np.all(np.isnan(rows[column][~used]))
        if "cells" in name:
            assert weights.tolist() == cube.n_cells[used, t].tolist()
        # The T cells keep every column of the design.
        x = design.matrix[used]
        if name == "voom cells":
            weighted = voom(counts, x, norm_factors=factors, weights=weights)
        elif name == "voom quality":
            weighted = voom_with_quality_weights(counts, x, norm_factors=factors)
            assert weighted.sample_weights is not None
            np.testing.assert_array_equal(weighted.sample_weights, weights)
        if name.startswith("voom"):
            values, weights, method = weighted.log_cpm, weighted.weights, "limma"
        else:
            values = log_cpm(counts, norm_factors=factors, prior_count=3.0)
            method = "exact"
        model = fit_linear_model(
            values, x, weights=weights, coefficient_names=design.columns
        )
        contrast = Contrasts(
            design.contrasts.matrix[:, :1], ("stim_vs_ctrl",), design.columns
        )
        moderated = moderated_statistics(
            contrasts_fit(model, contrast, method=method), trend="trend" in name
        )
        np.testing.assert_allclose(table["t"], moderated.t[:, 0], rtol=1e-12)
        np.testing.assert_allclose(table["p_value"], moderated.p_value[:, 0], rtol=1e-9)


def test_settings_record_every_option() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    defaults = {
        "min_counts": 0.0,
        "prior_weights": None,
        "quality_weights": None,
        "adjust_across_groups": False,
    }
    for function, method in ((limma_voom, "gene_by_gene"), (limma_trend, "reml")):
        settings = function(cube, design, groups="T").settings
        assert {key: settings[key] for key in defaults} == defaults
        settings = function(
            cube, design, groups="T", quality_weights={"prior_n": 5}
        ).settings
        # limma's settings: voomWithQualityWeights estimates the weights gene by
        # gene, and arrayWeights chooses REML for log2 CPM.
        assert settings["quality_weights"] == {
            "method": method,
            "prior_n": 5.0,
            "max_iterations": 50,
            "tolerance": 1e-5,
        }
    assert analysis("trend cells").settings["min_counts"] == MIN_COUNTS
    assert analysis("voom cells").settings["prior_weights"] == "n_cells"
    assert analysis("trend quality").settings["adjust_across_groups"] is True


# Each option changes what it should and nothing else.


def test_options_change_nothing_else() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    for function in (limma_voom, limma_trend):
        plain = function(cube, design)
        adjusted = function(cube, design, adjust_across_groups=True)
        for key, table in plain.tables.items():
            columns = list(adjusted.tables[key].columns)
            # The new column follows adjusted_p_value; the others are unchanged.
            assert columns.index(ACROSS) == columns.index("adjusted_p_value") + 1
            columns.remove(ACROSS)
            assert adjusted.tables[key].select(columns).equals(table)
        assert adjusted.summary.equals(plain.summary)
        assert adjusted.samples.equals(plain.samples)
        # Weights change neither the samples, the features nor the normalization.
        for options in ({"prior_weights": "n_cells"}, {"quality_weights": True}):
            weighted = function(cube, design, **options)
            new = "prior_weight" if "prior_weights" in options else "quality_weight"
            columns = [c for c in weighted.samples.columns if c != new]
            assert weighted.samples.select(columns).equals(plain.samples)
            for column in ("n_samples_used", "n_features_tested", "df_residual"):
                assert (
                    weighted.summary[column].tolist() == plain.summary[column].tolist()
                )
        # A minimum count that leaves out a T-cell sample changes nothing in NK.
        fewer = function(cube, design, min_counts=MIN_COUNTS)
        for contrast in CONTRASTS:
            assert fewer.table("NK", contrast).equals(plain.table("NK", contrast))
        assert group_rows(fewer, "NK").equals(group_rows(plain, "NK"))


def test_leaving_out_by_min_counts_equals_a_cube_without_the_sample() -> None:
    cube = cells_cube()
    others = [sid for sid in cube.samples["sample_id"] if sid != "D2_ctrl"]
    smaller = cube.select(samples=others, groups="T")
    design = paired_design(smaller.samples)
    assert design.columns == paired_design(cube.samples).columns
    for function in (limma_voom, limma_trend):
        left_out = function(
            cube, paired_design(cube.samples), groups="T", min_counts=MIN_COUNTS
        )
        without = function(smaller, design)
        for contrast in CONTRASTS:
            assert left_out.table("T", contrast).equals(without.table("T", contrast))


def test_a_sample_with_exactly_the_minimum_count_is_kept() -> None:
    # As decoupler's filter_samples keeps it (found by probing).
    cube = cells_cube()
    design = paired_design(cube.samples)
    t = cube.groups["group_id"].tolist().index("T")
    total = float(cube.layer("counts", t)[2].sum())  # D2_ctrl's
    assert total == 5342.0
    at = limma_trend(cube, design, groups="T", min_counts=total)
    assert group_rows(at, "T")["used"][2]
    above = limma_trend(cube, design, groups="T", min_counts=total + 0.5)
    assert group_rows(above, "T")["reason"][2] == (
        "5342 counts, fewer than min_counts (5342.5)"
    )
    assert record_of(above, "T")["left_out"]["D2_ctrl"].startswith("5342 counts")


def test_min_counts_applies_to_bulk_samples_too() -> None:
    counts = np.random.default_rng(4).poisson(30.0, (6, 40))
    counts[1] //= 10  # about 120 counts in all
    cube = from_bulk(
        counts,
        samples={
            "sample_id": [f"S{i}" for i in range(6)],
            "condition": ["ctrl", "stim"] * 3,
        },
    )
    design = Design(cube.samples, "~ condition", contrasts={"stim": "conditionstim"})
    total = int(counts[1].sum())
    result = limma_voom(cube, design, min_counts=500)
    rows = group_rows(result, "bulk")
    assert rows["used"].tolist() == [True, False, True, True, True, True]
    assert rows["reason"][1] == f"{total} counts, fewer than min_counts (500)"
    nothing = limma_trend(cube, design, min_counts=1e9)
    assert nothing.summary["status"].tolist() == ["no_samples"]
    assert nothing.summary["reason"][0] == (
        "every one of the 6 samples was left out (fewer than 1000000000 counts)"
    )


# Prior weights from the number of cells.


def with_cells(cube: Cube, n_cells: np.ndarray) -> Cube:
    return Cube(
        {"counts": cube.layer("counts")},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
    )


def test_only_the_ratios_of_the_prior_weights_matter() -> None:
    # Ten times as many cells in every sample: the same samples enter each fit
    # (min_cells ten times larger), and every result is the same up to rounding.
    cube = cells_cube()
    design = paired_design(cube.samples)
    scaled = with_cells(cube, cube.n_cells * 10)
    for function in (limma_voom, limma_trend):
        once = function(cube, design, prior_weights="n_cells")
        tenfold = function(scaled, design, prior_weights="n_cells", min_cells=100)
        for key, table in once.tables.items():
            assert_tables_agree(tenfold.tables[key], table, EXACT)
        weights = once.samples["prior_weight"]
        np.testing.assert_array_equal(tenfold.samples["prior_weight"], weights * 10)


def test_equal_cell_numbers_are_no_weights() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    equal = with_cells(cube, np.full(cube.n_cells.shape, 20))
    for function in (limma_voom, limma_trend):
        weighted = function(equal, design, prior_weights="n_cells")
        plain = function(equal, design)
        for key, table in plain.tables.items():
            assert_tables_agree(weighted.tables[key], table, EXACT)


def test_prior_weights_need_cell_numbers() -> None:
    cube = from_bulk(
        np.random.default_rng(5).poisson(50.0, (4, 10)),
        samples={"sample_id": ["A", "B", "C", "D"], "group": ["x", "x", "y", "y"]},
    )
    design = Design(cube.samples, "~ group", contrasts={"y": "groupy"})
    with pytest.raises(ValueError, match="this cube does not record"):
        limma_voom(cube, design, prior_weights="n_cells")


# Sample quality weights.


def test_quality_weights_with_one_residual_df_are_one_with_a_warning() -> None:
    # With min_cells=16, the T cells keep 6 samples of 4 donors: one residual df.
    cube = cells_cube()
    design = paired_design(cube.samples)
    message = (
        "group 'T': sample quality weights need a design with at least two residual "
        "degrees of freedom, got 1; every weight is 1"
    )
    for function in (limma_voom, limma_trend):
        with pytest.warns(RuntimeWarning) as caught:
            result = function(
                cube, design, groups="T", min_cells=16, quality_weights=True
            )
        assert [str(w.message) for w in caught] == [message]
        assert caught[0].filename == __file__
        record = record_of(result, "T")
        assert result.summary["df_residual"].tolist() == [1.0]
        assert record["quality_weights"] == [1.0] * 6
        assert record["warnings"] == [message.removeprefix("group 'T': ")]
        plain = function(cube, design, groups="T", min_cells=16)
        assert record_of(plain, "T")["warnings"] == []
        for key, table in plain.tables.items():
            assert result.tables[key].equals(table)


def test_quality_weights_that_cannot_be_estimated_stop_only_their_group() -> None:
    # A copy of the T cells, "T2", whose only control sample with 10 cells or more
    # is D4_ctrl. By ~ condition that sample is fitted exactly, so it carries no
    # information about its weight; without a prior (prior_n=0) the information is
    # singular there, and only there.
    cube = cells_cube()
    t = cube.groups["group_id"].tolist().index("T")
    cells = cube.n_cells[:, t].copy()
    cells[[0, 2, 4]] = 5  # D1_ctrl, D2_ctrl and D3_ctrl
    counts = cube.layer("counts", t)
    two = Cube(
        {"counts": np.stack([counts, counts])},
        samples=cube.samples,
        groups={"group_id": ["T", "T2"]},
        features=cube.features,
        n_cells=np.column_stack([cube.n_cells[:, t], cells]),
    )
    design = Design(
        two.samples,
        "~ condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    for function, step in (
        (limma_voom, "voom with quality weights"),
        (limma_trend, "sample quality weights"),
    ):
        result = function(two, design, quality_weights={"prior_n": 0})
        assert result.summary["status"].tolist() == ["fitted", "failed"]
        assert result.summary["reason"][1] == (
            f"{step}: the information about the sample weights is singular: some "
            "samples carry no information about their variance, such as a sample "
            "that the design fits exactly, and quality_weights['prior_n'] (0) is too "
            "small to make up for it; a larger prior_n keeps their weights near 1"
        )
        assert record_of(result, "T2")["quality_weights"] is None
        assert np.all(np.isnan(group_rows(result, "T2")["quality_weight"]))
        assert len(record_of(result, "T")["quality_weights"]) == 8


def test_prior_and_quality_weights_are_alternatives() -> None:
    cube = cells_cube()
    for function in (limma_voom, limma_trend):
        with pytest.raises(ValueError, match="prior_weights or quality_weights, not"):
            function(
                cube,
                paired_design(cube.samples),
                prior_weights="n_cells",
                quality_weights={"method": "reml"},
            )


# p-values adjusted across groups.


def test_across_one_group_the_adjustment_is_the_one_within_it() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    for function in (limma_voom, limma_trend):
        result = function(cube, design, groups="T", adjust_across_groups=True)
        for table in result.tables.values():
            np.testing.assert_array_equal(table[ACROSS], table["adjusted_p_value"])


def test_the_adjustment_across_groups_does_not_depend_on_their_order() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    first = limma_voom(cube, design, groups=["T", "NK"], adjust_across_groups=True)
    second = limma_voom(cube, design, groups=["NK", "T"], adjust_across_groups=True)
    for key, table in first.tables.items():
        assert second.tables[key].equals(table)
    # The family stacks the tests of every group: by Benjamini-Hochberg, the
    # adjusted value of the i-th smallest of n p-values lies between it and n / i
    # times it, and the adjusted values rise with the p-values.
    p_values = np.concatenate(
        [first.table(g, "stim_vs_ctrl")["p_value"] for g in first.groups]
    )
    adjusted = np.concatenate(
        [first.table(g, "stim_vs_ctrl")[ACROSS] for g in first.groups]
    )
    order = np.argsort(p_values, kind="stable")
    ranks = np.arange(1, p_values.size + 1)
    assert np.all(adjusted[order] >= p_values[order])
    # The bound up to rounding of the division.
    bound = p_values[order] * p_values.size / ranks
    assert np.all(adjusted[order] <= bound * (1.0 + 1e-14))
    assert np.all(np.diff(adjusted[order]) >= 0.0)
    within = np.concatenate(
        [first.table(g, "stim_vs_ctrl")["adjusted_p_value"] for g in first.groups]
    )
    assert not np.array_equal(adjusted, within)


def test_the_family_holds_the_groups_that_test_the_contrast() -> None:
    # Without donor D1, the NK cells cannot estimate D2_vs_D1, so its family holds
    # the T cells alone, while both cell types test stim_vs_ctrl. A copy of the T
    # cells whose samples have too few cells, "T2", is not fitted and joins no
    # family: adding it changes nothing.
    cube = cells_cube()
    t = cube.groups["group_id"].tolist().index("T")
    counts = cube.layer("counts", t)
    three = Cube(
        {"counts": np.concatenate([cube.layer("counts"), counts[np.newaxis]])},
        samples=cube.samples,
        groups={"group_id": ["NK", "T", "T2"]},
        features=cube.features,
        n_cells=np.column_stack([cube.n_cells, np.full(cube.n_samples, 5)]),
    )
    design = Design(
        cube.samples,
        "~ donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim", "D2_vs_D1": "donorD2"},
    )
    result = limma_trend(three, design, adjust_across_groups=True)
    assert result.summary["status"].tolist() == ["fitted", "fitted", "no_samples"]
    assert ("NK", "D2_vs_D1") not in result.tables
    alone = result.table("T", "D2_vs_D1")
    np.testing.assert_array_equal(alone[ACROSS], alone["adjusted_p_value"])
    both = limma_trend(three, design, groups=["NK", "T"], adjust_across_groups=True)
    for key, table in both.tables.items():
        assert result.tables[key].equals(table)
    # stim_vs_ctrl's family stacks both cell types, which test_options_match_limma
    # checks against p.adjust; it is not the NK cells alone.
    nk = result.table("NK", "stim_vs_ctrl")
    assert not np.array_equal(nk[ACROSS], nk["adjusted_p_value"])


def test_none_or_false_turns_quality_weights_off() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    for function in (limma_voom, limma_trend):
        plain = function(cube, design, groups="T")
        for off in (None, False):
            result = function(cube, design, groups="T", quality_weights=off)
            assert result.settings["quality_weights"] is None
            assert "quality_weight" not in result.samples
            for key, table in plain.tables.items():
                assert result.tables[key].equals(table)


# Errors.


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"min_counts": -1}, ValueError, "min_counts must be non-negative"),
        ({"min_counts": float("inf")}, ValueError, "min_counts must be finite"),
        ({"min_counts": "1000"}, TypeError, "min_counts must be a real number"),
        ({"min_counts": True}, TypeError, "min_counts must be a real number"),
        ({"prior_weights": "cells"}, ValueError, "prior_weights must be 'n_cells'"),
        ({"prior_weights": True}, TypeError, "prior_weights must be 'n_cells'"),
        (
            {"quality_weights": "reml"},
            TypeError,
            "quality_weights must be True, a mapping of settings",
        ),
        ({"quality_weights": {"methods": "reml"}}, ValueError, "unknown setting"),
        (
            {"quality_weights": {"method": "auto"}},
            ValueError,
            r"quality_weights\['method'\] must be one of 'gene_by_gene', 'reml'",
        ),
        (
            {"quality_weights": {"prior_n": -1}},
            ValueError,
            r"quality_weights\['prior_n'\] must be finite and at least 0",
        ),
        (
            {"quality_weights": {"max_iterations": 0}},
            ValueError,
            r"quality_weights\['max_iterations'\] must be at least 1",
        ),
        (
            {"quality_weights": {"tolerance": 0.0}},
            ValueError,
            r"quality_weights\['tolerance'\] must be positive",
        ),
        ({"adjust_across_groups": 1}, TypeError, "adjust_across_groups must be True"),
    ],
)
def test_invalid_options_are_refused(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    cube = cells_cube()
    for function in (limma_voom, limma_trend):
        with pytest.raises(error, match=message):
            function(cube, paired_design(cube.samples), **options)
