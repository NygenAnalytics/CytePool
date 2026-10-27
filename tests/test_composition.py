"""Tests for propeller, the test of differences in cell-type composition.

Reference values in ``tests/data/composition_tests/speckle.npz`` come from speckle's
``propeller``, ``propeller.ttest`` and ``propeller.anova``, run on one label per cell
for the cell numbers stored with them; ``validation/composition_tests.py`` writes the
file. The other tests compare with values derived by hand from the published
formulas, and check what a wrong implementation would break: the same tables in every
parametrisation of a comparison and for any order or naming of the cell types and
samples, a t-statistic that changes sign with the comparison, an F-test of two levels
that equals their t-test, the symmetry of two complementary cell types, and cell types
and samples without cells that change nothing.
"""

import math
import pickle
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool import (
    CompositionResult,
    Cube,
    Design,
    Table,
    from_bulk,
    from_cells,
    propeller,
)

DATA = Path(__file__).parent / "data" / "composition_tests"
EXACT = 1e-10
# limma solves the robust prior df of 10 cell types or more only to a tolerance.
CLOSE = 1e-6
LEVELS = {"condition": ["ctrl", "stim"]}
P_VALUES = {"p_value": "P.Value", "adjusted_p_value": "FDR"}
STATISTICS = ("t", "f_statistic")

# Cell numbers whose proportions are simple fractions; S5 has a single cell type.
CELLS = {
    "sample_id": ["S1", "S2", "S3", "S4", "S5"],
    "T": [1, 2, 0, 3, 0],
    "B": [3, 2, 4, 1, 6],
    "NK": [0, 4, 4, 0, 0],
}
SAMPLES = {
    "sample_id": ["S1", "S2", "S3", "S4", "S5"],
    "condition": ["ctrl", "ctrl", "stim", "stim", "ctrl"],
}
STIM_VS_CTRL = {"stim_vs_ctrl": "conditionstim - conditionctrl"}


@cache
def speckle() -> dict[str, np.ndarray]:
    with np.load(DATA / "speckle.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def fixture_cube(case: str) -> Cube:
    """The cube from_cells pools from one label per cell, as speckle read them."""
    data = speckle()
    n_cells, groups = data[f"{case} n_cells"], data[f"{case} groups"].tolist()
    samples = data[f"{case} sample_id"].tolist()
    per_entry = n_cells.reshape(-1)  # sample by sample
    sample_labels = np.repeat(np.repeat(samples, len(groups)), per_entry)
    group_labels = np.repeat(np.tile(groups, len(samples)), per_entry)
    columns = [key.split(" ", 1)[1] for key in data if key.startswith(f"{case} ")]
    variables = [c for c in columns if "|" not in c and c not in ("n_cells", "groups")]
    return from_cells(
        np.zeros((sample_labels.size, 1)),
        sample_labels=sample_labels,
        group_labels=group_labels,
        samples={name: data[f"{case} {name}"] for name in variables},
        groups={"group_id": groups},
        layers="counts",
    )


def covariate_design(formula: str = "~ 0 + group + age", **options: Any) -> Design:
    contrasts = {
        "B_vs_A": "groupB - groupA",
        "AB_vs_C": "(groupA + groupB)/2 - groupC",
        "age": "age",
        "A_plus_B": "groupA + groupB",
    }
    options.setdefault("contrasts", contrasts)
    return Design(fixture_cube("covariate").samples, formula, **options)


def scaled_difference(ours: np.ndarray, reference: np.ndarray, kind: str) -> float:
    """|ours - reference| over |reference| for proportions and ratios, over
    max(|reference|, 1) for statistics, and over max(p, 1e-300) max(1, |log p|) for
    p-values: about the error in log p relative to |log p|."""
    if kind == "p":
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    elif kind == "statistic":
        scale = np.maximum(np.abs(reference), 1.0)
    else:
        scale = np.abs(reference)
    return float(np.max(np.abs(ours - reference) / scale))


def assert_matches_speckle(
    table: Table, prefix: str, columns: dict[str, str | None], tolerance: float
) -> None:
    """Compare the columns of a table with speckle's, by cell type.

    ``columns`` maps every column of the table but ``group_id``, in order, to
    speckle's, or to None for one that speckle does not report. Proportions and
    ratios are compared in the exact tier, and statistics, p-values and FDRs within
    ``tolerance``.
    """
    data = speckle()
    rows = data[f"{prefix} | row"].tolist()
    assert sorted(rows) == sorted(table["group_id"].tolist())
    order = [rows.index(group) for group in table["group_id"].tolist()]
    assert table.columns == ("group_id", *columns)
    for ours, theirs in columns.items():
        if theirs is None:
            continue
        kind = "p" if ours in P_VALUES else "statistic" if ours in STATISTICS else ""
        difference = scaled_difference(
            table[ours], data[f"{prefix} | {theirs}"][order], kind
        )
        assert difference <= (tolerance if kind else EXACT), (ours, difference)


def two_group_design(samples: Any = SAMPLES, **options: Any) -> Design:
    options.setdefault("contrasts", STIM_VS_CTRL)
    return Design(samples, "~ 0 + condition", levels=LEVELS, **options)


def assert_tables_equal(first: Table, second: Table, tolerance: float = 1e-12) -> None:
    assert first.columns == second.columns
    assert first["group_id"].tolist() == second["group_id"].tolist()
    for column in first.columns[1:]:
        np.testing.assert_allclose(first[column], second[column], rtol=tolerance)


# Against speckle.


@pytest.mark.parametrize(
    ("label", "options"),
    [
        ("logit", {}),
        ("asin", {"transform": "asin"}),
        ("logit, robust = FALSE", {"robust": False}),
        ("logit, trend = TRUE", {"trend": True}),
    ],
)
def test_two_groups_match_speckle_propeller(
    label: str, options: dict[str, Any]
) -> None:
    cube = fixture_cube("two_groups")
    contrasts = {"ctrl_vs_stim": "conditionctrl - conditionstim"}
    design = two_group_design(cube.samples, contrasts=contrasts)
    result = propeller(cube, design, **options)

    columns = {
        "baseline_proportion": "BaselineProp.Freq",
        "mean_proportion_ctrl": "PropMean.ctrl",
        "mean_proportion_stim": "PropMean.stim",
        "proportion_ratio": "PropRatio",
        "t": "Tstatistic",
        **P_VALUES,
    }
    # Robust moderation of 10 cell types is solved by limma only to a tolerance.
    tolerance = EXACT if label.endswith("robust = FALSE") else CLOSE
    table = result.table("ctrl_vs_stim")
    assert_matches_speckle(table, f"two_groups | {label}", columns, tolerance)
    if label == "logit":
        # Robust moderation gave the cell types prior df of their own.
        assert np.ptp(result.summary["df_prior"]) > 0.1


@pytest.mark.parametrize(
    ("transform", "contrast", "levels"),
    [("logit", "B_vs_A", "AB"), ("logit", "AB_vs_C", "ABC"), ("asin", "B_vs_A", "AB")],
)
def test_contrasts_with_a_covariate_match_speckle_ttest(
    transform: str, contrast: str, levels: str
) -> None:
    result = propeller(
        fixture_cube("covariate"), covariate_design(), transform=transform
    )

    columns: dict[str, str | None] = {"baseline_proportion": None}
    columns |= {f"mean_proportion_{lv}": f"PropMean.group{lv}" for lv in levels}
    columns |= {"proportion_ratio": "PropRatio", "t": "Tstatistic", **P_VALUES}
    prefix = f"covariate | {transform}, {contrast}"
    assert_matches_speckle(result.table(contrast), prefix, columns, EXACT)


@pytest.mark.parametrize(
    ("contrast", "reference"), [("age", "age, against limma"), ("A_plus_B", "A + B")]
)
def test_contrasts_that_compare_no_levels_have_no_ratio(
    contrast: str, reference: str
) -> None:
    # The slope of age, which speckle refuses, against limma's eBayes on speckle's
    # transformed proportions; and A + B, for which speckle reports a "ratio".
    table = propeller(fixture_cube("covariate"), covariate_design()).table(contrast)

    columns: dict[str, str | None] = {"baseline_proportion": None}
    columns |= {"t": "Tstatistic", **P_VALUES}
    assert_matches_speckle(table, f"covariate | logit, {reference}", columns, EXACT)


def test_only_comparisons_of_levels_have_mean_proportions() -> None:
    samples = {**SAMPLES, "age": [30.0, 41.0, 35.0, 52.0, 47.0]}
    for formula, contrast in (
        ("~ 0 + condition", "conditionctrl + conditionstim"),
        ("~ condition", "(Intercept) + conditionstim"),
        ("~ condition + age", "age"),
        ("~ condition + age", "conditionstim + age"),
        ("~ 0 + condition", "conditionctrl"),
    ):
        design = Design(samples, formula, levels=LEVELS, contrasts={"c": contrast})
        table = propeller(CELLS, design).table("c")
        assert table.columns == (
            "group_id",
            "baseline_proportion",
            "t",
            "p_value",
            "adjusted_p_value",
        ), contrast


def test_f_test_with_a_covariate_matches_speckle_anova() -> None:
    cube = fixture_cube("covariate")
    result = propeller(cube, covariate_design(), contrasts=[], f_tests="group")

    assert list(result.tables) == ["group"] and result.f_tests == {
        "group": ("A", "B", "C")
    }
    columns: dict[str, str | None] = {"baseline_proportion": None}
    columns |= {f"mean_proportion_{lv}": f"PropMean.group{lv}" for lv in "ABC"}
    columns |= {"f_statistic": "Fstatistic", **P_VALUES}
    prefix = "covariate | logit, F-test of group"
    assert_matches_speckle(result.table("group"), prefix, columns, EXACT)


# Values derived by hand.


def test_proportions_and_transforms_follow_the_published_formulas() -> None:
    design = two_group_design()
    logit = propeller(CELLS, design).proportions
    asin = propeller(CELLS, design, transform="asin").proportions

    assert logit["group_id"].tolist() == ["T"] * 5 + ["B"] * 5 + ["NK"] * 5
    assert logit["sample_id"].tolist() == SAMPLES["sample_id"] * 3
    counts = [x for group in ("T", "B", "NK") for x in CELLS[group]]
    totals = [4, 8, 8, 4, 6] * 3
    assert logit["n_cells"].tolist() == counts
    proportions = [Fraction(x, n) for x, n in zip(counts, totals, strict=True)]
    assert logit["proportion"].tolist() == [float(p) for p in proportions]
    # logit(q) with q = (x + 1/2) / (n + 3/2), half a cell added to each of the
    # three counts (Phipson et al. 2022, supplementary section 1).
    q = [Fraction(2 * x + 1, 2 * n + 3) for x, n in zip(counts, totals, strict=True)]
    expected = [math.log(qi / (1 - qi)) for qi in q]
    np.testing.assert_allclose(logit["transformed_proportion"], expected, rtol=1e-15)
    # arcsin(sqrt(p)) of 0, 1/4, 1/2, 3/4 and 1 is 0, pi/6, pi/4, pi/3 and pi/2.
    angles = {
        0: 0.0,
        0.25: math.pi / 6,
        0.5: math.pi / 4,
        0.75: math.pi / 3,
        1: math.pi / 2,
    }
    expected = [angles[float(p)] for p in proportions]
    np.testing.assert_allclose(
        asin["transformed_proportion"], expected, rtol=1e-15, atol=0.0
    )


def test_baseline_and_mean_proportions_and_ratio_by_hand() -> None:
    table = propeller(CELLS, two_group_design()).table("stim_vs_ctrl")

    # 30 cells: 6 T, 16 B and 8 NK cells.
    np.testing.assert_allclose(
        table["baseline_proportion"], [6 / 30, 16 / 30, 8 / 30], rtol=1e-15
    )
    # ctrl: S1, S2 and S5; stim: S3 and S4.
    ctrl = [(1 / 4 + 1 / 4 + 0) / 3, (3 / 4 + 1 / 4 + 1) / 3, (0 + 1 / 2 + 0) / 3]
    stim = [(0 + 3 / 4) / 2, (1 / 2 + 1 / 4) / 2, (1 / 2 + 0) / 2]
    np.testing.assert_allclose(table["mean_proportion_ctrl"], ctrl, rtol=1e-15)
    np.testing.assert_allclose(table["mean_proportion_stim"], stim, rtol=1e-15)
    np.testing.assert_allclose(
        table["proportion_ratio"], [9 / 4, 9 / 16, 3 / 2], rtol=1e-15
    )


def test_the_table_of_the_only_contrast_needs_no_name() -> None:
    result = propeller(CELLS, two_group_design())
    assert result.table().equals(result.table("stim_vs_ctrl"))
    several = propeller(fixture_cube("covariate"), covariate_design())
    message = "the result has 4 contrasts; name one with contrast, one of 'B_vs_A'"
    with pytest.raises(ValueError, match=message):
        several.table()


def test_a_weighted_contrast_has_the_weighted_geometric_ratio() -> None:
    contrasts = {"S_vs_C": "(conditionstim + conditionctrl)/2 - conditionctrl"}
    table = propeller(CELLS, two_group_design(contrasts=contrasts)).table("S_vs_C")

    # stim/2 - ctrl/2: the square root of the ratio of means.
    np.testing.assert_allclose(
        table["proportion_ratio"], np.sqrt([9 / 4, 9 / 16, 3 / 2]), rtol=1e-15
    )


# Invariants.


def test_every_parametrisation_of_a_comparison_gives_the_same_table() -> None:
    cube = fixture_cube("covariate")
    group_means = propeller(cube, covariate_design()).table("B_vs_A")

    for formula, contrast in (
        ("~ group + age", "groupB"),
        ("~ age + group", "groupB"),
        ("~ 0 + age + group", "groupB - groupA"),
    ):
        design = covariate_design(formula, contrasts={"B_vs_A": contrast})
        assert_tables_equal(propeller(cube, design).table("B_vs_A"), group_means)
    f_tests = [
        propeller(cube, covariate_design(formula, contrasts={}), f_tests="group").table(
            "group"
        )
        for formula in ("~ 0 + group + age", "~ group + age", "~ age + group")
    ]
    assert_tables_equal(f_tests[1], f_tests[0])
    assert_tables_equal(f_tests[2], f_tests[0])


def test_renaming_and_reordering_cell_types_and_samples_changes_nothing() -> None:
    cube = fixture_cube("two_groups")
    contrasts = {"ctrl_vs_stim": "conditionctrl - conditionstim"}
    design = two_group_design(cube.samples, contrasts=contrasts)
    table = propeller(cube, design).table("ctrl_vs_stim")
    names = cube.groups["group_id"].tolist()
    renamed = cube.regroup({f"type {name}": name for name in names})
    reordered = renamed.select(groups=[f"type {name}" for name in reversed(names)])
    shuffled = design.samples.take(
        np.random.default_rng(1).permutation(len(design.samples))
    )

    other = propeller(reordered, two_group_design(shuffled, contrasts=contrasts))
    flipped = other.table("ctrl_vs_stim").take(np.arange(len(names))[::-1])
    assert flipped["group_id"].tolist() == [f"type {name}" for name in names]
    for column in table.columns[1:]:
        np.testing.assert_allclose(flipped[column], table[column], rtol=1e-12)


def test_swapping_the_conditions_negates_t_and_inverts_the_ratio() -> None:
    result = propeller(CELLS, two_group_design())
    swapped = {**SAMPLES, "condition": ["stim", "stim", "ctrl", "ctrl", "stim"]}
    other = propeller(CELLS, two_group_design(swapped))

    table, flipped = result.table("stim_vs_ctrl"), other.table("stim_vs_ctrl")
    np.testing.assert_allclose(flipped["t"], -table["t"], rtol=1e-13)
    np.testing.assert_allclose(flipped["p_value"], table["p_value"], rtol=1e-13)
    np.testing.assert_allclose(
        flipped["proportion_ratio"], 1.0 / table["proportion_ratio"], rtol=1e-15
    )
    np.testing.assert_array_equal(
        flipped["mean_proportion_ctrl"], table["mean_proportion_stim"]
    )


def test_proportions_that_do_not_change_give_no_difference() -> None:
    # Every stim sample has the cells of a ctrl sample.
    cells = {
        "sample_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
        "T": [10, 25, 7] * 2,
        "B": [30, 12, 40] * 2,
        "NK": [5, 9, 3] * 2,
    }
    samples = {
        "sample_id": cells["sample_id"],
        "condition": ["ctrl"] * 3 + ["stim"] * 3,
    }
    for transform in ("logit", "asin"):
        table = propeller(cells, two_group_design(samples), transform=transform).table(
            "stim_vs_ctrl"
        )
        np.testing.assert_allclose(table["t"], 0.0, atol=1e-12)
        np.testing.assert_allclose(table["p_value"], 1.0, rtol=1e-12)
        np.testing.assert_allclose(table["proportion_ratio"], 1.0, rtol=1e-15)


def test_an_f_test_of_two_levels_is_the_two_sided_t_test() -> None:
    cube = fixture_cube("two_groups")
    contrasts = {"ctrl_vs_stim": "conditionctrl - conditionstim"}
    result = propeller(
        cube, two_group_design(cube.samples, contrasts=contrasts), f_tests="condition"
    )
    table = result.table("ctrl_vs_stim")

    assert result.f_tests == {"condition": ("ctrl", "stim")}
    np.testing.assert_allclose(
        result.table("condition")["f_statistic"], table["t"] ** 2, rtol=1e-12
    )
    # limma caps the df of t-tests at the residual df of all cell types, and not
    # those of F-tests; here the cap is not reached, so the p-values agree.
    df_total = result.summary["df_residual"] + result.summary["df_prior"]
    assert np.all(df_total < np.sum(result.summary["df_residual"]))
    np.testing.assert_allclose(
        result.table("condition")["p_value"], table["p_value"], rtol=1e-12
    )


def test_f_tests_of_two_factors_are_those_of_each_factor_alone() -> None:
    cube = fixture_cube("covariate")
    # A batch that crosses the groups A, B, C, A, B, C, A, B, C.
    samples = cube.samples.with_columns(
        {"batch": ["x", "x", "y", "y", "y", "x", "x", "y", "x"]}
    )
    formula = "~ 0 + group + batch + age"
    design = Design(samples, formula, contrasts={"B_vs_A": "groupB - groupA"})
    both = propeller(cube, design, f_tests=["group", "batch"])

    assert both.f_tests == {"group": ("A", "B", "C"), "batch": ("x", "y")}
    assert list(both.tables) == ["B_vs_A", "group", "batch"]
    for factor in ("group", "batch"):
        alone = propeller(cube, design, contrasts=[], f_tests=factor)
        assert both.table(factor).equals(alone.table(factor))
        assert both.tables[factor].equals(alone.table())
    assert both.table("B_vs_A").equals(propeller(cube, design).table("B_vs_A"))
    # An F-test is named by its factor, so a contrast of that name is refused.
    named = Design(samples, formula, contrasts={"batch": "batchy"})
    with pytest.raises(
        ValueError, match="f_tests names 'batch', which is also the name of a contrast"
    ):
        propeller(cube, named, f_tests=["group", "batch"])


def test_two_complementary_cell_types_have_opposite_statistics() -> None:
    # With two cell types the logit of one is minus that of the other, with the half
    # cell added to both counts.
    cells = {
        "sample_id": CELLS["sample_id"],
        "T": [3, 9, 4, 12, 7],
        "B": [20, 15, 30, 9, 11],
    }
    for transform in ("logit", "asin"):
        result = propeller(cells, two_group_design(), transform=transform)
        table = result.table("stim_vs_ctrl")
        if transform == "logit":
            np.testing.assert_allclose(table["t"][1], -table["t"][0], rtol=1e-12)
            np.testing.assert_allclose(
                table["p_value"][1], table["p_value"][0], rtol=1e-12
            )
        # limma does not moderate two variances.
        assert result.summary["df_prior"].tolist() == [0.0, 0.0]


# Samples and cell types without cells, and other inputs.


def test_a_cell_type_without_cells_is_left_out_and_changes_nothing() -> None:
    cells = {**CELLS, "DC": [0, 0, 0, 0, 0]}
    result = propeller(cells, two_group_design())
    expected = propeller(CELLS, two_group_design())

    assert_tables_equal(result.table("stim_vs_ctrl"), expected.table("stim_vs_ctrl"), 0)
    summary = result.summary
    assert summary["group_id"].tolist() == ["T", "B", "NK", "DC"]
    assert summary["tested"].tolist() == [True, True, True, False]
    assert summary["reason"].tolist()[3] == "no cells in the samples used"
    assert np.isnan(summary["df_prior"][3])
    assert result.provenance["groups_left_out"] == {
        "DC": "no cells in the samples used"
    }


def test_a_sample_without_cells_is_left_out_and_changes_nothing() -> None:
    cells = {
        key: [*values, "S6" if key == "sample_id" else 0]
        for key, values in CELLS.items()
    }
    samples = {
        "sample_id": [*SAMPLES["sample_id"], "S6"],
        "condition": [*SAMPLES["condition"], "stim"],
    }
    for transform in ("logit", "asin"):
        result = propeller(cells, two_group_design(samples), transform=transform)
        expected = propeller(CELLS, two_group_design(), transform=transform)
        assert_tables_equal(
            result.table("stim_vs_ctrl"), expected.table("stim_vs_ctrl"), 0
        )
        assert result.samples["used"].tolist() == [True] * 5 + [False]
        assert result.samples["reason"].tolist()[5] == "no cells in any group"
        assert result.provenance["left_out"] == {"S6": "no cells in any group"}


def test_a_sample_whose_cells_are_one_cell_type_has_finite_values() -> None:
    # S5 has 6 B cells and no others: logit(6.5 / 7.5) = log(6.5) for B, and
    # logit(0.5 / 7.5) = log(1 / 14) for T and NK.
    proportions = propeller(CELLS, two_group_design()).proportions
    s5 = proportions.filter(proportions["sample_id"] == "S5")
    np.testing.assert_allclose(
        s5["transformed_proportion"],
        [math.log(1 / 14), math.log(13 / 2), math.log(1 / 14)],
        rtol=1e-15,
    )
    assert s5["proportion"].tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("cells", "transform"),
    [
        # Two cell types with the same cells: every logit is exactly 0.
        ({"T": [5, 10, 3, 7, 4], "B": [5, 10, 3, 7, 4]}, "logit"),
        # Every proportion is 1/2: the arcsines are equal up to rounding.
        ({"T": [5, 10, 3, 7, 4], "B": [5, 10, 3, 7, 4]}, "asin"),
        # The samples of each condition are copies of one another.
        (
            {"T": [5, 5, 9, 9, 5], "B": [20, 20, 30, 30, 20], "NK": [3, 3, 1, 1, 3]},
            "logit",
        ),
    ],
)
def test_data_the_design_fits_exactly_are_refused(
    cells: dict[str, list[int]], transform: str
) -> None:
    data = {"sample_id": CELLS["sample_id"], **cells}
    with pytest.raises(ValueError, match="fits the transformed proportions of every"):
        propeller(data, two_group_design(), transform=transform)


def test_a_table_of_cell_numbers_gives_the_result_of_the_cube() -> None:
    pandas = pytest.importorskip("pandas")
    cube = fixture_cube("two_groups")
    contrasts = {"ctrl_vs_stim": "conditionctrl - conditionstim"}
    design = two_group_design(cube.samples, contrasts=contrasts)
    expected = propeller(cube, design)
    ids = cube.samples["sample_id"].tolist()
    columns = {g: cube.n_cells[:, k] for k, g in enumerate(cube.groups["group_id"])}
    frame = pandas.DataFrame(columns, index=pandas.Index(ids, name="donor"))

    for data in (
        {"sample_id": ids, **columns},
        Table({"sample_id": ids, **columns}),
        frame,  # indexed by sample ids, as pandas.crosstab gives them
        frame.reset_index(names="sample_id").set_index("sample_id", drop=False),
    ):
        result = propeller(data, design)
        assert_tables_equal(
            result.table("ctrl_vs_stim"), expected.table("ctrl_vs_stim"), 0
        )
        assert result.provenance["input"] == "table"
        # The record identifies the same cell numbers as the cube's.
        digests = result.provenance["cube"]["digests"]
        assert digests["n_cells"] == expected.provenance["cube"]["digests"]["n_cells"]


def test_the_result_records_the_cube_and_settings_and_pickles() -> None:
    cube = fixture_cube("covariate")
    result = propeller(
        cube, covariate_design(), f_tests="group", transform="asin", trend=True
    )

    record = result.provenance
    assert record["method"] == "propeller" and record["input"] == "cube"
    assert record["cube"]["fingerprint"] == cube.fingerprint()
    assert record["settings"] == result.settings
    assert result.settings | {"winsor_tail_p": None} == {
        "transform": "asin",
        "robust": True,
        "trend": True,
        "f_tests": ["group"],
        "logit_offset": 0.5,
        "winsor_tail_p": None,
        "p_value_adjustment": "BH",
    }
    assert record["f_tests"]["group"]["levels"] == ["A", "B", "C"]
    assert record["design"]["formula"] == "~ 0 + group + age"
    assert record["samples"] == cube.samples["sample_id"].tolist()
    copy = pickle.loads(pickle.dumps(result))
    assert copy.provenance == record
    assert copy.table("B_vs_A").equals(result.table("B_vs_A"))
    assert copy.f_tests == result.f_tests == {"group": ("A", "B", "C")}
    assert copy.table("group").equals(result.table("group"))
    with pytest.raises(AttributeError, match="immutable"):
        result.settings = {}  # type: ignore[misc]
    with pytest.raises(
        KeyError, match="no contrast 'C_vs_A' was tested; contrasts are"
    ):
        result.table("C_vs_A")
    assert "F-test across the 3 levels of group" in repr(result)
    assert max(map(len, repr(result).splitlines())) <= 80
    assert isinstance(result, CompositionResult)


# Errors.


def bulk_cube() -> Cube:
    return from_bulk(np.ones((5, 2)), samples=SAMPLES["sample_id"])


@pytest.mark.parametrize(
    ("data", "design", "options", "error", "message"),
    [
        (bulk_cube, two_group_design, {}, ValueError, "records no cell numbers"),
        (lambda: [1, 2], two_group_design, {}, TypeError, "data must be a Cube"),
        (
            lambda: {k: v for k, v in CELLS.items() if k != "sample_id"},
            two_group_design,
            {},
            ValueError,
            "has no 'sample_id' column",
        ),
        (
            lambda: {**CELLS, "T": [1.0, 2, 0, 3, 0]},
            two_group_design,
            {},
            TypeError,
            r"data\['T'\] must hold whole numbers",
        ),
        (
            lambda: {**CELLS, "T": [1, -2, 0, 3, 0]},
            two_group_design,
            {},
            ValueError,
            r"has -2 cells for sample 'S2'",
        ),
        (
            lambda: {**CELLS, "sample_id": ["S1", "S1", "S3", "S4", "S5"]},
            two_group_design,
            {},
            ValueError,
            r"data has the sample id\(s\) 'S1' more than once, in its column 'sample_id'",
        ),
        (
            lambda: {**CELLS, "sample_id": ["S1", None, "S3", "S4", "S5"]},
            two_group_design,
            {},
            ValueError,
            "data has no sample id in position 1 of its column 'sample_id'",
        ),
        (lambda: CELLS, lambda: SAMPLES, {}, TypeError, "design must be a Design"),
        (
            lambda: CELLS,
            lambda: two_group_design(
                {**SAMPLES, "sample_id": ["S1", "S2", "S3", "S4", "S9"]}
            ),
            {},
            ValueError,
            "are not samples of the table of cell numbers: 'S9'",
        ),
        (
            lambda: {"sample_id": [], "T": [], "B": []},
            two_group_design,
            {},
            ValueError,
            "the table of cell numbers has no rows",
        ),
        (
            lambda: CELLS,
            lambda: two_group_design(contrasts=None),
            {},
            ValueError,
            "nothing to test",
        ),
        (
            lambda: CELLS,
            two_group_design,
            {"transform": "probit"},
            ValueError,
            "transform must be one of 'logit', 'asin'",
        ),
        (
            lambda: CELLS,
            two_group_design,
            {"robust": "yes"},
            TypeError,
            "robust must be True or False",
        ),
        (
            lambda: CELLS,
            two_group_design,
            {"f_tests": 1},
            TypeError,
            "f_tests must be the name of a factor",
        ),
        (
            lambda: CELLS,
            two_group_design,
            {"f_tests": "donor"},
            ValueError,
            "not a variable of the formula",
        ),
        (
            lambda: {**CELLS, "T": [0] * 5, "NK": [0] * 5},
            two_group_design,
            {},
            ValueError,
            "1 group of the data has cells",
        ),
        (
            lambda: {"sample_id": ["S1", "S3"], "T": [1, 2], "B": [3, 4]},
            lambda: two_group_design(
                {"sample_id": ["S1", "S3"], "condition": ["ctrl", "stim"]}
            ),
            {},
            ValueError,
            "no residual degrees of freedom",
        ),
        (
            lambda: {
                **CELLS,
                "T": [1, 2, 0, 0, 0],
                "B": [3, 2, 0, 0, 6],
                "NK": [0, 4, 0, 0, 0],
            },
            two_group_design,
            {},
            ValueError,
            "not of full column rank on the 3 samples used, once the 2 samples without cells",
        ),
    ],
)
def test_invalid_input_is_refused(
    data: Any, design: Any, options: dict[str, Any], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        propeller(data(), design(), **options)


def test_f_test_needs_a_factor_that_is_a_main_effect_only() -> None:
    cube = fixture_cube("covariate")
    with pytest.raises(ValueError, match="'age', a covariate"):
        propeller(cube, covariate_design(), f_tests="age")
    samples = {**fixture_cube("covariate").samples.to_dict()}
    samples["sex"] = np.array(["F", "M", "F"] * 3, dtype=object)
    design = Design(samples, "~ group * sex", contrasts={"B": "groupB"})
    with pytest.raises(ValueError, match="in the interaction group:sex"):
        propeller(cube, design, f_tests="group")
    with pytest.raises(ValueError, match="random effects? \\(1 \\| sex\\)"):
        propeller(
            cube,
            Design(samples, "~ group", random="(1 | sex)", contrasts={"B": "groupB"}),
        )
