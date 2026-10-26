"""Tests for over-representation analysis and its figure.

Reference values in ``tests/data/gene_set_files_and_ora`` come from limma's ``kegga``
(of a fit, which selects the genes up and down itself, and of lists of genes with a
universe), ``p.adjust`` and ``topKEGG``; ``validation/gene_set_files_and_ora.py``
writes them. The other tests check hypergeometric probabilities derived by hand,
invariants that a wrong implementation would break (the order of the sets, of the
genes and of the rows of a table; a universe that leaves genes out; sets that list
genes twice; the sign of the fold changes), the two ways kegga's fit differs on
purpose, a differential-expression result tested as a whole, edge cases and messages,
and what the figure draws.
"""

import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.gene_sets import GeneSets
from cytepool.over_representation import over_representation
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "gene_set_files_and_ora"
EXACT = 1e-10
LISTS = {"up": "Up", "down": "Down", "either": "Either", None: "DE"}
CASES = (
    "fit",
    "symbols",
    "missing_symbols",
    "restricted_universe",
    "gene_list",
    "missing_p_values",
)


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"ora_{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def case(name: str) -> tuple[Any, dict[str, list[str]], dict[str, Any]]:
    """The genes, the sets and the options that the validation gave CytePool."""
    data = fixture(name)
    ends = np.cumsum(data["set_sizes"])
    sets = {
        str(name): data["set_genes"][end - size : end].tolist()
        for name, size, end in zip(
            data["set_names"], data["set_sizes"], ends, strict=True
        )
    }
    if "genes" in data:
        genes: Any = data["genes"].tolist()
    else:
        columns = {
            key.removeprefix("table "): value
            for key, value in data.items()
            if key.startswith("table ")
        }
        columns["symbol"] = [None if s == "" else s for s in columns["symbol"].tolist()]
        genes = Table(columns)
    options: dict[str, Any] = {
        key.removeprefix("option "): value.tolist()
        for key, value in data.items()
        if key.startswith("option ")
    }
    return genes, sets, options


def kegga(name: str, key: str, prefix: str = "") -> Any:
    found = fixture(name)[f"kegga {prefix}{key}"]
    return found.tolist() if found.dtype.kind == "U" else found


def rows(table: Table, direction: str | None) -> Table:
    return table.filter(np.array([d == direction for d in table["direction"]]))


def assert_matches_kegga(table: Table, name: str, prefix: str = "") -> None:
    """Every set and direction of kegga's table: counts, p-values, adjusted p-values
    and topKEGG's order (p-values, ties by the number of genes, found by probing)."""
    pathways = kegga(name, "pathways", prefix)
    directions = [d for d in dict.fromkeys(table["direction"].tolist())]
    checked = 0
    for direction in directions:
        key = LISTS[direction]
        if f"kegga {prefix}P.{key}" not in fixture(name):
            continue
        block = rows(table, direction)
        row = {s: i for i, s in enumerate(block["gene_set"].tolist())}
        at = [row[s] for s in pathways]
        assert_array_equal(block["n_genes"][at], kegga(name, "N", prefix))
        assert_array_equal(block["overlap"][at], kegga(name, key, prefix))
        assert_allclose(
            block["p_value"][at], kegga(name, f"P.{key}", prefix), rtol=EXACT
        )
        assert_allclose(
            block["adjusted_p_value"][at], kegga(name, f"FDR.{key}", prefix), rtol=EXACT
        )
        order = np.lexsort((block["n_genes"], block["p_value"]))[: len(pathways)]
        assert [block["gene_set"][i] for i in order[:30]] == kegga(
            name, f"top {key}", prefix
        )
        # kegga leaves out the sets without genes in the universe.
        left_out = [row[s] for s in row if s not in set(pathways)]
        assert np.all(np.isnan(block["p_value"][left_out]))
        assert np.all(block["n_genes"][left_out] == 0)
        if not prefix:
            n = kegga(name, "selected")[list(LISTS.values()).index(key) % 3]
            universe = kegga(name, "universe")[0]
            assert np.all(block["n_selected"] == n)
            assert np.all(block["n_universe"] == universe)
            assert_allclose(
                block["expected"], block["n_genes"] * n / universe, rtol=1e-15
            )
        checked += 1
    assert checked == (2 if prefix else len(directions))


# kegga.


@pytest.mark.parametrize("name", CASES)
def test_every_set_and_direction_matches_kegga(name: str) -> None:
    genes, sets, options = case(name)
    table = over_representation(genes, sets, **options)

    assert_matches_kegga(table, name)
    assert len(table) == len(sets) * (1 if name == "gene_list" else 3)


@pytest.mark.parametrize("name", ["fit", "symbols"])
def test_genes_up_and_down_are_those_kegga_selects_from_a_fit(name: str) -> None:
    # kegga selects the genes whose adjusted p-value is below 0.05 itself, and in
    # "symbols" counts a symbol shared by two features once, up or down or both.
    genes, sets, options = case(name)

    assert_matches_kegga(over_representation(genes, sets, **options), name, "fit ")


def test_kegga_adjusts_the_features_with_an_id_and_cytepool_takes_the_results() -> None:
    genes, sets, options = case("missing_symbols")
    named = np.array([s is not None for s in genes["symbol"]])
    readjusted = np.full(len(genes), np.nan)
    readjusted[named] = adjust_p_values(genes["p_value"][named])

    # kegga of a fit adjusts the p-values of the features with a symbol only.
    again = genes.with_columns(adjusted_p_value=readjusted)
    assert_matches_kegga(
        over_representation(again, sets, **options), "missing_symbols", "fit "
    )
    # CytePool takes the adjusted p-values of the results: of 10 features, g10's is
    # 0.018 * 10 / 4 = 0.045, and over the 9 with a symbol it would be 0.054.
    p = np.array([0.001, 0.002, 0.003, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.018])
    table = Table(
        {
            "symbol": [None, *[f"g{i}" for i in range(2, 11)]],
            "log2_fold_change": np.ones(10),
            "adjusted_p_value": adjust_p_values(p),
        }
    )
    tested = over_representation(table, {"S": ["g10"]}, id_column="symbol")
    assert table["adjusted_p_value"][9] == pytest.approx(0.045, rel=1e-14)
    assert adjust_p_values(p[1:])[8] == pytest.approx(0.054, rel=1e-14)
    assert tested["overlap"].tolist() == [1, 0, 1]


def test_kegga_keeps_genes_without_a_p_value_in_the_universe_of_a_fit() -> None:
    genes, sets, _ = case("missing_p_values")

    with pytest.raises(AssertionError):
        assert_matches_kegga(
            over_representation(genes, sets), "missing_p_values", "fit "
        )
    everything = genes["feature_id"].tolist()
    assert_matches_kegga(
        over_representation(genes, sets, universe=everything),
        "missing_p_values",
        "fit ",
    )


# Values derived by hand.


def test_p_values_and_expected_overlaps_derived_by_hand() -> None:
    universe = [f"g{i}" for i in range(10)]
    sets = {
        "two of three": ["g0", "g1", "g2"],
        "one": ["g0"],
        "none selected": ["g3", "g4"],
        "everything": universe,
        "outside": ["x1"],
    }
    table = over_representation(["g0", "g1", "g5", "g6"], sets, universe=universe)

    # 4 draws from 10 genes: P(X >= 2) with 3 in the set is (C(3,2) C(7,2) +
    # C(3,3) C(7,1)) / C(10,4) = 70/210, and P(X >= 1) with 1 in it, 1 - 126/210.
    p = [70 / 210, 1 - 126 / 210, 1.0, 1.0, math.nan]
    assert_allclose(table["p_value"], p, rtol=1e-14)
    # Benjamini-Hochberg over the four sets with genes: 4/3 p_(i) / i, then minima.
    assert_allclose(
        table["adjusted_p_value"], [0.8, 0.8, 1.0, 1.0, math.nan], rtol=1e-14
    )
    assert_allclose(table["expected"], [1.2, 0.4, 0.8, 4.0, 0.0], rtol=1e-15)
    assert table["n_genes"].tolist() == [3, 1, 2, 10, 0]
    assert table["overlap"].tolist() == [2, 1, 0, 4, 0]
    assert table["overlap_genes"].tolist() == ["g0, g1", "g0", "", "g0, g1, g5, g6", ""]
    assert table["direction"].tolist() == [None] * 5
    assert set(table["n_selected"].tolist()) == {4}
    assert set(table["n_universe"].tolist()) == {10}


def test_a_p_value_far_in_the_tail_is_the_exact_hypergeometric_probability() -> None:
    universe = [f"g{i}" for i in range(1000)]
    table = over_representation(
        universe[:5], {"five": universe[:5], "half": universe[:3]}, universe=universe
    )

    # All 5 of 5 drawn genes in a set of 5: 1 / C(1000, 5); 3 of 3: C(997, 2) / C(1000, 5).
    assert_allclose(
        table["p_value"],
        [1 / math.comb(1000, 5), math.comb(997, 2) / math.comb(1000, 5)],
        rtol=1e-13,
    )


# Invariants.


@cache
def results() -> Table:
    genes, _, _ = case("fit")
    return genes


@cache
def sets_of_fit() -> dict[str, list[str]]:
    return case("fit")[1]


def test_the_order_of_sets_rows_and_genes_changes_nothing_else() -> None:
    table, sets = results(), sets_of_fit()
    tested = over_representation(table, sets)
    reversed_sets = dict(reversed(list(sets.items())))
    permutation = np.random.default_rng(1).permutation(len(table))

    again = over_representation(table, reversed_sets)
    for direction in ("up", "down", "either"):
        assert rows(again, direction).equals(
            rows(tested, direction).take(np.arange(len(sets))[::-1])
        )
    assert over_representation(table.take(permutation), sets).equals(tested)
    # The genes of a set in another order list the overlap in that order.
    shuffled = {name: list(dict.fromkeys(genes))[::-1] for name, genes in sets.items()}
    other = over_representation(table, shuffled)
    assert other.select(["n_genes", "overlap", "p_value"]).equals(
        tested.select(["n_genes", "overlap", "p_value"])
    )
    assert [text.split(", ")[::-1] for text in other["overlap_genes"]] == [
        text.split(", ") for text in tested["overlap_genes"]
    ]


def test_genes_listed_twice_and_a_collection_change_nothing() -> None:
    table, sets = results(), sets_of_fit()
    twice = {name: genes + genes[: len(genes) // 2] for name, genes in sets.items()}

    tested = over_representation(table, sets)
    assert over_representation(table, twice).equals(tested)
    assert over_representation(table, GeneSets(sets)).equals(tested)
    assert over_representation(table.to_pandas(), sets).equals(tested)


def test_a_universe_leaves_out_the_genes_outside_it() -> None:
    table, sets = results(), sets_of_fit()
    universe = table["feature_id"][::3].tolist() + ["Z1", "Z2"]
    inside = set(universe)
    with np.errstate(invalid="ignore"):
        selected = table["feature_id"][table["adjusted_p_value"] < 0.05].tolist()

    given = over_representation(selected, sets, universe=universe)
    # The same as a selection and sets that hold only genes of the universe.
    restricted = over_representation(
        [g for g in selected if g in inside],
        {name: [g for g in genes if g in inside] for name, genes in sets.items()},
        universe=universe,
    )
    assert given.equals(restricted)
    assert set(given["n_universe"].tolist()) == {len(universe)}
    # The genes of either direction of the table are those selected.
    either = rows(over_representation(table, sets, universe=universe), "either")
    columns = ["n_genes", "overlap", "p_value", "n_selected", "n_universe"]
    assert either.select(columns).equals(given.select(columns))


def test_the_sign_of_the_fold_changes_swaps_up_and_down() -> None:
    table, sets = results(), sets_of_fit()
    flipped = table.with_columns(log2_fold_change=-table["log2_fold_change"])

    tested, swapped = (
        over_representation(table, sets),
        over_representation(flipped, sets),
    )
    columns = ["gene_set", "n_genes", "overlap", "p_value", "overlap_genes"]
    assert (
        rows(swapped, "up").select(columns).equals(rows(tested, "down").select(columns))
    )
    assert (
        rows(swapped, "down").select(columns).equals(rows(tested, "up").select(columns))
    )
    assert rows(swapped, "either").equals(rows(tested, "either"))


def test_a_symbol_on_two_features_counts_once_in_each_direction() -> None:
    table = Table(
        {
            "symbol": ["A", "A", "B", "C", None, "D"],
            "log2_fold_change": [1.0, -1.0, 1.0, -0.5, 2.0, 0.3],
            "adjusted_p_value": [0.01, 0.02, 0.01, 0.5, 0.001, np.nan],
        }
    )
    tested = over_representation(table, {"set": ["A", "B", "D"]}, id_column="symbol")

    # The universe: A, B and C (D has no p-value, and a feature without a symbol is
    # left out); up: A and B; down: A; either: A and B.
    assert tested["n_universe"].tolist() == [3, 3, 3]
    assert tested["n_selected"].tolist() == [2, 1, 2]
    assert tested["overlap"].tolist() == [2, 1, 2]
    assert tested["n_genes"].tolist() == [2, 2, 2]
    assert tested["overlap_genes"].tolist() == ["A, B", "A", "A, B"]


# Edge cases.


def test_genes_are_selected_strictly_below_alpha_by_the_sign_of_the_change() -> None:
    # An infinite fold change has a direction, as a gene without counts in one
    # condition can have.
    table = Table(
        {
            "feature_id": ["a", "b", "c", "d"],
            "log2_fold_change": [1.0, np.inf, -np.inf, 1.0],
            "adjusted_p_value": [0.05, 0.0499, 0.01, 0.9],
        }
    )
    tested = over_representation(table, {"set": ["a", "b", "c"]})

    assert tested["overlap_genes"].tolist() == ["b", "c", "b, c"]
    assert tested["n_selected"].tolist() == [1, 1, 2]
    assert over_representation(table, {"set": ["a"]}, alpha=0.06)["overlap"][0] == 1


def test_lists_without_genes_give_p_values_of_one() -> None:
    table = Table(
        {
            "feature_id": ["a", "b", "c", "d"],
            "log2_fold_change": [1.0, 2.0, 1.0, 1.0],
            "adjusted_p_value": [0.01, 0.02, 0.5, 0.9],
        }
    )
    tested = over_representation(table, {"S": ["a", "c"], "T": ["d"]})
    down = rows(tested, "down")

    assert down["n_selected"].tolist() == [0, 0]
    assert down["p_value"].tolist() == [1.0, 1.0]
    assert down["adjusted_p_value"].tolist() == [1.0, 1.0]
    assert down["overlap_genes"].tolist() == ["", ""]


def test_a_set_without_genes_in_the_universe_is_left_out_of_the_adjustment() -> None:
    table, sets = results(), sets_of_fit()
    tested = over_representation(table, sets | {"elsewhere": ["X1", "X2"]})

    assert (
        rows(tested, "up")
        .take(np.arange(len(sets)))
        .equals(rows(over_representation(table, sets), "up"))
    )
    last = rows(tested, "up").take([len(sets)])
    assert last["n_genes"][0] == 0 and math.isnan(last["p_value"][0])
    assert math.isnan(last["adjusted_p_value"][0]) and last["expected"][0] == 0.0


def test_without_fold_changes_the_selected_genes_have_no_direction() -> None:
    table, sets = results(), sets_of_fit()
    plain = over_representation(table, sets, fold_change_column=None)

    assert plain["direction"].tolist() == [None] * len(sets)
    assert plain.select(["n_genes", "overlap", "p_value"]).equals(
        rows(over_representation(table, sets), "either").select(
            ["n_genes", "overlap", "p_value"]
        )
    )


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


def test_a_result_is_tested_as_its_table_of_a_group_and_contrast() -> None:
    result = de_result()
    sets = {"up": [f"g{i}" for i in range(0, 40, 2)], "other": ["g100", "g101", "g150"]}
    tested = over_representation(result, sets, group="T", contrast="stim")

    assert tested.equals(over_representation(result.table("T", "stim"), sets))
    up = rows(tested, "up")
    assert up["overlap"][0] > 0 and up["p_value"][0] < 1e-3
    assert over_representation(result, sets, group="T").equals(tested)
    with pytest.raises(
        ValueError, match="the result has 2 groups; name one with group"
    ):
        over_representation(result, sets)
    with pytest.raises(KeyError, match="no group 'NK' was analysed"):
        over_representation(result, sets, group="NK")
    with pytest.raises(TypeError, match="group must be a string"):
        over_representation(result, sets, group=1)  # type: ignore[arg-type]


# Messages.


@pytest.mark.parametrize(
    ("genes", "options", "error", "match"),
    [
        (["g1"], {}, ValueError, "a list of genes is tested against a universe"),
        (["g1"], {"group": "T"}, ValueError, "group and contrast choose a table"),
        ({"up": ["g1"]}, {}, TypeError, "got a mapping; test each list of genes"),
        (
            [1, 2],
            {"universe": ["g1"]},
            TypeError,
            "genes must hold gene ids as strings",
        ),
        (["g1"], {"universe": []}, ValueError, "universe holds no gene id"),
        (["x"], {"universe": ["g1", "g2"]}, ValueError, "none of the 1 selected genes"),
        (
            ["g1"],
            {"universe": ["y1"]},
            ValueError,
            "no gene of any set is in the universe",
        ),
        (["g1"], {"universe": ["g1"], "alpha": 1.0}, ValueError, "alpha must be in"),
        (
            ["g1"],
            {"universe": ["g1"], "alpha": True},
            TypeError,
            "alpha must be a real",
        ),
    ],
)
def test_invalid_arguments_are_refused_with_a_reason(
    genes: Any, options: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        over_representation(genes, {"set": ["g1", "g2"]}, **options)


def test_tables_that_cannot_be_read_are_refused_with_a_reason() -> None:
    table = Table(
        {
            "feature_id": ["a", "b"],
            "log2_fold_change": [1.0, np.nan],
            "adjusted_p_value": [0.01, 0.02],
        }
    )
    sets = {"set": ["a", "b"]}

    with pytest.raises(ValueError, match="the table has no column 'symbol'"):
        over_representation(table, sets, id_column="symbol")
    with pytest.raises(ValueError, match="is missing for 1 gene"):
        over_representation(table, sets)
    with pytest.raises(TypeError, match="must hold gene ids as text"):
        over_representation(table, sets, id_column="adjusted_p_value")
    with pytest.raises(TypeError, match="the column 'feature_id' must hold numbers"):
        over_representation(table, sets, p_value_column="feature_id")
    with pytest.raises(
        ValueError, match="holds values outside \\[0, 1\\], such as 2.5"
    ):
        over_representation(table.with_columns(t=[2.5, 0.1]), sets, p_value_column="t")
    with pytest.raises(TypeError, match="id_column must be a column name"):
        over_representation(table, sets, id_column=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gene_sets is empty"):
        over_representation(table, {})
    with pytest.raises(TypeError, match="gene_sets must be a GeneSets or a mapping"):
        over_representation(table, [["a"]])  # type: ignore[arg-type]
    untested = table.with_columns(adjusted_p_value=[np.nan, np.nan])
    with pytest.raises(ValueError, match="no gene of the table has a p-value"):
        over_representation(untested, sets)


# The figure.


def test_the_dot_plot_draws_the_top_sets_of_every_list() -> None:
    figure = pytest.importorskip("matplotlib.figure")
    colors = pytest.importorskip("matplotlib.colors")
    from cytepool import plotting

    table, sets = results(), sets_of_fit()
    tested = over_representation(table, sets)
    ax = plotting.plot_over_representation(
        tested, top=12, ax=figure.Figure().subplots()
    )
    dots = next(c for c in ax.collections if c.get_gid() == "gene sets")

    # The 12 sets with the smallest p-value in any list, the smallest at the top.
    best: dict[str, float] = {}
    for name, p in zip(tested["gene_set"], tested["p_value"], strict=True):
        if not math.isnan(p):
            best[name] = min(best.get(name, math.inf), p)
    shown = sorted(best, key=best.__getitem__)[:12]
    ticks = sorted(
        zip(ax.get_yticks(), ax.get_yticklabels(), strict=True), key=lambda t: -t[0]
    )
    n_genes = {s: n for s, n in zip(tested["gene_set"], tested["n_genes"], strict=True)}
    assert [label.get_text() for _, label in ticks] == [
        f"{s} ({n_genes[s]})" for s in shown
    ]
    labels = [label.get_text() for label in ax.get_xticklabels()]
    n = [rows(tested, d)["n_selected"][0] for d in ("up", "down", "either")]
    assert labels == [f"Up\n({n[0]})", f"Down\n({n[1]})", f"Either\n({n[2]})"]
    # A dot per set and list with genes in the overlap: its area in proportion to the
    # overlap, coloured where the adjusted p-value is below 0.05.
    position = {round(y): label.get_text().rsplit(" (", 1)[0] for y, label in ticks}
    found = []
    for (x, y), size, face in zip(
        dots.get_offsets(), dots.get_sizes(), dots.get_facecolors(), strict=True
    ):
        direction = ("up", "down", "either")[round(x)]
        block = rows(tested, direction)
        i = block["gene_set"].tolist().index(position[round(y)])
        found.append((block["overlap"][i], size, block["adjusted_p_value"][i] < 0.05,
                      tuple(face) != colors.to_rgba("#e1e0d9")))  # fmt: skip
    overlaps = np.array([f[0] for f in found], dtype=float)
    ratios = dots.get_sizes() / overlaps
    assert_allclose(ratios, ratios[0], rtol=1e-14)
    assert [f[2] for f in found] == [f[3] for f in found]
    expected = sum(
        int(
            rows(tested, d)["overlap"][rows(tested, d)["gene_set"].tolist().index(s)]
            > 0
        )
        for s in shown
        for d in ("up", "down", "either")
    )
    assert len(found) == expected


def test_the_dot_plot_of_a_list_and_its_messages() -> None:
    figure = pytest.importorskip("matplotlib.figure")
    from cytepool import plotting

    universe = [f"g{i}" for i in range(20)]
    tested = over_representation(universe[:6], {"A": universe[:4], "B": universe[10:]},
                                 universe=universe)  # fmt: skip
    ax = plotting.plot_over_representation(tested, ax=figure.Figure().subplots())

    assert [label.get_text() for label in ax.get_xticklabels()] == ["Selected\n(6)"]
    with pytest.raises(TypeError, match="table must be a Table"):
        plotting.plot_over_representation({"gene_set": ["A"]})  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="lacks the column\\(s\\) 'n_genes', 'overlap'"
    ):
        plotting.plot_over_representation(tested.select(["gene_set", "direction"]))
    empty = over_representation(
        ["g1"], {"A": ["x"], "B": ["g1"]}, universe=["g1", "g2"]
    )
    with pytest.raises(ValueError, match="no gene set of the table has a p-value"):
        plotting.plot_over_representation(empty.filter(np.array([True, False])))
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        plotting.plot_over_representation(tested, alpha=0.0)
    with pytest.raises(ValueError, match="top must be"):
        plotting.plot_over_representation(tested, top=0)
