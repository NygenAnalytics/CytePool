import copy
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.cube import Cube

SCANPY_FIXTURE = Path(__file__).parent / "data" / "cube" / "scanpy_aggregate.npz"
TARGET_SUM = 1e4
SAMPLES = ["S1", "S2", "S3", "S4"]
GROUPS = ["B", "T4", "T8", "NK"]
# Cells per sample (rows) and group (columns). S2 has no T4 cells, so that entry is
# empty, and a single NK cell, so its variance is undefined.
CELLS = np.array([[14, 9, 11, 6], [10, 0, 12, 1], [8, 13, 7, 9], [12, 10, 9, 8]])
N_FEATURES = 20
SAMPLE_TABLE = {
    "sample_id": SAMPLES,
    "donor": ["D1", "D2", "D1", "D1"],
    "condition": ["ctrl", "stim", "ctrl", "ctrl"],  # constant within each donor
    "age": [41.0, 29.0, 41.0, 41.0],  # constant within each donor
    "tissue": ["blood", "blood", "lung", "skin"],  # varies within donor D1
    "batch": [1, 2, 2, 1],  # varies within donor D1
}
GROUP_TABLE = {
    "group_id": GROUPS,
    "lineage": ["B", "T", "T", "NK"],
    "cell_ontology_id": ["CL:0000236", "CL:0000624", "CL:0000625", "CL:0000623"],
    "parent": ["lymphocyte"] * 4,
    "n_markers": [3, 4, 5, 2],
    "is_t_cell": [False, True, True, False],
}
FEATURE_TABLE = {
    "feature_id": [f"G{i:02d}" for i in range(N_FEATURES)],
    "symbol": [f"GENE{i}" for i in range(N_FEATURES)],
}
NORMALIZATION = {
    "method": "library_size",
    "target_sum": TARGET_SUM,
    "log": "natural",
    "pseudocount": 1.0,
}
INTEGER_LAYERS = ("counts", "detected")
FLOAT_LAYERS = ("log_sum", "log_sum_sq", "linear_sum")


def simulate_cells(seed: int = 20261022) -> dict[str, np.ndarray]:
    """Negative binomial counts for a few hundred cells, with their labels."""
    rng = np.random.default_rng(seed)
    samples, groups = [], []
    for s, sample in enumerate(SAMPLES):
        for g, group in enumerate(GROUPS):
            samples += [sample] * int(CELLS[s, g])
            groups += [group] * int(CELLS[s, g])
    order = rng.permutation(len(samples))
    sample_of_cell = np.array(samples)[order]
    group_of_cell = np.array(groups)[order]
    group_code = np.array([GROUPS.index(group) for group in group_of_cell])
    gene_mean = rng.lognormal(0.0, 1.0, size=N_FEATURES)
    group_effect = rng.lognormal(0.0, 0.7, size=(len(GROUPS), N_FEATURES))
    cell_size = rng.lognormal(0.0, 0.3, size=(order.size, 1))
    mean = gene_mean * group_effect[group_code] * cell_size
    counts = rng.negative_binomial(2, 2 / (2 + mean))
    counts[:, 0] = 0  # a feature that is never detected
    counts[counts.sum(axis=1) == 0, 1] = 1  # every cell has a positive library size
    linear = counts * TARGET_SUM / counts.sum(axis=1, keepdims=True)
    return {
        "sample": sample_of_cell,
        "group": group_of_cell,
        "counts": counts,
        "linear": linear,
        "log": np.log1p(linear),
    }


def aggregate(
    cells: dict[str, np.ndarray],
    sample_of_cell: np.ndarray,
    group_of_cell: np.ndarray,
    sample_ids: list[str],
    group_ids: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sum cells into cube layers, adding one cell at a time."""
    s = np.array([sample_ids.index(sample) for sample in sample_of_cell])
    g = np.array([group_ids.index(group) for group in group_of_cell])
    shape = (len(group_ids), len(sample_ids), cells["counts"].shape[1])
    layers = {name: np.zeros(shape, dtype=np.int64) for name in INTEGER_LAYERS}
    layers |= {name: np.zeros(shape) for name in FLOAT_LAYERS}
    np.add.at(layers["counts"], (g, s), cells["counts"])
    np.add.at(layers["detected"], (g, s), cells["counts"] > 0)
    np.add.at(layers["log_sum"], (g, s), cells["log"])
    np.add.at(layers["log_sum_sq"], (g, s), cells["log"] ** 2)
    np.add.at(layers["linear_sum"], (g, s), cells["linear"])
    n_cells = np.zeros((len(sample_ids), len(group_ids)), dtype=np.int64)
    np.add.at(n_cells, (s, g), 1)
    return layers, n_cells


@pytest.fixture(scope="module")
def cells() -> dict[str, np.ndarray]:
    return simulate_cells()


@pytest.fixture(scope="module")
def layers(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    built, n_cells = aggregate(cells, cells["sample"], cells["group"], SAMPLES, GROUPS)
    assert_array_equal(n_cells, CELLS)
    return built


@pytest.fixture(scope="module")
def cube(layers: dict[str, np.ndarray]) -> Cube:
    return make_cube(layers)


def make_cube(layers: dict[str, Any], **changes: Any) -> Cube:
    arguments: dict[str, Any] = {
        "samples": SAMPLE_TABLE,
        "groups": GROUP_TABLE,
        "features": FEATURE_TABLE,
        "n_cells": CELLS,
        "normalization": NORMALIZATION,
        "provenance": {"source": "simulated cells"},
    }
    arguments.update(changes)
    return Cube(layers, **arguments)


def entry_rows(cells: dict[str, np.ndarray], sample: str, group: str) -> np.ndarray:
    return (cells["sample"] == sample) & (cells["group"] == group)


class LazyArray:
    """Reads one group per integer key and fails if the whole array is converted."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.reads: list[int] = []

    def __getitem__(self, key: int) -> np.ndarray:
        if type(key) is not int:
            raise TypeError("only an integer on the first axis")
        self.reads.append(key)
        return self._array[key].copy()

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("the whole lazy array was converted")


# Statistics against direct computation from the cells.


def test_statistics_equal_direct_computation_from_the_cells(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    mean_log = cube.mean()
    mean_linear = cube.mean(scale="linear")
    variance = cube.variance()
    detected = cube.fraction_detected()
    library_sizes = cube.library_sizes()
    assert mean_log.shape == (4, 4, N_FEATURES)
    assert library_sizes.shape == (4, 4)
    assert library_sizes.dtype == np.int64
    for s, sample in enumerate(SAMPLES):
        for g, group in enumerate(GROUPS):
            rows = entry_rows(cells, sample, group)
            assert library_sizes[s, g] == cells["counts"][rows].sum()
            if not rows.any():
                for statistic in (mean_log, mean_linear, variance, detected):
                    assert np.isnan(statistic[g, s]).all()
                continue
            # The layers add cells one at a time; numpy's mean sums in another
            # order, so the two agree to rounding.
            assert_allclose(mean_log[g, s], cells["log"][rows].mean(axis=0), rtol=1e-13)
            assert_allclose(
                mean_linear[g, s], cells["linear"][rows].mean(axis=0), rtol=1e-13
            )
            assert_array_equal(
                detected[g, s], (cells["counts"][rows] > 0).sum(axis=0) / rows.sum()
            )
            if rows.sum() < 2:
                assert np.isnan(variance[g, s]).all()
            else:
                # Raw moments against numpy's two-pass variance: they differ by about
                # 1e-16 * (1 + mean**2 / variance), well inside the exact tier.
                assert_allclose(
                    variance[g, s],
                    np.var(cells["log"][rows], axis=0, ddof=1),
                    rtol=1e-10,
                )


def test_one_group_matches_the_matching_slice_of_all_groups(cube: Cube) -> None:
    assert_array_equal(cube.mean(group="T8"), cube.mean()[2])
    assert_array_equal(cube.variance(group=1), cube.variance()[1])
    assert_array_equal(cube.fraction_detected(group=-1), cube.fraction_detected()[3])
    assert_array_equal(cube.library_sizes(group="NK"), cube.library_sizes()[:, 3])


def test_empty_entries_give_nan_and_a_single_cell_has_no_variance(cube: Cube) -> None:
    assert np.isnan(cube.mean("T4")[1]).all()
    assert np.isnan(cube.mean("T4", scale="linear")[1]).all()
    assert np.isnan(cube.fraction_detected("T4")[1]).all()
    assert np.isnan(cube.variance("T4")[1]).all()
    assert (cube.layer("counts", "T4")[1] == 0).all()
    single = cube.variance("NK")[1]
    assert np.isnan(single).all()
    assert np.isfinite(cube.mean("NK")[1]).all()


def test_statistics_on_hand_computed_moments() -> None:
    # Sample A has 3 cells: log values 1, 2 and 4 for feature 0 (mean 7/3, variance
    # 7/3) and zeros for feature 1. Sample B has 7 cells: log values 1 and 0.5 in two
    # cells and zeros elsewhere for feature 0 (mean 3/14, variance 13/84), and 0.7 in
    # every cell for feature 1. Summed in float64, the moments of that constant
    # feature give a raw-moment numerator of -1.3e-15, which must come back as 0.
    # Sample C has a single cell, whose sum of squares (0.01) differs from the
    # square of its sum (0.1**2 = 0.010000000000000002) by rounding; one cell has
    # no sample variance whatever its moments.
    log_sum = np.array([[[7.0, 0.0], [1.5, 4.9], [0.1, 0.0]]])
    log_sum_sq = np.array([[[21.0, 0.0], [1.25, 3.4299999999999993], [0.01, 0.0]]])
    assert log_sum_sq[0, 1, 1] - log_sum[0, 1, 1] * (log_sum[0, 1, 1] / 7) < 0
    cube = Cube(
        {
            "counts": np.array([[[9, 0], [5, 7], [1, 0]]]),
            "detected": np.array([[[3, 0], [2, 7], [1, 0]]]),
            "log_sum": log_sum,
            "log_sum_sq": log_sum_sq,
            "linear_sum": np.array([[[30.0, 0.0], [12.0, 7.0], [0.2, 0.0]]]),
        },
        samples={"sample_id": ["A", "B", "C"]},
        groups={"group_id": ["g"]},
        features={"feature_id": ["f0", "f1"]},
        n_cells=np.array([[3], [7], [1]]),
        normalization=NORMALIZATION,
    )
    assert_allclose(
        cube.mean("g"), [[7 / 3, 0.0], [3 / 14, 0.7], [0.1, 0.0]], rtol=1e-15
    )
    variance = cube.variance("g")
    assert_allclose(variance[:2, 0], [7 / 3, 13 / 84], rtol=1e-14)
    assert variance[0, 1] == 0.0 and variance[1, 1] == 0.0
    assert np.isnan(variance[2]).all()
    assert_allclose(
        cube.mean("g", scale="linear"),
        [[10.0, 0.0], [12 / 7, 1.0], [0.2, 0.0]],
        rtol=1e-15,
    )
    assert_allclose(
        cube.fraction_detected("g"), [[1.0, 0.0], [2 / 7, 1.0], [1.0, 0.0]], rtol=1e-15
    )
    assert cube.library_sizes("g").tolist() == [9, 12, 1]


# Regrouping and collapsing against re-aggregation from the cells.


def assert_layers_equal(actual: Cube, expected: dict[str, np.ndarray]) -> None:
    for name in INTEGER_LAYERS:
        assert actual.layer(name).dtype == np.int64
        assert_array_equal(actual.layer(name), expected[name])
    for name in FLOAT_LAYERS:
        # Sums of entry sums and sums of cells add in different orders.
        assert_allclose(actual.layer(name), expected[name], rtol=1e-12, atol=0)


def test_regroup_equals_aggregation_with_merged_labels(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    merged = cube.regroup({"T": ["T8", "T4"]})
    labels = np.where(np.isin(cells["group"], ["T4", "T8"]), "T", cells["group"])
    expected, n_cells = aggregate(
        cells, cells["sample"], labels, SAMPLES, ["B", "T", "NK"]
    )
    assert merged.groups["group_id"].tolist() == ["B", "T", "NK"]
    assert_array_equal(merged.n_cells, n_cells)
    assert_layers_equal(merged, expected)
    rows = (cells["sample"] == "S2") & (labels == "T")
    assert_allclose(
        merged.variance("T")[1], np.var(cells["log"][rows], axis=0, ddof=1), rtol=1e-10
    )
    assert_allclose(merged.mean("T")[1], cells["log"][rows].mean(axis=0), rtol=1e-13)


def test_regroup_keeps_group_columns_only_where_members_agree(cube: Cube) -> None:
    groups = cube.regroup({"T": ["T4", "T8"]}).groups
    assert groups.columns == tuple(GROUP_TABLE)
    assert groups["lineage"].tolist() == ["B", "T", "NK"]
    assert groups["parent"].tolist() == ["lymphocyte"] * 3
    assert groups["cell_ontology_id"].tolist() == ["CL:0000236", None, "CL:0000623"]
    assert groups["n_markers"].dtype == np.float64
    assert_array_equal(groups["n_markers"], [3.0, np.nan, 2.0])
    assert groups["is_t_cell"].dtype == np.bool_
    assert groups["is_t_cell"].tolist() == [False, True, False]


def test_regroup_places_a_merge_at_its_earliest_member(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    merged = cube.regroup({"lymphoid": ["NK", "B"]})
    assert merged.groups["group_id"].tolist() == ["lymphoid", "T4", "T8"]
    assert_array_equal(
        merged.layer("counts", "lymphoid"), layers["counts"][[0, 3]].sum(0)
    )
    assert_array_equal(merged.layer("counts", "T8"), layers["counts"][2])
    assert_array_equal(merged.n_cells, np.c_[CELLS[:, 0] + CELLS[:, 3], CELLS[:, 1:3]])


def test_regroup_renames_swaps_and_reuses_a_member_id(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    renamed = cube.regroup({"B cell": "B"})
    assert renamed.groups["group_id"].tolist() == ["B cell", "T4", "T8", "NK"]
    swapped = cube.regroup({"B": ["T4"], "T4": ["B"]})
    assert swapped.groups["group_id"].tolist() == ["T4", "B", "T8", "NK"]
    assert_array_equal(swapped.layer("counts", "B"), layers["counts"][1])
    absorbed = cube.regroup({"T4": ["T4", "T8"]})
    assert absorbed.groups["group_id"].tolist() == ["B", "T4", "NK"]
    assert_array_equal(absorbed.layer("detected", "T4"), layers["detected"][1:3].sum(0))
    assert cube.regroup({}).groups.equals(cube.groups)


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ({"T": ["T4", "X"]}, KeyError, "lists 'X', which is not a group id"),
        ({"T": []}, ValueError, "lists no groups"),
        ({"T": ["T4", "T4"]}, ValueError, "listed twice in the merge into 'T'"),
        ({"T": ["T4"], "U": ["T8", "T4"]}, ValueError, "merges into 'T' and 'U'"),
        ({"B": ["T4", "T8"]}, ValueError, "'B' is already the id of a group that is"),
        ({"": ["B"]}, ValueError, "new group ids must be non-empty"),
        ({1: ["B"]}, TypeError, "new group ids must be strings, got int 1"),
        ({"T": [4]}, TypeError, "must list group ids"),
        ({"T": 4}, TypeError, "must list group ids, got int"),
        (["T4"], TypeError, "needs a mapping"),
    ],
)
def test_regroup_rejects_invalid_mappings(
    cube: Cube, mapping: Any, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        cube.regroup(mapping)


def test_collapse_samples_equals_aggregation_by_donor(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    pooled = cube.collapse_samples("donor")
    donor_of = dict(zip(SAMPLES, SAMPLE_TABLE["donor"], strict=True))
    donors = np.array([donor_of[sample] for sample in cells["sample"]])
    expected, n_cells = aggregate(cells, donors, cells["group"], ["D1", "D2"], GROUPS)
    assert pooled.samples["sample_id"].tolist() == ["D1", "D2"]
    assert_array_equal(pooled.n_cells, n_cells)
    assert_layers_equal(pooled, expected)
    rows = (donors == "D1") & (cells["group"] == "T4")
    assert_allclose(
        pooled.variance("T4")[0], np.var(cells["log"][rows], axis=0, ddof=1), rtol=1e-10
    )
    samples = pooled.samples
    assert samples.columns == (
        "sample_id",
        "donor",
        "condition",
        "age",
        "n_pooled_samples",
    )
    assert samples["condition"].tolist() == ["ctrl", "stim"]
    assert samples["age"].tolist() == [41.0, 29.0]
    assert samples["n_pooled_samples"].tolist() == [3, 1]


def test_collapsing_twice_counts_the_original_samples(cube: Cube) -> None:
    twice = cube.collapse_samples("donor").collapse_samples("condition")
    once = cube.collapse_samples("condition")
    assert twice.samples["sample_id"].tolist() == ["ctrl", "stim"]
    assert twice.samples["n_pooled_samples"].tolist() == [3, 1]
    assert once.samples.equals(twice.samples.select(once.samples.columns))
    for name in INTEGER_LAYERS:
        assert_array_equal(twice.layer(name), once.layer(name))
    for name in FLOAT_LAYERS:
        assert_allclose(twice.layer(name), once.layer(name), rtol=1e-13)


def test_collapse_samples_by_integer_column_and_invalid_columns(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    by_batch = cube.collapse_samples("batch")  # batches 1, 2, 2, 1
    assert by_batch.samples["sample_id"].tolist() == ["1", "2"]
    assert by_batch.samples["batch"].tolist() == [1, 2]
    counts_b = layers["counts"][0]
    expected = np.stack([counts_b[[0, 3]].sum(axis=0), counts_b[[1, 2]].sum(axis=0)])
    assert_array_equal(by_batch.layer("counts", "B"), expected)
    with pytest.raises(KeyError, match="no sample column named 'site'"):
        cube.collapse_samples("site")
    with pytest.raises(TypeError, match="'age' has dtype float64"):
        cube.collapse_samples("age")
    missing = dict(SAMPLE_TABLE, donor=["D1", None, "D1", "D1"])
    with pytest.raises(ValueError, match="missing or empty for sample 'S2'"):
        make_cube(layers, samples=missing).collapse_samples("donor")


def test_statistics_regrouping_and_pooling_match_scanpy() -> None:
    # scanpy.get.aggregate on cells that scanpy normalized itself, written by
    # validation/cube.py --write-fixtures.
    data = np.load(SCANPY_FIXTURE, allow_pickle=False)
    counts = data["counts"].astype(np.int64)
    linear = counts * float(data["target_sum"]) / counts.sum(axis=1, keepdims=True)
    cells = {"counts": counts, "linear": linear, "log": np.log1p(linear)}
    sample_ids, group_ids = data["sample_ids"].tolist(), data["group_ids"].tolist()
    layers, n_cells = aggregate(
        cells, data["sample"], data["group"], sample_ids, group_ids
    )
    cube = Cube(
        layers,
        samples={"sample_id": sample_ids, "donor": data["donor_of_sample"]},
        groups={"group_id": group_ids},
        features={"feature_id": [f"g{j}" for j in range(counts.shape[1])]},
        n_cells=n_cells,
        normalization=NORMALIZATION,
    )
    mapping: dict[str, list[str]] = {}
    for group, coarse in zip(group_ids, data["coarse_of_group"].tolist(), strict=True):
        mapping.setdefault(coarse, []).append(group)
    variants = {
        "fine": cube,
        "coarse": cube.regroup(mapping),
        "donor": cube.collapse_samples("donor"),
    }
    for name, variant in variants.items():
        variant_samples = variant.samples["sample_id"].tolist()
        variant_groups = variant.groups["group_id"].tolist()
        s = np.array([variant_samples.index(key) for key in data[f"{name}_keys0"]])
        g = np.array([variant_groups.index(key) for key in data[f"{name}_keys1"]])
        variant_cells = variant.n_cells
        assert variant_cells is not None
        assert_array_equal(variant_cells[s, g], data[f"{name}_n_cells"])
        assert_array_equal(variant.layer("counts")[g, s], data[f"{name}_sum"])
        assert_array_equal(
            variant.layer("detected")[g, s], data[f"{name}_count_nonzero"]
        )
        # scanpy sums with a sparse indicator product, in another order.
        assert_allclose(variant.mean()[g, s], data[f"{name}_mean_log"], rtol=1e-12)
        assert_allclose(
            variant.mean(scale="linear")[g, s], data[f"{name}_mean_linear"], rtol=1e-12
        )
        several = data[f"{name}_n_cells"] >= 2
        variance = variant.variance()[g, s]
        assert_allclose(variance[several], data[f"{name}_var_log"][several], rtol=1e-10)
        assert np.isnan(variance[~several]).all()
        assert np.isnan(data[f"{name}_var_log"][~several]).all()
        observed = np.zeros((variant.n_groups, variant.n_samples), dtype=bool)
        observed[g, s] = True
        assert (variant_cells.T[~observed] == 0).all()
        assert np.isnan(variant.mean()[~observed]).all()
    assert (~(data["fine_n_cells"] >= 2)).sum() == 1  # the fixture has one single cell


# Selection.


def test_select_by_ids_masks_and_positions(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    chosen = cube.select(
        samples=["S3", "S1"],
        groups=np.array([True, False, True, False]),
        features=[5, 0, -1],
    )
    assert chosen.samples["sample_id"].tolist() == ["S3", "S1"]
    assert chosen.groups["group_id"].tolist() == ["B", "T8"]
    assert chosen.features["feature_id"].tolist() == ["G05", "G00", "G19"]
    for name in INTEGER_LAYERS + FLOAT_LAYERS:
        assert chosen.layer(name).dtype == layers[name].dtype
        assert_array_equal(
            chosen.layer(name), layers[name][np.ix_([0, 2], [2, 0], [5, 0, 19])]
        )
    assert_array_equal(chosen.n_cells, CELLS[np.ix_([2, 0], [0, 2])])
    assert chosen.samples["donor"].tolist() == ["D1", "D1"]
    assert cube.select() is cube
    assert cube.select(groups="NK").n_groups == 1
    empty = cube.select(features=[])
    assert empty.n_features == 0
    assert empty.mean().shape == (4, 4, 0)


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ({"samples": ["S9"]}, KeyError, "no sample with id 'S9'"),
        ({"groups": [True, False]}, ValueError, "mask of groups needs 4 values, got 2"),
        ({"features": [20]}, IndexError, "feature position 20 is out of range"),
        ({"samples": ["S1", "S1"]}, ValueError, "samples repeats 'S1'"),
        ({"groups": [0, -4]}, ValueError, "groups repeats 'B'"),
        ({"features": [0.5]}, TypeError, "got dtype float64"),
        ({"groups": [["B"]]}, ValueError, "1-D selection"),
    ],
)
def test_select_rejects_invalid_selections(
    cube: Cube, selection: dict[str, Any], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        cube.select(**selection)


# Lazy layers.


def test_lazy_layers_are_read_one_group_at_a_time(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    lazy = {name: LazyArray(array) for name, array in layers.items()}
    lazy_cube = make_cube(lazy)
    assert all(not array.reads for array in lazy.values())
    assert "counts (int64, lazy)" in repr(lazy_cube)
    block = lazy_cube.layer("counts", "T8")
    assert lazy["counts"].reads == [2]
    assert_array_equal(block, layers["counts"][2])
    with pytest.raises(ValueError, match="read-only"):
        block[0, 0] = 1
    assert_array_equal(lazy_cube.variance(), cube.variance())
    assert_array_equal(lazy_cube.mean(scale="linear"), cube.mean(scale="linear"))
    assert_array_equal(lazy_cube.fraction_detected("B"), cube.fraction_detected("B"))
    assert_array_equal(lazy_cube.library_sizes(), cube.library_sizes())
    assert lazy["log_sum"].reads == [0, 1, 2, 3]
    assert_array_equal(lazy_cube.layer("log_sum"), layers["log_sum"])
    assert lazy["log_sum"].reads == [0, 1, 2, 3] * 2


def test_operations_on_a_lazy_cube_stay_lazy_and_match_the_eager_cube(
    cube: Cube, layers: dict[str, np.ndarray]
) -> None:
    lazy = {name: LazyArray(array) for name, array in layers.items()}

    def derive(source: Cube) -> Cube:
        chosen = source.select(
            samples=["S4", "S1", "S3", "S2"], features=list(range(12))
        )
        return chosen.regroup({"T": ["T4", "T8"]}).collapse_samples("donor")

    derived = derive(make_cube(lazy))
    assert all(not array.reads for array in lazy.values())
    expected = derive(cube)
    assert derived.samples.equals(expected.samples)
    assert derived.groups.equals(expected.groups)
    assert_array_equal(derived.n_cells, expected.n_cells)
    for name in layers:
        assert_array_equal(derived.layer(name), expected.layer(name))
        assert derived.layer(name).dtype == expected.layer(name).dtype
        assert_array_equal(derived.layer(name, "T"), expected.layer(name, "T"))
    in_memory = derived.to_memory()
    assert "lazy" not in repr(in_memory)
    assert_array_equal(in_memory.layer("log_sum_sq"), expected.layer("log_sum_sq"))
    assert cube.to_memory() is cube


def test_lazy_values_are_checked_when_a_group_is_read(
    layers: dict[str, np.ndarray],
) -> None:
    counts = layers["counts"].copy()
    counts[3, 1, 4] = -2
    lazy_cube = make_cube(dict(layers, counts=LazyArray(counts)))
    assert_array_equal(lazy_cube.layer("counts", "B"), layers["counts"][0])
    message = r"'counts' has a negative value \(-2\) at group 'NK', sample 'S2', feature 'G04'"
    with pytest.raises(ValueError, match=message):
        lazy_cube.layer("counts", "NK")
    with pytest.raises(ValueError, match=message):
        lazy_cube.to_memory()
    with pytest.raises(ValueError, match=message):
        lazy_cube.regroup({"all": GROUPS}).layer("counts", "all")


def test_lazy_layers_with_the_wrong_shape_or_dtype(
    layers: dict[str, np.ndarray],
) -> None:
    with pytest.raises(
        ValueError, match=r"must have shape .* = \(4, 4, 20\), got \(4, 4, 19\)"
    ):
        make_cube(dict(layers, counts=LazyArray(layers["counts"][:, :, :19])))
    with pytest.raises(
        TypeError, match="'log_sum' must have float64 dtype, got float32"
    ):
        make_cube(dict(layers, log_sum=LazyArray(layers["log_sum"].astype(np.float32))))

    class WrongShape(LazyArray):
        def __getitem__(self, key: int) -> np.ndarray:
            return super().__getitem__(key)[:, :3]

    class WrongDtype(LazyArray):
        def __getitem__(self, key: int) -> np.ndarray:
            return super().__getitem__(key).astype(np.float32)

    lazy_cube = make_cube(dict(layers, log_sum=WrongDtype(layers["log_sum"])))
    with pytest.raises(
        TypeError, match="group 'T4' of the lazy layer 'log_sum' gave dtype float32"
    ):
        lazy_cube.mean("T4")

    lazy_cube = make_cube(dict(layers, counts=WrongShape(layers["counts"])))
    with pytest.raises(
        ValueError, match=r"group 'B' of the lazy layer 'counts' gave shape \(4, 3\)"
    ):
        lazy_cube.layer("counts", 0)


# Dtypes, memory and alignment.


@pytest.mark.parametrize("dtype", [np.int32, np.uint16, np.uint64])
def test_small_and_unsigned_integer_layers_are_summed_in_int64(
    layers: dict[str, np.ndarray], dtype: type
) -> None:
    # 60,000 counts in every non-empty entry: sums over groups, samples or features
    # overflow uint16 (65,535) unless they are accumulated in int64.
    filled = (CELLS.T > 0)[:, :, None]
    counts = np.where(filled, 60_000, 0) * np.ones(N_FEATURES, dtype=np.int64)
    detected = np.minimum(layers["detected"], CELLS.T[:, :, None])
    cube = make_cube(
        dict(layers, counts=counts.astype(dtype), detected=detected.astype(dtype))
    )
    assert cube.layer("counts").dtype == dtype
    merged = cube.regroup({"all": GROUPS})
    pooled = cube.collapse_samples("donor")
    for derived in (merged, pooled):
        assert derived.layer("counts").dtype == np.int64
        assert derived.layer("detected").dtype == np.int64
    groups_with_cells = (CELLS > 0).sum(axis=1)  # 4, 3, 4, 4 groups per sample
    assert_array_equal(merged.layer("counts", "all")[:, 0], 60_000 * groups_with_cells)
    assert pooled.layer("counts", "T4")[:, 0].tolist() == [3 * 60_000, 0]
    library_sizes = cube.library_sizes()
    assert library_sizes.dtype == np.int64
    assert_array_equal(library_sizes, np.where(CELLS > 0, 60_000 * N_FEATURES, 0))


def test_big_endian_layers_are_stored_in_native_byte_order(
    layers: dict[str, np.ndarray],
) -> None:
    swapped = {
        name: array.astype(array.dtype.newbyteorder(">"))
        for name, array in layers.items()
    }
    eager = make_cube(swapped)
    lazy = make_cube({name: LazyArray(array) for name, array in swapped.items()})
    for cube in (eager, lazy):
        for name, array in layers.items():
            assert cube.layer(name).dtype == array.dtype
            assert cube.layer(name).dtype.isnative
            assert_array_equal(cube.layer(name), array)


def test_memmap_layers_are_lazy_and_reads_do_not_follow_later_changes(
    tmp_path: Path, layers: dict[str, np.ndarray]
) -> None:
    mapped = np.memmap(
        tmp_path / "counts.dat", dtype=np.int64, mode="w+", shape=layers["counts"].shape
    )
    mapped[:] = layers["counts"]
    cube = make_cube(dict(layers, counts=mapped))
    assert "counts (int64, lazy)" in repr(cube)
    block = cube.layer("counts", "B")
    in_memory = cube.to_memory()
    original = layers["counts"][0, 0, 0]
    mapped[0, 0, 0] = original + 1_000
    assert block[0, 0] == original
    assert in_memory.layer("counts")[0, 0, 0] == original
    assert in_memory.library_sizes()[0, 0] == layers["counts"][0, 0].sum()
    assert cube.layer("counts", "B")[0, 0] == original + 1_000  # a new read
    mapped[3, 1, 4] = -2
    with pytest.raises(ValueError, match="'counts' has a negative value"):
        cube.layer("counts", "NK")
    del mapped


def test_tables_without_rows_build_empty_axes() -> None:
    no_features = Cube(
        {"counts": np.zeros((4, 4, 0), dtype=np.int64)},
        samples=SAMPLE_TABLE,
        groups=GROUP_TABLE,
        features={"feature_id": [], "symbol": [], "measured": []},
        n_cells=CELLS,
    )
    assert no_features.n_features == 0
    assert no_features.features["feature_id"].dtype == object
    assert no_features.features["measured"].dtype == np.bool_
    assert (no_features.library_sizes() == 0).all()
    no_samples = Cube(
        {"counts": np.zeros((1, 0, 2), dtype=np.int64)},
        samples={"sample_id": [], "donor": []},
        groups={"group_id": ["g"]},
        features={"feature_id": ["a", "b"]},
    )
    assert no_samples.library_sizes().shape == (0, 1)
    pooled = no_samples.collapse_samples("donor")
    assert pooled.n_samples == 0
    assert pooled.samples.columns == ("sample_id", "donor", "n_pooled_samples")
    assert no_samples.regroup({"h": ["g"]}).groups["group_id"].tolist() == ["h"]


def test_moments_that_cannot_come_from_the_same_cells_are_rejected(
    layers: dict[str, np.ndarray],
) -> None:
    swapped = dict(layers, log_sum=layers["log_sum_sq"], log_sum_sq=layers["log_sum"])
    message = "'log_sum' and 'log_sum_sq' disagree at group 'B', sample 'S1'"
    with pytest.raises(ValueError, match=message):
        make_cube(swapped)
    lazy_cube = make_cube(
        dict(
            layers,
            log_sum=LazyArray(swapped["log_sum"]),
            log_sum_sq=LazyArray(swapped["log_sum_sq"]),
        )
    )
    with pytest.raises(ValueError, match=message):
        lazy_cube.variance()
    with pytest.raises(ValueError, match=message):
        lazy_cube.to_memory()


def test_moment_check_allows_rounding_but_not_more() -> None:
    # Four cells; log_sum**2 / 4 exceeds log_sum_sq = 1 by a relative 5e-7 (inside
    # the 1e-6 allowed for rounding, so the variance is 0) or by 2e-6 (rejected).
    def build(excess: float) -> Cube:
        return Cube(
            {
                "counts": np.array([[[8]]]),
                "log_sum": np.array([[[np.sqrt(4 * (1 + excess))]]]),
                "log_sum_sq": np.array([[[1.0]]]),
            },
            samples={"sample_id": ["A"]},
            groups={"group_id": ["g"]},
            features={"feature_id": ["f"]},
            n_cells=np.array([[4]]),
            normalization=NORMALIZATION,
        )

    assert build(5e-7).variance("g")[0, 0] == 0.0
    with pytest.raises(ValueError, match=r"log_sum\*\*2 / n_cells = 1 exceeds"):
        build(2e-6)


class LabelledMatrix:
    """The parts of a pandas DataFrame that label alignment uses."""

    def __init__(
        self, values: np.ndarray, index: list[str], columns: list[str]
    ) -> None:
        self._values = values
        self.index = index
        self.columns = columns

    def to_numpy(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)


def test_labelled_n_cells_are_aligned_by_ids(layers: dict[str, np.ndarray]) -> None:
    rows, columns = [2, 0, 3, 1], [3, 0, 2, 1]
    shuffled = LabelledMatrix(
        CELLS[np.ix_(rows, columns)],
        [SAMPLES[i] for i in rows],
        [GROUPS[j] for j in columns],
    )
    assert_array_equal(make_cube(layers, n_cells=shuffled).n_cells, CELLS)
    renamed = LabelledMatrix(CELLS, ["S9", *SAMPLES[1:]], GROUPS)
    message = r"rows of an n_cells DataFrame .* missing: \['S1'\], unexpected: \['S9'\]"
    with pytest.raises(ValueError, match=message):
        make_cube(layers, n_cells=renamed)
    transposed = LabelledMatrix(CELLS.T, GROUPS, SAMPLES)
    with pytest.raises(ValueError, match="rows of an n_cells DataFrame"):
        make_cube(layers, n_cells=transposed)
    repeated = LabelledMatrix(np.c_[CELLS, CELLS[:, :1]], SAMPLES, [*GROUPS, "B"])
    with pytest.raises(ValueError, match="columns of an n_cells DataFrame"):
        make_cube(layers, n_cells=repeated)
    # Object values, as pandas' nullable integer columns give, are converted when
    # they are whole numbers and rejected otherwise.
    objects = LabelledMatrix(CELLS.astype(object), SAMPLES, GROUPS)
    assert_array_equal(make_cube(layers, n_cells=objects).n_cells, CELLS)
    halves = LabelledMatrix((CELLS + 0.5).astype(object), SAMPLES, GROUPS)
    with pytest.raises(TypeError, match="n_cells must be an integer array"):
        make_cube(layers, n_cells=halves)


def test_pandas_n_cells_with_nullable_integers(layers: dict[str, np.ndarray]) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(CELLS, index=SAMPLES, columns=GROUPS).astype("Int64")
    shuffled = frame.loc[["S3", "S1", "S4", "S2"], ["NK", "B", "T8", "T4"]]
    assert_array_equal(make_cube(layers, n_cells=shuffled).n_cells, CELLS)
    frame.loc["S2", "T4"] = pd.NA
    with pytest.raises(ValueError, match="n_cells DataFrame has missing values"):
        make_cube(layers, n_cells=frame)
    fractions = pd.DataFrame(CELLS + 0.5, index=SAMPLES, columns=GROUPS).astype(object)
    with pytest.raises(TypeError, match="n_cells must be an integer array"):
        make_cube(layers, n_cells=fractions)


def test_copy_false_adopts_arrays_after_the_checks_pass(
    layers: dict[str, np.ndarray],
) -> None:
    owned = {name: array.copy() for name, array in layers.items()}
    cube = make_cube(owned, copy=False)
    assert np.shares_memory(cube.layer("counts"), owned["counts"])
    assert not owned["counts"].flags.writeable
    fortran = np.asfortranarray(layers["linear_sum"])
    copied = make_cube(dict(layers, linear_sum=fortran), copy=False)
    assert not np.shares_memory(copied.layer("linear_sum"), fortran)
    assert fortran.flags.writeable
    rejected = {name: array.copy() for name, array in layers.items()}
    rejected["detected"][3, 1, 2] = 5  # one NK cell in S2
    with pytest.raises(ValueError, match="exceeds n_cells"):
        make_cube(rejected, copy=False)
    assert rejected["counts"].flags.writeable
    untouched = {name: array.copy() for name, array in layers.items()}
    with pytest.raises(ValueError, match="JSON has no NaN"):
        make_cube(untouched, copy=False, provenance={"x": float("nan")})
    assert all(array.flags.writeable for array in untouched.values())


def test_n_pooled_samples_keeps_its_place_or_comes_last(
    layers: dict[str, np.ndarray],
) -> None:
    samples = {
        "sample_id": SAMPLES,
        "donor": SAMPLE_TABLE["donor"],
        "n_pooled_samples": [1, 2, 3, 1],  # varies within donor D1
        "batch": SAMPLE_TABLE["batch"],  # varies within donor D1, so it is dropped
        "condition": SAMPLE_TABLE["condition"],  # constant within each donor
    }
    pooled = make_cube(layers, samples=samples).collapse_samples("donor")
    expected = ("sample_id", "donor", "n_pooled_samples", "condition")
    assert pooled.samples.columns == expected
    assert pooled.samples["n_pooled_samples"].tolist() == [5, 2]
    fresh = make_cube(layers).collapse_samples("donor")
    assert fresh.samples.columns[-1] == "n_pooled_samples"
    floats = dict(samples, n_pooled_samples=[1.0, 2.0, 3.0, 1.0])
    with pytest.raises(ValueError, match="'n_pooled_samples' counts pooled samples"):
        make_cube(layers, samples=floats).collapse_samples("donor")


# Validation of the constructor.


def with_value(array: np.ndarray, index: tuple[int, ...], value: Any) -> np.ndarray:
    changed = array.copy()
    changed[index] = value
    return changed


def replace_layer(name: str, value: Callable[[dict[str, np.ndarray]], Any]) -> Any:
    return lambda arguments: arguments["layers"].__setitem__(
        name, value(arguments["layers"])
    )


INVALID_CUBES = [
    (
        "unknown layer",
        lambda a: a["layers"].update(logsum=a["layers"]["log_sum"]),
        ValueError,
        "unknown layer 'logsum'",
    ),
    (
        "no counts",
        lambda a: a["layers"].pop("counts"),
        ValueError,
        "'counts' layer is required",
    ),
    (
        "float32 counts",
        replace_layer("counts", lambda layers: layers["counts"].astype(np.float32)),
        TypeError,
        "'counts' must have an integer or float64 dtype, got float32",
    ),
    (
        "float32 moments",
        replace_layer(
            "log_sum_sq", lambda layers: layers["log_sum_sq"].astype(np.float32)
        ),
        TypeError,
        "lower precision loses digits",
    ),
    (
        "float detected",
        replace_layer("detected", lambda layers: layers["detected"].astype(np.float64)),
        TypeError,
        "'detected' must have an integer dtype",
    ),
    (
        "wrong shape",
        replace_layer("linear_sum", lambda layers: layers["linear_sum"][:, :, :19]),
        ValueError,
        r"'linear_sum' must have shape \(n_groups, n_samples, n_features\)",
    ),
    (
        "negative counts",
        replace_layer(
            "counts", lambda layers: with_value(layers["counts"], (2, 3, 7), -1)
        ),
        ValueError,
        r"'counts' has a negative value \(-1\) at group 'T8', sample 'S4', feature 'G07'",
    ),
    (
        "nan moment",
        replace_layer(
            "log_sum", lambda layers: with_value(layers["log_sum"], (0, 0, 1), np.nan)
        ),
        ValueError,
        r"'log_sum' has a non-finite value \(nan\) at group 'B', sample 'S1', feature 'G01'",
    ),
    (
        "detected above n_cells",
        replace_layer(
            "detected", lambda layers: with_value(layers["detected"], (3, 1, 2), 2)
        ),
        ValueError,
        "'detected' exceeds n_cells at group 'NK', sample 'S2', feature 'G02': 2 cells "
        "detected but 1 pooled",
    ),
    (
        "value in an empty entry",
        replace_layer(
            "linear_sum",
            lambda layers: with_value(layers["linear_sum"], (1, 1, 3), 0.5),
        ),
        ValueError,
        r"'linear_sum' is 0.5 at group 'T4', sample 'S2', feature 'G03', where n_cells is 0",
    ),
    (
        "no n_cells",
        lambda a: a.update(n_cells=None),
        ValueError,
        "n_cells is required with",
    ),
    (
        "no normalization",
        lambda a: a.update(normalization=None),
        ValueError,
        "normalization is required with the 'log_sum', 'log_sum_sq', 'linear_sum' layer",
    ),
    (
        "normalization without a method",
        lambda a: a.update(normalization={"target_sum": 1e4}),
        ValueError,
        "needs a 'method' string",
    ),
    (
        "n_cells shape",
        lambda a: a.update(n_cells=CELLS[:, :3]),
        ValueError,
        r"n_cells must have shape \(n_samples, n_groups\) = \(4, 4\), got \(4, 3\)",
    ),
    (
        "float n_cells",
        lambda a: a.update(n_cells=CELLS.astype(float)),
        TypeError,
        "n_cells must be an integer array",
    ),
    (
        "negative n_cells",
        lambda a: a.update(n_cells=with_value(CELLS, (0, 1), -3)),
        ValueError,
        r"n_cells is negative \(-3\) for sample 'S1' and group 'T4'",
    ),
    (
        "no sample ids",
        lambda a: a.update(samples={"donor": ["D1", "D2", "D1", "D1"]}),
        ValueError,
        "the samples table needs a 'sample_id' column; it has: donor",
    ),
    (
        "repeated ids",
        lambda a: a.update(groups={"group_id": ["B", "T4", "T4", "NK"]}),
        ValueError,
        r"groups\['group_id'\] repeats the id 'T4'",
    ),
    (
        "integer ids",
        lambda a: a.update(samples={"sample_id": [1, 2, 3, 4]}),
        TypeError,
        r"samples\['sample_id'\] must hold text ids, got dtype int64",
    ),
    (
        "empty id",
        lambda a: a.update(
            features={"feature_id": [""] + FEATURE_TABLE["feature_id"][1:]}
        ),
        ValueError,
        "missing or empty id at row 0",
    ),
    (
        "fewer groups than layers",
        lambda a: a.update(groups={"group_id": GROUPS[:3]}, n_cells=CELLS[:, :3]),
        ValueError,
        r"= \(3, 4, 20\), got \(4, 4, 20\)",
    ),
    (
        "measured as integers",
        lambda a: a.update(features=dict(FEATURE_TABLE, measured=[1] * N_FEATURES)),
        TypeError,
        r"features\['measured'\] must be boolean",
    ),
    (
        "parent as integers",
        lambda a: a.update(groups=dict(GROUP_TABLE, parent=[0, 0, 0, 0])),
        TypeError,
        r"groups\['parent'\] must hold text",
    ),
    (
        "ragged sample table",
        lambda a: a.update(samples={"sample_id": SAMPLES, "x": [1]}),
        ValueError,
        "equal lengths",
    ),
    (
        "provenance object",
        lambda a: a.update(provenance={"when": object()}),
        TypeError,
        r"provenance\['when'\] has a value of type 'object', which is not JSON",
    ),
    (
        "provenance nan",
        lambda a: a.update(provenance={"n": [1.0, np.nan]}),
        ValueError,
        r"provenance\['n'\]\[1\] is nan; JSON has no NaN",
    ),
    (
        "provenance key",
        lambda a: a.update(provenance={1: "a"}),
        TypeError,
        "JSON keys are strings",
    ),
    (
        "provenance operations",
        lambda a: a.update(provenance={"operations": "none"}),
        ValueError,
        "must be a list of operation records",
    ),
    (
        "layers list",
        lambda a: a.update(layers=[1, 2]),
        TypeError,
        "layers must be a mapping",
    ),
    (
        "masked layer",
        replace_layer("counts", lambda layers: np.ma.masked_array(layers["counts"])),
        TypeError,
        "masked array",
    ),
]


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [case[1:] for case in INVALID_CUBES],
    ids=[case[0] for case in INVALID_CUBES],
)
def test_invalid_cubes_are_rejected(
    layers: dict[str, np.ndarray],
    change: Callable[[dict[str, Any]], None],
    error: type,
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "layers": dict(layers),
        "samples": SAMPLE_TABLE,
        "groups": GROUP_TABLE,
        "features": FEATURE_TABLE,
        "n_cells": CELLS,
        "normalization": NORMALIZATION,
    }
    change(arguments)
    with pytest.raises(error, match=message):
        Cube(arguments.pop("layers"), **arguments)


# Access, metadata, immutability and bulk data.


def test_group_and_layer_lookup(cube: Cube, layers: dict[str, np.ndarray]) -> None:
    assert cube.layer_names == (
        "counts",
        "detected",
        "log_sum",
        "log_sum_sq",
        "linear_sum",
    )
    assert_array_equal(cube.layer("counts", 2), layers["counts"][2])
    assert_array_equal(cube.layer("counts", -1), cube.layer("counts", "NK"))
    assert_array_equal(cube.layer("log_sum"), layers["log_sum"])
    with pytest.raises(
        KeyError, match="no group with id 'X'; groups are: B, T4, T8, NK"
    ):
        cube.layer("counts", "X")
    with pytest.raises(
        IndexError, match="group position 4 is out of range for 4 groups"
    ):
        cube.mean(4)
    with pytest.raises(TypeError, match="not bool"):
        cube.mean(True)
    with pytest.raises(TypeError, match="got float"):
        cube.mean(2.0)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="unknown layer 'reads'"):
        cube.layer("reads")
    with pytest.raises(ValueError, match="scale must be 'log' or 'linear', got 'sqrt'"):
        cube.mean(scale="sqrt")


def test_bulk_counts_without_cells(layers: dict[str, np.ndarray]) -> None:
    estimates = layers["linear_sum"][:1] / 7.0  # non-integer counts, one group
    bulk = Cube(
        {"counts": estimates},
        samples=SAMPLE_TABLE,
        groups={"group_id": ["bulk"]},
        features=FEATURE_TABLE,
    )
    assert bulk.n_cells is None
    assert bulk.normalization is None
    assert "cells per entry: not recorded" in repr(bulk)
    assert bulk.library_sizes().dtype == np.float64
    assert_allclose(bulk.library_sizes()[:, 0], estimates[0].sum(axis=1), rtol=1e-15)
    with pytest.raises(
        KeyError, match="this cube has no 'detected' layer; it has: counts"
    ):
        bulk.layer("detected")
    with pytest.raises(
        ValueError, match=r"mean\(scale='log'\) needs the 'log_sum' layer"
    ):
        bulk.mean()
    with pytest.raises(ValueError, match="needs the 'log_sum' and 'log_sum_sq' layers"):
        bulk.variance()
    pooled = bulk.collapse_samples("donor")
    assert pooled.n_cells is None
    assert_allclose(pooled.layer("counts", 0)[1], estimates[0, 1], rtol=0)


def test_cube_is_immutable_and_owns_its_arrays(layers: dict[str, np.ndarray]) -> None:
    source = {name: array.copy() for name, array in layers.items()}
    normalization = dict(NORMALIZATION)
    cube = make_cube(source, normalization=normalization)
    source["counts"][0, 0, 0] += 100
    normalization["method"] = "changed"
    assert cube.layer("counts")[0, 0, 0] == layers["counts"][0, 0, 0]
    assert cube.normalization == NORMALIZATION
    returned = cube.normalization
    assert returned is not None
    returned["method"] = "changed"
    assert cube.normalization == NORMALIZATION
    with pytest.raises(ValueError, match="read-only"):
        cube.layer("counts")[0, 0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        cube.layer("detected", "B")[0, 0] = 1
    n_cells = cube.n_cells
    assert n_cells is not None
    with pytest.raises(ValueError, match="read-only"):
        n_cells[0, 0] = 1
    with pytest.raises(AttributeError, match="immutable"):
        cube.normalization = {}  # type: ignore[misc]
    assert copy.copy(cube) is cube and copy.deepcopy(cube) is cube


def test_operations_append_to_the_provenance(cube: Cube) -> None:
    derived = (
        cube.select(samples=["S1", "S2"], features=np.arange(5))
        .regroup({"T": ["T8", "T4"]})
        .collapse_samples("condition")
    )
    provenance = derived.provenance
    assert provenance == {
        "source": "simulated cells",
        "operations": [
            {
                "operation": "select",
                "samples": ["S1", "S2"],
                "groups": None,
                "n_features": 5,
            },
            {"operation": "regroup", "mapping": {"T": ["T8", "T4"]}},
            {"operation": "collapse_samples", "by": "condition"},
        ],
    }
    assert json.loads(json.dumps(provenance)) == provenance
    assert cube.provenance == {"source": "simulated cells"}
    provenance["source"] = "changed"
    assert derived.provenance["source"] == "simulated cells"
    assert derived.normalization == NORMALIZATION


def test_numpy_values_in_provenance_become_plain_json(
    layers: dict[str, np.ndarray],
) -> None:
    provenance = {
        "n": np.int64(3),
        "scale": np.float32(0.5),
        "kept": np.array(["a", "b"]),
        "flag": np.bool_(True),
        "steps": ({"k": 1},),
    }
    cube = make_cube(layers, provenance=provenance)
    assert cube.provenance == {
        "n": 3,
        "scale": 0.5,
        "kept": ["a", "b"],
        "flag": True,
        "steps": [{"k": 1}],
    }
    assert type(cube.provenance["n"]) is int


def test_repr_reports_sizes_layers_and_filled_entries(cube: Cube) -> None:
    lines = repr(cube).splitlines()
    assert lines[0] == "Cube with 4 samples x 4 groups x 20 features"
    assert lines[1] == (
        "  layers: counts (int64), detected (int64), log_sum (float64), "
        "log_sum_sq (float64), linear_sum (float64)"
    )
    assert lines[2] == "  filled: 15 of 16 sample x group entries have cells (93.8%)"
    assert lines[3] == "  samples: sample_id, donor, condition, age, tissue, batch"


def test_pickle_round_trip(cube: Cube) -> None:
    restored = pickle.loads(pickle.dumps(cube))
    for name in cube.layer_names:
        assert_array_equal(restored.layer(name), cube.layer(name))
    with pytest.raises(ValueError, match="read-only"):
        restored.layer("counts")[0, 0, 0] = 1
    assert restored.samples.equals(cube.samples)
    assert restored.provenance == cube.provenance
    assert_array_equal(restored.mean(), cube.mean())
