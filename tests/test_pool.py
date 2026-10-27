import re
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import Cube, Table, from_bulk, from_cells
from cytepool.cube import pool as pool_module
from cytepool.cube.cube import LAYER_NAMES

FIXTURE = Path(__file__).parent / "data" / "pool" / "scanpy_decoupler.npz"
TARGET_SUM = 1e4
SAMPLE_IDS = ["S1", "S2", "S3", "S4"]
GROUP_IDS = ["B", "Mono", "NK", "T"]
# Cells per sample (rows) and group (columns). S2 has no NK cells, so that entry is
# empty, and a single Mono cell.
CELLS = np.array([[9, 6, 4, 12], [7, 1, 0, 10], [5, 8, 6, 9], [11, 4, 3, 8]])
N_FEATURES = 24
FLOAT_LAYERS = ("log_sum", "log_sum_sq", "linear_sum")


class CSR:
    """The attributes of a CSR matrix that from_cells reads, without scipy."""

    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        layout: str | None = "csr",
    ) -> None:
        self.data, self.indices, self.indptr, self.shape = data, indices, indptr, shape
        if layout is not None:
            self.format = layout


def to_csr(
    dense: np.ndarray, data_dtype: Any = np.float32, index_dtype: Any = np.int32
) -> CSR:
    rows, columns = np.nonzero(dense)
    lengths = np.bincount(rows, minlength=dense.shape[0])
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    return CSR(
        dense[rows, columns].astype(data_dtype),
        columns.astype(index_dtype),
        indptr.astype(index_dtype),
        dense.shape,
    )


def scrambled_csr(dense: np.ndarray, seed: int = 3) -> CSR:
    """A CSR matrix that means ``dense`` but is far from canonical.

    Each row lists its entries in random order, splits some counts into two
    entries of the same column, and stores two explicit zeros.
    """
    rng = np.random.default_rng(seed)
    data: list[float] = []
    indices: list[int] = []
    lengths = []
    for row in dense:
        columns, values = [], []
        for column in np.flatnonzero(row).tolist():
            value = float(row[column])
            if value > 1 and rng.random() < 0.4:
                part = float(np.floor(value / 2))
                columns += [column, column]
                values += [part, value - part]
            else:
                columns.append(column)
                values.append(value)
        for column in rng.choice(dense.shape[1], size=2, replace=False).tolist():
            columns.append(column)
            values.append(0.0)
        order = rng.permutation(len(columns))
        data += [values[i] for i in order]
        indices += [columns[i] for i in order]
        lengths.append(len(columns))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    return CSR(np.array(data), np.array(indices, dtype=np.int64), indptr, dense.shape)


def simulate_cells(seed: int = 20261023) -> dict[str, np.ndarray]:
    """Negative binomial counts for about a hundred cells, in random order."""
    rng = np.random.default_rng(seed)
    samples, groups = [], []
    for s, sample in enumerate(SAMPLE_IDS):
        for g, group in enumerate(GROUP_IDS):
            samples += [sample] * int(CELLS[s, g])
            groups += [group] * int(CELLS[s, g])
    order = rng.permutation(len(samples))
    group_of_cell = np.array(groups)[order]
    codes = np.array([GROUP_IDS.index(group) for group in group_of_cell])
    mean = (
        rng.lognormal(0.0, 1.0, size=N_FEATURES)
        * rng.lognormal(0.0, 0.7, size=(len(GROUP_IDS), N_FEATURES))[codes]
        * rng.lognormal(0.0, 0.3, size=(order.size, 1))
    )
    counts = rng.negative_binomial(2, 2 / (2 + mean))
    counts[:, 0] = 0  # a feature that is never detected
    counts[counts.sum(axis=1) == 0, 1] = 1  # every cell has a count
    return {
        "counts": counts,
        "samples": np.array(samples)[order],
        "groups": group_of_cell,
    }


def pooled_with_masks(
    counts: np.ndarray,
    samples: np.ndarray,
    groups: np.ndarray,
    sample_ids: list[str],
    group_ids: list[str],
    library_sizes: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Pool the dense counts one sample x group entry at a time, with masks."""
    dense = np.asarray(counts, dtype=np.float64)
    sizes = dense.sum(axis=1) if library_sizes is None else library_sizes
    with np.errstate(divide="ignore", invalid="ignore"):  # NaN for cells without counts
        linear = dense * (TARGET_SUM / sizes)[:, None]
    log = np.log1p(linear)
    shape = (len(group_ids), len(sample_ids), dense.shape[1])
    expected = {name: np.zeros(shape) for name in LAYER_NAMES}
    n_cells = np.zeros((len(sample_ids), len(group_ids)), dtype=np.int64)
    for g, group in enumerate(group_ids):
        for s, sample in enumerate(sample_ids):
            rows = (samples == sample) & (groups == group)
            n_cells[s, g] = rows.sum()
            expected["counts"][g, s] = dense[rows].sum(axis=0)
            expected["detected"][g, s] = (dense[rows] > 0).sum(axis=0)
            expected["log_sum"][g, s] = log[rows].sum(axis=0)
            expected["log_sum_sq"][g, s] = (log[rows] ** 2).sum(axis=0)
            expected["linear_sum"][g, s] = linear[rows].sum(axis=0)
    return expected, n_cells


def assert_pooled(
    cube: Cube, expected: dict[str, np.ndarray], n_cells: Any, whole: bool = True
) -> None:
    """Compare every layer; ``whole`` says whether counts are whole numbers."""
    assert_array_equal(cube.n_cells, n_cells)
    for name in cube.layer_names:
        if name in FLOAT_LAYERS or (name == "counts" and not whole):
            # The cube adds cells in their order and uses its own log1p; the masks
            # sum in numpy's order with numpy's log1p. Both are within a few units
            # of rounding of the exact sums.
            assert_allclose(cube.layer(name), expected[name], rtol=1e-12, atol=0)
        else:
            assert_array_equal(cube.layer(name), expected[name])


def assert_same_cube(actual: Cube, expected: Cube) -> None:
    assert actual.layer_names == expected.layer_names
    for name in expected.layer_names:
        assert actual.layer(name).dtype == expected.layer(name).dtype
        assert_array_equal(actual.layer(name), expected.layer(name))
    assert_array_equal(actual.n_cells, expected.n_cells)
    assert actual.samples.equals(expected.samples)
    assert actual.groups.equals(expected.groups)
    assert actual.features.equals(expected.features)


def pool(counts: Any, cells: dict[str, np.ndarray], **options: Any) -> Cube:
    return from_cells(
        counts, sample_labels=cells["samples"], group_labels=cells["groups"], **options
    )


@pytest.fixture(scope="module")
def cells() -> dict[str, np.ndarray]:
    return simulate_cells()


@pytest.fixture(scope="module")
def cube(cells: dict[str, np.ndarray]) -> Cube:
    return pool(to_csr(cells["counts"]), cells)


# Layers against pooling with masks.


def test_every_layer_equals_pooling_with_masks(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    expected, n_cells = pooled_with_masks(
        cells["counts"], cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS
    )
    assert_array_equal(n_cells, CELLS)
    assert_pooled(cube, expected, CELLS)
    assert cube.layer_names == LAYER_NAMES
    assert cube.layer("counts").dtype == np.uint32
    assert cube.layer("detected").dtype == np.uint32
    assert all(cube.layer(name).dtype == np.float64 for name in FLOAT_LAYERS)
    assert cube.samples["sample_id"].tolist() == SAMPLE_IDS
    assert cube.groups["group_id"].tolist() == GROUP_IDS
    assert cube.features["feature_id"].tolist() == [
        f"feature_{j}" for j in range(N_FEATURES)
    ]
    assert cube.normalization == {
        "method": "library_size",
        "target_sum": TARGET_SUM,
        "log": "natural",
        "pseudocount": 1.0,
        "library_sizes": "cell_totals",
    }
    assert cube.provenance == {
        "source": None,
        "construction": {
            "function": "from_cells",
            "cytepool_version": cytepool.__version__,
            "n_cells": int(CELLS.sum()),
            "layers": list(LAYER_NAMES),
            "target_sum": TARGET_SUM,
            "library_sizes": "cell_totals",
            "samples": {"order": "sorted_labels", "labels": "text"},
            "groups": {"order": "sorted_labels", "labels": "text"},
            "features": "positions",
        },
    }


def test_entries_without_cells_are_zero(cube: Cube) -> None:
    n_cells = cube.n_cells
    assert n_cells is not None and n_cells[1, 2] == 0  # S2 has no NK cells
    for name in LAYER_NAMES:
        assert not cube.layer(name, "NK")[1].any()
    assert np.isnan(cube.mean("NK")[1]).all()
    assert np.isnan(cube.variance("Mono")[1]).all()  # a single cell
    assert not cube.layer("detected")[:, :, 0].any()  # a feature never detected


def test_single_cell_entries_hold_that_cells_values_to_a_few_ulps() -> None:
    # One cell per entry, with library size 1 and fractional counts from 1e-300 to
    # 1e12: log_sum must be log1p(count) within a few units in the last place (1e-15
    # is about 4.5 of them), which log(1 + count) misses by up to 1e-4 at 1e-12.
    # Below 1.1e-16, 1 + x rounds to 1 and log1p(x) is x itself.
    values = np.concatenate([[1e-300, 1e-20], 10.0 ** np.linspace(-12.0, 12.0, 25)])
    groups = [f"g{i:02d}" for i in range(values.size)]
    cube = from_cells(
        values[:, None],
        sample_labels=["s"] * values.size,
        group_labels=groups,
        target_sum=1.0,
        library_sizes=np.ones(values.size),
    )
    assert cube.layer("counts").dtype == np.float64
    assert_array_equal(cube.layer("counts")[:, 0, 0], values)
    assert_array_equal(cube.layer("linear_sum")[:, 0, 0], values)
    assert_allclose(cube.layer("log_sum")[:, 0, 0], np.log1p(values), rtol=1e-15)
    assert_array_equal(
        cube.layer("log_sum_sq")[:, 0, 0], cube.layer("log_sum")[:, 0, 0] ** 2
    )


# Input forms.


@pytest.mark.parametrize(
    "form",
    [
        "dense int64",
        "dense float32",
        "dense uint16",
        "dense Fortran order",
        "dense big-endian",
        "CSR float64, int64 indices",
        "CSR int32",
        "CSR uint64, int64 indptr",
        "CSR unsorted, repeated and zero entries",
    ],
)
def test_dense_and_csr_inputs_give_identical_cubes(
    cells: dict[str, np.ndarray], cube: Cube, form: str
) -> None:
    dense = cells["counts"]
    inputs = {
        "dense int64": lambda: dense.astype(np.int64),
        "dense float32": lambda: dense.astype(np.float32),
        "dense uint16": lambda: dense.astype(np.uint16),
        "dense Fortran order": lambda: np.asfortranarray(dense),
        "dense big-endian": lambda: dense.astype(">f8"),
        "CSR float64, int64 indices": lambda: to_csr(dense, np.float64, np.int64),
        "CSR int32": lambda: to_csr(dense, np.int32),
        "CSR uint64, int64 indptr": lambda: to_csr(dense, np.uint64, np.int64),
        "CSR unsorted, repeated and zero entries": lambda: scrambled_csr(dense),
    }
    assert_same_cube(pool(inputs[form](), cells), cube)


def test_scrambled_csr_really_is_unsorted_repeated_and_has_zeros(
    cells: dict[str, np.ndarray],
) -> None:
    matrix = scrambled_csr(cells["counts"])
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    pairs = rows * N_FEATURES + matrix.indices
    assert np.unique(pairs).size < pairs.size  # repeated entries
    assert (matrix.data == 0).sum() == 2 * matrix.shape[0]
    steps = np.diff(matrix.indices)[np.diff(rows) == 0]
    assert (steps < 0).any()
    rebuilt = np.zeros(matrix.shape)
    np.add.at(rebuilt, (rows, matrix.indices), matrix.data)
    assert_array_equal(rebuilt, cells["counts"])


def test_scipy_sparse_matrices(cells: dict[str, np.ndarray], cube: Cube) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    for matrix in (
        sparse.csr_matrix(cells["counts"]),
        sparse.csr_array(cells["counts"]),
    ):
        assert_same_cube(pool(matrix, cells), cube)
    with pytest.raises(TypeError, match=r"'csc' format.*counts.tocsr\(\)"):
        pool(sparse.csc_matrix(cells["counts"]), cells)


def test_results_do_not_depend_on_the_number_of_threads(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    before = numba.get_num_threads()
    assert_same_cube(pool(to_csr(cells["counts"]), cells, n_threads=1), cube)
    assert_same_cube(pool(to_csr(cells["counts"]), cells, n_threads=64), cube)
    assert numba.get_num_threads() == before


def test_n_threads_caps_the_threads_of_the_pooling_routine(
    cells: dict[str, np.ndarray], cube: Cube, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    routine = pool_module._pool_cells

    def recording(*arguments: Any) -> None:
        seen.append(numba.get_num_threads())
        routine(*arguments)

    monkeypatch.setattr(pool_module, "_pool_cells", recording)
    assert_same_cube(pool(to_csr(cells["counts"]), cells, n_threads=1), cube)
    assert seen == [1]


# Dtypes of the counts layer.


def test_sums_above_the_largest_uint32_switch_counts_to_uint64() -> None:
    samples, groups = ["a", "a", "b", "b"], ["g", "g", "g", "h"]
    # Entry (a, g) of feature 0 sums to 2**32, one above the largest uint32; every
    # other layer must still count each cell once after counting again in uint64.
    above = np.array([[2**31, 1], [2**31, 2], [5, 0], [0, 7]])
    expected, n_cells = pooled_with_masks(
        above, np.array(samples), np.array(groups), ["a", "b"], ["g", "h"]
    )
    cube = from_cells(above, sample_labels=samples, group_labels=groups)
    assert cube.layer("counts").dtype == np.uint64
    assert cube.layer("counts", "g").tolist() == [[2**32, 3], [5, 0]]
    assert_pooled(cube, expected, n_cells)
    # 2**31 + (2**31 - 1) is the largest uint32 itself.
    at_limit = np.array([[2**31, 1], [2**31 - 1, 2], [5, 0], [0, 7]])
    cube = from_cells(at_limit, sample_labels=samples, group_labels=groups)
    assert cube.layer("counts").dtype == np.uint32
    assert cube.layer("counts", "g")[0, 0] == 2**32 - 1
    # A single count above the largest uint32.
    cube = from_cells(np.array([[2**33 + 1]]), sample_labels=["a"], group_labels=["g"])
    assert cube.layer("counts").dtype == np.uint64
    assert cube.layer("counts")[0, 0, 0] == 2**33 + 1


def test_float_counts(cells: dict[str, np.ndarray]) -> None:
    whole = cells["counts"].astype(np.float64)
    assert pool(whole, cells).layer("counts").dtype == np.uint32
    fractional = whole * 0.37
    expected, n_cells = pooled_with_masks(
        fractional, cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS
    )
    cube = pool(to_csr(fractional, np.float64), cells)
    assert cube.layer("counts").dtype == np.float64
    assert_pooled(cube, expected, n_cells, whole=False)
    # float32, the usual dtype of h5ad files, with fractions.
    single = fractional.astype(np.float32)
    expected, _ = pooled_with_masks(
        single, cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS
    )
    cube = pool(to_csr(single, np.float32), cells)
    assert cube.layer("counts").dtype == np.float64
    assert_pooled(cube, expected, n_cells, whole=False)
    # Whole numbers everywhere but in one cell: pooled first as integers, then
    # again as floats, with every layer counted once.
    one_fraction = whole.copy()
    last = np.flatnonzero(one_fraction[-1])[0]
    one_fraction[-1, last] += 0.5
    expected, _ = pooled_with_masks(
        one_fraction, cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS
    )
    cube = pool(one_fraction, cells)
    assert cube.layer("counts").dtype == np.float64
    assert_pooled(cube, expected, CELLS)


# Labels, tables and features.


def test_tables_set_the_order_and_keep_metadata(
    cells: dict[str, np.ndarray], cube: Cube
) -> None:
    sample_table = {
        "sample_id": ["S3", "S5", "S1", "S4", "S2"],  # S5 has no cells
        "donor": ["D2", "D3", "D1", "D2", "D1"],
    }
    group_table = Table(
        {
            "group_id": ["T", "NK", "B", "Mono"],
            "cell_ontology_id": ["CL:0000084", "CL:0000623", "CL:0000236", None],
        }
    )
    features = {
        "feature_id": [f"G{j}" for j in range(N_FEATURES)],
        "symbol": [f"GENE{j}" for j in range(N_FEATURES)],
    }
    ordered = pool(
        to_csr(cells["counts"]),
        cells,
        samples=sample_table,
        groups=group_table,
        features=features,
    )
    assert ordered.samples.equals(Table(sample_table))
    assert ordered.groups.equals(group_table)
    assert ordered.features["symbol"].tolist() == features["symbol"]
    samples = [SAMPLE_IDS.index(sample) for sample in ["S3", "S1", "S4", "S2"]]
    groups = [GROUP_IDS.index(group) for group in ["T", "NK", "B", "Mono"]]
    n_cells = ordered.n_cells
    assert n_cells is not None
    assert_array_equal(n_cells[[0, 2, 3, 4]], CELLS[np.ix_(samples, groups)])
    assert not n_cells[1].any()
    for name in LAYER_NAMES:
        layer = ordered.layer(name)
        assert not layer[:, 1].any()
        assert_array_equal(
            layer[:, [0, 2, 3, 4]], cube.layer(name)[np.ix_(groups, samples)]
        )
    ids = [f"G{j}" for j in range(N_FEATURES)]
    assert pool(cells["counts"], cells, features=ids).features.equals(
        Table({"feature_id": ids})
    )


def test_integer_and_categorical_labels() -> None:
    counts = np.arange(1, 13).reshape(6, 2)
    # Integer labels sort in numeric order and become text.
    cube = from_cells(
        counts, sample_labels=[10, 2, 2, 10, 1, 2], group_labels=[0, 0, 1, 1, 0, 0]
    )
    assert cube.samples["sample_id"].tolist() == ["1", "2", "10"]
    assert cube.groups["group_id"].tolist() == ["0", "1"]
    assert cube.layer("counts", "0").tolist() == [[9, 10], [14, 16], [1, 2]]
    # numpy reads a list that mixes text with NaN or numbers as text ("nan", "1").
    with pytest.raises(ValueError, match="sample_labels has a missing label at cell 2"):
        from_cells(
            counts,
            sample_labels=["a", "b", np.nan, "a", "b", "a"],
            group_labels=[0] * 6,
        )
    with pytest.raises(TypeError, match="group_labels mixes text and integer labels"):
        from_cells(
            counts, sample_labels=[1] * 6, group_labels=["a", "b", 1, "a", "b", "a"]
        )
    with pytest.raises(ValueError, match=r"features\['feature_id'\] has a missing"):
        from_cells(
            counts, sample_labels=[1] * 6, group_labels=[0] * 6, features=["x", np.nan]
        )
    pd = pytest.importorskip("pandas")
    missing = pd.Categorical(["a", None, "b", "a", "b", "a"])
    with pytest.raises(ValueError, match="sample_labels has a missing label at cell 1"):
        from_cells(counts, sample_labels=missing, group_labels=[0] * 6)
    # Categories sort like other labels, and unused categories are dropped.
    labels = pd.Categorical(["b", "a", "b", "a", "a", "b"], categories=["z", "b", "a"])
    for samples in (labels, pd.Series(labels), pd.Series(np.asarray(labels))):
        by_category = from_cells(counts, sample_labels=samples, group_labels=["g"] * 6)
        assert by_category.samples["sample_id"].tolist() == ["a", "b"]
        assert by_category.layer("counts", "g").tolist() == [[19, 22], [17, 20]]


def test_library_sizes_replace_the_cell_totals(cells: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(5)
    sizes = cells["counts"].sum(axis=1) * rng.uniform(1.0, 3.0, size=CELLS.sum())
    expected, _ = pooled_with_masks(
        cells["counts"],
        cells["samples"],
        cells["groups"],
        SAMPLE_IDS,
        GROUP_IDS,
        library_sizes=sizes,
    )
    cube = pool(to_csr(cells["counts"]), cells, library_sizes=sizes)
    assert_pooled(cube, expected, CELLS)
    normalization = cube.normalization
    assert normalization is not None and normalization["library_sizes"] == "given"


def test_cells_without_counts(cells: dict[str, np.ndarray]) -> None:
    counts = cells["counts"].copy()
    counts[[4, 9]] = 0
    with pytest.raises(
        ValueError, match="cell 4 .* has no counts, so its library size"
    ):
        pool(to_csr(counts), cells)
    # Without normalized layers they are pooled, and counted in n_cells.
    cube = pool(to_csr(counts), cells, layers=["counts", "detected"])
    expected, n_cells = pooled_with_masks(
        counts, cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS
    )
    assert_array_equal(n_cells, CELLS)
    assert_pooled(cube, expected, CELLS)
    assert cube.normalization is None
    # With library sizes they can be normalized: their normalized values are 0.
    sizes = np.full(counts.shape[0], 1000.0)
    expected, _ = pooled_with_masks(
        counts, cells["samples"], cells["groups"], SAMPLE_IDS, GROUP_IDS, sizes
    )
    assert_pooled(pool(counts, cells, library_sizes=sizes), expected, CELLS)


@pytest.mark.parametrize(
    "layers",
    [["counts"], ["counts", "detected"], ["linear_sum", "counts"], "counts"],
)
def test_layer_subsets_equal_the_full_cube(
    cells: dict[str, np.ndarray], cube: Cube, layers: list[str] | str
) -> None:
    subset = pool(to_csr(cells["counts"]), cells, layers=layers)
    listed = [layers] if isinstance(layers, str) else layers
    assert subset.layer_names == tuple(name for name in LAYER_NAMES if name in listed)
    for name in subset.layer_names:
        assert_array_equal(subset.layer(name), cube.layer(name))
    assert (subset.normalization is None) == ("linear_sum" not in listed)


def test_empty_inputs() -> None:
    no_cells = from_cells(
        np.zeros((0, 3)),
        sample_labels=[],
        group_labels=[],
        samples={"sample_id": ["S1"]},
        groups={"group_id": ["g"]},
    )
    assert no_cells.layer("counts").shape == (1, 1, 3)
    assert not no_cells.layer("log_sum").any()
    assert no_cells.n_cells is not None and no_cells.n_cells.tolist() == [[0]]
    no_features = from_cells(
        CSR(np.zeros(0), np.zeros(0, np.int32), np.zeros(3, np.int32), (2, 0)),
        sample_labels=["a", "b"],
        group_labels=["g", "g"],
        layers=["counts", "detected"],
    )
    assert no_features.layer("counts").shape == (1, 2, 0)


def test_matches_scanpy_and_decoupler() -> None:
    # Written by validation/pool_cells.py --write-fixtures: scanpy.get.aggregate on
    # cells that scanpy normalized itself, and decoupler's pseudobulk sums.
    data = np.load(FIXTURE, allow_pickle=False)
    counts = CSR(data["data"], data["indices"], data["indptr"], tuple(data["shape"]))
    cube = from_cells(counts, sample_labels=data["sample"], group_labels=data["group"])
    groups = [cube.groups["group_id"].tolist().index(g) for g in data["keys_group"]]
    samples = [cube.samples["sample_id"].tolist().index(s) for s in data["keys_sample"]]
    n_cells = cube.n_cells
    assert n_cells is not None
    assert_array_equal(n_cells[samples, groups], data["scanpy_n_cells"])
    assert n_cells.sum() == n_cells[samples, groups].sum()  # no other filled entry
    assert_array_equal(cube.layer("counts")[groups, samples], data["scanpy_sum"])
    assert_array_equal(
        cube.layer("detected")[groups, samples], data["scanpy_count_nonzero"]
    )
    # scanpy sums with sparse matrix products, in another order than the cube.
    assert_allclose(cube.mean()[groups, samples], data["scanpy_mean_log"], rtol=1e-12)
    assert_allclose(
        cube.mean(scale="linear")[groups, samples],
        data["scanpy_mean_linear"],
        rtol=1e-12,
    )
    several = data["scanpy_n_cells"] >= 2
    variance = cube.variance()[groups, samples]
    assert_allclose(variance[several], data["scanpy_var_log"][several], rtol=1e-10)
    assert np.isnan(variance[~several]).all() and (~several).any()
    groups = [cube.groups["group_id"].tolist().index(g) for g in data["dc_group"]]
    samples = [cube.samples["sample_id"].tolist().index(s) for s in data["dc_sample"]]
    assert_array_equal(cube.layer("counts")[groups, samples], data["dc_sum"])
    assert_array_equal(n_cells[samples, groups], data["dc_n_cells"])
    assert_array_equal(cube.library_sizes()[samples, groups], data["dc_counts"])
    # Both divide the same whole numbers. decoupler keeps the entry without cells
    # and calls its fraction 0, where the cube leaves 0 / 0 undefined.
    filled = data["dc_n_cells"] > 0
    fraction = cube.fraction_detected()[groups, samples]
    assert_allclose(fraction[filled], data["dc_props"][filled], rtol=1e-15)
    assert (~filled).sum() == 1
    assert np.isnan(fraction[~filled]).all() and not data["dc_props"][~filled].any()


# Bulk counts.


def test_from_bulk() -> None:
    counts = np.array([[5, 0, 12], [7, 1, 9]])
    cube = from_bulk(counts, samples=["A", "B"], features=["x", "y", "z"])
    assert cube.layer_names == ("counts",)
    assert cube.layer("counts").dtype == np.uint32
    assert cube.layer("counts", "bulk").tolist() == counts.tolist()
    assert cube.groups["group_id"].tolist() == ["bulk"]
    assert cube.samples["sample_id"].tolist() == ["A", "B"]
    assert cube.features["feature_id"].tolist() == ["x", "y", "z"]
    assert cube.n_cells is None and cube.normalization is None
    assert cube.provenance == {
        "source": None,
        "construction": {
            "function": "from_bulk",
            "cytepool_version": cytepool.__version__,
            "samples": "ids",
            "features": "ids",
        },
    }
    assert cube.library_sizes()[:, 0].tolist() == [17, 17]
    default = from_bulk(counts.astype(np.float32))
    assert default.layer("counts").dtype == np.uint32
    assert default.samples["sample_id"].tolist() == ["sample_0", "sample_1"]
    assert default.features["feature_id"].tolist() == [
        "feature_0",
        "feature_1",
        "feature_2",
    ]
    estimates = from_bulk(counts * 0.5)
    assert estimates.layer("counts").dtype == np.float64
    assert estimates.layer("counts", 0).tolist() == (counts * 0.5).tolist()
    large = from_bulk(np.array([[2**32, 1]], dtype=np.int64))
    assert large.layer("counts").dtype == np.uint64
    assert large.layer("counts")[0, 0, 0] == 2**32
    largest = from_bulk(np.array([[2**63 - 1, 0]], dtype=np.uint64))
    assert largest.layer("counts")[0].tolist() == [[2**63 - 1, 0]]
    assert largest.library_sizes()[:, 0].tolist() == [2**63 - 1]
    tables = from_bulk(
        counts,
        samples=Table({"sample_id": ["A", "B"], "donor": ["D1", "D1"]}),
        features={"feature_id": ["x", "y", "z"], "symbol": ["X", "Y", "Z"]},
    )
    assert tables.features["symbol"].tolist() == ["X", "Y", "Z"]
    pooled = tables.collapse_samples("donor")
    assert pooled.layer("counts", 0).tolist() == [[12, 1, 21]]


def test_text_labels_of_a_data_frame_give_the_ids() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[5, 0, 12], [7, 1, 9]], index=["A", "B"], columns=list("xyz"))
    cube = from_bulk(frame)
    assert cube.samples["sample_id"].tolist() == ["A", "B"]
    assert cube.features["feature_id"].tolist() == ["x", "y", "z"]
    assert cube.layer("counts", 0).tolist() == [[5, 0, 12], [7, 1, 9]]
    # Positions are not ids, and given ids take precedence.
    unnamed = from_bulk(frame.reset_index(drop=True), features=["u", "v", "w"])
    assert unnamed.samples["sample_id"].tolist() == ["sample_0", "sample_1"]
    assert unnamed.features["feature_id"].tolist() == ["u", "v", "w"]
    pooled = from_cells(frame, sample_labels=["s", "s"], group_labels=["g", "g"])
    assert pooled.features["feature_id"].tolist() == ["x", "y", "z"]
    assert pooled.layer("counts", "g").tolist() == [[12, 1, 21]]


@pytest.mark.parametrize(
    ("counts", "options", "error", "message"),
    [
        (
            np.array([[1, -2]]),
            {},
            ValueError,
            r"negative value \(-2\) in row 0, column 1",
        ),
        (np.array([[1.0, np.nan]]), {}, ValueError, r"non-finite value \(nan\)"),
        (
            np.array([[0, 2**63]], dtype=np.uint64),
            {},
            ValueError,
            r"counts has a value above 2\*\*63 - 1 \(9223372036854775808\) in row 0, "
            r"column 1; integer counts must be at most 2\*\*63 - 1",
        ),
        (np.array([1, 2]), {}, ValueError, r"must be 2-D.*got shape \(2,\)"),
        (np.array([[True]]), {}, TypeError, "must hold numbers"),
        (np.ma.masked_array([[1, 2]], mask=[[0, 1]]), {}, TypeError, "masked array"),
        (np.ones((2, 3)), {"samples": ["A"]}, ValueError, "one id per row of counts"),
        (
            np.ones((2, 3)),
            {"features": {"feature_id": ["x", "y"]}},
            ValueError,
            "the features table has 2 rows, but counts has 3 columns",
        ),
    ],
)
def test_from_bulk_rejects_invalid_input(
    counts: np.ndarray, options: dict[str, Any], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        from_bulk(counts, **options)


# Invalid input.


def matrix_with(dense: np.ndarray, **changes: Any) -> CSR:
    matrix = to_csr(dense, np.float64)
    for name, value in changes.items():
        setattr(matrix, name, value(getattr(matrix, name)))
    return matrix


def csc_arrays(dense: np.ndarray) -> CSR:
    """The arrays of ``dense`` in CSC layout, with no format to say so."""
    transposed = to_csr(np.ascontiguousarray(dense.T))
    return CSR(
        transposed.data, transposed.indices, transposed.indptr, dense.shape, None
    )


def with_entry(array: np.ndarray, index: Any, value: Any) -> np.ndarray:
    changed = array.copy()
    changed[index] = value
    return changed


INVALID_CALLS = [
    (
        "CSC matrix",
        lambda d: {"counts": matrix_with(d, format=lambda _: "csc")},
        TypeError,
        r"'csc' format, but cells must be the rows of a CSR matrix; convert it with "
        r"counts.tocsr\(\)",
    ),
    (
        "COO matrix",
        lambda d: {"counts": matrix_with(d, format=lambda _: "coo")},
        TypeError,
        "'coo' format",
    ),
    (
        "CSC arrays without a format",
        lambda d: {"counts": csc_arrays(d)},
        ValueError,
        r"counts.indptr has 25 entries, but a CSR matrix with 103 rows needs 104; it "
        r"looks like CSC layout, so convert it with counts.tocsr\(\)",
    ),
    (
        "indptr too short",
        lambda d: {"counts": matrix_with(d, indptr=lambda p: p[:-1])},
        ValueError,
        r"has 103 entries, but a CSR matrix with 103 rows needs 104",
    ),
    (
        "indptr decreasing",
        lambda d: {"counts": matrix_with(d, indptr=lambda p: with_entry(p, 5, 0))},
        ValueError,
        "must start at 0, never decrease",
    ),
    (
        "data shorter than indptr says",
        lambda d: {"counts": matrix_with(d, data=lambda x: x[:-1])},
        ValueError,
        r"end at most at the number of stored values \(\d+\)",
    ),
    (
        "indptr starting at 1",
        lambda d: {"counts": matrix_with(d, indptr=lambda p: with_entry(p, 0, 1))},
        ValueError,
        "counts.indptr must start at 0",
    ),
    (
        "negative sizes",
        lambda d: {"counts": CSR(np.zeros(0), np.zeros(0), np.zeros(0), (-1, 3))},
        ValueError,
        r"counts must be 2-D, with sizes of 0 or more, got \(-1, 3\)",
    ),
    (
        "unsigned indices beyond int64",
        lambda d: {
            "counts": matrix_with(
                d, indices=lambda i: with_entry(i.astype(np.uint64), 3, 2**63)
            )
        },
        ValueError,
        "counts.indices holds the column index 9223372036854775808, but counts has "
        "24 columns",
    ),
    (
        "unsigned indptr beyond int64",
        lambda d: {
            "counts": matrix_with(
                d, indptr=lambda p: with_entry(p.astype(np.uint64), 103, 2**64 - 1)
            )
        },
        ValueError,
        "counts.indptr holds 18446744073709551615, more than the number of stored",
    ),
    (
        "negative column index",
        lambda d: {"counts": matrix_with(d, indices=lambda i: with_entry(i, 7, -1))},
        ValueError,
        r"column index -1 in row \d+, but counts has 24 columns",
    ),
    (
        "masked array",
        lambda d: {"counts": np.ma.masked_array(d, mask=d > 3)},
        TypeError,
        "counts is a numpy masked array; CytePool does not accept",
    ),
    (
        "column index outside the matrix",
        lambda d: {"counts": matrix_with(d, indices=lambda i: with_entry(i, 7, 24))},
        ValueError,
        r"column index 24 in row \d+, but counts has 24 columns",
    ),
    (
        "negative count",
        lambda d: {"counts": with_entry(d.astype(float), (3, 2), -1.0)},
        ValueError,
        r"counts has a negative value \(-1\) in row 3, column 2",
    ),
    (
        "NaN count",
        lambda d: {"counts": matrix_with(d, data=lambda x: with_entry(x, 0, np.nan))},
        ValueError,
        r"counts has a NaN in row 0, column \d+",
    ),
    (
        "infinite count",
        lambda d: {"counts": with_entry(d.astype(float), (5, 3), np.inf)},
        ValueError,
        "counts has an infinite value in row 5, column 3",
    ),
    (
        "count at 2**53",
        lambda d: {"counts": with_entry(d, (6, 1), 2**53)},
        ValueError,
        r"a value of 9.0072e\+15, at or above 2\*\*53 in row 6, column 1",
    ),
    (
        "boolean counts",
        lambda d: {"counts": d > 0},
        TypeError,
        "counts must hold numbers",
    ),
    (
        "1-D counts",
        lambda d: {"counts": d[0]},
        ValueError,
        r"counts must be 2-D.*got shape \(24,\)",
    ),
    (
        "too few labels",
        lambda d: {"sample_labels": np.array(["S1"] * 102)},
        ValueError,
        "counts has 103 rows, sample_labels has 102 labels",
    ),
    (
        "missing label",
        lambda d: {
            "group_labels": np.array(["B"] * 50 + [None] + ["B"] * 52, dtype=object)
        },
        ValueError,
        "group_labels has a missing label at cell 50",
    ),
    (
        "empty label",
        lambda d: {"sample_labels": np.array(["S1"] * 102 + [""])},
        ValueError,
        "sample_labels has an empty label at cell 102",
    ),
    (
        "float labels",
        lambda d: {"sample_labels": np.ones(103)},
        TypeError,
        "sample_labels must hold text or integer labels, got dtype float64",
    ),
    (
        "mixed labels",
        lambda d: {"sample_labels": np.array(["S1", 1] * 51 + ["S1"], dtype=object)},
        TypeError,
        "mixes text and integer labels",
    ),
    (
        "2-D labels",
        lambda d: {"sample_labels": np.array([["S1"] * 103])},
        ValueError,
        r"sample_labels must be 1-D, one label per cell, got shape \(1, 103\)",
    ),
    (
        "label without a row",
        lambda d: {"samples": {"sample_id": ["S1", "S2", "S4"]}},
        ValueError,
        r"the label 'S3' of cell \d+ in sample_labels is not in samples\['sample_id'\]",
    ),
    (
        "table without ids",
        lambda d: {"groups": {"name": GROUP_IDS}},
        ValueError,
        "the groups table needs a 'group_id' column",
    ),
    (
        "too few feature ids",
        lambda d: {"features": [f"g{j}" for j in range(23)]},
        ValueError,
        r"features must hold one id per column of counts \(24\), got 23",
    ),
    (
        "integer feature ids",
        lambda d: {"features": list(range(24))},
        TypeError,
        r"features\['feature_id'\] must hold text ids",
    ),
    (
        "unknown layer",
        lambda d: {"layers": ["counts", "means"]},
        ValueError,
        "unknown layer 'means'; layers are named: counts, detected",
    ),
    (
        "layers without counts",
        lambda d: {"layers": ["log_sum"]},
        ValueError,
        "layers must include 'counts'",
    ),
    (
        "repeated layer",
        lambda d: {"layers": ["counts", "counts"]},
        ValueError,
        "lists a layer more than once",
    ),
    (
        "zero target sum",
        lambda d: {"target_sum": 0},
        ValueError,
        "target_sum must be positive and finite, got 0.0",
    ),
    (
        "boolean target sum",
        lambda d: {"target_sum": True},
        TypeError,
        "target_sum must be a number, got bool",
    ),
    (
        "short library sizes",
        lambda d: {"library_sizes": np.ones(5)},
        ValueError,
        r"library_sizes must hold one value per cell \(103\), got shape \(5,\)",
    ),
    (
        "zero library size",
        lambda d: {"library_sizes": with_entry(np.ones(103), 8, 0.0)},
        ValueError,
        "library_sizes must be positive and finite; cell 8 has 0",
    ),
    (
        "no threads",
        lambda d: {"n_threads": 0},
        ValueError,
        "n_threads must be at least 1",
    ),
]


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [case[1:] for case in INVALID_CALLS],
    ids=[case[0] for case in INVALID_CALLS],
)
def test_invalid_calls_are_rejected(
    cells: dict[str, np.ndarray], change: Any, error: type, message: str
) -> None:
    arguments: dict[str, Any] = {
        "counts": cells["counts"],
        "sample_labels": cells["samples"],
        "group_labels": cells["groups"],
    }
    arguments.update(change(cells["counts"]))
    with pytest.raises(error, match=message):
        from_cells(**arguments)


def test_sums_up_to_the_largest_int64_are_exact_and_beyond_it_raise() -> None:
    # 1,024 cells of 2**53 - 1, the largest count, and one of 1,023 sum to 2**63 - 1,
    # the largest int64; float64 would round that sum to 2**63.
    counts = np.append(np.full(1024, 2**53 - 1), 1023).reshape(-1, 1)
    labels = {
        "sample_labels": ["a"] * counts.shape[0],
        "group_labels": ["g"] * counts.shape[0],
    }
    cube = from_cells(counts, **labels)
    assert cube.layer("counts").dtype == np.uint64
    assert cube.layer("counts")[0, 0, 0] == 2**63 - 1
    assert cube.library_sizes()[0, 0] == 2**63 - 1
    # Pooled again in uint64 after the uint32 sum stopped, every cell counts once:
    # each normalized value is count * (target_sum / count), target_sum to rounding,
    # and a cell counted twice would double the sum.
    assert cube.layer("detected")[0, 0, 0] == 1025
    assert cube.layer("linear_sum")[0, 0, 0] == pytest.approx(1025 * TARGET_SUM, 1e-12)
    message = re.escape(
        "layer 'counts' at group 'g', sample 'a', feature 'feature_0' sums over its "
        "cells to more than 2**63 - 1, the largest int64"
    )
    for beyond in (1024, 2**53 - 1):  # sums of exactly 2**63, and 2**63 + 2**53 - 1025
        counts[-1, 0] = beyond
        with pytest.raises(OverflowError, match=message):
            from_cells(counts, **labels)
    # One count that is not a whole number makes the counts float64, which have no
    # such limit, wherever that cell comes: here after the sum passes 2**63 - 1.
    labels = {"sample_labels": ["a"] * 1026, "group_labels": ["g"] * 1026}
    expected = sum(counts[:, 0].tolist()) + 0.5  # Python integers, exact
    for fractional in (np.vstack([counts, [[0.5]]]), np.vstack([[[0.5]], counts])):
        cube = from_cells(fractional, **labels, layers="counts")
        assert cube.layer("counts").dtype == np.float64
        # 1,025 float64 additions, each rounding by at most 2**-53 relative; a cell
        # counted twice would be off by 1e-3.
        assert cube.layer("counts")[0, 0, 0] == pytest.approx(expected, rel=1e-12)


def test_an_entry_with_more_cells_than_detected_counts_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2**32 cells would take gigabytes, so the limit is checked on cell numbers...
    groups, samples = np.array(["g"], dtype=object), np.array(["a", "b"], dtype=object)
    ids = (groups, samples, np.array(["f"], dtype=object))
    pool_module._check_detected_fits(np.array([7, 2**32 - 1]), ids)
    with pytest.raises(
        OverflowError,
        match=re.escape(
            "group 'g', sample 'b' has 4,294,967,296 cells, more than the "
            "4,294,967,295 that the uint32 layer 'detected' counts"
        ),
    ):
        pool_module._check_detected_fits(np.array([7, 2**32]), ids)
    # ... and from_cells is run with the limit lowered to two cells.
    monkeypatch.setattr(pool_module, "_MOST_DETECTED", 2)
    labels = {"sample_labels": ["a", "b", "a", "a", "b"], "group_labels": ["g"] * 5}
    with pytest.raises(OverflowError, match="sample 'a' has 3 cells, more than the 2"):
        from_cells(np.ones((5, 1)), **labels)
    counts_only = from_cells(np.ones((5, 1)), **labels, layers="counts")
    assert counts_only.layer("counts", "g").tolist() == [[3], [2]]
    at_limit = from_cells(
        np.ones((4, 1)), sample_labels=["a", "b"] * 2, group_labels=["g"] * 4
    )
    assert at_limit.layer("detected", "g").tolist() == [[2], [2]]


def test_normalized_values_that_sum_beyond_float64_are_rejected() -> None:
    # One cell's normalized values sum to target_sum, 1e308, which float64 holds;
    # two such cells in an entry do not fit.
    one = from_cells(
        np.array([[1]]), sample_labels=["a"], group_labels=["g"], target_sum=1e308
    )
    assert one.layer("linear_sum")[0, 0, 0] == 1e308
    with pytest.raises(
        OverflowError,
        match="layer 'linear_sum' at group 'g', sample 'a', feature 'feature_0' sums "
        "over its cells beyond the float64 range",
    ):
        from_cells(
            np.array([[1], [1]]),
            sample_labels=["a", "a"],
            group_labels=["g", "g"],
            target_sum=1e308,
        )


def test_normalized_values_beyond_float64_are_rejected() -> None:
    message = r"normalized values of cell 0 \(row 0 of counts\).*exceed the float64"
    # target_sum / library size is 5e309, beyond float64 for both counts.
    with pytest.raises(ValueError, match=message):
        from_cells(
            np.array([[1e-306, 1e-306]]), sample_labels=["a"], group_labels=["g"]
        )
    with pytest.raises(ValueError, match=message):
        from_cells(
            np.array([[1, 2]]),
            sample_labels=["a"],
            group_labels=["g"],
            library_sizes=[1e-306],
        )
    # Counts alone need no normalization.
    cube = from_cells(
        np.array([[1e-306, 1e-306]]),
        sample_labels=["a"],
        group_labels=["g"],
        layers="counts",
    )
    assert cube.layer("counts")[0, 0].tolist() == [1e-306, 1e-306]


def test_repeated_entries_that_sum_too_high_are_rejected() -> None:
    matrix = CSR(
        np.array([2.0**52, 2.0**52]), np.array([0, 0]), np.array([0, 2]), (1, 1)
    )
    with pytest.raises(ValueError, match="repeated entries that sum to 9.0072e"):
        from_cells(matrix, sample_labels=["a"], group_labels=["g"])
