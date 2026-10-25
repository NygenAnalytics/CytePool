"""Converting cubes to and from AnnData and MuData.

Imports are compared with ``from_cells`` on the same cells, a separate code path
that validation/pool_cells.py checks against scanpy and decoupler, using the outputs
of scanpy, decoupler and LIANA that validation/anndata_conversion.py saved in
tests/data/anndata_conversion. Round trips are checked by the fingerprint, which
covers every value, table and column type. Tests that need anndata or mudata are
skipped without them.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cytepool
from cytepool.cube import Cube, from_bulk, from_cells
from cytepool.cube.provenance import describe_provenance
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.io import read_cube, write_cube
from cytepool.io.anndata import from_anndata, from_mudata, to_anndata, to_mudata

# anndata and mudata warn about their own use of pandas options that pandas 4 removes.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:anndata"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning:mudata"),
]

DATA = Path(__file__).resolve().parent / "data" / "anndata_conversion"
LAYERS = ("counts", "detected", "log_sum", "log_sum_sq", "linear_sum")


@pytest.fixture(scope="module")
def ad() -> Any:
    return pytest.importorskip("anndata")


@pytest.fixture(scope="module")
def md() -> Any:
    pytest.importorskip("anndata")
    return pytest.importorskip("mudata")


@pytest.fixture(scope="module")
def pd() -> Any:
    return pytest.importorskip("pandas")


def _cells(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.5, size=(90, 6))
    counts[counts.sum(axis=1) == 0, 0] = 1
    samples = rng.choice(["S1", "S2", "S3"], 90)
    groups = rng.choice(["B", "T", "NK"], 90)
    samples[(samples == "S3") & (groups == "T")] = "S1"  # an entry without cells
    return counts, samples, groups


def varied_cube() -> Cube:
    """Every layer, tables of every column type, columns without any value, a sample
    and a group without cells, and names that anndata cannot write as they are."""
    counts, samples, groups = _cells()
    return from_cells(
        counts,
        samples=samples,
        groups=groups,
        sample_table={
            "sample_id": ["S1", "S2", "S3", "S4"],
            "collected": np.array(
                ["2024-01-05", "NaT", "2024-03-01", "2024-04-11"], "datetime64[D]"
            ),
            "age": [41.5, np.nan, -0.0, 63.0],
            "site/batch": ["x", None, "y", "z"],
            "n_cells": np.array([1, 2, 3, 2**64 - 1], dtype=np.uint64),
            "smoker": [True, False, False, True],
            "donor": ["D1", "D2", "D1", "D3"],
            "notes": [None, None, None, None],
            "..": [1, 2, 3, 4],
        },
        group_table={
            "group_id": ["B", "T", "NK", "Plasma"],
            "parent": ["L", "L", None, "L"],
            "since": np.full(4, "NaT", dtype="datetime64[s]"),
        },
        features={
            "feature_id": [f"g{j}" for j in range(6)],
            "symbol": ["CD3E", None, "GZMB", "NKG7", "MS4A1", "LYZ"],
            "length": np.arange(6).astype("timedelta64[ms]"),
            "weight": np.linspace(0, 1, 6).astype(np.float16),
            "phase": np.arange(6) * (1 + 1j),
            "_index": np.arange(6, dtype=np.int8),
            "alias": [None] * 6,
            "a\x00b": np.arange(6.0),
        },
        source={"dataset": "simulated", "version": 2},
    )


def whole_float_counts() -> Cube:
    """Counts stored as float64 although whole: the type is part of the fingerprint."""
    counts = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    return Cube(
        {"counts": counts},
        samples={"sample_id": ["a", "b", "c"]},
        groups={"group_id": ["g"]},
        features={"feature_id": ["f1", "f2", "f3", "f4"]},
    )


def cubes() -> dict[str, Cube]:
    cube = varied_cube()
    return {
        "varied": cube,
        "derived": cube.regroup({"lymphoid": ["B", "T"]})
        .collapse_samples("donor")
        .select(features=[4, 0, 2]),
        "bulk": from_bulk(np.arange(15).reshape(3, 5), samples=["x", "y", "z"]),
        "whole float counts": whole_float_counts(),
    }


CUBES = cubes()


# Round trips.


def _check_read_back(back: Cube, cube: Cube, function: str, source: Any) -> None:
    assert back.fingerprint() == cube.fingerprint()
    # The types of the layers come back too, not only their values.
    assert [back.layer(n).dtype for n in back.layer_names] == [
        cube.layer(n).dtype for n in cube.layer_names
    ]
    # The provenance continues with the import, which names the exported cube.
    operations = back.provenance["operations"]
    assert back.provenance == {**cube.provenance, "operations": operations}
    assert operations[-1] == {
        "operation": function,
        "input": cube.fingerprint(),
        "source": source,
    }


@pytest.mark.parametrize("name", list(CUBES))
def test_a_cube_comes_back_from_anndata_with_its_fingerprint(
    name: str, ad: Any
) -> None:
    cube = CUBES[name]

    _check_read_back(from_anndata(to_anndata(cube)), cube, "from_anndata", None)


@pytest.mark.parametrize("name", list(CUBES))
def test_a_cube_comes_back_from_mudata_with_its_fingerprint(name: str, md: Any) -> None:
    cube = CUBES[name]
    back = from_mudata(to_mudata(cube), source={"note": "read back"})

    _check_read_back(back, cube, "from_mudata", {"note": "read back"})


def test_files_that_anndata_and_mudata_write_read_back_as_the_cube(
    ad: Any, md: Any, tmp_path: Path
) -> None:
    # The cube whose tables hold every column type, a sample and a group without
    # cells, and names that anndata cannot write as they are. (.zarr files, which
    # zarr-python writes slowly, are read back in validation/anndata_conversion.py.)
    cube = CUBES["varied"]
    to_anndata(cube).write_h5ad(tmp_path / "cube.h5ad")
    to_mudata(cube).write(tmp_path / "cube.h5mu")

    back = from_anndata(ad.read_h5ad(tmp_path / "cube.h5ad"))
    _check_read_back(back, cube, "from_anndata", None)
    back = from_mudata(md.read_h5mu(tmp_path / "cube.h5mu"))
    _check_read_back(back, cube, "from_mudata", None)


def test_a_cube_read_from_a_cube_file_converts_as_in_memory(
    ad: Any, md: Any, tmp_path: Path
) -> None:
    cube = varied_cube()
    write_cube(cube, tmp_path / "x.cube.zarr")
    lazy = read_cube(tmp_path / "x.cube.zarr")

    assert from_anndata(to_anndata(lazy)).fingerprint() == cube.fingerprint()
    assert from_mudata(to_mudata(lazy)).fingerprint() == cube.fingerprint()


# What the exports hold.


def test_to_anndata_writes_a_row_for_each_entry_with_cells(ad: Any) -> None:
    cube = varied_cube()
    adata = to_anndata(cube)
    cells = cube.n_cells
    assert cells is not None
    # Group by group, samples in order; entries without cells have no row.
    groups, samples = np.nonzero(cells.T > 0)
    sample_ids = cube.samples["sample_id"]
    group_ids = cube.groups["group_id"]

    assert adata.X is None
    assert adata.obs["sample_id"].tolist() == sample_ids[samples].tolist()
    assert adata.obs["group_id"].tolist() == group_ids[groups].tolist()
    assert adata.obs["sample_id"].dtype.name == "category"
    assert list(adata.obs_names) == [
        f"{s}_{g}" for s, g in zip(sample_ids[samples], group_ids[groups], strict=True)
    ]
    assert_array_equal(adata.obs["n_cells"].to_numpy(), cells[samples, groups])
    assert set(adata.layers.keys()) - {None} == set(LAYERS)
    for name in LAYERS:
        assert adata.layers[name].dtype == cube.layer(name).dtype
        assert_array_equal(adata.layers[name], cube.layer(name)[groups, samples])
    assert list(adata.var_names) == cube.features["feature_id"].tolist()
    # Every column of the sample and group tables, in a form anndata writes: a
    # name with "/" or "_index" changed, a taken name prefixed, datetimes as ISO
    # 8601 text and timedeltas as int64 counts of their unit.
    assert list(adata.obs.columns) == [
        "sample_id",
        "group_id",
        "collected",
        "age",
        "site_batch",
        "sample_n_cells",
        "smoker",
        "donor",
        "notes",
        ".._",
        "parent",
        "since",
        "n_cells",
    ]
    assert adata.obs["collected"].iloc[0] == "2024-01-05"
    assert set(adata.obs["since"]) == {"NaT"}
    assert adata.obs["sample_n_cells"].dtype == np.uint64
    # A text column without any value, which anndata cannot write, is NaN.
    assert adata.obs["notes"].dtype == np.float64
    assert list(adata.var.columns) == [
        "symbol",
        "length",
        "weight",
        "phase",
        "_index_",
        "alias",
        "a_b",
    ]
    assert adata.var["length"].tolist() == list(range(6))
    record = adata.uns["cytepool"]
    assert record["format"] == "cytepool.cube.anndata"
    assert record["format_version"] == 1
    assert record["fingerprint"] == cube.fingerprint()


def test_to_anndata_names_rows_by_position_when_names_would_repeat(ad: Any) -> None:
    # "a_b" + "_" + "c" and "a" + "_" + "b_c" are the same name.
    cube = Cube(
        {"counts": np.ones((2, 2, 1), dtype=np.uint32)},
        samples={"sample_id": ["a_b", "a"]},
        groups={"group_id": ["c", "b_c"]},
        features={"feature_id": ["f"]},
    )

    adata = to_anndata(cube)

    assert list(adata.obs_names) == ["0", "1", "2", "3"]
    assert from_anndata(adata).fingerprint() == cube.fingerprint()


def test_to_anndata_of_a_bulk_cube_writes_every_entry(ad: Any) -> None:
    cube = CUBES["bulk"]
    adata = to_anndata(cube)

    assert list(adata.obs.columns) == ["sample_id", "group_id"]
    assert_array_equal(adata.layers["counts"], cube.layer("counts", "bulk"))


def test_to_mudata_writes_a_modality_for_each_group(md: Any, pd: Any) -> None:
    cube = varied_cube()
    mdata = to_mudata(cube)
    cells = cube.n_cells
    assert cells is not None
    sample_ids = cube.samples["sample_id"]

    assert list(mdata.mod) == ["B", "T", "NK", "Plasma"]
    for g, name in enumerate(mdata.mod):
        modality = mdata.mod[name]
        rows = np.flatnonzero(cells[:, g])
        assert list(modality.obs_names) == sample_ids[rows].tolist()
        assert_array_equal(modality.obs["n_cells"].to_numpy(), cells[rows, g])
        assert list(modality.var_names) == [
            f"{name}:{feature}" for feature in cube.features["feature_id"]
        ]
        assert modality.X is None
        for layer in LAYERS:
            assert_array_equal(modality.layers[layer], cube.layer(layer, g)[rows])
    assert mdata.mod["Plasma"].n_obs == 0  # a group without cells
    # Sample metadata on the MuData's obs, for the samples with rows.
    assert pd.isna(mdata.obs.loc["S2", "site_batch"])
    assert mdata.obs.loc["S3", "site_batch"] == "y"
    assert mdata.uns["cytepool"]["format"] == "cytepool.cube.mudata"


# Imports of what scanpy, decoupler and LIANA made of the same cells.


@pytest.fixture(scope="module")
def cells() -> dict[str, Any]:
    sparse = pytest.importorskip("scipy.sparse")
    data = dict(np.load(DATA / "cells.npz"))
    data["counts"] = sparse.csr_matrix(
        (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"])
    )
    return data


def _pooled(cells: dict[str, Any], **tables: Any) -> Cube:
    return from_cells(
        cells["counts"],
        samples=cells["sample"],
        groups=cells["group"],
        features={"feature_id": cells["feature_id"], "symbol": cells["symbol"]},
        layers=["counts", "detected"],
        **tables,
    )


def _metadata(cells: dict[str, Any], labels: str, values: str) -> dict[str, list[str]]:
    """The value of a per-cell column for each label, which is the same in every cell
    of a label in the simulation."""
    found = dict(zip(cells[labels].tolist(), cells[values].tolist(), strict=True))
    ids = sorted(found)
    return {f"{labels}_id": ids, values: [found[i] for i in ids]}


def test_scanpy_aggregate_imports_as_the_cube_of_its_cells(
    ad: Any, cells: dict[str, Any]
) -> None:
    aggregated = ad.read_h5ad(DATA / "scanpy_counts.h5ad")

    cube = from_anndata(aggregated, sample="sample", group="group")

    # Counts, detected cells and cell numbers, ids and tables: the same cube.
    assert cube.fingerprint() == _pooled(cells).fingerprint()
    assert cube.layer("counts").dtype == np.uint32
    construction = cube.provenance["construction"]
    assert construction["layout"] == "scanpy.get.aggregate"
    assert construction["layers"] == {"counts": "sum", "detected": "count_nonzero"}
    assert construction["n_cells"] == "n_obs_aggregated"


def test_decoupler_pseudobulk_imports_with_the_metadata_of_samples_and_groups(
    ad: Any, cells: dict[str, Any]
) -> None:
    pseudobulk = ad.read_h5ad(DATA / "decoupler.h5ad")
    # decoupler writes a row for the entry without cells, with metadata that need
    # not be its own.
    assert (pseudobulk.obs["psbulk_cells"] == 0).sum() == 1

    cube = from_anndata(pseudobulk, sample="sample", group="group")

    # Detected cells come from decoupler's fractions times its cell numbers.
    samples = _metadata(cells, "sample", "condition")
    groups = _metadata(cells, "group", "lineage")
    expected = _pooled(cells, sample_table=samples, group_table=groups)
    assert cube.fingerprint() == expected.fingerprint()
    assert cube.provenance["construction"]["obs_columns"] == {
        "samples": ["condition"],
        "groups": ["lineage"],
        "left_out": [],
    }


def test_liana_views_import_with_the_cell_numbers_of_the_cells(
    md: Any, pd: Any, cells: dict[str, Any]
) -> None:
    views = md.read_h5mu(DATA / "liana.h5mu")
    # LIANA keeps no cell numbers (keep_stats looks for decoupler 1's column).
    with pytest.raises(ValueError, match="no cell numbers to make it an entry without"):
        from_mudata(views)

    crosstab = pd.crosstab(cells["sample"], cells["group"])
    cube = from_mudata(views, n_cells=crosstab)

    # Feature ids without LIANA's "<view>:" prefix; groups sorted, as from_cells
    # sorts them; the condition of each sample from the MuData's obs.
    expected = _pooled(cells, sample_table=_metadata(cells, "sample", "condition"))
    assert cube.fingerprint() == expected.fingerprint()
    assert cube.provenance["construction"]["layout"] == "liana.multi.adata_to_views"
    # Without fractions, counts alone come without cell numbers... and so cannot
    # mark the entry without cells.
    with pytest.raises(ValueError, match="no cell numbers"):
        from_mudata(views, layers={"counts": "X"})


def test_an_imported_cube_feeds_a_design_and_limma_voom(
    ad: Any, cells: dict[str, Any]
) -> None:
    imported = from_anndata(
        ad.read_h5ad(DATA / "decoupler.h5ad"), sample="sample", group="group"
    )
    pooled = _pooled(
        cells,
        sample_table=_metadata(cells, "sample", "condition"),
        group_table=_metadata(cells, "group", "lineage"),
    )

    def analysis(cube: Cube) -> Any:
        # The condition of each sample came from decoupler's obs.
        design = Design(
            cube.samples,
            "~ condition",
            levels={"condition": ["ctrl", "stim"]},
            contrasts={"stim": [0.0, 1.0]},
        )
        return limma_voom(cube, design, min_cells=1)

    ours, reference = analysis(imported), analysis(pooled)
    assert ours.tables.keys() == reference.tables.keys()
    assert sum(len(table) for table in ours.tables.values()) > 0  # genes were tested
    for key, table in reference.tables.items():
        assert ours.tables[key].equals(table)
    assert ours.provenance["cube"]["fingerprint"] == imported.fingerprint()


def test_means_variances_and_medians_are_refused(ad: Any) -> None:
    means = ad.read_h5ad(DATA / "scanpy_means.h5ad")
    with pytest.raises(
        ValueError,
        match=r"no layer 'sum'; its layers are \['mean', 'var'\], which hold means and variances",
    ):
        from_anndata(means, sample="sample", group="group")
    for source, statistic in (("mean", "means"), ("var", "variances")):
        with pytest.raises(
            ValueError, match=f"fill with {statistic}. A cube holds sums"
        ):
            from_anndata(
                means, sample="sample", group="group", layers={"counts": source}
            )
    with pytest.raises(ValueError, match="'median', which .* fill with medians"):
        from_anndata(means, sample="sample", group="group", layers={"counts": "median"})


def test_decoupler_means_are_refused_because_x_adds_up_to_totals_over_cells(
    ad: Any,
) -> None:
    means = ad.read_h5ad(DATA / "decoupler_means.h5ad")

    # decoupler's psbulk_counts are the totals of the pooled counts whatever its
    # mode; X adds up to them divided by the cells of each row.
    message = r"X holds means: at row 0 \('S\d_\w+'\) it adds up to [\d.]+, the"
    with pytest.raises(ValueError, match=message):
        from_anndata(means, sample="sample", group="group")
    # Whatever X is read as.
    with pytest.raises(ValueError, match="X holds means"):
        from_anndata(means, sample="sample", group="group", layers={"counts": "X"})


def _medians(cells: dict[str, Any], samples: Any, groups: Any) -> np.ndarray:
    """The median count of each feature over the cells of each sample and group, as
    decoupler's mode='median' gives it; zeros for an entry without cells."""
    counts = cells["counts"].toarray()
    rows = []
    for sample, group in zip(samples, groups, strict=True):
        chosen = (cells["sample"] == str(sample)) & (cells["group"] == str(group))
        rows.append(
            np.median(counts[chosen], axis=0) if chosen.any() else counts[0] * 0
        )
    return np.array(rows)


def test_medians_are_refused_where_a_cell_detects_a_feature_and_x_is_zero(
    ad: Any, md: Any, pd: Any, cells: dict[str, Any]
) -> None:
    pseudobulk = ad.read_h5ad(DATA / "decoupler.h5ad")
    pseudobulk.X = _medians(cells, pseudobulk.obs["sample"], pseudobulk.obs["group"])
    views = md.read_h5mu(DATA / "liana.h5mu")
    for name, view in views.mod.items():
        view.X = _medians(cells, view.obs_names, [name] * view.n_obs)

    message = "X does not hold sums of the counts of its cells: at .* psbulk_props says"
    with pytest.raises(ValueError, match=message):
        from_anndata(pseudobulk, sample="sample", group="group")
    crosstab = pd.crosstab(cells["sample"], cells["group"])
    with pytest.raises(ValueError, match=message):
        from_mudata(views, n_cells=crosstab)


def test_decoupler_pseudobulk_of_fewer_features_still_imports(
    ad: Any, cells: dict[str, Any]
) -> None:
    pseudobulk = ad.read_h5ad(DATA / "decoupler.h5ad")
    kept = list(pseudobulk.var_names[::3])

    # As decoupler.pp.filter_by_expr leaves it: fewer features, and psbulk_counts
    # still the totals over all of them.
    cube = from_anndata(pseudobulk[:, kept], sample="sample", group="group")

    expected = _pooled(
        cells,
        sample_table=_metadata(cells, "sample", "condition"),
        group_table=_metadata(cells, "group", "lineage"),
    )
    assert cube.fingerprint() == expected.select(features=kept).fingerprint()


def test_x_that_adds_up_to_more_than_its_counts_is_refused(ad: Any) -> None:
    pseudobulk = ad.read_h5ad(DATA / "decoupler.h5ad")
    pseudobulk.X = pseudobulk.X * 1000.0  # whole numbers, zero where the counts are

    message = (
        r"X adds up to [\d.e+]+ at row 0 \('S1_\w+'\), more than the [\d.]+ counts"
    )
    with pytest.raises(ValueError, match=message):
        from_anndata(pseudobulk, sample="sample", group="group")


# Long AnnData made by hand.


def _long(
    ad: Any,
    pd: Any,
    rows: list[tuple[str, str]],
    values: Any,
    **obs: Any,
) -> Any:
    frame = pd.DataFrame(
        {"donor": [r[0] for r in rows], "cell_type": [r[1] for r in rows], **obs},
        index=[f"r{i}" for i in range(len(rows))],
    )
    return ad.AnnData(
        obs=frame,
        var=pd.DataFrame(index=[f"G{j}" for j in range(np.shape(values)[1])]),
        layers={"sum": np.asarray(values)},
    )


ROWS = [("D1", "B"), ("D2", "B"), ("D1", "T"), ("D2", "T")]
SUMS = np.array([[4.0, 0.0], [3.0, 1.0], [0.0, 2.0], [5.0, 5.0]])


def test_given_layers_cell_numbers_and_tables(ad: Any, pd: Any) -> None:
    adata = _long(ad, pd, ROWS, SUMS, n=[2, 1, 3, 4], batch=["x", "y", "x", "y"])

    cube = from_anndata(
        adata,
        sample="donor",
        group="cell_type",
        layers={"counts": "sum"},
        n_cells="n",
        sample_table={"sample_id": ["D2", "D1", "D3"], "age": [30, 40, 50]},
        source={"study": "hand-made"},
    )

    # The rows of sample_table set the order; D3 has no cells.
    assert cube.samples["sample_id"].tolist() == ["D2", "D1", "D3"]
    assert_array_equal(cube.n_cells, [[1, 4], [2, 3], [0, 0]])
    assert_array_equal(cube.layer("counts", "B"), [[3, 1], [4, 0], [0, 0]])
    # batch is constant within every sample: the group table cannot take it.
    assert cube.groups.columns == ("group_id",)
    construction = cube.provenance["construction"]
    assert cube.provenance["source"] == {"study": "hand-made"}
    assert construction == {
        "function": "from_anndata",
        "cytepool_version": cytepool.__version__,
        "layout": "given",
        "rows": 4,
        "sample_column": "donor",
        "group_column": "cell_type",
        "layers": {"counts": "sum"},
        "n_cells": "n",
        "samples": {"order": "sample_table", "labels": "text"},
        "groups": {"order": "sorted_labels", "labels": "text"},
        "obs_columns": {"samples": [], "groups": [], "left_out": ["batch"]},
    }


def test_obs_columns_go_to_the_table_they_are_constant_in(ad: Any, pd: Any) -> None:
    adata = _long(
        ad,
        pd,
        ROWS,
        SUMS,
        n=[2, 1, 3, 4],
        sex=pd.Categorical(["F", "M", "F", "M"]),
        lineage=["lymphoid B", "lymphoid B", "lymphoid T", "lymphoid T"],
        quality=[0.1, 0.2, 0.3, 0.4],
    )

    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers={"counts": "sum"}, n_cells="n"
    )

    assert cube.samples.equals(
        cube.samples.__class__({"sample_id": ["D1", "D2"], "sex": ["F", "M"]})
    )
    assert cube.groups["lineage"].tolist() == ["lymphoid B", "lymphoid T"]
    assert cube.provenance["construction"]["obs_columns"]["left_out"] == ["quality"]


def test_inference_keeps_the_ids_and_the_types_a_table_holds(ad: Any, pd: Any) -> None:
    adata = _long(
        ad,
        pd,
        ROWS,
        SUMS,
        n=[2, 1, 3, 4],
        sample_id=["L7", "L9", "L7", "L9"],  # constant within every donor
        parent=[1, 1, 2, 2],  # constant within every cell type, but not text
        cell_ontology_id=["CL:1", "CL:1", "CL:2", "CL:2"],
    )

    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers={"counts": "sum"}, n_cells="n"
    )

    assert cube.samples["sample_id"].tolist() == ["D1", "D2"]
    assert cube.samples.columns == ("sample_id",)
    assert cube.groups.columns == ("group_id", "cell_ontology_id")
    assert cube.provenance["construction"]["obs_columns"]["left_out"] == [
        "sample_id",
        "parent",
    ]


def test_feature_columns_that_a_cube_holds_otherwise_are_refused(
    ad: Any, pd: Any
) -> None:
    adata = _long(ad, pd, ROWS, SUMS)
    adata.var["symbol"] = [1, 2]
    with pytest.raises(TypeError, match="'symbol' of var has dtype int64, but a cube"):
        from_anndata(adata, sample="donor", group="cell_type", layers={"counts": "sum"})

    adata.var["symbol"] = ["CD3E", "MS4A1"]
    adata.var["seen"] = pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC")
    with pytest.raises(TypeError) as raised:
        from_anndata(adata, sample="donor", group="cell_type", layers={"counts": "sum"})
    assert "drop the column to read the data without it" in raised.value.__notes__[0]


def test_integer_labels_become_text_in_numeric_order(ad: Any, pd: Any) -> None:
    rows = [(10, "B"), (2, "B"), (1, "B")]
    adata = _long(ad, pd, rows, SUMS[:3])
    crosstab = pd.crosstab(pd.Series([10, 2, 1, 1]), pd.Series(["B"] * 4))

    cube = from_anndata(
        adata,
        sample="donor",
        group="cell_type",
        layers={"counts": "sum"},
        n_cells=crosstab,
    )

    assert cube.samples["sample_id"].tolist() == ["1", "2", "10"]
    assert_array_equal(cube.n_cells, [[2], [1], [1]])
    assert cube.provenance["construction"]["samples"]["labels"] == "integers"


def test_counts_that_a_layout_gives_must_be_whole_numbers(ad: Any, pd: Any) -> None:
    sums = SUMS.copy()
    sums[2, 1] = 2.5
    adata = _long(ad, pd, ROWS, sums, n_obs_aggregated=[2, 1, 3, 4])

    with pytest.raises(ValueError, match=r"holds 2.5 at row 2 \('r2'\), feature 'G1'"):
        from_anndata(adata, sample="donor", group="cell_type")
    # Counts that are not whole numbers, said so.
    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers={"counts": "sum"}
    )
    assert cube.layer("counts").dtype == np.float64
    assert cube.layer("counts", "T")[0, 1] == 2.5


def test_whole_counts_become_the_narrowest_integers_that_hold_them(
    ad: Any, pd: Any
) -> None:
    sums = SUMS.copy()
    large = _long(ad, pd, ROWS, sums * 2**32, n_obs_aggregated=[2, 1, 3, 4])
    small = _long(ad, pd, ROWS, sums, n_obs_aggregated=[2, 1, 3, 4])

    assert (
        from_anndata(small, sample="donor", group="cell_type").layer("counts").dtype
        == np.uint32
    )
    assert (
        from_anndata(large, sample="donor", group="cell_type").layer("counts").dtype
        == np.uint64
    )


def test_whole_floats_beyond_2_to_the_53_are_not_counts_of_a_layout(
    ad: Any, pd: Any
) -> None:
    sums = SUMS.copy()
    sums[3, 0] = 2.0**60
    adata = _long(ad, pd, ROWS, sums, n_obs_aggregated=[2, 1, 3, 4])

    with pytest.raises(
        ValueError, match=r"feature 'G0', at or above 2\*\*53, beyond which float64"
    ):
        from_anndata(adata, sample="donor", group="cell_type")
    # Said to be counts, they are float64 counts.
    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers={"counts": "sum"}
    )
    assert cube.layer("counts").dtype == np.float64


def test_counts_must_sum_over_the_cells_that_detected_counts(ad: Any, pd: Any) -> None:
    adata = _long(ad, pd, ROWS, SUMS, n_obs_aggregated=[2, 1, 3, 4])
    adata.layers["count_nonzero"] = np.array([[2, 0], [1, 1], [0, 2], [3, 3]])
    cube = from_anndata(adata, sample="donor", group="cell_type")
    assert_array_equal(cube.layer("detected", "T"), [[0, 2], [3, 3]])

    # A cell detects the feature, but the sum is 0: the sums are not of these cells.
    adata.layers["count_nonzero"][0, 1] = 1
    with pytest.raises(
        ValueError,
        match="at group 'B', sample 'D1', feature 'G1' the counts are 0 and 1 cells",
    ):
        from_anndata(adata, sample="donor", group="cell_type")
    # 3 cells with a positive count sum to 2: not a sum of whole counts.
    adata.layers["count_nonzero"][0, 1] = 0
    adata.layers["count_nonzero"][2, 1] = 3
    with pytest.raises(ValueError, match="the counts are 2 and 3 cells detect"):
        from_anndata(adata, sample="donor", group="cell_type")
    # Counts that are not whole numbers can sum to less than their cells.
    adata.layers["sum"] = SUMS / 4.0
    layers = {"counts": "sum", "detected": "count_nonzero"}
    from_anndata(adata, sample="donor", group="cell_type", layers=layers)


def test_whole_sums_of_counts_that_are_not_whole_are_read_when_said_so(
    ad: Any, pd: Any
) -> None:
    # Two cells with counts 0.5 sum to 1: a sum of fractional counts is no bound on
    # the number of cells detecting the feature.
    cube = from_cells(
        np.array([[0.5, 1.0], [0.5, 0.0], [1.5, 2.0], [0.5, 1.0]]),
        samples=["a", "a", "b", "b"],
        groups=["g"] * 4,
        layers=["counts", "detected"],
    )
    adata = to_anndata(cube)
    del adata.uns["cytepool"]  # as anndata.concat leaves it
    adata.layers["counts"] = adata.layers["counts"].astype(np.float64)
    layers = {"counts": "counts", "detected": "detected"}

    back = from_anndata(
        adata, sample="sample_id", group="group_id", layers=layers, n_cells="n_cells"
    )

    assert_array_equal(back.layer("counts", "g"), [[1, 1], [2, 3]])
    assert_array_equal(back.layer("detected", "g"), [[2, 1], [2, 2]])


def test_fractions_are_read_to_the_precision_they_are_stored_in(
    ad: Any, pd: Any
) -> None:
    detected = np.array([[7, 0], [19, 1], [0, 2], [5, 5]])
    adata = _long(ad, pd, ROWS, detected * 2, n=[19, 19, 19, 19])
    adata.layers["psbulk_props"] = (detected / 19).astype(np.float32)
    layers = {"counts": "sum", "detected": "psbulk_props"}

    # 7 / 19 in float32 is 0.36842105, which times 19 is 7.0000001.
    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers=layers, n_cells="n"
    )
    assert_array_equal(cube.layer("detected", "B"), detected[:2])
    # float16 holds too few digits for the cells of a row of 1,000 cells.
    adata.obs["n"] = 1000
    adata.layers["psbulk_props"] = (detected / 1000).astype(np.float16)
    with pytest.raises(
        ValueError, match=r"fractions in float16, too few digits .* 1000 cells of row 0"
    ):
        from_anndata(
            adata, sample="donor", group="cell_type", layers=layers, n_cells="n"
        )


def test_entries_without_rows_need_cell_numbers(ad: Any, pd: Any) -> None:
    adata = _long(ad, pd, ROWS[:3], SUMS[:3])

    with pytest.raises(ValueError, match="sample 'D2' has no row in group 'T', and"):
        from_anndata(adata, sample="donor", group="cell_type", layers={"counts": "sum"})
    crosstab = pd.DataFrame({"B": [1, 1], "T": [2, 0]}, index=["D1", "D2"])
    cube = from_anndata(
        adata,
        sample="donor",
        group="cell_type",
        layers={"counts": "sum"},
        n_cells=crosstab,
    )
    assert_array_equal(cube.n_cells, [[1, 2], [1, 0]])
    # A table that gives cells to an entry without a row: its profile is missing.
    crosstab.loc["D2", "T"] = 4
    with pytest.raises(
        ValueError, match="gives 4 cells to sample 'D2' in group 'T', which has no row"
    ):
        from_anndata(
            adata,
            sample="donor",
            group="cell_type",
            layers={"counts": "sum"},
            n_cells=crosstab,
        )


def test_a_table_of_cell_numbers_names_every_sample_and_group(ad: Any, pd: Any) -> None:
    adata = _long(ad, pd, ROWS, SUMS)
    crosstab = pd.DataFrame({"B": [1, 1], "T": [2, 0]}, index=["D1", "D9"])

    message = (
        r"n_cells must have the sample ids as its rows, each once; it lacks \['D2'\]"
    )
    with pytest.raises(ValueError, match=message + r" and has \['D9'\] besides"):
        from_anndata(
            adata,
            sample="donor",
            group="cell_type",
            layers={"counts": "sum"},
            n_cells=crosstab,
        )


def test_without_cell_numbers_every_entry_needs_a_row(ad: Any, pd: Any) -> None:
    cube = from_anndata(
        _long(ad, pd, ROWS, SUMS),
        sample="donor",
        group="cell_type",
        layers={"counts": "sum"},
    )

    assert cube.n_cells is None
    assert_array_equal(cube.layer("counts", "T"), SUMS[2:])


def test_fractions_give_detected_cells_only_when_they_are_whole_numbers(
    ad: Any, pd: Any
) -> None:
    adata = _long(ad, pd, ROWS, SUMS, n=[2, 1, 3, 4])
    adata.layers["psbulk_props"] = np.array(
        [[0.5, 0.0], [1.0, 1.0], [0.0, 2 / 3], [0.75, 0.25]]
    )
    layers = {"counts": "sum", "detected": "psbulk_props"}

    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers=layers, n_cells="n"
    )
    assert_array_equal(cube.layer("detected", "T"), [[0, 2], [3, 1]])

    adata.layers["psbulk_props"][3, 1] = 0.3  # 0.3 of 4 cells is 1.2 cells
    with pytest.raises(
        ValueError, match=r"fraction 0.3 of the 4 cells of row 3 \('r3'\)"
    ):
        from_anndata(
            adata, sample="donor", group="cell_type", layers=layers, n_cells="n"
        )
    with pytest.raises(
        ValueError, match="give 'detected' only with the number of cells"
    ):
        from_anndata(adata, sample="donor", group="cell_type", layers=layers)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"sample": None}, ValueError, "name the obs columns that hold the sample"),
        (
            {"group": "donor"},
            ValueError,
            "sample and group both name the obs column 'donor'",
        ),
        (
            {"group": "lineage"},
            KeyError,
            "group names 'lineage', which is not a column",
        ),
        ({"sample": 3}, TypeError, "sample must name an obs column"),
        ({"layers": {"count": "sum"}}, ValueError, "unknown cube layer 'count'"),
        (
            {"layers": {"detected": "sum"}},
            ValueError,
            "must give the source of 'counts'",
        ),
        (
            {"layers": {"counts": "total"}},
            KeyError,
            "has no layer 'total'; its layers are: sum",
        ),
        ({"layers": {"counts": "X"}}, KeyError, "X of the AnnData is empty"),
        (
            {"layers": {"counts": 1}},
            TypeError,
            r"layers\['counts'\] must name X or a layer",
        ),
        ({"layers": ["counts"]}, TypeError, "layers must map cube layer names"),
        (
            {"layers": {"counts": "sum", "log_sum": "count_nonzero"}},
            ValueError,
            "'count_nonzero', which counts cells",
        ),
        ({"n_cells": np.ones((2, 2))}, TypeError, "n_cells must name an obs column"),
        ({"n_cells": "cells"}, KeyError, "n_cells names 'cells'"),
        (
            {"layers": {"counts": "sum"}},
            ValueError,
            "say which source holds each cube layer",
        ),
    ],
)
def test_invalid_arguments_are_refused(
    change: dict[str, Any], error: type, message: str, ad: Any, pd: Any
) -> None:
    adata = _long(ad, pd, ROWS, SUMS)
    arguments: dict[str, Any] = {
        "sample": "donor",
        "group": "cell_type",
        "layers": {"counts": "sum"},
    }
    arguments.update(change)
    if message.startswith("say which"):
        del arguments["layers"]

    with pytest.raises(error, match=message):
        from_anndata(adata, **arguments)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            np.where(SUMS == 5.0, -1.0, SUMS),
            r"negative value \(-1.0\) at row 3 \('r3'\), feature 'G0'",
        ),
        (np.where(SUMS == 5.0, np.nan, SUMS), r"non-finite value \(nan\) at row 3"),
    ],
)
def test_values_that_cannot_be_counts_are_refused(
    values: np.ndarray, message: str, ad: Any, pd: Any
) -> None:
    with pytest.raises(ValueError, match=message):
        from_anndata(
            _long(ad, pd, ROWS, values),
            sample="donor",
            group="cell_type",
            layers={"counts": "sum"},
        )


def test_rows_must_name_each_entry_once_with_its_labels(ad: Any, pd: Any) -> None:
    repeated = _long(ad, pd, [*ROWS, ("D1", "T")], np.vstack([SUMS, SUMS[:1]]))
    with pytest.raises(
        ValueError,
        match=r"row 2 \('r2'\) and row 4 \('r4'\) are both sample 'D1' in group 'T'",
    ):
        from_anndata(
            repeated, sample="donor", group="cell_type", layers={"counts": "sum"}
        )
    missing = _long(ad, pd, ROWS, SUMS)
    missing.obs["donor"] = ["D1", None, "D1", "D2"]
    with pytest.raises(
        ValueError, match=r"obs\['donor'\] has a missing label at row 1"
    ):
        from_anndata(
            missing, sample="donor", group="cell_type", layers={"counts": "sum"}
        )


def test_sparse_layers_are_read(ad: Any, pd: Any) -> None:
    sparse = pytest.importorskip("scipy.sparse")
    adata = _long(ad, pd, ROWS, SUMS)
    adata.layers["sum"] = sparse.csr_matrix(SUMS)

    cube = from_anndata(
        adata, sample="donor", group="cell_type", layers={"counts": "sum"}
    )

    assert_array_equal(cube.layer("counts", "B"), SUMS[:2])


# Reading back an export that was changed, and the record.


def test_a_subset_of_features_reads_as_a_selection_of_the_cube(ad: Any) -> None:
    cube = varied_cube()
    adata = to_anndata(cube)

    back = from_anndata(adata[:, ["g4", "g1"]])

    assert back.fingerprint() == cube.select(features=["g4", "g1"]).fingerprint()
    assert back.provenance["operations"][-1]["input"] == cube.fingerprint()


def test_removed_rows_read_as_entries_without_cells(ad: Any) -> None:
    cube = varied_cube()
    adata = to_anndata(cube)
    kept = ~(
        (adata.obs["sample_id"] == "S2") & (adata.obs["group_id"] == "NK")
    ).to_numpy()

    back = from_anndata(adata[kept])

    layers = {name: cube.layer(name).copy() for name in LAYERS}
    for values in layers.values():
        values[2, 1] = 0  # group NK, sample S2
    n_cells = cube.n_cells.copy()  # type: ignore[union-attr]
    n_cells[1, 2] = 0
    expected = Cube(
        layers,
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
        normalization=cube.normalization,
    )
    assert back.fingerprint() == expected.fingerprint()


def test_an_exported_layer_keeps_its_type_or_is_refused_when_changed(ad: Any) -> None:
    cube = varied_cube()
    adata = to_anndata(cube)
    # Converted to other types, with the same values: converted back.
    adata.layers["counts"] = adata.layers["counts"].astype(np.float32)
    adata.layers["detected"] = adata.layers["detected"].astype(np.int64)
    assert from_anndata(adata).fingerprint() == cube.fingerprint()

    adata.layers["counts"] = adata.layers["counts"] / 3  # normalized in place, say
    message = (
        r"the layer 'counts' holds [\d.]+ at row 0 \('S1_B'\), feature 'g\d', which"
    )
    with pytest.raises(ValueError, match=message + " its type when CytePool exported"):
        from_anndata(adata)


def test_the_record_restores_every_argument_unless_one_is_given(ad: Any) -> None:
    cube = varied_cube()
    adata = to_anndata(cube)

    back = from_anndata(
        adata, layers={"counts": "counts"}, normalization={"method": "none"}
    )

    assert back.layer_names == ("counts",)
    assert back.normalization == {"method": "none"}
    assert back.samples.equals(cube.samples)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format_version": 2}, "has format version 2, newer than the version 1"),
        (
            {"format": "cytepool.cube.mudata"},
            r"to_mudata\(\) exported; read the MuData",
        ),
        ({"format": "other"}, "its format is 'other'"),
        (
            {"layers": "[1"},
            "is not a record of a cube that this version of CytePool reads",
        ),
        (
            {
                "samples": '{"columns": [{"name": "sample_id", "type": "int128", "values": []}]}'
            },
            "not a record",
        ),
        ({"n_cells": "1"}, "n_cells is not true or false"),
    ],
)
def test_a_record_that_cannot_be_read_is_reported(
    change: dict[str, Any], message: str, ad: Any
) -> None:
    adata = to_anndata(CUBES["bulk"])
    adata.uns["cytepool"] = {**adata.uns["cytepool"], **change}

    with pytest.raises(ValueError, match=message):
        from_anndata(adata)


def test_a_record_table_that_lacks_a_label_says_so(ad: Any) -> None:
    adata = to_anndata(CUBES["bulk"])
    adata.obs["sample_id"] = adata.obs["sample_id"].cat.rename_categories({"z": "w"})

    with pytest.raises(ValueError, match="'w' .* is not in sample_table") as raised:
        from_anndata(adata)
    assert "CytePool recorded in uns['cytepool']" in str(raised.value.__notes__)


# MuData made by hand.


def _views(md: Any, ad: Any, pd: Any, prefix: bool = True) -> Any:
    def view(name: str, samples: list[str], values: np.ndarray) -> Any:
        names = [f"{name}:G{j}" if prefix else f"G{j}" for j in range(2)]
        return ad.AnnData(
            obs=pd.DataFrame(index=samples),
            var=pd.DataFrame({"symbol": ["a", "b"]}, index=names),
            layers={"sum": values},
        )

    return md.MuData(
        {
            "T": view("T", ["D2", "D1"], SUMS[2:][::-1]),
            "B": view("B", ["D1", "D2"], SUMS[:2]),
        }
    )


# Without LIANA's prefixes, mudata finds the same var_names in both modalities.
@pytest.mark.filterwarnings("ignore:var_names are not unique")
def test_modalities_are_groups_and_their_rows_samples(
    md: Any, ad: Any, pd: Any
) -> None:
    for prefix in (True, False):
        cube = from_mudata(_views(md, ad, pd, prefix), layers={"counts": "sum"})

        assert cube.groups["group_id"].tolist() == ["B", "T"]
        assert cube.features["feature_id"].tolist() == ["G0", "G1"]
        assert_array_equal(cube.layer("counts", "T"), SUMS[2:])


def test_modalities_must_hold_the_same_features(md: Any, ad: Any, pd: Any) -> None:
    mdata = _views(md, ad, pd)
    mdata.mod["T"].var_names = ["T:G0", "T:G9"]

    with pytest.raises(
        ValueError,
        match=r"'B' lacks 1 of the features of 'T' \('G9'\) and has 1 others",
    ):
        from_mudata(mdata, layers={"counts": "sum"})
    # In another order they are the same features, in the order of the first
    # modality, T.
    reordered = _views(md, ad, pd)
    reordered.mod["T"] = reordered.mod["T"][:, ["T:G1", "T:G0"]]
    cube = from_mudata(reordered, layers={"counts": "sum"})
    assert cube.features["feature_id"].tolist() == ["G1", "G0"]
    assert_array_equal(cube.layer("counts", "T"), SUMS[2:, ::-1])
    assert_array_equal(cube.layer("counts", "B"), SUMS[:2, ::-1])


def test_a_modality_without_rows_is_a_group_without_cells(
    md: Any, ad: Any, pd: Any
) -> None:
    mdata = _views(md, ad, pd)
    empty = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], dtype=object)),
        var=mdata.mod["B"].var.rename(index=lambda name: name.replace("B:", "NK:")),
        layers={"sum": np.zeros((0, 2))},
    )
    with_empty = md.MuData({**dict(mdata.mod), "NK": empty})
    crosstab = pd.DataFrame(
        {"B": [1, 1], "NK": [0, 0], "T": [2, 2]}, index=["D1", "D2"]
    )

    cube = from_mudata(with_empty, layers={"counts": "sum"}, n_cells=crosstab)

    assert cube.groups["group_id"].tolist() == ["B", "NK", "T"]
    assert_array_equal(cube.n_cells, [[1, 0, 2], [1, 0, 2]])


def test_liana_statistics_give_the_cell_numbers_when_they_hold_them(
    md: Any, pd: Any, cells: dict[str, Any]
) -> None:
    views = md.read_h5mu(DATA / "liana.h5mu")
    crosstab = pd.crosstab(cells["sample"], cells["group"])
    stats = views.uns["psbulk_stats"].copy()
    for name in views.mod:  # as LIANA keeps them, with decoupler 1's column name
        stats[f"{name}:psbulk_n_cells"] = crosstab[name].reindex(stats.index)
    views.uns["psbulk_stats"] = stats

    cube = from_mudata(views)

    expected = _pooled(cells, sample_table=_metadata(cells, "sample", "condition"))
    assert cube.fingerprint() == expected.fingerprint()
    assert cube.provenance["construction"]["n_cells"] == "uns['psbulk_stats']"
    first = next(iter(views.mod))
    stats[f"{first}:psbulk_counts"] *= 0.5
    with pytest.raises(ValueError, match=r"X adds up to .* more than the"):
        from_mudata(views)


def test_views_of_maxima_are_refused_also_without_cell_numbers(
    md: Any, ad: Any, pd: Any
) -> None:
    mdata = _views(md, ad, pd)
    # 4 cells: the first feature is detected by 1 cell and the second by 3.
    for view in mdata.mod.values():
        view.layers["psbulk_props"] = np.tile([0.25, 0.75], (view.n_obs, 1))
        view.X = np.tile([2.0, 5.0], (view.n_obs, 1))  # sums: at least 1 and 3
    assert from_mudata(mdata).layer_names == ("counts",)
    # A maximum of 2 over the 3 cells detecting the second feature: no sum of
    # whole counts, which is at least 3.
    mdata.mod["T"].X[1, 1] = 2.0
    message = r"X holds 2.0, but at least 3 of its cells detect the feature"
    with pytest.raises(ValueError, match=message):
        from_mudata(mdata)


def test_views_of_medians_are_refused_also_without_cell_numbers(
    md: Any, ad: Any, pd: Any
) -> None:
    mdata = _views(md, ad, pd)
    for view in mdata.mod.values():
        view.layers["psbulk_props"] = (view.layers["sum"] > 0) * 0.5
        view.X = view.layers["sum"].copy()
    # Sums, positive where a cell detects the feature: read.
    assert from_mudata(mdata).layer_names == ("counts",)
    mdata.mod["T"].X[0, 0] = 0.0  # a median of 0 where a cell detects the feature
    with pytest.raises(ValueError, match="X does not hold sums of the counts of its"):
        from_mudata(mdata)


def test_containers_and_cubes_are_checked(ad: Any, md: Any) -> None:
    mdata = to_mudata(CUBES["bulk"])
    adata = to_anndata(CUBES["bulk"])

    with pytest.raises(TypeError, match=r"read a MuData with from_mudata\(\)"):
        from_anndata(mdata)
    with pytest.raises(TypeError, match=r"read an AnnData with from_anndata\(\)"):
        from_mudata(adata)
    with pytest.raises(
        TypeError, match=r"from_anndata\(\) reads an anndata.AnnData, got dict"
    ):
        from_anndata({})
    with pytest.raises(TypeError, match=r"to_anndata\(\) converts a Cube, got AnnData"):
        to_anndata(adata)
    without_groups = CUBES["bulk"].select(groups=[])
    with pytest.raises(ValueError, match="this cube has no groups"):
        to_mudata(without_groups)
    assert (
        from_anndata(to_anndata(without_groups)).fingerprint()
        == without_groups.fingerprint()
    )


def test_a_missing_optional_package_is_named_with_its_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cube = CUBES["bulk"]
    monkeypatch.setitem(sys.modules, "anndata", None)
    with pytest.raises(
        ImportError, match=r"to_anndata\(\) needs anndata.*'anndata' extra"
    ):
        to_anndata(cube)
    monkeypatch.setitem(sys.modules, "mudata", None)
    with pytest.raises(ImportError, match=r"to_mudata\(\) needs .*'mudata' extra"):
        to_mudata(cube)


def test_to_mudata_names_the_mudata_extra(
    ad: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mudata", None)

    with pytest.raises(
        ImportError, match=r"to_mudata\(\) needs mudata, .*'mudata' extra"
    ):
        to_mudata(CUBES["bulk"])


# How the provenance reads.


def test_the_history_describes_imports_in_plain_terms() -> None:
    construction = {
        "function": "from_anndata",
        "cytepool_version": "1.0",
        "layout": "decoupler.pp.pseudobulk",
        "rows": 12,
        "sample_column": "sample",
        "group_column": "group",
        "layers": {"counts": "X", "detected": "psbulk_props"},
        "n_cells": "psbulk_cells",
        "samples": {"order": "sorted_labels", "labels": "text"},
        "groups": {"order": "group_table", "labels": "integers"},
        "obs_columns": None,
    }
    imported = {"operation": "from_mudata", "input": "sha256:ab", "source": None}

    text = describe_provenance({"construction": construction, "operations": [imported]})

    assert text.splitlines() == [
        "Built by from_anndata (CytePool 1.0) from 12 profiles of an AnnData, in the "
        "layout of decoupler.pp.pseudobulk:",
        "  samples: the sorted distinct labels",
        "  groups: the rows of group_table; the labels were integers, written as "
        "decimal text",
        "  layers: counts from 'X', detected from 'psbulk_props'",
        '  sample_column: "sample"',
        '  group_column: "group"',
        '  n_cells: "psbulk_cells"',
        "  obs_columns: null",
        "1. from_mudata, reading a MuData that CytePool exported from the cube "
        "sha256:ab",
        "References: CytePool resolves none on its own. The source record's "
        "references need the tool that wrote them. A fingerprint names a cube "
        "without saying where it is: Cube.fingerprint() of a cube at hand says "
        "whether it is that cube.",
    ]
