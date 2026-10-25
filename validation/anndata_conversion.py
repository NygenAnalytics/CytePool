"""Check the conversion of cubes to and from AnnData and MuData against scverse tools.

The script simulates cells (18,914 cells x 1,500 genes, 8 samples x 10 cell types,
with an empty sample x cell type entry, entries with one and two cells, and genes
never detected), with a condition per sample and a lineage per cell type, pools them
with ``from_cells``, and checks:

- imports: ``from_anndata`` of the output of ``scanpy.get.aggregate`` (sums and
  counts of nonzero values of the counts; sums of normalized values, of their
  logarithms and of the squares of those) and of ``decoupler.pp.pseudobulk``, also
  after ``decoupler.pp.filter_by_expr``, and ``from_mudata`` of the views of
  ``liana.multi.adata_to_views``, each compared with the cube ``from_cells`` builds
  from the same cells;
- refusals: scanpy's means and variances, decoupler with ``mode="mean"`` and
  ``mode="median"``, LIANA's views of medians, and LIANA's views without cell
  numbers;
- layouts: ``to_anndata`` gives the rows and sums of ``scanpy.get.aggregate``, and
  ``to_mudata`` the modalities of LIANA; scanpy aggregates the exported AnnData by
  condition as ``Cube.collapse_samples`` pools it;
- round trips: ``to_anndata`` and ``to_mudata`` of four cubes (all five layers;
  merged, pooled and selected, with their provenance; bulk; tables of every column
  type, with columns without any value, names that anndata does not write as they
  are, and a group and a sample without cells), read back in memory, from
  ``.h5ad``, ``.zarr`` and ``.h5mu`` files, and from copies that anndata and mudata
  wrote in another environment (pandas 2), keep the fingerprint.

The tools run as separate programs that exchange files with this script in a
temporary directory.

Requirements:
    This script: numpy, scipy, pandas, anndata, mudata, and cytepool from the source
    tree (numba).
    scanpy reference: scanpy and anndata, in SCANPY_PYTHON (default: this
    interpreter). decoupler reference: decoupler and anndata, in DECOUPLER_PYTHON.
    LIANA reference: liana (with decoupler, mudata and pandas 2), in LIANA_PYTHON.

Usage, from the repository root:
    PYTHONPATH=. python validation/anndata_conversion.py
    PYTHONPATH=. python validation/anndata_conversion.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import anndata
import mudata
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cytepool.cube import Cube, from_bulk, from_cells
from cytepool.io.anndata import from_anndata, from_mudata, to_anndata, to_mudata

SEED = 20261025
EXACT = 1e-10
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "anndata_conversion"
)
# Cell types and their lineages, alternating, so that the first few hold both.
GROUPS = {
    "B": "lymphoid",
    "Mono_14": "myeloid",
    "CD4_T": "lymphoid",
    "Mono_16": "myeloid",
    "CD8_T": "lymphoid",
    "cDC": "myeloid",
    "NK": "lymphoid",
    "pDC": "myeloid",
    "MAIT": "lymphoid",
    "Plasma": "lymphoid",
}

PROGRAM = textwrap.dedent(
    '''
    """Run a scverse tool on the simulated cells and save what it returns."""

    import json
    import sys
    import warnings
    from importlib.metadata import version

    import anndata
    import numpy as np

    warnings.filterwarnings("ignore")
    tool, directory = sys.argv[1], sys.argv[2]
    cells = anndata.read_h5ad(f"{directory}/cells.h5ad")
    by = ["sample", "group"]
    if tool == "scanpy":
        import scanpy as sc

        aggregate = sc.get.aggregate
        aggregate(cells, by=by, func=["sum", "count_nonzero"]).write_h5ad(
            f"{directory}/scanpy_counts.h5ad"
        )
        aggregate(cells, by=by, func=["mean", "var"]).write_h5ad(
            f"{directory}/scanpy_means.h5ad"
        )
        cells.X = cells.X.astype(np.float64)  # normalized in float64, as CytePool does
        sc.pp.normalize_total(cells, target_sum=1e4)
        cells.layers["linear"] = cells.X.copy()
        sc.pp.log1p(cells)
        cells.layers["squares"] = cells.X.power(2)
        for name, layer in (("log", None), ("squares", "squares"), ("linear", "linear")):
            aggregate(cells, by=by, func="sum", layer=layer).write_h5ad(
                f"{directory}/scanpy_{name}.h5ad"
            )
        exported = anndata.read_h5ad(f"{directory}/export.h5ad")
        aggregate(
            exported, by=["condition", "group_id"], func="sum", layer="counts"
        ).write_h5ad(f"{directory}/scanpy_export.h5ad")
        packages = ("scanpy", "anndata", "pandas")
    elif tool == "decoupler":
        import decoupler as dc

        for mode, name in (("sum", ""), ("mean", "_means"), ("median", "_medians")):
            pooled = dc.pp.pseudobulk(
                cells, sample_col="sample", groups_col="group", mode=mode
            )
            pooled.write_h5ad(f"{directory}/decoupler{name}.h5ad")
            if mode == "sum":  # and with the genes that filter_by_expr keeps
                dc.pp.filter_by_expr(pooled, group="condition", inplace=True)
                pooled.write_h5ad(f"{directory}/decoupler_filtered.h5ad")
        packages = ("decoupler", "anndata", "pandas")
    else:
        import liana as li
        import mudata

        # LIANA's var_names are nullable strings, which anndata writes under pandas 2
        # only when allowed.
        anndata.settings.allow_write_nullable_strings = True
        for mode, name in (("sum", ""), ("median", "_medians")):
            li.multi.adata_to_views(
                cells, groupby="group", sample_key="sample", obs_keys=["condition"],
                keep_stats=True, psbulk_kwargs={"mode": mode},
            ).write(f"{directory}/liana{name}.h5mu")
        for name in json.load(open(f"{directory}/exports.json")):
            path = f"{directory}/{name}"
            if name.endswith(".h5ad"):
                anndata.read_h5ad(path).write_h5ad(f"{path}.copy.h5ad")
            elif name.endswith(".ad.zarr"):
                anndata.read_zarr(path).write_zarr(f"{path}.copy.zarr")
            else:
                mudata.read_h5mu(path).write(f"{path}.copy.h5mu")
        packages = ("liana", "decoupler", "mudata", "anndata", "pandas")
    versions = {package: version(package) for package in packages}
    versions["python"] = sys.version.split()[0]
    json.dump(versions, open(f"{directory}/{tool}.json", "w"))
    '''
)


def simulate(
    seed: int, n_cells: int, n_genes: int, n_samples: int, n_groups: int
) -> Any:
    """Simulate negative binomial counts of shuffled cells, as an AnnData."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i + 1}" for i in range(n_samples)]
    groups = list(GROUPS)[:n_groups]
    cells = np.array(
        [
            rng.multinomial(n_cells // n_samples, rng.dirichlet(np.full(n_groups, 2.0)))
            for _ in samples
        ]
    )
    cells[n_samples - 2, n_groups - 2] = 0  # an empty entry
    cells[n_samples - 3, n_groups - 1] = 1  # one cell
    cells[n_samples - 1, n_groups - 1] = 2
    base = rng.lognormal(np.log(0.2), 1.2, size=n_genes)
    base[:3] = 0.0  # genes never detected
    effect = rng.lognormal(0.0, 0.8, size=(n_groups, n_genes))
    blocks, labels = [], []
    for s, sample in enumerate(samples):
        for g, group in enumerate(groups):
            if cells[s, g]:
                mean = base * effect[g] * rng.lognormal(0, 0.3, size=(cells[s, g], 1))
                counts = rng.negative_binomial(1.5, 1.5 / (1.5 + mean))
                empty = np.flatnonzero(counts.sum(axis=1) == 0)
                counts[empty, rng.integers(3, n_genes, size=empty.size)] = 1
                blocks.append(sp.csr_matrix(counts.astype(np.float32)))
                labels += [(sample, group)] * int(cells[s, g])
    order = rng.permutation(len(labels))
    sample_of, group_of = (np.array(column)[order] for column in zip(*labels))
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(sample_of),
            "group": pd.Categorical(group_of),
            "condition": ["ctrl" if int(s[1:]) % 2 else "stim" for s in sample_of],
            "lineage": [GROUPS[g] for g in group_of],
            "quality": rng.random(len(labels)),  # varies within every profile
        },
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(
        {"symbol": [f"GENE{j}" for j in range(n_genes)]},
        index=[f"ENSG{j:05d}" for j in range(n_genes)],
    )
    matrix = sp.vstack(blocks, format="csr")[order]
    matrix.sort_indices()
    return anndata.AnnData(X=matrix, obs=obs, var=var)


def pooled(cells: Any, layers: Any = None, metadata: str | None = None) -> Cube:
    """from_cells on the cells, with the features of var, and with the condition of
    each sample (metadata "samples") and the lineage of each cell type ("both")."""
    obs = cells.obs
    tables: dict[str, Any] = {}
    for kind, labels, column in (
        ("sample", "sample", "condition"),
        ("group", "group", "lineage"),
    ):
        if metadata == "both" or (metadata == "samples" and kind == "sample"):
            values = obs.groupby(labels, observed=True)[column].first()
            tables[f"{kind}_table"] = {
                f"{kind}_id": list(values.index),
                column: list(values),
            }
    return from_cells(
        cells.X,
        samples=obs["sample"],
        groups=obs["group"],
        features={
            "feature_id": list(cells.var_names),
            "symbol": list(cells.var["symbol"]),
        },
        layers=layers,
        **tables,
    )


def run(tool: str, directory: Path, variable: str) -> dict[str, str]:
    path = directory / "reference.py"
    path.write_text(PROGRAM)
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    python = os.environ.get(variable, sys.executable)
    subprocess.run(
        [python, "-I", str(path), tool, str(directory)], check=True, env=environment
    )
    versions: dict[str, str] = json.loads((directory / f"{tool}.json").read_text())
    return versions


def difference(
    cube: Cube, reference: Cube, names: tuple[str, ...], relative: bool
) -> float:
    """The largest difference of the layers and cell numbers of two cubes, aligned by ids."""
    samples, groups = (
        [ids.tolist().index(i) for i in own]
        for ids, own in (
            (reference.samples["sample_id"], cube.samples["sample_id"]),
            (reference.groups["group_id"], cube.groups["group_id"]),
        )
    )
    if list(cube.features["feature_id"]) != list(reference.features["feature_id"]):
        return float("inf")
    largest = 0.0
    pairs = [
        (cube.layer(name), reference.layer(name)[np.ix_(groups, samples)])
        for name in names
    ]
    if not relative:
        pairs.append((cube.n_cells, reference.n_cells[np.ix_(samples, groups)]))
    for found, expected in pairs:
        gap = np.abs(found.astype(np.float64) - expected)
        if relative:
            scale = np.abs(expected.astype(np.float64))
            if (gap[scale == 0] != 0).any():
                return float("inf")
            gap = gap[scale > 0] / scale[scale > 0]
        largest = max(largest, float(gap.max(initial=0.0)))
    return largest


def refusal(read: Any, expected: str) -> str:
    try:
        read()
    except ValueError as error:
        return "refused" if expected in str(error) else f"other message: {error}"
    return "accepted"


def exotic_cube() -> Cube:
    """Tables of every column type, a group and a sample without cells."""
    rng = np.random.default_rng(SEED + 7)
    counts = rng.poisson(2.0, size=(80, 6))
    counts[counts.sum(axis=1) == 0, 0] = 1
    samples = rng.choice(["A", "B", "C"], 80)
    return from_cells(
        counts,
        samples=samples,
        groups=rng.choice(["T", "NK"], 80),
        sample_table={
            "sample_id": ["A", "B", "C", "D"],
            "collected": np.array(
                ["2024-01-05", "NaT", "2024-03-01", "2024-04-11"], "datetime64[D]"
            ),
            "age": [41.5, np.nan, -0.0, 63.0],
            "site/batch": ["x", None, "y", "z"],
            "n_cells": np.array([1, 2, 3, 2**64 - 1], dtype=np.uint64),
            "notes": [None] * 4,
            "..": [1, 2, 3, 4],
        },
        group_table={
            "group_id": ["T", "NK", "empty"],
            "parent/lineage": ["L", None, "M"],
            "since": np.full(3, "NaT", dtype="datetime64[s]"),
        },
        features={
            "feature_id": [f"g{j}" for j in range(6)],
            "symbol": ["CD3E", None, "GZMB", "NKG7", "MS4A1", "LYZ"],
            "length": np.arange(6).astype("timedelta64[ms]"),
            "weight": np.linspace(0, 1, 6).astype(np.float16),
            "phase": np.arange(6) * (1 + 1j),
            "measured": [True, True, False, True, True, True],
            "alias": [None] * 6,
            "a\x00b": np.arange(6.0),
        },
    )


def write_fixtures(directory: Path) -> None:
    """Write a small simulation and the tools' outputs for the tests."""
    cells = simulate(SEED + 100, 240, 30, 4, 2)
    cells.write_h5ad(directory / "cells.h5ad")
    exported = to_anndata(pooled(cells, metadata="both"))
    exported.write_h5ad(directory / "export.h5ad")
    (directory / "exports.json").write_text("[]")
    for tool, variable in (
        ("scanpy", "SCANPY_PYTHON"),
        ("decoupler", "DECOUPLER_PYTHON"),
        ("liana", "LIANA_PYTHON"),
    ):
        run(tool, directory, variable)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    obs = cells.obs
    np.savez_compressed(
        FIXTURE_DIR / "cells.npz",
        data=cells.X.data,
        indices=cells.X.indices,
        indptr=cells.X.indptr,
        shape=np.array(cells.shape),
        **{
            name: np.asarray(values, dtype=str)
            for name, values in (
                ("sample", obs["sample"]),
                ("group", obs["group"]),
                ("condition", obs["condition"]),
                ("lineage", obs["lineage"]),
                ("feature_id", cells.var_names),
                ("symbol", cells.var["symbol"]),
            )
        },
    )
    for name in (
        "scanpy_counts.h5ad",
        "scanpy_means.h5ad",
        "decoupler.h5ad",
        "decoupler_means.h5ad",
        "liana.h5mu",
    ):
        (FIXTURE_DIR / name).write_bytes((directory / name).read_bytes())
    for path in sorted(FIXTURE_DIR.iterdir()):
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")


def checks(directory: Path, cells: Any, cubes: dict[str, Cube]) -> list[tuple]:
    """Every check, from what the tools wrote to the directory and the exports."""

    def read(name: str) -> Any:
        return anndata.read_h5ad(directory / name)

    def by_label(name: str, **options: Any) -> Cube:
        return from_anndata(read(name), sample="sample", group="group", **options)

    cube = cubes["all five layers"]
    integer = ("counts", "detected")
    counts_only = pooled(cells, layers=integer)
    with_condition = pooled(cells, layers=integer, metadata="samples")
    with_tables = pooled(cells, layers=integer, metadata="both")
    scanpy = by_label("scanpy_counts.h5ad")
    normalized = read("scanpy_counts.h5ad")
    for layer, name in (
        ("log_sum", "log"),
        ("log_sum_sq", "squares"),
        ("linear_sum", "linear"),
    ):
        normalized.layers[layer] = read(f"scanpy_{name}.h5ad")[
            normalized.obs_names
        ].layers["sum"]
    sources = {"counts": "sum", "detected": "count_nonzero"}
    sources |= {name: name for name in ("log_sum", "log_sum_sq", "linear_sum")}
    scanpy_all = from_anndata(
        normalized,
        sample="sample",
        group="group",
        layers=sources,
        normalization=cube.normalization,
    )
    decoupler = by_label("decoupler.h5ad")
    filtered = read("decoupler_filtered.h5ad")
    kept = with_tables.select(features=list(filtered.var_names))
    views = mudata.read_h5mu(directory / "liana.h5mu")
    crosstab = pd.crosstab(cells.obs["sample"], cells.obs["group"])
    liana = from_mudata(views, n_cells=crosstab)
    medians = mudata.read_h5mu(directory / "liana_medians.h5mu")
    rows = [
        (
            "scanpy sums and counts of nonzero values: counts, detected, n_cells",
            "from_cells",
            "max abs diff",
            difference(scanpy, cube, integer, False),
            0.0,
        ),
        (
            "the same: fingerprint, ids and tables included",
            "from_cells",
            "identical",
            yes(scanpy.fingerprint() == counts_only.fingerprint()),
            "yes",
        ),
        (
            "scanpy sums of log-normalized values, of their squares, of normalized values",
            "from_cells",
            "max rel diff",
            difference(scanpy_all, cube, ("log_sum", "log_sum_sq", "linear_sum"), True),
            EXACT,
        ),
        (
            "decoupler sums, fractions detected times cells, cells",
            "from_cells",
            "max abs diff",
            difference(decoupler, cube, integer, False),
            0.0,
        ),
        (
            "decoupler obs columns: condition per sample, lineage per cell type",
            "the simulated metadata",
            "identical tables",
            yes(
                decoupler.samples.equals(cube.samples)
                and decoupler.groups.equals(cube.groups)
            ),
            "yes",
        ),
        (
            f"decoupler after filter_by_expr ({filtered.n_vars:,} of {cells.n_vars:,} genes)",
            "from_cells, the same genes",
            "identical fingerprint",
            yes(
                by_label("decoupler_filtered.h5ad").fingerprint() == kept.fingerprint()
            ),
            "yes",
        ),
        (
            "LIANA views with the cells' crosstab: counts, detected, n_cells",
            "from_cells",
            "max abs diff",
            difference(liana, cube, integer, False),
            0.0,
        ),
        (
            "the same: fingerprint",
            "from_cells with each sample's condition",
            "identical",
            yes(liana.fingerprint() == with_condition.fingerprint()),
            "yes",
        ),
    ]
    refusals = [
        (
            "scanpy means and variances",
            "not sums",
            "no layer 'sum'",
            lambda: by_label("scanpy_means.h5ad"),
        ),
        (
            "decoupler with mode='mean'",
            "totals over cells",
            "X holds means",
            lambda: by_label("decoupler_means.h5ad"),
        ),
        (
            "decoupler with mode='median'",
            "zero where cells detect",
            "X does not hold sums of the counts",
            lambda: by_label("decoupler_medians.h5ad"),
        ),
        (
            "LIANA views with mode='median'",
            "zero where cells detect",
            "X does not hold sums of the counts",
            lambda: from_mudata(medians, n_cells=crosstab),
        ),
        (
            "LIANA views without cell numbers, with an entry without cells",
            "no entry without cells",
            "no cell numbers",
            lambda: from_mudata(views),
        ),
    ]
    rows += [
        (check, f"refused: {reason}", "outcome", refusal(call, message), "refused")
        for check, reason, message, call in refusals
    ]
    # The exports have the layouts of scanpy and LIANA, and scanpy reads them.
    exported, reference = read("export.h5ad"), read("scanpy_counts.h5ad")
    mine = exported[
        [f"{s}_{g}" for s, g in zip(reference.obs["sample"], reference.obs["group"])]
    ]
    gap = max(
        np.abs(mine.layers["counts"] - reference.layers["sum"]).max(),
        np.abs(mine.layers["detected"] - reference.layers["count_nonzero"]).max(),
        abs(exported.n_obs - reference.n_obs),
    )
    ours = to_mudata(cube)
    same = sorted(ours.mod) == sorted(views.mod) and all(
        list(ours[name].var_names) == list(view.var_names)
        and sorted(ours[name].obs_names) == sorted(view.obs_names)
        and np.array_equal(ours[name][list(view.obs_names)].layers["counts"], view.X)
        for name, view in views.mod.items()
    )
    by_condition = read("scanpy_export.h5ad")
    collapsed = cube.collapse_samples("condition")
    ids = (
        collapsed.groups["group_id"].tolist(),
        collapsed.samples["sample_id"].tolist(),
    )
    expected = np.stack(
        [
            collapsed.layer("counts")[ids[0].index(g), ids[1].index(c)]
            for c, g in zip(by_condition.obs["condition"], by_condition.obs["group_id"])
        ]
    )
    rows += [
        (
            "to_anndata: rows, counts and detected",
            "scanpy.get.aggregate",
            "max abs diff",
            float(gap),
            0.0,
        ),
        (
            "to_mudata: modalities, var_names, samples and counts",
            "liana.multi.adata_to_views",
            "identical",
            yes(same),
            "yes",
        ),
        (
            "scanpy aggregates the export by condition and cell type",
            "Cube.collapse_samples",
            "max abs diff",
            float(np.abs(by_condition.layers["sum"] - expected).max()),
            0.0,
        ),
    ]
    # Round trips, in memory, through files, and through files written again in the
    # environment of LIANA, under pandas 2.
    kept_fingerprints, copied = [], []
    for index, exported_cube in enumerate(cubes.values()):
        fingerprint, path = exported_cube.fingerprint(), directory / str(index)
        readings = [
            from_anndata(to_anndata(exported_cube)),
            from_mudata(to_mudata(exported_cube)),
            from_anndata(anndata.read_h5ad(f"{path}.h5ad")),
            from_anndata(anndata.read_zarr(f"{path}.ad.zarr")),
            from_mudata(mudata.read_h5mu(f"{path}.h5mu")),
            from_mudata(mudata.read_zarr(f"{path}.mu.zarr")),
        ]
        copies = [
            from_anndata(anndata.read_h5ad(f"{path}.h5ad.copy.h5ad")),
            from_anndata(anndata.read_zarr(f"{path}.ad.zarr.copy.zarr")),
            from_mudata(mudata.read_h5mu(f"{path}.h5mu.copy.h5mu")),
        ]
        kept_fingerprints += [read.fingerprint() == fingerprint for read in readings]
        copied += [read.fingerprint() == fingerprint for read in copies]
    for check, found in (
        (
            f"round trips of {len(cubes)} cubes: AnnData and MuData, in memory and through "
            ".h5ad, .zarr and .h5mu files",
            kept_fingerprints,
        ),
        ("the same files, rewritten by anndata and mudata under pandas 2", copied),
    ):
        rows.append(
            (
                check,
                "the exported cube",
                "fingerprints kept",
                f"{sum(found)} of {len(found)}",
                f"{len(found)} of {len(found)}",
            )
        )
    return rows


def timings() -> dict[str, float]:
    """Seconds for each conversion of a cube of counts and detected cells of 20 cell
    types x 50 samples x 20,000 genes; to_anndata's includes the fingerprint."""
    rng = np.random.default_rng(SEED + 9)
    counts = rng.poisson(3.0, size=(20, 50, 20_000)).astype(np.uint32)
    cube = Cube(
        {"counts": counts, "detected": np.minimum(counts, 5)},
        samples={"sample_id": [f"S{i}" for i in range(50)]},
        groups={"group_id": [f"T{i}" for i in range(20)]},
        features={"feature_id": [f"G{i}" for i in range(20_000)]},
        n_cells=np.full((50, 20), 5),
    )
    seconds, results = {}, {}
    for name, call in (
        ("to_anndata", lambda: to_anndata(cube)),
        ("from_anndata", lambda: from_anndata(results["to_anndata"])),
        ("to_mudata", lambda: to_mudata(cube)),
        ("from_mudata", lambda: from_mudata(results["to_mudata"])),
    ):
        start = time.perf_counter()
        results[name] = call()
        seconds[name] = time.perf_counter() - start
    return seconds


def yes(condition: bool) -> str:
    return "yes" if condition else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)  # pandas, inside mudata
    cells = simulate(SEED, 20_000, 1_500, 8, 10)
    cube = pooled(cells, metadata="both")
    merged = cube.regroup({"all": cube.groups["group_id"].tolist()})
    cubes = {
        "all five layers": cube,
        "merged, pooled and selected": cube.regroup({"T": ["CD4_T", "CD8_T"]})
        .collapse_samples("condition")
        .select(features=np.arange(1_499, 0, -2)),
        "bulk": from_bulk(merged.layer("counts", "all")[:, :50], samples=cube.samples),
        "tables of every column type": exotic_cube(),
    }
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cells.write_h5ad(directory / "cells.h5ad")
        to_anndata(cube).write_h5ad(directory / "export.h5ad")
        files = []
        for index, kept in enumerate(cubes.values()):
            adata, mdata = to_anndata(kept), to_mudata(kept)
            adata.write_h5ad(directory / f"{index}.h5ad")
            adata.write_zarr(directory / f"{index}.ad.zarr")
            mdata.write(directory / f"{index}.h5mu")
            mdata.write_zarr(directory / f"{index}.mu.zarr")
            files += [f"{index}.h5ad", f"{index}.ad.zarr", f"{index}.h5mu"]
        (directory / "exports.json").write_text(json.dumps(files))
        versions = {
            tool: run(tool, directory, variable)
            for tool, variable in (
                ("scanpy", "SCANPY_PYTHON"),
                ("decoupler", "DECOUPLER_PYTHON"),
                ("liana", "LIANA_PYTHON"),
            )
        }
        rows = checks(directory, cells, cubes)
    if arguments.write_fixtures:
        with tempfile.TemporaryDirectory() as fixtures:
            write_fixtures(Path(fixtures))
    for tool, found in versions.items():
        print(
            f"{tool} reference versions: "
            + ", ".join(f"{k} {v}" for k, v in found.items())
        )
    print(
        f"CytePool side: numpy {np.__version__}, numba {numba.__version__}, anndata "
        f"{anndata.__version__}, mudata {mudata.__version__}, pandas {pd.__version__}"
    )
    print(
        f"Data: {cells.n_obs:,} cells x {cells.n_vars:,} genes, {cube.n_samples} samples "
        f"x {cube.n_groups} cell types; seed {SEED}"
    )
    print()
    print("| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    passed = True
    for check, reference, metric, value, tolerance in rows:
        ok = value == tolerance if isinstance(value, str) else bool(value <= tolerance)
        shown = f"{value:.2g}" if isinstance(value, float) else str(value)
        limit = f"{tolerance:.2g}" if isinstance(tolerance, float) else str(tolerance)
        passed &= ok
        print(
            f"| {check} | {reference} | {metric} | {shown} | {limit} | "
            f"{'pass' if ok else 'FAIL'} |"
        )
    print()
    print("| timing (one call; 20 x 50 x 20,000, counts and detected) | seconds |")
    print("|---|---|")
    for name, seconds in timings().items():
        print(f"| {name} | {seconds:.3f} |")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
