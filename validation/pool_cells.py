"""Check from_cells and from_bulk against scanpy.get.aggregate and decoupler.

The script simulates a cell-level count matrix (30,000 cells x 2,000 genes, 8
samples x 10 cell types, about 20% non-zero, cells in random order, with an empty
sample x cell type entry, entries with one and two cells, and genes that are never
detected), pools it with ``from_cells``, and compares the cube with two references
run on the same cells:

- ``scanpy.get.aggregate``: sum and count_nonzero of the counts, mean and var of
  log1p(normalize_total), and mean of normalize_total, after scanpy normalized the
  cells itself (in float64, so that rounding in float32 does not limit agreement).
- ``decoupler.pp.pseudobulk``: sums, cells and total counts per sample and cell
  type, the fraction of cells detecting each gene, and sums per sample.

Cases: integer counts stored as float32 (as in most h5ad files) and non-integer
counts. Invariants: the same counts as a CSR matrix whose rows list entries out of
order, split counts into repeated entries and store explicit zeros, or as a dense
array, and pooling with one thread, give the identical cube. ``from_bulk`` is
checked on decoupler's per-sample sums. Finally the script times ``from_cells``
against scanpy's normalization and aggregation (float32, as usually run) and
decoupler's pseudobulk.

The references run as separate Python programs that exchange data with this
script through files in a temporary directory.

Requirements:
    This script: numpy, scipy (to build the CSR matrices), and cytepool from the
    source tree.
    scanpy reference: numpy, scipy, pandas, anndata and scanpy. It runs in
    SCANPY_PYTHON, or in this interpreter when that is not set.
    decoupler reference: numpy, scipy, pandas, anndata and decoupler. It runs in
    DECOUPLER_PYTHON, or in this interpreter when that is not set.

Usage, from the repository root:
    PYTHONPATH=. python validation/pool_cells.py
    PYTHONPATH=. python validation/pool_cells.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy.sparse as sp

from cytepool import Cube, from_bulk, from_cells

SEED = 20261023
TARGET_SUM = 1e4
EXACT = 1e-10
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "pool"
SAMPLES = [f"S{i + 1}" for i in range(8)]
GROUPS = [
    "B",
    "CD4 T",
    "CD8 T",
    "NK",
    "CD14 Mono",
    "FCGR3A Mono",
    "cDC",
    "pDC",
    "Plasma",
    "Platelet",
]

SCANPY_PROGRAM = textwrap.dedent(
    '''
    """Reference: scanpy.get.aggregate on cells that scanpy normalizes itself."""

    import json
    import sys
    import time
    import warnings
    from importlib.metadata import version

    import anndata
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    timings = {}
    for case in json.load(open(f"{directory}/cases.json")):
        data = np.load(f"{directory}/{case}.npz")
        shape = tuple(int(n) for n in data["shape"])
        obs = pd.DataFrame(
            {
                "sample": pd.Categorical(data["sample"]),
                "group": pd.Categorical(data["group"]),
            },
            index=[f"cell{i}" for i in range(shape[0])],
        )

        def cells(dtype):
            matrix = sp.csr_matrix(
                (data["data"].astype(dtype), data["indices"], data["indptr"]),
                shape=shape,
            )
            return anndata.AnnData(X=matrix, obs=obs.copy())

        def aggregate(adata):
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=float(data["target_sum"]))
            adata.layers["linear"] = adata.X.copy()
            sc.pp.log1p(adata)
            by = ["sample", "group"]
            raw = sc.get.aggregate(
                adata, by=by, func=["sum", "count_nonzero"], layer="counts"
            )
            log = sc.get.aggregate(adata, by=by, func=["mean", "var"])
            linear = sc.get.aggregate(adata, by=by, func="mean", layer="linear")
            return raw, log[raw.obs_names], linear[raw.obs_names]

        raw, log, linear = aggregate(cells(np.float64))
        np.savez(
            f"{directory}/{case}_scanpy.npz",
            keys_sample=raw.obs["sample"].astype(str).to_numpy().astype(str),
            keys_group=raw.obs["group"].astype(str).to_numpy().astype(str),
            n_cells=raw.obs["n_obs_aggregated"].to_numpy(),
            sum=np.asarray(raw.layers["sum"]),
            count_nonzero=np.asarray(raw.layers["count_nonzero"]),
            mean_log=np.asarray(log.layers["mean"]),
            var_log=np.asarray(log.layers["var"]),
            mean_linear=np.asarray(linear.layers["mean"]),
        )
        if bool(data["timed"]):
            # As usually run, on the float32 matrix of an h5ad file: once to warm
            # up, since the first call in a process is much slower, then the best
            # of three runs.
            aggregate(cells(data["data"].dtype))
            best = float("inf")
            for _ in range(3):
                adata = cells(data["data"].dtype)
                start = time.perf_counter()
                aggregate(adata)
                best = min(best, time.perf_counter() - start)
            timings[case] = best
    packages = ("scanpy", "anndata", "numpy", "scipy", "pandas")
    versions = {package: version(package) for package in packages}
    versions["python"] = sys.version.split()[0]
    with open(f"{directory}/scanpy.json", "w") as handle:
        json.dump({"versions": versions, "timings": timings}, handle)
    '''
)

DECOUPLER_PROGRAM = textwrap.dedent(
    '''
    """Reference: decoupler's pseudobulk sums per sample and cell type."""

    import json
    import sys
    import time
    import warnings
    from importlib.metadata import version

    import anndata
    import decoupler as dc
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    timings = {}
    for case in json.load(open(f"{directory}/cases.json")):
        data = np.load(f"{directory}/{case}.npz")
        shape = tuple(int(n) for n in data["shape"])
        whole = bool(np.all(data["data"] == np.floor(data["data"])))
        obs = pd.DataFrame(
            {"sample": data["sample"], "group": data["group"]},
            index=[f"cell{i}" for i in range(shape[0])],
        )
        matrix = sp.csr_matrix(
            (data["data"].astype(np.float64), data["indices"], data["indptr"]),
            shape=shape,
        )
        cells = anndata.AnnData(X=matrix, obs=obs)

        def pseudobulk(groups_col):
            return dc.pp.pseudobulk(
                cells, sample_col="sample", groups_col=groups_col, skip_checks=not whole
            )

        pooled = pseudobulk("group")
        bulk = pseudobulk(None)
        if bool(data["timed"]):  # the best of three runs after the first
            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                pseudobulk("group")
                best = min(best, time.perf_counter() - start)
            timings[case] = best
        np.savez(
            f"{directory}/{case}_decoupler.npz",
            sample=pooled.obs["sample"].astype(str).to_numpy().astype(str),
            group=pooled.obs["group"].astype(str).to_numpy().astype(str),
            sum=np.asarray(pooled.X),
            n_cells=pooled.obs["psbulk_cells"].to_numpy(),
            counts=pooled.obs["psbulk_counts"].to_numpy(),
            props=np.asarray(pooled.layers["psbulk_props"]),
            bulk_sample=bulk.obs["sample"].astype(str).to_numpy().astype(str),
            bulk_sum=np.asarray(bulk.X),
        )
    packages = ("decoupler", "anndata", "numpy", "scipy", "pandas")
    versions = {package: version(package) for package in packages}
    versions["python"] = sys.version.split()[0]
    with open(f"{directory}/decoupler.json", "w") as handle:
        json.dump({"versions": versions, "timings": timings}, handle)
    '''
)


def cells_per_entry(rng: np.random.Generator, n_cells: int) -> np.ndarray:
    composition = rng.dirichlet(np.full(len(GROUPS), 2.0), size=len(SAMPLES))
    cells = np.array([rng.multinomial(n_cells // len(SAMPLES), p) for p in composition])
    cells[6, GROUPS.index("cDC")] = 0  # an empty entry
    cells[4, GROUPS.index("Plasma")] = 1  # one cell: no sample variance
    cells[7, GROUPS.index("Plasma")] = 2
    return cells


def simulate(
    seed: int, samples: list[str], groups: list[str], cells: np.ndarray, n_genes: int
) -> dict[str, Any]:
    """Simulate negative binomial counts, then shuffle the cells.

    Returns integer counts as a CSR matrix with float32 data, and the labels.
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(np.log(0.15), 1.2, size=n_genes)
    base[:5] = 0.0  # genes that are never detected
    effect = rng.lognormal(0.0, 0.8, size=(len(groups), n_genes))
    blocks, sample_of_cell, group_of_cell = [], [], []
    for s, sample in enumerate(samples):
        for g, group in enumerate(groups):
            n = int(cells[s, g])
            if n == 0:
                continue
            mean = base * effect[g] * rng.lognormal(0.0, 0.35, size=(n, 1))
            counts = rng.negative_binomial(1.5, 1.5 / (1.5 + mean))
            empty = np.flatnonzero(counts.sum(axis=1) == 0)
            counts[empty, rng.integers(5, n_genes, size=empty.size)] = 1
            blocks.append(sp.csr_matrix(counts.astype(np.float32)))
            sample_of_cell += [sample] * n
            group_of_cell += [group] * n
    order = rng.permutation(len(sample_of_cell))
    matrix = sp.vstack(blocks, format="csr")[order]
    matrix.sort_indices()
    return {
        "counts": matrix,
        "sample": np.array(sample_of_cell)[order],
        "group": np.array(group_of_cell)[order],
    }


def scrambled(matrix: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    """Return the same counts with rows out of order, repeats and explicit zeros.

    A fifth of the counts above 1 are split into two entries of the same column,
    every row gets one explicit zero, and each row lists its entries in random
    order.
    """
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    columns = matrix.indices.astype(np.int64)
    values = matrix.data.astype(np.float64)
    split = (values > 1) & (rng.random(values.size) < 0.2)
    part = np.floor(values[split] / 2)
    values = values.copy()
    values[split] -= part
    zero_columns = rng.integers(0, matrix.shape[1], size=matrix.shape[0])
    all_rows = np.concatenate([rows, rows[split], np.arange(matrix.shape[0])])
    all_columns = np.concatenate([columns, columns[split], zero_columns])
    all_values = np.concatenate([values, part, np.zeros(matrix.shape[0])])
    order = np.lexsort((rng.random(all_rows.size), all_rows))
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(all_rows, minlength=matrix.shape[0])))
    )
    result = sp.csr_matrix(
        (all_values[order], all_columns[order], indptr), shape=matrix.shape
    )
    assert not result.has_canonical_format
    return result


def run_references(
    directory: Path, cases: dict[str, dict[str, Any]], timed: set[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    for case, simulated in cases.items():
        matrix = simulated["counts"]
        np.savez(
            directory / f"{case}.npz",
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=np.array(matrix.shape),
            sample=simulated["sample"],
            group=simulated["group"],
            target_sum=TARGET_SUM,
            timed=case in timed,
        )
    (directory / "cases.json").write_text(json.dumps(list(cases)))
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    report: dict[str, Any] = {}
    for name, program, variable in (
        ("scanpy", SCANPY_PROGRAM, "SCANPY_PYTHON"),
        ("decoupler", DECOUPLER_PROGRAM, "DECOUPLER_PYTHON"),
    ):
        path = directory / f"{name}_reference.py"
        path.write_text(program)
        python = os.environ.get(variable, sys.executable)
        subprocess.run(
            [python, "-I", str(path), str(directory)], check=True, env=environment
        )
        report[name] = json.loads((directory / f"{name}.json").read_text())
    results = {
        case: {
            f"{name}_{key}": value
            for name in ("scanpy", "decoupler")
            for key, value in np.load(directory / f"{case}_{name}.npz").items()
        }
        for case in cases
    }
    return results, report


def comparable(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the entries to compare, or None when NaN appears in only one array.

    Entries that are NaN in both arrays agree and are left out.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        return None
    missing = np.isnan(actual)
    if np.any(missing != np.isnan(expected)):
        return None
    return actual[~missing], expected[~missing]


def relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    """Largest |actual - expected| / |expected|; where expected is 0, any difference fails."""
    pair = comparable(actual, expected)
    if pair is None:
        return float("inf")
    actual, expected = pair
    difference = np.abs(actual - expected)
    scale = np.abs(expected)
    if np.any(difference[scale == 0] != 0):
        return float("inf")
    nonzero = scale > 0
    return float(np.max(difference[nonzero] / scale[nonzero], initial=0.0))


def largest_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    pair = comparable(actual, expected)
    if pair is None:
        return float("inf")
    actual, expected = pair
    return float(np.max(np.abs(actual - expected), initial=0.0))


def positions(cube: Cube, samples: np.ndarray, groups: np.ndarray) -> tuple[Any, Any]:
    sample_index = {s: i for i, s in enumerate(cube.samples["sample_id"].tolist())}
    group_index = {g: i for i, g in enumerate(cube.groups["group_id"].tolist())}
    return (
        np.array([group_index[g] for g in groups.tolist()]),
        np.array([sample_index[s] for s in samples.tolist()]),
    )


def compare(
    cube: Cube, reference: dict[str, np.ndarray], label: str, counts_dtype: str
) -> list[tuple]:
    """Compare a cube with scanpy's and decoupler's results for one case."""
    whole = counts_dtype.startswith("uint")
    n_cells = cube.n_cells
    assert n_cells is not None
    groups, samples = positions(
        cube, reference["scanpy_keys_sample"], reference["scanpy_keys_group"]
    )
    several = reference["scanpy_n_cells"] >= 2
    counts = cube.layer("counts")[groups, samples]
    if whole:
        counts_row = (
            "max abs diff",
            largest_difference(counts, reference["scanpy_sum"]),
            0.0,
        )
    else:
        counts_row = (
            "max rel diff",
            relative_difference(counts, reference["scanpy_sum"]),
            EXACT,
        )
    rows = [
        (
            f"{label}: dtype of counts",
            "the rule: uint32 for whole sums below 2**32, float64 for fractions",
            "dtype",
            str(cube.layer("counts").dtype),
            counts_dtype,
        ),
        (f"{label}: counts", "scanpy sum", *counts_row),
        (
            f"{label}: detected, n_cells",
            "scanpy count_nonzero, n_obs_aggregated",
            "max abs diff",
            max(
                largest_difference(
                    cube.layer("detected")[groups, samples],
                    reference["scanpy_count_nonzero"],
                ),
                largest_difference(
                    n_cells[samples, groups], reference["scanpy_n_cells"]
                ),
            ),
            0.0,
        ),
        (
            f"{label}: mean of log values",
            "scanpy mean of log1p(normalize_total)",
            "max rel diff",
            relative_difference(
                cube.mean()[groups, samples], reference["scanpy_mean_log"]
            ),
            EXACT,
        ),
        (
            f"{label}: variance of log values (2 or more cells)",
            "scanpy var, dof=1",
            "max rel diff",
            relative_difference(
                cube.variance()[groups[several], samples[several]],
                reference["scanpy_var_log"][several],
            ),
            EXACT,
        ),
        (
            f"{label}: mean of linear values",
            "scanpy mean of normalize_total",
            "max rel diff",
            relative_difference(
                cube.mean(scale="linear")[groups, samples],
                reference["scanpy_mean_linear"],
            ),
            EXACT,
        ),
    ]
    observed = np.zeros((cube.n_groups, cube.n_samples), dtype=bool)
    observed[groups, samples] = True
    empty_ok = bool(
        (~observed).any()
        and (n_cells.T[~observed] == 0).all()
        and all(not cube.layer(name)[~observed].any() for name in cube.layer_names)
        and np.isnan(cube.mean()[~observed]).all()
    )
    single_ok = bool(
        (~several).any()
        and np.isnan(reference["scanpy_var_log"][~several]).all()
        and np.isnan(cube.variance()[groups[~several], samples[~several]]).all()
    )
    rows.append(
        (
            f"{label}: {int((~observed).sum())} empty and {int((~several).sum())} "
            "single-cell entries",
            "scanpy: no row; var NaN",
            "zero and NaN where expected",
            "yes" if empty_ok and single_ok else "no",
            "yes",
        )
    )
    groups, samples = positions(
        cube, reference["decoupler_sample"], reference["decoupler_group"]
    )
    counts = cube.layer("counts")[groups, samples]
    sums = (
        largest_difference(counts, reference["decoupler_sum"])
        if whole
        else relative_difference(counts, reference["decoupler_sum"])
    )
    # decoupler keeps entries without cells and gives them a detection fraction of
    # 0; the cube leaves 0 / 0 undefined (NaN).
    filled = reference["decoupler_n_cells"] > 0
    unfilled_ok = bool(
        (~filled).any()
        and (reference["decoupler_props"][~filled] == 0).all()
        and np.isnan(cube.fraction_detected()[groups[~filled], samples[~filled]]).all()
    )
    rows += [
        (
            f"{label}: counts",
            "decoupler pseudobulk sum",
            "max abs diff" if whole else "max rel diff",
            sums,
            0.0 if whole else EXACT,
        ),
        (
            f"{label}: n_cells, library sizes",
            "decoupler psbulk_cells, psbulk_counts",
            "max rel diff",
            max(
                relative_difference(
                    n_cells[samples, groups], reference["decoupler_n_cells"]
                ),
                relative_difference(
                    cube.library_sizes()[samples, groups], reference["decoupler_counts"]
                ),
            ),
            0.0 if whole else EXACT,
        ),
        (
            f"{label}: fraction detected (entries with cells)",
            "decoupler psbulk_props",
            "max rel diff",
            relative_difference(
                cube.fraction_detected()[groups[filled], samples[filled]],
                reference["decoupler_props"][filled],
            ),
            EXACT,
        ),
        (
            f"{label}: fraction detected without cells",
            "decoupler psbulk_props (0 by convention)",
            "NaN in the cube, 0 in decoupler",
            "yes" if unfilled_ok else "no",
            "yes",
        ),
    ]
    return rows


def bulk_rows(cube: Cube, reference: dict[str, np.ndarray]) -> list[tuple]:
    """Per-sample sums: the cube merged over cell types, and from_bulk."""
    order = [
        cube.samples["sample_id"].tolist().index(sample)
        for sample in reference["decoupler_bulk_sample"].tolist()
    ]
    expected = reference["decoupler_bulk_sum"]
    merged = cube.regroup({"all": cube.groups["group_id"].tolist()})
    bulk = from_bulk(expected, samples=reference["decoupler_bulk_sample"])
    return [
        (
            "per-sample sums: from_cells, then regroup into one group",
            "decoupler pseudobulk, groups_col=None",
            "max abs diff",
            largest_difference(merged.layer("counts", "all")[order], expected),
            0.0,
        ),
        (
            "from_bulk of those sums gives them back, as uint32",
            "decoupler's per-sample sums (round trip)",
            "max abs diff",
            largest_difference(bulk.layer("counts", "bulk"), expected)
            + (0.0 if bulk.layer("counts").dtype == np.uint32 else float("inf")),
            0.0,
        ),
    ]


def identical(first: Cube, second: Cube) -> bool:
    return first.layer_names == second.layer_names and all(
        first.layer(name).dtype == second.layer(name).dtype
        and np.array_equal(first.layer(name), second.layer(name))
        for name in first.layer_names
    )


def print_table(rows: list[tuple]) -> bool:
    print("| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    passed = True
    for check, reference, metric, value, tolerance in rows:
        shown = f"{value:.2g}" if isinstance(value, float) else str(value)
        ok = value == tolerance if isinstance(value, str) else bool(value <= tolerance)
        limit = f"{tolerance:.2g}" if isinstance(tolerance, float) else str(tolerance)
        passed &= ok
        result = "pass" if ok else "FAIL"
        print(f"| {check} | {reference} | {metric} | {shown} | {limit} | {result} |")
    return passed


def best_time(function: Any, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def write_fixtures(directory: Path) -> None:
    """Write a small simulation and the references' results for the tests."""
    rng = np.random.default_rng(SEED + 100)
    samples, groups = ["S1", "S2", "S3", "S4"], GROUPS[:5]
    cells = rng.integers(3, 30, size=(len(samples), len(groups)))
    cells[1, 2] = 0  # an empty entry
    cells[2, 4] = 1  # a single cell
    simulated = simulate(SEED + 101, samples, groups, cells, 30)
    results, _ = run_references(directory, {"fixture": simulated}, timed=set())
    reference = results["fixture"]
    matrix = simulated["counts"]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "scanpy_decoupler.npz"
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape),
        sample=simulated["sample"],
        group=simulated["group"],
        keys_sample=reference["scanpy_keys_sample"],
        keys_group=reference["scanpy_keys_group"],
        scanpy_n_cells=reference["scanpy_n_cells"],
        scanpy_sum=reference["scanpy_sum"],
        scanpy_count_nonzero=reference["scanpy_count_nonzero"],
        scanpy_mean_log=reference["scanpy_mean_log"],
        scanpy_mean_linear=reference["scanpy_mean_linear"],
        scanpy_var_log=reference["scanpy_var_log"],
        dc_sample=reference["decoupler_sample"],
        dc_group=reference["decoupler_group"],
        dc_sum=reference["decoupler_sum"],
        dc_n_cells=reference["decoupler_n_cells"],
        dc_counts=reference["decoupler_counts"],
        dc_props=reference["decoupler_props"],
    )
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    rng = np.random.default_rng(SEED)
    cells = cells_per_entry(rng, 30_000)
    integer = simulate(SEED + 1, SAMPLES, GROUPS, cells, 2_000)
    fraction = np.random.default_rng(SEED + 2).uniform(0.5, 1.0, integer["counts"].nnz)
    fractional_counts = integer["counts"].astype(np.float64)
    fractional_counts.data *= fraction
    fractional = dict(integer, counts=fractional_counts)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        results, report = run_references(
            directory,
            {"integer": integer, "fractional": fractional},
            timed={"integer"},
        )
        if arguments.write_fixtures:
            write_fixtures(directory)

    def pool(counts: Any, **options: Any) -> Cube:
        return from_cells(
            counts,
            sample_labels=integer["sample"],
            group_labels=integer["group"],
            **options,
        )

    start = time.perf_counter()
    cube = pool(integer["counts"])
    first_call = time.perf_counter() - start
    fractional_cube = pool(fractional["counts"])
    rows = compare(cube, results["integer"], "integer counts (float32 CSR)", "uint32")
    rows += compare(
        fractional_cube, results["fractional"], "non-integer counts", "float64"
    )
    rows += bulk_rows(cube, results["integer"])
    dense = integer["counts"].toarray()
    invariants = (
        (
            "unsorted, repeated and zero CSR entries",
            scrambled(integer["counts"], SEED + 3),
        ),
        ("dense array", dense),
    )
    for name, counts in invariants:
        rows.append(
            (
                f"invariant: {name}",
                "from_cells on the canonical CSR matrix",
                "identical cube",
                "yes" if identical(pool(counts), cube) else "no",
                "yes",
            )
        )
    rows.append(
        (
            "invariant: one thread",
            "from_cells with every thread",
            "identical cube",
            "yes" if identical(pool(integer["counts"], n_threads=1), cube) else "no",
            "yes",
        )
    )
    for name in ("scanpy", "decoupler"):
        versions = report[name]["versions"]
        print(
            f"{name} reference versions: "
            + ", ".join(f"{key} {value}" for key, value in versions.items())
        )
    print(f"CytePool side: numpy {np.__version__}, numba {numba.__version__}")
    matrix = integer["counts"]
    print(
        f"Data: {matrix.shape[0]:,} cells x {matrix.shape[1]:,} genes, "
        f"{matrix.nnz:,} stored counts (density {matrix.nnz / np.prod(matrix.shape):.3f}), "
        f"{cube.n_samples} samples x {cube.n_groups} cell types; seed {SEED}"
    )
    print()
    passed = print_table(rows)

    threads = numba.get_num_threads()
    timings = {
        f"from_cells, {threads} threads": best_time(lambda: pool(integer["counts"])),
        "from_cells, 1 thread": best_time(lambda: pool(integer["counts"], n_threads=1)),
        "from_cells, dense float32 input": best_time(lambda: pool(dense)),
        "from_cells, first call (compiles unless cached)": first_call,
        "scanpy normalize_total, log1p and 3 aggregate calls (float32), warm": report[
            "scanpy"
        ]["timings"]["integer"],
        "decoupler pseudobulk (sums only), warm": report["decoupler"]["timings"][
            "integer"
        ],
    }
    print()
    print("| timing (best of 3 after a first run; all five layers) | seconds |")
    print("|---|---|")
    for name, seconds in timings.items():
        print(f"| {name} | {seconds:.3f} |")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
