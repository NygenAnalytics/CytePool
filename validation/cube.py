"""Check the pseudobulk cube against scanpy.get.aggregate computed from the cells.

The script simulates a cell-level count matrix (about 29,000 cells x 2,000 genes, 8
samples x 10 cell types, with an empty sample x cell type entry and entries with one
and two cells), builds the cube layers from it with plain numpy sums, and compares the
cube's
means, variances, detection fractions and library sizes, and the results of
``Cube.regroup`` and ``Cube.collapse_samples``, with ``scanpy.get.aggregate`` run on
the cells. scanpy normalizes the cells itself (``normalize_total`` and ``log1p``), so
the comparison also confirms the definitions of the normalized layers.

It then checks the float64 contract of the variance from raw moments: on values with
mean-to-standard-deviation ratio r, the relative error must stay below
``64 * eps * (1 + r**2)``, where eps is the float64 machine epsilon. The reference
there is the exact variance of the same float64 values in rational arithmetic. The
constant is a worst-case rounding bound: numpy adds 1,000 values in blocks of up to
128, each with eight running partial sums, for an error of at most roughly 13 to 20
eps relative to the sum; ``sum_sq - sum**2 / n`` then carries at most about three
times that, times ``sum_sq``, which is roughly 40 to 60 eps (1 + r**2) relative to
the variance. Observed errors are 0.4 to 1 times eps (1 + r**2). At r = 8e6 the
bound exceeds 0.5, so that row only shows where raw moments stop being useful and
does not count.

The references run as a separate Python program that exchanges data with this script
through files in a temporary directory.

Requirements:
    This script: numpy, and cytepool from the source tree.
    Reference program: numpy, scipy, pandas, anndata and scanpy. It runs in the same
    interpreter unless REFERENCE_PYTHON names another Python executable.

Usage, from the repository root:
    PYTHONPATH=. python validation/cube.py
    PYTHONPATH=. python validation/cube.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.cube import Cube

SEED = 20261022
TARGET_SUM = 1e4
EXACT = 1e-10
EPS = np.finfo(np.float64).eps
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "cube"

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
MERGES = {
    "B lineage": ["B", "Plasma"],
    "T and NK": ["CD4 T", "CD8 T", "NK"],
    "Myeloid": ["CD14 Mono", "FCGR3A Mono", "cDC", "pDC"],
}  # "Platelet" is kept as it is
STRESS_RATIOS = [1e1, 8e2, 8e4, 8e6]
STRESS_CELLS = 1000

REFERENCE_PROGRAM = textwrap.dedent(
    '''
    """Reference statistics: scanpy.get.aggregate and exact rational variances."""

    import json
    import statistics
    import sys
    import warnings
    from importlib.metadata import version

    import anndata
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    warnings.filterwarnings("ignore")
    directory = sys.argv[1]
    data = np.load(f"{directory}/cells.npz")
    counts = sp.csr_matrix(
        (data["data"].astype(np.float64), data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )
    labels = ("sample", "group", "donor", "coarse")
    obs = pd.DataFrame({name: pd.Categorical(data[name]) for name in labels})
    obs.index = [f"cell{i}" for i in range(counts.shape[0])]
    cells = anndata.AnnData(X=counts, obs=obs)
    cells.layers["counts"] = cells.X.copy()
    sc.pp.normalize_total(cells, target_sum=float(data["target_sum"]))
    cells.layers["linear"] = cells.X.copy()
    sc.pp.log1p(cells)

    results = {}
    for name, by in (
        ("fine", ["sample", "group"]),
        ("coarse", ["sample", "coarse"]),
        ("donor", ["donor", "group"]),
    ):
        raw = sc.get.aggregate(cells, by=by, func=["sum", "count_nonzero"], layer="counts")
        log = sc.get.aggregate(cells, by=by, func=["mean", "var"])[raw.obs_names]
        linear = sc.get.aggregate(cells, by=by, func="mean", layer="linear")[raw.obs_names]
        results[f"{name}_keys0"] = raw.obs[by[0]].astype(str).to_numpy().astype(str)
        results[f"{name}_keys1"] = raw.obs[by[1]].astype(str).to_numpy().astype(str)
        results[f"{name}_n_cells"] = raw.obs["n_obs_aggregated"].to_numpy()
        results[f"{name}_sum"] = np.asarray(raw.layers["sum"])
        results[f"{name}_count_nonzero"] = np.asarray(raw.layers["count_nonzero"])
        results[f"{name}_mean_log"] = np.asarray(log.layers["mean"])
        results[f"{name}_var_log"] = np.asarray(log.layers["var"])
        results[f"{name}_mean_linear"] = np.asarray(linear.layers["mean"])

    # statistics.variance sums the float64 values exactly as fractions.
    results["stress_variance"] = np.array(
        [statistics.variance(row.tolist()) for row in data["stress"]]
    )
    np.savez(f"{directory}/reference.npz", **results)
    packages = ("scanpy", "anndata", "numpy", "scipy", "pandas")
    versions = {package: version(package) for package in packages}
    versions["python"] = sys.version.split()[0]
    with open(f"{directory}/versions.json", "w") as handle:
        json.dump(versions, handle)
    '''
)


def simulate(
    seed: int,
    samples: list[str],
    groups: list[str],
    cells_per_entry: np.ndarray,
    n_genes: int,
) -> dict[str, Any]:
    """Simulate negative binomial counts, one sample x group block at a time.

    Returns the counts as compressed sparse rows (cells x genes), the labels of
    every cell, and the cube layers built from the blocks with plain numpy sums.
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(np.log(0.15), 1.2, size=n_genes)
    base[:5] = 0.0  # genes that are never detected
    effect = rng.lognormal(0.0, 0.8, size=(len(groups), n_genes))
    size = 1.5
    shape = (len(groups), len(samples), n_genes)
    layers = {
        "counts": np.zeros(shape, dtype=np.int64),
        "detected": np.zeros(shape, dtype=np.int64),
        "log_sum": np.zeros(shape),
        "log_sum_sq": np.zeros(shape),
        "linear_sum": np.zeros(shape),
    }
    data, indices, row_lengths, sample_of_cell, group_of_cell = [], [], [], [], []
    for s, sample in enumerate(samples):
        for g, group in enumerate(groups):
            n = int(cells_per_entry[s, g])
            if n == 0:
                continue
            mean = base * effect[g] * rng.lognormal(0.0, 0.35, size=(n, 1))
            counts = rng.negative_binomial(size, size / (size + mean)).astype(np.int64)
            empty = np.flatnonzero(counts.sum(axis=1) == 0)
            counts[empty, rng.integers(5, n_genes, size=empty.size)] = 1
            linear = counts * TARGET_SUM / counts.sum(axis=1, keepdims=True)
            log = np.log1p(linear)
            layers["counts"][g, s] = counts.sum(axis=0)
            layers["detected"][g, s] = (counts > 0).sum(axis=0)
            layers["log_sum"][g, s] = log.sum(axis=0)
            layers["log_sum_sq"][g, s] = (log * log).sum(axis=0)
            layers["linear_sum"][g, s] = linear.sum(axis=0)
            rows, columns = np.nonzero(counts)
            data.append(counts[rows, columns].astype(np.int32))
            indices.append(columns.astype(np.int32))
            row_lengths.append(np.bincount(rows, minlength=n))
            sample_of_cell += [sample] * n
            group_of_cell += [group] * n
    lengths = np.concatenate(row_lengths)
    return {
        "data": np.concatenate(data),
        "indices": np.concatenate(indices),
        "indptr": np.concatenate(([0], np.cumsum(lengths))),
        "shape": np.array([lengths.size, n_genes]),
        "sample": np.array(sample_of_cell),
        "group": np.array(group_of_cell),
        "layers": layers,
    }


def cells_per_entry(
    rng: np.random.Generator, n_samples: int, n_cells: int
) -> np.ndarray:
    composition = rng.dirichlet(np.full(len(GROUPS), 2.0), size=n_samples)
    cells = np.array([rng.multinomial(n_cells // n_samples, p) for p in composition])
    cells[6, GROUPS.index("cDC")] = 0  # an empty entry
    cells[4, GROUPS.index("Plasma")] = 1  # one cell: no sample variance
    cells[7, GROUPS.index("Plasma")] = 2
    return cells


def stress_values(rng: np.random.Generator) -> np.ndarray:
    """Cells whose values have a large mean-to-standard-deviation ratio."""
    mean = 5.0
    return np.array(
        [
            mean + (mean / ratio) * rng.standard_normal(STRESS_CELLS)
            for ratio in STRESS_RATIOS
        ]
    )


def run_reference(
    directory: Path,
    simulated: dict[str, Any],
    labels: dict[str, np.ndarray],
    stress: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    np.savez(
        directory / "cells.npz",
        data=simulated["data"],
        indices=simulated["indices"],
        indptr=simulated["indptr"],
        shape=simulated["shape"],
        target_sum=TARGET_SUM,
        stress=stress,
        **labels,
    )
    program = directory / "reference.py"
    program.write_text(REFERENCE_PROGRAM)
    python = os.environ.get("REFERENCE_PYTHON", sys.executable)
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    subprocess.run(
        [python, "-I", str(program), str(directory)], check=True, env=environment
    )
    reference = dict(np.load(directory / "reference.npz"))
    versions = json.loads((directory / "versions.json").read_text())
    return reference, versions


def per_cell_labels(simulated: dict, donors: dict[str, str]) -> dict[str, np.ndarray]:
    coarse_of = {member: name for name, members in MERGES.items() for member in members}
    return {
        "sample": simulated["sample"],
        "group": simulated["group"],
        "donor": np.array([donors[sample] for sample in simulated["sample"]]),
        "coarse": np.array(
            [coarse_of.get(group, group) for group in simulated["group"]]
        ),
    }


def entries(cube: Cube, reference: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the cube's group and sample position of every reference row."""
    sample_ids = cube.samples["sample_id"].tolist()
    group_ids = cube.groups["group_id"].tolist()
    samples = np.array([sample_ids.index(key) for key in reference[f"{name}_keys0"]])
    groups = np.array([group_ids.index(key) for key in reference[f"{name}_keys1"]])
    return groups, samples


def relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    """Largest |actual - expected| / |expected|; where expected is 0, any difference fails."""
    difference = np.abs(actual - expected)
    scale = np.abs(expected)
    if np.any(difference[scale == 0] != 0):
        return float("inf")
    nonzero = scale > 0
    return float(np.max(difference[nonzero] / scale[nonzero], initial=0.0))


def compare_cube(cube: Cube, reference: dict, name: str, label: str) -> list[tuple]:
    """Compare every observed entry of a cube with scanpy's aggregate rows."""
    groups, samples = entries(cube, reference, name)
    n_cells = reference[f"{name}_n_cells"]
    several = n_cells >= 2
    cube_cells = cube.n_cells
    assert cube_cells is not None
    rows = [
        (
            f"{label}: counts, detected, n_cells, library sizes",
            "scanpy sum, count_nonzero, n_obs_aggregated",
            "max abs diff",
            max(
                np.abs(
                    cube.layer("counts")[groups, samples] - reference[f"{name}_sum"]
                ).max(),
                np.abs(
                    cube.layer("detected")[groups, samples]
                    - reference[f"{name}_count_nonzero"]
                ).max(),
                np.abs(cube_cells[samples, groups] - n_cells).max(),
                np.abs(
                    cube.library_sizes()[samples, groups]
                    - reference[f"{name}_sum"].sum(axis=1)
                ).max(),
            ),
            0,
        ),
        (
            f"{label}: mean, log scale",
            "scanpy mean of log1p(normalize_total)",
            "max rel diff",
            relative_difference(
                cube.mean()[groups, samples], reference[f"{name}_mean_log"]
            ),
            EXACT,
        ),
        (
            f"{label}: mean, linear scale",
            "scanpy mean of normalize_total",
            "max rel diff",
            relative_difference(
                cube.mean(scale="linear")[groups, samples],
                reference[f"{name}_mean_linear"],
            ),
            EXACT,
        ),
        (
            f"{label}: variance (entries with 2 or more cells)",
            "scanpy var, dof=1",
            "max rel diff",
            relative_difference(
                cube.variance()[groups[several], samples[several]],
                reference[f"{name}_var_log"][several],
            ),
            EXACT,
        ),
        (
            f"{label}: fraction_detected",
            "scanpy count_nonzero / n_obs_aggregated",
            "max rel diff",
            relative_difference(
                cube.fraction_detected()[groups, samples],
                reference[f"{name}_count_nonzero"] / n_cells[:, None],
            ),
            EXACT,
        ),
    ]
    # Entries without cells have no scanpy row; the cube must give NaN there.
    observed = np.zeros((cube.n_groups, cube.n_samples), dtype=bool)
    observed[groups, samples] = True
    empty_ok = bool(
        np.all(cube_cells.T[~observed] == 0)
        and np.isnan(cube.mean()[~observed]).all()
        and np.isnan(cube.variance()[~observed]).all()
        and np.isnan(cube.fraction_detected()[~observed]).all()
    )
    single = ~several
    single_ok = bool(
        np.isnan(reference[f"{name}_var_log"][single]).all()
        and np.isnan(cube.variance()[groups[single], samples[single]]).all()
    )
    n_empty, n_single = int((~observed).sum()), int(single.sum())
    if n_empty or n_single:
        rows.append(
            (
                f"{label}: {n_empty} empty and {n_single} single-cell entries",
                "scanpy: no row; var NaN",
                "NaN where expected",
                "yes" if empty_ok and single_ok else "no",
                "yes",
            )
        )
    return rows


def precision_rows(stress: np.ndarray, exact: np.ndarray) -> list[tuple]:
    log_sum = stress.sum(axis=1)
    log_sum_sq = (stress * stress).sum(axis=1)
    n = stress.shape[1]
    k = stress.shape[0]
    cube = Cube(
        {
            "counts": np.zeros((1, 1, k), dtype=np.int64),
            "log_sum": log_sum.reshape(1, 1, k),
            "log_sum_sq": log_sum_sq.reshape(1, 1, k),
        },
        samples={"sample_id": ["stress"]},
        groups={"group_id": ["stress"]},
        features={"feature_id": [f"r={ratio:g}" for ratio in STRESS_RATIOS]},
        n_cells=np.array([[n]]),
        normalization={"method": "none"},
    )
    variance = cube.variance("stress")[0]
    rows = []
    for ratio, value, truth in zip(STRESS_RATIOS, variance, exact, strict=True):
        bound = 64 * EPS * (1 + ratio**2)
        rows.append(
            (
                f"variance from raw moments, mean/sd = {ratio:g}"
                + (" (information only)" if bound > 0.5 else ""),
                "exact rational variance",
                "rel error",
                abs(value - truth) / truth,
                bound if bound <= 0.5 else None,
            )
        )
    return rows


def print_table(rows: list[tuple]) -> bool:
    print("| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    passed = True
    for check, reference, metric, value, tolerance in rows:
        shown = f"{value:.2g}" if isinstance(value, float) else str(value)
        if tolerance is None:  # shown for information, not checked
            print(f"| {check} | {reference} | {metric} | {shown} | none | info |")
            continue
        if isinstance(value, str):
            ok = value == tolerance
        else:
            ok = bool(value <= tolerance)
        limit = f"{tolerance:.2g}" if isinstance(tolerance, float) else str(tolerance)
        passed &= ok
        result = "pass" if ok else "FAIL"
        print(f"| {check} | {reference} | {metric} | {shown} | {limit} | {result} |")
    return passed


def build_cube(
    simulated: dict, samples: list[str], donors: dict[str, str], cells: np.ndarray
) -> Cube:
    return Cube(
        simulated["layers"],
        samples={"sample_id": samples, "donor": [donors[sample] for sample in samples]},
        groups={"group_id": GROUPS},
        features={"feature_id": [f"gene{j:04d}" for j in range(simulated["shape"][1])]},
        n_cells=cells,
        normalization={
            "method": "library_size",
            "target_sum": TARGET_SUM,
            "log": "natural",
            "pseudocount": 1.0,
        },
        provenance={"source": "validation/cube.py simulation", "seed": SEED},
    )


def run_case(
    seed: int, n_samples: int, n_cells: int, n_genes: int, directory: Path
) -> tuple[Cube, dict, dict, dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = [f"S{i + 1}" for i in range(n_samples)]
    donors = {sample: f"D{i // 2 + 1}" for i, sample in enumerate(samples)}
    cells = cells_per_entry(rng, n_samples, n_cells)
    simulated = simulate(seed + 1, samples, GROUPS, cells, n_genes)
    labels = per_cell_labels(simulated, donors)
    stress = stress_values(rng)
    reference, versions = run_reference(directory, simulated, labels, stress)
    cube = build_cube(simulated, samples, donors, cells)
    return cube, reference, versions, simulated, stress


def write_fixtures(directory: Path) -> None:
    """Write a small simulation and scanpy's aggregates of it for the tests."""
    rng = np.random.default_rng(SEED + 100)
    samples = ["S1", "S2", "S3", "S4"]
    donors = {"S1": "D1", "S2": "D1", "S3": "D2", "S4": "D2"}
    groups = GROUPS
    cells = rng.integers(2, 9, size=(len(samples), len(groups)))
    cells[1, groups.index("cDC")] = 0
    cells[2, groups.index("Plasma")] = 1
    simulated = simulate(SEED + 101, samples, groups, cells, 12)
    labels = per_cell_labels(simulated, donors)
    reference, _ = run_reference(
        directory, simulated, labels, stress_values(rng)[:, :10]
    )
    dense = np.zeros(tuple(simulated["shape"]), dtype=np.int32)
    rows = np.repeat(np.arange(simulated["shape"][0]), np.diff(simulated["indptr"]))
    dense[rows, simulated["indices"]] = simulated["data"]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    reference.pop("stress_variance")
    coarse_of = {member: name for name, members in MERGES.items() for member in members}
    np.savez_compressed(
        FIXTURE_DIR / "scanpy_aggregate.npz",
        counts=dense,
        sample=labels["sample"],
        group=labels["group"],
        sample_ids=np.array(samples),
        donor_of_sample=np.array([donors[sample] for sample in samples]),
        group_ids=np.array(groups),
        coarse_of_group=np.array([coarse_of.get(group, group) for group in groups]),
        target_sum=TARGET_SUM,
        **reference,
    )
    print(f"wrote {FIXTURE_DIR / 'scanpy_aggregate.npz'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cube, reference, versions, simulated, stress = run_case(
            SEED, n_samples=8, n_cells=30_000, n_genes=2_000, directory=directory
        )
        if arguments.write_fixtures:
            write_fixtures(directory)
    print(
        "Reference versions: "
        + ", ".join(f"{key} {value}" for key, value in versions.items())
    )
    print(f"CytePool side: numpy {np.__version__}")
    density = simulated["data"].size / np.prod(simulated["shape"])
    print(
        f"Data: {simulated['shape'][0]:,} cells x {simulated['shape'][1]:,} genes, "
        f"density {density:.3f}, {cube.n_samples} samples x {cube.n_groups} cell types; "
        f"regroup to {len(MERGES) + 1} groups, collapse to 4 donors; seed {SEED}"
    )
    print()
    rows = compare_cube(cube, reference, "fine", "cube")
    rows += compare_cube(
        cube.regroup(MERGES), reference, "coarse", "regroup 10 to 4 groups"
    )
    rows += compare_cube(
        cube.collapse_samples("donor"), reference, "donor", "collapse 8 to 4 donors"
    )
    rows += precision_rows(stress, reference["stress_variance"])
    passed = print_table(rows)
    print()
    print(
        "float64 contract: the variance from raw moments has relative error of about "
        "eps * (1 + mean**2 / variance); log-normalized expression stays in the exact tier."
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
