"""Check reads of selected samples and features against zarr-python, and measure them.

1. zarr-python's orthogonal indexing (``Array.oindex``) and ``ZarrArray.oindex`` read
   the same elements, for selections in increasing and other orders, with repeated
   and negative positions, boolean masks, empty selections and selections across
   chunk boundaries: on an array zarr-python writes (chunks along every axis, one of
   them not stored) and on the layers of a realistic cube file CytePool writes.
2. Cubes selected from that file give, for every layer, the values zarr-python reads
   at the same groups, samples and features, and merging groups or pooling samples of
   a selection gives the sums of those values.
3. Reading a selection opens exactly the chunk files that hold a selected feature of a
   selected group, each once. For each selection the script reports the chunk files
   opened, the bytes read from disk and decoded, the size of the result, the peak of
   Python's allocations (tracemalloc) and the time to read it into memory (the best
   of three reads after a first one).

The cube file has 20 groups, 100 samples and 30,000 genes: counts as uint32 and
log_sum as float64, gzip level 4. ``simulated_cube`` and ``measurement_table`` use only
functions that earlier versions of CytePool have, so the same measurements can be
taken of them. The reference runs as a separate program and exchanges data with this
script through files in a temporary directory.

Requirements:
    This script: numpy, and cytepool from the source tree.
    Reference program: zarr-python 3 and numpy. It runs as a separate process of the
    same Python executable unless REFERENCE_PYTHON names another one.

Usage, from the repository root:
    PYTHONPATH=. python validation/cube_reads.py
    PYTHONPATH=. python validation/cube_reads.py --write-fixtures
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.cube import Cube
from cytepool.io import read_cube, write_cube
from cytepool.io.zarr import open_array

SEED = 20261024
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "cube_reads"
SHAPE = (20, 100, 30_000)  # groups, samples, genes
SMALL = (3, 5, 11)  # the array zarr-python writes, in chunks of (2, 2, 4)
LAYERS = ("counts", "log_sum")
MERGED = ["type 3", "type 8", "type 11"]

REFERENCE_PROGRAM = textwrap.dedent(
    '''
    """zarr-python writes an array and reads orthogonal selections of arrays."""

    import json
    import sys
    from importlib.metadata import version
    from pathlib import Path

    import numpy as np
    import zarr
    from zarr.codecs import GzipCodec

    work = Path(sys.argv[1])
    plan = json.loads((work / "plan.json").read_text())

    def item(value):
        if isinstance(value, dict):
            if "slice" in value:
                return slice(*value["slice"])
            if "mask" in value:
                return np.array(value["mask"], dtype=bool)
            return np.array(value["positions"], dtype=np.int64)
        return value

    small = np.load(work / "small.npy")
    zarr.create_array(
        store=work / "small.zarr", shape=small.shape, chunks=(2, 2, 4),
        dtype=small.dtype, fill_value=7, compressors=[GzipCodec(level=1)],
    )[...] = small
    results = {}
    for case in plan:
        array = zarr.open_array(work / case["array"], mode="r")
        key = tuple(item(value) for value in case["key"])
        results[case["name"]] = np.asarray(array.oindex[key])
    np.savez(work / "reference.npz", **results)
    versions = {package: version(package) for package in ("zarr", "numpy")}
    versions["python"] = sys.version.split()[0]
    (work / "versions.json").write_text(json.dumps(versions))
    '''
)


class SimulatedLayer:
    """A layer of the simulated cube, made one group at a time from fixed seeds."""

    def __init__(self, name: str, n_cells: np.ndarray) -> None:
        self.name, self.n_cells, self.shape = name, n_cells, SHAPE
        self.dtype = np.dtype(np.uint32 if name == "counts" else np.float64)

    def __getitem__(self, group: int) -> np.ndarray:
        rng = np.random.default_rng([SEED, group])
        cells = self.n_cells[:, group, None]
        rate = rng.lognormal(-3.0, 2.0, SHAPE[2])  # genes from rare to common
        counts = rng.poisson(cells * rate).astype(np.uint32)
        if self.name == "counts":
            return counts
        return np.where(cells > 0, np.log1p(counts) + rng.random(counts.shape), 0.0)


def simulated_cube() -> Cube:
    """The cube of the measurements, with layers made as it is written."""
    rng = np.random.default_rng(SEED)
    n_cells = rng.integers(0, 400, size=(SHAPE[1], SHAPE[0]))
    n_cells[rng.random(n_cells.shape) < 0.05] = 0  # entries without cells
    return Cube(
        {name: SimulatedLayer(name, n_cells) for name in LAYERS},
        samples={
            "sample_id": [f"S{i:03d}" for i in range(SHAPE[1])],
            "donor": [f"D{i // 4:02d}" for i in range(SHAPE[1])],
        },
        groups={"group_id": [f"type {g}" for g in range(SHAPE[0])]},
        features={"feature_id": [f"ENSG{j:011d}" for j in range(SHAPE[2])]},
        n_cells=n_cells,
        normalization={"method": "library_size", "target_sum": 1e4, "log": "natural"},
    )


def selections() -> dict[str, dict[str, list[int]]]:
    """Selections of the cube file: positions of groups, samples and genes."""
    rng = np.random.default_rng(SEED + 1)
    genes = SHAPE[2]
    return {
        "50 neighbouring genes": {"features": list(range(10_000, 10_050))},
        "50 genes at random": {"features": sorted(rng.choice(genes, 50, False))},
        "2,000 genes at random": {"features": sorted(rng.choice(genes, 2000, False))},
        "50 genes at random, 2 groups": {
            "features": rng.choice(genes, 50, False).tolist(),  # in any order
            "groups": [7, 2],
        },
        "10 samples, every gene": {"samples": sorted(rng.choice(SHAPE[1], 10, False))},
        "one whole group": {"groups": [5]},
    }


def orthogonal_keys(shape: tuple[int, ...]) -> list[list[Any]]:
    """Orthogonal selections of a 3-D array of this shape, as JSON items."""
    g, s, f = shape
    every = {"slice": [None]}
    return [
        [0, every, {"positions": [f - 1, 0, 3, f // 2]}],
        [{"positions": [g - 1, 0]}, {"positions": [1, 1, 0]}, every],
        [every, {"slice": [1, None, 2]}, {"positions": [-1, -f, 2]}],
        [1, {"mask": [i % 3 != 1 for i in range(s)]}, {"positions": [2, 3, 4, 5, 6]}],
        [{"positions": []}, 0, {"positions": [0]}],
        [{"positions": [g - 1]}, {"positions": [s - 1, 0]}, f - 2],
        # In the small array, this one lies in the chunk that is not stored.
        [{"positions": [g - 1]}, {"slice": [2, 4]}, {"positions": [7, 4]}],
    ]


def item(value: Any) -> Any:
    """Turn a JSON item into an index: an integer, a slice, an array or a mask."""
    if isinstance(value, dict):
        if "slice" in value:
            return slice(*value["slice"])
        if "mask" in value:
            return np.array(value["mask"], dtype=bool)
        return np.array(value["positions"], dtype=np.int64)
    return value


def differences(actual: Any, expected: Any) -> int:
    """Count elements that differ, comparing bit patterns (NaN payloads, -0.0)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    native = expected.dtype.newbyteorder("=")
    if actual.shape != expected.shape or actual.dtype.newbyteorder("=") != native:
        return max(actual.size, expected.size, 1)
    bits = [np.ascontiguousarray(a, native).view(np.uint8) for a in (actual, expected)]
    return int((bits[0] != bits[1]).reshape(-1, native.itemsize).any(axis=1).sum())


def relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        return np.inf
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return float((np.abs(actual - expected) / scale).max(initial=0.0))


_opened: list[str] | None = None


def _record_open(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and _opened is not None:
        _opened.append(str(args[0]))


sys.addaudithook(_record_open)


def measure(path: Path, selection: dict[str, list[int]]) -> dict[str, Any]:
    """Read every layer of a selection of a cube file into memory, and measure it."""
    global _opened
    chosen = read_cube(path).select(**selection)
    _opened = []
    values = chosen.to_memory()
    opened, _opened = [p for p in _opened if f"{os.sep}layers{os.sep}" in p], None
    seconds = []
    for _ in range(3):
        start = time.perf_counter()
        chosen.to_memory()
        seconds.append(time.perf_counter() - start)
    tracemalloc.start()
    chosen.to_memory()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    stored = [p for p in opened if os.path.exists(p)]  # others read as zeros
    decoded = 0
    for name in values.layer_names:
        array = open_array(path / "layers" / name)
        chunk = int(np.prod(array.chunks)) * array.dtype.itemsize
        decoded += chunk * sum(f"{os.sep}{name}{os.sep}" in p for p in stored)
    return {
        "opened": opened,
        "read": sum(os.path.getsize(p) for p in stored),
        "decoded": decoded,
        "result": sum(values.layer(name).nbytes for name in values.layer_names),
        "peak": peak,
        "seconds": min(seconds),
        "cube": values,
    }


def measurement_table(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Measure the reads of every selection; return a markdown table and the data."""
    lines = [
        "| selection | chunk files opened | read from disk (MB) | decoded (MB) | "
        "result (MB) | peak memory (MB) | time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    measured = {}
    for name, selection in selections().items():
        result = measured[name] = measure(path, selection)
        mb = {key: f"{result[key] / 1e6:,.1f}" for key in ("read", "decoded", "peak")}
        lines.append(
            f"| {name} | {len(result['opened'])} | {mb['read']} | {mb['decoded']} | "
            f"{result['result'] / 1e6:,.2f} | {mb['peak']} | {result['seconds']:.2f} |"
        )
    return "\n".join(lines), measured


def expected_chunks(path: Path, selection: dict[str, list[int]]) -> list[str]:
    """The layers' chunk files that hold a selected feature of a selected group."""
    width = open_array(path / "layers" / "counts").chunks[2]
    groups = selection.get("groups", range(SHAPE[0]))
    chunks = {j // width for j in selection.get("features", range(SHAPE[2]))}
    root = path / "layers"
    return sorted(
        str(root / name / "c" / str(g) / "0" / str(j))
        for name in LAYERS
        for g in groups
        for j in chunks
    )


def run_reference(work: Path, store: Path, small: np.ndarray) -> tuple[dict, dict]:
    """Have zarr-python read the selections; return its versions and results."""
    np.save(work / "small.npy", small)
    arrays = [("small", "small.zarr", SMALL)]
    arrays += [(name, str(store / "layers" / name), SHAPE) for name in LAYERS]
    plan = [
        {"name": f"{name}:{k}", "array": array, "key": key}
        for name, array, shape in arrays
        for k, key in enumerate(orthogonal_keys(shape))
    ]
    for name, selection in selections().items():  # a cube selection, axis by axis
        key = [
            {"positions": selection[axis]} if axis in selection else {"slice": [None]}
            for axis in ("groups", "samples", "features")
        ]
        for layer in LAYERS:
            array = str(store / "layers" / layer)
            plan.append({"name": f"{name}:{layer}", "array": array, "key": key})
    (work / "plan.json").write_text(json.dumps(plan, default=int))
    (work / "reference.py").write_text(REFERENCE_PROGRAM)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    python = os.environ.get("REFERENCE_PYTHON", sys.executable)
    subprocess.run(
        [python, "-I", str(work / "reference.py"), str(work)],
        check=True,
        env=environment,
    )
    with np.load(work / "reference.npz") as data:
        results = dict(data)
    return json.loads((work / "versions.json").read_text()), results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixtures", action="store_true")
    arguments = parser.parse_args()
    small = np.random.default_rng(SEED).integers(0, 1000, SMALL).astype(np.int32)
    small[2:, 2:4, 4:8] = 7  # a chunk equal to the fill value, which is not stored
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        store = work / "simulated.cube.zarr"
        write_cube(simulated_cube(), store)
        versions, reference = run_reference(work, store, small)

        bad = cases = 0
        for name, array in [("small", work / "small.zarr")] + [
            (layer, store / "layers" / layer) for layer in LAYERS
        ]:
            for k, key in enumerate(orthogonal_keys(open_array(array).shape)):
                found = open_array(array).oindex[tuple(item(value) for value in key)]
                bad, cases = (
                    bad + differences(found, reference[f"{name}:{k}"]),
                    cases + 1,
                )
        check = f"ZarrArray.oindex, {cases} selections of a zarr-python array and of "
        check += "the cube file's layers"
        rows.append((check, "elements that differ", bad, "0", bad == 0))

        table, measured = measurement_table(store)
        bad = sum(
            differences(result["cube"].layer(layer), reference[f"{name}:{layer}"])
            for name, result in measured.items()
            for layer in LAYERS
        )
        check = f"Cube.select of the cube file, {len(measured)} selections, every layer"
        rows.append((check, "elements that differ", bad, "0", bad == 0))
        bad = sum(
            sorted(result["opened"]) != expected_chunks(store, selections()[name])
            for name, result in measured.items()
        )
        check = "chunk files each selection opens, against the chunks that hold it"
        rows.append((check, "selections with other files", bad, "0", bad == 0))

        genes = selections()["50 genes at random"]["features"]
        chosen = read_cube(store).select(features=genes)
        merged = chosen.regroup({"merged": MERGED})
        pooled = chosen.collapse_samples("donor")
        worst = {"counts": 0.0, "log_sum": 0.0}
        for layer in LAYERS:
            block = reference[f"50 genes at random:{layer}"]
            wide = block.astype(np.int64 if layer == "counts" else np.float64)
            members = [int(group.split()[1]) for group in MERGED]
            sums = [
                (merged.layer(layer, "merged"), wide[members].sum(axis=0)),
                (pooled.layer(layer), wide.reshape(SHAPE[0], -1, 4, 50).sum(axis=2)),
            ]
            for found, expected in sums:
                worst[layer] = max(worst[layer], relative_difference(found, expected))
        check = "merged groups and pooled samples of a 50-gene selection, counts"
        rows.append(
            (
                check,
                "largest relative difference",
                worst["counts"],
                "0",
                worst["counts"] == 0,
            )
        )
        check = check.replace("counts", "log_sum")
        rows.append(
            (
                check,
                "largest relative difference",
                f"{worst['log_sum']:.1e}",
                "1e-12",
                worst["log_sum"] <= 1e-12,
            )
        )
        if arguments.write_fixtures:
            write_fixtures(work, reference)

    print("Reference: " + ", ".join(f"{key} {item}" for key, item in versions.items()))
    print(f"CytePool: Python {sys.version.split()[0]}, numpy {np.__version__}")
    print(f"Machine: {platform.processor() or platform.machine()}, {platform.system()}")
    print("\n| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    reference_name = f"zarr-python {versions['zarr']}"
    for check, metric, value, tolerance, ok in rows:
        result = "pass" if ok else "FAIL"
        print(
            f"| {check} | {reference_name} | {metric} | {value} | {tolerance} | "
            f"{result} |"
        )
    print(
        f"\nReads of {SHAPE[0]} groups x {SHAPE[1]} samples x {SHAPE[2]:,} genes "
        "(counts and log_sum; every layer of the selection, into memory):\n"
    )
    print(table)
    return 0 if all(ok for *_, ok in rows) else 1


def write_fixtures(work: Path, reference: dict[str, np.ndarray]) -> None:
    """Save the small array zarr-python wrote, and what its selections read."""
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    shutil.copytree(work / "small.zarr", FIXTURE_DIR / "small.zarr")
    keys = orthogonal_keys(SMALL)
    results = {f"key {k}": reference[f"small:{k}"] for k in range(len(keys))}
    np.savez_compressed(FIXTURE_DIR / "oindex.npz", **results)
    lines = ",\n".join(f"  {json.dumps(key)}" for key in keys)  # a key per line
    (FIXTURE_DIR / "keys.json").write_text(f"[\n{lines}\n]\n")
    print(f"wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
