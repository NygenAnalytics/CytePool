"""Check cube files against zarr-python, an independent Zarr v3 implementation.

1. zarr-python opens the cube files CytePool writes (gzip, uncompressed, zstd) and,
   following the documented layout, gets the same tables, cell numbers, layers and
   attributes, bit for bit.
2. CytePool reads arrays zarr-python writes: every core data type, chunk shapes with
   partial edge chunks, chunks left unstored because they equal the fill value (such
   as NaN, -Infinity, 7 or true), both byte orders, gzip, zstd, gzip then zstd and no
   compression, and the default (both separators) and v2 chunk key encodings.
3. zarr-python copies cube files node by node with other chunks, big-endian bytes and
   other compression, and read_cube returns the original cube.

zstd needs Python 3.14 or newer (compression.zstd) here; without it the zstd cases are
skipped, as the table says. The reference runs as a separate program and exchanges
data with this script through files in a temporary directory.

Requirements:
    This script: numpy, and cytepool from the source tree (Python 3.14 or newer for
    the zstd cases).
    Reference program: zarr-python 3 and numpy. It runs in the same interpreter unless
    REFERENCE_PYTHON names another Python executable.

Usage, from the repository root:
    PYTHONPATH=. python validation/cube_files.py
    PYTHONPATH=. python validation/cube_files.py --write-fixtures
"""

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from cytepool import Cube, read_cube, write_cube
from cytepool.io.zarr import DATA_TYPES, open_array

SEED = 20261023
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "cube_files"
try:
    importlib.import_module("compression.zstd")
    HAS_ZSTD = True
except ImportError:  # Python 3.13, or a build without zstd
    HAS_ZSTD = False
COMPRESSIONS = ["gzip", None] + ["zstd"] * HAS_ZSTD
# Arrays zarr-python writes: every data type meets every (shape, chunks), with codec
# chains (byte order, compressors) and chunk key encodings in rotation. The arrays of
# shape 0 go into the test fixture.
SHAPES = [((5, 3), (2, 2)), ((37, 23), (8, 5)), ((37, 23), (64, 64))]
SHAPES += [((5, 7, 11), (2, 3, 4)), ((), ()), ((0, 4), (1, 4))]
CHAINS = [("little", ["gzip:1"]), ("big", []), ("little", ["zstd:3:checksum"])]
CHAINS += [("little", []), ("big", ["gzip:9", "zstd:1"])]
KEYS = [("default", "/"), ("default", "."), ("v2", "."), ("v2", "/")]

REFERENCE_PROGRAM = textwrap.dedent(
    '''
    """zarr-python reads cube files, writes arrays and copies cube files."""

    import json
    import sys
    from importlib.metadata import version
    from pathlib import Path

    import numpy as np
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec, ZstdCodec

    work = Path(sys.argv[1])
    plan = json.loads((work / "plan.json").read_text())

    def compressors(names):
        result = []
        for name in names:
            kind, level, *checksum = name.split(":")
            if kind == "gzip":
                result.append(GzipCodec(level=int(level)))
            else:
                result.append(ZstdCodec(level=int(level), checksum=bool(checksum)))
        return result

    # 1. Read cube files as their documentation describes them.
    for name in plan["read"]:
        root = zarr.open_group(work / "cytepool" / name, mode="r")
        arrays, text = {}, {"attributes": dict(root.attrs)}
        nodes = [("n_cells", root["n_cells"])] if root.attrs["has_n_cells"] else []
        layers = root.attrs["layers"]
        nodes += [(f"layer:{layer}", root["layers"][layer]) for layer in layers]
        for axis in ("samples", "groups", "features"):
            text[axis] = root[axis].attrs["columns"]
            nodes += [
                (f"{axis}:{position}", root[axis][node])
                for position, node in enumerate(root[axis].attrs["arrays"])
            ]
        for key, array in nodes:
            values = array[...]
            text[key + ":dims"] = list(array.metadata.dimension_names)
            if "levels" in array.attrs:
                levels = array.attrs["levels"]
                text[key] = [None if c < 0 else levels[c] for c in values.tolist()]
            elif "numpy_dtype" in array.attrs:
                arrays[key] = values.view(array.attrs["numpy_dtype"])
            else:
                arrays[key] = values
        np.savez(work / "read" / f"{name}.npz", **arrays)
        (work / "read" / f"{name}.json").write_text(json.dumps(text))

    # 2. Write arrays from the script's data, and count the chunks actually stored.
    inputs = np.load(work / "inputs.npz")
    stored = {}
    for case in plan["write"]:
        data, path = inputs[case["name"]], work / "zarr" / case["name"]
        zarr.create_array(
            store=path, shape=data.shape, chunks=tuple(case["chunks"]),
            dtype=data.dtype, fill_value=inputs[case["name"] + ":fill"][()],
            serializer=BytesCodec(
                endian=case["endian"] if data.dtype.itemsize > 1 else None
            ),
            compressors=compressors(case["compressors"]),
            chunk_key_encoding={"name": case["key"], "separator": case["separator"]},
        )[...] = data
        files = [p for p in path.rglob("*") if p.is_file() and p.name != "zarr.json"]
        stored[case["name"]] = len(files)
    (work / "stored.json").write_text(json.dumps(stored))

    # 3. Copy cube files node by node, with other chunks and codecs.
    def copy(source, target, names):
        for key, node in source.members():
            if isinstance(node, zarr.Group):
                copy(node, target.create_group(key, attributes=dict(node.attrs)), names)
                continue
            target.create_array(
                key, shape=node.shape, dtype=node.dtype, fill_value=node.fill_value,
                chunks=tuple(max(1, -(-n // 2)) for n in node.shape),
                serializer=BytesCodec(
                    endian="big" if node.dtype.itemsize > 1 else None
                ),
                compressors=compressors(names), attributes=dict(node.attrs),
                dimension_names=node.metadata.dimension_names,
            )[...] = node[...]

    for case in plan["copy"]:
        source = zarr.open_group(work / "cytepool" / case["source"], mode="r")
        target = zarr.create_group(
            work / "copies" / case["source"], attributes=dict(source.attrs)
        )
        copy(source, target, case["compressors"])

    versions = {package: version(package) for package in ("zarr", "numcodecs", "numpy")}
    versions["python"] = sys.version.split()[0]
    (work / "versions.json").write_text(json.dumps(versions))
    '''
)


def simulate_cube(
    rng: np.random.Generator, n_features: int, n_samples: int, n_groups: int
) -> Cube:
    """A cube with every layer, columns of many types and a group without cells."""
    shape = (n_groups, n_samples, n_features)
    days = rng.integers(0, 400, n_samples).astype("m8[D]")
    visits = np.datetime64("2024-01-01") + days
    visits[1] = np.datetime64("NaT", "D")
    samples = {
        "sample_id": [f"S{i:02d}" for i in range(n_samples)],
        "donor": [None if i % 5 == 3 else f"D{i // 2}" for i in range(n_samples)],
        "age": np.where(np.arange(n_samples) % 4 == 1, np.nan, 50.0),
        "batch": (np.arange(n_samples) % 3).astype(np.int16),
        "visit": visits,
        # Cube files store this column and the next two under positional array names.
        "Treated?": rng.random(n_samples) < 0.5,
        "CD4/CD8 ratio": rng.lognormal(0, 0.5, n_samples).astype(np.float32),
        "Größe": rng.integers(0, 255, n_samples).astype(np.uint8),
    }
    groups = {
        "group_id": [f"type {g}" for g in range(n_groups)],
        "cell_ontology_id": [
            None if g == 1 else f"CL:{g:07d}" for g in range(n_groups)
        ],
        "score": rng.standard_normal(n_groups) + 1j * rng.standard_normal(n_groups),
        "lifetime": rng.integers(1, 99, n_groups).astype("m8[h]"),
    }
    features = {
        "feature_id": [f"ENSG{j:011d}" for j in range(n_features)],
        "symbol": [None if j % 50 == 7 else f"GENE{j}" for j in range(n_features)],
        "measured": np.arange(n_features) % 97 != 0,
        "gc": rng.uniform(0.3, 0.7, n_features).astype(np.float16),
        "length": rng.integers(200, 90_000, n_features).astype(np.uint32),
    }
    cells = rng.integers(0, 300, size=(n_samples, n_groups))
    cells[:, -1] = 0  # a group without cells: its chunks hold zeros and are not stored
    cells[0, 0] = 1
    n = cells.T[:, :, None] * np.ones(shape)
    mean = rng.lognormal(0.0, 1.0, size=shape)
    spread = np.where(n > 0, (n - 1) * rng.lognormal(-1, 1, shape), 0.0)
    layers = {
        "counts": rng.poisson(mean * n / 10).astype(np.int32),
        "detected": rng.binomial(n.astype(np.int64), 0.3).astype(np.int16),
        "log_sum": n * mean,
        "log_sum_sq": spread + n * mean**2,  # raw moments of values with this spread
        "linear_sum": n * rng.lognormal(1.0, 1.0, size=shape),
    }
    normalization = {"method": "library_size", "target_sum": 1e4, "log": "natural"}
    provenance = {"source": "validation/cube_files.py", "seed": SEED, "note": "ünï"}
    tables = {"samples": samples, "groups": groups, "features": features}
    return Cube(
        layers,
        **tables,
        n_cells=cells,
        normalization=normalization,
        provenance=provenance,
    )


def simulate_cubes(rng: np.random.Generator) -> dict[str, Cube]:
    """A realistic cube, a bulk cube, a cube with empty axes and a tiny cube."""
    bulk = Cube(
        {"counts": rng.gamma(2.0, 50.0, size=(1, 40, 2000))},  # non-integer counts
        samples={"sample_id": [f"B{i}" for i in range(40)], "site": ["x", "y"] * 20},
        groups={"group_id": ["bulk"]},
        features={"feature_id": [f"g{j}" for j in range(2000)]},
        provenance={"source": "bulk RNA-seq"},
    )
    empty = Cube(
        {"counts": np.zeros((2, 0, 0), dtype=np.int64)},
        samples={"sample_id": [], "donor": []},
        groups={"group_id": ["a", "b"]},
        features={"feature_id": []},
        n_cells=np.zeros((0, 2), dtype=np.int64),
    )
    full, tiny = simulate_cube(rng, 5000, 12, 6), simulate_cube(rng, 7, 4, 3)
    return {"full": full, "bulk": bulk, "empty": empty, "tiny": tiny}


def contents(cube: Cube) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A cube's arrays and text, keyed as the reference program reports a cube file."""
    attributes = {
        "format": "cytepool.cube",
        "format_version": 1,
        "axes": ["groups", "samples", "features"],
        "layers": list(cube.layer_names),
        "has_n_cells": cube.n_cells is not None,
        "normalization": cube.normalization,
        "provenance": cube.provenance,
        "digests": cube.digests(),
    }
    arrays: dict[str, np.ndarray] = {}
    text: dict[str, Any] = {"attributes": attributes}
    if cube.n_cells is not None:
        arrays["n_cells"] = cube.n_cells.astype(np.int32)  # stored as int32
        text["n_cells:dims"] = ["samples", "groups"]
    for name in cube.layer_names:
        arrays[f"layer:{name}"] = cube.layer(name)
        text[f"layer:{name}:dims"] = ["groups", "samples", "features"]
    for axis in ("samples", "groups", "features"):
        table = getattr(cube, axis)
        text[axis] = list(table.columns)
        for position, name in enumerate(table.columns):
            key = f"{axis}:{position}"
            text[key + ":dims"] = [axis]
            if table[name].dtype.kind == "O":
                text[key] = table[name].tolist()
            else:
                arrays[key] = table[name]
    return arrays, text


def differences(actual: np.ndarray, expected: np.ndarray) -> int:
    """Count elements that differ, comparing bit patterns (NaN payloads, -0.0)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    native = expected.dtype.newbyteorder("=")
    if actual.shape != expected.shape or actual.dtype.newbyteorder("=") != native:
        return max(actual.size, expected.size, 1)
    bits = [
        np.frombuffer(np.asarray(a, native).tobytes(), np.uint8)
        for a in (actual, expected)
    ]
    return int((bits[0] != bits[1]).reshape(-1, native.itemsize).any(axis=1).sum())


def mismatches(found: tuple[dict, dict], expected: tuple[dict, dict]) -> int:
    """Count differing array elements and text entries, and missing or extra keys."""
    (arrays, text), (expected_arrays, expected_text) = found, expected
    bad = len(set(arrays) ^ set(expected_arrays)) + len(set(text) ^ set(expected_text))
    bad += sum(text.get(key) != value for key, value in expected_text.items())
    shared = [key for key in expected_arrays if key in arrays]
    return bad + sum(differences(arrays[key], expected_arrays[key]) for key in shared)


def array_cases(rng: np.random.Generator) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Arrays for zarr-python to write: random data, some chunks equal to the fill."""
    cases, inputs = [], {}
    for t, (type_name, dtype) in enumerate(DATA_TYPES.items()):
        for s, (shape, chunks) in enumerate(SHAPES):
            if dtype.kind == "b":
                data, fill = np.array(rng.random(shape) < 0.5), np.array(s % 2 == 0)
            elif dtype.kind in "iu":
                low, high = np.iinfo(dtype).min, np.iinfo(dtype).max
                data = np.array(rng.integers(low, high, shape, dtype, endpoint=True))
                fill = np.array([0, 7][(t + s) % 2], dtype=dtype)
            else:
                data = np.array(rng.standard_normal(shape) * 10, dtype=dtype)
                if dtype.kind == "c":
                    data += 1j * np.array(rng.standard_normal(shape), dtype=dtype)
                data.flat[:: max(1, data.size // 3)] = [np.nan, -0.0, np.inf][s % 3]
                fill = np.array([np.nan, -np.inf, 1.5][(t + s) % 3], dtype=dtype)
            for index in np.ndindex(*(math.ceil(n / c) for n, c in zip(shape, chunks))):
                region = tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks))
                if rng.random() < 0.35:  # a chunk equal to the fill value
                    data[region] = fill
            name = f"{type_name}-{s}"
            endian, compressors = CHAINS[(t + s) % len(CHAINS)]
            key, separator = KEYS[(t + 2 * s) % len(KEYS)]
            case = dict(name=name, chunks=chunks, endian=endian, key=key)
            cases.append(case | {"compressors": compressors, "separator": separator})
            inputs[name], inputs[name + ":fill"] = data, fill
    return cases, inputs


def run_reference(
    work: Path, cubes: dict[str, Cube], cases: list[dict], inputs: dict
) -> tuple[dict, dict]:
    """Write the cube files, run the reference program, and return its results."""
    for folder in ("cytepool", "read", "zarr", "copies"):
        (work / folder).mkdir()
    for name, cube in cubes.items():
        for compression in COMPRESSIONS:
            store = work / "cytepool" / f"{name}-{compression}"
            write_cube(cube, store, compression=compression)
    copies = [
        {"source": f"{name}-gzip", "compressors": ["gzip:1"] + ["zstd:1"] * zstd}
        for name, zstd in (("full", HAS_ZSTD), ("bulk", HAS_ZSTD), ("tiny", False))
    ]
    stores = [f"{name}-{compression}" for name in cubes for compression in COMPRESSIONS]
    plan = {"read": stores, "write": cases, "copy": copies}
    np.savez(work / "inputs.npz", **inputs)
    (work / "plan.json").write_text(json.dumps(plan))
    (work / "reference.py").write_text(REFERENCE_PROGRAM)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    python = os.environ.get("REFERENCE_PYTHON", sys.executable)
    subprocess.run(
        [python, "-I", str(work / "reference.py"), str(work)],
        check=True,
        env=environment,
    )
    versions = json.loads((work / "versions.json").read_text())
    return versions, json.loads((work / "stored.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixtures", action="store_true")
    arguments = parser.parse_args()
    rng = np.random.default_rng(SEED)
    cubes = simulate_cubes(rng)
    cases, inputs = array_cases(rng)
    skipped = "" if HAS_ZSTD else "; zstd skipped"
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        versions, stored = run_reference(work, cubes, cases, inputs)
        for name, cube in cubes.items():
            bad = 0
            for compression in COMPRESSIONS:
                path = work / "read" / f"{name}-{compression}"
                arrays = dict(np.load(path.with_suffix(".npz")))
                text = json.loads(path.with_suffix(".json").read_text())
                bad += mismatches((arrays, text), contents(cube))
            sizes = f"{cube.n_features:,} x {cube.n_samples} x {cube.n_groups}"
            shown = ", ".join(str(c).lower() for c in COMPRESSIONS) + skipped
            check = f"zarr-python reads cube '{name}' (genes x samples x groups "
            check += f"{sizes}; {shown})"
            rows.append(
                (check, "values and attributes that differ", bad, "0", bad == 0)
            )

        bad = missing = chunks = read = 0
        for case in cases:
            if any(c.startswith("zstd") for c in case["compressors"]) and not HAS_ZSTD:
                continue
            data, array = inputs[case["name"]], open_array(work / "zarr" / case["name"])
            grid = math.prod(
                math.ceil(n / c) for n, c in zip(data.shape, case["chunks"])
            )
            chunks, read = chunks + (grid if data.size else 0), read + 1
            missing += (grid if data.size else 0) - stored[case["name"]]
            bad += differences(np.asarray(array), data)
        check = f"CytePool reads {read} arrays zarr-python wrote{skipped}"
        rows.append((check, "elements that differ", bad, "0", bad == 0))
        check = f"chunks of those arrays not stored, as equal to the fill (of {chunks})"
        rows.append(
            (check, "chunks read as the fill value", missing, "> 0", missing > 0)
        )

        bad = 0
        for name in ("full", "bulk", "tiny"):
            copied = read_cube(work / "copies" / f"{name}-gzip")
            bad += mismatches(contents(copied), contents(cubes[name]))
        codecs = "gzip + zstd" if HAS_ZSTD else "gzip"
        check = f"read_cube of zarr-python copies of 'full', 'bulk', 'tiny' ({codecs})"
        rows.append((check, "values that differ", bad, "0", bad == 0))
        if arguments.write_fixtures:
            write_fixtures(work, cases, inputs, cubes["tiny"])

    print("Reference: " + ", ".join(f"{key} {item}" for key, item in versions.items()))
    python = (
        f"Python {sys.version.split()[0]} ({'with' if HAS_ZSTD else 'without'} zstd)"
    )
    print(f"CytePool: {python}, numpy {np.__version__}")
    print("\n| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    reference = f"zarr-python {versions['zarr']}"
    for check, metric, value, tolerance, ok in rows:
        result = "pass" if ok else "FAIL"
        print(
            f"| {check} | {reference} | {metric} | {value} | {tolerance} | {result} |"
        )
    return 0 if all(ok for *_, ok in rows) else 1


def write_fixtures(work: Path, cases: list[dict], inputs: dict, cube: Cube) -> None:
    """Copy the small zarr-python arrays and cube copy, with the values they hold."""
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    names = [case["name"] for case in cases if case["name"].endswith("-0")]
    for name in names:
        shutil.copytree(work / "zarr" / name, FIXTURE_DIR / "arrays" / name)
    shutil.copytree(work / "copies" / "tiny-gzip", FIXTURE_DIR / "tiny.cube.zarr")
    arrays, text = contents(cube)
    arrays |= {f"array:{name}": inputs[name] for name in names}
    np.savez_compressed(FIXTURE_DIR / "expected.npz", **arrays)
    expected = json.dumps(text, indent=1, ensure_ascii=False)
    (FIXTURE_DIR / "expected.json").write_text(expected + "\n")
    print(f"wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
