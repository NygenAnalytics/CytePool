"""Check cube fingerprints against a reimplementation written from their documentation.

No other tool computes a cube's fingerprint, so the reference is a separate program
that reads cube files with zarr-python and recomputes every digest and the
fingerprint with numpy and hashlib, following only the encoding that the Cube files
page of the documentation specifies (docs/source/reference/cube_files.md); it imports
nothing from CytePool. The script checks:

1. The digests that write_cube stores with each cube file equal the recomputed ones,
   part by part, and Cube.fingerprint() of each cube equals the recomputed
   fingerprint. The cubes have every table column type, missing values, -0.0 and NaN
   with another payload, empty axes, and no cell numbers or normalization.
2. Copies that zarr-python made with other chunks, big-endian bytes and other
   compression keep the fingerprint, recomputed and as read_cube verifies it, and
   so does a cube whose integer layers are int64 instead of narrower types.
3. Each change to a copy (a count, a float value, a sample id, a gene symbol, a
   cell number, the normalization's target sum) changes the recomputed fingerprint,
   and CytePool reports the changed file: read_cube refuses changed tables, cell
   numbers and normalization, and a changed layer group raises when it is read,
   whole or with its samples pooled, and in verify_fingerprint(). Changes that keep
   the values equal (-0.0 for 0.0, NaN with another payload) change neither.

The reference runs as a separate program and exchanges data with this script
through files in a temporary directory.

Requirements:
    This script: numpy, and cytepool from the source tree.
    Reference program: zarr-python 3 and numpy. It runs in the same interpreter unless
    REFERENCE_PYTHON names another Python executable.

Usage, from the repository root:
    PYTHONPATH=. python validation/cube_fingerprints.py
    PYTHONPATH=. python validation/cube_fingerprints.py --write-fixtures
"""

import argparse
import json
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

SEED = 20261025
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "cube_fingerprints"
)
# Changes the reference program makes to copies of the cube 'full', keeping its
# stored digests: (name, the layer changed or None, what CytePool must report, or
# None where the values stay equal).
LAYER_CHANGED = "does not match the digest stored with its cube file"
CHANGES = [
    ("count", "counts", LAYER_CHANGED),
    ("log_sum", "log_sum", LAYER_CHANGED),
    ("sample id", None, "the samples table does not match the digest stored"),
    ("symbol", None, "the features table does not match the digest stored"),
    ("n_cells", None, "n_cells does not match the digest stored"),
    ("target sum", None, "the normalization does not match the digest stored"),
    ("negative zero", None, None),
    ("NaN payload", None, None),
]

REFERENCE_PROGRAM = textwrap.dedent(
    '''
    """Recompute cube fingerprints from their documentation; copy and change cubes."""

    import hashlib
    import json
    import math
    import shutil
    import struct
    import sys
    from importlib.metadata import version
    from pathlib import Path

    import numpy as np
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    work = Path(sys.argv[1])
    plan = json.loads((work / "plan.json").read_text())
    LAYERS = ["counts", "detected", "log_sum", "log_sum_sq", "linear_sum"]
    # Bits of -0.0 and of the quiet NaN, for each float size.
    FLOATS = {
        2: ("<u2", 0x8000, 0x7E00),
        4: ("<u4", 0x80000000, 0x7FC00000),
        8: ("<u8", 0x8000000000000000, 0x7FF8000000000000),
    }

    def u64(n):
        return struct.pack("<Q", n)

    def text(value):
        data = value.encode("utf-8", "surrogatepass")
        return u64(len(data)) + data

    def digest(data):
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def floats(values):
        """Little-endian bits, -0.0 as 0.0 and every NaN as the quiet NaN."""
        unsigned, negative_zero, quiet = FLOATS[values.dtype.itemsize]
        values = values.astype(values.dtype.newbyteorder("<"))
        bits = values.view(unsigned).copy()
        bits[bits == negative_zero] = 0
        bits[np.isnan(values)] = quiet
        return bits.tobytes()

    def column(name, array):
        values, attributes = array[...], array.attrs
        if "levels" in attributes:
            levels = attributes["levels"]
            items = [None if c < 0 else levels[c] for c in values.tolist()]
            data = [None if v is None else v.encode("utf-8", "surrogatepass")
                    for v in items]
            sizes = b"".join(struct.pack("<q", -1 if d is None else len(d))
                             for d in data)
            return text(name) + text("text") + sizes + b"".join(
                d for d in data if d is not None)
        if "numpy_dtype" in attributes:
            body = values.astype("<i8").tobytes()
            return text(name) + text(attributes["numpy_dtype"]) + body
        dtype = values.dtype
        if dtype.kind == "b":
            body = values.astype(np.uint8).tobytes()
        elif dtype.kind in "iu":
            body = values.astype(dtype.newbyteorder("<")).tobytes()
        elif dtype.kind == "f":
            body = floats(values)
        else:  # complex: real and imaginary parts, a NaN part making both NaN
            half = np.dtype(f"f{dtype.itemsize // 2}")
            pairs = np.stack([values.real, values.imag], axis=-1).astype(half)
            pairs[np.isnan(pairs).any(axis=-1)] = np.nan
            body = floats(pairs.reshape(-1))
        return text(name) + text(np.dtype(dtype.name).name) + body

    def table(group):
        names, arrays = group.attrs["columns"], group.attrs["arrays"]
        rows = group[arrays[0]].shape[0]
        parts = [column(name, group[node]) for name, node in zip(names, arrays)]
        return digest(u64(rows) + u64(len(names)) + b"".join(parts))

    def json_value(value):
        if value is None:
            return b"n"
        if value is True or value is False:
            return b"t" if value else b"f"
        if isinstance(value, int):
            return b"i" + text(str(value))
        if isinstance(value, float):
            if value == math.floor(value):
                return b"i" + text(str(int(value)))
            return b"d" + struct.pack("<d", value)
        if isinstance(value, str):
            return b"s" + text(value)
        if isinstance(value, list):
            return b"a" + u64(len(value)) + b"".join(json_value(v) for v in value)
        keys = sorted(value, key=lambda key: key.encode("utf-8", "surrogatepass"))
        return b"o" + u64(len(keys)) + b"".join(
            text(key) + json_value(value[key]) for key in keys)

    def record(path):
        """The digest record of a cube file, from its documentation."""
        root = zarr.open_group(path, mode="r")
        attributes = root.attrs
        shape = [root[axis][root[axis].attrs["arrays"][0]].shape[0]
                 for axis in ("groups", "samples", "features")]
        found = {"version": 1, "shape": shape}
        for axis in ("samples", "groups", "features"):
            found[axis] = table(root[axis])
        if attributes["has_n_cells"]:
            cells = root["n_cells"][...].astype("<i8")
            found["n_cells"] = digest(u64(cells.shape[0]) + u64(cells.shape[1])
                                      + cells.tobytes())
        else:
            found["n_cells"] = None
        normalization = attributes["normalization"]
        found["normalization"] = (
            None if normalization is None else digest(json_value(normalization)))
        found["layers"] = {}
        for name in [n for n in LAYERS if n in attributes["layers"]]:
            array = root["layers"][name]
            kind = "int64" if array.dtype.kind in "iu" else "float64"
            groups = []
            for g in range(shape[0]):
                block = array[g]
                head = u64(block.shape[0]) + u64(block.shape[1])
                body = (block.astype("<i8").tobytes() if kind == "int64"
                        else floats(block.astype(np.float64)))
                groups.append(digest(head + body))
            found["layers"][name] = {"type": kind, "groups": groups}
        lines = ["cytepool cube fingerprint 1", "shape {} {} {}".format(*shape)]
        for part in ("samples", "groups", "features", "n_cells", "normalization"):
            lines.append(f"{part} {found[part] or 'none'}")
        for name, layer in found["layers"].items():
            for g, value in enumerate(layer["groups"]):
                lines.append(f"layer {name} {layer['type']} {g} {value}")
        manifest = "".join(line + "\\n" for line in lines)
        found["fingerprint"] = digest(manifest.encode("utf-8"))
        return found, manifest

    def copy(source, target, chunks_of, compressors, drop=()):
        """Copy a group node by node, with other chunks, big-endian bytes and codecs."""
        attributes = {k: v for k, v in source.attrs.items() if k not in drop}
        made = zarr.create_group(target, attributes=attributes)
        def walk(node_from, node_to):
            for key, node in node_from.members():
                if isinstance(node, zarr.Group):
                    walk(node, node_to.create_group(key, attributes=dict(node.attrs)))
                    continue
                node_to.create_array(
                    key, shape=node.shape, dtype=node.dtype,
                    fill_value=node.fill_value, chunks=chunks_of(node.shape),
                    serializer=BytesCodec(
                        endian="big" if node.dtype.itemsize > 1 else None),
                    compressors=compressors, attributes=dict(node.attrs),
                    dimension_names=node.metadata.dimension_names,
                )[...] = node[...]
        walk(source, made)

    def change(path, kind):
        """Change one thing in a copy of a cube file, keeping its stored digests."""
        root = zarr.open_group(path, mode="r+")
        if kind == "count":
            array = root["layers"]["counts"]
            array[0, 0, 0] = array[0, 0, 0] + 1
        elif kind == "log_sum":
            array = root["layers"]["log_sum"]
            array[0, 0, 1] = np.nextafter(array[0, 0, 1], np.inf)
        elif kind == "sample id":
            node = root["samples"]["sample_id"]
            levels = list(node.attrs["levels"])
            levels[1] = levels[1] + "x"
            node.attrs["levels"] = levels
        elif kind == "symbol":
            node = root["features"]["symbol"]
            levels = list(node.attrs["levels"])
            levels[0] = levels[0].lower()
            node.attrs["levels"] = levels
        elif kind == "n_cells":
            array = root["n_cells"]
            array[0, 0] = array[0, 0] + 1
        elif kind == "target sum":
            normalization = dict(root.attrs["normalization"])
            normalization["target_sum"] = normalization["target_sum"] * 2
            root.attrs["normalization"] = normalization
        elif kind == "negative zero":
            array = root["samples"]["score"]
            values = array[...]
            values[values == 0] = -0.0
            array[...] = values
        elif kind == "NaN payload":
            array = root["samples"]["age"]
            values = array[...]
            bits = values.view(np.uint64)
            bits[np.isnan(values)] = 0xFFF8000000000123  # sign and payload set
            array[...] = values

    results = {"records": {}, "manifests": {}}
    for name in plan["cubes"]:
        found, manifest = record(work / "cytepool" / name)
        results["records"][name], results["manifests"][name] = found, manifest
    for name in plan["copies"]:
        source = zarr.open_group(work / "cytepool" / name, mode="r")
        halves = lambda shape: tuple(max(1, -(-n // 2)) for n in shape)
        copy(source, work / "copies" / name, halves, [GzipCodec(level=1)])
        whole = lambda shape: tuple(max(1, n) for n in shape)
        copy(source, work / "plain" / name, whole, [])
        results["records"][f"copy:{name}"] = record(work / "copies" / name)[0]
        results["records"][f"plain:{name}"] = record(work / "plain" / name)[0]
    for name in plan["fixtures"]:
        source = zarr.open_group(work / "cytepool" / name, mode="r")
        halves = lambda shape: tuple(max(1, -(-n // 2)) for n in shape)
        copy(source, work / "fixtures" / name, halves, [GzipCodec(level=1)],
             drop=("digests",))
    for kind in plan["changes"]:
        target = work / "changed" / kind.replace(" ", "_")
        shutil.copytree(work / "cytepool" / "full", target)
        change(target, kind)
        results["records"][f"changed:{kind}"] = record(target)[0]
    versions = {package: version(package) for package in ("zarr", "numpy")}
    versions["python"] = sys.version.split()[0]
    results["versions"] = versions
    (work / "results.json").write_text(json.dumps(results, ensure_ascii=False))
    '''
)


def simulate_cube(
    rng: np.random.Generator, n_features: int, n_samples: int, n_groups: int
) -> Cube:
    """A cube with every layer, columns of every type and a group without cells."""
    shape = (n_groups, n_samples, n_features)
    visits = np.datetime64("2024-01-01") + rng.integers(0, 400, n_samples).astype(
        "m8[D]"
    )
    visits[1] = np.datetime64("NaT", "D")
    age = np.where(np.arange(n_samples) % 4 == 1, np.nan, 50.0)
    score = rng.standard_normal(n_samples)
    score[0], score[3] = 0.0, -0.0  # equal values, as Table.equals finds them
    samples = {
        "sample_id": [f"S{i:02d}" for i in range(n_samples)],
        "donor": [None if i % 5 == 3 else f"D{i // 2}" for i in range(n_samples)],
        "note": ["ünïcode ✓" if i == 0 else "" for i in range(n_samples)],
        "age": age,
        "score": score,
        "batch": (np.arange(n_samples) % 3).astype(np.int16),
        "visit": visits,
        "Treated?": rng.random(n_samples) < 0.5,
        "ratio": rng.lognormal(0, 0.5, n_samples).astype(np.float32),
        "plate": rng.integers(0, 255, n_samples).astype(np.uint8),
        "lane": rng.integers(-128, 127, n_samples).astype(np.int8),
        "depth": rng.integers(-(2**62), 2**62, n_samples, dtype=np.int64),
    }
    groups = {
        "group_id": [f"type {g}" for g in range(n_groups)],
        "cell_ontology_id": [
            None if g == 1 else f"CL:{g:07d}" for g in range(n_groups)
        ],
        "phase": rng.standard_normal(n_groups) + 1j * rng.standard_normal(n_groups),
        "phase64": (rng.standard_normal(n_groups) - 1j).astype(np.complex64),
        "rank": rng.permutation(n_groups).astype(np.int32),
        "lifetime": rng.integers(1, 99, n_groups).astype("m8[h]"),
    }
    groups["phase"][0] = complex(np.nan, 1.0)  # missing: a NaN part
    groups["phase64"][1] = complex(1.0, np.nan)
    features = {
        "feature_id": [f"ENSG{j:011d}" for j in range(n_features)],
        "symbol": [None if j % 50 == 7 else f"GENE{j}" for j in range(n_features)],
        "measured": np.arange(n_features) % 97 != 0,
        "gc": rng.uniform(0.3, 0.7, n_features).astype(np.float16),
        "length": rng.integers(200, 90_000, n_features).astype(np.uint32),
        "exons": rng.integers(1, 60, n_features).astype(np.uint16),
        "reads": rng.integers(0, 2**63, n_features, dtype=np.uint64) * 2 + 1,
    }
    cells = rng.integers(0, 300, size=(n_samples, n_groups))
    cells[:, -1] = 0  # a group without cells
    cells[0, 0] = 1
    n = cells.T[:, :, None] * np.ones(shape)
    mean = rng.lognormal(0.0, 1.0, size=shape)
    spread = np.where(n > 0, (n - 1) * rng.lognormal(-1, 1, shape), 0.0)
    log_sum = n * (mean - 1.0)  # negative values, and -0.0 where there are no cells
    layers = {
        "counts": rng.poisson(mean * n / 10).astype(np.uint16),
        "detected": rng.binomial(n.astype(np.int64), 0.3).astype(np.int16),
        "log_sum": np.where(n > 0, log_sum, -0.0),
        "log_sum_sq": spread + n * (mean - 1.0) ** 2,
        "linear_sum": n * rng.lognormal(1.0, 1.0, size=shape),
    }
    normalization = {
        "method": "library_size",
        "target_sum": 1e4,
        "log": "natural",
        "pseudocount": 1.0,
        "notes": {"ünï": [1, 2.5, None, True, False, "x"]},
    }
    return Cube(
        layers,
        samples=samples,
        groups=groups,
        features=features,
        n_cells=cells,
        normalization=normalization,
        provenance={"source": {"script": "validation/cube_fingerprints.py"}},
    )


def simulate_cubes(rng: np.random.Generator) -> dict[str, Cube]:
    """The cubes the reference recomputes: realistic, bulk, empty and tiny ones."""
    full = simulate_cube(rng, 3000, 12, 6)
    bulk = Cube(
        {"counts": rng.gamma(2.0, 50.0, size=(1, 30, 400))},  # non-integer counts
        samples={"sample_id": [f"B{i}" for i in range(30)], "site": ["x", "y"] * 15},
        groups={"group_id": ["bulk"]},
        features={"feature_id": [f"g{j}" for j in range(400)]},
    )
    empty = Cube(
        {"counts": np.zeros((2, 0, 0), dtype=np.int64)},
        samples={"sample_id": [], "donor": []},
        groups={"group_id": ["a", "b"]},
        features={"feature_id": []},
        n_cells=np.zeros((0, 2), dtype=np.int64),
    )
    tiny = simulate_cube(rng, 6, 4, 3)
    small_bulk = Cube(
        {"counts": np.array([[[0.5, 3.0, 0.0], [12.25, 0.0, 1.0]]])},
        samples={"sample_id": ["A", "B"]},
        groups={"group_id": ["bulk"]},
        features={"feature_id": ["x", "y", "z"]},
    )
    return {
        "full": full,
        "bulk": bulk,
        "empty": empty,
        "tiny": tiny,
        "small_bulk": small_bulk,
    }


def widened(cube: Cube) -> Cube:
    """The same cube with int64 integer layers instead of narrower types."""
    layers = {
        name: cube.layer(name).astype(np.int64)
        if cube.layer(name).dtype.kind in "iu"
        else cube.layer(name)
        for name in cube.layer_names
    }
    return Cube(
        layers,
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
        normalization=cube.normalization,
        provenance=cube.provenance,
    )


def run_reference(work: Path, cubes: dict[str, Cube]) -> dict[str, Any]:
    """Write the cube files, run the reference program, and return its results."""
    for folder in ("cytepool", "copies", "plain", "fixtures", "changed"):
        (work / folder).mkdir()
    for name, cube in cubes.items():
        write_cube(cube, work / "cytepool" / name)
    plan = {
        "cubes": list(cubes),
        "copies": ["full", "bulk", "tiny"],
        "fixtures": ["tiny", "small_bulk"],
        "changes": [name for name, *_ in CHANGES],
    }
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
    results: dict[str, Any] = json.loads((work / "results.json").read_text())
    return results


def parts_that_differ(found: dict[str, Any], expected: dict[str, Any]) -> int:
    """Count the digests of a record that differ from the expected record."""
    bad = sum(found.get(key) != expected[key] for key in expected if key != "layers")
    for name, layer in expected["layers"].items():
        theirs = found.get("layers", {}).get(name, {"groups": []})
        bad += sum(a != b for a, b in zip(theirs["groups"], layer["groups"]))
        bad += abs(len(theirs["groups"]) - len(layer["groups"]))
    return bad


def reported(path: Path, layer: str | None, message: str) -> bool:
    """Whether CytePool reports a cube file changed after it was written.

    A changed table, cell number or normalization must make read_cube raise. A
    changed group of a layer must raise when it is read, whole or with its samples
    pooled, and in verify_fingerprint(), while the other groups read.
    """
    try:
        cube = read_cube(path)
    except ValueError as error:
        return layer is None and message in str(error)
    if layer is None:
        return False
    for read in (
        lambda: cube.layer(layer, 0),
        lambda: cube.collapse_samples("batch").layer(layer, 0),
        cube.verify_fingerprint,
    ):
        try:
            read()
        except ValueError as error:
            if message not in str(error):
                return False
        else:
            return False
    return all(cube.layer(layer, g) is not None for g in range(1, cube.n_groups))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-fixtures", action="store_true")
    arguments = parser.parse_args()
    cubes = simulate_cubes(np.random.default_rng(SEED))
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        results = run_reference(work, cubes)
        records = results["records"]

        bad = 0
        for name in cubes:
            stored = json.loads((work / "cytepool" / name / "zarr.json").read_text())
            bad += parts_that_differ(stored["attributes"]["digests"], records[name])
        parts = sum(
            5 + sum(len(r["layers"][n]["groups"]) for n in r["layers"])
            for r in (records[name] for name in cubes)
        )
        check = (
            f"digests write_cube stores with {len(cubes)} cube files ({parts} parts)"
        )
        rows.append((check, "parts that differ", bad, "0", bad == 0))

        bad = sum(
            cube.fingerprint() != records[name]["fingerprint"]
            for name, cube in cubes.items()
        )
        check = f"Cube.fingerprint() of the {len(cubes)} cubes in memory"
        rows.append((check, "fingerprints that differ", bad, "0", bad == 0))

        bad = 0
        for name in ("full", "bulk", "tiny"):
            original = records[name]["fingerprint"]
            for kind in ("copy", "plain"):
                bad += records[f"{kind}:{name}"]["fingerprint"] != original
                folder = "copies" if kind == "copy" else "plain"
                bad += read_cube(work / folder / name).verify_fingerprint() != original
        check = (
            "zarr-python copies of 'full', 'bulk' and 'tiny' with other chunks, "
            "big-endian bytes, gzip level 1 or no compression"
        )
        rows.append((check, "fingerprints that differ", bad, "0", bad == 0))

        wide = widened(cubes["full"]).fingerprint()
        bad = int(wide != records["full"]["fingerprint"])
        check = "the cube 'full' with int64 instead of uint16 and int16 layers"
        rows.append((check, "fingerprints that differ", bad, "0", bad == 0))

        original = records["full"]["fingerprint"]
        changed = [
            records[f"changed:{name}"]["fingerprint"] != original
            for name, _, message in CHANGES
            if message is not None
        ]
        kept = [
            records[f"changed:{name}"]["fingerprint"] == original
            for name, _, message in CHANGES
            if message is None
        ]
        value = f"{sum(changed)} of {len(changed)}"
        check = "a changed count, log_sum value, sample id, symbol, cell number or "
        check += "target sum (in 'full')"
        rows.append((check, "changes the fingerprint", value, "all", all(changed)))
        value = f"{len(kept) - sum(kept)} of {len(kept)}"
        check = "-0.0 for 0.0 and NaN with another payload in sample columns"
        rows.append((check, "changes the fingerprint", value, "none", all(kept)))

        caught = []
        for name, layer, message in CHANGES:
            path = work / "changed" / name.replace(" ", "_")
            if message is not None:
                caught.append(reported(path, layer, message))
                continue
            try:
                caught.append(read_cube(path).verify_fingerprint() == original)
            except ValueError:  # an equal file refused
                caught.append(False)
        value = f"{sum(caught)} of {len(caught)}"
        check = "CytePool reports the 6 changed files and reads the 2 equal ones"
        rows.append((check, "files handled as expected", value, "all", all(caught)))
        if arguments.write_fixtures:
            write_fixtures(work, records, results["manifests"])

    versions = results["versions"]
    print("Reference: " + ", ".join(f"{key} {item}" for key, item in versions.items()))
    print(f"CytePool: Python {sys.version.split()[0]}, numpy {np.__version__}")
    print("\n| check | reference | metric | value | tolerance | result |")
    print("|---|---|---|---|---|---|")
    reference = f"documented encoding, zarr-python {versions['zarr']}"
    for check, metric, value, tolerance, ok in rows:
        result = "pass" if ok else "FAIL"
        print(
            f"| {check} | {reference} | {metric} | {value} | {tolerance} | {result} |"
        )
    return 0 if all(ok for *_, ok in rows) else 1


def write_fixtures(
    work: Path, records: dict[str, Any], manifests: dict[str, str]
) -> None:
    """Copy the zarr-python copies of two small cubes, without their stored digests,
    and the records and manifests the reference recomputed for them."""
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True)
    names = ["tiny", "small_bulk"]
    for name in names:
        shutil.copytree(work / "fixtures" / name, FIXTURE_DIR / f"{name}.cube.zarr")
    expected = {
        name: {"record": records[name], "manifest": manifests[name]} for name in names
    }
    text = json.dumps(expected, indent=1, ensure_ascii=False)
    (FIXTURE_DIR / "reference.json").write_text(text + "\n")
    print(f"wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    sys.exit(main())
