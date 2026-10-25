"""Fingerprints of cubes: the documented encoding, what they identify, and cube files.

The reference digests in tests/data/cube_fingerprints were recomputed by
validation/cube_fingerprints.py from the documentation alone (zarr-python, numpy and
hashlib, without CytePool), from copies of cube files that zarr-python wrote with
other chunks and big-endian bytes and without stored digests.
"""

import contextlib
import gzip
import hashlib
import json
import pickle
import re
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.cube import Cube, from_cells
from cytepool.io import cube_files, read_cube, write_cube

FIXTURES = Path(__file__).parent / "data" / "cube_fingerprints"
REFERENCE = json.loads((FIXTURES / "reference.json").read_text("utf-8"))
DIGEST = r"sha256:[0-9a-f]{64}"


_opened: list[str] | None = None


def _record_open(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and _opened is not None:
        _opened.append(str(args[0]))


sys.addaudithook(_record_open)


@contextlib.contextmanager
def opened_chunks() -> Iterator[list[str]]:
    """Record the layer chunk files opened inside the block, as layer/group.

    A chunk that is not stored, as the chunks of a group without cells, is recorded
    too: the audit event comes before the file is found missing.
    """
    global _opened
    _opened = []
    found: list[str] = []
    try:
        yield found
    finally:
        for path in _opened:
            parts = Path(path).parts
            if "layers" in parts and "c" in parts:
                at = parts.index("c")
                found.append(f"{parts[at - 1]}/{parts[at + 1]}")
        _opened = None


def rebuilt(cube: Cube, **changes: Any) -> Cube:
    """The cube built anew in memory from its values, with some arguments changed."""
    arguments: dict[str, Any] = {
        "layers": {name: cube.layer(name) for name in cube.layer_names},
        "samples": cube.samples,
        "groups": cube.groups,
        "features": cube.features,
        "n_cells": cube.n_cells,
        "normalization": cube.normalization,
        "provenance": cube.provenance,
    }
    arguments.update(changes)
    return Cube(arguments.pop("layers"), **arguments)


@pytest.fixture(scope="module")
def tiny() -> Cube:
    """The cube 'tiny' of the reference, in memory: every column type, 5 layers."""
    return rebuilt(read_cube(FIXTURES / "tiny.cube.zarr"))


@pytest.fixture(scope="module")
def store(tiny: Cube, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("store") / "tiny.cube.zarr"
    write_cube(tiny, path)
    return path


# The documented encoding.


@pytest.mark.parametrize("name", ["tiny", "small_bulk"])
def test_digests_follow_the_documented_encoding(name: str, tmp_path: Path) -> None:
    expected = REFERENCE[name]["record"]
    lazy = read_cube(FIXTURES / f"{name}.cube.zarr")  # no stored digests: computed

    assert lazy.digests() == expected
    assert rebuilt(lazy).fingerprint() == expected["fingerprint"]
    write_cube(lazy, tmp_path / "c")
    stored = json.loads((tmp_path / "c" / "zarr.json").read_text("utf-8"))
    assert stored["attributes"]["digests"] == expected


def test_a_digest_record_is_plain_json_and_new_each_time(tiny: Cube) -> None:
    record = tiny.digests()
    assert json.loads(json.dumps(record)) == record
    record["fingerprint"] = "changed"
    record["layers"]["counts"]["groups"][0] = "changed"
    assert tiny.digests() == REFERENCE["tiny"]["record"]


# A fingerprint identifies the values.


def _change_layer(name: str, change: Callable[[np.ndarray], None]) -> Callable:
    def changed(cube: Cube) -> Cube:
        values = cube.layer(name).copy()
        change(values)
        layers = {layer: cube.layer(layer) for layer in cube.layer_names}
        return rebuilt(cube, layers={**layers, name: values})

    return changed


def _change_table(axis: str, change: Callable[[dict], None]) -> Callable:
    def changed(cube: Cube) -> Cube:
        columns = getattr(cube, axis).to_dict()
        change(columns)
        return rebuilt(cube, **{axis: columns})

    return changed


def _next(array: np.ndarray, index: tuple[int, ...]) -> None:
    array[index] = np.nextafter(array[index], np.inf)


def _set(column: str, position: int, value: Any) -> Callable[[dict], None]:
    def change(columns: dict) -> None:
        values = np.array(columns[column], dtype=object)
        values[position] = value
        columns[column] = values.tolist()

    return change


def _normalization(change: Callable[[dict], None]) -> Callable[[Cube], Cube]:
    def changed(cube: Cube) -> Cube:
        normalization = cube.normalization
        assert normalization is not None
        change(normalization)
        return rebuilt(cube, normalization=normalization)

    return changed


def _reorder_samples(cube: Cube) -> Cube:
    order = [1, 0, 2, 3]
    layers = {name: cube.layer(name)[:, order] for name in cube.layer_names}
    n_cells = cube.n_cells
    assert n_cells is not None
    return rebuilt(
        cube, layers=layers, samples=cube.samples.take(order), n_cells=n_cells[order]
    )


def _without_layer(cube: Cube) -> Cube:
    layers = {name: cube.layer(name) for name in cube.layer_names}
    del layers["linear_sum"]
    return rebuilt(cube, layers=layers)


def _float_counts(cube: Cube) -> Cube:
    layers = {name: cube.layer(name) for name in cube.layer_names}
    return rebuilt(cube, layers={**layers, "counts": layers["counts"].astype(float)})


def _more_cells(cube: Cube) -> Cube:
    n_cells = cube.n_cells
    assert n_cells is not None
    changed = n_cells.copy()
    changed[1, 0] += 1
    return rebuilt(cube, n_cells=changed)


CHANGES: dict[str, Callable[[Cube], Cube]] = {
    "a count": _change_layer("counts", lambda v: v.__setitem__((0, 0, 0), 99)),
    "a detected cell": _change_layer(
        "detected", lambda v: v.__setitem__((0, 0, 0), 1 - v[0, 0, 0])
    ),
    "log_sum by 1 ulp": _change_layer("log_sum", lambda v: _next(v, (0, 0, 1))),
    "log_sum_sq by 1 ulp": _change_layer("log_sum_sq", lambda v: _next(v, (1, 2, 3))),
    "linear_sum by 1 ulp": _change_layer("linear_sum", lambda v: _next(v, (1, 0, 5))),
    "a sample id": _change_table("samples", _set("sample_id", 0, "S00b")),
    "a group id": _change_table("groups", _set("group_id", 2, "type 9")),
    "a feature id": _change_table("features", _set("feature_id", 5, "ENSG9")),
    "a symbol missing": _change_table("features", _set("symbol", 1, None)),
    "an empty note missing": _change_table("samples", _set("note", 1, None)),
    "an age by 1 ulp": _change_table(
        "samples", _set("age", 0, float(np.nextafter(50.0, 51.0)))
    ),
    "a column renamed": _change_table(
        "samples", lambda c: c.__setitem__("Batch", c.pop("batch"))
    ),
    "a column moved": _change_table(
        "samples", lambda c: c.__setitem__("donor", c.pop("donor"))
    ),
    "int16 batch as int32": _change_table(
        "samples", lambda c: c.__setitem__("batch", c["batch"].astype(np.int32))
    ),
    "a cell number": _more_cells,
    "the target sum": _normalization(lambda n: n.__setitem__("target_sum", 1e5)),
    "a nested note": _normalization(lambda n: n["notes"]["ünï"].append(3)),
    "samples reordered": _reorder_samples,
    "a layer left out": _without_layer,
    "counts as float64": _float_counts,
}


def test_every_change_of_a_value_id_or_count_changes_the_fingerprint(
    tiny: Cube,
) -> None:
    fingerprints = {
        name: change(tiny).fingerprint() for name, change in CHANGES.items()
    }
    fingerprints["original"] = tiny.fingerprint()

    assert len(set(fingerprints.values())) == len(fingerprints)


def test_cubes_with_equal_values_have_the_same_fingerprint(tiny: Cube) -> None:
    samples = tiny.samples.to_dict()
    score, age = samples["score"].copy(), samples["age"].copy()
    assert (score == 0).sum() == 2 and np.isnan(age).any()
    score[score == 0] = -score[score == 0]  # 0.0 and -0.0 swap signs
    age.view(np.uint64)[np.isnan(age)] = 0xFFF8_0000_0000_0123  # another NaN
    layers = {name: tiny.layer(name) for name in tiny.layer_names}
    wide = {
        name: v.astype(np.int64) if v.dtype.kind in "iu" else v
        for name, v in layers.items()
    }
    equal = [
        rebuilt(tiny, samples={**samples, "score": score, "age": age}),
        rebuilt(tiny, layers=wide),  # integers are hashed as int64, whatever the width
        rebuilt(tiny, provenance={"source": "anything else"}),
        tiny.regroup({"type 0": ["type 0"]}),  # renamed to itself: int64 layers
        pickle.loads(pickle.dumps(tiny)),
    ]

    # Computed anew from the values, not from digests a cube was given.
    assert {cube.verify_fingerprint() for cube in equal} == {tiny.fingerprint()}
    assert equal[3].layer("counts").dtype == np.int64
    assert equal[3].provenance != tiny.provenance


def test_pooling_some_cells_equals_selecting_their_samples() -> None:
    rng = np.random.default_rng(7)
    counts = rng.poisson(3.0, size=(300, 8)).astype(np.float32)
    samples = rng.choice(["S1", "S2", "S3"], 300)
    groups = rng.choice(["B", "T"], 300)
    counts[np.flatnonzero(samples == "S3")[0], 0] = 2.0**33  # S3 needs uint64 counts
    kept = samples != "S3"

    pooled = from_cells(counts, samples=samples, groups=groups).select(
        samples=["S1", "S2"]
    )
    subset = from_cells(counts[kept], samples=samples[kept], groups=groups[kept])

    assert pooled.layer("counts").dtype == np.uint64
    assert subset.layer("counts").dtype == np.uint32
    assert pooled.provenance != subset.provenance
    assert pooled.fingerprint() == subset.fingerprint()


# Cube files.


def test_a_saved_cube_takes_its_fingerprint_from_the_file(
    store: Path, tiny: Cube
) -> None:
    with opened_chunks() as chunks:
        back = read_cube(store)
        fingerprint = back.fingerprint()
        digests = back.digests()

    assert chunks == []
    assert (
        fingerprint == tiny.fingerprint() == REFERENCE["tiny"]["record"]["fingerprint"]
    )
    assert digests == REFERENCE["tiny"]["record"]
    # A pickled cube keeps them too.
    with opened_chunks() as chunks:
        assert pickle.loads(pickle.dumps(back)).fingerprint() == fingerprint
    assert chunks == []
    with opened_chunks() as chunks:
        assert back.verify_fingerprint() == fingerprint
    assert sorted(chunks) == sorted(
        f"{name}/{group}" for name in tiny.layer_names for group in range(3)
    )


def test_derived_cubes_read_only_the_groups_the_operation_changed(
    store: Path, tiny: Cube
) -> None:
    cases = [
        (lambda c: c.select(groups=["type 2", "type 0"]), set()),
        (lambda c: c.regroup({"merged": ["type 0", "type 1"]}), {0, 1}),
        (lambda c: c.regroup({"renamed": ["type 1"]}), set()),
        (lambda c: c.collapse_samples("batch"), {0, 1, 2}),
        (lambda c: c.select(features=[5, 0]), {0, 1, 2}),
    ]
    for derive, groups in cases:
        derived = derive(read_cube(store))
        with opened_chunks() as chunks:
            fingerprint = derived.fingerprint()
        assert fingerprint == derive(tiny).fingerprint()
        assert {int(chunk.split("/")[1]) for chunk in chunks} == groups
        if groups:
            assert {chunk.split("/")[0] for chunk in chunks} == set(tiny.layer_names)


def test_whole_reads_keep_digests_so_a_fingerprint_reads_the_rest(store: Path) -> None:
    # An analysis reads the counts of every group of a collapsed cube, then records
    # the fingerprint: only the other layers are read for it.
    collapsed = read_cube(store).collapse_samples("batch")
    for group in range(collapsed.n_groups):
        collapsed.layer("counts", group)
    with opened_chunks() as chunks:
        collapsed.fingerprint()
    assert chunks and not [chunk for chunk in chunks if chunk.startswith("counts/")]


def _with_cells(cube: Cube, group: int) -> int:
    """The first sample with cells in a group."""
    n_cells = cube.n_cells
    assert n_cells is not None
    return int(np.flatnonzero(n_cells[:, group])[0])


def _copy_chunk(source: Path, target: Path, chunk: str) -> None:
    (target / chunk).unlink()
    shutil.copyfile(source / chunk, target / chunk)


@pytest.fixture
def changed_store(store: Path, tiny: Cube, tmp_path: Path) -> Path:
    """A copy of the store whose counts of group 1 are those of another cube."""
    counts = tiny.layer("counts").copy()
    counts[1, _with_cells(tiny, 1), 0] += 1
    layers = {name: tiny.layer(name) for name in tiny.layer_names}
    write_cube(rebuilt(tiny, layers={**layers, "counts": counts}), tmp_path / "other")
    path = tmp_path / "changed"
    shutil.copytree(store, path)
    _copy_chunk(tmp_path / "other", path, "layers/counts/c/1/0/0")
    return path


def test_a_changed_group_is_reported_when_it_is_read(
    changed_store: Path, tiny: Cube
) -> None:
    back = read_cube(changed_store)  # the tables are unchanged
    stored = r"does not match the digest stored with its cube file"
    back.layer("counts", 0)
    back.layer("log_sum", 1)
    with pytest.raises(
        ValueError, match=r"group 'type 1' of the layer 'counts'.*" + stored
    ):
        back.layer("counts", 1)
    for read in (back.verify_fingerprint, back.to_memory, lambda: back.layer("counts")):
        with pytest.raises(ValueError, match=stored):
            read()
    with pytest.raises(ValueError, match=stored):
        back.select(groups=["type 1"]).layer("counts")
    # The fingerprint trusts the stored digests, and is that of the cube written.
    assert back.fingerprint() == tiny.fingerprint()


@pytest.mark.parametrize(
    "derive",
    [
        pytest.param(lambda c: c.collapse_samples("batch"), id="pooled samples"),
        pytest.param(lambda c: c.select(samples=[3, 1, 0, 2]), id="samples reordered"),
        pytest.param(lambda c: c.select(features=[5, 4, 3, 2, 1, 0]), id="features"),
        pytest.param(lambda c: c.regroup({"g": ["type 1", "type 0"]}), id="merged"),
    ],
)
def test_reads_that_cover_a_whole_group_in_another_order_are_checked(
    changed_store: Path, tmp_path: Path, derive: Callable[[Cube], Cube]
) -> None:
    derived = derive(read_cube(changed_store))
    stored = "does not match the digest stored with its cube file"
    reads = [
        lambda: derived.layer("counts"),
        derived.to_memory,
        derived.verify_fingerprint,
        lambda: write_cube(derived, tmp_path / "copy"),
    ]
    for read in reads:
        with pytest.raises(ValueError, match=stored):
            read()
    assert not (tmp_path / "copy").exists()


def test_reads_of_part_of_a_group_are_checked_by_verify_fingerprint(
    changed_store: Path, tiny: Cube
) -> None:
    # The changed count is that of feature 0; part of a group cannot be checked on
    # its own as it is read.
    part = read_cube(changed_store).select(features=[0, 2])
    assert part.layer("counts", 1)[_with_cells(tiny, 1), 0] == (
        tiny.layer("counts", 1)[_with_cells(tiny, 1), 0] + 1
    )
    with pytest.raises(ValueError, match="does not match the digest stored"):
        part.verify_fingerprint()
    with pytest.raises(ValueError, match="does not match the digest stored"):
        part.regroup({"g": ["type 1"]}).verify_fingerprint()


def test_a_file_without_digests_is_read_and_its_groups_kept_consistent(
    store: Path, tiny: Cube, tmp_path: Path
) -> None:
    path = tmp_path / "old"
    shutil.copytree(store, path)
    root = json.loads((path / "zarr.json").read_text("utf-8"))
    del root["attributes"]["digests"]  # as files written before digests existed
    (path / "zarr.json").write_text(json.dumps(root), "utf-8")
    back = read_cube(path)

    with opened_chunks() as chunks:
        assert back.fingerprint() == tiny.fingerprint()
    assert sorted(chunks) == sorted(  # every chunk, once
        f"{name}/{group}" for name in tiny.layer_names for group in range(3)
    )
    other = tmp_path / "other"
    counts = tiny.layer("counts").copy()
    counts[1, _with_cells(tiny, 1), 0] += 1
    layers = {name: tiny.layer(name) for name in tiny.layer_names}
    write_cube(rebuilt(tiny, layers={**layers, "counts": counts}), other)
    _copy_chunk(other, path, "layers/counts/c/1/0/0")
    with pytest.raises(ValueError, match="differs from when it was first read"):
        back.layer("counts", 1)


@pytest.mark.parametrize(
    ("part", "change", "message"),
    [
        ("samples", "donor", "the samples table does not match the digest stored"),
        ("groups", "cell_ontology_id", "the groups table does not match"),
        ("features", "symbol", "the features table does not match"),
        ("n_cells", None, "n_cells does not match"),
        ("normalization", None, "the normalization does not match"),
    ],
)
def test_changed_tables_cells_or_normalization_are_reported_on_opening(
    store: Path, tmp_path: Path, part: str, change: str | None, message: str
) -> None:
    path = tmp_path / "changed"
    shutil.copytree(store, path)
    if part == "normalization":
        root = json.loads((path / "zarr.json").read_text("utf-8"))
        root["attributes"]["normalization"]["target_sum"] = 1e5
        (path / "zarr.json").write_text(json.dumps(root), "utf-8")
    elif part == "n_cells":
        cells = path / "n_cells" / "c" / "0" / "0"
        values = np.frombuffer(gzip.decompress(cells.read_bytes()), "<i4").copy()
        values[0] += 1
        cells.write_bytes(gzip.compress(values.tobytes(), mtime=0))
    else:
        metadata = path / part / str(change) / "zarr.json"
        document = json.loads(metadata.read_text("utf-8"))
        levels = document["attributes"]["levels"]
        levels[0] = levels[0] + " (edited)"
        metadata.write_text(json.dumps(document), "utf-8")

    with pytest.raises(ValueError, match=message):
        read_cube(path)
    # The message says how to open a file changed on purpose.
    root = json.loads((path / "zarr.json").read_text("utf-8"))
    del root["attributes"]["digests"]
    (path / "zarr.json").write_text(json.dumps(root), "utf-8")
    assert read_cube(path).fingerprint() != REFERENCE["tiny"]["record"]["fingerprint"]


def _edit(record: dict[str, Any], change: str) -> None:
    if change == "version":
        record["version"] = 2
    elif change == "fingerprint":
        record["fingerprint"] = "sha256:" + "0" * 64
    elif change == "key":
        del record["n_cells"]
    elif change == "shape":
        record["shape"] = [3, 4, 7]
    elif change == "groups":
        record["layers"]["counts"]["groups"].pop()
    elif change == "type":
        record["layers"]["counts"]["type"] = "float64"
    elif change == "layers":
        record["layers"].pop("detected")
    elif change == "digest":
        record["samples"] = "sha256:XYZ"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("version", r"fingerprint version 2, but .* computes version 1; update"),
        ("fingerprint", "is not the digest of the manifest of the digests stored"),
        ("key", "must be an object with the keys features, fingerprint, groups"),
        ("shape", r"describes a cube of shape \[3, 4, 7\]"),
        ("groups", r"layer 'counts' must be an object .* one digest per group \(3\)"),
        ("type", r"layer 'counts' must be an object with the type 'int64'"),
        ("layers", "lists the layers"),
        ("digest", "samples must be a digest, got 'sha256:XYZ'"),
    ],
)
def test_malformed_digest_records_are_refused(
    store: Path, tmp_path: Path, change: str, message: str
) -> None:
    path = tmp_path / "edited"
    shutil.copytree(store, path)
    root = json.loads((path / "zarr.json").read_text("utf-8"))
    _edit(root["attributes"]["digests"], change)
    (path / "zarr.json").write_text(json.dumps(root), "utf-8")

    with pytest.raises(ValueError, match=message) as raised:
        read_cube(path)
    assert "delete its root attribute 'digests'" in raised.value.__notes__[0]


def test_the_order_of_the_members_of_a_stored_record_does_not_matter(
    store: Path, tiny: Cube, tmp_path: Path
) -> None:
    # JSON objects have no order; another writer may sort them.
    path = tmp_path / "sorted"
    shutil.copytree(store, path)
    root = json.loads((path / "zarr.json").read_text("utf-8"))
    (path / "zarr.json").write_text(json.dumps(root, sort_keys=True), "utf-8")
    stored = json.loads((path / "zarr.json").read_text("utf-8"))["attributes"]
    assert list(stored["digests"]["layers"]) != list(tiny.layer_names)

    assert read_cube(path).fingerprint() == tiny.fingerprint()
    assert read_cube(path).verify_fingerprint() == tiny.fingerprint()


def test_digests_do_not_depend_on_chunks_or_compression(
    tiny: Cube, store: Path, tmp_path: Path
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cube_files, "FEATURES_PER_CHUNK", 2)
        write_cube(tiny, tmp_path / "small_chunks", compression=None)
    stored = [
        json.loads((path / "zarr.json").read_text("utf-8"))["attributes"]["digests"]
        for path in (store, tmp_path / "small_chunks")
    ]
    assert stored[0] == stored[1]
    assert read_cube(tmp_path / "small_chunks").verify_fingerprint() == (
        tiny.fingerprint()
    )


def test_a_lazy_layer_of_another_array_like_is_read_on_every_call(tiny: Cube) -> None:
    class Rows:
        """An array-like whose values can change, read one group at a time."""

        def __init__(self, values: np.ndarray) -> None:
            self.values, self.shape, self.dtype = values, values.shape, values.dtype
            self.reads = 0

        def __getitem__(self, key: Any) -> np.ndarray:
            self.reads += 1
            return self.values[key]

    counts = Rows(tiny.layer("counts").copy())
    layers = {name: tiny.layer(name) for name in tiny.layer_names}
    lazy = rebuilt(tiny, layers={**layers, "counts": counts})
    assert lazy.fingerprint() == tiny.fingerprint()
    assert counts.reads == tiny.n_groups
    counts.values[0, 0, 0] += 1  # the source changes: the fingerprint follows
    assert lazy.fingerprint() != tiny.fingerprint()
    assert counts.reads == 2 * tiny.n_groups
    assert lazy.select(groups=["type 0"]).provenance["operations"][-1]["input"] is None


def test_a_lone_surrogate_is_hashed_in_its_three_byte_form() -> None:
    # A Python string can hold one, though a cube file cannot; its digest is the
    # SHA-256 of these bytes, written out from the documented encoding.
    cube = Cube(
        {"counts": np.zeros((1, 1, 1), dtype=np.uint32)},
        samples={"sample_id": ["\ud800"]},
        groups={"group_id": ["g"]},
        features={"feature_id": ["f"]},
    )
    encoded = b"".join(
        [
            (1).to_bytes(8, "little"),  # rows
            (1).to_bytes(8, "little"),  # columns
            (9).to_bytes(8, "little") + b"sample_id",
            (4).to_bytes(8, "little") + b"text",
            (3).to_bytes(8, "little", signed=True),  # the length of the value
            b"\xed\xa0\x80",  # U+D800 in its 3-byte form
        ]
    )
    expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert cube.digests()["samples"] == expected


def test_the_record_and_the_history_name_the_fingerprint(tiny: Cube) -> None:
    assert tiny.history().splitlines()[-1] == f"Fingerprint: {tiny.fingerprint()}"
    assert tiny.fingerprint() == tiny.digests()["fingerprint"]
    assert all(
        re.fullmatch(DIGEST, digest)
        for layer in tiny.digests()["layers"].values()
        for digest in layer["groups"]
    )
