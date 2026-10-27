"""Provenance records: what built a cube, what each operation pooled, and the chain."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import Cube, from_bulk, from_cells, read_cube, write_cube
from cytepool.cube.provenance import describe_provenance

SAMPLES = {
    "sample_id": ["S1", "S2", "S3", "S4", "S5"],
    "donor": ["D1", "D2", "D1", "D3", "D2"],
    "sex": ["F", "M", "F", "F", "M"],  # constant within each donor
    "condition": ["ctrl", "ctrl", "stim", "stim", "stim"],
}
GROUPS = {"group_id": ["B", "T4", "T8", "NK"], "lineage": ["B", "T", "T", "NK"]}
FEATURES = [f"G{j:02d}" for j in range(12)]
SOURCE = {
    "dataset": "simulated PBMC",
    "artifacts": {"membership": "membership.parquet#sha256:5e1f"},
    "exporter": {"name": "a cell store", "version": "2.1", "min_genes": 200},
}
# The texts of a history, in lines of at most 80 characters.
UNRESOLVED = [
    "Source, as the caller recorded it (unresolved: CytePool does not follow its",
    "    references):",
]
REFERENCES = [
    "References: CytePool resolves none on its own. The source record's references",
    "            need the tool that wrote them. A fingerprint names a cube without",
    "            saying where it is: Cube.fingerprint() of a cube at hand says",
    "            whether it is that cube.",
]


@pytest.fixture(scope="module")
def cells() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    counts = rng.poisson(2.0, size=(400, len(FEATURES)))
    counts[counts.sum(axis=1) == 0, 0] = 1  # every cell has a positive library size
    samples = rng.choice(SAMPLES["sample_id"], 400)
    groups = rng.choice(GROUPS["group_id"], 400)
    return counts, samples, groups


@pytest.fixture(scope="module")
def cube(cells: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Cube:
    counts, samples, groups = cells
    return from_cells(
        counts,
        sample_labels=samples,
        group_labels=groups,
        samples=SAMPLES,
        groups=GROUPS,
        features=FEATURES,
        source=SOURCE,
    )


# How a cube was built.


def test_from_cells_records_the_source_unchanged_and_its_settings(cube: Cube) -> None:
    assert cube.provenance == {
        "source": SOURCE,
        "construction": {
            "function": "from_cells",
            "cytepool_version": cytepool.__version__,
            "n_cells": 400,
            "layers": ["counts", "detected", "log_sum", "log_sum_sq", "linear_sum"],
            "target_sum": 10000.0,
            "library_sizes": "cell_totals",
            "samples": {"order": "sample_table", "labels": "text"},
            "groups": {"order": "group_table", "labels": "text"},
            "features": "ids",
        },
    }


def test_from_cells_records_other_settings(
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    counts, samples, groups = cells
    numbers = np.array([int(label[1:]) for label in samples])  # S3 is 3
    cube = from_cells(
        counts,
        sample_labels=numbers,
        group_labels=groups,
        features={"feature_id": FEATURES},
        target_sum=1e6,
        library_sizes=counts.sum(axis=1) + 1.0,
        layers=["counts", "linear_sum"],
    )
    construction = cube.provenance["construction"]
    assert cube.provenance["source"] is None
    assert construction["samples"] == {"order": "sorted_labels", "labels": "integers"}
    assert construction["groups"] == {"order": "sorted_labels", "labels": "text"}
    assert cube.samples["sample_id"].tolist() == ["1", "2", "3", "4", "5"]
    assert construction["features"] == "table"
    assert construction["target_sum"] == 1e6
    assert construction["library_sizes"] == "given"
    assert construction["layers"] == ["counts", "linear_sum"]
    positions = from_cells(
        counts, sample_labels=samples, group_labels=groups, layers="counts"
    )
    assert positions.provenance["construction"]["features"] == "positions"


def test_records_name_where_ids_of_data_frames_came_from(
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    pandas = pytest.importorskip("pandas")
    counts, samples, groups = cells
    frame = pandas.DataFrame(counts, columns=FEATURES)
    pooled = from_cells(
        frame, sample_labels=samples, group_labels=groups, layers="counts"
    )
    assert pooled.provenance["construction"]["features"] == "data_frame_columns"
    bulk = pandas.DataFrame(counts[:3], index=["a", "b", "c"], columns=FEATURES)
    assert from_bulk(bulk).provenance["construction"] == {
        "function": "from_bulk",
        "cytepool_version": cytepool.__version__,
        "samples": "data_frame_index",
        "features": "data_frame_columns",
    }


def test_from_bulk_records_the_source_and_where_ids_came_from() -> None:
    counts = np.array([[5, 0, 12], [7, 1, 9]])
    cube = from_bulk(counts, samples={"sample_id": ["A", "B"]}, source={"run": 7})
    assert cube.provenance == {
        "source": {"run": 7},
        "construction": {
            "function": "from_bulk",
            "cytepool_version": cytepool.__version__,
            "samples": "table",
            "features": "positions",
        },
    }


def test_a_source_record_is_stored_as_plain_json() -> None:
    source = {"n": np.int64(3), "ids": np.array(["a", "b"]), "steps": ({"x": 1.5},)}
    cube = from_bulk(np.ones((2, 2), dtype=int), source=source)
    assert cube.provenance["source"] == {
        "n": 3,
        "ids": ["a", "b"],
        "steps": [{"x": 1.5}],
    }
    assert type(cube.provenance["source"]["n"]) is int


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        (["dataset"], TypeError, r"source must be a mapping .* got list"),
        ("pbmc", TypeError, r"source must be a mapping .* got str"),
        ({1: "x"}, TypeError, "source has the key 1; JSON keys are strings"),
        ({"x": float("nan")}, ValueError, r"source\['x'\] is nan; JSON has no NaN"),
        ({"x": {"y": [1, np.inf]}}, ValueError, r"source\['x'\]\['y'\]\[1\] is inf"),
        ({"when": object()}, TypeError, r"source\['when'\] has a value of type 'obj"),
    ],
)
def test_invalid_source_records_are_refused(
    cells: tuple[np.ndarray, np.ndarray, np.ndarray],
    source: Any,
    error: type[Exception],
    message: str,
) -> None:
    counts, samples, groups = cells
    with pytest.raises(error, match=message):
        from_cells(counts, sample_labels=samples, group_labels=groups, source=source)
    with pytest.raises(error, match=message):
        from_bulk(np.ones((2, 2), dtype=int), source=source)


# What each operation kept or pooled.


def test_select_records_the_ids_it_kept_in_order(cube: Cube) -> None:
    selected = cube.select(
        samples=np.array([False, True, False, True, True]),
        groups=[3, 0],
        features=["G05", "G01"],
    )
    record = selected.provenance["operations"][-1]
    assert record == {
        "operation": "select",
        "input": cube.fingerprint(),
        "samples": ["S2", "S4", "S5"],
        "groups": ["NK", "B"],
        "features": ["G05", "G01"],
    }
    assert cube.select(groups="T8").provenance["operations"][-1]["samples"] is None


def test_regroup_records_each_merged_group_with_its_members(cube: Cube) -> None:
    regrouped = cube.regroup({"T": ["T8", "T4"], "natural killer": "NK"})
    record = regrouped.provenance["operations"][-1]
    # In the order of the result's groups, members in the order of the input's.
    assert record == {
        "operation": "regroup",
        "input": cube.fingerprint(),
        "membership": {"T": ["T4", "T8"], "natural killer": ["NK"]},
    }
    positions = {g: i for i, g in enumerate(cube.groups["group_id"].tolist())}
    for name in cube.layer_names:
        for new, members in record["membership"].items():
            merged = regrouped.layer(name, new)
            parts = [cube.layer(name, positions[member]) for member in members]
            assert_array_equal(merged, np.sum(parts, axis=0, dtype=merged.dtype))


def test_collapse_membership_says_exactly_what_each_new_sample_pools(
    cube: Cube,
) -> None:
    collapsed = cube.collapse_samples("donor")
    record = collapsed.provenance["operations"][-1]
    membership = record["membership"]
    assert record["by"] == "donor"
    assert membership == {"D1": ["S1", "S3"], "D2": ["S2", "S5"], "D3": ["S4"]}
    assert list(membership) == collapsed.samples["sample_id"].tolist()
    assert collapsed.samples["n_pooled_samples"].tolist() == [2, 2, 1]
    # Rebuilt from the record alone, with numpy sums over the original samples.
    position = {s: i for i, s in enumerate(cube.samples["sample_id"].tolist())}
    for name in cube.layer_names:
        layer = cube.layer(name)
        for new_index, members in enumerate(membership.values()):
            expected = layer[:, [position[m] for m in members]].sum(axis=1)
            # numpy may add the two members of a float sum in another order;
            # a sum of two terms is exact either way.
            assert_allclose(collapsed.layer(name)[:, new_index], expected, rtol=0)
    n_cells = cube.n_cells
    assert n_cells is not None and collapsed.n_cells is not None
    for new_index, members in enumerate(membership.values()):
        rows = n_cells[[position[m] for m in members]]
        assert_array_equal(collapsed.n_cells[new_index], rows.sum(axis=0))


def test_collapsing_twice_records_the_ids_of_the_cube_it_was_applied_to(
    cube: Cube,
) -> None:
    once = cube.collapse_samples("donor")
    twice = once.collapse_samples("sex")
    first, second = twice.provenance["operations"]
    assert first["membership"] == {"D1": ["S1", "S3"], "D2": ["S2", "S5"], "D3": ["S4"]}
    assert second == {
        "operation": "collapse_samples",
        "input": once.fingerprint(),
        "by": "sex",
        "membership": {"F": ["D1", "D3"], "M": ["D2"]},
    }
    # n_pooled_samples counts the original samples: F pools D1 (2) and D3 (1).
    assert twice.samples["n_pooled_samples"].tolist() == [3, 2]


def replay(cube: Cube, record: dict[str, Any]) -> Cube:
    """Apply an operation to a cube as its record describes it."""
    operation = record["operation"]
    if operation == "select":
        return cube.select(
            samples=record["samples"],
            groups=record["groups"],
            features=record["features"],
        )
    if operation == "regroup":
        return cube.regroup(record["membership"])
    return cube.collapse_samples(record["by"])


def test_a_chain_of_operations_can_be_checked_link_by_link(
    cube: Cube, tmp_path: Path
) -> None:
    derived = (
        cube.select(samples=["S5", "S1", "S2", "S3"], features=FEATURES[::-1][:7])
        .regroup({"T": ["T8", "T4"]})
        .collapse_samples("donor")
        .select(groups=["T", "NK"])
    )
    expected = derived.verify_fingerprint()  # computed anew from the values
    current = cube
    for record in derived.provenance["operations"]:
        assert record["input"] == current.verify_fingerprint()
        current = replay(current, record)
    assert current.verify_fingerprint() == expected
    assert current.provenance == derived.provenance
    # The same links, followed from the saved cube: each lazy cube is read for its
    # fingerprint.
    write_cube(cube, tmp_path / "c")
    current = read_cube(tmp_path / "c")
    for record in derived.provenance["operations"]:
        assert current.fingerprint() == record["input"]
        current = replay(current, record)
    assert current.verify_fingerprint() == expected


def test_operations_on_a_lazy_cube_record_inputs_known_without_reading(
    cube: Cube, tmp_path: Path
) -> None:
    write_cube(cube, tmp_path / "c")
    lazy = read_cube(tmp_path / "c")
    selected = lazy.select(features=["G01", "G02"])
    # The file stores its digests, so its fingerprint costs nothing.
    assert selected.provenance["operations"][-1]["input"] == cube.fingerprint()
    # The selection's would need its layers read, so it is not recorded ...
    unknown = selected.regroup({"T": ["T4", "T8"]})
    assert unknown.provenance["operations"][-1]["input"] is None
    # ... unless it was computed first.
    expected = cube.select(features=["G01", "G02"]).fingerprint()
    assert selected.fingerprint() == expected
    known = selected.regroup({"T": ["T4", "T8"]})
    assert known.provenance["operations"][-1]["input"] == expected
    # A selection of groups keeps the digests of the file: known at once.
    groups = lazy.select(groups=["B", "NK"]).regroup({"B cells": "B"})
    assert groups.provenance["operations"][-1]["input"] == (
        cube.select(groups=["B", "NK"]).fingerprint()
    )


def test_records_survive_saving_reading_pickling_and_to_memory(
    cube: Cube, tmp_path: Path
) -> None:
    derived = cube.regroup({"T": ["T8", "T4"]}).collapse_samples("donor")
    expected = derived.verify_fingerprint()
    write_cube(derived, tmp_path / "c")
    back = read_cube(tmp_path / "c")
    for copy in (back, back.to_memory(), pickle.loads(pickle.dumps(back))):
        assert copy.provenance == derived.provenance
        assert copy.verify_fingerprint() == expected  # from the values read
    assert back.provenance["source"] == SOURCE


def test_records_keep_the_order_of_their_keys_and_operations_append() -> None:
    layers = {"counts": np.ones((2, 2, 1), dtype=np.uint32)}
    axes = {
        "samples": {"sample_id": ["a", "b"]},
        "groups": {"group_id": ["g", "h"]},
        "features": {"feature_id": ["f"]},
    }
    given = {"operations": [], "note": "x"}
    cube = Cube(layers, **axes, provenance=given)
    assert cube.provenance == given and list(cube.provenance) == ["operations", "note"]
    selected = cube.select(samples=["b"]).select(groups=["h"])
    assert list(selected.provenance) == ["operations", "note"]
    assert [op["samples"] for op in selected.provenance["operations"]] == [["b"], None]
    plain = Cube(layers, **axes, provenance={"note": "x"}).select(features="f")
    assert list(plain.provenance) == ["note", "operations"]
    assert Cube(layers, **axes).provenance == {}


def test_cubes_pickle_with_every_protocol_and_from_earlier_versions(
    cube: Cube,
) -> None:
    expected = cube.verify_fingerprint()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(cube, protocol=protocol))
        assert restored.provenance == cube.provenance
        assert restored.verify_fingerprint() == expected
    # Before digests were kept, a pickled cube held its provenance as one JSON text
    # and nothing else.
    restore, state = cube.__reduce__()
    earlier = restore(*state[:6], json.dumps(cube.provenance))
    assert earlier.provenance == cube.provenance
    assert earlier.verify_fingerprint() == expected


@pytest.mark.parametrize(
    ("source", "subject"),
    [
        ({"name": "a\ud800b"}, r"source\['name'\]"),
        ({"x": ["ok", {"y": "\udc00"}]}, r"source\['x'\]\[1\]\['y'\]"),
        ({"a\ud800": 1}, "a key of source"),
    ],
)
def test_text_that_is_not_valid_unicode_is_refused(source: Any, subject: str) -> None:
    # Python strings can hold lone surrogates, which neither JSON files nor a
    # terminal take.
    message = subject + r" holds the lone surrogate '\\ud[8c]00'"
    with pytest.raises(ValueError, match=message):
        from_bulk(np.ones((2, 2), dtype=int), source=source)
    with pytest.raises(ValueError, match=message.replace("source", "provenance")):
        Cube(
            {"counts": np.ones((1, 1, 1), dtype=np.uint32)},
            samples={"sample_id": ["a"]},
            groups={"group_id": ["g"]},
            features={"feature_id": ["f"]},
            provenance=source,
        )


# Reading a record back.


def test_history_reads_the_chain_in_plain_terms(cube: Cube) -> None:
    regrouped = cube.regroup({"T": ["T8", "T4"]})
    collapsed = regrouped.collapse_samples("donor")
    selected = collapsed.select(groups=["T", "B"], features=FEATURES[::-1])

    history = selected.history().splitlines()
    assert history == [
        *UNRESOLVED,
        '  dataset: "simulated PBMC"',
        '  artifacts: {"membership": "membership.parquet#sha256:5e1f"}',
        '  exporter: {"name": "a cell store", "version": "2.1", "min_genes": 200}',
        f"Built by from_cells (CytePool {cytepool.__version__}) from 400 cells:",
        "  samples: the rows of the sample table given",
        "  groups: the rows of the group table given",
        "  features: ids given as a sequence",
        "  layers: counts, detected, log_sum, log_sum_sq, linear_sum",
        "  normalized values: each cell scaled to a total of 10,000, with library "
        "sizes",
        "                     the total count of each cell",
        "1. regroup, applied to the cube",
        f"   {cube.fingerprint()}",
        "  T: T4 + T8",
        "2. collapse_samples by 'donor', applied to the cube",
        f"   {regrouped.fingerprint()}",
        "  D1: S1 + S3",
        "  D2: S2 + S5",
        "  D3: S4",
        "3. select, applied to the cube",
        f"   {collapsed.fingerprint()}",
        "  samples: every sample, as they were",
        "  groups: T, B",
        "  features: G11, G10, G09, G08, G07, G06, G05, G04, G03, G02, ... (12 in all)",
        *REFERENCES,
        "Fingerprint:",
        f"    {selected.fingerprint()}",
    ]
    assert max(map(len, history)) <= 80


def test_a_lazy_history_says_what_was_not_computed(cube: Cube, tmp_path: Path) -> None:
    write_cube(cube, tmp_path / "c")
    lazy = read_cube(tmp_path / "c").select(features=["G01"]).regroup({"T": "T4"})
    lines = lazy.history().splitlines()
    assert "2. regroup, applied to a cube whose fingerprint was not recorded" in lines
    assert lines[-2:] == [
        "Fingerprint: not known without reading the lazy layers; fingerprint() reads",
        "             them.",
    ]


def test_long_memberships_are_shortened() -> None:
    membership = {f"P{i}": [f"S{i}-{j}" for j in range(12)] for i in range(25)}
    record = {"operation": "collapse_samples", "by": "pool", "membership": membership}
    lines = describe_provenance({"operations": [record]}).splitlines()
    assert lines[0] == "1. collapse_samples by 'pool'"
    # The first 10 of each pooled sample's 12 members, the line broken at 80.
    assert lines[1:3] == [
        "  P0: " + " + ".join(f"S0-{j}" for j in range(10)) + " + ...",
        "      (12 in all)",
    ]
    assert len(lines) == 1 + 2 * 20 + 1 and lines[-1] == "  ... (25 in all)"


def test_history_names_the_columns_added_and_each_way_of_pooling(cube: Cube) -> None:
    changed = cube.with_columns(
        samples={
            "pool": ["P1", "P1", "P2", "P2", "P1"],
            "sex": ["F", "M", "F", "F", "F"],
        },
        features={"length": np.arange(len(FEATURES))},
    )
    by_two = changed.collapse_samples(["pool", "sex"])
    everything = by_two.collapse_samples()
    lines = everything.history().splitlines()
    start = lines.index("1. with_columns, applied to the cube")
    assert lines[start : start + 12] == [
        "1. with_columns, applied to the cube",
        f"   {cube.fingerprint()}",
        "  samples: added pool; replaced sex",
        "  features: added length",
        "2. collapse_samples by 'pool' and 'sex', applied to the cube",
        f"   {changed.fingerprint()}",
        "  P1_F: S1 + S5",
        "  P1_M: S2",
        "  P2_F: S3 + S4",
        "3. collapse_samples of every sample into one, applied to the cube",
        f"   {by_two.fingerprint()}",
        "  all: P1_F + P1_M + P2_F",
    ]
    assert lines[start + 12 :] == [
        *REFERENCES,
        "Fingerprint:",
        f"    {everything.fingerprint()}",
    ]
    assert max(map(len, lines)) <= 80


def test_records_of_earlier_versions_and_other_entries_are_shown_as_json() -> None:
    earlier = {
        "source": "from_cells",
        "n_cells": 100,
        "operations": [
            {"operation": "select", "samples": ["S1"], "groups": None, "n_features": 5},
            {"operation": "regroup", "mapping": {"T": ["T8", "T4"]}},
            {"operation": "collapse_samples", "by": "donor"},
            "not a record",
        ],
        "construction": {"function": "elsewhere", "settings": [1]},
    }
    assert describe_provenance(earlier).splitlines() == [
        'Source: "from_cells"',
        "Built by elsewhere:",
        "  settings: [1]",
        "1. select",
        "  samples: S1",
        "  groups: every group, as they were",
        "  n_features: 5",
        "2. regroup",
        '  mapping: {"T": ["T8", "T4"]}',
        "3. collapse_samples by 'donor'",
        '4. "not a record"',
        "n_cells: 100",
    ]
    assert describe_provenance({}) == "Nothing is recorded."
    assert describe_provenance({"source": None}) == "Source: no record was given."


@pytest.mark.parametrize(
    ("provenance", "error", "message"),
    [
        ([1, 2], TypeError, "provenance must be a mapping, got list"),
        ({"x": np.nan}, ValueError, r"provenance\['x'\] is nan"),
        ({"operations": {"a": 1}}, ValueError, "must be a list of operation records"),
    ],
)
def test_describe_provenance_refuses_what_is_not_a_record(
    provenance: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        describe_provenance(provenance)
