"""Tests of CubeCollection and of StudyResults.table, which lines up the studies'
results.

The reference values come from validation/cube_collection.py --write-fixtures: three
small studies (``lupus``, ``covid`` and ``atlas``), anndata's ``concat`` of their
exported AnnData (outer and inner joins), pandas' ``concat`` of their sample tables,
and limma run in R on each study's own counts and on the stacked counts of two of
them. A smaller example below is worked out by hand.
"""

import copy
import json
import pickle
import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cytepool import (
    Cube,
    CubeCollection,
    Design,
    DifferentialExpressionResult,
    StudyResults,
    limma_voom,
    meta_analysis,
    read_cube,
    write_cube,
)

DATA = Path(__file__).parent / "data" / "cube_collection"
FORMULAS = {
    "lupus": "~ donor + condition",
    "covid": "~ donor + condition",
    "atlas": "~ condition",
}
LEVELS = {"condition": ["ctrl", "stim"]}
# limma's topTable columns, in the order the fixtures store them, and CytePool's.
STATISTICS = (
    "log2_fold_change",
    "average_log2_cpm",
    "t",
    "p_value",
    "adjusted_p_value",
    "log_odds",
    "standard_error",
)


@cache
def fixture() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = dict(np.load(DATA / "studies.npz"))
    record = json.loads((DATA / "studies.json").read_text())
    return arrays, record


@cache
def fixture_cubes() -> dict[str, Cube]:
    arrays, record = fixture()
    return {
        name: Cube(
            {layer: arrays[f"{name}/{layer}"] for layer in ("counts", "detected")},
            samples=tables["samples"],
            groups=tables["groups"],
            features=tables["features"],
            n_cells=arrays[f"{name}/n_cells"],
        )
        for name, tables in record["studies"].items()
    }


@cache
def fixture_collection() -> CubeCollection:
    return CubeCollection(
        fixture_cubes(),
        align_groups_by="cell_ontology_id",
        group_mapping=fixture()[1]["mapping"],
        within_study=["donor"],
    )


@cache
def fixture_results() -> dict[str, DifferentialExpressionResult]:
    collection = fixture_collection()
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        design = Design(cube.samples, FORMULAS[study], levels=LEVELS)
        results[study] = limma_voom(cube, design, contrasts={"stim": "conditionstim"})
    return results


def scaled_difference(ours: np.ndarray, reference: np.ndarray, p_value: bool) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|), as validation/cube_collection.py scales them."""
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale, initial=0.0))


def limma_table(key: str) -> tuple[list[str], np.ndarray]:
    arrays, _ = fixture()
    return arrays[f"limma/{key}/genes"].tolist(), arrays[f"limma/{key}/values"]


# The references: pandas, anndata and limma.


def test_samples_are_those_of_pandas_concat() -> None:
    samples = fixture_collection().samples
    reference = fixture()[1]["samples"]
    assert set(samples.columns) == set(reference)
    assert samples.columns[:2] == ("sample_id", "study")
    for column, expected in reference.items():
        ours = samples[column].tolist()
        missing = [v is None or (isinstance(v, float) and np.isnan(v)) for v in ours]
        assert missing == [value is None for value in expected["values"]], column
        present = [v for v, gap in zip(ours, missing, strict=True) if not gap]
        assert present == [v for v in expected["values"] if v is not None], column
    # Integer ages in two studies and floats in one become float64, as in pandas.
    assert samples["age"].dtype == np.float64
    assert reference["age"]["dtype"] == "float64"
    # Sample and donor ids repeat across studies and are kept apart by the study.
    assert len(set(samples["sample_id"].tolist())) == len(samples)
    assert samples["donor"][0] == "lupus::D1" and "covid::D1" in samples["donor"]


def test_features_and_measured_are_anndata_outer_join() -> None:
    collection = fixture_collection()
    arrays, _ = fixture()
    genes = collection.features["feature_id"].tolist()
    assert genes == arrays["first_listed"].tolist()
    assert sorted(genes) == sorted(arrays["outer_genes"].tolist())
    where = {gene: k for k, gene in enumerate(arrays["outer_genes"].tolist())}
    counts = arrays["outer_counts"][:, [where[gene] for gene in genes]]
    for row, study in enumerate(collection.studies):
        mine = counts[arrays["outer_study"] == study]
        # The outer join fills the genes a study lacks with NaN, stored as -1.
        assert_array_equal((mine >= 0).all(axis=0), collection.measured[row])
        assert_array_equal((mine < 0).all(axis=0), ~collection.measured[row])
        cube = collection.aligned_cube(study)
        groups = cube.groups["group_id"].tolist()
        samples = cube.samples["sample_id"].tolist()
        positions = (
            [groups.index(g) for g in arrays["outer_group"][arrays["outer_study"] == study]],
            [samples.index(s) for s in arrays["outer_sample_id"][arrays["outer_study"] == study]],
        )  # fmt: skip
        assert_array_equal(
            cube.layer("counts")[positions], mine[:, collection.measured[row]]
        )


def test_stack_is_anndata_inner_join() -> None:
    collection = fixture_collection()
    arrays, _ = fixture()
    stacked = collection.stack()
    assert stacked.features["feature_id"].tolist() == arrays["inner_genes"].tolist()
    assert set(stacked.groups["group_id"]) == set(arrays["inner_group"].tolist())
    assert stacked.samples.equals(collection.samples)
    groups = [
        stacked.groups["group_id"].tolist().index(g) for g in arrays["inner_group"]
    ]
    ids = stacked.samples["sample_id"].tolist()
    keys = zip(arrays["inner_study"], arrays["inner_sample_id"], strict=True)
    samples = [ids.index(f"{study}::{sample}") for study, sample in keys]
    for layer in ("counts", "detected"):
        assert_array_equal(
            stacked.layer(layer)[groups, samples], arrays[f"inner_{layer}"]
        )
    cells = stacked.n_cells
    assert cells is not None
    assert_array_equal(cells[samples, groups], arrays["inner_n_cells"])
    # An entry without a row in the join has no cells, and zeros in every layer.
    empty = np.ones(stacked.layer("counts").shape[:2], dtype=bool)
    empty[groups, samples] = False
    assert empty.any() and not cells.T[empty].any()
    assert not stacked.layer("counts")[empty].any()


def test_studies_analysed_alone_and_lined_up_give_limma() -> None:
    collection = fixture_collection()
    studies = StudyResults(collection, fixture_results())
    arrays, _ = fixture()
    for group in collection.groups["group_id"].tolist():
        table = studies.table(group, "stim")
        assert len(table) == collection.n_studies * collection.n_features
        for row, study in enumerate(collection.studies):
            mine = table.filter(table["study"] == study)
            key = f"{study}__{group.replace(':', '_')}"
            if f"limma/{key}/genes" not in arrays:  # the study lacks the group
                assert set(mine["status"].tolist()) <= {"not_analysed", "not_measured"}
                assert np.isnan(mine["t"]).all()
                continue
            genes, values = limma_table(key)
            where = {gene: k for k, gene in enumerate(mine["feature_id"].tolist())}
            positions = [where[gene] for gene in genes]
            expected = np.where(collection.measured[row], "not_tested", "not_measured")
            expected = expected.astype(object)
            expected[positions] = "tested"
            assert_array_equal(mine["status"], expected)
            for k, column in enumerate(STATISTICS):
                difference = scaled_difference(
                    mine[column][positions], values[k], "p_value" in column
                )
                assert difference <= 1e-10, (study, group, column, difference)


def test_the_stack_of_two_studies_fits_their_joint_model_as_limma() -> None:
    stacked = fixture_collection().select(studies=["lupus", "covid"]).stack()
    design = Design(stacked.samples, "~ donor + condition", levels=LEVELS)
    result = limma_voom(stacked, design, contrasts={"stim": "conditionstim"})
    for group in stacked.groups["group_id"].tolist():
        genes, values = limma_table(f"joint__{group.replace(':', '_')}")
        table = result.table(group, "stim")
        assert table["feature_id"].tolist() == genes
        for k, column in enumerate(STATISTICS):
            difference = scaled_difference(
                table[column], values[k], "p_value" in column
            )
            assert difference <= 1e-10, (group, column, difference)


def test_lined_up_results_feed_the_meta_analysis() -> None:
    results = fixture_results()
    meta = meta_analysis(results)
    collection = fixture_collection()
    group = collection.groups["group_id"][0]
    table = StudyResults(collection, results).table(group, "stim")
    lined_up = meta.table(group, "stim")
    tested = table["status"] == "tested"
    counts = {
        gene: int(np.count_nonzero(tested & (table["feature_id"] == gene)))
        for gene in lined_up["feature_id"].tolist()
    }
    assert lined_up["n_studies"].tolist() == list(counts.values())


# A small example worked out by hand.


def study_a() -> Cube:
    return Cube(
        {"counts": np.arange(1, 13, dtype=np.int64).reshape(2, 2, 3)},
        samples={
            "sample_id": ["S1", "S2"],
            "donor": ["D1", "D1"],
            "condition": ["ctrl", "stim"],
            "age": [30, 31],
        },
        groups={"group_id": ["B cell", "T cell"], "cell_ontology_id": ["CL:1", "CL:2"]},
        features={"feature_id": ["g1", "g2", "g3"], "symbol": ["G1", "G2", None]},
        n_cells=np.array([[5, 6], [7, 8]]),
        provenance={"source": {"study": "A"}},
    )


def study_b() -> Cube:
    # Features g2, g4, g3 and g5, which is not measured and filled with zeros.
    counts = np.array(
        [
            [[1, 0, 2, 0], [3, 1, 4, 0]],
            [[5, 1, 6, 0], [7, 2, 8, 0]],
            [[9, 3, 10, 0], [11, 4, 12, 0]],
            [[13, 5, 14, 0], [15, 6, 16, 0]],
        ],
        dtype=np.int64,
    )
    return Cube(
        {"counts": counts},
        samples={
            "sample_id": ["S1", "S3"],
            "donor": ["D1", "D2"],
            "condition": ["ctrl", "stim"],
            "age": [40.5, None],
            "batch": ["x", "y"],
        },
        groups={
            "group_id": ["B", "CD4", "CD8", "NK"],
            "cell_ontology_id": ["CL:1", "CL:2", "CL:2", "CL:3"],
        },
        features={
            "feature_id": ["g2", "g4", "g3", "g5"],
            "symbol": ["G2b", "G4", "G3", "G5"],
            "measured": [True, True, True, False],
        },
        n_cells=np.array([[3, 4, 5, 6], [7, 8, 9, 10]]),
    )


def test_a_small_collection_worked_out_by_hand() -> None:
    a, b = study_a(), study_b()
    collection = CubeCollection({"A": a, "B": b}, within_study="donor")
    assert collection.studies == ("A", "B")
    assert collection.samples.to_dict()["sample_id"].tolist() == [
        "A::S1", "A::S2", "B::S1", "B::S3"
    ]  # fmt: skip
    assert collection.samples["donor"].tolist() == ["A::D1", "A::D1", "B::D1", "B::D2"]
    assert collection.samples["study"].tolist() == ["A", "A", "B", "B"]
    assert_array_equal(collection.samples["age"], [30.0, 31.0, 40.5, np.nan])
    assert collection.samples["batch"].tolist() == [None, None, "x", "y"]
    # Genes in the order the studies first list them; g5 is measured by none.
    assert collection.features["feature_id"].tolist() == ["g1", "g2", "g3", "g4"]
    assert collection.features["symbol"].tolist() == ["G1", "G2", "G3", "G4"]
    assert_array_equal(
        collection.measured, [[True, True, True, False], [False, True, True, True]]
    )
    # Every cube has cell_ontology_id, which aligns the groups; CD4 and CD8 merge.
    assert collection.groups["group_id"].tolist() == ["CL:1", "CL:2", "CL:3"]
    assert collection.groups["cell_ontology_id"].tolist() == ["CL:1", "CL:2", "CL:3"]
    members = collection.group_members
    assert list(zip(*members.to_dict().values(), strict=True)) == [
        ("A", "CL:1", "B cell"),
        ("A", "CL:2", "T cell"),
        ("B", "CL:1", "B"),
        ("B", "CL:2", "CD4"),
        ("B", "CL:2", "CD8"),
        ("B", "CL:3", "NK"),
    ]
    aligned = collection.aligned_cube("B")
    assert aligned.groups["group_id"].tolist() == ["CL:1", "CL:2", "CL:3"]
    assert aligned.features["feature_id"].tolist() == ["g2", "g3", "g4"]
    assert aligned.samples["sample_id"].tolist() == ["S1", "S3"]
    assert_array_equal(
        aligned.layer("counts"),
        [
            [[1, 2, 0], [3, 4, 1]],
            [[5 + 9, 6 + 10, 1 + 3], [7 + 11, 8 + 12, 2 + 4]],
            [[13, 14, 5], [15, 16, 6]],
        ],
    )
    assert_array_equal(aligned.n_cells, [[3, 4 + 5, 6], [7, 8 + 9, 10]])
    stacked = collection.stack()
    assert stacked.groups["group_id"].tolist() == ["CL:1", "CL:2"]
    assert stacked.features["feature_id"].tolist() == ["g2", "g3"]
    assert stacked.samples.equals(collection.samples)
    assert_array_equal(
        stacked.layer("counts"),
        [
            [[2, 3], [5, 6], [1, 2], [3, 4]],
            [[8, 9], [11, 12], [14, 16], [18, 20]],
        ],
    )
    assert_array_equal(stacked.n_cells, [[5, 6], [7, 8], [3, 9], [7, 17]])
    assert stacked.normalization is None


def test_groups_by_id_with_a_mapping_that_renames_merges_and_leaves_out() -> None:
    mapping = {"B": {"B": "B cell", "CD4": "T cell", "CD8": "T cell", "NK": None}}
    collection = CubeCollection(
        {"A": study_a(), "B": study_b()},
        align_groups_by="group_id",
        group_mapping=mapping,
    )
    assert collection.groups["group_id"].tolist() == ["B cell", "T cell"]
    # The studies agree on the ontology ids of both groups.
    assert collection.groups["cell_ontology_id"].tolist() == ["CL:1", "CL:2"]
    aligned = collection.aligned_cube("B")
    assert aligned.groups["group_id"].tolist() == ["B cell", "T cell"]
    assert_array_equal(aligned.layer("counts", "T cell"), [[14, 16, 4], [18, 20, 6]])
    # The cube of A already has the aligned ids and the collection's features.
    assert collection.aligned_cube("A") is collection.cubes["A"]


def test_a_group_keeps_a_column_where_the_studies_that_have_it_agree() -> None:
    a = study_a()
    lineage = Cube(
        {"counts": a.layer("counts")},
        samples=a.samples,
        groups=a.groups.with_columns(lineage=["lymphoid", "lymphoid"]),
        features=a.features,
        n_cells=a.n_cells,
    )
    # T cell merges CD4, CD8 (CL:2) and NK (CL:3) of study B.
    mapping = {"B": {"B": "B cell", "CD4": "T cell", "CD8": "T cell", "NK": "T cell"}}
    collection = CubeCollection(
        {"A": lineage, "B": study_b()},
        align_groups_by="group_id",
        group_mapping=mapping,
    )
    # Study B has no lineage column, so A's values stand; the members of T cell
    # disagree on their ontology id, which is then missing.
    assert collection.groups["lineage"].tolist() == ["lymphoid", "lymphoid"]
    assert collection.groups["cell_ontology_id"].tolist() == ["CL:1", None]
    assert_array_equal(
        collection.aligned_cube("B").n_cells, [[3, 4 + 5 + 6], [7, 8 + 9 + 10]]
    )


def test_groups_without_an_ontology_id_take_one_from_the_mapping() -> None:
    a = study_a()
    c = Cube(
        {"counts": np.ones((2, 1, 2), dtype=np.int64)},
        samples={"sample_id": ["S1"]},
        groups={"group_id": ["Bcells", "junk"], "cell_ontology_id": ["CL:1", None]},
        features={"feature_id": ["g1", "g2"]},
    )
    with pytest.raises(ValueError, match=r"group 'junk' of study 'C' has no cell_onto"):
        CubeCollection({"A": a, "C": c})
    collection = CubeCollection({"A": a, "C": c}, group_mapping={"C": {"junk": None}})
    assert collection.group_members["study_group_id"].tolist() == [
        "B cell", "T cell", "Bcells"
    ]  # fmt: skip
    # A group column that only some studies have aligns only with a mapping for
    # the others.
    plain = Cube(
        {"counts": np.ones((1, 1, 1), dtype=np.int64)},
        samples={"sample_id": ["S1"]},
        groups={"group_id": ["B"]},
        features={"feature_id": ["g1"]},
    )
    with pytest.raises(ValueError, match=r"aligned by 'cell_ontology_id', which its"):
        CubeCollection({"A": a, "P": plain}, align_groups_by="cell_ontology_id")
    by_mapping = CubeCollection(
        {"A": a, "P": plain},
        align_groups_by="cell_ontology_id",
        group_mapping={"P": {"B": "CL:1"}},
    )
    assert by_mapping.groups["group_id"].tolist() == ["CL:1", "CL:2"]
    # Without the column in every cube, groups are aligned by group_id.
    assert CubeCollection({"A": a, "P": plain}, group_mapping={"P": {"B": "B cell"}})


def test_group_names_name_the_aligned_groups_and_change_no_value() -> None:
    cubes = {"A": study_a(), "B": study_b()}
    names = {"CL:2": "T cells", "CL:1": "B cells"}
    named = CubeCollection(cubes, group_names=names)
    plain = CubeCollection(cubes)
    assert named.groups["group_id"].tolist() == ["B cells", "T cells", "CL:3"]
    # The aligned column keeps each group's ontology id.
    assert named.groups["cell_ontology_id"].tolist() == ["CL:1", "CL:2", "CL:3"]
    assert named.group_members["group_id"].tolist() == [
        "B cells", "T cells", "B cells", "T cells", "T cells", "CL:3"
    ]  # fmt: skip
    for study in ("A", "B"):
        aligned, expected = named.aligned_cube(study), plain.aligned_cube(study)
        assert aligned.groups["group_id"].tolist() == [
            names.get(group, group) for group in expected.groups["group_id"].tolist()
        ]
        assert aligned.groups.select(["cell_ontology_id"]).equals(
            expected.groups.select(["cell_ontology_id"])
        )
        assert_array_equal(aligned.layer("counts"), expected.layer("counts"))
        assert_array_equal(aligned.n_cells, expected.n_cells)
    assert named.aligned_cube("B").provenance["operations"][-1]["membership"] == {
        "B cells": ["B"], "T cells": ["CD4", "CD8"], "CL:3": ["NK"]
    }  # fmt: skip
    stacked = named.stack()
    assert stacked.groups["group_id"].tolist() == ["B cells", "T cells"]
    assert_array_equal(stacked.layer("counts"), plain.stack().layer("counts"))
    # The records hold the names as given, and nothing when none are given.
    assert named.provenance["alignment"]["group_names"] == names
    assert list(named.provenance["alignment"]["group_names"]) == ["CL:2", "CL:1"]
    assert stacked.provenance["construction"]["group_names"] == names
    assert "group_names" not in plain.provenance["alignment"]
    assert "group_names" not in plain.stack().provenance["construction"]
    selected = named.select(groups=["T cells"])
    assert selected.groups["group_id"].tolist() == ["T cells"]
    assert selected.groups["cell_ontology_id"].tolist() == ["CL:2"]
    # Groups aligned by group_id are renamed in that column.
    mapping = {"B": {"B": "B cell", "CD4": "T cell", "CD8": "T cell", "NK": None}}
    by_id = CubeCollection(
        cubes,
        align_groups_by="group_id",
        group_mapping=mapping,
        group_names={"T cell": "T lymphocytes"},
    )
    assert by_id.groups["group_id"].tolist() == ["B cell", "T lymphocytes"]


@pytest.mark.parametrize(
    ("group_names", "error", "message"),
    [
        (["CL:1"], TypeError, "group_names must map aligned group ids to names"),
        ({1: "B cells"}, TypeError, "keyed by aligned group ids (str), got int 1"),
        ({"CL:1": 3}, TypeError, "group_names['CL:1'] must be a name (str), got int"),
        ({"CL:1": ""}, ValueError, "group_names['CL:1'] is empty"),
        ({"CL:9": "x"}, KeyError, "names the aligned group 'CL:9', which no study has"),
        (
            {"CL:1": "T", "CL:2": "T"},
            ValueError,
            "the aligned groups 'CL:1' and 'CL:2' would both be named 'T'",
        ),
        (
            {"CL:1": "CL:3"},
            ValueError,
            "the aligned groups 'CL:1' and 'CL:3' would both be named 'CL:3'",
        ),
    ],
)
def test_invalid_group_names_are_refused(
    group_names: Any, error: type, message: str
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        CubeCollection({"A": study_a(), "B": study_b()}, group_names=group_names)


# Invariants.


def test_the_order_of_studies_changes_no_value() -> None:
    a, b = study_a(), study_b()
    forward = CubeCollection({"A": a, "B": b})
    backward = CubeCollection({"B": b, "A": a})
    assert backward.features["feature_id"].tolist() == ["g2", "g4", "g3", "g1"]
    for study in ("A", "B"):
        ours = forward.aligned_cube(study)
        theirs = backward.aligned_cube(study)
        ids = theirs.features["feature_id"].tolist()
        assert ours.select(features=ids).fingerprint() == theirs.fingerprint()
        row = forward.studies.index(study)
        measured = dict(
            zip(forward.features["feature_id"].tolist(), forward.measured[row].tolist())
        )
        genes = backward.features["feature_id"].tolist()
        other = backward.studies.index(study)
        assert [measured[g] for g in genes] == backward.measured[other].tolist()
    stacked = backward.stack()
    reordered = forward.stack().select(samples=stacked.samples["sample_id"])
    assert_array_equal(reordered.layer("counts"), stacked.layer("counts"))
    assert_array_equal(reordered.n_cells, stacked.n_cells)
    assert reordered.samples.equals(stacked.samples)
    assert reordered.groups.equals(stacked.groups)
    # Only the symbol of g2 differs: the first study that gives one names it.
    assert reordered.features["symbol"].tolist() == ["G2", "G3"]
    assert stacked.features["symbol"].tolist() == ["G2b", "G3"]


def test_the_order_of_genes_within_a_cube_changes_no_value() -> None:
    a, b = study_a(), study_b()
    order = [3, 2, 0, 1]
    shuffled = b.select(features=order)
    first = CubeCollection({"A": a, "B": b})
    second = CubeCollection({"A": a, "B": shuffled})
    # The shared genes come in A's order, so the stacked cube is the same.
    assert first.stack().fingerprint() == second.stack().fingerprint()
    assert (
        first.aligned_cube("B").fingerprint() == second.aligned_cube("B").fingerprint()
    )
    assert_array_equal(first.measured, second.measured)


def test_selecting_then_stacking_is_stacking_then_selecting() -> None:
    collection = CubeCollection({"A": study_a(), "B": study_b()}, within_study="donor")
    chosen = {"groups": ["CL:2"], "features": ["g3", "g2"]}
    selected = collection.select(**chosen).stack()
    assert selected.fingerprint() == collection.stack().select(**chosen).fingerprint()
    assert collection.stack(**chosen).fingerprint() == selected.fingerprint()
    # The order of a selection is that of every study's aligned cube.
    reordered = collection.select(groups=["CL:2", "CL:1"]).aligned_cube("B")
    assert reordered.groups["group_id"].tolist() == ["CL:2", "CL:1"]
    assert_array_equal(
        reordered.layer("counts"),
        collection.aligned_cube("B").select(groups=["CL:2", "CL:1"]).layer("counts"),
    )
    # A study selected alone stacks to its own aligned cube with keyed samples.
    alone = collection.select(studies="B")
    assert alone.studies == ("B",)
    assert alone.features["feature_id"].tolist() == ["g2", "g3", "g4"]
    assert alone.groups["group_id"].tolist() == ["CL:1", "CL:2", "CL:3"]
    assert "age" in alone.samples and "batch" in alone.samples
    stacked = alone.stack()
    assert_array_equal(
        stacked.layer("counts"), collection.aligned_cube("B").layer("counts")
    )
    assert stacked.samples["sample_id"].tolist() == ["B::S1", "B::S3"]


def test_a_single_study_is_its_own_cube() -> None:
    a = study_a()
    collection = CubeCollection({"A": a}, align_groups_by="group_id")
    assert collection.aligned_cube("A") is a
    assert_array_equal(collection.measured, [[True, True, True]])
    stacked = collection.stack()
    assert_array_equal(stacked.layer("counts"), a.layer("counts"))
    assert_array_equal(stacked.n_cells, a.n_cells)
    assert stacked.groups.equals(a.groups)
    assert stacked.features.equals(a.features)
    assert stacked.samples["sample_id"].tolist() == ["A::S1", "A::S2"]


def test_cube_files_stay_lazy_and_align_as_cubes_in_memory(tmp_path: Path) -> None:
    a, b = study_a(), study_b()
    write_cube(a, tmp_path / "a")
    write_cube(b, tmp_path / "b")
    lazy = CubeCollection(
        {"A": read_cube(tmp_path / "a"), "B": read_cube(tmp_path / "b")}
    )
    eager = CubeCollection({"A": a, "B": b})
    aligned = lazy.aligned_cube("B")
    assert "lazy" in repr(aligned)
    assert aligned.fingerprint() == eager.aligned_cube("B").fingerprint()
    assert lazy.stack().fingerprint() == eager.stack().fingerprint()
    # A cube file's fingerprint is the one stored with it: no layer is read.
    assert lazy.provenance["studies"]["B"]["fingerprint"] == b.fingerprint()


def test_stacked_layers_take_a_type_that_holds_every_study() -> None:
    def cube(counts: np.ndarray) -> Cube:
        return Cube(
            {"counts": counts},
            samples={"sample_id": ["S1"]},
            groups={"group_id": ["B"]},
            features={"feature_id": ["g1", "g2"]},
        )

    small = cube(np.array([[[1, 2]]], dtype=np.uint32))
    wide = cube(np.array([[[3, 4]]], dtype=np.uint64))
    real = cube(np.array([[[0.5, 2.0]]]))
    assert CubeCollection({"A": small, "B": small}).stack().layer("counts").dtype == (
        np.uint32
    )
    assert CubeCollection({"A": small, "B": wide}).stack().layer("counts").dtype == (
        np.int64
    )
    mixed = CubeCollection({"A": small, "C": real}).stack().layer("counts")
    assert mixed.dtype == np.float64
    assert_array_equal(mixed, [[[1.0, 2.0], [0.5, 2.0]]])
    with pytest.raises(ValueError, match="holds whole numbers exactly only below"):
        CubeCollection(
            {"A": cube(np.array([[[2**53, 1]]], np.uint64)), "C": real}
        ).stack()


def _as_uint32(cube: Cube) -> Cube:
    return Cube(
        {"counts": cube.layer("counts").astype(np.uint32)},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
        provenance=cube.provenance,
    )


def test_groups_only_renamed_keep_their_integer_dtype() -> None:
    wide = {"A": study_a(), "B": study_b()}
    small = {study: _as_uint32(cube) for study, cube in wide.items()}
    # Aligned by ontology ids, A's groups are renamed (CL:1, CL:2) and B's CD4 and
    # CD8 merge into CL:2; without CD8, B's groups are only renamed too.
    for mapping, dtype_b in ((None, np.int64), ({"B": {"CD8": None}}, np.uint32)):
        collection = CubeCollection(small, group_mapping=mapping)
        reference = CubeCollection(wide, group_mapping=mapping)
        for study, dtype in (("A", np.uint32), ("B", dtype_b)):
            aligned = collection.aligned_cube(study)
            assert aligned.groups["group_id"].tolist()[:2] == ["CL:1", "CL:2"]
            assert aligned.layer("counts").dtype == dtype
            assert_array_equal(
                aligned.layer("counts"), reference.aligned_cube(study).layer("counts")
            )
        # The stack takes the type that holds every study's part.
        stacked = collection.stack().layer("counts")
        assert stacked.dtype == (np.int64 if dtype_b == np.int64 else np.uint32)
        assert_array_equal(stacked, reference.stack().layer("counts"))


def test_normalized_layers_stack_only_from_studies_normalized_alike() -> None:
    def cube(target: float) -> Cube:
        return Cube(
            {
                "counts": np.ones((1, 1, 2), dtype=np.int64),
                "log_sum": np.full((1, 1, 2), 0.5),
            },
            samples={"sample_id": ["S1"]},
            groups={"group_id": ["B"]},
            features={"feature_id": ["g1", "g2"]},
            n_cells=np.array([[2]]),
            normalization={"method": "library_size", "target_sum": target},
        )

    same = CubeCollection({"A": cube(1e4), "B": cube(1e4)}).stack()
    assert same.layer_names == ("counts", "log_sum")
    assert same.normalization == {"method": "library_size", "target_sum": 1e4}
    differing = CubeCollection({"A": cube(1e4), "B": cube(1e6)})
    with pytest.raises(ValueError, match=r"would mix two normalizations"):
        differing.stack()
    counts = differing.stack(layers="counts")
    assert counts.layer_names == ("counts",) and counts.normalization is None


# Provenance, pickling and immutability.


def test_provenance_records_each_cube_and_the_alignment() -> None:
    a, b = study_a(), study_b()
    collection = CubeCollection({"A": a, "B": b}, within_study=["donor"])
    record = collection.provenance
    assert json.loads(json.dumps(record)) == record
    assert record["studies"] == {
        "A": {"fingerprint": a.fingerprint(), "provenance": {"source": {"study": "A"}}},
        "B": {"fingerprint": b.fingerprint(), "provenance": {}},
    }
    alignment = record["alignment"]
    assert alignment["align_groups_by"] == "cell_ontology_id"
    assert alignment["group_mapping"] is None
    assert alignment["within_study"] == ["donor"]
    assert alignment["groups"]["B"] == {
        "CL:1": ["B"], "CL:2": ["CD4", "CD8"], "CL:3": ["NK"]
    }  # fmt: skip
    assert record["operations"] == []
    selected = collection.select(studies="B", features=["g4", "g2"])
    assert selected.provenance["operations"] == [
        {
            "operation": "select",
            "studies": ["B"],
            "groups": None,
            "features": ["g4", "g2"],
        }
    ]
    assert selected.provenance["alignment"] == alignment
    # The aligned cube records how it was derived from the cube as given.
    operations = collection.aligned_cube("B").provenance["operations"]
    assert [op["operation"] for op in operations] == ["select", "regroup"]
    assert operations[0]["input"] == b.fingerprint()
    assert operations[0]["features"] == ["g2", "g3", "g4"]
    assert operations[1]["membership"] == {"CL:2": ["CD4", "CD8"]} | {
        "CL:1": ["B"], "CL:3": ["NK"]
    }  # fmt: skip


def test_a_stacked_cube_records_the_studies_it_stacks() -> None:
    a, b = study_a(), study_b()
    stacked = CubeCollection({"A": a, "B": b}, within_study="donor").stack()
    construction = stacked.provenance["construction"]
    assert construction["function"] == "CubeCollection.stack"
    assert construction["studies"]["B"] == {
        "fingerprint": b.fingerprint(),
        "provenance": {},
        "groups": {"CL:1": ["B"], "CL:2": ["CD4", "CD8"]},
    }
    assert construction["studies"]["A"]["provenance"] == a.provenance
    assert construction["align_groups_by"] == "cell_ontology_id"
    assert construction["within_study"] == ["donor"]
    assert construction["layers"] == ["counts"]
    history = stacked.history()
    assert "from 2 studies" in history
    # Lines of at most 80 characters: a fingerprint goes on a line of its own.
    assert f"study 'B': the cube {b.fingerprint()}" in " ".join(history.split())
    assert max(map(len, history.splitlines())) <= 80
    assert "    CL:2: CD4 + CD8" in history
    assert "sample columns written study::value: donor" in history


def test_pickling_copying_and_immutability() -> None:
    collection = CubeCollection({"A": study_a(), "B": study_b()}).select(
        features=["g2", "g3"]
    )
    restored = pickle.loads(pickle.dumps(collection))
    assert restored.studies == collection.studies
    for name in ("samples", "groups", "features", "group_members"):
        assert getattr(restored, name).equals(getattr(collection, name))
    assert_array_equal(restored.measured, collection.measured)
    assert restored.provenance == collection.provenance
    assert repr(restored) == repr(collection)
    assert restored.stack().fingerprint() == collection.stack().fingerprint()
    for measured in (collection.measured, restored.measured):
        with pytest.raises(ValueError, match="read-only"):
            measured[0, 0] = False
    with pytest.raises(AttributeError, match="immutable"):
        collection.studies = ()  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del collection._studies
    assert copy.copy(collection) is collection
    assert copy.deepcopy(collection) is collection
    with pytest.raises(TypeError):
        collection.cubes["A"] = study_a()  # type: ignore[index]


def test_repr_reports_the_studies_and_what_they_share() -> None:
    lines = repr(CubeCollection({"A": study_a(), "B": study_b()})).splitlines()
    assert lines == [
        "CubeCollection of 2 studies: 4 samples, 3 groups, 4 features",
        "  groups aligned by 'cell_ontology_id'",
        "  A: 2 samples, 2 groups, 3 features measured",
        "  B: 2 samples, 3 groups (1 merged), 3 features measured",
        "  in every study: 2 groups, 2 features",
    ]


# What goes wrong in practice.


@pytest.mark.parametrize(
    ("cubes", "error", "message"),
    [
        ([study_a()], TypeError, "cubes must be a mapping of study names"),
        ({}, ValueError, "cubes is empty"),
        ({"a::b": study_a()}, ValueError, r"'a::b' holds '::', which separates"),
        ({"": study_a()}, ValueError, "study names must be non-empty"),
        ({1: study_a()}, TypeError, "study names must be strings"),
        ({"A": "cube"}, TypeError, r"cubes\['A'\] must be a Cube, got str"),
    ],
)
def test_invalid_studies_are_refused(cubes: Any, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        CubeCollection(cubes)


def test_studies_that_share_no_gene_or_group_are_refused() -> None:
    a = study_a()
    versioned = Cube(
        {"counts": np.ones((2, 1, 2), dtype=np.int64)},
        samples={"sample_id": ["S1"]},
        groups=a.groups,
        features={"feature_id": ["g1.7", "g2.7"]},
    )
    with pytest.raises(
        ValueError, match=r"study 'A' shares no feature .* differ only in a version"
    ):
        CubeCollection({"A": a, "V": versioned})
    other = Cube(
        {"counts": np.ones((1, 1, 2), dtype=np.int64)},
        samples={"sample_id": ["S1"]},
        groups={"group_id": ["neuron"], "cell_ontology_id": ["CL:0000540"]},
        features={"feature_id": ["g1", "g2"]},
    )
    with pytest.raises(
        ValueError, match=r"study 'A' shares no group .* theirs 'CL:0000540'"
    ):
        CubeCollection({"A": a, "N": other})
    unmeasured = Cube(
        {"counts": np.zeros((2, 1, 1), dtype=np.int64)},
        samples={"sample_id": ["S1"]},
        groups=a.groups,
        features={"feature_id": ["g1"], "measured": [False]},
    )
    with pytest.raises(ValueError, match="study 'U' measured no feature"):
        CubeCollection({"A": a, "U": unmeasured})


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ({"Z": {}}, KeyError, r"names the study 'Z', which is not in cubes"),
        ({"B": {"CD5": "T"}}, KeyError, r"group 'CD5', which the cube of study 'B'.*did you mean 'CD8' or 'CD4'"),
        ({"B": {"CD4": 4}}, TypeError, r"group_mapping\['B'\]\['CD4'\] must be an aligned id"),
        ({"B": {"CD4": ""}}, ValueError, "is empty; aligned ids are non-empty"),
        ({"B": ["CD4"]}, TypeError, r"group_mapping\['B'\] must map"),
        (["B"], TypeError, "group_mapping must map study names"),
        ({"B": {name: None for name in ("B", "CD4", "CD8", "NK")}}, ValueError, "study 'B' has no group"),
    ],
)  # fmt: skip
def test_invalid_group_mappings_are_refused(
    mapping: Any, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        CubeCollection({"A": study_a(), "B": study_b()}, group_mapping=mapping)


def test_sample_columns_must_hold_one_kind_of_values() -> None:
    a = study_a()
    b = study_b()
    text_age = Cube(
        {"counts": b.layer("counts")},
        samples=b.samples.with_columns(age=["forty", None]),
        groups=b.groups,
        features=b.features,
        n_cells=b.n_cells,
    )
    with pytest.raises(
        TypeError,
        match=r"sample column 'age' holds integers \(int64\) in study 'A' and text",
    ):
        CubeCollection({"A": a, "B": text_age})
    named = Cube(
        {"counts": b.layer("counts")},
        samples=b.samples.with_columns(study=["other", "other"]),
        groups=b.groups,
        features=b.features,
        n_cells=b.n_cells,
    )
    with pytest.raises(ValueError, match=r"sample column 'study' with the value 'oth"):
        CubeCollection({"A": a, "B": named})


@pytest.mark.parametrize(
    ("within", "error", "message"),
    [
        (["donr"], KeyError, r"'donr', which is not a sample column.*'donor'"),
        (["sample_id"], ValueError, "which the collection writes itself"),
        (["age"], TypeError, r"'age', whose values in study 'B' are float64"),
        (["donor", "donor"], ValueError, "names 'donor' more than once"),
        ([1], TypeError, "within_study must name sample columns"),
        (3, TypeError, "within_study must be a sample column name or a list"),
    ],
)
def test_invalid_within_study_columns_are_refused(
    within: Any, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        CubeCollection({"A": study_a(), "B": study_b()}, within_study=within)


def test_invalid_alignment_columns_are_refused() -> None:
    cubes = {"A": study_a(), "B": study_b()}
    with pytest.raises(KeyError, match=r"'cell_type', which is not a group column"):
        CubeCollection(cubes, align_groups_by="cell_type")
    with pytest.raises(TypeError, match="align_groups_by must be a group column name"):
        CubeCollection(cubes, align_groups_by=1)  # type: ignore[arg-type]


def test_stacking_refuses_what_a_study_lacks() -> None:
    collection = CubeCollection({"A": study_a(), "B": study_b()})
    with pytest.raises(ValueError, match=r"feature 'g4' is not in study 'A'"):
        collection.stack(features=["g2", "g4"])
    with pytest.raises(ValueError, match=r"group 'CL:3' is not in study 'A'"):
        collection.stack(groups=["CL:3"])
    with pytest.raises(ValueError, match="no group is in every study"):
        collection.select(groups=["CL:3"]).stack()
    with pytest.raises(ValueError, match="layers must include 'counts'"):
        collection.stack(layers=[])
    with pytest.raises(ValueError, match="unknown layer 'means'"):
        collection.stack(layers=["counts", "means"])
    with pytest.raises(ValueError, match=r"study 'A' has no 'detected' layer"):
        CubeCollection({"A": study_a(), "B": _with_detected(study_b())}).stack(
            layers=["counts", "detected"]
        )
    bulk = Cube(
        {"counts": np.ones((1, 2, 3), dtype=np.int64)},
        samples={"sample_id": ["S1", "S2"]},
        groups={"group_id": ["B cell"], "cell_ontology_id": ["CL:1"]},
        features={"feature_id": ["g1", "g2", "g3"]},
    )
    with pytest.raises(ValueError, match=r"of 'A' record the cells .* those of 'bulk'"):
        CubeCollection({"A": study_a(), "bulk": bulk}).stack()


def _with_detected(cube: Cube) -> Cube:
    counts = cube.layer("counts")
    return Cube(
        {"counts": counts, "detected": np.minimum(counts, 1)},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )


def test_selections_are_checked() -> None:
    collection = CubeCollection({"A": study_a(), "B": study_b()})
    assert collection.select() is collection
    with pytest.raises(KeyError, match="no study named 'C'"):
        collection.select(studies=["C"])
    with pytest.raises(ValueError, match="the selection of studies is empty"):
        collection.select(studies=[])
    with pytest.raises(
        ValueError, match=r"no study selected measured the feature 'g1'"
    ):
        collection.select(studies="B", features=["g1"])
    with pytest.raises(ValueError, match=r"no study selected has the group 'CL:3'"):
        collection.select(studies="A", groups="CL:3")
    with pytest.raises(ValueError, match="repeats 'g2'"):
        collection.select(features=["g2", "g2"])
    assert collection.select(studies=[False, True]).studies == ("B",)
    with pytest.raises(KeyError, match=r"no study named 'b'; studies are: 'A' and 'B'"):
        collection.aligned_cube("b")
    with pytest.raises(TypeError, match=r"study must be a study name \(str\)"):
        collection.aligned_cube(0)  # type: ignore[arg-type]


def test_lining_up_results_refuses_results_of_other_cubes() -> None:
    collection = fixture_collection()
    results = fixture_results()
    group = collection.groups["group_id"][0]
    with pytest.raises(TypeError, match="collection must be a CubeCollection"):
        StudyResults(fixture_cubes(), results).table(group, "stim")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"missing 'atlas'"):
        StudyResults(collection, {s: results[s] for s in ("lupus", "covid")}).table(
            group, "stim"
        )
    with pytest.raises(KeyError, match=r"no group 'T cell' in the collection"):
        StudyResults(collection, results).table("T cell", "stim")
    with pytest.raises(KeyError, match="no study tested the contrast 'ctrl'"):
        StudyResults(collection, results).table(group, "ctrl")
    # The covid study's own cube has its own group ids, which the collection merges.
    own = fixture_cubes()["covid"]
    design = Design(own.samples, FORMULAS["covid"], levels=LEVELS)
    unaligned = results | {
        "covid": limma_voom(own, design, contrasts={"stim": "conditionstim"})
    }
    with pytest.raises(ValueError, match=r"analyses the groups 'B', 'T4 naive'"):
        StudyResults(collection, unaligned).table(group, "stim")
    # Its unmeasured genes are filled with zeros, which a test without the
    # expression filter tests.
    alone = CubeCollection({"covid": own}, align_groups_by="group_id")
    tested_all = limma_voom(
        own, design, contrasts={"stim": "conditionstim"}, expression_filter=False
    )
    with pytest.raises(ValueError, match="which the study did not measure"):
        StudyResults(alone, {"covid": tested_all}).table("B", "stim")
