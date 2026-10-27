"""Tests for gene-set collections and GMT files.

Reference values in ``tests/data/gene_set_files_and_ora`` come from R, which read the
GMT files there by hand (fields split at tabs, as the GMT format defines them), and
from limma's ``ids2indices`` and ``camera``; ``validation/gene_set_files_and_ora.py``
writes them. The other tests check what a wrong reader or writer would break: files
written out exactly as derived by hand and read back unchanged, the quirks of real
files, the order of sets and genes, the record of the source, and the messages for
what a collection or a GMT file cannot hold.
"""

import gzip
import hashlib
import json
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import (
    GeneSets,
    camera,
    fry,
    mroast,
    read_gmt,
    roast,
    romer,
    write_gmt,
)

DATA = Path(__file__).parent / "data" / "gene_set_files_and_ora"
FILES = ("msigdb.gmt", "reactome.gmt", "wikipathways.gmt", "gene_ontology.gmt.gz")
EXACT = 1e-10


@cache
def by_hand() -> dict[str, Any]:
    readings: dict[str, Any] = json.loads((DATA / "gmt_by_hand.json").read_text())
    return readings


@cache
def matching() -> dict[str, np.ndarray]:
    with np.load(DATA / "matching.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def assert_same_reading(sets: GeneSets, reading: dict[str, Any]) -> None:
    assert list(sets) == reading["names"]
    assert [sets.descriptions[name] for name in sets] == reading["descriptions"]
    assert [list(sets[name]) for name in sets] == reading["genes"]


# Files against R's reading by hand.


@pytest.mark.parametrize("name", FILES)
def test_files_are_read_as_r_reads_them_by_hand(name: str) -> None:
    sets = read_gmt(DATA / name, name="simulated", version="1", licence="CC0")

    assert_same_reading(sets, by_hand()[name])
    assert (sets.name, sets.version, sets.licence) == ("simulated", "1", "CC0")
    # The record names the file and the digest of its bytes as stored.
    assert sets.file == name
    assert sets.sha256 == hashlib.sha256((DATA / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("name", FILES)
def test_files_written_read_back_as_r_reads_them(name: str, tmp_path: Path) -> None:
    sets = read_gmt(DATA / name)
    write_gmt(sets, tmp_path / name)
    again = read_gmt(tmp_path / name)

    # R read the files that write_gmt wrote of the same sets.
    assert_same_reading(again, by_hand()[f"written_{name}"])
    assert_same_reading(again, by_hand()[name])
    # A file in the form write_gmt writes, one set per line ended by a line feed, with
    # its genes once each, comes back byte for byte.
    if name == "msigdb.gmt":
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()
    else:
        assert again.sha256 != sets.sha256


def test_a_file_is_written_as_the_format_defines_it(tmp_path: Path) -> None:
    sets = GeneSets(
        {"first set": ["G1", "G2"], "SECOND": ("G3",), "empty": []},
        descriptions={"SECOND": "https://example.org/SECOND"},
    )
    write_gmt(sets, tmp_path / "sets.gmt")

    expected = (
        "first set\tna\tG1\tG2\nSECOND\thttps://example.org/SECOND\tG3\nempty\tna\n"
    )
    assert (tmp_path / "sets.gmt").read_bytes() == expected.encode()
    # A name ending in .gz is compressed, without a time stamp.
    write_gmt(sets, tmp_path / "sets.gmt.gz")
    data = (tmp_path / "sets.gmt.gz").read_bytes()
    assert data[:2] == b"\x1f\x8b"
    assert gzip.decompress(data) == expected.encode()
    write_gmt(sets, tmp_path / "again.gmt.gz")
    assert (tmp_path / "again.gmt.gz").read_bytes() == data


def test_the_quirks_of_real_files_are_read(tmp_path: Path) -> None:
    text = (
        "﻿A set\t na \t G1\tG2\t\tG1\t\r"  # a byte-order mark, spaces, "na"
        "\r\n"  # a blank line, and a carriage return alone ending the line before
        "B\t\tG3\r\n"  # no description
        "C\tthe description\n"  # a set without genes
        "\n   \n"
        "D\tdescribed\tG4"  # no line feed at the end
    )
    path = tmp_path / "sets.txt"
    path.write_bytes(gzip.compress(text.encode()))  # gzip, whatever the name
    sets = read_gmt(path)

    assert dict(sets) == {
        "A set": ("G1", "G2"),
        "B": ("G3",),
        "C": (),
        "D": ("G4",),
    }
    assert dict(sets.descriptions) == {
        "A set": None,
        "B": None,
        "C": "the description",
        "D": "described",
    }


def test_files_that_break_the_format_are_refused(tmp_path: Path) -> None:
    def refused(text: bytes, match: str) -> None:
        path = tmp_path / "bad.gmt"
        path.write_bytes(text)
        with pytest.raises(ValueError, match=match):
            read_gmt(path)

    refused(b"A\tna\tG1\nB G2 G3\n", r"bad\.gmt, line 2 has no tab")
    refused(b"A\tna\tG1\n\tna\tG2\n", "line 2: the gene set has no name")
    refused(
        b"A\tna\tG1\nB\tna\nA\tx\tG2\n", r"line 3: the gene set 'A' is also on line 1"
    )
    refused(b"A\tna\tG\xe91\n", "is not UTF-8 text: line 1 holds the byte 0xe9")
    refused(b"\n \n", "holds no gene set")
    refused(b"\x1f\x8b\x08\x00broken", "is not a valid gzip file")
    damaged = bytearray(gzip.compress(b"A\tna\tG1\tG2\tG3\n" * 50))
    damaged[12:16] = b"\xff\xff\xff\xff"
    refused(bytes(damaged), "is not a valid gzip file")
    with pytest.raises(FileNotFoundError):
        read_gmt(tmp_path / "missing.gmt")
    with pytest.raises(TypeError, match="path must be a str or os.PathLike"):
        read_gmt(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="licence must be a string or None"):
        read_gmt(DATA / "msigdb.gmt", licence=4)  # type: ignore[arg-type]


def test_what_a_file_cannot_hold_is_not_written(tmp_path: Path) -> None:
    path = tmp_path / "sets.gmt"
    for sets in (
        {"A\tB": ["G1"]},
        {"A": ["G1\nG2"]},
        {"A": [" G1"]},
    ):
        with pytest.raises(ValueError, match="gene set"):
            write_gmt(sets, path)
    with pytest.raises(ValueError, match="with a tab or a line break"):
        write_gmt(GeneSets({"A": ["G1"]}, descriptions={"A": "one\ttwo"}), path)
    assert not path.exists()
    write_gmt({"A": ["G1"]}, path)
    with pytest.raises(FileExistsError):
        write_gmt({"A": ["G2"]}, path)
    write_gmt({"A": ["G2"]}, path, overwrite=True)
    assert dict(read_gmt(path)) == {"A": ("G2",)}
    with pytest.raises(TypeError, match="overwrite must be True or False"):
        write_gmt({"A": ["G2"]}, path, overwrite=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sets must be a mapping"):
        write_gmt(["G1"], path)  # type: ignore[arg-type]


# The collection.


def test_genes_keep_their_first_place_and_each_counts_once() -> None:
    sets = GeneSets(
        {
            "listed twice": ["G3", "G1", "G3", "G2", "G1"],
            "array": np.array(["B", "A"]),
            "python set": {"Z", "A", "M"},
        }
    )

    assert sets["listed twice"] == ("G3", "G1", "G2")
    assert sets["array"] == ("B", "A")
    # A Python set has no order; its genes are sorted, so every run lists them alike.
    assert sets["python set"] == ("A", "M", "Z")
    assert list(sets) == ["listed twice", "array", "python set"]
    assert len(sets) == 3 and "array" in sets and "other" not in sets
    assert sets.get("other") is None and [] not in sets


def test_the_record_and_descriptions_are_kept_by_every_new_collection() -> None:
    digest = "0" * 64
    sets = GeneSets(
        {"A": ["G1", "G2", "G3"], "B": ["G2"], "C": ["G9"]},
        descriptions={"A": "first", "B": "na", "C": ""},
        name="a curated list",
        version="2",
        licence="CC BY 4.0",
        file="sets.gmt",
        sha256=digest,
    )
    record = ("a curated list", "2", "CC BY 4.0", "sets.gmt", digest)

    assert dict(sets.descriptions) == {"A": "first", "B": None, "C": None}
    for derived in (
        sets.select(["C", "A"]),
        sets.restrict_to(["G2", "G9"]),
        sets.filter_by_size(1, 1),
        pickle.loads(pickle.dumps(sets)),
    ):
        assert (
            derived.name,
            derived.version,
            derived.licence,
            derived.file,
            derived.sha256,
        ) == record
        assert all(derived.descriptions[n] == sets.descriptions[n] for n in derived)
    assert list(sets.select(["C", "A"])) == ["C", "A"]
    assert dict(pickle.loads(pickle.dumps(sets))) == dict(sets)
    with pytest.raises(AttributeError, match="GeneSets is immutable"):
        sets.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        sets.descriptions["A"] = "x"  # type: ignore[index]


def test_restrict_to_keeps_the_genes_ids2indices_finds() -> None:
    fixture = matching()
    identifiers = fixture["identifiers"].tolist()
    sets = read_gmt(DATA / "matching.gmt")
    restricted = sets.restrict_to(identifiers)
    position = {gene: i + 1 for i, gene in enumerate(identifiers)}

    assert list(restricted) == fixture["limma index names"].tolist()
    assert_array_equal(
        [len(genes) for genes in restricted.values()], fixture["limma index sizes"]
    )
    assert_array_equal(
        np.concatenate(
            [sorted(position[g] for g in genes) for genes in restricted.values()]
        ),
        fixture["limma index"],
    )
    # Each set keeps its own order.
    for name, genes in restricted.items():
        assert list(genes) == [g for g in sets[name] if g in position]


def test_a_collection_tested_by_camera_gives_limmas_p_values() -> None:
    fixture = matching()
    identifiers = fixture["identifiers"].tolist()
    kept = read_gmt(DATA / "matching.gmt").restrict_to(identifiers)
    kept = kept.filter_by_size(15, 200)
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    tested = camera(
        fixture["y"].T, kept, design=design, contrast=1, feature_ids=identifiers
    )

    assert list(kept) == fixture["limma camera names"].tolist()
    assert_allclose(tested["p_value"], fixture["limma camera"], rtol=EXACT)
    assert min(len(genes) for genes in kept.values()) >= 15
    assert max(len(genes) for genes in kept.values()) <= 200


def test_every_gene_set_test_takes_a_collection_as_a_mapping_of_ids() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=(8, 60))
    y[4:, :10] += 1.0
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    ids = [f"g{i}" for i in range(60)]
    sets = GeneSets({"up": [*ids[:10], "absent"], "other": ids[20:35]})
    plain = {name: list(genes) for name, genes in sets.items()}
    options = {"feature_ids": ids, "rng": 7, "n_rotations": 49, "n_threads": 1}

    assert camera(y, sets, design=design, contrast=1, feature_ids=ids).equals(
        camera(y, plain, design=design, contrast=1, feature_ids=ids)
    )
    assert fry(y, sets, design=design, contrast=1, feature_ids=ids).equals(
        fry(y, plain, design=design, contrast=1, feature_ids=ids)
    )
    assert mroast(y, sets, design=design, contrast=1, **options).equals(
        mroast(y, plain, design=design, contrast=1, **options)
    )
    assert romer(y, sets, design=design, contrast=1, **options).equals(
        romer(y, plain, design=design, contrast=1, **options)
    )
    assert roast(y, sets["up"], design=design, contrast=1, **options).equals(
        roast(y, plain["up"], design=design, contrast=1, **options)
    )
    assert camera(y, sets, design=design, contrast=1, feature_ids=ids)[
        "n_genes"
    ].tolist() == [10, 15]


def test_restrict_to_drops_empty_sets_ignores_missing_ids_and_needs_a_match() -> None:
    sets = GeneSets({"A": ["G1", "G2", "G3"], "B": ["G4"], "C": ["G5", "G1"]})

    assert dict(sets.restrict_to(["G1", None, "", np.nan, "G5"])) == {
        "A": ("G1",),
        "C": ("G5", "G1"),
    }
    assert dict(sets.restrict_to(["G1"], keep_empty=True)) == {
        "A": ("G1",),
        "B": (),
        "C": ("G1",),
    }
    with pytest.raises(ValueError, match="no gene of any set is among genes.*'G1'"):
        sets.restrict_to(["ENSG00000141510"])
    with pytest.raises(TypeError, match="genes must hold gene ids as strings"):
        sets.restrict_to([7157])
    with pytest.raises(TypeError, match="genes is a single string"):
        sets.restrict_to("G1")
    with pytest.raises(TypeError, match="keep_empty must be True or False"):
        sets.restrict_to(["G1"], keep_empty="yes")  # type: ignore[arg-type]


def test_filter_by_size_keeps_the_sizes_in_its_range() -> None:
    sets = GeneSets({f"S{m}": [f"G{i}" for i in range(m)] for m in (0, 1, 5, 15, 16)})

    assert list(sets.filter_by_size(5)) == ["S5", "S15", "S16"]
    assert list(sets.filter_by_size(1, 15)) == ["S1", "S5", "S15"]
    assert list(sets.filter_by_size(0, 0)) == ["S0"]
    assert list(sets.filter_by_size(17)) == []
    with pytest.raises(ValueError, match="max_size \\(4\\) is smaller than min_size"):
        sets.filter_by_size(5, 4)
    with pytest.raises(ValueError, match="min_size must be at least 0"):
        sets.filter_by_size(-1)
    with pytest.raises(TypeError, match="min_size must be an integer"):
        sets.filter_by_size(True)
    with pytest.raises(TypeError, match="max_size must be an integer"):
        sets.filter_by_size(1, 2.5)  # type: ignore[arg-type]


def test_names_that_are_not_sets_are_refused_with_close_names() -> None:
    sets = GeneSets({"HALLMARK_HYPOXIA": ["G1"], "HALLMARK_APOPTOSIS": ["G2"]})

    with pytest.raises(KeyError, match="did you mean 'HALLMARK_HYPOXIA'"):
        sets["HALLMARK_HYPOXIAA"]
    with pytest.raises(KeyError, match="did you mean 'HALLMARK_APOPTOSIS'"):
        sets.select(["HALLMARK_APOPTOSI"])
    with pytest.raises(ValueError, match="selected more than once"):
        sets.select(["HALLMARK_HYPOXIA", "HALLMARK_HYPOXIA"])
    with pytest.raises(TypeError, match="set names are strings"):
        sets.select([1])
    assert list(sets.select("HALLMARK_APOPTOSIS")) == ["HALLMARK_APOPTOSIS"]


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ((["G1"],), TypeError, "sets must be a mapping of set name to genes"),
        (({1: ["G1"]},), TypeError, "gene set names must be strings"),
        (({"": ["G1"]},), ValueError, "gene set names must not be empty"),
        (({"A": "G1"},), TypeError, "gene set 'A' is a single string"),
        (({"A": {"G1": 1.0}},), TypeError, "gene set 'A' maps genes to weights"),
        (({"A": ["G1", 2]},), TypeError, "gene set 'A' holds 2; genes are named by id"),
        (({"A": ["G1", None]},), TypeError, "gene set 'A' holds None"),
        (({"A": ["G1", ""]},), ValueError, "gene set 'A' holds an empty gene id"),
        (({"A": [["G1"]]},), TypeError, "gene set 'A' holds \\['G1'\\]"),
        (({"A": np.array([["G1"]])},), ValueError, "list of gene ids \\(1-D\\)"),
        (({"A": np.array([7157])},), TypeError, "got dtype int64; convert numeric"),
    ],
)
def test_sets_that_are_not_lists_of_ids_are_refused(
    arguments: tuple[Any, ...], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        GeneSets(*arguments)


def test_descriptions_and_the_record_are_checked() -> None:
    with pytest.raises(ValueError, match="descriptions names the set 'B'"):
        GeneSets({"A": ["G1"]}, descriptions={"B": "x"})
    with pytest.raises(TypeError, match="the description of 'A' must be a string"):
        GeneSets({"A": ["G1"]}, descriptions={"A": 1})
    with pytest.raises(TypeError, match="descriptions must be a mapping"):
        GeneSets({"A": ["G1"]}, descriptions=["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sha256 must be a SHA-256 digest"):
        GeneSets({"A": ["G1"]}, sha256="ABC")
    with pytest.raises(TypeError, match="name must be a string or None"):
        GeneSets({"A": ["G1"]}, name=1)  # type: ignore[arg-type]


def test_the_repr_says_what_the_collection_holds_and_where_it_comes_from() -> None:
    sets = read_gmt(DATA / "msigdb.gmt", name="simulated MSigDB")
    text = repr(sets)
    sizes = [len(genes) for genes in sets.values()]

    assert text.startswith(
        f"GeneSets: 6 sets of {min(sizes)} to {max(sizes)} genes, "
        f"{len({g for genes in sets.values() for g in genes}):,} genes in all"
    )
    assert "name: simulated MSigDB; version: not stated; licence: not stated" in text
    assert f"read from msigdb.gmt, SHA-256 {sets.sha256}" in " ".join(text.split())
    assert max(map(len, text.splitlines())) <= 80
    assert "and 3 more" in text
    assert repr(GeneSets({"A": ["G1"]})).startswith("GeneSets: 1 set of 1 gene,")
