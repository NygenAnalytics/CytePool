"""Tests of networks: the Network class and the loaders of CollecTRI and PROGENy.

The loaders read files in the formats of the published ones, their data simulated,
which validation/activity.py wrote to tests/data/activity and which are pinned here in
place of the real files: rows in the format of CollecTRI's file, against decoupler's
reading of them, and a package in the layout of progeny's with R data files that R
wrote, against the values R wrote and the genes dplyr's ``top_n`` keeps. Downloads,
the cache, the refusal of a file that is not the one pinned, and offline use are
checked with a stand-in for the connection: the tests never connect.
"""

import csv
import doctest
import hashlib
import io
import json
import pickle
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cytepool import networks
from cytepool.gene_sets import GeneSets
from cytepool.networks import Network, collectri, progeny
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "activity"
ROWS = (DATA / "collectri_rows.csv").read_bytes()
PACKAGE = (DATA / "progeny_package.tar.gz").read_bytes()
DECOUPLER = json.loads((DATA / "decoupler.json").read_text())
COLUMNS = ("source", "target", "weight", "resources", "references", "sign_decision")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty cache, the pieces pinned in place of the real files, and no
    connection."""
    for key, data in (("collectri", ROWS), ("progeny", PACKAGE)):
        pinned = networks._DOWNLOADS[key]._replace(size=len(data), sha256=_digest(data))
        monkeypatch.setitem(networks._DOWNLOADS, key, pinned)

    def no_connection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test connected to the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_connection)
    return tmp_path / "cache"


def _place(cache: Path, key: str, data: bytes) -> Path:
    path = cache / networks._DOWNLOADS[key].file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _rows(network: Network) -> set[tuple[Any, ...]]:
    edges = network.edges
    return set(zip(*(edges[name].tolist() for name in COLUMNS), strict=True))


def _file_rows() -> list[list[str]]:
    return list(csv.reader(io.StringIO(ROWS.decode())))[1:]


# CollecTRI.


def test_collectri_is_read_as_decoupler_reads_it(cache: Path) -> None:
    _place(cache, "collectri", ROWS)
    academic = collectri(licence="academic", cache_dir=cache)

    theirs = {tuple(row) for row in DECOUPLER["collectri"]}
    # decoupler drops the interactions without a reference; CytePool keeps them.
    without = {row for row in _rows(academic) if row[4] == ""}
    assert len(without) == sum(1 for row in _file_rows() if row[4] == "") > 0
    assert _rows(academic) - without == theirs
    assert len(academic.edges) == len(_file_rows())
    # The order of the file.
    assert academic.edges["source"].tolist() == [row[0] for row in _file_rows()]


def test_the_commercial_level_leaves_out_what_geredb_alone_supports(
    cache: Path,
) -> None:
    _place(cache, "collectri", ROWS)
    commercial = collectri(cache_dir=cache)
    academic = collectri(licence="academic", cache_dir=cache)

    alone = [row for row in _file_rows() if row[3] == "CollecTRI;GEREDB_CollecTRI"]
    assert len(alone) == 8
    left_out = _rows(academic) - _rows(commercial)
    assert {(row[0], row[1]) for row in left_out} == {(r[0], r[1]) for r in alone}
    assert "level 'commercial'" in commercial.licence
    assert "level 'academic'" in academic.licence
    assert commercial.name == "CollecTRI"
    assert commercial.file == "CollecTRI_regulons.csv"
    assert commercial.sha256 == _digest(ROWS)
    assert "Mueller-Dott" in commercial.citation
    with pytest.raises(ValueError, match="licence must be one of 'commercial'"):
        collectri(licence="nonprofit", cache_dir=cache)


def test_a_file_that_is_not_the_one_pinned_is_refused(cache: Path) -> None:
    changed = bytearray(ROWS)
    changed[100] ^= 1  # the same size, another digest
    path = _place(cache, "collectri", bytes(changed))
    with pytest.raises(ValueError, match="is not the file that CytePool pins"):
        collectri(cache_dir=cache)
    path.write_bytes(ROWS[:-1])
    with pytest.raises(ValueError, match=f"{len(ROWS) - 1:,} bytes"):
        collectri(cache_dir=cache)


class _Answer(io.BytesIO):
    """What urlopen returns: the bytes of a download, read in blocks."""

    def __enter__(self) -> "_Answer":
        return self


def test_a_download_is_kept_only_when_it_is_the_file_pinned(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    def serve(data: bytes) -> Any:
        def urlopen(request: urllib.request.Request, timeout: float) -> _Answer:
            requests.append(request)
            return _Answer(data)

        return urlopen

    monkeypatch.setattr(urllib.request, "urlopen", serve(ROWS[:-10]))
    with pytest.raises(ValueError, match="the download of .* is not the file"):
        collectri(cache_dir=cache)
    assert list(cache.iterdir()) == []  # nothing is kept, not even in part

    monkeypatch.setattr(urllib.request, "urlopen", serve(ROWS))
    first = collectri(cache_dir=cache)
    assert (cache / "CollecTRI_regulons.csv").read_bytes() == ROWS
    assert requests[-1].full_url == networks._DOWNLOADS["collectri"].url
    # The request names its client, which some servers require.
    assert requests[-1].header_items()[0][1].startswith("CytePool/")
    # Once in the cache, the file is read without a connection.
    monkeypatch.setattr(urllib.request, "urlopen", None)
    assert collectri(cache_dir=cache).edges.equals(first.edges)


def test_without_a_connection_the_message_says_where_to_put_the_file(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(request: Any, timeout: float) -> None:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(OSError, match=r"could not download CollecTRI_regulons\.csv"):
        collectri(cache_dir=cache)
    assert list(cache.iterdir()) == []


def test_the_cache_is_under_xdg_cache_home(
    cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home"))
    _place(tmp_path / "home" / "cytepool" / "networks", "collectri", ROWS)
    assert len(collectri(licence="academic").edges) == len(_file_rows())


# PROGENy.


def _r_table(organism: str) -> dict[tuple[str, str], tuple[float, float]]:
    """What R wrote of the model: (pathway, gene) to (weight, p-value)."""
    text = (DATA / f"progeny_model_{organism}_full.csv").read_text()
    return {
        (row["pathway"], row["gene"]): (float(row["weight"]), float(row["p.value"]))
        for row in csv.DictReader(io.StringIO(text))
    }


@pytest.mark.parametrize("organism", ["human", "mouse"])
def test_progenys_r_data_are_read_as_r_wrote_them(cache: Path, organism: str) -> None:
    # The human genes are a factor, the mouse genes text: both of R's codings.
    _place(cache, "progeny", PACKAGE)
    found = progeny(organism=organism, max_p_value=None, cache_dir=cache)
    edges = found.edges
    ours = {
        (s, t): (w, p)
        for s, t, w, p in zip(
            edges["source"].tolist(),
            edges["target"].tolist(),
            edges["weight"].tolist(),
            edges["p_value"].tolist(),
            strict=True,
        )
    }
    assert ours == _r_table(organism)
    # The pathways in alphabetical order, the genes of each by p-value.
    sources = edges["source"].tolist()
    assert sources == sorted(sources)
    for pathway in set(sources):
        p = edges["p_value"][edges["source"] == pathway]
        assert np.all(np.diff(p) >= 0.0)
    assert found.licence == "Apache License 2.0 (the R package progeny)"
    assert found.name == f"PROGENy ({organism})"


def test_progeny_keeps_the_top_genes_of_each_pathway_as_dplyr_does(cache: Path) -> None:
    _place(cache, "progeny", PACKAGE)
    top = progeny(top=3, max_p_value=None, cache_dir=cache)
    text = (DATA / "progeny_top3.csv").read_text()
    expected = {
        (row["pathway"], row["gene"]) for row in csv.DictReader(io.StringIO(text))
    }
    kept = set(zip(top.edges["source"].tolist(), top.edges["target"].tolist(),
                   strict=True))  # fmt: skip
    assert kept == expected
    # One pathway has a tie at its third gene, which stays.
    assert len(expected) == 3 * len(top) + 1


def test_progeny_keeps_the_genes_below_a_p_value(cache: Path) -> None:
    _place(cache, "progeny", PACKAGE)
    found = progeny(max_p_value=0.2, cache_dir=cache)
    expected = {key for key, (_, p) in _r_table("human").items() if p < 0.2}
    assert set(zip(found.edges["source"].tolist(), found.edges["target"].tolist(),
                   strict=True)) == expected  # fmt: skip
    default = progeny(cache_dir=cache)
    assert all(p < 0.05 for p in default.edges["p_value"].tolist())
    with pytest.raises(ValueError, match=r"max_p_value must be in \(0, 1\]"):
        progeny(max_p_value=0.0, cache_dir=cache)
    with pytest.raises(ValueError, match="top must be at least 1"):
        progeny(top=0, cache_dir=cache)
    with pytest.raises(ValueError, match="organism must be one of 'human', 'mouse'"):
        progeny(organism="rat", cache_dir=cache)


def test_r_data_that_cannot_be_read_are_refused() -> None:
    with pytest.raises(ValueError, match="not an R data file in the XDR format"):
        networks._read_r_data(b"not R")
    with pytest.raises(ValueError, match="end in the middle"):
        networks._read_r_data(b"RDX3\nX\n\x00\x00\x00\x03\x00\x04")


# The Network class.

EDGES = {
    "STAT1": {"IRF1": 1.0, "SOCS1": 1.0, "MYC": -1.0},
    "IRF1": {"CXCL10": 2.0, "MYC": -0.5},
}


def test_the_forms_of_a_network() -> None:
    network = Network(EDGES, name="mine", version="1", licence="CC0")
    assert network.sources == ("STAT1", "IRF1")
    assert dict(network["IRF1"]) == {"CXCL10": 2.0, "MYC": -0.5}
    assert network.edges.columns == ("source", "target", "weight")
    table = Table(
        {
            "source": ["STAT1", "STAT1", "STAT1", "IRF1", "IRF1"],
            "target": ["IRF1", "SOCS1", "MYC", "CXCL10", "MYC"],
            "weight": [1.0, 1.0, -1.0, 2.0, -0.5],
            "p_value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    from_table = Network(table)
    assert from_table.edges.select(["source", "target", "weight"]).equals(network.edges)
    assert_array_equal(from_table.edges["p_value"], [0.1, 0.2, 0.3, 0.4, 0.5])
    # A mapping of columns is a table; pandas DataFrames too.
    assert Network(table.to_dict()).edges.equals(from_table.edges)
    pandas = pytest.importorskip("pandas")
    assert Network(pandas.DataFrame(table.to_dict())).edges.equals(from_table.edges)
    # Lists of targets, and gene sets, weigh 1.
    sets = Network(GeneSets({"up": ["A", "B"], "down": ["C"]}))
    assert_array_equal(sets.edges["weight"], [1.0, 1.0, 1.0])
    assert (
        Network(table.select(["source", "target"])).edges["weight"].tolist()
        == [1.0] * 5
    )
    # The mapping.
    assert "STAT1" in network and "MYC" not in network and [] not in network
    assert len(network) == 2 and list(network) == ["STAT1", "IRF1"]
    with pytest.raises(KeyError, match="no source named 'STAT2' .did you mean 'STAT1'"):
        network["STAT2"]


def test_a_network_selects_records_and_pickles() -> None:
    network = Network(EDGES, name="mine", licence="CC0")
    chosen = network.select(["IRF1", "STAT1"])
    assert chosen.sources == ("IRF1", "STAT1")
    assert chosen.name == "mine" and chosen.licence == "CC0"
    assert network.select("IRF1").edges["target"].tolist() == ["CXCL10", "MYC"]
    copy = pickle.loads(pickle.dumps(network))
    assert copy.edges.equals(network.edges) and copy.name == "mine"
    assert repr(network).startswith("Network: 2 sources, 5 edges, 4 target genes")
    with pytest.raises(AttributeError, match="Network is immutable"):
        network.name = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="'IRF1' is selected more than once"):
        network.select(["IRF1", "IRF1"])
    with pytest.raises(KeyError, match="no source named 'X'"):
        network.select(["X"])


def _as_written_in_the_guides(network: Network) -> GeneSets:
    return GeneSets(
        {source: list(targets) for source, targets in network.items()},
        source=network.name,
        version=network.version,
        licence=network.licence,
        file=network.file,
        sha256=network.sha256,
    )


def _same_sets(ours: GeneSets, expected: GeneSets) -> None:
    assert list(ours) == list(expected)
    for name in expected:
        assert ours[name] == expected[name]
    for attribute in ("source", "version", "licence", "file", "sha256"):
        assert getattr(ours, attribute) == getattr(expected, attribute), attribute


def test_a_network_as_gene_sets_drops_the_weights_and_keeps_the_record(
    cache: Path,
) -> None:
    network = Network(EDGES, name="mine", version="1", licence="CC0")
    sets = network.to_gene_sets()
    # Each source's targets in the order of its edges, repressed ones included.
    assert dict(sets.items()) == {
        "STAT1": ("IRF1", "SOCS1", "MYC"),
        "IRF1": ("CXCL10", "MYC"),
    }
    _same_sets(sets, _as_written_in_the_guides(network))
    assert (sets.source, sets.version, sets.licence) == ("mine", "1", "CC0")
    # Back to a network, every target weighs 1.
    again = Network(sets)
    assert again.edges.select(["source", "target"]).equals(
        network.edges.select(["source", "target"])
    )
    assert again.edges["weight"].tolist() == [1.0] * 5
    # A loaded network keeps its file and digest.
    _place(cache, "collectri", ROWS)
    academic = collectri(licence="academic", cache_dir=cache)
    regulons = academic.to_gene_sets()
    _same_sets(regulons, _as_written_in_the_guides(academic))
    assert regulons.sha256 == academic.sha256 is not None
    assert len(regulons) == len(academic.sources)


def test_network_messages() -> None:
    with pytest.raises(ValueError, match="from 'A' to 'B' is given twice"):
        Network({"source": ["A", "A"], "target": ["B", "B"], "weight": [1.0, 2.0]})
    with pytest.raises(ValueError, match="the target of row 1 is missing"):
        Network({"source": ["A", "A"], "target": ["B", None]})
    with pytest.raises(ValueError, match="edge 0 has the weight inf"):
        Network({"A": {"B": np.inf}})
    with pytest.raises(ValueError, match="the network has no edges"):
        Network({"A": []})
    with pytest.raises(ValueError, match="have no column 'target'"):
        Network(Table({"source": ["A"], "gene": ["B"]}))
    with pytest.raises(TypeError, match="edges must be a Table or pandas DataFrame"):
        Network(3)
    with pytest.raises(TypeError, match="must be gene ids"):
        Network({"A": {1: 1.0}})
    with pytest.raises(ValueError, match="sha256 must be a SHA-256 digest"):
        Network(EDGES, sha256="abc")


def test_the_examples_run() -> None:
    assert doctest.testmod(networks, raise_on_error=False).failed == 0
