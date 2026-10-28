"""Networks of regulators and pathways and the genes they act on, with signed weights:
CollecTRI's transcription factors and PROGENy's pathways, downloaded from their
releases, and networks given as tables.

A :class:`~cytepool.Network` holds the target genes of its sources, transcription
factors or pathways, with signed weights, as the activity functions
(:func:`~cytepool.ulm`, :func:`~cytepool.mlm` and :func:`~cytepool.weighted_aggregate`)
take it, and as decoupler takes its networks (Badia-i-Mompel et al. 2022): a table with
the columns ``source``, ``target`` and ``weight``. A network records its name, version,
where it comes from, its licence and its citation, and, for a network read from a file,
the name of the file and its SHA-256 digest. :func:`~cytepool.collectri` and
:func:`~cytepool.progeny` return the two networks that CytePool downloads.
https://docs.nygen.io/CytePool/statistics/activities#regulator-networks explains the
networks, https://docs.nygen.io/CytePool/statistics/activities#activities-limits how
they differ from decoupler's, and
https://docs.nygen.io/CytePool/reference/datasets#networks gives their files, licences
and the terms of CollecTRI's resources.

CollecTRI (:func:`~cytepool.collectri`):
    The signed regulons of human transcription factors that Mueller-Dott et al. (2023)
    assembled from twelve resources of literature curation and text mining: version 2.0
    of the file ``CollecTRI_regulons.csv`` of Zenodo record 8192729
    (doi:10.5281/zenodo.8192729), which decoupler 2.2.0 reads too, under the Creative
    Commons Attribution 4.0 licence. Each interaction has a weight of 1 (activation) or
    -1 (repression), the resources that list it, the PubMed ids of its references and
    how its sign was decided. The complexes AP1 and NFKB are sources of their own, as in
    the file. Each resource keeps its own terms, which OmniPath records (Tuerei et al.
    2021): ``licence="commercial"``, the default, keeps the interactions that at least
    one resource listed for commercial use supports, which leaves out those that
    GEREDB, listed for academic use, alone supports; ``licence="academic"`` keeps every
    interaction. Use under any level is subject to the terms of the resources it keeps.

PROGENy (:func:`~cytepool.progeny`):
    The weights of the genes that respond to 14 signalling pathways, from the
    perturbation experiments of Schubert et al. (2018), for human, and their
    transfer to mouse (Holland, Szalai and Saez-Rodriguez 2020): the data frames
    ``model_human_full`` and ``model_mouse_full`` of the R package progeny 1.28.0, from
    its Bioconductor 3.20 release, under the Apache License 2.0, with a weight (a
    z-score) and a p-value for every gene and pathway. By default the genes with a
    p-value below 0.05 are kept, as decoupler's ``op.progeny`` keeps them; ``top``
    keeps the genes of each pathway with the smallest p-values, those tied with the
    last one included, as progeny's ``getModel`` does (100 genes by default there).
    Genes are named by the symbols that progeny 1.28.0 holds.

Downloads and the cache:
    Each network is downloaded on first use from the address it was published at,
    pinned by its size and SHA-256 digest, into the cache directory: ``cache_dir``,
    else ``cytepool/networks`` under ``XDG_CACHE_HOME`` or ``~/.cache``. A file
    whose size or digest differs from the pinned ones is refused, whether
    downloaded or found in the cache, so that a network is always the release its
    version names. With the file in the cache, as when copied there from another
    machine, no connection is made.

R's data files:
    progeny keeps its weights in R's data files (``.rda``), which this module reads
    with a reader of R's serialization format (version 2 or 3, XDR; R Core Team
    2025) limited to what data frames of numbers, text and factors use: vectors,
    pairlists of attributes, symbols and references to them, and R's compact
    sequences of numbers, compressed by xz, bzip2 or gzip.

References:
    Mueller-Dott S, Tsirvouli E, Vazquez M, Ramirez Flores RO, Badia-i-Mompel P,
    Fallegger R, Tuerei D, Laegreid A, Saez-Rodriguez J (2023). Expanding the
    coverage of regulons from high-confidence prior knowledge for accurate estimation
    of transcription factor activities. Nucleic Acids Research 51, 10934-10949.
    doi:10.1093/nar/gkad841

    Schubert M, Klinger B, Kluenemann M, Sieber A, Uhlitz F, Sauer S, Garnett MJ,
    Bluethgen N, Saez-Rodriguez J (2018). Perturbation-response genes reveal
    signaling footprints in cancer gene expression. Nature Communications 9, 20.
    doi:10.1038/s41467-017-02391-6

    Holland CH, Szalai B, Saez-Rodriguez J (2020). Transfer of regulatory knowledge
    from human to mouse for functional genomics analysis. Biochimica et Biophysica
    Acta (BBA) - Gene Regulatory Mechanisms 1863, 194431.
    doi:10.1016/j.bbagrm.2019.194431

    Tuerei D, Valdeolivas A, Gul L, Palacio-Escat N, Klein M, Ivanova O, Oelbei M,
    Gabor A, Theis F, Modos D, Korcsmaros T, Saez-Rodriguez J (2021). Integrated
    intra- and intercellular signaling knowledge for multicellular omics analysis.
    Molecular Systems Biology 17, e9923. doi:10.15252/msb.20209923

    Badia-i-Mompel P, Velez Santiago J, Braunger J, Geiss C, Dimitrov D, Mueller-Dott S,
    Taus P, Dugourd A, Holland CH, Ramirez Flores RO, Saez-Rodriguez J (2022).
    decoupleR: ensemble of computational methods to infer biological activities from
    omics data. Bioinformatics Advances 2, vbac016. doi:10.1093/bioadv/vbac016

    R Core Team (2025). R: A Language and Environment for Statistical Computing. R
    Foundation for Statistical Computing, Vienna. https://www.R-project.org/
"""

import bz2
import csv
import gzip
import hashlib
import io
import lzma
import os
import struct
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from . import __version__
from ._columns import is_data_frame, text_column
from ._gene_ids import is_missing, listed_values
from ._masked_arrays import real_array, reject_masked
from ._messages import KEEP_TOGETHER, close_matches, fitted_lines, listing, quoted
from ._options import checked_choice, positive_integer, real_number
from .gene_sets import GeneSets
from .table import Table

__all__ = ["Network", "collectri", "progeny"]


class _Download(NamedTuple):
    """A file of a network: where it is published, its name in the cache, its size in
    bytes and its SHA-256 digest."""

    url: str
    file: str
    size: int
    sha256: str


_DOWNLOADS = {
    "collectri": _Download(
        "https://zenodo.org/records/8192729/files/CollecTRI_regulons.csv?download=1",
        "CollecTRI_regulons.csv",
        4_345_649,
        "4473c9189dd53dacc80297709ad1452dda1086a1cc2185f9a56146c261668701",
    ),
    "progeny": _Download(
        "https://bioconductor.org/packages/3.20/bioc/src/contrib/progeny_1.28.0.tar.gz",
        "progeny_1.28.0.tar.gz",
        10_642_873,
        "2c41531ef6d088047e2830ccc5741aa0f28f48fb51d4c164cc8331e19b925d7b",
    ),
}
_TIMEOUT = 120.0  # seconds without data before a download gives up
# Some servers, such as Bioconductor's, refuse requests that do not name their client.
_CLIENT = f"CytePool/{__version__} (+https://github.com/NygenAnalytics/CytePool)"
_BLOCK = 1 << 20  # bytes read at a time

_COLLECTRI_COLUMNS = (
    "source",
    "target",
    "weight",
    "resources",
    "references",
    "sign_decision",
)
# CollecTRI's resources that OmniPath's licence records list for academic use only;
# the others it lists for commercial use (https://omnipathdb.org/resources, read on
# 26 October 2026).
_ACADEMIC_ONLY = frozenset({"GEREDB"})
_LICENCES = ("commercial", "academic")
_COLLECTRI_LICENCE = {
    "commercial": (
        "CC BY 4.0 (Zenodo record 8192729); level 'commercial': the interactions of "
        "resources that OmniPath lists for commercial use, without those that GEREDB "
        "(listed for academic use) alone supports; each resource keeps its own terms"
    ),
    "academic": (
        "CC BY 4.0 (Zenodo record 8192729); level 'academic': every interaction, "
        "including those that GEREDB, listed by OmniPath for academic use, alone "
        "supports; each resource keeps its own terms"
    ),
}
_COLLECTRI_CITATION = (
    "Mueller-Dott S et al. (2023). Expanding the coverage of regulons from "
    "high-confidence prior knowledge for accurate estimation of transcription factor "
    "activities. Nucleic Acids Research 51, 10934-10949. doi:10.1093/nar/gkad841"
)
_ORGANISMS = ("human", "mouse")
_PROGENY_CITATION = {
    "human": (
        "Schubert M et al. (2018). Perturbation-response genes reveal signaling "
        "footprints in cancer gene expression. Nature Communications 9, 20. "
        "doi:10.1038/s41467-017-02391-6"
    ),
    "mouse": (
        "Holland CH, Szalai B, Saez-Rodriguez J (2020). Transfer of regulatory "
        "knowledge from human to mouse for functional genomics analysis. Biochimica et "
        "Biophysica Acta (BBA) - Gene Regulatory Mechanisms 1863, 194431. "
        "doi:10.1016/j.bbagrm.2019.194431"
    ),
}
_PROGENY_MAX_P_VALUE = 0.05  # decoupler's thr_padj
_SOURCES_SHOWN = 3  # the sources that a repr names


class Network(Mapping[str, Mapping[str, float]]):
    """Sources, such as transcription factors or pathways, and their target genes,
    with signed weights.

    A read-only mapping of each source to its targets, a mapping of gene id to weight,
    so that ``network["STAT1"]["IRF1"]`` is the weight of the edge from STAT1 to IRF1.
    The edges keep the order given, and the sources the order in which they first
    appear. :func:`collectri` and :func:`progeny` build one from their releases.
    Immutable.

    Args:
        edges: A ``Table``, a pandas DataFrame or a mapping of column name to
            values, with the columns ``source``, ``target`` and, optionally,
            ``weight`` (1 for every edge without it), one row per edge; other
            columns, such as a p-value or the resources of an interaction, are kept
            with the edges. Or a mapping of source to its targets: a mapping of gene
            id to weight, or a list of gene ids, each of weight 1, so that a
            :class:`~cytepool.GeneSets` is a network of unweighted sets.
        name: The network's name, such as ``"CollecTRI"``.
        version: Its version or release.
        url: Where it was obtained.
        licence: The terms under which it is distributed, as its provider states
            them.
        citation: The work to cite for it.
        file: The name of the file it was read from.
        sha256: The SHA-256 digest of that file, 64 hexadecimal digits in lower case.

    Raises:
        TypeError: If ``edges`` is none of these, a source or target is not a
            string, a weight is not a real number, a text option is not a string, or
            an argument is or holds a numpy masked array.
        ValueError: If a table lacks the source or target column, a source or
            target is missing or empty, a weight is not finite, an edge is given
            twice, there are no edges, or ``sha256`` is not a SHA-256 digest.

    Example:
        >>> network = Network({"STAT1": {"IRF1": 1.0, "SOCS1": 1.0, "MYC": -1.0}})
        >>> network["STAT1"]["MYC"]
        -1.0
    """

    __slots__ = (
        "_edges",
        "_sources",
        "_targets",
        "_name",
        "_version",
        "_url",
        "_licence",
        "_citation",
        "_file",
        "_sha256",
    )

    _edges: Table
    _sources: tuple[str, ...]
    _targets: dict[str, dict[str, float]]
    _name: str | None
    _version: str | None
    _url: str | None
    _licence: str | None
    _citation: str | None
    _file: str | None
    _sha256: str | None

    def __init__(
        self,
        edges: Any,
        *,
        name: str | None = None,
        version: str | None = None,
        url: str | None = None,
        licence: str | None = None,
        citation: str | None = None,
        file: str | None = None,
        sha256: str | None = None,
    ) -> None:
        record = {
            "name": _text(name, "name"),
            "version": _text(version, "version"),
            "url": _text(url, "url"),
            "licence": _text(licence, "licence"),
            "citation": _text(citation, "citation"),
            "file": _text(file, "file"),
            "sha256": _digest(sha256),
        }
        _initialize(self, _edge_table(edges), record)

    # Immutability and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Network is immutable; select() returns a new network")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Network is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (self._edges, self._record()))

    # The mapping.

    def __getitem__(self, source: str) -> Mapping[str, float]:
        """Return the targets of ``source``, a mapping of gene id to weight.

        Raises:
            KeyError: If there is no such source; the message suggests close names.
        """
        try:
            return MappingProxyType(self._targets[source])
        except (KeyError, TypeError):
            hint = (
                close_matches(source, list(self._sources))
                if isinstance(source, str)
                else ""
            )
            raise KeyError(f"no source named {source!r}{hint}") from None

    def __contains__(self, source: object) -> bool:
        """Return whether a source has this name."""
        try:
            return source in self._targets
        except TypeError:  # unhashable
            return False

    def __iter__(self) -> Iterator[str]:
        """Iterate over the sources, in order."""
        return iter(self._sources)

    def __len__(self) -> int:
        """Return the number of sources."""
        return len(self._sources)

    # What the network records.

    @property
    def edges(self) -> Table:
        """The edges, one row per edge, with the columns ``source``, ``target`` and
        ``weight`` and any others given."""
        return self._edges

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources, in the order in which they first appear."""
        return self._sources

    @property
    def name(self) -> str | None:
        """The network's name; None when not stated."""
        return self._name

    @property
    def version(self) -> str | None:
        """Its version or release; None when not stated."""
        return self._version

    @property
    def url(self) -> str | None:
        """Where it was obtained; None when not stated."""
        return self._url

    @property
    def licence(self) -> str | None:
        """The terms under which it is distributed; None when not stated."""
        return self._licence

    @property
    def citation(self) -> str | None:
        """The work to cite for it; None when not stated."""
        return self._citation

    @property
    def file(self) -> str | None:
        """The name of the file it was read from; None for a network built in
        Python."""
        return self._file

    @property
    def sha256(self) -> str | None:
        """The SHA-256 digest of that file; None for a network built in Python."""
        return self._sha256

    # New networks.

    def select(self, sources: str | Any) -> "Network":
        """Return the edges of the given sources, in that order of sources.

        Args:
            sources: A source, or a sequence of them.

        Raises:
            KeyError: If a name is not a source's; the message suggests close names.
            TypeError: If a name is not a string, or ``sources`` is or holds a numpy
                masked array.
            ValueError: If a source is given twice, or none is given.
        """
        reject_masked(sources, "sources")
        given = [sources] if isinstance(sources, str) else _names(sources)
        chosen: list[str] = []
        for source in given:
            if not isinstance(source, str):
                raise TypeError(f"sources are named by strings, got {source!r}")
            if source not in self._targets:
                hint = close_matches(source, list(self._sources))
                raise KeyError(f"no source named {source!r}{hint}")
            if source in chosen:
                raise ValueError(f"the source {source!r} is selected more than once")
            chosen.append(str(source))
        if not chosen:
            raise ValueError("select needs at least one source")
        names = self._edges["source"].tolist()
        rows: dict[str, list[int]] = {source: [] for source in chosen}
        for row, source in enumerate(names):
            if source in rows:
                rows[source].append(row)
        taken = [row for source in chosen for row in rows[source]]
        return self._derived(np.array(taken, dtype=np.intp))

    def to_gene_sets(self) -> GeneSets:
        """Return the targets of each source as a gene set, without their weights.

        The gene-set tests and per-sample scores take gene sets, such as the regulons of
        CollecTRI to test with :func:`~cytepool.gsea` or score with
        :func:`~cytepool.gsva`. Each source becomes a set of its targets, in the order
        of the edges, named by the source. The weights, and so the signs that tell
        activated targets from repressed ones, are left out: :func:`~cytepool.ulm` and
        the other activity methods use them. The sets keep the network's name, version,
        licence, file and digest.

        Returns:
            A :class:`~cytepool.GeneSets` with one set per source.
        """
        return GeneSets(
            {source: list(self._targets[source]) for source in self._sources},
            name=self._name,
            version=self._version,
            licence=self._licence,
            file=self._file,
            sha256=self._sha256,
        )

    def __repr__(self) -> str:
        n_edges = len(self._edges)
        n_targets = len(set(self._edges["target"].tolist()))
        lines = [
            f"Network: {len(self._sources):,} source"
            f"{'' if len(self._sources) == 1 else 's'}, {n_edges:,} edge"
            f"{'' if n_edges == 1 else 's'}, {n_targets:,} target gene"
            f"{'' if n_targets == 1 else 's'}"
        ]
        stated = [
            f"{label}: {value if value is not None else 'not stated'}"
            for label, value in (
                ("name", self._name),
                ("version", self._version),
                ("licence", self._licence),
            )
        ]
        lines.append("  " + "; ".join(stated))
        if self._file is not None:
            digest = f", SHA-256 {self._sha256}" if self._sha256 else ""
            lines.append(f"  read from {self._file}{digest}")
        described = [
            f"{source} ({len(self._targets[source]):,})".replace(" ", KEEP_TOGETHER)
            for source in self._sources
        ]
        lines.append("  sources: " + listing(described, limit=_SOURCES_SHOWN))
        return fitted_lines("\n".join(lines))

    def _record(self) -> dict[str, str | None]:
        return {
            "name": self._name,
            "version": self._version,
            "url": self._url,
            "licence": self._licence,
            "citation": self._citation,
            "file": self._file,
            "sha256": self._sha256,
        }

    def _derived(self, rows: np.ndarray) -> "Network":
        """A network of some of these edges, with the same record."""
        derived = object.__new__(Network)
        _initialize(derived, self._edges.take(rows), self._record())
        return derived


def collectri(
    *,
    licence: str = "commercial",
    cache_dir: str | os.PathLike[str] | None = None,
) -> Network:
    """The CollecTRI network of human transcription factors and their target genes.

    The signed regulons of Mueller-Dott et al. (2023), version 2.0 of its Zenodo
    record: 43,175 interactions of 1,186 transcription factors, each with a weight of
    1 or -1, at the licence level asked for. Each of CollecTRI's twelve resources
    keeps its own terms, which OmniPath records;
    https://docs.nygen.io/CytePool/reference/datasets#networks lists them. The file
    is downloaded on first use, checked against its pinned size and SHA-256 digest,
    and kept in the cache.

    Args:
        licence: ``"commercial"``, the interactions that at least one resource that
            OmniPath lists for commercial use supports (39,896, which leaves out the
            3,279 that GEREDB, listed for academic use, alone supports), or
            ``"academic"``, every interaction (43,175).
        cache_dir: The directory that holds the downloaded file; by default
            ``cytepool/networks`` under ``XDG_CACHE_HOME`` or ``~/.cache``.

    Returns:
        A network with the columns ``source``, ``target``, ``weight`` (1 or -1),
        ``resources`` (the resources that list the interaction, separated by
        ``";"``), ``references`` (PubMed ids, separated by ``";"``; empty for the 185
        interactions that the file lists without any, which are kept) and
        ``sign_decision``, in the order of the file.

    Raises:
        ValueError: If ``licence`` is not one of the levels, or the file in the cache,
            or the one downloaded, is not the file pinned (its size or digest
            differs).
        TypeError: If an argument has the wrong type.
        OSError: If the file is not in the cache and cannot be downloaded.

    Example:
        >>> network = collectri()  # doctest: +SKIP
        >>> network["STAT1"]["IRF1"]  # doctest: +SKIP
        1.0
    """
    level = checked_choice(licence, "licence", _LICENCES)
    download = _DOWNLOADS["collectri"]
    path = _fetched(download, cache_dir)
    edges = _collectri_edges(path.read_bytes(), level)
    return Network(
        Table(edges),
        name="CollecTRI",
        version="2.0 (Zenodo record 8192729)",
        url=download.url,
        licence=_COLLECTRI_LICENCE[level],
        citation=_COLLECTRI_CITATION,
        file=download.file,
        sha256=download.sha256,
    )


def progeny(
    *,
    organism: str = "human",
    top: int | None = None,
    max_p_value: float | None = _PROGENY_MAX_P_VALUE,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Network:
    """The PROGENy network of 14 signalling pathways and the genes that respond to
    them.

    The weights of the R package progeny 1.28.0 (Schubert et al. 2018; Holland,
    Szalai and Saez-Rodriguez 2020), from Bioconductor 3.20, under the Apache
    License 2.0, with genes named by the symbols that the package holds: each
    pathway's genes in increasing order of their p-values, those with a p-value
    below ``max_p_value``, as decoupler's ``op.progeny`` keeps them, and, with
    ``top``, only the ``top`` genes with the smallest p-values, those tied with the
    last one included, as progeny's ``getModel`` keeps them. The package is
    downloaded on first use, checked against its pinned size and SHA-256 digest, and
    kept in the cache.

    Args:
        organism: ``"human"`` or ``"mouse"``.
        top: The most genes of each pathway, at least 1; None for no limit.
            progeny's ``getModel`` and ``progeny`` take 100 by default, decoupler's
            tutorials 500.
        max_p_value: Keep the genes with a p-value below it, in (0, 1]; None keeps
            every gene. 0.05 by default, as in decoupler.
        cache_dir: The directory that holds the downloaded file; by default
            ``cytepool/networks`` under ``XDG_CACHE_HOME`` or ``~/.cache``.

    Returns:
        A network with the columns ``source`` (the pathway), ``target`` (the gene),
        ``weight`` and ``p_value``, the pathways in alphabetical order and the genes
        of each in increasing order of their p-values.

    Raises:
        ValueError: If an option is out of range, or the file in the cache, or the one
            downloaded, is not the file pinned (its size or digest differs).
        TypeError: If an argument has the wrong type.
        OSError: If the file is not in the cache and cannot be downloaded.

    Example:
        >>> network = progeny(top=100)  # doctest: +SKIP
        >>> len(network)  # doctest: +SKIP
        14
    """
    kind = checked_choice(organism, "organism", _ORGANISMS)
    reject_masked(top, "top")
    reject_masked(max_p_value, "max_p_value")
    most = None if top is None else positive_integer(top, "top")
    threshold = None
    if max_p_value is not None:
        threshold = real_number(max_p_value, "max_p_value", finite=True)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"max_p_value must be in (0, 1], got {threshold}")
    download = _DOWNLOADS["progeny"]
    path = _fetched(download, cache_dir)
    edges = _progeny_edges(path.read_bytes(), kind, most, threshold)
    return Network(
        Table(edges),
        name=f"PROGENy ({kind})",
        version="progeny 1.28.0 (Bioconductor 3.20)",
        url=download.url,
        licence="Apache License 2.0 (the R package progeny)",
        citation=_PROGENY_CITATION[kind],
        file=download.file,
        sha256=download.sha256,
    )


# Building networks.


def _initialize(network: Network, edges: Table, record: Mapping[str, Any]) -> None:
    targets: dict[str, dict[str, float]] = {}
    weights = edges["weight"].tolist()
    for source, target, weight in zip(
        edges["source"].tolist(), edges["target"].tolist(), weights, strict=True
    ):
        targets.setdefault(source, {})[target] = weight
    object.__setattr__(network, "_edges", edges)
    object.__setattr__(network, "_sources", tuple(targets))
    object.__setattr__(network, "_targets", targets)
    for key in ("name", "version", "url", "licence", "citation", "file", "sha256"):
        object.__setattr__(network, f"_{key}", record[key])


def _rebuild(edges: Table, record: dict[str, str | None]) -> Network:
    network = object.__new__(Network)
    _initialize(network, edges, record)
    return network


def _edge_table(edges: Any) -> Table:
    """The edges given, checked: source, target and weight first."""
    reject_masked(edges, "edges", missing="None or NaN")
    columns = isinstance(edges, Mapping) and "source" in edges and "target" in edges
    if isinstance(edges, Table) or is_data_frame(edges) or columns:
        table = edges if isinstance(edges, Table) else Table(edges)
        missing = [name for name in ("source", "target") if name not in table]
        if missing:
            raise ValueError(
                f"the edges have no column {quoted(missing)}; a network's table has "
                "the columns 'source', 'target' and, optionally, 'weight', one row "
                f"per edge; its columns are {quoted(list(table.columns))}"
            )
        sources = _names_column(table["source"], "source")
        targets = _names_column(table["target"], "target")
        if "weight" in table:
            weights = _weights(table["weight"], "the column 'weight'")
        else:
            weights = np.ones(len(sources))
        others = {
            name: table[name]
            for name in table.columns
            if name not in ("source", "target", "weight")
        }
    elif isinstance(edges, Mapping):
        sources, targets, weight_list = [], [], []
        for source, members in edges.items():
            if not isinstance(source, str) or not source:
                raise TypeError(f"sources must be non-empty strings, got {source!r}")
            if isinstance(members, Mapping):
                reject_masked(members, f"edges[{source!r}]", missing="NaN")
                pairs = list(members.items())
            else:
                genes = listed_values(members, f"the targets of {source!r}")
                pairs = [(gene, 1.0) for gene in genes]
            for target, weight in pairs:
                if not isinstance(target, str) or not target:
                    raise TypeError(
                        f"the targets of {source!r} must be gene ids (non-empty "
                        f"strings), got {target!r}"
                    )
                sources.append(str(source))
                targets.append(str(target))
                weight_list.append(weight)
        weights = _weights(weight_list, "the weights")
        others = {}
    else:
        raise TypeError(
            "edges must be a Table or pandas DataFrame with the columns 'source', "
            "'target' and 'weight', or a mapping of source to targets, got "
            f"{type(edges).__name__}"
        )
    if not sources:
        raise ValueError("the network has no edges")
    seen: dict[tuple[str, str], int] = {}
    for row, pair in enumerate(zip(sources, targets, strict=True)):
        if pair in seen:
            raise ValueError(
                f"the edge from {pair[0]!r} to {pair[1]!r} is given twice (rows "
                f"{seen[pair]} and {row}); give each edge once, with one weight"
            )
        seen[pair] = row
    return Table(
        {
            "source": text_column(sources),
            "target": text_column(targets),
            "weight": weights,
            **others,
        }
    )


def _names_column(values: np.ndarray, name: str) -> list[str]:
    found = values.tolist()
    names = []
    for row, value in enumerate(found):
        if isinstance(value, str) and value:
            names.append(str(value))
        elif is_missing(value):
            raise ValueError(
                f"the {name} of row {row} is missing; every edge needs one"
            )
        else:
            raise TypeError(
                f"the column {name!r} must hold names (strings), got "
                f"{type(value).__name__} {value!r} on row {row}"
            )
    return names


def _weights(values: Any, name: str) -> np.ndarray:
    try:
        weights = real_array(values, name, boolean=False, objects=False)
    except TypeError:
        raise TypeError(f"{name} must hold real numbers") from None
    weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    if not np.isfinite(weights).all():
        row = int(np.flatnonzero(~np.isfinite(weights))[0])
        raise ValueError(
            f"{name} must be finite; edge {row} has the weight {weights[row]}"
        )
    return weights


def _names(values: Any) -> list[Any]:
    try:
        return list(values)
    except TypeError:
        raise TypeError(
            f"sources must be a source or a list of them, got {type(values).__name__}"
        ) from None


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    reject_masked(value, name)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
    return value


def _digest(value: Any) -> str | None:
    digest = _text(value, "sha256")
    if digest is not None and (
        len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(
            "sha256 must be a SHA-256 digest, 64 hexadecimal digits in lower case, "
            f"got {digest!r}"
        )
    return digest


# Downloads and the cache.


def _cache_directory(cache_dir: Any) -> Path:
    if cache_dir is not None:
        reject_masked(cache_dir, "cache_dir")
        if not isinstance(cache_dir, str | os.PathLike):
            raise TypeError(
                "cache_dir must be a str or os.PathLike, got "
                f"{type(cache_dir).__name__}"
            )
        return Path(os.fspath(cache_dir))
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cytepool" / "networks"


def _fetched(download: _Download, cache_dir: Any) -> Path:
    """The pinned file in the cache, downloaded when it is not there yet."""
    directory = _cache_directory(cache_dir)
    path = directory / download.file
    if path.exists():
        size, digest = _size_and_digest(path)
        if size != download.size or digest != download.sha256:
            raise ValueError(
                f"{path} is not the file that CytePool pins: {size:,} bytes with "
                f"SHA-256 {digest}, where {download.url} published "
                f"{download.size:,} bytes with SHA-256 {download.sha256}; delete it "
                "to download it again"
            )
        return path
    directory.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=directory, prefix=download.file, suffix=".part")
    partial = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            try:
                request = urllib.request.Request(
                    download.url, headers={"User-Agent": _CLIENT}
                )
                with urllib.request.urlopen(request, timeout=_TIMEOUT) as answer:
                    while block := answer.read(_BLOCK):
                        out.write(block)
            except (urllib.error.URLError, OSError) as error:
                raise OSError(
                    f"could not download {download.file} from {download.url}: "
                    f"{error}. Without a connection, copy the file into {directory} "
                    "from a machine that has one, or pass cache_dir"
                ) from error
        size, digest = _size_and_digest(partial)
        if size != download.size or digest != download.sha256:
            raise ValueError(
                f"the download of {download.url} is not the file that CytePool pins: "
                f"{size:,} bytes with SHA-256 {digest}, not {download.size:,} bytes "
                f"with SHA-256 {download.sha256}; the source may have changed it, or "
                "the download was cut short"
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _size_and_digest(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while block := file.read(_BLOCK):
            hasher.update(block)
            size += len(block)
    return size, hasher.hexdigest()


# CollecTRI.


def _collectri_edges(data: bytes, level: str) -> dict[str, np.ndarray]:
    """The interactions of CollecTRI's file at a licence level, with their resources
    and references written as decoupler writes them."""
    rows = csv.reader(io.StringIO(data.decode("utf-8")))
    header = next(rows, None)
    if header is None or tuple(header) != _COLLECTRI_COLUMNS:
        raise ValueError(
            f"the CollecTRI file has the columns {header}, not those of version 2.0, "
            f"{list(_COLLECTRI_COLUMNS)}"
        )
    columns: dict[str, list[Any]] = {name: [] for name in _COLLECTRI_COLUMNS}
    for row in rows:
        if not row:
            continue
        source, target, weight, resources, references, sign = row
        # "CollecTRI;ExTRI_CollecTRI;TRRUST_CollecTRI" is "ExTRI;TRRUST", as decoupler
        # writes it.
        listed = sorted(
            {
                name.replace("_", "")
                for name in resources.replace("CollecTRI", "").split(";")
                if name
            }
        )
        if level == "commercial" and set(listed) <= _ACADEMIC_ONLY:
            continue
        columns["source"].append(source)
        columns["target"].append(target)
        columns["weight"].append(float(weight))
        columns["resources"].append(";".join(listed))
        columns["references"].append(references.replace("CollecTRI:", ""))
        columns["sign_decision"].append(sign)
    return {
        name: np.array(values, dtype=np.float64)
        if name == "weight"
        else text_column(values)
        for name, values in columns.items()
    }


# PROGENy.


def _progeny_edges(
    data: bytes, organism: str, top: int | None, max_p_value: float | None
) -> dict[str, np.ndarray]:
    """The genes of each PROGENy pathway, by increasing p-value, as kept."""
    name = f"model_{organism}_full"
    member = f"progeny/data/{name}.rda"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        try:
            stored = archive.extractfile(member)
        except KeyError:
            stored = None
        if stored is None:
            raise ValueError(f"the progeny package holds no {member}")
        objects = _read_r_data(stored.read())
    if name not in objects:
        raise ValueError(f"{member} holds no data frame {name!r}")
    frame = _data_frame(objects[name], name)
    genes = text_column(frame["gene"])
    pathways = text_column(frame["pathway"])
    weights = np.asarray(frame["weight"], dtype=np.float64)
    p_values = np.asarray(frame["p.value"], dtype=np.float64)
    kept: list[int] = []
    for pathway in sorted(set(pathways.tolist())):
        rows = np.flatnonzero(pathways == pathway)
        rows = rows[np.argsort(p_values[rows], kind="stable")]
        if top is not None and rows.size > top:
            # Those tied with the last one kept stay, as dplyr's top_n keeps them.
            rows = rows[p_values[rows] <= p_values[rows[top - 1]]]
        if max_p_value is not None:
            rows = rows[p_values[rows] < max_p_value]
        kept.extend(rows.tolist())
    taken = np.array(kept, dtype=np.intp)
    return {
        "source": pathways[taken],
        "target": genes[taken],
        "weight": weights[taken],
        "p_value": p_values[taken],
    }


def _data_frame(value: Any, name: str) -> dict[str, Any]:
    """The columns of an R data frame of numbers, text and factors, by name; factors
    as their levels."""
    if not isinstance(value, _RVector) or value.kind != _LIST:
        raise ValueError(f"{name} is not an R data frame")
    names = value.attributes.get("names")
    if not isinstance(names, _RVector) or len(names.values) != len(value.values):
        raise ValueError(f"the columns of {name} have no names")
    columns = {}
    for column, item in zip(names.values, value.values, strict=True):
        if not isinstance(item, _RVector):
            raise ValueError(f"the column {column!r} of {name} is not a vector")
        levels = item.attributes.get("levels")
        if levels is not None:
            codes = np.asarray(item.values)
            if (codes < 1).any() or (codes > len(levels.values)).any():
                raise ValueError(f"the factor {column!r} of {name} has missing values")
            columns[column] = [levels.values[code - 1] for code in codes.tolist()]
        else:
            columns[column] = item.values
    return columns


# R's serialization format.

_NIL = 254
_REFERENCE = 255
_ALTREP = 238
_ENVIRONMENTS = (241, 242, 253)  # the base, empty and global environments
_SYMBOL, _PAIRLIST, _CHARACTER_STRING = 1, 2, 9
_LOGICAL, _INTEGER, _REAL, _STRINGS, _LIST = 10, 13, 14, 16, 19
_HAS_ATTRIBUTES = 1 << 9
_HAS_TAG = 1 << 10
_LATIN1 = 1 << 14  # the LATIN1 flag of a string's levels, which start at bit 12
_NA_INTEGER = -(2**31)
_HEADERS = (b"RDX2\n", b"RDX3\n")


class _RVector(NamedTuple):
    """An R vector: its type, its values and its attributes."""

    kind: int
    values: Any  # a numpy array of numbers, or a list of strings or items
    attributes: dict[str, Any]


def _read_r_data(data: bytes) -> dict[str, Any]:
    """The objects of an R data file (``save()``), by name.

    Numbers become numpy arrays (integers as int64, R's missing integer kept as
    -2**31), text a list of str (None where missing), lists a list, each as an
    ``_RVector`` with its attributes.

    Raises:
        ValueError: If the data are not an R data file in the XDR format, or hold an
            object this reader does not read.
    """
    if data.startswith(b"\xfd7zXZ\x00"):
        data = lzma.decompress(data)
    elif data.startswith(b"BZh"):
        data = bz2.decompress(data)
    elif data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    if data[:5] not in _HEADERS or data[5:7] != b"X\n":
        raise ValueError(
            "not an R data file in the XDR format (save() of R 2.4 or later)"
        )
    reader = _RReader(data, 7)
    version = reader.integer()
    reader.integer()  # the version of R that wrote it
    reader.integer()  # the oldest version of R that reads it
    if version not in (2, 3):
        raise ValueError(f"R's serialization format {version} is not read")
    if version == 3:
        length = reader.integer()
        reader.position += length  # the name of the native encoding
    saved = reader.item()
    if not isinstance(saved, list):
        raise ValueError("an R data file holds a pairlist of named objects")
    return {tag: value for tag, value in saved}


class _RReader:
    """Reads R's serialization format, XDR, item by item."""

    def __init__(self, data: bytes, position: int) -> None:
        self.data = data
        self.position = position
        self.references: list[Any] = []

    def integer(self) -> int:
        if self.position + 4 > len(self.data):
            raise ValueError("the R data end in the middle of an object")
        value: int = struct.unpack_from(">i", self.data, self.position)[0]
        self.position += 4
        return value

    def length(self) -> int:
        length = self.integer()
        if length == -1:  # a long vector: the length in two words
            upper, lower = self.integer(), self.integer()
            length = (upper << 32) + (lower & 0xFFFFFFFF)
        if length < 0:
            raise ValueError(f"the R data hold a vector of length {length}")
        return length

    def numbers(self, stored: str, size: int, length: int, kept: type) -> np.ndarray:
        """``length`` numbers stored big-endian as ``stored``, as ``kept``."""
        end = self.position + size * length
        if end > len(self.data):
            raise ValueError("the R data end in the middle of a vector")
        values = np.frombuffer(self.data, stored, length, self.position)
        self.position = end
        return values.astype(kept)

    def item(self) -> Any:
        flags = self.integer()
        kind = flags & 0xFF
        if kind == _NIL or kind in _ENVIRONMENTS:
            return None
        if kind == _REFERENCE:
            index = flags >> 8 or self.integer()
            if not 1 <= index <= len(self.references):
                raise ValueError(f"the R data refer to a missing object {index}")
            return self.references[index - 1]
        if kind == _SYMBOL:
            symbol = self.item()
            self.references.append(symbol)
            return symbol
        if kind == _PAIRLIST:
            return self.pairlist(flags)
        if kind == _CHARACTER_STRING:
            length = self.integer()
            if length == -1:
                return None
            text = self.data[self.position : self.position + length]
            self.position += length
            return text.decode("latin-1" if flags & _LATIN1 else "utf-8")
        if kind == _ALTREP:
            return self.altrep()
        if kind in (_LOGICAL, _INTEGER):
            values: Any = self.numbers(">i4", 4, self.length(), np.int64)
        elif kind == _REAL:
            values = self.numbers(">f8", 8, self.length(), np.float64)
        elif kind in (_STRINGS, _LIST):
            values = [self.item() for _ in range(self.length())]
        else:
            raise ValueError(f"the R data hold an object of type {kind}, not read here")
        attributes = self.attributes() if flags & _HAS_ATTRIBUTES else {}
        return _RVector(kind, values, attributes)

    def pairlist(self, flags: int) -> list[tuple[Any, Any]]:
        """The (tag, value) pairs of a pairlist, its cells read one after another."""
        cells = []
        while True:
            if flags & _HAS_ATTRIBUTES:
                self.item()  # the attributes of a cell, which data frames do not use
            tag = self.item() if flags & _HAS_TAG else None
            cells.append((tag, self.item()))
            flags = self.integer()
            if flags & 0xFF == _NIL:
                return cells
            if flags & 0xFF != _PAIRLIST:
                raise ValueError("a pairlist of the R data continues with another type")

    def attributes(self) -> dict[str, Any]:
        flags = self.integer()
        if flags & 0xFF != _PAIRLIST:
            raise ValueError("the attributes of an R vector are not a pairlist")
        return {tag: value for tag, value in self.pairlist(flags)}

    def altrep(self) -> _RVector:
        """A compact sequence, as R writes row names 1 to n; or a wrapped vector."""
        info = self.item()
        state = self.item()
        attributes = self.item()
        names = [value for _, value in info] if isinstance(info, list) else []
        name = names[0] if names else None
        if name in ("compact_intseq", "compact_realseq") and isinstance(
            state, _RVector
        ):
            length, first, step = (float(v) for v in np.asarray(state.values)[:3])
            kind = _INTEGER if name == "compact_intseq" else _REAL
            values = first + step * np.arange(int(length), dtype=np.float64)
            if kind == _INTEGER:
                values = values.astype(np.int64)
        elif isinstance(name, str) and name.startswith("wrap_") and state is not None:
            wrapped = state.values[0]
            kind, values = wrapped.kind, wrapped.values
        else:
            raise ValueError(
                f"the R data hold a compact vector {name!r}, not read here"
            )
        found = dict(attributes) if isinstance(attributes, list) else {}
        return _RVector(kind, values, found)
