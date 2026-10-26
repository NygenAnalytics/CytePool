"""Collections of gene sets, read from and written to GMT files.

A :class:`GeneSets` maps the name of each gene set to its genes, named by gene id, and
keeps the description of each set and where the collection comes from: its source,
version and licence, as the user states them from what the provider of the
collection documents, and, for a collection read from a file, the name of the file
and its SHA-256 digest. CytePool bundles no collection: MSigDB (Liberzon et al.
2011), Reactome, WikiPathways and the Gene Ontology distribute their sets under their
own terms, and :func:`read_gmt` reads the files that users download from them.

A ``GeneSets`` is a mapping of set name to genes, so every gene-set function of
CytePool takes it as it is: as ``gene_sets`` of
:func:`~cytepool.stats.gene_set_tests.camera`,
:func:`~cytepool.stats.gene_set_tests.fry`,
:func:`~cytepool.stats.rotation_gene_set_tests.mroast`,
:func:`~cytepool.stats.rotation_gene_set_tests.romer` and
:func:`~cytepool.over_representation.over_representation`, and one of its sets, as
``gene_sets[name]``, as ``gene_set`` of
:func:`~cytepool.stats.rotation_gene_set_tests.roast`. The tests of a linear model
match the ids to their ``gene_ids``.

Genes and sets:
    A set's genes are a tuple of gene ids (non-empty strings, compared exactly, case
    included), each once, in the order given: a gene listed twice keeps its first
    place, and the genes of a Python ``set`` are taken in sorted order, so that every
    run lists them alike. Set names are non-empty strings. A description of ``"na"``,
    which GSEA writes for a set without one, or an empty description, is None.

Matching and filtering:
    :meth:`GeneSets.restrict_to` keeps the genes of each set that an analysis measured
    and drops the sets left without genes, as limma's ``ids2indices`` does by default,
    and refuses ids that match no gene at all, as when the sets name genes by symbol
    and the analysis by Ensembl id. :meth:`GeneSets.filter_by_size` keeps the sets of a
    range of sizes; restricted first, the sizes are those within the analysis, as
    GSEA counts them before it leaves out sets of fewer than 15 or more than 500
    genes, its defaults (GSEA team 2026). :meth:`GeneSets.select` takes sets by name.
    Each returns a new ``GeneSets`` with the descriptions and the record of the
    source.

GMT files:
    The GMT (gene matrix transposed) format of GSEA (Subramanian et al. 2005; GSEA
    team 2026) holds one gene set per line, in fields separated by tabs: the set's
    name, a description, and the set's genes. :func:`read_gmt` reads:

    - fields separated by tabs only, so that names with spaces, such as Reactome's
      pathway names, are read whole;
    - lines that end in a line feed, a carriage return and line feed, or a carriage
      return; blank lines are skipped;
    - empty fields among the genes, such as a tab at the end of a line, which are
      skipped, and spaces at either end of a field, which are removed;
    - a set without genes (a name and a description only), kept with no genes;
    - UTF-8 text, with or without a byte-order mark, compressed by gzip or not
      (recognised by the first bytes of the file, whatever its name).

    A line with one field, a set without a name, or a name used for two sets is
    refused, with the number of the line. :func:`write_gmt` writes the same format,
    with ``"na"`` for a set without a description, so that :func:`read_gmt` reads
    back the same sets and descriptions. A GMT file holds no source, version or
    licence; they are given again when the file is read.

References:
    Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, Paulovich
    A, Pomeroy SL, Golub TR, Lander ES, Mesirov JP (2005). Gene set enrichment
    analysis: a knowledge-based approach for interpreting genome-wide expression
    profiles. Proceedings of the National Academy of Sciences 102, 15545-15550.
    doi:10.1073/pnas.0506580102

    Liberzon A, Subramanian A, Pinchback R, Thorvaldsdottir H, Tamayo P, Mesirov JP
    (2011). Molecular signatures database (MSigDB) 3.0. Bioinformatics 27, 1739-1740.
    doi:10.1093/bioinformatics/btr260

    GSEA team (2026). GSEA-MSigDB documentation: the GSEA User Guide and Data
    Formats. UC San Diego and Broad Institute, https://docs.gsea-msigdb.org/
"""

import gzip
import hashlib
import math
import numbers
import os
import re
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ._gene_ids import gene_ids, listed_values
from ._masked_arrays import reject_masked
from ._messages import close_matches, listing

__all__ = ["GeneSets", "read_gmt", "write_gmt"]

# The first bytes of a gzip stream (RFC 1952).
_GZIP_MAGIC = b"\x1f\x8b"
# What GSEA writes for a set without a description.
_NO_DESCRIPTION = "na"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SETS_SHOWN = 3  # the sets that a repr names


class GeneSets(Mapping[str, tuple[str, ...]]):
    """Named gene sets, with their descriptions and the source of the collection.

    A read-only mapping of set name to the set's genes, a tuple of gene ids, so that
    the gene-set functions of CytePool take it as their ``gene_sets``. The module
    docstring says how ids, names and descriptions are read. ``read_gmt`` builds one
    from a GMT file. Immutable.

    Args:
        sets: Mapping of set name to the set's genes, as gene ids (strings): a list,
            tuple, set or 1-D array of them. A gene listed twice counts once.
        descriptions: Mapping of set name to the set's description (a string), for
            some or all of the sets; the others have None.
        source: Where the sets come from, such as ``"MSigDB hallmark gene sets"``.
        version: The version or release of the collection, such as
            ``"2024.1.Hs"``.
        licence: The terms under which the provider distributes the sets, such as
            ``"CC BY 4.0"``, as its documentation states them.
        file: The name of the file the sets were read from.
        sha256: The SHA-256 digest of that file, 64 hexadecimal digits in lower case.

    Raises:
        TypeError: If ``sets`` or ``descriptions`` is not a mapping, a name, gene id
            or text is not a string, a set's genes are a single string or map genes
            to weights, or an argument is or holds a numpy masked array.
        ValueError: If a name or gene id is empty, a description names a set that
            is not in ``sets``, or ``sha256`` is not a SHA-256 digest.

    Example:
        >>> sets = GeneSets({"interferon": ["ISG15", "IFIT1", "MX1"]}, source="my lab")
        >>> sets["interferon"]
        ('ISG15', 'IFIT1', 'MX1')
    """

    __slots__ = (
        "_sets",
        "_descriptions",
        "_source",
        "_version",
        "_licence",
        "_file",
        "_sha256",
    )

    _sets: dict[str, tuple[str, ...]]
    _descriptions: dict[str, str | None]
    _source: str | None
    _version: str | None
    _licence: str | None
    _file: str | None
    _sha256: str | None

    def __init__(
        self,
        sets: Mapping[str, Any],
        *,
        descriptions: Mapping[str, Any] | None = None,
        source: str | None = None,
        version: str | None = None,
        licence: str | None = None,
        file: str | None = None,
        sha256: str | None = None,
    ) -> None:
        reject_masked(sets, "sets")
        reject_masked(descriptions, "descriptions")
        if not isinstance(sets, Mapping):
            raise TypeError(
                "sets must be a mapping of set name to genes, such as "
                f"{{'interferon': ['ISG15', 'IFIT1']}}; got {type(sets).__name__}"
            )
        ids: dict[str, str] = {}  # one string object per gene id
        checked = {}
        for name, genes in sets.items():
            checked[_set_name(name)] = _set_genes(name, genes, ids)
        texts: dict[str, str | None] = dict.fromkeys(checked)
        if descriptions is not None:
            if not isinstance(descriptions, Mapping):
                raise TypeError(
                    "descriptions must be a mapping of set name to description, got "
                    f"{type(descriptions).__name__}"
                )
            for name, text in descriptions.items():
                if name not in checked:
                    raise ValueError(
                        f"descriptions names the set {name!r}, which is not among the "
                        "sets"
                    )
                texts[name] = _description(text, f"the description of {name!r}")
        record = {
            "source": _text(source, "source"),
            "version": _text(version, "version"),
            "licence": _text(licence, "licence"),
            "file": _text(file, "file"),
            "sha256": _digest(sha256),
        }
        _initialize(self, checked, texts, record)

    # Immutability and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "GeneSets is immutable; select(), restrict_to() and filter_by_size() return "
            "new collections"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("GeneSets is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (self._sets, self._descriptions, self._record()))

    # The mapping.

    def __getitem__(self, name: str) -> tuple[str, ...]:
        """Return the genes of the set named ``name``.

        Raises:
            KeyError: If there is no such set; the message suggests close names.
        """
        try:
            return self._sets[name]
        except (KeyError, TypeError):
            hint = (
                close_matches(name, list(self._sets)) if isinstance(name, str) else ""
            )
            raise KeyError(f"no gene set named {name!r}{hint}") from None

    def __contains__(self, name: object) -> bool:
        """Return whether a set has this name."""
        try:
            return name in self._sets
        except TypeError:  # unhashable
            return False

    def __iter__(self) -> Iterator[str]:
        """Iterate over the set names, in order."""
        return iter(self._sets)

    def __len__(self) -> int:
        """Return the number of sets."""
        return len(self._sets)

    # What the collection records.

    @property
    def descriptions(self) -> Mapping[str, str | None]:
        """The description of every set, None where it has none, as a read-only
        mapping."""
        return MappingProxyType(self._descriptions)

    @property
    def source(self) -> str | None:
        """Where the sets come from, as stated; None when not stated."""
        return self._source

    @property
    def version(self) -> str | None:
        """The version or release of the collection, as stated; None when not
        stated."""
        return self._version

    @property
    def licence(self) -> str | None:
        """The terms under which the sets are distributed, as stated; None when not
        stated."""
        return self._licence

    @property
    def file(self) -> str | None:
        """The name of the file the sets were read from; None for sets not read from
        a file."""
        return self._file

    @property
    def sha256(self) -> str | None:
        """The SHA-256 digest of the bytes of that file, as stored (compressed or
        not); None for sets not read from a file."""
        return self._sha256

    # New collections.

    def select(self, names: str | Any) -> "GeneSets":
        """Return the sets with the given names, in that order.

        Args:
            names: A set name, or a sequence of them.

        Raises:
            KeyError: If a name is not a set's; the message suggests close names.
            TypeError: If a name is not a string, or ``names`` is or holds a numpy
                masked array.
            ValueError: If a name is given twice.
        """
        reject_masked(names, "names")
        given = [names] if isinstance(names, str) else _names_given(names)
        chosen: list[str] = []
        for name in given:
            if not isinstance(name, str):
                raise TypeError(f"set names are strings, got {name!r}")
            if name not in self._sets:
                hint = close_matches(name, list(self._sets))
                raise KeyError(f"no gene set named {name!r}{hint}")
            if name in chosen:
                raise ValueError(f"the set {name!r} is selected more than once")
            chosen.append(str(name))
        return self._derived({name: self._sets[name] for name in chosen})

    def restrict_to(self, genes: Any, *, keep_empty: bool = False) -> "GeneSets":
        """Return the sets with only their genes among ``genes``.

        Each set keeps its genes that are among ``genes``, in its own order, as the
        gene-set tests would match them; sets left without genes are dropped, as
        limma's ``ids2indices`` drops them by default, unless ``keep_empty``.

        Args:
            genes: The ids of the genes of an analysis, such as the ``feature_id`` or
                ``symbol`` column of a differential-expression table. Missing ids
                (None, NaN or empty strings) are ignored.
            keep_empty: Whether to keep the sets left without genes.

        Raises:
            TypeError: If ``genes`` does not hold strings, or is or holds a numpy
                masked array.
            ValueError: If no gene of any set is among ``genes``: the sets and the
                analysis name genes in different ways, for example by symbol and by
                Ensembl id.
        """
        if not isinstance(keep_empty, bool | np.bool_):
            raise TypeError(f"keep_empty must be True or False, got {keep_empty!r}")
        measured = set(gene_ids(genes, "genes")[0])
        kept: dict[str, tuple[str, ...]] = {}
        matched = 0
        for name, members in self._sets.items():
            inside = tuple(gene for gene in members if gene in measured)
            matched += len(inside)
            if inside or keep_empty:
                kept[name] = inside
        if matched == 0 and any(self._sets.values()):
            raise ValueError(
                "no gene of any set is among genes; the sets and the analysis must "
                "name genes the same way (for example both by symbol, or both by "
                f"Ensembl id): the sets name genes such as {_example_ids(self._sets)}"
            )
        return self._derived(kept)

    def filter_by_size(self, min_size: int, max_size: int | None = None) -> "GeneSets":
        """Return the sets of at least ``min_size`` and at most ``max_size`` genes.

        Sizes are the numbers of genes the sets hold: to count the genes of an
        analysis only, as GSEA does, call :meth:`restrict_to` first.

        Args:
            min_size: The fewest genes a set may hold, at least 0.
            max_size: The most genes a set may hold, at least ``min_size``; None for no
                limit.

        Raises:
            TypeError: If a size is not an integer.
            ValueError: If ``min_size`` is negative or ``max_size`` is smaller than
                ``min_size``.
        """
        smallest = _size(min_size, "min_size")
        largest: float = math.inf if max_size is None else _size(max_size, "max_size")
        if largest < smallest:
            raise ValueError(
                f"max_size ({max_size}) is smaller than min_size ({min_size}), so no "
                "set could be kept"
            )
        return self._derived(
            {
                name: members
                for name, members in self._sets.items()
                if smallest <= len(members) <= largest
            }
        )

    def __repr__(self) -> str:
        sizes = [len(members) for members in self._sets.values()]
        count = len(sizes)
        head = f"GeneSets: {count:,} set{'' if count == 1 else 's'}"
        if sizes:
            distinct = len(
                {gene for members in self._sets.values() for gene in members}
            )
            spread = (
                f"{min(sizes):,}" if min(sizes) == max(sizes) else
                f"{min(sizes):,} to {max(sizes):,}"
            )  # fmt: skip
            head += (
                f" of {spread} gene{'' if max(sizes) == 1 else 's'}, {distinct:,} "
                f"gene{'' if distinct == 1 else 's'} in all"
            )
        lines = [head]
        stated = [
            f"{label}: {value if value is not None else 'not stated'}"
            for label, value in (
                ("source", self._source),
                ("version", self._version),
                ("licence", self._licence),
            )
        ]
        lines.append("  " + "; ".join(stated))
        if self._file is not None:
            digest = f", SHA-256 {self._sha256}" if self._sha256 else ""
            lines.append(f"  read from {self._file}{digest}")
        if sizes:
            shown = [
                f"{name} ({len(members):,})"
                for name, members in list(self._sets.items())[:_SETS_SHOWN]
            ]
            more = count - len(shown)
            lines.append(
                "  sets: " + ", ".join(shown) + (f" and {more:,} more" if more else "")
            )
        return "\n".join(lines)

    def _record(self) -> dict[str, str | None]:
        return {
            "source": self._source,
            "version": self._version,
            "licence": self._licence,
            "file": self._file,
            "sha256": self._sha256,
        }

    def _derived(self, sets: dict[str, tuple[str, ...]]) -> "GeneSets":
        """A collection of some of these sets, with their descriptions and record."""
        descriptions = {name: self._descriptions[name] for name in sets}
        derived = object.__new__(GeneSets)
        _initialize(derived, sets, descriptions, self._record())
        return derived


def read_gmt(
    path: str | os.PathLike[str],
    *,
    source: str | None = None,
    version: str | None = None,
    licence: str | None = None,
) -> GeneSets:
    """Read gene sets from a GMT file.

    One set per line: its name, a description and its genes, separated by tabs, as
    MSigDB, Reactome, WikiPathways and the GMT exports of the Gene Ontology write
    them; the module docstring says what is read and how. The collection records the
    name of the file and the SHA-256 digest of its bytes, and the source, version and
    licence given here, which the file cannot hold: state them as the provider
    documents them, so that results can say which sets they used.

    Args:
        path: The GMT file, compressed by gzip or not.
        source: Where the sets come from, such as ``"MSigDB hallmark gene sets"``.
        version: The version or release of the collection, such as ``"2024.1.Hs"``.
        licence: The terms under which the provider distributes the sets, such as
            ``"CC BY 4.0"``.

    Returns:
        The sets, in the order of the file.

    Raises:
        TypeError: If ``path`` is not a str or os.PathLike, or an option is not a
            string.
        ValueError: If the file is not UTF-8 text or valid gzip, holds no gene set,
            or a line holds one field only, a set without a name or the name of an
            earlier set; the message gives the line.
        FileNotFoundError: If there is no file at ``path``.
    """
    location = _file_path(path)
    record = {
        "source": _text(source, "source"),
        "version": _text(version, "version"),
        "licence": _text(licence, "licence"),
    }
    data = location.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if data.startswith(_GZIP_MAGIC):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as error:
            raise ValueError(
                f"{location.name} is not a valid gzip file: {error}"
            ) from None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        number = data[: error.start].count(b"\n") + 1
        raise ValueError(
            f"{location.name} is not UTF-8 text: line {number} holds the byte "
            f"{data[error.start]:#04x}; GMT files are UTF-8"
        ) from None
    sets: dict[str, tuple[str, ...]] = {}
    descriptions: dict[str, str | None] = {}
    first_line: dict[str, int] = {}
    ids: dict[str, str] = {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        where = f"{location.name}, line {number}"
        if len(fields) < 2:
            raise ValueError(
                f"{where} has no tab; a GMT line holds a set's name, its description "
                "and its genes, separated by tabs"
            )
        name = fields[0].strip()
        if not name:
            raise ValueError(f"{where}: the gene set has no name (the first field)")
        if name in sets:
            raise ValueError(
                f"{where}: the gene set {name!r} is also on line {first_line[name]}; "
                "each set needs a name of its own"
            )
        genes = (field.strip() for field in fields[2:])
        sets[name] = tuple(
            dict.fromkeys(ids.setdefault(gene, gene) for gene in genes if gene)
        )
        descriptions[name] = _read_description(fields[1].strip())
        first_line[name] = number
    if not sets:
        raise ValueError(f"{location.name} holds no gene set")
    collection = object.__new__(GeneSets)
    record |= {"file": location.name, "sha256": digest}
    _initialize(collection, sets, descriptions, record)
    return collection


def write_gmt(
    gene_sets: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Write gene sets to a GMT file.

    One line per set, in order: its name, its description (``"na"`` without one) and
    its genes, separated by tabs, in UTF-8 with line feeds, compressed by gzip when
    the name of the file ends in ``.gz``. :func:`read_gmt` reads back the same sets
    and descriptions. The source, version and licence of a ``GeneSets`` are not
    written: GMT has no place for them.

    Args:
        gene_sets: A ``GeneSets``, or a mapping of set name to gene ids.
        path: The file to write.
        overwrite: Whether to replace a file that exists at ``path``.

    Raises:
        TypeError: If ``gene_sets`` is not a mapping of names to gene ids, or
            ``overwrite`` is not a bool.
        ValueError: If a name, description or gene id holds a tab or a line break,
            or starts or ends with spaces, which a GMT file cannot hold as written.
        FileExistsError: If a file exists at ``path`` and ``overwrite`` is False.
    """
    reject_masked(gene_sets, "gene_sets")
    collection = gene_sets if isinstance(gene_sets, GeneSets) else GeneSets(gene_sets)
    if not isinstance(overwrite, bool | np.bool_):
        raise TypeError(f"overwrite must be True or False, got {overwrite!r}")
    location = _file_path(path)
    lines = []
    for name, genes in collection.items():
        description = collection.descriptions[name]
        fields = [name, _NO_DESCRIPTION if description is None else description]
        for field in [*fields, *genes]:
            _check_field(field, name)
        lines.append("\t".join([*fields, *genes]) + "\n")
    data = "".join(lines).encode("utf-8")
    if location.name.endswith(".gz"):
        # No time stamp, so that the same sets give the same bytes.
        data = gzip.compress(data, mtime=0)
    with open(location, "wb" if overwrite else "xb") as out:
        out.write(data)


# Checks of names, genes and texts.


def _initialize(
    collection: GeneSets,
    sets: dict[str, tuple[str, ...]],
    descriptions: dict[str, str | None],
    record: Mapping[str, str | None],
) -> None:
    object.__setattr__(collection, "_sets", sets)
    object.__setattr__(collection, "_descriptions", descriptions)
    for key in ("source", "version", "licence", "file", "sha256"):
        object.__setattr__(collection, f"_{key}", record[key])


def _rebuild(
    sets: dict[str, tuple[str, ...]],
    descriptions: dict[str, str | None],
    record: dict[str, str | None],
) -> GeneSets:
    collection = object.__new__(GeneSets)
    _initialize(collection, dict(sets), dict(descriptions), record)
    return collection


def _set_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError(f"gene set names must be strings, got {name!r}")
    if not name:
        raise ValueError("gene set names must not be empty")
    return str(name)


def _set_genes(name: str, genes: Any, ids: dict[str, str]) -> tuple[str, ...]:
    """A set's genes as a tuple of distinct ids, each the one string of its id."""
    if isinstance(genes, Mapping):
        raise TypeError(
            f"gene set {name!r} maps genes to weights; a GeneSets holds the genes of "
            "each set, as a list"
        )
    members = []
    for gene in listed_values(genes, f"gene set {name!r}"):
        if not isinstance(gene, str):
            raise TypeError(
                f"gene set {name!r} holds {gene!r}; genes are named by id, as strings "
                "(convert numeric ids such as Entrez ids to text)"
            )
        if not gene:
            raise ValueError(f"gene set {name!r} holds an empty gene id")
        members.append(ids.setdefault(gene, gene))
    return tuple(dict.fromkeys(members))


def _description(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or None, got {type(value).__name__}")
    return _read_description(value)


def _read_description(text: str) -> str | None:
    return None if text in ("", _NO_DESCRIPTION) else text


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    reject_masked(value, name)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
    return value


def _digest(value: Any) -> str | None:
    digest = _text(value, "sha256")
    if digest is not None and not _SHA256.fullmatch(digest):
        raise ValueError(
            "sha256 must be a SHA-256 digest, 64 hexadecimal digits in lower case, "
            f"got {digest!r}"
        )
    return digest


def _size(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    size = int(value)
    if size < 0:
        raise ValueError(f"{name} must be at least 0, got {size}")
    return size


def _names_given(names: Any) -> list[Any]:
    try:
        return list(names)
    except TypeError:
        raise TypeError(
            f"names must be a set name or a list of them, got {type(names).__name__}"
        ) from None


def _example_ids(sets: Mapping[str, tuple[str, ...]]) -> str:
    examples = []
    for members in sets.values():
        for gene in members:
            if gene not in examples:
                examples.append(gene)
            if len(examples) == 3:
                return listing([repr(gene) for gene in examples])
    return listing([repr(gene) for gene in examples])


def _file_path(path: Any) -> Path:
    reject_masked(path, "path")
    if not isinstance(path, str | os.PathLike):
        raise TypeError(f"path must be a str or os.PathLike, got {type(path).__name__}")
    text = os.fspath(path)
    if not isinstance(text, str):
        raise TypeError("path must be text, not bytes")
    return Path(text)


def _check_field(field: str, name: str) -> None:
    if any(character in field for character in "\t\n\r"):
        raise ValueError(
            f"gene set {name!r} holds {field!r}, with a tab or a line break, which a "
            "GMT file cannot hold"
        )
    if field != field.strip():
        raise ValueError(
            f"gene set {name!r} holds {field!r}, with spaces at an end, which a GMT "
            "reader removes"
        )
