"""The pseudobulk cube: additive statistics per sample, group and feature."""

import json
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .._columns import values_equal, with_missing
from .._masked_arrays import reject_masked, reject_masked_table, unmasked_array
from ..table import Table
from .fingerprints import (
    _block_digest,
    _cells_digest,
    _check_record,
    _column_type,
    _digest_record,
    _layer_type,
    _normalization_digest,
    _table_digest,
)
from .provenance import (
    _add_record,
    _plain_json,
    _provenance_of,
    _provenance_texts,
    describe_provenance,
)

LAYER_NAMES = ("counts", "detected", "log_sum", "log_sum_sq", "linear_sum")
"""The layers a cube can hold, in the order a cube stores and lists them."""

POOLED_SAMPLES_COLUMN = "n_pooled_samples"
"""The sample column that ``Cube.collapse_samples`` fills with pooled sample counts."""

_MOMENT_TOLERANCE = 1e-6  # relative slack of the Cauchy-Schwarz check on the moments
# The largest int64. Integer layers and n_cells hold values from 0 to it, so every
# value converts to int64 exactly, and a sum is exact in int64 whenever it fits.
_INT64_MAX = 2**63 - 1
_CELL_LAYERS = ("detected", "log_sum", "log_sum_sq", "linear_sum")
_NORMALIZED_LAYERS = ("log_sum", "log_sum_sq", "linear_sum")
_NON_NEGATIVE_LAYERS = frozenset({"counts", "detected", "log_sum_sq", "linear_sum"})
_OPTIONAL_TEXT_COLUMNS = {
    "groups": ("cell_ontology_id", "parent"),
    "features": ("symbol",),
}
_OPTIONAL_BOOLEAN_COLUMNS = {"features": ("measured",)}
# The parts of a cube that have one digest each besides its layers, as messages name
# them.
_PART_NAMES = {
    "samples": "the samples table",
    "groups": "the groups table",
    "features": "the features table",
    "n_cells": "n_cells",
    "normalization": "the normalization",
}
# Reads the block of a group of a lazy layer at sample and feature positions (None
# for every position), as _LazyLayer.read does.
_BlockReader = Callable[[int, np.ndarray | None, np.ndarray | None], np.ndarray]


class Cube:
    """Additive statistics per sample, group and feature.

    A cube pools the cells of each sample and group (usually a cell type) and keeps,
    for every feature, sums over those cells. Every layer is a sum, so merging groups
    or pooling samples is an exact sum and never needs the cells again. A cube is
    immutable: methods that change something return a new cube.

    Layout:
        Layers have shape ``(n_groups, n_samples, n_features)``, so one group is a
        contiguous samples x features matrix, which is what a per-group model fits.
        Quantities with one value per sample and group, ``n_cells`` and
        ``library_sizes()``, have shape ``(n_samples, n_groups)``, like a table of
        cell type composition.

    Layers:
        counts: Sum of raw counts, the one required layer. An integer dtype, or
            float64 for non-integer counts such as transcript-level estimates.
        detected: Number of cells with a positive count. An integer dtype.
        log_sum: Sum over cells of the log-transformed normalized value, ``log1p``
            for the usual natural log with pseudocount 1 (``normalization`` records
            the transform). float64.
        log_sum_sq: Sum over cells of the squares of those values. float64.
        linear_sum: Sum over cells of the normalized value. float64.

    Args:
        layers: Layer name to an array of shape ``(n_groups, n_samples,
            n_features)``, or to a lazy array-like (see below).
        samples: Sample metadata, as a ``Table`` or anything ``Table`` accepts, with
            a unique text column ``sample_id``.
        groups: Group metadata with a unique text column ``group_id``, and optional
            text columns ``cell_ontology_id`` and ``parent``.
        features: Feature metadata with a unique text column ``feature_id``, an
            optional text column ``symbol`` and an optional boolean column
            ``measured``.
        n_cells: Integers of shape ``(n_samples, n_groups)``: the cells pooled into
            each entry. ``None`` for bulk data, where cells are unknown. Required
            with ``detected``, ``log_sum``, ``log_sum_sq`` or ``linear_sum``. A
            pandas DataFrame indexed by sample ids with group ids as columns, such
            as the output of ``pandas.crosstab``, is aligned by those labels.
        normalization: How normalized values were computed, as a JSON-serialisable
            mapping with a ``"method"`` string, for example ``{"method":
            "library_size", "target_sum": 10000.0, "log": "natural", "pseudocount":
            1.0}``. Required with ``log_sum``, ``log_sum_sq`` or ``linear_sum``.
        provenance: A JSON-serialisable mapping that says where the data came from
            and how the cube was made (``cytepool.cube.provenance`` documents the
            keys CytePool writes). Operations that derive a cube append a record of
            the operation, its input and what it kept or pooled to the list under
            ``"operations"``.
        copy: If False, a layer given as a C-contiguous numpy array with an
            accepted native dtype is stored without a copy and made read-only in
            place, so it must not be changed through other references afterwards.
            Other layers are always copied.

    Raises:
        TypeError: If an argument has the wrong type or dtype (a table column of a
            type below included), or is or holds a numpy masked array.
        ValueError: If shapes disagree, a required argument is missing, an id is
            missing or repeated, or a value breaks a constraint below.

    Constraints:
        Table columns hold text, booleans, integers, float16 to float64,
        complex64, complex128, datetimes or timedeltas: the types cube files store,
        and that a fingerprint encodes alike on every platform (longdouble, for
        example, is refused).

        ``counts``, ``detected``, ``log_sum_sq`` and ``linear_sum`` are
        non-negative; integer layers and ``n_cells`` are at most 2**63 - 1, the
        largest int64, whatever their dtype; float layers are finite; ``detected``
        is at most ``n_cells``; every layer is zero in entries where ``n_cells`` is
        zero; and, because both moments sum over the same cells, ``log_sum**2`` is
        at most ``n_cells * log_sum_sq`` (Cauchy-Schwarz), up to a relative 1e-6
        for rounding. ``log_sum`` may be negative, which happens with a pseudocount
        below 1. Arrays are stored read-only.

    Sums:
        Library sizes, merged groups and pooled samples are sums. Integer layers
        and ``n_cells`` are summed in int64, and every sum is exact: one that would
        exceed 2**63 - 1 raises OverflowError, naming the layer, the entry and the
        exact sum, instead of wrapping around. Float layers are summed in float64,
        and a sum beyond the float64 range raises OverflowError too.

    Lazy layers:
        A layer may be any object with ``shape``, ``dtype`` and a ``__getitem__``
        that accepts an integer on the first axis and returns one group, such as an
        HDF5 or zarr array; a numpy memmap is also kept lazy. Construction checks
        its shape and dtype and reads nothing. Each read puts what it needs in
        memory the cube owns, copying what the layer returns unless the layer is a
        ``ZarrArray`` (see below), and checks those values against the
        constraints, so a bad value raises when it is first used; the moment
        constraint is checked when ``variance()`` or ``to_memory()`` reads both
        moments. Selecting, regrouping and collapsing a lazy cube give a lazy cube,
        which computes its sums, and raises any OverflowError, when a group is
        read; ``to_memory()`` reads every layer.

        A lazy cube asks a layer only for the groups, samples and features it
        needs, such as those that ``select()`` kept. A layer that also has an
        ``oindex`` attribute for orthogonal indexing, as zarr-python arrays have,
        is read with ``layer.oindex[group, samples, features]``, where ``group`` is
        an integer and ``samples`` and ``features`` are each ``slice(None)`` or a
        1-D integer numpy array of positions in increasing order without repeats;
        it must return an array of shape ``(len(samples), len(features))``,
        counting a slice as the whole axis. Any other layer is read one whole group
        at a time, and the cube keeps the part it needs.

        The layers of ``read_cube`` are ``ZarrArray`` objects. They read only the
        chunks that hold the selected features, take positions in any order and
        return new arrays, which the cube keeps without a copy, so a read holds
        about the values it returns and one chunk (with zstd, also the decoder's
        window, 2 MiB at the default level).

        A read of a ``ZarrArray`` layer that covers a whole group, in any order (a
        group read whole, merged or with its samples pooled, or a selection that
        reorders samples or features), also hashes the group, to check it against
        its digest: stored with the cube file, or kept from the first such read. A
        read of a whole group of a layer derived from one hashes that group too, to
        keep its digest for ``fingerprint()``. SHA-256 runs at about 2 GB/s on
        processors with SHA instructions and 0.5 GB/s without, which adds about a
        fifth to the time of reading a gzip-compressed group on the former. A
        selection of some samples or features reads part of a group, which cannot
        be checked on its own; ``verify_fingerprint()`` reads the groups it comes
        from whole.

    Fingerprint and provenance:
        ``fingerprint()`` identifies the cube's values, tables and cell numbers
        included, and ``provenance`` records how the cube was made: the caller's
        source record, the function that built it with its settings, and each
        operation that derived it, with the fingerprint of its input.
        ``history()`` reads the record back in plain terms;
        ``cytepool.cube.fingerprints`` and ``cytepool.cube.provenance`` document
        both. To record the fingerprint of its input, the first operation applied
        to a cube in memory hashes its layers, once, at about 1 s per GB of values
        on processors with SHA instructions.
    """

    __slots__ = (
        "_layers",
        "_samples",
        "_groups",
        "_features",
        "_n_cells",
        "_normalization",
        "_provenance",
        "_sample_index",
        "_group_index",
        "_feature_index",
        "_digests",
    )

    _layers: dict[str, "np.ndarray | _LazyLayer"]
    _samples: Table
    _groups: Table
    _features: Table
    _n_cells: np.ndarray | None
    _normalization: str | None
    _provenance: tuple[str, ...]  # JSON texts, see provenance._provenance_texts
    _sample_index: dict[str, int]
    _group_index: dict[str, int]
    _feature_index: dict[str, int]
    _digests: "_Digests"

    def __init__(
        self,
        layers: Mapping[str, Any],
        *,
        samples: object,
        groups: object,
        features: object,
        n_cells: Any = None,
        normalization: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        copy: bool = True,
    ) -> None:
        for argument, value, id_column in (
            ("samples", samples, "sample_id"),
            ("groups", groups, "group_id"),
            ("features", features, "feature_id"),
        ):
            reject_masked_table(value, argument, id_column)
        # A flag read by its truth value would read a masked one through its mask.
        if not isinstance(copy, bool | np.bool_):
            raise TypeError(f"copy must be a bool, got {type(copy).__name__}")
        sample_table, sample_index = _axis_table(samples, "samples", "sample_id")
        group_table, group_index = _axis_table(groups, "groups", "group_id")
        feature_table, feature_index = _axis_table(features, "features", "feature_id")
        ids = (
            group_table["group_id"],
            sample_table["sample_id"],
            feature_table["feature_id"],
        )
        shape = (len(group_table), len(sample_table), len(feature_table))
        cells = None if n_cells is None else _as_n_cells(n_cells, ids)
        normalization_text = (
            None if normalization is None else _normalization_text(normalization)
        )
        provenance_texts = _provenance_texts(provenance)
        stored = _store_layers(layers, shape, cells, normalization_text, ids, copy)
        # Every check has passed: only now are adopted arrays made read-only.
        for array in stored.values():
            if isinstance(array, np.ndarray):
                array.flags.writeable = False
        _set_state(
            self,
            stored,
            sample_table,
            group_table,
            feature_table,
            cells,
            normalization_text,
            provenance_texts,
            (sample_index, group_index, feature_index),
        )

    # Immutability, copying and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "Cube is immutable; select(), regroup() and collapse_samples() return "
            "new cubes"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Cube is immutable")

    def __copy__(self) -> "Cube":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Cube":
        return self

    def __reduce__(self) -> tuple[Callable[..., "Cube"], tuple[Any, ...]]:
        return (
            _restore_cube,
            (
                self._layers,
                self._samples,
                self._groups,
                self._features,
                self._n_cells,
                self._normalization,
                self._provenance,
                self._digests,
            ),
        )

    # Sizes and metadata.

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return len(self._samples)

    @property
    def n_groups(self) -> int:
        """The number of groups."""
        return len(self._groups)

    @property
    def n_features(self) -> int:
        """The number of features."""
        return len(self._features)

    @property
    def layer_names(self) -> tuple[str, ...]:
        """The names of the layers this cube holds, in ``LAYER_NAMES`` order."""
        return tuple(self._layers)

    @property
    def samples(self) -> Table:
        """The sample table, one row per sample."""
        return self._samples

    @property
    def groups(self) -> Table:
        """The group table, one row per group."""
        return self._groups

    @property
    def features(self) -> Table:
        """The feature table, one row per feature."""
        return self._features

    @property
    def n_cells(self) -> np.ndarray | None:
        """Cells pooled per entry, int64 of shape ``(n_samples, n_groups)``, or None."""
        return None if self._n_cells is None else self._n_cells.view()

    @property
    def normalization(self) -> dict[str, Any] | None:
        """A copy of the normalization mapping, or None."""
        if self._normalization is None:
            return None
        result: dict[str, Any] = json.loads(self._normalization)
        return result

    @property
    def provenance(self) -> dict[str, Any]:
        """A copy of the provenance mapping (see ``cytepool.cube.provenance``)."""
        return _provenance_of(self._provenance)

    def __repr__(self) -> str:
        lines = [
            f"Cube with {self.n_samples:,} samples x {self.n_groups:,} groups x "
            f"{self.n_features:,} features"
        ]
        described = []
        for name, stored in self._layers.items():
            lazy = "" if isinstance(stored, np.ndarray) else ", lazy"
            described.append(f"{name} ({stored.dtype}{lazy})")
        lines.append(f"  layers: {', '.join(described)}")
        if self._n_cells is None:
            lines.append("  cells per entry: not recorded")
        else:
            total = self._n_cells.size
            filled = int(np.count_nonzero(self._n_cells))
            share = f" ({100 * filled / total:.1f}%)" if total else ""
            lines.append(
                f"  filled: {filled:,} of {total:,} sample x group entries have "
                f"cells{share}"
            )
        for label, table in (
            ("samples", self._samples),
            ("groups", self._groups),
            ("features", self._features),
        ):
            lines.append(f"  {label}: {', '.join(table.columns)}")
        return "\n".join(lines)

    # Layers and derived statistics.

    def layer(self, name: str, group: str | int | None = None) -> np.ndarray:
        """Return a layer for one group or for all groups.

        Args:
            name: The layer name.
            group: A group id or position; ``None`` returns every group.

        Returns:
            A read-only array of shape ``(n_samples, n_features)`` for one group, or
            ``(n_groups, n_samples, n_features)``. A lazy layer is read, and its
            values checked, on every call.

        Raises:
            KeyError: If the cube has no such layer, or ``group`` is an unknown id.
            IndexError: If ``group`` is a position out of range.
            TypeError: If ``group`` is neither a string nor an integer.
            ValueError: If a lazy layer gives a value that breaks a constraint.
            OverflowError: If a lazy layer that ``regroup()`` or
                ``collapse_samples()`` derived has a sum that does not fit.
        """
        self._check_layer_name(name)
        if group is None:
            return self._read_all(name)
        return self._read(name, self._group_position(group))

    def mean(self, group: str | int | None = None, scale: str = "log") -> np.ndarray:
        """Return the mean normalized value over the cells of each entry.

        Args:
            group: A group id or position; ``None`` returns every group.
            scale: ``"log"`` for ``log_sum / n_cells`` or ``"linear"`` for
                ``linear_sum / n_cells``.

        Returns:
            float64 of shape ``(n_samples, n_features)`` for one group, or
            ``(n_groups, n_samples, n_features)``; NaN where ``n_cells`` is zero.

        Raises:
            ValueError: If ``scale`` is unknown or the cube lacks the layer it needs.
            KeyError, IndexError, TypeError: For an invalid ``group``, as in
                ``layer()``.
            OverflowError: As in ``layer()``, for a derived lazy layer.
        """
        if scale == "log":
            name = "log_sum"
        elif scale == "linear":
            name = "linear_sum"
        else:
            raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
        self._require(f"mean(scale={scale!r})", (name,))
        return self._per_group(group, lambda g: self._per_cell(name, g))

    def variance(self, group: str | int | None = None) -> np.ndarray:
        """Return the sample variance over the cells of each entry, log scale.

        The variance comes from the raw moments,
        ``(log_sum_sq - log_sum**2 / n) / (n - 1)`` with ``n = n_cells``. Its
        relative rounding error is about ``eps * (1 + mean**2 / variance)``, with
        ``eps = 2.2e-16`` for float64: far below 1e-10 for log-normalized
        expression, but large when the values of an entry are nearly constant and
        far from zero (about 2e-6 at a mean-to-standard-deviation ratio of 1e5).
        For the same reason, the numerator of a constant entry comes out as zero
        give or take rounding (exactly zero when every value is zero); a numerator
        that rounding pushed below zero is returned as a variance of 0. A
        numerator below ``-1e-6 * log_sum_sq`` cannot come from rounding, so it
        raises.

        Args:
            group: A group id or position; ``None`` returns every group.

        Returns:
            float64 of shape ``(n_samples, n_features)`` for one group, or
            ``(n_groups, n_samples, n_features)``; NaN where ``n_cells < 2``.

        Raises:
            ValueError: If the cube lacks ``log_sum`` or ``log_sum_sq``, or the two
                layers break the moment constraint (they do not sum over the same
                cells).
            KeyError, IndexError, TypeError: For an invalid ``group``, as in
                ``layer()``.
            OverflowError: As in ``layer()``, for a derived lazy layer.
        """
        self._require("variance()", ("log_sum", "log_sum_sq"))
        return self._per_group(group, self._variance_of_group)

    def fraction_detected(self, group: str | int | None = None) -> np.ndarray:
        """Return the fraction of cells with a positive count, ``detected / n_cells``.

        Args:
            group: A group id or position; ``None`` returns every group.

        Returns:
            float64 of shape ``(n_samples, n_features)`` for one group, or
            ``(n_groups, n_samples, n_features)``; NaN where ``n_cells`` is zero.

        Raises:
            ValueError: If the cube lacks the ``detected`` layer.
            KeyError, IndexError, TypeError: For an invalid ``group``, as in
                ``layer()``.
            OverflowError: As in ``layer()``, for a derived lazy layer.
        """
        self._require("fraction_detected()", ("detected",))
        return self._per_group(group, lambda g: self._per_cell("detected", g))

    def library_sizes(self, group: str | int | None = None) -> np.ndarray:
        """Return the counts of each entry summed over features.

        Args:
            group: A group id or position; ``None`` returns every group.

        Returns:
            Shape ``(n_samples,)`` for one group, or ``(n_samples, n_groups)``;
            int64, and exact, for integer counts and float64 otherwise.

        Raises:
            OverflowError: If a library size exceeds 2**63 - 1 (integer counts) or
                the float64 range (float counts), or as in ``layer()`` for a
                derived lazy layer.
            KeyError, IndexError, TypeError: For an invalid ``group``, as in
                ``layer()``.
        """
        if group is not None:
            return self._group_library_sizes(self._group_position(group))
        dtype = np.int64 if self._layers["counts"].dtype.kind in "iu" else np.float64
        result = np.empty((self.n_samples, self.n_groups), dtype=dtype)
        for g in range(self.n_groups):
            result[:, g] = self._group_library_sizes(g)
        return result

    def to_memory(self) -> "Cube":
        """Return a cube with every lazy layer read into memory.

        Values are checked as they are read, the moment constraint included, and so
        are the digests of a cube file's groups. A cube without lazy layers is
        returned as it is. The result has the same values, so the same fingerprint
        and provenance.

        Raises:
            ValueError: If a value breaks a constraint, or a group of a cube file
                does not match its digest.
            OverflowError: If a sum of a derived lazy layer does not fit.
        """
        if all(isinstance(stored, np.ndarray) for stored in self._layers.values()):
            return self
        layers: dict[str, np.ndarray | _LazyLayer] = {
            name: self._read_all(name) for name in self._layers
        }
        _check_moment_layers(layers, self._n_cells, self._ids())
        # Whole reads check or keep the digests of lazy layers, so the copy knows them.
        digests = _Digests(
            dict(self._digests.parts),
            {name: self._known_layer_digests(name) for name in self._layers},
        )
        return _assemble(
            layers,
            self._samples,
            self._groups,
            self._features,
            self._n_cells,
            self._normalization,
            self._provenance,
            digests,
        )

    # Fingerprint and provenance.

    def fingerprint(self) -> str:
        """Return the fingerprint of the cube's values.

        The fingerprint identifies what an analysis reads: the sample, group and
        feature tables (ids included), ``n_cells``, the normalization and every
        layer. It leaves out the provenance, so cubes with the same values have the
        same fingerprint however they were made, and a result can name the cube it
        was computed from. It is ``"sha256:"`` followed by 64 hexadecimal digits;
        ``cytepool.cube.fingerprints`` documents how it is computed, so another
        program can compute it too.

        Cost:
            The fingerprint is built from a digest of each table, of ``n_cells``,
            of the normalization and of each group of each layer, and the cube keeps
            each digest once it is known.

            - A cube in memory hashes its values once.
            - A cube read from a cube file takes the digests stored in the file and
              reads no layer; the tables, ``n_cells`` and the normalization were
              checked against theirs when the file was opened, and each group is
              checked whenever a read covers all of it. A file written by an
              earlier version, without digests, is read whole once, a group at a
              time.
            - A lazy cube derived from a cube file with ``select()``, ``regroup()``
              or ``collapse_samples()`` reads, once, the groups whose values the
              operation changed and that were not read whole before, in every
              layer: none for a selection of groups only, the merged groups for
              ``regroup()``, and every group for ``collapse_samples()`` and for a
              selection of samples or features. Groups it keeps unchanged take
              the digests of the cube it was derived from.
            - A lazy layer of another array-like, which can change, is read on every
              call.

        Raises:
            ValueError: If a lazy layer gives a value that breaks a constraint, or a
                group of a cube file does not match its digest.
            OverflowError: If a sum of a derived lazy layer does not fit.
        """
        record = self._digest_record(read=True)
        assert record is not None  # every digest is computed when read is True
        fingerprint: str = record["fingerprint"]
        return fingerprint

    def digests(self) -> dict[str, Any]:
        """Return the digests of the cube's parts and the fingerprint they make.

        An analysis that reads some groups can record their digests, and check
        them later against a cube file, which checks a group whenever a read covers
        all of it. The digests are computed as ``fingerprint()`` computes them,
        at the same cost.

        Returns:
            A new JSON-serialisable dict in the form cube files store, which
            ``cytepool.cube.fingerprints`` documents. It holds the fingerprint
            version and the cube's shape; the digest of each table, of ``n_cells``
            and of the normalization (None without them); for each layer, its type
            and the digest of each group, by position; and the fingerprint.

        Raises:
            ValueError, OverflowError: As ``fingerprint()``.
        """
        record = self._digest_record(read=True)
        assert record is not None  # every digest is computed when read is True
        return record

    def verify_fingerprint(self) -> str:
        """Compute the fingerprint from the values, checking every digest known.

        Every group of every layer is read whole and hashed, and so are the tables,
        ``n_cells`` and the normalization. A lazy layer derived from another is
        computed from whole reads of the groups it comes from, so that what a
        selection of some samples or features reads is checked too. Each digest
        must equal the one the cube holds: stored with the cube file it was read
        from, or computed before. Use it to check that a cube file still holds what
        was written, and that a cube is the one a result was computed from:
        ``read_cube(path).verify_fingerprint() == fingerprint_the_result_recorded``.

        Returns:
            The fingerprint, computed from the values.

        Raises:
            ValueError: If a part does not match its digest (the message names it),
                or a lazy layer gives a value that breaks a constraint.
            OverflowError: If a sum of a derived lazy layer does not fit.
        """
        known = self._part_digests()
        parts = self._part_digests(fresh=True)
        for part, digest in parts.items():
            if known[part] != digest:
                raise ValueError(
                    f"{_PART_NAMES[part]} of this cube does not match its digest "
                    f"{known[part]}: it was changed after the digest was computed"
                )
        layers = {}
        for name, stored in self._layers.items():
            digests = []
            for group in range(self.n_groups):
                if isinstance(stored, np.ndarray):
                    digest = _block_digest(stored[group])
                    kept = self._digests.layers.setdefault(name, [None] * self.n_groups)
                    if kept[group] not in (None, digest):
                        raise ValueError(
                            f"group {self._groups['group_id'][group]!r} of the layer "
                            f"{name!r} does not match its digest {kept[group]}: the "
                            "array was changed after the digest was computed"
                        )
                    kept[group] = digest
                else:
                    # Every value is read with its whole group, and so checked.
                    block = stored.read_checked(group)
                    found = stored.known_digest(group)
                    digest = _block_digest(block) if found is None else found
                digests.append(digest)
            layers[name] = (_layer_type(stored.dtype), digests)
        shape = (self.n_groups, self.n_samples, self.n_features)
        fingerprint: str = _digest_record(shape, parts, layers)["fingerprint"]
        return fingerprint

    def history(self) -> str:
        """Describe how the cube was made, in plain terms, for printing.

        The text is ``describe_provenance(cube.provenance)`` (see
        ``cytepool.cube.provenance``): the caller's source record, how the cube was
        built, and each operation in order, with the fingerprint of the cube it was
        applied to. A last line gives the cube's own fingerprint when it is known
        without reading lazy layers (see ``fingerprint()``).
        """
        lines = [describe_provenance(self.provenance)]
        fingerprint = self._known_fingerprint()
        if fingerprint is None:
            lines.append(
                "Fingerprint: not known without reading the lazy layers; "
                "fingerprint() reads them."
            )
        else:
            lines.append(f"Fingerprint: {fingerprint}")
        return "\n".join(lines)

    # Deriving cubes.

    def select(
        self, samples: Any = None, groups: Any = None, features: Any = None
    ) -> "Cube":
        """Return a cube with a subset of samples, groups and features.

        Ids and positions also set the order of the result. A single id or position
        selects one entry. The provenance record lists the kept sample, group and
        feature ids, in order, and null for an axis given as None, and names this
        cube by its fingerprint (see "Fingerprint and provenance" in ``Cube``).
        Selecting from a lazy cube reads nothing. Reads of the result ask its layers only for
        the kept groups, samples and features: a cube file reads only the chunks
        that hold them, and a layer without ``oindex`` is read one whole kept group
        at a time (see "Lazy layers" in ``Cube``).

        Args:
            samples: Sample ids, a boolean mask or integer positions; ``None`` keeps
                every sample.
            groups: Group ids, a boolean mask or integer positions; ``None`` keeps
                every group.
            features: Feature ids, a boolean mask or integer positions; ``None``
                keeps every feature.

        Raises:
            KeyError: If an id is unknown.
            IndexError: If a position is out of range.
            TypeError: If a selection is not ids, a boolean mask or integers, or is
                a numpy masked array.
            ValueError: If a mask has the wrong length or a selection repeats an
                entry.
        """
        sample_positions = _positions(
            samples, self._samples["sample_id"], self._sample_index, "samples"
        )
        group_positions = _positions(
            groups, self._groups["group_id"], self._group_index, "groups"
        )
        feature_positions = _positions(
            features, self._features["feature_id"], self._feature_index, "features"
        )
        selections = (sample_positions, group_positions, feature_positions)
        if all(positions is None for positions in selections):
            return self
        kept_groups = (
            np.arange(self.n_groups) if group_positions is None else group_positions
        )
        sample_table = _take_rows(self._samples, sample_positions)
        group_table = _take_rows(self._groups, group_positions)
        feature_table = _take_rows(self._features, feature_positions)
        n_cells = None
        if self._n_cells is not None:
            n_cells = self._n_cells
            if sample_positions is not None:
                n_cells = n_cells[sample_positions]
            n_cells = np.ascontiguousarray(n_cells[:, kept_groups])
        record = {
            "operation": "select",
            "samples": _kept_ids(sample_table, "sample_id", sample_positions),
            "groups": _kept_ids(group_table, "group_id", group_positions),
            "features": _kept_ids(feature_table, "feature_id", feature_positions),
        }
        return self._derive(
            sample_table,
            group_table,
            feature_table,
            n_cells,
            record,
            group_sets=[kept_groups[i : i + 1] for i in range(kept_groups.size)],
            sample_order=sample_positions,
            feature_positions=feature_positions,
            sums=False,
        )

    def regroup(self, mapping: Mapping[str, Sequence[str] | str]) -> "Cube":
        """Return a cube with groups merged.

        Every layer and ``n_cells`` of a merged group is the sum over its members.
        Groups that are not listed are kept as they are. A merged group takes the
        position of its member that comes first in the current group order. In the
        group table, a merged row keeps a column's value where all members agree
        and is missing otherwise; an integer or boolean column that needs a missing
        value becomes float64 with NaN. Integer layers of the result are int64. The
        provenance record gives each merged or renamed group with its members, in
        the current group order, and names this cube by its fingerprint (see
        "Fingerprint and provenance" in ``Cube``).

        Args:
            mapping: New group id to the existing group ids it merges. A group id may
                appear in one merge only, and a new id must not be the id of a group
                that is kept. A single member renames a group.

        Raises:
            KeyError: If a member is not a group id.
            TypeError: If ``mapping`` is not a mapping of strings to strings, or
                holds a numpy masked array.
            ValueError: If a new id is empty, a merge is empty, a group is listed
                twice or a new id collides with a kept group.
            OverflowError: If a merged ``n_cells`` or a merged value of an
                in-memory layer does not fit (see "Sums" in ``Cube``). A lazy
                layer raises it when the merged group is read.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(
                "regroup() needs a mapping of new group id to existing group ids, got "
                f"{type(mapping).__name__}"
            )
        reject_masked(mapping, "mapping")
        ids = self._groups["group_id"].tolist()
        owner: dict[int, str] = {}
        merges: dict[str, list[int]] = {}
        for new_id, members in mapping.items():
            if not isinstance(new_id, str):
                raise TypeError(
                    f"new group ids must be strings, got {type(new_id).__name__} "
                    f"{new_id!r}"
                )
            if not new_id:
                raise ValueError("new group ids must be non-empty")
            if isinstance(members, str):
                listed = [members]
            else:
                try:
                    listed = list(members)
                except TypeError:
                    raise TypeError(
                        f"the merge into {new_id!r} must list group ids, got "
                        f"{type(members).__name__}"
                    ) from None
            if not listed:
                raise ValueError(f"the merge into {new_id!r} lists no groups")
            positions = []
            for member in listed:
                if not isinstance(member, str):
                    raise TypeError(
                        f"the merge into {new_id!r} must list group ids (str), got "
                        f"{type(member).__name__} {member!r}"
                    )
                position = self._group_index.get(member)
                if position is None:
                    raise KeyError(
                        f"the merge into {new_id!r} lists {member!r}, which is not a "
                        "group id"
                    )
                if position in owner:
                    where = (
                        f"twice in the merge into {new_id!r}"
                        if owner[position] == new_id
                        else f"in the merges into {owner[position]!r} and {new_id!r}"
                    )
                    raise ValueError(
                        f"group {member!r} is listed {where}; a group can be merged once"
                    )
                owner[position] = new_id
                positions.append(position)
            merges[new_id] = sorted(positions)
        kept = [position for position in range(self.n_groups) if position not in owner]
        kept_ids = {ids[position] for position in kept}
        for new_id in merges:
            if new_id in kept_ids:
                raise ValueError(
                    f"the new group id {new_id!r} is already the id of a group that is "
                    "not merged"
                )
        outputs = [(position, [position], ids[position]) for position in kept]
        outputs += [(members[0], members, new_id) for new_id, members in merges.items()]
        outputs.sort(key=lambda output: output[0])
        group_sets = [np.array(members, dtype=np.intp) for _, members, _ in outputs]
        order, starts = _runs(group_sets)
        new_ids = [new_id for _, _, new_id in outputs]
        group_table = _pool_rows(
            self._groups, "group_id", new_ids, order, starts, keep_varying=True
        )
        n_cells = None
        if self._n_cells is not None:
            sample_ids = self._samples["sample_id"]
            n_cells = _sum_runs(
                self._n_cells,
                order,
                starts,
                1,
                lambda index: (
                    f"n_cells of the merged group {new_ids[index[1]]!r} at sample "
                    f"{sample_ids[index[0]]!r}"
                ),
            )
        membership = {
            new_id: [ids[position] for position in members]
            for _, members, new_id in outputs
            if new_id in merges
        }
        record = {"operation": "regroup", "membership": membership}
        return self._derive(
            self._samples,
            group_table,
            self._features,
            n_cells,
            record,
            group_sets=group_sets,
            sums=True,
        )

    def collapse_samples(self, by: str) -> "Cube":
        """Return a cube with the samples that share a value of a column summed.

        Use it to pool, for example, the tissues of one donor. The new ``sample_id``
        is the shared value, in the order the values first appear. Sample columns
        that are constant within every pooled set are kept and the others are
        dropped. The column ``n_pooled_samples`` holds the number of original
        samples behind each new sample and comes last; if it already exists, it
        keeps its place and is summed, so collapsing twice counts the original
        samples. Integer layers of the result are int64. The provenance record gives
        each new sample with the ids of the samples it sums, in the current order,
        and names this cube by its fingerprint (see "Fingerprint and provenance" in
        ``Cube``).

        Args:
            by: A text or integer sample column without missing values.

        Raises:
            KeyError: If there is no such sample column.
            TypeError: If the column is not text or integer.
            ValueError: If the column has a missing or empty value, or an existing
                ``n_pooled_samples`` column does not hold non-negative integers.
            OverflowError: If a pooled ``n_cells``, ``n_pooled_samples`` or value
                of an in-memory layer does not fit (see "Sums" in ``Cube``). A lazy
                layer raises it when the group is read.
        """
        if not isinstance(by, str):
            raise TypeError(f"by must be a sample column name, got {type(by).__name__}")
        if by not in self._samples:
            raise KeyError(
                f"no sample column named {by!r}; columns are: "
                f"{', '.join(self._samples.columns)}"
            )
        column = self._samples[by]
        if column.size == 0:  # no samples: any dtype will do, as for [] (float64)
            keys: list[str] = []
        elif column.dtype.kind == "O":
            keys = column.tolist()
            sample_ids = self._samples["sample_id"]
            for position, key in enumerate(keys):
                if key is None or key == "":
                    raise ValueError(
                        f"sample column {by!r} is missing or empty for sample "
                        f"{sample_ids[position]!r}; every sample needs a value"
                    )
        elif column.dtype.kind in "iu":
            keys = [str(key) for key in column.tolist()]
        else:
            raise TypeError(
                "collapse_samples() pools by a text or integer sample column; "
                f"{by!r} has dtype {column.dtype}"
            )
        members: dict[str, list[int]] = {}
        for position, key in enumerate(keys):
            members.setdefault(key, []).append(position)
        sample_sets = [
            np.array(positions, dtype=np.intp) for positions in members.values()
        ]
        order, starts = _runs(sample_sets)
        new_ids = list(members)
        others = self._samples
        if POOLED_SAMPLES_COLUMN in self._samples:
            pooled = self._samples[POOLED_SAMPLES_COLUMN]
            if pooled.size and pooled.dtype.kind not in "iu":
                raise ValueError(
                    f"the sample column {POOLED_SAMPLES_COLUMN!r} counts pooled samples "
                    f"and must hold integers, got dtype {pooled.dtype}"
                )
            if pooled.size and pooled.min() < 0:
                position = int(np.argmax(pooled < 0))
                raise ValueError(
                    f"the sample column {POOLED_SAMPLES_COLUMN!r} counts pooled samples "
                    f"and must be non-negative, but sample "
                    f"{self._samples['sample_id'][position]!r} has {pooled[position]}"
                )
            counts = _sum_runs(
                pooled,
                order,
                starts,
                0,
                lambda index: (
                    f"the sample column {POOLED_SAMPLES_COLUMN!r} of the pooled sample "
                    f"{new_ids[index[0]]!r}"
                ),
            ).astype(np.int64, copy=False)  # a column without rows may be float64
            others = others.select(
                [name for name in others.columns if name != POOLED_SAMPLES_COLUMN]
            )
        else:
            counts = np.array([positions.size for positions in sample_sets], np.int64)
        kept = _pool_rows(
            others, "sample_id", new_ids, order, starts, keep_varying=False
        )
        names = [
            name
            for name in self._samples.columns
            if name in kept or name == POOLED_SAMPLES_COLUMN
        ]
        if POOLED_SAMPLES_COLUMN not in names:
            names.append(POOLED_SAMPLES_COLUMN)
        sample_table = Table(
            {
                name: counts if name == POOLED_SAMPLES_COLUMN else kept[name]
                for name in names
            }
        )
        n_cells = None
        if self._n_cells is not None:
            group_ids = self._groups["group_id"]
            n_cells = _sum_runs(
                self._n_cells,
                order,
                starts,
                0,
                lambda index: (
                    f"n_cells of the pooled sample {new_ids[index[0]]!r} at group "
                    f"{group_ids[index[1]]!r}"
                ),
            )
        original_ids = self._samples["sample_id"]
        record = {
            "operation": "collapse_samples",
            "by": by,
            "membership": {
                new_id: [original_ids[position] for position in positions]
                for new_id, positions in members.items()
            },
        }
        return self._derive(
            sample_table,
            self._groups,
            self._features,
            n_cells,
            record,
            group_sets=[np.array([g], dtype=np.intp) for g in range(self.n_groups)],
            sample_order=order,
            sample_starts=starts,
            sums=True,
        )

    # Internals.

    def _derive(
        self,
        samples: Table,
        groups: Table,
        features: Table,
        n_cells: np.ndarray | None,
        record: dict[str, Any],
        *,
        group_sets: list[np.ndarray],
        sample_order: np.ndarray | None = None,
        sample_starts: np.ndarray | None = None,
        feature_positions: np.ndarray | None = None,
        sums: bool,
    ) -> "Cube":
        """Return the cube that a ``_Plan`` of these parts derives from this one.

        ``record`` describes the operation; its input, this cube's fingerprint, is
        added when it is known without reading lazy layers. Groups that the plan
        leaves unchanged keep the digests this cube knows.
        """
        operation = {"operation": record.pop("operation")}
        operation["input"] = self._known_fingerprint()
        operation |= record
        plan = _Plan(
            group_sets,
            sample_order,
            sample_starts,
            feature_positions,
            source=self._ids(),
            target=(groups["group_id"], samples["sample_id"], features["feature_id"]),
        )
        shape = (len(groups), len(samples), len(features))
        layers: dict[str, np.ndarray | _LazyLayer] = {}
        digests = _Digests()
        for name, stored in self._layers.items():
            dtype = stored.dtype
            if sums and dtype.kind in "iu":
                dtype = np.dtype(np.int64)
            if isinstance(stored, np.ndarray):
                layers[name] = plan.apply(name, stored, shape, dtype)
                kept = self._known_layer_digests(name)
                digests.layers[name] = [
                    None if (source := plan.same_block(group)) is None else kept[source]
                    for group in range(shape[0])
                ]
            else:
                layers[name] = _DerivedLayer(stored, plan, name, dtype, shape)
        if n_cells is not None:
            n_cells.flags.writeable = False
        return _assemble(
            layers,
            samples,
            groups,
            features,
            n_cells,
            self._normalization,
            _add_record(self._provenance, operation),
            digests,
        )

    def _part_digests(self, fresh: bool = False) -> dict[str, str | None]:
        """Return the digests of the tables, n_cells and the normalization.

        They are computed once and kept, unless ``fresh`` asks for them computed
        anew (and not kept).
        """
        parts = {} if fresh else self._digests.parts
        if len(parts) < len(_PART_NAMES):
            for axis, table in (
                ("samples", self._samples),
                ("groups", self._groups),
                ("features", self._features),
            ):
                if axis not in parts:
                    parts[axis] = _table_digest(table)
            if "n_cells" not in parts:
                parts["n_cells"] = _cells_digest(self._n_cells)
            if "normalization" not in parts:
                parts["normalization"] = _normalization_digest(self.normalization)
        return parts

    def _known_layer_digests(self, name: str) -> list[str | None]:
        """Return the digest of each group of a layer, None where not known yet."""
        stored = self._layers[name]
        if isinstance(stored, np.ndarray):
            return list(self._digests.layers.get(name, [None] * self.n_groups))
        return [stored.known_digest(group) for group in range(self.n_groups)]

    def _layer_digest(self, name: str, group: int, read: bool) -> str | None:
        """Return the digest of one group of a layer.

        An in-memory group is hashed once. A lazy group's digest is taken from the
        layer, and otherwise computed from a whole read when ``read`` is true, and
        None when it is false.
        """
        stored = self._layers[name]
        if isinstance(stored, np.ndarray):
            kept = self._digests.layers.setdefault(name, [None] * self.n_groups)
            if kept[group] is None:
                kept[group] = _block_digest(stored[group])
            return kept[group]
        found = stored.known_digest(group)
        if found is not None or not read:
            return found
        block = stored.read(group)  # a whole read keeps the digest of a stable layer
        found = stored.known_digest(group)
        return _block_digest(block) if found is None else found

    def _digest_record(self, read: bool) -> dict[str, Any] | None:
        """Return the digest record, or None if ``read`` is false and a lazy group's
        digest is not known."""
        if not read:
            for stored in self._layers.values():
                if not isinstance(stored, np.ndarray) and not all(
                    stored.known_digest(group) is not None
                    for group in range(self.n_groups)
                ):
                    return None
        layers = {}
        for name, stored in self._layers.items():
            digests = []
            for group in range(self.n_groups):
                digest = self._layer_digest(name, group, read)
                assert digest is not None  # computed, or known as checked above
                digests.append(digest)
            layers[name] = (_layer_type(stored.dtype), digests)
        shape = (self.n_groups, self.n_samples, self.n_features)
        return _digest_record(shape, self._part_digests(), layers)

    def _known_fingerprint(self) -> str | None:
        """Return the fingerprint if it needs no read of a lazy layer, else None."""
        record = self._digest_record(read=False)
        return None if record is None else record["fingerprint"]

    def _check_layer_name(self, name: str) -> None:
        if name in self._layers:
            return
        if name in LAYER_NAMES:
            raise KeyError(
                f"this cube has no {name!r} layer; it has: {', '.join(self._layers)}"
            )
        raise KeyError(
            f"unknown layer {name!r}; layers are named: {', '.join(LAYER_NAMES)}"
        )

    def _require(self, statistic: str, names: tuple[str, ...]) -> None:
        missing = [name for name in names if name not in self._layers]
        if missing:
            named = " and ".join(map(repr, missing))
            plural = "s" if len(missing) > 1 else ""
            raise ValueError(
                f"{statistic} needs the {named} layer{plural}, which this cube does not "
                f"have; it has: {', '.join(self._layers)}"
            )

    def _group_library_sizes(self, group: int) -> np.ndarray:
        block = self._read("counts", group)
        group_id = self._groups["group_id"][group]
        sample_ids = self._samples["sample_id"]
        return _checked_sum(
            block,
            block.shape[1],
            lambda values, dtype: values.sum(axis=1, dtype=dtype),
            lambda index: (
                f"layer 'counts' over the features of group {group_id!r}, sample "
                f"{sample_ids[index[0]]!r}"
            ),
        )

    def _cells(self) -> np.ndarray:
        if self._n_cells is None:  # every layer that needs cells requires n_cells
            raise ValueError("this cube does not record n_cells")
        return self._n_cells

    def _read(self, name: str, group: int) -> np.ndarray:
        stored = self._layers[name]
        if isinstance(stored, np.ndarray):
            block: np.ndarray = stored[group]
            return block
        return stored.read(group)

    def _read_all(self, name: str) -> np.ndarray:
        stored = self._layers[name]
        if isinstance(stored, np.ndarray):
            return stored.view()
        return stored.read_all()

    def _per_group(
        self, group: str | int | None, compute: Callable[[int], np.ndarray]
    ) -> np.ndarray:
        if group is not None:
            return compute(self._group_position(group))
        result = np.empty((self.n_groups, self.n_samples, self.n_features))
        for g in range(self.n_groups):
            result[g] = compute(g)
        return result

    def _per_cell(self, name: str, group: int) -> np.ndarray:
        cells = self._cells()[:, group]
        with np.errstate(divide="ignore", invalid="ignore"):
            result: np.ndarray = self._read(name, group) / cells[:, None]
        result[cells == 0] = np.nan
        return result

    def _variance_of_group(self, group: int) -> np.ndarray:
        cells = self._cells()[:, group]
        squares = self._read("log_sum_sq", group)
        numerator = _central_sum_of_squares(
            self._read("log_sum", group), squares, cells
        )
        _check_moments(numerator, squares, group, self._ids())
        with np.errstate(divide="ignore", invalid="ignore"):
            result = numerator / (cells.astype(np.float64)[:, None] - 1.0)
        result[cells < 2] = np.nan
        np.maximum(result, 0.0, out=result)  # rounding below zero gives 0; NaN stays
        return result

    def _ids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._groups["group_id"],
            self._samples["sample_id"],
            self._features["feature_id"],
        )

    def _group_position(self, group: object) -> int:
        # operator.index accepts a 0-d masked integer and ignores its mask.
        reject_masked(group, "group")
        if isinstance(group, str):
            position = self._group_index.get(group)
            if position is None:
                raise KeyError(
                    f"no group with id {group!r}; groups are: "
                    f"{_preview(self._groups['group_id'])}"
                )
            return position
        if isinstance(group, bool | np.bool_):
            raise TypeError(
                "group must be a group id (str) or a position (int), not bool"
            )
        try:
            position = operator.index(group)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError(
                "group must be a group id (str) or a position (int), got "
                f"{type(group).__name__}"
            ) from None
        if not -self.n_groups <= position < self.n_groups:
            raise IndexError(
                f"group position {position} is out of range for {self.n_groups} groups"
            )
        return position % self.n_groups


class _LazyLayer:
    """A layer read one group, or one block of a group, at a time.

    ``digests`` holds the digest of each group once it is known (None until then),
    or is None for a layer whose values can change, such as one read from a memmap,
    whose digests are never kept. A read that covers a whole group of a layer that
    keeps digests hashes the group: it keeps the digest, or checks it against the
    one kept.
    """

    __slots__ = ("dtype", "shape", "digests")

    dtype: np.dtype
    shape: tuple[int, int, int]
    digests: list[str | None] | None

    def known_digest(self, group: int) -> str | None:
        """Return the digest of a group if it is known, else None."""
        return None if self.digests is None else self.digests[group]

    def _keep_digest(
        self,
        group: int,
        block: np.ndarray,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ) -> None:
        """Keep the digest of a group just read whole, or check it against the one
        kept.

        ``rows`` and ``columns`` say where each sample and feature of the group is
        in ``block`` when it holds them in another order (see ``_block_digest``).
        """
        if self.digests is None:
            return
        digest = _block_digest(block, rows, columns)
        kept = self.digests[group]
        if kept is None:
            self.digests[group] = digest
        elif digest != kept:
            raise self._changed(group)

    def _changed(self, group: int) -> ValueError:
        """The error for a group that does not match the digest kept for it."""
        raise NotImplementedError

    def read_checked(self, group: int) -> np.ndarray:
        """Return one group, every value it is computed from read with its group
        whole, so that each is checked against the digest kept for it."""
        raise NotImplementedError

    def read(
        self,
        group: int,
        samples: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one group, or its block at some sample and feature positions.

        ``samples`` and ``features`` are 1-D intp arrays of positions in any order,
        or None for every position. The block is read-only, of shape
        ``(len(samples), len(features))`` with None counting the whole axis.
        """
        raise NotImplementedError

    def read_all(self) -> np.ndarray:
        """Return every group, read-only, of shape ``shape``."""
        raise NotImplementedError


class _StoredLayer(_LazyLayer):
    """A lazy layer from a user's array-like.

    Every read puts what it needs in memory the cube owns, so a source that changes
    later (a memmap, say) cannot change what was read and checked. What a source
    returns is copied, except what a ``ZarrArray`` returns: a new array that nothing
    else holds (``reads_new_arrays``).

    A ``ZarrArray`` layer keeps digests (see ``_LazyLayer``): those stored with its
    cube file (``stored_digests``), which every whole read of a group checks, or
    those of its first whole reads. Another source can change, so its digests are
    not kept.
    """

    __slots__ = ("source", "name", "n_cells", "ids", "reads_new_arrays", "stored")

    def __init__(
        self,
        source: Any,
        name: str,
        dtype: np.dtype,
        shape: tuple[int, int, int],
        n_cells: np.ndarray | None,
        ids: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.source = source
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.n_cells = n_cells
        self.ids = ids
        self.reads_new_arrays = _reads_new_arrays(source)
        self.digests = [None] * shape[0] if self.reads_new_arrays else None
        self.stored = False  # whether the digests are those stored with a cube file

    def stored_digests(self, digests: list[str]) -> None:
        """Take the digests stored with the cube file this layer is read from."""
        assert self.reads_new_arrays and len(digests) == self.shape[0]
        self.digests = list(digests)
        self.stored = True

    def read(
        self,
        group: int,
        samples: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        if samples is None and features is None:
            block = self._owned(self._fetch(group), self.reads_new_arrays)
        else:
            block = self._fetch_block(group, samples, features)
        _check_values(
            self.name, block, group, self.n_cells, self.ids, samples, features
        )
        # A read of every sample and feature, in any order, is checked as a whole.
        every_sample, rows = _inverse(samples, self.shape[1])
        every_feature, columns = _inverse(features, self.shape[2])
        if every_sample and every_feature:
            self._keep_digest(group, block, rows, columns)
        block.flags.writeable = False
        return block

    def read_checked(self, group: int) -> np.ndarray:
        return self.read(group)

    def read_all(self) -> np.ndarray:
        if self.reads_new_arrays:  # one read, into the array it returns
            whole = unmasked_array(self.source[...], f"the lazy layer {self.name!r}")
            array = self._owned(whole, new=True)
            for group in range(self.shape[0]):
                _check_values(self.name, array[group], group, self.n_cells, self.ids)
                self._keep_digest(group, array[group])
        else:
            array = np.empty(self.shape, dtype=self.dtype)
            for group in range(self.shape[0]):
                array[group] = self._fetch(group)
                _check_values(self.name, array[group], group, self.n_cells, self.ids)
        array.flags.writeable = False
        return array

    def _changed(self, group: int) -> ValueError:
        what = f"group {self.ids[0][group]!r} of the layer {self.name!r}"
        where = getattr(self.source, "path", "its source")
        if self.stored:
            return ValueError(
                f"{what}, read from the array at {where}, does not match the digest "
                "stored with its cube file: the file was changed after it was "
                "written. If the change was meant, delete the root attribute "
                "'digests' of the cube file (with zarr-python, for example) and open "
                "it again"
            )
        return ValueError(
            f"{what}, read from the array at {where}, differs from when it was first "
            "read: the cube file changed while it was open; open it again with "
            "read_cube"
        )

    def _fetch(self, group: int) -> np.ndarray:
        """Read one group from the source and check its shape and dtype."""
        return self._checked(self.source[group], group, self.shape[1:])

    def _fetch_block(
        self, group: int, samples: np.ndarray | None, features: np.ndarray | None
    ) -> np.ndarray:
        """Read some samples and features of one group into a new array.

        A ``ZarrArray`` is asked for the positions in the order given. Another source
        with ``oindex`` is asked for the distinct positions in increasing order, and
        the block is then put in the order asked for. Any other source gives the whole
        group, and the block is taken from it.
        """
        rows = self.shape[1] if samples is None else samples.size
        columns = self.shape[2] if features is None else features.size
        if not rows or not columns:  # nothing to read
            return np.empty((rows, columns), dtype=self.dtype)
        outer = getattr(self.source, "oindex", None)
        if outer is None:
            return self._owned(_take(self._fetch(group), samples, features), new=True)
        if self.reads_new_arrays:
            key = (group, _every_if_none(samples), _every_if_none(features))
            block = self._checked(outer[key], group, (rows, columns), " with oindex")
            return self._owned(block, new=True)
        sample_key, sample_order = _increasing(samples)
        feature_key, feature_order = _increasing(features)
        block = self._checked(
            outer[group, sample_key, feature_key],
            group,
            (_length(sample_key, rows), _length(feature_key, columns)),
            " with oindex",
        )
        if sample_order is None and feature_order is None:
            return self._owned(block, new=False)
        return self._owned(_take(block, sample_order, feature_order), new=True)

    def _owned(self, block: np.ndarray, new: bool) -> np.ndarray:
        """Return a read as a C-contiguous array of the layer's dtype, the cube's own.

        A ``new`` block, which nothing else holds, is kept as it is if it already is
        such an array and owns its memory; anything else is copied. Only an array that
        owns its memory keeps the read-only flag the cube sets on it for every view of
        it, and numpy gives a view even for a dtype that only equals the array's.
        """
        if (
            new
            and block.flags.owndata
            and block.flags.c_contiguous
            and block.dtype == self.dtype
        ):
            return block
        return np.array(block, dtype=self.dtype, order="C")  # a copy

    def _checked(
        self,
        value: Any,
        group: int,
        shape: tuple[int, ...],
        how: str = "",
    ) -> np.ndarray:
        """Return what a read of a group gave as an array, after checking it."""
        what = f"group {self.ids[0][group]!r} of the lazy layer {self.name!r}{how}"
        block = unmasked_array(value, what)
        if block.shape != shape:
            raise ValueError(
                f"reading {what} gave shape {block.shape}, expected {shape}"
            )
        if block.dtype.newbyteorder("=") != self.dtype:
            raise TypeError(
                f"reading {what} gave dtype {block.dtype}, but the layer declares "
                f"{self.dtype}"
            )
        return block


class _DerivedLayer(_LazyLayer):
    """A lazy layer computed group by group from the layer of another cube.

    A read of some samples and features reads only what they need from the parent.
    It keeps digests when its parent does: a group that the plan leaves unchanged
    has the digest of its parent's group, and the digest of any other group is
    computed when the group is read whole.
    """

    __slots__ = ("parent", "plan", "name")

    def __init__(
        self,
        parent: _LazyLayer,
        plan: "_Plan",
        name: str,
        dtype: np.dtype,
        shape: tuple[int, int, int],
    ) -> None:
        self.parent = parent
        self.plan = plan
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.digests = None if parent.digests is None else [None] * shape[0]

    def known_digest(self, group: int) -> str | None:
        if self.digests is None:
            return None
        if self.digests[group] is None:
            source = self.plan.same_block(group)
            if source is not None:
                self.digests[group] = self.parent.known_digest(source)
        return self.digests[group]

    def read(
        self,
        group: int,
        samples: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        block = self.plan.block(
            self.name, self.parent.read, group, self.dtype, samples, features
        )
        if samples is None and features is None:
            self._keep_whole(group, block)
        block.flags.writeable = False
        return block

    def read_all(self) -> np.ndarray:
        if self.shape[0] == 1:  # the one block, without copying it into a new array
            return self.read(0)[np.newaxis]
        result = np.empty(self.shape, dtype=self.dtype)
        for group in range(self.shape[0]):
            block = self.plan.block(self.name, self.parent.read, group, self.dtype)
            self._keep_whole(group, block)
            result[group] = block
        result.flags.writeable = False
        return result

    def read_checked(self, group: int) -> np.ndarray:
        def read(
            source: int, samples: np.ndarray | None, features: np.ndarray | None
        ) -> np.ndarray:
            return _take(self.parent.read_checked(source), samples, features)

        block = self.plan.block(self.name, read, group, self.dtype)
        self._keep_whole(group, block)
        block.flags.writeable = False
        return block

    def _keep_whole(self, group: int, block: np.ndarray) -> None:
        """Keep or check the digest of a group just computed whole."""
        if self.digests is None:
            return
        if self.plan.same_block(group) is not None and self.known_digest(group):
            return  # the parent's whole read of the group checked its digest
        self._keep_digest(group, block)

    def _changed(self, group: int) -> ValueError:
        return ValueError(
            f"group {self.plan.target[0][group]!r} of the derived layer "
            f"{self.name!r} differs from when it was first read: the cube file it "
            "was derived from changed while it was open; open it again with "
            "read_cube and derive the cube anew"
        )


class _Plan:
    """How each group of a derived layer is computed from a source layer.

    Output group ``h`` is the sum of the source groups ``group_sets[h]``. Its samples
    are the source samples at ``sample_order`` (when ``sample_starts`` is None) or
    the sums of the runs of ``sample_order`` that begin at ``sample_starts``, and its
    features are the source features at ``feature_positions``. ``None`` keeps an axis
    as it is. ``source`` and ``target`` hold the group, sample and feature ids of the
    source and of the result, for error messages. Integer sums are exact or raise
    OverflowError, and so do float sums beyond the float64 range: ``_merge`` sums
    groups, and ``_pool`` the runs of samples.

    A block of an output group needs only some samples and features of the source
    groups, so ``block`` reads only those, and sums only them. Positions that keep
    an axis as it is, and runs of one sample each in place, are stored as None, so
    that an output group equal to a source group (``same_block``) is read whole.
    """

    __slots__ = (
        "group_sets",
        "sample_order",
        "sample_starts",
        "feature_positions",
        "source",
        "target",
    )

    def __init__(
        self,
        group_sets: list[np.ndarray],
        sample_order: np.ndarray | None,
        sample_starts: np.ndarray | None,
        feature_positions: np.ndarray | None,
        *,
        source: tuple[np.ndarray, np.ndarray, np.ndarray],
        target: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        n_samples, n_features = source[1].size, source[2].size
        if _is_identity(sample_order, n_samples) and (
            sample_starts is None or sample_starts.size == n_samples
        ):  # every sample in place, alone in its run if samples are pooled
            sample_order = sample_starts = None
        if _is_identity(feature_positions, n_features):
            feature_positions = None
        self.group_sets = group_sets
        self.sample_order = sample_order
        self.sample_starts = sample_starts
        self.feature_positions = feature_positions
        self.source = source
        self.target = target

    def same_block(self, group: int) -> int | None:
        """Return the source group whose block output group ``group`` is, unchanged.

        That is so when the output group has one member and every sample and
        feature stays in place; otherwise None.
        """
        members = self.group_sets[group]
        if members.size != 1 or not (
            self.sample_order is None
            and self.sample_starts is None
            and self.feature_positions is None
        ):
            return None
        return int(members[0])

    def block(
        self,
        name: str,
        read: _BlockReader,
        group: int,
        dtype: np.dtype,
        samples: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute output group ``group`` of layer ``name``, or a block of it.

        The block holds the sample and feature positions given (None for every
        position); ``read(g, samples, features)`` reads a block of source group ``g``.
        """
        source_features = _compose(self.feature_positions, features)
        if self.sample_starts is None:
            source_samples = _compose(self.sample_order, samples)
            block = self._merge(name, read, group, source_samples, source_features)
        else:
            source_samples, starts = self._runs_of(samples)
            block = self._merge(name, read, group, source_samples, source_features)
            block = self._pool(name, block, group, samples, starts, source_features)
        return block.astype(dtype, copy=False)

    def apply(
        self, name: str, array: np.ndarray, shape: tuple[int, int, int], dtype: np.dtype
    ) -> np.ndarray:
        """Compute every output group of an in-memory layer."""

        def read(
            group: int, samples: np.ndarray | None, features: np.ndarray | None
        ) -> np.ndarray:
            return _take(array[group], samples, features)

        result = np.empty(shape, dtype=dtype)
        for group in range(shape[0]):
            result[group] = self.block(name, read, group, dtype)
        result.flags.writeable = False
        return result

    def _runs_of(self, samples: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Return the runs of source samples that pooled samples sum, and their starts.

        ``samples`` are positions of pooled samples, None for all of them.
        """
        assert self.sample_order is not None and self.sample_starts is not None
        if samples is None:
            return self.sample_order, self.sample_starts
        ends = np.append(self.sample_starts[1:], self.sample_order.size)
        return _runs(
            [
                self.sample_order[self.sample_starts[pooled] : ends[pooled]]
                for pooled in samples.tolist()
            ]
        )

    def _pool(
        self,
        name: str,
        block: np.ndarray,
        group: int,
        samples: np.ndarray | None,
        starts: np.ndarray,
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Sum the runs of rows of ``block`` that begin at ``starts``.

        ``samples`` are the pooled samples the runs give and ``features`` the source
        features of the columns (None for all), for error messages.
        """
        if not starts.size:  # no samples to pool
            return block
        group_id, pooled_ids = self.target[0][group], self.target[1]
        feature_ids = self.source[2]
        longest = int(np.diff(np.append(starts, block.shape[0])).max())
        return _checked_sum(
            block,
            longest,
            lambda values, dtype: np.add.reduceat(values, starts, axis=0, dtype=dtype),
            lambda index: (
                f"layer {name!r} of the pooled sample "
                f"{pooled_ids[_at(samples, index[0])]!r} at group {group_id!r}, "
                f"feature {feature_ids[_at(features, index[1])]!r}"
            ),
        )

    def _merge(
        self,
        name: str,
        read: _BlockReader,
        group: int,
        samples: np.ndarray | None,
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Sum the source groups of output group ``group``, one member at a time.

        Only the source sample and feature positions given are read (None for all).
        Integers accumulate in int64 while the sum of the members' largest possible
        values fits, which proves that every sum does, and as Python integers once
        it does not. Every member is read once, and besides the sum only one is held
        at a time.
        """
        members = self.group_sets[group]
        first = read(int(members[0]), samples, features)
        if members.size == 1:
            return first
        merged_id = self.target[0][group]
        sample_ids, feature_ids = self.source[1], self.source[2]

        def subject(index: tuple[int, ...]) -> str:
            return (
                f"layer {name!r} of the merged group {merged_id!r} at sample "
                f"{sample_ids[_at(samples, index[0])]!r}, feature "
                f"{feature_ids[_at(features, index[1])]!r}"
            )

        if first.dtype.kind == "f":
            total = first.astype(np.float64)  # a new array to accumulate into
            del first  # from here on, the sum and one member are held
            for member in members[1:]:
                term = read(int(member), samples, features)
                try:
                    with np.errstate(over="raise"):
                        total += term
                except FloatingPointError:
                    # numpy writes the sums before it raises.
                    index = _first(~np.isfinite(total))
                    raise _float_overflow(subject(index)) from None
                del term  # before the next member is read
            return total
        total = first.astype(np.int64)  # values are at most 2**63 - 1: exact
        bound = _largest_possible(first)
        del first
        for member in members[1:]:
            term = read(int(member), samples, features)
            bound += _largest_possible(term)
            if bound > _INT64_MAX and total.dtype != object:
                total = total.astype(object)
            if total.dtype == object:
                # As Python integers: a Python integer plus a numpy integer would be
                # a numpy integer, which wraps.
                total += term.astype(object)
            else:
                # Values are at most 2**63 - 1, so uint64 ones read as int64 unchanged;
                # numpy would add uint64 to int64 in float64.
                if term.dtype == np.uint64:
                    term = term.view(np.int64)
                np.add(total, term, out=total)
            del term  # before the next member is read
        if total.dtype == object:
            too_large = total > _INT64_MAX
            if too_large.any():
                index = _first(too_large)
                raise _integer_overflow(subject(index), total[index])
            total = total.astype(np.int64)
        return total


class _Digests:
    """The digests of a cube's parts known so far: computed, or stored with a file.

    ``parts`` maps "samples", "groups", "features", "n_cells" and "normalization"
    to its digest (None for a cube without n_cells or normalization) once it is
    known. ``layers`` maps an in-memory layer to the digest of each group, None
    where not computed yet; a lazy layer keeps its own (``_LazyLayer.digests``).
    The values of a cube never change, so what is known stays true.
    """

    __slots__ = ("parts", "layers")

    def __init__(
        self,
        parts: dict[str, str | None] | None = None,
        layers: dict[str, list[str | None]] | None = None,
    ) -> None:
        self.parts = {} if parts is None else parts
        self.layers = {} if layers is None else layers

    def __reduce__(self) -> tuple[type["_Digests"], tuple[Any, ...]]:
        return (_Digests, (self.parts, self.layers))


def _set_state(
    cube: Cube,
    layers: dict[str, np.ndarray | _LazyLayer],
    samples: Table,
    groups: Table,
    features: Table,
    n_cells: np.ndarray | None,
    normalization: str | None,
    provenance: tuple[str, ...],
    indexes: tuple[dict[str, int], dict[str, int], dict[str, int]] | None = None,
    digests: _Digests | None = None,
) -> None:
    if indexes is None:
        indexes = (
            _index_of(samples["sample_id"]),
            _index_of(groups["group_id"]),
            _index_of(features["feature_id"]),
        )
    state = {
        "_layers": layers,
        "_samples": samples,
        "_groups": groups,
        "_features": features,
        "_n_cells": n_cells,
        "_normalization": normalization,
        "_provenance": provenance,
        "_sample_index": indexes[0],
        "_group_index": indexes[1],
        "_feature_index": indexes[2],
        "_digests": _Digests() if digests is None else digests,
    }
    for name, value in state.items():
        object.__setattr__(cube, name, value)


def _assemble(
    layers: dict[str, np.ndarray | _LazyLayer],
    samples: Table,
    groups: Table,
    features: Table,
    n_cells: np.ndarray | None,
    normalization: str | None,
    provenance: tuple[str, ...],
    digests: _Digests | None = None,
) -> Cube:
    """Build a cube from parts that are already checked, without copying them."""
    cube = object.__new__(Cube)
    _set_state(
        cube,
        layers,
        samples,
        groups,
        features,
        n_cells,
        normalization,
        provenance,
        digests=digests,
    )
    return cube


def _restore_cube(
    layers: dict[str, np.ndarray | _LazyLayer],
    samples: Table,
    groups: Table,
    features: Table,
    n_cells: np.ndarray | None,
    normalization: str | None,
    provenance: tuple[str, ...] | str,
    digests: _Digests | None = None,
) -> Cube:
    # A cube pickled before digests were kept has none, and one JSON text.
    if isinstance(provenance, str):
        provenance = _provenance_texts(json.loads(provenance))
    # Unpickled arrays are new and writable.
    for stored in layers.values():
        if isinstance(stored, np.ndarray):
            stored.flags.writeable = False
    if n_cells is not None:
        n_cells.flags.writeable = False
    return _assemble(
        layers, samples, groups, features, n_cells, normalization, provenance, digests
    )


# Digests of cube files, for cytepool.io.cube_files.


def _take_stored_digests(cube: Cube, record: Any, where: str) -> None:
    """Give a cube just read from a cube file the digests stored with the file.

    The tables, ``n_cells`` and the normalization, which are in memory, are checked
    now; each group of a layer is checked whenever a read covers all of it.

    Raises:
        ValueError: If the record is malformed, or a part does not match its digest.
    """
    shape = (cube.n_groups, cube.n_samples, cube.n_features)
    types = {name: _layer_type(layer.dtype) for name, layer in cube._layers.items()}
    try:
        _check_record(record, shape, types, where)
    except ValueError as error:
        error.add_note(
            "(to open the file without the digests stored with it, delete its root "
            "attribute 'digests', with zarr-python for example)"
        )
        raise
    for part, digest in cube._part_digests(fresh=True).items():
        if record[part] != digest:
            raise ValueError(
                f"{_PART_NAMES[part]} does not match the digest stored in {where}: "
                "the cube file was changed after it was written. If the change was "
                "meant, delete the root attribute 'digests' (with zarr-python, for "
                "example) and open the file again"
            )
    cube._digests.parts.update({part: record[part] for part in _PART_NAMES})
    for name, layer in cube._layers.items():
        assert isinstance(layer, _StoredLayer)  # read_cube's layers are ZarrArrays
        layer.stored_digests(record["layers"][name]["groups"])


def _digest_of_read_group(cube: Cube, name: str, group: int, block: np.ndarray) -> str:
    """Return the digest of a group of a layer that was just read whole as ``block``.

    A digest the cube knows is taken (a lazy layer keeps the digest of a whole
    read); otherwise the block is hashed, and the digest kept for an in-memory
    layer.
    """
    layer = cube._layers[name]
    found = (
        cube._digests.layers.get(name, [None] * cube.n_groups)[group]
        if isinstance(layer, np.ndarray)
        else layer.known_digest(group)
    )
    if found is not None:
        return found
    digest = _block_digest(block)
    if isinstance(layer, np.ndarray):
        cube._digests.layers.setdefault(name, [None] * cube.n_groups)[group] = digest
    return digest


def _digest_record_of(cube: Cube, layers: Mapping[str, list[str]]) -> dict[str, Any]:
    """Return a cube's digest record, given the digests of each group of each layer."""
    shape = (cube.n_groups, cube.n_samples, cube.n_features)
    typed = {
        name: (_layer_type(cube._layers[name].dtype), layers[name])
        for name in cube.layer_names
    }
    return _digest_record(shape, cube._part_digests(), typed)


def _axis_table(
    value: object, axis: str, id_column: str
) -> tuple[Table, dict[str, int]]:
    try:
        table = value if isinstance(value, Table) else Table(value)
    except (TypeError, ValueError) as error:
        error.add_note(f"(while reading the {axis} table)")
        raise
    if id_column not in table:
        raise ValueError(
            f"the {axis} table needs a {id_column!r} column; it has: "
            f"{', '.join(table.columns) or '(none)'}"
        )
    if len(table) == 0:
        # An empty column cannot hold a wrong value, but numpy reads [] as float64.
        text_columns = (id_column, *_OPTIONAL_TEXT_COLUMNS.get(axis, ()))
        empty: dict[str, np.ndarray] = {
            name: np.empty(0, dtype=object) for name in text_columns if name in table
        }
        for name in _OPTIONAL_BOOLEAN_COLUMNS.get(axis, ()):
            if name in table:
                empty[name] = np.empty(0, dtype=bool)
        table = table.with_columns(empty)
    ids = table[id_column]
    if ids.dtype != object:
        raise TypeError(
            f"{axis}[{id_column!r}] must hold text ids, got dtype {ids.dtype}; convert "
            "the ids to strings first"
        )
    index: dict[str, int] = {}
    for position, identifier in enumerate(ids.tolist()):
        if identifier is None or identifier == "":
            raise ValueError(
                f"{axis}[{id_column!r}] has a missing or empty id at row {position}"
            )
        if identifier in index:
            raise ValueError(
                f"{axis}[{id_column!r}] repeats the id {identifier!r}; ids must be "
                "unique"
            )
        index[identifier] = position
    for column in _OPTIONAL_TEXT_COLUMNS.get(axis, ()):
        if column in table and table[column].dtype != object:
            raise TypeError(
                f"{axis}[{column!r}] must hold text, got dtype {table[column].dtype}"
            )
    for column in _OPTIONAL_BOOLEAN_COLUMNS.get(axis, ()):
        if column in table and table[column].dtype != np.bool_:
            raise TypeError(
                f"{axis}[{column!r}] must be boolean, got dtype {table[column].dtype}"
            )
    for column in table.columns:  # the types a fingerprint and a cube file encode
        if table[column].dtype.kind != "O":
            _column_type(table[column].dtype, f"{axis}[{column!r}]")
    return table, index


def _index_of(ids: np.ndarray) -> dict[str, int]:
    return {value: position for position, value in enumerate(ids.tolist())}


def _as_n_cells(
    value: Any, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    group_ids, sample_ids, _ = ids
    if all(hasattr(value, attribute) for attribute in ("index", "columns", "to_numpy")):
        value = _aligned_by_labels(value, sample_ids, group_ids)
    array = unmasked_array(value, "n_cells")
    if array.dtype.kind not in "iu":
        raise TypeError(f"n_cells must be an integer array, got dtype {array.dtype}")
    expected = (sample_ids.size, group_ids.size)
    if array.shape != expected:
        raise ValueError(
            f"n_cells must have shape (n_samples, n_groups) = {expected}, got "
            f"{array.shape}"
        )
    # Only uint64 (in either byte order) holds values above 2**63 - 1.
    if array.dtype.kind == "u" and array.dtype.itemsize == 8 and array.size:
        too_large = array > _INT64_MAX
        if too_large.any():
            sample, group = _first(too_large)
            raise ValueError(
                f"n_cells is above 2**63 - 1 ({array[sample, group]}) for sample "
                f"{sample_ids[sample]!r} and group {group_ids[group]!r}; cell numbers "
                "must be at most 2**63 - 1, the largest int64"
            )
    cells = array.astype(np.int64)
    negative = np.argwhere(cells < 0)
    if negative.size:
        sample, group = negative[0]
        raise ValueError(
            f"n_cells is negative ({cells[sample, group]}) for sample "
            f"{sample_ids[sample]!r} and group {group_ids[group]!r}"
        )
    cells.flags.writeable = False
    return cells


def _aligned_by_labels(
    frame: Any, sample_ids: np.ndarray, group_ids: np.ndarray
) -> np.ndarray:
    """Reorder a DataFrame of cell numbers to the cube's sample and group order."""
    positions = []
    for axis, labels, ids in (
        ("rows", list(frame.index), sample_ids.tolist()),
        ("columns", list(frame.columns), group_ids.tolist()),
    ):
        where = {label: i for i, label in enumerate(labels)}
        known = set(ids)
        missing = [identifier for identifier in ids if identifier not in where]
        extra = [label for label in labels if label not in known]
        if missing or extra or len(where) != len(labels):
            kind = "sample" if axis == "rows" else "group"
            raise ValueError(
                f"the {axis} of an n_cells DataFrame must be the {kind} ids, each once; "
                f"missing: {missing[:5]}, unexpected: {extra[:5]}; pass a numpy array "
                "to give cell numbers by position"
            )
        positions.append([where[identifier] for identifier in ids])
    values = np.asarray(frame.to_numpy())
    if values.dtype == object:  # for example pandas' nullable integer columns
        if hasattr(frame, "isna") and np.asarray(frame.isna().to_numpy()).any():
            raise ValueError(
                "an n_cells DataFrame has missing values; every entry needs a number "
                "of cells"
            )
        # Keep a conversion only when it is exact: 1.5 would become 1, and pandas
        # turns a UInt64 count of 2**63 into -2**63 in int64. uint64 keeps such a
        # count for the caller to report; otherwise the caller reports the dtype.
        overflowed = False
        for dtype in (np.int64, np.uint64):
            try:
                integers = np.asarray(frame.to_numpy(dtype=dtype))
            except OverflowError:
                overflowed = True
                continue
            except (TypeError, ValueError):
                continue
            if bool(np.all(integers == values)):
                values = integers
                break
        else:
            if overflowed:
                raise ValueError(
                    "an n_cells DataFrame holds a number of cells outside the int64 "
                    "range; cell numbers must be from 0 to 2**63 - 1"
                )
    aligned: np.ndarray = values[np.ix_(*positions)]
    return aligned


def _store_layers(
    layers: Mapping[str, Any],
    shape: tuple[int, int, int],
    n_cells: np.ndarray | None,
    normalization: str | None,
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
    copy: bool,
) -> dict[str, np.ndarray | _LazyLayer]:
    if not isinstance(layers, Mapping):
        raise TypeError(
            f"layers must be a mapping of layer name to array, got {type(layers).__name__}"
        )
    reject_masked(layers, "layers")
    unknown = [name for name in layers if name not in LAYER_NAMES]
    if unknown:
        raise ValueError(
            f"unknown layer {unknown[0]!r}; layers are named: {', '.join(LAYER_NAMES)}"
        )
    if "counts" not in layers:
        raise ValueError("the 'counts' layer is required")
    needs_cells = [name for name in _CELL_LAYERS if name in layers]
    if needs_cells and n_cells is None:
        raise ValueError(
            f"n_cells is required with the {', '.join(map(repr, needs_cells))} layer"
        )
    needs_normalization = [name for name in _NORMALIZED_LAYERS if name in layers]
    if needs_normalization and normalization is None:
        raise ValueError(
            "normalization is required with the "
            f"{', '.join(map(repr, needs_normalization))} layer, to record how the "
            "normalized values were computed"
        )
    stored = {
        name: _store_layer(name, layers[name], shape, n_cells, ids, copy)
        for name in LAYER_NAMES
        if name in layers
    }
    _check_moment_layers(stored, n_cells, ids)
    return stored


def _store_layer(
    name: str,
    value: Any,
    shape: tuple[int, int, int],
    n_cells: np.ndarray | None,
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
    copy: bool,
) -> np.ndarray | _LazyLayer:
    lazy = isinstance(value, np.memmap) or (
        not isinstance(value, np.ndarray)
        and all(
            hasattr(value, attribute) for attribute in ("shape", "dtype", "__getitem__")
        )
    )
    if lazy:
        dtype = _layer_dtype(name, value.dtype)
        if tuple(value.shape) != shape:
            raise ValueError(_shape_message(name, shape, tuple(value.shape)))
        return _StoredLayer(value, name, dtype, shape, n_cells, ids)
    array = unmasked_array(value, f"layers[{name!r}]")
    dtype = _layer_dtype(name, array.dtype)
    if array.shape != shape:
        raise ValueError(_shape_message(name, shape, array.shape))
    adopt = (
        not copy
        and type(value) is np.ndarray
        and value.dtype == dtype
        and value.flags.c_contiguous
    )
    if not adopt:
        array = np.array(array, dtype=dtype, order="C", copy=True)
    for group in range(shape[0]):
        _check_values(name, array[group], group, n_cells, ids)
    return array


def _layer_dtype(name: str, dtype: Any) -> np.dtype:
    try:
        native: np.dtype = np.dtype(dtype).newbyteorder("=")
    except TypeError as error:
        raise TypeError(
            f"layer {name!r} has dtype {dtype!r}, which numpy cannot interpret"
        ) from error
    if name == "counts":
        ok, expected = (
            native.kind in "iu" or native == np.float64,
            "an integer or float64",
        )
    elif name == "detected":
        ok, expected = native.kind in "iu", "an integer"
    else:
        ok, expected = native == np.float64, "float64"
    if not ok:
        hint = ""
        if name in _NORMALIZED_LAYERS and native.kind == "f":
            hint = "; lower precision loses digits the variance from raw moments needs"
        raise TypeError(f"layer {name!r} must have {expected} dtype, got {dtype}{hint}")
    return native


def _shape_message(
    name: str, expected: tuple[int, ...], actual: tuple[int, ...]
) -> str:
    return (
        f"layer {name!r} must have shape (n_groups, n_samples, n_features) = {expected}, "
        f"got {actual}"
    )


def _check_values(
    name: str,
    block: np.ndarray,
    group: int,
    n_cells: np.ndarray | None,
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
    samples: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> None:
    """Check the values of one group of a layer, or of a block of it.

    ``block`` has shape ``(n_samples, n_features)``, or holds the sample and feature
    positions given (None for every position). Each check is a single reduction; the
    offending entry is located only when a check fails.
    """
    if block.size == 0:
        return

    def where(sample: int, feature: int) -> str:
        return _where(ids, group, _at(samples, sample), _at(features, feature))

    if block.dtype.kind == "f":
        # Finite values can sum to infinity, or to infinity minus infinity.
        with np.errstate(over="ignore", invalid="ignore"):
            total = block.sum()
        if not np.isfinite(total):
            bad = ~np.isfinite(block)  # the sum may also overflow with finite values
            if bad.any():
                sample, feature = _first(bad)
                raise ValueError(
                    f"layer {name!r} has a non-finite value ({block[sample, feature]}) "
                    f"at {where(sample, feature)}"
                )
    # Only uint64 holds values above 2**63 - 1, and no unsigned value is negative.
    if block.dtype.kind == "u" and block.dtype.itemsize == 8:
        if block.max() > _INT64_MAX:
            sample, feature = _first(block > _INT64_MAX)
            raise ValueError(
                f"layer {name!r} has a value above 2**63 - 1 ({block[sample, feature]}) "
                f"at {where(sample, feature)}; integer layers must be at most "
                "2**63 - 1, the largest int64"
            )
    elif name in _NON_NEGATIVE_LAYERS and block.dtype.kind != "u" and block.min() < 0:
        sample, feature = _first(block < 0)
        raise ValueError(
            f"layer {name!r} has a negative value ({block[sample, feature]}) at "
            f"{where(sample, feature)}; it must be non-negative"
        )
    if n_cells is None:
        return
    cells = n_cells[:, group] if samples is None else n_cells[samples, group]
    if name == "detected":
        over = np.flatnonzero(block.max(axis=1) > cells)
        if over.size:
            sample = int(over[0])
            feature = int(np.argmax(block[sample] > cells[sample]))
            raise ValueError(
                f"layer 'detected' exceeds n_cells at {where(sample, feature)}: "
                f"{block[sample, feature]} cells detected but {cells[sample]} pooled"
            )
    empty = np.flatnonzero(cells == 0)
    if empty.size and block[empty].any():
        row, feature = _first(block[empty] != 0)
        sample = int(empty[row])
        raise ValueError(
            f"layer {name!r} is {block[sample, feature]} at {where(sample, feature)}, "
            "where n_cells is 0; an entry without cells must be zero in every layer"
        )


def _first(mask: np.ndarray) -> tuple[int, ...]:
    """Return the index of the first entry, in C order, where a mask is true."""
    return tuple(int(position) for position in np.argwhere(mask)[0])


def _where(
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
    group: int,
    sample: int,
    feature: int,
) -> str:
    return (
        f"group {ids[0][group]!r}, sample {ids[1][sample]!r}, feature "
        f"{ids[2][feature]!r}"
    )


def _central_sum_of_squares(
    total: np.ndarray, squares: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Return ``log_sum_sq - log_sum**2 / n_cells`` for one group; 0 without cells."""
    n = cells.astype(np.float64)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = squares - total * (total / n)
    result[cells == 0] = 0.0
    return result


def _check_moments(
    numerator: np.ndarray,
    squares: np.ndarray,
    group: int,
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Raise when ``log_sum**2 / n_cells`` exceeds ``log_sum_sq`` beyond rounding.

    Both moments of an entry sum over the same cells, so by Cauchy-Schwarz the
    numerator of the variance is never negative. Rounding in float64 moves it by
    at most a few times ``n * eps * log_sum_sq``; ``_MOMENT_TOLERANCE`` is orders of
    magnitude above that for any realistic number of cells.
    """
    bad = numerator < -_MOMENT_TOLERANCE * squares
    if bad.any():
        sample, feature = _first(bad)
        square = squares[sample, feature]
        raise ValueError(
            "layers 'log_sum' and 'log_sum_sq' disagree at "
            f"{_where(ids, group, sample, feature)}: log_sum**2 / n_cells = "
            f"{square - numerator[sample, feature]:.6g} exceeds log_sum_sq = "
            f"{square:.6g} by more than rounding allows; both must be float64 sums "
            "over the same cells"
        )


def _check_moment_layers(
    layers: Mapping[str, np.ndarray | _LazyLayer],
    n_cells: np.ndarray | None,
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Check the moment constraint for in-memory moment layers."""
    total, squares = layers.get("log_sum"), layers.get("log_sum_sq")
    if not isinstance(total, np.ndarray) or not isinstance(squares, np.ndarray):
        return
    if n_cells is None:  # the moment layers require n_cells
        return
    for group in range(total.shape[0]):
        cells = n_cells[:, group]
        numerator = _central_sum_of_squares(total[group], squares[group], cells)
        _check_moments(numerator, squares[group], group, ids)


def _positions(
    selector: Any, ids: np.ndarray, index: dict[str, int], axis: str
) -> np.ndarray | None:
    """Convert ids, a boolean mask or positions along one axis to positions."""
    if selector is None:
        return None
    n = len(index)
    single = axis[:-1]
    if isinstance(selector, str) or (
        isinstance(selector, int | np.integer)
        and not isinstance(selector, bool | np.bool_)
    ):
        selector = [selector]
    values = unmasked_array(selector, axis)
    if values.ndim != 1:
        raise ValueError(f"{axis} must be a 1-D selection, got shape {values.shape}")
    kind = values.dtype.kind
    if kind == "b":
        if values.size != n:
            raise ValueError(
                f"a boolean mask of {axis} needs {n} values, got {values.size}"
            )
        return np.flatnonzero(values)
    if values.size == 0:
        return np.empty(0, dtype=np.intp)
    if kind in "iu":
        outside = (values < -n) | (values >= n)
        if outside.any():
            raise IndexError(
                f"{single} position {values[outside][0]} is out of range for {n} {axis}"
            )
        positions = values.astype(np.intp) % n
    elif kind in "UTO":
        positions = np.empty(values.size, dtype=np.intp)
        for i, value in enumerate(values.tolist()):
            if not isinstance(value, str):
                raise TypeError(
                    f"{axis} must be ids, a boolean mask or integer positions; got "
                    f"{type(value).__name__} {value!r} among ids"
                )
            position = index.get(value)
            if position is None:
                raise KeyError(f"no {single} with id {value!r}")
            positions[i] = position
    else:
        raise TypeError(
            f"{axis} must be ids, a boolean mask or integer positions, got dtype "
            f"{values.dtype}"
        )
    seen: set[int] = set()
    for position in positions.tolist():
        if position in seen:
            raise ValueError(
                f"the selection of {axis} repeats {ids[position]!r}; each {single} can "
                "be selected once"
            )
        seen.add(position)
    return positions


def _take_rows(table: Table, positions: np.ndarray | None) -> Table:
    return table if positions is None else table.take(positions)


def _take(
    block: np.ndarray, samples: np.ndarray | None, features: np.ndarray | None
) -> np.ndarray:
    """Return the rows ``samples`` and columns ``features`` of a block (None: all).

    With a selection, the result is a new C-contiguous array.
    """
    taken: np.ndarray
    if features is None:
        taken = block if samples is None else block[samples]
    elif samples is None:
        # numpy lays out block[:, features] in Fortran order; take gives C order.
        taken = np.take(block, features, axis=1)
    else:
        taken = block[np.ix_(samples, features)]
    return taken


def _compose(
    positions: np.ndarray | None, selected: np.ndarray | None
) -> np.ndarray | None:
    """Return the source positions of the ``selected`` positions of a derived axis.

    The derived axis holds the source positions ``positions``, None for the same
    positions as the source.
    """
    if positions is None:
        return selected
    return positions if selected is None else positions[selected]


def _inverse(positions: np.ndarray | None, n: int) -> tuple[bool, np.ndarray | None]:
    """Whether positions take each of ``n`` positions once, and where each one is.

    Returns ``(True, None)`` for None (every position, in place), ``(True, where)``
    for positions in another order, where ``where[p]`` is the index of position
    ``p``, and ``(False, None)`` for positions that leave some out.
    """
    if positions is None:
        return True, None
    if positions.size != n:
        return False, None
    where = np.full(n, -1, dtype=np.intp)
    where[positions] = np.arange(n)
    if (where < 0).any():  # a repeated position leaves another out
        return False, None
    return True, None if _is_identity(positions, n) else where


def _is_identity(positions: np.ndarray | None, n: int) -> bool:
    """Whether positions take each of ``n`` positions once, in order."""
    return (
        positions is not None
        and positions.size == n
        and bool((positions == np.arange(n)).all())
    )


def _at(positions: np.ndarray | None, index: int) -> int:
    """Return the position that entry ``index`` of a block stands for."""
    return index if positions is None else int(positions[index])


def _increasing(
    positions: np.ndarray | None,
) -> tuple[slice | np.ndarray, np.ndarray | None]:
    """Return the distinct positions in increasing order, and how to reorder them.

    For None, every position, the first value is ``slice(None)``. The second gives
    where to take each of ``positions`` from the distinct ones, or is None when they
    are already in place.
    """
    if positions is None:
        return slice(None), None
    distinct, inverse = np.unique(positions, return_inverse=True)
    if distinct.size == positions.size and np.array_equal(distinct, positions):
        return distinct, None
    return distinct, inverse


def _length(key: slice | np.ndarray, whole: int) -> int:
    return whole if isinstance(key, slice) else key.size


def _every_if_none(positions: np.ndarray | None) -> slice | np.ndarray:
    return slice(None) if positions is None else positions


def _reads_new_arrays(source: Any) -> bool:
    """Whether every read of a lazy layer's source returns a new array of its own.

    The cube can keep such an array without a copy. This is true of ``ZarrArray``,
    the reader of cube files, whose reads are new arrays in native byte order that
    own their memory.
    """
    from ..io.zarr import ZarrArray  # imported here: cytepool.io imports this module

    return type(source) is ZarrArray


def _kept_ids(
    table: Table, column: str, positions: np.ndarray | None
) -> list[str] | None:
    return None if positions is None else table[column].tolist()


def _runs(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate index sets, returning the positions and where each set starts."""
    if not sets:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    sizes = np.array([members.size for members in sets])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
    return np.concatenate(sets).astype(np.intp), starts


def _sum_runs(
    array: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    axis: int,
    subject: Callable[[tuple[int, ...]], str],
) -> np.ndarray:
    """Sum the runs of ``order`` that begin at ``starts`` along an axis.

    Integers, which must be non-negative, are summed in int64 and floats in float64,
    as ``_checked_sum`` describes; ``subject`` names an entry of the result.
    """
    taken = np.take(array, order, axis=axis)
    if starts.size == 0:
        return taken
    longest = int(np.diff(np.append(starts, order.size)).max())
    return _checked_sum(
        taken,
        longest,
        lambda values, dtype: np.add.reduceat(values, starts, axis=axis, dtype=dtype),
        subject,
    )


def _checked_sum(
    values: np.ndarray,
    n_terms: int,
    reduce: Callable[[np.ndarray, Any], np.ndarray],
    subject: Callable[[tuple[int, ...]], str],
) -> np.ndarray:
    """Return ``reduce(values, dtype)``: sums of at most ``n_terms`` values each.

    Floats are summed in float64. Integers, which must be non-negative, are summed
    in int64, exactly: directly when their largest possible value times
    ``n_terms`` is at most 2**63 - 1, which proves that every sum fits, and
    otherwise as Python integers, which do not overflow. A sum beyond 2**63 - 1, or
    beyond the float64 range, raises OverflowError for the first such entry of the
    result, described by ``subject(index)``.
    """
    if values.dtype.kind == "f":
        try:
            with np.errstate(over="raise"):
                return reduce(values, np.float64)
        except FloatingPointError:
            with np.errstate(over="ignore", invalid="ignore"):
                sums = reduce(values, np.float64)
            raise _float_overflow(subject(_first(~np.isfinite(sums)))) from None
    if _largest_possible(values) * n_terms <= _INT64_MAX:
        return reduce(values, np.int64)
    sums = reduce(values.astype(object), object)
    too_large = sums > _INT64_MAX
    if too_large.any():
        index = _first(too_large)
        raise _integer_overflow(subject(index), sums[index])
    return sums.astype(np.int64)


def _largest_possible(values: np.ndarray) -> int:
    """Return an upper bound on non-negative integers.

    For a dtype narrower than 64 bits it is the dtype's largest value, which needs no
    pass over the data and proves that a sum of up to 2**31 terms fits in int64. For
    64-bit dtypes it is the bitwise OR of the values, which holds every bit set in
    any of them and so is at least the largest; numpy computes it faster than the
    largest value itself.
    """
    if values.dtype.itemsize < 8:
        return int(np.iinfo(values.dtype).max)
    return int(np.bitwise_or.reduce(values, axis=None))


def _integer_overflow(subject: str, total: int) -> OverflowError:
    return OverflowError(
        f"{subject} sums to {total}, beyond 2**63 - 1, the largest int64"
    )


def _float_overflow(subject: str) -> OverflowError:
    return OverflowError(f"{subject} sums beyond the float64 range (about 1.8e308)")


def _pool_rows(
    table: Table,
    id_column: str,
    new_ids: list[str],
    order: np.ndarray,
    starts: np.ndarray,
    *,
    keep_varying: bool,
) -> Table:
    """Combine the rows of each run into one row with a new id.

    A column keeps a run's value where every row of the run agrees (missing values
    agree with each other). Elsewhere the value becomes missing when
    ``keep_varying`` is true; otherwise the column is dropped.
    """
    first_rows = order[starts]
    sizes = np.diff(np.append(starts, order.size))
    columns: dict[str, np.ndarray] = {}
    for name in table.columns:
        if name == id_column:
            ids = np.empty(len(new_ids), dtype=object)
            ids[:] = new_ids
            columns[name] = ids
            continue
        column = table[name]
        first = column[first_rows]
        if starts.size:
            same = values_equal(column[order], np.repeat(first, sizes))
            agree = np.logical_and.reduceat(same, starts)
        else:
            agree = np.ones(0, dtype=bool)
        if agree.all():
            columns[name] = first
        elif keep_varying:
            columns[name] = with_missing(first, ~agree)
    return Table(columns)


def _preview(ids: np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(ids[:limit].tolist())
    return shown + (f", ... ({ids.size} in all)" if ids.size > limit else "")


def _normalization_text(normalization: Mapping[str, Any]) -> str:
    if not isinstance(normalization, Mapping):
        raise TypeError(
            f"normalization must be a mapping, got {type(normalization).__name__}"
        )
    reject_masked(normalization, "normalization", missing="None")
    plain = _plain_json(normalization, "normalization")
    method = plain.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(
            "normalization needs a 'method' string that names how values were "
            "normalized, for example 'library_size'"
        )
    return json.dumps(plain)
