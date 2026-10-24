"""The pseudobulk cube: additive statistics per sample, group and feature."""

import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..table import Table, _values_equal, _with_missing

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

    Layers (``counts`` is required, the others are optional):
        counts: Sum of raw counts. An integer dtype, or float64 for non-integer
            counts such as transcript-level estimates.
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
        provenance: A JSON-serialisable mapping that says where the data came from.
            Operations that derive a cube append a record of the operation and its
            arguments to the list under ``"operations"``.
        copy: If False, a layer given as a C-contiguous numpy array with an
            accepted native dtype is stored without a copy and made read-only in
            place, so it must not be changed through other references afterwards.
            Other layers are always copied.

    Raises:
        TypeError: If an argument has the wrong type or dtype.
        ValueError: If shapes disagree, a required argument is missing, an id is
            missing or repeated, or a value breaks a constraint below.

    Constraints:
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
        its shape and dtype and reads nothing. Each read copies one group into
        memory and checks it against the constraints, so a bad value raises when it
        is first used; the moment constraint is checked when ``variance()`` or
        ``to_memory()`` reads both moments. Selecting, regrouping and collapsing a
        lazy cube give a lazy cube, which computes its sums, and raises any
        OverflowError, when a group is read; ``to_memory()`` reads every layer.
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
    )

    _layers: dict[str, "np.ndarray | _LazyLayer"]
    _samples: Table
    _groups: Table
    _features: Table
    _n_cells: np.ndarray | None
    _normalization: str | None
    _provenance: str
    _sample_index: dict[str, int]
    _group_index: dict[str, int]
    _feature_index: dict[str, int]

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
        provenance_text = _provenance_text(provenance)
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
            provenance_text,
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
        """A copy of the provenance mapping."""
        result: dict[str, Any] = json.loads(self._provenance)
        return result

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

        Values are checked as they are read, the moment constraint included. A
        cube without lazy layers is returned as it is.

        Raises:
            ValueError: If a value breaks a constraint.
            OverflowError: If a sum of a derived lazy layer does not fit.
        """
        if all(isinstance(stored, np.ndarray) for stored in self._layers.values()):
            return self
        layers: dict[str, np.ndarray | _LazyLayer] = {
            name: self._read_all(name) for name in self._layers
        }
        _check_moment_layers(layers, self._n_cells, self._ids())
        return _assemble(
            layers,
            self._samples,
            self._groups,
            self._features,
            self._n_cells,
            self._normalization,
            self._provenance,
        )

    # Deriving cubes.

    def select(
        self, samples: Any = None, groups: Any = None, features: Any = None
    ) -> "Cube":
        """Return a cube with a subset of samples, groups and features.

        Ids and positions also set the order of the result. A single id or position
        selects one entry. The provenance record lists the kept sample and group ids
        and the number of kept features, since feature lists can be long.

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
            TypeError: If a selection is not ids, a boolean mask or integers.
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
            "n_features": None if feature_positions is None else len(feature_table),
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
        value becomes float64 with NaN. Integer layers of the result are int64.

        Args:
            mapping: New group id to the existing group ids it merges. A group id may
                appear in one merge only, and a new id must not be the id of a group
                that is kept. A single member renames a group.

        Raises:
            KeyError: If a member is not a group id.
            TypeError: If ``mapping`` is not a mapping of strings to strings.
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
        ids = self._groups["group_id"].tolist()
        owner: dict[int, str] = {}
        merges: dict[str, list[int]] = {}
        record_mapping: dict[str, list[str]] = {}
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
            record_mapping[new_id] = listed
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
        record = {"operation": "regroup", "mapping": record_mapping}
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
        samples. Integer layers of the result are int64.

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
        record = {"operation": "collapse_samples", "by": by}
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
        """Return the cube that a ``_Plan`` of these parts derives from this one."""
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
        for name, stored in self._layers.items():
            dtype = stored.dtype
            if sums and dtype.kind in "iu":
                dtype = np.dtype(np.int64)
            if isinstance(stored, np.ndarray):
                layers[name] = plan.apply(name, stored, shape, dtype)
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
            _add_record(self._provenance, record),
        )

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
    """A layer read one group at a time."""

    __slots__ = ("dtype", "shape")

    dtype: np.dtype
    shape: tuple[int, int, int]

    def read(self, group: int) -> np.ndarray:
        """Return one group, read-only, of shape ``shape[1:]``."""
        raise NotImplementedError

    def read_all(self) -> np.ndarray:
        """Return every group, read-only, of shape ``shape``."""
        raise NotImplementedError


class _StoredLayer(_LazyLayer):
    """A lazy layer from a user's array-like.

    Every read copies one group into memory the cube owns, so a source that changes
    later (a memmap, say) cannot change what was read and checked.
    """

    __slots__ = ("source", "name", "n_cells", "ids")

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

    def read(self, group: int) -> np.ndarray:
        block = np.array(self._fetch(group), dtype=self.dtype, order="C")
        _check_values(self.name, block, group, self.n_cells, self.ids)
        block.flags.writeable = False
        return block

    def read_all(self) -> np.ndarray:
        array = np.empty(self.shape, dtype=self.dtype)
        for group in range(self.shape[0]):
            array[group] = self._fetch(group)
            _check_values(self.name, array[group], group, self.n_cells, self.ids)
        array.flags.writeable = False
        return array

    def _fetch(self, group: int) -> np.ndarray:
        """Read one group from the source and check its shape and dtype."""
        block = np.asarray(self.source[group])
        what = f"group {self.ids[0][group]!r} of the lazy layer {self.name!r}"
        if block.shape != self.shape[1:]:
            raise ValueError(
                f"reading {what} gave shape {block.shape}, expected {self.shape[1:]}"
            )
        if block.dtype.newbyteorder("=") != self.dtype:
            raise TypeError(
                f"reading {what} gave dtype {block.dtype}, but the layer declares "
                f"{self.dtype}"
            )
        return block


class _DerivedLayer(_LazyLayer):
    """A lazy layer computed group by group from the layer of another cube."""

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

    def read(self, group: int) -> np.ndarray:
        block = self.plan.block(self.name, self.parent.read, group, self.dtype)
        block.flags.writeable = False
        return block

    def read_all(self) -> np.ndarray:
        result = np.empty(self.shape, dtype=self.dtype)
        for group in range(self.shape[0]):
            result[group] = self.plan.block(
                self.name, self.parent.read, group, self.dtype
            )
        result.flags.writeable = False
        return result


class _Plan:
    """How each group of a derived layer is computed from a source layer.

    Output group ``h`` is the sum of the source groups ``group_sets[h]``. Its samples
    are then taken at ``sample_order`` (when ``sample_starts`` is None) or pooled by
    summing the runs of ``sample_order`` that begin at ``sample_starts``, and its
    features are taken at ``feature_positions``. ``None`` keeps an axis as it is.
    ``source`` and ``target`` hold the group, sample and feature ids of the source
    and of the result, for error messages. Integer sums are exact or raise
    OverflowError, and so do float sums beyond the float64 range: ``_merge`` sums
    groups, and ``_checked_sum`` the runs of samples.
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
        self.group_sets = group_sets
        self.sample_order = sample_order
        self.sample_starts = sample_starts
        self.feature_positions = feature_positions
        self.source = source
        self.target = target

    def block(
        self,
        name: str,
        read: Callable[[int], np.ndarray],
        group: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Compute one output group of layer ``name``; ``read`` reads a source group."""
        block = self._merge(name, read, group)
        if self.sample_order is not None:
            if self.sample_starts is None:
                block = block[self.sample_order]
            else:
                group_id, pooled_ids = self.target[0][group], self.target[1]
                feature_ids = self.source[2]  # features are taken last
                block = _sum_runs(
                    block,
                    self.sample_order,
                    self.sample_starts,
                    0,
                    lambda index: (
                        f"layer {name!r} of the pooled sample {pooled_ids[index[0]]!r} "
                        f"at group {group_id!r}, feature {feature_ids[index[1]]!r}"
                    ),
                )
        if self.feature_positions is not None:
            block = block[:, self.feature_positions]
        return block.astype(dtype, copy=False)

    def apply(
        self, name: str, array: np.ndarray, shape: tuple[int, int, int], dtype: np.dtype
    ) -> np.ndarray:
        """Compute every output group of an in-memory layer."""
        result = np.empty(shape, dtype=dtype)
        for group in range(shape[0]):
            result[group] = self.block(name, array.__getitem__, group, dtype)
        result.flags.writeable = False
        return result

    def _merge(
        self, name: str, read: Callable[[int], np.ndarray], group: int
    ) -> np.ndarray:
        """Sum the source groups of output group ``group``, one at a time.

        Integers accumulate in int64 while the sum of the members' largest possible
        values fits, which proves that every sum does, and as Python integers once
        it does not. Every member is read once, and only one is held at a time.
        """
        members = self.group_sets[group]
        first = read(int(members[0]))
        if members.size == 1:
            return first
        merged_id = self.target[0][group]
        sample_ids, feature_ids = self.source[1], self.source[2]

        def subject(index: tuple[int, ...]) -> str:
            return (
                f"layer {name!r} of the merged group {merged_id!r} at sample "
                f"{sample_ids[index[0]]!r}, feature {feature_ids[index[1]]!r}"
            )

        if first.dtype.kind == "f":
            total = first.astype(np.float64)  # a new array to accumulate into
            for member in members[1:]:
                term = read(int(member))
                try:
                    with np.errstate(over="raise"):
                        total += term
                except FloatingPointError:
                    # numpy writes the sums before it raises.
                    index = _first(~np.isfinite(total))
                    raise _float_overflow(subject(index)) from None
            return total
        total = first.astype(np.int64)  # values are at most 2**63 - 1: exact
        bound = _largest_possible(first)
        for member in members[1:]:
            term = read(int(member))
            bound += _largest_possible(term)
            if bound > _INT64_MAX and total.dtype != object:
                total = total.astype(object)
            # Cast first: numpy refuses to add uint64 into int64 in place, and a
            # Python integer plus a numpy integer is a numpy integer, which wraps.
            total += term.astype(total.dtype, copy=False)
        if total.dtype == object:
            too_large = total > _INT64_MAX
            if too_large.any():
                index = _first(too_large)
                raise _integer_overflow(subject(index), total[index])
            total = total.astype(np.int64)
        return total


def _set_state(
    cube: Cube,
    layers: dict[str, np.ndarray | _LazyLayer],
    samples: Table,
    groups: Table,
    features: Table,
    n_cells: np.ndarray | None,
    normalization: str | None,
    provenance: str,
    indexes: tuple[dict[str, int], dict[str, int], dict[str, int]] | None = None,
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
    provenance: str,
) -> Cube:
    """Build a cube from parts that are already checked, without copying them."""
    cube = object.__new__(Cube)
    _set_state(
        cube, layers, samples, groups, features, n_cells, normalization, provenance
    )
    return cube


def _restore_cube(
    layers: dict[str, np.ndarray | _LazyLayer],
    samples: Table,
    groups: Table,
    features: Table,
    n_cells: np.ndarray | None,
    normalization: str | None,
    provenance: str,
) -> Cube:
    # Unpickled arrays are new and writable.
    for stored in layers.values():
        if isinstance(stored, np.ndarray):
            stored.flags.writeable = False
    if n_cells is not None:
        n_cells.flags.writeable = False
    return _assemble(
        layers, samples, groups, features, n_cells, normalization, provenance
    )


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
    return table, index


def _index_of(ids: np.ndarray) -> dict[str, int]:
    return {value: position for position, value in enumerate(ids.tolist())}


def _as_n_cells(
    value: Any, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    group_ids, sample_ids, _ = ids
    if all(hasattr(value, attribute) for attribute in ("index", "columns", "to_numpy")):
        value = _aligned_by_labels(value, sample_ids, group_ids)
    array = np.asarray(value)
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
    if isinstance(value, np.ma.MaskedArray):
        raise TypeError(
            f"layer {name!r} is a masked array; layers have no missing values"
        )
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
    array = np.asarray(value)
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
) -> None:
    """Check one group of a layer, shape ``(n_samples, n_features)``.

    Each check is a single reduction; the offending entry is located only when a
    check fails.
    """
    if block.size == 0:
        return
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
                    f"at {_where(ids, group, sample, feature)}"
                )
    # Only uint64 holds values above 2**63 - 1, and no unsigned value is negative.
    if block.dtype.kind == "u" and block.dtype.itemsize == 8:
        if block.max() > _INT64_MAX:
            sample, feature = _first(block > _INT64_MAX)
            raise ValueError(
                f"layer {name!r} has a value above 2**63 - 1 ({block[sample, feature]}) "
                f"at {_where(ids, group, sample, feature)}; integer layers must be at "
                "most 2**63 - 1, the largest int64"
            )
    elif name in _NON_NEGATIVE_LAYERS and block.dtype.kind != "u" and block.min() < 0:
        sample, feature = _first(block < 0)
        raise ValueError(
            f"layer {name!r} has a negative value ({block[sample, feature]}) at "
            f"{_where(ids, group, sample, feature)}; it must be non-negative"
        )
    if n_cells is None:
        return
    cells = n_cells[:, group]
    if name == "detected":
        over = np.flatnonzero(block.max(axis=1) > cells)
        if over.size:
            sample = int(over[0])
            feature = int(np.argmax(block[sample] > cells[sample]))
            raise ValueError(
                f"layer 'detected' exceeds n_cells at "
                f"{_where(ids, group, sample, feature)}: {block[sample, feature]} "
                f"cells detected but {cells[sample]} pooled"
            )
    empty = np.flatnonzero(cells == 0)
    if empty.size and block[empty].any():
        row, feature = _first(block[empty] != 0)
        sample = int(empty[row])
        raise ValueError(
            f"layer {name!r} is {block[sample, feature]} at "
            f"{_where(ids, group, sample, feature)}, where n_cells is 0; an entry "
            "without cells must be zero in every layer"
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
    values = np.asarray(selector)
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
            same = _values_equal(column[order], np.repeat(first, sizes))
            agree = np.logical_and.reduceat(same, starts)
        else:
            agree = np.ones(0, dtype=bool)
        if agree.all():
            columns[name] = first
        elif keep_varying:
            columns[name] = _with_missing(first, ~agree)
    return Table(columns)


def _preview(ids: np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(ids[:limit].tolist())
    return shown + (f", ... ({ids.size} in all)" if ids.size > limit else "")


def _plain_json(value: Any, path: str) -> Any:
    """Return value as plain JSON types, raising an error that names a bad entry."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            raise ValueError(f"{path} is {value}; JSON has no NaN or infinity")
        return float(value)
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist(), path)
    if isinstance(value, Mapping):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has the key {key!r}; JSON keys are strings")
            plain[key] = _plain_json(item, f"{path}[{key!r}]")
        return plain
    if isinstance(value, list | tuple):
        return [_plain_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(
        f"{path} has a value of type {type(value).__name__!r}, which is not "
        "JSON-serialisable"
    )


def _normalization_text(normalization: Mapping[str, Any]) -> str:
    if not isinstance(normalization, Mapping):
        raise TypeError(
            f"normalization must be a mapping, got {type(normalization).__name__}"
        )
    plain = _plain_json(normalization, "normalization")
    method = plain.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(
            "normalization needs a 'method' string that names how values were "
            "normalized, for example 'library_size'"
        )
    return json.dumps(plain)


def _provenance_text(provenance: Mapping[str, Any] | None) -> str:
    if provenance is None:
        return "{}"
    if not isinstance(provenance, Mapping):
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}"
        )
    plain = _plain_json(provenance, "provenance")
    if "operations" in plain and not isinstance(plain["operations"], list):
        raise ValueError("provenance['operations'] must be a list of operation records")
    return json.dumps(plain)


def _add_record(provenance: str, record: dict[str, Any]) -> str:
    plain = json.loads(provenance)
    plain.setdefault("operations", []).append(record)
    return json.dumps(plain)
