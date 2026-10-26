"""Building cubes from a cell-level count matrix or from bulk counts.

``from_cells`` pools the cells of every sample and group in one pass over the
stored counts. Cells are sorted by sample and group, and each thread pools whole
sample x group entries into rows of the result that no other thread writes, so no
atomic updates are needed. The entries are shared among the threads by the
longest-processing-time rule (Graham 1969); repeated entries of a row of a sparse
matrix are summed through a sparse-set index (Briggs and Torczon 1993), and log1p is
computed as Goldberg (1991, Theorem 4) shows. ``from_bulk`` wraps a samples x features
count matrix as a cube with a single group.

References:
    Graham RL (1969). Bounds on multiprocessing timing anomalies. SIAM Journal on
    Applied Mathematics 17, 416-429. doi:10.1137/0117039

    Briggs P, Torczon L (1993). An efficient representation for sparse sets. ACM Letters
    on Programming Languages and Systems 2, 59-69. doi:10.1145/176454.176484

    Goldberg D (1991). What every computer scientist should know about floating-point
    arithmetic. ACM Computing Surveys 23, 5-48. doi:10.1145/103162.103163
"""

import heapq
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from numba import prange

from .. import __version__
from .._columns import is_data_frame
from .._jit import jit, thread_limit
from .._masked_arrays import reject_masked, reject_masked_table, unmasked_array
from ..table import Table
from ._axes import COUNT_LIMIT, UINT32_MAX, axis_codes, count_dtype
from .cube import LAYER_NAMES, Cube
from .provenance import _source_record

BULK_GROUP = "bulk"
"""The id of the single group of a cube built by ``from_bulk``."""

_NORMALIZED_LAYERS = ("log_sum", "log_sum_sq", "linear_sum")

# The largest value of a cube's integer layers, the largest int64. It is a uint64,
# like the sums it bounds: numba compares uint64 with an int64 through float64,
# which rounds 2**63 - 1 up to 2**63.
_INT64_MAX = np.uint64(2**63 - 1)
_MOST_DETECTED = 2**32 - 1  # the most cells the uint32 layer detected counts

# The compiled routine takes the counts in one of four arrays, by dtype, so that a
# single compiled version reads each of these dtypes without a copy.
_FLOAT64, _FLOAT32, _INT64, _INT32 = 0, 1, 2, 3
_VALUE_DTYPES = (np.float64, np.float32, np.int64, np.int32)
# Columns: dense rows (the column is the position in the row) or CSR indices.
_DENSE, _INDICES_INT32, _INDICES_INT64 = 0, 1, 2
# Dtypes of the counts layer, in the order they are tried.
_COUNTS_UINT32, _COUNTS_UINT64, _COUNTS_FLOAT64 = 0, 1, 2
_COUNTS_DTYPES = (np.uint32, np.uint64, np.float64)
# Bits that select the optional layers.
_DETECTED, _LOG_SUM, _LOG_SUM_SQ, _LINEAR_SUM = 1, 2, 4, 8
_LAYER_BITS = {
    "detected": _DETECTED,
    "log_sum": _LOG_SUM,
    "log_sum_sq": _LOG_SUM_SQ,
    "linear_sum": _LINEAR_SUM,
}
# What stopped the pooling of a segment. The first two mean that the counts layer
# needs a wider dtype, the others that the input is invalid.
_NOT_WHOLE = 1  # a count that is not a whole number, while counting in integers
_OVERFLOW = 2  # an integer sum beyond 2**32 - 1 (uint32) or 2**63 - 1 (uint64)
_BAD_VALUE = 4  # a negative, NaN, infinite or too large count
_BAD_COLUMN = 8  # a column index outside the matrix
_NO_LIBRARY = 16  # a library size of 0 where normalized values are needed
_BAD_SCALE = 32  # normalized values beyond the float64 range
_INVALID = _BAD_VALUE | _BAD_COLUMN | _NO_LIBRARY | _BAD_SCALE
_FLOAT64_MAX = float(np.finfo(np.float64).max)


def from_cells(
    counts: Any,
    *,
    samples: Any,
    groups: Any,
    sample_table: Any = None,
    group_table: Any = None,
    features: Any = None,
    target_sum: float = 10000.0,
    library_sizes: Any = None,
    layers: Sequence[str] | str | None = None,
    n_threads: int | None = None,
    source: Mapping[str, Any] | None = None,
) -> Cube:
    """Pool the cells of every sample and group into a cube.

    For every sample, group and feature, the cube sums over the cells of that
    sample and group: the raw counts (``counts``), the cells with a positive count
    (``detected``), and the normalized values ``x = count * target_sum / L``, where
    ``L`` is the library size of the cell: ``x`` itself (``linear_sum``),
    ``log1p(x)`` (``log_sum``) and ``log1p(x)**2`` (``log_sum_sq``). ``L`` is the
    total count of the cell, or its value in ``library_sizes``. One pass over the
    stored counts computes every layer.

    Args:
        counts: Counts with one row per cell and one column per feature: a 2-D numpy
            array, or a sparse matrix in CSR layout with ``data``, ``indices``,
            ``indptr`` and ``shape``, such as scipy's ``csr_matrix`` or
            ``csr_array``. Counts are non-negative and below 2**53, whole numbers or
            not. The column indices of a CSR row may be in any order, and repeated
            entries of one row and column are summed, as scipy does.
        samples: The sample label of every cell, as text or integers (integers
            become their decimal text), for example a list, a numpy array or a
            pandas Series or Categorical.
        groups: The group label, usually the cell type, of every cell, in the same
            forms.
        sample_table: Sample metadata, as a ``Table`` or anything ``Table``
            accepts, with a text column ``sample_id`` that holds every sample label.
            Its rows set the order of the samples, and a sample without cells gets
            zeros. By default the samples are the sorted distinct labels (integers
            in numeric order).
        group_table: Group metadata with a text column ``group_id``, used in the
            same way.
        features: Feature metadata with a text column ``feature_id`` and one row
            per column of ``counts``, or a sequence of feature ids. By default the
            ids are the columns of a pandas DataFrame ``counts`` when they are text,
            and otherwise ``feature_0``, ``feature_1``, and so on.
        target_sum: The total that normalization scales every cell to.
        library_sizes: One positive library size per cell to use instead of the
            total count of the cell, for example totals over all features when
            ``counts`` holds only some of them.
        layers: The layers to compute: ``counts``, which every cube needs, and any
            of ``detected``, ``log_sum``, ``log_sum_sq`` and ``linear_sum``. By
            default all five.
        n_threads: The most threads to use. ``None`` uses numba's setting for the
            calling thread (see ``numba.set_num_threads``).
        source: A record of where the cells came from, as a JSON-serialisable
            mapping, stored as JSON in the cube's provenance as it is given: for
            example a dataset identity, references to the artifacts of the tool that
            exported the cells (such as their frozen sample and group memberships,
            which stay in that tool's files) and its settings. CytePool does not
            interpret it.

    Returns:
        A cube with ``n_cells`` of shape (n_samples, n_groups). Its normalization
        is ``{"method": "library_size", "target_sum": target_sum, "log": "natural",
        "pseudocount": 1.0, "library_sizes": "cell_totals"}``, with
        ``"library_sizes": "given"`` when ``library_sizes`` is passed, and None
        without a normalized layer. Its provenance holds ``source`` and the
        settings: the number of cells, the layers, ``target_sum``, where library
        sizes came from, how samples and groups were ordered and where feature ids
        came from (``cytepool.cube.provenance`` documents the record).

    Raises:
        TypeError: If ``counts`` is neither a numeric array nor a CSR matrix (a CSC
            matrix, for example), an argument has the wrong type, ``source`` is not
            a mapping or holds a value JSON cannot hold, or an argument is or holds
            a numpy masked array.
        ValueError: If a count is negative, not finite or at least 2**53 (alone or
            summed over repeated entries), labels are missing or absent from a
            table, lengths disagree, a layer name is unknown, the normalized values
            of a cell are undefined (a library size of 0) or beyond the float64
            range, or ``source`` holds NaN, infinity or a lone surrogate.
        OverflowError: If, over the cells of an entry, the whole-number counts of
            a feature sum to more than 2**63 - 1 (the most a cube's integer layers
            hold) or its normalized values sum beyond the float64 range, or if
            ``detected`` is computed for an entry with more than 2**32 - 1 cells.

    Dtypes:
        ``counts`` is uint32 when every count is a whole number and every sum is
        at most 2**32 - 1, uint64 when a sum is larger (up to 2**63 - 1), and
        float64 when a count is not a whole number. Float input that holds whole
        numbers, as most h5ad files do, gives integer counts. ``detected`` is
        uint32, and the other layers are float64.

    Cells without counts:
        A cell with a library size of 0 adds nothing to ``counts`` and
        ``detected`` and is still counted in ``n_cells``. Its normalized values
        would be 0 / 0, so a normalized layer cannot be computed while it is
        present: remove such cells first, pass ``library_sizes``, or compute only
        ``counts`` and ``detected``.

    Memory:
        Every layer holds n_groups x n_samples x n_features values: 4 bytes each
        for ``counts`` (8 when it is uint64 or float64) and ``detected``, and 8
        bytes for each other layer, 32 bytes per value for all five. For example,
        30 groups x 100 samples x 30,000 features is 9e7 values: 2.9 GB for all
        five layers, 0.36 GB for ``counts`` alone. Beyond the result, pooling uses
        a few arrays with one value per cell, and each thread a few with one value
        per feature. The counts are not copied, except a dense array that is not
        C-contiguous and values of a dtype other than float64, float32, int64 and
        int32.

    Example:
        >>> cube = from_cells(
        ...     np.array([[1, 0], [3, 2], [0, 4]]),
        ...     samples=["S1", "S1", "S2"],
        ...     groups=["T", "T", "T"],
        ...     layers=["counts", "detected"],
        ... )
        >>> cube.layer("counts", "T").tolist()
        [[4, 2], [0, 4]]
    """
    names = _requested_layers(layers)
    target = _positive_number(target_sum, "target_sum")
    record = _source_record(source)
    with thread_limit(n_threads) as threads:
        matrix = _cell_matrix(counts)
        n_rows, n_features = matrix.shape
        sample_codes, sample_axis, sample_integers = axis_codes(
            samples, "samples", n_rows, sample_table, "sample_table", "sample_id"
        )
        group_codes, group_axis, group_integers = axis_codes(
            groups, "groups", n_rows, group_table, "group_table", "group_id"
        )
        feature_origin = _id_origin(features, counts, "columns")
        if features is None:
            features = _frame_labels(counts, "columns")
        feature_axis = _ids_or_table(features, n_features, "features", "feature_id")
        sizes = _library_sizes(library_sizes, n_rows)

        n_samples, n_groups = len(sample_axis), len(group_axis)
        ids = (
            group_axis["group_id"],
            sample_axis["sample_id"],
            feature_axis["feature_id"],
        )
        n_segments = n_groups * n_samples
        # Segment g * n_samples + s is row s of group g in every layer.
        segment_of_cell = group_codes * n_samples + sample_codes
        cells_per_segment = np.bincount(segment_of_cell, minlength=n_segments)
        if "detected" in names:
            _check_detected_fits(cells_per_segment, ids)
        segment_starts = np.zeros(n_segments + 1, dtype=np.int64)
        np.cumsum(cells_per_segment, out=segment_starts[1:])
        cell_order = np.argsort(segment_of_cell, kind="stable").astype(np.int64)
        filled = np.flatnonzero(cells_per_segment)
        # The work of a segment: its stored values, and one unit per cell.
        work = np.bincount(
            segment_of_cell,
            weights=np.diff(matrix.row_starts) + 1.0,
            minlength=n_segments,
        )
        worker_segments, worker_starts = _assign_workers(filled, work[filled], threads)
        pooled = _pool(
            matrix,
            _read_only(cell_order),
            _read_only(segment_starts),
            worker_segments,
            worker_starts,
            _read_only(np.empty(0) if sizes is None else sizes),
            target,
            sum(_LAYER_BITS[name] for name in names if name != "counts"),
            (n_segments, n_features),
            ids,
        )
    if "linear_sum" in names:
        _check_linear_sums(pooled["linear_sum"], ids)

    shape = (n_groups, n_samples, n_features)
    normalization = None
    if any(name in _NORMALIZED_LAYERS for name in names):
        normalization = {
            "method": "library_size",
            "target_sum": target,
            "log": "natural",
            "pseudocount": 1.0,
            "library_sizes": "cell_totals" if sizes is None else "given",
        }
    construction = {
        "function": "from_cells",
        "cytepool_version": __version__,
        "n_cells": n_rows,
        "layers": list(names),
        "target_sum": target,
        "library_sizes": "cell_totals" if sizes is None else "given",
        "samples": {
            "order": "sorted_labels" if sample_table is None else "sample_table",
            "labels": "integers" if sample_integers else "text",
        },
        "groups": {
            "order": "sorted_labels" if group_table is None else "group_table",
            "labels": "integers" if group_integers else "text",
        },
        "features": feature_origin,
    }
    return Cube(
        {name: pooled[name].reshape(shape) for name in names},
        samples=sample_axis,
        groups=group_axis,
        features=feature_axis,
        n_cells=cells_per_segment.reshape(n_groups, n_samples).T,
        normalization=normalization,
        provenance={"source": record, "construction": construction},
        copy=False,
    )


def from_bulk(
    counts: Any,
    *,
    samples: Any = None,
    features: Any = None,
    source: Mapping[str, Any] | None = None,
) -> Cube:
    """Build a cube with one group from bulk counts.

    Args:
        counts: Counts with one row per sample and one column per feature, as a 2-D
            array; a sparse matrix with ``toarray()`` is made dense. Counts are
            non-negative and finite, whole numbers or not (transcript-level
            estimates, for example).
        samples: Sample metadata with a text column ``sample_id`` and one row per
            row of ``counts`` (a ``Table`` or anything ``Table`` accepts), or a
            sequence of sample ids. By default the ids are the index of a pandas
            DataFrame ``counts`` when it holds text, and otherwise ``sample_0``,
            ``sample_1``, and so on.
        features: Feature metadata with a text column ``feature_id`` and one row
            per column of ``counts``, or a sequence of feature ids. By default the
            ids are the columns of a pandas DataFrame ``counts`` when they are text,
            and otherwise ``feature_0``, ``feature_1``, and so on.
        source: A record of where the counts came from, as a JSON-serialisable
            mapping, stored as JSON in the cube's provenance as it is given (see
            ``from_cells``).

    Returns:
        A cube with the single group ``"bulk"``, only the ``counts`` layer, and
        ``n_cells`` None. Counts are uint32 when every count is a whole number up
        to 2**32 - 1, uint64 for larger whole numbers (up to 2**63 - 1), and
        float64 when a count is not a whole number (or, for float input, is at
        least 2**53). Its provenance holds ``source`` and where the sample and
        feature ids came from (``cytepool.cube.provenance`` documents the record).

    Raises:
        TypeError: If ``counts`` is not numeric, ``source`` is not a mapping or
            holds a value JSON cannot hold, or an argument is or holds a numpy
            masked array.
        ValueError: If ``counts`` is not 2-D, has a negative or non-finite value
            or an integer above 2**63 - 1, the largest a cube holds, its shape
            disagrees with ``samples`` or ``features``, or ``source`` holds NaN,
            infinity or a lone surrogate.

    Example:
        >>> cube = from_bulk(np.array([[5, 0, 12], [7, 1, 9]]))
        >>> cube.layer("counts", "bulk").dtype
        dtype('uint32')
    """
    record = _source_record(source)
    construction = {
        "function": "from_bulk",
        "cytepool_version": __version__,
        "samples": _id_origin(samples, counts, "index"),
        "features": _id_origin(features, counts, "columns"),
    }
    if samples is None:
        samples = _frame_labels(counts, "index")
    if features is None:
        features = _frame_labels(counts, "columns")
    if hasattr(counts, "toarray") and not isinstance(counts, np.ndarray):
        counts = counts.toarray()
    array = unmasked_array(counts, "counts")
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must hold numbers (integers or floats), got dtype {array.dtype}"
        )
    if array.ndim != 2:
        raise ValueError(
            "counts must be 2-D, with one row per sample and one column per "
            f"feature, got shape {array.shape}"
        )
    if array.size and array.dtype.kind == "f" and not np.isfinite(array).all():
        row, column = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(
            f"counts has a non-finite value ({array[row, column]}) in row {row}, "
            f"column {column}; counts must be finite"
        )
    if array.size and array.dtype.kind != "u" and array.min() < 0:
        row, column = np.argwhere(array < 0)[0]
        raise ValueError(
            f"counts has a negative value ({array[row, column]}) in row {row}, "
            f"column {column}; counts must be non-negative"
        )
    largest = array.max() if array.size else 0
    if array.dtype.kind == "u" and largest > _INT64_MAX:
        row, column = np.argwhere(array > _INT64_MAX)[0]
        raise ValueError(
            f"counts has a value above 2**63 - 1 ({array[row, column]}) in row {row}, "
            f"column {column}; integer counts must be at most 2**63 - 1, the largest "
            "int64"
        )
    n_samples, n_features = array.shape
    return Cube(
        {"counts": array.astype(count_dtype(array, largest))[None]},
        samples=_ids_or_table(samples, n_samples, "samples", "sample_id"),
        groups={"group_id": [BULK_GROUP]},
        features=_ids_or_table(features, n_features, "features", "feature_id"),
        provenance={"source": record, "construction": construction},
        copy=False,
    )


# The compiled pooling routine.
#
# Array arguments are passed on to plain calls, never to inlined helpers: numba
# counts references to arrays that inlining binds to new names, and inside these
# loops that bookkeeping costs more than the arithmetic.


@jit(inline="always")
def _log1p(x: float) -> float:
    """Return log(1 + x) for x >= 0 within a few units in the last place.

    With u = 1 + x rounded, log(u) * (x / (u - 1)) cancels the rounding of 1 + x
    (Goldberg 1991, Theorem 4). It is about 2.5 times faster than the C library's
    log1p, which otherwise takes most of the time of pooling.
    """
    u = 1.0 + x
    if u == 1.0:
        return x
    return math.log(u) * (x / (u - 1.0))


@jit
def _merge_repeats(
    row_counts: np.ndarray, row_columns: np.ndarray, n: int, where: np.ndarray
) -> int:
    """Sum the entries of a row that share a column, in place.

    ``where`` is a sparse-set index (Briggs and Torczon 1993): ``where[column]`` is
    the position of a column among the merged entries when ``row_columns`` there
    holds that column, so it needs no reset between rows.

    Returns:
        The number of distinct columns, which now come first in the row, or
        ``-1 - column`` for a column whose entries sum to 2**53 or more.
    """
    merged = 0
    for entry in range(n):
        column = row_columns[entry]
        position = where[column]
        if 0 <= position < merged and row_columns[position] == column:
            total = row_counts[position] + row_counts[entry]
            if total >= COUNT_LIMIT:
                return -1 - column  # type: ignore[no-any-return]
            row_counts[position] = total
        else:
            where[column] = merged
            row_counts[merged] = row_counts[entry]
            row_columns[merged] = column
            merged += 1
    return merged


@jit
def _pool_segment(
    segment: int,
    first: int,
    last: int,
    cell_order: np.ndarray,
    values_f64: np.ndarray,
    values_f32: np.ndarray,
    values_i64: np.ndarray,
    values_i32: np.ndarray,
    value_kind: int,
    indices_i32: np.ndarray,
    indices_i64: np.ndarray,
    column_kind: int,
    row_starts: np.ndarray,
    n_features: int,
    library_sizes: np.ndarray,
    target_sum: float,
    counts_kind: int,
    counts_u32: np.ndarray,
    counts_u64: np.ndarray,
    counts_f64: np.ndarray,
    layer_bits: int,
    detected: np.ndarray,
    log_sum: np.ndarray,
    log_sum_sq: np.ndarray,
    linear_sum: np.ndarray,
    row_counts: np.ndarray,
    row_columns: np.ndarray,
    where: np.ndarray,
) -> tuple[int, int, int]:
    """Pool the cells ``cell_order[first:last]`` into row ``segment`` of the layers.

    Each row is read once: its counts are checked and summed, and the positive ones
    are kept in ``row_counts`` and ``row_columns``, from which they are added to the
    layers.

    Returns:
        (0, 0, 0) when every cell is pooled. Otherwise the segment stops at the
        first problem and returns its flag, the cell and the column (-1 when the
        problem is not one column).
    """
    check_whole = counts_kind != _COUNTS_FLOAT64 and value_kind <= _FLOAT32
    want_detected = (layer_bits & _DETECTED) != 0
    want_log_sum = (layer_bits & _LOG_SUM) != 0
    want_log_sum_sq = (layer_bits & _LOG_SUM_SQ) != 0
    want_linear_sum = (layer_bits & _LINEAR_SUM) != 0
    want_logs = want_log_sum or want_log_sum_sq
    normalize = want_logs or want_linear_sum
    given_sizes = library_sizes.size > 0
    for slot in range(first, last):
        cell = cell_order[slot]
        start = row_starts[cell]
        total = 0.0
        n_positive = np.int64(0)  # not a literal, so the merge compiles once
        increasing = True
        previous = -1
        for position in range(start, row_starts[cell + 1]):
            if column_kind == _DENSE:
                column = position - start
            elif column_kind == _INDICES_INT32:
                column = np.int64(indices_i32[position])
            else:
                column = indices_i64[position]
            if value_kind == _FLOAT64:
                count = values_f64[position]
            elif value_kind == _FLOAT32:
                count = np.float64(values_f32[position])
            elif value_kind == _INT64:
                count = np.float64(values_i64[position])
            else:
                count = np.float64(values_i32[position])
            if column < 0 or column >= n_features:
                return _BAD_COLUMN, cell, column
            if not (count >= 0.0 and count < COUNT_LIMIT):
                return _BAD_VALUE, cell, column
            if check_whole and count != np.floor(count):
                return _NOT_WHOLE, cell, column
            total += count
            if count > 0.0:
                if column <= previous:
                    increasing = False
                previous = column
                row_counts[n_positive] = count
                row_columns[n_positive] = column
                n_positive += 1
        if not increasing:  # columns out of order, perhaps repeated
            n_positive = _merge_repeats(row_counts, row_columns, n_positive, where)
            if n_positive < 0:
                return _BAD_VALUE, cell, -1 - n_positive
        scale = 0.0
        if normalize:
            size = library_sizes[cell] if given_sizes else total
            if not size > 0.0:
                return _NO_LIBRARY, cell, -1
            scale = target_sum / size
            # Every normalized value of the cell is at most total * scale.
            if not total * scale <= _FLOAT64_MAX:
                return _BAD_SCALE, cell, -1
        for entry in range(n_positive):
            count = row_counts[entry]
            column = row_columns[entry]
            if counts_kind == _COUNTS_UINT32:
                narrow = np.uint64(counts_u32[segment, column]) + np.uint64(count)
                if narrow > UINT32_MAX:
                    return _OVERFLOW, cell, column
                counts_u32[segment, column] = np.uint32(narrow)
            elif counts_kind == _COUNTS_UINT64:
                # At most 2**63 - 1 plus a count below 2**53: uint64 holds it.
                wide = counts_u64[segment, column] + np.uint64(count)
                if wide > _INT64_MAX:
                    return _OVERFLOW, cell, column
                counts_u64[segment, column] = wide
            else:
                counts_f64[segment, column] += count
            if want_detected:
                detected[segment, column] = np.uint32(detected[segment, column] + 1)
            if normalize:
                value = count * scale
                if want_linear_sum:
                    linear_sum[segment, column] += value
                if want_logs:
                    logged = _log1p(value)
                    if want_log_sum:
                        log_sum[segment, column] += logged
                    if want_log_sum_sq:
                        log_sum_sq[segment, column] += logged * logged
    return 0, 0, 0


@jit(parallel=True)
def _pool_cells(
    values_f64: np.ndarray,
    values_f32: np.ndarray,
    values_i64: np.ndarray,
    values_i32: np.ndarray,
    value_kind: int,
    indices_i32: np.ndarray,
    indices_i64: np.ndarray,
    column_kind: int,
    row_starts: np.ndarray,
    n_features: int,
    longest_row: int,
    cell_order: np.ndarray,
    segment_starts: np.ndarray,
    worker_segments: np.ndarray,
    worker_starts: np.ndarray,
    library_sizes: np.ndarray,
    target_sum: float,
    counts_kind: int,
    counts_u32: np.ndarray,
    counts_u64: np.ndarray,
    counts_f64: np.ndarray,
    layer_bits: int,
    detected: np.ndarray,
    log_sum: np.ndarray,
    log_sum_sq: np.ndarray,
    linear_sum: np.ndarray,
    status: np.ndarray,
) -> None:
    """Pool every segment; each worker pools its own list of segments.

    ``status[segment]`` receives what stopped a segment, if anything: its flag, the
    cell and the column.
    """
    for worker in prange(worker_starts.size - 1):
        row_counts = np.empty(longest_row)
        row_columns = np.empty(longest_row, dtype=np.int64)
        where = np.zeros(n_features, dtype=np.int64)
        for slot in range(worker_starts[worker], worker_starts[worker + 1]):
            segment = worker_segments[slot]
            flag, cell, column = _pool_segment(
                segment,
                segment_starts[segment],
                segment_starts[segment + 1],
                cell_order,
                values_f64,
                values_f32,
                values_i64,
                values_i32,
                value_kind,
                indices_i32,
                indices_i64,
                column_kind,
                row_starts,
                n_features,
                library_sizes,
                target_sum,
                counts_kind,
                counts_u32,
                counts_u64,
                counts_f64,
                layer_bits,
                detected,
                log_sum,
                log_sum_sq,
                linear_sum,
                row_counts,
                row_columns,
                where,
            )
            status[segment, 0] = flag
            status[segment, 1] = cell
            status[segment, 2] = column


# Running the routine.


class _Counts(NamedTuple):
    """A count matrix in the form the compiled routine reads.

    ``values`` holds one array per dtype in ``_VALUE_DTYPES``, all empty except the
    one at ``value_kind``; ``indices`` likewise holds int32 and int64 column
    indices, both empty for dense counts.
    """

    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    value_kind: int
    indices: tuple[np.ndarray, np.ndarray]
    column_kind: int
    row_starts: np.ndarray
    shape: tuple[int, int]
    longest_row: int


def _pool(
    matrix: _Counts,
    cell_order: np.ndarray,
    segment_starts: np.ndarray,
    worker_segments: np.ndarray,
    worker_starts: np.ndarray,
    library_sizes: np.ndarray,
    target_sum: float,
    layer_bits: int,
    shape: tuple[int, int],
    ids: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the pooling routine and return every layer, of shape ``shape``.

    Counts are first summed as uint32. A count that is not a whole number, or a
    sum above 2**32 - 1, stops the segment where it occurs, and the pooling runs
    again with float64 or uint64 counts. A uint64 sum above 2**63 - 1 raises
    OverflowError, naming its entry by the group, sample and feature ``ids``.
    """

    def layer(bit: int, dtype: type) -> np.ndarray:
        return np.zeros(shape if layer_bits & bit else (0, 0), dtype=dtype)

    detected = layer(_DETECTED, np.uint32)
    log_sum = layer(_LOG_SUM, np.float64)
    log_sum_sq = layer(_LOG_SUM_SQ, np.float64)
    linear_sum = layer(_LINEAR_SUM, np.float64)
    counts_kind = _COUNTS_UINT32
    while True:
        counts = [np.zeros((0, 0), dtype=dtype) for dtype in _COUNTS_DTYPES]
        counts[counts_kind] = np.zeros(shape, dtype=_COUNTS_DTYPES[counts_kind])
        status = np.zeros((shape[0], 3), dtype=np.int64)
        if worker_segments.size:
            _pool_cells(
                *matrix.values,
                matrix.value_kind,
                *matrix.indices,
                matrix.column_kind,
                matrix.row_starts,
                shape[1],
                matrix.longest_row,
                cell_order,
                segment_starts,
                worker_segments,
                worker_starts,
                library_sizes,
                target_sum,
                counts_kind,
                *counts,
                layer_bits,
                detected,
                log_sum,
                log_sum_sq,
                linear_sum,
                status,
            )
        flags = status[:, 0]
        invalid = np.flatnonzero(flags & _INVALID)
        if invalid.size:
            first = invalid[np.argmin(status[invalid, 1])]
            flag, cell, column = (int(value) for value in status[first])
            raise _invalid_input(matrix, flag, cell, column)
        if counts_kind != _COUNTS_FLOAT64 and (flags & _NOT_WHOLE).any():
            counts_kind = _COUNTS_FLOAT64
        elif (flags & _OVERFLOW).any():
            if counts_kind == _COUNTS_UINT32:
                counts_kind = _COUNTS_UINT64
            elif _has_fractions(matrix):
                # A segment stopped at the overflow before it reached a count that
                # is not a whole number, which makes the counts float64.
                counts_kind = _COUNTS_FLOAT64
            else:
                raise _sum_too_large(status, ids)
        else:
            return {
                "counts": counts[counts_kind],
                "detected": detected,
                "log_sum": log_sum,
                "log_sum_sq": log_sum_sq,
                "linear_sum": linear_sum,
            }
        # Pool again with the wider counts dtype.
        del counts
        for array in (detected, log_sum, log_sum_sq, linear_sum):
            array.fill(0)


def _has_fractions(matrix: _Counts) -> bool:
    """Return whether a stored count is not a whole number (or is NaN)."""
    if matrix.value_kind not in (_FLOAT64, _FLOAT32):
        return False
    stored = matrix.values[matrix.value_kind][: int(matrix.row_starts[-1])]
    return bool(np.any(stored != np.floor(stored)))


def _sum_too_large(
    status: np.ndarray, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> OverflowError:
    """Describe the uint64 sum above 2**63 - 1 that stopped at the earliest cell."""
    stopped = np.flatnonzero(status[:, 0] & _OVERFLOW)
    segment = int(stopped[np.argmin(status[stopped, 1])])
    group, sample = divmod(segment, ids[1].size)
    feature = int(status[segment, 2])
    return OverflowError(
        f"layer 'counts' at group {ids[0][group]!r}, sample {ids[1][sample]!r}, "
        f"feature {ids[2][feature]!r} sums over its cells to more than 2**63 - 1, "
        "the largest int64"
    )


def _check_detected_fits(
    cells_per_segment: np.ndarray, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Raise if an entry has more cells than the uint32 ``detected`` layer counts."""
    if cells_per_segment.size == 0:
        return
    segment = int(np.argmax(cells_per_segment))
    n_cells = int(cells_per_segment[segment])
    if n_cells > _MOST_DETECTED:
        group, sample = divmod(segment, ids[1].size)
        raise OverflowError(
            f"group {ids[0][group]!r}, sample {ids[1][sample]!r} has {n_cells:,} "
            f"cells, more than the {_MOST_DETECTED:,} that the uint32 layer "
            "'detected' counts; leave 'detected' out of layers"
        )


def _check_linear_sums(
    linear_sum: np.ndarray, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Raise if the normalized values of an entry sum beyond the float64 range.

    Only ``linear_sum`` can: a log1p value is below 710.
    """
    with np.errstate(over="ignore"):
        total = linear_sum.sum()  # not finite if an entry is not, or if they add up
    if np.isfinite(total):
        return
    beyond = ~np.isfinite(linear_sum)
    if beyond.any():
        segment, feature = (int(i) for i in np.argwhere(beyond)[0])
        group, sample = divmod(segment, ids[1].size)
        raise OverflowError(
            f"layer 'linear_sum' at group {ids[0][group]!r}, sample "
            f"{ids[1][sample]!r}, feature {ids[2][feature]!r} sums over its cells "
            "beyond the float64 range (about 1.8e308); lower target_sum, or check "
            "library_sizes"
        )


def _invalid_input(matrix: _Counts, flag: int, cell: int, column: int) -> ValueError:
    """Describe the problem that the pooling routine found in the input."""
    if flag == _NO_LIBRARY:
        return ValueError(
            f"cell {cell} (row {cell} of counts) has no counts, so its library size "
            "is 0 and its normalized values are undefined; remove cells without "
            "counts, pass library_sizes, or compute only the 'counts' and "
            "'detected' layers"
        )
    if flag == _BAD_SCALE:
        return ValueError(
            f"the normalized values of cell {cell} (row {cell} of counts), count * "
            "target_sum / library size, exceed the float64 range; check its library "
            "size, or lower target_sum"
        )
    if flag == _BAD_COLUMN:
        return ValueError(
            f"counts.indices holds the column index {column} in row {cell}, but "
            f"counts has {matrix.shape[1]} columns"
        )
    start, stop = (int(position) for position in matrix.row_starts[cell : cell + 2])
    stored = matrix.values[matrix.value_kind][start:stop]
    if matrix.column_kind == _DENSE:
        entries = stored[column : column + 1]
    else:
        entries = stored[matrix.indices[matrix.column_kind - 1][start:stop] == column]
    bad = [float(value) for value in entries if not 0 <= value < COUNT_LIMIT]
    if not bad:
        found = (
            f"repeated entries that sum to {float(entries.sum(dtype=np.float64)):g}, "
            "at or above 2**53"
        )
    elif math.isnan(bad[0]):
        found = "a NaN"
    elif bad[0] < 0:
        found = f"a negative value ({bad[0]:g})"
    elif math.isinf(bad[0]):
        found = "an infinite value"
    else:
        found = f"a value of {bad[0]:g}, at or above 2**53"
    return ValueError(
        f"counts has {found} in row {cell}, column {column}; counts must be "
        "non-negative and below 2**53, where float64 holds every whole number"
    )


def _assign_workers(
    segments: np.ndarray, work: np.ndarray, n_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Share segments among workers so that each gets about the same work.

    Segments are taken from the most work to the least, and each goes to the
    worker with the least work so far (the longest-processing-time rule, Graham
    1969). numba
    gives each thread an equal share of the iterations of a parallel loop, so the
    loop runs over workers, one per thread, rather than over segments of unequal
    size.

    Returns:
        The segments of all workers, worker by worker, and where each worker's
        segments start (one more entry than workers).
    """
    n_workers = max(1, min(n_workers, segments.size))
    order = np.argsort(-work, kind="stable")
    owners = [0] * order.size
    if n_workers > 1:
        loads = [(0.0, worker) for worker in range(n_workers)]
        for rank, amount in enumerate(work[order].tolist()):
            load, worker = heapq.heappop(loads)
            owners[rank] = worker
            heapq.heappush(loads, (load + amount, worker))
    owner = np.array(owners, dtype=np.int64)
    by_worker = np.argsort(owner, kind="stable")
    worker_segments = segments[order[by_worker]].astype(np.int64)
    worker_starts = np.searchsorted(owner[by_worker], np.arange(n_workers + 1))
    return _read_only(worker_segments), _read_only(worker_starts.astype(np.int64))


# Reading the arguments.


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a read-only view: one compiled version then serves any input."""
    view = array.view()
    view.flags.writeable = False
    return view


_EMPTY_VALUES = tuple(_read_only(np.empty(0, dtype=dtype)) for dtype in _VALUE_DTYPES)
_EMPTY_INDICES = (
    _read_only(np.empty(0, dtype=np.int32)),
    _read_only(np.empty(0, dtype=np.int64)),
)


def _cell_matrix(counts: Any) -> _Counts:
    """Read dense or CSR counts into the form the compiled routine takes."""
    if hasattr(counts, "indptr") or isinstance(getattr(counts, "format", None), str):
        return _csr_counts(counts)
    if hasattr(counts, "toarray") and not isinstance(counts, np.ndarray):
        raise TypeError(
            f"counts is a {type(counts).__name__} without CSR arrays; pass a numpy "
            "array or a CSR matrix with data, indices, indptr and shape"
        )
    array = unmasked_array(counts, "counts")
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must hold numbers (integers or floats), got dtype {array.dtype}"
        )
    if array.ndim != 2:
        raise ValueError(
            "counts must be 2-D, with one row per cell and one column per feature, "
            f"got shape {array.shape}"
        )
    n_rows, n_columns = array.shape
    values, value_kind = _value_slots(np.ascontiguousarray(array).reshape(-1))
    row_starts = np.arange(n_rows + 1, dtype=np.int64) * n_columns
    return _Counts(
        values,
        value_kind,
        _EMPTY_INDICES,
        _DENSE,
        _read_only(row_starts),
        (n_rows, n_columns),
        n_columns,
    )


def _csr_counts(counts: Any) -> _Counts:
    layout = getattr(counts, "format", "csr")
    if layout != "csr":
        raise TypeError(
            f"counts is a sparse matrix in {layout!r} format, but cells must be the "
            "rows of a CSR matrix; convert it with counts.tocsr()"
        )
    missing = [
        name
        for name in ("data", "indices", "indptr", "shape")
        if not hasattr(counts, name)
    ]
    if missing:
        raise TypeError(
            "a CSR counts matrix needs data, indices, indptr and shape; this "
            f"{type(counts).__name__} has no {', '.join(missing)}"
        )
    shape = tuple(int(size) for size in counts.shape)
    if len(shape) != 2 or min(shape) < 0:
        raise ValueError(f"counts must be 2-D, with sizes of 0 or more, got {shape}")
    n_rows, n_columns = shape
    data = unmasked_array(counts.data, "counts.data")
    indices = unmasked_array(counts.indices, "counts.indices")
    indptr = unmasked_array(counts.indptr, "counts.indptr")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("counts.data, counts.indices and counts.indptr must be 1-D")
    if data.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must hold numbers (integers or floats), got dtype {data.dtype}"
        )
    if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
        raise TypeError(
            "counts.indices and counts.indptr must hold integers, got dtypes "
            f"{indices.dtype} and {indptr.dtype}"
        )
    if indptr.size != n_rows + 1:
        hint = (
            "; it looks like CSC layout, so convert it with counts.tocsr()"
            if indptr.size == n_columns + 1
            else ""
        )
        raise ValueError(
            f"counts.indptr has {indptr.size} entries, but a CSR matrix with {n_rows} "
            f"rows needs {n_rows + 1}{hint}"
        )
    stored = min(data.size, indices.size)
    largest = np.iinfo(np.int64).max
    if indices.dtype.kind == "u" and indices.size and indices.max() > largest:
        raise ValueError(
            f"counts.indices holds the column index {indices.max()}, but counts has "
            f"{n_columns} columns"
        )
    if indptr.dtype.kind == "u" and indptr.size and indptr.max() > largest:
        raise ValueError(
            f"counts.indptr holds {indptr.max()}, more than the number of stored "
            f"values ({stored})"
        )
    row_starts = indptr.astype(np.int64)
    row_lengths = np.diff(row_starts)
    if row_starts[0] != 0 or row_starts[-1] > stored or (row_lengths < 0).any():
        raise ValueError(
            "counts.indptr must start at 0, never decrease and end at most at the "
            f"number of stored values ({stored}); it runs from {row_starts[0]} to "
            f"{row_starts[-1]}"
        )
    values, value_kind = _value_slots(data)
    if indices.dtype == np.int32:
        columns = (_read_only(np.ascontiguousarray(indices)), _EMPTY_INDICES[1])
        column_kind = _INDICES_INT32
    else:
        wide = np.ascontiguousarray(indices, dtype=np.int64)
        columns = (_EMPTY_INDICES[0], _read_only(wide))
        column_kind = _INDICES_INT64
    return _Counts(
        values,
        value_kind,
        columns,
        column_kind,
        _read_only(row_starts),
        (n_rows, n_columns),
        int(row_lengths.max(initial=0)),
    )


def _value_slots(
    data: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
    """Place the counts in the array of their dtype, converting rarer dtypes."""
    dtype = data.dtype
    if dtype.kind == "f" and dtype not in (np.float64, np.float32):
        data = data.astype(np.float64)  # float16, extended precision, byte-swapped
    elif dtype.kind == "u" and dtype.itemsize == 8:
        data = data.astype(np.float64)  # exact below 2**53, the largest count accepted
    elif dtype.kind in "iu" and dtype not in (np.int64, np.int32):
        data = data.astype(np.int64)
    kind = next(k for k, slot in enumerate(_VALUE_DTYPES) if data.dtype == slot)
    slots = list(_EMPTY_VALUES)
    slots[kind] = _read_only(np.ascontiguousarray(data))
    return (slots[0], slots[1], slots[2], slots[3]), kind


def _requested_layers(layers: Sequence[str] | str | None) -> tuple[str, ...]:
    if layers is None:
        return LAYER_NAMES
    reject_masked(layers, "layers")
    listed = [layers] if isinstance(layers, str) else list(layers)
    for name in listed:
        if not isinstance(name, str):
            raise TypeError(
                f"layers must be layer names (str), got {type(name).__name__} {name!r}"
            )
        if name not in LAYER_NAMES:
            raise ValueError(
                f"unknown layer {name!r}; layers are named: {', '.join(LAYER_NAMES)}"
            )
    if len(set(listed)) != len(listed):
        raise ValueError(f"layers lists a layer more than once: {listed}")
    if "counts" not in listed:
        raise ValueError(
            f"layers must include 'counts', which every cube holds; got {listed}"
        )
    return tuple(name for name in LAYER_NAMES if name in listed)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be positive and finite, got {number}")
    return number


def _library_sizes(value: Any, n_cells: int) -> np.ndarray | None:
    if value is None:
        return None
    array = unmasked_array(value, "library_sizes")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"library_sizes must hold numbers, got dtype {array.dtype}")
    if array.shape != (n_cells,):
        raise ValueError(
            f"library_sizes must hold one value per cell ({n_cells}), got shape "
            f"{array.shape}"
        )
    sizes = np.array(array, dtype=np.float64)
    bad = np.flatnonzero(~(np.isfinite(sizes) & (sizes > 0)))
    if bad.size:
        raise ValueError(
            f"library_sizes must be positive and finite; cell {bad[0]} has "
            f"{sizes[bad[0]]:g}"
        )
    return sizes


def _frame_labels(counts: Any, axis: str) -> np.ndarray | None:
    """Return the index or columns of a pandas DataFrame when they hold text."""
    if not is_data_frame(counts):
        return None
    labels = np.asarray(getattr(counts, axis), dtype=object)
    if all(isinstance(label, str) and label for label in labels.tolist()):
        return labels
    return None


def _id_origin(value: Any, counts: Any, labels: str) -> str:
    """Name where the ids of an axis come from, for the provenance record.

    ``labels`` is the attribute of a DataFrame ``counts`` that gives them by
    default: ``"index"`` or ``"columns"``.
    """
    if value is None:
        found = _frame_labels(counts, labels)
        return "positions" if found is None else f"data_frame_{labels}"
    if isinstance(value, Table | Mapping) or is_data_frame(value):
        return "table"
    return "ids"


def _ids_or_table(value: Any, n: int, axis: str, id_column: str) -> Table:
    """Return an axis table from a table, a sequence of ids, or None for default ids."""
    unit = "row" if axis == "samples" else "column"
    if value is None:
        ids = np.empty(n, dtype=object)
        ids[:] = [f"{axis[:-1]}_{i}" for i in range(n)]
        return Table({id_column: ids})
    if isinstance(value, Table | Mapping) or is_data_frame(value):
        reject_masked_table(value, axis, id_column)
        table = Table(value)
        if len(table) != n:
            raise ValueError(
                f"the {axis} table has {len(table)} rows, but counts has {n} {unit}s"
            )
        return table
    reject_masked(value, axis)
    try:
        table = Table({id_column: value})
    except (TypeError, ValueError) as error:
        error.add_note(f"(while reading the {axis} ids)")
        raise
    if len(table) != n:
        raise ValueError(
            f"{axis} must hold one id per {unit} of counts ({n}), got {len(table)}"
        )
    return table
