"""Refusing numpy masked arrays where CytePool reads array input, and reading reals.

numpy's conversions (``numpy.asarray``, ``numpy.array``, ``numpy.ascontiguousarray``)
keep the values under the mask of a ``numpy.ma.MaskedArray`` and drop the mask, so an
entry marked as missing would silently be read as data. ``numpy.ma.masked`` in a list
fares no better: numpy turns it into NaN among numbers and into the text ``"0.0"``
among strings. CytePool therefore accepts neither. Every public function, method and
constructor passes each argument that can hold array data to ``reject_masked``, and
the message says how to mark missing values in a form the argument accepts. Data
arguments (counts, expression values, designs, weights, labels, positions, table
columns, the arguments of the distributions) and the values read from lazy sources
are converted with ``unmasked_array``, which also refuses an array-like that converts
to a masked array, such as a netCDF4 variable; the small arrays that make up
contrasts and result objects are searched only. pandas objects are not inspected:
pandas marks missing values without numpy masks. ``real_array`` then reads arrays
of real numbers as float64, and ``expression_matrix`` the samples x features
values of a fit. ``read_only`` gives the read-only views that results hold.
"""

import numbers
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Missing = Literal["NaN", "None", "None or NaN"] | None
"""How an argument marks missing values: with NaN (numbers), with None (JSON
metadata), with None in text and NaN in numbers (table columns), or not at all
(None)."""

FloatArray = npt.NDArray[np.float64]

_REASON = (
    "CytePool does not accept numpy masked arrays, because numpy drops the mask when "
    "it converts one, and the masked entries would be read as data."
)
# For each way of marking missing values: the remedy, and how to apply it to a
# masked array, which numpy.ma.masked itself does not need.
_REMEDIES: dict[Missing, tuple[str, str]] = {
    None: ("Remove the masked entries, or fill them with valid values, first", ""),
    "NaN": (
        "Mark missing values with NaN instead",
        ", for example with numpy.ma.filled(values.astype(float), numpy.nan)",
    ),
    "None": (
        "Mark missing values with None instead",
        "; values.tolist() gives None for each masked entry",
    ),
    "None or NaN": (
        "Mark missing values with None, or NaN in numbers, instead",
        "; values.tolist() gives None for each masked entry",
    ),
}
_NOTHING_MASKED = "No entry is masked: pass numpy.ma.getdata(values) instead."
# Values that cannot be or hold a masked array, recognized by their exact type first,
# which keeps the check of a scalar argument to a few hundred nanoseconds.
_PLAIN_TYPES = frozenset({bool, int, float, complex, str, bytes, type(None)})


def reject_masked(value: object, name: str, *, missing: Missing = None) -> None:
    """Raise TypeError if an argument is, or holds, a numpy masked array.

    A masked array is found where it is the argument itself, a value of a mapping, an
    element of a list, tuple or object array, or a bound of a slice, at any depth;
    ``numpy.ma.masked`` counts as one. Slices are searched because Python reads their
    bounds through ``__index__``, which ignores the mask of a 0-d masked integer.
    Numeric arrays cannot hold a masked array and are not scanned, so the check costs
    one pass over the types of the elements of a list or object array, and nothing
    for a numeric array. A masked array is refused even when no entry is masked; the
    message then says how to drop the mask.

    Args:
        value: The argument as given.
        name: The argument as messages name it, such as ``"counts"`` or
            ``"layers['counts']"``. Where the masked array sits inside it is
            appended as subscripts, as in ``samples[2]``.
        missing: How the argument marks missing values, which decides the remedy
            the message gives: ``"NaN"``, ``"None"`` (JSON metadata), ``"None or
            NaN"`` (table columns), or None when it accepts no missing values.

    Raises:
        TypeError: If ``value`` is or holds a masked array.
    """
    found = _find_masked(value)
    if found is not None:
        where, masked = found
        raise _refusal(f"{name}{where}", masked, missing)


def reject_masked_table(table: object, name: str, id_column: str) -> None:
    """Raise TypeError if a table argument, such as a sample table, holds a masked array.

    The id column accepts no missing values, and the other columns accept None and
    NaN, as ``Table`` columns do; the message gives the remedy of the column.

    Args:
        table: The argument as given: a mapping of columns, a ``Table`` or a pandas
            DataFrame (neither of which can hold a masked array).
        name: The argument as messages name it.
        id_column: The column that holds the ids.

    Raises:
        TypeError: If a column is or holds a masked array.
    """
    if isinstance(table, Mapping) and id_column in table:
        reject_masked(table[id_column], f"{name}[{id_column!r}]")
    reject_masked(table, name, missing="None or NaN")


def unmasked_array(value: object, name: str, *, missing: Missing = None) -> np.ndarray:
    """Return an argument as a numpy array, refusing numpy masked arrays.

    The argument is searched as by ``reject_masked``, then converted once with
    ``numpy.asanyarray``. That keeps the masked array that an array-like returns from
    ``__array__``, as a netCDF4 variable or a dask array of masked chunks does, which
    ``numpy.asarray`` would drop the mask of; such a result is refused too.

    Args:
        value: The argument as given.
        name: The argument as messages name it; see ``reject_masked``.
        missing: How the argument marks missing values; see ``reject_masked``.

    Returns:
        The argument as a numpy array, of the base class ``numpy.ndarray``.

    Raises:
        TypeError: If ``value`` is, holds or converts to a masked array.
    """
    reject_masked(value, name, missing=missing)
    array = np.asanyarray(value)
    if type(array) is np.ndarray:
        return array
    if isinstance(array, np.ma.MaskedArray):
        raise _refusal(name, array, missing, converted=True)
    return np.asarray(array)


def real_array(
    values: object, name: str, *, boolean: bool = True, objects: bool = True
) -> np.ndarray:
    """Values as float64 without copying when possible.

    Integers and floats are read, booleans with ``boolean``, and object arrays whose
    every element is a real number with ``objects``, such as a table whose columns mix
    booleans and floats. The caller refuses masked arrays first.

    Raises:
        TypeError: If the values are not real numbers.
    """
    array = np.asarray(values)
    if objects and array.dtype.kind == "O":
        flags = (bool, np.bool_)
        kinds = {type(value) for value in array.flat}
        if not all(issubclass(kind, (numbers.Real, np.bool_)) for kind in kinds) or (
            not boolean and any(issubclass(kind, flags) for kind in kinds)
        ):
            raise TypeError(f"{name} must hold real numbers")
        array = array.astype(np.float64)
    if array.dtype.kind not in ("biuf" if boolean else "iuf"):
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    return array.astype(np.float64, copy=False)


def _refusal(
    subject: str, masked: np.ndarray, missing: Missing, *, converted: bool = False
) -> TypeError:
    """The standard error for a masked array found at ``subject``."""
    if masked is np.ma.masked:
        what = "numpy.ma.masked"
        remedy = _REMEDIES[missing][0] + "."
    else:
        what = (
            "an array-like that converts to a numpy masked array"
            if converted
            else "a numpy masked array"
        )
        if np.ma.is_masked(masked):
            remedy = "".join(_REMEDIES[missing]) + "."
        else:
            remedy = _NOTHING_MASKED
    return TypeError(f"{subject} is {what}; {_REASON} {remedy}")


def _find_masked(value: object) -> tuple[str, np.ndarray] | None:
    """Return where a masked array sits in ``value``, as subscripts, and the array.

    The subscripts are ``""`` for ``value`` itself, or for example ``"['x'][2]"``, or
    ``"[1].stop"`` for the stop of a slice.
    """
    if type(value) in _PLAIN_TYPES or isinstance(value, np.generic):
        return None  # numpy.ma.masked is an array, not a numpy scalar
    if isinstance(value, np.ndarray):
        if isinstance(value, np.ma.MaskedArray):
            return "", value
        if value.dtype != object or value.size == 0:
            return None
        if not _may_hold_masked(value.flat):
            return None
        for position, item in enumerate(value.flat):
            found = _find_masked(item)
            if found is not None:
                index = np.unravel_index(position, value.shape)
                text = ", ".join(str(int(i)) for i in index) if index else "()"
                return f"[{text}]{found[0]}", found[1]
        return None
    if isinstance(value, list | tuple):
        if not _may_hold_masked(value):
            return None
        for position, item in enumerate(value):
            found = _find_masked(item)
            if found is not None:
                return f"[{position}]{found[0]}", found[1]
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_masked(item)
            if found is not None:
                # A numpy string key is named as the text it holds.
                label = repr(str(key)) if isinstance(key, str) else repr(key)
                return f"[{label}]{found[0]}", found[1]
        return None
    if isinstance(value, slice):
        for part in ("start", "stop", "step"):
            found = _find_masked(getattr(value, part))
            if found is not None:
                return f".{part}{found[0]}", found[1]
    return None


def _may_hold_masked(items: Iterable[object]) -> bool:
    """Whether a sequence's first level holds an array, sequence, mapping or slice.

    Only such elements can be or hold a masked array. Collecting the distinct types
    runs at C speed, so a list of a million labels costs a few milliseconds.
    """
    kinds = set(map(type, items))
    return any(
        issubclass(kind, np.ndarray | list | tuple | Mapping | slice) for kind in kinds
    )


def expression_matrix(y: npt.ArrayLike, n_samples: int) -> FloatArray:
    """The values ``y`` of a fit as float64, samples x features: real numbers, one row
    per row of the design, NaN for a missing value and no infinity."""
    values = real_array(unmasked_array(y, "y", missing="NaN"), "y", boolean=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_features), "
            f"got shape {values.shape}"
        )
    if values.shape[0] != n_samples:
        raise ValueError(
            f"y has {values.shape[0]} rows but design has {n_samples}; "
            "rows of both are samples"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one feature (column)")
    if np.isinf(values).any():
        raise ValueError("y contains infinite values; mark missing values with NaN")
    return values


def read_only(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return a view of an array that cannot be written."""
    view = values.view()
    view.flags.writeable = False
    return view
