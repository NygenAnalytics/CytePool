"""Gene ids given as lists: strings, with None, NaN and empty strings missing.

Gene-set collections and over-representation analysis read lists of gene ids: the
genes of a set, of an analysis, of a selection and of a universe. numpy turns a list
that mixes text and numbers into text, so that the id ``7157`` given among symbols
would become ``"7157"``; the elements of lists and tuples are therefore read as
Python gives them.
"""

import math
from collections.abc import Set
from typing import Any

import numpy as np

from ._masked_arrays import Missing, reject_masked, unmasked_array


def listed_values(values: Any, label: str, *, missing: Missing = None) -> list[Any]:
    """The elements of a 1-D list, tuple, set or array, as Python objects.

    The elements of a ``set`` are sorted, so that every run lists them alike.

    Raises:
        TypeError: If ``values`` is a single string, an array of numbers, or is or
            holds a numpy masked array.
        ValueError: If an array is not one-dimensional.
    """
    if isinstance(values, str | bytes):
        raise TypeError(
            f"{label} is a single string; give the genes as a list, such as "
            f"[{values!r}]"
        )
    if isinstance(values, Set):
        values = sorted(values, key=str)
    if isinstance(values, list | tuple):
        reject_masked(values, label, missing=missing)
        return list(values)
    array = unmasked_array(values, label, missing=missing)
    if array.ndim != 1:
        raise ValueError(
            f"{label} must be a list of gene ids (1-D), got shape {array.shape}"
        )
    if array.dtype.kind not in "UO" and array.size:
        raise TypeError(
            f"{label} must hold gene ids as strings, got dtype {array.dtype}; convert "
            "numeric ids, such as Entrez ids, to text first"
        )
    values_read: list[Any] = array.tolist()
    return values_read


def gene_ids(values: Any, name: str) -> tuple[list[str], int]:
    """Gene ids as strings, in order, without the missing ones; and how many were
    missing.

    None, NaN, pandas' missing values and empty strings are missing.

    Args:
        values: A 1-D list, tuple, set or array of ids.
        name: The argument as messages name it.

    Raises:
        TypeError: If a value is neither a string nor missing, ``values`` is a single
            string, or it is or holds a numpy masked array.
        ValueError: If an array is not one-dimensional.
    """
    ids = []
    found = listed_values(values, name, missing="None")
    for value in found:
        if isinstance(value, str):
            if value:
                ids.append(str(value))
        elif not is_missing(value):
            raise TypeError(
                f"{name} must hold gene ids as strings, got {type(value).__name__} "
                f"{value!r}"
            )
    return ids, len(found) - len(ids)


def is_missing(value: object) -> bool:
    """Whether a value marks a missing id: None, NaN, pandas' NA or NaT, or ""."""
    if value is None or (isinstance(value, str) and not value):
        return True
    if isinstance(value, float | np.floating):
        return math.isnan(float(value))
    # pandas.NA and pandas.NaT, recognized without importing pandas.
    return type(value).__name__ in ("NAType", "NaTType")
