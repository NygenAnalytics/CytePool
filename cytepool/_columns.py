"""Columns for the tables that results hold."""

from collections.abc import Sequence

import numpy as np


def text_column(values: Sequence[str]) -> np.ndarray:
    """A text column for a ``Table``, also when it is empty.

    numpy reads an empty list as float64, which would make an empty column of ids or
    reasons a column of numbers; an object array stays a text column.
    """
    column = np.empty(len(values), dtype=object)
    column[:] = list(values)
    return column
