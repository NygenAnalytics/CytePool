"""Text in messages and reprs: lists of ids and names, hints, ranges of values."""

import difflib
from collections.abc import Sequence
from typing import Any

import numpy as np

LISTED = 8
"""The most items a message names before it says how many more there are."""


def listing(items: Sequence[Any]) -> str:
    """Items as "a, b and c", or the first few and how many more."""
    shown = [str(item) for item in items[:LISTED]]
    more = len(items) - len(shown)
    if more:
        return ", ".join(shown) + f" and {more} more"
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def quoted(items: Sequence[Any]) -> str:
    """Items as :func:`listing` gives them, each written as its repr, such as 'S1'."""
    return listing([repr(item) for item in items])


def close_matches(name: str, candidates: tuple[str, ...] | list[str]) -> str:
    """Return a ' (did you mean ...?)' hint for a misspelled name, or ''."""
    found = difflib.get_close_matches(name, list(candidates), n=3, cutoff=0.6)
    if not found:
        return ""
    return " (did you mean " + " or ".join(repr(match) for match in found) + "?)"


def value_range(values: float | np.ndarray) -> str:
    """A value, or the range of the finite values of an array, as reprs show them."""
    if isinstance(values, float):
        return f"{values:.6g}"
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return "[non-finite]"
    return f"[{finite.min():.6g} to {finite.max():.6g}]"
