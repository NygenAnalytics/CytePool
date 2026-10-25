"""Lists of ids and names in error messages."""

from collections.abc import Sequence
from typing import Any

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
