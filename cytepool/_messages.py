"""Text in messages and reprs: lists of ids and names, hints, ranges of values, and
lines that fit the width of a page."""

import difflib
import os
import re
import textwrap
from collections.abc import Sequence
from typing import Any

import numpy as np

LISTED = 8
"""The most items a message names before it says how many more there are."""

WIDTH = 80
"""The most characters in a line of a repr or a printed record, as pages show them."""

KEEP_TOGETHER = "\ue000"
"""A space at which :func:`fitted_lines` does not break a line, such as the one in
``counts (uint32)``; the text it returns has a plain space there."""

_NUMBERED = re.compile(r"\d+\. ")
# A line that begins with a short key and a colon, such as "  layers: ...".
_KEYED = re.compile(r" *[^\s:][^:]{0,29}: ")

PACKAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
"""The package's directory, ending in a separator: warnings skip the files under it,
so that a warning is attributed to the first caller outside the package."""


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


def fitted_lines(text: str, width: int = WIDTH) -> str:
    """Text whose lines are at most ``width`` characters, broken between words.

    A broken line continues below the text after its key, as in ``layers: ...``,
    or after the number of a numbered line (``1. ...``), and otherwise, or where its
    longest word would not fit there, four spaces deeper than it began. A word
    longer than a line stays whole, and so do words joined by ``KEEP_TOGETHER``; a
    fingerprint, 71 characters, fits on a line of its own.
    """
    lines = []
    for line in text.split("\n"):
        if len(line) <= width:
            lines.append(line.replace(KEEP_TOGETHER, " "))
            continue
        body = line.lstrip(" ")
        indent = line[: len(line) - len(body)]
        numbered = _NUMBERED.match(body)
        keyed = _KEYED.match(line)
        longest = max(len(word) for word in line.split())
        if numbered:
            hanging = indent + " " * numbered.end()
        elif keyed and keyed.end() <= width // 2:
            hanging = " " * keyed.end()
        else:
            hanging = indent + "    "
        if len(hanging) + longest > width:
            hanging = indent + "    "
        lines.extend(
            part.replace(KEEP_TOGETHER, " ")
            for part in textwrap.wrap(
                line,
                width=width,
                subsequent_indent=hanging,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def fitted_message(text: str, width: int = WIDTH) -> str:
    """A message, such as a warning's, broken into lines of at most ``width``
    characters between words."""
    return "\n".join(
        textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    )


def value_range(values: float | np.ndarray) -> str:
    """A value, or the range of the finite values of an array, as reprs show them."""
    if isinstance(values, float):
        return f"{values:.6g}"
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return "[non-finite]"
    return f"[{finite.min():.6g} to {finite.max():.6g}]"
