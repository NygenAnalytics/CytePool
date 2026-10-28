"""Checks of the options of public functions: flags, choices among names, numbers,
random generators.

Each check takes the option as the caller gave it and its name, and returns it as a
bool, str, float, int or numpy random generator, or raises TypeError for a value of
the wrong type and ValueError for one out of range. The messages name the option, say
what it must be and what it was: a TypeError ends with the type it got, and a
ValueError with the value.
"""

import math
import numbers
import operator
from typing import Any

import numpy as np

from ._masked_arrays import reject_masked


def checked_flag(value: Any, name: str) -> bool:
    """An option that is True or False, as a bool.

    Args:
        value: The option as given; a numpy bool is accepted.
        name: The option's name, for the message.

    Raises:
        TypeError: If ``value`` is not True or False.
    """
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be True or False, got {type(value).__name__}")
    return bool(value)


def checked_choice(value: Any, name: str, options: tuple[str, ...]) -> str:
    """An option that names one of ``options``.

    Args:
        value: The option as given.
        name: The option's name, for the message.
        options: The names it may take.

    Raises:
        ValueError: If ``value`` is a string that is not one of ``options``.
        TypeError: If ``value`` is not a string.
    """
    if isinstance(value, str) and value in options:
        return value
    error = ValueError if isinstance(value, str) else TypeError
    raise error(f"{name} must be one of {', '.join(map(repr, options))}, got {value!r}")


def real_number(value: Any, name: str, *, finite: bool | None = None) -> float:
    """A real number, as a float; a bool is not one.

    Args:
        value: The option as given.
        name: The option's name, for the message.
        finite: True refuses NaN and infinity, False refuses NaN, and None
            accepts any float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``finite`` refuses its value.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if finite is not None and (math.isnan(number) or (finite and math.isinf(number))):
        expected = "finite" if finite else "a number"
        raise ValueError(f"{name} must be {expected}, got {number}")
    return number


def non_negative_number(value: Any, name: str) -> float:
    """A finite real number of at least 0."""
    number = real_number(value, name, finite=True)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def positive_finite_number(value: Any, name: str) -> float:
    """A finite real number above 0."""
    number = real_number(value, name)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {number}")
    return number


def integral_number(value: Any, name: str, *, expected: str = "an integer") -> int:
    """An integer (``numbers.Integral``), as an int; a bool is not one.

    Args:
        value: The option as given.
        name: The option's name, for the message.
        expected: What the message says the option must be.

    Raises:
        TypeError: If ``value`` is not an integer.
    """
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")
    return operator.index(value)


def positive_integer(value: Any, name: str, *, expected: str = "an integer") -> int:
    """An integer of at least 1, as :func:`integral_number` reads it."""
    number = integral_number(value, name, expected=expected)
    if number < 1:
        raise ValueError(f"{name} must be at least 1, got {number}")
    return number


def index_integer(value: Any, name: str, *, expected: str) -> int:
    """An integer as ``operator.index`` reads it, which takes a 0-d integer array.

    The caller refuses a bool first, with its own message. ``operator.index``
    accepts a 0-d masked integer and ignores its mask, so masked values are refused
    before it.

    Raises:
        TypeError: If ``value`` is masked or is not an integer.
    """
    reject_masked(value, name)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be {expected}, got {type(value).__name__}"
        ) from None


def random_generator(rng: Any, name: str = "rng") -> np.random.Generator:
    """A numpy random generator: ``rng`` itself, one seeded by it, or a fresh one.

    Args:
        rng: A ``numpy.random.Generator``, which is used as it is; a non-negative
            integer, which seeds a new one; or None for a fresh, unseeded generator.
        name: The option's name, for the message.

    Raises:
        TypeError: If ``rng`` is none of these, or is a numpy masked array.
        ValueError: If the seed is negative.
    """
    reject_masked(rng, name)
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool | np.bool_) or not isinstance(rng, numbers.Integral):
        raise TypeError(
            f"{name} must be a numpy.random.Generator, an integer seed or None, got "
            f"{type(rng).__name__}"
        )
    seed = operator.index(rng)
    if seed < 0:
        raise ValueError(f"{name} must be a non-negative seed, got {seed}")
    return np.random.default_rng(seed)
