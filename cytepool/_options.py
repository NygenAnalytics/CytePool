"""Checks of the options of public functions: flags, and choices among names."""

from typing import Any

import numpy as np


def checked_flag(value: Any, name: str, *, expected: str = "True or False") -> bool:
    """An option that is True or False, as a bool.

    Args:
        value: The option as given; a numpy bool is accepted.
        name: The option's name, for the message.
        expected: What the message says the option must be: "True or False", or
            "a bool" as the statistics' messages say.

    Raises:
        TypeError: If ``value`` is not a bool.
    """
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")
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
