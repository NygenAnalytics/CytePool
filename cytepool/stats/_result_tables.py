"""Tables of the tests of fits on arrays, with the columns of the result tables.

``ModeratedFit.table``, ``QuasiLikelihoodTest.table`` and ``DESeq2Results.table``
lay out the statistics of every feature as the tables of the workflows on cubes lay
them out, a ``feature_id`` column first, so that an analysis built from the building
blocks reads like one of a cube and its calls can be found by gene.
"""

from typing import Any

import numpy as np

from cytepool._columns import text_column
from cytepool._masked_arrays import reject_masked, unmasked_array
from cytepool._messages import quoted
from cytepool._options import integral_number


def feature_id_column(feature_ids: Any, n: int) -> np.ndarray:
    """The ``feature_id`` column of a table of ``n`` features: the ids given, one per
    feature, or ``feature_0``, ``feature_1``, ... as ``from_bulk`` names the
    features of counts without ids.

    Raises:
        TypeError: If an id is not a string, or ``feature_ids`` is or holds a numpy
            masked array.
        ValueError: If there is not one id per feature, or an id is missing or
            empty.
    """
    if feature_ids is None:
        return text_column([f"feature_{j}" for j in range(n)])
    if isinstance(feature_ids, str):
        raise TypeError(
            "feature_ids must give one id per feature, as a list or an array, not a "
            "single string"
        )
    array = unmasked_array(feature_ids, "feature_ids")
    if array.ndim != 1 or array.size != n:
        raise ValueError(
            f"feature_ids must hold one id per feature ({n}), got shape {array.shape}"
        )
    ids = array.tolist()
    for position, value in enumerate(ids):
        if not isinstance(value, str):
            raise TypeError(
                f"feature_ids must hold text ids, got {type(value).__name__} "
                f"{value!r} at position {position}; convert numeric ids to text first"
            )
        if not value:
            raise ValueError(f"feature_ids has an empty id at position {position}")
    return text_column(ids)


def coefficient_position(value: Any, names: tuple[str, ...]) -> int:
    """The position of a coefficient or contrast of a fit, given by name or position,
    or the only one when ``value`` is None.

    Raises:
        ValueError: If ``value`` is None and the fit has several.
        KeyError: If no coefficient has the name.
        IndexError: If the position is out of range.
        TypeError: If ``value`` is neither a name nor a position.
    """
    if value is None:
        if len(names) != 1:
            raise ValueError(
                f"the fit has {len(names)} coefficients or contrasts; name one with "
                f"coefficient, one of {quoted(list(names))}"
            )
        return 0
    reject_masked(value, "coefficient")
    if isinstance(value, str):
        if value not in names:
            raise KeyError(
                f"no coefficient or contrast {value!r}; they are: {quoted(list(names))}"
            )
        return names.index(value)
    position = integral_number(value, "coefficient", expected="a name or a position")
    if not -len(names) <= position < len(names):
        raise IndexError(
            f"coefficient position {position} is out of range for {len(names)} "
            "coefficients"
        )
    return position % len(names)
