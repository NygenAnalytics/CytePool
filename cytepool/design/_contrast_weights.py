"""Contrasts read from names, expressions, mappings, arrays and pandas objects, and
matched to the columns of a design or to the coefficients of a fit.

A contrast gives a weight to each column; weights that name their columns are
matched to them by name, and the others are taken in order. The contrast matrix of a
fit has one row per coefficient, as limma's ``contrasts.fit`` takes it (Smyth 2004).

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027
"""

import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .._masked_arrays import real_array, reject_masked
from .._messages import close_matches, listing, quoted

FloatArray = npt.NDArray[np.float64]


def reject_masked_contrasts(contrasts: Any, name: str) -> None:
    """Refuse numpy masked arrays in contrasts given to a function.

    That is, in the contrasts themselves (a matrix, a vector, or the weights of a
    mapping), and in the ``matrix``, ``names`` and ``columns`` of named contrasts
    (see :func:`is_named_contrasts`).
    """
    reject_masked(contrasts, name)
    if is_named_contrasts(contrasts):
        for attribute in ("matrix", "names", "columns"):
            reject_masked(getattr(contrasts, attribute), f"{name}.{attribute}")


def is_pandas(values: Any) -> bool:
    """Whether values are a pandas Series or DataFrame, whose index labels its rows.

    These attributes are pandas' own, which no column of a DataFrame can hide.
    """
    return all(hasattr(values, name) for name in ("index", "iloc", "to_numpy"))


def is_named_contrasts(values: Any) -> bool:
    """Whether values are a ``Contrasts``, or laid out like one.

    That is, they have a ``(n_columns, n_contrasts)`` ``matrix``, the contrast
    ``names`` and the ``columns`` that name its rows. A pandas DataFrame is not,
    even one with columns of those names (pandas shows columns as attributes).
    """
    return not is_pandas(values) and all(
        hasattr(values, name) for name in ("matrix", "names", "columns")
    )


def weights_on_columns(
    weights: Any, columns: Sequence[str], *, label: str, target: str, remedy: str
) -> Any:
    """Contrast weights with their rows in the order of ``columns``.

    This decides, for every function that takes contrast weights, which weights
    name their rows. A pandas Series or DataFrame names them by its index when
    that holds strings, and named contrasts (see :func:`is_named_contrasts`) by
    their ``columns``. Such rows are matched to ``columns`` by name, so they must
    be exactly those names, in any order. An index that is pandas' default 0, 1,
    ..., n - 1 names no rows; any other index is an error, because its labels
    could be meant either as names or as positions. Weights whose rows are not
    named are returned as they are, to be read in the order of ``columns``.

    Args:
        weights: A vector, a matrix with one row per column, or named contrasts.
        columns: The names the rows must match, in the order wanted; unique.
        label: The weights in messages, such as "the contrasts".
        target: ``columns`` in messages, such as "the fit's coefficients".
        remedy: How to fix a mismatch, ending its message.

    Returns:
        The weights as an array with rows in the order of ``columns``, or
        ``weights`` unchanged (a pandas object as an array) when its rows are not
        named.

    Raises:
        TypeError: If ``weights`` has a matrix and columns but no contrast names,
            as a design does.
        ValueError: If the rows are named but not exactly by ``columns``, a name
            repeats, or a pandas index neither holds strings nor is the default.
    """
    pandas = is_pandas(weights)
    if pandas:
        values = weights.to_numpy()
        names = list(weights.index)
        if not all(isinstance(name, str) for name in names):
            if _default_positions(names):
                return values
            raise ValueError(
                f"the index of {label} must hold the names of {target}, or be the "
                f"default 0, 1, ..., n - 1 for weights in the order of {target}; got "
                + quoted(names)
            )
    elif is_named_contrasts(weights):
        values = np.asarray(weights.matrix)
        names = list(weights.columns)
        if values.ndim != 2 or values.shape[0] != len(names):
            raise ValueError(
                f"{label} must have one row of weights per name in their columns "
                f"({len(names)}), got a matrix of shape {values.shape}"
            )
    elif hasattr(weights, "matrix") and hasattr(weights, "columns"):
        raise TypeError(
            f"a {type(weights).__name__} has a matrix and columns but no contrast "
            f"names, as a design does, so it cannot give {label}; pass the contrasts "
            "of a design (design.contrasts), contrasts from make_contrasts, or the "
            "weights alone"
        )
    else:
        return weights
    rows: dict[Any, int] = {}
    for row, name in enumerate(names):
        if name in rows:
            raise ValueError(
                f"{name!r} labels more than one row of {label}; each row must weigh "
                f"a different one of {target}"
            )
        rows[name] = row
    missing = [name for name in columns if name not in rows]
    known = set(columns)
    unexpected = [name for name in names if name not in known]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"{quoted(missing)} {_is_or_are(missing)} missing")
        if unexpected:
            hinted = [
                repr(name) + close_matches(str(name), [str(m) for m in missing])
                for name in unexpected
            ]
            problems.append(f"{listing(hinted)} {_is_or_are(hinted)} not among them")
        if pandas and not unexpected:
            remedy += (
                "; to give the missing ones zero weight, reindex the weights with "
                "fill_value=0"
            )
        raise ValueError(
            f"{label} must weigh exactly {target} ({quoted(columns)}), in any "
            f"order, but {' and '.join(problems)}; {remedy}"
        )
    return values[[rows[name] for name in columns]]


def _default_positions(labels: list[Any]) -> bool:
    """Whether labels are 0, 1, ..., n - 1, as pandas' default RangeIndex."""
    return all(
        isinstance(label, numbers.Integral)
        and not isinstance(label, bool)
        and label == position
        for position, label in enumerate(labels)
    )


def _is_or_are(items: Sequence[Any]) -> str:
    return "is" if len(items) == 1 else "are"


def contrast_matrix(
    contrasts: Any, names: Sequence[str] | None, coefficient_names: tuple[str, ...]
) -> tuple[FloatArray, tuple[str, ...]]:
    """The contrast matrix, rows in the order of the coefficients, and its names.

    Weights that name their coefficients are matched to them by name, by the
    rule of :func:`weights_on_columns`; the others
    are taken in order.
    """
    reject_masked_contrasts(contrasts, "contrasts")
    reject_masked(names, "names")
    n_coefficients = len(coefficient_names)
    remedy = _contrasts_remedy(coefficient_names)
    named = is_named_contrasts(contrasts)
    if (named or isinstance(contrasts, Mapping)) and names is not None:
        raise ValueError(
            "names must not be given when contrasts is a mapping or a Contrasts, "
            "which carry their own names"
        )
    if named:
        weights = weights_on_columns(
            contrasts,
            coefficient_names,
            label="the contrasts",
            target="the fit's coefficients",
            remedy=remedy,
        )
        matrix = real_array(weights, "contrasts")
        if matrix.shape[1] == 0:
            raise ValueError("contrasts must hold at least one contrast")
        contrast_names = checked_names(
            contrasts.names, matrix.shape[1], "contrast names"
        )
    elif isinstance(contrasts, Mapping):
        if not contrasts:
            raise ValueError("contrasts must hold at least one contrast")
        vectors = []
        for key, vector in contrasts.items():
            weights = weights_on_columns(
                vector,
                coefficient_names,
                label=f"contrast {key!r}",
                target="the fit's coefficients",
                remedy=remedy,
            )
            values = real_array(weights, f"contrast {key!r}")
            if values.shape != (n_coefficients,):
                raise ValueError(
                    f"contrast {key!r} must have one value per coefficient "
                    f"({n_coefficients}), got shape {values.shape}"
                )
            vectors.append(values)
        matrix = np.column_stack(vectors)
        contrast_names = checked_names(
            list(contrasts.keys()), matrix.shape[1], "contrast names"
        )
    else:
        # The columns of a DataFrame, or the name of a Series, name its contrasts;
        # the index names its rows.
        columns = getattr(contrasts, "columns", None)
        if is_pandas(contrasts) and columns is None:
            name = getattr(contrasts, "name", None)
            columns = [name] if isinstance(name, str) else None
        weights = weights_on_columns(
            contrasts,
            coefficient_names,
            label="the contrasts",
            target="the fit's coefficients",
            remedy=remedy,
        )
        matrix = real_array(weights, "contrasts")
        if matrix.ndim == 1:
            matrix = matrix[:, np.newaxis]
        if matrix.ndim != 2 or matrix.shape[0] != n_coefficients:
            raise ValueError(
                "contrasts must have one row per coefficient of the fit "
                f"({n_coefficients}), got shape {matrix.shape}"
            )
        if matrix.shape[1] == 0:
            raise ValueError("contrasts must hold at least one contrast")
        if names is not None:
            contrast_names = checked_names(names, matrix.shape[1], "names")
            if columns is not None:
                check_renaming(
                    contrast_names, [str(c) for c in columns], "names", "the contrasts"
                )
        elif columns is not None:
            contrast_names = checked_names(
                [str(c) for c in columns], matrix.shape[1], "contrast columns"
            )
        else:
            contrast_names = tuple(f"contrast{j + 1}" for j in range(matrix.shape[1]))
    if not np.isfinite(matrix).all():
        raise ValueError("contrasts must be finite")
    return np.array(matrix, dtype=np.float64, order="C"), contrast_names


def _contrasts_remedy(coefficient_names: tuple[str, ...]) -> str:
    """How to make contrasts that match a fit, for error messages."""
    remedy = "make them with make_contrasts(..., fit.coefficient_names)"
    if coefficient_names == tuple(f"x{j + 1}" for j in range(len(coefficient_names))):
        remedy += (
            "; the fit has the default names x1, x2, ..., so fit the Design itself, "
            "not its matrix, to keep the names of its columns"
        )
    return remedy


def check_renaming(
    given: tuple[str, ...], existing: Sequence[str], label: str, owner: str
) -> None:
    """Refuse new names that give a column the name another column already has.

    Names given by position rename the columns in order. Coefficients and
    contrasts are matched by name downstream, so moving a name to another
    column would silently swap what it refers to.
    """
    existing = list(existing)
    for position, name in enumerate(given):
        own = position < len(existing) and existing[position] == name
        if name in existing and not own:
            raise ValueError(
                f"{label} give column {position + 1} the name {name!r}, the name of "
                f"column {existing.index(name) + 1} of {owner}; {label} rename "
                f"the columns in order ({', '.join(map(repr, existing))}), so a name "
                "already in use may only label its own column"
            )


def checked_names(names: Any, count: int, label: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{label} must be a sequence of strings, not a single string")
    checked = tuple(names)
    if not all(isinstance(name, str) for name in checked):
        raise TypeError(f"{label} must be strings")
    checked = tuple(str(name) for name in checked)
    if len(checked) != count:
        raise ValueError(f"{label} must have {count} entries, got {len(checked)}")
    if len(set(checked)) != count:
        raise ValueError(f"{label} must be unique, got {checked}")
    return checked
