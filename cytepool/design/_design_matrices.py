"""The design matrix of a fit, its column names, its rank, and the refusal of random
effects by fits that have none.

A fit takes a ``Design``, a ``ModelMatrix`` or another object with ``matrix`` and
``columns``, a pandas DataFrame, or a matrix, finite and with at least one row and one
column; its columns are named by the design, by ``coefficient_names`` or as x1, x2,
... A column that is a linear combination of the preceding columns, by the rule of
R's ``qr()``, makes the design rank deficient (Chambers and Hastie 1992). Fits of
fixed effects refuse a ``Design`` with random effects, saying what can be fitted.

References:
    Chambers JM, Hastie TJ (1992). Statistical models. Chapter 2 of Statistical Models
    in S, eds Chambers JM and Hastie TJ. Wadsworth & Brooks/Cole, Pacific Grove.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .._linear_algebra import column_space
from .._masked_arrays import real_array, reject_masked, unmasked_array
from ._contrast_weights import check_renaming, checked_names, is_pandas
from .checks import _same_partition
from .design import Design
from .formula import RandomEffect, quote_name

FloatArray = npt.NDArray[np.float64]


def _design_columns(design: Any) -> tuple[Any, Any]:
    """The matrix and column names of a design, as given.

    A design is a ``Design``, a ``ModelMatrix`` or another object with ``matrix``
    and ``columns`` attributes, a table with ``columns`` such as a pandas
    DataFrame, or a matrix. A function that fits a model also passes the design
    to :func:`refuse_dropped_random_effects`.

    Returns:
        The matrix, not yet checked or converted, and the column names, or None
        when the design has none.

    Raises:
        TypeError: If the design, or its matrix or columns, is a numpy masked array.
    """
    reject_masked(design, "design")
    if is_pandas(design):
        # Its columns, which pandas also shows as attributes, are the design's.
        return design, design.columns
    if hasattr(design, "matrix") and hasattr(design, "columns"):
        reject_masked(design.matrix, "design.matrix")
        reject_masked(design.columns, "design.columns")
        return design.matrix, design.columns
    columns = getattr(design, "columns", None)
    reject_masked(columns, "design.columns")
    return design, columns


def refuse_dropped_random_effects(
    design: Any, function: str, block: Any, correlation: float
) -> None:
    """Refuse a fit that would drop the random effects of a ``Design``.

    Every function that fits fixed effects and, through a correlation within
    blocks, at most one random intercept calls this; ``fit_mixed_model``, which
    estimates random intercepts, does not. A ``Design`` with random effects is
    accepted only when they are one random intercept, such as ``(1 | donor)``,
    whose grouping splits the samples as the blocks do, with a correlation of at
    least zero. Only a ``Design`` carries random effects; any other design is a
    matrix.

    Args:
        design: The design given to the fit.
        function: The name of the fitting function, for the message.
        block: The block index of each sample, already checked, or None.
        correlation: The correlation within blocks, already checked.

    Raises:
        ValueError: If the fit would drop a random effect.
    """
    if not isinstance(design, Design) or not design.random:
        return
    effects = design.random
    if block is None:
        reason = ""
    elif len(effects) != 1 or not is_random_intercept(effects[0]):
        reason = "; block= can stand only for a single random intercept"
    elif not _same_partition(np.asarray(block), design._random_parts[0][1]):
        reason = f"; block= does not group the samples as {effects[0]} does"
    elif correlation < 0.0:
        reason = (
            "; a random intercept gives the samples of a group a correlation of at "
            f"least zero, so correlation={correlation:g} does not stand for "
            f"{effects[0]}"
        )
    else:
        return
    raise ValueError(_random_effects_refusal(design, function, reason))


def refuse_random_effects(design: Any, function: str) -> None:
    """Refuse a fit of fixed effects only for a ``Design`` with random effects.

    Unlike :func:`refuse_dropped_random_effects`, for functions that take no
    ``block=`` and ``correlation=``, so that every random effect would be dropped.

    Raises:
        ValueError: If ``design`` is a ``Design`` with random effects.
    """
    if isinstance(design, Design) and design.random:
        raise ValueError(_random_effects_refusal(design, function, "", block=False))


def is_random_intercept(effect: RandomEffect) -> bool:
    return effect.effects.intercept and not effect.effects.terms


def _random_effects_refusal(
    design: Design, function: str, reason: str, *, block: bool = True
) -> str:
    """Why a fit refuses the random effects of a design, and what can be fitted.

    ``block`` says whether the function fits a random intercept with a known
    correlation through ``block=`` and ``correlation=``. Random intercepts, alone or
    with others, are estimated by ``fit_mixed_model``, which the message names
    first; random slopes are not estimated by any function yet.
    """
    effects = design.random
    listed = " + ".join(str(effect) for effect in effects)
    noun = "random effect" if len(effects) == 1 else "random effects"
    shared = len({effect.grouping for effect in effects}) == 1
    group = effects[0].group if shared else None
    single = shared and len(effects[0].grouping) == 1
    # In a formula a name such as "cell type" needs backticks; lme4 names a
    # combination of variables with them already.
    term = quote_name(effects[0].grouping[0]) if single else group
    intercept = len(effects) == 1 and is_random_intercept(effects[0])
    intercepts = all(is_random_intercept(effect) for effect in effects)
    if group is None:
        options = [
            "a fixed-effects analysis, with the Design built without random= and the "
            "grouping variables in the formula when the conditions compared vary "
            "within their groups (left out, the samples of a group count as "
            "independent)"
        ]
    else:
        options = [
            "a fixed-effects analysis, with the Design built without random= and "
            f"{term} in the formula when the conditions compared vary within each "
            f"{group} (left out, the samples of a {group} count as independent)"
        ]
        if intercept and block:
            labels = (
                f"block=design.samples[{group!r}]"
                if single
                else f"block= (the {group} of each sample)"
            )
            options.append(
                "this random intercept with a known correlation between the samples "
                f"of each {group}, given {labels} and correlation= ({function} does "
                "not estimate the correlation)"
            )
        by = (
            f"({group!r})"
            if single
            else f", by a column holding the {group} of each sample"
        )
        options.append(
            f"when the conditions compared differ between levels of {group}, not "
            f"within them, a fit of one profile per {group}, pooled first with "
            f"Cube.collapse_samples{by}"
        )
    fitted = (
        options[0]
        if len(options) == 1
        else ("; ".join(options[:-1]) + "; or, " + options[-1])
    )
    fits = (
        f"{function} fits fixed effects and, through block= and correlation=, one "
        "random intercept with a known correlation"
        if block
        else f"{function} fits fixed effects only"
    )
    dropped = f"{fits}, so it would drop the {noun} {listed} of this design{reason}."
    if intercepts:
        return (
            f"{dropped} fit_mixed_model estimates {'it' if len(effects) == 1 else 'them'}"
            " by REML, and fixed_effect_tests tests contrasts of the fixed effects with "
            f"Satterthwaite's degrees of freedom. Other analyses: {fitted}."
        )
    return (
        f"{dropped} Random slopes are not available yet: fit_mixed_model fits random "
        f"intercepts only. What can be fitted now: {fitted}."
    )


def design_matrix(
    design: Any, coefficient_names: Sequence[str] | None
) -> tuple[FloatArray, tuple[str, ...]]:
    matrix, columns = _design_columns(design)
    reject_masked(coefficient_names, "coefficient_names")
    x = real_array(unmasked_array(matrix, "design"), "design")
    if x.ndim != 2:
        raise ValueError(
            "design must be a 2-D array of shape (n_samples, n_coefficients), "
            f"got shape {x.shape}"
        )
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(
            f"design must have at least one row and one column, got shape {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("design must be finite; it contains NaN or infinite values")
    if coefficient_names is not None:
        names = checked_names(coefficient_names, x.shape[1], "coefficient_names")
        if columns is not None:
            check_renaming(
                names, [str(c) for c in columns], "coefficient_names", "the design"
            )
    elif columns is not None:
        names = checked_names([str(c) for c in columns], x.shape[1], "design columns")
    else:
        names = tuple(f"x{j + 1}" for j in range(x.shape[1]))
    # A copy, so that the fit does not change with the caller's array.
    return np.array(x, dtype=np.float64, order="C"), names


def aliased_columns(x: FloatArray) -> list[int]:
    """Indices of the columns of x that are linear combinations of earlier ones.

    The rule is R's ``qr()`` rule, as in :func:`column_space`; the compiled
    :func:`~cytepool._linear_algebra.householder_qr` applies the same tolerance to each
    feature.
    """
    return list(column_space(x).aliased)


def require_full_rank(x: FloatArray, names: Sequence[str]) -> None:
    """Refuse a design whose columns are not linearly independent, by R's rule.

    Raises:
        ValueError: Naming the columns that are linear combinations of the preceding
            columns.
    """
    aliased = aliased_columns(x)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            f"design is not of full column rank: column(s) {listed} are linear "
            "combinations of the preceding columns; remove them or reparametrise the "
            "design"
        )
