"""The designs of linear mixed models with random intercepts, read and checked.

A mixed model y = X beta + Z_1 b_1 + ... + Z_K b_K + e, with one random intercept per
level of each grouping factor (``(1 | donor)`` in lme4; Bates et al. 2015), is read
from a ``Design`` into the parts that the fits take, as a ``MixedDesign``: the
fixed-effects matrix X of full column rank, its column names, the level of each
sample for every grouping factor, numbered from 0, the number of levels of each
factor and the factors' names. The mixed models, dream and the workflows on cubes
share:

- ``random_intercepts``: refuses anything but a ``Design`` whose random effects are
  random intercepts.
- ``mixed_design``: the parts of a whole design, refusing random effects that cannot
  be estimated and fixed effects that are not of full column rank.
- ``mixed_design_of_samples``: the parts of some of a design's samples, checked on
  those samples as ``Design.check`` checks the design.
- ``maximum_likelihood``: whether a ``method`` asks for maximum likelihood or REML.

Both forms of the parts refuse fixed and random effects that together span every
sample, which would reproduce every feature exactly.

References:
    Bates D, Maechler M, Bolker B, Walker S (2015). Fitting linear mixed-effects models
    using lme4. Journal of Statistical Software 67(1), 1-48. doi:10.18637/jss.v067.i01
"""

from dataclasses import replace
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from cytepool._linear_algebra import column_space
from cytepool._masked_arrays import reject_masked
from cytepool.design._design_matrices import is_random_intercept
from cytepool.design.checks import _aliasing, _random_checks
from cytepool.design.design import Design

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class MixedDesign(NamedTuple):
    """The parts of a mixed model's design that the fits read."""

    x: FloatArray  # (n_samples, n_coefficients) fixed effects, of full column rank
    names: tuple[str, ...]  # the names of the fixed effects
    codes: IntArray  # (n_components, n_samples) level of each sample, from 0
    n_levels: IntArray  # the number of levels of each grouping factor
    components: tuple[str, ...]  # the names of the grouping factors


class _SomeSamples:
    """What the random-effect checks of a design read, for some of its samples."""

    def __init__(self, n_samples: int, random_parts: tuple[Any, ...]) -> None:
        self.n_samples = n_samples
        self._random_parts = random_parts


def mixed_design(design: Any) -> MixedDesign:
    """The fixed-effects matrix, its column names, and the random intercepts.

    Returns:
        The design matrix, its column names, the level of each sample for every
        grouping factor (``(n_components, n_samples)``, levels numbered from 0), the
        number of levels of each factor, and the factors' names.
    """
    random_intercepts(design)
    report = design.check()
    for issue in report.errors:
        if issue.kind.startswith("random_"):
            raise ValueError(f"the random effects cannot be estimated: {issue}")
    if report.aliased:
        raise ValueError(
            "the fixed effects are not of full column rank: column(s) "
            + ", ".join(repr(name) for name in report.aliased)
            + " are linear combinations of the preceding columns; remove them or "
            "reparametrise the design (see design.check())"
        )
    x = np.array(design.matrix, dtype=np.float64, order="C")
    parts = design._random_parts
    codes = np.array([part[1] for part in parts], dtype=np.int64).reshape(
        len(parts), x.shape[0]
    )
    return _spanning_checked(x, tuple(design.columns), codes, design)


def mixed_design_of_samples(
    design: Design, rows: npt.NDArray[np.intp], x: FloatArray, names: tuple[str, ...]
) -> MixedDesign:
    """The design parts of some samples of a design, checked as for the design.

    For a fit to a subset of a design's samples, such as those a workflow keeps in
    one group of a cube: the fixed effects are ``x``, rows of the design's matrix
    for those samples restricted to columns of full rank, and the random intercepts
    are the design's, with the levels those samples have, numbered from 0 in order
    of appearance. The random effects are checked on these samples as
    ``design.check()`` checks them on all: a factor with one level left, levels with
    one sample each, levels that the fixed effects separate, or two factors that
    group the samples alike cannot be estimated.

    Args:
        design: A ``Design`` with random intercepts, as fit_mixed_model takes.
        rows: The positions of the samples among the design's, in fitting order.
        x: ``(len(rows), n_coefficients)`` fixed effects of the samples.
        names: The names of the columns of ``x``.

    Raises:
        ValueError: As fit_mixed_model, for these samples; the message says why.
    """
    random_intercepts(design)
    x = np.array(x, dtype=np.float64, order="C")
    aliasing = _aliasing(x)
    if aliasing.aliased:
        raise ValueError(
            "the fixed effects are not of full column rank: column(s) "
            + ", ".join(repr(names[j]) for j in aliasing.aliased)
            + " are linear combinations of the preceding columns"
        )
    parts = []
    for effect, codes, effects in design._random_parts:
        index: dict[int, int] = {}
        recoded = np.array(
            [index.setdefault(int(code), len(index)) for code in codes[rows]],
            dtype=np.intp,
        )
        parts.append((effect, recoded, replace(effects, matrix=effects.matrix[rows])))
    errors: list[Any] = []
    # The checks read the number of samples and the random effects of a Design.
    _random_checks(
        cast(Design, _SomeSamples(len(rows), tuple(parts))), aliasing, errors
    )
    for issue in errors:
        raise ValueError(f"the random effects cannot be estimated: {issue}")
    codes = np.array([part[1] for part in parts], dtype=np.int64).reshape(
        len(parts), len(rows)
    )
    return _spanning_checked(x, tuple(names), codes, design)


def random_intercepts(design: Any) -> None:
    """Refuse anything but a ``Design`` whose random effects are random intercepts."""
    reject_masked(design, "design")
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design with random effects, such as Design(samples, "
            f"'~ condition', random='(1 | donor)'); got {type(design).__name__}"
        )
    if not design.random:
        raise ValueError(
            "the design has no random effects, so there is no mixed model to fit; "
            "fit it with fit_linear_model, or build it with random effects such as "
            "random='(1 | donor)'"
        )
    for effect in design.random:
        if not is_random_intercept(effect):
            raise ValueError(
                f"fit_mixed_model fits random intercepts, such as (1 | "
                f"{effect.group}), but the random effect {effect} has random slopes, "
                "which are not available yet"
            )


def _spanning_checked(
    x: FloatArray, names: tuple[str, ...], codes: IntArray, design: Design
) -> MixedDesign:
    """The parts, after refusing fixed and random effects that span every sample."""
    n_samples = x.shape[0]
    n_levels = codes.max(axis=1) + 1
    columns = [x]
    for row, count in zip(codes, n_levels):
        indicator = np.zeros((n_samples, int(count)))
        indicator[np.arange(n_samples), row] = 1.0
        columns.append(indicator)
    rank = column_space(np.column_stack(columns)).rank
    if rank >= n_samples:
        raise ValueError(
            f"the fixed and random effects together span every sample ({rank} "
            f"dimensions for {n_samples} samples), so they reproduce every feature "
            "exactly; fit_mixed_model does not fit such designs yet: add samples, or "
            "remove a random effect or a covariate"
        )
    components = tuple(effect.group for effect in design.random)
    return MixedDesign(x, names, codes, n_levels, components)


def maximum_likelihood(method: Any) -> bool:
    """Whether ``method`` asks for maximum likelihood ("ml") rather than REML."""
    if isinstance(method, str) and method in ("reml", "ml"):
        return method == "ml"
    error = ValueError if isinstance(method, str) else TypeError
    raise error(
        "method must be 'reml' (restricted maximum likelihood, lme4's REML = TRUE) "
        f"or 'ml' (maximum likelihood, REML = FALSE), got {method!r}"
    )
