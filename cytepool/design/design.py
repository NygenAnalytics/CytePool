"""The design of an analysis: samples, formula, model matrix, random effects, contrasts."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..table import Table, _missing_mask
from .checks import DesignReport, check_design
from .contrasts import Contrasts, make_contrasts
from .formula import RandomEffect, _close_matches, parse_formula, parse_random_effects
from .model_matrix import (
    ModelMatrix,
    _build_model_matrix,
    _Covariate,
    _describe_rows,
    _Factor,
    _normalize_levels,
    _read_samples,
)

__all__ = ["Design"]


class Design:
    """A sample table with its design matrix, random effects and named contrasts.

    Every model in the package takes a design: ``matrix`` and ``columns`` are what a
    linear model is fitted to, ``random`` describes the random effects of a mixed
    model, and ``contrasts`` holds the comparisons to test. ``check()`` looks for the
    problems that silently ruin an analysis before anything is fitted.

    Args:
        samples: The sample table: a ``Table``, a mapping of column name to 1-D
            array, or a pandas DataFrame, with one row per sample, in the order of
            the samples of the data the design will be fitted to. The categories
            of pandas categorical columns give their level order.
        formula: The fixed effects as an R formula, such as
            ``"~ condition + batch"`` or ``"~ 0 + group"``; see ``model_matrix``.
        random: Random-effect terms in lme4 notation, such as ``"(1 | donor)"`` or
            ``"(1 | donor) + (1 | plate)"``, or a list of such strings.
        contrasts: Named contrasts over the design columns, as accepted by
            ``make_contrasts``: for example ``{"stim_vs_ctrl": "conditionstim"}``.
        levels: Level order of factors, such as ``{"condition": ["ctrl", "stim"]}``;
            the first level is the reference. Names must be columns of the sample
            table; levels of columns the design does not use are ignored.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the formula or random effects are invalid, use columns that
            do not exist or have missing values, ``levels`` does not match the data,
            or a contrast is invalid.

    Example:
        >>> design = Design(
        ...     {"condition": ["ctrl", "stim"] * 3, "donor": ["D1", "D2", "D3"] * 2},
        ...     "~ condition",
        ...     random="(1 | donor)",
        ...     contrasts={"stim_vs_ctrl": "conditionstim"},
        ... )
        >>> design.columns
        ('(Intercept)', 'conditionstim')
    """

    __slots__ = (
        "_samples",
        "_model",
        "_random",
        "_contrasts",
        "_variables",
        "_random_parts",
    )

    _samples: Table
    _model: ModelMatrix
    _random: tuple[RandomEffect, ...]
    _contrasts: Contrasts
    _variables: dict[str, _Factor | _Covariate]
    _random_parts: tuple[tuple[RandomEffect, np.ndarray, ModelMatrix], ...]

    def __init__(
        self,
        samples: Any,
        formula: str,
        *,
        random: str | Sequence[str] | None = None,
        contrasts: Mapping[str, Any] | Sequence[str] | str | None = None,
        levels: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        table, categories = _read_samples(samples)
        parsed = parse_formula(formula)
        effects = _parse_random(random)
        given = _normalize_levels(levels)
        model, variables = _build_model_matrix(table, categories, parsed, given)
        random_parts = _random_parts(table, categories, effects, given)
        object.__setattr__(self, "_samples", table)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_random", effects)
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_random_parts", random_parts)
        # Without contrasts no name is looked up, so a design whose column names
        # repeat can still be built and check() can report the repeat.
        object.__setattr__(
            self,
            "_contrasts",
            Contrasts(np.empty((len(model.columns), 0)), (), model.columns)
            if contrasts is None
            else make_contrasts(contrasts, model.columns),
        )

    # Immutability and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "Design is immutable; with_contrasts() returns a new design, and a new "
            "Design(...) gives another formula"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Design is immutable")

    def __copy__(self) -> "Design":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Design":
        return self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # The state is pickled as it is rather than rebuilt from the arguments:
        # rebuilding would lose the category order of pandas categoricals, which
        # the sample table no longer holds, and with it the reference levels.
        return (_restore, tuple(getattr(self, name) for name in Design.__slots__))

    # Contents.

    @property
    def samples(self) -> Table:
        """The sample table, one row per sample."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return len(self._samples)

    @property
    def formula(self) -> str:
        """The fixed-effects formula."""
        return self._model.formula

    @property
    def model_matrix(self) -> ModelMatrix:
        """The fixed-effects model matrix with its terms and coding."""
        return self._model

    @property
    def matrix(self) -> np.ndarray:
        """The design matrix, float64 of shape ``(n_samples, n_columns)``, read-only."""
        return self._model.matrix

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names, as R names them."""
        return self._model.columns

    @property
    def terms(self) -> tuple[str, ...]:
        """The term labels, in column order."""
        return self._model.terms

    @property
    def assign(self) -> tuple[int, ...]:
        """For each column, its term: 0 for the intercept, ``i`` for ``terms[i-1]``."""
        return self._model.assign

    @property
    def levels(self) -> Mapping[str, tuple[str, ...]]:
        """The levels of each factor of the fixed effects; the first is the reference."""
        return self._model.levels

    @property
    def random(self) -> tuple[RandomEffect, ...]:
        """The random-effect terms, in the order given."""
        return self._random

    @property
    def contrasts(self) -> Contrasts:
        """The named contrasts; index by name for one contrast's weights."""
        return self._contrasts

    def with_contrasts(
        self, contrasts: Mapping[str, Any] | Sequence[str] | str
    ) -> "Design":
        """Return the same design with these contrasts in place of the current ones.

        Args:
            contrasts: Named contrasts, as accepted by ``make_contrasts``.

        Raises:
            ValueError: If a contrast is invalid.
        """
        design = object.__new__(Design)
        for name in Design.__slots__:
            object.__setattr__(design, name, getattr(self, name))
        object.__setattr__(
            design, "_contrasts", make_contrasts(contrasts, self._model.columns)
        )
        return design

    def check(self) -> DesignReport:
        """Check the design for problems that silently ruin an analysis.

        Finds rank deficiency (naming the aliased columns and the combination each
        equals), zero residual degrees of freedom, factors confounded with each
        other or with a covariate, constant variables, empty levels, levels with a
        single sample, samples fitted exactly or nearly so, numeric columns that
        look like batch or plate codes, labels that look like missing values or
        differ only in case or spaces, contrasts that are not estimable, and random
        effects whose variances cannot be estimated. See ``cytepool.design.checks``
        for the list.

        Returns:
            The errors and warnings found, with the rank, the aliased columns and
            the leverage of every sample.
        """
        return check_design(self)

    def __repr__(self) -> str:
        n, p = self._model.shape
        lines = [
            f"Design: {n} sample{'' if n == 1 else 's'} x {p} "
            f"column{'' if p == 1 else 's'}, formula {self.formula!r}",
            "  columns: " + (", ".join(self.columns) if self.columns else "(none)"),
        ]
        if self._random:
            lines.append("  random: " + " + ".join(str(e) for e in self._random))
        if len(self._contrasts):
            lines.append("  contrasts: " + ", ".join(self._contrasts.names))
        return "\n".join(lines)


def _restore(*state: Any) -> Design:
    design = object.__new__(Design)
    for name, value in zip(Design.__slots__, state, strict=True):
        object.__setattr__(design, name, value)
    return design


def _parse_random(random: str | Sequence[str] | None) -> tuple[RandomEffect, ...]:
    if random is None:
        return ()
    if isinstance(random, str):
        texts = [random]
    elif isinstance(random, Sequence):
        texts = list(random)
    else:
        raise TypeError(
            "random must be a string such as '(1 | donor)' or a list of such strings, "
            f"got {type(random).__name__}"
        )
    effects: list[RandomEffect] = []
    for text in texts:
        effects.extend(parse_random_effects(text))
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for effect in effects:
        key = (effect.effects.formula, effect.grouping)
        if key in seen:
            raise ValueError(f"the random effect {effect} is given more than once")
        seen.add(key)
    return tuple(effects)


def _random_parts(
    table: Table,
    categories: Mapping[str, tuple[str, ...]],
    effects: tuple[RandomEffect, ...],
    levels: Mapping[str, list[Any]],
) -> tuple[tuple[RandomEffect, np.ndarray, ModelMatrix], ...]:
    """Read each random effect: group codes per sample and its effect columns."""
    parts = []
    for effect in effects:
        label = str(effect)
        absent = [name for name in effect.grouping if name not in table]
        if absent:
            raise ValueError(
                f"the random effect {label} groups by {absent[0]!r}, which is not a "
                f"column of the sample table{_close_matches(absent[0], table.columns)}"
                f"; columns are: {', '.join(table.columns)}"
            )
        for name in effect.grouping:
            missing = np.flatnonzero(_missing_mask(table[name]))
            if missing.size:
                raise ValueError(
                    f"the grouping column {name!r} of the random effect {label} has "
                    f"missing values in {_describe_rows(table, missing)}; every "
                    "sample needs a group"
                )
        keys = list(
            zip(*(table[name].tolist() for name in effect.grouping), strict=True)
        )
        index: dict[tuple[Any, ...], int] = {}
        codes = np.array([index.setdefault(key, len(index)) for key in keys], np.intp)
        columns = _build_model_matrix(
            table, categories, effect.effects, levels, f"the random effect {label}"
        )[0]
        parts.append((effect, codes, columns))
    return tuple(parts)
