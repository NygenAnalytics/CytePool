"""The design of an analysis: samples, formula, model matrix, random effects, contrasts."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._masked_arrays import reject_masked
from ..table import Table, _missing_mask
from .checks import DesignReport, _same_partition, check_design
from .contrasts import Contrasts, _is_pandas, make_contrasts
from .formula import (
    RandomEffect,
    _close_matches,
    parse_formula,
    parse_random_effects,
    quote_name,
)
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
    problems that silently ruin an analysis before anything is fitted. No function
    estimates random effects yet: ``fit_linear_model`` fits one random intercept
    with a known correlation, given ``block=`` that groups the samples as the
    intercept does, and refuses a design whose random effects it would otherwise
    drop.

    Args:
        samples: The sample table: a ``Table``, a mapping of column name to 1-D
            array, or a pandas DataFrame, with one row per sample, in the order of
            the samples of the data the design will be fitted to. The categories
            of pandas categorical columns give their level order.
        formula: The fixed effects as an R formula, such as
            ``"~ condition + batch"`` or ``"~ 0 + group"``; see ``model_matrix``.
        random: Random-effect terms in lme4 notation, such as ``"(1 | donor)"`` or
            ``"(1 | donor) + (1 | plate)"``, or a list of such strings. ``check()``
            examines them, but no function estimates them yet.
        contrasts: Named contrasts over the design columns, as accepted by
            ``make_contrasts``: for example ``{"stim_vs_ctrl": "conditionstim"}``.
        levels: Level order of factors, such as ``{"condition": ["ctrl", "stim"]}``;
            the first level is the reference. Names must be columns of the sample
            table; levels of columns the design does not use are ignored.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
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
        table, categories = _read_samples(samples, "samples")
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
        """The term of each column, 0 for the intercept, ``i`` for ``terms[i-1]``."""
        return self._model.assign

    @property
    def levels(self) -> Mapping[str, tuple[str, ...]]:
        """The levels of each factor of the fixed effects; the first is the reference."""
        return self._model.levels

    @property
    def random(self) -> tuple[RandomEffect, ...]:
        """The random-effect terms, in the order given; none is estimated yet."""
        return self._random

    @property
    def contrasts(self) -> Contrasts:
        """The named contrasts; index by name for one contrast's weights.

        Pass them, or ``contrasts.select(...)``, to ``contrasts_fit``, which
        matches them to the fit's coefficients by name; the vectors that indexing
        returns carry no names and are read in order.
        """
        return self._contrasts

    def with_contrasts(
        self, contrasts: Mapping[str, Any] | Sequence[str] | str
    ) -> "Design":
        """Return the same design with these contrasts in place of the current ones.

        Args:
            contrasts: Named contrasts, as accepted by ``make_contrasts``.

        Raises:
            TypeError: If a contrast has the wrong type, or is or holds a numpy
                masked array.
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


def _design_columns(design: Any) -> tuple[Any, Any]:
    """The matrix and column names of a design, as given.

    A design is a ``Design``, a ``ModelMatrix`` or another object with ``matrix``
    and ``columns`` attributes, a table with ``columns`` such as a pandas
    DataFrame, or a matrix. A function that fits a model also passes the design
    to :func:`_refuse_dropped_random_effects`.

    Returns:
        The matrix, not yet checked or converted, and the column names, or None
        when the design has none.

    Raises:
        TypeError: If the design, or its matrix or columns, is a numpy masked array.
    """
    reject_masked(design, "design")
    if _is_pandas(design):
        # Its columns, which pandas also shows as attributes, are the design's.
        return design, design.columns
    if hasattr(design, "matrix") and hasattr(design, "columns"):
        reject_masked(design.matrix, "design.matrix")
        reject_masked(design.columns, "design.columns")
        return design.matrix, design.columns
    columns = getattr(design, "columns", None)
    reject_masked(columns, "design.columns")
    return design, columns


def _refuse_dropped_random_effects(
    design: Any, function: str, block: Any, correlation: float
) -> None:
    """Refuse a fit that would drop the random effects of a ``Design``.

    Every function that fits a model calls this. Such a fit takes fixed effects
    and, through a correlation within blocks, at most one random intercept: a
    ``Design`` with random effects is accepted only when they are one random
    intercept, such as ``(1 | donor)``, whose grouping splits the samples as the
    blocks do, with a correlation of at least zero. Only a ``Design`` carries
    random effects; any other design is a matrix.

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
    elif len(effects) != 1 or not _is_random_intercept(effects[0]):
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


def _refuse_random_effects(design: Any, function: str) -> None:
    """Refuse a fit of fixed effects only for a ``Design`` with random effects.

    Unlike :func:`_refuse_dropped_random_effects`, for functions that take no
    ``block=`` and ``correlation=``, so that every random effect would be dropped.

    Raises:
        ValueError: If ``design`` is a ``Design`` with random effects.
    """
    if isinstance(design, Design) and design.random:
        raise ValueError(_random_effects_refusal(design, function, "", block=False))


def _is_random_intercept(effect: RandomEffect) -> bool:
    return effect.effects.intercept and not effect.effects.terms


def _random_effects_refusal(
    design: Design, function: str, reason: str, *, block: bool = True
) -> str:
    """Why a fit refuses the random effects of a design, and what it can fit now.

    ``block`` says whether the function fits a random intercept with a known
    correlation through ``block=`` and ``correlation=``.
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
    intercept = len(effects) == 1 and _is_random_intercept(effects[0])
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
                f"of each {group}, given {labels} and correlation= (CytePool does not "
                "estimate the correlation)"
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
    return (
        f"{fits}, so it would drop the {noun} {listed} of this design{reason}. Fitting "
        "random effects (mixed models) is planned but not available yet. What it can "
        f"fit now: {fitted}."
    )


def _parse_random(random: str | Sequence[str] | None) -> tuple[RandomEffect, ...]:
    if random is None:
        return ()
    reject_masked(random, "random")
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
