"""Model matrices from a sample table and a formula, coded as R's ``model.matrix``.

Variables:
    Numeric columns (integer or float) are covariates and give one column each. Text,
    boolean and pandas categorical columns are factors. A boolean column always has
    the levels FALSE and TRUE, in that order, even when only one of them occurs, and
    gives names such as ``flagTRUE``.

Level order:
    ``levels={"condition": ["ctrl", "stim"]}`` fixes the order of a factor's levels;
    the first level is the reference. Without it, the levels of a pandas categorical
    are its categories, and the levels of a text column are its distinct values
    sorted by Unicode code point. R sorts by the collation of the locale instead
    (``a A b B`` in most UTF-8 locales, ``A B a b`` with ``LC_COLLATE=C``), so text
    levels match R's defaults only under ``LC_COLLATE=C``. Pass ``levels`` whenever
    the reference level matters, which is almost always. A level may be listed
    without having samples; like R, it then gives a column of zeros, which
    ``Design.check`` reports.

Coding:
    Factors are coded by treatment contrasts: indicators of every level except the
    first. A factor ``F`` in a term ``T`` is coded by indicators of all its levels
    instead when the term without ``F`` is not contained in an earlier term (for a
    main effect, the empty term is the intercept), the rule of Chambers and Hastie
    (1992) as R applies it: in ``~ a + a:b`` the term ``a:b`` codes ``a`` by all its
    levels and ``b`` by contrasts, one set of ``b`` contrasts within each level of
    ``a``. Without an intercept, the first term that contains a factor codes it by all
    its levels (Venables et al. 2025, section 11.1.1), as in ``~ 0 + group``. A
    term's columns are products of the columns of its variables, with the variable
    that appears first in the formula varying fastest, and are named by joining the
    parts with ``:``, as in ``conditionstim:sexM`` and ``x:groupB``.

Pandas categorical columns are coded like unordered R factors; the order of an
ordered categorical sets the level order, but R's polynomial contrasts for ordered
factors are not used. Their categories keep their order only when the DataFrame (or
the categorical) is passed itself: a ``Table`` holds the values as text. Categories
that are numbers or booleans are labelled as R labels factor levels (``1e+05``,
``TRUE``). Missing values in any column the formula uses raise an error that names
the column and the rows: samples are never dropped silently.

References:
    Chambers JM, Hastie TJ (1992). Statistical models. Chapter 2 of Statistical Models
    in S, eds Chambers JM and Hastie TJ. Wadsworth & Brooks/Cole, Pacific Grove.

    Venables WN, Smith DM, R Core Team (2025). An Introduction to R, version 4.4.3.
    https://cran.r-project.org/doc/manuals/R-intro.html
"""

import itertools
import math
import numbers
from decimal import Decimal
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .._masked_arrays import reject_masked
from ..table import Table, _is_data_frame, _missing_mask
from .formula import FormulaTerms, _close_matches, parse_formula, quote_name

__all__ = ["INTERCEPT", "ModelMatrix", "model_matrix"]

INTERCEPT = "(Intercept)"
"""The name of the intercept column, as R names it."""

_SAMPLE_ID = "sample_id"
_LISTED_ROWS = 8  # rows named in an error message before "and n more"


@dataclass(frozen=True, eq=False)
class ModelMatrix:
    """A design matrix with R's column names and the terms behind each column.

    Attributes:
        matrix: float64 array of shape ``(n_samples, n_columns)``, read-only.
        columns: Column names, as R's ``colnames(model.matrix(...))``.
        assign: For each column, the term it belongs to: 0 for the intercept and
            ``i`` for ``terms[i - 1]``, as R's ``attr(, "assign")``.
        terms: Term labels in column order, as R's ``attr(terms, "term.labels")``.
        intercept: Whether the first column is the intercept.
        formula: The formula the matrix was built from.
        levels: For every factor in the formula, its levels in order; the first is
            the reference level of treatment contrasts.
        variables: The variables the columns are built from, in formula order.
        coding: For each column, the ``(variable, level)`` pairs it multiplies, with
            level None for a covariate; the intercept has none. For example
            ``(("condition", "stim"), ("age", None))`` for ``conditionstim:age``.

    Raises:
        TypeError: If a field other than ``intercept`` and ``formula``, which are
            stored as given, is or holds a numpy masked array.
        ValueError: If ``matrix`` is not 2-D, or ``columns``, ``assign`` or
            ``coding`` does not have one entry per column.
    """

    matrix: np.ndarray
    columns: tuple[str, ...]
    assign: tuple[int, ...]
    terms: tuple[str, ...]
    intercept: bool
    formula: str
    levels: Mapping[str, tuple[str, ...]]
    variables: tuple[str, ...]
    coding: tuple[tuple[tuple[str, str | None], ...], ...]

    def __post_init__(self) -> None:
        for name in (
            "matrix",
            "columns",
            "assign",
            "terms",
            "levels",
            "variables",
            "coding",
        ):
            reject_masked(getattr(self, name), name)
        matrix = self.matrix
        if not (
            isinstance(matrix, np.ndarray)
            and matrix.dtype == np.float64
            and not matrix.flags.writeable
        ):
            matrix = np.array(matrix, dtype=np.float64)
            matrix.flags.writeable = False
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
        width = matrix.shape[1]
        for name in ("columns", "assign", "coding"):
            if len(getattr(self, name)) != width:
                raise ValueError(
                    f"{name} must have one entry per column ({width}), got "
                    f"{len(getattr(self, name))}"
                )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(
            self,
            "levels",
            MappingProxyType({name: tuple(lv) for name, lv in self.levels.items()}),
        )

    def __reduce__(self) -> tuple[type["ModelMatrix"], tuple[Any, ...]]:
        return (
            ModelMatrix,
            (
                np.array(self.matrix),
                self.columns,
                self.assign,
                self.terms,
                self.intercept,
                self.formula,
                dict(self.levels),
                self.variables,
                self.coding,
            ),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_samples, n_columns)``."""
        return (int(self.matrix.shape[0]), int(self.matrix.shape[1]))

    def __repr__(self) -> str:
        n, p = self.shape
        names = ", ".join(self.columns) if self.columns else "(none)"
        return (
            f"ModelMatrix: {n} sample{'' if n == 1 else 's'} x {p} "
            f"column{'' if p == 1 else 's'} from {self.formula!r}\n  columns: {names}"
        )


def model_matrix(
    data: Any,
    formula: str,
    *,
    levels: Mapping[str, Sequence[Any]] | None = None,
) -> ModelMatrix:
    """Build the design matrix of a formula, column for column as R's model.matrix.

    Args:
        data: The sample table: a ``Table``, a mapping of column name to 1-D array,
            or a pandas DataFrame, with one row per sample.
        formula: A right-hand-side formula such as ``"~ condition + batch"``,
            ``"~ 0 + group"``, ``"~ condition * sex"`` or ``"~ donor + condition"``.
            See ``parse_formula`` for the grammar.
        levels: Level order of factors, for example
            ``{"condition": ["ctrl", "stim"]}``; the first level is the reference.
            Each list must contain every value the column holds. For a boolean
            column, give ``[True, False]`` (or ``["TRUE", "FALSE"]``) to make TRUE
            the reference. Names must be columns of the table; levels of columns
            the formula does not use are ignored, so one mapping can serve several
            formulas.

    Returns:
        The model matrix with R's column names, column order and coding.

    Raises:
        TypeError: If an argument has the wrong type or is or holds a numpy masked
            array, or a used column holds datetimes or complex numbers.
        ValueError: If the formula is invalid, names a column that does not exist,
            uses a column with missing or infinite values, a factor has a single
            level, or ``levels`` does not match the data.

    Example:
        >>> samples = {"condition": ["ctrl", "stim", "ctrl", "stim"],
        ...            "age": [31.0, 45.0, 52.0, 38.0]}
        >>> model_matrix(samples, "~ condition + age").columns
        ('(Intercept)', 'conditionstim', 'age')
    """
    parsed = parse_formula(formula)
    table, categories = _read_samples(data, "data")
    return _build_model_matrix(table, categories, parsed, levels)[0]


def _read_samples(data: Any, name: str) -> tuple[Table, dict[str, tuple[str, ...]]]:
    """Return the sample table and the categories of its pandas categorical columns.

    Categorical columns become text columns, so that their categories, rather than
    numeric codes or sorted values, give the level order. ``name`` names the
    argument in messages.
    """
    reject_masked(data, name, missing="None or NaN")
    if isinstance(data, Table):
        return data, {}
    categories: dict[str, tuple[str, ...]] = {}
    if _is_data_frame(data):
        replaced: dict[int, np.ndarray] = {}
        for position, name in enumerate(data.columns):
            text = _categorical_as_text(data.iloc[:, position])
            if text is not None:
                replaced[position], categories[str(name)] = text
        if replaced:
            data = data.copy()
            for position, values in replaced.items():
                data.isetitem(position, values)
        return Table(data), categories
    if isinstance(data, Mapping):
        columns: dict[Any, Any] = {}
        for name, values in data.items():
            text = _categorical_as_text(values)
            if text is None:
                columns[name] = values
            else:
                columns[name], categories[str(name)] = text
        return Table(columns), categories
    raise TypeError(
        "the sample table must be a Table, a mapping of column names to 1-D arrays "
        f"or a pandas DataFrame; got {type(data).__name__}"
    )


def _build_model_matrix(
    table: Table,
    categories: Mapping[str, tuple[str, ...]],
    parsed: FormulaTerms,
    levels: Mapping[str, Sequence[Any]] | None,
    context: str | None = None,
) -> tuple[ModelMatrix, dict[str, "_Factor | _Covariate"]]:
    """Build a model matrix from a read sample table and a parsed formula.

    Args:
        table: The sample table.
        categories: The categories of columns that were pandas categoricals.
        parsed: The parsed formula.
        levels: Level orders; names that the formula does not use are ignored.
        context: What uses the columns, for error messages; the formula by
            default.

    Returns:
        The model matrix and the variables it was built from.
    """
    if len(table) == 0:
        raise ValueError("the sample table has no rows; a design needs samples")
    given = _normalize_levels(levels)
    _check_level_names(table, given)
    if context is None:
        context = f"the formula {parsed.formula!r}"
    used = parsed.used_variables
    variables = _read_variables(table, used, given, categories, context)
    return _assemble(parsed, variables, len(table)), variables


def _check_level_names(table: Table, levels: Mapping[str, Any]) -> None:
    """Reject levels for names that are not columns, which are usually typos."""
    for name in levels:
        if name not in table:
            raise ValueError(
                f"levels are given for {name!r}, which is not a column of the sample "
                f"table{_close_matches(name, table.columns)}; columns are: "
                f"{', '.join(table.columns) if table.columns else '(none)'}"
            )


@dataclass(frozen=True, eq=False)
class _Factor:
    """A factor read from the sample table: its levels and each sample's level."""

    levels: tuple[str, ...]
    codes: np.ndarray  # intp, the position of each sample's level in levels


@dataclass(frozen=True, eq=False)
class _Covariate:
    """A numeric covariate read from the sample table."""

    values: np.ndarray  # float64


def _normalize_levels(
    levels: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[Any]]:
    """Check the ``levels`` argument and return it as a dict of lists."""
    if levels is None:
        return {}
    reject_masked(levels, "levels")
    if not isinstance(levels, Mapping):
        raise TypeError(
            "levels must be a mapping of variable name to a list of levels, for "
            f"example {{'condition': ['ctrl', 'stim']}}; got {type(levels).__name__}"
        )
    result: dict[str, list[Any]] = {}
    for name, values in levels.items():
        if not isinstance(name, str):
            raise TypeError(f"levels keys must be variable names, got {name!r}")
        if isinstance(values, str | bytes) or not _is_iterable(values):
            raise TypeError(
                f"the levels of {name!r} must be a list of labels, such as "
                f"['ctrl', 'stim'], not {values!r}"
            )
        if isinstance(values, Set | Mapping):
            raise TypeError(
                f"the levels of {name!r} must be in order, as a list or tuple; a "
                f"{type(values).__name__} has no fixed order, and the first level is "
                "the reference"
            )
        result[name] = list(values)
    return result


def _read_variables(
    table: Table,
    names: Sequence[str],
    levels: Mapping[str, list[Any]],
    categories: Mapping[str, tuple[str, ...]],
    context: str,
) -> dict[str, _Factor | _Covariate]:
    """Read and check the columns that a formula uses.

    Args:
        table: The sample table.
        names: The columns to read.
        levels: Level orders by column name.
        categories: The categories of columns that were pandas categoricals.
        context: What uses the columns, for error messages, such as
            "the formula '~ a'".

    Raises:
        ValueError: If a column is missing, has missing or infinite values, or its
            levels do not match the data.
        TypeError: If a column has an unsupported type.
    """
    absent = [name for name in names if name not in table]
    if absent:
        name = absent[0]
        raise ValueError(
            f"{context} uses {quote_name(name)}, which is not a column of the "
            f"sample table{_close_matches(name, table.columns)}; columns are: "
            f"{', '.join(table.columns) if table.columns else '(none)'}"
        )
    gaps = []
    for name in names:
        missing = np.flatnonzero(_missing_mask(table[name]))
        if missing.size:
            gaps.append(f"{name!r} in {_describe_rows(table, missing)}")
    if gaps:
        raise ValueError(
            f"missing values in columns used by {context}: "
            + "; ".join(gaps)
            + ". Samples are not dropped silently: fill in these values or remove "
            "the samples from the table before building the design"
        )
    variables: dict[str, _Factor | _Covariate] = {}
    for name in names:
        variables[name] = _read_variable(
            table, name, levels.get(name), categories.get(name)
        )
    return variables


def _describe_rows(table: Table, positions: np.ndarray | Sequence[int]) -> str:
    """Describe rows by position, and by sample_id when the table has one."""
    positions = [int(position) for position in positions]
    shown = positions[:_LISTED_ROWS]
    more = len(positions) - len(shown)
    noun = "row" if len(positions) == 1 else "rows"
    text = f"{noun} " + ", ".join(str(position) for position in shown)
    if _SAMPLE_ID in table and table[_SAMPLE_ID].dtype == object:
        ids = table[_SAMPLE_ID]
        text += " (sample_id " + ", ".join(str(ids[p]) for p in shown) + ")"
    if more:
        text += f" and {more} more"
    return text


def _read_variable(
    table: Table,
    name: str,
    given: list[Any] | None,
    categories: tuple[str, ...] | None,
) -> _Factor | _Covariate:
    column = table[name]
    kind = column.dtype.kind
    if kind in "iuf":
        if given is not None:
            raise ValueError(
                f"levels are given for {name!r}, but it is a numeric column, so it is "
                "a covariate with one column (a slope), not a factor. To use it as a "
                "factor, convert it to text first, for example "
                f"table.with_columns({{{name!r}: table[{name!r}].astype(str)}})"
            )
        values = column.astype(np.float64)
        infinite = np.flatnonzero(~np.isfinite(values))
        if infinite.size:
            raise ValueError(
                f"column {name!r} has infinite values in "
                f"{_describe_rows(table, infinite)}; covariates must be finite"
            )
        return _Covariate(values)
    if kind == "b":
        labels = np.where(column, "TRUE", "FALSE").astype(object)
        default = ("FALSE", "TRUE")
        order = default if given is None else _boolean_levels(name, given)
        return _factor(table, name, labels, order)
    if kind == "O":
        if given is not None:
            if categories is not None:  # labels of a categorical, as R writes them
                given = [_category_label(value) for value in given]
            order = _text_levels(name, given)
        elif categories is not None:
            order = categories
        else:
            order = tuple(sorted(set(column.tolist())))
        return _factor(table, name, column, order)
    if kind in "mM":
        raise TypeError(
            f"column {name!r} holds dates or times, which a design cannot use "
            "directly; convert it to numbers (for example days since the first "
            "sample) for a covariate, or to text labels for a factor"
        )
    raise TypeError(
        f"column {name!r} has dtype {column.dtype}, which a design cannot use; "
        "covariates are numeric and factors are text or boolean"
    )


def _factor(
    table: Table, name: str, labels: np.ndarray, order: tuple[str, ...]
) -> _Factor:
    repeated = [level for level, count in Counter(order).items() if count > 1]
    if repeated:
        raise ValueError(f"the levels of {name!r} list {repeated[0]!r} more than once")
    position = {level: i for i, level in enumerate(order)}
    values = labels.tolist()
    unlisted = sorted({value for value in values if value not in position})
    if unlisted:
        counts = Counter(value for value in values if value not in position)
        listed = ", ".join(
            f"{value!r} ({counts[value]} sample{'' if counts[value] == 1 else 's'})"
            for value in unlisted
        )
        raise ValueError(
            f"column {name!r} has values that are not among its levels "
            f"{list(order)}: {listed}. The levels must list every value; check the "
            "spelling and capitalization"
        )
    if len(order) < 2:
        raise ValueError(
            f"factor {name!r} has a single level ({order[0]!r}), so it cannot have an "
            "effect; remove it from the formula"
        )
    codes = np.fromiter((position[value] for value in values), np.intp, len(values))
    return _Factor(order, codes)


def _text_levels(name: str, given: list[Any]) -> tuple[str, ...]:
    labels = []
    for value in given:
        if isinstance(value, np.str_):
            value = str(value)
        if not isinstance(value, str):
            raise TypeError(
                f"the levels of the text column {name!r} must be strings, got {value!r}"
            )
        labels.append(value)
    return tuple(labels)


def _boolean_levels(name: str, given: list[Any]) -> tuple[str, ...]:
    labels = []
    for value in given:
        if isinstance(value, bool | np.bool_):
            labels.append("TRUE" if value else "FALSE")
        elif value in ("TRUE", "FALSE"):
            labels.append(value)
        else:
            raise ValueError(
                f"the levels of the boolean column {name!r} are True and False (or "
                f"'TRUE' and 'FALSE'), got {value!r}"
            )
    return tuple(labels)


def _assemble(
    parsed: FormulaTerms, variables: Mapping[str, _Factor | _Covariate], n: int
) -> ModelMatrix:
    factors = {name for name, value in variables.items() if isinstance(value, _Factor)}
    contrasts = _factor_coding(parsed.terms, parsed.intercept, factors)
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    assign: list[int] = []
    coding: list[tuple[tuple[str, str | None], ...]] = []
    if parsed.intercept:
        blocks.append(np.ones(n))
        columns.append(INTERCEPT)
        assign.append(0)
        coding.append(())
    for index, (term, coded) in enumerate(zip(parsed.terms, contrasts, strict=True)):
        parts = [_parts(name, variables[name], coded.get(name)) for name in term]
        # itertools.product varies its last argument fastest; R varies the first
        # variable of the term fastest.
        for combination in itertools.product(*reversed(parts)):
            ordered = combination[::-1]
            values = ordered[0][2]
            with np.errstate(over="ignore"):  # overflow is reported below
                for part in ordered[1:]:
                    values = values * part[2]
            blocks.append(values)
            columns.append(":".join(part[0] for part in ordered))
            assign.append(index + 1)
            coding.append(tuple(part[1] for part in ordered))
    matrix = np.column_stack(blocks) if blocks else np.empty((n, 0))
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    overflowing = np.flatnonzero(~np.all(np.isfinite(matrix), axis=0))
    if overflowing.size:
        raise ValueError(
            f"the column {columns[overflowing[0]]} overflows float64 (its values "
            "multiply covariates beyond 1.8e308); rescale the covariates"
        )
    matrix.flags.writeable = False
    return ModelMatrix(
        matrix=matrix,
        columns=tuple(columns),
        assign=tuple(assign),
        terms=parsed.labels,
        intercept=parsed.intercept,
        formula=parsed.formula,
        levels={
            name: value.levels
            for name, value in variables.items()
            if isinstance(value, _Factor)
        },
        variables=tuple(variables),
        coding=tuple(coding),
    )


def _factor_coding(
    terms: Sequence[Sequence[str]], intercept: bool, factors: set[str]
) -> list[dict[str, bool]]:
    """Decide, for each factor in each term, contrasts (True) or all levels (False).

    A factor F in term T is coded by contrasts when T without F is contained in an
    earlier term; for a main effect (T without F empty) that is the intercept, or,
    without an intercept, any earlier term that contains a factor.
    """
    result = []
    factor_seen = False
    for index, term in enumerate(terms):
        members = set(term)
        coded = {}
        for name in term:
            if name not in factors:
                continue
            rest = members - {name}
            if not rest:
                coded[name] = True
            else:
                coded[name] = any(rest <= set(earlier) for earlier in terms[:index])
        if coded and not factor_seen and not intercept:
            coded = dict.fromkeys(coded, False)
        factor_seen = factor_seen or bool(coded)
        result.append(coded)
    return result


def _parts(
    name: str, variable: _Factor | _Covariate, contrasts: bool | None
) -> list[tuple[str, tuple[str, str | None], np.ndarray]]:
    label = quote_name(name)
    if isinstance(variable, _Covariate):
        return [(label, (name, None), variable.values)]
    first = 1 if contrasts else 0
    return [
        (label + level, (name, level), (variable.codes == k).astype(np.float64))
        for k, level in enumerate(variable.levels)
        if k >= first
    ]


def _categorical_as_text(values: Any) -> tuple[np.ndarray, tuple[str, ...]] | None:
    """Convert a pandas categorical to text labels and its categories, else None."""
    dtype = getattr(values, "dtype", None)
    if getattr(dtype, "name", None) != "category":
        return None
    accessor = getattr(values, "cat", values)  # a Series has .cat; a Categorical not
    categories = tuple(_category_label(category) for category in accessor.categories)
    codes = np.asarray(accessor.codes)
    text = np.empty(codes.size, dtype=object)
    text[:] = [categories[code] if code >= 0 else None for code in codes.tolist()]
    return text, categories


def _category_label(value: Any) -> str:
    """Write a category as R writes the level of a factor made from it.

    Booleans become TRUE and FALSE, integers are written in full, and other
    numbers as R's as.character writes doubles: up to 15 significant digits, in
    scientific notation when that is shorter (``1e+05``, ``0.5``, ``1e-05``).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return "TRUE" if value else "FALSE"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        number = float(value)
        if number == 0:
            return "0"
        # The fewest significant digits that write the same decimal as 15 digits;
        # compared as decimals, since near the largest float the rounded text
        # would read back as infinity.
        reference = Decimal(f"{number:.15g}")
        digits = next(
            d for d in range(1, 16) if Decimal(f"{number:.{d}g}") == reference
        )
        scientific = f"{number:.{digits - 1}e}"
        exponent = int(scientific.split("e")[1])
        fixed = f"{number:.{max(0, digits - 1 - exponent)}f}"
        return fixed if len(fixed) <= len(scientific) else scientific
    return str(value)


def _is_iterable(values: object) -> bool:
    try:
        iter(values)  # type: ignore[call-overload]
    except TypeError:
        return False
    return True
