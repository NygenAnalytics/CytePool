"""Named contrasts: linear combinations of design columns, written as expressions.

``make_contrasts({"b_vs_a": "groupB - groupA", "mean_bc": "(groupB + groupC)/2 -
groupA"}, columns)`` gives the weights of each contrast on the design columns, as
limma's ``makeContrasts`` does. Expressions may use ``+``, ``-``, ``*`` and ``/`` with
numbers, parentheses, and the design's column names written as they are, including
names that are not syntactic in R such as ``(Intercept)`` or ``conditionstim:sexM``;
a name can also be quoted with backticks. Where a name could be read more than one
way, the longest column name that matches is used. A column name that starts or
ends with a space (from a level label with stray spaces) must be quoted with
backticks, so that ``conditionctrl - conditionstim`` never reads the column
``conditionctrl `` by accident.

Unlike ``makeContrasts``, which evaluates the expression in R and so returns a
matrix for any expression, only linear combinations are accepted: an expression
that multiplies or divides two columns, adds a constant (``groupB - 1``), uses a
power or a function, or whose weights cancel to zero raises an error, and so does a
name that is not a column.

Weights that name the columns they refer to are matched to columns by name wherever
they are used, never by position: the rows of a ``Contrasts`` are named by its
``columns``, and those of a pandas Series or DataFrame by its index. This is how
``make_contrasts`` re-expresses contrasts made for another design with the same
columns in another order, and how ``contrasts_fit`` matches contrasts to the
coefficients of a fit.
"""

import numbers
import re
from collections.abc import (
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    Set,
    ValuesView,
)
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._masked_arrays import reject_masked
from .formula import _close_matches

__all__ = ["Contrasts", "make_contrasts"]

_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_WORD = re.compile(r"[\w.]+")
_LISTED = 8  # names listed in a message before "and n more"


@dataclass(frozen=True, eq=False)
class Contrasts:
    """Named contrasts: weights on the columns of a design.

    A read-only mapping of contrast name to weights: indexing by name returns one
    contrast's weights, ``len`` is the number of contrasts, iterating yields their
    names, and ``dict(contrasts)`` gives a plain dict.

    The weight vectors that indexing, ``values()``, ``items()`` and ``dict`` give
    follow the order of ``columns`` and carry no names, so other functions read
    them by position. Wherever the weights meet a fit or another design, pass the
    ``Contrasts`` itself, or ``select(...)`` for some of them: those are matched
    by name.

    Attributes:
        matrix: float64 array of shape ``(n_columns, n_contrasts)``, read-only;
            column ``j`` holds the weights of contrast ``names[j]``.
        names: The contrast names.
        columns: The design columns that the rows of ``matrix`` refer to. Rows are
            matched to the coefficients of a fit, or to the columns of another
            design, by these names.

    Raises:
        TypeError: If a field is or holds a numpy masked array.
        ValueError: If ``matrix`` does not have one row per column and one column
            per name.
    """

    matrix: np.ndarray
    names: tuple[str, ...]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("matrix", "names", "columns"):
            reject_masked(getattr(self, name), name)
        matrix = np.array(self.matrix, dtype=np.float64)
        if matrix.shape != (len(self.columns), len(self.names)):
            raise ValueError(
                f"the contrast matrix must have shape (n_columns, n_contrasts) = "
                f"({len(self.columns)}, {len(self.names)}), got {matrix.shape}"
            )
        matrix.flags.writeable = False
        object.__setattr__(self, "matrix", matrix)

    def __getitem__(self, name: str) -> np.ndarray:
        """Return the weights of one contrast, in the order of ``columns``."""
        return self.matrix[:, self._position(name)]

    def select(self, names: str | Sequence[str]) -> "Contrasts":
        """Return the given contrasts, in the given order, over the same columns.

        Unlike the vectors that indexing returns, the result keeps the names of the
        columns, so ``contrasts_fit`` and ``make_contrasts`` match it by name.

        Args:
            names: Contrast names; a single string selects one contrast.

        Raises:
            KeyError: If a name is not a contrast.
            TypeError: If ``names`` is a numpy masked array.
            ValueError: If a name is repeated.
        """
        reject_masked(names, "names")
        chosen = [names] if isinstance(names, str) else list(names)
        positions = [self._position(name) for name in chosen]
        repeated = [name for i, name in enumerate(chosen) if name in chosen[:i]]
        if repeated:
            raise ValueError(f"contrast {repeated[0]!r} is selected more than once")
        return Contrasts(self.matrix[:, positions], tuple(chosen), self.columns)

    def _position(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"no contrast named {name!r}; contrasts are: "
                f"{', '.join(self.names) if self.names else '(none)'}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def keys(self) -> KeysView[str]:
        """The contrast names."""
        return KeysView(self)  # type: ignore[arg-type]

    def values(self) -> ValuesView[np.ndarray]:
        """The weights of each contrast, in order."""
        return ValuesView(self)  # type: ignore[arg-type]

    def items(self) -> ItemsView[str, np.ndarray]:
        """Pairs of contrast name and weights, in order."""
        return ItemsView(self)  # type: ignore[arg-type]

    def get(self, name: str, default: Any = None) -> Any:
        """The weights of a contrast, or ``default`` if there is none by that name."""
        return self[name] if name in self.names else default

    def __reduce__(self) -> tuple[type["Contrasts"], tuple[Any, ...]]:
        return (Contrasts, (np.array(self.matrix), self.names, self.columns))

    def __repr__(self) -> str:
        lines = [f"Contrasts: {len(self.names)} over {len(self.columns)} columns"]
        for position, name in enumerate(self.names):
            lines.append(
                f"  {name}: {_describe(self.matrix[:, position], self.columns)}"
            )
        return "\n".join(lines)


def make_contrasts(
    contrasts: Mapping[str, Any] | Sequence[str] | str, columns: Any
) -> Contrasts:
    """Turn named linear expressions in design column names into a contrast matrix.

    Args:
        contrasts: A mapping of contrast name to an expression such as
            ``"groupB - groupA"`` or ``"(groupB + groupC)/2 - groupA"``, or to
            weights: a vector with one entry per column, in the order of
            ``columns``, or a pandas Series indexed by column name (one with the
            default index 0, 1, ..., n - 1 is read in order, and other indexes
            that do not hold strings are an error). A list of expressions (or a
            single expression) is also accepted; each expression is then its own
            name, as in limma. ``Contrasts`` made for other columns are
            re-expressed on these columns by name, so they must weigh the same
            columns, in any order.
        columns: The design column names, or an object with a ``columns``
            attribute such as a ``ModelMatrix`` or ``Design``. Contrasts weigh the
            fixed effects, so the random effects of a ``Design`` play no part.

    Returns:
        The contrasts, with a ``(n_columns, n_contrasts)`` weight matrix.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If a name is not a column, an expression is not linear (a
            product or ratio of columns, a constant term, a power or a function),
            a contrast is zero, names repeat, columns repeat, or weights that
            name their columns do not name exactly ``columns``.

    Example:
        >>> make_contrasts({"b_vs_a": "groupB - groupA"},
        ...                ["groupA", "groupB", "groupC"]).matrix[:, 0]
        array([-1.,  1.,  0.])
    """
    _reject_masked_contrasts(contrasts, "contrasts")
    names_of_columns = _column_names(columns)
    if _is_named_contrasts(contrasts):
        named: Any = contrasts
        aligned = _weights_on_columns(
            named,
            names_of_columns,
            label="the contrasts",
            target="the columns",
            remedy="write the contrasts for these columns",
        )
        named = Contrasts(aligned, tuple(named.names), names_of_columns)
        specifications = _specifications(named)
    else:
        specifications = _specifications(contrasts)
    weights = []
    for name, specification in specifications:
        if isinstance(specification, str):
            vector = _LinearExpression(specification, names_of_columns, name).parse()
        else:
            vector = _vector(name, specification, names_of_columns)
        if not np.any(vector):
            raise ValueError(
                f"contrast {name!r} is zero on every column, so it compares nothing"
            )
        weights.append(vector)
    matrix = (
        np.column_stack(weights) if weights else np.empty((len(names_of_columns), 0))
    )
    return Contrasts(
        matrix,
        tuple(name for name, _ in specifications),
        names_of_columns,
    )


Mapping.register(Contrasts)


def _column_names(columns: Any) -> tuple[str, ...]:
    reject_masked(columns, "columns")
    if hasattr(columns, "columns") and not isinstance(columns, str | Sequence):
        columns = columns.columns  # a Design, ModelMatrix or pandas DataFrame
        reject_masked(columns, "columns.columns")
    if isinstance(columns, str | bytes | Set | Mapping) or not hasattr(
        columns, "__iter__"
    ):
        raise TypeError(
            "columns must be the design's column names in order, or an object with a "
            f"columns attribute, such as a Design; got {type(columns).__name__}"
        )
    names = tuple(str(name) for name in columns)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"the design has more than one column named {name!r}, so contrasts "
                "cannot refer to it by name; rename a variable or level so the "
                "column names are unique"
            )
        seen.add(name)
    return names


def _specifications(contrasts: Any) -> list[tuple[str, Any]]:
    if isinstance(contrasts, Contrasts):
        items: list[tuple[Any, Any]] = list(contrasts.items())
    elif isinstance(contrasts, str):
        items = [(contrasts.strip(), contrasts)]
    elif isinstance(contrasts, Mapping):
        items = list(contrasts.items())
    elif isinstance(contrasts, Sequence):
        items = []
        for expression in contrasts:
            if not isinstance(expression, str):
                raise TypeError(
                    "a list of contrasts must hold expressions (strings); give a "
                    "mapping of name to weights for numeric contrasts"
                )
            items.append((expression.strip(), expression))
    else:
        raise TypeError(
            "contrasts must be a mapping of name to expression, or a list of "
            f"expressions; got {type(contrasts).__name__}"
        )
    seen: set[str] = set()
    for name, _ in items:
        if not isinstance(name, str) or not name:
            raise TypeError(f"contrast names must be non-empty strings, got {name!r}")
        if name in seen:
            raise ValueError(f"the contrast name {name!r} is used more than once")
        seen.add(name)
    return items


def _vector(name: str, values: Any, columns: tuple[str, ...]) -> np.ndarray:
    values = _weights_on_columns(
        values,
        columns,
        label=f"contrast {name!r}",
        target="the columns",
        remedy="index the weights by these columns, or give them in their order",
    )
    try:
        vector = np.array(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(
            f"contrast {name!r} must be an expression or a vector of numbers, got "
            f"{type(values).__name__}"
        ) from None
    if vector.shape != (len(columns),):
        raise ValueError(
            f"contrast {name!r} has {vector.size} weights; the design has "
            f"{len(columns)} columns ({', '.join(columns)})"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"contrast {name!r} has weights that are not finite")
    return vector


def _reject_masked_contrasts(contrasts: Any, name: str) -> None:
    """Refuse numpy masked arrays in contrasts given to a function.

    That is, in the contrasts themselves (a matrix, a vector, or the weights of a
    mapping), and in the ``matrix``, ``names`` and ``columns`` of named contrasts
    (see :func:`_is_named_contrasts`).
    """
    reject_masked(contrasts, name)
    if _is_named_contrasts(contrasts):
        for attribute in ("matrix", "names", "columns"):
            reject_masked(getattr(contrasts, attribute), f"{name}.{attribute}")


def _is_pandas(values: Any) -> bool:
    """Whether values are a pandas Series or DataFrame, whose index labels its rows.

    These attributes are pandas' own, which no column of a DataFrame can hide.
    """
    return all(hasattr(values, name) for name in ("index", "iloc", "to_numpy"))


def _is_named_contrasts(values: Any) -> bool:
    """Whether values are a ``Contrasts``, or laid out like one.

    That is, they have a ``(n_columns, n_contrasts)`` ``matrix``, the contrast
    ``names`` and the ``columns`` that name its rows. A pandas DataFrame is not,
    even one with columns of those names (pandas shows columns as attributes).
    """
    return not _is_pandas(values) and all(
        hasattr(values, name) for name in ("matrix", "names", "columns")
    )


def _weights_on_columns(
    weights: Any, columns: Sequence[str], *, label: str, target: str, remedy: str
) -> Any:
    """Contrast weights with their rows in the order of ``columns``.

    This decides, for every function that takes contrast weights, which weights
    name their rows. A pandas Series or DataFrame names them by its index when
    that holds strings, and named contrasts (see :func:`_is_named_contrasts`) by
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
    pandas = _is_pandas(weights)
    if pandas:
        values = weights.to_numpy()
        names = list(weights.index)
        if not all(isinstance(name, str) for name in names):
            if _default_positions(names):
                return values
            raise ValueError(
                f"the index of {label} must hold the names of {target}, or be the "
                f"default 0, 1, ..., n - 1 for weights in the order of {target}; got "
                + _quoted(names)
            )
    elif _is_named_contrasts(weights):
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
            problems.append(f"{_quoted(missing)} {_is_or_are(missing)} missing")
        if unexpected:
            hinted = [
                repr(name) + _close_matches(str(name), [str(m) for m in missing])
                for name in unexpected
            ]
            problems.append(f"{_listed(hinted)} {_is_or_are(hinted)} not among them")
        if pandas and not unexpected:
            remedy += (
                "; to give the missing ones zero weight, reindex the weights with "
                "fill_value=0"
            )
        raise ValueError(
            f"{label} must weigh exactly {target} ({_quoted(columns)}), in any "
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


def _quoted(names: Sequence[Any]) -> str:
    return _listed([repr(name) for name in names])


def _listed(items: list[str]) -> str:
    """Items as "a, b and c", or the first few and how many more."""
    if len(items) > _LISTED:
        return ", ".join(items[:_LISTED]) + f" and {len(items) - _LISTED} more"
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _is_or_are(items: Sequence[Any]) -> str:
    return "is" if len(items) == 1 else "are"


def _describe(vector: np.ndarray, columns: tuple[str, ...]) -> str:
    parts = []
    for weight, column in zip(vector.tolist(), columns, strict=True):
        if weight == 0:
            continue
        sign = "-" if weight < 0 else "+"
        size = abs(weight)
        text = column if size == 1 else f"{size:.6g}*{column}"
        parts.append((sign, text))
    if not parts:
        return "0"
    first_sign, first_text = parts[0]
    text = ("-" if first_sign == "-" else "") + first_text
    return text + "".join(f" {sign} {item}" for sign, item in parts[1:])


@dataclass(frozen=True)
class _Linear:
    """A linear form: weights on the columns plus a constant."""

    weights: np.ndarray
    constant: float
    has_columns: bool


class _LinearExpression:
    """Parses a linear expression in column names by recursive descent."""

    def __init__(self, text: str, columns: tuple[str, ...], name: str) -> None:
        self.text = text
        self.columns = columns
        self.name = name
        self.position_of = {column: i for i, column in enumerate(columns)}
        self.by_length = sorted(columns, key=len, reverse=True)
        self.tokens = self._tokenize()
        self.index = 0

    def error(self, start: int, end: int, problem: str) -> ValueError:
        caret = " " * start + "^" * max(1, end - start)
        shown = re.sub(r"\s", " ", self.text)
        return ValueError(
            f"contrast {self.name!r} ({self.text!r}): {problem}\n    {shown}\n    {caret}"
        )

    def _tokenize(self) -> list[tuple[str, str, int, int]]:
        text = self.text
        tokens: list[tuple[str, str, int, int]] = []
        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace():
                i += 1
                continue
            column = self._column_at(i)
            if column is not None:
                tokens.append(("name", column, i, i + len(column)))
                i += len(column)
                continue
            if char == "`":
                close = text.find("`", i + 1)
                if close < 0:
                    raise self.error(
                        i, len(text), "a backtick-quoted name is not closed"
                    )
                quoted = text[i + 1 : close]
                if quoted not in self.position_of:
                    raise self._unknown(quoted, i, close + 1)
                tokens.append(("name", quoted, i, close + 1))
                i = close + 1
                continue
            number = _NUMBER.match(text, i)
            if number and (char.isdigit() or char == "."):
                tokens.append(("number", number.group(), i, number.end()))
                i = number.end()
                continue
            if char in "+-*/()":
                tokens.append(("operator", char, i, i + 1))
                i += 1
                continue
            if char == "^":
                raise self.error(
                    i, i + 1, "powers are not linear; a contrast is a weighted sum"
                )
            word = _WORD.match(text, i)
            if word:
                end = word.end()
                if text.startswith("(", end):
                    raise self.error(
                        i,
                        end,
                        f"functions such as {word.group()}() are not linear; a "
                        "contrast is a weighted sum of columns",
                    )
                raise self._unknown(word.group(), i, end)
            raise self.error(i, i + 1, f"unexpected character {char!r}")
        tokens.append(("end", "", len(text), len(text)))
        return tokens

    def _column_at(self, i: int) -> str | None:
        text = self.text
        for column in self.by_length:
            if not column or column != column.strip():
                continue  # leading or trailing spaces: only with backticks
            if not text.startswith(column, i):
                continue
            end = i + len(column)
            # A column whose name ends in a word character must not continue into
            # a longer word, so that "groupBB" is not read as "groupB" then "B".
            if end < len(text) and _is_word(column[-1]) and _is_word(text[end]):
                continue
            return column
        return None

    def _unknown(self, word: str, start: int, end: int) -> ValueError:
        return self.error(
            start,
            end,
            f"{word!r} is not a column of the design{_close_matches(word, self.columns)}; "
            f"columns are: {', '.join(self.columns)}",
        )

    def parse(self) -> np.ndarray:
        if self.tokens[0][0] == "end":
            raise self.error(0, len(self.text), "the expression is empty")
        value = self._sum()
        kind, text, start, end = self.tokens[self.index]
        if kind != "end":
            if text == ")":
                raise self.error(start, end, "this ')' has no matching '('")
            raise self.error(start, end, "expected an operator such as + or - here")
        if value.constant != 0:
            raise self.error(
                0,
                len(self.text),
                f"the expression adds the constant {value.constant:g}; a contrast is "
                "a weighted sum of columns without a constant term",
            )
        return value.weights

    def _advance(self) -> tuple[str, str, int, int]:
        token = self.tokens[self.index]
        if token[0] != "end":
            self.index += 1
        return token

    def _sum(self) -> _Linear:
        value = self._product()
        while (
            self.tokens[self.index][1] in ("+", "-")
            and self.tokens[self.index][0] == "operator"
        ):
            operator = self._advance()[1]
            other = self._product()
            if operator == "+":
                value = _Linear(
                    value.weights + other.weights,
                    value.constant + other.constant,
                    value.has_columns or other.has_columns,
                )
            else:
                value = _Linear(
                    value.weights - other.weights,
                    value.constant - other.constant,
                    value.has_columns or other.has_columns,
                )
        return value

    def _product(self) -> _Linear:
        value = self._unary()
        while (
            self.tokens[self.index][1] in ("*", "/")
            and self.tokens[self.index][0] == "operator"
        ):
            _, operator, start, _ = self._advance()
            other = self._unary()
            if operator == "*":
                if value.has_columns and other.has_columns:
                    raise self.error(
                        start,
                        start + 1,
                        "a product of two columns is not linear; multiply columns "
                        "by numbers only",
                    )
                if other.has_columns:
                    value = _Linear(
                        value.constant * other.weights,
                        value.constant * other.constant,
                        True,
                    )
                else:
                    value = _Linear(
                        value.weights * other.constant,
                        value.constant * other.constant,
                        value.has_columns,
                    )
            else:
                if other.has_columns:
                    raise self.error(
                        start,
                        start + 1,
                        "dividing by a column is not linear; divide by numbers only",
                    )
                if other.constant == 0:
                    raise self.error(start, start + 1, "division by zero")
                value = _Linear(
                    value.weights / other.constant,
                    value.constant / other.constant,
                    value.has_columns,
                )
        return value

    def _unary(self) -> _Linear:
        kind, text, start, end = self.tokens[self.index]
        if kind == "operator" and text in ("+", "-"):
            self._advance()
            operand = self._unary()
            if text == "+":
                return operand
            return _Linear(-operand.weights, -operand.constant, operand.has_columns)
        return self._atom()

    def _atom(self) -> _Linear:
        kind, text, start, end = self._advance()
        zeros = np.zeros(len(self.columns))
        if kind == "number":
            number = float(text)
            if not np.isfinite(number):
                raise self.error(start, end, f"the number {text} is too large")
            return _Linear(zeros, number, False)
        if kind == "name":
            zeros[self.position_of[text]] = 1.0
            return _Linear(zeros, 0.0, True)
        if kind == "operator" and text == "(":
            value = self._sum()
            close = self._advance()
            if close[1] != ")":
                raise self.error(start, close[3], "this '(' is not closed by a ')'")
            return value
        if kind == "end":
            raise self.error(
                start, end, "the expression ends where a column or number was expected"
            )
        raise self.error(start, end, f"expected a column or a number, not {text!r}")


def _is_word(char: str) -> bool:
    return char.isalnum() or char in "._"
