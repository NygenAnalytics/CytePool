"""Parse R-style model formulas into terms.

A formula such as ``"~ condition * batch"`` describes the columns of a design matrix. It
is expanded to an intercept flag and a list of terms, each term a set of variables, by
the rules of R's ``terms.formula`` (Chambers and Hastie 1992, chapter 2; R-intro
section 11.1):

- ``a + b`` includes the terms of both sides; ``a - b`` removes the terms of ``b``.
- ``a:b`` is the interaction, every term of ``a`` combined with every term of ``b``,
  with the left side as the outer loop: ``(a + b):(c + d)`` gives ``a:c, a:d, b:c,
  b:d``.
- ``a * b`` is ``a + b + a:b``; ``(a + b + c)^2`` is all terms up to two-way
  interactions.
- ``a / b`` is ``a + b %in% a``, where ``b %in% a`` combines every term of ``b`` with
  all variables of ``a``, so ``(a + b) / c`` is ``a + b + a:b:c``.
- ``1`` and ``0`` set the intercept: ``~ 0 + group``, ``~ group - 1`` and
  ``~ group + 0`` have none. The last setting wins, and inside the right side of a
  ``-`` their meaning is reversed, as in R.

Terms are sorted by the number of variables they contain, keeping their order of
appearance within each order, and the variables of a term are listed in the order in
which they first appear in the formula. Term labels quote names that are not
syntactic R names with backticks, as R does.

Some formulas that R accepts are rejected because their R meaning is almost never
what was intended: a ``-`` that removes nothing (``~ x - z`` does not subtract z; it
is ``~ x`` in R), ``^`` applied to a single term (R reads ``~ age^2`` as ``~ age``),
and a power that is not a whole number (R truncates ``^2.5`` to ``^2``). Function
calls such as ``log(age)`` or ``I(age^2)`` are rejected with a suggestion to add a
column with the computed values.
"""

import difflib
import math
import re
from dataclasses import dataclass

__all__ = [
    "FormulaTerms",
    "RandomEffect",
    "parse_formula",
    "parse_random_effects",
    "quote_name",
]

_RESERVED_WORDS = frozenset(
    {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "in",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
    }
)
_DIGITS = "0123456789"
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?")
_NAME = re.compile(r"(?:[^\W\d_]|\.)[\w.]*")
_SPECIAL_OPERATOR = re.compile(r"%[^%\n]*%")
_OPERATORS = ("||", "|", "~", "+", "-", "*", "/", ":", "^", "(", ")", ",")
# Binding powers (left, right) of the binary operators, from R's ?Syntax: ^ binds
# tightest (and associates to the right), then unary minus and plus, then :, %in%,
# * and /, + and -; the vertical bars of random-effect terms bind loosest.
_BINARY_POWERS = {
    "|": (2, 3),
    "||": (2, 3),
    "+": (10, 11),
    "-": (10, 11),
    "*": (20, 21),
    "/": (20, 21),
    "%in%": (30, 31),
    ":": (40, 41),
    "^": (61, 60),
}
_UNARY_POWER = 50
_CALL_HINTS = {
    "I": "add a column with the computed values, for example age_squared = age**2, "
    "and use its name",
    "factor": "text and boolean columns are factors already; to use a numeric column "
    "as a factor, convert it to text, and use levels= to choose the level order",
    "relevel": "use levels= to choose the reference level, for example "
    "levels={'condition': ['ctrl', 'stim']}",
    "C": "factors are coded by treatment contrasts with the first level as the "
    "reference; use levels= to choose that level",
    "ns": "spline and polynomial bases are not built from formulas; add the basis "
    "columns to the sample table and use their names",
    "offset": "offsets are not columns of a design matrix; pass them to the model "
    "fitting function instead",
}
_CALL_HINTS["as.factor"] = _CALL_HINTS["factor"]
_CALL_HINTS["ordered"] = _CALL_HINTS["factor"]
_CALL_HINTS["bs"] = _CALL_HINTS["ns"]
_CALL_HINTS["poly"] = _CALL_HINTS["ns"]
_DEFAULT_CALL_HINT = (
    "add a column with the computed values to the sample table, for example "
    "log_age = np.log(age), and use its name"
)


def quote_name(name: str) -> str:
    """Return a variable name as R writes it in term labels and column names.

    Syntactic R names (letters, digits, ``.`` and ``_``, not starting with a digit,
    ``_`` or ``.`` followed by a digit, and not a reserved word) are returned as they
    are; other names are quoted with backticks: ``quote_name("cell type")`` returns
    ``"`cell type`"``.
    """
    return name if _is_syntactic(name) else f"`{name}`"


def _close_matches(name: str, candidates: tuple[str, ...] | list[str]) -> str:
    """Return a ' (did you mean ...?)' hint for a misspelled name, or ''."""
    found = difflib.get_close_matches(name, list(candidates), n=3, cutoff=0.6)
    if not found:
        return ""
    return " (did you mean " + " or ".join(repr(match) for match in found) + "?)"


def _is_syntactic(name: str) -> bool:
    if not name or name in _RESERVED_WORDS or re.fullmatch(r"\.\.(\.|\d+)", name):
        return False
    first = name[0]
    if not (first.isalpha() or first == "."):
        return False
    if first == "." and len(name) > 1 and name[1] in _DIGITS:
        return False
    return all(char.isalpha() or char in _DIGITS or char in "._" for char in name)


@dataclass(frozen=True)
class FormulaTerms:
    """The terms of the right-hand side of a formula, as R's ``terms()`` gives them.

    Attributes:
        formula: The formula as given.
        terms: Each term as a tuple of its variables, in R's term order: by the number
            of variables, then by order of appearance. The variables of a term follow
            ``variables``.
        intercept: Whether the formula includes an intercept.
        variables: Every variable named in the formula, in order of first
            appearance, including variables of terms that were removed. This is the
            order that decides the order of variables within terms.
    """

    formula: str
    terms: tuple[tuple[str, ...], ...]
    intercept: bool
    variables: tuple[str, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """The term labels, such as ``"condition"`` and ``"condition:batch"``."""
        return tuple(":".join(quote_name(name) for name in term) for term in self.terms)

    @property
    def used_variables(self) -> tuple[str, ...]:
        """The variables that appear in at least one term, in formula order."""
        used = {name for term in self.terms for name in term}
        return tuple(name for name in self.variables if name in used)


@dataclass(frozen=True)
class RandomEffect:
    """One random-effect term, such as ``(1 | donor)``, in lme4 notation.

    Attributes:
        effects: The left side of the bar: what varies between groups. ``1`` is a
            random intercept; ``1 + age`` (or ``age``) adds a random slope for age;
            ``0 + age`` is a slope without an intercept.
        grouping: The variables whose combinations define the groups, in the order
            of lme4's name for the grouping factor. A nested term
            ``(1 | donor/sample)`` gives two effects, the outer one grouped by
            ``("donor",)`` and the nested one by ``("sample", "donor")``, which
            lme4 names ``sample:donor``.
        correlated: False for the double-bar form ``(age || donor)``, whose effects
            are uncorrelated.
    """

    effects: FormulaTerms
    grouping: tuple[str, ...]
    correlated: bool = True

    @property
    def group(self) -> str:
        """The name of the grouping factor as lme4 gives it, such as ``"sample:donor"``.

        A single grouping variable keeps its name as it is (``cell type``); a
        combination of variables is written as in a formula, with backticks around
        names that need them (found by probing lme4).
        """
        if len(self.grouping) == 1:
            return self.grouping[0]
        return ":".join(quote_name(name) for name in self.grouping)

    def __str__(self) -> str:
        bar = "|" if self.correlated else "||"
        grouping = ":".join(quote_name(name) for name in self.grouping)
        return f"({self.effects.formula} {bar} {grouping})"


def parse_formula(formula: str) -> FormulaTerms:
    """Expand the right-hand side of an R model formula into its terms.

    Args:
        formula: A formula without a response, such as ``"~ condition + batch"``,
            ``"~ 0 + group"`` or ``"~ condition * sex"``. Names that are not
            syntactic in R are quoted with backticks: ``"~ `cell type` + age"``.

    Returns:
        The terms, the intercept flag and the variables of the formula.

    Raises:
        TypeError: If ``formula`` is not a string.
        ValueError: If the formula is not valid, has a response, contains a function
            call, a random-effect term or another unsupported construct; the message
            says where and how to fix it.

    Example:
        >>> parse_formula("~ a * b").labels
        ('a', 'b', 'a:b')
    """
    if not isinstance(formula, str):
        raise TypeError(f"the formula must be a string, got {type(formula).__name__}")
    parser = _Parser(formula, "formula")
    tokens = parser.tokens
    if not _is_tilde(tokens[0]):
        tilde = next((token for token in tokens if _is_tilde(token)), None)
        if tilde is not None:
            raise parser.error(
                tokens[0].start,
                tilde.start,
                "a design formula has no response (left-hand side); the response is "
                "the data the design is fitted to, so write '~ "
                + formula[tilde.end :].strip()
                + "'",
            )
        raise parser.error(
            0,
            len(formula),
            "a formula starts with '~', as in '~ " + formula.strip() + "'",
        )
    parser.advance()
    if parser.peek().kind == "end":
        raise parser.error(
            len(formula), len(formula), "the formula has nothing after '~'"
        )
    tree = parser.parse_expression(0)
    parser.expect_end()
    bar = _find_bar(tree)
    if bar is not None:
        raise parser.error(
            bar.start,
            bar.end,
            "random-effect terms such as (1 | donor) are not part of the "
            "fixed-effects design; pass them separately, as in "
            "Design(samples, '~ condition', random='(1 | donor)'), although no "
            "function estimates random effects yet",
        )
    return _expand(parser, tree, parser.variables)


def parse_random_effects(random: str) -> tuple[RandomEffect, ...]:
    """Parse lme4-style random-effect terms, such as ``"(1 | donor) + (1 | plate)"``.

    Args:
        random: One or more parenthesized terms ``(effects | grouping)`` joined by
            ``+``, optionally after ``~``. ``grouping`` is a variable, an
            interaction ``g1:g2``, or a nesting ``g1/g2``, which expands to
            ``(effects | g1) + (effects | g2:g1)`` as in lme4. ``||`` instead of ``|``
            makes the effects uncorrelated.

    Returns:
        One ``RandomEffect`` per term, in the order written; a nesting gives one
        effect per level of nesting, outer first. Grouping factors are named as
        lme4 names them.

    Raises:
        TypeError: If ``random`` is not a string.
        ValueError: If the text is not a sum of parenthesized bar terms.
    """
    if not isinstance(random, str):
        raise TypeError(
            f"random effects must be given as a string such as '(1 | donor)', got "
            f"{type(random).__name__}"
        )
    parser = _Parser(random, "random-effect formula")
    if _is_tilde(parser.peek()):
        parser.advance()
    if parser.peek().kind == "end":
        raise parser.error(
            0, len(random), "no random-effect terms; write them as '(1 | donor)'"
        )
    tree = parser.parse_expression(0)
    parser.expect_end()
    effects: list[RandomEffect] = []
    for summand in _summands(parser, tree):
        inner = summand.inner if isinstance(summand, _Group) else None
        if not isinstance(inner, _Bar):
            raise parser.error(
                summand.start,
                summand.end,
                "each random-effect term is written in parentheses with a bar, as in "
                "'(1 | donor)' or '(1 + age | donor)'",
            )
        lhs = random[inner.effects.start : inner.effects.end].strip()
        lhs_parser = _Parser(lhs, "random-effect formula")
        lhs_tree = lhs_parser.parse_expression(0)
        lhs_parser.expect_end()
        nested_bar = _find_bar(lhs_tree)
        if nested_bar is not None:
            raise parser.error(
                inner.start, inner.end, "a random-effect term has a single bar"
            )
        terms = _expand(lhs_parser, lhs_tree, lhs_parser.variables)
        if not terms.terms and not terms.intercept:
            raise parser.error(
                inner.start,
                inner.end,
                "this random-effect term has neither an intercept nor a slope, so it "
                "adds nothing; write (1 | group) for a random intercept",
            )
        for grouping in _groupings(parser, inner.grouping):
            effects.append(RandomEffect(terms, grouping, correlated=not inner.double))
    return tuple(effects)


# Tokens and syntax trees.


@dataclass(frozen=True)
class _Token:
    kind: str  # "name", "number", "string", "operator" or "end"
    text: str  # the name (without backticks), the number or the operator
    start: int
    end: int
    quoted: bool = False


@dataclass(frozen=True, eq=False)
class _Node:
    start: int
    end: int


@dataclass(frozen=True, eq=False)
class _Variable(_Node):
    name: str


@dataclass(frozen=True, eq=False)
class _Number(_Node):
    text: str
    value: float


@dataclass(frozen=True, eq=False)
class _Unary(_Node):
    operator: str
    operand: _Node


@dataclass(frozen=True, eq=False)
class _Binary(_Node):
    operator: str
    left: _Node
    right: _Node


@dataclass(frozen=True, eq=False)
class _Group(_Node):
    inner: _Node


@dataclass(frozen=True, eq=False)
class _Bar(_Node):
    effects: _Node
    grouping: _Node
    double: bool


class _Parser:
    """A precedence-climbing parser for R formula syntax."""

    def __init__(self, text: str, what: str) -> None:
        self.text = text
        self.what = what
        self.tokens = self._tokenize()
        self.position = 0
        self.variables: list[str] = []

    def error(self, start: int, end: int, problem: str) -> ValueError:
        shown = re.sub(r"\s", " ", self.text)
        caret = " " * start + "^" * max(1, end - start)
        return ValueError(
            f"invalid {self.what} {self.text!r}: {problem}\n    {shown}\n    {caret}"
        )

    def _tokenize(self) -> list[_Token]:
        text = self.text
        tokens: list[_Token] = []
        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace():
                i += 1
                continue
            if char == "`":
                close = text.find("`", i + 1)
                if close < 0:
                    raise self.error(
                        i, len(text), "a backtick-quoted name is not closed"
                    )
                name = text[i + 1 : close]
                if not name:
                    raise self.error(i, close + 1, "a quoted name is empty")
                tokens.append(_Token("name", name, i, close + 1, quoted=True))
                i = close + 1
                continue
            if char in "\"'":
                close = text.find(char, i + 1)
                end = len(text) if close < 0 else close + 1
                tokens.append(_Token("string", text[i:end], i, end))
                i = end
                continue
            match = _NUMBER.match(text, i)
            if match and (char in _DIGITS or char == "."):
                tokens.append(_Token("number", match.group(), i, match.end()))
                i = match.end()
                continue
            match = _NAME.match(text, i)
            if match:
                tokens.append(_Token("name", match.group(), i, match.end()))
                i = match.end()
                continue
            match = _SPECIAL_OPERATOR.match(text, i)
            if match:
                tokens.append(_Token("operator", match.group(), i, match.end()))
                i = match.end()
                continue
            operator = next((op for op in _OPERATORS if text.startswith(op, i)), None)
            if operator is None:
                raise self.error(
                    i,
                    i + 1,
                    f"unexpected character {char!r}; formulas use variable names and "
                    "the operators + - * / : ^ %in% ( )",
                )
            tokens.append(_Token("operator", operator, i, i + len(operator)))
            i += len(operator)
        tokens.append(_Token("end", "", len(text), len(text)))
        return tokens

    def peek(self) -> _Token:
        return self.tokens[self.position]

    def advance(self) -> _Token:
        token = self.tokens[self.position]
        if token.kind != "end":
            self.position += 1
        return token

    def expect_end(self) -> None:
        token = self.peek()
        if token.kind == "end":
            return
        if token.text == ")":
            raise self.error(token.start, token.end, "this ')' has no matching '('")
        if token.text == ",":
            raise self.error(
                token.start,
                token.end,
                "terms are joined with +, not with commas",
            )
        if token.kind in ("name", "number", "string"):
            raise self.error(
                token.start,
                token.end,
                "expected an operator such as + before this; a name with spaces or "
                "other characters must be quoted with backticks, as in `cell type`",
            )
        raise self.error(token.start, token.end, f"unexpected {token.text!r}")

    def parse_expression(self, min_power: int) -> _Node:
        token = self.advance()
        left: _Node
        if token.kind == "operator" and token.text in ("+", "-"):
            operand = self.parse_expression(_UNARY_POWER)
            left = _Unary(token.start, operand.end, token.text, operand)
        elif token.text == "(" and token.kind == "operator":
            if self.peek().text == ")":
                raise self.error(token.start, self.peek().end, "empty parentheses")
            inner = self.parse_expression(0)
            close = self.advance()
            if close.text == ",":
                raise self.error(
                    close.start, close.end, "terms are joined with +, not with commas"
                )
            if close.text != ")":
                raise self.error(
                    token.start, close.end, "this '(' is not closed by a ')'"
                )
            left = _Group(token.start, close.end, inner)
        elif token.kind == "name":
            following = self.peek()
            if following.text == "(" and following.kind == "operator":
                raise self._call_error(token)
            if not token.quoted:
                self._check_bare_name(token)
            if token.text not in self.variables:
                self.variables.append(token.text)
            left = _Variable(token.start, token.end, token.text)
        elif token.kind == "number":
            text = token.text[:-1] if token.text.endswith("L") else token.text
            left = _Number(token.start, token.end, token.text, float(text))
        elif token.kind == "string":
            raise self.error(
                token.start,
                token.end,
                "names in formulas are not quoted with quotation marks; quote a "
                "name that has spaces or other characters with backticks, as in "
                "`cell type`",
            )
        elif token.kind == "end":
            raise self.error(
                token.start,
                token.end,
                "the formula ends where a variable, 0, 1 or '(' was expected",
            )
        else:
            raise self.error(
                token.start,
                token.end,
                f"expected a variable, 0, 1 or '(' instead of {token.text!r}",
            )
        while True:
            token = self.peek()
            powers = (
                _BINARY_POWERS.get(token.text) if token.kind == "operator" else None
            )
            if token.kind == "operator" and token.text.startswith("%") and not powers:
                raise self.error(
                    token.start,
                    token.end,
                    f"the operator {token.text} is not a formula operator; formulas "
                    "use + - * / : ^ and %in%",
                )
            if powers is None or powers[0] < min_power:
                return left
            self.advance()
            if self.peek().kind == "end":
                raise self.error(
                    token.start,
                    token.end,
                    f"the formula ends after {token.text!r}; a variable, 0, 1 or '(' "
                    "was expected",
                )
            right = self.parse_expression(powers[1])
            if token.text in ("|", "||"):
                left = _Bar(left.start, right.end, left, right, token.text == "||")
            else:
                left = _Binary(left.start, right.end, token.text, left, right)

    def _check_bare_name(self, token: _Token) -> None:
        name = token.text
        if name == ".":
            raise self.error(
                token.start,
                token.end,
                "'.' (all other columns) is not supported; list the variables",
            )
        if name in _RESERVED_WORDS or re.fullmatch(r"\.\.(\.|\d+)", name):
            raise self.error(
                token.start,
                token.end,
                f"{name!r} is a reserved word in R; quote the column name with "
                f"backticks, as in `{name}`",
            )

    def _call_error(self, token: _Token) -> ValueError:
        hint = _CALL_HINTS.get(token.text, _DEFAULT_CALL_HINT)
        depth, end = 0, token.end
        for later in self.tokens[self.position :]:
            if later.text == "(":
                depth += 1
            elif later.text == ")":
                depth -= 1
                if depth == 0:
                    end = later.end
                    break
        call = self.text[token.start : end]
        return self.error(
            token.start,
            end,
            f"function calls such as {call} are not supported in formulas; {hint}",
        )


def _is_tilde(token: _Token) -> bool:
    return token.kind == "operator" and token.text == "~"


def _find_bar(tree: _Node) -> _Bar | None:
    """Return the first random-effect bar in a syntax tree, or None."""
    stack = [tree]
    while stack:
        match stack.pop():
            case _Bar() as bar:
                return bar
            case _Group(inner=inner):
                stack.append(inner)
            case _Unary(operand=operand):
                stack.append(operand)
            case _Binary(left=left, right=right):
                stack.extend((right, left))
    return None


def _summands(parser: _Parser, tree: _Node) -> list[_Node]:
    """Split a sum into its summands, in order."""
    summands: list[_Node] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        match node:
            case _Binary(operator="+", left=left, right=right):
                stack.extend((right, left))
            case _Binary(operator="-") | _Unary(operator="-"):
                raise parser.error(
                    node.start,
                    node.end,
                    "random-effect terms are joined with +, not with -",
                )
            case _:
                summands.append(node)
    return summands


def _groupings(parser: _Parser, node: _Node) -> list[tuple[str, ...]]:
    """Return the grouping variables of each effect, in lme4's name order.

    A nesting ``g1/g2`` expands outer first to ``g1`` and ``g2`` within ``g1``.
    lme4 names every grouping that a nesting produces by its variables in reverse
    order of appearance (``a:b/c`` gives ``b:a`` and ``c:b:a``), and a plain
    interaction by its variables as written (``a:b``); found by probing lme4.
    """
    groupings = _written_groupings(parser, node)
    if _find_nesting(node):
        groupings = [grouping[::-1] for grouping in groupings]
    return groupings


def _find_nesting(node: _Node) -> bool:
    match node:
        case _Group(inner=inner):
            return _find_nesting(inner)
        case _Binary(operator="/"):
            return True
        case _Binary(left=left, right=right):
            return _find_nesting(left) or _find_nesting(right)
    return False


def _written_groupings(parser: _Parser, node: _Node) -> list[tuple[str, ...]]:
    """The groupings with their variables in order of appearance."""
    match node:
        case _Variable(name=name):
            return [(name,)]
        case _Group(inner=inner):
            return _written_groupings(parser, inner)
        case _Binary(operator=":", left=left, right=right):
            first = _written_groupings(parser, left)
            second = _written_groupings(parser, right)
            if len(first) == 1 and len(second) == 1:
                return [_distinct(first[0] + second[0])]
        case _Binary(operator="/", left=left, right=right):
            outer = _written_groupings(parser, left)
            nested = _written_groupings(parser, right)
            if len(nested) == 1:
                return outer + [_distinct(outer[-1] + nested[0])]
    raise parser.error(
        node.start,
        node.end,
        "the grouping of a random effect is a variable, an interaction such as "
        "plate:donor, or a nesting such as donor/sample",
    )


def _distinct(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


# Expansion of the syntax tree into terms.


def _expand(parser: _Parser, tree: _Node, variables: list[str]) -> FormulaTerms:
    expansion = _Expansion(parser, variables)
    terms = expansion.terms(tree, negated=False)
    ordered = sorted(terms, key=len)  # stable: order of appearance within an order
    rank = {name: i for i, name in enumerate(variables)}
    return FormulaTerms(
        formula=parser.text.strip(),
        terms=tuple(tuple(sorted(term, key=rank.__getitem__)) for term in ordered),
        intercept=expansion.intercept,
        variables=tuple(variables),
    )


def _unique(terms: list[frozenset[str]]) -> list[frozenset[str]]:
    return list(dict.fromkeys(terms))


class _Expansion:
    """Expands a syntax tree into terms, tracking the intercept as R does.

    Every ``1`` sets the intercept and every ``0`` removes it, in the order they are
    met; inside the right side of a binary minus or under a unary minus the roles
    are swapped, so ``a - (b - 1)`` keeps the intercept. Interactions, crossings and
    nestings with an operand that has no terms behave as in R: ``a:1`` and ``1 * a``
    give no terms, while ``a * 1`` and ``a %in% 1`` give ``a``.
    """

    def __init__(self, parser: _Parser, variables: list[str]) -> None:
        self.parser = parser
        self.intercept = True
        self.rank = {name: i for i, name in enumerate(variables)}

    def label(self, term: frozenset[str]) -> str:
        ordered = sorted(term, key=self.rank.__getitem__)
        return ":".join(quote_name(name) for name in ordered)

    def terms(self, node: _Node, negated: bool) -> list[frozenset[str]]:
        match node:
            case _Variable(name=name):
                return [frozenset((name,))]
            case _Number(value=value, text=text):
                if value == 1:
                    self.intercept = not negated
                    return []
                if value == 0:
                    self.intercept = negated
                    return []
                raise self.parser.error(
                    node.start,
                    node.end,
                    f"the number {text} cannot be a term; only 1 (include the "
                    "intercept) and 0 (remove it) can appear in a sum",
                )
            case _Group(inner=inner):
                return self.terms(inner, negated)
            case _Unary(operator="+", operand=operand):
                return self.terms(operand, negated)
            case _Unary(operand=operand):
                removed = self.terms(operand, not negated)
                if removed:
                    raise self._removes_nothing(node, removed[0])
                return []
            case _Binary(operator="^", left=left, right=right):
                return self._power(node, left, right, negated)
            case _Binary(operator="+" | "-"):
                # A sum is a left-leaning chain; walk it in a loop so that a long
                # formula does not recurse once per term. Operands are still
                # expanded left to right, which decides the intercept.
                chain = []
                current: _Node = node
                while isinstance(current, _Binary) and current.operator in ("+", "-"):
                    chain.append(current)
                    current = current.left
                result = self.terms(current, negated)
                for link in reversed(chain):
                    flip = negated if link.operator == "+" else not negated
                    second = self.terms(link.right, flip)
                    result = self._combine(link, link.operator, result, second)
                return result
            case _Binary(operator=operator, left=left, right=right):
                first = self.terms(left, negated)
                second = self.terms(right, negated)
                return self._combine(node, operator, first, second)
        raise self.parser.error(node.start, node.end, "unsupported expression")

    def _combine(
        self,
        node: _Node,
        operator: str,
        first: list[frozenset[str]],
        second: list[frozenset[str]],
    ) -> list[frozenset[str]]:
        if operator == "+":
            return _unique(first + second)
        if operator == "-":
            present = set(first)
            for term in second:
                if term not in present:
                    raise self._removes_nothing(node, term)
            removed = set(second)
            return [term for term in first if term not in removed]
        if operator == ":":
            return _unique([a | b for a in first for b in second])
        if operator == "*":
            if not first:
                return []
            return _unique(first + second + [a | b for a in first for b in second])
        if operator == "/":
            if not first:
                return []
            outer = frozenset().union(*first)
            return _unique(first + [outer | b for b in second])
        if operator == "%in%":
            outer = frozenset().union(*second)
            return _unique([a | outer for a in first])
        raise self.parser.error(
            node.start, node.end, f"unsupported operator {operator!r}"
        )

    def _power(
        self, node: _Node, base: _Node, exponent: _Node, negated: bool
    ) -> list[frozenset[str]]:
        if (
            not isinstance(exponent, _Number)
            or not math.isfinite(exponent.value)
            or exponent.value != int(exponent.value)
        ):
            raise self.parser.error(
                exponent.start,
                exponent.end,
                "the power of a formula term is a whole number of at least 2 written "
                "directly after ^, as in (a + b + c)^2",
            )
        power = int(exponent.value)
        if power < 2:
            raise self.parser.error(
                exponent.start,
                exponent.end,
                "the power of a formula term must be at least 2, as in (a + b)^2",
            )
        terms = self.terms(base, negated)
        if len(terms) < 2:
            raise self.parser.error(
                node.start,
                node.end,
                "in a formula, ^ crosses the terms of a sum, as in (a + b + c)^2 for "
                "all main effects and two-way interactions; applied to a single term "
                "it changes nothing (R reads age^2 as age). For a squared covariate, "
                "add a column with the squared values and use its name",
            )
        # R builds a power by interacting the result with the base again and again
        # (found by probing): (c + x)^2 is c, c:x, x before terms are sorted by
        # order, which decides the order of same-order terms when the power is
        # crossed with other terms.
        result = terms
        for _ in range(power - 1):
            crossed = _unique([a | b for a in result for b in terms])
            # Once a step changes nothing, no later step can; the order may still
            # change one step after the set of terms stops growing, so compare lists.
            if crossed == result:
                break
            result = crossed
        return result

    def _removes_nothing(self, node: _Node, term: frozenset[str]) -> ValueError:
        label = self.label(term)
        return self.parser.error(
            node.start,
            node.end,
            f"removing {label} changes nothing, because {label} is not a term at this "
            "point of the formula. In a formula, - removes terms; it does not "
            "subtract values (for a difference of two columns, add a column with "
            "the computed values)",
        )
