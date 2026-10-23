"""Tests of formula parsing: terms, intercept, variable order and error messages.

Expected terms come from R's terms() (R 4.4.3) run as a black box: the fixture
r_terms.json written by validation/design.py, and hand-picked cases below.
"""

import json
import re
from pathlib import Path

import pytest

from cytepool.design import RandomEffect, parse_formula, parse_random_effects

DATA = Path(__file__).parent / "data" / "design"


@pytest.mark.parametrize(
    ("formula", "labels", "intercept"),
    [
        ("~ a * b", ("a", "b", "a:b"), True),
        ("~ 0 + group", ("group",), False),
        ("~ group - 1", ("group",), False),
        ("~ -1 + group", ("group",), False),
        ("~ group + 0", ("group",), False),
        # The last intercept setting wins.
        ("~ 0 + a + 1", ("a",), True),
        ("~ 1 + a - 1", ("a",), False),
        ("~ a + 1 + 0", ("a",), False),
        # Inside the right side of a minus the meaning of 0 and 1 is reversed.
        ("~ a - (-1)", ("a",), True),
        ("~ a*b - (a:b - 1)", ("a", "b"), True),
        # Interactions: left side is the outer loop.
        ("~ (a + b):(c + d)", ("a:c", "a:d", "b:c", "b:d"), True),
        ("~ (a+b+c)^2", ("a", "b", "c", "a:b", "a:c", "b:c"), True),
        ("~ (a+b+c)^2 - a:b", ("a", "b", "c", "a:c", "b:c"), True),
        ("~ (a + b:c)^2", ("a", "b:c", "a:b:c"), True),
        # Nesting combines with all variables of the left (or right) side.
        ("~ a/b", ("a", "a:b"), True),
        ("~ (a + b)/c", ("a", "b", "a:b:c"), True),
        ("~ a/b/c", ("a", "a:b", "a:b:c"), True),
        ("~ b %in% a", ("b:a",), True),
        ("~ a + b %in% a", ("a", "a:b"), True),
        ("~ (a + b) %in% c", ("a:c", "b:c"), True),
        ("~ a %in% (b + c)", ("a:b:c",), True),
        ("~ a/(1 + x) - 1", ("a", "a:x"), False),
        # Variables of a term follow their first appearance in the formula.
        ("~ b + a:b + a", ("b", "a", "b:a"), True),
        ("~ c + a:c", ("c", "c:a"), True),
        ("~ z - z + x:z", ("z:x",), True),
        # Repeated terms are kept once, at their first position.
        ("~ a + b + a", ("a", "b"), True),
        ("~ a:b + b:a", ("a:b",), True),
        ("~ a + b - a + a", ("b", "a"), True),
        # An operand without terms: a:1 and 1*a give nothing, a*1 and a %in% 1 a.
        ("~ a:1 + b", ("b",), True),
        ("~ 1*a + b", ("b",), True),
        ("~ a*1", ("a",), True),
        ("~ a %in% 1", ("a",), True),
        ("~ 1", (), True),
        ("~ 0", (), False),
        ("~ `cell type` + x", ("`cell type`", "x"), True),
        ("~ `x.y`", ("x.y",), True),
        ("~\n  condition +\n  batch", ("condition", "batch"), True),
    ],
)
def test_terms_and_intercept_follow_r(
    formula: str, labels: tuple[str, ...], intercept: bool
) -> None:
    terms = parse_formula(formula)
    assert terms.labels == labels
    assert terms.intercept is intercept


def test_grammar_cases_match_r_terms() -> None:
    # R's terms() on 282 formulas, written by validation/design.py. Formulas that R
    # accepts but CytePool rejects must fail for one of the deliberate reasons.
    fixture = json.loads((DATA / "r_terms.json").read_text(encoding="utf-8"))
    deliberate = re.compile(
        "changes nothing|crosses the terms|whole number|function calls"
    )
    compared = 0
    for case in fixture["cases"]:
        if "error" in case:
            with pytest.raises(ValueError):
                parse_formula(case["formula"])
            continue
        try:
            terms = parse_formula(case["formula"])
        except ValueError as error:
            assert deliberate.search(str(error)), case["formula"]
            continue
        assert list(terms.labels) == case["labels"], case["formula"]
        assert terms.intercept == case["intercept"], case["formula"]
        compared += 1
    assert compared == 236


def test_power_order_inside_an_interaction_follows_r() -> None:
    # R expands (c + x)^2 to c, c:x, x before sorting by order, which shows when
    # the power is crossed with terms of different orders (found by probing).
    assert parse_formula("~ ((c + x)^2):(z + a:b)").labels == (
        "c:z",
        "x:z",
        "c:a:b",
        "c:x:z",
        "x:a:b",
        "c:x:a:b",
    )
    # A crossing with * keeps its left side, right side, then interactions.
    assert parse_formula("~ (c*x):(z + a:b)").labels == (
        "c:z",
        "x:z",
        "c:a:b",
        "x:a:b",
        "c:x:z",
        "c:x:a:b",
    )


def test_variables_include_removed_terms_and_terms_are_tuples() -> None:
    terms = parse_formula("~ b*a - b:a + c:a")
    assert terms.variables == ("b", "a", "c")
    assert terms.terms == (("b",), ("a",), ("a", "c"))
    assert terms.used_variables == ("b", "a", "c")
    assert terms.formula == "~ b*a - b:a + c:a"


def test_unicode_letters_make_syntactic_names() -> None:
    # R in a UTF-8 locale treats letters such as a-circumflex as letters.
    assert parse_formula("~ âge").labels == ("âge",)
    assert parse_formula("~ `1x` + `if` + `_y`").labels == ("`1x`", "`if`", "`_y`")


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("a + b", r"a formula starts with '~', as in '~ a \+ b'"),
        ("y ~ a + b", r"no response .*write '~ a \+ b'"),
        ("~", "nothing after '~'"),
        ("~ log(age)", r"function calls such as log\(age\) are not supported"),
        ("~ I(age^2) + x", r"I\(age\^2\).*age_squared = age\*\*2"),
        ("~ factor(batch)", "text and boolean columns are factors already"),
        ("~ relevel(group, 'b')", "use levels= to choose the reference level"),
        ("~ condition + (1 | donor)", r"random-effect terms such as \(1 \| donor\)"),
        ("~ .", "'.' \\(all other columns\\) is not supported"),
        ("~ if + a", "'if' is a reserved word in R"),
        ('~ "cell type"', "quote a name that has spaces .* with backticks"),
        ("~ a $ b", "unexpected character '\\$'"),
        ("~ a %o% b", "the operator %o% is not a formula operator"),
        ("~ (a + b", "this '\\(' is not closed"),
        ("~ a + b)", "this '\\)' has no matching '\\('"),
        ("~ a +", "ends after '\\+'"),
        ("~ a, b", "joined with \\+, not with commas"),
        ("~ (a, b)", "joined with \\+, not with commas"),
        ("~ cell type", "must be quoted with backticks"),
        ("~ a + 2", "the number 2 cannot be a term"),
        ("~ (a + b)^1", "must be at least 2"),
        ("~ (a + b)^(2)", "whole number of at least 2 written directly after"),
        ("~ (a + b)^2.5", "whole number"),
        ("~ (a + b)^x", "whole number"),
        ("~ age^2", r"applied to a single term it changes nothing \(R reads age\^2"),
        ("~ a - b", "removing b changes nothing"),
        ("~ x - z + x:z", "removing z changes nothing"),
        ("~ -a + b", "removing a changes nothing"),
        ("~ a*b - b:c", "removing b:c changes nothing"),
        ("~ ``", "a quoted name is empty"),
        ("~ `cell type", "backtick-quoted name is not closed"),
        ("~ ()", "empty parentheses"),
        ("~ * a", "expected a variable, 0, 1 or '\\(' instead of '\\*'"),
    ],
)
def test_invalid_formulas_raise_with_a_fix(formula: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_formula(formula)


def test_error_message_points_at_the_problem() -> None:
    with pytest.raises(ValueError) as caught:
        parse_formula("~ a + log(x)")
    lines = str(caught.value).splitlines()
    assert lines[1] == "    ~ a + log(x)"
    assert lines[2] == "          ^^^^^^"


def test_formula_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="the formula must be a string, got int"):
        parse_formula(3)  # type: ignore[arg-type]


def test_random_effects_parse_like_lme4() -> None:
    effects = parse_random_effects("~ (1 | donor) + (1 + age | plate:donor)")
    assert [effect.grouping for effect in effects] == [("donor",), ("plate", "donor")]
    assert [effect.group for effect in effects] == ["donor", "plate:donor"]
    assert effects[0].effects.terms == () and effects[0].effects.intercept
    assert effects[1].effects.labels == ("age",) and effects[1].effects.intercept
    assert [str(effect) for effect in effects] == [
        "(1 | donor)",
        "(1 + age | plate:donor)",
    ]


def test_nested_grouping_expands_inner_first_like_lme4() -> None:
    # lme4 names the nested factor of (1 | g1/g2/d) d:g2:g1 (found by probing).
    effects = parse_random_effects("(1 | g1/g2/d)")
    assert [effect.group for effect in effects] == ["g1", "g2:g1", "d:g2:g1"]
    assert all(isinstance(effect, RandomEffect) for effect in effects)


def test_double_bar_and_no_intercept() -> None:
    (uncorrelated,) = parse_random_effects("(age || donor)")
    assert not uncorrelated.correlated
    assert str(uncorrelated) == "(age || donor)"
    (slope,) = parse_random_effects("(0 + age | `donor id`)")
    assert not slope.effects.intercept and slope.effects.labels == ("age",)
    # lme4 names a single grouping variable as it is and quotes combinations.
    assert slope.group == "donor id" and str(slope) == "(0 + age | `donor id`)"
    (combined,) = parse_random_effects("(1 | `cell type`:donor)")
    assert combined.group == "`cell type`:donor"


@pytest.mark.parametrize(
    ("random", "message"),
    [
        ("1 | donor", "written in parentheses with a bar"),
        ("(1 | donor) - (1 | plate)", "joined with \\+, not with -"),
        ("(1 | 1)", "the grouping of a random effect is a variable"),
        ("(1 | donor | plate)", "a random-effect term has a single bar"),
        ("(log(age) | donor)", "function calls such as log\\(age\\)"),
        ("", "no random-effect terms"),
        ("(donor)", "written in parentheses with a bar"),
    ],
)
def test_invalid_random_effects_raise(random: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_random_effects(random)


def test_random_effects_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="random effects must be given as a string"):
        parse_random_effects(["(1 | donor)"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("random", "groups"),
    [
        # lme4 names every grouping of a nesting in reverse order of appearance and
        # keeps a plain interaction as written (found by probing lme4 2.0.6).
        ("(1 | a:b/c)", ["b:a", "c:b:a"]),
        ("(1 | a/b:c)", ["a", "c:b:a"]),
        ("(1 | c:b/a)", ["b:c", "a:b:c"]),
        ("(1 | a/(b:c))", ["a", "c:b:a"]),
        ("(1 | (a/b)/c)", ["a", "b:a", "c:b:a"]),
        ("(1 | b:a)", ["b:a"]),
    ],
)
def test_grouping_names_follow_lme4(random: str, groups: list[str]) -> None:
    assert [effect.group for effect in parse_random_effects(random)] == groups


@pytest.mark.parametrize("random", ["(0 | donor)", "(-1 | donor)", "(1 + 0 | donor)"])
def test_random_terms_without_effects_are_rejected(random: str) -> None:
    with pytest.raises(ValueError, match="has neither an intercept nor a slope"):
        parse_random_effects(random)


def test_long_sums_do_not_hit_the_recursion_limit() -> None:
    names = [f"x{i}" for i in range(1500)]
    terms = parse_formula("~ " + " + ".join(names) + " - x7 - 1")
    assert len(terms.terms) == 1499 and not terms.intercept
    assert terms.labels[7] == "x8"
