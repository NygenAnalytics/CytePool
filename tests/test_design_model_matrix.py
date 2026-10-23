"""Tests of model_matrix against R's model.matrix and hand-derived codings.

The fixtures were written by validation/design.py from R 4.4.3: model.matrix on a
small sample table for about a hundred listed formulas, once with R's own default
level order under LC_COLLATE=C and once with explicit levels that change the
reference levels and list a level without samples, and 60 random formulas. Products
of covariates are exact in both programs, so the matrices are compared exactly.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cytepool.design import INTERCEPT, ModelMatrix, model_matrix
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "design"


def _load(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def listed() -> dict[str, Any]:
    return _load("r_model_matrix.json")


@pytest.fixture(scope="module")
def random_formulas() -> dict[str, Any]:
    return _load("r_random_formulas.json")


def _samples(fixture: dict[str, Any]) -> dict[str, Any]:
    samples = dict(fixture["samples"])
    samples["w"] = np.array(samples["w"], dtype=np.int64)
    return samples


def _levels(fixture: dict[str, Any], order: str) -> dict[str, Any] | None:
    # Levels of every factor; those a formula does not use are ignored.
    return fixture["level_orders"].get(order)


def _assert_matches_r(ours: ModelMatrix, case: dict[str, Any]) -> None:
    assert list(ours.columns) == case["columns"], case["formula"]
    assert list(ours.assign) == case["assign"], case["formula"]
    assert list(ours.terms) == case["terms"], case["formula"]
    assert ours.intercept == case["intercept"], case["formula"]
    expected = np.array(case["matrix"], dtype=np.float64).reshape(ours.shape)
    assert_array_equal(ours.matrix, expected, err_msg=case["formula"])


def test_listed_formulas_match_r_with_explicit_levels(listed: dict[str, Any]) -> None:
    samples = _samples(listed)
    cases = [case for case in listed["cases"] if case["levels"] == "custom"]
    assert len(cases) == 114
    for case in cases:
        levels = _levels(listed, case["levels"])
        _assert_matches_r(model_matrix(samples, case["formula"], levels=levels), case)


def test_default_level_order_is_r_under_lc_collate_c(listed: dict[str, Any]) -> None:
    # R decided these level orders itself (LC_COLLATE=C, no explicit levels).
    samples = _samples(listed)
    implicit = [case for case in listed["cases"] if case["levels"] == "implicit_C"]
    assert len(implicit) == 117
    for case in implicit:
        _assert_matches_r(model_matrix(samples, case["formula"]), case)


def test_random_formulas_match_r(random_formulas: dict[str, Any]) -> None:
    samples = _samples(random_formulas)
    assert len(random_formulas["cases"]) == 60
    for case in random_formulas["cases"]:
        levels = _levels(random_formulas, case["levels"])
        _assert_matches_r(model_matrix(samples, case["formula"], levels=levels), case)


@pytest.fixture(scope="module")
def two_by_two() -> dict[str, Any]:
    return {
        "a": ["A", "A", "B", "B"],
        "b": ["X", "Y", "X", "Y"],
        "x": [1.0, 2.0, 3.0, 4.0],
    }


def test_nested_coding_by_hand(two_by_two: dict[str, Any]) -> None:
    # b within a: a by contrasts, b by contrasts within every level of a.
    mm = model_matrix(two_by_two, "~ a + a:b")
    assert mm.columns == (INTERCEPT, "aB", "aA:bY", "aB:bY")
    assert mm.assign == (0, 1, 2, 2)
    assert_array_equal(
        mm.matrix,
        [[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]],
    )


def test_interaction_without_margins_codes_all_levels(
    two_by_two: dict[str, Any],
) -> None:
    mm = model_matrix(two_by_two, "~ a:b")
    # The first variable varies fastest.
    assert mm.columns == (INTERCEPT, "aA:bX", "aB:bX", "aA:bY", "aB:bY")
    assert_array_equal(mm.matrix[:, 1:], np.eye(4)[[0, 2, 1, 3]])


def test_no_intercept_codes_the_first_factor_term_by_all_levels(
    two_by_two: dict[str, Any],
) -> None:
    mm = model_matrix(two_by_two, "~ 0 + x + x:a")
    # x:a would get contrasts by the margin rule, but it is the first term with a
    # factor in a model without intercept (R-intro 11.1.1), so the matrix has
    # rank 2 like R's.
    assert mm.columns == ("x", "x:aA", "x:aB")
    assert_array_equal(mm.matrix, [[1, 1, 0], [2, 2, 0], [3, 0, 3], [4, 0, 4]])
    assert np.linalg.matrix_rank(mm.matrix) == 2
    mm = model_matrix(two_by_two, "~ 0 + x + a")
    assert mm.columns == ("x", "aA", "aB")


def test_margin_contained_in_an_earlier_term_gives_contrasts() -> None:
    # In (a + b):(c + d), d in a:d is coded by contrasts because {a} is part of
    # the earlier term a:c, not because a is a term.
    samples = {
        "a": ["A", "B"] * 4,
        "b": ["X", "X", "Y", "Y"] * 2,
        "c": ["p"] * 4 + ["q"] * 4,
        "d": ["r", "s", "s", "r", "r", "s", "s", "r"],
    }
    mm = model_matrix(samples, "~ (a + b):(c + d)")
    assert mm.columns == (
        INTERCEPT,
        "aA:cp",
        "aB:cp",
        "aA:cq",
        "aB:cq",
        "aA:ds",
        "aB:ds",
        "bY:cp",
        "bY:cq",
        "bY:ds",
    )
    assert mm.terms == ("a:c", "a:d", "b:c", "b:d")
    assert mm.assign == (0, 1, 1, 1, 1, 2, 2, 3, 3, 4)
    assert mm.coding[5] == (("a", "A"), ("d", "s"))


def test_numeric_interactions_multiply_and_names_quote(
    two_by_two: dict[str, Any],
) -> None:
    samples = dict(two_by_two, **{"cell type": ["T", "B", "T", "B"], "z": [2, 3, 5, 7]})
    mm = model_matrix(samples, "~ `cell type`:x + x:z")
    assert mm.columns == (INTERCEPT, "`cell type`B:x", "`cell type`T:x", "x:z")
    assert_array_equal(mm.matrix[:, 1], [0.0, 2.0, 0.0, 4.0])
    assert_array_equal(mm.matrix[:, 3], [2.0, 6.0, 15.0, 28.0])
    assert mm.variables == ("cell type", "x", "z")
    assert mm.levels == {"cell type": ("B", "T")}
    # With x:z first, {x} is part of an earlier term, so `cell type` within x is
    # coded by contrasts; x now comes first in the term label.
    mm = model_matrix(samples, "~ x:z + `cell type`:x")
    assert mm.columns == (INTERCEPT, "x:z", "x:`cell type`T")
    assert mm.terms == ("x:z", "x:`cell type`")


def test_booleans_have_levels_false_true() -> None:
    samples = {"flag": [True, True, True], "other": [True, False, True]}
    mm = model_matrix(samples, "~ flag + other")
    # Both levels exist even when only TRUE occurs, as in R.
    assert mm.columns == (INTERCEPT, "flagTRUE", "otherTRUE")
    assert_array_equal(mm.matrix, [[1, 1, 1], [1, 1, 0], [1, 1, 1]])
    mm = model_matrix(samples, "~ other", levels={"other": [True, False]})
    assert mm.columns == (INTERCEPT, "otherFALSE")
    mm = model_matrix(samples, "~ 0 + other", levels={"other": ["TRUE", "FALSE"]})
    assert mm.columns == ("otherTRUE", "otherFALSE")


def test_default_levels_sort_by_code_point_and_levels_fix_the_reference() -> None:
    samples = {"e": ["b", "a", "B", "A", "_x"]}
    assert model_matrix(samples, "~ e").columns == (INTERCEPT, "eB", "e_x", "ea", "eb")
    mm = model_matrix(samples, "~ e", levels={"e": ["b", "a", "B", "A", "_x"]})
    assert mm.columns == (INTERCEPT, "ea", "eB", "eA", "e_x")
    assert mm.levels["e"] == ("b", "a", "B", "A", "_x")


def test_listed_level_without_samples_gives_a_zero_column() -> None:
    mm = model_matrix({"g": ["A", "B", "A"]}, "~ g", levels={"g": ["A", "B", "C"]})
    assert mm.columns == (INTERCEPT, "gB", "gC")
    assert_array_equal(mm.matrix[:, 2], [0.0, 0.0, 0.0])


def test_level_labels_are_pasted_as_they_are() -> None:
    samples = {"lv": ["CD4 T", "B-cell", "(other)", ""]}
    mm = model_matrix(samples, "~ 0 + lv")
    assert mm.columns == ("lv", "lv(other)", "lvB-cell", "lvCD4 T")


def test_inputs_table_mapping_and_pandas_agree() -> None:
    pandas = pytest.importorskip("pandas")
    values = {"g": ["ctrl", "stim", "ctrl", "stim"], "age": [30.0, 41.0, 52.0, 38.0]}
    expected = [[1, 0, 30], [1, 1, 41], [1, 0, 52], [1, 1, 38]]
    for data in (values, Table(values), pandas.DataFrame(values)):
        mm = model_matrix(data, "~ g + age")
        assert mm.columns == (INTERCEPT, "gstim", "age")
        assert_array_equal(mm.matrix, expected)


def test_pandas_categorical_categories_give_the_level_order() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "g": pandas.Categorical(
                ["stim", "ctrl", "stim"], categories=["stim", "ctrl", "unused"]
            ),
            "batch": pandas.Categorical([2, 1, 2], categories=[2, 1]),
        }
    )
    mm = model_matrix(frame, "~ g + batch")
    # Integer categories make a factor, not a covariate.
    assert mm.columns == (INTERCEPT, "gctrl", "gunused", "batch1")
    assert mm.levels == {"g": ("stim", "ctrl", "unused"), "batch": ("2", "1")}
    mm = model_matrix(
        {"g": frame["g"].array}, "~ g", levels={"g": ["ctrl", "stim", "unused"]}
    )
    assert mm.columns == (INTERCEPT, "gstim", "gunused")


def test_model_matrix_is_read_only_and_pickles(two_by_two: dict[str, Any]) -> None:
    mm = model_matrix(two_by_two, "~ a * x")
    assert mm.matrix.dtype == np.float64 and not mm.matrix.flags.writeable
    with pytest.raises(ValueError):
        mm.matrix[0, 0] = 2.0
    copy = pickle.loads(pickle.dumps(mm))
    assert copy.columns == mm.columns and copy.levels == mm.levels
    assert_array_equal(copy.matrix, mm.matrix)
    assert "4 samples x 4 columns from '~ a * x'" in repr(mm)


def test_empty_and_intercept_only_formulas() -> None:
    samples = {"a": ["A", "B", "A"]}
    mm = model_matrix(samples, "~ 1")
    assert mm.columns == (INTERCEPT,) and mm.assign == (0,)
    assert_array_equal(mm.matrix, np.ones((3, 1)))
    mm = model_matrix(samples, "~ 0")
    assert mm.shape == (3, 0) and mm.terms == ()


def test_missing_values_name_the_column_and_rows() -> None:
    samples = {
        "sample_id": ["S1", "S2", "S3", "S4"],
        "age": [30.0, np.nan, 41.0, np.nan],
        "g": ["A", None, "B", "B"],
        "unused": [None, None, None, None],
    }
    with pytest.raises(
        ValueError,
        match=r"'age' in rows 1, 3 \(sample_id S2, S4\); 'g' in row 1 "
        r"\(sample_id S2\)\. Samples are not dropped silently",
    ):
        model_matrix(samples, "~ age + g")


@pytest.mark.parametrize(
    ("formula", "levels", "error", "message"),
    [
        (
            "~ conditon",
            None,
            ValueError,
            r"uses conditon, which is not a column .*did you mean 'condition'",
        ),
        (
            "~ condition",
            {"conditon": ["a", "b"]},
            ValueError,
            r"levels are given for 'conditon'.*did you mean 'condition'",
        ),
        (
            "~ condition",
            {"condition": ["ctrl"]},
            ValueError,
            r"values that are not among its levels \['ctrl'\]: 'Stim' \(1 sample\), 'stim' \(2 samples\).*capitalization",
        ),
        (
            "~ condition",
            {"condition": ["ctrl", "stim", "Stim", "ctrl"]},
            ValueError,
            "list 'ctrl' more than once",
        ),
        ("~ condition", {"condition": "ctrl"}, TypeError, "must be a list of labels"),
        ("~ condition", {"condition": [1, 2]}, TypeError, "must be strings, got 1"),
        (
            "~ age",
            {"age": ["30", "40"]},
            ValueError,
            "numeric column, so it is a covariate .* convert it to text first",
        ),
        (
            "~ flag",
            {"flag": ["yes", "no"]},
            ValueError,
            "boolean column 'flag' are True and False",
        ),
        ("~ batch", None, ValueError, r"factor 'batch' has a single level \('b1'\)"),
        ("~ age", ["age"], TypeError, "levels must be a mapping"),
        ("~ big", None, ValueError, "column 'big' has infinite values in row 2"),
        ("~ when", None, TypeError, "holds dates or times"),
        ("~ wave", None, TypeError, "dtype complex128"),
    ],
)
def test_invalid_inputs_raise_with_a_fix(
    formula: str, levels: Any, error: type[Exception], message: str
) -> None:
    samples = {
        "condition": ["ctrl", "stim", "Stim", "stim"],
        "age": [30.0, 41.0, 52.0, 38.0],
        "flag": [True, False, True, False],
        "batch": ["b1"] * 4,
        "big": [1.0, 2.0, np.inf, 3.0],
        "when": np.array(["2024-01-01"] * 4, dtype="M8[D]"),
        "wave": np.ones(4, dtype=complex),
    }
    with pytest.raises(error, match=message):
        model_matrix(samples, formula, levels=levels)


def test_unsupported_data_and_empty_tables() -> None:
    with pytest.raises(TypeError, match="must be a Table, a mapping"):
        model_matrix([1, 2, 3], "~ x")
    with pytest.raises(ValueError, match="the sample table has no rows"):
        model_matrix({"x": np.array([], dtype=float)}, "~ x")


@pytest.mark.parametrize("unordered", [{"ctrl", "stim"}, {"ctrl": 0, "stim": 1}])
def test_unordered_levels_are_rejected(unordered: Any) -> None:
    # A set has no fixed order, so the reference level could change between runs.
    with pytest.raises(TypeError, match="must be in order, as a list or tuple"):
        model_matrix({"g": ["ctrl", "stim"]}, "~ g", levels={"g": unordered})


def test_number_and_boolean_categories_are_labelled_as_in_r() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "flag": pandas.Categorical([True, False, True]),
            "dose": pandas.Categorical([0.5, 1e5, 1.0]),
            "plate": pandas.Categorical([3, 1, 3]),
        }
    )
    # R writes factor levels made from these values as TRUE, 0.5, 1 and 1e+05.
    mm = model_matrix(frame, "~ flag + dose + plate")
    assert mm.columns == (INTERCEPT, "flagTRUE", "dose1", "dose1e+05", "plate3")
    mm = model_matrix(frame, "~ plate", levels={"plate": [3, 1]})
    assert mm.columns == (INTERCEPT, "plate1")
    # R's as.character of the largest double and of the smallest subnormal.
    extremes = pandas.Categorical([1.7976931348623157e308, 5e-324] * 2)
    assert model_matrix({"v": extremes}, "~ v").levels["v"] == (
        "4.94065645841247e-324",
        "1.79769313486232e+308",
    )


def test_overflowing_products_raise() -> None:
    samples = {"x": [1e200, 2e200, 3e200], "z": [1e200, 1e200, 2e200]}
    with pytest.raises(ValueError, match="the column x:z overflows float64"):
        model_matrix(samples, "~ x:z")
