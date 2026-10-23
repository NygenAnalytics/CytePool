"""Tests of make_contrasts against limma's makeContrasts and hand-derived weights."""

import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.design import Contrasts, make_contrasts, model_matrix

DATA = Path(__file__).parent / "data" / "design"
COLUMNS = ["(Intercept)", "groupB", "groupC", "conditionstim:sexM", "cell_typeCD4 T"]


def test_matches_limma_make_contrasts() -> None:
    fixture = json.loads((DATA / "r_model_matrix.json").read_text())["contrasts"]
    ours = make_contrasts(fixture["expressions"], fixture["columns"])
    # Same arithmetic in the same order as R evaluates it, so the weights agree to
    # the last bit; 1e-12 still fails any wrong weight.
    assert_allclose(ours.matrix, np.array(fixture["matrix"]), rtol=1e-12, atol=0)
    assert list(ours.names) == fixture["names"]


def test_named_expressions_by_hand() -> None:
    contrasts = make_contrasts(
        {
            "b_vs_a": "groupB",
            "c_vs_b": "groupC - groupB",
            "mean_bc": "(groupB + groupC)/2",
            "scaled": "2*(groupC - groupB)/4 + 0.25*groupB",
            "interaction": "conditionstim:sexM",
            "spaces": "`cell_typeCD4 T` - (Intercept)",
        },
        COLUMNS,
    )
    assert contrasts.names == (
        "b_vs_a",
        "c_vs_b",
        "mean_bc",
        "scaled",
        "interaction",
        "spaces",
    )
    assert contrasts.matrix.shape == (5, 6)
    assert_array_equal(
        contrasts.matrix,
        [
            [0, 0, 0, 0, 0, -1],
            [1, -1, 0.5, -0.25, 0, 0],
            [0, 1, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
    )
    assert_array_equal(contrasts["c_vs_b"], [0, -1, 1, 0, 0])
    assert "mean_bc" in contrasts and len(contrasts) == 6
    assert list(contrasts) == list(contrasts.names)


def test_longest_column_name_wins_and_words_do_not_split() -> None:
    columns = ["groupB", "groupB:x", "groupBB", "conditionIFN-g", "x"]
    contrasts = make_contrasts(
        ["groupB:x - groupB", "groupBB - groupB", "conditionIFN-g - x"], columns
    )
    assert_array_equal(
        contrasts.matrix.T,
        [[-1, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, 1, -1]],
    )
    with pytest.raises(ValueError, match="'groupBBB' is not a column"):
        make_contrasts(["groupBBB"], columns)


def test_numeric_weights_and_designs_as_columns() -> None:
    design = model_matrix({"g": ["a", "b", "c", "a"]}, "~ g")
    contrasts = make_contrasts({"c_vs_b": [0, -1, 1], "b": "gb"}, design)
    assert contrasts.columns == ("(Intercept)", "gb", "gc")
    assert_array_equal(contrasts.matrix, [[0, 0], [-1, 1], [1, 0]])
    assert contrasts.matrix.dtype == np.float64 and not contrasts.matrix.flags.writeable
    copy = pickle.loads(pickle.dumps(contrasts))
    assert isinstance(copy, Contrasts) and copy.names == contrasts.names
    assert "c_vs_b: -gb + gc" in repr(contrasts)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("groupD - groupB", r"'groupD' is not a column of the design \(did you mean"),
        ("Intercept", r"did you mean '\(Intercept\)'"),
        ("groupB - 1", "adds the constant -1"),
        ("groupB*groupC", "a product of two columns is not linear"),
        ("groupB/groupC", "dividing by a column is not linear"),
        ("groupB/0", "division by zero"),
        ("groupB^2", "powers are not linear"),
        ("log(groupB)", r"functions such as log\(\) are not linear"),
        ("groupB - groupB", "is zero on every column"),
        ("0*groupB", "is zero on every column"),
        ("", "the expression is empty"),
        ("groupB groupC", "expected an operator"),
        ("(groupB - groupC", "not closed"),
        ("groupB)", "has no matching"),
        ("groupB -", "the expression ends where a column or number was expected"),
        ("`groupD`", "'groupD' is not a column"),
        ("groupB & groupC", "unexpected character '&'"),
        ("groupB*1e400", "the number 1e400 is too large"),
    ],
)
def test_invalid_expressions_raise(expression: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_contrasts({"test": expression}, COLUMNS)


def test_invalid_arguments_raise() -> None:
    # Expressions given as a list are their own names, so a repeat is a clash.
    with pytest.raises(ValueError, match="the contrast name 'groupB' is used more"):
        make_contrasts(["groupB", " groupB "], COLUMNS)
    with pytest.raises(ValueError, match="more than one column named 'abX'"):
        make_contrasts(["abX"], ["abX", "abX"])
    with pytest.raises(ValueError, match="has 2 weights; the design has 5 columns"):
        make_contrasts({"v": [1, -1]}, COLUMNS)
    with pytest.raises(ValueError, match="weights that are not finite"):
        make_contrasts({"v": [0, 1, np.nan, 0, 0]}, COLUMNS)
    with pytest.raises(TypeError, match="contrasts must be a mapping"):
        make_contrasts(3, COLUMNS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="columns must be the design's column names"):
        make_contrasts(["groupB"], "groupB")
    with pytest.raises(TypeError, match="contrast names must be non-empty strings"):
        make_contrasts({"": "groupB"}, COLUMNS)


def test_single_expression_is_its_own_name() -> None:
    contrasts = make_contrasts(" groupC - groupB ", COLUMNS)
    assert contrasts.names == ("groupC - groupB",)
    with pytest.raises(KeyError, match="no contrast named 'other'"):
        contrasts["other"]


def test_names_with_stray_spaces_need_backticks() -> None:
    # A level "ctrl " next to "ctrl" gives the column "conditionctrl "; reading it
    # in "conditionctrl - conditionstim" would silently weigh the wrong group.
    columns = ["conditionctrl", "conditionctrl ", "conditionstim"]
    contrasts = make_contrasts(
        {
            "ctrl_vs_stim": "conditionctrl - conditionstim",
            "stim_vs_ctrl": "conditionstim - conditionctrl",
            "stray": "`conditionctrl ` - conditionctrl",
        },
        columns,
    )
    assert_array_equal(contrasts["ctrl_vs_stim"], -contrasts["stim_vs_ctrl"])
    assert_array_equal(contrasts["ctrl_vs_stim"], [1, 0, -1])
    assert_array_equal(contrasts["stray"], [-1, 1, 0])


def test_contrasts_are_a_mapping_and_accept_any_sequence_of_names() -> None:
    pandas = pytest.importorskip("pandas")
    columns = pandas.Index(["(Intercept)", "gb", "gc"])
    contrasts = make_contrasts({"c_vs_b": "gc - gb"}, columns)
    assert isinstance(contrasts, Mapping)
    assert list(dict(contrasts)) == ["c_vs_b"]
    assert contrasts.keys() == {"c_vs_b"}
    assert_array_equal(list(contrasts.values())[0], [0, -1, 1])
    assert_array_equal(contrasts.get("c_vs_b"), [0, -1, 1])
    assert contrasts.get("other") is None
    assert [name for name, _ in contrasts.items()] == ["c_vs_b"]
    frame = pandas.DataFrame(columns=columns)
    assert make_contrasts(["gb"], frame).columns == tuple(columns)
    with pytest.raises(TypeError, match="column names in order"):
        make_contrasts(["gb"], {"gb", "gc"})
