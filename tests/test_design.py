"""Tests of the Design class: contents, random effects, contrasts, immutability."""

import copy
import pickle
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cytepool.design import Design
from cytepool.table import Table


@pytest.fixture(scope="module")
def samples() -> dict[str, Any]:
    return {
        "sample_id": [f"S{i}" for i in range(1, 9)],
        "condition": ["ctrl", "stim"] * 4,
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
        "plate": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
        "age": [30.0, 30.0, 41.0, 41.0, 52.0, 52.0, 38.0, 38.0],
    }


def test_design_holds_matrix_columns_terms_and_levels(samples: dict[str, Any]) -> None:
    design = Design(
        samples,
        "~ condition + age",
        levels={"condition": ["stim", "ctrl"]},
        contrasts={"ctrl_vs_stim": "conditionctrl", "age": "age"},
    )
    assert design.columns == ("(Intercept)", "conditionctrl", "age")
    assert_array_equal(design.matrix[:, 1], [1, 0, 1, 0, 1, 0, 1, 0])
    assert_array_equal(design.matrix[:, 2], samples["age"])
    assert design.terms == ("condition", "age")
    assert design.assign == (0, 1, 2)
    assert design.levels == {"condition": ("stim", "ctrl")}
    assert design.formula == "~ condition + age"
    assert design.n_samples == 8
    assert isinstance(design.samples, Table)
    assert design.samples["sample_id"].tolist() == samples["sample_id"]
    assert design.model_matrix.coding[1] == (("condition", "ctrl"),)
    assert design.contrasts.names == ("ctrl_vs_stim", "age")
    assert_array_equal(design.contrasts["ctrl_vs_stim"], [0, 1, 0])
    assert design.random == ()
    assert repr(design).splitlines() == [
        "Design: 8 samples x 3 columns, formula '~ condition + age'",
        "  columns: (Intercept), conditionctrl, age",
        "  contrasts: ctrl_vs_stim, age",
    ]


def test_random_effects_are_parsed_and_their_columns_validated(
    samples: dict[str, Any],
) -> None:
    design = Design(samples, "~ condition", random=["(1 | donor)", "(1 | plate/donor)"])
    assert [str(effect) for effect in design.random] == [
        "(1 | donor)",
        "(1 | plate)",
        "(1 | donor:plate)",
    ]
    assert "random: (1 | donor) + (1 | plate) + (1 | donor:plate)" in repr(design)
    with pytest.raises(ValueError, match=r"groups by 'donr'.*did you mean 'donor'"):
        Design(samples, "~ condition", random="(1 | donr)")
    with pytest.raises(ValueError, match="uses agee, which is not a column"):
        Design(samples, "~ condition", random="(agee | donor)")
    with pytest.raises(
        ValueError, match=r"the random effect \(1 \| donor\) is given more"
    ):
        Design(samples, "~ condition", random="(1 | donor) + (1 | donor)")
    with pytest.raises(ValueError, match="random-effect terms such as"):
        Design(samples, "~ condition + (1 | donor)")
    with pytest.raises(TypeError, match="random must be a string"):
        Design(samples, "~ condition", random=3)  # type: ignore[arg-type]


def test_missing_grouping_values_name_the_rows(samples: dict[str, Any]) -> None:
    broken = dict(samples, donor=["D1", None, "D2", "D2", "D3", "D3", "D4", "D4"])
    with pytest.raises(
        ValueError,
        match=r"grouping column 'donor' .* missing values in row 1 \(sample_id S2\)",
    ):
        Design(broken, "~ condition", random="(1 | donor)")


def test_levels_may_name_random_variables_but_not_others(
    samples: dict[str, Any],
) -> None:
    design = Design(
        samples,
        "~ condition",
        random="(1 | donor)",
        levels={"donor": ["D4", "D3", "D2", "D1"]},
    )
    # Only fixed-effect factors have levels in the design; the grouping order
    # does not change a mixed model.
    assert design.levels == {"condition": ("ctrl", "stim")}
    with pytest.raises(ValueError, match=r"levels are given for 'plat'.*'plate'"):
        Design(samples, "~ condition + plate", levels={"plat": ["p2", "p1"]})
    # Levels of a factor in a random slope are checked against the data too.
    with pytest.raises(ValueError, match=r"'plate' has values that are not among"):
        Design(
            samples, "~ condition", random="(plate | donor)", levels={"plate": ["p1"]}
        )


def test_with_contrasts_returns_a_new_design(samples: dict[str, Any]) -> None:
    design = Design(samples, "~ condition")
    assert len(design.contrasts) == 0 and design.contrasts.matrix.shape == (2, 0)
    other = design.with_contrasts({"stim": "conditionstim"})
    assert other.contrasts.names == ("stim",) and len(design.contrasts) == 0
    assert other.matrix is design.matrix
    with pytest.raises(ValueError, match="'conditionstm' is not a column"):
        design.with_contrasts(["conditionstm"])


def test_design_is_immutable_copies_to_itself_and_pickles(
    samples: dict[str, Any],
) -> None:
    design = Design(
        samples,
        "~ condition + age",
        random="(1 | plate/donor)",
        contrasts={"ctrl": "conditionctrl"},
        levels={"condition": ["stim", "ctrl"]},
    )
    with pytest.raises(AttributeError, match="Design is immutable"):
        design.formula = "~ age"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="Design is immutable"):
        del design.formula  # type: ignore[misc]
    assert copy.copy(design) is design and copy.deepcopy(design) is design
    restored = pickle.loads(pickle.dumps(design))
    assert restored.columns == design.columns
    assert_array_equal(restored.matrix, design.matrix)
    assert [str(e) for e in restored.random] == [str(e) for e in design.random]
    assert restored.contrasts.names == ("ctrl",)
    assert restored.levels == design.levels


def test_design_works_for_any_object_with_matrix_and_columns(
    samples: dict[str, Any],
) -> None:
    # What fit functions rely on: a float64 samples x columns matrix and names.
    design = Design(samples, "~ 0 + condition")
    assert design.matrix.shape == (8, 2) and design.matrix.dtype == np.float64
    assert design.columns == ("conditionctrl", "conditionstim")
    assert not design.matrix.flags.writeable


def test_pickling_keeps_category_order_and_unused_categories() -> None:
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "condition": pandas.Categorical(
                ["stim", "ctrl"] * 3, categories=["stim", "ctrl"]
            ),
            "dose": pandas.Categorical(
                ["lo", "mid", "hi"] * 2, categories=["mid", "hi", "lo", "unused"]
            ),
            "donor": ["D1", "D1", "D2", "D2", "D3", "D3"],
        }
    )
    design = Design(
        frame,
        "~ condition + dose",
        random="(condition | donor)",
        contrasts={"ctrl_vs_stim": "conditionctrl"},
    )
    assert design.columns == (
        "(Intercept)",
        "conditionctrl",
        "dosehi",
        "doselo",
        "doseunused",
    )
    restored = pickle.loads(pickle.dumps(design))
    # The sample table holds the categories as text, so a rebuilt design would
    # sort them and flip the reference level; the restored state must not.
    assert restored.columns == design.columns
    assert restored.levels == design.levels
    assert_array_equal(restored.contrasts["ctrl_vs_stim"], [0, 1, 0, 0, 0])
    assert restored.check().kinds() == design.check().kinds()


def test_levels_must_name_columns_and_unused_ones_are_ignored(
    samples: dict[str, Any],
) -> None:
    shared = {"condition": ["stim", "ctrl"], "plate": ["p2", "p1"]}
    design = Design(samples, "~ condition", levels=shared)
    assert design.columns == ("(Intercept)", "conditionctrl")
    with pytest.raises(ValueError, match="'conditon', which is not a column"):
        Design(samples, "~ condition", levels={"conditon": ["stim", "ctrl"]})


def test_contrasts_of_one_design_serve_another_with_the_same_columns(
    samples: dict[str, Any],
) -> None:
    first = Design(samples, "~ condition", contrasts={"stim": "conditionstim"})
    second = Design(dict(samples, age=samples["age"][::-1]), "~ condition")
    assert second.with_contrasts(first.contrasts).contrasts.names == ("stim",)
    assert dict(first.contrasts).keys() == {"stim"}
    # The same columns in another order: the weights follow their columns by name.
    adjusted = Design(samples, "~ condition + age", contrasts={"aged": "age"})
    reordered = Design(samples, "~ age + condition").with_contrasts(adjusted.contrasts)
    assert reordered.columns == ("(Intercept)", "age", "conditionstim")
    assert_array_equal(reordered.contrasts["aged"], [0, 1, 0])
    other = Design(samples, "~ condition + age")
    with pytest.raises(
        ValueError,
        match=r"must weigh exactly the columns \('\(Intercept\)', 'conditionstim' "
        r"and 'age'\), in any order, but 'age' is missing",
    ):
        other.with_contrasts(first.contrasts)


def test_random_effect_errors_name_the_random_effect(samples: dict[str, Any]) -> None:
    broken = dict(samples, age=[30.0, np.nan] + samples["age"][2:])
    with pytest.raises(
        ValueError, match=r"missing values in columns used by the random effect"
    ):
        Design(broken, "~ condition", random="(age | donor)")
