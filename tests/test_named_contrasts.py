"""Tests that contrast weights are matched to coefficients by name, never by position.

Weights that name the coefficients they refer to (a ``Contrasts``, an object laid out
like one, a pandas DataFrame or Series indexed by name) must give the same contrasts
in any row order. The references are limma's ``contrasts.fit`` with the contrasts in
limma's own order, and limma refitted on a design whose coefficient is the contrast
(``tests/data/linear_models``, written by ``validation/linear_models.py``); group
means and their standard errors computed by hand; and fits of designs reparametrised
by their formula, which use no contrast code.
"""

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import Design
from cytepool.design import Contrasts, make_contrasts
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model

LINEAR_MODELS = Path(__file__).parent / "data" / "linear_models"
# The tier for closed-form results.
EXACT = 1e-10
REMEDY = r"make them with make_contrasts\(\.\.\., fit\.coefficient_names\)"


def assert_agrees(
    actual: np.ndarray, desired: np.ndarray, floor: np.ndarray | float = 0.0
) -> None:
    """Agreement to the exact tier, relative to max(|desired|, floor)."""
    scale = np.fmax(np.abs(desired), floor)
    difference = np.abs(actual - desired)
    assert np.all(difference <= EXACT * scale), np.max(difference / scale)


@pytest.fixture(scope="module")
def groups() -> SimpleNamespace:
    """Three groups fitted by their means, so that contrasts have closed forms."""
    labels = np.array(["A"] * 3 + ["B"] * 4 + ["C"] * 5)
    shift = np.select([labels == "B", labels == "C"], [10.0, -4.0], 0.0)
    y = np.random.default_rng(17).normal(5.0, 1.0, (12, 6)) + shift[:, np.newaxis]
    design = Design({"group": labels}, "~ 0 + group")
    return SimpleNamespace(
        y=y,
        design=design,
        fit=fit_linear_model(y, design),
        means={g: y[labels == g].mean(axis=0) for g in "ABC"},
        sizes={g: int(np.sum(labels == g)) for g in "ABC"},
    )


def test_contrast_over_reordered_columns_is_not_reversed() -> None:
    # The reported defect: group B is 10 higher, and contrasts made over the same
    # columns in another order gave about -10.
    labels = np.repeat(["A", "B"], 4)
    noise = np.random.default_rng(3).normal(0.0, 0.5, (8, 5))
    y = noise + 10.0 * (labels == "B")[:, np.newaxis]
    fit = fit_linear_model(y, Design({"group": labels}, "~ 0 + group"))
    contrasts = make_contrasts({"B_vs_A": "groupB - groupA"}, ["groupB", "groupA"])
    assert contrasts.columns != fit.coefficient_names
    contrast = contrasts_fit(fit, contrasts)
    expected = y[labels == "B"].mean(axis=0) - y[labels == "A"].mean(axis=0)
    assert_allclose(contrast.coefficients[:, 0], expected, rtol=1e-12)
    assert np.all(contrast.coefficients > 8.0)
    assert contrast.coefficient_names == ("B_vs_A",)


def limma_case(case: str) -> dict[str, np.ndarray]:
    with np.load(LINEAR_MODELS / f"{case}.npz") as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", ["groups_and_covariate", "observation_weights"])
def test_every_row_order_of_named_contrasts_gives_limmas_contrasts(case: str) -> None:
    data = limma_case(case)
    names = tuple(str(name) for name in data["coefficient_names"])
    contrast_names = tuple(str(name) for name in data["contrast_names"])
    fit = fit_linear_model(
        data["y"], data["design"], weights=data.get("weights"), coefficient_names=names
    )
    results = []
    for order in itertools.permutations(range(len(names))):
        rows = list(order)
        contrasts = Contrasts(
            data["contrasts"][rows], contrast_names, tuple(names[i] for i in rows)
        )
        results.append(
            (
                contrasts_fit(fit, contrasts, method="limma"),
                contrasts_fit(fit, contrasts, method="exact"),
            )
        )
    assert len(results) == 24
    limma, exact = results[0]
    for other_limma, other_exact in results[1:]:
        for name in ("coefficients", "stdev_unscaled"):
            assert_array_equal(getattr(other_limma, name), getattr(limma, name))
            assert_array_equal(getattr(other_exact, name), getattr(exact, name))
        assert other_exact.coefficient_names == contrast_names
    sigma = data["sigma"][:, np.newaxis]
    se = data["contrast_stdev_unscaled"] * sigma
    assert_agrees(limma.coefficients, data["contrast_coefficients"], floor=se)
    assert_agrees(limma.stdev_unscaled, data["contrast_stdev_unscaled"])
    se = data["refit_stdev_unscaled"] * sigma
    assert_agrees(exact.coefficients, data["refit_coefficients"], floor=se)
    assert_agrees(exact.stdev_unscaled, data["refit_stdev_unscaled"])

    # The same contrasts written as expressions over the columns in reverse order.
    written = make_contrasts(
        {
            "BvsA": "groupB",
            "CvsB": "groupC - groupB",
            "B_at_age_50": "Intercept + groupB + 50*age",
        },
        names[::-1],
    )
    assert_array_equal(written.matrix[::-1], data["contrasts"])
    by_expression = contrasts_fit(fit, written, method="limma")
    assert_array_equal(by_expression.coefficients, limma.coefficients)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            ("groupA", "groupB"),
            r"^the contrasts must weigh exactly the fit's coefficients \('groupA', "
            r"'groupB' and 'groupC'\), in any order, but 'groupC' is missing; "
            + REMEDY
            + "$",
        ),
        (
            ("groupC", "age", "groupA", "groupB"),
            r"in any order, but 'age' is not among them; " + REMEDY,
        ),
        (
            ("groupA", "groupb", "groupC"),
            r"but 'groupB' is missing and 'groupb' \(did you mean 'groupB'\?\) is not "
            r"among them",
        ),
        (
            ("groupA", "groupA", "groupC"),
            r"'groupA' labels more than one row of the contrasts",
        ),
    ],
    ids=["missing", "unexpected", "misspelled", "repeated"],
)
def test_contrasts_that_do_not_name_exactly_the_coefficients_raise(
    groups: SimpleNamespace, columns: tuple[str, ...], message: str
) -> None:
    contrasts = Contrasts(np.ones((len(columns), 1)), ("c",), columns)
    with pytest.raises(ValueError, match=message):
        contrasts_fit(groups.fit, contrasts)


def test_a_fit_of_a_bare_matrix_points_to_fitting_the_design(
    groups: SimpleNamespace,
) -> None:
    design = groups.design.with_contrasts({"B_vs_A": "groupB - groupA"})
    bare = fit_linear_model(groups.y, design.matrix)
    with pytest.raises(ValueError, match="fit the Design itself, not its matrix"):
        contrasts_fit(bare, design.contrasts)
    # Weights without names are taken in the order of the coefficients.
    contrast = contrasts_fit(bare, design.contrasts["B_vs_A"])
    assert_allclose(
        contrast.coefficients[:, 0],
        groups.means["B"] - groups.means["A"],
        rtol=1e-12,
    )


def test_pandas_weights_are_matched_by_their_index(groups: SimpleNamespace) -> None:
    pandas = pytest.importorskip("pandas")
    b_vs_a = groups.means["B"] - groups.means["A"]
    c_vs_b = groups.means["C"] - groups.means["B"]
    frame = pandas.DataFrame(
        {"B_vs_A": [0.0, -1.0, 1.0], "C_vs_B": [1.0, 0.0, -1.0]},
        index=["groupC", "groupA", "groupB"],
    )
    by_frame = contrasts_fit(groups.fit, frame)
    assert by_frame.coefficient_names == ("B_vs_A", "C_vs_B")
    assert_allclose(
        by_frame.coefficients, np.column_stack([b_vs_a, c_vs_b]), rtol=1e-12
    )
    sizes = groups.sizes
    assert_allclose(
        by_frame.stdev_unscaled[0],
        [
            np.sqrt(1 / sizes["A"] + 1 / sizes["B"]),
            np.sqrt(1 / sizes["B"] + 1 / sizes["C"]),
        ],
        rtol=1e-13,
    )
    single = contrasts_fit(groups.fit, frame["B_vs_A"])
    assert_allclose(single.coefficients[:, 0], b_vs_a, rtol=1e-12)
    # In a mapping, a Series is matched by name and a list is taken in order.
    mixed = contrasts_fit(groups.fit, {"B_vs_A": frame["B_vs_A"], "C_vs_B": [0, -1, 1]})
    assert_allclose(mixed.coefficients, by_frame.coefficients, rtol=1e-15)

    # The default index 0, 1, ..., n - 1 names nothing: rows are in the fit's order.
    positional = pandas.DataFrame({"B_vs_A": [-1.0, 1.0, 0.0]})
    integers = pandas.DataFrame({"B_vs_A": [-1.0, 1.0, 0.0]}, index=[0, 1, 2])
    assert type(integers.index).__name__ != type(positional.index).__name__
    for weights in (positional, integers, positional["B_vs_A"]):
        contrast = contrasts_fit(groups.fit, weights)
        assert_allclose(contrast.coefficients[:, 0], b_vs_a, rtol=1e-12)


def test_pandas_labels_of_any_string_type_name_rows(groups: SimpleNamespace) -> None:
    pandas = pytest.importorskip("pandas")
    b_vs_a = groups.means["B"] - groups.means["A"]
    labels = ["groupC", "groupA", "groupB"]
    weights = [0.0, -1.0, 1.0]
    for index in (
        pandas.CategoricalIndex(labels),
        pandas.Index(labels, dtype="string"),
        pandas.Index(np.array(labels, dtype=np.str_)),
    ):
        series = pandas.Series(weights, index=index, name="B_vs_A")
        contrast = contrasts_fit(groups.fit, series)
        assert_allclose(contrast.coefficients[:, 0], b_vs_a, rtol=1e-12)
        # A Series names its contrast, as the columns of a DataFrame do.
        assert contrast.coefficient_names == ("B_vs_A",)
    # Columns that pandas shows as attributes do not make a DataFrame a Contrasts.
    frame = pandas.DataFrame({"matrix": weights, "names": weights}, index=labels)
    contrast = contrasts_fit(groups.fit, frame)
    assert contrast.coefficient_names == ("matrix", "names")
    assert_allclose(
        contrast.coefficients, np.column_stack([b_vs_a, b_vs_a]), rtol=1e-12
    )
    # Weights for some of the coefficients only are refused, with a way to add zeros.
    with pytest.raises(
        ValueError,
        match=r"but 'groupC' is missing; .*; to give the missing ones zero weight, "
        r"reindex the weights with fill_value=0$",
    ):
        contrasts_fit(groups.fit, pandas.Series({"groupB": 1.0, "groupA": -1.0}))


def test_selected_contrasts_keep_the_names_that_indexing_drops(
    groups: SimpleNamespace,
) -> None:
    contrasts = make_contrasts(
        {"B_vs_A": "groupB - groupA", "C_vs_A": "groupC - groupA"},
        ["groupB", "groupA", "groupC"],
    )
    assert contrasts.columns != groups.fit.coefficient_names
    # Indexing gives the weights in the order of the contrasts' own columns.
    assert_array_equal(contrasts["C_vs_A"], [0.0, -1.0, 1.0])
    chosen = contrasts.select(["C_vs_A"])
    assert chosen.names == ("C_vs_A",) and chosen.columns == contrasts.columns
    contrast = contrasts_fit(groups.fit, chosen)
    assert_allclose(
        contrast.coefficients[:, 0], groups.means["C"] - groups.means["A"], rtol=1e-12
    )
    assert contrasts.select("B_vs_A").names == ("B_vs_A",)
    both = contrasts.select(["C_vs_A", "B_vs_A"])
    assert both.names == ("C_vs_A", "B_vs_A")
    for name in both:
        assert_array_equal(both[name], contrasts[name])
    with pytest.raises(KeyError, match="no contrast named 'C_vs_B'"):
        contrasts.select(["C_vs_B"])
    with pytest.raises(ValueError, match="'B_vs_A' is selected more than once"):
        contrasts.select(["B_vs_A", "B_vs_A"])


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (
            [2, 0, 1],
            r"the index of the contrasts must hold the names of the fit's coefficients, "
            r"or be the default 0, 1, \.\.\., n - 1 for weights in the order of the "
            r"fit's coefficients; got 2, 0 and 1",
        ),
        ([1, 2, 3], "got 1, 2 and 3"),
        ([True, False, True], "must hold the names of the fit's coefficients"),
        (["groupA", np.nan, "groupC"], "must hold the names of the fit's coefficients"),
        (
            ["groupA", "groupB", "groupD"],
            r"but 'groupC' is missing and 'groupD' \(did you mean 'groupC'\?\) is not "
            r"among them; " + REMEDY,
        ),
        ("multi", "must hold the names of the fit's coefficients"),
    ],
    ids=[
        "reordered_positions",
        "positions_from_one",
        "booleans",
        "missing_label",
        "unknown_name",
        "multi_index",
    ],
)
def test_pandas_index_that_neither_names_nor_counts_raises(
    groups: SimpleNamespace, index: Any, message: str
) -> None:
    pandas = pytest.importorskip("pandas")
    if isinstance(index, str):
        index = pandas.MultiIndex.from_tuples(
            [("groupA", 1), ("groupB", 1), ("groupC", 1)]
        )
    frame = pandas.DataFrame({"c": [-1.0, 1.0, 0.0]}, index=index)
    with pytest.raises(ValueError, match=message):
        contrasts_fit(groups.fit, frame)
    with pytest.raises(
        ValueError, match=message.replace("the contrasts", "contrast 'c'")
    ):
        contrasts_fit(groups.fit, {"c": frame["c"]})


def test_names_given_in_order_may_rename_columns_but_not_move_names(
    groups: SimpleNamespace,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"coefficient_names give column 1 the name 'groupB', the name of column 2 "
        r"of the design; coefficient_names rename the columns in order \('groupA', "
        r"'groupB', 'groupC'\)",
    ):
        fit_linear_model(
            groups.y, groups.design, coefficient_names=["groupB", "groupA", "C"]
        )
    renamed = fit_linear_model(
        groups.y, groups.design, coefficient_names=["A", "groupB", "C"]
    )
    assert renamed.coefficient_names == ("A", "groupB", "C")
    assert_array_equal(renamed.coefficients, groups.fit.coefficients)

    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {"B_vs_A": [-1.0, 1.0, 0.0], "C_vs_B": [0.0, -1.0, 1.0]},
        index=list(groups.fit.coefficient_names),
    )
    with pytest.raises(
        ValueError, match="names give column 1 the name 'C_vs_B', the name of column 2"
    ):
        contrasts_fit(groups.fit, frame, names=["C_vs_B", "B_vs_A"])
    relabelled = contrasts_fit(groups.fit, frame, names=["B_minus_A", "C_vs_B"])
    assert relabelled.coefficient_names == ("B_minus_A", "C_vs_B")


def test_objects_laid_out_like_contrasts_are_matched_by_name(
    groups: SimpleNamespace,
) -> None:
    laid_out = SimpleNamespace(
        matrix=np.array([[1.0], [0.0], [-1.0]]),
        names=["B_vs_A"],
        columns=["groupB", "groupC", "groupA"],
    )
    contrast = contrasts_fit(groups.fit, laid_out)
    assert_allclose(
        contrast.coefficients[:, 0], groups.means["B"] - groups.means["A"], rtol=1e-12
    )
    assert contrast.coefficient_names == ("B_vs_A",)
    with pytest.raises(
        TypeError,
        match="a Design has a matrix and columns but no contrast names, as a design "
        "does, so it cannot give the contrasts; pass the contrasts of a design",
    ):
        contrasts_fit(groups.fit, groups.design)
    with pytest.raises(ValueError, match="names must not be given"):
        contrasts_fit(groups.fit, laid_out, names=["other"])
    with pytest.raises(ValueError, match="at least one contrast"):
        contrasts_fit(groups.fit, groups.design.contrasts)  # a design without any
    shapeless = SimpleNamespace(matrix=np.ones((2, 1)), names=["c"], columns=["groupA"])
    with pytest.raises(
        ValueError, match="one row of weights per name in their columns"
    ):
        contrasts_fit(groups.fit, shapeless)


def test_contrasts_of_contrasts_are_matched_by_contrast_name(
    groups: SimpleNamespace,
) -> None:
    design = groups.design.with_contrasts(
        {"B_vs_A": "groupB - groupA", "C_vs_A": "groupC - groupA"}
    )
    first = contrasts_fit(groups.fit, design.contrasts)
    second = make_contrasts({"C_vs_B": "C_vs_A - B_vs_A"}, ["C_vs_A", "B_vs_A"])
    contrast = contrasts_fit(first, second)
    assert_allclose(
        contrast.coefficients[:, 0], groups.means["C"] - groups.means["B"], rtol=1e-12
    )
    sizes = groups.sizes
    assert_allclose(
        contrast.stdev_unscaled[:, 0],
        np.sqrt(1 / sizes["B"] + 1 / sizes["C"]),
        rtol=1e-13,
    )
    assert_array_equal(contrast.contrasts[:, 0], [0.0, -1.0, 1.0])


def test_make_contrasts_matches_named_weights_to_the_columns_by_name() -> None:
    columns = ("(Intercept)", "groupB", "groupC", "age")
    reordered = make_contrasts(
        {"C_vs_B": "groupC - groupB", "aged": "age"},
        ("age", "groupC", "(Intercept)", "groupB"),
    )
    assert_array_equal(reordered.matrix, [[0, 1], [1, 0], [0, 0], [-1, 0]])
    contrasts = make_contrasts(reordered, columns)
    assert contrasts.columns == columns and contrasts.names == ("C_vs_B", "aged")
    assert_array_equal(contrasts.matrix, [[0, 0], [-1, 0], [1, 0], [0, 1]])
    samples = {"group": ["A", "B", "C"] * 2, "age": [30.0, 41, 52, 38, 45, 60]}
    design = Design(samples, "~ group + age", contrasts=reordered)
    assert_array_equal(design.contrasts.matrix, contrasts.matrix)
    with pytest.raises(
        ValueError,
        match=r"must weigh exactly the columns \('\(Intercept\)', 'groupB' and "
        r"'groupC'\), in any order, but 'age' is not among them; write the contrasts "
        r"for these",
    ):
        make_contrasts(reordered, columns[:3])


def test_make_contrasts_matches_pandas_weights_by_their_index() -> None:
    pandas = pytest.importorskip("pandas")
    columns = ["groupA", "groupB", "groupC"]
    named = pandas.Series({"groupC": 0.0, "groupB": 1.0, "groupA": -1.0})
    unnamed = pandas.Series([0.0, -1.0, 1.0])  # the default index: in column order
    contrasts = make_contrasts({"b_vs_a": named, "c_vs_b": unnamed}, columns)
    assert_array_equal(contrasts.matrix, [[-1, 0], [1, -1], [0, 1]])
    with pytest.raises(
        ValueError,
        match=r"contrast 'b_vs_a' must weigh exactly the columns .* but 'groupC' is "
        r"missing; index the weights by these columns",
    ):
        make_contrasts({"b_vs_a": named.drop("groupC")}, columns)


@pytest.fixture(scope="module")
def experiment() -> SimpleNamespace:
    """Three unbalanced groups with an age effect, gene-specific variances (so the
    prior degrees of freedom are finite) and precision weights."""
    rng = np.random.default_rng(29)
    labels = np.array(["A"] * 5 + ["B"] * 4 + ["C"] * 6)
    age = rng.uniform(20.0, 70.0, labels.size)
    n_features = 60
    effects = rng.normal(0.0, 1.0, (3, n_features)) * (
        rng.random((3, n_features)) < 0.4
    )
    sd = 0.3 * np.sqrt(4.0 / rng.chisquare(4.0, n_features))
    y = (
        rng.uniform(4.0, 10.0, n_features)
        + np.outer(labels == "B", effects[0])
        + np.outer(labels == "C", effects[1])
        + np.outer(age - 45.0, 0.02 * effects[2])
        + rng.normal(0.0, 1.0, (labels.size, n_features)) * sd
    )
    return SimpleNamespace(
        samples={"group": labels, "age": age},
        y=y,
        weights=rng.uniform(0.5, 2.0, y.shape),
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("formula", ["~ group + age", "~ age + group"])
def test_moderated_statistics_of_design_contrasts_match_a_reparametrised_design(
    experiment: SimpleNamespace, formula: str, weighted: bool
) -> None:
    samples, y = experiment.samples, experiment.y
    weights = experiment.weights if weighted else None
    design = Design(
        samples,
        "~ group + age",
        contrasts={
            "C_vs_B": "groupC - groupB",
            "B_at_age_50": "(Intercept) + groupB + 50*age",
        },
    )
    # The fit's coefficients are the design's columns, or the same columns in another
    # order; the design's contrasts must be matched to them by name.
    fitted_design = Design(samples, formula)
    fit = fit_linear_model(y, fitted_design, weights=weights)
    assert set(fit.coefficient_names) == set(design.columns)
    assert (fit.coefficient_names == design.columns) == (formula == design.formula)
    contrasted = contrasts_fit(fit, design.contrasts)
    moderated = moderated_statistics(contrasted)
    assert np.isfinite(moderated.df_prior)  # the variances are moderated

    # The independent route: designs whose coefficients are the contrasts, from a
    # change of reference level and from age centred at 50 with group means.
    routes = {
        "C_vs_B": (
            Design(samples, "~ group + age", levels={"group": ["B", "A", "C"]}),
            "groupC",
        ),
        "B_at_age_50": (
            Design(
                {"group": samples["group"], "age_50": samples["age"] - 50.0},
                "~ 0 + group + age_50",
            ),
            "groupB",
        ),
    }
    for j, name in enumerate(contrasted.coefficient_names):
        refit_design, coefficient = routes[name]
        refit = fit_linear_model(y, refit_design, weights=weights)
        k = refit.coefficient_names.index(coefficient)
        reference = moderated_statistics(refit)
        se = refit.stdev_unscaled[:, k] * refit.sigma
        assert_agrees(contrasted.coefficients[:, j], refit.coefficients[:, k], floor=se)
        assert_agrees(contrasted.stdev_unscaled[:, j], refit.stdev_unscaled[:, k])
        # t-statistics and log-odds near zero carry absolute, not relative, precision.
        assert_agrees(moderated.t[:, j], reference.t[:, k], floor=1.0)
        assert_agrees(moderated.p_value[:, j], reference.p_value[:, k])
        assert_agrees(moderated.log_odds[:, j], reference.log_odds[:, k], floor=1.0)
    # Not a check of the contrasts, which these do not depend on, but of what the
    # contrasted fit carries over: the residual variances of each feature, and so
    # the prior and posterior variances of any parametrisation.
    assert_agrees(moderated.posterior_variance, reference.posterior_variance)
    assert_agrees(moderated.df_total, reference.df_total)
    # The contrasts are not trivial: they differ from every coefficient of the fit.
    for j in range(contrasted.coefficients.shape[1]):
        gaps = np.abs(fit.coefficients - contrasted.coefficients[:, [j]]).max(axis=0)
        assert np.all(gaps > 0.1)
