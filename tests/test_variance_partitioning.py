"""Tests for variance partitioning: variance_fractions, variance_partitioning, figures.

Reference values in ``tests/data/variance_partitioning`` come from variancePartition
1.36.2 (lme4 1.1-38), as ``validation/variance_partitioning.py`` ran it:
``fitExtractVarPartModel`` and ``calcVarPart`` of the first 20 genes of each case (one
random effect; several random effects with two covariates and observation weights,
also by REML; unbalanced donors with sample weights; nested random effects; a linear
model with factors), lme4 and ``calcVarPart`` evaluated at CytePool's estimates, the
fractions of two cell types of a cube after ``voomWithDreamWeights``, and what
``plotVarPart`` and ``plotPercentBars`` draw for CytePool's fractions. The other tests
check closed forms (the maximum likelihood and REML estimates of a balanced one-way
design, the squared correlation of a single covariate, the sums of squares of an
orthogonal design), invariants (fractions sum to one; relabelling levels, reordering
samples, an affine change of the values and the scale of the weights change nothing;
a variable without effect explains almost nothing) and messages.
"""

import json
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.cube import Cube, from_bulk
from cytepool.design import Design
from cytepool.normalization import filter_by_expression, tmm_factors
from cytepool.stats.dream import voom_with_dream_weights
from cytepool.stats.voom import voom
from cytepool.table import Table
from cytepool.variance_partitioning import (
    VarianceFractions,
    VariancePartitioningResult,
    variance_fractions,
    variance_partitioning,
)

DATA = Path(__file__).parent / "data" / "variance_partitioning"
# lme4's fits stop at about 1e-8 of the optimum (see the validation script); closed
# forms at given estimates and linear models agree to rounding.
CLOSE = 1e-6
EXACT = 1e-10
# Fits of the same data in another order or scale reach the same optimum to the
# precision of Newton's method, which stops when no step exceeds 1e-10.
REFIT = 1e-8
CASES = ["one_random", "several_random", "unbalanced", "nested", "linear", "reml"]


@cache
def fixture(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((DATA / f"{name}.json").read_text())


@cache
def fixture_fractions(name: str) -> VarianceFractions:
    data, case = fixture(name)
    design = Design(case["samples"], case["fixed"], random=case["random"])
    return variance_fractions(
        data["values"], design, weights=data.get("weights"), method=case["method"]
    )


def scaled(ours: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference relative to each gene's total variance."""
    total = np.sum(reference, axis=1, keepdims=True)
    return float(np.max(np.abs(ours - reference) / total))


def one_way(seed: int, groups: int, per_group: int, spread: float) -> tuple:
    """Values of a balanced one-way design with a random group effect of SD spread."""
    rng = np.random.default_rng(seed)
    labels = [f"G{g}" for g in np.repeat(np.arange(groups), per_group)]
    effects = np.repeat(rng.normal(0.0, spread, (groups, 3)), per_group, axis=0)
    return labels, 4.0 + effects + rng.normal(size=(groups * per_group, 3))


def donors_and_batches(seed: int, n_donors: int, per_donor: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = n_donors * per_donor
    return {
        "donor": [f"D{d}" for d in np.repeat(np.arange(n_donors), per_donor)],
        "batch": [f"B{b}" for b in rng.integers(3, size=n)],
        "age": np.repeat(rng.normal(50.0, 10.0, n_donors), per_donor),
        "rin": rng.normal(7.0, 1.0, n),
    }


# Against variancePartition.


@pytest.mark.parametrize("name", CASES)
def test_the_fractions_and_variances_match_variance_partition(name: str) -> None:
    data, case = fixture(name)
    result = fixture_fractions(name)
    genes = data["genes"]

    assert result.variables == tuple(data["variables"].tolist())
    tolerance = EXACT if case["random"] is None else CLOSE
    assert scaled(result.fractions[genes], data["fractions"]) <= tolerance
    assert scaled(result.variances[genes], data["variances"]) <= tolerance
    assert_allclose(result.fractions.sum(axis=1), 1.0, rtol=EXACT)
    assert result.method == case["method"]


@pytest.mark.parametrize("name", [c for c in CASES if c != "linear"])
def test_the_variances_are_lme4s_and_calcvarparts_at_the_same_estimates(
    name: str,
) -> None:
    data, _ = fixture(name)
    result = fixture_fractions(name)

    assert scaled(result.variances, data["given"]) <= EXACT


def cube_fixture() -> tuple[Cube, Design, dict[str, np.ndarray]]:
    with np.load(DATA / "cube.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    samples = json.loads((DATA / "cube.json").read_text())
    cube = Cube(
        {"counts": data["counts"]},
        samples=samples,
        groups={"group_id": ["B", "NK", "rare"]},
        features={"feature_id": [f"gene{i}" for i in range(data["counts"].shape[2])]},
        n_cells=data["n_cells"],
    )
    design = Design(cube.samples, "~ age", random="(1 | donor) + (1 | condition)")
    return cube, design, data


@cache
def cube_result() -> VariancePartitioningResult:
    cube, design, _ = cube_fixture()
    return variance_partitioning(cube, design, prior_weights="n_cells", span=0.5)


def test_variance_partitioning_matches_variance_partition_on_every_cell_type() -> None:
    _, _, data = cube_fixture()
    result = cube_result()

    assert result.variables == ("donor", "condition", "age", "residuals")
    for group in ("B", "NK"):
        table = result.table(group)
        fractions = np.column_stack([table[name] for name in result.variables])
        genes = data[f"{group} genes"]
        assert scaled(fractions[genes], data[f"{group} fractions"]) <= CLOSE
    # Every donor keeps one sample in "rare": the donor variance cannot be told
    # from the residual variance, and variancePartition stops.
    summary = result.summary
    assert summary["status"].tolist() == [
        "fitted",
        "fitted",
        "random_effects_not_estimable",
    ]
    assert "rare" not in result.tables


# Closed forms.


@pytest.mark.parametrize(
    ("spread", "method"), [(2.0, "ml"), (2.0, "reml"), (0.0, "ml"), (0.0, "reml")]
)
def test_a_balanced_one_way_design_has_the_classical_estimates(
    spread: float, method: str
) -> None:
    # Searle, Casella and McCulloch (1992, sections 3.7 and 3.8): with q groups of m
    # samples, sigma_e^2 = MSW and sigma_a^2 = (SSB / q - MSW) / m by maximum
    # likelihood, (MSB - MSW) / m by REML; at zero, sigma_e^2 = SST / (qm) or
    # SST / (qm - 1).
    q, m = 6, 4
    labels, y = one_way(3, q, m, spread)
    result = variance_fractions(
        y, Design({"group": labels}, "~ 1", random="(1 | group)"), method=method
    )

    for g in range(3):
        values = y[:, g].reshape(q, m)
        means = values.mean(axis=1)
        within = float(np.sum((values - means[:, None]) ** 2))
        between = m * float(np.sum((means - values.mean()) ** 2))
        msw = within / (q * (m - 1))
        group = (
            (between / q - msw) / m if method == "ml" else (between / (q - 1) - msw) / m
        )
        if group > 0.0:
            expected = [group, msw]
        else:
            total = within + between
            expected = [0.0, total / (q * m if method == "ml" else q * m - 1)]
        assert_allclose(result.variances[g], expected, rtol=REFIT, atol=0.0)
        assert_allclose(
            result.fractions[g], np.divide(expected, sum(expected)), rtol=REFIT
        )
    if spread == 0.0:
        assert np.any(result.fractions[:, 0] == 0.0)


def test_a_covariate_alone_explains_its_squared_correlation() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=20)
    y = np.column_stack([0.5 * x + rng.normal(size=20), rng.normal(size=20)])
    result = variance_fractions(y, Design({"x": x}, "~ x"))

    for g in range(2):
        r = np.corrcoef(x, y[:, g])[0, 1]
        assert_allclose(result.fractions[g], [r**2, 1.0 - r**2], rtol=EXACT)
    assert result.variables == ("x", "residuals")
    assert result.random_effects == () and result.fixed_effects == ("x",)


def test_the_terms_of_an_orthogonal_design_split_the_sum_of_squares() -> None:
    # A balanced two-way layout without interaction: each term's variance is its sum
    # of squares over n, from the means of its levels, and the residual is the rest.
    a = np.tile(["a1", "a2"], 12)
    b = np.repeat(["b1", "b2", "b3"], 8)
    rng = np.random.default_rng(5)
    y = rng.normal(size=(24, 4)) + (a == "a2")[:, None] * 0.8 + (b == "b3")[:, None]
    result = variance_fractions(y, Design({"a": a, "b": b}, "~ a + b"))

    for g in range(4):
        total = float(np.sum((y[:, g] - y[:, g].mean()) ** 2))
        squares = [
            sum(
                np.sum(labels == level)
                * (y[labels == level, g].mean() - y[:, g].mean()) ** 2
                for level in np.unique(labels)
            )
            for labels in (a, b)
        ]
        expected = [*squares, total - sum(squares)]
        assert_allclose(result.variances[g], np.divide(expected, 24), rtol=EXACT)
        assert_allclose(result.fractions[g], np.divide(expected, total), rtol=EXACT)


def test_the_fixed_effects_of_a_mixed_model_share_the_variance_of_their_fit() -> None:
    # With random effects, the fixed terms share the weighted variance of all the
    # fitted fixed effects; without them, each has the variance of its own fit.
    samples = donors_and_batches(6, 8, 4)
    rng = np.random.default_rng(6)
    y = (
        rng.normal(size=(32, 5))
        + 0.05 * samples["age"][:, None]
        + samples["rin"][:, None]
    )
    weights = rng.uniform(0.5, 2.0, (32, 5))
    mixed = variance_fractions(
        y, Design(samples, "~ age + rin", random="(1 | donor)"), weights=weights
    )
    fit = mixed.fit
    fitted = (fit.coefficients[:, 1:] @ fit.design[:, 1:].T).T  # samples x genes
    means = np.sum(weights * fitted, axis=0) / np.sum(weights, axis=0)
    total = np.sum(weights * (fitted - means) ** 2, axis=0) / np.sum(weights, axis=0)

    assert_allclose(mixed.variances[:, 1:3].sum(axis=1), total, rtol=EXACT)


# Invariants.


def test_relabelling_reordering_rescaling_and_reweighting_change_nothing() -> None:
    samples = donors_and_batches(7, 8, 3)
    rng = np.random.default_rng(7)
    y = rng.normal(size=(24, 6)) + np.repeat(rng.normal(size=(8, 6)), 3, axis=0)
    weights = rng.uniform(0.5, 2.0, (24, 6))
    design = Design(samples, "~ age", random="(1 | donor) + (1 | batch)")
    reference = variance_fractions(y, design, weights=weights)

    order = rng.permutation(24)
    relabelled = {
        "donor": [f"X{9 - int(d[1:])}" for d in np.asarray(samples["donor"])[order]],
        "batch": [
            {"B0": "c", "B1": "a", "B2": "b"}[b]
            for b in np.asarray(samples["batch"])[order]
        ],
        "age": samples["age"][order],
    }
    moved = variance_fractions(
        y[order],
        Design(relabelled, "~ age", random="(1 | donor) + (1 | batch)"),
        weights=weights[order] * 10.0,
    )
    assert_allclose(moved.fractions, reference.fractions, rtol=REFIT, atol=REFIT)
    shifted = variance_fractions(3.0 * y + 7.0, design, weights=weights)
    assert_allclose(shifted.fractions, reference.fractions, rtol=REFIT, atol=REFIT)
    assert_allclose(
        shifted.variances, 9.0 * reference.variances, rtol=REFIT, atol=REFIT
    )
    assert_allclose(reference.fractions.sum(axis=1), 1.0, rtol=EXACT)


def test_weights_of_one_are_no_weights_and_constant_weights_change_no_fraction() -> (
    None
):
    labels, y = one_way(8, 5, 3, 1.0)
    design = Design({"donor": labels}, "~ 1", random="(1 | donor)")
    unweighted = variance_fractions(y, design)

    assert_array_equal(
        variance_fractions(y, design, weights=np.ones(15)).fractions,
        unweighted.fractions,
    )
    assert_allclose(
        variance_fractions(y, design, weights=np.full(15, 7.0)).fractions,
        unweighted.fractions,
        rtol=REFIT,
    )


def test_a_variable_without_effect_explains_almost_nothing() -> None:
    samples = donors_and_batches(9, 10, 4)
    rng = np.random.default_rng(9)
    donor = np.repeat(rng.normal(0.0, 1.5, (10, 200)), 4, axis=0)
    y = donor + rng.normal(size=(40, 200))  # no batch effect
    result = variance_fractions(
        y, Design(samples, "~ 1", random="(1 | donor) + (1 | batch)")
    )
    batch = result.fractions[:, 1]

    assert np.median(batch) < 0.01 and np.mean(batch) < 0.03
    assert np.mean(batch == 0.0) > 0.4  # held at the boundary
    assert np.median(result.fractions[:, 0]) > 0.5


def test_features_without_a_fit_have_no_fractions() -> None:
    labels, y = one_way(10, 5, 3, 1.0)
    y[:, 0] = 2.0  # constant
    y[:, 1] = np.repeat(np.arange(5.0), 3)  # constant within every donor
    result = variance_fractions(
        y, Design({"donor": labels}, "~ 1", random="(1 | donor)")
    )

    assert np.all(np.isnan(result.fractions[:2])) and not np.any(result.converged[:2])
    assert result.converged[2] and np.all(np.isfinite(result.fractions[2]))
    linear = variance_fractions(y, Design({"donor": labels}, "~ donor"))
    assert np.all(np.isnan(linear.fractions[0])) and not linear.converged[0]
    assert_allclose(linear.fractions[1], [1.0, 0.0], atol=EXACT)


# Messages.


def samples_of_six() -> dict[str, Any]:
    return {
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3"],
        "condition": ["ctrl", "stim"] * 3,
        "age": [30.0, 30.0, 40.0, 40.0, 55.0, 55.0],
        "months": [360.0, 360.0, 480.0, 480.0, 660.0, 660.0],
        "residuals": ["a", "a", "b", "b", "c", "c"],
    }


@pytest.mark.parametrize(
    ("formula", "random", "message"),
    [
        ("~ condition", "(1 | donor)", "every categorical variable as a random effect"),
        ("~ 0 + age", "(1 | donor)", "has no intercept"),
        ("~ age", "(1 | residuals)", "named 'residuals'"),
        ("~ age", "(1 | age)", "both random and fixed effects"),
        ("~ 1", "(age | donor)", "random slopes"),
        ("~ age + months", None, "not of full column rank"),
        ("~ donor + condition + age", None, "not of full column rank"),
    ],
)
def test_designs_that_cannot_be_partitioned_are_refused(
    formula: str, random: str | None, message: str
) -> None:
    design = Design(samples_of_six(), formula, random=random)
    with pytest.raises(ValueError, match=message):
        variance_fractions(np.random.default_rng(11).normal(size=(6, 2)), design)


def test_invalid_values_weights_and_options_are_refused() -> None:
    design = Design(samples_of_six(), "~ age", random="(1 | donor)")
    linear = Design(samples_of_six(), "~ age")
    y = np.random.default_rng(12).normal(size=(6, 2))
    for target in (design, linear):
        with pytest.raises(ValueError, match="must be finite"):
            variance_fractions(np.where(y > 1.0, np.nan, y), target)
        with pytest.raises(ValueError, match="finite and positive"):
            variance_fractions(y, target, weights=np.array([1, 1, 0, 1, 1, 1.0]))
        with pytest.raises(ValueError, match=r"shape \(6,\)"):
            variance_fractions(y, target, weights=np.ones(5))
        with pytest.raises(ValueError, match="rows"):
            variance_fractions(y[:5], target)
    with pytest.raises(ValueError, match="'reml'"):
        variance_fractions(y, design, method="bayes")
    with pytest.raises(TypeError, match="method"):
        variance_fractions(y, design, method=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a Design"):
        variance_fractions(y, np.ones((6, 1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fits them exactly"):
        variance_fractions(y[:2], Design({"x": [1.0, 2.0]}, "~ x"))


# Every group of a cube.


def small_cube(rng: np.random.Generator) -> Cube:
    """Four groups of 8 donors x 2 conditions: one fine, one without samples, one whose
    samples have one age, one with one gene with counts."""
    donors = np.repeat(np.arange(8), 2)
    samples = {
        "sample_id": [f"S{i}" for i in range(16)],
        "donor": [f"D{d}" for d in donors],
        "condition": ["ctrl", "stim"] * 8,
        "age": np.repeat([30.0, 30.0, 41.0, 47.0, 52.0, 58.0, 63.0, 70.0], 2),
    }
    counts = rng.negative_binomial(10, 0.2, size=(4, 16, 30))
    counts[3, :, 1:] = 0  # one gene with counts
    n_cells = np.full((16, 4), 50)
    n_cells[:, 1] = 3
    n_cells[4:, 2] = 2  # only D0 and D1, both 30 years old
    return Cube(
        {"counts": counts},
        samples=samples,
        groups={"group_id": ["fine", "empty", "one_age", "sparse"]},
        features={"feature_id": [f"g{j}" for j in range(30)]},
        n_cells=n_cells,
    )


def test_groups_that_cannot_be_partitioned_say_why_and_the_rest_are_fitted() -> None:
    cube = small_cube(np.random.default_rng(13))
    design = Design(cube.samples, "~ age", random="(1 | donor) + (1 | condition)")
    result = variance_partitioning(
        cube, design, expression_filter=False, prior_weights="n_cells"
    )

    statuses = dict(zip(result.groups, result.summary["status"].tolist(), strict=True))
    assert statuses == {
        "fine": "fitted",
        "empty": "no_samples",
        "one_age": "fixed_effects_not_estimable",
        "sparse": "too_few_features",
    }
    reasons = dict(zip(result.groups, result.summary["reason"].tolist(), strict=True))
    assert "age = 30*(Intercept)" in reasons["one_age"]
    with pytest.raises(KeyError, match="was not fitted \\(no_samples\\)"):
        result.table("empty")
    with pytest.raises(KeyError, match="no group 'other'"):
        result.table("other")
    with pytest.raises(
        ValueError, match="the result has 4 groups; name one with group"
    ):
        result.table()
    table = result.table("fine")
    assert table.columns == ("feature_id", "donor", "condition", "age", "residuals")
    fractions = np.column_stack([table[name] for name in result.variables])
    assert_allclose(fractions.sum(axis=1), 1.0, rtol=EXACT)
    # The weights of a group are those voomWithDreamWeights gives its counts.
    counts = np.asarray(cube.layer("counts", 0), dtype=float)
    weighted = voom_with_dream_weights(
        counts,
        design,
        norm_factors=tmm_factors(counts),
        weights=cube.n_cells[:, 0].astype(float),
    )
    direct = variance_fractions(weighted.log_cpm, design, weights=weighted.weights)
    assert_allclose(fractions, direct.fractions, rtol=EXACT, atol=EXACT)
    assert result.summary["n_features"].tolist() == [30, 0, 0, 0]
    assert int(result.summary["n_samples_left_out"][1]) == 16


def test_a_design_without_random_effects_is_fitted_after_voom() -> None:
    cube = small_cube(np.random.default_rng(14)).select(groups=["fine"])
    design = Design(cube.samples, "~ age + condition")
    result = variance_partitioning(
        cube, design, expression_filter=False, normalization=None
    )
    counts = np.asarray(cube.layer("counts", 0), dtype=float)
    weighted = voom(counts, design.matrix)
    direct = variance_fractions(weighted.log_cpm, design, weights=weighted.weights)
    table = result.table("fine")

    assert result.table().equals(table)  # the only group needs no name
    assert result.variables == ("age", "condition", "residuals")
    assert_allclose(
        np.column_stack([table[name] for name in result.variables]),
        direct.fractions,
        rtol=EXACT,
        atol=EXACT,
    )


def test_a_filter_by_sample_groups_keeps_the_genes_of_any_one_group() -> None:
    fine = small_cube(np.random.default_rng(14)).select(groups=["fine"])
    counts = fine.layer("counts", "fine").copy()
    counts[2:, 0] = 0  # g0: counted in the two samples of donor D0 only
    counts[1:, 1] = 0  # g1: counted in one sample only
    cube = Cube(
        {"counts": counts[np.newaxis]},
        samples=fine.samples,
        groups=fine.groups,
        features=fine.features,
        n_cells=fine.n_cells,
    )
    design = Design(cube.samples, "~ age + condition")
    by_donor = variance_partitioning(
        cube, design, expression_filter={"sample_groups": "donor"}, normalization=None
    )
    keep = filter_by_expression(counts, group=cube.samples["donor"])
    tested = by_donor.table()["feature_id"].tolist()
    assert tested == cube.features["feature_id"][keep].tolist()
    assert "g0" in tested and "g1" not in tested
    by_design = variance_partitioning(cube, design, normalization=None)
    assert "g0" not in by_design.table()["feature_id"].tolist()


def test_the_result_records_its_samples_settings_and_provenance() -> None:
    result = cube_result()
    record = result.provenance
    design = record["design"]

    assert design["random"] == ["(1 | donor)", "(1 | condition)"]
    assert {"donor", "condition", "age", "sample_id"} <= set(design["samples"])
    assert record["method"] == "variance_partitioning"
    assert result.settings["method"] == "ml" and result.settings["weights_fit"] == "ml"
    groups = {entry["group_id"]: entry for entry in record["groups"]}
    nk = groups["NK"]
    assert len(nk["left_out"]) == 3 and len(nk["samples"]) == 13
    assert nk["features_not_fitted"] == 0 and len(nk["prior_weights"]) == 13
    assert groups["rare"]["prior_weights"] is None
    samples = result.samples
    used = samples["used"][samples["group_id"] == "NK"]
    assert int(np.count_nonzero(~used)) == 3
    assert np.all(np.isnan(samples["prior_weight"][samples["group_id"] == "rare"]))
    restored = pickle.loads(pickle.dumps(result))
    assert restored.table("B").equals(result.table("B"))
    with pytest.raises(AttributeError, match="immutable"):
        result.summary = result.samples  # type: ignore[misc]
    assert "3 groups, 2 fitted" in repr(result)
    assert "rare: not fitted (random_effects_not_estimable)" in repr(result)
    assert max(map(len, repr(result).splitlines())) <= 80


def test_cube_options_and_names_are_checked() -> None:
    cube = from_bulk(
        np.random.default_rng(15).poisson(50, (6, 10)),
        samples={"sample_id": [f"S{i}" for i in range(6)]},
    )
    samples = {
        "sample_id": [f"S{i}" for i in range(6)],
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3"],
        "symbol": ["a", "a", "b", "b", "c", "c"],
    }
    with pytest.raises(ValueError, match="named 'symbol'"):
        variance_partitioning(cube, Design(samples, "~ 1", random="(1 | symbol)"))
    design = Design(samples, "~ 1", random="(1 | donor)")
    with pytest.raises(ValueError, match="does not record"):
        variance_partitioning(cube, design, prior_weights="n_cells")
    with pytest.raises(ValueError, match="'reml'"):
        variance_partitioning(cube, design, method="map")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a Cube"):
        variance_partitioning(np.ones((6, 10)), design)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="no group with id 'T'"):
        variance_partitioning(cube, design, groups="T")


# Figures.


def figure_table() -> tuple[Table, dict[str, np.ndarray]]:
    with np.load(DATA / "figures.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    fractions = data["fractions"]
    columns: dict[str, Any] = {
        "feature_id": [f"g{i + 1}" for i in range(len(fractions))]
    }
    for j, name in enumerate(data["variables"].tolist()):
        columns[name] = fractions[:, j]
    return Table(columns), data


def test_the_figures_draw_what_plotvarpart_and_plotpercentbars_draw() -> None:
    figure = pytest.importorskip("matplotlib.figure")
    from cytepool import plotting

    table, data = figure_table()
    ax = plotting.plot_variance_fractions(table, ax=figure.Figure().subplots())
    labels = [text.get_text() for text in ax.get_xticklabels()]
    assert labels == data["labels"].tolist()
    children = ax.get_children()
    boxes = [a for a in children if a.get_gid() == "boxes"]
    assert_allclose(
        [b.get_y() for b in boxes], data["box_lower"], rtol=EXACT, atol=1e-12
    )
    assert_allclose(
        [b.get_y() + b.get_height() for b in boxes], data["box_upper"], rtol=EXACT
    )
    medians = [a.get_ydata()[0] for a in children if a.get_gid() == "medians"]
    assert_allclose(medians, data["box_middle"], rtol=EXACT, atol=1e-12)
    whiskers = [a.get_ydata()[1] for a in children if a.get_gid() == "whiskers"]
    assert_allclose(whiskers[0::2], data["box_ymin"], rtol=EXACT, atol=1e-12)
    assert_allclose(whiskers[1::2], data["box_ymax"], rtol=EXACT)
    outliers = [
        np.sort(a.get_offsets()[:, 1]) for a in children if a.get_gid() == "outliers"
    ]
    assert [o.size for o in outliers] == data["outlier_counts"].tolist()
    expected = np.split(data["outliers"], np.cumsum(data["outlier_counts"])[:-1])
    for drawn, theirs in zip(outliers, expected, strict=True):
        assert_allclose(drawn, np.sort(theirs), rtol=EXACT)
    # The violins: R's density() bins the values, so the outlines agree to 1e-3 of
    # their widest point (see the validation script).
    violins = {a.get_label(): a for a in children if a.get_gid() == "violins"}
    for position, label in enumerate(labels, start=1):
        vertices = violins[label].get_paths()[0].vertices
        right = vertices[vertices[:, 0] >= position]
        order = np.argsort(right[:, 1], kind="stable")
        rows = data["violin_group"] == position
        drawn = np.interp(
            data["violin_y"][rows], right[order, 1], right[order, 0] - position
        )
        assert np.max(np.abs(drawn - 0.45 * data["violin_width"][rows])) <= 0.45e-3

    genes = data["bar_genes"].tolist()
    ax = plotting.plot_gene_variance_fractions(
        table, genes, ax=figure.Figure().subplots()
    )
    segments = {
        (gene, container.get_label()): (patch.get_x(), patch.get_width())
        for container in ax.containers
        for gene, patch in zip(genes, container, strict=True)
    }
    for gene, variable, left, width in zip(
        data["bar_gene"],
        data["bar_variable"],
        data["bar_left"],
        data["bar_width"],
        strict=True,
    ):
        x, drawn_width = segments[gene, variable]
        assert drawn_width == pytest.approx(width, abs=1e-12)
        if width > 0.0:  # ggplot stacks a segment without width anywhere
            assert x == pytest.approx(left, abs=1e-12)
    top = sorted(ax.get_yticklabels(), key=lambda text: -text.get_position()[1])
    assert [text.get_text() for text in top] == genes


def test_figures_take_fractions_results_and_tables_and_say_what_is_wrong() -> None:
    figure = pytest.importorskip("matplotlib.figure")
    from cytepool import plotting

    fractions = fixture_fractions("several_random")
    ax = plotting.plot_variance_fractions(
        fractions, sort=False, ax=figure.Figure().subplots()
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "donor",
        "batch",
        "condition",
        "age",
        "rin",
        "Residuals",
    ]
    ax = plotting.plot_gene_variance_fractions(
        fractions, [2, 0], ax=figure.Figure().subplots()
    )
    widths = np.array([[patch.get_width() for patch in c] for c in ax.containers]).T
    assert_allclose(widths.sum(axis=1), 100.0, rtol=EXACT)
    labelled = plotting.plot_gene_variance_fractions(
        fractions,
        ["x2"],
        labels=[f"x{i}" for i in range(20)],
        ax=figure.Figure().subplots(),
    )
    assert [t.get_text() for t in labelled.get_yticklabels()] == ["x2"]
    result = cube_result()
    plotting.plot_variance_fractions(result, group="B", ax=figure.Figure().subplots())
    table, _ = figure_table()

    with pytest.raises(ValueError, match="2 fitted groups; name one"):
        plotting.plot_variance_fractions(result, ax=figure.Figure().subplots())
    with pytest.raises(KeyError, match="not fitted"):
        plotting.plot_variance_fractions(result, group="rare")
    with pytest.raises(ValueError, match="no 'residuals' column"):
        plotting.plot_variance_fractions(table.select(["feature_id", "donor"]))
    with pytest.raises(ValueError, match="group names a group"):
        plotting.plot_variance_fractions(fractions, group="B")
    with pytest.raises(TypeError, match="data must be"):
        plotting.plot_variance_fractions(np.ones((3, 2)))
    with pytest.raises(KeyError, match="no gene 'g99'"):
        plotting.plot_gene_variance_fractions(table, ["g99"])
    with pytest.raises(KeyError, match="name no genes"):
        plotting.plot_gene_variance_fractions(fractions, ["g1"])
    with pytest.raises(ValueError, match="more than once"):
        plotting.plot_gene_variance_fractions(table, ["g1", "g1"])
    with pytest.raises(ValueError, match="genes is empty"):
        plotting.plot_gene_variance_fractions(table, [])
    with pytest.raises(IndexError, match="out of range"):
        plotting.plot_gene_variance_fractions(fractions, [20])
    many = Table(
        {f"v{i}": np.full(3, 0.1) for i in range(9)} | {"residuals": np.full(3, 0.1)}
    )
    with pytest.raises(ValueError, match="more than the 8 colours"):
        plotting.plot_variance_fractions(many)
    unfitted = Table({"feature_id": ["a"], "donor": [np.nan], "residuals": [np.nan]})
    with pytest.raises(ValueError, match="no gene has fractions"):
        plotting.plot_variance_fractions(unfitted)
    with pytest.raises(ValueError, match="'a' has no fractions"):
        plotting.plot_gene_variance_fractions(unfitted, ["a"])
