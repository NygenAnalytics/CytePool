"""Tests for the gene-set tests camera and fry.

Reference values in ``tests/data/gene_set_tests`` come from limma's ``camera`` and
``fry`` (and, for camera's robust option, which limma's camera lacks, from ``cameraPR``
and ``rankSumTestWithCorrelation`` on limma's robust moderated z-scores);
``validation/gene_set_tests.py`` writes them. The other tests check properties that a
wrong implementation would break: symmetries, invariances, closed forms, limits of the
standardizations and the documented edge cases. A differential-expression result is
tested, bit for bit, as the voom values of its group's fit with the design rows and
contrast weights fitted and the moderation the result used.
"""

import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from cytepool import (
    Cube,
    Design,
    DifferentialExpressionResult,
    Table,
    camera,
    dream_voom,
    fry,
    limma_trend,
    limma_voom,
)
from cytepool.gene_set_tests import _trimmed_mean_square_expectation
from cytepool.stats.voom import voom

DATA = Path(__file__).parent / "data" / "gene_set_tests"
EXACT = 1e-10
# Robust moderation: limma solves the robust prior df only to a tolerance.
CLOSE = 1e-6
# fry's posterior_sd: limma's null expectation of the trimmed mean square is about
# 7.3e-5 (d + 1) / d above the exact one, which moves p-values by about 1e-4 of their
# value; with d df instead of limma's 0.92 d they would differ by 2e-2.
POSTERIOR = 1e-3
# Run name: (test, inter-gene correlation, ranks, negative allowed, trend,
# standardize, robust), as in the validation script.
RUNS = {
    "camera": ("camera", 0.01, False, False, False, "", False),
    "camera_estimated": ("camera", None, False, False, False, "", False),
    "camera_negative": ("camera", None, False, True, False, "", False),
    "camera_ranks": ("camera", 0.01, True, False, False, "", False),
    "camera_ranks_estimated": ("camera", None, True, False, False, "", False),
    "camera_trend": ("camera", 0.01, False, False, True, "", False),
    "camera_robust": ("camera", 0.01, False, False, False, "", True),
    "camera_robust_estimated": ("camera", None, False, False, True, "", True),
    "camera_robust_ranks": ("camera", None, True, False, False, "", True),
    "fry_none": ("fry", None, False, False, False, "none", False),
    "fry_residual": ("fry", None, False, False, False, "residual_sd", False),
    "fry_posterior": ("fry", None, False, False, False, "posterior_sd", False),
    "fry_posterior_trend": ("fry", None, False, False, True, "posterior_sd", False),
    "fry_posterior_robust": ("fry", None, False, False, False, "posterior_sd", True),
}
COLUMNS = {
    "NGenes": "n_genes",
    "Correlation": "correlation",
    "PValue": "p_value",
    "FDR": "adjusted_p_value",
    "PValue.Mixed": "mixed_p_value",
    "FDR.Mixed": "mixed_adjusted_p_value",
}
CASES = (
    "groups",
    "groups_large_sets",
    "groups_weights",
    "voom",
    "voom_weights",
    "sample_weights",
    "equal_variances",
    "small_sets",
    "zero_variance",
    "few_genes",
)


@cache
def fixture(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def gene_sets(case: str) -> dict[str, np.ndarray]:
    data = fixture(case)
    return {key[4:]: data[key] for key in data if key.startswith("set ")}


def contrast(case: str) -> Any:
    """The contrast as limma got it: weights, or the position of a column."""
    data = fixture(case)
    if "contrast weights" in data:
        return data["contrast weights"]
    return data["columns"].tolist().index(str(data["contrast"]))


def run(case: str, name: str, **changes: Any) -> Table:
    test, rho, ranks, negative, trend, standardize, robust = RUNS[name]
    data = fixture(case)
    arguments = {
        "data": data["y"],
        "design": data["design"],
        "contrast": contrast(case),
        "gene_sets": gene_sets(case),
        "weights": data.get("weights"),
        "trend": trend,
        "robust": robust,
    }
    if test == "fry":
        return fry(**(arguments | {"standardize": standardize} | changes))
    options = {
        "inter_gene_correlation": rho,
        "allow_negative_correlation": negative,
        "use_ranks": ranks,
    }
    return camera(**(arguments | options | changes))


def compared_runs() -> list[Any]:
    """Every case and run compared with limma."""
    return [
        pytest.param(case, name, id=f"{case}-{name}")
        for case in CASES
        for name in fixture(case)["runs"].tolist()
    ]


def tolerance_of(case: str, name: str) -> float:
    if name.startswith("fry_posterior") and case != "equal_variances":
        return POSTERIOR  # with an infinite prior df both standardize as "none"
    return CLOSE if RUNS[name][-1] else EXACT


def assert_p_values_agree(ours: Any, desired: np.ndarray, tolerance: float) -> None:
    """|ours - desired| <= tolerance * max(p, 1e-300) * max(1, |log p|)."""
    ours = np.asarray(ours, dtype=float)
    assert_array_equal(np.isnan(ours), np.isnan(desired))
    keep = ~np.isnan(desired)
    scale = np.maximum(desired[keep], 1e-300)
    scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    assert np.all(np.abs(ours[keep] - desired[keep]) <= tolerance * scale)


def directions(table: Table) -> np.ndarray:
    return np.array(
        [{"up": 1.0, "down": -1.0, None: np.nan}[v] for v in table["direction"]]
    )


@pytest.mark.parametrize(("case", "name"), compared_runs())
def test_every_column_matches_limma(case: str, name: str) -> None:
    data = fixture(case)
    table = run(case, name)
    tolerance = tolerance_of(case, name)
    d = data["y"].shape[0] - data["design"].shape[1]
    sizes = table["n_genes"]
    # fry's mixed p-values of sets of 2 to d genes differ from limma's on purpose.
    small = (sizes >= 2) & (sizes <= d) if name.startswith("fry") else sizes < 0
    reference = {
        key.split(" ")[2]: value
        for key, value in data.items()
        if key.startswith(f"limma {name} ")
    }
    assert set(reference) >= {"NGenes", "Direction", "PValue"}
    for column, values in reference.items():
        if column == "Direction":
            # limma reports "Up" where the p-value is NaN; CytePool no direction.
            ours = directions(table)
            values = np.where(np.isnan(table["p_value"]), np.nan, values)
            assert_array_equal(ours, values)
        elif column == "NGenes":
            assert_array_equal(table["n_genes"], values)
        elif column == "Correlation":
            assert_allclose(table["correlation"], values, rtol=0, atol=EXACT)
        elif column == "FDR.Mixed" and small.any():
            continue  # depends on the small sets' mixed p-values
        elif column == "PValue.Mixed":
            assert_p_values_agree(
                table[COLUMNS[column]][~small], values[~small], tolerance
            )
        elif column == "PValue":
            # limma's rank test can report more than 1, CytePool 1.
            assert_p_values_agree(table["p_value"], np.minimum(values, 1.0), tolerance)
        else:
            assert_p_values_agree(table[COLUMNS[column]], values, tolerance)


def test_a_design_with_named_contrasts_and_feature_ids_gives_limmas_values() -> None:
    # The weighted case rebuilt from its sample table, with genes named by id and the
    # contrast named in the design; and the unweighted case with a column by name.
    genes = np.array([f"gene{i}" for i in range(200)])
    for case, design, contrast in (
        (
            "groups_weights",
            Design(
                {"group": np.tile(["A", "B", "C"], 3)},
                "~ group",
                contrasts={"c_vs_b": "groupC - groupB"},
            ),
            "c_vs_b",
        ),
        (
            "groups",
            Design({"group": np.repeat(["A", "B", "C"], 3)}, "~ group"),
            "groupB",
        ),
    ):
        data = fixture(case)
        assert_array_equal(design.matrix, data["design"])
        named = {name: genes[rows] for name, rows in gene_sets(case).items()}
        common = {"feature_ids": genes, "weights": data.get("weights")}

        estimated = camera(
            data["y"],
            named,
            design=design,
            contrast=contrast,
            inter_gene_correlation=None,
            **common,
        )
        residual = fry(
            data["y"],
            named,
            design=design,
            contrast=contrast,
            standardize="residual_sd",
            **common,
        )

        assert_p_values_agree(
            estimated["p_value"], data["limma camera_estimated PValue"], EXACT
        )
        assert_allclose(
            estimated["correlation"],
            data["limma camera_estimated Correlation"],
            atol=EXACT,
        )
        assert_p_values_agree(
            residual["p_value"], data["limma fry_residual PValue"], EXACT
        )


# Symmetries and invariances.


def three_groups(seed: int, n_genes: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(9), np.repeat([0, 1, 0], 3), np.repeat([0, 0, 1], 3)])
    y = rng.normal(size=(9, n_genes)) * rng.uniform(0.5, 2.0, n_genes)
    y[3:6, :10] += 1.0
    return y, x


@pytest.mark.parametrize("use_ranks", [False, True])
def test_a_set_and_its_complement_get_the_same_p_value_and_opposite_directions(
    use_ranks: bool,
) -> None:
    # Without a correlation the two-sample and rank sum tests are symmetric in the
    # two groups.
    y, x = three_groups(1)
    inside = np.arange(0, 25)
    sets = {"set": inside, "complement": np.arange(25, 60)}
    table = camera(
        y, sets, design=x, contrast=1, inter_gene_correlation=0.0, use_ranks=use_ranks
    )

    assert_allclose(table["p_value"][0], table["p_value"][1], rtol=1e-12)
    assert set(table["direction"]) == {"up", "down"}


def test_the_order_of_the_genes_does_not_matter() -> None:
    y, x = three_groups(2)
    genes = np.array([f"g{i}" for i in range(60)])
    sets = {"a": genes[:12], "b": genes[[3, 30, 31, 40, 50]], "c": genes[20:45]}
    order = np.random.default_rng(3).permutation(60)

    for test in (camera, fry):
        original = test(y, sets, design=x, contrast=1, feature_ids=genes)
        shuffled = test(
            y[:, order], sets, design=x, contrast=1, feature_ids=genes[order]
        )
        for column in original.columns:
            if original[column].dtype.kind == "f":
                assert_allclose(shuffled[column], original[column], rtol=1e-12)
            else:
                assert_array_equal(shuffled[column], original[column])


def test_negating_the_contrast_flips_directions_and_keeps_p_values() -> None:
    y, x = three_groups(4)
    sets = {"up": np.arange(10), "mixed": np.arange(5, 20), "null": np.arange(30, 50)}
    weights = np.exp(np.random.default_rng(5).normal(0.0, 0.3, y.shape))
    # c and -c have the same Householder reflection when c_0 is not zero (R's sign
    # convention gives a zero c_0 a positive norm), so each gene keeps its residual
    # basis and only the contrast effect changes sign.
    contrast = np.array([0.25, 1.0, -0.5])
    for options in (
        {"inter_gene_correlation": None},
        {"use_ranks": True},
        {"standardize": "residual_sd"},
    ):
        test = fry if "standardize" in options else camera
        plus = test(y, sets, design=x, contrast=contrast, weights=weights, **options)
        minus = test(y, sets, design=x, contrast=-contrast, weights=weights, **options)
        assert_allclose(minus["p_value"], plus["p_value"], rtol=1e-12)
        flipped = {"up": "down", "down": "up"}
        assert [flipped[v] for v in plus["direction"]] == list(minus["direction"])


def test_rescaling_and_shifting_the_values_changes_nothing() -> None:
    y, x = three_groups(6)
    sets = {"a": np.arange(10), "b": np.arange(10, 40)}
    shifted = 10.0 * y + np.arange(60.0)
    for test, options in (
        (camera, {"inter_gene_correlation": None}),
        (camera, {"use_ranks": True, "trend": False}),
        (fry, {}),
    ):
        original = test(y, sets, design=x, contrast=2, **options)
        changed = test(shifted, sets, design=x, contrast=2, **options)
        for column in ("p_value", "adjusted_p_value"):
            assert_allclose(changed[column], original[column], rtol=1e-10)


def test_a_column_named_or_counted_from_either_end_is_the_same_contrast() -> None:
    y, x = three_groups(7)
    design = Design({"group": np.repeat(["A", "B", "C"], 3)}, "~ group")
    sets = {"a": np.arange(12), "b": np.arange(12, 30)}
    expected = camera(y, sets, design=x, contrast=2, inter_gene_correlation=None)
    for given in ("groupC", -1, 2):
        table = camera(
            y, sets, design=design, contrast=given, inter_gene_correlation=None
        )
        assert_allclose(table["p_value"], expected["p_value"], rtol=1e-14)
    # Without observation weights the residual basis does not matter to camera, so
    # the column as weights gives the same test.
    weights = camera(
        y, sets, design=x, contrast=[0.0, 0.0, 1.0], inter_gene_correlation=None
    )
    assert_allclose(weights["p_value"], expected["p_value"], rtol=1e-12)
    assert_allclose(weights["correlation"], expected["correlation"], atol=1e-13)


def test_a_voom_result_is_its_log2_cpm_with_its_weights() -> None:
    rng = np.random.default_rng(8)
    counts = rng.poisson(rng.uniform(20, 500, 80), size=(8, 80))
    x = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])
    result = voom(counts, x)
    sets = {"a": np.arange(10), "b": np.arange(5, 40)}
    for test in (camera, fry):
        direct = test(result, sets, design=x, contrast=1)
        explicit = test(
            result.log2_cpm, sets, design=x, contrast=1, weights=result.weights
        )
        assert_array_equal(direct["p_value"], explicit["p_value"])


# Closed forms and limits.


def test_a_set_of_one_gene_in_fry_is_that_genes_t_test() -> None:
    # The scale of one gene cancels, so every standardization gives the ordinary
    # t-test of the contrast on the residual degrees of freedom, directional and mixed.
    y, x = three_groups(9)
    coefficients, residuals, *_ = np.linalg.lstsq(x, y[:, [17]], rcond=None)
    d = 9 - 3
    unscaled = math.sqrt(np.linalg.inv(x.T @ x)[1, 1])
    t = coefficients[1, 0] / (unscaled * math.sqrt(residuals[0] / d))
    expected = 2.0 * stats.t.sf(abs(t), d)
    for standardize in ("posterior_sd", "residual_sd", "none"):
        table = fry(y, {"one": [17]}, design=x, contrast=1, standardize=standardize)
        assert_allclose(table["p_value"], [expected], rtol=1e-12)
        assert_allclose(table["mixed_p_value"], [expected], rtol=1e-12)
        assert table["direction"][0] == ("up" if t > 0 else "down")


def test_the_estimated_correlation_is_the_mean_pairwise_residual_correlation() -> None:
    # Wu and Smyth (2012): the VIF estimate equals 1 + (m - 1) times the average of
    # the pairwise correlations of the genes' residuals.
    y, x = three_groups(10)
    rows = np.arange(4, 15)
    residuals = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    within = residuals[:, rows]
    d = 9 - 3
    unit = within / np.sqrt(np.sum(within**2, axis=0) / d)
    pairwise = unit.T @ unit / d
    m = rows.size
    expected = (pairwise.sum() - np.trace(pairwise)) / (m * (m - 1))

    table = camera(y, {"set": rows}, design=x, contrast=1, inter_gene_correlation=None)

    assert_allclose(table["correlation"], [expected], rtol=1e-12)


def test_the_null_expectation_of_the_trimmed_mean_square() -> None:
    # E[(sum - max) / d] of d + 1 chi-squared(1): the largest of two has mean
    # 1 + 2 / pi and of three 1 + 2 sqrt(3) / pi.
    assert_allclose(_trimmed_mean_square_expectation(1), 1 - 2 / math.pi, rtol=1e-14)
    assert_allclose(
        _trimmed_mean_square_expectation(2), 1 - math.sqrt(3) / math.pi, rtol=1e-14
    )
    # Larger d, against a simulation: within four standard errors (about 0.005).
    rng = np.random.default_rng(11)
    for d in (5, 20):
        draws = rng.chisquare(1.0, size=(40_000, d + 1))
        trimmed = (draws.sum(axis=1) - draws.max(axis=1)) / d
        error = trimmed.std() / math.sqrt(draws.shape[0])
        assert abs(_trimmed_mean_square_expectation(d) - trimmed.mean()) < 4 * error


def test_posterior_standardization_has_the_limits_of_its_prior() -> None:
    # An infinite prior df gives every gene the prior variance: "none"
    # (equal_variances, matched to limma above). Without moderation, with at most two
    # genes, the prior has no weight: "residual_sd".
    y, x = three_groups(12, n_genes=2)
    sets = {"both": [0, 1]}
    posterior = fry(y, sets, design=x, contrast=1, standardize="posterior_sd")
    residual = fry(y, sets, design=x, contrast=1, standardize="residual_sd")
    assert_allclose(posterior["p_value"], residual["p_value"], rtol=1e-12)
    assert_allclose(posterior["mixed_p_value"], residual["mixed_p_value"], rtol=1e-12)

    data = fixture("equal_variances")
    for name in ("fry_posterior", "fry_none"):
        assert_p_values_agree(
            run("equal_variances", name)["p_value"],
            data["limma fry_none PValue"],
            EXACT,
        )


def test_mixed_p_values_of_small_sets_approximate_the_rotation_test() -> None:
    # Sets of 2 to d genes without a change, against 40,000 random rotations of their
    # effects (standard error at most 0.0025). The beta approximation, in all d + 1
    # dimensions, is itself off by up to about 0.07 at such p-values, as it is for
    # larger sets, where limma's is the same (validation/gene_set_tests.py); limma
    # treats these sets as problems of m dimensions and is off by up to about 0.6.
    data = fixture("small_sets")
    y, x = data["y"], data["design"]
    n, p = x.shape
    q = np.linalg.qr(x, mode="complete")[0]
    effects = (y.T @ q)[:, p - 1 :]
    rotations = np.random.default_rng(13).normal(size=(40_000, n - p + 1))
    rotations /= np.linalg.norm(rotations, axis=1)[:, np.newaxis]
    table = run("small_sets", "fry_none")
    limma = data["limma fry_none PValue.Mixed"]
    worst_limma = 0.0
    for i, (name, rows) in enumerate(gene_sets("small_sets").items()):
        if rows.size > n - p or name.startswith("tail"):
            continue
        block = effects[rows]
        observed = np.sum(block[:, 0] ** 2)
        simulated = np.mean(np.sum((rotations @ block.T) ** 2, axis=1) >= observed)
        assert abs(table["mixed_p_value"][i] - simulated) < 0.1
        worst_limma = max(worst_limma, abs(limma[i] - simulated))
    assert worst_limma > 0.3


# Edge cases.


def test_a_gene_listed_twice_counts_once() -> None:
    y, x = three_groups(14)
    once = {"a": [1, 2, 3, 8], "b": [10, 11]}
    twice = {"a": [3, 1, 2, 3, 8, 1], "b": [11, 10, 10]}
    for test in (camera, fry):
        expected = test(y, once, design=x, contrast=1)
        table = test(y, twice, design=x, contrast=1)
        assert_array_equal(table["n_genes"], [4, 2])
        assert_allclose(table["p_value"], expected["p_value"], rtol=1e-14)


def test_empty_sets_and_sets_of_every_gene_have_no_p_value() -> None:
    y, x = three_groups(15)
    genes = np.array([f"g{i}" for i in range(60)])
    sets = {"empty": ["not a gene"], "a": genes[:10], "all": genes, "b": genes[20:30]}
    table = camera(y, sets, design=x, contrast=1, feature_ids=genes)
    assert_array_equal(table["n_genes"], [0, 10, 60, 10])
    assert np.isnan(table["p_value"][[0, 2]]).all()
    assert table["direction"][0] is None and table["direction"][2] is None
    # The other sets are adjusted among themselves.
    alone = camera(
        y, {"a": genes[:10], "b": genes[20:30]}, design=x, contrast=1, feature_ids=genes
    )
    assert_allclose(table["adjusted_p_value"][[1, 3]], alone["adjusted_p_value"])

    tested = fry(y, sets, design=x, contrast=1, feature_ids=genes)
    assert np.isnan(tested["p_value"][0]) and np.isnan(tested["mixed_p_value"][0])
    assert np.isfinite(tested["p_value"][2])  # fry's test is self-contained


def test_the_rank_test_p_value_is_at_most_one() -> None:
    # A set of the lowest and the highest z-score: its rank sum is centred, and with
    # the continuity correction both tails exceed 1/2. limma reports twice the smaller.
    y, x = three_groups(16)
    t = camera(
        y,
        {f"g{i}": [i] for i in range(60)},
        design=x,
        contrast=1,
        inter_gene_correlation=0.0,
    )
    signed = np.where(np.array(t["direction"]) == "up", 1.0, -1.0) * (1 - t["p_value"])
    ends = [int(np.argmin(signed)), int(np.argmax(signed))]
    table = camera(
        y,
        {"ends": ends},
        design=x,
        contrast=1,
        inter_gene_correlation=0.0,
        use_ranks=True,
    )
    assert table["p_value"][0] == 1.0
    assert table["direction"][0] == "up"


def test_camera_estimates_correlation_with_a_floor_on_tiny_variances() -> None:
    # A gene the design fits exactly has residual effects of rounding size; with its
    # variance bounded below by 1e-8 they add nothing to the set's mean (limma's rule).
    y, x = three_groups(17)
    y[:, 5] = x @ np.array([3.0, 1.0, -1.0])
    sets = {"with": [5, *range(20, 30)], "without": list(range(20, 30))}
    table = camera(y, sets, design=x, contrast=1, inter_gene_correlation=None)
    m = 11
    vif_without = 1 + 9 * table["correlation"][1]
    expected = ((10 / m) ** 2 * m / 10 * vif_without - 1) / (m - 1)
    assert_allclose(table["correlation"][0], expected, rtol=1e-10)


def test_a_pivot_that_is_zero_up_to_rounding_counts_as_zero() -> None:
    # Samples sorted by group and C - B as weights: a pivot of each gene's QR
    # decomposition is zero in exact arithmetic. Its sign is fixed, so weights a
    # rounding error apart give the same residual bases and the same results; were
    # rounding to decide it, a few genes would change basis (2 of these 200), which
    # moves p-values by more than 1e-3.
    data = fixture("sorted_groups_weights")
    sets = gene_sets("sorted_groups_weights")
    jitter = 1.0 + 1e-13 * np.random.default_rng(21).normal(size=data["weights"].shape)
    for test, options in (
        (camera, {"inter_gene_correlation": None}),
        (fry, {"standardize": "residual_sd"}),
    ):
        given = {"design": data["design"], "contrast": data["contrast weights"]}
        exact = test(data["y"], sets, **given, weights=data["weights"], **options)
        jittered = test(
            data["y"], sets, **given, weights=data["weights"] * jitter, **options
        )
        assert_allclose(jittered["p_value"], exact["p_value"], rtol=1e-9)


def test_genes_with_equal_residuals_have_a_correlation_of_one() -> None:
    # Identical genes: their standardized residuals are equal, so the VIF is m and
    # the correlation 1, which rounding must not push above 1 (the rank test's
    # variance takes asin((rho + 1) / 2)).
    y, x = three_groups(22)
    y[:, 10:20] = y[:, [10]]
    sets = {"identical": np.arange(10, 20), "other": np.arange(30, 45)}
    for use_ranks in (False, True):
        table = camera(
            y,
            sets,
            design=x,
            contrast=1,
            inter_gene_correlation=None,
            use_ranks=use_ranks,
        )
        assert table["correlation"][0] == 1.0
        assert np.isfinite(table["p_value"]).all()


def test_a_contrast_can_name_its_columns_or_be_a_zero_dimensional_position() -> None:
    y, x = three_groups(23)
    design = Design({"group": np.repeat(["A", "B", "C"], 3)}, "~ group")
    sets = {"a": np.arange(12), "b": np.arange(12, 30)}
    vector = camera(
        y, sets, design=design, contrast=[0.0, -1.0, 1.0], inter_gene_correlation=None
    )
    mapping = camera(
        y,
        sets,
        design=design,
        contrast={"groupC": 1.0, "groupB": -1.0},
        inter_gene_correlation=None,
    )
    assert_array_equal(mapping["p_value"], vector["p_value"])
    column = camera(y, sets, design=design, contrast="groupC")
    assert_array_equal(
        camera(y, sets, design=design, contrast=np.array(2))["p_value"],
        column["p_value"],
    )


# Arguments.


def test_invalid_arguments_are_refused_with_a_reason() -> None:
    y, x = three_groups(18)
    sets = {"a": [0, 1, 2]}
    genes = [f"g{i}" for i in range(60)]
    voomed = voom(np.random.default_rng(1).poisson(50, size=(9, 60)), x)
    nan = y.copy()
    nan[0, 0] = np.nan
    cases = [
        ({"data": nan}, ValueError, "data must be finite"),
        ({"data": y[:, :2]}, ValueError, "at least 3 genes"),
        ({"weights": np.zeros(9)}, ValueError, "positive and finite"),
        ({"weights": np.ones(5)}, ValueError, r"shape \(9,\) for sample weights"),
        ({"data": voomed, "weights": np.ones(9)}, ValueError, "carries the precision"),
        ({"design": x[:, [0, 1, 1]]}, ValueError, "not of full column rank"),
        ({"design": np.eye(9)}, ValueError, "no residual degrees of freedom"),
        ({"design": x[:8]}, ValueError, "data has 9 rows but the design has 8"),
        ({"contrast": "groupD"}, ValueError, "neither a named contrast"),
        ({"contrast": 3}, ValueError, "positions 0 to 2"),
        ({"contrast": [0.0, 0.0, 0.0]}, ValueError, "zero on every column"),
        ({"contrast": [0.0, 1.0]}, ValueError, "one weight per design column"),
        ({"contrast": np.eye(3)}, ValueError, "one contrast is tested at a time"),
        ({"contrast": 1.5}, TypeError, "contrast must be a contrast name"),
        ({"gene_sets": [0, 1]}, TypeError, "gene_sets must be a mapping"),
        ({"gene_sets": {}}, ValueError, "gene_sets is empty"),
        ({"gene_sets": {"a": "g1"}}, TypeError, "single string"),
        ({"gene_sets": {"a": [0, 60]}}, ValueError, "position 60"),
        ({"gene_sets": {"a": [0.0, 1.0]}}, TypeError, "0-based gene positions"),
        ({"gene_sets": {"a": [True, False]}}, ValueError, "one value per gene"),
        ({"gene_sets": {1: [0, 1]}}, TypeError, "names must be strings"),
        ({"gene_sets": {"": [0, 1]}}, ValueError, "must not be empty"),
        ({"contrast": {"x4": 1.0}}, ValueError, "not a design column"),
        ({"contrast": {"x2": "1"}}, TypeError, "one number per design column"),
        (
            {"feature_ids": genes, "gene_sets": {"a": ["x"]}},
            ValueError,
            "no gene of any",
        ),
        ({"feature_ids": ["g1"] * 60}, ValueError, "feature_ids must be unique"),
        ({"feature_ids": genes[:3]}, ValueError, "one id per gene"),
        ({"inter_gene_correlation": 1.0}, ValueError, r"lie in \(-1, 1\)"),
        ({"inter_gene_correlation": True}, TypeError, "real number or None"),
        ({"use_ranks": 1}, TypeError, "use_ranks must be a bool"),
    ]
    for changes, error, message in cases:
        arguments = {"data": y, "design": x, "contrast": 1, "gene_sets": sets} | changes
        with pytest.raises(error, match=message):
            camera(**arguments)


def test_fry_refuses_options_without_effect_and_unknown_standardizations() -> None:
    y, x = three_groups(19)
    sets = {"a": [0, 1, 2]}
    with pytest.raises(ValueError, match="only with standardize='posterior_sd'"):
        fry(y, sets, design=x, contrast=1, standardize="none", trend=True)
    with pytest.raises(ValueError, match="standardize must be one of"):
        fry(y, sets, design=x, contrast=1, standardize="posterior.sd")
    with pytest.raises(TypeError, match="robust must be a bool"):
        fry(y, sets, design=x, contrast=1, robust="yes")


def test_a_design_with_random_effects_is_refused() -> None:
    y, _ = three_groups(20)
    design = Design(
        {
            "group": np.repeat(["A", "B", "C"], 3),
            "donor": np.tile(["D1", "D2", "D3"], 3),
        },
        "~ group",
        random="(1 | donor)",
    )
    for test in (camera, fry):
        with pytest.raises(ValueError, match="fits fixed effects only"):
            test(y, {"a": [0, 1]}, design=design, contrast="groupB")


# Results of differential expression.

CONTRASTS = {"stim_vs_ctrl": "conditionstim", "D2_vs_D1": "donorD2"}
LEVELS = {"condition": ["ctrl", "stim"]}
# Sets of the simulated genes, named by symbol: up, down, both and unchanged.
SYMBOL_SETS = {
    "up": [f"G{j}" for j in range(15)],
    "down": [f"G{j}" for j in range(15, 25)],
    "both": [f"G{j}" for j in range(5, 20)],
    "null": [f"G{j}" for j in range(40, 70)],
}


@cache
def two_groups_of_six_donors() -> Cube:
    """Negative binomial counts of 120 genes, with ids and symbols, in two groups of
    6 donors x 2 conditions; genes G0 to G14 go up and G15 to G24 down."""
    rng = np.random.default_rng(31)
    condition = np.tile(["ctrl", "stim"], 6)
    change = np.zeros(120)
    change[:15], change[15:25] = 1.0, -1.0
    mean = np.exp(rng.normal(4.0, 1.0, size=(2, 1, 120)))
    mean = mean * np.where(condition[None, :, None] == "stim", 2.0**change, 1.0)
    return Cube(
        {"counts": rng.negative_binomial(10, 10 / (10 + mean))},
        samples={
            "sample_id": [f"S{i}" for i in range(12)],
            "condition": condition,
            "donor": np.repeat([f"D{i}" for i in range(1, 7)], 2),
        },
        groups={"group_id": ["B", "T"]},
        features={
            "feature_id": [f"ENSG{j:05d}" for j in range(120)],
            "symbol": [f"G{j}" for j in range(120)],
        },
        n_cells=rng.integers(20, 60, size=(12, 2)),
    )


@cache
def results() -> dict[str, DifferentialExpressionResult]:
    """The cube's limma-voom, with robust moderation; its limma-trend, with the cell
    numbers as prior weights; and its dream-voom of group B."""
    cube = two_groups_of_six_donors()
    paired = Design(
        cube.samples, "~ donor + condition", levels=LEVELS, contrasts=CONTRASTS
    )
    mixed = Design(
        cube.samples,
        "~ condition",
        random="(1 | donor)",
        levels=LEVELS,
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    return {
        "limma_voom": limma_voom(cube, paired, robust=True),
        "limma_trend": limma_trend(cube, paired, prior_weights="n_cells"),
        "dream_voom": dream_voom(cube, mixed, groups=["B"]),
    }


def fitted(
    result: DifferentialExpressionResult, group: str, contrast: str
) -> tuple[Any, dict[str, Any]]:
    """A group's fit, and what a gene-set test of the result reads from it: the
    design rows and contrast weights fitted and the symbols of the features tested."""
    fit = result.fit(group)
    linear = fit.moderated.fit
    column = linear.coefficient_names.index(contrast)
    return fit, {
        "design": linear.design,
        "contrast": linear.contrasts[:, column],
        "feature_ids": fit.features["symbol"],
    }


@pytest.mark.parametrize("contrast", list(CONTRASTS))
@pytest.mark.parametrize("test", [camera, fry])
def test_a_limma_voom_result_is_tested_as_the_voom_values_of_its_groups_fit(
    test: Any, contrast: str
) -> None:
    # Group B's filter leaves a gene out, so its ids are those of its fit.
    result = results()["limma_voom"]
    fit, read = fitted(result, "B", contrast)
    assert len(fit.features) < result.cube.n_features and fit.voom is not None
    assert result.settings["robust"]
    table = test(result, SYMBOL_SETS, group="B", contrast=contrast, id_column="symbol")
    expected = test(fit.voom, SYMBOL_SETS, **read, trend=False, robust=True)
    assert table.equals(expected)
    # The sets named by feature id, the default column, are the same sets.
    symbols = result.cube.features["symbol"].tolist()
    ids = result.cube.features["feature_id"]
    by_id = {
        name: [ids[symbols.index(gene)] for gene in genes]
        for name, genes in SYMBOL_SETS.items()
    }
    assert test(result, by_id, group="B", contrast=contrast).equals(table)


def test_a_result_gives_the_design_weights_and_ids_of_its_fit() -> None:
    result = results()["limma_voom"]
    fit, read = fitted(result, "B", "stim_vs_ctrl")
    tested = {"group": "B", "contrast": "stim_vs_ctrl", "id_column": "symbol"}
    for test in (camera, fry):
        for name, value, owner in (
            ("design", read["design"], "the design of its group's fit"),
            ("feature_ids", read["feature_ids"], "the ids of the column id_column"),
            ("weights", fit.weights, "the weights of its group's fit"),
        ):
            message = (
                f"{name} applies to an array or a VoomResult; a "
                f"differential-expression result gives {owner}"
            )
            with pytest.raises(ValueError, match=message):
                test(result, SYMBOL_SETS, **tested, **{name: value})
        # A group names a group of a result, not of an array.
        with pytest.raises(
            ValueError,
            match="group names a group of a differential-expression result, and data "
            "is a ndarray",
        ):
            test(fit.log2_cpm, SYMBOL_SETS, group="B", **read)


def test_a_result_of_dream_voom_is_refused() -> None:
    result = results()["dream_voom"]
    for test in (camera, fry):
        with pytest.raises(
            ValueError,
            match=f"{test.__name__} tests the contrast and residual effects of a "
            "linear model, and this result is from dream_voom",
        ):
            test(result, SYMBOL_SETS, contrast="stim_vs_ctrl", id_column="symbol")
