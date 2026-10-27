"""Tests for voom precision weights.

Reference values in ``tests/data/voom`` come from limma's ``voom`` (after edgeR's
``calcNormFactors`` in the cases with TMM factors) and, downstream, from ``lmFit``,
``contrasts.fit`` and ``eBayes``, and from limma's ``chooseLowessSpan``;
``validation/voom.py`` writes them. The cases named ``adaptive_*`` choose the span
from the number of genes (limma's ``adaptive.span = TRUE``); the others fix it.
Designs are built from each case's sample table, by R's ``model.matrix`` for the
reference and by ``Design`` here. The other tests check properties that a wrong
implementation would break: invariances, closed forms and the documented edge cases.
"""

import contextlib
import dataclasses
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool import Design
from cytepool.normalization import tmm_factors
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.smoothing import choose_lowess_span
from cytepool.stats.voom import VoomResult, voom

DATA = Path(__file__).parent / "data" / "voom"
CASES = (
    "two_groups",
    "span_0.3",
    "span_1",
    "library_sizes_given",
    "two_groups_tmm",
    "library_sizes_and_tmm",
    "groups_covariate_tmm",
    "group_means_tmm",
    "covariate_only",
    "one_residual_df",
    "small_libraries",
    "extreme_libraries",
    "intercept_only",
    "ties",
    # Blocks of genes with one read: ties wider than the LOWESS window, which give
    # limma's trend only when x is computed to the last bit as limma computes it.
    "tie_blocks",
    "rarefied",
    "non_integer",
    "no_replication",
    "two_genes",
    # Spans chosen from the number of genes: 1 up to 50 genes. About a fifth of the
    # 30, 51 and 120 genes have no counts, and count all the same.
    "adaptive_two_genes",
    "adaptive_30_genes",
    "adaptive_51_genes",
    "adaptive_120_genes",
    "adaptive_group_means_tmm",
    "adaptive_groups_covariate_tmm",
    "adaptive_no_replication",
)
WITH_TREND = tuple(case for case in CASES if not case.endswith("no_replication"))
ADAPTIVE = tuple(case for case in CASES if case.startswith("adaptive_"))
TWO_GROUPS = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])


@cache
def fixture(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def tolerance(case: str) -> float:
    """The validation's tier: exact (1e-10), or close (1e-6) for rarefied counts,
    where genes with the same count in every sample have a residual standard
    deviation that is rounding noise in either program."""
    return float(fixture(case)["tolerance"])


def design_of(case: str) -> Design:
    data = fixture(case)
    samples = {
        key.removeprefix("samples "): data[key]
        for key in data
        if key.startswith("samples ")
    }
    contrasts = None
    if "contrast names" in data:
        names = data["contrast names"].tolist()
        contrasts = dict(zip(names, data["contrast expressions"].tolist()))
    return Design(samples, str(data["formula"]), contrasts=contrasts)


def run_voom(case: str) -> VoomResult:
    """voom on a case's counts, from the counts alone (TMM factors included)."""
    data = fixture(case)
    library_sizes = data.get("lib")
    norm_factors = None
    if data["tmm"]:
        norm_factors = tmm_factors(data["counts"], library_sizes=library_sizes)
    expected = (
        pytest.warns(RuntimeWarning, match="no residual degrees of freedom")
        if case.endswith("no_replication")
        else contextlib.nullcontext()
    )
    # Cases without a span choose it from the number of genes, as limma did.
    span = float(data["span"]) if "span" in data else None
    with expected:
        return voom(
            data["counts"],
            design_of(case),
            library_sizes=library_sizes,
            norm_factors=norm_factors,
            span=span,
        )


def limma_matrix(case: str, key: str, n_columns: int) -> np.ndarray:
    """A limma matrix written column by column, as rows of n_columns columns."""
    return fixture(case)[f"limma {key}"].reshape(n_columns, -1).T


def assert_p_values_agree(ours: Any, desired: np.ndarray, tolerance: float) -> None:
    """|ours - desired| <= tolerance * max(p, 1e-300) * max(1, |log p|).

    About the error in log p relative to |log p|: rounding in the variances moves a
    far-tail p-value in proportion to |log p|.
    """
    scale = np.maximum(desired, 1e-300)
    scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    assert np.all(np.abs(np.asarray(ours) - desired) <= tolerance * scale)


@pytest.mark.parametrize("case", CASES)
def test_log2_cpm_weights_and_library_sizes_match_limma(case: str) -> None:
    data = fixture(case)
    result = run_voom(case)
    n_samples = data["counts"].shape[0]
    exact = 1e-10
    # log2 CPM is a closed form: limma's to the last bit on the validation inputs.
    np.testing.assert_allclose(
        result.log2_cpm, limma_matrix(case, "E", n_samples).T, rtol=exact, atol=exact
    )
    np.testing.assert_allclose(
        result.weights,
        limma_matrix(case, "weights", n_samples).T,
        rtol=tolerance(case),
    )
    sizes = result.effective_library_sizes
    np.testing.assert_allclose(sizes, data["limma lib.size"], rtol=exact)
    if data["tmm"]:
        factors = sizes / data.get("lib", data["counts"].sum(axis=1))
        np.testing.assert_allclose(factors, data["limma norm.factors"], rtol=exact)


@pytest.mark.parametrize("case", WITH_TREND)
def test_trend_matches_limma(case: str) -> None:
    data = fixture(case)
    result = run_voom(case)
    in_trend = ~np.isnan(result.sqrt_sd)
    # limma leaves out the features without counts (found by probing).
    np.testing.assert_array_equal(in_trend, data["counts"].any(axis=0))
    np.testing.assert_array_equal(np.isnan(result.mean_log_count), ~in_trend)
    np.testing.assert_array_equal(np.isnan(result.sqrt_sd_trend), ~in_trend)
    x = result.mean_log_count[in_trend]
    np.testing.assert_allclose(x, data["limma xy.x"], rtol=1e-10, atol=1e-10)
    # Square-root standard deviations near zero are rounding noise in both, so they
    # are compared on the scale of the typical one.
    scale = float(np.median(data["limma xy.y"]))
    np.testing.assert_allclose(
        result.sqrt_sd[in_trend],
        data["limma xy.y"],
        rtol=tolerance(case),
        atol=tolerance(case) * scale,
    )
    order = np.argsort(x, kind="stable")
    np.testing.assert_allclose(
        result.sqrt_sd_trend[in_trend][order],
        data["limma line.y"],
        rtol=tolerance(case),
        atol=tolerance(case) * scale,
    )


@pytest.mark.parametrize("case", WITH_TREND)
def test_moderated_statistics_of_the_weighted_fit_match_limma(case: str) -> None:
    """counts -> voom -> fit_linear_model -> contrasts_fit -> moderated_statistics."""
    design = design_of(case)
    result = run_voom(case)
    fit = fit_linear_model(result.log2_cpm, design, weights=result.weights)
    if len(design.contrasts):
        # limma's contrasts.fit approximates the standard errors of contrasts of
        # correlated coefficients in a weighted fit; method="limma" reproduces it.
        fit = contrasts_fit(fit, design.contrasts, method="limma")
    moderated = moderated_statistics(fit)
    n_columns = moderated.t.shape[1]
    rtol = tolerance(case)
    np.testing.assert_allclose(
        moderated.t, limma_matrix(case, "t", n_columns), rtol=rtol, atol=rtol
    )
    assert_p_values_agree(
        moderated.p_value, limma_matrix(case, "p.value", n_columns), rtol
    )


@pytest.mark.parametrize("case", ADAPTIVE)
def test_chosen_span_matches_limma(case: str) -> None:
    """limma returns the span it chose, and none when no trend is fitted."""
    data = fixture(case)
    result = run_voom(case)
    if "limma span" in data:
        # A closed form: limma's to the last bit on the validation inputs.
        np.testing.assert_allclose(result.span, data["limma span"][0], rtol=1e-10)
    else:
        assert case.endswith("no_replication") and np.isnan(result.span)


def test_span_rule_matches_limmas_choose_lowess_span() -> None:
    """choose_lowess_span against chooseLowessSpan over 0 to 1e9 points with limma's
    settings, and over other settings of small.n, min.span and power."""
    data = fixture("span_rule")
    spans = [
        choose_lowess_span(int(n), small_n=int(small_n), min_span=lower, power=rate)
        for n, small_n, lower, rate in data["settings"]
    ]
    # A closed form: limma's to the last bit on the validation inputs.
    np.testing.assert_allclose(spans, data["limma"], rtol=1e-10)
    assert data["settings"][:, 0].max() == 1e9 and len(spans) > 300


def test_default_span_is_chosen_from_every_feature() -> None:
    """The default is the span of the rule for the number of features, those
    without counts included, as in limma: 51 features give a span below 1,
    although at most 50 of them enter the trend, which alone would give 1."""
    data = fixture("adaptive_51_genes")
    counts, design = data["counts"], design_of("adaptive_51_genes")
    in_trend = int(counts.any(axis=0).sum())
    assert counts.shape[1] == 51 and in_trend <= 50
    span = choose_lowess_span(51)
    assert span < 1.0 and choose_lowess_span(in_trend) == 1.0

    chosen = voom(counts, design)
    given = voom(counts, design, span=span)
    assert chosen.span == span
    np.testing.assert_array_equal(chosen.weights, given.weights)
    np.testing.assert_array_equal(chosen.sqrt_sd_trend, given.sqrt_sd_trend)
    # The span matters: the weights of a span of 1 are not those.
    assert not np.allclose(voom(counts, design, span=1.0).weights, chosen.weights)


def test_counts_to_moderated_statistics_with_a_design_and_named_contrasts() -> None:
    """The whole workflow on public objects, with the defaults: the span chosen
    from the number of genes, and exact contrasts.

    The coefficients of a group-means design are uncorrelated in every feature's
    weighted fit, so the exact standard errors of their contrasts are also limma's.
    """
    case = "adaptive_group_means_tmm"
    data = fixture(case)
    counts = data["counts"]
    design = Design(
        {"group": data["samples group"]},
        "~ 0 + group",
        contrasts={"BvsA": "groupB - groupA", "CvsB": "groupC - groupB"},
    )
    result = voom(counts, design, norm_factors=tmm_factors(counts))
    fit = fit_linear_model(result.log2_cpm, design, weights=result.weights)
    contrasted = contrasts_fit(fit, design.contrasts)
    moderated = moderated_statistics(contrasted)
    assert contrasted.coefficient_names == ("BvsA", "CvsB")
    assert contrasted.exact_standard_errors
    # A closed form: limma's to the last bit on the validation inputs.
    np.testing.assert_allclose(result.span, data["limma span"][0], rtol=1e-10)
    np.testing.assert_allclose(
        moderated.t, limma_matrix(case, "t", 2), rtol=1e-10, atol=1e-10
    )
    assert_p_values_agree(moderated.p_value, limma_matrix(case, "p.value", 2), 1e-10)


def test_no_replication_gives_unit_weights_and_no_trend() -> None:
    result = run_voom("no_replication")
    np.testing.assert_array_equal(result.weights, 1.0)
    assert np.isnan(result.sqrt_sd).all() and np.isnan(result.sqrt_sd_trend).all()
    assert not np.isnan(result.mean_log_count).all()
    # No trend, so no span, although one was given.
    assert float(fixture("no_replication")["span"]) == 0.5 and np.isnan(result.span)


def test_scaling_library_sizes_plus_one_shifts_log2_cpm_and_keeps_weights() -> None:
    """With R + 1 scaled by c, log2 CPM falls by log2(c) and nothing else changes.

    log2((r + 0.5) / (c (R + 1)) 1e6) = y - log2(c); with an intercept the fitted
    values fall by log2(c) too, the residuals stay, and the average and fitted log2
    counts, which add log2(c (R + 1)) back, are unchanged. The libraries of 5e3 and
    2e4 reads make a misplaced + 1 visible at about 1e-4.
    """
    counts = fixture("small_libraries")["counts"]
    design = design_of("small_libraries")
    sizes = counts.sum(axis=1)
    base = voom(counts, design, library_sizes=sizes)
    scaled = voom(counts, design, library_sizes=4.0 * (sizes + 1.0) - 1.0)
    np.testing.assert_allclose(scaled.log2_cpm, base.log2_cpm - 2.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaled.weights, base.weights, rtol=1e-12)
    for name in ("mean_log_count", "sqrt_sd", "sqrt_sd_trend"):
        np.testing.assert_allclose(
            getattr(scaled, name), getattr(base, name), rtol=1e-12, atol=1e-12
        )


def test_weights_depend_on_a_feature_only_through_its_fitted_values() -> None:
    """Two features with the same group means of log2 CPM but different residuals.

    Their fitted values are equal, so their weights must be, while the residual
    standard deviation of the second is three times that of the first.
    """
    sizes = np.array([1e6, 2e6, 1.5e6, 1e6, 3e6, 2e6])
    means = np.repeat([5.0, 7.0], 3)
    spread = np.array([-0.2, 0.0, 0.2, -0.1, 0.0, 0.1])
    log_cpm = np.column_stack([means + spread, means + 3.0 * spread])
    pair = 2.0**log_cpm * (sizes[:, np.newaxis] + 1.0) / 1e6 - 0.5
    counts = np.column_stack([fixture("two_groups")["counts"], pair])
    result = voom(counts, TWO_GROUPS, library_sizes=sizes)
    np.testing.assert_allclose(result.log2_cpm[:, -2:], log_cpm, rtol=1e-13)
    np.testing.assert_allclose(result.weights[:, -1], result.weights[:, -2], rtol=1e-12)
    np.testing.assert_allclose(
        result.sqrt_sd[-1] / result.sqrt_sd[-2], np.sqrt(3.0), rtol=1e-12
    )


def test_identical_features_are_weighted_by_their_inverse_variance() -> None:
    """Two features with the same counts are one point of the trend: the curve is
    constant at the square root of their residual standard deviation s, so every
    weight is 1 / s^2. limma fails here (a deliberate deviation)."""
    column = np.array([10.0, 25.0, 14.0, 40.0, 22.0, 35.0])
    counts = np.column_stack([column, column])
    sizes = np.array([1e3, 1.2e3, 0.9e3, 1.5e3, 1.1e3, 1.3e3])
    result = voom(counts, TWO_GROUPS, library_sizes=sizes)
    _, residual_sum, *_ = np.linalg.lstsq(TWO_GROUPS, result.log2_cpm[:, 0])
    np.testing.assert_allclose(result.weights, 4.0 / residual_sum[0], rtol=1e-12)


def test_reparametrised_design_gives_the_same_trend_and_weights() -> None:
    counts = fixture("two_groups")["counts"]
    samples = {"group": np.repeat(["A", "B"], 3)}
    treatment = voom(counts, Design(samples, "~ group"))
    means = voom(counts, Design(samples, "~ 0 + group"))
    np.testing.assert_allclose(means.weights, treatment.weights, rtol=1e-12)
    for name in ("sqrt_sd", "sqrt_sd_trend"):
        np.testing.assert_allclose(
            getattr(means, name), getattr(treatment, name), rtol=1e-12
        )


def test_features_without_counts_leave_the_others_unchanged() -> None:
    """Features without counts are left out of the trend. With equal library sizes
    their fitted log2 counts are log2(0.5), below every feature with counts, so
    their weights come from the lower end of the curve. The span is fixed: the
    default would count the features without counts too."""
    counts = fixture("two_groups")["counts"]
    counts = counts[:, counts.any(axis=0)]
    sizes = np.full(6, 2e6)
    base = voom(counts, TWO_GROUPS, library_sizes=sizes, span=0.5)
    zero = np.array([0, 5, counts.shape[1] + 2])
    padded = np.insert(counts, zero - np.arange(3), 0.0, axis=1)
    result = voom(padded, TWO_GROUPS, library_sizes=sizes, span=0.5)
    others = np.setdiff1d(np.arange(padded.shape[1]), zero)
    np.testing.assert_allclose(result.weights[:, others], base.weights, rtol=1e-12)
    np.testing.assert_allclose(result.sqrt_sd[others], base.sqrt_sd, rtol=1e-12)
    assert np.isnan(result.mean_log_count[zero]).all()
    lowest = base.sqrt_sd_trend[np.argmin(base.mean_log_count)]
    np.testing.assert_allclose(result.weights[:, zero], lowest**-4.0, rtol=1e-12)


def test_permuting_samples_and_features_permutes_the_results() -> None:
    data = fixture("groups_covariate_tmm")
    counts = data["counts"]
    samples = {"group": data["samples group"], "age": data["samples age"]}
    rows = np.random.default_rng(5).permutation(counts.shape[0])
    columns = np.random.default_rng(6).permutation(counts.shape[1])
    base = voom(counts, Design(samples, "~ group + age"))
    shuffled_samples = {name: values[rows] for name, values in samples.items()}
    shuffled = voom(counts[rows][:, columns], Design(shuffled_samples, "~ group + age"))
    np.testing.assert_allclose(
        shuffled.weights, base.weights[rows][:, columns], rtol=1e-12
    )
    np.testing.assert_allclose(
        shuffled.log2_cpm, base.log2_cpm[rows][:, columns], rtol=1e-14
    )
    np.testing.assert_allclose(
        shuffled.sqrt_sd_trend, base.sqrt_sd_trend[columns], rtol=1e-12
    )


def test_result_does_not_change_with_the_callers_library_sizes() -> None:
    counts = fixture("two_groups")["counts"]
    sizes = counts.sum(axis=1)
    result = voom(counts, TWO_GROUPS, library_sizes=sizes)
    sizes[:] = 1.0
    np.testing.assert_array_equal(result.effective_library_sizes, counts.sum(axis=1))


def test_result_is_read_only_with_a_summary() -> None:
    result = run_voom("two_genes")
    with pytest.raises(ValueError, match="read-only"):
        result.weights[0, 0] = 1.0
    assert repr(result) == (
        "VoomResult(n_samples=6, n_features=2, features_in_trend=2, span=0.5)"
    )
    with pytest.raises(ValueError, match="effective_library_sizes must have shape"):
        VoomResult(
            result.log2_cpm,
            result.weights,
            result.effective_library_sizes[:-1],
            result.mean_log_count,
            result.sqrt_sd,
            result.sqrt_sd_trend,
            result.span,
        )


@pytest.mark.parametrize(
    ("span", "error", "message"),
    [
        (0.0, ValueError, r"span must be in \(0, 1\], or NaN when no trend"),
        (1.5, ValueError, r"span must be in \(0, 1\], or NaN when no trend"),
        ("0.5", TypeError, "span must be a real number, got str"),
        (None, TypeError, "span must be a real number, got NoneType"),
    ],
)
def test_result_refuses_an_invalid_span(span: Any, error: type, message: str) -> None:
    result = run_voom("two_genes")
    with pytest.raises(error, match=message):
        dataclasses.replace(result, span=span)
    # NaN marks a result without a trend.
    assert np.isnan(dataclasses.replace(result, span=np.nan).span)


# Counts of 8 genes in 4 samples whose LOWESS trend is negative at its upper end;
# limma returns weights from it all the same.
NEGATIVE_TREND = np.array(
    [
        [2, 101, 23, 131, 509, 87, 230, 150],
        [0, 253, 77, 60, 1306, 46, 51, 240],
        [1, 156, 169, 74, 3276, 19, 106, 265],
        [1, 17, 190, 30, 3434, 124, 162, 573],
    ],
    dtype=float,
)
COUNTS = fixture("two_groups")["counts"]
FEATURE = COUNTS[:, [int(COUNTS.sum(axis=0).argmax())]]
SATURATED = np.eye(6)


@pytest.mark.parametrize(
    ("counts", "design", "keywords", "error", "message"),
    [
        (-COUNTS, TWO_GROUPS, {}, ValueError, "finite and non-negative"),
        (COUNTS * np.nan, TWO_GROUPS, {}, ValueError, "finite and non-negative"),
        (COUNTS[0], TWO_GROUPS, {}, ValueError, "2-D array"),
        (COUNTS.astype(str), TWO_GROUPS, {}, TypeError, "integer or float"),
        (COUNTS, TWO_GROUPS[:5], {}, ValueError, "counts have 6 rows"),
        (COUNTS, TWO_GROUPS[:, [0, 1, 1]], {}, ValueError, "full column rank"),
        (COUNTS, TWO_GROUPS * np.nan, {}, ValueError, "design must be finite"),
        (COUNTS, TWO_GROUPS, {"library_sizes": [1.0] * 5}, ValueError, "per sample"),
        (COUNTS, TWO_GROUPS, {"library_sizes": [0.0] * 6}, ValueError, "positive"),
        (COUNTS, TWO_GROUPS, {"norm_factors": [-1.0] * 6}, ValueError, "positive"),
        (
            np.vstack([COUNTS[:5], np.zeros(COUNTS.shape[1])]),
            TWO_GROUPS,
            {},
            ValueError,
            "library size is zero",
        ),
        (COUNTS, TWO_GROUPS, {"span": 0.0}, ValueError, r"in \(0, 1\]"),
        (COUNTS, TWO_GROUPS, {"span": 1.5}, ValueError, r"in \(0, 1\]"),
        (COUNTS, TWO_GROUPS, {"span": np.nan}, ValueError, r"in \(0, 1\]"),
        (COUNTS, TWO_GROUPS, {"span": "0.5"}, TypeError, "real number, or None"),
        (COUNTS, TWO_GROUPS, {"span": True}, TypeError, "real number, or None"),
        (FEATURE, TWO_GROUPS, {}, ValueError, "at least two features, got 1"),
        # One feature is an error without replication too, as in limma.
        (FEATURE, SATURATED, {}, ValueError, "at least two features, got 1"),
        (
            np.column_stack([FEATURE, np.zeros((6, 3))]),
            TWO_GROUPS,
            {"library_sizes": [1e3] * 6},
            ValueError,
            "at least two features with counts, got 1",
        ),
        (
            NEGATIVE_TREND,
            TWO_GROUPS[[0, 1, 3, 4]],
            {"span": 0.5},
            ValueError,
            "trend falls to -0.442",
        ),
    ],
)
def test_invalid_input_is_rejected(
    counts: Any, design: Any, keywords: dict[str, Any], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        voom(counts, design, **keywords)
