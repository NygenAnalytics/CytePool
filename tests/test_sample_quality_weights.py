"""Tests for sample quality weights, voom with prior weights and voom with quality weights.

Reference values in ``tests/data/sample_quality_weights`` come from limma's
``arrayWeights``, ``voom(weights = ...)`` and ``voomWithQualityWeights``, and
downstream from ``lmFit``, ``contrasts.fit`` and ``eBayes`` (``legacy = TRUE``);
``validation/sample_quality_weights.py`` writes them. limma's voom returns the
trend's weights alone, so the fixtures hold them times the prior weights. The other
tests check closed forms derived by hand and properties that a wrong implementation
would break.
"""

import contextlib
import dataclasses
import re
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.design import Design
from cytepool.normalization import tmm_factors
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.sample_weights import sample_quality_weights
from cytepool.stats.voom import VoomResult, voom, voom_with_quality_weights

DATA = Path(__file__).parent / "data" / "sample_quality_weights"
CASES = sorted(path.stem for path in DATA.glob("*.npz"))
# Iterative estimates, and everything computed from them: the close tier.
CLOSE = 1e-6
EXACT = 1e-10
TWO_GROUPS = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])


@cache
def fixture(case: str) -> dict[str, Any]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def cases_of(kind: str) -> list[str]:
    return [case for case in CASES if str(fixture(case)["kind"]) == kind]


def design_of(case: str) -> Design:
    data = fixture(case)
    samples = {
        k.removeprefix("samples "): v for k, v in data.items() if "samples " in k
    }
    contrasts = None
    if "contrast names" in data:
        names = data["contrast names"].tolist()
        contrasts = dict(zip(names, data["contrast expressions"].tolist(), strict=True))
    return Design(samples, str(data["formula"]), contrasts=contrasts)


def run_case(case: str) -> Any:
    """CytePool on a case's inputs, as the validation runs it."""
    data = fixture(case)
    design = design_of(case)
    variance = {"variance_design": data.get("var_design")}
    variance["variance_groups"] = data.get("var_group")
    method = str(data["method"])
    kind = str(data["kind"])
    span = float(data["span"]) if "span" in data else None
    no_replication = case.endswith("no_replication") and kind == "quality"
    expected = (
        pytest.warns(RuntimeWarning, match="no replication")
        if no_replication
        else contextlib.nullcontext()
    )
    with expected, warnings.catch_warnings():
        if not no_replication:
            # Cases with one gene, no replication or missing values warn.
            warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "weights":
            return sample_quality_weights(
                data["data"],
                design,
                weights=data.get("weights"),
                method=method,
                prior_n=float(data["prior_n"]),
                **variance,
            )
        if kind == "voom":
            return voom(data["data"], design, weights=data.get("weights"), span=span)
        lib = data.get("lib")
        factors = tmm_factors(data["data"], library_sizes=lib) if data["tmm"] else None
        return voom_with_quality_weights(
            data["data"],
            design,
            library_sizes=lib,
            norm_factors=factors,
            span=span,
            method=method,
            **variance,
        )


def limma(case: str, key: str, n_rows: int) -> np.ndarray:
    """A limma matrix written column by column, as ``n_rows`` rows."""
    return fixture(case)[f"limma {key}"].reshape(n_rows, -1)


def assert_p_values_agree(ours: Any, desired: np.ndarray, tolerance: float) -> None:
    """|ours - desired| <= tolerance * max(p, 1e-300) * max(1, |log p|): rounding in
    the variances moves a far-tail p-value in proportion to |log p|."""
    scale = np.maximum(desired, 1e-300)
    scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    assert np.all(np.abs(np.asarray(ours) - desired) <= tolerance * scale)


def assert_moderated_statistics_agree(
    case: str, result: VoomResult, tolerance: float
) -> None:
    """result -> fit_linear_model -> contrasts_fit -> moderated_statistics, as the
    fixture's lmFit -> contrasts.fit -> eBayes."""
    design = design_of(case)
    fit = fit_linear_model(result.log_cpm, design, weights=result.weights)
    if len(design.contrasts):
        # limma's contrasts.fit approximates the standard errors of contrasts of
        # correlated coefficients in a weighted fit; method="limma" reproduces it.
        fit = contrasts_fit(fit, design.contrasts, method="limma")
    moderated = moderated_statistics(fit)
    n_columns = moderated.t.shape[1]
    desired = limma(case, "t", n_columns).T
    np.testing.assert_allclose(moderated.t, desired, rtol=tolerance, atol=tolerance)
    desired_p = limma(case, "p.value", n_columns).T
    assert_p_values_agree(moderated.p_value, desired_p, tolerance)


# Agreement with limma.


@pytest.mark.parametrize("case", cases_of("weights"))
def test_sample_weights_match_limmas_array_weights(case: str) -> None:
    weights = run_case(case)
    np.testing.assert_allclose(
        weights, fixture(case)["limma sample.weights"], rtol=CLOSE
    )
    np.testing.assert_allclose(np.prod(weights), 1.0, rtol=1e-12)


@pytest.mark.parametrize("case", cases_of("voom"))
def test_voom_with_prior_weights_matches_limma(case: str) -> None:
    """E, the weights (limma's times the prior weights), the trend, and the
    moderated statistics of the weighted fit, all in the exact tier."""
    data = fixture(case)
    result = run_case(case)
    n_samples = data["data"].shape[0]
    np.testing.assert_allclose(
        result.log_cpm, limma(case, "E", n_samples), rtol=EXACT, atol=EXACT
    )
    np.testing.assert_allclose(
        result.weights, limma(case, "weights", n_samples), rtol=EXACT
    )
    in_trend = ~np.isnan(result.sqrt_sd)
    x = result.mean_log_count[in_trend]
    np.testing.assert_allclose(x, data["limma xy.x"], rtol=EXACT, atol=EXACT)
    # Square-root standard deviations near zero are rounding noise in both.
    scale = float(np.median(data["limma xy.y"]))
    order = np.argsort(x, kind="stable")
    np.testing.assert_allclose(
        result.sqrt_sd_trend[in_trend][order],
        data["limma line.y"],
        rtol=EXACT,
        atol=EXACT * scale,
    )
    if "span" not in data:
        # A closed form: limma's to the last bit on the validation inputs.
        np.testing.assert_allclose(result.span, data["limma span"][0], rtol=EXACT)
    assert_moderated_statistics_agree(case, result, EXACT)


@pytest.mark.parametrize("case", cases_of("quality"))
def test_voom_with_quality_weights_matches_limma(case: str) -> None:
    data = fixture(case)
    result = run_case(case)
    n_samples = data["data"].shape[0]
    np.testing.assert_allclose(
        result.log_cpm, limma(case, "E", n_samples), rtol=EXACT, atol=EXACT
    )
    np.testing.assert_allclose(
        result.sample_weights, data["limma sample.weights"], rtol=CLOSE
    )
    np.testing.assert_allclose(
        result.weights, limma(case, "weights", n_samples), rtol=CLOSE
    )
    if "limma span" in data:
        np.testing.assert_allclose(result.span, data["limma span"][0], rtol=EXACT)
    if "limma t" in data:
        assert_moderated_statistics_agree(case, result, CLOSE)


def test_noisy_pseudobulk_samples_get_the_lowest_quality_weights() -> None:
    """Samples 2 and 6 of the quality cases have extra noise, as samples pooled from
    few cells do; both algorithms, and limma, give them the lowest weights."""
    for case in ("quality_gene_by_gene", "quality_reml"):
        weights = run_case(case).sample_weights
        assert set(np.argsort(weights)[:2]) == {2, 6}
        assert np.all(weights[[2, 6]] < 1.0)


def test_reml_reports_the_genes_it_leaves_out_as_limma_does() -> None:
    """limma's message names the number of genes with missing values that REML
    removes; CytePool's warning gives the same number."""
    data = fixture("missing_values_reml")
    (message,) = [w for w in data["limma warnings"] if "removing" in w]
    removed = int(re.findall(r"\d+", message)[0])
    with pytest.warns(RuntimeWarning, match=f"leaves out {removed} of 100"):
        sample_quality_weights(data["data"], TWO_GROUPS, method="reml")


# Closed forms and properties.


def simulate(seed: int, scales: Any, n_genes: int = 400) -> np.ndarray:
    """Samples x genes normal values; sample i's noise is multiplied by scales[i]."""
    rng = np.random.default_rng(seed)
    means = rng.normal(6.0, 2.0, n_genes)
    noise = rng.normal(0.0, 1.0, (len(scales), n_genes))
    return means + np.asarray(scales, dtype=float)[:, np.newaxis] * noise


@pytest.mark.parametrize("prior_n", [0.5, 10.0])
def test_one_informative_gene_moves_the_weights_by_its_score(prior_n: float) -> None:
    """With an intercept-only design, J samples and one gene with residuals e, every
    leverage is 1/J, so the gene's information is (J - 1)/J times the identity and
    the gene-by-gene update is u = z / (prior_n + (J - 1)/J), with z_j = e_j^2 / s^2
    - (J - 1)/J and s^2 = sum(e^2) / (J - 1). The weights are exp(-u). The second
    gene is constant: no residual variance, so it is skipped."""
    values = np.array([3.1, 2.4, 5.0, 3.3, 4.2])
    n = values.size
    y = np.column_stack([values, np.full(n, 7.0)])
    residuals = values - values.mean()
    z = residuals**2 / (residuals @ residuals / (n - 1)) - (n - 1) / n
    expected = np.exp(-z / (prior_n + (n - 1) / n))
    weights = sample_quality_weights(
        y, np.ones((n, 1)), prior_n=prior_n, method="gene_by_gene"
    )
    np.testing.assert_allclose(weights, expected, rtol=1e-13)


def test_samples_of_equal_quality_get_equal_weights_by_reml() -> None:
    """Every cyclic shift of a set of genes across samples: the data, and the
    intercept-only design, are unchanged by shifting the samples, so the REML
    estimate is too, and every weight is one."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (6, 40)) * np.array(
        [[1.0], [1.0], [1.0], [3.0], [1.0], [1.0]]
    )
    y = np.hstack([np.roll(base, shift, axis=0) for shift in range(6)])
    weights = sample_quality_weights(y, np.ones((6, 1)), method="reml")
    np.testing.assert_allclose(weights, 1.0, rtol=1e-12)


@pytest.mark.parametrize("method", ["gene_by_gene", "reml"])
def test_a_noisier_sample_gets_a_lower_weight(method: str) -> None:
    previous = 1.0
    for scale in (1.5, 2.0, 3.0):
        y = simulate(5, [1, 1, scale, 1, 1, 1, 1, 1])
        weights = sample_quality_weights(y, TWO_GROUPS, method=method)
        assert np.argmin(weights) == 2
        assert weights[2] < previous
        previous = weights[2]


@pytest.mark.parametrize("method", ["gene_by_gene", "reml"])
def test_weights_do_not_depend_on_the_order_of_the_samples(method: str) -> None:
    y = simulate(6, [1, 2, 1, 1, 1, 3, 1, 1])
    if method == "gene_by_gene":
        y[np.random.default_rng(7).random(y.shape) < 0.05] = np.nan
    groups = np.array(["a", "a", "b", "b", "c", "c", "a", "b"])
    order = np.random.default_rng(8).permutation(8)
    for variance in ({}, {"variance_groups": groups}):
        base = sample_quality_weights(y, TWO_GROUPS, method=method, **variance)
        shuffled = {k: v[order] for k, v in variance.items()}
        permuted = sample_quality_weights(
            y[order], TWO_GROUPS[order], method=method, **shuffled
        )
        np.testing.assert_allclose(permuted, base[order], rtol=1e-12)


def test_reml_weights_do_not_depend_on_the_order_of_the_genes() -> None:
    """REML sums over genes; the gene-by-gene update, by design, does depend on it."""
    y = simulate(9, [1, 1, 1, 2, 1, 1, 1, 1])
    order = np.random.default_rng(10).permutation(y.shape[1])
    base = sample_quality_weights(y, TWO_GROUPS, method="reml")
    permuted = sample_quality_weights(y[:, order], TWO_GROUPS, method="reml")
    np.testing.assert_allclose(permuted, base, rtol=1e-12)


@pytest.mark.parametrize("method", ["gene_by_gene", "reml"])
def test_only_the_spans_of_the_designs_matter(method: str) -> None:
    """Reparametrising the design or the variance design, adding an intercept to the
    variance design, or scaling and shifting the values changes no weight."""
    y = simulate(11, [1, 1, 2, 1, 1, 1, 3, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    indicators = (groups[:, np.newaxis] == np.arange(3)).astype(float)
    base = sample_quality_weights(y, TWO_GROUPS, variance_groups=groups, method=method)
    means = np.column_stack([1.0 - TWO_GROUPS[:, 1], TWO_GROUPS[:, 1]])
    other = np.column_stack([np.ones(8), indicators[:, 1:] * 3.0 + 1.0])
    for design, variance, values in (
        (means, {"variance_groups": groups}, y),
        (TWO_GROUPS, {"variance_design": indicators}, y),
        (TWO_GROUPS, {"variance_design": other}, y),
        (TWO_GROUPS, {"variance_groups": groups}, 7.0 * y - 3.0),
    ):
        weights = sample_quality_weights(values, design, method=method, **variance)
        np.testing.assert_allclose(weights, base, rtol=1e-11)


def test_samples_of_a_variance_group_share_a_weight() -> None:
    y = simulate(12, [1, 1, 2, 1, 1, 1, 3, 1])
    groups = ["x", "y", "x", "y", "x", "y", "x", "y"]
    for method in ("gene_by_gene", "reml"):
        weights = sample_quality_weights(
            y, TWO_GROUPS, variance_groups=groups, method=method
        )
        np.testing.assert_allclose(weights[0::2], weights[0], rtol=1e-13)
        np.testing.assert_allclose(weights[1::2], weights[1], rtol=1e-13)
        assert weights[0] < 1.0 < weights[1]


def test_a_large_prior_squeezes_the_weights_to_one() -> None:
    y = simulate(13, [1, 1, 1, 1, 1, 1, 3, 1])
    for method in ("gene_by_gene", "reml"):
        weights = sample_quality_weights(y, TWO_GROUPS, prior_n=1e12, method=method)
        np.testing.assert_allclose(weights, 1.0, atol=1e-8)


def test_gene_by_gene_ignores_weights_that_are_constant_within_a_gene() -> None:
    """Weights that scale every value of a gene alike change neither its standardised
    residuals relative to their variance nor its leverages."""
    y = simulate(14, [1, 1, 1, 2, 1, 1, 1, 1], n_genes=100)
    base = sample_quality_weights(y, TWO_GROUPS, method="gene_by_gene")
    feature = np.random.default_rng(15).uniform(0.2, 5.0, (1, 100))
    for weights in (feature, np.full(y.shape, 3.0)):
        result = sample_quality_weights(
            y, TWO_GROUPS, weights=weights, method="gene_by_gene"
        )
        np.testing.assert_allclose(result, base, rtol=1e-12)


def test_reml_with_missing_values_uses_the_complete_genes() -> None:
    y = simulate(16, [1, 1, 1, 2, 1, 1, 1, 1], n_genes=60)
    y[np.random.default_rng(17).random(y.shape) < 0.05] = np.nan
    complete = ~np.isnan(y).any(axis=0)
    with pytest.warns(RuntimeWarning, match=f"leaves out {60 - complete.sum()} of 60"):
        weights = sample_quality_weights(y, TWO_GROUPS, method="reml")
    expected = sample_quality_weights(y[:, complete], TWO_GROUPS, method="reml")
    np.testing.assert_array_equal(weights, expected)


def test_reml_warns_when_it_stops_at_the_iteration_limit() -> None:
    y = simulate(18, [1, 1, 1, 1, 1, 1, 3, 1])
    converged = sample_quality_weights(y, TWO_GROUPS, method="reml")
    with pytest.warns(RuntimeWarning, match="did not converge in 1 iterations"):
        one_step = sample_quality_weights(
            y, TWO_GROUPS, method="reml", max_iterations=1
        )
    assert not np.allclose(one_step, converged, rtol=1e-3)
    # More iterations than it needs change nothing, and do not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = sample_quality_weights(y, TWO_GROUPS, method="reml", max_iterations=100)
    np.testing.assert_array_equal(again, converged)


def test_reml_stops_where_its_criterion_rises_as_limma_does() -> None:
    """On these data the criterion rises at the fourth iteration: limma stops
    without that step, with a warning, and so does CytePool. The weights are those
    of the first three iterations, and limma's."""
    case = "reml_criterion_rises"
    data = fixture(case)
    assert any("prematurely" in w for w in data["limma warnings"])
    with pytest.warns(RuntimeWarning, match="stopped after 3 iterations without"):
        weights = sample_quality_weights(data["data"], design_of(case), method="reml")
    with pytest.warns(RuntimeWarning, match="did not converge in 3 iterations"):
        three = sample_quality_weights(
            data["data"], design_of(case), method="reml", max_iterations=3
        )
    np.testing.assert_array_equal(weights, three)
    np.testing.assert_allclose(weights, data["limma sample.weights"], rtol=CLOSE)


def test_a_design_can_be_the_variance_design() -> None:
    """A Design of the batches spans what their labels do; one with random effects
    is refused."""
    y = simulate(22, [1, 1, 2, 1, 1, 1, 3, 1])
    samples = {"batch": np.array(["p", "q", "p", "q", "r", "r", "p", "q"])}
    for method in ("gene_by_gene", "reml"):
        from_labels = sample_quality_weights(
            y, TWO_GROUPS, variance_groups=samples["batch"], method=method
        )
        from_design = sample_quality_weights(
            y, TWO_GROUPS, variance_design=Design(samples, "~ batch"), method=method
        )
        np.testing.assert_allclose(from_design, from_labels, rtol=1e-12)
    samples["donor"] = np.repeat(["d1", "d2", "d3", "d4"], 2)
    random = Design(samples, "~ batch", random="(1 | donor)")
    with pytest.raises(ValueError, match="variance_design must not have random"):
        sample_quality_weights(y, TWO_GROUPS, variance_design=random)


@pytest.mark.parametrize(
    ("y", "design", "message"),
    [
        (simulate(19, [1] * 8)[:, :1], TWO_GROUPS, "at least two features, got 1"),
        (simulate(19, [1] * 8), np.eye(8)[:, :7], "at least two residual degrees"),
        (np.ones((8, 5)), TWO_GROUPS, "positive residual variance"),
    ],
)
def test_weights_are_one_without_the_information_to_estimate_them(
    y: np.ndarray, design: np.ndarray, message: str
) -> None:
    """limma returns weights of one in these cases; CytePool also warns."""
    for method in ("gene_by_gene", "reml"):
        expected = message
        if method == "gene_by_gene" and "residual" in message:
            expected = "no feature has at least two residual degrees of freedom"
        with pytest.warns(RuntimeWarning, match=expected):
            weights = sample_quality_weights(y, design, method=method)
        np.testing.assert_array_equal(weights, 1.0)


@pytest.mark.parametrize("method", ["gene_by_gene", "reml"])
def test_a_sample_without_information_needs_the_prior(method: str) -> None:
    """The only sample of a group has a leverage of one and zero residuals: nothing
    is known of its variance. Without a prior the information is singular; with one,
    its weight stays near one."""
    y = simulate(20, [1] * 8)
    alone = np.column_stack([np.ones(8), np.eye(8)[:, 7]])
    with pytest.raises(ValueError, match="information about the sample weights is"):
        sample_quality_weights(y, alone, prior_n=0.0, method=method)
    weights = sample_quality_weights(y, alone, method=method)
    assert abs(np.log(weights[7])) < 0.05


Y = simulate(21, [1] * 8, n_genes=20)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"y": Y[0]}, ValueError, "y must be a 2-D array"),
        ({"y": Y[:6]}, ValueError, "y has 6 rows but design has 8"),
        ({"y": np.where(Y > 7, np.inf, Y)}, ValueError, "infinite values"),
        ({"design": TWO_GROUPS * np.nan}, ValueError, "design must be finite"),
        ({"design": TWO_GROUPS[:, [0, 1, 1]]}, ValueError, "full column rank"),
        ({"weights": np.ones(20)}, ValueError, r"shape \(8,\) for sample weights"),
        ({"weights": -np.ones(8)}, ValueError, "sample weights must be finite and"),
        ({"weights": np.full((8, 20), np.nan)}, ValueError, "observation weights must"),
        ({"variance_design": np.ones((7, 1))}, ValueError, "one row per sample"),
        ({"variance_design": np.ones((8, 2))}, ValueError, "must vary between"),
        ({"variance_design": np.full(8, np.nan)}, ValueError, "must be finite"),
        ({"variance_groups": [1, 2]}, ValueError, "one label per sample"),
        ({"variance_groups": ["a"] * 8}, ValueError, "at least two groups"),
        ({"variance_groups": [1.0, np.nan] * 4}, ValueError, "must not contain NaN"),
        ({"variance_groups": ["a", None] * 4}, ValueError, "must not contain None"),
        (
            {"variance_groups": [1, 2] * 4, "variance_design": np.eye(8)},
            ValueError,
            "not both",
        ),
        ({"prior_n": -1.0}, ValueError, "prior_n must be finite and at least 0"),
        ({"prior_n": np.inf}, ValueError, "prior_n must be finite"),
        ({"prior_n": True}, TypeError, "prior_n must be a real number"),
        ({"method": "genebygene"}, ValueError, "method must be one of 'auto'"),
        ({"max_iterations": 0}, ValueError, "max_iterations must be at least 1"),
        ({"max_iterations": 2.5}, TypeError, "max_iterations must be an integer"),
        ({"tolerance": 0.0}, ValueError, "tolerance must be positive"),
        ({"tolerance": "small"}, TypeError, "tolerance must be a real number"),
    ],
)
def test_invalid_arguments_are_refused(
    arguments: dict[str, Any], error: type, message: str
) -> None:
    call = {"y": Y, "design": TWO_GROUPS} | arguments
    with pytest.raises(error, match=message):
        sample_quality_weights(**call)


def test_a_design_with_random_effects_is_refused() -> None:
    samples = {"group": np.repeat(["A", "B"], 4), "donor": np.tile(["d1", "d2"], 4)}
    design = Design(samples, "~ group", random="(1 | donor)")
    match = "sample_quality_weights fits fixed effects only, so it would drop"
    with pytest.raises(ValueError, match=match):
        sample_quality_weights(Y, design)


# voom with prior weights.

COUNTS = fixture("voom_observation_weights")["data"]
PRIOR = fixture("voom_observation_weights")["weights"]


def test_constant_prior_weights_change_no_weight() -> None:
    """Weights c for every observation scale the residual standard deviations by
    c^(1/2), the trend's weights by 1/c, and the prior multiplies them back."""
    base = voom(COUNTS, TWO_GROUPS, span=0.5)
    for weights in (np.full(8, 3.0), np.full(COUNTS.shape, 0.25)):
        result = voom(COUNTS, TWO_GROUPS, span=0.5, weights=weights)
        np.testing.assert_allclose(result.weights, base.weights, rtol=1e-12)


def test_a_gene_whose_prior_weights_are_all_zero_is_left_out_of_the_trend() -> None:
    """Its weights are zero, and every other gene gets the weights it gets without
    that gene (with the library sizes of every gene). limma's trend fails here."""
    weights = PRIOR.copy()
    weights[:, 3] = 0.0
    sizes = COUNTS.sum(axis=1)
    result = voom(COUNTS, TWO_GROUPS, span=0.5, weights=weights, library_sizes=sizes)
    assert np.isnan(result.sqrt_sd[3]) and not np.isnan(result.mean_log_count[3])
    np.testing.assert_array_equal(result.weights[:, 3], 0.0)
    others = np.delete(np.arange(COUNTS.shape[1]), 3)
    reduced = voom(
        COUNTS[:, others],
        TWO_GROUPS,
        span=0.5,
        weights=weights[:, others],
        library_sizes=sizes,
    )
    np.testing.assert_allclose(result.weights[:, others], reduced.weights, rtol=1e-12)


def test_zero_weights_that_leave_coefficients_inestimable() -> None:
    """Gene 0 has zero weight in every sample of group A, so neither coefficient of
    the two-group design is estimable for it, but its fitted value in group B is the
    weighted mean of its group B values. Its weights are the trend's at that fitted
    log count times the prior, read here with numpy's interpolation; zero in group
    A. limma returns missing weights for the gene."""
    weights = PRIOR.copy()
    weights[:4, 0] = 0.0
    result = voom(COUNTS, TWO_GROUPS, span=0.5, weights=weights)
    np.testing.assert_array_equal(result.weights[:4, 0], 0.0)
    prior = weights[4:, 0]
    fitted = np.sum(prior * result.log_cpm[4:, 0]) / prior.sum()
    sizes = result.effective_library_sizes[4:]
    fitted_log_count = np.log2(2.0**fitted * (sizes + 1.0) * 1e-6)
    in_trend = ~np.isnan(result.sqrt_sd)
    knots, first = np.unique(result.mean_log_count[in_trend], return_index=True)
    curve = result.sqrt_sd_trend[in_trend][first]
    expected = np.interp(fitted_log_count, knots, curve) ** -4.0 * prior
    np.testing.assert_allclose(result.weights[4:, 0], expected, rtol=1e-12)


def test_a_vector_of_weights_is_sample_weights_also_for_square_counts() -> None:
    """With as many genes as samples a vector could be either; it is sample weights,
    as for every other shape (limma's voom takes it for gene weights)."""
    counts = COUNTS[:, :8]
    weights = np.linspace(0.5, 2.0, 8)
    vector = voom(counts, TWO_GROUPS, span=0.5, weights=weights)
    columns = np.repeat(weights[:, np.newaxis], 8, axis=1)
    as_samples = voom(counts, TWO_GROUPS, span=0.5, weights=columns)
    np.testing.assert_allclose(vector.weights, as_samples.weights, rtol=1e-12)
    as_genes = voom(counts, TWO_GROUPS, span=0.5, weights=weights[np.newaxis, :])
    assert not np.allclose(vector.weights, as_genes.weights)


def test_without_residual_df_the_weights_are_the_prior_weights() -> None:
    """Zero sample weights leave one sample per group, so no trend can be fitted."""
    weights = np.array([1.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    with pytest.warns(RuntimeWarning, match="every weight is set to its prior weight"):
        result = voom(COUNTS, TWO_GROUPS, weights=weights)
    np.testing.assert_array_equal(
        result.weights, np.repeat(weights[:, np.newaxis], COUNTS.shape[1], axis=1)
    )


def test_voom_refuses_random_effects_under_its_own_name() -> None:
    samples = {"group": np.repeat(["A", "B"], 4), "donor": np.tile(["d1", "d2"], 4)}
    design = Design(samples, "~ group", random="(1 | donor)")
    for function in (voom, voom_with_quality_weights):
        with pytest.raises(ValueError, match=f"{function.__name__} fits fixed effects"):
            function(COUNTS, design)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (PRIOR[0], r"\(8,\) for sample weights, \(1, 100\) for feature weights"),
        (-PRIOR, "observation weights must be finite and non-negative"),
        (np.full(8, np.nan), "sample weights must be finite and non-negative"),
        (np.ones((1, 8)), r"got \(1, 8\)"),
    ],
)
def test_voom_refuses_invalid_prior_weights(weights: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        voom(COUNTS, TWO_GROUPS, weights=weights)


# voom with quality weights.


def test_quality_weights_combine_voom_and_sample_weights_in_limmas_steps() -> None:
    """voom, sample weights from its weights, voom again with them as prior weights,
    sample weights again from the second trend's weights, and their product. This
    checks how the public functions fit together; the limma fixtures above check
    that these are limma's steps."""
    counts = fixture("quality_gene_by_gene")["data"]
    first = voom(counts, TWO_GROUPS, span=0.5)
    quality = sample_quality_weights(
        first.log_cpm, TWO_GROUPS, weights=first.weights, method="gene_by_gene"
    )
    second = voom(counts, TWO_GROUPS, span=0.5, weights=quality)
    trend = second.weights / quality[:, np.newaxis]
    final = sample_quality_weights(
        second.log_cpm, TWO_GROUPS, weights=trend, method="gene_by_gene"
    )
    result = voom_with_quality_weights(counts, TWO_GROUPS, span=0.5)
    np.testing.assert_allclose(result.sample_weights, final, rtol=1e-12)
    np.testing.assert_allclose(result.weights, trend * final[:, np.newaxis], rtol=1e-12)
    np.testing.assert_allclose(result.sqrt_sd_trend, second.sqrt_sd_trend, rtol=1e-12)


def test_quality_weights_without_replication_warn_once_and_are_one() -> None:
    counts = fixture("quality_no_replication")["data"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = voom_with_quality_weights(counts, np.eye(4))
    assert [str(w.message) for w in caught] == [
        "the design leaves no residual degrees of freedom (no replication), so no "
        "mean-variance trend can be fitted; every weight is set to 1"
    ]
    np.testing.assert_array_equal(result.weights, 1.0)
    np.testing.assert_array_equal(result.sample_weights, 1.0)
    assert np.isnan(result.span)


def test_quality_weights_with_one_residual_df_are_voom_weights() -> None:
    counts = fixture("quality_gene_by_gene")["data"][:3]
    design = TWO_GROUPS[[0, 4, 5]]
    with pytest.warns(RuntimeWarning, match="at least two residual degrees of freedom"):
        result = voom_with_quality_weights(counts, design, span=0.5)
    np.testing.assert_array_equal(result.sample_weights, 1.0)
    expected = voom(counts, design, span=0.5)
    np.testing.assert_allclose(result.weights, expected.weights, rtol=1e-14)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"method": "auto"}, "method must be one of 'gene_by_gene', 'reml'"),
        ({"variance_groups": ["a"] * 8}, "at least two groups"),
        ({"prior_n": -2.0}, "prior_n must be finite and at least 0"),
        ({"span": 2.0}, r"span must be a fraction of the features in \(0, 1\]"),
    ],
)
def test_voom_with_quality_weights_refuses_invalid_arguments(
    arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        voom_with_quality_weights(COUNTS, TWO_GROUPS, **arguments)


def test_result_holds_read_only_sample_weights_with_a_summary() -> None:
    result = run_case("quality_reml")
    with pytest.raises(ValueError, match="read-only"):
        result.sample_weights[0] = 1.0
    assert repr(result) == (
        "VoomResult(n_samples=8, n_features=100, features_in_trend=100, span=0.5, "
        "with sample weights)"
    )
    with pytest.raises(ValueError, match=r"sample_weights must have shape \(8,\)"):
        dataclasses.replace(result, sample_weights=np.ones(7))
    with pytest.raises(ValueError, match="sample_weights must be positive"):
        dataclasses.replace(result, sample_weights=np.zeros(8))
