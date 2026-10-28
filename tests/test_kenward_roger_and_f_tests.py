"""Tests for Kenward-Roger's tests and the F-tests of mixed models, and dream's use of them.

Reference values in ``tests/data/kenward_roger_and_f_tests`` come from pbkrtest 0.5.3 and
lmerTest 3.1-3 on lme4 1.1-38, evaluated at CytePool's REML estimates, and from
variancePartition 1.36.2's ``dream``, ``eBayes`` and ``topTable``, written by
``validation/kenward_roger_and_f_tests.py``. The other tests check closed forms of
balanced designs, where Kenward-Roger's degrees of freedom are those of the analysis
of variance and the adjustment vanishes, and invariants that a wrong implementation
breaks: reparametrised designs and equivalent contrasts, relabelled donors and
reordered samples, the scale of the weights, and an F-test of one contrast, which is the
square of its t-test.
"""

import json
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool import Cube, Design, dream_voom, from_bulk
from cytepool._jit import thread_limit
from cytepool.stats._fixed_effect_inference import combined_df
from cytepool.stats.dream import moderated_fixed_effect_tests, voom_with_dream_weights
from cytepool.stats.mixed_models import (
    FixedEffectFTest,
    MixedModelFit,
    fit_mixed_model,
    fixed_effect_f_test,
    fixed_effect_tests,
)

DATA = Path(__file__).parent / "data" / "kenward_roger_and_f_tests"
# Closed forms at given estimates.
EXACT = 1e-10
# variancePartition's fits converge to about 1e-8 of the optimum (see the validation
# script), and its moderated tests inherit it.
CLOSE = 1e-6
# lmerTest differentiates the deviance numerically for Satterthwaite's F-tests
# (validation/mixed_models.py).
SATTERTHWAITE = 1e-5
CASES = (
    "time_course",
    "unbalanced",
    "crossed",
    "nested",
    "boundary",
    "weights",
    "weights_precision",
    "weights_crossed",
    "weights_crossed_precision",
)
CONTRASTS = {
    "t2_vs_t1": [0, -1, 1, 0],
    "(Intercept)": [1, 0, 0, 0],
    "timet1": [0, 1, 0, 0],
    "timet2": [0, 0, 1, 0],
    "age": [0, 0, 0, 1],
}


@cache
def fixture(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((DATA / f"{name}.json").read_text())


@cache
def fitted(name: str) -> tuple[MixedModelFit, str]:
    """The fit of a case's fixture genes, and the treatment of its weights."""
    data, meta = fixture(name)
    design = Design(meta["samples"], meta["formula"], random=meta["random"])
    fit = fit_mixed_model(data["y"], design, weights=data.get("weights"))
    return fit, "precision" if meta["variant"] == "precision" else "average"


def assert_relative(
    actual: Any, desired: Any, tolerance: float, floor: Any = 0.0
) -> None:
    """|actual - desired| <= tolerance * max(|desired|, floor), NaN where desired is."""
    actual, desired = np.asarray(actual, float), np.asarray(desired, float)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(desired))
    scale = np.fmax(np.abs(desired), floor)
    finite = ~np.isnan(desired)
    excess = np.abs(actual - desired)[finite] - tolerance * scale[finite]
    assert np.all(excess <= 0.0), float(np.max(excess))


def p_scale(p: np.ndarray) -> np.ndarray:
    """max(p, 1e-300) max(1, |log p|): p-values compared on the scale of log p."""
    bounded = np.fmax(p, 1e-300)
    return bounded * np.fmax(1.0, np.abs(np.log(bounded)))


# Against pbkrtest and lmerTest.


@pytest.mark.parametrize("name", CASES)
def test_kenward_roger_t_tests_match_pbkrtest_and_lmertest(name: str) -> None:
    data, _ = fixture(name)
    fit, variant = fitted(name)
    tests = fixed_effect_tests(
        fit, df_method="kenward_roger", kenward_roger_weights=variant
    )
    adjusted = data["adjusted"]
    scale = np.max(np.abs(adjusted), axis=(1, 2))[:, None, None]
    assert_relative(tests.covariance, adjusted, EXACT, scale)
    coef = data["coef"]  # estimate, standard error, df, t, p
    assert_relative(tests.estimate, coef[..., 0], EXACT, coef[..., 1])
    assert_relative(tests.standard_error, coef[..., 1], EXACT)
    assert_relative(tests.t, coef[..., 3], EXACT, 1.0)
    assert_relative(tests.df, coef[..., 2], EXACT)
    assert_relative(tests.p_value, coef[..., 4], EXACT, p_scale(coef[..., 4]))
    contrast = fixed_effect_tests(
        fit,
        data["contrast_weights"],
        df_method="kenward_roger",
        kenward_roger_weights=variant,
    )
    expected = data["contrast"]
    assert_relative(contrast.standard_error[:, 0], expected[:, 1], EXACT)
    assert_relative(contrast.df[:, 0], expected[:, 2], EXACT)
    assert_relative(contrast.t[:, 0], expected[:, 3], EXACT, 1.0)
    assert tests.df_method == "kenward_roger" and tests.kenward_roger_weights == variant


@pytest.mark.parametrize("name", CASES)
def test_f_tests_match_krmodcomp_and_contestmd(name: str) -> None:
    data, _ = fixture(name)
    fit, variant = fitted(name)
    for key in [key.split(" ", 1)[1] for key in data if key.startswith("f_matrix ")]:
        matrix = data[f"f_matrix {key}"].T
        # KRmodcomp's stat, ndf, ddf, F.scaling, p; contestMD's F, NumDF, DenDF, p by
        # Kenward-Roger, then by Satterthwaite.
        f = data[f"f {key}"]
        kr = fixed_effect_f_test(
            fit, matrix, df_method="kenward_roger", kenward_roger_weights=variant
        )
        assert kr.df1 == 2 and np.all(f[:, 1] == 2)  # 'pairs' spans 2 dimensions
        assert_relative(kr.f_statistic, f[:, 0], EXACT)
        assert_relative(kr.df2, f[:, 2], EXACT)
        assert_relative(kr.scaling, f[:, 3], EXACT)
        assert_relative(kr.p_value, f[:, 4], EXACT, p_scale(f[:, 4]))
        satterthwaite = fixed_effect_f_test(fit, matrix)
        assert_relative(satterthwaite.f_statistic, f[:, 9], EXACT)
        np.testing.assert_array_equal(satterthwaite.scaling, 1.0)
        assert_relative(satterthwaite.df2, f[:, 11], SATTERTHWAITE)
        assert_relative(
            satterthwaite.p_value, f[:, 12], SATTERTHWAITE, p_scale(f[:, 12])
        )


def test_the_cases_cover_weights_two_random_effects_and_the_boundary() -> None:
    # Without them the comparison would leave pbkrtest's treatment of weights, the
    # residual's derivatives and the boundary components untested.
    assert fixture("weights")[0]["weights"].std() > 0.3
    assert fitted("weights_crossed")[0].component_names == ("donor", "batch")
    assert fitted("nested")[0].component_names == ("donor", "sample:donor")
    assert np.count_nonzero(fitted("boundary")[0].boundary) >= 3
    # The two treatments of the weights differ on these data.
    fit, _ = fitted("weights")
    average = fixed_effect_tests(fit, df_method="kenward_roger")
    precision = fixed_effect_tests(
        fit, df_method="kenward_roger", kenward_roger_weights="precision"
    )
    assert np.max(np.abs(average.df - precision.df) / precision.df) > 0.01


# Against dream.


def study(name: str, method: str) -> tuple[Any, Any]:
    data, samples = fixture(f"study_{name}")
    design = Design(samples, "~ time + age", random="(1 | donor)")
    voom = voom_with_dream_weights(data["counts"], design, span=0.5)
    fit = fit_mixed_model(voom.log2_cpm, design, weights=voom.weights)
    tests = fixed_effect_tests(fit, CONTRASTS, df_method=method)  # type: ignore[arg-type]
    time = fixed_effect_tests(
        fit,
        {key: CONTRASTS[key] for key in ("timet1", "timet2")},
        df_method=method,  # type: ignore[arg-type]
    )
    return moderated_fixed_effect_tests(tests), moderated_fixed_effect_tests(time)


@pytest.mark.parametrize(
    ("name", "ddf", "method"),
    [
        ("n12", "Kenward-Roger", "kenward_roger"),
        ("n12", "adaptive", "kenward_roger"),
        ("n12", "Satterthwaite", "satterthwaite"),
        ("n20", "adaptive", "satterthwaite"),
    ],
)
def test_moderated_tests_match_dream(name: str, ddf: str, method: str) -> None:
    """dream's adaptive choice is Kenward-Roger's method for 12 samples and
    Satterthwaite's for 20, as the tests of each method reproduce."""
    data, _ = fixture(f"study_{name}")
    moderated, time = study(name, method)
    reference = {
        key.split(" ", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{ddf} ")
    }
    assert_relative(moderated.tests.standard_error, reference["se"], CLOSE)
    assert_relative(moderated.tests.df, reference["df"], CLOSE)
    assert_relative(moderated.t, reference["t"], CLOSE, 1.0)
    assert_relative(moderated.log_odds, reference["lods"], CLOSE, 1.0)
    assert_relative(moderated.z, reference["z"], CLOSE, 1.0)
    assert_relative(moderated.df_total, reference["df_total"], CLOSE)
    assert_relative(moderated.p_value, reference["p"], CLOSE, p_scale(reference["p"]))
    # topTable's F; its p-values use the first gene's degrees of freedom for all.
    assert_relative(time.f_statistic, reference["F"], CLOSE)
    assert time.f_df1 == 2 and np.ptp(time.f_df2) > 1.0


@cache
def cube_and_design() -> tuple[Cube, Design]:
    data, samples = fixture("cube")
    cube = Cube(
        {"counts": data["counts"]},
        samples=samples,
        groups={"group_id": ["small", "large"]},
        features={"feature_id": [f"gene{i}" for i in range(data["counts"].shape[2])]},
        n_cells=data["n_cells"],
    )
    design = Design(
        cube.samples,
        "~ time + age",
        random="(1 | donor)",
        levels={"time": ["t0", "t1", "t2"]},
        contrasts={
            "t1_vs_t0": "timet1",
            "t2_vs_t0": "timet2",
            "t2_vs_t1": "timet2 - timet1",
        },
    )
    return cube, design


@cache
def cube_result() -> Any:
    cube, design = cube_and_design()
    return dream_voom(
        cube, design, span=0.5, f_tests={"time": ["t1_vs_t0", "t2_vs_t0"]}
    )


def test_dream_voom_by_default_chooses_the_degrees_of_freedom_as_dream_does() -> None:
    """12 samples left in "small": Kenward-Roger's method; 24 in "large":
    Satterthwaite's; the tables are dream's with ddf = "adaptive"."""
    data, _ = fixture("cube")
    result = cube_result()
    assert result.summary["df_method"].tolist() == ["kenward_roger", "satterthwaite"]
    assert result.summary["n_samples_used"].tolist() == [12, 24]
    for group in ("small", "large"):
        for name in ("t1_vs_t0", "t2_vs_t1"):
            table = result.table(group, name)
            reference = {
                key.split(" ")[-1]: value
                for key, value in data.items()
                if key.startswith(f"{group} {name} ")
            }
            assert_relative(table["t"], reference["t"], CLOSE, 1.0)
            assert_relative(table["df"], reference["df_total"], CLOSE)
            assert_relative(table["log_odds"], reference["lods"], CLOSE, 1.0)
            assert_relative(table["z_statistic"], reference["z"], CLOSE, 1.0)
            assert_relative(
                table["p_value"], reference["p"], CLOSE, p_scale(reference["p"])
            )
        f_table = result.table(group, "time")
        assert_relative(f_table["f_statistic"], data[f"{group} time F"], CLOSE)


def test_the_f_tables_of_dream_voom() -> None:
    result = cube_result()
    table = result.table("small", "time")
    assert table.columns == (
        "feature_id",
        "log2_fold_change_t1_vs_t0",
        "log2_fold_change_t2_vs_t0",
        "average_log2_cpm",
        "f_statistic",
        "df",
        "p_value",
        "adjusted_p_value",
        "standardized_f_statistic",
    )
    contrast = result.table("small", "t1_vs_t0")
    np.testing.assert_array_equal(
        table["log2_fold_change_t1_vs_t0"], contrast["log2_fold_change"]
    )
    np.testing.assert_array_equal(table["feature_id"], contrast["feature_id"])
    # F.std is the F-statistic on (2, infinity) df, half a chi-squared on 2 df, with
    # the test's p-value: exp(-F.std) is that p-value.
    np.testing.assert_allclose(
        np.exp(-table["standardized_f_statistic"]), table["p_value"], rtol=1e-12
    )
    assert result.f_tests == {"time": ("t1_vs_t0", "t2_vs_t0")}
    groups = result.provenance["groups"]
    assert [group["f_test_df1"] for group in groups] == [{"time": 2}, {"time": 2}]
    assert [group["features_satterthwaite_fallback"] for group in groups] == [0, 0]
    assert result.provenance["settings"]["f_tests"] == {
        "time": ["t1_vs_t0", "t2_vs_t0"]
    }
    random = result.provenance["design"]
    assert random["random"] == ["(1 | donor)"] and "donor" in random["samples"]


# Closed forms of balanced designs.


def test_a_balanced_one_way_design_has_kenward_rogers_closed_forms() -> None:
    """In a balanced one-way layout the mean has the variance MSB / n on q - 1 degrees
    of freedom exactly, and the adjustment vanishes: G_k maps the fixed effects'
    columns into their span (Kenward and Roger 1997, section 5)."""
    q, m = 6, 4
    rng = np.random.default_rng(31)
    groups = np.repeat([f"G{i}" for i in range(q)], m)
    design = Design({"group": groups}, "~ 1", random="(1 | group)")
    y = 5.0 + rng.normal(size=(q, 5))[np.repeat(np.arange(q), m)] * [
        0.6,
        1.0,
        2.0,
        3.0,
        5.0,
    ]
    y = y + rng.normal(size=(q * m, 5))
    fit = fit_mixed_model(y, design)
    assert not fit.boundary.any()
    kr = fixed_effect_tests(fit, df_method="kenward_roger")
    satterthwaite = fixed_effect_tests(fit)
    np.testing.assert_allclose(kr.df[:, 0], q - 1, rtol=EXACT)
    np.testing.assert_allclose(kr.covariance, satterthwaite.covariance, rtol=EXACT)
    means = y.reshape(q, m, -1).mean(axis=1)
    msb = m * np.sum((means - y.mean(axis=0)) ** 2, axis=0) / (q - 1)
    np.testing.assert_allclose(kr.standard_error[:, 0] ** 2, msb / (q * m), rtol=EXACT)


def test_a_randomised_block_design_has_the_classical_degrees_of_freedom() -> None:
    """Donors as random blocks with each treatment once per donor: a difference of
    treatments lies within donors, on (q - 1)(t - 1) df, and a treatment's mean has
    Satterthwaite's combination of the donor and residual mean squares, as
    Kenward-Roger's method gives; the adjustment vanishes. The F-test of the treatments
    is the analysis of variance's, MS treatments / MSE on (t - 1, (q - 1)(t - 1))."""
    q, t = 7, 3
    rng = np.random.default_rng(5)
    donors = np.repeat([f"D{i}" for i in range(q)], t)
    treatment = np.tile(["a", "b", "c"], q)
    design = Design(
        {"donor": donors, "treatment": treatment},
        "~ 0 + treatment",
        random="(1 | donor)",
    )
    y = (
        np.array([4.0, 5.0, 4.5])[np.tile(np.arange(t), q)][:, None]
        + rng.normal(0.0, 2.0, (q, 4))[np.repeat(np.arange(q), t)]
        + rng.normal(0.0, 0.5, (q * t, 4))
    )
    fit = fit_mixed_model(y, design)
    assert not fit.boundary.any()
    tests = fixed_effect_tests(
        fit, {"a": [1.0, 0, 0], "b_vs_a": [-1.0, 1.0, 0]}, df_method="kenward_roger"
    )
    cells = y.reshape(q, t, -1)
    donor_means, treatment_means = cells.mean(axis=1), cells.mean(axis=0)
    grand = cells.mean(axis=(0, 1))
    msd = t * np.sum((donor_means - grand) ** 2, axis=0) / (q - 1)
    residual = cells - donor_means[:, None] - treatment_means[None] + grand
    mse = np.sum(residual**2, axis=(0, 1)) / ((q - 1) * (t - 1))
    combined = msd + (t - 1) * mse
    expected = combined**2 / (
        msd**2 / (q - 1) + ((t - 1) * mse) ** 2 / ((q - 1) * (t - 1))
    )
    np.testing.assert_allclose(tests.df[:, 0], expected, rtol=EXACT)
    np.testing.assert_allclose(tests.df[:, 1], (q - 1) * (t - 1), rtol=EXACT)
    np.testing.assert_allclose(tests.standard_error[:, 1] ** 2, 2 * mse / q, rtol=EXACT)
    f = fixed_effect_f_test(
        fit,
        {"b_vs_a": [-1.0, 1.0, 0], "c_vs_a": [-1.0, 0, 1.0]},
        df_method="kenward_roger",
    )
    mst = q * np.sum((treatment_means - grand) ** 2, axis=0) / (t - 1)
    np.testing.assert_allclose(f.f_statistic, mst / mse, rtol=EXACT)
    np.testing.assert_allclose(f.df2, (q - 1) * (t - 1), rtol=EXACT)
    np.testing.assert_allclose(f.scaling, 1.0, rtol=EXACT)
    satterthwaite = fixed_effect_f_test(
        fit, {"b_vs_a": [-1.0, 1.0, 0], "c_vs_a": [-1.0, 0, 1.0]}
    )
    np.testing.assert_allclose(satterthwaite.f_statistic, mst / mse, rtol=EXACT)
    np.testing.assert_allclose(satterthwaite.df2, (q - 1) * (t - 1), rtol=EXACT)


def test_the_denominator_df_of_an_f_test_combine_those_of_its_t_tests() -> None:
    # Values derived by hand from the rule lmerTest documents for get_Fstat_ddf.
    df = np.array(
        [
            [5.0, 5.0],
            [4.0, 12.0],
            [2.0, 30.0],
            [np.inf, 6.0],
            [np.inf, np.inf],
            [7.0, np.nan],
        ]
    )
    # [4, 12]: E = 4/2 + 12/10 = 3.2, 2 E / (E - 2) = 16/3; [inf, 6]: E = 1 + 3/2.
    expected = [5.0, 16.0 / 3.0, 2.0, 10.0, np.inf, np.nan]
    np.testing.assert_allclose(combined_df(df), expected, rtol=1e-15)
    # One keeps its own, below 2 too, as lmerTest's contestMD does (found by running it).
    np.testing.assert_allclose(combined_df(np.array([[9.0], [1.5]])), [9.0, 1.5])


# Invariants.


def test_a_reparametrised_design_changes_only_the_coefficients() -> None:
    data, meta = fixture("time_course")
    fit, _ = fitted("time_course")
    # Means of time points instead of a reference, and age in months: x_new = x t.
    samples = dict(
        meta["samples"], age_months=[12.0 * a for a in meta["samples"]["age"]]
    )
    other = Design(samples, "~ 0 + time + age_months", random="(1 | donor)")
    refit = fit_mixed_model(data["y"], other)
    old = Design(meta["samples"], meta["formula"], random=meta["random"]).matrix
    t = np.linalg.lstsq(old, other.matrix, rcond=None)[0]
    first = fixed_effect_tests(fit, df_method="kenward_roger")
    second = fixed_effect_tests(refit, df_method="kenward_roger")
    inverse = np.linalg.inv(t)
    transformed = np.einsum("ij,gjk,lk->gil", inverse, first.covariance, inverse)
    scale = np.max(np.abs(second.covariance), axis=(1, 2))[:, None, None]
    assert_relative(second.covariance, transformed, 1e-8, scale)
    # The same contrasts in both parametrisations.
    one = {"t2_vs_t1": [0, -1, 1, 0], "age": [0, 0, 0, 1]}
    two = {"t2_vs_t1": [0, -1, 1, 0], "age": [0, 0, 0, 12.0]}
    a = fixed_effect_tests(fit, one, df_method="kenward_roger")
    b = fixed_effect_tests(refit, two, df_method="kenward_roger")
    for name in ("estimate", "standard_error", "t", "df", "p_value"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-8)
    for method in ("kenward_roger", "satterthwaite"):
        f1 = fixed_effect_f_test(
            fit, {"t1": [0, 1, 0, 0], "t2": [0, 0, 1, 0]}, df_method=method
        )  # type: ignore[arg-type]
        f2 = fixed_effect_f_test(
            refit, {"t1": [-1, 1, 0, 0], "t2": [-1, 0, 1, 0]}, df_method=method
        )  # type: ignore[arg-type]
        np.testing.assert_allclose(f2.f_statistic, f1.f_statistic, rtol=1e-8)
        np.testing.assert_allclose(f2.p_value, f1.p_value, rtol=1e-7)


def test_equivalent_contrasts_give_the_same_kenward_roger_f_test() -> None:
    fit, _ = fitted("weights")
    spans = (
        {"t1": [0, 1, 0, 0], "t2": [0, 0, 1, 0]},
        {"t1": [0, 2, 0, 0], "t2_vs_t1": [0, -1, 1, 0]},
        {"t1": [0, 1, 0, 0], "t2": [0, 0, 1, 0], "t2_vs_t1": [0, -1, 1, 0]},
    )
    tests = [
        fixed_effect_f_test(fit, span, df_method="kenward_roger") for span in spans
    ]
    for other in tests[1:]:
        assert other.df1 == 2
        for name in ("f_statistic", "df2", "scaling", "p_value"):
            np.testing.assert_allclose(
                getattr(other, name), getattr(tests[0], name), rtol=1e-9, err_msg=name
            )
    # Satterthwaite's F-statistic is the same Wald statistic; its df depend on how the
    # hypothesis is spanned, as lmerTest's do.
    plain = [fixed_effect_f_test(fit, span) for span in spans]
    for other in plain[1:]:
        np.testing.assert_allclose(other.f_statistic, plain[0].f_statistic, rtol=1e-9)


@pytest.mark.parametrize("method", ["kenward_roger", "satterthwaite"])
def test_an_f_test_of_one_contrast_is_the_square_of_its_t_test(method: str) -> None:
    fit, _ = fitted("crossed")
    contrast = {"stim_minus_x": [0.0, 1.0, -1.0]}
    t = fixed_effect_tests(fit, contrast, df_method=method)  # type: ignore[arg-type]
    f = fixed_effect_f_test(fit, contrast, df_method=method)  # type: ignore[arg-type]
    assert f.df1 == 1
    np.testing.assert_allclose(f.f_statistic, t.t[:, 0] ** 2, rtol=1e-12)
    np.testing.assert_allclose(f.df2, t.df[:, 0], rtol=1e-12)
    np.testing.assert_allclose(f.p_value, t.p_value[:, 0], rtol=1e-10)
    np.testing.assert_allclose(f.scaling, 1.0, rtol=1e-12)
    # So is dream's moderated F-test of one contrast.
    moderated = moderated_fixed_effect_tests(t)
    assert moderated.f_df1 == 1
    np.testing.assert_allclose(
        moderated.f_statistic, moderated.t[:, 0] ** 2, rtol=1e-12
    )
    np.testing.assert_allclose(moderated.f_df2, moderated.df_total[:, 0], rtol=1e-12)
    np.testing.assert_allclose(moderated.f_p_value, moderated.p_value[:, 0], rtol=1e-10)


@pytest.mark.parametrize("method", ["kenward_roger", "satterthwaite"])
def test_an_f_test_of_one_contrast_keeps_degrees_of_freedom_below_two(
    method: str,
) -> None:
    """A difference between donors, two against one, has about one degree of freedom.
    Its F-test is still its t-test, as lmerTest's contestMD gives it (found by running
    it), not the 2 that the rule for several contrasts gives when one is at most 2."""
    rng = np.random.default_rng(3)
    donors = np.repeat(["A", "B", "C"], 3)
    group = np.repeat(["x", "x", "y"], 3)
    design = Design({"donor": donors, "group": group}, "~ group", random="(1 | donor)")
    y = (
        rng.normal(size=(9, 6))
        + rng.normal(0.0, 2.0, (3, 6))[np.repeat(np.arange(3), 3)]
        + (group == "y")[:, np.newaxis]
    )
    fit = fit_mixed_model(y, design)
    contrast = {"y_vs_x": [0.0, 1.0]}
    t = fixed_effect_tests(fit, contrast, df_method=method)  # type: ignore[arg-type]
    f = fixed_effect_f_test(fit, contrast, df_method=method)  # type: ignore[arg-type]
    assert np.count_nonzero(t.df[:, 0] < 2.0) >= 3
    np.testing.assert_allclose(f.df2, t.df[:, 0], rtol=1e-12)
    np.testing.assert_allclose(f.p_value, t.p_value[:, 0], rtol=1e-10)


def test_relabelled_donors_and_reordered_samples_change_nothing() -> None:
    data, meta = fixture("weights_crossed")
    fit, _ = fitted("weights_crossed")
    order = np.random.default_rng(2).permutation(data["y"].shape[0])
    samples = {key: np.asarray(value)[order] for key, value in meta["samples"].items()}
    samples["donor"] = np.array(["donor " + str(d)[::-1] for d in samples["donor"]])
    samples["batch"] = np.array(["x" + str(b) for b in samples["batch"]])
    design = Design(samples, meta["formula"], random=meta["random"])
    refit = fit_mixed_model(data["y"][order], design, weights=data["weights"][order])
    for variant in ("average", "precision"):
        first = fixed_effect_tests(
            fit,
            df_method="kenward_roger",
            kenward_roger_weights=variant,  # type: ignore[arg-type]
        )
        second = fixed_effect_tests(
            refit,
            df_method="kenward_roger",
            kenward_roger_weights=variant,  # type: ignore[arg-type]
        )
        for name in ("standard_error", "df", "p_value"):
            # Sums taken in another order differ in rounding.
            np.testing.assert_allclose(
                getattr(second, name), getattr(first, name), rtol=1e-8, err_msg=name
            )


def test_the_scale_of_the_weights_changes_nothing_and_equal_weights_are_no_weights() -> (
    None
):
    data, meta = fixture("weights")
    design = Design(meta["samples"], meta["formula"], random=meta["random"])
    fit = fit_mixed_model(data["y"], design, weights=data["weights"])
    larger = fit_mixed_model(data["y"], design, weights=7.0 * data["weights"])
    for variant in ("average", "precision"):
        a = fixed_effect_tests(
            fit, df_method="kenward_roger", kenward_roger_weights=variant
        )  # type: ignore[arg-type]
        b = fixed_effect_tests(
            larger, df_method="kenward_roger", kenward_roger_weights=variant
        )  # type: ignore[arg-type]
        np.testing.assert_allclose(b.df, a.df, rtol=1e-8)
        np.testing.assert_allclose(b.standard_error, a.standard_error, rtol=1e-8)
    plain = fit_mixed_model(data["y"], design)
    equal = fit_mixed_model(data["y"], design, weights=np.full(data["y"].shape[0], 3.0))
    reference = fixed_effect_tests(plain, df_method="kenward_roger")
    for variant in ("average", "precision"):
        tests = fixed_effect_tests(
            equal, df_method="kenward_roger", kenward_roger_weights=variant
        )  # type: ignore[arg-type]
        np.testing.assert_allclose(tests.df, reference.df, rtol=1e-8)
        np.testing.assert_allclose(
            tests.standard_error, reference.standard_error, rtol=1e-8
        )


def test_the_number_of_threads_changes_nothing() -> None:
    # The features are computed in runs on Python threads, each by the same
    # operations.
    fit, _ = fitted("weights_crossed")
    with thread_limit(1):
        serial = fixed_effect_tests(fit, df_method="kenward_roger")
    with thread_limit(3):
        threaded = fixed_effect_tests(fit, df_method="kenward_roger")
    for name in ("standard_error", "df", "covariance"):
        np.testing.assert_array_equal(getattr(threaded, name), getattr(serial, name))


# dream_voom: its choices, its records and its errors.


def ill_conditioned_cube() -> tuple[Cube, Design]:
    """15 samples whose age, 32 years on average and about 1 apart, leaves the
    condition numbers of the adjusted covariance matrices near 1e6."""
    rng = np.random.default_rng(41)
    donor = np.repeat(np.arange(5), 3)
    samples = {
        "sample_id": [f"S{i}" for i in range(15)],
        "donor": [f"D{d}" for d in donor],
        "time": ["t0", "t1", "t2"] * 5,
        "age": (32.0 + np.round(rng.normal(size=15), 3)).tolist(),
    }
    log_mean = rng.normal(3.0, 1.0, 40) + rng.normal(0.0, 0.4, (5, 40))[donor]
    counts = rng.negative_binomial(
        10, 10 / (10 + np.exp(log_mean) * rng.uniform(0.5, 2.0, (15, 1)))
    )
    cube = from_bulk(counts, samples=samples)
    design = Design(
        cube.samples, "~ time + age", random="(1 | donor)", contrasts={"t1": "timet1"}
    )
    return cube, design


def test_dream_voom_falls_back_to_satterthwaite_where_dream_does() -> None:
    """Features whose adjusted covariance has a condition number above 1e6 take
    Satterthwaite's tests; the others Kenward-Roger's."""
    cube, design = ill_conditioned_cube()
    kr = dream_voom(cube, design, df_method="kenward_roger", expression_filter=False)
    satterthwaite = dream_voom(
        cube, design, df_method="satterthwaite", expression_filter=False
    )
    group = kr.provenance["groups"][0]
    first, second = kr.table("bulk", "t1"), satterthwaite.table("bulk", "t1")
    same = np.isclose(first["df"], second["df"], rtol=1e-12)
    assert 0 < group["features_satterthwaite_fallback"] < len(same)
    assert np.count_nonzero(same) == group["features_satterthwaite_fallback"]
    np.testing.assert_allclose(
        np.asarray(first["t"])[same], np.asarray(second["t"])[same], rtol=1e-10
    )
    # dream_voom with its default is Kenward-Roger's for 15 samples.
    default = dream_voom(cube, design, expression_filter=False)
    np.testing.assert_array_equal(default.table("bulk", "t1")["df"], first["df"])
    assert default.summary["df_method"][0] == "kenward_roger"


def test_an_f_test_with_a_contrast_a_group_cannot_estimate_is_not_tested() -> None:
    cube, design = cube_and_design()
    cells = np.asarray(cube.n_cells).copy()
    cells[np.asarray(cube.samples["time"]) == "t2", 0] = 5  # no t2 left in "small"
    trimmed = Cube(
        {"counts": np.asarray(cube.layer("counts"))},
        samples=cube.samples.to_dict(),
        groups={"group_id": ["small", "large"]},
        features=cube.features.to_dict(),
        n_cells=cells,
    )
    result = dream_voom(
        trimmed,
        design,
        contrasts={"t1_vs_t0": "timet1", "t2_vs_t0": "timet2"},
        f_tests={"time": ["t1_vs_t0", "t2_vs_t0"]},
        span=0.5,
    )
    reason = result.provenance["groups"][0]["not_estimable"]["time"]
    assert "cannot estimate its contrast 't2_vs_t0'" in reason
    assert result.summary["not_estimable"][0] == "t2_vs_t0, time"
    with pytest.raises(KeyError, match="cannot estimate its contrast"):
        result.table("small", "time")
    assert ("large", "time") in result.tables


def test_invalid_arguments_are_refused_with_what_to_do() -> None:
    cube, design = cube_and_design()
    with pytest.raises(ValueError, match="df_method must be one of 'adaptive'"):
        dream_voom(cube, design, df_method="lme4")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="f_tests must be a mapping"):
        dream_voom(cube, design, f_tests=["t1_vs_t0"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"f_tests\['time'\] must be a list"):
        dream_voom(cube, design, f_tests={"time": "t1_vs_t0"})
    with pytest.raises(ValueError, match="must name at least one contrast"):
        dream_voom(cube, design, f_tests={"time": []})
    with pytest.raises(ValueError, match="'t1_vs_t0' more than once"):
        dream_voom(cube, design, f_tests={"time": ["t1_vs_t0", "t1_vs_t0"]})
    with pytest.raises(ValueError, match="names 't3_vs_t0', which is not among"):
        dream_voom(cube, design, f_tests={"time": ["t1_vs_t0", "t3_vs_t0"]})
    with pytest.raises(ValueError, match="has the name of a contrast"):
        dream_voom(cube, design, f_tests={"t1_vs_t0": ["t1_vs_t0", "t2_vs_t0"]})
    fit, _ = fitted("time_course")
    ml = fit_mixed_model(
        fixture("time_course")[0]["y"],
        Design(
            fixture("time_course")[1]["samples"], "~ time + age", random="(1 | donor)"
        ),
        method="ml",
    )
    with pytest.raises(ValueError, match="defined for REML fits"):
        fixed_effect_tests(ml, df_method="kenward_roger")
    with pytest.raises(ValueError, match="defined for REML fits"):
        fixed_effect_f_test(ml, np.eye(4)[:, 1:3], df_method="kenward_roger")
    with pytest.raises(ValueError, match="the contrasts are all zero"):
        fixed_effect_f_test(fit, np.zeros((4, 2)))
    with pytest.raises(TypeError, match="fit must be a MixedModelFit"):
        fixed_effect_f_test(fit.design, np.eye(4))  # type: ignore[arg-type]


def test_features_without_a_fit_have_nan_kenward_roger_tests() -> None:
    data, meta = fixture("time_course")
    design = Design(meta["samples"], meta["formula"], random=meta["random"])
    y = np.column_stack([data["y"][:, 0], np.full(design.n_samples, 4.0)])
    fit = fit_mixed_model(y, design)
    tests = fixed_effect_tests(fit, df_method="kenward_roger")
    f = fixed_effect_f_test(fit, np.eye(4)[:, 1:3], df_method="kenward_roger")
    assert np.isfinite(tests.df[0]).all() and np.isnan(tests.df[1]).all()
    assert np.isnan(tests.standard_error[1]).all() and np.isnan(f.p_value[1])
    assert np.isfinite(f.p_value[0])


def test_the_f_test_is_a_read_only_record() -> None:
    fit, _ = fitted("unbalanced")
    f = fixed_effect_f_test(
        fit, {"stim": [0, 1, 0], "x": [0, 0, 1]}, df_method="kenward_roger"
    )
    assert isinstance(f, FixedEffectFTest) and f.fit is fit
    assert f.contrast_names == ("stim", "x") and f.kenward_roger_weights == "average"
    assert repr(f) == (
        "FixedEffectFTest(n_features=6, contrast_names=('stim', 'x'), df1=2, "
        "df_method='kenward_roger')"
    )
    for array in (f.f_statistic, f.df2, f.p_value, f.scaling, f.contrasts):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    copy = pickle.loads(pickle.dumps(f))
    np.testing.assert_array_equal(copy.p_value, f.p_value)
    assert fixed_effect_f_test(fit, {"stim": [0, 1, 0]}).kenward_roger_weights is None
