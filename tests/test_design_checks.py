"""Tests of Design.check on deliberately broken designs and against R's QR.

Each broken design is built so that the finding follows from first principles (for
example, condition is constant within batch, so the condition column is the sum of
two batch columns). The rank, aliased columns, leverages and contrast estimability of
the fixture designs come from R's qr() with its default tolerance of 1e-7, which
lm.fit uses, written by validation/design.py, which also draws the contrasts whose
estimability is known by construction.
"""

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool import Design
from cytepool.design import DesignIssue, DesignReport
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.mixed_models import fit_mixed_model

DATA = Path(__file__).parent / "data" / "design"
IDS = [f"S{i}" for i in range(1, 9)]


def _issue(report: DesignReport, kind: str) -> DesignIssue:
    found = [issue for issue in report.errors + report.warnings if issue.kind == kind]
    assert len(found) == 1, report
    return found[0]


def test_rank_aliasing_leverage_and_estimability_match_r_qr() -> None:
    fixture = json.loads((DATA / "r_model_matrix.json").read_text())
    qr = json.loads((DATA / "r_qr.json").read_text())["cases"]
    samples = dict(fixture["samples"], w=np.array(fixture["samples"]["w"]))
    stress = re.compile(r"big|offset|eps|tiny|huge|below|above")
    deficient = verdicts = 0
    for case in qr:
        levels = fixture["level_orders"].get(case["levels"])
        named = {
            f"c{k}": weights for k, weights in enumerate(case.get("contrasts", []))
        }
        design = Design(
            samples, case["formula"], levels=levels, contrasts=named or None
        )
        report = design.check()
        assert report.rank == case["rank"], case["formula"]
        # R's pivot lists unvisited columns before the ones it moved, so compare
        # the sets; the report lists them in column order.
        assert set(report.aliased) == set(case["aliased"]), case["formula"]
        assert list(report.aliased) == [
            c for c in design.columns if c in report.aliased
        ]
        # A kept column that differs from the others by r of its size carries
        # rounding errors of about eps / r, up to 2e-9 for the stress covariates.
        tolerance = 1e-8 if stress.search(case["formula"]) else 1e-10
        assert np.allclose(report.leverage, case["leverage"], rtol=0, atol=tolerance)
        if named:
            deficient += 1
            flagged = {
                issue.message.split("'")[1]
                for issue in report.errors
                if issue.kind == "contrast_not_estimable"
            }
            # X'u is estimable by construction; a random contrast and the unit
            # contrast of an aliased column are not.
            ours = [name not in flagged for name in named]
            assert ours == [True, False, False], case["formula"]
            verdicts += len(ours)
    assert deficient >= 60 and verdicts == 3 * deficient


@pytest.fixture(scope="module")
def nested_batches() -> dict[str, Any]:
    return {
        "sample_id": IDS,
        "condition": ["ctrl"] * 4 + ["stim"] * 4,
        "batch": ["b1", "b1", "b2", "b2", "b3", "b3", "b4", "b4"],
    }


def test_batch_nested_in_condition(nested_batches: dict[str, Any]) -> None:
    design = Design(
        nested_batches,
        "~ batch + condition",
        contrasts={"stim_vs_ctrl": "conditionstim", "b2_vs_b1": "batchb2"},
    )
    report = design.check()
    assert report.kinds() == (
        "rank_deficient",
        "confounded",
        "contrast_not_estimable",
    )
    assert (report.n_samples, report.n_columns, report.rank) == (8, 5, 4)
    assert report.df_residual == 4 and not report.ok
    assert report.aliased == ("conditionstim",)
    # conditionstim is exactly batchb3 + batchb4.
    assert (
        "conditionstim = batchb3 + batchb4" in _issue(report, "rank_deficient").message
    )
    confounded = _issue(report, "confounded").message
    assert "every level of batch holds a single level of condition" in confounded
    assert "condition 'ctrl': batch 'b1' and 'b2'" in confounded
    assert (
        "the right model has it as a random effect, random='(1 | batch)'" in confounded
    )
    # The message says how that model can be fitted.
    assert confounded.endswith(
        "fit_mixed_model estimates this random intercept by REML; fit_linear_model "
        "fits it with a known correlation, given block= (the batch of each sample) "
        "and correlation=; or pool the samples of each batch with "
        "Cube.collapse_samples('batch')"
    )
    # b2 vs b1 lies within ctrl and stays estimable; stim vs ctrl does not.
    assert "'stim_vs_ctrl' is not estimable" in str(report)
    assert "b2_vs_b1" not in str(report)


def test_the_advice_for_repeated_samples_can_be_followed(
    nested_batches: dict[str, Any],
) -> None:
    # The confounded message advises random='(1 | batch)' and says how to fit it:
    # by REML, or with block= and a known correlation. Both are fitted, not refused.
    design = Design(nested_batches, "~ condition", random="(1 | batch)")
    assert design.check().ok
    y = np.random.default_rng(47).normal(5.0, 1.0, (8, 3))
    mixed = fit_mixed_model(y, design)
    assert mixed.coefficient_names == ("(Intercept)", "conditionstim")
    assert mixed.component_names == ("batch",) and mixed.converged.all()
    fit = fit_linear_model(y, design, block=design.samples["batch"], correlation=0.3)
    assert fit.coefficient_names == ("(Intercept)", "conditionstim")
    with pytest.raises(ValueError, match=r"drop the random effect \(1 \| batch\)"):
        fit_linear_model(y, design)


def test_identical_groupings(nested_batches: dict[str, Any]) -> None:
    samples = dict(nested_batches, run=["r1"] * 4 + ["r2"] * 4)
    report = Design(samples, "~ condition + run").check()
    message = _issue(report, "confounded").message
    assert message.startswith("condition and run group the samples identically")
    assert "runr2 = conditionstim" in _issue(report, "rank_deficient").message


def test_covariate_constant_within_a_factor(nested_batches: dict[str, Any]) -> None:
    samples = dict(nested_batches, age=[30.0, 30, 41, 41, 52, 52, 38, 38])
    report = Design(samples, "~ batch + age").check()
    assert _issue(report, "confounded").message.startswith(
        "age has a single value within every level of batch"
    )
    assert report.aliased == ("age",)


def test_no_residual_degrees_of_freedom() -> None:
    samples = {"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]}
    report = Design(samples, "~ a * b").check()
    assert report.kinds() == ("no_residual_df",)
    assert report.rank == 4 and report.df_residual == 0
    assert "no residual degrees of freedom: 4 samples and rank 4" in (
        _issue(report, "no_residual_df").message
    )


def test_constant_variables_and_empty_levels() -> None:
    samples = {
        "sample_id": IDS[:4],
        "group": ["A", "B", "A", "B"],
        "treated": [True] * 4,
        "dose": [2.5] * 4,
    }
    report = Design(samples, "~ group + treated").check()
    assert _issue(report, "constant_variable").message.startswith(
        "treated is 'TRUE' in every sample"
    )
    assert "treatedTRUE = (Intercept)" in _issue(report, "rank_deficient").message
    report = Design(samples, "~ group + dose").check()
    assert "dose has the same value (2.5) in every sample" in str(report)
    assert "dose = 2.5*(Intercept)" in str(report)
    report = Design(samples, "~ group", levels={"group": ["A", "B", "C"]}).check()
    message = _issue(report, "empty_level").message
    assert message.startswith("group has no samples at level 'C', so groupC is zero")
    assert "groupC is zero in every sample" in _issue(report, "rank_deficient").message


def test_a_factor_with_a_level_per_sample() -> None:
    samples = {"sample_id": IDS[:4], "condition": ["a", "a", "b", "b"]}
    report = Design(samples | {"id": IDS[:4]}, "~ condition + id").check()
    assert "id has a different level for every sample" in str(report)
    message = _issue(report, "one_sample_per_level").message
    # No column of this design groups repeated samples, so none is named.
    assert message.endswith(
        "the right model has that grouping as a random effect, such as "
        "random='(1 | g)' for a grouping column g; fit_mixed_model estimates such "
        "a random intercept by REML; fit_linear_model fits one whose correlation is "
        "known, given block= (the group of each sample) and correlation=; and, when "
        "the conditions compared differ between groups, not within them, "
        "Cube.collapse_samples pools the repeated samples of each group into one"
    )
    assert set(report.kinds()) == {
        "rank_deficient",
        "no_residual_df",
        "one_sample_per_level",
    }


def test_single_sample_levels_and_labels_that_look_missing() -> None:
    samples = {
        "sample_id": IDS[:7],
        "group": ["A", "A", "B", "B", "C", "NA", "NA"],
        "batch": ["x", "y", "x", "y", "x", "y", "x"],
    }
    report = Design(samples, "~ group + batch").check()
    assert report.ok
    lonely = _issue(report, "single_sample_level").message
    assert lonely.startswith("group has a single sample at level 'C' (S5)")
    assert "'NA', which looks like a missing value" in str(report)
    # S5 is fitted exactly, but its single-sample level already explains it.
    assert "unreplicated_sample" not in report.kinds()


def test_samples_fitted_exactly_in_an_interaction() -> None:
    samples = {
        "sample_id": IDS,
        "a": ["x", "x", "x", "y", "y", "y", "x", "y"],
        "b": ["p", "p", "q", "p", "q", "q", "p", "p"],
    }
    report = Design(samples, "~ a * b").check()
    assert report.kinds() == ("unreplicated_sample",)
    assert _issue(report, "unreplicated_sample").message.startswith(
        "the model fits sample S3 (a = x, b = q) exactly (leverage 1)"
    )


def test_numeric_columns_that_look_like_group_labels() -> None:
    samples = {"plate": [1, 1, 2, 2, 3, 3], "age": [31, 31, 40, 52, 47, 38]}
    report = Design(samples, "~ plate + age").check()
    assert report.kinds() == ("numeric_codes",)
    assert report.warnings[0].message.startswith(
        "plate is numeric with the consecutive whole values 1 to 3"
    )
    # Ages per donor, time points and doses are ordinary covariates.
    samples = {
        "age": [30, 30, 41, 41, 52, 52, 38, 38],
        "hours": [0, 0, 6, 6, 24, 24, 24, 0],
        "dose": [0, 10, 100, 1000] * 2,
    }
    assert Design(samples, "~ age + hours + dose").check().warnings == ()


def test_numeric_ids_that_are_not_consecutive() -> None:
    # Kang's donors 101, 107 and 1015, each in both conditions.
    samples = {
        "donor": [101, 101, 107, 107, 1015, 1015],
        "condition": ["ctrl", "stim"] * 3,
    }
    report = Design(samples, "~ donor + condition").check()
    assert report.kinds() == ("numeric_ids",)
    assert report.warnings[0].message.startswith(
        "donor holds whole numbers (3 distinct values, from 101 to 1015), each "
        "shared by several samples, and its name says they are ids"
    )
    assert "convert the column to text" in report.warnings[0].message
    # The names that say ids: by their last word, or a number after an id word.
    for name in ("patient_id", "subjectID", "Batches", "plate2", "run_number", "id"):
        renamed = {name: samples["donor"], "condition": samples["condition"]}
        found = Design(renamed, f"~ {name} + condition").check()
        assert found.kinds() == ("numeric_ids",), name
    # Not ids: names of measurements and counts, the same numbers in a column whose
    # name says nothing of ids, a sample of its own per value, too few values,
    # values that are not whole, and consecutive codes (numeric_codes warns).
    others = {
        "hours": [0, 0, 6, 6, 24, 24],
        "donor_age": [31, 31, 47, 47, 52, 52],
        "n_donors": [3, 3, 5, 5, 9, 9],
        "dose": [101, 101, 107, 107, 1015, 1015],
        "batch": [3, 7, 12, 3, 7, 13],
        "plate": [10, 10, 30, 30, 10, 30],
        "lane": [1.5, 1.5, 2.5, 2.5, 4.5, 4.5],
        "run": [4, 4, 5, 5, 6, 6],
    }
    for name, values in others.items():
        table = {name: values, "condition": samples["condition"]}
        kinds = Design(table, f"~ {name} + condition").check().kinds()
        assert "numeric_ids" not in kinds, name


def test_duplicate_column_names_are_reported_not_raised() -> None:
    # a = "bX" and ab = "X" both give a column named abX.
    samples = {"a": ["bX", "c", "d", "c"], "ab": ["X", "Y", "Y", "X"]}
    design = Design(samples, "~ 0 + ab + a", levels={"a": ["c", "bX", "d"]})
    assert design.columns == ("abX", "abY", "abX", "ad")
    report = design.check()
    assert report.kinds()[0] == "duplicate_columns"
    assert "more than one column is named 'abX'" in report.errors[0].message
    with pytest.raises(ValueError, match="more than one column named 'abX'"):
        design.with_contrasts(["ad"])


def test_a_formula_without_columns() -> None:
    report = Design({"a": ["x", "y"]}, "~ 0").check()
    assert report.kinds() == ("empty_design",)
    assert report.n_columns == 0 and report.df_residual == 2


def test_random_effect_checks() -> None:
    samples = {
        "sample_id": IDS,
        "condition": ["ctrl"] * 4 + ["stim"] * 4,
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
        "site": ["s1"] * 8,
    }
    assert Design(samples, "~ condition", random="(1 | donor)").check().ok
    report = Design(samples, "~ condition + donor", random="(1 | donor)").check()
    assert "random_confounded" in report.kinds()
    report = Design(samples, "~ condition", random="(1 | sample_id)").check()
    assert report.kinds() == ("random_unreplicated",)
    text = str(report)
    assert "every group of the random effect (1 | sample_id) has a single" in " ".join(
        text.split()
    )
    assert max(map(len, text.splitlines())) <= 80
    report = Design(samples, "~ condition", random="(1 | site)").check()
    assert report.kinds() == ("random_single_group",)
    # A slope per donor is absorbed when the fixed effects have one per donor.
    samples = samples | {"x": np.arange(1.0, 9.0)}
    assert Design(samples, "~ condition", random="(0 + x | donor)").check().ok
    report = Design(samples, "~ condition + donor:x", random="(0 + x | donor)").check()
    assert report.kinds() == ("random_confounded",)


def test_a_clean_design_reports_nothing() -> None:
    samples = {
        "sample_id": IDS,
        "condition": ["ctrl", "stim"] * 4,
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
    }
    report = Design(samples, "~ donor + condition", contrasts=["conditionstim"]).check()
    assert report.ok and report.errors == () and report.warnings == ()
    # In lines of at most 80 characters.
    assert str(report).splitlines() == [
        "Design check: 8 samples, 5 columns, rank 5, 3 residual degrees of freedom; no",
        "              problems found.",
    ]


def test_report_text_lists_errors_then_warnings(nested_batches: dict[str, Any]) -> None:
    lines = str(Design(nested_batches, "~ batch + condition").check()).splitlines()
    assert lines[0] == (
        "Design check: 8 samples, 5 columns, rank 4, 4 residual degrees of freedom."
    )
    assert lines[1] == "2 errors:"
    assert lines[2].startswith("  1. [rank_deficient] the design matrix has 5")
    assert any(line.startswith("  2. [confounded] every level") for line in lines)
    assert max(len(line) for line in lines) <= 88
    samples = {"g": ["A", "A", "B", "B", "C"], "x": [1.0, 2.0, 4.0, 3.0, 5.0]}
    lines = str(Design(samples, "~ g + x").check()).splitlines()
    assert lines[1] == "1 warning:"
    assert lines[2].startswith("  1. [single_sample_level] g has a single sample")


def test_leverages_by_hand() -> None:
    # Cell means: 1 over the size of the sample's group.
    report = Design({"g": ["A", "A", "B", "B", "B"]}, "~ g").check()
    assert np.allclose(report.leverage, [1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3])
    # A line: 1/n + (x - mean)^2 / sum of squares, here 5/6, 1/3, 5/6.
    report = Design({"x": [0.0, 1.0, 2.0]}, "~ x").check()
    assert np.allclose(report.leverage, [5 / 6, 1 / 3, 5 / 6])


def test_a_sample_nearly_fitted_exactly_by_an_extreme_covariate() -> None:
    samples = {
        "sample_id": IDS[:5],
        "g": ["A", "A", "B", "B", "B"],
        "x": [1.0, 2.0, 3.0, 4.0, 3000.0],
    }
    report = Design(samples, "~ g + x").check()
    assert report.kinds() == ("high_leverage",)
    # Separate intercepts and a common slope: h = 1/n_g + (x - mean_g)^2 / S,
    # with S the within-group sum of squares of x.
    x = np.array(samples["x"])
    centered = np.concatenate([x[:2] - x[:2].mean(), x[2:] - x[2:].mean()])
    expected = 1 / 3 + centered[4] ** 2 / np.sum(centered**2)
    assert report.leverage[4] == pytest.approx(expected, abs=1e-12)
    assert 0.99 < expected < 1 - 1e-10
    message = report.warnings[0].message
    assert message.startswith(
        f"the model nearly fits S5 (leverage 1 - {1 - expected:.2g};"
    )
    assert "x = 3000, the others 1 to 4" in message


def test_constant_covariate_without_intercept_is_a_warning() -> None:
    samples = {"x": [2.5] * 4, "g": ["A", "B", "A", "B"]}
    report = Design(samples, "~ 0 + x").check()
    assert report.ok and report.kinds() == ("constant_variable",)
    assert "only rescale the means" in report.warnings[0].message
    assert not Design(samples, "~ x").check().ok


def test_near_aliasing_advice_depends_on_the_cause() -> None:
    # A covariate that hardly varies around a large value is aliased with the
    # intercept under R's tolerance; centering it fixes that.
    depth = Design({"depth": 3.2e7 + np.arange(4.0)}, "~ depth").check()
    assert depth.aliased == ("depth",)
    assert "depth hardly varies" in depth.errors[0].message
    assert "centering it (subtracting its mean)" in depth.errors[0].message
    # Two nearly identical covariates cannot be told apart; centering would not help.
    x1 = np.arange(1.0, 6.0)
    x2 = x1 + 1e-9 * np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    pair = Design({"x1": x1, "x2": x2}, "~ x1 + x2").check()
    assert pair.aliased == ("x2",)
    assert "remove or combine the nearly collinear variables" in str(pair.errors[0])


def test_extreme_scales_are_not_aliased() -> None:
    for scale in (1e200, 1e-200):
        report = Design({"x": scale * np.array([1.0, 2.0, 4.0])}, "~ x").check()
        assert report.rank == 2 and report.aliased == ()


def test_labels_that_differ_only_in_case_or_spaces() -> None:
    samples = {"condition": ["ctrl", "ctrl ", "Ctrl", "stim", "stim", "ctrl"]}
    report = Design(samples, "~ condition").check()
    message = _issue(report, "similar_levels").message
    assert "('Ctrl', 'ctrl' and 'ctrl ')" in message
    # A control group called none is not a missing value.
    samples = {"stimulation": ["none", "LPS", "IFNg"] * 2}
    assert Design(samples, "~ stimulation").check().warnings == ()


def test_many_single_sample_levels_are_reported_once() -> None:
    labels = [f"L{i}" for i in range(8)] + ["L9", "L9"]
    report = Design(
        {"g": labels, "sample_id": [f"S{i}" for i in range(10)]}, "~ g"
    ).check()
    assert report.kinds() == ("single_sample_level",)
    assert report.warnings[0].message.endswith(
        "and 2 more; effects of such a level rest on one sample with no replicate. "
        "Add replicates, merge the level with a similar one, or remove the sample"
    )


def test_random_effects_that_lme4_cannot_identify() -> None:
    samples = {
        "condition": ["ctrl", "stim"] * 4,
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
        "donor2": ["E1", "E1", "E2", "E2", "E3", "E3", "E4", "E4"],
    }
    # Intercept and slope for each of 4 donors: 8 random effects for 8 samples.
    report = Design(samples, "~ condition", random="(condition | donor)").check()
    assert report.kinds() == ("random_unreplicated",)
    assert "2 effects in each of 4 groups, 8 random effects in all" in (
        report.errors[0].message
    )
    report = Design(samples, "~ condition", random="(1 | donor) + (1 | donor2)").check()
    assert report.kinds() == ("random_duplicate",)


def _exact_null_space(matrix: np.ndarray) -> list[list[Fraction]]:
    """The null space of a matrix of dyadic numbers, in exact rational arithmetic."""
    rows = [[Fraction(float(value)) for value in row] for row in matrix]
    n, p = len(rows), len(rows[0])
    pivots: list[int] = []
    rank = 0
    for column in range(p):
        pivot = next((i for i in range(rank, n) if rows[i][column] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        rows[rank] = [value / rows[rank][column] for value in rows[rank]]
        for i in range(n):
            if i != rank and rows[i][column] != 0:
                factor = rows[i][column]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[rank])]
        pivots.append(column)
        rank += 1
    basis = []
    for free in (column for column in range(p) if column not in pivots):
        vector = [Fraction(0)] * p
        vector[free] = Fraction(1)
        for i, column in enumerate(pivots):
            vector[column] = -rows[i][free]
        basis.append(vector)
    return basis


def test_estimability_matches_exact_arithmetic_in_any_units() -> None:
    # Dyadic designs with exact aliasing (batch nested in condition, x2 = 2x + 3u,
    # an unused level) and covariates in units from 2^-40 to 2^40: the verdict on
    # unit, difference and mixed contrasts must equal the exact rational one.
    rng = np.random.default_rng(3)
    condition = ["ctrl"] * 6 + ["stim"] * 6
    batch = ["b1", "b1", "b2", "b2", "b2", "b1", "b3", "b3", "b4", "b4", "b3", "b4"]
    checked = 0
    for unit in (2.0**-40, 1.0, 2.0**40):
        x = unit * rng.integers(-40, 41, 12) / 8
        samples = {
            "condition": condition,
            "batch": batch,
            "x": x,
            "x2": 2 * x + 3 * unit,
        }
        samples["g"] = ["A", "B", "C"] * 4
        for formula in (
            "~ batch + condition + x",
            "~ g + x + x2",
            "~ condition*x + batch",
        ):
            design = Design(samples, formula, levels={"g": ["A", "B", "C", "D"]})
            null = _exact_null_space(design.matrix)
            p = design.matrix.shape[1]
            eye = np.eye(p)
            candidates = {f"e{k}": eye[k] for k in range(p)}
            candidates |= {f"e{k}-e{k + 1}": eye[k] - eye[k + 1] for k in range(p - 1)}
            candidates |= {f"e{k}+e0/4": eye[k] + eye[0] / 4 for k in range(1, p)}
            report = design.with_contrasts(candidates).check()
            flagged = {
                issue.message.split("'")[1]
                for issue in report.errors
                if issue.kind == "contrast_not_estimable"
            }
            for name, weights in candidates.items():
                exact = [Fraction(float(w)) for w in weights]
                estimable = all(sum(a * b for a, b in zip(exact, v)) == 0 for v in null)
                assert (name not in flagged) == estimable, (formula, unit, name)
                checked += 1
    assert checked > 150


def test_a_small_non_estimable_part_is_not_masked() -> None:
    # reads_m is reads in other units, so reads and reads_m cannot be separated: a
    # contrast that touches reads is not estimable however large its other weights.
    reads = 2.0**20 * np.array([24, 31, 18, 40, 27, 35, 22, 29])
    samples = {"g": ["A", "B"] * 4, "reads": reads, "reads_m": reads / 2**20}
    design = Design(samples, "~ g + reads + reads_m")
    report = design.with_contrasts(
        {"mixed": "gB + reads", "reads": "reads", "group": "gB"}
    ).check()
    flagged = [
        issue.message.split("'")[1]
        for issue in report.errors
        if issue.kind == "contrast_not_estimable"
    ]
    assert report.aliased == ("reads_m",)
    assert flagged == ["mixed", "reads"]


def test_double_bar_terms_count_one_effect_per_term() -> None:
    samples = {
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
        "x": [0.5, 1.5, 2.0, 0.25, 1.0, 3.0, 2.5, 0.75],
    }
    # lme4 splits (x || donor) into (1 | donor) and (0 + x | donor): 4 random
    # effects each for 8 samples, which it fits; (x | donor) has 8 for 8.
    assert Design(samples, "~ x", random="(x || donor)").check().ok
    report = Design(samples, "~ x", random="(x | donor)").check()
    assert report.kinds() == ("random_unreplicated",)


def test_a_random_intercept_absorbed_by_fixed_effects() -> None:
    samples = {
        "donor": ["D1", "D1", "D1", "D2", "D2", "D2", "D3", "D3", "D3"],
        "x": [0.0, 1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 4.0],
    }
    report = Design(samples, "~ donor + x", random="(x | donor)").check()
    assert report.kinds() == ("random_confounded",)
    message = report.errors[0].message
    assert "for the random intercept of the random effect (x | donor)" in message
    assert "write 0 + on its left side to drop the intercept" in message
    assert Design(samples, "~ donor + x", random="(0 + x | donor)").check().ok


def test_high_leverage_from_few_residual_degrees_of_freedom() -> None:
    # Three donors, two conditions and a covariate leave 1 residual degree of
    # freedom; no RIN value is far from the others, so none is blamed.
    samples = {
        "sample_id": IDS[:6],
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3"],
        "condition": ["ctrl", "stim"] * 3,
        "rin": [7.0, 7.3, 6.7, 6.1, 6.5, 6.0],
    }
    report = Design(samples, "~ donor + condition + rin").check()
    assert report.kinds() == ("high_leverage",) and report.df_residual == 1
    message = report.warnings[0].message
    assert message.startswith("the model nearly fits S1 (leverage 0.99")
    assert "with 1 residual degree of freedom" in message
    assert "rin =" not in message and "log scale" not in message
