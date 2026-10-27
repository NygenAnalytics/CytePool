"""Tests of what a differential-expression result keeps and computes again.

A result keeps the cube and design it analysed, and ``fit`` computes one group's fit
again. Each fit is compared, bit for bit, with the tables the analysis wrote, and with
the steps of limma-voom run by hand on the group's counts (``filter_by_expression``,
``tmm_factors``, ``voom``, ``fit_linear_model``, ``contrasts_fit`` and
``moderated_statistics``), the building blocks that the validation scripts check
against edgeR and limma. The cube is the fixture of the differential-expression tests,
whose NK cells leave out both samples of donor D1, the reference level, so that the
group's design is rank deficient. Also here: the tables of a result by default keys,
``ModeratedFit.table`` against the tables of ``limma_voom``, and the trend figures of a
result's group against those of the building blocks.
"""

import pickle
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from test_differential_expression import cells_cube, paired_design, voom_result

from cytepool import (
    Design,
    DifferentialExpressionResult,
    GroupFit,
    dream_voom,
    from_bulk,
    limma_trend,
    limma_voom,
)
from cytepool.design import Contrasts
from cytepool.design.checks import _aliasing
from cytepool.normalization import filter_by_expression, log2_cpm, tmm_factors
from cytepool.stats.empirical_bayes import ModeratedFit, moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.voom import voom

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from cytepool.plotting import (  # noqa: E402
    plot_mean_variance,
    plot_residual_variances,
)

LEVELS = {"condition": ["ctrl", "stim"]}


@cache
def trend_result() -> DifferentialExpressionResult:
    """limma-trend with prior weights from the cells, robust."""
    cube = cells_cube()
    return limma_trend(
        cube, paired_design(cube.samples), prior_weights="n_cells", robust=True
    )


@cache
def quality_result() -> DifferentialExpressionResult:
    cube = cells_cube()
    return limma_voom(cube, paired_design(cube.samples), quality_weights=True)


@cache
def dream_result() -> DifferentialExpressionResult:
    cube = cells_cube()
    design = Design(
        cube.samples,
        "~ condition",
        random="(1 | donor)",
        levels=LEVELS,
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    return dream_voom(cube, design, span=0.5)


RESULTS = {
    "voom": voom_result,
    "voom, quality weights": quality_result,
    "trend": trend_result,
    "dream": dream_result,
}


def axes() -> Any:
    return Figure().subplots()


def used_rows(result: DifferentialExpressionResult, group: str) -> np.ndarray:
    """The cube's positions of the samples a group's fit used."""
    used = result.samples.filter(group_id=group, used=True)["sample_id"].tolist()
    ids = result.cube.samples["sample_id"].tolist()  # type: ignore[union-attr]
    return np.array([ids.index(sample) for sample in used], dtype=np.intp)


# What a result keeps.


def test_a_result_keeps_the_cube_and_design_it_analysed() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples)
    result = limma_voom(cube, design, groups="T")

    assert result.cube is cube and result.design is design
    restored = pickle.loads(pickle.dumps(result))
    assert restored.cube.fingerprint() == cube.fingerprint()
    assert restored.design.formula == design.formula
    assert_array_equal(restored.fit().moderated.t, result.fit().moderated.t)
    built = DifferentialExpressionResult(
        result.method,
        result.contrasts,
        result.summary,
        result.samples,
        result.tables,
        result.provenance,
    )
    assert built.cube is None and built.design is None
    with pytest.raises(ValueError, match="keeps no cube and design"):
        built.fit("T")
    with pytest.raises(ValueError, match="both the cube and the design"):
        DifferentialExpressionResult(
            result.method,
            result.contrasts,
            result.summary,
            result.samples,
            result.tables,
            result.provenance,
            cube=cube,
        )
    with pytest.raises(TypeError, match="design must be a Design"):
        DifferentialExpressionResult(
            result.method,
            result.contrasts,
            result.summary,
            result.samples,
            result.tables,
            result.provenance,
            cube=cube,
            design=design.matrix,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("name", list(RESULTS))
def test_a_group_fitted_again_gives_the_tables_of_the_analysis(name: str) -> None:
    result = RESULTS[name]()
    for group in result.groups:
        fit = result.fit(group)
        moderated = fit.moderated
        names = (
            moderated.tests.contrast_names
            if result.method == "dream_voom"
            else moderated.fit.coefficient_names
        )
        for j, contrast in enumerate(names):
            table = result.table(group, contrast)
            assert_array_equal(table["feature_id"], fit.features["feature_id"])
            estimate = (
                moderated.tests.estimate
                if result.method == "dream_voom"
                else moderated.fit.coefficients
            )
            assert_array_equal(table["log2_fold_change"], estimate[:, j])
            assert_array_equal(table["t"], moderated.t[:, j])
            assert_array_equal(table["p_value"], moderated.p_value[:, j])
            assert_array_equal(table["log_odds"], moderated.log_odds[:, j])
        # The samples fitted are those the result says it used, with their numbers.
        used = result.samples.filter(group_id=group, used=True)
        assert fit.samples["sample_id"].tolist() == used["sample_id"].tolist()
        for column in ("n_cells", "library_size", "norm_factor"):
            assert_array_equal(fit.samples[column], used[column])
        for column in ("prior_weight", "quality_weight"):
            assert (column in fit.samples) == (column in used)
            if column in used:
                assert_array_equal(fit.samples[column], used[column])
        assert fit.log2_cpm.shape == (len(used), len(fit.features))
        assert not fit.log2_cpm.flags.writeable
    assert repr(fit).startswith(f"GroupFit(group={fit.group!r}, n_samples=")


def test_a_fit_is_limma_voom_by_hand_in_a_rank_deficient_group() -> None:
    result = voom_result()
    cube, design = cells_cube(), result.design
    for group in ("T", "NK"):
        rows = used_rows(result, group)
        y = np.asarray(cube.layer("counts", group), dtype=np.float64)[rows]
        x = design.matrix[rows]
        kept = list(_aliasing(x).kept)
        x = np.ascontiguousarray(x[:, kept])
        keep = filter_by_expression(y, design=x)
        y = np.ascontiguousarray(y[:, keep])
        weighted = voom(y, x, norm_factors=tmm_factors(y))
        columns = tuple(design.columns[j] for j in kept)
        model = fit_linear_model(
            weighted.log2_cpm, x, weights=weighted.weights, coefficient_names=columns
        )
        estimable = [
            p for p, name in enumerate(design.contrasts.names)
            if name not in result.provenance["groups"][
                result.groups.index(group)
            ]["not_estimable"]
        ]  # fmt: skip
        chosen = Contrasts(
            design.contrasts.matrix[np.ix_(kept, estimable)],
            tuple(design.contrasts.names[p] for p in estimable),
            columns,
        )
        moderated = moderated_statistics(contrasts_fit(model, chosen, method="limma"))

        fit = result.fit(group)
        assert isinstance(fit, GroupFit) and fit.group == group
        assert_array_equal(fit.log2_cpm, weighted.log2_cpm)
        assert_array_equal(fit.weights, weighted.weights)
        assert fit.voom.span == weighted.span
        assert_array_equal(fit.voom.sqrt_sd_trend, weighted.sqrt_sd_trend)
        assert isinstance(fit.moderated, ModeratedFit)
        assert fit.moderated.fit.coefficient_names == chosen.names
        assert_array_equal(fit.moderated.t, moderated.t)
        assert_array_equal(
            fit.moderated.posterior_variance, moderated.posterior_variance
        )
        assert_array_equal(
            fit.features["feature_id"], cube.features["feature_id"][keep]
        )
    # In NK, the reference donor's samples were left out: D2_vs_D1 is not tested.
    assert result.fit("NK").moderated.fit.coefficient_names == (
        "stim_vs_ctrl",
        "D2_vs_D4",
    )


def test_limma_trend_fits_log2_cpm_with_the_sample_weights() -> None:
    result = trend_result()
    fit = result.fit("T")
    rows = used_rows(result, "T")
    cube = cells_cube()
    y = np.asarray(cube.layer("counts", "T"), dtype=np.float64)[rows]
    x = result.design.matrix[rows]
    y = y[:, filter_by_expression(y, design=x)]
    assert_array_equal(
        fit.log2_cpm, log2_cpm(y, norm_factors=tmm_factors(y), prior_count=3.0)
    )
    assert fit.voom is None
    assert_array_equal(fit.weights, np.asarray(cube.n_cells)[rows, 1].astype(float))
    assert_array_equal(fit.weights, fit.samples["prior_weight"])


def test_the_group_fitted_last_is_kept() -> None:
    result = limma_voom(cells_cube(), paired_design(cells_cube().samples))
    first = result.fit("T")
    assert result.fit("T") is first
    other = result.fit("NK")
    assert other is not first and result.fit("NK") is other
    again = result.fit("T")
    assert again is not first
    assert_array_equal(again.moderated.t, first.moderated.t)


def test_fit_says_which_group_it_cannot_fit() -> None:
    cube = cells_cube()
    result = limma_voom(cube, paired_design(cube.samples), min_cells=20)
    with pytest.raises(
        ValueError, match="the result has 2 groups; name one with group"
    ):
        result.fit()
    with pytest.raises(
        KeyError, match="no group 'B' was analysed; groups are: NK and T"
    ):
        result.fit("B")
    with pytest.raises(TypeError, match="group must be a group id"):
        result.fit(1)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="group 'NK' was not fitted"):
        result.fit("NK")
    one = limma_voom(cube, paired_design(cube.samples), groups="T")
    assert one.fit().group == "T"


# Tables by default keys.


def test_the_table_of_the_only_group_and_contrast_needs_no_keys() -> None:
    cube = cells_cube()
    design = paired_design(cube.samples, contrasts={"stim": "conditionstim"})
    one = limma_voom(cube, design, groups="NK")
    assert one.table().equals(one.table("NK", "stim"))
    assert one.table(contrast="stim").equals(one.table("NK"))
    several = voom_result()
    with pytest.raises(
        ValueError, match="the result has 2 groups; name one with group"
    ):
        several.table(contrast="stim_vs_ctrl")
    with pytest.raises(
        ValueError,
        match="the result has 3 contrasts; name one with contrast",
    ):
        several.table("T")
    # A dream F-test counts among the keys.
    design = Design(
        cube.samples, "~ condition", random="(1 | donor)", levels=LEVELS,
        contrasts={"stim": "conditionstim"},
    )  # fmt: skip
    tested = dream_voom(cube, design, groups="T", f_tests={"both": ["stim"]})
    with pytest.raises(ValueError, match="2 contrasts and F-tests"):
        tested.table()


# The tests of a fit as a table, as limma_voom's.


def test_a_moderated_fit_as_a_table_is_the_table_of_limma_voom() -> None:
    result = voom_result()
    for group in result.groups:
        fit = result.fit(group)
        ids = fit.features["feature_id"]
        for contrast in fit.moderated.fit.coefficient_names:
            mine = fit.moderated.table(contrast, feature_ids=ids)
            theirs = result.table(group, contrast)
            assert mine.columns == (
                "feature_id", "log2_fold_change", "standard_error",
                "average_log2_cpm", "t", "p_value", "adjusted_p_value", "log_odds",
            )  # fmt: skip
            for column in theirs.columns:
                assert_array_equal(mine[column], theirs[column])
    moderated = result.fit("T").moderated
    by_position = moderated.table(1)
    assert by_position.equals(
        moderated.table("D2_vs_D1").with_columns(feature_id=by_position["feature_id"])
    )
    assert by_position["feature_id"][:2].tolist() == ["feature_0", "feature_1"]
    with pytest.raises(ValueError, match="the fit has 3 coefficients or contrasts"):
        moderated.table()
    with pytest.raises(KeyError, match="no coefficient or contrast 'x'"):
        moderated.table("x")
    with pytest.raises(IndexError, match="out of range for 3 coefficients"):
        moderated.table(3)
    with pytest.raises(ValueError, match="one id per feature"):
        moderated.table(0, feature_ids=["a"])
    with pytest.raises(TypeError, match="feature_ids must hold text ids"):
        moderated.table(0, feature_ids=list(range(len(by_position))))


# The trend figures of a result's group.


def drawn(ax: Any, gid: str) -> np.ndarray:
    """The points or the line of an artist, by its gid."""
    for artist in [*ax.collections, *ax.lines]:
        if artist.get_gid() == gid:
            if hasattr(artist, "get_offsets") and not hasattr(artist, "get_xdata"):
                return np.asarray(artist.get_offsets())
            return np.column_stack([artist.get_xdata(), artist.get_ydata()])
    raise AssertionError(f"no artist {gid!r}")


def test_the_trend_figures_of_a_group_are_those_of_its_fit() -> None:
    result = voom_result()
    fit = result.fit("NK")
    by_result = plot_mean_variance(result, group="NK", ax=axes())
    by_hand = plot_mean_variance(fit.voom, ax=axes())
    for gid in ("points", "trend"):
        assert_array_equal(drawn(by_result, gid), drawn(by_hand, gid))
    by_result = plot_residual_variances(result, group="NK", ax=axes())
    by_hand = plot_residual_variances(fit.moderated, ax=axes())
    assert_array_equal(drawn(by_result, "points"), drawn(by_hand, "points"))
    one = trend_result()
    residuals = plot_residual_variances(one, group="T", ax=axes())
    # limma-trend's prior follows the trend: a curve, drawn as a line.
    assert drawn(residuals, "prior").shape[0] == one.fit("T").log2_cpm.shape[1]
    single = from_bulk(cells_cube().layer("counts", "T"), samples=cells_cube().samples)
    bulk = limma_voom(single, paired_design(single.samples))
    assert_array_equal(
        drawn(plot_mean_variance(bulk, ax=axes()), "points"),
        drawn(plot_mean_variance(bulk.fit().voom, ax=axes()), "points"),
    )


def test_the_trend_figures_refuse_what_they_cannot_draw() -> None:
    with pytest.raises(ValueError, match="limma-trend weights no observation"):
        plot_mean_variance(trend_result(), group="T")
    with pytest.raises(ValueError, match="dream moderates those of mixed models"):
        plot_residual_variances(dream_result(), group="T")
    with pytest.raises(ValueError, match="group applies to a differential-expression"):
        plot_mean_variance(voom_result().fit("T").voom, group="T")
    with pytest.raises(ValueError, match="group applies to a differential-expression"):
        plot_residual_variances(voom_result().fit("T").moderated, group="T")
    with pytest.raises(ValueError, match="the result has 2 groups"):
        plot_mean_variance(voom_result())
    with pytest.raises(TypeError, match="result must be a VoomResult, or the"):
        plot_mean_variance(np.ones(3))
    with pytest.raises(TypeError, match="fit must be a ModeratedFit or a TreatFit, or"):
        plot_residual_variances(np.ones(3))
