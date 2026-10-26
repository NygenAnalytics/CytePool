"""Tests for the figures of cytepool.plotting.

The figures are drawn on the Agg canvas of matplotlib Figures that pyplot does not
manage, and the data of the drawn artists, found by their gid, are compared with:

- what R draws for the same figure on the same data, saved by
  ``validation/plotting.py`` in ``tests/data/plotting/``: limma's ``plotMDS``,
  ``plotSA``, ``plotMD``, ``volcanoplot`` and ``barcodeplot``, voom's saved plot, the
  bars of ``voomWithQualityWeights``, ``hist``, DESeq2's ``plotDispEsts``, edgeR's
  ``plotQLDisp``, metafor's ``forest``, and speckle's ``plotCellTypeProps`` and
  ``propeller``;
- values derived by hand: leading log2 fold changes of three samples, counts of
  p-values per bin, the impulse response of limma's tricube moving average as R
  printed it;
- invariants that a wrong figure would break: bars that sum to one, cell numbers and
  sample weights read back from the result they came from, labels on their points,
  and colours that stay apart under simulated colour-vision deficiencies.

Errors are checked for their messages, and drawing changes no setting of matplotlib.
"""

import ast
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import plotting
from cytepool.composition import propeller
from cytepool.cube import Cube, from_bulk
from cytepool.design import Design
from cytepool.differential_expression import limma_voom
from cytepool.meta_analysis import differential_expression_meta_analysis
from cytepool.normalization import log_cpm
from cytepool.stats.deseq2 import deseq2_wald_test, fit_deseq2
from cytepool.stats.empirical_bayes import moderated_statistics, treat
from cytepool.stats.gene_set_tests import camera
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.quasi_likelihood import (
    fit_quasi_likelihood,
    quasi_likelihood_f_test,
)
from cytepool.stats.voom import voom, voom_with_quality_weights
from cytepool.table import Table

matplotlib = pytest.importorskip("matplotlib")
FIGURE = pytest.importorskip("matplotlib.figure")
COLORS = pytest.importorskip("matplotlib.colors")

DATA = Path(__file__).parent / "data" / "plotting"
EXACT = 1e-10
# Robust moderation, sample weights and REML are iterative fits.
CLOSE = 1e-6
GENES = [f"g{j:03d}" for j in range(300)]
SAMPLES = {
    "sample_id": [f"S{i}" for i in range(1, 9)],
    "condition": ["ctrl"] * 4 + ["stim"] * 4,
}
LEVELS = {"condition": ["ctrl", "stim"]}
ACROSS = "adjusted_p_value_across_groups"


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def axes() -> Any:
    return FIGURE.Figure().subplots()


def artists(ax: Any, gid: str) -> list[Any]:
    return [artist for artist in ax.get_children() if artist.get_gid() == gid]


def pairs(x: Any, y: Any) -> np.ndarray:
    """Points as rows (x, y), sorted by x and then y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    order = np.lexsort((y, x))
    return np.column_stack([x[order], y[order]])


def points(ax: Any, *gids: str) -> np.ndarray:
    offsets = [c.get_offsets() for gid in gids for c in artists(ax, gid)]
    stacked = np.concatenate(offsets) if offsets else np.empty((0, 2))
    return pairs(stacked[:, 0], stacked[:, 1])


def scaled(ours: Any, reference: Any) -> float:
    """Largest difference over max(|reference|, 1), as validation/plotting.py has it."""
    ours, reference = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    assert ours.shape == reference.shape
    if ours.size == 0:
        return 0.0
    return float(np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1.0)))


@cache
def design() -> Design:
    return Design(
        SAMPLES, "~ condition", levels=LEVELS, contrasts={"stim": "conditionstim"}
    )


@cache
def counts() -> np.ndarray:
    return fixture("limma")["counts"]


@cache
def table() -> Table:
    cube = from_bulk(counts(), samples=SAMPLES, features=GENES)
    return limma_voom(cube, design()).table("bulk", "stim")


def bulk_cube(n_cells: Any = None) -> Cube:
    return Cube(
        {"counts": counts()[np.newaxis].astype(np.uint32)},
        samples=SAMPLES,
        groups={"group_id": ["T"]},
        features={"feature_id": GENES},
        n_cells=n_cells,
    )


# The figures R draws for the same data.


@pytest.mark.parametrize("case", ["mds1", "mds2", "mds3", "mds_cube"])
def test_mds_coordinates_are_those_of_plotMDS_up_to_sign(case: str) -> None:
    reference = fixture("limma")
    options = {
        "mds1": {"top": 50},
        "mds2": {"top": 500},
        "mds3": {"top": 100, "gene_selection": "common", "dimensions": (2, 3)},
    }
    if case == "mds_cube":
        n_cells = np.where(np.arange(8) < 6, 50, 5)[:, np.newaxis]
        ax = plotting.plot_mds(bulk_cube(n_cells), labels=True, ax=axes())
        # The samples with fewer than 10 cells are left out.
        assert [t.get_text() for t in artists(ax, "sample labels")] == [
            f"S{i}" for i in range(1, 7)
        ]
    else:
        ax = plotting.plot_mds(log_cpm(counts()), ax=axes(), **options[case])
    ours = artists(ax, "samples")[0].get_offsets()
    theirs = reference[f"{case} points"]
    signs = np.sign(np.sum(ours * theirs, axis=0))
    assert scaled(ours * signs, theirs) <= EXACT
    shares = [ax.get_xlabel().rsplit("(", 1)[1], ax.get_ylabel().rsplit("(", 1)[1]]
    assert shares == list(reference[f"{case} percentages"])


def test_mean_variance_trend_is_voom_saved_plot() -> None:
    reference = fixture("limma")
    ax = plotting.plot_mean_variance(voom(counts(), design()), ax=axes())

    assert scaled(artists(ax, "points")[0].get_offsets(), reference["voom points"]) <= (
        EXACT
    )
    assert scaled(artists(ax, "trend")[0].get_xydata(), reference["voom line"]) <= EXACT


def test_sample_weights_are_the_bars_of_voomWithQualityWeights() -> None:
    result = voom_with_quality_weights(counts(), design())
    ax = plotting.plot_sample_weights(result, ax=axes())
    heights = [patch.get_height() for patch in artists(ax, "weights")]

    assert scaled(heights, fixture("limma")["weights"]) <= CLOSE
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        str(i) for i in range(1, 9)
    ]
    assert artists(ax, "equal weight")[0].get_ydata()[0] == 1.0


@pytest.mark.parametrize("case", ["sa_robust", "sa_plain"])
def test_residual_variances_are_plotSA(case: str) -> None:
    reference = fixture("limma")
    values = log_cpm(counts())
    options: dict[str, Any] = {}
    if case == "sa_robust":
        values[:, 10:16] += reference["noise"]
        options = {"trend": True, "robust": True}
    fit = moderated_statistics(fit_linear_model(values, design()), **options)
    ax = plotting.plot_residual_variances(fit, ax=axes())

    assert (
        scaled(points(ax, "points", "outliers"), reference[f"{case} points"]) <= CLOSE
    )
    # The variances plotSA marks as outliers, in both directions.
    assert scaled(points(ax, "outliers"), reference[f"{case} outliers"]) <= CLOSE
    prior = artists(ax, "prior")[0].get_xydata()
    if case == "sa_plain":
        prior = prior[:, 1]  # a horizontal line
    assert scaled(prior, reference[f"{case} prior"]) <= CLOSE


def test_md_volcano_and_p_values_are_those_limma_draws() -> None:
    reference = fixture("limma")
    ax = plotting.plot_md(table(), ax=axes())
    for gid in ("up", "down", "not significant"):
        assert scaled(points(ax, gid), reference[f"md {gid}"]) <= EXACT
    ax = plotting.plot_volcano(table(), ax=axes())
    drawn = points(ax, "up", "down", "not significant")
    assert scaled(drawn, reference["volcano points"]) <= EXACT
    labels = sorted(text.get_text() for text in artists(ax, "gene labels"))
    assert labels == list(reference["volcano labels"])
    ax = plotting.plot_p_values(table(), ax=axes())
    heights = [patch.get_height() for patch in artists(ax, "p-values")]
    assert_array_equal(heights, reference["histogram"])


@pytest.mark.parametrize("case", ["barcode", "barcode_span"])
def test_barcode_bars_and_worm_are_barcodeplot(case: str) -> None:
    reference = fixture("limma")
    t = reference["t"]
    if case == "barcode":
        ax = plotting.plot_barcode(t, reference["members"], ax=axes())
    else:
        ax = plotting.plot_barcode(
            t[:60], reference["members_span"], span=1.0, ax=axes()
        )
    bars = np.sort([s[0, 0] for s in artists(ax, "set")[0].get_segments()])

    assert_array_equal(bars, reference[f"{case} bars"])
    worm = artists(ax, "worm")[0].get_ydata()
    assert scaled(worm, reference[f"{case} worm"]) <= EXACT


def test_dispersions_draw_DESeq2s_estimates_as_plotDispEsts() -> None:
    reference = fixture("count_models")
    estimates = fit_deseq2(counts(), design()).dispersion_estimates
    assert estimates is not None
    given = replace(
        estimates,
        base_mean=reference["deseq2 base_mean"],
        genewise=reference["deseq2 genewise"],
        trended=reference["deseq2 trended"],
        dispersion=reference["deseq2 dispersion"],
        outlier=reference["deseq2 outlier"] == 1.0,
        all_zero=reference["deseq2 all_zero"] == 1.0,
    )
    ax = plotting.plot_dispersions(given, ax=axes())

    assert scaled(points(ax, "gene-wise"), reference["deseq2 drawn gene-wise"]) <= EXACT
    final = points(ax, "final", "outliers")
    assert scaled(final, reference["deseq2 drawn final"]) <= EXACT
    # Outliers keep their gene-wise estimate, drawn as open circles.
    assert scaled(points(ax, "outliers"), reference["deseq2 drawn outliers"]) <= EXACT
    assert len(points(ax, "outliers")) == 4
    assert artists(ax, "outliers")[0].get_facecolors().size == 0
    fitted = artists(ax, "fitted")[0].get_xydata()
    assert scaled(fitted, reference["deseq2 drawn fitted"]) <= EXACT


def test_quasi_dispersions_draw_edgeRs_estimates_as_plotQLDisp() -> None:
    reference = fixture("count_models")
    fit = fit_quasi_likelihood(
        counts(), design(), norm_factors=fixture("limma")["norm_factors"]
    )
    with np.errstate(invalid="ignore"):
        quasi = reference["ql deviance"] / reference["ql df"]
    given = replace(
        fit,
        covariate=reference["ql covariate"],
        quasi_dispersion=quasi,
        posterior_quasi_dispersion=reference["ql posterior"],
        prior_quasi_dispersion=reference["ql prior"],
    )
    ax = plotting.plot_quasi_dispersions(given, ax=axes())

    # edgeR also draws the two genes without counts at 0; they have no
    # quasi-dispersion here.
    assert scaled(points(ax, "raw"), reference["ql drawn raw"]) <= EXACT
    assert_array_equal(reference["ql drawn raw without df"], [0.0, 0.0])
    assert scaled(points(ax, "squeezed"), reference["ql drawn squeezed"]) <= EXACT
    trend = artists(ax, "prior")[0].get_xydata()
    assert scaled(trend, reference["ql drawn trend"]) <= EXACT


@cache
def studies() -> dict[str, Any]:
    """The three studies of the forest plots, as validation/plotting.py tests them."""
    data = fixture("metafor")
    results = {}
    for name in ("A", "B", "C"):
        values = data[f"forest counts {name}"]
        donors = [f"{name}{d}" for d in range(values.shape[0] // 2)]
        samples = {
            "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
            "donor": [d for d in donors for _ in range(2)],
            "condition": ["ctrl", "stim"] * len(donors),
        }
        genes = [f"g{j:03d}" for j in range(values.shape[1])]
        cube = from_bulk(values, samples=samples, features=genes)
        study_design = Design(
            cube.samples,
            "~ donor + condition",
            levels=LEVELS,
            contrasts={"stim": "conditionstim"},
        )
        results[name] = limma_voom(cube, study_design)
    return results


@pytest.mark.parametrize(("method", "test"), [("reml", "z"), ("dl", "knapp_hartung")])
def test_forest_intervals_and_pooled_estimate_are_metafors(
    method: str, test: str
) -> None:
    data = fixture("metafor")
    meta = differential_expression_meta_analysis(studies(), method=method, test=test)
    for feature in data["forest features"].tolist():
        ax = plotting.plot_forest(meta, studies(), feature, ax=axes())
        segments = sorted(
            artists(ax, "intervals")[0].get_segments(), key=lambda s: -s[0, 1]
        )
        key = f"forest {method} {feature}"
        tolerance = CLOSE if method == "reml" else EXACT
        assert (
            scaled([s[:, 0] for s in segments], data[f"{key} intervals"]) <= tolerance
        )
        diamond = artists(ax, "pooled")[0].get_xy()[:4]
        assert scaled(diamond[:3, 0], data[f"{key} pooled"]) <= tolerance
        # The diamond's top and bottom are at the estimate.
        assert diamond[3, 0] == diamond[1, 0]


def test_a_study_that_did_not_test_the_gene_keeps_its_row() -> None:
    data = fixture("metafor")
    meta = differential_expression_meta_analysis(studies())
    feature = str(data["forest features"][2])  # tested by two studies
    ax = plotting.plot_forest(meta, studies(), feature, ax=axes())

    names = [label.get_text() for label in ax.get_yticklabels()]
    assert names[:3] == ["A", "B", "C"]
    assert names[3].startswith("Pooled, REML (I² ")
    texts = [text.get_text() for text in ax.texts]
    assert texts.count("not tested") == 1
    assert len(artists(ax, "intervals")[0].get_segments()) == 2
    # The line of no effect is in view, beside every interval.
    low, high = ax.get_xlim()
    ends = [s[:, 0] for s in artists(ax, "intervals")[0].get_segments()]
    assert low < min(0.0, np.min(ends)) and high > max(0.0, np.max(ends))


def composition_cube() -> Cube:
    cells = fixture("speckle")["speckle cells"]
    return Cube(
        {"counts": (cells.T[:, :, np.newaxis] > 0).astype(np.uint32)},
        samples={
            "sample_id": [f"S{i:02d}" for i in range(1, 13)],
            "condition": ["ctrl"] * 6 + ["stim"] * 6,
        },
        groups={"group_id": ["T", "B", "NK", "Mono", "DC"]},
        features={"feature_id": ["x"]},
        n_cells=cells,
    )


def test_composition_bars_are_those_of_plotCellTypeProps() -> None:
    reference = fixture("speckle")
    ax = plotting.plot_composition(composition_cube(), by="condition", ax=axes())
    samples = [label.get_text() for label in ax.get_xticklabels()]
    heights = {
        (samples[i], container.get_label()): patch.get_height()
        for container in ax.containers
        for i, patch in enumerate(container)
    }
    keys = zip(
        reference["speckle bar_sample"].tolist(),
        reference["speckle bar_cluster"].tolist(),
        strict=True,
    )

    assert scaled([heights[key] for key in keys], reference["speckle bar_height"]) <= (
        EXACT
    )


def test_proportions_means_and_adjusted_p_values_are_propellers() -> None:
    reference = fixture("speckle")
    cube = composition_cube()
    result = propeller(
        cube,
        Design(
            cube.samples,
            "~ condition",
            levels=LEVELS,
            contrasts={"stim": "conditionstim"},
        ),
    )
    ax = plotting.plot_proportions(result, by="condition", ax=axes())
    groups = cube.groups["group_id"].tolist()
    order = [reference["speckle table_clusters"].tolist().index(g) for g in groups]
    means = [line.get_ydata()[0] for line in artists(ax, "means")]
    expected = np.concatenate(
        [reference["speckle mean_ctrl"][order], reference["speckle mean_stim"][order]]
    )

    assert scaled(means, expected) <= EXACT
    clusters = reference["speckle clusters"].tolist()
    proportions = reference["speckle proportions"]
    by_level = [
        [proportions[clusters.index(g), s] for g in groups for s in samples]
        for samples in (range(6), range(6, 12))
    ]
    drawn = [c.get_offsets()[:, 1] for c in artists(ax, "proportions")]
    assert scaled(np.concatenate(drawn), np.concatenate(by_level)) <= EXACT
    texts = [text.get_text() for text in artists(ax, "adjusted p-values")]
    assert texts == [f"adj. p {reference['speckle fdr'][i]:.2g}" for i in order]


# Values derived by hand, and invariants.


def test_mds_distances_are_leading_log2_fold_changes() -> None:
    # Squared differences between the samples, gene by gene: (1, 4, 0, 0) for the
    # first two, (0, 0, 9, 1) for the first and third, (1, 4, 9, 1) for the last two;
    # the means of the two largest are 2.5, 5 and 6.5, which three points place
    # exactly in two dimensions.
    values = [[0.0, 0, 0, 0], [1, 2, 0, 0], [0, 0, 3, 1]]
    ax = plotting.plot_mds(values, top=2, ax=axes())
    p = artists(ax, "samples")[0].get_offsets()
    squared = [np.sum((p[i] - p[j]) ** 2) for i, j in ((0, 1), (0, 2), (1, 2))]

    assert_allclose(squared, [2.5, 5.0, 6.5], rtol=1e-12)
    # Each dimension's largest coordinate is positive.
    assert np.all(p[np.argmax(np.abs(p), axis=0), [0, 1]] > 0)
    assert ax.get_xlabel().startswith("Leading log2 fold change, dimension 1 (")
    # With every gene, the plot is a principal component plot: the same distances.
    ax = plotting.plot_mds(values, gene_selection="common", ax=axes())
    assert ax.get_xlabel().startswith("Principal component 1 (")


def test_mds_colours_and_markers_follow_the_sample_variables() -> None:
    cube = Cube(
        {"counts": np.tile(counts()[:, np.newaxis, :], (1, 1, 1)).swapaxes(0, 1)},
        samples={
            **SAMPLES,
            "donor": ["D1", "D2", "D3", "D4"] * 2,
        },
        groups={"group_id": ["bulk"]},
        features={"feature_id": GENES},
    )
    ax = plotting.plot_mds(cube, color_by="condition", marker_by="donor", ax=axes())
    collections = artists(ax, "samples")
    assert len(collections) == 4  # one per marker, of the donors
    assert {len(c.get_offsets()) for c in collections} == {2}
    for collection in collections:  # each donor has a control and a stimulated sample
        assert len({tuple(color) for color in collection.get_facecolors()}) == 2
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["condition", "ctrl", "stim", "donor", "D1", "D2", "D3", "D4"]
    # More than three levels of colour alone get a marker each too.
    ax = plotting.plot_mds(cube, color_by="donor", labels=True, ax=axes())
    assert len(artists(ax, "samples")) == 4
    for label in artists(ax, "sample labels"):
        position = label.xy
        drawn = np.concatenate([c.get_offsets() for c in artists(ax, "samples")])
        assert np.min(np.hypot(*(drawn - position).T)) == 0.0


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"data": [[0.0, 1.0], [1.0, 0.0]]}, ValueError, "need at least 3 samples"),
        ({"data": [[0.0, np.nan], [1, 0], [2, 1]]}, ValueError, "must be finite"),
        ({"data": [0.0, 1.0, 2.0]}, ValueError, "2-D array"),
        ({"group": "T"}, ValueError, "group applies to a cube"),
        ({"min_cells": 3}, ValueError, "min_cells applies to a cube"),
        ({"color_by": ["a", "b"]}, ValueError, r"one label per sample \(3\)"),
        ({"color_by": ["a", None, "b"]}, ValueError, "missing label"),
        ({"color_by": "condition"}, ValueError, "names a variable"),
        ({"top": 0}, ValueError, "top must be at least 1"),
        ({"gene_selection": "all"}, ValueError, "gene_selection must be one of"),
        ({"dimensions": (1, 1)}, ValueError, "two different dimensions"),
        ({"dimensions": 2}, TypeError, "pair of dimensions"),
        ({"ax": "axes"}, TypeError, "ax must be a matplotlib Axes"),
    ],
)
def test_mds_refuses_invalid_arguments(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    call = {"data": [[0.0, 1, 2], [1, 0, 2], [2, 2, 0]]} | arguments
    with pytest.raises(error, match=message):
        plotting.plot_mds(**call)


def test_mds_of_a_cube_needs_the_group_and_known_variables() -> None:
    cube = Cube(
        {"counts": np.repeat(counts()[np.newaxis].astype(np.uint32), 2, axis=0)},
        samples=SAMPLES,
        groups={"group_id": ["T", "B"]},
        features={"feature_id": GENES},
    )
    with pytest.raises(ValueError, match="name the one to plot with group"):
        plotting.plot_mds(cube)
    with pytest.raises(KeyError, match="no variable 'donor'"):
        plotting.plot_mds(cube, group="T", color_by="donor")


def test_mds_refuses_more_levels_than_colours_or_markers() -> None:
    values = np.random.default_rng(1).normal(size=(13, 6))
    levels = [f"L{i}" for i in range(13)]
    with pytest.raises(ValueError, match="more than the 8 colours"):
        plotting.plot_mds(values, color_by=levels)
    with pytest.raises(ValueError, match="more than the 12 markers"):
        plotting.plot_mds(values, marker_by=levels)
    # Labels name any number of samples.
    ax = plotting.plot_mds(values, labels=levels, ax=axes())
    assert [t.get_text() for t in artists(ax, "sample labels")] == levels


def test_cell_numbers_are_the_cubes_with_entries_left_out_pale() -> None:
    cells = np.array([[5, 20], [0, 0], [12, 3]])
    layer = np.zeros((2, 3, 1), dtype=np.uint32)
    layer[:, :, 0] = (cells.T > 0).astype(np.uint32)
    cube = Cube(
        {"counts": layer},
        samples={"sample_id": ["a", "b", "c"]},
        groups={"group_id": ["T", "B"]},
        features={"feature_id": ["x"]},
        n_cells=cells,
    )
    ax = plotting.plot_cell_numbers(cube, ax=axes())
    image = artists(ax, "cell numbers")[0].get_array()

    # Groups are rows and samples columns; entries below 10 cells have no shade.
    assert_array_equal(
        np.ma.getmaskarray(image), [[True, True, False], [False, True, True]]
    )
    assert_array_equal(np.ma.getdata(image)[~np.ma.getmaskarray(image)], [12.0, 20.0])
    numbers = [text.get_text() for text in artists(ax, "numbers")]
    assert numbers == ["5", "0", "12", "20", "0", "3"]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b", "c"]
    ax = plotting.plot_cell_numbers(cube, min_cells=None, numbers=False, ax=axes())
    assert artists(ax, "numbers") == []
    assert np.ma.getmaskarray(artists(ax, "cell numbers")[0].get_array()).sum() == 2
    with pytest.raises(ValueError, match="does not record cell numbers"):
        plotting.plot_cell_numbers(bulk_cube())
    with pytest.raises(ValueError, match="min_cells must be at least 0"):
        plotting.plot_cell_numbers(cube, min_cells=-1)
    with pytest.raises(TypeError, match="cube must be a Cube"):
        plotting.plot_cell_numbers(cells)  # type: ignore[arg-type]


def test_composition_bars_sum_to_one_with_the_first_group_on_top() -> None:
    cube = composition_cube()
    ax = plotting.plot_composition(cube, ax=axes())
    containers = ax.containers
    heights = np.array([[p.get_height() for p in c] for c in containers])
    bottoms = np.array([[p.get_y() for p in c] for c in containers])

    assert_allclose(heights.sum(axis=0), 1.0, rtol=1e-15)
    assert_allclose(bottoms[0] + heights[0], 1.0, rtol=1e-15)
    assert_allclose(bottoms[-1], 0.0, atol=1e-15)
    # Within a sample, the shares are in the ratios of the cell numbers.
    cells = fixture("speckle")["speckle cells"]
    assert_allclose(heights[0] / heights[3], cells[:, 0] / cells[:, 3], rtol=1e-12)
    # Grouped by condition, with a gap between the levels and a header above each.
    ax = plotting.plot_composition(cube, by="condition", ax=axes())
    x = [p.get_x() for p in ax.containers[0]]
    assert np.diff(x)[5] == pytest.approx(2.0)
    assert [t.get_text() for t in ax.texts] == ["ctrl", "stim"]


def test_composition_of_a_result_is_its_samples_and_groups() -> None:
    cube = composition_cube()
    labels = ["ctrl", "stim"] * 6
    result = propeller(
        cube,
        Design(
            {"sample_id": cube.samples["sample_id"], "condition": labels},
            "~ condition",
            levels=LEVELS,
            contrasts={"stim": "conditionstim"},
        ),
    )
    ax = plotting.plot_composition(result, by="condition", ax=axes())
    # The bars follow the design's variable, not the cube's sample table.
    shown = [label.get_text() for label in ax.get_xticklabels()]
    assert shown[:6] == [f"S{i:02d}" for i in range(1, 13, 2)]
    by_sequence = plotting.plot_composition(result, by=labels, ax=axes())
    assert [t.get_text() for t in by_sequence.get_xticklabels()] == shown


def test_composition_refuses_what_it_cannot_draw() -> None:
    cells = np.ones((2, 9), dtype=np.int64)
    many = Cube(
        {"counts": np.ones((9, 2, 1), dtype=np.uint32)},
        samples={"sample_id": ["a", "b"]},
        groups={"group_id": [f"t{i}" for i in range(9)]},
        features={"feature_id": ["x"]},
        n_cells=cells,
    )
    with pytest.raises(ValueError, match="merge groups with Cube.regroup"):
        plotting.plot_composition(many)
    with pytest.raises(ValueError, match="does not record cell numbers"):
        plotting.plot_composition(bulk_cube())
    with pytest.raises(TypeError, match="Cube or a CompositionResult"):
        plotting.plot_composition(table())  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="no variable 'donor'"):
        plotting.plot_composition(composition_cube(), by="donor")


def test_proportions_need_a_test_to_show() -> None:
    cube = composition_cube()
    result = propeller(
        cube,
        Design(
            cube.samples,
            "~ condition",
            levels=LEVELS,
            contrasts={"stim": "conditionstim", "ctrl": "-conditionstim"},
        ),
    )
    with pytest.raises(ValueError, match="name one with contrast"):
        plotting.plot_proportions(result, by="condition")
    ax = plotting.plot_proportions(result, by="condition", contrast="ctrl", ax=axes())
    texts = [text.get_text() for text in artists(ax, "adjusted p-values")]
    adjusted = result.table("ctrl")["adjusted_p_value"]
    assert texts == [f"adj. p {p:.2g}" for p in adjusted]
    with pytest.raises(KeyError, match="no contrast 'other'"):
        plotting.plot_proportions(result, by="condition", contrast="other")
    with pytest.raises(TypeError, match="result must be a CompositionResult"):
        plotting.plot_proportions(cube, by="condition")  # type: ignore[arg-type]


def test_sample_weights_of_a_result_are_its_quality_weights() -> None:
    cube = from_bulk(counts(), samples=SAMPLES, features=GENES)
    result = limma_voom(cube, design(), quality_weights=True)
    ax = plotting.plot_sample_weights(result, ax=axes())
    heights = [patch.get_height() for patch in artists(ax, "weights")]

    assert_array_equal(heights, result.samples["quality_weight"])
    assert [label.get_text() for label in ax.get_xticklabels()] == SAMPLES["sample_id"]
    assert ax.get_ylabel() == "Sample quality weight"
    with pytest.raises(KeyError, match="no group 'T' was analysed"):
        plotting.plot_sample_weights(result, group="T")
    with pytest.raises(ValueError, match="holds no sample weights"):
        plotting.plot_sample_weights(limma_voom(cube, design()))
    with pytest.raises(ValueError, match="holds no sample weights"):
        plotting.plot_sample_weights(voom(counts(), design()))
    with pytest.raises(ValueError, match="positive and finite"):
        plotting.plot_sample_weights([1.0, 0.0])
    with pytest.raises(ValueError, match=r"one label per sample \(2\)"):
        plotting.plot_sample_weights([1.0, 2.0], labels=["a"])
    ax = plotting.plot_sample_weights([0.5, 2.0], labels=["a", "b"], ax=axes())
    assert [p.get_height() for p in artists(ax, "weights")] == [0.5, 2.0]


def test_fits_and_tests_are_drawn_from_their_own_columns() -> None:
    values = voom(counts(), design())
    fit = fit_linear_model(values.log_cpm, design(), weights=values.weights)
    moderated = moderated_statistics(fit)
    ax = plotting.plot_md(moderated, coefficient="conditionstim", ax=axes())
    drawn = points(ax, "up", "down", "not significant")
    assert_array_equal(drawn, pairs(fit.mean_expression, fit.coefficients[:, 1]))
    same = plotting.plot_md(moderated, coefficient=-1, ax=axes())
    assert_array_equal(points(same, "up"), points(ax, "up"))
    with pytest.raises(ValueError, match="name one with coefficient"):
        plotting.plot_md(moderated)
    with pytest.raises(KeyError, match="no coefficient or contrast 'x'"):
        plotting.plot_md(moderated, coefficient="x")
    with pytest.raises(IndexError, match="out of range for 2 coefficients"):
        plotting.plot_md(moderated, coefficient=2)
    # TREAT's p-values highlight fewer genes than the moderated t-tests.
    treated = treat(fit, lfc=1.0)
    ax = plotting.plot_volcano(treated, coefficient=1, labels=GENES, top=3, ax=axes())
    assert len(artists(ax, "gene labels")) == 3

    results = deseq2_wald_test(fit_deseq2(counts(), design()), "conditionstim")
    ax = plotting.plot_md(results, ax=axes())
    assert ax.get_xscale() == "log"
    drawn = points(ax, "up", "down", "not significant")
    shown = np.isfinite(results.log2_fold_change)
    assert_array_equal(
        drawn, pairs(results.base_mean[shown], results.log2_fold_change[shown])
    )
    with np.errstate(invalid="ignore"):
        up = (results.adjusted_p_value < 0.05) & (results.log2_fold_change > 0)
    assert len(points(ax, "up")) == np.count_nonzero(up)

    test = quasi_likelihood_f_test(
        fit_quasi_likelihood(counts(), design()), coefficients="conditionstim"
    )
    ax = plotting.plot_volcano(test, ax=axes())
    assert artists(ax, "gene labels") == []  # a test holds no gene names
    drawn = points(ax, "up", "down", "not significant")
    assert_allclose(
        drawn,
        pairs(test.log_fold_change[:, 0], -np.log10(test.p_value)),
        rtol=1e-15,
    )


def test_significant_genes_are_counted_in_each_group() -> None:
    layers = np.stack([counts(), counts()]).astype(np.uint32)
    n_cells = np.column_stack([np.full(8, 50), np.full(8, 3)])  # B: too few cells
    cube = Cube(
        {"counts": layers},
        samples=SAMPLES,
        groups={"group_id": ["T", "B"]},
        features={"feature_id": GENES},
        n_cells=n_cells,
    )
    result = limma_voom(cube, design(), adjust_across_groups=True)
    table = result.table("T", "stim")
    for column, across in (("adjusted_p_value", False), (ACROSS, True)):
        ax = plotting.plot_significant_genes(result, across_groups=across, ax=axes())
        significant = np.asarray(table[column]) < 0.05
        change = np.asarray(table["log2_fold_change"])
        up = int(np.count_nonzero(significant & (change > 0)))
        down = int(np.count_nonzero(significant & (change < 0)))
        assert up + down > 0
        # The top row is T; B has no bars.
        assert [bar.get_width() for bar in artists(ax, "up")] == [up, 0]
        assert [bar.get_width() for bar in artists(ax, "down")] == [-down, 0]
        assert [t.get_text() for t in artists(ax, "numbers")] == [str(up), str(down)]
        ticks = sorted(
            zip(ax.get_yticks(), ax.get_yticklabels(), strict=True), key=lambda t: -t[0]
        )
        assert [label.get_text() for _, label in ticks] == ["T", "B (not fitted)"]
    with pytest.raises(ValueError, match="no p-values adjusted across groups"):
        plotting.plot_significant_genes(limma_voom(cube, design()), across_groups=True)
    with pytest.raises(KeyError, match="no contrast 'other' was tested"):
        plotting.plot_significant_genes(result, contrast="other")
    with pytest.raises(TypeError, match="DifferentialExpressionResult"):
        plotting.plot_significant_genes(table)  # type: ignore[arg-type]


def test_volcano_labels_the_genes_with_the_smallest_p_values() -> None:
    ax = plotting.plot_volcano(table(), top=4, ax=axes())
    labels = artists(ax, "gene labels")
    p = table()["p_value"]
    smallest = np.argsort(p, kind="stable")[:4]
    assert sorted(label.get_text() for label in labels) == sorted(
        table()["feature_id"][smallest]
    )
    for label in labels:  # each label points at its gene
        row = table()["feature_id"].tolist().index(label.get_text())
        assert label.xy == pytest.approx(
            (table()["log2_fold_change"][row], -np.log10(p[row]))
        )
    names = Table({"feature_id": ["a", "b"], "symbol": ["A", None]})
    tiny = names.with_columns(
        log2_fold_change=[1.0, -1.0], p_value=[0.0, 1e-3], adjusted_p_value=[0.0, 1e-3]
    )
    ax = plotting.plot_volcano(tiny, ax=axes())
    # A p-value of 0 is drawn at the smallest positive one; the symbol, else the id.
    assert_allclose(points(ax, "up", "down")[:, 1], [3.0, 3.0])
    assert sorted(t.get_text() for t in artists(ax, "gene labels")) == ["A", "b"]


def test_tables_need_the_columns_of_their_figure() -> None:
    meta_like = table().select(["feature_id", "log2_fold_change", "p_value"])
    with pytest.raises(ValueError, match="needs the column 'average_log2_cpm'"):
        plotting.plot_md(meta_like)
    with pytest.raises(ValueError, match="needs the column 'adjusted_p_value'"):
        plotting.plot_volcano(meta_like)
    with pytest.raises(ValueError, match="coefficient applies to fits and tests"):
        plotting.plot_md(table(), coefficient=1)
    with pytest.raises(TypeError, match="result must be a table"):
        plotting.plot_md(counts())
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        plotting.plot_md(table(), alpha=1.0)


def test_p_value_histogram_counts_tests_per_bin() -> None:
    # Bins of 0.25: two p-values in the first, none in the second, one in each of
    # the others; NaN is left out. The line is at the 4 / 4 expected per bin.
    ax = plotting.plot_p_values([0.01, 0.02, 0.5, 0.97, np.nan], bins=4, ax=axes())
    heights = [patch.get_height() for patch in artists(ax, "p-values")]

    assert heights == [2, 0, 1, 1]
    assert artists(ax, "uniform")[0].get_ydata()[0] == 1.0
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        plotting.plot_p_values([0.5, 1.5])
    with pytest.raises(ValueError, match="no p-value to plot"):
        plotting.plot_p_values([np.nan])
    with pytest.raises(ValueError, match="bins must be at least 1"):
        plotting.plot_p_values([0.5], bins=0)


def test_the_worm_is_limmas_tricube_moving_average() -> None:
    # One gene of 20, ranked 11th: the worm is the impulse response of
    # tricubeMovingAverage(span = 0.45), which R printed as below, over its largest
    # value.
    response = [
        0.0038737068, 0.0646234304, 0.1461272941, 0.1880585326, 0.1946340723,
        0.1880585326, 0.1461272941, 0.0646234304, 0.0038737068,
    ]  # fmt: skip
    statistics = np.arange(20.0)
    ax = plotting.plot_barcode(statistics, [10], ax=axes())
    worm = artists(ax, "worm")[0].get_ydata() - 1.1
    expected = np.zeros(20)
    expected[6:15] = response
    assert_allclose(worm, expected / max(response), atol=1e-9)
    # At the start the window is cut short, as R's response to the first gene shows.
    ax = plotting.plot_barcode(statistics, [0], ax=axes())
    edge = [0.3258471808, 0.2394504490, 0.1568726194, 0.0648747361, 0.0038737068]
    worm = artists(ax, "worm")[0].get_ydata()[:5] - 1.1
    assert_allclose(worm, np.array(edge) / edge[0], atol=1e-9)


def test_barcode_takes_ids_positions_or_masks() -> None:
    t = table()["t"]
    ids = table()["feature_id"].tolist()
    by_id = plotting.plot_barcode(table(), [ids[3], ids[40], "unknown"], ax=axes())
    by_position = plotting.plot_barcode(t, [3, 40], ax=axes())
    mask = np.zeros(t.size, dtype=bool)
    mask[[3, 40]] = True
    by_mask = plotting.plot_barcode(t, mask, ax=axes())
    by_gene_ids = plotting.plot_barcode(t, [ids[40], ids[3]], gene_ids=ids, ax=axes())
    worms = [
        artists(ax, "worm")[0].get_ydata()
        for ax in (by_id, by_position, by_mask, by_gene_ids)
    ]
    for worm in worms[1:]:
        assert_array_equal(worm, worms[0])
    ranks = np.argsort(np.argsort(t, kind="stable"), kind="stable") + 1
    bars = sorted(s[0, 0] for s in artists(by_id, "set")[0].get_segments())
    assert bars == sorted(ranks[[3, 40]])
    assert by_id.get_xlabel() == "Moderated t-statistic"


@pytest.mark.parametrize(
    ("gene_set", "options", "message"),
    [
        (["unknown"], {"gene_ids": list("abcde")}, "no gene of the set"),
        (list(range(5)), {}, "holds every gene"),
        ([7], {}, "position outside 0 to 4"),
        (["a"], {}, "give gene_ids too"),
        ([0], {"span": 0.0}, r"span must be in \(0, 1\]"),
        ([0], {"gene_ids": ["a", "a", "b", "c", "d"]}, None),
    ],
)
def test_barcode_refuses_sets_it_cannot_place(
    gene_set: Any, options: dict[str, Any], message: str | None
) -> None:
    statistics = [0.5, -1.0, 2.0, 0.1, 3.0]
    if message is None:  # repeated ids
        with pytest.raises(ValueError, match="gene ids must be unique"):
            plotting.plot_barcode(statistics, ["a"], **options)
        return
    with pytest.raises(ValueError, match=message):
        plotting.plot_barcode(statistics, gene_set, **options)


def test_gene_sets_are_drawn_by_p_value_and_direction() -> None:
    values = voom(counts(), design())
    sets = {f"set{i}": GENES[10 * i : 10 * i + 15] for i in range(12)}
    tested = camera(values, design(), "conditionstim", sets, gene_ids=GENES)
    ax = plotting.plot_gene_sets(tested, top=5, ax=axes())
    bars = artists(ax, "gene sets")
    order = np.argsort(tested["p_value"], kind="stable")[:5]
    signs = np.where(tested["direction"][order] == "up", 1.0, -1.0)

    # The most significant set on top.
    tops = sorted(bars, key=lambda bar: -bar.get_y())
    assert_allclose(
        [bar.get_width() for bar in tops],
        signs * -np.log10(tested["adjusted_p_value"][order]),
        rtol=1e-15,
    )
    ticks = sorted(
        zip(ax.get_yticks(), ax.get_yticklabels(), strict=True), key=lambda t: -t[0]
    )
    names = [label.get_text() for _, label in ticks]
    assert names == [f"{tested['gene_set'][i]} ({tested['n_genes'][i]})" for i in order]
    with pytest.raises(ValueError, match="lacks the column"):
        plotting.plot_gene_sets(tested.select(["gene_set", "p_value"]))
    with pytest.raises(TypeError, match="table must be a Table"):
        plotting.plot_gene_sets({"gene_set": ["a"]})  # type: ignore[arg-type]


def test_count_model_figures_refuse_other_objects() -> None:
    given = fit_deseq2(counts(), design(), dispersion=np.full(300, 0.1))
    with pytest.raises(ValueError, match="given its dispersions"):
        plotting.plot_dispersions(given)
    with pytest.raises(TypeError, match="DESeq2Dispersions or a DESeq2Fit"):
        plotting.plot_dispersions(table())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="QuasiLikelihoodFit or a QuasiLikelihoodTest"):
        plotting.plot_quasi_dispersions(table())  # type: ignore[arg-type]
    # Without a trend, the prior is a line at its value.
    flat = fit_quasi_likelihood(counts(), design(), trend=False)
    ax = plotting.plot_quasi_dispersions(flat, ax=axes())
    level = artists(ax, "prior")[0].get_ydata()
    assert_allclose(level, np.sqrt(np.sqrt(flat.prior_quasi_dispersion)), rtol=1e-15)


def test_trend_figures_refuse_other_objects() -> None:
    with pytest.raises(TypeError, match="result must be a VoomResult"):
        plotting.plot_mean_variance(table())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ModeratedFit or a TreatFit"):
        plotting.plot_residual_variances(table())  # type: ignore[arg-type]
    with pytest.warns(RuntimeWarning, match="no residual degrees of freedom"):
        unreplicated = voom(counts()[:2], np.eye(2))
    with pytest.raises(ValueError, match="no mean-variance trend was fitted"):
        plotting.plot_mean_variance(unreplicated)


def test_forest_refuses_what_it_cannot_find() -> None:
    meta = differential_expression_meta_analysis(studies())
    with pytest.raises(KeyError, match="gene 'nothing' was not meta-analysed"):
        plotting.plot_forest(meta, studies(), "nothing")
    with pytest.raises(TypeError, match="studies must map study names"):
        plotting.plot_forest(meta, {"A": table()}, "g001")
    with pytest.raises(TypeError, match="feature must be a gene id"):
        plotting.plot_forest(meta, studies(), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="studies holds no study"):
        plotting.plot_forest(meta, {}, "g001")
    with pytest.raises(KeyError, match="no group 'T' was analysed"):
        plotting.plot_forest(meta, studies(), "g001", group="T", contrast="stim")


# Colours, settings and imports.

# Machado, Oliveira and Fernandes (2009), severity 1, on linear RGB.
PROTANOPIA = [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
]
DEUTERANOPIA = [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
]


def oklab(rgb: np.ndarray, simulation: Any = None) -> np.ndarray:
    """OKLab coordinates (Ottosson 2020) of sRGB colours, rows of rgb in [0, 1]."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    if simulation is not None:
        linear = np.clip(linear @ np.asarray(simulation).T, 0.0, 1.0)
    lms = (
        linear
        @ np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        ).T
    )
    root = np.cbrt(lms)
    return (
        root
        @ np.array(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        ).T
    )


def separation(colors: np.ndarray, pairs_of: list[tuple[int, int]]) -> float:
    """The smallest distance (OKLab, times 100) of the pairs under simulated
    protanopia and deuteranopia."""
    smallest = np.inf
    for simulation in (PROTANOPIA, DEUTERANOPIA):
        lab = oklab(colors, simulation)
        for i, j in pairs_of:
            smallest = min(smallest, 100.0 * np.linalg.norm(lab[i] - lab[j]))
    return smallest


def test_categorical_colours_stay_apart_under_colour_vision_deficiencies() -> None:
    eight = Cube(
        {"counts": np.ones((8, 2, 1), dtype=np.uint32)},
        samples={"sample_id": ["a", "b"]},
        groups={"group_id": [f"t{i}" for i in range(8)]},
        features={"feature_id": ["x"]},
        n_cells=np.ones((2, 8), dtype=np.int64),
    )
    ax = plotting.plot_composition(eight, ax=axes())
    colors = np.array([c.patches[0].get_facecolor()[:3] for c in ax.containers])
    assert len({tuple(c) for c in colors}) == 8
    # Neighbouring segments, as stacked bars place them, and any two of the first
    # three hues, as a scatter plot of three levels can place them.
    assert separation(colors, [(i, i + 1) for i in range(7)]) >= 8.0
    assert separation(colors, [(0, 1), (0, 2), (1, 2)]) >= 8.0
    # Neighbours also stay apart for full colour vision.
    lab = oklab(colors)
    assert min(100 * np.linalg.norm(lab[i] - lab[i + 1]) for i in range(7)) >= 15.0
    # Up and down.
    up_down = np.array([COLORS.to_rgb(c) for c in ("#e34948", "#2a78d6")])
    md = plotting.plot_md(table(), ax=axes())
    drawn = np.array(
        [artists(md, gid)[0].get_facecolor()[0][:3] for gid in ("up", "down")]
    )
    assert_allclose(drawn, up_down)
    assert separation(drawn, [(0, 1)]) >= 8.0


def test_drawing_changes_no_setting_of_matplotlib() -> None:
    before = dict(matplotlib.rcParams)
    plotting.plot_mds(log_cpm(counts()), color_by=SAMPLES["condition"], ax=axes())
    plotting.plot_md(table(), ax=axes())
    plotting.plot_cell_numbers(composition_cube(), ax=axes())

    assert dict(matplotlib.rcParams) == before


def test_a_new_figure_is_made_when_no_axes_are_given() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    ax = plotting.plot_p_values([0.1, 0.5])
    try:
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax.figure.get_layout_engine() is not None
    finally:
        pyplot.close(ax.figure)


def test_no_module_of_cytepool_imports_matplotlib_when_imported() -> None:
    # Every import that runs when a module is imported: those at its top level,
    # outside functions and the blocks that only type checkers read.
    package = Path(plotting.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_text()
        if "matplotlib" not in source:
            continue
        tree = ast.parse(source)
        statements = list(tree.body)
        while statements:
            node = statements.pop()
            if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
                statements.extend(node.orelse)
                continue
            if isinstance(node, ast.If | ast.Try | ast.With):
                statements.extend(ast.iter_child_nodes(node))  # type: ignore[arg-type]
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            found += [f"{path.name}: {n}" for n in names if n.startswith("matplotlib")]
    assert found == []
    # plotting.py imports it in the functions that draw, and names it for type checkers.
    assert "matplotlib" in (package / "plotting.py").read_text()
