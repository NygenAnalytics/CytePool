"""Tests for principal components, their figure and dot plots.

They compare with:

- what DESeq2's ``plotPCA`` returned and drew, and what scanpy's ``dotplot`` computed
  and drew from the cells, saved by ``validation/pca_and_dot_plots.py`` in
  ``tests/data/pca_and_dot_plots/``;
- values derived by hand: three samples on two genes whose components are the genes,
  two samples, one gene, and a gene expressed in half the cells of two samples;
- invariants that a wrong implementation would break: the order of the samples and of
  the genes, the sign and the shift of the values, a group merged or samples pooled
  in the cube against the same cells given one label, and scaled means that run from
  0 to 1.

Figures are drawn on Figures that pyplot does not manage, and read back from the
artists that carry their gid.
"""

import importlib
import json
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import plotting
from cytepool.cube import Cube, from_cells
from cytepool.principal_components import PrincipalComponents, principal_components

FIGURE = pytest.importorskip("matplotlib.figure")

DATA = Path(__file__).parent / "data" / "pca_and_dot_plots"
EXACT = 1e-10
SAMPLES = [f"S{i}" for i in range(1, 13)]
CONDITIONS = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
BATCHES = ["x", "y"] * 6
GROUPS = ["B", "T", "NK", "pDC"]
CELL_SAMPLES = [f"D{i}" for i in range(1, 7)]
MARKERS = {"B cells": ["g00", "g01", "g02"], "T cells": ["g03", "g04"]}
MARKERS["others"] = ["g10", "g20", "g30", "g39"]
GENES = [gene for genes in MARKERS.values() for gene in genes]
DOT_CASES: dict[str, dict[str, Any]] = {
    "plain": {},
    "scaled by gene": {"standard_scale": "gene"},
    "scaled by group": {"standard_scale": "group"},
    "expressing cells": {"mean_only_expressed": True},
    "chosen samples and groups": {
        "samples": ["D2", "D3", "D5"],
        "groups": ["NK", "B", "T"],
    },
    "genes in groups": {"genes": MARKERS},
}


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@cache
def dot_texts() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((DATA / "dot_plots.json").read_text())
    return result


def axes() -> Any:
    return FIGURE.Figure().subplots()


def artists(ax: Any, gid: str) -> list[Any]:
    return [artist for artist in ax.get_children() if artist.get_gid() == gid]


def scaled(ours: Any, reference: Any) -> float:
    """Largest difference over max(|reference|, 1), as the validation script has it."""
    ours, reference = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    assert ours.shape == reference.shape
    return float(np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1.0)))


def up_to_sign(ours: np.ndarray, theirs: np.ndarray) -> float:
    signs = np.sign(np.sum(ours * theirs, axis=0))
    return scaled(ours * signs, theirs)


def pca_cube() -> Cube:
    data = fixture("pca")
    return Cube(
        {"counts": data["counts"][np.newaxis].astype(np.uint32)},
        samples={"sample_id": SAMPLES, "condition": CONDITIONS, "batch": BATCHES},
        groups={"group_id": ["T"]},
        features={"feature_id": [f"g{j:03d}" for j in range(data["counts"].shape[1])]},
        n_cells=data["n_cells"][:, np.newaxis],
    )


@cache
def dot_cube() -> Cube:
    data = fixture("dot_plots")
    return Cube(
        {
            "counts": data["detected"],  # counts are not drawn; any valid layer will do
            "detected": data["detected"],
            "log_sum": data["log_sum"],
        },
        samples={"sample_id": CELL_SAMPLES},
        groups={"group_id": GROUPS},
        features={"feature_id": [f"g{j:02d}" for j in range(40)]},
        n_cells=data["n_cells"],
        normalization={"method": "library_size", "target_sum": 1e4, "pseudocount": 1.0},
    )


# What DESeq2 and scanpy computed and drew.


@pytest.mark.parametrize(
    ("case", "top", "components"),
    [("top100", 100, (1, 2)), ("top100_pc34", 100, (3, 4)), ("top500", 500, (1, 2))],
)
def test_components_are_those_of_plotPCA_up_to_sign(
    case: str, top: int, components: tuple[int, int]
) -> None:
    data = fixture("pca")
    result = principal_components(data["values"], top=top)
    chosen = [components[0] - 1, components[1] - 1]

    assert up_to_sign(result.coordinates[:, chosen], data[f"{case} coordinates"]) <= (
        EXACT
    )
    assert scaled(result.variance_explained[chosen], data[f"{case} percent"]) <= EXACT
    # 500 is more than the genes: every gene, in decreasing order of variance.
    assert result.genes.size == min(top, data["values"].shape[1])


def test_genes_of_equal_variance_keep_their_order_as_plotPCA_keeps_them() -> None:
    data = fixture("pca")
    result = principal_components(data["ties"], top=20)

    # Genes 18 to 21 (from 0) have equal variance; the 20 kept end with 18 and 19.
    assert result.genes.tolist()[-2:] == [18, 19]
    assert up_to_sign(result.coordinates[:, :2], data["ties coordinates"]) <= EXACT


def test_a_cube_group_is_prepared_as_edgeR_prepares_it_for_plotPCA() -> None:
    data = fixture("pca")
    result = principal_components(pca_cube())

    # The two samples with 5 cells are left out.
    assert result.samples["sample_id"].tolist() == SAMPLES[:10]
    assert result.samples["condition"].tolist() == CONDITIONS[:10]
    assert up_to_sign(result.coordinates[:, :2], data["cube coordinates"]) <= EXACT
    assert result.settings == {"group": "T", "min_cells": 10, "top": 500}


def test_components_of_a_cube_record_the_cube_they_read() -> None:
    cube = pca_cube()
    result = principal_components(cube)
    record = result.provenance
    assert record is not None
    assert record["method"] == "principal_components"
    assert record["cube"]["fingerprint"] == cube.fingerprint()
    assert record["settings"] == result.settings
    assert json.loads(json.dumps(record)) == record
    of_values = principal_components(fixture("pca")["values"])
    assert of_values.provenance is None
    # Both pickle, the provenance with them; the copy keeps none it did not have.
    for found in (result, of_values):
        copy = pickle.loads(pickle.dumps(found))
        for name in ("coordinates", "variance_explained", "loadings", "genes"):
            assert_array_equal(getattr(copy, name), getattr(found, name))
            assert not getattr(copy, name).flags.writeable
        assert copy.samples.equals(found.samples)
        assert dict(copy.settings) == dict(found.settings)
        assert copy.provenance == found.provenance
    assert "_provenance" not in vars(pickle.loads(pickle.dumps(of_values)))


def test_the_figure_draws_the_points_and_labels_of_plotPCA() -> None:
    data = fixture("pca")
    ax = plotting.plot_pca(data["values"], top=100, labels=True, ax=axes())
    drawn = np.array([label.xy for label in artists(ax, "sample labels")])

    assert up_to_sign(drawn, data["figure coordinates"]) <= EXACT
    assert [ax.get_xlabel(), ax.get_ylabel()] == data["figure labels"].tolist()
    assert ax.get_aspect() == 1.0  # plotPCA's coord_fixed


@pytest.mark.parametrize("case", list(DOT_CASES))
def test_dot_plot_areas_and_colours_are_scanpys(case: str) -> None:
    options = {"genes": GENES} | DOT_CASES[case]
    genes = options.pop("genes")
    ax = plotting.plot_dotplot(dot_cube(), genes, ax=axes(), **options)
    dots = artists(ax, "dots")[0]
    reference = fixture("dot_plots")
    texts = dot_texts()[case]

    assert scaled(dots.get_sizes(), reference[f"{case} sizes"]) <= EXACT
    assert scaled(dots.get_array(), reference[f"{case} color"].ravel()) <= EXACT
    assert [t.get_text() for t in ax.get_yticklabels()] == texts["rows"]
    assert [t.get_text() for t in ax.get_xticklabels()] == texts["columns"]
    legends = [a.get_legend() for a in ax.figure.axes if a.get_legend() is not None]
    assert [t.get_text() for t in legends[0].get_texts()] == texts["key"]
    if texts["gene_groups"] is not None:
        labels, positions = texts["gene_groups"]
        brackets = [line.get_xdata() for line in artists(ax, "gene groups")]
        assert [[round(x[0] + 0.3), round(x[-1] - 0.3)] for x in brackets] == positions
        assert [t.get_text() for t in artists(ax, "gene group names")] == labels


@pytest.mark.parametrize("standard_scale", [None, "gene"])
def test_dot_plot_labels_its_colour_bar_clear_of_the_key(
    standard_scale: str | None,
) -> None:
    """The colour bar's label is written above the bar, as scanpy writes it, and the
    key of the dots' areas below it: a label along the bar ran into the key."""
    ax = plotting.plot_dotplot(
        dot_cube(), GENES, standard_scale=standard_scale, ax=axes()
    )
    figure = ax.figure
    canvas = importlib.import_module("matplotlib.backends.backend_agg").FigureCanvasAgg(
        figure
    )
    canvas.draw()
    renderer = canvas.get_renderer()
    bar = figure.axes[-1]
    title = bar.get_title(loc="left")
    written = [
        child
        for child in bar.get_children()
        if hasattr(child, "get_text") and child.get_text() == title
    ]
    label = written[0].get_window_extent(renderer)
    key = bar.get_legend().get_window_extent(renderer)
    frame = bar.get_window_extent(renderer)

    assert title.startswith("Mean expression")
    assert not label.overlaps(key)
    assert not label.overlaps(frame) and label.y0 >= frame.y1
    assert not key.overlaps(frame) and key.y1 <= frame.y0


# Values derived by hand, and invariants.


def test_three_samples_on_two_genes_whose_components_are_the_genes() -> None:
    # Centred, gene 0 is (-1, 1, 0) and gene 1 (-1/3, -1/3, 2/3): orthogonal, with
    # sums of squares 2 and 2/3, so the components are the genes with shares 3/4 and
    # 1/4. Of PC1's tied largest coordinates, the first sample's is made positive.
    values = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    result = principal_components(values)

    assert_allclose(
        result.coordinates, [[1, -1 / 3], [-1, -1 / 3], [0, 2 / 3]], atol=1e-15
    )
    assert_allclose(result.variance_explained, [0.75, 0.25], rtol=1e-15)
    assert_allclose(np.abs(result.loadings), np.eye(2), atol=1e-15)
    assert result.genes.tolist() == [0, 1]
    assert result.samples["sample_id"].tolist() == ["1", "2", "3"]


def test_two_samples_have_one_component_along_their_difference() -> None:
    # The genes of variance 8 and 4.5 are kept, the constant one is not; centred, the
    # first sample is (-2, -1.5) on them, at distance 2.5 from the centre.
    values = np.array([[0.0, 0.0, 5.0], [3.0, 4.0, 5.0]])
    result = principal_components(values, top=2)

    assert result.genes.tolist() == [1, 0]
    assert_allclose(result.coordinates, [[2.5], [-2.5]], rtol=1e-15)
    assert_allclose(result.loadings, [[-0.8], [-0.6]], rtol=1e-15)
    assert_allclose(result.variance_explained, [1.0])
    with pytest.raises(ValueError, match="need 2 components; the data have 1"):
        plotting.plot_pca(result, ax=axes())


def test_one_gene_gives_its_centred_values() -> None:
    values = np.array([[1.0, 9.0], [4.0, 9.5], [7.0, 10.0], [0.0, 11.0]])
    result = principal_components(values, top=1)

    assert result.genes.tolist() == [0]  # variance 10 against 0.729
    assert_allclose(result.coordinates[:, 0], [-2.0, 1.0, 4.0, -3.0])


@cache
def random_values() -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(9, 60))
    values[:4, :12] += 2.5
    values[::3, 30:40] -= 1.5
    return values


def test_the_order_of_samples_and_genes_and_the_sign_of_the_values_do_not_matter() -> (
    None
):
    values = random_values()
    base = principal_components(values, top=25)
    samples = np.array([4, 0, 8, 2, 6, 1, 3, 7, 5])
    genes = np.random.default_rng(8).permutation(60)

    by_sample = principal_components(values[samples], top=25)
    assert_allclose(by_sample.coordinates, base.coordinates[samples], atol=1e-12)
    by_gene = principal_components(values[:, genes], top=25)
    assert_allclose(by_gene.coordinates, base.coordinates, atol=1e-12)
    assert_array_equal(genes[by_gene.genes], base.genes)
    negated = principal_components(-values, top=25)
    assert_allclose(negated.coordinates, base.coordinates, atol=1e-12)
    assert_allclose(negated.loadings, -base.loadings, atol=1e-12)
    shifted = principal_components(values * 3.0 + np.arange(60.0), top=25)
    assert_allclose(shifted.coordinates, 3.0 * base.coordinates, atol=1e-11)
    assert_allclose(shifted.variance_explained, base.variance_explained, rtol=1e-12)


def test_components_are_orthonormal_and_their_shares_sum_to_one() -> None:
    result = principal_components(random_values(), top=25)
    centred = random_values()[:, result.genes] - random_values()[:, result.genes].mean(
        0
    )

    assert result.n_components == 8  # one fewer than the samples
    assert_allclose(result.loadings.T @ result.loadings, np.eye(8), atol=1e-12)
    assert_allclose(centred @ result.loadings, result.coordinates, atol=1e-12)
    assert_allclose(result.variance_explained.sum(), 1.0, rtol=1e-15)
    assert np.all(np.diff(result.variance_explained) <= 0.0)
    largest = np.argmax(np.abs(result.coordinates), axis=0)
    assert np.all(result.coordinates[largest, np.arange(8)] > 0.0)


def test_results_are_read_only_records() -> None:
    result = principal_components(random_values(), top=10)

    assert isinstance(result, PrincipalComponents)
    with pytest.raises(ValueError, match="read-only"):
        result.coordinates[0, 0] = 1.0
    assert "n_components=8" in repr(result)


def test_the_figure_colours_marks_and_labels_samples_by_their_variables() -> None:
    cube = pca_cube()
    ax = plotting.plot_pca(
        cube, color_by="condition", marker_by="batch", labels=True, ax=axes()
    )

    collections = artists(ax, "samples")
    assert len(collections) == 2  # one per batch
    hues = {tuple(face) for c in collections for face in c.get_facecolors().tolist()}
    assert len(hues) == 3  # one per condition
    assert [t.get_text() for t in artists(ax, "sample labels")] == SAMPLES[:10]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["condition", "a", "b", "c", "batch", "x", "y"]
    # Labels given one per sample of the cube, of which the plotted ten are shown.
    ax = plotting.plot_pca(cube, labels=[f"L{i}" for i in range(12)], ax=axes())
    assert [t.get_text() for t in artists(ax, "sample labels")] == [
        f"L{i}" for i in range(10)
    ]


def test_the_figure_draws_a_result_and_chosen_components() -> None:
    result = principal_components(random_values(), top=25)
    ax = plotting.plot_pca(result, components=(2, 3), labels=True, ax=axes())
    drawn = np.array([label.xy for label in artists(ax, "sample labels")])

    assert_array_equal(drawn, result.coordinates[:, 1:3])
    share = 100 * result.variance_explained
    assert ax.get_xlabel() == f"PC2: {share[1]:.0f}% variance"
    assert ax.get_ylabel() == f"PC3: {share[2]:.0f}% variance"


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"data": np.ones((1, 3))}, ValueError, "at least two samples"),
        ({"data": np.ones(3)}, ValueError, "2-D array"),
        ({"data": np.array([[1.0, np.nan], [2.0, 3.0]])}, ValueError, "finite"),
        ({"data": np.ones((3, 4))}, ValueError, "do not vary"),
        # The mean of three values of 0.1 is not 0.1 in floating point: the centred
        # values are rounding errors, not variation.
        ({"data": np.full((3, 4), 0.1)}, ValueError, "do not vary"),
        ({"data": np.ones((3, 0))}, ValueError, "no genes"),
        ({"data": np.eye(3), "top": 0}, ValueError, "top must be at least 1"),
        ({"data": np.eye(3), "top": 2.5}, TypeError, "top must be an integer"),
        ({"data": np.eye(3), "group": "T"}, ValueError, "group applies to a cube"),
        (
            {"data": np.eye(3), "min_cells": 3},
            ValueError,
            "min_cells applies to a cube",
        ),
        ({"data": "values"}, TypeError, "real numbers"),
    ],
)
def test_principal_components_refuse_invalid_arguments(
    arguments: dict[str, Any], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        principal_components(**arguments)


def test_a_cube_needs_its_group_and_enough_samples() -> None:
    cube = dot_cube()
    with pytest.raises(ValueError, match="the cube has 4 groups; name the one"):
        principal_components(cube)
    with pytest.raises(KeyError, match="Mono"):
        principal_components(cube, group="Mono")
    with pytest.raises(ValueError, match="principal components need two or more"):
        principal_components(cube, group="pDC", min_cells=10_000)
    with pytest.raises(ValueError, match="min_cells must be at least 0"):
        principal_components(cube, group="B", min_cells=-1)


def test_the_figure_refuses_options_a_result_has_used() -> None:
    result = principal_components(random_values(), top=25)
    with pytest.raises(ValueError, match="top applies to the values"):
        plotting.plot_pca(result, top=10, ax=axes())
    with pytest.raises(ValueError, match="two different components"):
        plotting.plot_pca(result, components=(1, 1), ax=axes())
    with pytest.raises(TypeError, match="pair of components"):
        plotting.plot_pca(result, components=1, ax=axes())


def hand_cube(**options: Any) -> Cube:
    """Two samples of one group: gene a detected in 2 of 4 and 3 of 6 cells, gene b in
    every cell."""
    return Cube(
        {
            "counts": np.array([[[3, 9], [5, 12]]], dtype=np.uint32),
            "detected": np.array([[[2, 4], [3, 6]]], dtype=np.uint32),
            "log_sum": np.array([[[2.0, 4.0], [4.0, 6.0]]]),
        },
        samples={"sample_id": ["s1", "s2"]},
        groups={"group_id": ["T"]},
        features={"feature_id": ["a", "b"], "symbol": ["GA", "GB"]},
        n_cells=np.array([[4], [6]]),
        normalization=options.get(
            "normalization", {"method": "library_size", "pseudocount": 1.0}
        ),
    )


def test_a_gene_expressed_in_half_the_cells_of_two_samples() -> None:
    ax = plotting.plot_dotplot(hand_cube(), ["GA", "b"], ax=axes())
    dots = artists(ax, "dots")[0]

    # Fractions 5/10 and 10/10; the largest is 1, so the areas are 200 (f / 1)^1.5.
    assert_allclose(dots.get_sizes(), [200.0 * 0.5**1.5, 200.0], rtol=1e-15)
    # Means 6/10 and 10/10 over every cell.
    assert_allclose(dots.get_array(), [0.6, 1.0], rtol=1e-15)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["GA", "b"]
    ax = plotting.plot_dotplot(hand_cube(), ["a"], mean_only_expressed=True, ax=axes())
    # 6 over the 5 cells that express it; the largest fraction, 0.5, has the largest dot.
    assert_allclose(artists(ax, "dots")[0].get_array(), [1.2], rtol=1e-15)
    assert_allclose(artists(ax, "dots")[0].get_sizes(), [200.0], rtol=1e-15)
    ax = plotting.plot_dotplot(hand_cube(), ["a", "b"], samples=["s1"], ax=axes())
    assert_allclose(artists(ax, "dots")[0].get_array(), [0.5, 1.0], rtol=1e-15)


@cache
def cells() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    n = 600
    counts = rng.poisson(rng.uniform(0.05, 3.0, 12), size=(n, 12)).astype(np.float64)
    counts[counts.sum(axis=1) == 0, 0] = 1.0
    return {
        "counts": counts,
        "samples": rng.choice(["s1", "s2", "s3"], n),
        "groups": rng.choice(["B", "T", "NK"], n),
        "features": [f"x{j}" for j in range(12)],
    }


def test_a_merged_group_and_pooled_samples_are_the_same_cells_given_one_label() -> None:
    data = cells()
    cube = from_cells(**data)
    genes = ["x0", "x3", "x7", "x11"]
    merged = cube.regroup({"BT": ["B", "T"]})
    relabelled = from_cells(
        **(data | {"groups": np.where(data["groups"] == "NK", "NK", "BT")})
    )
    one_sample = from_cells(**(data | {"samples": np.full(data["groups"].size, "all")}))

    for first, second, groups in (
        (merged, relabelled, ["BT"]),
        (cube, one_sample, ["NK", "T"]),
    ):
        ours = artists(
            plotting.plot_dotplot(first, genes, groups=groups, ax=axes()), "dots"
        )
        theirs = artists(
            plotting.plot_dotplot(second, genes, groups=groups, ax=axes()), "dots"
        )
        assert_allclose(ours[0].get_sizes(), theirs[0].get_sizes(), rtol=1e-14)
        assert_allclose(ours[0].get_array(), theirs[0].get_array(), rtol=1e-14)


def test_scaled_means_run_from_zero_to_one() -> None:
    cube = from_cells(**cells())
    genes = ["x0", "x3", "x7", "x11"]
    for scale, axis in (("gene", 0), ("group", 1)):
        ax = plotting.plot_dotplot(cube, genes, standard_scale=scale, ax=axes())
        values = np.asarray(artists(ax, "dots")[0].get_array()).reshape(3, 4)
        assert_allclose(values.min(axis=axis), 0.0, atol=0)
        assert_allclose(values.max(axis=axis), 1.0, rtol=1e-15)
    # A gene equal in every group is 0 throughout.
    ax = plotting.plot_dotplot(
        hand_cube(), ["a", "b"], standard_scale="gene", ax=axes()
    )
    assert_array_equal(artists(ax, "dots")[0].get_array(), [0.0, 0.0])


def test_colour_limits_put_dot_plots_on_one_scale() -> None:
    cube = from_cells(**cells())
    genes = ["x0", "x3", "x7", "x11"]
    plain = artists(plotting.plot_dotplot(cube, genes, ax=axes()), "dots")[0]
    values = np.asarray(plain.get_array())
    # By default the colours run from the smallest value shown to the largest.
    assert (plain.norm.vmin, plain.norm.vmax) == (values.min(), values.max())
    ax = plotting.plot_dotplot(cube, genes, vmin=0.0, vmax=3.0, ax=axes())
    dots = artists(ax, "dots")[0]
    assert (dots.norm.vmin, dots.norm.vmax) == (0.0, 3.0)
    assert_array_equal(dots.get_array(), values)  # the values are those drawn before
    assert ax.figure.axes[-1].get_ylim() == (0.0, 3.0)  # the colour bar
    # One limit given, the other is the value shown at that end.
    upper = artists(plotting.plot_dotplot(cube, genes, vmax=3.0, ax=axes()), "dots")[0]
    assert (upper.norm.vmin, upper.norm.vmax) == (values.min(), 3.0)
    # Two plots of other groups: on one scale, a group's dots have the same colours.
    shared, alone = [], []
    for chosen in (["B", "T"], ["T", "NK"]):
        position = chosen.index("T")
        for limits, found in (({"vmin": 0.0, "vmax": 3.0}, shared), ({}, alone)):
            drawn = artists(
                plotting.plot_dotplot(cube, genes, groups=chosen, ax=axes(), **limits),
                "dots",
            )[0]
            rows = np.asarray(drawn.get_array()).reshape(2, len(genes))
            found.append(drawn.to_rgba(rows[position]))
    assert_array_equal(shared[0], shared[1])
    assert not np.array_equal(alone[0], alone[1])


def test_dot_plots_refuse_what_they_cannot_draw() -> None:
    bulk = Cube(
        {"counts": np.ones((1, 2, 2), dtype=np.uint32)},
        samples={"sample_id": ["s1", "s2"]},
        groups={"group_id": ["bulk"]},
        features={"feature_id": ["a", "b"]},
    )
    with pytest.raises(ValueError, match="the cube lacks cell numbers"):
        plotting.plot_dotplot(bulk, ["a"], ax=axes())
    with pytest.raises(KeyError, match="no gene 'c'"):
        plotting.plot_dotplot(hand_cube(), ["c"], ax=axes())
    with pytest.raises(ValueError, match="genes is empty"):
        plotting.plot_dotplot(hand_cube(), [], ax=axes())
    with pytest.raises(ValueError, match="the group of genes 'g' is empty"):
        plotting.plot_dotplot(hand_cube(), {"g": []}, ax=axes())
    with pytest.raises(KeyError, match="no group 'B'"):
        plotting.plot_dotplot(hand_cube(), ["a"], groups=["B"], ax=axes())
    with pytest.raises(ValueError, match="names a sample more than once"):
        plotting.plot_dotplot(hand_cube(), ["a"], samples=["s1", "s1"], ax=axes())
    with pytest.raises(ValueError, match="one entry per sample"):
        plotting.plot_dotplot(hand_cube(), ["a"], samples=[True], ax=axes())
    with pytest.raises(ValueError, match="standard_scale must be one of"):
        plotting.plot_dotplot(hand_cube(), ["a"], standard_scale="var", ax=axes())
    without = hand_cube(normalization={"method": "library_size", "pseudocount": 0.5})
    with pytest.raises(ValueError, match="pseudocount of 1"):
        plotting.plot_dotplot(without, ["a"], mean_only_expressed=True, ax=axes())
    empty = from_cells(
        **(cells() | {"groups": np.where(cells()["samples"] == "s1", "A", "B")})
    )
    with pytest.raises(
        ValueError, match="group 'A' has no cells in the samples pooled"
    ):
        plotting.plot_dotplot(empty, ["x0"], samples=["s2"], ax=axes())
    twice = Cube(
        {
            "counts": np.ones((1, 1, 2), dtype=np.uint32),
            "detected": np.ones((1, 1, 2), dtype=np.uint32),
            "log_sum": np.ones((1, 1, 2)),
        },
        samples={"sample_id": ["s1"]},
        groups={"group_id": ["T"]},
        features={"feature_id": ["a", "b"], "symbol": ["G", "G"]},
        n_cells=np.array([[1]]),
        normalization={"method": "library_size"},
    )
    with pytest.raises(ValueError, match="the symbol 'G' names 2 genes"):
        plotting.plot_dotplot(twice, ["G"], ax=axes())
    with pytest.raises(ValueError, match=r"vmin \(3\) must be below vmax \(1\)"):
        plotting.plot_dotplot(hand_cube(), ["a"], vmin=3, vmax=1, ax=axes())
    with pytest.raises(ValueError, match=r"the values shown run from 0\.6 to 1"):
        plotting.plot_dotplot(hand_cube(), ["a", "b"], vmax=0.5, ax=axes())
    with pytest.raises(ValueError, match="vmin must be finite"):
        plotting.plot_dotplot(hand_cube(), ["a"], vmin=np.nan, ax=axes())
    with pytest.raises(TypeError, match="vmax must be a real number"):
        plotting.plot_dotplot(hand_cube(), ["a"], vmax="high", ax=axes())
