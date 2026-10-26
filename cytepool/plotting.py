"""Figures of pseudobulk and bulk analyses, drawn as limma, edgeR, DESeq2, speckle and
metafor draw them.

Each function draws one figure from the objects that CytePool's analyses return: a
cube, a voom result, a fit, the table of a differential-expression or gene-set test,
or a composition or meta-analysis result. It reads the values from them, so that a
figure shows what the analysis computed, draws into ``ax``, a matplotlib Axes, or into
a new figure when ``ax`` is None, and returns the Axes, whose title, limits and labels
can then be changed. The artists that show data carry a ``gid`` that names what they
show, such as ``"trend"``, so that they can be found and restyled; each function lists
its own.

matplotlib is an optional dependency: install CytePool's ``plot`` extra, or matplotlib
3.9 or newer. These functions import it when they draw, and change none of its
settings.

The figures:

- Samples: :func:`plot_mds`, the multidimensional scaling plot of limma's ``plotMDS``
  (Ritchie et al. 2015), whose distances are leading log2 fold changes.
- Cubes: :func:`plot_cell_numbers`, the cells pooled into each sample and group.
- Composition: :func:`plot_composition`, the cell-type proportions of every sample, as
  speckle's ``plotCellTypeProps`` draws them; :func:`plot_proportions`, the proportions
  of every cell type by condition, with the adjusted p-values of ``propeller``
  (Phipson et al. 2022).
- Mean-variance trends: :func:`plot_mean_variance`, voom's plot (Law et al. 2014);
  :func:`plot_residual_variances`, limma's ``plotSA``.
- Sample weights: :func:`plot_sample_weights`, the sample quality weights that limma's
  ``voomWithQualityWeights`` plots (Liu et al. 2015).
- Differential expression: :func:`plot_md`, limma's mean-difference plot ``plotMD``;
  :func:`plot_volcano`, limma's ``volcanoplot``; :func:`plot_p_values`, a histogram of
  p-values; :func:`plot_significant_genes`, the genes up and down in every group.
- Gene sets: :func:`plot_barcode`, limma's ``barcodeplot``; :func:`plot_gene_sets`, the
  sets of a ``camera``, ``fry`` or ``mroast`` table.
- Count models: :func:`plot_dispersions`, DESeq2's ``plotDispEsts`` (Love et al.
  2014); :func:`plot_quasi_dispersions`, edgeR's ``plotQLDisp`` (Chen et al. 2016).
- Meta-analysis: :func:`plot_forest`, the forest plot (Lewis and Clarke 2001) that
  metafor's ``forest`` draws (Viechtbauer 2010).

``validation/plotting.py`` compares what each figure shows with what the R function it
follows draws for the same data, where one exists.

Colours:
    Categories, such as the cell types of a composition or the levels of a sample
    variable, take eight hues in a fixed order: blue, orange, aqua, yellow, magenta,
    green, violet and red. Neighbouring hues stay apart for readers with
    colour-vision deficiencies, as measured under simulated protanopia and
    deuteranopia (Machado et al. 2009), and so do any two of the first three. A
    scatter plot of more than three levels therefore also gives each level a marker
    of its own, so that colour is never the only cue. Up and down are red and blue,
    values shown for context are grey, and numbers take shades of blue. A figure that
    would need more than eight hues raises an error that says how to merge or select
    the categories.

References:
    Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK (2015). limma powers
    differential expression analyses for RNA-sequencing and microarray studies. Nucleic
    Acids Research 43, e47. doi:10.1093/nar/gkv007

    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Liu R, Holik AZ, Su S, Jansz N, Chen K, Leong HS, Blewitt ME, Asselin-Labat ML,
    Smyth GK, Ritchie ME (2015). Why weight? Modelling sample and observational level
    variability improves power in RNA-seq analyses. Nucleic Acids Research 43, e97.
    doi:10.1093/nar/gkv412

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Gower JC (1966). Some distance properties of latent root and vector methods used in
    multivariate analysis. Biometrika 53, 325-338. doi:10.1093/biomet/53.3-4.325

    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2

    Phipson B, Sim CB, Porrello ER, Hewitt AW, Powell J, Oshlack A (2022). propeller:
    testing for differences in cell type proportions in single cell data. Bioinformatics
    38, 4720-4726. doi:10.1093/bioinformatics/btac582

    Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and
    dispersion for RNA-seq data with DESeq2. Genome Biology 15, 550.
    doi:10.1186/s13059-014-0550-8

    Lund SP, Nettleton D, McCarthy DJ, Smyth GK (2012). Detecting differential
    expression in RNA-sequence data using quasi-likelihood with shrunken dispersion
    estimates. Statistical Applications in Genetics and Molecular Biology 11(5),
    article 8. doi:10.1515/1544-6115.1826

    Wu D, Smyth GK (2012). Camera: a competitive gene set test accounting for inter-gene
    correlation. Nucleic Acids Research 40, e133. doi:10.1093/nar/gks461

    Lewis S, Clarke M (2001). Forest plots: trying to see the wood and the trees. BMJ
    322, 1479-1480. doi:10.1136/bmj.322.7300.1479

    Viechtbauer W (2010). Conducting meta-analyses in R with the metafor package.
    Journal of Statistical Software 36(3), 1-48. doi:10.18637/jss.v036.i03

    Machado GM, Oliveira MM, Fernandes LAF (2009). A physiologically-based model for
    simulation of color vision deficiency. IEEE Transactions on Visualization and
    Computer Graphics 15, 1291-1298. doi:10.1109/TVCG.2009.113
"""

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import real_array, reject_masked, unmasked_array
from ._messages import listing, quoted
from ._options import (
    checked_choice,
    checked_flag,
    integral_number,
    positive_integer,
    real_number,
)
from .composition import CompositionResult
from .cube import Cube
from .differential_expression import DifferentialExpressionResult
from .meta_analysis import MetaAnalysisResult
from .normalization import filter_by_expression, log_cpm, tmm_factors
from .stats.deseq2 import DESeq2Dispersions, DESeq2Fit, DESeq2Results
from .stats.distributions import f_cdf, f_sf, norm_isf
from .stats.empirical_bayes import ModeratedFit, TreatFit
from .stats.multiple_testing import adjust_p_values
from .stats.quasi_likelihood import QuasiLikelihoodFit, QuasiLikelihoodTest
from .stats.voom import VoomResult
from .table import Table

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_barcode",
    "plot_cell_numbers",
    "plot_composition",
    "plot_dispersions",
    "plot_forest",
    "plot_gene_sets",
    "plot_md",
    "plot_mds",
    "plot_mean_variance",
    "plot_p_values",
    "plot_proportions",
    "plot_quasi_dispersions",
    "plot_residual_variances",
    "plot_sample_weights",
    "plot_significant_genes",
    "plot_volcano",
]

FloatArray = npt.NDArray[np.float64]

# Categorical hues in their fixed order: blue, orange, aqua, yellow, magenta, green,
# violet and red. Neighbours, and any two of the first three, stay distinguishable
# under simulated protanopia and deuteranopia.
_CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_BLUE, _ORANGE, _AQUA = _CATEGORICAL[:3]
_UP, _DOWN = _CATEGORICAL[7], _CATEGORICAL[0]
_SCATTER_HUES = 3  # the hues that stay apart in scatter plots, where any two can meet
_GREY = "#898781"  # values shown for context
_PALE = "#e1e0d9"  # entries left out, and the middle of a barcode
_INK = "#0b0b0b"
# A ramp of blue, light to dark, for numbers.
_BLUES = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#1c5cab",
    "#104281",
    "#0d366b",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*")
_DENSE = 8.0  # area of the points of scatter plots of genes, in points squared
_SPARSE = 48.0  # and of samples
# limma's plotMDS takes the 500 largest log fold changes, and edgeR's plotMDS of
# counts adds an average count of 2 before taking logs.
_MDS_TOP = 500
_MDS_PRIOR_COUNT = 2.0
_MIN_CELLS = 10  # as limma_voom leaves samples out
# plotSA marks as outliers the variances whose two-sided p-value, adjusted by Benjamini
# and Hochberg, is below this (found by probing limma 3.62.1).
_OUTLIER_LEVEL = 0.5
# barcodeplot's quantiles, which colour the ends of the bed, and the span of its
# enrichment worm.
_BARCODE_QUANTILE = math.sqrt(2.0)
_WORM_SPAN = 0.45
# barcodeplot draws the worm between these heights, above the bed (0 to 0.5) and the
# bars (0 to 1).
_WORM_BASE, _WORM_TOP = 1.1, 2.1
_NUMBERED_ENTRIES = 800  # the most heatmap entries whose numbers are written
_METHODS = {
    "reml": "REML",
    "dl": "DerSimonian-Laird",
    "pm": "Paule-Mandel",
    "fixed": "fixed effect",
}


# matplotlib.


def _module(name: str) -> Any:
    """A module of matplotlib, imported when a figure is drawn."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            "cytepool.plotting needs matplotlib, which is not installed: install "
            "CytePool with its 'plot' extra (pip install 'cytepool[plot]'), or "
            "install matplotlib itself"
        ) from error


def _check_axes(ax: object) -> None:
    """Raise TypeError unless ``ax`` is a matplotlib Axes or None."""
    if ax is None:
        return
    if not isinstance(ax, _module("matplotlib.axes").Axes):
        raise TypeError(
            "ax must be a matplotlib Axes, or None for a new figure, got "
            f"{type(ax).__name__}"
        )


def _axes(ax: "Axes | None", size: tuple[float, float]) -> "Axes":
    """``ax``, or the Axes of a new figure of ``size`` inches."""
    if ax is not None:
        return ax
    _, created = _module("matplotlib.pyplot").subplots(
        figsize=size, layout="constrained"
    )
    axes: Axes = created
    return axes


def _key(label: str, **properties: Any) -> Any:
    """A legend entry: a marker, or a line with ``linestyle``, or a header with
    ``marker=""``."""
    settings: dict[str, Any] = {"linestyle": "none", "marker": "o", "markersize": 7.0}
    settings.update(properties)
    return _module("matplotlib.lines").Line2D([], [], label=label, **settings)


def _patch(label: str, color: str, **properties: Any) -> Any:
    """A legend entry for a filled area."""
    return _module("matplotlib.patches").Patch(
        facecolor=color, label=label, **properties
    )


def _legend(ax: "Axes", handles: list[Any]) -> None:
    """A legend to the right of the plot; entries without a marker or line are
    headers, set in bold."""
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    for handle, text in zip(handles, legend.get_texts(), strict=True):
        header = getattr(handle, "get_marker", lambda: "o")() in ("", "None")
        if header and handle.get_linestyle() in ("None", "none", ""):
            text.set_fontweight("bold")


def _text_transform(ax: "Axes") -> Any:
    """x in data units and y in axes units, for text above the plot."""
    return _module("matplotlib.transforms").blended_transform_factory(
        ax.transData, ax.transAxes
    )


# Arguments.


def _labels(values: object, name: str, n: int) -> list[str]:
    """One label per sample, as text; refuses masked arrays and missing labels."""
    array = unmasked_array(values, name)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(
            f"{name} must hold one label per sample ({n}), got shape {array.shape}"
        )
    labels = []
    for value in array.tolist():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"{name} has a missing label; give every sample one")
        labels.append(str(value))
    return labels


def _fraction(value: Any, name: str) -> float:
    number = real_number(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {number}")
    return number


def _min_cells(value: Any) -> int:
    number = integral_number(value, "min_cells")
    if number < 0:
        raise ValueError(f"min_cells must be at least 0, got {number}")
    return number


def _optional_text(value: Any, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    reject_masked(value, name)
    raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")


@dataclass(frozen=True)
class _Samples:
    """The samples a figure shows, and what their labels are read from.

    Attributes:
        ids: The ids of the samples shown, in the data's order.
        rows: Their rows in the table that labels given one per sample follow.
        n_rows: The rows of that table.
        variables: Variables by name, one value per row, such as the columns of a
            cube's sample table.
        levels: The order of the levels of the factors among them.
        source: Where the variables come from, for messages.
    """

    ids: list[str]
    rows: list[int]
    n_rows: int
    variables: Mapping[str, Any] = field(default_factory=dict)
    levels: Mapping[str, Sequence[str]] = field(default_factory=dict)
    source: str = ""

    def labels(self, value: Any, name: str) -> tuple[list[str], list[str]]:
        """The labels of the samples shown, from a variable's name or one label per
        row, and their levels in order."""
        if isinstance(value, str):
            if value not in self.variables:
                if not self.source:
                    raise ValueError(
                        f"{name} names a variable, but the data have none; give one "
                        "label per sample"
                    )
                raise KeyError(
                    f"{name}: {self.source} has no variable {value!r}; variables "
                    f"are: {quoted(list(self.variables))}"
                )
            every = _labels(self.variables[value], f"{name} {value!r}", self.n_rows)
        else:
            every = _labels(value, name, self.n_rows)
        labels = [every[row] for row in self.rows]
        seen = list(dict.fromkeys(labels))
        known = (
            [str(level) for level in self.levels.get(value, ())]
            if isinstance(value, str)
            else []
        )
        ordered = [level for level in known if level in seen]
        return labels, ordered + [level for level in seen if level not in ordered]


def _cube_samples(cube: Cube, rows: npt.NDArray[np.intp]) -> _Samples:
    table = cube.samples
    return _Samples(
        ids=[str(value) for value in table["sample_id"][rows].tolist()],
        rows=rows.tolist(),
        n_rows=len(table),
        variables={name: table[name] for name in table.columns},
        source="the cube's sample table",
    )


def _check_variable(value: object, name: str) -> None:
    """Refuse a masked sample variable before anything is computed."""
    if not isinstance(value, str):
        reject_masked(value, name)


def _hues(levels: list[str], name: str, remedy: str) -> dict[str, str]:
    if len(levels) > len(_CATEGORICAL):
        raise ValueError(
            f"{name} has {len(levels)} levels, more than the {len(_CATEGORICAL)} "
            f"colours that stay distinguishable; {remedy}"
        )
    return dict(zip(levels, _CATEGORICAL, strict=False))


def _shapes(levels: list[str], name: str) -> dict[str, str]:
    if len(levels) > len(_MARKERS):
        raise ValueError(
            f"{name} has {len(levels)} levels, more than the {len(_MARKERS)} markers; "
            "give it as labels instead"
        )
    return dict(zip(levels, _MARKERS, strict=False))


# Samples.


def plot_mds(
    data: Any,
    *,
    group: str | None = None,
    color_by: str | Sequence[Any] | None = None,
    marker_by: str | Sequence[Any] | None = None,
    labels: bool | Sequence[Any] = False,
    top: int = _MDS_TOP,
    gene_selection: str = "pairwise",
    dimensions: Sequence[int] = (1, 2),
    min_cells: int | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot samples so that their distances are typical log2 fold changes between them.

    limma's ``plotMDS`` (Ritchie et al. 2015). The distance between two samples is
    their leading log2 fold change: the root mean square of the ``top`` largest
    absolute log2 fold changes between them, each pair over its own genes
    (``gene_selection="pairwise"``). With ``"common"``, or when ``top`` is at least the
    number of genes, it is the root mean square over the ``top`` genes with the largest
    variance over the samples, which makes the plot a principal component plot. The
    samples are placed by classical multidimensional scaling (Gower 1966): the
    coordinates of a dimension are an eigenvector of the doubly centred matrix of
    squared distances, -1/2 J D J, times the square root of its eigenvalue, and the
    axis labels give each eigenvalue's share of their sum, limma's
    ``var.explained``. A dimension's sign is arbitrary; here its largest coordinate is
    positive.

    A cube gives the pseudobulk samples of one group, prepared as the edgeR User's
    Guide prepares counts for ``plotMDS``: samples with fewer than ``min_cells`` cells
    or no counts are left out, as
    :func:`~cytepool.differential_expression.limma_voom` leaves them out; genes are
    kept by :func:`~cytepool.normalization.filter_by_expression` with the samples as
    one group; and the log2 counts per million are those that edgeR's ``plotMDS``
    takes of counts, with TMM normalization factors and an average count of 2 added
    (:func:`~cytepool.normalization.log_cpm`).

    Draws the samples as points (gid ``"samples"``), coloured by ``color_by`` and
    marked by ``marker_by``, with a legend of their levels, and labels the samples
    (gid ``"sample labels"``) when asked.

    Args:
        data: A :class:`~cytepool.cube.Cube`; a
            :class:`~cytepool.stats.voom.VoomResult`, whose log2 counts per million are
            used; or ``(n_samples, n_genes)`` log-expression values, finite.
        group: The group (cell type) of a cube to plot; it may be left out when the
            cube has one group. Only cubes take it.
        color_by: The variable whose levels colour the points: a column of the cube's
            sample table, or one label per sample (of the cube, for a cube). With more
            than three levels and no ``marker_by``, each level also gets a marker.
        marker_by: The variable whose levels mark the points, as ``color_by``; at most
            12 levels.
        labels: True to write each sample's id next to it (numbers from 1 for data
            other than a cube), or one label per sample.
        top: The number of genes behind each distance, at least 1.
        gene_selection: ``"pairwise"`` or ``"common"``.
        dimensions: The two dimensions to plot, counted from 1 in decreasing order of
            their eigenvalues.
        min_cells: The fewest cells a sample needs in the group to be plotted, 10 by
            default as for ``limma_voom``; for a cube that records cell numbers.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the values are not finite or not 2-D, too few samples remain for
            the dimensions, the group of a cube of several groups is not given,
            ``group`` or ``min_cells`` is given for data other than a cube, no gene
            passes the filter, a sample variable has the wrong length, a missing label
            or too many levels, or an option is out of range.
        KeyError: If the group, or a variable named, is not in the cube.
    """
    _check_axes(ax)
    count = positive_integer(top, "top")
    selection = checked_choice(gene_selection, "gene_selection", ("pairwise", "common"))
    first, second = _dimensions(dimensions)
    _check_variable(color_by, "color_by")
    _check_variable(marker_by, "marker_by")
    if not isinstance(labels, bool | np.bool_):
        reject_masked(labels, "labels")
    values, samples = _mds_values(data, group, min_cells)
    n_samples, n_genes = values.shape
    if n_samples <= max(first, second):
        raise ValueError(
            f"dimensions {first} and {second} need at least {max(first, second) + 1} "
            f"samples, got {n_samples}"
        )
    colors = None if color_by is None else samples.labels(color_by, "color_by")
    markers = None if marker_by is None else samples.labels(marker_by, "marker_by")
    remedy = "give it as marker_by, or as labels"
    hues = {} if colors is None else _hues(colors[1], "color_by", remedy)
    # More levels than hues that scatter plots keep apart get a marker each, too.
    shapes_follow_colors = (
        markers is None and colors is not None and len(colors[1]) > _SCATTER_HUES
    )
    if markers is not None:
        shapes, by_shape = _shapes(markers[1], "marker_by"), markers[0]
    elif colors is not None and shapes_follow_colors:
        shapes, by_shape = _shapes(colors[1], "color_by"), colors[0]
    else:
        shapes, by_shape = {}, None
    if isinstance(labels, bool | np.bool_):
        texts = list(samples.ids) if labels else None
    else:
        texts = samples.labels(labels, "labels")[0]
    pairwise = selection == "pairwise" and count < n_genes
    coordinates, eigenvalues = _mds(values, count, pairwise)
    share = np.maximum(eigenvalues / np.sum(eigenvalues), 0.0)
    x, y = coordinates[:, first - 1], coordinates[:, second - 1]

    ax = _axes(ax, (6.5, 4.5))
    color_of = [hues[level] for level in colors[0]] if colors else [_BLUE] * n_samples
    marker_of = [shapes[level] for level in by_shape] if by_shape else None
    marker_of = marker_of or [_MARKERS[0]] * n_samples
    for marker in dict.fromkeys(marker_of):
        rows = [i for i in range(n_samples) if marker_of[i] == marker]
        ax.scatter(
            x[rows],
            y[rows],
            c=[color_of[i] for i in rows],
            marker=marker,
            s=_SPARSE,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            gid="samples",
        )
    if texts is not None:
        for i, text in enumerate(texts):
            ax.annotate(
                text,
                (x[i], y[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                gid="sample labels",
            )
    axis = "Leading log2 fold change, dimension" if pairwise else "Principal component"
    ax.set_xlabel(f"{axis} {first} ({100 * share[first - 1]:.0f}%)")
    ax.set_ylabel(f"{axis} {second} ({100 * share[second - 1]:.0f}%)")
    handles: list[Any] = []
    if colors is not None:
        if isinstance(color_by, str):
            handles.append(_key(color_by, marker=""))
        for level in colors[1]:
            shape = shapes[level] if shapes_follow_colors else _MARKERS[0]
            handles.append(_key(level, marker=shape, color=hues[level]))
    if markers is not None:
        if isinstance(marker_by, str):
            handles.append(_key(marker_by, marker=""))
        for level in markers[1]:
            handles.append(_key(level, marker=shapes[level], color=_GREY))
    if handles:
        _legend(ax, handles)
    return ax


def _dimensions(value: Any) -> tuple[int, int]:
    reject_masked(value, "dimensions")
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(
            f"dimensions must be a pair of dimensions, such as (1, 2), got {value!r}"
        )
    first = positive_integer(value[0], "dimensions[0]")
    second = positive_integer(value[1], "dimensions[1]")
    if first == second:
        raise ValueError(f"dimensions must be two different dimensions, got {value!r}")
    return first, second


def _mds_values(data: Any, group: Any, min_cells: Any) -> tuple[FloatArray, _Samples]:
    """The log-expression values of the samples plotted, and the samples."""
    if not isinstance(data, Cube):
        if group is not None or min_cells is not None:
            given = "group" if group is not None else "min_cells"
            raise ValueError(
                f"{given} applies to a cube, and data is a {type(data).__name__}"
            )
        source = data.log_cpm if isinstance(data, VoomResult) else data
        values = real_array(unmasked_array(source, "data"), "data", boolean=False)
        if values.ndim != 2:
            raise ValueError(
                "data must be a Cube, a VoomResult or a 2-D array of shape "
                f"(n_samples, n_genes), got shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                "data must be finite; leave out the genes with missing values first"
            )
        n = values.shape[0]
        return values, _Samples([str(i + 1) for i in range(n)], list(range(n)), n)
    fewest = _MIN_CELLS if min_cells is None else _min_cells(min_cells)
    if group is None:
        if data.n_groups != 1:
            raise ValueError(
                f"the cube has {data.n_groups} groups; name the one to plot with "
                f"group, one of {quoted(data.groups['group_id'].tolist())}"
            )
        group = data.groups["group_id"][0]
    if not isinstance(group, str):
        reject_masked(group, "group")
        raise TypeError(f"group must be a group id (str), got {type(group).__name__}")
    counts = np.asarray(data.layer("counts", group), dtype=np.float64)
    used = counts.sum(axis=1) > 0.0
    if data.n_cells is not None:
        position = data.groups["group_id"].tolist().index(group)
        used &= data.n_cells[:, position] >= fewest
    rows = np.flatnonzero(used)
    if rows.size < 2:
        raise ValueError(
            f"{rows.size} sample(s) of group {group!r} have counts and at least "
            f"{fewest} cells; an MDS plot compares two or more"
        )
    kept = counts[rows]
    keep = filter_by_expression(kept, group=np.zeros(rows.size))
    if not keep.any():
        raise ValueError(
            f"no gene of group {group!r} passes filter_by_expression, so no distance "
            "can be computed"
        )
    kept = kept[:, keep]
    values = log_cpm(kept, norm_factors=tmm_factors(kept), prior_count=_MDS_PRIOR_COUNT)
    return values, _cube_samples(data, rows)


def _mds(values: FloatArray, top: int, pairwise: bool) -> tuple[FloatArray, FloatArray]:
    """Classical scaling of the rows: the coordinates of every dimension, in
    decreasing order of eigenvalue, and the eigenvalues."""
    n, n_genes = values.shape
    if not pairwise:
        top = min(top, n_genes)
        variances = np.var(values, axis=0, ddof=1)
        values = values[:, np.argsort(-variances, kind="stable")[:top]]
    squared = np.zeros((n, n))
    for i in range(n - 1):
        differences = (values[i + 1 :] - values[i]) ** 2
        if pairwise:
            differences = np.partition(differences, n_genes - top, axis=1)
            differences = differences[:, n_genes - top :]
        squared[i, i + 1 :] = np.mean(differences, axis=1)
    squared += squared.T
    centred = (
        squared
        - squared.mean(axis=0)
        - squared.mean(axis=1)[:, np.newaxis]
        + squared.mean()
    )
    eigenvalues, vectors = np.linalg.eigh(-0.5 * centred)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    coordinates: FloatArray = vectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    # The largest coordinate of each dimension is positive; of near ties, the first.
    magnitude = np.abs(coordinates)
    largest = np.argmax(magnitude >= magnitude.max(axis=0) * (1.0 - 1e-9), axis=0)
    signs = np.sign(coordinates[largest, np.arange(n)])
    coordinates *= np.where(signs == 0.0, 1.0, signs)
    return coordinates, eigenvalues


# Cubes.


def plot_cell_numbers(
    cube: Cube,
    *,
    min_cells: int | None = _MIN_CELLS,
    numbers: bool | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the number of cells pooled into each sample and group of a cube.

    A heatmap of ``Cube.n_cells``, one column per sample and one row per group (cell
    type) in the cube's order, shaded in blue on a logarithmic scale. Entries with
    fewer than ``min_cells`` cells, which
    :func:`~cytepool.differential_expression.limma_voom` leaves out of a group's fit
    with the same ``min_cells``, and entries without cells are pale grey, as the end
    of the colour bar shows.

    Draws the image (gid ``"cell numbers"``) and, when asked, the number of each entry
    (gid ``"numbers"``).

    Args:
        cube: A cube that records cell numbers.
        min_cells: The fewest cells an entry needs to be shaded, at least 0; None
            shades every entry with cells.
        numbers: Whether to write each entry's number; by default, when the cube has
            at most 800 entries.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``cube`` is not a Cube or an option has the wrong type.
        ValueError: If the cube does not record cell numbers, or ``min_cells`` is
            negative.
    """
    _check_axes(ax)
    if not isinstance(cube, Cube):
        raise TypeError(f"cube must be a Cube, got {type(cube).__name__}")
    fewest = None if min_cells is None else _min_cells(min_cells)
    if numbers is not None:
        numbers = checked_flag(numbers, "numbers")
    if cube.n_cells is None:
        raise ValueError(
            "the cube does not record cell numbers (bulk data), so there is nothing "
            "to plot"
        )
    cells = np.asarray(cube.n_cells, dtype=np.int64).T  # groups x samples
    lowest = max(1, fewest or 0)
    shown = np.where(cells >= lowest, cells, np.nan).astype(np.float64)
    write = cells.size <= _NUMBERED_ENTRIES if numbers is None else numbers
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_blues", _BLUES)
    ramp = ramp.with_extremes(bad=_PALE, under=_PALE)
    norm = colors.LogNorm(
        vmin=lowest, vmax=max(float(cells.max(initial=0)), lowest + 1)
    )

    n_groups, n_samples = cells.shape
    width = min(14.0, 2.5 + 0.32 * n_samples)
    ax = _axes(ax, (width, min(10.0, 1.6 + 0.32 * n_groups)))
    image = ax.imshow(shown, cmap=ramp, norm=norm, aspect="auto", interpolation="none")
    image.set_gid("cell numbers")
    ax.set_xticks(range(n_samples), cube.samples["sample_id"].tolist(), rotation=90)
    ax.set_yticks(range(n_groups), cube.groups["group_id"].tolist())
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if write:
        pale = colors.to_rgba(_PALE)
        for g in range(n_groups):
            for s in range(n_samples):
                face = ramp(norm(shown[g, s])) if cells[g, s] >= lowest else pale
                ax.text(
                    s,
                    g,
                    f"{cells[g, s]:d}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if _luminance(face) < 0.3 else _INK,
                    gid="numbers",
                )
    label = (
        "Cells (pale: no cells)"
        if lowest == 1
        else f"Cells (pale: fewer than {lowest})"
    )
    ax.figure.colorbar(image, ax=ax, extend="min", label=label)
    return ax


def _luminance(rgba: Sequence[float]) -> float:
    """The relative luminance of an sRGB colour."""
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


# Composition.


def plot_composition(
    data: Cube | CompositionResult,
    *,
    by: str | Sequence[Any] | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the cell-type composition of every sample as stacked bars.

    speckle's ``plotCellTypeProps``: one bar per sample, divided into the proportions
    of its cells in each group (cell type), the first group on top. A cube gives the
    proportions ``n_cells`` over each sample's cells in all groups, without the samples
    that have no cells; a :class:`~cytepool.composition.CompositionResult` gives those
    that :func:`~cytepool.composition.propeller` tested, of its samples used and groups
    tested. ``by`` groups the bars by a sample variable, such as the condition, whose
    levels are written above their bars.

    Draws the segments (gid ``"proportions"``), with a legend of the groups.

    Args:
        data: A cube that records cell numbers, or the result of ``propeller``.
        by: A sample variable: a column of the cube's sample table, a variable of the
            result's design, or one label per sample (of the cube, or of the result's
            sample table).
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``data`` is neither a Cube nor a CompositionResult, or ``by`` is
            or holds a numpy masked array.
        ValueError: If the cube does not record cell numbers, no sample has cells,
            there are more than eight groups (merge some with ``Cube.regroup``, or
            select some with ``Cube.select``, first), or ``by`` has the wrong length or
            a missing label.
        KeyError: If ``by`` names no variable of the data.
    """
    _check_axes(ax)
    _check_variable(by, "by")
    proportions, samples, group_ids = _composition(data)
    if len(group_ids) > len(_CATEGORICAL):
        raise ValueError(
            f"{len(group_ids)} groups are more than the {len(_CATEGORICAL)} colours "
            "that stay distinguishable; merge groups with Cube.regroup, or select some "
            "with Cube.select, first"
        )
    n_samples = len(samples.ids)
    if by is None:
        order, blocks = np.arange(n_samples), [("", 0, n_samples)]
    else:
        labels, levels = samples.labels(by, "by")
        order = np.array(
            sorted(range(n_samples), key=lambda i: levels.index(labels[i])), dtype=int
        )
        sorted_labels = [labels[i] for i in order]
        blocks = []
        for level in levels:
            start = sorted_labels.index(level)
            blocks.append((level, start, start + sorted_labels.count(level)))
    # A gap of one bar between the levels.
    positions = np.arange(n_samples, dtype=np.float64)
    for _, start, _ in blocks[1:]:
        positions[start:] += 1.0
    ax = _axes(ax, (min(14.0, 3.5 + 0.3 * n_samples), 4.0))
    shares = proportions[order]
    top = np.ones(n_samples)
    for g, group in enumerate(group_ids):
        bottom = top - shares[:, g]
        ax.bar(
            positions,
            shares[:, g],
            bottom=bottom,
            width=0.85,
            color=_CATEGORICAL[g],
            edgecolor="white",
            linewidth=0.8,
            label=group,
            gid="proportions",
        )
        top = bottom
    ax.set_xticks(positions, [samples.ids[i] for i in order], rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion of cells")
    ax.margins(x=0.01)
    if by is not None:
        transform = _text_transform(ax)
        for level, start, end in blocks:
            left, right = positions[start] - 0.42, positions[end - 1] + 0.42
            ax.plot(
                [left, right],
                [1.02, 1.02],
                transform=transform,
                color=_INK,
                linewidth=0.8,
                clip_on=False,
            )
            ax.text(
                (left + right) / 2.0,
                1.04,
                level,
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    _legend(ax, [_patch(group, _CATEGORICAL[g]) for g, group in enumerate(group_ids)])
    return ax


def _composition(data: Any) -> tuple[FloatArray, _Samples, list[str]]:
    """The proportions of every sample shown (rows) in every group (columns), the
    samples, and the group ids."""
    if isinstance(data, Cube):
        if data.n_cells is None:
            raise ValueError(
                "the cube does not record cell numbers (bulk data), so it has no "
                "composition"
            )
        cells = np.asarray(data.n_cells, dtype=np.float64)
        totals = cells.sum(axis=1)
        rows = np.flatnonzero(totals > 0.0)
        if rows.size == 0:
            raise ValueError("no sample of the cube has cells")
        proportions: FloatArray = cells[rows] / totals[rows, np.newaxis]
        group_ids = [str(value) for value in data.groups["group_id"].tolist()]
        return proportions, _cube_samples(data, rows), group_ids
    if not isinstance(data, CompositionResult):
        raise TypeError(
            f"data must be a Cube or a CompositionResult, got {type(data).__name__}"
        )
    table = data.proportions
    group_ids = list(dict.fromkeys(table["group_id"].tolist()))
    sample_ids = list(dict.fromkeys(table["sample_id"].tolist()))
    proportions = (
        np.asarray(table["proportion"], dtype=np.float64)
        .reshape(len(group_ids), len(sample_ids))
        .T
    )
    return proportions, _result_samples(data, sample_ids), group_ids


def _result_samples(result: CompositionResult, sample_ids: list[str]) -> _Samples:
    """The samples a composition result used, with the variables of its design."""
    every = result.samples["sample_id"].tolist()
    design = result.provenance["design"]
    position = {value: i for i, value in enumerate(design["samples"]["sample_id"])}
    variables = {
        name: [values[position[sample]] for sample in every]
        for name, values in design["samples"].items()
        if name != "sample_id"
    }
    return _Samples(
        ids=sample_ids,
        rows=[every.index(sample) for sample in sample_ids],
        n_rows=len(every),
        variables=variables,
        levels=design["levels"],
        source="the result's design",
    )


def plot_proportions(
    result: CompositionResult,
    *,
    by: str | Sequence[Any],
    contrast: str | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the proportion of every cell type in every sample, by condition, with the
    adjusted p-values of propeller.

    The figure of a :func:`~cytepool.composition.propeller` result (Phipson et al.
    2022): for every group (cell type) tested, the proportion of its cells in each
    sample used, as a point, side by side for the levels of ``by``, with a bar at the
    mean proportion of each level (speckle's ``PropMean``); above each group, the
    adjusted p-value (speckle's ``FDR``) of the F-test, or of a contrast.

    Draws the points of each level (gid ``"proportions"``), the means (gid
    ``"means"``) and the adjusted p-values (gid ``"adjusted p-values"``), with a legend
    of the levels.

    Args:
        result: The result of ``propeller``.
        by: A variable of the result's design, such as the condition, or one label per
            sample of the result's sample table.
        contrast: The contrast whose adjusted p-values are shown. By default, the
            F-test's when the result has one, else those of its only contrast.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a CompositionResult, or ``by`` or ``contrast``
            has the wrong type or is a numpy masked array.
        ValueError: If ``by`` has the wrong length, a missing label or more than eight
            levels, or no contrast is named and the result has several and no F-test.
        KeyError: If ``by`` names no variable of the design, or ``contrast`` was not
            tested.
    """
    _check_axes(ax)
    if not isinstance(result, CompositionResult):
        raise TypeError(
            f"result must be a CompositionResult, got {type(result).__name__}"
        )
    _check_variable(by, "by")
    name = _optional_text(contrast, "contrast")
    if name is not None:
        tests = result.table(name)
    elif result.f_test is not None:
        tests = result.f_test
    elif len(result.tables) == 1:
        tests = next(iter(result.tables.values()))
    else:
        raise ValueError(
            "the result tests several contrasts and no F-test; name one with contrast, "
            f"one of {quoted(list(result.tables))}"
        )
    proportions, samples, group_ids = _composition(result)
    labels, levels = samples.labels(by, "by")
    hues = _hues(levels, "by", "merge some of its levels")
    shapes = _shapes(levels, "by") if len(levels) > _SCATTER_HUES else {}
    adjusted = dict(
        zip(tests["group_id"].tolist(), tests["adjusted_p_value"].tolist(), strict=True)
    )

    n_groups, n_levels = len(group_ids), len(levels)
    ax = _axes(ax, (min(14.0, 2.5 + 0.9 * n_groups), 4.0))
    slot = 0.8 / n_levels
    for k, level in enumerate(levels):
        members = [i for i, label in enumerate(labels) if label == level]
        spread = (
            np.linspace(-0.3, 0.3, len(members)) * slot
            if len(members) > 1
            else np.zeros(1)
        )
        xs: list[float] = []
        ys: list[float] = []
        for g in range(n_groups):
            centre = g + (k - (n_levels - 1) / 2.0) * slot
            xs.extend(centre + spread)
            ys.extend(proportions[members, g])
            mean = float(np.mean(proportions[members, g]))
            ax.plot(
                [centre - 0.4 * slot, centre + 0.4 * slot],
                [mean, mean],
                color=_INK,
                linewidth=1.5,
                zorder=4,
                gid="means",
            )
        ax.scatter(
            xs,
            ys,
            color=hues[level],
            marker=shapes.get(level, _MARKERS[0]),
            s=30.0,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            gid="proportions",
        )
    transform = _text_transform(ax)
    for g, group in enumerate(group_ids):
        ax.text(
            g,
            1.01,
            f"adj. p {adjusted[group]:.2g}",
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=8,
            gid="adjusted p-values",
        )
    ax.set_xticks(range(n_groups), group_ids)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Proportion of cells")
    handles = [_key(by, marker="")] if isinstance(by, str) else []
    for level in levels:
        handles.append(
            _key(level, marker=shapes.get(level, _MARKERS[0]), color=hues[level])
        )
    _legend(ax, handles)
    return ax


# Mean-variance trends.


def plot_mean_variance(result: VoomResult, *, ax: "Axes | None" = None) -> "Axes":
    """Plot voom's mean-variance trend.

    voom's plot (Law et al. 2014), as limma's ``voom(plot = TRUE)`` draws it: the square
    root of every gene's residual standard deviation against its average log2 count
    (``VoomResult.sqrt_sd`` against ``mean_log_count``), with the LOWESS trend from
    which the precision weights are read (``sqrt_sd_trend``). Genes left out of the
    trend, such as those without counts, are not drawn.

    Draws the genes (gid ``"points"``) and the trend (gid ``"trend"``).

    Args:
        result: The result of :func:`~cytepool.stats.voom.voom` or
            :func:`~cytepool.stats.voom.voom_with_quality_weights`.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a VoomResult.
        ValueError: If no trend was fitted, because the design leaves no residual
            degrees of freedom.
    """
    _check_axes(ax)
    if not isinstance(result, VoomResult):
        raise TypeError(f"result must be a VoomResult, got {type(result).__name__}")
    shown = np.isfinite(result.sqrt_sd)
    if not shown.any():
        raise ValueError(
            "no mean-variance trend was fitted: the design leaves no residual degrees "
            "of freedom"
        )
    x = result.mean_log_count[shown]
    order = np.argsort(x, kind="stable")
    ax = _axes(ax, (6.5, 4.0))
    ax.scatter(
        x,
        result.sqrt_sd[shown],
        s=_DENSE,
        color=_GREY,
        alpha=0.6,
        linewidths=0,
        gid="points",
    )
    ax.plot(
        x[order],
        result.sqrt_sd_trend[shown][order],
        color=_ORANGE,
        linewidth=2,
        gid="trend",
    )
    ax.set_xlabel("Average log2 count")
    ax.set_ylabel("Square root of residual standard deviation")
    _legend(
        ax,
        [
            _key("Genes", color=_GREY),
            _key(
                f"LOWESS trend (span {result.span:.2f})",
                marker="",
                linestyle="-",
                color=_ORANGE,
                linewidth=2,
            ),
        ],
    )
    return ax


def plot_residual_variances(
    fit: ModeratedFit | TreatFit, *, ax: "Axes | None" = None
) -> "Axes":
    """Plot the residual standard deviations of the genes and the prior they were
    moderated towards.

    limma's ``plotSA``: the square root of every gene's residual standard deviation
    against its average log-expression, with the square root of the prior standard
    deviation (Smyth 2004), a line at its value or, with a trend, a curve. After robust
    moderation (Phipson et al. 2016), the variances that are outliers are marked: those
    whose two-sided p-value as a variance of the prior, from the F distribution on the
    gene's residual and the prior's degrees of freedom (those of the genes that are not
    outliers), is below 0.5 once adjusted by Benjamini and Hochberg (1995), as limma
    marks them (found by probing).

    Draws the genes (gid ``"points"``), the outliers (gid ``"outliers"``) and the prior
    (gid ``"prior"``).

    Args:
        fit: The result of :func:`~cytepool.stats.empirical_bayes.moderated_statistics`
            or :func:`~cytepool.stats.empirical_bayes.treat`.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``fit`` is neither a ModeratedFit nor a TreatFit.
        ValueError: If no gene has a residual standard deviation.
    """
    _check_axes(ax)
    if not isinstance(fit, ModeratedFit | TreatFit):
        raise TypeError(
            f"fit must be a ModeratedFit or a TreatFit, got {type(fit).__name__}"
        )
    linear = fit.fit
    sigma, x, df = linear.sigma, linear.mean_expression, linear.df_residual
    shown = np.isfinite(sigma) & np.isfinite(x) & (df > 0.0)
    if not shown.any():
        raise ValueError("no gene has a residual standard deviation to plot")
    prior = np.broadcast_to(
        np.asarray(fit.prior_variance, dtype=np.float64), sigma.shape
    )
    outlier = np.zeros(sigma.shape, dtype=bool)
    robust = np.ndim(fit.prior_df) == 1
    if robust:
        ratio = np.where(shown, sigma**2 / prior, np.nan)
        dfd = float(np.max(np.asarray(fit.prior_df)[shown]))
        lower, upper = f_cdf(ratio, df, dfd), f_sf(ratio, df, dfd)
        tails = np.where(shown, np.minimum(1.0, 2.0 * np.minimum(lower, upper)), np.nan)
        with np.errstate(invalid="ignore"):
            outlier = adjust_p_values(tails, "bh") < _OUTLIER_LEVEL
    normal = shown & ~outlier
    ax = _axes(ax, (6.5, 4.0))
    ax.scatter(
        x[normal],
        np.sqrt(sigma[normal]),
        s=_DENSE,
        color=_GREY,
        alpha=0.6,
        linewidths=0,
        gid="points",
    )
    handles = [_key("Genes", color=_GREY)]
    if outlier.any():
        ax.scatter(
            x[outlier],
            np.sqrt(sigma[outlier]),
            s=2 * _DENSE,
            color=_AQUA,
            linewidths=0,
            gid="outliers",
        )
        handles.append(_key("Outlier variances", color=_AQUA))
    level = np.sqrt(np.sqrt(prior))
    if np.ndim(fit.prior_variance) == 0:
        ax.axhline(level[0], color=_ORANGE, linewidth=2, gid="prior")
    else:
        trend = shown & np.isfinite(level)
        order = np.argsort(x[trend], kind="stable")
        ax.plot(
            x[trend][order],
            level[trend][order],
            color=_ORANGE,
            linewidth=2,
            gid="prior",
        )
    handles.append(_key("Prior", marker="", linestyle="-", color=_ORANGE, linewidth=2))
    ax.set_xlabel("Average log-expression")
    ax.set_ylabel("Square root of residual standard deviation")
    _legend(ax, handles)
    return ax


# Sample weights.


def plot_sample_weights(
    weights: Any,
    *,
    group: str | None = None,
    labels: Sequence[Any] | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the weight of every sample, with a line at the weight of equal samples.

    The sample-weight panel of limma's ``voomWithQualityWeights(plot = TRUE)`` (Liu et
    al. 2015): a bar per sample, and a dashed line at 1. Samples below the line follow
    the linear model less closely, and count for less in the fits.

    Draws the bars (gid ``"weights"``) and the line (gid ``"equal weight"``).

    Args:
        weights: The ``sample_weights`` of a :class:`~cytepool.stats.voom.VoomResult`
            from :func:`~cytepool.stats.voom.voom_with_quality_weights`; the quality
            weights, or prior weights, of the samples used in one group of a
            :class:`~cytepool.differential_expression.DifferentialExpressionResult`;
            or the weights, positive and finite, such as those of
            :func:`~cytepool.stats.sample_weights.sample_quality_weights`.
        group: The group of a differential-expression result to plot; it may be left
            out when the result has one group.
        labels: One label per sample shown; by default the sample ids of a
            differential-expression result, and numbers from 1 otherwise.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the result holds no sample weights, the group of a result of
            several groups is not given, the group was not fitted, the weights are not
            positive and finite, or ``labels`` has the wrong length.
        KeyError: If the group was not analysed.
    """
    _check_axes(ax)
    name = _optional_text(group, "group")
    if labels is not None:
        reject_masked(labels, "labels")
    kind = "Sample weight"
    if isinstance(weights, VoomResult):
        if name is not None:
            raise ValueError("group applies to a differential-expression result")
        if weights.sample_weights is None:
            raise ValueError(
                "the VoomResult holds no sample weights; voom_with_quality_weights "
                "estimates them"
            )
        values = np.asarray(weights.sample_weights, dtype=np.float64)
        names = [str(i + 1) for i in range(values.size)]
        kind = "Sample quality weight"
    elif isinstance(weights, DifferentialExpressionResult):
        values, names, kind = _weights_of_group(weights, name)
    else:
        if name is not None:
            raise ValueError("group applies to a differential-expression result")
        values = real_array(
            unmasked_array(weights, "weights"), "weights", boolean=False
        )
        if values.ndim != 1:
            raise ValueError(f"weights must be a 1-D array, got shape {values.shape}")
        names = [str(i + 1) for i in range(values.size)]
    if values.size == 0 or not np.all(np.isfinite(values) & (values > 0.0)):
        raise ValueError("the sample weights must be positive and finite")
    if labels is not None:
        names = _labels(labels, "labels", values.size)
    ax = _axes(ax, (min(14.0, 2.5 + 0.35 * values.size), 3.5))
    positions = np.arange(values.size)
    ax.bar(positions, values, width=0.7, color=_BLUE, gid="weights")
    ax.axhline(1.0, color=_INK, linewidth=1, linestyle="--", gid="equal weight")
    ax.set_xticks(positions, names, rotation=90 if max(map(len, names)) > 3 else 0)
    ax.set_ylabel(kind)
    return ax


def _weights_of_group(
    result: DifferentialExpressionResult, group: str | None
) -> tuple[FloatArray, list[str], str]:
    """The sample weights of the samples used in one group, their ids and the kind."""
    samples = result.samples
    if "quality_weight" in samples:
        column, kind = "quality_weight", "Sample quality weight"
    elif "prior_weight" in samples:
        column, kind = "prior_weight", "Prior weight"
    else:
        raise ValueError(
            "the result holds no sample weights; limma_voom estimates them with "
            "quality_weights=True, and takes cell numbers as prior weights with "
            "prior_weights='n_cells'"
        )
    if group is None:
        if len(result.groups) != 1:
            raise ValueError(
                f"the result has {len(result.groups)} groups; name the one to plot "
                f"with group, one of {quoted(list(result.groups))}"
            )
        group = result.groups[0]
    if group not in result.groups:
        raise KeyError(
            f"no group {group!r} was analysed; groups are: "
            f"{quoted(list(result.groups))}"
        )
    rows = (samples["group_id"] == group) & samples["used"]
    values = np.asarray(samples[column][rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"group {group!r} was not fitted, so it has no sample weights")
    return values, [str(value) for value in samples["sample_id"][rows].tolist()], kind


# Differential expression.


@dataclass(frozen=True)
class _Tests:
    """The columns of a test that the figures of differential expression read."""

    log_fold_change: FloatArray | None
    abundance: FloatArray | None
    abundance_label: str
    log_abundance: bool  # whether the abundance is drawn on a logarithmic axis
    p_value: FloatArray | None
    adjusted_p_value: FloatArray | None
    names: list[str] | None


_TESTED = (
    "a table of a DifferentialExpressionResult or MetaAnalysisResult, a ModeratedFit, "
    "a TreatFit, a QuasiLikelihoodTest or DESeq2Results"
)


def _tests(result: Any, coefficient: Any) -> _Tests:
    """The columns of a table or test result, for one coefficient or contrast."""
    if isinstance(result, Table):
        _no_coefficient(coefficient, "a table")

        def column(name: str) -> FloatArray | None:
            return (
                np.asarray(result[name], dtype=np.float64) if name in result else None
            )

        names = None
        if "feature_id" in result:
            ids = result["feature_id"].tolist()
            symbols = (
                result["symbol"].tolist() if "symbol" in result else [None] * len(ids)
            )
            names = [
                str(s) if s is not None else str(i)
                for s, i in zip(symbols, ids, strict=True)
            ]
        return _Tests(
            column("log2_fold_change"),
            column("average_log2_cpm"),
            "Average log2 CPM",
            False,
            column("p_value"),
            column("adjusted_p_value"),
            names,
        )
    if isinstance(result, ModeratedFit | TreatFit):
        j = _coefficient(coefficient, result.fit.coefficient_names)
        p = np.asarray(result.p_value[:, j], dtype=np.float64)
        return _Tests(
            np.asarray(result.fit.coefficients[:, j], dtype=np.float64),
            np.asarray(result.fit.mean_expression, dtype=np.float64),
            "Average log-expression",
            False,
            p,
            adjust_p_values(p, "bh"),
            None,
        )
    if isinstance(result, QuasiLikelihoodTest):
        j = _coefficient(coefficient, result.names)
        return _Tests(
            np.asarray(result.log_fold_change[:, j], dtype=np.float64),
            np.asarray(result.average_log_cpm, dtype=np.float64),
            "Average log2 CPM",
            False,
            np.asarray(result.p_value, dtype=np.float64),
            np.asarray(result.adjusted_p_value, dtype=np.float64),
            None,
        )
    if isinstance(result, DESeq2Results):
        _no_coefficient(coefficient, "DESeq2Results")
        return _Tests(
            np.asarray(result.log2_fold_change, dtype=np.float64),
            np.asarray(result.base_mean, dtype=np.float64),
            "Mean of normalized counts",
            True,
            np.asarray(result.p_value, dtype=np.float64),
            np.asarray(result.adjusted_p_value, dtype=np.float64),
            None,
        )
    raise TypeError(f"result must be {_TESTED}, got {type(result).__name__}")


def _no_coefficient(coefficient: Any, what: str) -> None:
    if coefficient is not None:
        reject_masked(coefficient, "coefficient")
        raise ValueError(
            f"coefficient applies to fits and tests of several; {what} holds one"
        )


def _coefficient(value: Any, names: Sequence[str]) -> int:
    """The position of a coefficient or contrast, given by name or position."""
    if value is None:
        if len(names) != 1:
            raise ValueError(
                f"the result holds {len(names)} coefficients or contrasts; name one "
                f"with coefficient, one of {quoted(list(names))}"
            )
        return 0
    reject_masked(value, "coefficient")
    if isinstance(value, str):
        if value not in names:
            raise KeyError(
                f"no coefficient or contrast {value!r}; they are: {quoted(list(names))}"
            )
        return list(names).index(value)
    position = integral_number(value, "coefficient", expected="a name or a position")
    if not -len(names) <= position < len(names):
        raise IndexError(
            f"coefficient position {position} is out of range for {len(names)} "
            "coefficients"
        )
    return position % len(names)


def _require(values: FloatArray | None, column: str, function: str) -> FloatArray:
    if values is None:
        raise ValueError(
            f"{function} needs the column {column!r}, which the table lacks"
        )
    return values


def _directions(
    log_fold_change: FloatArray, adjusted: FloatArray, alpha: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """The genes significant up and down at ``alpha``."""
    with np.errstate(invalid="ignore"):
        significant = adjusted < alpha
        return significant & (log_fold_change > 0.0), significant & (
            log_fold_change < 0.0
        )


def _significance_keys(n_up: int, n_down: int, n_rest: int) -> list[Any]:
    return [
        _key(f"Up ({n_up:,})", color=_UP),
        _key(f"Down ({n_down:,})", color=_DOWN),
        _key(f"Not significant ({n_rest:,})", color=_GREY),
    ]


def _draw_significance(
    ax: "Axes", x: FloatArray, y: FloatArray, up: Any, down: Any, rest: Any
) -> None:
    ax.scatter(
        x[rest],
        y[rest],
        s=_DENSE,
        color=_GREY,
        alpha=0.5,
        linewidths=0,
        gid="not significant",
    )
    ax.scatter(x[down], y[down], s=_DENSE, color=_DOWN, linewidths=0, gid="down")
    ax.scatter(x[up], y[up], s=_DENSE, color=_UP, linewidths=0, gid="up")


def plot_md(
    result: Any,
    *,
    coefficient: str | int | None = None,
    alpha: float = 0.05,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot log fold changes against average expression, the significant genes
    highlighted.

    limma's mean-difference plot of a fit, ``plotMD`` (an MA plot), with the genes
    whose adjusted p-value is below ``alpha`` coloured by direction, as
    ``plotMD(fit, status = decideTests(fit))`` colours them. The average expression is
    the average log2 counts per million of a table of
    :class:`~cytepool.differential_expression.DifferentialExpressionResult` or of a
    :class:`~cytepool.stats.quasi_likelihood.QuasiLikelihoodTest`, the average
    log-expression of a fit, or, for
    :class:`~cytepool.stats.deseq2.DESeq2Results`, the mean of the normalized counts
    on a logarithmic axis, as DESeq2's ``plotMA``. The adjusted p-values of a fit are
    Benjamini and Hochberg's over its genes, as ``topTable`` gives them; those of a
    quasi-likelihood test of several contrasts are of the test of all of them, whichever
    contrast's log fold changes are drawn.

    Draws the genes up (gid ``"up"``), down (gid ``"down"``) and not significant (gid
    ``"not significant"``), with a line at zero.

    Args:
        result: A table of a differential-expression result (with the columns
            ``log2_fold_change``, ``average_log2_cpm`` and ``adjusted_p_value``), a
            ModeratedFit, a TreatFit, a QuasiLikelihoodTest or DESeq2Results.
        coefficient: The coefficient or contrast of a fit or test of several, by name
            or position.
        alpha: The adjusted p-value below which genes are highlighted, in (0, 1).
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` or an option has the wrong type.
        ValueError: If a table lacks a column, ``coefficient`` is needed and not given
            or given for a single test, or ``alpha`` is not in (0, 1).
        KeyError, IndexError: If ``coefficient`` names or points to no coefficient.
    """
    _check_axes(ax)
    tests = _tests(result, coefficient)
    level = _fraction(alpha, "alpha")
    y = _require(tests.log_fold_change, "log2_fold_change", "plot_md")
    x = _require(tests.abundance, "average_log2_cpm", "plot_md")
    adjusted = _require(tests.adjusted_p_value, "adjusted_p_value", "plot_md")
    shown = np.isfinite(x) & np.isfinite(y)
    if tests.log_abundance:
        shown &= x > 0.0
    up, down = _directions(y, adjusted, level)
    up, down = up & shown, down & shown
    rest = shown & ~up & ~down
    ax = _axes(ax, (6.5, 4.0))
    _draw_significance(ax, x, y, up, down, rest)
    ax.axhline(0.0, color=_INK, linewidth=0.8, zorder=0)
    if tests.log_abundance:
        ax.set_xscale("log")
    ax.set_xlabel(tests.abundance_label)
    ax.set_ylabel("log2 fold change")
    _legend(ax, _significance_keys(int(up.sum()), int(down.sum()), int(rest.sum())))
    return ax


def plot_volcano(
    result: Any,
    *,
    coefficient: str | int | None = None,
    alpha: float = 0.05,
    top: int = 10,
    labels: Sequence[Any] | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot -log10 p-values against log fold changes, labelling the top genes.

    limma's ``volcanoplot``, with the genes whose adjusted p-value is below ``alpha``
    coloured by direction, as in :func:`plot_md`, and the ``top`` genes with the
    smallest p-values labelled: by their symbol, or else their id, in a table, or by
    ``labels``. A p-value of 0 is drawn at the smallest positive p-value.

    Draws the genes up (gid ``"up"``), down (gid ``"down"``) and not significant (gid
    ``"not significant"``), and the labels (gid ``"gene labels"``).

    Args:
        result: A table of a differential-expression or meta-analysis result (with
            the columns ``log2_fold_change``, ``p_value`` and ``adjusted_p_value``), a
            ModeratedFit, a TreatFit, a QuasiLikelihoodTest or DESeq2Results.
        coefficient: The coefficient or contrast of a fit or test of several, by name
            or position.
        alpha: The adjusted p-value below which genes are highlighted, in (0, 1).
        top: How many genes to label, at least 0.
        labels: One name per gene; needed to label the genes of fits and tests, which
            hold no names.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or ``labels`` is or holds a numpy
            masked array.
        ValueError: As :func:`plot_md`, or if ``labels`` has the wrong length or
            ``top`` is negative.
        KeyError, IndexError: If ``coefficient`` names or points to no coefficient.
    """
    _check_axes(ax)
    tests = _tests(result, coefficient)
    level = _fraction(alpha, "alpha")
    count = integral_number(top, "top")
    if count < 0:
        raise ValueError(f"top must be at least 0, got {count}")
    if labels is not None:
        reject_masked(labels, "labels")
    x = _require(tests.log_fold_change, "log2_fold_change", "plot_volcano")
    p = _require(tests.p_value, "p_value", "plot_volcano")
    adjusted = _require(tests.adjusted_p_value, "adjusted_p_value", "plot_volcano")
    names = tests.names if labels is None else _gene_labels(labels, x.size)
    shown = np.isfinite(x) & np.isfinite(p)
    positive = p[shown & (p > 0.0)]
    floor = float(positive.min()) if positive.size else 1.0
    y = np.full(p.shape, np.nan)
    y[shown] = -np.log10(np.maximum(p[shown], floor))
    up, down = _directions(x, adjusted, level)
    up, down = up & shown, down & shown
    rest = shown & ~up & ~down
    ax = _axes(ax, (6.5, 4.5))
    _draw_significance(ax, x, y, up, down, rest)
    if names is not None and count:
        candidates = np.flatnonzero(shown)
        chosen = candidates[np.argsort(p[candidates], kind="stable")[:count]]
        _label_points(ax, x, y, names, chosen)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 p-value")
    _legend(ax, _significance_keys(int(up.sum()), int(down.sum()), int(rest.sum())))
    return ax


def _label_points(
    ax: "Axes", x: FloatArray, y: FloatArray, names: list[str], chosen: Any
) -> None:
    """Label points towards the middle of the plot (x = 0), each side's labels moved
    apart vertically where they would overlap, with a line to a label that moved."""
    low, high = ax.get_xlim()
    bottom, top = ax.get_ylim()
    # The size of the plot in points, and the span of 6 points across and of a line
    # of text, 11 points, down.
    width, height = (
        ax.bbox.width * 72.0 / ax.figure.dpi,
        ax.bbox.height * 72.0 / ax.figure.dpi,
    )
    across = (high - low) * 6.0 / width
    line = (top - bottom) * 11.0 / height
    for right in (True, False):
        side = [i for i in chosen if (x[i] > 0.0) == right]
        placed: list[float] = []
        for i in sorted(side, key=lambda i: -y[i]):
            level = y[i] if not placed else min(y[i], placed[-1] - line)
            placed.append(level)
            moved = abs(level - y[i]) > 0.1 * line
            ax.annotate(
                names[i],
                (x[i], y[i]),
                xytext=(x[i] - across if right else x[i] + across, level),
                textcoords="data",
                ha="right" if right else "left",
                va="center",
                fontsize=8,
                arrowprops=(
                    {"arrowstyle": "-", "color": _GREY, "linewidth": 0.6}
                    if moved
                    else None
                ),
                gid="gene labels",
            )


def _gene_labels(labels: Any, n: int) -> list[str]:
    array = unmasked_array(labels, "labels")
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(
            f"labels must hold one name per gene ({n}), got shape {array.shape}"
        )
    return [str(value) for value in array.tolist()]


def plot_significant_genes(
    result: DifferentialExpressionResult,
    *,
    contrast: str | None = None,
    alpha: float = 0.05,
    across_groups: bool = False,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot how many genes go up and down in every group of a differential-expression
    result.

    For every group (cell type) of a
    :class:`~cytepool.differential_expression.DifferentialExpressionResult`, in its
    order from the top, the genes whose adjusted p-value for ``contrast`` is below
    ``alpha``: a bar to the right for those up and to the left for those down, each
    labelled with its number. A group that was not fitted, or cannot estimate the
    contrast, has no bars, and its label says so. ``across_groups`` counts by the
    p-values adjusted over every group (``adjusted_p_value_across_groups``, which
    :func:`~cytepool.differential_expression.limma_voom` adds with
    ``adjust_across_groups=True``) instead of within each group.

    Draws the bars of genes up (gid ``"up"``) and down (gid ``"down"``), and their
    numbers (gid ``"numbers"``).

    Args:
        result: The result of ``limma_voom``, ``limma_trend`` or ``dream_voom``.
        contrast: The contrast; it may be left out when the result tests one.
        alpha: The adjusted p-value below which genes count, in (0, 1).
        across_groups: Whether to count by the p-values adjusted across groups.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a DifferentialExpressionResult or an option
            has the wrong type.
        ValueError: If no contrast is named and the result tests several,
            ``alpha`` is not in (0, 1), or the p-values adjusted across groups are
            asked for and the result has none.
        KeyError: If the contrast was not tested.
    """
    _check_axes(ax)
    if not isinstance(result, DifferentialExpressionResult):
        raise TypeError(
            "result must be a DifferentialExpressionResult, got "
            f"{type(result).__name__}"
        )
    name = _optional_text(contrast, "contrast")
    level = _fraction(alpha, "alpha")
    across = checked_flag(across_groups, "across_groups")
    names = list(result.contrasts.names)
    if name is None:
        if len(names) != 1:
            raise ValueError(
                f"the result tests {len(names)} contrasts; name one with contrast, one "
                f"of {quoted(names)}"
            )
        name = names[0]
    elif name not in names:
        raise KeyError(
            f"no contrast {name!r} was tested; contrasts are: {quoted(names)}"
        )
    column = "adjusted_p_value_across_groups" if across else "adjusted_p_value"
    groups = list(result.groups)
    up = np.zeros(len(groups), dtype=np.int64)
    down = np.zeros(len(groups), dtype=np.int64)
    tested = np.zeros(len(groups), dtype=bool)
    labels = []
    for g, group in enumerate(groups):
        table = result.tables.get((group, name))
        tested[g] = table is not None
        if table is None:
            entry = result.summary["status"][g]
            labels.append(
                f"{group} (not fitted)"
                if entry != "fitted"
                else f"{group} (not estimable)"
            )
            continue
        if column not in table:
            raise ValueError(
                "the result has no p-values adjusted across groups; limma_voom adds "
                "them with adjust_across_groups=True"
            )
        lfc = np.asarray(table["log2_fold_change"], dtype=np.float64)
        adjusted = np.asarray(table[column], dtype=np.float64)
        ups, downs = _directions(lfc, adjusted, level)
        up[g], down[g] = int(ups.sum()), int(downs.sum())
        labels.append(group)
    positions = np.arange(len(groups))[::-1]
    ax = _axes(ax, (6.5, min(12.0, 1.4 + 0.45 * len(groups))))
    ax.barh(positions, up, height=0.6, color=_UP, gid="up")
    ax.barh(positions, -down, height=0.6, color=_DOWN, gid="down")
    ax.axvline(0.0, color=_INK, linewidth=0.8)
    reach = max(int(up.max(initial=0)), int(down.max(initial=0)), 1)
    gap = 0.02 * reach
    for y, n_up, n_down in zip(
        positions[tested], up[tested], down[tested], strict=True
    ):
        for value, side in ((n_up, 1.0), (n_down, -1.0)):
            ax.text(
                side * (value + gap),
                y,
                f"{value:,}",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=8,
                gid="numbers",
            )
    ax.set_yticks(positions, labels)
    ax.set_xlim(-1.2 * reach, 1.2 * reach)
    # Genes down are drawn to the left; the axis counts them as numbers of genes.
    ax.xaxis.set_major_formatter(
        _module("matplotlib.ticker").FuncFormatter(lambda value, _: f"{abs(value):g}")
    )
    adjusted_over = "over every group" if across else "within each group"
    ax.set_xlabel(f"Genes at adjusted p < {level:g} ({adjusted_over})")
    _legend(ax, [_patch("Up", _UP), _patch("Down", _DOWN)])
    return ax


def plot_p_values(
    result: Any,
    *,
    coefficient: str | int | None = None,
    bins: int = 20,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot a histogram of p-values, with the count expected without effects.

    Without differential expression, p-values are uniform on [0, 1], so every bin
    holds about as many tests as the dashed line: a peak at small p-values shows tests
    with effects, and a histogram that rises towards 1, or a peak that holds most
    tests, shows that the model does not fit the data, as when cells are tested as if
    they were independent replicates.

    Draws the bars (gid ``"p-values"``) and the expected count (gid ``"uniform"``).

    Args:
        result: The p-values of a table (its ``p_value`` column), a fit or a test, as
            for :func:`plot_md`, or the p-values themselves, in [0, 1]; NaN marks a
            missing one, which is left out.
        coefficient: The coefficient or contrast of a fit or test of several, by name
            or position.
        bins: The number of bins of equal width over [0, 1], at least 1.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or the p-values are a numpy
            masked array.
        ValueError: If there is no p-value, one lies outside [0, 1], or a table has no
            ``p_value`` column.
    """
    _check_axes(ax)
    n_bins = positive_integer(bins, "bins")
    if isinstance(
        result, Table | ModeratedFit | TreatFit | QuasiLikelihoodTest | DESeq2Results
    ):
        p = _require(_tests(result, coefficient).p_value, "p_value", "plot_p_values")
    else:
        if coefficient is not None:
            _no_coefficient(coefficient, "an array of p-values")
        p = real_array(
            unmasked_array(result, "result", missing="NaN"), "result", boolean=False
        )
        if p.ndim != 1:
            raise ValueError(f"p-values must be a 1-D array, got shape {p.shape}")
    p = p[~np.isnan(p)]
    if p.size == 0:
        raise ValueError("there is no p-value to plot")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    ax = _axes(ax, (6.5, 4.0))
    counts, edges = np.histogram(p, bins=n_bins, range=(0.0, 1.0))
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=_BLUE,
        edgecolor="white",
        linewidth=0.8,
        gid="p-values",
    )
    ax.axhline(p.size / n_bins, color=_INK, linewidth=1, linestyle="--", gid="uniform")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Tests")
    _legend(
        ax,
        [
            _patch(f"p-values ({p.size:,})", _BLUE),
            _key("Expected without effects", marker="", linestyle="--", color=_INK),
        ],
    )
    return ax


# Gene sets.


def plot_barcode(
    statistics: Any,
    gene_set: Any,
    *,
    gene_ids: Any = None,
    span: float = _WORM_SPAN,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot where the genes of a set fall among all genes ranked by a statistic.

    limma's ``barcodeplot`` (Ritchie et al. 2015) for one set: the genes ranked from
    the smallest statistic, on the left, to the largest, as a bed shaded at the ends
    where the statistic is below -sqrt(2) or above sqrt(2); a bar at the rank of every
    gene of the set; and above them the enrichment worm, the relative enrichment of
    the set along the ranking: the moving average of the set's indicator over the
    ranked genes, with tricube weights over a window of ``span`` of the genes (limma's
    ``tricubeMovingAverage``, whose weights were found by probing), divided by the
    set's share of all genes. A worm above 1 (the dashed line) at the right shows a
    set enriched among the genes with large statistics. The worm is drawn above the
    bars, scaled to its largest value, which the axis gives, as limma draws it. The
    x-axis gives the statistic at five ranks.

    Draws the bed (gid ``"bed"``), the bars (gid ``"set"``) and the worm (gid
    ``"worm"``).

    Args:
        statistics: A table of a differential-expression result, whose ``t`` column
            (or ``statistic``, for a meta-analysis) ranks its genes, named by its
            ``feature_id``; or one statistic per gene, finite.
        gene_set: The genes of the set: ids (of the table's genes, or of ``gene_ids``),
            0-based positions, or a boolean mask over the genes. Ids that are not
            among the genes are left out, as limma's ``ids2indices`` leaves them.
        gene_ids: One id per gene, unique, to name the genes of ``gene_set`` by id when
            ``statistics`` is not a table.
        span: The share of the genes in each window of the worm, in (0, 1].
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the statistics are not finite and 1-D, fewer than two, the ids
            are not unique or have the wrong length, the set holds no gene, every gene,
            or a position out of range, or ``span`` is not in (0, 1].
    """
    _check_axes(ax)
    window = real_number(span, "span")
    if not 0.0 < window <= 1.0:
        raise ValueError(f"span must be in (0, 1], got {window}")
    reject_masked(gene_set, "gene_set")
    label = "Statistic"
    if isinstance(statistics, Table):
        if gene_ids is not None:
            reject_masked(gene_ids, "gene_ids")
            raise ValueError("gene_ids applies to statistics given as an array")
        column = "t" if "t" in statistics else "statistic"
        if column not in statistics or "feature_id" not in statistics:
            raise ValueError(
                "a table of statistics needs a 't' or 'statistic' column and a "
                "'feature_id' column"
            )
        values = np.asarray(statistics[column], dtype=np.float64)
        ids: list[Any] | None = statistics["feature_id"].tolist()
        label = "Moderated t-statistic" if column == "t" else "Statistic"
    else:
        values = real_array(
            unmasked_array(statistics, "statistics"), "statistics", boolean=False
        )
        ids = (
            None if gene_ids is None else unmasked_array(gene_ids, "gene_ids").tolist()
        )
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("the statistics must be 1-D, finite and at least two")
    n = values.size
    members = _set_members(gene_set, ids, n)
    order = np.argsort(values, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    indicator = np.zeros(n)
    indicator[rank[members] - 1] = 1.0
    enrichment = _tricube_moving_average(indicator, window) / indicator.mean()
    highest = float(enrichment.max())
    ordered = values[order]

    ax = _axes(ax, (7.0, 3.2))
    patches = _module("matplotlib.patches")
    low = int(np.count_nonzero(ordered < -_BARCODE_QUANTILE))
    high = int(np.count_nonzero(ordered > _BARCODE_QUANTILE))
    for left, width, color, alpha in (
        (0.5, float(n), _PALE, 1.0),
        (0.5, float(low), _DOWN, 0.3),
        (n + 0.5 - high, float(high), _UP, 0.3),
    ):
        if width > 0:
            ax.add_patch(
                patches.Rectangle(
                    (left, 0.0),
                    width,
                    0.5,
                    facecolor=color,
                    alpha=alpha,
                    linewidth=0,
                    gid="bed",
                )
            )
    ax.vlines(np.sort(rank[members]), 0.0, 1.0, color=_INK, linewidth=1.0, gid="set")
    worm = _WORM_BASE + enrichment / highest * (_WORM_TOP - _WORM_BASE)
    ax.plot(np.arange(1, n + 1), worm, color=_ORANGE, linewidth=2, gid="worm")
    one = _WORM_BASE + (_WORM_TOP - _WORM_BASE) / highest
    ax.axhline(one, color=_INK, linewidth=0.8, linestyle="--")
    # The dashed line at 1, no enrichment, is always labelled; 0 where it stands apart.
    ticks, texts = [one, _WORM_TOP], ["1", f"{highest:.1f}"]
    if (one - _WORM_BASE) / (_WORM_TOP - _WORM_BASE) > 0.15:
        ticks.insert(0, _WORM_BASE)
        texts.insert(0, "0")
    ax.set_yticks(ticks, texts)
    ax.set_ylabel("Enrichment", loc="top")
    ax.set_ylim(0.0, _WORM_TOP + 0.1)
    ax.set_xlim(0.5, n + 0.5)
    ranks = np.unique(np.round(np.linspace(1, n, 5)).astype(int))
    ax.set_xticks(ranks, [f"{ordered[r - 1]:.2g}" for r in ranks])
    ax.set_xlabel(label)
    return ax


def _set_members(gene_set: Any, ids: list[Any] | None, n: int) -> npt.NDArray[np.intp]:
    """The positions of the set's genes: from ids, positions or a boolean mask."""
    members = unmasked_array(gene_set, "gene_set")
    if members.ndim != 1:
        raise ValueError(f"gene_set must be 1-D, got shape {members.shape}")
    if members.dtype == bool:
        if members.size != n:
            raise ValueError(
                f"a boolean gene_set must have one entry per gene ({n}), got "
                f"{members.size}"
            )
        found = np.flatnonzero(members)
    elif members.size and all(isinstance(v, str) for v in members.tolist()):
        if ids is None:
            raise ValueError("gene_set names genes by id; give gene_ids too")
        if len(ids) != n:
            raise ValueError(
                f"gene_ids must have one id per gene ({n}), got {len(ids)}"
            )
        position = {str(value): i for i, value in enumerate(ids)}
        if len(position) != n:
            raise ValueError("the gene ids must be unique")
        found = np.array(
            sorted({position[v] for v in members.tolist() if v in position}),
            dtype=np.intp,
        )
    else:
        found = np.unique(
            np.array(
                [
                    integral_number(
                        v, "gene_set", expected="ids, positions or a boolean mask"
                    )
                    for v in members.tolist()
                ],
                dtype=np.intp,
            )
        )
        if found.size and (found.min() < 0 or found.max() >= n):
            raise ValueError(f"gene_set holds a position outside 0 to {n - 1}")
    if found.size == 0:
        raise ValueError("no gene of the set is among the genes")
    if found.size == n:
        raise ValueError("the set holds every gene, so it cannot be enriched anywhere")
    return found


def _tricube_moving_average(values: FloatArray, span: float) -> FloatArray:
    """limma's tricubeMovingAverage: weighted means over windows of 2h + 1 values.

    h is the largest whole number up to span n / 2 that leaves the window within n
    values. The weight of a value k places away is (1 - (k / b)^3)^3 with
    b = 2h(h + 1) / (2h + 1), so that all 2h + 1 values weigh more than zero; near the
    ends the window holds the values that exist (found by probing limma 3.62.1).
    """
    n = values.size
    h = min(int(math.floor(span * n / 2.0)), (n - 1) // 2)
    if h < 1:
        return values.copy()
    reach = 2.0 * h * (h + 1) / (2.0 * h + 1.0)
    weights = (1.0 - (np.abs(np.arange(-h, h + 1)) / reach) ** 3) ** 3
    padded = np.concatenate([np.zeros(h), values, np.zeros(h)])
    present = np.concatenate([np.zeros(h), np.ones(n), np.zeros(h)])
    total = np.convolve(padded, weights, mode="valid")
    weight = np.convolve(present, weights, mode="valid")
    averaged: FloatArray = total / weight
    return averaged


def plot_gene_sets(
    table: Table,
    *,
    top: int = 20,
    alpha: float = 0.05,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the most significant gene sets of a gene-set test, by direction.

    For the ``top`` sets with the smallest p-values of a table of
    :func:`~cytepool.stats.gene_set_tests.camera` (Wu and Smyth 2012),
    :func:`~cytepool.stats.gene_set_tests.fry` or
    :func:`~cytepool.stats.rotation_gene_set_tests.mroast`, a bar of length -log10 of
    the adjusted p-value, to the right for sets up and to the left for sets down,
    with dashed lines at ``alpha``. Each set is named with its number of genes.

    Draws the bars (gid ``"gene sets"``) and the lines (gid ``"alpha"``).

    Args:
        table: The table of a gene-set test, with the columns ``gene_set``,
            ``n_genes``, ``direction``, ``p_value`` and ``adjusted_p_value``.
        top: How many sets to show, at least 1.
        alpha: The adjusted p-value of the dashed lines, in (0, 1).
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``table`` is not a Table or an option has the wrong type.
        ValueError: If the table lacks a column or has no set with a p-value.
    """
    _check_axes(ax)
    if not isinstance(table, Table):
        raise TypeError(f"table must be a Table, got {type(table).__name__}")
    count = positive_integer(top, "top")
    level = _fraction(alpha, "alpha")
    needed = ("gene_set", "n_genes", "direction", "p_value", "adjusted_p_value")
    missing = [column for column in needed if column not in table]
    if missing:
        raise ValueError(
            f"the table lacks the column(s) {quoted(missing)}; plot_gene_sets reads "
            "the tables of camera, fry and mroast"
        )
    p = np.asarray(table["p_value"], dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(p))
    if rows.size == 0:
        raise ValueError("no gene set of the table has a p-value")
    rows = rows[np.argsort(p[rows], kind="stable")[:count]]
    adjusted = np.asarray(table["adjusted_p_value"], dtype=np.float64)[rows]
    up = np.array([value == "up" for value in table["direction"][rows].tolist()])
    lengths = -np.log10(np.maximum(adjusted, np.finfo(np.float64).tiny))
    signed = np.where(up, lengths, -lengths)
    names = [
        f"{name} ({n})"
        for name, n in zip(
            table["gene_set"][rows].tolist(),
            table["n_genes"][rows].tolist(),
            strict=True,
        )
    ]
    ax = _axes(ax, (7.0, min(12.0, 1.2 + 0.3 * rows.size)))
    positions = np.arange(rows.size)[::-1]
    ax.barh(
        positions, signed, height=0.7, color=np.where(up, _UP, _DOWN), gid="gene sets"
    )
    ax.axvline(0.0, color=_INK, linewidth=0.8)
    threshold = -math.log10(level)
    for side in (-threshold, threshold):
        ax.axvline(side, color=_INK, linewidth=0.8, linestyle="--", gid="alpha")
    ax.set_yticks(positions, names)
    ax.set_xlabel("-log10 adjusted p-value, signed by direction")
    ax.set_xlim(
        1.08 * min(float(signed.min()), -threshold),
        1.08 * max(float(signed.max()), threshold),
    )
    _legend(
        ax,
        [
            _patch("Up", _UP),
            _patch("Down", _DOWN),
            _key(f"Adjusted p-value {level:g}", marker="", linestyle="--", color=_INK),
        ],
    )
    return ax


# Count models.


def plot_dispersions(
    dispersions: DESeq2Dispersions | DESeq2Fit, *, ax: "Axes | None" = None
) -> "Axes":
    """Plot DESeq2's dispersion estimates against the mean normalized count.

    DESeq2's ``plotDispEsts`` (Love et al. 2014), on logarithmic axes: the gene-wise
    estimates (``dispGeneEst``), the fitted trend (``dispFit``), drawn as a curve, and
    the final estimates (``dispersion``), the maximum a posteriori estimates shrunk
    towards the trend, with the dispersion outliers, which keep their gene-wise
    estimate, as open circles. Genes without counts are not drawn.

    Draws the gene-wise estimates (gid ``"gene-wise"``), the trend (gid
    ``"fitted"``), the final estimates (gid ``"final"``) and the outliers (gid
    ``"outliers"``).

    Args:
        dispersions: The result of :func:`~cytepool.stats.deseq2.deseq2_dispersions`,
            or a fit of :func:`~cytepool.stats.deseq2.fit_deseq2` that estimated them.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``dispersions`` is neither DESeq2Dispersions nor a DESeq2Fit.
        ValueError: If the fit was given its dispersions, or no gene has counts.
    """
    _check_axes(ax)
    if isinstance(dispersions, DESeq2Fit):
        if dispersions.dispersion_estimates is None:
            raise ValueError(
                "the fit was given its dispersions, so it holds no estimates to plot"
            )
        dispersions = dispersions.dispersion_estimates
    if not isinstance(dispersions, DESeq2Dispersions):
        raise TypeError(
            "dispersions must be DESeq2Dispersions or a DESeq2Fit, got "
            f"{type(dispersions).__name__}"
        )
    shown = ~np.asarray(dispersions.all_zero) & np.isfinite(dispersions.base_mean)
    if not shown.any():
        raise ValueError("no gene has counts, so there is no dispersion to plot")
    x = np.asarray(dispersions.base_mean)[shown]
    outlier = np.asarray(dispersions.outlier)[shown]
    final = np.asarray(dispersions.dispersion)[shown]
    order = np.argsort(x, kind="stable")
    ax = _axes(ax, (6.5, 4.0))
    ax.scatter(
        x,
        np.asarray(dispersions.genewise)[shown],
        s=_DENSE,
        color=_GREY,
        alpha=0.6,
        linewidths=0,
        gid="gene-wise",
    )
    ax.scatter(
        x[~outlier], final[~outlier], s=_DENSE, color=_BLUE, linewidths=0, gid="final"
    )
    ax.scatter(
        x[outlier],
        final[outlier],
        s=4 * _DENSE,
        facecolors="none",
        edgecolors=_BLUE,
        linewidths=1.0,
        gid="outliers",
    )
    ax.plot(
        x[order],
        np.asarray(dispersions.trended)[shown][order],
        color=_ORANGE,
        linewidth=2,
        gid="fitted",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean of normalized counts")
    ax.set_ylabel("Dispersion")
    handles = [
        _key("Gene-wise", color=_GREY),
        _key("Fitted", marker="", linestyle="-", color=_ORANGE, linewidth=2),
        _key("Final", color=_BLUE),
    ]
    if outlier.any():
        handles.append(
            _key("Outlier, gene-wise", markerfacecolor="none", markeredgecolor=_BLUE)
        )
    _legend(ax, handles)
    return ax


def plot_quasi_dispersions(
    fit: QuasiLikelihoodFit | QuasiLikelihoodTest, *, ax: "Axes | None" = None
) -> "Axes":
    """Plot edgeR's quasi-dispersions before and after moderation, with their prior.

    edgeR's ``plotQLDisp`` (Chen et al. 2016) for the quasi-likelihood model of Lund et
    al. (2012): the quarter-root of every gene's quasi-dispersion (the adjusted
    residual deviance over its degrees of freedom), of its moderated value, and of the
    prior, a trend in the average log2 counts per million or a line, against the
    abundance the trend was fitted to (``covariate``), or without a trend the average
    log2 counts per million.

    Draws the quasi-dispersions (gid ``"raw"``), the moderated ones (gid
    ``"squeezed"``) and the prior (gid ``"prior"``).

    Args:
        fit: The result of
            :func:`~cytepool.stats.quasi_likelihood.fit_quasi_likelihood`, or a test of
            it.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``fit`` is neither a QuasiLikelihoodFit nor a QuasiLikelihoodTest.
        ValueError: If no gene has a quasi-dispersion.
    """
    _check_axes(ax)
    if isinstance(fit, QuasiLikelihoodTest):
        fit = fit.fit
    if not isinstance(fit, QuasiLikelihoodFit):
        raise TypeError(
            "fit must be a QuasiLikelihoodFit or a QuasiLikelihoodTest, got "
            f"{type(fit).__name__}"
        )
    trended = fit.covariate is not None
    x = np.asarray(fit.covariate if trended else fit.average_log_cpm, dtype=np.float64)
    raw = np.asarray(fit.quasi_dispersion, dtype=np.float64)
    squeezed = np.asarray(fit.posterior_quasi_dispersion, dtype=np.float64)
    prior = np.broadcast_to(
        np.asarray(fit.prior_quasi_dispersion, dtype=np.float64), x.shape
    )
    shown = np.isfinite(x) & np.isfinite(raw)
    if not shown.any():
        raise ValueError("no gene has a quasi-dispersion to plot")
    root = np.sqrt(np.sqrt(raw[shown]))
    ax = _axes(ax, (6.5, 4.0))
    ax.scatter(
        x[shown], root, s=_DENSE, color=_GREY, alpha=0.6, linewidths=0, gid="raw"
    )
    moderated = np.isfinite(x) & np.isfinite(squeezed)
    ax.scatter(
        x[moderated],
        np.sqrt(np.sqrt(squeezed[moderated])),
        s=_DENSE,
        color=_BLUE,
        linewidths=0,
        gid="squeezed",
    )
    if np.ndim(fit.prior_quasi_dispersion) == 0:
        ax.axhline(
            float(np.sqrt(np.sqrt(prior[0]))), color=_ORANGE, linewidth=2, gid="prior"
        )
    else:
        line = np.isfinite(x) & np.isfinite(prior)
        order = np.argsort(x[line], kind="stable")
        ax.plot(
            x[line][order],
            np.sqrt(np.sqrt(prior[line][order])),
            color=_ORANGE,
            linewidth=2,
            gid="prior",
        )
    ax.set_xlabel("Average log2 CPM")
    ax.set_ylabel("Quarter-root of quasi-dispersion")
    _legend(
        ax,
        [
            _key("Raw", color=_GREY),
            _key("Squeezed", color=_BLUE),
            _key(
                "Prior" + (" trend" if trended else ""),
                marker="",
                linestyle="-",
                color=_ORANGE,
                linewidth=2,
            ),
        ],
    )
    return ax


# Meta-analysis.


def plot_forest(
    meta: MetaAnalysisResult,
    studies: Mapping[str, DifferentialExpressionResult],
    feature: str,
    *,
    group: str | None = None,
    contrast: str | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot each study's log2 fold change of a gene with its confidence interval, and
    the pooled estimate.

    The forest plot (Lewis and Clarke 2001) that metafor's ``forest`` draws for a model
    without moderators (Viechtbauer 2010), of one gene of a
    :func:`~cytepool.meta_analysis.differential_expression_meta_analysis`: a row per
    study, in the order of ``studies``, with its log2 fold change as a square and the
    interval y_i +/- z s_i, z the normal quantile of the meta-analysis's confidence
    level and s_i the study's standard error, as metafor draws it whatever the test;
    and at the bottom the pooled estimate as a diamond spanning its confidence
    interval. A square's area grows with the study's weight in the pooled estimate,
    1 / (s_i^2 + tau^2), scaled as metafor scales it. A study that did not test the
    gene keeps its row, marked "not tested". The estimates and intervals are written
    to the right of the plot, and the pooled row is named with the estimator of
    tau^2 and I^2.

    Draws the intervals (gid ``"intervals"``), the squares (gid ``"estimates"``), the
    diamond (gid ``"pooled"``) and a dotted line at zero.

    Args:
        meta: The meta-analysis.
        studies: The differential-expression results it combined, by study name.
        feature: The id of the gene.
        group: The group (cell type) of the meta-analysis; it may be left out when the
            meta-analysis has one group, or one with ``contrast``.
        contrast: The contrast; likewise.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If the group and contrast are not given and the meta-analysis has
            several, or ``studies`` is empty.
        KeyError: If the group and contrast, or the gene, were not meta-analysed.
    """
    _check_axes(ax)
    if not isinstance(meta, MetaAnalysisResult):
        raise TypeError(f"meta must be a MetaAnalysisResult, got {type(meta).__name__}")
    if not isinstance(studies, Mapping) or not all(
        isinstance(result, DifferentialExpressionResult) for result in studies.values()
    ):
        raise TypeError(
            "studies must map study names to DifferentialExpressionResult, the "
            "results the meta-analysis combined"
        )
    if not studies:
        raise ValueError("studies holds no study")
    if not isinstance(feature, str):
        reject_masked(feature, "feature")
        raise TypeError(
            f"feature must be a gene id (str), got {type(feature).__name__}"
        )
    group = _optional_text(group, "group")
    contrast = _optional_text(contrast, "contrast")
    key = _analysis(meta, group, contrast)
    pooled = meta.table(*key)
    found = np.flatnonzero(pooled["feature_id"] == feature)
    if found.size == 0:
        raise KeyError(
            f"gene {feature!r} was not meta-analysed in group {key[0]!r} and contrast "
            f"{key[1]!r}: fewer studies than min_studies tested it, or none did"
        )
    row = int(found[0])
    settings = meta.settings
    z = float(norm_isf((1.0 - settings["level"]) / 2.0))
    tau2 = float(pooled["tau_squared"][row])
    names = [str(name) for name in studies]
    estimates = np.full(len(names), np.nan)
    errors = np.full(len(names), np.nan)
    for i, result in enumerate(studies.values()):
        table = result.tables.get(key)
        if table is not None:
            hit = np.flatnonzero(table["feature_id"] == feature)
            if hit.size:
                estimates[i] = float(table["log2_fold_change"][hit[0]])
                errors[i] = float(table["standard_error"][hit[0]])
    tested = np.isfinite(estimates) & np.isfinite(errors)
    lows, highs = estimates - z * errors, estimates + z * errors
    weight = np.sqrt(1.0 / (errors**2 + tau2))
    spread = np.nanmax(weight) - np.nanmin(weight) if tested.any() else 0.0
    scale = np.where(
        spread > 0.0, 0.5 + (weight - np.nanmin(weight)) / (spread or 1.0), 1.0
    )
    estimate = float(pooled["log2_fold_change"][row])
    lower, upper = float(pooled["ci_lower"][row]), float(pooled["ci_upper"][row])

    k = len(names)
    ax = _axes(ax, (7.0, 1.4 + 0.4 * (k + 2)))
    rows = np.arange(k, 0, -1, dtype=np.float64)
    ax.hlines(
        rows[tested],
        lows[tested],
        highs[tested],
        color=_INK,
        linewidth=1.2,
        gid="intervals",
    )
    ax.scatter(
        estimates[tested],
        rows[tested],
        marker="s",
        s=(7.0 * scale[tested]) ** 2,
        color=_BLUE,
        zorder=3,
        gid="estimates",
    )
    diamond = _module("matplotlib.patches").Polygon(
        [(lower, -1.0), (estimate, -0.7), (upper, -1.0), (estimate, -1.3)],
        closed=True,
        facecolor=_ORANGE,
        edgecolor=_ORANGE,
        gid="pooled",
    )
    ax.add_patch(diamond)
    ax.axvline(0.0, color=_GREY, linewidth=1, linestyle=":", zorder=0)
    method = _METHODS.get(settings["method"], settings["method"])
    label = f"Pooled, {method}"
    if settings["method"] != "fixed":
        label += f" (I² {100 * float(pooled['i_squared'][row]):.0f}%)"
    ax.set_yticks([*rows, -1.0], [*names, label])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-2.0, k + 1.0)
    # The intervals and the line of no effect, at zero, with a margin.
    ends = np.concatenate([lows[tested], highs[tested], [lower, upper, 0.0]])
    margin = 0.05 * (ends.max() - ends.min()) or 1.0
    ax.set_xlim(ends.min() - margin, ends.max() + margin)
    transform = _module("matplotlib.transforms").blended_transform_factory(
        ax.transAxes, ax.transData
    )
    texts = [
        f"{estimates[i]:.2f} [{lows[i]:.2f}, {highs[i]:.2f}]"
        if tested[i]
        else "not tested"
        for i in range(k)
    ]
    for y, text in zip(
        [*rows, -1.0],
        [*texts, f"{estimate:.2f} [{lower:.2f}, {upper:.2f}]"],
        strict=True,
    ):
        ax.text(1.02, y, text, transform=transform, ha="left", va="center", fontsize=9)
    ax.set_xlabel("log2 fold change")
    return ax


def _analysis(
    meta: MetaAnalysisResult, group: str | None, contrast: str | None
) -> tuple[str, str]:
    """The group and contrast of a meta-analysis that are meant."""
    keys = [
        key
        for key in meta.tables
        if (group is None or key[0] == group)
        and (contrast is None or key[1] == contrast)
    ]
    if group is not None and contrast is not None:
        return group, contrast
    if len(keys) != 1:
        pairs = listing([f"{g!r} and {c!r}" for g, c in meta.tables])
        raise ValueError(
            "name the group and contrast to plot; the meta-analysis has: "
            f"{pairs or '(none)'}"
        )
    return keys[0]
