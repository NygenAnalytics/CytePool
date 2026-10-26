"""Figures of pseudobulk and bulk analyses, drawn as limma, edgeR, DESeq2, speckle,
metafor, variancePartition, fgsea and decoupler draw them.

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
  (Ritchie et al. 2015), whose distances are leading log2 fold changes;
  :func:`plot_pca`, the principal component plot of DESeq2's ``plotPCA`` (Love et al.
  2014; Pearson 1901).
- Cubes: :func:`plot_cell_numbers`, the cells pooled into each sample and group;
  :func:`plot_dotplot`, scanpy's dot plot of the fraction of cells that express each
  gene and its mean expression in each group (Wolf et al. 2018), from the cube's sums.
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
  sets of a ``camera``, ``fry`` or ``mroast`` table; :func:`plot_over_representation`,
  the sets of an over-representation analysis, as clusterProfiler's dot plots show
  them (Yu et al. 2012); :func:`plot_enrichment`, GSEA's enrichment plot (Subramanian
  et al. 2005), and :func:`plot_gsea_table`, the top sets of a GSEA with their genes
  on the ranking, as fgsea's ``plotEnrichment`` and ``plotGseaTable`` draw them
  (Korotkevich et al. 2021); :func:`plot_pathway_scores`, the scores of gene sets in
  every sample as a heatmap, as GSVA's guide draws them (Haenzelmann, Castelo and
  Guinney 2013).
- Activities: :func:`plot_activities`, the activities of transcription factors or
  pathways in every sample or contrast as a heatmap, and :func:`plot_top_activities`,
  the sources with the largest scores of one profile as bars, as decoupler's
  ``pl.barplot`` draws them (Badia-i-Mompel et al. 2022).
- Count models: :func:`plot_dispersions`, DESeq2's ``plotDispEsts`` (Love et al.
  2014); :func:`plot_quasi_dispersions`, edgeR's ``plotQLDisp`` (Chen et al. 2016).
- Meta-analysis: :func:`plot_forest`, the forest plot (Lewis and Clarke 2001) that
  metafor's ``forest`` draws (Viechtbauer 2010).
- Replication across studies: :func:`plot_study_agreement`, pi1 or the sign agreement
  of every pair of studies (Storey and Tibshirani 2003); :func:`plot_replicated_genes`,
  the genes that each number of studies finds (Nakatsuka et al. 2025).
- Several studies: :func:`plot_sumrank`, the SumRank p-value of every gene, the genes
  in alphabetical order, as the authors of SumRank draw it, and
  :func:`plot_held_out_auc`, the AUC of every held-out study by the method that
  chose its genes (Nakatsuka et al. 2025).
- Variance partitioning: :func:`plot_variance_fractions`, the violins (Hintze and
  Nelson 1998) of the fractions of variance of every variable over the genes, as
  variancePartition's ``plotVarPart`` draws them (Hoffman and Schadt 2016);
  :func:`plot_gene_variance_fractions`, the fractions of chosen genes as stacked
  bars, as its ``plotPercentBars`` draws them.

``validation/plotting.py`` compares what each figure shows with what the R function it
follows draws for the same data, where one exists, and
``validation/combined_p_values_and_replication.py`` the replication figures with what R
computes for them.

Colours:
    Categories, such as the cell types of a composition or the levels of a sample
    variable, take eight hues in a fixed order: blue, orange, aqua, yellow, magenta,
    green, violet and red. Neighbouring hues stay apart for readers with
    colour-vision deficiencies, as measured under simulated protanopia and
    deuteranopia (Machado et al. 2009), and so do any two of the first three. A
    scatter plot of more than three levels therefore also gives each level a marker
    of its own, so that colour is never the only cue. Up and down are red and blue,
    values shown for context are grey, and numbers take shades of blue; values about
    zero, as in a heatmap of scores, run from blue through a neutral grey to red, the
    two arms of equal lightness at each step. A figure that would need more than eight
    hues raises an error that says how to merge or select the categories.

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

    Yu G, Wang LG, Han Y, He QY (2012). clusterProfiler: an R package for comparing
    biological themes among gene clusters. OMICS 16, 284-287. doi:10.1089/omi.2011.0118

    Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, Paulovich
    A, Pomeroy SL, Golub TR, Lander ES, Mesirov JP (2005). Gene set enrichment
    analysis: a knowledge-based approach for interpreting genome-wide expression
    profiles. Proceedings of the National Academy of Sciences 102, 15545-15550.
    doi:10.1073/pnas.0506580102

    Korotkevich G, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A (2021).
    Fast gene set enrichment analysis. bioRxiv 060012, version 3.
    doi:10.1101/060012

    Haenzelmann S, Castelo R, Guinney J (2013). GSVA: gene set variation analysis for
    microarray and RNA-seq data. BMC Bioinformatics 14, 7.
    doi:10.1186/1471-2105-14-7

    Badia-i-Mompel P, Velez Santiago J, Braunger J, Geiss C, Dimitrov D, Mueller-Dott S,
    Taus P, Dugourd A, Holland CH, Ramirez Flores RO, Saez-Rodriguez J (2022).
    decoupleR: ensemble of computational methods to infer biological activities from
    omics data. Bioinformatics Advances 2, vbac016. doi:10.1093/bioadv/vbac016

    Lewis S, Clarke M (2001). Forest plots: trying to see the wood and the trees. BMJ
    322, 1479-1480. doi:10.1136/bmj.322.7300.1479

    Viechtbauer W (2010). Conducting meta-analyses in R with the metafor package.
    Journal of Statistical Software 36(3), 1-48. doi:10.18637/jss.v036.i03

    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Machado GM, Oliveira MM, Fernandes LAF (2009). A physiologically-based model for
    simulation of color vision deficiency. IEEE Transactions on Visualization and
    Computer Graphics 15, 1291-1298. doi:10.1109/TVCG.2009.113

    Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of variation
    in complex gene expression studies. BMC Bioinformatics 17, 483.
    doi:10.1186/s12859-016-1323-z

    Hintze JL, Nelson RD (1998). Violin plots: a box plot-density trace synergism. The
    American Statistician 52, 181-184. doi:10.1080/00031305.1998.10480559

    Silverman BW (1986). Density Estimation for Statistics and Data Analysis. Chapman
    and Hall, London.

    Storey JD, Tibshirani R (2003). Statistical significance for genomewide studies.
    Proceedings of the National Academy of Sciences 100, 9440-9445.
    doi:10.1073/pnas.1530509100

    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Pearson K (1901). On lines and planes of closest fit to systems of points in space.
    The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2,
    559-572. doi:10.1080/14786440109462720

    Wolf FA, Angerer P, Theis FJ (2018). SCANPY: large-scale single-cell gene expression
    data analysis. Genome Biology 19, 15. doi:10.1186/s13059-017-1382-0
"""

import importlib
import math
import numbers
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
    number_at_least,
    positive_integer,
    real_number,
)
from ._ranked_genes import (
    gene_set_collection,
    matched_positions,
    ranked_genes,
    running_sum,
)
from .activity import Activities
from .composition import CompositionResult
from .cube import Cube
from .differential_expression import DifferentialExpressionResult
from .gene_sets import GeneSets
from .held_out_prediction import HeldOutPrediction
from .meta_analysis import MetaAnalysisResult
from .normalization import filter_by_expression, log_cpm, tmm_factors
from .pathway_scores import PathwayScores
from .principal_components import PrincipalComponents, principal_components
from .replication import ReplicationResult
from .stats.deseq2 import DESeq2Dispersions, DESeq2Fit, DESeq2Results
from .stats.distributions import f_cdf, f_sf, norm_isf
from .stats.empirical_bayes import ModeratedFit, TreatFit
from .stats.multiple_testing import adjust_p_values
from .stats.quasi_likelihood import QuasiLikelihoodFit, QuasiLikelihoodTest
from .stats.voom import VoomResult
from .sumrank import SumRankResult
from .table import Table
from .variance_partitioning import (
    RESIDUALS,
    VarianceFractions,
    VariancePartitioningResult,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "plot_activities",
    "plot_barcode",
    "plot_cell_numbers",
    "plot_composition",
    "plot_dispersions",
    "plot_dotplot",
    "plot_enrichment",
    "plot_forest",
    "plot_gene_sets",
    "plot_gene_variance_fractions",
    "plot_gsea_table",
    "plot_held_out_auc",
    "plot_md",
    "plot_mds",
    "plot_mean_variance",
    "plot_over_representation",
    "plot_p_values",
    "plot_pathway_scores",
    "plot_pca",
    "plot_proportions",
    "plot_quasi_dispersions",
    "plot_replicated_genes",
    "plot_residual_variances",
    "plot_sample_weights",
    "plot_significant_genes",
    "plot_study_agreement",
    "plot_sumrank",
    "plot_top_activities",
    "plot_variance_fractions",
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
# A diverging ramp for values about zero: the blue ramp, dark to light, a neutral
# grey midpoint, and a red ramp of the same lightness at each step, light to dark.
_DIVERGING = (
    "#0d366b",
    "#1c5cab",
    "#3987e5",
    "#86b6ef",
    "#cde2fb",
    "#f0efec",
    "#fad6d2",
    "#ea9a93",
    "#d75853",
    "#9e3432",
    "#621b1a",
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
# The area of the largest dot of a dot plot of gene sets, in points squared, and the
# labels of its lists of genes.
_LARGEST_DOT = 300.0
_LIST_LABELS = {"up": "Up", "down": "Down", "either": "Either", None: "Selected"}
_TINY = np.finfo(np.float64).tiny  # the smallest p-value shown on a log scale
# A GSEA table: the half-height of the bars of a row; where the headers of its columns
# stand, the middle of the ranks and the starts of the numbers, and its width, in
# numbers of genes ranked.
_TABLE_BAR = 0.45
_TABLE_COLUMNS = (0.5, 1.04, 1.17, 1.37)
_TABLE_WIDTH = 1.55
# A heatmap labels its rows and columns up to these numbers, and draws the strip of
# its columns' groups about a third of a row high, within these shares of its height.
_LABELLED_ROWS = 80
_LABELLED_COLUMNS = 60
_STRIP_HEIGHT = (0.04, 0.1)
_SCORE_LABELS = {
    "gsva": "GSVA score",
    "ssgsea": "ssGSEA score",
    "combined_z_scores": "Combined z-score",
    "plage": "PLAGE score",
}
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


def plot_over_representation(
    table: Table,
    *,
    top: int = 20,
    alpha: float = 0.05,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the most over-represented gene sets, a dot for each list of genes.

    For the ``top`` sets with the smallest p-value in any list of a table of
    :func:`~cytepool.over_representation.over_representation`, most significant at
    the top, a dot in the column of each list of selected genes (up, down and either,
    or the genes given), as clusterProfiler draws the lists of genes it compares (Yu
    et al. 2012). The area of a dot is proportional to the overlap, the number of the
    set's genes among the selected genes, and its colour gives the adjusted p-value
    as -log10, darker for smaller; a dot whose adjusted p-value is not below
    ``alpha`` is pale grey, and a list that holds none of the set's genes has no dot.
    Each set is named with its number of genes in the universe, and each list with
    its number of selected genes.

    Draws the dots (gid ``"gene sets"``).

    Args:
        table: The table of ``over_representation``, with the columns ``gene_set``,
            ``direction``, ``n_genes``, ``overlap``, ``p_value``,
            ``adjusted_p_value`` and ``n_selected``.
        top: How many sets to show, at least 1.
        alpha: The adjusted p-value below which dots are coloured, in (0, 1).
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
    needed = (
        "gene_set",
        "direction",
        "n_genes",
        "overlap",
        "p_value",
        "adjusted_p_value",
        "n_selected",
    )
    missing = [column for column in needed if column not in table]
    if missing:
        raise ValueError(
            f"the table lacks the column(s) {quoted(missing)}; "
            "plot_over_representation reads the tables of over_representation"
        )
    names = table["gene_set"].tolist()
    directions = table["direction"].tolist()
    p = np.asarray(table["p_value"], dtype=np.float64)
    best: dict[str, float] = {}
    for name, value in zip(names, p.tolist(), strict=True):
        if math.isfinite(value) and value < best.get(name, math.inf):
            best[name] = value
    if not best:
        raise ValueError("no gene set of the table has a p-value")
    shown = sorted(best, key=best.__getitem__)[:count]  # stable: ties in table order
    lists = list(dict.fromkeys(directions))
    row_of = {name: len(shown) - 1 - i for i, name in enumerate(shown)}  # top first
    column_of = {direction: j for j, direction in enumerate(lists)}
    overlap = np.asarray(table["overlap"], dtype=np.float64)
    drawn = np.array(
        [
            name in row_of and math.isfinite(value) and size > 0
            for name, value, size in zip(
                names, p.tolist(), overlap.tolist(), strict=True
            )
        ],
        dtype=bool,
    )
    rows = np.flatnonzero(drawn)
    x = np.array([column_of[directions[i]] for i in rows], dtype=np.float64)
    y = np.array([row_of[names[i]] for i in rows], dtype=np.float64)
    largest = float(overlap[rows].max(initial=1.0))
    areas = _LARGEST_DOT * overlap[rows] / largest
    strength = -np.log10(
        np.maximum(np.asarray(table["adjusted_p_value"], dtype=np.float64)[rows], _TINY)
    )
    significant = strength > -math.log10(level)
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_blues", _BLUES)
    floor = -math.log10(level)
    norm = colors.Normalize(
        vmin=floor, vmax=max(float(strength.max(initial=floor)), floor + 1.0)
    )
    faces = np.array([colors.to_rgba(_PALE)] * rows.size).reshape(-1, 4)
    if significant.any():
        faces[significant] = ramp(norm(strength[significant]))

    # Tall enough for the colour bar and the key below it.
    height = min(12.0, max(4.5, 1.8 + 0.32 * len(shown)))
    ax = _axes(ax, (6.0 + 1.0 * len(lists), height))
    ax.scatter(x, y, s=areas, c=faces, linewidths=0, gid="gene sets")
    size_of = {name: int(n) for name, n in zip(names, table["n_genes"].tolist())}
    ax.set_yticks(
        [row_of[name] for name in shown],
        [f"{name} ({size_of[name]:,})" for name in shown],
    )
    selected_of = {
        direction: int(n)
        for direction, n in zip(directions, table["n_selected"].tolist(), strict=True)
    }
    ax.set_xticks(
        range(len(lists)),
        [
            f"{_LIST_LABELS.get(direction, str(direction))}\n({selected_of[direction]:,})"
            for direction in lists
        ],
    )
    ax.set_xlim(-0.6, len(lists) - 0.4)
    ax.set_ylim(-0.6, len(shown) - 0.4)
    ax.set_xlabel("Selected genes")
    mappable = _module("matplotlib.cm").ScalarMappable(norm=norm, cmap=ramp)
    bar = ax.figure.colorbar(
        mappable, ax=ax, label="-log10 adjusted p-value", shrink=0.5, anchor=(0.0, 1.0)
    )
    keys = sorted({max(1, round(largest * share)) for share in (1.0, 0.5, 0.25)})
    handles = [
        *(
            _key(
                f"{n:,}", color=_GREY, markersize=math.sqrt(_LARGEST_DOT * n / largest)
            )
            for n in reversed(keys)
        ),
        _key(f"Adjusted p-value {level:g} or more", color=_PALE),
    ]
    # The key of the areas below the colour bar, as part of its Axes.
    key = bar.ax.legend(
        handles=handles,
        title="Overlap",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.06),
        borderaxespad=0.0,
        frameon=False,
        alignment="left",
    )
    key.get_title().set_fontweight("bold")
    return ax


def plot_enrichment(
    statistics: Any,
    gene_set: Any,
    *,
    group: str | None = None,
    contrast: str | None = None,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    exponent: float = 1.0,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the running enrichment score of a gene set along the ranked genes.

    GSEA's enrichment plot (Subramanian et al. 2005) as fgsea's ``plotEnrichment``
    draws it (Korotkevich et al. 2021): the genes ranked by decreasing statistic, as
    :func:`~cytepool.gsea.gsea` ranks them, along the x-axis, from rank 0 to N + 1;
    the walk that rises at each gene of the set by its share of the set's weight
    |r|^p and falls at each other gene, drawn through its values just before and at
    each gene of the set; a tick at the rank of each gene of the set, spanning a
    sixteenth of the walk's range above and below zero; and dashed lines at the
    walk's largest and smallest values, of which the larger in absolute value is the
    enrichment score.

    Draws the walk (gid ``"running sum"``), the ticks (gid ``"set"``) and the dashed
    lines (gid ``"extremes"``).

    Args:
        statistics: The statistics, in any form :func:`~cytepool.gsea.gsea` takes.
        gene_set: The genes of the set, as gene ids, such as ``gene_sets[name]``;
            ids that are not ranked are left out.
        group: The group of a ``DifferentialExpressionResult``, as for ``gsea``.
        contrast: Its contrast, as for ``gsea``.
        id_column: The column of gene ids of a table.
        statistic_column: The column of statistics of a table.
        exponent: The power p of the weights, at least 0, as for ``gsea``.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the statistics cannot be ranked, as for ``gsea``, the set holds
            no ranked gene or every one, or ``exponent`` is negative or not finite.
        KeyError: If a ``DifferentialExpressionResult`` has no table of the group and
            contrast.
    """
    _check_axes(ax)
    power = number_at_least(exponent, "exponent", 0.0)
    reject_masked(gene_set, "gene_set")
    if isinstance(gene_set, str) or isinstance(gene_set, Mapping):
        raise TypeError(
            "gene_set must be a list of gene ids, such as gene_sets[name], got "
            f"{type(gene_set).__name__}"
        )
    genes = GeneSets({"gene_set": gene_set})["gene_set"]
    ranking = ranked_genes(
        statistics,
        group=group,
        contrast=contrast,
        id_column=id_column,
        statistic_column=statistic_column,
    )
    n = len(ranking.ids)
    index = {gene: i for i, gene in enumerate(ranking.ids)}
    positions = np.sort(matched_positions(genes, index))
    if positions.size == 0:
        raise ValueError("no gene of the set is among the ranked genes")
    if positions.size == n:
        raise ValueError(
            "the set holds every ranked gene, so it has no enrichment score"
        )
    weights = np.ascontiguousarray(np.abs(ranking.values) ** power)
    tops, bottoms = running_sum(weights, positions)
    ranks = positions + 1.0
    x = np.concatenate(
        [[0.0], np.column_stack([ranks - 1.0, ranks]).ravel(), [n + 1.0]]
    )
    y = np.concatenate([[0.0], np.column_stack([bottoms, tops]).ravel(), [0.0]])
    highest, lowest = float(tops.max()), float(bottoms.min())
    spread = highest - lowest

    ax = _axes(ax, (7.0, 3.6))
    ax.plot(x, y, color=_AQUA, linewidth=1.5, gid="running sum")
    ax.vlines(
        ranks, -spread / 16.0, spread / 16.0, color=_INK, linewidth=0.6, gid="set"
    )
    for value in (highest, lowest):
        ax.axhline(value, color=_UP, linewidth=1.0, linestyle="--", gid="extremes")
    ax.axhline(0.0, color=_INK, linewidth=0.8)
    ax.set_xlim(0.0, n + 1.0)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Enrichment score")
    return ax


def plot_gsea_table(
    statistics: Any,
    gene_sets: Mapping[str, Any],
    table: Table,
    *,
    group: str | None = None,
    contrast: str | None = None,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    top: int = 10,
    exponent: float = 1.0,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the most significant gene sets of a GSEA, each with its genes on the
    ranking.

    The table of fgsea's ``plotGseaTable`` (Korotkevich et al. 2021): a row for each
    of the ``top`` sets with a positive enrichment score and the smallest p-values
    of a table of :func:`~cytepool.gsea.gsea`, most significant first, followed by
    the ``top`` with a negative score, most significant last, as fgsea's guide
    chooses them; the ties of p-values in the order of the table. Each row shows the
    set's name, its genes as bars at their ranks among the genes ranked by
    decreasing statistic, of height sign(r) |r|^p scaled by its largest absolute
    value over all genes, and the set's normalized enrichment score, p-value and
    adjusted p-value, written as fgsea writes them.

    Draws the bars (gid ``"gene ranks"``) and the numbers (gid ``"numbers"``).

    Args:
        statistics: The statistics that ``table`` was computed from, in any form
            :func:`~cytepool.gsea.gsea` takes.
        gene_sets: The gene sets of ``table``: a
            :class:`~cytepool.gene_sets.GeneSets` or a mapping of set name to gene
            ids.
        table: The table of ``gsea``, with the columns ``gene_set``,
            ``enrichment_score``, ``normalized_enrichment_score``, ``p_value`` and
            ``adjusted_p_value``.
        group: The group of a ``DifferentialExpressionResult``, as for ``gsea``.
        contrast: Its contrast, as for ``gsea``.
        id_column: The column of gene ids of a table.
        statistic_column: The column of statistics of a table.
        top: How many sets of each sign to show, at least 1.
        exponent: The power p of the heights of the bars, at least 0; fgsea suggests
            0.5 to flatten them.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the statistics cannot be ranked, the table lacks a column, has
            no set with a p-value, or names a set that ``gene_sets`` does not hold, or
            an option is out of range.
        KeyError: If a ``DifferentialExpressionResult`` has no table of the group and
            contrast.
    """
    _check_axes(ax)
    if not isinstance(table, Table):
        raise TypeError(f"table must be a Table, got {type(table).__name__}")
    count = positive_integer(top, "top")
    power = number_at_least(exponent, "exponent", 0.0)
    reject_masked(gene_sets, "gene_sets")
    collection = gene_set_collection(gene_sets)
    needed = (
        "gene_set",
        "enrichment_score",
        "normalized_enrichment_score",
        "p_value",
        "adjusted_p_value",
    )
    missing = [column for column in needed if column not in table]
    if missing:
        raise ValueError(
            f"the table lacks the column(s) {quoted(missing)}; plot_gsea_table reads "
            "the tables of gsea"
        )
    p = np.asarray(table["p_value"], dtype=np.float64)
    scores = np.asarray(table["enrichment_score"], dtype=np.float64)
    tested = np.isfinite(p)
    if not tested.any():
        raise ValueError("no gene set of the table has a p-value")
    up = np.flatnonzero(tested & (scores > 0.0))
    down = np.flatnonzero(tested & (scores < 0.0))
    up = up[np.argsort(p[up], kind="stable")[:count]]
    down = down[np.argsort(p[down], kind="stable")[:count]][::-1]
    rows = np.concatenate([up, down]).astype(np.intp)
    names = table["gene_set"][rows].tolist()
    absent = [name for name in names if name not in collection]
    if absent:
        raise ValueError(
            f"gene_sets has no set {quoted(absent)}; give the gene sets that the "
            "table was computed from"
        )
    ranking = ranked_genes(
        statistics,
        group=group,
        contrast=contrast,
        id_column=id_column,
        statistic_column=statistic_column,
    )
    n = len(ranking.ids)
    index = {gene: i for i, gene in enumerate(ranking.ids)}
    heights = np.sign(ranking.values) * np.abs(ranking.values) ** power
    largest = float(np.abs(heights).max())
    if largest > 0.0:
        heights = heights / largest

    shown = len(rows)
    ax = _axes(ax, (8.0, min(14.0, 1.4 + 0.4 * shown)))
    xs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for row, name in enumerate(names):
        positions = matched_positions(collection[name], index)
        base = float(shown - 1 - row)
        xs.append(positions + 1.0)
        bases.append(np.full(positions.size, base))
        ends.append(base + _TABLE_BAR * heights[positions])
    ax.vlines(
        np.concatenate(xs),
        np.concatenate(bases),
        np.concatenate(ends),
        color=_INK,
        linewidth=0.5,
        gid="gene ranks",
    )
    levels = np.arange(shown - 1, -1, -1, dtype=np.float64)
    ax.set_yticks(levels, names)
    ax.tick_params(axis="y", length=0)
    # The numbers stand in columns to the right of the ranks, inside the Axes.
    ax.set_xlim(0.0, _TABLE_WIDTH * n)
    ax.set_ylim(-0.6, shown + 0.4)
    ticks = np.unique(np.round(np.linspace(0, n, 5)).astype(int))
    ax.set_xticks(ticks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0.0, float(n))
    ax.set_xlabel("Rank")
    ax.xaxis.label.set_x(0.5 / _TABLE_WIDTH)
    columns = (
        ("Gene ranks", None),
        ("NES", np.asarray(table["normalized_enrichment_score"], dtype=np.float64)),
        ("p-value", p),
        ("Adjusted\np-value", np.asarray(table["adjusted_p_value"], dtype=np.float64)),
    )
    for (header, values), start in zip(columns, _TABLE_COLUMNS, strict=True):
        ax.text(
            start * n,
            shown - 0.3,
            header,
            ha="center" if values is None else "left",
            va="bottom",
            fontweight="bold",
        )
        if values is None:
            continue
        for level, value in zip(levels.tolist(), values[rows].tolist(), strict=True):
            text = f"{value:.2f}" if header == "NES" else _power_of_ten(value)
            ax.text(start * n, level, text, ha="left", va="center", gid="numbers")
    return ax


def _power_of_ten(value: float) -> str:
    """A p-value as fgsea's tables write it, such as 1.2·10^-5, in mathtext."""
    if not math.isfinite(value) or value <= 0.0:
        return f"{value:g}"
    exponent = math.floor(math.log10(value))
    return rf"${value / 10.0**exponent:.1f} \cdot 10^{{{exponent}}}$"


def plot_pathway_scores(
    scores: PathwayScores,
    *,
    sets: Sequence[str] | None = None,
    color_by: str | Sequence[Any] | None = None,
    standardize: bool = False,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the scores of gene sets in every sample as a heatmap.

    The heatmap of GSVA's guide (Haenzelmann, Castelo and Guinney 2013): a row for
    each set and a column for each sample, each cell coloured by the set's score in
    the sample, on a scale from blue (low) through grey (zero) to red (high) that is
    symmetric about zero. With ``color_by``, the samples are grouped by its levels,
    in the order the levels first appear, under a strip of the groups' colours, as
    the guide's ``heatmap(ColSideColors = ...)`` shows them, with white lines between
    the groups; within a group they keep their order. ``standardize`` centres each
    set's scores on their mean over the samples shown and divides them by their
    standard deviation, as R's ``heatmap(scale = "row")`` does, so that sets whose
    scores spread differently compare. NaN scores are dark grey.

    Draws the cells (gid ``"scores"``) and, with ``color_by``, the strip (gid
    ``"groups"``) and the lines between groups (gid ``"group boundaries"``).

    Args:
        scores: The scores of :func:`~cytepool.pathway_scores.gsva`,
            :func:`~cytepool.pathway_scores.ssgsea`,
            :func:`~cytepool.pathway_scores.combined_z_scores` or
            :func:`~cytepool.pathway_scores.plage`.
        sets: The sets to show, top to bottom; by default every set, in order.
        color_by: A column of ``scores.samples``, such as ``"condition"`` for the
            scores of a cube's group, or one label per sample, at most eight levels.
        standardize: Whether to standardize each set's scores across the samples.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes of the heatmap.

    Raises:
        TypeError: If ``scores`` is not a ``PathwayScores``, an option has the wrong
            type, or an argument is or holds a numpy masked array.
        ValueError: If a set is not among the scores or is named twice, the labels do
            not match the samples, or ``color_by`` has more than eight levels.
        KeyError: If ``color_by`` names a column the samples do not have.
    """
    _check_axes(ax)
    if not isinstance(scores, PathwayScores):
        raise TypeError(
            "scores must be a PathwayScores, as gsva, ssgsea, combined_z_scores and "
            f"plage return, got {type(scores).__name__}"
        )
    scaled = checked_flag(standardize, "standardize")
    names = list(scores.gene_sets)
    if sets is None:
        chosen = names
    else:
        reject_masked(sets, "sets")
        chosen = [sets] if isinstance(sets, str) else list(sets)
        for name in chosen:
            if not isinstance(name, str):
                raise TypeError(f"sets must hold set names (str), got {name!r}")
            if name not in scores.gene_sets:
                raise ValueError(
                    f"no set {name!r} among the scores; they hold {quoted(names)}"
                )
        if len(set(chosen)) != len(chosen):
            raise ValueError("sets names a set twice")
        if not chosen:
            raise ValueError("sets is empty; name at least one set")
    values = scores.scores[:, [names.index(name) for name in chosen]].T
    table = scores.samples
    n_samples = values.shape[1]
    samples = _Samples(
        ids=[str(value) for value in table["sample_id"].tolist()],
        rows=list(range(n_samples)),
        n_rows=n_samples,
        variables={name: table[name] for name in table.columns},
        source="the scores' sample table",
    )
    order = np.arange(n_samples)
    groups = None
    if color_by is not None:
        _check_variable(color_by, "color_by")
        labels, levels = samples.labels(color_by, "color_by")
        hues = _hues(levels, "color_by", "merge levels or give fewer labels")
        position = {level: i for i, level in enumerate(levels)}
        order = np.argsort([position[label] for label in labels], kind="stable")
        groups = ([labels[i] for i in order.tolist()], hues)
    values = values[:, order]
    if scaled:
        values = _standardized_rows(values)
    label = _SCORE_LABELS[scores.method]
    ax = _axes(
        ax,
        (min(14.0, 3.5 + 0.22 * n_samples), min(14.0, 1.8 + 0.28 * len(chosen))),
    )
    _draw_heatmap(
        ax,
        values,
        chosen,
        [samples.ids[i] for i in order.tolist()],
        "Standardized score" if scaled else label,
        groups,
    )
    return ax


def _draw_heatmap(
    ax: "Axes",
    values: FloatArray,
    row_labels: list[str],
    column_labels: list[str],
    colorbar_label: str,
    groups: tuple[list[str], dict[str, str]] | None,
) -> None:
    """Draw a rows x columns matrix of values about zero as a heatmap.

    The scale is the diverging ramp, symmetric about zero; NaN is dark grey. Rows
    are labelled when there are at most ``_LABELLED_ROWS`` of them, and columns when
    there are at most ``_LABELLED_COLUMNS``. ``groups`` gives each column's group,
    with the groups' hues: a strip above the matrix shows them, white lines part
    them, and a legend above the strip names them.
    """
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_diverging", _DIVERGING)
    ramp = ramp.with_extremes(bad=_GREY)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(finite.max()) if finite.size and finite.max() > 0.0 else 1.0
    n_rows, n_columns = values.shape
    image = ax.imshow(
        values,
        cmap=ramp,
        norm=colors.Normalize(vmin=-limit, vmax=limit),
        aspect="auto",
        interpolation="none",
    )
    image.set_gid("scores")
    ax.set_yticks(
        range(n_rows) if n_rows <= _LABELLED_ROWS else [],
        row_labels if n_rows <= _LABELLED_ROWS else [],
    )
    ax.set_xticks(
        range(n_columns) if n_columns <= _LABELLED_COLUMNS else [],
        column_labels if n_columns <= _LABELLED_COLUMNS else [],
        rotation=90,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if groups is not None:
        labels, hues = groups
        height = min(_STRIP_HEIGHT[1], max(_STRIP_HEIGHT[0], 0.3 / n_rows))
        strip = ax.inset_axes((0.0, 1.01, 1.0, height), sharex=ax)
        shown = colors.ListedColormap([hues[level] for level in hues])
        codes = {level: i for i, level in enumerate(hues)}
        bar = strip.imshow(
            np.array([[codes[label] for label in labels]], dtype=float),
            cmap=shown,
            norm=colors.Normalize(vmin=-0.5, vmax=len(hues) - 0.5),
            aspect="auto",
            interpolation="none",
        )
        bar.set_gid("groups")
        strip.set_axis_off()
        boundaries = [
            i + 0.5 for i in range(n_columns - 1) if labels[i] != labels[i + 1]
        ]
        for target in (ax, strip):
            target.vlines(
                boundaries,
                -0.5,
                (n_rows if target is ax else 1) - 0.5,
                color="white",
                linewidth=2.0,
                gid="group boundaries",
            )
        strip.legend(
            handles=[_patch(level, hue) for level, hue in hues.items()],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=min(len(hues), 4),
            frameon=False,
            borderaxespad=0.0,
        )
    ax.figure.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)


def _standardized_rows(values: FloatArray) -> FloatArray:
    """Each row centred on the mean of its finite values and divided by their
    standard deviation, as R's ``heatmap(scale = "row")``; NaN for a row of fewer
    than two values, or of equal ones."""
    observed = np.isfinite(values)
    counts = observed.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(observed, values, 0.0).sum(axis=1, keepdims=True) / counts
        centred = values - means
        squares = np.where(observed, centred**2, 0.0).sum(axis=1, keepdims=True)
        scaled: FloatArray = centred / np.sqrt(squares / (counts - 1.0))
    return scaled


# Activities.


def plot_activities(
    activities: Activities,
    *,
    sources: Sequence[str] | None = None,
    top: int | None = 25,
    color_by: str | Sequence[Any] | None = None,
    standardize: bool = False,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the activities of the sources of a network in every profile as a
    heatmap.

    A row for each source and a column for each profile, a sample or a contrast, each
    cell coloured by the source's score in the profile, on the scale of
    :func:`plot_pathway_scores`, from blue (low) through grey (zero) to red (high),
    symmetric about zero. The sources are those named by ``sources``, top to bottom,
    or else the ``top`` sources whose largest absolute score over the profiles is
    largest, in the order of the network. ``color_by`` groups the profiles under a
    strip of the groups' colours, and ``standardize`` centres and scales each
    source's scores across the profiles, as for :func:`plot_pathway_scores`. NaN
    scores, of sources with too few targets among a profile's genes, are dark grey.

    Draws the cells (gid ``"scores"``) and, with ``color_by``, the strip (gid
    ``"groups"``) and the lines between groups (gid ``"group boundaries"``).

    Args:
        activities: The activities of :func:`~cytepool.activity.ulm`,
            :func:`~cytepool.activity.mlm` or
            :func:`~cytepool.activity.weighted_aggregate`.
        sources: The sources to show, top to bottom; by default the ``top``
            sources.
        top: When ``sources`` is None, how many sources to show, at least 1; None
            shows every source.
        color_by: A column of ``activities.profiles``, such as ``"condition"`` for
            the activities of a cube's group, or one label per profile, at most
            eight levels.
        standardize: Whether to standardize each source's scores across the
            profiles.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes of the heatmap.

    Raises:
        TypeError: If ``activities`` is not an ``Activities``, an option has the
            wrong type, or an argument is or holds a numpy masked array.
        ValueError: If a source is not among the activities or is named twice, an
            option is out of range, the labels do not match the profiles, or
            ``color_by`` has more than eight levels.
        KeyError: If ``color_by`` names a column the profiles do not have.
    """
    _check_axes(ax)
    _check_activities(activities)
    scaled = checked_flag(standardize, "standardize")
    reject_masked(top, "top")
    count = None if top is None else positive_integer(top, "top")
    names = list(activities.sources)
    if sources is not None:
        chosen = _chosen_sources(sources, names)
    elif count is None:
        chosen = names
    else:
        with np.errstate(invalid="ignore"):
            largest = np.nanmax(np.abs(activities.scores), axis=0, initial=-np.inf)
        shown = np.argsort(-largest, kind="stable")[:count]
        chosen = [names[j] for j in np.sort(shown).tolist()]
    values = activities.scores[:, [names.index(name) for name in chosen]].T
    table = activities.profiles
    n_profiles = values.shape[1]
    profiles = _Samples(
        ids=[str(value) for value in table["profile_id"].tolist()],
        rows=list(range(n_profiles)),
        n_rows=n_profiles,
        variables={name: table[name] for name in table.columns},
        source="the activities' profiles",
    )
    order = np.arange(n_profiles)
    groups = None
    if color_by is not None:
        _check_variable(color_by, "color_by")
        labels, levels = profiles.labels(color_by, "color_by")
        hues = _hues(levels, "color_by", "merge levels or give fewer labels")
        position = {level: i for i, level in enumerate(levels)}
        order = np.argsort([position[label] for label in labels], kind="stable")
        groups = ([labels[i] for i in order.tolist()], hues)
    values = values[:, order]
    if scaled:
        values = _standardized_rows(values)
    ax = _axes(
        ax,
        (min(14.0, 3.5 + 0.22 * n_profiles), min(14.0, 1.8 + 0.28 * len(chosen))),
    )
    _draw_heatmap(
        ax,
        values,
        chosen,
        [profiles.ids[i] for i in order.tolist()],
        "Standardized activity" if scaled else _activity_label(activities),
        groups,
    )
    return ax


def plot_top_activities(
    activities: Activities,
    profile: str | None = None,
    *,
    top: int = 25,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the sources with the largest activity scores of one profile as bars.

    decoupler's ``pl.barplot`` (Badia-i-Mompel et al. 2022): the ``top`` sources with
    the largest absolute scores in the profile, in decreasing order of their scores
    from top to bottom, each bar as long as the score and coloured by it on the
    scale of :func:`plot_activities`, symmetric about zero. Sources without a score
    in the profile (NaN) are left out.

    Draws the bars (gid ``"activities"``).

    Args:
        activities: The activities of :func:`~cytepool.activity.ulm`,
            :func:`~cytepool.activity.mlm` or
            :func:`~cytepool.activity.weighted_aggregate`.
        profile: The ``profile_id`` of the profile to show; it may be left out when
            the activities have one profile.
        top: How many sources to show, at least 1.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``activities`` is not an ``Activities``, an option has the
            wrong type, or an argument is or holds a numpy masked array.
        ValueError: If ``profile`` is left out of several profiles, ``top`` is below
            1, or the profile has no score.
        KeyError: If no profile has the id ``profile``.
    """
    _check_axes(ax)
    _check_activities(activities)
    count = positive_integer(top, "top")
    ids = [str(value) for value in activities.profiles["profile_id"].tolist()]
    if profile is None:
        if len(ids) != 1:
            raise ValueError(
                f"the activities have {len(ids)} profiles; name the one to show with "
                f"profile, one of {quoted(ids)}"
            )
        row = 0
    else:
        if not isinstance(profile, str):
            reject_masked(profile, "profile")
            raise TypeError(
                f"profile must be a profile id (str), got {type(profile).__name__}"
            )
        if profile not in ids:
            raise KeyError(
                f"no profile {profile!r} among the activities: {quoted(ids)}"
            )
        row = ids.index(profile)
    scores = activities.scores[row]
    scored = np.flatnonzero(~np.isnan(scores))
    if scored.size == 0:
        raise ValueError(f"the profile {ids[row]!r} has no score")
    shown = scored[np.argsort(-np.abs(scores[scored]), kind="stable")[:count]]
    shown = shown[np.argsort(-scores[shown], kind="stable")]
    values = scores[shown]
    names = [activities.sources[j] for j in shown.tolist()]
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_diverging", _DIVERGING)
    limit = float(np.max(np.abs(values))) or 1.0
    shades = ramp(colors.Normalize(vmin=-limit, vmax=limit)(values))
    ax = _axes(ax, (5.0, min(14.0, 1.2 + 0.26 * values.size)))
    positions = np.arange(values.size, dtype=np.float64)
    ax.barh(positions, values, color=shades, height=0.8, gid="activities")
    ax.axvline(0.0, color=_INK, linewidth=0.8)
    ax.set_yticks(positions, names)
    ax.set_ylim(values.size - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(_activity_label(activities))
    ax.set_title(ids[row], loc="left")
    return ax


def _check_activities(activities: Any) -> None:
    if not isinstance(activities, Activities):
        raise TypeError(
            "activities must be an Activities, as ulm, mlm and weighted_aggregate "
            f"return, got {type(activities).__name__}"
        )


def _chosen_sources(sources: Any, names: list[str]) -> list[str]:
    """The sources named, checked against those of the activities."""
    reject_masked(sources, "sources")
    chosen = [sources] if isinstance(sources, str) else list(sources)
    for name in chosen:
        if not isinstance(name, str):
            raise TypeError(f"sources must hold source names (str), got {name!r}")
        if name not in names:
            raise ValueError(
                f"no source {name!r} among the activities; they hold {quoted(names)}"
            )
    if len(set(chosen)) != len(chosen):
        raise ValueError("sources names a source twice")
    if not chosen:
        raise ValueError("sources is empty; name at least one source")
    return chosen


def _activity_label(activities: Activities) -> str:
    """What the scores of the activities are, for an axis."""
    if activities.method != "weighted_aggregate":
        return "Activity (t-statistic)"
    kind = "mean" if activities.settings.get("statistic") == "mean" else "sum"
    if activities.settings.get("n_permutations"):
        return f"Normalized weighted {kind}"
    return f"Weighted {kind}"


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


# Replication across studies.

_AGREEMENT = {
    "pi1": "π₁ of the genes the discovery study finds",
    "sign_agreement": "Share of the genes the discovery study finds\nwith the same sign",
}


def plot_study_agreement(
    result: ReplicationResult,
    *,
    measure: str = "pi1",
    numbers: bool | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot how well each study replicates the genes that each other study finds.

    A heatmap of the pairs of studies of a
    :func:`~cytepool.replication.differential_expression_replication`: a row per
    discovery study and a column per replication study, in the collection's order,
    shaded from 0 to 1 by ``measure`` of the genes the discovery study finds and the
    replication study tested (``study_pairs``): pi1, Storey's estimate of the share of
    them with an effect in the replication study (Storey and Tibshirani 2003), or the
    share whose log2 fold changes have the same sign.
    A pair without such genes, and the diagonal, are pale.

    Draws the image (gid ``"agreement"``) and, when asked, each entry's value with the
    number of genes it covers (gid ``"numbers"``).

    Args:
        result: The replication of a group and contrast.
        measure: ``"pi1"`` or ``"sign_agreement"``.
        numbers: Whether to write each entry; by default, when there are at most 800.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a ReplicationResult or an option has the
            wrong type.
        ValueError: If ``measure`` is unknown.
    """
    _check_axes(ax)
    if not isinstance(result, ReplicationResult):
        raise TypeError(
            f"result must be a ReplicationResult, got {type(result).__name__}"
        )
    chosen = checked_choice(measure, "measure", tuple(_AGREEMENT))
    if numbers is not None:
        numbers = checked_flag(numbers, "numbers")
    studies = [
        study
        for study, analysed in zip(
            result.studies["study"].tolist(),
            result.studies["analysed"].tolist(),
            strict=True,
        )
        if analysed
    ]
    index = {study: position for position, study in enumerate(studies)}
    k = len(studies)
    shown = np.full((k, k), np.nan)
    counts = np.zeros((k, k), dtype=np.int64)
    pairs = result.study_pairs
    for discovery, replication, value, count in zip(
        pairs["discovery"].tolist(),
        pairs["replication"].tolist(),
        pairs[chosen].tolist(),
        pairs["n_found"].tolist(),
        strict=True,
    ):
        shown[index[discovery], index[replication]] = value
        counts[index[discovery], index[replication]] = count
    write = k * k <= _NUMBERED_ENTRIES if numbers is None else numbers
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_blues", _BLUES)
    ramp = ramp.with_extremes(bad=_PALE)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    ax = _axes(ax, (min(12.0, 3.0 + 0.9 * k), min(10.0, 1.8 + 0.8 * k)))
    image = ax.imshow(shown, cmap=ramp, norm=norm, aspect="auto", interpolation="none")
    image.set_gid("agreement")
    ax.set_xticks(range(k), studies, rotation=45, ha="right")
    ax.set_yticks(range(k), studies)
    ax.set_xlabel("Replication study")
    ax.set_ylabel("Discovery study")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if write:
        for row in range(k):
            for column in range(k):
                if row == column or np.isnan(shown[row, column]):
                    continue
                face = ramp(norm(shown[row, column]))
                ax.text(
                    column,
                    row,
                    f"{shown[row, column]:.2f}\n{counts[row, column]:,} genes",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if _luminance(face) < 0.3 else _INK,
                    gid="numbers",
                )
    settings = result.settings
    ax.set_title(
        f"{settings['group']}, {settings['contrast']}: genes at adjusted p < "
        f"{settings['alpha']:g}",
        fontsize=10,
    )
    ax.figure.colorbar(image, ax=ax, label=_AGREEMENT[chosen])
    return ax


def plot_replicated_genes(
    result: ReplicationResult, *, ax: "Axes | None" = None
) -> "Axes":
    """Plot how many genes each number of studies finds.

    For every k from 1 to the number of studies that analysed the group and contrast
    of a :func:`~cytepool.replication.differential_expression_replication`, a bar of
    the genes that exactly k studies find (adjusted p-value below its ``alpha`` in
    each), split into those that every one of them finds in the same direction and
    those found up in some and down in others, with the numbers above. Most genes
    that one study finds are usually found by no other (Nakatsuka et al. 2025).

    Draws the bars of genes found in one direction (gid ``"same direction"``) and in
    both (gid ``"both directions"``), and their totals (gid ``"numbers"``).

    Args:
        result: The replication of a group and contrast.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a ReplicationResult.
    """
    _check_axes(ax)
    if not isinstance(result, ReplicationResult):
        raise TypeError(
            f"result must be a ReplicationResult, got {type(result).__name__}"
        )
    genes = result.genes
    found = np.asarray(genes["n_found"], dtype=np.int64)
    one_way = np.maximum(genes["n_found_up"], genes["n_found_down"]) == found
    k = len(result.found_in)
    studies = np.arange(1, k + 1)
    same = np.array([np.count_nonzero((found == j) & one_way) for j in studies])
    both = np.array([np.count_nonzero((found == j) & ~one_way) for j in studies])
    ax = _axes(ax, (min(10.0, 3.0 + 0.7 * k), 4.0))
    ax.bar(studies, same, width=0.7, color=_BLUE, gid="same direction")
    ax.bar(studies, both, width=0.7, bottom=same, color=_ORANGE, gid="both directions")
    tallest = int((same + both).max(initial=0))
    for x, total in zip(studies.tolist(), (same + both).tolist(), strict=True):
        ax.text(
            x,
            total + 0.01 * max(tallest, 1),
            f"{total:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            gid="numbers",
        )
    ax.set_xticks(studies)
    ax.set_xlim(0.4, k + 0.6)
    ax.set_ylim(0.0, 1.12 * max(tallest, 1))
    ax.set_xlabel("Studies that find the gene")
    settings = result.settings
    ax.set_ylabel(f"Genes at adjusted p < {settings['alpha']:g}")
    ax.set_title(f"{settings['group']}, {settings['contrast']}", fontsize=10)
    _legend(
        ax,
        [_patch("Same direction", _BLUE), _patch("Up and down", _ORANGE)],
    )
    return ax


# Variance partitioning.

# plotVarPart's violins: a kernel density at 512 points from the smallest to the
# largest value, every violin scaled to the same largest width, around a box of the
# quartiles 0.07 wide whose whiskers reach the furthest values within 1.5
# interquartile ranges, its labels turned by 20 degrees (the settings of its ggplot2
# layers, found by probing variancePartition 1.36.2).
_VIOLIN_POINTS = 512
_VIOLIN_WIDTH = 0.9
_BOX_WIDTH = 0.07
_WHISKER_REACH = 1.5
_LABEL_ANGLE = 20.0
_DENSITY_BLOCK = 64  # grid points of the density computed at a time
_FRACTIONS = "a VarianceFractions, a VariancePartitioningResult or one of its tables"
_FEATURE_COLUMNS = ("feature_id", "symbol")


@dataclass(frozen=True)
class _Partition:
    """The fractions of variance that a figure shows, and the names of the features."""

    fractions: FloatArray  # features x variables
    variables: list[str]
    ids: list[str] | None  # the feature ids of a table
    symbols: list[Any] | None  # the symbols of a table, None where missing


def plot_variance_fractions(
    data: Any,
    *,
    group: str | None = None,
    sort: bool = True,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the fraction of variance each variable explains over the genes, as violins.

    variancePartition's ``plotVarPart`` (Hoffman and Schadt 2016): for each variable
    of a variance partition, and the residuals, the distribution over the genes of
    the percentage of variance it explains, as a violin (Hintze and Nelson 1998)
    around a narrow box plot. A violin is a Gaussian kernel density of the
    percentages, with Silverman's (1986) rule-of-thumb bandwidth (R's ``bw.nrd0``:
    0.9 min(sd, IQR / 1.34) n^(-1/5), the standard deviation alone where the
    quartiles coincide), drawn from the smallest to the largest percentage; every
    violin has the same largest width, as ``plotVarPart`` scales them. The box spans
    the quartiles, with the median, whiskers to the furthest percentages within 1.5
    interquartile ranges of the box, and the percentages beyond them as points.
    With ``sort``, the variables are in decreasing order of their median, the
    residuals last, as ``plotVarPart(sortCols(...))`` orders them. Genes without
    fractions (NaN) are left out.

    Draws the violins (gid ``"violins"``, labelled with their variable; none for a
    variable whose percentages are all equal), the boxes (gid ``"boxes"``), the
    medians (gid ``"medians"``), the whiskers (gid ``"whiskers"``, the lower one of a
    variable first) and the outlying percentages (gid ``"outliers"``, one collection
    per variable).

    Args:
        data: The fractions: a :class:`~cytepool.variance_partitioning.VarianceFractions`,
            a :class:`~cytepool.variance_partitioning.VariancePartitioningResult`
            (with ``group``), or a table of one.
        group: The group (cell type) of a result; it may be left out when one group
            was fitted.
        sort: Whether to order the variables by their median, the residuals last;
            False keeps the order of the design.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``data`` or an option has the wrong type.
        ValueError: If no gene has fractions, there are more than eight variables
            besides the residuals, a table has no ``residuals`` column, or
            ``group`` is needed and not given, or given for data of one group.
        KeyError: If ``group`` was not analysed or not fitted.
    """
    _check_axes(ax)
    name = _optional_text(group, "group")
    ordered = checked_flag(sort, "sort")
    partition = _partition(data, name)
    finite = np.all(np.isfinite(partition.fractions), axis=1)
    if not finite.any():
        raise ValueError(
            "no gene has fractions of variance to plot: every gene's mixed model "
            "lacks a fit"
        )
    fractions = partition.fractions[finite]
    order = _variable_order(fractions, partition.variables, ordered)
    names = [partition.variables[j] for j in order]
    colors = _variable_colors(names)
    percent = 100.0 * fractions[:, order]
    labels = [_variable_label(variable) for variable in names]
    ax = _axes(ax, (min(14.0, 2.0 + 0.9 * len(order)), 4.5))
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        x = i + 1.0
        values = percent[:, i]
        _draw_violin(ax, x, values, color, label)
        _draw_box(ax, x, values)
    ax.set_xticks(
        range(1, len(order) + 1),
        labels,
        rotation=_LABEL_ANGLE,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(0.4, len(order) + 0.6)
    ax.set_ylim(-3.0, 103.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Variance explained (%)")
    return ax


def plot_gene_variance_fractions(
    data: Any,
    genes: Any,
    *,
    group: str | None = None,
    labels: Sequence[Any] | None = None,
    sort: bool = True,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the fractions of variance of chosen genes as stacked bars.

    variancePartition's ``plotPercentBars``: one horizontal bar per gene, the first
    at the top, divided into the percentage of the gene's variance that each
    variable explains, the residuals last. The variables are in the order of
    :func:`plot_variance_fractions`: with ``sort``, in decreasing order of their
    median over every gene of the data, as ``plotPercentBars(sortCols(x)[genes, ])``
    orders them.

    Draws the segments of each variable (gid ``"fractions"``, one bar container per
    variable, labelled with it), with a legend of the variables.

    Args:
        data: The fractions, as for :func:`plot_variance_fractions`.
        genes: The genes: feature ids, or symbols, of a table or result; names given
            in ``labels``; or positions among the features.
        group: The group (cell type) of a result; it may be left out when one group
            was fitted.
        labels: One name per feature of the data, to find and label the genes by;
            a table's genes are named by their symbol, or else their id.
        sort: Whether to order the variables by their median, the residuals last.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or ``genes`` or ``labels`` is
            or holds a numpy masked array.
        ValueError: As :func:`plot_variance_fractions`, or if ``genes`` is empty or
            names a gene twice, a symbol names several genes, a gene has no
            fractions, or ``labels`` has the wrong length.
        KeyError: If a gene is not among the data's, or ``group`` was not analysed
            or not fitted.
        IndexError: If a position is out of range.
    """
    _check_axes(ax)
    reject_masked(genes, "genes")
    if labels is not None:
        reject_masked(labels, "labels")
    name = _optional_text(group, "group")
    ordered = checked_flag(sort, "sort")
    partition = _partition(data, name)
    n = partition.fractions.shape[0]
    given = None if labels is None else _gene_labels(labels, n)
    rows, shown = _chosen_genes(genes, partition, given)
    chosen = partition.fractions[rows]
    missing = [shown[i] for i in range(len(rows)) if not np.all(np.isfinite(chosen[i]))]
    if missing:
        raise ValueError(
            f"{quoted(missing)} {'has' if len(missing) == 1 else 'have'} no fractions "
            "of variance: the mixed model of such a gene lacks a fit"
        )
    finite = np.all(np.isfinite(partition.fractions), axis=1)
    order = _variable_order(partition.fractions[finite], partition.variables, ordered)
    names = [partition.variables[j] for j in order]
    colors = _variable_colors(names)
    percent = 100.0 * chosen[:, order]
    ax = _axes(ax, (7.0, min(12.0, 1.4 + 0.35 * len(rows))))
    positions = np.arange(len(rows), dtype=np.float64)[::-1]
    left = np.zeros(len(rows))
    handles = []
    for j, (variable, color) in enumerate(zip(names, colors, strict=True)):
        label = _variable_label(variable)
        ax.barh(
            positions,
            percent[:, j],
            left=left,
            height=0.8,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            gid="fractions",
        )
        left = left + percent[:, j]
        handles.append(_patch(label, color))
    ax.set_yticks(positions, shown)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Variance explained (%)")
    _legend(ax, handles)
    return ax


def _partition(data: Any, group: str | None) -> _Partition:
    """The fractions of variance of the data, with the names of the features."""
    if isinstance(data, VariancePartitioningResult):
        data = data.table(_partitioned_group(data, group))
    elif group is not None:
        raise ValueError(
            "group names a group of a VariancePartitioningResult; the data hold one "
            "group's fractions"
        )
    if isinstance(data, VarianceFractions):
        return _Partition(
            np.asarray(data.fractions, dtype=np.float64),
            list(data.variables),
            None,
            None,
        )
    if isinstance(data, Table):
        if RESIDUALS not in data:
            raise ValueError(
                f"the table has no {RESIDUALS!r} column: give a table of a "
                "VariancePartitioningResult, whose columns besides feature_id and "
                "symbol are the fractions of every variable"
            )
        variables = [
            column for column in data.columns if column not in _FEATURE_COLUMNS
        ]
        fractions = np.column_stack(
            [np.asarray(data[column], dtype=np.float64) for column in variables]
        )
        ids = data["feature_id"].tolist() if "feature_id" in data else None
        symbols = data["symbol"].tolist() if "symbol" in data else None
        return _Partition(fractions, variables, ids, symbols)
    raise TypeError(f"data must be {_FRACTIONS}, got {type(data).__name__}")


def _partitioned_group(result: VariancePartitioningResult, group: str | None) -> str:
    """The group of a result that is meant: the one given, or the only one fitted."""
    if group is not None:
        return group
    fitted = list(result.tables)
    if len(fitted) != 1:
        raise ValueError(
            f"the result has {len(fitted)} fitted groups; name one with group, one of "
            f"{quoted(fitted) or '(none)'}"
        )
    return fitted[0]


def _variable_order(
    fractions: FloatArray, variables: Sequence[str], ordered: bool
) -> list[int]:
    """The variables in decreasing order of their median fraction, ties in the order
    given, the residuals last; or in the order given."""
    others = [j for j, name in enumerate(variables) if name != RESIDUALS]
    residual = [j for j, name in enumerate(variables) if name == RESIDUALS]
    if ordered and fractions.shape[0]:
        medians = np.median(fractions[:, others], axis=0)
        others = [others[k] for k in np.argsort(-medians, kind="stable")]
    return others + residual


def _variable_colors(variables: list[str]) -> list[str]:
    """A hue for each variable in order, and pale grey for the residuals."""
    others = [name for name in variables if name != RESIDUALS]
    hues = _hues(
        others,
        "the variance partition",
        "partition the variance among fewer variables, or merge some",
    )
    return [_PALE if name == RESIDUALS else hues[name] for name in variables]


def _variable_label(variable: str) -> str:
    return "Residuals" if variable == RESIDUALS else variable


def _chosen_genes(
    genes: Any, partition: _Partition, labels: list[str] | None
) -> tuple[list[int], list[str]]:
    """The rows of the genes asked for, and the names to show for them."""
    n = partition.fractions.shape[0]
    if isinstance(genes, str | numbers.Integral):
        genes = [genes]
    if not isinstance(genes, Sequence | np.ndarray):
        raise TypeError(
            "genes must be a list of gene ids, symbols or positions, got "
            f"{type(genes).__name__}"
        )
    if len(genes) == 0:
        raise ValueError("genes is empty; give at least one gene")
    names: list[str] | None = labels
    if names is None and partition.ids is not None:
        symbols = partition.symbols or [None] * n
        names = [
            str(symbol) if symbol is not None else str(identifier)
            for symbol, identifier in zip(symbols, partition.ids, strict=True)
        ]
    rows: list[int] = []
    for gene in list(genes):
        if isinstance(gene, bool | np.bool_):
            raise TypeError("genes must be gene ids, symbols or positions, not bool")
        if isinstance(gene, numbers.Integral):
            position = int(gene)
            if not -n <= position < n:
                raise IndexError(
                    f"gene position {position} is out of range for {n} genes"
                )
            row = position % n
        elif isinstance(gene, str):
            row = _gene_row(gene, partition, labels)
        else:
            raise TypeError(
                f"genes must be gene ids, symbols or positions, got "
                f"{type(gene).__name__} {gene!r}"
            )
        if row in rows:
            raise ValueError(f"the gene {gene!r} is given more than once")
        rows.append(row)
    shown = [names[row] if names is not None else str(row) for row in rows]
    return rows, shown


def _gene_row(gene: str, partition: _Partition, labels: list[str] | None) -> int:
    """The row of a gene named by its id, its symbol or a label."""
    for kind, values in (
        ("label", labels),
        ("feature id", partition.ids),
        ("symbol", partition.symbols),
    ):
        if values is None:
            continue
        found = [row for row, value in enumerate(values) if value == gene]
        if len(found) > 1:
            raise ValueError(
                f"the {kind} {gene!r} names {len(found)} genes; give their feature ids"
            )
        if found:
            return found[0]
    if labels is None and partition.ids is None:
        raise KeyError(
            f"gene {gene!r} cannot be found: the fractions name no genes; give "
            "positions, or one name per gene with labels"
        )
    raise KeyError(f"no gene {gene!r} among the {partition.fractions.shape[0]} genes")


def _draw_violin(
    ax: "Axes", x: float, values: FloatArray, color: str, label: str
) -> None:
    """The violin of the values at x, unless they span no range."""
    low, high = float(np.min(values)), float(np.max(values))
    if values.size < 2 or not high > low:
        return
    grid = np.linspace(low, high, _VIOLIN_POINTS)
    density = _kernel_density(values, grid, _bandwidth(values))
    half = 0.5 * _VIOLIN_WIDTH * density / np.max(density)
    ax.fill_betweenx(
        grid,
        x - half,
        x + half,
        facecolor=color,
        edgecolor=_INK,
        linewidth=0.6,
        label=label,
        gid="violins",
    )


def _bandwidth(values: FloatArray) -> float:
    """Silverman's rule of thumb, R's bw.nrd0, for values that span a range.

    Where the quartiles coincide, bw.nrd0 takes the standard deviation alone (found
    by probing R 4.4.3; its help page says only that the result stays positive).
    """
    spread = float(np.std(values, ddof=1))
    first, third = np.percentile(values, [25.0, 75.0])
    quartiles = float(third - first) / 1.34
    if quartiles > 0.0:
        spread = min(spread, quartiles)
    return 0.9 * spread * math.pow(values.size, -0.2)


def _kernel_density(
    values: FloatArray, grid: FloatArray, bandwidth: float
) -> FloatArray:
    """The Gaussian kernel density estimate of the values at the grid points."""
    density = np.empty(grid.size)
    for start in range(0, grid.size, _DENSITY_BLOCK):
        z = (grid[start : start + _DENSITY_BLOCK, np.newaxis] - values) / bandwidth
        density[start : start + _DENSITY_BLOCK] = np.sum(np.exp(-0.5 * z * z), axis=1)
    scaled: FloatArray = density / (values.size * bandwidth * math.sqrt(2.0 * math.pi))
    return scaled


def _draw_box(ax: "Axes", x: float, values: FloatArray) -> None:
    """A narrow box plot of the values at x."""
    first, median, third = np.percentile(values, [25.0, 50.0, 75.0])
    reach = _WHISKER_REACH * (third - first)
    within = values[(values >= first - reach) & (values <= third + reach)]
    low = float(np.min(within)) if within.size else float(first)
    high = float(np.max(within)) if within.size else float(third)
    outlying = values[(values < first - reach) | (values > third + reach)]
    half = _BOX_WIDTH / 2.0
    box = _module("matplotlib.patches").Rectangle(
        (x - half, first),
        _BOX_WIDTH,
        third - first,
        facecolor=_GREY,
        edgecolor=_INK,
        linewidth=0.6,
        zorder=3,
        gid="boxes",
    )
    ax.add_patch(box)
    ax.plot([x, x], [first, low], color=_INK, linewidth=0.6, zorder=3, gid="whiskers")
    ax.plot([x, x], [third, high], color=_INK, linewidth=0.6, zorder=3, gid="whiskers")
    ax.plot(
        [x - half, x + half],
        [median, median],
        color=_INK,
        linewidth=1.2,
        zorder=4,
        gid="medians",
    )
    ax.scatter(
        np.full(outlying.size, x),
        outlying,
        s=6.0,
        color=_INK,
        linewidths=0,
        zorder=4,
        gid="outliers",
    )


# SumRank and held-out prediction.

_METHOD_LABELS = {"sumrank": "SumRank", "meta_analysis": "Meta-analysis"}


def plot_sumrank(
    result: SumRankResult,
    *,
    direction: str = "up",
    alpha: float = 0.05,
    n_labels: int = 20,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot the SumRank p-value of every gene, the genes in alphabetical order.

    The Manhattan plot that the authors of SumRank draw (Nakatsuka et al. 2025,
    Figure 4, and their ``MakeManhattanPlot``): a point per gene ranked, at -log10 of
    its p-value (``up_p_value`` or ``down_p_value``, from permutations when they were
    run), the genes in alphabetical order of their symbol, or of their id without
    one. The genes whose adjusted p-value is below ``alpha`` are coloured, red for
    genes up and blue for genes down, the ``n_labels`` with the smallest p-values are
    named, and a dotted line lies at the largest -log10 p-value of the other genes,
    where the authors draw it, when some gene passes.

    Draws the genes (gid ``"genes"``), those that pass (gid ``"significant"``), the
    line (gid ``"threshold"``) and the names (gid ``"labels"``).

    Args:
        result: The result of
            :func:`~cytepool.sumrank.differential_expression_sumrank`.
        direction: ``"up"`` or ``"down"``.
        alpha: The adjusted p-value below which genes pass, in (0, 1).
        n_labels: The most genes named, at least 0.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``result`` is not a SumRankResult or an option has the wrong
            type.
        ValueError: If ``direction`` is unknown or ``alpha`` is not in (0, 1).
    """
    _check_axes(ax)
    if not isinstance(result, SumRankResult):
        raise TypeError(f"result must be a SumRankResult, got {type(result).__name__}")
    side = checked_choice(direction, "direction", ("up", "down"))
    level = _fraction(alpha, "alpha")
    reject_masked(n_labels, "n_labels")
    named = integral_number(n_labels, "n_labels")
    if named < 0:
        raise ValueError(f"n_labels must be at least 0, got {named}")
    table = result.table
    ids = [str(value) for value in table["feature_id"].tolist()]
    names = ids
    if "symbol" in table:
        names = [
            str(symbol) if symbol is not None else identifier
            for symbol, identifier in zip(table["symbol"].tolist(), ids, strict=True)
        ]
    order = sorted(range(len(names)), key=lambda i: (names[i], ids[i]))
    p = np.asarray(table[f"{side}_p_value"], dtype=np.float64)[order]
    adjusted = np.asarray(table[f"{side}_adjusted_p_value"], dtype=np.float64)[order]
    heights = -np.log10(np.maximum(p, _TINY))
    passed = adjusted < level
    x = np.arange(len(order), dtype=np.float64)
    color = _UP if side == "up" else _DOWN
    ax = _axes(ax, (8.0, 4.0))
    ax.scatter(
        x[~passed], heights[~passed], s=_DENSE, color=_INK, linewidths=0, gid="genes"
    )
    ax.scatter(
        x[passed],
        heights[passed],
        s=_DENSE,
        color=color,
        linewidths=0,
        gid="significant",
    )
    if passed.any() and (~passed).any():
        ax.axhline(
            float(heights[~passed].max()),
            color=color,
            linewidth=1.2,
            linestyle=":",
            gid="threshold",
        )
    shown = np.flatnonzero(passed)
    shown = shown[np.lexsort((shown, p[shown]))][:named]
    for i in sorted(shown.tolist()):
        ax.annotate(
            names[order[i]],
            (x[i], heights[i]),
            xytext=(0.0, 3.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            gid="labels",
        )
    ax.set_xticks([])
    ax.set_xlim(-1.0, len(order))
    ax.set_ylim(0.0, max(1.0, float(heights.max(initial=0.0))) * 1.12)
    settings = result.settings
    source = (
        f"{settings['n_permutations']:,} permutations"
        if settings["p_values"] == "permutation"
        else "Irwin-Hall"
    )
    ax.set_xlabel(f"Genes, in alphabetical order ({len(order):,})")
    ax.set_ylabel(f"-log10 p-value, genes {side} ({source})")
    _legend(
        ax,
        [
            _key(f"Adjusted p < {level:g} ({int(passed.sum()):,})", color=color),
            _key("Other genes", color=_INK),
        ],
    )
    return ax


def plot_held_out_auc(
    prediction: HeldOutPrediction, *, ax: "Axes | None" = None
) -> "Axes":
    """Plot the AUC of every held-out study, by the method that chose its genes.

    A row per study of a :func:`~cytepool.held_out_prediction.held_out_prediction`,
    in the collection's order from the top, with a point for each method at the AUC
    of the study's cases against its controls, scored by the genes that the method
    chose from the other studies; the legend gives each method's mean AUC, and a
    dotted line marks 0.5, the AUC of a score that does not separate them, as the
    paper compares its methods (Nakatsuka et al. 2025, Table 1). A study without an
    AUC keeps its row, marked "no AUC".

    Draws the points of each method (gid its name, ``"sumrank"`` or
    ``"meta_analysis"``) and the line (gid ``"chance"``).

    Args:
        prediction: The result of
            :func:`~cytepool.held_out_prediction.held_out_prediction`.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If ``prediction`` is not a HeldOutPrediction.
    """
    _check_axes(ax)
    if not isinstance(prediction, HeldOutPrediction):
        raise TypeError(
            f"prediction must be a HeldOutPrediction, got {type(prediction).__name__}"
        )
    table = prediction.auc
    studies = list(dict.fromkeys(table["study"].tolist()))
    methods = prediction.settings["methods"]
    rows = {study: len(studies) - 1 - i for i, study in enumerate(studies)}
    ax = _axes(ax, (6.5, min(12.0, 1.6 + 0.4 * len(studies))))
    ax.axvline(0.5, color=_GREY, linewidth=1, linestyle=":", zorder=0, gid="chance")
    handles = []
    offsets = np.linspace(-0.12, 0.12, len(methods)) if len(methods) > 1 else [0.0]
    values = np.asarray(table["auc"], dtype=np.float64)
    with_auc = set(table["study"][~np.isnan(values)].tolist())
    for method, hue, marker, offset in zip(
        methods, _CATEGORICAL, _MARKERS, offsets, strict=False
    ):
        chosen = table["method"] == method
        auc = values[chosen]
        names = table["study"][chosen].tolist()
        known = ~np.isnan(auc)
        y = np.array([rows[name] for name in names], dtype=np.float64) + offset
        ax.scatter(
            auc[known],
            y[known],
            s=_SPARSE,
            color=hue,
            marker=marker,
            zorder=3,
            clip_on=False,  # an AUC of 0 or 1 lies on the frame
            gid=method,
        )
        mean = f"mean {auc[known].mean():.2f}" if known.any() else "no AUC"
        label = _METHOD_LABELS.get(method, method)
        handles.append(_key(f"{label} ({mean})", color=hue, marker=marker))
    labels = [study if study in with_auc else f"{study} (no AUC)" for study in studies]
    ax.set_yticks([rows[study] for study in studies], labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(studies) - 0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("AUC of the held-out study's cases against its controls")
    _legend(ax, handles)
    return ax


# Principal components and dot plots.

# DESeq2's plotPCA takes the 500 genes of largest variance.
_PCA_TOP = 500
# scanpy's dot plots: the area of the dot of the largest fraction, in points squared,
# and the power of the fraction that sets a dot's area.
_DOT_AREA = 200.0
_DOT_EXPONENT = 1.5
_DOT_SCALES = ("gene", "group")


def plot_pca(
    data: Any,
    *,
    group: str | None = None,
    color_by: str | Sequence[Any] | None = None,
    marker_by: str | Sequence[Any] | None = None,
    labels: bool | Sequence[Any] = False,
    top: int | None = None,
    components: Sequence[int] = (1, 2),
    min_cells: int | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot samples on two principal components of their most variable genes.

    DESeq2's ``plotPCA`` (Love et al. 2014): the samples' coordinates on two components
    of :func:`~cytepool.principal_components.principal_components`, which takes the
    ``top`` genes of largest variance over the samples, centres each on its mean, and
    decomposes them by singular values (Pearson 1901); the sign of each component makes
    its largest coordinate positive. The axes are labelled as ``plotPCA`` labels them,
    with each component's share of the variance rounded to a whole percent, as in
    "PC1: 45% variance", and drawn to the same scale, as ``plotPCA`` draws them, so that
    the distances between samples are drawn true. The coordinates are those of
    ``principal_components``; pass its result as ``data`` to draw the components it
    holds.

    Draws the samples as points (gid ``"samples"``), coloured by ``color_by`` and
    marked by ``marker_by`` as :func:`plot_mds` draws them, with a legend of their
    levels, and labels the samples (gid ``"sample labels"``) when asked.

    Args:
        data: A :class:`~cytepool.principal_components.PrincipalComponents`, or what
            ``principal_components`` takes: a :class:`~cytepool.cube.Cube`, a
            :class:`~cytepool.stats.voom.VoomResult`, or ``(n_samples, n_genes)``
            log-expression values, finite.
        group: The group (cell type) of a cube to plot; it may be left out when the
            cube has one group. Only cubes take it.
        color_by: The variable whose levels colour the points: a column of the
            samples' table (a cube's, or that of a PrincipalComponents), or one label
            per sample (of the cube, for a cube). With more than three levels and no
            ``marker_by``, each level also gets a marker.
        marker_by: The variable whose levels mark the points, as ``color_by``; at most
            12 levels.
        labels: True to write each sample's id next to it (numbers from 1 for an
            array), or one label per sample.
        top: The number of genes of largest variance used, at least 1; 500 by
            default, as DESeq2's ``ntop``. Not for a PrincipalComponents.
        components: The two components to plot, counted from 1 in decreasing order
            of the variance they explain.
        min_cells: The fewest cells a sample needs in the group to be plotted, 10 by
            default as for ``limma_voom``; for a cube that records cell numbers.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As ``principal_components``, or if the data have fewer components
            than asked for, ``group``, ``top`` or ``min_cells`` is given with a
            PrincipalComponents, or a sample variable has the wrong length, a missing
            label or too many levels.
        KeyError: If the group, or a variable named, is not in the data.
    """
    _check_axes(ax)
    first, second = _component_pair(components)
    _check_variable(color_by, "color_by")
    _check_variable(marker_by, "marker_by")
    if not isinstance(labels, bool | np.bool_):
        reject_masked(labels, "labels")
    if isinstance(data, PrincipalComponents):
        for name, value in (("group", group), ("top", top), ("min_cells", min_cells)):
            if value is not None:
                reject_masked(value, name)
                raise ValueError(
                    f"{name} applies to the values that principal_components takes; "
                    "data holds its components already"
                )
        result = data
    else:
        result = principal_components(
            data,
            group=group,
            top=_PCA_TOP if top is None else top,
            min_cells=min_cells,
        )
    if result.n_components < max(first, second):
        raise ValueError(
            f"components {first} and {second} need {max(first, second)} components; "
            f"the data have {result.n_components}, one fewer than the samples and at "
            "most the genes used"
        )
    samples = _component_samples(data, result)
    x = np.asarray(result.coordinates[:, first - 1], dtype=np.float64)
    y = np.asarray(result.coordinates[:, second - 1], dtype=np.float64)
    ax = _axes(ax, (6.5, 4.5))
    _draw_samples(ax, x, y, samples, color_by, marker_by, labels)
    share = result.variance_explained
    ax.set_xlabel(f"PC{first}: {100 * share[first - 1]:.0f}% variance")
    ax.set_ylabel(f"PC{second}: {100 * share[second - 1]:.0f}% variance")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _component_pair(value: Any) -> tuple[int, int]:
    reject_masked(value, "components")
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(
            f"components must be a pair of components, such as (1, 2), got {value!r}"
        )
    first = positive_integer(value[0], "components[0]")
    second = positive_integer(value[1], "components[1]")
    if first == second:
        raise ValueError(f"components must be two different components, got {value!r}")
    return first, second


def _component_samples(data: Any, result: PrincipalComponents) -> _Samples:
    """The samples of principal components, with the variables of their table: a
    cube's sample table for a cube, so that labels follow its samples."""
    table = result.samples
    if isinstance(data, Cube):
        position = {
            value: i for i, value in enumerate(data.samples["sample_id"].tolist())
        }
        rows = [position[value] for value in table["sample_id"].tolist()]
        return _cube_samples(data, np.array(rows, dtype=np.intp))
    n = len(table)
    ids = [str(value) for value in table["sample_id"].tolist()]
    if not isinstance(data, PrincipalComponents):
        return _Samples(ids, list(range(n)), n)
    return _Samples(
        ids=ids,
        rows=list(range(n)),
        n_rows=n,
        variables={name: table[name] for name in table.columns},
        source="the samples' table",
    )


def _draw_samples(
    ax: "Axes",
    x: FloatArray,
    y: FloatArray,
    samples: _Samples,
    color_by: Any,
    marker_by: Any,
    labels: Any,
) -> None:
    """Samples as points, coloured and marked by their variables, as plot_mds draws
    them: a hue per level of color_by, a marker per level of marker_by, or of color_by
    when it has more levels than scatter plots keep apart; labels when asked; and a
    legend of the levels."""
    n_samples = x.size
    colors = None if color_by is None else samples.labels(color_by, "color_by")
    markers = None if marker_by is None else samples.labels(marker_by, "marker_by")
    remedy = "give it as marker_by, or as labels"
    hues = {} if colors is None else _hues(colors[1], "color_by", remedy)
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


def plot_dotplot(
    cube: Cube,
    genes: Any,
    *,
    groups: Sequence[str] | None = None,
    samples: Any = None,
    standard_scale: str | None = None,
    mean_only_expressed: bool = False,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot how many cells of each group express each gene, and how much, as dots.

    scanpy's dot plot (``scanpy.pl.dotplot``; Wolf et al. 2018), drawn from the sums
    that a cube keeps instead of from the cells: for each group (cell type), a row,
    and each gene, a column, the fraction of the group's cells with a positive count,
    ``detected`` over ``n_cells``, sets the area of the dot, and the mean of the cells'
    log-normalized values, ``log_sum`` over ``n_cells``, its colour. Both are pooled
    over the samples, every sample's sums added before dividing, so that they are the
    fractions and means scanpy computes from the cells the cube was pooled from (with
    its default ``expression_cutoff`` of 0); ``samples`` pools some of them only.
    A dot's area is 200 (f / f_max)^1.5 points squared, f its fraction and f_max the
    largest fraction shown rounded up to a tenth, as scanpy sizes its dots, and the
    key of the areas shows fractions in scanpy's steps. The colours run from the
    smallest to the largest value shown.

    ``standard_scale`` scales the means of each gene over the groups (``"gene"``,
    scanpy's ``"var"``), or of each group over the genes (``"group"``), to run from 0
    to 1, (m - min) / (max - min), 0 where they are all equal, as scanpy does.
    ``mean_only_expressed`` takes the mean over the cells that express the gene,
    ``log_sum`` over ``detected`` (0 where none does), as scanpy's option of that name;
    it needs a cube whose transform maps a zero count to zero, one whose normalization
    records a pseudocount of 1, as :func:`~cytepool.cube.from_cells` records it. Genes
    given as a mapping are
    drawn in groups, under a bracket labelled with the group's name, as scanpy draws
    ``var_names`` given as a dict.

    Draws the dots (gid ``"dots"``: their sizes are the areas, and their array the
    values that the colours show) and, for genes in groups, the brackets (gid
    ``"gene groups"``) and their names (gid ``"gene group names"``).

    Args:
        cube: A cube with cell numbers and the layers ``detected`` and ``log_sum``,
            as :func:`~cytepool.cube.from_cells` builds it.
        genes: The genes, the columns from left to right: feature ids or symbols of
            the cube's features; or a mapping of names to such lists, for groups of
            genes.
        groups: The groups, the rows from the top; by default every group, in the
            cube's order.
        samples: The samples whose cells are pooled: sample ids, or a boolean mask
            over the cube's samples; by default every sample.
        standard_scale: ``"gene"``, ``"group"`` or None.
        mean_only_expressed: Whether the mean is taken over the cells that express
            the gene only.
        ax: The Axes to draw into, or None for a new figure.

    Returns:
        The Axes.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the cube lacks cell numbers, ``detected`` or ``log_sum``, a
            group has no cells in the samples pooled, ``genes``, ``groups`` or
            ``samples`` is empty or names one twice, a symbol names several genes,
            ``mean_only_expressed`` is asked of a cube without a pseudocount of 1, or
            an option is out of range.
        KeyError: If a gene, group or sample is not in the cube.
    """
    _check_axes(ax)
    if not isinstance(cube, Cube):
        raise TypeError(f"cube must be a Cube, got {type(cube).__name__}")
    scale = (
        None
        if standard_scale is None
        else checked_choice(standard_scale, "standard_scale", _DOT_SCALES)
    )
    expressed_only = checked_flag(mean_only_expressed, "mean_only_expressed")
    lacking = [name for name in ("detected", "log_sum") if name not in cube.layer_names]
    if cube.n_cells is None or lacking:
        what = (
            "cell numbers"
            if cube.n_cells is None
            else f"the layer(s) {quoted(lacking)}"
        )
        raise ValueError(
            f"the cube lacks {what}; a dot plot needs cell numbers and the layers "
            "'detected' and 'log_sum', as from_cells computes them"
        )
    columns, names, gene_groups = _dot_genes(cube, genes)
    rows = _chosen_ids(groups, cube.groups["group_id"].tolist(), "groups", "group")
    pooled = _pooled_samples(cube, samples)
    if expressed_only:
        normalization = cube.normalization or {}
        if normalization.get("pseudocount") != 1:
            raise ValueError(
                "mean_only_expressed needs a cube whose transform maps a zero count to "
                "zero, one whose normalization records a pseudocount of 1, as from_cells "
                f"records it; this cube's normalization is {normalization or None!r}"
            )
    fractions, means = _dot_values(cube, rows, columns, pooled, expressed_only)
    shown = (
        means if scale is None else _min_max_scaled(means, 0 if scale == "gene" else 1)
    )

    largest = float(fractions.max())
    f_max = math.ceil(largest * 10) / 10 if largest > 0.0 else 1.0
    areas = _DOT_AREA * (fractions / f_max) ** _DOT_EXPONENT
    n_groups, n_genes = fractions.shape
    colors = _module("matplotlib.colors")
    ramp = colors.LinearSegmentedColormap.from_list("cytepool_blues", _BLUES)
    low, high = float(shown.min()), float(shown.max())
    norm = colors.Normalize(vmin=low, vmax=high if high > low else low + 1.0)
    width = min(14.0, 3.0 + 0.4 * n_genes)
    ax = _axes(ax, (width, min(12.0, max(4.0, 1.6 + 0.4 * n_groups))))
    y, x = np.indices((n_groups, n_genes))
    ax.scatter(
        x.ravel(),
        y.ravel(),
        s=areas.ravel(),
        c=shown.ravel(),
        cmap=ramp,
        norm=norm,
        edgecolors=_INK,
        linewidths=0.2,
        gid="dots",
    )
    group_ids = cube.groups["group_id"].tolist()
    ax.set_yticks(range(n_groups), [group_ids[g] for g in rows])
    ax.set_xticks(range(n_genes), names, rotation=90)
    ax.set_xlim(-0.6, n_genes - 0.4)
    ax.set_ylim(n_groups - 0.4, -0.6)
    ax.tick_params(length=0)
    if gene_groups:
        _draw_gene_groups(ax, gene_groups)
    if scale is not None:
        label = f"Mean expression\nin group, scaled\nper {scale}"
    elif expressed_only:
        label = "Mean expression in\nexpressing cells"
    else:
        label = "Mean expression\nin group"
    mappable = _module("matplotlib.cm").ScalarMappable(norm=norm, cmap=ramp)
    # The bar starts below the top of the plot, to leave room for its label above
    # it, where scanpy writes it; a label along the bar ran into the key below it.
    bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.4, anchor=(0.0, 0.85))
    handles = [
        _key(
            f"{round(100 * share):d}",
            color=_GREY,
            markeredgecolor=_INK,
            markeredgewidth=0.2,
            markersize=math.sqrt(_DOT_AREA * (share / f_max) ** _DOT_EXPONENT),
        )
        for share in _fraction_keys(f_max)
    ]
    key = bar.ax.legend(
        handles=handles,
        title="Fraction of cells\nin group (%)",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.06),
        borderaxespad=0.0,
        frameon=False,
        alignment="left",
    )
    key.get_title().set_fontweight("bold")
    bar.ax.set_title(
        label,
        loc="left",
        fontsize=key.get_title().get_fontsize(),
        fontweight="bold",
    )
    return ax


def _dot_genes(
    cube: Cube, genes: Any
) -> tuple[list[int], list[str], list[tuple[str, int, int]]]:
    """The cube's features of the genes asked for, the names shown, and the groups of
    genes as (name, first column, last column)."""
    reject_masked(genes, "genes")
    if isinstance(genes, str):
        named: dict[str, Any] = {"": [genes]}
    elif isinstance(genes, Mapping):
        named = dict(genes)
    else:
        named = {"": genes}
    features = cube.features
    ids = features["feature_id"].tolist()
    index = {value: j for j, value in enumerate(ids)}
    symbols = features["symbol"].tolist() if "symbol" in features else []
    by_symbol: dict[Any, list[int]] = {}
    for j, symbol in enumerate(symbols):
        by_symbol.setdefault(symbol, []).append(j)
    columns: list[int] = []
    names: list[str] = []
    gene_groups: list[tuple[str, int, int]] = []
    for label, members in named.items():
        if not isinstance(label, str):
            raise TypeError(
                f"the names of groups of genes must be text, got {type(label).__name__}"
            )
        if isinstance(members, str):
            members = [members]
        if not isinstance(members, Sequence | np.ndarray):
            raise TypeError(
                "genes must be feature ids or symbols, or a mapping of names to them, "
                f"got {type(members).__name__}"
            )
        start = len(columns)
        for gene in list(members):
            if not isinstance(gene, str):
                raise TypeError(
                    f"genes must be feature ids or symbols (str), got "
                    f"{type(gene).__name__} {gene!r}"
                )
            if gene in index:
                columns.append(index[gene])
            elif gene in by_symbol:
                found = by_symbol[gene]
                if len(found) > 1:
                    raise ValueError(
                        f"the symbol {gene!r} names {len(found)} genes, "
                        f"{quoted([ids[j] for j in found])}; give their feature ids"
                    )
                columns.append(found[0])
            else:
                raise KeyError(
                    f"no gene {gene!r} among the cube's {len(ids):,} features, by "
                    "feature_id or symbol"
                )
            names.append(gene)
        if len(columns) == start:
            raise ValueError(
                "genes is empty; give at least one gene"
                if not label
                else f"the group of genes {label!r} is empty"
            )
        if label:
            gene_groups.append((label, start, len(columns) - 1))
    return columns, names, gene_groups


def _chosen_ids(value: Any, ids: list[str], name: str, kind: str) -> list[int]:
    """The positions of the ids chosen, in the order given; every id for None."""
    if value is None:
        return list(range(len(ids)))
    reject_masked(value, name)
    chosen = [value] if isinstance(value, str) else value
    if not isinstance(chosen, Sequence | np.ndarray):
        raise TypeError(
            f"{name} must be a list of {kind} ids, got {type(value).__name__}"
        )
    index = {identifier: i for i, identifier in enumerate(ids)}
    positions = []
    for identifier in list(chosen):
        if not isinstance(identifier, str):
            raise TypeError(
                f"{name} must hold {kind} ids (str), got {type(identifier).__name__} "
                f"{identifier!r}"
            )
        if identifier not in index:
            raise KeyError(f"no {kind} {identifier!r}; {kind}s are: {quoted(ids)}")
        positions.append(index[identifier])
    if not positions:
        raise ValueError(f"{name} is empty; give at least one {kind}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{name} names a {kind} more than once")
    return positions


def _pooled_samples(cube: Cube, samples: Any) -> list[int]:
    """The positions of the samples pooled: ids, a boolean mask, or every sample."""
    ids = cube.samples["sample_id"].tolist()
    if samples is not None and not isinstance(samples, str):
        reject_masked(samples, "samples")
        mask = np.asarray(samples)
        if mask.dtype == bool:
            if mask.shape != (len(ids),):
                raise ValueError(
                    f"a boolean mask of samples needs one entry per sample "
                    f"({len(ids)}), got shape {mask.shape}"
                )
            if not mask.any():
                raise ValueError("samples is empty; the mask selects no sample")
            return np.flatnonzero(mask).tolist()
    return _chosen_ids(samples, ids, "samples", "sample")


def _dot_values(
    cube: Cube,
    rows: list[int],
    columns: list[int],
    pooled: list[int],
    expressed_only: bool,
) -> tuple[FloatArray, FloatArray]:
    """The fraction of cells detected and the mean log value, groups x genes, each
    from the sums over the samples pooled."""
    n_cells = np.asarray(cube.n_cells, dtype=np.int64)
    fractions = np.empty((len(rows), len(columns)))
    means = np.empty((len(rows), len(columns)))
    group_ids = cube.groups["group_id"].tolist()
    for i, g in enumerate(rows):
        cells = int(n_cells[pooled, g].sum())
        if cells == 0:
            raise ValueError(
                f"group {group_ids[g]!r} has no cells in the samples pooled, so its "
                "fractions and means are undefined; leave it out with groups"
            )
        detected = np.asarray(cube.layer("detected", g))[pooled][:, columns]
        log_sum = np.asarray(cube.layer("log_sum", g), dtype=np.float64)[pooled]
        expressing = detected.sum(axis=0, dtype=np.int64).astype(np.float64)
        total = log_sum[:, columns].sum(axis=0)
        fractions[i] = expressing / cells
        if expressed_only:
            means[i] = np.divide(
                total, expressing, out=np.zeros_like(total), where=expressing > 0
            )
        else:
            means[i] = total / cells
    return fractions, means


def _min_max_scaled(values: FloatArray, axis: int) -> FloatArray:
    """(v - min) / (max - min) along an axis, 0 where every value is equal."""
    shifted = values - values.min(axis=axis, keepdims=True)
    spread = shifted.max(axis=axis, keepdims=True)
    scaled: FloatArray = np.divide(
        shifted, spread, out=np.zeros_like(shifted), where=spread > 0.0
    )
    return scaled


def _fraction_keys(f_max: float) -> list[float]:
    """The fractions of the key of a dot plot's areas, in scanpy's steps: 0.2 apart
    from f_max down, 0.1 when f_max is at most 0.6 and 0.05 when at most 0.3."""
    step = 0.2 if f_max > 0.6 else (0.1 if f_max > 0.3 else 0.05)
    return [float(value) for value in np.arange(f_max, 0.0, -step)[::-1]]


def _draw_gene_groups(ax: "Axes", gene_groups: list[tuple[str, int, int]]) -> None:
    """Brackets above the columns of each group of genes, labelled with its name;
    the names turned upright when any is longer than four characters."""
    transform = _text_transform(ax)
    rotation = 90 if max(len(name) for name, _, _ in gene_groups) > 4 else 0
    for name, start, end in gene_groups:
        left, right = start - 0.3, end + 0.3
        ax.plot(
            [left, left, right, right],
            [1.01, 1.04, 1.04, 1.01],
            transform=transform,
            color=_INK,
            linewidth=0.8,
            clip_on=False,
            gid="gene groups",
        )
        ax.text(
            (left + right) / 2.0,
            1.06,
            name,
            transform=transform,
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=9,
            gid="gene group names",
        )
