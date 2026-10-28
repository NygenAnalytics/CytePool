"""Per-sample scores of gene sets: GSVA, ssGSEA, combined z-scores and PLAGE, as the R
package GSVA computes them.

The functions compute the four methods of GSVA with its options and defaults:
:func:`~cytepool.gsva`, the method of Haenzelmann, Castelo and Guinney (2013);
:func:`~cytepool.ssgsea`, single sample GSEA (Barbie et al. 2009);
:func:`~cytepool.combined_z_scores` (Lee et al. 2008); and :func:`~cytepool.plage`
(Tomfohr, Lu and Kepler 2005). Each returns a :class:`~cytepool.PathwayScores`, one
score per sample and gene set, which :func:`~cytepool.plotting.plot_pathway_scores`
draws as a heatmap and :func:`~cytepool.limma` tests with a design.
https://docs.nygen.io/CytePool/statistics/pathway-scores#gsva explains the methods and
how they differ, and
https://docs.nygen.io/CytePool/statistics/pathway-scores#pathway-scores-limits how they
differ from GSVA's; each function's docstring gives those that change what it returns.

The data:
    Values are samples x genes: a 2-D array with one gene id per column, or a group
    (cell type) of a :class:`~cytepool.Cube`, of which the samples with at least
    ``min_cells`` cells and some counts are scored, over the genes that edgeR's
    ``filterByExpr`` keeps (:func:`~cytepool.normalization.filter_by_expression`),
    as log2 counts per million after TMM normalization with a prior count of 2
    (``values="log2_cpm"``, edgeR's ``cpm(log = TRUE)``) or as counts
    (``values="counts"``), the values GSVA's guide passes. NaN marks a missing value;
    infinite values are refused. Each method's docstring says how it treats missing
    values. :func:`~cytepool.gsva`, :func:`~cytepool.combined_z_scores` and
    :func:`~cytepool.plage` leave out the genes whose values are all equal
    (constant), before the sets are matched, and :func:`~cytepool.ssgsea` keeps them,
    as GSVA does.

Gene sets:
    The sets are a :class:`~cytepool.GeneSets` or a mapping of set name to gene ids,
    matched to the genes' ids exactly. A set is scored when ``min_size`` to ``max_size``
    of its genes are among the genes scored (GSVA's ``minSize`` and ``maxSize``, counted
    after matching); the others are left out, as GSVA leaves them out. The result keeps
    the order of the sets.

Matching GSVA:
    The scores match GSVA 2.0.0, which is licensed under the GPL (>= 2): the methods are
    implemented from the papers and GSVA's documentation, and running GSVA on chosen
    inputs settled what these leave open, among them the conventions below and those
    that each function's docstring gives.

    - GSVA evaluates the normal distribution function of its Gaussian kernel from a
      table of its values at multiples of 0.001: at |t| rounded down to such a
      multiple, and as one minus that value for negative t, which moves a kernel
      estimate by less than 4e-4. :func:`~cytepool.gsva` does the same, and sums each
      estimate over the samples in their order, so that genes are ranked as in GSVA.
    - In GSVA's method, genes with equal statistics take their ranks in reverse order
      of the genes, and equal deviations above and below zero give
      ``score_type="max_deviation"`` the one below.
    - In ssGSEA, genes with equal values weigh their average rank truncated to an
      integer, and the walk passes them in the order of the genes.

References:
    Haenzelmann S, Castelo R, Guinney J (2013). GSVA: gene set variation analysis for
    microarray and RNA-seq data. BMC Bioinformatics 14, 7.
    doi:10.1186/1471-2105-14-7

    Barbie DA, Tamayo P, Boehm JS, Kim SY, Moody SE, Dunn IF, Schinzel AC, Sandy P,
    Meylan E, Scholl C, Froehling S, Chan EM, Sos ML, Michel K, Mermel C, Silver SJ,
    Weir BA, Reiling JH, Sheng Q, Gupta PB, Wadlow RC, Le H, Hoersch S, Wittner BS,
    Ramaswamy S, Livingston DM, Sabatini DM, Meyerson M, Thomas RK, Lander ES,
    Mesirov JP, Root DE, Gilliland DG, Jacks T, Hahn WC (2009). Systematic RNA
    interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature
    462, 108-112. doi:10.1038/nature08460

    Lee E, Chuang HY, Kim JW, Ideker T, Lee D (2008). Inferring pathway activity
    toward precise disease classification. PLoS Computational Biology 4, e1000217.
    doi:10.1371/journal.pcbi.1000217

    Tomfohr J, Lu J, Kepler TB (2005). Pathway level analysis of gene expression
    using singular value decomposition. BMC Bioinformatics 6, 225.
    doi:10.1186/1471-2105-6-225

    Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, Paulovich
    A, Pomeroy SL, Golub TR, Lander ES, Mesirov JP (2005). Gene set enrichment
    analysis: a knowledge-based approach for interpreting genome-wide expression
    profiles. Proceedings of the National Academy of Sciences 102, 15545-15550.
    doi:10.1073/pnas.0506580102

    Robinson MD, Oshlack A (2010). A scaling normalization method for differential
    expression analysis of RNA-seq data. Genome Biology 11, R25.
    doi:10.1186/gb-2010-11-3-r25

    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import prange

from ._analysis_records import (
    keep_provenance,
    provenance_of,
    provenance_text,
    restore_provenance,
)
from ._columns import text_column
from ._expression_values import (
    MIN_CELLS,
    PROFILE_ID,
    ExpressionValues,
    expression_values,
    profile_table,
)
from ._jit import by_chunks, jit, thread_limit
from ._masked_arrays import read_only, reject_masked
from ._messages import quoted
from ._options import (
    checked_choice,
    checked_flag,
    non_negative_number,
    positive_integer,
)
from ._ranked_genes import gene_set_collection
from ._summation import observed_means
from .gene_sets import GeneSets
from .stats.distributions import norm_cdf
from .stats.special import gammaincc_scalar
from .table import Table

__all__ = ["PathwayScores", "combined_z_scores", "gsva", "plage", "ssgsea"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_KERNELS = ("auto", "gaussian", "poisson", "none")
_SCORE_TYPES = ("difference", "kuiper", "max_deviation")
_KERNEL_CODES = {"gaussian": 0, "poisson": 1, "none": 2}
# GSVA's table of the normal distribution function: its values at multiples of
# 0.001 (found by probing GSVA 2.0.0).
_TABLE_STEP = 0.001
_BANDWIDTH_DIVISOR = 4.0  # the bandwidth is a quarter of the standard deviation
_POISSON_OFFSET = 0.5  # the Poisson kernel's mean, x + 1/2 (Haenzelmann et al. 2013)
# PLAGE's sign: a sum of loadings below this share of their sizes is zero.
_ZERO_SUM = 1e-10


@dataclass(frozen=True, eq=False, repr=False)
class PathwayScores:
    """Scores of gene sets in every profile: every sample of an array, or of a
    cube's group.

    Rows of ``scores`` are profiles, in the order of ``profiles``, and columns are
    the sets scored, in the order of ``gene_sets``, so that :func:`~cytepool.limma`
    tests them with a design of the samples, matched by ``sample_id``. The arrays are
    read-only.

    Attributes:
        scores: ``(n_profiles, n_sets)`` float64 scores; NaN where a set has none in
            a profile.
        profiles: One row per profile: ``profile_id``, the sample's id, then
            ``sample_id`` and, for a cube's group, the other columns of the cube's
            sample table, as :class:`~cytepool.Activities` names its profiles.
        gene_sets: The sets scored, each with the genes that were used, in the order
            given, with the description and source of the collection.
        method: ``"gsva"``, ``"ssgsea"``, ``"combined_z_scores"`` or ``"plage"``.
        settings: The options of the method, the kernel that ``"auto"`` chose
            included, and for a cube how its values were taken.
    """

    scores: FloatArray
    profiles: Table
    gene_sets: GeneSets
    method: str
    settings: Mapping[str, Any]

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # The settings are a read-only view, which pickle cannot write: a dict
        # travels in their place, and the provenance kept outside the fields with it.
        return (
            _restored_scores,
            (
                self.scores,
                self.profiles,
                self.gene_sets,
                self.method,
                dict(self.settings),
                provenance_text(self),
            ),
        )

    @property
    def provenance(self) -> dict[str, Any] | None:
        """A JSON-serialisable record of what produced the scores of a cube's group,
        as a new dict: the method, the CytePool, numpy, numba and Python versions,
        the cube's fingerprint, the digests it is built from and its provenance
        (``"cube"``), the layer read and the settings; None for scores of values
        given as an array. The cube's record costs what it costs for
        :func:`~cytepool.limma_voom` ("Provenance")."""
        return provenance_of(self)

    @property
    def n_genes(self) -> IntArray:
        """The number of genes of each set that were used, in the order of
        ``gene_sets``."""
        sizes = [len(genes) for genes in self.gene_sets.values()]
        return read_only(np.array(sizes, dtype=np.int64))

    def table(self) -> Table:
        """Return the scores as a table with one row per set and profile.

        Returns:
            A table with ``gene_set``, ``n_genes``, ``profile_id`` and ``score``, the
            rows of each set together, in the order of ``gene_sets``, and its
            profiles in the order of ``profiles``.
        """
        n_profiles, n_sets = self.scores.shape
        names = list(self.gene_sets)
        ids = self.profiles[PROFILE_ID].tolist()
        return Table(
            {
                "gene_set": text_column([name for name in names for _ in ids]),
                "n_genes": np.repeat(self.n_genes, n_profiles),
                "profile_id": text_column(ids * n_sets),
                "score": self.scores.T.ravel().copy(),
            }
        )

    def __repr__(self) -> str:
        n_profiles, n_sets = self.scores.shape
        return (
            f"PathwayScores(method={self.method!r}, n_profiles={n_profiles}, "
            f"n_sets={n_sets})"
        )


def _restored_scores(
    scores: FloatArray,
    profiles: Table,
    gene_sets: GeneSets,
    method: str,
    settings: dict[str, Any],
    provenance: str | None,
) -> PathwayScores:
    """Pathway scores unpickled: read-only again, with their provenance."""
    restored = PathwayScores(
        read_only(np.asarray(scores)),
        profiles,
        gene_sets,
        method,
        MappingProxyType(settings),
    )
    return restore_provenance(restored, provenance)


def gsva(
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    id_column: str = "feature_id",
    feature_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    values: str = "log2_cpm",
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    kernel: str = "auto",
    min_samples_without_kernel: int = 200,
    exponent: float = 1.0,
    score_type: str = "difference",
    min_size: int = 1,
    max_size: int | None = None,
    n_threads: int | None = None,
) -> PathwayScores:
    """Score gene sets in every sample by gene set variation analysis (GSVA).

    The method of Haenzelmann, Castelo and Guinney (2013) as GSVA 2.0.0's
    ``gsva(gsvaParam(...))`` computes it:

    1. Each gene's values are turned into the probability of a value at most as high
       under the gene's distribution across samples, estimated by a kernel: for gene
       i in sample j, F_ij = (1/n) sum_k K(x_ij, x_ik) over the n samples, with K the
       Gaussian kernel Phi((x_ij - x_ik) / h_i), h_i a quarter of the gene's standard
       deviation, for continuous values; the Poisson kernel P(Y <= x_ij) for
       Y ~ Poisson(x_ik + 1/2), for counts; or, without a kernel, the indicator of
       x_ik <= x_ij, the empirical distribution function.
    2. The genes of each sample are ranked by the statistic z_ij = -log((1 - F_ij) /
       F_ij), from 1 for the lowest to p for the highest of the p genes, and each gene
       weighs |p/2 - r_ij|^tau, so that both ends of the ranking weigh most
       (``exponent``, GSVA's ``tau``).
    3. A walk down the genes from rank p rises at each gene of a set by its share of
       the set's weight and falls at each other gene by 1/(p - k), for k genes in the
       set, as in gene set enrichment analysis (Subramanian et al. 2005). With E+ and
       E- the sizes of the walk's largest deviations above and below zero, the score
       is E+ - E- (``score_type="difference"``, GSVA's ``maxDiff = TRUE``), which is
       large for a set whose genes are high and low for one whose genes are low;
       E+ + E- (``"kuiper"``, ``absRanking = TRUE``), Kuiper's statistic, large for a
       set whose genes are at either end; or the larger of the two deviations, with
       its sign (``"max_deviation"``, ``maxDiff = FALSE``).

    Scores lie in [-1, 1], or in [0, 2] with ``score_type="kuiper"``. A set with every
    gene, or whose genes all weigh zero (one gene at rank p/2), has a NaN score, as in
    GSVA. With missing values, each gene's distribution is estimated from its observed
    samples and each sample ranks its observed genes, as GSVA's ``use = "na.rm"``
    does; the walk then falls by 1/(p_j - k_j) at each of the sample's observed genes
    outside the set, with p_j and k_j counting the genes observed in sample j, so that
    it ends at zero; GSVA's falls by 1/(p - k_j), counting the missing genes too, and
    ends above zero. A sample without any value has NaN scores, where GSVA stops with
    an error.
    https://docs.nygen.io/CytePool/statistics/pathway-scores#gsva explains the method,
    and the module documentation the conventions it shares with GSVA.

    Args:
        data: ``(n_samples, n_genes)`` expression values, such as log-CPM or, for the
            Poisson kernel, counts; NaN marks a missing value. Or a
            :class:`~cytepool.Cube`, whose ``group`` is scored.
        gene_sets: A :class:`~cytepool.GeneSets`, such as one read by
            :func:`~cytepool.read_gmt`, or a mapping of set name to gene ids.
        id_column: For a cube, the column of its features table with the genes'
            ids, which the sets name: ``"feature_id"`` by default, or ``"symbol"``.
        feature_ids: For an array, the genes' ids, one per column (None, NaN or an
            empty string for a gene without one).
        sample_ids: For an array, one id per row; by default ``"1"``, ``"2"``, ...
        group: The group (cell type) of a cube to score; it may be left out when the
            cube has one.
        values: For a cube, the values scored: ``"log2_cpm"``, log2 counts per million
            after TMM normalization, or ``"counts"``.
        min_cells: For a cube, the fewest cells a sample needs in the group to be
            scored, at least 0, as for
            :func:`~cytepool.limma_voom`.
        expression_filter: For a cube, whether to score only the genes that
            :func:`~cytepool.normalization.filter_by_expression` keeps, with the
            samples as one group; or a mapping of its settings, which may name a
            column of the cube's samples whose values group them, such as
            ``{"sample_groups": "condition"}``, as
            :func:`~cytepool.limma_voom` takes it.
        kernel: How each gene's distribution is estimated (GSVA's ``kcdf``):
            ``"gaussian"`` for continuous values, ``"poisson"`` for counts (whole
            numbers of at least 0), ``"none"`` for the empirical distribution
            function, or ``"auto"``: the Poisson kernel for an integer dtype, as
            GSVA chooses it for R's integer matrices, the Gaussian kernel otherwise,
            and none from ``min_samples_without_kernel`` samples on, also for an
            integer dtype, as GSVA documents (GSVA 2.0.0 stops with an error there).
        min_samples_without_kernel: The number of samples from which ``"auto"``
            uses no kernel, at least 1 (GSVA's ``kcdfNoneMinSampleSize``).
        exponent: The power tau of the weights |p/2 - r|^tau of the ranks, at least
            0 (GSVA's ``tau``).
        score_type: ``"difference"``, ``"kuiper"`` or ``"max_deviation"``, as
            above (GSVA's ``maxDiff`` and ``absRanking``).
        min_size: The fewest of a set's genes among the genes scored for the set to
            be scored, at least 1 (GSVA's ``minSize``).
        max_size: The most of them, at least ``min_size``; None for no limit (GSVA's
            ``maxSize``).
        n_threads: The most threads to use; None uses numba's current setting. The
            result does not depend on it.

    Returns:
        The scores, samples x sets, of the sets with ``min_size`` to ``max_size``
        genes, in the order of ``gene_sets``.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range,
            the Poisson kernel is given values that are not counts, no gene varies
            across samples, no gene of any set is among the genes, or no set has
            ``min_size`` to ``max_size`` genes.
        KeyError: If the cube has no such group or features column.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> y = rng.normal(size=(6, 40))
        >>> y[3:, :5] += 2.0  # the first five genes are higher in the last samples
        >>> genes = [f"g{j}" for j in range(40)]
        >>> sets = {"up": genes[:5], "other": genes[20:30]}
        >>> result = gsva(y, sets, feature_ids=genes)
        >>> result.scores[:, 0].round(2).tolist()
        [-0.54, -0.82, -0.69, 0.78, 0.34, 0.77]
    """
    checked = _checked_size(min_size, max_size)
    kind = checked_choice(kernel, "kernel", _KERNELS)
    without_kernel = positive_integer(
        min_samples_without_kernel, "min_samples_without_kernel"
    )
    power = non_negative_number(exponent, "exponent")
    score = _SCORE_TYPES.index(checked_choice(score_type, "score_type", _SCORE_TYPES))
    collection = _collection(gene_sets)
    expression = expression_values(
        data,
        id_column,
        feature_ids,
        sample_ids,
        group,
        values,
        min_cells,
        expression_filter,
    )
    n_samples = expression.values.shape[0]
    # GSVA's kcdf = "auto": Poisson for R's integer matrices, Gaussian otherwise, and
    # no kernel from kcdfNoneMinSampleSize samples on (found by probing GSVA 2.0.0,
    # which stops with an error on integer data from that many samples).
    if kind == "auto":
        kind = (
            "none"
            if n_samples >= without_kernel
            else "poisson"
            if expression.integer
            else "gaussian"
        )
    if kind == "poisson":
        _check_counts(expression.values)
    kept = _varying_genes(expression.values)
    members, names = _matched_sets(collection, expression.gene_ids, kept, checked)
    settings = {
        "kernel": kind,
        "min_samples_without_kernel": without_kernel,
        "exponent": power,
        "score_type": _SCORE_TYPES[score],
        "min_size": checked[0],
        "max_size": checked[1],
        **expression.settings,
    }
    columns = expression.values[:, kept]
    with thread_limit(n_threads):
        statistics = _kernel_statistics(
            np.ascontiguousarray(columns.T), _KERNEL_CODES[kind]
        )
        ranks = _gene_ranks(statistics)
        scores = _gsva_scores(ranks, members, power, score)
    return _result(
        scores, expression, collection, names, members, kept, "gsva", settings
    )


def ssgsea(
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    id_column: str = "feature_id",
    feature_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    values: str = "log2_cpm",
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    exponent: float = 0.25,
    normalize: bool = True,
    min_size: int = 1,
    max_size: int | None = None,
    n_threads: int | None = None,
) -> PathwayScores:
    """Score gene sets in every sample by single-sample GSEA (ssGSEA).

    The method of Barbie et al. (2009) as GSVA 2.0.0's ``gsva(ssgseaParam(...))``
    computes it: each sample ranks its genes by their values, and the walk down the
    ranking rises at each gene of a set by its rank to the power alpha
    (``exponent``), as a share of the set's total, and falls at each other gene by
    1/(p - k), for k of the p genes in the set. The score is the sum of the walk over
    all p genes, not its extreme:

        ES = sum_{g in G} w_g a_g / sum_{g in G} w_g - sum_{g not in G} a_g / (p - k),

    with a_g the gene's rank from the bottom of the walk and w_g = a_g^alpha. Only the
    order of the values within each sample matters. ``normalize=True`` divides every
    score by the range of all the scores, over sets and samples, as Barbie et al. did
    and GSVA does by default. Missing values are left out of each sample's ranking,
    as GSVA's ``use = "na.rm"`` does. A set of every gene has a NaN score, which the
    normalization leaves out, where GSVA stops with an error.
    https://docs.nygen.io/CytePool/statistics/pathway-scores#ssgsea explains the
    method, and the module documentation its ties.

    Args:
        data: ``(n_samples, n_genes)`` expression values; NaN marks a missing value,
            which is left out of its sample's ranking. Or a
            :class:`~cytepool.Cube`, whose ``group`` is scored.
        gene_sets: A :class:`~cytepool.GeneSets` or a mapping of set name to
            gene ids.
        id_column: For a cube, the column of its gene ids, as for :func:`gsva`.
        feature_ids: For an array, the genes' ids, as for :func:`gsva`.
        sample_ids: For an array, one id per row, as for :func:`gsva`.
        group: The group of a cube, as for :func:`gsva`.
        values: For a cube, ``"log2_cpm"`` or ``"counts"``, as for :func:`gsva`; both
            rank the genes of a sample alike.
        min_cells: For a cube, as for :func:`gsva`.
        expression_filter: For a cube, as for :func:`gsva`.
        exponent: The power alpha of the weights of the ranks, at least 0 (GSVA's
            ``alpha``).
        normalize: Whether to divide every score by the range of all the scores
            (GSVA's ``normalize``).
        min_size: The fewest of a set's genes among the genes for the set to be
            scored, at least 1.
        max_size: The most of them, at least ``min_size``; None for no limit.
        n_threads: The most threads to use; None uses numba's current setting. The
            result does not depend on it.

    Returns:
        The scores, samples x sets, of the sets with ``min_size`` to ``max_size``
        genes, in the order of ``gene_sets``.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range, no
            gene of any set is among the genes, no set has ``min_size`` to
            ``max_size`` genes, or ``normalize`` would divide by a range of zero.
        KeyError: If the cube has no such group or features column.
    """
    checked = _checked_size(min_size, max_size)
    power = non_negative_number(exponent, "exponent")
    scaled = checked_flag(normalize, "normalize")
    collection = _collection(gene_sets)
    expression = expression_values(
        data,
        id_column,
        feature_ids,
        sample_ids,
        group,
        values,
        min_cells,
        expression_filter,
    )
    kept = np.arange(expression.values.shape[1])
    members, names = _matched_sets(collection, expression.gene_ids, kept, checked)
    settings = {
        "exponent": power,
        "normalize": scaled,
        "min_size": checked[0],
        "max_size": checked[1],
        **expression.settings,
    }
    with thread_limit(n_threads):
        scores = _ssgsea_scores(expression.values, members, power)
    if scaled:
        finite = scores[np.isfinite(scores)]
        spread = float(finite.max() - finite.min()) if finite.size else 0.0
        if not spread > 0.0:
            raise ValueError(
                "normalize divides the scores by their range, which is zero here "
                f"({finite.size} finite score(s)); score more sets or samples, or "
                "pass normalize=False"
            )
        scores = scores / spread
    return _result(
        scores, expression, collection, names, members, kept, "ssgsea", settings
    )


def combined_z_scores(
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    id_column: str = "feature_id",
    feature_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    values: str = "log2_cpm",
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    min_size: int = 1,
    max_size: int | None = None,
) -> PathwayScores:
    """Score gene sets in every sample by their combined z-scores.

    The method of Lee et al. (2008) as GSVA 2.0.0's ``gsva(zscoreParam(...))``
    computes it: each gene is standardized across samples, and a set's score in a
    sample is the sum of its genes' z-scores divided by the square root of their
    number, which is a standard normal score when the genes are independent standard
    normal values. A missing value leaves its gene out of the mean and standard
    deviation, and makes the score of every set that holds the gene NaN in that
    sample, as in GSVA.
    https://docs.nygen.io/CytePool/statistics/pathway-scores#combined-z-scores
    explains the method.

    Args:
        data: ``(n_samples, n_genes)`` expression values on a log scale; NaN marks a
            missing value. Or a :class:`~cytepool.Cube`, whose ``group`` is
            scored.
        gene_sets: A :class:`~cytepool.GeneSets` or a mapping of set name to
            gene ids.
        id_column: For a cube, the column of its gene ids, as for :func:`gsva`.
        feature_ids: For an array, the genes' ids, as for :func:`gsva`.
        sample_ids: For an array, one id per row, as for :func:`gsva`.
        group: The group of a cube, as for :func:`gsva`.
        values: For a cube, ``"log2_cpm"`` or ``"counts"``, as for :func:`gsva`.
        min_cells: For a cube, as for :func:`gsva`.
        expression_filter: For a cube, as for :func:`gsva`.
        min_size: The fewest of a set's genes among the genes for the set to be
            scored, at least 1.
        max_size: The most of them, at least ``min_size``; None for no limit.

    Returns:
        The scores, samples x sets; NaN where a gene of the set is missing in the
        sample.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D or hold an infinite value, the ids do
            not match the values, an id names two genes, an option is out of range, no
            gene varies across samples, no gene of any set is among the genes, or no
            set has ``min_size`` to ``max_size`` genes.
        KeyError: If the cube has no such group or features column.
    """
    checked = _checked_size(min_size, max_size)
    collection = _collection(gene_sets)
    expression = expression_values(
        data,
        id_column,
        feature_ids,
        sample_ids,
        group,
        values,
        min_cells,
        expression_filter,
    )
    kept = _varying_genes(expression.values)
    members, names = _matched_sets(collection, expression.gene_ids, kept, checked)
    z = _standardized(expression.values[:, kept])
    scores = np.empty((z.shape[0], len(names)))
    for s in range(len(names)):
        genes = members[s]
        scores[:, s] = z[:, genes].sum(axis=1) / math.sqrt(genes.size)
    settings = {"min_size": checked[0], "max_size": checked[1], **expression.settings}
    return _result(
        scores,
        expression,
        collection,
        names,
        members,
        kept,
        "combined_z_scores",
        settings,
    )


def plage(
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    id_column: str = "feature_id",
    feature_ids: Any = None,
    sample_ids: Any = None,
    group: str | None = None,
    values: str = "log2_cpm",
    min_cells: int = MIN_CELLS,
    expression_filter: bool | Mapping[str, Any] = True,
    min_size: int = 1,
    max_size: int | None = None,
) -> PathwayScores:
    """Score gene sets in every sample by pathway level analysis of gene expression
    (PLAGE).

    The method of Tomfohr, Lu and Kepler (2005) as GSVA 2.0.0's
    ``gsva(plageParam(...))`` computes it: the standardized values of a set's genes
    form a genes x samples matrix, and the set's scores are its first right singular
    vector, the samples' coordinates on the axis along which the genes vary most
    together. The scores of a set have mean zero and length one across samples, and
    rise with the mean standardized expression of its genes: the sum of the set's
    loadings (the first left singular vector) is positive, or, when it is zero, the
    loading of largest size, the first of equal ones. GSVA's sign is that of its
    singular value decomposition, so the two agree up to sign. A set of one gene
    scores the gene's standardized values divided by their length, where GSVA gives
    1 in every sample. Missing values are refused, as GSVA refuses them.
    https://docs.nygen.io/CytePool/statistics/pathway-scores#plage explains the
    method.

    Args:
        data: ``(n_samples, n_genes)`` expression values on a log scale, without
            missing values. Or a :class:`~cytepool.Cube`, whose ``group`` is
            scored.
        gene_sets: A :class:`~cytepool.GeneSets` or a mapping of set name to
            gene ids.
        id_column: For a cube, the column of its gene ids, as for :func:`gsva`.
        feature_ids: For an array, the genes' ids, as for :func:`gsva`.
        sample_ids: For an array, one id per row, as for :func:`gsva`.
        group: The group of a cube, as for :func:`gsva`.
        values: For a cube, ``"log2_cpm"`` or ``"counts"``, as for :func:`gsva`.
        min_cells: For a cube, as for :func:`gsva`.
        expression_filter: For a cube, as for :func:`gsva`.
        min_size: The fewest of a set's genes among the genes for the set to be
            scored, at least 1.
        max_size: The most of them, at least ``min_size``; None for no limit.

    Returns:
        The scores, samples x sets.

    Raises:
        TypeError: If an argument has the wrong type, values are not real numbers,
            gene ids are not strings, or an argument is or holds a numpy masked array.
        ValueError: If the values are not 2-D, hold an infinite or a missing value,
            the ids do not match the values, an id names two genes, an option is out
            of range, no gene varies across samples, no gene of any set is among the
            genes, or no set has ``min_size`` to ``max_size`` genes.
        KeyError: If the cube has no such group or features column.
    """
    checked = _checked_size(min_size, max_size)
    collection = _collection(gene_sets)
    expression = expression_values(
        data,
        id_column,
        feature_ids,
        sample_ids,
        group,
        values,
        min_cells,
        expression_filter,
        missing=None,
    )
    missing = np.isnan(expression.values)
    if missing.any():
        row, column = np.argwhere(missing)[0]
        raise ValueError(
            f"{int(missing.sum())} value(s) are missing (NaN), such as that of gene "
            f"{expression.gene_ids[column]!r} in sample "
            f"{expression.samples['sample_id'][row]!r}; PLAGE's singular value "
            "decomposition needs every value: leave out the genes or samples with "
            "missing values first"
        )
    kept = _varying_genes(expression.values)
    members, names = _matched_sets(collection, expression.gene_ids, kept, checked)
    z = _standardized(expression.values[:, kept])
    scores = np.empty((z.shape[0], len(names)))
    for s in range(len(names)):
        scores[:, s] = _first_singular_vector(z[:, members[s]].T)
    settings = {"min_size": checked[0], "max_size": checked[1], **expression.settings}
    return _result(
        scores, expression, collection, names, members, kept, "plage", settings
    )


# The data.


def _check_counts(values: FloatArray) -> None:
    # GSVA's Poisson kernel stops with an error on negative values and scores
    # fractional ones (found by probing GSVA 2.0.0); here both are refused.
    observed = values[~np.isnan(values)]
    if observed.size and (
        float(observed.min()) < 0.0 or not np.all(observed == np.floor(observed))
    ):
        raise ValueError(
            "the Poisson kernel takes counts, non-negative whole numbers; use "
            "kernel='gaussian' for continuous values such as log-CPM"
        )


def _varying_genes(values: FloatArray) -> IntArray:
    """The genes whose observed values are not all equal, as GSVA keeps them for its
    method, combined z-scores and PLAGE; for ssGSEA it keeps every gene (found by
    probing GSVA 2.0.0)."""
    missing = np.isnan(values)
    highest = np.where(missing, -np.inf, values).max(axis=0)
    lowest = np.where(missing, np.inf, values).min(axis=0)
    kept = np.flatnonzero(highest > lowest)
    if kept.size == 0:
        raise ValueError(
            "every gene has equal values in all samples, so none can be scored; "
            "the methods compare each gene's values across samples"
        )
    return kept.astype(np.int64)


def _checked_size(min_size: Any, max_size: Any) -> tuple[int, int | None]:
    reject_masked(min_size, "min_size")
    reject_masked(max_size, "max_size")
    smallest = positive_integer(min_size, "min_size")
    if max_size is None:
        return smallest, None
    largest = positive_integer(max_size, "max_size")
    if largest < smallest:
        raise ValueError(
            f"max_size ({largest}) is smaller than min_size ({smallest}), so no set "
            "could be scored"
        )
    return smallest, largest


def _collection(gene_sets: Any) -> GeneSets:
    reject_masked(gene_sets, "gene_sets")
    return gene_set_collection(gene_sets)


def _matched_sets(
    collection: GeneSets,
    gene_ids: list[str | None],
    kept: IntArray,
    sizes: tuple[int, int | None],
) -> tuple[list[IntArray], list[str]]:
    """The positions among the kept genes of each scored set's genes, in the set's
    order, and the names of the sets scored."""
    index = {
        gene_ids[j]: position
        for position, j in enumerate(kept.tolist())
        if gene_ids[j] is not None
    }
    smallest, largest = sizes
    members: list[IntArray] = []
    names: list[str] = []
    matched = 0
    for name, genes in collection.items():
        positions = np.array(
            [index[gene] for gene in genes if gene in index], dtype=np.int64
        )
        matched += positions.size
        if (
            positions.size
            and smallest <= positions.size
            and (largest is None or positions.size <= largest)
        ):
            members.append(positions)
            names.append(name)
    if matched == 0:
        examples = [gene for genes in collection.values() for gene in genes[:3]][:3]
        ids = [gene for gene in gene_ids if gene is not None][:3]
        raise ValueError(
            "no gene of any set is among the genes scored; the sets and the genes' "
            "ids must name genes the same way (for example both by symbol, or both by "
            f"Ensembl id): the sets name genes such as {quoted(examples)}, the data "
            f"genes such as {quoted(ids)}"
        )
    if not names:
        limit = "no limit" if largest is None else str(largest)
        raise ValueError(
            f"no gene set has between min_size ({smallest}) and max_size ({limit}) "
            "genes among the genes scored"
        )
    return members, names


def _result(
    scores: FloatArray,
    expression: ExpressionValues,
    collection: GeneSets,
    names: list[str],
    members: list[IntArray],
    kept: IntArray,
    method: str,
    settings: dict[str, Any],
) -> PathwayScores:
    genes = [expression.gene_ids[j] for j in kept.tolist()]
    used = GeneSets(
        {
            name: [genes[i] for i in positions.tolist()]
            for name, positions in zip(names, members, strict=True)
        },
        descriptions={name: collection.descriptions[name] for name in names},
        name=collection.name,
        version=collection.version,
        licence=collection.licence,
        file=collection.file,
        sha256=collection.sha256,
    )
    result = PathwayScores(
        read_only(np.ascontiguousarray(scores, dtype=np.float64)),
        profile_table(expression.samples, "the scores"),
        used,
        method,
        MappingProxyType(dict(settings)),
    )
    if expression.cube is not None:
        keep_provenance(result, method, expression.cube, settings)
    return result


# GSVA: the kernel estimates, their ranks and the walks.


@cache
def _normal_table() -> FloatArray:
    """GSVA's table of the normal distribution function at multiples of 0.001, up to
    the first value that rounds to 1."""
    steps = np.arange(10_000, dtype=np.float64)
    table = np.asarray(norm_cdf(steps * _TABLE_STEP), dtype=np.float64)
    last = int(np.argmax(table == 1.0))
    return read_only(np.ascontiguousarray(table[: last + 1]))


def _kernel_statistics(values: FloatArray, kernel: int) -> FloatArray:
    """The statistics z = -log((1 - F) / F) of the kernel estimates F, samples x
    genes, from the genes x samples ``values``; NaN where a value is missing."""
    n_genes, n_samples = values.shape
    out = np.empty((n_samples, n_genes))
    table = _normal_table()
    by_chunks(
        _kernel_statistics_parallel,
        _kernel_statistics_serial,
        (values, kernel, table, out),
        n_genes,
    )
    return out


@jit(parallel=True)
def _kernel_statistics_parallel(
    values: FloatArray, kernel: int, table: FloatArray, out: FloatArray, n_runs: int
) -> None:
    n_genes = values.shape[0]
    for run in prange(n_runs):
        for i in range(run * n_genes // n_runs, (run + 1) * n_genes // n_runs):
            _gene_statistics(values[i], kernel, table, out, i)


@jit
def _kernel_statistics_serial(
    values: FloatArray, kernel: int, table: FloatArray, out: FloatArray
) -> None:
    for i in range(values.shape[0]):
        _gene_statistics(values[i], kernel, table, out, i)


@jit(error_model="numpy")
def _gene_statistics(
    x: FloatArray, kernel: int, table: FloatArray, out: FloatArray, gene: int
) -> None:
    """The statistics of one gene in every sample, written to ``out[:, gene]``.

    The kernel estimate of sample j is the mean over the observed samples k, summed
    in their order, of the kernel: ``kernel`` 0 for the Gaussian kernel, which looks
    up the normal distribution function in ``table`` at |t| rounded down to a
    multiple of 0.001, and takes one minus it for negative t, as GSVA does (found by
    probing GSVA 2.0.0), 1 for the Poisson kernel, and 2 for none.
    """
    n = x.size
    observed = np.empty(n, dtype=np.int64)
    m = 0
    total = 0.0
    for k in range(n):
        if not math.isnan(x[k]):
            observed[m] = k
            m += 1
            total += x[k]
    bandwidth = 0.0
    if kernel == 0:
        mean = total / m
        squares = 0.0
        for q in range(m):
            deviation = x[observed[q]] - mean
            squares += deviation * deviation
        bandwidth = math.sqrt(squares / (m - 1)) / _BANDWIDTH_DIVISOR
    last = table.size - 1
    for j in range(n):
        if math.isnan(x[j]):
            out[j, gene] = np.nan
            continue
        s = 0.0
        for q in range(m):
            k = observed[q]
            if kernel == 0:
                t = (x[j] - x[k]) / bandwidth
                step = abs(t) / _TABLE_STEP
                phi = table[int(step)] if step < last else 1.0
                s += phi if t >= 0.0 else 1.0 - phi
            elif kernel == 1:
                mean_k = x[k] + _POISSON_OFFSET
                if x[j] == 0.0:
                    s += math.exp(-mean_k)
                else:
                    s += gammaincc_scalar(x[j] + 1.0, mean_k)
            elif x[k] <= x[j]:
                s += 1.0
        f = s / m
        out[j, gene] = -math.log((1.0 - f) / f)


def _gene_ranks(statistics: FloatArray) -> IntArray:
    """Ranks of the genes of each sample by their statistics, 1 for the lowest, and 0
    where missing; equal statistics take their ranks in reverse order of the genes, as
    in GSVA (found by probing GSVA 2.0.0)."""
    ranks = np.zeros(statistics.shape, dtype=np.int64)
    by_chunks(
        _gene_ranks_parallel,
        _gene_ranks_serial,
        (statistics, ranks),
        statistics.shape[0],
    )
    return ranks


@jit(parallel=True)
def _gene_ranks_parallel(statistics: FloatArray, ranks: IntArray, n_runs: int) -> None:
    n_samples = statistics.shape[0]
    for run in prange(n_runs):
        for j in range(run * n_samples // n_runs, (run + 1) * n_samples // n_runs):
            _sample_ranks(statistics[j], ranks[j])


@jit
def _gene_ranks_serial(statistics: FloatArray, ranks: IntArray) -> None:
    for j in range(statistics.shape[0]):
        _sample_ranks(statistics[j], ranks[j])


@jit
def _sample_ranks(values: FloatArray, ranks: IntArray) -> int:
    """Ranks 1 to p of the observed values, the later of equal values lower (R's
    ``rank(ties.method = "last")``); returns p."""
    n = values.size
    genes = np.empty(n, dtype=np.int64)
    p = 0
    for i in range(n - 1, -1, -1):  # the later genes first: they rank lower in ties
        if not math.isnan(values[i]):
            genes[p] = i
            p += 1
    genes = genes[:p]
    order = np.argsort(values[genes], kind="mergesort")
    for position in range(p):
        ranks[genes[order[position]]] = position + 1
    return p


def _set_arrays(members: list[IntArray]) -> tuple[IntArray, IntArray]:
    """The sets' genes, one after another, and where each set starts."""
    starts = np.zeros(len(members) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([genes.size for genes in members])
    flat = np.concatenate(members) if members else np.zeros(0, dtype=np.int64)
    return np.ascontiguousarray(flat, dtype=np.int64), starts


def _gsva_scores(
    ranks: IntArray, members: list[IntArray], exponent: float, score_type: int
) -> FloatArray:
    flat, starts = _set_arrays(members)
    out = np.empty((ranks.shape[0], len(members)))
    by_chunks(
        _gsva_scores_parallel,
        _gsva_scores_serial,
        (ranks, flat, starts, exponent, score_type, out),
        ranks.shape[0],
    )
    return out


@jit(parallel=True)
def _gsva_scores_parallel(
    ranks: IntArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    score_type: int,
    out: FloatArray,
    n_runs: int,
) -> None:
    n_samples = ranks.shape[0]
    for run in prange(n_runs):
        for j in range(run * n_samples // n_runs, (run + 1) * n_samples // n_runs):
            _gsva_sample(ranks[j], flat, starts, exponent, score_type, out[j])


@jit
def _gsva_scores_serial(
    ranks: IntArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    score_type: int,
    out: FloatArray,
) -> None:
    for j in range(ranks.shape[0]):
        _gsva_sample(ranks[j], flat, starts, exponent, score_type, out[j])


@jit(error_model="numpy")
def _gsva_sample(
    ranks: IntArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    score_type: int,
    out: FloatArray,
) -> None:
    """The scores of every set in one sample, from the ranks of its genes (0 where
    missing)."""
    p = 0
    for i in range(ranks.size):
        if ranks[i] > 0:
            p += 1
    half = p / 2.0
    largest = 0
    for s in range(starts.size - 1):
        largest = max(largest, starts[s + 1] - starts[s])
    positions = np.empty(largest, dtype=np.int64)
    weights = np.empty(largest)
    for s in range(starts.size - 1):
        k = 0
        for g in flat[starts[s] : starts[s + 1]]:
            r = ranks[g]
            if r > 0:
                positions[k] = p - r + 1  # from the top of the ranking
                weights[k] = abs(half - r) ** exponent
                k += 1
        if k == 0 or k == p:
            out[s] = np.nan
            continue
        order = np.argsort(positions[:k])
        total = 0.0
        for i in range(k):
            total += weights[order[i]]
        if total == 0.0:
            out[s] = np.nan
            continue
        # p and k count the sample's observed genes, so the walk ends at zero. GSVA's
        # use = "na.rm" counts the missing genes in p too, and its default use =
        # "everything" gives scores without meaning (found by probing GSVA 2.0.0).
        others = float(p - k)
        reached = 0.0
        highest = -np.inf
        lowest = np.inf
        for i in range(k):
            missed = (positions[order[i]] - (i + 1)) / others
            lowest = min(lowest, reached / total - missed)
            reached += weights[order[i]]
            highest = max(highest, reached / total - missed)
        if score_type == 0:
            out[s] = highest + lowest
        elif score_type == 1:
            out[s] = highest - lowest
        else:  # equal deviations give the one below, as in GSVA (found by probing)
            out[s] = highest if highest > -lowest else lowest


# ssGSEA.


def _ssgsea_scores(
    values: FloatArray, members: list[IntArray], exponent: float
) -> FloatArray:
    flat, starts = _set_arrays(members)
    out = np.empty((values.shape[0], len(members)))
    by_chunks(
        _ssgsea_scores_parallel,
        _ssgsea_scores_serial,
        (values, flat, starts, exponent, out),
        values.shape[0],
    )
    return out


@jit(parallel=True)
def _ssgsea_scores_parallel(
    values: FloatArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    out: FloatArray,
    n_runs: int,
) -> None:
    n_samples = values.shape[0]
    for run in prange(n_runs):
        for j in range(run * n_samples // n_runs, (run + 1) * n_samples // n_runs):
            _ssgsea_sample(values[j], flat, starts, exponent, out[j])


@jit
def _ssgsea_scores_serial(
    values: FloatArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    out: FloatArray,
) -> None:
    for j in range(values.shape[0]):
        _ssgsea_sample(values[j], flat, starts, exponent, out[j])


@jit(error_model="numpy")
def _ssgsea_sample(
    values: FloatArray,
    flat: IntArray,
    starts: IntArray,
    exponent: float,
    out: FloatArray,
) -> None:
    """The ssGSEA scores of every set in one sample.

    Each observed gene has its rank from the bottom, the later of equal values lower,
    and a weight, its average rank truncated to an integer to the power
    ``exponent`` (found by probing GSVA 2.0.0).
    """
    n = values.size
    ranks = np.zeros(n, dtype=np.int64)
    p = _sample_ranks(values, ranks)
    order = np.empty(p, dtype=np.int64)
    for i in range(n):
        if ranks[i] > 0:
            order[ranks[i] - 1] = i
    weights = np.zeros(n)
    start = 0
    while start < p:
        end = start
        while end + 1 < p and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end + 2) / 2.0  # the average of ranks start+1 to end+1
        weight = math.floor(average) ** exponent
        for position in range(start, end + 1):
            weights[order[position]] = weight
        start = end + 1
    total_rank = p * (p + 1) / 2.0
    for s in range(starts.size - 1):
        k = 0
        weighted = 0.0
        weight_sum = 0.0
        rank_sum = 0.0
        for g in flat[starts[s] : starts[s + 1]]:
            r = ranks[g]
            if r > 0:
                k += 1
                weighted += weights[g] * r
                weight_sum += weights[g]
                rank_sum += r
        if k == 0 or k == p:
            out[s] = np.nan
            continue
        out[s] = weighted / weight_sum - (total_rank - rank_sum) / (p - k)


# Combined z-scores and PLAGE.


def _standardized(values: FloatArray) -> FloatArray:
    """Each gene's values minus their mean, over their standard deviation, both of
    the observed values."""
    means = observed_means(values)
    centred = values - means
    observed = np.sum(~np.isnan(values), axis=0)
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.nansum(centred * centred, axis=0) / (observed - 1))
    z: FloatArray = centred / sd
    return z


def _first_singular_vector(matrix: FloatArray) -> FloatArray:
    """The first right singular vector of a genes x samples matrix, with the sign that
    makes the genes' loadings sum to a positive number, or else the largest loading,
    the first of equal ones, positive. GSVA returns the sign its singular value
    decomposition gives (found by probing GSVA 2.0.0)."""
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    loadings = u[:, 0]
    total = float(loadings.sum())
    sizes = np.abs(loadings)
    if abs(total) > _ZERO_SUM * float(sizes.sum()):
        sign = math.copysign(1.0, total)
    else:
        first = int(np.flatnonzero(sizes >= sizes.max() * (1.0 - _ZERO_SUM))[0])
        sign = math.copysign(1.0, float(loadings[first]))
    vector: FloatArray = sign * vt[0]
    return vector
