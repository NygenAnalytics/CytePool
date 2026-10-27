"""SumRank: genes whose differential expression ranks high in study after study.

Most genes that one single-cell study finds differentially expressed are not found by
the next (Nakatsuka et al. 2025). SumRank asks instead whether a gene ranks high in
every study: each study ranks its genes by their signed evidence of change, and a
gene's ranks are summed over the studies, so that a gene ranked high everywhere has a
small sum and a gene found by one study alone does not, however small its p-value
there. :func:`sumrank` computes the statistic and its p-values from a signed statistic
of every gene in every study; :func:`differential_expression_sumrank` does it for the
results of the studies of a :class:`~cytepool.cube.CubeCollection`, one group (cell
type) and contrast at a time, with p-values from permutations that re-run each study's
analysis on its samples with their labels permuted.

The statistic, as the paper defines it and its authors' code (Apache-2.0) computes it:
    1. Each study ranks the genes by their signed statistic, from 1 for the largest
       (the gene most up); tied genes share the average of their ranks, as R's
       ``rank``, which the authors' code calls, gives them. Results of
       differential expression give the signed -log10 p-value: -log10(p_value),
       negative where the log2 fold change is negative, as the paper ranks DESeq2's
       results.
    2. A gene of rank r among G genes has the normalized rank u = (r - 1) / (G - 1):
       0 for the gene most up and 1 for the gene most down.
    3. For each gene, the m smallest of its normalized ranks in the k studies are
       summed: those of the studies that rank it best, which leaves out studies that
       rank it low for reasons of their own. m = floor(p k), at least 1, for the
       proportion p of the studies (``proportion_of_studies``); the product is rounded
       to 9 decimals first, so that 0.57 of 100 studies is 57. The paper sums every
       study when there are few (p = 1 for its 3 to 6 studies of Parkinson's,
       Huntington's disease and schizophrenia) and chose p by prediction in held-out
       studies when there are many (0.65 of 17 studies of Alzheimer's disease, 0.55
       of 16 of COVID-19); :func:`~cytepool.held_out_prediction.held_out_prediction`
       gives that prediction for a chosen p.
    4. A sum above m / 2, the mean of a sum of m uniform ranks, is set to m / 2, so
       that genes without consistent evidence share one value.

    The result is the statistic of genes up (``up_statistic``): small for a gene that
    ranks high in its best m studies. Genes down have the same statistic of the
    negated statistics (``down_statistic``), whose normalized ranks are 1 - u, as the
    paper tests genes down by testing controls against cases.

Genes that a study did not measure or did not test:
    The genes ranked are those that every study tested (a finite or infinite
    statistic, not NaN), as the paper ranks the genes that every dataset measured,
    the intersection of their genes. A rank says where a gene stands among the genes
    ranked with it, so the studies must rank the same genes; and a gene ranked in
    fewer studies would have a statistic, and a distribution under the null, of its
    own. The other genes have NaN. A study that did not analyse the group or the
    contrast at all is left out by :func:`differential_expression_sumrank`, as the
    authors' code leaves out a dataset without the cell type: k counts the studies
    that did.

Irwin-Hall p-values:
    If every study ranked the genes in an order of its own, at random and
    independently of the others, each normalized rank would be uniform on [0, 1],
    and the sum of m of them would follow the Irwin-Hall distribution of a sum of m
    independent uniform variables (Irwin 1927; Hall 1927), whose distribution
    function is

        F_m(x) = (1/m!) sum_{j=0}^{floor(x)} (-1)^j C(m, j) (x - j)^m.

    ``up_irwin_hall_p_value`` is F_m(s) for the statistic s of a gene below the cap,
    and 1 at the cap: the probability under that null of a statistic at most s.
    It treats the ranks as continuous: a gene ranked first in every study (s = 0)
    has p-value 0, where its probability is G^-m.

    With every study summed (m = k), the p-value is exact under that null, apart
    from the steps of the ranks. A sum of the m best of k ranks (m < k) is smaller
    than a sum of m ranks, so its Irwin-Hall p-values are too small: the paper takes
    its p-values from permutations instead, as below, and so should an analysis
    with m < k, as :func:`differential_expression_sumrank` warns. Genes whose ranks
    are not uniform under the null, such as genes with a p-value of 1 in many
    permutations, depart from it too.

    The alternating sum above loses every digit to cancellation near the middle of
    the distribution for tens of studies. F_m is computed instead as a sum of
    positive terms, F_m(x) = sum_{i=0}^{floor(x)} N_{m+1}(x - floor(x) + i), where
    N_{m+1} is the cardinal B-spline of order m + 1, the density of the Irwin-Hall
    distribution of m + 1 variables, which de Boor's recurrence evaluates with
    positive terms only (de Boor 1978): N_{m+1}' (x) = N_m(x) - N_m(x - 1), so the
    sum telescopes to the integral of N_m. It agrees with a high-precision
    computation to about 1e-14 relative for up to 200 studies
    (``validation/sumrank.py``).

Permutation p-values:
    The paper permutes the case and control labels of the samples of every study,
    runs each study's analysis and SumRank again, and pools the statistics of every
    gene in every permutation into one null distribution: with B permutations of G
    genes, N = B G null statistics. The p-value of a gene of statistic s is

        p = (1 + the number of null statistics at most s) / (1 + N),

    as Phipson and Smyth (2010) form p-values from random permutations, so that it
    is never zero, and a gene at the cap has p = 1. Pooling over the genes gives
    small p-values from few permutations (100 permutations of 10,000 genes give a
    p-value as small as 1e-6), for genes whose null distributions are alike. The
    statistics of a permutation are computed as those of the data, over the genes
    that every study's analysis of that permutation tested.

    :func:`sumrank` takes the statistics of the permutations as an array;
    :func:`differential_expression_sumrank` permutes the values of one variable of
    each study's design (``permute``, such as the condition) among the samples of
    the study's fit of the group, with a ``numpy.random.Generator``, and re-runs the
    study's analysis, ``limma_voom`` or ``limma_trend`` with the design and settings
    its result records, on its aligned cube, for that group and contrast only. The
    permutations are drawn in a fixed order, permutation after permutation and
    study after study, so a seed gives the same permutations, and the analyses do
    not depend on the number of threads.

Cost:
    A permutation runs the analysis of every study once, for one group: with
    ``limma_voom``, 0.21 s for a study of 40 samples and 10,000 genes on a laptop on
    2 threads (``limma_trend`` 0.14 s; ``validation/sumrank.py --benchmark``), so 100
    permutations of 20 such studies take about 7 minutes; the SumRank statistics of a
    permutation take milliseconds. The memory is that of one study's analysis and a
    few arrays of the genes, whatever the number of permutations.

Deliberate differences from the paper and its authors' code:
    - The Irwin-Hall p-value is the lower tail of the distribution, as above. The
      paper calls its Irwin-Hall p-values two-sided, which would double these, and
      the authors' code computes the density of the Irwin-Hall distribution at the
      statistic instead (the unifed package's ``dirwin.hall``), which is not a
      probability. Its permutation p-values compare -log10 of that density, which
      falls as the statistic rises up to the cap for m >= 2: they order genes as the
      statistic does, so the permutation p-values here do not depend on this choice.
    - The authors' code takes as the permutation p-value the share of null
      statistics strictly below the gene's (``1 - ecdf`` of -log10 of the density),
      without the one above: a gene beyond every null statistic has the p-value 0,
      and a gene at the cap the share of null statistics below the cap rather than
      1. Here ties count as at least as extreme, which keeps the p-value valid.
    - The labels are permuted among the samples of each study's fit of the group,
      which keeps the number of cases and controls of the fit. The paper permutes
      them among all the individuals of a dataset, for all its cell types at once,
      and draws again until every cell type has at least two cases and two controls.
    - The authors' code sums the normalized ranks in floating point, in each gene's
      own order, so that genes with equal statistics can compare as unequal in its
      calibration; here the sums are exact. It stops when it sums one study (m = 1),
      and takes floor(p k) of the product in floating point, which for 0.57 of 100
      studies is 56.

Not done:
    Studies are not weighted, a limit the paper notes too. Permutations re-run
    designs without random effects only (``limma_voom`` and ``limma_trend``), whose
    samples are exchangeable under the null; a design with several samples of a
    donor needs the donors' labels permuted instead.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Irwin JO (1927). On the frequency distribution of the means of samples from a
    population having any law of frequency with finite moments, with special
    reference to Pearson's Type II. Biometrika 19, 225-239.
    doi:10.1093/biomet/19.3-4.225

    Hall P (1927). The distribution of means for samples of size N drawn from a
    population in which the variate takes values between 0 and 1, all such values
    being equally probable. Biometrika 19, 240-244. doi:10.1093/biomet/19.3-4.240

    de Boor C (1978). A Practical Guide to Splines. Springer, New York.
    doi:10.1007/978-1-4612-6333-3

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import inspect
import json
import math
import numbers
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._analysis_records import software_versions
from ._columns import text_column
from ._jit import jit
from ._masked_arrays import read_only, reject_masked, unmasked_array
from ._messages import (
    PACKAGE_DIRECTORY,
    fitted_lines,
    fitted_message,
    listing,
    quoted,
)
from ._options import index_integer, random_generator
from ._ranks import average_ranks
from ._study_statistics import signed_statistics, study_statistics, studies_summed
from .cube import Cube, CubeCollection
from .design import Design
from .differential_expression import (
    DifferentialExpressionResult,
    limma_trend,
    limma_voom,
)
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = [
    "SumRankFit",
    "SumRankResult",
    "differential_expression_sumrank",
    "sumrank",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_DIRECTIONS = ("up", "down")
# The analyses that permutations re-run: a design without random effects, whose
# samples are exchangeable under the null.
_RERUN: dict[str, Callable[..., DifferentialExpressionResult]] = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
}


@dataclass(frozen=True, eq=False, repr=False)
class SumRankFit:
    """SumRank of every gene across studies, from :func:`sumrank`.

    The per-gene arrays hold one value per gene (feature), NaN for a gene that some
    study did not test; they are read-only.

    Attributes:
        up_statistic: The sum of the gene's ``n_studies_summed`` smallest normalized
            ranks, set to ``n_studies_summed / 2`` where it is larger: small for a
            gene up in the studies that rank it best.
        down_statistic: The same for the negated statistics: small for a gene down.
        up_irwin_hall_p_value: The Irwin-Hall distribution function of the sum of
            ``n_studies_summed`` uniform ranks at ``up_statistic``, or 1 at the cap.
        down_irwin_hall_p_value: The same for ``down_statistic``.
        up_permutation_p_value: (1 + the number of null statistics at most
            ``up_statistic``) / (1 + ``n_null_values``), over the statistics of
            every gene in every permutation; None without permutations.
        down_permutation_p_value: The same for ``down_statistic``.
        normalized_ranks: ``(n_studies, n_features)``: the normalized rank of every
            gene in every study, (r - 1) / (G - 1) for the rank r from 1 for the
            largest statistic among the G genes ranked; NaN for a gene not ranked.
            The ranks of the genes down are 1 minus these.
        ranked: Whether each gene was ranked: tested by every study.
        n_studies: The number of studies, k.
        n_studies_summed: The number of ranks summed for each gene, m.
        proportion_of_studies: The proportion of the studies that sets m.
        n_permutations: The number of permutations, 0 without them.
        n_null_values: The number of null statistics of each direction, the genes
            ranked in each permutation, summed over the permutations; 0 without
            them.
    """

    up_statistic: FloatArray
    down_statistic: FloatArray
    up_irwin_hall_p_value: FloatArray
    down_irwin_hall_p_value: FloatArray
    up_permutation_p_value: FloatArray | None
    down_permutation_p_value: FloatArray | None
    normalized_ranks: FloatArray
    ranked: BoolArray
    n_studies: int
    n_studies_summed: int
    proportion_of_studies: float
    n_permutations: int
    n_null_values: int

    def __repr__(self) -> str:
        text = (
            f"SumRankFit(n_features={self.ranked.size}, ranked="
            f"{int(np.count_nonzero(self.ranked))}, n_studies={self.n_studies}, "
            f"n_studies_summed={self.n_studies_summed}"
        )
        if self.n_permutations:
            text += f", n_permutations={self.n_permutations}"
        return text + ")"


def sumrank(
    statistics: npt.ArrayLike,
    *,
    proportion_of_studies: float = 1.0,
    null_statistics: npt.ArrayLike | None = None,
) -> SumRankFit:
    """Rank every gene in every study and sum its best ranks, as SumRank does.

    Each study ranks the genes that every study tested by their statistic, from the
    largest; a gene's normalized ranks in the studies that rank it best are summed and
    capped at their mean under the null, for genes up and, with the statistics
    negated, for genes down. The module description gives the statistic, its
    Irwin-Hall p-values and the permutation p-values.

    Args:
        statistics: ``(n_studies, n_features)`` signed statistics, larger for a gene
            more up, such as the signed -log10 p-values of each study's test; NaN
            where a study did not test a gene, and infinite values are allowed (a
            p-value of 0). Only their order within each study matters.
        proportion_of_studies: The proportion p of the k studies whose ranks are
            summed for each gene: the m = floor(p k) studies that rank it best, at
            least one, in (0, 1]; 1 sums every study.
        null_statistics: ``(n_permutations, n_studies, n_features)``: the statistics
            of the same studies with their labels permuted, one study's analysis
            re-run on each permutation, NaN where an analysis did not test a gene;
            or None for no permutation p-values.

    Returns:
        The statistics of every gene, up and down, their Irwin-Hall p-values and,
        with ``null_statistics``, their permutation p-values.

    Raises:
        TypeError: If an argument is not an array of real numbers, is a numpy masked
            array, or ``proportion_of_studies`` is not a real number.
        ValueError: If the shapes are wrong, a study tested no gene, fewer than two
            genes were tested by every study (in the data or in a permutation), or
            ``proportion_of_studies`` is not in (0, 1].

    Example:
        Two studies of four genes: the first gene is the largest in both, so its
        ranks sum to 0; the others reach the cap of 1 (two ranks of mean 1/2).

        >>> fit = sumrank([[3.0, 1.0, -2.0, 0.5], [2.0, -1.0, 0.0, 1.5]])
        >>> fit.up_statistic.tolist()
        [0.0, 1.0, 1.0, 1.0]
        >>> fit.down_statistic.tolist()
        [1.0, 0.6666666666666666, 0.3333333333333333, 1.0]
        >>> fit.down_irwin_hall_p_value.tolist()  # (1/3)^2 / 2 = 1/18 for the third
        [1.0, 0.2222222222222222, 0.05555555555555555, 1.0]
    """
    values = _statistics(statistics, "statistics", 2)
    n_studies = values.shape[0]
    summed = studies_summed(proportion_of_studies, n_studies)
    null = None
    if null_statistics is not None:
        null = _statistics(null_statistics, "null_statistics", 3)
        if null.shape[1:] != values.shape:
            raise ValueError(
                "null_statistics must be (n_permutations, n_studies, n_features) with "
                f"the studies and features of statistics, {values.shape}; got "
                f"{null.shape}"
            )
        if null.shape[0] == 0:
            raise ValueError(
                "null_statistics holds no permutation; give at least one, or None for "
                "no permutation p-values"
            )
    observed = _capped_sums(values, summed, "statistics")
    up = _full(observed.up, observed.ranked)
    down = _full(observed.down, observed.ranked)
    permutation: list[FloatArray | None] = [None, None]
    n_null = 0
    if null is not None:
        counts = _NullCounts(observed)
        for b, permuted in enumerate(null):
            counts.add(_capped_sums(permuted, summed, f"permutation {b}"))
        permutation = [read_only(_full(p, observed.ranked)) for p in counts.p_values()]
        n_null = counts.n_values
    return SumRankFit(
        up_statistic=read_only(up),
        down_statistic=read_only(down),
        up_irwin_hall_p_value=read_only(_full(observed.up_p, observed.ranked)),
        down_irwin_hall_p_value=read_only(_full(observed.down_p, observed.ranked)),
        up_permutation_p_value=permutation[0],
        down_permutation_p_value=permutation[1],
        normalized_ranks=read_only(observed.ranks_of_every_feature()),
        ranked=read_only(observed.ranked),
        n_studies=n_studies,
        n_studies_summed=summed,
        proportion_of_studies=float(proportion_of_studies),
        n_permutations=0 if null is None else int(null.shape[0]),
        n_null_values=n_null,
    )


# The statistic.


def _statistics(values: Any, name: str, ndim: int) -> FloatArray:
    array = unmasked_array(values, name, missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if array.ndim != ndim:
        shape = (
            "(n_studies, n_features)"
            if ndim == 2
            else "(n_permutations, n_studies, n_features)"
        )
        raise ValueError(f"{name} must be {shape}, got {array.ndim} dimensions")
    converted: FloatArray = np.array(array, dtype=np.float64)
    if ndim == 2 and converted.shape[0] == 0:
        raise ValueError(f"{name} has no study: give one row per study")
    return converted


@dataclass(frozen=True)
class _Sums:
    """The capped sums of the genes every study tested, in units that are exact.

    ``up`` and ``down`` are the statistics of the ``ranked`` genes, in their order,
    and ``up_p`` and ``down_p`` their Irwin-Hall p-values; ``ranks`` holds the
    normalized ranks of these genes in every study.
    """

    ranked: BoolArray  # per feature
    up: FloatArray  # per ranked gene
    down: FloatArray
    up_p: FloatArray
    down_p: FloatArray
    ranks: FloatArray  # (n_studies, n_ranked)

    def ranks_of_every_feature(self) -> FloatArray:
        full = np.full((self.ranks.shape[0], self.ranked.size), np.nan)
        full[:, self.ranked] = self.ranks
        return full


def _capped_sums(values: FloatArray, summed: int, where: str) -> _Sums:
    """The SumRank statistics of the genes every study tested, up and down.

    The rank r of a gene among G (from 1 for the largest value) is an integer or, for
    ties, a half-integer, so q = 2 (r - 1) is an integer and a sum of normalized ranks
    is (sum of q) / (2 (G - 1)): it is formed exactly in integers, and compared with
    the cap m (G - 1) there, so that equal sums are equal in every permutation.
    """
    tested = ~np.isnan(values)
    empty = np.flatnonzero(~tested.any(axis=1))
    if empty.size:
        raise ValueError(
            f"{where}: study {int(empty[0])} tested no gene (every value is NaN); "
            "leave out a study that did not test the group and contrast"
        )
    ranked = tested.all(axis=0)
    n_ranked = int(np.count_nonzero(ranked))
    if n_ranked < 2:
        genes = "1 gene was" if n_ranked == 1 else f"{n_ranked} genes were"
        raise ValueError(
            f"{where}: {genes} tested by every study, and SumRank ranks the genes "
            "that every study tested against each other; it needs at least two"
        )
    kept = values[:, ranked]
    n_studies = kept.shape[0]
    twice_up = np.empty(kept.shape, dtype=np.int64)  # 2 (r - 1), r from the largest
    for row in range(n_studies):
        ascending = average_ranks(np.ascontiguousarray(kept[row]))  # 1 = smallest
        # r = G + 1 - ascending rank, so 2 (r - 1) = 2 (G - ascending rank).
        twice_up[row] = np.rint(2.0 * (n_ranked - ascending)).astype(np.int64)
    twice_down = 2 * (n_ranked - 1) - twice_up
    scale = 2.0 * (n_ranked - 1)
    statistics = []
    for twice in (twice_up, twice_down):
        if summed < n_studies:
            best = np.partition(twice, summed - 1, axis=0)[:summed]
        else:
            best = twice
        total = best.sum(axis=0)
        cap = summed * (n_ranked - 1)
        capped = np.minimum(total, cap)
        statistic = capped / scale
        p_value = np.ones(statistic.size)
        below = capped < cap
        p_value[below] = _irwin_hall_cdf(statistic[below], summed)
        statistics.append((statistic, p_value))
    return _Sums(
        ranked=ranked,
        up=statistics[0][0],
        down=statistics[1][0],
        up_p=statistics[0][1],
        down_p=statistics[1][1],
        ranks=twice_up / scale,
    )


def _full(values: FloatArray, ranked: BoolArray) -> FloatArray:
    """Values of the ranked genes in an array of every feature, NaN elsewhere."""
    full = np.full(ranked.size, np.nan)
    full[ranked] = values
    return full


class _NullCounts:
    """For every gene, the number of null statistics at most its own, per direction,
    accumulated one permutation at a time, so that the null distribution is never
    held whole."""

    def __init__(self, observed: _Sums) -> None:
        self._observed = (observed.up, observed.down)
        self._counts = [np.zeros(observed.up.size, dtype=np.int64) for _ in range(2)]
        self.n_values = 0

    def add(self, permutation: _Sums) -> None:
        for counts, statistic, null in zip(
            self._counts,
            self._observed,
            (permutation.up, permutation.down),
            strict=True,
        ):
            counts += np.searchsorted(np.sort(null), statistic, side="right")
        self.n_values += permutation.up.size

    def p_values(self) -> list[FloatArray]:
        return [
            np.divide(1.0 + counts, 1.0 + self.n_values, dtype=np.float64)
            for counts in self._counts
        ]


# The Irwin-Hall distribution function.


def _irwin_hall_cdf(x: FloatArray, n: int) -> FloatArray:
    """F_n at each x, the distribution function of a sum of n uniform variables."""
    out = np.empty(x.size)
    _irwin_hall_cdf_into(np.ascontiguousarray(x, dtype=np.float64), n, out)
    return out


@jit
def _irwin_hall_cdf_into(x: FloatArray, n: int, out: FloatArray) -> None:
    """Fill ``out`` with F_n(x), as a sum of values of the cardinal B-spline of order
    n + 1 at t + i for the fractional part t of x, which de Boor's recurrence
    evaluates with positive terms only; above n / 2 by the symmetry F_n(x) =
    1 - F_n(n - x)."""
    b = np.empty(n + 1)
    for g in range(x.size):
        value = x[g]
        if not value > 0.0:
            out[g] = 0.0 if value <= 0.0 else math.nan
            continue
        if not value < n:
            out[g] = 1.0
            continue
        upper = value > 0.5 * n
        if upper:
            value = n - value
        whole = int(math.floor(value))
        t = value - whole
        # b[i] = N_k(t + i) for the cardinal B-spline N_k of order k, from k = 1
        # (the indicator of [0, 1)) to k = n + 1, updating from the top so that
        # b[i - 1] still holds order k - 1.
        b[0] = 1.0
        for i in range(1, n + 1):
            b[i] = 0.0
        for k in range(2, n + 2):
            for i in range(k - 1, -1, -1):
                term = (t + i) * b[i]
                if i > 0:
                    term += (k - t - i) * b[i - 1]
                b[i] = term / (k - 1)
        total = 0.0
        for i in range(whole + 1):
            total += b[i]
        out[g] = 1.0 - total if upper else total


# SumRank of the studies of a collection.


class SumRankResult:
    """SumRank of the differential expression of the studies of a collection.

    Built by :func:`differential_expression_sumrank`, for one group (cell type) and
    contrast. The table has one row per gene that every study tested, in the
    collection's order of the features:

    - ``feature_id`` and, when the collection's features have it, ``symbol``;
    - ``up_statistic``: the SumRank statistic of the gene up, the capped sum of its
      best normalized ranks (small for a gene up in the studies that rank it best);
    - ``up_p_value``: its p-value, from permutations when they were run
      (``settings["p_values"]`` is ``"permutation"``), from the Irwin-Hall
      distribution otherwise (``"irwin_hall"``);
    - ``up_adjusted_p_value``: Benjamini-Hochberg over the genes of the table
      (Benjamini and Hochberg 1995);
    - ``up_irwin_hall_p_value``: the Irwin-Hall p-value, also with permutations;
    - ``down_statistic``, ``down_p_value``, ``down_adjusted_p_value`` and
      ``down_irwin_hall_p_value``: the same for the gene down.

    The table of studies has one row per study of the collection: ``study``,
    ``analysed`` (whether its results test the group and contrast), ``reason`` (why
    not; empty when analysed) and ``n_features_tested`` (0 when not analysed).

    Immutable.

    Raises:
        TypeError: If ``table`` or ``studies`` is not a ``Table``, or ``provenance``
            not a mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_table", "_studies", "_record")

    _table: Table
    _studies: Table
    _record: str

    def __init__(
        self, table: Table, studies: Table, provenance: Mapping[str, Any]
    ) -> None:
        for name, given in (("table", table), ("studies", studies)):
            if not isinstance(given, Table):
                raise TypeError(f"{name} must be a Table, got {type(given).__name__}")
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_table": table,
            "_studies": studies,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SumRankResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SumRankResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            SumRankResult,
            (self._table, self._studies, json.loads(self._record)),
        )

    @property
    def table(self) -> Table:
        """The statistics and p-values of every gene ranked."""
        return self._table

    @property
    def studies(self) -> Table:
        """Every study of the collection, whether it analysed the group and
        contrast, and why not."""
        return self._studies

    @property
    def settings(self) -> dict[str, Any]:
        """The settings, as a new dict (also in the provenance), with the group, the
        contrast, the proportion of studies, the numbers of studies and of ranks
        summed, the source of the p-values and the permutations."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions; the settings; the
        number of genes ranked and of genes left out, measured by a study but not
        tested by every study that analysed the group and contrast; and the
        provenance of every study's differential-expression result, by study name,
        which identifies its cube by its fingerprint and gives its design and
        settings and the samples of every group's fit.
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def __repr__(self) -> str:
        settings = self.settings
        table = self._table
        lines = [
            f"SumRankResult for group {settings['group']!r} and contrast "
            f"{settings['contrast']!r}: {settings['n_studies']:,} studies, the best "
            f"{settings['n_studies_summed']:,} summed; {len(table):,} genes"
        ]
        if settings["p_values"] == "permutation":
            lines.append(
                f"  p-values from {settings['n_permutations']:,} permutations of "
                f"{settings['permute']!r}"
            )
        else:
            lines.append("  p-values from the Irwin-Hall distribution")
        counts = [
            int(np.count_nonzero(table[f"{d}_adjusted_p_value"] < 0.05))
            for d in _DIRECTIONS
        ]
        lines.append(
            f"  adjusted p-value below 0.05: {counts[0]:,} up, {counts[1]:,} down"
        )
        return fitted_lines("\n".join(lines))


def differential_expression_sumrank(
    collection: CubeCollection,
    results: Mapping[str, DifferentialExpressionResult],
    group: str,
    contrast: str,
    *,
    proportion_of_studies: float = 1.0,
    n_permutations: int = 0,
    permute: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> SumRankResult:
    """SumRank of one group and contrast across the studies of a collection.

    Each study's result is the analysis of its aligned cube
    (:meth:`~cytepool.cube.CubeCollection.aligned_cube`), such as
    :func:`~cytepool.differential_expression.limma_voom` with the study's own design,
    as for :func:`~cytepool.meta_analysis.align_study_results`. For the group and
    contrast, the studies whose results test them each rank the genes that every one
    of them tested by their signed -log10 p-value, and :func:`sumrank` sums each
    gene's best ranks; the module description gives the statistic and its p-values.

    With ``n_permutations``, the values of the design variable ``permute``, such as
    the condition, are permuted among the samples of each study's fit of the group,
    and each study's analysis is run again on its aligned cube with the design and
    settings its result records (``limma_voom`` or ``limma_trend``), for this group
    and contrast only; the SumRank statistics of every permutation make the null
    distribution of the p-values. Each permutation costs one analysis of the group in
    every study (see "Cost" in the module description).

    Args:
        collection: The collection whose aligned cubes the studies' results
            analysed.
        results: The differential-expression result of every study of the
            collection, by study name, from
            :func:`~cytepool.differential_expression.limma_voom`,
            :func:`~cytepool.differential_expression.limma_trend` or
            :func:`~cytepool.differential_expression.dream_voom` (without
            permutations).
        group: An aligned group id of the collection.
        contrast: A contrast name that the studies tested.
        proportion_of_studies: The proportion of the studies that analysed the group
            whose ranks are summed for each gene, those that rank it best, in
            (0, 1]; 1 sums every study.
        n_permutations: The number of permutations, at least 0; 0 gives Irwin-Hall
            p-values only.
        permute: The variable of the studies' designs whose values are permuted,
            such as ``"condition"``; needed with permutations.
        rng: A ``numpy.random.Generator`` to draw the permutations from, an integer
            seed for one, or None for a fresh, unseeded generator.

    Returns:
        The statistics and p-values of every gene that every study analysing the
        group and contrast tested, a table of the studies, and a provenance record.

    Warns:
        RuntimeWarning: If fewer than all the studies are summed without
            permutations: the p-values are then the Irwin-Hall p-values, which are
            too small for such sums.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``results`` does not give the result of every study of the
            collection, a result did not analyse its study's aligned cube, no study
            analysed the group and contrast, the contrast is an F-test, fewer than
            two genes were tested by every study, an option is invalid,
            ``permute`` is given without permutations or missing with them, or a
            study cannot be permuted: its analysis has random effects, its design
            has no variable ``permute``, or a permutation leaves the contrast
            untestable.
        KeyError: If ``group`` is not a group of the collection, or no study tested
            ``contrast``.
    """
    reject_masked(n_permutations, "n_permutations")
    permutations = _n_permutations(n_permutations)
    if permute is not None and not isinstance(permute, str):
        reject_masked(permute, "permute")
        raise TypeError(
            f"permute must be a design variable name (str), got {type(permute).__name__}"
        )
    if permutations and permute is None:
        raise ValueError(
            "permutations permute the values of one variable of each study's design, "
            "such as the condition: give its name as permute"
        )
    if not permutations and permute is not None:
        raise ValueError(
            "permute names the variable that permutations permute, but n_permutations "
            "is 0; give the number of permutations"
        )
    generator = random_generator(rng) if permutations else None
    inputs = study_statistics(collection, results, group, contrast)
    used = inputs.analysed
    if not used.any():
        raise ValueError(
            f"no study tested group {group!r} and contrast {contrast!r}: "
            + "; ".join(
                f"{study}: {reason}"
                for study, reason in zip(
                    collection.studies, inputs.reasons, strict=True
                )
            )
        )
    studies = [
        s for s, flag in zip(collection.studies, used.tolist(), strict=True) if flag
    ]
    statistics = inputs.statistics[used]
    summed = studies_summed(proportion_of_studies, len(studies))
    if summed < len(studies) and not permutations:
        warnings.warn(
            fitted_message(
                f"the Irwin-Hall p-values of a sum of the best {summed} of "
                f"{len(studies)} studies' ranks are too small, as the sum of the best "
                "ranks is smaller than a sum of uniform ranks; the p_value columns "
                "hold them, so take p-values from permutations (n_permutations and "
                "permute)"
            ),
            RuntimeWarning,
            skip_file_prefixes=(PACKAGE_DIRECTORY,),
        )
    observed = _capped_sums(statistics, summed, "the studies' results")
    counts = None
    if permutations:
        assert generator is not None and permute is not None
        counts = _NullCounts(observed)
        for b, permuted in enumerate(
            _permuted_statistics(
                collection,
                results,
                studies,
                group,
                contrast,
                permute,
                permutations,
                generator,
            )
        ):
            counts.add(_capped_sums(permuted, summed, f"permutation {b}"))
    ranked = observed.ranked
    features = collection.features
    columns: dict[str, Any] = {"feature_id": features["feature_id"][ranked]}
    if "symbol" in features:
        columns["symbol"] = features["symbol"][ranked]
    p_values = [observed.up_p, observed.down_p] if counts is None else counts.p_values()
    for direction, statistic, p_value, irwin_hall in zip(
        _DIRECTIONS,
        (observed.up, observed.down),
        p_values,
        (observed.up_p, observed.down_p),
        strict=True,
    ):
        columns[f"{direction}_statistic"] = statistic
        columns[f"{direction}_p_value"] = p_value
        columns[f"{direction}_adjusted_p_value"] = adjust_p_values(p_value, "bh")
        columns[f"{direction}_irwin_hall_p_value"] = irwin_hall
    measured = inputs.measured[used].any(axis=0)
    settings = {
        "group": group,
        "contrast": contrast,
        "statistic": "signed -log10 p-value",
        "proportion_of_studies": float(proportion_of_studies),
        "n_studies": len(studies),
        "n_studies_summed": summed,
        "p_values": "irwin_hall" if counts is None else "permutation",
        "n_permutations": permutations,
        "permute": permute,
        "seed": (
            int(rng)
            if isinstance(rng, numbers.Integral)
            and not isinstance(rng, bool | np.bool_)
            else None
        ),
        "n_null_values": 0 if counts is None else counts.n_values,
        "p_value_adjustment": "BH",
    }
    return SumRankResult(
        Table(columns),
        Table(
            {
                "study": text_column(list(collection.studies)),
                "analysed": used.copy(),
                "reason": text_column(inputs.reasons),
                "n_features_tested": inputs.n_tested,
            }
        ),
        {
            "analysis": "differential_expression_sumrank",
            "versions": software_versions(),
            "settings": settings,
            "n_features": int(np.count_nonzero(ranked)),
            "n_features_left_out": int(np.count_nonzero(measured & ~ranked)),
            "studies": {name: results[name].provenance for name in collection.studies},
        },
    )


def _n_permutations(value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError("n_permutations must be a whole number, not bool")
    number = index_integer(value, "n_permutations", expected="a whole number")
    if number < 0:
        raise ValueError(f"n_permutations must be at least 0, got {number}")
    return number


# Permutations: each study's analysis re-run with the labels of its samples permuted.


@dataclass(frozen=True)
class _Rerun:
    """What it takes to run a study's analysis again with permuted labels."""

    study: str
    cube: Cube
    function: Callable[..., DifferentialExpressionResult]
    options: dict[str, Any]
    formula: str
    levels: dict[str, list[Any]]
    samples: dict[str, list[Any]]
    weights: list[float]  # of the contrast on the design's columns
    used: list[int]  # the design rows of the samples of the group's fit
    fingerprint: str
    settings: dict[str, Any]


def _permuted_statistics(
    collection: CubeCollection,
    results: Mapping[str, DifferentialExpressionResult],
    studies: list[str],
    group: str,
    contrast: str,
    permute: str,
    n_permutations: int,
    generator: np.random.Generator,
) -> Iterator[FloatArray]:
    """The signed statistics of every study for each permutation, ``(n_studies,
    n_features)`` in the collection's order, NaN where an analysis did not test a
    gene. The permutations are drawn permutation by permutation, study by study."""
    reruns = [
        _rerun(collection, results[study], study, group, contrast, permute)
        for study in studies
    ]
    index = {
        feature: position
        for position, feature in enumerate(collection.features["feature_id"].tolist())
    }
    checked = [False] * len(reruns)
    for _ in range(n_permutations):
        statistics = np.full((len(reruns), len(index)), np.nan)
        for row, rerun in enumerate(reruns):
            values = list(rerun.samples[permute])
            drawn = generator.permutation(len(rerun.used))
            for position, source in zip(rerun.used, drawn.tolist(), strict=True):
                values[position] = rerun.samples[permute][rerun.used[source]]
            design = Design(
                {**rerun.samples, permute: values},
                rerun.formula,
                levels=rerun.levels,
                contrasts={contrast: rerun.weights},
            )
            result = rerun.function(rerun.cube, design, groups=[group], **rerun.options)
            if not checked[row]:
                _check_rerun(rerun, result)
                checked[row] = True
            table = result.tables.get((group, contrast))
            if table is None:
                raise ValueError(
                    f"a permutation of {permute!r} in study {rerun.study!r} leaves "
                    f"contrast {contrast!r} untestable in group {group!r}: "
                    f"{_absence(result, group, contrast)}"
                )
            positions = np.array(
                [index[feature] for feature in table["feature_id"].tolist()],
                dtype=np.intp,
            )
            statistics[row, positions] = signed_statistics(
                table["p_value"], table["log2_fold_change"]
            )
        yield statistics


def _rerun(
    collection: CubeCollection,
    result: DifferentialExpressionResult,
    study: str,
    group: str,
    contrast: str,
    permute: str,
) -> _Rerun:
    """How to run a study's analysis again: its function and options, its design,
    and the samples of its fit of the group, from what its result records."""
    function = _RERUN.get(result.method)
    if function is None:
        raise ValueError(
            f"the result of study {study!r} is from {result.method}, whose design has "
            "random effects: its samples are not exchangeable under the null, so its "
            "labels cannot be permuted sample by sample; permutations re-run "
            "limma_voom and limma_trend"
        )
    record = result.provenance
    design = record["design"]
    samples: dict[str, list[Any]] = design["samples"]
    variables = [name for name in samples if name != "sample_id"]
    if permute not in variables:
        raise ValueError(
            f"permute names {permute!r}, which is not a variable of the design of "
            f"study {study!r}; its variables are: {quoted(variables) or '(none)'}"
        )
    settings: dict[str, Any] = record["settings"]
    parameters = inspect.signature(function).parameters
    options = {
        name: settings[name]
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in ("groups", "contrasts")
        and name in settings
    }
    if options.get("expression_filter", True) is None:
        options["expression_filter"] = False  # recorded as None when it was off
    ids = samples["sample_id"]
    rows = {sample: position for position, sample in enumerate(ids)}
    table = result.samples
    in_group = (table["group_id"] == group) & table["used"].astype(bool)
    used = [rows[sample] for sample in table["sample_id"][in_group].tolist()]
    return _Rerun(
        study=study,
        cube=collection.aligned_cube(study),
        function=function,
        options=options,
        formula=design["formula"],
        levels=design["levels"],
        samples=samples,
        weights=design["contrasts"][contrast],
        used=used,
        fingerprint=record["cube"]["fingerprint"],
        settings=settings,
    )


def _check_rerun(rerun: _Rerun, result: DifferentialExpressionResult) -> None:
    """Refuse a re-run on another cube, or with other settings, than the study's
    analysis: its permutations would not be those of the study's results."""
    record = result.provenance
    if record["cube"]["fingerprint"] != rerun.fingerprint:
        raise ValueError(
            f"the result of study {rerun.study!r} was computed from a cube whose "
            f"fingerprint, {rerun.fingerprint}, is not that of the study's aligned "
            f"cube, {record['cube']['fingerprint']}: analyse "
            f"collection.aligned_cube({rerun.study!r})"
        )
    if record["settings"] != rerun.settings:
        changed = [
            name
            for name in sorted(set(record["settings"]) | set(rerun.settings))
            if record["settings"].get(name) != rerun.settings.get(name)
        ]
        raise ValueError(  # pragma: no cover - the options are those recorded
            f"the analysis of study {rerun.study!r} could not be re-run with its "
            f"recorded settings; these differ: {listing(changed)}"
        )


def _absence(result: DifferentialExpressionResult, group: str, contrast: str) -> str:
    """Why a result has no table of a group and contrast: its own message."""
    try:
        result.table(group, contrast)
    except KeyError as error:
        return str(error.args[0])
    raise AssertionError("a table that is not in result.tables")  # pragma: no cover
