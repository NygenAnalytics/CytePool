"""SumRank: genes whose differential expression ranks high in study after study.

Most genes that one single-cell study finds differentially expressed are not found by
the next (Nakatsuka et al. 2025). SumRank asks instead whether a gene ranks high in
every study: each study ranks its genes by their signed evidence of change, and a
gene's ranks are summed over the studies, so that a gene ranked high everywhere has a
small sum and a gene found by one study alone does not, however small its p-value
there. :func:`fit_sumrank` computes the statistic and its p-values from a signed
statistic of every gene in every study; :func:`cytepool.sumrank` does it for the
studies' differential-expression results of one group (cell type) and contrast, with
p-values from permutations that re-run each study's analysis on its samples with
their labels permuted.

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
       rank it low for reasons of their own. m is given (``n_studies_summed``), or
       m = floor(p k), at least 1, for a proportion p of the studies
       (``proportion_of_studies``), the product rounded to 9 decimals first, so that
       0.57 of 100 studies is 57; by default every study is summed (m = k). The paper
       sums every study when there are few (its 3 to 6 studies of Parkinson's,
       Huntington's disease and schizophrenia) and chose p by prediction in held-out
       studies when there are many (0.65 of 17 studies of Alzheimer's disease, 0.55
       of 16 of COVID-19); :func:`~cytepool.held_out_prediction` gives that
       prediction for a chosen m or p.
    4. A sum above m / 2, the mean of a sum of m uniform ranks, is set to m / 2, so
       that genes without consistent evidence share one value.

    The result is the statistic of genes up (``up_statistic``): small for a gene that
    ranks high in its best m studies. Genes down have the same statistic of the
    negated statistics (``down_statistic``), whose normalized ranks are 1 - u, as the
    paper tests genes down by testing controls against cases.

Genes that a study did not measure or did not test:
    The genes ranked are those that every study tested (a finite or infinite statistic,
    not NaN), as the paper ranks the genes that every dataset measured, the intersection
    of their genes. A rank says where a gene stands among the genes ranked with it, so
    the studies must rank the same genes; and a gene ranked in fewer studies would have
    a statistic, and a distribution under the null, of its own. The other genes have
    NaN. A study that did not analyse the group or the contrast at all is left out by
    :func:`cytepool.sumrank`, as the authors' code leaves out a dataset without the cell
    type: k counts the studies that did.

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

    With every study summed (m = k), the p-value is exact under that null, apart from
    the steps of the ranks. A sum of the m best of k ranks (m < k) is smaller than a sum
    of m ranks, so its Irwin-Hall p-values are too small: the paper takes its p-values
    from permutations instead, as below, and so should an analysis with m < k, as
    :func:`cytepool.sumrank` warns. Genes whose ranks are not uniform under the null,
    such as genes with a p-value of 1 in many permutations, depart from it too.

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

    :func:`fit_sumrank` takes the statistics of the permutations as an array;
    :func:`cytepool.sumrank` draws them by permuting the labels of each study's
    samples and running its analysis again.

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
    - The authors' code sums the normalized ranks in floating point, in each gene's
      own order, so that genes with equal statistics can compare as unequal in its
      calibration; here the sums are exact. It stops when it sums one study (m = 1),
      and takes floor(p k) of the product in floating point, which for 0.57 of 100
      studies is 56.

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
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .._jit import jit
from .._masked_arrays import read_only, reject_masked, unmasked_array
from .._options import index_integer, real_number
from .._ranks import average_ranks

__all__ = ["SumRankFit", "fit_sumrank"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, eq=False, repr=False)
class SumRankFit:
    """SumRank of every gene across studies, from :func:`fit_sumrank`.

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
        proportion_of_studies: The proportion of the studies that set m, 1.0 when
            every study is summed by default, or None when m was given.
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
    proportion_of_studies: float | None
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


def fit_sumrank(
    statistics: npt.ArrayLike,
    *,
    n_studies_summed: int | None = None,
    proportion_of_studies: float | None = None,
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
        n_studies_summed: The number m of studies whose ranks are summed for each
            gene, those that rank it best, from 1 to the number of studies k; by
            default every study.
        proportion_of_studies: Instead of ``n_studies_summed``, the proportion p of
            the k studies whose ranks are summed, in (0, 1]: m = floor(p k), at
            least one; 1 sums every study.
        null_statistics: ``(n_permutations, n_studies, n_features)``: the statistics
            of the same studies with their labels permuted, one study's analysis
            re-run on each permutation, NaN where an analysis did not test a gene;
            or None for no permutation p-values.

    Returns:
        The statistics of every gene, up and down, their Irwin-Hall p-values and,
        with ``null_statistics``, their permutation p-values.

    Raises:
        TypeError: If an argument is not an array of real numbers, is a numpy masked
            array, ``n_studies_summed`` is not a whole number or
            ``proportion_of_studies`` not a real number.
        ValueError: If the shapes are wrong, a study tested no gene, fewer than two
            genes were tested by every study (in the data or in a permutation),
            ``n_studies_summed`` is not from 1 to the number of studies,
            ``proportion_of_studies`` is not in (0, 1], or both are given.

    Example:
        Two studies of four genes: the first gene is the largest in both, so its
        ranks sum to 0; the others reach the cap of 1 (two ranks of mean 1/2).

        >>> fit = fit_sumrank([[3.0, 1.0, -2.0, 0.5], [2.0, -1.0, 0.0, 1.5]])
        >>> fit.up_statistic.tolist()
        [0.0, 1.0, 1.0, 1.0]
        >>> fit.down_statistic.tolist()
        [1.0, 0.6666666666666666, 0.3333333333333333, 1.0]
        >>> fit.down_irwin_hall_p_value.tolist()  # (1/3)^2 / 2 = 1/18 for the third
        [1.0, 0.2222222222222222, 0.05555555555555555, 1.0]
    """
    values = _statistics(statistics, "statistics", 2)
    n_studies = values.shape[0]
    summed, proportion = _studies_summed(
        n_studies_summed, proportion_of_studies, n_studies
    )
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
        proportion_of_studies=proportion,
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


def _studies_summed(
    n_studies_summed: Any, proportion: Any, n_studies: int
) -> tuple[int, float | None]:
    """The number m of ranks SumRank sums for each gene, and the proportion of the
    studies that set it: ``n_studies_summed`` when given (no proportion), floor(p k)
    of a proportion p, at least 1, the product rounded to 9 decimals first so that
    0.57 of 100 studies is 57, or every study (the proportion 1).

    Raises:
        TypeError: If ``n_studies_summed`` is not a whole number or ``proportion``
            not a real number, or either is a numpy masked array.
        ValueError: If both are given, ``n_studies_summed`` is not from 1 to
            ``n_studies``, or ``proportion`` is not in (0, 1].
    """
    reject_masked(n_studies_summed, "n_studies_summed")
    reject_masked(proportion, "proportion_of_studies")
    if n_studies_summed is not None and proportion is not None:
        raise ValueError(
            "give the number of studies summed (n_studies_summed) or their proportion "
            "(proportion_of_studies), not both"
        )
    if n_studies_summed is not None:
        if isinstance(n_studies_summed, bool | np.bool_):
            raise TypeError(
                "n_studies_summed must be a whole number of studies, not bool"
            )
        summed = index_integer(
            n_studies_summed, "n_studies_summed", expected="a whole number of studies"
        )
        if not 1 <= summed <= n_studies:
            raise ValueError(
                f"n_studies_summed must be from 1 to the number of studies, "
                f"{n_studies:,}, got {summed:,}"
            )
        return summed, None
    if proportion is None:
        return n_studies, 1.0
    p = real_number(proportion, "proportion_of_studies", finite=True)
    if not 0.0 < p <= 1.0:
        raise ValueError(
            f"proportion_of_studies must be in (0, 1], such as 1 to sum every study, "
            f"got {p}"
        )
    return max(1, math.floor(round(p * n_studies, 9))), p
