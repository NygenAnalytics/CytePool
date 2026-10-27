"""Gene set enrichment analysis (GSEA) of genes ranked by a statistic, as fgsea computes
it.

:func:`~cytepool.gsea` asks of each gene set whether its genes gather at the top or at
the bottom of a ranking of all genes by a statistic, such as the moderated t-statistic
of a contrast: preranked GSEA (Subramanian et al. 2005) as fgsea 1.32.2 computes it
(Korotkevich et al. 2021), with p-values by fgsea's default multilevel method
(``fgseaMultilevel``) or by permutations of the genes (``fgseaSimple``). Results match
fgsea's exactly when CytePool is given the random seeds that fgsea draws, and within
Monte Carlo error otherwise, except where noted under "Deliberate differences". The
routines port fgsea's (MIT licence), as ``THIRD_PARTY_NOTICES.md`` records.
:func:`~cytepool.plotting.plot_enrichment` and
:func:`~cytepool.plotting.plot_gsea_table` draw its figures.

The genes and the sets:
    The statistics come from a group and contrast of a
    :class:`~cytepool.DifferentialExpressionResult`, from a table of results or from a
    mapping of gene id to statistic. Genes are ranked by decreasing statistic, and genes
    with equal statistics keep the order in which they are given, as in fgsea: the score
    of a set whose genes tie with other genes depends on that order, and a warning says
    how many statistics tie. Rows without a gene id or without a statistic (NaN) are
    left out, where fgsea refuses them; infinite statistics and ids on two rows are
    refused. Each set's genes are matched to the ranked genes, and a set is tested when
    ``min_size`` to ``max_size`` of its genes are ranked (fgsea's ``minSize`` and
    ``maxSize``); a set of every ranked gene is never tested, as its score is not
    defined. The table keeps the order of ``gene_sets``, where fgsea orders the sets by
    name.

The enrichment score:
    With weights |r|^p of the statistics r, p the ``exponent`` (fgsea's
    ``gseaParam``), the walk down the ranking rises at each gene of a set by its share
    of the set's weight and falls at each other gene by 1 / (N - m), for m of the N
    genes in the set. The enrichment score (ES) is the walk's largest deviation from
    zero: with ``score_type="std"``, its maximum where that exceeds the absolute
    minimum, its minimum where the absolute minimum is larger, and 0 where they are
    equal; ``"pos"`` and ``"neg"`` take the maximum and the minimum, for one-sided
    tests. An exponent of 0 gives the unweighted statistic of Mootha et al. (2003), 1
    the weighted one of Subramanian et al. (2005). The leading edge is the set's genes
    from the top of the ranking down to the maximum, for a positive score, or from
    the minimum to the bottom, for a negative one, listed from the most extreme
    statistic inwards: the genes that drive the score.

Random gene sets:
    The null distribution of a set's score is that of random sets of the same size
    (gene permutation). As fgsea does, each of the ``n_permutations`` permutations
    draws one random sample of as many genes as the largest set, and the scores of all
    its first genes, one score for each size, are computed together in
    O(K^(3/2)) time for K genes, from the upper convex hulls of the walk in blocks
    of about sqrt(K) genes (fgsea's ``calcGseaStatCumulative``).

    The normalized enrichment score (NES) divides the ES by the mean of the random
    scores of its sign: of those at least 0 for a positive ES, of the absolute values
    of those at most 0 otherwise; it is NaN where that mean is 0 or there is no such
    score. The permutation p-value is (1 + b) / (1 + n), with b the random scores at
    least as extreme as the ES and n those of its sign, so that it is conditional on
    the sign of the score, as GSEA's is.

P-values by the multilevel method:
    ``method="multilevel"`` (fgsea's default) starts from the permutation p-values of
    ``n_permutations`` permutations. A set with fewer than 10 random scores of the sign
    of its ES gets NaN for its p-value and NES. A set goes to adaptive multilevel
    splitting (Korotkevich et al. 2021) when its permutation p-value would be less
    accurate than the multilevel estimate: with b' of the n random scores beyond its ES
    on its side, when half the larger distance from log2 of (b' + 1) / (n + 1) to the
    ends of a 95% interval of the proportion (from the 2.5% and 97.5% quantiles of beta
    distributions) exceeds the expected error of the multilevel estimate, as fgsea
    decides; with fgsea's defaults, when b' is at most 23. For each set size,
    ``sample_size`` random sets
    are drawn; at each level the half with the lower positive scores (the largest
    deviation of the walk above zero) is replaced by copies of the other half, each
    of which a Markov chain then moves, swapping one gene at a time and keeping its
    score above the level's median, about ``sample_size`` times the size moves in all.
    The levels continue until the median reaches the ES, the median stops rising, or
    the p-value would fall below ``eps``. The probability of each level, about 1/2, is
    estimated on the log scale by the expected logarithm of an order statistic of
    uniform variables, psi(k) - psi(n + 1) for the k-th of n with psi the digamma
    function, and the p-value is the product over the levels, times the share of the
    sampled sets beyond the ES whose signed score is positive, divided by the share of
    random scores of the ES's sign among the permutations. ``log2_error`` gives the
    expected standard deviation of log2 of the p-value: from the number of levels for
    the multilevel method, and from the trigamma function of the counts for the
    permutation p-values (fgsea's ``log2err``). It is NaN where fewer than half of the
    last level's sets have a positive signed score, which makes the p-value likely
    too large, and where the p-value is below ``eps``, which is then its value: when
    the multilevel method has run, p-values below ``eps`` are reported as ``eps``.

Random numbers:
    ``rng`` draws the seeds of the random sets: two distinct integers from 1 to 10^9
    for ``method="multilevel"``, the first for the permutations and the second for the
    multilevel method, and one for each chunk of the permutations for
    ``method="permutation"``, which, as ``fgseaSimple``, splits the permutations into
    chunks of max(1000, ceiling(n / 128)), run in parallel. Each seed starts the
    Mersenne Twister MT19937 (Matsumoto and Nishimura 1998), initialised as C++'s
    ``std::mt19937`` that fgsea uses, from which random integers are drawn by
    rejection as fgsea draws them. Given fgsea's seeds, which R draws with
    ``sample.int(10^9, 1)``, the results are fgsea's: ``validation/gsea.py`` replays
    them. A seeded ``rng`` gives the same results every time, with any number of
    threads.

Deliberate differences from fgsea:
    - Adjusted p-values are Benjamini and Hochberg's over the sets with a p-value.
      When no set goes to the multilevel method, fgsea adjusts them over the sets
      before it removes those with fewer than 10 random scores of their sign, as
      though those sets had been tested; its adjusted p-values are then smaller or
      larger than those of the sets it reports.
    - fgsea tests a single set with random sets from R's ``sample.int``, and several
      sets with its own sampler; CytePool uses fgsea's sampler for one set too, so
      that the two agree within Monte Carlo error.
    - Random samples of at least half of the genes are put in random order by the
      Fisher-Yates shuffle (Durstenfeld 1964) with fgsea's random integers, where
      fgsea calls C++'s ``std::shuffle``, whose algorithm the standard leaves to each
      library; the results agree within Monte Carlo error.

References:
    Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, Paulovich
    A, Pomeroy SL, Golub TR, Lander ES, Mesirov JP (2005). Gene set enrichment
    analysis: a knowledge-based approach for interpreting genome-wide expression
    profiles. Proceedings of the National Academy of Sciences 102, 15545-15550.
    doi:10.1073/pnas.0506580102

    Korotkevich G, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A (2021).
    Fast gene set enrichment analysis. bioRxiv 060012, version 3.
    doi:10.1101/060012

    Mootha VK, Lindgren CM, Eriksson KF, Subramanian A, Sihag S, Lehar J, Puigserver P,
    Carlsson E, Ridderstrale M, Laurila E, Houstis N, Daly MJ, Patterson N, Mesirov JP,
    Golub TR, Tamayo P, Spiegelman B, Lander ES, Hirschhorn JN, Altshuler D, Groop LC
    (2003). PGC-1alpha-responsive genes involved in oxidative phosphorylation are
    coordinately downregulated in human diabetes. Nature Genetics 34, 267-273.
    doi:10.1038/ng1180

    Matsumoto M, Nishimura T (1998). Mersenne twister: a 623-dimensionally
    equidistributed uniform pseudo-random number generator. ACM Transactions on
    Modeling and Computer Simulation 8, 3-30. doi:10.1145/272991.272995

    Durstenfeld R (1964). Algorithm 235: random permutation. Communications of the
    ACM 7, 420. doi:10.1145/364520.364540

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import math
import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numba import prange

from ._columns import text_column
from ._jit import by_chunks, jit, thread_limit
from ._masked_arrays import reject_masked
from ._messages import PACKAGE_DIRECTORY, fitted_message, quoted
from ._options import (
    checked_choice,
    number_at_least,
    positive_integer,
    random_generator,
    real_number,
)
from ._ranked_genes import (
    RankedGenes,
    gene_set_collection,
    matched_positions,
    ranked_genes,
    running_sum,
)
from ._summation import accurate_sum
from .gene_sets import GeneSets
from .stats.multiple_testing import adjust_p_values
from .stats.special import betainc, digamma_scalar, trigamma
from .table import Table

__all__ = ["gsea"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_SCORE_TYPES = ("std", "pos", "neg")
_METHODS = ("multilevel", "permutation")
_SEPARATOR = ", "  # between the genes of a leading edge
_SEED_RANGE = 10**9  # fgsea's seeds: sample.int(10^9, 1), from 1 to 10^9
# fgseaMultilevel: a set with fewer random scores of its sign gets no p-value.
_FEWEST_OF_SIGN = 10
# fgseaSimple's chunks of permutations: at least 1000, and at most 128 chunks.
_CHUNK, _MOST_CHUNKS = 1000, 128
# The Mersenne Twister MT19937, as C++'s std::mt19937 (Matsumoto and Nishimura 1998):
# its state of 624 words, the index of the next word kept after them.
_MT_WORDS = 624
_LARGEST_DRAW = 4294967295  # 2^32 - 1, the largest output of MT19937
# fgsea's constants: the tolerance of its convex hulls, of its multilevel levels,
# the share of a set's genes swapped by one round of its Markov chain, the most
# draws for one gene of a random sample, and the floor of the statistics of a
# random sample (1e-5, divided by 1024 under the smallest nonzero one).
_HULL_TOLERANCE = 1e-13
_LEVEL_TOLERANCE = 1e-10
_SWAP_SHARE = 0.1
_DRAWS_PER_GENE = 100
_STATISTIC_FLOOR, _FLOOR_DIVISOR = 1e-5, 1024.0
_COLUMNS = (
    "gene_set",
    "n_genes",
    "enrichment_score",
    "normalized_enrichment_score",
    "p_value",
    "adjusted_p_value",
    "log2_error",
    "leading_edge",
)


class _Settings(NamedTuple):
    min_size: int
    max_size: int | None
    exponent: float
    score_type: int  # 0 std, 1 pos, 2 neg
    multilevel: bool
    n_permutations: int
    sample_size: int
    eps: float


def gsea(
    statistics: Any,
    gene_sets: Mapping[str, Any],
    *,
    group: str | None = None,
    contrast: str | None = None,
    id_column: str = "feature_id",
    statistic_column: str = "t",
    min_size: int = 1,
    max_size: int | None = None,
    exponent: float = 1.0,
    score_type: str = "std",
    method: str = "multilevel",
    n_permutations: int = 1000,
    sample_size: int = 101,
    eps: float = 1e-50,
    rng: np.random.Generator | int | None = None,
    n_threads: int | None = None,
) -> Table:
    """Gene set enrichment analysis of genes ranked by a statistic, as fgsea.

    Preranked GSEA (Subramanian et al. 2005) as fgsea's ``fgsea`` computes it
    (Korotkevich et al. 2021): whether the genes of each set gather at the top or the
    bottom of the ranking, with the enrichment score, its leading edge, the score
    normalized by those of random sets of the same size, and p-values by fgsea's
    multilevel method or by permutations of the genes. The module docstring gives the
    method, the random numbers and the deliberate differences from fgsea.

    Args:
        statistics: The statistics that rank the genes: a
            :class:`~cytepool.DifferentialExpressionResult`, with ``group`` and
            ``contrast``; a table of results, a ``Table`` or a pandas DataFrame, with a
            column of gene ids and a column of statistics; or a pandas Series of
            statistics indexed by gene id, or a mapping of gene id to statistic. NaN
            marks a gene without a statistic.
        gene_sets: A :class:`~cytepool.GeneSets`, such as one read by
            :func:`~cytepool.read_gmt`, or a mapping of set name to gene ids.
        group: The group (cell type) of a ``DifferentialExpressionResult``; it may be
            left out when the result has one.
        contrast: The contrast of a ``DifferentialExpressionResult``; it may be left
            out when the result tests one.
        id_column: The column of gene ids of a table, such as ``"feature_id"`` or
            ``"symbol"``.
        statistic_column: The column of statistics of a table, such as ``"t"`` (the
            moderated t-statistic) or ``"log2_fold_change"``.
        min_size: The fewest of a set's genes among the ranked genes for the set to be
            tested, at least 1 (fgsea's ``minSize``).
        max_size: The most of them, at least ``min_size``; None, and any value of at
            least the number of ranked genes, mean one fewer than the ranked genes
            (fgsea's ``maxSize``).
        exponent: The power p of the weights |r|^p of the statistics, at least 0
            (fgsea's ``gseaParam``): 1 for GSEA's weighted score, 0 for the unweighted
            one.
        score_type: ``"std"`` for the score of the larger deviation in either
            direction, ``"pos"`` for enrichment at the top only and ``"neg"`` at the
            bottom only (fgsea's ``scoreType``).
        method: ``"multilevel"`` (fgsea's default) or ``"permutation"`` (fgsea's
            ``fgseaSimple``).
        n_permutations: The random samples of genes, at least 1: for the
            normalized scores and the permutation p-values, which are at least
            1 / (n_permutations + 1); fgsea's ``nPermSimple`` for the multilevel
            method and ``nperm`` for permutations.
        sample_size: The random sets of each level of the multilevel method, at least
            3; an even number is raised by one, as fgsea raises it (fgsea's
            ``sampleSize``).
        eps: The smallest p-value the multilevel method estimates, in [0, 1); 0 for no
            limit (fgsea's ``eps``).
        rng: A ``numpy.random.Generator`` to draw the seeds of the random sets from,
            or an integer seed for one, or None for a fresh, unseeded generator.
        n_threads: The most threads to use; None uses numba's current setting. The
            result does not depend on it.

    Returns:
        A table with one row per set tested, in the order of ``gene_sets``:
        ``gene_set``; ``n_genes``, the set's genes among the ranked genes;
        ``enrichment_score``; ``normalized_enrichment_score``; ``p_value``;
        ``adjusted_p_value``, Benjamini-Hochberg over the sets with a p-value;
        ``log2_error``, the expected standard deviation of log2 of the p-value; and
        ``leading_edge``, the genes of the leading edge, from the most extreme
        statistic inwards, separated by ``", "``. A set with too few random scores of
        its sign has NaN normalized score, p-values and error. The table has no row
        when no set has between ``min_size`` and ``max_size`` genes ranked.

    Raises:
        TypeError: If an argument has the wrong type, gene ids are not strings,
            statistics are not numbers, or an argument is or holds a numpy masked
            array.
        ValueError: If ``gene_sets`` is empty, an option is out of range, a table
            lacks a column, a statistic is infinite, a gene id is on two rows, fewer
            than two genes are ranked, every statistic is zero, ``group`` or
            ``contrast`` is given without a ``DifferentialExpressionResult`` or left
            out of one that has several, or no gene of any set is among the ranked
            genes.
        KeyError: If the result has no table of the group and contrast; the message
            says why.

    Example:
        >>> import numpy as np
        >>> genes = [f"g{i}" for i in range(500)]
        >>> statistics = dict(zip(genes, np.linspace(2.0, -2.0, 500)))
        >>> sets = {"near the top": genes[:50:2], "spread": genes[::20]}
        >>> table = gsea(statistics, sets, rng=1)
        >>> table["enrichment_score"].round(3).tolist()
        [0.949, 0.184]
        >>> table["normalized_enrichment_score"].round(2).tolist()
        [3.51, 0.68]
    """
    reject_masked(gene_sets, "gene_sets")
    collection = gene_set_collection(gene_sets)
    settings = _settings(
        min_size,
        max_size,
        exponent,
        score_type,
        method,
        n_permutations,
        sample_size,
        eps,
    )
    ranking = ranked_genes(
        statistics,
        group=group,
        contrast=contrast,
        id_column=id_column,
        statistic_column=statistic_column,
    )
    generator = random_generator(rng)
    with thread_limit(n_threads):
        return _gene_set_enrichment(ranking, collection, settings, generator)


# The analysis.


def _gene_set_enrichment(
    ranking: RankedGenes,
    collection: GeneSets,
    settings: _Settings,
    generator: np.random.Generator,
) -> Table:
    n = len(ranking.ids)
    if not np.any(ranking.values):
        raise ValueError(
            "every statistic is zero, so the statistics do not rank the genes"
        )
    # fgsea's weights: abs(stats)^gseaParam, in ranked order.
    weights = np.ascontiguousarray(np.abs(ranking.values) ** settings.exponent)
    _warn_about_statistics(ranking.values, settings.score_type)
    index = {gene: position for position, gene in enumerate(ranking.ids)}
    smallest = settings.min_size
    largest = n - 1 if settings.max_size is None else min(settings.max_size, n - 1)
    names: list[str] = []
    members: list[IntArray] = []
    matched = 0
    for name, genes in collection.items():
        positions = matched_positions(genes, index)
        matched += positions.size
        if smallest <= positions.size <= largest:
            names.append(name)
            members.append(np.sort(positions))
    if matched == 0:
        raise ValueError(
            "no gene of any set is among the ranked genes; the sets and the "
            "statistics must name genes the same way (for example both by symbol, "
            f"or both by Ensembl id): the sets name genes such as "
            f"{_example_ids(collection)}, the statistics genes such as "
            f"{quoted(ranking.ids[:3])}"
        )
    seeds = _seeds(generator, settings)
    if not names:
        empty = np.zeros(0)
        return _table(
            [], np.zeros(0, dtype=np.int64), empty, empty, empty, empty, empty, []
        )
    sizes = np.array([positions.size for positions in members], dtype=np.int64)
    scores = np.empty(len(names))
    edges = []
    for s, positions in enumerate(members):
        scores[s], edge = _enrichment_score(weights, positions, settings.score_type)
        edges.append(_SEPARATOR.join(ranking.ids[i] for i in edge.tolist()))
    counts = _random_counts(weights, scores, sizes, settings, seeds)
    normalized, p_value, beyond, of_sign = _permutation_results(
        scores, counts, settings.score_type
    )
    n_perm = settings.n_permutations
    log2_error = np.full(len(names), np.nan)
    if not settings.multilevel:
        missing = int(np.count_nonzero(np.isnan(p_value)))
        if missing:
            _warn(
                f"{missing} gene set(s) have no random score of the sign of their "
                "enrichment score, or only zeros, so that their normalized scores and "
                "p-values are NaN; the statistics are mostly of one sign"
            )
        tested = ~np.isnan(p_value)
        log2_error[tested] = _permutation_error(beyond[tested], n_perm)
    else:
        unbalanced = of_sign < _FEWEST_OF_SIGN
        none = of_sign == 0
        few = unbalanced & ~none
        if few.any():
            _warn(
                fitted_message(
                    f"{int(np.count_nonzero(few))} gene set(s) have fewer than "
                    f"{_FEWEST_OF_SIGN} random scores of the sign of their enrichment "
                    f"score among {n_perm} permutations, but some; their normalized "
                    "scores and p-values are NaN. More permutations "
                    f"(n_permutations={10 * n_perm}) may give them enough"
                )
            )
        if none.any():
            _warn(
                fitted_message(
                    f"{int(np.count_nonzero(none))} gene set(s) have no random score of "
                    f"the sign of their enrichment score among {n_perm} permutations: "
                    "random sets of their size almost always score with the other "
                    "sign, because most statistics share one sign, so more "
                    "permutations are unlikely to give them any; their normalized "
                    "scores and p-values are NaN"
                )
            )
        normalized[unbalanced] = np.nan
        p_value[unbalanced] = np.nan
        tested = ~np.isnan(p_value)
        multilevel = tested.copy()
        multilevel[tested] = _needs_multilevel(
            beyond[tested], n_perm, settings.sample_size
        )
        simple = tested & ~multilevel
        log2_error[simple] = _permutation_error(beyond[simple], n_perm)
        if multilevel.any():
            rows = np.flatnonzero(multilevel)
            denominator = (of_sign[rows] + 1.0) / (n_perm + 1.0)
            estimated, good = _multilevel_p_values(
                weights, scores[rows], sizes[rows], denominator, settings, seeds[1]
            )
            p_value[rows] = np.minimum(1.0, estimated / denominator)
            log2_error[rows] = np.where(
                good, _multilevel_error(p_value[rows], settings.sample_size), np.nan
            )
            if not good.all():
                _warn(
                    f"the p-values of {int(np.count_nonzero(~good))} gene set(s) are "
                    "likely too large: fewer than half of the random sets of the "
                    "multilevel method's last level had a positive signed score; "
                    "their log2_error is NaN"
                )
            with np.errstate(invalid="ignore"):
                floored = p_value < settings.eps
            if floored.any():
                p_value[floored] = settings.eps
                log2_error[floored] = np.nan
                _warn(
                    f"the p-values of {int(np.count_nonzero(floored))} gene set(s) "
                    f"are below eps={settings.eps:g} and are reported as eps; eps=0 "
                    "estimates them"
                )
    adjusted = adjust_p_values(p_value, "bh")
    return _table(
        names, sizes, scores, normalized, p_value, adjusted, log2_error, edges
    )


def _enrichment_score(
    weights: FloatArray, positions: IntArray, score_type: int
) -> tuple[float, IntArray]:
    """fgsea's calcGseaStat: the ES of a set and its leading edge, as positions."""
    tops, bottoms = running_sum(weights, positions)
    highest = float(tops.max())
    lowest = float(bottoms.min())
    if score_type == 1 or (score_type == 0 and highest > -lowest):
        score = highest
        edge = positions[: int(np.argmax(tops)) + 1]
    elif score_type == 2 or (score_type == 0 and highest < -lowest):
        score = lowest
        edge = positions[int(np.argmin(bottoms)) :][::-1]
    else:  # the deviations are equal: no direction
        score = 0.0
        edge = positions[:0]
    return score, edge


def _seeds(generator: np.random.Generator, settings: _Settings) -> IntArray:
    """The seeds of the random sets, distinct integers from 1 to 10^9."""
    count = 2 if settings.multilevel else len(_chunks(settings.n_permutations))
    drawn = generator.choice(_SEED_RANGE, size=count, replace=False)
    return np.asarray(drawn, dtype=np.int64) + 1


def _chunks(n_permutations: int) -> list[int]:
    """fgseaSimple's chunks: of max(1000, ceiling(n / 128)) permutations, and the
    rest."""
    size = max(_CHUNK, -(-n_permutations // _MOST_CHUNKS))
    chunks = [size] * (n_permutations // size)
    if n_permutations - sum(chunks) > 0:
        chunks.append(n_permutations - sum(chunks))
    return chunks


class _Counts(NamedTuple):
    """The random scores of each set: below or at its ES, above or at it, at most
    and at least 0, and the sums of those at most and at least 0."""

    below: FloatArray
    above: FloatArray
    at_most_zero: FloatArray
    at_least_zero: FloatArray
    negative_sum: FloatArray
    positive_sum: FloatArray


def _random_counts(
    weights: FloatArray,
    scores: FloatArray,
    sizes: IntArray,
    settings: _Settings,
    seeds: IntArray,
) -> _Counts:
    if settings.multilevel:
        chunks = np.array([settings.n_permutations], dtype=np.int64)
        seeds = seeds[:1]
    else:
        chunks = np.array(_chunks(settings.n_permutations), dtype=np.int64)
    arguments = (weights, scores, sizes, chunks, seeds, settings.score_type)
    by_chunk: FloatArray = by_chunks(
        _counts_by_chunk_parallel, _counts_by_chunk, arguments, chunks.size
    )
    # The counts are whole numbers; the sums of the chunks are rounded once, as R's
    # sum() of fgsea's chunks is.
    totals = [by_chunk[:, column].sum(axis=0) for column in range(4)]
    sums = [
        np.array([accurate_sum(by_chunk[:, column, s])[0] for s in range(sizes.size)])
        for column in (4, 5)
    ]
    return _Counts(*totals, *sums)


def _permutation_results(
    scores: FloatArray, counts: _Counts, score_type: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """fgseaSimpleImpl: the NES, the permutation p-values, the random scores beyond
    each ES on its side, and those of its sign (fgsea's modeFraction)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        negative_mean = counts.negative_sum / counts.at_most_zero
        positive_mean = counts.positive_sum / counts.at_least_zero
    positive = scores > 0.0
    usable_positive = ~np.isnan(positive_mean) & (positive_mean != 0.0)
    usable_negative = ~np.isnan(negative_mean) & (negative_mean != 0.0)
    if score_type == 0:
        normalizable = (positive & usable_positive) | (~positive & usable_negative)
        beyond = np.where(positive, counts.above, counts.below)
        of_sign = np.where(scores >= 0.0, counts.at_least_zero, counts.at_most_zero)
    elif score_type == 1:
        normalizable = (scores >= 0.0) & usable_positive
        beyond = counts.above
        of_sign = counts.at_least_zero
    else:
        normalizable = (scores <= 0.0) & usable_negative
        beyond = counts.below
        of_sign = counts.at_most_zero
    normalized = np.full(scores.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        divisor = np.where(scores > 0.0, positive_mean, np.abs(negative_mean))
        if score_type == 1:
            divisor = positive_mean
        elif score_type == 2:
            divisor = np.abs(negative_mean)
        normalized[normalizable] = scores[normalizable] / divisor[normalizable]
        p_value = np.minimum(
            (1.0 + counts.below) / (1.0 + counts.at_most_zero),
            (1.0 + counts.above) / (1.0 + counts.at_least_zero),
        )
    p_value[~normalizable] = np.nan
    return normalized, p_value, beyond, of_sign


def _permutation_error(beyond: FloatArray, n_permutations: int) -> FloatArray:
    """fgsea's log2err of a permutation p-value."""
    variance = trigamma(beyond + 1.0) - trigamma(float(n_permutations + 1))
    error: FloatArray = np.sqrt(variance) / math.log(2.0)
    return error


def _multilevel_error(p_value: FloatArray, sample_size: int) -> FloatArray:
    """fgsea's multilevelError: the expected standard deviation of log2 p from the
    number of levels."""
    level = trigamma((sample_size + 1) / 2) - trigamma(float(sample_size + 1))
    with np.errstate(divide="ignore"):
        levels = np.floor(-np.log2(p_value) + 1.0)
    error: FloatArray = np.sqrt(levels * level) / math.log(2.0)
    return error


def _needs_multilevel(
    beyond: FloatArray, n_permutations: int, sample_size: int
) -> BoolArray:
    """Whether the multilevel method estimates log2 p more accurately than the
    permutations did (fgseaMultilevel's comparison of multError and simpleError)."""
    with np.errstate(divide="ignore"):
        crude = np.log2((beyond + 1.0) / (n_permutations + 1.0))
        left = np.log2(_beta_quantile(0.025, beyond, n_permutations - beyond + 1.0))
        right = np.log2(_beta_quantile(0.975, beyond + 1.0, n_permutations - beyond))
    simple_error = 0.5 * np.maximum(crude - left, right - crude)
    multiple_error = _multilevel_error(
        (beyond + 1.0) / (n_permutations + 1.0), sample_size
    )
    needed: BoolArray = multiple_error < simple_error
    return needed


def _beta_quantile(q: float, a: FloatArray, b: FloatArray) -> FloatArray:
    """The q-quantile of beta distributions with shapes a and b, by bisection of the
    regularized incomplete beta function; 0 for a = 0 and 1 for b = 0, the point
    masses of the limits."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    low = np.zeros(a.shape)
    high = np.ones(a.shape)
    inside = (a > 0.0) & (b > 0.0)
    for _ in range(1100):  # far beyond the 1074 halvings that reach the smallest double
        middle = 0.5 * (low + high)
        settled = ~inside | (middle == low) | (middle == high)
        if settled.all():
            break
        below = np.asarray(betainc(a, b, middle), dtype=np.float64) < q
        low = np.where(~settled & below, middle, low)
        high = np.where(~settled & ~below, middle, high)
    quantile: FloatArray = np.where(a == 0.0, 0.0, np.where(b == 0.0, 1.0, high))
    return quantile


def _multilevel_p_values(
    weights: FloatArray,
    scores: FloatArray,
    sizes: IntArray,
    denominator: FloatArray,
    settings: _Settings,
    seed: int,
) -> tuple[FloatArray, BoolArray]:
    """fgsea's multilevelImpl: the multilevel estimates of the sets of each size,
    with one seed for every size, and whether each is good (isCpGeHalf)."""
    by_size = np.argsort(sizes, kind="stable")
    group_sizes, starts = np.unique(sizes[by_size], return_index=True)
    starts = np.append(starts, by_size.size).astype(np.int64)
    ordered_scores = np.ascontiguousarray(scores[by_size])
    boundaries = np.array(
        [
            settings.eps * float(denominator[by_size[starts[g] : starts[g + 1]]].min())
            for g in range(group_sizes.size)
        ]
    )
    arguments = (
        weights,
        ordered_scores,
        group_sizes.astype(np.int64),
        starts,
        boundaries,
        settings.sample_size,
        int(seed),
        settings.score_type != 0,
    )
    estimated, good = by_chunks(
        _multilevel_groups_parallel, _multilevel_groups, arguments, group_sizes.size
    )
    p_value = np.empty(scores.size)
    p_value[by_size] = estimated
    is_good = np.empty(scores.size, dtype=bool)
    is_good[by_size] = good
    return p_value, is_good


# The Mersenne Twister and fgsea's random integers.


@jit
def _mt_seeded(seed: int) -> IntArray:
    """The state of std::mt19937 seeded with ``seed``, words in int64."""
    state = np.empty(_MT_WORDS + 1, dtype=np.int64)
    state[0] = seed & 0xFFFFFFFF
    for i in range(1, _MT_WORDS):
        previous = state[i - 1]
        state[i] = (1812433253 * (previous ^ (previous >> 30)) + i) & 0xFFFFFFFF
    state[_MT_WORDS] = _MT_WORDS
    return state


@jit
def _mt_next(state: IntArray) -> int:
    """The next 32-bit output of MT19937."""
    index = state[_MT_WORDS]
    if index >= _MT_WORDS:
        for i in range(_MT_WORDS):
            y = (state[i] & 0x80000000) | (state[(i + 1) % _MT_WORDS] & 0x7FFFFFFF)
            word = state[(i + 397) % _MT_WORDS] ^ (y >> 1)
            if y & 1:
                word ^= 0x9908B0DF
            state[i] = word
        index = 0
    y = state[index]
    state[_MT_WORDS] = index + 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    output: int = y
    return output


@jit
def _uniform_integer(state: IntArray, low: int, high: int) -> int:
    """An integer from low to high, as fgsea's uid_wrapper draws it: by rejecting the
    outputs above the largest multiple of the range, then the remainder."""
    length = high - low + 1
    complete = _LARGEST_DRAW - _LARGEST_DRAW % length
    while True:
        x = _mt_next(state)
        if x < complete:
            drawn: int = low + x % length
            return drawn


@jit
def _random_sample(
    state: IntArray, low: int, high: int, k: int, used: BoolArray, out: IntArray
) -> None:
    """k distinct integers from low to high in random order, as fgsea's
    ``combination`` draws them; ``used`` is all False before and after."""
    n = high - low + 1
    count = 0
    if k < n * 1.0 / 2:
        for _ in range(k):
            for _ in range(_DRAWS_PER_GENE):
                x = _uniform_integer(state, low, high)
                if not used[x - low]:
                    out[count] = x
                    count += 1
                    used[x - low] = True
                    break
    else:
        # Floyd's selection, as fgsea's, then the Fisher-Yates shuffle (Durstenfeld
        # 1964) where fgsea calls std::shuffle.
        for r in range(n - k, n):
            x = _uniform_integer(state, 0, r)
            if not used[x]:
                out[count] = low + x
                used[x] = True
            else:
                out[count] = low + r
                used[r] = True
            count += 1
        for i in range(count - 1, 0, -1):
            j = _uniform_integer(state, 0, i)
            out[i], out[j] = out[j], out[i]
    if count < k:  # 100 draws in a row fell on genes drawn already
        raise RuntimeError("could not draw a random sample of distinct genes")
    for i in range(count):
        used[out[i] - low] = False


# The scores of random samples and of all their first genes (fgsea's
# calcGseaStatCumulative), from prefix sums kept in square-root blocks.


@jit
def _tree_shape(size: int) -> tuple[int, int, int]:
    """The block length, its base-2 logarithm and the number of blocks of fgsea's
    SegmentTree of ``size`` entries."""
    length = 1
    log_length = 0
    while length * length < size:
        length <<= 1
        log_length += 1
    return length, log_length, (size - 1) // length + 1


@jit
def _tree_add(
    within: Any, between: Any, length: int, log_length: int, p: int, delta: Any
) -> None:
    """Adds ``delta`` at entry ``p``: to the sums of its block from ``p`` on, and to
    the sums before every later block."""
    mask = length - 1
    end = p - (p & mask) + mask + 1
    while p < end:
        within[p] += delta
        p += 1
    block = p >> log_length
    while block < between.size:
        between[block] += delta
        block += 1


@jit
def _tree_sum(within: Any, between: Any, log_length: int, r: int) -> Any:
    """The sum of the entries before ``r``."""
    if r == 0:
        return 0
    r -= 1
    return within[r] + between[r >> log_length]


@jit(error_model="numpy")
def _prefix_maxima(
    weights: FloatArray,
    selected: IntArray,
    order: IntArray,
    reverse: bool,
    floor: float,
    out: FloatArray,
) -> None:
    """fgsea's gseaStats1: for each number i of the first genes of ``selected``
    (1-based positions, in the order drawn), the largest deviation of their walk
    above zero, or below zero with ``reverse``.

    The walk at the end of the j-th selected gene in ranked order is the point
    (x_j, y_j) of the genes passed that are not among the first i and of the weight
    of those that are; its deviation is (y_j (N - i) / W - x_j) / (N - i). Adding a
    gene adds (-1, w) to the points after it, which keeps the shape of the upper
    convex hull of the points of each later block, so only the block of the gene is
    rebuilt, and the point of each hull farthest along the current direction moves
    one way as the genes are added.
    """
    n = weights.size
    k = selected.size
    length, log_length, n_blocks = _tree_shape(k + 1)
    x_within = np.zeros(length * length, dtype=np.int64)
    x_between = np.zeros(n_blocks, dtype=np.int64)
    y_within = np.zeros(length * length)
    y_between = np.zeros(n_blocks)
    rank = np.empty(k, dtype=np.int64)
    for i in range(k):
        rank[order[i]] = i
    if not reverse:
        previous = -1
        for i in range(k):
            t = selected[order[i]] - 1
            _tree_add(x_within, x_between, length, log_length, i, t - previous)
            previous = t
        _tree_add(x_within, x_between, length, log_length, k, n - 1 - previous)
    else:
        previous = n
        for i in range(k):
            rank[i] = k - 1 - rank[i]
            t = selected[order[k - 1 - i]] - 1
            _tree_add(x_within, x_between, length, log_length, i, previous - t)
            previous = t
        _tree_add(x_within, x_between, length, log_length, k, previous)

    # The hull of each block of points, as links to the previous point; a point off
    # the hull has no next point (-1).
    to_previous = np.empty(k + 2, dtype=np.int64)
    to_next = np.empty(k + 2, dtype=np.int64)
    block_length = int(math.sqrt(k + 1))
    n_hulls = (k + 1) // block_length + 1
    summit = np.empty(n_hulls, dtype=np.int64)
    first = np.empty(n_hulls, dtype=np.int64)
    last = np.empty(n_hulls, dtype=np.int64)
    i = 0
    while i <= k + 1:
        block = i // block_length
        first[block] = i
        last[block] = min(i + block_length - 1, k + 1)
        j = 1
        while i + j <= last[block]:
            to_previous[i + j] = i + j - 1
            to_next[i + j - 1] = i + j
            j += 1
        to_previous[i] = i
        to_next[last[block]] = last[block]
        summit[block] = last[block]
        i += block_length

    total = 0.0
    for i in range(k):
        t = selected[i] - 1
        position = rank[i]
        weight = weights[t]
        if weight < floor:  # max(|stat|, statEps), raised to gseaParam = 1
            weight = floor
        _tree_add(x_within, x_between, length, log_length, position, -1)
        _tree_add(y_within, y_between, length, log_length, position, weight)
        total += weight
        m = i + 1
        block = (position + 1) // block_length
        start = first[block]
        end = last[block]
        top = max(position, start)
        for c in range(position + 1, end + 1):
            xc = float(_tree_sum(x_within, x_between, log_length, c))
            yc = _tree_sum(y_within, y_between, log_length, c)
            b = top
            xb = float(_tree_sum(x_within, x_between, log_length, b))
            yb = _tree_sum(y_within, y_between, log_length, b)
            while to_previous[top] != top:
                a = to_previous[top]
                xa = float(_tree_sum(x_within, x_between, log_length, a))
                ya = _tree_sum(y_within, y_between, log_length, a)
                turn = (xb - xa) * (yc - yb) - (yb - ya) * (xc - xb)
                if yc - ya < _HULL_TOLERANCE:
                    turn = 0.0
                if turn <= 0.0:
                    break
                top = a
                to_next[b] = -1
                b = a
                xb = xa
                yb = ya
            to_previous[c] = top
            to_next[top] = c
            top = c
            if to_next[c] != -1:
                break
        coefficient = (n - m) / total
        largest = 0.0
        summit[block] = max(top, summit[block])
        for h in range(n_hulls):
            point = summit[h]
            distance = _tree_sum(
                y_within, y_between, log_length, point
            ) * coefficient - _tree_sum(x_within, x_between, log_length, point)
            while True:
                after = to_previous[point]
                farther = _tree_sum(
                    y_within, y_between, log_length, after
                ) * coefficient - _tree_sum(x_within, x_between, log_length, after)
                if farther <= distance:
                    break
                distance = farther
                point = after
            summit[h] = point
            if largest < distance:
                largest = distance
        out[i] = largest / (n - m)


@jit(error_model="numpy")
def _prefix_scores(
    weights: FloatArray, selected: IntArray, score_type: int, out: FloatArray
) -> None:
    """fgsea's calcGseaStatCumulative: the ES of each number of the first genes of
    ``selected``."""
    k = selected.size
    order = np.argsort(selected)
    floor = _STATISTIC_FLOOR
    for i in range(k):
        x = abs(weights[selected[i] - 1])
        if x > 0.0 and x < floor:
            floor = x
    floor /= _FLOOR_DIVISOR
    if score_type == 1:
        _prefix_maxima(weights, selected, order, False, floor, out)
        return
    down = np.empty(k)
    _prefix_maxima(weights, selected, order, True, floor, down)
    if score_type == 2:
        for i in range(k):
            out[i] = -down[i]
        return
    _prefix_maxima(weights, selected, order, False, floor, out)
    for i in range(k):
        if out[i] == down[i]:
            out[i] = 0.0
        elif out[i] < down[i]:
            out[i] = -down[i]


@jit(error_model="numpy")
def _chunk_counts(
    weights: FloatArray,
    scores: FloatArray,
    sizes: IntArray,
    n_iterations: int,
    seed: int,
    score_type: int,
    out: FloatArray,
) -> None:
    """fgsea's calcGseaStatCumulativeBatch: ``out[column, set]`` holds the six
    counts and sums of :class:`_Counts` over ``n_iterations`` random samples."""
    n = weights.size
    k = sizes.max()
    state = _mt_seeded(seed)
    used = np.zeros(n, dtype=np.bool_)
    selected = np.empty(k, dtype=np.int64)
    random_scores = np.empty(k)
    out[:] = 0.0
    for _ in range(n_iterations):
        _random_sample(state, 1, n, k, used, selected)
        _prefix_scores(weights, selected, score_type, random_scores)
        for s in range(sizes.size):
            r = random_scores[sizes[s] - 1]
            if r <= scores[s]:
                out[0, s] += 1.0
            if r >= scores[s]:
                out[1, s] += 1.0
            if r <= 0.0:
                out[2, s] += 1.0
            if r >= 0.0:
                out[3, s] += 1.0
            out[4, s] = out[4, s] + (r if r < 0.0 else 0.0)
            out[5, s] = out[5, s] + (r if r > 0.0 else 0.0)


@jit(parallel=True)
def _counts_by_chunk_parallel(
    weights: FloatArray,
    scores: FloatArray,
    sizes: IntArray,
    chunks: IntArray,
    seeds: IntArray,
    score_type: int,
    n_runs: int,
) -> FloatArray:
    """The counts of each chunk of permutations, ``(chunk, column, set)``, in
    ``n_runs`` runs of chunks on numba's threads."""
    out = np.empty((chunks.size, 6, sizes.size))
    for run in prange(n_runs):
        for c in range(run * chunks.size // n_runs, (run + 1) * chunks.size // n_runs):
            _chunk_counts(
                weights, scores, sizes, chunks[c], seeds[c], score_type, out[c]
            )
    return out


@jit
def _counts_by_chunk(
    weights: FloatArray,
    scores: FloatArray,
    sizes: IntArray,
    chunks: IntArray,
    seeds: IntArray,
    score_type: int,
) -> FloatArray:
    """The counts of :func:`_counts_by_chunk_parallel` on the calling thread."""
    out = np.empty((chunks.size, 6, sizes.size))
    for c in range(chunks.size):
        _chunk_counts(weights, scores, sizes, chunks[c], seeds[c], score_type, out[c])
    return out


# The multilevel method (fgsea's EsRuler).


@jit(error_model="numpy")
def _signed_score(ranks: FloatArray, sample: IntArray) -> float:
    """fgsea's calcES: the deviation of the walk of a sorted sample that is largest in
    absolute value, the first of equal ones."""
    n = ranks.size
    k = sample.size
    total = 0.0
    for position in sample:
        total += ranks[position]
    result = 0.0
    current = 0.0
    q1 = 1.0 / (n - k)
    q2 = 1.0 / total
    last = -1
    for position in sample:
        current -= q1 * (position - last - 1)
        if abs(current) > abs(result):
            result = current
        current += q2 * ranks[position]
        if abs(current) > abs(result):
            result = current
        last = position
    return result


@jit(error_model="numpy")
def _positive_score(ranks: FloatArray, sample: IntArray) -> float:
    """fgsea's calcPositiveES: the largest deviation of the walk above zero."""
    n = ranks.size
    k = sample.size
    total = 0.0
    for position in sample:
        total += ranks[position]
    result = 0.0
    current = 0.0
    q1 = 1.0 / (n - k)
    q2 = 1.0 / total
    last = -1
    for position in sample:
        current += q2 * ranks[position] - q1 * (position - last - 1)
        if result < current:  # std::max, which keeps the first value against NaN
            result = current
        last = position
    return result


@jit
def _appended(values: Any, count: int, value: Any) -> Any:
    """``values`` with ``value`` at ``count``, enlarged when full."""
    if count == values.size:
        larger = np.empty(2 * values.size, dtype=values.dtype)
        larger[:count] = values
        values = larger
    values[count] = value
    return values


@jit(error_model="numpy")
def _next_level(
    ranks: FloatArray,
    samples: IntArray,
    levels: FloatArray,
    positives: IntArray,
    count: int,
) -> tuple[IntArray, FloatArray, IntArray, int]:
    """fgsea's duplicateSamples: records the positive scores of the lower half of the
    samples, the median included, with the number of the samples above them whose
    signed score is positive, and replaces that half by copies of the upper half."""
    n_samples, k = samples.shape
    positive_scores = np.empty(n_samples)
    signed_positive = np.zeros(n_samples, dtype=np.bool_)
    n_positive = 0
    for s in range(n_samples):
        positive_scores[s] = _positive_score(ranks, samples[s])
        if _signed_score(ranks, samples[s]) > 0.0:
            n_positive += 1
            signed_positive[s] = True
    # Sorted by score, ties by sample, as std::sort sorts pairs.
    order = np.argsort(positive_scores, kind="mergesort")
    i = 0
    while 2 * i < n_samples:
        levels = _appended(levels, count, positive_scores[order[i]])
        if signed_positive[order[i]]:
            n_positive -= 1
        positives = _appended(positives, count, n_positive)
        count += 1
        i += 1
    copies = np.empty((n_samples, k), dtype=np.int64)
    row = 0
    i = 0
    while 2 * i < n_samples - 2:
        for _ in range(2):
            copies[row] = samples[order[n_samples - 1 - i]]
            row += 1
        i += 1
    copies[row] = samples[order[n_samples >> 1]]
    return copies, levels, positives, count


@jit
def _lower_bound(values: Any, start: int, end: int, value: Any) -> int:
    """The first index from ``start`` before ``end`` whose value is not below
    ``value``, as std::lower_bound finds it."""
    count = end - start
    while count > 0:
        step = count // 2
        middle = start + step
        if values[middle] < value:
            start = middle + 1
            count -= step + 1
        else:
            count = step
    return start


@jit
def _upper_bound(values: IntArray, value: int) -> int:
    """The first index whose value is above ``value``, as std::upper_bound finds
    it."""
    start = 0
    count = values.size
    while count > 0:
        step = count // 2
        middle = start + step
        if not value < values[middle]:
            start = middle + 1
            count -= step + 1
        else:
            count = step
    return start


@jit
def _spread_chunks(
    slots: IntArray, starts: IntArray, sizes: IntArray, spare: int
) -> None:
    """Moves the chunks of a sample apart so that each has ``spare`` free slots after
    its genes, the last one all that remain."""
    n_chunks = sizes.size
    genes = np.empty(sizes.sum(), dtype=np.int64)
    count = 0
    for i in range(n_chunks):
        for j in range(starts[i], starts[i] + sizes[i]):
            genes[count] = slots[j]
            count += 1
    count = 0
    for i in range(n_chunks):
        starts[i + 1] = starts[i] + sizes[i] + spare
        for j in range(sizes[i]):
            slots[starts[i] + j] = genes[count]
            count += 1
    starts[n_chunks] = slots.size


@jit(error_model="numpy")
def _perturb(
    ranks: FloatArray,
    slots: IntArray,
    starts: IntArray,
    sizes: IntArray,
    chunk_sums: FloatArray,
    chunk_ends: IntArray,
    bound: float,
    state: IntArray,
) -> int:
    """fgsea's perturbate: about a tenth as many swaps of one gene of a sample for
    another as it has genes, each kept when the sample's positive score stays above
    ``bound``; returns the swaps kept.

    The sample's genes, sorted, are split into chunks of positions, as fgsea splits
    them: chunk i holds ``sizes[i]`` genes from slot ``starts[i]`` of ``slots`` on,
    with free slots before the next chunk; ``chunk_sums`` holds the sum of the ranks
    of each chunk and ``chunk_ends`` the position after its last possible one. A
    chunk whose whole sum cannot lift the walk above the bound is passed in one
    step, and the first gene above the bound is remembered, so that later swaps that
    keep it above need no walk. The walk of a proposed sample passes over the gene
    that leaves and takes the gene that enters in its place, so that the sample is
    changed only when the swap is kept; the sums are updated, and restored, in
    fgsea's order of operations.
    """
    n = ranks.size
    n_chunks = chunk_sums.size
    k = 0
    total = 0.0
    for i in range(n_chunks):
        k += sizes[i]
        for j in range(starts[i], starts[i] + sizes[i]):
            total += ranks[slots[j]]
    q1 = 1.0 / (n - k)
    iterations = max(1, int(k * _SWAP_SHARE))
    moves = 0
    candidate = -1
    has_candidate = False
    candidate_x = 0
    candidate_y = 0.0
    for _ in range(iterations):
        old_index = _uniform_integer(state, 0, k - 1)
        old_chunk = 0
        while sizes[old_chunk] <= old_index:
            old_index -= sizes[old_chunk]
            old_chunk += 1
        old_slot = starts[old_chunk] + old_index
        old = slots[old_slot]
        new = _uniform_integer(state, 0, n - 1)
        new_chunk = _upper_bound(chunk_ends, new)
        end = starts[new_chunk] + sizes[new_chunk]
        target = _lower_bound(slots, starts[new_chunk], end, new)
        if target < end and slots[target] == new:
            if new == old:
                moves += 1
            continue
        total = total - ranks[old] + ranks[new]
        chunk_sums[old_chunk] -= ranks[old]
        chunk_sums[new_chunk] += ranks[new]
        if has_candidate and old == candidate:
            has_candidate = False
        if has_candidate:
            if old < candidate:
                candidate_x += 1
                candidate_y -= ranks[old]
            if new < candidate:
                candidate_x -= 1
                candidate_y += ranks[new]
        q2 = 1.0 / total
        kept = has_candidate and -q1 * candidate_x + q2 * candidate_y > bound
        if not kept:
            x = 0
            y = 0.0
            last = -1
            for i in range(n_chunks):
                size = sizes[i]
                if i == old_chunk:
                    size -= 1
                if i == new_chunk:
                    size += 1
                if q2 * (y + chunk_sums[i]) - q1 * x < bound:
                    y += chunk_sums[i]
                    x += chunk_ends[i] - last - 1 - size
                    last = chunk_ends[i] - 1
                    continue
                j = starts[i]
                stop = starts[i] + sizes[i]
                pending = i == new_chunk  # the new gene, which enters before target
                while True:
                    if pending and j == target:
                        position = new
                        pending = False
                    elif j < stop:
                        position = slots[j]
                        j += 1
                        if j - 1 == old_slot:
                            continue
                    else:
                        break
                    y += ranks[position]
                    x += position - last - 1
                    if q2 * y - q1 * x > bound:
                        kept = True
                        has_candidate = True
                        candidate_x = x
                        candidate_y = y
                        candidate = position
                        break
                    last = position
                if kept:
                    break
                x += chunk_ends[i] - 1 - last
                last = chunk_ends[i] - 1
        if kept:
            moves += 1
            # The old gene leaves its chunk, and the new one enters its own.
            stop = starts[old_chunk] + sizes[old_chunk] - 1
            for j in range(old_slot, stop):
                slots[j] = slots[j + 1]
            sizes[old_chunk] -= 1
            if new_chunk == old_chunk and old_slot < target:
                target -= 1
            if starts[new_chunk] + sizes[new_chunk] == starts[new_chunk + 1]:
                # The chunk is full: spread the chunks again.
                offset = target - starts[new_chunk]
                spare = (slots.size - (k - 1)) // n_chunks
                _spread_chunks(slots, starts, sizes, spare)
                target = starts[new_chunk] + offset
            for j in range(starts[new_chunk] + sizes[new_chunk], target, -1):
                slots[j] = slots[j - 1]
            slots[target] = new
            sizes[new_chunk] += 1
        else:
            total = total - ranks[new] + ranks[old]
            chunk_sums[old_chunk] += ranks[old]
            chunk_sums[new_chunk] -= ranks[new]
            if has_candidate and new == candidate:
                has_candidate = False
            if has_candidate:
                if old < candidate:
                    candidate_x -= 1
                    candidate_y += ranks[old]
                if new < candidate:
                    candidate_x += 1
                    candidate_y -= ranks[new]
    return moves


@jit(error_model="numpy")
def _ruler(
    ranks: FloatArray, sample_size: int, k: int, score: float, seed: int, eps: float
) -> tuple[FloatArray, IntArray]:
    """fgsea's EsRuler::extend: the positive scores recorded at each level until
    their median reaches ``score``, and the counts of positive signed scores."""
    n = ranks.size
    state = _mt_seeded(seed)
    used = np.zeros(n, dtype=np.bool_)
    samples = np.empty((sample_size, k), dtype=np.int64)
    for s in range(sample_size):
        _random_sample(state, 0, n - 1, k, used, samples[s])
        samples[s].sort()
    n_chunks = max(1, int(math.sqrt(k)))
    chunk_ends = np.empty(n_chunks, dtype=np.int64)
    chunk_ends[n_chunks - 1] = n
    levels = np.empty(64)
    positives = np.empty(64, dtype=np.int64)
    samples, levels, positives, count = _next_level(
        ranks, samples, levels, positives, 0
    )
    # Each chunk holds its genes and as many free slots as genes on average.
    spare = (k + n_chunks - 1) // n_chunks + 4
    slots = np.empty((sample_size, k + spare * n_chunks), dtype=np.int64)
    starts = np.empty((sample_size, n_chunks + 1), dtype=np.int64)
    sizes = np.empty((sample_size, n_chunks), dtype=np.int64)
    chunk_sums = np.empty((sample_size, n_chunks))
    column = np.empty(sample_size, dtype=np.int64)
    half = (sample_size + 1) // 2
    while levels[count - 1] <= score - _LEVEL_TOLERANCE:
        position = 0
        for i in range(n_chunks - 1):
            position += (k + i) // n_chunks
            for s in range(sample_size):
                column[s] = samples[s, position]
            chunk_ends[i] = np.sort(column)[sample_size // 2]
        for s in range(sample_size):
            chunk_sums[s, :] = 0.0
            sizes[s, :] = 0
            chunk = 0
            for j in range(k):
                position = samples[s, j]
                while chunk_ends[chunk] <= position:
                    chunk += 1
                sizes[s, chunk] += 1
                chunk_sums[s, chunk] += ranks[position]
            starts[s, 0] = 0
            for i in range(n_chunks):
                starts[s, i + 1] = starts[s, i] + sizes[s, i]
            slots[s, :k] = samples[s]
            _spread_chunks(slots[s], starts[s], sizes[s], spare)
        moves = 0
        while moves < sample_size * k:
            for s in range(sample_size):
                moves += _perturb(
                    ranks,
                    slots[s],
                    starts[s],
                    sizes[s],
                    chunk_sums[s],
                    chunk_ends,
                    levels[count - 1],
                    state,
                )
        for s in range(sample_size):
            count_in = 0
            for i in range(n_chunks):
                for j in range(starts[s, i], starts[s, i] + sizes[s, i]):
                    samples[s, count_in] = slots[s, j]
                    count_in += 1
        previous = levels[count - 1]
        samples, levels, positives, count = _next_level(
            ranks, samples, levels, positives, count
        )
        if levels[count - 1] <= previous:
            break  # the median did not rise
        if eps != 0.0 and count // half > -math.log2(0.5 * eps):
            break
    return levels[:count].copy(), positives[:count].copy()


@jit
def _beta_mean_log(a: float, b: float) -> float:
    """fgsea's betaMeanLog: E log X for X ~ Beta(a, b + 1 - a), psi(a) - psi(b + 1)."""
    return digamma_scalar(a) - digamma_scalar(b + 1.0)


@jit
def _ruler_p_value(
    levels: FloatArray,
    positives: IntArray,
    sample_size: int,
    score: float,
    one_sided: bool,
) -> tuple[float, bool]:
    """fgsea's EsRuler::getPvalue: the probability of a positive score of at least
    ``score``, and of a positive signed one unless ``one_sided``; and whether the
    estimate is good."""
    half = (sample_size + 1) // 2
    size = levels.size
    good = True
    if score >= levels[size - 1]:
        index = size - 1
        if score > levels[size - 1] + _LEVEL_TOLERANCE:
            good = False  # beyond the last level
    else:
        index = _lower_bound(levels, 0, size, score)
    full_levels = index // half
    remainder = sample_size - index % half
    log_level = _beta_mean_log(float(half), float(sample_size))
    log_p = full_levels * log_level + _beta_mean_log(
        float(remainder + 1), float(sample_size)
    )
    if not one_sided:
        correction = _beta_mean_log(float(positives[index] + 1), float(remainder))
        log_p += correction
        good = good and math.exp(correction) >= 0.5
    return max(0.0, min(1.0, math.exp(log_p))), good


@jit(error_model="numpy")
def _multilevel_group(
    weights: FloatArray,
    scores: FloatArray,
    k: int,
    eps: float,
    sample_size: int,
    seed: int,
    one_sided: bool,
    p_out: FloatArray,
    good_out: BoolArray,
) -> None:
    """fgsea's fgseaMultilevelCpp for the sets of one size."""
    largest = scores.max()
    smallest = scores.min()
    up_levels = np.empty(0)
    up_positives = np.empty(0, dtype=np.int64)
    down_levels = np.empty(0)
    down_positives = np.empty(0, dtype=np.int64)
    if largest >= 0.0:
        up_levels, up_positives = _ruler(
            weights, sample_size, k, abs(largest), seed, eps
        )
    if smallest < 0.0:
        down_levels, down_positives = _ruler(
            weights[::-1].copy(), sample_size, k, abs(smallest), seed, eps
        )
    for i in range(scores.size):
        if scores[i] >= 0.0:
            p_out[i], good_out[i] = _ruler_p_value(
                up_levels, up_positives, sample_size, abs(scores[i]), one_sided
            )
        else:
            p_out[i], good_out[i] = _ruler_p_value(
                down_levels, down_positives, sample_size, abs(scores[i]), one_sided
            )


@jit(parallel=True)
def _multilevel_groups_parallel(
    weights: FloatArray,
    scores: FloatArray,
    group_sizes: IntArray,
    starts: IntArray,
    boundaries: FloatArray,
    sample_size: int,
    seed: int,
    one_sided: bool,
    n_runs: int,
) -> tuple[FloatArray, BoolArray]:
    """The multilevel estimates of the groups of sets of each size, in ``n_runs``
    runs of groups on numba's threads."""
    p_value = np.empty(scores.size)
    good = np.empty(scores.size, dtype=np.bool_)
    n_groups = group_sizes.size
    for run in prange(n_runs):
        for g in range(run * n_groups // n_runs, (run + 1) * n_groups // n_runs):
            _multilevel_group(
                weights,
                scores[starts[g] : starts[g + 1]],
                group_sizes[g],
                boundaries[g],
                sample_size,
                seed,
                one_sided,
                p_value[starts[g] : starts[g + 1]],
                good[starts[g] : starts[g + 1]],
            )
    return p_value, good


@jit
def _multilevel_groups(
    weights: FloatArray,
    scores: FloatArray,
    group_sizes: IntArray,
    starts: IntArray,
    boundaries: FloatArray,
    sample_size: int,
    seed: int,
    one_sided: bool,
) -> tuple[FloatArray, BoolArray]:
    """The estimates of :func:`_multilevel_groups_parallel` on the calling thread."""
    p_value = np.empty(scores.size)
    good = np.empty(scores.size, dtype=np.bool_)
    for g in range(group_sizes.size):
        _multilevel_group(
            weights,
            scores[starts[g] : starts[g + 1]],
            group_sizes[g],
            boundaries[g],
            sample_size,
            seed,
            one_sided,
            p_value[starts[g] : starts[g + 1]],
            good[starts[g] : starts[g + 1]],
        )
    return p_value, good


# Arguments, warnings and the table.


def _settings(
    min_size: Any,
    max_size: Any,
    exponent: Any,
    score_type: Any,
    method: Any,
    n_permutations: Any,
    sample_size: Any,
    eps: Any,
) -> _Settings:
    for value, name in (
        (min_size, "min_size"),
        (max_size, "max_size"),
        (exponent, "exponent"),
        (score_type, "score_type"),
        (method, "method"),
        (n_permutations, "n_permutations"),
        (sample_size, "sample_size"),
        (eps, "eps"),
    ):
        reject_masked(value, name)
    smallest = positive_integer(min_size, "min_size")
    largest = None
    if max_size is not None:
        largest = positive_integer(max_size, "max_size")
        if largest < smallest:
            raise ValueError(
                f"max_size ({largest}) is smaller than min_size ({smallest}), so no "
                "set could be tested"
            )
    power = number_at_least(exponent, "exponent", 0.0)
    kind = _SCORE_TYPES.index(checked_choice(score_type, "score_type", _SCORE_TYPES))
    how = checked_choice(method, "method", _METHODS)
    count = positive_integer(n_permutations, "n_permutations")
    samples = positive_integer(sample_size, "sample_size")
    if samples < 3:
        raise ValueError(f"sample_size must be at least 3, got {samples}")
    boundary = real_number(eps, "eps")
    if not 0.0 <= boundary < 1.0:
        raise ValueError(f"eps must be in [0, 1), got {boundary}")
    return _Settings(
        smallest,
        largest,
        power,
        kind,
        how == "multilevel",
        count,
        samples + 1 if samples % 2 == 0 else samples,
        boundary,
    )


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, skip_file_prefixes=(PACKAGE_DIRECTORY,))


def _warn_about_statistics(values: FloatArray, score_type: int) -> None:
    """fgsea's warnings about the statistics: ties, and all positive with "std"."""
    nonzero = values[values != 0.0]
    ties = nonzero.size - np.unique(nonzero).size
    if ties:
        _warn(
            f"{ties} of the {values.size} statistics ({100 * ties / values.size:.2f}%) "
            "tie with another one; genes with equal statistics keep the order in "
            "which they are given, which decides the scores of sets that hold some "
            "of them"
        )
    if score_type == 0 and bool(np.all(values > 0.0)):
        _warn(
            "all statistics are positive and score_type is 'std'; for statistics "
            "that are all positive, score_type='pos' tests for enrichment at the top"
        )


def _example_ids(collection: GeneSets) -> str:
    examples: list[str] = []
    for genes in collection.values():
        for gene in genes:
            if gene not in examples:
                examples.append(gene)
            if len(examples) == 3:
                return quoted(examples)
    return quoted(examples)


def _table(
    names: list[str],
    sizes: IntArray,
    scores: FloatArray,
    normalized: FloatArray,
    p_value: FloatArray,
    adjusted: FloatArray,
    log2_error: FloatArray,
    edges: list[str],
) -> Table:
    values = (
        text_column(names),
        sizes.astype(np.int64),
        scores,
        normalized,
        p_value,
        adjusted,
        log2_error,
        text_column(edges),
    )
    return Table(dict(zip(_COLUMNS, values, strict=True)))
