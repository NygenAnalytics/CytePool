"""Test whether each gene's mean rises or falls steadily with dose, by Williams' trend
test as BMDExpress 2 computes it.

NTP's and ETAP's workflows keep a gene for curve fitting when its mean rises or
falls monotonically with dose, by Williams' trend test as BMDExpress 2 runs it (NTP
2018; US EPA 2024; Phillips et al. 2019), whose p-values come from random
relabellings of the samples' doses. :func:`williams_trend_test` computes that test
for every gene of a samples x genes matrix of log-scale values, and
:class:`WilliamsTrendTest` holds each gene's statistic, direction and p-value. How the
test works, how it differs from the Williams-type contrasts of
:func:`~cytepool.dose_response` and how to read it are in
https://docs.nygen.io/CytePool/statistics/dose-response#williams-trend-test.

The statistic:
    With ybar_k the mean of the n_k samples of dose group k, the vehicle k = 0, and s
    the standard deviation pooled within the K dose groups on N - K degrees of
    freedom, isotonic regression (Bartholomew 1959) fits the means of all K groups,
    the vehicle included, by pool-adjacent-violators with weights n_k: once
    non-decreasing, f, and once non-increasing, g. The two sides' statistics are t_up
    = (f_top - f_0) / (s sqrt(1/n_top + 1/n_0)) and t_down = (g_0 - g_top) / (s
    sqrt(1/n_top + 1/n_0)), both at least 0, and the gene's direction is the side of
    the larger, up on a tie; a side whose fit is flat scores 0, and the other side is
    infinite, or very large where rounding leaves a trace of spread, when no group
    has any spread within it. Williams (1971) fits the treated groups alone and
    compares them with the vehicle's observed mean; BMDExpress 2.30.520 fits the
    vehicle with them, as found by comparing its p-values with exact enumeration of
    every relabelling of small designs.

The p-value:
    B random relabellings of the samples' dose groups, drawn once and shared by every
    gene, each give the two statistics. Of the n relabellings whose larger statistic
    lies on the gene's side, b reach its statistic, and the p-value is (b + 1) / (n +
    1), which counts the observed labelling among them, so that it is never 0
    (Phipson and Smyth 2010). About half the relabellings fall on each side, so the
    smallest p-value is about 2 / B; BMDExpress also counts only the relabellings on
    the gene's side. A relabelling reaches the statistic when its own is at least the
    gene's less a relative 1e-9, so that relabellings whose statistics equal the
    gene's mathematically, such as those that move samples between equal groups that
    pool-adjacent-violators pools, count as reaching it whatever the rounding.
    Each gene's values are centred on their mean first, which leaves the statistics
    as they are and keeps the within-group sums of squares accurate where the spread
    is small beside the level.

References:
    Williams DA (1971). A test for differences between treatment means when several
    dose levels are compared with a zero dose control. Biometrics 27, 103.
    doi:10.2307/2528930

    Bartholomew DJ (1959). A test of homogeneity for ordered alternatives. Biometrika
    46, 36-48. doi:10.1093/biomet/46.1-2.36

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero:
    calculating exact p-values when permutations are randomly drawn. Statistical
    Applications in Genetics and Molecular Biology 9, article 39.
    doi:10.2202/1544-6115.1585

    Phillips JR, Svoboda DL, Tandon A, Patel S, Sedykh A, Mav D, Kuo B, Yauk CL, Yang
    L, Thomas RS, Gift JS, Davis JA, Olszyk L, Merrick BA, Paules RS, Parham F, Saddler
    T, Shah RR, Auerbach SS (2019). BMDExpress 2: enhanced transcriptomic
    dose-response analysis workflow. Bioinformatics 35, 1780-1782.
    doi:10.1093/bioinformatics/bty878

    NTP (2018). NTP research report on National Toxicology Program approach to
    genomic dose-response modeling. NTP Research Report 5. National Institute of
    Environmental Health Sciences. doi:10.22427/ntp-rr-5

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496
"""

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._columns import text_column
from cytepool._doses import checked_doses, distinct_doses
from cytepool._jit import by_chunks, jit, thread_limit
from cytepool._masked_arrays import read_only, real_array, reject_masked, unmasked_array
from cytepool._options import positive_integer, recorded_seed
from cytepool.stats._result_tables import feature_id_column
from cytepool.table import Table

__all__ = ["WilliamsTrendTest", "williams_trend_test"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

UP, DOWN = "up", "down"
_UP, _DOWN, _NONE = 1, -1, 0
REACHED = 1e-9
"""A relabelling reaches a gene's statistic when its own is at least the gene's less
this, relative to it."""
_LARGEST_VALUE = 1e100
# Relabellings are drawn and counted in blocks of this many, which bounds the memory
# of the labels at any number of permutations.
_BLOCK = 10_000


@dataclass(frozen=True, eq=False, repr=False)
class WilliamsTrendTest:
    """Williams' trend test of every gene, as BMDExpress 2 computes it.

    Arrays are read-only; genes are in the order of the columns of the values.
    https://docs.nygen.io/CytePool/statistics/dose-response#williams-trend-test says
    how to read them.

    Attributes:
        doses: ``(K,)`` the dose of each dose group, ascending, the vehicle (0) first.
        n_samples: ``(K,)`` the samples of each dose group.
        statistic: ``(n_genes,)`` the larger of the two sides' statistics, t_up and
            t_down; infinite, or very large after rounding, where the groups' means
            differ and no group has any spread within it, NaN where every sample of
            the gene has one value.
        direction: ``(n_genes,)`` ``"up"`` where t_up is at least t_down, otherwise
            ``"down"``; empty where the statistic is NaN.
        p_value: ``(n_genes,)`` the permutation p-value, (b + 1) / (n + 1) for b of
            the n relabellings on the gene's side that reach its statistic; NaN
            where the statistic is.
        isotonic_means: ``(n_genes, K)`` the dose groups' means fitted by isotonic
            regression in the gene's direction, the vehicle first; the means
            themselves where the statistic is NaN.
        n_permutations: B, the relabellings drawn.
        seed: The seed of the relabellings; passing it back as ``rng`` draws the same
            ones.
    """

    doses: FloatArray
    n_samples: IntArray
    statistic: FloatArray
    direction: np.ndarray
    p_value: FloatArray
    isotonic_means: FloatArray
    n_permutations: int
    seed: int

    def table(self, *, feature_ids: Any = None) -> Table:
        """Return the test as a table, one row per gene, in order.

        Columns: ``feature_id``, ``direction``, ``statistic`` and ``p_value``.

        Args:
            feature_ids: One id per gene, as text; by default ``feature_0``,
                ``feature_1``, ...

        Raises:
            ValueError: If ``feature_ids`` does not give one id per gene, or an id is
                empty.
            TypeError: If an id is not text, or ``feature_ids`` is a numpy masked
                array.
        """
        return Table(
            {
                "feature_id": feature_id_column(feature_ids, self.statistic.size),
                "direction": text_column(self.direction.tolist()),
                "statistic": self.statistic,
                "p_value": self.p_value,
            }
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return (_frozen_test, (values,))

    def __repr__(self) -> str:
        n_genes = self.statistic.size
        k = self.doses.size
        below = self.p_value < 0.05
        up = int(np.count_nonzero(below & (self.direction == UP)))
        down = int(np.count_nonzero(below & (self.direction == DOWN)))
        lines = [
            f"WilliamsTrendTest: {n_genes:,} gene{'' if n_genes == 1 else 's'}; "
            f"{k} dose groups (the vehicle and {k - 1} dose{'' if k == 2 else 's'}), "
            f"{int(self.n_samples.sum()):,} samples; {self.n_permutations:,} "
            f"permutations, seed {self.seed}",
            f"  p < 0.05: {up + down:,} genes ({up:,} up, {down:,} down)",
        ]
        undefined = int(np.count_nonzero(np.isnan(self.statistic)))
        if undefined:
            lines.append(
                f"  no test for {undefined:,} gene{'' if undefined == 1 else 's'} "
                "whose samples all have one value"
            )
        return "\n".join(lines)


def _frozen_test(values: dict[str, Any]) -> WilliamsTrendTest:
    """A WilliamsTrendTest of read-only arrays, as williams_trend_test builds it;
    pickled results are rebuilt by it."""
    frozen: dict[str, Any] = {
        name: read_only(value) if isinstance(value, np.ndarray) else value
        for name, value in values.items()
    }
    return WilliamsTrendTest(**frozen)


def williams_trend_test(
    doses: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    n_permutations: int = 10_000,
    rng: Any = None,
    n_threads: int | None = None,
) -> WilliamsTrendTest:
    """Test every gene for a monotone trend with dose, by Williams' trend test as
    BMDExpress 2 computes it.

    The statistic compares the vehicle and the top dose after isotonic regression of
    the dose groups' means, in the direction the data take, and the p-value counts
    the random relabellings of the samples' doses, among those that point the same
    way, whose statistic reaches the gene's. NTP's and ETAP's rule sets in
    :func:`~cytepool.dose_response` run this test on log2(CPM + 1) values; its
    p-values are not adjusted for the number of genes there, as BMDExpress's are not.
    https://docs.nygen.io/CytePool/statistics/dose-response#williams-trend-test
    explains the test and how it differs from the Williams-type contrasts of
    ``dose_response(tests=["williams"])``.

    Args:
        doses: One dose per sample, finite and at least 0, with 0 for the vehicle;
            the dose groups are the distinct doses.
        values: ``(n_samples, n_genes)`` log-scale values, such as log2(CPM + 1),
            finite and at most 1e100 in size.
        n_permutations: B, the random relabellings of the samples' dose groups, at
            least 1; the smallest p-value is about 2 / B.
        rng: A ``numpy.random.Generator``, an integer seed or None (fresh entropy)
            for the relabellings; the seed is recorded in the result.
        n_threads: The most threads to use; None keeps numba's setting. Results do
            not depend on it.

    Returns:
        Each gene's statistic, direction, p-value and isotonic means.

    Raises:
        TypeError: If ``doses`` or ``values`` do not hold real numbers or are numpy
            masked arrays, ``n_permutations`` is not an integer, or ``rng`` is not a
            generator, an integer seed or None.
        ValueError: If ``values`` is not 2-D with one row per dose and a column or
            more, a value or dose is not finite, a value is larger than 1e100 in
            size, a dose is negative, two distinct doses differ by less than 1e-8 of
            the dose, no dose is 0, the doses form one group, no dose group has two
            samples, ``n_permutations`` is below 1, or the seed is negative.

    Example:
        >>> import numpy as np
        >>> doses = np.repeat([0.0, 1.0, 3.0, 10.0], 3)
        >>> rng = np.random.default_rng(1)
        >>> values = np.column_stack(
        ...     [rng.normal(0.0, 0.2, 12) + 0.1 * doses, rng.normal(0.0, 0.2, 12)]
        ... )
        >>> test = williams_trend_test(doses, values, rng=1)
        >>> test.direction.tolist(), bool(test.p_value[0] < 0.01)
        (['up', 'up'], True)
    """
    reject_masked(doses, "doses")
    reject_masked(values, "values")
    x = checked_doses(doses, "doses")
    y = real_array(unmasked_array(values, "values"), "values", boolean=False)
    if y.ndim != 2:
        raise ValueError(
            "values must be a 2-D array of shape (n_samples, n_genes), got shape "
            f"{y.shape}"
        )
    if y.shape[0] != x.size:
        raise ValueError(
            f"values has {y.shape[0]} rows and doses {x.size} entries; rows of values "
            "are samples, one dose each"
        )
    if y.shape[1] == 0:
        raise ValueError("values must have at least one gene (column)")
    if not np.isfinite(y).all():
        raise ValueError("values must be finite; leave out genes with missing values")
    largest = float(np.abs(y).max())
    if largest > _LARGEST_VALUE:
        raise ValueError(
            f"values must be at most {_LARGEST_VALUE:g} in size, as log-scale values "
            f"such as log2 CPM are, got {largest:g}"
        )
    n_perm = positive_integer(n_permutations, "n_permutations")
    seed = recorded_seed(rng)
    levels = distinct_doses(x, "(the doses' unit)")
    if levels[0] != 0.0:
        raise ValueError(
            "Williams' trend test compares the doses with the vehicle, and no dose is "
            "0; give the vehicle samples dose 0"
        )
    if levels.size < 2:
        raise ValueError(
            "Williams' trend test needs a dose group besides the vehicle, and every "
            "sample has dose 0"
        )
    labels = np.searchsorted(levels, x).astype(np.int32)
    sizes = np.bincount(labels, minlength=levels.size).astype(np.int64)
    if x.size - levels.size < 1:
        raise ValueError(
            "Williams' trend test pools the spread within the dose groups, and every "
            "dose group has one sample; give a dose group two samples or more"
        )
    centred = np.ascontiguousarray((y - y.mean(axis=0)).T)
    # A gene of one value gets no test; its centred values are made exactly 0, which
    # rounding of its mean need not give.
    centred[np.ptp(y, axis=0) == 0.0] = 0.0
    total = np.einsum("ij,ij->i", centred, centred)
    with thread_limit(n_threads):
        statistic, side, fitted = _observed(centred, total, labels, sizes)
        on_side, reaching = _counts(
            centred, total, labels, sizes, statistic, side, n_perm, seed
        )
    defined = side != _NONE
    p_value = np.where(defined, (reaching + 1.0) / (on_side + 1.0), np.nan)
    direction = np.where(side == _UP, UP, np.where(side == _DOWN, DOWN, ""))
    return _frozen_test(
        {
            "doses": levels,
            "n_samples": sizes,
            "statistic": statistic,
            "direction": direction.astype(object),
            "p_value": p_value,
            "isotonic_means": fitted + y.mean(axis=0)[:, None],
            "n_permutations": n_perm,
            "seed": seed,
        }
    )


def _observed(
    centred: FloatArray,
    total: FloatArray,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
) -> tuple[FloatArray, npt.NDArray[np.int64], FloatArray]:
    """Each gene's statistic, side (1 up, -1 down, 0 for a gene of one value) and
    isotonic means in that direction, of the centred values."""
    n_genes = centred.shape[0]
    statistic = np.empty(n_genes)
    side = np.empty(n_genes, dtype=np.int64)
    fitted = np.empty((n_genes, sizes.size))
    by_chunks(
        _observed_parallel,
        _observed_serial,
        (centred, total, labels, sizes, statistic, side, fitted),
        n_genes,
    )
    return statistic, side, fitted


def _counts(
    centred: FloatArray,
    total: FloatArray,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    n_permutations: int,
    seed: int,
) -> tuple[IntArray, IntArray]:
    """For each gene, the relabellings on its side and those of them that reach its
    statistic, over ``n_permutations`` relabellings drawn in blocks from ``seed``."""
    n_genes = centred.shape[0]
    on_side = np.zeros(n_genes, dtype=np.int64)
    reaching = np.zeros(n_genes, dtype=np.int64)
    generator = np.random.default_rng(seed)
    drawn = 0
    while drawn < n_permutations:
        m = min(_BLOCK, n_permutations - drawn)
        block = generator.permuted(np.broadcast_to(labels, (m, labels.size)), axis=1)
        by_chunks(
            _counts_parallel,
            _counts_serial,
            (centred, total, block, sizes, statistic, side, on_side, reaching),
            n_genes,
        )
        drawn += m
    return on_side, reaching


@jit(parallel=True)
def _observed_parallel(
    centred: FloatArray,
    total: FloatArray,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    fitted: FloatArray,
    n_runs: int,
) -> None:
    n_genes = centred.shape[0]
    for run in prange(n_runs):
        for g in range(run * n_genes // n_runs, (run + 1) * n_genes // n_runs):
            _observed_gene(
                centred[g], total[g], labels, sizes, statistic, side, fitted, g
            )


@jit
def _observed_serial(
    centred: FloatArray,
    total: FloatArray,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    fitted: FloatArray,
) -> None:
    for g in range(centred.shape[0]):
        _observed_gene(centred[g], total[g], labels, sizes, statistic, side, fitted, g)


@jit(parallel=True)
def _counts_parallel(
    centred: FloatArray,
    total: FloatArray,
    block: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    on_side: IntArray,
    reaching: IntArray,
    n_runs: int,
) -> None:
    n_genes = centred.shape[0]
    for run in prange(n_runs):
        for g in range(run * n_genes // n_runs, (run + 1) * n_genes // n_runs):
            _count_gene(
                centred[g],
                total[g],
                block,
                sizes,
                statistic,
                side,
                on_side,
                reaching,
                g,
            )


@jit
def _counts_serial(
    centred: FloatArray,
    total: FloatArray,
    block: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    on_side: IntArray,
    reaching: IntArray,
) -> None:
    for g in range(centred.shape[0]):
        _count_gene(
            centred[g], total[g], block, sizes, statistic, side, on_side, reaching, g
        )


@jit(error_model="numpy")
def _observed_gene(
    y: FloatArray,
    total: float,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    fitted: FloatArray,
    g: int,
) -> None:
    """The statistic, side and isotonic means of gene ``g``, written to the arrays."""
    k = sizes.size
    sums = np.empty(k)
    block_sums = np.empty(k)
    block_sizes = np.empty(k)
    block_groups = np.empty(k, dtype=np.int64)
    up, down = _side_statistics(
        y, total, labels, sizes, sums, block_sums, block_sizes, block_groups
    )
    if total == 0.0:
        statistic[g] = np.nan
        side[g] = _NONE
        for j in range(k):
            fitted[g, j] = sums[j] / sizes[j]
        return
    sign = 1.0 if up >= down else -1.0
    statistic[g] = up if up >= down else down
    side[g] = _UP if up >= down else _DOWN
    _isotonic_fit(sums, sizes, sign, block_sums, block_sizes, block_groups, fitted[g])


@jit(error_model="numpy")
def _count_gene(
    y: FloatArray,
    total: float,
    block: npt.NDArray[np.int32],
    sizes: IntArray,
    statistic: FloatArray,
    side: npt.NDArray[np.int64],
    on_side: IntArray,
    reaching: IntArray,
    g: int,
) -> None:
    """Add gene ``g``'s relabellings of ``block`` that lie on its side, and those of
    them that reach its statistic, to its counts."""
    if side[g] == _NONE:
        return
    k = sizes.size
    sums = np.empty(k)
    block_sums = np.empty(k)
    block_sizes = np.empty(k)
    block_groups = np.empty(k, dtype=np.int64)
    threshold = statistic[g] * (1.0 - REACHED)
    upward = side[g] == _UP
    n_on_side = 0
    n_reaching = 0
    for b in range(block.shape[0]):
        up, down = _side_statistics(
            y, total, block[b], sizes, sums, block_sums, block_sizes, block_groups
        )
        if (up >= down) == upward:
            n_on_side += 1
            if (up if upward else down) >= threshold:
                n_reaching += 1
    on_side[g] += n_on_side
    reaching[g] += n_reaching


@jit(error_model="numpy")
def _side_statistics(
    y: FloatArray,
    total: float,
    labels: npt.NDArray[np.int32],
    sizes: IntArray,
    sums: FloatArray,
    block_sums: FloatArray,
    block_sizes: FloatArray,
    block_groups: IntArray,
) -> tuple[float, float]:
    """t_up and t_down of centred values ``y`` under ``labels``; ``sums`` receives
    the groups' sums, and the other arrays are work space of one entry per group.
    ``total`` is the sum of squares of ``y``."""
    k = sizes.size
    for j in range(k):
        sums[j] = 0.0
    for i in range(y.size):
        sums[labels[i]] += y[i]
    between = 0.0
    for j in range(k):
        between += sums[j] * sums[j] / sizes[j]
    within = max(total - between, 0.0)
    s = math.sqrt(within / (y.size - k))
    scale = s * math.sqrt(1.0 / sizes[k - 1] + 1.0 / sizes[0])
    up_first, up_last = _isotonic_ends(
        sums, sizes, 1.0, block_sums, block_sizes, block_groups
    )
    down_first, down_last = _isotonic_ends(
        sums, sizes, -1.0, block_sums, block_sizes, block_groups
    )
    # A side whose fit is flat shows no change: 0, also where no group has spread.
    rise = up_last - up_first
    fall = down_first - down_last
    return (rise / scale if rise > 0.0 else 0.0), (fall / scale if fall > 0.0 else 0.0)


@jit
def _pooled_blocks(
    sums: FloatArray,
    sizes: IntArray,
    sign: float,
    block_sums: FloatArray,
    block_sizes: FloatArray,
    block_groups: IntArray,
) -> int:
    """Pool-adjacent-violators on the groups' means sums / sizes, weighted by sizes,
    non-decreasing for sign 1 and non-increasing for sign -1: the blocks' sums of
    sign times the values, sizes and numbers of groups, in order; returns their
    number. Two blocks are pooled while the earlier's mean exceeds the later's,
    compared as products of sums and sizes."""
    m = 0
    for j in range(sizes.size):
        block_sums[m] = sign * sums[j]
        block_sizes[m] = sizes[j]
        block_groups[m] = 1
        m += 1
        while (
            m > 1
            and block_sums[m - 2] * block_sizes[m - 1]
            > block_sums[m - 1] * block_sizes[m - 2]
        ):
            block_sums[m - 2] += block_sums[m - 1]
            block_sizes[m - 2] += block_sizes[m - 1]
            block_groups[m - 2] += block_groups[m - 1]
            m -= 1
    return m


@jit
def _isotonic_ends(
    sums: FloatArray,
    sizes: IntArray,
    sign: float,
    block_sums: FloatArray,
    block_sizes: FloatArray,
    block_groups: IntArray,
) -> tuple[float, float]:
    """The isotonic means of the first and the last group."""
    m = _pooled_blocks(sums, sizes, sign, block_sums, block_sizes, block_groups)
    return (
        sign * block_sums[0] / block_sizes[0],
        sign * block_sums[m - 1] / block_sizes[m - 1],
    )


@jit
def _isotonic_fit(
    sums: FloatArray,
    sizes: IntArray,
    sign: float,
    block_sums: FloatArray,
    block_sizes: FloatArray,
    block_groups: IntArray,
    out: FloatArray,
) -> None:
    """The isotonic means of every group, written to ``out``."""
    m = _pooled_blocks(sums, sizes, sign, block_sums, block_sizes, block_groups)
    j = 0
    for b in range(m):
        value = sign * block_sums[b] / block_sizes[b]
        for _ in range(block_groups[b]):
            out[j] = value
            j += 1
