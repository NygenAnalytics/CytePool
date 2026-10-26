"""Average ranks: the mean of the ranks that tied values share.

Values are ranked from 1 for the smallest; values that compare equal (with ``==``,
under which -0.0 equals 0.0 and a NaN equals nothing) share the mean of the ranks they
would take, Kendall's (1945) treatment of ties. The ranks are half-integers, exact in
float64 for fewer than 2^52 values, so they do not depend on the order in which tied
values are visited. Normalization (TMM's trimming), the robust estimation of the
prior and the gene-set and rotation tests rank with it.

References:
    Kendall MG (1945). The treatment of ties in ranking problems. Biometrika 33,
    239-251. doi:10.1093/biomet/33.3.239
"""

import numpy as np
import numpy.typing as npt

from ._jit import jit

FloatArray = npt.NDArray[np.float64]


def average_ranks(values: FloatArray) -> FloatArray:
    """Ranks from 1 for the smallest value, with ties given their average rank.

    The values are ordered by numpy's stable sort, which puts NaN last.
    """
    ranks = np.empty(values.size)
    average_ranks_in_order(values, np.argsort(values, kind="stable"), ranks)
    return ranks


@jit
def average_ranks_in_order(
    values: FloatArray, order: npt.NDArray[np.int64], ranks: FloatArray
) -> None:
    """Fill ``ranks`` with the average ranks of the values that ``order`` sorts.

    ``order`` holds the positions of the values to rank, in increasing order of
    their values; ranks are written at those positions only.
    """
    n = order.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + 1 + j)  # the mean of ranks i + 1 to j
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
