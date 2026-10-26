"""The results of the studies of a collection for one group and contrast, as arrays,
and the number of studies whose ranks SumRank sums.

SumRank and held-out prediction read, from every study's differential expression of
one group and contrast, a signed statistic of every gene of the collection, with the
log2 fold changes and standard errors of a meta-analysis: one row per study and one
column per gene, NaN where the study did not test the gene. They are read from
:func:`~cytepool.meta_analysis.align_study_results`, which checks that each result
analysed its study's aligned cube and says why a study has no result for a gene. The
signed statistic is -log10 of the p-value, negative where the log2 fold change is
negative, as Nakatsuka et al. (2025) rank DESeq2's results, and a proportion p of k
studies sums floor(p k) ranks, as their code does.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import reject_masked
from ._options import real_number
from .cube import CubeCollection
from .differential_expression import DifferentialExpressionResult
from .meta_analysis import align_study_results

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class StudyStatistics:
    """Every study's results of one group and contrast, ``(n_studies, n_features)``
    in the collection's order.

    Attributes:
        statistics: The signed -log10 p-value of every gene, NaN where the study did
            not test it or its test has no p-value.
        log2_fold_change: The log2 fold changes, NaN likewise.
        standard_error: The standard errors, NaN likewise, or where the results have
            none (dream's).
        measured: Whether each study measured each gene.
        analysed: Whether each study's results test the group and contrast.
        reasons: Why each study does not, as its result says; empty when it does.
        n_tested: The number of genes each study tested.
    """

    statistics: FloatArray
    log2_fold_change: FloatArray
    standard_error: FloatArray
    measured: BoolArray
    analysed: BoolArray
    reasons: list[str]
    n_tested: IntArray


def signed_statistics(
    p_value: npt.ArrayLike, log2_fold_change: npt.ArrayLike
) -> FloatArray:
    """-log10 of the p-values, negative where the log2 fold change is negative, as
    the authors of SumRank sign DESeq2's p-values; infinite for a p-value of 0, and
    NaN where either value is NaN."""
    p = np.asarray(p_value, dtype=np.float64)
    change = np.asarray(log2_fold_change, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = -np.log10(p)
    signed: FloatArray = np.where(change < 0.0, -magnitude, magnitude)
    signed[np.isnan(change)] = np.nan
    return signed


def study_statistics(
    collection: CubeCollection,
    results: Mapping[str, DifferentialExpressionResult],
    group: str,
    contrast: str,
) -> StudyStatistics:
    """Every study's signed statistics, log2 fold changes and standard errors of one
    group and contrast, with which studies analysed them and why the others did not.

    Raises:
        TypeError, ValueError, KeyError: As
            :func:`~cytepool.meta_analysis.align_study_results` raises them, and
            ValueError if ``contrast`` is an F-test, which has no direction.
    """
    table = align_study_results(collection, results, group, contrast)
    shape = (collection.n_studies, collection.n_features)
    if "log2_fold_change" not in table:
        raise ValueError(
            f"{contrast!r} is an F-test, which tests several contrasts at once and "
            "has no direction; SumRank ranks the genes of one contrast, up and down"
        )
    status = table["status"].reshape(shape)
    analysed = ((status == "tested") | (status == "not_tested")).any(axis=1)
    reasons = []
    for row, flag in enumerate(analysed.tolist()):
        if flag:
            reasons.append("")
            continue
        found = table["reason"].reshape(shape)[row]
        # A study that measured some gene says why it has no table; one that measured
        # none of the collection's genes cannot be in a collection.
        reasons.append(
            next(
                (
                    text
                    for text, kind in zip(
                        found.tolist(), status[row].tolist(), strict=True
                    )
                    if kind == "not_analysed"
                ),
                found[0],
            )
        )
    tested = status == "tested"

    def column(name: str) -> FloatArray:
        if name not in table:
            return np.full(shape, np.nan)
        values: FloatArray = np.array(table[name], dtype=np.float64).reshape(shape)
        values[~tested] = np.nan
        return values

    change = column("log2_fold_change")
    statistics = signed_statistics(column("p_value"), change)
    return StudyStatistics(
        statistics=statistics,
        log2_fold_change=change,
        standard_error=column("standard_error"),
        measured=np.array(collection.measured, dtype=bool),
        analysed=analysed,
        reasons=reasons,
        n_tested=tested.sum(axis=1).astype(np.int64),
    )


def studies_summed(proportion: Any, n_studies: int) -> int:
    """The number of ranks SumRank sums for each gene, m = floor(p k), at least 1; the
    product is rounded to 9 decimals first, so that 0.57 of 100 studies is 57.

    Raises:
        TypeError: If ``proportion`` is not a real number, or is a numpy masked array.
        ValueError: If it is not in (0, 1].
    """
    reject_masked(proportion, "proportion_of_studies")
    p = real_number(proportion, "proportion_of_studies", finite=True)
    if not 0.0 < p <= 1.0:
        raise ValueError(
            f"proportion_of_studies must be in (0, 1], such as 1 to sum every study, "
            f"got {p}"
        )
    return max(1, math.floor(round(p * n_studies, 9)))
