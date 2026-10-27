"""The results of the studies of a collection for one group and contrast, as arrays,
and the factor and levels that a contrast compares in the studies' designs.

SumRank and held-out prediction read, from every study's differential expression of
one group and contrast, a signed statistic of every gene of the collection, with the
log2 fold changes and standard errors of a meta-analysis: one row per study and one
column per gene, NaN where the study did not test the gene. They are read from
:meth:`StudyResults.table <cytepool.StudyResults.table>`, which checks that each result
analysed its study's aligned cube and says why a study has no result for a gene. The
signed statistic is -log10 of the p-value, negative where the log2 fold change is
negative, as Nakatsuka et al. (2025) rank DESeq2's results. Permutations of SumRank
permute the factor that the contrast compares, and held-out prediction tells its cases
from its controls by the two levels it compares, when the studies' designs say so.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import reject_masked
from .design import Design
from .study_results import StudyResults

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
    studies: StudyResults, group: str, contrast: str
) -> StudyStatistics:
    """Every study's signed statistics, log2 fold changes and standard errors of one
    group and contrast, with which studies analysed them and why the others did not.

    Raises:
        TypeError, ValueError, KeyError: As :meth:`StudyResults.table` raises them,
            and ValueError if ``contrast`` is an F-test, which has no direction.
    """
    collection = studies.collection
    table = studies.table(group, contrast)
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


def study_results(value: Any, function: str) -> StudyResults:
    """The studies' results that a function combining studies takes, checked.

    Raises:
        TypeError: If ``value`` is not a ``StudyResults``.
    """
    if not isinstance(value, StudyResults):
        reject_masked(value, "studies")
        raise TypeError(
            f"{function} takes the studies' results with their collection, a "
            "StudyResults such as analyse_each_study gives, or "
            f"StudyResults(collection, results); got {type(value).__name__}"
        )
    return value


def contrast_levels(
    studies: StudyResults, contrast: str, names: Sequence[str]
) -> tuple[str, str, str] | None:
    """The factor and the two levels that a contrast compares in the designs of the
    studies named, as ``(factor, case, control)``; None unless it compares the same
    two levels of one factor in every one of them.

    A contrast compares two levels of a factor when its weights fall on the columns of
    that factor alone, which enters the design as a main effect only: one column with
    a positive weight, a level against the reference level, which has no column of its
    own (``conditioncase`` in ``~ condition``), or two columns with weights of equal
    size and opposite sign, one level minus another (``conditionB - conditionA`` in
    ``~ 0 + condition``). A negative weight on one column compares the reference
    against the level. A study's result without its design (built by hand) gives
    None.
    """
    found = set()
    for name in names:
        result = studies[name]
        design = result.design
        if design is None or contrast not in result.contrasts.names:
            return None
        compared = _compared_levels(design, result.contrasts[contrast])
        if compared is None:
            return None
        found.add(compared)
    if len(found) != 1:
        return None
    return found.pop()


def _compared_levels(
    design: Design, weights: FloatArray
) -> tuple[str, str, str] | None:
    """The factor, case level and control level that contrast weights over a design's
    columns compare, or None; see :func:`contrast_levels`."""
    coding = design.model_matrix.coding
    nonzero = [
        j for j, weight in enumerate(np.asarray(weights).tolist()) if weight != 0
    ]
    if not nonzero or len(nonzero) > 2:
        return None
    factors = set()
    for j in nonzero:
        parts = coding[j]
        if len(parts) != 1 or parts[0][1] is None:
            return None  # the intercept, a covariate or an interaction
        factors.add(parts[0][0])
    if len(factors) != 1:
        return None
    factor = factors.pop()
    if any(len(parts) > 1 and any(v == factor for v, _ in parts) for parts in coding):
        return None  # the factor interacts with another variable
    levels = design.levels[factor]
    own = {parts[0][1] for parts in coding if len(parts) == 1 and parts[0][0] == factor}
    w = [float(weights[j]) for j in nonzero]
    named = [str(coding[j][0][1]) for j in nonzero]
    if len(nonzero) == 1:
        reference = str(levels[0])
        if reference in own:
            return None  # a level's mean, in a design that has a column per level
        return (
            (factor, named[0], reference) if w[0] > 0 else (factor, reference, named[0])
        )
    if w[0] != -w[1]:
        return None
    return (factor, named[0], named[1]) if w[0] > 0 else (factor, named[1], named[0])
