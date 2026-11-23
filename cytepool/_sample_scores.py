"""Scores of samples, as limma, check_dose_design and dose_response read and fit them.

Pathway scores (:class:`~cytepool.PathwayScores`) and activities of samples
(:class:`~cytepool.Activities`) give one value per sample and gene set or source. Their
values are read with the sample of each row, the names of the scores and the cell type
they were computed in, and fitted to a design by limma's ``lmFit``, ``contrasts.fit``
and ``eBayes`` (Smyth 2004), the steps GSVA's guide takes to test scores (Haenzelmann,
Castelo and Guinney 2013).

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Haenzelmann S, Castelo R, Guinney J (2013). GSVA: gene set variation analysis for
    microarray and RNA-seq data. BMC Bioinformatics 14, 7.
    doi:10.1186/1471-2105-14-7
"""

from collections import Counter
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._analysis_records import design_sample_ids
from ._columns import text_column
from ._messages import quoted
from .activity import Activities
from .cube.pool import BULK_GROUP
from .design import Contrasts, Design
from .pathway_scores import PathwayScores
from .stats.empirical_bayes import ModeratedFit, moderated_statistics
from .stats.linear_models import contrasts_fit, fit_linear_model
from .stats.multiple_testing import adjust_p_values
from .table import Table

FloatArray = npt.NDArray[np.float64]

NO_SAMPLE_IDS = (
    "the design's sample table has no 'sample_id' column, so its rows cannot be "
    "matched to the samples of the scores; build the Design from the cube's samples"
)
"""The message for a design whose samples cannot be matched to those of scores."""


class SampleScores(NamedTuple):
    """The values of scores of samples, with the samples, the scores' names and the
    cell type they were computed in."""

    values: FloatArray  # samples x scores, NaN where missing
    sample_ids: list[str]  # the sample of each row
    id_column: str  # "gene_set" or "source", the column that names the scores
    ids: np.ndarray  # the scores' names, as a text column
    group: str  # the cell type the scores were computed in


def sample_scores(data: PathwayScores | Activities, caller: str) -> SampleScores:
    """The values of pathway scores, or of activities of samples, for ``caller``.

    Raises:
        ValueError: If the activities are of the statistics of contrasts, whose
            profiles name no sample, the profiles name a sample twice, or a score is
            infinite.
    """
    if isinstance(data, PathwayScores):
        id_column, names = "gene_set", list(data.gene_sets)
    else:
        if "sample_id" not in data.profiles:
            raise ValueError(
                f"{caller} tests scores of samples, and these activities are of the "
                "statistics of contrasts (their profiles are groups and contrasts, "
                "with no sample_id), which are tested already; score the samples of a "
                "cube's group, as ulm(cube, network, group=...) does"
            )
        id_column, names = "source", list(data.sources)
    samples = data.profiles["sample_id"].tolist()
    repeated = sorted(sample for sample, n in Counter(samples).items() if n > 1)
    if repeated:
        raise ValueError(
            f"the profiles of the scores name the sample(s) {quoted(repeated)} more "
            "than once; each profile must be one sample, as gsva and ulm give them"
        )
    values = np.asarray(data.scores, dtype=np.float64)
    infinite = np.argwhere(np.isinf(values))
    if infinite.size:
        row, column = infinite[0]
        raise ValueError(
            f"the scores hold {len(infinite)} infinite value(s), such as the score of "
            f"{names[column]!r} in sample {samples[row]!r}; mark a missing score with "
            "NaN"
        )
    group = data.settings.get("group")
    return SampleScores(
        values,
        samples,
        id_column,
        text_column(names),
        BULK_GROUP if group is None else str(group),
    )


def design_rows(scores: SampleScores, design: Design) -> tuple[list[str], np.ndarray]:
    """The design's samples, in its order, and the row of the scores that holds each,
    -1 for a sample the scores do not hold.

    Raises:
        TypeError, ValueError: If the design's sample ids are missing, not text or
            repeated, or none is among the scores' samples.
    """
    sample_ids = design_sample_ids(
        design, no_ids=NO_SAMPLE_IDS, text_ids="the scores' profiles do"
    )
    row_of = {sample: row for row, sample in enumerate(scores.sample_ids)}
    rows = np.array([row_of.get(sample, -1) for sample in sample_ids], dtype=np.intp)
    if not np.any(rows >= 0):
        raise ValueError(
            "none of the design's samples is among the samples of the scores: the "
            f"scores' samples are such as {quoted(scores.sample_ids[:3])}, the "
            f"design's such as {quoted(sample_ids[:3])}"
        )
    return sample_ids, rows


def fit_scores(
    values: FloatArray,
    x: FloatArray,
    contrasts: Contrasts,
    *,
    weights: Any = None,
    trend: Any = False,
    robust: bool = False,
) -> ModeratedFit:
    """limma's fit of scores: ``lmFit`` of the scores, samples x scores, on a design
    matrix of full rank, whose columns ``contrasts`` names; ``contrasts.fit`` with
    exact standard errors; and ``eBayes`` with ``trend`` and ``robust``.

    A score with missing values is fitted on the samples it has, as ``lmFit`` fits
    it, and its contrasts' standard errors come from its own covariance."""
    fit = fit_linear_model(
        np.ascontiguousarray(values),
        np.ascontiguousarray(x),
        weights=weights,
        coefficient_names=contrasts.columns,
    )
    tested = contrasts_fit(fit, contrasts, method="exact")
    return moderated_statistics(tested, trend=trend, robust=robust)


def contrast_table(
    id_column: str, ids: np.ndarray, moderated: ModeratedFit, column: int
) -> Table:
    """limma's table of one contrast of a fit of scores, as ``topTable`` gives it:
    the contrast, its moderated standard error, the scores' mean, t, the p-values
    and the B-statistic."""
    fit = moderated.fit
    p_value = moderated.p_value[:, column]
    return Table(
        {
            id_column: ids,
            "estimate": fit.coefficients[:, column],
            "standard_error": fit.stdev_unscaled[:, column]
            * np.sqrt(moderated.posterior_variance),
            "average": fit.mean_expression,
            "t": moderated.t[:, column],
            "p_value": p_value,
            "adjusted_p_value": adjust_p_values(p_value, "bh"),
            "log_odds": moderated.log_odds[:, column],
        }
    )
