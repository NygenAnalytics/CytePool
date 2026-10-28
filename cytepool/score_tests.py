"""Tests of per-sample scores with a design: limma on pathway scores and activities.

:func:`~cytepool.limma` tests a contrast of a design on values of samples: pathway
scores (:func:`~cytepool.gsva` and the others), activities (:func:`~cytepool.ulm` and the
others) or an array, by limma's ``lmFit``, ``contrasts.fit``, ``eBayes`` (Smyth 2004)
and ``topTable``, the steps that GSVA's guide takes to test its scores, and returns a
table with one row per set, source or feature.
https://docs.nygen.io/CytePool/statistics/pathway-scores#testing-scores explains the
tests.

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Haenzelmann S, Castelo R, Guinney J (2013). GSVA: gene set variation analysis for
    microarray and RNA-seq data. BMC Bioinformatics 14, 7.
    doi:10.1186/1471-2105-14-7

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt

from ._columns import text_column
from ._masked_arrays import real_array, reject_masked, unmasked_array
from ._messages import listing, quoted
from .activity import Activities
from .design import Contrasts, Design
from .design._design_matrices import refuse_random_effects
from .design.checks import _aliasing, _contrast_violations
from .pathway_scores import PathwayScores
from .stats._result_tables import feature_id_column
from .stats.empirical_bayes import moderated_statistics
from .stats.linear_models import contrasts_fit, fit_linear_model
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = ["limma"]

FloatArray = npt.NDArray[np.float64]


def limma(
    data: Any,
    design: Design,
    *,
    contrast: str | None = None,
    feature_ids: Any = None,
    sample_ids: Any = None,
    weights: Any = None,
    trend: Any = False,
    robust: bool = False,
) -> Table:
    """Test a contrast of a design on scores of samples, by limma.

    limma's ``lmFit``, ``contrasts.fit``, ``eBayes`` and ``topTable(sort.by =
    "none")`` for one contrast: a linear model of the scores of every gene set,
    source or feature on the design, moderated t-tests of the contrast, and
    Benjamini-Hochberg adjusted p-values over them.
    https://docs.nygen.io/CytePool/statistics/pathway-scores#testing-scores explains the
    tests. Sort the table to read it, as ``limma(scores, design).sort("p_value")``.

    The design's rows are matched to the scores by ``sample_id``, so that the samples a
    score leaves out, such as those with too few cells, leave the fit too: the samples
    fitted are those of the scores that the design has, in the order of the scores. The
    design matrix is the design's rows for them, so every column keeps its meaning;
    when those rows do not determine every column (the samples left out took a donor's
    every sample, say), the fit keeps the columns that span them, as R's ``qr()`` keeps
    them and :func:`~cytepool.limma_voom` fits a group, and the contrast is tested when
    it is estimable from these samples
    (https://docs.nygen.io/CytePool/statistics/linear-models#estimable-contrasts).
    Missing scores (NaN) leave a set out of the fit of the samples they miss, as limma's
    ``lmFit`` does.

    The standard errors of the contrast are exact, as
    :func:`~cytepool.stats.linear_models.contrasts_fit` computes them with
    ``method="exact"``. They are limma's when one design fits every score: no missing
    scores, and no weights or one weight per sample. Otherwise limma's
    ``contrasts.fit`` approximates them where the contrast combines correlated
    coefficients.

    Args:
        data: :class:`~cytepool.PathwayScores`, such as
            :func:`~cytepool.gsva` returns; :class:`~cytepool.Activities` of samples,
            such as :func:`~cytepool.ulm` returns for a cube's group; or an array of
            values, samples x features, NaN where missing, such as log-expression values
            of another platform.
        design: A :class:`~cytepool.Design` whose sample table has a
            ``sample_id`` column, such as the design built for the cube, with named
            contrasts; random effects are refused, as by
            :func:`~cytepool.limma_voom`.
        contrast: The name of one of the design's contrasts; it may be left out
            when the design has one.
        feature_ids: For an array, one id per column, as text; ``feature_0``,
            ``feature_1``, ... by default.
        sample_ids: For an array, one sample id per row, among the design's; by
            default the rows are the design's samples, in order.
        weights: Weights of the fit, as ``lmFit`` takes them: one per sample, in the
            order of the rows of the scores, or one per sample and score; positive
            and finite. None weighs every value alike.
        trend: Whether the prior variance of the moderation follows the mean of the
            scores (``eBayes(trend = TRUE)``), or one covariate value per gene set,
            source or feature, such as ``scores.n_genes``, as GSVA's guide passes the
            sizes of the sets.
        robust: Whether to estimate the prior of the moderation robustly, as
            ``eBayes(robust = TRUE)`` does.

    Returns:
        A table with one row per gene set, source or feature, in the order of the
        data: ``gene_set``, ``source`` or ``feature_id``; ``estimate``, the
        contrast of the scores (``logFC``); ``standard_error``, its moderated
        standard error; ``average``, the mean of the scores over the samples fitted
        (``AveExpr``); ``t``; ``p_value``; ``adjusted_p_value``; and ``log_odds``,
        the B-statistic (``B``).

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the activities are of statistics rather than samples,
            ``feature_ids`` or ``sample_ids`` is given with scores or activities, the
            design has random effects or no ``sample_id`` column, no sample of the
            data is in the design, the design leaves no residual degrees of freedom
            on its samples, the contrast is not estimable from them, or the design
            has several contrasts and none is named.
        KeyError: If the design has no contrast of that name.
    """
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design whose sample table has a sample_id column, got "
            f"{type(design).__name__}"
        )
    refuse_random_effects(design, "limma")
    values, rows, id_column, ids = _scores(data, feature_ids, sample_ids, design)
    name, column = _contrast(design, contrast)
    data_rows, design_rows = _matched(rows, design)
    x = design.matrix[design_rows]
    aliasing = _aliasing(x)
    kept = list(aliasing.kept)
    n_used = len(design_rows)
    if n_used <= len(kept):
        raise ValueError(
            f"no residual degrees of freedom: the design has rank {len(kept)} on the "
            f"{n_used} sample{'' if n_used == 1 else 's'} of the data, so the fit "
            "cannot estimate a variance"
        )
    chosen = Contrasts(
        design.contrasts.matrix[:, [column]], (name,), tuple(design.columns)
    )
    violations = _contrast_violations(
        replace(design.model_matrix, matrix=x), aliasing, chosen
    )
    if violations:
        raise ValueError(
            f"the contrast {name!r} is not estimable from the samples of the data, "
            "which cannot separate the effects it depends on: "
            + listing(violations[0], separator="; ")
        )
    columns = tuple(design.columns[j] for j in kept)
    fit = fit_linear_model(
        np.ascontiguousarray(values[data_rows]),
        np.ascontiguousarray(x[:, kept]),
        weights=_weights(weights, values.shape, data_rows),
        coefficient_names=columns,
    )
    tested = contrasts_fit(
        fit, Contrasts(chosen.matrix[kept], (name,), columns), method="exact"
    )
    moderated = moderated_statistics(tested, trend=trend, robust=robust)
    p_value = moderated.p_value[:, 0]
    return Table(
        {
            id_column: ids,
            "estimate": tested.coefficients[:, 0],
            "standard_error": tested.stdev_unscaled[:, 0]
            * np.sqrt(moderated.posterior_variance),
            "average": fit.mean_expression,
            "t": moderated.t[:, 0],
            "p_value": p_value,
            "adjusted_p_value": adjust_p_values(p_value, "bh"),
            "log_odds": moderated.log_odds[:, 0],
        }
    )


def _scores(
    data: Any, feature_ids: Any, sample_ids: Any, design: Design
) -> tuple[FloatArray, list[str] | None, str, np.ndarray]:
    """The values (samples x scores), the sample id of each row (None: the design's
    rows), the name of the id column and the ids of the scores."""
    if isinstance(data, PathwayScores | Activities):
        for name, value in (("feature_ids", feature_ids), ("sample_ids", sample_ids)):
            if value is not None:
                reject_masked(value, name)
                raise ValueError(
                    f"{name} applies to an array; the {type(data).__name__} name "
                    "their samples and scores"
                )
    if isinstance(data, PathwayScores):
        rows = data.profiles["sample_id"].tolist()
        ids = text_column(list(data.gene_sets))
        return np.asarray(data.scores, dtype=np.float64), rows, "gene_set", ids
    if isinstance(data, Activities):
        if "sample_id" not in data.profiles:
            raise ValueError(
                "limma tests scores of samples, and these activities are of the "
                "statistics of contrasts (their profiles are groups and contrasts, "
                "with no sample_id), which are tested already; score the samples of a "
                "cube's group, as ulm(cube, network, group=...) does"
            )
        rows = data.profiles["sample_id"].tolist()
        ids = text_column(list(data.sources))
        return np.asarray(data.scores, dtype=np.float64), rows, "source", ids
    if not isinstance(data, np.ndarray | list | tuple):
        raise TypeError(
            "data must be PathwayScores, Activities of samples, or an array of values "
            f"(samples x features), got {type(data).__name__}"
        )
    array = unmasked_array(data, "data", missing="NaN")
    values = real_array(array, "data", boolean=False, objects=False)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            "data must be a 2-D array of values with at least one sample and one "
            f"feature, samples x features, got shape {values.shape}"
        )
    if np.isinf(values).any():
        raise ValueError("data holds infinite values; mark missing values with NaN")
    ids = feature_id_column(feature_ids, values.shape[1])
    if sample_ids is None:
        if values.shape[0] != design.n_samples:
            raise ValueError(
                f"data has {values.shape[0]} rows and the design {design.n_samples} "
                "samples; give sample_ids, the sample id of each row, to match them"
            )
        return np.asarray(values, dtype=np.float64), None, "feature_id", ids
    given = unmasked_array(sample_ids, "sample_ids")
    if given.ndim != 1 or given.size != values.shape[0]:
        raise ValueError(
            f"sample_ids must hold one id per row of data ({values.shape[0]}), got "
            f"shape {given.shape}"
        )
    rows = given.tolist()
    if not all(isinstance(row, str) and row for row in rows):
        raise TypeError("sample_ids must hold text ids, as the design's sample_id does")
    if len(set(rows)) != len(rows):
        raise ValueError("sample_ids must be unique: each row is one sample")
    return np.asarray(values, dtype=np.float64), rows, "feature_id", ids


def _contrast(design: Design, contrast: Any) -> tuple[str, int]:
    """The name of the contrast tested and its column among the design's."""
    names = list(design.contrasts.names)
    if contrast is None:
        if len(names) != 1:
            raise ValueError(
                f"the design has {len(names)} contrasts; name one with contrast"
                + (f", one of {quoted(names)}" if names else "")
                + (
                    ""
                    if names
                    else ", after giving the design contrasts, as in "
                    "design.with_contrasts({'stim_vs_ctrl': 'conditionstim'})"
                )
            )
        return names[0], 0
    if not isinstance(contrast, str):
        reject_masked(contrast, "contrast")
        raise TypeError(
            f"contrast must be the name of a contrast of the design, got "
            f"{type(contrast).__name__}"
        )
    if contrast not in names:
        raise KeyError(
            f"the design has no contrast {contrast!r}; its contrasts are: "
            f"{listing(names) or '(none)'}"
        )
    return contrast, names.index(contrast)


def _matched(rows: list[str] | None, design: Design) -> tuple[list[int], list[int]]:
    """The rows of the data and of the design fitted: the data's samples that the
    design has, in the data's order."""
    if rows is None:
        every = list(range(design.n_samples))
        return every, every
    if "sample_id" not in design.samples:
        raise ValueError(
            "the design's sample table has no 'sample_id' column, so its rows cannot "
            "be matched to the samples of the scores; build the Design from the "
            "cube's samples"
        )
    position = {
        value: i for i, value in enumerate(design.samples["sample_id"].tolist())
    }
    data_rows = [i for i, sample in enumerate(rows) if sample in position]
    if not data_rows:
        raise ValueError(
            "none of the samples of the data is in the design: the data's samples "
            f"are such as {quoted(rows[:3])}, the design's such as "
            f"{quoted(list(position)[:3])}"
        )
    return data_rows, [position[rows[i]] for i in data_rows]


def _weights(weights: Any, shape: tuple[int, int], rows: list[int]) -> Any:
    """The weights of the samples fitted: one per sample, or per sample and score."""
    if weights is None:
        return None
    array = real_array(unmasked_array(weights, "weights"), "weights", boolean=False)
    if array.shape not in ((shape[0],), shape):
        raise ValueError(
            f"weights must hold one weight per sample ({shape[0]}) or per sample and "
            f"score {shape}, in the order of the rows of the data, got shape "
            f"{array.shape}"
        )
    return np.ascontiguousarray(array[rows], dtype=np.float64)
