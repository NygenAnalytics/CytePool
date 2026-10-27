"""Replication of differential expression across studies.

Most genes that one study finds are not found in the next (Nakatsuka et al. 2025), so a
result of several studies says how well they agree next to what they find together.
:func:`~cytepool.replication` reads the differential expression of one group (cell type)
and contrast in every study of a collection, a :class:`~cytepool.StudyResults`, lined up
gene by gene by :meth:`StudyResults.table <cytepool.StudyResults.table>`, and gives for
every gene the p-value combined over the studies that tested it, the partial conjunction
p-value that the effect is found in at least u studies in the same direction (Benjamini
and Heller 2008), how many studies find it and in which direction, and the heterogeneity
of its log2 fold changes; for every pair of studies, pi1 of one study's p-values among
the genes the other finds (Storey and Tibshirani 2003) and their sign agreement; and the
number of genes found in at least k studies.
:func:`~cytepool.plotting.plot_study_agreement` and
:func:`~cytepool.plotting.plot_replicated_genes` draw them.

The genes:
    A study tested a gene when its table of the group and contrast has it; one that
    did not measure it, measured and left it out, or did not analyse the group is left
    out of that gene's numbers, and :meth:`StudyResults.table
    <cytepool.StudyResults.table>` says which. Each study's two-sided p-values
    become one-sided ones by the sign of its log2 fold change (``directions`` of
    :func:`~cytepool.stats.combined_p_values.combine_p_values`), so that the combined
    and partial conjunction p-values are two-sided and effects in opposite directions
    count against each other. Stouffer's method, the default, weighs each study by the
    square root of the number of samples its fit of the group used (Zaykin 2011). A
    study finds a gene when its adjusted p-value (Benjamini and Hochberg within the
    study's group) is below ``alpha``. Both p-values are adjusted by Benjamini and
    Hochberg's procedure over the genes that have them.

Agreement between studies:
    For an ordered pair of studies, the discovery and the replication, the genes the
    discovery finds that the replication tested are compared: pi1 = 1 - pi0 of their
    p-values in the replication study, Storey's estimate of the proportion of them with
    an effect there (:func:`~cytepool.stats.multiple_testing.proportion_of_true_nulls`),
    which counts effects that the replication study is too small to find on its own; the
    share of them whose log2 fold changes have the same sign; and how many it finds too.
    pi1 measures replication in the replication study's p-values, so it is not
    symmetric.

    pi0 is estimated at lambda = 0.5 by default: the share of p-values above 0.5,
    over 0.5 (Storey and Tibshirani 2003). qvalue's default, a smoother of the
    estimates at lambda = 0.05, ..., 0.95 (``null_proportion={}``), reads the
    largest lambdas, where the few dozen p-values of the genes one study finds
    leave too few values, so that it often gives pi1 of 0 or 1 there.

Heterogeneity:
    Cochran's Q, its p-value, I^2 and tau^2 of the log2 fold changes and their
    standard errors, by :func:`~cytepool.stats.meta_analysis.fit_meta_analysis` with
    ``meta_method``, for the genes that two studies or more tested; NaN for the others.

Not done:
    The r-values of Heller, Bogomolov and Benjamini (2014), which claim replication
    in a follow-up study of the genes a primary study selects. Held-out prediction
    of case status from the genes the studies find (Nakatsuka et al. 2025) is
    :func:`cytepool.held_out_prediction`.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Benjamini Y, Heller R (2008). Screening for partial conjunction hypotheses.
    Biometrics 64, 1215-1222. doi:10.1111/j.1541-0420.2007.00984.x

    Storey JD, Tibshirani R (2003). Statistical significance for genomewide studies.
    Proceedings of the National Academy of Sciences 100, 9440-9445.
    doi:10.1073/pnas.1530509100

    Zaykin DV (2011). Optimally weighted Z-test is a powerful method for combining
    probabilities in meta-analysis. Journal of Evolutionary Biology 24, 1836-1841.
    doi:10.1111/j.1420-9101.2011.02297.x

    Heller R, Bogomolov M, Benjamini Y (2014). Deciding whether follow-up studies have
    replicated findings in a preliminary large-scale omics study. Proceedings of the
    National Academy of Sciences 111, 16262-16267. doi:10.1073/pnas.1314814111

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ._analysis_records import software_versions, study_keys
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import fitted_lines, quoted
from ._options import checked_choice, integral_number, real_number
from ._study_statistics import study_results
from .stats.combined_p_values import combine_p_values, partial_conjunction
from .stats.meta_analysis import fit_meta_analysis
from .stats.multiple_testing import adjust_p_values, proportion_of_true_nulls
from .study_results import StudyResults
from .table import Table

__all__ = ["ReplicationResult", "replication"]

FloatArray = npt.NDArray[np.float64]

_METHODS = ("stouffer", "fisher", "tippett", "pearson", "mudholkar_george", "simes")
_CONJUNCTION_METHODS = ("simes", "fisher", "stouffer")
_META_METHODS = ("reml", "dl", "pm", "fixed")
# The columns of the studies' tables that the analysis reads.
_COLUMNS = ("log2_fold_change", "standard_error", "p_value", "adjusted_p_value")
_NULL_PROPORTION_OPTIONS = ("lambdas", "method", "smooth_df", "smooth_log")
# pi0 of pi1 by default: at one lambda, steady on the few dozen p-values of the genes
# a study finds, where qvalue's smoother often gives 0 or 1.
_NULL_PROPORTION_DEFAULT = {"lambdas": 0.5}


class ReplicationResult:
    """The replication of one group and contrast across the studies of a collection.

    Built by :func:`replication`. Immutable.

    ``features`` has a row for every feature of the collection that at least one study
    tested, in the collection's order: ``feature_id`` and, when the collection's
    features have it, ``symbol``; ``n_studies``, the studies that tested it;
    ``n_up`` and ``n_down``, those whose log2 fold change is above and below zero;
    ``n_found``, those that find it (adjusted p-value below ``alpha``), and
    ``n_found_up`` and ``n_found_down`` by direction; ``sign``, 1 or -1, of the
    combined effect, and ``sign_agreement``, the share of the studies that tested the
    gene whose log2 fold change has that sign; ``p_value`` and ``adjusted_p_value``,
    the combined p-value of every study that tested the gene, two-sided; and
    ``replication_p_value`` and ``replication_adjusted_p_value``, the partial
    conjunction p-value that at least ``at_least`` studies have an effect in the
    same direction (NaN where fewer tested the gene); and ``q_statistic``,
    ``q_p_value``, ``i_squared`` and ``tau_squared``, the heterogeneity of its log2
    fold changes (NaN where one study tested it).

    ``study_pairs`` has a row for every ordered pair of the studies that analysed the
    group and contrast: ``discovery`` and ``replication``; ``n_features_found``, the
    features the discovery study finds that the replication study tested;
    ``n_features_found_in_both``, those it finds too; ``sign_agreement``, the share
    of the ``n_features_found`` features whose log2 fold changes have the same sign in
    both; and ``pi1``, 1 - pi0 of the replication study's p-values of those features
    (NaN where there is none).

    ``found_in`` has a row for every k from 1 to the number of studies that analysed
    the group and contrast: ``n_studies``, k; ``n_features``, the features that at
    least k studies find; and ``n_features_same_direction``, those that at least k
    studies find in one direction.

    ``studies`` has a row for every study of the collection: ``study``; ``analysed``,
    whether its result has a table of the group and contrast; ``reason``, why not
    (empty when analysed); ``n_samples``, the samples its fit of the group used (0
    when not analysed); ``weight``, its weight in Stouffer's method (NaN for another
    method, or when not analysed); ``n_features_tested`` and ``n_features_found``, the
    features it tested and finds.

    Raises:
        TypeError: If a table is not a ``Table``, or ``provenance`` is not a mapping
            that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_features", "_study_pairs", "_found_in", "_studies", "_record")

    _features: Table
    _study_pairs: Table
    _found_in: Table
    _studies: Table
    _record: str

    def __init__(
        self,
        features: Table,
        study_pairs: Table,
        found_in: Table,
        studies: Table,
        provenance: Mapping[str, Any],
    ) -> None:
        tables = {
            "features": features,
            "study_pairs": study_pairs,
            "found_in": found_in,
            "studies": studies,
        }
        for name, table in tables.items():
            if not isinstance(table, Table):
                raise TypeError(f"{name} must be a Table, got {type(table).__name__}")
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state: dict[str, Any] = {f"_{name}": table for name, table in tables.items()}
        # JSON holds no arrays: json.dumps raises TypeError for one.
        state["_record"] = json.dumps(provenance, allow_nan=False)
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ReplicationResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ReplicationResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            ReplicationResult,
            (
                self._features,
                self._study_pairs,
                self._found_in,
                self._studies,
                json.loads(self._record),
            ),
        )

    @property
    def features(self) -> Table:
        """Every feature that a study tested, with its combined and replication
        p-values, the studies that find it and its heterogeneity."""
        return self._features

    @property
    def study_pairs(self) -> Table:
        """Every ordered pair of studies, with pi1 and the sign agreement of the genes
        the first finds."""
        return self._study_pairs

    @property
    def found_in(self) -> Table:
        """The genes found in at least k studies, for every k."""
        return self._found_in

    @property
    def studies(self) -> Table:
        """Every study, whether it analysed the group and contrast, its samples and
        weight, and the genes it tested and finds."""
        return self._studies

    @property
    def settings(self) -> dict[str, Any]:
        """The group, contrast and settings, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions; the settings, with
        the group and contrast; the collection's provenance, which identifies each
        study's cube by its fingerprint and gives the alignment; and the provenance of
        every study's differential-expression result, by study name.
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def __repr__(self) -> str:
        settings = self.settings
        analysed = int(np.count_nonzero(self._studies["analysed"]))
        replicated = self._features["replication_adjusted_p_value"]
        with np.errstate(invalid="ignore"):
            n_replicated = int(np.count_nonzero(replicated < settings["alpha"]))
        return fitted_lines(
            f"ReplicationResult of group {settings['group']!r}, contrast "
            f"{settings['contrast']!r}: {analysed:,} of {len(self._studies):,} "
            f"studies, {len(self._features):,} genes; {n_replicated:,} found in at "
            "least "
            f"{settings['at_least']:,} studies in one direction at adjusted p < "
            f"{settings['alpha']:g} ({settings['conjunction_method']})"
        )


def replication(
    studies: StudyResults,
    group: str | None = None,
    contrast: str | None = None,
    *,
    alpha: float = 0.05,
    method: str = "stouffer",
    at_least: int = 2,
    conjunction_method: str = "simes",
    weights: Mapping[str, float] | None = None,
    meta_method: str = "reml",
    null_proportion: Mapping[str, Any] | None = None,
) -> ReplicationResult:
    """How well the studies of a collection replicate each other's differential
    expression, feature by feature and study by study.

    For one group (cell type) and contrast, the studies' results, each the analysis
    of its aligned cube (:meth:`~cytepool.CubeCollection.aligned_cube`), are lined up
    by :meth:`StudyResults.table <cytepool.StudyResults.table>`, and every feature
    gets its combined p-value, its partial conjunction p-value of an effect in at
    least ``at_least`` studies in one direction, the studies that find it, and the
    heterogeneity of its log2 fold changes; every pair of studies its pi1 and sign
    agreement; and every k the features found in at least k studies. The module
    description gives the methods.

    Args:
        studies: The differential-expression result of every study of a
            collection, from :func:`~cytepool.limma_voom`,
            :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom`, as
            :func:`~cytepool.analyse_each_study` gives them.
        group: An aligned group id of the collection; it may be left out when the
            studies analysed one group.
        contrast: A contrast name that at least two studies tested; it may be left
            out when the studies tested one.
        alpha: The adjusted p-value below which a study finds a feature, in (0, 1).
        method: The combination of every study's p-value:
            ``"stouffer"`` (weighted, by default by the square roots of the studies'
            samples), ``"fisher"``, ``"tippett"``, ``"pearson"``,
            ``"mudholkar_george"`` or ``"simes"``
            (:func:`~cytepool.stats.combined_p_values.combine_p_values`).
        at_least: u of the partial conjunction: the fewest studies with an effect in
            one direction, from 1 to the number of studies that analysed the group
            and contrast; 2 by default.
        conjunction_method: ``"simes"``, ``"fisher"`` or ``"stouffer"``
            (:func:`~cytepool.stats.combined_p_values.partial_conjunction`).
        weights: Stouffer's weight of each study, by study name, positive and finite;
            by default the square root of the number of samples its fit of the group
            used. For ``method="stouffer"`` only.
        meta_method: The estimator of tau^2 for the heterogeneity statistics:
            ``"reml"``, ``"dl"``, ``"pm"`` or ``"fixed"``
            (:func:`~cytepool.stats.meta_analysis.fit_meta_analysis`).
        null_proportion: Options of
            :func:`~cytepool.stats.multiple_testing.proportion_of_true_nulls` for pi1:
            ``lambdas``, ``method``, ``smooth_df`` and ``smooth_log``. None, the
            default, is ``{"lambdas": 0.5}``; ``{}`` takes qvalue's defaults (see
            the module description).

    Returns:
        The tables of features, pairs of studies, counts by number of studies and
        studies, with a provenance record.

    Raises:
        TypeError: If ``studies`` is not a ``StudyResults``, an argument has the
            wrong type, or is a numpy masked array.
        ValueError: If a key is left out and the studies analysed several, fewer
            than two studies analysed the group and contrast, a study's table lacks
            a column the analysis reads (as an F-test's does), ``at_least`` exceeds
            the studies that analysed it, an option is out of range or unknown, or
            ``weights`` are given for a method other than Stouffer's or do not give
            every study that analysed it.
        KeyError: If ``group`` is not a group of the collection, or no study tested
            ``contrast``.
    """
    for name, value in (("weights", weights), ("null_proportion", null_proportion)):
        reject_masked(value, name)
    level = real_number(alpha, "alpha", finite=True)
    if not 0.0 < level < 1.0:
        raise ValueError(f"alpha must be in (0, 1), such as 0.05, got {level}")
    combination = checked_choice(method, "method", _METHODS)
    conjunction = checked_choice(
        conjunction_method, "conjunction_method", _CONJUNCTION_METHODS
    )
    estimator = checked_choice(meta_method, "meta_method", _META_METHODS)
    u = integral_number(at_least, "at_least", expected="a whole number of studies")
    pi0_options = _null_proportion_options(null_proportion)
    results = study_results(studies, "replication")
    collection = results.collection
    group, contrast = study_keys(results, group, contrast)
    aligned = results.table(group, contrast)
    names = collection.studies
    n_studies, n_features = collection.n_studies, collection.n_features
    status = np.asarray(aligned["status"].tolist(), dtype=object).reshape(
        n_studies, n_features
    )
    missing = [column for column in _COLUMNS if column not in aligned]
    if missing:
        raise ValueError(
            f"the studies' tables of group {group!r} and contrast {contrast!r} have no "
            f"column {quoted(missing)}; replication reads the log2 fold changes, "
            "standard errors and p-values of a contrast, which an F-test does not have"
        )
    values = {
        column: np.asarray(aligned[column], dtype=np.float64).reshape(
            n_studies, n_features
        )
        for column in _COLUMNS
    }
    # A gene counts as tested where its study's table gives it a p-value and a log2
    # fold change: dream's tables keep genes whose model has no minimum, with NaN.
    tested = (status == "tested") & ~np.isnan(values["p_value"])
    tested &= ~np.isnan(values["log2_fold_change"])
    analysed = (status == "tested").any(axis=1)
    reasons = aligned["reason"].tolist()
    used = [study for study, flag in zip(names, analysed, strict=True) if flag]
    if len(used) < 2:
        raise ValueError(
            f"group {group!r} and contrast {contrast!r} were analysed by "
            f"{len(used)} of the {n_studies} studies; replication needs at least two"
        )
    if not 1 <= u <= len(used):
        raise ValueError(
            f"at_least must be from 1 to the {len(used)} studies that analysed group "
            f"{group!r} and contrast {contrast!r}, got {u}"
        )
    n_samples = np.zeros(n_studies, dtype=np.int64)
    for row, study in enumerate(names):
        if analysed[row]:
            summary = results[study].summary
            position = summary["group_id"].tolist().index(group)
            n_samples[row] = int(summary["n_samples_used"][position])
    study_weights = _study_weights(weights, combination, names, analysed, n_samples)

    genes = tested.any(axis=0)
    rows = tested[:, genes]
    lfc = np.where(rows, values["log2_fold_change"][:, genes], np.nan)
    se = np.where(rows, values["standard_error"][:, genes], np.nan)
    p = np.where(rows, values["p_value"][:, genes], np.nan)
    found = rows & (values["adjusted_p_value"][:, genes] < level)
    directions = np.where(rows, lfc, np.nan)
    combined = combine_p_values(
        p,
        method=combination,
        weights=None
        if study_weights is None
        else np.where(rows, study_weights[:, np.newaxis], np.nan),
        directions=directions,
    )
    replicated = partial_conjunction(p, u, method=conjunction, directions=directions)
    heterogeneity = fit_meta_analysis(
        lfc,
        se,
        method=estimator,  # type: ignore[arg-type]
    )
    several = combined.n_studies >= 2
    assert combined.sign is not None
    direction = np.asarray(combined.sign)
    up, down = rows & (lfc > 0.0), rows & (lfc < 0.0)
    agreeing = np.where(direction > 0.0, up.sum(axis=0), down.sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        sign_agreement = np.where(
            direction != 0.0, agreeing / combined.n_studies, np.nan
        )
    kept = np.flatnonzero(genes).tolist()
    ids = collection.features["feature_id"].tolist()
    columns: dict[str, Any] = {"feature_id": text_column([ids[i] for i in kept])}
    if "symbol" in collection.features:
        symbols = collection.features["symbol"].tolist()
        # A text column whose missing symbols are None.
        columns["symbol"] = np.array([symbols[i] for i in kept], dtype=object)
    found_up, found_down = found & up, found & down
    columns |= {
        "n_studies": np.asarray(combined.n_studies, dtype=np.int64),
        "n_up": up.sum(axis=0).astype(np.int64),
        "n_down": down.sum(axis=0).astype(np.int64),
        "n_found": found.sum(axis=0).astype(np.int64),
        "n_found_up": found_up.sum(axis=0).astype(np.int64),
        "n_found_down": found_down.sum(axis=0).astype(np.int64),
        "sign": direction,
        "sign_agreement": sign_agreement,
        "p_value": np.asarray(combined.p_value),
        "adjusted_p_value": adjust_p_values(combined.p_value, "bh"),
        "replication_p_value": np.asarray(replicated.p_value),
        "replication_adjusted_p_value": adjust_p_values(replicated.p_value, "bh"),
        "q_statistic": np.where(several, heterogeneity.q_statistic, np.nan),
        "q_p_value": np.where(several, heterogeneity.q_p_value, np.nan),
        "i_squared": np.where(several, heterogeneity.i_squared, np.nan),
        "tau_squared": np.where(several, heterogeneity.tau_squared, np.nan),
    }
    feature_table = Table(columns)
    pairs = _study_pairs(names, analysed, rows, found, lfc, p, pi0_options)
    k = np.arange(1, len(used) + 1)
    same_direction = np.maximum(found_up.sum(axis=0), found_down.sum(axis=0))
    found_in = Table(
        {
            "n_studies": k.astype(np.int64),
            "n_features": np.array(
                [int(np.count_nonzero(found.sum(axis=0) >= j)) for j in k], np.int64
            ),
            "n_features_same_direction": np.array(
                [int(np.count_nonzero(same_direction >= j)) for j in k], np.int64
            ),
        }
    )
    study_table = Table(
        {
            "study": text_column(list(names)),
            "analysed": analysed.copy(),
            "reason": text_column(
                [
                    "" if analysed[row] else _absence(reasons, status, row, n_features)
                    for row in range(n_studies)
                ]
            ),
            "n_samples": n_samples,
            "weight": np.full(n_studies, np.nan)
            if study_weights is None
            else study_weights,
            "n_features_tested": tested.sum(axis=1).astype(np.int64),
            "n_features_found": found.sum(axis=1).astype(np.int64),
        }
    )
    if study_weights is None:
        weight_setting: Any = None
    elif weights is None:
        weight_setting = "square roots of the samples"
    else:
        weight_setting = {
            study: float(weight)
            for study, weight, flag in zip(names, study_weights, analysed, strict=True)
            if flag
        }
    settings = {
        "group": group,
        "contrast": contrast,
        "alpha": level,
        "method": combination,
        "weights": weight_setting,
        "at_least": u,
        "conjunction_method": conjunction,
        "meta_method": estimator,
        "null_proportion": pi0_options,
        "p_value_adjustment": "BH",
    }
    return ReplicationResult(
        feature_table,
        pairs,
        found_in,
        study_table,
        {
            "analysis": "replication",
            "versions": software_versions(),
            "settings": settings,
            "collection": collection.provenance,
            "studies": {study: results[study].provenance for study in names},
        },
    )


def _null_proportion_options(value: Any) -> dict[str, Any]:
    if value is None:
        return dict(_NULL_PROPORTION_DEFAULT)
    if not isinstance(value, Mapping):
        raise TypeError(
            "null_proportion must be a mapping of options of proportion_of_true_nulls, "
            f"such as {{'method': 'bootstrap'}}, got {type(value).__name__}"
        )
    unknown = [name for name in value if name not in _NULL_PROPORTION_OPTIONS]
    if unknown:
        raise ValueError(
            f"null_proportion has no option {quoted(unknown)}; its options are "
            f"{quoted(list(_NULL_PROPORTION_OPTIONS))}"
        )
    options = dict(value)
    if "lambdas" in options and options["lambdas"] is not None:
        lambdas = np.asarray(options["lambdas"], dtype=np.float64)
        options["lambdas"] = lambdas.tolist()
    # Checked once here, on p-values that every option accepts, so that a wrong
    # option fails before any pair of studies.
    proportion_of_true_nulls(np.linspace(0.0, 1.0, 101), **options)
    return options


def _study_weights(
    weights: Any,
    method: str,
    studies: tuple[str, ...],
    analysed: npt.NDArray[np.bool_],
    n_samples: npt.NDArray[np.int64],
) -> FloatArray | None:
    if method != "stouffer":
        if weights is not None:
            raise ValueError(
                f"weights apply to Stouffer's method only; method is {method!r}"
            )
        return None
    if weights is None:
        return np.where(analysed, np.sqrt(n_samples.astype(np.float64)), np.nan)
    if not isinstance(weights, Mapping):
        raise TypeError(
            "weights must be a mapping of study names to weights, got "
            f"{type(weights).__name__}"
        )
    extra = [name for name in weights if name not in studies]
    if extra:
        raise ValueError(
            f"weights names studies not in the collection: {quoted(extra)}"
        )
    chosen = np.full(len(studies), np.nan)
    for row, study in enumerate(studies):
        if not analysed[row]:
            continue
        if study not in weights:
            raise ValueError(
                f"weights must give every study that analysed the group and contrast; "
                f"{study!r} has none"
            )
        number = real_number(weights[study], f"weights[{study!r}]")
        if not (math.isfinite(number) and number > 0.0):
            raise ValueError(
                f"weights[{study!r}] must be positive and finite, got {number}"
            )
        chosen[row] = number
    return chosen


def _study_pairs(
    studies: tuple[str, ...],
    analysed: npt.NDArray[np.bool_],
    rows: npt.NDArray[np.bool_],
    found: npt.NDArray[np.bool_],
    lfc: FloatArray,
    p: FloatArray,
    options: dict[str, Any],
) -> Table:
    """pi1 and the sign agreement of every ordered pair of studies that analysed the
    group and contrast."""
    order = [row for row in range(len(studies)) if analysed[row]]
    names: list[tuple[str, str]] = []
    counts: list[tuple[int, int]] = []
    agreement: list[float] = []
    pi1: list[float] = []
    for a in order:
        for b in order:
            if a == b:
                continue
            chosen = found[a] & rows[b]
            names.append((studies[a], studies[b]))
            counts.append(
                (
                    int(np.count_nonzero(chosen)),
                    int(np.count_nonzero(chosen & found[b])),
                )
            )
            if chosen.any():
                same = np.sign(lfc[a, chosen]) == np.sign(lfc[b, chosen])
                agreement.append(float(np.mean(same)))
                pi1.append(proportion_of_true_nulls(p[b, chosen], **options).pi1)
            else:
                agreement.append(math.nan)
                pi1.append(math.nan)
    return Table(
        {
            "discovery": text_column([pair[0] for pair in names]),
            "replication": text_column([pair[1] for pair in names]),
            "n_features_found": np.array(
                [count[0] for count in counts], dtype=np.int64
            ),
            "n_features_found_in_both": np.array(
                [count[1] for count in counts], dtype=np.int64
            ),
            "sign_agreement": np.array(agreement, dtype=np.float64),
            "pi1": np.array(pi1, dtype=np.float64),
        }
    )


def _absence(
    reasons: list[str], status: npt.NDArray[Any], row: int, n_features: int
) -> str:
    """Why a study has no table of the group and contrast: the reason that
    StudyResults.table gives a feature it measured, or that it measured none."""
    start = row * n_features
    for column in range(n_features):
        if status[row, column] == "not_analysed":
            return reasons[start + column]
    return "the study measured none of the collection's features"
