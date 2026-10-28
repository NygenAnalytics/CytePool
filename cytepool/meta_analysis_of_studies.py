"""Meta-analysis of the differential expression of several studies, feature by feature.

:func:`~cytepool.meta_analysis` combines, for every group (cell type), contrast and
feature, the log2 fold changes and standard errors of several studies' results into an
estimate of a common log2 fold change with its between-study variance and
heterogeneity statistics, as metafor's ``rma.uni`` computes them for a model without
moderators (Viechtbauer 2010), and returns a :class:`~cytepool.MetaAnalysisResult`
whose tables adjust the p-values over the features (Benjamini and Hochberg 1995);
:func:`~cytepool.stats.meta_analysis.fit_meta_analysis` computes each meta-analysis.
https://docs.nygen.io/CytePool/statistics/meta-analysis explains the method.

References:
    Viechtbauer W (2010). Conducting meta-analyses in R with the metafor package.
    Journal of Statistical Software 36(3), 1-48. doi:10.18637/jss.v036.i03

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import json
import warnings
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._analysis_records import absence, only_key, software_versions
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import PACKAGE_DIRECTORY, fitted_lines, listing, quoted
from ._options import index_integer
from .differential_expression import DifferentialExpressionResult
from .stats.meta_analysis import _checked_options, fit_meta_analysis
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = ["MetaAnalysisResult", "meta_analysis"]

_ANALYSED = "analysed"  # the status of a group and contrast that was meta-analysed


class MetaAnalysisResult:
    """Meta-analyses of the differential expression of several studies.

    Built by :func:`meta_analysis`. Tables hold, for every group and contrast
    meta-analysed, one row per feature that at least ``min_studies`` studies tested, in
    the order in which the studies, taken in the order given, first list them:

    - ``feature_id`` and, when a study's tables have one, ``symbol``: the first symbol
      that a study gives for the feature, taking the studies in the order given;
    - ``n_studies``: the number of studies that tested the feature;
    - ``log2_fold_change`` and ``standard_error``: the estimate of the common log2
      fold change and its standard error;
    - ``statistic`` and ``p_value``: the z-statistic, or the Knapp-Hartung
      t-statistic on ``n_studies - 1`` degrees of freedom, and its two-sided p-value.
      A feature that one study tested, kept with ``min_studies=1``, has the z-test of
      that study's log2 fold change against its standard error, also with
      ``"knapp_hartung"``;
    - ``adjusted_p_value``: Benjamini-Hochberg over the features of the table
      (Benjamini and Hochberg 1995);
    - ``ci_lower`` and ``ci_upper``: the confidence interval of the log2 fold change;
    - ``tau_squared``: the between-study variance;
    - ``i_squared`` and ``h_squared``: I^2, as a proportion, and H^2;
    - ``q_statistic`` and ``q_p_value``: Cochran's Q and its p-value.

    The summary has one row per group and contrast: ``group_id``, ``contrast``,
    ``status`` (``"analysed"``; ``"too_few_studies"`` when fewer than ``min_studies``
    studies tested it; ``"no_features"`` when no feature was tested by that many),
    ``reason`` (empty when analysed), ``n_studies`` (the studies that tested it),
    ``n_features`` (in its table) and ``n_features_left_out`` (tested by fewer than
    ``min_studies`` studies). The table of studies has one row per group, contrast and
    study: ``group_id``, ``contrast``, ``study``, ``analysed`` (whether the study's
    results have a table of the group and contrast), ``reason`` (why not; empty when
    analysed) and ``n_features_tested`` (in the study's table; 0 when not analysed).

    The result keeps the studies' results it combined (``study_results``, the object
    given, not a copy), so that :func:`~cytepool.plotting.plot_forest` can draw a
    feature's estimate in every study; a pickled result carries them. Immutable.

    Raises:
        TypeError: If ``summary`` or ``studies`` is not a ``Table``, ``tables`` not a
            mapping of (group id, contrast name) to ``Table``, ``provenance`` not a
            mapping that JSON can write, or ``study_results`` not a mapping of study
            names to ``DifferentialExpressionResult``.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_summary", "_studies", "_tables", "_record", "_study_results")

    _summary: Table
    _studies: Table
    _tables: Mapping[tuple[str, str], Table]
    _record: str
    _study_results: Mapping[str, DifferentialExpressionResult] | None

    def __init__(
        self,
        summary: Table,
        studies: Table,
        tables: Mapping[tuple[str, str], Table],
        provenance: Mapping[str, Any],
        *,
        study_results: Mapping[str, DifferentialExpressionResult] | None = None,
    ) -> None:
        for name, table in (("summary", summary), ("studies", studies)):
            if not isinstance(table, Table):
                raise TypeError(f"{name} must be a Table, got {type(table).__name__}")
        if not isinstance(tables, Mapping):
            raise TypeError(
                f"tables must be a mapping of (group id, contrast name) to Table, got "
                f"{type(tables).__name__}"
            )
        for key, table in tables.items():
            pair = isinstance(key, tuple) and len(key) == 2
            if not (pair and all(isinstance(part, str) for part in key)):
                raise TypeError(
                    "tables must be keyed by (group id, contrast name), got the key "
                    f"{key!r}"
                )
            if not isinstance(table, Table):
                raise TypeError(
                    f"tables[{key!r}] must be a Table, got {type(table).__name__}"
                )
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        if study_results is not None:
            _studies(study_results, fewest=0, name="study_results")
        state = {
            "_summary": summary,
            "_studies": studies,
            "_tables": MappingProxyType(dict(tables)),
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
            "_study_results": study_results,
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MetaAnalysisResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MetaAnalysisResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_result,
            (
                self._summary,
                self._studies,
                dict(self._tables),
                json.loads(self._record),
                self._study_results,
            ),
        )

    @property
    def summary(self) -> Table:
        """The status, studies and features of every group and contrast."""
        return self._summary

    @property
    def studies(self) -> Table:
        """Every study of every group and contrast, whether it analysed them, and
        why not."""
        return self._studies

    @property
    def study_results(self) -> Mapping[str, DifferentialExpressionResult] | None:
        """The studies' results that were combined, as given (a
        :class:`~cytepool.StudyResults` or a mapping of study name to result), or None
        for a result built without them."""
        return self._study_results

    @property
    def tables(self) -> Mapping[tuple[str, str], Table]:
        """The table of every group and contrast meta-analysed, by (group, contrast)."""
        return self._tables

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the meta-analysis, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions; the settings; the
        provenance of every study's differential-expression result, by study name,
        which identifies its cube by its fingerprint and gives its design and
        settings and the samples of every group's fit; and, for every group and
        contrast, its status, the studies that tested it and those that did not with
        the reason, the number of features in its table and left out of it, and the
        features whose estimate of tau^2 did not converge (``not_converged``).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None, contrast: str | None = None) -> Table:
        """Return the table of one group and contrast.

        Args:
            group: A group id; it may be left out when the result has one group.
            contrast: A contrast name; it may be left out when the result has one
                contrast.

        Raises:
            ValueError: If a key is left out and the result has several.
            KeyError: If the group or contrast was not analysed, fewer than
                ``min_studies`` studies tested it, or none of its features; the message
                says why.
        """
        groups = list(dict.fromkeys(self._summary["group_id"].tolist()))
        contrasts = list(dict.fromkeys(self._summary["contrast"].tolist()))
        group = only_key(group, groups, "group", "groups")
        contrast = only_key(contrast, contrasts, "contrast", "contrasts")
        found = self._tables.get((group, contrast))
        if found is not None:
            return found
        if group not in groups:
            raise KeyError(
                f"no group {group!r} was analysed; groups are: "
                f"{listing(groups) or '(none)'}"
            )
        if contrast not in contrasts:
            raise KeyError(
                f"no contrast {contrast!r} was analysed; contrasts are: "
                f"{listing(contrasts) or '(none)'}"
            )
        row = np.flatnonzero(
            (self._summary["group_id"] == group)
            & (self._summary["contrast"] == contrast)
        )[0]
        raise KeyError(
            f"group {group!r} and contrast {contrast!r} were not meta-analysed: "
            f"{self._summary['reason'][row]}"
        )

    def __repr__(self) -> str:
        record = json.loads(self._record)
        settings = record["settings"]
        summary = self._summary
        n = len(summary)
        analysed = summary["status"].tolist().count(_ANALYSED)
        lines = [
            f"MetaAnalysisResult ({settings['method']}, test {settings['test']}): "
            f"{len(record['studies']):,} studies; {n:,} group and contrast "
            f"pair{'' if n == 1 else 's'}, {analysed:,} analysed"
        ]
        for row in range(n):
            label = f"  {summary['group_id'][row]}, {summary['contrast'][row]}: "
            used = int(summary["n_studies"][row])
            if summary["status"][row] == _ANALYSED:
                features = int(summary["n_features"][row])
                lines.append(f"{label}{used:,} studies, {features:,} features")
            else:
                lines.append(f"{label}not analysed ({summary['status'][row]})")
        return fitted_lines("\n".join(lines))


def meta_analysis(
    studies: Mapping[str, DifferentialExpressionResult],
    *,
    groups: str | Sequence[str] | None = None,
    contrasts: str | Sequence[str] | None = None,
    method: Literal["reml", "dl", "pm", "fixed"] = "reml",
    test: Literal["z", "knapp_hartung"] = "z",
    confidence_level: float = 0.95,
    min_studies: int = 2,
) -> MetaAnalysisResult:
    """Meta-analyse the differential expression of several studies, feature by feature.

    Each study is the result of :func:`~cytepool.limma_voom` or
    :func:`~cytepool.limma_trend` on its own cube, such as the cube of one population,
    tissue or platform: a :class:`~cytepool.StudyResults` from
    :func:`~cytepool.analyse_each_study`, or a mapping of study name to result. For
    every group (cell type) and contrast, the log2 fold changes of the studies that
    tested it, with their standard errors (the tables' ``log2_fold_change`` and
    ``standard_error``), are matched by feature id and meta-analysed by
    :func:`~cytepool.stats.meta_analysis.fit_meta_analysis`. Groups are matched by
    group id and contrasts by name, so the studies must name them alike, and a
    contrast should mean the same comparison in every study. A feature that some
    studies did not test, because their expression filter left it out or their cube
    does not have it, is analysed over the studies that tested it.

    The standard errors are the moderated ones of limma's empirical Bayes moderation,
    with which the tables' t-statistics are formed; the meta-analysis takes them as
    known, as it does any sampling variance, and so does not use their degrees of
    freedom.

    Args:
        studies: The differential-expression result of every study: a
            ``StudyResults``, or a mapping of study name to result; at least two.
        groups: A group id or a list of them; by default every group of any study,
            in the order in which the studies first list them.
        contrasts: A contrast name or a list of them; by default every contrast of
            any study, in the order in which the studies first list them.
        method: The estimator of the between-study variance, as for
            :func:`~cytepool.stats.meta_analysis.fit_meta_analysis`: ``"reml"``,
            ``"dl"``, ``"pm"`` or ``"fixed"``.
        test: ``"z"`` or ``"knapp_hartung"``, as for
            :func:`~cytepool.stats.meta_analysis.fit_meta_analysis`.
        confidence_level: The confidence level of the intervals, in (0, 1).
        min_studies: The fewest studies that must test a feature for it to be
            meta-analysed, from 1 to the number of studies; 2 by default, since one
            study gives no information about the between-study variance. A group and
            contrast that fewer studies test is not analysed. With 1, a feature that
            one study tested has the z-test of that study's log2 fold change against
            its standard error, taken as known, as metafor gives it for one study: its
            p-value is smaller than that of the study's own moderated t-test, whose
            degrees of freedom are finite.

    Returns:
        The table of every group and contrast meta-analysed, a summary of every group
        and contrast, the studies that tested them, the studies' results, and a
        provenance record.

    Warns:
        RuntimeWarning: If the estimate of tau^2 of a feature did not converge; the
            message names the group, the contrast and the features, which the record
            lists too.

    Raises:
        TypeError: If ``studies`` is not a mapping of study names to
            ``DifferentialExpressionResult``, an option has the wrong type, or
            ``groups``, ``contrasts`` or ``min_studies`` is or holds a numpy masked
            array.
        ValueError: If there are fewer than two studies, a group or contrast is
            listed twice, an option is invalid, ``min_studies`` exceeds the number of
            studies, or a study's table has no ``standard_error`` column, a log2 fold
            change that is infinite, or a standard error that is not positive and
            finite (NaN marks a feature the study did not test).
        KeyError: If a group or contrast is in no study's results.
    """
    reject_masked(groups, "groups")
    reject_masked(contrasts, "contrasts")
    results = _studies(studies)
    estimator, chosen_test, confidence = _checked_options(
        method, test, confidence_level
    )
    fewest = _min_studies(min_studies)
    if fewest > len(results):
        raise ValueError(
            f"min_studies ({fewest}) is larger than the number of studies "
            f"({len(results)}), so no feature could be meta-analysed"
        )
    group_ids = _chosen(
        "group", groups, [list(result.groups) for result in results.values()]
    )
    names = _chosen(
        "contrast",
        contrasts,
        [list(result.contrasts.names) for result in results.values()],
    )
    settings = {
        "method": estimator,
        "test": chosen_test,
        "confidence_level": confidence,
        "min_studies": fewest,
        "p_value_adjustment": "BH",
    }
    rows: list[dict[str, Any]] = []
    study_rows: list[dict[str, Any]] = []
    tables: dict[tuple[str, str], Table] = {}
    for group in group_ids:
        for contrast in names:
            tested: dict[str, Table] = {}
            left_out: dict[str, str] = {}
            for name, result in results.items():
                table = result.tables.get((group, contrast))
                if table is None:
                    left_out[name] = absence(result, group, contrast)
                else:
                    _check_table(table, name, group, contrast)
                    tested[name] = table
                study_rows.append(
                    {
                        "group_id": group,
                        "contrast": contrast,
                        "study": name,
                        "analysed": table is not None,
                        "reason": left_out.get(name, ""),
                        "n_features_tested": 0 if table is None else len(table),
                    }
                )
            row: dict[str, Any] = {
                "group_id": group,
                "contrast": contrast,
                "status": _ANALYSED,
                "reason": "",
                "studies": list(tested),
                "left_out": left_out,
                "n_features": 0,
                "n_features_left_out": 0,
                "not_converged": [],
            }
            if len(tested) < fewest:
                row["status"] = "too_few_studies"
                row["reason"] = (
                    f"{len(tested)} of the {len(results)} studies tested it, fewer "
                    f"than min_studies ({fewest})"
                )
                rows.append(row)
                continue
            table, left, failed = _meta_analysed_table(
                tested, estimator, chosen_test, confidence, fewest
            )
            row["n_features"], row["n_features_left_out"] = len(table), left
            row["not_converged"] = failed
            if len(table):
                tables[(group, contrast)] = table
            else:
                row["status"] = "no_features"
                row["reason"] = (
                    f"none of the {left} features that the {len(tested)} studies "
                    f"tested was tested by min_studies ({fewest}) of them"
                )
            if failed:
                warnings.warn(
                    f"group {group!r}, contrast {contrast!r}: the estimate of tau^2 "
                    f"did not converge for {len(failed)} feature"
                    f"{'' if len(failed) == 1 else 's'}, {quoted(failed)}; the record "
                    "lists them under not_converged",
                    RuntimeWarning,
                    skip_file_prefixes=(PACKAGE_DIRECTORY,),
                )
            rows.append(row)
    return MetaAnalysisResult(
        _summary_table(rows),
        Table(
            {
                column: (
                    text_column([row[column] for row in study_rows])
                    if column in ("group_id", "contrast", "study", "reason")
                    else np.array(
                        [row[column] for row in study_rows],
                        dtype=bool if column == "analysed" else np.int64,
                    )
                )
                for column in (
                    "group_id",
                    "contrast",
                    "study",
                    "analysed",
                    "reason",
                    "n_features_tested",
                )
            }
        ),
        tables,
        {
            "analysis": "meta_analysis",
            "versions": software_versions(),
            "settings": settings,
            "studies": {name: result.provenance for name, result in results.items()},
            "analyses": rows,
        },
        study_results=studies,
    )


def _meta_analysed_table(
    tested: Mapping[str, Table], method: str, test: str, level: float, fewest: int
) -> tuple[Table, int, list[str]]:
    """The meta-analysis of one group and contrast, the number of features that fewer
    than ``fewest`` studies tested, and the ids of those in the table whose estimate of
    tau^2 did not converge."""
    index: dict[str, int] = {}
    positions = []
    for table in tested.values():
        ids = table["feature_id"].tolist()
        for feature in ids:
            index.setdefault(feature, len(index))
        positions.append(np.array([index[feature] for feature in ids], dtype=np.intp))
    effects = np.full((len(tested), len(index)), np.nan)
    errors = np.full((len(tested), len(index)), np.nan)
    symbols: list[str | None] | None = None
    for row, (table, where) in enumerate(zip(tested.values(), positions, strict=True)):
        effects[row, where] = table["log2_fold_change"]
        errors[row, where] = table["standard_error"]
        if "symbol" in table:
            if symbols is None:
                symbols = [None] * len(index)
            for position, symbol in zip(
                where.tolist(), table["symbol"].tolist(), strict=True
            ):
                if symbols[position] is None:
                    symbols[position] = symbol
    fit = fit_meta_analysis(
        effects,
        errors,
        method=method,  # type: ignore[arg-type]
        test=test,  # type: ignore[arg-type]
        confidence_level=level,
    )
    keep = fit.n_studies >= fewest
    ids = list(index)
    columns: dict[str, Any] = {
        "feature_id": text_column([f for f, k in zip(ids, keep, strict=True) if k])
    }
    if symbols is not None:
        # A text column whose missing symbols are None.
        columns["symbol"] = np.array(
            [s for s, k in zip(symbols, keep, strict=True) if k], dtype=object
        )
    columns |= {
        "n_studies": fit.n_studies[keep],
        "log2_fold_change": fit.estimate[keep],
        "standard_error": fit.standard_error[keep],
        "statistic": fit.statistic[keep],
        "p_value": fit.p_value[keep],
        "adjusted_p_value": adjust_p_values(fit.p_value[keep], "bh"),
        "ci_lower": fit.ci_lower[keep],
        "ci_upper": fit.ci_upper[keep],
        "tau_squared": fit.tau_squared[keep],
        "i_squared": fit.i_squared[keep],
        "h_squared": fit.h_squared[keep],
        "q_statistic": fit.q_statistic[keep],
        "q_p_value": fit.q_p_value[keep],
    }
    failed = [ids[i] for i in np.flatnonzero(keep & ~fit.converged)]
    return Table(columns), int(np.count_nonzero(~keep)), failed


def _studies(
    results: Any, fewest: int = 2, name: str = "studies"
) -> dict[str, DifferentialExpressionResult]:
    """The results of the studies, checked, as a dict in their order."""
    if not isinstance(results, Mapping):
        raise TypeError(
            f"{name} must be a StudyResults, or a mapping of study names to "
            "DifferentialExpressionResult, such as {'cohort_a': limma_voom(cube_a, "
            f"design_a), ...}}, got {type(results).__name__}"
        )
    for study, result in results.items():
        if not isinstance(study, str):
            raise TypeError(
                f"{name} must be keyed by study names (str), got "
                f"{type(study).__name__} {study!r}"
            )
        if not isinstance(result, DifferentialExpressionResult):
            raise TypeError(
                f"{name}[{study!r}] must be a DifferentialExpressionResult, from "
                f"limma_voom or limma_trend, got {type(result).__name__}"
            )
    if len(results) < fewest:
        raise ValueError(
            f"a meta-analysis combines at least two studies, got {len(results)}"
        )
    return dict(results)


def _chosen(kind: str, value: Any, listed: list[list[str]]) -> list[str]:
    """The groups or contrasts to analyse: those given, or every one of any study."""
    available = list(dict.fromkeys(name for names in listed for name in names))
    if value is None:
        return available
    if isinstance(value, str):
        chosen: list[Any] = [value]
    elif isinstance(value, Sequence | np.ndarray):
        chosen = list(value)
    else:
        raise TypeError(
            f"{kind}s must be a {kind} name or a list of them, got "
            f"{type(value).__name__}"
        )
    if not chosen:
        raise ValueError(
            f"{kind}s is empty; give at least one {kind}, or None for every {kind}"
        )
    for name in chosen:
        if not isinstance(name, str):
            raise TypeError(
                f"{kind}s must be {kind} names (str), got {type(name).__name__} "
                f"{name!r}"
            )
    repeated = [name for name in dict.fromkeys(chosen) if chosen.count(name) > 1]
    if repeated:
        raise ValueError(f"{kind}s lists {quoted(repeated)} more than once")
    missing = [name for name in chosen if name not in available]
    if missing:
        raise KeyError(
            f"no study analysed the {kind} {quoted(missing)}; the studies' {kind}s "
            f"are: {listing(available) or '(none)'}"
        )
    return chosen


def _check_table(table: Table, study: str, group: str, contrast: str) -> None:
    """Refuse a study's table without the columns the meta-analysis reads, or with a
    log2 fold change or standard error it cannot use; the message names the feature."""
    where = (
        f"the table of study {study!r} for group {group!r} and contrast {contrast!r}"
    )
    missing = [
        column
        for column in ("feature_id", "log2_fold_change", "standard_error")
        if column not in table
    ]
    if missing:
        raise ValueError(
            f"{where} has no column {quoted(missing)}; the meta-analysis reads the "
            "log2 fold changes and standard errors of limma_voom and limma_trend "
            "results"
        )
    effects = np.asarray(table["log2_fold_change"], dtype=np.float64)
    errors = np.asarray(table["standard_error"], dtype=np.float64)
    for values, wrong, expected in (
        (effects, np.isinf(effects), "finite"),
        (
            errors,
            ~np.isnan(errors) & ~((errors > 0.0) & np.isfinite(errors)),
            "positive and finite",
        ),
    ):
        if wrong.any():
            row = int(np.flatnonzero(wrong)[0])
            column = "log2_fold_change" if values is effects else "standard_error"
            raise ValueError(
                f"{where} has the {column} {values[row]} for feature "
                f"{table['feature_id'][row]!r}; it must be {expected}, or NaN for a "
                "feature the study did not test"
            )


def _min_studies(value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError("min_studies must be a whole number of studies, got bool")
    number = index_integer(value, "min_studies", expected="a whole number of studies")
    if number < 1:
        raise ValueError(f"min_studies must be at least 1, got {number}")
    return number


def _summary_table(rows: list[dict[str, Any]]) -> Table:
    columns: dict[str, Any] = {
        name: text_column([row[name] for row in rows])
        for name in ("group_id", "contrast", "status", "reason")
    }
    columns["n_studies"] = np.array([len(row["studies"]) for row in rows], np.int64)
    for name in ("n_features", "n_features_left_out"):
        columns[name] = np.array([row[name] for row in rows], dtype=np.int64)
    return Table(columns)


def _restore_result(
    summary: Table,
    studies: Table,
    tables: Mapping[tuple[str, str], Table],
    provenance: Mapping[str, Any],
    study_results: Mapping[str, DifferentialExpressionResult] | None,
) -> MetaAnalysisResult:
    return MetaAnalysisResult(
        summary, studies, tables, provenance, study_results=study_results
    )
