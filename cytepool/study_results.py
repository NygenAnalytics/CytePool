"""The differential expression of every study of a collection, analysed alike.

Each study of a :class:`~cytepool.CubeCollection` is analysed on the collection's axes:
its aligned cube (:meth:`~cytepool.CubeCollection.aligned_cube`), with a design built
from its own samples. :func:`~cytepool.analyse_each_study` runs one analysis per study
that way, and :class:`~cytepool.StudyResults` keeps the results with their collection:
it lines them up feature by feature (:meth:`~cytepool.StudyResults.table`), with the
reason a study has no result for a feature (it did not measure it, did not test it, or
did not analyse the group or contrast), and gives one column of every study as a
studies x features matrix (:meth:`~cytepool.StudyResults.matrix`), the form that
:func:`~cytepool.stats.combined_p_values.combine_p_values` takes.
:func:`~cytepool.meta_analysis`, :func:`~cytepool.replication`,
:func:`~cytepool.sumrank` and :func:`~cytepool.held_out_prediction` take a
``StudyResults``.
https://docs.nygen.io/CytePool/statistics/aligning-studies#one-analysis-per-study
explains it.

Memory:
    Each result keeps the cube it analysed (see
    :class:`~cytepool.DifferentialExpressionResult`), here the study's aligned cube. The
    aligned cube of a cube read with :func:`~cytepool.read_cube` stays on disk; that of
    a cube in memory, when its groups or features are selected or renamed, holds a copy
    of the study's values, which the results keep for as long as they exist.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from ._analysis_records import absence, study_keys, study_union
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import fitted_lines, listing, quoted
from .cube import CubeCollection
from .design import Design
from .differential_expression import (
    DifferentialExpressionResult,
    dream_voom,
    limma_trend,
    limma_voom,
)
from .table import Table

__all__ = ["StudyResults", "analyse_each_study"]

_ANALYSES = (limma_voom, limma_trend, dream_voom)
# Why a study has no result for a feature, as StudyResults.table gives it.
_NOT_TESTED = (
    "the study measured the feature and tested the group and contrast, but not the "
    "feature, as when its expression filter left the feature out"
)


class StudyResults(Mapping[str, DifferentialExpressionResult]):
    """The differential-expression result of every study of a collection.

    A read-only mapping of study name to result, in the order of the collection's
    studies, with the collection the results analysed. Each result is the analysis
    of the study's aligned cube, such as :func:`analyse_each_study` makes, so that
    its groups are the collection's aligned groups and its features those the study
    measured. Immutable.

    Args:
        collection: The collection whose aligned cubes the results analysed.
        results: The result of every study of the collection, by study name, from
            :func:`~cytepool.limma_voom`, :func:`~cytepool.limma_trend` or
            :func:`~cytepool.dream_voom`.

    Raises:
        TypeError: If ``collection`` is not a ``CubeCollection``, ``results`` not a
            mapping, or a result not a ``DifferentialExpressionResult``.
        ValueError: If ``results`` does not give the result of every study of the
            collection, and no other.
    """

    __slots__ = ("_collection", "_results")

    _collection: CubeCollection
    _results: dict[str, DifferentialExpressionResult]

    def __init__(
        self,
        collection: CubeCollection,
        results: Mapping[str, DifferentialExpressionResult],
    ) -> None:
        if not isinstance(collection, CubeCollection):
            raise TypeError(
                f"collection must be a CubeCollection, got {type(collection).__name__}"
            )
        if not isinstance(results, Mapping):
            raise TypeError(
                "results must be a mapping of study names to "
                "DifferentialExpressionResult, one for every study of the collection, "
                f"got {type(results).__name__}"
            )
        studies = collection.studies
        missing = [study for study in studies if study not in results]
        extra = [name for name in results if name not in studies]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {quoted(missing)}")
            if extra:
                parts.append(f"not in the collection: {quoted(extra)}")
            raise ValueError(
                "results must give the result of every study of the collection, and "
                "no other: " + "; ".join(parts)
            )
        for study in studies:
            if not isinstance(results[study], DifferentialExpressionResult):
                raise TypeError(
                    f"results[{study!r}] must be a DifferentialExpressionResult, from "
                    "limma_voom, limma_trend or dream_voom, got "
                    f"{type(results[study]).__name__}"
                )
        object.__setattr__(self, "_collection", collection)
        object.__setattr__(
            self, "_results", {study: results[study] for study in studies}
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("StudyResults is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("StudyResults is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (StudyResults, (self._collection, self._results))

    # The mapping.

    def __getitem__(self, study: str) -> DifferentialExpressionResult:
        """Return the result of one study.

        Raises:
            KeyError: If there is no such study.
        """
        try:
            return self._results[study]
        except (KeyError, TypeError):
            raise KeyError(
                f"no study named {study!r}; studies are: {quoted(list(self._results))}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the study names, in the collection's order."""
        return iter(self._results)

    def __len__(self) -> int:
        """Return the number of studies."""
        return len(self._results)

    # What the results hold together.

    @property
    def collection(self) -> CubeCollection:
        """The collection whose aligned cubes the results analysed."""
        return self._collection

    @property
    def studies(self) -> tuple[str, ...]:
        """The study names, in the collection's order."""
        return tuple(self._results)

    @property
    def designs(self) -> Mapping[str, Design | None]:
        """The design each study's result analysed, by study name (read-only); None
        for a result built without one."""
        return MappingProxyType(
            {study: result.design for study, result in self._results.items()}
        )

    @property
    def summary(self) -> Table:
        """The summary of every study's result, one after another, with ``study``
        first: one row per study and group, with the status, samples, features and
        degrees of freedom of the group's fit. A column that some studies' summaries
        lack, such as dream's ``df_method``, is missing in their rows."""
        parts = [(study, result.summary) for study, result in self._results.items()]
        names = list(
            dict.fromkeys(name for _, table in parts for name in table.columns)
        )
        columns: dict[str, Any] = {
            "study": text_column(
                [study for study, table in parts for _ in range(len(table))]
            )
        }
        for name in names:
            present = [table[name] for _, table in parts if name in table]
            text = all(column.dtype.kind == "O" for column in present)
            if text:
                values: list[Any] = []
                for _, table in parts:
                    values += (
                        table[name].tolist() if name in table else [None] * len(table)
                    )
                columns[name] = text_column(values)
            elif (
                len(present) == len(parts)
                and len({column.dtype for column in present}) == 1
            ):
                columns[name] = np.concatenate(present)
            else:
                columns[name] = np.concatenate(
                    [
                        np.asarray(table[name], dtype=np.float64)
                        if name in table
                        else np.full(len(table), np.nan)
                        for _, table in parts
                    ]
                )
        return Table(columns)

    def table(self, group: str | None = None, contrast: str | None = None) -> Table:
        """Line up the studies' results of one group and contrast, feature by feature.

        The table has a row for every study and every feature of the collection: the
        studies in order, and the features in the collection's order within each
        study, as :meth:`matrix` lays them out. ``status`` says whether the study
        tested the feature, and why not:

        - ``"tested"``: the study's table of the group and contrast has the feature;
        - ``"not_tested"``: the study measured the feature and has a table of the
          group and contrast, without the feature, as when its expression filter
          left it out;
        - ``"not_measured"``: the study did not measure the feature
          (:attr:`CubeCollection.measured <cytepool.CubeCollection.measured>`), so no
          analysis of it could test it;
        - ``"not_analysed"``: the study measured the feature but has no table of the
          group and contrast: it does not have the group, its result leaves the group
          out, the group was not fitted, or the contrast cannot be estimated from its
          samples; ``reason`` says which, in the words of the study's result.

        The columns are ``study``, ``feature_id``, ``symbol`` (when the collection's
        features have it), ``status``, ``reason`` (empty where tested), and then the
        numeric columns of the studies' tables, such as ``log2_fold_change``,
        ``standard_error``, ``t``, ``p_value`` and ``adjusted_p_value``, in the order
        in which the studies first have them, NaN where a study did not test the
        feature. A missing feature is thus never a zero, and a feature that a study
        did not measure is told apart from one that it measured and left out.

        Args:
            group: An aligned group id; it may be left out when the studies analysed
                one group.
            contrast: A contrast name, or the name of an F-test of
                :func:`~cytepool.dream_voom`; it may be left out when the studies tested
                one.

        Returns:
            A table of ``collection.n_studies * collection.n_features`` rows.

        Raises:
            ValueError: If a key is left out and the studies analysed several, or a
                study's result did not analyse its aligned cube: it analyses a group
                that the study does not have in the collection, or tests a feature
                that the study did not measure there.
            KeyError: If ``group`` is not a group of the collection, or no study
                tested ``contrast``.
            TypeError: If a key is not a string.
        """
        group, contrast = study_keys(self._results, group, contrast)
        return _aligned_table(self._collection, self._results, group, contrast)

    def matrix(
        self, column: str, group: str | None = None, contrast: str | None = None
    ) -> np.ndarray:
        """Return one numeric column of every study, studies x features.

        The rows are the collection's studies and the columns its features, in its
        order (``collection.features``), as :meth:`table` lines them up: for
        example the p-values of every study, as
        :func:`~cytepool.stats.combined_p_values.combine_p_values` takes them, or
        the t-statistics. A feature that a study did not test is NaN.

        Args:
            column: A numeric column of the studies' tables, such as ``"p_value"``,
                ``"t"`` or ``"log2_fold_change"``.
            group: An aligned group id; it may be left out when the studies analysed
                one group.
            contrast: A contrast name; it may be left out when the studies tested
                one.

        Returns:
            A read-only float64 array of shape ``(n_studies, n_features)``.

        Raises:
            KeyError: If no study's table has the column, ``group`` is not a group of
                the collection, or no study tested ``contrast``.
            ValueError: If the column is not numeric, or a key is left out and the
                studies analysed several.
            TypeError: If an argument is not a string.
        """
        if not isinstance(column, str):
            reject_masked(column, "column")
            raise TypeError(
                f"column must be a column name, got {type(column).__name__}"
            )
        table = self.table(group, contrast)
        numeric = [
            name
            for name in table.columns
            if table[name].dtype.kind in "biuf" and name != "study"
        ]
        if column not in table.columns:
            raise KeyError(
                f"no study's table has a column {column!r}; the numeric columns are "
                f"{quoted(numeric)}"
            )
        if column not in numeric:
            raise ValueError(
                f"matrix gives a numeric column, and {column!r} holds text; the numeric "
                f"columns are {quoted(numeric)}, and table() gives every column"
            )
        values = np.asarray(table[column], dtype=np.float64).reshape(
            self._collection.n_studies, self._collection.n_features
        )
        values.flags.writeable = False
        return values

    def __repr__(self) -> str:
        groups = study_union(self._results, "groups")
        contrasts = study_union(self._results, "contrasts")
        lines = [
            f"StudyResults of {len(self._results):,} "
            f"stud{'y' if len(self._results) == 1 else 'ies'}: {len(groups):,} "
            f"group{'' if len(groups) == 1 else 's'}, {len(contrasts):,} "
            f"contrast{'' if len(contrasts) == 1 else 's'}: " + ", ".join(contrasts)
        ]
        for study, result in self._results.items():
            statuses = result.summary["status"].tolist()
            fitted = statuses.count("fitted")
            lines.append(
                f"  {study}: {result.method}, {len(statuses):,} "
                f"group{'' if len(statuses) == 1 else 's'}, {fitted:,} fitted"
            )
        return fitted_lines("\n".join(lines))


def analyse_each_study(
    collection: CubeCollection,
    formula: str,
    *,
    analysis: Any = limma_voom,
    random: str | None = None,
    levels: Mapping[str, Any] | None = None,
    contrasts: Mapping[str, Any] | None = None,
    **options: Any,
) -> StudyResults:
    """Analyse every study of a collection alike, each on its aligned cube.

    For each study, in the collection's order: its aligned cube
    (:meth:`~cytepool.CubeCollection.aligned_cube`), a :class:`~cytepool.Design` built
    from the cube's own samples with the formula, random effects, levels and contrasts
    given, and ``analysis`` of the cube with that design and ``options``. The loop a
    meta-analysis or a replication analysis starts from, in one call:
    ``analyse_each_study(collection, "~ condition", levels={"condition": ["control", "case"]}, contrasts={"case_vs_control": "conditioncase"})``.
    A study whose design or analysis fails stops the run, with a note that names the
    study.

    Args:
        collection: The studies, aligned.
        formula: The design's formula, such as ``"~ condition"``, over the columns
            of each study's sample table.
        analysis: :func:`~cytepool.limma_voom` (by default),
            :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom`.
        random: The design's random effects, such as ``"(1 | donor)"``, for
            ``dream_voom``.
        levels: The levels of the design's factors, the reference first, such as
            ``{"condition": ["control", "case"]}``.
        contrasts: The design's named contrasts, such as ``{"case_vs_control":
            "conditioncase"}``.
        **options: The options of ``analysis``, such as ``groups``,
            ``prior_weights`` or ``expression_filter``.

    Returns:
        The result of every study, with the collection.

    Raises:
        TypeError: If ``collection`` is not a ``CubeCollection``, ``formula`` not a
            string, or ``analysis`` not one of the three.
        ValueError, KeyError: As ``Design`` and ``analysis`` raise them for a study,
            with a note that names the study.
    """
    if not isinstance(collection, CubeCollection):
        raise TypeError(
            f"collection must be a CubeCollection, got {type(collection).__name__}"
        )
    if not isinstance(formula, str):
        reject_masked(formula, "formula")
        raise TypeError(
            f"formula must be a formula such as '~ condition', got "
            f"{type(formula).__name__}"
        )
    # Refused before any study is analysed, so that the message names no study.
    for name, value in (
        ("random", random),
        ("levels", levels),
        ("contrasts", contrasts),
        *options.items(),
    ):
        reject_masked(value, name)
    if not any(analysis is known for known in _ANALYSES):
        raise TypeError(
            "analysis must be limma_voom, limma_trend or dream_voom, the analyses of "
            f"a cube that StudyResults holds; got {analysis!r}"
        )
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        try:
            design = Design(
                cube.samples, formula, random=random, levels=levels, contrasts=contrasts
            )
            results[study] = analysis(cube, design, **options)
        except (TypeError, ValueError, KeyError) as error:
            error.add_note(f"(in study {study!r} of the collection)")
            raise
    return StudyResults(collection, results)


def _aligned_table(
    collection: CubeCollection,
    results: Mapping[str, DifferentialExpressionResult],
    group: str,
    contrast: str,
) -> Table:
    """The table of :meth:`StudyResults.table`, of results checked to be one per
    study of the collection."""
    studies = collection.studies
    for name, value in (("group", group), ("contrast", contrast)):
        reject_masked(value, name)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    group_ids = collection.groups["group_id"].tolist()
    if group not in group_ids:
        raise KeyError(
            f"no group {group!r} in the collection; its groups are: "
            f"{listing(group_ids) or '(none)'}"
        )
    tested_names = list(
        dict.fromkeys(
            name
            for study in studies
            for name in (*results[study].contrasts.names, *results[study].f_tests)
        )
    )
    if contrast not in tested_names:
        raise KeyError(
            f"no study tested the contrast {contrast!r}; the studies tested: "
            f"{listing(tested_names) or '(none)'}"
        )
    members = collection.group_members
    held: dict[str, set[str]] = {study: set() for study in studies}
    for study, aligned in zip(
        members["study"].tolist(), members["group_id"].tolist(), strict=True
    ):
        held[study].add(aligned)
    feature_ids = collection.features["feature_id"].tolist()
    index = {feature: position for position, feature in enumerate(feature_ids)}
    n_features = len(feature_ids)
    measured = collection.measured
    statuses: list[np.ndarray] = []
    reasons: list[list[str]] = []
    values: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for row, study in enumerate(studies):
        result = results[study]
        foreign = [name for name in result.groups if name not in held[study]]
        if foreign:
            raise ValueError(
                f"the result of study {study!r} analyses the group"
                f"{'' if len(foreign) == 1 else 's'} {quoted(foreign)}, which the "
                "study does not have in this collection; analyse the study's aligned "
                f"cube, collection.aligned_cube({study!r})"
            )
        table = None
        if group not in held[study]:
            why = f"study {study!r} has no group {group!r}"
        else:
            table = result.tables.get((group, contrast))
            why = "" if table is not None else absence(result, group, contrast)
        status = np.where(measured[row], "not_analysed", "not_measured").astype(object)
        reason = [
            why if flag else f"study {study!r} did not measure the feature"
            for flag in measured[row].tolist()
        ]
        if table is not None:
            positions = np.array(
                [index.get(feature, -1) for feature in table["feature_id"].tolist()],
                dtype=np.intp,
            )
            outside = (positions < 0) | ~measured[row, np.maximum(positions, 0)]
            if outside.any():
                feature = table["feature_id"][int(np.flatnonzero(outside)[0])]
                raise ValueError(
                    f"the result of study {study!r} tests the feature {feature!r}, "
                    "which the study did not measure in this collection; analyse the "
                    f"study's aligned cube, collection.aligned_cube({study!r})"
                )
            status[measured[row]] = "not_tested"
            status[positions] = "tested"
            for position in np.flatnonzero(measured[row]).tolist():
                reason[position] = _NOT_TESTED
            for position in positions.tolist():
                reason[position] = ""
            for name in table.columns:
                column = table[name]
                if name not in ("feature_id", "symbol") and column.dtype.kind in "biuf":
                    values.setdefault(name, []).append((row, positions, column))
        statuses.append(status)
        reasons.append(reason)
    columns: dict[str, Any] = {
        "study": text_column([study for study in studies for _ in range(n_features)]),
        "feature_id": text_column(feature_ids * len(studies)),
    }
    if "symbol" in collection.features:
        columns["symbol"] = text_column(
            collection.features["symbol"].tolist() * len(studies)
        )
    columns["status"] = text_column(
        [status for array in statuses for status in array.tolist()]
    )
    columns["reason"] = text_column([text for part in reasons for text in part])
    for name, parts in values.items():
        column = np.full((len(studies), n_features), np.nan)
        for row, positions, found in parts:
            column[row, positions] = found
        columns[name] = column.reshape(-1)
    return Table(columns)
