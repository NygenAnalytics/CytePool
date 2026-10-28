"""SumRank of the differential expression of the studies of a collection.

:func:`~cytepool.sumrank` computes SumRank (Nakatsuka et al. 2025) for a
:class:`~cytepool.StudyResults`, such as :func:`~cytepool.analyse_each_study` gives,
whose studies each analysed their aligned cube, in one group (cell type) and contrast,
and returns a :class:`~cytepool.SumRankResult`.
:func:`~cytepool.stats.sumrank.fit_sumrank` computes the statistic, its Irwin-Hall
p-values and its permutation p-values.
https://docs.nygen.io/CytePool/statistics/replication#sum-of-ranks explains them, and
https://docs.nygen.io/CytePool/statistics/replication#replication-limits how the
permutations differ from the paper's.

The studies' statistics:
    Each study ranks the genes by its signed -log10 p-value of the contrast:
    -log10(p_value), negative where the log2 fold change is negative, as the paper
    ranks DESeq2's results. The genes ranked are those that every study analysing
    the group and contrast tested; a study that did not analyse them at all is left
    out, as the authors' code leaves out a dataset without the cell type.

Permutations:
    With ``n_permutations``, the values of one variable of each study's design, such
    as the condition, are permuted among the samples of the study's fit of the
    group, with a ``numpy.random.Generator``, and the study's analysis,
    ``limma_voom`` or ``limma_trend`` with the design and settings its result
    records, is run again on its aligned cube, for that group and contrast only.
    Only designs without random effects are permuted, whose samples are exchangeable
    under the null; a design with several samples of a donor would need the donors'
    labels permuted instead. The variable (``permute``) is read from the contrast
    when it compares two levels of one factor in every study's design, such as
    ``conditioncase``, the level ``case`` of ``condition`` against its reference;
    otherwise it is given.
    The permutations are drawn in a fixed order, permutation after permutation and
    study after study, so a seed gives the same permutations, and the analyses do
    not depend on the number of threads.

Cost:
    A permutation runs the analysis of every study once, for one group; the SumRank
    statistics of a permutation take milliseconds. The memory is that of one study's
    analysis and a few arrays of the genes, whatever the number of permutations.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

import inspect
import json
import numbers
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._analysis_records import absence, software_versions, study_keys
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import PACKAGE_DIRECTORY, fitted_lines, fitted_message, listing, quoted
from ._options import index_integer, random_generator
from ._study_statistics import (
    contrast_levels,
    signed_statistics,
    study_results,
    study_statistics,
)
from .cube import Cube, CubeCollection
from .design import Design
from .differential_expression import (
    DifferentialExpressionResult,
    limma_trend,
    limma_voom,
)
from .stats.multiple_testing import adjust_p_values
from .stats.sumrank import _capped_sums, _NullCounts, _studies_summed
from .study_results import StudyResults
from .table import Table

__all__ = ["SumRankResult", "sumrank"]

FloatArray = np.ndarray

_DIRECTIONS = ("up", "down")
# The analyses that permutations re-run: a design without random effects, whose
# samples are exchangeable under the null.
_RERUN: dict[str, Callable[..., DifferentialExpressionResult]] = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
}


# SumRank of the studies of a collection.


class SumRankResult:
    """SumRank of the differential expression of the studies of a collection.

    Built by :func:`sumrank`, for one group (cell type) and contrast. The table of
    features (``features``) has one row per gene that every study tested, in the
    collection's order of the features:

    - ``feature_id`` and, when the collection's features have it, ``symbol``;
    - ``up_statistic``: the SumRank statistic of the gene up, the capped sum of its
      best normalized ranks (small for a gene up in the studies that rank it best);
    - ``up_p_value``: its p-value, from permutations when they were run
      (``settings["p_values"]`` is ``"permutation"``), from the Irwin-Hall
      distribution otherwise (``"irwin_hall"``);
    - ``up_adjusted_p_value``: Benjamini-Hochberg over the genes of the table
      (Benjamini and Hochberg 1995);
    - ``up_irwin_hall_p_value``: the Irwin-Hall p-value, also with permutations;
    - ``down_statistic``, ``down_p_value``, ``down_adjusted_p_value`` and
      ``down_irwin_hall_p_value``: the same for the gene down.

    The table of studies has one row per study of the collection: ``study``,
    ``analysed`` (whether its results test the group and contrast), ``reason`` (why
    not; empty when analysed) and ``n_features_tested`` (0 when not analysed).

    Immutable.

    Raises:
        TypeError: If ``features`` or ``studies`` is not a ``Table``, or
            ``provenance`` not a mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_features", "_studies", "_record")

    _features: Table
    _studies: Table
    _record: str

    def __init__(
        self, features: Table, studies: Table, provenance: Mapping[str, Any]
    ) -> None:
        for name, given in (("features", features), ("studies", studies)):
            if not isinstance(given, Table):
                raise TypeError(f"{name} must be a Table, got {type(given).__name__}")
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_features": features,
            "_studies": studies,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SumRankResult is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SumRankResult is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            SumRankResult,
            (self._features, self._studies, json.loads(self._record)),
        )

    @property
    def features(self) -> Table:
        """The statistics and p-values of every feature ranked."""
        return self._features

    @property
    def studies(self) -> Table:
        """Every study of the collection, whether it analysed the group and
        contrast, and why not."""
        return self._studies

    @property
    def settings(self) -> dict[str, Any]:
        """The settings, as a new dict (also in the provenance), with the group, the
        contrast, the proportion of studies, the numbers of studies and of ranks
        summed, the source of the p-values and the permutations."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions; the settings; the
        number of genes ranked and of genes left out, measured by a study but not
        tested by every study that analysed the group and contrast; and the
        provenance of every study's differential-expression result, by study name,
        which identifies its cube by its fingerprint and gives its design and
        settings and the samples of every group's fit.
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def __repr__(self) -> str:
        settings = self.settings
        table = self._features
        lines = [
            f"SumRankResult for group {settings['group']!r} and contrast "
            f"{settings['contrast']!r}: {settings['n_studies']:,} studies, the best "
            f"{settings['n_studies_summed']:,} summed; {len(table):,} genes"
        ]
        if settings["p_values"] == "permutation":
            lines.append(
                f"  p-values from {settings['n_permutations']:,} permutations of "
                f"{settings['permute']!r}"
            )
        else:
            lines.append("  p-values from the Irwin-Hall distribution")
        counts = [
            int(np.count_nonzero(table[f"{d}_adjusted_p_value"] < 0.05))
            for d in _DIRECTIONS
        ]
        lines.append(
            f"  adjusted p-value below 0.05: {counts[0]:,} up, {counts[1]:,} down"
        )
        return fitted_lines("\n".join(lines))


def sumrank(
    studies: StudyResults,
    group: str | None = None,
    contrast: str | None = None,
    *,
    n_studies_summed: int | None = None,
    proportion_of_studies: float | None = None,
    n_permutations: int = 0,
    permute: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> SumRankResult:
    """SumRank of one group and contrast across the studies of a collection.

    Each study's result is the analysis of its aligned cube
    (:meth:`~cytepool.CubeCollection.aligned_cube`), such as
    :func:`~cytepool.analyse_each_study` gives. For the group and contrast, the
    studies whose results test them each rank the genes that every one of them
    tested by their signed -log10 p-value, and each gene's best ranks are summed, as
    :func:`~cytepool.stats.sumrank.fit_sumrank` sums them, with Irwin-Hall p-values.
    https://docs.nygen.io/CytePool/statistics/replication#sum-of-ranks explains the
    statistic and its p-values.

    With ``n_permutations``, the values of the design variable ``permute``, such as
    the condition, are permuted among the samples of each study's fit of the group,
    and each study's analysis is run again on its aligned cube with the design and
    settings its result records (``limma_voom`` or ``limma_trend``), for this group
    and contrast only; the SumRank statistics of every permutation make the null
    distribution of the p-values. Each permutation runs the analysis of the group
    once in every study; the permutations are drawn permutation after permutation
    and study after study, so a seed gives the same permutations.

    Args:
        studies: The differential-expression result of every study of a
            collection, from :func:`~cytepool.limma_voom`,
            :func:`~cytepool.limma_trend` or :func:`~cytepool.dream_voom` (without
            permutations), as :func:`~cytepool.analyse_each_study` gives them.
        group: An aligned group id of the collection; it may be left out when the
            studies analysed one group.
        contrast: A contrast name that the studies tested; it may be left out when
            they tested one.
        n_studies_summed: The number of studies, among those that analysed the
            group and contrast, whose ranks are summed for each gene: those that
            rank it best; by default every study.
        proportion_of_studies: Instead of ``n_studies_summed``, the proportion of
            those studies whose ranks are summed, in (0, 1]; 1 sums every study.
        n_permutations: The number of permutations, at least 0; 0 gives Irwin-Hall
            p-values only.
        permute: The variable of the studies' designs whose values are permuted,
            such as ``"condition"``. By default, the factor whose two levels the
            contrast compares in every study's design (a level against the
            reference, or one level minus another); it must be given otherwise.
        rng: A ``numpy.random.Generator`` to draw the permutations from, an integer
            seed for one, or None for a fresh, unseeded generator.

    Returns:
        The statistics and p-values of every gene that every study analysing the
        group and contrast tested, a table of the studies, and a provenance record.

    Warns:
        RuntimeWarning: If fewer than all the studies are summed without
            permutations: the p-values are then the Irwin-Hall p-values, which are
            too small for such sums.

    Raises:
        TypeError: If ``studies`` is not a ``StudyResults``, an argument has the
            wrong type, or is or holds a numpy masked array.
        ValueError: If a key is left out and the studies analysed several, a result
            did not analyse its study's aligned cube, no study analysed the group and
            contrast, the contrast is an F-test, fewer than two genes were tested by
            every study, an option is invalid, both ``n_studies_summed`` and
            ``proportion_of_studies`` are given, ``permute`` is given without
            permutations, or is needed with them and the contrast does not compare
            two levels of one factor in every study's design, or a study cannot be
            permuted: its analysis has random effects, its design has no variable
            ``permute``, or a permutation leaves the contrast untestable.
        KeyError: If ``group`` is not a group of the collection, or no study tested
            ``contrast``.
    """
    studies = study_results(studies, "sumrank")
    collection = studies.collection
    reject_masked(n_permutations, "n_permutations")
    permutations = _n_permutations(n_permutations)
    if permute is not None and not isinstance(permute, str):
        reject_masked(permute, "permute")
        raise TypeError(
            f"permute must be a design variable name (str), got {type(permute).__name__}"
        )
    if not permutations and permute is not None:
        raise ValueError(
            "permute names the variable that permutations permute, but n_permutations "
            "is 0; give the number of permutations"
        )
    generator = random_generator(rng) if permutations else None
    group, contrast = study_keys(studies, group, contrast)
    inputs = study_statistics(studies, group, contrast)
    used = inputs.analysed
    if not used.any():
        raise ValueError(
            f"no study tested group {group!r} and contrast {contrast!r}: "
            + "; ".join(
                f"{study}: {reason}"
                for study, reason in zip(
                    collection.studies, inputs.reasons, strict=True
                )
            )
        )
    analysed = [
        s for s, flag in zip(collection.studies, used.tolist(), strict=True) if flag
    ]
    statistics = inputs.statistics[used]
    summed, proportion = _studies_summed(
        n_studies_summed, proportion_of_studies, len(analysed)
    )
    if permutations and permute is None:
        compared = contrast_levels(studies, contrast, analysed)
        if compared is None:
            raise ValueError(
                f"permutations permute the values of one variable of each study's "
                f"design, and contrast {contrast!r} does not compare two levels of one "
                "factor in every study's design; give the variable's name as permute"
            )
        permute = compared[0]
    if summed < len(analysed) and not permutations:
        warnings.warn(
            fitted_message(
                f"the Irwin-Hall p-values of a sum of the best {summed} of "
                f"{len(analysed)} studies' ranks are too small, as the sum of the best "
                "ranks is smaller than a sum of uniform ranks; the p_value columns "
                "hold them, so take p-values from permutations (n_permutations and "
                "permute)"
            ),
            RuntimeWarning,
            skip_file_prefixes=(PACKAGE_DIRECTORY,),
        )
    observed = _capped_sums(statistics, summed, "the studies' results")
    counts = None
    if permutations:
        assert generator is not None and permute is not None
        counts = _NullCounts(observed)
        for b, permuted in enumerate(
            _permuted_statistics(
                studies,
                analysed,
                group,
                contrast,
                permute,
                permutations,
                generator,
            )
        ):
            counts.add(_capped_sums(permuted, summed, f"permutation {b}"))
    ranked = observed.ranked
    features = collection.features
    columns: dict[str, Any] = {"feature_id": features["feature_id"][ranked]}
    if "symbol" in features:
        columns["symbol"] = features["symbol"][ranked]
    p_values = [observed.up_p, observed.down_p] if counts is None else counts.p_values()
    for direction, statistic, p_value, irwin_hall in zip(
        _DIRECTIONS,
        (observed.up, observed.down),
        p_values,
        (observed.up_p, observed.down_p),
        strict=True,
    ):
        columns[f"{direction}_statistic"] = statistic
        columns[f"{direction}_p_value"] = p_value
        columns[f"{direction}_adjusted_p_value"] = adjust_p_values(p_value, "bh")
        columns[f"{direction}_irwin_hall_p_value"] = irwin_hall
    measured = inputs.measured[used].any(axis=0)
    settings = {
        "group": group,
        "contrast": contrast,
        "statistic": "signed -log10 p-value",
        "proportion_of_studies": proportion,
        "n_studies": len(analysed),
        "n_studies_summed": summed,
        "p_values": "irwin_hall" if counts is None else "permutation",
        "n_permutations": permutations,
        "permute": permute,
        "seed": (
            int(rng)
            if isinstance(rng, numbers.Integral)
            and not isinstance(rng, bool | np.bool_)
            else None
        ),
        "n_null_values": 0 if counts is None else counts.n_values,
        "p_value_adjustment": "BH",
    }
    return SumRankResult(
        Table(columns),
        Table(
            {
                "study": text_column(list(collection.studies)),
                "analysed": used.copy(),
                "reason": text_column(inputs.reasons),
                "n_features_tested": inputs.n_tested,
            }
        ),
        {
            "analysis": "sumrank",
            "versions": software_versions(),
            "settings": settings,
            "n_features": int(np.count_nonzero(ranked)),
            "n_features_left_out": int(np.count_nonzero(measured & ~ranked)),
            "studies": {name: studies[name].provenance for name in collection.studies},
        },
    )


def _n_permutations(value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError("n_permutations must be a whole number, got bool")
    number = index_integer(value, "n_permutations", expected="a whole number")
    if number < 0:
        raise ValueError(f"n_permutations must be non-negative, got {number}")
    return number


# Permutations: each study's analysis re-run with the labels of its samples permuted.


@dataclass(frozen=True)
class _Rerun:
    """What it takes to run a study's analysis again with permuted labels."""

    study: str
    cube: Cube
    function: Callable[..., DifferentialExpressionResult]
    options: dict[str, Any]
    formula: str
    levels: dict[str, list[Any]]
    samples: dict[str, list[Any]]
    weights: list[float]  # of the contrast on the design's columns
    used: list[int]  # the design rows of the samples of the group's fit
    fingerprint: str
    settings: dict[str, Any]


def _permuted_statistics(
    studies: StudyResults,
    analysed: list[str],
    group: str,
    contrast: str,
    permute: str,
    n_permutations: int,
    generator: np.random.Generator,
) -> Iterator[FloatArray]:
    """The signed statistics of every study for each permutation, ``(n_studies,
    n_features)`` in the collection's order, NaN where an analysis did not test a
    gene. The permutations are drawn permutation by permutation, study by study."""
    collection = studies.collection
    reruns = [
        _rerun(collection, studies[study], study, group, contrast, permute)
        for study in analysed
    ]
    index = {
        feature: position
        for position, feature in enumerate(collection.features["feature_id"].tolist())
    }
    checked = [False] * len(reruns)
    for _ in range(n_permutations):
        statistics = np.full((len(reruns), len(index)), np.nan)
        for row, rerun in enumerate(reruns):
            values = list(rerun.samples[permute])
            drawn = generator.permutation(len(rerun.used))
            for position, source in zip(rerun.used, drawn.tolist(), strict=True):
                values[position] = rerun.samples[permute][rerun.used[source]]
            design = Design(
                {**rerun.samples, permute: values},
                rerun.formula,
                levels=rerun.levels,
                contrasts={contrast: rerun.weights},
            )
            result = rerun.function(rerun.cube, design, groups=[group], **rerun.options)
            if not checked[row]:
                _check_rerun(rerun, result)
                checked[row] = True
            table = result.tables.get((group, contrast))
            if table is None:
                raise ValueError(
                    f"a permutation of {permute!r} in study {rerun.study!r} leaves "
                    f"contrast {contrast!r} untestable in group {group!r}: "
                    f"{absence(result, group, contrast)}"
                )
            positions = np.array(
                [index[feature] for feature in table["feature_id"].tolist()],
                dtype=np.intp,
            )
            statistics[row, positions] = signed_statistics(
                table["p_value"], table["log2_fold_change"]
            )
        yield statistics


def _rerun(
    collection: CubeCollection,
    result: DifferentialExpressionResult,
    study: str,
    group: str,
    contrast: str,
    permute: str,
) -> _Rerun:
    """How to run a study's analysis again: its function and options, its design,
    and the samples of its fit of the group, from what its result records."""
    function = _RERUN.get(result.method)
    if function is None:
        raise ValueError(
            f"the result of study {study!r} is from {result.method}, whose design has "
            "random effects: its samples are not exchangeable under the null, so its "
            "labels cannot be permuted sample by sample; permutations re-run "
            "limma_voom and limma_trend"
        )
    record = result.provenance
    design = record["design"]
    samples: dict[str, list[Any]] = design["samples"]
    variables = [name for name in samples if name != "sample_id"]
    if permute not in variables:
        raise ValueError(
            f"permute names {permute!r}, which is not a variable of the design of "
            f"study {study!r}; its variables are: {quoted(variables) or '(none)'}"
        )
    settings: dict[str, Any] = record["settings"]
    parameters = inspect.signature(function).parameters
    options = {
        name: settings[name]
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and name not in ("groups", "contrasts")
        and name in settings
    }
    if options.get("expression_filter", True) is None:
        options["expression_filter"] = False  # recorded as None when it was off
    ids = samples["sample_id"]
    rows = {sample: position for position, sample in enumerate(ids)}
    table = result.samples
    in_group = (table["group_id"] == group) & table["used"].astype(bool)
    used = [rows[sample] for sample in table["sample_id"][in_group].tolist()]
    return _Rerun(
        study=study,
        cube=collection.aligned_cube(study),
        function=function,
        options=options,
        formula=design["formula"],
        levels=design["levels"],
        samples=samples,
        weights=design["contrasts"][contrast],
        used=used,
        fingerprint=record["cube"]["fingerprint"],
        settings=settings,
    )


def _check_rerun(rerun: _Rerun, result: DifferentialExpressionResult) -> None:
    """Refuse a re-run on another cube, or with other settings, than the study's
    analysis: its permutations would not be those of the study's results."""
    record = result.provenance
    if record["cube"]["fingerprint"] != rerun.fingerprint:
        raise ValueError(
            f"the result of study {rerun.study!r} was computed from a cube whose "
            f"fingerprint, {rerun.fingerprint}, is not that of the study's aligned "
            f"cube, {record['cube']['fingerprint']}: analyse "
            f"collection.aligned_cube({rerun.study!r})"
        )
    if record["settings"] != rerun.settings:
        changed = [
            name
            for name in sorted(set(record["settings"]) | set(rerun.settings))
            if record["settings"].get(name) != rerun.settings.get(name)
        ]
        raise ValueError(  # pragma: no cover - the options are those recorded
            f"the analysis of study {rerun.study!r} could not be re-run with its "
            f"recorded settings; these differ: {listing(changed)}"
        )
