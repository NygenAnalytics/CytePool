"""Prediction of case status in held-out studies, from genes chosen in the others.

:func:`~cytepool.held_out_prediction` leaves out the studies of a collection in turn,
chooses genes up and down in each cell type from the other studies, by SumRank
(:func:`~cytepool.stats.sumrank.fit_sumrank`) and by a random-effects meta-analysis
(:func:`~cytepool.stats.meta_analysis.fit_meta_analysis`), scores each sample of the
held-out study by how high its profiles rank those genes, and returns the area under the
ROC curve (AUC) of the scores of the study's cases against its controls, in a
:class:`~cytepool.HeldOutPrediction`, as the paper that introduced SumRank compares
methods (Nakatsuka et al. 2025). :func:`~cytepool.plotting.plot_held_out_auc` draws the
AUCs by study and method.
https://docs.nygen.io/CytePool/statistics/replication#held-out-prediction explains the
method, and https://docs.nygen.io/CytePool/statistics/replication#replication-limits how
it differs from the paper's.

Choosing the genes:
    In every group (cell type), from the studies other than the held-out one whose
    results test the group and contrast, and among the genes that every one of them
    tested, ``n_features`` genes up and ``n_features`` down, the same number for each
    method, as the paper compares methods by gene sets of one size:

    - ``"sumrank"``: the genes with the smallest ``up_statistic``, and those with the
      smallest ``down_statistic``, below the cap, with the given number or proportion
      of the studies summed; ties in the collection's order of the genes. Every gene has
      the same numbers of studies and ranks, so this is also the order of the
      permutation p-values, which therefore need no permutations here.
    - ``"meta_analysis"``: the genes with a positive estimate and the smallest
      p-values, and those with a negative one, from the random-effects
      meta-analysis of the studies' log2 fold changes and standard errors with REML
      and the Knapp-Hartung adjustment (``method="reml"``,
      ``test="knapp_hartung"``), as the paper ran metagen (``method.tau = "REML"``,
      ``hakn = TRUE``); ties in the collection's order.

Scoring a sample:
    The genes of a sample's profile in a group, its counts summed over the cells,
    are ranked from the highest count, tied genes sharing their average rank, among
    all the genes the held-out study measured. A set of n of these genes, those of a
    chosen set that the study measured, has UCell's score (Andreatta and Carmona
    2021), 1 - U / (n G), with U the sum of the set's ranks minus n (n + 1) / 2 and
    G the number of genes ranked: the maximum rank is the number of genes, as the
    paper set it to rank every gene (``maxRank = 16000``). Then, as the authors'
    code does:

    1. In each group, the scores of the genes up, and those of the genes down, are
       scaled over the held-out study's samples to [0, 1], minus their smallest
       value and divided by their range (0 for all when they are equal), which
       makes the score's own scale irrelevant.
    2. A sample without a profile in the group, that is not among the samples of
       the held-out study's fit of the group, takes the mean of the other samples'
       scaled scores, and every sample takes 0 when none has one, as for a set
       without a gene the study measured.
    3. A sample's score is the sum over the groups of its scaled scores of the
       genes up, minus the sum of its scaled scores of the genes down.

The AUC:
    The probability that a case scores higher than a control, ties counting one
    half: the Mann-Whitney statistic divided by the number of pairs of a case and a
    control (Hanley and McNeil 1982). The samples are those of the held-out study's
    design at one of the two levels that the contrast compares, cases at the level it
    tests and controls at the other, such as ``case`` and ``control`` of
    ``condition`` for the contrast ``conditioncase``; or those whose value of
    ``variable`` is ``case`` or ``control``, when these are given. 0.5 says the
    score does not separate them, and below 0.5 the genes predict the wrong
    direction; the AUC is NaN, with the reason, for a study without a case or
    without a control.

References:
    Nakatsuka N, Adler D, Jiang L, Hartman A, Cheng E, Klann E, Satija R (2025).
    Improving reproducibility of differentially expressed genes in single-cell
    transcriptomic studies of neurodegenerative diseases through meta-analysis.
    Nature Communications 16, 7436. doi:10.1038/s41467-025-62579-z

    Andreatta M, Carmona SJ (2021). UCell: robust and scalable single-cell gene
    signature scoring. Computational and Structural Biotechnology Journal 19,
    3796-3798. doi:10.1016/j.csbj.2021.06.043

    Hanley JA, McNeil BJ (1982). The meaning and use of the area under a receiver
    operating characteristic (ROC) curve. Radiology 143, 29-36.
    doi:10.1148/radiology.143.1.7063747

    Robin X, Turck N, Hainard A, Tiberti N, Lisacek F, Sanchez JC, Mueller M (2011).
    pROC: an open-source package for R and S+ to analyze and compare ROC curves. BMC
    Bioinformatics 12, 77. doi:10.1186/1471-2105-12-77
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._analysis_records import only_study_key, software_versions
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import fitted_lines, listing, quoted
from ._options import index_integer
from ._ranks import average_ranks
from ._study_statistics import (
    StudyStatistics,
    contrast_levels,
    study_results,
    study_statistics,
)
from .cube import CubeCollection
from .differential_expression import DifferentialExpressionResult
from .stats.meta_analysis import fit_meta_analysis
from .stats.sumrank import _studies_summed, fit_sumrank
from .study_results import StudyResults
from .table import Table

__all__ = ["HeldOutPrediction", "held_out_prediction"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

METHODS = ("sumrank", "meta_analysis")
"""The methods that choose the genes, in the order the tables list them."""

_DIRECTIONS = ("up", "down")
# The meta-analysis that chooses genes for comparison: the paper's metagen settings.
_META_ANALYSIS = {"method": "reml", "test": "knapp_hartung"}


class HeldOutPrediction:
    """The prediction of case status in every held-out study, by method.

    Built by :func:`held_out_prediction`. Three tables, the studies in the
    collection's order and the methods in the order ``"sumrank"``,
    ``"meta_analysis"``:

    - ``auc``: one row per held-out study and method: ``study``, ``method``,
      ``auc`` (NaN where it cannot be computed), ``n_cases``, ``n_controls``,
      ``n_groups`` (the groups whose genes scored the study), ``n_features_up`` and
      ``n_features_down`` (the genes chosen, summed over the groups, that the study
      measured) and ``reason`` (why the AUC is NaN; empty otherwise);
    - ``scores``: one row per held-out study, method and sample of its design that
      is a case or a control: ``study``, ``method``, ``sample_id``, ``case``
      (whether it is a case) and ``score``;
    - ``features``: the genes chosen for each held-out study from the other studies:
      ``study`` (held out), ``method``, ``group_id``, ``direction`` (``"up"`` or
      ``"down"``), ``rank`` (from 1) and ``feature_id``.

    Immutable.

    Raises:
        TypeError: If a table is not a ``Table``, or ``provenance`` not a mapping
            that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_auc", "_scores", "_features", "_record")

    _auc: Table
    _scores: Table
    _features: Table
    _record: str

    def __init__(
        self, auc: Table, scores: Table, features: Table, provenance: Mapping[str, Any]
    ) -> None:
        for name, given in (("auc", auc), ("scores", scores), ("features", features)):
            if not isinstance(given, Table):
                raise TypeError(f"{name} must be a Table, got {type(given).__name__}")
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_auc": auc,
            "_scores": scores,
            "_features": features,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("HeldOutPrediction is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("HeldOutPrediction is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            HeldOutPrediction,
            (self._auc, self._scores, self._features, json.loads(self._record)),
        )

    @property
    def auc(self) -> Table:
        """The AUC of every held-out study and method."""
        return self._auc

    @property
    def scores(self) -> Table:
        """The score of every case and control of every held-out study, by method."""
        return self._scores

    @property
    def features(self) -> Table:
        """The genes chosen from the other studies for every held-out study."""
        return self._features

    @property
    def settings(self) -> dict[str, Any]:
        """The settings, as a new dict (also in the provenance)."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the CytePool, numpy, numba and Python versions, the settings, and
        the provenance of every study's differential-expression result, by study
        name, which identifies its cube by its fingerprint and gives its design and
        the samples of every group's fit.
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def __repr__(self) -> str:
        table = self._auc
        settings = self.settings
        lines = [
            f"HeldOutPrediction of {settings['case']!r} against {settings['control']!r}"
            f" ({settings['variable']}), contrast {settings['contrast']!r}: "
            f"{len(set(table['study'].tolist())):,} held-out studies, "
            f"{settings['n_features']:,} genes up and down per group"
        ]
        for method in settings["methods"]:
            values = table["auc"][(table["method"] == method) & ~np.isnan(table["auc"])]
            mean = f"mean AUC {values.mean():.3f}" if values.size else "no AUC"
            lines.append(f"  {method}: {mean} over {values.size:,} studies")
        return fitted_lines("\n".join(lines))


def held_out_prediction(
    studies: StudyResults,
    contrast: str | None = None,
    *,
    groups: str | Sequence[str] | None = None,
    n_features: int = 50,
    n_studies_summed: int | None = None,
    proportion_of_studies: float | None = None,
    variable: str | None = None,
    case: str | None = None,
    control: str | None = None,
) -> HeldOutPrediction:
    """Leave each study out, choose genes from the others, and predict its cases.

    For every study of the collection in turn, genes up and down in each group are
    chosen from the results of the other studies, by SumRank and by a
    random-effects meta-analysis; each case and control of the held-out study is
    scored by how high its profiles rank those genes, as UCell scores them; and the
    AUC says how well the scores separate cases from controls. A sample's score is
    UCell's rank score of the chosen genes in each cell type, scaled over the
    held-out study's samples, the genes up counted for it and the genes down against
    it, summed over the cell types; the AUC is the probability that a case scores
    higher than a control, ties counting one half.
    https://docs.nygen.io/CytePool/statistics/replication#held-out-prediction explains
    the method. The results are those that
    :func:`~cytepool.sumrank` takes: each study's analysis of its aligned cube.

    Args:
        studies: The differential-expression result of every study of a
            collection, from :func:`~cytepool.limma_voom` or
            :func:`~cytepool.limma_trend`, whose tables hold the standard errors of
            the meta-analysis, as :func:`~cytepool.analyse_each_study` gives them.
        contrast: The contrast that the studies tested, cases against controls; it
            may be left out when they tested one.
        groups: A group id or a list of them, whose scores are summed; by default
            every group of the collection. A group that no other study tested gives
            a held-out study no genes.
        n_features: The number of genes chosen up, and down, in each group, by each
            method; at least 1.
        n_studies_summed: The number of the other studies summed by SumRank, as for
            :func:`~cytepool.sumrank`; by default every one.
        proportion_of_studies: Instead of ``n_studies_summed``, the proportion of
            the other studies summed by SumRank.
        variable: The variable of the studies' designs that tells cases from
            controls, such as ``"condition"``. By default, the factor whose two
            levels the contrast compares in every study's design (a level against
            the reference, or one level minus another), with ``case`` the level it
            tests and ``control`` the other; give ``variable``, ``case`` and
            ``control`` together otherwise.
        case: The variable's value for a case, such as ``"AD"``.
        control: Its value for a control.

    Returns:
        The AUC of every held-out study and method, the score of every sample, and
        the genes chosen.

    Raises:
        TypeError: If ``studies`` is not a ``StudyResults``, an argument has the
            wrong type, or is or holds a numpy masked array.
        ValueError: If a result did not analyse its study's aligned cube, the
            contrast is left out and the studies tested several, the contrast is an
            F-test, the results have no standard errors (dream's), some but not all
            of ``variable``, ``case`` and ``control`` are given, none is given and
            the contrast does not compare two levels of one factor in every study's
            design, a study's design has no variable ``variable``, ``case`` equals
            ``control``, a group is listed twice, or an option is invalid.
        KeyError: If a group is not a group of the collection, or no study tested
            ``contrast``.
    """
    studies = study_results(studies, "held_out_prediction")
    collection = studies.collection
    for name, value in (
        ("contrast", contrast),
        ("variable", variable),
        ("case", case),
        ("control", control),
    ):
        reject_masked(value, name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    contrast = only_study_key(studies, contrast, "contrast", "contrasts")
    given = [value is not None for value in (variable, case, control)]
    if any(given) and not all(given):
        raise ValueError(
            "variable, case and control are given together, or none of them, to read "
            "the two levels that the contrast compares from the studies' designs"
        )
    if not any(given):
        compared = contrast_levels(studies, contrast, collection.studies)
        if compared is None:
            raise ValueError(
                f"contrast {contrast!r} does not compare two levels of one factor in "
                "every study's design; give the variable that tells cases from "
                "controls, and its values, as variable, case and control"
            )
        variable, case, control = compared
    assert variable is not None and case is not None and control is not None
    if case == control:
        raise ValueError(
            f"case and control are both {case!r}; give the two values of {variable!r} "
            "that the AUC compares"
        )
    chosen = _groups(collection, groups)
    count = _n_features(n_features)
    # Checked here, against the studies other than one held out; a group that fewer
    # studies tested is checked when its genes are chosen.
    _studies_summed(
        n_studies_summed, proportion_of_studies, max(1, collection.n_studies - 1)
    )
    # Every group's results, read once: the held-out studies take rows of them.
    inputs = {group: study_statistics(studies, group, contrast) for group in chosen}
    for group, found in inputs.items():
        missing = found.analysed & np.isnan(found.standard_error).all(axis=1)
        if missing.any():
            study = collection.studies[int(np.flatnonzero(missing)[0])]
            raise ValueError(
                f"the results of study {study!r} have no standard errors for group "
                f"{group!r}, which the meta-analysis reads: dream_voom's tables have "
                "none; give results of limma_voom or limma_trend"
            )
    summed = (n_studies_summed, proportion_of_studies)
    names = list(collection.studies)
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    feature_ids = collection.features["feature_id"]
    for held, study in enumerate(names):
        samples = _held_out_samples(studies[study], study, variable, case, control)
        profiles = _profiles(collection, studies[study], study, chosen)
        for method in METHODS:
            sets = {}
            for group in chosen:
                found = inputs[group]
                training = found.analysed.copy()
                training[held] = False
                if not training.any():
                    continue
                try:
                    chosen_genes = _chosen_genes(method, found, training, count, summed)
                except ValueError as error:
                    error.add_note(
                        f"(choosing the genes of group {group!r} without study "
                        f"{study!r}, from the {int(training.sum())} other studies that "
                        "tested it)"
                    )
                    raise
                sets[group] = chosen_genes
                for direction, positions in zip(_DIRECTIONS, chosen_genes, strict=True):
                    for rank, position in enumerate(positions.tolist(), start=1):
                        gene_rows.append(
                            {
                                "study": study,
                                "method": method,
                                "group_id": group,
                                "direction": direction,
                                "rank": rank,
                                "feature_id": feature_ids[position],
                            }
                        )
            scores, summary = _scores(samples, profiles, sets, collection)
            row = {"study": study, "method": method, **summary}
            row |= _auc(scores, samples.case)
            rows.append(row)
            for sample, is_case, score in zip(
                samples.ids, samples.case.tolist(), scores.tolist(), strict=True
            ):
                score_rows.append(
                    {
                        "study": study,
                        "method": method,
                        "sample_id": sample,
                        "case": is_case,
                        "score": score,
                    }
                )
    settings = {
        "contrast": contrast,
        "variable": variable,
        "case": case,
        "control": control,
        "groups": chosen,
        "n_features": count,
        "n_studies_summed": n_studies_summed,
        "proportion_of_studies": (
            None
            if n_studies_summed is not None
            else 1.0
            if proportion_of_studies is None
            else float(proportion_of_studies)
        ),
        "methods": list(METHODS),
        "meta_analysis": dict(_META_ANALYSIS),
        "score": "UCell, maxRank = the number of genes ranked",
    }
    return HeldOutPrediction(
        _auc_table(rows),
        Table(
            {
                "study": text_column([r["study"] for r in score_rows]),
                "method": text_column([r["method"] for r in score_rows]),
                "sample_id": text_column([r["sample_id"] for r in score_rows]),
                "case": np.array([r["case"] for r in score_rows], dtype=bool),
                "score": np.array([r["score"] for r in score_rows], dtype=np.float64),
            }
        ),
        Table(
            {
                "study": text_column([r["study"] for r in gene_rows]),
                "method": text_column([r["method"] for r in gene_rows]),
                "group_id": text_column([r["group_id"] for r in gene_rows]),
                "direction": text_column([r["direction"] for r in gene_rows]),
                "rank": np.array([r["rank"] for r in gene_rows], dtype=np.int64),
                "feature_id": text_column([r["feature_id"] for r in gene_rows]),
            }
        ),
        {
            "analysis": "held_out_prediction",
            "versions": software_versions(),
            "settings": settings,
            "studies": {name: studies[name].provenance for name in names},
        },
    )


# Arguments.


def _groups(collection: CubeCollection, value: Any) -> list[str]:
    reject_masked(value, "groups")
    available: list[str] = collection.groups["group_id"].tolist()
    if value is None:
        return available
    if isinstance(value, str):
        listed: list[Any] = [value]
    elif isinstance(value, Sequence | np.ndarray):
        listed = list(value)
    else:
        raise TypeError(
            f"groups must be a group id or a list of them, got {type(value).__name__}"
        )
    if not listed:
        raise ValueError("groups is empty; give at least one, or None for every group")
    for group in listed:
        if not isinstance(group, str):
            raise TypeError(
                f"groups must be group ids (str), got {type(group).__name__} {group!r}"
            )
    repeated = [g for g in dict.fromkeys(listed) if listed.count(g) > 1]
    if repeated:
        raise ValueError(f"groups lists {quoted(repeated)} more than once")
    unknown = [g for g in listed if g not in available]
    if unknown:
        raise KeyError(
            f"no group {quoted(unknown)} in the collection; its groups are: "
            f"{listing(available)}"
        )
    return listed


def _n_features(value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError("n_features must be a whole number of genes, got bool")
    number = index_integer(value, "n_features", expected="a whole number of genes")
    if number < 1:
        raise ValueError(f"n_features must be at least 1, got {number}")
    return number


# The genes chosen from the other studies.


def _chosen_genes(
    method: str,
    found: StudyStatistics,
    training: npt.NDArray[np.bool_],
    count: int,
    summed: tuple[Any, Any],
) -> tuple[IntArray, IntArray]:
    """The positions, in the collection's features, of the genes chosen up and down
    from the training studies, best first."""
    statistics = found.statistics[training]
    universe = ~np.isnan(statistics).any(axis=0)
    if np.count_nonzero(universe) < 2:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    chosen: list[IntArray] = []
    if method == "sumrank":
        ranking = fit_sumrank(
            statistics, n_studies_summed=summed[0], proportion_of_studies=summed[1]
        )
        cap = ranking.n_studies_summed / 2.0
        for statistic in (ranking.up_statistic, ranking.down_statistic):
            candidates = np.flatnonzero(ranking.ranked & (statistic < cap))
            order = np.lexsort((candidates, statistic[candidates]))
            chosen.append(candidates[order][:count].astype(np.intp))
        return chosen[0], chosen[1]
    effects = found.log2_fold_change[training][:, universe]
    errors = found.standard_error[training][:, universe]
    combined = fit_meta_analysis(effects, errors, method="reml", test="knapp_hartung")
    positions = np.flatnonzero(universe)
    for sign in (1.0, -1.0):
        candidates = np.flatnonzero(sign * combined.estimate > 0.0)
        order = np.lexsort((candidates, combined.p_value[candidates]))
        chosen.append(positions[candidates[order][:count]].astype(np.intp))
    return chosen[0], chosen[1]


# The held-out study's samples, their profiles and their scores.


@dataclass(frozen=True)
class _Samples:
    """The cases and controls of a held-out study, in its cube's order."""

    ids: list[str]
    case: npt.NDArray[np.bool_]


def _held_out_samples(
    result: DifferentialExpressionResult,
    study: str,
    variable: str,
    case: str,
    control: str,
) -> _Samples:
    design = result.provenance["design"]
    samples = design["samples"]
    if variable not in samples or variable == "sample_id":
        variables = [name for name in samples if name != "sample_id"]
        raise ValueError(
            f"variable names {variable!r}, which is not a variable of the design of "
            f"study {study!r}; its variables are: {quoted(variables) or '(none)'}"
        )
    labels = {
        sample: str(value)
        for sample, value in zip(samples["sample_id"], samples[variable], strict=True)
    }
    # The design's samples in the cube's order, as the result's sample table has them.
    table = result.samples
    first = table["group_id"] == table["group_id"][0] if len(table) else []
    ordered = table["sample_id"][first].tolist() if len(table) else list(labels)
    kept = [sample for sample in ordered if labels[sample] in (case, control)]
    return _Samples(
        ids=kept, case=np.array([labels[s] == case for s in kept], dtype=bool)
    )


@dataclass(frozen=True)
class _Profiles:
    """A held-out study's profiles in one group: the counts of the samples of its fit
    of the group, sorted within each sample for ranking."""

    samples: list[str]
    counts: FloatArray  # samples x genes measured
    sorted_counts: FloatArray
    features: dict[str, int]  # feature id: column


def _profiles(
    collection: CubeCollection,
    result: DifferentialExpressionResult,
    study: str,
    groups: list[str],
) -> dict[str, _Profiles]:
    """The profiles of the held-out study in each group that its result analysed,
    for the samples of its fit of the group."""
    table = result.samples
    if not len(table):
        return {}
    cube = collection.aligned_cube(study)
    sample_rows = {s: i for i, s in enumerate(cube.samples["sample_id"].tolist())}
    features = {f: i for i, f in enumerate(cube.features["feature_id"].tolist())}
    profiles = {}
    analysed = set(result.groups)
    for group in groups:
        if group not in analysed:
            continue
        in_group = (table["group_id"] == group) & table["used"].astype(bool)
        used = table["sample_id"][in_group].tolist()
        if not used:
            continue
        rows = np.array([sample_rows[s] for s in used], dtype=np.intp)
        counts = np.asarray(cube.layer("counts", group), dtype=np.float64)[rows]
        profiles[group] = _Profiles(
            samples=used,
            counts=counts,
            sorted_counts=np.sort(counts, axis=1),
            features=features,
        )
    return profiles


def _set_scores(profiles: _Profiles, genes: list[str]) -> FloatArray | None:
    """UCell's score of a set of genes in every profile, with the maximum rank the
    number of genes ranked; None when the study measured none of the genes."""
    columns = np.array(
        [profiles.features[g] for g in genes if g in profiles.features], dtype=np.intp
    )
    if columns.size == 0:
        return None
    n_genes = profiles.counts.shape[1]
    n = columns.size
    scores = np.empty(len(profiles.samples))
    for row in range(len(profiles.samples)):
        ordered = profiles.sorted_counts[row]
        values = profiles.counts[row, columns]
        above = np.searchsorted(ordered, values, side="right")
        below = np.searchsorted(ordered, values, side="left")
        # The rank from the highest count: the genes above, plus the mean position
        # among the equal ones.
        ranks = (n_genes - above) + 0.5 * (above - below + 1)
        u = ranks.sum() - 0.5 * n * (n + 1)
        scores[row] = 1.0 - u / (n * n_genes)
    return scores


def _scaled(scores: FloatArray) -> FloatArray:
    """Scores minus their smallest value, divided by their range; 0 when equal."""
    low, high = scores.min(), scores.max()
    if not high > low:
        return np.zeros(scores.size)
    scaled: FloatArray = (scores - low) / (high - low)
    return scaled


def _scores(
    samples: _Samples,
    profiles: dict[str, _Profiles],
    sets: dict[str, tuple[IntArray, IntArray]],
    collection: CubeCollection,
) -> tuple[FloatArray, dict[str, Any]]:
    """The score of every case and control: over the groups, the scaled scores of
    the genes up minus those of the genes down, with the mean for a sample without
    a profile in a group."""
    feature_ids = collection.features["feature_id"]
    position = {sample: i for i, sample in enumerate(samples.ids)}
    total = np.zeros(len(samples.ids))
    n_groups = 0
    measured = {"up": 0, "down": 0}
    for group, chosen in sets.items():
        found = profiles.get(group)
        contributed = False
        for direction, sign, positions in zip(
            _DIRECTIONS, (1.0, -1.0), chosen, strict=True
        ):
            genes = [feature_ids[p] for p in positions.tolist()]
            scores = None if found is None else _set_scores(found, genes)
            if scores is None:
                continue  # every sample takes 0
            assert found is not None
            measured[direction] += sum(g in found.features for g in genes)
            scaled = _scaled(scores)
            values = np.full(len(samples.ids), scaled.mean())
            for sample, value in zip(found.samples, scaled.tolist(), strict=True):
                if sample in position:
                    values[position[sample]] = value
            total += sign * values
            contributed = True
        n_groups += contributed
    return total, {
        "n_groups": n_groups,
        "n_features_up": measured["up"],
        "n_features_down": measured["down"],
    }


def _auc(scores: FloatArray, case: npt.NDArray[np.bool_]) -> dict[str, Any]:
    """The AUC of the scores for cases against controls, and the numbers of each."""
    n_cases = int(np.count_nonzero(case))
    n_controls = int(case.size - n_cases)
    row: dict[str, Any] = {"n_cases": n_cases, "n_controls": n_controls}
    if n_cases == 0 or n_controls == 0:
        row["auc"] = np.nan
        row["reason"] = (
            f"the study has {n_cases} cases and {n_controls} controls; the AUC needs "
            "at least one of each"
        )
        return row
    ranks = average_ranks(np.ascontiguousarray(scores))
    statistic = ranks[case].sum() - 0.5 * n_cases * (n_cases + 1)
    row["auc"] = float(statistic / (n_cases * n_controls))
    row["reason"] = ""
    return row


def _auc_table(rows: list[dict[str, Any]]) -> Table:
    columns: dict[str, Any] = {
        "study": text_column([r["study"] for r in rows]),
        "method": text_column([r["method"] for r in rows]),
        "auc": np.array([r["auc"] for r in rows], dtype=np.float64),
    }
    for name in (
        "n_cases",
        "n_controls",
        "n_groups",
        "n_features_up",
        "n_features_down",
    ):
        columns[name] = np.array([r[name] for r in rows], dtype=np.int64)
    columns["reason"] = text_column([r["reason"] for r in rows])
    return Table(columns)
