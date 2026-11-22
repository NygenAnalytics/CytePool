"""Whether a dose series supports tests in each cell type, checked before anything is
fitted.

A dose analysis compares each dose with the vehicle, the samples given no dose, and
judges each difference against the variation between replicates given the same dose.
:func:`~cytepool.check_dose_design` says what each cell type (group) of a cube
supports, and returns a :class:`~cytepool.DoseDesignReport`
(https://docs.nygen.io/CytePool/statistics/dose-response#dose-design-check):

1. Samples. A design sample enters a cell type as it enters the cell type's fit in
   limma_voom: with at least ``min_cells`` cells there, when the cube records cells,
   some counts, and at least ``min_counts`` counts summed over all features. For the
   pathway scores or activities of one cell type's samples, a design sample enters
   when the scores hold it with a score that is not missing.
   Filtering can leave a dose group without a sample in some cell types, as
   dreamlet's documentation warns for the time points of an ordered covariate
   (Hoffman et al. 2026).
2. Dose groups. The distinct doses of the design's samples, the vehicle (dose 0)
   first. In a cell type, a dose group that keeps no sample is dropped. The
   replicates of a dose group are its samples or, when the design has random
   intercepts, the fewest levels of a random intercept's grouping among them, such
   as the animals whose samples it holds.
3. The dose-group model. The design's terms, an intercept, and an indicator of each
   treated dose group that keeps a sample, on the samples used. Its rank, by R's
   ``qr()`` rule, gives the residual degrees of freedom. A dose group is estimable,
   comparable with the vehicle apart from the design's other terms, when its column
   lies outside the span of the other columns (Searle 1971), which R's ``qr()`` and
   limma's ``nonEstimable`` report with the column placed last. Random intercepts
   are checked on these samples as ``Design.check`` checks them.
4. The batch. A dose group is confounded with a processing batch, such as a lane or
   a plate, when its difference from the vehicle is also a difference between
   batches: it is estimable, and not once the batches join the model. Batches of a
   single sample are pooled into one first, since such a batch cannot be told apart
   from its sample's own deviation; a sample alone in its batch, when it is the only
   one, joins no batch. A batch that coincides with the condition breaks the
   control of false positives, which summing the counts within each batch restores
   (Lun and Marioni 2017). The rule finds exact confounding only: a batch that is
   unbalanced across the doses without confounding them belongs in the design too.

Statuses:
    - ``"refused"``: no dose analysis. Every sample was left out; the design, or the
      cell type, has no vehicle sample; no treated sample is left; no treated dose
      group is estimable; or the random intercepts cannot be estimated beside the
      dose groups.
    - ``"descriptive"``: curves that describe the means of the dose groups, without
      p-values or bounds. No estimable treated dose group has two replicates, so no
      dose has a replicate; the model leaves no residual degrees of freedom; or a
      batch is confounded with an estimable dose group, unless the design has the
      batch as a random intercept and the vehicle and each such group span two or
      more batches.
    - ``"inferential"``: tests with p-values, and curves with bounds.

ETAP's design minimums:
    With ``rules="etap_2024"``, the report lists the design minimums of EPA's
    Transcriptomic Assessment Products (US EPA 2024) that samples show, each met or
    not: at least eight doses and a vehicle, spaced by half a log10 down from the
    highest dose, the lowest a full log10 below the second lowest (Sec. 3.4.1.3); at
    least four animals of each sex in each dose group (Sec. 3.4.2); and, per cell
    type and sex, at least two passing samples at each dose level and three passing
    vehicle samples (Sec. 3.4.5.2), passing being the cell type's filter of samples.
    A step counts as half a log10 within 0.05 of 0.5, and as a full log10 within
    0.05 of 1. ETAP excludes a dose level with fewer than two passing samples and
    goes on, so such a level is listed without changing the rules' status; any other
    unmet minimum makes the status a deviation.

References:
    Hoffman GE, Lee D, Bendl J, Prashant NM, Hong A, Casey C, Alvia M, Shao Z,
    Argyriou S, Therrien K, Venkatesh S, Voloudakis G, Haroutunian V, Fullard JF,
    Roussos P (2026). Efficient differential expression analysis of large-scale
    single-cell transcriptomics data using Dreamlet. Nature Communications 17, 9597.
    doi:10.1038/s41467-026-75680-8

    Lun ATL, Marioni JC (2017). Overcoming confounding plate effects in differential
    expression analyses of single-cell RNA-seq data. Biostatistics 18, 451-464.
    doi:10.1093/biostatistics/kxw055

    Searle SR (1971). Linear Models. Wiley, New York.

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496
"""

import json
import numbers
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._analysis_records import cube_samples, design_sample_ids, only_key
from ._columns import missing_mask, text_column
from ._doses import column_doses, distinct_doses, dose_text, is_axis_column
from ._group_fits import (
    group_positions,
    group_samples,
    min_cells_setting,
    min_counts_setting,
    no_samples_reason,
)
from ._linear_algebra import column_space
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import checked_choice
from ._sample_scores import SampleScores, sample_scores
from .activity import Activities
from .cube import Cube
from .design import Design
from .design._design_matrices import refuse_random_effects
from .design.checks import DesignReport, _aliasing, _relation, _same_partition
from .design.formula import RandomEffect
from .dose_axes import DoseAxis
from .normalization import library_sizes
from .pathway_scores import PathwayScores
from .stats._mixed_model_designs import mixed_design_of_samples, random_intercepts
from .table import Table

__all__ = ["DoseDesignReport", "check_dose_design"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

INFERENTIAL, DESCRIPTIVE, REFUSED = "inferential", "descriptive", "refused"
RULE_SETS = ("etap_2024",)
MET, DEVIATION = "met", "deviation"

# US EPA (2024): a vehicle and at least 8 doses, spaced by half a log10 down from the
# highest, the lowest a full log10 below the second lowest (Sec. 3.4.1.3); 4 animals
# of each sex per dose group (Sec. 3.4.2); 2 passing samples at each dose level and
# 3 passing vehicle samples (Sec. 3.4.5.2). A step's log10 is read to one decimal
# place, the precision of "half-log10".
ETAP_DOSES = 8
ETAP_STEP, ETAP_LOWEST_STEP, ETAP_STEP_TOLERANCE = 0.5, 1.0, 0.05
ETAP_ANIMALS = 4
ETAP_PASSING_SAMPLES = 2
ETAP_VEHICLE_SAMPLES = 3
# What ETAP does when a minimum is unmet.
DESIGN_DEVIATION = "deviation from the design"
LEVEL_EXCLUDED = "dose level excluded"
NOT_MODELLED = "dose response not modelled"


class DoseDesignReport:
    """What each cell type of a dose series supports, from :func:`check_dose_design`.

    Tables, one row per cell type checked, or per cell type and dose group or sample,
    in the order of the cube's groups (or of ``groups``), doses ascending:

    - ``summary``: ``group_id``; ``status``, ``"inferential"``, ``"descriptive"`` or
      ``"refused"``; ``reason``, why the cell type is not inferential, empty when it
      is; ``n_dose_groups``, the dose groups that keep a sample, the vehicle
      included, and ``n_dose_groups_dropped``, those that keep none;
      ``n_dose_groups_estimable``, the treated dose groups that the design can
      compare with the vehicle, which with the vehicle are the dose groups a curve
      can use; ``n_samples_used`` and ``n_samples_left_out``; ``df_residual``, the
      residual degrees of freedom of the dose-group model, NaN when no sample is
      used; and, with a rule set, ``rules_status``, ``"met"`` or ``"deviation"``.
    - ``doses``: ``group_id``; ``dose``, in the axis's unit; ``n_samples_used`` and
      ``n_samples_left_out``; ``n_cells_min``, when the cube records cells, and
      ``total_counts_min``, over the samples used (NaN for a group that keeps no
      sample), neither for scores; ``estimable``, whether the design can compare
      the treated dose group with the vehicle apart from its other terms (False for
      the vehicle, the reference, and for groups that keep no sample); and, with a
      batch, ``confounded_with_batch``, whether the group's difference from the
      vehicle is also a difference between batches (False for groups that are not
      estimable). A design that has the batch as a random intercept compares such
      groups against the variance between batches, so they do not make the cell type
      descriptive when the vehicle and each of them span two or more batches.
    - ``samples``: ``group_id``; ``sample_id``; ``dose``; ``n_cells``, when the cube
      records cells; ``total_counts``, the counts summed over all features, which
      ``min_counts`` reads, neither for scores; ``used``; and ``reason``, why the
      sample was left out, empty when used.
    - ``minimums``: with a rule set, one row per minimum, met or not, and None
      without: ``group_id``, None for a minimum of the design; ``sex``, None for a
      minimum of the doses; ``minimum``, ``"dose_levels"``, ``"dose_spacing"``,
      ``"sexes"``, ``"animals_per_sex_and_dose"``, ``"passing_samples_per_dose"`` or
      ``"passing_vehicle_samples"``; ``section``, of the rule set's document;
      ``required`` and ``observed``, as text; ``met``; and ``when_unmet``, what the
      rule set does when the minimum is unmet.

    ``design_check`` is :meth:`Design.check` of the design given, and ``settings``
    the settings of the check, the dose axis as :meth:`DoseAxis.to_dict` gives it,
    with None for ``min_cells`` and ``min_counts`` for scores.
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-design-check
    explains how to read the report.

    Built by :func:`check_dose_design`. Immutable.

    Raises:
        TypeError: If ``summary``, ``doses`` or ``samples`` is not a ``Table``,
            ``minimums`` is neither a ``Table`` nor None, ``design_check`` is not a
            ``DesignReport``, or ``settings`` is not a mapping that JSON can write.
        ValueError: If ``settings`` holds a NaN or an infinity.
    """

    __slots__ = (
        "_summary",
        "_doses",
        "_samples",
        "_minimums",
        "_design_check",
        "_settings",
    )

    _summary: Table
    _doses: Table
    _samples: Table
    _minimums: Table | None
    _design_check: DesignReport
    _settings: str

    def __init__(
        self,
        summary: Table,
        doses: Table,
        samples: Table,
        minimums: Table | None,
        design_check: DesignReport,
        settings: Mapping[str, Any],
    ) -> None:
        for label, table in (
            ("summary", summary),
            ("doses", doses),
            ("samples", samples),
        ):
            if not isinstance(table, Table):
                raise TypeError(f"{label} must be a Table, got {type(table).__name__}")
        if minimums is not None and not isinstance(minimums, Table):
            raise TypeError(
                f"minimums must be a Table or None, got {type(minimums).__name__}"
            )
        if not isinstance(design_check, DesignReport):
            raise TypeError(
                "design_check must be a DesignReport, got "
                f"{type(design_check).__name__}"
            )
        if not isinstance(settings, Mapping):
            raise TypeError(
                f"settings must be a mapping, got {type(settings).__name__}"
            )
        state = {
            "_summary": summary,
            "_doses": doses,
            "_samples": samples,
            "_minimums": minimums,
            "_design_check": design_check,
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_settings": json.dumps(settings, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DoseDesignReport is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DoseDesignReport is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            DoseDesignReport,
            (
                self._summary,
                self._doses,
                self._samples,
                self._minimums,
                self._design_check,
                self.settings,
            ),
        )

    @property
    def summary(self) -> Table:
        """One row per cell type, with its status, the reason, its dose groups and its
        samples."""
        return self._summary

    @property
    def doses(self) -> Table:
        """One row per cell type and dose group, with its samples, cells and counts, and
        whether it is estimable."""
        return self._doses

    @property
    def samples(self) -> Table:
        """One row per cell type and design sample, used or left out, and why."""
        return self._samples

    @property
    def minimums(self) -> Table | None:
        """One row per minimum of the rule set checked, met or not, or None without
        one."""
        return self._minimums

    @property
    def design_check(self) -> DesignReport:
        """:meth:`Design.check` of the design given."""
        return self._design_check

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the check as a new dict, with ``dose``, the axis as
        :meth:`DoseAxis.to_dict` gives it, which ``DoseAxis(**settings["dose"])``
        rebuilds, and ``batch``, ``rules``, ``sex``, ``min_cells``, ``min_counts`` and
        ``groups``, as given."""
        settings: dict[str, Any] = json.loads(self._settings)
        return settings

    def status(self, group: str | None = None) -> str:
        """The status of a cell type: ``"inferential"``, ``"descriptive"`` or
        ``"refused"``.

        Args:
            group: A group id; may be left out when the report checked one group.

        Raises:
            ValueError: If ``group`` is left out and the report checked several.
            KeyError: If the report did not check ``group``.
        """
        return str(self._summary["status"][self._row(group)])

    def rules_status(self, group: str | None = None) -> str:
        """Whether a cell type meets the minimums of the rule set checked:
        ``"met"`` or ``"deviation"``.

        A deviation means that a minimum of the design, or the minimum of passing
        vehicle samples of the cell type, is unmet; dose levels that the rule set
        excludes for want of passing samples are listed in ``minimums`` and leave the
        status unchanged.

        Args:
            group: A group id; may be left out when the report checked one group.

        Raises:
            ValueError: If no rule set was checked, or ``group`` is left out and the
                report checked several groups.
            KeyError: If the report did not check ``group``.
        """
        if "rules_status" not in self._summary:
            raise ValueError(
                "no rule set was checked; pass rules='etap_2024' and sex to "
                "check_dose_design"
            )
        return str(self._summary["rules_status"][self._row(group)])

    def _row(self, group: str | None) -> int:
        ids = self._summary["group_id"].tolist()
        key = only_key(group, ids, "group", "groups")
        if key not in ids:
            raise KeyError(
                f"the report did not check {key!r}; it checked {quoted(ids)}"
            )
        return int(ids.index(key))

    def __repr__(self) -> str:
        settings = self.settings
        unit = settings["dose"]["unit"]
        doses = np.unique(self._doses["dose"])
        header = (
            f"DoseDesignReport: {_count(len(self._summary), 'cell type')}, "
            f"{_count(doses.size, 'dose group')} from {doses[0]:g} to {doses[-1]:g} "
            f"{unit}"
        )
        if settings["batch"] is not None:
            header += f", batch {settings['batch']!r}"
        if settings["rules"] is not None:
            header += f", rules {settings['rules']}"
        lines = [header]
        for row in range(len(self._summary)):
            lines.append(self._cell_type_line(row, unit, settings["batch"]))
        check = self._design_check
        if check.errors or check.warnings:
            found = [
                _count(len(issues), kind)
                for kind, issues in (
                    ("error", check.errors),
                    ("warning", check.warnings),
                )
                if issues
            ]
            lines.append(
                f"  design check: {' and '.join(found)} ({', '.join(check.kinds())}); "
                "print report.design_check for the messages"
            )
        return fitted_lines("\n".join(lines))

    def _cell_type_line(self, row: int, unit: str, batch: str | None) -> str:
        summary = self._summary
        group = summary["group_id"][row]
        doses = self._doses.filter(group_id=group)
        parts = [str(summary["status"][row])]
        samples = doses["n_samples_used"]
        kept = samples[samples > 0]
        if kept.size:
            low, high = int(kept.min()), int(kept.max())
            each = f"{low}" if low == high else f"{low} to {high}"
            parts.append(
                f"{kept.size} of {samples.size} dose groups keep a sample, {each} each"
            )
        df = float(summary["df_residual"][row])
        if not np.isnan(df):
            parts.append(f"{df:g} residual df")
        if summary["reason"][row]:
            parts.append(str(summary["reason"][row]))
        # Kept treated dose groups that the design's other terms absorb, beside
        # others that it can compare with the vehicle.
        estimable = doses["estimable"]
        absorbed = (doses["dose"] > 0.0) & (samples > 0) & ~estimable
        if estimable.any() and absorbed.any():
            parts.append(
                f"{dose_text(doses['dose'][absorbed], unit)} cannot be compared with "
                "the vehicle apart from the design's other terms"
            )
        # Confounded dose groups of an inferential cell type are those that a random
        # intercept of the batch compares with the vehicle.
        if batch is not None and summary["status"][row] == INFERENTIAL:
            confounded = doses["confounded_with_batch"]
            if confounded.any():
                parts.append(
                    f"{dose_text(doses['dose'][confounded], unit)} confounded with "
                    f"{batch}, which a random intercept of the design models"
                )
        if self._minimums is not None:
            text = f"rules: {summary['rules_status'][row]}"
            excluded = _excluded_levels(self._minimums, group)
            if excluded:
                text += f", ETAP excludes {excluded}"
            parts.append(text)
        return f"  {group}: " + "; ".join(parts)


def check_dose_design(
    data: Cube | PathwayScores | Activities,
    design: Design,
    *,
    dose: DoseAxis,
    batch: str | None = None,
    rules: str | None = None,
    sex: str | None = None,
    min_cells: int = 10,
    min_counts: float = 0,
    groups: str | Sequence[str] | None = None,
) -> DoseDesignReport:
    """Check what a dose series supports in every cell type of a cube, or in the
    scores of one cell type's samples.

    For every cell type (group), before anything is fitted: the samples that the
    cell type's fit in :func:`~cytepool.limma_voom` would use, with the same
    ``min_cells`` and ``min_counts``; the samples, cells and counts each dose group
    keeps; the dose groups that the design's other terms, or a processing batch,
    make impossible to compare with the vehicle; the residual degrees of freedom;
    and a status, inferential, descriptive or refused, with the reason (the module
    documentation gives the rules). With ``rules="etap_2024"``, the design minimums
    of EPA's Transcriptomic Assessment Products, each met or not (US EPA 2024).
    https://docs.nygen.io/CytePool/statistics/dose-response#dose-design-check
    explains the check. For pathway scores or activities, the samples are those the
    scores hold with a score that is not missing, and the cell type is the scores'
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-signatures).

    Args:
        data: A cube with a ``counts`` layer, as :func:`~cytepool.limma_voom` takes;
            or the :class:`~cytepool.PathwayScores` or :class:`~cytepool.Activities`
            of the samples of one cell type, as :func:`~cytepool.dose_response`
            takes them, matched to the design's rows by ``sample_id``.
        design: A :class:`~cytepool.Design` of the terms other than dose, such as
            ``Design(cube.samples, "~ 1")`` or ``"~ sex"``, with random intercepts if
            the analysis has them, whose sample table has a ``sample_id`` column with
            cube sample ids and the dose column; its rows may be some of the cube's
            samples, in any order. The check adds the dose groups itself, so a design
            whose terms use the dose column or the axis's columns
            (``{column}_treated``, ``{column}_s1``, ...) is refused. The dose groups
            are the distinct doses of all the design's samples, so check a screen of
            several compounds one compound at a time, with a design of that
            compound's samples and the vehicle's.
        dose: The :class:`~cytepool.DoseAxis` whose ``column`` holds each sample's
            dose, 0 for the vehicle. The dose groups are the distinct doses of the
            design's samples.
        batch: A column of the design's samples that names each sample's processing
            batch, such as a lane, plate or run, read as labels whatever their type;
            None for none.
        rules: ``"etap_2024"`` to list the design minimums of EPA's Transcriptomic
            Assessment Products, each met or not, or None.
        sex: With ``rules``, and only then: the column of the design's samples that
            holds each sample's sex; a study of one sex has a single value there.
        min_cells: The fewest cells a sample needs in a cell type, as in
            :func:`~cytepool.limma_voom`. Ignored when the cube does not record cell
            numbers.
        min_counts: The fewest counts, summed over all features, a sample needs in
            a cell type, as in :func:`~cytepool.limma_voom`.
        groups: A group id or a list of them; all groups by default. ``groups``,
            ``min_cells`` and ``min_counts`` stay at their defaults for scores,
            which chose their samples when they were computed.

    Returns:
        The tables of the cell types, their dose groups, samples and, with
        ``rules``, the minimums, with :meth:`Design.check` of the design.

    Raises:
        TypeError: If ``data`` is not a Cube, PathwayScores or Activities, ``design``
            not a Design, ``dose`` not a DoseAxis, an option has the wrong type, the
            dose column does not hold
            real numbers, or ``groups`` or ``min_cells`` is or holds a numpy masked
            array.
        ValueError: If the design has random effects that are not random
            intercepts, has no ``sample_id`` column, names samples the cube does not
            have or names a sample twice; if its terms use the dose column or the
            axis's columns; if the dose, ``batch`` or ``sex``
            column is missing or holds a missing value; if a dose is not finite or is
            negative, or two doses differ by less than 1e-8 of the dose; if
            ``rules`` names no rule set, ``rules`` is given without ``sex`` or
            ``sex`` without ``rules``; or if an option is out of range. For scores,
            if the activities are of contrasts, no design sample is among the
            scores' samples, the design has random effects, or ``groups``,
            ``min_cells`` or ``min_counts`` is given.
        KeyError: If a group is not in the cube.
    """
    inputs = _checked_inputs(
        data, design, dose, batch, rules, sex, min_cells, min_counts, groups
    )
    checks = [_check_cell_type(inputs, k) for k in range(len(inputs.group_ids))]
    minimums = None
    if inputs.sexes is not None:  # sex comes with rules
        used = {check.group_id: check.used for check in checks}
        minimums = _etap_2024_minimums(inputs.doses, inputs.sexes, used, dose.unit)
    settings = {
        "dose": dose.to_dict(),
        "batch": batch,
        "rules": rules,
        "sex": sex,
        "min_cells": inputs.min_cells,
        "min_counts": inputs.min_counts,
        "groups": None if groups is None else [check.group_id for check in checks],
    }
    return DoseDesignReport(
        _summary_table(checks, minimums),
        _dose_table(inputs, checks),
        _sample_table(inputs, checks),
        minimums,
        design.check(),
        settings,
    )


# The inputs.


@dataclass(frozen=True)
class _Inputs:
    """The checked inputs, with the design's samples in cube order, or in the
    design's order for scores."""

    cube: Cube | None  # None for scores
    scores: SampleScores | None  # None for a cube
    design: Design
    axis: DoseAxis
    group_ids: list[str]  # the cell types to check
    positions: list[int]  # their positions among the cube's groups; none for scores
    sample_ids: list[str]  # the id of each design sample, in the order of the check
    data_rows: IntArray  # its cube position, increasing, or its row of the scores (-1)
    model_rows: IntArray  # the design row of each
    doses: FloatArray  # the dose of each
    levels: FloatArray  # the distinct doses, ascending
    level_of: IntArray  # the dose group of each design sample
    batch: str | None  # the batch column
    batches: list[str] | None  # the batch of each, as a label
    batch_modelled: bool  # a random intercept groups the samples as the batch does
    sexes: list[str] | None  # the sex of each, as a label
    min_cells: int | None  # None for scores, which chose their samples
    min_counts: float | None


def _checked_inputs(
    data: Any,
    design: Any,
    axis: Any,
    batch: Any,
    rules: Any,
    sex: Any,
    min_cells: Any,
    min_counts: Any,
    groups: Any,
) -> _Inputs:
    reject_masked(groups, "groups")
    scored = isinstance(data, PathwayScores | Activities)
    if not scored and not isinstance(data, Cube):
        raise TypeError(
            "data must be a Cube, or the PathwayScores or Activities of the samples of "
            f"one cell type, got {type(data).__name__}"
        )
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design of the terms other than dose, such as "
            "Design(cube.samples, '~ 1'), got " + type(design).__name__
        )
    if not isinstance(axis, DoseAxis):
        raise TypeError(
            "dose must be a DoseAxis, such as DoseAxis('dose', unit='ug/kg', "
            f"metric='administered'), got {type(axis).__name__}"
        )
    if scored:
        refuse_random_effects(design, "dose_response on scores")
    elif design.random:
        random_intercepts(design)
    if rules is not None:
        checked_choice(rules, "rules", RULE_SETS)
        if sex is None:
            raise ValueError(
                f"rules={rules!r} counts the animals of each sex, so it needs sex, the "
                "column of each sample's sex; a study of one sex has a single value "
                "there"
            )
    elif sex is not None:
        raise ValueError(
            "sex is read by a rule set's minimums; give rules='etap_2024' with it, or "
            "leave sex out"
        )
    _refuse_dose_terms(design, axis)
    cube, scores = (
        (None, sample_scores(data, "check_dose_design")) if scored else (data, None)
    )
    if scores is not None:
        _refuse_cube_options(groups, min_cells, min_counts)
        cells, counts, positions = None, None, []
        group_ids = [scores.group]
        sample_ids = design_sample_ids(
            design,
            no_ids="the design's sample table has no 'sample_id' column, so its rows "
            "cannot be matched to the samples of the scores; build the Design from the "
            "cube's samples",
            text_ids="the scores' profiles do",
        )
        row_of = {sample: row for row, sample in enumerate(scores.sample_ids)}
        data_rows = np.array([row_of.get(s, -1) for s in sample_ids], dtype=np.intp)
        if not np.any(data_rows >= 0):
            raise ValueError(
                "none of the design's samples is among the samples of the scores: the "
                f"scores' samples are such as {quoted(scores.sample_ids[:3])}, the "
                f"design's such as {quoted(sample_ids[:3])}"
            )
        model_rows = np.arange(len(sample_ids), dtype=np.intp)
    else:
        assert cube is not None  # data is a cube
        cells, counts = min_cells_setting(min_cells), min_counts_setting(min_counts)
        positions = group_positions(cube, groups)
        group_ids = [str(cube.groups["group_id"][g]) for g in positions]
        data_rows, model_rows = cube_samples(cube, design)
        sample_ids = cube.samples["sample_id"][data_rows].tolist()
    doses = column_doses(design.samples, axis.column)[model_rows]
    levels = distinct_doses(doses, axis.unit)
    batches = None
    if batch is not None:
        labels = _labels(design.samples, batch, "batch")
        batches = [labels[row] for row in model_rows]
    sexes = None
    if sex is not None:
        labels = _labels(design.samples, sex, "sex")
        sexes = [labels[row] for row in model_rows]
    batch_modelled = batches is not None and any(
        _same_partition(_codes(batches), _codes(_levels(design, effect, model_rows)))
        for effect in design.random
    )
    return _Inputs(
        cube=cube,
        scores=scores,
        design=design,
        axis=axis,
        group_ids=group_ids,
        positions=positions,
        sample_ids=sample_ids,
        data_rows=data_rows,
        model_rows=model_rows,
        doses=doses,
        levels=levels,
        level_of=np.searchsorted(levels, doses),
        batch=batch,
        batches=batches,
        batch_modelled=batch_modelled,
        sexes=sexes,
        min_cells=cells,
        min_counts=counts,
    )


def _refuse_cube_options(groups: Any, min_cells: Any, min_counts: Any) -> None:
    """Refuse the options that choose a cube's cell types and samples, given with
    scores, which are of one cell type and chose their samples when they were
    computed."""
    if groups is not None:
        raise ValueError(
            "groups chooses cell types of a cube, and scores are of the one cell type "
            "they were computed in; leave groups out"
        )
    # The defaults of check_dose_design and dose_response.
    for name, value, default in (
        ("min_cells", min_cells, 10),
        ("min_counts", min_counts, 0),
    ):
        reject_masked(value, name)
        given = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if not (given and value == default):
            raise ValueError(
                f"{name} chooses the samples of a cube's cell type, and the scores "
                "chose theirs when they were computed, as gsva(cube, ..., "
                f"min_cells=...) does; leave {name} out"
            )


def _refuse_dose_terms(design: Design, axis: DoseAxis) -> None:
    """Refuse a design whose fixed or random terms use the dose column or the
    columns of the axis, since the check adds the dose groups itself."""
    names = list(design.model_matrix.variables)
    for effect in design.random:
        names += [*effect.effects.variables, *effect.grouping]
    used = sorted({name for name in names if is_axis_column(name, axis.column)})
    if used:
        raise ValueError(
            f"the design uses {quoted(used)}, but the dose groups are added from the "
            "dose axis; give the design of the other terms, such as "
            "Design(cube.samples, '~ 1'), or '~ sex' for a study of both sexes"
        )


def _labels(samples: Table, column: Any, argument: str) -> list[str]:
    """The values of a column of the design's samples, as labels."""
    if not isinstance(column, str):
        raise TypeError(
            f"{argument} must name a column of the design's samples (str), got "
            f"{type(column).__name__}"
        )
    if column not in samples:
        raise ValueError(
            f"{argument} names {column!r}, which is not a column of the design's "
            f"samples{close_matches(column, samples.columns)}; columns are: "
            f"{listing([repr(name) for name in samples.columns])}"
        )
    values = samples[column]
    missing = np.flatnonzero(missing_mask(values))
    if missing.size:
        # The design's samples were matched to the cube's by their text ids.
        ids = samples["sample_id"][missing].tolist()
        raise ValueError(
            f"the {argument} column {column!r} is missing for sample(s) "
            f"{quoted(ids)}; every sample needs a {argument}"
        )
    return [str(value) for value in values.tolist()]


def _levels(design: Design, effect: RandomEffect, rows: IntArray) -> list[tuple]:
    """The level of a random intercept's grouping for each design row in ``rows``."""
    return list(zip(*(design.samples[name][rows].tolist() for name in effect.grouping)))


def _codes(values: Sequence[Hashable]) -> IntArray:
    """Integer codes of values, one per distinct value, in the order they appear."""
    index: dict[Hashable, int] = {}
    return np.array([index.setdefault(value, len(index)) for value in values])


# One cell type: the facts of its dose-group model, then its status.


@dataclass(frozen=True)
class _Model:
    """The facts of a cell type's dose-group model, from which its status follows."""

    kept: IntArray  # the samples each dose group keeps
    replicates: IntArray  # those samples, or the fewest levels of a grouping
    batch_spread: bool  # the vehicle and each confounded group span two batches
    rank: int
    df_residual: float  # NaN without samples
    estimable: np.ndarray  # bool, per dose group
    confounded: np.ndarray  # bool, per dose group
    relations: str  # the dose columns' relations, when no group is estimable
    random_error: str  # why the random intercepts cannot be estimated, or ""
    unit: str | None  # a random intercept's grouping nested in the dose groups


@dataclass(frozen=True)
class _CellType:
    """What one cell type keeps and supports."""

    group_id: str
    used: np.ndarray  # bool, per design sample in the order of the check
    reasons: list[str]  # why each design sample was left out; "" when used
    cells: np.ndarray | None  # cells of each design sample, when the cube records them
    totals: FloatArray | None  # each design sample's counts over all features; a cube's
    model: _Model
    status: str
    reason: str


class _Samples(NamedTuple):
    """The design samples a cell type uses, why the others are left out, their cells
    and total counts when the data are a cube, and why none is used when none is."""

    used: np.ndarray
    reasons: list[str]
    cells: np.ndarray | None
    totals: FloatArray | None
    none_used: str


def _check_cell_type(inputs: _Inputs, k: int) -> _CellType:
    found = _samples_of(inputs, k)
    model = _dose_group_model(inputs, found.used)
    status, reason = _status(inputs, model, found.none_used)
    return _CellType(
        inputs.group_ids[k],
        found.used,
        found.reasons,
        found.cells,
        found.totals,
        model,
        status,
        reason,
    )


def _samples_of(inputs: _Inputs, k: int) -> _Samples:
    """The samples of the k-th cell type checked: for a cube, those its fit in
    limma_voom would use; for scores, those the scores hold with a score that is not
    missing."""
    scores = inputs.scores
    if scores is None:
        assert inputs.cube is not None and inputs.min_cells is not None
        assert inputs.min_counts is not None
        counts, cells, reasons, used = group_samples(
            inputs.cube,
            inputs.data_rows,
            inputs.positions[k],
            inputs.min_cells,
            inputs.min_counts,
        )
        totals = library_sizes(counts) if counts.size else np.zeros(used.size)
        none_used = no_samples_reason(
            used.size, inputs.min_cells, inputs.min_counts, cells is not None
        )
        return _Samples(used, reasons, cells, totals, none_used)
    rows = inputs.data_rows
    held = rows >= 0
    used = np.zeros(rows.size, dtype=bool)
    used[held] = ~np.isnan(scores.values[rows[held]]).all(axis=1)
    reasons = [
        ""
        if seen
        else "every score of this sample is missing"
        if kept
        else "the scores do not hold this sample"
        for kept, seen in zip(held.tolist(), used.tolist(), strict=True)
    ]
    none_used = (
        f"every one of the {rows.size} samples was left out: every score of each "
        "sample the scores hold is missing"
    )
    return _Samples(used, reasons, None, None, none_used)


def _dose_group_model(inputs: _Inputs, used: np.ndarray) -> _Model:
    """The dose-group model of the samples a cell type uses: the design's terms, an
    intercept, and an indicator of each treated dose group that keeps a sample."""
    design, levels = inputs.design, inputs.levels
    rows, groups = inputs.model_rows[used], inputs.level_of[used]
    kept = np.bincount(groups, minlength=levels.size)
    replicates = kept.copy()
    for effect in design.random:
        replicates = np.minimum(
            replicates, _levels_per_group(_levels(design, effect, rows), groups, kept)
        )
    estimable = np.zeros(levels.size, dtype=bool)
    confounded = np.zeros(levels.size, dtype=bool)
    if not rows.size:
        return _Model(
            kept,
            replicates,
            False,
            0,
            float("nan"),
            estimable,
            confounded,
            relations="",
            random_error="",
            unit=None,
        )
    treated = np.flatnonzero((levels > 0.0) & (kept > 0))
    others = np.column_stack([design.matrix[rows], np.ones(rows.size)])
    indicators = (groups[:, None] == treated).astype(np.float64)
    x = np.column_stack([others, indicators])
    kept_columns = list(column_space(x).kept)
    compared = levels[0] == 0.0 and kept[0] > 0 and treated.size > 0
    if compared:
        estimable[treated] = _outside_span(others, indicators)
    relations = ""
    if compared and not estimable.any():
        relations = _relations(inputs, x, others.shape[1], treated)
    spread = False
    if estimable.any() and inputs.batches is not None:
        labels = [inputs.batches[i] for i in np.flatnonzero(used)]
        with_batch = np.column_stack([others, _batch_columns(labels)])
        outside = _outside_span(with_batch, indicators)
        confounded[treated] = estimable[treated] & ~outside
        batches = _levels_per_group(labels, groups, kept)
        spread = batches[0] >= 2 and bool(np.all(batches[confounded] >= 2))
    random_error, unit = "", None
    if estimable.any() and design.random:
        names = (*design.columns, "(Intercept)", *_group_names(inputs, treated))
        try:
            mixed_design_of_samples(
                design, rows, x[:, kept_columns], tuple(names[j] for j in kept_columns)
            )
        except ValueError as error:
            random_error = str(error)
            unit = _unit_per_dose_group(design, rows, groups)
    rank = len(kept_columns)
    return _Model(
        kept,
        replicates,
        spread,
        rank,
        float(rows.size - rank),
        estimable,
        confounded,
        relations,
        random_error,
        unit,
    )


def _status(inputs: _Inputs, model: _Model, none_used: str) -> tuple[str, str]:
    """The status of a cell type and the reason, from the facts of its dose-group
    model, by the rules of the module documentation in order; ``none_used`` is the
    reason when no sample is used."""
    levels, level_of = inputs.levels, inputs.level_of
    n_used = int(model.kept.sum())
    if not n_used:
        return REFUSED, none_used
    if levels[0] != 0.0:
        return REFUSED, (
            "the design has no sample of dose 0, the vehicle, which every dose "
            "analysis compares the doses with; compare the doses with each other "
            "with limma_voom"
        )
    if not model.kept[0]:
        n_vehicle = int(np.count_nonzero(level_of == 0))
        return REFUSED, (
            f"no vehicle sample is left ({_count(n_vehicle, 'vehicle sample')} left "
            "out); compare the doses with each other with limma_voom"
        )
    if not model.kept[1:].any():
        n_treated = int(np.count_nonzero(level_of > 0))
        if not n_treated:
            return REFUSED, "the design has no treated sample, only the vehicle"
        return REFUSED, (
            f"no treated sample is left ({_count(n_treated, 'treated sample')} left "
            "out)"
        )
    if not model.estimable.any():
        return REFUSED, (
            "no treated dose group can be compared with the vehicle apart from the "
            f"design's other terms: {model.relations}"
        )
    if model.random_error:
        unit = model.unit
        if unit is None:
            return REFUSED, model.random_error
        return REFUSED, (
            f"each dose group holds the samples of a single {unit}, so no dose has a "
            f"second {unit} to replicate it and the random intercept of {unit} "
            f"cannot be estimated beside the dose groups; pool the samples of each "
            f"{unit} at each dose, with cube.collapse_samples({unit!r}) when each "
            f"{unit} received one dose, and check the pooled cube"
        )
    if not np.any(model.replicates[model.estimable] >= 2):
        return DESCRIPTIVE, _replication_reason(inputs, model)
    if model.df_residual == 0:
        return DESCRIPTIVE, (
            f"no residual degrees of freedom: the model has rank {model.rank} on the "
            f"{n_used} samples left, so it fits them exactly"
        )
    # A random intercept of the batch compares the confounded dose groups with the
    # vehicle when the vehicle and each of them span two batches or more.
    if model.confounded.any() and not (inputs.batch_modelled and model.batch_spread):
        return DESCRIPTIVE, _batch_reason(inputs, model)
    return INFERENTIAL, ""


def _replication_reason(inputs: _Inputs, model: _Model) -> str:
    """Why no estimable treated dose group has a replicate: it keeps one sample, or
    its samples come from one level of a random intercept's grouping."""
    groupings = [effect.group for effect in inputs.design.random]
    if groupings and np.any(model.kept[model.estimable] >= 2):
        second = " and a second ".join(groupings)
        return (
            "no treated dose group that can be compared with the vehicle keeps "
            f"samples of a second {second}, so no dose has a replicate"
        )
    return (
        "no treated dose group that can be compared with the vehicle keeps a second "
        "sample, so no dose has a replicate"
    )


def _batch_reason(inputs: _Inputs, model: _Model) -> str:
    """Why a confounded batch makes a cell type descriptive, and what can be done."""
    batch = inputs.batch
    doses = dose_text(inputs.levels[model.confounded], inputs.axis.unit)
    reason = (
        f"the difference of {doses} from the vehicle is also a difference between "
        f"values of {batch}"
    )
    fixed = bool(np.any(model.estimable & ~model.confounded))
    if inputs.batch_modelled:
        reason += (
            f", which the random intercept of {batch} cannot separate unless the "
            f"vehicle and each of these doses span two or more values of {batch}"
        )
        if fixed:
            reason += f"; add {batch} as a fixed term instead, which sets them aside"
        return reason
    ways = []
    if model.batch_spread:
        ways.append(f"as a random intercept, random='(1 | {batch})'")
    if fixed:
        ways.append("as a fixed term, which sets these doses aside")
    if ways:
        reason += f"; add {batch} to the design " + ", or ".join(ways)
    return reason


def _outside_span(base: FloatArray, columns: FloatArray) -> np.ndarray:
    """Whether each column lies outside the span of ``base`` and the other columns:
    whether R's ``qr()`` keeps it when it comes last, so that its coefficient is
    estimable whatever the order of the columns."""
    found = np.zeros(columns.shape[1], dtype=bool)
    for k in range(columns.shape[1]):
        x = np.column_stack([base, np.delete(columns, k, axis=1), columns[:, k]])
        found[k] = x.shape[1] - 1 not in column_space(x).aliased
    return found


def _batch_columns(labels: list[str]) -> FloatArray:
    """Columns that, beside an intercept, span the batches of the samples used:
    contrasts of the batches that sum to zero. Batches of a single sample are pooled
    into one first; a sample alone in its batch, the only one, is 0 in every column,
    so that it joins no batch and its batch absorbs nothing."""
    sizes = Counter(labels)
    pooled = [label if sizes[label] > 1 else None for label in labels]
    alone = sum(size == 1 for size in sizes.values()) == 1
    batches = [b for b in dict.fromkeys(pooled) if b is not None or not alone]
    member = np.array([[p == b for b in batches] for p in pooled], dtype=np.float64)
    contrasts: FloatArray = member[:, :-1] - member[:, -1:]
    return contrasts


def _levels_per_group(
    labels: Sequence[Hashable], groups: IntArray, kept: IntArray
) -> IntArray:
    """The distinct labels among the samples of each dose group."""
    found: list[set[Hashable]] = [set() for _ in kept]
    for label, group in zip(labels, groups.tolist(), strict=True):
        found[group].add(label)
    return np.array([len(distinct) for distinct in found], dtype=np.intp)


def _relations(inputs: _Inputs, x: FloatArray, n_others: int, treated: IntArray) -> str:
    """How the dose columns that R's ``qr()`` pivots out of ``x`` depend on the
    columns before them, as ``Design.check`` writes it."""
    design, axis = inputs.design, inputs.axis
    model = replace(
        design.model_matrix,
        matrix=x,
        columns=(*design.columns, "(Intercept)", *_group_names(inputs, treated)),
        assign=(*design.assign, *[0] * (1 + treated.size)),
        coding=(
            *design.model_matrix.coding,
            (),
            *[((axis.column, f"{inputs.levels[k]:g}"),) for k in treated],
        ),
    )
    aliasing = _aliasing(x)
    relations = [
        _relation(model, aliasing, j) for j in aliasing.aliased if j >= n_others
    ]
    return listing(relations, separator="; ")


def _unit_per_dose_group(
    design: Design, rows: IntArray, groups: IntArray
) -> str | None:
    """The grouping of a random intercept of which every dose group holds the
    samples of a single level, with two levels or more and a level of two samples
    or more; or None."""
    for effect in design.random:
        keys = _levels(design, effect, rows)
        if not 1 < len(set(keys)) < len(keys):
            continue
        single = all(
            len({key for key, k in zip(keys, groups) if k == group}) == 1
            for group in set(groups.tolist())
        )
        if single:
            return effect.group
    return None


def _group_names(inputs: _Inputs, groups: IntArray) -> list[str]:
    """Names of dose groups for messages, such as "dose 30 ug/kg"."""
    axis = inputs.axis
    return [f"{axis.column} {inputs.levels[k]:g} {axis.unit}" for k in groups]


def _count(number: int, noun: str) -> str:
    """A count with its noun, as in "1 sample" and "2 samples"."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


# Tables.


def _summary_table(checks: list[_CellType], minimums: Table | None) -> Table:
    models = [check.model for check in checks]
    columns: dict[str, Any] = {
        "group_id": text_column([check.group_id for check in checks]),
        "status": text_column([check.status for check in checks]),
        "reason": text_column([check.reason for check in checks]),
        "n_dose_groups": np.array(
            [np.count_nonzero(model.kept) for model in models], dtype=np.int64
        ),
        "n_dose_groups_dropped": np.array(
            [np.count_nonzero(model.kept == 0) for model in models], dtype=np.int64
        ),
        "n_dose_groups_estimable": np.array(
            [np.count_nonzero(model.estimable) for model in models], dtype=np.int64
        ),
        "n_samples_used": np.array(
            [np.count_nonzero(check.used) for check in checks], dtype=np.int64
        ),
        "n_samples_left_out": np.array(
            [np.count_nonzero(~check.used) for check in checks], dtype=np.int64
        ),
        "df_residual": np.array([model.df_residual for model in models]),
    }
    if minimums is not None:
        columns["rules_status"] = text_column(
            [_rules_status(minimums, check.group_id) for check in checks]
        )
    return Table(columns)


def _dose_table(inputs: _Inputs, checks: list[_CellType]) -> Table:
    k, level_of = inputs.levels.size, inputs.level_of
    columns: dict[str, Any] = {
        "group_id": text_column([check.group_id for check in checks for _ in range(k)]),
        "dose": np.tile(inputs.levels, len(checks)),
        "n_samples_used": np.concatenate([check.model.kept for check in checks]).astype(
            np.int64
        ),
        "n_samples_left_out": np.concatenate(
            [np.bincount(level_of[~check.used], minlength=k) for check in checks]
        ).astype(np.int64),
    }
    if checks[0].cells is not None:
        columns["n_cells_min"] = np.concatenate(
            [_group_min(check.cells, check, level_of) for check in checks]
        )
    if checks[0].totals is not None:
        columns["total_counts_min"] = np.concatenate(
            [_group_min(check.totals, check, level_of) for check in checks]
        )
    columns["estimable"] = np.concatenate([check.model.estimable for check in checks])
    if inputs.batches is not None:
        columns["confounded_with_batch"] = np.concatenate(
            [check.model.confounded for check in checks]
        )
    return Table(columns)


def _group_min(values: Any, check: _CellType, level_of: IntArray) -> FloatArray:
    """The fewest value of each dose group over the samples used, NaN where a group
    keeps none."""
    found = np.full(check.model.kept.size, np.inf)
    used = check.used
    np.minimum.at(found, level_of[used], np.asarray(values, dtype=np.float64)[used])
    found[check.model.kept == 0] = np.nan
    return found


def _sample_table(inputs: _Inputs, checks: list[_CellType]) -> Table:
    ids = inputs.sample_ids
    columns: dict[str, Any] = {
        "group_id": text_column([check.group_id for check in checks for _ in ids]),
        "sample_id": text_column(ids * len(checks)),
        "dose": np.tile(inputs.doses, len(checks)),
    }
    cells = [check.cells for check in checks if check.cells is not None]
    if cells:
        columns["n_cells"] = np.concatenate(cells)
    totals = [check.totals for check in checks if check.totals is not None]
    if totals:
        columns["total_counts"] = np.concatenate(totals)
    columns |= {
        "used": np.concatenate([check.used for check in checks]),
        "reason": text_column([reason for check in checks for reason in check.reasons]),
    }
    return Table(columns)


# ETAP's design minimums, checked on plain inputs: the dose and sex of each design
# sample, and the samples each cell type uses, which are those that pass.


class _Minimum(NamedTuple):
    """One minimum of a rule set, met or not."""

    group_id: str | None
    sex: str | None
    minimum: str
    section: str
    required: str
    observed: str
    met: bool
    when_unmet: str


def _etap_2024_minimums(
    doses: FloatArray,
    sexes: list[str],
    used_by_group: Mapping[str, np.ndarray],
    unit: str,
) -> Table:
    """ETAP's minimums: one row per minimum of the design, then per cell type and sex.

    Args:
        doses: The dose of each design sample.
        sexes: The sex of each design sample, as a label.
        used_by_group: For each cell type, whether each design sample passes there.
        unit: The doses' unit, for the text of the rows.
    """
    rows = _etap_design_minimums(doses, sexes, unit)
    for group_id, used in used_by_group.items():
        rows += _etap_cell_type_minimums(group_id, doses, sexes, used, unit)
    return Table(
        {
            "group_id": text_column([row.group_id for row in rows]),
            "sex": text_column([row.sex for row in rows]),
            "minimum": text_column([row.minimum for row in rows]),
            "section": text_column([row.section for row in rows]),
            "required": text_column([row.required for row in rows]),
            "observed": text_column([row.observed for row in rows]),
            "met": np.array([row.met for row in rows], dtype=bool),
            "when_unmet": text_column([row.when_unmet for row in rows]),
        }
    )


def _rules_status(minimums: Table, group_id: str) -> str:
    """Deviation when a minimum of the design, or a vehicle minimum of the cell type,
    is unmet; dose levels that the rule set excludes do not count."""
    mine = np.array([g is None or g == group_id for g in minimums["group_id"].tolist()])
    unmet = mine & ~minimums["met"] & (minimums["when_unmet"] != LEVEL_EXCLUDED)
    return DEVIATION if unmet.any() else MET


def _excluded_levels(minimums: Table, group_id: str) -> str:
    """The dose levels that the rule set excludes in a cell type, as "doses short of
    2 passing samples (M: 1 at 1 mg/kg)", or "" when it excludes none."""
    rows = (
        (minimums["group_id"] == group_id)
        & ~minimums["met"]
        & (minimums["when_unmet"] == LEVEL_EXCLUDED)
    )
    if not rows.any():
        return ""
    short = [
        f"{sex}: {observed}"
        for sex, observed in zip(
            minimums["sex"][rows].tolist(), minimums["observed"][rows].tolist()
        )
    ]
    return f"doses short of {ETAP_PASSING_SAMPLES} passing samples ({'; '.join(short)})"


def _etap_design_minimums(
    doses: FloatArray, sexes: list[str], unit: str
) -> list[_Minimum]:
    """The minimums of Sec. 3.4.1.3 and 3.4.2 that the doses and sexes of the design's
    samples show: the dose levels, their spacing, the sexes and the animals of each
    sex in each dose group."""
    levels = np.unique(doses)
    positive = levels[levels > 0.0]
    vehicle = bool(levels[0] == 0.0)
    rows = [
        _Minimum(
            None,
            None,
            "dose_levels",
            "3.4.1.3",
            f"{ETAP_DOSES} doses and a vehicle",
            f"{positive.size} dose{'' if positive.size == 1 else 's'} and "
            f"{'a' if vehicle else 'no'} vehicle",
            positive.size >= ETAP_DOSES and vehicle,
            DESIGN_DEVIATION,
        )
    ]
    spacing = (
        f"steps of {ETAP_STEP:g} log10 down from the highest dose, the lowest step "
        f"{ETAP_LOWEST_STEP:g} log10, each within {ETAP_STEP_TOLERANCE:g}"
    )
    if positive.size < 2:
        observed, met = "fewer than two doses", False
    else:
        steps = np.diff(np.log10(positive))  # from the lowest dose up
        observed = (
            f"steps of {', '.join(f'{step:.2f}' for step in steps[::-1])} log10 down "
            "from the highest dose"
        )
        met = bool(
            abs(steps[0] - ETAP_LOWEST_STEP) <= ETAP_STEP_TOLERANCE
            and np.all(np.abs(steps[1:] - ETAP_STEP) <= ETAP_STEP_TOLERANCE)
        )
    rows.append(
        _Minimum(
            None,
            None,
            "dose_spacing",
            "3.4.1.3",
            spacing,
            observed,
            met,
            DESIGN_DEVIATION,
        )
    )
    distinct = sorted(set(sexes))
    rows.append(
        _Minimum(
            None,
            None,
            "sexes",
            "3.4.2",
            "male and female",
            f"{len(distinct)} ({', '.join(distinct)})",
            len(distinct) >= 2,
            DESIGN_DEVIATION,
        )
    )
    sex_of = np.array(sexes, dtype=object)
    for sex in distinct:
        counts = [int(np.sum((doses == level) & (sex_of == sex))) for level in levels]
        rows.append(
            _Minimum(
                None,
                sex,
                "animals_per_sex_and_dose",
                "3.4.2",
                f"{ETAP_ANIMALS} in each dose group",
                _shortfalls(levels, counts, ETAP_ANIMALS, unit, "in each dose group"),
                min(counts) >= ETAP_ANIMALS,
                DESIGN_DEVIATION,
            )
        )
    return rows


def _etap_cell_type_minimums(
    group_id: str, doses: FloatArray, sexes: list[str], used: np.ndarray, unit: str
) -> list[_Minimum]:
    """The minimums of Sec. 3.4.5.2 in one cell type, for each sex: passing samples
    at each dose level and passing vehicle samples."""
    levels = np.unique(doses)
    positive = levels[levels > 0.0]
    sex_of = np.array(sexes, dtype=object)
    rows = []
    for sex in sorted(set(sexes)):
        passing = used & (sex_of == sex)
        counts = [int(np.sum(passing & (doses == level))) for level in positive]
        rows.append(
            _Minimum(
                group_id,
                sex,
                "passing_samples_per_dose",
                "3.4.5.2",
                f"{ETAP_PASSING_SAMPLES} at each dose",
                _shortfalls(
                    positive, counts, ETAP_PASSING_SAMPLES, unit, "at each dose"
                ),
                min(counts, default=ETAP_PASSING_SAMPLES) >= ETAP_PASSING_SAMPLES,
                LEVEL_EXCLUDED,
            )
        )
        vehicle = int(np.sum(passing & (doses == 0.0)))
        rows.append(
            _Minimum(
                group_id,
                sex,
                "passing_vehicle_samples",
                "3.4.5.2",
                f"{ETAP_VEHICLE_SAMPLES} passing vehicle samples",
                f"{vehicle}",
                vehicle >= ETAP_VEHICLE_SAMPLES,
                NOT_MODELLED,
            )
        )
    return rows


def _shortfalls(
    doses: FloatArray, counts: list[int], required: int, unit: str, where: str
) -> str:
    """The doses whose count falls short, as "3 at 1 ug/kg, 2 at 3 ug/kg", or the
    fewest count when none does, as "at least 4 in each dose group"."""
    short = [
        f"{count} at {dose:g} {unit}"
        for dose, count in zip(doses, counts, strict=True)
        if count < required
    ]
    if short:
        return ", ".join(short)
    if not counts:
        return "no doses"
    return f"at least {min(counts)} {where}"
