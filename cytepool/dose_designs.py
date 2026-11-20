"""Whether a dose series supports tests in each cell type, checked before anything is
fitted.

A dose analysis compares each dose with the vehicle, the samples given no dose, and
judges each difference against the variation between replicates given the same dose.
:func:`check_dose_design` says what each cell type (group) of a cube supports, and
returns a :class:`DoseDesignReport`
(https://docs.nygen.io/CytePool/statistics/dose-response#dose-design-check):

1. Samples. A design sample enters a cell type as it enters the cell type's fit in
   limma_voom: with at least ``min_cells`` cells there, when the cube records cells,
   some counts, and at least ``min_counts`` counts over every feature. Filtering can
   leave a dose group without a sample in some cell types, as dreamlet's
   documentation warns for the time points of an ordered covariate (Hoffman et al.
   2026).
2. Dose groups. The distinct doses of the design's samples, the vehicle (dose 0)
   first. In a cell type, a dose group that keeps no sample is dropped.
3. The dose-group model. The design's terms, an intercept, and an indicator of each
   treated dose group that keeps a sample, on the samples used. Its rank, by R's
   ``qr()`` rule, gives the residual degrees of freedom. A dose group is estimable,
   comparable with the vehicle apart from the design's other terms, when its column
   lies outside the span of the other columns (Searle 1971), which R's ``qr()`` and
   limma's ``nonEstimable`` report with the column placed last. Random intercepts
   are checked on these samples as ``Design.check`` checks them.
4. The batch. A dose group is confounded with a processing batch, such as a lane or
   a plate, when its difference from the vehicle is also a difference between
   batches: it is estimable, and not once an indicator of each batch joins the
   model. Batches of a single sample are pooled into one first, since such a batch
   cannot be told apart from its sample's own deviation. A batch that coincides with
   the condition breaks the control of false positives, which summing the counts
   within each batch restores (Lun and Marioni 2017).

Statuses:
    - ``"refused"``: no dose analysis. Every sample was left out; the design, or the
      cell type, has no vehicle sample; no treated sample is left; no treated dose
      group is estimable; or the random intercepts cannot be estimated beside the
      dose groups.
    - ``"descriptive"``: curves that describe the means of the dose groups, without
      p-values or bounds. No estimable treated dose group keeps two samples, so no
      dose has a replicate; the model leaves no residual degrees of freedom; or a
      batch that the design does not model is confounded with an estimable dose
      group.
    - ``"inferential"``: tests with p-values, and curves with bounds.

ETAP's design minimums:
    With ``rules="etap_2024"``, the report lists the design minimums of EPA's
    Transcriptomic Assessment Products (US EPA 2024) that samples show: at least
    eight doses and a vehicle, spaced by half a log10 down from the highest dose, the
    lowest a full log10 below the second lowest (Sec. 3.4.1.3); at least four animals
    of each sex in each dose group (Sec. 3.4.2); and, per cell type and sex, at least
    two passing samples at each dose level and three passing vehicle samples (Sec.
    3.4.5.2), passing being the cell type's filter of samples. A step counts as half a
    log10 within 0.05 of 0.5, and as a full log10 within 0.05 of 1. ETAP excludes a
    dose level with fewer than two passing samples and goes on, so such a level is
    listed without changing the rules' status; any other unmet minimum makes the
    status a deviation.

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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._analysis_records import matched_samples, only_key
from ._columns import missing_mask, text_column
from ._doses import column_doses, distinct_doses
from ._group_fits import (
    COUNTS_LAYER,
    amount,
    group_positions,
    left_out,
    min_cells_setting,
    min_counts_setting,
)
from ._linear_algebra import column_space
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import checked_choice
from .cube import Cube
from .design import Design
from .design.checks import DesignReport, _aliasing, _relation
from .dose_axes import DoseAxis
from .normalization import library_sizes
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
      is; ``n_dose_groups``, the dose groups a dose analysis can use, the vehicle and
      the treated dose groups it can compare with the vehicle; ``n_dose_groups_dropped``,
      the dose groups that keep no sample; ``n_samples_used`` and
      ``n_samples_left_out``; ``df_residual``, the residual degrees of freedom of the
      dose-group model, NaN when no sample is used; and, with a rule set,
      ``rules_status``, ``"met"`` or ``"deviation"``.
    - ``doses``: ``group_id``; ``dose``, in the axis's unit; ``n_samples_used`` and
      ``n_samples_left_out``; ``n_cells_min``, when the cube records cells, and
      ``total_counts_min``, over the samples used (NaN for a group that keeps no
      sample); ``estimable``, whether the design can compare the treated dose group
      with the vehicle apart from its other terms (False for the vehicle, the
      reference, and for groups that keep no sample); and, with a batch,
      ``confounded_with_batch``, whether the group's difference from the vehicle is
      also a difference between batches (False for groups that are not estimable).
      A design that has the batch as a random intercept compares such groups against
      the variance between batches, so they do not make the cell type descriptive.
    - ``samples``: ``group_id``; ``sample_id``; ``dose``; ``n_cells``, when the cube
      records cells; ``total_counts``, the counts over every feature, which
      ``min_counts`` reads; ``used``; and ``reason``, why the sample was left out,
      empty when used.
    - ``minimums``: with a rule set, one row per minimum, and None without:
      ``group_id``, None for a minimum of the design; ``sex``, None for a minimum of
      the doses; ``minimum``, ``"dose_levels"``, ``"dose_spacing"``, ``"sexes"``,
      ``"animals_per_sex_and_dose"``, ``"passing_samples_per_dose"`` or
      ``"passing_vehicle_samples"``; ``section``, of the rule set's document;
      ``required`` and ``observed``, as text; ``met``; and ``consequence``, what the
      rule set does when the minimum is unmet.

    ``design_check`` is :meth:`Design.check` of the design given, and ``settings``
    the settings of the check, the dose axis as :meth:`DoseAxis.to_dict` gives it.
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
        """One row per cell type: its status, why, its dose groups and samples."""
        return self._summary

    @property
    def doses(self) -> Table:
        """One row per cell type and dose group: samples, cells, counts, estimable."""
        return self._doses

    @property
    def samples(self) -> Table:
        """One row per cell type and design sample: used or left out, and why."""
        return self._samples

    @property
    def minimums(self) -> Table | None:
        """One row per minimum of the rule set checked, or None without one."""
        return self._minimums

    @property
    def design_check(self) -> DesignReport:
        """:meth:`Design.check` of the design given."""
        return self._design_check

    @property
    def settings(self) -> dict[str, Any]:
        """The settings of the check, as a new dict: ``dose``, the axis as
        :meth:`DoseAxis.to_dict` gives it, which ``DoseAxis(**settings["dose"])``
        rebuilds; ``batch``, ``rules``, ``sex``, ``min_cells``, ``min_counts`` and
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
            lines.append(self._cell_type_line(row))
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

    def _cell_type_line(self, row: int) -> str:
        summary = self._summary
        group = summary["group_id"][row]
        parts = [str(summary["status"][row])]
        samples = self._doses.filter(group_id=group)["n_samples_used"]
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
        if "rules_status" in summary:
            parts.append(f"rules: {summary['rules_status'][row]}")
        return f"  {group}: " + "; ".join(parts)


def check_dose_design(
    cube: Cube,
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
    """Check what a dose series supports in every cell type of a cube.

    For every cell type (group), before anything is fitted: the samples that the
    cell type's fit in :func:`~cytepool.limma_voom` would use, with the same
    ``min_cells`` and ``min_counts``; the samples, cells and counts each dose group
    keeps; the dose groups that the design's other terms, or a processing batch,
    make impossible to compare with the vehicle; the residual degrees of freedom;
    and a status, inferential, descriptive or refused, with the reason (the module
    documentation gives the rules). With ``rules="etap_2024"``, the design minimums
    of EPA's Transcriptomic Assessment Products that the design misses (US EPA
    2024). https://docs.nygen.io/CytePool/statistics/dose-response#dose-design-check
    explains the check.

    Args:
        cube: A cube with a ``counts`` layer, as :func:`~cytepool.limma_voom` takes.
        design: A :class:`~cytepool.Design` of the terms other than dose, such as
            ``Design(cube.samples, "~ 1")`` or ``"~ sex"``, with random intercepts if
            the analysis has them, whose sample table has a ``sample_id`` column with
            cube sample ids and the dose column; its rows may be some of the cube's
            samples, in any order. The check adds the dose groups itself, so a design
            whose terms use the dose column or the axis's columns
            (``{column}_treated``, ``{column}_s1``, ...) is refused.
        dose: The :class:`~cytepool.DoseAxis` whose ``column`` holds each sample's
            dose, 0 for the vehicle. The dose groups are the distinct doses of the
            design's samples.
        batch: A column of the design's samples that names each sample's processing
            batch, such as a lane, plate or run, read as labels whatever their type;
            None for none.
        rules: ``"etap_2024"`` to list the design minimums of EPA's Transcriptomic
            Assessment Products that the design misses, or None.
        sex: With ``rules``, and only then: the column of the design's samples that
            holds each sample's sex; a study of one sex has a single value there.
        min_cells: The fewest cells a sample needs in a cell type, as in
            :func:`~cytepool.limma_voom`. Ignored when the cube does not record cell
            numbers.
        min_counts: The fewest counts, summed over every feature, a sample needs in
            a cell type, as in :func:`~cytepool.limma_voom`.
        groups: A group id or a list of them; all groups by default.

    Returns:
        The tables of the cell types, their dose groups, samples and, with
        ``rules``, the minimums, with :meth:`Design.check` of the design.

    Raises:
        TypeError: If ``cube`` is not a Cube, ``design`` not a Design, ``dose`` not a
            DoseAxis, an option has the wrong type, the dose column does not hold
            real numbers, or ``groups`` or ``min_cells`` is or holds a numpy masked
            array.
        ValueError: If the design has random effects that are not random
            intercepts, has no ``sample_id`` column, names samples the cube does not
            have or names a sample twice; if its terms use the dose column or the
            axis's columns; if the dose, ``batch`` or ``sex``
            column is missing or holds a missing value; if a dose is not finite or is
            negative, or two doses differ by less than 1e-8 of the dose; if
            ``rules`` names no rule set, ``rules`` is given without ``sex`` or
            ``sex`` without ``rules``; or if an option is out of range.
        KeyError: If a group is not in the cube.
    """
    inputs = _checked_inputs(
        cube, design, dose, batch, rules, sex, min_cells, min_counts, groups
    )
    checks = [_check_cell_type(inputs, g) for g in inputs.positions]
    minimums = None
    if inputs.sexes is not None:  # sex comes with rules
        minimums = _etap_design_minimums(inputs.doses, inputs.sexes, dose.unit)
        for check in checks:
            minimums += _etap_cell_type_minimums(
                check.group_id, inputs.doses, inputs.sexes, check.used, dose.unit
            )
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
        _summary_table(inputs, checks, minimums),
        _dose_table(inputs, checks),
        _sample_table(inputs, checks),
        None if minimums is None else _minimum_table(minimums),
        design.check(),
        settings,
    )


# The inputs.


@dataclass(frozen=True)
class _Inputs:
    """The checked inputs, with the design's samples in cube order."""

    cube: Cube
    design: Design
    axis: DoseAxis
    positions: list[int]  # the cube's groups to check
    cube_rows: IntArray  # the cube position of each design sample, increasing
    model_rows: IntArray  # the design row of each
    doses: FloatArray  # the dose of each
    levels: FloatArray  # the distinct doses, ascending
    level_of: IntArray  # the dose group of each design sample
    batch: str | None  # the batch column
    batches: list[str] | None  # the batch of each, as a label
    batch_modelled: bool  # the batch groups a random intercept of the design
    sexes: list[str] | None  # the sex of each, as a label
    min_cells: int
    min_counts: float


def _checked_inputs(
    cube: Any,
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
    if not isinstance(cube, Cube):
        raise TypeError(f"cube must be a Cube, got {type(cube).__name__}")
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
    if design.random:
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
    min_cells, min_counts = min_cells_setting(min_cells), min_counts_setting(min_counts)
    positions = group_positions(cube, groups)
    cube_rows, model_rows = matched_samples(
        cube,
        design,
        no_ids="the design's sample table has no 'sample_id' column, so its rows "
        "cannot be matched to the cube's samples; build the Design from "
        "cube.samples, or from a table whose 'sample_id' column holds cube sample ids",
        text_ids="the cube's does",
        data="the cube",
        data_ids="the cube's sample ids",
    )
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
    return _Inputs(
        cube=cube,
        design=design,
        axis=axis,
        positions=positions,
        cube_rows=cube_rows,
        model_rows=model_rows,
        doses=doses,
        levels=levels,
        level_of=np.searchsorted(levels, doses),
        batch=batch,
        batches=batches,
        batch_modelled=any(effect.grouping == (batch,) for effect in design.random),
        sexes=sexes,
        min_cells=min_cells,
        min_counts=min_counts,
    )


def _refuse_dose_terms(design: Design, axis: DoseAxis) -> None:
    """Refuse a design whose fixed or random terms use the dose column or the
    columns of the axis, since the check adds the dose groups itself."""
    names = list(design.model_matrix.variables)
    for effect in design.random:
        names += [*effect.effects.variables, *effect.grouping]
    own = re.compile(rf"{re.escape(axis.column)}(_treated|_s\d+)?")
    used = sorted({name for name in names if own.fullmatch(name)})
    if used:
        raise ValueError(
            f"the design uses {quoted(used)}, but check_dose_design adds the dose "
            "groups itself; give the design of the other terms, such as "
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


# One cell type.


@dataclass(frozen=True)
class _CellType:
    """What one cell type keeps and supports."""

    group_id: str
    used: np.ndarray  # bool, per design sample in cube order
    reasons: list[str]  # why each design sample was left out; "" when used
    cells: np.ndarray | None  # cells of each design sample, when recorded
    totals: FloatArray  # counts of each design sample over every feature
    estimable: np.ndarray  # bool, per dose group
    confounded: np.ndarray  # bool, per dose group
    df_residual: float
    status: str
    reason: str


def _check_cell_type(inputs: _Inputs, g: int) -> _CellType:
    cube, levels, level_of = inputs.cube, inputs.levels, inputs.level_of
    counts = cube.layer(COUNTS_LAYER, g)[inputs.cube_rows]
    cells = None if cube.n_cells is None else cube.n_cells[inputs.cube_rows, g]
    reasons = left_out(counts, cells, inputs.min_cells, inputs.min_counts)
    used = np.array([not reason for reason in reasons], dtype=bool)
    totals = library_sizes(counts) if counts.size else np.zeros(used.size)
    # Filled in place as they are found; False where nothing was found.
    estimable = np.zeros(levels.size, dtype=bool)
    confounded = np.zeros(levels.size, dtype=bool)
    found = _CellType(
        str(cube.groups["group_id"][g]),
        used,
        reasons,
        cells,
        totals,
        estimable,
        confounded,
        float("nan"),
        REFUSED,
        "",
    )
    n_used = int(used.sum())
    kept_per_group = np.bincount(level_of[used], minlength=levels.size)
    if n_used == 0:
        rules = (
            f"fewer than {amount(inputs.min_counts)} counts"
            if inputs.min_counts
            else "no counts"
        )
        if cells is not None:
            rules = f"fewer than {inputs.min_cells} cells, or {rules}"
        reason = f"every one of the {used.size} samples was left out ({rules})"
        return replace(found, reason=reason)

    # The design's terms, an intercept, and the treated dose groups that keep a
    # sample, the vehicle being the reference.
    treated = [k for k in range(levels.size) if levels[k] > 0.0 and kept_per_group[k]]
    groups_used = level_of[used]
    others = np.column_stack(
        [inputs.design.matrix[inputs.model_rows][used], np.ones(n_used)]
    )
    indicators = (groups_used[:, None] == np.array(treated, dtype=np.intp)).astype(
        np.float64
    )
    x = np.column_stack([others, indicators])
    space = column_space(x)
    found = replace(found, df_residual=float(n_used - len(space.kept)))
    vehicle_kept = levels[0] == 0.0 and kept_per_group[0] > 0
    if levels[0] != 0.0:
        reason = (
            "the design has no sample of dose 0, the vehicle, which every dose "
            "analysis compares the doses with; compare the doses with each other "
            "with limma_voom"
        )
        return replace(found, reason=reason)
    if not vehicle_kept:
        n_vehicle = int(np.count_nonzero(level_of == 0))
        reason = (
            f"no vehicle sample is left ({_count(n_vehicle, 'vehicle sample')} left "
            "out); compare the doses with each other with limma_voom"
        )
        return replace(found, reason=reason)
    if not treated:
        n_treated = int(np.count_nonzero(level_of > 0))
        reason = (
            f"no treated sample is left ({_count(n_treated, 'treated sample')} left "
            "out)"
            if n_treated
            else "the design has no treated sample, only the vehicle"
        )
        return replace(found, reason=reason)
    comparable = _outside_span(others, indicators)
    estimable[treated] = comparable
    if not comparable.any():
        relations = _relations(inputs, others, indicators, treated)
        reason = (
            "no treated dose group can be compared with the vehicle apart from the "
            "design's other terms: " + listing(relations, separator="; ")
        )
        return replace(found, reason=reason)
    if inputs.batches is not None:
        labels = [inputs.batches[i] for i in np.flatnonzero(used)]
        batch_columns = _pooled_batch_columns(labels)
        if batch_columns.shape[1]:
            with_batch = np.column_stack([others, batch_columns])
            outside = _outside_span(with_batch, indicators)
            confounded[treated] = comparable & ~outside
    if inputs.design.random:
        rows = inputs.model_rows[used]
        names = (*inputs.design.columns, "(Intercept)", *_group_names(inputs, treated))
        kept = list(space.kept)
        try:
            mixed_design_of_samples(
                inputs.design,
                rows,
                x[:, kept],
                tuple(names[j] for j in kept),
            )
        except ValueError as error:
            unit = _unit_per_dose_group(inputs.design, rows, groups_used)
            if unit is None:
                return replace(found, reason=str(error))
            reason = (
                f"each dose group holds the samples of a single {unit}, so no dose "
                f"has a second {unit} to replicate it and the random intercept of "
                f"{unit} cannot be estimated beside the dose groups; pool the samples "
                f"of each {unit} at each dose, with cube.collapse_samples({unit!r}) "
                f"when each {unit} received one dose, and check the pooled cube"
            )
            return replace(found, reason=reason)

    if not np.any(kept_per_group[treated][comparable] >= 2):
        reason = (
            "no treated dose group that can be compared with the vehicle keeps a "
            "second sample, so no dose has a replicate"
        )
        return replace(found, status=DESCRIPTIVE, reason=reason)
    if found.df_residual == 0:
        reason = (
            f"no residual degrees of freedom: the model has rank {len(space.kept)} on "
            f"the {n_used} samples left, so it fits them exactly"
        )
        return replace(found, status=DESCRIPTIVE, reason=reason)
    if confounded.any() and not inputs.batch_modelled:
        batch = inputs.batch
        doses = _dose_list(levels[confounded], inputs.axis.unit)
        reason = (
            f"the difference of {doses} from the vehicle is also a difference between "
            f"values of {batch}; add {batch} to the design, as a random intercept, "
            f"random='(1 | {batch})', when each dose spans two or more values of it, "
            "or as a fixed term, which sets the confounded doses aside"
        )
        return replace(found, status=DESCRIPTIVE, reason=reason)
    return replace(found, status=INFERENTIAL)


def _outside_span(base: FloatArray, columns: FloatArray) -> np.ndarray:
    """Whether each column lies outside the span of ``base`` and the other columns:
    whether R's ``qr()`` keeps it when it comes last, so that its coefficient is
    estimable whatever the order of the columns."""
    found = np.zeros(columns.shape[1], dtype=bool)
    for k in range(columns.shape[1]):
        x = np.column_stack([base, np.delete(columns, k, axis=1), columns[:, k]])
        found[k] = x.shape[1] - 1 not in column_space(x).aliased
    return found


def _pooled_batch_columns(labels: list[str]) -> FloatArray:
    """Indicators of the batches of the samples used, but the first, with the
    batches of a single sample pooled into one."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    pooled = [label if counts[label] > 1 else None for label in labels]
    batches = list(dict.fromkeys(pooled))
    columns = np.zeros((len(labels), len(batches) - 1))
    for j, batch in enumerate(batches[1:]):
        columns[:, j] = [label == batch for label in pooled]
    return columns


def _relations(
    inputs: _Inputs, others: FloatArray, indicators: FloatArray, treated: list[int]
) -> list[str]:
    """How each treated dose group's column depends on the design's other columns,
    as ``Design.check`` writes it, with the group's column placed last."""
    design, axis = inputs.design, inputs.axis
    names = _group_names(inputs, treated)
    coding = [((axis.column, f"{inputs.levels[k]:g}"),) for k in treated]
    relations = []
    for position in range(len(treated)):
        order = [j for j in range(len(treated)) if j != position] + [position]
        x = np.column_stack([others, indicators[:, order]])
        model = replace(
            design.model_matrix,
            matrix=x,
            columns=(*design.columns, "(Intercept)", *[names[j] for j in order]),
            assign=(*design.assign, 0, *[0] * len(order)),
            coding=(*design.model_matrix.coding, (), *[coding[j] for j in order]),
        )
        relations.append(_relation(model, _aliasing(x), x.shape[1] - 1))
    return relations


def _unit_per_dose_group(
    design: Design, rows: IntArray, groups_used: IntArray
) -> str | None:
    """The grouping of a random intercept of which every dose group holds the
    samples of a single level, with two levels or more and a level of two samples
    or more; or None."""
    for effect in design.random:
        keys = list(
            zip(*(design.samples[name][rows].tolist() for name in effect.grouping))
        )
        if not 1 < len(set(keys)) < len(keys):
            continue
        single = all(
            len({key for key, k in zip(keys, groups_used) if k == group}) == 1
            for group in set(groups_used.tolist())
        )
        if single:
            return effect.group
    return None


def _group_names(inputs: _Inputs, groups: list[int]) -> list[str]:
    """Names of dose groups for messages, such as "dose 30 ug/kg"."""
    axis = inputs.axis
    return [f"{axis.column} {inputs.levels[k]:g} {axis.unit}" for k in groups]


def _dose_list(doses: FloatArray, unit: str) -> str:
    """Doses for messages, such as "10 and 30 ug/kg"."""
    return f"{listing([f'{dose:g}' for dose in doses])} {unit}"


def _count(number: int, noun: str) -> str:
    """A count with its noun, as in "1 sample" and "2 samples"."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


# ETAP's design minimums, from plain inputs: the doses and sexes of the design's
# samples, and the samples each cell type uses.


class _Minimum(NamedTuple):
    """One minimum of a rule set, met or not."""

    group_id: str | None
    sex: str | None
    minimum: str
    section: str
    required: str
    observed: str
    met: bool
    consequence: str


def _etap_design_minimums(
    doses: FloatArray, sexes: list[str], unit: str
) -> list[_Minimum]:
    """The minimums of US EPA (2024) Sec. 3.4.1.3 and 3.4.2 that the doses and sexes
    of the design's samples show: the dose levels, their spacing, the sexes and the
    animals of each sex in each dose group."""
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
            f"{_count(positive.size, 'dose')} and {'a' if vehicle else 'no'} vehicle",
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
        steps = np.diff(np.log10(positive))
        observed = (
            f"steps of {', '.join(f'{step:.2f}' for step in steps)} log10 from the "
            "lowest dose"
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
            "2",
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
    """The minimums of US EPA (2024) Sec. 3.4.5.2 in one cell type, for each sex:
    passing samples at each dose level and passing vehicle samples, the samples
    that the cell type uses being those that pass."""
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
                f"{ETAP_VEHICLE_SAMPLES}",
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


def _rules_status(rows: list[_Minimum], group_id: str) -> str:
    """Deviation when a minimum of the design, or a vehicle minimum of the cell
    type, is unmet; levels that the rule set excludes do not count."""
    unmet = [
        row
        for row in rows
        if not row.met
        and row.consequence != LEVEL_EXCLUDED
        and row.group_id in (None, group_id)
    ]
    return DEVIATION if unmet else MET


# Tables.


def _summary_table(
    inputs: _Inputs, checks: list[_CellType], minimums: list[_Minimum] | None
) -> Table:
    vehicle = inputs.levels[0] == 0.0
    used_groups = [np.unique(inputs.level_of[check.used]) for check in checks]
    columns: dict[str, Any] = {
        "group_id": text_column([check.group_id for check in checks]),
        "status": text_column([check.status for check in checks]),
        "reason": text_column([check.reason for check in checks]),
        "n_dose_groups": np.array(
            [
                int(vehicle and 0 in kept) + int(check.estimable.sum())
                for check, kept in zip(checks, used_groups, strict=True)
            ],
            dtype=np.int64,
        ),
        "n_dose_groups_dropped": np.array(
            [inputs.levels.size - kept.size for kept in used_groups], dtype=np.int64
        ),
        "n_samples_used": np.array(
            [int(check.used.sum()) for check in checks], dtype=np.int64
        ),
        "n_samples_left_out": np.array(
            [int((~check.used).sum()) for check in checks], dtype=np.int64
        ),
        "df_residual": np.array([check.df_residual for check in checks]),
    }
    if minimums is not None:
        columns["rules_status"] = text_column(
            [_rules_status(minimums, check.group_id) for check in checks]
        )
    return Table(columns)


def _dose_table(inputs: _Inputs, checks: list[_CellType]) -> Table:
    levels, level_of = inputs.levels, inputs.level_of
    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "group_id",
            "dose",
            "n_samples_used",
            "n_samples_left_out",
            "n_cells_min",
            "total_counts_min",
            "estimable",
            "confounded_with_batch",
        )
    }
    for check in checks:
        for k, dose in enumerate(levels.tolist()):
            members = level_of == k
            kept = members & check.used
            rows["group_id"].append(check.group_id)
            rows["dose"].append(dose)
            rows["n_samples_used"].append(int(kept.sum()))
            rows["n_samples_left_out"].append(int((members & ~check.used).sum()))
            cells = np.nan
            if check.cells is not None and kept.any():
                cells = float(check.cells[kept].min())
            rows["n_cells_min"].append(cells)
            rows["total_counts_min"].append(
                float(check.totals[kept].min()) if kept.any() else np.nan
            )
            rows["estimable"].append(bool(check.estimable[k]))
            rows["confounded_with_batch"].append(bool(check.confounded[k]))
    columns: dict[str, Any] = {
        "group_id": text_column(rows.pop("group_id")),
        **{name: np.array(values) for name, values in rows.items()},
    }
    columns["n_samples_used"] = columns["n_samples_used"].astype(np.int64)
    columns["n_samples_left_out"] = columns["n_samples_left_out"].astype(np.int64)
    columns["dose"] = columns["dose"].astype(np.float64)
    columns["n_cells_min"] = columns["n_cells_min"].astype(np.float64)
    columns["total_counts_min"] = columns["total_counts_min"].astype(np.float64)
    columns["estimable"] = columns["estimable"].astype(bool)
    columns["confounded_with_batch"] = columns["confounded_with_batch"].astype(bool)
    if inputs.cube.n_cells is None:
        del columns["n_cells_min"]
    if inputs.batches is None:
        del columns["confounded_with_batch"]
    return Table(columns)


def _sample_table(inputs: _Inputs, checks: list[_CellType]) -> Table:
    n = inputs.cube_rows.size
    ids = inputs.cube.samples["sample_id"][inputs.cube_rows].tolist()
    columns: dict[str, Any] = {
        "group_id": text_column([check.group_id for check in checks for _ in ids]),
        "sample_id": text_column(ids * len(checks)),
        "dose": np.tile(inputs.doses, len(checks)),
    }
    if inputs.cube.n_cells is not None:
        cells = [check.cells for check in checks if check.cells is not None]
        columns["n_cells"] = np.concatenate(cells)
    columns |= {
        "total_counts": np.concatenate([check.totals for check in checks]),
        "used": np.concatenate([check.used for check in checks]),
        "reason": text_column([reason for check in checks for reason in check.reasons]),
    }
    assert columns["used"].size == n * len(checks)
    return Table(columns)


def _minimum_table(rows: list[_Minimum]) -> Table:
    return Table(
        {
            "group_id": text_column([row.group_id for row in rows]),
            "sex": text_column([row.sex for row in rows]),
            "minimum": text_column([row.minimum for row in rows]),
            "section": text_column([row.section for row in rows]),
            "required": text_column([row.required for row in rows]),
            "observed": text_column([row.observed for row in rows]),
            "met": np.array([row.met for row in rows], dtype=bool),
            "consequence": text_column([row.consequence for row in rows]),
        }
    )
