"""Could the genes a dose series calls come from noise? A permutation check of the
gate at the series' own design.

:func:`~cytepool.permutation_check` permutes the dose labels of each cell type among
the samples that its design treats alike, runs :func:`~cytepool.dose_response` again
on each permuted labelling with the result's settings, and counts the genes that pass
the result's gate: how often the gate passes genes with the doses permuted, and a
permutation p-value that tests whether the true doses pass more
(https://docs.nygen.io/CytePool/statistics/dose-response#dose-permutation-check).

Exchangeable units:
    Under the null hypothesis that no gene responds to dose, samples that the design
    treats alike apart from their dose are exchangeable. Without random effects the
    units are the samples, in blocks of the same row of the design's matrix, its
    terms other than dose: a factor such as sex splits them, and a numeric covariate
    leaves each sample in a block of its own. With one random intercept, such as
    ``(1 | animal)``, units that each hold one dose are permuted whole, in blocks of
    units with the same rows, so that each run keeps the samples and units of every
    dose group; when units hold several doses, as donors whose cells receive every
    dose, the doses are permuted within each unit and block of rows.

References:
    Lee H, Stead JDH, Williams A, Cortés Ramírez SA, Atlas E, Mennigen JA, O'Brien JM,
    Yauk C (2024). Empirical characterization of false discovery rates of
    differentially expressed genes and transcriptomic benchmark concentrations in
    zebrafish embryos. Environmental Science & Technology 58, 6128-6137.
    doi:10.1021/acs.est.3c10543

    Bruns IB, Schultz DR, Demuynck E, et al. (2026). Analytical choices drive
    toxicogenomic potency estimates: a systematic evaluation of transcriptomic points
    of departure. Toxicological Sciences 209, kfag115. doi:10.1093/toxsci/kfag115

    Bourgon R, Gentleman R, Huber W (2010). Independent filtering increases detection
    power for high-throughput experiments. Proceedings of the National Academy of
    Sciences 107, 9546-9551. doi:10.1073/pnas.0914005107

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585
"""

import itertools
import json
import math
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ._analysis_records import absence, only_key, software_versions
from ._columns import text_column
from ._group_fits import group_positions
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing
from ._options import positive_integer, recorded_seed
from ._reruns import recorded_options
from .cube import Cube
from .design import Design
from .dose_axes import DoseAxis
from .dose_responses import (
    ANALYSES,
    DOSE_GROUPS,
    EACH_DOSE,
    GATE,
    MCP,
    SET_BY_DOSE_RESPONSE,
    DoseResponseResult,
    dose_response,
    gate_text,
)
from .table import Table

__all__ = ["PermutationCheck", "permutation_check"]

# The third word of each cell type's seed sequence, a stream of its own beside those
# of dose_response's multiple contrast tests (their positions, 0 to 2).
_STREAM = 1000
# The relative difference within which two gates' p-values are the same up to
# rounding: runs of one labelling on other machines, or with other tests beside the
# gate's, differ by about 1e-13.
_ROUNDING = 1e-9
# The statistics of a cell type's runs, NaN for a cell type not checked.
_STATISTICS = (
    "share_of_runs_with_passing_genes",
    "mean_passing_genes_per_run",
    "p_value",
    "smallest_possible_p_value",
)


class PermutationCheck:
    """How often the gate of a dose series passes genes with its doses permuted, and
    whether the true doses pass more, from :func:`permutation_check`.

    ``summary`` has one row per cell type asked for: ``group_id``; ``checked``;
    ``reason``, why a cell type was not checked, empty when checked;
    ``n_assignments``, the distinct assignments of the dose labels to the cell type's
    exchangeable units, as a float (the exact integer is in the provenance);
    ``exhaustive``, whether the runs were every other assignment; ``n_runs``;
    ``n_genes_passing_gate``, the genes the gate passes with the true doses (NaN
    where the gate did not run); ``share_of_runs_with_passing_genes``;
    ``mean_passing_genes_per_run``; ``p_value``, (1 + b) / (1 + B) for the b of the
    B runs that pass at least as many genes as the true doses (Phipson and Smyth
    2010); and ``smallest_possible_p_value``, the smallest p-value these runs can
    give, (1 + t) / (1 + B) for the t runs whose gate gives the true doses' p-values
    up to rounding and so passes the same genes whatever the data: runs that only
    rename dose groups that the gate's test cannot tell apart, such as dose groups
    of equal sizes under the dose-group test, and the reflection of doses that sit
    symmetrically on the axis under the spline. With every other assignment run, B
    is the number of assignments less one. Dunnett's test of three or more treated
    doses, whose p-values come from Monte Carlo integration, gives its renamings
    p-values apart by that error, so they do not count. Statistics of cell types not
    checked are NaN, and their ``n_runs`` 0.

    :meth:`table` gives each run of a cell type, with ``run`` and
    ``n_genes_passing_gate``. The provenance records each run's assignment, so that
    any run can be made again with :func:`~cytepool.dose_response` and the result's
    seed, which every run takes. Built by :func:`permutation_check`; immutable.

    Raises:
        TypeError: If ``summary`` is not a ``Table``, ``tables`` not a mapping of cell
            type to ``Table``, or ``provenance`` not a mapping that JSON can write.
        ValueError: If ``provenance`` holds a NaN or an infinity.
    """

    __slots__ = ("_summary", "_tables", "_record")

    _summary: Table
    _tables: Mapping[str, Table]
    _record: str

    def __init__(
        self,
        summary: Table,
        tables: Mapping[str, Table],
        provenance: Mapping[str, Any],
    ) -> None:
        if not isinstance(summary, Table):
            raise TypeError(f"summary must be a Table, got {type(summary).__name__}")
        if not isinstance(tables, Mapping):
            raise TypeError(
                "tables must be a mapping of cell type to Table, got "
                f"{type(tables).__name__}"
            )
        for key, table in tables.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"tables must be keyed by cell type, got the key {key!r}"
                )
            if not isinstance(table, Table):
                raise TypeError(
                    f"tables[{key!r}] must be a Table, got {type(table).__name__}"
                )
        if not isinstance(provenance, Mapping):
            raise TypeError(
                f"provenance must be a mapping, got {type(provenance).__name__}"
            )
        state = {
            "_summary": summary,
            "_tables": MappingProxyType(dict(tables)),
            # JSON holds no arrays: json.dumps raises TypeError for one.
            "_record": json.dumps(provenance, allow_nan=False),
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PermutationCheck is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("PermutationCheck is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            PermutationCheck,
            (self._summary, dict(self._tables), self.provenance),
        )

    @property
    def summary(self) -> Table:
        """One row per cell type, with whether it was checked, its assignments and
        runs, the genes the gate passes, how often its runs pass genes, and the
        p-value."""
        return self._summary

    @property
    def settings(self) -> dict[str, Any]:
        """The settings ``n_permutations``, ``groups`` as given, and ``seed``, the
        seed of the runs, given or drawn, as a new dict, also in the provenance."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the versions of the software; the settings; the result checked
        (``result``): its cube's fingerprint, its gate and its seed, which every run
        takes; and, for every cell type (``groups``), whether it was checked and why
        not, and the genes its gate passes with the true doses. A cell type whose
        units were read also records the kind of units permuted (``"samples"``,
        ``"whole units"`` or ``"doses within units"``) and the random effect's
        grouping (``unit``), the samples used, in the order of the fit, the doses,
        each sample's dose as an index into them (``labels``), the blocks of units,
        each unit given by its samples, and the number of assignments as an exact
        integer; a cell type checked, whether the runs were every other assignment,
        each run's assignment (``assignments``), each sample's dose index in the
        order of ``samples``, and the runs whose gate gives the true doses' p-values
        up to rounding (``tied_runs``, numbered from 1).
        """
        record: dict[str, Any] = json.loads(self._record)
        return record

    def table(self, group: str | None = None) -> Table:
        """Return the runs of a cell type: ``run``, from 1, and
        ``n_genes_passing_gate``.

        Args:
            group: A cell type that was checked; it may be left out when one cell
                type was asked for.

        Raises:
            ValueError: If ``group`` is left out and several cell types were asked
                for.
            TypeError: If ``group`` is not a string.
            KeyError: If the cell type was not asked for or not checked; the message
                says why.
        """
        groups = self._summary["group_id"].tolist()
        name = only_key(group, groups, "group", "cell types")
        if not isinstance(name, str):
            reject_masked(name, "group")
            raise TypeError(
                f"group must be a cell type (str), got {type(name).__name__}"
            )
        found = self._tables.get(name)
        if found is not None:
            return found
        if name not in groups:
            raise KeyError(
                f"no cell type {name!r} was asked for; cell types are: "
                f"{listing(groups) or '(none)'}{close_matches(name, groups)}"
            )
        reason = self._summary["reason"][groups.index(name)]
        raise KeyError(f"cell type {name!r} was not checked: {reason}")

    def __repr__(self) -> str:
        record = self.provenance
        summary = self._summary
        checked = int(np.count_nonzero(summary["checked"]))
        n = len(summary)
        lines = [
            f"PermutationCheck of the gate ({gate_text(record['result']['gate'])}): "
            f"{n:,} cell type{'' if n == 1 else 's'}, {checked:,} checked; seed "
            f"{record['settings']['seed']}"
        ]
        for row, entry in enumerate(record["groups"]):
            group = entry["group_id"]
            if not entry["checked"]:
                lines.append(f"  {group}: not checked: {entry['reason']}")
                continue
            lines.append(f"  {group}: " + _row_text(summary, row, entry))
        return fitted_lines("\n".join(lines))


def _row_text(summary: Table, row: int, entry: Mapping[str, Any]) -> str:
    """What the check found in a cell type, for the printout."""
    passing = int(summary["n_genes_passing_gate"][row])
    runs = int(summary["n_runs"][row])
    share = float(summary["share_of_runs_with_passing_genes"][row])
    count = entry["n_assignments"]
    size = f"{count:,}" if count < 10**6 else f"{float(count):.2g}".replace("e+", "e")
    over = (
        f"all {runs:,} other assignments"
        if entry["exhaustive"]
        else f"{runs:,} random run{'' if runs == 1 else 's'} of {size} assignments"
    )
    smallest = float(summary["smallest_possible_p_value"][row])
    p = float(summary["p_value"][row])
    with_genes = int(round(share * runs))
    text = (
        f"{passing:,} gene{' passes' if passing == 1 else 's pass'} the gate; over "
        f"{over}, the gate passed genes in {with_genes:,} run"
        f"{'' if with_genes == 1 else 's'} ({share:.2f}), "
        f"{float(summary['mean_passing_genes_per_run'][row]):.2g} genes per run; "
        f"p = {p:.2g}"
    )
    if p <= smallest:
        return text + ", the smallest these runs can give"
    if smallest > 0.05:
        return text + f"; the smallest these runs can give is {smallest:.2g}"
    return text


def permutation_check(
    result: DoseResponseResult,
    *,
    n_permutations: int = 100,
    groups: str | Sequence[str] | None = None,
    rng: np.random.Generator | int | None = None,
) -> PermutationCheck:
    """Check whether the true doses of a dose series pass more genes through its gate
    than permuted doses do.

    For each cell type, the dose labels of the samples it uses are permuted among the
    samples that the design treats alike (the module documentation gives the units),
    :func:`~cytepool.dose_response` runs again on that cell type with each permuted
    labelling and the result's settings and gate, and the genes that pass the gate
    are counted. The answer is the permutation p-value, (1 + b) / (1 + B) for the b
    of the B runs that pass at least as many genes as the true doses (Phipson and
    Smyth 2010). Under the null hypothesis that no gene responds to dose, every
    assignment of the labels is as likely as the true one, so a small p-value says
    that the true doses pass more genes than permuted ones, and so that not all of
    the gate's calls are noise; it does not say which calls are false. The summary
    also gives how often the gate passes genes with the doses permuted, given the
    cell type's own samples, which varies widely between series of one design. That
    share estimates how often noise alone passes genes only where no gene responds,
    as in vehicle-only data: responses scrambled over the permuted dose groups still
    pass in some runs.

    A design that allows at most ``n_permutations`` other assignments runs each of
    them, which makes the p-value exact; otherwise ``n_permutations`` assignments
    are drawn at random. Each run keeps the samples of every dose group and the
    doses of the design, and so the spline's knots and the genes of the dose-blind
    expression filter (Bourgon, Gentleman and Huber 2010); ``batch``, which a rerun
    could find confounded by chance and which no fit models, is left out. Vehicle-only
    data that the design places in mock dose groups, as Lee et al. (2024) did, runs
    through :func:`~cytepool.dose_response` and this check like any series
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-permutation-check).

    Each run fits the gate's test alone, with the result's seed for a multiple
    contrast test. Before the runs, each cell type runs once with its true doses
    and must give the result's gate: the same genes, p-values equal up to rounding
    and the same calls, so that the runs are the analysis the result records. Each
    run takes the time of the cell type's analysis and of the gate's test: about
    0.7 s for 20,000 genes and 24 samples with the spline test on two threads, twice
    that with a fold change, and more for a multiple contrast test or under dream.

    Args:
        result: The result of :func:`~cytepool.dose_response` to check, which keeps
            its cube and design.
        n_permutations: The runs per cell type, at least 1: random assignments, or
            every other assignment when there are at most this many.
        groups: A cell type or a list of them, each tested with its gate run; all
            cell types by default, those not checked with the reason.
        rng: A numpy random generator, an integer seed, or None, for the runs. The
            check takes one seed, the integer given or one drawn from the
            generator, and records it in ``settings["seed"]``; each cell type draws
            its runs from it and the CRC-32 of its name, so that a cell type's runs
            depend neither on the other cell types nor on the threads.

    Returns:
        The summary of each cell type, its runs and a provenance record.

    Raises:
        TypeError: If ``result`` is not a :class:`~cytepool.DoseResponseResult`,
            ``n_permutations`` is not an integer, or ``groups`` or ``rng`` has the
            wrong type, or is or holds a numpy masked array.
        ValueError: If ``n_permutations`` is below 1, ``groups`` is empty or names a
            cell type twice, ``rng`` is a negative seed, the result keeps no cube
            and design, its design has more than one random effect or a random
            slope, a cell type's true doses do not give the result's gate again, or
            a run does not run the gate.
        KeyError: If a cell type asked for is not in the cube, was not checked by
            the result, not tested, or did not run the gate.
    """
    if not isinstance(result, DoseResponseResult):
        raise TypeError(
            "result must be a DoseResponseResult, from dose_response, got "
            f"{type(result).__name__}"
        )
    reject_masked(n_permutations, "n_permutations")
    runs = positive_integer(n_permutations, "n_permutations")
    rerun = _Rerun.of(result)
    asked = _asked_groups(groups, result, rerun.cube)
    seed = recorded_seed(rng)
    records, tables = [], {}
    for group in asked:
        record, counts = _check_cell_type(result, rerun, group, runs, seed)
        records.append(record)
        if record["checked"]:
            tables[group] = Table(
                {
                    "run": np.arange(1, len(counts) + 1, dtype=np.int64),
                    "n_genes_passing_gate": np.array(counts, dtype=np.int64),
                }
            )
    provenance = {
        "method": "permutation_check",
        "versions": software_versions(),
        "settings": {
            "n_permutations": runs,
            "groups": None if groups is None else list(asked),
            "seed": seed,
        },
        "result": {
            "cube": result.provenance["cube"]["fingerprint"],
            "gate": result.gate,
            "seed": rerun.seed,
        },
        "groups": records,
    }
    return PermutationCheck(_summary(records, tables), tables, provenance)


def _asked_groups(groups: Any, result: DoseResponseResult, cube: Cube) -> list[str]:
    """The cell types asked for, in the result's order; each one named must have run
    the gate, or the result's table of the gate says why not."""
    reject_masked(groups, "groups")
    if groups is None:
        return list(result.groups)
    ids = cube.groups["group_id"].tolist()
    named = [ids[position] for position in group_positions(cube, groups)]
    for name in named:
        result.table(name, GATE)
    return [group for group in result.groups if group in named]


def _why_unchecked(result: DoseResponseResult, group: str) -> str:
    """Why a cell type has no gate to check, in the words of the result's table of
    the gate, or ''."""
    return "" if (group, GATE) in result.tables else absence(result, group, GATE)


# The rerun of one cell type.


@dataclass(frozen=True)
class _Rerun:
    """How to run dose_response again on one cell type with other doses: the cube,
    the design, the axis, the result's seed, and the other arguments of
    dose_response, the analysis's options among them as the result records them."""

    cube: Cube
    design: Design
    axis: DoseAxis
    seed: int | None
    arguments: dict[str, Any]

    @classmethod
    def of(cls, result: DoseResponseResult) -> "_Rerun":
        cube, design = result.cube, result.design
        if cube is None or design is None:
            raise ValueError(
                "the result keeps no cube and design, so its cell types cannot be run "
                "again: it was built without them; dose_response keeps both"
            )
        if len(design.random) > 1 or any(
            effect.effects.terms or not effect.effects.intercept
            for effect in design.random
        ):
            effects = ", ".join(str(effect) for effect in design.random)
            raise ValueError(
                f"the design's random effects, {effects}, leave no exchangeable units "
                "that permutation_check can tell: it permutes samples, or the units of "
                "one random intercept such as '(1 | donor)'"
            )
        settings, gate = result.settings, result.gate
        analysis = ANALYSES[result.method]
        recorded = result.provenance["analysis_settings"]
        test = gate["test"]
        return cls(
            cube=cube,
            design=design,
            axis=result.dose,
            seed=settings["seed"],
            arguments={
                # A fold change reads each dose's fit, which every test holds.
                "tests": [DOSE_GROUPS if test == EACH_DOSE else test],
                "spline_df": settings["spline_df"],
                "candidates": result.candidates if test == MCP else None,
                "gate": gate,
                "analysis": analysis,
                "min_cells": settings["min_cells"],
                "min_counts": settings["min_counts"],
                **recorded_options(
                    analysis, recorded, set_by_caller=SET_BY_DOSE_RESPONSE
                ),
            },
        )

    def __call__(
        self, group: str, rows: list[int], doses: Sequence[float]
    ) -> DoseResponseResult:
        """dose_response on one cell type, with the doses of the design's rows
        replaced."""
        samples = self.design.samples
        values = np.array(samples[self.axis.column], dtype=np.float64)
        values[rows] = doses
        design = Design(
            samples.with_columns({self.axis.column: values}),
            self.design.formula,
            random=[str(effect) for effect in self.design.random] or None,
            levels={name: list(v) for name, v in self.design.levels.items()},
        )
        return dose_response(
            self.cube,
            design,
            dose=self.axis,
            groups=[group],
            rng=self.seed,
            **self.arguments,
        )


def _same_gate(theirs: Table, ours: Table) -> bool:
    """Whether a run's gate holds the genes of another, with the same p-values up to
    rounding and the same calls."""
    return (
        theirs["feature_id"].tolist() == ours["feature_id"].tolist()
        and bool(
            np.allclose(
                np.asarray(theirs["p_value"]),
                np.asarray(ours["p_value"]),
                rtol=_ROUNDING,
                atol=0.0,
                equal_nan=True,
            )
        )
        and np.array_equal(theirs["passes_gate"], ours["passes_gate"])
    )


# Exchangeable units and assignments.


@dataclass(frozen=True)
class _Units:
    """A cell type's exchangeable units under the null, in blocks.

    ``samples`` are the used samples in the order of the fit, ``rows`` their rows of
    the design's sample table, ``doses`` their distinct doses ascending, ``items``
    the positions among ``samples`` of each unit's samples (one sample each, or a
    unit permuted whole), ``blocks`` the units of each block, and ``labels`` each
    unit's dose as an index into ``doses``, the vehicle 0.
    """

    kind: str
    unit: str | None
    samples: tuple[str, ...]
    rows: tuple[int, ...]
    doses: tuple[float, ...]
    items: tuple[tuple[int, ...], ...]
    blocks: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]

    def count(self) -> int:
        """The distinct assignments of the labels to the units within blocks: the
        product over blocks of n! over the product of each label's count's
        factorial."""
        total = 1
        for block in self.blocks:
            counts = Counter(self.labels[item] for item in block)
            total *= math.factorial(len(block)) // math.prod(
                math.factorial(n) for n in counts.values()
            )
        return total

    def every_other(self) -> Iterator[tuple[int, ...]]:
        """Each assignment but the true one: the blocks' arrangements, each in
        lexicographic order, in their Cartesian product."""
        arrangements = [
            list(_arrangements([self.labels[item] for item in block]))
            for block in self.blocks
        ]
        for chosen in itertools.product(*arrangements):
            labels = list(self.labels)
            for block, arrangement in zip(self.blocks, chosen, strict=True):
                for item, label in zip(block, arrangement, strict=True):
                    labels[item] = label
            if tuple(labels) != self.labels:
                yield tuple(labels)

    def draw(self, generator: np.random.Generator) -> tuple[int, ...]:
        """An assignment drawn uniformly: each block's labels in a random order,
        which reaches each distinct arrangement equally often."""
        labels = list(self.labels)
        for block in self.blocks:
            order = generator.permutation(len(block))
            for item, source in zip(block, order.tolist(), strict=True):
                labels[item] = self.labels[block[source]]
        return tuple(labels)

    def by_sample(self, labels: Sequence[int]) -> list[int]:
        """Each used sample's label under an assignment of the units."""
        found = [0] * len(self.samples)
        for item, positions in enumerate(self.items):
            for position in positions:
                found[position] = labels[item]
        return found

    def record(self) -> dict[str, Any]:
        """The units, for the provenance."""
        return {
            "units": self.kind,
            "unit": self.unit,
            "samples": list(self.samples),
            "doses": list(self.doses),
            "labels": self.by_sample(self.labels),
            "blocks": [
                [[self.samples[i] for i in self.items[item]] for item in block]
                for block in self.blocks
            ],
            "n_assignments": self.count(),
        }


def _arrangements(labels: list[int]) -> Iterator[tuple[int, ...]]:
    """The distinct arrangements of a multiset of labels, in lexicographic order, by
    the next-permutation step."""
    current = sorted(labels)
    n = len(current)
    while True:
        yield tuple(current)
        i = n - 2
        while i >= 0 and current[i] >= current[i + 1]:
            i -= 1
        if i < 0:
            return
        j = n - 1
        while current[j] <= current[i]:
            j -= 1
        current[i], current[j] = current[j], current[i]
        current[i + 1 :] = reversed(current[i + 1 :])


def _units(design: Design, axis: DoseAxis, used: Sequence[str]) -> _Units:
    """The exchangeable units of a cell type's used samples, by the design: the
    samples in blocks of the same row of the design's matrix; with a random
    intercept, its units whole when each holds one dose, in blocks of units with the
    same rows, and otherwise the samples within each unit and row."""
    samples = design.samples
    position = {sample: row for row, sample in enumerate(samples["sample_id"].tolist())}
    rows = [position[sample] for sample in used]
    values = np.asarray(samples[axis.column], dtype=np.float64)[rows]
    doses = np.unique(values)
    labels = np.searchsorted(doses, values).tolist()
    matrix = np.asarray(design.matrix, dtype=np.float64)[rows]
    keys: list[Any] = [matrix[i].tobytes() for i in range(len(rows))]
    unit = None
    items: list[tuple[int, ...]] = [(i,) for i in range(len(rows))]
    if not design.random:
        kind = "samples"
    else:
        (effect,) = design.random
        unit = effect.group
        columns = [samples[name].tolist() for name in effect.grouping]
        of = [tuple(column[row] for column in columns) for row in rows]
        members: dict[Any, list[int]] = {}
        for i, key in enumerate(of):
            members.setdefault(key, []).append(i)
        if all(len({labels[i] for i in m}) == 1 for m in members.values()):
            kind = "whole units"
            items = [tuple(m) for m in members.values()]
            keys = [tuple(sorted(keys[i] for i in m)) for m in items]
        else:
            kind = "doses within units"
            keys = [(of[i], keys[i]) for i in range(len(rows))]
    blocks: dict[Any, list[int]] = {}
    for item, key in enumerate(keys):
        blocks.setdefault(key, []).append(item)
    return _Units(
        kind=kind,
        unit=unit,
        samples=tuple(used),
        rows=tuple(rows),
        doses=tuple(doses.tolist()),
        items=tuple(items),
        blocks=tuple(tuple(block) for block in blocks.values()),
        labels=tuple(labels[item[0]] for item in items),
    )


# One cell type.


def _check_cell_type(
    result: DoseResponseResult,
    rerun: _Rerun,
    group: str,
    n_permutations: int,
    seed: int,
) -> tuple[dict[str, Any], list[int]]:
    """Run the gate of a cell type on permuted dose labels, or say why not: the cell
    type's record, and the genes that each run passes."""
    record: dict[str, Any] = {
        "group_id": group,
        "checked": False,
        "reason": _why_unchecked(result, group),
        "n_genes_passing_gate": None,
    }
    if record["reason"]:
        return record, []
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
    units = _units(rerun.design, rerun.axis, entry["fit"]["samples"])
    ours = result.table(group, GATE)
    record |= units.record()
    record["n_genes_passing_gate"] = int(np.count_nonzero(ours["passes_gate"]))
    if record["n_assignments"] == 1:
        record["reason"] = _single_assignment(units)
        return record, []
    # The true doses through the same path as the runs.
    again = rerun(group, list(units.rows), [units.doses[i] for i in record["labels"]])
    true = again.tables.get((group, GATE))
    if true is None or not _same_gate(true, ours):
        raise ValueError(
            f"{group!r} run again with its true doses does not give the result's gate, "
            f"the same genes with p-values equal up to a relative {_ROUNDING:g} and "
            "the same calls: the cube's values, the options or CytePool changed since "
            "the analysis; analyse the cube again and check that result"
        )
    exhaustive = record["n_assignments"] - 1 <= n_permutations
    if exhaustive:
        assignments = list(units.every_other())
    else:
        generator = np.random.default_rng(
            [seed, zlib.crc32(group.encode("utf-8")), _STREAM]
        )
        assignments = [units.draw(generator) for _ in range(n_permutations)]
    counts, tied = _runs(rerun, group, units, assignments, true)
    record |= {
        "checked": True,
        "exhaustive": exhaustive,
        "assignments": [units.by_sample(labels) for labels in assignments],
        "tied_runs": tied,
    }
    return record, counts


def _runs(
    rerun: _Rerun,
    group: str,
    units: _Units,
    assignments: list[tuple[int, ...]],
    true: Table,
) -> tuple[list[int], list[int]]:
    """The genes that the gate passes in each run, and the runs, numbered from 1,
    whose gate gives the true doses' p-values up to rounding."""
    counts, tied = [], []
    for run, labels in enumerate(assignments, start=1):
        doses = [units.doses[label] for label in units.by_sample(labels)]
        found = rerun(group, list(units.rows), doses)
        gate = found.tables.get((group, GATE))
        if gate is None:
            raise ValueError(
                f"run {run} of {group!r} did not run the gate: "
                f"{absence(found, group, GATE)}; a run keeps the samples of every dose "
                "group, so the analysis failed on the permuted doses"
            )
        counts.append(int(np.count_nonzero(gate["passes_gate"])))
        if _same_gate(gate, true):
            tied.append(run)
    return counts, tied


def _statistics(observed: int, passing: Any, ties: int) -> dict[str, float]:
    """What the runs say of the genes the true doses pass: the share of runs that
    pass genes, the mean genes per run, the p-value (1 + b) / (1 + B), with b the
    B runs that pass at least as many genes (Phipson and Smyth 2010), and the
    smallest p-value they can give, (1 + t) / (1 + B), with t the runs that tie with
    the true doses."""
    values = np.asarray(passing, dtype=np.float64)
    runs = values.size
    return {
        "share_of_runs_with_passing_genes": float(np.mean(values >= 1)),
        "mean_passing_genes_per_run": float(np.mean(values)),
        "p_value": (1 + int(np.count_nonzero(values >= observed))) / (1 + runs),
        "smallest_possible_p_value": (1 + ties) / (1 + runs),
    }


def _single_assignment(units: _Units) -> str:
    """Why the units allow no other assignment."""
    if all(len(block) == 1 for block in units.blocks):
        return (
            "each unit is a block of its own, as a numeric covariate of the design "
            "makes it, so no dose label can move; drop the covariate, or bin it into "
            "a factor"
        )
    return (
        "each block of units that the design treats alike holds one dose, so no "
        "dose label can move"
    )


def _summary(records: list[dict[str, Any]], tables: Mapping[str, Table]) -> Table:
    """One row per cell type, from its record and its runs; the statistics of a cell
    type not checked are NaN."""
    found = [
        _statistics(
            record["n_genes_passing_gate"],
            tables[record["group_id"]]["n_genes_passing_gate"],
            len(record["tied_runs"]),
        )
        if record["checked"]
        else {}
        for record in records
    ]

    def numbers(values: list[Any]) -> np.ndarray:
        return np.array(values, dtype=np.float64)  # None as NaN

    return Table(
        {
            "group_id": text_column([record["group_id"] for record in records]),
            "checked": np.array([record["checked"] for record in records], dtype=bool),
            "reason": text_column([record["reason"] for record in records]),
            "n_assignments": numbers(
                [record.get("n_assignments") for record in records]
            ),
            "exhaustive": np.array(
                [record.get("exhaustive", False) for record in records], dtype=bool
            ),
            "n_runs": np.array(
                [len(tables.get(record["group_id"], ())) for record in records],
                dtype=np.int64,
            ),
            "n_genes_passing_gate": numbers(
                [record["n_genes_passing_gate"] for record in records]
            ),
            **{name: numbers([row.get(name) for row in found]) for name in _STATISTICS},
        }
    )
