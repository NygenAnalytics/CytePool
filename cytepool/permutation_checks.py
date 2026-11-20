"""Could the genes a dose series calls come from noise? A permutation check of the
gate at the series' own design.

:func:`~cytepool.permutation_check` permutes the dose labels of each cell type among
the samples that its design treats alike, runs :func:`~cytepool.dose_response` again
on each permuted labelling with the result's settings, and counts the genes that pass
the result's gate: how often the gate passes genes when the doses carry no
information, and a permutation p-value for the genes the true doses pass
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

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585
"""

import itertools
import json
import math
import zlib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ._analysis_records import only_key, software_versions
from ._columns import text_column
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import positive_integer, recorded_seed
from ._reruns import changed_settings, recorded_options
from .cube import Cube
from .design import Design
from .differential_expression import (
    DifferentialExpressionResult,
    dream_voom,
    limma_trend,
    limma_voom,
)
from .dose_axes import DoseAxis
from .dose_responses import (
    DOSE_GROUPS,
    DUNNETT,
    EACH_DOSE,
    GATE,
    MCP,
    SPLINE,
    TESTS,
    DoseResponseResult,
    dose_response,
)
from .table import Table

__all__ = ["PermutationCheck", "permutation_check"]

FloatArray = npt.NDArray[np.float64]

_ANALYSES: dict[str, Callable[..., DifferentialExpressionResult]] = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
    "dream_voom": dream_voom,
}
# The options of the analysis that dose_response sets, or takes as arguments of its
# own, which a rerun does not read back from the record.
_SET_BY_DOSE_RESPONSE = (
    "groups",
    "contrasts",
    "f_tests",
    "adjust_across_groups",
    "min_cells",
    "min_counts",
)
# The third word of each cell type's seed sequence, a stream of its own beside those
# of dose_response's multiple contrast tests (their positions, 0 to 2).
_STREAM = 1000
_SEED_BOUND = 2**63


class PermutationCheck:
    """How often the gate of a dose series passes genes with its doses permuted,
    from :func:`permutation_check`.

    ``summary`` has one row per cell type asked for: ``group_id``; ``checked``;
    ``reason``, why a cell type was not checked, empty when checked;
    ``n_assignments``, the distinct assignments of the dose labels to the cell type's
    exchangeable units, as a float (the exact integer is in the provenance);
    ``exhaustive``, whether the runs were every other assignment; ``n_runs``;
    ``n_genes_passing_gate``, the genes the gate passes with the true doses (NaN
    where the gate did not run); ``share_of_runs_with_passing_genes``;
    ``mean_passing_genes_per_run``; ``p_value``, (1 + b) / (1 + B) for the b of the
    B runs that pass at least as many genes as the true doses (Phipson and Smyth
    2010); and ``smallest_p_value``, the smallest p-value the runs can give. With
    random runs it is 1 / (B + 1); with every other assignment it is r / A, where A
    counts the assignments and r those that only rename dose groups that the gate
    cannot tell apart, the true one included, which pass the same genes: every
    renaming of dose groups of equal sizes for the dose-group test and a spline that
    passes through every group's mean; those of the treated doses for the
    fold-change gate and Dunnett's test; none for the Williams-type test, the MCP
    step and other splines. Statistics of cell types not checked are NaN, and their
    ``n_runs`` 0.

    :meth:`table` gives each run of a cell type, with ``run`` and
    ``n_genes_passing_gate``. The provenance records each run's assignment, so that
    any run can be made again with :func:`~cytepool.dose_response`. Built by
    :func:`permutation_check`; immutable.

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
        """Each cell type: whether it was checked, its assignments and runs, the genes
        the gate passes, how often its runs pass genes, and the p-value."""
        return self._summary

    @property
    def settings(self) -> dict[str, Any]:
        """The settings, as a new dict (also in the provenance): ``n_permutations``,
        ``groups`` as given, and ``seed``, the seed of the runs, given or drawn."""
        settings: dict[str, Any] = json.loads(self._record)["settings"]
        return settings

    @property
    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record of what produced every number, as a new dict.

        It holds the versions of the software; the settings; the result checked
        (``result``): its cube's fingerprint, its gate and its seed; and, for every
        cell type (``groups``), whether it was checked and why not, the kind of units
        permuted (``"samples"``, ``"whole units"`` or ``"doses within units"``) and
        the random effect's grouping (``unit``), the samples used, in the order of the
        fit, the doses, each sample's dose as an index into them (``labels``), the
        blocks of units, each unit given by its samples, the number of assignments as
        an exact integer, the renamings that pass the same genes, whether the runs
        were every other assignment, and each run's assignment (``assignments``),
        each sample's dose index in the order of ``samples``.
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
            f"PermutationCheck of the gate, {_gate_text(record['result']['gate'])}: "
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


def _gate_text(gate: Mapping[str, Any]) -> str:
    """The gate, for the printout: its test and level."""
    level = f"at {100.0 * gate['fdr']:g}% FDR"
    if gate["fold_change"] is None:
        return f"the {gate['test']} test {level}"
    return f"TREAT beyond {gate['fold_change']:g}-fold at each dose, {level}"


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
    smallest = float(summary["smallest_p_value"][row])
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
    """Check how often the gate of a dose series passes genes when the doses carry no
    information.

    For each cell type, the dose labels of the samples it uses are permuted among the
    samples that the design treats alike (the module documentation gives the units),
    :func:`~cytepool.dose_response` runs again on that cell type with each permuted
    labelling and the result's settings and gate, and the genes that pass the gate
    are counted. Under the null hypothesis that no gene responds to dose, every
    assignment of the labels is as likely as the true one, so the share of runs that
    pass genes is the chance that noise alone passes genes at this design, and the
    genes the true doses pass get a permutation p-value, (1 + b) / (1 + B) for the b
    of the B runs that pass at least as many (Phipson and Smyth 2010). Where genes
    respond, strong responses scrambled over the permuted dose groups still pass in
    some runs, so the share then overstates that chance, and the p-value comes out
    larger than noise alone would make it. A design that allows at most
    ``n_permutations`` other assignments runs each of them, which makes the p-value
    exact; otherwise ``n_permutations`` assignments are drawn at random. Each run
    keeps the samples of every dose group and the doses of the design, and so the
    spline's knots and the genes of the dose-blind expression filter (Bourgon et al.
    2010); ``batch``, which a rerun could find confounded by chance and which no fit
    models, is left out. Vehicle-only data placed in mock dose groups by the design,
    the check of Lee et al. (2024), run through :func:`~cytepool.dose_response` and
    this check alike
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-permutation-check).

    Before the runs, each cell type runs once with its true doses through the same
    path and must give the result's gate p-values, bit for bit, so that the runs are
    the analysis the result records. Each run takes the time of the cell type's
    analysis and of the gate's test: about 0.6 s per 20,000 genes for the spline
    test on two threads, and more for a multiple contrast test or under dream.

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
        ValueError: If ``n_permutations`` is below 1, ``rng`` is a negative seed,
            the result keeps no cube and design, its design has more than one random
            effect or a random slope, a cell type's true doses do not give the
            result's gate p-values again, or a run does not run the gate.
        KeyError: If a cell type asked for was not checked by the result, not
            tested, or did not run the gate.
    """
    if not isinstance(result, DoseResponseResult):
        raise TypeError(
            "result must be a DoseResponseResult, from dose_response, got "
            f"{type(result).__name__}"
        )
    reject_masked(n_permutations, "n_permutations")
    runs = positive_integer(n_permutations, "n_permutations")
    asked = _asked_groups(groups, result)
    seed = recorded_seed(rng)
    rerun = _Rerun.of(result)
    rows, tables, records = [], {}, []
    for group in result.groups:
        if group not in asked:
            continue
        found = _check_cell_type(result, rerun, group, runs, seed)
        rows.append(found.row)
        records.append(found.record)
        if found.table is not None:
            tables[group] = found.table
    record = result.provenance
    provenance = {
        "method": "permutation_check",
        "versions": software_versions(),
        "settings": {
            "n_permutations": runs,
            "groups": None if groups is None else list(asked),
            "seed": seed,
        },
        "result": {
            "cube": record["cube"]["fingerprint"],
            "gate": result.gate,
            "seed": result.settings["seed"],
        },
        "groups": records,
    }
    return PermutationCheck(_summary(rows), tables, provenance)


def _asked_groups(groups: Any, result: DoseResponseResult) -> list[str]:
    """The cell types asked for, in the result's order; each one named must have run
    the gate."""
    reject_masked(groups, "groups")
    if groups is None:
        return list(result.groups)
    named = [groups] if isinstance(groups, str) else groups
    if isinstance(named, Mapping) or not isinstance(named, Sequence | np.ndarray):
        raise TypeError(
            "groups must be a cell type or a list of them, or None for every cell "
            f"type, got {type(groups).__name__}"
        )
    listed = list(named)
    known = list(result.groups)
    for name in listed:
        if not isinstance(name, str):
            raise TypeError(
                f"groups must be cell types (str), got {type(name).__name__} {name!r}"
            )
        if name not in known:
            raise KeyError(
                f"no cell type {name!r} was checked by the result; cell types are: "
                f"{listing(known) or '(none)'}{close_matches(name, known)}"
            )
        why = _why_unchecked(result, name)
        if why:
            raise KeyError(f"cell type {name!r} cannot be checked: {why}")
    return [group for group in known if group in listed]


def _why_unchecked(result: DoseResponseResult, group: str) -> str:
    """Why a cell type has no gate to check, or ''."""
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
    if not entry["tested"]:
        return f"not tested ({entry['status']}): {entry['reason']}"
    if (group, GATE) not in result.tables:
        test = result.gate["test"]
        return (
            f"the gate did not run: its test, {test!r}, was not run there: "
            f"{entry['tests_not_run'].get(test, '')}"
        )
    return ""


# The rerun of one cell type.


@dataclass(frozen=True)
class _Rerun:
    """How to run dose_response again on one cell type with other doses: the cube,
    the design, the axis, the analysis with its options as the result records them,
    and the arguments of dose_response."""

    cube: Cube
    design: Design
    axis: DoseAxis
    analysis: Callable[..., DifferentialExpressionResult]
    options: dict[str, Any]
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
        settings = result.settings
        gate = result.gate
        analysis = _ANALYSES[result.method]
        recorded = result.provenance["analysis_settings"]
        return cls(
            cube=cube,
            design=design,
            axis=result.dose,
            analysis=analysis,
            options=recorded_options(
                analysis, recorded, set_by_caller=_SET_BY_DOSE_RESPONSE
            ),
            arguments={
                "tests": _rerun_tests(settings["tests"], gate["test"]),
                "spline_df": settings["spline_df"],
                "candidates": result.candidates if gate["test"] == MCP else None,
                "gate": gate,
                "min_cells": settings["min_cells"],
                "min_counts": settings["min_counts"],
            },
        )

    def __call__(
        self, group: str, rows: list[int], doses: Sequence[float], seed: int | None
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
            analysis=self.analysis,
            groups=[group],
            rng=seed,
            **self.arguments,
            **self.options,
        )


def _rerun_tests(given: list[str] | None, gate: str) -> list[str] | None:
    """The tests of a rerun: the tests along dose that the result named, or None
    when it named none, and the gate's test, so that the fit holds the contrasts of
    the result's and its gate's p-values are the result's, bit for bit; the multiple
    contrast tests other than the gate's, which cost the most and do not change the
    fit, are left out. A gate with a fold change and no such test takes the dose
    groups' test, whose contrasts are the doses that every fit holds."""
    if given is None:
        return None
    tests = [test for test in given if test in TESTS]
    if gate != EACH_DOSE and gate not in tests:
        tests.append(gate)
    return tests or [DOSE_GROUPS]


def _check_reproduced(
    result: DoseResponseResult, again: DoseResponseResult, group: str
) -> None:
    """Refuse a check whose run with the true doses does not give the result's gate:
    its runs would not be those of the result's analysis."""
    record, rerun = result.provenance, again.provenance
    if rerun["cube"]["fingerprint"] != record["cube"]["fingerprint"]:
        raise ValueError(
            f"{group!r} ran again on a cube other than the result's (fingerprint "
            f"{rerun['cube']['fingerprint']}, where the result records "
            f"{record['cube']['fingerprint']}): its values changed since the analysis; "
            "analyse the cube again and check that result"
        )
    changed = changed_settings(
        record["analysis_settings"], rerun["analysis_settings"], ignored=("f_tests",)
    )
    ours = result.table(group, GATE)
    theirs = again.tables.get((group, GATE))
    same = (
        theirs is not None
        and theirs["feature_id"].tolist() == ours["feature_id"].tolist()
        and np.array_equal(
            np.asarray(theirs["p_value"]), np.asarray(ours["p_value"]), equal_nan=True
        )
    )
    if changed or not same:
        differ = (
            f"the settings {quoted(changed)} differ"
            if changed
            else "the gate's p-values differ"
        )
        raise ValueError(
            f"{group!r} run again with its true doses does not give the result's gate "
            f"({differ}): the cube's values, the options or CytePool changed since the "
            "analysis; analyse the cube again and check that result"
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

    def renamings(self, vehicle_held: bool) -> int:
        """The renamings of the dose labels that give an assignment, the identity
        included: the product of the factorials of the sizes of the classes of labels
        with the same count in each block, the vehicle in a class of its own when
        held."""
        counts = {
            label: tuple(
                sum(self.labels[item] == label for item in block)
                for block in self.blocks
            )
            for label in set(self.labels)
        }
        classes = Counter(
            ("vehicle",) if vehicle_held and label == 0 else vector
            for label, vector in counts.items()
        )
        return math.prod(math.factorial(size) for size in classes.values())

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


@dataclass(frozen=True)
class _Checked:
    """What the check of one cell type gives: its row of the summary, its table of
    runs (None when not checked) and its record."""

    row: dict[str, Any]
    table: Table | None
    record: dict[str, Any]


def _check_cell_type(
    result: DoseResponseResult,
    rerun: _Rerun,
    group: str,
    n_permutations: int,
    seed: int,
) -> _Checked:
    """Run the gate of a cell type on permuted dose labels, or say why not."""
    record: dict[str, Any] = {
        "group_id": group,
        "checked": False,
        "reason": _why_unchecked(result, group),
        "units": None,
        "unit": None,
        "samples": [],
        "doses": [],
        "labels": [],
        "blocks": [],
        "n_assignments": None,
        "renamings": None,
        "exhaustive": False,
        "assignments": [],
    }
    if record["reason"]:
        return _unchecked(record)
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
    units = _units(rerun.design, rerun.axis, entry["fit"]["samples"])
    observed = int(np.count_nonzero(result.table(group, GATE)["passes_gate"]))
    count = units.count()
    record |= {
        "units": units.kind,
        "unit": units.unit,
        "samples": list(units.samples),
        "doses": list(units.doses),
        "labels": units.by_sample(units.labels),
        "blocks": [
            [[units.samples[i] for i in units.items[item]] for item in block]
            for block in units.blocks
        ],
        "n_assignments": count,
    }
    if count == 1:
        return _unchecked(record | {"reason": _single_assignment(units)}, observed)
    # The true doses through the same path, with the result's seed.
    again = rerun(
        group,
        list(units.rows),
        [units.doses[label] for label in record["labels"]],
        result.settings["seed"],
    )
    _check_reproduced(result, again, group)
    generator = np.random.default_rng(
        [seed, zlib.crc32(group.encode("utf-8")), _STREAM]
    )
    exhaustive = count - 1 <= n_permutations
    assignments = (
        list(units.every_other())
        if exhaustive
        else [units.draw(generator) for _ in range(n_permutations)]
    )
    counts = []
    for run, labels in enumerate(assignments, start=1):
        doses = [units.doses[label] for label in units.by_sample(labels)]
        found = rerun(
            group,
            list(units.rows),
            doses,
            int(generator.integers(0, _SEED_BOUND)),
        )
        counts.append(_outcomes(found, group, run)["n_genes_passing_gate"])
    passing = np.array(counts, dtype=np.float64)
    renamings = _renamings(result, entry, units) if exhaustive else 1
    record |= {
        "checked": True,
        "renamings": renamings,
        "exhaustive": exhaustive,
        "assignments": [units.by_sample(labels) for labels in assignments],
    }
    row = {
        "group_id": group,
        "checked": True,
        "reason": "",
        "n_assignments": float(count),
        "exhaustive": exhaustive,
        "n_runs": len(counts),
        "n_genes_passing_gate": float(observed),
        **_statistics(observed, passing),
        "smallest_p_value": (
            renamings / count if exhaustive else 1.0 / (1 + len(counts))
        ),
    }
    table = Table(
        {
            "run": np.arange(1, len(counts) + 1, dtype=np.int64),
            "n_genes_passing_gate": passing.astype(np.int64),
        }
    )
    return _Checked(row, table, record)


def _statistics(observed: int, passing: FloatArray) -> dict[str, float]:
    """What the runs say of the genes the true doses pass: the share of runs that
    pass genes, the mean genes per run, and the p-value (1 + b) / (1 + B), with b
    the B runs that pass at least as many genes (Phipson and Smyth 2010)."""
    runs = passing.size
    return {
        "share_of_runs_with_passing_genes": float(np.mean(passing >= 1)),
        "mean_passing_genes_per_run": float(np.mean(passing)),
        "p_value": (1 + int(np.count_nonzero(passing >= observed))) / (1 + runs),
    }


def _outcomes(found: DoseResponseResult, group: str, run: int) -> dict[str, int]:
    """What one run gives, by name: the genes that pass the gate."""
    table = found.tables.get((group, GATE))
    if table is None:
        why = _why_unchecked(found, group)
        raise ValueError(
            f"run {run} of {group!r} did not run the gate: {why}; a run keeps the "
            "samples of every dose group, so the analysis failed on the permuted "
            "doses"
        )
    return {"n_genes_passing_gate": int(np.count_nonzero(table["passes_gate"]))}


def _renamings(
    result: DoseResponseResult, entry: Mapping[str, Any], units: _Units
) -> int:
    """The assignments, the true one included, that only rename dose groups the
    gate cannot tell apart, so that the gate passes the same genes: every dose
    group for the dose-group test and a spline through every group's mean, the
    treated ones for the fold-change gate and Dunnett's test, none otherwise."""
    test = result.gate["test"]
    treated = len(entry["doses"]) - 1
    through_means = test == SPLINE and entry["f_test_df1"].get(SPLINE) == treated
    if test == DOSE_GROUPS or through_means:
        return units.renamings(vehicle_held=False)
    if test in (EACH_DOSE, DUNNETT):
        return units.renamings(vehicle_held=True)
    return 1


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


def _unchecked(record: dict[str, Any], observed: int | None = None) -> _Checked:
    """The row of a cell type that is not checked, from its record; the genes its
    gate passes when the gate ran."""
    count = record["n_assignments"]
    row = {
        "group_id": record["group_id"],
        "checked": False,
        "reason": record["reason"],
        "n_assignments": np.nan if count is None else float(count),
        "exhaustive": False,
        "n_runs": 0,
        "n_genes_passing_gate": np.nan if observed is None else float(observed),
        "share_of_runs_with_passing_genes": np.nan,
        "mean_passing_genes_per_run": np.nan,
        "p_value": np.nan,
        "smallest_p_value": np.nan,
    }
    return _Checked(row, None, record)


def _summary(rows: list[dict[str, Any]]) -> Table:
    def column(name: str, dtype: Any) -> np.ndarray:
        return np.array([row[name] for row in rows], dtype=dtype)

    return Table(
        {
            "group_id": text_column([row["group_id"] for row in rows]),
            "checked": column("checked", bool),
            "reason": text_column([row["reason"] for row in rows]),
            "n_assignments": column("n_assignments", np.float64),
            "exhaustive": column("exhaustive", bool),
            "n_runs": column("n_runs", np.int64),
            "n_genes_passing_gate": column("n_genes_passing_gate", np.float64),
            "share_of_runs_with_passing_genes": column(
                "share_of_runs_with_passing_genes", np.float64
            ),
            "mean_passing_genes_per_run": column(
                "mean_passing_genes_per_run", np.float64
            ),
            "p_value": column("p_value", np.float64),
            "smallest_p_value": column("smallest_p_value", np.float64),
        }
    )
