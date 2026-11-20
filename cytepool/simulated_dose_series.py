"""Simulated dose series: pseudobulk counts per cell type drawn from known curves of
expression along dose, with the truth that estimates are compared with.

Whether a method finds the genes that respond to a dose, the shape of each response
and the dose at which it starts can only be checked on data whose truth is known. The
one published single-cell dose benchmark set the means of its responsive genes from
the benchmark-dose equations of the US EPA, added the variation of 3 individuals per
dose and compared tests on cells, without curves (Nault et al. 2022).
:func:`~cytepool.simulate_dose_series` draws what the dose methods read, the counts
of each replicate (an animal, or one donor's cells in a well) in each cell type, with
the effects of each replicate that muscat's and scDesign3's simulations also give
(Crowell et al. 2020; Song et al. 2024); :class:`~cytepool.DoseTruth` holds each
gene's true curve, EC50 and benchmark doses. The curves include shapes inside and
outside the model sets of BMDS (US EPA 2026), drc (Ritz et al. 2015) and DRomics
(Delignette-Muller et al. 2023), so that a method is not checked only on curves of
its own form; Hill, sigmoid Emax and four-parameter log-logistic curves are one
family (Kappenberg et al. 2023)
(https://docs.nygen.io/CytePool/statistics/linear-models#simulated-dose-series).

References:
    Crowell HL, Soneson C, Germain PL, Calini D, Collin L, Raposo C, Malhotra D,
    Robinson MD (2020). muscat detects subpopulation-specific state transitions from
    multi-sample multi-condition single-cell transcriptomics data. Nature
    Communications 11, 6077. doi:10.1038/s41467-020-19894-4

    Crump KS (1984). A new method for determining allowable daily intakes. Fundamental
    and Applied Toxicology 4, 854-871. doi:10.1016/0272-0590(84)90107-6

    Delignette-Muller ML, Siberchicot A, Larras F, Billoir E (2023). DRomics, a
    workflow to exploit dose-response omics data in ecotoxicology. Peer Community
    Journal 3, e90. doi:10.24072/pcjournal.325

    Kappenberg F, Duda JC, Schürmeyer L, Gül O, Brecklinghaus T, Hengstler JG,
    Schorning K, Rahnenführer J (2023). Guidance for statistical design and analysis
    of toxicological dose-response experiments, based on a comprehensive literature
    review. Archives of Toxicology 97, 2741-2761. doi:10.1007/s00204-023-03561-w

    Nault R, Saha S, Bhattacharya S, Dodson J, Sinha S, Maiti T, Zacharewski T
    (2022). Benchmarking of a Bayesian single cell RNAseq differential gene expression
    test for dose-response study designs. Nucleic Acids Research 50, e48.
    doi:10.1093/nar/gkac019

    Ritz C, Baty F, Streibig JC, Gerhard D (2015). Dose-response analysis using R.
    PLOS ONE 10, e0146021. doi:10.1371/journal.pone.0146021

    Song D, Wang Q, Yan G, Liu T, Sun T, Li JJ (2024). scDesign3 generates realistic
    in silico data for multimodal single-cell and spatial omics. Nature Biotechnology
    42, 247-252. doi:10.1038/s41587-023-01772-1

    US EPA (2026). BMDS User Guide: continuous endpoints. Online documentation of
    BMDS Online, BMDS Desktop and pybmds 25.2.
    https://usepa.github.io/bmds-user-guide/continuous.html
"""

import math
import numbers
import operator
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from . import __version__
from ._counts import real_values
from ._masked_arrays import reject_masked
from ._messages import close_matches, fitted_lines, listing, quoted
from ._options import (
    checked_choice,
    non_negative_number,
    positive_finite_number,
    positive_integer,
    random_generator,
    real_number,
)
from .cube import Cube
from .cube._axes import count_dtype, selected_positions
from .dose_axes import METRICS
from .table import Table

__all__ = ["DoseTruth", "simulate_dose_series"]

FloatArray = npt.NDArray[np.float64]

SHAPES = (
    "hill",
    "exponential",
    "linear",
    "power",
    "log_gauss_probit",
    "frechet",
    "step",
)
"""The curve shapes of responsive genes; a gene that does not respond is ``"flat"``."""

FLAT = "flat"

DEFAULT_SHAPES = MappingProxyType(
    {
        "hill": 0.30,
        "exponential": 0.15,
        "linear": 0.05,
        "power": 0.10,
        "log_gauss_probit": 0.20,
        "frechet": 0.10,
        "step": 0.10,
    }
)
"""The shares of the shapes among the responsive genes by default: 30% Hill curves, in
the model sets of BMDS, drc's LL.4 and DRomics; 60% in BMDS's set and 70% in DRomics';
20% Fréchet curves and steps, outside every set."""

FOLD_CHANGE = 1.1
"""The fold change of ``bmd_fold_change``: a change of log2(1.1) in either direction."""

# The parameter columns of a truth table, and the columns that each shape uses.
PARAMETERS = (
    "max_change",
    "ec50",
    "slope",
    "peak_dose",
    "peak_width",
    "counts_per_cell",
    "expression_sd",
)
_USES = {
    "hill": ("ec50", "slope"),
    "exponential": ("ec50",),
    "linear": (),
    "power": ("slope",),
    "log_gauss_probit": ("peak_dose", "peak_width"),
    "frechet": ("ec50", "slope"),
    "step": ("ec50",),
    FLAT: (),
}
_SHAPE_PARAMETERS = ("ec50", "slope", "peak_dose", "peak_width")
# The shapes whose curve reaches a change c only when |max_change| exceeds c (a
# plateau that the curve approaches), and those that reach it at |max_change| = c too
# (a peak or a step that the curve attains).
_APPROACHED = ("hill", "exponential", "frechet")
_ATTAINED = ("log_gauss_probit", "step")
# The shapes placed by their dose of a 1.1-fold change.
_PLACED = ("hill", "exponential", "frechet", "step")
# Expected counts per cell at the vehicle: log-normal around 0.1, a gene seen in about
# one cell in ten, with an SD (natural log) of 1.5 that the cell types share and 0.5 of
# each cell type's own.
_COUNTS_PER_CELL = 0.1
_SHARED_LEVEL_SD = 1.5
_OWN_LEVEL_SD = 0.5
# The SD (natural log) of the genes' factors on biological_sd and donor_sd.
_SD_FACTOR_SD = 0.3
# The spreads (natural log) of the cells pooled into a profile and of the sequencing
# depth between profiles, each log-normal with a mean of 1.
_CELLS_SD = 0.3
_DEPTH_SD = 0.2
# The dispersion of one cell's counts; a profile of n cells has 0.5 / n.
_CELL_DISPERSION = 0.5
# Slopes are log-uniform between these bounds. A power curve with an exponent below 1
# rises infinitely steeply at the vehicle.
_SLOPES = {"hill": (0.5, 4.0), "frechet": (0.5, 4.0), "power": (1.0, 4.0)}
# The SD of a bell along log10 dose, uniform between these multiples of the median
# log10 spacing of the positive doses, so that most bells exceed half their peak at
# two or more of the doses.
_BELL_WIDTHS = (0.75, 1.5)
# The doses of a 1.1-fold change of the placed shapes are log-uniform from an eighth
# of the log10 span of the positive doses below the lowest to an eighth above the top
# dose: a tenth lies below the doses and a tenth above.
_BEYOND = 0.125
# Distinct doses closer than this, relative to the larger, are one dose that a
# conversion of units left as two, as DoseAxis reads them.
_SAME_DOSE = 1e-8
# The shares of the shapes must sum to 1 within this.
_SUM_TOLERANCE = 1e-9
# The streams that each cell type's stream spawns, in this order.
_STREAMS = ("genes", "curves", "cells", "depth", "samples", "donors", "counts")


class _Curves(NamedTuple):
    """The curve of each gene: its shape and the parameters that shape uses (NaN
    for the others)."""

    shape: npt.NDArray[np.object_]
    max_change: FloatArray
    ec50: FloatArray
    slope: FloatArray
    peak_dose: FloatArray
    peak_width: FloatArray

    def take(self, rows: npt.NDArray[np.intp]) -> "_Curves":
        return _Curves._make(values[rows] for values in self)


class DoseTruth:
    """The true curves of a simulated dose series, from :func:`simulate_dose_series`.

    For each cell type and gene, the curve of expression along dose: the true change
    in log2 expression from the vehicle (a log2 fold change) at any dose
    (:meth:`curve`), the gene's EC50, the dose that gives half the maximal change,
    and its benchmark doses (Crump 1984), the lowest dose at which the curve departs
    from the vehicle by a stated change, at any change (:meth:`benchmark_doses`).
    :meth:`table` gives the parameters and two benchmark doses per gene, with the
    columns

    - ``feature_id``: the gene;
    - ``shape``: ``"flat"`` for a gene that does not respond, or one of ``"hill"``,
      ``"exponential"``, ``"linear"``, ``"power"``, ``"log_gauss_probit"``,
      ``"frechet"`` and ``"step"`` (the curves below);
    - ``max_change``: the signed maximal change in log2 expression: the plateau of
      the hill, exponential, frechet and step curves, the change at the top dose of
      the linear and power curves, and the peak (or trough) of the log_gauss_probit
      curve; 0 for flat;
    - ``ec50``: the dose of half the maximal change (hill, exponential, frechet,
      step); NaN otherwise;
    - ``slope``: the Hill slope (hill), the shape exponent (frechet) or the exponent
      (power), positive whatever the direction, which ``max_change`` carries; NaN
      otherwise;
    - ``peak_dose`` and ``peak_width``: the dose of the peak and the SD of the bell
      along log10 dose, in log10 units (log_gauss_probit); NaN otherwise;
    - ``counts_per_cell``: the expected counts of one cell at the vehicle, for a
      replicate without biological effect and of average depth;
    - ``expression_sd``: the SD of the gene's log2 expression between vehicle
      replicates from biological and donor variation, before counting: the SD of
      true expression, not of observed log-CPM values, which counting widens most
      for genes with few counts;
    - ``bmd_expression_sd``: the benchmark dose at a change of one ``expression_sd``;
    - ``bmd_fold_change``: the benchmark dose at a fold change of 1.1 in either
      direction, a rise of 10% or a fall of 9.1% (a change of log2(1.1) = 0.1375 in
      log2 expression).

    With M = ``max_change``, E = ``ec50``, n = ``slope``, P = ``peak_dose``, w =
    ``peak_width`` and T the top dose, the change from the vehicle at dose d > 0 is

    - hill: M / (1 + (E/d)^n), the Hill equation, drc's LL.4 and the sigmoid Emax
      model;
    - exponential: M (1 - 2^(-d/E)), BMDS's exponential 4 model;
    - linear: M d / T;
    - power: M (d/T)^n, BMDS's power model;
    - log_gauss_probit: M exp(-(log10(d/P) / w)^2 / 2), a symmetric bell along log
      dose, DRomics' log-Gauss-probit model with the same level on both sides;
    - frechet: M 2^(-(E/d)^n), the Fréchet curve, drc's Weibull type 1 with b < 0:
      flat at low doses, then a steep rise and a slow approach to the plateau, the
      Weibull curve 1 - exp(-(d/e)^n) of BMDS's exponential 5 model turned end for end
      on a log-dose axis;
    - step: M where d >= E, else 0;
    - flat: 0;

    and 0 at d = 0 for every shape. A benchmark dose is NaN where the curve never
    reaches the change: for flat genes; for hill, exponential and frechet curves when
    the change is |M| or more; for log_gauss_probit and step curves when it exceeds
    |M|. Linear and power curves reach any change, beyond the top dose when it exceeds
    |M|. A bell reaches a change twice, and the benchmark dose is the lower crossing.

    Built by :func:`simulate_dose_series`, or from tables with the parameter columns,
    as :meth:`table` gives them, so that a truth can be saved and read back:
    ``DoseTruth(truth.doses, {c: truth.table(c) for c in truth.cell_types},
    unit=truth.unit, metric=truth.metric)`` rebuilds ``truth``. Other columns of the
    tables, such as the benchmark doses, are ignored and computed again. Immutable
    (https://docs.nygen.io/CytePool/statistics/linear-models#simulated-dose-series).

    Args:
        doses: The dose levels of the series, as :func:`simulate_dose_series` takes
            them: strictly increasing from 0, the vehicle, with at least two doses
            above 0. The last is the top dose of the linear and power curves.
        curves: Each cell type's table, a ``Table``, a mapping of column name to 1-D
            values or a pandas DataFrame, with the columns ``feature_id``, ``shape``,
            ``max_change``, ``ec50``, ``slope``, ``peak_dose``, ``peak_width``,
            ``counts_per_cell`` and ``expression_sd``, and the same genes in the
            same order in every table.
        unit: The unit of the doses, as :class:`~cytepool.DoseAxis` takes it.
        metric: What the dose measures, one of ``"administered"``, ``"nominal"``,
            ``"measured"`` and ``"internal"``, as :class:`~cytepool.DoseAxis` takes
            it.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``doses`` are not as described, there is no cell type, a
            table lacks a column, its genes are repeated or differ from those of
            another table, a shape is unknown, or a parameter is not finite and
            above 0 where the shape uses it and NaN where it does not; ``max_change``
            must be 0 for flat genes and finite and not 0 otherwise, and
            ``counts_per_cell`` and ``expression_sd`` finite and above 0.

    Example:
        >>> truth = DoseTruth(
        ...     [0.0, 0.1, 1.0, 10.0],
        ...     {
        ...         "hepatocyte": {
        ...             "feature_id": ["Cyp1a1", "Alb"],
        ...             "shape": ["hill", "flat"],
        ...             "max_change": [2.0, 0.0],
        ...             "ec50": [1.0, np.nan],
        ...             "slope": [2.0, np.nan],
        ...             "peak_dose": [np.nan, np.nan],
        ...             "peak_width": [np.nan, np.nan],
        ...             "counts_per_cell": [0.5, 3.0],
        ...             "expression_sd": [0.25, 0.25],
        ...         }
        ...     },
        ...     unit="ug/kg",
        ...     metric="administered",
        ... )
        >>> truth.curve("hepatocyte", at=[0.0, 1.0, 10.0]).round(4).tolist()
        [[0.0, 1.0, 1.9802], [0.0, 0.0, 0.0]]
        >>> truth.benchmark_doses("hepatocyte", change=1.0).tolist()
        [1.0, nan]
    """

    __slots__ = ("_doses", "_unit", "_metric", "_tables", "_curves", "_index")

    _doses: tuple[float, ...]
    _unit: str
    _metric: str
    _tables: Mapping[str, Table]
    _curves: Mapping[str, _Curves]
    _index: dict[str, int]

    def __init__(
        self,
        doses: npt.ArrayLike,
        curves: Mapping[str, Any],
        *,
        unit: str,
        metric: str,
    ) -> None:
        reject_masked(doses, "doses")
        reject_masked(curves, "curves", missing="None or NaN")
        checked = _checked_series_doses(doses)
        unit = _checked_unit(unit)
        metric = checked_choice(metric, "metric", METRICS)
        if not isinstance(curves, Mapping):
            raise TypeError(
                "curves must be a mapping of cell type to its table, got "
                f"{type(curves).__name__}"
            )
        if not curves:
            raise ValueError("curves must hold the table of at least one cell type")
        top = float(checked[-1])
        tables: dict[str, Table] = {}
        parsed: dict[str, _Curves] = {}
        feature_ids: np.ndarray | None = None
        for cell_type, values in curves.items():
            if not isinstance(cell_type, str) or not cell_type:
                raise TypeError(
                    f"the cell types of curves must be non-empty str, got {cell_type!r}"
                )
            where = f"curves[{cell_type!r}]"
            ids, gene_curves, expression = _checked_truth_table(values, where)
            if feature_ids is None:
                feature_ids = ids
            elif not np.array_equal(ids, feature_ids):
                first = next(iter(curves))
                raise ValueError(
                    f"{where} must have the genes of curves[{first!r}], in the same "
                    "order: every cell type of a series has the same genes"
                )
            parsed[cell_type] = gene_curves
            tables[cell_type] = _truth_table(ids, gene_curves, expression, top)
        assert feature_ids is not None  # curves is not empty
        state = {
            "_doses": tuple(float(dose) for dose in checked),
            "_unit": unit,
            "_metric": metric,
            "_tables": MappingProxyType(tables),
            "_curves": MappingProxyType(parsed),
            "_index": {gene: i for i, gene in enumerate(feature_ids.tolist())},
        }
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DoseTruth is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DoseTruth is immutable")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _rebuilt_truth,
            (self._doses, dict(self._tables), self._unit, self._metric),
        )

    @property
    def doses(self) -> tuple[float, ...]:
        """The dose levels of the series, from 0, the vehicle."""
        return self._doses

    @property
    def unit(self) -> str:
        """The unit of the doses."""
        return self._unit

    @property
    def metric(self) -> str:
        """What the doses measure, one of administered, nominal, measured and
        internal."""
        return self._metric

    @property
    def cell_types(self) -> tuple[str, ...]:
        """The cell types, in the order of the cube's groups."""
        return tuple(self._tables)

    def table(self, cell_type: str) -> Table:
        """The true curve of each gene of a cell type, and its two benchmark doses.

        One row per gene, in the cube's feature order, with the columns that the
        class describes.

        Raises:
            KeyError: If the truth has no such cell type.
            TypeError: If ``cell_type`` is not a str.
        """
        return self._tables[self._cell_type(cell_type)]

    def curve(
        self, cell_type: str, genes: Any = None, *, at: npt.ArrayLike
    ) -> FloatArray:
        """The true change in log2 expression from the vehicle, at any doses.

        Args:
            cell_type: The cell type.
            genes: Feature ids, a boolean mask or integer positions, as
                ``Cube.select`` takes features; None (default) gives every gene, in
                table order.
            at: ``(m,)`` doses in the unit of the series, finite and at least 0.

        Returns:
            ``(genes, m)`` float64 log2 fold changes against the vehicle.

        Raises:
            KeyError: If the truth has no such cell type or gene.
            TypeError: If ``at`` is not real-valued, or an argument is or holds a
                numpy masked array.
            ValueError: If ``at`` is not 1-D or holds a dose that is not finite or is
                negative, or ``genes`` repeats a gene.
        """
        reject_masked(genes, "genes")
        reject_masked(at, "at")
        curves = self._selected(cell_type, genes)
        doses = _checked_doses(at, "at")
        return _changes(curves, self._doses[-1], doses)

    def benchmark_doses(
        self, cell_type: str, genes: Any = None, *, change: npt.ArrayLike
    ) -> FloatArray:
        """The lowest dose at which each gene's true curve departs from the vehicle
        by a change, in either direction.

        For a change c in log2 expression, the lowest dose d at which |f(d)| reaches
        c, f being the curve of :meth:`curve`; NaN where the curve never does. A
        rule of BMDS turns into a change here: one SD of the vehicle is ``change``
        equal to ``table(cell_type)["expression_sd"]``, which ``bmd_expression_sd``
        holds, and a fold change of 1.1 is log2(1.1), which ``bmd_fold_change``
        holds.

        Args:
            cell_type: The cell type.
            genes: Feature ids, a boolean mask or integer positions, as
                ``Cube.select`` takes features; None (default) gives every gene, in
                table order.
            change: The change in log2 expression, above 0 and finite: one value
                for every gene, or one per gene selected.

        Returns:
            ``(genes,)`` float64 doses in the unit of the series.

        Raises:
            KeyError: If the truth has no such cell type or gene.
            TypeError: If ``change`` is not real-valued, or an argument is or holds
                a numpy masked array.
            ValueError: If ``change`` has neither one value nor one per gene, or is
                not finite and above 0, or ``genes`` repeats a gene.
        """
        reject_masked(genes, "genes")
        reject_masked(change, "change")
        curves = self._selected(cell_type, genes)
        changes = real_values(change, "change")
        if changes.ndim == 0:
            changes = np.full(curves.shape.size, float(changes))
        elif changes.shape != (curves.shape.size,):
            raise ValueError(
                f"change must be one value or one per gene ({curves.shape.size}), got "
                f"shape {changes.shape}"
            )
        bad = ~(np.isfinite(changes) & (changes > 0.0))
        if bad.any():
            raise ValueError(
                "change must be finite and above 0, a change in log2 expression; got "
                f"{listing([repr(float(value)) for value in changes[bad]])}"
            )
        return _benchmark_doses(curves, self._doses[-1], changes)

    def __repr__(self) -> str:
        n_genes = len(self._index)
        cell_types = listing([repr(name) for name in self._tables])
        doses = f"{len(self._doses):,} doses from 0 to {self._doses[-1]:g} {self._unit}"
        return fitted_lines(
            f"DoseTruth: {len(self._tables):,} cell type"
            f"{'' if len(self._tables) == 1 else 's'} ({cell_types}) x {n_genes:,} "
            f"genes; {doses} ({self._metric})"
        )

    def _cell_type(self, cell_type: str) -> str:
        """A cell type of the truth, checked."""
        if not isinstance(cell_type, str):
            raise TypeError(f"cell_type must be a str, got {type(cell_type).__name__}")
        if cell_type not in self._tables:
            names = list(self._tables)
            raise KeyError(
                f"no cell type {cell_type!r}{close_matches(cell_type, names)}; the "
                f"truth has {listing([repr(name) for name in names])}"
            )
        return cell_type

    def _selected(self, cell_type: str, genes: Any) -> _Curves:
        """The curves of the genes selected in a cell type."""
        curves = self._curves[self._cell_type(cell_type)]
        ids = self._tables[cell_type]["feature_id"]
        rows = selected_positions(genes, ids, self._index, "genes")
        return curves if rows is None else curves.take(rows)


def simulate_dose_series(
    doses: npt.ArrayLike,
    *,
    replicates: int | Sequence[int] = 3,
    cell_types: int | Sequence[str] = 2,
    cells: float | Sequence[float] = 100.0,
    n_genes: int = 1000,
    responsive: float | Sequence[float] = 0.2,
    shapes: Mapping[str, float] | None = None,
    max_changes: float | Sequence[float] = (0.5, 1.0, 2.0),
    biological_sd: float = 0.25,
    donor_sd: float | None = None,
    composition: Mapping[str, Sequence[float]] | None = None,
    unit: str = "ug/kg",
    metric: str = "administered",
    rng: np.random.Generator | int | None = None,
) -> tuple[Cube, DoseTruth]:
    """Draw a dose series of pseudobulk counts from known curves, with its truth.

    Each cell type's profile of each replicate sums the counts of a random number of
    cells. For cell type k, replicate s at dose d and gene g, with independent
    standard normal draws:

    - the expected counts of one cell at the vehicle, r = 0.1 exp(1.5 z_g + 0.5
      z_gk), with z_g shared by the cell types, so that a gene is expressed at a
      similar level in each;
    - a factor v = exp(0.3 y_gk) on ``biological_sd`` and ``donor_sd``, so that the
      genes' SDs differ, log-normally;
    - the cells pooled, n ~ Poisson(cells c_k(d) exp(0.3 x_sk - 0.045)), with mean
      ``cells`` times the ``composition`` factor c_k(d), and the depth, e =
      exp(0.2 u_sk - 0.02), with mean 1;
    - the biological effect in log2 units, b = v (``biological_sd`` eps_sgk +
      ``donor_sd`` eta_jgk), with eta shared by the replicates of donor j;
    - the counts, negative binomial with mean n e r 2^(f_gk(d) + b) and dispersion
      0.5 / n, the sum of n cells of dispersion 0.5 each, so that profiles of fewer
      cells vary more beyond their counts; 0 where n = 0.

    f is the gene's true curve, a change in log2 expression from the vehicle that
    :class:`DoseTruth` describes. A share ``responsive`` of the genes respond,
    rounded half up, the first of a random order of the genes, so that a larger share
    adds genes and keeps the others; their shapes follow the shares of ``shapes`` by the largest
    remainder (ties in the order given), the maximal changes ``max_changes`` in turn
    within each shape, and rises and falls in turn within each shape and change, the
    first a rise, so that the counts of every shape, change and direction are exact.
    The dose of a 1.1-fold change of the hill, exponential, frechet and step curves
    is log-uniform from an eighth of the log10 span of the positive doses below the
    lowest to an eighth above the top dose, so that a tenth lies below the doses and
    a tenth above, and their EC50s follow from it; linear and power curves reach
    their maximal change at the top dose. Bells peak at a dose log-uniform over the
    positive doses, with an SD along log10 dose from 0.75 to 1.5 times the median
    log10 spacing of the positive doses, so that most exceed half their peak at two
    or more doses. Slopes are log-uniform from 0.5 to 4 (hill, frechet) and from 1 to
    4 (power). Doses multiplied by a factor give EC50s, peak doses and benchmark
    doses multiplied by it.

    The random draws come in independent streams: one for the levels the cell types
    share, and for each cell type one each for the genes' levels and SD factors, the
    curves, the cells, the depths, the biological effects, the donor effects and the
    counts. A cell type's curves, levels and SD factors therefore depend only on the
    seed, its position among the cell types, ``doses``, ``n_genes``,
    ``responsive``, ``shapes`` and ``max_changes``: other ``replicates``, ``cells``,
    ``composition``, ``biological_sd`` or ``donor_sd`` give the same genes, and so
    does adding a cell type, which compares designs on the same truth
    (https://docs.nygen.io/CytePool/statistics/linear-models#simulated-dose-series).

    Args:
        doses: The dose levels in ``unit``, strictly increasing from 0, the vehicle,
            with at least two doses above 0, no two of which differ by less than
            1e-8 of the dose.
        replicates: The replicates of each dose: one number, or one per dose.
        cell_types: How many cell types, named ``cell_type_1``, ``cell_type_2``,
            ..., or their names.
        cells: The expected number of cells pooled into a profile at a factor of 1
            in ``composition``, the mean (the median is about 4% lower): one number,
            or one per cell type.
        n_genes: The genes, the same in every cell type, with ids ``gene_1``, ...,
            padded with zeros to the width of ``n_genes``.
        responsive: The share of the genes that respond to dose: one number, or one
            per cell type; 0 gives a series without responses.
        shapes: The shares of the curve shapes among the responsive genes, by name
            (``"hill"``, ``"exponential"``, ``"linear"``, ``"power"``,
            ``"log_gauss_probit"``, ``"frechet"``, ``"step"``), summing to 1; None
            gives 30% hill, 15% exponential, 5% linear, 10% power, 20%
            log_gauss_probit, 10% frechet and 10% step.
        max_changes: The maximal changes of the responsive genes in log2 units, each
            above log2(1.1) = 0.1375, given in equal shares, half rises and half
            falls.
        biological_sd: The SD of a gene's log2 expression between replicates of one
            dose from biological variation between animals or wells, the median over
            genes.
        donor_sd: None (default) for one animal or donor per replicate. A number, 0
            or more, for donors whose cells receive every dose, as when one donor's
            cells are split across the doses in culture: the r-th replicate of each
            dose comes from donor r, and donor_sd is the SD of the donor's effect,
            shared by the doses, the median over genes.
        composition: For the cell types named, a factor per dose on the expected
            cells per profile, such as ``[1, 1, 1, 1, 1, 0.5]`` to halve a cell type
            at the top dose, or above 1 for one that expands; other cell types keep
            1.
        unit: The unit of the doses, recorded in the truth for
            :class:`~cytepool.DoseAxis`.
        metric: What the dose measures, one of ``"administered"`` (default),
            ``"nominal"``, ``"measured"`` and ``"internal"``, as
            :class:`~cytepool.DoseAxis` takes it.
        rng: A ``numpy.random.Generator``, which the call draws the series' seed
            from, so that calling again with it gives another series; an integer
            seed, which gives the same series as ``numpy.random.default_rng(seed)``;
            or None for a fresh, unseeded generator.

    Returns:
        The cube and its truth. The cube has the cell types as groups; the samples
        in dose order, replicates within, with ``sample_id`` ``d{k}_r{r}`` (k the
        dose's position from 0 for the vehicle, r the replicate from 1, each padded
        with zeros), ``dose`` (0 for the vehicle), ``dose_group`` (``d{k}``, which
        sorts in dose order, the vehicle first) and ``donor``; the layer ``counts``;
        ``n_cells``; and a provenance record of how it was simulated.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If a value is out of range or a per-dose or per-cell-type
            argument has the wrong length, a shape or a cell type of
            ``composition`` is unknown, the shares of ``shapes`` do not sum to 1, or
            ``composition`` is 0 at every dose for a cell type.

    Example:
        >>> cube, truth = simulate_dose_series([0, 0.1, 1, 10], replicates=2, rng=1)
        >>> cube.samples["dose_group"].tolist()
        ['d0', 'd0', 'd1', 'd1', 'd2', 'd2', 'd3', 'd3']
        >>> table = truth.table("cell_type_1")
        >>> int((table["shape"] != "flat").sum())
        200
    """
    settings = _checked_settings(
        doses=doses,
        replicates=replicates,
        cell_types=cell_types,
        cells=cells,
        n_genes=n_genes,
        responsive=responsive,
        shapes=shapes,
        max_changes=max_changes,
        biological_sd=biological_sd,
        donor_sd=donor_sd,
        composition=composition,
        unit=unit,
        metric=metric,
    )
    generator = random_generator(rng)
    # The series' seed is drawn from the generator's state, so that generators in the
    # same state give the same series, whatever seed sequence built them.
    root = np.random.SeedSequence(generator.integers(0, 2**63, size=4).tolist())
    shared_stream, *cell_type_streams = root.spawn(1 + len(settings.cell_types))
    shared_levels = np.random.default_rng(shared_stream).standard_normal(
        settings.n_genes
    )
    dose_of_sample = np.repeat(np.arange(settings.doses.size), settings.replicates)
    replicate_of_sample = np.concatenate(
        [np.arange(count) for count in settings.replicates]
    )
    n_samples = dose_of_sample.size
    counts = np.empty((len(settings.cell_types), n_samples, settings.n_genes), np.int64)
    n_cells = np.empty((n_samples, len(settings.cell_types)), np.int64)
    gene_ids = _gene_ids(settings.n_genes)
    curves: dict[str, dict[str, Any]] = {}
    for k, stream in enumerate(cell_type_streams):
        draws = dict(
            zip(_STREAMS, map(np.random.default_rng, stream.spawn(len(_STREAMS))))
        )
        level_draws = draws["genes"].standard_normal(settings.n_genes)
        factor_draws = draws["genes"].standard_normal(settings.n_genes)
        counts_per_cell = _COUNTS_PER_CELL * np.exp(
            _SHARED_LEVEL_SD * shared_levels + _OWN_LEVEL_SD * level_draws
        )
        sd_factor = np.exp(_SD_FACTOR_SD * factor_draws)
        gene_curves = _true_curves(draws["curves"], settings, k)
        changes = _changes(gene_curves, float(settings.doses[-1]), settings.doses)
        expected_cells = (
            settings.cells[k] * np.asarray(settings.composition[k])[dose_of_sample]
        )
        cells_drawn = draws["cells"].poisson(
            expected_cells
            * np.exp(
                _CELLS_SD * draws["cells"].standard_normal(n_samples) - _CELLS_SD**2 / 2
            )
        )
        depth = np.exp(
            _DEPTH_SD * draws["depth"].standard_normal(n_samples) - _DEPTH_SD**2 / 2
        )
        effects = settings.biological_sd * draws["samples"].standard_normal(
            (n_samples, settings.n_genes)
        )
        if settings.donor_sd is not None:
            donor_effects = draws["donors"].standard_normal(
                (max(settings.replicates), settings.n_genes)
            )
            effects += settings.donor_sd * donor_effects[replicate_of_sample]
        means = (
            (cells_drawn * depth)[:, None]
            * counts_per_cell
            * np.exp2(changes[:, dose_of_sample].T + sd_factor * effects)
        )
        # numpy's negative binomial takes a size, 1 / dispersion, and a probability;
        # a profile without cells has mean 0, which gives 0.
        size = (np.maximum(cells_drawn, 1) / _CELL_DISPERSION)[:, None]
        counts[k] = draws["counts"].negative_binomial(size, size / (size + means))
        n_cells[:, k] = cells_drawn
        curves[settings.cell_types[k]] = {
            "feature_id": gene_ids,
            **gene_curves._asdict(),
            "counts_per_cell": counts_per_cell,
            "expression_sd": sd_factor
            * math.hypot(settings.biological_sd, settings.donor_sd or 0.0),
        }
    truth = DoseTruth(
        settings.doses, curves, unit=settings.unit, metric=settings.metric
    )
    cube = Cube(
        {"counts": counts.astype(count_dtype(counts, int(counts.max(initial=0))))},
        samples=_sample_table(settings, dose_of_sample, replicate_of_sample),
        groups={"group_id": list(settings.cell_types)},
        features={"feature_id": gene_ids},
        n_cells=n_cells,
        provenance={"construction": _construction(settings, rng)},
        copy=False,
    )
    return cube, truth


# The truth: curves, benchmark doses and the placement of the curves.


def _changes(curves: _Curves, top: float, doses: FloatArray) -> FloatArray:
    """The change in log2 expression of each gene from the vehicle at each dose,
    ``(genes, doses)``, by the formulas of :class:`DoseTruth`."""
    values = np.zeros((curves.shape.size, doses.size))
    treated = np.flatnonzero(doses > 0.0)
    d = doses[treated]
    # Doses far below an EC50 or a peak overflow (E/d)^n and underflow d/P, which give
    # the curve's limit of 0 there.
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        for shape in SHAPES:
            rows = np.flatnonzero(curves.shape == shape)
            if rows.size == 0:
                continue
            m = curves.max_change[rows, None]
            e, n = curves.ec50[rows, None], curves.slope[rows, None]
            if shape == "hill":
                change = m / (1.0 + (e / d) ** n)
            elif shape == "exponential":
                change = -m * np.expm1(-math.log(2.0) * d / e)
            elif shape == "linear":
                change = m * d / top
            elif shape == "power":
                change = m * (d / top) ** n
            elif shape == "log_gauss_probit":
                p, w = curves.peak_dose[rows, None], curves.peak_width[rows, None]
                change = m * np.exp(-0.5 * (np.log10(d / p) / w) ** 2)
            elif shape == "frechet":
                change = m * np.exp2(-((e / d) ** n))
            else:
                change = m * (d >= e)
            values[np.ix_(rows, treated)] = change
    return values


def _benchmark_doses(curves: _Curves, top: float, change: FloatArray) -> FloatArray:
    """The lowest dose at which each gene's |change| reaches ``change``, NaN where it
    never does, by the closed form of each shape."""
    reach = np.abs(curves.max_change)
    found = np.full(reach.size, np.nan)
    for shape in SHAPES:
        rows = curves.shape == shape
        if shape in _APPROACHED:
            rows &= reach > change
        elif shape in _ATTAINED:
            rows &= reach >= change
        if not rows.any():
            continue
        a, c = reach[rows], change[rows]
        e, n = curves.ec50[rows], curves.slope[rows]
        if shape == "hill":
            found[rows] = e * (c / (a - c)) ** (1.0 / n)
        elif shape == "exponential":
            found[rows] = -e * np.log1p(-c / a) / math.log(2.0)
        elif shape == "linear":
            found[rows] = top * c / a
        elif shape == "power":
            found[rows] = top * (c / a) ** (1.0 / n)
        elif shape == "log_gauss_probit":
            p, w = curves.peak_dose[rows], curves.peak_width[rows]
            found[rows] = p * 10.0 ** (-w * np.sqrt(2.0 * np.log(a / c)))
        elif shape == "frechet":
            found[rows] = e * np.log2(a / c) ** (-1.0 / n)
        else:
            found[rows] = e
    return found


def _true_curves(rng: np.random.Generator, settings: "_Settings", k: int) -> _Curves:
    """The curve of each gene of cell type k: which genes respond, with which shape,
    maximal change and direction, and where along dose."""
    n_genes = settings.n_genes
    order = rng.permutation(n_genes)
    # Drawn for every gene, so that a gene's place does not depend on which respond.
    position, slope_draw, peak_draw, width_draw = rng.random((4, n_genes))
    shape = np.full(n_genes, FLAT, dtype=object)
    max_change = np.zeros(n_genes)
    responding = order[: math.floor(settings.responsive[k] * n_genes + 0.5)]
    labels, changes = _allocation(
        responding.size, settings.shapes, settings.max_changes
    )
    shape[responding] = labels
    max_change[responding] = changes
    positive = np.log10(settings.doses[1:])
    span = positive[-1] - positive[0]
    spacing = float(np.median(np.diff(positive)))
    ec50, slope, peak_dose, peak_width = (np.full(n_genes, np.nan) for _ in range(4))
    for name, (low, high) in _SLOPES.items():
        rows = shape == name
        slope[rows] = low * (high / low) ** slope_draw[rows]
    bells = shape == "log_gauss_probit"
    peak_dose[bells] = 10.0 ** (positive[0] + span * peak_draw[bells])
    low, high = _BELL_WIDTHS
    peak_width[bells] = spacing * (low + (high - low) * width_draw[bells])
    placed = np.isin(shape, _PLACED)
    fold_change_dose = 10.0 ** (
        positive[0] - _BEYOND * span + (1.0 + 2.0 * _BEYOND) * span * position[placed]
    )
    # The dose of a change is proportional to the EC50 for these shapes, so the EC50
    # is the dose placed divided by the dose of the change at an EC50 of 1.
    at_unit_ec50 = _Curves(
        shape[placed],
        max_change[placed],
        np.ones(int(placed.sum())),
        slope[placed],
        peak_dose[placed],
        peak_width[placed],
    )
    change = np.full(int(placed.sum()), math.log2(FOLD_CHANGE))
    ec50[placed] = fold_change_dose / _benchmark_doses(at_unit_ec50, 1.0, change)
    return _Curves(shape, max_change, ec50, slope, peak_dose, peak_width)


def _allocation(
    count: int, shapes: Mapping[str, float], max_changes: tuple[float, ...]
) -> tuple[npt.NDArray[np.object_], FloatArray]:
    """The shape and signed maximal change of ``count`` responsive genes, in nested
    order: shapes by the largest remainder of their shares (ties in the order
    given), the maximal changes in turn within each shape, and rises and falls in
    turn within each shape and change, the first a rise."""
    names = list(shapes)
    shares = np.array([shapes[name] for name in names])
    # Rounded, so that a quota such as 200 x 0.15 is 30 and not 30.000000000000004.
    quotas = np.round(count * shares / shares.sum(), 9)
    seats = np.floor(quotas).astype(np.int64)
    left = count - int(seats.sum())
    by_remainder = np.argsort(-(quotas - seats), kind="stable")
    seats[by_remainder[:left]] += 1
    within = np.concatenate([np.arange(n) for n in seats])
    turn = len(max_changes)
    sign = np.where((within // turn) % 2 == 0, 1.0, -1.0)
    labels = np.repeat(np.array(names, dtype=object), seats)
    return labels, sign * np.asarray(max_changes)[within % turn]


def _truth_table(
    feature_ids: Any, curves: _Curves, expression: Mapping[str, FloatArray], top: float
) -> Table:
    """A cell type's table: the parameters and the two benchmark doses."""
    columns: dict[str, Any] = {"feature_id": feature_ids, "shape": curves.shape}
    columns |= {name: getattr(curves, name) for name in _Curves._fields[1:]}
    columns |= {name: expression[name] for name in ("counts_per_cell", "expression_sd")}
    columns["bmd_expression_sd"] = _benchmark_doses(
        curves, top, np.asarray(expression["expression_sd"], dtype=np.float64)
    )
    columns["bmd_fold_change"] = _benchmark_doses(
        curves, top, np.full(curves.shape.size, math.log2(FOLD_CHANGE))
    )
    return Table(columns)


def _checked_truth_table(
    values: Any, where: str
) -> tuple[np.ndarray, _Curves, dict[str, FloatArray]]:
    """The genes, curves and expression of a cell type's table, checked."""
    table = values if isinstance(values, Table) else Table(values)
    missing = [
        name for name in ("feature_id", "shape", *PARAMETERS) if name not in table
    ]
    if missing:
        raise ValueError(f"{where} lacks the columns {quoted(missing)}")
    ids = np.asarray(table["feature_id"], dtype=object)
    if any(not isinstance(gene, str) or not gene for gene in ids.tolist()):
        raise TypeError(f"{where}: feature_id must hold non-empty str")
    repeated = sorted(gene for gene, n in Counter(ids.tolist()).items() if n > 1)
    if repeated:
        raise ValueError(f"{where}: feature_id repeats {quoted(repeated)}")
    shape = np.asarray(table["shape"], dtype=object)
    unknown = sorted({str(name) for name in shape.tolist()} - {*SHAPES, FLAT})
    if unknown:
        raise ValueError(
            f"{where}: shape must be one of {quoted([*SHAPES, FLAT])}, got "
            f"{quoted(unknown)}{close_matches(unknown[0], [*SHAPES, FLAT])}"
        )
    numbers_by_name = {
        name: real_values(table[name], f"{where}[{name!r}]") for name in PARAMETERS
    }
    for name in _SHAPE_PARAMETERS:
        values_of = numbers_by_name[name]
        used = np.array([name in _USES[s] for s in shape.tolist()], dtype=bool)
        bad = used & ~(np.isfinite(values_of) & (values_of > 0.0))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{where}: {name} of {ids[i]} must be finite and above 0 for a "
                f"{shape[i]} curve, got {values_of[i]!r}"
            )
        bad = ~used & ~np.isnan(values_of)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{where}: {name} of {ids[i]} must be NaN, since a {shape[i]} curve "
                f"has no {name}; got {values_of[i]!r}"
            )
    change = numbers_by_name["max_change"]
    flat = shape == FLAT
    bad = (flat & (change != 0.0)) | (~flat & ~(np.isfinite(change) & (change != 0.0)))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        expected = "0" if flat[i] else "finite and not 0"
        raise ValueError(
            f"{where}: max_change of {ids[i]} must be {expected} for a {shape[i]} "
            f"curve, got {change[i]!r}"
        )
    for name in ("counts_per_cell", "expression_sd"):
        values_of = numbers_by_name[name]
        bad = ~(np.isfinite(values_of) & (values_of > 0.0))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{where}: {name} of {ids[i]} must be finite and above 0, got "
                f"{values_of[i]!r}"
            )
    curves = _Curves(shape, *(numbers_by_name[name] for name in _Curves._fields[1:]))
    expression = {
        name: numbers_by_name[name] for name in ("counts_per_cell", "expression_sd")
    }
    return ids, curves, expression


def _rebuilt_truth(
    doses: tuple[float, ...], curves: dict[str, Table], unit: str, metric: str
) -> DoseTruth:
    """A truth rebuilt from its parts, for pickling."""
    return DoseTruth(doses, curves, unit=unit, metric=metric)


# The arguments.


@dataclass(frozen=True)
class _Settings:
    """The arguments of simulate_dose_series, checked and resolved per dose and per
    cell type."""

    doses: FloatArray
    replicates: tuple[int, ...]
    cell_types: tuple[str, ...]
    cells: tuple[float, ...]
    n_genes: int
    responsive: tuple[float, ...]
    shapes: Mapping[str, float]
    max_changes: tuple[float, ...]
    biological_sd: float
    donor_sd: float | None
    composition: tuple[tuple[float, ...], ...]
    unit: str
    metric: str


def _checked_settings(
    *,
    doses: Any,
    replicates: Any,
    cell_types: Any,
    cells: Any,
    n_genes: Any,
    responsive: Any,
    shapes: Any,
    max_changes: Any,
    biological_sd: Any,
    donor_sd: Any,
    composition: Any,
    unit: Any,
    metric: Any,
) -> _Settings:
    for name, value in (
        ("doses", doses),
        ("replicates", replicates),
        ("cell_types", cell_types),
        ("cells", cells),
        ("responsive", responsive),
        ("shapes", shapes),
        ("max_changes", max_changes),
        ("composition", composition),
    ):
        reject_masked(value, name)
    checked_doses = _checked_series_doses(doses)
    n_doses = checked_doses.size
    names = _checked_cell_types(cell_types)
    n_types = len(names)
    replicate_counts = _per_item(
        replicates, "replicates", n_doses, "dose", positive_integer
    )
    cell_numbers = _per_item(
        cells, "cells", n_types, "cell type", positive_finite_number
    )
    shares_responsive = _per_item(
        responsive, "responsive", n_types, "cell type", _checked_share
    )
    changes = _per_item(max_changes, "max_changes", None, "change", _checked_change)
    return _Settings(
        doses=checked_doses,
        replicates=tuple(int(count) for count in replicate_counts),
        cell_types=names,
        cells=tuple(float(value) for value in cell_numbers),
        n_genes=positive_integer(n_genes, "n_genes"),
        responsive=tuple(float(value) for value in shares_responsive),
        shapes=_checked_shapes(shapes),
        max_changes=tuple(float(value) for value in changes),
        biological_sd=positive_finite_number(biological_sd, "biological_sd"),
        donor_sd=None
        if donor_sd is None
        else non_negative_number(donor_sd, "donor_sd"),
        composition=_checked_composition(composition, names, n_doses),
        unit=_checked_unit(unit),
        metric=checked_choice(metric, "metric", METRICS),
    )


def _checked_series_doses(doses: Any) -> FloatArray:
    """The dose levels of a series: strictly increasing from 0, at least two above 0,
    none that differ by less than 1e-8 of the dose."""
    values = _checked_doses(doses, "doses")
    if values.size == 0 or values[0] != 0.0:
        found = f"{values[0]!r}" if values.size else "nothing"
        raise ValueError(
            f"doses must start with 0, the vehicle, and increase; got {found} first"
        )
    steps = np.flatnonzero(np.diff(values) <= 0.0)
    if steps.size:
        i = int(steps[0])
        raise ValueError(
            "doses must increase strictly, each dose level once; got "
            f"{values[i + 1]!r} after {values[i]!r}"
        )
    if values.size < 3:
        raise ValueError(
            "doses need at least two doses above 0, the vehicle, so that the curves "
            f"have a span to be placed on; got {listing([f'{v:g}' for v in values])}"
        )
    close = np.flatnonzero(np.diff(values[1:]) <= _SAME_DOSE * values[2:])
    if close.size:
        a, b = float(values[close[0] + 1]), float(values[close[0] + 2])
        raise ValueError(
            f"the doses {a!r} and {b!r} differ by less than 1e-8 of the dose, which "
            "DoseAxis reads as one dose; round them, for example with numpy.round"
        )
    return values


def _checked_doses(values: Any, name: str) -> FloatArray:
    """Doses as a new float64 array: 1-D, real, finite and at least 0."""
    doses = real_values(values, name)
    if doses.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of doses, got shape {doses.shape}"
        )
    bad = ~np.isfinite(doses) | (doses < 0.0)
    if bad.any():
        raise ValueError(
            f"{name} must be finite and at least 0, with 0 for the vehicle; got "
            f"{listing([repr(float(dose)) for dose in doses[bad]])}"
        )
    return doses


def _checked_cell_types(cell_types: Any) -> tuple[str, ...]:
    """The names of the cell types: given, or cell_type_1, ... for a number."""
    if isinstance(cell_types, numbers.Integral) and not isinstance(
        cell_types, bool | np.bool_
    ):
        count = positive_integer(cell_types, "cell_types")
        return tuple(f"cell_type_{i}" for i in range(1, count + 1))
    if isinstance(cell_types, str) or not isinstance(cell_types, Sequence | np.ndarray):
        raise TypeError(
            "cell_types must be a number of cell types or a list of their names, got "
            f"{type(cell_types).__name__}"
        )
    names = list(np.asarray(cell_types, dtype=object).tolist())
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(
            "cell_types must name at least one cell type, each by a non-empty str; "
            f"got {names!r}"
        )
    repeated = sorted(name for name, n in Counter(names).items() if n > 1)
    if repeated:
        raise ValueError(f"cell_types repeats {quoted(repeated)}; each name once")
    return tuple(names)


def _per_item(
    value: Any,
    name: str,
    count: int | None,
    item: str,
    check: Callable[[Any, str], Any],
) -> list[Any]:
    """One checked value per item: a single value for every item, or a sequence of
    ``count`` values (any length when ``count`` is None)."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, str) or not isinstance(value, Sequence | np.ndarray):
        values = [check(value, name)]
        return values * (1 if count is None else count)
    items = list(value)
    if count is None and not items:
        raise ValueError(f"{name} must hold at least one value")
    if count is not None and len(items) != count:
        raise ValueError(
            f"{name} must be one value or one per {item} ({count}), got {len(items)}"
        )
    return [check(entry, f"{name}[{i}]") for i, entry in enumerate(items)]


def _checked_share(value: Any, name: str) -> float:
    """A share from 0 to 1."""
    share = real_number(value, name, finite=True)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"{name} must be a share from 0 to 1, got {share}")
    return share


def _checked_change(value: Any, name: str) -> float:
    """A maximal change, above log2(1.1)."""
    change = real_number(value, name, finite=True)
    if not change > math.log2(FOLD_CHANGE):
        raise ValueError(
            f"{name} must be above log2(1.1) = 0.1375, a change in log2 expression "
            "that a 1.1-fold change can place, and is given as a size, the direction "
            f"being drawn; got {change}"
        )
    return change


def _checked_shapes(shapes: Any) -> Mapping[str, float]:
    """The shares of the shapes, by name, summing to 1."""
    if shapes is None:
        return DEFAULT_SHAPES
    if not isinstance(shapes, Mapping):
        raise TypeError(
            "shapes must be a mapping of shape name to share, such as {'hill': 0.5, "
            f"'step': 0.5}}, got {type(shapes).__name__}"
        )
    checked: dict[str, float] = {}
    for name, share in shapes.items():
        if name not in SHAPES:
            raise ValueError(
                f"shapes has an unknown shape {name!r}"
                f"{close_matches(str(name), list(SHAPES))}; shapes are "
                f"{quoted(list(SHAPES))}"
            )
        checked[name] = non_negative_number(share, f"shapes[{name!r}]")
    total = math.fsum(checked.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"the shares of shapes must sum to 1, got {total!r}")
    return MappingProxyType(checked)


def _checked_composition(
    composition: Any, names: tuple[str, ...], n_doses: int
) -> tuple[tuple[float, ...], ...]:
    """Each cell type's factor at each dose, 1 where not given."""
    factors = [(1.0,) * n_doses for _ in names]
    if composition is None:
        return tuple(factors)
    if not isinstance(composition, Mapping):
        raise TypeError(
            "composition must be a mapping of cell type to a factor per dose, got "
            f"{type(composition).__name__}"
        )
    for cell_type, values in composition.items():
        if cell_type not in names:
            raise ValueError(
                f"composition names an unknown cell type {cell_type!r}"
                f"{close_matches(str(cell_type), list(names))}; cell types are "
                f"{quoted(list(names))}"
            )
        where = f"composition[{cell_type!r}]"
        if isinstance(values, str) or not isinstance(values, Sequence | np.ndarray):
            raise TypeError(
                f"{where} must hold one factor per dose, got {type(values).__name__}"
            )
        checked = _per_item(values, where, n_doses, "dose", non_negative_number)
        if not any(checked):
            raise ValueError(
                f"{where} is 0 at every dose, so no profile of {cell_type!r} would "
                "have cells; leave the cell type out of cell_types instead"
            )
        factors[names.index(cell_type)] = tuple(float(value) for value in checked)
    return tuple(factors)


def _checked_unit(unit: Any) -> str:
    """The unit of the doses, as DoseAxis takes it: a non-empty str."""
    if not isinstance(unit, str):
        raise TypeError(f"unit must be a str, got {type(unit).__name__}")
    if not unit.strip():
        raise ValueError("unit must not be empty")
    return unit


# The cube.


def _gene_ids(n_genes: int) -> list[str]:
    """gene_1, ..., padded with zeros to the width of n_genes."""
    width = len(str(n_genes))
    return [f"gene_{i:0{width}d}" for i in range(1, n_genes + 1)]


def _sample_table(
    settings: _Settings, dose_of_sample: np.ndarray, replicate_of_sample: np.ndarray
) -> dict[str, Any]:
    """The samples: ids, doses, dose groups and donors, in dose order."""
    dose_width = len(str(settings.doses.size - 1))
    replicate_width = len(str(max(settings.replicates)))
    groups = [f"d{k:0{dose_width}d}" for k in dose_of_sample.tolist()]
    replicates = [f"{r + 1:0{replicate_width}d}" for r in replicate_of_sample.tolist()]
    if settings.donor_sd is None:
        sample_width = len(str(dose_of_sample.size))
        donors = [f"donor_{i:0{sample_width}d}" for i in range(1, len(groups) + 1)]
    else:
        donors = [f"donor_{r}" for r in replicates]
    return {
        "sample_id": [f"{g}_r{r}" for g, r in zip(groups, replicates)],
        "dose": settings.doses[dose_of_sample],
        "dose_group": groups,
        "donor": donors,
    }


def _construction(settings: _Settings, rng: Any) -> dict[str, Any]:
    """The cube's record of how it was simulated: every argument as resolved."""
    seed = (
        operator.index(rng)
        if isinstance(rng, numbers.Integral) and not isinstance(rng, bool | np.bool_)
        else None
    )
    return {
        "function": "simulate_dose_series",
        "cytepool_version": __version__,
        "numpy_version": np.__version__,
        "doses": settings.doses.tolist(),
        "replicates": list(settings.replicates),
        "cell_types": list(settings.cell_types),
        "cells": list(settings.cells),
        "n_genes": settings.n_genes,
        "responsive": list(settings.responsive),
        "shapes": dict(settings.shapes),
        "max_changes": list(settings.max_changes),
        "biological_sd": settings.biological_sd,
        "donor_sd": settings.donor_sd,
        "composition": {
            name: list(factors)
            for name, factors in zip(settings.cell_types, settings.composition)
        },
        "unit": settings.unit,
        "metric": settings.metric,
        "seed": seed,
    }
