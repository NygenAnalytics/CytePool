"""Dose axes: where each dose sits along the axis of a dose-response curve, and the
design columns of a spline of dose.

A dose series has a vehicle, the samples given no dose, and treated samples at a few
doses, often spaced by a constant factor. A curve of dose follows an axis: the
logarithm of the dose, on which equal factors are equal steps, or the dose itself.
The vehicle's dose of 0 has no logarithm, so :class:`DoseAxis` either keeps the
vehicle as a group of its own, separated from the treated samples by a column of the
design, or places it on the axis by adding a recorded offset to every dose, as
DoseFinding's model ``log(dose + off)`` does (Bornkamp et al. 2026). sci-Plex-GxE
added 0.001 to every dose, a pseudocount chosen from experiments whose data are not
shown (McFaline-Figueroa et al. 2024); the offset decides where the vehicle sits and
the slope at the low end, which is why it is recorded rather than built in. The axis
also records the unit and the kind of dose, administered, nominal, measured or
internal, so that knots and curves print in the dose's own terms
(https://docs.nygen.io/CytePool/statistics/linear-models#dose-axes).

References:
    Bornkamp B, Pinheiro J, Bretz F, Sandig L, Thomas M, Sabanes Bove D (2026).
    DoseFinding: Planning and Analyzing Dose Finding Experiments. R package version
    1.4-2. https://cran.r-project.org/package=DoseFinding

    McFaline-Figueroa JL, Srivatsan S, Hill AJ, Gasperini M, Jackson DL, Saunders L,
    Domcke S, Regalado SG, Lazarchuck P, Alvarez S, Monnat RJ, Shendure J, Trapnell C
    (2024). Multiplex single-cell chemical genomics reveals the kinase dependence of
    the response to targeted therapy. Cell Genomics 4, 100487.
    doi:10.1016/j.xgen.2023.100487
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._columns import is_data_frame
from ._masked_arrays import reject_masked, unmasked_array
from ._messages import close_matches, listing
from ._options import checked_choice, positive_finite_number, positive_integer
from .stats.smoothing import NaturalSplineBasis
from .table import Table

__all__ = ["DoseAxis"]

FloatArray = npt.NDArray[np.float64]

METRICS = ("administered", "nominal", "measured", "internal")
SCALES = ("log10", "linear")
VEHICLES = ("own_group", "on_axis")
# How messages name the rules of NaturalSplineBasis that place knots.
_RULE_NAMES = {
    "levels": "R's quantile rule on the distinct doses",
    "quantiles": "R's quantile rule",
    "harrell": "Harrell's rule",
}
# Distinct doses closer than this, relative to the larger, are one dose that a
# conversion of units left as two.
_SAME_DOSE = 1e-8


@dataclass(frozen=True)
class DoseAxis:
    """The axis along which a curve of dose is fitted, with the vehicle's place on it.

    Records the dose column of a sample table, the dose's unit and kind, whether
    the curve follows log10 of the dose or the dose itself, and where the vehicle, the
    samples of dose 0, goes: in a group of its own (``"own_group"``), or on the axis
    (``"on_axis"``), on the log10 scale at an offset added to every dose. With the
    vehicle in its own group, a column of the design marks the treated samples, the
    spline columns are 0 for the vehicle, and the spline is fitted along the treated
    doses; a dose of 0 has no position on the axis. From the axis,
    :meth:`spline_basis` places the knots of a natural cubic spline on the samples'
    doses, and :meth:`design_columns` gives the columns that enter the design
    (https://docs.nygen.io/CytePool/statistics/linear-models#dose-axes).

    Args:
        column: The column of the sample table that holds each sample's dose, finite
            and at least 0, with 0 for the vehicle.
        unit: The unit of the doses, such as ``"ug/kg"``, ``"uM"`` or ``"Gy"``, which
            messages and records print.
        metric: What the dose measures: ``"administered"``, the amount given to the
            animal or person, as mg/kg; ``"nominal"``, the amount added to the
            medium; ``"measured"``, the amount measured in the medium, blood or
            tissue; or ``"internal"``, the amount at the target estimated by a model.
        scale: ``"log10"``, positions at log10 of the dose, or ``"linear"``, at the
            dose itself.
        vehicle: ``"own_group"`` or ``"on_axis"``.
        offset: With ``scale="log10"`` and ``vehicle="on_axis"``, and only then: a
            dose, above 0, added to every dose before its logarithm is taken, so
            that the vehicle sits at log10(offset).

    Raises:
        TypeError: If ``column`` or ``unit`` is not a str, an option is not a str,
            or ``offset`` is not a real number.
        ValueError: If ``column`` or ``unit`` is empty, an option is not one of its
            choices, or ``offset`` is missing where the vehicle sits on a log10 axis,
            given elsewhere, or not positive and finite.

    Example:
        >>> axis = DoseAxis("dose", unit="ug/kg", metric="administered")
        >>> samples = {"dose": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 10.0, 10.0]}
        >>> basis = axis.spline_basis(samples, df=2)
        >>> axis.doses(basis.knots).round(6).tolist(), basis.boundary_knots
        ([1.0], (-1.0, 1.0))
        >>> columns = axis.design_columns(samples, basis)
        >>> list(columns)
        ['dose_treated', 'dose_s1', 'dose_s2']
        >>> columns["dose_s1"].tolist()
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    """

    column: str
    _: KW_ONLY
    unit: str
    metric: str
    scale: str = "log10"
    vehicle: str = "own_group"
    offset: float | None = None

    def __post_init__(self) -> None:
        for name in ("column", "unit"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str, got {type(value).__name__}")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        checked_choice(self.metric, "metric", METRICS)
        checked_choice(self.scale, "scale", SCALES)
        checked_choice(self.vehicle, "vehicle", VEHICLES)
        if self.scale == "log10" and self.vehicle == "on_axis":
            if self.offset is None:
                raise ValueError(
                    "the vehicle's dose of 0 has no logarithm; with scale='log10' and "
                    "vehicle='on_axis', give offset, a dose added to every dose, or "
                    "keep the vehicle in a group of its own with vehicle='own_group'"
                )
            offset = positive_finite_number(self.offset, "offset")
            object.__setattr__(self, "offset", offset)
        elif self.offset is not None:
            raise ValueError(
                "offset places the vehicle on a log10 axis, and applies only with "
                f"scale='log10' and vehicle='on_axis'; got scale={self.scale!r} and "
                f"vehicle={self.vehicle!r}"
            )

    def positions(self, doses: npt.ArrayLike) -> FloatArray:
        """Where each dose sits on the axis.

        log10(dose) on the log10 scale, log10(dose + offset) with the vehicle on a
        log10 axis, and the dose itself on the linear scale; NaN for the vehicle,
        dose 0, when it is a group of its own.

        Args:
            doses: ``(n,)`` doses in the axis's unit, finite and at least 0.

        Returns:
            ``(n,)`` float64 positions.

        Raises:
            TypeError: If ``doses`` is not real-valued or is a numpy masked array.
            ValueError: If ``doses`` is not 1-D, or holds a value that is not finite
                or is negative.
        """
        return self._positions(_checked_doses(doses, "doses"))

    def doses(self, positions: npt.ArrayLike) -> FloatArray:
        """The doses at positions on the axis, the inverse of :meth:`positions`.

        Prints knots and reads curves in the dose's unit: 10^position, minus the
        offset when the vehicle sits on a log10 axis, or the position itself on the
        linear scale. The round trip through :meth:`positions` can differ from the
        dose in the last bit.

        Args:
            positions: ``(n,)`` real positions; NaN marks a missing value, and stays
                NaN.

        Returns:
            ``(n,)`` float64 doses.

        Raises:
            TypeError: If ``positions`` is not real-valued or is a numpy masked
                array.
            ValueError: If ``positions`` is not 1-D or holds an infinite value.
        """
        array = unmasked_array(positions, "positions", missing="NaN")
        if array.dtype.kind not in "iuf":
            raise TypeError(
                f"positions must hold real numbers, got dtype {array.dtype}"
            )
        if array.ndim != 1:
            raise ValueError(f"positions must be a 1-D array, got shape {array.shape}")
        values = array.astype(np.float64)
        if np.isinf(values).any():
            raise ValueError(
                "positions must not be infinite; NaN marks a missing value"
            )
        if self.scale == "linear":
            return values
        doses: FloatArray = 10.0**values
        if self.offset is not None:
            doses = doses - self.offset
        return doses

    def spline_basis(
        self,
        samples: Any,
        df: int | None = None,
        *,
        knots: str | npt.ArrayLike = "levels",
        form: str = "restricted_cubic",
    ) -> NaturalSplineBasis:
        """The basis of a natural cubic spline of dose, with knots placed on the
        samples' doses.

        The knots and the boundary knots, by default the lowest and highest doses on
        the axis, are placed on the positions of the samples that have one: the
        treated samples when the vehicle is a group of its own, and every sample
        when it is on the axis. Pass the basis to :meth:`design_columns`, and read
        its knots in doses with :meth:`doses`. A spline of ``df`` columns and the
        level of the treated samples fit ``df + 1`` distinct doses, so ``df`` is at
        most the number of distinct doses on the axis minus one; at that number the
        spline fits every dose group.

        Args:
            samples: The sample table, a ``Table``, a mapping of column name to 1-D
                values, or a pandas DataFrame, with the dose column.
            df: The number of spline columns, from 1 up. Needed with a rule; with
                knots given, it must agree with them (``len(knots) + 1``).
            knots: A rule of :class:`~cytepool.stats.smoothing.NaturalSplineBasis`:
                ``"levels"``, R's quantile rule on the distinct doses, each counted
                once whatever its number of replicates; ``"quantiles"``, R's rule
                for ``ns`` on the samples' positions; or ``"harrell"``, Harrell's
                percentiles. Or the interior knots as doses in the axis's unit,
                strictly between the lowest and highest dose on the axis.
            form: ``"restricted_cubic"``, whose first column is the position minus
                the lower boundary knot, so that the other columns test departure
                from a straight line, or ``"b_spline"``, the columns of R's ``ns``
                up to a change of basis.

        Returns:
            The basis, without an intercept.

        Raises:
            TypeError: If ``samples`` is not a table or holds a numpy masked array,
                its dose column is not numeric, ``df`` is not an integer, or the
                knots are not real numbers.
            ValueError: If the dose column is missing or holds a dose that is not
                finite or is negative; fewer than two distinct doses are on the
                axis, or two of them differ by less than 1e-8 of the dose; ``df``
                is missing with a rule, disagrees with given knots or exceeds the
                distinct doses on the axis minus one; a given knot is not strictly
                between the lowest and highest dose on the axis; any two knots
                coincide; or the rule cannot place the knots.
        """
        doses = self._sample_doses(samples)
        positions = self._positions(doses)
        on_axis = ~np.isnan(positions)
        levels = np.unique(doses[on_axis])
        self._check_levels(levels)
        if isinstance(knots, str):
            if df is None:
                raise ValueError(
                    f"df, the number of spline columns, is needed with knots={knots!r}; "
                    f"it can be at most {levels.size - 1} here, one less than the "
                    "distinct doses on the axis"
                )
            columns = positive_integer(df, "df")
            placing: str | FloatArray = knots
            source = _RULE_NAMES.get(knots, repr(knots))
        else:
            placing = self._knot_positions(knots, levels)
            columns = placing.size + 1
            if df is not None and positive_integer(df, "df") != columns:
                raise ValueError(
                    f"df={df} disagrees with {placing.size} knots, which give "
                    f"{columns} spline columns; give one of them"
                )
            source = "the knots given"
        if columns > levels.size - 1:
            raise ValueError(
                f"a spline of {columns} columns needs {columns + 1} distinct doses on "
                f"the axis, and the samples have {levels.size}, "
                f"{self._dose_text(levels)}{self._which_doses()}; lower df to at most "
                f"{levels.size - 1}"
            )
        try:
            placed = NaturalSplineBasis(positions[on_axis], columns, knots=placing)
        except ValueError as error:
            raise ValueError(
                f"{error} (on this axis, x is {self._position_text()})"
            ) from error
        interior = np.array(placed.knots)
        repeated = interior[1:][np.diff(interior) == 0.0]
        if repeated.size:
            dose = self.doses(repeated[:1])[0]
            raise ValueError(
                f"the knots that {source} places on the doses coincide at "
                f"{dose:g} {self.unit}, which would make the curve less smooth there; "
                "use knots='levels', a lower df, or give distinct knots"
            )
        if form == "b_spline":
            return placed
        return NaturalSplineBasis(
            knots=placed.knots, boundary_knots=placed.boundary_knots, form=form
        )

    def design_columns(
        self, samples: Any, basis: NaturalSplineBasis
    ) -> dict[str, FloatArray]:
        """The design columns of a spline of dose, at each sample's dose.

        In this order: ``{column}_treated``, 1 for a dose above 0 and 0 for the
        vehicle, when the vehicle is a group of its own; then ``{column}_s1`` to
        ``{column}_s{df}``, the basis at each sample's position, 0 for the vehicle
        when it is a group of its own. Every spline column vanishes at the lower
        boundary knot, so that ``{column}_treated`` is the difference between the
        curve there and the vehicle's mean, at the lowest dose with the rules
        ``"levels"`` and ``"quantiles"``. The vehicle's rows change no F-test that
        keeps the intercept and ``{column}_treated``, and no fitted curve: only what
        those two coefficients mean.

        The result is a dict of float64 arrays, which ``Table.with_columns`` adds to
        the sample table. ``samples`` can be any table with the dose column, such as
        a grid of doses at which to draw a fitted curve.

        Args:
            samples: The sample table, a ``Table``, a mapping of column name to 1-D
                values, or a pandas DataFrame, with the dose column.
            basis: A basis of this axis, from :meth:`spline_basis`: its knots are
                positions on this axis, and a basis placed on other values gives
                columns without meaning.

        Returns:
            The columns by name.

        Raises:
            TypeError: If ``samples`` is not a table or holds a numpy masked array,
                its dose column is not numeric, or ``basis`` is not a
                ``NaturalSplineBasis``.
            ValueError: If the dose column is missing or holds a dose that is not
                finite or is negative, or the basis has an intercept.
        """
        if not isinstance(basis, NaturalSplineBasis):
            raise TypeError(
                "basis must be a NaturalSplineBasis, such as spline_basis() gives, got "
                f"{type(basis).__name__}"
            )
        if basis.intercept:
            raise ValueError(
                "the basis has an intercept, which the design's intercept and "
                f"{self.column}_treated already give; build it without one, as "
                "spline_basis() does"
            )
        doses = self._sample_doses(samples)
        positions = self._positions(doses)
        values = basis.matrix(positions)
        values[np.isnan(positions)] = 0.0
        columns: dict[str, FloatArray] = {}
        if self.vehicle == "own_group":
            columns[f"{self.column}_treated"] = (doses > 0.0).astype(np.float64)
        for j in range(basis.df):
            columns[f"{self.column}_s{j + 1}"] = np.ascontiguousarray(values[:, j])
        return columns

    def to_dict(self) -> dict[str, Any]:
        """The axis as plain values that JSON stores: its six fields.

        ``DoseAxis(**axis.to_dict())`` rebuilds the axis.
        """
        return dataclasses.asdict(self)

    def _positions(self, doses: FloatArray) -> FloatArray:
        """Positions of checked doses."""
        if self.vehicle == "on_axis":
            if self.scale == "linear":
                return doses.copy()
            assert self.offset is not None  # set by the constructor
            on_log10_axis: FloatArray = np.log10(doses + self.offset)
            return on_log10_axis
        positions = np.full(doses.size, np.nan)
        treated = doses > 0.0
        positions[treated] = (
            np.log10(doses[treated]) if self.scale == "log10" else doses[treated]
        )
        return positions

    def _sample_doses(self, samples: Any) -> FloatArray:
        """The checked doses of a sample table's dose column."""
        reject_masked(samples, "samples", missing="None or NaN")
        if not (isinstance(samples, Table | Mapping) or is_data_frame(samples)):
            raise TypeError(
                "samples must be a Table, a mapping of column names to 1-D values or "
                f"a pandas DataFrame, got {type(samples).__name__}"
            )
        table = samples if isinstance(samples, Table) else Table(samples)
        if self.column not in table:
            raise ValueError(
                f"the dose column {self.column!r} is not a column of the sample "
                f"table{close_matches(self.column, table.columns)}; columns are: "
                f"{listing([repr(name) for name in table.columns])}"
            )
        return _checked_doses(table[self.column], f"samples[{self.column!r}]")

    def _knot_positions(self, knots: npt.ArrayLike, levels: FloatArray) -> FloatArray:
        """The positions of interior knots given as doses, which must lie strictly
        between the lowest and highest dose on the axis."""
        # Searched before np.atleast_1d could turn numpy.ma.masked in a list into NaN.
        reject_masked(knots, "knots")
        doses = np.sort(_checked_doses(np.atleast_1d(knots), "knots"))
        outside = (doses <= levels[0]) | (doses >= levels[-1])
        if outside.any():
            raise ValueError(
                f"knots must lie strictly between the lowest and highest dose on the "
                f"axis, {levels[0]:g} and {levels[-1]:g} {self.unit}"
                f"{self._which_doses()}; got {self._dose_text(doses[outside])}"
            )
        return self._positions(doses)

    def _check_levels(self, levels: FloatArray) -> None:
        """Refuse fewer than two distinct doses on the axis, and doses that differ
        only by rounding."""
        if levels.size < 2:
            found = self._dose_text(levels) if levels.size else "none"
            raise ValueError(
                "a spline of dose needs at least two distinct doses on the axis, and "
                f"the samples have {found}{self._which_doses()}"
            )
        close = np.flatnonzero(np.diff(levels) <= _SAME_DOSE * levels[1:])
        if close.size:
            a, b = float(levels[close[0]]), float(levels[close[0] + 1])
            raise ValueError(
                f"the doses {a!r} and {b!r} {self.unit} differ by less than 1e-8 of "
                "the dose, as a conversion of units can leave one dose as two; round "
                "the dose column first, for example with numpy.round"
            )

    def _position_text(self) -> str:
        """What a position on the axis is, for messages."""
        doses = "the treated doses" if self.vehicle == "own_group" else "the doses"
        if self.scale == "linear":
            return f"{doses} in {self.unit}"
        if self.offset is None:
            return f"log10 of {doses} in {self.unit}"
        return f"log10 of {doses} plus {self.offset:g} {self.unit}"

    def _which_doses(self) -> str:
        """Which doses are on the axis, for messages."""
        if self.vehicle == "own_group":
            return " (the treated doses: the vehicle is a group of its own)"
        return ""

    def _dose_text(self, doses: FloatArray) -> str:
        """Doses for messages, such as "0.1, 1 and 10 ug/kg"."""
        return f"{listing([f'{dose:g}' for dose in doses])} {self.unit}"


def _checked_doses(values: Any, name: str) -> FloatArray:
    """Doses as float64: a 1-D array of real numbers, finite and at least 0."""
    array = unmasked_array(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numbers, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    doses = array.astype(np.float64)
    bad = ~np.isfinite(doses) | (doses < 0.0)
    if bad.any():
        raise ValueError(
            f"{name} must hold doses that are finite and at least 0, with 0 for the "
            f"vehicle; got {listing([repr(float(dose)) for dose in doses[bad]])}"
        )
    return doses
