"""Candidate shapes of a dose response, for the MCP step of MCP-Mod.

The MCP step of MCP-Mod tests for a dose signal under uncertainty about the shape of
the response (Bretz, Pinheiro and Branson 2005; Pinheiro et al. 2014): the analyst
guesses a few plausible shapes, each becomes the contrast of the dose groups that
best detects it, and the largest of the contrasts' t-statistics is referred to the
multivariate t distribution. :class:`CandidateShapes` holds the guesses in the model
families of DoseFinding (Bornkamp et al. 2026), as the change from the vehicle along
dose: Emax, sigmoid Emax, exponential and beta. Its
:meth:`~cytepool.CandidateShapes.spread` places a few shapes of each family over the
doses of a series, so that their contrasts cover monotone responses that start early,
midway or late, and umbrella responses that rise and fall;
:meth:`~cytepool.CandidateShapes.means` gives each shape's mean change at the doses,
from which :func:`~cytepool.stats.multiple_contrasts.optimal_contrasts` builds the
contrasts. :func:`~cytepool.dose_response` runs the step with
``tests=("mcp",)``
(https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-contrasts).

References:
    Bretz F, Pinheiro JC, Branson M (2005). Combining multiple comparisons and
    modeling techniques in dose-response studies. Biometrics 61, 738-748.
    doi:10.1111/j.1541-0420.2005.00344.x

    Pinheiro J, Bornkamp B, Glimm E, Bretz F (2014). Model-based dose finding under
    model uncertainty using general parametric models. Statistics in Medicine 33,
    1646-1661. doi:10.1002/sim.6052

    Bornkamp B, Pinheiro J, Bretz F, Sandig L, Thomas M, Sabanes Bove D (2026).
    DoseFinding: Planning and Analyzing Dose Finding Experiments. R package version
    1.4-2. https://cran.r-project.org/package=DoseFinding
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._doses import checked_doses, distinct_doses, dose_text, sample_doses
from ._masked_arrays import reject_masked
from ._options import integral_number, positive_finite_number, real_number
from .dose_axes import DoseAxis

__all__ = ["CandidateShapes"]

FloatArray = npt.NDArray[np.float64]

FAMILIES = ("emax", "sigmoid_emax", "exponential", "beta")
"""The model families, in the order of the shapes."""
SPREAD_HILL_SLOPE = 3.0
"""The Hill slope of the sigmoid Emax shapes that :meth:`CandidateShapes.spread`
places."""
BETA_SCALE_FACTOR = 1.2
"""The beta model's scale over the top dose that :meth:`CandidateShapes.spread`
takes, DoseFinding's default."""


@dataclass(frozen=True)
class CandidateShapes:
    """Candidate shapes of a dose response: guesses of the mean change from the
    vehicle along dose, for the MCP step of MCP-Mod.

    Each shape is a curve of dose d in one of DoseFinding's model families (Bornkamp
    et al. 2026), up to its size, which the step does not need:

    - ``emax``: d / (ED50 + d), a rise that slows from the start and reaches half
      its plateau at ED50;
    - ``sigmoid_emax``: d^h / (ED50^h + d^h), the Hill equation with ED50 and Hill
      slope h, a rise that starts slowly when h is above 1;
    - ``exponential``: exp(d / delta) - 1, a rise that grows with dose, the change
      from the vehicle of BMDS's exponential 2 model a exp(b d) with b = 1 / delta
      (US EPA 2026); a small delta keeps the shape flat until the top doses;
    - ``beta``: (d / s)^delta1 (1 - d / s)^delta2 with the scale s =
      ``beta_scale``, an umbrella shape, which rises to a peak at s delta1 /
      (delta1 + delta2) and falls.

    There is no linear shape, as DoseFinding's advice for MCP-Mod has it: an Emax
    shape with a large ED50 and an exponential shape with a large delta are nearly
    linear. The step tests each shape's direction with the contrast of the dose
    groups that best detects it, so a shape and the same shape scaled or shifted give
    one contrast. Build the shapes from prior guesses of ED50s and peaks, or let
    :meth:`spread` place them over the doses of a series
    (https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-contrasts).

    Args:
        emax: The ED50 of each Emax shape, in the dose's unit.
        sigmoid_emax: (ED50, Hill slope) of each sigmoid Emax shape.
        exponential: delta of each exponential shape, in the dose's unit.
        beta: (delta1, delta2) of each beta shape.
        beta_scale: The beta model's scale s, at least the top dose, in the dose's
            unit; given with beta shapes, and only then.

    Each family takes a list with one entry per shape, such as ``emax=[0.5]`` for
    one shape; every parameter is finite and above 0, and there is at least one
    shape.

    Raises:
        TypeError: If a family is not a list of numbers (``emax``,
            ``exponential``) or of pairs of numbers (``sigmoid_emax``, ``beta``),
            or a value is not a real number, or an argument is a numpy masked array.
        ValueError: If there is no shape, a parameter is not finite and above 0, or
            ``beta_scale`` is missing with beta shapes or given without them.

    Example:
        >>> shapes = CandidateShapes(emax=[0.2, 2.0], beta=[(0.5, 2.0)], beta_scale=12.0)
        >>> shapes.names
        ('emax_1', 'emax_2', 'beta')
        >>> shapes.means([0.0, 0.1, 1.0, 10.0]).round(3).tolist()[0]
        [0.0, 0.34, 0.85, 1.0]
    """

    emax: tuple[float, ...] = ()
    sigmoid_emax: tuple[tuple[float, float], ...] = ()
    exponential: tuple[float, ...] = ()
    beta: tuple[tuple[float, float], ...] = ()
    beta_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "emax", _numbers(self.emax, "emax", "ED50"))
        object.__setattr__(
            self,
            "sigmoid_emax",
            _pairs(self.sigmoid_emax, "sigmoid_emax", ("ED50", "Hill slope")),
        )
        object.__setattr__(
            self, "exponential", _numbers(self.exponential, "exponential", "delta")
        )
        object.__setattr__(
            self, "beta", _pairs(self.beta, "beta", ("delta1", "delta2"))
        )
        reject_masked(self.beta_scale, "beta_scale")
        if self.beta and self.beta_scale is None:
            raise ValueError(
                "beta shapes need beta_scale, the beta model's scale, of at least the "
                "top dose; DoseFinding takes 1.2 times the top dose"
            )
        if not self.beta and self.beta_scale is not None:
            raise ValueError(
                "beta_scale is the scale of the beta shapes, and there are none; leave "
                "it out"
            )
        if self.beta_scale is not None:
            object.__setattr__(
                self,
                "beta_scale",
                positive_finite_number(self.beta_scale, "beta_scale"),
            )
        if not self.names:
            raise ValueError(
                "there are no candidate shapes; give at least one, or place them with "
                "CandidateShapes.spread"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """The shapes' names in order, each the name of its family, such as
        ``"emax"``, with ``_1``, ``_2``, ... when the family has several shapes."""
        names: list[str] = []
        for family in FAMILIES:
            count = len(getattr(self, family))
            names += (
                [family] if count == 1 else [f"{family}_{j + 1}" for j in range(count)]
            )
        return tuple(names)

    def means(self, doses: npt.ArrayLike) -> FloatArray:
        """Each shape's mean change from the vehicle at doses, scaled to a largest
        change of 1.

        A shape's change is 0 at dose 0 and is scaled to 1 at the highest of
        ``doses`` for the rising shapes, and at its peak for a beta shape, wherever
        the peak lies, as DoseFinding's ``getResp(Mods(..., doses = doses))`` gives
        the means with its default placebo effect 0 and maximal effect 1. The
        contrasts of the MCP step do not depend on the scale.

        Args:
            doses: ``(n_doses,)`` doses in the dose's unit, finite and at least 0,
                at most ``beta_scale`` with beta shapes, and at least one above 0.

        Returns:
            ``(n_shapes, n_doses)`` float64 means, the shapes in the order of
            :attr:`names`.

        Raises:
            TypeError: If ``doses`` is not real-valued or is a numpy masked array.
            ValueError: If ``doses`` is not 1-D, holds a dose that is not finite or
                is negative, has no dose above 0, or holds a dose above
                ``beta_scale``.
        """
        reject_masked(doses, "doses")
        values = checked_doses(np.atleast_1d(np.asarray(doses)), "doses")
        top = float(values.max()) if values.size else 0.0
        if top <= 0.0:
            raise ValueError(
                "doses must hold a dose above 0, which the shapes are scaled to; got "
                f"{dose_text(np.unique(values), '')}".rstrip()
            )
        if self.beta_scale is not None and top > self.beta_scale:
            raise ValueError(
                f"the beta shapes are defined up to their scale, beta_scale = "
                f"{self.beta_scale:g}, and doses reach {top:g}; give a beta_scale of at "
                "least the top dose"
            )
        rows = [_emax(values, ed50, top) for ed50 in self.emax]
        rows += [_sigmoid_emax(values, ed50, h, top) for ed50, h in self.sigmoid_emax]
        rows += [_exponential(values, delta, top) for delta in self.exponential]
        if self.beta_scale is not None:
            rows += [
                _beta(values, delta1, delta2, self.beta_scale)
                for delta1, delta2 in self.beta
            ]
        return np.array(rows, dtype=np.float64).reshape(len(rows), values.size)

    def to_dict(self) -> dict[str, Any]:
        """The shapes as plain values that JSON stores.

        ``CandidateShapes(**shapes.to_dict())`` rebuilds them.
        """
        return {
            "emax": list(self.emax),
            "sigmoid_emax": [list(pair) for pair in self.sigmoid_emax],
            "exponential": list(self.exponential),
            "beta": [list(pair) for pair in self.beta],
            "beta_scale": self.beta_scale,
        }

    @classmethod
    def spread(
        cls,
        axis: DoseAxis,
        samples: Any,
        *,
        n_emax: int = 2,
        n_sigmoid_emax: int = 1,
        n_exponential: int = 1,
        n_beta: int = 1,
    ) -> "CandidateShapes":
        """Place candidate shapes over the doses of a series.

        Between the lowest and highest treated dose of the samples, along the axis's
        positions (log10 of the dose, of the dose plus the offset when the vehicle
        sits on a log10 axis, or the dose on a linear axis), the j-th of m shapes of
        a family is placed at the fraction j / (m + 1) of the way:

        - Emax: ED50 there;
        - sigmoid Emax: ED50 there, with a Hill slope of 3, steeper than an Emax
          shape's slope of 1 along log dose;
        - exponential: half its change at the top dose D reached at
          (1/2 + j / (2 (m + 1))) D, on the dose scale, since an exponential shape
          reaches half its change only above D / 2 (0.75 D for one shape);
        - beta: the peak there, half the peak's change halfway between the lowest
          treated dose and the peak, along the axis, and the scale 1.2 D,
          DoseFinding's default.

        The defaults, two Emax, one sigmoid Emax, one exponential and one beta shape,
        give five shapes from four families, within the 3 to 7 shapes from 2 to 4
        families that DoseFinding's advice for MCP-Mod finds sufficient (Bornkamp et
        al. 2026): early, midway and late monotone rises, and an umbrella peaking
        mid-range.
        https://docs.nygen.io/CytePool/statistics/dose-response#dose-response-contrasts
        explains the rule.

        Args:
            axis: The :class:`~cytepool.DoseAxis` of the series.
            samples: The sample table, a ``Table``, a mapping of column name to 1-D
                values, or a pandas DataFrame, with the axis's dose column, such as
                ``design.samples``.
            n_emax: The number of Emax shapes, 0 or more.
            n_sigmoid_emax: The number of sigmoid Emax shapes, 0 or more.
            n_exponential: The number of exponential shapes, 0 or more.
            n_beta: The number of beta shapes, 0 or more.

        Returns:
            The shapes, in the dose's unit.

        Raises:
            TypeError: If ``axis`` is not a ``DoseAxis``, ``samples`` is not a table
                or holds a numpy masked array, its dose column is not numeric, or a
                count is not an integer.
            ValueError: If a count is negative or all are 0; the dose column is
                missing, holds a dose that is not finite or is negative, or has
                fewer than two distinct doses above 0.

        Example:
            >>> from cytepool import DoseAxis
            >>> axis = DoseAxis("dose", unit="ug/kg", metric="administered")
            >>> samples = {"dose": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 10.0, 10.0]}
            >>> shapes = CandidateShapes.spread(axis, samples)
            >>> shapes.names
            ('emax_1', 'emax_2', 'sigmoid_emax', 'exponential', 'beta')
            >>> [round(ed50, 4) for ed50 in shapes.emax]
            [0.4642, 2.1544]
        """
        if not isinstance(axis, DoseAxis):
            raise TypeError(f"axis must be a DoseAxis, got {type(axis).__name__}")
        counts = {
            family: _count(value, name)
            for family, name, value in (
                ("emax", "n_emax", n_emax),
                ("sigmoid_emax", "n_sigmoid_emax", n_sigmoid_emax),
                ("exponential", "n_exponential", n_exponential),
                ("beta", "n_beta", n_beta),
            )
        }
        if not any(counts.values()):
            raise ValueError(
                "n_emax, n_sigmoid_emax, n_exponential and n_beta are all 0; place at "
                "least one shape"
            )
        doses = sample_doses(samples, axis.column)
        treated = distinct_doses(doses[doses > 0.0], axis.unit)
        if treated.size < 2:
            found = dose_text(treated, axis.unit) if treated.size else "none"
            raise ValueError(
                "candidate shapes spread between the lowest and highest treated dose, "
                f"and the samples have fewer than two doses above 0: {found}"
            )
        top = float(treated[-1])
        low, high = axis.positions(treated[[0, -1]])

        def placed(count: int) -> list[float]:
            """The positions j / (count + 1) of the way, j = 1, ..., count."""
            return [low + j / (count + 1) * (high - low) for j in range(1, count + 1)]

        def dose_at(position: float) -> float:
            return float(axis.doses(np.array([position]))[0])

        beta_scale = BETA_SCALE_FACTOR * top if counts["beta"] else None
        beta = []
        for peak in placed(counts["beta"]):
            assert beta_scale is not None
            beta.append(
                _beta_parameters(dose_at((low + peak) / 2.0), dose_at(peak), beta_scale)
            )
        exponential = [
            _exponential_delta(
                (0.5 + j / (2.0 * (counts["exponential"] + 1))) * top, top
            )
            for j in range(1, counts["exponential"] + 1)
        ]
        return cls(
            emax=tuple(dose_at(p) for p in placed(counts["emax"])),
            sigmoid_emax=tuple(
                (dose_at(p), SPREAD_HILL_SLOPE) for p in placed(counts["sigmoid_emax"])
            ),
            exponential=tuple(exponential),
            beta=tuple(beta),
            beta_scale=beta_scale,
        )


# The shapes, each scaled to a change of 1 at the top dose, or at the beta's peak.


def _emax(d: FloatArray, ed50: float, top: float) -> FloatArray:
    # d / (ED50 + d) over its value at the top dose.
    result: FloatArray = d / (ed50 + d) * ((ed50 + top) / top)
    return result


def _sigmoid_emax(d: FloatArray, ed50: float, h: float, top: float) -> FloatArray:
    # d^h / (ED50^h + d^h) = 1 / (1 + (ED50 / d)^h), whose power overflows to infinity
    # (a change of 0) far below ED50, over its value at the top dose.
    with np.errstate(divide="ignore", over="ignore"):
        ratio = np.exp(h * (math.log(ed50) - np.log(d)))
    level = 1.0 / (1.0 + ratio)
    at_top = 1.0 / (1.0 + math.exp(h * (math.log(ed50) - math.log(top))))
    result: FloatArray = level / at_top
    return result


def _exponential(d: FloatArray, delta: float, top: float) -> FloatArray:
    # (exp(d / delta) - 1) / (exp(D / delta) - 1), written so that neither a small
    # delta (overflow) nor a large one (cancellation) loses digits.
    result: FloatArray = (
        np.exp((d - top) / delta) * np.expm1(-d / delta) / math.expm1(-top / delta)
    )
    return result


def _beta(d: FloatArray, delta1: float, delta2: float, scale: float) -> FloatArray:
    # In logarithms, over its value at the peak, s delta1 / (delta1 + delta2), wherever
    # the peak lies, as DoseFinding scales it.
    def log_level(x: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            values: FloatArray = delta1 * np.log(x / scale) + delta2 * np.log1p(
                -x / scale
            )
        return values

    peak = scale * delta1 / (delta1 + delta2)
    reference = float(log_level(np.array([peak]))[0])
    result: FloatArray = np.exp(log_level(d) - reference)
    return result


# The parameters that spread places.


def _exponential_delta(half: float, top: float) -> float:
    """The delta of an exponential shape that reaches half its change at the top dose
    at the dose ``half``, between top / 2 and top: the root of
    expm1(half / delta) / expm1(top / delta) = 1/2, unique since the ratio falls from
    half / top to 0 as top / delta grows."""
    fraction = half / top

    def excess(x: float) -> float:
        # The ratio at top / delta = x, minus 1/2; exp((fraction - 1) x) beyond 700.
        if x > 700.0:
            return math.exp((fraction - 1.0) * x) - 0.5
        return math.expm1(fraction * x) / math.expm1(x) - 0.5

    low, high = 0.0, 1.0
    while excess(high) > 0.0:
        low, high = high, 2.0 * high
    # Bisection to the last bit: the ratio's slope is bounded away from 0 at the root.
    while True:
        middle = 0.5 * (low + high)
        if middle in (low, high):
            break
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return top / (0.5 * (low + high))


def _beta_parameters(half: float, peak: float, scale: float) -> tuple[float, float]:
    """(delta1, delta2) of the beta shape of scale s that peaks at ``peak`` and is half
    its peak at ``half``, below it: with r = peak / s and k = (1 - r) / r, delta2 = k
    delta1, and (half / peak)^delta1 ((s - half) / (s - peak))^delta2 = 1/2."""
    r = peak / scale
    k = (1.0 - r) / r
    delta1 = math.log(0.5) / (
        math.log(half / peak) + k * math.log((scale - half) / (scale - peak))
    )
    return delta1, k * delta1


# Reading the arguments.


def _parameter(value: Any, name: str) -> float:
    number = real_number(value, name)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be finite and above 0, got {number}")
    return number


def _entries(value: Any, name: str, example: str) -> list[Any]:
    reject_masked(value, name)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence | np.ndarray):
        raise TypeError(
            f"{name} must be a list with one entry per shape, such as {example}, got "
            f"{type(value).__name__}"
        )
    return list(value)


def _numbers(value: Any, name: str, what: str) -> tuple[float, ...]:
    entries = _entries(value, name, f"{name}=[0.5] for one shape")
    return tuple(
        _parameter(entry, f"the {what} of {name} shape {j + 1}")
        for j, entry in enumerate(entries)
    )


def _pairs(
    value: Any, name: str, what: tuple[str, str]
) -> tuple[tuple[float, float], ...]:
    example = f"{name}=[({what[0].lower().replace(' ', '_')}, ...)]"
    entries = _entries(value, name, example)
    pairs = []
    for j, entry in enumerate(entries):
        if isinstance(entry, str | bytes) or not (
            isinstance(entry, Sequence | np.ndarray) and len(entry) == 2
        ):
            raise TypeError(
                f"{name} must be a list of ({what[0]}, {what[1]}) pairs, one per shape; "
                f"entry {j + 1} is {entry!r}; wrap one shape as "
                f"{name}=[({what[0].lower().replace(' ', '_')}, "
                f"{what[1].lower().replace(' ', '_')})]"
            )
        pairs.append(
            (
                _parameter(entry[0], f"the {what[0]} of {name} shape {j + 1}"),
                _parameter(entry[1], f"the {what[1]} of {name} shape {j + 1}"),
            )
        )
    return tuple(pairs)


def _count(value: Any, name: str) -> int:
    count = integral_number(value, name)
    if count < 0:
        raise ValueError(f"{name} must be 0 or more, got {count}")
    return count
