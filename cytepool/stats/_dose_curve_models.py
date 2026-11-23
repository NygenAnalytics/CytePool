"""The dose-response models of :mod:`cytepool.stats.dose_curves`: their means, analytic
gradients, bounds and the grids of their nonlinear parameters.

Each model has a kind, an integer that the compiled routines switch on, and a vector of
internal parameters theta, ordered as its reported parameters, on which its bounds are
a box. For every kind but DRomics' exponential, theta is the reported vector; that
model is searched as d + beta (exp(s x) - 1) / s, which is smooth in s, and reported as
DRomics writes it, b = beta / s and e = 1 / s. The parameters split into q linear ones,
solved exactly at each point of a grid of the r nonlinear ones (variable projection),
and the nonlinear ones.

Means at the limits where floating point gives 0/0 or 0 times infinity take their
limits: a Hill or log-logistic term is 0 at dose 0 for any location, and 1 at doses
above 0 when the location is 0; x^n log x, (b x)^d log(b x) and z exp(-z^2/2) are 0
where their first factor is 0. Exponentials are capped at infinity rather than
overflowing, so that a grid point whose mean overflows is skipped.

References:
    US EPA (2026). BMDS User Guide: continuous endpoints. Online documentation of
    BMDS Online, BMDS Desktop and pybmds 25.2.
    https://usepa.github.io/bmds-user-guide/continuous.html

    Ritz C, Baty F, Streibig JC, Gerhard D (2015). Dose-response analysis using R.
    PLOS ONE 10, e0146021. doi:10.1371/journal.pone.0146021

    Larras F, Billoir E, Baillard V, Siberchicot A, Scholz S, Wubet T, Tarkka M,
    Schmitt-Jansen M, Delignette-Muller ML (2018). DRomics: a turnkey tool to support
    the use of the dose-response framework for omics data in ecological risk
    assessment. Environmental Science & Technology 52, 14461-14468.
    doi:10.1021/acs.est.8b04752
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit_numpy_errors
from cytepool.stats.distributions import norm_cdf_scalar

FloatArray = npt.NDArray[np.float64]

# The kinds of model.
LINE, POLYNOMIAL, POWER, HILL = 0, 1, 2, 3
EXP2, EXP3, EXP4, EXP5 = 4, 5, 6, 7
LOG_LOGISTIC, DROMICS_EXPONENTIAL = 8, 9
GAUSS_PROBIT, GAUSS_PROBIT_MONOTONE, GAUSS_PROBIT_SYMMETRIC = 10, 11, 12
LOG_GAUSS_PROBIT, LOG_GAUSS_PROBIT_MONOTONE, LOG_GAUSS_PROBIT_SYMMETRIC = 13, 14, 15
_GAUSS = (GAUSS_PROBIT, GAUSS_PROBIT_MONOTONE, GAUSS_PROBIT_SYMMETRIC)
_LOG_GAUSS = (LOG_GAUSS_PROBIT, LOG_GAUSS_PROBIT_MONOTONE, LOG_GAUSS_PROBIT_SYMMETRIC)
TURNING = (
    GAUSS_PROBIT,
    GAUSS_PROBIT_SYMMETRIC,
    LOG_GAUSS_PROBIT,
    LOG_GAUSS_PROBIT_SYMMETRIC,
)
"""The kinds whose slope can change sign within the doses; the others are monotone
for doses at least 0."""

# How the linear parameters are constrained: not at all; one parameter at least 0 (the
# a of exponential 2 and 3); the cone of exponential 4 and 5, a >= 0 with
# c = 1 + alpha_2 / a in [0, C_MAX]; or the last two of one sign (b1 and b2 of the
# restricted quadratic).
FREE, NON_NEGATIVE, CONE, SAME_SIGN = 0, 1, 2, 3
# BMDS states no upper bound on c for a rising curve. This one keeps finite the fits
# that approach a straight line, where c grows without bound as b shrinks, and lies far
# beyond the ratio of two log-scale levels: from 0.02 to 20 log2 CPM is c = 1,000.
C_MAX = 1000.0

# The bounds of BMDS's continuous models (US EPA 2026; pybmds 25.2's defaults), written
# on doses in their unit where BMDS states them on doses divided by the top dose, with
# C_MAX above.
EXPONENT_MIN, EXPONENT_MAX = 1.0, 18.0
LOCATION_MAX = 5.0  # times the top dose
RATE_MAX = 100.0  # divided by the top dose
LOG_LOGISTIC_SLOPE_MIN = 0.1
RATE_MIN = 1e-4  # of DRomics' exponential, divided by the top dose
LOCATION_GRID_MIN = 0.01  # of the grids of locations, times the lowest positive dose
WIDTH_MIN = 1.0 / 3.0  # of a bell's width, times the largest gap between doses
WIDTH_MAX = 10.0  # of a bell's width, times the span of the doses

# The sizes of the grids of nonlinear parameters.
GRID_SIZES = {
    "locations": 48,
    "exponents": 24,
    "power_exponents": 64,
    "rates": 64,
    "slopes": 24,
    "widths": 24,
}


@dataclass(frozen=True)
class Model:
    """A model of a set: its name, kind, reported parameters (in the order of theta)
    and mean, as the documentation writes it."""

    name: str
    kind: int
    parameters: tuple[str, ...]
    formula: str

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)


_GAUSS_TEXT = "z = (x - e) / b, Phi the standard normal distribution function"
_LOG_GAUSS_TEXT = "z = (ln x - ln e) / b, and d at dose 0"

SETS: dict[str, tuple[Model, ...]] = {
    "epa": (
        Model("linear", LINE, ("g", "b1"), "g + b1 x"),
        Model("polynomial_2", POLYNOMIAL, ("g", "b1", "b2"), "g + b1 x + b2 x^2"),
        Model("power", POWER, ("g", "v", "n"), "g + v x^n"),
        Model("hill", HILL, ("g", "v", "k", "n"), "g + v x^n / (k^n + x^n)"),
        Model("exponential_2", EXP2, ("a", "b"), "a exp(b x)"),
        Model("exponential_3", EXP3, ("a", "b", "d"), "a exp(sign(b) (|b| x)^d)"),
        Model("exponential_4", EXP4, ("a", "b", "c"), "a (c - (c - 1) exp(-b x))"),
        Model(
            "exponential_5",
            EXP5,
            ("a", "b", "c", "d"),
            "a (c - (c - 1) exp(-(b x)^d))",
        ),
    ),
    "ll4": (
        Model("ll4", LOG_LOGISTIC, ("b", "c", "d", "e"), "c + (d - c) / (1 + (x/e)^b)"),
    ),
    "dromics": (
        Model("linear", LINE, ("d", "b"), "d + b x"),
        Model(
            "exponential",
            DROMICS_EXPONENTIAL,
            ("b", "d", "e"),
            "d + b (exp(x / e) - 1)",
        ),
        Model("ll4", LOG_LOGISTIC, ("b", "c", "d", "e"), "c + (d - c) / (1 + (x/e)^b)"),
        Model(
            "gauss_probit",
            GAUSS_PROBIT,
            ("b", "c", "d", "e", "f"),
            f"f exp(-z^2/2) + d + (c - d) Phi(z), {_GAUSS_TEXT}",
        ),
        Model(
            "gauss_probit_monotone",
            GAUSS_PROBIT_MONOTONE,
            ("b", "c", "d", "e"),
            f"d + (c - d) Phi(z), {_GAUSS_TEXT}",
        ),
        Model(
            "gauss_probit_symmetric",
            GAUSS_PROBIT_SYMMETRIC,
            ("b", "d", "e", "f"),
            f"f exp(-z^2/2) + d, {_GAUSS_TEXT}",
        ),
        Model(
            "log_gauss_probit",
            LOG_GAUSS_PROBIT,
            ("b", "c", "d", "e", "f"),
            f"f exp(-z^2/2) + d + (c - d) Phi(z), {_LOG_GAUSS_TEXT}",
        ),
        Model(
            "log_gauss_probit_monotone",
            LOG_GAUSS_PROBIT_MONOTONE,
            ("b", "c", "d", "e"),
            f"d + (c - d) Phi(z), {_LOG_GAUSS_TEXT}",
        ),
        Model(
            "log_gauss_probit_symmetric",
            LOG_GAUSS_PROBIT_SYMMETRIC,
            ("b", "d", "e", "f"),
            f"f exp(-z^2/2) + d, {_LOG_GAUSS_TEXT}",
        ),
    ),
}

SET_PARAMETERS: dict[str, tuple[str, ...]] = {
    "epa": ("g", "v", "k", "n", "b1", "b2", "a", "b", "c", "d"),
    "ll4": ("b", "c", "d", "e"),
    "dromics": ("b", "c", "d", "e", "f"),
}
"""The parameter columns of each set's table, in order."""

POSITIVE_MEANS = (EXP2, EXP3, EXP4, EXP5)
"""The kinds whose means are positive, not fitted where a dose-group mean is not."""


def model_of(model_set: str, name: str) -> Model:
    """A model of a set, by its name."""
    return next(model for model in SETS[model_set] if model.name == name)


def n_linear(kind: int) -> int:
    """The linear parameters q of a kind that the grid solves exactly."""
    if kind in (EXP2, EXP3):
        return 1
    if kind in (GAUSS_PROBIT, LOG_GAUSS_PROBIT, POLYNOMIAL):
        return 3
    return 2


def constraint(kind: int) -> int:
    """How a kind's linear parameters are constrained."""
    if kind in (EXP2, EXP3):
        return NON_NEGATIVE
    if kind in (EXP4, EXP5):
        return CONE
    if kind == POLYNOMIAL:
        return SAME_SIGN
    return FREE


# ---------------------------------------------------------------------------------------
# Bounds and grids, from the doses
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseScale:
    """What bounds and grids are made of: the top dose D, the lowest positive dose,
    the largest gap between adjacent doses, and the same on ln dose over the
    positive doses."""

    top: float
    lowest: float
    gap: float
    log_span: float
    log_gap: float

    @classmethod
    def of(cls, doses: FloatArray) -> "DoseScale":
        levels = np.unique(doses)
        positive = levels[levels > 0.0]
        logs = np.log(positive)
        return cls(
            top=float(levels[-1]),
            lowest=float(positive[0]),
            gap=float(np.max(np.diff(levels))),
            log_span=float(logs[-1] - logs[0]),
            log_gap=float(np.max(np.diff(logs))) if logs.size > 1 else math.inf,
        )


def bounds(kind: int, scale: DoseScale) -> tuple[FloatArray, FloatArray]:
    """The lower and upper bounds of a kind's internal parameters."""
    inf = math.inf
    top = scale.top
    rate = RATE_MAX / top
    location = LOCATION_MAX * top
    exponents = (EXPONENT_MIN, EXPONENT_MAX)
    if kind == LINE:
        pairs = [(-inf, inf), (-inf, inf)]
    elif kind == POLYNOMIAL:
        pairs = [(-inf, inf), (-inf, inf), (-inf, inf)]
    elif kind == POWER:
        pairs = [(-inf, inf), (-inf, inf), exponents]
    elif kind == HILL:
        pairs = [(-inf, inf), (-inf, inf), (0.0, location), exponents]
    elif kind == EXP2:
        pairs = [(0.0, inf), (-rate, rate)]
    elif kind == EXP3:
        pairs = [(0.0, inf), (-rate, rate), exponents]
    elif kind == EXP4:
        pairs = [(0.0, inf), (0.0, rate), (0.0, C_MAX)]
    elif kind == EXP5:
        pairs = [(0.0, inf), (0.0, rate), (0.0, C_MAX), exponents]
    elif kind == LOG_LOGISTIC:
        low = LOCATION_GRID_MIN * scale.lowest
        pairs = [
            (LOG_LOGISTIC_SLOPE_MIN, EXPONENT_MAX),
            (-inf, inf),
            (-inf, inf),
            (low, location),
        ]
    elif kind == DROMICS_EXPONENTIAL:
        # s keeps the sign of its start, away from 0 (exponential_sign_gap).
        pairs = [(-inf, inf), (-inf, inf), (-rate, rate)]
    else:
        log = kind in _LOG_GAUSS
        # A peak or inflection within the doses, and a width of at least a third of
        # the largest gap: every point of the range lies within half a gap of a
        # dose, where a bell keeps exp(-9/8) = 32% of its height or more, so that no
        # bell hides between two doses or beyond them with a diverging height.
        width = (
            (WIDTH_MIN * scale.log_gap, WIDTH_MAX * scale.log_span)
            if log
            else (WIDTH_MIN * scale.gap, WIDTH_MAX * top)
        )
        peak = (scale.lowest if log else 0.0, top)
        free = (-inf, inf)
        if kind in (GAUSS_PROBIT, LOG_GAUSS_PROBIT):
            pairs = [width, free, free, peak, free]  # b, c, d, e, f
        elif kind in (GAUSS_PROBIT_MONOTONE, LOG_GAUSS_PROBIT_MONOTONE):
            pairs = [width, free, free, peak]  # b, c, d, e
        else:
            pairs = [width, free, peak, free]  # b, d, e, f
    lower = np.array([pair[0] for pair in pairs], dtype=np.float64)
    upper = np.array([pair[1] for pair in pairs], dtype=np.float64)
    return lower, upper


def exponential_sign_gap(scale: DoseScale) -> float:
    """The smallest |s| of DRomics' exponential: below it the curve is a straight
    line to within 5e-5 of its change over the doses."""
    return RATE_MIN / scale.top


def _log_points(low: float, high: float, n: int) -> FloatArray:
    """n points evenly spaced on log scale from low to high, both included."""
    points: FloatArray = np.exp(np.linspace(math.log(low), math.log(high), n))
    points[0], points[-1] = low, high
    return points


def grid(kind: int, scale: DoseScale) -> tuple[FloatArray, int]:
    """The grid of a kind's nonlinear parameters, one row per point, in the order of
    theta, and the size of its second axis: every grid is the product of one or two
    axes, the second varying fastest, so that each point's neighbours are known.
    Linear and polynomial models have none."""
    sizes = GRID_SIZES
    top = scale.top
    lower, upper = bounds(kind, scale)
    locations = _log_points(
        LOCATION_GRID_MIN * scale.lowest, LOCATION_MAX * top, sizes["locations"]
    )
    exponents = _log_points(EXPONENT_MIN, EXPONENT_MAX, sizes["exponents"])
    rates = _log_points(RATE_MIN / top, RATE_MAX / top, sizes["rates"])
    if kind == POWER:
        points = _log_points(EXPONENT_MIN, EXPONENT_MAX, sizes["power_exponents"])
        return points[:, None], 1
    if kind == HILL:
        return _pairs(np.r_[0.0, locations], exponents), exponents.size
    if kind == EXP2:
        return np.r_[-rates[::-1], 0.0, rates][:, None], 1
    if kind == EXP3:
        return _pairs(np.r_[-rates[::-1], 0.0, rates], exponents), exponents.size
    if kind == EXP4:
        return np.r_[0.0, rates][:, None], 1
    if kind == EXP5:
        return _pairs(np.r_[0.0, rates], exponents), exponents.size
    if kind == LOG_LOGISTIC:
        slopes = _log_points(LOG_LOGISTIC_SLOPE_MIN, EXPONENT_MAX, sizes["slopes"])
        return _pairs(slopes, locations), locations.size
    if kind == DROMICS_EXPONENTIAL:
        return np.r_[-rates[::-1], rates][:, None], 1
    widths = _log_points(lower[0], upper[0], sizes["widths"])
    peaks = _log_points(scale.lowest, top, sizes["locations"])
    peaks = peaks if kind in _LOG_GAUSS else np.r_[0.0, peaks]
    return _pairs(widths, peaks), peaks.size


def _pairs(first: FloatArray, second: FloatArray) -> FloatArray:
    """Every pair of a value of ``first`` and a value of ``second``."""
    a, b = np.meshgrid(first, second, indexing="ij")
    return np.ascontiguousarray(np.column_stack([a.ravel(), b.ravel()]))


# ---------------------------------------------------------------------------------------
# Reported parameters and turns
# ---------------------------------------------------------------------------------------


def reported(kind: int, theta: FloatArray) -> FloatArray:
    """The reported parameters of internal ones, ``(..., p)``: DRomics' exponential
    as b = beta / s and e = 1 / s; the others unchanged."""
    if kind != DROMICS_EXPONENTIAL:
        return theta
    out = theta.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., 0] = theta[..., 0] / theta[..., 2]
        out[..., 2] = 1.0 / theta[..., 2]
    return out


def internal(kind: int, parameters: FloatArray) -> FloatArray:
    """The internal parameters of reported ones, ``(..., p)``: the inverse of
    ``reported``."""
    if kind != DROMICS_EXPONENTIAL:
        return parameters
    out = parameters.copy()
    out[..., 0] = parameters[..., 0] / parameters[..., 2]
    out[..., 2] = 1.0 / parameters[..., 2]
    return out


def turn_doses(kind: int, theta: FloatArray, top: float) -> FloatArray:
    """The dose x* at which the slope of each fit changes sign, where it lies strictly
    between 0 and the top dose; NaN for fits that do not turn there and for monotone
    kinds. A curve whose slope changes sign at the top dose, or at 0, is monotone over
    the doses.

    The slope of a Gauss-probit curve is exp(-z^2/2) (-f z + (c - d) / sqrt(2 pi))
    times dz/dx > 0, which is 1 / b, or 1 / (b x) on ln dose, so it changes sign once,
    at z* = (c - d) / (f sqrt(2 pi)): x* is e + b z*, or e exp(b z*) on ln dose, and e
    when c = d.
    """
    out = np.full(theta.shape[:-1], np.nan)
    if kind not in TURNING:
        return out
    b, e = (
        theta[..., 0],
        theta[..., 3 if kind in (GAUSS_PROBIT, LOG_GAUSS_PROBIT) else 2],
    )
    if kind in (GAUSS_PROBIT, LOG_GAUSS_PROBIT):
        c, d, f = theta[..., 1], theta[..., 2], theta[..., 4]
    else:
        f = theta[..., 3]
        c = d = np.zeros_like(f)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (c - d) / (f * math.sqrt(2.0 * math.pi))
        # A symmetric curve turns at e itself: taken as it is, a peak at the top
        # dose's bound stays there whatever the rounding of exp.
        x = (
            np.where(z == 0.0, e, e * np.exp(b * z))
            if kind in _LOG_GAUSS
            else e + b * z
        )
    turns = (f != 0.0) & np.isfinite(x) & (x > 0.0) & (x < top)
    out[turns] = x[turns]
    return out


# ---------------------------------------------------------------------------------------
# Compiled means, gradients and bases
# ---------------------------------------------------------------------------------------

_EXP_MAX = 709.0
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@jit_numpy_errors
def safe_exp(t: float) -> float:
    """exp(t), infinity beyond the largest double instead of an overflow."""
    if t > _EXP_MAX:
        return math.inf
    return math.exp(t)


@jit_numpy_errors
def logistic(t: float) -> float:
    """1 / (1 + exp(-t)) without overflow."""
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    u = math.exp(t)
    return u / (1.0 + u)


@jit_numpy_errors
def _hill_pair(x: float, location: float, exponent: float) -> tuple[float, float]:
    """h = x^n / (k^n + x^n) and 1 - h, each without cancellation: h is 0 at dose 0,
    and 1 above it when k is 0."""
    if x <= 0.0:
        return 0.0, 1.0
    if location <= 0.0:
        return 1.0, 0.0
    t = exponent * (math.log(x) - math.log(location))
    return logistic(t), logistic(-t)


@jit_numpy_errors
def _probit_pair(z: float) -> tuple[float, float]:
    """Phi(z) and 1 - Phi(z), the second as Phi(-z)."""
    if math.isinf(z):
        return (0.0, 1.0) if z < 0.0 else (1.0, 0.0)
    return norm_cdf_scalar(z), norm_cdf_scalar(-z)


@jit_numpy_errors
def _power_log(x: float, exponent: float) -> tuple[float, float]:
    """x^n and x^n ln x, both 0 at x = 0 (n >= 1)."""
    if x <= 0.0:
        return 0.0, 0.0
    lx = math.log(x)
    value = safe_exp(exponent * lx)
    return value, value * lx


@jit_numpy_errors
def _exp_rate_term(rate: float, x: float) -> float:
    """(exp(s x) - 1) / s, x at s = 0, and infinity beyond the largest double, as
    ``safe_exp`` gives it, instead of an overflow."""
    if rate == 0.0:
        return x
    if rate * x > _EXP_MAX:
        return math.inf
    return math.expm1(rate * x) / rate


@jit_numpy_errors
def _exp_rate_derivative(rate: float, x: float) -> float:
    """The derivative of (exp(s x) - 1) / s in s, x^2 / 2 at s = 0."""
    w = rate * x
    if abs(w) < 1e-2:
        # x^2 sum_n (n - 1) w^(n - 2) / n! over n >= 2, to n = 8: the first term left
        # out is below 1e-18 of the sum, where the closed form below would lose the
        # digits of its difference.
        return (
            x
            * x
            * (
                0.5
                + w
                * (
                    1.0 / 3.0
                    + w
                    * (
                        1.0 / 8.0
                        + w
                        * (
                            1.0 / 30.0
                            + w * (1.0 / 144.0 + w * (1.0 / 840.0 + w / 5760.0))
                        )
                    )
                )
            )
        )
    return (x * math.exp(w) - math.expm1(w) / rate) / rate


@jit_numpy_errors
def _probit_z(kind: int, x: float, width: float, peak: float) -> float:
    """z of a Gauss-probit kind: (x - e) / b, or (ln x - ln e) / b on ln dose, where
    dose 0 gives minus infinity."""
    if kind >= LOG_GAUSS_PROBIT:
        if x <= 0.0:
            return -math.inf
        return (math.log(x) - math.log(peak)) / width
    return (x - peak) / width


@jit_numpy_errors
def _bell(z: float) -> float:
    """exp(-z^2/2), 0 at infinite z."""
    if math.isinf(z):
        return 0.0
    return math.exp(-0.5 * z * z)


@jit_numpy_errors
def basis_row(kind: int, phi0: float, phi1: float, x: float, out: FloatArray) -> None:
    """The columns of the linear parameters at dose x for the nonlinear parameters
    (phi0, phi1), in the order in which ``assemble`` reads them."""
    if kind == LINE:
        out[0], out[1] = 1.0, x
    elif kind == POLYNOMIAL:
        out[0], out[1], out[2] = 1.0, x, x * x
    elif kind == POWER:
        out[0], out[1] = 1.0, _power_log(x, phi0)[0]
    elif kind == HILL:
        out[0], out[1] = 1.0, _hill_pair(x, phi0, phi1)[0]
    elif kind == EXP2:
        out[0] = safe_exp(phi0 * x)
    elif kind == EXP3:
        t = abs(phi0) * x
        power = 0.0 if t <= 0.0 else safe_exp(phi1 * math.log(t))
        out[0] = safe_exp(power if phi0 >= 0.0 else -power)
    elif kind == EXP4:
        out[0], out[1] = 1.0, -math.expm1(-phi0 * x)
    elif kind == EXP5:
        t = phi0 * x
        power = 0.0 if t <= 0.0 else safe_exp(phi1 * math.log(t))
        out[0], out[1] = 1.0, -math.expm1(-power)
    elif kind == LOG_LOGISTIC:
        # r = 1 / (1 + (x/e)^b), the weight of d; phi = (b, e).
        h, r = _hill_pair(x, phi1, phi0)
        out[0], out[1] = h, r
    elif kind == DROMICS_EXPONENTIAL:
        out[0], out[1] = 1.0, _exp_rate_term(phi0, x)
    else:
        z = _probit_z(kind, x, phi0, phi1)
        if kind == GAUSS_PROBIT or kind == LOG_GAUSS_PROBIT:
            p, q = _probit_pair(z)
            out[0], out[1], out[2] = p, q, _bell(z)
        elif kind == GAUSS_PROBIT_MONOTONE or kind == LOG_GAUSS_PROBIT_MONOTONE:
            p, q = _probit_pair(z)
            out[0], out[1] = p, q
        else:
            out[0], out[1] = 1.0, _bell(z)


@jit_numpy_errors
def assemble(
    kind: int, phi0: float, phi1: float, alpha: FloatArray, theta: FloatArray
) -> None:
    """The internal parameters theta of a grid point and its linear parameters."""
    if kind == LINE:
        theta[0], theta[1] = alpha[0], alpha[1]
    elif kind == POLYNOMIAL:
        theta[0], theta[1], theta[2] = alpha[0], alpha[1], alpha[2]
    elif kind == POWER:
        theta[0], theta[1], theta[2] = alpha[0], alpha[1], phi0
    elif kind == HILL:
        theta[0], theta[1], theta[2], theta[3] = alpha[0], alpha[1], phi0, phi1
    elif kind == EXP2:
        theta[0], theta[1] = alpha[0], phi0
    elif kind == EXP3:
        theta[0], theta[1], theta[2] = alpha[0], phi0, phi1
    elif kind == EXP4 or kind == EXP5:
        a = alpha[0]
        c = 1.0 + alpha[1] / a if a > 0.0 else 1.0
        theta[0], theta[1], theta[2] = a, phi0, min(max(c, 0.0), C_MAX)
        if kind == EXP5:
            theta[3] = phi1
    elif kind == LOG_LOGISTIC:
        theta[0], theta[1], theta[2], theta[3] = phi0, alpha[0], alpha[1], phi1
    elif kind == DROMICS_EXPONENTIAL:
        theta[0], theta[1], theta[2] = alpha[1], alpha[0], phi0
    elif kind == GAUSS_PROBIT or kind == LOG_GAUSS_PROBIT:
        theta[0], theta[1], theta[2] = phi0, alpha[0], alpha[1]
        theta[3], theta[4] = phi1, alpha[2]
    elif kind == GAUSS_PROBIT_MONOTONE or kind == LOG_GAUSS_PROBIT_MONOTONE:
        theta[0], theta[1], theta[2], theta[3] = phi0, alpha[0], alpha[1], phi1
    else:
        theta[0], theta[1], theta[2], theta[3] = phi0, alpha[0], phi1, alpha[1]


@jit_numpy_errors
def is_linear(kind: int, j: int) -> bool:
    """Whether internal parameter j of a kind moves with the linear parameters that
    the grid solves exactly: the levels and heights, and the a and c of exponential
    4 and 5, which span the same columns as a and a (c - 1)."""
    if (
        kind == LOG_LOGISTIC
        or kind == GAUSS_PROBIT_MONOTONE
        or kind == LOG_GAUSS_PROBIT_MONOTONE
    ):
        return j == 1 or j == 2
    if kind == GAUSS_PROBIT or kind == LOG_GAUSS_PROBIT:
        return j == 1 or j == 2 or j == 4
    if kind == GAUSS_PROBIT_SYMMETRIC or kind == LOG_GAUSS_PROBIT_SYMMETRIC:
        return j == 1 or j == 3
    if kind == EXP4 or kind == EXP5:
        return j == 0 or j == 2
    if kind == EXP2 or kind == EXP3:
        return j == 0
    return j == 0 or j == 1  # power, Hill, DRomics' exponential


@jit_numpy_errors
def nonlinear_of(kind: int, theta: FloatArray) -> tuple[float, float]:
    """The nonlinear parameters (phi0, phi1) of internal parameters theta, as the
    grid orders them; 0 where a kind has fewer, as the line and the quadratic have
    none."""
    if kind == POWER or kind == EXP2 or kind == EXP4:
        return theta[2] if kind == POWER else theta[1], 0.0
    if kind == HILL:
        return theta[2], theta[3]
    if kind == EXP3:
        return theta[1], theta[2]
    if kind == EXP5:
        return theta[1], theta[3]
    if kind == DROMICS_EXPONENTIAL:
        return theta[2], 0.0
    if kind == GAUSS_PROBIT_SYMMETRIC or kind == LOG_GAUSS_PROBIT_SYMMETRIC:
        return theta[0], theta[2]
    if kind == LINE or kind == POLYNOMIAL:
        return 0.0, 0.0
    return theta[0], theta[3]  # log-logistic and the other Gauss-probit kinds


@jit_numpy_errors
def mean_value(kind: int, theta: FloatArray, x: float) -> float:
    """The mean of a kind at dose x."""
    if kind == LINE:
        return float(theta[0] + theta[1] * x)
    if kind == POLYNOMIAL:
        return float(theta[0] + (theta[1] + theta[2] * x) * x)
    if kind == POWER:
        return float(theta[0] + theta[1] * _power_log(x, theta[2])[0])
    if kind == HILL:
        return float(theta[0] + theta[1] * _hill_pair(x, theta[2], theta[3])[0])
    if kind == EXP2:
        return float(theta[0] * safe_exp(theta[1] * x))
    if kind == EXP3:
        t = abs(theta[1]) * x
        power = 0.0 if t <= 0.0 else safe_exp(theta[2] * math.log(t))
        return float(theta[0] * safe_exp(power if theta[1] >= 0.0 else -power))
    if kind == EXP4:
        a, c = theta[0], theta[2]
        return float(a * (c - (c - 1.0) * math.exp(-theta[1] * x)))
    if kind == EXP5:
        a, c = theta[0], theta[2]
        t = theta[1] * x
        power = 0.0 if t <= 0.0 else safe_exp(theta[3] * math.log(t))
        return float(a * (c - (c - 1.0) * math.exp(-power)))
    if kind == LOG_LOGISTIC:
        h, r = _hill_pair(x, theta[3], theta[0])
        return float(theta[1] * h + theta[2] * r)
    if kind == DROMICS_EXPONENTIAL:
        return float(theta[1] + theta[0] * _exp_rate_term(theta[2], x))
    if kind == GAUSS_PROBIT or kind == LOG_GAUSS_PROBIT:
        z = _probit_z(kind, x, theta[0], theta[3])
        p, q = _probit_pair(z)
        return float(theta[4] * _bell(z) + theta[1] * p + theta[2] * q)
    if kind == GAUSS_PROBIT_MONOTONE or kind == LOG_GAUSS_PROBIT_MONOTONE:
        z = _probit_z(kind, x, theta[0], theta[3])
        p, q = _probit_pair(z)
        return float(theta[1] * p + theta[2] * q)
    z = _probit_z(kind, x, theta[0], theta[2])
    return float(theta[1] + theta[3] * _bell(z))


@jit_numpy_errors
def mean_gradient(kind: int, theta: FloatArray, x: float, out: FloatArray) -> None:
    """The derivatives of a kind's mean at dose x in its internal parameters."""
    if kind == LINE:
        out[0], out[1] = 1.0, x
    elif kind == POLYNOMIAL:
        out[0], out[1], out[2] = 1.0, x, x * x
    elif kind == POWER:
        value, log_term = _power_log(x, theta[2])
        out[0], out[1], out[2] = 1.0, value, theta[1] * log_term
    elif kind == HILL:
        k, n = theta[2], theta[3]
        h, rest = _hill_pair(x, k, n)
        out[0], out[1] = 1.0, h
        if x <= 0.0:
            out[2], out[3] = 0.0, 0.0
        elif k <= 0.0:
            # The limits at k = 0: x / (k + x) has derivative -1/x when n = 1.
            out[2] = -theta[1] / x if n == 1.0 else 0.0
            out[3] = 0.0
        else:
            spread = h * rest
            out[2] = -theta[1] * n * spread / k
            out[3] = -theta[1] * spread * (math.log(k) - math.log(x))
    elif kind == EXP2:
        e = safe_exp(theta[1] * x)
        out[0], out[1] = e, theta[0] * x * e
    elif kind == EXP3:
        b, d = theta[1], theta[2]
        t = abs(b) * x
        if t <= 0.0:
            out[0] = 1.0
            # (|b| x)^(d - 1) at t = 0 is 1 when d = 1, else 0.
            out[1] = theta[0] * x if d == 1.0 else 0.0
            out[2] = 0.0
        else:
            lt = math.log(t)
            power = safe_exp(d * lt)
            e = safe_exp(power if b >= 0.0 else -power)
            out[0] = e
            out[1] = theta[0] * e * d * safe_exp((d - 1.0) * lt) * x
            out[2] = theta[0] * e * (power if b >= 0.0 else -power) * lt
    elif kind == EXP4:
        a, b, c = theta[0], theta[1], theta[2]
        e = math.exp(-b * x)
        out[0] = c - (c - 1.0) * e
        out[1] = a * (c - 1.0) * x * e
        out[2] = a * (1.0 - e)
    elif kind == EXP5:
        a, b, c, d = theta[0], theta[1], theta[2], theta[3]
        t = b * x
        if t <= 0.0:
            power, rate_term, log_term = 0.0, (x if d == 1.0 else 0.0), 0.0
        else:
            lt = math.log(t)
            power = safe_exp(d * lt)
            rate_term = d * safe_exp((d - 1.0) * lt) * x
            log_term = power * lt
        e = math.exp(-power)
        out[0] = c - (c - 1.0) * e
        out[1] = a * (c - 1.0) * e * rate_term
        out[2] = a * (1.0 - e)
        out[3] = a * (c - 1.0) * e * log_term
    elif kind == LOG_LOGISTIC:
        b, c, d, e_ = theta[0], theta[1], theta[2], theta[3]
        h, r = _hill_pair(x, e_, b)  # h = (x/e)^b / (1 + (x/e)^b) = 1 - r
        out[1], out[2] = h, r
        if x <= 0.0:
            out[0], out[3] = 0.0, 0.0
        else:
            spread = h * r
            # dr/db = -r (1 - r) ln(x/e); dr/de = r (1 - r) b / e.
            out[0] = -(d - c) * spread * (math.log(x) - math.log(e_))
            out[3] = (d - c) * spread * b / e_
    elif kind == DROMICS_EXPONENTIAL:
        out[0] = _exp_rate_term(theta[2], x)
        out[1] = 1.0
        out[2] = theta[0] * _exp_rate_derivative(theta[2], x)
    else:
        full = kind == GAUSS_PROBIT or kind == LOG_GAUSS_PROBIT
        monotone = kind == GAUSS_PROBIT_MONOTONE or kind == LOG_GAUSS_PROBIT_MONOTONE
        width = theta[0]
        peak = theta[3] if (full or monotone) else theta[2]
        z = _probit_z(kind, x, width, peak)
        bell = _bell(z)
        p, q = _probit_pair(z)
        if full or monotone:
            c, d = theta[1], theta[2]
            f = theta[4] if full else 0.0
            slope = (c - d) * _INV_SQRT_2PI * bell - (
                f * z * bell if bell > 0.0 else 0.0
            )
        else:
            f = theta[3]
            slope = -f * z * bell if bell > 0.0 else 0.0
        # dz/db = -z / b; dz/de = -1/b, or -1/(b e) on ln dose; 0 at dose 0 on ln dose.
        if math.isinf(z):
            dz_width, dz_peak = 0.0, 0.0
        else:
            dz_width = -z / width
            dz_peak = -1.0 / width
            if kind >= LOG_GAUSS_PROBIT:
                dz_peak /= peak
        out[0] = slope * dz_width
        if full or monotone:
            out[1], out[2], out[3] = p, q, slope * dz_peak
            if full:
                out[4] = bell
        else:
            out[1], out[2], out[3] = 1.0, slope * dz_peak, bell


@jit_numpy_errors
def mean_values(kind: int, theta: FloatArray, x: FloatArray) -> FloatArray:
    """The means of a kind for each row of internal parameters ``(n, p)`` at each
    dose of ``x``, ``(n, len(x))``."""
    out = np.empty((theta.shape[0], x.size))
    for g in range(theta.shape[0]):
        for i in range(x.size):
            out[g, i] = mean_value(kind, theta[g], x[i])
    return out


def curve_values(model: Model, parameters: FloatArray, x: FloatArray) -> FloatArray:
    """A model's curves at the doses ``x`` for each row of reported parameters
    ``(n, p)``, ``(n, len(x))``."""
    theta = np.ascontiguousarray(internal(model.kind, parameters), dtype=np.float64)
    return mean_values(model.kind, theta, np.ascontiguousarray(x, dtype=np.float64))
