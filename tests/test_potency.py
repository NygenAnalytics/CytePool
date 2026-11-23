"""benchmark_doses and BenchmarkDoses: benchmark doses against closed forms written
here, bounds against the straight line's closed form and reference fixtures,
statuses, invariants and refusals."""

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from cytepool import BenchmarkResponse
from cytepool.stats.dose_curves import DoseCurveFits, fit_dose_curves
from cytepool.stats.potency import BenchmarkDoses, benchmark_doses

DATA = Path(__file__).parent / "data" / "benchmark_doses"
CHAPTER = Path(__file__).parents[1] / "docs" / "source" / "statistics" / "potency.md"
HALF_LOG = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
CHI_SQUARE_90 = 2.705543454095404  # chi-square(1) at 0.90


def replicated(doses: np.ndarray, reps: int) -> np.ndarray:
    return np.repeat(doses, reps)


def noisy(mean: np.ndarray, sd: float, seed: int, n_genes: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean[:, None] + rng.normal(0.0, sd, (mean.size, n_genes))


def with_parameters(
    fits: DoseCurveFits, model: str, parameters: dict[str, float]
) -> DoseCurveFits:
    """The fits with one model's parameters of every gene replaced."""
    position = fits.models.index(model)
    values = np.array(fits.parameters)
    values[:, position, :] = np.nan
    for name, value in parameters.items():
        values[:, position, fits.parameter_names.index(name)] = value
    return dataclasses.replace(fits, parameters=values)


# Closed forms of the BMD at an absolute change R, written from each mean.
def bmd_linear(p: dict[str, float], r: float) -> float:
    return r / abs(p["b1"])


def bmd_polynomial(p: dict[str, float], r: float) -> float:
    # b1 x + b2 x^2 = s R: the smaller positive root.
    b1, b2 = p["b1"], p["b2"]
    s = math.copysign(1.0, b1 if b1 != 0 else b2)
    root = math.sqrt(b1 * b1 + 4.0 * b2 * s * r)
    return min(x for x in ((-b1 + root) / (2 * b2), (-b1 - root) / (2 * b2)) if x > 0)


def bmd_power(p: dict[str, float], r: float) -> float:
    return (r / abs(p["v"])) ** (1.0 / p["n"])


def bmd_hill(p: dict[str, float], r: float) -> float:
    rho = r / abs(p["v"])
    return p["k"] * (rho / (1.0 - rho)) ** (1.0 / p["n"])


def bmd_exponential_2(p: dict[str, float], r: float) -> float:
    s = math.copysign(1.0, p["b"])
    return math.log1p(s * r / p["a"]) / p["b"]


def bmd_exponential_3(p: dict[str, float], r: float) -> float:
    s = math.copysign(1.0, p["b"])
    return (s * math.log1p(s * r / p["a"])) ** (1.0 / p["d"]) / abs(p["b"])


def bmd_exponential_4(p: dict[str, float], r: float) -> float:
    span = p["a"] * (p["c"] - 1.0)
    return -math.log1p(-r / abs(span)) / p["b"]


def bmd_exponential_5(p: dict[str, float], r: float) -> float:
    span = p["a"] * (p["c"] - 1.0)
    return (-math.log1p(-r / abs(span))) ** (1.0 / p["d"]) / p["b"]


def bmd_ll4(p: dict[str, float], r: float) -> float:
    rho = r / abs(p["c"] - p["d"])
    return p["e"] * (rho / (1.0 - rho)) ** (1.0 / p["b"])


def bmd_dromics_exponential(p: dict[str, float], r: float) -> float:
    s = math.copysign(1.0, p["b"] * p["e"])
    return p["e"] * math.log1p(s * r / p["b"]) if p["e"] > 0 else math.nan


EPA_CASES = [
    ("linear", {"g": 5.0, "b1": 0.04}, bmd_linear),
    ("linear", {"g": 5.0, "b1": -0.04}, bmd_linear),
    ("polynomial_2", {"g": 5.0, "b1": 0.02, "b2": 0.001}, bmd_polynomial),
    ("polynomial_2", {"g": 5.0, "b1": -0.02, "b2": -0.001}, bmd_polynomial),
    ("power", {"g": 5.0, "v": 0.1, "n": 1.7}, bmd_power),
    ("power", {"g": 5.0, "v": -0.2, "n": 3.0}, bmd_power),
    ("hill", {"g": 5.0, "v": 1.5, "k": 0.7, "n": 2.3}, bmd_hill),
    ("hill", {"g": 5.0, "v": -1.0, "k": 4.0, "n": 1.0}, bmd_hill),
    ("exponential_2", {"a": 5.0, "b": 0.05}, bmd_exponential_2),
    ("exponential_2", {"a": 5.0, "b": -0.05}, bmd_exponential_2),
    ("exponential_3", {"a": 5.0, "b": 0.08, "d": 1.8}, bmd_exponential_3),
    ("exponential_3", {"a": 5.0, "b": -0.08, "d": 1.8}, bmd_exponential_3),
    ("exponential_4", {"a": 5.0, "b": 0.6, "c": 1.4}, bmd_exponential_4),
    ("exponential_4", {"a": 5.0, "b": 0.6, "c": 0.7}, bmd_exponential_4),
    ("exponential_5", {"a": 5.0, "b": 0.6, "c": 1.4, "d": 2.5}, bmd_exponential_5),
]


@pytest.fixture(scope="module")
def epa_fits() -> DoseCurveFits:
    x = replicated(HALF_LOG, 3)
    mean = 5.0 + x / (1.0 + x)
    return fit_dose_curves(x, noisy(mean, 0.2, 1), models="epa")


@pytest.mark.parametrize(("model", "parameters", "closed_form"), EPA_CASES)
@pytest.mark.parametrize("change", [0.05, 0.4])
def test_benchmark_doses_of_known_curves(
    epa_fits: DoseCurveFits, model: str, parameters: dict, closed_form, change: float
) -> None:
    fits = with_parameters(epa_fits, model, parameters)
    result = benchmark_doses(
        fits, BenchmarkResponse.absolute(change), model=model, bounds=False
    )
    expected = closed_form(parameters, change)
    got = float(result.bmd[0, 0])
    assert got == pytest.approx(expected, rel=1e-12)
    rising = documented_change(model, parameters, expected) > 0
    assert result.direction[0, 0] == ("up" if rising else "down")
    lowest, top = 0.01, 30.0
    status = (
        "within the doses"
        if lowest <= expected <= top
        else "below the lowest dose"
        if expected < lowest
        else "above the top dose"
    )
    assert result.status[0, 0] == status


def documented_change(model: str, p: dict[str, float], x: float) -> float:
    """f(x) - f(0) of the EPA set's models, from their documented means."""
    if model == "linear":
        return p["b1"] * x
    if model == "polynomial_2":
        return p["b1"] * x + p["b2"] * x * x
    if model == "power":
        return p["v"] * x ** p["n"]
    if model == "hill":
        return p["v"] * x ** p["n"] / (p["k"] ** p["n"] + x ** p["n"])
    if model == "exponential_2":
        return p["a"] * math.expm1(p["b"] * x)
    if model == "exponential_3":
        s = math.copysign(1.0, p["b"])
        return p["a"] * math.expm1(s * (abs(p["b"]) * x) ** p["d"])
    if model == "exponential_4":
        return p["a"] * (p["c"] - 1.0) * -math.expm1(-p["b"] * x)
    if model == "exponential_5":
        return p["a"] * (p["c"] - 1.0) * -math.expm1(-((p["b"] * x) ** p["d"]))
    raise ValueError(model)


# The DRomics set, with the Gauss-probit forms' crossings found by bisection here.
def gauss_probit_change(model: str, p: dict[str, float], x: float) -> float:
    def mean(at: float) -> float:
        b, d, e = p["b"], p["d"], p["e"]
        c = p.get("c", d)
        f = p.get("f", 0.0)
        if model.startswith("log"):
            if at <= 0.0:
                return d
            z = (math.log(at) - math.log(e)) / b
        else:
            z = (at - e) / b
        phi = 0.5 * math.erfc(-z / math.sqrt(2.0))
        return f * math.exp(-z * z / 2.0) + d + (c - d) * phi

    return mean(x) - mean(0.0)


def first_root(change, r: float, low: float, high: float) -> float:
    """The lowest x in [low, high] with |change(x)| = r, from a grid of 20,001 doses on
    log dose and bisection."""
    grid = np.exp(np.linspace(math.log(low), math.log(high), 20001))
    values = np.array([abs(change(float(x))) - r for x in grid])
    j = int(np.flatnonzero(values >= 0)[0])
    a, b = math.log(grid[j - 1]), math.log(grid[j])
    for _ in range(200):
        m = 0.5 * (a + b)
        if abs(change(math.exp(m))) >= r:
            b = m
        else:
            a = m
    return math.exp(0.5 * (a + b))


@pytest.fixture(scope="module")
def dromics_fits() -> DoseCurveFits:
    x = replicated(HALF_LOG, 3)
    mean = 5.0 + x / (1.0 + x)
    return fit_dose_curves(x, noisy(mean, 0.2, 2), models="dromics")


DROMICS_CASES = [
    ("ll4", {"b": 1.8, "c": 7.0, "d": 5.0, "e": 0.5}),
    ("ll4", {"b": 0.9, "c": 3.5, "d": 5.0, "e": 2.0}),
    ("exponential", {"b": 0.3, "d": 5.0, "e": 8.0}),
    ("exponential", {"b": -0.3, "d": 5.0, "e": 8.0}),
    # A rate of 2 per dose unit overflows long before 50 times the top dose, rising
    # and falling.
    ("exponential", {"b": 0.3, "d": 5.0, "e": 0.5}),
    ("exponential", {"b": -0.3, "d": 5.0, "e": 0.5}),
    ("gauss_probit_monotone", {"b": 3.0, "c": 6.0, "d": 5.0, "e": 4.0}),
    ("log_gauss_probit_monotone", {"b": 1.2, "c": 4.2, "d": 5.0, "e": 0.3}),
    # Bells: the first crossing, before the peak.
    ("gauss_probit_symmetric", {"b": 4.0, "d": 5.0, "e": 12.0, "f": 1.0}),
    ("log_gauss_probit_symmetric", {"b": 1.0, "d": 5.0, "e": 0.3, "f": -1.2}),
    ("gauss_probit", {"b": 4.0, "c": 4.0, "d": 5.0, "e": 10.0, "f": 2.0}),
    ("log_gauss_probit", {"b": 1.0, "c": 5.6, "d": 5.0, "e": 0.3, "f": 1.5}),
]


@pytest.mark.parametrize(("model", "parameters"), DROMICS_CASES)
def test_dromics_models_and_bells(
    dromics_fits: DoseCurveFits, model: str, parameters: dict
) -> None:
    fits = with_parameters(dromics_fits, model, parameters)
    change = 0.4
    result = benchmark_doses(
        fits, BenchmarkResponse.absolute(change), model=model, bounds=False
    )
    if model == "ll4":
        expected = bmd_ll4(parameters, change)
    elif model == "exponential":
        expected = bmd_dromics_exponential(parameters, change)
    else:
        expected = first_root(
            lambda x: gauss_probit_change(model, parameters, x), change, 1e-4, 1500.0
        )
    assert float(result.bmd[0, 0]) == pytest.approx(expected, rel=1e-9)


def test_a_bell_crosses_first_before_its_turn(dromics_fits: DoseCurveFits) -> None:
    # A bell rising by 1 at 1 ug/kg: a change of 0.5 is reached twice, and the BMD is
    # the lower crossing; a change of 1.2 is never reached.
    bell = {"b": 0.8, "d": 5.0, "e": 1.0, "f": 1.0}
    fits = with_parameters(dromics_fits, "log_gauss_probit_symmetric", bell)
    result = benchmark_doses(
        fits,
        [BenchmarkResponse.absolute(0.5), BenchmarkResponse.absolute(1.2)],
        model="log_gauss_probit_symmetric",
        bounds=False,
    )
    lower = math.exp(math.log(1.0) - 0.8 * math.sqrt(2.0 * math.log(2.0)))
    assert float(result.bmd[0, 0]) == pytest.approx(lower, rel=1e-10)
    assert result.direction[0, 0] == "up"
    assert result.status[1, 0] == "not reached"
    assert math.isnan(result.bmd[1, 0])


def test_a_bell_that_crosses_back_reaches_the_other_direction(
    dromics_fits: DoseCurveFits,
) -> None:
    # Up by less than 0.5 near dose 0.1, then down to a plateau 1.0 below the level at
    # dose 0: a change of 0.5 is first reached on the way down, beyond the turn.
    bell = {"b": 0.7, "c": 4.0, "d": 5.0, "e": 0.3, "f": 0.5}
    fits = with_parameters(dromics_fits, "log_gauss_probit", bell)
    result = benchmark_doses(
        fits, BenchmarkResponse.absolute(0.5), model="log_gauss_probit", bounds=False
    )
    expected = first_root(
        lambda x: gauss_probit_change("log_gauss_probit", bell, x), 0.5, 1e-4, 1500.0
    )
    assert gauss_probit_change("log_gauss_probit", bell, expected) < 0
    assert float(result.bmd[0, 0]) == pytest.approx(expected, rel=1e-9)
    assert result.direction[0, 0] == "down"


def test_statuses_at_the_search_range(epa_fits: DoseCurveFits) -> None:
    # Linear slopes put the BMD of a change of 0.1 at chosen doses.
    lowest, top = 0.01, 30.0
    floor, ceiling = lowest / 100, 50 * top
    cases = {
        "within the doses": 1.0,
        "below the lowest dose": lowest / 2,
        "below a hundredth of the lowest dose": floor / 2,
        "above the top dose": top * 2,
        "not reached": ceiling * 2,
    }
    for status, bmd in cases.items():
        fits = with_parameters(epa_fits, "linear", {"g": 5.0, "b1": 0.1 / bmd})
        result = benchmark_doses(
            fits, BenchmarkResponse.absolute(0.1), model="linear", bounds=False
        )
        assert result.status[0, 0] == status
        expected = {
            "below a hundredth of the lowest dose": floor,
            "not reached": math.nan,
        }
        got = float(result.bmd[0, 0])
        want = expected.get(status, bmd)
        assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(
            want, rel=1e-12
        )


def test_status_edges() -> None:
    # A crossing exactly at the search's lowest dose is found there; one below it is
    # given as that dose. Doses written so that the changes are exact.
    from cytepool.stats import _benchmark_dose_search as search
    from cytepool.stats._dose_curve_models import LINE

    line = np.array([5.0, 1.0])  # g + b1 x
    at_floor = search.first_crossing(LINE, line, 0.5, math.nan, 0.5, 100.0)
    assert at_floor == (0.5, 1.0, search.CROSSED)
    below = search.first_crossing(LINE, line, 0.25, math.nan, 0.5, 100.0)
    assert below == (0.5, 1.0, search.CROSSED_BELOW_RANGE)
    at_ceiling = search.first_crossing(LINE, line, 100.0, math.nan, 0.5, 100.0)
    assert at_ceiling == (100.0, 1.0, search.CROSSED)
    beyond = search.first_crossing(LINE, line, 100.5, math.nan, 0.5, 100.0)
    assert beyond[2] == search.NEVER_CROSSED
    # Where each BMD lies against the doses 0.01 to 30, edges included.
    from cytepool.stats.potency import _statuses
    from cytepool.stats._dose_curve_models import DoseScale

    scale = DoseScale.of(HALF_LOG)
    bmd = np.array([0.01, 30.0, 0.0001, 30.000000000000004, 0.009999999999999998])
    how = np.full(bmd.size, search.CROSSED)

    chosen = np.zeros(bmd.size, dtype=np.int64)
    got = _statuses(chosen, np.ones(bmd.size, bool), how, bmd, scale)
    assert got.tolist() == [
        "within the doses",
        "within the doses",
        "below the lowest dose",
        "above the top dose",
        "below the lowest dose",
    ]


@pytest.fixture(scope="module")
def line_data() -> tuple[np.ndarray, np.ndarray]:
    x = replicated(HALF_LOG, 3)
    values = noisy(5.0 + 0.03 * x, 0.25, 3, n_genes=7)
    values[:, 3:6] = noisy(5.0 - 0.02 * x, 0.25, 4, n_genes=3)
    # A weak slope, whose likelihood region reaches 50 times the top dose.
    values[:, 6:] = noisy(5.0 + 0.006 * x, 0.25, 5, n_genes=1)
    return x, values


def test_bounds_of_a_straight_line(line_data: tuple[np.ndarray, np.ndarray]) -> None:
    # Constant variance, absolute R: the profile at dose b fixes the slope at s R / b,
    # adding Sxx (b1 - s R / b)^2 to the residual sum of squares RSS of the samples,
    # so the deviance n log(1 + Sxx (b1 - s R / b)^2 / RSS) reaches chi-square(1) at
    # 0.90 where |b1| - R / b = -+ sqrt((exp(2.7055 / n) - 1) RSS / Sxx).
    x, values = line_data
    fits = fit_dose_curves(x, values, models="epa")
    change = 0.3
    result = benchmark_doses(fits, BenchmarkResponse.absolute(change), model="linear")
    n = x.size
    sxx = float(np.sum((x - x.mean()) ** 2))
    kappa = math.expm1(CHI_SQUARE_90 / n)
    for g in range(values.shape[1]):
        y = values[:, g]
        b1 = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
        rss = float(np.sum((y - y.mean() - b1 * (x - x.mean())) ** 2))
        delta = math.sqrt(kappa * rss / sxx)
        assert float(result.bmd[0, g]) == pytest.approx(change / abs(b1), rel=1e-12)
        assert float(result.bmdl[0, g]) == pytest.approx(
            change / (abs(b1) + delta), rel=1e-9
        )
        upper = change / (abs(b1) - delta) if abs(b1) > delta else math.inf
        upper = upper if upper <= 50 * 30.0 else math.inf
        assert float(result.bmdu[0, g]) == pytest.approx(upper, rel=1e-9)
        assert result.direction[0, g] == ("up" if b1 > 0 else "down")
    assert np.all(result.threshold[0] == pytest.approx(CHI_SQUARE_90, rel=1e-14))
    assert math.isinf(result.bmdu[0, 6]) and np.isfinite(result.bmdu[0, :6]).all()


def test_one_sd_is_the_curve_maximum_likelihood_sd(
    line_data: tuple[np.ndarray, np.ndarray],
) -> None:
    # The BMR's size at 1 SD is sigma = sqrt(RSS / n), the n of BMDS's estimate.
    x, values = line_data
    fits = fit_dose_curves(x, values, models="epa")
    result = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="linear")
    sxx = float(np.sum((x - x.mean()) ** 2))
    for g in range(values.shape[1]):
        y = values[:, g]
        b1 = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
        rss = float(np.sum((y - y.mean() - b1 * (x - x.mean())) ** 2))
        sigma = math.sqrt(rss / x.size)
        assert float(result.bmr_change[0, g]) == pytest.approx(1.349 * sigma, rel=1e-12)
        assert float(result.bmr_change[1, g]) == pytest.approx(sigma, rel=1e-12)
        assert float(result.bmd[0, g]) == pytest.approx(
            1.349 * sigma / abs(b1), rel=1e-12
        )


def test_the_one_sd_comparison_is_always_given(epa_fits: DoseCurveFits) -> None:
    def labels(bmr) -> list[str]:
        return [
            b.label
            for b in benchmark_doses(epa_fits, bmr, bounds=False).benchmark_responses
        ]

    assert labels(BenchmarkResponse.sd(1.349)) == ["1.349 SD", "1 SD"]
    assert labels(BenchmarkResponse.absolute(0.5)) == ["change of 0.5", "1 SD"]
    assert labels(BenchmarkResponse.sd(1.0)) == ["1 SD"]
    assert labels([BenchmarkResponse.sd(1.0), BenchmarkResponse.absolute(0.5)]) == [
        "1 SD",
        "change of 0.5",
    ]
    assert labels([BenchmarkResponse.absolute(0.5), BenchmarkResponse.sd(2)]) == [
        "change of 0.5",
        "2 SD",
        "1 SD",
    ]
    # A multiple that prints as 1 SD is the comparison, which is not added again.
    assert labels(BenchmarkResponse.sd(1.0000001)) == ["1 SD"]


def test_arguments_are_refused_with_a_fix(epa_fits: DoseCurveFits) -> None:
    bmr = BenchmarkResponse.sd(1.0)
    with pytest.raises(TypeError, match="fits must be a DoseCurveFits"):
        benchmark_doses("fits", bmr)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"BenchmarkResponse\.sd\(1\.349\)"):
        benchmark_doses(epa_fits, 1.349)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bmr must hold BenchmarkResponse"):
        benchmark_doses(epa_fits, [bmr, 0.5])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        benchmark_doses(epa_fits, [])
    with pytest.raises(ValueError, match="bmr repeats '1 SD'"):
        benchmark_doses(epa_fits, [bmr, BenchmarkResponse.sd(1)])
    # Values that print alike are one BMR to a table, and refused together.
    with pytest.raises(ValueError, match="bmr repeats 'change of 0.584963'"):
        benchmark_doses(
            epa_fits,
            [
                BenchmarkResponse.absolute(math.log2(1.5)),
                BenchmarkResponse.absolute(0.584963),
            ],
        )
    with pytest.raises(ValueError, match="did you mean 'hill'"):
        benchmark_doses(epa_fits, bmr, model="hil")
    with pytest.raises(TypeError, match="model must be a str"):
        benchmark_doses(epa_fits, bmr, model=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bounds must be True or False"):
        benchmark_doses(epa_fits, bmr, bounds=1)  # type: ignore[arg-type]
    result = benchmark_doses(epa_fits, bmr, bounds=False)
    with pytest.raises(ValueError, match="unknown set of checks 'etap'.*etap_2024"):
        result.table(flags="etap")
    with pytest.raises(TypeError, match="flags must name a set of checks"):
        result.table(flags=3)
    with pytest.raises(TypeError, match="as text, got int"):
        result.table(flags=["ntp_2018", 3])


def test_a_model_not_fitted_is_refused() -> None:
    # Four dose groups leave Hill without a spare group.
    x = replicated(np.array([0.0, 0.1, 1.0, 10.0]), 3)
    fits = fit_dose_curves(x, noisy(5.0 + 0.05 * x, 0.2, 5), models="epa")
    with pytest.raises(
        ValueError, match="model 'hill' was not fitted: no spare dose group"
    ):
        benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="hill")


def test_bounds_need_replicates() -> None:
    x = np.array([0.0, 0.0, 0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    values = noisy(5.0 + x / (1 + x), 0.2, 6, n_genes=2)
    fits = fit_dose_curves(x, values, models="epa")
    # The vehicle's replicates give the variance degrees of freedom here.
    assert np.all(fits.df > 0)
    single = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    fits = fit_dose_curves(
        single, noisy(5.0 + single / (1 + single), 0.2, 7, 2), models="epa"
    )
    with pytest.raises(ValueError, match="pass bounds=False"):
        benchmark_doses(fits, BenchmarkResponse.absolute(0.5))
    result = benchmark_doses(fits, BenchmarkResponse.absolute(0.5), bounds=False)
    assert np.all(np.isnan(result.bmdl)) and np.all(np.isnan(result.bmdu))
    assert np.isfinite(result.bmd).any()
    assert not result.bounds and "no bounds" in repr(result)


def test_the_table_reads_back(epa_fits: DoseCurveFits) -> None:
    result = benchmark_doses(
        epa_fits, [BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.5)]
    )
    table = result.table(feature_ids=["gene_a"], flags=["etap_2024", "ntp_2018"])
    assert list(table.columns) == [
        "feature_id",
        "bmr",
        "model",
        "direction",
        "bmr_change",
        "bmd",
        "bmdl",
        "bmdu",
        "bmd_status",
        "flags_etap_2024",
        "flags_ntp_2018",
    ]
    assert table["bmr"].tolist() == ["1.349 SD", "change of 0.5", "1 SD"]
    assert table["feature_id"].tolist() == ["gene_a"] * 3
    np.testing.assert_array_equal(table["bmd"], result.bmd.ravel())
    np.testing.assert_array_equal(table["bmdl"], result.bmdl.ravel())
    assert table["model"].tolist() == [epa_fits.models[epa_fits.best[0]]] * 3
    plain = result.table()
    assert list(plain.columns)[-1] == "bmd_status"
    text = repr(result)
    assert text.splitlines()[0].startswith(
        "BenchmarkDoses (epa, best model, constant variance): 1 gene"
    )
    assert "1.349 SD:" in text and "change of 0.5:" in text
    with pytest.raises(ValueError, match="more than once"):
        result.table(flags=["etap_2024", "etap_2024"])
    for array in (result.bmd, result.status, result.models):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = array


@pytest.fixture(scope="module")
def mixed() -> tuple[np.ndarray, np.ndarray]:
    """Six genes of different shapes at the vehicle and 8 half-log doses x 3."""
    x = replicated(HALF_LOG, 3)
    pos = np.maximum(x, 1e-300)
    means = np.column_stack(
        [
            5.0 + 1.2 * x**2 / (0.5**2 + x**2),
            6.0 - 0.9 * (1.0 - np.exp(-x / 2.0)),
            5.0 + 0.03 * x,
            7.0 + 0.0 * x,
            5.0 + 0.8 * np.exp(-0.5 * np.log(pos / 0.3) ** 2) * (x > 0),
            5.0 - 1.0 * x**3 / (5.0**3 + x**3),
        ]
    )
    rng = np.random.default_rng(11)
    return x, means + rng.normal(0.0, 0.25, means.shape)


def results_close(a: BenchmarkDoses, b: BenchmarkDoses, *, scale: float = 1.0) -> None:
    for name in ("bmd", "bmdl", "bmdu"):
        np.testing.assert_allclose(
            getattr(a, name) * scale,
            getattr(b, name),
            rtol=1e-8,
            atol=0,
            equal_nan=True,
        )
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_array_equal(a.direction, b.direction)


def test_doses_in_other_units(mixed: tuple[np.ndarray, np.ndarray]) -> None:
    x, values = mixed
    bmr = [BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.4)]
    base = benchmark_doses(fit_dose_curves(x, values), bmr)
    scaled = benchmark_doses(fit_dose_curves(x * 1000.0, values), bmr)
    results_close(base, scaled, scale=1000.0)


def test_order_and_threads(mixed: tuple[np.ndarray, np.ndarray]) -> None:
    x, values = mixed
    bmr = BenchmarkResponse.sd(1.0)
    fits = fit_dose_curves(x, values)
    one = benchmark_doses(fits, bmr, n_threads=1)
    four = benchmark_doses(fits, bmr, n_threads=4)
    for name in ("bmd", "bmdl", "bmdu", "bmr_change"):
        np.testing.assert_array_equal(getattr(one, name), getattr(four, name))
    order = np.array([3, 0, 5, 1, 4, 2])
    samples = np.random.default_rng(1).permutation(x.size)
    shuffled = benchmark_doses(
        fit_dose_curves(x[samples], values[samples][:, order]), bmr
    )
    for name in ("bmd", "bmdl", "bmdu"):
        np.testing.assert_allclose(
            getattr(one, name)[:, order],
            getattr(shuffled, name),
            rtol=1e-8,
            equal_nan=True,
        )


def test_values_scaled_or_shifted(mixed: tuple[np.ndarray, np.ndarray]) -> None:
    # Doubling the values leaves SD-based BMDs unchanged and turns a change R into
    # 2R; adding 3 leaves the additive models' results unchanged.
    x, values = mixed
    fits = fit_dose_curves(x, values, models="epa")
    sd = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="hill")
    absolute = benchmark_doses(fits, BenchmarkResponse.absolute(0.4), model="hill")
    doubled = fit_dose_curves(x, 2.0 * values, models="epa")
    results_close(
        sd, benchmark_doses(doubled, BenchmarkResponse.sd(1.349), model="hill")
    )
    results_close(
        absolute,
        benchmark_doses(doubled, BenchmarkResponse.absolute(0.8), model="hill"),
    )
    shifted = fit_dose_curves(x, values + 3.0, models="epa")
    for model in ("linear", "hill", "power"):
        results_close(
            benchmark_doses(fits, BenchmarkResponse.sd(1.0), model=model),
            benchmark_doses(shifted, BenchmarkResponse.sd(1.0), model=model),
        )


def test_bounds_hold_the_bmd(mixed: tuple[np.ndarray, np.ndarray]) -> None:
    x, values = mixed
    for model_set in ("epa", "dromics", "ll4"):
        result = benchmark_doses(
            fit_dose_curves(x, values, models=model_set),
            [BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.3)],
        )
        found = np.isfinite(result.bmd) & np.isfinite(result.bmdl)
        assert found.any()
        assert np.all(result.bmdl[found] <= result.bmd[found])
        assert np.all(result.bmd[found] <= result.bmdu[found])
        assert np.all((result.bmdl[found] > 0) | (result.bmdl[found] == 0))


def test_what_bmds_checks_read(mixed: tuple[np.ndarray, np.ndarray]) -> None:
    # Test 2: n log(S_w / n) - sum_k n_k log(S_k / n_k) on K - 1 degrees of freedom;
    # the curve's SD over the vehicle's sample SD; scaled residuals with the curve's
    # maximum-likelihood SD.
    from scipy.stats import chi2

    x, values = mixed
    fits = fit_dose_curves(x, values, models="epa")
    result = benchmark_doses(fits, BenchmarkResponse.sd(1.0))
    levels = np.unique(x)
    for g in range(values.shape[1]):
        y = values[:, g]
        groups = [y[x == level] for level in levels]
        sums = np.array([np.sum((v - v.mean()) ** 2) for v in groups])
        n = x.size
        statistic = n * math.log(sums.sum() / n) - sum(
            v.size * math.log(s / v.size) for v, s in zip(groups, sums)
        )
        p = chi2.sf(statistic, levels.size - 1)
        assert float(result.constant_variance_p_value[g]) == pytest.approx(p, rel=1e-10)
        model = fits.models[fits.best[g]]
        mean_at = fits.means[g] - documented_residuals(fits, g, model)
        rss = sums.sum() + float(np.sum(3.0 * (fits.means[g] - mean_at) ** 2))
        sigma = math.sqrt(rss / n)
        vehicle_sd = float(np.std(groups[0], ddof=1))
        assert float(result.vehicle_sd_ratio[g]) == pytest.approx(
            sigma / vehicle_sd, rel=1e-9
        )
        scaled = (fits.means[g] - mean_at) / (sigma / math.sqrt(3.0))
        assert float(result.scaled_residual_at_vehicle[g]) == pytest.approx(
            scaled[0], rel=1e-8, abs=1e-12
        )
        bmd = float(result.bmd[0, g])
        if math.isfinite(bmd):
            # the dose group nearest the BMD on the dose scale, the lower on a tie
            nearest = int(np.argmin(np.abs(levels - bmd)))
            assert float(result.scaled_residual_near_bmd[0, g]) == pytest.approx(
                scaled[nearest], rel=1e-8, abs=1e-12
            )


def documented_residuals(fits: DoseCurveFits, g: int, model: str) -> np.ndarray:
    """ybar_k - f(x_k) of a gene's model from the curves' table of fitted means."""
    from cytepool.stats import _dose_curve_models as models_

    names = fits.parameter_names
    spec = next(m for m in models_.SETS[fits.model_set] if m.name == model)
    m = fits.models.index(model)
    theta = models_.internal(
        spec.kind,
        np.array([fits.parameters[g, m, names.index(p)] for p in spec.parameters]),
    )
    return fits.means[g] - models_.mean_values(spec.kind, theta[None], fits.doses)[0]


def test_the_residual_of_interest_takes_the_lower_dose_on_a_tie(
    epa_fits: DoseCurveFits,
) -> None:
    # A straight line whose BMD of 0.1 falls halfway between 1 and 3.
    fits = with_parameters(epa_fits, "linear", {"g": 5.0, "b1": 0.05})
    result = benchmark_doses(
        fits, BenchmarkResponse.absolute(0.1), model="linear", bounds=False
    )
    assert float(result.bmd[0, 0]) == pytest.approx(2.0)
    from cytepool.stats._dose_curve_models import LINE  # noqa: F401

    residuals = fits.means[0] - (5.0 + 0.05 * fits.doses)
    sigma_part = residuals / residuals  # the same scale on both doses
    assert sigma_part[5] == 1.0
    expected_dose = 5  # dose 1.0, not 3.0
    # scaled residuals share the denominator sigma / sqrt(3) at every dose here
    ratio = float(result.scaled_residual_near_bmd[0, 0]) / residuals[expected_dose]
    other = float(result.scaled_residual_near_bmd[0, 0]) / residuals[expected_dose + 1]
    assert (
        ratio > 0
        and abs(
            ratio * residuals[expected_dose + 1] - other * residuals[expected_dose + 1]
        )
        > 0
    )


def test_curves_without_a_vehicle() -> None:
    # Doses from 0.1 up, no group at 0: the change is from the curve's level at 0.
    x = replicated(np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0]), 3)
    values = noisy(5.0 + 0.04 * x, 0.2, 9, n_genes=2)
    fits = fit_dose_curves(x, values, models="epa")
    result = benchmark_doses(fits, BenchmarkResponse.absolute(0.5), model="linear")
    slope = [np.polyfit(x, values[:, g], 1)[0] for g in range(2)]
    np.testing.assert_allclose(result.bmd[0], 0.5 / np.abs(slope), rtol=1e-9)
    assert np.all(np.isnan(result.scaled_residual_at_vehicle))
    assert np.all(np.isnan(result.vehicle_sd_ratio))


@pytest.mark.parametrize(
    ("means", "model"),
    [
        # A step between the vehicle and the lowest dose: Hill's k at 0 or n at 18.
        (lambda x: 5.0 + 1.0 * (x > 0), "hill"),
        (lambda x: 5.0 + x**30 / (0.5**30 + x**30), "hill"),
        # A fall that would cross 0 beyond the doses puts exponential 4's c at 0, and
        # a rise of 2,000-fold at its bound of 1,000.
        (lambda x: 5.0 - 0.133 * x, "exponential_4"),
        (lambda x: 0.5 * (2000.0 - 1999.0 * np.exp(-x / 2.0)), "exponential_4"),
    ],
)
def test_bounds_with_parameters_at_their_bounds(means, model: str) -> None:
    x = replicated(HALF_LOG, 3)
    values = noisy(means(x), 0.2, 12, n_genes=1)
    fits = fit_dose_curves(x, values, models="epa")
    m = fits.models.index(model)
    assert fits.parameters_at_bounds[0, m] != ""
    result = benchmark_doses(fits, BenchmarkResponse.absolute(0.3), model=model)
    bmd, low, high = (
        float(result.bmd[0, 0]),
        float(result.bmdl[0, 0]),
        float(result.bmdu[0, 0]),
    )
    assert math.isfinite(bmd)
    assert 0.0 <= low <= bmd <= high


def test_a_vehicle_without_spread(epa_fits: DoseCurveFits) -> None:
    # Vehicle replicates that agree exactly: the ratio reads their SD as 1e-6, as
    # BMDS does, so the check of the curve's SD fails.
    x = replicated(HALF_LOG, 3)
    values = noisy(5.0 + x / (1.0 + x), 0.2, 13)
    values[x == 0.0, 0] = 5.0
    fits = fit_dose_curves(x, values, models="epa")
    result = benchmark_doses(fits, BenchmarkResponse.sd(1.0))
    n = x.size
    model = fits.models[fits.best[0]]
    residual = documented_residuals(fits, 0, model)
    within = sum(
        float(np.sum((values[x == d, 0] - values[x == d, 0].mean()) ** 2))
        for d in HALF_LOG
    )
    sigma = math.sqrt((within + 3.0 * float(np.sum(residual**2))) / n)
    assert float(result.vehicle_sd_ratio[0]) == pytest.approx(sigma / 1e-6, rel=1e-9)
    assert (
        "curve's vehicle SD / observed vehicle SD > 1.5 (warning)"
        in result.table(flags="bmds_2026")["flags_bmds_2026"][0]
    )
    # Test 2 needs spread in every group.
    assert math.isnan(float(result.constant_variance_p_value[0]))


def profile_minus_fit_at_the_bmd(
    fits: DoseCurveFits, bmr: BenchmarkResponse, model: str
) -> np.ndarray:
    """The profile objective at each gene's BMD, maximised at the fit's own nonlinear
    parameters, minus the objective at the fit."""
    from cytepool.stats import potency
    from cytepool.stats._benchmark_dose_search import profile_point, workspace
    from cytepool.stats._dose_curve_models import constraint, is_linear, n_linear

    result = benchmark_doses(fits, bmr, model=model, bounds=False)
    genes = potency._GeneFits.of(fits, potency._chosen_models(fits, model))
    likelihood = potency._likelihood(fits, genes, bmr)
    found = (result.status[0] != "below a hundredth of the lowest dose") & np.isfinite(
        result.bmd[0]
    )
    out = []
    for g in np.flatnonzero(found):
        kind, theta = int(genes.kinds[g]), genes.theta[g]
        q = n_linear(kind)
        index = [
            j for j in range(theta.size) if theta.size > q and not is_linear(kind, j)
        ]
        phi = np.ascontiguousarray(theta[index] if index else np.zeros(1))
        root = np.ascontiguousarray(genes.roots[0 if genes.roots.shape[0] == 1 else g])
        value = profile_point(
            kind,
            constraint(kind),
            q,
            phi,
            float(result.bmd[0, g]),
            fits.doses,
            root,
            root @ fits.means[g],
            int(likelihood.forms[g]),
            float(likelihood.values[g]),
            1.0 if result.direction[0, g] == "up" else -1.0,
            float(likelihood.a0[g]),
            float(likelihood.n_eff[g]),
            float(likelihood.s2[g]),
            np.zeros(theta.size),
            np.zeros(3),
            workspace(fits.doses.size),
        )
        scale = max(1.0, abs(float(likelihood.objective_fit[g])))
        out.append((value - float(likelihood.objective_fit[g])) / scale)
    return np.array(out)


@pytest.mark.parametrize("model_set", ["epa", "dromics", "ll4"])
def test_the_profile_at_the_bmd_is_the_fit(
    mixed: tuple[np.ndarray, np.ndarray], model_set: str
) -> None:
    # The fit meets the constraint at its own BMD, so the profile there is the fit:
    # a wrong sign, level at dose 0 or SD in the constraint moves it.
    x, values = mixed
    fits = fit_dose_curves(x, values, models=model_set)
    checked = 0
    for model in fits.models:
        for bmr in (BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.3)):
            gaps = profile_minus_fit_at_the_bmd(fits, bmr, model)
            assert np.all(np.abs(gaps) <= 1e-9), (model, bmr.label, gaps)
            checked += gaps.size
    assert checked >= 8 * len(fits.models)


@pytest.mark.parametrize("variance", ["weighted", "moderated"])
def test_the_plug_in_profile_at_the_bmd_is_the_fit(variance: str) -> None:
    from cytepool import Design, DoseAxis, dose_response, simulate_dose_series

    cube, _ = simulate_dose_series(
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], replicates=2, cell_types=1, n_genes=60, rng=3
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=DoseAxis("dose", unit="ug/kg", metric="administered"),
        models="epa",
        variance=variance,
    )
    fits = result.curve_fits(result.groups[0])
    checked = 0
    for model in ("linear", "hill", "exponential_4", "power"):
        for bmr in (BenchmarkResponse.sd(1.0), BenchmarkResponse.absolute(0.3)):
            gaps = profile_minus_fit_at_the_bmd(fits, bmr, model)
            assert np.all(np.abs(gaps) <= 1e-9), (model, bmr.label, gaps)
            checked += gaps.size
    assert checked > 20


# Fixtures of validation/benchmark_doses.py --write-fixtures.


def fixture(name: str) -> dict:
    found: dict = json.loads((DATA / f"{name}.json").read_text())
    return found


def refitted(data: dict) -> DoseCurveFits:
    """The curves of a fixture case, fitted again as the validation fitted them: to
    the gene's samples under constant variance, or to its dose-group means with their
    covariance, variance and degrees of freedom."""
    from cytepool.stats.dose_curves import _CurveData, _fit_means

    if data["variance"] == "constant":
        return fit_dose_curves(
            np.asarray(data["doses"]),
            np.asarray(data["values"])[:, None],
            models=data["model_set"],
        )
    curve_data = _CurveData(
        data["variance"],
        np.asarray(data["doses"]),
        np.asarray(data["n_samples"]),
        np.asarray(data["means"])[None, :],
        np.asarray(data["covariance"])[None, :, :],
        np.array([data["covariance_scale"]]),
        np.array([data["df"]]),
        None,
    )
    return _fit_means(curve_data, data["model_set"])


def fixture_bmr(task: dict) -> BenchmarkResponse:
    if task["form"] == "sd":
        # The plug-in modes' inputs carry the multiple scaled to a vehicle replicate.
        key = task["key"].rsplit("/", 1)[1]
        return BenchmarkResponse.sd(float(key.removesuffix(" SD")))
    return BenchmarkResponse.absolute(task["value"])


def test_bounds_match_the_scipy_profile() -> None:
    profiles = fixture("profiles")
    results = {r["key"]: r for r in profiles["results"]}
    checked = 0
    for task in profiles["inputs"]:
        reference = results[task["key"]]
        fits = refitted(task["data"])
        found = benchmark_doses(fits, fixture_bmr(task), model=task["model"])
        for name in ("bmdl", "bmdu"):
            ours = float(getattr(found, name)[0, 0])
            if reference.get(f"{name}_at_ours") is not None:
                # scipy's root of deviance = threshold near the bound, maximising the
                # documented likelihood from its own starts and from the curve
                # CytePool found at the bound, which the documented means checked;
                # scipy's own starts found no curve below the threshold there.
                assert ours == pytest.approx(reference[name], rel=1e-6), (
                    task["key"],
                    name,
                )
                assert reference[f"{name}_at_ours"] >= -1e-6
                witness = reference[f"{name}_witness"]
                assert witness["inside"] and witness["constraint"] <= 1e-8
                assert abs(witness["deviance"]) <= 1e-6
                checked += 1
            if reference.get(f"{name}_end") is not None:
                # An open end: scipy's deviance at the search's end is within the
                # threshold, so the region reaches it.
                assert ours in (0.0, math.inf)
                assert reference[f"{name}_end"] <= 1e-6
            if reference.get(f"{name}_beyond") is not None:
                assert reference[f"{name}_beyond"] > 0.0
    assert checked > 100


def test_bounds_of_straight_lines_match_mpmath() -> None:
    lines = fixture("linear")
    results = {r["key"]: r for r in lines["results"]}
    for task in lines["inputs"]:
        reference = results[task["key"]]
        found = benchmark_doses(
            refitted(task["data"]), fixture_bmr(task), model="linear"
        )
        assert float(found.bmd[0, 0]) == pytest.approx(reference["bmd"], rel=1e-10)
        for name in ("bmdl", "bmdu"):
            ours, theirs = float(getattr(found, name)[0, 0]), reference[name]
            if theirs in (0.0, math.inf):
                assert ours == theirs, (task["key"], name)
            else:
                assert ours == pytest.approx(theirs, rel=1e-10), (task["key"], name)


def test_dromics_benchmark_doses_given_its_parameters() -> None:
    dromics = fixture("dromics")
    doses = np.asarray(dromics["doses"])
    rng = np.random.default_rng(0)
    fits = fit_dose_curves(
        doses, 5.0 + rng.normal(0.0, 0.2, (doses.size, 1)), models="dromics"
    )
    floor = float(dromics["minBMD"])
    for row in dromics["rows"]:
        letters = dict(zip("bcdef", row["parameters"], strict=True))
        parameters = {k: v for k, v in letters.items() if v is not None}
        one = with_parameters(fits, row["model"], parameters)
        found = benchmark_doses(
            one,
            BenchmarkResponse.absolute(row["size"]),
            model=row["model"],
            bounds=False,
        )
        bmd, status = float(found.bmd[0, 0]), str(found.status[0, 0])
        if row["code"] == "NaN":
            assert status == "not reached"
        elif row["code"] == "NA":
            assert status in ("above the top dose", "not reached")
        elif row["bmd"] == floor:
            assert status == "below a hundredth of the lowest dose" and bmd == floor
        elif row["model"].startswith(("gauss", "log_gauss")) and not row[
            "model"
        ].endswith("monotone"):
            # bmdcalc searches bells with uniroot, within 1.22e-4 on its log dose.
            assert abs(math.log(bmd) - math.log(row["bmd"])) <= 1.22e-4
        else:
            assert bmd == pytest.approx(row["bmd"], rel=1e-10)


def test_benchmark_doses_given_pybmds_parameters() -> None:
    from cytepool.stats._dose_curve_models import model_of

    found_any = 0
    pybmds = fixture("pybmds")
    for task, result in zip(pybmds["tasks"], pybmds["results"], strict=True):
        doses = np.asarray(task["doses"])
        fits = fit_dose_curves(doses, np.asarray(task["values"])[:, None], models="epa")
        top = float(doses.max())
        for row in result["rows"]:
            failed = any(
                "Did not successfully execute." in messages
                for messages in row["notes"].values()
            )
            if failed or row["bmd"] in (None, -9999.0) or row["bmd"] > 50 * top:
                # A fit pybmds did not complete, as its notes say, or no BMD.
                continue
            values = dict(zip(row["names"], row["values"], strict=True))
            parameters = {
                name: values[name] for name in model_of("epa", row["model"]).parameters
            }
            if row["model"] == "exponential_3" and not row["is_increasing"]:
                parameters["b"] = -parameters["b"]
            if task["form"] == "sd":
                alpha = values.get("alpha", math.exp(values.get("log-alpha", math.nan)))
                size = math.sqrt(alpha)
            else:
                size = task["value"]
            ours = benchmark_doses(
                with_parameters(fits, row["model"], parameters),
                BenchmarkResponse.absolute(size),
                model=row["model"],
                bounds=False,
            )
            # pybmds's BMD search stops within about 1e-6 of the root.
            assert float(ours.bmd[0, 0]) == pytest.approx(row["bmd"], rel=1e-5)
            found_any += 1
    assert found_any > 20


def test_flags_match_the_sets_coded_in_r() -> None:
    from cytepool.stats._benchmark_dose_flags import (
        FLAG_SETS,
        FlagInputs,
        failed_checks,
    )

    flags = fixture("flags")
    for table, reference in zip(flags["tables"], flags["r"], strict=True):

        def column(name: str, table: dict = table) -> np.ndarray:
            return np.array(
                [
                    np.nan if v is None else (np.inf if v == "inf" else v)
                    for v in table["rows"][name]
                ],
                dtype=float,
            )

        inputs = FlagInputs(
            bmd=column("bmd"),
            bmdl=column("bmdl"),
            bmdu=column("bmdu"),
            bounds=table["bounds"],
            lowest_dose=table["lowest_dose"],
            top_dose=table["top_dose"],
            lack_of_fit_p_value=column("lack_of_fit_p_value"),
            residual_near_bmd=column("residual_near_bmd"),
            residual_at_vehicle=column("residual_at_vehicle"),
            vehicle_sd_ratio=column("vehicle_sd_ratio"),
            constant_variance_p_value=column("constant_variance_p_value"),
        )
        for flag_set in FLAG_SETS:
            assert failed_checks(flag_set, inputs) == reference[flag_set], flag_set


def test_the_chapter_example_runs() -> None:
    """The chapter's example, on the series of the dose chapter, and the numbers its
    text states."""
    text = CHAPTER.read_text()
    (block,) = re.findall(r"```python\n(.*?)```", text, re.S)
    namespace: dict = {}
    exec(block, namespace)  # noqa: S102
    result = namespace["result"]
    line = (
        "BMDs at 1.349 SD: 149 within the doses, 4 below the lowest dose, 2 below a "
        "hundredth of the lowest dose, 598 above the top dose, 64 not reached"
    )
    assert line in " ".join(repr(result).split())
    assert f"`{line}`" in text
    assert "for the hepatocytes' 817 genes" in text
    # The genes that pass the gate, by the gate's own table, and those of them that
    # pass ETAP's checks at 1.349 SD.
    gate = result.table("hepatocyte", "gate")
    passed = set(gate.filter(gate["passes_gate"])["feature_id"].tolist())
    bmds = namespace["bmds"]
    rows = bmds.filter(
        np.isin(bmds["feature_id"], list(passed)) & (bmds["bmr"] == "1.349 SD")
    )
    assert len(rows) == len(passed)
    assert rows["passes_gate"].all()
    passing = int(np.sum(np.asarray(rows["flags_etap_2024"]) == ""))
    assert f"{passing} of the {len(rows)} genes that pass the gate pass" in text
    medians = [
        f"{np.nanmedian(result.benchmark_doses(group).bmr_change[0]):.2f}"
        for group in ("hepatocyte", "stellate")
    ]
    assert (
        f"a median change of {medians[0]} log2 units in the hepatocytes" in text
        and f"of {medians[1]} in the stellate cells" in text
    )


def test_plug_in_curves_without_a_vehicle_have_no_sd_to_scale_to() -> None:
    """Means of a fit without a vehicle group: an SD-based BMR is undefined, since
    the SD of a vehicle replicate needs the vehicle's mean; an absolute BMR is not."""
    from cytepool.stats.dose_curves import _CurveData, _fit_means

    levels = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    n_samples = np.full(levels.size, 3.0)
    means = (5.0 + 0.04 * levels)[None, :] + np.array(
        [[0.02, -0.03, 0.01, 0.0, -0.02, 0.03]]
    )
    data = _CurveData(
        "weighted",
        levels,
        n_samples,
        means,
        np.diag(1.0 / n_samples)[None],
        np.array([0.04]),
        np.array([12.0]),
        None,
    )
    fits = _fit_means(data, "epa")
    found = benchmark_doses(
        fits,
        [BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.5)],
        model="linear",
    )
    assert found.status[0, 0] == "BMR undefined" and math.isnan(found.bmr_change[0, 0])
    assert math.isnan(found.bmd[0, 0]) and math.isnan(found.bmdl[0, 0])
    assert found.status[1, 0] == "within the doses"
    slope = fits.parameters[
        0, fits.models.index("linear"), fits.parameter_names.index("b1")
    ]
    assert float(found.bmd[1, 0]) == pytest.approx(0.5 / abs(slope), rel=1e-12)
    assert math.isnan(found.scaled_residual_at_vehicle[0])


def test_a_region_that_reaches_the_search_s_lowest_dose_gives_a_bmdl_of_0() -> None:
    # A Hill response whose EC50 lies below the lowest dose: a Hill curve that
    # reaches the BMR at a hundredth of the lowest dose, the fit's own curve with its
    # k moved there, fits within the threshold, so the likelihood's region reaches
    # the search's lowest dose and the BMDL is given as 0.
    x = replicated(HALF_LOG, 3)
    rng = np.random.default_rng(21)
    values = 5.0 + 0.77 * x / (0.0012 + x) + rng.normal(0.0, 0.3, x.size)
    fits = fit_dose_curves(x, values[:, None], models="epa")
    change = 0.3
    found = benchmark_doses(fits, BenchmarkResponse.absolute(change), model="hill")
    assert found.status[0, 0] == "below the lowest dose"
    assert float(found.bmdl[0, 0]) == 0.0
    names = fits.parameter_names
    m = fits.models.index("hill")
    g, v, k, n = (float(fits.parameters[0, m, names.index(p)]) for p in "gvkn")
    floor = 0.01 / 100
    rho = change / abs(v)
    k_floor = floor * ((1.0 - rho) / rho) ** (1.0 / n)

    def total_squares(location: float) -> float:
        curve = g + v * x**n / (location**n + x**n)
        return float(np.sum((values - curve) ** 2))

    deviance = x.size * math.log(total_squares(k_floor) / total_squares(k))
    assert deviance <= CHI_SQUARE_90


def test_bounds_are_not_computed_where_the_fit_is_not_the_maximum(
    line_data: tuple[np.ndarray, np.ndarray],
) -> None:
    # A line with its slope halved is not the curve of largest likelihood: the
    # profile at its BMD finds a better line, so its bounds are not computed.
    x, values = line_data
    fits = fit_dose_curves(x, values[:, :2], models="epa")
    m = fits.models.index("linear")
    halved = np.array(fits.parameters)
    halved[0, m, fits.parameter_names.index("b1")] *= 0.5
    found = benchmark_doses(
        dataclasses.replace(fits, parameters=halved),
        BenchmarkResponse.absolute(0.3),
        model="linear",
    )
    assert math.isfinite(found.bmd[0, 0]) and math.isnan(found.bmdl[0, 0])
    assert math.isnan(found.bmdu[0, 0])
    assert np.isfinite(found.bmdl[0, 1])
    assert "bounds not computed for 2 BMDs: the profile found a likelihood above" in (
        " ".join(repr(found).split())
    )


def test_genes_that_never_reach_the_bmr_get_no_bounds(epa_fits: DoseCurveFits) -> None:
    fits = with_parameters(epa_fits, "linear", {"g": 5.0, "b1": 1e-6})
    found = benchmark_doses(fits, BenchmarkResponse.absolute(0.5), model="linear")
    assert found.status[0, 0] == "not reached"
    assert np.isnan([found.bmd[0, 0], found.bmdl[0, 0], found.bmdu[0, 0]]).all()


def test_bounds_of_a_rising_dromics_exponential() -> None:
    # DRomics' exponential of a positive rate keeps its sign in the profile's search;
    # the bounds of a clear rise hold the BMD and the fit is the profile's maximum.
    x = replicated(HALF_LOG, 3)
    rng = np.random.default_rng(31)
    values = 5.0 + 0.3 * np.expm1(x / 8.0) + rng.normal(0.0, 0.2, x.size)
    fits = fit_dose_curves(x, values[:, None], models="dromics")
    m = fits.models.index("exponential")
    assert fits.parameters[0, m, fits.parameter_names.index("e")] > 0
    found = benchmark_doses(fits, BenchmarkResponse.absolute(0.5), model="exponential")
    low, bmd, high = (
        float(found.bmdl[0, 0]),
        float(found.bmd[0, 0]),
        float(found.bmdu[0, 0]),
    )
    assert 0.0 < low < bmd < high < math.inf
    assert (
        np.abs(
            profile_minus_fit_at_the_bmd(
                fits, BenchmarkResponse.absolute(0.5), "exponential"
            )
        ).max()
        <= 1e-9
    )


def test_a_crossing_beyond_a_turn_below_the_search_is_given_as_its_floor() -> None:
    # A Gauss-probit curve that rises a little, turns at once and falls by 2: beyond
    # the turn, the fall already exceeds the BMR at the search's lowest dose.
    from cytepool.stats import _benchmark_dose_search as search
    from cytepool.stats._dose_curve_models import GAUSS_PROBIT, internal, turn_doses

    theta = internal(GAUSS_PROBIT, np.array([[1e-6, 3.0, 5.0, 5e-5, 0.1]]))[0]
    turn = float(turn_doses(GAUSS_PROBIT, theta[None, :], 30.0)[0])
    assert 0.0 < turn < 1e-4
    found = search.first_crossing(GAUSS_PROBIT, theta, 0.5, turn, 1e-4, 1500.0)
    assert found == (1e-4, -1.0, search.CROSSED_BELOW_RANGE)


def test_the_constant_variance_test_matches_mpmath() -> None:
    """BMDS's test 2, the likelihood ratio of one variance against a variance per dose
    group at their maximum-likelihood estimates, against mpmath at 50 digits."""
    lines = fixture("linear")
    assert len(lines["test2"]) >= 10
    for task, reference in zip(lines["test2_inputs"], lines["test2"], strict=True):
        fits = fit_dose_curves(
            np.asarray(task["doses"]), np.asarray(task["values"])[:, None]
        )
        found = benchmark_doses(fits, BenchmarkResponse.absolute(0.5), bounds=False)
        assert float(found.constant_variance_p_value[0]) == pytest.approx(
            reference["p_value"], rel=1e-10
        )


def test_a_start_whose_curves_cannot_meet_the_bmr_is_left_out() -> None:
    # Exponential 2 at a rate of 0 is flat: no curve of that rate changes at any dose,
    # so the profile there has no feasible curve, its refinement stops at once with
    # an objective of infinity, and the optima kept for the next dose stay as they
    # were.
    from cytepool.stats import _benchmark_dose_search as search
    from cytepool.stats._dose_curve_models import EXP2, constraint

    x = np.array([0.0, 1.0, 10.0])
    ybar = np.array([5.0, 5.2, 6.0])
    root = np.diag(np.sqrt([3.0, 3.0, 3.0]))
    phi, theta = np.array([0.0]), np.zeros(2)
    objective = search.refine_profile(
        EXP2,
        constraint(EXP2),
        1,
        phi,
        np.array([-1.0]),
        np.array([1.0]),
        1.0,
        x,
        ybar,
        root,
        root @ ybar,
        search.CHANGE_PROFILED_VARIANCE,
        0.5,
        1.0,
        0.3,
        9.0,
        1.0,
        np.array([1], dtype=np.int64),
        theta,
        search.workspace(3),
        10,
    )
    assert objective == math.inf and phi[0] == 0.0
    kept, values = np.full((2, 1), 7.0), np.array([1.0, math.inf])
    assert search._keep_optimum(math.inf, phi, 1, kept, values, 1) == 1
    assert kept.tolist() == [[7.0], [7.0]] and values.tolist() == [1.0, math.inf]


def test_a_region_in_two_pieces_ends_at_the_far_piece() -> None:
    # A weak Hill response, found by simulation, whose likelihood region below the BMD
    # has two pieces: one around the BMD, and one from below a hundredth of the
    # lowest dose to about 0.03 ug/kg, where a step just above dose 0 fits almost as
    # well. The BMDL is the region's lowest dose, so it is 0, not the end of the
    # piece around the BMD.
    x = replicated(HALF_LOG, 3)
    y = np.array(
        [5.8463, 5.7155, 6.2424, 6.6158, 6.4698, 6.3317, 6.774, 6.882, 6.1706]
        + [6.3315, 6.8764, 6.1794, 6.4309, 6.9166, 6.3524, 6.663, 6.4653, 6.5553]
        + [6.5106, 5.9315, 6.5692, 7.2186, 6.923, 6.8938, 7.0501, 6.6016, 7.0952]
    )
    fits = fit_dose_curves(x, y[:, None], models="epa")
    change = 0.3
    found = benchmark_doses(fits, BenchmarkResponse.absolute(change), model="hill")
    assert found.direction[0, 0] == "up" and 3.0 < float(found.bmd[0, 0]) < 10.0
    assert float(found.bmdl[0, 0]) == 0.0
    # A Hill curve that changes by the BMR at a hundredth of the lowest dose, a step,
    # lies within the threshold of the fit: on a grid of k and n, with g and v by
    # least squares under the constraint, which fixes v, in closed form.
    m = fits.models.index("hill")
    names = fits.parameter_names
    g, v, k, n = (float(fits.parameters[0, m, names.index(p)]) for p in "gvkn")
    fit_rss = float(np.sum((y - g - v * x**n / (k**n + x**n)) ** 2))
    location = np.geomspace(1e-5, 1e-2, 60)[:, None, None]
    exponent = np.linspace(1.0, 18.0, 60)[None, :, None]
    at = 1e-4
    shape = x**exponent / (location**exponent + x**exponent)
    slope = change / (at**exponent / (location**exponent + at**exponent))
    level = np.mean(y - slope * shape, axis=-1, keepdims=True)
    rss = np.sum((y - level - slope * shape) ** 2, axis=-1)
    assert x.size * math.log(float(rss.min()) / fit_rss) <= CHI_SQUARE_90
