"""The checks that regulatory guidance makes of a benchmark dose, as named sets.

Each set lists its checks with the text a table prints when a check fails. A check
reads the BMD, its bounds and statistics of the fit, among them the fit p-value of
BMDS's test 4; a row that lacks what a check reads, such as the bounds of a cell type
that supports only description, does not get that check. The sets, their sources and
how each document uses a failed check are in
https://docs.nygen.io/CytePool/statistics/potency#benchmark-dose-flags.

- ``"bmds_2026"``: the default rules of BMDS's model recommendation for continuous
  models, as pybmds 25.2 runs them (US EPA 2026), with the category of each failure:
  unusable, questionable or warning.
- ``"ntp_2018"``: NTP's acceptance of a gene's BMD (NTP 2018, pp. 8 to 9).
- ``"etap_2024"``: EPA's acceptance of a gene's BMD in Transcriptomic Assessment
  Products (US EPA 2024, Sec. 3.4.5.3).
- ``"efsa_2022"``: EFSA's indicators for expert consideration (EFSA 2022, Sec. 2.6.5).

References:
    US EPA (2026). BMDS User Guide: model recommendations and decision logic. Online
    documentation of BMDS Online, BMDS Desktop and pybmds 25.2.
    https://usepa.github.io/bmds-user-guide/model-recommendation.html

    NTP (2018). NTP research report on National Toxicology Program approach to
    genomic dose-response modeling. NTP Research Report 5. National Institute of
    Environmental Health Sciences. doi:10.22427/ntp-rr-5

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496

    EFSA Scientific Committee (2022). Guidance on the use of the benchmark dose
    approach in risk assessment. EFSA Journal 20, e07584. doi:10.2903/j.efsa.2022.7584
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._masked_arrays import reject_masked
from cytepool._messages import close_matches, listing

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

FLAG_SETS = ("bmds_2026", "ntp_2018", "etap_2024", "efsa_2022")


def checked_flag_sets(flags: Any) -> tuple[str, ...]:
    """The sets of checks named by ``flags``: None, a name or a list of names.

    Raises:
        TypeError: If ``flags`` or a set is not text, or ``flags`` is a numpy masked
            array.
        ValueError: If a set is unknown or repeated.
    """
    if flags is None:
        return ()
    reject_masked(flags, "flags")
    if not isinstance(flags, str | list | tuple):
        raise TypeError(
            "flags must name a set of checks or list several, such as 'etap_2024' or "
            f"['ntp_2018', 'etap_2024'], got {type(flags).__name__}"
        )
    names = [flags] if isinstance(flags, str) else list(flags)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"flags must name sets of checks as text, got {type(name).__name__}"
            )
        if name not in FLAG_SETS:
            raise ValueError(
                f"unknown set of checks {name!r}; the sets are {listing(FLAG_SETS)}"
                f"{_suggestion(name)}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"flags names a set more than once: {names}")
    return tuple(names)


def _suggestion(name: str) -> str:
    """A hint for a mistyped set: the sets whose name starts with the given one,
    whatever its case, or else close spellings."""
    starting = [s for s in FLAG_SETS if s.startswith(name.lower())]
    if not starting:
        return close_matches(name, FLAG_SETS)
    return " (did you mean " + " or ".join(repr(s) for s in starting) + "?)"


@dataclass(frozen=True)
class FlagInputs:
    """What the checks read, one value per row: the BMD and its bounds (NaN where not
    computed; a BMDL of 0 where the likelihood's region reaches the lowest dose
    searched, a BMDU of infinity where it reaches the highest), whether bounds were
    computed, the lowest positive and the highest dose, the fit p-value (BMDS's test
    4), the scaled residuals of the dose group nearest the BMD and of the vehicle, the
    curve's SD of a vehicle replicate over the vehicle's sample SD, and the p-value
    of the test of constant variance across dose groups (NaN where not made)."""

    bmd: FloatArray
    bmdl: FloatArray
    bmdu: FloatArray
    bounds: bool
    lowest_dose: float
    top_dose: float
    fit_p_value: FloatArray
    residual_near_bmd: FloatArray
    residual_at_vehicle: FloatArray
    vehicle_sd_ratio: FloatArray
    constant_variance_p_value: FloatArray


@dataclass(frozen=True)
class _Check:
    """A check: the text a table prints when a row fails it, which rows fail it, and
    whether the document keeps a row that fails it, as NTP keeps a gene whose BMD
    lies more than 10-fold below the lowest dose and BMDS keeps a model with a
    warning among the viable ones."""

    text: str
    failed: Callable[[FlagInputs], BoolArray]
    kept: bool = False


def _above(values: FloatArray, threshold: float) -> BoolArray:
    """values > threshold where finite; False elsewhere."""
    with np.errstate(invalid="ignore"):
        return np.asarray(np.isfinite(values) & (values > threshold))


def _at_least(values: FloatArray, threshold: float) -> BoolArray:
    """values >= threshold, infinity included; False for NaN."""
    with np.errstate(invalid="ignore"):
        return np.asarray(~np.isnan(values) & (values >= threshold))


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """numerator / denominator where both are positive, infinity where only the
    numerator is (a bound of 0 or infinity widens the ratio), NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    defined = (numerator > 0.0) & (denominator >= 0.0) & ~np.isnan(ratio)
    return np.where(defined, ratio, np.nan)


def _bmdl_missing(inputs: FlagInputs) -> BoolArray:
    """No usable BMDL: not computed, or 0 because the region reaches the search's
    end (pybmds counts a BMDL of 0 as not estimated)."""
    return np.asarray(inputs.bounds & ~(np.isfinite(inputs.bmdl) & (inputs.bmdl > 0.0)))


def _positive_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """numerator / denominator where both are finite and above 0, NaN elsewhere, as
    pybmds computes its ratios."""
    ok = np.isfinite(numerator) & np.isfinite(denominator) & (numerator > 0.0)
    ok &= denominator > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, numerator / denominator, np.nan)


def _bmds(level: str, text: str, failed: Callable[[FlagInputs], BoolArray]) -> _Check:
    """A check of BMDS's recommendation; a model that fails only warnings stays
    viable."""
    return _Check(f"{text} ({level})", failed, kept=level == "warning")


def _bmds_checks() -> tuple[_Check, ...]:
    def lowest_over(values: FloatArray, inputs: FlagInputs) -> FloatArray:
        with np.errstate(divide="ignore"):
            return np.where(
                np.isfinite(values) & (values > 0.0),
                inputs.lowest_dose / values,
                np.nan,
            )

    unusable, questionable, warning = "unusable", "questionable", "warning"
    return (
        _bmds(unusable, "BMD not computed", lambda i: ~np.isfinite(i.bmd)),
        _bmds(unusable, "BMDL not computed", _bmdl_missing),
        _bmds(
            questionable,
            "fit p-value < 0.1",
            lambda i: np.isfinite(i.fit_p_value) & (i.fit_p_value < 0.1),
        ),
        _bmds(
            questionable,
            "|scaled residual near the BMD| > 2",
            lambda i: _above(np.abs(i.residual_near_bmd), 2.0),
        ),
        _bmds(
            questionable,
            "lowest dose / BMD > 10",
            lambda i: _above(lowest_over(i.bmd, i), 10.0),
        ),
        _bmds(
            questionable,
            "lowest dose / BMDL > 10",
            lambda i: _above(lowest_over(i.bmdl, i), 10.0),
        ),
        _bmds(
            questionable,
            "BMD / BMDL > 20",
            lambda i: _above(_positive_ratio(i.bmd, i.bmdl), 20.0),
        ),
        _bmds(
            questionable,
            "constant variance test p-value < 0.05",
            lambda i: (
                np.isfinite(i.constant_variance_p_value)
                & (i.constant_variance_p_value < 0.05)
            ),
        ),
        _bmds(
            warning,
            "BMD / highest dose > 1",
            lambda i: _above(
                _positive_ratio(i.bmd, np.full_like(i.bmd, i.top_dose)), 1.0
            ),
        ),
        _bmds(
            warning,
            "BMDL / highest dose > 1",
            lambda i: _above(
                _positive_ratio(i.bmdl, np.full_like(i.bmdl, i.top_dose)), 1.0
            ),
        ),
        _bmds(
            warning,
            "lowest dose / BMD > 3",
            lambda i: _above(lowest_over(i.bmd, i), 3.0),
        ),
        _bmds(
            warning,
            "lowest dose / BMDL > 3",
            lambda i: _above(lowest_over(i.bmdl, i), 3.0),
        ),
        _bmds(
            warning,
            "BMD / BMDL > 3",
            lambda i: _above(_positive_ratio(i.bmd, i.bmdl), 3.0),
        ),
        _bmds(
            warning,
            "curve's vehicle SD / observed vehicle SD > 1.5",
            lambda i: _above(i.vehicle_sd_ratio, 1.5),
        ),
        _bmds(
            warning,
            "|scaled residual at the vehicle| > 2",
            lambda i: _above(np.abs(i.residual_at_vehicle), 2.0),
        ),
    )


def _not_below_top(inputs: FlagInputs) -> BoolArray:
    return np.asarray(np.isfinite(inputs.bmd) & (inputs.bmd >= inputs.top_dose))


def _fit_p_at_most(inputs: FlagInputs) -> BoolArray:
    p = inputs.fit_p_value
    return np.asarray(np.isfinite(p) & (p <= 0.1))


def _ten_fold_below(inputs: FlagInputs) -> BoolArray:
    return np.asarray(
        np.isfinite(inputs.bmd) & (inputs.bmd < inputs.lowest_dose / 10.0)
    )


def _bound_ratio(inputs: FlagInputs, upper: bool) -> FloatArray:
    """BMD / BMDL, or BMDU / BMDL; infinite where the BMDL is 0 or the BMDU
    infinite; NaN without bounds, whose BMDL is NaN."""
    return _ratio(inputs.bmdu if upper else inputs.bmd, inputs.bmdl)


# NTP flags a BMD more than 10-fold below the lowest dose and keeps the gene.
KEPT = "(NTP keeps the gene)"

_CHECKS: dict[str, tuple[_Check, ...]] = {
    "bmds_2026": _bmds_checks(),
    "ntp_2018": (
        _Check(
            f"BMD more than 10-fold below the lowest dose {KEPT}",
            _ten_fold_below,
            kept=True,
        ),
        _Check(
            "BMD, BMDL or BMDU not computed, or beyond the search",
            lambda i: (
                ~np.isfinite(i.bmd)
                | (
                    i.bounds
                    & ~(np.isfinite(i.bmdl) & (i.bmdl > 0.0) & np.isfinite(i.bmdu))
                )
            ),
        ),
        _Check("BMD not below the highest dose", _not_below_top),
        _Check("fit p-value <= 0.1", _fit_p_at_most),
        _Check(
            "BMDU / BMDL >= 40", lambda i: _at_least(_bound_ratio(i, upper=True), 40.0)
        ),
    ),
    "etap_2024": (
        _Check(
            "BMD or BMDL not computed",
            lambda i: ~np.isfinite(i.bmd) | (i.bounds & np.isnan(i.bmdl)),
        ),
        _Check("BMD not below the highest dose", _not_below_top),
        _Check("fit p-value <= 0.1", _fit_p_at_most),
        _Check(
            "BMD / BMDL >= 20", lambda i: _at_least(_bound_ratio(i, upper=False), 20.0)
        ),
    ),
    "efsa_2022": (
        _Check(
            "BMD / BMDL > 20",
            lambda i: _above_or_infinite(_bound_ratio(i, upper=False), 20.0),
        ),
        _Check(
            "BMDU / BMDL > 50",
            lambda i: _above_or_infinite(_bound_ratio(i, upper=True), 50.0),
        ),
        _Check("BMD more than 10 times below the lowest dose", _ten_fold_below),
    ),
}


def _above_or_infinite(values: FloatArray, threshold: float) -> BoolArray:
    with np.errstate(invalid="ignore"):
        return np.asarray(~np.isnan(values) & (values > threshold))


def failed_checks(flag_set: str, inputs: FlagInputs) -> list[str]:
    """For each row, the texts of the checks it fails, joined by "; ", empty when it
    fails none."""
    n = inputs.bmd.size
    failures: list[list[str]] = [[] for _ in range(n)]
    for check in _CHECKS[flag_set]:
        for row in np.flatnonzero(check.failed(inputs)):
            failures[row].append(check.text)
    return ["; ".join(texts) for texts in failures]


def kept_rows(flag_set: str, inputs: FlagInputs) -> BoolArray:
    """Whether each row fails none of a set's checks but those whose document keeps
    the row: a gene that NTP flags and keeps, a model that BMDS keeps viable with
    warnings."""
    failed = np.zeros(inputs.bmd.size, dtype=bool)
    for check in _CHECKS[flag_set]:
        if not check.kept:
            failed |= check.failed(inputs)
    return ~failed


def accepted_rows(flag_set: str, inputs: FlagInputs) -> BoolArray:
    """Whether each row passes a document's acceptance, ``"ntp_2018"`` or
    ``"etap_2024"``: a fit p-value that is a number and no failed check but those
    the document keeps aside. The checks skip what a row lacks; acceptance asks for
    all that a criterion reads, so rows without bounds are refused."""
    if not inputs.bounds:
        raise ValueError(
            f"{flag_set!r} accepts a BMD by its bounds (NTP: BMDL and BMDU "
            "computed, BMDU / BMDL; ETAP: BMD / BMDL), and these BMDs have none, "
            "as for a cell type that supports only description"
        )
    return np.asarray(kept_rows(flag_set, inputs) & np.isfinite(inputs.fit_p_value))
