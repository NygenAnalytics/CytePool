"""Benchmark responses: the change in a dose-response curve that defines a benchmark
dose.

A benchmark dose (BMD) is the dose at which a fitted curve departs from its level at
dose 0 by a stated change, the benchmark response (BMR) (Crump 1984). Regulatory
guidance states the BMR of a gene in two ways. One is a multiple of the standard
deviation (SD) of a vehicle replicate: 1 SD in BMDS and NTP 2018, which US EPA 2012
asks to present beside any other BMR, and 1.349 SD in EPA's Transcriptomic
Assessment Products (US EPA 2024, after Thomas et al. 2007), since a gene's adverse
direction is unknown. The other is a change in the values' units, such as a 1.5-fold
change of expression, log2(1.5) on log2 values, which compares cell types: the SD of
a pseudobulk profile shrinks as its cells add up, so the same SD multiple is a
smaller change where cells are many.
:class:`BenchmarkResponse` holds the BMR as a logged value; how BMDs and their bounds
are computed is in
https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses.

References:
    Crump KS (1984). A new method for determining allowable daily intakes.
    Fundamental and Applied Toxicology 4, 854-871. doi:10.1016/0272-0590(84)90107-6

    US EPA (2012). Benchmark Dose Technical Guidance. EPA/100/R-12/001, Risk
    Assessment Forum, US Environmental Protection Agency, Washington, DC.

    US EPA (2024). Standard Methods for Development of EPA Transcriptomic Assessment
    Products (ETAPs). EPA/600/X-23/083, US Environmental Protection Agency, Office of
    Research and Development. doi:10.23645/epacomptox.25365496

    Thomas RS, Allen BC, Nong A, Yang L, Bermudez E, Clewell HJ, Andersen ME (2007). A
    method to integrate benchmark dose estimates with genomic data to assess the
    functional effects of chemical exposure. Toxicological Sciences 98, 240-248.
    doi:10.1093/toxsci/kfm092
"""

from dataclasses import dataclass
from typing import Any

from ._options import checked_choice, positive_finite_number, real_number

__all__ = ["BenchmarkResponse"]

SD = "sd"
ABSOLUTE = "absolute"
KINDS = (SD, ABSOLUTE)


@dataclass(frozen=True)
class BenchmarkResponse:
    """The change from a curve's level at dose 0 that defines a benchmark dose.

    Build one with :meth:`sd` or :meth:`absolute`. The BMD is the lowest dose x
    above 0 at which a fitted curve f reaches |f(x) - f(0)| = R, R the BMR's size:
    ``value`` times the SD of a vehicle replicate (``kind="sd"``), or ``value`` in
    the values' units (``kind="absolute"``)
    (https://docs.nygen.io/CytePool/statistics/potency#benchmark-doses).

    Args:
        kind: ``"sd"`` or ``"absolute"``.
        value: The multiple of the SD, or the change; finite and above 0.

    Raises:
        TypeError: If ``kind`` is not a str, or ``value`` is not a real number (a
            bool or a numpy masked array is not one).
        ValueError: If ``kind`` is not ``"sd"`` or ``"absolute"``, or ``value`` is
            not finite and above 0.

    Example:
        >>> bmr = BenchmarkResponse.sd(1.349)
        >>> bmr.label
        '1.349 SD'
        >>> BenchmarkResponse(**bmr.to_dict()) == bmr
        True
    """

    kind: str
    value: float

    def __post_init__(self) -> None:
        checked_choice(self.kind, "kind", KINDS)
        object.__setattr__(self, "value", positive_finite_number(self.value, "value"))

    @classmethod
    def sd(cls, multiple: float) -> "BenchmarkResponse":
        """A change of ``multiple`` times the SD of a vehicle replicate.

        The SD is the one of the fitted curve's likelihood, as BMDS estimates it
        with the curve. Under constant variance it is the curve's maximum-likelihood
        SD, sqrt((S_w + Q) / n), with S_w the within-group sum of squares, Q the
        curve's sum of squares about the dose-group means and n the samples. Where
        the curves weigh the dose-group means of a moderated fit, whose variance s^2
        has df degrees of freedom, the likelihood holds s^2 with the curve, and the
        SD is its maximum-likelihood value, sqrt((df s^2 + Q) / (df + K)) at K dose
        groups, scaled to one vehicle replicate by sqrt(n_0 C_00), with C_00 the
        unscaled variance of the vehicle's mean over its n_0 samples. The bounds of
        the BMD treat the SD as estimated, so it enters their likelihood
        (https://docs.nygen.io/CytePool/statistics/potency#benchmark-dose-bounds).
        The multiples of regulatory guidance are 1 (BMDS's default, NTP 2018) and
        1.349 (EPA's transcriptomic assessment products, US EPA 2024, after Thomas
        et al. 2007).

        Args:
            multiple: The multiple of the SD, finite and above 0.

        Raises:
            TypeError: If ``multiple`` is not a real number.
            ValueError: If ``multiple`` is not finite and above 0.
        """
        return cls(SD, _size(multiple, "multiple"))

    @classmethod
    def absolute(cls, change: float) -> "BenchmarkResponse":
        """A change of ``change`` in the values' units.

        On log2 values, ``absolute(np.log2(1.5))`` is a 1.5-fold change of
        expression, and ``absolute(np.log2(1.1))`` a 1.1-fold change, up or
        down. The same change compares across cell types, whose SDs differ
        with their cells per profile.

        Args:
            change: The size of the change, finite and above 0: the BMD is the
                first dose at which the curve rises or falls by it.

        Raises:
            TypeError: If ``change`` is not a real number.
            ValueError: If ``change`` is not finite and above 0.
        """
        return cls(ABSOLUTE, _size(change, "change"))

    @property
    def label(self) -> str:
        """The BMR as tables print it, with 6 significant digits, such as
        ``"1.349 SD"``, or ``"change of 0.584963"`` for ``absolute(np.log2(1.5))``."""
        number = f"{self.value:.6g}"
        return f"{number} SD" if self.kind == SD else f"change of {number}"

    def to_dict(self) -> dict[str, Any]:
        """The BMR as plain values that JSON stores: ``kind`` and ``value``.

        ``BenchmarkResponse(**bmr.to_dict())`` rebuilds it.
        """
        return {"kind": self.kind, "value": self.value}


def _size(value: Any, name: str) -> float:
    """A BMR's multiple or change: a finite real number above 0. A negative one asks
    for a fall, which every BMR reads already."""
    number = real_number(value, name)
    if number < 0.0:
        raise ValueError(
            f"{name} must be above 0, got {number}: give the size of the change; the "
            "BMD is read in either direction, up or down"
        )
    return positive_finite_number(number, name)
