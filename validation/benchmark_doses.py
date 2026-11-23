"""Check benchmark_doses and BenchmarkResponse against DRomics, pybmds, scipy and
mpmath, and the checks of BMDS, NTP, ETAP and EFSA against the same rules coded in R.

Series simulated by ``simulate_dose_series`` from fixed seeds (20261230 to 20261234):
``half_log``, the vehicle and 8 half-log doses from 0.01 to 30 ug/kg x 3, 150 genes,
two cell types of 500 and 30 cells per profile; ``decades``, the vehicle and 4 doses a
decade apart x 3; a series of the vehicle and 5 doses x 2 whose curves weigh the
dose-group means of a weighted and of a moderated fit (``dose_response`` with
``variance``); a series with one sample per treated dose, whose cell type supports only
description; and seven genes simulated here whose responses test the search: a step
below the lowest dose, a small plateau, a weak line, a bell, a falling exponential, a
late Hill curve and a weak response that a step at dose 0 fits almost as well. The
log2-CPM values of limma_voom's fit in ``dose_response`` are the samples' values.

References, each a separate program that sees the values, doses and settings and
nothing that CytePool computed from them, except where a row says "given":

- DRomics 2.6.3 (``DOSE_RSCRIPT``) fits half_log's large cell type, and two straight
  lines steep enough that their BMDs lie below ``minBMD``, with ``drcfit`` and
  computes ``bmdcalc(z = 1, x = 10)``; CytePool computes the BMD of DRomics' own
  parameters at a change of DRomics' SDres (BMD-zSD) and of 10% of its y0 (BMD-xfold),
  as absolute BMRs. Linear, exponential, Hill and the monotone Gauss-probit forms are
  closed forms (exact, 1e-10); the bells, which ``bmdcalc`` searches with ``uniroot``,
  are close within uniroot's default tolerance, 1.22e-4 on log dose; BMDs floored at
  ``minBMD``, a hundredth of the lowest dose, and the NA (reached beyond the top dose)
  and NaN (never reached) codes are identical to CytePool's statuses.
- pybmds 25.2 (``BMDS_PYTHON``) fits 12 genes and the edge genes with linear,
  polynomial 2, power, Hill and exponential 3 and 5 under normal errors of constant
  variance, at 1 SD and at a change of 0.5, with NLopt's fallback optimizer, since
  the wheel lacks NLopt's Luksan algorithms. BMDs given pybmds's parameters and SD are
  close (1e-5; pybmds stops its search within about 5e-7 of the root); test 2 against
  mpmath is exact (1e-10); test 4, the fit p-value that the checks read, of the fits
  where pybmds reaches CytePool's maximum, against mpmath's likelihood ratio of
  pybmds's fit to the exact maximum of a mean per dose group, every parameter
  counted, is close (1e-6 relative to p max(1, |log p|)), and pybmds's own test 4,
  which leaves out of K - p the parameters at a bound, is a deliberate difference;
  the checks of ``bmds_2026`` on pybmds's own numbers equal its recommender's default
  rules (identical); BMDLs and BMDUs where pybmds reaches
  CytePool's maximum agree within 5e-3, pybmds's root precision, and where they do
  not, the documented profile at pybmds's bound decides: a curve of the documented
  likelihood within the threshold there puts pybmds's bound inside the region, and
  scipy's and CytePool's profiles above the threshold put it outside.
- scipy 1.18.1 refits every case from 40 starts with the documented means and bounds,
  finds the BMD by brentq, and the root of deviance = threshold within 2% of each of
  CytePool's bounds, the constrained profile maximised by SLSQP from 12 starts and
  from the curve CytePool found at the bound: close (1e-6). From its own starts alone,
  scipy finds no curve below the threshold at any of CytePool's bounds, and every dose
  of a scan beyond each bound lies outside the region (identical). Given CytePool's
  curve at each bound, the documented means check that it lies within the parameters'
  bounds, changes by the BMR at the bound and has the threshold's deviance (close,
  1e-6).
- mpmath 1.3.0 gives the straight line's profile in closed form at 50 digits, under
  constant variance and for weighted means (Fieller's bound), with the threshold it
  computes from the level and the degrees of freedom: exact (1e-10).
- The sets of checks coded in R 4.4.3 from pybmds's rules, NTP 2018, US EPA 2024 and
  EFSA 2022, on every BMD table computed here: identical.

Requirements:
    Python 3.13 with numpy and numba, scipy and mpmath, run from the repository root.
    pybmds 25.2 (``BMDS_PYTHON``, a Python with pybmds; default ``python``).
    R 4.4 with DRomics 2.6.3 and jsonlite (``DOSE_RSCRIPT``).

Usage:
    PYTHONPATH=. python validation/benchmark_doses.py
    PYTHONPATH=. python validation/benchmark_doses.py --write-fixtures
    PYTHONPATH=. python validation/benchmark_doses.py --benchmark
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "benchmark_doses"
)

FLAGS_R = r"""
# The checks of benchmark doses by BMDS, NTP, ETAP and EFSA, coded in R from pybmds
# 25.2's default rules of BMDS's model recommendation (default.json and checks.py),
# NTP 2018 pp. 8 to 9, US EPA 2024 Sec. 3.4.5.3 and EFSA 2022 Sec. 2.6.5. The fit
# p-value is BMDS's test 4.
# Reads rows of BMDs, bounds and fit statistics; writes, per set, the failed checks
# of each row joined by "; ".
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1])
num <- function(x) {
  x <- unlist(lapply(x, function(v) if (is.null(v)) NA_real_ else v))
  x[x == "inf"] <- Inf
  as.numeric(x)
}
rows <- input$rows
bmd <- num(rows$bmd)
bmdl <- num(rows$bmdl)
bmdu <- num(rows$bmdu)
p <- num(rows$fit_p_value)
roi <- num(rows$residual_near_bmd)
vehicle_residual <- num(rows$residual_at_vehicle)
sd_ratio <- num(rows$vehicle_sd_ratio)
test2 <- num(rows$constant_variance_p_value)
bounds <- isTRUE(input$bounds)
lowest <- input$lowest_dose
top <- input$top_dose
n <- length(bmd)

known <- function(x) !is.na(x) & is.finite(x)
gt <- function(x, threshold) !is.na(x) & is.finite(x) & x > threshold
# A ratio with a bound of 0 below or infinity above is infinite.
ratio <- function(upper, lower) {
  out <- rep(NA_real_, length(upper))
  ok <- !is.na(upper) & !is.na(lower) & upper > 0 & lower >= 0
  out[ok] <- ifelse(lower[ok] == 0, Inf, upper[ok] / lower[ok])
  out
}
# pybmds's ratios: both values finite and above 0.
positive_ratio <- function(upper, lower) {
  ok <- known(upper) & known(lower) & upper > 0 & lower > 0
  ifelse(ok, upper / lower, NA_real_)
}
lowest_over <- function(x) ifelse(known(x) & x > 0, lowest / x, NA_real_)
bmdl_usable <- known(bmdl) & bmdl > 0

checks <- list(
  bmds_2026 = list(
    list("BMD not computed (unusable)", !known(bmd)),
    list("BMDL not computed (unusable)", rep(bounds, n) & !bmdl_usable),
    list("fit p-value < 0.1 (questionable)", known(p) & p < 0.1),
    list("|scaled residual near the BMD| > 2 (questionable)", gt(abs(roi), 2)),
    list("lowest dose / BMD > 10 (questionable)", gt(lowest_over(bmd), 10)),
    list("lowest dose / BMDL > 10 (questionable)", gt(lowest_over(bmdl), 10)),
    list("BMD / BMDL > 20 (questionable)", gt(positive_ratio(bmd, bmdl), 20)),
    list("constant variance test p-value < 0.05 (questionable)",
         known(test2) & test2 < 0.05),
    list("BMD / highest dose > 1 (warning)", gt(positive_ratio(bmd, rep(top, n)), 1)),
    list("BMDL / highest dose > 1 (warning)",
         gt(positive_ratio(bmdl, rep(top, n)), 1)),
    list("lowest dose / BMD > 3 (warning)", gt(lowest_over(bmd), 3)),
    list("lowest dose / BMDL > 3 (warning)", gt(lowest_over(bmdl), 3)),
    list("BMD / BMDL > 3 (warning)", gt(positive_ratio(bmd, bmdl), 3)),
    list("curve's vehicle SD / observed vehicle SD > 1.5 (warning)", gt(sd_ratio, 1.5)),
    list("|scaled residual at the vehicle| > 2 (warning)",
         gt(abs(vehicle_residual), 2))
  ),
  ntp_2018 = list(
    list("BMD more than 10-fold below the lowest dose (NTP keeps the gene)",
         known(bmd) & bmd < lowest / 10),
    list("BMD, BMDL or BMDU not computed, or beyond the search",
         !known(bmd) | (rep(bounds, n) & !(bmdl_usable & known(bmdu)))),
    list("BMD not below the highest dose", known(bmd) & bmd >= top),
    list("fit p-value <= 0.1", known(p) & p <= 0.1),
    list("BMDU / BMDL >= 40", {
      r <- if (bounds) ratio(bmdu, bmdl) else rep(NA_real_, n)
      !is.na(r) & r >= 40
    })
  ),
  etap_2024 = list(
    list("BMD or BMDL not computed", !known(bmd) | (rep(bounds, n) & is.na(bmdl))),
    list("BMD not below the highest dose", known(bmd) & bmd >= top),
    list("fit p-value <= 0.1", known(p) & p <= 0.1),
    list("BMD / BMDL >= 20", {
      r <- if (bounds) ratio(bmd, bmdl) else rep(NA_real_, n)
      !is.na(r) & r >= 20
    })
  ),
  efsa_2022 = list(
    list("BMD / BMDL > 20", {
      r <- if (bounds) ratio(bmd, bmdl) else rep(NA_real_, n)
      !is.na(r) & r > 20
    }),
    list("BMDU / BMDL > 50", {
      r <- if (bounds) ratio(bmdu, bmdl) else rep(NA_real_, n)
      !is.na(r) & r > 50
    }),
    list("BMD more than 10 times below the lowest dose", known(bmd) & bmd < lowest / 10)
  )
)
out <- list()
for (set in names(checks)) {
  texts <- character(n)
  for (check in checks[[set]]) {
    failed <- which(check[[2]])
    texts[failed] <- ifelse(texts[failed] == "", check[[1]],
                            paste(texts[failed], check[[1]], sep = "; "))
  }
  out[[set]] <- texts
}
out$versions <- list(R = paste(R.version$major, R.version$minor, sep = "."))
write_json(out, args[2], auto_unbox = TRUE, digits = NA)
"""


# ---------------------------------------------------------------------------------------
# References: none of this imports CytePool
# ---------------------------------------------------------------------------------------

CHI_SQUARE_LEVEL = 0.90  # each bound one-sided 95%


def documented_mean(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each model's mean as the chapter "Dose response" documents it, written apart
    from CytePool, with the reported parameters in the set's order."""
    from scipy.special import ndtr

    t = theta
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        if model == "linear":
            return t[0] + t[1] * x
        if model == "polynomial_2":
            return t[0] + t[1] * x + t[2] * x * x
        if model == "power":
            return t[0] + t[1] * np.where(x > 0, x ** t[2], 0.0)
        if model == "hill":
            g, v, k, n = t
            ratio = np.where(x > 0, 1.0 / (1.0 + (k / np.maximum(x, 1e-300)) ** n), 0.0)
            return g + v * ratio
        if model == "exponential_2":
            return t[0] * np.exp(t[1] * x)
        if model == "exponential_3":
            a, b, d = t
            return a * np.exp(np.sign(b) * (abs(b) * x) ** d)
        if model == "exponential_4":
            a, b, c = t
            return a * (c - (c - 1) * np.exp(-b * x))
        if model == "exponential_5":
            a, b, c, d = t
            return a * (c - (c - 1) * np.exp(-((b * x) ** d)))
        if model == "ll4":
            b, c, d, e = t
            u = np.where(x > 0, (x / e) ** b, 0.0)
            return c * (u / (1 + u)) + d / (1 + u)
        if model == "exponential":
            b, d, e = t  # DRomics: d + b (exp(x / e) - 1)
            return d + b * np.expm1(x / e)
        if "gauss_probit" in model:
            if model.endswith("symmetric"):
                b, d, e, f = t
                c = d
            elif model.endswith("monotone"):
                b, c, d, e = t
                f = 0.0
            else:
                b, c, d, e, f = t
            if model.startswith("log"):
                z = np.where(
                    x > 0, (np.log(np.maximum(x, 1e-300)) - np.log(e)) / b, -np.inf
                )
            else:
                z = (x - e) / b
            return f * np.exp(-z * z / 2) + d + (c - d) * ndtr(z)
    raise ValueError(model)


def documented_bounds(
    model: str, doses: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each model's bounds as the chapter documents them, from the doses; DRomics'
    exponential, whose abs(e) lies between D / 100 and 10,000 D, on the side of its
    fit's e."""
    inf = np.inf
    levels = np.unique(doses)
    top = levels[-1]
    positive = levels[levels > 0]
    lowest = positive[0]
    logs = np.log(positive)
    gap = np.max(np.diff(levels))
    log_gap = np.max(np.diff(logs)) if logs.size > 1 else inf
    rate = 100 / top
    table = {
        "linear": [(-inf, inf)] * 2,
        "polynomial_2": [(-inf, inf)] * 3,
        "power": [(-inf, inf), (-inf, inf), (1, 18)],
        "hill": [(-inf, inf), (-inf, inf), (0, 5 * top), (1, 18)],
        "exponential_2": [(0, inf), (-rate, rate)],
        "exponential_3": [(0, inf), (-rate, rate), (1, 18)],
        "exponential_4": [(0, inf), (0, rate), (0, 1000)],
        "exponential_5": [(0, inf), (0, rate), (0, 1000), (1, 18)],
        "ll4": [(0.1, 18), (-inf, inf), (-inf, inf), (lowest / 100, 5 * top)],
        "exponential": [(-inf, inf)] * 3,
    }
    if model == "polynomial_2":
        # b1 and b2 of one sign: that of the fit's change, which the profile keeps.
        side = (0.0, inf) if theta[1] + theta[2] > 0 else (-inf, 0.0)
        table["polynomial_2"] = [(-inf, inf), side, side]
    if model == "exponential":
        table["exponential"][2] = (
            (top / 100, 10_000 * top) if theta[2] > 0 else (-10_000 * top, -top / 100)
        )
    if model in table:
        pairs = table[model]
    else:
        log = model.startswith("log")
        width = (log_gap / 3, 10 * (logs[-1] - logs[0])) if log else (gap / 3, 10 * top)
        peak = (lowest, top) if log else (0.0, top)
        free = (-inf, inf)
        if model.endswith("symmetric"):
            pairs = [width, free, peak, free]
        elif model.endswith("monotone"):
            pairs = [width, free, free, peak]
        else:
            pairs = [width, free, free, peak, free]
    return np.array([p[0] for p in pairs], float), np.array(
        [p[1] for p in pairs], float
    )


def _change(model: str, theta: np.ndarray, b: float) -> float:
    values = documented_mean(model, theta, np.array([0.0, b]))
    return float(values[1] - values[0])


class _Profile:
    """The likelihood of one gene's dose-group means as the chapter defines it, and
    its maximum under a constraint on the change at a dose, from many starts.

    Constant variance: -2 LL = n log(S_w + Q) up to a constant, sigma profiled,
    with Q = sum_k n_k (ybar_k - f_k)^2. Weighted means of a first stage with
    variance s^2 on df: Q / s^2 for an absolute BMR (F threshold); for an SD-based
    BMR, (df + K) log sigma^2 + (df s^2 + Q) / sigma^2, sigma in the parameters."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.model = task["model"]
        self.doses = np.asarray(task["doses"], dtype=np.float64)
        self.means = np.asarray(task["means"], dtype=np.float64)
        self.weight = np.asarray(task["weight"], dtype=np.float64)  # W, K x K
        self.mode = task["mode"]  # "constant" or "plugin"
        self.a0 = float(task["a0"])  # S_w, or df s^2
        self.n_eff = float(task["n_eff"])  # n, or df + K
        self.s2 = float(task["s2"])
        self.form = task["form"]  # "absolute" or "sd"
        self.value = float(task["value"])  # R, or the SD multiple scaled to a replicate
        self.lower, self.upper = documented_bounds(
            self.model, self.doses, np.asarray(task["theta"], dtype=np.float64)
        )

    def q(self, theta: np.ndarray) -> float:
        r = self.means - documented_mean(self.model, theta, self.doses)
        return float(r @ self.weight @ r)

    def objective(self, theta: np.ndarray, sigma: float | None = None) -> float:
        q = self.q(theta)
        if self.form == "sd":
            assert sigma is not None
            return self.n_eff * math.log(sigma * sigma) + (self.a0 + q) / (
                sigma * sigma
            )
        if self.mode == "constant":
            return self.n_eff * math.log(self.a0 + q)
        return q / self.s2

    def unconstrained(self, starts: list[np.ndarray]) -> tuple[float, np.ndarray]:
        """The fit: the smallest Q from the starts, by bounded least squares."""
        from scipy.optimize import least_squares

        root = np.linalg.cholesky(self.weight).T  # W = R'R
        best: tuple[float, np.ndarray] = (math.inf, starts[0])
        lower = np.where(np.isfinite(self.lower), self.lower, -np.inf)
        upper = np.where(np.isfinite(self.upper), self.upper, np.inf)
        for start in starts:
            # strictly inside the finite bounds, as least_squares asks
            with np.errstate(invalid="ignore"):
                inside_low = np.where(
                    np.isfinite(lower), lower + 1e-12 * (abs(lower) + 1), -np.inf
                )
                inside_high = np.where(
                    np.isfinite(upper), upper - 1e-12 * (abs(upper) + 1), np.inf
                )
            x0 = np.clip(start, inside_low, inside_high)

            def residuals(theta: np.ndarray) -> np.ndarray:
                r = root @ (self.means - documented_mean(self.model, theta, self.doses))
                return np.where(np.isfinite(r), r, 1e10)

            try:
                fit = least_squares(
                    residuals,
                    x0,
                    bounds=(lower, upper),
                    method="trf",
                    xtol=1e-15,
                    ftol=1e-15,
                    gtol=1e-15,
                    max_nfev=4000,
                )
            except ValueError:
                continue
            value = self.q(fit.x)
            if value < best[0]:
                best = (value, fit.x)
        return best

    def constrained(
        self,
        b: float,
        s: float,
        starts: list[np.ndarray],
        sigma0: float,
        given: dict[str, Any] | None = None,
    ) -> float:
        """The smallest objective of curves whose change at b is s R (s R = s v sigma
        for an SD-based BMR), by SLSQP from each start, the SD from sigma0, and from a
        curve given with its own SD, on the parameters divided by the start's
        magnitudes, so that a parameter of 1e-27 and one of 18 move alike."""
        from scipy.optimize import minimize

        sd = self.form == "sd"
        points = [np.r_[start, sigma0] if sd else np.asarray(start) for start in starts]
        if given is not None:
            theta = np.asarray(given["theta"], dtype=np.float64)
            points.append(np.r_[theta, given["sigma"]] if sd else theta)
        lower = np.r_[self.lower, 1e-12] if sd else self.lower
        upper = np.r_[self.upper, np.inf] if sd else self.upper
        best = math.inf
        for point in points:
            unit = np.where(np.abs(point) > 0, np.abs(point), 1.0)
            bounds = [
                (
                    None if not math.isfinite(lo) else float(lo / u),
                    None if not math.isfinite(hi) else float(hi / u),
                )
                for lo, hi, u in zip(lower, upper, unit, strict=True)
            ]
            if sd:

                def fun(p: np.ndarray, unit: np.ndarray = unit) -> float:
                    v = p * unit
                    return self.objective(v[:-1], v[-1])

                def constraint(p: np.ndarray, unit: np.ndarray = unit) -> float:
                    v = p * unit
                    return _change(self.model, v[:-1], b) - s * self.value * v[-1]

            else:

                def fun(p: np.ndarray, unit: np.ndarray = unit) -> float:
                    return self.objective(p * unit)

                def constraint(p: np.ndarray, unit: np.ndarray = unit) -> float:
                    return _change(self.model, p * unit, b) - s * self.value

            x0 = point / unit
            try:
                with np.errstate(all="ignore"):
                    fit = minimize(
                        fun,
                        x0,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=[{"type": "eq", "fun": constraint}],
                        options={"ftol": 1e-15, "maxiter": 3000},
                    )
            except (ValueError, OverflowError, ZeroDivisionError, AssertionError):
                continue
            tolerance = 1e-9 * max(1.0, abs(self.value))
            if (
                math.isfinite(fit.fun)
                and abs(constraint(fit.x)) <= tolerance
                and fit.fun < best
            ):
                best = float(fit.fun)
        return best


def _starts(
    profile: _Profile, theta: np.ndarray, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """The fit, then perturbations of it within the bounds."""
    out = [np.asarray(theta, dtype=np.float64)]
    for _ in range(count - 1):
        trial = theta * np.exp(rng.normal(0.0, 0.4, theta.size)) + rng.normal(
            0.0, 0.05, theta.size
        )
        lo = np.where(np.isfinite(profile.lower), profile.lower, -1e6)
        hi = np.where(np.isfinite(profile.upper), profile.upper, 1e6)
        out.append(np.clip(trial, lo, hi))
    return out


@dataclass
class _ScipyFit:
    """scipy's fit of one case from 40 starts, with what its profile reads: the BMR's
    size, the objective at the fit, the threshold and the fit's SD."""

    profile: _Profile
    q_fit: float
    theta: np.ndarray
    size: float
    fit_objective: float
    threshold: float
    sigma0: float


def _scipy_fit(task: dict[str, Any], rng: np.random.Generator) -> _ScipyFit:
    from scipy.stats import chi2, f as f_dist

    profile = _Profile(task)
    theta0 = np.asarray(task["theta"], dtype=np.float64)
    q_fit, theta = profile.unconstrained(_starts(profile, theta0, 40, rng))
    if profile.form == "sd":
        sigma2 = (profile.a0 + q_fit) / profile.n_eff
        return _ScipyFit(
            profile,
            q_fit,
            theta,
            profile.value * math.sqrt(sigma2),
            profile.n_eff * math.log(sigma2) + profile.n_eff,
            float(chi2.ppf(CHI_SQUARE_LEVEL, 1)),
            math.sqrt(sigma2),
        )
    threshold = float(
        chi2.ppf(CHI_SQUARE_LEVEL, 1)
        if profile.mode == "constant"
        else f_dist.ppf(CHI_SQUARE_LEVEL, 1, task["df"])
    )
    return _ScipyFit(
        profile, q_fit, theta, profile.value, profile.objective(theta), threshold, 0.0
    )


def _first_crossing(
    fitted: _ScipyFit, low: float, high: float
) -> tuple[float, float] | None:
    """The BMD of scipy's fit and the direction of its change there, from a grid of
    20,001 doses refined by brentq; a crossing below the search is given as its lowest
    dose. None where the BMR is not reached."""
    from scipy.optimize import brentq

    model, theta, size = fitted.profile.model, fitted.theta, fitted.size
    grid = np.exp(np.linspace(math.log(low), math.log(high), 20001))
    changes = documented_mean(model, theta, grid) - documented_mean(
        model, theta, np.zeros(1)
    )
    hit = np.flatnonzero(np.abs(changes) >= size)
    if hit.size == 0:
        return None
    j = hit[0]
    s = 1.0 if changes[j] > 0 else -1.0
    if j == 0:
        return low, s
    bmd = brentq(
        lambda lx: _change(model, theta, math.exp(lx)) - s * size,
        math.log(grid[j - 1]),
        math.log(grid[j]),
        xtol=1e-15,
        rtol=1e-15,
    )
    return math.exp(bmd), s


def profile_bounds(task: dict[str, Any]) -> dict[str, Any]:
    """scipy's BMD and bounds of one case: its own fit from many starts, the first
    crossing of the BMR, the deviance at CytePool's bounds, the bounds as roots of
    deviance = threshold near each of CytePool's, with the deviance on a scan beyond
    each bound, which must exceed the threshold, and CytePool's curve at each bound
    checked with the documented means."""
    from scipy.optimize import brentq

    rng = np.random.default_rng(task["seed"])
    fitted = _scipy_fit(task, rng)
    profile, fit_objective, threshold = (
        fitted.profile,
        fitted.fit_objective,
        fitted.threshold,
    )
    low, high = task["low"], task["high"]
    out: dict[str, Any] = {
        "key": task["key"],
        "q_fit": fitted.q_fit,
        "size": fitted.size,
    }
    crossing = _first_crossing(fitted, low, high)
    if crossing is None:
        out.update(bmd=None, bmdl=None, bmdu=None)
        return out
    bmd, s = crossing
    starts = _starts(profile, fitted.theta, task["starts"], rng)

    def deviance(b: float) -> float:
        return profile.constrained(b, s, starts, fitted.sigma0) - fit_objective

    out.update(bmd=bmd, direction=s, threshold=threshold)
    for name in ("bmdl", "bmdu"):
        given = task[name]
        if given is not None and (given == 0.0 or given == math.inf):
            # An open end: the region reaches the search's end on that side.
            out[name] = None
            out[f"{name}_end"] = deviance(low if name == "bmdl" else high) - threshold
            continue
        if given is None or not math.isfinite(given) or given <= 0.0:
            out[name] = None
            continue
        # The deviance at CytePool's bound from scipy's own starts, and that of the
        # curve CytePool found there, by the documented means.
        out[f"{name}_at_ours"] = deviance(given) - threshold
        witness = task.get("witnesses", {}).get(name)
        if witness is not None:
            out[f"{name}_witness"] = checked_witness(
                profile, witness, s, fit_objective, threshold
            )

        # A root of deviance = threshold within 2% of CytePool's bound, the profile
        # maximised from scipy's starts and from CytePool's curve at the bound.
        def near(b: float, witness: dict[str, Any] | None = witness) -> float:
            return (
                profile.constrained(b, s, starts, fitted.sigma0, witness)
                - fit_objective
                - threshold
            )

        a, c = given * 0.98, given * 1.02
        da, dc = near(a), near(c)
        root = None
        if math.isfinite(da) and math.isfinite(dc) and da * dc < 0:
            root = math.exp(
                brentq(
                    lambda lx: near(math.exp(lx)),
                    math.log(a),
                    math.log(c),
                    xtol=1e-12,
                    rtol=1e-12,
                )
            )
        # Beyond the bound, every dose of a scan must lie outside the region.
        side = -1 if name == "bmdl" else 1
        far = low if side < 0 else high
        n = max(2, int(math.ceil(abs(math.log10(far / given)) * 6)))
        beyond = np.exp(np.linspace(math.log(given), math.log(far), n + 1))[1:]
        out[name] = root
        out[f"{name}_beyond"] = float(
            min(deviance(float(b)) for b in beyond) - threshold
        )
    return out


def checked_witness(
    profile: _Profile,
    witness: dict[str, Any],
    s: float,
    fit_objective: float,
    threshold: float,
) -> dict[str, Any]:
    """A curve given at a dose b, checked with the documented means and bounds: whether
    its parameters lie within the bounds, how far its change at b lies from s R (s v
    sigma for an SD-based BMR, sigma the curve's SD), and its deviance from the fit
    less the threshold."""
    theta = np.asarray(witness["theta"], dtype=np.float64)
    b = float(witness["dose"])
    slack = 1e-12 * (np.abs(theta) + 1.0)
    inside = bool(
        np.all(theta >= profile.lower - slack)
        and np.all(theta <= profile.upper + slack)
    )
    if profile.form == "sd":
        sigma = float(witness["sigma"])
        target = s * profile.value * sigma
        objective = profile.objective(theta, sigma)
    else:
        target = s * profile.value
        objective = profile.objective(theta)
    return {
        "inside": inside,
        "constraint": abs(_change(profile.model, theta, b) - target)
        / max(abs(target), 1e-300),
        "deviance": objective - fit_objective - threshold,
    }


def profile_probe(task: dict[str, Any]) -> dict[str, Any]:
    """The deviance at one dose less the threshold, maximised from scipy's starts and
    from the curve CytePool found there, and that curve's own, checked with the
    documented means: where pybmds's bound lies against the region."""
    rng = np.random.default_rng(task["seed"])
    fitted = _scipy_fit(task, rng)
    out: dict[str, Any] = {"key": task["key"], "bound": task["probe_bound"]}
    crossing = _first_crossing(fitted, task["low"], task["high"])
    if crossing is None:
        out.update(at_probe=None, witness=None)
        return out
    _, s = crossing
    starts = _starts(fitted.profile, fitted.theta, task["starts"], rng)
    b = float(task["probe"])
    given = task["witnesses"]["probe"]
    out["at_probe"] = (
        fitted.profile.constrained(b, s, starts, fitted.sigma0, given)
        - fitted.fit_objective
        - fitted.threshold
    )
    out["witness"] = checked_witness(
        fitted.profile, given, s, fitted.fit_objective, fitted.threshold
    )
    return out


def scipy_main(directory: Path) -> None:
    """The profile bounds of every case, and the deviance at each probe, in parallel
    processes."""
    from concurrent.futures import ProcessPoolExecutor

    import scipy

    tasks = json.loads((directory / "scipy_inputs.json").read_text())
    with ProcessPoolExecutor(
        max_workers=int(os.environ.get("SCIPY_WORKERS", "4"))
    ) as pool:
        results = list(pool.map(profile_bounds, tasks["cases"], chunksize=1))
        probes = list(pool.map(profile_probe, tasks["probes"], chunksize=1))
    out = {
        "versions": {"scipy": scipy.__version__},
        "results": results,
        "probes": probes,
    }
    (directory / "scipy.json").write_text(json.dumps(out))


def line_bounds(task: dict[str, Any]) -> dict[str, Any]:
    """The straight line's BMD and bounds in closed form at 50 digits (mpmath).

    With the slope b1 of the weighted fit, its unscaled variance v = [(X'WX)^-1]_11,
    the weighted residual sum of squares Q and A = a0 + Q, the profile at dose b fixes
    the slope at t / b, t = s R or s R(sigma), and adds (b1 - t / b)^2 / v to Q:
    - constant variance, absolute R: n log(A + (b1 - s R / b)^2 / v) - n log A;
    - weighted means, absolute R: (b1 - s R / b)^2 / (v s^2), Fieller's (1954) bound;
    - SD-based R = v_R sigma: sigma solves N sigma^2 + (b1 u / v) sigma - (A + b1^2
      / v) = 0 with u = s v_R / b, the -2 log-likelihood N log sigma^2 + (A + (b1 - u
      sigma)^2 / v) / sigma^2 against its value at the fit, N log(A / N) + N.
    Each bound is the root of deviance = threshold on the side of the BMD, by
    bisection at 50 digits; 0 or infinity where none lies within [low, high]. The
    threshold is computed here from the level, 0.90: the chi-square(1) quantile, or,
    for an absolute BMR of weighted means, the F(1, df) quantile."""
    import mpmath as mp

    mp.mp.dps = 50
    x = [mp.mpf(v) for v in task["doses"]]
    y = [mp.mpf(v) for v in task["means"]]
    w = mp.matrix([[mp.mpf(v) for v in row] for row in task["weight"]])
    k = len(x)
    design = mp.matrix([[1, xi] for xi in x])
    gram = design.T * w * design
    inverse = gram**-1
    beta = inverse * (design.T * w * mp.matrix(y))
    g, b1 = beta[0], beta[1]
    residual = mp.matrix([y[i] - g - b1 * x[i] for i in range(k)])
    q = (residual.T * w * residual)[0]
    v = inverse[1, 1]
    a = mp.mpf(task["a0"]) + q
    n_eff = mp.mpf(task["n_eff"])
    s2 = mp.mpf(task["s2"])
    threshold = line_threshold(task["mode"], task["form"], task["df"])
    s = 1 if b1 > 0 else -1
    form, mode = task["form"], task["mode"]
    value = mp.mpf(task["value"])

    def objective(b: Any) -> Any:
        if form == "sd":
            u = s * value / b
            quad_b = b1 * u / v
            constant = a + b1 * b1 / v
            sigma = (-quad_b + mp.sqrt(quad_b * quad_b + 4 * n_eff * constant)) / (
                2 * n_eff
            )
            return (
                n_eff * mp.log(sigma * sigma)
                + (a + (b1 - u * sigma) ** 2 / v) / sigma**2
            )
        excess = (b1 - s * value / b) ** 2 / v
        if mode == "constant":
            return n_eff * mp.log(a + excess)
        return (q + excess) / s2

    if form == "sd":
        fit_objective = n_eff * mp.log(a / n_eff) + n_eff
        size = value * mp.sqrt(a / n_eff)
    else:
        fit_objective = n_eff * mp.log(a) if mode == "constant" else q / s2
        size = value
    bmd = size / abs(b1)

    def excess(b: Any) -> Any:
        return objective(b) - fit_objective - threshold

    low, high = mp.mpf(task["low"]), mp.mpf(task["high"])
    out: dict[str, Any] = {"key": task["key"], "bmd": float(bmd), "slope": float(b1)}
    for name, end in (("bmdl", low), ("bmdu", high)):
        if excess(end) <= 0:
            out[name] = 0.0 if name == "bmdl" else math.inf
            continue
        inner, outer = mp.log(bmd), mp.log(end)
        for _ in range(400):
            middle = (inner + outer) / 2
            if excess(mp.exp(middle)) <= 0:
                inner = middle
            else:
                outer = middle
        out[name] = float(mp.exp((inner + outer) / 2))
    # Fieller's bound beside the constant-variance one: the t-based bound with the
    # pooled variance on its degrees of freedom, as information.
    return out


def line_threshold(mode: str, form: str, df: float) -> Any:
    """The 0.90 quantile that a straight line's bounds read, at 50 digits: chi-square
    with 1 degree of freedom, the square of the standard normal's 0.95 quantile; or,
    for an absolute BMR of weighted means, F(1, df), the root of its distribution
    function, the regularized incomplete beta function I_{f / (f + df)}(1/2, df/2),
    at 0.90."""
    import mpmath as mp

    mp.mp.dps = 50
    level = mp.mpf(CHI_SQUARE_LEVEL)
    chi_square = 2 * mp.erfinv(level) ** 2
    if mode == "constant" or form == "sd" or math.isinf(df):
        return chi_square
    half_df = mp.mpf(df) / 2

    def excess(f: Any) -> Any:
        return (
            mp.betainc(mp.mpf(1) / 2, half_df, 0, f / (f + df), regularized=True)
            - level
        )

    return mp.findroot(excess, chi_square)


def constant_variance_test(task: dict[str, Any]) -> dict[str, Any]:
    """BMDS's test 2 at 50 digits: the likelihood ratio of one variance against a
    variance per dose group, each at its maximum-likelihood estimate (the sum of
    squares over the samples), n log(S_w / n) - sum_k n_k log(S_k / n_k), and its
    chi-square tail on K - 1 degrees of freedom."""
    import mpmath as mp

    mp.mp.dps = 50
    groups: dict[float, list[Any]] = {}
    for dose, value in zip(task["doses"], task["values"], strict=True):
        groups.setdefault(dose, []).append(mp.mpf(value))
    n = sum(len(values) for values in groups.values())
    statistic = mp.mpf(0)
    total = mp.mpf(0)
    for values in groups.values():
        mean = mp.fsum(values) / len(values)
        squares = mp.fsum((value - mean) ** 2 for value in values)
        total += squares
        statistic -= len(values) * mp.log(squares / len(values))
    statistic += n * mp.log(total / n)
    df = len(groups) - 1
    p_value = mp.gammainc(mp.mpf(df) / 2, statistic / 2, mp.inf, regularized=True)
    # The log-likelihood of a variance per dose group at its maximum.
    a2 = mp.fsum(
        -len(values)
        / mp.mpf(2)
        * (mp.log(2 * mp.pi * squares_of(values) / len(values)) + 1)
        for values in groups.values()
    )
    # ... and of a mean per dose group with one variance, at S_w / n.
    a1 = -n / mp.mpf(2) * (mp.log(2 * mp.pi * total / n) + 1)
    return {
        "key": task["key"],
        "statistic": float(statistic),
        "p_value": float(p_value),
        "a2_log_likelihood": float(a2),
        "a1_log_likelihood": float(a1),
    }


def fit_test(task: dict[str, Any], a1: Any) -> dict[str, Any]:
    """Test 4 of a fit at 50 digits: twice its log-likelihood's distance from the
    maximum of a mean per dose group with one variance, on a chi-square of K - p
    degrees of freedom, every mean parameter counted."""
    import mpmath as mp

    mp.mp.dps = 50
    statistic = max(2 * (a1 - mp.mpf(task["log_likelihood"])), mp.mpf(0))
    half_df = mp.mpf(task["df"]) / 2
    p_value = mp.gammainc(half_df, statistic / 2, mp.inf, regularized=True)
    return {"key": task["key"], "p_value": float(p_value)}


def squares_of(values: list[Any]) -> Any:
    """The sum of squares of values about their mean, at mpmath's precision."""
    import mpmath as mp

    mean = mp.fsum(values) / len(values)
    return mp.fsum((value - mean) ** 2 for value in values)


def mpmath_main(directory: Path) -> None:
    import mpmath as mp

    tasks = json.loads((directory / "mpmath_inputs.json").read_text())
    test2 = [constant_variance_test(t) for t in tasks["test2"]]
    a1 = {r["key"]: mp.mpf(r["a1_log_likelihood"]) for r in test2}
    out = {
        "versions": {"mpmath": mp.__version__},
        "results": [line_bounds(t) for t in tasks["lines"]],
        "test2": test2,
        "test4": [fit_test(t, a1[t["gene"]]) for t in tasks["test4"]],
    }
    (directory / "mpmath.json").write_text(json.dumps(out))


# The mean parameters of each of BMDS's models, as their documented means count them.
MEAN_PARAMETERS = {
    "linear": 2,
    "polynomial_2": 3,
    "power": 3,
    "hill": 4,
    "exponential_3": 3,
    "exponential_5": 4,
}

PYBMDS_CLASSES = {
    "linear": "Linear",
    "polynomial_2": "Polynomial",
    "power": "Power",
    "hill": "Hill",
    "exponential_3": "ExponentialM3",
    "exponential_5": "ExponentialM5",
}


def _pybmds_case(task: dict[str, Any]) -> dict[str, Any]:
    """pybmds's fits of one gene with each model, normal errors of constant variance,
    at one BMR, with its recommender's verdicts."""
    import pybmds

    dataset = pybmds.ContinuousIndividualDataset(
        doses=task["doses"], responses=task["values"]
    )
    session = pybmds.Session(dataset=dataset)
    risk = {
        "sd": pybmds.ContinuousRiskType.StandardDeviation,
        "absolute": pybmds.ContinuousRiskType.AbsoluteDeviation,
    }[task["form"]]
    for model in task["models"]:
        settings = {
            "disttype": pybmds.ContinuousDistType.normal,
            "bmr_type": risk,
            "bmr": task["value"],
        }
        if model == "polynomial_2":
            settings["degree"] = 2
        session.add_model(
            getattr(pybmds.Models, PYBMDS_CLASSES[model]), settings=settings
        )
    session.execute()
    session.recommend()
    recommendation = session.recommender.results
    rows = []
    for i, (model, fitted) in enumerate(zip(task["models"], session.models)):
        result = fitted.results.model_dump()
        notes = recommendation.bmds_model_notes[i]
        rows.append(
            {
                "model": model,
                "names": result["parameters"]["names"],
                "values": result["parameters"]["values"],
                "is_increasing": bool(fitted.settings.is_increasing),
                "log_likelihood": result["fit"]["loglikelihood"],
                "bmd": result["bmd"],
                "bmdl": result["bmdl"],
                "bmdu": result["bmdu"],
                "test2_p": result["tests"]["p_values"][1],
                "test4_p": result["tests"]["p_values"][3],
                "a2_log_likelihood": result["deviance"]["loglikelihoods"][1],
                "roi": fitted.results.get_parameter("roi"),
                "roi_control": fitted.results.get_parameter("roi_control"),
                "est_sd_control": result["gof"]["est_sd"][0],
                "calc_sd_control": result["gof"]["calc_sd"][0],
                "notes": {str(int(b)): list(messages) for b, messages in notes.items()},
            }
        )
    return {"key": task["key"], "rows": rows}


def pybmds_main(directory: Path) -> None:
    """pybmds's fits of every case, in parallel processes."""
    from concurrent.futures import ProcessPoolExecutor
    from importlib.metadata import version

    tasks = json.loads((directory / "pybmds_inputs.json").read_text())
    with ProcessPoolExecutor(
        max_workers=int(os.environ.get("SCIPY_WORKERS", "4"))
    ) as pool:
        results = list(pool.map(_pybmds_case, tasks, chunksize=1))
    versions = {"pybmds": version("pybmds"), "python": sys.version.split()[0]}
    (directory / "pybmds.json").write_text(
        json.dumps({"versions": versions, "results": results})
    )


DROMICS_R = r"""
suppressPackageStartupMessages({library(jsonlite); library(DRomics)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
doses <- as.numeric(input$doses); values <- as.matrix(input$values)
frame <- data.frame(item = c("dose", paste0("g", seq_len(nrow(values)) - 1)),
                    rbind(doses, values), check.names = FALSE)
o <- continuousomicdata(frame, check = TRUE)
selected <- itemselect(o, select.method = "quadratic", FDR = 0.99999999)
fits <- drcfit(selected, progressbar = FALSE, postfitfilter = FALSE,
               deltaAICminfromnullmodel = -1e9)
r <- bmdcalc(fits, z = input$z, x = input$x)
res <- r$res
out <- list(versions = list(R = as.character(getRversion()),
                            DRomics = as.character(packageVersion("DRomics"))),
            minBMD = r$minBMD,
            rows = lapply(seq_len(nrow(res)), function(i) {
              row <- res[i, ]
              list(id = as.character(row$id), model = as.character(row$model),
                   typology = as.character(row$typology),
                   b = row$b, c = row$c, d = row$d, e = row$e, f = row$f,
                   SDres = row$SDres, y0 = row$y0,
                   BMD_zSD = row$BMD.zSD, BMD_xfold = row$BMD.xfold,
                   zSD_code = if (is.nan(row$BMD.zSD)) "NaN" else if (is.na(row$BMD.zSD)) "NA" else "value",
                   xfold_code = if (is.nan(row$BMD.xfold)) "NaN" else if (is.na(row$BMD.xfold)) "NA" else "value")
            }))
write_json(out, args[[2]], digits = NA, auto_unbox = TRUE, null = "null", na = "null")
"""


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    """Run an R program on a JSON payload; the variable names its Rscript."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "program.R").write_text(program)
        (directory / "input.json").write_text(json.dumps(payload))
        command = [
            os.environ.get(variable, "Rscript"),
            str(directory / "program.R"),
            str(directory / "input.json"),
            str(directory / "output.json"),
        ]
        subprocess.run(command, check=True)
        result: dict[str, Any] = json.loads((directory / "output.json").read_text())
    return result


def run_python(
    flag: str, interpreter: str, inputs: dict[str, str], output: str
) -> dict[str, Any]:
    """Run this script's reference part in another interpreter on the inputs. The
    references print nothing of their own; pybmds's parallel fits print NLopt's message
    that the wheel lacks the Luksan algorithms, once per optimizer call, and pybmds's
    warning that a BMD read from the model's CDF differs from the model's, interleaved,
    so their counts are printed instead."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, text in inputs.items():
            (directory / name).write_text(text)
        completed = subprocess.run(
            [interpreter, "-I", __file__, flag, str(directory)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        if completed.stdout:
            print(
                f"{flag[2:]} printed {len(completed.stdout.splitlines()):,} lines: "
                "NLopt's 'attempting to use NLOPT_LD_LBFGS, but Luksan code disabled' "
                f"{completed.stdout.count('Luksan code disabled'):,} times, 'CDF BMD "
                f"differs from model BMD' {completed.stdout.count('CDF BMD differs'):,} "
                "times"
            )
        result: dict[str, Any] = json.loads((directory / output).read_text())
    return result


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------

HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
DECADES = [0.0, 0.01, 0.1, 1.0, 10.0]
TWO_REPLICATES = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
EXACT, CLOSE = 1e-10, 1e-6
BELL_TOLERANCE = 1.22e-4  # uniroot's default tolerance on DRomics' searched scale
PYBMDS_BMD = 1e-5  # pybmds stops its search within about 5e-7 of the root
PYBMDS_BOUND = 5e-3  # pybmds's bounds missed the exact profile by up to 1.9e-3
OPEN_END = 1e-6  # deviance above the threshold that still counts as reaching the end
WITNESS_CONSTRAINT = 1e-8  # relative miss of a given curve's change at its dose
FLAG_SETS = ("bmds_2026", "ntp_2018", "etap_2024", "efsa_2022")
EPA = ("linear", "polynomial_2", "power", "hill", "exponential_3", "exponential_5")


class Rows:
    """The printed checks, each with its reference, value, tolerance and outcome."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.failures = 0

    def add(
        self,
        check: str,
        reference: str,
        value: str,
        tolerance: str,
        passed: bool | None,
        deliberate: bool = False,
    ) -> None:
        """A check: passed True or False, None for information, or a deliberate
        difference, which the page lists with its reason."""
        if deliberate:
            outcome = "deliberate difference"
        else:
            outcome = (
                "information" if passed is None else ("pass" if passed else "FAIL")
            )
        self.failures += outcome == "FAIL"
        self.rows.append([check, reference, value, tolerance, outcome])

    def print(self) -> None:
        print("| check | reference | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def relative(ours: float, reference: float) -> float:
    if math.isinf(ours) or math.isinf(reference):
        return 0.0 if ours == reference else math.inf
    return abs(ours - reference) / max(abs(reference), 1e-300)


def axis() -> Any:
    from cytepool.dose_axes import DoseAxis

    return DoseAxis("dose", unit="ug/kg", metric="administered")


def simulated(
    doses: list[float], replicates: int, seed: int, n_genes: int, cells: Any
) -> Any:
    from cytepool.simulated_dose_series import simulate_dose_series

    return simulate_dose_series(
        doses,
        replicates=replicates,
        cell_types=2,
        cells=cells,
        n_genes=n_genes,
        responsive=0.6,
        rng=seed,
    )


@dataclass
class Series:
    """A cell type's samples: doses and log2-CPM values from dose_response's fit."""

    name: str
    doses: np.ndarray
    values: np.ndarray  # (n_samples, n_genes)
    feature_ids: list[str]


def constant_series() -> list[Series]:
    """The cell types of three series, each fitted by limma_voom in dose_response,
    whose log2-CPM values the curves are fitted to under constant variance."""
    from cytepool.design import Design
    from cytepool.dose_responses import dose_response

    out = []
    for name, doses, replicates, seed, cells in (
        ("half_log", HALF_LOG, 3, 20261230, (500.0, 30.0)),
        ("decades", DECADES, 3, 20261231, (200.0, 200.0)),
    ):
        cube, _ = simulated(doses, replicates, seed, 150, cells)
        result = dose_response(cube, Design(cube.samples, "~ 1"), dose=axis())
        for group in result.groups:
            fit = result.fit(group)
            out.append(
                Series(
                    f"{name}/{group}",
                    np.asarray(fit.samples["dose"], dtype=np.float64),
                    np.asarray(fit.log2_cpm, dtype=np.float64),
                    list(fit.features["feature_id"]),
                )
            )
    return out


def edge_series() -> Series:
    """Genes simulated here with responses that test the search: a step below the
    lowest dose, a small plateau, a weak line, a bell, a falling exponential, a late
    Hill curve and a weak response that a step at dose 0 fits almost as well."""
    rng = np.random.default_rng(20261232)
    x = np.repeat(np.asarray(HALF_LOG), 3)
    pos = np.maximum(x, 1e-300)
    means = [
        6.0 + 1.0 * x**6 / (0.002**6 + x**6),
        6.0 + 0.25 * x**2 / (0.5**2 + x**2),
        6.0 + 0.01 * x,
        6.0 + 1.0 * np.exp(-0.5 * (np.log10(pos / 0.3) / 0.5) ** 2) * (x > 0),
        6.0 - 1.5 * (1.0 - np.exp(-x / 3.0)),
        6.0 + 1.2 * x**4 / (20.0**4 + x**4),
        6.0 + 0.35 * x**1.5 / (1.0 + x**1.5),
    ]
    values = np.column_stack([m + rng.normal(0.0, 0.3, x.size) for m in means])
    return Series("edge", x, values, [f"edge_{i}" for i in range(values.shape[1])])


# ---- DRomics, given its parameters --------------------------------------------------


def steep_items(doses: np.ndarray) -> np.ndarray:
    """Two straight lines steep enough that a change of their SD, or of 10% of their
    level at dose 0, comes below a hundredth of the lowest dose, DRomics' minBMD: a
    rise and a fall of 10,000 per dose unit, with noise of SD 0.3 about a level of 6."""
    rng = np.random.default_rng(20261235)
    lines = [6.0 + 1e4 * doses, 6.0 - 1e4 * doses]
    return np.column_stack([line + rng.normal(0.0, 0.3, doses.size) for line in lines])


DROMICS_NAMES = {"linear": "linear", "exponential": "exponential", "Hill": "ll4"}


def dromics_model(row: dict[str, Any]) -> str:
    """The model of the "dromics" set that a row of bmdcalc's results is."""
    name = row["model"]
    if name in DROMICS_NAMES:
        return DROMICS_NAMES[name]
    prefix = "log_gauss_probit" if name == "log-Gauss-probit" else "gauss_probit"
    if row["f"] in (None, 0.0) or row["f"] == 0:
        return f"{prefix}_monotone"
    if row["c"] is None or row["c"] == row["d"]:
        return f"{prefix}_symmetric"
    return prefix


def compare_dromics(
    rows: Rows, series: Series, reference: dict[str, Any]
) -> list[dict]:
    """CytePool's BMD at DRomics' parameters and BMRs: z = 1 SDres and x = 10% of y0
    as absolute changes, without bounds, against DRomics' BMD-zSD and BMD-xfold."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    fits = fit_dose_curves(series.doses, series.values, models="dromics")
    min_bmd = float(reference["minBMD"])
    closed, bells, codes, floors = [], [], [], []
    records = []
    for row in reference["rows"]:
        g = int(row["id"][1:])
        model = dromics_model(row)
        if model not in fits.models:
            continue
        m = fits.models.index(model)
        parameters = np.array(fits.parameters)
        parameters[g, m, :] = np.nan
        for letter in "bcdef":
            if letter in fits.parameter_names and row.get(letter) is not None:
                parameters[g, m, fits.parameter_names.index(letter)] = row[letter]
        one = replace_gene(fits, g, parameters)
        for label, size, ref, code in (
            ("zSD", row["SDres"], row["BMD_zSD"], row["zSD_code"]),
            ("xfold", 0.1 * abs(row["y0"]), row["BMD_xfold"], row["xfold_code"]),
        ):
            result = benchmark_doses(
                one, BenchmarkResponse.absolute(size), model=model, bounds=False
            )
            bmd, status = float(result.bmd[0, 0]), str(result.status[0, 0])
            records.append(
                {
                    "id": row["id"],
                    "model": model,
                    "bmr": label,
                    "size": size,
                    "parameters": [row.get(k) for k in "bcdef"],
                    "bmd": ref,
                    "code": code,
                    "status": status,
                }
            )
            if code == "value":
                if bmd == min_bmd or status == "below a hundredth of the lowest dose":
                    floors.append(
                        status == "below a hundredth of the lowest dose"
                        and ref == min_bmd
                    )
                    continue
                difference = relative(bmd, ref)
                if model.startswith(("gauss", "log_gauss")) and not model.endswith(
                    "monotone"
                ):
                    bells.append(abs(math.log(bmd) - math.log(ref)))
                else:
                    closed.append(difference)
            else:
                # NA: reached beyond the top dose; NaN: never reached.
                allowed = {
                    "NA": ("above the top dose", "not reached"),
                    "NaN": ("not reached",),
                }
                codes.append(status in allowed[code])
    rows.add(
        f"BMD-zSD and BMD-xfold given DRomics' parameters, closed forms ({len(closed)})",
        "DRomics 2.6.3 bmdcalc",
        f"largest relative difference {max(closed, default=0.0):.1e}",
        f"exact ({EXACT:g})",
        max(closed, default=0.0) <= EXACT,
    )
    rows.add(
        f"The same for bells, which bmdcalc searches with uniroot ({len(bells)})",
        "DRomics 2.6.3 bmdcalc",
        f"largest difference on log dose {max(bells, default=0.0):.1e}",
        f"close ({BELL_TOLERANCE:g} on log dose)",
        max(bells, default=0.0) <= BELL_TOLERANCE,
    )
    rows.add(
        f"BMDs at minBMD, the lowest dose / 100 ({len(floors)}), and NA or NaN ({len(codes)})",
        "DRomics 2.6.3 bmdcalc",
        f"{sum(floors) + sum(codes)} of {len(floors) + len(codes)} as DRomics codes them",
        "identical, with at least one BMD at minBMD",
        bool(floors) and all(floors) and all(codes),
    )
    return records


def replace_gene(fits: Any, g: int, parameters: np.ndarray) -> Any:
    """The fits of one gene, with its parameters replaced."""
    import dataclasses

    keep = slice(g, g + 1)
    fields = {}
    for field in dataclasses.fields(fits):
        value = getattr(fits, field.name)
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == fits.means.shape[0]
            and field.name not in ("doses", "n_samples")
        ):
            fields[field.name] = value[keep]
    fields["parameters"] = parameters[keep]
    return dataclasses.replace(fits, **fields)


# ---- pybmds -------------------------------------------------------------------------

PYBMDS_MESSAGES = {
    "BMD not computed (unusable)": ("BMD does not exist", 2),
    "BMDL not computed (unusable)": ("BMDL does not exist", 2),
    "fit p-value < 0.1 (questionable)": ("Goodness of fit p-value < 0.1", 1),
    "|scaled residual near the BMD| > 2 (questionable)": (
        "|Residual near BMD| > 2.0",
        1,
    ),
    "lowest dose / BMD > 10 (questionable)": ("lowest dose/BMD ratio > 10.0", 1),
    "lowest dose / BMDL > 10 (questionable)": ("lowest dose/BMDL ratio > 10.0", 1),
    "BMD / BMDL > 20 (questionable)": ("BMD/BMDL ratio > 20.0", 1),
    "constant variance test p-value < 0.05 (questionable)": (
        "Constant variance test failed (Test 2 p-value < 0.05)",
        1,
    ),
    "BMD / highest dose > 1 (warning)": ("BMD/highest dose ratio > 1.0", 0),
    "BMDL / highest dose > 1 (warning)": ("BMDL/highest dose ratio > 1.0", 0),
    "lowest dose / BMD > 3 (warning)": ("lowest dose/BMD ratio > 3.0", 0),
    "lowest dose / BMDL > 3 (warning)": ("lowest dose/BMDL ratio > 3.0", 0),
    "BMD / BMDL > 3 (warning)": ("BMD/BMDL ratio > 3.0", 0),
    "curve's vehicle SD / observed vehicle SD > 1.5 (warning)": (
        "Control stdev. fit > 1.5",
        0,
    ),
    "|scaled residual at the vehicle| > 2 (warning)": ("Residual at control > 2.0", 0),
}


def p_distance(ours: float, reference: float) -> float:
    """The difference of two p-values relative to p max(1, |log p|)."""
    scale = max(reference * max(1.0, abs(math.log(max(reference, 1e-300)))), 1e-300)
    return abs(ours - reference) / scale


def pybmds_value(value: Any) -> float:
    """pybmds's number, NaN for its blank value -9999 or None."""
    if value is None or value == -9999.0:
        return math.nan
    return float(value)


def our_parameters(model: str, row: dict[str, Any]) -> dict[str, float]:
    """pybmds's parameters in the letters and signs of CytePool's "epa" set, and its
    SD sqrt(alpha), from alpha or log-alpha: pybmds's exponential 3 has b >= 0 and
    its direction as a setting."""
    from cytepool.stats._dose_curve_models import model_of

    values = dict(zip(row["names"], row["values"]))
    out = {name: values[name] for name in model_of("epa", model).parameters}
    if model == "exponential_3":
        out["b"] = out["b"] if row["is_increasing"] else -out["b"]
    if "alpha" in values:
        sd = math.sqrt(values["alpha"])
    else:
        sd = math.exp(0.5 * values["log-alpha"])
    return {**out, "sd": sd}


def pybmds_tasks(found: list[Series]) -> list[dict[str, Any]]:
    """The genes pybmds fits: the first 12 of half_log's large cell type and the edge
    genes, at 1 SD and at a change of 0.5."""
    tasks = []
    for series in found:
        if series.name not in ("half_log/cell_type_1", "edge"):
            continue
        for g in range(min(12, series.values.shape[1])):
            for form, value in (("sd", 1.0), ("absolute", 0.5)):
                tasks.append(
                    {
                        "key": f"{series.name}/{g}/{form}",
                        "series": series.name,
                        "gene": g,
                        "form": form,
                        "value": value,
                        "models": list(EPA),
                        "doses": series.doses.tolist(),
                        "values": series.values[:, g].tolist(),
                    }
                )
    return tasks


NOT_COMPLETED = "Did not successfully execute."


def compare_pybmds(
    rows: Rows,
    found: list[Series],
    reference: dict[str, Any],
    test2: dict[str, Any],
    test4: dict[str, Any],
) -> dict:
    """BMDs at pybmds's parameters; test 2 against mpmath's, with pybmds's own; test 4
    of the fits where pybmds reaches CytePool's maximum against mpmath's, from
    pybmds's log-likelihood and the exact maximum of a mean per dose group, with
    pybmds's own; the recommender's checks; bounds where pybmds reaches CytePool's
    maximum."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats._benchmark_dose_flags import FlagInputs, failed_checks
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    by_name = {series.name: series for series in found}
    exact_test2 = {r["key"]: r for r in test2}
    exact_test4 = {r["key"]: r["p_value"] for r in test4}
    bmd_differences, bound_differences, checks = [], [], []
    test2_differences, a2_shortfalls, p_ratios = [], [], []
    test4_differences: list[float] = []
    test4_pybmds: list[float] = []
    beyond, shortfalls, not_completed = 0, 0, 0
    negative: list[str] = []
    one_only: list[str] = []
    disagreements = []
    probe_cases: list[tuple[Case, dict[str, Any]]] = []
    fits_cache: dict[str, Any] = {}
    for case in reference["results"]:
        name, g, form = (
            case["key"].rsplit("/", 2)[0],
            int(case["key"].rsplit("/", 2)[1]),
            case["key"].rsplit("/", 1)[1],
        )
        series = by_name[name]
        if name not in fits_cache:
            fits_cache[name] = fit_dose_curves(
                series.doses, series.values, models="epa"
            )
        fits = fits_cache[name]
        x = series.doses
        top = float(np.max(x))
        lowest = float(np.min(x[x > 0]))
        exact = exact_test2[f"{name}/{g}"]
        ours_test2 = float(
            benchmark_doses(
                replace_gene(fits, g, np.array(fits.parameters)),
                BenchmarkResponse.absolute(0.5),
                bounds=False,
            ).constant_variance_p_value[0]
        )
        if form == "sd":
            test2_differences.append(relative(ours_test2, exact["p_value"]))
        for row in case["rows"]:
            model = row["model"]
            completed = not any(
                NOT_COMPLETED in message
                for messages in row["notes"].values()
                for message in messages
            )
            if not completed or not math.isfinite(pybmds_value(row["log_likelihood"])):
                not_completed += 1
                continue
            bmd_ref = pybmds_value(row["bmd"])
            if bmd_ref < 0.0:
                # pybmds's BMD of a fit whose slope is opposite to its direction
                # setting, as it solves for the change in that direction.
                negative.append(f"{case['key']} {model} {bmd_ref:.6g}")
                continue
            a2_shortfalls.append(exact["a2_log_likelihood"] - row["a2_log_likelihood"])
            if math.isfinite(pybmds_value(row["test2_p"])):
                p_ratios.append(pybmds_value(row["test2_p"]) / exact["p_value"])
            parameters = our_parameters(model, row)
            sd = parameters.pop("sd")
            size = sd if form == "sd" else 0.5
            m = fits.models.index(model)
            values = np.array(fits.parameters)
            values[g, m, :] = np.nan
            for letter, value in parameters.items():
                values[g, m, fits.parameter_names.index(letter)] = value
            ours = benchmark_doses(
                replace_gene(fits, g, values),
                BenchmarkResponse.absolute(size),
                model=model,
                bounds=False,
            )
            bmd = float(ours.bmd[0, 0])
            if math.isfinite(bmd_ref) and bmd_ref > 50 * top:
                beyond += 1
            elif math.isfinite(bmd_ref) and math.isfinite(bmd):
                bmd_differences.append(relative(bmd, bmd_ref))
            elif math.isfinite(bmd_ref) != math.isfinite(bmd):
                one_only.append(
                    f"{case['key']} {model}: CytePool {bmd:.6g} ({ours.status[0, 0]}), "
                    f"pybmds {bmd_ref:.6g}"
                )
            # The recommender's checks on pybmds's own numbers.
            inputs = FlagInputs(
                bmd=np.array([bmd_ref]),
                bmdl=np.array([pybmds_value(row["bmdl"])]),
                bmdu=np.array([pybmds_value(row["bmdu"])]),
                bounds=True,
                lowest_dose=lowest,
                top_dose=top,
                fit_p_value=np.array([pybmds_value(row["test4_p"])]),
                residual_near_bmd=np.array([pybmds_value(row["roi"])]),
                residual_at_vehicle=np.array([pybmds_value(row["roi_control"])]),
                vehicle_sd_ratio=np.array(
                    [row["est_sd_control"] / max(row["calc_sd_control"], 1e-6)]
                ),
                constant_variance_p_value=np.array([pybmds_value(row["test2_p"])]),
            )
            texts = [t for t in failed_checks("bmds_2026", inputs)[0].split("; ") if t]
            mine = sorted(PYBMDS_MESSAGES[t] for t in texts)
            theirs = sorted(
                (message, int(b))
                for b, messages in row["notes"].items()
                for message in messages
            )
            checks.append(mine == theirs)
            if mine != theirs:
                disagreements.append(
                    {"key": case["key"], "model": model, "ours": mine, "pybmds": theirs}
                )
        # Bounds of CytePool's own fits, where pybmds reaches the same maximum.
        n = x.size
        for row in case["rows"]:
            model = row["model"]
            m = fits.models.index(model)
            ll_ours = float(fits.log_likelihood[g, m])
            ll_ref = pybmds_value(row["log_likelihood"])
            if not math.isfinite(ll_ref) or ll_ref < ll_ours - n / 2 * 1e-8:
                shortfalls += 1
                continue
            one = replace_gene(fits, g, np.array(fits.parameters))
            bmr = (
                BenchmarkResponse.sd(1.0)
                if form == "sd"
                else BenchmarkResponse.absolute(0.5)
            )
            mine_model = benchmark_doses(one, bmr, model=model)
            exact = exact_test4.get(f"{name}/{g}/{model}")
            if form == "sd" and exact is not None:
                # Test 4 of the same maximum, which the flags read as the fit p-value,
                # and pybmds's own, which leaves out parameters at a bound.
                test4_differences.append(
                    p_distance(float(mine_model.fit_p_value[0]), exact)
                )
                if math.isfinite(pybmds_value(row["test4_p"])):
                    test4_pybmds.append(p_distance(pybmds_value(row["test4_p"]), exact))
            for name_ in ("bmdl", "bmdu"):
                ours_bound = float(getattr(mine_model, name_)[0, 0])
                ref_bound = pybmds_value(row[name_])
                if (
                    math.isfinite(ours_bound)
                    and math.isfinite(ref_bound)
                    and ours_bound > 0
                    and ref_bound > 0
                ):
                    entry = {
                        "key": f"{case['key']}/{model}/{name_}",
                        "bound": name_,
                        "ours": ours_bound,
                        "pybmds": ref_bound,
                        "bmd": float(mine_model.bmd[0, 0]),
                        "difference": relative(ours_bound, ref_bound),
                    }
                    bound_differences.append(entry)
                    if entry["difference"] > PYBMDS_BOUND:
                        # The documented profile at pybmds's bound decides.
                        probe_cases.append(
                            (
                                Case(entry["key"], one, 0, model, bmr, mine_model, {}),
                                entry,
                            )
                        )
    rows.add(
        f"BMDs given pybmds's parameters and SD, 1 SD and a change of 0.5 "
        f"({len(bmd_differences)} fits)",
        "pybmds 25.2 (normal, constant variance)",
        f"largest relative difference {max(bmd_differences, default=0.0):.1e}; "
        f"{beyond} beyond 50 times the top dose",
        f"close ({PYBMDS_BMD:g})",
        max(bmd_differences, default=0.0) <= PYBMDS_BMD,
    )
    rows.add(
        f"BMDs that one program reaches and the other does not ({len(one_only)} fits)",
        "pybmds 25.2",
        f"{len(one_only)}, listed below",
        "n/a",
        None,
    )
    rows.add(
        f"pybmds's runs it did not complete ({not_completed}) and its negative BMDs "
        f"({len(negative)}), left out",
        "pybmds 25.2",
        "listed below",
        "n/a",
        None,
        deliberate=True,
    )
    rows.add(
        f"Test 2, one variance against one per dose group, p-values ({len(test2_differences)} genes)",
        "mpmath 1.3.0: the likelihood ratio at the maximum-likelihood variances, 50 digits",
        f"largest relative difference {max(test2_differences, default=0.0):.1e}",
        f"exact ({EXACT:g})",
        max(test2_differences, default=0.0) <= EXACT,
    )
    rows.add(
        f"pybmds's test 2: its variance per dose group below its maximum ({len(a2_shortfalls)} fits)",
        "pybmds 25.2, against mpmath 1.3.0",
        f"log-likelihood short by up to {max(a2_shortfalls, default=0.0):.3g} "
        f"(median {np.median(a2_shortfalls) if a2_shortfalls else 0.0:.2g}); its "
        f"p-values up to {max(p_ratios, default=1.0):.3g} times the exact ones",
        "n/a",
        None,
        deliberate=True,
    )
    rows.add(
        f"Test 4, the fit p-value, of the fits where pybmds reaches CytePool's maximum "
        f"({len(test4_differences)} fits)",
        "mpmath 1.3.0: pybmds 25.2's log-likelihood against the exact maximum of a "
        "mean per dose group, every parameter counted, 50 digits",
        f"largest difference relative to p max(1, |log p|) "
        f"{max(test4_differences, default=0.0):.1e}",
        f"close ({CLOSE:g})",
        bool(test4_differences) and max(test4_differences) <= CLOSE,
    )
    moved = sum(d > CLOSE for d in test4_pybmds)
    rows.add(
        f"pybmds's own test 4, with its default count of parameters ({len(test4_pybmds)} "
        "fits)",
        "pybmds 25.2, against mpmath 1.3.0",
        f"{moved} differ by more than {CLOSE:g}: BMDS leaves out of K - p the "
        f"parameters at a bound; {len(test4_pybmds) - moved} the same",
        "n/a",
        None,
        deliberate=True,
    )
    rows.add(
        f"bmds_2026's checks on pybmds's own numbers ({len(checks)} completed fits)",
        "pybmds 25.2's recommender, default rules",
        f"{sum(checks)} of {len(checks)} the same checks in the same bins",
        "identical",
        all(checks),
    )
    return {
        "bounds": bound_differences,
        "shortfalls": shortfalls,
        "disagreements": disagreements,
        "one_only": one_only,
        "negative": negative,
        "not_completed": not_completed,
        "probe_cases": probe_cases,
    }


def probe_tasks(probe_cases: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
    """scipy's inputs at each of pybmds's bounds that differ from CytePool's by more
    than pybmds's root precision, with CytePool's curve there."""
    tasks = []
    for case, entry in probe_cases:
        doses = case.fits.doses
        task = case_inputs(case, doses[doses > 0][0] / 100, 50 * doses[-1])
        task.update(
            probe=entry["pybmds"],
            probe_bound=entry["bound"],
            witnesses={"probe": profile_witness(case, entry["pybmds"])},
        )
        tasks.append(task)
    return tasks


def compare_pybmds_bounds(
    rows: Rows, parts: dict[str, Any], probes: list[dict[str, Any]]
) -> list[str]:
    """pybmds's bounds against CytePool's: within pybmds's root precision, or, beyond
    it, decided by the documented profile at pybmds's bound. A bound nearer the BMD
    than CytePool's lies inside the region when a curve of the documented likelihood
    there is within the threshold; a bound farther out lies outside it when scipy's
    profile and CytePool's are both above the threshold there. A bound nearer the BMD
    that both profiles put outside the region ends the piece of the region around the
    BMD, where CytePool's bound ends the whole region; anything else fails."""
    bounds = parts["bounds"]
    by_key = {probe["key"]: probe for probe in probes}
    close = [b for b in bounds if b["difference"] <= PYBMDS_BOUND]
    outcomes: dict[str, list[str]] = {
        "inside": [],
        "outside": [],
        "piece": [],
        "failed": [],
    }
    for _, entry in parts["probe_cases"]:
        probe = by_key[entry["key"]]
        witness = probe["witness"]
        ours, theirs, bmd = entry["ours"], entry["pybmds"], entry["bmd"]
        nearer = min(ours, bmd) < theirs < max(ours, bmd)
        verified = (
            witness is not None
            and witness["inside"]
            and witness["constraint"] <= WITNESS_CONSTRAINT
        )
        within = verified and witness["deviance"] < -CLOSE
        above = (
            verified
            and witness["deviance"] > CLOSE
            and probe["at_probe"] is not None
            and probe["at_probe"] > CLOSE
        )
        if nearer and within:
            outcome = "inside"
        elif not nearer and above:
            outcome = "outside"
        elif nearer and above:
            outcome = "piece"
        else:
            outcome = "failed"
        outcomes[outcome].append(
            f"{entry['key']}: CytePool {ours:.6g}, pybmds {theirs:.6g}, BMD {bmd:.6g}; "
            f"deviance - threshold at pybmds's bound: CytePool's curve "
            f"{witness['deviance'] if witness else math.nan:.3g}, scipy "
            f"{probe['at_probe'] if probe['at_probe'] is not None else math.nan:.3g}"
        )
    rows.add(
        f"BMDLs and BMDUs where pybmds reaches CytePool's maximum ({len(bounds)})",
        "pybmds 25.2",
        f"{len(close)} within {PYBMDS_BOUND:g} (largest relative difference "
        f"{max((b['difference'] for b in close), default=0.0):.1e}); "
        f"{len(bounds) - len(close)} beyond it, of which "
        f"{len(outcomes['failed'])} the documented profile does not explain",
        f"close ({PYBMDS_BOUND:g}), or decided by the documented profile",
        not outcomes["failed"],
    )
    rows.add(
        f"pybmds's bounds beyond {PYBMDS_BOUND:g} of CytePool's "
        f"({len(bounds) - len(close)})",
        "pybmds 25.2; the documented profile by scipy 1.18.1 at pybmds's bound",
        f"{len(outcomes['inside'])} inside the region, by a curve of the documented "
        f"likelihood within the threshold there; {len(outcomes['outside'])} outside "
        f"it, by CytePool's profile and scipy's from its starts and CytePool's curve; "
        f"{len(outcomes['piece'])} at the end of the region's piece around the BMD",
        "n/a",
        None,
        deliberate=True,
    )
    return [f"{kind}: {line}" for kind, lines in outcomes.items() for line in lines]


# ---- scipy and mpmath: the profile and the straight line ----------------------------


PROFILE_SKIPPED: list[str] = []


@dataclass
class Case:
    """One gene's curves and BMR, with what CytePool found."""

    key: str
    fits: Any
    gene: int
    model: str
    bmr: Any
    result: Any  # BenchmarkDoses of the gene alone, the model named
    data: dict[str, Any]  # what the curves were fitted to, for the tests' fixtures


def case_data(fits: Any, g: int, samples: Any) -> dict[str, Any]:
    """What a gene's curves were fitted to: its samples' doses and values under
    constant variance, or the dose-group means with their covariance, variance and
    degrees of freedom in the weighted and moderated modes."""
    if samples is not None:
        doses, values = samples
        return {
            "model_set": fits.model_set,
            "variance": "constant",
            "doses": doses.tolist(),
            "values": values[:, g].tolist(),
        }
    return {
        "model_set": fits.model_set,
        "variance": fits.variance,
        "doses": fits.doses.tolist(),
        "n_samples": fits.n_samples.tolist(),
        "means": fits.means[g].tolist(),
        "covariance": fits.covariance[g].tolist(),
        "covariance_scale": float(fits.covariance_scale[g]),
        "df": float(fits.df[g]),
    }


def plugin_fits() -> list[tuple[str, Any]]:
    """Curves of the dose-group means of a weighted and a moderated fit, from
    dose_response on a series of the vehicle and 5 doses x 2 under "~ 1"."""
    from cytepool.design import Design
    from cytepool.dose_responses import dose_response

    cube, _ = simulated(TWO_REPLICATES, 2, 20261233, 120, (300.0, 300.0))
    out = []
    for mode in ("weighted", "moderated"):
        result = dose_response(
            cube, Design(cube.samples, "~ 1"), dose=axis(), models="epa", variance=mode
        )
        group = result.groups[0]
        out.append((f"{mode}/{group}", result.curve_fits(group)))
    return out


def case_inputs(case: Case, low: float, high: float) -> dict[str, Any]:
    """What the references see of a case: the means, their weights, the likelihood's
    constants, the BMR and the model, its documented name and parameters."""
    fits, g = case.fits, case.gene
    n_groups = fits.doses.size
    constant = fits.variance == "constant"
    weight = (
        np.diag(fits.n_samples)
        if constant
        else np.linalg.inv(np.asarray(fits.covariance[g], dtype=np.float64))
    )
    s2 = float(fits.covariance_scale[g])
    df = float(fits.df[g])
    m = fits.models.index(case.model)
    names = [p for p in model_letters(fits.model_set, case.model)]
    theta = [float(fits.parameters[g, m, fits.parameter_names.index(p)]) for p in names]
    if constant:
        a0 = s2 * df if df > 0 else 0.0
        n_eff = float(fits.n_samples.sum())
    else:
        a0, n_eff = df * s2, df + n_groups
    if case.bmr.kind == "sd" and not constant:
        value = case.bmr.value * math.sqrt(fits.n_samples[0] * fits.covariance[g, 0, 0])
    else:
        value = case.bmr.value
    return {
        "key": case.key,
        "model": case.model
        if case.model != "linear" or fits.model_set == "epa"
        else "linear",
        "doses": fits.doses.tolist(),
        "means": fits.means[g].tolist(),
        "weight": weight.tolist(),
        "mode": "constant" if constant else "plugin",
        "a0": a0,
        "n_eff": n_eff,
        "s2": s2,
        "df": df,
        "form": case.bmr.kind,
        "value": value,
        "theta": theta,
        "low": low,
        "high": high,
        "bmdl": float(case.result.bmdl[0, 0]),
        "bmdu": float(case.result.bmdu[0, 0]),
        "seed": zlib.crc32(case.key.encode()),
        "starts": 12,
        "data": case.data,
    }


def profile_witness(case: Case, b: float) -> dict[str, Any]:
    """CytePool's curve of largest likelihood among those whose change at the dose b is
    the BMR in the BMD's direction, as its search finds it at a dose of its scan: the
    grid of the model's nonlinear parameters at b, refined from the fit and from the
    grid's lowest minima by ``profile_at``. Its reported parameters, its SD for an
    SD-based BMR, and CytePool's deviance at b."""
    from cytepool.stats._benchmark_dose_search import (
        CHANGE_IN_SD,
        FORM,
        GRID_MARGIN,
        MAX_CARRIED,
        OBJECTIVE_FIT,
        THRESHOLD,
        grid_faces,
        grid_values,
        profile_at,
        profile_point,
        workspace,
    )
    from cytepool.stats._dose_curve_models import reported
    from cytepool.stats._dose_curve_search import grid_bases, grid_minima
    from cytepool.stats.potency import (
        _chosen_models,
        _gene_numbers,
        _GeneFits,
        _grid_changes,
        _likelihood,
        _search_model,
    )

    fits = case.fits
    genes = _GeneFits.of(fits, _chosen_models(fits, case.model))
    likelihood = _likelihood(fits, genes, case.bmr)
    kind = int(genes.kinds[0])
    model, points, inner, phi_fit = _search_model(fits, kind, genes.theta[0][None])
    linear, q, r = model[1], model[2], model[3].size
    s = 1.0 if case.result.direction[0, 0] == "up" else -1.0
    row = _gene_numbers(
        likelihood, np.zeros(1, dtype=np.int64), np.array([s]), case.result.bmd[0]
    )[0]
    x = fits.doses
    root = np.ascontiguousarray(genes.roots[0])
    ybar = np.ascontiguousarray(fits.means[0], dtype=np.float64)
    z = root @ ybar
    space = workspace(x.size)
    # The grid's values at b and their lowest minima within the margin, after the fit.
    starts = [phi_fit[0]]
    if r:
        bases, usable = grid_bases(kind, q, points, x)
        n_points = points.shape[0]
        inverses, betas = np.zeros((n_points, 3, 3, 3)), np.zeros((n_points, 3, 3))
        sums, sizes = np.zeros((n_points, 3)), np.zeros((n_points, 3), dtype=np.int64)
        grid_faces(linear, q, bases, usable, root, z, inverses, betas, sums, sizes)
        values = np.full(n_points, np.inf)
        change = _grid_changes(kind, q, points, np.array([b]))[:, 0]
        grid_values(
            linear, q, usable, inverses, betas, sums, sizes, change, row, values
        )
        margin = row[THRESHOLD] + GRID_MARGIN
        starts += [
            points[m]
            for m in grid_minima(values, inner)
            if m >= 0 and values[m] - row[OBJECTIVE_FIT] <= margin
        ]
    kept = np.zeros((MAX_CARRIED, max(r, 1)))
    kept_values = np.zeros(MAX_CARRIED)
    profile_at(
        model,
        row,
        b,
        np.ascontiguousarray(np.array(starts)),
        len(starts),
        x,
        ybar,
        root,
        z,
        kept,
        kept_values,
        space,
    )
    theta, extra = np.zeros(genes.theta[0].size), np.zeros(3)
    objective = profile_point(model, row, kept[0], b, x, root, z, theta, extra, space)
    return {
        "dose": b,
        "theta": reported(kind, theta).tolist(),
        "sigma": math.sqrt(extra[1]) if row[FORM] == CHANGE_IN_SD else None,
        "deviance": objective - row[OBJECTIVE_FIT],
    }


def profile_task(case: Case, low: float, high: float) -> dict[str, Any]:
    """The scipy reference's inputs of a case, with CytePool's curve at each of its
    finite bounds above 0."""
    task = case_inputs(case, low, high)
    task["witnesses"] = {
        name: profile_witness(case, task[name])
        for name in ("bmdl", "bmdu")
        if math.isfinite(task[name]) and task[name] > 0.0
    }
    return task


def model_letters(model_set: str, model: str) -> tuple[str, ...]:
    """The reported parameters of a model, in the order the chapter writes its mean."""
    from cytepool.stats._dose_curve_models import model_of

    return model_of(model_set, model).parameters


def profile_cases(found: list[Series], plugin: list[tuple[str, Any]]) -> list[Case]:
    """Cases for the scipy profile: in each set, the first genes of each best model
    of half_log's large cell type and of the edge genes, at 1.349 SD and at a change
    of 0.4, and the plug-in fits' genes of each best model."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    bmrs = (BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.4))
    sources: list[tuple[str, Any, Any]] = []
    for series in found:
        if series.name in ("half_log/cell_type_1", "edge"):
            for model_set in ("epa", "ll4", "dromics"):
                fits = fit_dose_curves(series.doses, series.values, models=model_set)
                sources.append(
                    (f"{series.name}/{model_set}", fits, (series.doses, series.values))
                )
    sources += [(label, fits, None) for label, fits in plugin]
    cases = []
    for label, fits, samples in sources:
        seen: set[str] = set()
        for g in range(fits.means.shape[0]):
            if fits.best[g] < 0:
                continue
            model = fits.models[fits.best[g]]
            if model in seen and not label.startswith("edge"):
                continue
            # A parameter at a bound, such as an exponent of 18 with a scale of
            # 1e-18, leaves scipy's SLSQP without a usable gradient: such fits are
            # counted, not compared.
            if fits.parameters_at_bounds[g, fits.best[g]]:
                PROFILE_SKIPPED.append(f"{label}/{g}/{model}")
                continue
            seen.add(model)
            one = replace_gene(fits, g, np.array(fits.parameters))
            for bmr in bmrs:
                result = benchmark_doses(one, bmr, model=model)
                if math.isfinite(float(result.bmd[0, 0])):
                    cases.append(
                        Case(
                            f"{label}/{g}/{model}/{bmr.label}",
                            one,
                            0,
                            model,
                            bmr,
                            result,
                            case_data(fits, g, samples),
                        )
                    )
    return cases


def compare_profiles(
    rows: Rows, cases: list[Case], reference: dict[str, Any]
) -> list[str]:
    """CytePool's bounds against scipy's profile: the roots of deviance = threshold
    near each bound, the profile maximised from scipy's starts and from CytePool's
    curve at the bound; scipy's deviance at each bound from its own starts, which
    reaches CytePool's or lies above it, while below it scipy would have found a curve
    that CytePool missed; and CytePool's curve, checked with the documented means.
    Returns the bounds that scipy's own starts do not reach, and the failures."""
    by_key = {r["key"]: r for r in reference["results"]}
    bmds, roots, beyond, ends, witnesses = [], [], [], [], []
    largest: list[tuple[float, str]] = []
    reached = 0
    unreached: list[str] = []
    missed: list[str] = []
    rootless: list[str] = []
    for case in cases:
        ref = by_key[case.key]
        if ref["bmd"] is None:
            continue
        bmds.append(relative(float(case.result.bmd[0, 0]), ref["bmd"]))
        for name in ("bmdl", "bmdu"):
            ours = float(getattr(case.result, name)[0, 0])
            at_ours = ref.get(f"{name}_at_ours")
            if at_ours is not None:
                witnesses.append(ref[f"{name}_witness"])
                if ref.get(name) is None:
                    rootless.append(f"{case.key} {name} {ours:.6g}: no root within 2%")
                else:
                    roots.append(relative(ours, ref[name]))
                    largest.append((roots[-1], f"{case.key} {name}"))
                if abs(at_ours) <= CLOSE:
                    reached += 1
                elif at_ours > CLOSE:
                    unreached.append(
                        f"{case.key} {name} {ours:.6g}: scipy's deviance "
                        f"{at_ours:.3g} above the threshold"
                    )
                else:
                    missed.append(
                        f"{case.key} {name} {ours:.6g}: scipy's deviance "
                        f"{at_ours:.3g} below the threshold"
                    )
            if ref.get(f"{name}_beyond") is not None:
                beyond.append(ref[f"{name}_beyond"] > 0.0)
            if ref.get(f"{name}_end") is not None:
                ends.append(ref[f"{name}_end"] <= OPEN_END)
    # Each tool's BMD comes from its own fit: along a flat direction of a likelihood,
    # fits whose likelihoods agree to 1e-12 can place a BMD 1e-3 apart, so BMDs are
    # compared given the same parameters (DRomics, pybmds) and here as information.
    rows.add(
        f"BMDs, each from its tool's own fit, the same cases ({len(bmds)})",
        "scipy 1.18.1: the documented means, fits from 40 starts, brentq",
        f"largest relative difference {max(bmds, default=0.0):.1e}",
        "n/a",
        None,
    )
    n_bounds = len(roots) + len(rootless)
    rows.add(
        f"BMDLs and BMDUs of every model of the three sets, constant, weighted and "
        f"moderated, 1.349 SD and a change of 0.4 ({n_bounds})",
        "scipy 1.18.1: the constrained profile by SLSQP from 12 starts and from "
        "CytePool's curve at the bound, brentq",
        f"largest relative difference {max(roots, default=0.0):.1e}; "
        f"{len(rootless)} without a root within 2%",
        f"close ({CLOSE:g})",
        not rootless and max(roots, default=0.0) <= CLOSE,
    )
    rows.add(
        f"scipy's own starts at each of these bounds ({n_bounds})",
        "scipy 1.18.1: the constrained profile by SLSQP from 12 starts",
        f"{reached} reach CytePool's curve, the deviance within {CLOSE:g} of the "
        f"threshold; {len(unreached)} lie above it; {len(missed)} below it",
        f"none below the threshold by more than {CLOSE:g}",
        not missed,
    )
    worst_constraint = max((w["constraint"] for w in witnesses), default=0.0)
    worst_deviance = max((abs(w["deviance"]) for w in witnesses), default=0.0)
    rows.add(
        f"CytePool's curve at each of these bounds, given, by the documented means "
        f"({len(witnesses)})",
        "scipy 1.18.1's process: the documented means, bounds and likelihood",
        f"{sum(w['inside'] for w in witnesses)} within the parameters' bounds; change "
        f"at the bound within {worst_constraint:.1e} of the BMR, relative; deviance "
        f"within {worst_deviance:.1e} of the threshold",
        f"close ({CLOSE:g})",
        all(w["inside"] for w in witnesses)
        and worst_constraint <= WITNESS_CONSTRAINT
        and worst_deviance <= CLOSE,
    )
    rows.add(
        f"Best fits with a parameter at a bound, left out of the scipy rows ({len(PROFILE_SKIPPED)})",
        "scipy 1.18.1",
        ", ".join(sorted({key.rsplit("/", 1)[1] for key in PROFILE_SKIPPED})) or "none",
        "n/a",
        None,
    )
    rows.add(
        f"No dose beyond a bound lies in the region, 6 per decade to the search's end ({len(beyond)})",
        "scipy 1.18.1",
        f"{sum(beyond)} of {len(beyond)} outside",
        "identical",
        all(beyond),
    )
    rows.add(
        f"A BMDL of 0 or a BMDU of infinity: the region reaches the search's end ({len(ends)})",
        "scipy 1.18.1: the deviance at the lowest dose / 100 or 50 times the top dose",
        f"{sum(ends)} of {len(ends)} within the threshold",
        f"identical (within {OPEN_END:g} of the threshold)",
        all(ends),
    )
    largest.sort(reverse=True)
    return (
        [f"the largest difference of a root: {d:.2e} {key}" for d, key in largest[:3]]
        + unreached
        + [f"failed: {line}" for line in missed + rootless]
    )


def line_cases(found: list[Series], plugin: list[tuple[str, Any]]) -> list[Case]:
    """Straight lines: 8 genes of half_log's large cell type and the plug-in fits'
    genes, at a change of 0.4 and at 1.349 SD."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    sources = [
        (s.name, fit_dose_curves(s.doses, s.values, models="epa"), (s.doses, s.values))
        for s in found
        if s.name == "half_log/cell_type_1"
    ] + [(label, fits, None) for label, fits in plugin]
    cases = []
    for label, fits, samples in sources:
        for g in range(min(8, fits.means.shape[0])):
            one = replace_gene(fits, g, np.array(fits.parameters))
            for bmr in (BenchmarkResponse.absolute(0.4), BenchmarkResponse.sd(1.349)):
                result = benchmark_doses(one, bmr, model="linear")
                cases.append(
                    Case(
                        f"{label}/{g}/linear/{bmr.label}",
                        one,
                        0,
                        "linear",
                        bmr,
                        result,
                        case_data(fits, g, samples),
                    )
                )
    return cases


def compare_lines(rows: Rows, cases: list[Case], reference: dict[str, Any]) -> None:
    by_key = {r["key"]: r for r in reference["results"]}
    groups: dict[str, list[float]] = {}
    for case in cases:
        ref = by_key[case.key]
        mode = case.fits.variance
        kind = "constant variance" if mode == "constant" else f"{mode} means"
        label = f"{kind}, {'SD-based' if case.bmr.kind == 'sd' else 'absolute'}"
        values = groups.setdefault(label, [])
        values.append(relative(float(case.result.bmd[0, 0]), ref["bmd"]))
        for name in ("bmdl", "bmdu"):
            values.append(relative(float(getattr(case.result, name)[0, 0]), ref[name]))
    for label, values in groups.items():
        rows.add(
            f"Straight lines, {label}: BMDs and bounds ({len(values)})",
            "mpmath 1.3.0: closed-form profile, roots at 50 digits",
            f"largest relative difference {max(values):.1e}",
            f"exact ({EXACT:g})",
            max(values) <= EXACT,
        )


# ---- the checks of BMDS, NTP, ETAP and EFSA in R ---------------------------------------


def flag_rows(results: list[Any]) -> dict[str, Any]:
    """Every row of the BMD tables, with what the checks read."""
    keys = (
        "bmd",
        "bmdl",
        "bmdu",
        "fit_p_value",
        "residual_near_bmd",
        "residual_at_vehicle",
        "vehicle_sd_ratio",
        "constant_variance_p_value",
    )
    tables = []
    for result in results:
        inputs = result._flag_inputs()
        tables.append(
            {
                "bounds": bool(inputs.bounds),
                "lowest_dose": inputs.lowest_dose,
                "top_dose": inputs.top_dose,
                "rows": {
                    k: [
                        None
                        if not math.isfinite(v) and not math.isinf(v)
                        else ("inf" if v == math.inf else v)
                        for v in getattr(inputs, k).tolist()
                    ]
                    for k in keys
                },
                "ours": {
                    s: result.table(flags=s)[f"flags_{s}"].tolist() for s in FLAG_SETS
                },
            }
        )
    return {"tables": tables}


def compare_flags(rows: Rows, payload: dict[str, Any]) -> list[dict]:
    outputs = []
    agree = total = 0
    for table in payload["tables"]:
        reference = run_r(FLAGS_R, table, "DOSE_RSCRIPT")
        outputs.append(reference)
        for flag_set in FLAG_SETS:
            for ours, theirs in zip(table["ours"][flag_set], reference[flag_set]):
                total += 1
                agree += ours == theirs
    rows.add(
        f"Failed checks of the four sets on every BMD table ({total:,} rows and sets)",
        "the sets coded in R 4.4.3 from pybmds's rules and the guidance",
        f"{agree:,} of {total:,} identical",
        "identical",
        agree == total,
    )
    return outputs


def descriptive_check(rows: Rows) -> None:
    """A cell type of one sample per treated dose gets BMDs without bounds in
    dose_response, and benchmark_doses refuses bounds for curves of dose groups
    without replicates."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.design import Design
    from cytepool.dose_responses import dose_response
    from cytepool.simulated_dose_series import simulate_dose_series
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    cube, _ = simulate_dose_series(
        TWO_REPLICATES,
        replicates=[3, 1, 1, 1, 1, 1],
        cell_types=1,
        n_genes=60,
        rng=20261234,
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=axis(),
        models="epa",
        bmr=BenchmarkResponse.absolute(0.5),
    )
    table = result.table(result.groups[0], "benchmark_doses")
    has_bmds = bool(np.isfinite(np.asarray(table["bmd"], dtype=float)).any())
    no_bounds = bool(np.all(np.isnan(np.asarray(table["bmdl"], dtype=float))))
    rows.add(
        "A cell type that supports only description: BMDs without bounds",
        "the rule: bounds only where the cell type supports tests",
        f"BMDs {has_bmds}, bounds absent {no_bounds}",
        "identical",
        has_bmds and no_bounds,
    )
    rng = np.random.default_rng(20261234)
    doses = np.asarray(TWO_REPLICATES)
    single = fit_dose_curves(
        doses, 6.0 + 0.1 * doses[:, None] + rng.normal(0.0, 0.2, (doses.size, 3))
    )
    try:
        benchmark_doses(single, BenchmarkResponse.absolute(0.5))
        refused = False
    except ValueError:
        refused = True
    rows.add(
        "Bounds for curves of dose groups without replicates",
        "the rule: no degrees of freedom for the variance",
        f"refused {refused}",
        "identical",
        refused,
    )


# ---- fixtures, benchmark, main ------------------------------------------------------


def write_fixtures(parts: dict[str, Any]) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, payload in parts.items():
        path = FIXTURE_DIRECTORY / f"{name}.json"
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(
            f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
        )


def benchmark() -> None:
    """benchmark_doses on 10,000 genes x 9 dose groups x 3, the "epa" set's best
    models, at 1.349 SD with the 1 SD comparison, with the ordinary least squares fit
    of linear_models.py's benchmark as the machine's baseline."""
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.linear_models import fit_linear_model
    from cytepool.stats.potency import benchmark_doses

    rng = np.random.default_rng(20261235)
    x = np.repeat(np.asarray(HALF_LOG), 3)
    pos = np.maximum(x, 1e-300)
    n_genes = 10_000
    ec50 = 10 ** rng.uniform(-2.0, 1.5, n_genes)
    top = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], n_genes)
    flat = rng.random(n_genes) < 0.8
    curves = top / (1.0 + (ec50[None, :] / pos[:, None]) ** 2) * (x[:, None] > 0)
    curves[:, flat] = 0.0
    values = 6.0 + curves + rng.normal(0.0, 0.25, (x.size, n_genes))
    fits = fit_dose_curves(x, values, models="epa")
    bmr = BenchmarkResponse.sd(1.349)

    y = rng.normal(size=(200, 20_000))
    design = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    timings = {}
    for name, run in (
        ("bounds", lambda: benchmark_doses(fits, bmr)),
        ("no_bounds", lambda: benchmark_doses(fits, bmr, bounds=False)),
        ("ols", lambda: fit_linear_model(y, design)),
    ):
        run()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        timings[name] = float(np.median(times))
    baseline = timings["ols"]
    print("| run | seconds | baselines |")
    print("| --- | --- | --- |")
    for name, label in (
        ("bounds", "benchmark_doses, 10,000 genes, 1.349 SD and 1 SD, with bounds"),
        ("no_bounds", "the same without bounds"),
        ("ols", "fit_linear_model, 20,000 genes x 200 samples (baseline)"),
    ):
        seconds = timings[name]
        print(f"| {label} | {seconds:.2f} | {seconds / baseline:.0f} |")
    print("target: 60 s per cell type of 10,000 genes, about 350 baselines")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time benchmark_doses")
    parser.add_argument("--scipy", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mpmath", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pybmds", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.scipy is not None:
        scipy_main(arguments.scipy)
        return 0
    if arguments.mpmath is not None:
        mpmath_main(arguments.mpmath)
        return 0
    if arguments.pybmds is not None:
        pybmds_main(arguments.pybmds)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0

    rows = Rows()
    found = constant_series()
    found.append(edge_series())
    plugin = plugin_fits()

    # DRomics, given its parameters, on half_log's large cell type and two steep lines
    # whose BMDs lie below minBMD.
    half_log = next(s for s in found if s.name == "half_log/cell_type_1")
    items = Series(
        "dromics",
        half_log.doses,
        np.column_stack([half_log.values, steep_items(half_log.doses)]),
        [*half_log.feature_ids, "steep_rise", "steep_fall"],
    )
    dromics = run_r(
        DROMICS_R,
        {
            "doses": items.doses.tolist(),
            "values": items.values.T.tolist(),
            "z": 1,
            "x": 10,
        },
        "DOSE_RSCRIPT",
    )
    print("DRomics: " + ", ".join(f"{k} {v}" for k, v in dromics["versions"].items()))
    dromics_records = compare_dromics(rows, items, dromics)

    # pybmds.
    tasks = pybmds_tasks(found)
    pybmds = run_python(
        "--pybmds",
        os.environ.get("BMDS_PYTHON", "python"),
        {"pybmds_inputs.json": json.dumps(tasks)},
        "pybmds.json",
    )
    print("pybmds: " + ", ".join(f"{k} {v}" for k, v in pybmds["versions"].items()))

    # mpmath: the straight lines' bounds, and test 2 of the genes pybmds fits.
    lines = line_cases(found, plugin)
    line_inputs = [
        case_inputs(
            case,
            case.fits.doses[case.fits.doses > 0][0] / 100,
            50 * case.fits.doses[-1],
        )
        for case in lines
    ]
    test2_inputs = list(
        {
            task["key"].rsplit("/", 1)[0]: {
                "key": task["key"].rsplit("/", 1)[0],
                "doses": task["doses"],
                "values": task["values"],
            }
            for task in tasks
        }.values()
    )
    test4_inputs = [
        {
            "key": f"{task['key'].rsplit('/', 1)[0]}/{row['model']}",
            "gene": task["key"].rsplit("/", 1)[0],
            "log_likelihood": pybmds_value(row["log_likelihood"]),
            "df": len(set(task["doses"])) - MEAN_PARAMETERS[row["model"]],
        }
        for task, case in zip(tasks, pybmds["results"], strict=True)
        if task["key"].endswith("/sd")
        for row in case["rows"]
        if math.isfinite(pybmds_value(row["log_likelihood"]))
        and not any(
            NOT_COMPLETED in message
            for messages in row["notes"].values()
            for message in messages
        )
    ]
    mpmath = run_python(
        "--mpmath",
        sys.executable,
        {
            "mpmath_inputs.json": json.dumps(
                {"lines": line_inputs, "test2": test2_inputs, "test4": test4_inputs}
            )
        },
        "mpmath.json",
    )
    print("mpmath: " + ", ".join(f"{k} {v}" for k, v in mpmath["versions"].items()))
    pybmds_parts = compare_pybmds(rows, found, pybmds, mpmath["test2"], mpmath["test4"])

    # scipy.
    cases = profile_cases(found, plugin)
    scale_of = {
        case.key: (
            case.fits.doses[case.fits.doses > 0][0] / 100,
            50 * case.fits.doses[-1],
        )
        for case in cases
    }
    profile_inputs = [profile_task(case, *scale_of[case.key]) for case in cases]
    probes = probe_tasks(pybmds_parts["probe_cases"])
    scipy = run_python(
        "--scipy",
        sys.executable,
        {"scipy_inputs.json": json.dumps({"cases": profile_inputs, "probes": probes})},
        "scipy.json",
    )
    print("scipy: " + ", ".join(f"{k} {v}" for k, v in scipy["versions"].items()))
    unreached = compare_profiles(rows, cases, scipy)
    compare_lines(rows, lines, mpmath)

    # pybmds's bounds against CytePool's, with the documented profile deciding where
    # they differ.
    pybmds_bounds = compare_pybmds_bounds(rows, pybmds_parts, scipy["probes"])
    rows.add(
        f"pybmds's fits below CytePool's maximum, whose bounds are left out "
        f"({pybmds_parts['shortfalls']})",
        "pybmds 25.2",
        "left out: their bounds are those of a curve of lower likelihood",
        "n/a",
        None,
    )

    # The checks of BMDS, NTP, ETAP and EFSA in R, on every table computed here.
    from cytepool.benchmark_responses import BenchmarkResponse
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    tables = [
        benchmark_doses(
            fit_dose_curves(series.doses, series.values, models="epa"),
            [BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(0.5)],
        )
        for series in found
    ]
    tables += [benchmark_doses(fits, BenchmarkResponse.sd(1.349)) for _, fits in plugin]
    flag_payload = flag_rows(tables)
    flag_outputs = compare_flags(rows, flag_payload)
    descriptive_check(rows)

    print()
    rows.print()
    print()
    print("BMDs that one program reaches and the other does not:")
    for line in pybmds_parts["one_only"]:
        print(f"  {line}")
    print("pybmds's negative BMDs:")
    for line in pybmds_parts["negative"]:
        print(f"  {line}")
    print("Disagreements of the recommender's checks:")
    for item in pybmds_parts["disagreements"]:
        print(f"  {item}")
    print("pybmds's bounds beyond its root precision, by the documented profile:")
    for line in pybmds_bounds:
        print(f"  {line}")
    print("scipy's profile at CytePool's bounds, where its own starts do not reach:")
    for line in unreached:
        print(f"  {line}")
    if arguments.write_fixtures:
        if rows.failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(
                {
                    "dromics": {
                        "minBMD": dromics["minBMD"],
                        "doses": items.doses.tolist(),
                        "rows": dromics_records,
                    },
                    "pybmds": {"tasks": tasks[:8], "results": pybmds["results"][:8]},
                    "profiles": {"inputs": profile_inputs, "results": scipy["results"]},
                    "linear": {
                        "inputs": line_inputs,
                        "results": mpmath["results"],
                        "test2_inputs": test2_inputs,
                        "test2": mpmath["test2"],
                    },
                    "flags": {
                        "tables": flag_payload["tables"][:2],
                        "r": flag_outputs[:2],
                    },
                }
            )
    print(f"{len(rows.rows)} checks, {rows.failures} failed")
    return 1 if rows.failures else 0


if __name__ == "__main__":
    sys.exit(main())
